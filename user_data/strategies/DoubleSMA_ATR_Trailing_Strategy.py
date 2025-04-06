"""
Double SMA + ATR Trailing Strategy
Chiến lược kết hợp hai đường SMA (nhanh/chậm) với ATR Trailing Stop
Tối ưu cho khung thời gian 4h
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import ta
from pandas import DataFrame
from functools import reduce
from typing import Dict, List, Optional, Tuple
import logging
import os

# Định nghĩa các class và hàm cần thiết để chiến lược có thể chạy mà không cần freqtrade
class IStrategy:
    """Base class for strategies"""
    def __init__(self, config: dict) -> None:
        self.config = config
        self.dp = DataProvider()

class DataProvider:
    """Mock DataProvider class"""
    def __init__(self):
        self.data = {}
        
    def current_whitelist(self):
        return ["BTC/USDT"]
        
    def get_pair_dataframe(self, pair, timeframe):
        # Return empty dataframe
        return pd.DataFrame()
        
    def get_analyzed_dataframe(self, pair, timeframe):
        # Return empty dataframe and empty dict
        return pd.DataFrame(), {}

class CategoricalParameter:
    """Parameter with categorical values"""
    def __init__(self, categories, default, space="buy", optimize=True):
        self.categories = categories
        self.default = default
        self.value = default
        self.space = space
        self.optimize = optimize

class DecimalParameter:
    """Parameter with decimal values"""
    def __init__(self, low, high, default, space="buy", optimize=True):
        self.low = low
        self.high = high
        self.default = default
        self.value = default
        self.space = space
        self.optimize = optimize

class IntParameter:
    """Parameter with integer values"""
    def __init__(self, low, high, default, space="buy", optimize=True):
        self.low = low
        self.high = high
        self.default = default
        self.value = default
        self.space = space
        self.optimize = optimize

class Trade:
    """Mock Trade class"""
    def __init__(self, is_short=False):
        self.is_short = is_short

def calculate_atr_trailing_stop(dataframe, period=14, multiplier=2.0, is_short=False):
    """
    Tính trailing stop dựa trên ATR
    
    period: số nến để tính ATR
    multiplier: hệ số nhân cho ATR
    is_short: True nếu tính cho vị thế short, False cho vị thế long
    """
    atr = ta.volatility.average_true_range(
        high=dataframe['high'],
        low=dataframe['low'],
        close=dataframe['close'],
        window=period
    )
    
    # Long Trail Stop = Close - ATR * multiplier
    # Short Trail Stop = Close + ATR * multiplier
    if not is_short:
        # Long: giá - ATR*multiplier
        initial_stop = dataframe['close'] - atr * multiplier
        
        # Chỉ tăng, không giảm
        highest_stop = initial_stop.expanding().max()
        trailing_stop = highest_stop.shift(1).fillna(0)
        
        # Sử dụng initial stop cho nến đầu tiên, sau đó chọn max giữa stop trước đó và initial stop
        result = np.where(trailing_stop == 0, initial_stop, np.maximum(initial_stop, trailing_stop))
    else:
        # Short: giá + ATR*multiplier
        initial_stop = dataframe['close'] + atr * multiplier
        
        # Chỉ giảm, không tăng
        lowest_stop = initial_stop.expanding().min()
        trailing_stop = lowest_stop.shift(1).fillna(float('inf'))
        
        # Sử dụng initial stop cho nến đầu tiên, sau đó chọn min giữa stop trước đó và initial stop
        result = np.where(trailing_stop == float('inf'), initial_stop, np.minimum(initial_stop, trailing_stop))
    
    return pd.Series(result, index=dataframe.index)

class DoubleSMA_ATR_Trailing_Strategy(IStrategy):
    """
    Chiến lược kết hợp hai đường SMA (nhanh/chậm) với ATR Trailing Stop
    
    - Sử dụng SMA crossover làm tín hiệu giao dịch chính
    - Kết hợp ATR Trailing Stop để tối ưu điểm thoát
    - Thêm lọc Volume và ADX để cải thiện chất lượng tín hiệu
    - Tối ưu cho khung thời gian 4h
    """
    
    # Tham số cho SMA
    sma_fast_length = IntParameter(5, 50, default=20, space="buy", optimize=True)
    sma_slow_length = IntParameter(30, 200, default=50, space="buy", optimize=True)
    
    # Tham số cho ATR
    atr_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    atr_multiplier_long = DecimalParameter(1.0, 4.0, default=2.0, space="sell", optimize=True)
    atr_multiplier_short = DecimalParameter(1.0, 4.0, default=2.0, space="sell", optimize=True)
    
    # Tham số cho ADX
    adx_period = IntParameter(7, 30, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 35, default=25, space="buy", optimize=True)
    
    # Tham số cho Volume
    volume_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.05  # ATR trailing sẽ được sử dụng thay vì stoploss cố định
    
    # Time-in-force và ROI
    timeframe = '4h'
    
    minimal_roi = {
        "0": 0.08,     # 8%
        "48": 0.05,    # 5% sau 48 giờ 
        "120": 0.03,   # 3% sau 120 giờ
        "240": 0.02    # 2% sau 240 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.04  # 4%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # SMA indicators
        for length in range(5, 201, 5):  # [5, 10, 15, ..., 200]
            dataframe[f'sma_{length}'] = ta.trend.sma_indicator(
                close=dataframe['close'], 
                window=length
            )
        
        # ATR
        for period in range(10, 31, 2):  # [10, 12, ..., 30]
            dataframe[f'atr_{period}'] = ta.volatility.average_true_range(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=period
            )
        
        # ADX
        for period in range(7, 31, 7):  # [7, 14, 21, 28]
            adx = ta.trend.ADXIndicator(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=period
            )
            dataframe[f'adx_{period}'] = adx.adx()
            dataframe[f'plus_di_{period}'] = adx.adx_pos()
            dataframe[f'minus_di_{period}'] = adx.adx_neg()
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Bollinger Bands
        for window in [20, 50]:
            for std in [2.0, 2.5]:
                bollinger = ta.volatility.BollingerBands(
                    close=dataframe['close'],
                    window=window,
                    window_dev=std
                )
                dataframe[f'bb_{window}_upper_{std}'] = bollinger.bollinger_hband()
                dataframe[f'bb_{window}_lower_{std}'] = bollinger.bollinger_lband()
                dataframe[f'bb_{window}_mid_{std}'] = bollinger.bollinger_mavg()
                dataframe[f'bb_{window}_width_{std}'] = (
                    (dataframe[f'bb_{window}_upper_{std}'] - dataframe[f'bb_{window}_lower_{std}']) / 
                    dataframe[f'bb_{window}_mid_{std}']
                )
        
        # RSI
        for period in [7, 14, 21]:
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=period
            )
        
        # Lấy tham số từ config
        sma_fast = self.sma_fast_length.value
        sma_slow = self.sma_slow_length.value
        atr_period = self.atr_period.value
        adx_period = self.adx_period.value
        
        # Tính SMA crossover
        dataframe['sma_fast'] = dataframe[f'sma_{sma_fast}']
        dataframe['sma_slow'] = dataframe[f'sma_{sma_slow}']
        
        dataframe['sma_cross_up'] = (
            (dataframe['sma_fast'] > dataframe['sma_slow']) & 
            (dataframe['sma_fast'].shift(1) <= dataframe['sma_slow'].shift(1))
        )
        
        dataframe['sma_cross_down'] = (
            (dataframe['sma_fast'] < dataframe['sma_slow']) & 
            (dataframe['sma_fast'].shift(1) >= dataframe['sma_slow'].shift(1))
        )
        
        # Xác định xu hướng
        dataframe['uptrend'] = dataframe['sma_fast'] > dataframe['sma_slow']
        dataframe['downtrend'] = dataframe['sma_fast'] < dataframe['sma_slow']
        
        # Tính ATR Trailing Stops
        dataframe['atr_trailing_stop_long'] = calculate_atr_trailing_stop(
            dataframe, 
            period=atr_period, 
            multiplier=self.atr_multiplier_long.value,
            is_short=False
        )
        
        dataframe['atr_trailing_stop_short'] = calculate_atr_trailing_stop(
            dataframe, 
            period=atr_period, 
            multiplier=self.atr_multiplier_short.value,
            is_short=True
        )
        
        # Xác định ADX trend strength
        dataframe['adx'] = dataframe[f'adx_{adx_period}']
        dataframe['plus_di'] = dataframe[f'plus_di_{adx_period}']
        dataframe['minus_di'] = dataframe[f'minus_di_{adx_period}']
        
        dataframe['strong_trend'] = dataframe['adx'] > self.adx_threshold.value
        dataframe['bullish_trend'] = (dataframe['plus_di'] > dataframe['minus_di']) & dataframe['strong_trend']
        dataframe['bearish_trend'] = (dataframe['minus_di'] > dataframe['plus_di']) & dataframe['strong_trend']
        
        # Xác định khi giá chạm trailing stop
        dataframe['hit_trailing_stop_long'] = (
            (dataframe['low'] <= dataframe['atr_trailing_stop_long']) &
            (dataframe['low'].shift(1) > dataframe['atr_trailing_stop_long'].shift(1))
        )
        
        dataframe['hit_trailing_stop_short'] = (
            (dataframe['high'] >= dataframe['atr_trailing_stop_short']) &
            (dataframe['high'].shift(1) < dataframe['atr_trailing_stop_short'].shift(1))
        )
        
        # Xác định pullback trong trend
        dataframe['pullback_in_uptrend'] = (
            dataframe['uptrend'] &
            (dataframe['low'] <= dataframe['sma_fast']) &
            (dataframe['close'] > dataframe['low'])  # Có rejection, không đóng cửa ở mức thấp
        )
        
        dataframe['pullback_in_downtrend'] = (
            dataframe['downtrend'] &
            (dataframe['high'] >= dataframe['sma_fast']) &
            (dataframe['close'] < dataframe['high'])  # Có rejection, không đóng cửa ở mức cao
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Lấy tham số từ config
        adx_period = self.adx_period.value
        adx_threshold = self.adx_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. SMA cross up + strong ADX
        conditions_long.append(
            (dataframe['sma_cross_up']) &
            (dataframe[f'adx_{adx_period}'] > adx_threshold) &
            (dataframe['volume_ratio'] > self.volume_threshold.value)
        )
        
        # 2. Pullback to fast SMA trong uptrend mạnh
        conditions_long.append(
            (dataframe['pullback_in_uptrend']) &
            (dataframe['bullish_trend']) &
            (dataframe['close'] > dataframe['sma_fast']) &  # Close trên fast SMA
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Bullish ADX crossover
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['plus_di'] > dataframe['minus_di']) &
            (dataframe['plus_di'].shift(1) <= dataframe['minus_di'].shift(1)) &
            (dataframe[f'adx_{adx_period}'] > 20) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 4. RSI oversold bounce trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['rsi_14'] < 40) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &
            (dataframe['rsi_14'].shift(1) < dataframe['rsi_14'].shift(2)) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bollinger Band bounce lower trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['low'] < dataframe['bb_20_lower_2.0']) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. SMA cross down + strong ADX
        conditions_short.append(
            (dataframe['sma_cross_down']) &
            (dataframe[f'adx_{adx_period}'] > adx_threshold) &
            (dataframe['volume_ratio'] > self.volume_threshold.value)
        )
        
        # 2. Pullback to fast SMA trong downtrend mạnh
        conditions_short.append(
            (dataframe['pullback_in_downtrend']) &
            (dataframe['bearish_trend']) &
            (dataframe['close'] < dataframe['sma_fast']) &  # Close dưới fast SMA
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Bearish ADX crossover
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['minus_di'] > dataframe['plus_di']) &
            (dataframe['minus_di'].shift(1) <= dataframe['plus_di'].shift(1)) &
            (dataframe[f'adx_{adx_period}'] > 20) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 4. RSI overbought drop trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['rsi_14'] > 60) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1)) &
            (dataframe['rsi_14'].shift(1) > dataframe['rsi_14'].shift(2)) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bollinger Band bounce upper trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['high'] > dataframe['bb_20_upper_2.0']) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Hit ATR trailing stop
        conditions_exit_long.append(
            dataframe['hit_trailing_stop_long']
        )
        
        # 2. SMA cross down
        conditions_exit_long.append(
            dataframe['sma_cross_down']
        )
        
        # 3. ADX trend change to bearish
        conditions_exit_long.append(
            (dataframe['bearish_trend']) &
            (~dataframe['bearish_trend'].shift(1))
        )
        
        # 4. RSI overbought + bearish divergence
        conditions_exit_long.append(
            (dataframe['rsi_14'] > 70) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1)) &
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Giá cao hơn nhưng RSI giảm
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bollinger Band upper touch + rejection
        conditions_exit_long.append(
            (dataframe['high'] > dataframe['bb_20_upper_2.0']) &
            (dataframe['close'] < (dataframe['high'] - 0.5 * (dataframe['high'] - dataframe['low']))) &  # Rejection
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Hit ATR trailing stop
        conditions_exit_short.append(
            dataframe['hit_trailing_stop_short']
        )
        
        # 2. SMA cross up
        conditions_exit_short.append(
            dataframe['sma_cross_up']
        )
        
        # 3. ADX trend change to bullish
        conditions_exit_short.append(
            (dataframe['bullish_trend']) &
            (~dataframe['bullish_trend'].shift(1))
        )
        
        # 4. RSI oversold + bullish divergence
        conditions_exit_short.append(
            (dataframe['rsi_14'] < 30) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Giá thấp hơn nhưng RSI tăng
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bollinger Band lower touch + rejection
        conditions_exit_short.append(
            (dataframe['low'] < dataframe['bb_20_lower_2.0']) &
            (dataframe['close'] > (dataframe['low'] + 0.5 * (dataframe['high'] - dataframe['low']))) &  # Rejection
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ mạnh của xu hướng
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số từ config
        adx_period = self.adx_period.value
        adx_threshold = self.adx_threshold.value
        
        # Xác định ADX
        adx = last_candle[f'adx_{adx_period}']
        
        if side == 'long':
            # ADX rất mạnh và xu hướng
            if adx > adx_threshold + 15 and last_candle['uptrend']:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # ADX mạnh và xu hướng tăng
            elif adx > adx_threshold + 5 and last_candle['uptrend']:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # ADX vừa phải và xu hướng tăng
            elif adx > adx_threshold and last_candle['uptrend']:
                leverage = min(3, max_leverage)  # Đòn bẩy mặc định
            
            # Xu hướng không rõ ràng
            else:
                leverage = min(2, max_leverage)  # Đòn bẩy thấp
                
        else:  # short
            # ADX rất mạnh và xu hướng
            if adx > adx_threshold + 15 and last_candle['downtrend']:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # ADX mạnh và xu hướng giảm
            elif adx > adx_threshold + 5 and last_candle['downtrend']:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # ADX vừa phải và xu hướng giảm
            elif adx > adx_threshold and last_candle['downtrend']:
                leverage = min(3, max_leverage)  # Đòn bẩy mặc định
            
            # Xu hướng không rõ ràng
            else:
                leverage = min(2, max_leverage)  # Đòn bẩy thấp
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên ATR Trailing Stop
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        if not trade.is_short:
            # Long trade: sử dụng ATR trailing stop
            atr_stop = last_candle['atr_trailing_stop_long']
            sl_distance = (atr_stop / current_rate) - 1
            
            # Không để stoploss xa hơn giá trị mặc định
            return max(sl_distance, self.stoploss)
        else:
            # Short trade: sử dụng ATR trailing stop
            atr_stop = last_candle['atr_trailing_stop_short']
            sl_distance = 1 - (atr_stop / current_rate)
            
            # Không để stoploss xa hơn giá trị mặc định
            return max(sl_distance, self.stoploss)
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận vào lệnh với lọc thêm
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        
        # Kiểm tra volume
        if last_candle['volume_ratio'] < 0.5:
            return False
        
        # Kiểm tra biến động quá cao
        if last_candle['bb_20_width_2.0'] > 0.1:  # BB width quá rộng
            return False
        
        # Kiểm tra gap quá lớn
        if side == 'long' and rate > last_candle['close'] * 1.03:  # Giá vào lệnh cao hơn close 3%
            return False
        
        if side == 'short' and rate < last_candle['close'] * 0.97:  # Giá vào lệnh thấp hơn close 3%
            return False
        
        return True
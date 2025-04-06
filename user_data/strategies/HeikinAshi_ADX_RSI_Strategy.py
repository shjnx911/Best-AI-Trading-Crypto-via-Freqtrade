"""
Heikin-Ashi + ADX + RSI Strategy
Chiến lược kết hợp nến Heikin-Ashi với ADX và RSI
Tối ưu cho khung thời gian 5m
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

def heikin_ashi(dataframe):
    """
    Chuyển đổi nến thông thường thành nến Heikin-Ashi
    """
    heikin_ashi_df = dataframe.copy()
    
    # Tính giá của nến Heikin-Ashi
    heikin_ashi_df['ha_close'] = (dataframe['open'] + dataframe['high'] + 
                                  dataframe['low'] + dataframe['close']) / 4
    
    # Sử dụng giá OHLC thông thường cho nến đầu tiên
    heikin_ashi_df.loc[0, 'ha_open'] = dataframe['open'].iloc[0]
    
    # Tính ha_open từ nến thứ 2 trở đi
    for i in range(1, len(heikin_ashi_df)):
        heikin_ashi_df.loc[i, 'ha_open'] = (heikin_ashi_df['ha_open'].iloc[i-1] + 
                                            heikin_ashi_df['ha_close'].iloc[i-1]) / 2
    
    # Tính ha_high và ha_low
    heikin_ashi_df['ha_high'] = heikin_ashi_df[['high', 'ha_open', 'ha_close']].max(axis=1)
    heikin_ashi_df['ha_low'] = heikin_ashi_df[['low', 'ha_open', 'ha_close']].min(axis=1)
    
    return heikin_ashi_df

def consecutive_candles(dataframe, num_candles=3, which='green'):
    """
    Xác định chuỗi nến liên tiếp cùng màu
    which: 'green' hoặc 'red'
    """
    
    if which == 'green':
        # Nến xanh: close > open
        candle_condition = dataframe['ha_close'] > dataframe['ha_open']
    else:
        # Nến đỏ: close < open
        candle_condition = dataframe['ha_close'] < dataframe['ha_open']
    
    # Khởi tạo cột kết quả
    result = np.zeros(len(dataframe))
    
    # Đếm số nến liên tiếp
    count = 0
    
    for i in range(len(dataframe)):
        if candle_condition.iloc[i]:
            count += 1
        else:
            count = 0
        
        # Khi đạt đủ số lượng nến liên tiếp
        if count >= num_candles:
            result[i] = 1
    
    return pd.Series(result, index=dataframe.index)

class HeikinAshi_ADX_RSI_Strategy(IStrategy):
    """
    Chiến lược kết hợp nến Heikin-Ashi với ADX và RSI
    
    - Sử dụng nến Heikin-Ashi để lọc nhiễu và xác định xu hướng rõ ràng
    - Kết hợp với ADX để đo lường sức mạnh xu hướng
    - Thêm RSI để xác định vùng quá mua/quá bán
    - Tối ưu cho giao dịch ngắn hạn (5m)
    """
    
    # Tham số cho Heikin-Ashi
    ha_consecutive_green = IntParameter(2, 6, default=3, space="buy", optimize=True)
    ha_consecutive_red = IntParameter(2, 6, default=3, space="buy", optimize=True)
    
    # Tham số cho ADX
    adx_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 35, default=25, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho EMA
    ema_fast = IntParameter(5, 30, default=8, space="buy", optimize=True)
    ema_slow = IntParameter(15, 50, default=21, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.02,     # 2%
        "10": 0.015,   # 1.5% sau 10 phút
        "20": 0.01,    # 1% sau 20 phút
        "30": 0.005    # 0.5% sau 30 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Chuyển đổi sang nến Heikin-Ashi
        ha_df = heikin_ashi(dataframe)
        
        # Thêm các cột Heikin-Ashi vào dataframe gốc
        dataframe['ha_open'] = ha_df['ha_open']
        dataframe['ha_close'] = ha_df['ha_close']
        dataframe['ha_high'] = ha_df['ha_high']
        dataframe['ha_low'] = ha_df['ha_low']
        
        # Xác định màu nến Heikin-Ashi
        dataframe['ha_color'] = np.where(dataframe['ha_close'] >= dataframe['ha_open'], 'green', 'red')
        
        # Thêm ADX
        for period in range(7, 22, 7):  # [7, 14, 21]
            adx = ta.trend.ADXIndicator(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=period
            )
            dataframe[f'adx_{period}'] = adx.adx()
            dataframe[f'plus_di_{period}'] = adx.adx_pos()
            dataframe[f'minus_di_{period}'] = adx.adx_neg()
        
        # Thêm RSI
        for period in range(7, 22, 7):  # [7, 14, 21]
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=period
            )
        
        # Thêm EMA
        for length in range(5, 51, 5):  # [5, 10, 15, ..., 50]
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'],
                window=length
            )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=dataframe['close'],
            window=20,
            window_dev=2
        )
        dataframe['bb_upperband'] = bollinger.bollinger_hband()
        dataframe['bb_lowerband'] = bollinger.bollinger_lband()
        dataframe['bb_middleband'] = bollinger.bollinger_mavg()
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        
        # Tính chuỗi nến liên tiếp
        dataframe['consecutive_green'] = consecutive_candles(dataframe, num_candles=self.ha_consecutive_green.value, which='green')
        dataframe['consecutive_red'] = consecutive_candles(dataframe, num_candles=self.ha_consecutive_red.value, which='red')
        
        # Lấy tham số từ config
        adx_period = self.adx_period.value
        adx_threshold = self.adx_threshold.value
        rsi_period = self.rsi_period.value
        ema_fast = self.ema_fast.value
        ema_slow = self.ema_slow.value
        
        # Xác định xu hướng theo EMA
        dataframe['uptrend'] = dataframe[f'ema_{ema_fast}'] > dataframe[f'ema_{ema_slow}']
        dataframe['downtrend'] = dataframe[f'ema_{ema_fast}'] < dataframe[f'ema_{ema_slow}']
        
        # Xác định tín hiệu ADX
        dataframe['adx'] = dataframe[f'adx_{adx_period}']
        dataframe['plus_di'] = dataframe[f'plus_di_{adx_period}']
        dataframe['minus_di'] = dataframe[f'minus_di_{adx_period}']
        
        dataframe['strong_trend'] = dataframe['adx'] > adx_threshold
        dataframe['bullish_trend'] = (dataframe['plus_di'] > dataframe['minus_di']) & dataframe['strong_trend']
        dataframe['bearish_trend'] = (dataframe['minus_di'] > dataframe['plus_di']) & dataframe['strong_trend']
        
        # Xác định tín hiệu Heikin-Ashi + ADX + RSI
        # 1. Strong bullish trend: chuỗi nến xanh HA + ADX cao + RSI không quá cao
        dataframe['ha_strong_bullish'] = (
            (dataframe['consecutive_green'] == 1) &
            (dataframe['bullish_trend']) &
            (dataframe[f'rsi_{rsi_period}'] < 70)
        )
        
        # 2. Strong bearish trend: chuỗi nến đỏ HA + ADX cao + RSI không quá thấp
        dataframe['ha_strong_bearish'] = (
            (dataframe['consecutive_red'] == 1) &
            (dataframe['bearish_trend']) &
            (dataframe[f'rsi_{rsi_period}'] > 30)
        )
        
        # 3. Overbought: 3+ nến xanh HA + RSI cao + ADX giảm
        dataframe['ha_overbought'] = (
            (dataframe['consecutive_green'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] > 75) &
            (dataframe['adx'] < dataframe['adx'].shift(1))
        )
        
        # 4. Oversold: 3+ nến đỏ HA + RSI thấp + ADX giảm
        dataframe['ha_oversold'] = (
            (dataframe['consecutive_red'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] < 25) &
            (dataframe['adx'] < dataframe['adx'].shift(1))
        )
        
        # 5. HA Color change: từ đỏ sang xanh
        dataframe['ha_color_change_bullish'] = (
            (dataframe['ha_color'] == 'green') &
            (dataframe['ha_color'].shift(1) == 'red')
        )
        
        # 6. HA Color change: từ xanh sang đỏ
        dataframe['ha_color_change_bearish'] = (
            (dataframe['ha_color'] == 'red') &
            (dataframe['ha_color'].shift(1) == 'green')
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Lấy tham số từ config
        rsi_period = self.rsi_period.value
        rsi_buy = self.rsi_buy_threshold.value
        rsi_sell = self.rsi_sell_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Heikin-Ashi strong bullish trend
        conditions_long.append(
            (dataframe['ha_strong_bullish']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. HA color change (đỏ sang xanh) + RSI oversold + ADX tăng
        conditions_long.append(
            (dataframe['ha_color_change_bullish']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy + 5) &
            (dataframe['adx'] > dataframe['adx'].shift(1)) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 3. Heikin-Ashi oversold + EMA uptrend
        conditions_long.append(
            (dataframe['ha_oversold']) &
            (dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bollinger Band bounce trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['low'] < dataframe['bb_lowerband'] * 1.005) &
            (dataframe['ha_color'] == 'green') &
            (dataframe[f'rsi_{rsi_period}'] < 50) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Strong RSI oversold bounce with green HA candle
        conditions_long.append(
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy) &
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1)) &
            (dataframe['ha_color'] == 'green') &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Heikin-Ashi strong bearish trend
        conditions_short.append(
            (dataframe['ha_strong_bearish']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. HA color change (xanh sang đỏ) + RSI overbought + ADX tăng
        conditions_short.append(
            (dataframe['ha_color_change_bearish']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell - 5) &
            (dataframe['adx'] > dataframe['adx'].shift(1)) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 3. Heikin-Ashi overbought + EMA downtrend
        conditions_short.append(
            (dataframe['ha_overbought']) &
            (dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bollinger Band bounce trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['high'] > dataframe['bb_upperband'] * 0.995) &
            (dataframe['ha_color'] == 'red') &
            (dataframe[f'rsi_{rsi_period}'] > 50) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Strong RSI overbought drop with red HA candle
        conditions_short.append(
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell) &
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1)) &
            (dataframe['ha_color'] == 'red') &
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
        
        # Lấy tham số từ config
        rsi_period = self.rsi_period.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. HA color change (từ xanh sang đỏ)
        conditions_exit_long.append(
            (dataframe['ha_color_change_bearish']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI overbought
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_period}'] > 75) &
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang giảm
        )
        
        # 3. Consecutive red HA candles
        conditions_exit_long.append(
            (dataframe['consecutive_red'] == 1) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. ADX bearish cross
        conditions_exit_long.append(
            (dataframe['minus_di'] > dataframe['plus_di']) &
            (dataframe['minus_di'].shift(1) <= dataframe['plus_di'].shift(1))
        )
        
        # 5. EMA fast crosses below slow
        conditions_exit_long.append(
            (dataframe[f'ema_{self.ema_fast.value}'] < dataframe[f'ema_{self.ema_slow.value}']) &
            (dataframe[f'ema_{self.ema_fast.value}'].shift(1) >= dataframe[f'ema_{self.ema_slow.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. HA color change (từ đỏ sang xanh)
        conditions_exit_short.append(
            (dataframe['ha_color_change_bullish']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI oversold
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_period}'] < 25) &
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang tăng
        )
        
        # 3. Consecutive green HA candles
        conditions_exit_short.append(
            (dataframe['consecutive_green'] == 1) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. ADX bullish cross
        conditions_exit_short.append(
            (dataframe['plus_di'] > dataframe['minus_di']) &
            (dataframe['plus_di'].shift(1) <= dataframe['minus_di'].shift(1))
        )
        
        # 5. EMA fast crosses above slow
        conditions_exit_short.append(
            (dataframe[f'ema_{self.ema_fast.value}'] > dataframe[f'ema_{self.ema_slow.value}']) &
            (dataframe[f'ema_{self.ema_fast.value}'].shift(1) <= dataframe[f'ema_{self.ema_slow.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ mạnh của tín hiệu Heikin-Ashi và ADX
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy tham số từ config
        adx_threshold = self.adx_threshold.value
        
        if side == 'long':
            # HA strong bullish + high ADX
            if last_candle['ha_strong_bullish'] and last_candle['adx'] > adx_threshold + 10:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # HA color change bullish + ADX increasing
            elif last_candle['ha_color_change_bullish'] and last_candle['adx'] > last_candle['adx'].shift(1):
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # Consecutive green HA candles + ADX above threshold
            elif last_candle['consecutive_green'] == 1 and last_candle['adx'] > adx_threshold:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # RSI oversold + green HA candle
            elif last_candle[f'rsi_{self.rsi_period.value}'] < self.rsi_buy_threshold.value and last_candle['ha_color'] == 'green':
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # HA strong bearish + high ADX
            if last_candle['ha_strong_bearish'] and last_candle['adx'] > adx_threshold + 10:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # HA color change bearish + ADX increasing
            elif last_candle['ha_color_change_bearish'] and last_candle['adx'] > last_candle['adx'].shift(1):
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # Consecutive red HA candles + ADX above threshold
            elif last_candle['consecutive_red'] == 1 and last_candle['adx'] > adx_threshold:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # RSI overbought + red HA candle
            elif last_candle[f'rsi_{self.rsi_period.value}'] > self.rsi_sell_threshold.value and last_candle['ha_color'] == 'red':
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Heikin-Ashi, RSI và ADX
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận dương, điều chỉnh stoploss
        if current_profit > 0.01:  # > 1% lợi nhuận
            if not trade.is_short:
                # Long trade
                
                # Nếu có tín hiệu thoát rõ ràng từ HA
                if last_candle['ha_color_change_bearish'] or last_candle['consecutive_red'] == 1:
                    return -0.005  # Stoploss 0.5% để bảo vệ lợi nhuận
                
                # Nếu RSI quá cao
                if last_candle[f'rsi_{self.rsi_period.value}'] > 75:
                    return -0.01  # Stoploss 1%
                
                # Nếu ADX giảm mạnh
                if last_candle['adx'] < last_candle['adx'].shift(2) * 0.9:  # Giảm > 10%
                    return -0.01  # Stoploss 1%
                
                # Nếu lợi nhuận > 2%, đặt breakeven
                if current_profit > 0.02:
                    return 0.001  # Breakeven + 0.1%
                
            else:
                # Short trade
                
                # Nếu có tín hiệu thoát rõ ràng từ HA
                if last_candle['ha_color_change_bullish'] or last_candle['consecutive_green'] == 1:
                    return -0.005  # Stoploss 0.5% để bảo vệ lợi nhuận
                
                # Nếu RSI quá thấp
                if last_candle[f'rsi_{self.rsi_period.value}'] < 25:
                    return -0.01  # Stoploss 1%
                
                # Nếu ADX giảm mạnh
                if last_candle['adx'] < last_candle['adx'].shift(2) * 0.9:  # Giảm > 10%
                    return -0.01  # Stoploss 1%
                
                # Nếu lợi nhuận > 2%, đặt breakeven
                if current_profit > 0.02:
                    return 0.001  # Breakeven + 0.1%
        
        return self.stoploss  # Sử dụng stoploss mặc định
    
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
        
        # Kiểm tra biến động quá cao (BB width)
        if last_candle['bb_width'] > 0.1:  # BB width quá rộng
            return False
        
        # Kiểm tra RSI extreme
        rsi = last_candle[f'rsi_{self.rsi_period.value}']
        if side == 'long' and rsi > 75:  # không vào long khi RSI quá cao
            return False
        if side == 'short' and rsi < 25:  # không vào short khi RSI quá thấp
            return False
        
        # Kiểm tra nến HA phù hợp
        if side == 'long' and last_candle['ha_color'] == 'red':  # không vào long khi nến HA đỏ
            return False
        if side == 'short' and last_candle['ha_color'] == 'green':  # không vào short khi nến HA xanh
            return False
        
        return True
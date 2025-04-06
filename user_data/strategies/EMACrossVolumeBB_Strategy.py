"""
EMA Cross + Volume + Bollinger Bands Strategy
Chiến lược kết hợp giữa EMA Cross, phân tích khối lượng và Bollinger Bands
Tối ưu cho khung thời gian 15m
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

logger = logging.getLogger(__name__)

class EMACrossVolumeBB_Strategy(IStrategy):
    """
    Chiến lược kết hợp EMA Cross, Volume và Bollinger Bands
    
    - Sử dụng cắt EMA để xác định xu hướng
    - Bollinger Bands để xác định vùng quá mua/quá bán và volatility squeeze
    - Volume để lọc và xác nhận tín hiệu
    """
    
    # Tham số EMA
    buy_ema_short = IntParameter(3, 15, default=8, space="buy", optimize=True)
    buy_ema_long = IntParameter(15, 50, default=21, space="buy", optimize=True)
    sell_ema_short = IntParameter(3, 15, default=8, space="sell", optimize=True)
    sell_ema_long = IntParameter(15, 50, default=21, space="sell", optimize=True)
    
    # Tham số Bollinger Bands
    bb_period = IntParameter(15, 30, default=20, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=True)
    
    # Tham số Volume
    volume_factor = DecimalParameter(1.0, 5.0, default=2.0, space="buy", optimize=True)
    
    # Tham số RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '15m'
    
    minimal_roi = {
        "0": 0.05,    # 5%
        "30": 0.025,  # 2.5% sau 30 phút
        "60": 0.015,  # 1.5% sau 60 phút
        "90": 0.01    # 1% sau 90 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%
    trailing_stop_positive_offset = 0.02  # 2%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # EMA ngắn và dài
        for val in self.buy_ema_short.range:
            dataframe[f'ema_short_{val}'] = ta.trend.ema_indicator(
                close=dataframe['close'], window=val
            )
            
        for val in self.buy_ema_long.range:
            dataframe[f'ema_long_{val}'] = ta.trend.ema_indicator(
                close=dataframe['close'], window=val
            )
        
        # Bollinger Bands
        for val in self.bb_period.range:
            for std in [2.0, 2.5, 3.0]:
                bollinger = ta.volatility.BollingerBands(
                    close=dataframe['close'], window=val, window_dev=std
                )
                dataframe[f'bb_lowerband_{val}_{std}'] = bollinger.bollinger_lband()
                dataframe[f'bb_middleband_{val}_{std}'] = bollinger.bollinger_mavg()
                dataframe[f'bb_upperband_{val}_{std}'] = bollinger.bollinger_hband()
                dataframe[f'bb_width_{val}_{std}'] = (
                    dataframe[f'bb_upperband_{val}_{std}'] - dataframe[f'bb_lowerband_{val}_{std}']
                ) / dataframe[f'bb_middleband_{val}_{std}']
                
                # Phát hiện Bollinger Bands squeeze (squeeze = bands thắt chặt)
                dataframe[f'bb_squeeze_{val}_{std}'] = (
                    dataframe[f'bb_width_{val}_{std}'] < 
                    dataframe[f'bb_width_{val}_{std}'].rolling(window=20).mean() * 0.8
                )
        
        # Chỉ báo volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Chỉ báo RSI
        for val in self.rsi_period.range:
            dataframe[f'rsi_{val}'] = ta.momentum.rsi(close=dataframe['close'], window=val)
            
        # MACD
        macd = ta.trend.MACD(
            close=dataframe['close'], window_slow=26, window_fast=12, window_sign=9
        )
        dataframe['macd'] = macd.macd()
        dataframe['macdsignal'] = macd.macd_signal()
        dataframe['macdhist'] = macd.macd_diff()
        
        # OBV (On Balance Volume)
        dataframe['obv'] = ta.volume.on_balance_volume(
            close=dataframe['close'], volume=dataframe['volume']
        )
        dataframe['obv_ema'] = ta.trend.ema_indicator(close=dataframe['obv'], window=21)
        
        # Đánh dấu xu hướng
        ema_short = self.buy_ema_short.value
        ema_long = self.buy_ema_long.value
        
        dataframe['uptrend'] = (
            dataframe[f'ema_short_{ema_short}'] > dataframe[f'ema_long_{ema_long}']
        )
        dataframe['downtrend'] = (
            dataframe[f'ema_short_{ema_short}'] < dataframe[f'ema_long_{ema_long}']
        )
        
        # EMA Cross
        dataframe['ema_cross_up'] = (
            (dataframe[f'ema_short_{ema_short}'] > dataframe[f'ema_long_{ema_long}']) &
            (dataframe[f'ema_short_{ema_short}'].shift(1) <= dataframe[f'ema_long_{ema_long}'].shift(1))
        )
        dataframe['ema_cross_down'] = (
            (dataframe[f'ema_short_{ema_short}'] < dataframe[f'ema_long_{ema_long}']) &
            (dataframe[f'ema_short_{ema_short}'].shift(1) >= dataframe[f'ema_long_{ema_long}'].shift(1))
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số cho các điều kiện
        ema_short = self.buy_ema_short.value
        ema_long = self.buy_ema_long.value
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        volume_factor = self.volume_factor.value
        rsi_period = self.rsi_period.value
        rsi_buy = self.rsi_buy_threshold.value
        rsi_sell = self.rsi_sell_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. EMA Cross Up + RSI thấp
        conditions_long.append(
            (dataframe['ema_cross_up']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # 2. Giá chạm BB lower trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['close'] <= dataframe[f'bb_lowerband_{bb_period}_{bb_std}']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # 3. BB Squeeze followed by expansion
        conditions_long.append(
            (dataframe[f'bb_squeeze_{bb_period}_{bb_std}'].shift(1)) &
            (~dataframe[f'bb_squeeze_{bb_period}_{bb_std}']) &
            (dataframe['close'] > dataframe[f'bb_middleband_{bb_period}_{bb_std}']) &
            (dataframe['volume_ratio'] > volume_factor * 1.5)
        )
        
        # 4. OBV xác nhận xu hướng
        conditions_long.append(
            (dataframe['obv'] > dataframe['obv_ema']) &
            (dataframe['uptrend']) &
            (dataframe['close'] > dataframe[f'ema_short_{ema_short}']) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. EMA Cross Down + RSI cao
        conditions_short.append(
            (dataframe['ema_cross_down']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # 2. Giá chạm BB upper trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['close'] >= dataframe[f'bb_upperband_{bb_period}_{bb_std}']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # 3. BB Squeeze followed by expansion
        conditions_short.append(
            (dataframe[f'bb_squeeze_{bb_period}_{bb_std}'].shift(1)) &
            (~dataframe[f'bb_squeeze_{bb_period}_{bb_std}']) &
            (dataframe['close'] < dataframe[f'bb_middleband_{bb_period}_{bb_std}']) &
            (dataframe['volume_ratio'] > volume_factor * 1.5)
        )
        
        # 4. OBV xác nhận xu hướng
        conditions_short.append(
            (dataframe['obv'] < dataframe['obv_ema']) &
            (dataframe['downtrend']) &
            (dataframe['close'] < dataframe[f'ema_short_{ema_short}']) &
            (dataframe['volume_ratio'] > volume_factor)
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Tham số cho các điều kiện
        ema_short = self.sell_ema_short.value
        ema_long = self.sell_ema_long.value
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        rsi_period = self.rsi_period.value
        rsi_sell = self.rsi_sell_threshold.value
        rsi_buy = self.rsi_buy_threshold.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. EMA Cross Down
        conditions_exit_long.append(dataframe['ema_cross_down'])
        
        # 2. Chạm BB upper + RSI cao
        conditions_exit_long.append(
            (dataframe['close'] >= dataframe[f'bb_upperband_{bb_period}_{bb_std}']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell)
        )
        
        # 3. MACD cross down
        conditions_exit_long.append(
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1))
        )
        
        # 4. OBV giảm mạnh
        conditions_exit_long.append(
            (dataframe['obv'] < dataframe['obv'].shift(3) * 0.95)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. EMA Cross Up
        conditions_exit_short.append(dataframe['ema_cross_up'])
        
        # 2. Chạm BB lower + RSI thấp
        conditions_exit_short.append(
            (dataframe['close'] <= dataframe[f'bb_lowerband_{bb_period}_{bb_std}']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy)
        )
        
        # 3. MACD cross up
        conditions_exit_short.append(
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1))
        )
        
        # 4. OBV tăng mạnh
        conditions_exit_short.append(
            (dataframe['obv'] > dataframe['obv'].shift(3) * 1.05)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên biến động (Bollinger Band width)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có dữ liệu
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        
        # Lấy BB width hiện tại
        bb_width = last_candle[f'bb_width_{bb_period}_{bb_std}']
        
        # Điều chỉnh đòn bẩy dựa trên BB width
        # BB width nhỏ = biến động thấp = đòn bẩy cao hơn
        # BB width lớn = biến động cao = đòn bẩy thấp hơn
        
        if bb_width < 0.03:  # Biến động rất thấp
            leverage = min(self.leverage_max, max_leverage)
        elif bb_width < 0.05:  # Biến động thấp
            leverage = min(5, max_leverage)
        elif bb_width < 0.08:  # Biến động trung bình
            leverage = min(4, max_leverage)
        else:  # Biến động cao
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Bollinger Bands
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        
        # Với lợi nhuận tích cực, sử dụng Bollinger Band làm stoploss động
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại middle band
                bb_middleband = last_candle[f'bb_middleband_{bb_period}_{bb_std}']
                return (bb_middleband / current_rate) - 1
            else:
                # Short trade: stoploss tại middle band
                bb_middleband = last_candle[f'bb_middleband_{bb_period}_{bb_std}']
                return 1 - (bb_middleband / current_rate)
        
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
        
        # Lọc theo khối lượng
        if last_candle['volume'] < last_candle['volume_mean'] * 0.5:
            return False
        
        # Không vào lệnh khi BB quá rộng (biến động cao)
        bb_period = self.bb_period.value
        bb_std = self.bb_std.value
        if last_candle[f'bb_width_{bb_period}_{bb_std}'] > 0.1:
            return False
        
        return True
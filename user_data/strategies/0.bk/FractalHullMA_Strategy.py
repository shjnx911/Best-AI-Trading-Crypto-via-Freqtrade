"""
Fractal + Hull Moving Average Strategy
Chiến lược kết hợp Fractals của Bill Williams với Hull Moving Average
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

logger = logging.getLogger(__name__)

def wma(series, period):
    """Weighted Moving Average"""
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)

def hull_moving_average(dataframe, period):
    """
    Hull Moving Average
    Formula: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    """
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(dataframe, half_period)
    wma_full = wma(dataframe, period)
    
    hull = 2 * wma_half - wma_full
    return wma(hull, sqrt_period)

def detect_fractals(dataframe, window=2):
    """
    Phát hiện Fractals của Bill Williams
    
    - Fractal up: Đỉnh cục bộ, high cao hơn 'window' nến hai bên
    - Fractal down: Đáy cục bộ, low thấp hơn 'window' nến hai bên
    """
    dataframe['fractal_up'] = False
    dataframe['fractal_down'] = False
    
    for i in range(window, len(dataframe) - window):
        # Phát hiện fractal up (bullish)
        if all(dataframe['high'].iloc[i] > dataframe['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(dataframe['high'].iloc[i] > dataframe['high'].iloc[i+j] for j in range(1, window+1)):
            dataframe['fractal_up'].iloc[i] = True
        
        # Phát hiện fractal down (bearish)
        if all(dataframe['low'].iloc[i] < dataframe['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(dataframe['low'].iloc[i] < dataframe['low'].iloc[i+j] for j in range(1, window+1)):
            dataframe['fractal_down'].iloc[i] = True
    
    # Giá trị Fractal
    dataframe['fractal_up_value'] = np.where(dataframe['fractal_up'], dataframe['high'], np.nan)
    dataframe['fractal_down_value'] = np.where(dataframe['fractal_down'], dataframe['low'], np.nan)
    
    # Forward fill để có giá trị Fractal gần nhất
    dataframe['last_fractal_up'] = dataframe['fractal_up_value'].fillna(method='ffill')
    dataframe['last_fractal_down'] = dataframe['fractal_down_value'].fillna(method='ffill')
    
    return dataframe

class FractalHullMA_Strategy(IStrategy):
    """
    Chiến lược kết hợp Fractals với Hull Moving Average
    
    - Sử dụng Fractals của Bill Williams để xác định điểm đảo chiều
    - Kết hợp với Hull Moving Average để xác định xu hướng
    - Xác định entry points ở các fractal gần HMA trong xu hướng
    - Tối ưu cho giao dịch xu hướng ngắn (5m)
    """
    
    # Tham số cho Fractals
    fractal_window = IntParameter(1, 5, default=2, space="buy", optimize=True)
    
    # Tham số cho Hull Moving Average
    hma_length_fast = IntParameter(6, 24, default=9, space="buy", optimize=True)
    hma_length_slow = IntParameter(25, 100, default=49, space="buy", optimize=True)
    
    # Tham số cho xác định tín hiệu
    fractal_distance = DecimalParameter(0.1, 1.0, default=0.3, space="buy", optimize=True)
    hma_angle_threshold = DecimalParameter(0.1, 1.0, default=0.3, space="buy", optimize=True)
    
    # Tham số cho Dynamic Trailing
    close_above_hma_pct = DecimalParameter(0.1, 1.0, default=0.5, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 7
    stoploss = -0.025
    
    # Time-in-force và ROI
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.03,    # 3%
        "15": 0.02,   # 2% sau 15 phút
        "30": 0.01,   # 1% sau 30 phút
        "45": 0.005   # 0.5% sau 45 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.008  # 0.8%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Tính HMA trên các khung thời gian khác nhau
        for length in range(6, 101, 3):
            dataframe[f'hma_{length}'] = hull_moving_average(dataframe['close'], length)
            
            # Tính độ dốc (angle) của HMA
            dataframe[f'hma_{length}_angle'] = 100 * (dataframe[f'hma_{length}'] - dataframe[f'hma_{length}'].shift(3)) / dataframe[f'hma_{length}'].shift(3)
        
        # Phát hiện Fractals
        for window in range(1, 6):
            dataframe = detect_fractals(dataframe, window=window)
            
            # Thêm thông tin khoảng cách từ fractal đến HMA
            for length in [9, 16, 25, 49, 81]:
                # Khoảng cách % từ fractal up đến HMA
                dataframe[f'fractal_up_hma_{length}_dist_{window}'] = 100 * (dataframe['last_fractal_up'] - dataframe[f'hma_{length}']) / dataframe[f'hma_{length}']
                
                # Khoảng cách % từ fractal down đến HMA
                dataframe[f'fractal_down_hma_{length}_dist_{window}'] = 100 * (dataframe[f'hma_{length}'] - dataframe['last_fractal_down']) / dataframe[f'hma_{length}']
        
        # Thêm các chỉ báo bổ sung
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=14)
        
        # ATR - Average True Range
        dataframe['atr'] = ta.volatility.average_true_range(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            window=14
        )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Lấy các giá trị từ tham số
        hma_length_fast = self.hma_length_fast.value
        hma_length_slow = self.hma_length_slow.value
        fractal_window = self.fractal_window.value
        
        # Tính toán tín hiệu
        # Uptrend: Fast HMA trên Slow HMA + Fast HMA góc dương
        dataframe['uptrend'] = (
            (dataframe[f'hma_{hma_length_fast}'] > dataframe[f'hma_{hma_length_slow}']) &
            (dataframe[f'hma_{hma_length_fast}_angle'] > self.hma_angle_threshold.value)
        )
        
        # Downtrend: Fast HMA dưới Slow HMA + Fast HMA góc âm
        dataframe['downtrend'] = (
            (dataframe[f'hma_{hma_length_fast}'] < dataframe[f'hma_{hma_length_slow}']) &
            (dataframe[f'hma_{hma_length_fast}_angle'] < -self.hma_angle_threshold.value)
        )
        
        # Long Signal: Uptrend + Fractal down gần HMA (within fractal_distance %)
        dataframe['fractal_long_signal'] = (
            dataframe['uptrend'] &
            (dataframe[f'fractal_down_hma_{hma_length_fast}_dist_{fractal_window}'] < self.fractal_distance.value) &
            (dataframe[f'fractal_down_hma_{hma_length_fast}_dist_{fractal_window}'] > 0)
        )
        
        # Short Signal: Downtrend + Fractal up gần HMA (within fractal_distance %)
        dataframe['fractal_short_signal'] = (
            dataframe['downtrend'] &
            (dataframe[f'fractal_up_hma_{hma_length_fast}_dist_{fractal_window}'] < self.fractal_distance.value) &
            (dataframe[f'fractal_up_hma_{hma_length_fast}_dist_{fractal_window}'] > 0)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số từ config
        hma_length_fast = self.hma_length_fast.value
        fractal_window = self.fractal_window.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Fractal Long Signal - Uptrend + Fractal down gần HMA
        conditions_long.append(
            (dataframe['fractal_long_signal']) &
            (dataframe['volume_ratio'] > 0.8) &
            (dataframe['rsi'] > 40)
        )
        
        # 2. Pullback to HMA trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['close'] < dataframe[f'hma_{hma_length_fast}'] * 1.005) &
            (dataframe['close'] > dataframe[f'hma_{hma_length_fast}'] * 0.995) &
            (dataframe['rsi'] > 40) &
            (dataframe['rsi'] < 60) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Breakout trên HMA sau fractal down
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['fractal_down'].shift(1) | dataframe['fractal_down'].shift(2)) &
            (dataframe['close'] > dataframe[f'hma_{hma_length_fast}']) &
            (dataframe['close'].shift(1) < dataframe[f'hma_{hma_length_fast}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Fractal Short Signal - Downtrend + Fractal up gần HMA
        conditions_short.append(
            (dataframe['fractal_short_signal']) &
            (dataframe['volume_ratio'] > 0.8) &
            (dataframe['rsi'] < 60)
        )
        
        # 2. Pullback to HMA trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['close'] > dataframe[f'hma_{hma_length_fast}'] * 0.995) &
            (dataframe['close'] < dataframe[f'hma_{hma_length_fast}'] * 1.005) &
            (dataframe['rsi'] < 60) &
            (dataframe['rsi'] > 40) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Breakdown dưới HMA sau fractal up
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['fractal_up'].shift(1) | dataframe['fractal_up'].shift(2)) &
            (dataframe['close'] < dataframe[f'hma_{hma_length_fast}']) &
            (dataframe['close'].shift(1) > dataframe[f'hma_{hma_length_fast}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Tham số từ config
        hma_length_fast = self.hma_length_fast.value
        hma_length_slow = self.hma_length_slow.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Giá vượt HMA quá xa
        conditions_exit_long.append(
            (dataframe['close'] > dataframe[f'hma_{hma_length_fast}'] * (1 + self.close_above_hma_pct.value / 100)) &
            (dataframe['rsi'] > 70)
        )
        
        # 2. Fractal up mới xuất hiện
        conditions_exit_long.append(
            dataframe['fractal_up']
        )
        
        # 3. HMA chuyển xu hướng xuống
        conditions_exit_long.append(
            (dataframe[f'hma_{hma_length_fast}'] < dataframe[f'hma_{hma_length_slow}']) &
            (dataframe[f'hma_{hma_length_fast}'].shift(1) >= dataframe[f'hma_{hma_length_slow}'].shift(1))
        )
        
        # 4. RSI quá cao
        conditions_exit_long.append(
            (dataframe['rsi'] > 80)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Giá dưới HMA quá xa
        conditions_exit_short.append(
            (dataframe['close'] < dataframe[f'hma_{hma_length_fast}'] * (1 - self.close_above_hma_pct.value / 100)) &
            (dataframe['rsi'] < 30)
        )
        
        # 2. Fractal down mới xuất hiện
        conditions_exit_short.append(
            dataframe['fractal_down']
        )
        
        # 3. HMA chuyển xu hướng lên
        conditions_exit_short.append(
            (dataframe[f'hma_{hma_length_fast}'] > dataframe[f'hma_{hma_length_slow}']) &
            (dataframe[f'hma_{hma_length_fast}'].shift(1) <= dataframe[f'hma_{hma_length_slow}'].shift(1))
        )
        
        # 4. RSI quá thấp
        conditions_exit_short.append(
            (dataframe['rsi'] < 20)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên khoảng cách từ fractal đến HMA
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số
        hma_length_fast = self.hma_length_fast.value
        fractal_window = self.fractal_window.value
        
        # Lấy thông tin khoảng cách fractal
        if side == 'long':
            fractal_dist = last_candle[f'fractal_down_hma_{hma_length_fast}_dist_{fractal_window}']
            rsi = last_candle['rsi']
            
            # Fractal gần HMA + RSI thấp = đòn bẩy cao hơn
            if 0 < fractal_dist < 0.2 and rsi < 40:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif 0 < fractal_dist < 0.4 and rsi < 45:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif 0 < fractal_dist < 0.6:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            fractal_dist = last_candle[f'fractal_up_hma_{hma_length_fast}_dist_{fractal_window}']
            rsi = last_candle['rsi']
            
            # Fractal gần HMA + RSI cao = đòn bẩy cao hơn
            if 0 < fractal_dist < 0.2 and rsi > 60:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif 0 < fractal_dist < 0.4 and rsi > 55:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif 0 < fractal_dist < 0.6:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên fractal gần nhất
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận dương, sử dụng fractal gần nhất làm stoploss
        if current_profit > 0.01:  # > 1% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại fractal down gần nhất
                fractal_down = last_candle['last_fractal_down']
                if not np.isnan(fractal_down):
                    return (fractal_down / current_rate) - 1
            else:
                # Short trade: stoploss tại fractal up gần nhất
                fractal_up = last_candle['last_fractal_up']
                if not np.isnan(fractal_up):
                    return 1 - (fractal_up / current_rate)
        
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
        
        # Tham số
        hma_length_fast = self.hma_length_fast.value
        hma_length_slow = self.hma_length_slow.value
        
        # Không vào lệnh khi volume quá thấp
        if last_candle['volume_ratio'] < 0.5:
            return False
        
        # Không vào lệnh long khi HMA angle âm
        if side == 'long' and last_candle[f'hma_{hma_length_fast}_angle'] < 0:
            return False
            
        # Không vào lệnh short khi HMA angle dương
        if side == 'short' and last_candle[f'hma_{hma_length_fast}_angle'] > 0:
            return False
        
        return True
"""
Pivot Point + Fibonacci Strategy
Chiến lược kết hợp Pivot Points và Fibonacci Extensions/Retracements
Tối ưu cho khung thời gian 1h
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

def calculate_pivot_points(dataframe: pd.DataFrame, mode: str = 'fibonacci') -> pd.DataFrame:
    """
    Tính toán Pivot Points cho các phiên giao dịch.
    
    Các chế độ:
    - 'standard': Standard Pivot Points
    - 'fibonacci': Fibonacci Pivot Points
    - 'camarilla': Camarilla Pivot Points
    """
    # Tạo một bản sao để tránh thay đổi dataframe gốc
    df = dataframe.copy()
    
    # Tính giá trị Pivot Points (dựa trên high, low, close của phiên trước)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    
    # Pivot point (PP) - giống nhau cho tất cả các loại
    df['pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3
    
    if mode == 'standard':
        # Standard Pivot Points
        df['r1'] = 2 * df['pivot'] - df['prev_low']
        df['s1'] = 2 * df['pivot'] - df['prev_high']
        df['r2'] = df['pivot'] + (df['prev_high'] - df['prev_low'])
        df['s2'] = df['pivot'] - (df['prev_high'] - df['prev_low'])
        df['r3'] = df['pivot'] + 2 * (df['prev_high'] - df['prev_low'])
        df['s3'] = df['pivot'] - 2 * (df['prev_high'] - df['prev_low'])
        
    elif mode == 'fibonacci':
        # Fibonacci Pivot Points
        df['r1'] = df['pivot'] + 0.382 * (df['prev_high'] - df['prev_low'])
        df['s1'] = df['pivot'] - 0.382 * (df['prev_high'] - df['prev_low'])
        df['r2'] = df['pivot'] + 0.618 * (df['prev_high'] - df['prev_low'])
        df['s2'] = df['pivot'] - 0.618 * (df['prev_high'] - df['prev_low'])
        df['r3'] = df['pivot'] + 1.0 * (df['prev_high'] - df['prev_low'])
        df['s3'] = df['pivot'] - 1.0 * (df['prev_high'] - df['prev_low'])
        
    elif mode == 'camarilla':
        # Camarilla Pivot Points
        df['r1'] = df['prev_close'] + 1.1 * (df['prev_high'] - df['prev_low']) / 12
        df['s1'] = df['prev_close'] - 1.1 * (df['prev_high'] - df['prev_low']) / 12
        df['r2'] = df['prev_close'] + 1.1 * 2 * (df['prev_high'] - df['prev_low']) / 12
        df['s2'] = df['prev_close'] - 1.1 * 2 * (df['prev_high'] - df['prev_low']) / 12
        df['r3'] = df['prev_close'] + 1.1 * 3 * (df['prev_high'] - df['prev_low']) / 12
        df['s3'] = df['prev_close'] - 1.1 * 3 * (df['prev_high'] - df['prev_low']) / 12
        df['r4'] = df['prev_close'] + 1.1 * 4 * (df['prev_high'] - df['prev_low']) / 12
        df['s4'] = df['prev_close'] - 1.1 * 4 * (df['prev_high'] - df['prev_low']) / 12
    
    return df

def calculate_fibonacci_levels(dataframe: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Tính toán các mức Fibonacci Retracement và Extension
    dựa trên swing high/low gần nhất
    """
    df = dataframe.copy()
    
    # Tìm swing high và swing low trong cửa sổ
    df['rolling_max'] = df['high'].rolling(window=window).max()
    df['rolling_min'] = df['low'].rolling(window=window).min()
    
    # Xác định khi nào có swing high mới
    df['is_swing_high'] = (df['rolling_max'] == df['high'])
    
    # Xác định khi nào có swing low mới
    df['is_swing_low'] = (df['rolling_min'] == df['low'])
    
    # Forward fill để duy trì giá trị swing high/low gần nhất
    df['last_swing_high'] = df.loc[df['is_swing_high'], 'high']
    df['last_swing_high'] = df['last_swing_high'].fillna(method='ffill')
    
    df['last_swing_low'] = df.loc[df['is_swing_low'], 'low']
    df['last_swing_low'] = df['last_swing_low'].fillna(method='ffill')
    
    # Tính range của swing
    df['swing_range'] = df['last_swing_high'] - df['last_swing_low']
    
    # Fibonacci Retracement levels (giảm giá từ swing high)
    df['fib_23_6_retracement'] = df['last_swing_high'] - 0.236 * df['swing_range']
    df['fib_38_2_retracement'] = df['last_swing_high'] - 0.382 * df['swing_range']
    df['fib_50_0_retracement'] = df['last_swing_high'] - 0.5 * df['swing_range']
    df['fib_61_8_retracement'] = df['last_swing_high'] - 0.618 * df['swing_range']
    df['fib_78_6_retracement'] = df['last_swing_high'] - 0.786 * df['swing_range']
    
    # Fibonacci Extension levels (tăng giá từ swing low)
    df['fib_127_extension'] = df['last_swing_low'] + 1.27 * df['swing_range']
    df['fib_138_extension'] = df['last_swing_low'] + 1.382 * df['swing_range']
    df['fib_150_extension'] = df['last_swing_low'] + 1.5 * df['swing_range']
    df['fib_161_extension'] = df['last_swing_low'] + 1.618 * df['swing_range']
    df['fib_200_extension'] = df['last_swing_low'] + 2.0 * df['swing_range']
    
    # Fibonacci Extension levels (giảm giá từ swing high)
    df['fib_127_extension_down'] = df['last_swing_high'] - 1.27 * df['swing_range']
    df['fib_138_extension_down'] = df['last_swing_high'] - 1.382 * df['swing_range']
    df['fib_150_extension_down'] = df['last_swing_high'] - 1.5 * df['swing_range']
    df['fib_161_extension_down'] = df['last_swing_high'] - 1.618 * df['swing_range']
    df['fib_200_extension_down'] = df['last_swing_high'] - 2.0 * df['swing_range']
    
    return df

class PivotPointFib_Strategy(IStrategy):
    """
    Chiến lược kết hợp Pivot Points với Fibonacci
    
    - Sử dụng Pivot Points để xác định hỗ trợ/kháng cự key
    - Kết hợp với Fibonacci Retracement để xác định mức giá chốt lời và vào lệnh
    - Tìm các mức hội tụ giữa Pivot và Fibonacci để có tín hiệu mạnh hơn
    - Thích hợp cho giao dịch tầm trung (1h)
    """
    
    # Tham số cho Pivot Point
    pivot_type = CategoricalParameter(['standard', 'fibonacci', 'camarilla'], default='fibonacci', space="buy", optimize=True)
    
    # Tham số cho Fibonacci
    fib_window = IntParameter(10, 50, default=20, space="buy", optimize=True)
    
    # Tham số cho xác định tín hiệu
    pivot_bounce_pct = DecimalParameter(0.05, 0.5, default=0.1, space="buy", optimize=True)
    fib_tolerance_pct = DecimalParameter(0.05, 0.5, default=0.1, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.05
    
    # Time-in-force và ROI
    timeframe = '1h'
    
    minimal_roi = {
        "0": 0.05,     # 5%
        "12": 0.035,   # 3.5% sau 12 giờ
        "24": 0.025,   # 2.5% sau 24 giờ
        "48": 0.015    # 1.5% sau 48 giờ
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
        
        # Tính Pivot Points
        for pivot_mode in ['standard', 'fibonacci', 'camarilla']:
            pivot_df = calculate_pivot_points(dataframe, mode=pivot_mode)
            
            # Thêm Pivot Points vào dataframe
            dataframe[f'pivot_{pivot_mode}'] = pivot_df['pivot']
            for level in ['r1', 'r2', 'r3', 's1', 's2', 's3']:
                dataframe[f'{level}_{pivot_mode}'] = pivot_df[level]
            
            # Thêm r4/s4 nếu có (chỉ có ở Camarilla)
            if pivot_mode == 'camarilla':
                dataframe['r4_camarilla'] = pivot_df['r4']
                dataframe['s4_camarilla'] = pivot_df['s4']
        
        # Tính Fibonacci levels
        for window in range(10, 51, 10):
            fib_df = calculate_fibonacci_levels(dataframe, window=window)
            
            # Fibonacci Retracement
            for level in ['23_6', '38_2', '50_0', '61_8', '78_6']:
                dataframe[f'fib_{level}_retracement_{window}'] = fib_df[f'fib_{level}_retracement']
            
            # Fibonacci Extension
            for level in ['127', '138', '150', '161', '200']:
                dataframe[f'fib_{level}_extension_{window}'] = fib_df[f'fib_{level}_extension']
                dataframe[f'fib_{level}_extension_down_{window}'] = fib_df[f'fib_{level}_extension_down']
                
            # Swing high/low
            dataframe[f'swing_high_{window}'] = fib_df['last_swing_high']
            dataframe[f'swing_low_{window}'] = fib_df['last_swing_low']
            dataframe[f'swing_range_{window}'] = fib_df['swing_range']
            
        # Thêm các chỉ báo thông thường
        # RSI
        for period in range(7, 22):
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(close=dataframe['close'], window=period)
        
        # SMA
        for length in [20, 50, 100, 200]:
            dataframe[f'sma_{length}'] = ta.trend.sma_indicator(close=dataframe['close'], window=length)
        
        # ATR
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
        pivot_type_val = self.pivot_type.value
        fib_window_val = self.fib_window.value
        rsi_period_val = self.rsi_period.value
        pivot_bounce_pct_val = self.pivot_bounce_pct.value
        fib_tolerance_pct_val = self.fib_tolerance_pct.value
        
        # Xác định các confluence zones (vùng hội tụ) giữa Pivot và Fibonacci
        # 1. Confluent Support Zones
        dataframe['support_pivot_fib'] = False
        
        for s_level in ['s1', 's2', 's3']:
            pivot_level = dataframe[f'{s_level}_{pivot_type_val}']
            
            # Kiểm tra hội tụ với các mức fibonacci retracement
            for fib_level in ['38_2', '50_0', '61_8']:
                fib_level_val = dataframe[f'fib_{fib_level}_retracement_{fib_window_val}']
                
                # Xác định nếu pivot và fib ở gần nhau
                pct_diff = 100 * abs(pivot_level - fib_level_val) / pivot_level
                is_confluent = pct_diff < fib_tolerance_pct_val
                
                dataframe['support_pivot_fib'] = dataframe['support_pivot_fib'] | is_confluent
        
        # 2. Confluent Resistance Zones
        dataframe['resistance_pivot_fib'] = False
        
        for r_level in ['r1', 'r2', 'r3']:
            pivot_level = dataframe[f'{r_level}_{pivot_type_val}']
            
            # Kiểm tra hội tụ với các mức fibonacci extension
            for fib_level in ['127', '138', '161']:
                fib_level_val = dataframe[f'fib_{fib_level}_extension_{fib_window_val}']
                
                # Xác định nếu pivot và fib ở gần nhau
                pct_diff = 100 * abs(pivot_level - fib_level_val) / pivot_level
                is_confluent = pct_diff < fib_tolerance_pct_val
                
                dataframe['resistance_pivot_fib'] = dataframe['resistance_pivot_fib'] | is_confluent
        
        # Xác định khi giá gần mức pivot/fibonacci
        # 1. Gần mức hỗ trợ (support)
        dataframe['near_support_pivot'] = False
        for s_level in ['s1', 's2', 's3']:
            pivot_level = dataframe[f'{s_level}_{pivot_type_val}']
            pct_diff = 100 * (dataframe['close'] - pivot_level) / pivot_level
            is_near = (-pivot_bounce_pct_val <= pct_diff) & (pct_diff <= pivot_bounce_pct_val)
            dataframe['near_support_pivot'] = dataframe['near_support_pivot'] | is_near
        
        # 2. Gần mức kháng cự (resistance)
        dataframe['near_resistance_pivot'] = False
        for r_level in ['r1', 'r2', 'r3']:
            pivot_level = dataframe[f'{r_level}_{pivot_type_val}']
            pct_diff = 100 * (dataframe['close'] - pivot_level) / pivot_level
            is_near = (-pivot_bounce_pct_val <= pct_diff) & (pct_diff <= pivot_bounce_pct_val)
            dataframe['near_resistance_pivot'] = dataframe['near_resistance_pivot'] | is_near
        
        # Xác định xu hướng dựa trên SMA
        dataframe['uptrend'] = (
            (dataframe['close'] > dataframe['sma_50']) &
            (dataframe['sma_20'] > dataframe['sma_50'])
        )
        
        dataframe['downtrend'] = (
            (dataframe['close'] < dataframe['sma_50']) &
            (dataframe['sma_20'] < dataframe['sma_50'])
        )
        
        # Phát hiện bounce từ pivot levels
        # 1. Bounce từ support lên (cho long)
        dataframe['bounce_from_support'] = (
            (dataframe['near_support_pivot']) &
            (dataframe['close'] > dataframe['open']) &  # Nến tăng
            (dataframe['close'].shift(1) < dataframe['open'].shift(1))  # Nến trước giảm
        )
        
        # 2. Bounce từ resistance xuống (cho short)
        dataframe['bounce_from_resistance'] = (
            (dataframe['near_resistance_pivot']) &
            (dataframe['close'] < dataframe['open']) &  # Nến giảm
            (dataframe['close'].shift(1) > dataframe['open'].shift(1))  # Nến trước tăng
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số từ config
        pivot_type_val = self.pivot_type.value
        rsi_period_val = self.rsi_period.value
        rsi_buy_val = self.rsi_buy_threshold.value
        rsi_sell_val = self.rsi_sell_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bounce từ support level (pivot + fibonacci confluence) trong uptrend
        conditions_long.append(
            (dataframe['bounce_from_support']) &
            (dataframe['support_pivot_fib']) &  # Vùng hội tụ
            (dataframe['uptrend']) &
            (dataframe[f'rsi_{rsi_period_val}'] < rsi_buy_val) &  # RSI oversold
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Pullback to support trong uptrend mạnh
        conditions_long.append(
            (dataframe['near_support_pivot']) &
            (dataframe['uptrend']) &
            (dataframe['sma_20'] > dataframe['sma_20'].shift(5)) &  # SMA20 tăng
            (dataframe['close'] > dataframe['open']) &  # Nến tăng
            (dataframe[f'rsi_{rsi_period_val}'] > 40) &  # RSI không quá thấp
            (dataframe[f'rsi_{rsi_period_val}'] < 60) &  # RSI không quá cao
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Phá vỡ kháng cự R1 sau khi test thành công
        conditions_long.append(
            (dataframe['close'] > dataframe[f'r1_{pivot_type_val}']) &
            (dataframe['close'].shift(1) < dataframe[f'r1_{pivot_type_val}'].shift(1)) &
            (dataframe['uptrend']) &
            (dataframe['close'] - dataframe['open'] > dataframe['atr'] * 0.5) &  # Nến tăng mạnh
            (dataframe['volume_ratio'] > 1.5)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bounce từ resistance level (pivot + fibonacci confluence) trong downtrend
        conditions_short.append(
            (dataframe['bounce_from_resistance']) &
            (dataframe['resistance_pivot_fib']) &  # Vùng hội tụ
            (dataframe['downtrend']) &
            (dataframe[f'rsi_{rsi_period_val}'] > rsi_sell_val) &  # RSI overbought
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Pullback to resistance trong downtrend mạnh
        conditions_short.append(
            (dataframe['near_resistance_pivot']) &
            (dataframe['downtrend']) &
            (dataframe['sma_20'] < dataframe['sma_20'].shift(5)) &  # SMA20 giảm
            (dataframe['close'] < dataframe['open']) &  # Nến giảm
            (dataframe[f'rsi_{rsi_period_val}'] < 60) &  # RSI không quá cao
            (dataframe[f'rsi_{rsi_period_val}'] > 40) &  # RSI không quá thấp
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Phá vỡ hỗ trợ S1 sau khi test thành công
        conditions_short.append(
            (dataframe['close'] < dataframe[f's1_{pivot_type_val}']) &
            (dataframe['close'].shift(1) > dataframe[f's1_{pivot_type_val}'].shift(1)) &
            (dataframe['downtrend']) &
            (dataframe['open'] - dataframe['close'] > dataframe['atr'] * 0.5) &  # Nến giảm mạnh
            (dataframe['volume_ratio'] > 1.5)
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
        pivot_type_val = self.pivot_type.value
        rsi_period_val = self.rsi_period.value
        rsi_sell_val = self.rsi_sell_threshold.value
        rsi_buy_val = self.rsi_buy_threshold.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Chạm R1/R2 Pivot point
        conditions_exit_long.append(
            (dataframe['near_resistance_pivot']) &
            (dataframe[f'rsi_{rsi_period_val}'] > rsi_sell_val)
        )
        
        # 2. Phá vỡ SMA 20 trong uptrend
        conditions_exit_long.append(
            (dataframe['close'] < dataframe['sma_20']) &
            (dataframe['close'].shift(1) > dataframe['sma_20'].shift(1)) &
            (dataframe['uptrend'])
        )
        
        # 3. RSI quá cao
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_period_val}'] > 80)
        )
        
        # 4. Bắt đầu downtrend
        conditions_exit_long.append(
            (~dataframe['uptrend']) &
            (dataframe['uptrend'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Chạm S1/S2 Pivot point
        conditions_exit_short.append(
            (dataframe['near_support_pivot']) &
            (dataframe[f'rsi_{rsi_period_val}'] < rsi_buy_val)
        )
        
        # 2. Phá vỡ SMA 20 trong downtrend
        conditions_exit_short.append(
            (dataframe['close'] > dataframe['sma_20']) &
            (dataframe['close'].shift(1) < dataframe['sma_20'].shift(1)) &
            (dataframe['downtrend'])
        )
        
        # 3. RSI quá thấp
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_period_val}'] < 20)
        )
        
        # 4. Bắt đầu uptrend
        conditions_exit_short.append(
            (~dataframe['downtrend']) &
            (dataframe['downtrend'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên khoảng cách từ Pivot Points
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số từ config
        pivot_type_val = self.pivot_type.value
        rsi_period_val = self.rsi_period.value
        
        # Long: Đánh giá khoảng cách đến kháng cự gần nhất
        if side == 'long':
            # Kiểm tra hội tụ support
            support_confluence = last_candle['support_pivot_fib']
            
            # Kiểm tra khoảng cách đến kháng cự (r1)
            r1 = last_candle[f'r1_{pivot_type_val}']
            distance_to_r1 = (r1 - current_rate) / current_rate * 100
            
            # Kiểm tra RSI
            rsi = last_candle[f'rsi_{rsi_period_val}']
            
            # Tính toán đòn bẩy
            if support_confluence and distance_to_r1 > 2.0 and rsi < 40:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif support_confluence and distance_to_r1 > 1.5:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif distance_to_r1 > 1.0:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        # Short: Đánh giá khoảng cách đến hỗ trợ gần nhất
        else:
            # Kiểm tra hội tụ resistance
            resistance_confluence = last_candle['resistance_pivot_fib']
            
            # Kiểm tra khoảng cách đến hỗ trợ (s1)
            s1 = last_candle[f's1_{pivot_type_val}']
            distance_to_s1 = (current_rate - s1) / current_rate * 100
            
            # Kiểm tra RSI
            rsi = last_candle[f'rsi_{rsi_period_val}']
            
            # Tính toán đòn bẩy
            if resistance_confluence and distance_to_s1 > 2.0 and rsi > 60:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif resistance_confluence and distance_to_s1 > 1.5:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif distance_to_s1 > 1.0:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên pivot levels
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số từ config
        pivot_type_val = self.pivot_type.value
        
        # Nếu lợi nhuận dương, sử dụng pivot làm stoploss
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại pivot
                pivot = last_candle[f'pivot_{pivot_type_val}']
                sl_distance = (pivot / current_rate) - 1
                
                # Nếu giá đã vượt xa pivot, sử dụng s1 thay thế
                if current_rate > last_candle[f'r1_{pivot_type_val}']:
                    pivot = last_candle[f'pivot_{pivot_type_val}']
                
                return max(sl_distance, -0.05)  # Không ít hơn -5%
            
            else:
                # Short trade: stoploss tại pivot
                pivot = last_candle[f'pivot_{pivot_type_val}']
                sl_distance = 1 - (pivot / current_rate)
                
                # Nếu giá đã vượt xa pivot, sử dụng r1 thay thế
                if current_rate < last_candle[f's1_{pivot_type_val}']:
                    pivot = last_candle[f'pivot_{pivot_type_val}']
                
                return max(sl_distance, -0.05)  # Không ít hơn -5%
        
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
        
        # Không vào lệnh khi volume quá thấp
        if last_candle['volume_ratio'] < 0.7:
            return False
        
        # Kiểm tra thêm cho long
        if side == 'long':
            # Không vào lệnh long nếu đang có pivot resistance gần
            if last_candle['near_resistance_pivot']:
                return False
            
            # Không vào lệnh nếu RSI quá cao
            if last_candle[f'rsi_{self.rsi_period.value}'] > 75:
                return False
        
        # Kiểm tra thêm cho short
        else:
            # Không vào lệnh short nếu đang có pivot support gần
            if last_candle['near_support_pivot']:
                return False
            
            # Không vào lệnh nếu RSI quá thấp
            if last_candle[f'rsi_{self.rsi_period.value}'] < 25:
                return False
        
        return True
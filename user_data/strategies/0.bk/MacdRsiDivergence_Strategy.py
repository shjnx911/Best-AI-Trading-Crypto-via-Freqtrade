"""
MACD RSI Divergence Strategy
Chiến lược kết hợp phát hiện phân kỳ (divergence) MACD và RSI
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
        self.optimize = True

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

def detect_divergence(dataframe: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Phát hiện phân kỳ (divergence) giữa giá và các chỉ báo
    
    - Bullish divergence: Giá tạo đáy thấp hơn nhưng chỉ báo tạo đáy cao hơn
    - Bearish divergence: Giá tạo đỉnh cao hơn nhưng chỉ báo tạo đỉnh thấp hơn
    """
    df = dataframe.copy()
    
    # Tìm swing high và swing low cho giá
    df['price_swing_high'] = False
    df['price_swing_low'] = False
    
    for i in range(window, len(df) - window):
        # Swing High: điểm cao nhất trong cửa sổ
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            df['price_swing_high'].iloc[i] = True
        
        # Swing Low: điểm thấp nhất trong cửa sổ
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            df['price_swing_low'].iloc[i] = True
    
    # Tìm phân kỳ cho RSI
    for rsi_period in [6, 14]:
        rsi_col = f'rsi_{rsi_period}'
        if rsi_col in df.columns:
            # Tìm swing high và swing low cho RSI
            df[f'{rsi_col}_swing_high'] = False
            df[f'{rsi_col}_swing_low'] = False
            
            for i in range(window, len(df) - window):
                # RSI Swing High
                if all(df[rsi_col].iloc[i] > df[rsi_col].iloc[i-j] for j in range(1, window+1)) and \
                   all(df[rsi_col].iloc[i] > df[rsi_col].iloc[i+j] for j in range(1, window+1)):
                    df[f'{rsi_col}_swing_high'].iloc[i] = True
                
                # RSI Swing Low
                if all(df[rsi_col].iloc[i] < df[rsi_col].iloc[i-j] for j in range(1, window+1)) and \
                   all(df[rsi_col].iloc[i] < df[rsi_col].iloc[i+j] for j in range(1, window+1)):
                    df[f'{rsi_col}_swing_low'].iloc[i] = True
            
            # Tìm phân kỳ
            df[f'{rsi_col}_bull_div'] = False
            df[f'{rsi_col}_bear_div'] = False
            
            # Lấy tất cả các vị trí của swing points
            price_swing_high_indices = df.index[df['price_swing_high']].tolist()
            price_swing_low_indices = df.index[df['price_swing_low']].tolist()
            rsi_swing_high_indices = df.index[df[f'{rsi_col}_swing_high']].tolist()
            rsi_swing_low_indices = df.index[df[f'{rsi_col}_swing_low']].tolist()
            
            # Phát hiện Bullish Divergence (giá xuống, RSI lên)
            for i in range(1, len(price_swing_low_indices)):
                curr_price_idx = price_swing_low_indices[i]
                prev_price_idx = price_swing_low_indices[i-1]
                
                # Chỉ xét phân kỳ trong khoảng 10 nến
                if curr_price_idx - prev_price_idx > 10:
                    continue
                
                # Tìm rsi swing lows trong khoảng này
                rsi_swing_lows_in_range = [idx for idx in rsi_swing_low_indices 
                                           if prev_price_idx <= idx <= curr_price_idx]
                
                if len(rsi_swing_lows_in_range) >= 2:
                    rsi_curr_idx = rsi_swing_lows_in_range[-1]
                    rsi_prev_idx = rsi_swing_lows_in_range[-2]
                    
                    # Bullish divergence: giá thấp hơn nhưng RSI cao hơn
                    if (df['low'].iloc[curr_price_idx] < df['low'].iloc[prev_price_idx]) and \
                       (df[rsi_col].iloc[rsi_curr_idx] > df[rsi_col].iloc[rsi_prev_idx]):
                        df[f'{rsi_col}_bull_div'].iloc[curr_price_idx] = True
            
            # Phát hiện Bearish Divergence (giá lên, RSI xuống)
            for i in range(1, len(price_swing_high_indices)):
                curr_price_idx = price_swing_high_indices[i]
                prev_price_idx = price_swing_high_indices[i-1]
                
                # Chỉ xét phân kỳ trong khoảng 10 nến
                if curr_price_idx - prev_price_idx > 10:
                    continue
                
                # Tìm rsi swing highs trong khoảng này
                rsi_swing_highs_in_range = [idx for idx in rsi_swing_high_indices 
                                            if prev_price_idx <= idx <= curr_price_idx]
                
                if len(rsi_swing_highs_in_range) >= 2:
                    rsi_curr_idx = rsi_swing_highs_in_range[-1]
                    rsi_prev_idx = rsi_swing_highs_in_range[-2]
                    
                    # Bearish divergence: giá cao hơn nhưng RSI thấp hơn
                    if (df['high'].iloc[curr_price_idx] > df['high'].iloc[prev_price_idx]) and \
                       (df[rsi_col].iloc[rsi_curr_idx] < df[rsi_col].iloc[rsi_prev_idx]):
                        df[f'{rsi_col}_bear_div'].iloc[curr_price_idx] = True
    
    # Tìm phân kỳ cho MACD
    for fast_period, slow_period, signal_period in [(12, 26, 9), (8, 21, 5)]:
        macd_col = f'macd_{fast_period}_{slow_period}_{signal_period}'
        if macd_col in df.columns:
            # Tìm swing high và swing low cho MACD
            df[f'{macd_col}_swing_high'] = False
            df[f'{macd_col}_swing_low'] = False
            
            for i in range(window, len(df) - window):
                # MACD Swing High
                if all(df[macd_col].iloc[i] > df[macd_col].iloc[i-j] for j in range(1, window+1)) and \
                   all(df[macd_col].iloc[i] > df[macd_col].iloc[i+j] for j in range(1, window+1)):
                    df[f'{macd_col}_swing_high'].iloc[i] = True
                
                # MACD Swing Low
                if all(df[macd_col].iloc[i] < df[macd_col].iloc[i-j] for j in range(1, window+1)) and \
                   all(df[macd_col].iloc[i] < df[macd_col].iloc[i+j] for j in range(1, window+1)):
                    df[f'{macd_col}_swing_low'].iloc[i] = True
            
            # Tìm phân kỳ
            df[f'{macd_col}_bull_div'] = False
            df[f'{macd_col}_bear_div'] = False
            
            # Lấy tất cả các vị trí của swing points
            macd_swing_high_indices = df.index[df[f'{macd_col}_swing_high']].tolist()
            macd_swing_low_indices = df.index[df[f'{macd_col}_swing_low']].tolist()
            
            # Phát hiện Bullish Divergence (giá xuống, MACD lên)
            for i in range(1, len(price_swing_low_indices)):
                curr_price_idx = price_swing_low_indices[i]
                prev_price_idx = price_swing_low_indices[i-1]
                
                # Chỉ xét phân kỳ trong khoảng 10 nến
                if curr_price_idx - prev_price_idx > 10:
                    continue
                
                # Tìm macd swing lows trong khoảng này
                macd_swing_lows_in_range = [idx for idx in macd_swing_low_indices 
                                            if prev_price_idx <= idx <= curr_price_idx]
                
                if len(macd_swing_lows_in_range) >= 2:
                    macd_curr_idx = macd_swing_lows_in_range[-1]
                    macd_prev_idx = macd_swing_lows_in_range[-2]
                    
                    # Bullish divergence: giá thấp hơn nhưng MACD cao hơn
                    if (df['low'].iloc[curr_price_idx] < df['low'].iloc[prev_price_idx]) and \
                       (df[macd_col].iloc[macd_curr_idx] > df[macd_col].iloc[macd_prev_idx]):
                        df[f'{macd_col}_bull_div'].iloc[curr_price_idx] = True
            
            # Phát hiện Bearish Divergence (giá lên, MACD xuống)
            for i in range(1, len(price_swing_high_indices)):
                curr_price_idx = price_swing_high_indices[i]
                prev_price_idx = price_swing_high_indices[i-1]
                
                # Chỉ xét phân kỳ trong khoảng 10 nến
                if curr_price_idx - prev_price_idx > 10:
                    continue
                
                # Tìm macd swing highs trong khoảng này
                macd_swing_highs_in_range = [idx for idx in macd_swing_high_indices 
                                             if prev_price_idx <= idx <= curr_price_idx]
                
                if len(macd_swing_highs_in_range) >= 2:
                    macd_curr_idx = macd_swing_highs_in_range[-1]
                    macd_prev_idx = macd_swing_highs_in_range[-2]
                    
                    # Bearish divergence: giá cao hơn nhưng MACD thấp hơn
                    if (df['high'].iloc[curr_price_idx] > df['high'].iloc[prev_price_idx]) and \
                       (df[macd_col].iloc[macd_curr_idx] < df[macd_col].iloc[macd_prev_idx]):
                        df[f'{macd_col}_bear_div'].iloc[curr_price_idx] = True
    
    return df

class MacdRsiDivergence_Strategy(IStrategy):
    """
    Chiến lược kết hợp phát hiện phân kỳ (divergence) MACD và RSI
    
    - Phát hiện phân kỳ dương (bullish divergence) khi thị trường oversold
    - Phát hiện phân kỳ âm (bearish divergence) khi thị trường overbought
    - Kết hợp với xác nhận xu hướng từ EMA và khối lượng
    - Tối ưu cho giao dịch tầm trung (1h)
    """
    
    # Tham số cho MACD
    macd_fast = IntParameter(8, 16, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(16, 32, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(5, 12, default=9, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(6, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho swing window (phát hiện divergence)
    swing_window = IntParameter(3, 10, default=5, space="buy", optimize=True)
    
    # Tham số cho EMA (xác nhận xu hướng)
    ema_short = IntParameter(5, 30, default=20, space="buy", optimize=True)
    ema_long = IntParameter(20, 60, default=50, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.04
    
    # Time-in-force và ROI
    timeframe = '1h'
    
    minimal_roi = {
        "0": 0.05,     # 5%
        "24": 0.035,   # 3.5% sau 24 giờ
        "48": 0.025,   # 2.5% sau 48 giờ
        "72": 0.015    # 1.5% sau 72 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.015  # 1.5%
    trailing_stop_positive_offset = 0.03  # 3%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # MACD
        for fast_period in range(8, 17, 4):  # [8, 12, 16]
            for slow_period in range(16, 33, 8):  # [16, 24, 32]
                for signal_period in range(5, 13, 4):  # [5, 9, 13]
                    macd = ta.trend.MACD(
                        close=dataframe['close'],
                        window_fast=fast_period,
                        window_slow=slow_period,
                        window_sign=signal_period
                    )
                    dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'] = macd.macd()
                    dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}'] = macd.macd_signal()
                    dataframe[f'macdhist_{fast_period}_{slow_period}_{signal_period}'] = macd.macd_diff()
        
        # RSI
        for period in range(6, 22, 4):  # [6, 10, 14, 18]
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(close=dataframe['close'], window=period)
        
        # EMA
        for length in range(5, 61, 5):  # [5, 10, 15, ..., 60]
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(close=dataframe['close'], window=length)
        
        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            window=14,
            smooth_window=3
        )
        dataframe['stoch_k'] = stoch.stoch()
        dataframe['stoch_d'] = stoch.stoch_signal()
        
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
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Detect divergence
        div_df = detect_divergence(dataframe, window=self.swing_window.value)
        
        # Thêm các cột divergence vào dataframe
        for col in div_df.columns:
            if '_bull_div' in col or '_bear_div' in col or 'swing_' in col:
                dataframe[col] = div_df[col]
        
        # Tham số từ config
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        rsi_period = self.rsi_period.value
        ema_short = self.ema_short.value
        ema_long = self.ema_long.value
        
        # Xác định xu hướng từ EMA
        dataframe['uptrend'] = dataframe[f'ema_{ema_short}'] > dataframe[f'ema_{ema_long}']
        dataframe['downtrend'] = dataframe[f'ema_{ema_short}'] < dataframe[f'ema_{ema_long}']
        
        # Xác định xác nhận từ MACD
        dataframe['macd_cross_up'] = (
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] > 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1) <= 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1))
        )
        
        dataframe['macd_cross_down'] = (
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] < 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1) >= 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1))
        )
        
        # Forward fill các tín hiệu phân kỳ
        macd_col = f'macd_{macd_fast}_{macd_slow}_{macd_signal}'
        rsi_col = f'rsi_{rsi_period}'
        
        # Tạo các cột signaling để forward fill 3 nến
        dataframe['macd_bull_div_signal'] = False
        dataframe['rsi_bull_div_signal'] = False
        dataframe['macd_bear_div_signal'] = False
        dataframe['rsi_bear_div_signal'] = False
        
        # Forward fill các tín hiệu phân kỳ 3 nến
        for i in range(len(dataframe)):
            if i > 0 and i < len(dataframe) - 3:
                if dataframe[f'{macd_col}_bull_div'].iloc[i]:
                    dataframe['macd_bull_div_signal'].iloc[i:i+4] = True
                    
                if dataframe[f'{rsi_col}_bull_div'].iloc[i]:
                    dataframe['rsi_bull_div_signal'].iloc[i:i+4] = True
                    
                if dataframe[f'{macd_col}_bear_div'].iloc[i]:
                    dataframe['macd_bear_div_signal'].iloc[i:i+4] = True
                    
                if dataframe[f'{rsi_col}_bear_div'].iloc[i]:
                    dataframe['rsi_bear_div_signal'].iloc[i:i+4] = True
        
        # Thêm tín hiệu kết hợp
        dataframe['div_buy_signal'] = (
            (dataframe['macd_bull_div_signal'] | dataframe['rsi_bull_div_signal']) &
            (dataframe[f'rsi_{rsi_period}'] < 40)
        )
        
        dataframe['div_sell_signal'] = (
            (dataframe['macd_bear_div_signal'] | dataframe['rsi_bear_div_signal']) &
            (dataframe[f'rsi_{rsi_period}'] > 60)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số
        rsi_period = self.rsi_period.value
        rsi_buy = self.rsi_buy_threshold.value
        rsi_sell = self.rsi_sell_threshold.value
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bullish divergence với RSI thấp
        conditions_long.append(
            (dataframe['div_buy_signal']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy + 5) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI quá thấp + MACD cross up
        conditions_long.append(
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy) &
            (dataframe['macd_cross_up']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Giá gần BB lower + RSI thấp + không downtrend mạnh
        conditions_long.append(
            (dataframe['close'] < dataframe['bb_lowerband'] * 1.01) &
            (dataframe[f'rsi_{rsi_period}'] < 40) &
            (~dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Stoch oversold + MACD bullish
        conditions_long.append(
            (dataframe['stoch_k'] < 20) &
            (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) &
            (dataframe[f'macdhist_{macd_fast}_{macd_slow}_{macd_signal}'] > 
             dataframe[f'macdhist_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bearish divergence với RSI cao
        conditions_short.append(
            (dataframe['div_sell_signal']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell - 5) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI quá cao + MACD cross down
        conditions_short.append(
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell) &
            (dataframe['macd_cross_down']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Giá gần BB upper + RSI cao + không uptrend mạnh
        conditions_short.append(
            (dataframe['close'] > dataframe['bb_upperband'] * 0.99) &
            (dataframe[f'rsi_{rsi_period}'] > 60) &
            (~dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Stoch overbought + MACD bearish
        conditions_short.append(
            (dataframe['stoch_k'] > 80) &
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) &
            (dataframe[f'macdhist_{macd_fast}_{macd_slow}_{macd_signal}'] < 
             dataframe[f'macdhist_{macd_fast}_{macd_slow}_{macd_signal}'].shift(1)) &
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
        
        # Tham số
        rsi_period = self.rsi_period.value
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. RSI overbought
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_period}'] > 75) &
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1))
        )
        
        # 2. MACD cross down từ vùng cao
        conditions_exit_long.append(
            (dataframe['macd_cross_down']) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] > 0)
        )
        
        # 3. Bearish divergence
        conditions_exit_long.append(
            dataframe['div_sell_signal']
        )
        
        # 4. Stoch cross down từ vùng overbought
        conditions_exit_long.append(
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) < dataframe['stoch_d'].shift(1)) &
            (dataframe['stoch_k'].shift(1) > 80)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. RSI oversold
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_period}'] < 25) &
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1))
        )
        
        # 2. MACD cross up từ vùng thấp
        conditions_exit_short.append(
            (dataframe['macd_cross_up']) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] < 0)
        )
        
        # 3. Bullish divergence
        conditions_exit_short.append(
            dataframe['div_buy_signal']
        )
        
        # 4. Stoch cross up từ vùng oversold
        conditions_exit_short.append(
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) > dataframe['stoch_d'].shift(1)) &
            (dataframe['stoch_k'].shift(1) < 20)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ rõ ràng của tín hiệu phân kỳ và vị trí RSI/Stoch
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số
        rsi_period = self.rsi_period.value
        
        if side == 'long':
            # Phân kỳ rõ ràng và RSI rất thấp
            if (last_candle['div_buy_signal'] or last_candle['macd_bull_div_signal']) and last_candle[f'rsi_{rsi_period}'] < 25:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Phân kỳ rõ ràng hoặc RSI rất thấp + stoch oversold
            elif (last_candle['div_buy_signal'] or last_candle[f'rsi_{rsi_period}'] < 25) and last_candle['stoch_k'] < 20:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Chỉ có phân kỳ hoặc chỉ có RSI/Stoch thấp
            elif last_candle['div_buy_signal'] or last_candle[f'rsi_{rsi_period}'] < 30 or last_candle['stoch_k'] < 20:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # Phân kỳ rõ ràng và RSI rất cao
            if (last_candle['div_sell_signal'] or last_candle['macd_bear_div_signal']) and last_candle[f'rsi_{rsi_period}'] > 75:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Phân kỳ rõ ràng hoặc RSI rất cao + stoch overbought
            elif (last_candle['div_sell_signal'] or last_candle[f'rsi_{rsi_period}'] > 75) and last_candle['stoch_k'] > 80:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Chỉ có phân kỳ hoặc chỉ có RSI/Stoch cao
            elif last_candle['div_sell_signal'] or last_candle[f'rsi_{rsi_period}'] > 70 or last_candle['stoch_k'] > 80:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Bollinger Bands
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận dương, sử dụng BB middle làm stoploss
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại BB middle
                bb_middle = last_candle['bb_middleband']
                sl_distance = (bb_middle / current_rate) - 1
                return max(sl_distance, -0.04)  # Không quá -4%
            else:
                # Short trade: stoploss tại BB middle
                bb_middle = last_candle['bb_middleband']
                sl_distance = 1 - (bb_middle / current_rate)
                return max(sl_distance, -0.04)  # Không quá -4%
        
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
        if last_candle['bb_width'] > 0.1:  # BB width quá rộng, biến động cao
            return False
        
        # Tham số
        rsi_period = self.rsi_period.value
        
        # Không vào lệnh khi RSI extreme
        rsi = last_candle[f'rsi_{rsi_period}']
        if side == 'long' and rsi > 60:  # không vào long khi rsi quá cao
            return False
        if side == 'short' and rsi < 40:  # không vào short khi rsi quá thấp
            return False
        
        return True
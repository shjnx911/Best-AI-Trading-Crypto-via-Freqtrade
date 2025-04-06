"""
Momentum Pinbar Strategy
Chiến lược kết hợp Momentum và mô hình nến Pinbar
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

logger = logging.getLogger(__name__)

def detect_pinbar(dataframe, body_percentage=0.3, wick_percentage=2.0, window=5):
    """
    Phát hiện mô hình nến Pinbar
    
    - Bullish Pinbar: Có bóng dưới dài, thân ngắn, xuất hiện trong downtrend
    - Bearish Pinbar: Có bóng trên dài, thân ngắn, xuất hiện trong uptrend
    
    Tham số:
    - body_percentage: Phần trăm của thân nến (càng nhỏ càng tốt cho pinbar)
    - wick_percentage: Phần trăm của bóng nến so với thân (càng lớn càng tốt)
    - window: Số nến để xác định xu hướng
    """
    # Tính toán các thành phần của nến
    df = dataframe.copy()
    
    # Xác định chiều của nến (xanh/đỏ)
    df['candle_color'] = np.where(df['close'] >= df['open'], 'green', 'red')
    
    # Tính toán thân nến
    df['body_size'] = abs(df['close'] - df['open'])
    
    # Tính toán bóng trên và bóng dưới
    df['upper_wick'] = df.apply(lambda x: x['high'] - max(x['open'], x['close']), axis=1)
    df['lower_wick'] = df.apply(lambda x: min(x['open'], x['close']) - x['low'], axis=1)
    
    # Tính toán tỷ lệ giữa bóng và thân
    # Tránh chia cho 0 cho các doji
    df['body_size_adj'] = df['body_size'].apply(lambda x: max(x, 0.0000001))
    df['upper_wick_ratio'] = df['upper_wick'] / df['body_size_adj']
    df['lower_wick_ratio'] = df['lower_wick'] / df['body_size_adj']
    
    # Tính tổng kích thước của nến
    df['candle_size'] = df['high'] - df['low']
    df['body_percentage_of_candle'] = df['body_size'] / df['candle_size']
    
    # Phát hiện Bullish Pinbar
    df['bullish_pinbar'] = (
        (df['body_percentage_of_candle'] < body_percentage) &  # Thân nến nhỏ
        (df['lower_wick_ratio'] > wick_percentage) &  # Bóng dưới dài
        (df['lower_wick'] > df['upper_wick'] * 2) &  # Bóng dưới dài hơn bóng trên
        (df['candle_size'] > df['candle_size'].rolling(window=window).mean() * 0.8)  # Nến đủ lớn
    )
    
    # Phát hiện Bearish Pinbar
    df['bearish_pinbar'] = (
        (df['body_percentage_of_candle'] < body_percentage) &  # Thân nến nhỏ
        (df['upper_wick_ratio'] > wick_percentage) &  # Bóng trên dài
        (df['upper_wick'] > df['lower_wick'] * 2) &  # Bóng trên dài hơn bóng dưới
        (df['candle_size'] > df['candle_size'].rolling(window=window).mean() * 0.8)  # Nến đủ lớn
    )
    
    # Xác định xu hướng trước pinbar để xác định có phải là pinbar đảo chiều không
    df['ma_fast'] = ta.trend.sma_indicator(close=df['close'], window=5)
    df['ma_slow'] = ta.trend.sma_indicator(close=df['close'], window=10)
    
    df['uptrend'] = df['ma_fast'] > df['ma_slow']
    df['downtrend'] = df['ma_fast'] < df['ma_slow']
    
    # Bullish Pinbar có ý nghĩa khi xuất hiện trong downtrend
    df['bullish_reversal_pinbar'] = df['bullish_pinbar'] & df['downtrend']
    
    # Bearish Pinbar có ý nghĩa khi xuất hiện trong uptrend
    df['bearish_reversal_pinbar'] = df['bearish_pinbar'] & df['uptrend']
    
    return df

def detect_momentum(dataframe, n_fast=12, n_slow=26, ma_type='ema'):
    """
    Phát hiện Momentum từ các chỉ báo Momentum như MACD, RSI, Stochastic
    
    Tham số:
    - n_fast: Chu kỳ nhanh cho MACD
    - n_slow: Chu kỳ chậm cho MACD
    - ma_type: Loại MA sử dụng ('ema' hoặc 'sma')
    """
    df = dataframe.copy()
    
    # MACD
    if ma_type == 'ema':
        # Calculate EMA
        ema_fast = ta.trend.ema_indicator(close=df['close'], window=n_fast)
        ema_slow = ta.trend.ema_indicator(close=df['close'], window=n_slow)
        macd_line = ema_fast - ema_slow
        signal_line = ta.trend.ema_indicator(close=macd_line, window=9)
    else:
        # Calculate SMA
        sma_fast = ta.trend.sma_indicator(close=df['close'], window=n_fast)
        sma_slow = ta.trend.sma_indicator(close=df['close'], window=n_slow)
        macd_line = sma_fast - sma_slow
        signal_line = ta.trend.sma_indicator(close=macd_line, window=9)
    
    # Lưu kết quả vào dataframe
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_line - signal_line
    
    # Xác định các điểm giao cắt của MACD
    df['macd_cross_up'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['macd_cross_down'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    
    # RSI
    df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Xác định Stochastic cross
    df['stoch_cross_up'] = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
    df['stoch_cross_down'] = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))
    
    # Kết hợp Momentum Indicators
    # Bullish Momentum: MACD cross up + RSI tăng + Stochastic overbought
    df['bullish_momentum'] = (
        (df['macd_cross_up'] | (df['macd_hist'] > 0) & (df['macd_hist'] > df['macd_hist'].shift(1))) &
        (df['rsi'] > df['rsi'].shift(1)) &
        ((df['stoch_k'] > df['stoch_d']) | df['stoch_cross_up'])
    )
    
    # Bearish Momentum: MACD cross down + RSI giảm + Stochastic oversold
    df['bearish_momentum'] = (
        (df['macd_cross_down'] | (df['macd_hist'] < 0) & (df['macd_hist'] < df['macd_hist'].shift(1))) &
        (df['rsi'] < df['rsi'].shift(1)) &
        ((df['stoch_k'] < df['stoch_d']) | df['stoch_cross_down'])
    )
    
    return df

def detect_supports_resistances(dataframe, window=20, tolerance=0.02):
    """
    Phát hiện các vùng hỗ trợ/kháng cự từ các swing high/low
    
    Tham số:
    - window: Số nến để xác định swing point
    - tolerance: Độ chênh lệch cho phép để gộp các mức gần nhau
    """
    df = dataframe.copy()
    
    # Phát hiện swing high và swing low
    df['swing_high'] = 0.0
    df['swing_low'] = 0.0
    
    # Xác định các swing high
    for i in range(window, len(df) - window):
        # Swing High
        if all(df['high'].iloc[i] >= df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] >= df['high'].iloc[i+j] for j in range(1, window+1)):
            df['swing_high'].iloc[i] = df['high'].iloc[i]
    
    # Xác định các swing low
    for i in range(window, len(df) - window):
        # Swing Low
        if all(df['low'].iloc[i] <= df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] <= df['low'].iloc[i+j] for j in range(1, window+1)):
            df['swing_low'].iloc[i] = df['low'].iloc[i]
    
    # Lọc ra các swing high/low có giá trị != 0
    swing_highs = df[df['swing_high'] > 0]['swing_high'].values
    swing_lows = df[df['swing_low'] > 0]['swing_low'].values
    
    # Gộp các mức gần nhau
    consolidated_highs = []
    for h in swing_highs:
        if not consolidated_highs or all(abs(h - ch) / ch > tolerance for ch in consolidated_highs):
            consolidated_highs.append(h)
    
    consolidated_lows = []
    for l in swing_lows:
        if not consolidated_lows or all(abs(l - cl) / cl > tolerance for cl in consolidated_lows):
            consolidated_lows.append(l)
    
    # Xác định mức hỗ trợ và kháng cự gần nhất
    df['nearest_resistance'] = 0.0
    df['nearest_support'] = 0.0
    df['distance_to_resistance'] = 0.0
    df['distance_to_support'] = 0.0
    
    for i in range(len(df)):
        # Tìm mức kháng cự gần nhất ở phía trên
        resistances_above = [r for r in consolidated_highs if r > df['close'].iloc[i]]
        if resistances_above:
            nearest_res = min(resistances_above)
            df['nearest_resistance'].iloc[i] = nearest_res
            df['distance_to_resistance'].iloc[i] = (nearest_res - df['close'].iloc[i]) / df['close'].iloc[i] * 100  # %
        
        # Tìm mức hỗ trợ gần nhất ở phía dưới
        supports_below = [s for s in consolidated_lows if s < df['close'].iloc[i]]
        if supports_below:
            nearest_sup = max(supports_below)
            df['nearest_support'].iloc[i] = nearest_sup
            df['distance_to_support'].iloc[i] = (df['close'].iloc[i] - nearest_sup) / df['close'].iloc[i] * 100  # %
    
    return df

class MomentumPinbar_Strategy(IStrategy):
    """
    Chiến lược kết hợp Momentum và mô hình nến Pinbar
    
    - Sử dụng Pinbar để xác định điểm đảo chiều tiềm năng
    - Kết hợp với xác nhận từ các chỉ báo Momentum như MACD, RSI, Stochastic
    - Áp dụng phân tích vùng hỗ trợ/kháng cự để lọc tín hiệu
    - Tối ưu cho khung thời gian 4h
    """
    
    # Tham số cho Pinbar
    pinbar_body_pct = DecimalParameter(0.1, 0.5, default=0.3, space="buy", optimize=True)
    pinbar_wick_pct = DecimalParameter(1.0, 3.0, default=2.0, space="buy", optimize=True)
    
    # Tham số cho Momentum
    macd_fast = IntParameter(8, 20, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(20, 40, default=26, space="buy", optimize=True)
    
    # Tham số cho Support/Resistance
    sr_window = IntParameter(10, 30, default=20, space="buy", optimize=True)
    sr_tolerance = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    
    # Tham số cho filter
    min_dist_to_sr_pct = DecimalParameter(0.5, 3.0, default=1.0, space="buy", optimize=True)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.05
    
    # Time-in-force và ROI
    timeframe = '4h'
    
    minimal_roi = {
        "0": 0.08,     # 8%
        "24": 0.05,    # 5% sau 24 giờ
        "48": 0.03,    # 3% sau 48 giờ
        "72": 0.02     # 2% sau 72 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.04  # 4%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Phát hiện Pinbar
        pinbar_df = detect_pinbar(
            dataframe,
            body_percentage=self.pinbar_body_pct.value,
            wick_percentage=self.pinbar_wick_pct.value
        )
        
        # Thêm các cột Pinbar
        for column in pinbar_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = pinbar_df[column]
        
        # Phát hiện Momentum
        momentum_df = detect_momentum(
            dataframe,
            n_fast=self.macd_fast.value,
            n_slow=self.macd_slow.value
        )
        
        # Thêm các cột Momentum
        for column in momentum_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = momentum_df[column]
        
        # Phát hiện các vùng hỗ trợ/kháng cự
        sr_df = detect_supports_resistances(
            dataframe,
            window=self.sr_window.value,
            tolerance=self.sr_tolerance.value
        )
        
        # Thêm các cột Support/Resistance
        for column in sr_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = sr_df[column]
        
        # EMAs để xác định xu hướng
        for length in [20, 50, 100, 200]:
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'],
                window=length
            )
        
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
        
        # Dynamic Trend Detection
        dataframe['uptrend_strong'] = (
            (dataframe['ema_20'] > dataframe['ema_50']) &
            (dataframe['ema_50'] > dataframe['ema_100']) &
            (dataframe['close'] > dataframe['ema_20'])
        )
        
        dataframe['downtrend_strong'] = (
            (dataframe['ema_20'] < dataframe['ema_50']) &
            (dataframe['ema_50'] < dataframe['ema_100']) &
            (dataframe['close'] < dataframe['ema_20'])
        )
        
        # Phát hiện Pinbar tại vùng SR
        # Bullish Pinbar tại vùng hỗ trợ
        dataframe['bullish_pinbar_at_support'] = (
            dataframe['bullish_pinbar'] &
            (dataframe['nearest_support'] > 0) &
            (dataframe['distance_to_support'] < self.min_dist_to_sr_pct.value)
        )
        
        # Bearish Pinbar tại vùng kháng cự
        dataframe['bearish_pinbar_at_resistance'] = (
            dataframe['bearish_pinbar'] &
            (dataframe['nearest_resistance'] > 0) &
            (dataframe['distance_to_resistance'] < self.min_dist_to_sr_pct.value)
        )
        
        # Momentum + Pinbar Setup
        # Bullish: Pinbar + Momentum
        dataframe['bullish_pinbar_momentum'] = (
            (dataframe['bullish_pinbar'] | dataframe['bullish_reversal_pinbar']) &
            (dataframe['bullish_momentum']) &
            (dataframe['rsi'] < 60)  # RSI không quá cao
        )
        
        # Bearish: Pinbar + Momentum
        dataframe['bearish_pinbar_momentum'] = (
            (dataframe['bearish_pinbar'] | dataframe['bearish_reversal_pinbar']) &
            (dataframe['bearish_momentum']) &
            (dataframe['rsi'] > 40)  # RSI không quá thấp
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bullish Pinbar tại vùng hỗ trợ với Momentum xác nhận
        conditions_long.append(
            (dataframe['bullish_pinbar_at_support']) &
            (dataframe['bullish_momentum']) &
            (dataframe['rsi'] < 50) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Bullish Reversal Pinbar với Momentum mạnh
        conditions_long.append(
            (dataframe['bullish_reversal_pinbar']) &
            (dataframe['macd_cross_up']) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 3. Bullish Pinbar Momentum tại mức giá thấp
        conditions_long.append(
            (dataframe['bullish_pinbar_momentum']) &
            (dataframe['rsi'] < self.rsi_oversold.value + 5) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bullish Pinbar bouncing from EMA in uptrend
        conditions_long.append(
            (dataframe['bullish_pinbar']) &
            (dataframe['uptrend_strong']) &
            (dataframe['low'] <= dataframe['ema_50'] * 1.01) &
            (dataframe['close'] > dataframe['ema_50']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bullish Pinbar at support with oversold RSI
        conditions_long.append(
            (dataframe['bullish_pinbar']) &
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['nearest_support'] > 0) &
            (dataframe['distance_to_support'] < 2.0) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bearish Pinbar tại vùng kháng cự với Momentum xác nhận
        conditions_short.append(
            (dataframe['bearish_pinbar_at_resistance']) &
            (dataframe['bearish_momentum']) &
            (dataframe['rsi'] > 50) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Bearish Reversal Pinbar với Momentum mạnh
        conditions_short.append(
            (dataframe['bearish_reversal_pinbar']) &
            (dataframe['macd_cross_down']) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 3. Bearish Pinbar Momentum tại mức giá cao
        conditions_short.append(
            (dataframe['bearish_pinbar_momentum']) &
            (dataframe['rsi'] > self.rsi_overbought.value - 5) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bearish Pinbar bouncing from EMA in downtrend
        conditions_short.append(
            (dataframe['bearish_pinbar']) &
            (dataframe['downtrend_strong']) &
            (dataframe['high'] >= dataframe['ema_50'] * 0.99) &
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bearish Pinbar at resistance with overbought RSI
        conditions_short.append(
            (dataframe['bearish_pinbar']) &
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['nearest_resistance'] > 0) &
            (dataframe['distance_to_resistance'] < 2.0) &
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
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Bearish Pinbar
        conditions_exit_long.append(
            (dataframe['bearish_pinbar']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI Overbought
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))  # RSI đang giảm
        )
        
        # 3. MACD cross down
        conditions_exit_long.append(
            (dataframe['macd_cross_down']) &
            (dataframe['macd'] > 0)  # MACD vẫn dương
        )
        
        # 4. Approaching strong resistance
        conditions_exit_long.append(
            (dataframe['nearest_resistance'] > 0) &
            (dataframe['distance_to_resistance'] < 0.5) &  # Rất gần kháng cự
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Break down from EMA in uptrend
        conditions_exit_long.append(
            (dataframe['uptrend_strong'].shift(1)) &
            (dataframe['close'] < dataframe['ema_20']) &
            (dataframe['close'].shift(1) > dataframe['ema_20'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Bullish Pinbar
        conditions_exit_short.append(
            (dataframe['bullish_pinbar']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI Oversold
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI đang tăng
        )
        
        # 3. MACD cross up
        conditions_exit_short.append(
            (dataframe['macd_cross_up']) &
            (dataframe['macd'] < 0)  # MACD vẫn âm
        )
        
        # 4. Approaching strong support
        conditions_exit_short.append(
            (dataframe['nearest_support'] > 0) &
            (dataframe['distance_to_support'] < 0.5) &  # Rất gần hỗ trợ
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Break up from EMA in downtrend
        conditions_exit_short.append(
            (dataframe['downtrend_strong'].shift(1)) &
            (dataframe['close'] > dataframe['ema_20']) &
            (dataframe['close'].shift(1) < dataframe['ema_20'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu Pinbar và Momentum
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        if side == 'long':
            # Pinbar tại vùng hỗ trợ mạnh với RSI thấp và volume cao
            if (last_candle['bullish_pinbar_at_support'] and 
                last_candle['rsi'] < self.rsi_oversold.value and 
                last_candle['volume_ratio'] > 1.5):
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Bullish Reversal Pinbar với MACD cross up
            elif (last_candle['bullish_reversal_pinbar'] and 
                last_candle['macd_cross_up'] and 
                last_candle['volume_ratio'] > 1.2):
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Pinbar trong uptrend
            elif (last_candle['bullish_pinbar'] and 
                last_candle['uptrend_strong'] and 
                last_candle['volume_ratio'] > 1.0):
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Pinbar với momentum trung bình
            elif (last_candle['bullish_pinbar'] and 
                last_candle['bullish_momentum']):
                leverage = min(3, max_leverage)  # Đòn bẩy mặc định
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # Pinbar tại vùng kháng cự mạnh với RSI cao và volume cao
            if (last_candle['bearish_pinbar_at_resistance'] and 
                last_candle['rsi'] > self.rsi_overbought.value and 
                last_candle['volume_ratio'] > 1.5):
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Bearish Reversal Pinbar với MACD cross down
            elif (last_candle['bearish_reversal_pinbar'] and 
                last_candle['macd_cross_down'] and 
                last_candle['volume_ratio'] > 1.2):
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Pinbar trong downtrend
            elif (last_candle['bearish_pinbar'] and 
                last_candle['downtrend_strong'] and 
                last_candle['volume_ratio'] > 1.0):
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Pinbar với momentum trung bình
            elif (last_candle['bearish_pinbar'] and 
                last_candle['bearish_momentum']):
                leverage = min(3, max_leverage)  # Đòn bẩy mặc định
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên ATR và vùng hỗ trợ/kháng cự
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Dynamic stoploss based on ATR
        atr = last_candle['atr']
        
        if current_profit > 0.03:  # > 3% lợi nhuận
            if not trade.is_short:
                # Long trade
                
                # Nếu có vùng hỗ trợ gần, đặt stoploss tại vùng hỗ trợ
                if last_candle['nearest_support'] > 0:
                    support_level = last_candle['nearest_support']
                    sl_distance = (support_level / current_rate) - 1
                    return max(sl_distance, -0.1)  # Không quá 10%
                
                # Nếu không, sử dụng ATR
                atr_sl = -(atr * 2) / current_rate
                return max(atr_sl, -0.1)  # Không quá 10%
            else:
                # Short trade
                
                # Nếu có vùng kháng cự gần, đặt stoploss tại vùng kháng cự
                if last_candle['nearest_resistance'] > 0:
                    resistance_level = last_candle['nearest_resistance']
                    sl_distance = 1 - (resistance_level / current_rate)
                    return max(sl_distance, -0.1)  # Không quá 10%
                
                # Nếu không, sử dụng ATR
                atr_sl = -(atr * 2) / current_rate
                return max(atr_sl, -0.1)  # Không quá 10%
        
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
        if last_candle['volume_ratio'] < 0.7:
            return False
        
        # Kiểm tra biến động quá cao
        if last_candle['bb_width'] > 0.1:  # BB width quá rộng
            return False
        
        # Kiểm tra khoảng cách entry
        if side == 'long' and rate > last_candle['close'] * 1.01:  # Giá entry cao hơn 1%
            return False
        
        if side == 'short' and rate < last_candle['close'] * 0.99:  # Giá entry thấp hơn 1%
            return False
        
        # Kiểm tra RSI extreme
        rsi = last_candle['rsi']
        if side == 'long' and rsi > 70:  # RSI quá cao cho long
            return False
        
        if side == 'short' and rsi < 30:  # RSI quá thấp cho short
            return False
        
        return True
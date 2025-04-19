"""
Swing DCA Strategy (4h)
Chiến lược swing trading kết hợp DCA thông minh - tối ưu cho khung thời gian 4h
Tập trung vào giao dịch với mục tiêu lợi nhuận trung-dài hạn (5-10%)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import ta
from pandas import DataFrame
from functools import reduce
from typing import Dict, List, Optional, Tuple, Any, Union
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

def detect_swing_points(dataframe, window=5):
    """
    Phát hiện các điểm swing (swing high/low) trong dataframe
    
    Tham số:
    - window: Cửa sổ để xác định swing point (mặc định: 5)
    
    Trả về dataframe với các cột mới:
    - swing_high: True nếu là swing high, ngược lại False
    - swing_low: True nếu là swing low, ngược lại False
    - swing_high_price: Giá tại swing high
    - swing_low_price: Giá tại swing low
    """
    df = dataframe.copy()
    
    # Khởi tạo các cột swing
    df['swing_high'] = False
    df['swing_low'] = False
    df['swing_high_price'] = np.nan
    df['swing_low_price'] = np.nan
    
    # Xác định swing high
    for i in range(window, len(df) - window):
        # Swing High
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            df['swing_high'].iloc[i] = True
            df['swing_high_price'].iloc[i] = df['high'].iloc[i]
        
        # Swing Low
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            df['swing_low'].iloc[i] = True
            df['swing_low_price'].iloc[i] = df['low'].iloc[i]
    
    return df

def analyze_trend(dataframe, ema_short=8, ema_medium=21, ema_long=50):
    """
    Phân tích xu hướng dài hạn/trung hạn dựa trên EMA
    
    Tham số:
    - ema_short: Chu kỳ EMA ngắn
    - ema_medium: Chu kỳ EMA trung hạn
    - ema_long: Chu kỳ EMA dài hạn
    
    Trả về dataframe với các cột mới:
    - uptrend_strong: True nếu là uptrend mạnh
    - uptrend_weak: True nếu là uptrend yếu
    - downtrend_strong: True nếu là downtrend mạnh
    - downtrend_weak: True nếu là downtrend yếu
    - sideways: True nếu là sideway
    """
    df = dataframe.copy()
    
    # Tính EMA
    df[f'ema_{ema_short}'] = ta.trend.ema_indicator(close=df['close'], window=ema_short)
    df[f'ema_{ema_medium}'] = ta.trend.ema_indicator(close=df['close'], window=ema_medium)
    df[f'ema_{ema_long}'] = ta.trend.ema_indicator(close=df['close'], window=ema_long)
    
    # Các EMA dài hơn
    df['ema_100'] = ta.trend.ema_indicator(close=df['close'], window=100)
    df['ema_200'] = ta.trend.ema_indicator(close=df['close'], window=200)
    
    # Xác định xu hướng dựa trên các EMA
    # Uptrend mạnh: EMA ngắn > EMA trung > EMA dài và giá trên EMA ngắn
    df['uptrend_strong'] = (
        (df[f'ema_{ema_short}'] > df[f'ema_{ema_medium}']) &
        (df[f'ema_{ema_medium}'] > df[f'ema_{ema_long}']) &
        (df['close'] > df[f'ema_{ema_short}']) &
        (df['ema_100'] > df['ema_200'])
    )
    
    # Uptrend yếu: EMA ngắn > EMA trung nhưng không thỏa mãn điều kiện uptrend mạnh
    df['uptrend_weak'] = (
        (df[f'ema_{ema_short}'] > df[f'ema_{ema_medium}']) &
        (~df['uptrend_strong'])
    )
    
    # Downtrend mạnh: EMA ngắn < EMA trung < EMA dài và giá dưới EMA ngắn
    df['downtrend_strong'] = (
        (df[f'ema_{ema_short}'] < df[f'ema_{ema_medium}']) &
        (df[f'ema_{ema_medium}'] < df[f'ema_{ema_long}']) &
        (df['close'] < df[f'ema_{ema_short}']) &
        (df['ema_100'] < df['ema_200'])
    )
    
    # Downtrend yếu: EMA ngắn < EMA trung nhưng không thỏa mãn điều kiện downtrend mạnh
    df['downtrend_weak'] = (
        (df[f'ema_{ema_short}'] < df[f'ema_{ema_medium}']) &
        (~df['downtrend_strong'])
    )
    
    # Sideway: các EMA gần nhau, không có xu hướng rõ ràng
    df['sideways'] = (
        (~df['uptrend_strong']) &
        (~df['downtrend_strong']) &
        (abs(df[f'ema_{ema_short}'] - df[f'ema_{ema_medium}']) / df['close'] < 0.01) &
        (abs(df[f'ema_{ema_medium}'] - df[f'ema_{ema_long}']) / df['close'] < 0.02)
    )
    
    # Tính ADX để xác định độ mạnh của xu hướng
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # Xác định market condition tổng hợp
    conditions = [
        df['uptrend_strong'] & (df['adx'] > 25),
        df['uptrend_weak'],
        df['downtrend_strong'] & (df['adx'] > 25),
        df['downtrend_weak'],
        df['sideways']
    ]
    choices = ['strong_uptrend', 'weak_uptrend', 'strong_downtrend', 'weak_downtrend', 'sideways']
    df['market_condition'] = np.select(conditions, choices, default='uncertain')
    
    # DCA suitability
    df['dca_suitable'] = (
        (df['market_condition'].isin(['weak_uptrend', 'weak_downtrend', 'sideways', 'strong_downtrend'])) &
        (df['adx'] < 50)  # Tránh xu hướng quá mạnh
    )
    
    return df

def detect_support_resistance(dataframe, window=20, swing_window=5, atr_multiplier=1.5):
    """
    Phát hiện các vùng hỗ trợ/kháng cự dựa trên swing points
    
    Tham số:
    - window: Cửa sổ để gom nhóm các mức hỗ trợ/kháng cự gần nhau
    - swing_window: Cửa sổ để xác định swing point
    - atr_multiplier: Hệ số nhân ATR để xác định vùng hỗ trợ/kháng cự
    
    Trả về dataframe với các cột mới:
    - support_zone_high: Giá cao nhất của vùng hỗ trợ
    - support_zone_low: Giá thấp nhất của vùng hỗ trợ
    - resistance_zone_high: Giá cao nhất của vùng kháng cự
    - resistance_zone_low: Giá thấp nhất của vùng kháng cự
    """
    df = dataframe.copy()
    
    # Tính ATR
    df['atr'] = ta.volatility.average_true_range(high=df['high'], low=df['low'], close=df['close'], window=14)
    
    # Phát hiện swing points
    swing_df = detect_swing_points(df, window=swing_window)
    
    # Thêm các cột swing
    for column in swing_df.columns:
        if column not in df.columns:
            df[column] = swing_df[column]
    
    # Khởi tạo các cột support/resistance
    df['support_zone_high'] = np.nan
    df['support_zone_low'] = np.nan
    df['resistance_zone_high'] = np.nan
    df['resistance_zone_low'] = np.nan
    
    # Tìm các swing low/high trong cửa sổ window nến gần nhất
    for i in range(window, len(df)):
        # Current price
        current_price = df['close'].iloc[i]
        recent_atr = df['atr'].iloc[i]
        
        # Lấy các swing low/high trong window nến gần nhất
        window_df = df.iloc[i-window:i]
        swing_lows = window_df[window_df['swing_low']]['low'].values
        swing_highs = window_df[window_df['swing_high']]['high'].values
        
        # Gom nhóm các mức hỗ trợ gần nhau trong khoảng ATR
        if len(swing_lows) > 0:
            # Chỉ quan tâm đến các mức thấp hơn giá hiện tại
            valid_lows = [low for low in swing_lows if low < current_price]
            
            if len(valid_lows) > 0:
                # Tìm vùng hỗ trợ gần nhất
                support_level = max(valid_lows)  # Mức hỗ trợ cao nhất trong số các mức thấp hơn giá hiện tại
                
                # Tạo vùng hỗ trợ quanh mức này
                support_zone_high = support_level + (recent_atr * 0.3)  # +30% ATR
                support_zone_low = support_level - (recent_atr * 0.7)   # -70% ATR
                
                # Lưu vùng hỗ trợ
                df['support_zone_high'].iloc[i] = support_zone_high
                df['support_zone_low'].iloc[i] = support_zone_low
        
        # Gom nhóm các mức kháng cự gần nhau trong khoảng ATR
        if len(swing_highs) > 0:
            # Chỉ quan tâm đến các mức cao hơn giá hiện tại
            valid_highs = [high for high in swing_highs if high > current_price]
            
            if len(valid_highs) > 0:
                # Tìm vùng kháng cự gần nhất
                resistance_level = min(valid_highs)  # Mức kháng cự thấp nhất trong số các mức cao hơn giá hiện tại
                
                # Tạo vùng kháng cự quanh mức này
                resistance_zone_high = resistance_level + (recent_atr * 0.7)  # +70% ATR
                resistance_zone_low = resistance_level - (recent_atr * 0.3)   # -30% ATR
                
                # Lưu vùng kháng cự
                df['resistance_zone_high'].iloc[i] = resistance_zone_high
                df['resistance_zone_low'].iloc[i] = resistance_zone_low
    
    # Tính DCA levels dựa trên vùng hỗ trợ
    df['dca_level_1_pct'] = -2.0  # Default -2% for first DCA
    df['dca_level_2_pct'] = -4.0  # Default -4% for second DCA
    df['dca_level_3_pct'] = -7.0  # Default -7% for third DCA
    
    # Điều chỉnh mức DCA theo support zones
    for i in range(1, len(df)):
        current_price = df['close'].iloc[i]
        
        # Sử dụng vùng hỗ trợ làm mức DCA nếu có
        if not pd.isna(df['support_zone_high'].iloc[i]):
            support_zone_high = df['support_zone_high'].iloc[i]
            support_zone_low = df['support_zone_low'].iloc[i]
            
            # DCA Level 1: Sử dụng vùng phía trên support zone
            if support_zone_high < current_price:
                dca_level_1 = support_zone_high
                dca_level_1_pct = (dca_level_1 / current_price - 1) * 100
                df['dca_level_1_pct'].iloc[i] = max(dca_level_1_pct, -3.0)  # Không thấp hơn -3%
            
            # DCA Level 2: Sử dụng middle of support zone
            middle_support = (support_zone_high + support_zone_low) / 2
            if middle_support < current_price:
                dca_level_2 = middle_support
                dca_level_2_pct = (dca_level_2 / current_price - 1) * 100
                df['dca_level_2_pct'].iloc[i] = max(dca_level_2_pct, -5.0)  # Không thấp hơn -5%
            
            # DCA Level 3: Sử dụng bottom of support zone
            if support_zone_low < current_price:
                dca_level_3 = support_zone_low
                dca_level_3_pct = (dca_level_3 / current_price - 1) * 100
                df['dca_level_3_pct'].iloc[i] = max(dca_level_3_pct, -8.0)  # Không thấp hơn -8%
    
    # Điều chỉnh DCA levels dựa trên market condition
    for i in range(1, len(df)):
        market_condition = df['market_condition'].iloc[i]
        
        if market_condition == 'strong_downtrend':
            # Trong downtrend mạnh, DCA levels nên thấp hơn để bắt knife tốt hơn
            df['dca_level_1_pct'].iloc[i] = min(df['dca_level_1_pct'].iloc[i] * 1.3, -2.5)  # Ít nhất -2.5%
            df['dca_level_2_pct'].iloc[i] = min(df['dca_level_2_pct'].iloc[i] * 1.2, -5.0)  # Ít nhất -5%
            df['dca_level_3_pct'].iloc[i] = min(df['dca_level_3_pct'].iloc[i] * 1.1, -8.0)  # Ít nhất -8%
            
        elif market_condition == 'sideways':
            # Trong sideway, DCA levels có thể cao hơn do price action thường trong range
            df['dca_level_1_pct'].iloc[i] = max(df['dca_level_1_pct'].iloc[i], -1.5)  # Không thấp hơn -1.5%
            df['dca_level_2_pct'].iloc[i] = max(df['dca_level_2_pct'].iloc[i], -3.0)  # Không thấp hơn -3%
            df['dca_level_3_pct'].iloc[i] = max(df['dca_level_3_pct'].iloc[i], -5.0)  # Không thấp hơn -5%
    
    # Tính DCA allocation dựa trên market condition
    df['dca_initial_allocation'] = 0.4  # Default 40% cho entry ban đầu
    
    # Điều chỉnh allocation dựa trên market condition
    df.loc[df['market_condition'] == 'strong_uptrend', 'dca_initial_allocation'] = 0.6  # 60% cho entry ban đầu
    df.loc[df['market_condition'] == 'weak_uptrend', 'dca_initial_allocation'] = 0.5  # 50% cho entry ban đầu
    df.loc[df['market_condition'] == 'sideways', 'dca_initial_allocation'] = 0.4  # 40% cho entry ban đầu
    df.loc[df['market_condition'] == 'weak_downtrend', 'dca_initial_allocation'] = 0.3  # 30% cho entry ban đầu
    df.loc[df['market_condition'] == 'strong_downtrend', 'dca_initial_allocation'] = 0.2  # 20% cho entry ban đầu
    
    # Tính toán allocation cho các lần DCA
    for i in range(1, len(df)):
        initial_alloc = df['dca_initial_allocation'].iloc[i]
        remaining_alloc = 1 - initial_alloc
        
        if df['market_condition'].iloc[i] in ['strong_downtrend', 'weak_downtrend']:
            # Trong downtrend, phân bổ nhiều hơn cho DCA levels sau
            df.loc[i, 'dca_level_1_allocation'] = remaining_alloc * 0.4  # 40% của remaining
            df.loc[i, 'dca_level_2_allocation'] = remaining_alloc * 0.3  # 30% của remaining
            df.loc[i, 'dca_level_3_allocation'] = remaining_alloc * 0.3  # 30% của remaining
            
        elif df['market_condition'].iloc[i] == 'sideways':
            # Trong sideway, phân bổ đều cho các DCA levels
            df.loc[i, 'dca_level_1_allocation'] = remaining_alloc / 3  # 33% của remaining
            df.loc[i, 'dca_level_2_allocation'] = remaining_alloc / 3  # 33% của remaining
            df.loc[i, 'dca_level_3_allocation'] = remaining_alloc / 3  # 33% của remaining
            
        else:  # Uptrend/uncertain
            # Trong uptrend, phân bổ nhiều hơn cho DCA levels đầu
            df.loc[i, 'dca_level_1_allocation'] = remaining_alloc * 0.5  # 50% của remaining
            df.loc[i, 'dca_level_2_allocation'] = remaining_alloc * 0.3  # 30% của remaining
            df.loc[i, 'dca_level_3_allocation'] = remaining_alloc * 0.2  # 20% của remaining
    
    return df

class Swing_DCA_Strategy_4h(IStrategy):
    """
    Chiến lược Swing Trading kết hợp DCA thông minh
    
    - Tối ưu cho khung thời gian 4h
    - Tập trung vào giao dịch trung-dài hạn với mục tiêu lợi nhuận 5-10%
    - Sử dụng DCA thông minh dựa trên vùng hỗ trợ và cấu trúc thị trường
    - Đặc biệt hiệu quả trong thị trường downtrend và sideway
    """
    
    # Tham số cho chiến lược DCA
    use_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    max_dca_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    max_dca_orders = IntParameter(1, 3, default=3, space="buy", optimize=True)
    
    # Tham số cho trend analysis
    ema_short = IntParameter(5, 20, default=8, space="buy", optimize=True)
    ema_medium = IntParameter(15, 50, default=21, space="buy", optimize=True)
    ema_long = IntParameter(30, 100, default=50, space="buy", optimize=True)
    
    # Tham số cho support/resistance
    sr_window = IntParameter(10, 40, default=20, space="buy", optimize=True)
    swing_window = IntParameter(3, 10, default=5, space="buy", optimize=True)
    
    # Tham số RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số profit-taking
    profit_target_1 = DecimalParameter(2.0, 5.0, default=3.0, space="sell", optimize=True)
    profit_target_2 = DecimalParameter(4.0, 8.0, default=5.0, space="sell", optimize=True)
    profit_target_3 = DecimalParameter(7.0, 15.0, default=10.0, space="sell", optimize=True)
    
    # Tham số risk management
    initial_stoploss = DecimalParameter(3.0, 8.0, default=5.0, space="sell", optimize=True)
    stoploss_after_dca_1 = DecimalParameter(5.0, 12.0, default=8.0, space="sell", optimize=True)
    stoploss_after_dca_2 = DecimalParameter(8.0, 15.0, default=12.0, space="sell", optimize=True)
    stoploss_after_dca_3 = DecimalParameter(10.0, 20.0, default=15.0, space="sell", optimize=True)
    
    # Core parameters
    timeframe = '4h'
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.05  # -5% stoploss mặc định, sẽ được điều chỉnh theo DCA
    
    # Time-in-force và ROI
    minimal_roi = {
        "0": 0.10,     # 10%
        "24": 0.07,    # 7% sau 24 giờ (6 candles)
        "48": 0.05,    # 5% sau 48 giờ (12 candles)
        "72": 0.03     # 3% sau 72 giờ (18 candles)
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.04  # 4%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200  # Cần nhiều dữ liệu lịch sử để phân tích swing
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Phân tích xu hướng
        trend_df = analyze_trend(
            dataframe,
            ema_short=self.ema_short.value,
            ema_medium=self.ema_medium.value,
            ema_long=self.ema_long.value
        )
        
        # Thêm các cột trend
        for column in trend_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = trend_df[column]
        
        # Phát hiện support/resistance
        sr_df = detect_support_resistance(
            dataframe,
            window=self.sr_window.value,
            swing_window=self.swing_window.value
        )
        
        # Thêm các cột support/resistance
        for column in sr_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = sr_df[column]
        
        # Thêm các chỉ báo phổ biến nếu chưa có
        if 'rsi' not in dataframe.columns:
            dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=self.rsi_period.value)
        
        # MACD
        macd = ta.trend.MACD(
            close=dataframe['close'],
            window_slow=26,
            window_fast=12,
            window_sign=9
        )
        dataframe['macd'] = macd.macd()
        dataframe['macdsignal'] = macd.macd_signal()
        dataframe['macdhist'] = macd.macd_diff()
        
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
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        
        # Volume indicators
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean_20']
        
        # Money Flow Index
        dataframe['mfi'] = ta.volume.money_flow_index(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            volume=dataframe['volume'],
            window=14
        )
        
        # Phát hiện các xác nhận entry
        # Swing low setup - mua tại các điểm swing low
        dataframe['swing_low_setup'] = (
            dataframe['swing_low'] &
            (dataframe['rsi'] < 40) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Support bounce setup - mua khi giá tiếp cận vùng hỗ trợ
        dataframe['support_bounce_setup'] = False
        
        for i in range(1, len(dataframe)):
            if not pd.isna(dataframe['support_zone_high'].iloc[i]) and not pd.isna(dataframe['support_zone_low'].iloc[i]):
                # Price approaching support zone
                if (dataframe['low'].iloc[i] <= dataframe['support_zone_high'].iloc[i] and
                    dataframe['low'].iloc[i] >= dataframe['support_zone_low'].iloc[i]):
                    
                    # Confirmation: RSI low and bullish candle
                    if (dataframe['rsi'].iloc[i] < 45 and dataframe['close'].iloc[i] > dataframe['open'].iloc[i]):
                        dataframe['support_bounce_setup'].iloc[i] = True
        
        # Mua tại pullback trong uptrend
        dataframe['pullback_setup'] = (
            (dataframe['market_condition'].isin(['strong_uptrend', 'weak_uptrend'])) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_long.value}']) &
            (dataframe['rsi'] < 45) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Mua sau khi breakout khỏi resistance
        dataframe['breakout_setup'] = False
        
        for i in range(5, len(dataframe)):
            if not pd.isna(dataframe['resistance_zone_high'].iloc[i-1]):
                # Price breaking above resistance zone
                if (dataframe['close'].iloc[i-1] < dataframe['resistance_zone_high'].iloc[i-1] and
                    dataframe['close'].iloc[i] > dataframe['resistance_zone_high'].iloc[i-1]):
                    
                    # Confirmation: Volume surge and bullish candle
                    if (dataframe['volume_ratio'].iloc[i] > 1.2 and dataframe['close'].iloc[i] > dataframe['open'].iloc[i]):
                        dataframe['breakout_setup'].iloc[i] = True
        
        # Xác định điểm stop loss tối ưu
        dataframe['optimal_stoploss_pct'] = np.nan
        
        for i in range(1, len(dataframe)):
            current_price = dataframe['close'].iloc[i]
            
            # Default stoploss based on ATR
            atr_stoploss = dataframe['atr'].iloc[i] * 3 / current_price * 100  # 3x ATR as percentage
            optimal_sl = max(atr_stoploss, self.initial_stoploss.value)  # At least initial_stoploss
            
            # If there's a swing low below current price, use it as stoploss
            recent_swing_lows = []
            for j in range(max(0, i-20), i):
                if dataframe['swing_low'].iloc[j]:
                    recent_swing_lows.append(dataframe['low'].iloc[j])
            
            if recent_swing_lows:
                # Find closest swing low below current price
                valid_lows = [low for low in recent_swing_lows if low < current_price]
                if valid_lows:
                    closest_swing_low = max(valid_lows)
                    swing_sl_pct = (1 - closest_swing_low / current_price) * 100  # Convert to percentage
                    
                    # Use swing low if it's not too far
                    if swing_sl_pct < 15:  # Not more than 15% away
                        optimal_sl = swing_sl_pct
            
            # Use support zone if available and closer
            if not pd.isna(dataframe['support_zone_low'].iloc[i]) and dataframe['support_zone_low'].iloc[i] < current_price:
                support_sl_pct = (1 - dataframe['support_zone_low'].iloc[i] / current_price) * 100
                
                # Use support zone if it's not too far
                if support_sl_pct < 12:  # Not more than 12% away
                    optimal_sl = min(optimal_sl, support_sl_pct)  # Use the closer one
            
            dataframe['optimal_stoploss_pct'].iloc[i] = optimal_sl
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Swing Low Setup
        conditions_long.append(
            (dataframe['swing_low_setup']) &
            (dataframe['market_condition'].isin(['weak_uptrend', 'sideways', 'weak_downtrend'])) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Support Bounce Setup
        conditions_long.append(
            (dataframe['support_bounce_setup']) &
            (dataframe['market_condition'].isin(['strong_uptrend', 'weak_uptrend', 'sideways'])) &
            (dataframe['rsi'] < 45) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Pullback in Uptrend
        conditions_long.append(
            (dataframe['pullback_setup']) &
            (dataframe['close'] > dataframe['open']) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Breakout Setup with Confirmation
        conditions_long.append(
            (dataframe['breakout_setup']) &
            (dataframe['market_condition'].isin(['strong_uptrend', 'weak_uptrend'])) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 5. Oversold Bounce in Favorable Trend
        conditions_long.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI turning up
            (dataframe['close'] > dataframe['open']) &  # Bullish candle
            (dataframe['market_condition'].isin(['strong_uptrend', 'weak_uptrend', 'sideways'])) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT (hạn chế trong 4h swing strategy)
        conditions_short = []
        
        # 1. Swing High với Trend Confirmation
        conditions_short.append(
            (dataframe['swing_high']) &
            (dataframe['market_condition'] == 'strong_downtrend') &
            (dataframe['rsi'] > 60) &
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Resistance Rejection
        conditions_short.append(
            (~pd.isna(dataframe['resistance_zone_high'])) &
            (dataframe['high'] >= dataframe['resistance_zone_low']) &
            (dataframe['high'] <= dataframe['resistance_zone_high']) &
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['market_condition'].isin(['strong_downtrend', 'weak_downtrend'])) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. RSI Overbought in Downtrend
        conditions_short.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI turning down
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['market_condition'].isin(['strong_downtrend', 'weak_downtrend'])) &
            (dataframe['volume_ratio'] > 0.8)
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
        
        # 1. Đạt vùng kháng cự
        conditions_exit_long.append(
            (~pd.isna(dataframe['resistance_zone_high'])) &
            (dataframe['high'] >= dataframe['resistance_zone_low']) &
            (dataframe['high'] <= dataframe['resistance_zone_high']) &
            (dataframe['close'] < dataframe['open']) &  # Bearish rejection
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI Overbought + Bearish Divergence
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Higher close
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # Lower RSI (divergence)
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Strong Trend Reversal
        conditions_exit_long.append(
            (dataframe['market_condition'].shift(1).isin(['strong_uptrend', 'weak_uptrend'])) &
            (dataframe['market_condition'].isin(['strong_downtrend', 'weak_downtrend'])) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. Swing High Formed
        conditions_exit_long.append(
            (dataframe['swing_high']) &
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 5. Break below EMA after extended uptrend
        conditions_exit_long.append(
            (dataframe['market_condition'].shift(1) == 'strong_uptrend') &
            (dataframe['close'] < dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema_medium.value}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Đạt vùng hỗ trợ
        conditions_exit_short.append(
            (~pd.isna(dataframe['support_zone_high'])) &
            (dataframe['low'] <= dataframe['support_zone_high']) &
            (dataframe['low'] >= dataframe['support_zone_low']) &
            (dataframe['close'] > dataframe['open']) &  # Bullish rejection
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI Oversold + Bullish Divergence
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Lower close
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # Higher RSI (divergence)
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Strong Trend Reversal
        conditions_exit_short.append(
            (dataframe['market_condition'].shift(1).isin(['strong_downtrend', 'weak_downtrend'])) &
            (dataframe['market_condition'].isin(['strong_uptrend', 'weak_uptrend'])) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. Swing Low Formed
        conditions_exit_short.append(
            (dataframe['swing_low']) &
            (dataframe['close'] > dataframe['open']) &  # Bullish candle
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 5. Break above EMA after extended downtrend
        conditions_exit_short.append(
            (dataframe['market_condition'].shift(1) == 'strong_downtrend') &
            (dataframe['close'] > dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe['close'].shift(1) < dataframe[f'ema_{self.ema_medium.value}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def adjust_trade_position(self, trade: 'Trade', current_time: datetime, 
                           current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        """
        Điều chỉnh vị thế - thực hiện chiến lược DCA
        Đây là hàm đặc biệt trong FreqTrade để thêm vào vị thế hiện tại
        """
        # Safety check - Chỉ áp dụng DCA nếu tham số use_dca = True
        if not self.use_dca.value:
            return None
        
        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if len(dataframe) == 0:
            return None
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Safety checks
        if trade.pair not in dataframe['pair'].values:
            return None
        
        # Get DCA information
        dca_info = {}
        
        # Tính số lần đã DCA
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Đã đạt số lần DCA tối đa
        if dca_count >= self.max_dca_orders.value:
            return None
        
        # Kiểm tra xem thị trường có phù hợp cho DCA không
        if not last_candle['dca_suitable']:
            return None
        
        # Calculate units to buy for DCA
        stake_amount = trade.stake_amount
        dca_stake = 0.0
        
        current_dca_level = dca_count + 1
        
        # Xác định mức giá và stake amount cho DCA tiếp theo
        # DCA level 1
        if current_dca_level == 1:
            dca_threshold = last_candle['dca_level_1_pct']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
            
            # Phân bổ vốn dựa trên tính toán trong indicator
            if 'dca_level_1_allocation' in last_candle:
                dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle['dca_level_1_allocation']) / float(last_candle['dca_initial_allocation'])
            else:
                # Default nếu không có thông tin allocation
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.5
        
        # DCA level 2
        elif current_dca_level == 2:
            dca_threshold = last_candle['dca_level_2_pct']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
            
            # Phân bổ vốn dựa trên tính toán trong indicator
            if 'dca_level_2_allocation' in last_candle:
                dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle['dca_level_2_allocation']) / float(last_candle['dca_initial_allocation'])
            else:
                # Default nếu không có thông tin allocation
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.3
        
        # DCA level 3
        elif current_dca_level == 3:
            dca_threshold = last_candle['dca_level_3_pct']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
            
            # Phân bổ vốn dựa trên tính toán trong indicator
            if 'dca_level_3_allocation' in last_candle:
                dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle['dca_level_3_allocation']) / float(last_candle['dca_initial_allocation'])
            else:
                # Default nếu không có thông tin allocation
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.2
        
        # Kiểm tra trend direction
        if not trade.is_short and last_candle['market_condition'] == 'strong_downtrend':
            # In strong downtrend, increase DCA stake for long positions
            dca_stake *= 1.2
        
        if trade.is_short and last_candle['market_condition'] == 'strong_uptrend':
            # In strong uptrend, increase DCA stake for short positions
            dca_stake *= 1.2
        
        # Kiểm tra thêm confirmation signals
        # For long positions
        if not trade.is_short:
            # Check if price is near support
            has_support_signal = False
            
            # Near support zone
            if not pd.isna(last_candle['support_zone_high']) and not pd.isna(last_candle['support_zone_low']):
                if (current_rate <= last_candle['support_zone_high'] and 
                    current_rate >= last_candle['support_zone_low']):
                    has_support_signal = True
            
            # Oversold with bouncing RSI
            if last_candle['rsi'] < 40 and last_candle['rsi'] > last_candle['rsi'].shift(1):
                has_support_signal = True
            
            # Adjust DCA stake based on confirmation
            if has_support_signal:
                dca_stake *= 1.2  # Increase stake with confirmation
            else:
                dca_stake *= 0.8  # Decrease stake without confirmation
        
        # For short positions
        else:
            # Check if price is near resistance
            has_resistance_signal = False
            
            # Near resistance zone
            if not pd.isna(last_candle['resistance_zone_high']) and not pd.isna(last_candle['resistance_zone_low']):
                if (current_rate >= last_candle['resistance_zone_low'] and 
                    current_rate <= last_candle['resistance_zone_high']):
                    has_resistance_signal = True
            
            # Overbought with falling RSI
            if last_candle['rsi'] > 60 and last_candle['rsi'] < last_candle['rsi'].shift(1):
                has_resistance_signal = True
            
            # Adjust DCA stake based on confirmation
            if has_resistance_signal:
                dca_stake *= 1.2  # Increase stake with confirmation
            else:
                dca_stake *= 0.8  # Decrease stake without confirmation
        
        # Thêm giới hạn cuối cùng cho safety
        dca_stake = min(dca_stake, stake_amount * self.max_dca_multiplier.value * 1.5)
        
        # Trả về số lượng vốn cho DCA
        return dca_stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên market condition và setup quality
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Base leverage on market condition
        market_condition = last_candle['market_condition']
        
        if side == 'long':
            # Long position leverage
            if market_condition == 'strong_uptrend':
                # Higher leverage in strong uptrend
                leverage = min(5, max_leverage)
            elif market_condition == 'weak_uptrend':
                leverage = min(4, max_leverage)
            elif market_condition == 'sideways':
                leverage = min(3, max_leverage)
            elif market_condition == 'weak_downtrend':
                leverage = min(2, max_leverage)
            elif market_condition == 'strong_downtrend':
                # Lower leverage in strong downtrend
                leverage = min(2, max_leverage)
            else:
                leverage = self.leverage_default
            
            # Adjust based on setup quality
            if last_candle['swing_low_setup']:
                leverage = min(leverage + 1, max_leverage)
            if last_candle['support_bounce_setup']:
                leverage = min(leverage + 1, max_leverage)
            if last_candle['breakout_setup']:
                leverage = min(leverage + 1, max_leverage)
            if last_candle['rsi'] < self.rsi_oversold.value:
                leverage = min(leverage + 1, max_leverage)
            
        else:  # short
            # Short position leverage - more conservative
            if market_condition == 'strong_downtrend':
                leverage = min(4, max_leverage)
            elif market_condition == 'weak_downtrend':
                leverage = min(3, max_leverage)
            elif market_condition == 'sideways':
                leverage = min(2, max_leverage)
            elif market_condition in ['weak_uptrend', 'strong_uptrend']:
                leverage = min(2, max_leverage)
            else:
                leverage = self.leverage_default
            
            # Adjust based on setup quality
            if last_candle['swing_high']:
                leverage = min(leverage + 1, max_leverage)
            if not pd.isna(last_candle['resistance_zone_high']):
                if (current_rate >= last_candle['resistance_zone_low'] and 
                    current_rate <= last_candle['resistance_zone_high']):
                    leverage = min(leverage + 1, max_leverage)
            if last_candle['rsi'] > self.rsi_overbought.value:
                leverage = min(leverage + 1, max_leverage)
        
        # Limit leverage by volatility (using BB width as proxy)
        if last_candle['bb_width'] > 0.1:  # High volatility
            leverage = min(leverage, 3)  # Cap at 3x in high volatility
        
        # Ensure minimum leverage
        leverage = max(2, leverage)  # At least 2x
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên support zones và số lần DCA
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy số lần DCA đã thực hiện
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Base stoploss on DCA count
        if dca_count == 0:
            # Use optimal_stoploss_pct if available
            if 'optimal_stoploss_pct' in last_candle and not pd.isna(last_candle['optimal_stoploss_pct']):
                dynamic_stoploss = -last_candle['optimal_stoploss_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value / 100
        elif dca_count == 1:
            dynamic_stoploss = -self.stoploss_after_dca_1.value / 100
        elif dca_count == 2:
            dynamic_stoploss = -self.stoploss_after_dca_2.value / 100
        else:
            dynamic_stoploss = -self.stoploss_after_dca_3.value / 100
        
        # Adjust stoploss when in profit
        if current_profit > 0:
            if current_profit >= self.profit_target_3.value / 100:  # > target 3
                return max(dynamic_stoploss, -0.02)  # Lock in 98% of profit
            elif current_profit >= self.profit_target_2.value / 100:  # > target 2
                return max(dynamic_stoploss, -0.03)  # Lock in 97% of profit
            elif current_profit >= self.profit_target_1.value / 100:  # > target 1
                return max(dynamic_stoploss, -0.05)  # Lock in 95% of profit
        
        # Use swing points for stoploss if available
        if not trade.is_short:  # Long position
            # Find recent swing lows for potential stoploss
            recent_swing_lows = []
            
            for i in range(max(0, len(dataframe)-20), len(dataframe)):
                if dataframe['swing_low'].iloc[i]:
                    # Only consider swing lows below current price
                    if dataframe['low'].iloc[i] < current_rate:
                        recent_swing_lows.append(dataframe['low'].iloc[i])
            
            if recent_swing_lows:
                # Use the highest swing low below current price
                closest_swing_low = max(recent_swing_lows)
                swing_sl_distance = (closest_swing_low / current_rate) - 1
                
                # Only use if not too far (avoid setting SL too far away)
                if abs(swing_sl_distance) < abs(dynamic_stoploss) and abs(swing_sl_distance) < 0.15:
                    return swing_sl_distance * 0.95  # Set SL slightly below swing low
        
        else:  # Short position
            # Find recent swing highs for potential stoploss
            recent_swing_highs = []
            
            for i in range(max(0, len(dataframe)-20), len(dataframe)):
                if dataframe['swing_high'].iloc[i]:
                    # Only consider swing highs above current price
                    if dataframe['high'].iloc[i] > current_rate:
                        recent_swing_highs.append(dataframe['high'].iloc[i])
            
            if recent_swing_highs:
                # Use the lowest swing high above current price
                closest_swing_high = min(recent_swing_highs)
                swing_sl_distance = 1 - (closest_swing_high / current_rate)
                
                # Only use if not too far (avoid setting SL too far away)
                if abs(swing_sl_distance) < abs(dynamic_stoploss) and abs(swing_sl_distance) < 0.15:
                    return swing_sl_distance * 0.95  # Set SL slightly above swing high
        
        return dynamic_stoploss
    
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
        
        # Check trade direction vs market condition
        market_condition = last_candle['market_condition']
        
        if side == 'long':
            # Không vào lệnh long trong strong downtrend trừ khi có xác nhận mạnh
            if market_condition == 'strong_downtrend':
                # Cần xác nhận mạnh để vào lệnh
                has_strong_confirmation = (
                    last_candle['swing_low_setup'] or
                    last_candle['support_bounce_setup'] or
                    last_candle['rsi'] < self.rsi_oversold.value
                )
                
                if not has_strong_confirmation:
                    return False
            
            # Kiểm tra risk:reward
            # Xác định target gần nhất
            nearest_target = None
            
            if not pd.isna(last_candle['resistance_zone_low']):
                nearest_target = last_candle['resistance_zone_low']
            
            # Xác định stoploss
            stoploss_level = rate * (1 + self.stoploss)  # Default stoploss
            
            if 'optimal_stoploss_pct' in last_candle and not pd.isna(last_candle['optimal_stoploss_pct']):
                stoploss_level = rate * (1 - last_candle['optimal_stoploss_pct']/100)
            
            # Tính RR nếu có target
            if nearest_target is not None:
                reward = nearest_target - rate
                risk = rate - stoploss_level
                
                if reward / risk < 1.5:  # RR < 1.5
                    return False
        
        else:  # short
            # Không vào lệnh short trong strong uptrend trừ khi có xác nhận mạnh
            if market_condition == 'strong_uptrend':
                # Cần xác nhận mạnh để vào lệnh
                has_strong_confirmation = (
                    last_candle['swing_high'] and
                    last_candle['rsi'] > self.rsi_overbought.value and
                    last_candle['close'] < last_candle['open']
                )
                
                if not has_strong_confirmation:
                    return False
            
            # Hạn chế lệnh short trong 4h timeframe 
            # (thường không tốt cho swing trading)
            if market_condition not in ['strong_downtrend', 'weak_downtrend']:
                return False
        
        # Kiểm tra khoảng cách entry
        if side == 'long' and rate > last_candle['close'] * 1.01:  # Giá entry cao hơn 1%
            return False
        
        if side == 'short' and rate < last_candle['close'] * 0.99:  # Giá entry thấp hơn 1%
            return False
        
        return True
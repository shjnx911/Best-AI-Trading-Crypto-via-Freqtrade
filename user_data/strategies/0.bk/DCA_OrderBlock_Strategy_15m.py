"""
DCA OrderBlock Strategy (15m)
Chiến lược DCA (Dollar Cost Averaging) thông minh kết hợp với Order Block và Fair Value Gap
Tối ưu cho khung thời gian 15m
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

def detect_fair_value_gaps(dataframe, min_gap_size=0.2, with_atr=True, atr_multiplier=0.5, window=14):
    """
    Phát hiện Fair Value Gaps (FVG) trong dataframe
    
    Fair Value Gap là khoảng trống giữa các nến không chồng chéo, thể hiện bất đối xứng về 
    cung cầu mà thị trường có thể quay lại lấp đầy
    
    Tham số:
    - min_gap_size: Kích thước tối thiểu của gap (tính theo %, mặc định 0.2%)
    - with_atr: Có tính toán dựa trên ATR hay không
    - atr_multiplier: Hệ số nhân ATR để xác định kích thước gap có ý nghĩa
    - window: Cửa sổ tính ATR
    """
    # Tạo bản sao dataframe
    df = dataframe.copy()
    
    # Tính toán ATR nếu cần
    if with_atr:
        df['atr'] = ta.volatility.average_true_range(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=window
        )
    
    # Khởi tạo cột FVG
    df['bullish_fvg'] = False
    df['bearish_fvg'] = False
    df['bullish_fvg_zone_high'] = np.nan
    df['bullish_fvg_zone_low'] = np.nan
    df['bearish_fvg_zone_high'] = np.nan
    df['bearish_fvg_zone_low'] = np.nan
    
    # Khởi tạo cột để lưu thông tin gap
    df['gap_size'] = 0.0
    
    # Phát hiện Bullish FVG (khi nến ở giữa có high thấp hơn low của nến trước đó)
    # Nến 1 - Nến 2 - Nến 3: Nếu low(Nến 1) > high(Nến 2) thì hình thành Bullish FVG
    for i in range(2, len(df)):
        # Bullish FVG: 
        # - Low của nến thứ nhất cao hơn high của nến thứ hai
        # - Giá nến thứ ba cao hơn giá nến thứ hai
        if df['low'].iloc[i-2] > df['high'].iloc[i-1] and df['close'].iloc[i] > df['close'].iloc[i-1]:
            # Tính kích thước gap
            gap_size = df['low'].iloc[i-2] - df['high'].iloc[i-1]
            gap_percent = gap_size / df['high'].iloc[i-1] * 100
            
            # Kiểm tra kích thước gap có đủ lớn không
            if with_atr:
                min_gap = df['atr'].iloc[i-1] * atr_multiplier
                is_significant = gap_size > min_gap
            else:
                is_significant = gap_percent > min_gap_size
            
            if is_significant:
                df['bullish_fvg'].iloc[i] = True
                df['bullish_fvg_zone_high'].iloc[i] = df['low'].iloc[i-2]
                df['bullish_fvg_zone_low'].iloc[i] = df['high'].iloc[i-1]
                df['gap_size'].iloc[i] = gap_size
    
    # Phát hiện Bearish FVG (khi nến ở giữa có low cao hơn high của nến sau đó)
    # Nến 1 - Nến 2 - Nến 3: Nếu high(Nến 1) < low(Nến 2) thì hình thành Bearish FVG
    for i in range(2, len(df)):
        # Bearish FVG: 
        # - High của nến thứ nhất thấp hơn low của nến thứ hai
        # - Giá nến thứ ba thấp hơn giá nến thứ hai
        if df['high'].iloc[i-2] < df['low'].iloc[i-1] and df['close'].iloc[i] < df['close'].iloc[i-1]:
            # Tính kích thước gap
            gap_size = df['low'].iloc[i-1] - df['high'].iloc[i-2]
            gap_percent = gap_size / df['high'].iloc[i-2] * 100
            
            # Kiểm tra kích thước gap có đủ lớn không
            if with_atr:
                min_gap = df['atr'].iloc[i-1] * atr_multiplier
                is_significant = gap_size > min_gap
            else:
                is_significant = gap_percent > min_gap_size
            
            if is_significant:
                df['bearish_fvg'].iloc[i] = True
                df['bearish_fvg_zone_high'].iloc[i] = df['low'].iloc[i-1]
                df['bearish_fvg_zone_low'].iloc[i] = df['high'].iloc[i-2]
                df['gap_size'].iloc[i] = gap_size
    
    # Track FVG zone cho các nến sau đó (FVG sẽ tồn tại cho đến khi được lấp đầy)
    for i in range(1, len(df)):
        # Nếu nến hiện tại không có FVG, kiểm tra nến trước đó
        if pd.isna(df['bullish_fvg_zone_high'].iloc[i]) and not pd.isna(df['bullish_fvg_zone_high'].iloc[i-1]):
            # Kiểm tra xem FVG đã bị lấp đầy chưa
            if df['low'].iloc[i] <= df['bullish_fvg_zone_high'].iloc[i-1]:
                # FVG đã bị lấp đầy, không còn hiệu lực
                pass
            else:
                # FVG vẫn còn hiệu lực, copy sang nến hiện tại
                df['bullish_fvg_zone_high'].iloc[i] = df['bullish_fvg_zone_high'].iloc[i-1]
                df['bullish_fvg_zone_low'].iloc[i] = df['bullish_fvg_zone_low'].iloc[i-1]
        
        # Tương tự cho bearish FVG
        if pd.isna(df['bearish_fvg_zone_high'].iloc[i]) and not pd.isna(df['bearish_fvg_zone_high'].iloc[i-1]):
            # Kiểm tra xem FVG đã bị lấp đầy chưa
            if df['high'].iloc[i] >= df['bearish_fvg_zone_low'].iloc[i-1]:
                # FVG đã bị lấp đầy, không còn hiệu lực
                pass
            else:
                # FVG vẫn còn hiệu lực, copy sang nến hiện tại
                df['bearish_fvg_zone_high'].iloc[i] = df['bearish_fvg_zone_high'].iloc[i-1]
                df['bearish_fvg_zone_low'].iloc[i] = df['bearish_fvg_zone_low'].iloc[i-1]
    
    # Phát hiện khi giá quay lại FVG (mitigate)
    df['bullish_fvg_mitigation'] = False
    df['bearish_fvg_mitigation'] = False
    
    for i in range(1, len(df)):
        # Bullish FVG mitigation: giá quay lại vùng FVG bullish
        if not pd.isna(df['bullish_fvg_zone_high'].iloc[i]):
            if (df['low'].iloc[i] <= df['bullish_fvg_zone_high'].iloc[i] and 
                df['low'].iloc[i] >= df['bullish_fvg_zone_low'].iloc[i]):
                df['bullish_fvg_mitigation'].iloc[i] = True
        
        # Bearish FVG mitigation: giá quay lại vùng FVG bearish
        if not pd.isna(df['bearish_fvg_zone_high'].iloc[i]):
            if (df['high'].iloc[i] >= df['bearish_fvg_zone_low'].iloc[i] and 
                df['high'].iloc[i] <= df['bearish_fvg_zone_high'].iloc[i]):
                df['bearish_fvg_mitigation'].iloc[i] = True
    
    return df

def detect_order_blocks(dataframe, strength_threshold=0.6, atr_multiplier=1.5, window=14):
    """
    Phát hiện Order Blocks trong dataframe
    
    Order Block là khu vực nơi một lượng lớn lệnh được đặt, thể hiện sự hiện diện của smart money
    và thường đóng vai trò là vùng hỗ trợ/kháng cự mạnh
    
    Tham số:
    - strength_threshold: Ngưỡng độ mạnh của nến (0-1, mặc định 0.6)
    - atr_multiplier: Hệ số nhân ATR để xác định nến đủ lớn
    - window: Cửa sổ tính ATR
    """
    # Tạo bản sao dataframe
    df = dataframe.copy()
    
    # Tính toán ATR
    df['atr'] = ta.volatility.average_true_range(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=window
    )
    
    # Tính body size và wick size
    df['body_size'] = abs(df['close'] - df['open'])
    df['candle_size'] = df['high'] - df['low']
    df['body_percentage'] = df['body_size'] / df['candle_size']
    
    # Xác định màu nến
    df['is_bullish'] = df['close'] > df['open']
    df['is_bearish'] = df['close'] < df['open']
    
    # Xác định nến có biến động lớn (impulsive move)
    df['is_large_move'] = df['candle_size'] > (df['atr'] * atr_multiplier)
    
    # Khởi tạo các cột Order Block
    df['bullish_ob'] = False
    df['bearish_ob'] = False
    df['bullish_ob_zone_high'] = np.nan
    df['bullish_ob_zone_low'] = np.nan
    df['bearish_ob_zone_high'] = np.nan
    df['bearish_ob_zone_low'] = np.nan
    
    # Phát hiện Bullish Order Block (nến giảm trước chuỗi tăng mạnh)
    for i in range(2, len(df) - 1):
        # Tìm nến giảm trước chuỗi tăng mạnh
        if (df['is_bearish'].iloc[i-1] and 
            df['is_bullish'].iloc[i] and 
            df['is_bullish'].iloc[i+1] and 
            df['is_large_move'].iloc[i]):
            
            # Xác nhận đây là bullish order block
            # - Nến i-1 là nến đỏ
            # - Nến i và i+1 là nến xanh mạnh
            # - Body của nến i phải đủ lớn (> strength_threshold)
            if df['body_percentage'].iloc[i] > strength_threshold:
                df['bullish_ob'].iloc[i-1] = True
                df['bullish_ob_zone_high'].iloc[i-1] = df['high'].iloc[i-1]
                df['bullish_ob_zone_low'].iloc[i-1] = df['low'].iloc[i-1]
    
    # Phát hiện Bearish Order Block (nến tăng trước chuỗi giảm mạnh)
    for i in range(2, len(df) - 1):
        # Tìm nến tăng trước chuỗi giảm mạnh
        if (df['is_bullish'].iloc[i-1] and 
            df['is_bearish'].iloc[i] and 
            df['is_bearish'].iloc[i+1] and 
            df['is_large_move'].iloc[i]):
            
            # Xác nhận đây là bearish order block
            # - Nến i-1 là nến xanh
            # - Nến i và i+1 là nến đỏ mạnh
            # - Body của nến i phải đủ lớn (> strength_threshold)
            if df['body_percentage'].iloc[i] > strength_threshold:
                df['bearish_ob'].iloc[i-1] = True
                df['bearish_ob_zone_high'].iloc[i-1] = df['high'].iloc[i-1]
                df['bearish_ob_zone_low'].iloc[i-1] = df['low'].iloc[i-1]
    
    # Track Order Block zone cho các nến sau đó (Order Block tồn tại trong 20 nến)
    max_ob_life = 20  # Order Block tồn tại tối đa 20 nến
    
    for i in range(1, len(df)):
        # Bullish Order Block
        if df['bullish_ob'].iloc[i-1]:
            # Cập nhật vùng Order Block cho 20 nến tiếp theo
            end_index = min(i + max_ob_life, len(df))
            for j in range(i, end_index):
                # Nếu giá không return về vùng Order Block, vùng này vẫn còn hiệu lực
                if pd.isna(df['bullish_ob_zone_high'].iloc[j]):
                    df['bullish_ob_zone_high'].iloc[j] = df['bullish_ob_zone_high'].iloc[i-1]
                    df['bullish_ob_zone_low'].iloc[j] = df['bullish_ob_zone_low'].iloc[i-1]
        
        # Bearish Order Block
        if df['bearish_ob'].iloc[i-1]:
            # Cập nhật vùng Order Block cho 20 nến tiếp theo
            end_index = min(i + max_ob_life, len(df))
            for j in range(i, end_index):
                # Nếu giá không return về vùng Order Block, vùng này vẫn còn hiệu lực
                if pd.isna(df['bearish_ob_zone_high'].iloc[j]):
                    df['bearish_ob_zone_high'].iloc[j] = df['bearish_ob_zone_high'].iloc[i-1]
                    df['bearish_ob_zone_low'].iloc[j] = df['bearish_ob_zone_low'].iloc[i-1]
    
    # Phát hiện khi giá quay lại Order Block (revisit)
    df['bullish_ob_revisit'] = False
    df['bearish_ob_revisit'] = False
    
    for i in range(1, len(df)):
        # Bullish Order Block revisit: giá quay lại vùng Order Block bullish
        if not pd.isna(df['bullish_ob_zone_high'].iloc[i]):
            if (df['low'].iloc[i] <= df['bullish_ob_zone_high'].iloc[i] and 
                df['low'].iloc[i] >= df['bullish_ob_zone_low'].iloc[i]):
                df['bullish_ob_revisit'].iloc[i] = True
        
        # Bearish Order Block revisit: giá quay lại vùng Order Block bearish
        if not pd.isna(df['bearish_ob_zone_high'].iloc[i]):
            if (df['high'].iloc[i] >= df['bearish_ob_zone_low'].iloc[i] and 
                df['high'].iloc[i] <= df['bearish_ob_zone_high'].iloc[i]):
                df['bearish_ob_revisit'].iloc[i] = True
    
    return df

class DCA_OrderBlock_Strategy_15m(IStrategy):
    """
    Chiến lược DCA kết hợp Order Block và Fair Value Gap
    
    - Áp dụng DCA thông minh dựa trên vùng Order Block và Fair Value Gap
    - Tối ưu hóa vùng giá và timing DCA dựa trên Smart Money Concept
    - Sử dụng mức giá DCA dựa trên các vùng hỗ trợ/kháng cự thực tế
    - Đặc biệt phù hợp với khung thời gian 15m
    """
    
    # Tham số cho Order Block và FVG
    ob_strength = DecimalParameter(0.4, 0.8, default=0.6, space="buy", optimize=True)
    ob_atr_multiplier = DecimalParameter(1.0, 2.0, default=1.5, space="buy", optimize=True)
    fvg_min_gap = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True)
    fvg_atr_multiplier = DecimalParameter(0.3, 1.0, default=0.5, space="buy", optimize=True)
    
    # Tham số cho chiến lược DCA
    use_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    max_dca_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    max_dca_orders = IntParameter(1, 3, default=3, space="buy", optimize=True)
    use_ob_for_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    use_fvg_for_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    
    # Tham số Price Action
    ema_short = IntParameter(5, 30, default=8, space="buy", optimize=True)
    ema_medium = IntParameter(10, 50, default=21, space="buy", optimize=True)
    
    # Tham số RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_ob_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    rsi_os_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    
    # Tham số profit-taking
    profit_target_1 = DecimalParameter(0.5, 2.0, default=1.0, space="sell", optimize=True)
    profit_target_2 = DecimalParameter(1.0, 3.0, default=2.0, space="sell", optimize=True)
    profit_target_3 = DecimalParameter(2.0, 5.0, default=3.0, space="sell", optimize=True)
    
    # Tham số risk management
    initial_stoploss = DecimalParameter(1.0, 5.0, default=3.0, space="sell", optimize=True)
    stoploss_after_dca_1 = DecimalParameter(2.0, 7.0, default=4.0, space="sell", optimize=True)
    stoploss_after_dca_2 = DecimalParameter(3.0, 10.0, default=6.0, space="sell", optimize=True)
    stoploss_after_dca_3 = DecimalParameter(4.0, 15.0, default=8.0, space="sell", optimize=True)
    
    # Core parameters
    timeframe = '15m'
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.03  # -3% stoploss mặc định, sẽ được điều chỉnh theo DCA
    
    # Time-in-force và ROI
    minimal_roi = {
        "0": 0.03,    # 3%
        "30": 0.02,   # 2% sau 30 phút
        "60": 0.01,   # 1% sau 60 phút
        "90": 0.005   # 0.5% sau 90 phút
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
        
        # Phát hiện Order Blocks
        ob_df = detect_order_blocks(
            dataframe,
            strength_threshold=self.ob_strength.value,
            atr_multiplier=self.ob_atr_multiplier.value
        )
        
        # Thêm các cột Order Blocks
        for column in ob_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = ob_df[column]
        
        # Phát hiện Fair Value Gaps
        fvg_df = detect_fair_value_gaps(
            dataframe,
            min_gap_size=self.fvg_min_gap.value,
            atr_multiplier=self.fvg_atr_multiplier.value
        )
        
        # Thêm các cột FVG
        for column in fvg_df.columns:
            if column not in dataframe.columns and column != 'atr':  # Tránh ghi đè ATR
                dataframe[column] = fvg_df[column]
        
        # EMAs
        for length in [self.ema_short.value, self.ema_medium.value, 50, 100, 200]:
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(close=dataframe['close'], window=length)
        
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=self.rsi_period.value)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=dataframe['close'], window=20, window_dev=2)
        dataframe['bb_upperband'] = bollinger.bollinger_hband()
        dataframe['bb_lowerband'] = bollinger.bollinger_lband()
        dataframe['bb_middleband'] = bollinger.bollinger_mavg()
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        
        # MACD
        macd = ta.trend.MACD(close=dataframe['close'], window_slow=26, window_fast=12, window_sign=9)
        dataframe['macd'] = macd.macd()
        dataframe['macdsignal'] = macd.macd_signal()
        dataframe['macdhist'] = macd.macd_diff()
        
        # ATR nếu chưa có
        if 'atr' not in dataframe.columns:
            dataframe['atr'] = ta.volatility.average_true_range(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=14
            )
        
        # Volume
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean_20']
        
        # Trend Detection
        dataframe['uptrend'] = (
            (dataframe[f'ema_{self.ema_short.value}'] > dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe[f'ema_{self.ema_medium.value}'] > dataframe['ema_50']) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}'])
        )
        
        dataframe['downtrend'] = (
            (dataframe[f'ema_{self.ema_short.value}'] < dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe[f'ema_{self.ema_medium.value}'] < dataframe['ema_50']) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}'])
        )
        
        # Xác định tín hiệu mua
        # Bullish Setup: Bullish OB Revisit hoặc Bullish FVG Mitigation
        dataframe['bullish_setup'] = (
            (dataframe['bullish_ob_revisit'] | dataframe['bullish_fvg_mitigation']) &
            (dataframe['rsi'] < 60) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Bearish Setup: Bearish OB Revisit hoặc Bearish FVG Mitigation
        dataframe['bearish_setup'] = (
            (dataframe['bearish_ob_revisit'] | dataframe['bearish_fvg_mitigation']) &
            (dataframe['rsi'] > 40) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Tính DCA levels và allocation - dựa trên Order Block và FVG
        
        # Initial allocation
        dataframe['dca_initial_allocation'] = 0.4  # Default 40% for initial entry
        
        # DCA setup thích hợp
        dataframe['dca_suitable'] = (
            (~dataframe['uptrend'] | dataframe['downtrend']) &
            (dataframe['rsi'] > 20) &
            (dataframe['rsi'] < 80)
        )
        
        # Tính toán mức tối ưu cho DCA dựa trên FVG và OB
        dataframe['dca_level_1_pct'] = -1.5  # Default -1.5%
        dataframe['dca_level_2_pct'] = -3.0  # Default -3.0%
        dataframe['dca_level_3_pct'] = -5.0  # Default -5.0%
        
        # Price levels for DCA - sử dụng vùng OB và FVG nếu có
        for i in range(1, len(dataframe)):
            current_price = dataframe['close'].iloc[i]
            
            # Tìm các vùng OB và FVG gần đây
            ob_levels = []
            fvg_levels = []
            
            # Collect OB levels
            if not pd.isna(dataframe['bullish_ob_zone_high'].iloc[i]):
                ob_levels.append(dataframe['bullish_ob_zone_high'].iloc[i])
                ob_levels.append(dataframe['bullish_ob_zone_low'].iloc[i])
            
            # Collect FVG levels
            if not pd.isna(dataframe['bullish_fvg_zone_high'].iloc[i]):
                fvg_levels.append(dataframe['bullish_fvg_zone_high'].iloc[i])
                fvg_levels.append(dataframe['bullish_fvg_zone_low'].iloc[i])
            
            # Sắp xếp các mức theo thứ tự giảm dần từ giá hiện tại
            all_levels = [(level, (level / current_price - 1) * 100) for level in ob_levels + fvg_levels if level < current_price]
            all_levels.sort(key=lambda x: x[1], reverse=True)  # Sort by percentage, highest first
            
            # Điều chỉnh DCA levels nếu có đủ vùng OB và FVG
            if len(all_levels) >= 3:
                dca_candidates = [level for level, pct in all_levels if pct < -0.5]  # Only consider levels at least 0.5% below current price
                
                if len(dca_candidates) >= 1:
                    level_1 = dca_candidates[0]
                    dataframe['dca_level_1_pct'].iloc[i] = (level_1 / current_price - 1) * 100
                
                if len(dca_candidates) >= 2:
                    level_2 = dca_candidates[1]
                    dataframe['dca_level_2_pct'].iloc[i] = (level_2 / current_price - 1) * 100
                
                if len(dca_candidates) >= 3:
                    level_3 = dca_candidates[2]
                    dataframe['dca_level_3_pct'].iloc[i] = (level_3 / current_price - 1) * 100
        
        # Calculate dynamic stoploss based on ATR
        dataframe['dynamic_stoploss_pct'] = dataframe['atr'] / dataframe['close'] * 100 * 3  # 3x ATR
        dataframe['dynamic_stoploss_pct'] = dataframe['dynamic_stoploss_pct'].clip(2.0, 5.0)  # Clip between 2% and 5%
        
        # Calculate potential target based on risk:reward of at least 2:1
        dataframe['potential_target_pct'] = dataframe['dynamic_stoploss_pct'] * 2
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bullish OB Revisit in non-uptrend
        conditions_long.append(
            (dataframe['bullish_ob_revisit']) &
            (~dataframe['uptrend']) &
            (dataframe['rsi'] < 50) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Bullish FVG Mitigation in non-uptrend
        conditions_long.append(
            (dataframe['bullish_fvg_mitigation']) &
            (~dataframe['uptrend']) &
            (dataframe['rsi'] < 50) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Strong buy signal - both OB & FVG
        conditions_long.append(
            (dataframe['bullish_ob_revisit']) &
            (dataframe['bullish_fvg_mitigation']) &
            (dataframe['close'] > dataframe['open']) &  # Bullish candle
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. Bullish Setup + RSI Oversold
        conditions_long.append(
            (dataframe['bullish_setup']) &
            (dataframe['rsi'] < self.rsi_os_threshold.value) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 5. Bounce from EMA in slight uptrend
        conditions_long.append(
            (dataframe['close'] > dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe['low'] <= dataframe[f'ema_{self.ema_medium.value}'] * 1.005) &  # Touched or very close to EMA
            (dataframe['close'] > dataframe['open']) &  # Bullish candle
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Higher close
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bearish OB Revisit in non-downtrend
        conditions_short.append(
            (dataframe['bearish_ob_revisit']) &
            (~dataframe['downtrend']) &
            (dataframe['rsi'] > 50) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Bearish FVG Mitigation in non-downtrend
        conditions_short.append(
            (dataframe['bearish_fvg_mitigation']) &
            (~dataframe['downtrend']) &
            (dataframe['rsi'] > 50) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. Strong sell signal - both OB & FVG
        conditions_short.append(
            (dataframe['bearish_ob_revisit']) &
            (dataframe['bearish_fvg_mitigation']) &
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 4. Bearish Setup + RSI Overbought
        conditions_short.append(
            (dataframe['bearish_setup']) &
            (dataframe['rsi'] > self.rsi_ob_threshold.value) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 5. Rejection from EMA in slight downtrend
        conditions_short.append(
            (dataframe['close'] < dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe['high'] >= dataframe[f'ema_{self.ema_medium.value}'] * 0.995) &  # Touched or very close to EMA
            (dataframe['close'] < dataframe['open']) &  # Bearish candle
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Lower close
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
        
        # 1. RSI Overbought
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_ob_threshold.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))  # RSI đang giảm
        )
        
        # 2. Bearish Order Block Created
        conditions_exit_long.append(
            dataframe['bearish_ob']
        )
        
        # 3. Bearish FVG Created
        conditions_exit_long.append(
            dataframe['bearish_fvg']
        )
        
        # 4. MACD Cross Down
        conditions_exit_long.append(
            (dataframe['macd'] < dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) >= dataframe['macdsignal'].shift(1)) &
            (dataframe['macd'] > 0)  # Still in positive MACD territory
        )
        
        # 5. Break below EMA in uptrend
        conditions_exit_long.append(
            (dataframe['uptrend'].shift(1)) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema_short.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. RSI Oversold
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_os_threshold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI đang tăng
        )
        
        # 2. Bullish Order Block Created
        conditions_exit_short.append(
            dataframe['bullish_ob']
        )
        
        # 3. Bullish FVG Created
        conditions_exit_short.append(
            dataframe['bullish_fvg']
        )
        
        # 4. MACD Cross Up
        conditions_exit_short.append(
            (dataframe['macd'] > dataframe['macdsignal']) &
            (dataframe['macd'].shift(1) <= dataframe['macdsignal'].shift(1)) &
            (dataframe['macd'] < 0)  # Still in negative MACD territory
        )
        
        # 5. Break above EMA in downtrend
        conditions_exit_short.append(
            (dataframe['downtrend'].shift(1)) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) < dataframe[f'ema_{self.ema_short.value}'].shift(1))
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
                
            # Phân bổ vốn cho DCA level 1
            if trade.is_short:
                # Short position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.3  # 30% of initial position for first DCA
            else:
                # Long position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.3  # 30% of initial position for first DCA
        
        # DCA level 2
        elif current_dca_level == 2:
            dca_threshold = last_candle['dca_level_2_pct']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
                
            # Phân bổ vốn cho DCA level 2
            if trade.is_short:
                # Short position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.2  # 20% of initial position for second DCA
            else:
                # Long position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.2  # 20% of initial position for second DCA
        
        # DCA level 3
        elif current_dca_level == 3:
            dca_threshold = last_candle['dca_level_3_pct']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
                
            # Phân bổ vốn cho DCA level 3
            if trade.is_short:
                # Short position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.1  # 10% of initial position for third DCA
            else:
                # Long position
                dca_stake = stake_amount * self.max_dca_multiplier.value * 0.1  # 10% of initial position for third DCA
        
        # Điều kiện bổ sung - kiểm tra phân kỳ RSI cho DCA level cuối
        if current_dca_level == self.max_dca_orders.value:
            # Kiểm tra xem có dấu hiệu reversal hay không
            if trade.is_short:
                # Short position - tìm dấu hiệu giá sẽ giảm
                if last_candle['rsi'] < 40 or last_candle['close'] > last_candle['close'].shift(1):
                    # RSI đã giảm nhiều hoặc giá đảo chiều tăng - không DCA nữa
                    return None
            else:
                # Long position - tìm dấu hiệu giá sẽ tăng
                if last_candle['rsi'] > 60 or last_candle['close'] < last_candle['close'].shift(1):
                    # RSI đã tăng nhiều hoặc giá đảo chiều giảm - không DCA nữa
                    return None
        
        # Điều kiện đặc biệt cho long position - kiểm tra nếu có vùng OB hoặc FVG gần đó
        if not trade.is_short:
            # Kiểm tra có vùng hỗ trợ OB/FVG gần giá hiện tại không (+/- 1%)
            has_support = False
            
            if self.use_ob_for_dca.value and not pd.isna(last_candle['bullish_ob_zone_high']):
                ob_high = last_candle['bullish_ob_zone_high']
                ob_low = last_candle['bullish_ob_zone_low']
                
                # Kiểm tra xem giá hiện tại có gần vùng OB không
                if current_rate >= ob_low * 0.99 and current_rate <= ob_high * 1.01:
                    has_support = True
            
            if self.use_fvg_for_dca.value and not pd.isna(last_candle['bullish_fvg_zone_high']):
                fvg_high = last_candle['bullish_fvg_zone_high']
                fvg_low = last_candle['bullish_fvg_zone_low']
                
                # Kiểm tra xem giá hiện tại có gần vùng FVG không
                if current_rate >= fvg_low * 0.99 and current_rate <= fvg_high * 1.01:
                    has_support = True
            
            # Nếu không có vùng hỗ trợ gần, tăng threshold DCA (phải giảm nhiều hơn mới DCA)
            if not has_support and current_dca_level < 3:
                dca_adjusted_threshold = dca_threshold * 1.2  # Thêm 20% threshold
                if current_profit >= dca_adjusted_threshold / 100:
                    return None
        
        # Trả về số lượng vốn cho DCA
        return dca_stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên tín hiệu OB/FVG và cấu trúc thị trường
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        if side == 'long':
            # Đòn bẩy cho long position
            
            # 1. Strong buy signal - both OB & FVG
            if last_candle['bullish_ob_revisit'] and last_candle['bullish_fvg_mitigation']:
                return min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # 2. Bullish OB Revisit
            elif last_candle['bullish_ob_revisit']:
                if not last_candle['uptrend'] and last_candle['rsi'] < 40:
                    return min(5, max_leverage)  # Đòn bẩy cao
                else:
                    return min(4, max_leverage)  # Đòn bẩy trung bình cao
            
            # 3. Bullish FVG Mitigation
            elif last_candle['bullish_fvg_mitigation']:
                if not last_candle['uptrend'] and last_candle['rsi'] < 40:
                    return min(5, max_leverage)  # Đòn bẩy cao
                else:
                    return min(4, max_leverage)  # Đòn bẩy trung bình cao
            
            # 4. Bounce from EMA in uptrend
            elif (last_candle['close'] > last_candle[f'ema_{self.ema_medium.value}'] and 
                  last_candle['low'] <= last_candle[f'ema_{self.ema_medium.value}'] * 1.005):
                  
                return min(4, max_leverage)  # Đòn bẩy trung bình cao
                
            # 5. RSI Oversold
            elif last_candle['rsi'] < self.rsi_os_threshold.value:
                return min(4, max_leverage)  # Đòn bẩy trung bình cao
                
            # Default
            else:
                return self.leverage_default
                
        else:  # side == 'short'
            # Đòn bẩy cho short position
            
            # 1. Strong sell signal - both OB & FVG
            if last_candle['bearish_ob_revisit'] and last_candle['bearish_fvg_mitigation']:
                return min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # 2. Bearish OB Revisit
            elif last_candle['bearish_ob_revisit']:
                if not last_candle['downtrend'] and last_candle['rsi'] > 60:
                    return min(5, max_leverage)  # Đòn bẩy cao
                else:
                    return min(4, max_leverage)  # Đòn bẩy trung bình cao
            
            # 3. Bearish FVG Mitigation
            elif last_candle['bearish_fvg_mitigation']:
                if not last_candle['downtrend'] and last_candle['rsi'] > 60:
                    return min(5, max_leverage)  # Đòn bẩy cao
                else:
                    return min(4, max_leverage)  # Đòn bẩy trung bình cao
            
            # 4. Rejection from EMA in downtrend
            elif (last_candle['close'] < last_candle[f'ema_{self.ema_medium.value}'] and 
                  last_candle['high'] >= last_candle[f'ema_{self.ema_medium.value}'] * 0.995):
                  
                return min(4, max_leverage)  # Đòn bẩy trung bình cao
                
            # 5. RSI Overbought
            elif last_candle['rsi'] > self.rsi_ob_threshold.value:
                return min(4, max_leverage)  # Đòn bẩy trung bình cao
                
            # Default
            else:
                return self.leverage_default
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên OB/FVG và số lần DCA
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy số lần DCA đã thực hiện
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Stoploss mặc định dựa trên DCA count
        if dca_count == 0:
            # Chưa DCA - sử dụng dynamic stoploss hoặc initial stoploss
            if 'dynamic_stoploss_pct' in last_candle:
                dynamic_stoploss = -last_candle['dynamic_stoploss_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value / 100
                
        elif dca_count == 1:
            # Sau DCA 1 lần
            dynamic_stoploss = -self.stoploss_after_dca_1.value / 100
                
        elif dca_count == 2:
            # Sau DCA 2 lần
            dynamic_stoploss = -self.stoploss_after_dca_2.value / 100
                
        else:
            # Sau DCA 3 lần hoặc nhiều hơn
            dynamic_stoploss = -self.stoploss_after_dca_3.value / 100
        
        # Điều chỉnh stoploss nếu trade đang có lợi nhuận
        if current_profit > 0:
            # Đang lãi - bảo vệ lợi nhuận
            if current_profit >= self.profit_target_2.value / 100:  # Đạt target 2
                return max(dynamic_stoploss, -0.01)  # Maximum -1% (protect 99% of profit)
                
            elif current_profit >= self.profit_target_1.value / 100:  # Đạt target 1
                return max(dynamic_stoploss, -0.02)  # Maximum -2%
        
        # Điều chỉnh stoploss dựa trên vùng hỗ trợ/kháng cự OB/FVG nếu có
        if not trade.is_short:
            # Long position - tìm vùng hỗ trợ OB/FVG
            support_levels = []
            
            # Add OB support levels
            if self.use_ob_for_dca.value and not pd.isna(last_candle['bullish_ob_zone_low']):
                support_levels.append(last_candle['bullish_ob_zone_low'])
            
            # Add FVG support levels
            if self.use_fvg_for_dca.value and not pd.isna(last_candle['bullish_fvg_zone_low']):
                support_levels.append(last_candle['bullish_fvg_zone_low'])
            
            # Nếu có vùng hỗ trợ dưới giá hiện tại, sử dụng vùng hỗ trợ gần giá hiện tại nhất
            if support_levels:
                valid_levels = [level for level in support_levels if level < current_rate]
                
                if valid_levels:
                    best_support = max(valid_levels)  # Vùng hỗ trợ cao nhất (gần giá hiện tại nhất)
                    sl_distance = (best_support / current_rate) - 1
                    
                    # Không đặt stoploss quá xa so với support level
                    adjusted_sl = sl_distance * 0.95  # Đặt stoploss ở 95% của support level
                    
                    # Nếu có nhiều DCA, vẫn đảm bảo stoploss không quá gần
                    if dca_count > 1:
                        return max(adjusted_sl, dynamic_stoploss)
                    else:
                        return max(adjusted_sl, -0.1)  # Không quá -10%
        
        else:  # Short position
            # Short position - tìm vùng kháng cự OB/FVG
            resistance_levels = []
            
            # Add OB resistance levels
            if self.use_ob_for_dca.value and not pd.isna(last_candle['bearish_ob_zone_high']):
                resistance_levels.append(last_candle['bearish_ob_zone_high'])
            
            # Add FVG resistance levels
            if self.use_fvg_for_dca.value and not pd.isna(last_candle['bearish_fvg_zone_high']):
                resistance_levels.append(last_candle['bearish_fvg_zone_high'])
            
            # Nếu có vùng kháng cự trên giá hiện tại, sử dụng vùng kháng cự gần giá hiện tại nhất
            if resistance_levels:
                valid_levels = [level for level in resistance_levels if level > current_rate]
                
                if valid_levels:
                    best_resistance = min(valid_levels)  # Vùng kháng cự thấp nhất (gần giá hiện tại nhất)
                    sl_distance = 1 - (best_resistance / current_rate)
                    
                    # Không đặt stoploss quá xa so với resistance level
                    adjusted_sl = sl_distance * 0.95  # Đặt stoploss ở 95% của resistance level
                    
                    # Nếu có nhiều DCA, vẫn đảm bảo stoploss không quá gần
                    if dca_count > 1:
                        return max(adjusted_sl, dynamic_stoploss)
                    else:
                        return max(adjusted_sl, -0.1)  # Không quá -10%
        
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
        
        # Kiểm tra nếu risk:reward tốt
        if 'potential_target_pct' in last_candle and 'dynamic_stoploss_pct' in last_candle:
            # Risk:reward tối thiểu 1:1.5
            min_rr_ratio = 1.5
            
            if side == 'long':
                potential_reward = last_candle['potential_target_pct']
                potential_risk = last_candle['dynamic_stoploss_pct']
                
                if potential_reward / potential_risk < min_rr_ratio:
                    return False
                    
            else:  # short
                potential_reward = last_candle['potential_target_pct']
                potential_risk = last_candle['dynamic_stoploss_pct']
                
                if potential_reward / potential_risk < min_rr_ratio:
                    return False
        
        return True
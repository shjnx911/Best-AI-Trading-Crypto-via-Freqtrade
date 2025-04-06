"""
FVG (Fair Value Gap) + Order Block Strategy
Chiến lược kết hợp phân tích Fair Value Gap và Order Block
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

class FVG_OrderBlock_Strategy(IStrategy):
    """
    Chiến lược kết hợp phân tích Fair Value Gap và Order Block
    
    - Sử dụng FVG (Fair Value Gap) để xác định khu vực mất cân bằng cung cầu
    - Kết hợp Order Block để xác định các vùng hỗ trợ/kháng cự mạnh
    - Phát hiện khi giá quay lại các vùng này để xác định cơ hội giao dịch
    - Tối ưu cho khung thời gian 15m
    """
    
    # Tham số cho FVG
    fvg_min_gap = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True)
    fvg_atr_multiplier = DecimalParameter(0.3, 1.0, default=0.5, space="buy", optimize=True)
    
    # Tham số cho Order Block
    ob_strength = DecimalParameter(0.4, 0.8, default=0.6, space="buy", optimize=True)
    ob_atr_multiplier = DecimalParameter(1.0, 2.0, default=1.5, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_ob_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    rsi_os_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    
    # Tham số cho EMAs
    ema_short = IntParameter(5, 30, default=8, space="buy", optimize=True)
    ema_medium = IntParameter(10, 50, default=21, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '15m'
    
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
        
        # Tìm Fair Value Gaps
        fvg_df = detect_fair_value_gaps(
            dataframe,
            min_gap_size=self.fvg_min_gap.value,
            atr_multiplier=self.fvg_atr_multiplier.value
        )
        
        # Thêm các cột FVG
        for column in fvg_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = fvg_df[column]
        
        # Tìm Order Blocks
        ob_df = detect_order_blocks(
            dataframe,
            strength_threshold=self.ob_strength.value,
            atr_multiplier=self.ob_atr_multiplier.value
        )
        
        # Thêm các cột Order Blocks
        for column in ob_df.columns:
            if column not in dataframe.columns and column != 'atr':  # Tránh ghi đè ATR
                dataframe[column] = ob_df[column]
        
        # Thêm các chỉ báo khác
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(
            close=dataframe['close'],
            window=self.rsi_period.value
        )
        
        # EMAs
        for length in [self.ema_short.value, self.ema_medium.value, 50, 100, 200]:
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
        
        # ATR nếu chưa có
        if 'atr' not in dataframe.columns:
            dataframe['atr'] = ta.volatility.average_true_range(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=14
            )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Trend Detection
        dataframe['uptrend'] = dataframe[f'ema_{self.ema_short.value}'] > dataframe[f'ema_{self.ema_medium.value}']
        dataframe['downtrend'] = dataframe[f'ema_{self.ema_short.value}'] < dataframe[f'ema_{self.ema_medium.value}']
        
        # Kết hợp FVG và Order Block
        # Bullish: FVG bullish mitigation + Order Block bullish revisit
        dataframe['bullish_fvg_ob_zone'] = (
            dataframe['bullish_fvg_mitigation'] | dataframe['bullish_ob_revisit']
        )
        
        # Bearish: FVG bearish mitigation + Order Block bearish revisit
        dataframe['bearish_fvg_ob_zone'] = (
            dataframe['bearish_fvg_mitigation'] | dataframe['bearish_ob_revisit']
        )
        
        # Cấu hình tín hiệu mua
        # Bullish Setup: Bullish FVG/OB Zone + RSI thấp + Uptrend/Không downtrend mạnh
        dataframe['bullish_setup'] = (
            dataframe['bullish_fvg_ob_zone'] &
            (dataframe['rsi'] < 50) &
            (~dataframe['downtrend'] | dataframe['uptrend'])
        )
        
        # Bearish Setup: Bearish FVG/OB Zone + RSI cao + Downtrend/Không uptrend mạnh
        dataframe['bearish_setup'] = (
            dataframe['bearish_fvg_ob_zone'] &
            (dataframe['rsi'] > 50) &
            (~dataframe['uptrend'] | dataframe['downtrend'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bullish FVG mitigation + RSI thấp
        conditions_long.append(
            (dataframe['bullish_fvg_mitigation']) &
            (dataframe['rsi'] < self.rsi_os_threshold.value + 10) &
            (~dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Bullish Order Block revisit + RSI thấp
        conditions_long.append(
            (dataframe['bullish_ob_revisit']) &
            (dataframe['rsi'] < 45) &
            (~dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Bullish FVG+OB Zone + Bullish candle + Uptrend
        conditions_long.append(
            (dataframe['bullish_fvg_ob_zone']) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bullish FVG+OB Zone + RSI oversold bounce
        conditions_long.append(
            (dataframe['bullish_fvg_ob_zone']) &
            (dataframe['rsi'] < self.rsi_os_threshold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI đang tăng
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bullish setup + Break above EMA
        conditions_long.append(
            (dataframe['bullish_setup']) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) <= dataframe[f'ema_{self.ema_short.value}'].shift(1)) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bearish FVG mitigation + RSI cao
        conditions_short.append(
            (dataframe['bearish_fvg_mitigation']) &
            (dataframe['rsi'] > self.rsi_ob_threshold.value - 10) &
            (~dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Bearish Order Block revisit + RSI cao
        conditions_short.append(
            (dataframe['bearish_ob_revisit']) &
            (dataframe['rsi'] > 55) &
            (~dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Bearish FVG+OB Zone + Bearish candle + Downtrend
        conditions_short.append(
            (dataframe['bearish_fvg_ob_zone']) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bearish FVG+OB Zone + RSI overbought drop
        conditions_short.append(
            (dataframe['bearish_fvg_ob_zone']) &
            (dataframe['rsi'] > self.rsi_ob_threshold.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI đang giảm
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Bearish setup + Break below EMA
        conditions_short.append(
            (dataframe['bearish_setup']) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) >= dataframe[f'ema_{self.ema_short.value}'].shift(1)) &
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
        
        # 1. Bearish FVG Created
        conditions_exit_long.append(
            dataframe['bearish_fvg']
        )
        
        # 2. Bearish Order Block Created
        conditions_exit_long.append(
            dataframe['bearish_ob']
        )
        
        # 3. RSI Overbought
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_ob_threshold.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))  # RSI đang giảm
        )
        
        # 4. Break below EMA trong uptrend
        conditions_exit_long.append(
            (dataframe['uptrend'].shift(1)) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema_short.value}'].shift(1))
        )
        
        # 5. Price hits first bearish FVG/OB zone
        conditions_exit_long.append(
            dataframe['bearish_fvg_ob_zone']
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Bullish FVG Created
        conditions_exit_short.append(
            dataframe['bullish_fvg']
        )
        
        # 2. Bullish Order Block Created
        conditions_exit_short.append(
            dataframe['bullish_ob']
        )
        
        # 3. RSI Oversold
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_os_threshold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI đang tăng
        )
        
        # 4. Break above EMA trong downtrend
        conditions_exit_short.append(
            (dataframe['downtrend'].shift(1)) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) < dataframe[f'ema_{self.ema_short.value}'].shift(1))
        )
        
        # 5. Price hits first bullish FVG/OB zone
        conditions_exit_short.append(
            dataframe['bullish_fvg_ob_zone']
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu FVG và Order Block
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        if side == 'long':
            # Bullish FVG mitigation + Bullish OB revisit + RSI thấp
            if (last_candle['bullish_fvg_mitigation'] and
                last_candle['bullish_ob_revisit'] and
                last_candle['rsi'] < self.rsi_os_threshold.value):
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Bullish FVG mitigation hoặc Bullish OB revisit + RSI thấp + Uptrend
            elif ((last_candle['bullish_fvg_mitigation'] or last_candle['bullish_ob_revisit']) and
                 last_candle['rsi'] < 40 and
                 last_candle['uptrend']):
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # Bullish FVG+OB zone + Không downtrend mạnh
            elif last_candle['bullish_fvg_ob_zone'] and not last_candle['downtrend']:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # Bullish setup đơn thuần
            elif last_candle['bullish_setup']:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # Bearish FVG mitigation + Bearish OB revisit + RSI cao
            if (last_candle['bearish_fvg_mitigation'] and
                last_candle['bearish_ob_revisit'] and
                last_candle['rsi'] > self.rsi_ob_threshold.value):
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Bearish FVG mitigation hoặc Bearish OB revisit + RSI cao + Downtrend
            elif ((last_candle['bearish_fvg_mitigation'] or last_candle['bearish_ob_revisit']) and
                 last_candle['rsi'] > 60 and
                 last_candle['downtrend']):
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # Bearish FVG+OB zone + Không uptrend mạnh
            elif last_candle['bearish_fvg_ob_zone'] and not last_candle['uptrend']:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # Bearish setup đơn thuần
            elif last_candle['bearish_setup']:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên FVG và Order Block
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        if current_profit > 0.01:  # > 1% lợi nhuận
            if not trade.is_short:
                # Long trade
                
                # Sử dụng vùng bullish FVG/OB gần nhất làm stoploss
                if not pd.isna(last_candle['bullish_fvg_zone_low']):
                    fvg_stop = last_candle['bullish_fvg_zone_low']
                    sl_distance = (fvg_stop / current_rate) - 1
                    return max(sl_distance, -0.05)  # Không quá -5%
                
                if not pd.isna(last_candle['bullish_ob_zone_low']):
                    ob_stop = last_candle['bullish_ob_zone_low']
                    sl_distance = (ob_stop / current_rate) - 1
                    return max(sl_distance, -0.05)  # Không quá -5%
                
                # Nếu không có FVG/OB, sử dụng EMA
                ema_short = last_candle[f'ema_{self.ema_short.value}']
                if current_rate > ema_short:
                    sl_distance = (ema_short / current_rate) - 1
                    return max(sl_distance, -0.05)  # Không quá -5%
                
            else:
                # Short trade
                
                # Sử dụng vùng bearish FVG/OB gần nhất làm stoploss
                if not pd.isna(last_candle['bearish_fvg_zone_high']):
                    fvg_stop = last_candle['bearish_fvg_zone_high']
                    sl_distance = 1 - (fvg_stop / current_rate)
                    return max(sl_distance, -0.05)  # Không quá -5%
                
                if not pd.isna(last_candle['bearish_ob_zone_high']):
                    ob_stop = last_candle['bearish_ob_zone_high']
                    sl_distance = 1 - (ob_stop / current_rate)
                    return max(sl_distance, -0.05)  # Không quá -5%
                
                # Nếu không có FVG/OB, sử dụng EMA
                ema_short = last_candle[f'ema_{self.ema_short.value}']
                if current_rate < ema_short:
                    sl_distance = 1 - (ema_short / current_rate)
                    return max(sl_distance, -0.05)  # Không quá -5%
        
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
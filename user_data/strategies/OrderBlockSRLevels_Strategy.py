"""
Order Block + Support/Resistance Levels Strategy
Chiến lược kết hợp Order Block và phân tích vùng hỗ trợ/kháng cự
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

def detect_order_blocks(dataframe: pd.DataFrame, window: int = 20, strength_threshold: float = 0.5) -> pd.DataFrame:
    """
    Phát hiện Order Blocks - khu vực nơi các "smart money" đặt lệnh lớn
    
    - Bullish Order Block: Nến giảm trước chuỗi tăng mạnh
    - Bearish Order Block: Nến tăng trước chuỗi giảm mạnh
    """
    df = dataframe.copy()
    
    # Tính biến động trung bình (ATR simplification)
    df['avg_candle_size'] = (df['high'] - df['low']).rolling(window=window).mean()
    
    # Xác định các nến có biến động lớn (impulsive moves)
    df['big_move_up'] = (
        (df['close'] > df['open']) &
        ((df['close'] - df['open']) > strength_threshold * df['avg_candle_size'])
    )
    
    df['big_move_down'] = (
        (df['close'] < df['open']) &
        ((df['open'] - df['close']) > strength_threshold * df['avg_candle_size'])
    )
    
    # Đánh dấu khu vực Order Block
    # Bullish Order Block: nến giảm trước chuỗi tăng mạnh
    df['bullish_ob_start'] = False
    for i in range(3, len(df)):
        # Tìm nến giảm trước chuỗi tăng
        if (df['close'].iloc[i-3] < df['open'].iloc[i-3]) and \
           (df['big_move_up'].iloc[i-2] or df['big_move_up'].iloc[i-1] or df['big_move_up'].iloc[i]):
            df['bullish_ob_start'].iloc[i-3] = True
    
    # Bearish Order Block: nến tăng trước chuỗi giảm mạnh
    df['bearish_ob_start'] = False
    for i in range(3, len(df)):
        # Tìm nến tăng trước chuỗi giảm
        if (df['close'].iloc[i-3] > df['open'].iloc[i-3]) and \
           (df['big_move_down'].iloc[i-2] or df['big_move_down'].iloc[i-1] or df['big_move_down'].iloc[i]):
            df['bearish_ob_start'].iloc[i-3] = True
    
    # Xác định giới hạn của OB
    df['bullish_ob_high'] = np.where(df['bullish_ob_start'], df['high'], np.nan)
    df['bullish_ob_low'] = np.where(df['bullish_ob_start'], df['low'], np.nan)
    
    df['bearish_ob_high'] = np.where(df['bearish_ob_start'], df['high'], np.nan)
    df['bearish_ob_low'] = np.where(df['bearish_ob_start'], df['low'], np.nan)
    
    # Forward fill để duy trì Order Block gần nhất
    df['last_bullish_ob_high'] = df['bullish_ob_high'].fillna(method='ffill')
    df['last_bullish_ob_low'] = df['bullish_ob_low'].fillna(method='ffill')
    
    df['last_bearish_ob_high'] = df['bearish_ob_high'].fillna(method='ffill')
    df['last_bearish_ob_low'] = df['bearish_ob_low'].fillna(method='ffill')
    
    # Kiểm tra xem giá hiện tại có về được Order Block không
    df['near_bullish_ob'] = (
        (df['low'] <= df['last_bullish_ob_high']) &
        (df['close'] >= df['last_bullish_ob_low'])
    )
    
    df['near_bearish_ob'] = (
        (df['high'] >= df['last_bearish_ob_low']) &
        (df['close'] <= df['last_bearish_ob_high'])
    )
    
    return df

def detect_sr_levels(dataframe: pd.DataFrame, window: int = 14, confirmation: int = 3, threshold_pct: float = 0.2) -> pd.DataFrame:
    """
    Phát hiện các mức hỗ trợ/kháng cự dựa trên swing high/low
    
    window: Số nến để xác định swing high/low
    confirmation: Số lần chạm để xác nhận level
    threshold_pct: % sai số cho phép khi tìm mức giá tương tự
    """
    df = dataframe.copy()
    
    # Xác định swing high/low
    df['swing_high'] = False
    df['swing_low'] = False
    
    for i in range(window, len(df) - window):
        # Swing High: high cao hơn 'window' nến bên trái và phải
        if all(df['high'].iloc[i] > df['high'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+j] for j in range(1, window+1)):
            df['swing_high'].iloc[i] = True
        
        # Swing Low: low thấp hơn 'window' nến bên trái và phải
        if all(df['low'].iloc[i] < df['low'].iloc[i-j] for j in range(1, window+1)) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+j] for j in range(1, window+1)):
            df['swing_low'].iloc[i] = True
    
    # Tạo danh sách các mức kháng cự/hỗ trợ
    resistance_levels = []
    support_levels = []
    
    # Trích xuất giá trị của swing high/low
    swing_high_values = df.loc[df['swing_high'], 'high'].values
    swing_low_values = df.loc[df['swing_low'], 'low'].values
    
    # Gom nhóm các mức giá gần nhau
    def group_similar_levels(levels, threshold):
        result = []
        for level in levels:
            if not result:
                result.append(level)
                continue
            
            # Kiểm tra xem level có gần mức nào trong result không
            is_grouped = False
            for i, grouped_level in enumerate(result):
                if abs(level - grouped_level) / grouped_level < threshold / 100:
                    # Lấy trung bình
                    result[i] = (grouped_level + level) / 2
                    is_grouped = True
                    break
            
            if not is_grouped:
                result.append(level)
        
        return sorted(result)
    
    # Gom nhóm các mức kháng cự/hỗ trợ
    resistance_levels = group_similar_levels(swing_high_values, threshold_pct)
    support_levels = group_similar_levels(swing_low_values, threshold_pct)
    
    # Tính số lần chạm mỗi mức
    def count_touches(df, levels, is_resistance=True):
        touches = {level: 0 for level in levels}
        tolerance = threshold_pct / 100  # % sai số cho phép
        
        for i in range(len(df)):
            for level in levels:
                if is_resistance:
                    # Chạm vào kháng cự: high gần level
                    if abs(df['high'].iloc[i] - level) / level < tolerance:
                        touches[level] += 1
                else:
                    # Chạm vào hỗ trợ: low gần level
                    if abs(df['low'].iloc[i] - level) / level < tolerance:
                        touches[level] += 1
        
        return touches
    
    # Đếm số lần chạm
    resistance_touches = count_touches(df, resistance_levels, is_resistance=True)
    support_touches = count_touches(df, support_levels, is_resistance=False)
    
    # Lọc các mức có đủ xác nhận
    valid_resistance = [level for level, touches in resistance_touches.items() if touches >= confirmation]
    valid_support = [level for level, touches in support_touches.items() if touches >= confirmation]
    
    # Khởi tạo cột cho các mức SR
    df['nearest_resistance'] = np.nan
    df['nearest_support'] = np.nan
    df['distance_to_resistance'] = np.nan
    df['distance_to_support'] = np.nan
    df['sr_zone'] = False
    
    # Tìm mức SR gần nhất cho mỗi nến
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        # Tìm kháng cự gần nhất (ở trên giá)
        higher_resistances = [r for r in valid_resistance if r > price]
        if higher_resistances:
            nearest_resistance = min(higher_resistances)
            df['nearest_resistance'].iloc[i] = nearest_resistance
            df['distance_to_resistance'].iloc[i] = (nearest_resistance - price) / price * 100
        
        # Tìm hỗ trợ gần nhất (ở dưới giá)
        lower_supports = [s for s in valid_support if s < price]
        if lower_supports:
            nearest_support = max(lower_supports)
            df['nearest_support'].iloc[i] = nearest_support
            df['distance_to_support'].iloc[i] = (price - nearest_support) / price * 100
        
        # Xác định nếu giá đang ở vùng SR
        if (df['distance_to_resistance'].iloc[i] is not np.nan and df['distance_to_resistance'].iloc[i] < 1.0) or \
           (df['distance_to_support'].iloc[i] is not np.nan and df['distance_to_support'].iloc[i] < 1.0):
            df['sr_zone'].iloc[i] = True
    
    return df, valid_resistance, valid_support

class OrderBlockSRLevels_Strategy(IStrategy):
    """
    Chiến lược kết hợp phân tích Order Block và Support/Resistance Levels
    
    - Sử dụng Order Block để phát hiện khu vực lệnh lớn từ smart money
    - Kết hợp với vùng hỗ trợ/kháng cự để tìm điểm vào lệnh mạnh
    - Tối ưu cho giao dịch xu hướng tầm trung (1h)
    """
    
    # Tham số cho Order Block
    ob_window = IntParameter(10, 50, default=20, space="buy", optimize=True)
    ob_strength = DecimalParameter(0.2, 1.5, default=0.5, space="buy", optimize=True)
    ob_confirmation = IntParameter(1, 4, default=2, space="buy", optimize=True)
    
    # Tham số cho Support/Resistance
    sr_window = IntParameter(5, 30, default=14, space="buy", optimize=True)
    sr_confirmation = IntParameter(2, 6, default=3, space="buy", optimize=True)
    sr_threshold = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho trend
    ema_short = IntParameter(5, 30, default=21, space="buy", optimize=True)
    ema_long = IntParameter(30, 100, default=55, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.04
    
    # Time-in-force và ROI
    timeframe = '1h'
    timeframe_lower = '15m'
    
    minimal_roi = {
        "0": 0.06,     # 6%
        "24": 0.04,    # 4% sau 24 giờ 
        "48": 0.025,   # 2.5% sau 48 giờ
        "72": 0.015    # 1.5% sau 72 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.03  # 3%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200  # Cần dữ liệu dài để phát hiện SR
    process_only_new_candles = True
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_lower) for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Thêm Order Blocks vào dataframe
        ob_df = detect_order_blocks(
            dataframe,
            window=self.ob_window.value,
            strength_threshold=self.ob_strength.value
        )
        
        # Thêm các cột OB vào dataframe
        for column in ob_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = ob_df[column]
        
        # Thêm Support/Resistance Levels vào dataframe
        sr_df, resistance_levels, support_levels = detect_sr_levels(
            dataframe,
            window=self.sr_window.value,
            confirmation=self.sr_confirmation.value,
            threshold_pct=self.sr_threshold.value
        )
        
        # Thêm các cột SR vào dataframe
        for column in sr_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = sr_df[column]
        
        # Lấy dữ liệu từ timeframe thấp hơn
        if self.dp:
            informative = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe=self.timeframe_lower
            )
            
            if len(informative) > 0:
                # Thêm chỉ báo cho timeframe thấp hơn
                # RSI
                for length in range(7, 22):
                    informative[f'rsi_{length}'] = ta.momentum.rsi(
                        close=informative['close'],
                        window=length
                    )
                
                # Đổi tên để tránh trùng lặp
                informative_columns = {
                    f"inf_{col}": informative[col] for col in informative.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']
                }
                
                # Merge với dataframe gốc
                dataframe = pd.merge(
                    dataframe, informative_columns, left_on='date', right_on='date', how='left'
                )
        
        # Thêm các chỉ báo thông thường
        # RSI
        for length in range(7, 22):
            dataframe[f'rsi_{length}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=length
            )
        
        # EMA
        for length in range(5, 101, 5):
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
        
        # Xác định xu hướng
        ema_short = self.ema_short.value
        ema_long = self.ema_long.value
        
        dataframe['uptrend'] = dataframe[f'ema_{ema_short}'] > dataframe[f'ema_{ema_long}']
        dataframe['downtrend'] = dataframe[f'ema_{ema_short}'] < dataframe[f'ema_{ema_long}']
        
        # Xác định strong trend
        dataframe['strong_uptrend'] = (
            dataframe['uptrend'] &
            (dataframe[f'ema_{ema_short}'].shift(10) < dataframe[f'ema_{ema_short}']) &
            (dataframe['close'] > dataframe[f'ema_{ema_short}'])
        )
        
        dataframe['strong_downtrend'] = (
            dataframe['downtrend'] &
            (dataframe[f'ema_{ema_short}'].shift(10) > dataframe[f'ema_{ema_short}']) &
            (dataframe['close'] < dataframe[f'ema_{ema_short}'])
        )
        
        # Xác định tín hiệu Order Block + SR
        # 1. Bullish signal: giá trở lại Order Block bullish gần vùng hỗ trợ
        dataframe['ob_sr_bullish'] = (
            dataframe['near_bullish_ob'] &
            (dataframe['distance_to_support'] < 2.0) &  # Gần vùng hỗ trợ (within 2%)
            (dataframe[f'rsi_{self.rsi_period.value}'] < 45) &  # RSI không quá cao
            (~dataframe['strong_downtrend'])  # Không trong downtrend mạnh
        )
        
        # 2. Bearish signal: giá trở lại Order Block bearish gần vùng kháng cự
        dataframe['ob_sr_bearish'] = (
            dataframe['near_bearish_ob'] &
            (dataframe['distance_to_resistance'] < 2.0) &  # Gần vùng kháng cự (within 2%)
            (dataframe[f'rsi_{self.rsi_period.value}'] > 55) &  # RSI không quá thấp
            (~dataframe['strong_uptrend'])  # Không trong uptrend mạnh
        )
        
        # Sức mạnh của tín hiệu
        # Điểm cho bullish signal dựa trên:
        # - Điểm 1: Gần vùng hỗ trợ (càng gần càng tốt)
        # - Điểm 2: RSI thấp
        # - Điểm 3: Volume cao
        # - Điểm 4: Gần Order Block
        dataframe['bullish_signal_strength'] = 0
        mask = dataframe['ob_sr_bullish']
        if mask.any():
            # Điểm cho khoảng cách đến hỗ trợ (max 30)
            support_score = np.where(
                dataframe['distance_to_support'] < 0.5, 30,
                np.where(dataframe['distance_to_support'] < 1.0, 25,
                np.where(dataframe['distance_to_support'] < 1.5, 20,
                np.where(dataframe['distance_to_support'] < 2.0, 15, 0)))
            )
            
            # Điểm cho RSI (max 30)
            rsi = dataframe[f'rsi_{self.rsi_period.value}']
            rsi_score = np.where(
                rsi < 30, 30,
                np.where(rsi < 35, 25,
                np.where(rsi < 40, 20,
                np.where(rsi < 45, 15, 0)))
            )
            
            # Điểm cho volume (max 20)
            volume_score = np.where(
                dataframe['volume_ratio'] > 2.0, 20,
                np.where(dataframe['volume_ratio'] > 1.5, 15,
                np.where(dataframe['volume_ratio'] > 1.0, 10, 0))
            )
            
            # Điểm cho Order Block (max 20)
            # Khoảng thời gian từ khi OB hình thành (càng gần càng tốt)
            ob_score = np.where(
                dataframe['bullish_ob_start'].shift(1) | dataframe['bullish_ob_start'].shift(2), 20,
                np.where(dataframe['bullish_ob_start'].shift(3) | dataframe['bullish_ob_start'].shift(4), 15,
                np.where(dataframe['bullish_ob_start'].shift(5) | dataframe['bullish_ob_start'].shift(10), 10, 5))
            )
            
            # Tổng điểm
            dataframe.loc[mask, 'bullish_signal_strength'] = support_score + rsi_score + volume_score + ob_score
        
        # Tương tự cho bearish signal
        dataframe['bearish_signal_strength'] = 0
        mask = dataframe['ob_sr_bearish']
        if mask.any():
            # Điểm cho khoảng cách đến kháng cự (max 30)
            resistance_score = np.where(
                dataframe['distance_to_resistance'] < 0.5, 30,
                np.where(dataframe['distance_to_resistance'] < 1.0, 25,
                np.where(dataframe['distance_to_resistance'] < 1.5, 20,
                np.where(dataframe['distance_to_resistance'] < 2.0, 15, 0)))
            )
            
            # Điểm cho RSI (max 30)
            rsi = dataframe[f'rsi_{self.rsi_period.value}']
            rsi_score = np.where(
                rsi > 70, 30,
                np.where(rsi > 65, 25,
                np.where(rsi > 60, 20,
                np.where(rsi > 55, 15, 0)))
            )
            
            # Điểm cho volume (max 20)
            volume_score = np.where(
                dataframe['volume_ratio'] > 2.0, 20,
                np.where(dataframe['volume_ratio'] > 1.5, 15,
                np.where(dataframe['volume_ratio'] > 1.0, 10, 0))
            )
            
            # Điểm cho Order Block (max 20)
            # Khoảng thời gian từ khi OB hình thành (càng gần càng tốt)
            ob_score = np.where(
                dataframe['bearish_ob_start'].shift(1) | dataframe['bearish_ob_start'].shift(2), 20,
                np.where(dataframe['bearish_ob_start'].shift(3) | dataframe['bearish_ob_start'].shift(4), 15,
                np.where(dataframe['bearish_ob_start'].shift(5) | dataframe['bearish_ob_start'].shift(10), 10, 5))
            )
            
            # Tổng điểm
            dataframe.loc[mask, 'bearish_signal_strength'] = resistance_score + rsi_score + volume_score + ob_score
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Order Block bullish + SR + RSI thấp
        conditions_long.append(
            (dataframe['ob_sr_bullish']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < self.rsi_buy.value) &
            (dataframe['bullish_signal_strength'] > 60) &  # Tín hiệu đủ mạnh
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Chạm vùng hỗ trợ mạnh + RSI thấp + Uptrend
        conditions_long.append(
            (dataframe['distance_to_support'] < 0.5) &  # Rất gần vùng hỗ trợ
            (dataframe[f'rsi_{self.rsi_period.value}'] < 40) &
            (dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Pull back to EMA trong uptrend mạnh
        conditions_long.append(
            (dataframe['strong_uptrend']) &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}'] * 1.01) &  # Giá gần EMA ngắn
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}'] * 0.99) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 45) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bollinger Band bounce trong uptrend
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['close'] < dataframe['bb_lowerband'] * 1.01) &  # Giá gần BB dưới
            (dataframe[f'rsi_{self.rsi_period.value}'] < 40) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Order Block bearish + SR + RSI cao
        conditions_short.append(
            (dataframe['ob_sr_bearish']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > self.rsi_sell.value) &
            (dataframe['bearish_signal_strength'] > 60) &  # Tín hiệu đủ mạnh
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Chạm vùng kháng cự mạnh + RSI cao + Downtrend
        conditions_short.append(
            (dataframe['distance_to_resistance'] < 0.5) &  # Rất gần vùng kháng cự
            (dataframe[f'rsi_{self.rsi_period.value}'] > 60) &
            (dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Pull back to EMA trong downtrend mạnh
        conditions_short.append(
            (dataframe['strong_downtrend']) &
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}'] * 0.99) &  # Giá gần EMA ngắn
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}'] * 1.01) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 55) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. Bollinger Band bounce trong downtrend
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['close'] > dataframe['bb_upperband'] * 0.99) &  # Giá gần BB trên
            (dataframe[f'rsi_{self.rsi_period.value}'] > 60) &
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
        
        # 1. Chạm vùng kháng cự
        conditions_exit_long.append(
            (dataframe['distance_to_resistance'] < 0.75) &  # Gần kháng cự
            (dataframe[f'rsi_{self.rsi_period.value}'] > 65)  # RSI cao
        )
        
        # 2. Break EMA trong uptrend
        conditions_exit_long.append(
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema_short.value}'].shift(1)) &
            (dataframe['uptrend'])
        )
        
        # 3. RSI quá cao
        conditions_exit_long.append(
            (dataframe[f'rsi_{self.rsi_period.value}'] > 78)
        )
        
        # 4. Trend change
        conditions_exit_long.append(
            (dataframe[f'ema_{self.ema_short.value}'] < dataframe[f'ema_{self.ema_long.value}']) &
            (dataframe[f'ema_{self.ema_short.value}'].shift(1) > dataframe[f'ema_{self.ema_long.value}'].shift(1))
        )
        
        # 5. Near Bearish Order Block
        conditions_exit_long.append(
            (dataframe['near_bearish_ob']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 65)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Chạm vùng hỗ trợ
        conditions_exit_short.append(
            (dataframe['distance_to_support'] < 0.75) &  # Gần hỗ trợ
            (dataframe[f'rsi_{self.rsi_period.value}'] < 35)  # RSI thấp
        )
        
        # 2. Break EMA trong downtrend
        conditions_exit_short.append(
            (dataframe['close'] > dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) < dataframe[f'ema_{self.ema_short.value}'].shift(1)) &
            (dataframe['downtrend'])
        )
        
        # 3. RSI quá thấp
        conditions_exit_short.append(
            (dataframe[f'rsi_{self.rsi_period.value}'] < 22)
        )
        
        # 4. Trend change
        conditions_exit_short.append(
            (dataframe[f'ema_{self.ema_short.value}'] > dataframe[f'ema_{self.ema_long.value}']) &
            (dataframe[f'ema_{self.ema_short.value}'].shift(1) < dataframe[f'ema_{self.ema_long.value}'].shift(1))
        )
        
        # 5. Near Bullish Order Block
        conditions_exit_short.append(
            (dataframe['near_bullish_ob']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 35)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên sức mạnh tín hiệu và khoảng cách đến S/R
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Xác định độ mạnh của tín hiệu và khoảng cách đến SR
        if side == 'long':
            signal_strength = last_candle['bullish_signal_strength']
            
            if signal_strength > 80 and last_candle['volume_ratio'] > 1.5:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif signal_strength > 70:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif signal_strength > 60:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            signal_strength = last_candle['bearish_signal_strength']
            
            if signal_strength > 80 and last_candle['volume_ratio'] > 1.5:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif signal_strength > 70:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif signal_strength > 60:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên SR level gần nhất
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Tính SL dựa trên SR level
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: SL tại mức hỗ trợ gần nhất
                if not np.isnan(last_candle['nearest_support']):
                    support = last_candle['nearest_support']
                    sl_distance = (support / current_rate) - 1
                    return max(sl_distance, -0.1)  # Không quá 10%
            else:
                # Short trade: SL tại mức kháng cự gần nhất
                if not np.isnan(last_candle['nearest_resistance']):
                    resistance = last_candle['nearest_resistance']
                    sl_distance = 1 - (resistance / current_rate)
                    return max(sl_distance, -0.1)  # Không quá 10%
        
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
        
        # Kiểm tra sức mạnh tín hiệu
        if side == 'long' and last_candle['bullish_signal_strength'] < 50:
            return False
        
        if side == 'short' and last_candle['bearish_signal_strength'] < 50:
            return False
        
        # Không long khi giá gần kháng cự
        if side == 'long' and not np.isnan(last_candle['distance_to_resistance']) and last_candle['distance_to_resistance'] < 1.0:
            return False
        
        # Không short khi giá gần hỗ trợ
        if side == 'short' and not np.isnan(last_candle['distance_to_support']) and last_candle['distance_to_support'] < 1.0:
            return False
        
        return True
"""
Volume Profile Strategy
Chiến lược dựa trên phân tích Volume Profile và POC (Point of Control)
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

def calculate_volume_profile(dataframe, period=100, bins=20):
    """
    Tính Volume Profile và POC (Point of Control)
    
    period: số nến để tính Volume Profile
    bins: số vùng giá để phân tích
    """
    df = dataframe.copy().iloc[-period:]
    
    # Tìm high và low của toàn bộ phạm vi giá
    min_price = df['low'].min()
    max_price = df['high'].max()
    
    # Tạo các mức giá từ min đến max
    price_levels = np.linspace(min_price, max_price, bins+1)
    
    # Khởi tạo mass để lưu volume tại mỗi mức giá
    volumes = [0] * bins
    
    # Tính volume ở mỗi mức giá
    for index, row in df.iterrows():
        # Với mỗi nến, phân bổ volume vào các khoảng giá mà nến chạm đến
        candle_min = row['low']
        candle_max = row['high']
        volume = row['volume']
        
        # Tính tỉ lệ và phân bổ volume vào các khoảng giá
        for i in range(bins):
            level_min = price_levels[i]
            level_max = price_levels[i+1]
            
            # Nếu nến có phần overlap với mức giá này
            if candle_max >= level_min and candle_min <= level_max:
                # Tính phần overlap
                overlap_min = max(candle_min, level_min)
                overlap_max = min(candle_max, level_max)
                
                # Tỉ lệ overlap so với tổng biên độ của nến
                overlap_ratio = (overlap_max - overlap_min) / (candle_max - candle_min)
                
                # Phân bổ volume theo tỉ lệ
                volumes[i] += volume * overlap_ratio
    
    # Tìm POC (Point of Control) - mức giá có volume cao nhất
    poc_index = np.argmax(volumes)
    poc_price = (price_levels[poc_index] + price_levels[poc_index+1]) / 2
    
    # Tìm giá trị VAH (Value Area High) và VAL (Value Area Low)
    # Value Area chứa 70% volume giao dịch
    total_volume = sum(volumes)
    target_volume = 0.7 * total_volume
    
    current_volume = volumes[poc_index]
    
    vah_index = poc_index
    val_index = poc_index
    
    while current_volume < target_volume and (vah_index < bins-1 or val_index > 0):
        # Mở rộng hai bên từ POC
        volume_above = volumes[vah_index+1] if vah_index < bins-1 else 0
        volume_below = volumes[val_index-1] if val_index > 0 else 0
        
        # Ưu tiên mở rộng theo hướng có volume cao hơn
        if volume_above > volume_below and vah_index < bins-1:
            vah_index += 1
            current_volume += volume_above
        elif val_index > 0:
            val_index -= 1
            current_volume += volume_below
        else:
            vah_index += 1
            current_volume += volume_above
    
    # Tính giá trị VAH và VAL
    vah_price = price_levels[vah_index+1]
    val_price = price_levels[val_index]
    
    # Tạo dataframe kết quả
    result = pd.DataFrame({
        'price_level': [(price_levels[i] + price_levels[i+1])/2 for i in range(bins)],
        'volume': volumes
    })
    
    # Thêm POC, VAH, VAL
    result = result.sort_values('price_level', ascending=False).reset_index(drop=True)
    
    return result, poc_price, vah_price, val_price

class VolumeProfile_Strategy(IStrategy):
    """
    Chiến lược dựa trên phân tích Volume Profile và POC (Point of Control)
    
    - Sử dụng Volume Profile để xác định vùng giao dịch quan trọng
    - Kết hợp EMA và Stochastic để xác định xu hướng
    - Giao dịch reversal tại vùng Value Area
    - Tối ưu cho khung thời gian 4h
    """
    
    # Tham số cho Volume Profile
    vp_period = IntParameter(50, 200, default=100, space="buy", optimize=True)
    vp_bins = IntParameter(10, 50, default=20, space="buy", optimize=True)
    
    # Tham số cho EMA
    ema_fast = IntParameter(5, 50, default=21, space="buy", optimize=True)
    ema_medium = IntParameter(20, 100, default=55, space="buy", optimize=True)
    ema_slow = IntParameter(50, 200, default=100, space="buy", optimize=True)
    
    # Tham số cho Stochastic
    stoch_k = IntParameter(5, 21, default=14, space="buy", optimize=True)
    stoch_d = IntParameter(3, 10, default=3, space="buy", optimize=True)
    stoch_smooth = IntParameter(1, 5, default=3, space="buy", optimize=True)
    
    # Tham số cho xác nhận Value Area
    va_width_min = DecimalParameter(1.0, 10.0, default=3.0, space="buy", optimize=True)
    poc_volume_min = DecimalParameter(1.0, 5.0, default=2.0, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.04
    
    # Time-in-force và ROI
    timeframe = '4h'
    
    minimal_roi = {
        "0": 0.08,     # 8%
        "24": 0.05,    # 5% sau 24 giờ
        "72": 0.03,    # 3% sau 72 giờ
        "120": 0.02    # 2% sau 120 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.04  # 4%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # EMA
        for length in range(5, 201, 5):  # [5, 10, 15, ..., 200]
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'], 
                window=length
            )
        
        # Stochastic Oscillator
        for k in range(5, 22, 3):  # [5, 8, 11, 14, 17, 20]
            for d in range(3, 11, 3):  # [3, 6, 9]
                for smooth in range(1, 6):  # [1, 2, 3, 4, 5]
                    stoch = ta.momentum.StochasticOscillator(
                        high=dataframe['high'],
                        low=dataframe['low'],
                        close=dataframe['close'],
                        window=k,
                        smooth_window=smooth,
                        fillna=True
                    )
                    dataframe[f'stoch_{k}_{d}_{smooth}_k'] = stoch.stoch()
                    dataframe[f'stoch_{k}_{d}_{smooth}_d'] = stoch.stoch_signal()
        
        # RSI
        for length in range(7, 22, 7):  # [7, 14, 21]
            dataframe[f'rsi_{length}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=length,
                fillna=True
            )
        
        # ATR
        for length in range(7, 22, 7):  # [7, 14, 21]
            dataframe[f'atr_{length}'] = ta.volatility.average_true_range(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=length,
                fillna=True
            )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Volume Profile (tính cho mỗi nến dựa trên n nến trước đó)
        vp_period = self.vp_period.value
        vp_bins = self.vp_bins.value
        
        # Khởi tạo các cột Volume Profile
        dataframe['poc'] = np.nan
        dataframe['vah'] = np.nan
        dataframe['val'] = np.nan
        dataframe['poc_distance'] = np.nan
        dataframe['in_value_area'] = False
        dataframe['va_width'] = np.nan
        dataframe['poc_strength'] = np.nan
        
        # Tính Volume Profile cho mỗi nến
        # Chỉ tính cho vài nến cuối để tránh tính toán quá nhiều
        for i in range(len(dataframe) - min(100, len(dataframe)), len(dataframe)):
            if i >= vp_period:
                df_slice = dataframe.iloc[i-vp_period:i].copy()
                _, poc, vah, val = calculate_volume_profile(df_slice, vp_period, vp_bins)
                
                # Tính POC strength (tỉ lệ volume tại POC so với trung bình)
                bin_range = (df_slice['high'].max() - df_slice['low'].min()) / vp_bins
                poc_high = poc + bin_range/2
                poc_low = poc - bin_range/2
                
                # Tính volume tại vùng POC
                poc_volume = 0
                for _, row in df_slice.iterrows():
                    if row['low'] <= poc_high and row['high'] >= poc_low:
                        # Tính phần overlap
                        overlap_min = max(row['low'], poc_low)
                        overlap_max = min(row['high'], poc_high)
                        
                        # Tỉ lệ overlap so với tổng biên độ của nến
                        if row['high'] != row['low']:  # Tránh chia cho 0
                            overlap_ratio = (overlap_max - overlap_min) / (row['high'] - row['low'])
                            poc_volume += row['volume'] * overlap_ratio
                
                # So sánh với volume trung bình
                avg_volume = df_slice['volume'].mean()
                poc_strength = poc_volume / (avg_volume * bin_range) if avg_volume > 0 else 0
                
                # Thêm vào dataframe
                dataframe.loc[dataframe.index[i], 'poc'] = poc
                dataframe.loc[dataframe.index[i], 'vah'] = vah
                dataframe.loc[dataframe.index[i], 'val'] = val
                dataframe.loc[dataframe.index[i], 'va_width'] = (vah - val) / val * 100  # % width
                dataframe.loc[dataframe.index[i], 'poc_strength'] = poc_strength
                
                # Tính khoảng cách từ giá hiện tại đến POC
                close = dataframe.loc[dataframe.index[i], 'close']
                dataframe.loc[dataframe.index[i], 'poc_distance'] = (close - poc) / poc * 100  # % distance
                
                # Kiểm tra nếu giá đang trong vùng Value Area
                if val <= close <= vah:
                    dataframe.loc[dataframe.index[i], 'in_value_area'] = True
        
        # Forward fill các giá trị Volume Profile
        dataframe['poc'] = dataframe['poc'].ffill()
        dataframe['vah'] = dataframe['vah'].ffill()
        dataframe['val'] = dataframe['val'].ffill()
        dataframe['va_width'] = dataframe['va_width'].ffill()
        dataframe['poc_strength'] = dataframe['poc_strength'].ffill()
        
        # Xác định xu hướng từ EMA
        ema_fast = self.ema_fast.value
        ema_medium = self.ema_medium.value
        ema_slow = self.ema_slow.value
        
        dataframe['uptrend'] = (
            (dataframe[f'ema_{ema_fast}'] > dataframe[f'ema_{ema_medium}']) & 
            (dataframe[f'ema_{ema_medium}'] > dataframe[f'ema_{ema_slow}'])
        )
        
        dataframe['downtrend'] = (
            (dataframe[f'ema_{ema_fast}'] < dataframe[f'ema_{ema_medium}']) & 
            (dataframe[f'ema_{ema_medium}'] < dataframe[f'ema_{ema_slow}'])
        )
        
        # Stochastic K & D lines
        stoch_k = self.stoch_k.value
        stoch_d = self.stoch_d.value
        stoch_smooth = self.stoch_smooth.value
        
        dataframe['stoch_k'] = dataframe[f'stoch_{stoch_k}_{stoch_d}_{stoch_smooth}_k']
        dataframe['stoch_d'] = dataframe[f'stoch_{stoch_k}_{stoch_d}_{stoch_smooth}_d']
        
        # Tín hiệu giao dịch dựa trên Volume Profile
        # 1. Bouncing from Value Area Low (VAL)
        dataframe['val_bounce_up'] = (
            (dataframe['low'] <= dataframe['val'] * 1.005) &  # Giá chạm hoặc dưới VAL
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['close'] > dataframe['val']) &  # Giá đóng cửa trên VAL
            (dataframe['volume_ratio'] > 1.0)  # Volume lớn
        )
        
        # 2. Bouncing from Value Area High (VAH)
        dataframe['vah_bounce_down'] = (
            (dataframe['high'] >= dataframe['vah'] * 0.995) &  # Giá chạm hoặc trên VAH
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['close'] < dataframe['vah']) &  # Giá đóng cửa dưới VAH
            (dataframe['volume_ratio'] > 1.0)  # Volume lớn
        )
        
        # 3. Approaching POC from below (bullish)
        dataframe['approaching_poc_from_below'] = (
            (dataframe['poc_distance'] < 0) &  # Giá dưới POC
            (dataframe['poc_distance'] > -1.0) &  # Giá gần POC (trong khoảng 1% dưới POC)
            (dataframe['poc_distance'].shift(1) < dataframe['poc_distance']) &  # Đang di chuyển lên gần POC
            (dataframe['poc_strength'] > self.poc_volume_min.value)  # POC đủ mạnh
        )
        
        # 4. Approaching POC from above (bearish)
        dataframe['approaching_poc_from_above'] = (
            (dataframe['poc_distance'] > 0) &  # Giá trên POC
            (dataframe['poc_distance'] < 1.0) &  # Giá gần POC (trong khoảng 1% trên POC)
            (dataframe['poc_distance'].shift(1) > dataframe['poc_distance']) &  # Đang di chuyển xuống gần POC
            (dataframe['poc_strength'] > self.poc_volume_min.value)  # POC đủ mạnh
        )
        
        # 5. Breaking out of Value Area (bullish)
        dataframe['va_breakout_up'] = (
            (dataframe['close'] > dataframe['vah']) &  # Giá đóng cửa trên VAH
            (dataframe['close'].shift(1) <= dataframe['vah'].shift(1)) &  # Nến trước đóng cửa dưới VAH
            (dataframe['volume_ratio'] > 1.2)  # Volume cao
        )
        
        # 6. Breaking out of Value Area (bearish)
        dataframe['va_breakout_down'] = (
            (dataframe['close'] < dataframe['val']) &  # Giá đóng cửa dưới VAL
            (dataframe['close'].shift(1) >= dataframe['val'].shift(1)) &  # Nến trước đóng cửa trên VAL
            (dataframe['volume_ratio'] > 1.2)  # Volume cao
        )
        
        # 7. POC acting as support
        dataframe['poc_support'] = (
            (dataframe['low'] <= dataframe['poc'] * 1.005) &  # Giá chạm hoặc dưới POC một chút
            (dataframe['close'] > dataframe['poc']) &  # Giá đóng cửa trên POC
            (dataframe['poc_strength'] > self.poc_volume_min.value) &  # POC đủ mạnh
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        # 8. POC acting as resistance
        dataframe['poc_resistance'] = (
            (dataframe['high'] >= dataframe['poc'] * 0.995) &  # Giá chạm hoặc trên POC một chút
            (dataframe['close'] < dataframe['poc']) &  # Giá đóng cửa dưới POC
            (dataframe['poc_strength'] > self.poc_volume_min.value) &  # POC đủ mạnh
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Lấy tham số
        va_width_min = self.va_width_min.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Bouncing from Value Area Low (VAL) trong uptrend
        conditions_long.append(
            (dataframe['val_bounce_up']) &
            (~dataframe['downtrend']) &
            (dataframe['va_width'] > va_width_min) &  # Value Area đủ rộng
            (dataframe['stoch_k'] < 30) &  # Stochastic oversold
            (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1))  # Stochastic đang tăng
        )
        
        # 2. Approaching POC from below + bullish momentum
        conditions_long.append(
            (dataframe['approaching_poc_from_below']) &
            (~dataframe['downtrend']) &
            (dataframe['stoch_k'] < 50) &  # Stochastic không quá cao
            (dataframe['stoch_k'] > dataframe['stoch_d']) &  # Stochastic K cắt lên trên D
            (dataframe['rsi_14'] > 40)  # RSI không quá thấp
        )
        
        # 3. POC acting as support trong uptrend
        conditions_long.append(
            (dataframe['poc_support']) &
            (dataframe['uptrend']) &
            (dataframe['stoch_k'] < 60) &  # Stochastic không quá cao
            (dataframe['volume_ratio'] > 1.0)  # Volume khá tốt
        )
        
        # 4. Breaking out of Value Area (bullish)
        conditions_long.append(
            (dataframe['va_breakout_up']) &
            (dataframe['va_width'] > va_width_min) &  # Value Area đủ rộng
            (dataframe['stoch_k'] > 40) &  # Stochastic không quá thấp
            (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1)) &  # Stochastic đang tăng
            (dataframe['volume_ratio'] > 1.5)  # Volume cao
        )
        
        # 5. Strong uptrend + Value Area expanding upward
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['stoch_k'] > 40) & (dataframe['stoch_k'] < 70) &  # Stochastic trung bình
            (dataframe['vah'] > dataframe['vah'].shift(1)) &  # VAH đang tăng
            (dataframe['poc'] > dataframe['poc'].shift(1)) &  # POC đang tăng
            (dataframe['close'] > dataframe['poc']) &  # Giá trên POC
            (dataframe['volume_ratio'] > 1.2)  # Volume khá tốt
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Bouncing from Value Area High (VAH) trong downtrend
        conditions_short.append(
            (dataframe['vah_bounce_down']) &
            (~dataframe['uptrend']) &
            (dataframe['va_width'] > va_width_min) &  # Value Area đủ rộng
            (dataframe['stoch_k'] > 70) &  # Stochastic overbought
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1))  # Stochastic đang giảm
        )
        
        # 2. Approaching POC from above + bearish momentum
        conditions_short.append(
            (dataframe['approaching_poc_from_above']) &
            (~dataframe['uptrend']) &
            (dataframe['stoch_k'] > 50) &  # Stochastic không quá thấp
            (dataframe['stoch_k'] < dataframe['stoch_d']) &  # Stochastic K cắt xuống dưới D
            (dataframe['rsi_14'] < 60)  # RSI không quá cao
        )
        
        # 3. POC acting as resistance trong downtrend
        conditions_short.append(
            (dataframe['poc_resistance']) &
            (dataframe['downtrend']) &
            (dataframe['stoch_k'] > 40) &  # Stochastic không quá thấp
            (dataframe['volume_ratio'] > 1.0)  # Volume khá tốt
        )
        
        # 4. Breaking out of Value Area (bearish)
        conditions_short.append(
            (dataframe['va_breakout_down']) &
            (dataframe['va_width'] > va_width_min) &  # Value Area đủ rộng
            (dataframe['stoch_k'] < 60) &  # Stochastic không quá cao
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1)) &  # Stochastic đang giảm
            (dataframe['volume_ratio'] > 1.5)  # Volume cao
        )
        
        # 5. Strong downtrend + Value Area expanding downward
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['stoch_k'] < 60) & (dataframe['stoch_k'] > 30) &  # Stochastic trung bình
            (dataframe['val'] < dataframe['val'].shift(1)) &  # VAL đang giảm
            (dataframe['poc'] < dataframe['poc'].shift(1)) &  # POC đang giảm
            (dataframe['close'] < dataframe['poc']) &  # Giá dưới POC
            (dataframe['volume_ratio'] > 1.2)  # Volume khá tốt
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
        
        # 1. Reaching Value Area High (VAH) với volume yếu
        conditions_exit_long.append(
            (dataframe['high'] >= dataframe['vah'] * 0.998) &  # Giá chạm hoặc gần VAH
            (dataframe['volume_ratio'] < 1.2) &  # Volume không đủ mạnh để phá vỡ
            (dataframe['stoch_k'] > 75)  # Stochastic overbought
        )
        
        # 2. Rejected at POC từ trên xuống
        conditions_exit_long.append(
            (dataframe['approaching_poc_from_above']) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['stoch_k'] < dataframe['stoch_k'].shift(1))  # Stochastic đang giảm
        )
        
        # 3. Stochastic overbought + bearish cross
        conditions_exit_long.append(
            (dataframe['stoch_k'] > 80) &
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) > dataframe['stoch_d'].shift(1))  # Vừa cắt xuống
        )
        
        # 4. Falling back into Value Area sau khi breakout
        conditions_exit_long.append(
            (dataframe['close'] < dataframe['vah']) &
            (dataframe['close'].shift(1) > dataframe['vah'].shift(1)) &  # Nến trước đóng cửa trên VAH
            (dataframe['volume_ratio'] > 1.0)  # Volume không quá thấp
        )
        
        # 5. EMA cross bearish
        conditions_exit_long.append(
            (dataframe[f'ema_{self.ema_fast.value}'] < dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe[f'ema_{self.ema_fast.value}'].shift(1) >= dataframe[f'ema_{self.ema_medium.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Reaching Value Area Low (VAL) với volume yếu
        conditions_exit_short.append(
            (dataframe['low'] <= dataframe['val'] * 1.002) &  # Giá chạm hoặc gần VAL
            (dataframe['volume_ratio'] < 1.2) &  # Volume không đủ mạnh để phá vỡ
            (dataframe['stoch_k'] < 25)  # Stochastic oversold
        )
        
        # 2. Rejected at POC từ dưới lên
        conditions_exit_short.append(
            (dataframe['approaching_poc_from_below']) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['stoch_k'] > dataframe['stoch_k'].shift(1))  # Stochastic đang tăng
        )
        
        # 3. Stochastic oversold + bullish cross
        conditions_exit_short.append(
            (dataframe['stoch_k'] < 20) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) < dataframe['stoch_d'].shift(1))  # Vừa cắt lên
        )
        
        # 4. Climbing back into Value Area sau khi breakout
        conditions_exit_short.append(
            (dataframe['close'] > dataframe['val']) &
            (dataframe['close'].shift(1) < dataframe['val'].shift(1)) &  # Nến trước đóng cửa dưới VAL
            (dataframe['volume_ratio'] > 1.0)  # Volume không quá thấp
        )
        
        # 5. EMA cross bullish
        conditions_exit_short.append(
            (dataframe[f'ema_{self.ema_fast.value}'] > dataframe[f'ema_{self.ema_medium.value}']) &
            (dataframe[f'ema_{self.ema_fast.value}'].shift(1) <= dataframe[f'ema_{self.ema_medium.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ rõ ràng của tín hiệu Volume Profile
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Kiểm tra độ mạnh của POC và độ rộng của Value Area
        poc_strength = last_candle['poc_strength']
        va_width = last_candle['va_width']
        
        if side == 'long':
            # Tín hiệu bouncing từ VAL rõ ràng
            if last_candle['val_bounce_up'] and poc_strength > self.poc_volume_min.value * 1.5:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Breaking out khỏi Value Area với volume cao
            elif last_candle['va_breakout_up'] and last_candle['volume_ratio'] > 2.0:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # POC acting as strong support
            elif last_candle['poc_support'] and poc_strength > self.poc_volume_min.value:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # Các tín hiệu khác
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # Tín hiệu bouncing từ VAH rõ ràng
            if last_candle['vah_bounce_down'] and poc_strength > self.poc_volume_min.value * 1.5:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Breaking out khỏi Value Area với volume cao
            elif last_candle['va_breakout_down'] and last_candle['volume_ratio'] > 2.0:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            
            # POC acting as strong resistance
            elif last_candle['poc_resistance'] and poc_strength > self.poc_volume_min.value:
                leverage = min(4, max_leverage)  # Đòn bẩy trên trung bình
            
            # Các tín hiệu khác
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Value Area và POC
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy giá trị Volume Profile
        poc = last_candle['poc']
        vah = last_candle['vah']
        val = last_candle['val']
        
        if not trade.is_short:
            # Long: stoploss dựa trên Value Area Low hoặc POC (tùy theo tình huống)
            # Nếu giá đang ở trên POC, đặt stoploss tại POC
            if current_rate > poc and poc > 0:
                sl_distance = (poc / current_rate) - 1
                return max(sl_distance, -0.1)  # Không quá -10%
            
            # Nếu giá đang ở trong Value Area, đặt stoploss tại VAL
            elif current_rate > val and val > 0:
                sl_distance = (val / current_rate) - 1
                return max(sl_distance, -0.1)  # Không quá -10%
        else:
            # Short: stoploss dựa trên Value Area High hoặc POC (tùy theo tình huống)
            # Nếu giá đang ở dưới POC, đặt stoploss tại POC
            if current_rate < poc and poc > 0:
                sl_distance = 1 - (poc / current_rate)
                return max(sl_distance, -0.1)  # Không quá -10%
            
            # Nếu giá đang ở trong Value Area, đặt stoploss tại VAH
            elif current_rate < vah and vah > 0:
                sl_distance = 1 - (vah / current_rate)
                return max(sl_distance, -0.1)  # Không quá -10%
        
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
        
        # Kiểm tra độ mạnh của POC
        if last_candle['poc_strength'] < self.poc_volume_min.value * 0.7:
            return False
        
        # Kiểm tra độ rộng của Value Area
        if last_candle['va_width'] < self.va_width_min.value * 0.7:
            return False
        
        # Không chấp nhận nếu giá đã di chuyển quá xa từ tín hiệu
        if side == 'long' and rate > last_candle['close'] * 1.01:  # Giá vào lệnh cao hơn close 1%
            return False
        
        if side == 'short' and rate < last_candle['close'] * 0.99:  # Giá vào lệnh thấp hơn close 1%
            return False
        
        return True
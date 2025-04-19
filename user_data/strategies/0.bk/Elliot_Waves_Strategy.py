"""
Elliot Waves Strategy
Chiến lược phân tích sóng Elliott kết hợp với Fibonacci và RSI
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

def find_swings(dataframe, window=5, swing_type='both'):
    """
    Tìm swing high và swing low trong dataframe
    
    window: số nến nhìn trước/sau để xác định swing
    swing_type: 'high', 'low' hoặc 'both'
    """
    # Khởi tạo cột swing high và swing low
    dataframe['swing_high'] = 0
    dataframe['swing_low'] = 0
    
    # Tìm swing high
    if swing_type in ['high', 'both']:
        for i in range(window, len(dataframe) - window):
            # Tìm các điểm có giá high cao nhất trong cửa sổ
            if all(dataframe['high'].iloc[i] >= dataframe['high'].iloc[i-j] for j in range(1, window+1)) and \
               all(dataframe['high'].iloc[i] >= dataframe['high'].iloc[i+j] for j in range(1, window+1)):
                dataframe.loc[dataframe.index[i], 'swing_high'] = dataframe['high'].iloc[i]
    
    # Tìm swing low
    if swing_type in ['low', 'both']:
        for i in range(window, len(dataframe) - window):
            # Tìm các điểm có giá low thấp nhất trong cửa sổ
            if all(dataframe['low'].iloc[i] <= dataframe['low'].iloc[i-j] for j in range(1, window+1)) and \
               all(dataframe['low'].iloc[i] <= dataframe['low'].iloc[i+j] for j in range(1, window+1)):
                dataframe.loc[dataframe.index[i], 'swing_low'] = dataframe['low'].iloc[i]
    
    return dataframe

def detect_impulse_wave(dataframe, window=5, threshold_pct=2.0):
    """
    Phát hiện sóng động lực (impulse wave) trong Elliott Wave Theory
    
    window: số nến nhìn trước/sau để xác định swing
    threshold_pct: ngưỡng % tối thiểu giữa các đỉnh/đáy để được coi là sóng hợp lệ
    """
    # Tìm swing high và swing low
    dataframe = find_swings(dataframe, window=window)
    
    # Khởi tạo cột cho impulse wave
    dataframe['impulse_start'] = 0  # Điểm bắt đầu sóng (sóng 1)
    dataframe['impulse_end'] = 0    # Điểm kết thúc sóng (sóng 5)
    dataframe['impulse_phase'] = 0  # Pha hiện tại của sóng (1-5)
    dataframe['impulse_direction'] = ''  # 'up' hoặc 'down'
    
    # Tìm các swing high và swing low có giá trị != 0
    swing_highs = dataframe[dataframe['swing_high'] > 0]
    swing_lows = dataframe[dataframe['swing_low'] > 0]
    
    # Nếu không có đủ swing để phân tích
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return dataframe
    
    # Tìm các sóng động lực đi lên (impulse wave up)
    # Pattern: low - high - low - high - low - high (5 sóng)
    last_wave_up_end = 0
    for i in range(len(dataframe) - window*5):
        # Kiểm tra nếu đã tìm thấy sóng đi lên gần đây
        if last_wave_up_end > i:
            continue
            
        # Tìm swing low đầu tiên sau vị trí hiện tại
        start_lows = swing_lows[swing_lows.index > dataframe.index[i]]
        if len(start_lows) == 0:
            continue
        wave_start_idx = start_lows.index[0]
        wave_start_val = start_lows['swing_low'].iloc[0]
        
        # Tìm swing high kế tiếp (sóng 1)
        wave1_highs = swing_highs[swing_highs.index > wave_start_idx]
        if len(wave1_highs) == 0:
            continue
        wave1_idx = wave1_highs.index[0]
        wave1_val = wave1_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng 1 đủ lớn
        wave1_change = (wave1_val - wave_start_val) / wave_start_val * 100
        if wave1_change < threshold_pct:
            continue
        
        # Tìm swing low kế tiếp (sóng 2)
        wave2_lows = swing_lows[swing_lows.index > wave1_idx]
        if len(wave2_lows) == 0:
            continue
        wave2_idx = wave2_lows.index[0]
        wave2_val = wave2_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng 2 không chiết khấu quá 100% so với sóng 1
        if wave2_val <= wave_start_val:
            continue
        
        # Tìm swing high kế tiếp (sóng 3)
        wave3_highs = swing_highs[swing_highs.index > wave2_idx]
        if len(wave3_highs) == 0:
            continue
        wave3_idx = wave3_highs.index[0]
        wave3_val = wave3_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng 3 cao hơn sóng 1
        if wave3_val <= wave1_val:
            continue
        
        # Tìm swing low kế tiếp (sóng 4)
        wave4_lows = swing_lows[swing_lows.index > wave3_idx]
        if len(wave4_lows) == 0:
            continue
        wave4_idx = wave4_lows.index[0]
        wave4_val = wave4_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng 4 không chiết khấu xuống dưới đỉnh sóng 1
        if wave4_val <= wave1_val:
            continue
        
        # Tìm swing high kế tiếp (sóng 5)
        wave5_highs = swing_highs[swing_highs.index > wave4_idx]
        if len(wave5_highs) == 0:
            continue
        wave5_idx = wave5_highs.index[0]
        wave5_val = wave5_highs['swing_high'].iloc[0]
        
        # Đánh dấu các điểm sóng
        start_i = dataframe.index.get_loc(wave_start_idx)
        wave5_i = dataframe.index.get_loc(wave5_idx)
        
        # Đánh dấu điểm bắt đầu và kết thúc
        dataframe.loc[wave_start_idx, 'impulse_start'] = 1
        dataframe.loc[wave5_idx, 'impulse_end'] = 1
        dataframe.loc[wave_start_idx:wave5_idx, 'impulse_direction'] = 'up'
        
        # Đánh dấu các phase
        wave1_i = dataframe.index.get_loc(wave1_idx)
        wave2_i = dataframe.index.get_loc(wave2_idx)
        wave3_i = dataframe.index.get_loc(wave3_idx)
        wave4_i = dataframe.index.get_loc(wave4_idx)
        
        dataframe.loc[dataframe.index[start_i:wave1_i], 'impulse_phase'] = 1
        dataframe.loc[dataframe.index[wave1_i:wave2_i], 'impulse_phase'] = 2
        dataframe.loc[dataframe.index[wave2_i:wave3_i], 'impulse_phase'] = 3
        dataframe.loc[dataframe.index[wave3_i:wave4_i], 'impulse_phase'] = 4
        dataframe.loc[dataframe.index[wave4_i:wave5_i+1], 'impulse_phase'] = 5
        
        # Cập nhật last_wave_end để tránh tìm thấy các sóng chồng chéo
        last_wave_up_end = wave5_i
    
    # Tìm các sóng động lực đi xuống (impulse wave down)
    # Pattern: high - low - high - low - high - low (5 sóng)
    last_wave_down_end = 0
    for i in range(len(dataframe) - window*5):
        # Kiểm tra nếu đã tìm thấy sóng đi xuống gần đây
        if last_wave_down_end > i:
            continue
            
        # Tìm swing high đầu tiên sau vị trí hiện tại
        start_highs = swing_highs[swing_highs.index > dataframe.index[i]]
        if len(start_highs) == 0:
            continue
        wave_start_idx = start_highs.index[0]
        wave_start_val = start_highs['swing_high'].iloc[0]
        
        # Tìm swing low kế tiếp (sóng 1)
        wave1_lows = swing_lows[swing_lows.index > wave_start_idx]
        if len(wave1_lows) == 0:
            continue
        wave1_idx = wave1_lows.index[0]
        wave1_val = wave1_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng 1 đủ lớn
        wave1_change = (wave_start_val - wave1_val) / wave_start_val * 100
        if wave1_change < threshold_pct:
            continue
        
        # Tìm swing high kế tiếp (sóng 2)
        wave2_highs = swing_highs[swing_highs.index > wave1_idx]
        if len(wave2_highs) == 0:
            continue
        wave2_idx = wave2_highs.index[0]
        wave2_val = wave2_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng 2 không hồi phục quá 100% so với sóng 1
        if wave2_val >= wave_start_val:
            continue
        
        # Tìm swing low kế tiếp (sóng 3)
        wave3_lows = swing_lows[swing_lows.index > wave2_idx]
        if len(wave3_lows) == 0:
            continue
        wave3_idx = wave3_lows.index[0]
        wave3_val = wave3_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng 3 thấp hơn sóng 1
        if wave3_val >= wave1_val:
            continue
        
        # Tìm swing high kế tiếp (sóng 4)
        wave4_highs = swing_highs[swing_highs.index > wave3_idx]
        if len(wave4_highs) == 0:
            continue
        wave4_idx = wave4_highs.index[0]
        wave4_val = wave4_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng 4 không hồi phục cao hơn đáy sóng 1
        if wave4_val >= wave1_val:
            continue
        
        # Tìm swing low kế tiếp (sóng 5)
        wave5_lows = swing_lows[swing_lows.index > wave4_idx]
        if len(wave5_lows) == 0:
            continue
        wave5_idx = wave5_lows.index[0]
        wave5_val = wave5_lows['swing_low'].iloc[0]
        
        # Đánh dấu các điểm sóng
        start_i = dataframe.index.get_loc(wave_start_idx)
        wave5_i = dataframe.index.get_loc(wave5_idx)
        
        # Đánh dấu điểm bắt đầu và kết thúc
        dataframe.loc[wave_start_idx, 'impulse_start'] = -1  # -1 for down impulse
        dataframe.loc[wave5_idx, 'impulse_end'] = -1  # -1 for down impulse
        dataframe.loc[wave_start_idx:wave5_idx, 'impulse_direction'] = 'down'
        
        # Đánh dấu các phase
        wave1_i = dataframe.index.get_loc(wave1_idx)
        wave2_i = dataframe.index.get_loc(wave2_idx)
        wave3_i = dataframe.index.get_loc(wave3_idx)
        wave4_i = dataframe.index.get_loc(wave4_idx)
        
        dataframe.loc[dataframe.index[start_i:wave1_i], 'impulse_phase'] = -1
        dataframe.loc[dataframe.index[wave1_i:wave2_i], 'impulse_phase'] = -2
        dataframe.loc[dataframe.index[wave2_i:wave3_i], 'impulse_phase'] = -3
        dataframe.loc[dataframe.index[wave3_i:wave4_i], 'impulse_phase'] = -4
        dataframe.loc[dataframe.index[wave4_i:wave5_i+1], 'impulse_phase'] = -5
        
        # Cập nhật last_wave_end để tránh tìm thấy các sóng chồng chéo
        last_wave_down_end = wave5_i
    
    return dataframe

def detect_fibonacci_levels(dataframe, direction='both'):
    """
    Phát hiện các mức Fibonacci retracement và extension
    
    direction: 'up', 'down' hoặc 'both' để xác định hướng tìm kiếm
    """
    # Fib Retracement Levels
    fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    # Fib Extension Levels
    fib_ext_levels = [1.0, 1.272, 1.414, 1.618, 2.0, 2.618]
    
    # Khởi tạo các cột Fibonacci
    for level in fib_levels:
        dataframe[f'fib_retr_{level:.3f}'] = 0
    
    for level in fib_ext_levels:
        dataframe[f'fib_ext_{level:.3f}'] = 0
    
    # Đánh dấu Fibonacci Retracement cho các sóng đi lên (bullish)
    if direction in ['up', 'both']:
        up_impulse_end_indices = dataframe[dataframe['impulse_end'] == 1].index
        
        for end_idx in up_impulse_end_indices:
            # Tìm điểm bắt đầu của sóng
            start_indices = dataframe[dataframe['impulse_start'] == 1].index
            start_indices = start_indices[start_indices < end_idx]
            
            if len(start_indices) == 0:
                continue
                
            start_idx = start_indices[-1]  # Điểm bắt đầu gần nhất
            
            # Lấy giá trị thấp nhất và cao nhất trong khoảng sóng
            low_val = dataframe.loc[start_idx:end_idx, 'low'].min()
            high_val = dataframe.loc[start_idx:end_idx, 'high'].max()
            
            # Tính các mức Fibonacci Retracement
            price_range = high_val - low_val
            for level in fib_levels:
                fib_price = high_val - price_range * level
                # Đánh dấu các mức Fibonacci trong 10 nến sau kết thúc sóng
                end_i = dataframe.index.get_loc(end_idx)
                if end_i + 10 < len(dataframe):
                    dataframe.loc[dataframe.index[end_i:end_i+10], f'fib_retr_{level:.3f}'] = fib_price
            
            # Tính các mức Fibonacci Extension
            for level in fib_ext_levels:
                fib_ext_price = low_val + price_range * level
                # Đánh dấu các mức Fibonacci Extension trong 10 nến sau kết thúc sóng
                dataframe.loc[dataframe.index[end_i:end_i+10], f'fib_ext_{level:.3f}'] = fib_ext_price
    
    # Đánh dấu Fibonacci Retracement cho các sóng đi xuống (bearish)
    if direction in ['down', 'both']:
        down_impulse_end_indices = dataframe[dataframe['impulse_end'] == -1].index
        
        for end_idx in down_impulse_end_indices:
            # Tìm điểm bắt đầu của sóng
            start_indices = dataframe[dataframe['impulse_start'] == -1].index
            start_indices = start_indices[start_indices < end_idx]
            
            if len(start_indices) == 0:
                continue
                
            start_idx = start_indices[-1]  # Điểm bắt đầu gần nhất
            
            # Lấy giá trị thấp nhất và cao nhất trong khoảng sóng
            low_val = dataframe.loc[start_idx:end_idx, 'low'].min()
            high_val = dataframe.loc[start_idx:end_idx, 'high'].max()
            
            # Tính các mức Fibonacci Retracement (đảo ngược cho sóng đi xuống)
            price_range = high_val - low_val
            for level in fib_levels:
                fib_price = low_val + price_range * level
                # Đánh dấu các mức Fibonacci trong 10 nến sau kết thúc sóng
                end_i = dataframe.index.get_loc(end_idx)
                if end_i + 10 < len(dataframe):
                    dataframe.loc[dataframe.index[end_i:end_i+10], f'fib_retr_{level:.3f}'] = fib_price
            
            # Tính các mức Fibonacci Extension
            for level in fib_ext_levels:
                fib_ext_price = high_val - price_range * level
                # Đánh dấu các mức Fibonacci Extension trong 10 nến sau kết thúc sóng
                dataframe.loc[dataframe.index[end_i:end_i+10], f'fib_ext_{level:.3f}'] = fib_ext_price
    
    return dataframe

def detect_corrective_waves(dataframe, window=5, threshold_pct=1.0):
    """
    Phát hiện các sóng điều chỉnh (corrective waves) theo Elliott Wave Theory
    
    window: số nến nhìn trước/sau để xác định swing
    threshold_pct: ngưỡng % tối thiểu giữa các đỉnh/đáy để được coi là sóng hợp lệ
    """
    # Đánh dấu các sóng điều chỉnh sau một sóng động lực đi lên
    up_impulse_end_indices = dataframe[dataframe['impulse_end'] == 1].index
    
    for end_idx in up_impulse_end_indices:
        end_i = dataframe.index.get_loc(end_idx)
        
        # Xem xét 20 nến sau kết thúc sóng động lực
        if end_i + 20 >= len(dataframe):
            continue
            
        # Tìm swing high và swing low trong phạm vi này
        correction_df = dataframe.iloc[end_i:end_i+20].copy()
        correction_df = find_swings(correction_df, window=window)
        
        # Tính mức cao nhất và thấp nhất trong phạm vi
        high_val = dataframe.loc[end_idx, 'high']  # Mức cao nhất của sóng động lực
        
        # Tìm swing low đầu tiên (sóng A)
        swing_lows = correction_df[correction_df['swing_low'] > 0]
        if len(swing_lows) == 0:
            continue
        
        wave_a_idx = swing_lows.index[0]
        wave_a_val = swing_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng A đủ lớn
        wave_a_change = (high_val - wave_a_val) / high_val * 100
        if wave_a_change < threshold_pct:
            continue
        
        # Tìm swing high kế tiếp (sóng B)
        swing_highs = correction_df[correction_df['swing_high'] > 0]
        swing_highs = swing_highs[swing_highs.index > wave_a_idx]
        if len(swing_highs) == 0:
            continue
        
        wave_b_idx = swing_highs.index[0]
        wave_b_val = swing_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng B không vượt quá mức cao nhất của sóng động lực
        if wave_b_val > high_val:
            continue
        
        # Tìm swing low kế tiếp (sóng C)
        swing_lows = correction_df[correction_df['swing_low'] > 0]
        swing_lows = swing_lows[swing_lows.index > wave_b_idx]
        if len(swing_lows) == 0:
            continue
        
        wave_c_idx = swing_lows.index[0]
        wave_c_val = swing_lows['swing_low'].iloc[0]
        
        # Đánh dấu sóng điều chỉnh
        dataframe.loc[end_idx:wave_c_idx, 'corrective_wave'] = 1
        
        # Đánh dấu các phase ABC
        a_i = dataframe.index.get_loc(wave_a_idx)
        b_i = dataframe.index.get_loc(wave_b_idx)
        c_i = dataframe.index.get_loc(wave_c_idx)
        
        dataframe.loc[dataframe.index[end_i:a_i+1], 'corrective_phase'] = 'A'
        dataframe.loc[dataframe.index[a_i+1:b_i+1], 'corrective_phase'] = 'B'
        dataframe.loc[dataframe.index[b_i+1:c_i+1], 'corrective_phase'] = 'C'
    
    # Đánh dấu các sóng điều chỉnh sau một sóng động lực đi xuống
    down_impulse_end_indices = dataframe[dataframe['impulse_end'] == -1].index
    
    for end_idx in down_impulse_end_indices:
        end_i = dataframe.index.get_loc(end_idx)
        
        # Xem xét 20 nến sau kết thúc sóng động lực
        if end_i + 20 >= len(dataframe):
            continue
            
        # Tìm swing high và swing low trong phạm vi này
        correction_df = dataframe.iloc[end_i:end_i+20].copy()
        correction_df = find_swings(correction_df, window=window)
        
        # Tính mức thấp nhất của sóng động lực
        low_val = dataframe.loc[end_idx, 'low']
        
        # Tìm swing high đầu tiên (sóng A)
        swing_highs = correction_df[correction_df['swing_high'] > 0]
        if len(swing_highs) == 0:
            continue
        
        wave_a_idx = swing_highs.index[0]
        wave_a_val = swing_highs['swing_high'].iloc[0]
        
        # Kiểm tra nếu sóng A đủ lớn
        wave_a_change = (wave_a_val - low_val) / low_val * 100
        if wave_a_change < threshold_pct:
            continue
        
        # Tìm swing low kế tiếp (sóng B)
        swing_lows = correction_df[correction_df['swing_low'] > 0]
        swing_lows = swing_lows[swing_lows.index > wave_a_idx]
        if len(swing_lows) == 0:
            continue
        
        wave_b_idx = swing_lows.index[0]
        wave_b_val = swing_lows['swing_low'].iloc[0]
        
        # Kiểm tra nếu sóng B không thấp hơn mức thấp nhất của sóng động lực
        if wave_b_val < low_val:
            continue
        
        # Tìm swing high kế tiếp (sóng C)
        swing_highs = correction_df[correction_df['swing_high'] > 0]
        swing_highs = swing_highs[swing_highs.index > wave_b_idx]
        if len(swing_highs) == 0:
            continue
        
        wave_c_idx = swing_highs.index[0]
        wave_c_val = swing_highs['swing_high'].iloc[0]
        
        # Đánh dấu sóng điều chỉnh
        dataframe.loc[end_idx:wave_c_idx, 'corrective_wave'] = -1
        
        # Đánh dấu các phase ABC
        a_i = dataframe.index.get_loc(wave_a_idx)
        b_i = dataframe.index.get_loc(wave_b_idx)
        c_i = dataframe.index.get_loc(wave_c_idx)
        
        dataframe.loc[dataframe.index[end_i:a_i+1], 'corrective_phase'] = 'A'
        dataframe.loc[dataframe.index[a_i+1:b_i+1], 'corrective_phase'] = 'B'
        dataframe.loc[dataframe.index[b_i+1:c_i+1], 'corrective_phase'] = 'C'
    
    return dataframe

class Elliot_Waves_Strategy(IStrategy):
    """
    Chiến lược phân tích sóng Elliott kết hợp với Fibonacci và RSI
    
    - Phát hiện các mẫu sóng Elliott (Impulse và Corrective Waves)
    - Sử dụng Fibonacci retracement để xác định các mức hỗ trợ/kháng cự
    - Kết hợp với RSI để xác nhận tín hiệu
    - Tối ưu cho khung thời gian 15m
    """
    
    # Tham số cho Elliot Waves
    swing_window = IntParameter(3, 10, default=5, space="buy", optimize=True)
    impulse_threshold = DecimalParameter(1.0, 5.0, default=2.0, space="buy", optimize=True)
    corrective_threshold = DecimalParameter(0.5, 3.0, default=1.0, space="buy", optimize=True)
    
    # Fibonacci levels
    fib_buy_level = DecimalParameter(0.5, 0.786, default=0.618, space="buy", optimize=True)
    fib_sell_level = DecimalParameter(1.0, 2.0, default=1.618, space="sell", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho EMA
    ema_fast = IntParameter(5, 30, default=8, space="buy", optimize=True)
    ema_slow = IntParameter(15, 50, default=21, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '15m'
    
    minimal_roi = {
        "0": 0.05,     # 5%
        "30": 0.03,    # 3% sau 30 phút
        "60": 0.02,    # 2% sau 60 phút
        "90": 0.01     # 1% sau 90 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%
    trailing_stop_positive_offset = 0.02  # 2%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # RSI
        for period in range(7, 22, 7):  # [7, 14, 21]
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=period
            )
        
        # EMA
        for length in range(5, 51, 5):  # [5, 10, 15, ..., 50]
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'],
                window=length
            )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
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
        
        # Khởi tạo các cột cho Elliot Waves
        dataframe['corrective_wave'] = 0
        dataframe['corrective_phase'] = ''
        
        # Lấy tham số từ config
        swing_window = self.swing_window.value
        impulse_threshold = self.impulse_threshold.value
        corrective_threshold = self.corrective_threshold.value
        
        # Phát hiện các sóng Elliott
        dataframe = detect_impulse_wave(dataframe, window=swing_window, threshold_pct=impulse_threshold)
        dataframe = detect_fibonacci_levels(dataframe, direction='both')
        dataframe = detect_corrective_waves(dataframe, window=swing_window, threshold_pct=corrective_threshold)
        
        # Xác định xu hướng theo EMA
        ema_fast = self.ema_fast.value
        ema_slow = self.ema_slow.value
        
        dataframe['uptrend'] = dataframe[f'ema_{ema_fast}'] > dataframe[f'ema_{ema_slow}']
        dataframe['downtrend'] = dataframe[f'ema_{ema_fast}'] < dataframe[f'ema_{ema_slow}']
        
        # Fibonacci trading setups
        # 1. Bullish setup: end of corrective wave (ABC) với C gần fib_buy_level
        dataframe['fib_buy_setup'] = False
        
        # Tìm điểm kết thúc của sóng điều chỉnh (C) sau sóng đi xuống
        c_end_indices = dataframe[(dataframe['corrective_phase'] == 'C') & 
                                 (dataframe['corrective_wave'] == -1)].index
        
        for idx in c_end_indices:
            # Tìm các mức Fibonacci levels
            fib_level = self.fib_buy_level.value
            fib_col = f'fib_retr_{fib_level:.3f}'
            
            # Nếu đóng cửa gần mức Fibonacci và RSI thấp
            if fib_col in dataframe.columns and dataframe.loc[idx, fib_col] > 0:
                fib_price = dataframe.loc[idx, fib_col]
                
                # Kiểm tra nếu giá gần mức Fibonacci (±2%)
                if (dataframe.loc[idx, 'low'] <= fib_price * 1.02) and (dataframe.loc[idx, 'high'] >= fib_price * 0.98):
                    # Đánh dấu buy setup trong 3 nến
                    idx_i = dataframe.index.get_loc(idx)
                    if idx_i + 3 < len(dataframe):
                        dataframe.loc[dataframe.index[idx_i:idx_i+3], 'fib_buy_setup'] = True
        
        # 2. Bearish setup: end of corrective wave (ABC) với C gần fib_sell_level
        dataframe['fib_sell_setup'] = False
        
        # Tìm điểm kết thúc của sóng điều chỉnh (C) sau sóng đi lên
        c_end_indices = dataframe[(dataframe['corrective_phase'] == 'C') & 
                                 (dataframe['corrective_wave'] == 1)].index
        
        for idx in c_end_indices:
            # Tìm các mức Fibonacci levels
            fib_level = self.fib_sell_level.value
            fib_col = f'fib_ext_{fib_level:.3f}'
            
            # Nếu đóng cửa gần mức Fibonacci và RSI cao
            if fib_col in dataframe.columns and dataframe.loc[idx, fib_col] > 0:
                fib_price = dataframe.loc[idx, fib_col]
                
                # Kiểm tra nếu giá gần mức Fibonacci (±2%)
                if (dataframe.loc[idx, 'low'] <= fib_price * 1.02) and (dataframe.loc[idx, 'high'] >= fib_price * 0.98):
                    # Đánh dấu sell setup trong 3 nến
                    idx_i = dataframe.index.get_loc(idx)
                    if idx_i + 3 < len(dataframe):
                        dataframe.loc[dataframe.index[idx_i:idx_i+3], 'fib_sell_setup'] = True
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Lấy tham số từ config
        rsi_period = self.rsi_period.value
        rsi_buy = self.rsi_buy_threshold.value
        rsi_sell = self.rsi_sell_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Fibonacci buy setup với RSI oversold
        conditions_long.append(
            (dataframe['fib_buy_setup']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy + 5) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. End of impulse wave down (5 waves down), start of corrective wave up
        conditions_long.append(
            (dataframe['impulse_end'] == -1) &
            (dataframe[f'rsi_{rsi_period}'] < 40) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Nến xanh sau kết thúc sóng điều chỉnh ABC sau sóng đi xuống
        conditions_long.append(
            (dataframe['corrective_phase'] == 'C') &
            (dataframe['corrective_wave'] == -1) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe[f'rsi_{rsi_period}'] < 45) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 4. Giá hồi phục đến mức fibonacci 0.618 sau sóng đi xuống
        conditions_long.append(
            (dataframe['fib_retr_0.618'] > 0) &
            (dataframe['low'] <= dataframe['fib_retr_0.618'] * 1.01) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe[f'rsi_{rsi_period}'] < 45) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Start of new impulse wave up after corrective wave
        conditions_long.append(
            (dataframe['impulse_phase'] == 1) &
            (dataframe['impulse_direction'] == 'up') &
            (dataframe[f'rsi_{rsi_period}'] < 60) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Fibonacci sell setup với RSI overbought
        conditions_short.append(
            (dataframe['fib_sell_setup']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell - 5) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. End of impulse wave up (5 waves up), start of corrective wave down
        conditions_short.append(
            (dataframe['impulse_end'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] > 60) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. Nến đỏ sau kết thúc sóng điều chỉnh ABC sau sóng đi lên
        conditions_short.append(
            (dataframe['corrective_phase'] == 'C') &
            (dataframe['corrective_wave'] == 1) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe[f'rsi_{rsi_period}'] > 55) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # 4. Giá hồi phục đến mức fibonacci 0.618 sau sóng đi lên
        conditions_short.append(
            (dataframe['fib_retr_0.618'] > 0) &
            (dataframe['high'] >= dataframe['fib_retr_0.618'] * 0.99) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe[f'rsi_{rsi_period}'] > 55) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Start of new impulse wave down after corrective wave
        conditions_short.append(
            (dataframe['impulse_phase'] == -1) &
            (dataframe['impulse_direction'] == 'down') &
            (dataframe[f'rsi_{rsi_period}'] > 40) &
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
        
        # Lấy tham số từ config
        rsi_period = self.rsi_period.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. RSI overbought
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_period}'] > 75) &
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang giảm
        )
        
        # 2. Đạt mức Fibonacci extension 1.618
        conditions_exit_long.append(
            (dataframe['fib_ext_1.618'] > 0) &
            (dataframe['high'] >= dataframe['fib_ext_1.618'] * 0.995)
        )
        
        # 3. End of impulse wave up (5 waves up)
        conditions_exit_long.append(
            (dataframe['impulse_end'] == 1)
        )
        
        # 4. Giá chạm mức kháng cự Fibonacci (đạt đến 0.786 retracement)
        conditions_exit_long.append(
            (dataframe['fib_retr_0.786'] > 0) &
            (dataframe['high'] >= dataframe['fib_retr_0.786'] * 0.99) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Start of new impulse wave down
        conditions_exit_long.append(
            (dataframe['impulse_phase'] == -1) &
            (dataframe['impulse_direction'] == 'down') &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. RSI oversold
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_period}'] < 25) &
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang tăng
        )
        
        # 2. Đạt mức Fibonacci extension 1.618
        conditions_exit_short.append(
            (dataframe['fib_ext_1.618'] > 0) &
            (dataframe['low'] <= dataframe['fib_ext_1.618'] * 1.005)
        )
        
        # 3. End of impulse wave down (5 waves down)
        conditions_exit_short.append(
            (dataframe['impulse_end'] == -1)
        )
        
        # 4. Giá chạm mức hỗ trợ Fibonacci (đạt đến 0.786 retracement)
        conditions_exit_short.append(
            (dataframe['fib_retr_0.786'] > 0) &
            (dataframe['low'] <= dataframe['fib_retr_0.786'] * 1.01) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. Start of new impulse wave up
        conditions_exit_short.append(
            (dataframe['impulse_phase'] == 1) &
            (dataframe['impulse_direction'] == 'up') &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ rõ ràng của tín hiệu Elliott Waves
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        if side == 'long':
            # End of 5-wave down impulse với RSI thấp
            if last_candle['impulse_end'] == -1 and last_candle[f'rsi_{self.rsi_period.value}'] < 30:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Fib buy setup với RSI thấp
            elif last_candle['fib_buy_setup'] and last_candle[f'rsi_{self.rsi_period.value}'] < 30:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # End of ABC correction sau sóng đi xuống
            elif last_candle['corrective_phase'] == 'C' and last_candle['corrective_wave'] == -1:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Fibonacci level bounce
            elif any(last_candle[f'fib_retr_{level:.3f}'] > 0 for level in [0.618, 0.786]):
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # End of 5-wave up impulse với RSI cao
            if last_candle['impulse_end'] == 1 and last_candle[f'rsi_{self.rsi_period.value}'] > 70:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # Fib sell setup với RSI cao
            elif last_candle['fib_sell_setup'] and last_candle[f'rsi_{self.rsi_period.value}'] > 70:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # End of ABC correction sau sóng đi lên
            elif last_candle['corrective_phase'] == 'C' and last_candle['corrective_wave'] == 1:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # Fibonacci level bounce
            elif any(last_candle[f'fib_retr_{level:.3f}'] > 0 for level in [0.618, 0.786]):
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Elliott Waves và Fibonacci
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại fibonacci level
                for level in [0.382, 0.5, 0.618]:
                    fib_col = f'fib_retr_{level:.3f}'
                    if fib_col in dataframe.columns and last_candle[fib_col] > 0:
                        fib_price = last_candle[fib_col]
                        if fib_price < current_rate:  # Fibonacci level dưới giá hiện tại
                            sl_distance = (fib_price / current_rate) - 1
                            return max(sl_distance, -0.05)  # Không quá -5%
                
                # Nếu không có fib level, sử dụng EMA slow
                ema_slow = last_candle[f'ema_{self.ema_slow.value}']
                if ema_slow < current_rate:
                    sl_distance = (ema_slow / current_rate) - 1
                    return max(sl_distance, -0.05)  # Không quá -5%
            else:
                # Short trade: stoploss tại fibonacci level
                for level in [0.382, 0.5, 0.618]:
                    fib_col = f'fib_retr_{level:.3f}'
                    if fib_col in dataframe.columns and last_candle[fib_col] > 0:
                        fib_price = last_candle[fib_col]
                        if fib_price > current_rate:  # Fibonacci level trên giá hiện tại
                            sl_distance = 1 - (fib_price / current_rate)
                            return max(sl_distance, -0.05)  # Không quá -5%
                
                # Nếu không có fib level, sử dụng EMA slow
                ema_slow = last_candle[f'ema_{self.ema_slow.value}']
                if ema_slow > current_rate:
                    sl_distance = 1 - (ema_slow / current_rate)
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
        
        # Kiểm tra RSI extreme
        rsi = last_candle[f'rsi_{self.rsi_period.value}']
        
        if side == 'long':
            # Không long khi RSI quá cao
            if rsi > 75:
                return False
            
            # Không long khi trong sóng impulse đi xuống tích cực
            if last_candle['impulse_phase'] < 0 and last_candle['impulse_phase'] != -5:
                return False
            
            # Không long khi trong sóng corrective B sau sóng đi lên
            if last_candle['corrective_phase'] == 'B' and last_candle['corrective_wave'] == 1:
                return False
        else:  # short
            # Không short khi RSI quá thấp
            if rsi < 25:
                return False
            
            # Không short khi trong sóng impulse đi lên tích cực
            if last_candle['impulse_phase'] > 0 and last_candle['impulse_phase'] != 5:
                return False
            
            # Không short khi trong sóng corrective B sau sóng đi xuống
            if last_candle['corrective_phase'] == 'B' and last_candle['corrective_wave'] == -1:
                return False
        
        return True
"""
WaveTrend Multi-Timeframe Strategy
Chiến lược sử dụng chỉ báo WaveTrend kết hợp với phân tích đa khung thời gian (MTF)
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

def calculate_wavetrend(dataframe, n1=10, n2=21):
    """
    Tính toán chỉ báo WaveTrend
    
    Tham số:
    - n1: Chu kỳ đầu tiên (mặc định: 10)
    - n2: Chu kỳ thứ hai (mặc định: 21)
    
    Trả về dataframe với các cột mới:
    - wt1: WaveTrend Line
    - wt2: WaveTrend Trigger Line
    - wt_cross_up: WaveTrend cross up signal
    - wt_cross_down: WaveTrend cross down signal
    - wt_overbought: WaveTrend overbought (>70)
    - wt_oversold: WaveTrend oversold (<-70)
    - wt_extreme_overbought: WaveTrend extreme overbought (>100)
    - wt_extreme_oversold: WaveTrend extreme oversold (<-100)
    """
    # Create copy of dataframe
    df = dataframe.copy()
    
    # Calculate HLC3 price
    hlc3 = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate average price (AP)
    esa = ta.trend.ema_indicator(close=hlc3, window=n1)
    
    # Calculate absolute price distance (D)
    d = ta.trend.ema_indicator(close=abs(hlc3 - esa), window=n1)
    
    # Calculate smoothed CI (Choppiness Index)
    ci = (hlc3 - esa) / (0.015 * d)
    
    # Calculate WaveTrend
    wt1 = ta.trend.ema_indicator(close=ci, window=n2)
    wt2 = ta.trend.sma_indicator(close=wt1, window=4)
    
    # Store in dataframe
    df['wt1'] = wt1
    df['wt2'] = wt2
    
    # Calculate cross signals
    df['wt_cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
    df['wt_cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))
    
    # Calculate overbought/oversold conditions
    df['wt_overbought'] = df['wt1'] > 70
    df['wt_oversold'] = df['wt1'] < -70
    df['wt_extreme_overbought'] = df['wt1'] > 100
    df['wt_extreme_oversold'] = df['wt1'] < -100
    
    # Calculate divergences
    # Regular Bullish Divergence (Giá tạo đáy thấp hơn, WT tạo đáy cao hơn)
    df['wt_reg_bull_div'] = False
    
    # Regular Bearish Divergence (Giá tạo đỉnh cao hơn, WT tạo đỉnh thấp hơn)
    df['wt_reg_bear_div'] = False
    
    # Xác định các đáy của WT trong vùng oversold
    df['wt_valley'] = ((df['wt1'] < -60) & 
                      (df['wt1'] < df['wt1'].shift(1)) & 
                      (df['wt1'] < df['wt1'].shift(2)) &
                      (df['wt1'] < df['wt1'].shift(-1)) & 
                      (df['wt1'] < df['wt1'].shift(-2)))
    
    # Xác định các đỉnh của WT trong vùng overbought
    df['wt_peak'] = ((df['wt1'] > 60) & 
                    (df['wt1'] > df['wt1'].shift(1)) & 
                    (df['wt1'] > df['wt1'].shift(2)) &
                    (df['wt1'] > df['wt1'].shift(-1)) & 
                    (df['wt1'] > df['wt1'].shift(-2)))
    
    # Tìm divergence (Scan cửa sổ 5-20 nến)
    for i in range(20, len(df)):
        # Tìm bullish divergence
        if df['wt_valley'].iloc[i]:
            # Tìm đáy trước đó trong vùng oversold
            found_prev_valley = False
            for j in range(i-5, i-20, -1):
                if j > 0 and df['wt_valley'].iloc[j]:
                    # Kiểm tra divergence: giá thấp hơn, wt cao hơn
                    if (df['low'].iloc[i] < df['low'].iloc[j]) and (df['wt1'].iloc[i] > df['wt1'].iloc[j]):
                        df['wt_reg_bull_div'].iloc[i] = True
                        found_prev_valley = True
                        break
            
        # Tìm bearish divergence  
        if df['wt_peak'].iloc[i]:
            # Tìm đỉnh trước đó trong vùng overbought
            found_prev_peak = False
            for j in range(i-5, i-20, -1):
                if j > 0 and df['wt_peak'].iloc[j]:
                    # Kiểm tra divergence: giá cao hơn, wt thấp hơn
                    if (df['high'].iloc[i] > df['high'].iloc[j]) and (df['wt1'].iloc[i] < df['wt1'].iloc[j]):
                        df['wt_reg_bear_div'].iloc[i] = True
                        found_prev_peak = True
                        break
    
    return df

def analyze_higher_timeframe(dataframe, timeframe_higher_data, conversion_factor=4):
    """
    Phân tích dữ liệu khung thời gian cao hơn và map xuống khung thời gian hiện tại
    
    Tham số:
    - dataframe: DataFrame của timeframe hiện tại
    - timeframe_higher_data: DataFrame của timeframe cao hơn
    - conversion_factor: Hệ số chuyển đổi giữa hai timeframe (ví dụ: 4 nếu từ 1h xuống 15m)
    
    Trả về dataframe với các cột từ timeframe cao hơn
    """
    # Check if higher timeframe data is available
    if timeframe_higher_data is None or len(timeframe_higher_data) == 0:
        return dataframe
    
    # Create copy of dataframe
    df = dataframe.copy()
    df_higher = timeframe_higher_data.copy()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('date', inplace=True)
    
    if not isinstance(df_higher.index, pd.DatetimeIndex):
        df_higher.set_index('date', inplace=True)
    
    # Add '_higher' suffix to higher timeframe columns
    df_higher.columns = [f"{col}_higher" for col in df_higher.columns]
    
    # Resample higher timeframe data to match current timeframe
    # We repeat each higher timeframe value for conversion_factor times
    df_resampled = pd.DataFrame(index=df.index)
    
    for col in df_higher.columns:
        # Forward fill to populate higher timeframe data into the current timeframe
        df_resampled[col] = None
        
        # For each row in higher timeframe, map to corresponding rows in lower timeframe
        for i, row in df_higher.iterrows():
            # Find matching rows in lower timeframe
            # For 1h -> 15m, each 1h candle maps to 4 15m candles
            matching_rows = df.index[(df.index >= i) & (df.index < i + pd.Timedelta(minutes=60))]
            if len(matching_rows) > 0:
                df_resampled.loc[matching_rows, col] = row[col]
        
        # Forward fill any missing values
        df_resampled[col] = df_resampled[col].fillna(method='ffill')
    
    # Merge resampled higher timeframe data with current timeframe
    df = pd.concat([df, df_resampled], axis=1)
    
    return df

class WaveTrend_MTF_Strategy(IStrategy):
    """
    Chiến lược sử dụng chỉ báo WaveTrend kết hợp với phân tích đa khung thời gian (MTF)
    
    - Sử dụng WaveTrend để xác định xu hướng và khu vực quá mua/quá bán
    - Kết hợp WaveTrend từ nhiều khung thời gian để có tín hiệu chính xác hơn
    - Phát hiện divergence để xác định đảo chiều tiềm năng
    - Tối ưu cho khung thời gian 1h
    """
    
    # Tham số cho WaveTrend
    wt_n1 = IntParameter(5, 15, default=10, space="buy", optimize=True)
    wt_n2 = IntParameter(15, 30, default=21, space="buy", optimize=True)
    
    # Tham số cho WaveTrend Bands
    wt_overbought = IntParameter(60, 80, default=70, space="sell", optimize=True)
    wt_oversold = IntParameter(-80, -60, default=-70, space="buy", optimize=True)
    
    # Tham số cho Price Action
    ema1 = IntParameter(5, 30, default=20, space="buy", optimize=True)
    ema2 = IntParameter(30, 100, default=50, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.05
    
    # Time-in-force và ROI
    timeframe = '1h'
    timeframe_higher = '4h'
    
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
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_higher) for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Tính WaveTrend cho current timeframe
        wt_df = calculate_wavetrend(
            dataframe,
            n1=self.wt_n1.value,
            n2=self.wt_n2.value
        )
        
        # Thêm các cột WaveTrend
        for column in wt_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = wt_df[column]
        
        # Lấy dữ liệu khung thời gian cao hơn
        higher_tf_df = None
        
        if self.dp:
            informative = self.dp.get_pair_dataframe(
                pair=metadata['pair'],
                timeframe=self.timeframe_higher
            )
            
            if len(informative) > 0:
                # Tính WaveTrend cho khung thời gian cao hơn
                wt_higher_df = calculate_wavetrend(
                    informative,
                    n1=self.wt_n1.value,
                    n2=self.wt_n2.value
                )
                
                # Kết hợp dữ liệu từ khung thời gian cao hơn
                higher_tf_df = analyze_higher_timeframe(
                    dataframe,
                    wt_higher_df,
                    conversion_factor=4  # 4h -> 1h
                )
                
                # Thêm các cột từ khung thời gian cao hơn
                for column in higher_tf_df.columns:
                    if column.endswith('_higher') and column not in dataframe.columns:
                        dataframe[column] = higher_tf_df[column]
        
        # EMAs
        for length in [self.ema1.value, self.ema2.value, 100, 200]:
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'],
                window=length
            )
        
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(
            close=dataframe['close'],
            window=self.rsi_period.value
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
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # WaveTrend Crosses - Kết hợp cả 2 timeframes
        # Lưu ý: cột có hậu tố '_higher' là từ khung thời gian cao hơn
        
        # WaveTrend buy setup (cross up + oversold in higher TF)
        if 'wt_cross_up_higher' in dataframe.columns and 'wt_oversold_higher' in dataframe.columns:
            dataframe['wt_buy_setup'] = (
                (dataframe['wt_cross_up']) &  # Cross up ở timeframe hiện tại
                (dataframe['wt_oversold'].shift(1)) &  # Oversold ở timeframe hiện tại (nến trước)
                (dataframe['wt_oversold_higher'])  # Oversold ở timeframe cao hơn
            )
        else:
            dataframe['wt_buy_setup'] = (
                (dataframe['wt_cross_up']) &  # Cross up ở timeframe hiện tại
                (dataframe['wt_oversold'].shift(1))  # Oversold ở timeframe hiện tại (nến trước)
            )
        
        # WaveTrend sell setup (cross down + overbought in higher TF)
        if 'wt_cross_down_higher' in dataframe.columns and 'wt_overbought_higher' in dataframe.columns:
            dataframe['wt_sell_setup'] = (
                (dataframe['wt_cross_down']) &  # Cross down ở timeframe hiện tại
                (dataframe['wt_overbought'].shift(1)) &  # Overbought ở timeframe hiện tại (nến trước)
                (dataframe['wt_overbought_higher'])  # Overbought ở timeframe cao hơn
            )
        else:
            dataframe['wt_sell_setup'] = (
                (dataframe['wt_cross_down']) &  # Cross down ở timeframe hiện tại
                (dataframe['wt_overbought'].shift(1))  # Overbought ở timeframe hiện tại (nến trước)
            )
        
        # WaveTrend Divergence buy/sell signals
        dataframe['wt_div_buy'] = (
            (dataframe['wt_reg_bull_div']) &  # Bullish divergence
            (dataframe['wt1'] < -50) &  # WT1 trong vùng oversold
            (dataframe['wt1'] > dataframe['wt1'].shift(1))  # WT1 đang tăng
        )
        
        dataframe['wt_div_sell'] = (
            (dataframe['wt_reg_bear_div']) &  # Bearish divergence
            (dataframe['wt1'] > 50) &  # WT1 trong vùng overbought
            (dataframe['wt1'] < dataframe['wt1'].shift(1))  # WT1 đang giảm
        )
        
        # Trend Identification
        dataframe['uptrend'] = dataframe[f'ema_{self.ema1.value}'] > dataframe[f'ema_{self.ema2.value}']
        dataframe['downtrend'] = dataframe[f'ema_{self.ema1.value}'] < dataframe[f'ema_{self.ema2.value}']
        
        # WaveTrend với Price Action
        # Bullish Setup
        dataframe['wt_bullish_setup'] = (
            ((dataframe['wt_buy_setup']) | (dataframe['wt_div_buy'])) &
            (dataframe['close'] > dataframe[f'ema_{self.ema1.value}']) &  # Giá trên EMA ngắn
            (dataframe['rsi'] > 40) &  # RSI không quá thấp
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        # Bearish Setup
        dataframe['wt_bearish_setup'] = (
            ((dataframe['wt_sell_setup']) | (dataframe['wt_div_sell'])) &
            (dataframe['close'] < dataframe[f'ema_{self.ema1.value}']) &  # Giá dưới EMA ngắn
            (dataframe['rsi'] < 60) &  # RSI không quá cao
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. WaveTrend Buy Setup (cross up từ oversold) + xác nhận từ higher timeframe
        conditions_long.append(
            (dataframe['wt_buy_setup']) &
            (dataframe['wt1'] < 0) &  # WaveTrend vẫn dưới 0 (không quá cao)
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. WaveTrend Bullish Divergence
        conditions_long.append(
            (dataframe['wt_div_buy']) &
            (~dataframe['downtrend']) &  # Không trong downtrend mạnh
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. WaveTrend Oversold Bounce
        conditions_long.append(
            (dataframe['wt1'] < self.wt_oversold.value) &
            (dataframe['wt1'] > dataframe['wt1'].shift(1)) &  # WT1 đang tăng
            (dataframe['wt2'] > dataframe['wt2'].shift(1)) &  # WT2 đang tăng
            (dataframe['rsi'] < 40) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. WaveTrend Cross Up khi WT dưới 0 (vẫn trong vùng âm)
        conditions_long.append(
            (dataframe['wt_cross_up']) &
            (dataframe['wt1'] < 0) &
            (dataframe['uptrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. WaveTrend Bullish Setup với giá trên EMA
        conditions_long.append(
            (dataframe['wt_bullish_setup']) &
            (dataframe['close'] > dataframe[f'ema_{self.ema1.value}']) &
            (dataframe['rsi'] > 45) &
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. WaveTrend Sell Setup (cross down từ overbought) + xác nhận từ higher timeframe
        conditions_short.append(
            (dataframe['wt_sell_setup']) &
            (dataframe['wt1'] > 0) &  # WaveTrend vẫn trên 0 (không quá thấp)
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. WaveTrend Bearish Divergence
        conditions_short.append(
            (dataframe['wt_div_sell']) &
            (~dataframe['uptrend']) &  # Không trong uptrend mạnh
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. WaveTrend Overbought Bounce
        conditions_short.append(
            (dataframe['wt1'] > self.wt_overbought.value) &
            (dataframe['wt1'] < dataframe['wt1'].shift(1)) &  # WT1 đang giảm
            (dataframe['wt2'] < dataframe['wt2'].shift(1)) &  # WT2 đang giảm
            (dataframe['rsi'] > 60) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 4. WaveTrend Cross Down khi WT trên 0 (vẫn trong vùng dương)
        conditions_short.append(
            (dataframe['wt_cross_down']) &
            (dataframe['wt1'] > 0) &
            (dataframe['downtrend']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 5. WaveTrend Bearish Setup với giá dưới EMA
        conditions_short.append(
            (dataframe['wt_bearish_setup']) &
            (dataframe['close'] < dataframe[f'ema_{self.ema1.value}']) &
            (dataframe['rsi'] < 55) &
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
        
        # 1. WaveTrend Cross Down từ overbought
        conditions_exit_long.append(
            (dataframe['wt_cross_down']) &
            (dataframe['wt1'].shift(1) > 70)  # WT1 trước đó trong vùng overbought
        )
        
        # 2. WaveTrend Extreme Overbought
        conditions_exit_long.append(
            (dataframe['wt_extreme_overbought']) &
            (dataframe['wt1'] < dataframe['wt1'].shift(1))  # WT1 đang giảm
        )
        
        # 3. WaveTrend + RSI Overbought
        conditions_exit_long.append(
            (dataframe['wt1'] > 70) &
            (dataframe['rsi'] > 70) &
            (dataframe['wt1'] < dataframe['wt1'].shift(1)) &  # WT1 đang giảm
            (dataframe['rsi'] < dataframe['rsi'].shift(1))  # RSI đang giảm
        )
        
        # 4. WaveTrend Bearish Divergence
        conditions_exit_long.append(
            dataframe['wt_div_sell']
        )
        
        # 5. Break down from EMA
        conditions_exit_long.append(
            (dataframe['close'] < dataframe[f'ema_{self.ema1.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema1.value}'].shift(1)) &
            (dataframe['wt1'] < 0)  # WT1 đã vào vùng âm
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. WaveTrend Cross Up từ oversold
        conditions_exit_short.append(
            (dataframe['wt_cross_up']) &
            (dataframe['wt1'].shift(1) < -70)  # WT1 trước đó trong vùng oversold
        )
        
        # 2. WaveTrend Extreme Oversold
        conditions_exit_short.append(
            (dataframe['wt_extreme_oversold']) &
            (dataframe['wt1'] > dataframe['wt1'].shift(1))  # WT1 đang tăng
        )
        
        # 3. WaveTrend + RSI Oversold
        conditions_exit_short.append(
            (dataframe['wt1'] < -70) &
            (dataframe['rsi'] < 30) &
            (dataframe['wt1'] > dataframe['wt1'].shift(1)) &  # WT1 đang tăng
            (dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI đang tăng
        )
        
        # 4. WaveTrend Bullish Divergence
        conditions_exit_short.append(
            dataframe['wt_div_buy']
        )
        
        # 5. Break up from EMA
        conditions_exit_short.append(
            (dataframe['close'] > dataframe[f'ema_{self.ema1.value}']) &
            (dataframe['close'].shift(1) < dataframe[f'ema_{self.ema1.value}'].shift(1)) &
            (dataframe['wt1'] > 0)  # WT1 đã vào vùng dương
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên sức mạnh của tín hiệu WaveTrend
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        if side == 'long':
            # WaveTrend đang trong vùng oversold mạnh
            if last_candle['wt1'] < -90:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # WaveTrend bullish divergence
            elif last_candle['wt_div_buy']:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend buy setup từ oversold
            elif last_candle['wt_buy_setup'] and last_candle['wt1'].shift(1) < -70:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend cross up với xác nhận từ higher timeframe
            elif last_candle['wt_cross_up'] and 'wt1_higher' in last_candle and last_candle['wt1_higher'] < 0:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend đang trong vùng oversold vừa phải
            elif last_candle['wt1'] < -60:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # WaveTrend đang trong vùng overbought mạnh
            if last_candle['wt1'] > 90:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            
            # WaveTrend bearish divergence
            elif last_candle['wt_div_sell']:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend sell setup từ overbought
            elif last_candle['wt_sell_setup'] and last_candle['wt1'].shift(1) > 70:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend cross down với xác nhận từ higher timeframe
            elif last_candle['wt_cross_down'] and 'wt1_higher' in last_candle and last_candle['wt1_higher'] > 0:
                leverage = min(4, max_leverage)  # Đòn bẩy cao
            
            # WaveTrend đang trong vùng overbought vừa phải
            elif last_candle['wt1'] > 60:
                leverage = min(3, max_leverage)  # Đòn bẩy trung bình
            
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên WaveTrend
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade
                
                # Nếu WaveTrend vào vùng overbought, đặt stoploss tại giá hiện tại - 1%
                if last_candle['wt1'] > 70:
                    return -0.01
                
                # Nếu giá trên EMA nhanh, đặt stoploss tại EMA nhanh
                ema_fast = last_candle[f'ema_{self.ema1.value}']
                if current_rate > ema_fast:
                    sl_distance = (ema_fast / current_rate) - 1
                    return max(sl_distance, -0.03)  # Không quá -3%
                
                # Nếu đã có lãi tốt (>5%), bảo toàn lợi nhuận
                if current_profit > 0.05:
                    return -0.01  # Chỉ mất 1% nếu giá quay đầu
                
            else:
                # Short trade
                
                # Nếu WaveTrend vào vùng oversold, đặt stoploss tại giá hiện tại + 1%
                if last_candle['wt1'] < -70:
                    return -0.01
                
                # Nếu giá dưới EMA nhanh, đặt stoploss tại EMA nhanh
                ema_fast = last_candle[f'ema_{self.ema1.value}']
                if current_rate < ema_fast:
                    sl_distance = 1 - (ema_fast / current_rate)
                    return max(sl_distance, -0.03)  # Không quá -3%
                
                # Nếu đã có lãi tốt (>5%), bảo toàn lợi nhuận
                if current_profit > 0.05:
                    return -0.01  # Chỉ mất 1% nếu giá quay đầu
        
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
        rsi = last_candle['rsi']
        if side == 'long' and rsi > 75:  # RSI quá cao cho long
            return False
        if side == 'short' and rsi < 25:  # RSI quá thấp cho short
            return False
        
        # Kiểm tra tín hiệu xung đột từ WaveTrend
        if side == 'long' and last_candle['wt1'] > 80:  # WaveTrend quá cao cho long
            return False
        if side == 'short' and last_candle['wt1'] < -80:  # WaveTrend quá thấp cho short
            return False
        
        # Kiểm tra tín hiệu xung đột từ higher timeframe
        if 'wt1_higher' in last_candle:
            if side == 'long' and last_candle['wt1_higher'] > 70:  # WaveTrend higher TF overbought
                return False
            if side == 'short' and last_candle['wt1_higher'] < -70:  # WaveTrend higher TF oversold
                return False
        
        return True
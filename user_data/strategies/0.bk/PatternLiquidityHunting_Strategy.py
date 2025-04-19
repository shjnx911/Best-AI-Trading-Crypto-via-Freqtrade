"""
Pattern Recognition + Liquidity Hunting Strategy
Chiến lược kết hợp nhận diện mẫu hình nến và săn thanh khoản
Tối ưu cho khung thời gian 5m
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

def detect_candlestick_patterns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Phát hiện các mẫu hình nến Nhật
    """
    # Tạo một bản sao để tránh thay đổi dataframe gốc
    df = dataframe.copy()
    
    # Tính thân nến, bóng trên và bóng dưới
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_pct'] = 100 * df['body'] / (df['high'] - df['low'])
    df['upper_shadow_pct'] = 100 * df['upper_shadow'] / (df['high'] - df['low'])
    df['lower_shadow_pct'] = 100 * df['lower_shadow'] / (df['high'] - df['low'])
    
    # Xác định nến tăng/giảm
    df['bullish_candle'] = df['close'] > df['open']
    df['bearish_candle'] = df['close'] < df['open']
    
    # Hammer (búa) - nến tăng hoặc giảm với bóng dưới dài ít nhất 2x thân
    df['hammer'] = (
        (df['lower_shadow'] > 2 * df['body']) &
        (df['upper_shadow'] < 0.3 * df['body']) &
        (df['body_pct'] < 40) &
        (df['lower_shadow_pct'] > 50)
    )
    
    # Inverted Hammer (búa ngược) - nến tăng hoặc giảm với bóng trên dài ít nhất 2x thân
    df['inverted_hammer'] = (
        (df['upper_shadow'] > 2 * df['body']) &
        (df['lower_shadow'] < 0.3 * df['body']) &
        (df['body_pct'] < 40) &
        (df['upper_shadow_pct'] > 50)
    )
    
    # Bullish Engulfing (nến bao phủ tăng)
    df['bullish_engulfing'] = (
        (df['bullish_candle']) &
        (df['bearish_candle'].shift(1)) &
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )
    
    # Bearish Engulfing (nến bao phủ giảm)
    df['bearish_engulfing'] = (
        (df['bearish_candle']) &
        (df['bullish_candle'].shift(1)) &
        (df['close'] < df['open'].shift(1)) &
        (df['open'] > df['close'].shift(1))
    )
    
    # Morning Star (sao mai)
    df['morning_star'] = (
        (df['bearish_candle'].shift(2)) &
        (df['body'].shift(1) < 0.5 * df['body'].shift(2)) &
        (df['bullish_candle']) &
        (df['body'] > 0.5 * df['body'].shift(2)) &
        (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)
    )
    
    # Evening Star (sao hôm)
    df['evening_star'] = (
        (df['bullish_candle'].shift(2)) &
        (df['body'].shift(1) < 0.5 * df['body'].shift(2)) &
        (df['bearish_candle']) &
        (df['body'] > 0.5 * df['body'].shift(2)) &
        (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)
    )
    
    # Doji (chữ thập)
    df['doji'] = (df['body'] < 0.1 * (df['high'] - df['low']))
    
    # Spinning Top (con quay)
    df['spinning_top'] = (
        (df['body'] < 0.4 * (df['high'] - df['low'])) &
        (df['upper_shadow'] > 0.2 * (df['high'] - df['low'])) &
        (df['lower_shadow'] > 0.2 * (df['high'] - df['low']))
    )
    
    # Three White Soldiers (ba chiến binh trắng)
    df['three_white_soldiers'] = (
        (df['bullish_candle']) &
        (df['bullish_candle'].shift(1)) &
        (df['bullish_candle'].shift(2)) &
        (df['close'] > df['close'].shift(1)) &
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['open'] > df['open'].shift(1)) &
        (df['open'].shift(1) > df['open'].shift(2)) &
        (df['open'] > df['close'].shift(1) * 0.97) &  # Open không quá thấp
        (df['open'].shift(1) > df['close'].shift(2) * 0.97)
    )
    
    # Three Black Crows (ba quạ đen)
    df['three_black_crows'] = (
        (df['bearish_candle']) &
        (df['bearish_candle'].shift(1)) &
        (df['bearish_candle'].shift(2)) &
        (df['close'] < df['close'].shift(1)) &
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['open'] < df['open'].shift(1)) &
        (df['open'].shift(1) < df['open'].shift(2)) &
        (df['open'] < df['close'].shift(1) * 1.03) &  # Open không quá cao
        (df['open'].shift(1) < df['close'].shift(2) * 1.03)
    )
    
    # Piercing Line (đường xuyên thủng)
    df['piercing_line'] = (
        (df['bearish_candle'].shift(1)) &
        (df['bullish_candle']) &
        (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2) &
        (df['close'] < df['open'].shift(1))
    )
    
    # Dark Cloud Cover (mây đen che phủ)
    df['dark_cloud_cover'] = (
        (df['bullish_candle'].shift(1)) &
        (df['bearish_candle']) &
        (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2) &
        (df['close'] > df['close'].shift(1))
    )
    
    # Tweezer Top/Bottom (nhíp trên/dưới)
    df['tweezer_top'] = (
        (df['bullish_candle'].shift(1)) &
        (df['bearish_candle']) &
        (abs(df['high'] - df['high'].shift(1)) / df['high'] < 0.001) &
        (df['close'] < df['close'].shift(1))
    )
    
    df['tweezer_bottom'] = (
        (df['bearish_candle'].shift(1)) &
        (df['bullish_candle']) &
        (abs(df['low'] - df['low'].shift(1)) / df['low'] < 0.001) &
        (df['close'] > df['close'].shift(1))
    )
    
    # Gập (Gap) - khoảng trống
    df['gap_up'] = (df['low'] > df['high'].shift(1) * 1.005)
    df['gap_down'] = (df['high'] < df['low'].shift(1) * 0.995)
    
    return df

def detect_liquidity_hunting(dataframe: pd.DataFrame, atr_multiplier: float = 1.5) -> pd.DataFrame:
    """
    Phát hiện liquidity hunts - những nơi giá đột biến để săn stop loss
    
    - Stop Hunt Up: Giá vượt lên cao và sau đó quay đầu giảm
    - Stop Hunt Down: Giá rớt xuống thấp và sau đó quay đầu tăng
    """
    df = dataframe.copy()
    
    # Tính ATR (Average True Range)
    df['atr'] = ta.volatility.average_true_range(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=14
    )
    
    # Tính Bollinger Bands
    df['upper_band'] = ta.volatility.bollinger_hband(
        close=df['close'],
        window=20,
        window_dev=2
    )
    
    df['lower_band'] = ta.volatility.bollinger_lband(
        close=df['close'],
        window=20,
        window_dev=2
    )
    
    # Mẫu hình Stop Hunt Up (Giá vượt lên cao rồi quay đầu)
    df['stop_hunt_up'] = (
        (df['high'] > df['upper_band']) &  # Vượt upper band
        (df['close'] < df['close'].shift(1)) &  # Giá đóng cửa giảm
        ((df['high'] - df['close']) > atr_multiplier * df['atr']) &  # Bóng trên dài
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.2)  # Volume cao
    )
    
    # Mẫu hình Stop Hunt Down (Giá giảm mạnh rồi quay đầu)
    df['stop_hunt_down'] = (
        (df['low'] < df['lower_band']) &  # Giảm dưới lower band
        (df['close'] > df['close'].shift(1)) &  # Giá đóng cửa tăng
        ((df['close'] - df['low']) > atr_multiplier * df['atr']) &  # Bóng dưới dài
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.2)  # Volume cao
    )
    
    # Phát hiện vùng tập trung lệnh - nơi nhiều stop losses tập trung
    # Thường là nơi giá dao động trong range nhỏ rồi đột ngột breakout
    
    # Tính Bollinger Band Width
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['close']
    
    # Tích lũy trước bùng nổ (vùng nhiều lệnh tích lũy)
    df['consolidation'] = (
        (df['bb_width'] < df['bb_width'].rolling(window=20).mean() * 0.85) &
        (df['volume'] < df['volume'].rolling(window=20).mean() * 0.8)
    )
    
    # Breakout khỏi vùng tích lũy với volume cao
    df['breakout_up'] = (
        df['consolidation'].shift(1) &
        (df['close'] > df['upper_band']) &
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
    )
    
    df['breakout_down'] = (
        df['consolidation'].shift(1) &
        (df['close'] < df['lower_band']) &
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
    )
    
    # Phát hiện Fakeout (giả vỡ)
    df['fakeout_up'] = (
        (df['close'].shift(1) > df['upper_band'].shift(1)) &  # Đóng cửa trên upper band
        (df['close'] < df['upper_band']) &  # Đóng cửa dưới upper band
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.3)  # Volume cao
    )
    
    df['fakeout_down'] = (
        (df['close'].shift(1) < df['lower_band'].shift(1)) &  # Đóng cửa dưới lower band
        (df['close'] > df['lower_band']) &  # Đóng cửa trên lower band
        (df['volume'] > df['volume'].rolling(window=20).mean() * 1.3)  # Volume cao
    )
    
    return df

class PatternLiquidityHunting_Strategy(IStrategy):
    """
    Chiến lược kết hợp nhận diện mẫu hình nến và săn thanh khoản
    
    - Sử dụng các mẫu hình nến Nhật để xác định xu hướng đảo chiều
    - Kết hợp săn Liquidity Hunts - tín hiệu mà large players/bots săn lệnh Stop Loss
    - Tìm các vùng thanh khoản cao để chờ phá vỡ
    - Tối ưu cho giao dịch tần suất cao (5m)
    """
    
    # Tham số cho nhận diện mẫu hình
    candle_low_high_ratio = DecimalParameter(1.5, 4.0, default=2.0, space="buy", optimize=True)
    candle_body_pct = DecimalParameter(10.0, 50.0, default=30.0, space="buy", optimize=True)
    
    # Tham số cho liquidity hunting
    atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    volume_surge_threshold = DecimalParameter(1.2, 2.5, default=1.5, space="buy", optimize=True)
    consolidation_threshold = DecimalParameter(0.7, 0.9, default=0.8, space="buy", optimize=True)
    
    # Tham số cho trend
    ema_short = IntParameter(5, 15, default=8, space="buy", optimize=True)
    ema_long = IntParameter(20, 50, default=26, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 8
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '5m'
    
    minimal_roi = {
        "0": 0.03,     # 3%
        "15": 0.02,    # 2% sau 15 phút
        "30": 0.01,    # 1% sau 30 phút
        "60": 0.005    # 0.5% sau 60 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%
    trailing_stop_positive_offset = 0.015  # 1.5%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Nhận diện mẫu hình nến
        pattern_df = detect_candlestick_patterns(dataframe)
        for column in pattern_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = pattern_df[column]
        
        # Phát hiện liquidity hunts
        hunt_df = detect_liquidity_hunting(dataframe, atr_multiplier=self.atr_multiplier.value)
        for column in hunt_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = hunt_df[column]
        
        # Thêm EMA
        for length in range(5, 51, 5):
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(
                close=dataframe['close'],
                window=length
            )
        
        # Thêm RSI
        for length in range(7, 22):
            dataframe[f'rsi_{length}'] = ta.momentum.rsi(
                close=dataframe['close'],
                window=length
            )
        
        # Chỉ báo Bollinger Bands
        bollinger = ta.volatility.BollingerBands(
            close=dataframe['close'],
            window=20,
            window_dev=2
        )
        dataframe['bb_middle'] = bollinger.bollinger_mavg()
        dataframe['bb_upper'] = bollinger.bollinger_hband()
        dataframe['bb_lower'] = bollinger.bollinger_lband()
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        
        # Chỉ báo Stochastic
        stoch = ta.momentum.StochasticOscillator(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            window=14,
            smooth_window=3
        )
        dataframe['stoch_k'] = stoch.stoch()
        dataframe['stoch_d'] = stoch.stoch_signal()
        
        # Xác định xu hướng tổng thể
        ema_short = self.ema_short.value
        ema_long = self.ema_long.value
        
        dataframe['uptrend'] = dataframe[f'ema_{ema_short}'] > dataframe[f'ema_{ema_long}']
        dataframe['downtrend'] = dataframe[f'ema_{ema_short}'] < dataframe[f'ema_{ema_long}']
        
        # Xác định điểm lý tưởng để vào lệnh
        
        # 1. Mẫu hình nến đảo chiều bullish
        dataframe['reversal_bullish'] = (
            dataframe['hammer'] |
            dataframe['bullish_engulfing'] |
            dataframe['morning_star'] |
            dataframe['piercing_line'] |
            dataframe['tweezer_bottom'] |
            dataframe['three_white_soldiers']
        )
        
        # 2. Mẫu hình nến đảo chiều bearish
        dataframe['reversal_bearish'] = (
            dataframe['inverted_hammer'] |
            dataframe['bearish_engulfing'] |
            dataframe['evening_star'] |
            dataframe['dark_cloud_cover'] |
            dataframe['tweezer_top'] |
            dataframe['three_black_crows']
        )
        
        # 3. Điểm săn thanh khoản trong xu hướng
        dataframe['liquidity_signal_long'] = (
            (dataframe['stop_hunt_down'] | dataframe['fakeout_down']) &
            (~dataframe['downtrend'])
        )
        
        dataframe['liquidity_signal_short'] = (
            (dataframe['stop_hunt_up'] | dataframe['fakeout_up']) &
            (~dataframe['uptrend'])
        )
        
        # 4. Breakout từ vùng tích lũy
        dataframe['breakout_signal_long'] = (
            dataframe['breakout_up'] &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 70)
        )
        
        dataframe['breakout_signal_short'] = (
            dataframe['breakout_down'] &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 30)
        )
        
        # 5. Tổng hợp tất cả các tín hiệu long/short
        dataframe['combined_buy_signal'] = (
            (dataframe['reversal_bullish'] & (dataframe[f'rsi_{self.rsi_period.value}'] < self.rsi_buy.value)) |
            (dataframe['liquidity_signal_long']) |
            (dataframe['breakout_signal_long'])
        )
        
        dataframe['combined_sell_signal'] = (
            (dataframe['reversal_bearish'] & (dataframe[f'rsi_{self.rsi_period.value}'] > self.rsi_sell.value)) |
            (dataframe['liquidity_signal_short']) |
            (dataframe['breakout_signal_short'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh Long
        conditions_long = []
        
        # 1. Mẫu hình đảo chiều bullish + RSI thấp + Volume cao
        conditions_long.append(
            (dataframe['reversal_bullish']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < self.rsi_buy.value) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * 1.2)
        )
        
        # 2. Stop Hunt Down (săn thanh khoản dưới và bật lên)
        conditions_long.append(
            (dataframe['stop_hunt_down'] | dataframe['fakeout_down']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * self.volume_surge_threshold.value) &
            (~dataframe['downtrend'])
        )
        
        # 3. Breakout lên từ vùng tích lũy
        conditions_long.append(
            (dataframe['breakout_up']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 70) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * self.volume_surge_threshold.value)
        )
        
        # 4. Hammer cạnh vùng hỗ trợ
        conditions_long.append(
            (dataframe['hammer']) &
            (dataframe['close'] < dataframe['bb_lower'] * 1.01) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 40) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())
        )
        
        # 5. Gap up với volume cao
        conditions_long.append(
            (dataframe['gap_up']) &
            (dataframe['uptrend']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * 1.5)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh Short
        conditions_short = []
        
        # 1. Mẫu hình đảo chiều bearish + RSI cao + Volume cao
        conditions_short.append(
            (dataframe['reversal_bearish']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > self.rsi_sell.value) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * 1.2)
        )
        
        # 2. Stop Hunt Up (săn thanh khoản trên và đi xuống)
        conditions_short.append(
            (dataframe['stop_hunt_up'] | dataframe['fakeout_up']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * self.volume_surge_threshold.value) &
            (~dataframe['uptrend'])
        )
        
        # 3. Breakout xuống từ vùng tích lũy
        conditions_short.append(
            (dataframe['breakout_down']) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 30) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * self.volume_surge_threshold.value)
        )
        
        # 4. Inverted Hammer cạnh vùng kháng cự
        conditions_short.append(
            (dataframe['inverted_hammer']) &
            (dataframe['close'] > dataframe['bb_upper'] * 0.99) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 60) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean())
        )
        
        # 5. Gap down với volume cao
        conditions_short.append(
            (dataframe['gap_down']) &
            (dataframe['downtrend']) &
            (dataframe['volume'] > dataframe['volume'].rolling(window=20).mean() * 1.5)
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Điều kiện thoát Long
        conditions_exit_long = []
        
        # 1. Mẫu hình đảo chiều bearish
        conditions_exit_long.append(
            dataframe['reversal_bearish']
        )
        
        # 2. Đụng Bollinger Band trên + RSI cao
        conditions_exit_long.append(
            (dataframe['close'] > dataframe['bb_upper'] * 0.98) &
            (dataframe[f'rsi_{self.rsi_period.value}'] > 70)
        )
        
        # 3. Stop hunt up xuất hiện
        conditions_exit_long.append(
            dataframe['stop_hunt_up']
        )
        
        # 4. EMA cross down
        conditions_exit_long.append(
            (dataframe[f'ema_{self.ema_short.value}'] < dataframe[f'ema_{self.ema_long.value}']) &
            (dataframe[f'ema_{self.ema_short.value}'].shift(1) > dataframe[f'ema_{self.ema_long.value}'].shift(1))
        )
        
        # 5. Stochastic cross down từ vùng overbought
        conditions_exit_long.append(
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) > dataframe['stoch_d'].shift(1)) &
            (dataframe['stoch_k'].shift(1) > 80)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát Short
        conditions_exit_short = []
        
        # 1. Mẫu hình đảo chiều bullish
        conditions_exit_short.append(
            dataframe['reversal_bullish']
        )
        
        # 2. Đụng Bollinger Band dưới + RSI thấp
        conditions_exit_short.append(
            (dataframe['close'] < dataframe['bb_lower'] * 1.02) &
            (dataframe[f'rsi_{self.rsi_period.value}'] < 30)
        )
        
        # 3. Stop hunt down xuất hiện
        conditions_exit_short.append(
            dataframe['stop_hunt_down']
        )
        
        # 4. EMA cross up
        conditions_exit_short.append(
            (dataframe[f'ema_{self.ema_short.value}'] > dataframe[f'ema_{self.ema_long.value}']) &
            (dataframe[f'ema_{self.ema_short.value}'].shift(1) < dataframe[f'ema_{self.ema_long.value}'].shift(1))
        )
        
        # 5. Stochastic cross up từ vùng oversold
        conditions_exit_short.append(
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) < dataframe['stoch_d'].shift(1)) &
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
        Điều chỉnh đòn bẩy dựa trên mẫu hình nến và chất lượng của tín hiệu
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Đánh giá mức độ rõ ràng của mẫu hình
        clear_pattern = False
        high_volume = False
        strong_signal = False
        
        # Kiểm tra tín hiệu rõ ràng và khối lượng cao
        if side == 'long':
            # Mẫu hình rõ ràng
            if last_candle['reversal_bullish'] or last_candle['hammer'] or last_candle['morning_star']:
                clear_pattern = True
            
            # Stop hunt down hoặc fakeout rõ ràng
            if last_candle['stop_hunt_down'] or (last_candle['fakeout_down'] and last_candle['close'] > last_candle['bb_lower'] * 1.05):
                strong_signal = True
                
            # Khối lượng cao
            if last_candle['volume'] > last_candle['volume'].rolling(window=20).mean() * 2.0:
                high_volume = True
                
        else:  # side == 'short'
            # Mẫu hình rõ ràng
            if last_candle['reversal_bearish'] or last_candle['inverted_hammer'] or last_candle['evening_star']:
                clear_pattern = True
            
            # Stop hunt up hoặc fakeout rõ ràng
            if last_candle['stop_hunt_up'] or (last_candle['fakeout_up'] and last_candle['close'] < last_candle['bb_upper'] * 0.95):
                strong_signal = True
                
            # Khối lượng cao
            if last_candle['volume'] > last_candle['volume'].rolling(window=20).mean() * 2.0:
                high_volume = True
        
        # Tính toán đòn bẩy dựa trên chất lượng tín hiệu
        if clear_pattern and high_volume and strong_signal:
            leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
        elif (clear_pattern and high_volume) or (high_volume and strong_signal) or (clear_pattern and strong_signal):
            leverage = min(6, max_leverage)  # Đòn bẩy cao
        elif clear_pattern or high_volume or strong_signal:
            leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
        else:
            leverage = self.leverage_default  # Đòn bẩy mặc định
            
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên mẫu hình nến và Bollinger Bands
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận dương, sử dụng Bollinger Band làm stoploss
        if current_profit > 0.015:  # > 1.5% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại middle band
                bb_middle = last_candle['bb_middle']
                sl_distance = (bb_middle / current_rate) - 1
                return max(sl_distance, -0.03)  # Không ít hơn -3%
            else:
                # Short trade: stoploss tại middle band
                bb_middle = last_candle['bb_middle']
                sl_distance = 1 - (bb_middle / current_rate)
                return max(sl_distance, -0.03)  # Không ít hơn -3%
        
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
        
        # Không vào lệnh khi BB Width quá nhỏ (volatility quá thấp)
        if last_candle['bb_width'] < 0.03:
            return False
        
        # Không vào lệnh khi volume quá thấp
        if last_candle['volume'] < last_candle['volume'].rolling(window=20).mean() * 0.7:
            return False
        
        # Không vào lệnh long khi RSI quá cao
        if side == 'long' and last_candle[f'rsi_{self.rsi_period.value}'] > 75:
            return False
            
        # Không vào lệnh short khi RSI quá thấp
        if side == 'short' and last_candle[f'rsi_{self.rsi_period.value}'] < 25:
            return False
        
        return True
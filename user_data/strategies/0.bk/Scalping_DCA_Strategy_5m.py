"""
Scalping DCA Strategy (5m)
Chiến lược scalping kết hợp DCA thông minh - tối ưu cho khung thời gian 5m
Tập trung vào giao dịch tần suất cao với mục tiêu lợi nhuận nhỏ (0.5-1%)
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

def detect_scalping_opportunity(dataframe, window_short=5, window_medium=14, window_long=30):
    """
    Phát hiện cơ hội scalping trong khung thời gian ngắn
    
    Tham số:
    - window_short: Cửa sổ ngắn cho các chỉ báo nhanh
    - window_medium: Cửa sổ trung bình
    - window_long: Cửa sổ dài cho trend
    
    Trả về dataframe với các thông tin:
    - scalping_long_signal: Tín hiệu scalping long
    - scalping_short_signal: Tín hiệu scalping short
    - scalping_dca_suitable: Thích hợp cho DCA trong scalping không
    """
    df = dataframe.copy()
    
    # EMAs
    for period in [3, 5, 8, 13, 21, 50, 100]:
        df[f'ema_{period}'] = ta.trend.ema_indicator(close=df['close'], window=period)
    
    # RSI
    df['rsi'] = ta.momentum.rsi(close=df['close'], window=window_medium)
    df['rsi_fast'] = ta.momentum.rsi(close=df['close'], window=window_short)
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(
        high=df['high'], 
        low=df['low'], 
        close=df['close'],
        window=window_medium,
        smooth_window=3
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=df['close'], 
        window=window_medium, 
        window_dev=2
    )
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume
    df['volume_mean'] = df['volume'].rolling(window=window_medium).mean()
    df['volume_ratio'] = df['volume'] / df['volume_mean']
    
    # Tính chỉ báo momentum và volatility
    df['close_change_pct'] = df['close'].pct_change() * 100
    df['volatility'] = df['close_change_pct'].rolling(window=window_medium).std()
    
    # MACD
    macd = ta.trend.MACD(
        close=df['close'],
        window_slow=26,
        window_fast=12,
        window_sign=9
    )
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Xác định Market Type - Đặc biệt quan trọng cho scalping
    # 1. Range-bound (Sideway)
    df['range_bound'] = ((df['bb_width'] < 0.05) & 
                         (df['volatility'] < df['volatility'].rolling(window=window_long).mean() * 0.8) &
                         (abs(df['ema_8'] - df['ema_21']) / df['close'] < 0.005))
    
    # 2. Trending
    df['uptrend'] = ((df['ema_5'] > df['ema_13']) & 
                     (df['ema_8'] > df['ema_21']) & 
                     (df['close'] > df['ema_8']))
    
    df['downtrend'] = ((df['ema_5'] < df['ema_13']) & 
                        (df['ema_8'] < df['ema_21']) & 
                        (df['close'] < df['ema_8']))
    
    # 3. Volatile/Choppy
    df['choppy'] = ((df['volatility'] > df['volatility'].rolling(window=window_long).mean() * 1.2) &
                    (~df['range_bound']) & 
                    (~df['uptrend']) & 
                    (~df['downtrend']))
    
    # Xác định market type tổng hợp
    conditions = [
        df['range_bound'],
        df['uptrend'],
        df['downtrend'],
        df['choppy']
    ]
    choices = ['range_bound', 'uptrend', 'downtrend', 'choppy']
    df['market_type'] = np.select(conditions, choices, default='uncertain')
    
    # Tín hiệu Scalping Long
    df['scalping_long_signal'] = (
        # Điều kiện price action
        (df['close'] > df['open']) &  # Nến xanh
        (df['close'] > df['close'].shift(1)) &  # Giá đang tăng
        
        # Điều kiện kỹ thuật
        (
            # Trường hợp 1: Bounce từ SMA ở uptrend
            ((df['uptrend']) & (df['low'] <= df['ema_21']) & (df['close'] > df['ema_21'])) |
            
            # Trường hợp 2: Breakout khỏi range
            ((df['range_bound'].shift(1)) & (df['close'] > df['bb_upper'].shift(1)) & (df['volume_ratio'] > 1.2)) |
            
            # Trường hợp 3: RSI oversold bounce
            ((df['rsi_fast'] < 30) & (df['rsi_fast'] > df['rsi_fast'].shift(1)) & (df['close'] > df['open'])) |
            
            # Trường hợp 4: Stochastic oversold cross
            ((df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1)))
        )
    )
    
    # Tín hiệu Scalping Short
    df['scalping_short_signal'] = (
        # Điều kiện price action
        (df['close'] < df['open']) &  # Nến đỏ
        (df['close'] < df['close'].shift(1)) &  # Giá đang giảm
        
        # Điều kiện kỹ thuật
        (
            # Trường hợp 1: Reject từ SMA ở downtrend
            ((df['downtrend']) & (df['high'] >= df['ema_21']) & (df['close'] < df['ema_21'])) |
            
            # Trường hợp 2: Breakdown khỏi range
            ((df['range_bound'].shift(1)) & (df['close'] < df['bb_lower'].shift(1)) & (df['volume_ratio'] > 1.2)) |
            
            # Trường hợp 3: RSI overbought reversal
            ((df['rsi_fast'] > 70) & (df['rsi_fast'] < df['rsi_fast'].shift(1)) & (df['close'] < df['open'])) |
            
            # Trường hợp 4: Stochastic overbought cross
            ((df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1)))
        )
    )
    
    # Xác định DCA suitability - trong scalping, DCA chỉ phù hợp trong một số trường hợp cụ thể
    df['scalping_dca_suitable'] = (
        # Phù hợp DCA trong range-bound và downtrend
        (df['market_type'].isin(['range_bound', 'downtrend'])) &
        
        # Không phù hợp trong thị trường choppy
        (~df['choppy']) &
        
        # Chỉ phù hợp khi volatility không quá cao
        (df['volatility'] < df['volatility'].rolling(window=window_long).mean() * 1.2)
    )
    
    # Tính toán potential profit target cho scalping
    # Trong 5m, mục tiêu lợi nhuận thường nhỏ
    df['scalping_profit_target'] = np.where(
        df['market_type'] == 'range_bound',
        0.5,  # 0.5% trong sideway
        np.where(
            df['market_type'].isin(['uptrend', 'downtrend']),
            0.8,  # 0.8% trong trend
            0.3   # 0.3% trong uncertain/choppy
        )
    )
    
    # Tính DCA levels cho scalping - thường nhỏ hơn các khung thời gian khác
    df['dca_level_1_pct'] = np.where(
        df['market_type'] == 'range_bound',
        -0.3,  # -0.3% trong sideway
        np.where(
            df['market_type'] == 'downtrend',
            -0.5,  # -0.5% trong downtrend
            -0.4   # -0.4% trong các trường hợp khác
        )
    )
    
    df['dca_level_2_pct'] = df['dca_level_1_pct'] * 2  # Gấp đôi level 1
    df['dca_level_3_pct'] = df['dca_level_1_pct'] * 3  # Gấp ba level 1
    
    # DCA allocation - tùy thuộc vào market type
    df['dca_initial_allocation'] = np.where(
        df['market_type'] == 'range_bound',
        0.4,  # 40% ban đầu trong sideway
        np.where(
            df['market_type'] == 'downtrend',
            0.3,  # 30% ban đầu trong downtrend
            0.5   # 50% ban đầu trong các trường hợp khác
        )
    )
    
    # Stop loss levels - phụ thuộc vào volatility và market type
    df['initial_stoploss_pct'] = np.where(
        df['market_type'] == 'range_bound',
        1.0,  # 1% stoploss trong sideway
        np.where(
            df['market_type'] == 'choppy',
            2.0,  # 2% stoploss trong choppy
            1.5   # 1.5% stoploss trong các trường hợp khác
        )
    )
    
    # Tăng stoploss sau mỗi lần DCA
    df['stoploss_after_dca1_pct'] = df['initial_stoploss_pct'] * 1.5
    df['stoploss_after_dca2_pct'] = df['initial_stoploss_pct'] * 2.0
    df['stoploss_after_dca3_pct'] = df['initial_stoploss_pct'] * 2.5
    
    return df

class Scalping_DCA_Strategy_5m(IStrategy):
    """
    Chiến lược Scalping kết hợp DCA thông minh
    
    - Tối ưu cho khung thời gian 5m
    - Tập trung vào giao dịch tần suất cao với mục tiêu lợi nhuận nhỏ (0.5-1%)
    - Sử dụng DCA chọn lọc trong các điều kiện thị trường phù hợp
    - Đặc biệt hiệu quả trong thị trường sideway (range-bound)
    """
    
    # Tham số cho chiến lược DCA
    use_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    max_dca_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    max_dca_orders = IntParameter(1, 3, default=3, space="buy", optimize=True)
    
    # Tham số scalping
    ema_short = IntParameter(3, 8, default=5, space="buy", optimize=True)
    ema_medium = IntParameter(8, 21, default=13, space="buy", optimize=True)
    
    # Tham số RSI
    rsi_period = IntParameter(4, 14, default=6, space="buy", optimize=True)
    rsi_oversold = IntParameter(10, 30, default=20, space="buy", optimize=True)
    rsi_overbought = IntParameter(70, 90, default=80, space="sell", optimize=True)
    
    # Tham số stochastic
    stoch_k_period = IntParameter(4, 14, default=9, space="buy", optimize=True)
    stoch_oversold = IntParameter(10, 30, default=20, space="buy", optimize=True)
    stoch_overbought = IntParameter(70, 90, default=80, space="sell", optimize=True)
    
    # Tham số Bollinger
    bb_period = IntParameter(10, 20, default=14, space="buy", optimize=True)
    bb_std = DecimalParameter(1.5, 2.5, default=2.0, space="buy", optimize=True)
    
    # Tham số profit-taking
    profit_target_1 = DecimalParameter(0.3, 1.0, default=0.5, space="sell", optimize=True)
    profit_target_2 = DecimalParameter(0.5, 2.0, default=1.0, space="sell", optimize=True)
    
    # Tham số risk management
    initial_stoploss = DecimalParameter(0.5, 2.0, default=1.0, space="sell", optimize=True)
    trailing_stop_positive_offset = DecimalParameter(0.1, 0.5, default=0.3, space="sell", optimize=True)
    
    # Core parameters
    timeframe = '5m'
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.01  # -1% stoploss mặc định, sẽ được điều chỉnh theo DCA
    
    # ROI mục tiêu - cho scalping, ROI timeline nên rất ngắn
    minimal_roi = {
        "0": 0.01,     # 1%
        "5": 0.007,    # 0.7% sau 5 phút
        "10": 0.005,   # 0.5% sau 10 phút
        "20": 0.003    # 0.3% sau 20 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.002  # 0.2%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Phát hiện cơ hội scalping
        scalping_df = detect_scalping_opportunity(
            dataframe,
            window_short=self.rsi_period.value,
            window_medium=self.bb_period.value
        )
        
        # Thêm các cột scalping
        for column in scalping_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = scalping_df[column]
        
        # Thêm các chỉ báo phổ biến nếu chưa có trong scalping_df
        if 'rsi' not in dataframe.columns:
            dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=self.rsi_period.value)
        
        # Volume OBV
        dataframe['obv'] = ta.volume.on_balance_volume(dataframe['close'], dataframe['volume'])
        dataframe['obv_ma'] = ta.trend.sma_indicator(dataframe['obv'], window=10)
        
        # Phát hiện pinbar - đặc biệt quan trọng cho scalping
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['wick_upper'] = dataframe['high'] - dataframe[['close', 'open']].max(axis=1)
        dataframe['wick_lower'] = dataframe[['close', 'open']].min(axis=1) - dataframe['low']
        
        # Định nghĩa pinbar cho scalping
        dataframe['bullish_pinbar'] = (
            (dataframe['body_size'] < dataframe['body_size'].rolling(10).mean() * 0.8) &
            (dataframe['wick_lower'] > dataframe['body_size'] * 2) &
            (dataframe['wick_lower'] > dataframe['wick_upper']) &
            (dataframe['close'] > dataframe['open'])  # Nến xanh
        )
        
        dataframe['bearish_pinbar'] = (
            (dataframe['body_size'] < dataframe['body_size'].rolling(10).mean() * 0.8) &
            (dataframe['wick_upper'] > dataframe['body_size'] * 2) &
            (dataframe['wick_upper'] > dataframe['wick_lower']) &
            (dataframe['close'] < dataframe['open'])  # Nến đỏ
        )
        
        # Thêm chỉ báo momentum cho scalping nhanh
        dataframe['close_change_1'] = dataframe['close'].pct_change(1)
        dataframe['close_change_2'] = dataframe['close'].pct_change(2)
        
        dataframe['acceleration'] = dataframe['close_change_1'] - dataframe['close_change_2']
        
        # Tạo composite signal
        dataframe['long_signal_strength'] = 0
        dataframe['short_signal_strength'] = 0
        
        # Bullish signals
        dataframe.loc[dataframe['scalping_long_signal'], 'long_signal_strength'] += 1
        dataframe.loc[dataframe['bullish_pinbar'], 'long_signal_strength'] += 1
        dataframe.loc[dataframe['rsi'] < self.rsi_oversold.value, 'long_signal_strength'] += 1
        dataframe.loc[dataframe['stoch_k'] < self.stoch_oversold.value, 'long_signal_strength'] += 1
        dataframe.loc[dataframe['obv'] > dataframe['obv_ma'], 'long_signal_strength'] += 0.5
        
        # Bearish signals
        dataframe.loc[dataframe['scalping_short_signal'], 'short_signal_strength'] += 1
        dataframe.loc[dataframe['bearish_pinbar'], 'short_signal_strength'] += 1
        dataframe.loc[dataframe['rsi'] > self.rsi_overbought.value, 'short_signal_strength'] += 1
        dataframe.loc[dataframe['stoch_k'] > self.stoch_overbought.value, 'short_signal_strength'] += 1
        dataframe.loc[dataframe['obv'] < dataframe['obv_ma'], 'short_signal_strength'] += 0.5
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Strong scalping signal
        conditions_long.append(
            (dataframe['long_signal_strength'] >= 2) &
            (dataframe['scalping_long_signal']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Bounce from support in range-bound market
        conditions_long.append(
            (dataframe['market_type'] == 'range_bound') &
            (dataframe['bb_percent'] < 0.2) &  # Gần BB lower band
            (dataframe['rsi'] < 40) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. RSI oversold bounce
        conditions_long.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI đang tăng
            (dataframe['close'] > dataframe['open']) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['volume_ratio'] > 0.7)
        )
        
        # 4. Bullish Pinbar
        conditions_long.append(
            (dataframe['bullish_pinbar']) &
            (dataframe['market_type'].isin(['range_bound', 'uptrend'])) &
            (dataframe['volume_ratio'] > 0.7)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Strong scalping signal
        conditions_short.append(
            (dataframe['short_signal_strength'] >= 2) &
            (dataframe['scalping_short_signal']) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Rejection from resistance in range-bound market
        conditions_short.append(
            (dataframe['market_type'] == 'range_bound') &
            (dataframe['bb_percent'] > 0.8) &  # Gần BB upper band
            (dataframe['rsi'] > 60) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 3. RSI overbought reversal
        conditions_short.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI đang giảm
            (dataframe['close'] < dataframe['open']) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['volume_ratio'] > 0.7)
        )
        
        # 4. Bearish Pinbar
        conditions_short.append(
            (dataframe['bearish_pinbar']) &
            (dataframe['market_type'].isin(['range_bound', 'downtrend'])) &
            (dataframe['volume_ratio'] > 0.7)
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
        
        # 1. Take profit at target - Scalping nên chốt lời sớm
        conditions_exit_long.append(
            (dataframe['close'] > dataframe['open'] * (1 + self.profit_target_1.value/100)) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI overbought
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1))  # RSI đang giảm
        )
        
        # 3. Price reaches upper band
        conditions_exit_long.append(
            (dataframe['close'] > dataframe['bb_upper']) &
            (dataframe['close'] < dataframe['open'])  # Nến đỏ
        )
        
        # 4. Stochastic overbought cross down
        conditions_exit_long.append(
            (dataframe['stoch_k'] > self.stoch_overbought.value) &
            (dataframe['stoch_k'] < dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) >= dataframe['stoch_d'].shift(1))
        )
        
        # 5. Price drops below entry EMA in uptrend
        conditions_exit_long.append(
            (dataframe['market_type'] == 'uptrend') &
            (dataframe['close'] < dataframe[f'ema_{self.ema_short.value}']) &
            (dataframe['close'].shift(1) > dataframe[f'ema_{self.ema_short.value}'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Take profit at target
        conditions_exit_short.append(
            (dataframe['close'] < dataframe['open'] * (1 - self.profit_target_1.value/100)) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. RSI oversold
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1))  # RSI đang tăng
        )
        
        # 3. Price reaches lower band
        conditions_exit_short.append(
            (dataframe['close'] < dataframe['bb_lower']) &
            (dataframe['close'] > dataframe['open'])  # Nến xanh
        )
        
        # 4. Stochastic oversold cross up
        conditions_exit_short.append(
            (dataframe['stoch_k'] < self.stoch_oversold.value) &
            (dataframe['stoch_k'] > dataframe['stoch_d']) &
            (dataframe['stoch_k'].shift(1) <= dataframe['stoch_d'].shift(1))
        )
        
        # 5. Price rises above entry EMA in downtrend
        conditions_exit_short.append(
            (dataframe['market_type'] == 'downtrend') &
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
        
        # Scalping thích hợp cho cả long và short DCA
        
        # Get DCA information
        dca_info = {}
        
        # Tính số lần đã DCA
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Đã đạt số lần DCA tối đa
        if dca_count >= self.max_dca_orders.value:
            return None
        
        # Kiểm tra xem thị trường có phù hợp cho DCA không
        if not last_candle['scalping_dca_suitable']:
            return None
        
        # Calculate units to buy for DCA
        stake_amount = trade.stake_amount
        dca_stake = 0.0
        
        # DCA parameters for scalping need to be more responsive
        current_dca_level = dca_count + 1
        
        # Xác định mức giá và stake amount cho DCA tiếp theo
        if current_dca_level == 1:
            # Sử dụng mức DCA được tính toán trong indicator
            dca_threshold = last_candle['dca_level_1_pct']
            
            # Allocation tùy thuộc vào loại market
            alloc_ratio = (1 - last_candle['dca_initial_allocation']) / self.max_dca_orders.value
            dca_stake = stake_amount * self.max_dca_multiplier.value * alloc_ratio / last_candle['dca_initial_allocation']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
                
        elif current_dca_level == 2:
            dca_threshold = last_candle['dca_level_2_pct']
            
            # Allocation tùy thuộc vào loại market
            alloc_ratio = (1 - last_candle['dca_initial_allocation']) / self.max_dca_orders.value
            dca_stake = stake_amount * self.max_dca_multiplier.value * alloc_ratio / last_candle['dca_initial_allocation']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
                
        elif current_dca_level == 3:
            dca_threshold = last_candle['dca_level_3_pct']
            
            # Allocation tùy thuộc vào loại market
            alloc_ratio = (1 - last_candle['dca_initial_allocation']) / self.max_dca_orders.value
            dca_stake = stake_amount * self.max_dca_multiplier.value * alloc_ratio / last_candle['dca_initial_allocation']
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
        
        # Scalping cần thêm kiểm tra để đảm bảo không DCA khi thị trường biến động quá mạnh
        if last_candle['volatility'] > last_candle['volatility'].rolling(30).mean() * 1.5:
            return None
        
        # Không DCA khi volume quá thấp
        if last_candle['volume_ratio'] < 0.7:
            return None
        
        # Chỉ DCA khi có signal xác nhận theo hướng của trade
        if trade.is_short and not last_candle['short_signal_strength'] >= 1:
            return None
            
        if not trade.is_short and not last_candle['long_signal_strength'] >= 1:
            return None
            
        # Trả về số lượng vốn cho DCA - trong scalping thường nhỏ hơn
        return dca_stake * 0.8  # Giảm 20% so với chiến lược thông thường
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên market type và signal strength
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Trong scalping, đòn bẩy thường cao hơn do take profit nhanh
        leverage = self.leverage_default
        
        # Điều chỉnh đòn bẩy dựa trên market type
        if side == 'long':
            # Long signal strength
            signal_strength = last_candle['long_signal_strength']
            
            if last_candle['market_type'] == 'range_bound':
                if signal_strength >= 3:
                    leverage = min(5, max_leverage)  # Strong signal
                elif signal_strength >= 2:
                    leverage = min(4, max_leverage)  # Medium signal
                else:
                    leverage = min(3, max_leverage)  # Default
                    
            elif last_candle['market_type'] == 'uptrend':
                if signal_strength >= 2:
                    leverage = min(6, max_leverage)  # Strong uptrend signal
                else:
                    leverage = min(4, max_leverage)  # Regular uptrend
                    
            elif last_candle['market_type'] == 'downtrend':
                # Hạn chế đòn bẩy khi long trong downtrend
                leverage = min(2, max_leverage)
                
            elif last_candle['market_type'] == 'choppy':
                # Hạn chế đòn bẩy trong thị trường choppy
                leverage = min(2, max_leverage)
                
        else:  # short
            # Short signal strength
            signal_strength = last_candle['short_signal_strength']
            
            if last_candle['market_type'] == 'range_bound':
                if signal_strength >= 3:
                    leverage = min(5, max_leverage)  # Strong signal
                elif signal_strength >= 2:
                    leverage = min(4, max_leverage)  # Medium signal
                else:
                    leverage = min(3, max_leverage)  # Default
                    
            elif last_candle['market_type'] == 'downtrend':
                if signal_strength >= 2:
                    leverage = min(6, max_leverage)  # Strong downtrend signal
                else:
                    leverage = min(4, max_leverage)  # Regular downtrend
                    
            elif last_candle['market_type'] == 'uptrend':
                # Hạn chế đòn bẩy khi short trong uptrend
                leverage = min(2, max_leverage)
                
            elif last_candle['market_type'] == 'choppy':
                # Hạn chế đòn bẩy trong thị trường choppy
                leverage = min(2, max_leverage)
        
        # Giảm đòn bẩy nếu volatility cao
        if 'volatility' in last_candle and last_candle['volatility'] > last_candle['volatility'].rolling(30).mean() * 1.5:
            leverage = max(2, leverage - 2)  # Giảm 2 bậc nhưng không dưới 2
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động cho scalping - thường hẹp hơn chiến lược khác
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy số lần DCA đã thực hiện
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Với mỗi lần DCA, mở rộng stoploss để tránh stopped out quá sớm
        if dca_count == 0:
            # Stoploss mặc định dựa trên market type
            if 'initial_stoploss_pct' in last_candle:
                dynamic_stoploss = -last_candle['initial_stoploss_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value / 100
                
        elif dca_count == 1:
            # Sau DCA lần 1
            if 'stoploss_after_dca1_pct' in last_candle:
                dynamic_stoploss = -last_candle['stoploss_after_dca1_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value * 1.5 / 100
                
        elif dca_count == 2:
            # Sau DCA lần 2
            if 'stoploss_after_dca2_pct' in last_candle:
                dynamic_stoploss = -last_candle['stoploss_after_dca2_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value * 2.0 / 100
                
        else:
            # Sau DCA lần 3 trở lên
            if 'stoploss_after_dca3_pct' in last_candle:
                dynamic_stoploss = -last_candle['stoploss_after_dca3_pct'] / 100
            else:
                dynamic_stoploss = -self.initial_stoploss.value * 2.5 / 100
        
        # Điều chỉnh stoploss khi có lợi nhuận
        # Trong scalping, nhanh chóng bảo vệ lợi nhuận
        if current_profit > self.profit_target_1.value / 100:
            # Đã đạt profit target 1, thắt chặt stoploss
            return max(dynamic_stoploss, -0.005)  # Không quá -0.5%
            
        elif current_profit > self.profit_target_1.value / 200:  # Đạt 50% của target 1
            # Đã đạt 50% của profit target 1, giảm rủi ro
            return max(dynamic_stoploss, -0.01)  # Không quá -1%
        
        return dynamic_stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận vào lệnh với lọc thêm - đặc biệt quan trọng trong scalping
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        
        # Kiểm tra slippage - trong scalping, slippage control rất quan trọng
        current_price = last_candle['close']
        max_allowed_slippage = 0.1  # 0.1% maximum allowed slippage
        
        if side == 'long' and rate > current_price * (1 + max_allowed_slippage / 100):
            return False
            
        if side == 'short' and rate < current_price * (1 - max_allowed_slippage / 100):
            return False
        
        # Kiểm tra volatility - không vào lệnh khi volatility quá cao
        if 'volatility' in last_candle and last_candle['volatility'] > last_candle['volatility'].rolling(30).mean() * 2:
            return False
        
        # Kiểm tra volume không quá thấp
        if last_candle['volume_ratio'] < 0.7:
            return False
        
        # Kiểm tra BBand width - không vào lệnh khi bands quá rộng (volatility cao)
        if last_candle['bb_width'] > 0.1:
            return False
        
        # Kiểm tra signal strength - cần tối thiểu 1.5 
        if side == 'long' and last_candle['long_signal_strength'] < 1.5:
            return False
            
        if side == 'short' and last_candle['short_signal_strength'] < 1.5:
            return False
        
        # Kiểm tra xung đột tín hiệu - không nên có tín hiệu đối nghịch mạnh
        if side == 'long' and last_candle['short_signal_strength'] >= 2:
            return False
            
        if side == 'short' and last_candle['long_signal_strength'] >= 2:
            return False
        
        return True
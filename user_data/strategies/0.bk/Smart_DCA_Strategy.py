"""
Smart DCA Strategy
Chiến lược DCA thông minh (Dollar Cost Averaging) - tối ưu cho thị trường sideway/downtrend
Tối ưu cho các khung thời gian 1h và 4h
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

def detect_market_structure(dataframe, window=20, trend_lookback=5):
    """
    Phát hiện cấu trúc thị trường để quyết định có nên áp dụng DCA hay không
    
    Tham số:
    - window: Cửa sổ xem xét các đỉnh và đáy
    - trend_lookback: Số nến để xác định xu hướng hiện tại
    
    Trả về dataframe với các thông tin:
    - market_structure: 'uptrend', 'downtrend', 'sideway', 'uncertain'
    - dca_suitable: True/False (có nên áp dụng DCA không)
    - trend_strength: 0-100 (độ mạnh của xu hướng)
    """
    df = dataframe.copy()
    
    # Tính toán các chỉ báo kỹ thuật cơ bản
    # ADX để đo lường độ mạnh của xu hướng
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
    
    # EMAs để xác định xu hướng
    for period in [8, 13, 21, 50, 100]:
        df[f'ema_{period}'] = ta.trend.ema_indicator(close=df['close'], window=period)
    
    # Bollinger Bands để xác định volatility và kênh giá
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upperband'] = bollinger.bollinger_hband()
    df['bb_lowerband'] = bollinger.bollinger_lband()
    df['bb_middleband'] = bollinger.bollinger_mavg()
    df['bb_width'] = (df['bb_upperband'] - df['bb_lowerband']) / df['bb_middleband']
    
    # Phát hiện Higher Highs, Lower Lows
    df['rolling_max'] = df['high'].rolling(window=window).max()
    df['rolling_min'] = df['low'].rolling(window=window).min()
    
    df['higher_high'] = (df['high'] > df['rolling_max'].shift(1)) & (df['close'] > df['open'])
    df['lower_low'] = (df['low'] < df['rolling_min'].shift(1)) & (df['close'] < df['open'])
    
    # Xác định cấu trúc thị trường
    df['market_structure'] = 'uncertain'
    
    # Uptrend: EMAs tăng dần (8 > 13 > 21), giá trên EMA50
    df.loc[
        (df['ema_8'] > df['ema_13']) & 
        (df['ema_13'] > df['ema_21']) & 
        (df['close'] > df['ema_50']) &
        (df['adx'] > 25),
        'market_structure'
    ] = 'uptrend'
    
    # Downtrend: EMAs giảm dần (8 < 13 < 21), giá dưới EMA50
    df.loc[
        (df['ema_8'] < df['ema_13']) & 
        (df['ema_13'] < df['ema_21']) & 
        (df['close'] < df['ema_50']) &
        (df['adx'] > 25),
        'market_structure'
    ] = 'downtrend'
    
    # Sideway: BB width nhỏ, EMAs gần nhau, ADX thấp
    df.loc[
        (df['bb_width'] < 0.1) & 
        (abs(df['ema_8'] - df['ema_21']) / df['close'] < 0.01) &
        (df['adx'] < 20),
        'market_structure'
    ] = 'sideway'
    
    # DCA thích hợp cho sideway hoặc downtrend
    df['dca_suitable'] = (df['market_structure'].isin(['sideway', 'downtrend'])) | (df['bb_width'] < 0.05)
    
    # Độ mạnh của xu hướng (sử dụng ADX)
    df['trend_strength'] = df['adx'].clip(0, 100)
    
    # Xác định xác suất xu hướng tiếp tục
    df['trend_continuation_probability'] = 50.0  # Mặc định 50%
    
    # Uptrend mạnh có xác suất tiếp tục cao hơn
    df.loc[
        (df['market_structure'] == 'uptrend') & 
        (df['higher_high']) & 
        (df['adx'] > 30),
        'trend_continuation_probability'
    ] = 70.0
    
    # Downtrend mạnh có xác suất tiếp tục cao hơn
    df.loc[
        (df['market_structure'] == 'downtrend') & 
        (df['lower_low']) & 
        (df['adx'] > 30),
        'trend_continuation_probability'
    ] = 70.0
    
    # Xác định mức DCA lý tưởng
    df['ideal_dca_level_1'] = np.nan
    df['ideal_dca_level_2'] = np.nan
    df['ideal_dca_level_3'] = np.nan
    
    # Mức DCA dựa trên hỗ trợ từ BB bands và EMAs
    for i in range(1, len(df)):
        if df['market_structure'].iloc[i] == 'downtrend' or df['market_structure'].iloc[i] == 'sideway':
            current_price = df['close'].iloc[i]
            
            # Long position DCA levels (buy lower)
            df['ideal_dca_level_1'].iloc[i] = current_price * 0.985  # 1.5% below current price
            df['ideal_dca_level_2'].iloc[i] = current_price * 0.97   # 3% below current price
            df['ideal_dca_level_3'].iloc[i] = current_price * 0.95   # 5% below current price
            
            # Điều chỉnh dựa trên BB bands
            if df['bb_lowerband'].iloc[i] > df['ideal_dca_level_1'].iloc[i]:
                df['ideal_dca_level_1'].iloc[i] = df['bb_lowerband'].iloc[i]
            
            if df['ema_50'].iloc[i] > df['ideal_dca_level_2'].iloc[i] and df['ema_50'].iloc[i] < current_price:
                df['ideal_dca_level_2'].iloc[i] = df['ema_50'].iloc[i]
            
            if df['ema_100'].iloc[i] > df['ideal_dca_level_3'].iloc[i] and df['ema_100'].iloc[i] < current_price:
                df['ideal_dca_level_3'].iloc[i] = df['ema_100'].iloc[i]
    
    return df

def analyze_dca_opportunity(dataframe, max_dca_count=3):
    """
    Phân tích cơ hội DCA và xác định phân bổ vốn tối ưu
    
    Tham số:
    - dataframe: DataFrame với dữ liệu giá và các chỉ báo
    - max_dca_count: Số lần DCA tối đa
    
    Trả về:
    - dataframe với thông tin DCA tối ưu
    """
    df = dataframe.copy()
    
    # Xác định mức giá hiện tại và các mức DCA tiềm năng
    price = df['close'].iloc[-1]
    
    # Chỉ số RSI để đánh giá oversold/overbought
    df['rsi'] = ta.momentum.rsi(close=df['close'], window=14)
    
    # Fibonacci retracement levels
    high_point = df['high'].iloc[-100:].max()
    low_point = df['low'].iloc[-100:].min()
    fib_diff = high_point - low_point
    
    df['fib_0'] = low_point
    df['fib_0.236'] = low_point + 0.236 * fib_diff
    df['fib_0.382'] = low_point + 0.382 * fib_diff
    df['fib_0.5'] = low_point + 0.5 * fib_diff
    df['fib_0.618'] = low_point + 0.618 * fib_diff
    df['fib_0.786'] = low_point + 0.786 * fib_diff
    df['fib_1'] = high_point
    
    # Xác định phân bổ vốn tối ưu dựa trên cấu trúc thị trường
    df['dca_entry_allocation'] = 0.4  # Default 40% for initial entry
    
    # Downtrend: Phân bổ entry nhỏ hơn, giữ vốn cho DCA
    df.loc[df['market_structure'] == 'downtrend', 'dca_entry_allocation'] = 0.3  # 30% for initial entry
    
    # Sideway: Phân bổ entry vừa phải
    df.loc[df['market_structure'] == 'sideway', 'dca_entry_allocation'] = 0.4  # 40% for initial entry
    
    # Uptrend: Phân bổ entry lớn hơn
    df.loc[df['market_structure'] == 'uptrend', 'dca_entry_allocation'] = 0.6  # 60% for initial entry
    
    # Calculate DCA allocations based on market structure
    for i in range(1, max_dca_count + 1):
        # Downtrend: Phân bổ nhiều hơn cho DCA sau
        df.loc[df['market_structure'] == 'downtrend', f'dca_{i}_allocation'] = (0.7 / max_dca_count) * (1 + 0.2 * (i-1))
        
        # Sideway: Phân bổ đều
        df.loc[df['market_structure'] == 'sideway', f'dca_{i}_allocation'] = 0.6 / max_dca_count
        
        # Uptrend: Phân bổ ít cho DCA
        df.loc[df['market_structure'] == 'uptrend', f'dca_{i}_allocation'] = (0.4 / max_dca_count) * (1 - 0.1 * (i-1))
    
    # Calculate ideal price levels for DCA
    last_close = df['close'].iloc[-1]
    
    # DCA levels as percentage drops from entry
    # Adjusted based on market structure and volatility
    df['dca_1_pct'] = np.where(
        df['market_structure'] == 'downtrend',
        -1.0 - (df['bb_width'] * 10),  # More aggressive in downtrend
        -1.5
    )
    
    df['dca_2_pct'] = np.where(
        df['market_structure'] == 'downtrend',
        -2.5 - (df['bb_width'] * 15),  # More aggressive in downtrend
        -3.0
    )
    
    df['dca_3_pct'] = np.where(
        df['market_structure'] == 'downtrend',
        -4.0 - (df['bb_width'] * 20),  # More aggressive in downtrend
        -5.0
    )
    
    # Calculate expected averaged entry price if all DCAs are executed
    df['avg_entry_after_dca'] = 0.0
    for i in range(len(df)):
        if pd.isna(df['dca_entry_allocation'].iloc[i]):
            continue
            
        entry_price = df['close'].iloc[i]
        
        # Calculate prices at each DCA level
        dca_1_price = entry_price * (1 + df['dca_1_pct'].iloc[i]/100)
        dca_2_price = entry_price * (1 + df['dca_2_pct'].iloc[i]/100)
        dca_3_price = entry_price * (1 + df['dca_3_pct'].iloc[i]/100)
        
        # Calculate allocations
        entry_alloc = df['dca_entry_allocation'].iloc[i]
        dca_1_alloc = df['dca_1_allocation'].iloc[i] if 'dca_1_allocation' in df.columns else (1 - entry_alloc) / 3
        dca_2_alloc = df['dca_2_allocation'].iloc[i] if 'dca_2_allocation' in df.columns else (1 - entry_alloc) / 3
        dca_3_alloc = df['dca_3_allocation'].iloc[i] if 'dca_3_allocation' in df.columns else (1 - entry_alloc) / 3
        
        # Calculate weighted average
        avg_entry = (
            (entry_price * entry_alloc) +
            (dca_1_price * dca_1_alloc) +
            (dca_2_price * dca_2_alloc) +
            (dca_3_price * dca_3_alloc)
        ) / (entry_alloc + dca_1_alloc + dca_2_alloc + dca_3_alloc)
        
        df['avg_entry_after_dca'].iloc[i] = avg_entry
    
    # Calculate potential profit % based on bounce to recent high
    recent_high = df['high'].iloc[-20:].max()
    df['potential_profit_pct'] = ((recent_high / df['avg_entry_after_dca']) - 1) * 100
    
    return df

class Smart_DCA_Strategy(IStrategy):
    """
    Chiến lược DCA thông minh (Dollar Cost Averaging)
    
    - Phát hiện cấu trúc thị trường để xác định khi nào DCA phù hợp
    - Tự động tính toán mức giá DCA tối ưu dựa trên phân tích kỹ thuật
    - Phân bổ vốn thông minh dựa trên độ tin cậy của tín hiệu
    - Tập trung vào mục tiêu lợi nhuận 3% mỗi ngày bằng cách tối ưu hóa exit
    """
    
    # Tham số cho chiến lược DCA
    use_dca = CategoricalParameter([True, False], default=True, space="buy", optimize=True)
    max_dca_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    max_dca_orders = IntParameter(1, 3, default=3, space="buy", optimize=True)
    
    # Tham số cho cấu trúc thị trường
    market_trend_window = IntParameter(10, 200, default=100, space="buy", optimize=True)
    adx_threshold = IntParameter(15, 35, default=25, space="buy", optimize=True)
    
    # Tham số cho entry
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", optimize=True)
    bb_width_threshold = DecimalParameter(0.05, 0.2, default=0.1, space="buy", optimize=True)
    
    # Tham số cho exit
    profit_target_1 = DecimalParameter(1.0, 5.0, default=1.5, space="sell", optimize=True)
    profit_target_2 = DecimalParameter(2.0, 6.0, default=3.0, space="sell", optimize=True)
    profit_target_3 = DecimalParameter(3.0, 8.0, default=5.0, space="sell", optimize=True)
    
    # Tham số cho quản lý rủi ro
    initial_stoploss = DecimalParameter(2.0, 10.0, default=5.0, space="sell", optimize=True)
    stoploss_after_dca_1 = DecimalParameter(3.0, 12.0, default=7.0, space="sell", optimize=True)
    stoploss_after_dca_2 = DecimalParameter(5.0, 15.0, default=10.0, space="sell", optimize=True)
    stoploss_after_dca_3 = DecimalParameter(7.0, 20.0, default=15.0, space="sell", optimize=True)
    
    # Core parameters
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.05  # -5% stoploss mặc định, sẽ được điều chỉnh theo dca trong code
    
    # Time-in-force và ROI
    timeframe = '1h'
    
    minimal_roi = {
        "0": 0.03,     # 3%
        "60": 0.02,    # 2% sau 60 phút
        "120": 0.01,   # 1% sau 120 phút
        "240": 0.005   # 0.5% sau 240 phút
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
        
        # Phát hiện cấu trúc thị trường
        structure_df = detect_market_structure(
            dataframe,
            window=self.market_trend_window.value
        )
        
        # Thêm các cột cấu trúc thị trường
        for column in structure_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = structure_df[column]
        
        # Phân tích cơ hội DCA
        dca_df = analyze_dca_opportunity(
            dataframe,
            max_dca_count=self.max_dca_orders.value
        )
        
        # Thêm các cột phân tích DCA
        for column in dca_df.columns:
            if column not in dataframe.columns:
                dataframe[column] = dca_df[column]
        
        # Thêm các chỉ báo phổ biến
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=14)
        dataframe['rsi_ma'] = ta.trend.sma_indicator(close=dataframe['rsi'], window=14)
        
        # MACD
        macd = ta.trend.macd(close=dataframe['close'], window_fast=12, window_slow=26, window_sign=9)
        dataframe['macd'] = macd.macd()
        dataframe['macdsignal'] = macd.macd_signal()
        dataframe['macdhist'] = macd.macd_diff()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=dataframe['close'], window=20, window_dev=2)
        dataframe['bb_upperband'] = bollinger.bollinger_hband()
        dataframe['bb_lowerband'] = bollinger.bollinger_lband()
        dataframe['bb_middleband'] = bollinger.bollinger_mavg()
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        dataframe['bb_percent'] = (dataframe['close'] - dataframe['bb_lowerband']) / (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        
        # EMAs
        for period in [8, 13, 21, 50, 100, 200]:
            dataframe[f'ema_{period}'] = ta.trend.ema_indicator(close=dataframe['close'], window=period)
        
        # ATR
        dataframe['atr'] = ta.volatility.average_true_range(high=dataframe['high'], low=dataframe['low'], close=dataframe['close'], window=14)
        
        # Volume
        dataframe['volume_mean_20'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean_20']
        
        # Chỉ báo cho entry
        dataframe['buy_signal'] = (
            (dataframe['rsi'] < self.rsi_oversold.value) & 
            (dataframe['bb_percent'] < 0.2) &  # Giá gần BB lower
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        # Xác định nếu thị trường phù hợp cho DCA
        dataframe['dca_market'] = (
            (dataframe['market_structure'].isin(['sideway', 'downtrend'])) |
            (dataframe['bb_width'] < self.bb_width_threshold.value)
        )
        
        # Xác định tín hiệu reversal
        dataframe['potential_reversal'] = (
            (dataframe['rsi'] < 30) &  # RSI thấp
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI đang tăng
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Giá đang tăng
            (dataframe['volume'] > dataframe['volume_mean_20'])  # Volume cao
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Signal dựa trên RSI oversold và giá gần Bollinger lower band
        conditions_long.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['bb_percent'] < 0.2) &  # Giá gần BB lower
            (dataframe['volume_ratio'] > 0.8)  # Volume không quá thấp
        )
        
        # 2. Reversal signal trong market structure phù hợp
        conditions_long.append(
            (dataframe['potential_reversal']) &
            (dataframe['dca_market']) &
            (dataframe['potential_profit_pct'] > self.profit_target_2.value)
        )
        
        # 3. Signal đặc biệt cho market structure downtrend - vào tại vùng hỗ trợ mạnh
        conditions_long.append(
            (dataframe['market_structure'] == 'downtrend') &
            (dataframe['close'] < dataframe['ema_200'] * 0.95) &  # Giá dưới EMA200 ít nhất 5%
            (dataframe['rsi'] < 35) &  # RSI thấp
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI đang tăng
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['volume'] > dataframe['volume_mean_20'] * 1.5)  # Volume cao
        )
        
        # 4. Signal dựa trên chỉ báo DCA và cấu trúc thị trường
        conditions_long.append(
            (dataframe['dca_market']) &
            (dataframe['close'] < dataframe['ema_50']) &  # Giá dưới EMA50
            (dataframe['rsi'] < 40) &  # RSI thấp
            (dataframe['potential_profit_pct'] > self.profit_target_3.value) &  # Lợi nhuận tiềm năng cao
            (dataframe['volume_ratio'] > 1.0)  # Volume tốt
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # SHORT entries are very limited in DCA strategy because DCA works best for long positions
        # in downtrend/sideway markets
        
        # Điều kiện vào lệnh SHORT - rất hạn chế, chỉ trong điều kiện uptrend mạnh + overbought
        conditions_short = []
        
        # 1. Signal dựa trên RSI overbought và giá gần Bollinger upper band trong uptrend
        conditions_short.append(
            (dataframe['market_structure'] == 'uptrend') &
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['bb_percent'] > 0.8) &  # Giá gần BB upper
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['volume_ratio'] > 1.0)  # Volume tốt
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
        
        # 1. Thoát khi đạt profit target
        conditions_exit_long.append(
            (dataframe['close'] > dataframe['avg_entry_after_dca'] * (1 + self.profit_target_2.value/100)) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Thoát khi trend chuyển sang bearish mạnh
        conditions_exit_long.append(
            (dataframe['market_structure'] == 'downtrend') &
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['close'] < dataframe['ema_50']) &
            (dataframe['adx'] > self.adx_threshold.value)
        )
        
        # 3. Thoát khi RSI overbought và giá tạo đỉnh
        conditions_exit_long.append(
            (dataframe['rsi'] > self.rsi_overbought.value) &
            (dataframe['rsi'] < dataframe['rsi'].shift(1)) &  # RSI đang giảm
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Giá đang giảm
            (dataframe['volume'] > dataframe['volume_mean_20'])  # Volume cao
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Thoát khi đạt profit target
        conditions_exit_short.append(
            (dataframe['close'] < dataframe['avg_entry_after_dca'] * (1 - self.profit_target_2.value/100)) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. Thoát khi trend chuyển sang bullish mạnh
        conditions_exit_short.append(
            (dataframe['market_structure'] == 'uptrend') &
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['close'] > dataframe['ema_50']) &
            (dataframe['adx'] > self.adx_threshold.value)
        )
        
        # 3. Thoát khi RSI oversold và giá tạo đáy
        conditions_exit_short.append(
            (dataframe['rsi'] < self.rsi_oversold.value) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # RSI đang tăng
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Giá đang tăng
            (dataframe['volume'] > dataframe['volume_mean_20'])  # Volume cao
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
        
        # Chỉ áp dụng DCA cho long positions (thông thường)
        if trade.is_short:
            return None
        
        # Get DCA information
        dca_info = {}
        
        # Tính số lần đã DCA
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Đã đạt số lần DCA tối đa
        if dca_count >= self.max_dca_orders.value:
            return None
        
        # Calculate units to buy for DCA
        stake_amount = trade.stake_amount
        dca_stake = 0.0
        
        current_dca_level = dca_count + 1
        
        # Xác định mức giá và stake amount cho DCA tiếp theo
        # DCA level 1
        if current_dca_level == 1:
            dca_threshold = last_candle[f'dca_1_pct']
            dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle[f'dca_1_allocation']) / float(last_candle['dca_entry_allocation'])
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
            
        # DCA level 2
        elif current_dca_level == 2:
            dca_threshold = last_candle[f'dca_2_pct']
            dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle[f'dca_2_allocation']) / float(last_candle['dca_entry_allocation'])
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
            
        # DCA level 3
        elif current_dca_level == 3:
            dca_threshold = last_candle[f'dca_3_pct']
            dca_stake = stake_amount * self.max_dca_multiplier.value * float(last_candle[f'dca_3_allocation']) / float(last_candle['dca_entry_allocation'])
            
            # Kiểm tra xem đã đạt ngưỡng giá DCA chưa
            if current_profit >= dca_threshold / 100:  # convert pct to ratio
                return None
        
        # Thêm kiểm tra xem thị trường có phù hợp để DCA không
        if not last_candle['dca_market']:
            return None
        
        # Thêm kiểm tra volume để đảm bảo có đủ thanh khoản
        if last_candle['volume_ratio'] < 0.8:
            return None
        
        # Thêm kiểm tra xem giá hiện tại có ở mức "giá trị" không
        # Ví dụ: Gần Bollinger lower band hoặc các vùng hỗ trợ
        if last_candle['bb_percent'] > 0.3:  # Không đủ gần band dưới
            return None
        
        # Kiểm tra xem lợi nhuận tiềm năng có đủ lớn không
        if last_candle['potential_profit_pct'] < self.profit_target_2.value:
            return None
        
        # Trả về số lượng vốn cho DCA
        return dca_stake
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên cấu trúc thị trường và chiến lược DCA
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Thị trường phù hợp cho DCA - sử dụng đòn bẩy thấp hơn để giảm rủi ro
        if last_candle['dca_market']:
            if side == 'long':
                # Trong thị trường thích hợp cho DCA (sideway/downtrend), giảm đòn bẩy cho LONG
                if last_candle['market_structure'] == 'downtrend':
                    return min(2, max_leverage)  # max 2x trong downtrend
                else:  # sideway
                    return min(3, max_leverage)  # max 3x trong sideway
            else:  # short
                # Không sử dụng DCA cho short, nên có thể tăng đòn bẩy nếu trong downtrend
                if last_candle['market_structure'] == 'downtrend':
                    return min(4, max_leverage)  # max 4x trong downtrend
                else:  # sideway
                    return min(3, max_leverage)  # max 3x trong sideway
        
        # Thị trường uptrend
        if last_candle['market_structure'] == 'uptrend':
            if side == 'long':
                # Đòn bẩy cao hơn cho LONG trong uptrend
                return min(5, max_leverage)  # max 5x
            else:  # short
                # Đòn bẩy thấp cho SHORT trong uptrend
                return min(2, max_leverage)  # max 2x
        
        # Default leverage
        return self.leverage_default
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên số lần DCA đã thực hiện
        """
        # Lấy số lần DCA đã thực hiện
        filled_entries = trade.select_filled_orders('enter')
        dca_count = max(0, len(filled_entries) - 1)
        
        # Với mỗi lần DCA, mở rộng stoploss để tránh stopped out quá sớm
        if dca_count == 0:
            # Chưa DCA lần nào - stoploss ban đầu
            dynamic_stoploss = -self.initial_stoploss.value / 100
        elif dca_count == 1:
            # Đã DCA 1 lần
            dynamic_stoploss = -self.stoploss_after_dca_1.value / 100
        elif dca_count == 2:
            # Đã DCA 2 lần
            dynamic_stoploss = -self.stoploss_after_dca_2.value / 100
        else:
            # Đã DCA 3 lần hoặc nhiều hơn
            dynamic_stoploss = -self.stoploss_after_dca_3.value / 100
        
        # Trả về stoploss tính theo phần trăm
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
        
        # Kiểm tra thị trường có phù hợp với chiến lược không
        if side == 'long':
            # Đối với long, chỉ vào lệnh trong thị trường sideway hoặc uptrend, hoặc tại vùng hỗ trợ mạnh
            if last_candle['market_structure'] == 'downtrend':
                # Trong downtrend, chỉ vào lệnh khi:
                # 1. Có tín hiệu reversal mạnh
                # 2. Giá ở vùng hỗ trợ quan trọng
                # 3. Giá ở vùng oversold
                
                if not (last_candle['potential_reversal'] or 
                       last_candle['close'] < last_candle['ema_200'] * 0.95 or
                       last_candle['rsi'] < 35):
                    return False
            
            # Kiểm tra xem có lợi nhuận tiềm năng đủ lớn không
            if last_candle['potential_profit_pct'] < self.profit_target_1.value:
                return False
            
        else:  # short
            # Đối với short, chỉ vào lệnh trong uptrend mạnh và overbought
            if last_candle['market_structure'] != 'uptrend' or last_candle['rsi'] < self.rsi_overbought.value:
                return False
        
        # Kiểm tra volume
        if last_candle['volume_ratio'] < 0.7:
            return False
        
        # Kiểm tra khoảng cách entry
        if side == 'long' and rate > last_candle['close'] * 1.01:  # Giá entry cao hơn 1%
            return False
        
        if side == 'short' and rate < last_candle['close'] * 0.99:  # Giá entry thấp hơn 1%
            return False
        
        return True
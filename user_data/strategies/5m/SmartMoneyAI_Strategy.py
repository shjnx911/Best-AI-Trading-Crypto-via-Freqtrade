"""
Smart Money Concept + AI Strategy for FreqTrade
Kết hợp giữa concept Smart Money và mô hình AI để xác định điểm vào lệnh tối ưu
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
# Sử dụng ta thay vì talib
import ta

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

from pandas import DataFrame
from functools import reduce
from typing import Dict, List, Optional, Tuple
import logging
import os
import joblib
from sklearn.preprocessing import StandardScaler

# Define a qtpylib replacement for indicators
class qtpylib:
    @staticmethod
    def bollinger_bands(dataframe, window=20, stds=2):
        """Calculate Bollinger Bands"""
        mean = dataframe.rolling(window=window).mean()
        std = dataframe.rolling(window=window).std()
        upper = mean + std * stds
        lower = mean - std * stds
        return {'upper': upper, 'mid': mean, 'lower': lower}
        
    @staticmethod
    def typical_price(dataframe):
        """Calculate typical price"""
        return (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3

logger = logging.getLogger(__name__)

def normalize_data(df, columns):
    """Chuẩn hóa dữ liệu để đưa vào mô hình AI"""
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

class SmartMoneyAI_Strategy(IStrategy):
    """
    Chiến lược kết hợp Smart Money Concept với AI
    
    Smart Money Concept dựa trên việc xác định các Order Block, 
    Fair Value Gap (FVG), Equal High/Low (EQH/EQL), và phân tích cấu trúc thị trường
    (Market Structure).
    
    Kết hợp với AI để xác định độ tin cậy của mỗi tín hiệu.
    """
    
    # Thông số tối ưu hóa cho chiến lược
    order_block_length = IntParameter(3, 20, default=5, space="buy", optimize=True)
    fvg_threshold = DecimalParameter(0.2, 1.0, default=0.5, space="buy", optimize=True)
    eqhl_bars = IntParameter(3, 10, default=5, space="buy", optimize=True)
    min_rsi = IntParameter(10, 40, default=30, space="buy", optimize=True)
    max_rsi = IntParameter(60, 90, default=70, space="sell", optimize=True)
    trend_ema_period = IntParameter(20, 200, default=50, space="buy", optimize=True)
    score_threshold = DecimalParameter(60, 80, default=70, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 8
    stoploss = -0.02
    
    # Time-in-force và ROI
    timeframe = '5m'
    timeframe_higher = '1h'
    
    minimal_roi = {
        "0": 0.05,
        "30": 0.025,
        "60": 0.015,
        "120": 0.01
    }
    
    # Trailing stoploss
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.025
    trailing_only_offset_is_reached = True
    
    # Thời gian để khởi tạo các chỉ báo
    startup_candle_count = 100
    process_only_new_candles = True
    
    # Thư mục lưu mô hình AI
    models_dir = os.path.join('user_data', 'models')
    
    def __init__(self, config: dict) -> None:
        """Khởi tạo chiến lược"""
        super().__init__(config)
        self.config = config
        
        # Tạo thư mục models nếu chưa tồn tại
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Tải mô hình AI để dự đoán độ tin cậy
        self.ai_model = self._load_model('smart_money_model.joblib')
    
    def _load_model(self, model_name):
        """Tải mô hình AI từ thư mục models"""
        model_path = os.path.join(self.models_dir, model_name)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Đã tải mô hình AI từ {model_path}")
                return model
            except Exception as e:
                logger.warning(f"Không thể tải mô hình AI: {e}")
        else:
            logger.warning(f"Không tìm thấy mô hình AI tại {model_path}")
        return None
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_higher) for pair in pairs]
        return informative_pairs
    
    def get_informative_indicators(self, metadata: dict, dataframe: DataFrame) -> DataFrame:
        """Thêm các chỉ báo từ khung thời gian cao hơn"""
        if not self.dp:
            return dataframe
            
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], 
            timeframe=self.timeframe_higher
        )
        
        # Thêm các chỉ báo ở timeframe cao hơn
        informative['ema200'] = ta.EMA(informative, timeperiod=200)
        informative['ema100'] = ta.EMA(informative, timeperiod=100)
        informative['ema50'] = ta.EMA(informative, timeperiod=50)
        
        # Thêm các mẫu nến đảo chiều ở timeframe cao hơn
        informative['engulfing'] = self.detect_engulfing_pattern(informative)
        informative['pinbar'] = self.detect_pinbar_pattern(informative)
        
        # Xác định các Order Block
        informative['bullish_ob'] = self.detect_bullish_order_blocks(informative, self.order_block_length.value)
        informative['bearish_ob'] = self.detect_bearish_order_blocks(informative, self.order_block_length.value)
        
        # Xác định cấu trúc thị trường
        informative['market_structure'] = self.analyze_market_structure(informative)
        
        # Đổi tên để tránh trùng lặp
        informative_indicators = {
            f"inf_{col}": informative[col] for col in informative.columns if col != 'date'
        }
        
        # Merge với dataframe gốc
        dataframe = pd.merge(
            dataframe, informative_indicators, left_on='date', right_on='date', how='left'
        )
        
        return dataframe
    
    def detect_engulfing_pattern(self, dataframe: DataFrame) -> pd.Series:
        """Phát hiện mẫu nến Engulfing - một kiểu Smart Money pattern"""
        # Bullish Engulfing: cây nến hiện tại mở dưới đóng trên và "nuốt" cây nến trước
        bullish_engulfing = (
            (dataframe['open'] < dataframe['close']) &  # Nến tăng
            (dataframe['open'].shift(1) > dataframe['close'].shift(1)) &  # Nến trước giảm
            (dataframe['open'] <= dataframe['close'].shift(1)) &  # Mở dưới đóng của nến trước
            (dataframe['close'] >= dataframe['open'].shift(1))  # Đóng trên mở của nến trước
        )
        
        # Bearish Engulfing: cây nến hiện tại mở trên đóng dưới và "nuốt" cây nến trước
        bearish_engulfing = (
            (dataframe['open'] > dataframe['close']) &  # Nến giảm
            (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &  # Nến trước tăng
            (dataframe['open'] >= dataframe['close'].shift(1)) &  # Mở trên đóng của nến trước
            (dataframe['close'] <= dataframe['open'].shift(1))  # Đóng dưới mở của nến trước
        )
        
        # 1 cho bullish, -1 cho bearish, 0 cho không có mẫu
        result = pd.Series(0, index=dataframe.index)
        result.loc[bullish_engulfing] = 1
        result.loc[bearish_engulfing] = -1
        
        return result
    
    def detect_pinbar_pattern(self, dataframe: DataFrame) -> pd.Series:
        """Phát hiện mẫu nến Pinbar - thường xuất hiện ở vùng reversal"""
        # Tính độ dài của thân nến, bóng trên và bóng dưới
        body = abs(dataframe['open'] - dataframe['close'])
        upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # Bullish Pinbar: bóng dưới dài ít nhất 2 lần thân, bóng trên ngắn
        bullish_pinbar = (
            (lower_wick > 2 * body) &
            (lower_wick > upper_wick * 3) &
            (body > 0)  # Thân nến phải có độ dài
        )
        
        # Bearish Pinbar: bóng trên dài ít nhất 2 lần thân, bóng dưới ngắn
        bearish_pinbar = (
            (upper_wick > 2 * body) &
            (upper_wick > lower_wick * 3) &
            (body > 0)  # Thân nến phải có độ dài
        )
        
        # 1 cho bullish, -1 cho bearish, 0 cho không có mẫu
        result = pd.Series(0, index=dataframe.index)
        result.loc[bullish_pinbar] = 1
        result.loc[bearish_pinbar] = -1
        
        return result
    
    def detect_bullish_order_blocks(self, dataframe: DataFrame, length: int) -> pd.Series:
        """
        Phát hiện Order Block tăng (Bullish)
        Order Block là nến giảm mạnh ngay trước chuỗi tăng
        """
        result = pd.Series(0, index=dataframe.index)
        
        # Tìm nến giảm mạnh
        strong_bearish = (
            (dataframe['open'] > dataframe['close']) &  # Nến giảm
            (dataframe['close'] < dataframe['low'].shift(1)) &  # Đóng dưới low của nến trước
            (dataframe['high'] - dataframe['low'] > 1.5 * (dataframe['high'].rolling(10).mean() - dataframe['low'].rolling(10).mean()))  # Biên độ cao
        )
        
        # Kiểm tra nếu sau nến giảm mạnh là một chuỗi nến tăng
        for i in range(1, len(dataframe) - length):
            if strong_bearish.iloc[i-1]:
                # Kiểm tra nếu sau nến giảm có chuỗi tăng
                if (dataframe['close'].iloc[i:i+length] > dataframe['open'].iloc[i:i+length]).all():
                    result.iloc[i-1] = 1
        
        return result
    
    def detect_bearish_order_blocks(self, dataframe: DataFrame, length: int) -> pd.Series:
        """
        Phát hiện Order Block giảm (Bearish)
        Order Block là nến tăng mạnh ngay trước chuỗi giảm
        """
        result = pd.Series(0, index=dataframe.index)
        
        # Tìm nến tăng mạnh
        strong_bullish = (
            (dataframe['open'] < dataframe['close']) &  # Nến tăng
            (dataframe['close'] > dataframe['high'].shift(1)) &  # Đóng trên high của nến trước
            (dataframe['high'] - dataframe['low'] > 1.5 * (dataframe['high'].rolling(10).mean() - dataframe['low'].rolling(10).mean()))  # Biên độ cao
        )
        
        # Kiểm tra nếu sau nến tăng mạnh là một chuỗi nến giảm
        for i in range(1, len(dataframe) - length):
            if strong_bullish.iloc[i-1]:
                # Kiểm tra nếu sau nến tăng có chuỗi giảm
                if (dataframe['close'].iloc[i:i+length] < dataframe['open'].iloc[i:i+length]).all():
                    result.iloc[i-1] = 1
        
        return result
    
    def analyze_market_structure(self, dataframe: DataFrame) -> pd.Series:
        """
        Phân tích cấu trúc thị trường (Market Structure)
        Trả về 1 cho uptrend, -1 cho downtrend, 0 cho sideways
        """
        # Xác định Higher Highs & Higher Lows cho uptrend
        dataframe['higher_high'] = (dataframe['high'] > dataframe['high'].shift(1)) & (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        dataframe['higher_low'] = (dataframe['low'] > dataframe['low'].shift(1)) & (dataframe['low'].shift(1) > dataframe['low'].shift(2))
        
        # Xác định Lower Highs & Lower Lows cho downtrend
        dataframe['lower_high'] = (dataframe['high'] < dataframe['high'].shift(1)) & (dataframe['high'].shift(1) < dataframe['high'].shift(2))
        dataframe['lower_low'] = (dataframe['low'] < dataframe['low'].shift(1)) & (dataframe['low'].shift(1) < dataframe['low'].shift(2))
        
        # Uptrend: có Higher Highs và Higher Lows
        uptrend = (dataframe['higher_high'] & dataframe['higher_low'])
        
        # Downtrend: có Lower Highs và Lower Lows
        downtrend = (dataframe['lower_high'] & dataframe['lower_low'])
        
        # EMA slope cũng là một chỉ báo tốt về cấu trúc thị trường
        dataframe['ema50_slope'] = dataframe['ema50'].diff(5)
        
        # Kết hợp các yếu tố để xác định cấu trúc
        result = pd.Series(0, index=dataframe.index)  # Mặc định là sideways
        result.loc[uptrend | (dataframe['ema50_slope'] > 0)] = 1  # Uptrend
        result.loc[downtrend | (dataframe['ema50_slope'] < 0)] = -1  # Downtrend
        
        return result
    
    def detect_fair_value_gap(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện Fair Value Gap (FVG)
        - Bullish FVG: Low của nến hiện tại > High của nến thứ 2 trước đó
        - Bearish FVG: High của nến hiện tại < Low của nến thứ 2 trước đó
        """
        dataframe['bullish_fvg'] = 0
        dataframe['bearish_fvg'] = 0
        
        # Phát hiện Bullish FVG
        bullish_fvg_condition = dataframe['low'] > dataframe['high'].shift(2)
        dataframe.loc[bullish_fvg_condition, 'bullish_fvg'] = 1
        
        # Phát hiện Bearish FVG
        bearish_fvg_condition = dataframe['high'] < dataframe['low'].shift(2)
        dataframe.loc[bearish_fvg_condition, 'bearish_fvg'] = 1
        
        # Tính kích thước của FVG
        dataframe['bullish_fvg_size'] = 0
        dataframe['bearish_fvg_size'] = 0
        
        dataframe.loc[bullish_fvg_condition, 'bullish_fvg_size'] = (dataframe['low'] - dataframe['high'].shift(2)) / dataframe['low']
        dataframe.loc[bearish_fvg_condition, 'bearish_fvg_size'] = (dataframe['low'].shift(2) - dataframe['high']) / dataframe['high']
        
        return dataframe
    
    def detect_equal_highs_lows(self, dataframe: DataFrame, bars: int) -> DataFrame:
        """
        Phát hiện Equal Highs (EQH) và Equal Lows (EQL)
        Được sử dụng để xác định các điểm tích lũy và breakout
        """
        dataframe['eqh'] = 0
        dataframe['eql'] = 0
        
        # Phát hiện các swing high và swing low
        for i in range(bars, len(dataframe) - bars):
            # Kiểm tra Equal Highs (các đỉnh bằng nhau)
            if all(abs(dataframe['high'].iloc[i] - dataframe['high'].iloc[i-j]) / dataframe['high'].iloc[i] < 0.001 for j in range(1, 3)):
                dataframe['eqh'].iloc[i] = 1
                
            # Kiểm tra Equal Lows (các đáy bằng nhau)
            if all(abs(dataframe['low'].iloc[i] - dataframe['low'].iloc[i-j]) / dataframe['low'].iloc[i] < 0.001 for j in range(1, 3)):
                dataframe['eql'].iloc[i] = 1
                
        return dataframe
    
    def detect_fibonacci_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        Xác định các mức Fibonacci từ swing high và swing low gần nhất
        """
        # Tìm swing high và swing low gần nhất (21 nến)
        window = 21
        dataframe['swing_high'] = dataframe['high'].rolling(window=window, center=True).max()
        dataframe['swing_low'] = dataframe['low'].rolling(window=window, center=True).min()
        
        # Tính toán các mức Fibonacci Retracement cho downtrend (từ high xuống low)
        # 0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
        fib_levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for level in fib_levels:
            # Tính toán mức Fibonacci cho cả uptrend và downtrend
            dataframe[f'fib_{int(level*1000)}'] = dataframe['swing_low'] + (dataframe['swing_high'] - dataframe['swing_low']) * level
        
        # Thêm cột chỉ ra khi giá gần các mức Fibonacci
        dataframe['near_fib_level'] = 0
        
        # Xem xét giá gần các mức Fibonacci nào (trong khoảng 0.2%)
        price = dataframe['close']
        for level in fib_levels:
            fib_price = dataframe[f'fib_{int(level*1000)}']
            is_near = (abs(price - fib_price) / price < 0.002)  # Trong khoảng 0.2%
            dataframe.loc[is_near, 'near_fib_level'] = level
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        # Lấy thông tin từ timeframe cao hơn
        dataframe = self.get_informative_indicators(metadata, dataframe)
        
        # Tính các chỉ báo kỹ thuật cơ bản
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma200'] = ta.SMA(dataframe, timeperiod=200)
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        # Phát hiện mẫu nến
        dataframe['engulfing'] = self.detect_engulfing_pattern(dataframe)
        dataframe['pinbar'] = self.detect_pinbar_pattern(dataframe)
        
        # Phát hiện Order Blocks
        dataframe['bullish_ob'] = self.detect_bullish_order_blocks(dataframe, self.order_block_length.value)
        dataframe['bearish_ob'] = self.detect_bearish_order_blocks(dataframe, self.order_block_length.value)
        
        # Phát hiện Fair Value Gap
        dataframe = self.detect_fair_value_gap(dataframe)
        
        # Phát hiện Equal Highs/Lows
        dataframe = self.detect_equal_highs_lows(dataframe, self.eqhl_bars.value)
        
        # Tính các mức Fibonacci
        dataframe = self.detect_fibonacci_levels(dataframe)
        
        # Tính toán độ tin cậy của mỗi tín hiệu khi kết hợp Smart Money với các chỉ báo
        dataframe['entry_score'] = self.calculate_entry_score(dataframe)
        dataframe['exit_score'] = self.calculate_exit_score(dataframe)
        
        return dataframe
    
    def calculate_entry_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Tính toán điểm tin cậy cho tín hiệu vào lệnh (0-100)
        Kết hợp tất cả các yếu tố Smart Money và chỉ báo kỹ thuật
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. Smart Money Factors (60%)
        
        # Order Blocks (15%)
        score += dataframe['bullish_ob'] * 15  # Bullish OB: +15 điểm
        score -= dataframe['bearish_ob'] * 15  # Bearish OB: -15 điểm
        
        # Fair Value Gap (10%)
        score += (dataframe['bullish_fvg'] * dataframe['bullish_fvg_size'] * 1000).clip(0, 10)  # Bullish FVG: +10 điểm tối đa
        score -= (dataframe['bearish_fvg'] * dataframe['bearish_fvg_size'] * 1000).clip(0, 10)  # Bearish FVG: -10 điểm tối đa
        
        # Market Structure (15%)
        score += (dataframe['market_structure'] == 1) * 15  # Uptrend: +15 điểm
        score -= (dataframe['market_structure'] == -1) * 15  # Downtrend: -15 điểm
        
        # Candle Patterns (10%)
        score += (dataframe['engulfing'] == 1) * 10  # Bullish Engulfing: +10 điểm
        score -= (dataframe['engulfing'] == -1) * 10  # Bearish Engulfing: -10 điểm
        score += (dataframe['pinbar'] == 1) * 10  # Bullish Pinbar: +10 điểm
        score -= (dataframe['pinbar'] == -1) * 10  # Bearish Pinbar: -10 điểm
        
        # Equal Highs/Lows (5%)
        score += dataframe['eql'] * 5  # Equal Lows: +5 điểm (tích lũy ở đáy)
        score -= dataframe['eqh'] * 5  # Equal Highs: -5 điểm (tích lũy ở đỉnh)
        
        # Fibonacci Levels (5%)
        score += ((dataframe['near_fib_level'] == 0.618) | (dataframe['near_fib_level'] == 0.5)) * 5  # Gần mức Fib quan trọng: +5 điểm
        
        # 2. Technical Indicators (40%)
        
        # EMA Trend (10%)
        score += ((dataframe['close'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])) * 10  # Uptrend: +10 điểm
        score -= ((dataframe['close'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])) * 10  # Downtrend: -10 điểm
        
        # RSI (10%)
        score += ((dataframe['rsi'] < self.min_rsi.value) & (dataframe['rsi'].shift(1) < dataframe['rsi'])) * 10  # RSI oversold & tăng: +10 điểm
        score -= ((dataframe['rsi'] > self.max_rsi.value) & (dataframe['rsi'].shift(1) > dataframe['rsi'])) * 10  # RSI overbought & giảm: -10 điểm
        
        # MACD (10%)
        score += ((dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macdhist'] > 0) & (dataframe['macdhist'] > dataframe['macdhist'].shift(1))) * 10  # MACD tăng: +10 điểm
        score -= ((dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macdhist'] < 0) & (dataframe['macdhist'] < dataframe['macdhist'].shift(1))) * 10  # MACD giảm: -10 điểm
        
        # Bollinger Bands (10%)
        score += ((dataframe['close'] < dataframe['bb_lowerband']) & (dataframe['close'].shift(1) < dataframe['close'])) * 10  # Giá dưới BB dưới & tăng: +10 điểm
        score -= ((dataframe['close'] > dataframe['bb_upperband']) & (dataframe['close'].shift(1) > dataframe['close'])) * 10  # Giá trên BB trên & giảm: -10 điểm
        
        # Đảm bảo điểm nằm trong khoảng 0-100
        return score.clip(0, 100)
    
    def calculate_exit_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Tính toán điểm tin cậy cho tín hiệu thoát lệnh (0-100)
        Kết hợp tất cả các yếu tố Smart Money và chỉ báo kỹ thuật
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. Smart Money Factors (60%)
        
        # Order Blocks (15%)
        score += dataframe['bearish_ob'] * 15  # Bearish OB: +15 điểm cho thoát lệnh
        score -= dataframe['bullish_ob'] * 15  # Bullish OB: -15 điểm cho thoát lệnh
        
        # Fair Value Gap (10%)
        score += (dataframe['bearish_fvg'] * dataframe['bearish_fvg_size'] * 1000).clip(0, 10)  # Bearish FVG: +10 điểm tối đa
        score -= (dataframe['bullish_fvg'] * dataframe['bullish_fvg_size'] * 1000).clip(0, 10)  # Bullish FVG: -10 điểm tối đa
        
        # Market Structure (15%)
        score += (dataframe['market_structure'] == -1) * 15  # Downtrend: +15 điểm
        score -= (dataframe['market_structure'] == 1) * 15  # Uptrend: -15 điểm
        
        # Candle Patterns (10%)
        score += (dataframe['engulfing'] == -1) * 10  # Bearish Engulfing: +10 điểm
        score -= (dataframe['engulfing'] == 1) * 10  # Bullish Engulfing: -10 điểm
        score += (dataframe['pinbar'] == -1) * 10  # Bearish Pinbar: +10 điểm
        score -= (dataframe['pinbar'] == 1) * 10  # Bullish Pinbar: -10 điểm
        
        # Equal Highs/Lows (5%)
        score += dataframe['eqh'] * 5  # Equal Highs: +5 điểm (tích lũy ở đỉnh)
        score -= dataframe['eql'] * 5  # Equal Lows: -5 điểm (tích lũy ở đáy)
        
        # Fibonacci Levels (5%)
        # Thoát khi giá gần 0.786 hoặc 1.0 (có thể đảo chiều)
        score += ((dataframe['near_fib_level'] == 0.786) | (dataframe['near_fib_level'] == 1.0)) * 5
        
        # 2. Technical Indicators (40%)
        
        # EMA Trend (10%)
        score += ((dataframe['close'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])) * 10  # Downtrend: +10 điểm
        score -= ((dataframe['close'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])) * 10  # Uptrend: -10 điểm
        
        # RSI (10%)
        score += ((dataframe['rsi'] > self.max_rsi.value) & (dataframe['rsi'].shift(1) < dataframe['rsi'])) * 10  # RSI overbought & tăng: +10 điểm
        score -= ((dataframe['rsi'] < self.min_rsi.value) & (dataframe['rsi'].shift(1) > dataframe['rsi'])) * 10  # RSI oversold & giảm: -10 điểm
        
        # MACD (10%)
        score += ((dataframe['macd'] < dataframe['macdsignal']) & (dataframe['macdhist'] < 0) & (dataframe['macdhist'] < dataframe['macdhist'].shift(1))) * 10  # MACD giảm: +10 điểm
        score -= ((dataframe['macd'] > dataframe['macdsignal']) & (dataframe['macdhist'] > 0) & (dataframe['macdhist'] > dataframe['macdhist'].shift(1))) * 10  # MACD tăng: -10 điểm
        
        # Bollinger Bands (10%)
        score += ((dataframe['close'] > dataframe['bb_upperband']) & (dataframe['close'].shift(1) < dataframe['close'])) * 10  # Giá trên BB trên & tăng: +10 điểm
        score -= ((dataframe['close'] < dataframe['bb_lowerband']) & (dataframe['close'].shift(1) > dataframe['close'])) * 10  # Giá dưới BB dưới & giảm: -10 điểm
        
        # Đảm bảo điểm nằm trong khoảng 0-100
        return score.clip(0, 100)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phương thức xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tính toán điểm tin cậy cho từng tín hiệu
        conditions_long = []
        conditions_short = []
        
        # Chỉ vào lệnh khi điểm tin cậy vượt ngưỡng
        conditions_long.append(dataframe['entry_score'] > self.score_threshold.value)
        conditions_short.append(dataframe['exit_score'] > self.score_threshold.value)
        
        # Kết hợp các điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_long), 'enter_long'] = 1
        
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Phương thức xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Tính toán điểm tin cậy cho từng tín hiệu
        conditions_exit_long = []
        conditions_exit_short = []
        
        # Chỉ thoát lệnh khi điểm tin cậy vượt ngưỡng
        conditions_exit_long.append(dataframe['exit_score'] > self.score_threshold.value)
        conditions_exit_short.append(dataframe['entry_score'] > self.score_threshold.value)
        
        # Kết hợp các điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_exit_long), 'exit_long'] = 1
        
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x & y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu và biến động thị trường
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Sử dụng điểm tin cậy để điều chỉnh đòn bẩy
        entry_score = last_candle['entry_score']
        
        # Điều chỉnh đòn bẩy theo độ tin cậy
        # Càng tin cậy cao càng sử dụng đòn bẩy lớn
        if entry_score > 85:  # Tín hiệu rất mạnh
            leverage = min(self.leverage_max, max_leverage)
        elif entry_score > 75:  # Tín hiệu khá mạnh
            leverage = min(6, max_leverage)
        elif entry_score > 65:  # Tín hiệu trung bình
            leverage = min(4, max_leverage)
        else:  # Tín hiệu yếu
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss dựa trên Smart Money Concept và mức lãi/lỗ hiện tại
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Không đủ dữ liệu, sử dụng stoploss mặc định
        if len(dataframe) < 10:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Sử dụng thông tin từ Order Blocks để điều chỉnh stoploss
        # Nếu là Long, đặt stoploss dưới Order Block gần nhất
        if trade.is_short:
            # Với vị thế Short, sử dụng Bearish Order Block làm vùng kháng cự
            # Tìm Bullish Order Block gần nhất để đặt stoploss
            ob_index = dataframe['bullish_ob'].iloc[-20:].idxmax()
            if ob_index and dataframe['bullish_ob'].iloc[ob_index] > 0:
                # Đặt stoploss trên high của Order Block
                ob_high = dataframe['high'].iloc[ob_index]
                dynamic_sl = (ob_high / current_rate - 1) * (1 + 0.005)  # Thêm 0.5% buffer
                return max(self.stoploss, dynamic_sl)
        else:
            # Với vị thế Long, sử dụng Bullish Order Block làm vùng hỗ trợ
            # Tìm Bearish Order Block gần nhất để đặt stoploss
            ob_index = dataframe['bearish_ob'].iloc[-20:].idxmax()
            if ob_index and dataframe['bearish_ob'].iloc[ob_index] > 0:
                # Đặt stoploss dưới low của Order Block
                ob_low = dataframe['low'].iloc[ob_index]
                dynamic_sl = (ob_low / current_rate - 1) * (1 - 0.005)  # Thêm 0.5% buffer
                return min(self.stoploss, dynamic_sl)
        
        # Stoploss dựa trên lợi nhuận hiện tại
        if current_profit >= 0.03:  # 3% lợi nhuận
            return -0.01  # 1% stoploss
        elif current_profit >= 0.02:  # 2% lợi nhuận
            return -0.015  # 1.5% stoploss
        elif current_profit >= 0.01:  # 1% lợi nhuận
            return -0.02  # 2% stoploss
        
        return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận việc vào lệnh với thêm một lớp kiểm tra bổ sung
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Độ tin cậy của tín hiệu
        entry_score = last_candle['entry_score']
        
        # Từ chối lệnh có độ tin cậy thấp
        if entry_score < self.score_threshold.value:
            return False
        
        # Thêm điều kiện bảo vệ dựa trên AI model (nếu có)
        if self.ai_model is not None:
            try:
                # Chuẩn bị dữ liệu cho mô hình AI
                features = self._prepare_features_for_ai(dataframe.iloc[-20:])
                
                # Dự đoán xác suất thành công
                prediction = self.ai_model.predict_proba(features)[0, 1]
                
                # Nếu mô hình AI cho điểm thấp, từ chối lệnh
                if prediction < 0.6:  # Ngưỡng 60%
                    return False
            except Exception as e:
                logger.warning(f"Lỗi khi sử dụng mô hình AI: {e}")
        
        # Đảm bảo không vào lệnh ngược xu hướng
        if side == 'long' and last_candle['market_structure'] == -1:  # Vào Long trong downtrend
            return False
        elif side == 'short' and last_candle['market_structure'] == 1:  # Vào Short trong uptrend
            return False
        
        return True
    
    def _prepare_features_for_ai(self, dataframe: DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho mô hình AI
        """
        # Chọn và chuẩn hóa các đặc trưng
        feature_columns = [
            'rsi', 'macd', 'macdsignal', 'macdhist',
            'ema50', 'ema200', 'sma50', 'sma200',
            'bb_lowerband', 'bb_middleband', 'bb_upperband',
            'bullish_ob', 'bearish_ob', 'bullish_fvg', 'bearish_fvg',
            'market_structure', 'engulfing', 'pinbar', 'eqh', 'eql'
        ]
        
        # Chỉ giữ lại các cột hiện có trong dataframe
        available_features = [col for col in feature_columns if col in dataframe.columns]
        features = dataframe[available_features].copy()
        
        # Giữ chỉ dòng cuối cùng
        features = features.iloc[-1:].reset_index(drop=True)
        
        # Xử lý dữ liệu thiếu
        features = features.fillna(0)
        
        return features
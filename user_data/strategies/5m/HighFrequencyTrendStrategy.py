"""
High Frequency Trend Following Strategy
Chiến lược giao dịch tần suất cao dựa trên xu hướng ngắn hạn, 
kết hợp với phân tích nến và liquidation hunting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, CategoricalParameter, DecimalParameter, IntParameter
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from functools import reduce
from typing import Dict, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)

class HighFrequencyTrendStrategy(IStrategy):
    """
    High Frequency Trend Following + Liquidity Hunting Strategy
    
    Chiến lược này kết hợp:
    1. Trend following (theo dõi xu hướng) ngắn hạn với tần suất cao
    2. Stop hunt & liquidity grab detection (phát hiện khi big players đang hunt stop loss)
    3. Phân tích mẫu hình nến đảo chiều
    4. Bộ lọc thông minh để hạn chế giao dịch giả
    """
    
    # Thông số tối ưu hóa cho chiến lược
    trend_ema1 = IntParameter(3, 15, default=8, space="buy", optimize=True)
    trend_ema2 = IntParameter(15, 50, default=21, space="buy", optimize=True)
    
    rsi_enter_long = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_enter_short = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho scalping
    profit_factor = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    
    # Tham số phát hiện stop hunt
    hunt_threshold = DecimalParameter(0.5, 3.0, default=1.2, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 8
    stoploss = -0.015  # Stoploss thấp hơn cho giao dịch tần suất cao
    
    # Time-in-force và ROI
    timeframe = '5m'
    timeframe_lower = '1m'  # Sử dụng cho high frequency
    
    minimal_roi = {
        "0": 0.02,    # Mục tiêu lãi nhỏ cho giao dịch tần số cao
        "10": 0.015,  # Sau 10 phút
        "20": 0.01,   # Sau 20 phút
        "30": 0.005   # Sau 30 phút
    }
    
    # Trailing stop cho scalping
    trailing_stop = True
    trailing_stop_positive = 0.005  # 0.5%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo hệ thống
    startup_candle_count = 100
    process_only_new_candles = True
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_lower) for pair in pairs]
        return informative_pairs
    
    def detect_liquidity_hunt(self, dataframe: DataFrame, threshold: float = 1.2) -> DataFrame:
        """
        Phát hiện stop hunt / liquidity grab
        Khi giá tạo ra cú lừa giả đột phá hỗ trợ/kháng cự trước khi đảo chiều mạnh
        """
        # Tìm các swing high và swing low (điểm hỗ trợ/kháng cự)
        window = 20
        dataframe['swing_high'] = dataframe['high'].rolling(window=window, center=True).max()
        dataframe['swing_low'] = dataframe['low'].rolling(window=window, center=True).min()
        
        # Xác định vùng hỗ trợ/kháng cự gần nhất
        dataframe['resistance'] = dataframe['swing_high'].rolling(window=5).min()
        dataframe['support'] = dataframe['swing_low'].rolling(window=5).max()
        
        # Phát hiện stop hunt trên (giá giả vờ phá kháng cự rồi quay đầu giảm mạnh)
        dataframe['stop_hunt_up'] = (
            (dataframe['high'] > dataframe['resistance'].shift(1)) &  # Giá vượt kháng cự
            (dataframe['close'] < dataframe['resistance'].shift(1)) &  # Nhưng đóng cửa dưới kháng cự
            (dataframe['close'] < dataframe['open']) &  # Nến giảm
            ((dataframe['high'] - dataframe['close']) > 
             threshold * (dataframe['open'] - dataframe['close']))  # Bóng trên dài
        )
        
        # Phát hiện stop hunt dưới (giá giả vờ phá hỗ trợ rồi quay đầu tăng mạnh)
        dataframe['stop_hunt_down'] = (
            (dataframe['low'] < dataframe['support'].shift(1)) &  # Giá dưới hỗ trợ
            (dataframe['close'] > dataframe['support'].shift(1)) &  # Nhưng đóng cửa trên hỗ trợ
            (dataframe['close'] > dataframe['open']) &  # Nến tăng
            ((dataframe['close'] - dataframe['low']) > 
             threshold * (dataframe['close'] - dataframe['open']))  # Bóng dưới dài
        )
        
        return dataframe
    
    def detect_candle_patterns(self, dataframe: DataFrame) -> DataFrame:
        """
        Phát hiện các mẫu hình nến đảo chiều
        """
        # Bullish Engulfing
        dataframe['engulfing_bullish'] = (
            (dataframe['open'] < dataframe['close']) &  # Nến tăng
            (dataframe['open'].shift(1) > dataframe['close'].shift(1)) &  # Nến trước giảm
            (dataframe['open'] <= dataframe['close'].shift(1)) &  # Mở cửa dưới đóng cửa nến trước
            (dataframe['close'] >= dataframe['open'].shift(1))  # Đóng cửa trên mở cửa nến trước
        )
        
        # Bearish Engulfing
        dataframe['engulfing_bearish'] = (
            (dataframe['open'] > dataframe['close']) &  # Nến giảm
            (dataframe['open'].shift(1) < dataframe['close'].shift(1)) &  # Nến trước tăng
            (dataframe['open'] >= dataframe['close'].shift(1)) &  # Mở cửa trên đóng cửa nến trước
            (dataframe['close'] <= dataframe['open'].shift(1))  # Đóng cửa dưới mở cửa nến trước
        )
        
        # Pinbar / Rejection
        body = abs(dataframe['open'] - dataframe['close'])
        upper_wick = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        lower_wick = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # Bullish Pinbar
        dataframe['pinbar_bullish'] = (
            (lower_wick > 2 * body) &  # Bóng dưới ít nhất gấp 2 lần thân
            (lower_wick > upper_wick * 3) &  # Bóng dưới dài hơn bóng trên ít nhất 3 lần
            (body > 0)  # Có thân nến
        )
        
        # Bearish Pinbar
        dataframe['pinbar_bearish'] = (
            (upper_wick > 2 * body) &  # Bóng trên ít nhất gấp 2 lần thân
            (upper_wick > lower_wick * 3) &  # Bóng trên dài hơn bóng dưới ít nhất 3 lần
            (body > 0)  # Có thân nến
        )
        
        # Inside Bar (nến nằm trong range của nến trước)
        dataframe['inside_bar'] = (
            (dataframe['high'] <= dataframe['high'].shift(1)) &  # High thấp hơn high nến trước
            (dataframe['low'] >= dataframe['low'].shift(1))  # Low cao hơn low nến trước
        )
        
        # Outside Bar (nến có range lớn hơn bao trùm nến trước)
        dataframe['outside_bar'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &  # High cao hơn high nến trước
            (dataframe['low'] < dataframe['low'].shift(1))  # Low thấp hơn low nến trước
        )
        
        # 3 Soldiers (3 nến tăng liên tiếp)
        dataframe['three_soldiers'] = (
            (dataframe['close'] > dataframe['open']) &  # Nến hiện tại tăng
            (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &  # Nến trước tăng
            (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &  # Nến trước nữa tăng
            (dataframe['close'] > dataframe['close'].shift(1)) &  # Đóng cửa cao hơn nến trước
            (dataframe['close'].shift(1) > dataframe['close'].shift(2))  # Nến trước đóng cửa cao hơn nến trước nữa
        )
        
        # 3 Crows (3 nến giảm liên tiếp)
        dataframe['three_crows'] = (
            (dataframe['close'] < dataframe['open']) &  # Nến hiện tại giảm
            (dataframe['close'].shift(1) < dataframe['open'].shift(1)) &  # Nến trước giảm
            (dataframe['close'].shift(2) < dataframe['open'].shift(2)) &  # Nến trước nữa giảm
            (dataframe['close'] < dataframe['close'].shift(1)) &  # Đóng cửa thấp hơn nến trước
            (dataframe['close'].shift(1) < dataframe['close'].shift(2))  # Nến trước đóng cửa thấp hơn nến trước nữa
        )
        
        # Morning Star (mẫu hình 3 nến đảo chiều tăng)
        dataframe['morning_star'] = (
            (dataframe['close'].shift(2) < dataframe['open'].shift(2)) &  # Nến 1 giảm
            (abs(dataframe['close'].shift(1) - dataframe['open'].shift(1)) < 
             0.3 * abs(dataframe['close'].shift(2) - dataframe['open'].shift(2))) &  # Nến 2 có thân nhỏ (doji)
            (dataframe['close'] > dataframe['open']) &  # Nến 3 tăng
            (dataframe['close'] > (dataframe['open'].shift(2) + dataframe['close'].shift(2)) / 2)  # Nến 3 đóng cửa trên giữa nến 1
        )
        
        # Evening Star (mẫu hình 3 nến đảo chiều giảm)
        dataframe['evening_star'] = (
            (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &  # Nến 1 tăng
            (abs(dataframe['close'].shift(1) - dataframe['open'].shift(1)) < 
             0.3 * abs(dataframe['close'].shift(2) - dataframe['open'].shift(2))) &  # Nến 2 có thân nhỏ (doji)
            (dataframe['close'] < dataframe['open']) &  # Nến 3 giảm
            (dataframe['close'] < (dataframe['open'].shift(2) + dataframe['close'].shift(2)) / 2)  # Nến 3 đóng cửa dưới giữa nến 1
        )
        
        return dataframe
    
    def hf_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Thêm các chỉ báo cho high frequency trading
        """
        # EMA ngắn hạn
        for length in [5, 10, 20, 50]:
            dataframe[f'ema_{length}'] = ta.EMA(dataframe, timeperiod=length)
        
        # MACD siêu ngắn
        macd = ta.MACD(dataframe, fastperiod=6, slowperiod=13, signalperiod=4)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # RSI các khung thời gian
        for length in [6, 14]:
            dataframe[f'rsi_{length}'] = ta.RSI(dataframe, timeperiod=length)
        
        # Stochastic siêu ngắn
        stoch = ta.STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3)
        dataframe['stoch_k'] = stoch['slowk']
        dataframe['stoch_d'] = stoch['slowd']
        
        # ATR - Average True Range
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Bollinger Bands siêu ngắn
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
        
        # Tính momentum và volatility
        dataframe['momentum'] = dataframe['close'].pct_change(5) * 100
        dataframe['volatility'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Squeeze Momentum Indicator
        dataframe['squeeze_on'] = (
            (dataframe['bb_upperband'] - dataframe['bb_lowerband']) < 
            (dataframe['atr'] * 1.5)
        )
        
        # Chỉ báo chi phí giao dịch (đánh giá xem biên độ có đủ lớn để bù chi phí không)
        dataframe['trading_cost'] = dataframe['atr'] * 0.1  # Giả sử chi phí = 10% của ATR
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        if not self.dp:
            # Xử lý khi DP không có sẵn (backtest)
            return dataframe
            
        # Lấy dữ liệu từ timeframe thấp hơn (1m)
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], 
            timeframe=self.timeframe_lower
        )
        
        if len(informative) > 0:
            # Thêm chỉ báo cho timeframe thấp hơn
            informative = self.hf_indicators(informative)
            
            # Đổi tên để tránh trùng lặp
            informative_columns = {
                f"inf_{col}": informative[col] for col in informative.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']
            }
            
            # Merge với dataframe gốc
            dataframe = pd.merge(
                dataframe, informative_columns, left_on='date', right_on='date', how='left'
            )
        
        # Các chỉ báo tần suất cao
        dataframe = self.hf_indicators(dataframe)
        
        # Thêm chỉ báo phát hiện stop hunt
        dataframe = self.detect_liquidity_hunt(dataframe, self.hunt_threshold.value)
        
        # Thêm các mẫu hình nến
        dataframe = self.detect_candle_patterns(dataframe)
        
        # Thêm mô hình phát hiện xu hướng
        ema1 = self.trend_ema1.value
        ema2 = self.trend_ema2.value
        
        dataframe[f'ema_{ema1}'] = ta.EMA(dataframe, timeperiod=ema1)
        dataframe[f'ema_{ema2}'] = ta.EMA(dataframe, timeperiod=ema2)
        
        # Xác định xu hướng
        dataframe['uptrend'] = dataframe[f'ema_{ema1}'] > dataframe[f'ema_{ema2}']
        dataframe['downtrend'] = dataframe[f'ema_{ema1}'] < dataframe[f'ema_{ema2}']
        
        # Tính điểm tín hiệu (0-100) cho mỗi tín hiệu
        dataframe['long_signal_score'] = self.calculate_long_signal_score(dataframe)
        dataframe['short_signal_score'] = self.calculate_short_signal_score(dataframe)
        
        return dataframe
    
    def calculate_long_signal_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Tính toán điểm tin cậy cho tín hiệu LONG (0-100)
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. Trend following (30%)
        score += dataframe['uptrend'] * 15  # EMA uptrend
        score += ((dataframe['close'] > dataframe['ema_50']) & 
                  (dataframe['ema_20'] > dataframe['ema_50'])) * 15  # Price > EMA50 & EMA20 > EMA50
        
        # 2. Mẫu hình nến (25%)
        score += dataframe['engulfing_bullish'] * 10
        score += dataframe['pinbar_bullish'] * 10 
        score += dataframe['three_soldiers'] * 5
        score += dataframe['morning_star'] * 15
        score += (dataframe['outside_bar'] & (dataframe['close'] > dataframe['open'])) * 5
        
        # 3. Stop hunt detection (20%)
        score += dataframe['stop_hunt_down'] * 20
        
        # 4. Momentum & Oscillators (15%)
        score += ((dataframe['rsi_14'] < self.rsi_enter_long.value) & 
                 (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1))) * 5  # RSI oversold và đang tăng
        score += ((dataframe['stoch_k'] < 30) & (dataframe['stoch_k'] > dataframe['stoch_d'])) * 5  # Stoch oversold và cắt lên
        score += ((dataframe['macd'] > dataframe['macdsignal']) & 
                 (dataframe['macdhist'] > 0) & 
                 (dataframe['macdhist'] > dataframe['macdhist'].shift(1))) * 5  # MACD tăng
        
        # 5. Volatility (10%)
        score += (dataframe['atr'] > dataframe['atr'].rolling(10).mean()) * 5  # ATR tăng
        score += (dataframe['squeeze_on'].shift(1) & ~dataframe['squeeze_on']) * 5  # Squeeze vừa kết thúc
        
        return score.clip(0, 100)
    
    def calculate_short_signal_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Tính toán điểm tin cậy cho tín hiệu SHORT (0-100)
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. Trend following (30%)
        score += dataframe['downtrend'] * 15  # EMA downtrend
        score += ((dataframe['close'] < dataframe['ema_50']) & 
                  (dataframe['ema_20'] < dataframe['ema_50'])) * 15  # Price < EMA50 & EMA20 < EMA50
        
        # 2. Mẫu hình nến (25%)
        score += dataframe['engulfing_bearish'] * 10
        score += dataframe['pinbar_bearish'] * 10
        score += dataframe['three_crows'] * 5
        score += dataframe['evening_star'] * 15
        score += (dataframe['outside_bar'] & (dataframe['close'] < dataframe['open'])) * 5
        
        # 3. Stop hunt detection (20%)
        score += dataframe['stop_hunt_up'] * 20
        
        # 4. Momentum & Oscillators (15%)
        score += ((dataframe['rsi_14'] > self.rsi_enter_short.value) & 
                 (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1))) * 5  # RSI overbought và đang giảm
        score += ((dataframe['stoch_k'] > 70) & (dataframe['stoch_k'] < dataframe['stoch_d'])) * 5  # Stoch overbought và cắt xuống
        score += ((dataframe['macd'] < dataframe['macdsignal']) & 
                 (dataframe['macdhist'] < 0) & 
                 (dataframe['macdhist'] < dataframe['macdhist'].shift(1))) * 5  # MACD giảm
        
        # 5. Volatility (10%)
        score += (dataframe['atr'] > dataframe['atr'].rolling(10).mean()) * 5  # ATR tăng
        score += (dataframe['squeeze_on'].shift(1) & ~dataframe['squeeze_on']) * 5  # Squeeze vừa kết thúc
        
        return score.clip(0, 100)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh Long
        conditions_long = []
        
        # 1. Signal score cao
        conditions_long.append(dataframe['long_signal_score'] > 70)
        
        # 2. Momentum & Oscillators
        conditions_long.append(
            (dataframe['rsi_14'] < self.rsi_enter_long.value) &  # RSI dưới ngưỡng
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1))  # RSI đang tăng
        )
        
        # 3. EMA Cross
        conditions_long.append(
            (dataframe[f'ema_{self.trend_ema1.value}'] > dataframe[f'ema_{self.trend_ema2.value}']) &
            (dataframe[f'ema_{self.trend_ema1.value}'].shift(1) <= dataframe[f'ema_{self.trend_ema2.value}'].shift(1))
        )
        
        # 4. Stop Hunt Down (tín hiệu liquidity grab dưới)
        conditions_long.append(dataframe['stop_hunt_down'])
        
        # 5. Bullish Patterns
        bullish_patterns = (
            dataframe['engulfing_bullish'] |
            dataframe['pinbar_bullish'] |
            dataframe['morning_star'] |
            dataframe['three_soldiers']
        )
        conditions_long.append(bullish_patterns)
        
        # 6. ATR đủ lớn để bù chi phí giao dịch
        conditions_long.append(dataframe['atr'] > dataframe['trading_cost'] * 5)
        
        # Kết hợp các điều kiện "OR"
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh Short
        conditions_short = []
        
        # 1. Signal score cao
        conditions_short.append(dataframe['short_signal_score'] > 70)
        
        # 2. Momentum & Oscillators
        conditions_short.append(
            (dataframe['rsi_14'] > self.rsi_enter_short.value) &  # RSI trên ngưỡng
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(1))  # RSI đang giảm
        )
        
        # 3. EMA Cross
        conditions_short.append(
            (dataframe[f'ema_{self.trend_ema1.value}'] < dataframe[f'ema_{self.trend_ema2.value}']) &
            (dataframe[f'ema_{self.trend_ema1.value}'].shift(1) >= dataframe[f'ema_{self.trend_ema2.value}'].shift(1))
        )
        
        # 4. Stop Hunt Up (tín hiệu liquidity grab trên)
        conditions_short.append(dataframe['stop_hunt_up'])
        
        # 5. Bearish Patterns
        bearish_patterns = (
            dataframe['engulfing_bearish'] |
            dataframe['pinbar_bearish'] |
            dataframe['evening_star'] |
            dataframe['three_crows']
        )
        conditions_short.append(bearish_patterns)
        
        # 6. ATR đủ lớn để bù chi phí giao dịch
        conditions_short.append(dataframe['atr'] > dataframe['trading_cost'] * 5)
        
        # Kết hợp các điều kiện "OR"
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Take Profit thông minh dựa trên ATR và biến động
        take_profit = dataframe['atr'] * self.profit_factor.value
        conditions_exit_long.append(
            (dataframe['high'] >= dataframe['close'].shift(1) + take_profit)
        )
        
        # 2. EMA Cross
        conditions_exit_long.append(
            (dataframe[f'ema_{self.trend_ema1.value}'] < dataframe[f'ema_{self.trend_ema2.value}']) &
            (dataframe[f'ema_{self.trend_ema1.value}'].shift(1) >= dataframe[f'ema_{self.trend_ema2.value}'].shift(1))
        )
        
        # 3. Đảo chiều giá mạnh
        conditions_exit_long.append(
            (dataframe['close'] < dataframe['low'].shift(1)) &
            (dataframe['close'].shift(1) > dataframe['open'].shift(1))
        )
        
        # 4. Bearish Patterns
        bearish_patterns = (
            dataframe['engulfing_bearish'] |
            dataframe['pinbar_bearish'] |
            dataframe['evening_star'] |
            dataframe['three_crows']
        )
        conditions_exit_long.append(bearish_patterns)
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Take Profit thông minh dựa trên ATR và biến động
        conditions_exit_short.append(
            (dataframe['low'] <= dataframe['close'].shift(1) - take_profit)
        )
        
        # 2. EMA Cross
        conditions_exit_short.append(
            (dataframe[f'ema_{self.trend_ema1.value}'] > dataframe[f'ema_{self.trend_ema2.value}']) &
            (dataframe[f'ema_{self.trend_ema1.value}'].shift(1) <= dataframe[f'ema_{self.trend_ema2.value}'].shift(1))
        )
        
        # 3. Đảo chiều giá mạnh
        conditions_exit_short.append(
            (dataframe['close'] > dataframe['high'].shift(1)) &
            (dataframe['close'].shift(1) < dataframe['open'].shift(1))
        )
        
        # 4. Bullish Patterns
        bullish_patterns = (
            dataframe['engulfing_bullish'] |
            dataframe['pinbar_bullish'] |
            dataframe['morning_star'] |
            dataframe['three_soldiers']
        )
        conditions_exit_short.append(bullish_patterns)
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên khả năng sinh lợi và biến động
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có dữ liệu
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy điểm tín hiệu
        if side == 'long':
            signal_score = last_candle['long_signal_score']
        else:
            signal_score = last_candle['short_signal_score']
        
        # Lấy chỉ số biến động
        volatility = last_candle['volatility']
        
        # Điều chỉnh đòn bẩy dựa trên điểm tín hiệu và biến động
        if signal_score > 85:  # Tín hiệu rất mạnh
            if volatility < 1.5:  # Biến động thấp
                leverage = min(self.leverage_max, max_leverage)
            else:  # Biến động cao
                leverage = min(6, max_leverage)
        elif signal_score > 75:  # Tín hiệu khá mạnh
            if volatility < 1.5:  # Biến động thấp
                leverage = min(6, max_leverage)
            else:  # Biến động cao
                leverage = min(5, max_leverage)
        elif signal_score > 65:  # Tín hiệu trung bình
            leverage = min(4, max_leverage)
        else:  # Tín hiệu yếu
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss dựa trên ATR và giá xác định dựa theo thời gian 
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # ATR-based stoploss
        atr_value = last_candle['atr']
        
        # Tính stoploss dựa trên ATR (1.5x ATR cho vị thế mới)
        atr_stoploss = 1.5 * atr_value / current_rate
        
        # Với lợi nhuận tích cực, giảm dần stoploss
        if current_profit > 0.01:  # > 1%
            # Trailing stop thấp hơn
            return min(self.stoploss, -0.5 * atr_stoploss)
        
        # Mặc định: Sử dụng ATR stoploss
        return -atr_stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận vào lệnh với điều kiện bổ sung
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có dữ liệu
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        
        # 1. Xác nhận dựa trên điểm tin cậy
        if side == 'long' and last_candle['long_signal_score'] < 65:
            return False
        elif side == 'short' and last_candle['short_signal_score'] < 65:
            return False
        
        # 2. Không giao dịch khi biến động quá thấp
        if last_candle['atr'] < last_candle['atr'].rolling(20).mean() * 0.7:
            return False
        
        # 3. Đảm bảo biên độ đủ lớn để bù chi phí
        if last_candle['atr'] < last_candle['trading_cost'] * 3:
            return False
        
        # 4. Không giao dịch khi khối lượng quá thấp
        volume_sma = dataframe['volume'].rolling(20).mean().iloc[-1]
        if last_candle['volume'] < 0.7 * volume_sma:
            return False
        
        return True
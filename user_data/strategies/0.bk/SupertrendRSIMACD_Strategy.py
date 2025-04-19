"""
Supertrend + RSI + MACD Strategy
Chiến lược kết hợp Supertrend với RSI và MACD
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

def supertrend(dataframe: pd.DataFrame, period: int = 7, atr_multiplier: float = 3.0) -> pd.DataFrame:
    """
    Tính toán chỉ báo Supertrend
    
    Tham số:
    - dataframe: DataFrame với dữ liệu OHLC
    - period: Số nến để tính ATR
    - atr_multiplier: Hệ số nhân ATR
    
    Trả về:
    DataFrame với các cột:
    - supertrend: Giá trị Supertrend
    - trend: 1 cho uptrend, -1 cho downtrend
    """
    
    # Tính ATR (Average True Range)
    df = dataframe.copy()
    
    # Tính True Range
    df['tr0'] = abs(df["high"] - df["low"])
    df['tr1'] = abs(df["high"] - df["close"].shift(1))
    df['tr2'] = abs(df["low"] - df["close"].shift(1))
    df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
    
    # ATR: Exponential Moving Average của True Range
    df['atr'] = df['tr'].ewm(alpha=1/period, adjust=False).mean()
    
    # Tính Upper và Lower Bands
    df['upperband'] = (df["high"] + df["low"]) / 2 + (atr_multiplier * df['atr'])
    df['lowerband'] = (df["high"] + df["low"]) / 2 - (atr_multiplier * df['atr'])
    
    # Tính toán Supertrend
    df['in_uptrend'] = True
    
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['upperband'].iloc[i-1]:
            df['in_uptrend'].iloc[i] = True
        elif df['close'].iloc[i] < df['lowerband'].iloc[i-1]:
            df['in_uptrend'].iloc[i] = False
        else:
            df['in_uptrend'].iloc[i] = df['in_uptrend'].iloc[i-1]
            
            if df['in_uptrend'].iloc[i] and df['lowerband'].iloc[i] < df['lowerband'].iloc[i-1]:
                df['lowerband'].iloc[i] = df['lowerband'].iloc[i-1]
            if not df['in_uptrend'].iloc[i] and df['upperband'].iloc[i] > df['upperband'].iloc[i-1]:
                df['upperband'].iloc[i] = df['upperband'].iloc[i-1]
    
    # Giá trị Supertrend
    df['supertrend'] = np.where(df['in_uptrend'], df['lowerband'], df['upperband'])
    df['trend'] = np.where(df['in_uptrend'], 1, -1)
    
    # Xác định các điểm chuyển xu hướng (trend change)
    df['trend_change'] = df['trend'].diff().fillna(0)
    df['trend_change_up'] = (df['trend_change'] > 0).astype(int)
    df['trend_change_down'] = (df['trend_change'] < 0).astype(int)
    
    # Trả về các cột cần thiết
    result_df = pd.DataFrame(index=dataframe.index)
    result_df['supertrend'] = df['supertrend']
    result_df['trend'] = df['trend']
    result_df['trend_change_up'] = df['trend_change_up']
    result_df['trend_change_down'] = df['trend_change_down']
    result_df['upperband'] = df['upperband']
    result_df['lowerband'] = df['lowerband']
    result_df['atr'] = df['atr']
    
    return result_df

class SupertrendRSIMACD_Strategy(IStrategy):
    """
    Chiến lược kết hợp Supertrend, RSI và MACD
    
    Phát hiện trend và reversal dựa trên ba chỉ báo này với các tầng lọc
    - Supertrend: Xác định xu hướng tổng thể
    - RSI: Xác định vùng quá mua/quá bán
    - MACD: Xác nhận momentum và xu hướng
    """
    
    # Tham số cho Supertrend
    supertrend_period = IntParameter(5, 21, default=10, space="buy", optimize=True)
    supertrend_multiplier = DecimalParameter(1.0, 4.0, default=3.0, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho MACD
    macd_fast = IntParameter(8, 16, default=12, space="buy", optimize=True)
    macd_slow = IntParameter(18, 34, default=26, space="buy", optimize=True)
    macd_signal = IntParameter(6, 14, default=9, space="buy", optimize=True)
    
    # Tham số cho xác nhận tín hiệu
    signal_consensus = IntParameter(2, 3, default=2, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.05
    
    # Time-in-force và ROI
    timeframe = '1h'
    
    minimal_roi = {
        "0": 0.06,    # 6%
        "180": 0.04,  # 4% sau 3 giờ
        "300": 0.03,  # 3% sau 5 giờ
        "480": 0.02   # 2% sau 8 giờ
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.03  # 3%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Supertrend
        for period in range(5, 22, 3):  # [5, 8, 11, 14, 17, 20]
            for multiplier in [1.0, 2.0, 3.0, 4.0]:
                st = supertrend(dataframe, period=period, atr_multiplier=multiplier)
                dataframe[f'supertrend_{period}_{multiplier}'] = st['supertrend']
                dataframe[f'supertrend_trend_{period}_{multiplier}'] = st['trend']
                dataframe[f'supertrend_trend_up_{period}_{multiplier}'] = (st['trend'] == 1).astype(int)
                dataframe[f'supertrend_trend_down_{period}_{multiplier}'] = (st['trend'] == -1).astype(int)
                dataframe[f'supertrend_change_up_{period}_{multiplier}'] = st['trend_change_up']
                dataframe[f'supertrend_change_down_{period}_{multiplier}'] = st['trend_change_down']
        
        # RSI
        for period in range(7, 22, 2):  # [7, 9, 11, 13, 15, 17, 19, 21]
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(close=dataframe['close'], window=period)
        
        # MACD
        for fast_period in [8, 12, 16]:
            for slow_period in [18, 26, 34]:
                for signal_period in [6, 9, 12]:
                    macd = ta.trend.MACD(
                        close=dataframe['close'],
                        window_fast=fast_period,
                        window_slow=slow_period,
                        window_sign=signal_period
                    )
                    dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'] = macd.macd()
                    dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}'] = macd.macd_signal()
                    dataframe[f'macdhist_{fast_period}_{slow_period}_{signal_period}'] = macd.macd_diff()
                    
                    # MACD Cross
                    dataframe[f'macd_cross_up_{fast_period}_{slow_period}_{signal_period}'] = (
                        (dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'] > 
                         dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}']) &
                        (dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'].shift(1) <= 
                         dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}'].shift(1))
                    ).astype(int)
                    
                    dataframe[f'macd_cross_down_{fast_period}_{slow_period}_{signal_period}'] = (
                        (dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'] < 
                         dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}']) &
                        (dataframe[f'macd_{fast_period}_{slow_period}_{signal_period}'].shift(1) >= 
                         dataframe[f'macdsignal_{fast_period}_{slow_period}_{signal_period}'].shift(1))
                    ).astype(int)
        
        # ADX - để xác định độ mạnh của xu hướng
        dataframe['adx'] = ta.trend.adx(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            window=14
        )
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Giá trị tham chiếu từ tham số
        st_period = self.supertrend_period.value
        st_multiplier = self.supertrend_multiplier.value
        rsi_period = self.rsi_period.value
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        
        # Tín hiệu consensus - tính điểm tín hiệu (0-3)
        dataframe['long_signal_count'] = (
            dataframe[f'supertrend_trend_up_{st_period}_{st_multiplier}'] +
            (dataframe[f'rsi_{rsi_period}'] < self.rsi_buy_threshold.value).astype(int) +
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] > 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']).astype(int)
        )
        
        dataframe['short_signal_count'] = (
            dataframe[f'supertrend_trend_down_{st_period}_{st_multiplier}'] +
            (dataframe[f'rsi_{rsi_period}'] > self.rsi_sell_threshold.value).astype(int) +
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] < 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']).astype(int)
        )
        
        # Thêm chỉ báo cho reversal
        dataframe['potential_reversal_up'] = (
            (dataframe[f'supertrend_change_up_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] < 40) &
            (dataframe[f'macd_cross_up_{macd_fast}_{macd_slow}_{macd_signal}'] == 1)
        ).astype(int)
        
        dataframe['potential_reversal_down'] = (
            (dataframe[f'supertrend_change_down_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] > 60) &
            (dataframe[f'macd_cross_down_{macd_fast}_{macd_slow}_{macd_signal}'] == 1)
        ).astype(int)
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số
        st_period = self.supertrend_period.value
        st_multiplier = self.supertrend_multiplier.value
        rsi_period = self.rsi_period.value
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        signal_consensus = self.signal_consensus.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Supertrend uptrend + RSI thấp + MACD tăng
        conditions_long.append(
            (dataframe[f'supertrend_trend_up_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] < self.rsi_buy_threshold.value) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] > 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Chuyển xu hướng lên (reversal)
        conditions_long.append(
            (dataframe[f'supertrend_change_up_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] < 45) &
            (dataframe[f'macd_cross_up_{macd_fast}_{macd_slow}_{macd_signal}'] == 1) &
            (dataframe['volume_ratio'] > 1.5)
        )
        
        # 3. Consensus signal
        conditions_long.append(
            (dataframe['long_signal_count'] >= signal_consensus) &
            (dataframe['adx'] > 25) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Supertrend downtrend + RSI cao + MACD giảm
        conditions_short.append(
            (dataframe[f'supertrend_trend_down_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] > self.rsi_sell_threshold.value) &
            (dataframe[f'macd_{macd_fast}_{macd_slow}_{macd_signal}'] < 
             dataframe[f'macdsignal_{macd_fast}_{macd_slow}_{macd_signal}']) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 2. Chuyển xu hướng xuống (reversal)
        conditions_short.append(
            (dataframe[f'supertrend_change_down_{st_period}_{st_multiplier}'] == 1) &
            (dataframe[f'rsi_{rsi_period}'] > 55) &
            (dataframe[f'macd_cross_down_{macd_fast}_{macd_slow}_{macd_signal}'] == 1) &
            (dataframe['volume_ratio'] > 1.5)
        )
        
        # 3. Consensus signal
        conditions_short.append(
            (dataframe['short_signal_count'] >= signal_consensus) &
            (dataframe['adx'] > 25) &
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
        
        # Tham số
        st_period = self.supertrend_period.value
        st_multiplier = self.supertrend_multiplier.value
        rsi_period = self.rsi_period.value
        macd_fast = self.macd_fast.value
        macd_slow = self.macd_slow.value
        macd_signal = self.macd_signal.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Supertrend chuyển xuống
        conditions_exit_long.append(
            dataframe[f'supertrend_change_down_{st_period}_{st_multiplier}'] == 1
        )
        
        # 2. RSI quá cao
        conditions_exit_long.append(
            dataframe[f'rsi_{rsi_period}'] > 80
        )
        
        # 3. MACD cross down
        conditions_exit_long.append(
            dataframe[f'macd_cross_down_{macd_fast}_{macd_slow}_{macd_signal}'] == 1
        )
        
        # 4. Consensus ngược lại
        conditions_exit_long.append(
            dataframe['short_signal_count'] >= 2
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Supertrend chuyển lên
        conditions_exit_short.append(
            dataframe[f'supertrend_change_up_{st_period}_{st_multiplier}'] == 1
        )
        
        # 2. RSI quá thấp
        conditions_exit_short.append(
            dataframe[f'rsi_{rsi_period}'] < 20
        )
        
        # 3. MACD cross up
        conditions_exit_short.append(
            dataframe[f'macd_cross_up_{macd_fast}_{macd_slow}_{macd_signal}'] == 1
        )
        
        # 4. Consensus ngược lại
        conditions_exit_short.append(
            dataframe['long_signal_count'] >= 2
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ mạnh của tín hiệu
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Đánh giá độ mạnh của xu hướng
        adx = last_candle['adx']
        
        # Lấy số lượng tín hiệu consensus
        if side == 'long':
            signal_count = last_candle['long_signal_count']
            potential_reversal = last_candle['potential_reversal_up']
        else:
            signal_count = last_candle['short_signal_count']
            potential_reversal = last_candle['potential_reversal_down']
        
        # Điều chỉnh đòn bẩy dựa trên độ mạnh tín hiệu
        if signal_count == 3 and adx > 30:  # Tín hiệu rất mạnh
            leverage = min(self.leverage_max, max_leverage)
        elif signal_count == 3 or (signal_count == 2 and adx > 25):  # Tín hiệu mạnh
            leverage = min(4, max_leverage)
        elif potential_reversal == 1:  # Tín hiệu reversal
            leverage = min(4, max_leverage)
        else:  # Tín hiệu trung bình
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Supertrend
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số
        st_period = self.supertrend_period.value
        st_multiplier = self.supertrend_multiplier.value
        
        # Stoploss dựa trên band của Supertrend
        # Với lợi nhuận dương, sử dụng supertrend làm stoploss
        if current_profit > 0.02:  # > 2% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại supertrend
                supertrend = last_candle[f'supertrend_{st_period}_{st_multiplier}']
                return (supertrend / current_rate) - 1
            else:
                # Short trade: stoploss tại supertrend
                supertrend = last_candle[f'supertrend_{st_period}_{st_multiplier}']
                return 1 - (supertrend / current_rate)
        
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
        
        # Lọc bổ sung: ADX phải đủ cao để xác nhận xu hướng
        if last_candle['adx'] < 20:
            return False
        
        # Lọc bổ sung: Volume phải đủ lớn
        if last_candle['volume_ratio'] < 0.75:
            return False
        
        # Kiểm tra số lượng tín hiệu
        if side == 'long' and last_candle['long_signal_count'] < self.signal_consensus.value:
            return False
        
        if side == 'short' and last_candle['short_signal_count'] < self.signal_consensus.value:
            return False
        
        return True
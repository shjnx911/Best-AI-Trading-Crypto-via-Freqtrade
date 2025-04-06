"""
Triple EMA + VWAP Strategy
Chiến lược kết hợp Triple EMA và VWAP
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

def triple_ema(dataframe, period):
    """
    Tính Triple EMA
    TEMA = 3 * EMA(period) - 3 * EMA(EMA(period)) + EMA(EMA(EMA(period)))
    """
    # First EMA
    ema1 = ta.trend.ema_indicator(close=dataframe['close'], window=period)
    
    # Second EMA (EMA of first EMA)
    ema2 = ta.trend.ema_indicator(close=ema1, window=period)
    
    # Third EMA (EMA of second EMA)
    ema3 = ta.trend.ema_indicator(close=ema2, window=period)
    
    # Calculate TEMA
    tema = 3 * ema1 - 3 * ema2 + ema3
    
    return tema

def calculate_vwap(dataframe, period=20):
    """
    Tính VWAP (Volume Weighted Average Price)
    """
    # Tạo dataframe để tính toán
    df = dataframe.copy()
    
    # Tính typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Tính volume * typical price
    df['vol_x_price'] = df['typical_price'] * df['volume']
    
    # Tính tổng vol_x_price và volume trong số nến gần nhất
    df['vol_x_price_cumsum'] = df['vol_x_price'].rolling(window=period).sum()
    df['volume_cumsum'] = df['volume'].rolling(window=period).sum()
    
    # Tính VWAP
    df['vwap'] = df['vol_x_price_cumsum'] / df['volume_cumsum']
    
    return df['vwap']

class TEMA_VWAP_Strategy(IStrategy):
    """
    Chiến lược kết hợp Triple EMA và VWAP
    
    - Triple EMA phản ứng nhanh hơn EMA thông thường, tốt cho xu hướng
    - VWAP cung cấp mức hỗ trợ/kháng cự dựa trên khối lượng
    - Kết hợp RSI và Volume để xác nhận tín hiệu
    """
    
    # Tham số cho Triple EMA
    tema_fast = IntParameter(5, 15, default=8, space="buy", optimize=True)
    tema_mid = IntParameter(15, 30, default=21, space="buy", optimize=True)
    tema_slow = IntParameter(30, 60, default=50, space="buy", optimize=True)
    
    # Tham số cho VWAP
    vwap_period = IntParameter(10, 30, default=20, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 5
    stoploss = -0.03
    
    # Time-in-force và ROI
    timeframe = '15m'
    
    minimal_roi = {
        "0": 0.04,     # 4%
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
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Triple EMA
        for period in range(5, 61, 5):
            dataframe[f'tema_{period}'] = triple_ema(dataframe, period)
        
        # VWAP
        for period in range(10, 31, 5):
            dataframe[f'vwap_{period}'] = calculate_vwap(dataframe, period)
        
        # RSI
        for period in range(7, 22):
            dataframe[f'rsi_{period}'] = ta.momentum.rsi(close=dataframe['close'], window=period)
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # ATR
        dataframe['atr'] = ta.volatility.average_true_range(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            window=14
        )
        
        # Tham số
        tema_fast = self.tema_fast.value
        tema_mid = self.tema_mid.value
        tema_slow = self.tema_slow.value
        vwap_period = self.vwap_period.value
        
        # Xác định xu hướng
        dataframe['uptrend'] = (
            (dataframe[f'tema_{tema_fast}'] > dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_mid}'] > dataframe[f'tema_{tema_slow}'])
        )
        
        dataframe['downtrend'] = (
            (dataframe[f'tema_{tema_fast}'] < dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_mid}'] < dataframe[f'tema_{tema_slow}'])
        )
        
        # TEMA Cross
        dataframe['tema_fast_cross_up'] = (
            (dataframe[f'tema_{tema_fast}'] > dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_fast}'].shift(1) <= dataframe[f'tema_{tema_mid}'].shift(1))
        )
        
        dataframe['tema_fast_cross_down'] = (
            (dataframe[f'tema_{tema_fast}'] < dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_fast}'].shift(1) >= dataframe[f'tema_{tema_mid}'].shift(1))
        )
        
        # VWAP Relationship
        dataframe['above_vwap'] = dataframe['close'] > dataframe[f'vwap_{vwap_period}']
        dataframe['below_vwap'] = dataframe['close'] < dataframe[f'vwap_{vwap_period}']
        
        # Price crossing VWAP
        dataframe['cross_vwap_up'] = (
            (dataframe['close'] > dataframe[f'vwap_{vwap_period}']) &
            (dataframe['close'].shift(1) <= dataframe[f'vwap_{vwap_period}'].shift(1))
        )
        
        dataframe['cross_vwap_down'] = (
            (dataframe['close'] < dataframe[f'vwap_{vwap_period}']) &
            (dataframe['close'].shift(1) >= dataframe[f'vwap_{vwap_period}'].shift(1))
        )
        
        # Tín hiệu vào lệnh
        # 1. TEMA bullish setup with VWAP cross
        dataframe['tema_vwap_bullish'] = (
            (dataframe['uptrend']) &
            (dataframe['cross_vwap_up'] | 
             ((dataframe['above_vwap']) & (dataframe['tema_fast_cross_up'])))
        )
        
        # 2. TEMA bearish setup with VWAP cross
        dataframe['tema_vwap_bearish'] = (
            (dataframe['downtrend']) &
            (dataframe['cross_vwap_down'] | 
             ((dataframe['below_vwap']) & (dataframe['tema_fast_cross_down'])))
        )
        
        # 3. VWAP Bounce in trend
        dataframe['vwap_bounce_up'] = (
            (dataframe['uptrend']) &
            (dataframe['close'].shift(1) < dataframe[f'vwap_{vwap_period}'].shift(1)) &
            (dataframe['close'] > dataframe[f'vwap_{vwap_period}']) &
            (dataframe['low'] <= dataframe[f'vwap_{vwap_period}'] * 1.003)  # Touching or slightly below VWAP
        )
        
        dataframe['vwap_bounce_down'] = (
            (dataframe['downtrend']) &
            (dataframe['close'].shift(1) > dataframe[f'vwap_{vwap_period}'].shift(1)) &
            (dataframe['close'] < dataframe[f'vwap_{vwap_period}']) &
            (dataframe['high'] >= dataframe[f'vwap_{vwap_period}'] * 0.997)  # Touching or slightly above VWAP
        )
        
        # 4. TEMA and VWAP convergence
        dataframe['tema_vwap_convergence_bullish'] = (
            (abs(dataframe[f'tema_{tema_fast}'] - dataframe[f'vwap_{vwap_period}']) / 
             dataframe[f'vwap_{vwap_period}'] < 0.002) &  # TEMA and VWAP very close
            (dataframe[f'tema_{tema_fast}'] > dataframe[f'tema_{tema_fast}'].shift(3)) &  # TEMA rising
            (dataframe['volume_ratio'] > 1.2)
        )
        
        dataframe['tema_vwap_convergence_bearish'] = (
            (abs(dataframe[f'tema_{tema_fast}'] - dataframe[f'vwap_{vwap_period}']) / 
             dataframe[f'vwap_{vwap_period}'] < 0.002) &  # TEMA and VWAP very close
            (dataframe[f'tema_{tema_fast}'] < dataframe[f'tema_{tema_fast}'].shift(3)) &  # TEMA falling
            (dataframe['volume_ratio'] > 1.2)
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số
        rsi_period = self.rsi_period.value
        rsi_buy = self.rsi_buy_threshold.value
        rsi_sell = self.rsi_sell_threshold.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. TEMA bullish + VWAP cross-up + RSI không quá cao
        conditions_long.append(
            (dataframe['tema_vwap_bullish']) &
            (dataframe[f'rsi_{rsi_period}'] < 65) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. VWAP bounce-up trong uptrend + RSI oversold
        conditions_long.append(
            (dataframe['vwap_bounce_up']) &
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy + 10) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. TEMA/VWAP convergence bullish
        conditions_long.append(
            (dataframe['tema_vwap_convergence_bullish']) &
            (dataframe[f'rsi_{rsi_period}'] < 60) &
            (dataframe[f'rsi_{rsi_period}'] > 40)  # Không quá thấp hoặc quá cao
        )
        
        # 4. Strong uptrend with RSI reset
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['above_vwap']) &
            (dataframe[f'rsi_{rsi_period}'].shift(1) < 40) &  # RSI vừa ở mức thấp
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1)) &  # RSI tăng
            (dataframe['volume_ratio'] > 1.2)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. TEMA bearish + VWAP cross-down + RSI không quá thấp
        conditions_short.append(
            (dataframe['tema_vwap_bearish']) &
            (dataframe[f'rsi_{rsi_period}'] > 35) &
            (dataframe['volume_ratio'] > 0.8)
        )
        
        # 2. VWAP bounce-down trong downtrend + RSI overbought
        conditions_short.append(
            (dataframe['vwap_bounce_down']) &
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell - 10) &
            (dataframe['volume_ratio'] > 1.0)
        )
        
        # 3. TEMA/VWAP convergence bearish
        conditions_short.append(
            (dataframe['tema_vwap_convergence_bearish']) &
            (dataframe[f'rsi_{rsi_period}'] > 40) &
            (dataframe[f'rsi_{rsi_period}'] < 60)  # Không quá thấp hoặc quá cao
        )
        
        # 4. Strong downtrend with RSI reset
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['below_vwap']) &
            (dataframe[f'rsi_{rsi_period}'].shift(1) > 60) &  # RSI vừa ở mức cao
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1)) &  # RSI giảm
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
        
        # Tham số
        tema_fast = self.tema_fast.value
        tema_mid = self.tema_mid.value
        rsi_period = self.rsi_period.value
        rsi_sell = self.rsi_sell_threshold.value
        rsi_buy = self.rsi_buy_threshold.value
        vwap_period = self.vwap_period.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. TEMA fast cross down
        conditions_exit_long.append(
            (dataframe[f'tema_{tema_fast}'] < dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_fast}'].shift(1) >= dataframe[f'tema_{tema_mid}'].shift(1))
        )
        
        # 2. Cross below VWAP in uptrend
        conditions_exit_long.append(
            (dataframe['cross_vwap_down']) &
            (dataframe['uptrend'])
        )
        
        # 3. RSI overbought
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_period}'] > rsi_sell + 5) &
            (dataframe[f'rsi_{rsi_period}'] < dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang giảm
        )
        
        # 4. Strong volume spike with price rejection
        conditions_exit_long.append(
            (dataframe['volume_ratio'] > 2.5) &
            (dataframe['close'] < dataframe['open']) &  # Nến đỏ
            (dataframe['high'] - dataframe['close'] > 1.5 * (dataframe['close'] - dataframe['low']))  # Bóng trên dài
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. TEMA fast cross up
        conditions_exit_short.append(
            (dataframe[f'tema_{tema_fast}'] > dataframe[f'tema_{tema_mid}']) &
            (dataframe[f'tema_{tema_fast}'].shift(1) <= dataframe[f'tema_{tema_mid}'].shift(1))
        )
        
        # 2. Cross above VWAP in downtrend
        conditions_exit_short.append(
            (dataframe['cross_vwap_up']) &
            (dataframe['downtrend'])
        )
        
        # 3. RSI oversold
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_period}'] < rsi_buy - 5) &
            (dataframe[f'rsi_{rsi_period}'] > dataframe[f'rsi_{rsi_period}'].shift(1))  # RSI đang tăng
        )
        
        # 4. Strong volume spike with price rejection
        conditions_exit_short.append(
            (dataframe['volume_ratio'] > 2.5) &
            (dataframe['close'] > dataframe['open']) &  # Nến xanh
            (dataframe['close'] - dataframe['low'] > 1.5 * (dataframe['high'] - dataframe['close']))  # Bóng dưới dài
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên khoảng cách đến VWAP và mức RSI
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số
        vwap_period = self.vwap_period.value
        rsi_period = self.rsi_period.value
        
        # Tính khoảng cách đến VWAP
        vwap = last_candle[f'vwap_{vwap_period}']
        
        if side == 'long':
            # Tính khoảng cách % từ giá đến VWAP (cho long)
            vwap_distance = (vwap - current_rate) / current_rate * 100  # Âm là tốt cho long (dưới VWAP)
            rsi = last_candle[f'rsi_{rsi_period}']
            
            # Giá dưới VWAP và RSI thấp = đòn bẩy cao hơn
            if vwap_distance > 0.5 and rsi < 40:
                leverage = self.leverage_default  # Không tốt cho long
            elif vwap_distance > 0 and vwap_distance <= 0.5 and rsi < 45:
                leverage = min(3, max_leverage)  # Không tốt lắm cho long
            elif vwap_distance <= 0 and vwap_distance > -0.5 and rsi < 65:
                leverage = min(4, max_leverage)  # Khá tốt cho long
            elif vwap_distance <= -0.5 and rsi < 40:
                leverage = min(self.leverage_max, max_leverage)  # Rất tốt cho long
            else:
                leverage = min(4, max_leverage)  # Trung bình
                
        else:  # short
            # Tính khoảng cách % từ giá đến VWAP (cho short)
            vwap_distance = (current_rate - vwap) / current_rate * 100  # Âm là tốt cho short (trên VWAP)
            rsi = last_candle[f'rsi_{rsi_period}']
            
            # Giá trên VWAP và RSI cao = đòn bẩy cao hơn
            if vwap_distance > 0.5 and rsi > 60:
                leverage = min(self.leverage_max, max_leverage)  # Rất tốt cho short
            elif vwap_distance > 0 and vwap_distance <= 0.5 and rsi > 55:
                leverage = min(4, max_leverage)  # Khá tốt cho short
            elif vwap_distance <= 0 and vwap_distance > -0.5 and rsi > 35:
                leverage = min(3, max_leverage)  # Không tốt lắm cho short
            elif vwap_distance <= -0.5 and rsi > 60:
                leverage = self.leverage_default  # Không tốt cho short
            else:
                leverage = min(4, max_leverage)  # Trung bình
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên TEMA và VWAP
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số
        tema_mid = self.tema_mid.value
        vwap_period = self.vwap_period.value
        
        # Lợi nhuận dương sử dụng TEMA/VWAP làm stoploss
        if current_profit > 0.015:  # > 1.5% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại TEMA mid hoặc VWAP (cái nào gần hơn)
                tema = last_candle[f'tema_{tema_mid}']
                vwap = last_candle[f'vwap_{vwap_period}']
                
                # Chọn giá trị cao hơn giữa tema và vwap (xa giá hiện tại hơn cho short)
                sl_level = max(tema, vwap)
                sl_distance = (sl_level / current_rate) - 1
                
                return max(sl_distance, -0.05)  # Không ít hơn -5%
            else:
                # Short trade: stoploss tại TEMA mid hoặc VWAP (cái nào gần hơn)
                tema = last_candle[f'tema_{tema_mid}']
                vwap = last_candle[f'vwap_{vwap_period}']
                
                # Chọn giá trị thấp hơn giữa tema và vwap (xa giá hiện tại hơn cho short)
                sl_level = min(tema, vwap)
                sl_distance = 1 - (sl_level / current_rate)
                
                return max(sl_distance, -0.05)  # Không ít hơn -5%
        
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
        
        # Tham số
        rsi_period = self.rsi_period.value
        vwap_period = self.vwap_period.value
        
        # Kiểm tra volume
        if last_candle['volume_ratio'] < 0.7:
            return False
        
        # Kiểm tra RSI extreme
        rsi = last_candle[f'rsi_{rsi_period}']
        if side == 'long' and rsi > 75:
            return False
        if side == 'short' and rsi < 25:
            return False
        
        # Kiểm tra khoảng cách đến VWAP quá xa
        vwap = last_candle[f'vwap_{vwap_period}']
        vwap_distance_pct = abs(rate - vwap) / vwap * 100
        
        if vwap_distance_pct > 2.0:  # Khoảng cách quá xa
            return False
        
        return True
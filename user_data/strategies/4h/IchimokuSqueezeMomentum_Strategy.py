"""
Ichimoku Cloud + Squeeze Momentum Strategy
Chiến lược kết hợp Ichimoku Cloud và Squeeze Momentum
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

def squeeze_momentum(dataframe, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5):
    """
    Tính toán chỉ báo Squeeze Momentum (kết hợp BB & KC)
    
    - Bollinger Bands: 20, 2
    - Keltner Channels: 20, 1.5
    - Squeeze xảy ra khi BB nằm trong KC
    """
    # Tính Bollinger Bands
    mid = dataframe['close'].rolling(window=bb_length).mean()
    std = dataframe['close'].rolling(window=bb_length).std()
    
    bb_upperband = mid + bb_mult * std
    bb_lowerband = mid - bb_mult * std
    
    # Tính Keltner Channel
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    tr = pd.DataFrame()
    tr['h-l'] = dataframe['high'] - dataframe['low']
    tr['h-pc'] = abs(dataframe['high'] - dataframe['close'].shift(1))
    tr['l-pc'] = abs(dataframe['low'] - dataframe['close'].shift(1))
    tr['tr'] = tr[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    atr = tr['tr'].rolling(window=kc_length).mean()
    
    kc_upperband = typical_price.rolling(window=kc_length).mean() + atr * kc_mult
    kc_lowerband = typical_price.rolling(window=kc_length).mean() - atr * kc_mult
    
    # Phát hiện squeeze
    squeeze = ((bb_lowerband > kc_lowerband) & (bb_upperband < kc_upperband))
    
    # Tính momentum: sử dụng Linear Regression Slope của 5 nến
    # Sử dụng phương pháp tính momentum đơn giản
    momentum = dataframe['close'] - dataframe['close'].rolling(window=bb_length).mean()
    
    result = pd.DataFrame(index=dataframe.index)
    result['squeeze_on'] = squeeze
    result['squeeze_off'] = ~squeeze
    result['momentum'] = momentum
    
    # Momentum tăng hay giảm
    result['momentum_increasing'] = momentum > momentum.shift(1)
    result['momentum_decreasing'] = momentum < momentum.shift(1)
    
    # Squeeze vừa mới kết thúc (squeeze vừa tắt - thường là tín hiệu giao dịch tốt)
    result['squeeze_fired'] = squeeze.shift(1) & ~squeeze
    
    return result

class IchimokuSqueezeMomentum_Strategy(IStrategy):
    """
    Chiến lược kết hợp Ichimoku Cloud và Squeeze Momentum
    
    - Sử dụng Ichimoku Cloud để xác định xu hướng và hỗ trợ/kháng cự
    - Sử dụng Squeeze Momentum để xác định khi thị trường tích lũy và sắp bùng nổ
    - Tối ưu cho giao dịch khung 4h
    """
    
    # Tham số cho Ichimoku
    tenkan_period = IntParameter(5, 15, default=9, space="buy", optimize=True)
    kijun_period = IntParameter(20, 40, default=26, space="buy", optimize=True)
    senkou_span_b_period = IntParameter(40, 60, default=52, space="buy", optimize=True)
    
    # Tham số cho Squeeze Momentum
    bb_length = IntParameter(15, 25, default=20, space="buy", optimize=True)
    bb_mult = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=True)
    kc_length = IntParameter(15, 25, default=20, space="buy", optimize=True)
    kc_mult = DecimalParameter(1.0, 2.0, default=1.5, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.07
    
    # Time-in-force và ROI
    timeframe = '4h'
    
    minimal_roi = {
        "0": 0.08,     # 8%
        "24": 0.05,    # 5% sau 24 giờ (6 nến 4h)
        "48": 0.03,    # 3% sau 48 giờ (12 nến 4h)
        "72": 0.02     # 2% sau 72 giờ (18 nến 4h)
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.03  # 3%
    trailing_stop_positive_offset = 0.04  # 4%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 200  # Ichimoku cần nhiều nến lịch sử
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # Tính toán Ichimoku Cloud
        for tenkan in range(5, 16):
            for kijun in range(20, 41):
                for senkou_b in range(40, 61):
                    ichimoku = ta.trend.IchimokuIndicator(
                        high=dataframe['high'],
                        low=dataframe['low'],
                        window1=tenkan,
                        window2=kijun,
                        window3=senkou_b
                    )
                    dataframe[f'tenkan_{tenkan}_{kijun}_{senkou_b}'] = ichimoku.ichimoku_conversion_line()
                    dataframe[f'kijun_{tenkan}_{kijun}_{senkou_b}'] = ichimoku.ichimoku_base_line()
                    dataframe[f'senkou_a_{tenkan}_{kijun}_{senkou_b}'] = ichimoku.ichimoku_a()
                    dataframe[f'senkou_b_{tenkan}_{kijun}_{senkou_b}'] = ichimoku.ichimoku_b()
                    dataframe[f'chikou_{tenkan}_{kijun}_{senkou_b}'] = dataframe['close'].shift(-kijun)
                    
                    # Xác định vị trí giá so với cloud
                    dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}'] = (
                        (dataframe['close'] > dataframe[f'senkou_a_{tenkan}_{kijun}_{senkou_b}']) &
                        (dataframe['close'] > dataframe[f'senkou_b_{tenkan}_{kijun}_{senkou_b}'])
                    )
                    dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}'] = (
                        (dataframe['close'] < dataframe[f'senkou_a_{tenkan}_{kijun}_{senkou_b}']) &
                        (dataframe['close'] < dataframe[f'senkou_b_{tenkan}_{kijun}_{senkou_b}'])
                    )
                    dataframe[f'in_cloud_{tenkan}_{kijun}_{senkou_b}'] = (
                        ~dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}'] &
                        ~dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}']
                    )
                    
                    # Xác định TK Cross (Tenkan/Kijun Cross)
                    dataframe[f'tk_cross_up_{tenkan}_{kijun}_{senkou_b}'] = (
                        (dataframe[f'tenkan_{tenkan}_{kijun}_{senkou_b}'] > 
                         dataframe[f'kijun_{tenkan}_{kijun}_{senkou_b}']) &
                        (dataframe[f'tenkan_{tenkan}_{kijun}_{senkou_b}'].shift(1) <= 
                         dataframe[f'kijun_{tenkan}_{kijun}_{senkou_b}'].shift(1))
                    )
                    dataframe[f'tk_cross_down_{tenkan}_{kijun}_{senkou_b}'] = (
                        (dataframe[f'tenkan_{tenkan}_{kijun}_{senkou_b}'] < 
                         dataframe[f'kijun_{tenkan}_{kijun}_{senkou_b}']) &
                        (dataframe[f'tenkan_{tenkan}_{kijun}_{senkou_b}'].shift(1) >= 
                         dataframe[f'kijun_{tenkan}_{kijun}_{senkou_b}'].shift(1))
                    )
        
        # Tính toán Squeeze Momentum
        for bb_length in range(15, 26):
            for bb_mult in [1.5, 2.0, 2.5, 3.0]:
                for kc_length in range(15, 26):
                    for kc_mult in [1.0, 1.5, 2.0]:
                        squeeze = squeeze_momentum(
                            dataframe,
                            bb_length=bb_length,
                            bb_mult=bb_mult,
                            kc_length=kc_length,
                            kc_mult=kc_mult
                        )
                        dataframe[f'squeeze_on_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['squeeze_on']
                        dataframe[f'squeeze_off_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['squeeze_off']
                        dataframe[f'momentum_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['momentum']
                        dataframe[f'momentum_increasing_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['momentum_increasing']
                        dataframe[f'momentum_decreasing_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['momentum_decreasing']
                        dataframe[f'squeeze_fired_{bb_length}_{bb_mult}_{kc_length}_{kc_mult}'] = squeeze['squeeze_fired']
        
        # Tính toán các chỉ báo bổ sung
        # RSI
        dataframe['rsi'] = ta.momentum.rsi(close=dataframe['close'], window=14)
        
        # Volume
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean']
        
        # Params for easy access
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        bb_length = self.bb_length.value
        bb_mult_val = self.bb_mult.value
        kc_length = self.kc_length.value
        kc_mult_val = self.kc_mult.value
        
        # Tính toán tín hiệu tổng hợp
        # Long signal
        dataframe['ichimoku_long'] = (
            (dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'tk_cross_up_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'chikou_{tenkan}_{kijun}_{senkou_b}'] > dataframe['close'].shift(kijun))
        )
        
        dataframe['squeeze_long'] = (
            (dataframe[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] > 0) &
            (dataframe[f'momentum_increasing_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'])
        )
        
        # Short signal
        dataframe['ichimoku_short'] = (
            (dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'tk_cross_down_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'chikou_{tenkan}_{kijun}_{senkou_b}'] < dataframe['close'].shift(kijun))
        )
        
        dataframe['squeeze_short'] = (
            (dataframe[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] < 0) &
            (dataframe[f'momentum_decreasing_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Tham số từ config
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        bb_length = self.bb_length.value
        bb_mult_val = self.bb_mult.value
        kc_length = self.kc_length.value
        kc_mult_val = self.kc_mult.value
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. Ichimoku + Squeeze combo
        conditions_long.append(
            (dataframe['ichimoku_long']) &
            (dataframe['squeeze_long']) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['rsi'] > 40)
        )
        
        # 2. Tín hiệu Ichimoku mạnh
        conditions_long.append(
            (dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'tk_cross_up_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'chikou_{tenkan}_{kijun}_{senkou_b}'] > dataframe['close'].shift(kijun)) &
            (dataframe['close'] > dataframe[f'senkou_a_{tenkan}_{kijun}_{senkou_b}'].shift(kijun)) &
            (dataframe['close'] > dataframe[f'senkou_b_{tenkan}_{kijun}_{senkou_b}'].shift(kijun)) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['rsi'] > 45)
        )
        
        # 3. Squeeze Momentum bùng nổ mạnh
        conditions_long.append(
            (dataframe[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] > 
             dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].rolling(window=5).mean() * 1.5) &
            (dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['rsi'] > 40) &
            (dataframe['rsi'] < 80)
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. Ichimoku + Squeeze combo
        conditions_short.append(
            (dataframe['ichimoku_short']) &
            (dataframe['squeeze_short']) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['rsi'] < 60)
        )
        
        # 2. Tín hiệu Ichimoku mạnh
        conditions_short.append(
            (dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'tk_cross_down_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe[f'chikou_{tenkan}_{kijun}_{senkou_b}'] < dataframe['close'].shift(kijun)) &
            (dataframe['close'] < dataframe[f'senkou_a_{tenkan}_{kijun}_{senkou_b}'].shift(kijun)) &
            (dataframe['close'] < dataframe[f'senkou_b_{tenkan}_{kijun}_{senkou_b}'].shift(kijun)) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['rsi'] < 55)
        )
        
        # 3. Squeeze Momentum bùng nổ mạnh
        conditions_short.append(
            (dataframe[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] < 
             dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].rolling(window=5).mean() * 1.5) &
            (dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}']) &
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['rsi'] < 60) &
            (dataframe['rsi'] > 20)
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Tham số từ config
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        bb_length = self.bb_length.value
        bb_mult_val = self.bb_mult.value
        kc_length = self.kc_length.value
        kc_mult_val = self.kc_mult.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. Ichimoku signal đảo chiều
        conditions_exit_long.append(
            dataframe[f'tk_cross_down_{tenkan}_{kijun}_{senkou_b}']
        )
        
        # 2. Giá dưới cloud
        conditions_exit_long.append(
            dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}'] &
            (dataframe[f'below_cloud_{tenkan}_{kijun}_{senkou_b}'].shift(1) == False)
        )
        
        # 3. Momentum đảo chiều mạnh
        conditions_exit_long.append(
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] < 0) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].shift(1) > 0) &
            (dataframe[f'momentum_decreasing_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'])
        )
        
        # 4. RSI quá cao
        conditions_exit_long.append(
            (dataframe['rsi'] > 80) &
            (dataframe['rsi'].shift(1) <= 80)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. Ichimoku signal đảo chiều
        conditions_exit_short.append(
            dataframe[f'tk_cross_up_{tenkan}_{kijun}_{senkou_b}']
        )
        
        # 2. Giá trên cloud
        conditions_exit_short.append(
            dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}'] &
            (dataframe[f'above_cloud_{tenkan}_{kijun}_{senkou_b}'].shift(1) == False)
        )
        
        # 3. Momentum đảo chiều mạnh
        conditions_exit_short.append(
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'] > 0) &
            (dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].shift(1) < 0) &
            (dataframe[f'momentum_increasing_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'])
        )
        
        # 4. RSI quá thấp
        conditions_exit_short.append(
            (dataframe['rsi'] < 20) &
            (dataframe['rsi'].shift(1) >= 20)
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên vị trí giá so với Ichimoku Cloud
        và độ mạnh của squeeze momentum
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số từ config
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        bb_length = self.bb_length.value
        bb_mult_val = self.bb_mult.value
        kc_length = self.kc_length.value
        kc_mult_val = self.kc_mult.value
        
        # Xác định tín hiệu và vị trí
        if side == 'long':
            # Trên cloud rõ ràng
            if last_candle[f'above_cloud_{tenkan}_{kijun}_{senkou_b}'] == True:
                cloud_position = 1
            # Trong cloud
            elif last_candle[f'in_cloud_{tenkan}_{kijun}_{senkou_b}'] == True:
                cloud_position = 0
            # Dưới cloud
            else:
                cloud_position = -1
            
            # Momentum mạnh
            momentum_value = last_candle[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']
            momentum_mean = dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].rolling(window=5).mean().iloc[-1]
            
            if momentum_value > 0 and momentum_value > momentum_mean * 2:
                momentum_strength = 2  # Rất mạnh
            elif momentum_value > 0 and momentum_value > momentum_mean:
                momentum_strength = 1  # Mạnh
            else:
                momentum_strength = 0  # Bình thường
            
            # Squeeze vừa kết thúc
            squeeze_fired = last_candle[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']
            
            # Tính toán đòn bẩy dựa trên các yếu tố
            if cloud_position == 1 and momentum_strength == 2 and squeeze_fired:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif cloud_position == 1 and momentum_strength >= 1:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif cloud_position >= 0 and momentum_strength >= 1:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        else:  # Short
            # Dưới cloud rõ ràng
            if last_candle[f'below_cloud_{tenkan}_{kijun}_{senkou_b}'] == True:
                cloud_position = 1
            # Trong cloud
            elif last_candle[f'in_cloud_{tenkan}_{kijun}_{senkou_b}'] == True:
                cloud_position = 0
            # Trên cloud
            else:
                cloud_position = -1
            
            # Momentum mạnh
            momentum_value = last_candle[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']
            momentum_mean = dataframe[f'momentum_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}'].rolling(window=5).mean().iloc[-1]
            
            if momentum_value < 0 and momentum_value < momentum_mean * 2:
                momentum_strength = 2  # Rất mạnh
            elif momentum_value < 0 and momentum_value < momentum_mean:
                momentum_strength = 1  # Mạnh
            else:
                momentum_strength = 0  # Bình thường
            
            # Squeeze vừa kết thúc
            squeeze_fired = last_candle[f'squeeze_fired_{bb_length}_{bb_mult_val}_{kc_length}_{kc_mult_val}']
            
            # Tính toán đòn bẩy dựa trên các yếu tố
            if cloud_position == 1 and momentum_strength == 2 and squeeze_fired:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif cloud_position == 1 and momentum_strength >= 1:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif cloud_position >= 0 and momentum_strength >= 1:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên Ichimoku Cloud
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Tham số từ config
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        
        # Stoploss dựa trên Kijun Sen (đường cơ sở)
        if current_profit > 0.03:  # > 3% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại Kijun Sen
                kijun_sen = last_candle[f'kijun_{tenkan}_{kijun}_{senkou_b}']
                return (kijun_sen / current_rate) - 1
            else:
                # Short trade: stoploss tại Kijun Sen
                kijun_sen = last_candle[f'kijun_{tenkan}_{kijun}_{senkou_b}']
                return 1 - (kijun_sen / current_rate)
        
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
        
        # Tham số từ config
        tenkan = self.tenkan_period.value
        kijun = self.kijun_period.value
        senkou_b = self.senkou_span_b_period.value
        bb_length = self.bb_length.value
        bb_mult_val = self.bb_mult.value
        kc_length = self.kc_length.value
        kc_mult_val = self.kc_mult.value
        
        # Không vào lệnh khi RSI extreme
        if side == 'long' and last_candle['rsi'] > 85:
            return False
            
        if side == 'short' and last_candle['rsi'] < 15:
            return False
        
        # Không vào lệnh khi chikou span không thỏa mãn
        if side == 'long' and last_candle[f'chikou_{tenkan}_{kijun}_{senkou_b}'] <= last_candle['close'].shift(kijun):
            return False
            
        if side == 'short' and last_candle[f'chikou_{tenkan}_{kijun}_{senkou_b}'] >= last_candle['close'].shift(kijun):
            return False
        
        # Không vào lệnh khi volume quá thấp
        if last_candle['volume_ratio'] < 0.5:
            return False
        
        return True
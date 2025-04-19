"""
ADX Momentum Volume Strategy
Chiến lược kết hợp ADX, chỉ báo Momentum và phân tích khối lượng
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

class ADXMomentumVolume_Strategy(IStrategy):
    """
    Chiến lược kết hợp ADX, Momentum và Volume
    
    - Sử dụng ADX để xác định sức mạnh xu hướng
    - Kết hợp với các chỉ báo momentum (RSI, TSI, MFI)
    - Phân tích khối lượng giao dịch để xác nhận tín hiệu
    - Tối ưu cho giao dịch tầm ngắn đến trung bình (15m)
    """
    
    # Tham số cho ADX
    adx_length = IntParameter(10, 30, default=14, space="buy", optimize=True)
    adx_threshold = IntParameter(20, 40, default=25, space="buy", optimize=True)
    
    # Tham số cho DMI (Directional Movement Index)
    dmi_length = IntParameter(10, 30, default=14, space="buy", optimize=True)
    dmi_threshold = DecimalParameter(0.1, 0.5, default=0.2, space="buy", optimize=True)
    
    # Tham số cho Momentum
    rsi_length = IntParameter(7, 21, default=14, space="buy", optimize=True)
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    tsi_long = IntParameter(8, 25, default=13, space="buy", optimize=True)
    tsi_short = IntParameter(1, 7, default=2, space="buy", optimize=True)
    tsi_signal = IntParameter(7, 15, default=9, space="buy", optimize=True)
    
    # Tham số cho Volume
    volume_ma_length = IntParameter(10, 50, default=20, space="buy", optimize=True)
    volume_threshold = DecimalParameter(1.0, 3.0, default=1.5, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 6
    stoploss = -0.028
    
    # Time-in-force và ROI
    timeframe = '15m'
    
    minimal_roi = {
        "0": 0.04,    # 4%
        "30": 0.03,   # 3% sau 30 phút
        "60": 0.02,   # 2% sau 60 phút
        "90": 0.01    # 1% sau 90 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.008  # 0.8%
    trailing_stop_positive_offset = 0.01  # 1%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo
    startup_candle_count = 100
    process_only_new_candles = True
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        
        # ADX (Average Directional Index)
        for length in range(10, 31, 2):
            dataframe[f'adx_{length}'] = ta.trend.adx(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=length
            )
            
            # DMI (Directional Movement Index)
            dataframe[f'plus_di_{length}'] = ta.trend.adx_pos(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=length
            )
            
            dataframe[f'minus_di_{length}'] = ta.trend.adx_neg(
                high=dataframe['high'],
                low=dataframe['low'],
                close=dataframe['close'],
                window=length
            )
            
            # DI Difference and Ratio
            dataframe[f'di_diff_{length}'] = abs(dataframe[f'plus_di_{length}'] - dataframe[f'minus_di_{length}'])
            dataframe[f'di_ratio_{length}'] = np.where(
                dataframe[f'minus_di_{length}'] != 0,
                dataframe[f'plus_di_{length}'] / dataframe[f'minus_di_{length}'],
                999  # Avoid division by zero
            )
        
        # RSI (Relative Strength Index)
        for length in range(7, 22):
            dataframe[f'rsi_{length}'] = ta.momentum.rsi(close=dataframe['close'], window=length)
        
        # TSI (True Strength Index)
        for long_length in [8, 13, 21, 25]:
            for short_length in [1, 2, 4, 7]:
                for signal_length in [7, 9, 13, 15]:
                    # Tính PC (Price Change)
                    pc = dataframe['close'].diff()
                    
                    # Smooth PC lần thứ nhất với EMA dài
                    pc_ema_long = ta.trend.ema_indicator(pc, window=long_length)
                    
                    # Smooth lần thứ hai với EMA ngắn
                    pc_ema_smooth = ta.trend.ema_indicator(pc_ema_long, window=short_length)
                    
                    # Tính APC (Absolute Price Change)
                    apc = abs(dataframe['close'].diff())
                    
                    # Smooth APC lần thứ nhất với EMA dài
                    apc_ema_long = ta.trend.ema_indicator(apc, window=long_length)
                    
                    # Smooth lần thứ hai với EMA ngắn
                    apc_ema_smooth = ta.trend.ema_indicator(apc_ema_long, window=short_length)
                    
                    # Tính TSI
                    tsi_value = 100 * (pc_ema_smooth / apc_ema_smooth)
                    dataframe[f'tsi_{long_length}_{short_length}'] = tsi_value
                    
                    # Tính Signal Line
                    dataframe[f'tsi_signal_{long_length}_{short_length}_{signal_length}'] = ta.trend.ema_indicator(
                        tsi_value, window=signal_length
                    )
                    
                    # Tính Histogram
                    dataframe[f'tsi_hist_{long_length}_{short_length}_{signal_length}'] = (
                        dataframe[f'tsi_{long_length}_{short_length}'] - 
                        dataframe[f'tsi_signal_{long_length}_{short_length}_{signal_length}']
                    )
        
        # MFI (Money Flow Index)
        dataframe['mfi'] = ta.volume.money_flow_index(
            high=dataframe['high'],
            low=dataframe['low'],
            close=dataframe['close'],
            volume=dataframe['volume'],
            window=14
        )
        
        # Volume Analysis
        for length in [10, 20, 30, 50]:
            # Volume Moving Average
            dataframe[f'volume_ma_{length}'] = ta.trend.sma_indicator(dataframe['volume'], window=length)
            
            # Volume Ratio
            dataframe[f'volume_ratio_{length}'] = dataframe['volume'] / dataframe[f'volume_ma_{length}']
            
            # On-Balance Volume (OBV)
            dataframe['obv'] = ta.volume.on_balance_volume(
                close=dataframe['close'],
                volume=dataframe['volume']
            )
            
            # OBV Moving Average
            dataframe[f'obv_ma_{length}'] = ta.trend.sma_indicator(dataframe['obv'], window=length)
        
        # Price Moving Averages
        for length in [8, 13, 21, 55, 89]:
            dataframe[f'ema_{length}'] = ta.trend.ema_indicator(close=dataframe['close'], window=length)
            dataframe[f'sma_{length}'] = ta.trend.sma_indicator(close=dataframe['close'], window=length)
        
        # Lấy các giá trị từ tham số
        adx_length = self.adx_length.value
        dmi_length = self.dmi_length.value
        rsi_length = self.rsi_length.value
        tsi_long = self.tsi_long.value
        tsi_short = self.tsi_short.value
        tsi_signal = self.tsi_signal.value
        volume_ma_length = self.volume_ma_length.value
        
        # Xác định xu hướng
        dataframe['uptrend'] = (
            (dataframe[f'plus_di_{dmi_length}'] > dataframe[f'minus_di_{dmi_length}']) &
            (dataframe[f'di_diff_{dmi_length}'] > self.dmi_threshold.value * 100) &
            (dataframe[f'adx_{adx_length}'] > self.adx_threshold.value) &
            (dataframe['ema_8'] > dataframe['ema_21'])
        )
        
        dataframe['downtrend'] = (
            (dataframe[f'minus_di_{dmi_length}'] > dataframe[f'plus_di_{dmi_length}']) &
            (dataframe[f'di_diff_{dmi_length}'] > self.dmi_threshold.value * 100) &
            (dataframe[f'adx_{adx_length}'] > self.adx_threshold.value) &
            (dataframe['ema_8'] < dataframe['ema_21'])
        )
        
        # Tín hiệu Momentum
        # RSI Divergence
        dataframe['rsi_uptrend'] = (
            (dataframe[f'rsi_{rsi_length}'] > dataframe[f'rsi_{rsi_length}'].shift(1)) &
            (dataframe[f'rsi_{rsi_length}'].shift(1) > dataframe[f'rsi_{rsi_length}'].shift(2))
        )
        
        dataframe['rsi_downtrend'] = (
            (dataframe[f'rsi_{rsi_length}'] < dataframe[f'rsi_{rsi_length}'].shift(1)) &
            (dataframe[f'rsi_{rsi_length}'].shift(1) < dataframe[f'rsi_{rsi_length}'].shift(2))
        )
        
        # TSI Cross
        dataframe['tsi_cross_up'] = (
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] > dataframe[f'tsi_signal_{tsi_long}_{tsi_short}_{tsi_signal}']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'].shift(1) <= dataframe[f'tsi_signal_{tsi_long}_{tsi_short}_{tsi_signal}'].shift(1))
        )
        
        dataframe['tsi_cross_down'] = (
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] < dataframe[f'tsi_signal_{tsi_long}_{tsi_short}_{tsi_signal}']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'].shift(1) >= dataframe[f'tsi_signal_{tsi_long}_{tsi_short}_{tsi_signal}'].shift(1))
        )
        
        # Volume Surge
        dataframe['volume_surge'] = dataframe[f'volume_ratio_{volume_ma_length}'] > self.volume_threshold.value
        
        # Strong ADX
        dataframe['strong_adx'] = dataframe[f'adx_{adx_length}'] > self.adx_threshold.value
        
        # Long Entry Signals
        dataframe['entry_long_rsi_adx'] = (
            (dataframe[f'rsi_{rsi_length}'] < self.rsi_buy_threshold.value) &
            (dataframe['rsi_uptrend']) &
            (dataframe['uptrend']) &
            (dataframe['strong_adx'])
        )
        
        dataframe['entry_long_tsi_volume'] = (
            (dataframe['tsi_cross_up']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] < 0) &  # TSI crossing up while below zero
            (dataframe['volume_surge']) &
            (dataframe['uptrend'])
        )
        
        dataframe['entry_long_dmi_mfi'] = (
            (dataframe[f'plus_di_{dmi_length}'].shift(1) < dataframe[f'minus_di_{dmi_length}'].shift(1)) &
            (dataframe[f'plus_di_{dmi_length}'] > dataframe[f'minus_di_{dmi_length}']) &  # DMI cross
            (dataframe['mfi'] < 40) &
            (dataframe['strong_adx'])
        )
        
        # Short Entry Signals
        dataframe['entry_short_rsi_adx'] = (
            (dataframe[f'rsi_{rsi_length}'] > self.rsi_sell_threshold.value) &
            (dataframe['rsi_downtrend']) &
            (dataframe['downtrend']) &
            (dataframe['strong_adx'])
        )
        
        dataframe['entry_short_tsi_volume'] = (
            (dataframe['tsi_cross_down']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] > 0) &  # TSI crossing down while above zero
            (dataframe['volume_surge']) &
            (dataframe['downtrend'])
        )
        
        dataframe['entry_short_dmi_mfi'] = (
            (dataframe[f'minus_di_{dmi_length}'].shift(1) < dataframe[f'plus_di_{dmi_length}'].shift(1)) &
            (dataframe[f'minus_di_{dmi_length}'] > dataframe[f'plus_di_{dmi_length}']) &  # DMI cross
            (dataframe['mfi'] > 60) &
            (dataframe['strong_adx'])
        )
        
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh LONG
        conditions_long = []
        
        # 1. RSI oversold + ADX strong + Uptrend
        conditions_long.append(
            (dataframe['entry_long_rsi_adx']) &
            (dataframe['volume_surge'])
        )
        
        # 2. TSI cross up + Volume surge
        conditions_long.append(
            (dataframe['entry_long_tsi_volume'])
        )
        
        # 3. DMI cross + MFI < 40
        conditions_long.append(
            (dataframe['entry_long_dmi_mfi']) &
            (dataframe['volume_surge'])
        )
        
        # 4. OBV trend confirmation
        conditions_long.append(
            (dataframe['uptrend']) &
            (dataframe['obv'] > dataframe['obv_ma_20']) &
            (dataframe['strong_adx']) &
            (dataframe[f'rsi_{self.rsi_length.value}'] < 60) &
            (dataframe['volume_surge'])
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh SHORT
        conditions_short = []
        
        # 1. RSI overbought + ADX strong + Downtrend
        conditions_short.append(
            (dataframe['entry_short_rsi_adx']) &
            (dataframe['volume_surge'])
        )
        
        # 2. TSI cross down + Volume surge
        conditions_short.append(
            (dataframe['entry_short_tsi_volume'])
        )
        
        # 3. DMI cross + MFI > 60
        conditions_short.append(
            (dataframe['entry_short_dmi_mfi']) &
            (dataframe['volume_surge'])
        )
        
        # 4. OBV trend confirmation
        conditions_short.append(
            (dataframe['downtrend']) &
            (dataframe['obv'] < dataframe['obv_ma_20']) &
            (dataframe['strong_adx']) &
            (dataframe[f'rsi_{self.rsi_length.value}'] > 40) &
            (dataframe['volume_surge'])
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
        adx_length = self.adx_length.value
        dmi_length = self.dmi_length.value
        rsi_length = self.rsi_length.value
        tsi_long = self.tsi_long.value
        tsi_short = self.tsi_short.value
        tsi_signal = self.tsi_signal.value
        
        # Điều kiện thoát lệnh LONG
        conditions_exit_long = []
        
        # 1. ADX giảm mạnh
        conditions_exit_long.append(
            (dataframe[f'adx_{adx_length}'] < self.adx_threshold.value) &
            (dataframe[f'adx_{adx_length}'].shift(1) > self.adx_threshold.value)
        )
        
        # 2. DMI bearish cross
        conditions_exit_long.append(
            (dataframe[f'plus_di_{dmi_length}'] < dataframe[f'minus_di_{dmi_length}']) &
            (dataframe[f'plus_di_{dmi_length}'].shift(1) > dataframe[f'minus_di_{dmi_length}'].shift(1))
        )
        
        # 3. RSI overbought
        conditions_exit_long.append(
            (dataframe[f'rsi_{rsi_length}'] > self.rsi_sell_threshold.value)
        )
        
        # 4. TSI cross down
        conditions_exit_long.append(
            (dataframe['tsi_cross_down']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] > 0.2)  # Only when TSI is high
        )
        
        # 5. EMA cross
        conditions_exit_long.append(
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['ema_8'].shift(1) > dataframe['ema_21'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Điều kiện thoát lệnh SHORT
        conditions_exit_short = []
        
        # 1. ADX giảm mạnh
        conditions_exit_short.append(
            (dataframe[f'adx_{adx_length}'] < self.adx_threshold.value) &
            (dataframe[f'adx_{adx_length}'].shift(1) > self.adx_threshold.value)
        )
        
        # 2. DMI bullish cross
        conditions_exit_short.append(
            (dataframe[f'minus_di_{dmi_length}'] < dataframe[f'plus_di_{dmi_length}']) &
            (dataframe[f'minus_di_{dmi_length}'].shift(1) > dataframe[f'plus_di_{dmi_length}'].shift(1))
        )
        
        # 3. RSI oversold
        conditions_exit_short.append(
            (dataframe[f'rsi_{rsi_length}'] < self.rsi_buy_threshold.value)
        )
        
        # 4. TSI cross up
        conditions_exit_short.append(
            (dataframe['tsi_cross_up']) &
            (dataframe[f'tsi_{tsi_long}_{tsi_short}'] < -0.2)  # Only when TSI is low
        )
        
        # 5. EMA cross
        conditions_exit_short.append(
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['ema_8'].shift(1) < dataframe['ema_21'].shift(1))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên độ mạnh của ADX và tín hiệu
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy thông tin ADX
        adx_value = last_candle[f'adx_{self.adx_length.value}']
        
        # Lấy thông tin DI
        plus_di = last_candle[f'plus_di_{self.dmi_length.value}']
        minus_di = last_candle[f'minus_di_{self.dmi_length.value}']
        di_diff = abs(plus_di - minus_di)
        
        # Lấy thông tin khối lượng
        volume_ratio = last_candle[f'volume_ratio_{self.volume_ma_length.value}']
        
        # Điều chỉnh đòn bẩy dựa trên độ mạnh của tín hiệu
        if side == 'long':
            # ADX mạnh, DI+ cao và khối lượng lớn
            if adx_value > 40 and plus_di > minus_di * 1.5 and volume_ratio > 2.0:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif adx_value > 30 and plus_di > minus_di and volume_ratio > 1.5:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif adx_value > self.adx_threshold.value and plus_di > minus_di:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
                
        else:  # short
            # ADX mạnh, DI- cao và khối lượng lớn
            if adx_value > 40 and minus_di > plus_di * 1.5 and volume_ratio > 2.0:
                leverage = min(self.leverage_max, max_leverage)  # Đòn bẩy tối đa
            elif adx_value > 30 and minus_di > plus_di and volume_ratio > 1.5:
                leverage = min(5, max_leverage)  # Đòn bẩy cao
            elif adx_value > self.adx_threshold.value and minus_di > plus_di:
                leverage = min(4, max_leverage)  # Đòn bẩy trung bình cao
            else:
                leverage = self.leverage_default  # Đòn bẩy mặc định
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên ATR và mức giá trung bình khi ADX yếu
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận dương, sử dụng EMA làm stoploss
        if current_profit > 0.015:  # > 1.5% lợi nhuận
            if not trade.is_short:
                # Long trade: stoploss tại EMA 21
                ema21 = last_candle['ema_21']
                sl_distance = (ema21 / current_rate) - 1
                return max(sl_distance, -0.05)  # Không ít hơn -5%
            else:
                # Short trade: stoploss tại EMA 21
                ema21 = last_candle['ema_21']
                sl_distance = 1 - (ema21 / current_rate)
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
        
        # Không vào lệnh khi volume quá thấp
        if last_candle[f'volume_ratio_{self.volume_ma_length.value}'] < 0.8:
            return False
        
        # Không vào lệnh khi ADX quá thấp
        if last_candle[f'adx_{self.adx_length.value}'] < 20:
            return False
        
        # Không vào lệnh khi xu hướng DI yếu
        if side == 'long' and last_candle[f'plus_di_{self.dmi_length.value}'] < last_candle[f'minus_di_{self.dmi_length.value}']:
            return False
            
        if side == 'short' and last_candle[f'minus_di_{self.dmi_length.value}'] < last_candle[f'plus_di_{self.dmi_length.value}']:
            return False
        
        return True
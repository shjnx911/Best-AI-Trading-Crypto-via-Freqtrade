"""
Breakout Trading Strategy kết hợp với VWAP Mean Reversion
Phát hiện và giao dịch các breakout có volume cao và sử dụng VWAP làm điểm tham chiếu
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

def calculate_vwap(dataframe):
    """Tính Volume Weighted Average Price (VWAP)"""
    v = dataframe['volume'].values
    tp = (dataframe['high'] + dataframe['low'] + dataframe['close']).values / 3
    return pd.Series(data=np.cumsum(tp * v) / np.cumsum(v), index=dataframe.index)

class Breakout_VWAP_Strategy(IStrategy):
    """
    Chiến lược Breakout kết hợp với VWAP Mean Reversion
    
    - Phát hiện breakout từ các mô hình tích lũy (Darvas Box)
    - Xác thực breakout bằng volume
    - Sử dụng VWAP để xác định mean reversion
    - Kết hợp Fibonacci để xác định mục tiêu lợi nhuận
    """
    
    # Tham số tối ưu hóa cho chiến lược
    darvas_lookback = IntParameter(5, 30, default=15, space="buy", optimize=True)
    darvas_threshold = DecimalParameter(0.5, 3.0, default=1.0, space="buy", optimize=True)
    volume_threshold = IntParameter(150, 300, default=200, space="buy", optimize=True)
    vwap_period = IntParameter(10, 50, default=20, space="buy", optimize=True)
    vwap_deviation = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True)
    breakout_trend_period = IntParameter(5, 20, default=10, space="buy", optimize=True)
    
    # Fibonacci retracement levels
    fib_retracement = CategoricalParameter([0.236, 0.382, 0.5, 0.618, 0.786], default=0.618, space="sell", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 8
    stoploss = -0.05  # Stoploss cao hơn vì là chiến lược breakout
    
    # ROI và trailing stop
    minimal_roi = {
        "0": 0.10,  # 10% lợi nhuận là mục tiêu
        "30": 0.05,  # Sau 30 phút, mục tiêu giảm xuống 5%
        "60": 0.025,  # Sau 60 phút, mục tiêu giảm xuống 2.5%
        "120": 0.01   # Sau 120 phút, mục tiêu giảm xuống 1%
    }
    
    trailing_stop = True
    trailing_stop_positive = 0.02  # 2%
    trailing_stop_positive_offset = 0.03  # 3%
    trailing_only_offset_is_reached = True
    
    # Khung thời gian và khởi tạo
    timeframe = '5m'
    timeframe_higher = '1h'
    startup_candle_count = 200
    process_only_new_candles = True
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_higher) for pair in pairs]
        return informative_pairs
    
    def get_darvas_boxes(self, dataframe: DataFrame, lookback: int, threshold: float) -> DataFrame:
        """
        Tạo Darvas Boxes để xác định các vùng tích lũy
        - lookback: số nến để tìm high/low trước đó
        - threshold: % chênh lệch để xác nhận box đã bị phá vỡ
        """
        dataframe['darvas_high'] = dataframe['high'].rolling(lookback).max()
        dataframe['darvas_low'] = dataframe['low'].rolling(lookback).min()
        
        # Phát hiện vùng tích lũy (box)
        dataframe['in_box'] = (
            (dataframe['high'] <= dataframe['darvas_high'].shift(1) * 1.005) &  # Giá không vượt high trước đó quá 0.5%
            (dataframe['low'] >= dataframe['darvas_low'].shift(1) * 0.995)  # Giá không dưới low trước đó quá 0.5%
        )
        
        # Đếm số nến liên tiếp trong vùng tích lũy
        dataframe['box_count'] = 0
        
        # Tính số nến trong box
        box_count = 0
        for i in range(1, len(dataframe)):
            if dataframe['in_box'].iloc[i]:
                box_count += 1
            else:
                box_count = 0
            dataframe['box_count'].iloc[i] = box_count
        
        # Phát hiện breakout (giá phá vỡ vùng tích lũy)
        dataframe['box_breakout_up'] = (
            (dataframe['close'] > dataframe['darvas_high'].shift(1) * (1 + threshold/100)) &  # Giá đóng cửa vượt high của box
            (dataframe['box_count'].shift(1) >= 5) &  # Đã có ít nhất 5 nến trong box
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5)  # Volume tăng
        )
        
        dataframe['box_breakout_down'] = (
            (dataframe['close'] < dataframe['darvas_low'].shift(1) * (1 - threshold/100)) &  # Giá đóng cửa dưới low của box
            (dataframe['box_count'].shift(1) >= 5) &  # Đã có ít nhất 5 nến trong box
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5)  # Volume tăng
        )
        
        return dataframe
    
    def detect_vwap_deviation(self, dataframe: DataFrame, period: int, deviation_pct: float) -> DataFrame:
        """
        Phát hiện độ lệch giá so với VWAP
        - period: số nến để tính VWAP
        - deviation_pct: % độ lệch cho phép
        """
        # Tính VWAP cho mỗi phiên (reset theo ngày)
        dataframe['date_str'] = pd.to_datetime(dataframe['date']).dt.date.astype(str)
        groups = dataframe.groupby('date_str')
        
        vwap_values = []
        for _, group in groups:
            v = group['volume'].values
            tp = (group['high'] + group['low'] + group['close']).values / 3
            group_vwap = np.cumsum(tp * v) / np.cumsum(v)
            vwap_values.extend(group_vwap)
        
        dataframe['vwap'] = vwap_values
        
        # Tính độ lệch từ VWAP
        dataframe['vwap_deviation'] = 100 * (dataframe['close'] - dataframe['vwap']) / dataframe['vwap']
        
        # Xác định khi giá vượt xa VWAP
        dataframe['above_vwap_threshold'] = dataframe['vwap_deviation'] > deviation_pct
        dataframe['below_vwap_threshold'] = dataframe['vwap_deviation'] < -deviation_pct
        
        # Xác định khi giá quay trở lại VWAP (mean reversion)
        dataframe['vwap_cross_up'] = (
            (dataframe['close'].shift(1) < dataframe['vwap'].shift(1)) &
            (dataframe['close'] > dataframe['vwap'])
        )
        
        dataframe['vwap_cross_down'] = (
            (dataframe['close'].shift(1) > dataframe['vwap'].shift(1)) &
            (dataframe['close'] < dataframe['vwap'])
        )
        
        return dataframe
    
    def fibonacci_levels(self, dataframe: DataFrame) -> DataFrame:
        """Tính toán các mức Fibonacci Retracement"""
        # Xác định swing high/low trong 100 nến gần nhất
        rolling_max = dataframe['high'].rolling(100).max()
        rolling_min = dataframe['low'].rolling(100).min()
        
        # Các mức Fibonacci cơ bản
        fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
        
        # Uptrend: Retracement từ low lên high
        for level in fib_levels:
            dataframe[f'fib_uptrend_{int(level*1000)}'] = rolling_min + (rolling_max - rolling_min) * level
        
        # Downtrend: Retracement từ high xuống low
        for level in fib_levels:
            dataframe[f'fib_downtrend_{int(level*1000)}'] = rolling_max - (rolling_max - rolling_min) * level
        
        return dataframe
    
    def detect_high_volume_bars(self, dataframe: DataFrame, threshold: int = 200) -> DataFrame:
        """Phát hiện các nến có volume cao"""
        dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean'] * 100
        dataframe['high_volume'] = dataframe['volume_ratio'] > threshold
        
        return dataframe
    
    def detect_trend(self, dataframe: DataFrame, period: int = 10) -> DataFrame:
        """Xác định xu hướng dựa trên EMA"""
        dataframe['ema_short'] = ta.EMA(dataframe, timeperiod=period)
        dataframe['ema_long'] = ta.EMA(dataframe, timeperiod=period*2)
        
        dataframe['uptrend'] = (
            (dataframe['ema_short'] > dataframe['ema_long']) &
            (dataframe['ema_short'].shift(1) > dataframe['ema_long'].shift(1)) &
            (dataframe['ema_short'] > dataframe['ema_short'].shift(1))
        )
        
        dataframe['downtrend'] = (
            (dataframe['ema_short'] < dataframe['ema_long']) &
            (dataframe['ema_short'].shift(1) < dataframe['ema_long'].shift(1)) &
            (dataframe['ema_short'] < dataframe['ema_short'].shift(1))
        )
        
        return dataframe
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        if not self.dp:
            # Chủ động tránh lỗi khi backtest/hyperopt
            return dataframe
        
        # Lấy dữ liệu từ timeframe cao hơn để phân tích xu hướng tổng thể
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], 
            timeframe=self.timeframe_higher
        )
        
        if len(informative) == 0:
            # Xử lý khi không có dữ liệu informative
            return dataframe
        
        # Thêm các chỉ báo cho khung thời gian cao hơn
        informative = self.detect_trend(informative, self.breakout_trend_period.value)
        
        # Đổi tên cột để tránh trùng lặp
        informative_columns = {
            f"inf_{col}": informative[col] for col in informative.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']
        }
        
        # Merge với dataframe gốc
        dataframe = pd.merge(
            dataframe, informative_columns, left_on='date', right_on='date', how='left'
        )
        
        # Darvas Boxes
        dataframe = self.get_darvas_boxes(
            dataframe, 
            self.darvas_lookback.value, 
            self.darvas_threshold.value
        )
        
        # VWAP và độ lệch
        dataframe = self.detect_vwap_deviation(
            dataframe,
            self.vwap_period.value,
            self.vwap_deviation.value
        )
        
        # Fibonacci levels
        dataframe = self.fibonacci_levels(dataframe)
        
        # Volume cao
        dataframe = self.detect_high_volume_bars(dataframe, self.volume_threshold.value)
        
        # Thêm ADX để đo lường sức mạnh xu hướng
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        
        # Thêm chỉ báo tin cậy cho tín hiệu breakout
        dataframe['breakout_score'] = self.calculate_breakout_score(dataframe)
        
        return dataframe
    
    def calculate_breakout_score(self, dataframe: DataFrame) -> pd.Series:
        """
        Tính toán điểm tin cậy cho tín hiệu breakout (0-100)
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. Volume (30%)
        score += (dataframe['high_volume']) * 30
        
        # 2. Vùng tích lũy (25%)
        # Càng nhiều nến trong box càng đáng tin cậy
        box_score = dataframe['box_count'].clip(0, 15) / 15 * 25
        score += box_score
        
        # 3. Sức mạnh breakout (20%)
        # Khoảng cách vượt qua high/low của box
        breakout_strength_up = ((dataframe['close'] - dataframe['darvas_high'].shift(1)) / dataframe['darvas_high'].shift(1) * 100).clip(0, 5) * 4
        breakout_strength_down = ((dataframe['darvas_low'].shift(1) - dataframe['close']) / dataframe['darvas_low'].shift(1) * 100).clip(0, 5) * 4
        score += breakout_strength_up + breakout_strength_down
        
        # 4. ADX - Sức mạnh xu hướng (15%)
        score += (dataframe['adx'] / 50).clip(0, 1) * 15
        
        # 5. Trend alignment (10%)
        # Breakout lên nên cùng xu hướng với uptrend
        score += (dataframe['box_breakout_up'] & dataframe['inf_uptrend']) * 10
        # Breakout xuống nên cùng xu hướng với downtrend
        score += (dataframe['box_breakout_down'] & dataframe['inf_downtrend']) * 10
        
        return score.clip(0, 100)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Điều kiện vào lệnh Long
        conditions_long = []
        
        # 1. Breakout lên từ vùng tích lũy với volume cao và điểm tin cậy cao
        conditions_long.append(
            (dataframe['box_breakout_up']) &  # Breakout lên
            (dataframe['breakout_score'] > 70) &  # Điểm tin cậy cao
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5)  # Volume cao
        )
        
        # 2. Các kiểu vào lệnh VWAP Mean Reversion
        conditions_long.append(
            (dataframe['close'] < dataframe['vwap']) &  # Giá dưới VWAP
            (dataframe['below_vwap_threshold']) &  # Độ lệch đủ lớn
            (dataframe['vwap_cross_up']) &  # Giá bắt đầu cắt lên VWAP
            (dataframe['inf_uptrend'])  # Uptrend ở timeframe cao hơn
        )
        
        # Kết hợp điều kiện
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Điều kiện vào lệnh Short
        conditions_short = []
        
        # 1. Breakout xuống từ vùng tích lũy với volume cao và điểm tin cậy cao
        conditions_short.append(
            (dataframe['box_breakout_down']) &  # Breakout xuống
            (dataframe['breakout_score'] > 70) &  # Điểm tin cậy cao
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5)  # Volume cao
        )
        
        # 2. Các kiểu vào lệnh VWAP Mean Reversion
        conditions_short.append(
            (dataframe['close'] > dataframe['vwap']) &  # Giá trên VWAP
            (dataframe['above_vwap_threshold']) &  # Độ lệch đủ lớn
            (dataframe['vwap_cross_down']) &  # Giá bắt đầu cắt xuống VWAP
            (dataframe['inf_downtrend'])  # Downtrend ở timeframe cao hơn
        )
        
        # Kết hợp điều kiện
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Các điều kiện thoát Long
        conditions_exit_long = []
        
        # 1. Thoát khi giá chạm mức Fibonacci cao
        fib_level = int(self.fib_retracement.value * 1000)
        conditions_exit_long.append(
            (dataframe['high'] > dataframe[f'fib_uptrend_{fib_level}']) &
            (dataframe['high'].shift(1) <= dataframe[f'fib_uptrend_{fib_level}'].shift(1))
        )
        
        # 2. Thoát khi giá vượt quá xa VWAP
        conditions_exit_long.append(
            (dataframe['above_vwap_threshold']) &
            (dataframe['vwap_deviation'] > 2 * self.vwap_deviation.value)
        )
        
        # 3. Thoát khi RSI quá cao
        conditions_exit_long.append(dataframe['rsi'] > 75)
        
        # 4. Thoát khi cắt xuống dưới VWAP sau khi đã lên cao
        conditions_exit_long.append(
            (dataframe['close'] < dataframe['vwap']) &
            (dataframe['close'].shift(1) > dataframe['vwap'].shift(1)) &
            (dataframe['close'].shift(2) > dataframe['vwap'].shift(2))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Các điều kiện thoát Short
        conditions_exit_short = []
        
        # 1. Thoát khi giá chạm mức Fibonacci thấp
        conditions_exit_short.append(
            (dataframe['low'] < dataframe[f'fib_downtrend_{fib_level}']) &
            (dataframe['low'].shift(1) >= dataframe[f'fib_downtrend_{fib_level}'].shift(1))
        )
        
        # 2. Thoát khi giá vượt quá xa VWAP
        conditions_exit_short.append(
            (dataframe['below_vwap_threshold']) &
            (dataframe['vwap_deviation'] < -2 * self.vwap_deviation.value)
        )
        
        # 3. Thoát khi RSI quá thấp
        conditions_exit_short.append(dataframe['rsi'] < 25)
        
        # 4. Thoát khi cắt lên trên VWAP sau khi đã xuống thấp
        conditions_exit_short.append(
            (dataframe['close'] > dataframe['vwap']) &
            (dataframe['close'].shift(1) < dataframe['vwap'].shift(1)) &
            (dataframe['close'].shift(2) < dataframe['vwap'].shift(2))
        )
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên điểm tin cậy của breakout
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có đủ dữ liệu, sử dụng đòn bẩy mặc định
        if len(dataframe) < 1:
            return self.leverage_default
        
        # Lấy nến gần nhất
        last_candle = dataframe.iloc[-1]
        
        # Lấy điểm tin cậy của breakout
        breakout_score = last_candle['breakout_score']
        
        # Điều chỉnh đòn bẩy dựa trên điểm tin cậy
        if breakout_score > 85:  # Tín hiệu rất tin cậy
            leverage = min(self.leverage_max, max_leverage)
        elif breakout_score > 75:  # Tín hiệu khá tin cậy
            leverage = min(6, max_leverage)
        elif breakout_score > 65:  # Tín hiệu trung bình
            leverage = min(4, max_leverage)
        else:  # Tín hiệu yếu
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên VWAP và khoảng nghịch đảo tại mức giá darvas box
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu, sử dụng stoploss mặc định
        if len(dataframe) < 1:
            return self.stoploss
        
        # Lấy nến gần nhất
        last_candle = dataframe.iloc[-1]
        
        # Nếu đã có lợi nhuận, sử dụng VWAP làm stoploss
        if current_profit > 0.03:  # >3% lợi nhuận
            # Với vị thế Long, đặt stoploss ở VWAP
            if not trade.is_short:
                vwap_sl = (last_candle['vwap'] / current_rate) - 1
                return max(vwap_sl, -0.02)  # Không thấp hơn -2%
            # Với vị thế Short, đặt stoploss ở VWAP
            else:
                vwap_sl = 1 - (last_candle['vwap'] / current_rate)
                return max(vwap_sl, -0.02)  # Không thấp hơn -2%
        
        # Nếu là breakout, đặt stoploss ở mức darvas box trước đó
        if not trade.is_short and last_candle['box_breakout_up'] > 0:
            return (last_candle['darvas_high'] / current_rate) - 1
        elif trade.is_short and last_candle['box_breakout_down'] > 0:
            return 1 - (last_candle['darvas_low'] / current_rate)
        
        return self.stoploss  # Sử dụng stoploss mặc định
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận vào lệnh với lọc thêm
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu
        if len(dataframe) < 1:
            return False
        
        # Lấy nến gần nhất
        last_candle = dataframe.iloc[-1]
        
        # Chỉ chấp nhận tín hiệu có điểm tin cậy cao
        if last_candle['breakout_score'] < 65:
            return False
        
        # Không vào lệnh nếu sắp có tin lớn (cần thêm dữ liệu calendar)
        
        # Không vào lệnh nếu khối lượng giao dịch quá thấp
        if last_candle['volume'] < 0.5 * last_candle['volume_mean']:
            return False
        
        return True
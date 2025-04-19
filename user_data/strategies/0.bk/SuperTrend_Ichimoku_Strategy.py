"""
SuperTrend + Ichimoku Cloud Hybrid Strategy
Kết hợp sức mạnh của SuperTrend và Ichimoku Cloud với thêm xác nhận từ RSI và khối lượng
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

def calculate_supertrend(dataframe, multiplier, period):
    """
    Tính toán chỉ báo SuperTrend
    
    SuperTrend = (High + Low) / 2 +/- Multiplier * ATR
    """
    # Tính ATR
    df = dataframe.copy()
    df['atr'] = ta.ATR(df, timeperiod=period)
    
    # Tính upper và lower band
    df['upperband'] = ((df['high'] + df['low']) / 2) + (multiplier * df['atr'])
    df['lowerband'] = ((df['high'] + df['low']) / 2) - (multiplier * df['atr'])
    
    # Khởi tạo giá trị
    df['supertrend'] = 0.0
    df['supertrend_direction'] = 1  # 1 = uptrend, -1 = downtrend
    
    # Tính toán SuperTrend
    for i in range(1, len(df)):
        curr = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Tính upperband
        if (curr['upperband'] < prev['upperband']) or (prev['close'] > prev['upperband']):
            curr_upperband = curr['upperband']
        else:
            curr_upperband = prev['upperband']
            
        # Tính lowerband
        if (curr['lowerband'] > prev['lowerband']) or (prev['close'] < prev['lowerband']):
            curr_lowerband = curr['lowerband']
        else:
            curr_lowerband = prev['lowerband']
            
        # Xác định xu hướng
        if prev['supertrend'] == prev['upperband']:
            if curr['close'] < curr_upperband:
                curr_supertrend = curr_upperband
                curr_direction = -1
            else:
                curr_supertrend = curr_lowerband
                curr_direction = 1
        else:
            if curr['close'] > curr_lowerband:
                curr_supertrend = curr_lowerband
                curr_direction = 1
            else:
                curr_supertrend = curr_upperband
                curr_direction = -1
        
        # Cập nhật giá trị
        df.at[i, 'supertrend'] = curr_supertrend
        df.at[i, 'supertrend_direction'] = curr_direction
    
    return df['supertrend'], df['supertrend_direction']

class SuperTrend_Ichimoku_Strategy(IStrategy):
    """
    SuperTrend + Ichimoku Cloud Hybrid Strategy
    
    Chiến lược này kết hợp:
    1. SuperTrend để xác định xu hướng chính và entry/exit
    2. Ichimoku Cloud để xác nhận trạng thái thị trường
    3. RSI để lọc vào lệnh và tránh vùng quá mua/quá bán
    4. Kết hợp khối lượng để xác thực tín hiệu
    
    Phù hợp với khung thời gian 15m và 1h
    """
    
    # Tham số cho SuperTrend
    supertrend_multiplier = DecimalParameter(1.0, 5.0, default=3.0, space="buy", optimize=True)
    supertrend_period = IntParameter(7, 30, default=10, space="buy", optimize=True)
    
    # Tham số cho Ichimoku
    tenkan_period = IntParameter(5, 15, default=9, space="buy", optimize=True)
    kijun_period = IntParameter(20, 40, default=26, space="buy", optimize=True)
    senkou_span_b_period = IntParameter(40, 80, default=52, space="buy", optimize=True)
    
    # Tham số cho RSI
    rsi_period = IntParameter(10, 30, default=14, space="buy", optimize=True)
    rsi_oversold = IntParameter(20, 40, default=30, space="buy", optimize=True)
    rsi_overbought = IntParameter(60, 80, default=70, space="sell", optimize=True)
    
    # Tham số cho khối lượng
    volume_threshold = IntParameter(150, 300, default=200, space="buy", optimize=True)
    
    # Đòn bẩy và stoploss
    leverage_default = 3
    leverage_max = 7
    stoploss = -0.05
    
    # Time-in-force và ROI
    timeframe = '15m'
    timeframe_higher = '1h'
    
    minimal_roi = {
        "0": 0.08,    # 8% ban đầu
        "60": 0.04,   # 4% sau 60 phút
        "120": 0.02,  # 2% sau 120 phút
        "300": 0.01   # 1% sau 300 phút
    }
    
    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01  # 1%
    trailing_stop_positive_offset = 0.025  # 2.5%
    trailing_only_offset_is_reached = True
    
    # Khởi tạo hệ thống
    startup_candle_count = 100
    process_only_new_candles = True
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """Trả về các cặp thông tin bổ sung"""
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.timeframe_higher) for pair in pairs]
        return informative_pairs
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Thêm các chỉ báo vào dataframe"""
        if not self.dp:
            # Xử lý khi DP không có sẵn (backtest)
            return dataframe
            
        # Lấy dữ liệu từ timeframe cao hơn
        informative = self.dp.get_pair_dataframe(
            pair=metadata['pair'], 
            timeframe=self.timeframe_higher
        )
        
        if len(informative) > 0:
            # Tính toán SuperTrend cho timeframe cao hơn
            informative['inf_supertrend'], informative['inf_supertrend_direction'] = calculate_supertrend(
                informative, 
                self.supertrend_multiplier.value,
                self.supertrend_period.value
            )
            
            # Tính toán Ichimoku cho timeframe cao hơn
            informative['inf_tenkan_sen'] = ta.SMA(informative, timeperiod=self.tenkan_period.value)
            informative['inf_kijun_sen'] = ta.SMA(informative, timeperiod=self.kijun_period.value)
            informative['inf_senkou_span_a'] = (informative['inf_tenkan_sen'] + informative['inf_kijun_sen']) / 2
            informative['inf_senkou_span_b'] = ta.SMA(informative, timeperiod=self.senkou_span_b_period.value)
            
            # Tính toán RSI
            informative['inf_rsi'] = ta.RSI(informative, timeperiod=self.rsi_period.value)
            
            # Xác định trạng thái thị trường
            informative['inf_market_bull'] = (
                (informative['close'] > informative['inf_senkou_span_a']) &
                (informative['close'] > informative['inf_senkou_span_b']) &
                (informative['inf_tenkan_sen'] > informative['inf_kijun_sen'])
            )
            
            informative['inf_market_bear'] = (
                (informative['close'] < informative['inf_senkou_span_a']) &
                (informative['close'] < informative['inf_senkou_span_b']) &
                (informative['inf_tenkan_sen'] < informative['inf_kijun_sen'])
            )
            
            # Đổi tên để tránh trùng lặp
            informative_columns = {
                f"inf_{col}": informative[col] for col in informative.columns if col not in ['date', 'open', 'high', 'low', 'close', 'volume']
            }
            
            # Merge với dataframe gốc
            dataframe = pd.merge(
                dataframe, informative_columns, left_on='date', right_on='date', how='left'
            )
        
        # Tính SuperTrend cho timeframe hiện tại
        dataframe['supertrend'], dataframe['supertrend_direction'] = calculate_supertrend(
            dataframe, 
            self.supertrend_multiplier.value,
            self.supertrend_period.value
        )
        
        # Tính Ichimoku cho timeframe hiện tại
        dataframe['tenkan_sen'] = ta.SMA(dataframe, timeperiod=self.tenkan_period.value)
        dataframe['kijun_sen'] = ta.SMA(dataframe, timeperiod=self.kijun_period.value)
        dataframe['senkou_span_a'] = (dataframe['tenkan_sen'] + dataframe['kijun_sen']) / 2
        dataframe['senkou_span_b'] = ta.SMA(dataframe, timeperiod=self.senkou_span_b_period.value)
        dataframe['chikou_span'] = dataframe['close'].shift(-26)
        
        # Phân tích xu hướng Ichimoku
        dataframe['cloud_green'] = dataframe['senkou_span_a'] > dataframe['senkou_span_b']
        dataframe['above_cloud'] = (
            (dataframe['close'] > dataframe['senkou_span_a']) &
            (dataframe['close'] > dataframe['senkou_span_b'])
        )
        dataframe['below_cloud'] = (
            (dataframe['close'] < dataframe['senkou_span_a']) &
            (dataframe['close'] < dataframe['senkou_span_b'])
        )
        dataframe['in_cloud'] = ~(dataframe['above_cloud'] | dataframe['below_cloud'])
        
        # TK Cross
        dataframe['tk_cross_up'] = qtpylib.crossed_above(
            dataframe['tenkan_sen'], dataframe['kijun_sen']
        )
        dataframe['tk_cross_down'] = qtpylib.crossed_below(
            dataframe['tenkan_sen'], dataframe['kijun_sen']
        )
        
        # Price Cross Kijun
        dataframe['price_cross_kijun_up'] = qtpylib.crossed_above(
            dataframe['close'], dataframe['kijun_sen']
        )
        dataframe['price_cross_kijun_down'] = qtpylib.crossed_below(
            dataframe['close'], dataframe['kijun_sen']
        )
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.rsi_period.value)
        
        # Khối lượng
        dataframe['volume_mean'] = dataframe['volume'].rolling(20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_mean'] * 100
        dataframe['high_volume'] = dataframe['volume_ratio'] > self.volume_threshold.value
        
        # Phân tích xu hướng tổng hợp
        dataframe['uptrend'] = (
            (dataframe['supertrend_direction'] == 1) &
            (dataframe['close'] > dataframe['senkou_span_a']) &
            (dataframe['tenkan_sen'] > dataframe['kijun_sen'])
        )
        
        dataframe['downtrend'] = (
            (dataframe['supertrend_direction'] == -1) &
            (dataframe['close'] < dataframe['senkou_span_a']) &
            (dataframe['tenkan_sen'] < dataframe['kijun_sen'])
        )
        
        # Score calculation (0-100)
        dataframe['buy_score'] = self.calculate_buy_score(dataframe)
        dataframe['sell_score'] = self.calculate_sell_score(dataframe)
        
        return dataframe
    
    def calculate_buy_score(self, dataframe):
        """
        Tính điểm tín hiệu mua (0-100)
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. SuperTrend direction (30%)
        score += (dataframe['supertrend_direction'] == 1) * 30
        
        # 2. Ichimoku signals (40%)
        score += dataframe['above_cloud'] * 15
        score += dataframe['cloud_green'] * 10
        score += dataframe['tk_cross_up'] * 10
        score += dataframe['price_cross_kijun_up'] * 5
        
        # 3. RSI (15%)
        score += ((dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)) * 7
        score += (dataframe['rsi'] > dataframe['rsi'].shift(1)) * 8
        
        # 4. Volume (15%)
        score += dataframe['high_volume'] * 15
        
        # 5. Higher timeframe alignment (20% bonus if already > 0)
        # Only add if score is already positive
        score = score.apply(lambda x: x * 1.2 if (x > 0 and dataframe['inf_market_bull'].iloc[score.index.get_loc(x)]) else x)
        
        return score.clip(0, 100)
    
    def calculate_sell_score(self, dataframe):
        """
        Tính điểm tín hiệu bán (0-100)
        """
        score = pd.Series(0, index=dataframe.index)
        
        # 1. SuperTrend direction (30%)
        score += (dataframe['supertrend_direction'] == -1) * 30
        
        # 2. Ichimoku signals (40%)
        score += dataframe['below_cloud'] * 15
        score += (~dataframe['cloud_green']) * 10
        score += dataframe['tk_cross_down'] * 10
        score += dataframe['price_cross_kijun_down'] * 5
        
        # 3. RSI (15%)
        score += (dataframe['rsi'] > 70) * 7
        score += (dataframe['rsi'] < dataframe['rsi'].shift(1)) * 8
        
        # 4. Volume (15%)
        score += dataframe['high_volume'] * 15
        
        # 5. Higher timeframe alignment (20% bonus if already > 0)
        # Only add if score is already positive
        score = score.apply(lambda x: x * 1.2 if (x > 0 and dataframe['inf_market_bear'].iloc[score.index.get_loc(x)]) else x)
        
        return score.clip(0, 100)
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu vào lệnh"""
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Các điều kiện vào lệnh Long
        conditions_long = []
        
        # 1. Score cao
        conditions_long.append(dataframe['buy_score'] > 70)
        
        # 2. SuperTrend + Ichimoku
        conditions_long.append(
            (dataframe['supertrend_direction'] == 1) &  # SuperTrend lên
            (dataframe['close'] > dataframe['supertrend']) &  # Giá trên SuperTrend
            (dataframe['above_cloud'])  # Giá trên mây
        )
        
        # 3. RSI không quá mua
        conditions_long.append(dataframe['rsi'] < self.rsi_overbought.value)
        
        # 4. Khối lượng đủ
        conditions_long.append(dataframe['volume_ratio'] > 80)
        
        # 5. Kijun cross
        conditions_long.append(
            (dataframe['price_cross_kijun_up']) &
            (dataframe['close'] > dataframe['kijun_sen'])
        )
        
        # 6. TK cross
        conditions_long.append(
            (dataframe['tk_cross_up']) &
            (dataframe['close'] > dataframe['kijun_sen'])
        )
        
        # Kết hợp điều kiện - chỉ cần 1 trong các điều kiện thỏa mãn
        if conditions_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_long), 'enter_long'] = 1
        
        # Các điều kiện vào lệnh Short
        conditions_short = []
        
        # 1. Score cao
        conditions_short.append(dataframe['sell_score'] > 70)
        
        # 2. SuperTrend + Ichimoku
        conditions_short.append(
            (dataframe['supertrend_direction'] == -1) &  # SuperTrend xuống
            (dataframe['close'] < dataframe['supertrend']) &  # Giá dưới SuperTrend
            (dataframe['below_cloud'])  # Giá dưới mây
        )
        
        # 3. RSI không quá bán
        conditions_short.append(dataframe['rsi'] > self.rsi_oversold.value)
        
        # 4. Khối lượng đủ
        conditions_short.append(dataframe['volume_ratio'] > 80)
        
        # 5. Kijun cross
        conditions_short.append(
            (dataframe['price_cross_kijun_down']) &
            (dataframe['close'] < dataframe['kijun_sen'])
        )
        
        # 6. TK cross
        conditions_short.append(
            (dataframe['tk_cross_down']) &
            (dataframe['close'] < dataframe['kijun_sen'])
        )
        
        # Kết hợp điều kiện - chỉ cần 1 trong các điều kiện thỏa mãn
        if conditions_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_short), 'enter_short'] = 1
        
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Xác định tín hiệu thoát lệnh"""
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Các điều kiện thoát Long
        conditions_exit_long = []
        
        # 1. SuperTrend đảo chiều
        conditions_exit_long.append(
            (dataframe['supertrend_direction'] == -1) &
            (dataframe['supertrend_direction'].shift(1) == 1)
        )
        
        # 2. Giá cắt xuống dưới SuperTrend
        conditions_exit_long.append(
            (dataframe['close'] < dataframe['supertrend']) &
            (dataframe['close'].shift(1) >= dataframe['supertrend'].shift(1))
        )
        
        # 3. Giá cắt xuống dưới Kijun-sen
        conditions_exit_long.append(dataframe['price_cross_kijun_down'])
        
        # 4. TK Cross down
        conditions_exit_long.append(dataframe['tk_cross_down'])
        
        # 5. RSI quá mua
        conditions_exit_long.append(dataframe['rsi'] > 80)
        
        # Kết hợp điều kiện
        if conditions_exit_long:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_long), 'exit_long'] = 1
        
        # Các điều kiện thoát Short
        conditions_exit_short = []
        
        # 1. SuperTrend đảo chiều
        conditions_exit_short.append(
            (dataframe['supertrend_direction'] == 1) &
            (dataframe['supertrend_direction'].shift(1) == -1)
        )
        
        # 2. Giá cắt lên trên SuperTrend
        conditions_exit_short.append(
            (dataframe['close'] > dataframe['supertrend']) &
            (dataframe['close'].shift(1) <= dataframe['supertrend'].shift(1))
        )
        
        # 3. Giá cắt lên trên Kijun-sen
        conditions_exit_short.append(dataframe['price_cross_kijun_up'])
        
        # 4. TK Cross up
        conditions_exit_short.append(dataframe['tk_cross_up'])
        
        # 5. RSI quá bán
        conditions_exit_short.append(dataframe['rsi'] < 20)
        
        # Kết hợp điều kiện
        if conditions_exit_short:
            dataframe.loc[reduce(lambda x, y: x | y, conditions_exit_short), 'exit_short'] = 1
        
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên điểm tín hiệu
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có dữ liệu
        if len(dataframe) < 1:
            return self.leverage_default
        
        last_candle = dataframe.iloc[-1]
        
        # Lấy điểm tín hiệu
        if side == 'long':
            signal_score = last_candle['buy_score']
        else:
            signal_score = last_candle['sell_score']
        
        # Điều chỉnh đòn bẩy dựa trên điểm tín hiệu
        if signal_score > 90:  # Tín hiệu rất mạnh
            leverage = min(self.leverage_max, max_leverage)
        elif signal_score > 80:  # Tín hiệu khá mạnh
            leverage = min(6, max_leverage)
        elif signal_score > 70:  # Tín hiệu trung bình
            leverage = min(5, max_leverage)
        elif signal_score > 60:  # Tín hiệu yếu
            leverage = min(4, max_leverage)
        else:  # Tín hiệu rất yếu
            leverage = self.leverage_default
        
        return leverage
    
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Stoploss động dựa trên SuperTrend và Kijun-sen
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu
        if len(dataframe) < 1:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1]
        
        # Với lợi nhuận tích cực, sử dụng Kijun-sen làm stoploss động
        if current_profit > 0.02:  # >2% lợi nhuận
            # Với vị thế Long, đặt stoploss ở Kijun-sen nếu nó cao hơn SuperTrend
            if not trade.is_short:
                kijun_sl = (last_candle['kijun_sen'] / current_rate) - 1
                supertrend_sl = (last_candle['supertrend'] / current_rate) - 1
                
                # Chọn stoploss cao hơn (ít âm hơn)
                dynamic_sl = max(kijun_sl, supertrend_sl)
                
                return max(dynamic_sl, -0.05)  # Không thấp hơn -5%
                
            # Với vị thế Short, đặt stoploss ở Kijun-sen nếu nó thấp hơn SuperTrend
            else:
                kijun_sl = 1 - (last_candle['kijun_sen'] / current_rate)
                supertrend_sl = 1 - (last_candle['supertrend'] / current_rate)
                
                # Chọn stoploss cao hơn (ít âm hơn)
                dynamic_sl = max(kijun_sl, supertrend_sl)
                
                return max(dynamic_sl, -0.05)  # Không thấp hơn -5%
        
        # Nếu lợi nhuận còn thấp, sử dụng stoploss mặc định
        return self.stoploss
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận vào lệnh với lọc thêm
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không có dữ liệu
        if len(dataframe) < 1:
            return False
        
        last_candle = dataframe.iloc[-1]
        
        # 1. Xác nhận dựa trên điểm tín hiệu
        if side == 'long' and last_candle['buy_score'] < 60:
            return False
        elif side == 'short' and last_candle['sell_score'] < 60:
            return False
        
        # 2. Kiểm tra alignment với timeframe cao hơn
        if side == 'long' and not last_candle['inf_market_bull']:
            # Tín hiệu mua nhưng timeframe cao đang bearish, cần score cao hơn
            if last_candle['buy_score'] < 80:
                return False
        
        elif side == 'short' and not last_candle['inf_market_bear']:
            # Tín hiệu bán nhưng timeframe cao đang bullish, cần score cao hơn
            if last_candle['sell_score'] < 80:
                return False
        
        # 3. Kiểm tra khối lượng
        if last_candle['volume'] < 0.8 * last_candle['volume_mean']:
            return False
        
        return True
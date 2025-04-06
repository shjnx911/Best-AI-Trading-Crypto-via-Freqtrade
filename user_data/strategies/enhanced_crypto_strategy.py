import logging
import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from pandas import DataFrame

# Các hàm technical analysis cơ bản để thay thế TA-Lib
def EMA(dataframe, timeperiod=20):
    return dataframe['close'].ewm(span=timeperiod, adjust=False).mean()

def SMA(series, timeperiod=5):
    return series.rolling(window=timeperiod).mean()

def RSI(dataframe, timeperiod=14):
    delta = dataframe['close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(timeperiod).mean()
    roll_down = down.abs().rolling(timeperiod).mean()
    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def ATR(dataframe, timeperiod=14):
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    tr1 = abs(high - low)
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=timeperiod).mean()
    return atr

def BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2):
    middle = dataframe['close'].rolling(window=timeperiod).mean()
    std = dataframe['close'].rolling(window=timeperiod).std()
    upper = middle + nbdevup * std
    lower = middle - nbdevdn * std
    
    return {
        'upperband': upper,
        'middleband': middle,
        'lowerband': lower
    }

def MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9):
    fast = dataframe['close'].ewm(span=fastperiod, adjust=False).mean()
    slow = dataframe['close'].ewm(span=slowperiod, adjust=False).mean()
    macd = fast - slow
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    
    return {
        'macd': macd,
        'macdsignal': signal,
        'macdhist': hist
    }

def STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3):
    high_max = dataframe['high'].rolling(window=fastk_period).max()
    low_min = dataframe['low'].rolling(window=fastk_period).min()
    
    fastk = 100 * ((dataframe['close'] - low_min) / (high_max - low_min))
    slowk = fastk.rolling(window=slowk_period).mean()
    slowd = slowk.rolling(window=slowd_period).mean()
    
    return {
        'slowk': slowk,
        'slowd': slowd
    }

def MFI(dataframe, timeperiod=14):
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    money_flow = typical_price * dataframe['volume']
    
    delta = typical_price.diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    
    up_money_flow = money_flow.copy()
    up_money_flow[delta <= 0] = 0
    
    down_money_flow = money_flow.copy()
    down_money_flow[delta >= 0] = 0
    
    up_sum = up_money_flow.rolling(window=timeperiod).sum()
    down_sum = down_money_flow.abs().rolling(window=timeperiod).sum()
    
    money_ratio = up_sum / down_sum
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi

# Chiến lược FreqTrade cơ bản
class EnhancedCryptoStrategy:
    """
    Chiến lược AI nâng cao cho giao dịch cryptocurrency.
    Kết hợp các chỉ báo kỹ thuật với dự đoán từ mô hình AI để tạo tín hiệu giao dịch.
    """
    # Cài đặt cơ bản
    minimal_roi = {
        "0": 0.03,  # Mục tiêu lợi nhuận 3%
        "30": 0.02,  # Sau 30 phút, chốt lời ở 2%
        "60": 0.01,  # Sau 60 phút, chốt lời ở 1%
        "180": 0.005,  # Sau 180 phút, chốt lời ở 0.5%
    }
    stoploss = -0.02  # Cắt lỗ ở 2%
    timeframe = '5m'  # Khung thời gian 5 phút
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 100
    
    # Parameters for dynamic ROI using a custom stoploss
    # Get custom stoploss based on profit
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        # Dynamic stoploss that adjusts based on profit level
        if current_profit > 0.03:
            return -0.01  # Very tight stoploss when in good profit
        elif current_profit > 0.02:
            return -0.015  # Slightly looser stoploss at medium profit
        else:
            return -0.02  # Original stoploss with low profit
    
    def __init__(self, config: dict) -> None:
        """
        Khởi tạo chiến lược với cấu hình được cung cấp.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Đường dẫn đến thư mục mô hình
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'models')
        
        # Đường dẫn đến các tệp mô hình
        trend_model_path = os.path.join(model_dir, 'trend_model.pkl')
        volatility_model_path = os.path.join(model_dir, 'volatility_model.pkl')
        reversal_model_path = os.path.join(model_dir, 'reversal_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        # Kiểm tra xem mô hình đã tồn tại hay chưa
        self.models_exist = (
            os.path.exists(trend_model_path) and
            os.path.exists(volatility_model_path) and
            os.path.exists(reversal_model_path) and
            os.path.exists(scaler_path)
        )
        
        # Tải mô hình nếu chúng tồn tại
        if self.models_exist:
            try:
                self.trend_model = pickle.load(open(trend_model_path, 'rb'))
                self.volatility_model = pickle.load(open(volatility_model_path, 'rb'))
                self.reversal_model = pickle.load(open(reversal_model_path, 'rb'))
                self.scaler = pickle.load(open(scaler_path, 'rb'))
                self.logger.info("Đã tải thành công các mô hình AI từ %s", model_dir)
            except Exception as e:
                self.logger.error("Không thể tải mô hình AI: %s", str(e))
                self.models_exist = False
        else:
            self.logger.warning("Không tìm thấy mô hình AI. Chỉ sử dụng tín hiệu kỹ thuật.")
    
    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        Return các cặp thông tin bổ sung cho các cặp giao dịch hiện tại
        """
        return []
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Thêm các chỉ báo vào dataframe
        """
        # Chỉ báo xác định xu hướng
        dataframe['ema_short'] = EMA(dataframe, timeperiod=8)
        dataframe['ema_medium'] = EMA(dataframe, timeperiod=21)
        dataframe['ema_long'] = EMA(dataframe, timeperiod=50)
        dataframe['trend'] = ((dataframe['ema_short'] > dataframe['ema_medium']) & 
                            (dataframe['ema_medium'] > dataframe['ema_long'])).astype(int)
        
        # Chỉ báo biến động
        dataframe['atr'] = ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'] * 100
        
        # Chỉ báo xác định điểm đảo chiều
        dataframe['rsi'] = RSI(dataframe, timeperiod=14)
        dataframe['rsi_ma'] = SMA(dataframe['rsi'], timeperiod=5)
        dataframe['mfi'] = MFI(dataframe, timeperiod=14)
        
        # MACD
        macd = MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # Volume profile
        dataframe['volume_pct_change'] = dataframe['volume'].pct_change() * 100
        
        # Liquidity indicators
        dataframe['spread'] = (dataframe['high'] - dataframe['low']) / dataframe['low'] * 100
        
        # Thêm các chỉ báo đặc biệt - Tính năng dựa trên price action
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open']) / dataframe['open'] * 100
        dataframe['upper_wick'] = (dataframe['high'] - dataframe[['open', 'close']].max(axis=1)) / dataframe['open'] * 100
        dataframe['lower_wick'] = (dataframe[['open', 'close']].min(axis=1) - dataframe['low']) / dataframe['open'] * 100
        
        # Bollinger Bands
        bollinger = BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_width'] = ((dataframe['bb_upperband'] - dataframe['bb_lowerband']) / 
                                dataframe['bb_middleband'])
        
        # Stochastic
        stoch = STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']
        
        # Thêm các feature dẫn xuất
        dataframe['ema_diff'] = ((dataframe['ema_short'] - dataframe['ema_medium']) / 
                                dataframe['ema_medium'] * 100)
        dataframe['rsi_diff'] = dataframe['rsi'] - dataframe['rsi'].shift(1)
        
        # Phát hiện phân kỳ (Divergence)
        dataframe['price_higher_high'] = (dataframe['close'] > dataframe['close'].shift(1)) & (dataframe['close'].shift(1) > dataframe['close'].shift(2))
        dataframe['rsi_lower_high'] = (dataframe['rsi'] < dataframe['rsi'].shift(1)) & (dataframe['rsi'].shift(1) > dataframe['rsi'].shift(2))
        dataframe['divergence_bearish'] = dataframe['price_higher_high'] & dataframe['rsi_lower_high']
        
        dataframe['price_lower_low'] = (dataframe['close'] < dataframe['close'].shift(1)) & (dataframe['close'].shift(1) < dataframe['close'].shift(2))
        dataframe['rsi_higher_low'] = (dataframe['rsi'] > dataframe['rsi'].shift(1)) & (dataframe['rsi'].shift(1) < dataframe['rsi'].shift(2))
        dataframe['divergence_bullish'] = dataframe['price_lower_low'] & dataframe['rsi_higher_low']
        
        return dataframe
    
    def ai_prediction(self, dataframe: DataFrame) -> DataFrame:
        """
        Thêm dự đoán AI vào dataframe nếu mô hình tồn tại
        """
        import sys
        import os
        
        # Đảm bảo có thể import module market_sentiment
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        if root_dir not in sys.path:
            sys.path.append(root_dir)
        
        # Thêm phân tích tâm lý thị trường nếu có thể
        try:
            import market_sentiment
            pair = dataframe.get('pair', 'BTC/USDT')
            if len(dataframe) > 0:
                # Chỉ phân tích nến cuối cùng để tiết kiệm API call
                dataframe = market_sentiment.integrate_sentiment_with_technical(dataframe, pair)
        except Exception as e:
            print(f"Không thể sử dụng market_sentiment: {e}")
            # Nếu không có cột sentiment thì tạo
            if 'sentiment_score' not in dataframe.columns:
                dataframe['sentiment_score'] = 0.0
            if 'sentiment_signal' not in dataframe.columns:
                dataframe['sentiment_signal'] = 0
        
        # Chỉ áp dụng dự đoán AI nếu mô hình đã được tải
        if not self.models_exist:
            # Tạo cột mặc định cho dự đoán
            dataframe['trend_strength'] = 0.5
            dataframe['volatility_score'] = 0.5
            dataframe['reversal_chance'] = 0.5
            dataframe['long_confidence'] = 0.5
            dataframe['short_confidence'] = 0.5
            return dataframe
        
        # Chuẩn bị features cho mô hình
        features = dataframe[[
            'ema_short', 'ema_medium', 'ema_long', 'atr', 'atr_pct',
            'rsi', 'rsi_ma', 'mfi', 'macd', 'macdsignal', 'macdhist',
            'volume_pct_change', 'spread', 'body_size', 'upper_wick', 'lower_wick',
            'bb_width', 'slowk', 'slowd', 'ema_diff', 'rsi_diff'
        ]].copy().fillna(0)
        
        if len(features) >= 30:
            try:
                # Chuẩn hóa dữ liệu
                features_scaled = self.scaler.transform(features.tail(30))
                
                # Dự đoán xu hướng (1: tăng, 0: giảm)
                trend_pred = self.trend_model.predict_proba(features_scaled)
                
                # Dự đoán biến động (Đánh giá sức mạnh của thị trường)
                volatility_pred = self.volatility_model.predict_proba(features_scaled)
                
                # Dự đoán điểm đảo chiều
                reversal_pred = self.reversal_model.predict_proba(features_scaled)
                
                # Áp dụng dự đoán vào dataframe
                dataframe.loc[:, 'trend_strength'] = 0.5
                dataframe.loc[:, 'volatility_score'] = 0.5
                dataframe.loc[:, 'reversal_chance'] = 0.5
                
                # Chỉ áp dụng dự đoán cho 30 nến gần nhất
                dataframe.iloc[-30:, dataframe.columns.get_loc('trend_strength')] = trend_pred[:, 1]
                dataframe.iloc[-30:, dataframe.columns.get_loc('volatility_score')] = volatility_pred[:, 1]  
                dataframe.iloc[-30:, dataframe.columns.get_loc('reversal_chance')] = reversal_pred[:, 1]
                
                # Tính toán điểm confidence tổng thể
                # Kết hợp cả điểm tâm lý thị trường nếu có
                sentiment_weight = 0.0
                sentiment_score = 0.0
                if 'sentiment_score' in dataframe.columns:
                    sentiment_weight = 0.2  # Trọng số 20% cho phân tích tâm lý
                    sentiment_score = dataframe['sentiment_score'].fillna(0)
                    
                # Điều chỉnh trọng số của các thành phần khác
                trend_weight = 0.5 * (1 - sentiment_weight)
                volatility_weight = 0.3 * (1 - sentiment_weight)
                reversal_weight = 0.2 * (1 - sentiment_weight)
                
                # Tính confidence mới có tích hợp tâm lý thị trường
                dataframe['long_confidence'] = (
                    dataframe['trend_strength'] * trend_weight + 
                    dataframe['volatility_score'] * volatility_weight + 
                    (1 - dataframe['reversal_chance']) * reversal_weight +
                    (sentiment_score > 0) * sentiment_score * sentiment_weight
                )
                
                dataframe['short_confidence'] = (
                    (1 - dataframe['trend_strength']) * trend_weight + 
                    dataframe['volatility_score'] * volatility_weight + 
                    dataframe['reversal_chance'] * reversal_weight +
                    (sentiment_score < 0) * abs(sentiment_score) * sentiment_weight
                )
                
            except Exception as e:
                print(f"Lỗi khi thực hiện dự đoán AI: {e}")
                # Tạo cột mặc định cho dự đoán
                dataframe['trend_strength'] = 0.5
                dataframe['volatility_score'] = 0.5
                dataframe['reversal_chance'] = 0.5
                dataframe['long_confidence'] = 0.5
                dataframe['short_confidence'] = 0.5
        else:
            # Tạo cột mặc định cho dự đoán
            dataframe['trend_strength'] = 0.5
            dataframe['volatility_score'] = 0.5
            dataframe['reversal_chance'] = 0.5
            dataframe['long_confidence'] = 0.5
            dataframe['short_confidence'] = 0.5
        
        return dataframe
    
    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Đánh dấu các điểm vào lệnh mua (long)
        """
        # Áp dụng dự đoán AI nếu có
        dataframe = self.ai_prediction(dataframe)
        
        # Tín hiệu LONG dựa trên AI và xác nhận bởi chỉ báo kỹ thuật
        dataframe.loc[
            (
                # Dựa trên AI nếu có
                (
                    (self.models_exist & (dataframe['long_confidence'] > 0.75)) |
                    (~self.models_exist)
                ) &
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] > 40) & (dataframe['rsi'] < 65) &
                (dataframe['ema_short'] > dataframe['ema_medium']) &
                (dataframe['volume_pct_change'] > 5) &
                (dataframe['slowk'] > dataframe['slowd']) &
                # Phân kỳ tăng giá
                (dataframe['divergence_bullish'] | True)
            ),
            'buy'] = 1
        
        return dataframe
    
    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Đánh dấu các điểm thoát lệnh long
        """
        # Phân tích dữ liệu
        dataframe = self.ai_prediction(dataframe)
        
        # Đóng vị thế long
        dataframe.loc[
            (
                # Dựa trên AI
                (
                    (self.models_exist & (dataframe['reversal_chance'] > 0.7)) |
                    (self.models_exist & (dataframe['trend_strength'] < 0.3))
                ) |
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] > 78) |
                (dataframe['ema_short'] < dataframe['ema_medium']) |
                (dataframe['slowk'] < dataframe['slowd']) |
                # Phân kỳ giảm giá
                (dataframe['divergence_bearish'])
            ),
            'sell'] = 1
            
        return dataframe
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Phương thức mới dùng cho cả long và short (thay thế cho populate_buy_trend)
        """
        dataframe = self.ai_prediction(dataframe)
        
        # Tín hiệu LONG
        dataframe.loc[
            (
                # Dựa trên AI nếu có
                (
                    (self.models_exist & (dataframe['long_confidence'] > 0.75)) |
                    (~self.models_exist)
                ) &
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] > 40) & (dataframe['rsi'] < 65) &
                (dataframe['ema_short'] > dataframe['ema_medium']) &
                (dataframe['volume_pct_change'] > 5) &
                (dataframe['slowk'] > dataframe['slowd']) &
                # Phân kỳ tăng giá
                (dataframe['divergence_bullish'] | True)
            ),
            'enter_long'] = 1
        
        # Tín hiệu SHORT
        dataframe.loc[
            (
                # Dựa trên AI nếu có
                (
                    (self.models_exist & (dataframe['short_confidence'] > 0.75)) |
                    (~self.models_exist)
                ) &
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] < 60) & (dataframe['rsi'] > 35) &
                (dataframe['ema_short'] < dataframe['ema_medium']) &
                (dataframe['volume_pct_change'] > 5) &
                (dataframe['slowk'] < dataframe['slowd']) &
                # Phân kỳ giảm giá
                (dataframe['divergence_bearish'] | True)
            ),
            'enter_short'] = 1
            
        return dataframe
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Phương thức mới dùng cho cả đóng long và short (thay thế cho populate_sell_trend)
        """
        dataframe = self.ai_prediction(dataframe)
        
        # Đóng vị thế LONG
        dataframe.loc[
            (
                # Dựa trên AI
                (
                    (self.models_exist & (dataframe['reversal_chance'] > 0.7)) |
                    (self.models_exist & (dataframe['trend_strength'] < 0.3))
                ) |
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] > 78) |
                (dataframe['ema_short'] < dataframe['ema_medium']) |
                (dataframe['slowk'] < dataframe['slowd']) |
                # Phân kỳ giảm giá
                (dataframe['divergence_bearish'])
            ),
            'exit_long'] = 1
        
        # Đóng vị thế SHORT
        dataframe.loc[
            (
                # Dựa trên AI
                (
                    (self.models_exist & (dataframe['reversal_chance'] > 0.7)) |
                    (self.models_exist & (dataframe['trend_strength'] > 0.7))
                ) |
                # Dựa trên chỉ báo truyền thống
                (dataframe['rsi'] < 22) |
                (dataframe['ema_short'] > dataframe['ema_medium']) |
                (dataframe['slowk'] > dataframe['slowd']) |
                # Phân kỳ tăng giá
                (dataframe['divergence_bullish'])
            ),
            'exit_short'] = 1
            
        return dataframe
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str], side: str,
                dataframe=None, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên biến động của thị trường.
        """
        # Trong môi trường giả lập, sử dụng dataframe cung cấp
        if dataframe is not None and len(dataframe) > 0:
            current_volatility = dataframe['atr_pct'].iloc[-1] if 'atr_pct' in dataframe.columns else 2.0
            
            # Biến động thấp -> đòn bẩy cao hơn, biến động cao -> đòn bẩy thấp hơn
            if current_volatility < 1.0:
                return min(5.0, max_leverage)  # Biến động thấp, an toàn hơn
            elif current_volatility < 2.0:
                return min(3.0, max_leverage)  # Biến động trung bình
            else:
                return min(2.0, max_leverage)  # Biến động cao, thận trọng
        
        # Mặc định
        return min(3.0, max_leverage)
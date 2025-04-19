# --- Thư viện cần thiết ---
import os
import sys
import numpy as np
import pandas as pd
import logging
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# Thêm đường dẫn cha vào sys.path để import các module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from market_sentiment import get_sentiment_signal, integrate_sentiment_with_technical

# Import các hàm và lớp cần thiết cho chiến lược
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (merge_informative_pair, 
                               DecimalParameter, IntParameter, 
                               CategoricalParameter, BooleanParameter)
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes

logger = logging.getLogger(__name__)

# --- Định nghĩa các hàm chỉ báo kỹ thuật ---
def EMA(dataframe, timeperiod=20):
    """Exponential Moving Average"""
    from pandas import Series
    return Series(dataframe).ewm(span=timeperiod, min_periods=timeperiod).mean()

def SMA(series, timeperiod=5):
    """Simple Moving Average"""
    return series.rolling(timeperiod).mean()

def RSI(dataframe, timeperiod=14):
    """Relative Strength Index"""
    from ta.momentum import RSIIndicator
    return RSIIndicator(close=dataframe, window=timeperiod).rsi()

def ATR(dataframe, timeperiod=14):
    """Average True Range"""
    from ta.volatility import AverageTrueRange
    return AverageTrueRange(high=dataframe['high'], 
                           low=dataframe['low'], 
                           close=dataframe['close'], 
                           window=timeperiod).average_true_range()

def BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    from ta.volatility import BollingerBands
    bollinger = BollingerBands(close=dataframe['close'], window=timeperiod, window_dev=nbdevup)
    return {
        'upper': bollinger.bollinger_hband(),
        'mid': bollinger.bollinger_mavg(),
        'lower': bollinger.bollinger_lband()
    }

def MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    from ta.trend import MACD as ta_MACD
    macd_indicator = ta_MACD(close=dataframe['close'], 
                             window_slow=slowperiod, 
                             window_fast=fastperiod, 
                             window_sign=signalperiod)
    return {
        'macd': macd_indicator.macd(),
        'signal': macd_indicator.macd_signal(),
        'histogram': macd_indicator.macd_diff()
    }

def STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    from ta.momentum import StochasticOscillator
    stoch = StochasticOscillator(high=dataframe['high'], 
                                low=dataframe['low'], 
                                close=dataframe['close'], 
                                window=fastk_period, 
                                smooth_window=slowk_period)
    return {
        'fastk': stoch.stoch(),
        'fastd': stoch.stoch_signal()
    }

def MFI(dataframe, timeperiod=14):
    """Money Flow Index"""
    from ta.volume import MFIIndicator
    return MFIIndicator(high=dataframe['high'], 
                        low=dataframe['low'], 
                        close=dataframe['close'], 
                        volume=dataframe['volume'], 
                        window=timeperiod).money_flow_index()

class EnhancedCryptoStrategy(IStrategy):
    """
    Chiến lược AI nâng cao cho giao dịch cryptocurrency.
    Kết hợp các chỉ báo kỹ thuật với dự đoán từ mô hình AI để tạo tín hiệu giao dịch.
    """
    
    # --- Thông số chiến lược ---
    # Mục tiêu lợi nhuận
    minimal_roi = {
        "0": 0.03,  # Mục tiêu lợi nhuận 3%
        "30": 0.02,  # Sau 30 phút, chốt lời ở 2%
        "60": 0.01,  # Sau 60 phút, chốt lời ở 1%
        "180": 0.005,  # Sau 180 phút, chốt lời ở 0.5%
    }

    # Cắt lỗ ở 2%
    stoploss = -0.02
    
    # Khung thời gian 5 phút
    timeframe = '5m'
    
    # Tính năng trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True
    
    # Sử dụng custom stoploss
    use_custom_stoploss = True
    
    # Chỉ xử lý nến mới
    process_only_new_candles = True
    
    # Số nến cần thiết để khởi động
    startup_candle_count = 100
    
    # Tham số có thể tối ưu hóa
    # Ngưỡng buy RSI
    buy_rsi = IntParameter(25, 40, default=30, space="buy", optimize=True)
    
    # Ngưỡng sell RSI
    sell_rsi = IntParameter(65, 80, default=70, space="sell", optimize=True)
    
    # Ngưỡng mua dựa trên dự đoán AI
    buy_ai_threshold = DecimalParameter(0.65, 0.85, default=0.75, space="buy", optimize=True)
    
    # Ngưỡng bán dựa trên dự đoán AI
    sell_ai_threshold = DecimalParameter(0.65, 0.85, default=0.75, space="sell", optimize=True)
    
    # Độ lệch so với Bollinger Band để mua
    buy_bb_offset = DecimalParameter(0.97, 0.99, default=0.98, space="buy", optimize=True)
    
    # Đòn bẩy mặc định
    leverage_default = 3
    
    # Đòn bẩy tối đa
    leverage_max = 8
    
    # Phần trăm tối đa của tài sản dùng cho một cặp
    max_pair_stake = 33.0  # Điều chỉnh để tối đa 3 cặp (3 x 33% ~ 100% tài sản)
    
    # Đường dẫn đến thư mục mô hình AI
    models_dir = os.path.join('user_data', 'models')

    def __init__(self, config: dict) -> None:
        """
        Khởi tạo chiến lược với cấu hình được cung cấp.
        """
        super().__init__(config)
        self.config = config
        
        # Kiểm tra và tạo thư mục models nếu chưa tồn tại
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Tải các mô hình AI nếu có
        self.trend_model = self._load_model('trend_model.joblib')
        self.volatility_model = self._load_model('volatility_model.joblib')
        self.reversal_model = self._load_model('reversal_model.joblib')
        
        # Thời gian nạp lại mô hình
        self.last_model_load_time = datetime.now()
        
        logger.info("Đã khởi tạo EnhancedCryptoStrategy với cấu hình")

    def _load_model(self, model_name):
        """
        Tải mô hình từ file.
        """
        model_path = os.path.join(self.models_dir, model_name)
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
                logger.info(f"Đã tải mô hình từ {model_path}")
                return model
            except Exception as e:
                logger.warning(f"Không thể tải mô hình từ {model_path}: {e}")
        else:
            logger.warning(f"Không tìm thấy mô hình tại {model_path}")
        return None

    def informative_pairs(self) -> List[Tuple[str, str]]:
        """
        Return các cặp thông tin bổ sung cho các cặp giao dịch hiện tại
        """
        # Lấy danh sách các cặp giao dịch hiện tại
        pairs = self.dp.current_whitelist()
        
        # Thêm thông tin từ khung thời gian cao hơn (15m)
        informative_pairs = [(pair, '15m') for pair in pairs]
        
        # Thêm thông tin từ khung thời gian cao hơn nữa (1h)
        informative_pairs.extend([(pair, '1h') for pair in pairs])
        
        return informative_pairs

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Thêm các chỉ báo vào dataframe
        """
        # Kiểm tra xem đã quá lâu kể từ lần cuối nạp mô hình hay chưa (mỗi 6 giờ)
        if (datetime.now() - self.last_model_load_time).total_seconds() > 21600:
            self.trend_model = self._load_model('trend_model.joblib')
            self.volatility_model = self._load_model('volatility_model.joblib')
            self.reversal_model = self._load_model('reversal_model.joblib')
            self.last_model_load_time = datetime.now()
        
        # --- THÊM CÁC CHỈ BÁO KỸ THUẬT ---
        # 1. RSI
        dataframe['rsi'] = RSI(dataframe['close'], timeperiod=14)
        
        # 2. MACD
        macd = MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['signal']
        dataframe['macdhist'] = macd['histogram']
        
        # 3. Bollinger Bands
        bollinger = BBANDS(dataframe)
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_mid'] = bollinger['mid']
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_width'] = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_mid']
        
        # 4. Average True Range (ATR)
        dataframe['atr'] = ATR(dataframe)
        
        # 5. EMA
        dataframe['ema_9'] = EMA(dataframe['close'], 9)
        dataframe['ema_21'] = EMA(dataframe['close'], 21)
        dataframe['ema_50'] = EMA(dataframe['close'], 50)
        dataframe['ema_200'] = EMA(dataframe['close'], 200)
        
        # 6. SMA
        dataframe['sma_50'] = SMA(dataframe['close'], 50)
        dataframe['sma_200'] = SMA(dataframe['close'], 200)
        
        # 7. Stochastic Oscillator
        stoch = STOCH(dataframe)
        dataframe['stoch_k'] = stoch['fastk']
        dataframe['stoch_d'] = stoch['fastd']
        
        # 8. Money Flow Index
        dataframe['mfi'] = MFI(dataframe)
        
        # 9. Price distance from EMA
        dataframe['distance_ema_200'] = ((dataframe['close'] - dataframe['ema_200']) / dataframe['close']) * 100
        
        # 10. Volume Rate of Change
        dataframe['volume_mean'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_roc'] = (dataframe['volume'] / dataframe['volume_mean'])
        
        # 11. Trend Direction (1=Uptrend, -1=Downtrend, 0=Sideways)
        dataframe['trend'] = 0
        dataframe.loc[(dataframe['ema_50'] > dataframe['sma_200']) & 
                     (dataframe['close'] > dataframe['ema_50']), 'trend'] = 1
        dataframe.loc[(dataframe['ema_50'] < dataframe['sma_200']) & 
                     (dataframe['close'] < dataframe['ema_50']), 'trend'] = -1
        
        # 12. Overbought/Oversold
        dataframe['is_oversold'] = 0
        dataframe.loc[(dataframe['rsi'] < 30) & 
                     (dataframe['stoch_k'] < 20) & 
                     (dataframe['mfi'] < 20), 'is_oversold'] = 1
        
        dataframe['is_overbought'] = 0
        dataframe.loc[(dataframe['rsi'] > 70) & 
                     (dataframe['stoch_k'] > 80) & 
                     (dataframe['mfi'] > 80), 'is_overbought'] = 1
        
        # 13. Volatility
        dataframe['volatility'] = dataframe['atr'] / dataframe['close'] * 100
        
        # 14. Candle Patterns
        dataframe['body'] = abs(dataframe['open'] - dataframe['close'])
        dataframe['wick_high'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['wick_low'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        dataframe['is_hammer'] = 0
        dataframe.loc[(dataframe['body'] > 0) &
                     (dataframe['wick_low'] >= 2 * dataframe['body']) &
                     (dataframe['wick_high'] <= 0.2 * dataframe['wick_low']), 'is_hammer'] = 1
        
        dataframe['is_shooting_star'] = 0
        dataframe.loc[(dataframe['body'] > 0) &
                     (dataframe['wick_high'] >= 2 * dataframe['body']) &
                     (dataframe['wick_low'] <= 0.2 * dataframe['wick_high']), 'is_shooting_star'] = 1
        
        # 15. Thêm phân tích tâm lý thị trường
        try:
            symbol = metadata['pair'].split('/')[0]
            dataframe = integrate_sentiment_with_technical(dataframe, symbol)
        except Exception as e:
            logger.warning(f"Không thể tích hợp phân tích tâm lý thị trường: {e}")
        
        # 16. Thêm dự đoán từ mô hình AI
        dataframe = self.ai_prediction(dataframe)
        
        # Xử lý dữ liệu thiếu
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)
        dataframe = dataframe.fillna(0)
        
        return dataframe

    def ai_prediction(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm dự đoán AI vào dataframe
        """
        # Chỉ thêm dự đoán nếu đủ dữ liệu và mô hình tồn tại
        if len(dataframe) < 50 or (self.trend_model is None and self.volatility_model is None and self.reversal_model is None):
            # Tạo các cột với giá trị mặc định
            dataframe['trend_predict'] = 0.5
            dataframe['volatility_predict'] = 0.5
            dataframe['reversal_predict'] = 0.5
            dataframe['ai_signal'] = 0
            return dataframe
            
        try:
            # Chuẩn bị dữ liệu cho mô hình
            features = self._prepare_features(dataframe)
            
            # Dự đoán xu hướng (1=Up, 0=Down)
            if self.trend_model is not None:
                try:
                    dataframe['trend_predict'] = self.trend_model.predict_proba(features)[:, 1]
                except Exception as e:
                    logger.warning(f"Lỗi khi dự đoán xu hướng: {e}")
                    dataframe['trend_predict'] = 0.5
            else:
                dataframe['trend_predict'] = 0.5
            
            # Dự đoán biến động (1=Cao, 0=Thấp)
            if self.volatility_model is not None:
                try:
                    dataframe['volatility_predict'] = self.volatility_model.predict_proba(features)[:, 1]
                except Exception as e:
                    logger.warning(f"Lỗi khi dự đoán biến động: {e}")
                    dataframe['volatility_predict'] = 0.5
            else:
                dataframe['volatility_predict'] = 0.5
            
            # Dự đoán điểm đảo chiều (1=Có, 0=Không)
            if self.reversal_model is not None:
                try:
                    dataframe['reversal_predict'] = self.reversal_model.predict_proba(features)[:, 1]
                except Exception as e:
                    logger.warning(f"Lỗi khi dự đoán điểm đảo chiều: {e}")
                    dataframe['reversal_predict'] = 0.5
            else:
                dataframe['reversal_predict'] = 0.5
            
            # Tính toán tín hiệu AI kết hợp
            dataframe['ai_signal'] = 0
            
            # Tín hiệu Long: Xu hướng tăng, biến động trong kiểm soát, không có đảo chiều
            dataframe.loc[(dataframe['trend_predict'] > self.buy_ai_threshold.value) & 
                         (dataframe['volatility_predict'] < 0.7) & 
                         (dataframe['reversal_predict'] < 0.3), 'ai_signal'] = 1
            
            # Tín hiệu Short: Xu hướng giảm, biến động trong kiểm soát, không có đảo chiều
            dataframe.loc[(dataframe['trend_predict'] < (1 - self.buy_ai_threshold.value)) & 
                         (dataframe['volatility_predict'] < 0.7) & 
                         (dataframe['reversal_predict'] < 0.3), 'ai_signal'] = -1
            
            # Tín hiệu đóng vị thế: Phát hiện điểm đảo chiều
            dataframe.loc[dataframe['reversal_predict'] > 0.7, 'ai_signal'] = 0
            
        except Exception as e:
            logger.error(f"Lỗi khi thêm dự đoán AI: {e}")
            dataframe['trend_predict'] = 0.5
            dataframe['volatility_predict'] = 0.5
            dataframe['reversal_predict'] = 0.5
            dataframe['ai_signal'] = 0
            
        return dataframe

    def _prepare_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu đặc trưng cho mô hình AI
        """
        # Chọn các tính năng quan trọng từ dataframe
        features = dataframe[[
            'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_width',
            'atr', 'volatility', 'mfi', 'stoch_k', 'stoch_d',
            'distance_ema_200', 'volume_roc', 'trend'
        ]].copy()
        
        # Thêm các tính năng tương đối
        features['close_rel_ema9'] = dataframe['close'] / dataframe['ema_9'] - 1
        features['close_rel_ema21'] = dataframe['close'] / dataframe['ema_21'] - 1
        features['close_rel_ema50'] = dataframe['close'] / dataframe['ema_50'] - 1
        features['close_rel_sma50'] = dataframe['close'] / dataframe['sma_50'] - 1
        
        # Thêm lag features (dữ liệu lịch sử)
        for lag in [1, 2, 3, 5]:
            features[f'close_change_{lag}'] = dataframe['close'].pct_change(lag)
            features[f'volume_change_{lag}'] = dataframe['volume'].pct_change(lag)
            features[f'rsi_change_{lag}'] = features['rsi'].diff(lag)
        
        # Xử lý dữ liệu thiếu
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        return features

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Phương thức mới dùng cho cả long và short (thay thế cho populate_buy_trend)
        """
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        
        # Không giao dịch nếu không đủ dữ liệu
        if len(dataframe) < 50:
            return dataframe

        # Điều kiện vào lệnh Long
        long_conditions = []
        
        # Điều kiện 1: Tín hiệu AI
        long_conditions.append(dataframe['ai_signal'] == 1)
        
        # Điều kiện 2: Các chỉ báo kỹ thuật truyền thống
        long_conditions.append(
            (dataframe['rsi'] < self.buy_rsi.value) &  # RSI thấp
            (dataframe['close'] < dataframe['bb_lower'] * self.buy_bb_offset.value) &  # Giá gần BB dưới
            (dataframe['macd'] > dataframe['macdsignal']) &  # MACD trên Signal
            (dataframe['ema_50'] > dataframe['ema_200'])  # EMA50 trên EMA200 (uptrend)
        )
        
        # Điều kiện 3: Đánh giá tâm lý thị trường
        if 'sentiment_score' in dataframe.columns:
            long_conditions.append(dataframe['sentiment_score'] > 0.3)  # Tâm lý tích cực
        
        # Kết hợp các điều kiện với "hoặc"
        if long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, long_conditions),
                'enter_long'
            ] = 1
        
        # Điều kiện vào lệnh Short
        short_conditions = []
        
        # Điều kiện 1: Tín hiệu AI
        short_conditions.append(dataframe['ai_signal'] == -1)
        
        # Điều kiện 2: Các chỉ báo kỹ thuật truyền thống
        short_conditions.append(
            (dataframe['rsi'] > (100 - self.buy_rsi.value)) &  # RSI cao
            (dataframe['close'] > dataframe['bb_upper'] * (2 - self.buy_bb_offset.value)) &  # Giá gần BB trên
            (dataframe['macd'] < dataframe['macdsignal']) &  # MACD dưới Signal
            (dataframe['ema_50'] < dataframe['ema_200'])  # EMA50 dưới EMA200 (downtrend)
        )
        
        # Điều kiện 3: Đánh giá tâm lý thị trường
        if 'sentiment_score' in dataframe.columns:
            short_conditions.append(dataframe['sentiment_score'] < -0.3)  # Tâm lý tiêu cực
        
        # Kết hợp các điều kiện với "hoặc"
        if short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, short_conditions),
                'enter_short'
            ] = 1
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Phương thức mới dùng cho cả đóng long và short (thay thế cho populate_sell_trend)
        """
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        
        # Không giao dịch nếu không đủ dữ liệu
        if len(dataframe) < 50:
            return dataframe
        
        # Điều kiện thoát lệnh Long
        exit_long_conditions = []
        
        # Điều kiện 1: Phát hiện đảo chiều từ AI
        exit_long_conditions.append(dataframe['reversal_predict'] > 0.7)
        
        # Điều kiện 2: Các chỉ báo kỹ thuật truyền thống
        exit_long_conditions.append(
            (dataframe['rsi'] > self.sell_rsi.value) &  # RSI cao
            (dataframe['close'] > dataframe['bb_upper']) &  # Giá vượt BB trên
            (dataframe['macdhist'] < 0)  # MACD Histogram âm
        )
        
        # Điều kiện 3: Đánh giá tâm lý thị trường đổi chiều
        if 'sentiment_score' in dataframe.columns:
            exit_long_conditions.append(
                (dataframe['sentiment_score'] < -0.3) &  # Tâm lý chuyển tiêu cực
                (dataframe['sentiment_score'].shift(1) > 0)  # Trước đó tích cực
            )
        
        # Kết hợp các điều kiện với "hoặc"
        if exit_long_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, exit_long_conditions),
                'exit_long'
            ] = 1
        
        # Điều kiện thoát lệnh Short
        exit_short_conditions = []
        
        # Điều kiện 1: Phát hiện đảo chiều từ AI
        exit_short_conditions.append(dataframe['reversal_predict'] > 0.7)
        
        # Điều kiện 2: Các chỉ báo kỹ thuật truyền thống
        exit_short_conditions.append(
            (dataframe['rsi'] < (100 - self.sell_rsi.value)) &  # RSI thấp
            (dataframe['close'] < dataframe['bb_lower']) &  # Giá dưới BB dưới
            (dataframe['macdhist'] > 0)  # MACD Histogram dương
        )
        
        # Điều kiện 3: Đánh giá tâm lý thị trường đổi chiều
        if 'sentiment_score' in dataframe.columns:
            exit_short_conditions.append(
                (dataframe['sentiment_score'] > 0.3) &  # Tâm lý chuyển tích cực
                (dataframe['sentiment_score'].shift(1) < 0)  # Trước đó tiêu cực
            )
        
        # Kết hợp các điều kiện với "hoặc"
        if exit_short_conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, exit_short_conditions),
                'exit_short'
            ] = 1
        
        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Điều chỉnh đòn bẩy dựa trên biến động của thị trường.
        """
        # Lấy dữ liệu từ 20 nến gần nhất
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Chỉ lấy 20 nến gần nhất
        dataframe = dataframe.iloc[-20:]
        
        # Nếu không đủ dữ liệu, sử dụng đòn bẩy mặc định
        if len(dataframe) < 20:
            return self.leverage_default
        
        try:
            # Tính volatility trung bình
            volatility = dataframe['volatility'].mean()
            
            # Điều chỉnh đòn bẩy dựa trên biến động
            if volatility > 2.5:  # Biến động cao
                return 3  # Đòn bẩy tối thiểu
            elif volatility > 1.5:  # Biến động trung bình
                return 5  # Đòn bẩy trung bình
            else:  # Biến động thấp
                return min(8, max_leverage)  # Đòn bẩy cao nhất, không vượt quá max_leverage
                
        except Exception as e:
            logger.warning(f"Lỗi khi tính toán đòn bẩy: {e}")
            return self.leverage_default

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss dựa trên ATR và mức lãi/lỗ hiện tại.
        """
        # Lấy dữ liệu
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Nếu không đủ dữ liệu, sử dụng stoploss mặc định
        if len(dataframe) < 20:
            return self.stoploss
        
        last_candle = dataframe.iloc[-1].squeeze()
        
        try:
            # Sử dụng ATR để tính stoploss động
            atr_value = last_candle['atr']
            atr_stoploss = -2.0 * atr_value / current_rate  # 2x ATR
            
            # Điều chỉnh stoploss dựa trên lợi nhuận hiện tại
            if current_profit > 0.02:  # Lợi nhuận > 2%
                # Bảo vệ 50% lợi nhuận
                return max(atr_stoploss, -(current_profit * 0.5))
            elif current_profit > 0.01:  # Lợi nhuận > 1%
                # Bảo vệ 20% lợi nhuận
                return max(atr_stoploss, -(current_profit * 0.8))
            else:
                return atr_stoploss
                
        except Exception as e:
            logger.warning(f"Lỗi khi tính toán custom stoploss: {e}")
            return self.stoploss

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """
        Xác nhận việc vào lệnh với thêm một lớp kiểm tra bổ sung.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Không mở thêm vị thế nếu đã có nhiều giao dịch mở
        open_trades = self.wallets.get_open_trades()
        
        if side == 'long':
            # Ngừng mở vị thế long nếu xu hướng giảm
            if last_candle['trend'] == -1:
                return False
                
            # Kiểm tra tâm lý thị trường
            if 'sentiment_score' in dataframe.columns:
                if last_candle['sentiment_score'] < -0.5:  # Tâm lý rất tiêu cực
                    return False
        
        elif side == 'short':
            # Ngừng mở vị thế short nếu xu hướng tăng
            if last_candle['trend'] == 1:
                return False
                
            # Kiểm tra tâm lý thị trường
            if 'sentiment_score' in dataframe.columns:
                if last_candle['sentiment_score'] > 0.5:  # Tâm lý rất tích cực
                    return False
        
        # Giới hạn số lượng vị thế mở cho mỗi cặp
        pair_open_trades = [trade for trade in open_trades if trade.pair == pair]
        if len(pair_open_trades) >= 2:
            return False
            
        # Giới hạn phần trăm tài sản dùng cho một cặp
        stake_amount = self.wallets.get_total_stake_amount()
        pair_stake = sum(trade.stake_amount for trade in pair_open_trades)
        if (pair_stake + amount) / stake_amount * 100 > self.max_pair_stake:
            return False
        
        return True

# Import ở đây để tránh lỗi circular import
from functools import reduce
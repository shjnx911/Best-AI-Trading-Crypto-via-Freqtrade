"""
AI Profit Optimizer
Module tối ưu hóa lợi nhuận dựa trên AI để đạt mục tiêu 3% lợi nhuận hàng ngày
"""

import numpy as np
import pandas as pd
import logging
import os
import time
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import matplotlib.pyplot as plt
import requests
from ccxt.base.errors import AuthenticationError, NetworkError
import talib.abstract as ta
from dotenv import load_dotenv
import traceback
import sys
import openai
from threading import Lock

# Cố gắng load biến môi trường
load_dotenv()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ai_profit_optimizer.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ai_profit_optimizer")

# Biến toàn cục để theo dõi trạng thái
class TrackingState:
    def __init__(self):
        self.daily_profit_pct = 0.0
        self.daily_profit_target_pct = 3.0
        self.num_trades_today = 0
        self.max_trades_per_day = 20
        self.active_pairs = []
        self.max_active_pairs = 3
        self.potential_trades = []
        self.last_update_time = datetime.now()
        self.pairs_in_watchlist = []
        self.market_conditions = {
            "trend": "neutral",  # neutral, bullish, bearish
            "volatility": "medium",  # low, medium, high
            "sentiment": 0.0,  # -1 to 1 scale
            "risk_level": "medium"  # low, medium, high
        }
        self.lock = Lock()  # Thread lock for concurrency
        self.todays_trades = []
        self.profitable_trades = 0
        self.losing_trades = 0
        self.is_target_reached = False
        self.ai_enabled = True
        self.ai_prediction_cache = {}
        self.opportunity_score_threshold = 75
        self.opportunity_score_threshold_after_target = 85

    def update_daily_profit(self, profit_pct):
        with self.lock:
            self.daily_profit_pct += profit_pct
            if self.daily_profit_pct >= self.daily_profit_target_pct:
                self.is_target_reached = True
                logger.info(f"🎯 Mục tiêu lợi nhuận hàng ngày {self.daily_profit_target_pct}% đã đạt được!")
            
    def reset_daily_tracking(self):
        with self.lock:
            self.daily_profit_pct = 0.0
            self.num_trades_today = 0
            self.todays_trades = []
            self.profitable_trades = 0
            self.losing_trades = 0
            self.is_target_reached = False
            self.ai_prediction_cache = {}
            self.last_update_time = datetime.now()
            
    def add_trade(self, trade_info):
        with self.lock:
            self.todays_trades.append(trade_info)
            self.num_trades_today += 1
            if trade_info['profit_pct'] > 0:
                self.profitable_trades += 1
            else:
                self.losing_trades += 1
            
    def can_open_new_trade(self, pair, opportunity_score):
        with self.lock:
            # Kiểm tra xem đã đạt mục tiêu lợi nhuận hàng ngày chưa
            if self.is_target_reached:
                # Nếu đã đạt mục tiêu, chỉ mở trade với opportunity score cao hơn ngưỡng
                if opportunity_score < self.opportunity_score_threshold_after_target:
                    return False, f"Đã đạt mục tiêu lợi nhuận {self.daily_profit_target_pct}%, cần score >= {self.opportunity_score_threshold_after_target}"
            else:
                # Chưa đạt mục tiêu, áp dụng ngưỡng thông thường
                if opportunity_score < self.opportunity_score_threshold:
                    return False, f"Opportunity score {opportunity_score} thấp hơn ngưỡng {self.opportunity_score_threshold}"
            
            # Kiểm tra số lượng trade tối đa mỗi ngày
            if self.num_trades_today >= self.max_trades_per_day:
                return False, f"Đã đạt số lượng trade tối đa trong ngày ({self.max_trades_per_day})"
            
            # Kiểm tra số lượng cặp tiền đang giao dịch
            if len(self.active_pairs) >= self.max_active_pairs and pair not in self.active_pairs:
                return False, f"Đã đạt số lượng cặp tiền giao dịch đồng thời tối đa ({self.max_active_pairs})"
            
            return True, "Có thể mở trade mới"

# Khởi tạo trạng thái toàn cục
state = TrackingState()

class AIModel:
    """Class quản lý các mô hình AI dùng trong tối ưu hóa lợi nhuận"""
    
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.profit_predictor = None
        self.entry_classifier = None
        self.exit_classifier = None
        self.scaler = None
        self.features = None
        self.load_models()
        
    def load_models(self):
        """Load các mô hình đã được huấn luyện từ đĩa"""
        try:
            # Tạo thư mục chứa model nếu chưa tồn tại
            os.makedirs(self.model_dir, exist_ok=True)
            
            if os.path.exists(f"{self.model_dir}/profit_predictor.pkl"):
                self.profit_predictor = joblib.load(f"{self.model_dir}/profit_predictor.pkl")
                logger.info("Đã tải mô hình profit_predictor")
            
            if os.path.exists(f"{self.model_dir}/entry_classifier.pkl"):
                self.entry_classifier = joblib.load(f"{self.model_dir}/entry_classifier.pkl")
                logger.info("Đã tải mô hình entry_classifier")
            
            if os.path.exists(f"{self.model_dir}/exit_classifier.pkl"):
                self.exit_classifier = joblib.load(f"{self.model_dir}/exit_classifier.pkl")
                logger.info("Đã tải mô hình exit_classifier")
            
            if os.path.exists(f"{self.model_dir}/scaler.pkl"):
                self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
                logger.info("Đã tải scaler")
            
            if os.path.exists(f"{self.model_dir}/features.json"):
                with open(f"{self.model_dir}/features.json", "r") as f:
                    self.features = json.load(f)
                logger.info(f"Đã tải danh sách {len(self.features)} features")
            
        except Exception as e:
            logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            logger.exception(e)
    
    def save_models(self):
        """Lưu các mô hình đã được huấn luyện ra đĩa"""
        try:
            # Tạo thư mục chứa model nếu chưa tồn tại
            os.makedirs(self.model_dir, exist_ok=True)
            
            if self.profit_predictor:
                joblib.dump(self.profit_predictor, f"{self.model_dir}/profit_predictor.pkl")
            
            if self.entry_classifier:
                joblib.dump(self.entry_classifier, f"{self.model_dir}/entry_classifier.pkl")
            
            if self.exit_classifier:
                joblib.dump(self.exit_classifier, f"{self.model_dir}/exit_classifier.pkl")
            
            if self.scaler:
                joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")
            
            if self.features:
                with open(f"{self.model_dir}/features.json", "w") as f:
                    json.dump(self.features, f)
            
            logger.info("Đã lưu tất cả mô hình")
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu mô hình: {str(e)}")
            logger.exception(e)
    
    def train_profit_predictor(self, X_train, y_train):
        """Huấn luyện mô hình dự đoán lợi nhuận"""
        try:
            # Sử dụng LightGBM cho dự đoán lợi nhuận
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=32,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            self.profit_predictor = model
            
            # Đánh giá quan trọng của features
            feature_importance = model.feature_importances_
            features_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            logger.info(f"Top 10 features quan trọng nhất cho dự đoán lợi nhuận:\n{features_df.head(10)}")
            
            # Lưu mô hình
            self.save_models()
            return model
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình dự đoán lợi nhuận: {str(e)}")
            logger.exception(e)
            return None
    
    def train_entry_classifier(self, X_train, y_train):
        """Huấn luyện mô hình phân loại điểm vào lệnh tốt/xấu"""
        try:
            # Sử dụng LightGBM cho phân loại điểm vào lệnh
            model = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                num_leaves=32,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight='balanced',
                random_state=42
            )
            
            model.fit(X_train, y_train)
            self.entry_classifier = model
            
            # Đánh giá quan trọng của features
            feature_importance = model.feature_importances_
            features_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': feature_importance
            }).sort_values(by='Importance', ascending=False)
            
            logger.info(f"Top 10 features quan trọng nhất cho phân loại điểm vào lệnh:\n{features_df.head(10)}")
            
            # Lưu mô hình
            self.save_models()
            return model
            
        except Exception as e:
            logger.error(f"Lỗi khi huấn luyện mô hình phân loại điểm vào lệnh: {str(e)}")
            logger.exception(e)
            return None
    
    def predict_profit(self, features_df):
        """Dự đoán lợi nhuận tiềm năng từ một trade"""
        try:
            if self.profit_predictor is None:
                logger.warning("Chưa có mô hình dự đoán lợi nhuận được tải")
                return None
            
            # Chuẩn hóa features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            
            # Dự đoán
            predicted_profit = self.profit_predictor.predict(features_df)
            return predicted_profit
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán lợi nhuận: {str(e)}")
            logger.exception(e)
            return None
    
    def predict_entry_quality(self, features_df):
        """Dự đoán chất lượng của điểm vào lệnh (xác suất thành công)"""
        try:
            if self.entry_classifier is None:
                logger.warning("Chưa có mô hình phân loại điểm vào lệnh được tải")
                return None
            
            # Chuẩn hóa features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
            
            # Dự đoán xác suất
            entry_proba = self.entry_classifier.predict_proba(features_df)[:, 1]  # Lấy xác suất class 1 (good entry)
            return entry_proba
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán chất lượng điểm vào lệnh: {str(e)}")
            logger.exception(e)
            return None
    
    def calculate_opportunity_score(self, dataframe, pair, current_time, timeframe="15m"):
        """
        Tính điểm cơ hội giao dịch (opportunity score) cho một cặp tiền
        
        Thang điểm 0-100:
        - < 50: Tín hiệu yếu, bỏ qua
        - 50-75: Tín hiệu trung bình, xem xét nếu không có tùy chọn tốt hơn
        - 75-90: Tín hiệu tốt, xem xét vào lệnh
        - > 90: Tín hiệu rất tốt, khả năng cao đem lại lợi nhuận
        """
        try:
            # Kiểm tra cache
            cache_key = f"{pair}_{timeframe}_{current_time.timestamp()}"
            if cache_key in state.ai_prediction_cache:
                return state.ai_prediction_cache[cache_key]
            
            # Chuyển đổi dataframe thành features
            features_df = self._extract_features(dataframe)
            
            if features_df is None or len(features_df) == 0:
                return 0
            
            # Lấy last row làm features hiện tại
            current_features = features_df.iloc[-1:].copy()
            
            # Component 1: Dự đoán lợi nhuận tiềm năng (0-40 điểm)
            profit_score = 0
            if self.profit_predictor is not None:
                predicted_profit = self.predict_profit(current_features)
                if predicted_profit is not None and len(predicted_profit) > 0:
                    # Scale predicted profit to 0-40 points
                    # Ví dụ 3% profit -> 40 điểm
                    # 0% -> 0 điểm
                    # <0% -> 0 điểm
                    profit_pct = predicted_profit[0]
                    profit_score = min(40, max(0, profit_pct * 40 / 3.0))
            
            # Component 2: Chất lượng điểm vào lệnh (0-40 điểm)
            entry_score = 0
            if self.entry_classifier is not None:
                entry_proba = self.predict_entry_quality(current_features)
                if entry_proba is not None and len(entry_proba) > 0:
                    # Xác suất success * 40 điểm
                    entry_score = entry_proba[0] * 40
            
            # Component 3: Điều kiện thị trường và vị thế hiện tại (0-20 điểm)
            market_score = self._calculate_market_conditions_score(dataframe, pair)
            
            # Tính tổng điểm
            total_score = profit_score + entry_score + market_score
            
            # Thêm vào cache
            state.ai_prediction_cache[cache_key] = total_score
            
            logger.info(f"Opportunity Score cho {pair}: {total_score:.2f} (Profit: {profit_score:.2f}, Entry: {entry_score:.2f}, Market: {market_score:.2f})")
            
            return total_score
            
        except Exception as e:
            logger.error(f"Lỗi khi tính opportunity score: {str(e)}")
            logger.exception(e)
            return 0
    
    def _extract_features(self, dataframe):
        """Trích xuất features từ dataframe để đưa vào mô hình"""
        try:
            df = dataframe.copy()
            
            # Thêm các chỉ báo kỹ thuật cơ bản
            # RSI
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            df['rsi_6'] = ta.RSI(df['close'], timeperiod=6)
            
            # MACD
            macd = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd['macd']
            df['macdsignal'] = macd['macdsignal']
            df['macdhist'] = macd['macdhist']
            
            # Bollinger Bands
            upperband, middleband, lowerband = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            df['bb_upper'] = upperband
            df['bb_middle'] = middleband
            df['bb_lower'] = lowerband
            df['bb_width'] = (upperband - lowerband) / middleband
            df['bb_percent'] = (df['close'] - lowerband) / (upperband - lowerband)
            
            # EMAs
            for period in [5, 8, 13, 21, 34, 55, 89, 200]:
                df[f'ema_{period}'] = ta.EMA(df['close'], timeperiod=period)
            
            # Đặc trưng về khối lượng
            df['volume_mean_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_mean_20']
            
            # ADX
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # ATR
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            df['atr_percent'] = df['atr'] / df['close'] * 100
            
            # Stochastic
            slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Ichimoku
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            df['tenkan_sen'] = (high_9 + low_9) / 2  # Conversion line
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            df['kijun_sen'] = (high_26 + low_26) / 2  # Base line
            
            # Candle patterns
            df['body_size'] = abs(df['close'] - df['open'])
            df['shadow_upper'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['shadow_lower'] = df[['close', 'open']].min(axis=1) - df['low']
            df['body_percentage'] = df['body_size'] / (df['high'] - df['low'])
            
            # Đặc trưng momentum
            df['momentum'] = df['close'].pct_change(periods=14) * 100
            
            # Price action features
            df['price_up'] = (df['close'] > df['open']).astype(int)
            df['consecutive_ups'] = df['price_up'].rolling(window=3).sum()
            
            # Xử lý giá trị NaN
            df = df.dropna()
            
            if len(df) == 0:
                return None
            
            # Chọn features từ danh sách nếu đã được xác định
            if self.features is not None:
                available_features = [f for f in self.features if f in df.columns]
                return df[available_features]
            
            # Nếu chưa có danh sách features, hãy sử dụng tất cả
            features_to_exclude = ['date', 'open_time']
            features_cols = [col for col in df.columns if col not in features_to_exclude]
            
            return df[features_cols]
            
        except Exception as e:
            logger.error(f"Lỗi khi trích xuất features: {str(e)}")
            logger.exception(e)
            return None
    
    def _calculate_market_conditions_score(self, dataframe, pair):
        """Tính điểm cho điều kiện thị trường hiện tại"""
        try:
            # Lấy 100 nến cuối cùng để phân tích
            df = dataframe.tail(100).copy()
            
            score = 10  # Điểm cơ bản
            
            # 1. Xác định xu hướng
            ema_8 = df['close'].ewm(span=8, adjust=False).mean()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean()
            
            trend = 'neutral'
            if ema_8.iloc[-1] > ema_21.iloc[-1] and ema_8.iloc[-2] > ema_21.iloc[-2]:
                trend = 'bullish'
                score += 3
            elif ema_8.iloc[-1] < ema_21.iloc[-1] and ema_8.iloc[-2] < ema_21.iloc[-2]:
                trend = 'bearish'
                score -= 2
            
            # 2. Xác định volatility
            atr = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            atr_pct = atr.iloc[-1] / df['close'].iloc[-1] * 100
            
            volatility = 'medium'
            if atr_pct < 1.0:
                volatility = 'low'
                score -= 1
            elif atr_pct > 3.0:
                volatility = 'high'
                score -= 2
            
            # 3. Kiểm tra volume
            volume_ratio = df['volume'].iloc[-1] / df['volume'].rolling(20).mean().iloc[-1]
            if volume_ratio > 1.5:
                score += 2
            elif volume_ratio < 0.7:
                score -= 1
            
            # 4. Kiểm tra nếu pair đã có trong danh sách theo dõi
            if pair in state.pairs_in_watchlist:
                score += 2
            
            # 5. Kiểm tra RSI
            rsi = ta.RSI(df['close'], timeperiod=14).iloc[-1]
            if 30 <= rsi <= 70:
                score += 2
            elif rsi < 30:
                # Oversold
                if trend == 'bullish':
                    score += 3  # Oversold + xu hướng tăng: cơ hội tốt
                else:
                    score += 1
            elif rsi > 70:
                # Overbought
                if trend == 'bearish':
                    score += 3  # Overbought + xu hướng giảm: cơ hội tốt
                else:
                    score += 1
            
            # Đảm bảo score trong khoảng 0-20
            market_score = max(0, min(20, score))
            
            # Cập nhật market conditions trong state
            state.market_conditions['trend'] = trend
            state.market_conditions['volatility'] = volatility
            
            return market_score
            
        except Exception as e:
            logger.error(f"Lỗi khi tính điểm điều kiện thị trường: {str(e)}")
            logger.exception(e)
            return 10  # Điểm mặc định

class ProfitOptimizer:
    """
    Class chính quản lý việc tối ưu hóa lợi nhuận hàng ngày
    """
    
    def __init__(self, exchange_config=None, backtest_data_dir="./backtest_data"):
        self.exchange_config = exchange_config
        self.backtest_data_dir = backtest_data_dir
        self.ai_model = AIModel()
        self.current_pairs = []
        self.last_analysis_time = datetime.now() - timedelta(hours=1)
        self.min_trades_needed = self._calculate_min_trades_needed()
        
        # Initialize directories
        os.makedirs(backtest_data_dir, exist_ok=True)
        
        # Fetch initial market data
        self._update_market_data()
    
    def _calculate_min_trades_needed(self):
        """
        Tính số lượng giao dịch tối thiểu cần thiết để đạt mục tiêu 3% hàng ngày
        dựa trên dữ liệu quá khứ về lợi nhuận trung bình mỗi giao dịch
        """
        # Default if no data: assuming 0.5% profit per trade, need 6 trades
        avg_profit_per_trade = 0.5
        
        # TODO: Phân tích dữ liệu quá khứ để có ước tính chính xác hơn
        
        min_trades = max(1, round(state.daily_profit_target_pct / avg_profit_per_trade))
        logger.info(f"Cần khoảng {min_trades} giao dịch để đạt mục tiêu {state.daily_profit_target_pct}% mỗi ngày")
        
        return min_trades
    
    def _update_market_data(self):
        """Cập nhật dữ liệu thị trường cho các cặp tiền đang theo dõi"""
        try:
            # Thêm code cập nhật dữ liệu từ exchange
            # Đây chỉ là phần giả lập, cần thay thế bằng API thực tế
            logger.info("Đang cập nhật dữ liệu thị trường...")
            
            # Giả lập cập nhật danh sách cặp tiền theo dõi
            if not state.pairs_in_watchlist:
                state.pairs_in_watchlist = [
                    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
                    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "DOGE/USDT", "SHIB/USDT"
                ]
            
            self.last_analysis_time = datetime.now()
            logger.info(f"Đã cập nhật dữ liệu thị trường cho {len(state.pairs_in_watchlist)} cặp tiền")
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật dữ liệu thị trường: {str(e)}")
            logger.exception(e)
    
    def analyze_daily_profit_needed(self):
        """
        Phân tích lợi nhuận cần thiết để đạt mục tiêu 3% hàng ngày
        Xác định số lượng và loại giao dịch cần thiết
        """
        try:
            # Cập nhật thông tin thị trường nếu đã quá 1 giờ từ lần cập nhật trước
            if (datetime.now() - self.last_analysis_time).seconds > 3600:
                self._update_market_data()
            
            # Tình trạng lợi nhuận hiện tại
            current_profit = state.daily_profit_pct
            target_profit = state.daily_profit_target_pct
            remaining_profit = max(0, target_profit - current_profit)
            
            # Nếu đã đạt mục tiêu, báo cáo và thoát
            if remaining_profit == 0:
                logger.info(f"Mục tiêu lợi nhuận hàng ngày {target_profit}% đã đạt được!")
                
                # Vẫn tính toán các chỉ số để chuẩn bị cho giao dịch trong tương lai
                avg_profit_per_trade = 0.5  # Giả định 0.5% mỗi trade
                remaining_trades = 0
                confidence_level = 100  # Độ tin cậy tối đa vì đã đạt mục tiêu
                
                return {
                    "status": "target_reached",
                    "current_profit": current_profit,
                    "target_profit": target_profit,
                    "remaining_profit": remaining_profit,
                    "remaining_trades_needed": remaining_trades,
                    "confidence_level": confidence_level,
                    "recommendation": "Đã đạt mục tiêu lợi nhuận hàng ngày, chỉ nên thực hiện các giao dịch có opportunity score > 85"
                }
            
            # Tính toán số lượng giao dịch còn cần để đạt mục tiêu
            avg_profit_this_far = 0
            if state.num_trades_today > 0:
                avg_profit_this_far = current_profit / state.num_trades_today
            
            # Sử dụng dữ liệu quá khứ hoặc hiện tại để ước tính lợi nhuận trung bình
            if avg_profit_this_far > 0:
                avg_profit_per_trade = avg_profit_this_far
            else:
                # Assume 0.5% average profit if no data
                avg_profit_per_trade = 0.5
            
            remaining_trades = max(1, round(remaining_profit / avg_profit_per_trade))
            
            # Tính toán thời gian còn lại trong ngày
            now = datetime.now()
            end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59)
            hours_remaining = (end_of_day - now).seconds / 3600
            
            # Tính toán độ tin cậy dựa trên khả năng hoàn thành mục tiêu
            trades_per_hour_needed = remaining_trades / max(1, hours_remaining)
            
            confidence_level = 100
            if trades_per_hour_needed > 3:
                # Quá khó để có hơn 3 giao dịch mỗi giờ
                confidence_level = 50
            elif trades_per_hour_needed > 2:
                confidence_level = 70
            elif trades_per_hour_needed > 1:
                confidence_level = 85
            
            # Chuẩn bị đề xuất
            recommendation = f"Cần thêm {remaining_trades} giao dịch với lợi nhuận trung bình {avg_profit_per_trade:.2f}% để đạt mục tiêu {target_profit}%"
            
            if confidence_level < 70:
                recommendation += ". Khó có thể đạt được mục tiêu hôm nay, nên xem xét các giao dịch với opportunity score cao hơn 80"
            
            # Ưu tiên các cặp tiền dựa trên opportunity score
            priority_pairs = self._get_priority_pairs()
            
            return {
                "status": "in_progress",
                "current_profit": current_profit,
                "target_profit": target_profit,
                "remaining_profit": remaining_profit,
                "avg_profit_per_trade": avg_profit_per_trade,
                "remaining_trades_needed": remaining_trades,
                "trades_per_hour_needed": trades_per_hour_needed,
                "hours_remaining": hours_remaining,
                "confidence_level": confidence_level,
                "recommendation": recommendation,
                "priority_pairs": priority_pairs
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích lợi nhuận hàng ngày: {str(e)}")
            logger.exception(e)
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_priority_pairs(self):
        """Xác định các cặp tiền ưu tiên dựa trên opportunity score"""
        try:
            # Đây là phần giả lập, cần thay thế bằng tính toán thực tế
            priority_pairs = []
            
            # Giả lập opportunity score cho các cặp tiền trong watchlist
            for pair in state.pairs_in_watchlist:
                # Random score from 0-100 for simulation
                score = np.random.uniform(0, 100)
                priority_pairs.append({
                    "pair": pair,
                    "opportunity_score": score,
                    "estimated_profit": score * 0.05 / 100  # 0.05% profit for each score point
                })
            
            # Sắp xếp theo opportunity score giảm dần
            priority_pairs = sorted(priority_pairs, key=lambda x: x["opportunity_score"], reverse=True)
            
            # Chỉ giữ top 5
            return priority_pairs[:5]
            
        except Exception as e:
            logger.error(f"Lỗi khi xác định cặp tiền ưu tiên: {str(e)}")
            logger.exception(e)
            return []
    
    def get_trade_recommendation(self, dataframe, pair, current_time, side="long", timeframe="15m"):
        """
        Đưa ra khuyến nghị giao dịch dựa trên dữ liệu hiện tại
        và các mô hình AI
        
        Returns:
            dict: Thông tin khuyến nghị giao dịch
        """
        try:
            # Kiểm tra tình trạng mục tiêu lợi nhuận hàng ngày
            daily_profit_status = self.analyze_daily_profit_needed()
            
            # Tính opportunity score cho cặp tiền này
            opportunity_score = self.ai_model.calculate_opportunity_score(dataframe, pair, current_time, timeframe)
            
            # Kiểm tra xem có thể mở giao dịch mới không
            can_open, reason = state.can_open_new_trade(pair, opportunity_score)
            
            if not can_open:
                return {
                    "pair": pair,
                    "timeframe": timeframe,
                    "side": side,
                    "recommendation": "no_trade",
                    "opportunity_score": opportunity_score,
                    "reason": reason,
                    "daily_profit_status": daily_profit_status
                }
            
            # Ước tính lợi nhuận dự kiến
            estimated_profit = 0.5  # Default 0.5%
            if opportunity_score > 90:
                estimated_profit = 1.5
            elif opportunity_score > 80:
                estimated_profit = 1.0
            elif opportunity_score > 70:
                estimated_profit = 0.7
            
            # Tính toán confidence level (độ tin cậy)
            confidence_level = min(opportunity_score, 100)
            
            # Xác định đòn bẩy dựa trên confidence level
            recommended_leverage = 3  # Default
            if confidence_level > 90:
                recommended_leverage = 6
            elif confidence_level > 80:
                recommended_leverage = 5
            elif confidence_level > 70:
                recommended_leverage = 4
            
            # Khuyến nghị
            if opportunity_score > state.opportunity_score_threshold_after_target:
                recommendation = "strong_entry"
                action = "Mở lệnh với đòn bẩy cao"
            elif opportunity_score > state.opportunity_score_threshold:
                recommendation = "entry"
                action = "Mở lệnh với đòn bẩy trung bình"
            else:
                recommendation = "monitor"
                action = "Theo dõi, chưa nên vào lệnh"
            
            # Tính toán stoploss đề xuất
            # Giả định 1% of capital risk per trade
            risk_per_trade = 0.01  # 1%
            price = dataframe['close'].iloc[-1]
            
            if side == "long":
                sl_pct = 3.0  # 3% stoploss
                if opportunity_score > 90:
                    sl_pct = 2.0  # Tín hiệu mạnh có thể dùng stoploss khít hơn
                
                sl_price = price * (1 - sl_pct/100)
                
            else:  # short
                sl_pct = 3.0  # 3% stoploss
                if opportunity_score > 90:
                    sl_pct = 2.0
                
                sl_price = price * (1 + sl_pct/100)
            
            # Tính lượng vốn để đầu tư
            capital_fraction = min(0.3, risk_per_trade / (sl_pct/100))  # Max 30% of capital
            
            return {
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "current_price": price,
                "recommendation": recommendation,
                "action": action,
                "opportunity_score": opportunity_score,
                "confidence_level": confidence_level,
                "estimated_profit": estimated_profit,
                "recommended_leverage": recommended_leverage,
                "stoploss_percentage": sl_pct,
                "stoploss_price": sl_price,
                "capital_allocation": capital_fraction,
                "daily_profit_status": daily_profit_status
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy khuyến nghị giao dịch: {str(e)}")
            logger.exception(e)
            return {
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "recommendation": "error",
                "error": str(e)
            }
    
    def get_exit_recommendation(self, dataframe, pair, current_time, entry_price, side="long", 
                              current_profit=0.0, timeframe="15m"):
        """
        Đưa ra khuyến nghị thoát lệnh
        
        Args:
            dataframe: Dữ liệu giá
            pair: Cặp tiền
            current_time: Thời gian hiện tại
            entry_price: Giá vào lệnh
            side: 'long' hoặc 'short'
            current_profit: Lợi nhuận hiện tại
            timeframe: Khung thời gian
            
        Returns:
            dict: Thông tin khuyến nghị thoát lệnh
        """
        try:
            # Cung cấp giá hiện tại để tính lợi nhuận nếu cần
            current_price = dataframe['close'].iloc[-1]
            
            # Nếu chưa cung cấp lợi nhuận, tính lợi nhuận
            if current_profit == 0.0:
                if side == "long":
                    current_profit = (current_price / entry_price - 1) * 100
                else:  # short
                    current_profit = (entry_price / current_price - 1) * 100
            
            # Mặc định không thoát
            exit_recommendation = "hold"
            reason = "Chưa đạt điều kiện thoát lệnh"
            
            # Kiểm tra các điều kiện thoát
            
            # 1. Nếu lợi nhuận đã đạt mục tiêu bảo vệ (3%)
            if current_profit >= 3.0:
                exit_recommendation = "exit_full"
                reason = f"Đã đạt mục tiêu lợi nhuận 3% (thực tế: {current_profit:.2f}%)"
            
            # 2. Thoát một phần nếu đã có lợi nhuận khá (2%)
            elif current_profit >= 2.0:
                exit_recommendation = "exit_partial"
                reason = f"Đã đạt lợi nhuận 2%, khuyến nghị chốt một phần (thực tế: {current_profit:.2f}%)"
            
            # 3. Kiểm tra chỉ báo kỹ thuật (ví dụ RSI)
            if side == "long":
                rsi = ta.RSI(dataframe['close'], timeperiod=14).iloc[-1]
                if rsi > 70 and current_profit > 0.5:
                    exit_recommendation = "exit_full"
                    reason = f"RSI overbought ({rsi:.2f}) khi có lợi nhuận {current_profit:.2f}%"
                
                # Kiểm tra xu hướng giảm
                ema8 = ta.EMA(dataframe['close'], timeperiod=8).iloc[-1]
                ema21 = ta.EMA(dataframe['close'], timeperiod=21).iloc[-1]
                
                if ema8 < ema21 and current_profit > 1.0:
                    exit_recommendation = "exit_full"
                    reason = f"Xu hướng đã đảo chiều (EMA8 < EMA21) khi có lợi nhuận {current_profit:.2f}%"
                
            else:  # short
                rsi = ta.RSI(dataframe['close'], timeperiod=14).iloc[-1]
                if rsi < 30 and current_profit > 0.5:
                    exit_recommendation = "exit_full"
                    reason = f"RSI oversold ({rsi:.2f}) khi có lợi nhuận {current_profit:.2f}%"
                
                # Kiểm tra xu hướng tăng
                ema8 = ta.EMA(dataframe['close'], timeperiod=8).iloc[-1]
                ema21 = ta.EMA(dataframe['close'], timeperiod=21).iloc[-1]
                
                if ema8 > ema21 and current_profit > 1.0:
                    exit_recommendation = "exit_full"
                    reason = f"Xu hướng đã đảo chiều (EMA8 > EMA21) khi có lợi nhuận {current_profit:.2f}%"
            
            # 4. Nếu có lợi nhuận và đã đạt mục tiêu 3% hàng ngày
            if current_profit > 0.5 and state.daily_profit_pct >= state.daily_profit_target_pct:
                exit_recommendation = "exit_full"
                reason = f"Đã đạt mục tiêu lợi nhuận hàng ngày {state.daily_profit_target_pct}%, chốt lời {current_profit:.2f}% để bảo vệ vốn"
            
            return {
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "entry_price": entry_price,
                "current_price": current_price,
                "current_profit": current_profit,
                "recommendation": exit_recommendation,
                "reason": reason
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy khuyến nghị thoát lệnh: {str(e)}")
            logger.exception(e)
            return {
                "pair": pair,
                "timeframe": timeframe,
                "side": side,
                "recommendation": "error",
                "error": str(e)
            }
    
    def simulate_trading_day(self, num_simulations=100):
        """
        Mô phỏng một ngày giao dịch để tính toán xác suất đạt mục tiêu 3%
        
        Args:
            num_simulations: Số lần mô phỏng
            
        Returns:
            dict: Kết quả mô phỏng
        """
        try:
            results = []
            success_count = 0
            target_profit = state.daily_profit_target_pct
            
            for i in range(num_simulations):
                daily_profit = 0
                trades = 0
                max_trades = 20  # Giới hạn số lượng giao dịch mỗi ngày
                
                while daily_profit < target_profit and trades < max_trades:
                    # Mô phỏng một trade
                    # Giả định phân phối lợi nhuận dạng chuẩn với mean=0.5%, std=1.5%
                    trade_profit = np.random.normal(0.5, 1.5)
                    
                    # Đặt giới hạn stoploss -3%
                    if trade_profit < -3:
                        trade_profit = -3
                    
                    daily_profit += trade_profit
                    trades += 1
                
                # Lưu kết quả
                success = daily_profit >= target_profit
                results.append({
                    "daily_profit": daily_profit,
                    "trades": trades,
                    "success": success
                })
                
                if success:
                    success_count += 1
            
            # Tính toán thống kê
            success_rate = success_count / num_simulations * 100
            avg_trades_needed = np.mean([r["trades"] for r in results if r["success"]])
            avg_profit = np.mean([r["daily_profit"] for r in results])
            
            return {
                "success_rate": success_rate,
                "avg_trades_needed": avg_trades_needed,
                "avg_profit": avg_profit,
                "target_profit": target_profit,
                "simulations": num_simulations
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi mô phỏng ngày giao dịch: {str(e)}")
            logger.exception(e)
            return {
                "error": str(e)
            }
    
    def track_completed_trade(self, trade_info):
        """
        Theo dõi thông tin về một giao dịch đã hoàn thành
        
        Args:
            trade_info: Thông tin giao dịch (dictionary)
        """
        try:
            # Cập nhật lợi nhuận hàng ngày
            profit_pct = trade_info.get('profit_pct', 0)
            state.update_daily_profit(profit_pct)
            
            # Thêm giao dịch vào danh sách
            state.add_trade(trade_info)
            
            # Log thông tin
            logger.info(f"Đã hoàn thành giao dịch {trade_info.get('pair')} với lợi nhuận {profit_pct}%")
            logger.info(f"Tổng lợi nhuận hôm nay: {state.daily_profit_pct}% ({state.num_trades_today} giao dịch)")
            
            # Xóa cặp tiền khỏi danh sách active nếu cần
            if trade_info.get('pair') in state.active_pairs:
                state.active_pairs.remove(trade_info.get('pair'))
            
        except Exception as e:
            logger.error(f"Lỗi khi theo dõi giao dịch hoàn thành: {str(e)}")
            logger.exception(e)
    
    def reset_daily_tracking(self):
        """Reset theo dõi hàng ngày khi bắt đầu ngày mới"""
        try:
            state.reset_daily_tracking()
            logger.info(f"Đã reset theo dõi hàng ngày. Mục tiêu lợi nhuận: {state.daily_profit_target_pct}%")
            
        except Exception as e:
            logger.error(f"Lỗi khi reset theo dõi hàng ngày: {str(e)}")
            logger.exception(e)
    
    def analyze_trade_history(self, trade_history=None):
        """
        Phân tích lịch sử giao dịch để cải thiện chiến lược
        
        Args:
            trade_history: Danh sách các giao dịch trong quá khứ
            
        Returns:
            dict: Kết quả phân tích
        """
        try:
            # Sử dụng dữ liệu mô phỏng nếu không có lịch sử
            if trade_history is None:
                # Giả lập 100 giao dịch
                np.random.seed(42)  # Để kết quả có thể tái tạo
                profit_results = np.random.normal(0.5, 1.5, 100)  # mean 0.5%, std 1.5%
                
                # Giới hạn stoploss
                profit_results = np.clip(profit_results, -3, None)
                
                # Tạo chi tiết giao dịch
                trade_history = []
                for i, profit in enumerate(profit_results):
                    # Random pair
                    pair = np.random.choice(["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "ADA/USDT"])
                    
                    # Random timeframe
                    timeframe = np.random.choice(["5m", "15m", "1h", "4h"])
                    
                    # Random side
                    side = np.random.choice(["long", "short"])
                    
                    trade_history.append({
                        "id": i,
                        "pair": pair,
                        "timeframe": timeframe,
                        "side": side,
                        "profit_pct": profit,
                        "open_date": datetime.now() - timedelta(days=np.random.randint(1, 30)),
                        "close_date": datetime.now() - timedelta(days=np.random.randint(0, 29))
                    })
            
            # Chuyển đổi thành DataFrame
            df = pd.DataFrame(trade_history)
            
            # Thống kê cơ bản
            total_trades = len(df)
            winning_trades = len(df[df['profit_pct'] > 0])
            losing_trades = len(df[df['profit_pct'] <= 0])
            win_rate = winning_trades / total_trades * 100
            
            avg_profit = df['profit_pct'].mean()
            avg_win = df[df['profit_pct'] > 0]['profit_pct'].mean()
            avg_loss = df[df['profit_pct'] <= 0]['profit_pct'].mean()
            
            # Profit factor
            profit_factor = abs(df[df['profit_pct'] > 0]['profit_pct'].sum() / df[df['profit_pct'] <= 0]['profit_pct'].sum()) if df[df['profit_pct'] <= 0]['profit_pct'].sum() != 0 else float('inf')
            
            # Phân tích theo cặp tiền
            pair_performance = df.groupby('pair')['profit_pct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            # Phân tích theo timeframe
            timeframe_performance = df.groupby('timeframe')['profit_pct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            # Phân tích theo side
            side_performance = df.groupby('side')['profit_pct'].agg(['mean', 'count']).sort_values('mean', ascending=False)
            
            # Tính số ngày giao dịch thành công
            if 'open_date' in df.columns:
                df['open_date'] = pd.to_datetime(df['open_date'])
                df['date'] = df['open_date'].dt.date
                
                daily_profits = df.groupby('date')['profit_pct'].sum()
                successful_days = len(daily_profits[daily_profits >= state.daily_profit_target_pct])
                total_days = len(daily_profits)
                daily_success_rate = successful_days / total_days * 100 if total_days > 0 else 0
            else:
                daily_success_rate = "N/A"
            
            # Đề xuất cải thiện
            recommendations = []
            
            # Đề xuất dựa trên win rate
            if win_rate < 50:
                recommendations.append("Cải thiện chất lượng tín hiệu vào lệnh để tăng win rate")
            
            # Đề xuất dựa trên profit factor
            if profit_factor < 1.5:
                recommendations.append("Cải thiện tỷ lệ R:R (Reward:Risk) bằng cách giảm stoploss hoặc tăng take profit")
            
            # Đề xuất dựa trên pair performance
            best_pairs = pair_performance.head(3).index.tolist()
            worst_pairs = pair_performance.tail(3).index.tolist()
            
            recommendations.append(f"Ưu tiên giao dịch các cặp hiệu quả nhất: {', '.join(best_pairs)}")
            recommendations.append(f"Hạn chế hoặc cải thiện chiến lược cho các cặp kém hiệu quả: {', '.join(worst_pairs)}")
            
            # Đề xuất dựa trên timeframe performance
            best_timeframes = timeframe_performance.head(2).index.tolist()
            recommendations.append(f"Tập trung vào các khung thời gian hiệu quả nhất: {', '.join(best_timeframes)}")
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_profit": avg_profit,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "pair_performance": pair_performance.to_dict(),
                "timeframe_performance": timeframe_performance.to_dict(),
                "side_performance": side_performance.to_dict(),
                "daily_target_success_rate": daily_success_rate,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích lịch sử giao dịch: {str(e)}")
            logger.exception(e)
            return {
                "error": str(e)
            }
    
    def predict_profit_potential(self, capital=1000, days=30):
        """
        Dự đoán tiềm năng lợi nhuận trong một khoảng thời gian dựa trên dữ liệu quá khứ
        
        Args:
            capital: Vốn ban đầu
            days: Số ngày dự đoán
            
        Returns:
            dict: Kết quả dự đoán
        """
        try:
            # Phân tích lịch sử giao dịch
            history_analysis = self.analyze_trade_history()
            
            # Mô phỏng ngày giao dịch
            day_simulation = self.simulate_trading_day(num_simulations=500)
            
            # Tỷ lệ ngày thành công
            day_success_rate = day_simulation['success_rate'] / 100
            
            # Tính toán compound
            daily_return = state.daily_profit_target_pct / 100
            compound_factor = (1 + daily_return) ** (day_success_rate * days)
            expected_capital = capital * compound_factor
            
            # Tính lợi nhuận dự kiến
            expected_profit = expected_capital - capital
            expected_profit_pct = (expected_profit / capital) * 100
            
            # Tạo bảng dự báo theo ngày
            daily_projection = []
            current_capital = capital
            
            for day in range(1, days + 1):
                # Xác suất thành công trong ngày
                success = np.random.random() < day_success_rate
                
                if success:
                    day_profit_pct = daily_return * 100
                    day_profit = current_capital * daily_return
                else:
                    # Ngày thất bại, giả định lợi nhuận 0%
                    day_profit_pct = 0
                    day_profit = 0
                
                current_capital += day_profit
                
                daily_projection.append({
                    "day": day,
                    "success": success,
                    "profit_pct": day_profit_pct,
                    "profit": day_profit,
                    "capital": current_capital
                })
            
            return {
                "initial_capital": capital,
                "prediction_days": days,
                "day_success_rate": day_success_rate * 100,
                "expected_capital": expected_capital,
                "expected_profit": expected_profit,
                "expected_profit_pct": expected_profit_pct,
                "daily_projection": daily_projection,
                "history_analysis": history_analysis,
                "day_simulation": day_simulation
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi dự đoán tiềm năng lợi nhuận: {str(e)}")
            logger.exception(e)
            return {
                "error": str(e)
            }
    
    def analyze_with_openai_gpt(self, trade_data, prompt_type="trade_analysis"):
        """
        Sử dụng OpenAI API để phân tích dữ liệu giao dịch và cung cấp insight
        
        Args:
            trade_data: Dữ liệu giao dịch cần phân tích
            prompt_type: Loại phân tích cần thực hiện
            
        Returns:
            dict: Kết quả phân tích từ GPT
        """
        try:
            # Kiểm tra API key
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return {
                    "status": "error",
                    "message": "Không tìm thấy OPENAI_API_KEY trong biến môi trường"
                }
            
            # Cấu hình OpenAI API
            openai.api_key = openai_api_key
            
            # Chuẩn bị prompt dựa trên loại phân tích
            prompt = ""
            
            if prompt_type == "trade_analysis":
                # Phân tích giao dịch
                prompt = f"""
                Phân tích dữ liệu giao dịch sau và cung cấp insight hữu ích:
                
                Dữ liệu giao dịch:
                {json.dumps(trade_data, indent=2)}
                
                Hãy phân tích:
                1. Tỷ lệ thắng/thua và lợi nhuận trung bình
                2. Cặp tiền nào hiệu quả nhất và kém nhất
                3. Khung thời gian nào có kết quả tốt nhất
                4. Đề xuất các lĩnh vực cần cải thiện
                5. Chiến lược để đạt mục tiêu lợi nhuận 3% hàng ngày một cách nhất quán
                
                Trả lời bằng tiếng Việt, định dạng rõ ràng dễ đọc.
                """
            
            elif prompt_type == "market_analysis":
                # Phân tích thị trường
                prompt = f"""
                Phân tích dữ liệu thị trường sau và cung cấp insight hữu ích:
                
                Dữ liệu thị trường:
                {json.dumps(trade_data, indent=2)}
                
                Hãy phân tích:
                1. Tình trạng thị trường hiện tại (trend, volatility)
                2. Các cặp tiền có tiềm năng nhất trong điều kiện hiện tại
                3. Rủi ro chính và cách giảm thiểu
                4. Chiến lược tối ưu cho tình hình hiện tại
                5. Dự báo ngắn hạn (24-48 giờ tới)
                
                Trả lời bằng tiếng Việt, định dạng rõ ràng dễ đọc.
                """
            
            elif prompt_type == "profit_optimization":
                # Tối ưu hóa lợi nhuận
                prompt = f"""
                Phân tích dữ liệu sau và đề xuất chiến lược tối ưu hóa để đạt mục tiêu lợi nhuận 3% hàng ngày:
                
                Dữ liệu:
                {json.dumps(trade_data, indent=2)}
                
                Hãy phân tích:
                1. Đánh giá tình trạng hiện tại so với mục tiêu 3%/ngày
                2. Số lượng giao dịch tối ưu và lợi nhuận mục tiêu mỗi giao dịch
                3. Phân bổ vốn và quản lý rủi ro
                4. Điều chỉnh đòn bẩy và position sizing
                5. Kế hoạch hành động cụ thể cho 24 giờ tới
                
                Trả lời bằng tiếng Việt, định dạng rõ ràng dễ đọc.
                """
            
            else:
                # Prompt mặc định
                prompt = f"""
                Phân tích dữ liệu sau và cung cấp insight:
                
                {json.dumps(trade_data, indent=2)}
                
                Trả lời bằng tiếng Việt, định dạng rõ ràng dễ đọc.
                """
            
            # Gọi OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4-0125-preview",  # Sử dụng model mới nhất
                messages=[
                    {"role": "system", "content": "Bạn là trợ lý phân tích giao dịch crypto chuyên nghiệp, giỏi phân tích dữ liệu và đưa ra insight sâu sắc."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.5
            )
            
            analysis = response.choices[0].message.content
            
            return {
                "status": "success",
                "analysis": analysis,
                "prompt_type": prompt_type
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích với OpenAI GPT: {str(e)}")
            logger.exception(e)
            return {
                "status": "error",
                "message": str(e)
            }

def main():
    """Hàm main để chạy chương trình"""
    try:
        logger.info("Khởi động AI Profit Optimizer...")
        
        # Khởi tạo optimizer
        optimizer = ProfitOptimizer()
        
        # Phân tích lợi nhuận cần thiết
        analysis = optimizer.analyze_daily_profit_needed()
        logger.info(f"Phân tích lợi nhuận: {json.dumps(analysis, indent=2)}")
        
        # Mô phỏng giao dịch
        simulation = optimizer.simulate_trading_day(num_simulations=1000)
        logger.info(f"Kết quả mô phỏng: {json.dumps(simulation, indent=2)}")
        
        logger.info("AI Profit Optimizer đã sẵn sàng.")
        
    except Exception as e:
        logger.error(f"Lỗi khi chạy AI Profit Optimizer: {str(e)}")
        logger.exception(e)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Module huấn luyện mô hình AI cho chiến lược giao dịch.
Sử dụng LightGBM để xây dựng mô hình dự đoán xu hướng, biến động và điểm đảo chiều thị trường.
"""

import os
import sys
import time
import ccxt
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("train_ai_models.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Định nghĩa các hàm chỉ báo kỹ thuật ---
def EMA(dataframe, timeperiod=20):
    """Exponential Moving Average"""
    return dataframe.ewm(span=timeperiod, min_periods=timeperiod).mean()

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

def collect_training_data(symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], 
                         timeframe='5m', lookback_days=30):
    """
    Thu thập dữ liệu từ Binance để huấn luyện mô hình
    """
    logger.info(f"Collecting data for {symbols} on {timeframe} timeframe (past {lookback_days} days)")
    
    # Kiểm tra nếu đã có file cache
    cache_file = f"training_data_{timeframe}_{lookback_days}days.csv"
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:  # 1 day
        logger.info(f"Using cached training data from {cache_file}")
        return pd.read_csv(cache_file)
    
    try:
        # Khởi tạo API Binance
        binance = ccxt.binance({
            'apiKey': os.environ.get('BINANCE_API_KEY', ''),
            'secret': os.environ.get('BINANCE_API_SECRET', ''),
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future',  # Sử dụng futures 
            }
        })
        
        # Tính toán thời gian bắt đầu và kết thúc
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Convert to milliseconds timestamp
        since = int(start_time.timestamp() * 1000)
        
        all_data = []
        
        # Thu thập dữ liệu cho từng symbol
        for symbol in symbols:
            logger.info(f"Fetching data for {symbol}...")
            
            try:
                # Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume)
                candles = binance.fetch_ohlcv(symbol, timeframe, since)
                
                # Chuyển đổi thành DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Thêm symbol vào DataFrame
                df['symbol'] = symbol
                
                # Chuyển đổi timestamp sang datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Thêm vào danh sách kết quả
                all_data.append(df)
                
                logger.info(f"Collected {len(df)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        # Kết hợp dữ liệu từ tất cả symbols
        if all_data:
            combined_df = pd.concat(all_data)
            combined_df.to_csv(cache_file, index=False)
            logger.info(f"Saved combined data to {cache_file}")
            return combined_df
        else:
            logger.warning("No data collected. Using simulated data.")
            # Trong trường hợp không có API Binance, tạo dữ liệu giả lập
            return generate_simulated_data()
            
    except Exception as e:
        logger.error(f"Error in data collection: {e}")
        logger.warning("Using simulated data instead.")
        return generate_simulated_data()

def generate_simulated_data():
    """
    Tạo dữ liệu giả lập để huấn luyện khi không thể kết nối với Binance API.
    """
    logger.warning("Using simulated data is not recommended for production. Please provide valid Binance API keys.")
    
    # Tạo một DataFrame giả lập
    # Lưu ý: Điều này không nên được sử dụng trong môi trường thực tế
    # Chỉ để phát triển và kiểm tra mã nguồn khi không có API keys thực
    data = []
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Không triển khai phần này để tránh tạo dữ liệu giả lập
    # Thay vào đó, tải dữ liệu mẫu từ file nếu có
    sample_file = "sample_ohlcv_data.csv"
    if os.path.exists(sample_file):
        logger.info(f"Loading sample data from {sample_file}")
        return pd.read_csv(sample_file)
    else:
        logger.error("No sample data available and cannot connect to API.")
        raise ValueError("Cannot generate training data. Please provide valid API keys or sample data.")

def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame
    """
    logger.info("Adding technical indicators...")
    
    # Đảm bảo rằng df được sắp xếp theo thời gian
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    
    # Xử lý từng symbol riêng biệt
    all_processed_data = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # 1. RSI
        symbol_df['rsi'] = RSI(symbol_df['close'], timeperiod=14)
        
        # 2. MACD
        macd = MACD(symbol_df)
        symbol_df['macd'] = macd['macd']
        symbol_df['macdsignal'] = macd['signal']
        symbol_df['macdhist'] = macd['histogram']
        
        # 3. Bollinger Bands
        bollinger = BBANDS(symbol_df)
        symbol_df['bb_upper'] = bollinger['upper']
        symbol_df['bb_mid'] = bollinger['mid']
        symbol_df['bb_lower'] = bollinger['lower']
        symbol_df['bb_width'] = (symbol_df['bb_upper'] - symbol_df['bb_lower']) / symbol_df['bb_mid']
        
        # 4. Average True Range (ATR)
        symbol_df['atr'] = ATR(symbol_df)
        
        # 5. EMA
        symbol_df['ema_9'] = EMA(symbol_df['close'], 9)
        symbol_df['ema_21'] = EMA(symbol_df['close'], 21)
        symbol_df['ema_50'] = EMA(symbol_df['close'], 50)
        symbol_df['ema_200'] = EMA(symbol_df['close'], 200)
        
        # 6. SMA
        symbol_df['sma_50'] = SMA(symbol_df['close'], 50)
        symbol_df['sma_200'] = SMA(symbol_df['close'], 200)
        
        # 7. Stochastic Oscillator
        stoch = STOCH(symbol_df)
        symbol_df['stoch_k'] = stoch['fastk']
        symbol_df['stoch_d'] = stoch['fastd']
        
        # 8. Money Flow Index
        symbol_df['mfi'] = MFI(symbol_df)
        
        # 9. Price distance from EMA
        symbol_df['distance_ema_200'] = ((symbol_df['close'] - symbol_df['ema_200']) / symbol_df['close']) * 100
        
        # 10. Volume Rate of Change
        symbol_df['volume_mean'] = symbol_df['volume'].rolling(window=20).mean()
        symbol_df['volume_roc'] = (symbol_df['volume'] / symbol_df['volume_mean'])
        
        # 11. Trend Direction (1=Uptrend, -1=Downtrend, 0=Sideways)
        symbol_df['trend'] = 0
        symbol_df.loc[(symbol_df['ema_50'] > symbol_df['sma_200']) & 
                     (symbol_df['close'] > symbol_df['ema_50']), 'trend'] = 1
        symbol_df.loc[(symbol_df['ema_50'] < symbol_df['sma_200']) & 
                     (symbol_df['close'] < symbol_df['ema_50']), 'trend'] = -1
        
        # 12. Volatility
        symbol_df['volatility'] = symbol_df['atr'] / symbol_df['close'] * 100
        
        # 13. Candle Patterns
        symbol_df['body'] = abs(symbol_df['open'] - symbol_df['close'])
        symbol_df['wick_high'] = symbol_df['high'] - symbol_df[['open', 'close']].max(axis=1)
        symbol_df['wick_low'] = symbol_df[['open', 'close']].min(axis=1) - symbol_df['low']
        
        # Thêm mẫu hình nến
        symbol_df['is_hammer'] = 0
        symbol_df.loc[(symbol_df['body'] > 0) &
                     (symbol_df['wick_low'] >= 2 * symbol_df['body']) &
                     (symbol_df['wick_high'] <= 0.2 * symbol_df['wick_low']), 'is_hammer'] = 1
        
        symbol_df['is_shooting_star'] = 0
        symbol_df.loc[(symbol_df['body'] > 0) &
                     (symbol_df['wick_high'] >= 2 * symbol_df['body']) &
                     (symbol_df['wick_low'] <= 0.2 * symbol_df['wick_high']), 'is_shooting_star'] = 1
        
        # Thêm biến đổi của giá
        for lag in [1, 3, 5, 10]:
            symbol_df[f'close_change_{lag}'] = symbol_df['close'].pct_change(lag)
            symbol_df[f'volume_change_{lag}'] = symbol_df['volume'].pct_change(lag)
            symbol_df[f'high_change_{lag}'] = symbol_df['high'].pct_change(lag)
            symbol_df[f'low_change_{lag}'] = symbol_df['low'].pct_change(lag)
        
        # Thêm biến đổi của các chỉ báo
        symbol_df['rsi_change_1'] = symbol_df['rsi'].diff(1)
        symbol_df['macd_change_1'] = symbol_df['macd'].diff(1)
        
        # Thêm vào kết quả
        all_processed_data.append(symbol_df)
    
    # Kết hợp tất cả dữ liệu đã xử lý
    processed_df = pd.concat(all_processed_data)
    
    # Xử lý các giá trị NaN và vô cùng
    processed_df = processed_df.replace([np.inf, -np.inf], np.nan)
    processed_df = processed_df.dropna()
    
    logger.info(f"Added technical indicators. Final shape: {processed_df.shape}")
    
    return processed_df

def create_features_labels(df):
    """
    Tạo tính năng và nhãn cho việc huấn luyện mô hình
    """
    logger.info("Creating features and labels...")
    
    # Xử lý từng symbol riêng biệt
    datasets = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        
        # 1. Tạo nhãn cho dự đoán xu hướng (1 = tăng, 0 = giảm)
        # Dùng sự thay đổi giá trong tương lai (sau 3 nến)
        symbol_df['future_change'] = symbol_df['close'].shift(-3) / symbol_df['close'] - 1
        symbol_df['trend_label'] = (symbol_df['future_change'] > 0).astype(int)
        
        # 2. Tạo nhãn cho dự đoán biến động (1 = biến động cao, 0 = biến động thấp)
        # Biến động cao được định nghĩa là khi ATR > 1.5 lần trung bình di động của ATR
        symbol_df['atr_ma'] = symbol_df['atr'].rolling(window=20).mean()
        symbol_df['volatility_label'] = (symbol_df['atr'] > 1.5 * symbol_df['atr_ma']).astype(int)
        
        # 3. Tạo nhãn cho dự đoán điểm đảo chiều
        # Điểm đảo chiều được định nghĩa là khi giá đảo chiều hơn 2% trong 2 nến tiếp theo
        symbol_df['max_future_high'] = symbol_df['high'].rolling(window=2, min_periods=1).max().shift(-2)
        symbol_df['min_future_low'] = symbol_df['low'].rolling(window=2, min_periods=1).min().shift(-2)
        
        # Điểm đảo chiều tăng: giá đang giảm và sắp đảo chiều tăng
        symbol_df['upward_reversal'] = ((symbol_df['close_change_5'] < -0.02) & 
                                      (symbol_df['max_future_high'] / symbol_df['close'] - 1 > 0.02)).astype(int)
        
        # Điểm đảo chiều giảm: giá đang tăng và sắp đảo chiều giảm
        symbol_df['downward_reversal'] = ((symbol_df['close_change_5'] > 0.02) & 
                                        (1 - symbol_df['min_future_low'] / symbol_df['close'] > 0.02)).astype(int)
        
        # Kết hợp thành nhãn đảo chiều
        symbol_df['reversal_label'] = (symbol_df['upward_reversal'] | symbol_df['downward_reversal']).astype(int)
        
        # Thêm vào danh sách kết quả
        datasets.append(symbol_df)
    
    # Kết hợp tất cả dữ liệu
    result_df = pd.concat(datasets)
    
    # Xử lý các giá trị NaN và vô cùng
    result_df = result_df.replace([np.inf, -np.inf], np.nan)
    result_df = result_df.dropna()
    
    logger.info(f"Created features and labels. Final shape: {result_df.shape}")
    
    return result_df

def train_models(df, model_dir='user_data/models'):
    """
    Huấn luyện các mô hình AI cho chiến lược giao dịch
    """
    logger.info("Starting model training...")
    
    # Tạo thư mục lưu mô hình nếu chưa tồn tại
    os.makedirs(model_dir, exist_ok=True)
    
    # Các tính năng dùng cho mô hình
    features = [
        'rsi', 'macd', 'macdsignal', 'macdhist', 'bb_width', 'atr', 'volatility',
        'ema_9', 'ema_21', 'ema_50', 'distance_ema_200', 'stoch_k', 'stoch_d', 'mfi',
        'trend', 'volume_roc', 'body', 'wick_high', 'wick_low', 'is_hammer', 'is_shooting_star',
        'close_change_1', 'close_change_3', 'close_change_5', 'volume_change_1',
        'high_change_1', 'low_change_1', 'rsi_change_1', 'macd_change_1'
    ]
    
    # 1. Huấn luyện mô hình dự đoán xu hướng
    logger.info("Training trend prediction model...")
    train_trend_model(df, features, model_dir)
    
    # 2. Huấn luyện mô hình dự đoán biến động
    logger.info("Training volatility prediction model...")
    train_volatility_model(df, features, model_dir)
    
    # 3. Huấn luyện mô hình dự đoán điểm đảo chiều
    logger.info("Training reversal prediction model...")
    train_reversal_model(df, features, model_dir)
    
    logger.info("Model training completed!")

def train_trend_model(df, features, model_dir):
    """Huấn luyện mô hình dự đoán xu hướng"""
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = df[features]
    y = df['trend_label']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lưu lại bộ chuẩn hóa
    joblib.dump(scaler, os.path.join(model_dir, 'trend_scaler.joblib'))
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Thiết lập tham số LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device': 'gpu',  # Sử dụng GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    # Tạo bộ dữ liệu LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Huấn luyện mô hình
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=500, 
        valid_sets=[test_data], 
        early_stopping_rounds=50,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # Đánh giá mô hình
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Trend model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Lưu mô hình
    model_path = os.path.join(model_dir, 'trend_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Trend model saved to {model_path}")
    
    # Tạo feature importance plot
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Trend Model - Feature Importance")
    plot_path = os.path.join(model_dir, 'trend_feature_importance.png')
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")

def train_volatility_model(df, features, model_dir):
    """Huấn luyện mô hình dự đoán biến động"""
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = df[features]
    y = df['volatility_label']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lưu lại bộ chuẩn hóa
    joblib.dump(scaler, os.path.join(model_dir, 'volatility_scaler.joblib'))
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Thiết lập tham số LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device': 'gpu',  # Sử dụng GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    # Tạo bộ dữ liệu LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Huấn luyện mô hình
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=500, 
        valid_sets=[test_data], 
        early_stopping_rounds=50,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # Đánh giá mô hình
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Volatility model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Lưu mô hình
    model_path = os.path.join(model_dir, 'volatility_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Volatility model saved to {model_path}")
    
    # Tạo feature importance plot
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Volatility Model - Feature Importance")
    plot_path = os.path.join(model_dir, 'volatility_feature_importance.png')
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")

def train_reversal_model(df, features, model_dir):
    """Huấn luyện mô hình dự đoán điểm đảo chiều"""
    
    # Chia dữ liệu thành tập huấn luyện và kiểm tra
    X = df[features]
    y = df['reversal_label']
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Lưu lại bộ chuẩn hóa
    joblib.dump(scaler, os.path.join(model_dir, 'reversal_scaler.joblib'))
    
    # Chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Thiết lập tham số LightGBM
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'device': 'gpu',  # Sử dụng GPU
        'gpu_platform_id': 0,
        'gpu_device_id': 0
    }
    
    # Tạo bộ dữ liệu LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Huấn luyện mô hình
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=500, 
        valid_sets=[test_data], 
        early_stopping_rounds=50,
        callbacks=[lgb.log_evaluation(100)]
    )
    
    # Đánh giá mô hình
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    logger.info(f"Reversal model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Lưu mô hình
    model_path = os.path.join(model_dir, 'reversal_model.joblib')
    joblib.dump(model, model_path)
    logger.info(f"Reversal model saved to {model_path}")
    
    # Tạo feature importance plot
    plt.figure(figsize=(10, 6))
    lgb.plot_importance(model, max_num_features=20)
    plt.title("Reversal Model - Feature Importance")
    plot_path = os.path.join(model_dir, 'reversal_feature_importance.png')
    plt.savefig(plot_path)
    logger.info(f"Feature importance plot saved to {plot_path}")

def main():
    """
    Hàm chính để chạy toàn bộ quá trình huấn luyện
    """
    logger.info("Starting AI model training process...")
    
    # 1. Thu thập dữ liệu
    df = collect_training_data(
        symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT'],
        timeframe='5m',
        lookback_days=30
    )
    
    # 2. Thêm các chỉ báo kỹ thuật
    df = add_technical_indicators(df)
    
    # 3. Tạo tính năng và nhãn
    df = create_features_labels(df)
    
    # 4. Huấn luyện mô hình
    train_models(df, model_dir='user_data/models')
    
    logger.info("Training process completed!")

if __name__ == "__main__":
    main()
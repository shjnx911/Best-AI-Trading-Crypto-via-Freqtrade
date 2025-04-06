import pandas as pd
import numpy as np
import os
import pickle
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import ccxt

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

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("user_data/logs/ai_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ai_training")

def collect_training_data(symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'], 
                         timeframe='5m', lookback_days=30):
    """
    Thu thập dữ liệu từ Binance để huấn luyện mô hình
    """
    logger.info(f"Bắt đầu thu thập dữ liệu cho {symbols} với khung thời gian {timeframe}")
    
    # Khởi tạo API Binance
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',
        }
    })
    
    all_data = []
    start_time = (datetime.now() - timedelta(days=lookback_days))
    since = exchange.parse8601(start_time.isoformat())
    
    for symbol in symbols:
        try:
            logger.info(f"Đang thu thập dữ liệu {symbol} từ {start_time.strftime('%Y-%m-%d')}")
            
            # Tải lịch sử giá
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not ohlcv or len(ohlcv) < 100:
                logger.warning(f"Không đủ dữ liệu cho {symbol}, bỏ qua")
                continue
                
            # Chuyển đổi sang DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['symbol'] = symbol
            
            # Lưu dữ liệu
            filename = f"user_data/data/{symbol.replace('/', '_')}_{timeframe}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Đã lưu {len(df)} nến cho {symbol} vào {filename}")
            
            all_data.append(df)
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu {symbol}: {str(e)}")
    
    if not all_data:
        logger.error("Không có dữ liệu nào được thu thập")
        return None
        
    return pd.concat(all_data)

def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame
    """
    logger.info("Thêm chỉ báo kỹ thuật vào dữ liệu")
    
    # Nhóm theo cặp tiền
    grouped = df.groupby('symbol')
    result_dfs = []
    
    for name, group in grouped:
        # Đặt lại index để sử dụng các hàm TA-Lib
        group = group.reset_index(drop=True)
        
        # Chỉ báo xác định xu hướng
        group['ema_short'] = EMA(group, timeperiod=8)
        group['ema_medium'] = EMA(group, timeperiod=21)
        group['ema_long'] = EMA(group, timeperiod=50)
        
        # Chỉ báo biến động
        group['atr'] = ATR(group, timeperiod=14)
        group['atr_pct'] = group['atr'] / group['close'] * 100
        
        # Chỉ báo xác định điểm đảo chiều
        group['rsi'] = RSI(group, timeperiod=14)
        group['rsi_ma'] = SMA(group['rsi'], timeperiod=5)
        group['mfi'] = MFI(group, timeperiod=14)
        
        # MACD
        macd = MACD(group)
        group['macd'] = macd['macd']
        group['macdsignal'] = macd['macdsignal']
        group['macdhist'] = macd['macdhist']
        
        # Volume profile
        group['volume_pct_change'] = group['volume'].pct_change() * 100
        
        # Liquidity indicators
        group['spread'] = (group['high'] - group['low']) / group['low'] * 100
        
        # Price action features
        group['body_size'] = abs(group['close'] - group['open']) / group['open'] * 100
        group['upper_wick'] = (group['high'] - group[['open', 'close']].max(axis=1)) / group['open'] * 100
        group['lower_wick'] = (group[['open', 'close']].min(axis=1) - group['low']) / group['open'] * 100
        
        # Bollinger Bands
        bollinger = BBANDS(group, timeperiod=20, nbdevup=2, nbdevdn=2)
        group['bb_upperband'] = bollinger['upperband']
        group['bb_middleband'] = bollinger['middleband']
        group['bb_lowerband'] = bollinger['lowerband']
        group['bb_width'] = (group['bb_upperband'] - group['bb_lowerband']) / group['bb_middleband']
        
        # Stochastic
        stoch = STOCH(group)
        group['slowk'] = stoch['slowk']
        group['slowd'] = stoch['slowd']
        
        # Thêm các feature dẫn xuất
        group['ema_diff'] = (group['ema_short'] - group['ema_medium']) / group['ema_medium'] * 100
        group['rsi_diff'] = group['rsi'] - group['rsi'].shift(1)
        
        result_dfs.append(group)
    
    return pd.concat(result_dfs)

def create_features_labels(df):
    """
    Tạo tính năng và nhãn cho việc huấn luyện mô hình
    """
    logger.info("Tạo các tính năng và nhãn cho mô hình")
    
    # Nhóm theo cặp tiền
    grouped = df.groupby('symbol')
    result_dfs = []
    
    for name, group in grouped:
        # Tạo nhãn cho các mô hình
        
        # Trend labels - lên/xuống trong 12 nến tiếp theo
        group['future_return'] = group['close'].pct_change(12).shift(-12)
        group['trend_label'] = (group['future_return'] > 0).astype(int)
        
        # Volatility labels - biến động cao/thấp
        group['future_volatility'] = group['close'].rolling(12).std().shift(-12) / group['close'] * 100
        # Dùng ngưỡng cụ thể thay vì median
        group['volatility_label'] = (group['future_volatility'] > 0.5).astype(int)
        
        # Reversal labels - đảo chiều trong 6 nến tiếp theo
        group['price_direction'] = np.sign(group['close'].diff())
        group['future_direction'] = group['price_direction'].shift(-6)
        group['reversal_label'] = (group['price_direction'] != group['future_direction']).astype(int)
        
        result_dfs.append(group)
    
    result = pd.concat(result_dfs)
    
    # Loại bỏ các hàng có giá trị NaN
    result = result.dropna()
    
    # Hiển thị phân phối của nhãn
    logger.info(f"Phân phối nhãn xu hướng: {result['trend_label'].value_counts(normalize=True)}")
    logger.info(f"Phân phối nhãn biến động: {result['volatility_label'].value_counts(normalize=True)}")
    logger.info(f"Phân phối nhãn đảo chiều: {result['reversal_label'].value_counts(normalize=True)}")
    
    return result

def train_models(df, model_dir='user_data/models'):
    """
    Huấn luyện các mô hình AI cho chiến lược giao dịch
    """
    logger.info("Bắt đầu huấn luyện mô hình")
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(model_dir, exist_ok=True)
    
    # Danh sách tính năng
    features = [
        'ema_short', 'ema_medium', 'ema_long', 'atr', 'atr_pct',
        'rsi', 'rsi_ma', 'mfi', 'macd', 'macdsignal', 'macdhist',
        'volume_pct_change', 'spread', 'body_size', 'upper_wick', 'lower_wick',
        'bb_width', 'slowk', 'slowd', 'ema_diff', 'rsi_diff'
    ]
    
    # Chọn dữ liệu
    X = df[features]
    y_trend = df['trend_label']
    y_volatility = df['volatility_label']
    y_reversal = df['reversal_label']
    
    # Chia dữ liệu
    X_train, X_test, y_trend_train, y_trend_test = train_test_split(
        X, y_trend, test_size=0.2, random_state=42)
    _, _, y_vol_train, y_vol_test = train_test_split(
        X, y_volatility, test_size=0.2, random_state=42)
    _, _, y_rev_train, y_rev_test = train_test_split(
        X, y_reversal, test_size=0.2, random_state=42)
    
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Thiết lập các tham số cho LGBM (Tăng tốc GPU nếu có)
    gpu_params = {}
    try:
        # Thử với ROCm GPU (AMD)
        gpu_params = {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
        logger.info("Đang thử sử dụng GPU cho huấn luyện")
    except:
        logger.info("Không có GPU khả dụng, sử dụng CPU")
    
    # Huấn luyện mô hình xu hướng (Trend)
    logger.info("Huấn luyện mô hình xu hướng")
    trend_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        **gpu_params
    )
    trend_model.fit(X_train_scaled, y_trend_train)
    
    # Huấn luyện mô hình biến động (Volatility)
    logger.info("Huấn luyện mô hình biến động")
    volatility_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        **gpu_params
    )
    volatility_model.fit(X_train_scaled, y_vol_train)
    
    # Huấn luyện mô hình đảo chiều (Reversal)
    logger.info("Huấn luyện mô hình đảo chiều")
    reversal_model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        **gpu_params
    )
    reversal_model.fit(X_train_scaled, y_rev_train)
    
    # Lưu mô hình
    pickle.dump(trend_model, open(os.path.join(model_dir, 'trend_model.pkl'), 'wb'))
    pickle.dump(volatility_model, open(os.path.join(model_dir, 'volatility_model.pkl'), 'wb'))
    pickle.dump(reversal_model, open(os.path.join(model_dir, 'reversal_model.pkl'), 'wb'))
    pickle.dump(scaler, open(os.path.join(model_dir, 'scaler.pkl'), 'wb'))
    
    # In độ chính xác
    trend_accuracy = trend_model.score(X_test_scaled, y_trend_test)
    volatility_accuracy = volatility_model.score(X_test_scaled, y_vol_test)
    reversal_accuracy = reversal_model.score(X_test_scaled, y_rev_test)
    
    logger.info(f"Độ chính xác mô hình xu hướng: {trend_accuracy:.4f}")
    logger.info(f"Độ chính xác mô hình biến động: {volatility_accuracy:.4f}")
    logger.info(f"Độ chính xác mô hình đảo chiều: {reversal_accuracy:.4f}")
    
    # Tạo biểu đồ tầm quan trọng của tính năng
    plt.figure(figsize=(12, 16))
    
    plt.subplot(3, 1, 1)
    plt.title('Tầm quan trọng của tính năng trong mô hình xu hướng')
    trend_importance = trend_model.feature_importances_
    indices = np.argsort(trend_importance)[::-1]
    plt.barh(range(len(indices)), trend_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    
    plt.subplot(3, 1, 2)
    plt.title('Tầm quan trọng của tính năng trong mô hình biến động')
    vol_importance = volatility_model.feature_importances_
    indices = np.argsort(vol_importance)[::-1]
    plt.barh(range(len(indices)), vol_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    
    plt.subplot(3, 1, 3)
    plt.title('Tầm quan trọng của tính năng trong mô hình đảo chiều')
    rev_importance = reversal_model.feature_importances_
    indices = np.argsort(rev_importance)[::-1]
    plt.barh(range(len(indices)), rev_importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
    
    return trend_accuracy, volatility_accuracy, reversal_accuracy

def main():
    """
    Hàm chính để chạy toàn bộ quá trình huấn luyện
    """
    # Danh sách cặp tiền muốn huấn luyện
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'LINK/USDT', 
              'OP/USDT', 'ARB/USDT', 'MATIC/USDT', 'ATOM/USDT', 'NEAR/USDT']
    
    # Thu thập dữ liệu
    data = collect_training_data(symbols=symbols, timeframe='5m', lookback_days=30)
    if data is None:
        logger.error("Không thu thập được dữ liệu, dừng quá trình huấn luyện")
        return
    
    # Thêm chỉ báo kỹ thuật
    data_with_indicators = add_technical_indicators(data)
    
    # Tạo tính năng và nhãn
    processed_data = create_features_labels(data_with_indicators)
    
    # Huấn luyện mô hình
    accuracies = train_models(processed_data)
    
    logger.info("Hoàn thành quá trình huấn luyện mô hình AI")
    logger.info(f"Kết quả: Xu hướng={accuracies[0]:.4f}, Biến động={accuracies[1]:.4f}, Đảo chiều={accuracies[2]:.4f}")

if __name__ == "__main__":
    main()
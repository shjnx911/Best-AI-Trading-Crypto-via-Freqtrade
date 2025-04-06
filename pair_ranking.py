#!/usr/bin/env python3
"""
Module xếp hạng các cặp tiền dựa trên tiềm năng lợi nhuận.
Phân tích và xếp hạng các cặp tiền từ Binance để xác định top N cặp tiền có tiềm năng nhất.
"""

import os
import sys
import ccxt
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Thêm đường dẫn cha vào sys.path để import các module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from market_sentiment import get_sentiment_signal

# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pair_ranking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def fetch_market_data(timeframe='1h', lookback_days=3, top_volume_pairs=20):
    """
    Lấy dữ liệu thị trường từ Binance và xác định các cặp tiền có khối lượng giao dịch cao nhất
    """
    logger.info(f"Fetching market data for top {top_volume_pairs} pairs by volume")
    
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
        
        # Lấy thông tin thị trường
        markets = binance.load_markets()
        
        # Lọc ra các cặp USDT trong futures
        usdt_futures = [symbol for symbol in markets.keys() if symbol.endswith(':USDT') and markets[symbol]['active']]
        
        # Lấy thông tin khối lượng giao dịch 24h
        tickers = binance.fetch_tickers(usdt_futures)
        
        # Sắp xếp theo khối lượng giao dịch
        volume_data = []
        for symbol, ticker in tickers.items():
            if 'quoteVolume' in ticker and ticker['quoteVolume'] is not None:
                volume_data.append({
                    'symbol': symbol,
                    'volume': ticker['quoteVolume']
                })
        
        # Sắp xếp và lấy top N cặp có khối lượng cao nhất
        volume_df = pd.DataFrame(volume_data)
        volume_df = volume_df.sort_values('volume', ascending=False).reset_index(drop=True)
        top_pairs = volume_df.head(top_volume_pairs)['symbol'].tolist()
        
        logger.info(f"Top {len(top_pairs)} volume pairs: {top_pairs}")
        
        # Tính toán thời gian bắt đầu và kết thúc
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback_days)
        
        # Convert to milliseconds timestamp
        since = int(start_time.timestamp() * 1000)
        
        # Thu thập dữ liệu cho top pairs
        pair_data = {}
        
        for symbol in top_pairs:
            logger.info(f"Fetching data for {symbol}...")
            
            try:
                # Lấy dữ liệu OHLCV (Open, High, Low, Close, Volume)
                candles = binance.fetch_ohlcv(symbol, timeframe, since)
                
                # Chuyển đổi thành DataFrame
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Chuyển đổi timestamp sang datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Thêm vào dictionary
                pair_data[symbol] = df
                
                logger.info(f"Collected {len(df)} candles for {symbol}")
                
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return pair_data
    
    except Exception as e:
        logger.error(f"Error in fetching market data: {e}")
        return {}

def calculate_metrics(pair_data):
    """
    Tính toán các chỉ số cho từng cặp tiền để xếp hạng
    """
    logger.info("Calculating metrics for pairs...")
    
    metrics = []
    
    for symbol, df in pair_data.items():
        try:
            # 1. Tính Volatility (ATR / Price)
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['volatility'] = df['atr'] / df['close'] * 100
            recent_volatility = df['volatility'].iloc[-1]
            
            # 2. Tính Volume Trend (Volume Change)
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            recent_volume_trend = df['volume_ratio'].iloc[-1]
            
            # 3. Tính Price Momentum
            df['price_change_1d'] = df['close'].pct_change(periods=24).iloc[-1] * 100  # 24 periods for 1h timeframe = 1 day
            df['price_change_3d'] = df['close'].pct_change(periods=72).iloc[-1] * 100  # 72 periods for 3 days
            recent_momentum = df['price_change_1d']
            
            # 4. Tính Trend Strength
            df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df['trend_strength'] = (df['close'] / df['ema_50'] - 1) * 100
            recent_trend = df['trend_strength'].iloc[-1]
            
            # 5. Lấy phân tích tâm lý thị trường
            base_symbol = symbol.split('/')[0] if '/' in symbol else symbol.split(':')[0]
            try:
                sentiment_score = get_sentiment_signal(base_symbol)
            except Exception as e:
                logger.warning(f"Could not get sentiment for {base_symbol}: {e}")
                sentiment_score = 0
            
            # Tạo điểm xếp hạng tổng hợp
            # Không ưu tiên volatility quá lớn, nhưng cũng không quá nhỏ
            volatility_score = 0
            if recent_volatility < 0.5:  # Quá ít biến động
                volatility_score = recent_volatility * 2
            elif recent_volatility > 3:  # Quá nhiều biến động
                volatility_score = 6 - recent_volatility
            else:  # Biến động vừa phải (tốt nhất)
                volatility_score = 10 - abs(2 - recent_volatility) * 2
            
            # Tính điểm tổng hợp
            combined_score = (
                volatility_score * 0.3 +  # 30% trọng số cho biến động phù hợp
                recent_volume_trend * 0.2 +  # 20% cho xu hướng khối lượng
                abs(recent_momentum) * 0.2 +  # 20% cho động lượng giá (bất kể lên hay xuống)
                abs(recent_trend) * 0.1 +  # 10% cho độ mạnh của xu hướng
                abs(sentiment_score) * 0.2  # 20% cho điểm tâm lý
            )
            
            # Thêm vào danh sách kết quả
            metrics.append({
                'symbol': symbol,
                'volatility': recent_volatility,
                'volume_trend': recent_volume_trend,
                'momentum': recent_momentum,
                'trend_strength': recent_trend,
                'sentiment': sentiment_score,
                'combined_score': combined_score
            })
            
            logger.info(f"Metrics for {symbol}: Score={combined_score:.2f}, Volatility={recent_volatility:.2f}%, Volume={recent_volume_trend:.2f}, Momentum={recent_momentum:.2f}%")
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {symbol}: {e}")
    
    # Tạo DataFrame từ danh sách kết quả
    metrics_df = pd.DataFrame(metrics)
    
    # Sắp xếp theo điểm tổng hợp
    metrics_df = metrics_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
    
    return metrics_df

def save_ranking_to_file(metrics_df, output_file='pair_ranking.json'):
    """
    Lưu bảng xếp hạng vào file
    """
    try:
        # Lưu vào JSON
        metrics_df.to_json(output_file, orient='records')
        logger.info(f"Saved pair ranking to {output_file}")
        
        # Hiển thị top 5
        print("\nTop 5 Pairs by Trading Potential:")
        print("=================================")
        for i, row in metrics_df.head(5).iterrows():
            print(f"{i+1}. {row['symbol']} - Score: {row['combined_score']:.2f}")
            print(f"   Volatility: {row['volatility']:.2f}%, Momentum: {row['momentum']:.2f}%, Sentiment: {row['sentiment']:.2f}")
        print("\n")
        
        return True
    except Exception as e:
        logger.error(f"Error saving ranking to file: {e}")
        return False

def generate_freqtrade_config(metrics_df, top_n=5, output_file='config/top_pairs_config.json'):
    """
    Tạo file cấu hình FreqTrade với top N cặp tiền
    """
    try:
        # Lấy top N cặp tiền
        top_pairs = metrics_df.head(top_n)['symbol'].tolist()
        
        # Đọc config mẫu
        with open('config/config.json', 'r') as f:
            config = json.load(f)
        
        # Cập nhật danh sách cặp tiền
        config['exchange']['pair_whitelist'] = top_pairs
        
        # Giới hạn số cặp giao dịch đồng thời
        config['max_open_trades'] = 3
        
        # Lưu config mới
        with open(output_file, 'w') as f:
            json.dump(config, f, indent=4)
        
        logger.info(f"Generated FreqTrade config with top {top_n} pairs: {top_pairs}")
        logger.info(f"Saved to {output_file}")
        
        return True
    except Exception as e:
        logger.error(f"Error generating FreqTrade config: {e}")
        return False

def rank_trading_pairs(top_n=5):
    """
    Hàm chính để xếp hạng các cặp tiền dựa trên tiềm năng giao dịch
    """
    logger.info(f"Starting to rank trading pairs to find top {top_n}...")
    
    # 1. Lấy dữ liệu thị trường
    pair_data = fetch_market_data(timeframe='1h', lookback_days=3, top_volume_pairs=20)
    
    if not pair_data:
        logger.error("Failed to fetch market data. Exiting.")
        return False
    
    # 2. Tính toán các chỉ số để xếp hạng
    metrics_df = calculate_metrics(pair_data)
    
    if len(metrics_df) == 0:
        logger.error("Failed to calculate metrics. Exiting.")
        return False
    
    # 3. Lưu bảng xếp hạng
    save_ranking_to_file(metrics_df)
    
    # 4. Tạo file cấu hình FreqTrade với top N cặp
    generate_freqtrade_config(metrics_df, top_n=top_n)
    
    return metrics_df.head(top_n)['symbol'].tolist()

if __name__ == "__main__":
    # Xếp hạng và lấy top 5 cặp tiền
    top_pairs = rank_trading_pairs(top_n=5)
    
    if top_pairs:
        print(f"Successfully ranked pairs and generated config with top 5: {top_pairs}")
    else:
        print("Failed to rank trading pairs.")
#!/usr/bin/env python3
"""
Script để chạy FreqTrade trong môi trường không cài đặt đầy đủ FreqTrade
"""
import argparse
import json
import logging
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("user_data/logs/freqtrade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("freqtrade")

# Path cho thư mục chiến lược
STRATEGY_PATH = Path("user_data/strategies")
CONFIG_PATH = Path("config")
DEFAULT_CONFIG = CONFIG_PATH / "config.json"
BACKTEST_CONFIG = CONFIG_PATH / "backtest_config.json"

def load_config(config_path):
    """Tải file cấu hình FreqTrade"""
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        logger.error(f"Lỗi khi tải cấu hình: {e}")
        sys.exit(1)

def load_strategy(strategy_name):
    """Tải chiến lược từ thư mục user_data/strategies"""
    try:
        sys.path.append(str(STRATEGY_PATH.parent))
        module_path = f"strategies.{strategy_name}"
        module = __import__(module_path, fromlist=[strategy_name])
        strategy_class = getattr(module, strategy_name)
        return strategy_class
    except Exception as e:
        logger.error(f"Lỗi khi tải chiến lược {strategy_name}: {e}")
        sys.exit(1)

def train_ai_models():
    """Huấn luyện mô hình AI cho chiến lược"""
    try:
        from train_ai_models import main as train_main
        logger.info("Bắt đầu huấn luyện mô hình AI...")
        train_main()
        logger.info("Hoàn thành huấn luyện mô hình AI")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình huấn luyện mô hình AI: {e}")
        sys.exit(1)

def backtest(config_path, strategy_name, timerange=None):
    """
    Thực hiện backtest với chiến lược đã chọn
    """
    from user_data.strategies.enhanced_crypto_strategy import EnhancedCryptoStrategy
    
    # Tải cấu hình
    config = load_config(config_path)
    if timerange:
        config['timerange'] = timerange
    
    logger.info(f"Thực hiện backtest với chiến lược {strategy_name}")
    logger.info(f"Khung thời gian: {config['timerange'] if 'timerange' in config else 'tất cả dữ liệu có sẵn'}")
    
    # Thực hiện mô phỏng đơn giản
    # Lưu ý: Đây là mô phỏng đơn giản, không phải backtest đầy đủ như FreqTrade
    
    try:
        # Giả lập một số cặp tiền và khung thời gian để kiểm tra logic chiến lược
        pairs = config['exchange']['pair_whitelist'][:3]  # Chỉ lấy 3 cặp đầu tiên để đơn giản
        
        results = {}
        for pair in pairs:
            # Đơn giản hóa tên cặp
            simple_pair = pair.replace("/USDT:USDT", "-USDT")
            
            # Nếu có dữ liệu thực từ CCXT
            try:
                import ccxt
                exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'future',
                    }
                })
                
                # Tải dữ liệu lịch sử
                timeframe = config['timeframe']
                since = exchange.parse8601(datetime.now() - timedelta(days=7))
                ohlcv = exchange.fetch_ohlcv(pair, timeframe, since)
                
                if ohlcv and len(ohlcv) > 0:
                    # Chuyển đổi sang DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')
                    
                    # Tạo một giá instance của chiến lược
                    strategy = EnhancedCryptoStrategy(config)
                    
                    # Thêm các chỉ báo
                    df = strategy.populate_indicators(df, {'pair': pair})
                    
                    # Thêm tín hiệu vào/thoát
                    if hasattr(strategy, 'populate_entry_trend') and hasattr(strategy, 'populate_exit_trend'):
                        df = strategy.populate_entry_trend(df, {'pair': pair})
                        df = strategy.populate_exit_trend(df, {'pair': pair})
                    else:
                        df = strategy.populate_buy_trend(df, {'pair': pair})
                        df = strategy.populate_sell_trend(df, {'pair': pair})
                    
                    # Tính toán lợi nhuận đơn giản (mô phỏng, không phải kết quả thực tế)
                    total_profit = 0
                    position_open = False
                    entry_price = 0
                    trades = []
                    
                    for i in range(len(df) - 1):
                        # Mua (long)
                        if not position_open and ('enter_long' in df.columns and df['enter_long'].iloc[i]) or ('buy' in df.columns and df['buy'].iloc[i]):
                            position_open = True
                            entry_price = df['close'].iloc[i]
                            trade = {
                                'entry_time': df.index[i],
                                'entry_price': entry_price,
                                'type': 'long'
                            }
                        
                        # Bán (thoát long)
                        elif position_open and ('exit_long' in df.columns and df['exit_long'].iloc[i]) or ('sell' in df.columns and df['sell'].iloc[i]):
                            exit_price = df['close'].iloc[i]
                            profit_pct = (exit_price - entry_price) / entry_price * 100
                            total_profit += profit_pct
                            position_open = False
                            
                            trade['exit_time'] = df.index[i]
                            trade['exit_price'] = exit_price
                            trade['profit_pct'] = profit_pct
                            trades.append(trade)
                    
                    results[simple_pair] = {
                        'total_profit_pct': total_profit,
                        'num_trades': len(trades),
                        'avg_profit_per_trade': total_profit / len(trades) if trades else 0,
                        'trades': trades
                    }
                    
                    logger.info(f"{simple_pair}: Lợi nhuận {total_profit:.2f}% trên {len(trades)} giao dịch")
                
            except Exception as e:
                logger.error(f"Lỗi khi thực hiện backtest cho {pair}: {e}")
                results[simple_pair] = {
                    'error': str(e)
                }
        
        # Hiển thị kết quả backtest
        logger.info("\n=== KẾT QUẢ BACKTEST ===")
        total_profit = sum(r['total_profit_pct'] for p, r in results.items() if 'total_profit_pct' in r)
        total_trades = sum(r['num_trades'] for p, r in results.items() if 'num_trades' in r)
        
        logger.info(f"Tổng lợi nhuận: {total_profit:.2f}%")
        logger.info(f"Tổng số giao dịch: {total_trades}")
        logger.info(f"Lợi nhuận trung bình/giao dịch: {total_profit/total_trades:.2f}% nếu total_trades > 0 else 'N/A'")
        
        # Vẽ biểu đồ kết quả
        plt.figure(figsize=(10, 6))
        profits = [r['total_profit_pct'] for p, r in results.items() if 'total_profit_pct' in r]
        pairs = [p for p, r in results.items() if 'total_profit_pct' in r]
        
        if profits and pairs:
            plt.bar(pairs, profits)
            plt.axhline(y=0, color='r', linestyle='-')
            plt.title('Lợi nhuận theo cặp tiền (%)')
            plt.ylabel('Lợi nhuận (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('user_data/backtest_results.png')
            logger.info("Đã lưu biểu đồ kết quả tại user_data/backtest_results.png")
        
        return results
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện backtest: {e}")
        return {'error': str(e)}

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Simple FreqTrade implementation')
    parser.add_argument('--train', action='store_true', help='Train AI models')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--strategy', type=str, default='EnhancedCryptoStrategy', help='Strategy name')
    parser.add_argument('--timerange', type=str, help='Timerange for backtest (format: YYYYMMDD-YYYYMMDD)')
    
    args = parser.parse_args()
    
    # Đảm bảo thư mục logs tồn tại
    os.makedirs("user_data/logs", exist_ok=True)
    
    if args.train:
        train_ai_models()
        
    if args.backtest:
        results = backtest(BACKTEST_CONFIG, args.strategy, args.timerange)
        logger.info(f"Kết quả backtest: {json.dumps(results, default=str, indent=2)}")
    
    if not args.train and not args.backtest:
        logger.error("Cần chỉ định ít nhất một trong các tùy chọn --train hoặc --backtest")
        parser.print_help()

if __name__ == "__main__":
    main()
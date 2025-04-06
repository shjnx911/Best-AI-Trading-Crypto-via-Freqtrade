#!/usr/bin/env python3
"""
Script để chạy FreqTrade trong môi trường không cài đặt đầy đủ FreqTrade
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from datetime import datetime
import re

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("run_freqtrade.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Tải file cấu hình FreqTrade"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Lỗi khi tải cấu hình từ {config_path}: {e}")
        return None

def load_strategy(strategy_name):
    """Tải chiến lược từ thư mục user_data/strategies"""
    try:
        # Kiểm tra xem strategy_name có phần mở rộng .py hay không
        if not strategy_name.endswith('.py'):
            strategy_file = f"user_data/strategies/{strategy_name}.py"
        else:
            strategy_file = f"user_data/strategies/{strategy_name}"
        
        if not os.path.exists(strategy_file):
            logger.error(f"File chiến lược {strategy_file} không tồn tại")
            return False
        
        # Hiển thị thông tin chiến lược
        with open(strategy_file, 'r') as f:
            content = f.read()
            # Tìm class name
            match = re.search(r'class\s+(\w+)\(IStrategy\)', content)
            if match:
                class_name = match.group(1)
                logger.info(f"Tìm thấy lớp chiến lược: {class_name}")
                return True
            else:
                logger.warning(f"Không tìm thấy lớp chiến lược trong {strategy_file}")
                return False
    except Exception as e:
        logger.error(f"Lỗi khi tải chiến lược {strategy_name}: {e}")
        return False

def train_ai_models():
    """Huấn luyện mô hình AI cho chiến lược"""
    logger.info("Bắt đầu huấn luyện mô hình AI...")
    try:
        import train_ai_models
        train_ai_models.main()
        logger.info("Huấn luyện mô hình AI thành công")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi huấn luyện mô hình AI: {e}")
        return False

def backtest(config_path, strategy_name, timerange=None):
    """
    Thực hiện backtest với chiến lược đã chọn
    """
    logger.info(f"Bắt đầu backtest với chiến lược {strategy_name}...")
    
    try:
        cmd = [
            "freqtrade", "backtesting",
            "--config", config_path,
            "--strategy", strategy_name
        ]
        
        if timerange:
            cmd.extend(["--timerange", timerange])
        
        # Thêm các tham số khác
        cmd.extend(["--timeframe-detail", "5m"])
        
        # Chạy lệnh
        logger.info(f"Chạy lệnh: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Hiển thị kết quả
        logger.info("Backtest hoàn thành")
        print(result.stdout)
        
        # Tạo báo cáo
        report_cmd = [
            "freqtrade", "plot-dataframe",
            "--config", config_path,
            "--strategy", strategy_name
        ]
        
        if timerange:
            report_cmd.extend(["--timerange", timerange])
        
        logger.info(f"Tạo báo cáo: {' '.join(report_cmd)}")
        subprocess.run(report_cmd, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy backtest: {e}")
        print(e.stderr)
        return False
    except Exception as e:
        logger.error(f"Lỗi không xác định: {e}")
        return False

def trade(config_path, strategy_name):
    """
    Chạy FreqTrade trong chế độ trading
    """
    logger.info(f"Bắt đầu trading với chiến lược {strategy_name}...")
    
    try:
        # Tạo thư mục logs nếu chưa tồn tại
        os.makedirs("logs", exist_ok=True)
        
        cmd = [
            "freqtrade", "trade",
            "--config", config_path,
            "--strategy", strategy_name,
            "--logfile", f"logs/freqtrade_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ]
        
        # Chạy lệnh
        logger.info(f"Chạy lệnh: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi chạy trading: {e}")
        return False
    except Exception as e:
        logger.error(f"Lỗi không xác định: {e}")
        return False

def check_freqtrade_installed():
    """Kiểm tra xem FreqTrade đã được cài đặt chưa"""
    try:
        result = subprocess.run(["freqtrade", "--version"], capture_output=True, text=True)
        logger.info(f"FreqTrade đã được cài đặt: {result.stdout.strip()}")
        return True
    except Exception:
        logger.error("FreqTrade chưa được cài đặt")
        print("FreqTrade chưa được cài đặt. Vui lòng cài đặt FreqTrade trước khi chạy script này.")
        print("Hướng dẫn cài đặt: https://www.freqtrade.io/en/stable/installation/")
        return False

def check_environment():
    """Kiểm tra môi trường và cài đặt các phụ thuộc cần thiết"""
    # Kiểm tra FreqTrade
    if not check_freqtrade_installed():
        return False
    
    # Kiểm tra thư mục user_data
    os.makedirs("user_data/strategies", exist_ok=True)
    os.makedirs("user_data/models", exist_ok=True)
    os.makedirs("user_data/data", exist_ok=True)
    os.makedirs("user_data/backtest_results", exist_ok=True)
    
    return True

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Chạy FreqTrade với các tùy chọn')
    parser.add_argument('--mode', choices=['backtest', 'trade', 'train'], required=True,
                        help='Chế độ chạy: backtest, trade, hoặc train')
    parser.add_argument('--config', default='config/config.json',
                        help='Đường dẫn đến file cấu hình (mặc định: config/config.json)')
    parser.add_argument('--strategy', default='EnhancedCryptoStrategy',
                        help='Tên chiến lược (mặc định: EnhancedCryptoStrategy)')
    parser.add_argument('--timerange', help='Khoảng thời gian cho backtest (ví dụ: 20210101-20210201)')
    
    args = parser.parse_args()
    
    # Kiểm tra môi trường
    if not check_environment():
        sys.exit(1)
    
    # Chọn chế độ chạy
    if args.mode == 'train':
        success = train_ai_models()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'backtest':
        # Nếu mode là backtest và không có config cụ thể, sử dụng backtest_config.json
        if args.config == 'config/config.json' and os.path.exists('config/backtest_config.json'):
            args.config = 'config/backtest_config.json'
            logger.info(f"Sử dụng cấu hình backtest mặc định: {args.config}")
        
        # Tải cấu hình
        config = load_config(args.config)
        if not config:
            sys.exit(1)
            
        # Tải chiến lược
        if not load_strategy(args.strategy):
            sys.exit(1)
            
        # Chạy backtest
        success = backtest(args.config, args.strategy, args.timerange)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'trade':
        # Tải cấu hình
        config = load_config(args.config)
        if not config:
            sys.exit(1)
            
        # Tải chiến lược
        if not load_strategy(args.strategy):
            sys.exit(1)
            
        # Chạy trading
        success = trade(args.config, args.strategy)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
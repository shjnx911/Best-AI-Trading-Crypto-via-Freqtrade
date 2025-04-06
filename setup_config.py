#!/usr/bin/env python3
"""
Script để thiết lập và tùy chỉnh cấu hình cho FreqTrade Bot.
Cung cấp giao diện dòng lệnh để điều chỉnh các tham số quan trọng mà không cần sửa file cấu hình trực tiếp.
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("setup_config.log"),
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

def save_config(config, config_path):
    """Lưu cấu hình vào file"""
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        logger.info(f"Đã lưu cấu hình vào {config_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi lưu cấu hình vào {config_path}: {e}")
        return False

def update_max_open_trades(config, value):
    """Cập nhật số lượng giao dịch đồng thời tối đa"""
    try:
        config['max_open_trades'] = value
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật max_open_trades: {e}")
        return False

def update_stake_amount(config, value):
    """Cập nhật số tiền đặt cược cho mỗi giao dịch"""
    try:
        if value.lower() == "unlimited":
            config['stake_amount'] = "unlimited"
        else:
            config['stake_amount'] = float(value)
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật stake_amount: {e}")
        return False

def update_stoploss(config, value):
    """Cập nhật giá trị stoploss trong chiến lược"""
    try:
        # Stoploss trong config.json
        if 'stoploss' in config:
            config['stoploss'] = -abs(float(value)) / 100
        
        # Tìm và cập nhật file chiến lược
        strategy_path = os.path.join(config.get('strategy_path', 'user_data/strategies'), f"{config.get('strategy', 'EnhancedCryptoStrategy')}.py")
        
        if os.path.exists(strategy_path):
            with open(strategy_path, 'r') as f:
                content = f.read()
            
            # Tìm và thay thế dòng stoploss
            import re
            pattern = r'stoploss\s*=\s*-[\d.]+'
            replacement = f"stoploss = -{value/100}"
            
            new_content = re.sub(pattern, replacement, content)
            
            with open(strategy_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Đã cập nhật stoploss trong file chiến lược {strategy_path}")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật stoploss: {e}")
        return False

def update_max_leverage(config, value):
    """Cập nhật đòn bẩy tối đa trong chiến lược"""
    try:
        # Tìm và cập nhật file chiến lược
        strategy_path = os.path.join(config.get('strategy_path', 'user_data/strategies'), f"{config.get('strategy', 'EnhancedCryptoStrategy')}.py")
        
        if os.path.exists(strategy_path):
            with open(strategy_path, 'r') as f:
                content = f.read()
            
            # Tìm và thay thế dòng leverage_max
            import re
            pattern = r'leverage_max\s*=\s*[\d.]+'
            replacement = f"leverage_max = {value}"
            
            new_content = re.sub(pattern, replacement, content)
            
            # Tìm và thay thế dòng leverage_default nếu cần
            pattern = r'leverage_default\s*=\s*[\d.]+'
            default_leverage = min(int(value), 3)  # Mặc định là tối đa 3x hoặc giá trị đòn bẩy tối đa
            replacement = f"leverage_default = {default_leverage}"
            
            new_content = re.sub(pattern, replacement, new_content)
            
            with open(strategy_path, 'w') as f:
                f.write(new_content)
                
            logger.info(f"Đã cập nhật đòn bẩy trong file chiến lược {strategy_path}")
        
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật đòn bẩy: {e}")
        return False

def update_pair_list(config, pairs):
    """Cập nhật danh sách cặp tiền trong config"""
    try:
        # Đảm bảo pairs là danh sách
        if isinstance(pairs, str):
            pair_list = [p.strip() for p in pairs.split(',')]
        else:
            pair_list = pairs
        
        # Thêm :USDT nếu cần
        for i, pair in enumerate(pair_list):
            if not pair.endswith(':USDT') and not pair.endswith('/USDT'):
                if '/' not in pair and ':' not in pair:
                    pair_list[i] = f"{pair}/USDT:USDT"
        
        config['exchange']['pair_whitelist'] = pair_list
        logger.info(f"Đã cập nhật danh sách cặp tiền: {pair_list}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật danh sách cặp tiền: {e}")
        return False

def update_api_keys(config, api_key, api_secret):
    """Cập nhật API keys cho sàn giao dịch"""
    try:
        config['exchange']['key'] = api_key
        config['exchange']['secret'] = api_secret
        logger.info("Đã cập nhật API keys")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi cập nhật API keys: {e}")
        return False

def show_current_config(config_path):
    """Hiển thị cấu hình hiện tại"""
    config = load_config(config_path)
    if not config:
        return
    
    print("\nCấu hình hiện tại:")
    print(f"- Số giao dịch đồng thời tối đa: {config.get('max_open_trades', 'N/A')}")
    print(f"- Số tiền đặt cược: {config.get('stake_amount', 'N/A')}")
    print(f"- Stoploss: {config.get('stoploss', 'N/A')}")
    print(f"- Danh sách cặp tiền: {config.get('exchange', {}).get('pair_whitelist', [])}")
    print(f"- API Key: {'Đã cấu hình' if config.get('exchange', {}).get('key') else 'Chưa cấu hình'}")
    print(f"- API Secret: {'Đã cấu hình' if config.get('exchange', {}).get('secret') else 'Chưa cấu hình'}")
    
    # Lấy thông tin đòn bẩy từ file chiến lược
    strategy_path = os.path.join(config.get('strategy_path', 'user_data/strategies'), f"{config.get('strategy', 'EnhancedCryptoStrategy')}.py")
    if os.path.exists(strategy_path):
        try:
            with open(strategy_path, 'r') as f:
                content = f.read()
            
            import re
            leverage_max_match = re.search(r'leverage_max\s*=\s*([\d.]+)', content)
            leverage_default_match = re.search(r'leverage_default\s*=\s*([\d.]+)', content)
            
            leverage_max = leverage_max_match.group(1) if leverage_max_match else "N/A"
            leverage_default = leverage_default_match.group(1) if leverage_default_match else "N/A"
            
            print(f"- Đòn bẩy mặc định: {leverage_default}")
            print(f"- Đòn bẩy tối đa: {leverage_max}")
        except Exception as e:
            logger.error(f"Lỗi khi đọc thông tin đòn bẩy: {e}")

def setup_wizard(config_path):
    """Trình wizard hướng dẫn thiết lập cấu hình"""
    config = load_config(config_path)
    if not config:
        print("Không thể tải cấu hình. Thoát wizard.")
        return
    
    print("\n=== Trình hướng dẫn thiết lập FreqTrade Bot ===\n")
    
    # 1. Số giao dịch đồng thời
    max_trades = input(f"Số giao dịch đồng thời tối đa [{config.get('max_open_trades', 3)}]: ")
    if max_trades.strip():
        update_max_open_trades(config, int(max_trades))
    
    # 2. Số tiền đặt cược
    stake_amount = input(f"Số tiền đặt cược (USDT) cho mỗi giao dịch [{config.get('stake_amount', 'unlimited')}]: ")
    if stake_amount.strip():
        update_stake_amount(config, stake_amount)
    
    # 3. Stoploss
    stoploss_percent = input("Cắt lỗ (%) [2]: ")
    if stoploss_percent.strip():
        update_stoploss(config, float(stoploss_percent))
    
    # 4. Đòn bẩy tối đa
    leverage = input("Đòn bẩy tối đa [5]: ")
    if leverage.strip():
        update_max_leverage(config, int(leverage))
    
    # 5. API Keys
    use_api = input("Bạn có muốn cấu hình API keys không? (y/n) [n]: ")
    if use_api.lower() == 'y':
        api_key = input("Binance API Key: ")
        api_secret = input("Binance API Secret: ")
        if api_key and api_secret:
            update_api_keys(config, api_key, api_secret)
    
    # 6. Cặp tiền
    use_top_pairs = input("Bạn có muốn sử dụng danh sách Top 5 cặp tiền có tiềm năng nhất? (y/n) [y]: ")
    if use_top_pairs.lower() != 'n':
        print("Sẽ sử dụng module pair_ranking.py để xác định Top 5 cặp tiền có tiềm năng nhất.")
        config['use_top_pairs'] = True
    else:
        pairs = input("Nhập danh sách cặp tiền (phân cách bằng dấu phẩy): ")
        if pairs.strip():
            update_pair_list(config, pairs)
    
    # Lưu cấu hình
    save_config(config, config_path)
    print("\nĐã lưu cấu hình thành công!")

def main():
    """Hàm main để chạy script"""
    parser = argparse.ArgumentParser(description='Thiết lập cấu hình cho FreqTrade Bot')
    parser.add_argument('--config', default='config/config.json', help='Đường dẫn đến file cấu hình')
    parser.add_argument('--max-trades', type=int, help='Số giao dịch đồng thời tối đa')
    parser.add_argument('--stake-amount', help='Số tiền đặt cược cho mỗi giao dịch')
    parser.add_argument('--stoploss', type=float, help='Phần trăm cắt lỗ')
    parser.add_argument('--leverage', type=int, help='Đòn bẩy tối đa')
    parser.add_argument('--pairs', help='Danh sách cặp tiền (phân cách bằng dấu phẩy)')
    parser.add_argument('--api-key', help='Binance API key')
    parser.add_argument('--api-secret', help='Binance API secret')
    parser.add_argument('--show', action='store_true', help='Hiển thị cấu hình hiện tại')
    parser.add_argument('--wizard', action='store_true', help='Chạy trình hướng dẫn thiết lập')
    
    args = parser.parse_args()
    
    # Kiểm tra xem file cấu hình tồn tại không
    if not os.path.exists(args.config):
        logger.error(f"File cấu hình {args.config} không tồn tại.")
        return 1
    
    # Hiển thị cấu hình hiện tại
    if args.show:
        show_current_config(args.config)
        return 0
    
    # Chạy trình wizard
    if args.wizard:
        setup_wizard(args.config)
        return 0
    
    # Cập nhật các tham số riêng lẻ
    config = load_config(args.config)
    if not config:
        return 1
    
    changes_made = False
    
    if args.max_trades is not None:
        changes_made |= update_max_open_trades(config, args.max_trades)
    
    if args.stake_amount is not None:
        changes_made |= update_stake_amount(config, args.stake_amount)
    
    if args.stoploss is not None:
        changes_made |= update_stoploss(config, args.stoploss)
    
    if args.leverage is not None:
        changes_made |= update_max_leverage(config, args.leverage)
    
    if args.pairs is not None:
        changes_made |= update_pair_list(config, args.pairs)
    
    if args.api_key is not None and args.api_secret is not None:
        changes_made |= update_api_keys(config, args.api_key, args.api_secret)
    
    # Lưu cấu hình nếu có thay đổi
    if changes_made:
        save_config(config, args.config)
        print("Đã cập nhật cấu hình thành công!")
    else:
        print("Không có thay đổi nào được thực hiện.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
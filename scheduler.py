#!/usr/bin/env python3
"""
Script quản lý việc lập lịch các tác vụ tự động cho FreqTrade Bot
Sử dụng crontab để thiết lập các tác vụ định kỳ
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import logging

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_crontab():
    """
    Thiết lập crontab để lập lịch các tác vụ tự động
    """
    logger.info("Thiết lập crontab cho FreqTrade Bot...")
    
    # Tạo file crontab tạm thời
    crontab_path = os.path.join(BASE_DIR, "freqtrade_crontab")
    
    with open(crontab_path, "w") as f:
        # Thêm comment để nhận biết
        f.write("# FreqTrade Bot Scheduler\n")
        
        # 1. Khởi động bot khi hệ thống khởi động lại
        f.write("@reboot cd {} && python run_freqtrade.py --mode trade\n".format(BASE_DIR))
        
        # 2. Huấn luyện lại mô hình AI hàng tuần (Chủ nhật lúc 1 giờ sáng)
        f.write("0 1 * * 0 cd {} && python train_ai_models.py\n".format(BASE_DIR))
        
        # 3. Sao lưu dữ liệu giao dịch hàng ngày (lúc 0 giờ)
        backup_dir = os.path.join(BASE_DIR, "backups")
        f.write("0 0 * * * mkdir -p {0} && cp -r {1}/user_data/models {0}/models_$(date +%Y%m%d) && cd {1} && python -c \"import json; import datetime; open('{0}/trades_$(date +%Y%m%d).json', 'w').write(json.dumps(dict(date=str(datetime.datetime.now()), data=dict())))\"\n".format(backup_dir, BASE_DIR))
        
        # 4. Kiểm tra trạng thái bot hàng giờ
        f.write("0 * * * * cd {} && python scheduler.py --check-status\n".format(BASE_DIR))
    
    # Cài đặt crontab
    try:
        subprocess.run(["crontab", crontab_path], check=True)
        logger.info("Đã thiết lập crontab thành công!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Lỗi khi thiết lập crontab: {e}")
    
    # Xóa file tạm
    os.remove(crontab_path)

def check_bot_status():
    """
    Kiểm tra trạng thái của FreqTrade Bot
    """
    logger.info("Kiểm tra trạng thái của FreqTrade Bot...")
    
    # Kiểm tra xem bot có đang chạy không
    try:
        result = subprocess.run(["pgrep", "-f", "freqtrade"], stdout=subprocess.PIPE)
        if result.stdout:
            logger.info("FreqTrade Bot đang chạy.")
            return True
        else:
            logger.warning("FreqTrade Bot không hoạt động! Đang khởi động lại...")
            # Khởi động lại bot
            subprocess.Popen(["python", os.path.join(BASE_DIR, "run_freqtrade.py"), "--mode", "trade"], 
                             start_new_session=True)
            return False
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra trạng thái bot: {e}")
        return False

def backup_data():
    """
    Sao lưu dữ liệu giao dịch và mô hình AI
    """
    logger.info("Đang sao lưu dữ liệu...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(BASE_DIR, "backups", timestamp)
    
    # Tạo thư mục sao lưu
    os.makedirs(backup_dir, exist_ok=True)
    
    # Sao lưu các mô hình AI
    models_dir = os.path.join(BASE_DIR, "user_data", "models")
    if os.path.exists(models_dir):
        try:
            subprocess.run(["cp", "-r", models_dir, os.path.join(backup_dir, "models")])
            logger.info(f"Đã sao lưu mô hình AI vào {backup_dir}/models")
        except Exception as e:
            logger.error(f"Lỗi khi sao lưu mô hình AI: {e}")
    
    # Sao lưu cấu hình
    config_dir = os.path.join(BASE_DIR, "config")
    if os.path.exists(config_dir):
        try:
            subprocess.run(["cp", "-r", config_dir, os.path.join(backup_dir, "config")])
            logger.info(f"Đã sao lưu cấu hình vào {backup_dir}/config")
        except Exception as e:
            logger.error(f"Lỗi khi sao lưu cấu hình: {e}")

def restart_bot():
    """
    Khởi động lại FreqTrade Bot
    """
    logger.info("Đang khởi động lại FreqTrade Bot...")
    
    # Tìm và kết thúc các tiến trình FreqTrade hiện tại
    try:
        subprocess.run(["pkill", "-f", "freqtrade"])
        logger.info("Đã kết thúc các tiến trình FreqTrade hiện tại.")
    except Exception as e:
        logger.warning(f"Không tìm thấy tiến trình FreqTrade để kết thúc: {e}")
    
    # Khởi động lại bot
    try:
        subprocess.Popen(["python", os.path.join(BASE_DIR, "run_freqtrade.py"), "--mode", "trade"], 
                         start_new_session=True)
        logger.info("Đã khởi động lại FreqTrade Bot.")
    except Exception as e:
        logger.error(f"Lỗi khi khởi động lại bot: {e}")

def main():
    """
    Xử lý tham số dòng lệnh và thực hiện tác vụ tương ứng
    """
    parser = argparse.ArgumentParser(description='FreqTrade Bot Scheduler')
    parser.add_argument('--setup', action='store_true', help='Thiết lập crontab')
    parser.add_argument('--check-status', action='store_true', help='Kiểm tra trạng thái bot')
    parser.add_argument('--backup', action='store_true', help='Sao lưu dữ liệu')
    parser.add_argument('--restart', action='store_true', help='Khởi động lại bot')
    
    args = parser.parse_args()
    
    if args.setup:
        setup_crontab()
    elif args.check_status:
        check_bot_status()
    elif args.backup:
        backup_data()
    elif args.restart:
        restart_bot()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Script hệ thống giám sát (monitoring) cho FreqTrade Bot
- Giám sát hiệu suất và trạng thái của bot
- Gửi thông báo qua Telegram hoặc email khi có vấn đề
- Cung cấp báo cáo định kỳ về hiệu suất giao dịch
"""

import os
import sys
import json
import time
import logging
import smtplib
import requests
import argparse
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitoring.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Đường dẫn đến thư mục dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cấu hình thông báo
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_RECEIVER = os.environ.get("EMAIL_RECEIVER", "")

def send_telegram_message(message):
    """
    Gửi thông báo qua Telegram
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Không có thông tin Telegram để gửi thông báo")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        response = requests.post(url, data=payload)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Lỗi khi gửi thông báo Telegram: {e}")
        return False

def send_email_notification(subject, message):
    """
    Gửi thông báo qua email
    """
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_RECEIVER:
        logger.warning("Không có thông tin email để gửi thông báo")
        return False
    
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECEIVER
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        logger.error(f"Lỗi khi gửi email: {e}")
        return False

def check_bot_health():
    """
    Kiểm tra sức khỏe của FreqTrade Bot
    """
    logger.info("Kiểm tra sức khỏe của FreqTrade Bot...")
    
    issues = []
    
    # 1. Kiểm tra xem bot có đang chạy không
    try:
        result = os.popen("pgrep -f freqtrade").read().strip()
        if not result:
            issues.append("Bot không hoạt động!")
    except Exception as e:
        issues.append(f"Lỗi khi kiểm tra trạng thái bot: {e}")
    
    # 2. Kiểm tra kết nối với sàn giao dịch
    # Giả định rằng chúng ta có một endpoint để kiểm tra kết nối
    # Đây chỉ là mô phỏng, cần điều chỉnh theo triển khai thực tế
    try:
        # Đọc trạng thái từ file log gần nhất của FreqTrade
        log_files = [f for f in os.listdir(os.path.join(BASE_DIR, "logs")) if f.endswith(".log")]
        if log_files:
            latest_log = max(log_files, key=lambda x: os.path.getmtime(os.path.join(BASE_DIR, "logs", x)))
            with open(os.path.join(BASE_DIR, "logs", latest_log), "r") as f:
                log_content = f.read()
                if "Error connecting to exchange" in log_content:
                    issues.append("Lỗi kết nối với sàn giao dịch!")
    except Exception as e:
        issues.append(f"Lỗi khi kiểm tra kết nối với sàn giao dịch: {e}")
    
    # 3. Kiểm tra dung lượng đĩa
    try:
        disk_usage = os.popen("df -h / | tail -1").read().strip().split()
        disk_percent = int(disk_usage[4].replace('%', ''))
        if disk_percent > 90:
            issues.append(f"Cảnh báo: Dung lượng đĩa đã sử dụng {disk_percent}%!")
    except Exception as e:
        issues.append(f"Lỗi khi kiểm tra dung lượng đĩa: {e}")
    
    # 4. Kiểm tra RAM
    try:
        mem_info = os.popen("free -m | grep Mem").read().strip().split()
        total_mem = int(mem_info[1])
        used_mem = int(mem_info[2])
        mem_percent = (used_mem / total_mem) * 100
        if mem_percent > 90:
            issues.append(f"Cảnh báo: Sử dụng RAM {mem_percent:.1f}%!")
    except Exception as e:
        issues.append(f"Lỗi khi kiểm tra RAM: {e}")
    
    return issues

def generate_performance_report():
    """
    Tạo báo cáo hiệu suất giao dịch
    """
    logger.info("Tạo báo cáo hiệu suất giao dịch...")
    
    # Giả định rằng chúng ta có cách truy cập dữ liệu giao dịch
    # Đây chỉ là mô phỏng, cần điều chỉnh theo triển khai thực tế
    
    try:
        # Đọc dữ liệu giao dịch từ FreqTrade
        trades_file = os.path.join(BASE_DIR, "user_data", "trades", "trades.json")
        if not os.path.exists(trades_file):
            return "Không tìm thấy dữ liệu giao dịch!"
        
        with open(trades_file, "r") as f:
            trades_data = json.load(f)
        
        # Lọc giao dịch trong 24 giờ qua
        now = datetime.now()
        yesterday = now - timedelta(days=1)
        
        recent_trades = [trade for trade in trades_data if datetime.fromisoformat(trade["close_date"]) > yesterday]
        
        # Tính toán thống kê
        total_trades = len(recent_trades)
        profitable_trades = sum(1 for trade in recent_trades if float(trade["profit_ratio"]) > 0)
        win_rate = profitable_trades / total_trades * 100 if total_trades > 0 else 0
        
        total_profit = sum(float(trade["profit_abs"]) for trade in recent_trades)
        
        # Tạo báo cáo
        report = f"""
        # Báo cáo hiệu suất giao dịch (24 giờ qua)

        - Tổng số giao dịch: {total_trades}
        - Giao dịch có lãi: {profitable_trades}
        - Tỷ lệ thắng: {win_rate:.2f}%
        - Tổng lợi nhuận: {total_profit:.4f} USDT
        
        ## Top 3 giao dịch có lãi cao nhất:
        """
        
        # Thêm top 3 giao dịch có lãi nhất
        top_profit_trades = sorted(recent_trades, key=lambda x: float(x["profit_abs"]), reverse=True)[:3]
        for i, trade in enumerate(top_profit_trades, start=1):
            report += f"""
            {i}. {trade["pair"]} - Lợi nhuận: {float(trade["profit_abs"]):.4f} USDT ({float(trade["profit_ratio"]) * 100:.2f}%)
            """
        
        return report
    
    except Exception as e:
        logger.error(f"Lỗi khi tạo báo cáo hiệu suất: {e}")
        return f"Lỗi khi tạo báo cáo: {e}"

def check_balance_changes():
    """
    Kiểm tra sự thay đổi số dư và cảnh báo nếu có sự sụt giảm đáng kể
    """
    logger.info("Kiểm tra sự thay đổi số dư...")
    
    try:
        # Đọc số dư từ file lưu trữ
        balance_file = os.path.join(BASE_DIR, "user_data", "balance_history.json")
        
        current_balance = 0
        
        # Giả định rằng chúng ta có cách truy cập số dư hiện tại
        # Đây chỉ là mô phỏng, cần điều chỉnh theo triển khai thực tế
        # current_balance = get_current_balance()
        
        if os.path.exists(balance_file):
            with open(balance_file, "r") as f:
                balance_history = json.load(f)
            
            # Lấy số dư gần nhất
            if balance_history:
                last_balance = balance_history[-1]["balance"]
                
                # Tính phần trăm thay đổi
                percent_change = (current_balance - last_balance) / last_balance * 100
                
                # Cảnh báo nếu giảm hơn 5%
                if percent_change < -5:
                    message = f"⚠️ Cảnh báo: Số dư đã giảm {abs(percent_change):.2f}% trong 24 giờ qua!"
                    send_telegram_message(message)
                    send_email_notification("Cảnh báo giảm số dư", message)
        
        # Lưu số dư hiện tại vào lịch sử
        if current_balance > 0:
            if not os.path.exists(balance_file):
                balance_history = []
            
            balance_history.append({
                "timestamp": datetime.now().isoformat(),
                "balance": current_balance
            })
            
            # Chỉ giữ lịch sử 30 ngày
            if len(balance_history) > 30:
                balance_history = balance_history[-30:]
            
            with open(balance_file, "w") as f:
                json.dump(balance_history, f, indent=2)
            
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra sự thay đổi số dư: {e}")

def monitor_and_report():
    """
    Hàm chính để giám sát và báo cáo trạng thái của bot
    """
    logger.info("Bắt đầu giám sát FreqTrade Bot...")
    
    # Kiểm tra sức khỏe của bot
    issues = check_bot_health()
    
    if issues:
        # Nếu có vấn đề, gửi thông báo
        message = "⚠️ FreqTrade Bot gặp vấn đề:\n\n" + "\n".join(issues)
        send_telegram_message(message)
        send_email_notification("FreqTrade Bot - Cảnh báo", message)
    
    # Kiểm tra sự thay đổi số dư
    check_balance_changes()
    
    # Tạo báo cáo hiệu suất giao dịch hàng ngày (vào lúc 8 giờ sáng)
    current_hour = datetime.now().hour
    if current_hour == 8:
        performance_report = generate_performance_report()
        send_telegram_message(performance_report)
        send_email_notification("FreqTrade Bot - Báo cáo hiệu suất hàng ngày", performance_report)

def main():
    """
    Xử lý tham số dòng lệnh và thực hiện tác vụ tương ứng
    """
    parser = argparse.ArgumentParser(description='FreqTrade Bot Monitoring System')
    parser.add_argument('--check-health', action='store_true', help='Kiểm tra sức khỏe của bot')
    parser.add_argument('--generate-report', action='store_true', help='Tạo báo cáo hiệu suất')
    parser.add_argument('--monitor', action='store_true', help='Chạy hệ thống giám sát')
    parser.add_argument('--telegram-test', action='store_true', help='Kiểm tra kết nối Telegram')
    parser.add_argument('--email-test', action='store_true', help='Kiểm tra kết nối email')
    
    args = parser.parse_args()
    
    if args.check_health:
        issues = check_bot_health()
        if issues:
            print("Bot gặp vấn đề:")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("Bot hoạt động bình thường.")
    
    elif args.generate_report:
        report = generate_performance_report()
        print(report)
    
    elif args.monitor:
        monitor_and_report()
    
    elif args.telegram_test:
        success = send_telegram_message("Đây là tin nhắn kiểm tra từ FreqTrade Bot Monitoring System.")
        print(f"Gửi telegram {'thành công' if success else 'thất bại'}")
    
    elif args.email_test:
        success = send_email_notification("Kiểm tra Email", "Đây là email kiểm tra từ FreqTrade Bot Monitoring System.")
        print(f"Gửi email {'thành công' if success else 'thất bại'}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
# AI Trading Bot cho Binance Futures

Hệ thống giao dịch tiên tiến tích hợp AI cho Binance Futures, giúp đạt mục tiêu lợi nhuận nhất quán với quản lý rủi ro tối ưu.

## Tính năng chính

- **Chiến lược giao dịch AI** kết hợp phân tích kỹ thuật truyền thống và dự đoán từ mô hình máy học
- **Phân tích tâm lý thị trường** từ tin tức sử dụng OpenAI API
- **Tối ưu hóa GPU** cho việc huấn luyện mô hình nhanh hơn sử dụng AMD RX6600
- **Quản lý rủi ro thông minh** với cắt lỗ động, đòn bẩy linh hoạt và phân bổ vốn tối ưu 
- **Tự động hóa hoàn toàn** với lịch huấn luyện lại mô hình, sao lưu và giám sát
- **Xếp hạng cặp tiền** tự động để chọn top 5 cặp tiền có tiềm năng giao dịch tốt nhất
- **Hỗ trợ cả Long và Short** trên thị trường Binance Futures

## Cài đặt trên Windows

### Yêu cầu hệ thống
- Windows 10/11
- Python 3.9+ 
- AMD RX6600 GPU
- RAM 32GB
- CPU Intel i5-12400k trở lên

### Bước 1: Cài đặt

Chạy file `install.bat` để cài đặt tất cả các phụ thuộc cần thiết:

```
install.bat
```

File này sẽ:
- Tạo môi trường ảo Python
- Cài đặt FreqTrade và các thư viện cần thiết
- Thiết lập các thư mục cần thiết 
- Tạo file `run.bat` để dễ dàng khởi động bot

### Bước 2: Cấu hình

Sửa file `.env` để thêm API keys của bạn:

```
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret
OPENAI_API_KEY=your_openai_api_key

# Thêm thông tin Telegram nếu muốn nhận thông báo
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

Hoặc chạy trình hướng dẫn thiết lập cấu hình:

```
python setup_config.py --wizard
```

### Bước 3: Huấn luyện mô hình AI

Huấn luyện các mô hình AI trước khi giao dịch:

```
python run_freqtrade.py --mode train
```

hoặc chọn tùy chọn "1. Train AI models" trong menu `run.bat`

## Sử dụng

### Chạy Bot

Chạy file `run.bat` và chọn một trong các tùy chọn:

1. **Train AI models** - Huấn luyện lại các mô hình AI
2. **Run backtest** - Chạy backtesting để kiểm tra hiệu suất chiến lược
3. **Run trading bot** - Khởi động bot giao dịch thực
4. **Run simulation** - Chạy mô phỏng trên dữ liệu giả lập
5. **Exit** - Thoát

### Tìm Top Cặp Tiền

Để xác định top 5 cặp tiền có tiềm năng giao dịch cao nhất:

```
python pair_ranking.py
```

Kết quả sẽ được lưu vào `pair_ranking.json` và tạo cấu hình mới `config/top_pairs_config.json`.

### Giám sát Bot

Giám sát trạng thái và hiệu suất của bot:

```
python monitoring.py --check-health
python monitoring.py --generate-report
```

## Cấu trúc Dự án

```
├── config/                      # Thư mục chứa file cấu hình
│   ├── config.json              # Cấu hình chính cho giao dịch thật
│   └── backtest_config.json     # Cấu hình cho backtesting
├── user_data/                   # Thư mục dữ liệu người dùng (FreqTrade)
│   ├── strategies/              # Thư mục chiến lược
│   │   └── enhanced_crypto_strategy.py  # Chiến lược giao dịch chính
│   ├── models/                  # Thư mục lưu mô hình AI
│   ├── data/                    # Dữ liệu thị trường
│   ├── logs/                    # Logs
│   └── backups/                 # Sao lưu
├── market_sentiment.py          # Module phân tích tâm lý thị trường
├── train_ai_models.py           # Script huấn luyện mô hình AI
├── run_freqtrade.py             # Script chạy FreqTrade
├── pair_ranking.py              # Xếp hạng và lọc cặp tiền tiềm năng
├── simulation.py                # Mô phỏng chiến lược
├── monitoring.py                # Hệ thống giám sát bot
├── scheduler.py                 # Quản lý lịch các tác vụ tự động
├── setup_config.py              # Trình thiết lập cấu hình
├── run.bat                      # Script chạy bot trên Windows
├── install.bat                  # Script cài đặt trên Windows
└── .env                         # File chứa biến môi trường và API keys
```

## Cấu hình Khuyến nghị

### Cấu hình Giao dịch
- **Số cặp giao dịch đồng thời**: 3 (đã được tối ưu)
- **Cắt lỗ**: 2% (có thể điều chỉnh tùy theo khẩu vị rủi ro)
- **Đòn bẩy**: 3x mặc định, tối đa 5x (điều chỉnh động theo biến động thị trường)
- **Khung thời gian**: 5m (hỗ trợ 15m, 1h, 4h)

### Cấu hình GPU (AMD RX6600)
Sử dụng OpenCL để tăng tốc LightGBM:
```
GPU_PLATFORM_ID=0
GPU_DEVICE_ID=0
```

## Quản lý Rủi ro

Hệ thống tích hợp nhiều lớp quản lý rủi ro:

1. **Cắt lỗ Thông minh**: Điều chỉnh dựa trên ATR và biến động thị trường
2. **Trailing Stop**: Bảo vệ lợi nhuận khi thị trường đảo chiều
3. **Đòn bẩy Động**: Tự động điều chỉnh đòn bẩy từ 2x đến 5x tùy thuộc vào biến động
4. **Phân bổ Vốn Tối ưu**: Mỗi cặp sử dụng tối đa 33% vốn giao dịch
5. **Lọc Cặp Thông minh**: Chỉ giao dịch top 5 cặp tiền có tiềm năng cao nhất

## Giấy phép

[MIT License](LICENSE)

# Chiến lược High Frequency Trend + Liquidation Hunting

Chiến lược này kết hợp các kỹ thuật giao dịch tần suất cao (HFT) với các mô hình phát hiện stop hunt / liquidity grab và phân tích mẫu hình nến, tạo ra một hệ thống giao dịch toàn diện cho các thị trường biến động.

## Tổng quan

Chiến lược High Frequency Trend + Liquidation Hunting (viết tắt: HFT-LH) tập trung vào việc:

1. Giao dịch theo xu hướng ngắn hạn với tần suất cao
2. Phát hiện và tận dụng các "stop hunt" (liquidity grab) do các whales tạo ra
3. Sử dụng phân tích mẫu hình nến để xác nhận tín hiệu
4. Quản lý vốn và rủi ro động dựa trên biến động thị trường

## Kỹ thuật giao dịch chính

### 1. High Frequency Trend Following

- Sử dụng các EMA siêu ngắn (3-15 nến)
- Phát hiện xu hướng sớm thông qua EMA crossover
- Xác nhận xu hướng với các chỉ báo momentum (RSI, Stochastic)

### 2. Stop Hunt & Liquidation Detection

- Phát hiện khi whales đang "hunt" các stop loss
- Nhận diện các điểm giá giả vờ phá vỡ hỗ trợ/kháng cự rồi nhanh chóng quay đầu
- Tận dụng psychology của thị trường để vào lệnh ngược dòng sau các cú "stop hunt"

### 3. Candlestick Pattern Analysis

Phát hiện các mẫu hình nến quan trọng:
- Engulfing (Nuốt chửng)
- Pinbar (Rejection)
- Inside & Outside Bar
- 3 Soldiers & 3 Crows (3 chiến binh & 3 quạ đen)
- Morning & Evening Star (Sao mai & Sao hôm)

### 4. VWAP Mean Reversion

- Đo lường độ lệch của giá so với VWAP trong ngày
- Giao dịch mean reversion khi độ lệch quá lớn
- Kết hợp với phân tích xu hướng để hạn chế giao dịch counter-trend yếu

## Hệ thống tính điểm tín hiệu

### Điểm Long Signal (0-100)

1. **Trend following (30%)**: EMA uptrend, giá trên các đường trung bình
2. **Mẫu hình nến (25%)**: Engulfing, Pinbar, Three Soldiers, Morning Star
3. **Stop hunt detection (20%)**: Phát hiện stop hunt dưới
4. **Momentum & Oscillators (15%)**: RSI, Stochastic, MACD
5. **Volatility (10%)**: ATR, Squeeze

### Điểm Short Signal (0-100)

1. **Trend following (30%)**: EMA downtrend, giá dưới các đường trung bình 
2. **Mẫu hình nến (25%)**: Engulfing, Pinbar, Three Crows, Evening Star
3. **Stop hunt detection (20%)**: Phát hiện stop hunt trên
4. **Momentum & Oscillators (15%)**: RSI, Stochastic, MACD
5. **Volatility (10%)**: ATR, Squeeze

## Quản lý vốn và rủi ro

### Đòn bẩy động

Dựa trên điểm tín hiệu và biến động (ATR):

- Điểm > 85 + Biến động thấp: Đòn bẩy tối đa (8x)
- Điểm > 85 + Biến động cao: Đòn bẩy 6x
- Điểm > 75 + Biến động thấp: Đòn bẩy 6x
- Điểm > 75 + Biến động cao: Đòn bẩy 5x
- Điểm > 65: Đòn bẩy 4x
- Điểm < 65: Đòn bẩy mặc định (3x)

### Stoploss động

- **ATR-based**: Sử dụng 1.5x ATR cho stoploss ban đầu
- **Trailing stop**: Khi lợi nhuận > 1%, giảm stoploss xuống 0.5x ATR
- **Breakeven**: Điều chỉnh stoploss về giá vào lệnh khi lợi nhuận đủ lớn

### ROI

Chiến lược sử dụng ROI giảm dần nhanh chóng theo thời gian:
- 0 phút: 2% (mục tiêu ban đầu) 
- 10 phút: 1.5%
- 20 phút: 1%
- 30 phút: 0.5%

## Lọc tín hiệu thông minh

### Xác nhận vào lệnh

- Điểm tín hiệu > 65 
- ATR đủ lớn (biến động đủ để sinh lợi)
- Khối lượng đủ (> 70% khối lượng trung bình)
- Chi phí giao dịch thấp hơn tiềm năng lợi nhuận

### Xác nhận thoát lệnh

- Đạt mục tiêu ATR (profit factor x ATR)
- Đảo chiều EMA
- Mẫu hình nến đảo chiều mạnh
- Thay đổi đột ngột trong khối lượng và biến động

## Tham số tối ưu hóa

- **trend_ema1**: EMA ngắn (3-15, default 8)
- **trend_ema2**: EMA dài (15-50, default 21) 
- **rsi_enter_long**: Ngưỡng RSI cho long (20-40, default 30)
- **rsi_enter_short**: Ngưỡng RSI cho short (60-80, default 70)
- **profit_factor**: Hệ số lợi nhuận dựa trên ATR (1.0-3.0, default 1.5)
- **hunt_threshold**: Ngưỡng phát hiện stop hunt (0.5-3.0, default 1.2)

## Ưu điểm và hạn chế

### Ưu điểm
- Tần suất giao dịch cao, khai thác nhiều cơ hội
- Khai thác tâm lý thị trường và hành vi của whales
- Quản lý rủi ro chặt chẽ với đòn bẩy và stoploss động
- Kết hợp nhiều kỹ thuật giao dịch bổ sung cho nhau

### Hạn chế
- Cần độ trễ thấp để thực hiện hiệu quả
- Chi phí giao dịch cao do tần suất giao dịch nhiều
- Có thể bị "overfit" nếu tối ưu hóa quá nhiều
- Cần tập dữ liệu chất lượng cao để huấn luyện và tối ưu hóa

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Biến động trung bình đến cao
- Chi phí giao dịch thấp
- Khối lượng giao dịch lớn
- Có nhiều whales và tổ chức tham gia

Nên sử dụng với staking thấp hơn và số lệnh đồng thời cao hơn để phân tán rủi ro.

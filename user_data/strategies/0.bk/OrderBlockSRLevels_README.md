# Chiến lược Order Block + Support/Resistance Levels

Chiến lược này kết hợp phân tích Order Block (khối lệnh) và Support/Resistance Levels (vùng hỗ trợ/kháng cự). Được tối ưu hóa cho khung thời gian 1 giờ (1h).

## Tổng quan

OrderBlockSRLevels Strategy là chiến lược giao dịch trung hạn kết hợp hai phương pháp phân tích kỹ thuật tiên tiến:

1. **Order Block**: Mô hình giao dịch xác định các vùng giá nơi các "smart money" (tổ chức, nhà đầu tư lớn) đã đặt lệnh. Các vùng này thường là điểm bật giá mạnh mẽ khi được test lại.
2. **Support/Resistance Levels**: Xác định các mức giá quan trọng đã được thị trường kiểm nghiệm nhiều lần, là nơi giá thường có phản ứng.

Chiến lược này tập trung vào việc tìm kiếm vùng hội tụ (confluence) giữa Order Blocks và SR Levels để xác định điểm vào lệnh với xác suất thành công cao.

## Phương pháp giao dịch

### 1. Order Block Analysis

- **Bullish Order Block**: Nến giảm xuất hiện trước chuỗi nến tăng mạnh
- **Bearish Order Block**: Nến tăng xuất hiện trước chuỗi nến giảm mạnh
- **OB Identification**: Sử dụng thuật toán phân tích các chuỗi nến có biến động lớn để xác định OB
- **Tham số tối ưu**: ob_window (10-50, mặc định 20), ob_strength (0.2-1.5, mặc định 0.5)

### 2. Support/Resistance Analysis

- **Swing High/Low Detection**: Xác định các điểm swing high/low qua chuỗi thời gian
- **Level Clustering**: Nhóm các mức giá gần nhau thành một mức SR chính
- **Confirmation**: Chỉ sử dụng các mức đã được xác nhận bởi nhiều lần chạm (3+ lần)
- **Tham số tối ưu**: sr_window (5-30, mặc định 14), sr_confirmation (2-6, mặc định 3)

### 3. Trend Analysis

- **EMA Crossover**: Sử dụng EMA ngắn và dài để xác định xu hướng
- **Trend Strength**: Đánh giá sức mạnh xu hướng thông qua độ dốc của EMA
- **Tham số tối ưu**: ema_short (5-30, mặc định 21), ema_long (30-100, mặc định 55)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Order Block + SR Confluence**:
   - Giá trở lại vùng Bullish Order Block
   - Vùng OB gần với mức hỗ trợ (<2% khoảng cách)
   - RSI < 45 (không quá cao)
   - Không trong downtrend mạnh
   - Tín hiệu đủ mạnh (>60 điểm)

2. **Strong Support Bounce**:
   - Giá rất gần vùng hỗ trợ (<0.5%)
   - RSI < 40
   - Trong uptrend
   - Volume tốt

3. **Pull back to EMA trong uptrend mạnh**:
   - Uptrend mạnh và ổn định
   - Giá pullback về EMA ngắn (21 EMA)
   - RSI < 45
   - Volume ổn định

4. **Bollinger Band Bounce trong uptrend**:
   - Trong uptrend
   - Giá chạm BB lower band
   - RSI < 40
   - Volume tốt

### Tín hiệu Short

1. **Order Block + SR Confluence**:
   - Giá trở lại vùng Bearish Order Block
   - Vùng OB gần với mức kháng cự (<2% khoảng cách)
   - RSI > 55 (không quá thấp)
   - Không trong uptrend mạnh
   - Tín hiệu đủ mạnh (>60 điểm)

2. **Strong Resistance Rejection**:
   - Giá rất gần vùng kháng cự (<0.5%)
   - RSI > 60
   - Trong downtrend
   - Volume tốt

3. **Pull back to EMA trong downtrend mạnh**:
   - Downtrend mạnh và ổn định
   - Giá pullback về EMA ngắn (21 EMA)
   - RSI > 55
   - Volume ổn định

4. **Bollinger Band Bounce trong downtrend**:
   - Trong downtrend
   - Giá chạm BB upper band
   - RSI > 60
   - Volume tốt

## Hệ thống tính điểm tín hiệu

Chiến lược sử dụng hệ thống tính điểm toàn diện (0-100) để đánh giá chất lượng tín hiệu:

### Tín hiệu Bullish (max 100 điểm)

1. **Khoảng cách đến hỗ trợ** (max 30 điểm):
   - < 0.5%: 30 điểm
   - < 1.0%: 25 điểm
   - < 1.5%: 20 điểm
   - < 2.0%: 15 điểm

2. **RSI** (max 30 điểm):
   - < 30: 30 điểm
   - < 35: 25 điểm
   - < 40: 20 điểm
   - < 45: 15 điểm

3. **Volume** (max 20 điểm):
   - > 2.0x avg: 20 điểm
   - > 1.5x avg: 15 điểm
   - > 1.0x avg: 10 điểm

4. **Order Block recency** (max 20 điểm):
   - OB trong 1-2 nến: 20 điểm
   - OB trong 3-4 nến: 15 điểm
   - OB trong 5-10 nến: 10 điểm
   - OB cũ hơn: 5 điểm

### Tín hiệu Bearish (max 100 điểm)

1. **Khoảng cách đến kháng cự** (max 30 điểm):
   - < 0.5%: 30 điểm
   - < 1.0%: 25 điểm
   - < 1.5%: 20 điểm
   - < 2.0%: 15 điểm

2. **RSI** (max 30 điểm):
   - > 70: 30 điểm
   - > 65: 25 điểm
   - > 60: 20 điểm
   - > 55: 15 điểm

3. **Volume** (max 20 điểm):
   - > 2.0x avg: 20 điểm
   - > 1.5x avg: 15 điểm
   - > 1.0x avg: 10 điểm

4. **Order Block recency** (max 20 điểm):
   - OB trong 1-2 nến: 20 điểm
   - OB trong 3-4 nến: 15 điểm
   - OB trong 5-10 nến: 10 điểm
   - OB cũ hơn: 5 điểm

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên sức mạnh tín hiệu:

- **Tín hiệu rất mạnh** (>80 điểm + volume cao): Đòn bẩy tối đa (6x)
- **Tín hiệu mạnh** (>70 điểm): Đòn bẩy cao (5x)
- **Tín hiệu khá** (>60 điểm): Đòn bẩy trung bình cao (4x)
- **Tín hiệu cơ bản**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -4%
- **Với lợi nhuận > 2%**: Stoploss di chuyển lên/xuống theo mức SR gần nhất
- **Trailing stop**: Kích hoạt ở mức 3% lợi nhuận, step 2%

### ROI (Return on Investment)

- **0 giờ**: 6% lợi nhuận
- **24 giờ**: 4% lợi nhuận
- **48 giờ**: 2.5% lợi nhuận
- **72 giờ**: 1.5% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **Sức mạnh tín hiệu**: Không giao dịch khi điểm tín hiệu < 50
3. **Gần mức SR ngược**: Không long khi gần kháng cự, không short khi gần hỗ trợ (<1%)
4. **Timeframe thấp hơn**: Kiểm tra các chỉ báo trên timeframe 15m để xác nhận tín hiệu

## Thoát lệnh thông minh

### Thoát Long

1. **Chạm vùng kháng cự**: Khi giá gần mức kháng cự (<0.75%) và RSI > 65
2. **Break EMA**: Khi giá từ trên cắt xuống dưới EMA ngắn trong uptrend
3. **RSI quá cao**: Khi RSI > 78
4. **Trend change**: Khi EMA ngắn cắt xuống dưới EMA dài
5. **Near Bearish OB**: Khi giá gần Bearish Order Block và RSI > 65

### Thoát Short

1. **Chạm vùng hỗ trợ**: Khi giá gần mức hỗ trợ (<0.75%) và RSI < 35
2. **Break EMA**: Khi giá từ dưới cắt lên trên EMA ngắn trong downtrend
3. **RSI quá thấp**: Khi RSI < 22
4. **Trend change**: Khi EMA ngắn cắt lên trên EMA dài
5. **Near Bullish OB**: Khi giá gần Bullish Order Block và RSI < 35

## Tham số tối ưu hóa

### Tham số Order Block
- **ob_window**: Cửa sổ xác định OB (10-50, mặc định 20)
- **ob_strength**: Ngưỡng độ mạnh OB (0.2-1.5, mặc định 0.5)
- **ob_confirmation**: Số nến xác nhận (1-4, mặc định 2)

### Tham số Support/Resistance
- **sr_window**: Cửa sổ xác định swing high/low (5-30, mặc định 14)
- **sr_confirmation**: Số lần chạm xác nhận (2-6, mặc định 3)
- **sr_threshold**: % sai số cho phép (0.1-0.5, mặc định 0.2)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

### Tham số Trend
- **ema_short**: EMA ngắn (5-30, mặc định 21)
- **ema_long**: EMA dài (30-100, mặc định 55)

## Ưu điểm của chiến lược

1. **Tìm kiếm giá trị thực**: Xác định các vùng giá thực sự quan trọng (nơi smart money giao dịch)
2. **Đa lớp xác nhận**: Kết hợp Order Block, SR Levels, RSI, và trend analysis
3. **Tránh giao dịch giả**: Hệ thống điểm số toàn diện giúp lọc bỏ các tín hiệu yếu
4. **Quản lý rủi ro tối ưu**: Stoploss dựa trên mức SR gần nhất thay vì % cố định

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 1 giờ
- Xu hướng rõ ràng, không quá nhiều tích lũy
- Volume đủ lớn và ổn định
- Có sự tham gia của tổ chức và nhà đầu tư lớn (smart money)

# Chiến lược EMA Cross + Volume + Bollinger Bands

Chiến lược này kết hợp phân tích xu hướng dựa trên EMA, biến động dựa trên Bollinger Bands và xác nhận khối lượng giao dịch. Được tối ưu hóa cho khung thời gian 15 phút (15m).

## Tổng quan

EMACrossVolumeBB Strategy là chiến lược toàn diện kết hợp 3 phương pháp phân tích kỹ thuật phổ biến:

1. **EMA Cross**: Phát hiện xu hướng và điểm vào lệnh.
2. **Bollinger Bands**: Xác định vùng quá mua/quá bán và biến động thị trường.
3. **Volume Analysis**: Lọc và xác nhận tín hiệu dựa trên khối lượng.

Thêm vào đó, chiến lược sử dụng OBV (On Balance Volume) và RSI để xác nhận tín hiệu và tránh các giao dịch giả.

## Phương pháp giao dịch

### 1. EMA Cross

- **Xác định xu hướng**: Sử dụng EMA ngắn và dài để xác định xu hướng thị trường.
- **Điểm vào lệnh**: Khi EMA ngắn cắt lên trên EMA dài (mua) hoặc cắt xuống dưới EMA dài (bán).
- **Tham số tối ưu**: EMA ngắn (3-15, mặc định 8) và EMA dài (15-50, mặc định 21).

### 2. Bollinger Bands

- **Vùng quá mua/quá bán**: Mua khi giá chạm BB lower, bán khi giá chạm BB upper.
- **Phát hiện Squeeze**: Giao dịch khi BB squeeze (thắt chặt) và sau đó mở rộng.
- **Tham số tối ưu**: Chu kỳ (15-30, mặc định 20) và độ lệch chuẩn (1.5-3.0, mặc định 2.0).

### 3. Volume Analysis

- **Xác nhận tín hiệu**: Chỉ giao dịch khi volume cao hơn trung bình.
- **Volume Ratio**: So sánh volume hiện tại với trung bình động 20 nến.
- **Tham số tối ưu**: Hệ số volume (1.0-5.0, mặc định 2.0).

### 4. Oscillators

- **RSI**: Xác định điều kiện quá mua/quá bán.
- **MACD**: Xác nhận tín hiệu và thoát lệnh.
- **OBV**: Xác thực xu hướng dựa trên khối lượng tích lũy.

## Các tín hiệu vào lệnh

### Tín hiệu Long

1. **EMA Cross Up + RSI thấp**: EMA ngắn cắt lên trên EMA dài và RSI dưới ngưỡng, với volume cao.
2. **BB Lower Touch trong Uptrend**: Giá chạm BB lower trong xu hướng tăng, với RSI thấp.
3. **BB Squeeze Breakout**: BB squeeze kết thúc với giá tăng trên BB middle.
4. **OBV Confirmation**: OBV trên EMA của nó, trong xu hướng tăng.

### Tín hiệu Short

1. **EMA Cross Down + RSI cao**: EMA ngắn cắt xuống dưới EMA dài và RSI trên ngưỡng, với volume cao.
2. **BB Upper Touch trong Downtrend**: Giá chạm BB upper trong xu hướng giảm, với RSI cao.
3. **BB Squeeze Breakdown**: BB squeeze kết thúc với giá giảm dưới BB middle.
4. **OBV Confirmation**: OBV dưới EMA của nó, trong xu hướng giảm.

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên biến động (BB width):

- **Biến động rất thấp** (BB width < 0.03): Đòn bẩy tối đa (6x)
- **Biến động thấp** (BB width < 0.05): Đòn bẩy cao (5x)
- **Biến động trung bình** (BB width < 0.08): Đòn bẩy trung bình (4x)
- **Biến động cao** (BB width >= 0.08): Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -3%
- **Với lợi nhuận > 2%**: Stoploss di chuyển lên/xuống theo BB middle band
- **Trailing stop**: Kích hoạt ở mức 2% lợi nhuận, step 1%

### ROI (Return on Investment)

- **0 phút**: 5% lợi nhuận
- **30 phút**: 2.5% lợi nhuận
- **60 phút**: 1.5% lợi nhuận
- **90 phút**: 1% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 50% trung bình.
2. **Biến động cao**: Không giao dịch khi BB width > 0.1 (biến động quá cao).
3. **Nến không ổn định**: Bỏ qua các mẫu nến không rõ ràng hoặc có bóng quá dài.

## Thoát lệnh thông minh

### Thoát Long

1. **EMA Cross Down**: Khi EMA ngắn cắt xuống dưới EMA dài.
2. **BB Upper + RSI cao**: Khi giá chạm BB upper và RSI > ngưỡng bán.
3. **MACD Cross Down**: Khi MACD cắt xuống dưới signal line.
4. **OBV giảm mạnh**: Khi OBV giảm hơn 5% trong 3 nến.

### Thoát Short

1. **EMA Cross Up**: Khi EMA ngắn cắt lên trên EMA dài.
2. **BB Lower + RSI thấp**: Khi giá chạm BB lower và RSI < ngưỡng mua.
3. **MACD Cross Up**: Khi MACD cắt lên trên signal line.
4. **OBV tăng mạnh**: Khi OBV tăng hơn 5% trong 3 nến.

## Tham số tối ưu hóa

### Tham số EMA
- **buy_ema_short**: EMA ngắn cho tín hiệu mua (3-15, mặc định 8)
- **buy_ema_long**: EMA dài cho tín hiệu mua (15-50, mặc định 21)
- **sell_ema_short**: EMA ngắn cho tín hiệu bán (3-15, mặc định 8)
- **sell_ema_long**: EMA dài cho tín hiệu bán (15-50, mặc định 21)

### Tham số Bollinger Bands
- **bb_period**: Chu kỳ cho Bollinger Bands (15-30, mặc định 20)
- **bb_std**: Độ lệch chuẩn cho Bollinger Bands (1.5-3.0, mặc định 2.0)

### Tham số Volume
- **volume_factor**: Hệ số khối lượng so với trung bình (1.0-5.0, mặc định 2.0)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. **Đa dạng tín hiệu**: Kết hợp nhiều loại tín hiệu từ các phương pháp khác nhau.
2. **Tránh tín hiệu giả**: Lọc tín hiệu bằng khối lượng và oscillators.
3. **Thích ứng với biến động**: Điều chỉnh đòn bẩy và stoploss theo biến động thị trường.
4. **Thoát lệnh linh hoạt**: Nhiều cơ chế thoát lệnh kết hợp giảm rủi ro.

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 15 phút
- Xu hướng rõ ràng (không sideways)
- Khối lượng giao dịch ổn định
- Biến động trung bình

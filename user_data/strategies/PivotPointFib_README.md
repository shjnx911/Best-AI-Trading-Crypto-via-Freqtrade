# Chiến lược Pivot Point + Fibonacci

Chiến lược này kết hợp Pivot Points với Fibonacci Retracement/Extension để xác định các mức hỗ trợ và kháng cự quan trọng. Được tối ưu hóa cho khung thời gian 1 giờ (1h).

## Tổng quan

PivotPointFib Strategy là chiến lược giao dịch tầm trung kết hợp hai công cụ phân tích kỹ thuật mạnh mẽ:

1. **Pivot Points**: Mức giá tham chiếu được tính toán từ giá high, low, close của phiên trước, giúp xác định các mức hỗ trợ và kháng cự quan trọng mà giá thường phản ứng.
2. **Fibonacci Retracement/Extension**: Công cụ xác định các mức giá tiềm năng dựa trên tỷ lệ Fibonacci, giúp dự đoán mức giá sẽ chạm đến trong khi pullback hoặc breakout.

Chiến lược tập trung vào việc giao dịch các vùng hội tụ (confluence) - nơi các mức Pivot Points và Fibonacci trùng nhau, tạo ra các mức hỗ trợ/kháng cự mạnh.

## Phương pháp giao dịch

### 1. Pivot Points

Có ba loại Pivot Points được hỗ trợ:
- **Standard**: Phương pháp truyền thống, dựa trên high, low, close của phiên trước
- **Fibonacci**: Sử dụng các tỷ lệ Fibonacci để xác định mức hỗ trợ/kháng cự
- **Camarilla**: Sử dụng phép nhân cụ thể để xác định các mức hỗ trợ/kháng cự chi tiết hơn

Mỗi loại Pivot Points tạo ra các mức giá:
- **Pivot (PP)**: Mức cơ sở
- **Resistance levels (R1, R2, R3)**: Các mức kháng cự
- **Support levels (S1, S2, S3)**: Các mức hỗ trợ

### 2. Fibonacci Retracement/Extension

- **Retracement levels**: 23.6%, 38.2%, 50.0%, 61.8%, 78.6% (giảm giá từ đỉnh)
- **Extension levels**: 127.2%, 138.2%, 150.0%, 161.8%, 200.0% (mở rộng từ đáy)
- **Swing high/low**: Được xác định trong một cửa sổ nhất định (10-50 nến)

### 3. Confluence Zones (Vùng hội tụ)

- **Support Confluence**: Khi mức hỗ trợ Pivot (S1, S2, S3) trùng với mức Fibonacci Retracement (38.2%, 50%, 61.8%)
- **Resistance Confluence**: Khi mức kháng cự Pivot (R1, R2, R3) trùng với mức Fibonacci Extension (127.2%, 138.2%, 161.8%)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Bounce từ Support Confluence**: 
   - Giá phản ứng từ vùng hội tụ hỗ trợ (Pivot + Fibonacci)
   - Trong uptrend (giá trên SMA50, SMA20 trên SMA50)
   - RSI trong vùng oversold (< 30)
   - Volume ổn định (> 80% trung bình)

2. **Pullback to Support trong Uptrend mạnh**:
   - Giá gần mức hỗ trợ Pivot
   - SMA20 có xu hướng tăng
   - Nến tăng (close > open)
   - RSI hợp lý (40-60)
   - Volume tốt

3. **Breakout trên R1 Pivot Point**:
   - Giá vượt qua R1 sau khi test thành công
   - Trong uptrend
   - Nến tăng mạnh (> 0.5 ATR)
   - Volume tăng đột biến

### Tín hiệu Short

1. **Bounce từ Resistance Confluence**:
   - Giá phản ứng từ vùng hội tụ kháng cự (Pivot + Fibonacci)
   - Trong downtrend (giá dưới SMA50, SMA20 dưới SMA50)
   - RSI trong vùng overbought (> 70)
   - Volume ổn định

2. **Pullback to Resistance trong Downtrend mạnh**:
   - Giá gần mức kháng cự Pivot
   - SMA20 có xu hướng giảm
   - Nến giảm (close < open)
   - RSI hợp lý (40-60)
   - Volume tốt

3. **Breakdown dưới S1 Pivot Point**:
   - Giá phá S1 sau khi test thành công
   - Trong downtrend
   - Nến giảm mạnh (> 0.5 ATR)
   - Volume tăng đột biến

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên khoảng cách từ giá hiện tại đến mức hỗ trợ/kháng cự gần nhất và hội tụ:

#### Long:
- **Tối ưu** (Hội tụ support + Khoảng cách đến R1 > 2% + RSI < 40): Đòn bẩy tối đa (6x)
- **Tốt** (Hội tụ support + Khoảng cách đến R1 > 1.5%): Đòn bẩy cao (5x)
- **Khá** (Khoảng cách đến R1 > 1%): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

#### Short:
- **Tối ưu** (Hội tụ resistance + Khoảng cách đến S1 > 2% + RSI > 60): Đòn bẩy tối đa (6x)
- **Tốt** (Hội tụ resistance + Khoảng cách đến S1 > 1.5%): Đòn bẩy cao (5x)
- **Khá** (Khoảng cách đến S1 > 1%): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -5%
- **Với lợi nhuận > 2%**: Stoploss di chuyển lên/xuống theo Pivot level
- **Trailing stop**: Kích hoạt ở mức 2% lợi nhuận, step 1%

### ROI (Return on Investment)

- **0 giờ**: 5% lợi nhuận
- **12 giờ**: 3.5% lợi nhuận
- **24 giờ**: 2.5% lợi nhuận
- **48 giờ**: 1.5% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **Pivot ngược chiều gần**: Không long khi gần resistance pivot, không short khi gần support pivot
3. **RSI Extreme**: Không long khi RSI > 75, không short khi RSI < 25
4. **Vùng hội tụ (confluence)**: Ưu tiên giao dịch ở các vùng hội tụ giữa Pivot và Fibonacci

## Thoát lệnh thông minh

### Thoát Long

1. **Chạm kháng cự**: Khi giá gần mức R1/R2 Pivot và RSI > 70
2. **Phá vỡ SMA20**: Khi giá từ trên cắt xuống dưới SMA20 trong uptrend
3. **RSI quá cao**: Khi RSI > 80
4. **Chuyển xu hướng**: Khi uptrend kết thúc

### Thoát Short

1. **Chạm hỗ trợ**: Khi giá gần mức S1/S2 Pivot và RSI < 30
2. **Phá vỡ SMA20**: Khi giá từ dưới cắt lên trên SMA20 trong downtrend
3. **RSI quá thấp**: Khi RSI < 20
4. **Chuyển xu hướng**: Khi downtrend kết thúc

## Tham số tối ưu hóa

### Tham số Pivot Point
- **pivot_type**: Loại Pivot Point ('standard', 'fibonacci', 'camarilla', mặc định 'fibonacci')

### Tham số Fibonacci
- **fib_window**: Cửa sổ tính swing high/low (10-50, mặc định 20)

### Tham số xác định tín hiệu
- **pivot_bounce_pct**: % khoảng cách coi là gần pivot (0.05-0.5, mặc định 0.1)
- **fib_tolerance_pct**: % dung sai cho vùng hội tụ (0.05-0.5, mặc định 0.1)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. **Dựa trên nguyên lý thị trường**: Pivot Points và Fibonacci là các nguyên lý được trader sử dụng rộng rãi.
2. **Định lượng chính xác**: Cung cấp các mức giá cụ thể để vào lệnh, chốt lời và cắt lỗ.
3. **Xác định vùng hội tụ**: Tìm ra các vùng có nhiều mức hỗ trợ/kháng cự trùng nhau, tăng độ tin cậy.
4. **Linh hoạt**: Hỗ trợ nhiều loại Pivot Points và có thể tùy chỉnh cho phù hợp với từng thị trường.

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 1 giờ
- Biến động trung bình (không quá dữ dội hoặc quá ít)
- Coin và token có vốn hóa trung bình đến lớn
- Thị trường có xu hướng rõ ràng (uptrend hoặc downtrend)

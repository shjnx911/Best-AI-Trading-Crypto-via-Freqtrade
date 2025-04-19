# Chiến lược Breakout Trading kết hợp với VWAP Mean Reversion

Chiến lược này kết hợp kỹ thuật Breakout Trading (giao dịch đột phá) với VWAP Mean Reversion (hồi quy trung bình theo khối lượng), cung cấp hai phương pháp giao dịch có thể bổ sung cho nhau và tạo ra nhiều cơ hội giao dịch chất lượng cao hơn.

## Tổng quan

Chiến lược Breakout Trading kết hợp với VWAP Mean Reversion có hai phương pháp giao dịch chính:

1. **Breakout Trading**: Phát hiện và giao dịch các breakout từ mô hình tích lũy (Darvas Box)
2. **VWAP Mean Reversion**: Giao dịch khi giá quay trở lại VWAP sau khi đã lệch xa

Hệ thống cũng tích hợp các kỹ thuật Fibonacci để xác định mục tiêu lợi nhuận và thoát lệnh hiệu quả.

## Phương pháp giao dịch

### 1. Darvas Box Breakout

- **Nguyên lý**: Xác định các vùng tích lũy (Box) và phát hiện khi giá đột phá khỏi vùng này với volume cao
- **Lọc tín hiệu**: Chỉ giao dịch các breakout có volume cao và tương thích với xu hướng ở timeframe cao hơn
- **Đánh giá**: Mỗi tín hiệu breakout được đánh giá độ tin cậy từ 0-100 dựa trên nhiều yếu tố

### 2. VWAP Mean Reversion

- **Nguyên lý**: Khi giá lệch xa VWAP, có xu hướng quay trở lại mức này
- **Lọc tín hiệu**: Chờ giá bắt đầu quay trở lại VWAP (đảo chiều) sau khi đã lệch xa
- **Kết hợp xu hướng**: Chỉ giao dịch Mean Reversion theo đúng xu hướng chính của timeframe cao hơn

### 3. Fibonacci Levels

- Sử dụng các mức Fibonacci để xác định mục tiêu lợi nhuận
- Các mức chính được sử dụng: 23.6%, 38.2%, 50%, 61.8%, 78.6%
- Tham số tối ưu hóa cho phép chọn mức Fibonacci tối ưu

## Các yếu tố kỹ thuật chính

### 1. Darvas Box
- Sử dụng để xác định các vùng tích lũy trước breakout
- Phát hiện breakout khi giá vượt qua high/low của box với đủ margin

### 2. VWAP (Volume Weighted Average Price)
- Chỉ báo tham chiếu trung bình giá theo khối lượng
- Đo lường sự lệch của giá so với VWAP
- Xác định các giao cắt VWAP

### 3. Volume Analysis
- Đo lường khối lượng giao dịch so với trung bình
- Xác nhận breakout bằng volume tăng đột biến
- Lọc bỏ các tín hiệu khi volume quá thấp

### 4. Trend Analysis
- Sử dụng EMA để xác định xu hướng tổng thể
- Kết hợp ADX để đo lường sức mạnh xu hướng
- Phân tích xu hướng ở timeframe cao hơn

## Hệ thống tính điểm tin cậy

Mỗi tín hiệu breakout được đánh giá theo thang điểm 0-100 dựa trên:

1. **Volume (30%)**: Mức độ tăng của volume so với trung bình
2. **Vùng tích lũy (25%)**: Độ dài của vùng tích lũy trước breakout
3. **Sức mạnh breakout (20%)**: % khoảng cách vượt qua high/low của box
4. **ADX - Sức mạnh xu hướng (15%)**: Đo lường bằng chỉ báo ADX
5. **Phù hợp xu hướng (10%)**: Breakout có phù hợp với xu hướng tổng thể không

## Quản lý vốn và rủi ro

### Đòn bẩy động
- Điểm tin cậy > 85: Đòn bẩy tối đa (8x)
- Điểm tin cậy > 75: Đòn bẩy cao (6x)
- Điểm tin cậy > 65: Đòn bẩy trung bình (4x)
- Điểm tin cậy < 65: Đòn bẩy mặc định (3x)

### Stoploss động
- Với breakout: Stoploss đặt tại mức của Darvas Box
- Sau khi có lợi nhuận > 3%: Stoploss di chuyển lên/xuống theo VWAP
- ROI giảm dần theo thời gian: 10%, 5%, 2.5%, 1%

### Trailing Stop
- Trailing stop kích hoạt ở mức 3% lợi nhuận
- Trailing step 2%

## Thoát lệnh thông minh

Chiến lược sử dụng nhiều điều kiện thoát lệnh:

1. **Chạm mức Fibonacci**: Thoát khi giá chạm mức Fibonacci đã cấu hình
2. **Lệch VWAP quá xa**: Thoát khi độ lệch với VWAP vượt quá ngưỡng
3. **RSI quá cao/thấp**: Thoát khi RSI > 75 (long) hoặc < 25 (short)
4. **Đảo chiều VWAP**: Thoát khi giá cắt ngược qua VWAP sau một đà tăng/giảm

## Các tham số tối ưu hóa

- **darvas_lookback**: Số nến để tìm high/low trước đó (5-30, mặc định 15)
- **darvas_threshold**: % chênh lệch để xác nhận box đã bị phá vỡ (0.5-3.0, mặc định 1.0)
- **volume_threshold**: Ngưỡng % volume so với trung bình (150-300, mặc định 200)
- **vwap_period**: Số nến để tính VWAP (10-50, mặc định 20)
- **vwap_deviation**: % độ lệch cho phép so với VWAP (0.5-3.0, mặc định 1.5)
- **breakout_trend_period**: Số nến để xác định xu hướng (5-20, mặc định 10)
- **fib_retracement**: Mức Fibonacci để lấy lợi nhuận (0.236, 0.382, 0.5, 0.618, 0.786)

## Ưu điểm của chiến lược

1. **Đa dạng cơ hội**: Kết hợp hai phương pháp giao dịch trong một chiến lược
2. **Hệ thống lọc tín hiệu**: Đánh giá độ tin cậy chi tiết cho từng tín hiệu
3. **Quản lý rủi ro thông minh**: Stoploss động và trailing stop thích ứng
4. **Đòn bẩy động**: Tối ưu hóa đòn bẩy dựa trên chất lượng tín hiệu
5. **Tích hợp khối lượng**: Sử dụng volume làm yếu tố xác nhận quan trọng

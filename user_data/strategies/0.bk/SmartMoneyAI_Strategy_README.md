# Chiến lược Smart Money Concept kết hợp AI

Chiến lược này kết hợp các kỹ thuật phân tích Smart Money Concept (SMC) với mô hình AI để tạo ra hệ thống giao dịch có độ tin cậy cao.

## Tổng quan

Smart Money Concept là phương pháp phân tích giá dựa trên hành vi của các nhà giao dịch lớn ("smart money") trong thị trường. Phương pháp này tập trung vào việc xác định các khu vực giá mà các tổ chức lớn đang tham gia giao dịch, thông qua các mẫu hình Order Block, Fair Value Gap, và Equal High/Low.

Chiến lược này kết hợp SMC với AI để xác định chính xác điểm vào lệnh, với mỗi tín hiệu được đánh giá mức độ tin cậy từ 0-100.

## Các phương pháp phân tích chính

### 1. Order Blocks (OB)
- **Order Block Tăng (Bullish)**: Nến giảm mạnh ngay trước chuỗi tăng giá
- **Order Block Giảm (Bearish)**: Nến tăng mạnh ngay trước chuỗi giảm giá
- Các Order Block thường là điểm giao dịch tối ưu khi giá quay lại kiểm tra vùng này

### 2. Fair Value Gap (FVG)
- Khoảng trống giá không có giao dịch, thường xảy ra khi thị trường di chuyển nhanh
- **Bullish FVG**: Low của nến hiện tại > High của nến thứ 2 trước đó
- **Bearish FVG**: High của nến hiện tại < Low của nến thứ 2 trước đó
- FVG thường được lấp đầy khi giá quay lại vùng này

### 3. Cấu trúc thị trường (Market Structure)
- Phân tích chuỗi đỉnh/đáy (Higher Highs, Higher Lows, Lower Highs, Lower Lows)
- Xác định Break of Structure (BOS) và Change of Character (CHoCH)
- Xác định xu hướng tổng thể của thị trường

### 4. Equal Highs/Lows (EQH/EQL)
- Equal High: Khi giá tạo nhiều đỉnh bằng nhau
- Equal Low: Khi giá tạo nhiều đáy bằng nhau
- Thường được sử dụng để xác định các vùng tích lũy và breakout

### 5. Mẫu nến (Pinbar, Engulfing)
- Sử dụng mẫu nến Pinbar, Engulfing, và các mẫu nến đảo chiều khác
- Các mẫu nến này thường xác nhận lại Order Block và cấu trúc thị trường

### 6. Fibonacci Retracement
- Xác định các mức Fibonacci từ swing high/low gần nhất
- Giao dịch tại các mức quan trọng 0.5, 0.618, 0.786

## Hệ thống tính điểm tin cậy (0-100)

Điểm tin cậy được tính dựa trên sự kết hợp của:

### 1. Smart Money Factors (60%)
- Order Blocks (15%)
- Fair Value Gap (10%)
- Market Structure (15%)
- Candle Patterns (10%)
- Equal Highs/Lows (5%)
- Fibonacci Levels (5%)

### 2. Technical Indicators (40%)
- EMA Trend (10%)
- RSI (10%)
- MACD (10%)
- Bollinger Bands (10%)

## Quản lý vốn và rủi ro

### Đòn bẩy động
- Điểm tin cậy > 85: Sử dụng đòn bẩy tối đa (max 8x)
- Điểm tin cậy > 75: Sử dụng đòn bẩy 6x
- Điểm tin cậy > 65: Sử dụng đòn bẩy 4x
- Điểm tin cậy < 65: Sử dụng đòn bẩy mặc định (3x)

### Stoploss động
- Đặt dưới/trên Order Block gần nhất (tùy theo vị thế Long/Short)
- Điều chỉnh dựa trên mức lợi nhuận hiện tại (trailing stoploss)
- Lợi nhuận > 3%: Stoploss = -1%
- Lợi nhuận > 2%: Stoploss = -1.5%
- Lợi nhuận > 1%: Stoploss = -2%

## Lọc tín hiệu và xác nhận

- Chỉ vào lệnh khi điểm tin cậy vượt ngưỡng (mặc định 70/100)
- Sử dụng mô hình AI để xác nhận (nếu có) với ngưỡng xác suất thành công > 60%
- Không giao dịch ngược xu hướng (theo cấu trúc thị trường)

## Cách sử dụng

1. Đặt ngưỡng điểm tin cậy (score_threshold) tùy thuộc khẩu vị rủi ro
2. Huấn luyện mô hình AI (smart_money_model.joblib) để nâng cao độ chính xác
3. Tối ưu các tham số order_block_length, fvg_threshold, eqhl_bars, min_rsi, max_rsi

## Tối ưu hóa

Các tham số có thể tối ưu hóa:
- **order_block_length**: Độ dài của các Order Block (3-20, mặc định 5)
- **fvg_threshold**: Ngưỡng kích thước của Fair Value Gap (0.2-1.0, mặc định 0.5)
- **eqhl_bars**: Số nến để xác định Equal Highs/Lows (3-10, mặc định 5)
- **min_rsi**: Ngưỡng RSI thấp (10-40, mặc định 30)
- **max_rsi**: Ngưỡng RSI cao (60-90, mặc định 70)
- **score_threshold**: Ngưỡng điểm tin cậy (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. Kết hợp cả phân tích Smart Money (các tổ chức lớn) và phân tích kỹ thuật truyền thống
2. Hệ thống tính điểm tin cậy giúp lọc bỏ các tín hiệu yếu
3. Đòn bẩy động cho phép tận dụng tối đa các cơ hội chất lượng cao
4. Stoploss thông minh giúp bảo vệ vốn và tối ưu lợi nhuận
5. Tích hợp AI để nâng cao độ chính xác của dự đoán

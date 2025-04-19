# Chiến lược WaveTrend Multi-Timeframe

Chiến lược này sử dụng chỉ báo WaveTrend với phân tích đa khung thời gian (MTF - Multi-TimeFrame). Được tối ưu hóa cho khung thời gian 1 giờ (1h).

## Tổng quan

WaveTrend_MTF_Strategy là chiến lược tiên tiến kết hợp chỉ báo WaveTrend (được phát triển bởi LazyBear) với phân tích đa khung thời gian:

1. **WaveTrend**: Chỉ báo kỹ thuật mạnh mẽ kết hợp các khía cạnh của RSI, MACD và các chỉ báo dao động, hiệu quả trong việc xác định oversold/overbought và xác nhận xu hướng
2. **Multi-Timeframe Analysis**: Kết hợp tín hiệu giữa khung thời gian hiện tại (1h) và khung thời gian cao hơn (4h) để lọc bỏ nhiễu và tăng chất lượng tín hiệu
3. **Divergence Detection**: Phát hiện phân kỳ (divergence) giữa giá và WaveTrend để xác định các điểm đảo chiều tiềm năng

Chiến lược này đặc biệt hiệu quả trong việc xác định các điểm reversal và continuation có xác suất cao.

## Phương pháp giao dịch

### 1. WaveTrend Oscillator

- **Công thức cơ bản**: Kết hợp EMA và giá trị độ lệch chuẩn hóa
- **Chỉ số WaveTrend Line (WT1)**: Đường chính phản ánh momentum
- **Chỉ số WaveTrend Signal (WT2)**: Đường tín hiệu làm cơ sở cho các giao cắt
- **Vùng overbought/oversold**: >70 (overbought), <-70 (oversold)
- **Tham số tối ưu**: n1 (5-15, mặc định 10), n2 (15-30, mặc định 21)

### 2. Multi-Timeframe Analysis

- **Timeframe chính**: 1h (phân tích chính và tín hiệu giao dịch)
- **Timeframe cao hơn**: 4h (xác nhận và lọc tín hiệu)
- **Phương pháp kết hợp**: Tìm kiếm sự xác nhận từ cả hai khung thời gian, ưu tiên tín hiệu có sự đồng thuận

### 3. WaveTrend Divergence

- **Bullish Divergence**: Giá tạo đáy thấp hơn, WaveTrend tạo đáy cao hơn
- **Bearish Divergence**: Giá tạo đỉnh cao hơn, WaveTrend tạo đỉnh thấp hơn
- **Phát hiện tự động**: Sử dụng thuật toán phát hiện swing high/low và so sánh giữa giá và WaveTrend

## Tín hiệu giao dịch

### Tín hiệu Long

1. **WaveTrend Buy Setup**:
   - WaveTrend cross up từ vùng oversold trong 1h timeframe
   - WaveTrend oversold trong 4h timeframe (xác nhận)
   - WaveTrend vẫn ở vùng âm (<0)
   - Volume tốt

2. **WaveTrend Bullish Divergence**:
   - Phát hiện bullish divergence giữa giá và WaveTrend
   - Không trong downtrend mạnh
   - Volume tốt

3. **WaveTrend Oversold Bounce**:
   - WaveTrend trong vùng oversold (<-70)
   - WaveTrend đang tăng (cả WT1 và WT2)
   - RSI < 40
   - Volume khá tốt

4. **WaveTrend Cross Up trong vùng âm**:
   - WaveTrend cross up khi WT vẫn < 0
   - Trong uptrend (EMA xác nhận)
   - Volume tốt

5. **WaveTrend Bullish Setup + Price Action**:
   - WaveTrend bullish setup (cross up hoặc divergence)
   - Giá trên EMA ngắn
   - RSI > 45
   - Volume cao

### Tín hiệu Short

1. **WaveTrend Sell Setup**:
   - WaveTrend cross down từ vùng overbought trong 1h timeframe
   - WaveTrend overbought trong 4h timeframe (xác nhận)
   - WaveTrend vẫn ở vùng dương (>0)
   - Volume tốt

2. **WaveTrend Bearish Divergence**:
   - Phát hiện bearish divergence giữa giá và WaveTrend
   - Không trong uptrend mạnh
   - Volume tốt

3. **WaveTrend Overbought Bounce**:
   - WaveTrend trong vùng overbought (>70)
   - WaveTrend đang giảm (cả WT1 và WT2)
   - RSI > 60
   - Volume khá tốt

4. **WaveTrend Cross Down trong vùng dương**:
   - WaveTrend cross down khi WT vẫn > 0
   - Trong downtrend (EMA xác nhận)
   - Volume tốt

5. **WaveTrend Bearish Setup + Price Action**:
   - WaveTrend bearish setup (cross down hoặc divergence)
   - Giá dưới EMA ngắn
   - RSI < 55
   - Volume cao

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu WaveTrend:

#### Long:
- **WaveTrend oversold mạnh** (WT1 < -90): Đòn bẩy tối đa (5x)
- **WaveTrend bullish divergence**: Đòn bẩy cao (4x)
- **WaveTrend buy setup từ oversold** (WT1 < -70): Đòn bẩy cao (4x)
- **WaveTrend cross up + higher TF confirmation**: Đòn bẩy cao (4x)
- **WaveTrend oversold vừa phải** (WT1 < -60): Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

#### Short:
- **WaveTrend overbought mạnh** (WT1 > 90): Đòn bẩy tối đa (5x)
- **WaveTrend bearish divergence**: Đòn bẩy cao (4x)
- **WaveTrend sell setup từ overbought** (WT1 > 70): Đòn bẩy cao (4x)
- **WaveTrend cross down + higher TF confirmation**: Đòn bẩy cao (4x)
- **WaveTrend overbought vừa phải** (WT1 > 60): Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -5%
- **Với lợi nhuận > 2%**:
  - Long:
    - WaveTrend vào vùng overbought: -1%
    - Giá trên EMA nhanh: Stoploss tại EMA nhanh (max -3%)
    - Lợi nhuận > 5%: -1% để bảo toàn lợi nhuận
  - Short:
    - WaveTrend vào vùng oversold: -1%
    - Giá dưới EMA nhanh: Stoploss tại EMA nhanh (max -3%)
    - Lợi nhuận > 5%: -1% để bảo toàn lợi nhuận

### ROI (Return on Investment)

- **0 giờ**: 5% lợi nhuận
- **24 giờ**: 3.5% lợi nhuận
- **48 giờ**: 2.5% lợi nhuận
- **72 giờ**: 1.5% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **WaveTrend Cross Down từ overbought**:
   - WT1 đang trong vùng overbought (>70)
   - WaveTrend vừa cắt xuống

2. **WaveTrend Extreme Overbought**:
   - WT1 > 100 (extreme overbought)
   - WT1 đang giảm

3. **WaveTrend + RSI Overbought**:
   - WT1 > 70 và đang giảm
   - RSI > 70 và đang giảm

4. **WaveTrend Bearish Divergence**:
   - Phát hiện bearish divergence

5. **Break down from EMA**:
   - Giá từ trên EMA nhanh cắt xuống dưới
   - WT1 đã vào vùng âm (<0)

### Thoát Short

1. **WaveTrend Cross Up từ oversold**:
   - WT1 đang trong vùng oversold (<-70)
   - WaveTrend vừa cắt lên

2. **WaveTrend Extreme Oversold**:
   - WT1 < -100 (extreme oversold)
   - WT1 đang tăng

3. **WaveTrend + RSI Oversold**:
   - WT1 < -70 và đang tăng
   - RSI < 30 và đang tăng

4. **WaveTrend Bullish Divergence**:
   - Phát hiện bullish divergence

5. **Break up from EMA**:
   - Giá từ dưới EMA nhanh cắt lên trên
   - WT1 đã vào vùng dương (>0)

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **RSI extreme**: Không long khi RSI > 75, không short khi RSI < 25
3. **WaveTrend extreme**: Không long khi WT1 > 80, không short khi WT1 < -80
4. **Higher timeframe conflict**: Không long khi WT1 khung cao hơn > 70, không short khi WT1 khung cao hơn < -70

## Tham số tối ưu hóa

### Tham số WaveTrend
- **wt_n1**: Chu kỳ đầu tiên (5-15, mặc định 10)
- **wt_n2**: Chu kỳ thứ hai (15-30, mặc định 21)
- **wt_overbought**: Ngưỡng overbought (60-80, mặc định 70)
- **wt_oversold**: Ngưỡng oversold (-80 đến -60, mặc định -70)

### Tham số EMA
- **ema1**: EMA nhanh (5-30, mặc định 20)
- **ema2**: EMA chậm (30-100, mặc định 50)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)

## Ưu điểm của chiến lược

1. **Chỉ báo mạnh mẽ**: WaveTrend là chỉ báo kết hợp nhiều khía cạnh của các chỉ báo truyền thống, cho tín hiệu chất lượng
2. **Lọc nhiễu hiệu quả**: Phân tích đa khung thời gian giúp giảm tín hiệu giả
3. **Phát hiện đảo chiều**: Hiệu quả trong việc xác định các điểm đảo chiều
4. **Quản lý rủi ro linh hoạt**: Stoploss động thích nghi với điều kiện thị trường

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 1 giờ
- Biến động trung bình đến cao
- Xu hướng rõ ràng
- Cần xác định các điểm đảo chiều có xác suất cao

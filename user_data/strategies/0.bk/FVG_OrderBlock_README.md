# Chiến lược FVG (Fair Value Gap) + Order Block

Chiến lược này kết hợp phân tích Fair Value Gap và Order Block. Được tối ưu hóa cho khung thời gian 15 phút (15m).

## Tổng quan

FVG_OrderBlock_Strategy là chiến lược tinh vi dựa trên các khái niệm Smart Money Concept (SMC):

1. **Fair Value Gap (FVG)**: Khoảng trống giữa các nến không chồng chéo, thể hiện sự mất cân bằng cung cầu mà thị trường thường quay lại để "lấp đầy"
2. **Order Block (OB)**: Khu vực nơi các nhà giao dịch lớn (smart money) đặt lệnh, thường đóng vai trò là vùng hỗ trợ/kháng cự mạnh
3. **Kết hợp phân tích kỹ thuật**: RSI, EMA, và volume để xác nhận tín hiệu

Chiến lược này thích hợp cho giao dịch futures scalping, tận dụng những động thái nhỏ nhưng có xác suất cao của thị trường.

## Phương pháp giao dịch

### 1. Fair Value Gap (FVG)

- **Bullish FVG**: Khoảng trống giữa nến thứ nhất và nến thứ ba, khi low của nến thứ nhất > high của nến thứ hai, và nến thứ ba tăng
- **Bearish FVG**: Khoảng trống giữa nến thứ nhất và nến thứ ba, khi high của nến thứ nhất < low của nến thứ hai, và nến thứ ba giảm
- **FVG Mitigation**: Khi giá quay trở lại vùng FVG, tạo cơ hội giao dịch
- **Tham số tối ưu**: min_gap (0.1-0.5%, mặc định 0.2%), atr_multiplier (0.3-1.0, mặc định 0.5)

### 2. Order Block (OB)

- **Bullish Order Block**: Nến giảm xuất hiện trước chuỗi tăng mạnh, thường đóng vai trò là vùng hỗ trợ
- **Bearish Order Block**: Nến tăng xuất hiện trước chuỗi giảm mạnh, thường đóng vai trò là vùng kháng cự
- **OB Revisit**: Khi giá quay trở lại vùng Order Block, tạo cơ hội giao dịch
- **Tham số tối ưu**: strength (0.4-0.8, mặc định 0.6), atr_multiplier (1.0-2.0, mặc định 1.5)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Bullish FVG Mitigation + RSI thấp**:
   - Giá quay lại vùng Bullish FVG
   - RSI < 40 (gần oversold)
   - Không trong downtrend mạnh
   - Volume tốt

2. **Bullish Order Block Revisit + RSI thấp**:
   - Giá quay lại vùng Bullish Order Block
   - RSI < 45
   - Không trong downtrend mạnh
   - Volume tốt

3. **Bullish FVG+OB Zone + Bullish candle + Uptrend**:
   - Giá trong vùng Bullish FVG hoặc OB
   - Nến xanh (close > open)
   - Trong uptrend (EMAs xác nhận)
   - Volume ít nhất 80% trung bình

4. **Bullish FVG+OB Zone + RSI oversold bounce**:
   - Giá trong vùng Bullish FVG hoặc OB
   - RSI < 30 (oversold)
   - RSI đang tăng (bouncing)
   - Volume tốt

5. **Bullish setup + Break above EMA**:
   - Bullish setup (FVG/OB + RSI + không downtrend)
   - Giá vừa break lên trên EMA ngắn
   - Volume cao (>120% trung bình)

### Tín hiệu Short

1. **Bearish FVG Mitigation + RSI cao**:
   - Giá quay lại vùng Bearish FVG
   - RSI > 60 (gần overbought)
   - Không trong uptrend mạnh
   - Volume tốt

2. **Bearish Order Block Revisit + RSI cao**:
   - Giá quay lại vùng Bearish Order Block
   - RSI > 55
   - Không trong uptrend mạnh
   - Volume tốt

3. **Bearish FVG+OB Zone + Bearish candle + Downtrend**:
   - Giá trong vùng Bearish FVG hoặc OB
   - Nến đỏ (close < open)
   - Trong downtrend (EMAs xác nhận)
   - Volume ít nhất 80% trung bình

4. **Bearish FVG+OB Zone + RSI overbought drop**:
   - Giá trong vùng Bearish FVG hoặc OB
   - RSI > 70 (overbought)
   - RSI đang giảm (dropping)
   - Volume tốt

5. **Bearish setup + Break below EMA**:
   - Bearish setup (FVG/OB + RSI + không uptrend)
   - Giá vừa break xuống dưới EMA ngắn
   - Volume cao (>120% trung bình)

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu FVG và Order Block:

#### Long:
- **FVG mitigation + OB revisit + RSI thấp**: Đòn bẩy tối đa (6x)
- **FVG mitigation/OB revisit + RSI thấp + Uptrend**: Đòn bẩy cao (5x)
- **Bullish FVG+OB zone + Không downtrend**: Đòn bẩy trên trung bình (4x)
- **Bullish setup đơn thuần**: Đòn bẩy trung bình (3x)

#### Short:
- **FVG mitigation + OB revisit + RSI cao**: Đòn bẩy tối đa (6x)
- **FVG mitigation/OB revisit + RSI cao + Downtrend**: Đòn bẩy cao (5x)
- **Bearish FVG+OB zone + Không uptrend**: Đòn bẩy trên trung bình (4x)
- **Bearish setup đơn thuần**: Đòn bẩy trung bình (3x)

### Stoploss động

- **Stoploss mặc định**: -3%
- **Với lợi nhuận > 1%**:
  - Long: Stoploss tại vùng Bullish FVG/OB gần nhất, hoặc tại EMA ngắn
  - Short: Stoploss tại vùng Bearish FVG/OB gần nhất, hoặc tại EMA ngắn
  - Không quá -5% trong mọi trường hợp

### ROI (Return on Investment)

- **0 phút**: 3% lợi nhuận
- **30 phút**: 2% lợi nhuận
- **60 phút**: 1% lợi nhuận
- **90 phút**: 0.5% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **Bearish FVG Created**:
   - Xuất hiện FVG bearish mới

2. **Bearish Order Block Created**:
   - Xuất hiện Order Block bearish mới

3. **RSI Overbought**:
   - RSI > 70 và đang giảm

4. **Break below EMA trong uptrend**:
   - Giá từ trên EMA ngắn cắt xuống dưới
   - Nến trước đó vẫn trong uptrend

5. **Price hits bearish FVG/OB zone**:
   - Giá chạm vùng FVG/OB bearish

### Thoát Short

1. **Bullish FVG Created**:
   - Xuất hiện FVG bullish mới

2. **Bullish Order Block Created**:
   - Xuất hiện Order Block bullish mới

3. **RSI Oversold**:
   - RSI < 30 và đang tăng

4. **Break above EMA trong downtrend**:
   - Giá từ dưới EMA ngắn cắt lên trên
   - Nến trước đó vẫn trong downtrend

5. **Price hits bullish FVG/OB zone**:
   - Giá chạm vùng FVG/OB bullish

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **Biến động quá cao**: Không giao dịch khi BB width > 0.1 (biến động cao)
3. **Khoảng cách entry**: Không giao dịch khi giá entry cách giá đóng cửa > 1%
4. **RSI Extreme**: Không long khi RSI > 70, không short khi RSI < 30

## Tham số tối ưu hóa

### Tham số FVG
- **fvg_min_gap**: Kích thước tối thiểu của gap (0.1-0.5%, mặc định 0.2%)
- **fvg_atr_multiplier**: Hệ số ATR để xác định gap (0.3-1.0, mặc định 0.5)

### Tham số Order Block
- **ob_strength**: Ngưỡng độ mạnh của nến (0.4-0.8, mặc định 0.6)
- **ob_atr_multiplier**: Hệ số ATR để xác định large move (1.0-2.0, mặc định 1.5)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_ob_threshold**: Ngưỡng overbought (60-80, mặc định 70)
- **rsi_os_threshold**: Ngưỡng oversold (20-40, mặc định 30)

### Tham số EMAs
- **ema_short**: EMA nhanh (5-30, mặc định 8)
- **ema_medium**: EMA trung bình (10-50, mặc định 21)

## Ưu điểm của chiến lược

1. **Dựa trên Smart Money Concept**: Khai thác các nguyên tắc SMC đã được chứng minh hiệu quả trong thực tế
2. **Độ chính xác cao**: Các tín hiệu FVG và OB có xác suất thành công cao hơn chỉ báo truyền thống
3. **Phù hợp cho scalping**: ROI ngắn (0.5-3%) lý tưởng cho scalping trong thị trường sideway và trending
4. **Đòn bẩy thông minh**: Điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu giúp tối ưu hóa lợi nhuận
5. **Quản lý rủi ro hiệu quả**: Stoploss động dựa trên các vùng FVG/OB thực tế giúp giảm thiểu rủi ro

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 15 phút
- Biến động trung bình (không quá volatility)
- Tạo thành các mô hình FVG và OB rõ ràng
- Volume đủ lớn để xác định sự hiện diện của smart money

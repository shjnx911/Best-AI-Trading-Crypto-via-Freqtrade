# Chiến lược Double SMA + ATR Trailing

Chiến lược này kết hợp hai đường trung bình động đơn giản (SMA) với hệ thống ATR Trailing Stop. Được tối ưu cho khung thời gian 4 giờ (4h).

## Tổng quan

DoubleSMA_ATR_Trailing Strategy là chiến lược giao dịch kết hợp những công cụ kinh điển và hiệu quả:

1. **Double SMA Crossover**: Hệ thống giao dịch kinh điển dùng hai đường trung bình động để xác định xu hướng và điểm vào lệnh
2. **ATR Trailing Stop**: Hệ thống thoát lệnh động dựa trên biến động giá thực tế, tối ưu hóa điểm thoát
3. **ADX Filter**: Lọc tín hiệu dựa trên chỉ báo Directional Movement để chỉ giao dịch trong xu hướng mạnh

Chiến lược này phù hợp với khung thời gian 4h, tận dụng biến động giá trong thị trường có xu hướng rõ ràng.

## Phương pháp giao dịch

### 1. Double SMA Crossover

- **Short SMA**: SMA ngắn (mặc định 20 nến) phản ứng nhanh với thay đổi giá
- **Long SMA**: SMA dài (mặc định 50 nến) xác định xu hướng
- **Tín hiệu Long**: Khi SMA ngắn cắt lên trên SMA dài
- **Tín hiệu Short**: Khi SMA ngắn cắt xuống dưới SMA dài
- **Tham số tối ưu**: sma_fast_length (5-50, mặc định 20), sma_slow_length (30-200, mặc định 50)

### 2. ATR Trailing Stop

- **Công thức Long**: Trailing Stop = Price - (ATR * Multiplier)
- **Công thức Short**: Trailing Stop = Price + (ATR * Multiplier)
- **Đặc tính**: Chỉ di chuyển theo hướng có lợi, không di chuyển ngược lại
- **Tham số tối ưu**: atr_period (10-30, mặc định 14), atr_multiplier_long/short (1.0-4.0, mặc định 2.0)

### 3. ADX Trend Strength

- **ADX**: Đo lường sức mạnh của xu hướng
- **+DI & -DI**: Xác định hướng của xu hướng
- **Tham số tối ưu**: adx_period (7-30, mặc định 14), adx_threshold (15-35, mặc định 25)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **SMA Crossover + Strong ADX**:
   - SMA ngắn cắt lên trên SMA dài
   - ADX > ngưỡng (mặc định 25)
   - Volume > ngưỡng trung bình

2. **Pullback trong Uptrend**:
   - Giá pullback về chạm SMA ngắn trong uptrend
   - ADX cùng chiều (bullish)
   - Giá đóng cửa trên SMA ngắn
   - Volume ổn định

3. **Bullish ADX Crossover**:
   - Trong uptrend (+DI > -DI)
   - +DI cắt lên trên -DI
   - ADX > 20
   - Volume tăng

4. **RSI Oversold Bounce trong Uptrend**:
   - Trong uptrend
   - RSI < 40 và đang tăng
   - Nến xanh (close > open)
   - Volume tốt

5. **Bollinger Band Bounce trong Uptrend**:
   - Trong uptrend
   - Giá chạm BB dưới
   - Nến xanh (close > open)
   - Volume tăng

### Tín hiệu Short

1. **SMA Crossover + Strong ADX**:
   - SMA ngắn cắt xuống dưới SMA dài
   - ADX > ngưỡng (mặc định 25)
   - Volume > ngưỡng trung bình

2. **Pullback trong Downtrend**:
   - Giá pullback lên chạm SMA ngắn trong downtrend
   - ADX cùng chiều (bearish)
   - Giá đóng cửa dưới SMA ngắn
   - Volume ổn định

3. **Bearish ADX Crossover**:
   - Trong downtrend (-DI > +DI)
   - -DI cắt lên trên +DI
   - ADX > 20
   - Volume tăng

4. **RSI Overbought Drop trong Downtrend**:
   - Trong downtrend
   - RSI > 60 và đang giảm
   - Nến đỏ (close < open)
   - Volume tốt

5. **Bollinger Band Bounce trong Downtrend**:
   - Trong downtrend
   - Giá chạm BB trên
   - Nến đỏ (close < open)
   - Volume tăng

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ mạnh của xu hướng:

#### Long/Short:
- **ADX rất mạnh** (ADX > threshold + 15) và cùng chiều: Đòn bẩy tối đa (5x)
- **ADX mạnh** (ADX > threshold + 5) và cùng chiều: Đòn bẩy cao (4x)
- **ADX vừa phải** (ADX > threshold) và cùng chiều: Đòn bẩy trung bình (3x)
- **Xu hướng không rõ ràng**: Đòn bẩy thấp (2x)

### Stoploss động

- **Stoploss mặc định**: -5%
- **ATR Trailing Stop**: Tự động điều chỉnh stoploss theo biến động giá
- Công thức:
  - **Long**: Stop = Close - (ATR * Multiplier)
  - **Short**: Stop = Close + (ATR * Multiplier)
- Chỉ di chuyển theo hướng có lợi (tăng cho long, giảm cho short)

### ROI (Return on Investment)

- **0 giờ**: 8% lợi nhuận
- **48 giờ**: 5% lợi nhuận
- **120 giờ**: 3% lợi nhuận
- **240 giờ**: 2% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **ATR Trailing Stop**: Khi giá chạm xuống dưới ATR trailing stop
2. **SMA Cross Down**: Khi SMA ngắn cắt xuống dưới SMA dài
3. **ADX Trend Change**: Khi xu hướng ADX chuyển từ bullish sang bearish
4. **RSI Overbought + Bearish Divergence**: RSI > 70 và giảm trong khi giá tăng
5. **BB Upper Rejection**: Giá chạm BB trên và có rejection mạnh

### Thoát Short

1. **ATR Trailing Stop**: Khi giá chạm lên trên ATR trailing stop
2. **SMA Cross Up**: Khi SMA ngắn cắt lên trên SMA dài
3. **ADX Trend Change**: Khi xu hướng ADX chuyển từ bearish sang bullish
4. **RSI Oversold + Bullish Divergence**: RSI < 30 và tăng trong khi giá giảm
5. **BB Lower Rejection**: Giá chạm BB dưới và có rejection mạnh

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch trong điều kiện không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 50% trung bình
2. **Biến động quá cao**: Không giao dịch khi BB width > 0.1 (biến động cao)
3. **Gap quá lớn**: Không giao dịch khi giá vào lệnh > 3% so với giá đóng cửa gần nhất
4. **Trend Strength**: Ưu tiên giao dịch khi ADX > 25 (xu hướng rõ ràng)

## Tham số tối ưu hóa

### Tham số SMA
- **sma_fast_length**: SMA ngắn (5-50, mặc định 20)
- **sma_slow_length**: SMA dài (30-200, mặc định 50)

### Tham số ATR
- **atr_period**: Chu kỳ tính ATR (10-30, mặc định 14)
- **atr_multiplier_long**: Hệ số ATR cho long (1.0-4.0, mặc định 2.0)
- **atr_multiplier_short**: Hệ số ATR cho short (1.0-4.0, mặc định 2.0)

### Tham số ADX
- **adx_period**: Chu kỳ tính ADX (7-30, mặc định 14)
- **adx_threshold**: Ngưỡng ADX xác định xu hướng mạnh (15-35, mặc định 25)

### Tham số Volume
- **volume_threshold**: Hệ số volume tối thiểu (0.5-3.0, mặc định 1.5)

## Ưu điểm của chiến lược

1. **Phương pháp kinh điển**: Dựa trên những công cụ phân tích kỹ thuật đã được chứng minh hiệu quả
2. **Quản lý rủi ro chặt chẽ**: ATR trailing stop tự động điều chỉnh theo biến động thị trường
3. **Lọc trend mạnh**: ADX giúp tập trung vào những xu hướng rõ ràng
4. **Đa dạng tín hiệu**: Nhiều phương pháp xác định điểm vào lệnh, phù hợp với nhiều điều kiện thị trường

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 4 giờ
- Xu hướng rõ ràng và ổn định
- Không quá nhiều nhiễu trong dao động giá
- Khối lượng giao dịch trung bình đến cao

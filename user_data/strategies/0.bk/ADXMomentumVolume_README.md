# Chiến lược ADX Momentum Volume

Chiến lược này kết hợp Average Directional Index (ADX) với các chỉ báo momentum và phân tích khối lượng. Được tối ưu hóa cho khung thời gian 15 phút (15m).

## Tổng quan

ADXMomentumVolume Strategy là chiến lược giao dịch theo xu hướng, sử dụng kết hợp ba thành phần chính:

1. **ADX/DMI**: Xác định sức mạnh xu hướng và hướng của xu hướng hiện tại
2. **Momentum**: Sử dụng nhiều chỉ báo momentum (RSI, TSI, MFI) để xác nhận tín hiệu
3. **Volume**: Phân tích khối lượng giao dịch để xác nhận tín hiệu và tránh các tín hiệu giả

Chiến lược này đặc biệt hiệu quả trong việc phát hiện các xu hướng mạnh và tận dụng các đợt pullback để vào lệnh.

## Phương pháp giao dịch

### 1. ADX (Average Directional Index)

- **Mục đích**: Đo lường sức mạnh của xu hướng, bất kể hướng xu hướng
- **Giá trị**: ADX > 25 cho thấy xu hướng mạnh, ADX > 40 cho thấy xu hướng rất mạnh
- **Ứng dụng**: Chỉ giao dịch khi ADX > ngưỡng (25 mặc định) để đảm bảo xu hướng đủ mạnh
- **Tham số tối ưu**: adx_length (10-30, mặc định 14), adx_threshold (20-40, mặc định 25)

### 2. DMI (Directional Movement Index)

- **Mục đích**: Xác định hướng của xu hướng (tăng hoặc giảm)
- **Thành phần**: +DI (positive directional indicator) và -DI (negative directional indicator)
- **Tín hiệu**:
  - Khi +DI > -DI, có xu hướng tăng
  - Khi -DI > +DI, có xu hướng giảm
  - Khi +DI cắt lên trên -DI, xu hướng tăng bắt đầu
  - Khi -DI cắt lên trên +DI, xu hướng giảm bắt đầu
- **Tham số tối ưu**: dmi_length (10-30, mặc định 14), dmi_threshold (0.1-0.5, mặc định 0.2)

### 3. Momentum Indicators

#### 3.1. RSI (Relative Strength Index)

- Đo lường tốc độ và sự thay đổi của giá
- Phát hiện vùng quá mua (>70) và quá bán (<30)
- Tham số: rsi_length (7-21, mặc định 14)

#### 3.2. TSI (True Strength Index)

- Chỉ báo momentum hai lần làm mịn
- Phát hiện đảo chiều sớm và tín hiệu cross
- Tham số: tsi_long (8-25, mặc định 13), tsi_short (1-7, mặc định 2), tsi_signal (7-15, mặc định 9)

#### 3.3. MFI (Money Flow Index)

- Kết hợp cả giá và khối lượng
- Phát hiện vùng quá mua/quá bán với xác nhận khối lượng

### 4. Volume Analysis

- **Volume ratio**: So sánh khối lượng hiện tại với trung bình động
- **OBV (On-Balance Volume)**: Xác nhận xu hướng giá bằng xu hướng khối lượng
- **Tham số tối ưu**: volume_ma_length (10-50, mặc định 20), volume_threshold (1.0-3.0, mặc định 1.5)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **RSI oversold + ADX strong + Uptrend**:
   - RSI < ngưỡng mua (30 mặc định)
   - RSI đang tăng
   - ADX > ngưỡng (25 mặc định)
   - +DI > -DI
   - Volume cao

2. **TSI cross up + Volume surge**:
   - TSI cắt lên trên Signal line
   - TSI < 0 (đang từ vùng âm đi lên)
   - +DI > -DI
   - Volume cao

3. **DMI cross + MFI < 40**:
   - +DI vừa cắt lên trên -DI
   - MFI < 40 (không quá cao)
   - ADX > ngưỡng
   - Volume cao

4. **OBV trend confirmation**:
   - OBV > OBV MA
   - +DI > -DI và ADX > ngưỡng
   - RSI < 60 (không quá cao)
   - Volume cao

### Tín hiệu Short

1. **RSI overbought + ADX strong + Downtrend**:
   - RSI > ngưỡng bán (70 mặc định)
   - RSI đang giảm
   - ADX > ngưỡng (25 mặc định)
   - -DI > +DI
   - Volume cao

2. **TSI cross down + Volume surge**:
   - TSI cắt xuống dưới Signal line
   - TSI > 0 (đang từ vùng dương đi xuống)
   - -DI > +DI
   - Volume cao

3. **DMI cross + MFI > 60**:
   - -DI vừa cắt lên trên +DI
   - MFI > 60 (không quá thấp)
   - ADX > ngưỡng
   - Volume cao

4. **OBV trend confirmation**:
   - OBV < OBV MA
   - -DI > +DI và ADX > ngưỡng
   - RSI > 40 (không quá thấp)
   - Volume cao

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ mạnh của xu hướng (ADX), chênh lệch DI và khối lượng:

#### Long:
- **Tối ưu** (ADX > 40, +DI > -DI*1.5, Volume ratio > 2.0): Đòn bẩy tối đa (6x)
- **Tốt** (ADX > 30, +DI > -DI, Volume ratio > 1.5): Đòn bẩy cao (5x)
- **Khá** (ADX > ngưỡng, +DI > -DI): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

#### Short:
- **Tối ưu** (ADX > 40, -DI > +DI*1.5, Volume ratio > 2.0): Đòn bẩy tối đa (6x)
- **Tốt** (ADX > 30, -DI > +DI, Volume ratio > 1.5): Đòn bẩy cao (5x)
- **Khá** (ADX > ngưỡng, -DI > +DI): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -2.8%
- **Với lợi nhuận > 1.5%**: Stoploss di chuyển lên/xuống theo EMA 21
- **Trailing stop**: Kích hoạt ở mức 1% lợi nhuận, step 0.8%

### ROI (Return on Investment)

- **0 phút**: 4% lợi nhuận
- **30 phút**: 3% lợi nhuận
- **60 phút**: 2% lợi nhuận
- **90 phút**: 1% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume ratio < 0.8
2. **ADX yếu**: Không giao dịch khi ADX < 20
3. **DI mâu thuẫn**: Không long khi +DI < -DI, không short khi -DI < +DI
4. **Tín hiệu mâu thuẫn**: Kết hợp nhiều indicator để lọc tín hiệu không rõ ràng

## Thoát lệnh thông minh

### Thoát Long

1. **ADX giảm mạnh**: Khi ADX giảm xuống dưới ngưỡng
2. **DMI bearish cross**: Khi +DI cắt xuống dưới -DI
3. **RSI overbought**: Khi RSI > ngưỡng bán (70)
4. **TSI cross down**: Khi TSI cắt xuống dưới Signal và TSI > 0.2
5. **EMA cross**: Khi EMA 8 cắt xuống dưới EMA 21

### Thoát Short

1. **ADX giảm mạnh**: Khi ADX giảm xuống dưới ngưỡng
2. **DMI bullish cross**: Khi -DI cắt xuống dưới +DI
3. **RSI oversold**: Khi RSI < ngưỡng mua (30)
4. **TSI cross up**: Khi TSI cắt lên trên Signal và TSI < -0.2
5. **EMA cross**: Khi EMA 8 cắt lên trên EMA 21

## Tham số tối ưu hóa

### Tham số ADX
- **adx_length**: Chu kỳ tính ADX (10-30, mặc định 14)
- **adx_threshold**: Ngưỡng ADX để xác định xu hướng mạnh (20-40, mặc định 25)

### Tham số DMI
- **dmi_length**: Chu kỳ tính DMI (10-30, mặc định 14)
- **dmi_threshold**: Ngưỡng chênh lệch DI (0.1-0.5, mặc định 0.2)

### Tham số Momentum
- **rsi_length**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)
- **tsi_long**: Chu kỳ dài cho TSI (8-25, mặc định 13)
- **tsi_short**: Chu kỳ ngắn cho TSI (1-7, mặc định 2)
- **tsi_signal**: Chu kỳ signal cho TSI (7-15, mặc định 9)

### Tham số Volume
- **volume_ma_length**: Chu kỳ MA volume (10-50, mặc định 20)
- **volume_threshold**: Ngưỡng volume ratio (1.0-3.0, mặc định 1.5)

## Ưu điểm của chiến lược

1. **Lọc xu hướng chặt chẽ**: Chỉ giao dịch trong xu hướng mạnh được xác định bởi ADX
2. **Xác nhận đa chỉ báo**: Kết hợp nhiều loại indicator để xác nhận tín hiệu
3. **Quản lý rủi ro thích ứng**: Đòn bẩy và stoploss tùy chỉnh theo độ mạnh của tín hiệu
4. **Tích hợp volume**: Dùng khối lượng giao dịch để xác nhận tín hiệu

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 15 phút
- Xu hướng mạnh (tránh thị trường sideway)
- Biến động vừa phải đến cao
- Khối lượng giao dịch ổn định

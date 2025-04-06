# Chiến lược Momentum Pinbar

Chiến lược này kết hợp phân tích Momentum với mô hình nến Pinbar. Được tối ưu hóa cho khung thời gian 4 giờ (4h).

## Tổng quan

MomentumPinbar_Strategy là chiến lược mạnh mẽ kết hợp phân tích mô hình nến Pinbar với xác nhận từ các chỉ báo Momentum:

1. **Pinbar (Pin Bar)**: Mô hình nến phản ánh đảo chiều tiềm năng, có đặc điểm là bóng dài và thân nhỏ
2. **Momentum Indicators**: Bao gồm MACD, RSI và Stochastic, dùng để xác nhận sức mạnh và hướng của động lượng giá
3. **Support/Resistance Analysis**: Xác định các vùng hỗ trợ/kháng cự quan trọng từ các swing high/low

Chiến lược nhằm mục đích xác định các điểm đảo chiều có xác suất cao dựa trên sự kết hợp của các mô hình nến đặc biệt và các chỉ báo Momentum.

## Phương pháp giao dịch

### 1. Phân tích Pinbar

- **Bullish Pinbar**: Thân nến nhỏ, bóng dưới dài (ít nhất 2x thân nến), xuất hiện trong downtrend hoặc tại vùng hỗ trợ
- **Bearish Pinbar**: Thân nến nhỏ, bóng trên dài (ít nhất 2x thân nến), xuất hiện trong uptrend hoặc tại vùng kháng cự
- **Tham số tối ưu**: body_pct (0.1-0.5, mặc định 0.3), wick_pct (1.0-3.0, mặc định 2.0)

### 2. Phân tích Momentum

- **MACD**: Xác định strength và direction của trend, phát hiện các điểm cross và divergence
- **RSI**: Xác định vùng oversold/overbought
- **Stochastic**: Xác nhận xu hướng và phát hiện tín hiệu giao cắt
- **Tham số tối ưu**: macd_fast (8-20, mặc định 12), macd_slow (20-40, mặc định 26)

### 3. Support/Resistance Analysis

- **Swing Detection**: Xác định các swing high/low
- **Level Grouping**: Gộp các mức giá gần nhau
- **Tham số tối ưu**: sr_window (10-30, mặc định 20), sr_tolerance (0.01-0.05, mặc định 0.02)

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Bullish Pinbar tại vùng hỗ trợ**:
   - Pinbar có thân nhỏ, bóng dưới dài
   - Xuất hiện tại hoặc gần vùng hỗ trợ (<1% khoảng cách)
   - Momentum dương (MACD, RSI, Stochastic)
   - RSI < 50
   - Volume tốt

2. **Bullish Reversal Pinbar with Momentum**:
   - Pinbar đảo chiều trong downtrend
   - MACD cross up
   - Volume tăng cao (>120% trung bình)

3. **Bullish Pinbar Momentum tại vùng giá thấp**:
   - Pinbar bullish với MACD, RSI, Stochastic bullish
   - RSI < 35 (gần oversold)
   - Volume >= 80% trung bình

4. **Bullish Pinbar bouncing from EMA**:
   - Pinbar bullish trong uptrend mạnh
   - Giá thấp nhất chạm hoặc gần EMA50
   - Đóng cửa trên EMA50
   - Volume tốt

5. **Bullish Pinbar at support with oversold RSI**:
   - Pinbar bullish
   - RSI < 30 (oversold)
   - Gần vùng hỗ trợ (<2%)
   - Volume tốt

### Tín hiệu Short

1. **Bearish Pinbar tại vùng kháng cự**:
   - Pinbar có thân nhỏ, bóng trên dài
   - Xuất hiện tại hoặc gần vùng kháng cự (<1% khoảng cách)
   - Momentum âm (MACD, RSI, Stochastic)
   - RSI > 50
   - Volume tốt

2. **Bearish Reversal Pinbar with Momentum**:
   - Pinbar đảo chiều trong uptrend
   - MACD cross down
   - Volume tăng cao (>120% trung bình)

3. **Bearish Pinbar Momentum tại vùng giá cao**:
   - Pinbar bearish với MACD, RSI, Stochastic bearish
   - RSI > 65 (gần overbought)
   - Volume >= 80% trung bình

4. **Bearish Pinbar bouncing from EMA**:
   - Pinbar bearish trong downtrend mạnh
   - Giá cao nhất chạm hoặc gần EMA50
   - Đóng cửa dưới EMA50
   - Volume tốt

5. **Bearish Pinbar at resistance with overbought RSI**:
   - Pinbar bearish
   - RSI > 70 (overbought)
   - Gần vùng kháng cự (<2%)
   - Volume tốt

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu Pinbar và Momentum:

#### Long:
- **Pinbar tại hỗ trợ mạnh + RSI thấp + Volume cao**: Đòn bẩy tối đa (5x)
- **Reversal Pinbar + MACD cross up**: Đòn bẩy cao (4x)
- **Pinbar trong uptrend mạnh**: Đòn bẩy cao (4x)
- **Pinbar với momentum trung bình**: Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

#### Short:
- **Pinbar tại kháng cự mạnh + RSI cao + Volume cao**: Đòn bẩy tối đa (5x)
- **Reversal Pinbar + MACD cross down**: Đòn bẩy cao (4x)
- **Pinbar trong downtrend mạnh**: Đòn bẩy cao (4x)
- **Pinbar với momentum trung bình**: Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -5%
- **Với lợi nhuận > 3%**:
  - Long: Stoploss tại vùng hỗ trợ gần nhất hoặc dựa trên ATR (2x ATR)
  - Short: Stoploss tại vùng kháng cự gần nhất hoặc dựa trên ATR (2x ATR)
  - Không quá -10% trong mọi trường hợp

### ROI (Return on Investment)

- **0 giờ**: 8% lợi nhuận
- **24 giờ**: 5% lợi nhuận
- **48 giờ**: 3% lợi nhuận
- **72 giờ**: 2% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **Bearish Pinbar**:
   - Xuất hiện pinbar đảo chiều giảm
   - Volume không quá thấp

2. **RSI Overbought**:
   - RSI > 70 và đang giảm

3. **MACD cross down**:
   - MACD cắt xuống dưới Signal line
   - MACD vẫn > 0 (vẫn trong vùng dương)

4. **Approaching Strong Resistance**:
   - Giá rất gần vùng kháng cự mạnh (<0.5%)
   - Volume tốt

5. **Break EMA trong uptrend**:
   - Giá từ trên EMA20 cắt xuống dưới
   - Nến trước đó vẫn trong uptrend

### Thoát Short

1. **Bullish Pinbar**:
   - Xuất hiện pinbar đảo chiều tăng
   - Volume không quá thấp

2. **RSI Oversold**:
   - RSI < 30 và đang tăng

3. **MACD cross up**:
   - MACD cắt lên trên Signal line
   - MACD vẫn < 0 (vẫn trong vùng âm)

4. **Approaching Strong Support**:
   - Giá rất gần vùng hỗ trợ mạnh (<0.5%)
   - Volume tốt

5. **Break EMA trong downtrend**:
   - Giá từ dưới EMA20 cắt lên trên
   - Nến trước đó vẫn trong downtrend

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **Biến động quá cao**: Không giao dịch khi BB width > 0.1 (biến động cao)
3. **Khoảng cách entry**: Không giao dịch khi giá entry cách giá đóng cửa > 1%
4. **RSI Extreme**: Không long khi RSI > 70, không short khi RSI < 30

## Tham số tối ưu hóa

### Tham số Pinbar
- **pinbar_body_pct**: Tỷ lệ thân/cả nến (0.1-0.5, mặc định 0.3)
- **pinbar_wick_pct**: Tỷ lệ bóng/thân (1.0-3.0, mặc định 2.0)

### Tham số Momentum
- **macd_fast**: Chu kỳ nhanh MACD (8-20, mặc định 12)
- **macd_slow**: Chu kỳ chậm MACD (20-40, mặc định 26)

### Tham số Support/Resistance
- **sr_window**: Cửa sổ xác định swing (10-30, mặc định 20)
- **sr_tolerance**: Độ chênh lệch cho phép (0.01-0.05, mặc định 0.02)

### Tham số Filter
- **min_dist_to_sr_pct**: Khoảng cách tối thiểu đến SR (0.5-3.0, mặc định 1.0)
- **rsi_oversold**: Ngưỡng RSI oversold (20-40, mặc định 30)
- **rsi_overbought**: Ngưỡng RSI overbought (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. **Xác định điểm đảo chiều**: Pinbar là một trong những mô hình nến đáng tin cậy nhất để xác định điểm đảo chiều
2. **Đa lớp xác nhận**: Kết hợp mô hình nến, momentum và vùng hỗ trợ/kháng cự
3. **Quản lý rủi ro hiệu quả**: Stoploss động dựa trên các mức hỗ trợ/kháng cự thực tế
4. **Thích ứng với nhiều điều kiện thị trường**: Các tín hiệu khác nhau cho các giai đoạn thị trường khác nhau

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 4 giờ
- Biến động trung bình đến cao
- Tạo thành các mô hình nến rõ ràng
- Có xu hướng thay vì sideway

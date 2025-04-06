# Chiến lược Triple EMA + VWAP

Chiến lược này kết hợp Triple Exponential Moving Average (TEMA) với Volume Weighted Average Price (VWAP) để tạo ra một hệ thống giao dịch vừa nhạy với xu hướng vừa có xác nhận từ khối lượng. Được tối ưu hóa cho khung thời gian 15 phút (15m).

## Tổng quan

TEMA_VWAP Strategy kết hợp hai công cụ phân tích kỹ thuật mạnh mẽ:

1. **Triple EMA (TEMA)**: Biến thể cải tiến của EMA giúp giảm lag và phản ứng nhanh hơn với thay đổi giá
2. **VWAP (Volume Weighted Average Price)**: Chỉ báo kết hợp giá và khối lượng, thường được các trader lớn sử dụng

Chiến lược này đánh giá các tương tác giữa giá, TEMA và VWAP để tìm điểm vào lệnh có xác suất cao, khai thác các xu hướng ngay khi chúng mới bắt đầu.

## Phương pháp giao dịch

### 1. Triple EMA (TEMA)

- **Công thức**: TEMA = 3 * EMA - 3 * EMA(EMA) + EMA(EMA(EMA))
- **Ưu điểm**: Phản ứng nhanh với thay đổi giá, giảm lag so với MA thông thường
- **Ứng dụng**: Sử dụng ba TEMA (fast, mid, slow) để xác định xu hướng
- **Tham số tối ưu**: fast (5-15, mặc định 8), mid (15-30, mặc định 21), slow (30-60, mặc định 50)

### 2. VWAP (Volume Weighted Average Price)

- **Công thức**: ∑(Price * Volume) / ∑(Volume)
- **Ưu điểm**: Kết hợp khối lượng giao dịch, phản ánh mức giá trung bình được giao dịch thực tế
- **Ứng dụng**: Làm mức hỗ trợ/kháng cự động, xác định điểm đảo chiều
- **Tham số tối ưu**: period (10-30, mặc định 20)

### 3. TEMA-VWAP Relationship

Chiến lược phân tích đặc biệt mối quan hệ giữa TEMA và VWAP:
- **Convergence**: Khi TEMA và VWAP hội tụ (rất gần nhau)
- **Cross**: Khi giá cắt qua VWAP
- **Bounce**: Khi giá chạm và nảy lên/xuống từ VWAP

## Tín hiệu giao dịch

### Tín hiệu Long

1. **TEMA Bullish + VWAP Cross**:
   - Hệ thống TEMA cho thấy xu hướng tăng (Fast > Mid > Slow TEMA)
   - Giá cắt lên trên VWAP hoặc TEMA Fast cắt lên trên TEMA Mid khi giá trên VWAP
   - RSI < 65 (không quá cao)
   - Volume ổn định

2. **VWAP Bounce trong Uptrend**:
   - Xu hướng tăng đã được xác nhận (Fast > Mid > Slow TEMA)
   - Giá pullback về chạm VWAP rồi bật lên
   - RSI không quá cao
   - Volume tốt

3. **TEMA-VWAP Convergence Bullish**:
   - TEMA Fast và VWAP hội tụ (rất gần nhau)
   - TEMA Fast đang tăng (độ dốc dương)
   - RSI trong vùng 40-60 (trung tính)
   - Volume tăng

4. **Strong Uptrend with RSI Reset**:
   - Xu hướng tăng đã xác định
   - Giá trên VWAP
   - RSI vừa từ vùng thấp (<40) và đang tăng
   - Volume tăng cao

### Tín hiệu Short

1. **TEMA Bearish + VWAP Cross**:
   - Hệ thống TEMA cho thấy xu hướng giảm (Fast < Mid < Slow TEMA)
   - Giá cắt xuống dưới VWAP hoặc TEMA Fast cắt xuống dưới TEMA Mid khi giá dưới VWAP
   - RSI > 35 (không quá thấp)
   - Volume ổn định

2. **VWAP Bounce trong Downtrend**:
   - Xu hướng giảm đã được xác nhận (Fast < Mid < Slow TEMA)
   - Giá pullback lên chạm VWAP rồi bật xuống
   - RSI không quá thấp
   - Volume tốt

3. **TEMA-VWAP Convergence Bearish**:
   - TEMA Fast và VWAP hội tụ (rất gần nhau)
   - TEMA Fast đang giảm (độ dốc âm)
   - RSI trong vùng 40-60 (trung tính)
   - Volume tăng

4. **Strong Downtrend with RSI Reset**:
   - Xu hướng giảm đã xác định
   - Giá dưới VWAP
   - RSI vừa từ vùng cao (>60) và đang giảm
   - Volume tăng cao

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên khoảng cách đến VWAP và chỉ số RSI:

#### Long:
- **Rất tốt** (Giá dưới VWAP > 0.5% + RSI < 40): Đòn bẩy tối đa (5x)
- **Khá tốt** (Giá dưới VWAP 0-0.5% + RSI < 65): Đòn bẩy cao (4x)
- **Không tốt lắm** (Giá trên VWAP 0-0.5% + RSI < 45): Đòn bẩy thấp (3x)
- **Không tốt** (Giá trên VWAP > 0.5% + RSI < 40): Đòn bẩy mặc định (3x)

#### Short:
- **Rất tốt** (Giá trên VWAP > 0.5% + RSI > 60): Đòn bẩy tối đa (5x)
- **Khá tốt** (Giá trên VWAP 0-0.5% + RSI > 55): Đòn bẩy cao (4x)
- **Không tốt lắm** (Giá dưới VWAP 0-0.5% + RSI > 35): Đòn bẩy thấp (3x)
- **Không tốt** (Giá dưới VWAP > 0.5% + RSI > 60): Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -3%
- **Với lợi nhuận > 1.5%**: Stoploss di chuyển lên/xuống theo TEMA mid hoặc VWAP (cái nào gần hơn)
- **Trailing stop**: Kích hoạt ở mức 2% lợi nhuận, step 1%

### ROI (Return on Investment)

- **0 phút**: 4% lợi nhuận
- **30 phút**: 3% lợi nhuận
- **60 phút**: 2% lợi nhuận
- **90 phút**: 1% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **RSI Extreme**: Không long khi RSI > 75, không short khi RSI < 25
3. **VWAP Distance**: Không giao dịch khi giá cách VWAP > 2% (quá xa)
4. **Trend Alignment**: Ưu tiên giao dịch theo xu hướng TEMA

## Thoát lệnh thông minh

### Thoát Long

1. **TEMA Fast cross down**: Khi TEMA Fast cắt xuống dưới TEMA Mid
2. **Cross below VWAP**: Khi giá từ trên cắt xuống dưới VWAP trong uptrend
3. **RSI overbought**: Khi RSI > 75 và đang giảm
4. **Volume spike với rejection**: Khi volume tăng vọt (>2.5x) với nến đỏ có bóng trên dài

### Thoát Short

1. **TEMA Fast cross up**: Khi TEMA Fast cắt lên trên TEMA Mid
2. **Cross above VWAP**: Khi giá từ dưới cắt lên trên VWAP trong downtrend
3. **RSI oversold**: Khi RSI < 25 và đang tăng
4. **Volume spike với rejection**: Khi volume tăng vọt (>2.5x) với nến xanh có bóng dưới dài

## Tham số tối ưu hóa

### Tham số Triple EMA
- **tema_fast**: TEMA fast period (5-15, mặc định 8)
- **tema_mid**: TEMA mid period (15-30, mặc định 21)
- **tema_slow**: TEMA slow period (30-60, mặc định 50)

### Tham số VWAP
- **vwap_period**: Chu kỳ tính VWAP (10-30, mặc định 20)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. **Phản ứng nhanh**: TEMA giúp phát hiện xu hướng sớm hơn các MA thông thường
2. **Xác nhận khối lượng**: VWAP kết hợp khối lượng giúp lọc tín hiệu hiệu quả
3. **Quản lý rủi ro thông minh**: Stoploss và đòn bẩy động dựa trên mối quan hệ giữa giá-TEMA-VWAP
4. **Lọc tín hiệu đa chiều**: Nhiều lớp lọc giúp loại bỏ tín hiệu yếu

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 15 phút
- Biến động trung bình đến cao
- Xu hướng rõ ràng
- Khối lượng giao dịch ổn định

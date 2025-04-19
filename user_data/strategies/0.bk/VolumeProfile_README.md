# Chiến lược Volume Profile

Chiến lược này dựa trên phân tích Volume Profile và POC (Point of Control). Được tối ưu hóa cho khung thời gian 4 giờ (4h).

## Tổng quan

VolumeProfile_Strategy áp dụng phương pháp phân tích thị trường tiên tiến dựa trên phân bố khối lượng giao dịch theo mức giá, được sử dụng rộng rãi bởi các nhà giao dịch chuyên nghiệp:

1. **Volume Profile**: Phân tích khối lượng giao dịch theo mức giá (không phải theo thời gian) để xác định các vùng giá quan trọng
2. **Point of Control (POC)**: Mức giá có khối lượng giao dịch cao nhất trong một phạm vi thời gian
3. **Value Area**: Vùng giá tập trung 70% khối lượng giao dịch, bao gồm VAH (Value Area High) và VAL (Value Area Low)

Chiến lược này hoạt động dựa trên nguyên tắc rằng giá thường xuyên quay trở lại các vùng giao dịch tích cực và tìm sự hỗ trợ/kháng cự tại các vùng này.

## Phương pháp giao dịch

### 1. Volume Profile Analysis

- **Point of Control (POC)**: Xác định mức giá có khối lượng giao dịch cao nhất
- **Value Area High (VAH)**: Giới hạn trên của vùng tập trung 70% khối lượng
- **Value Area Low (VAL)**: Giới hạn dưới của vùng tập trung 70% khối lượng
- **POC Strength**: Đo lường mức độ tập trung của khối lượng tại POC
- **Value Area Width**: Đo lường độ rộng của Value Area (% từ VAL đến VAH)

### 2. EMA & Stochastic

- **Triple EMA System**: Sử dụng 3 EMA (nhanh, trung bình, chậm) để xác định xu hướng
- **Stochastic Oscillator**: Xác định vùng quá mua/quá bán và các tín hiệu đảo chiều
- **Tham số tối ưu**: ema_fast (5-50), ema_medium (20-100), ema_slow (50-200)

### 3. S/R từ Volume Profile

- **POC Support/Resistance**: POC thường đóng vai trò là vùng hỗ trợ/kháng cự mạnh
- **Value Area Bounce**: Giá thường có phản ứng tại các rìa của Value Area (VAH & VAL)
- **Value Area Breakout**: Khi giá phá vỡ Value Area với khối lượng lớn, thường dẫn đến chuyển động mạnh

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Value Area Low Bounce**:
   - Giá chạm hoặc dưới VAL (±0.5%)
   - Nến xanh (close > open)
   - Giá đóng cửa trên VAL
   - Không trong downtrend mạnh
   - Stochastic < 30 và đang tăng
   - Value Area đủ rộng (> 3%)

2. **Approaching POC from Below**:
   - Giá đang ở gần dưới POC (trong khoảng 1%)
   - Đang di chuyển lên gần POC
   - Không trong downtrend mạnh
   - Stochastic < 50 và K > D
   - RSI > 40

3. **POC Support**:
   - Giá chạm POC từ trên xuống
   - Đóng cửa trên POC
   - Trong uptrend (EMA fast > medium > slow)
   - POC đủ mạnh (volume trên trung bình)
   - Stochastic < 60

4. **Value Area Breakout Bullish**:
   - Giá đóng cửa trên VAH
   - Nến trước đóng dưới VAH
   - Value Area đủ rộng (> 3%)
   - Volume > 150% trung bình
   - Stochastic > 40 và đang tăng

5. **Strong Uptrend + Expanding VA**:
   - Trong uptrend (EMA fast > medium > slow)
   - VAH và POC đang tăng
   - Giá trên POC
   - Stochastic 40-70
   - Volume > 120% trung bình

### Tín hiệu Short

1. **Value Area High Bounce**:
   - Giá chạm hoặc trên VAH (±0.5%)
   - Nến đỏ (close < open)
   - Giá đóng cửa dưới VAH
   - Không trong uptrend mạnh
   - Stochastic > 70 và đang giảm
   - Value Area đủ rộng (> 3%)

2. **Approaching POC from Above**:
   - Giá đang ở gần trên POC (trong khoảng 1%)
   - Đang di chuyển xuống gần POC
   - Không trong uptrend mạnh
   - Stochastic > 50 và K < D
   - RSI < 60

3. **POC Resistance**:
   - Giá chạm POC từ dưới lên
   - Đóng cửa dưới POC
   - Trong downtrend (EMA fast < medium < slow)
   - POC đủ mạnh (volume trên trung bình)
   - Stochastic > 40

4. **Value Area Breakout Bearish**:
   - Giá đóng cửa dưới VAL
   - Nến trước đóng trên VAL
   - Value Area đủ rộng (> 3%)
   - Volume > 150% trung bình
   - Stochastic < 60 và đang giảm

5. **Strong Downtrend + Expanding VA**:
   - Trong downtrend (EMA fast < medium < slow)
   - VAL và POC đang giảm
   - Giá dưới POC
   - Stochastic 30-60
   - Volume > 120% trung bình

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ rõ ràng của tín hiệu Volume Profile:

#### Long:
- **VAL Bounce với POC mạnh**: Đòn bẩy tối đa (6x)
- **Breakout với volume cao**: Đòn bẩy cao (5x)
- **POC Support mạnh**: Đòn bẩy trên trung bình (4x)
- **Các tín hiệu khác**: Đòn bẩy mặc định (3x)

#### Short:
- **VAH Bounce với POC mạnh**: Đòn bẩy tối đa (6x)
- **Breakout với volume cao**: Đòn bẩy cao (5x)
- **POC Resistance mạnh**: Đòn bẩy trên trung bình (4x)
- **Các tín hiệu khác**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -4%
- **Với Long trade**:
  - Giá trên POC: Stoploss tại POC
  - Giá trong Value Area: Stoploss tại VAL
- **Với Short trade**:
  - Giá dưới POC: Stoploss tại POC
  - Giá trong Value Area: Stoploss tại VAH

### ROI (Return on Investment)

- **0 giờ**: 8% lợi nhuận
- **24 giờ**: 5% lợi nhuận
- **72 giờ**: 3% lợi nhuận
- **120 giờ**: 2% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **Reaching VAH với volume yếu**:
   - Giá chạm/gần VAH
   - Volume < 120% trung bình (không đủ mạnh để breakout)
   - Stochastic > 75

2. **Rejected at POC từ trên xuống**:
   - Giá đang ở gần trên POC và giảm
   - Nến đỏ (close < open)
   - Stochastic đang giảm

3. **Stochastic Overbought + Bearish Cross**:
   - Stochastic > 80
   - K vừa cắt xuống dưới D

4. **Falling back sau khi Breakout**:
   - Giá từ trên VAH quay trở lại dưới VAH
   - Volume tốt

5. **EMA Cross Bearish**:
   - EMA nhanh cắt xuống dưới EMA trung bình

### Thoát Short

1. **Reaching VAL với volume yếu**:
   - Giá chạm/gần VAL
   - Volume < 120% trung bình (không đủ mạnh để breakout)
   - Stochastic < 25

2. **Rejected at POC từ dưới lên**:
   - Giá đang ở gần dưới POC và tăng
   - Nến xanh (close > open)
   - Stochastic đang tăng

3. **Stochastic Oversold + Bullish Cross**:
   - Stochastic < 20
   - K vừa cắt lên trên D

4. **Climbing back sau khi Breakout**:
   - Giá từ dưới VAL quay trở lại trên VAL
   - Volume tốt

5. **EMA Cross Bullish**:
   - EMA nhanh cắt lên trên EMA trung bình

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **POC yếu**: Không giao dịch khi POC strength < 70% ngưỡng tối thiểu
3. **Value Area hẹp**: Không giao dịch khi VA width < 70% ngưỡng tối thiểu
4. **Gap quá lớn**: Không giao dịch khi giá vào lệnh cách giá đóng cửa gần nhất > 1%

## Tham số tối ưu hóa

### Tham số Volume Profile
- **vp_period**: Số nến để tính Volume Profile (50-200, mặc định 100)
- **vp_bins**: Số vùng giá để phân tích (10-50, mặc định 20)
- **va_width_min**: Độ rộng tối thiểu của Value Area (1.0-10.0%, mặc định 3.0%)
- **poc_volume_min**: Độ mạnh tối thiểu của POC (1.0-5.0, mặc định 2.0)

### Tham số EMA
- **ema_fast**: EMA nhanh (5-50, mặc định 21)
- **ema_medium**: EMA trung bình (20-100, mặc định 55)
- **ema_slow**: EMA chậm (50-200, mặc định 100)

### Tham số Stochastic
- **stoch_k**: Period K (5-21, mặc định 14)
- **stoch_d**: Period D (3-10, mặc định 3)
- **stoch_smooth**: Smoothing K (1-5, mặc định 3)

## Ưu điểm của chiến lược

1. **Xác định giá trị thực**: Volume Profile giúp phát hiện mức giá thực sự được thị trường quan tâm
2. **Mean Reversion + Trend Following**: Chiến lược kết hợp cả hai phương pháp giao dịch
3. **Bằng chứng thống kê**: Dựa trên dữ liệu thực tế về sự phân bố khối lượng, không đơn thuần dựa trên pattern
4. **Stoploss tối ưu**: Stoploss được đặt tại các mức hỗ trợ/kháng cự thực từ dữ liệu volume

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 4 giờ
- Khối lượng giao dịch ổn định
- Sự tham gia của cả nhà đầu tư tổ chức và cá nhân
- Các vùng giá giúp xác định điểm tham gia hiệu quả

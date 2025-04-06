# Chiến lược Supertrend + RSI + MACD

Chiến lược này kết hợp ba chỉ báo kỹ thuật mạnh mẽ: Supertrend, RSI và MACD để tạo ra một hệ thống giao dịch toàn diện. Được tối ưu hóa cho khung thời gian 1 giờ (1h).

## Tổng quan

SupertrendRSIMACD là chiến lược giao dịch toàn diện nhằm xác định và giao dịch theo xu hướng thị trường thông qua sự kết hợp của ba chỉ báo kỹ thuật phổ biến:

1. **Supertrend**: Xác định xu hướng tổng thể và các điểm chuyển xu hướng
2. **RSI (Relative Strength Index)**: Xác định vùng quá mua/quá bán và xác nhận đảo chiều
3. **MACD (Moving Average Convergence Divergence)**: Xác nhận động lượng và sức mạnh xu hướng

Chiến lược còn sử dụng ADX để xác nhận độ mạnh của xu hướng và phân tích khối lượng giao dịch để lọc tín hiệu.

## Phương pháp giao dịch

### 1. Supertrend

- **Phát hiện xu hướng**: Sử dụng Supertrend để xác định xu hướng thị trường (uptrend hoặc downtrend)
- **Đảo chiều**: Phát hiện các điểm chuyển xu hướng (trend change) từ Supertrend
- **Tham số tối ưu**: Chu kỳ (5-21, mặc định 10) và hệ số ATR (1.0-4.0, mặc định 3.0)

### 2. RSI (Relative Strength Index)

- **Quá mua/quá bán**: Xác định khi thị trường trong vùng quá mua/quá bán
- **Phân kỳ**: Kết hợp với các chỉ báo khác để phát hiện phân kỳ
- **Tham số tối ưu**: Chu kỳ (7-21, mặc định 14), ngưỡng mua (20-40, mặc định 30), ngưỡng bán (60-80, mặc định 70)

### 3. MACD (Moving Average Convergence Divergence)

- **Động lượng**: Xác định xu hướng và sức mạnh động lượng
- **Cross**: Phát hiện khi MACD cắt qua Signal Line
- **Tham số tối ưu**: Fast period (8-16, mặc định 12), Slow period (18-34, mặc định 26), Signal period (6-14, mặc định 9)

### 4. ADX (Average Directional Index)

- Sử dụng để đo lường độ mạnh của xu hướng
- Chỉ giao dịch khi ADX > 20 (xu hướng đủ mạnh)

## Hệ thống tín hiệu consensus

Chiến lược sử dụng phương pháp "Signal Consensus" để đánh giá độ tin cậy của tín hiệu. Mỗi tín hiệu được tính 1 điểm, tổng điểm từ 0-3:

- **Supertrend**: +1 nếu xu hướng phù hợp (uptrend cho long, downtrend cho short)
- **RSI**: +1 nếu RSI trong vùng phù hợp (quá bán cho long, quá mua cho short)
- **MACD**: +1 nếu MACD phù hợp (MACD > Signal cho long, MACD < Signal cho short)

Chỉ giao dịch khi tổng điểm đạt hoặc vượt ngưỡng consensus (thường là 2 hoặc 3).

## Các tín hiệu vào lệnh

### Tín hiệu Long

1. **Supertrend Uptrend + RSI thấp + MACD tăng**: Kết hợp cả 3 chỉ báo với volume cao
2. **Reversal lên**: Supertrend chuyển từ downtrend sang uptrend, RSI thấp và MACD cắt lên
3. **Consensus Signal**: Ít nhất 2/3 chỉ báo cho tín hiệu mua và ADX > 25

### Tín hiệu Short

1. **Supertrend Downtrend + RSI cao + MACD giảm**: Kết hợp cả 3 chỉ báo với volume cao
2. **Reversal xuống**: Supertrend chuyển từ uptrend sang downtrend, RSI cao và MACD cắt xuống
3. **Consensus Signal**: Ít nhất 2/3 chỉ báo cho tín hiệu bán và ADX > 25

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ mạnh của tín hiệu và ADX:

- **Tín hiệu rất mạnh** (3/3 chỉ báo + ADX > 30): Đòn bẩy tối đa (5x)
- **Tín hiệu mạnh** (3/3 chỉ báo hoặc 2/3 chỉ báo + ADX > 25): Đòn bẩy 4x
- **Tín hiệu reversal**: Đòn bẩy 4x
- **Tín hiệu trung bình**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -5%
- **Với lợi nhuận > 2%**: Stoploss di chuyển lên/xuống theo đường Supertrend
- **Trailing stop**: Kích hoạt ở mức 3% lợi nhuận, step 2%

### ROI (Return on Investment)

- **0 phút**: 6% lợi nhuận
- **180 phút (3 giờ)**: 4% lợi nhuận
- **300 phút (5 giờ)**: 3% lợi nhuận
- **480 phút (8 giờ)**: 2% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **ADX thấp**: Không giao dịch khi ADX < 20 (xu hướng yếu)
2. **Volume thấp**: Không giao dịch khi volume < 75% trung bình
3. **Consensus thấp**: Yêu cầu ít nhất 2/3 chỉ báo cho tín hiệu nhất quán
4. **Reversal chưa hoàn thành**: Đợi xác nhận từ cả 3 chỉ báo trước khi giao dịch reversal

## Thoát lệnh thông minh

### Thoát Long

1. **Supertrend chuyển xuống**: Khi Supertrend chuyển từ uptrend sang downtrend
2. **RSI quá cao**: Khi RSI > 80 (vùng quá mua mạnh)
3. **MACD cross down**: Khi MACD cắt xuống dưới Signal Line
4. **Consensus ngược lại**: Khi có ít nhất 2 tín hiệu short

### Thoát Short

1. **Supertrend chuyển lên**: Khi Supertrend chuyển từ downtrend sang uptrend
2. **RSI quá thấp**: Khi RSI < 20 (vùng quá bán mạnh)
3. **MACD cross up**: Khi MACD cắt lên trên Signal Line
4. **Consensus ngược lại**: Khi có ít nhất 2 tín hiệu long

## Tham số tối ưu hóa

### Tham số Supertrend
- **supertrend_period**: Chu kỳ cho Supertrend (5-21, mặc định 10)
- **supertrend_multiplier**: Hệ số ATR cho Supertrend (1.0-4.0, mặc định 3.0)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

### Tham số MACD
- **macd_fast**: Chu kỳ EMA nhanh (8-16, mặc định 12)
- **macd_slow**: Chu kỳ EMA chậm (18-34, mặc định 26)
- **macd_signal**: Chu kỳ EMA signal (6-14, mặc định 9)

### Tham số Consensus
- **signal_consensus**: Số lượng tín hiệu cần thiết (2-3, mặc định 2)

## Ưu điểm của chiến lược

1. **Đa chiều**: Kết hợp nhiều loại phân tích (xu hướng, động lượng, quá mua/bán).
2. **Consensus**: Sử dụng phương pháp consensus giảm tín hiệu giả.
3. **Thích ứng**: Điều chỉnh đòn bẩy và stoploss dựa trên độ mạnh của tín hiệu.
4. **Thời gian giữ lệnh linh hoạt**: ROI tối ưu cho các giao dịch kéo dài nhiều giờ.

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 1 giờ
- Xu hướng rõ ràng (ADX > 25)
- Biến động trung bình
- Giao dịch dài hạn hơn (vài giờ đến vài ngày)

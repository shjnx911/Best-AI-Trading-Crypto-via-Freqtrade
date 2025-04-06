# Chiến lược Ichimoku Cloud + Squeeze Momentum

Chiến lược này kết hợp hệ thống Ichimoku Kinko Hyo (Ichimoku Cloud) với chỉ báo Squeeze Momentum để xác định cả xu hướng và thời điểm bùng nổ giá. Được tối ưu hóa cho khung thời gian 4 giờ (4h).

## Tổng quan

IchimokuSqueezeMomentum Strategy kết hợp hai hệ thống giao dịch mạnh mẽ:

1. **Ichimoku Cloud**: Hệ thống phân tích xu hướng toàn diện từ Nhật Bản, cung cấp thông tin về xu hướng, hỗ trợ/kháng cự, và điểm vào lệnh
2. **Squeeze Momentum**: Kỹ thuật phát hiện khi thị trường đang tích lũy (squeeze) và sắp bùng nổ, được phát triển bởi John Carter

Chiến lược này đặc biệt hiệu quả với giao dịch xu hướng dài hạn hơn và các breakout sau giai đoạn tích lũy.

## Phương pháp giao dịch

### 1. Ichimoku Cloud (Ichimoku Kinko Hyo)

Ichimoku Cloud là hệ thống phân tích kỹ thuật toàn diện với 5 thành phần chính:

- **Tenkan-sen (Đường chuyển đổi)**: Trung bình của high và low trong 9 nến gần nhất
- **Kijun-sen (Đường cơ sở)**: Trung bình của high và low trong 26 nến gần nhất
- **Senkou Span A (Cloud A)**: Trung bình của Tenkan-sen và Kijun-sen, dịch chuyển 26 nến về tương lai
- **Senkou Span B (Cloud B)**: Trung bình của high và low trong 52 nến gần nhất, dịch chuyển 26 nến về tương lai
- **Chikou Span (Đường trễ)**: Giá đóng cửa hiện tại dịch chuyển 26 nến về quá khứ

Hệ thống này giúp xác định:
- Xu hướng tổng thể dựa trên vị trí giá so với cloud (đám mây)
- Điểm vào lệnh dựa trên TK Cross (Tenkan/Kijun Cross)
- Xác nhận tín hiệu bằng Chikou Span

### 2. Squeeze Momentum

Squeeze Momentum kết hợp hai dải biến động: Bollinger Bands và Keltner Channels:

- **Bollinger Bands**: Dựa trên độ lệch chuẩn của giá
- **Keltner Channels**: Dựa trên Average True Range (ATR)

Chỉ báo này phát hiện:
- **Squeeze On**: Khi Bollinger Bands nằm trong Keltner Channels (thị trường đang tích lũy)
- **Squeeze Off**: Khi Bollinger Bands mở rộng ra ngoài Keltner Channels (hết tích lũy)
- **Squeeze Fired**: Thời điểm squeeze vừa kết thúc (thường là thời điểm breakout)
- **Momentum**: Hướng và độ mạnh của đà chuyển động giá

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Ichimoku + Squeeze Combo**: Kết hợp tín hiệu từ cả hai hệ thống
   - Giá trên cloud
   - TK Cross Up (Tenkan cắt lên trên Kijun)
   - Chikou Span trên giá quá khứ
   - Squeeze vừa kết thúc với momentum dương và tăng
   - Volume cao và RSI > 40

2. **Tín hiệu Ichimoku mạnh**: Tất cả các thành phần của Ichimoku đều đưa ra tín hiệu mua
   - Giá trên cloud
   - TK Cross Up
   - Chikou Span trên giá quá khứ
   - Giá trên Senkou Span A và B
   - Volume cao và RSI > 45

3. **Squeeze Momentum bùng nổ mạnh**: Breakout mạnh sau thời gian tích lũy
   - Squeeze vừa kết thúc
   - Momentum dương và lớn hơn 150% trung bình 5 nến
   - Giá trên cloud
   - Volume rất cao
   - RSI hợp lý (40-80)

### Tín hiệu Short

1. **Ichimoku + Squeeze Combo**: Kết hợp tín hiệu từ cả hai hệ thống
   - Giá dưới cloud
   - TK Cross Down (Tenkan cắt xuống dưới Kijun)
   - Chikou Span dưới giá quá khứ
   - Squeeze vừa kết thúc với momentum âm và giảm
   - Volume cao và RSI < 60

2. **Tín hiệu Ichimoku mạnh**: Tất cả các thành phần của Ichimoku đều đưa ra tín hiệu bán
   - Giá dưới cloud
   - TK Cross Down
   - Chikou Span dưới giá quá khứ
   - Giá dưới Senkou Span A và B
   - Volume cao và RSI < 55

3. **Squeeze Momentum bùng nổ mạnh**: Breakdown mạnh sau thời gian tích lũy
   - Squeeze vừa kết thúc
   - Momentum âm và nhỏ hơn 150% trung bình 5 nến
   - Giá dưới cloud
   - Volume rất cao
   - RSI hợp lý (20-60)

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên vị trí giá so với Ichimoku Cloud và độ mạnh của Squeeze Momentum:

- **Điều kiện tốt nhất** (giá rõ ràng trên/dưới cloud + momentum rất mạnh + squeeze vừa kết thúc): Đòn bẩy tối đa (6x)
- **Điều kiện tốt** (giá rõ ràng trên/dưới cloud + momentum mạnh): Đòn bẩy cao (5x)
- **Điều kiện khá** (giá trong cloud hoặc trên/dưới cloud + momentum khá): Đòn bẩy trung bình cao (4x)
- **Điều kiện trung bình**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -7%
- **Với lợi nhuận > 3%**: Stoploss di chuyển lên/xuống theo Kijun-sen (đường cơ sở)
- **Trailing stop**: Kích hoạt ở mức 4% lợi nhuận, step 3%

### ROI (Return on Investment)

- **0 giờ**: 8% lợi nhuận
- **24 giờ (6 nến 4h)**: 5% lợi nhuận
- **48 giờ (12 nến 4h)**: 3% lợi nhuận
- **72 giờ (18 nến 4h)**: 2% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **RSI Extreme**: Không vào lệnh khi RSI > 85 (long) hoặc < 15 (short)
2. **Chikou Span**: Không vào lệnh khi Chikou Span không xác nhận tín hiệu
3. **Volume thấp**: Không giao dịch khi volume < 50% trung bình

## Thoát lệnh thông minh

### Thoát Long

1. **TK Cross Down**: Khi Tenkan cắt xuống dưới Kijun
2. **Giá dưới cloud**: Khi giá rơi xuống dưới cloud sau khi đã ở trên
3. **Momentum đảo chiều**: Khi momentum chuyển từ dương sang âm và giảm
4. **RSI quá cao**: Khi RSI > 80 và bắt đầu giảm

### Thoát Short

1. **TK Cross Up**: Khi Tenkan cắt lên trên Kijun
2. **Giá trên cloud**: Khi giá tăng lên trên cloud sau khi đã ở dưới
3. **Momentum đảo chiều**: Khi momentum chuyển từ âm sang dương và tăng
4. **RSI quá thấp**: Khi RSI < 20 và bắt đầu tăng

## Tham số tối ưu hóa

### Tham số Ichimoku Cloud
- **tenkan_period**: Chu kỳ tính Tenkan-sen (5-15, mặc định 9)
- **kijun_period**: Chu kỳ tính Kijun-sen (20-40, mặc định 26)
- **senkou_span_b_period**: Chu kỳ tính Senkou Span B (40-60, mặc định 52)

### Tham số Squeeze Momentum
- **bb_length**: Chu kỳ tính Bollinger Bands (15-25, mặc định 20)
- **bb_mult**: Hệ số độ lệch chuẩn cho Bollinger Bands (1.5-3.0, mặc định 2.0)
- **kc_length**: Chu kỳ tính Keltner Channels (15-25, mặc định 20)
- **kc_mult**: Hệ số ATR cho Keltner Channels (1.0-2.0, mặc định 1.5)

## Ưu điểm của chiến lược

1. **Toàn diện**: Kết hợp phân tích xu hướng và phát hiện breakout.
2. **Thời gian nắm giữ dài**: Phù hợp cho giao dịch từ nhiều giờ đến nhiều ngày.
3. **Quản lý rủi ro thích ứng**: Stoploss và đòn bẩy động dựa trên vị trí giá và biến động.
4. **Ít giao dịch giả**: Kết hợp nhiều bộ lọc để hạn chế tín hiệu giả.

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 4 giờ
- Xu hướng mạnh hoặc breakout sau giai đoạn tích lũy
- Coin và token có vốn hóa trung bình đến lớn (ít biến động bất thường)
- Chỉ số thị trường rõ ràng (không sideways hoặc dao động hẹp kéo dài)

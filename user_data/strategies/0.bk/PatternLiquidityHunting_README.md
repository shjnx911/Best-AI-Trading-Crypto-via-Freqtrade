# Chiến lược Pattern Recognition + Liquidity Hunting

Chiến lược này kết hợp nhận diện mẫu hình nến Nhật và kỹ thuật săn thanh khoản (Liquidity Hunting). Được tối ưu hóa cho khung thời gian 5 phút (5m).

## Tổng quan

PatternLiquidityHunting Strategy là chiến lược giao dịch kết hợp hai phương pháp tiếp cận chính:

1. **Pattern Recognition**: Nhận diện các mẫu hình nến Nhật truyền thống để xác định điểm đảo chiều và breakout
2. **Liquidity Hunting**: Phát hiện các điểm "săn thanh khoản" - nơi các "cá mập" thường săn lệnh stop loss của trader nhỏ lẻ

Chiến lược này đặc biệt hiệu quả trong việc xác định các điểm đảo chiều chính xác với rủi ro thấp và tiềm năng lợi nhuận cao.

## Phương pháp giao dịch

### 1. Nhận diện mẫu hình nến Nhật

Chiến lược phát hiện và giao dịch các mẫu hình nến sau:

#### Mẫu hình nến đảo chiều tăng (Bullish)
- **Hammer (Búa)**: Nến với bóng dưới dài, thân nhỏ, bóng trên rất ngắn hoặc không có
- **Bullish Engulfing (Bao phủ tăng)**: Nến tăng lớn bao phủ toàn bộ nến giảm trước đó
- **Morning Star (Sao Mai)**: Mẫu hình 3 nến với nến giảm, tiếp theo là doji, rồi đến nến tăng
- **Piercing Line (Đường xuyên thủng)**: Nến tăng mở cửa dưới mức thấp nhất của nến giảm trước đó, nhưng đóng cửa trên giữa thân nến trước
- **Tweezer Bottom (Nhíp đáy)**: Hai nến với mức thấp giống hệt nhau, nến đầu giảm và nến sau tăng
- **Three White Soldiers (Ba chiến binh trắng)**: Ba nến tăng liên tiếp, mỗi nến mở cửa trong thân nến trước và đóng cửa cao hơn

#### Mẫu hình nến đảo chiều giảm (Bearish)
- **Inverted Hammer (Búa ngược)**: Nến với bóng trên dài, thân nhỏ, bóng dưới rất ngắn hoặc không có
- **Bearish Engulfing (Bao phủ giảm)**: Nến giảm lớn bao phủ toàn bộ nến tăng trước đó
- **Evening Star (Sao Hôm)**: Mẫu hình 3 nến với nến tăng, tiếp theo là doji, rồi đến nến giảm
- **Dark Cloud Cover (Mây đen che phủ)**: Nến giảm mở cửa trên mức cao nhất của nến tăng trước đó, nhưng đóng cửa dưới giữa thân nến trước
- **Tweezer Top (Nhíp đỉnh)**: Hai nến với mức cao giống hệt nhau, nến đầu tăng và nến sau giảm
- **Three Black Crows (Ba quạ đen)**: Ba nến giảm liên tiếp, mỗi nến mở cửa trong thân nến trước và đóng cửa thấp hơn

### 2. Săn thanh khoản (Liquidity Hunting)

- **Stop Hunt Up**: Khi giá tăng vượt Bollinger Upper Band với bóng trên dài, rồi quay đầu giảm
- **Stop Hunt Down**: Khi giá giảm xuống dưới Bollinger Lower Band với bóng dưới dài, rồi quay đầu tăng
- **Fakeout**: Phá vỡ giả - nến bứt phá qua vùng hỗ trợ/kháng cự rồi quay trở lại
- **Consolidation**: Vùng tích lũy - nơi giá đi ngang trong range hẹp trước khi breakout
- **Breakout**: Phá vỡ khỏi vùng tích lũy với volume cao

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Mẫu hình đảo chiều bullish + RSI thấp**:
   - Xuất hiện mẫu hình nến đảo chiều tăng (Hammer, Bullish Engulfing, Morning Star, v.v.)
   - RSI < ngưỡng mua (mặc định 30)
   - Volume cao hơn trung bình

2. **Stop Hunt Down hoặc Fakeout Down**:
   - Giá giảm xuống dưới Bollinger Lower Band tạo bóng dưới dài
   - Sau đó quay đầu tăng (đóng cửa trên mở cửa)
   - Không trong downtrend (tránh bắt dao rơi)
   - Volume cao

3. **Breakout từ vùng tích lũy**:
   - Giá bứt phá lên trên Bollinger Upper Band
   - Sau một giai đoạn tích lũy (BB Width thấp)
   - RSI không quá cao (< 70)
   - Volume tăng đột biến

4. **Hammer gần vùng hỗ trợ**:
   - Nến Hammer xuất hiện gần Bollinger Lower Band
   - RSI < 40
   - Volume trên trung bình

5. **Gap Up trong uptrend**:
   - Giá mở cửa cao hơn high của nến trước
   - Trong uptrend
   - Volume cao

### Tín hiệu Short

1. **Mẫu hình đảo chiều bearish + RSI cao**:
   - Xuất hiện mẫu hình nến đảo chiều giảm (Inverted Hammer, Bearish Engulfing, Evening Star, v.v.)
   - RSI > ngưỡng bán (mặc định 70)
   - Volume cao hơn trung bình

2. **Stop Hunt Up hoặc Fakeout Up**:
   - Giá tăng vượt Bollinger Upper Band tạo bóng trên dài
   - Sau đó quay đầu giảm (đóng cửa dưới mở cửa)
   - Không trong uptrend (tránh bán trong xu hướng tăng mạnh)
   - Volume cao

3. **Breakdown từ vùng tích lũy**:
   - Giá phá xuống dưới Bollinger Lower Band
   - Sau một giai đoạn tích lũy (BB Width thấp)
   - RSI không quá thấp (> 30)
   - Volume tăng đột biến

4. **Inverted Hammer gần vùng kháng cự**:
   - Nến Inverted Hammer xuất hiện gần Bollinger Upper Band
   - RSI > 60
   - Volume trên trung bình

5. **Gap Down trong downtrend**:
   - Giá mở cửa thấp hơn low của nến trước
   - Trong downtrend
   - Volume cao

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên chất lượng tín hiệu:

- **Tín hiệu tối ưu** (Mẫu hình rõ ràng + Volume cao + Stop hunt/Fakeout mạnh): Đòn bẩy tối đa (8x)
- **Tín hiệu tốt** (2/3 yếu tố trên): Đòn bẩy cao (6x)
- **Tín hiệu khá** (1/3 yếu tố trên): Đòn bẩy trung bình cao (4x)
- **Tín hiệu cơ bản**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -3%
- **Với lợi nhuận > 1.5%**: Stoploss di chuyển lên/xuống theo Bollinger Middle Band
- **Trailing stop**: Kích hoạt ở mức 1.5% lợi nhuận, step 1%

### ROI (Return on Investment)

- **0 phút**: 3% lợi nhuận
- **15 phút**: 2% lợi nhuận
- **30 phút**: 1% lợi nhuận
- **60 phút**: 0.5% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volatility quá thấp**: Không giao dịch khi BB Width < 0.03
2. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
3. **RSI Extreme**: Không long khi RSI > 75, không short khi RSI < 25
4. **Price Action mâu thuẫn**: Đánh giá tổng thể các mẫu hình và thanh khoản

## Thoát lệnh thông minh

### Thoát Long

1. **Mẫu hình đảo chiều bearish**: Khi xuất hiện mẫu hình nến đảo chiều giảm rõ ràng
2. **Chạm BB Upper + RSI cao**: Khi giá chạm gần Bollinger Upper Band và RSI > 70
3. **Stop Hunt Up**: Khi phát hiện tín hiệu săn thanh khoản hướng lên
4. **EMA Cross Down**: Khi EMA ngắn cắt xuống dưới EMA dài
5. **Stochastic Cross Down**: Khi Stochastic %K cắt xuống dưới %D từ vùng overbought

### Thoát Short

1. **Mẫu hình đảo chiều bullish**: Khi xuất hiện mẫu hình nến đảo chiều tăng rõ ràng
2. **Chạm BB Lower + RSI thấp**: Khi giá chạm gần Bollinger Lower Band và RSI < 30
3. **Stop Hunt Down**: Khi phát hiện tín hiệu săn thanh khoản hướng xuống
4. **EMA Cross Up**: Khi EMA ngắn cắt lên trên EMA dài
5. **Stochastic Cross Up**: Khi Stochastic %K cắt lên trên %D từ vùng oversold

## Tham số tối ưu hóa

### Tham số Pattern Recognition
- **candle_low_high_ratio**: Tỷ lệ bóng:thân tối thiểu (1.5-4.0, mặc định 2.0)
- **candle_body_pct**: % thân nến tối đa (10.0-50.0, mặc định 30.0)

### Tham số Liquidity Hunting
- **atr_multiplier**: Hệ số ATR để phát hiện stop hunt (1.0-3.0, mặc định 1.5)
- **volume_surge_threshold**: Ngưỡng volume tăng vọt (1.2-2.5, mặc định 1.5)
- **consolidation_threshold**: Ngưỡng tích lũy (0.7-0.9, mặc định 0.8)

### Tham số Trend
- **ema_short**: EMA ngắn (5-15, mặc định 8)
- **ema_long**: EMA dài (20-50, mặc định 26)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy**: Ngưỡng RSI cho tín hiệu mua (20-40, mặc định 30)
- **rsi_sell**: Ngưỡng RSI cho tín hiệu bán (60-80, mặc định 70)

## Ưu điểm của chiến lược

1. **Xác định điểm đảo chiều mạnh**: Phát hiện các điểm cá lớn đang săn thanh khoản
2. **Kết hợp đa tầng**: Lọc tín hiệu qua nhiều lớp phân tích (mẫu hình, thanh khoản, kỹ thuật)
3. **Tỷ lệ risk:reward cao**: Vào lệnh ở các điểm đảo chiều với stoploss hợp lý
4. **Tận dụng tâm lý thị trường**: Hiểu và tận dụng hành vi của các nhà giao dịch lớn

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 5 phút
- Biến động trung bình đến cao
- Khối lượng giao dịch đủ lớn và ổn định
- Thị trường có các "cá mập" săn thanh khoản nhiều

# Chiến lược Sóng Elliott (Elliott Waves)

Chiến lược này dựa trên phân tích sóng Elliott kết hợp với các mức Fibonacci và RSI. Được tối ưu hóa cho khung thời gian 15 phút (15m).

## Tổng quan

Elliot_Waves_Strategy áp dụng lý thuyết sóng Elliott - một phương pháp phân tích kỹ thuật cổ điển kết hợp với các công cụ hiện đại:

1. **Lý thuyết sóng Elliott**: Xác định các mẫu sóng xác định (5 sóng động lực + 3 sóng điều chỉnh) trong chuyển động giá
2. **Fibonacci Retracement/Extension**: Xác định các mức hỗ trợ/kháng cự tự nhiên và mục tiêu giá
3. **RSI (Relative Strength Index)**: Xác nhận tín hiệu và lọc các điểm vào lệnh

Chiến lược này phù hợp với các nhà giao dịch kỹ thuật có kiến thức về mô hình sóng và muốn tận dụng lợi thế tâm lý đám đông trong thị trường.

## Phương pháp giao dịch

### 1. Phân tích Sóng Elliott

- **Sóng động lực (Impulse Waves)**: Bao gồm 5 sóng theo hướng xu hướng chính (1-2-3-4-5)
- **Sóng điều chỉnh (Corrective Waves)**: Bao gồm 3 sóng ngược hướng xu hướng chính (A-B-C)
- **Quy tắc cơ bản**: 
  - Sóng 2 không vượt quá điểm bắt đầu sóng 1
  - Sóng 3 không ngắn nhất trong các sóng 1, 3, 5
  - Sóng 4 không chồng lên sóng 1

### 2. Phân tích Fibonacci

- **Retracement Levels**: 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
- **Extension Levels**: 1.0, 1.272, 1.414, 1.618, 2.0, 2.618
- **Ứng dụng**: Xác định điểm vào/ra lệnh tại các mức quan trọng
- **Tham số tối ưu**: fib_buy_level (0.5-0.786, mặc định 0.618), fib_sell_level (1.0-2.0, mặc định 1.618)

### 3. RSI (Relative Strength Index)

- **Ngưỡng**: 30/70 (tiêu chuẩn)
- **Xác nhận**: Kết hợp với phân tích sóng và Fibonacci để lọc tín hiệu

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Fibonacci Buy Setup**:
   - Giá chạm mức Fibonacci Retracement (0.618) sau sóng đi xuống
   - RSI < 35 (gần vùng oversold)
   - Volume tốt

2. **End of Impulse Wave Down**:
   - Sau khi hoàn thành 5 sóng đi xuống
   - RSI < 40
   - Volume tốt

3. **End of Corrective ABC Wave** (sau sóng đi xuống):
   - Kết thúc sóng C của mẫu ABC sau impulse wave đi xuống
   - Nến xanh (close > open)
   - RSI < 45
   - Volume tăng

4. **Fibonacci 0.618 Bounce**:
   - Giá hồi phục đến mức 0.618 sau sóng đi xuống
   - Nến xanh
   - RSI < 45
   - Volume tốt

5. **Start of New Impulse Wave Up**:
   - Phát hiện bắt đầu sóng 1 của impulse wave mới đi lên
   - RSI < 60
   - Volume tăng cao

### Tín hiệu Short

1. **Fibonacci Sell Setup**:
   - Giá chạm mức Fibonacci Extension (1.618) sau sóng đi lên
   - RSI > 65 (gần vùng overbought)
   - Volume tốt

2. **End of Impulse Wave Up**:
   - Sau khi hoàn thành 5 sóng đi lên
   - RSI > 60
   - Volume tốt

3. **End of Corrective ABC Wave** (sau sóng đi lên):
   - Kết thúc sóng C của mẫu ABC sau impulse wave đi lên
   - Nến đỏ (close < open)
   - RSI > 55
   - Volume tăng

4. **Fibonacci 0.618 Bounce**:
   - Giá hồi phục đến mức 0.618 sau sóng đi lên
   - Nến đỏ
   - RSI > 55
   - Volume tốt

5. **Start of New Impulse Wave Down**:
   - Phát hiện bắt đầu sóng 1 của impulse wave mới đi xuống
   - RSI > 40
   - Volume tăng cao

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên độ rõ ràng của tín hiệu sóng Elliott:

#### Long:
- **End of 5-wave down + RSI thấp**: Đòn bẩy tối đa (5x)
- **Fib buy setup + RSI thấp**: Đòn bẩy cao (4x)
- **End of ABC correction sau sóng xuống**: Đòn bẩy cao (4x)
- **Fibonacci bounce**: Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

#### Short:
- **End of 5-wave up + RSI cao**: Đòn bẩy tối đa (5x)
- **Fib sell setup + RSI cao**: Đòn bẩy cao (4x)
- **End of ABC correction sau sóng lên**: Đòn bẩy cao (4x)
- **Fibonacci bounce**: Đòn bẩy trung bình (3x)
- **Tín hiệu khác**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -3%
- **Với lợi nhuận > 2%**:
  - Long: Stoploss tại mức Fibonacci (0.382, 0.5, 0.618) gần nhất dưới giá hiện tại, hoặc EMA chậm
  - Short: Stoploss tại mức Fibonacci (0.382, 0.5, 0.618) gần nhất trên giá hiện tại, hoặc EMA chậm
  - Không quá -5% trong mọi trường hợp

### ROI (Return on Investment)

- **0 phút**: 5% lợi nhuận
- **30 phút**: 3% lợi nhuận
- **60 phút**: 2% lợi nhuận
- **90 phút**: 1% lợi nhuận

## Thoát lệnh thông minh

### Thoát Long

1. **RSI Overbought**:
   - RSI > 75 và đang giảm

2. **Fibonacci Extension Target**:
   - Giá đạt đến mức Fibonacci extension 1.618

3. **End of Impulse Wave Up**:
   - Hoàn thành 5 sóng đi lên (sóng 5)

4. **Fibonacci Resistance Level**:
   - Giá chạm mức Fibonacci retracement 0.786 với volume tốt

5. **Start of New Impulse Wave Down**:
   - Phát hiện bắt đầu sóng 1 của impulse wave mới đi xuống

### Thoát Short

1. **RSI Oversold**:
   - RSI < 25 và đang tăng

2. **Fibonacci Extension Target**:
   - Giá đạt đến mức Fibonacci extension 1.618

3. **End of Impulse Wave Down**:
   - Hoàn thành 5 sóng đi xuống (sóng 5)

4. **Fibonacci Support Level**:
   - Giá chạm mức Fibonacci retracement 0.786 với volume tốt

5. **Start of New Impulse Wave Up**:
   - Phát hiện bắt đầu sóng 1 của impulse wave mới đi lên

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 70% trung bình
2. **RSI Extreme**: Không long khi RSI > 75, không short khi RSI < 25
3. **Sóng không phù hợp**:
   - Không long khi đang trong sóng impulse đi xuống (sóng 1-4)
   - Không short khi đang trong sóng impulse đi lên (sóng 1-4)
   - Không long khi đang trong sóng B của corrective wave lên
   - Không short khi đang trong sóng B của corrective wave xuống

## Tham số tối ưu hóa

### Tham số Elliott Waves
- **swing_window**: Cửa sổ xác định swing point (3-10, mặc định 5)
- **impulse_threshold**: Ngưỡng % tối thiểu cho sóng động lực (1.0-5.0, mặc định 2.0)
- **corrective_threshold**: Ngưỡng % tối thiểu cho sóng điều chỉnh (0.5-3.0, mặc định 1.0)

### Tham số Fibonacci
- **fib_buy_level**: Mức Fibonacci cho tín hiệu mua (0.5-0.786, mặc định 0.618)
- **fib_sell_level**: Mức Fibonacci cho tín hiệu bán (1.0-2.0, mặc định 1.618)

### Tham số RSI
- **rsi_period**: Chu kỳ tính RSI (7-21, mặc định 14)
- **rsi_buy_threshold**: Ngưỡng mua (20-40, mặc định 30)
- **rsi_sell_threshold**: Ngưỡng bán (60-80, mặc định 70)

### Tham số EMA
- **ema_fast**: EMA nhanh (5-30, mặc định 8)
- **ema_slow**: EMA chậm (15-50, mặc định 21)

## Ưu điểm của chiến lược

1. **Tận dụng tâm lý đám đông**: Sóng Elliott phản ánh tâm lý thị trường và hành vi người giao dịch
2. **Xác định điểm đảo chiều**: Đặc biệt hiệu quả trong việc xác định điểm kết thúc xu hướng và điểm đảo chiều mới
3. **Mục tiêu rõ ràng**: Fibonacci extension cung cấp mục tiêu lợi nhuận có cơ sở
4. **Quản lý rủi ro chính xác**: Stoploss tại các mức Fibonacci giúp đặt SL tại các mức hỗ trợ/kháng cự tự nhiên

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 15 phút
- Chuyển động giá rõ ràng với xu hướng có thể nhận diện
- Đủ khối lượng giao dịch
- Không quá nhiều can thiệp từ ngoại lực (tin tức)

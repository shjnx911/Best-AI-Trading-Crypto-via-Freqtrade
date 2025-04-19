# Chiến lược Fractal + Hull Moving Average

Chiến lược này kết hợp Fractals của Bill Williams với Hull Moving Average để xác định điểm đảo chiều và điểm vào lệnh trong xu hướng. Được tối ưu hóa cho khung thời gian 5 phút (5m).

## Tổng quan

FractalHullMA Strategy là chiến lược giao dịch kết hợp hai công cụ kỹ thuật mạnh mẽ:

1. **Fractals**: Chỉ báo của Bill Williams dùng để xác định các điểm đảo chiều tiềm năng (swing highs và swing lows).
2. **Hull Moving Average (HMA)**: Đường trung bình động tiên tiến cho phép phản ứng nhanh với sự thay đổi giá và giảm lag.

Chiến lược này tập trung vào việc giao dịch các pullback về HMA hoặc các fractal gần đường HMA trong xu hướng rõ ràng, cung cấp điểm vào với rủi ro thấp và tiềm năng lợi nhuận cao.

## Phương pháp giao dịch

### 1. Fractals

- **Định nghĩa**: Một fractal up (đảo chiều xuống tiềm năng) được tạo ra khi high của một nến cao hơn high của 'n' nến trước và sau nó. Một fractal down (đảo chiều lên tiềm năng) được tạo ra khi low của một nến thấp hơn low của 'n' nến trước và sau nó.
- **Ứng dụng**: Fractal up là vùng kháng cự tiềm năng, fractal down là vùng hỗ trợ tiềm năng.
- **Tham số tối ưu**: window (1-5, mặc định 2) - số nến cần xét để xác định fractal.

### 2. Hull Moving Average (HMA)

- **Định nghĩa**: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n), là đường trung bình động cải tiến giúp giảm lag và tăng độ nhạy.
- **Ứng dụng**: HMA dùng để xác định xu hướng và làm mức hỗ trợ/kháng cự động.
- **Tham số tối ưu**: Fast HMA (6-24, mặc định 9) và Slow HMA (25-100, mặc định 49).

### 3. Phân tích xu hướng

- **Uptrend**: Khi Fast HMA > Slow HMA và HMA đang có độ dốc dương.
- **Downtrend**: Khi Fast HMA < Slow HMA và HMA đang có độ dốc âm.
- **Độ dốc (Angle)**: Đo lường bằng % thay đổi của HMA qua 3 nến.

## Tín hiệu giao dịch

### Tín hiệu Long

1. **Fractal Long Signal**: 
   - Trong uptrend (Fast HMA > Slow HMA, góc dương)
   - Fractal down gần Fast HMA (trong khoảng fractal_distance)
   - Volume không quá thấp
   - RSI > 40

2. **Pullback to HMA**:
   - Trong uptrend
   - Giá chạm HMA (trong vùng ±0.5%)
   - RSI hợp lý (40-60)
   - Volume không quá thấp

3. **Breakout trên HMA sau fractal**:
   - Trong uptrend
   - Có fractal down trong 1-2 nến trước
   - Giá từ dưới HMA cắt lên trên
   - Volume tốt

### Tín hiệu Short

1. **Fractal Short Signal**:
   - Trong downtrend (Fast HMA < Slow HMA, góc âm)
   - Fractal up gần Fast HMA (trong khoảng fractal_distance)
   - Volume không quá thấp
   - RSI < 60

2. **Pullback to HMA**:
   - Trong downtrend
   - Giá chạm HMA (trong vùng ±0.5%)
   - RSI hợp lý (40-60)
   - Volume không quá thấp

3. **Breakdown dưới HMA sau fractal**:
   - Trong downtrend
   - Có fractal up trong 1-2 nến trước
   - Giá từ trên HMA cắt xuống dưới
   - Volume tốt

## Quản lý vốn và rủi ro

### Đòn bẩy động

Chiến lược điều chỉnh đòn bẩy dựa trên khoảng cách từ fractal đến HMA và chỉ số RSI:

#### Long:
- **Tối ưu** (Fractal gần HMA < 0.2% + RSI < 40): Đòn bẩy tối đa (7x)
- **Tốt** (Fractal gần HMA < 0.4% + RSI < 45): Đòn bẩy cao (5x)
- **Khá** (Fractal gần HMA < 0.6%): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

#### Short:
- **Tối ưu** (Fractal gần HMA < 0.2% + RSI > 60): Đòn bẩy tối đa (7x)
- **Tốt** (Fractal gần HMA < 0.4% + RSI > 55): Đòn bẩy cao (5x)
- **Khá** (Fractal gần HMA < 0.6%): Đòn bẩy trung bình cao (4x)
- **Bình thường**: Đòn bẩy mặc định (3x)

### Stoploss động

- **Stoploss mặc định**: -2.5%
- **Với lợi nhuận > 1%**: Stoploss di chuyển lên/xuống theo fractal gần nhất
- **Trailing stop**: Kích hoạt ở mức 1% lợi nhuận, step 0.8%

### ROI (Return on Investment)

- **0 phút**: 3% lợi nhuận
- **15 phút**: 2% lợi nhuận
- **30 phút**: 1% lợi nhuận
- **45 phút**: 0.5% lợi nhuận

## Lọc tín hiệu bổ sung

Chiến lược sử dụng các lọc bổ sung để tránh giao dịch không tốt:

1. **Volume thấp**: Không giao dịch khi volume < 50% trung bình
2. **HMA Angle mâu thuẫn**: Không long khi HMA có góc âm, không short khi HMA có góc dương
3. **RSI Extreme**: Giao dịch với thận trọng khi RSI quá cao/thấp
4. **Khoảng cách fractal**: Ưu tiên fractal gần HMA hơn là xa

## Thoát lệnh thông minh

### Thoát Long

1. **Giá vượt HMA quá xa**: Khi giá > HMA + close_above_hma_pct% và RSI > 70
2. **Fractal up xuất hiện**: Khi xuất hiện fractal up mới
3. **HMA chuyển xu hướng**: Khi Fast HMA từ trên cắt xuống dưới Slow HMA
4. **RSI quá cao**: Khi RSI > 80

### Thoát Short

1. **Giá dưới HMA quá xa**: Khi giá < HMA - close_above_hma_pct% và RSI < 30
2. **Fractal down xuất hiện**: Khi xuất hiện fractal down mới
3. **HMA chuyển xu hướng**: Khi Fast HMA từ dưới cắt lên trên Slow HMA
4. **RSI quá thấp**: Khi RSI < 20

## Tham số tối ưu hóa

### Tham số Fractals
- **fractal_window**: Số nến để xác định fractal (1-5, mặc định 2)

### Tham số Hull Moving Average
- **hma_length_fast**: Chu kỳ cho Fast HMA (6-24, mặc định 9)
- **hma_length_slow**: Chu kỳ cho Slow HMA (25-100, mặc định 49)

### Tham số xác định tín hiệu
- **fractal_distance**: % khoảng cách tối đa từ fractal đến HMA (0.1-1.0, mặc định 0.3)
- **hma_angle_threshold**: Ngưỡng tối thiểu cho góc HMA (0.1-1.0, mặc định 0.3)

### Tham số Dynamic Trailing
- **close_above_hma_pct**: % khoảng cách HMA để thoát lệnh (0.1-1.0, mặc định 0.5)

## Ưu điểm của chiến lược

1. **Phát hiện đảo chiều sớm**: Fractals giúp phát hiện sớm điểm đảo chiều tiềm năng.
2. **Giảm trễ tín hiệu**: Hull MA phản ứng nhanh hơn so với các MA thông thường.
3. **Rủi ro thấp**: Giao dịch ở các điểm pullback về HMA hoặc fractal cung cấp tỷ lệ risk:reward tốt.
4. **Thích ứng với biến động**: Điều chỉnh đòn bẩy và stoploss dựa trên khoảng cách fractal và HMA.

## Khuyến nghị sử dụng

Chiến lược này phù hợp nhất với thị trường có:
- Khung thời gian 5 phút
- Xu hướng rõ ràng, tránh thị trường đi ngang
- Biến động trung bình đến cao
- Khối lượng giao dịch ổn định

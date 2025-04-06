# AI Profit Optimizer

AI Profit Optimizer là module tối ưu hóa lợi nhuận dựa trên AI để quản lý và tối đa hóa lợi nhuận hàng ngày cho hệ thống giao dịch tự động crypto.

## Mục tiêu chính

- Phân tích và tối ưu hóa giao dịch để đạt mục tiêu lợi nhuận 3% hàng ngày
- Đánh giá các cơ hội giao dịch dựa trên mô hình AI
- Điều chỉnh chiến lược để bảo toàn lợi nhuận sau khi đạt mục tiêu
- Quản lý đòn bẩy và phân bổ vốn một cách thông minh

## Tính năng chính

### 1. Phân tích lợi nhuận hàng ngày
- Tính toán lợi nhuận cần thiết để đạt mục tiêu 3% mỗi ngày
- Theo dõi tiến độ thực hiện mục tiêu hàng ngày
- Điều chỉnh chiến lược dựa trên tiến độ hiện tại

### 2. Đánh giá cơ hội giao dịch
- Tính điểm cơ hội (Opportunity Score) cho các cặp tiền dựa trên mô hình AI
- Thang điểm từ 0-100:
  - < 50: Tín hiệu yếu, bỏ qua
  - 50-75: Tín hiệu trung bình, xem xét nếu không có tùy chọn tốt hơn
  - 75-90: Tín hiệu tốt, xem xét vào lệnh
  - > 90: Tín hiệu rất tốt, khả năng cao đem lại lợi nhuận
- Lọc tín hiệu dựa trên điều kiện thị trường và tiến độ lợi nhuận

### 3. Mô hình AI tiên tiến
- Mô hình dự đoán lợi nhuận (Profit Predictor)
- Mô hình phân loại điểm vào lệnh (Entry Classifier)
- Mô hình phân loại điểm thoát lệnh (Exit Classifier)
- Tích hợp với OpenAI GPT-4 để phân tích chuyên sâu

### 4. Quản lý rủi ro thông minh
- Điều chỉnh đòn bẩy dựa trên độ tin cậy của tín hiệu (3x-6x)
- Tính toán stoploss động dựa trên các mức hỗ trợ/kháng cự
- Quản lý vốn thích ứng theo mục tiêu lợi nhuận

### 5. Tối ưu hóa danh mục đầu tư
- Xác định cặp tiền có tiềm năng cao nhất
- Giới hạn số lượng giao dịch đồng thời (tối đa 3 cặp)
- Phân tích hiệu suất để ưu tiên cặp tiền hiệu quả nhất

### 6. Thích ứng sau khi đạt mục tiêu
- Tăng ngưỡng điểm cơ hội (từ 75 lên 85) sau khi đạt mục tiêu 3%
- Tập trung vào bảo toàn lợi nhuận đã đạt được
- Chỉ giao dịch các cơ hội có xác suất cao nhất

## Cấu trúc module

### 1. `TrackingState`
- Theo dõi trạng thái giao dịch hàng ngày
- Quản lý các cặp tiền đang hoạt động
- Lưu trữ thông tin về điều kiện thị trường

### 2. `AIModel`
- Quản lý các mô hình học máy
- Tính toán opportunity score
- Trích xuất features từ dữ liệu thị trường

### 3. `ProfitOptimizer`
- Cung cấp các phương thức phân tích và tối ưu hóa
- Đưa ra khuyến nghị giao dịch
- Mô phỏng kịch bản giao dịch

## Quy trình hoạt động

1. **Phân tích lợi nhuận cần thiết**
   - Tính toán tiến độ hướng tới mục tiêu 3%
   - Xác định số lượng giao dịch và lợi nhuận trung bình cần thiết

2. **Đánh giá các cặp tiền**
   - Tính toán opportunity score cho mỗi cặp tiền
   - Xếp hạng các cặp tiền theo tiềm năng lợi nhuận

3. **Đưa ra khuyến nghị**
   - Đề xuất giao dịch dựa trên điểm cơ hội và tiến độ lợi nhuận
   - Tính toán điểm vào, điểm ra, stoploss và đòn bẩy tối ưu

4. **Theo dõi giao dịch**
   - Cập nhật lợi nhuận hàng ngày
   - Điều chỉnh chiến lược khi đạt mục tiêu

5. **Phân tích hiệu suất**
   - Đánh giá hiệu quả của các cặp tiền và khung thời gian
   - Đưa ra đề xuất cải thiện chiến lược

## Sử dụng module

### Khởi tạo

```python
from ai_profit_optimizer import ProfitOptimizer

# Khởi tạo optimizer
optimizer = ProfitOptimizer()
```

### Phân tích lợi nhuận cần thiết

```python
# Phân tích lợi nhuận
analysis = optimizer.analyze_daily_profit_needed()
print(f"Cần thêm {analysis['remaining_trades_needed']} giao dịch để đạt mục tiêu {analysis['target_profit']}%")
```

### Đánh giá cơ hội giao dịch

```python
# Đánh giá cơ hội giao dịch
recommendation = optimizer.get_trade_recommendation(dataframe, "BTC/USDT", datetime.now(), "long", "15m")
print(f"Opportunity Score: {recommendation['opportunity_score']}")
print(f"Khuyến nghị: {recommendation['action']}")
```

### Xác định điểm thoát lệnh

```python
# Xác định điểm thoát lệnh
exit_recommendation = optimizer.get_exit_recommendation(dataframe, "BTC/USDT", datetime.now(), entry_price=28500, "long", 1.5)
print(f"Khuyến nghị thoát lệnh: {exit_recommendation['recommendation']}")
print(f"Lý do: {exit_recommendation['reason']}")
```

### Theo dõi giao dịch hoàn thành

```python
# Cập nhật giao dịch đã hoàn thành
trade_info = {
    "pair": "BTC/USDT",
    "profit_pct": 1.5,
    "side": "long",
    "timeframe": "15m"
}
optimizer.track_completed_trade(trade_info)
```

### Mô phỏng tiềm năng lợi nhuận

```python
# Mô phỏng tiềm năng lợi nhuận
prediction = optimizer.predict_profit_potential(capital=1000, days=30)
print(f"Lợi nhuận dự kiến sau 30 ngày: ${prediction['expected_profit']:.2f} ({prediction['expected_profit_pct']:.2f}%)")
```

### Phân tích với OpenAI GPT

```python
# Phân tích sâu với OpenAI GPT
analysis_data = optimizer.analyze_trade_history()
gpt_analysis = optimizer.analyze_with_openai_gpt(analysis_data, "trade_analysis")
print(gpt_analysis['analysis'])
```

## Yêu cầu về môi trường

- Python 3.8+
- LightGBM
- Scikit-learn
- Pandas, NumPy
- CCXT
- OpenAI (tùy chọn, cho phân tích GPT)

## Cấu hình

Tạo file `.env` với nội dung:

```
OPENAI_API_KEY=sk-your-openai-api-key  # Tùy chọn, cho phân tích với GPT
```

## Thống số hiệu suất

- **Độ chính xác dự đoán lợi nhuận**: ~65-75%
- **Tỷ lệ thành công theo ngày**: ~60-70% đạt mục tiêu 3%
- **ROI hàng tháng dự kiến**: ~60-80% (compound)

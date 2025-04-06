import os
import json
import requests
from datetime import datetime, timedelta
import openai
import pandas as pd
import numpy as np

# Thiết lập API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

def fetch_crypto_news(symbols=["BTC", "ETH", "SOL"], days=1):
    """
    Lấy tin tức liên quan đến các loại tiền điện tử.
    Trong môi trường thực tế, hàm này sẽ gọi API tin tức.
    Hiện tại chúng ta sẽ giả lập dữ liệu tin tức.
    """
    # Danh sách tin tức giả lập
    current_date = datetime.now()
    news_list = []
    
    for i in range(days):
        date = current_date - timedelta(days=i)
        date_str = date.strftime("%Y-%m-%d")
        
        for symbol in symbols:
            # Tạo các tin tức giả cho mỗi symbol
            if symbol == "BTC":
                news_list.append({
                    "title": f"Bitcoin tăng giá mạnh sau khi các nhà đầu tư tổ chức mua vào - {date_str}",
                    "description": "Giá Bitcoin đã tăng 5% trong 24 giờ qua sau khi có thông tin về việc các quỹ đầu tư lớn gia tăng vị thế.",
                    "date": date_str,
                    "symbol": symbol
                })
                news_list.append({
                    "title": f"Chuyên gia dự đoán Bitcoin sẽ tiếp tục tăng trong ngắn hạn - {date_str}",
                    "description": "Các chuyên gia phân tích kỹ thuật nhận định BTC có thể sẽ đạt mức giá cao mới trong quý này.",
                    "date": date_str,
                    "symbol": symbol
                })
            elif symbol == "ETH":
                news_list.append({
                    "title": f"Ethereum hoàn thành cập nhật mạng lưới quan trọng - {date_str}",
                    "description": "Bản cập nhật mới của Ethereum giúp cải thiện tốc độ xử lý giao dịch và giảm phí gas.",
                    "date": date_str,
                    "symbol": symbol
                })
            elif symbol == "SOL":
                news_list.append({
                    "title": f"Solana gặp sự cố kỹ thuật khiến giá giảm - {date_str}",
                    "description": "Mạng lưới Solana đã gặp sự cố kỹ thuật ngắn hạn, khiến giá SOL giảm nhẹ trước khi phục hồi.",
                    "date": date_str,
                    "symbol": symbol
                })
    
    return news_list

def analyze_market_sentiment_with_ai(news_list):
    """
    Sử dụng OpenAI API để phân tích tâm lý thị trường dựa trên tin tức.
    Khi không có API key hoặc quota hết, sẽ sử dụng phân tích đơn giản dựa trên từ khóa.
    """
    if not news_list:
        return {"overall_sentiment": "neutral", "sentiment_score": 0.0, "symbol_sentiments": {}}
    
    # Chuẩn bị nội dung để gửi tới OpenAI
    # Tạo một chuỗi văn bản từ tin tức
    news_text = "\n\n".join([f"Date: {news['date']}, Symbol: {news['symbol']}, Title: {news['title']}, Content: {news['description']}" for news in news_list])
    
    try:
        # Kiểm tra nếu API key OpenAI có khả dụng
        if openai.api_key and openai.api_key != "":
            # Gọi OpenAI API để phân tích
            response = openai.chat.completions.create(
                model="gpt-4o",  # Sử dụng mô hình mới nhất
                messages=[
                    {"role": "system", "content": "Bạn là một chuyên gia phân tích thị trường tiền điện tử. Hãy phân tích những tin tức sau đây và đưa ra đánh giá về tâm lý thị trường. Trả lời dưới dạng JSON với các trường: overall_sentiment (positive/neutral/negative), sentiment_score (-1.0 đến 1.0), và symbol_sentiments (dict với key là ký hiệu tiền và value là sentiment_score cho tiền đó)."},
                    {"role": "user", "content": f"Phân tích tin tức tiền điện tử sau đây:\n\n{news_text}"}
                ],
                response_format={"type": "json_object"}
            )
            
            # Phân tích kết quả từ API
            sentiment_analysis = json.loads(response.choices[0].message.content)
            return sentiment_analysis
        else:
            # Phương pháp phân tích từ khóa đơn giản khi không có API key
            raise ValueError("API key không khả dụng, sử dụng phân tích từ khóa thay thế")
            
    except Exception as e:
        print(f"Lỗi khi phân tích tâm lý thị trường: {e}")
        
        # Phân tích bằng từ khóa đơn giản
        symbol_sentiments = {}
        overall_sentiment_score = 0.0
        sentiment_count = 0
        
        # Từ khóa tích cực
        positive_keywords = [
            "tăng", "tăng giá", "phục hồi", "tích cực", "mua vào", "cao", "thành công", 
            "triển vọng", "tốt", "cải thiện", "lạc quan", "bứt phá", "vượt trội"
        ]
        
        # Từ khóa tiêu cực
        negative_keywords = [
            "giảm", "giảm giá", "sụt giảm", "tiêu cực", "bán ra", "thấp", "thất bại", 
            "sụp đổ", "rủi ro", "kém", "bi quan", "sụt giảm", "sự cố", "khó khăn"
        ]
        
        # Phân tích từng bài báo
        for news in news_list:
            symbol = news['symbol']
            title = news['title'].lower()
            description = news['description'].lower()
            
            # Tính điểm tâm lý
            sentiment_score = 0.0
            
            # Kiểm tra từ khóa tích cực
            for keyword in positive_keywords:
                if keyword in title or keyword in description:
                    sentiment_score += 0.2
            
            # Kiểm tra từ khóa tiêu cực
            for keyword in negative_keywords:
                if keyword in title or keyword in description:
                    sentiment_score -= 0.2
            
            # Giới hạn điểm trong khoảng -1.0 đến 1.0
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # Cập nhật điểm cho symbol
            if symbol in symbol_sentiments:
                symbol_sentiments[symbol] = (symbol_sentiments[symbol] + sentiment_score) / 2
            else:
                symbol_sentiments[symbol] = sentiment_score
            
            # Cập nhật điểm tổng thể
            overall_sentiment_score += sentiment_score
            sentiment_count += 1
        
        # Tính điểm tâm lý trung bình
        if sentiment_count > 0:
            overall_sentiment_score /= sentiment_count
        
        # Xác định tâm lý tổng thể dựa trên điểm
        if overall_sentiment_score > 0.2:
            overall_sentiment = "positive"
        elif overall_sentiment_score < -0.2:
            overall_sentiment = "negative"
        else:
            overall_sentiment = "neutral"
        
        return {
            "overall_sentiment": overall_sentiment,
            "sentiment_score": overall_sentiment_score,
            "symbol_sentiments": symbol_sentiments
        }

def get_sentiment_signal(symbol):
    """
    Lấy tín hiệu giao dịch dựa trên phân tích tâm lý thị trường.
    Trả về giá trị từ -1 đến 1, với:
    - Giá trị dương: Tâm lý tích cực, có thể xem xét vị thế Long
    - Giá trị âm: Tâm lý tiêu cực, có thể xem xét vị thế Short
    - Giá trị gần 0: Tâm lý trung tính
    """
    # Lấy tin tức và phân tích
    symbol_short = symbol.split('/')[0] if '/' in symbol else symbol
    news = fetch_crypto_news(symbols=[symbol_short], days=1)
    sentiment = analyze_market_sentiment_with_ai(news)
    
    # Lấy điểm tâm lý cho symbol cụ thể
    symbol_sentiment = sentiment["symbol_sentiments"].get(symbol_short, sentiment["sentiment_score"])
    
    return symbol_sentiment

def integrate_sentiment_with_technical(dataframe, symbol):
    """
    Tích hợp phân tích tâm lý với phân tích kỹ thuật.
    """
    # Chỉ lấy tín hiệu tâm lý cho hàng cuối cùng để tránh gọi API quá nhiều
    # Trong ứng dụng thực tế, bạn có thể lưu cache tín hiệu này và cập nhật định kỳ
    sentiment_score = get_sentiment_signal(symbol)
    
    # Thêm cột tâm lý vào DataFrame
    dataframe.loc[dataframe.index[-1], 'sentiment_score'] = sentiment_score
    
    # Cũng có thể tạo cột tín hiệu tâm lý dựa trên điểm số
    if sentiment_score > 0.3:
        dataframe.loc[dataframe.index[-1], 'sentiment_signal'] = 1  # Tín hiệu Long
    elif sentiment_score < -0.3:
        dataframe.loc[dataframe.index[-1], 'sentiment_signal'] = -1  # Tín hiệu Short
    else:
        dataframe.loc[dataframe.index[-1], 'sentiment_signal'] = 0  # Trung tính
    
    return dataframe

if __name__ == "__main__":
    # Kiểm tra mô-đun
    news = fetch_crypto_news(["BTC", "ETH", "SOL"], days=1)
    sentiment = analyze_market_sentiment_with_ai(news)
    print(f"Phân tích tâm lý thị trường: {json.dumps(sentiment, indent=2, ensure_ascii=False)}")
    
    # Tạo DataFrame mẫu
    df = pd.DataFrame({
        'open': [50000, 51000, 52000],
        'high': [52000, 53000, 54000],
        'low': [49000, 50000, 51000],
        'close': [51000, 52000, 53000],
        'volume': [1000, 1100, 1200]
    })
    
    # Thêm tâm lý thị trường
    df = integrate_sentiment_with_technical(df, "BTC")
    print("\nDataFrame với tâm lý thị trường:")
    print(df)
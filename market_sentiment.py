"""
Module phân tích tâm lý thị trường dựa trên tin tức và mạng xã hội.
Sử dụng OpenAI API để phân tích tin tức và trích xuất tâm lý của thị trường.
"""

import os
import requests
import logging
import json
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import numpy as np

# Tải biến môi trường từ file .env
load_dotenv()

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("market_sentiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API Key OpenAI từ biến môi trường
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Keywords để phân tích tâm lý khi không có API
POSITIVE_KEYWORDS = [
    "bullish", "surge", "rally", "outperform", "growth", "uptrend", "buy", "gains",
    "positive", "optimistic", "breakthrough", "adoption", "partnership", "success",
    "soar", "climb", "rise", "jump", "strengthen", "upgrade", "recovery", "strong"
]

NEGATIVE_KEYWORDS = [
    "bearish", "crash", "decline", "dump", "plunge", "downtrend", "sell", "losses",
    "negative", "pessimistic", "setback", "ban", "regulation", "risk", "investigation",
    "tumble", "drop", "fall", "weak", "downgrade", "slump", "warning", "concern"
]

def fetch_crypto_news(symbols=["BTC", "ETH", "SOL"], days=1):
    """
    Lấy tin tức liên quan đến các loại tiền điện tử.
    Trong môi trường thực tế, hàm này sẽ gọi API tin tức như CryptoCompare, CoinAPI, hoặc NewsAPI.
    Hiện tại chúng ta sẽ giả lập dữ liệu tin tức.
    """
    logger.info(f"Fetching news for {symbols} for the past {days} days")
    
    # Trong môi trường thực tế, thay thế đoạn này bằng cuộc gọi API tin tức thực
    try:
        # Ví dụ API call (uncomment và chỉnh sửa khi sử dụng API thực)
        # url = "https://newsapi.org/v2/everything"
        # params = {
        #     "q": " OR ".join(symbols),
        #     "from": (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d"),
        #     "sortBy": "publishedAt",
        #     "language": "en",
        #     "apiKey": NEWS_API_KEY
        # }
        # response = requests.get(url, params=params)
        # news = response.json()['articles']
        
        # Thay vì sử dụng API thực tế, chúng ta sẽ giả lập dữ liệu tin tức
        # Điều này chỉ được sử dụng cho mục đích phát triển
        # Trong môi trường thực tế, đoạn code này sẽ được thay thế bằng cuộc gọi API tin tức thực
        
        # Số tin tức mẫu được tạo cho mỗi symbol
        sample_count = 3
        
        # Tạo dữ liệu tin tức ví dụ ngẫu nhiên cho mỗi symbol
        news_list = []
        
        logger.warning("Crypto news API not available. Using real-time news from API is recommended in production.")
        
        return []
        
    except Exception as e:
        logger.error(f"Error fetching crypto news: {e}")
        return []

def analyze_market_sentiment_with_ai(news_list):
    """
    Sử dụng OpenAI API để phân tích tâm lý thị trường dựa trên tin tức.
    Khi không có API key hoặc quota hết, sẽ sử dụng phân tích đơn giản dựa trên từ khóa.
    """
    if not news_list:
        logger.warning("No news data available for sentiment analysis")
        return 0, "Neutral"
    
    # Khi có API key OpenAI
    if OPENAI_API_KEY:
        try:
            # Chuẩn bị dữ liệu để gửi tới API
            news_text = "\n\n".join([f"Title: {news['title']}\nContent: {news['content']}" for news in news_list])
            
            # Tạo prompt cho phân tích tâm lý
            prompt = f"""
            Analyze the crypto market sentiment based on the following news articles.
            Rate the sentiment on a scale from -1.0 (extremely bearish) to 1.0 (extremely bullish).
            Please provide the numerical rating and a brief explanation.

            News Articles:
            {news_text}

            Format your response as a JSON object with 'score' (number) and 'explanation' (string) keys.
            """

            # Gọi OpenAI API
            import openai
            openai.api_key = OPENAI_API_KEY
            
            response = openai.Completion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial sentiment analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            
            # Phân tích kết quả
            result = json.loads(response.choices[0].message.content)
            score = float(result["score"])
            explanation = result["explanation"]
            
            logger.info(f"AI sentiment analysis completed: Score={score}, Explanation={explanation}")
            return score, explanation
            
        except Exception as e:
            logger.error(f"Error using OpenAI API for sentiment analysis: {e}")
            logger.info("Falling back to keyword-based sentiment analysis")
    else:
        logger.warning("OpenAI API key not available. Using keyword-based sentiment analysis.")
    
    # Phân tích đơn giản dựa trên từ khóa khi không có API
    return analyze_sentiment_with_keywords(news_list)

def analyze_sentiment_with_keywords(news_list):
    """
    Phân tích tâm lý thị trường dựa trên từ khóa khi không có API OpenAI.
    """
    if not news_list:
        return 0, "Neutral"
    
    positive_count = 0
    negative_count = 0
    
    # Đếm số lượng từ khóa tích cực và tiêu cực
    for news in news_list:
        text = (news['title'] + " " + news['content']).lower()
        
        for keyword in POSITIVE_KEYWORDS:
            if keyword.lower() in text:
                positive_count += 1
        
        for keyword in NEGATIVE_KEYWORDS:
            if keyword.lower() in text:
                negative_count += 1
    
    # Tính điểm tâm lý từ -1 đến 1
    total = positive_count + negative_count
    if total == 0:
        score = 0
    else:
        score = (positive_count - negative_count) / total
    
    # Xác định loại tâm lý
    if score > 0.5:
        sentiment = "Very Bullish"
    elif score > 0.2:
        sentiment = "Bullish"
    elif score > -0.2:
        sentiment = "Neutral"
    elif score > -0.5:
        sentiment = "Bearish"
    else:
        sentiment = "Very Bearish"
    
    logger.info(f"Keyword-based sentiment analysis: Score={score}, Sentiment={sentiment}")
    return score, sentiment

def get_sentiment_signal(symbol):
    """
    Lấy tín hiệu giao dịch dựa trên phân tích tâm lý thị trường.
    Trả về giá trị từ -1 đến 1, với:
    - Giá trị dương: Tâm lý tích cực, có thể xem xét vị thế Long
    - Giá trị âm: Tâm lý tiêu cực, có thể xem xét vị thế Short
    - Giá trị gần 0: Tâm lý trung tính
    """
    # Loại bỏ phần /USDT nếu có
    if '/' in symbol:
        symbol = symbol.split('/')[0]
    
    # Tránh gọi API quá nhanh
    cache_file = f"sentiment_cache_{symbol}.json"
    
    # Kiểm tra xem đã phân tích gần đây chưa (cache trong 60 phút)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Nếu dữ liệu cache còn mới (< 60 phút)
            if time.time() - cache_data['timestamp'] < 3600:
                logger.info(f"Using cached sentiment for {symbol}: {cache_data['score']}")
                return cache_data['score']
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
    
    # Lấy tin tức cho symbol
    news = fetch_crypto_news([symbol], days=1)
    
    # Phân tích tâm lý thị trường
    score, _ = analyze_market_sentiment_with_ai(news)
    
    # Lưu kết quả vào cache
    try:
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': time.time(),
                'score': score
            }, f)
    except Exception as e:
        logger.warning(f"Error writing cache: {e}")
    
    return score

def integrate_sentiment_with_technical(dataframe, symbol):
    """
    Tích hợp phân tích tâm lý với phân tích kỹ thuật.
    """
    try:
        # Lấy điểm tâm lý thị trường
        sentiment_score = get_sentiment_signal(symbol)
        
        # Thêm cột điểm tâm lý vào dataframe
        dataframe['sentiment_score'] = sentiment_score
        
        # Kết hợp tâm lý với các chỉ báo kỹ thuật
        # Ví dụ: điều chỉnh RSI dựa trên tâm lý
        if 'rsi' in dataframe.columns:
            # Điều chỉnh RSI dựa trên tâm lý (tăng RSI khi tâm lý tích cực và ngược lại)
            dataframe['rsi_adjusted'] = dataframe['rsi'] + (sentiment_score * 5)
            
            # Đảm bảo RSI điều chỉnh nằm trong khoảng 0-100
            dataframe['rsi_adjusted'] = np.clip(dataframe['rsi_adjusted'], 0, 100)
        
        logger.info(f"Added sentiment analysis for {symbol}: {sentiment_score}")
    except Exception as e:
        logger.error(f"Error integrating sentiment with technical indicators: {e}")
    
    return dataframe

if __name__ == "__main__":
    # Thử nghiệm module
    symbols = ["BTC", "ETH", "SOL"]
    
    for symbol in symbols:
        score = get_sentiment_signal(symbol)
        print(f"Sentiment for {symbol}: {score}")
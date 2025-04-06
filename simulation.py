#!/usr/bin/env python3
"""
Module giả lập để kiểm tra chiến lược giao dịch khi không có kết nối API Binance.
Tạo dữ liệu giả lập và mô phỏng cách chiến lược hoạt động.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import json
import random

# Thêm đường dẫn cha vào sys.path để import các module từ thư mục gốc
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from market_sentiment import get_sentiment_signal

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Định nghĩa các hàm chỉ báo kỹ thuật ---
def EMA(dataframe, timeperiod=20):
    """Exponential Moving Average"""
    return dataframe.ewm(span=timeperiod, min_periods=timeperiod).mean()

def SMA(series, timeperiod=5):
    """Simple Moving Average"""
    return series.rolling(timeperiod).mean()

def RSI(dataframe, timeperiod=14):
    """Relative Strength Index"""
    from ta.momentum import RSIIndicator
    return RSIIndicator(close=dataframe, window=timeperiod).rsi()

def ATR(dataframe, timeperiod=14):
    """Average True Range"""
    from ta.volatility import AverageTrueRange
    return AverageTrueRange(high=dataframe['high'], 
                           low=dataframe['low'], 
                           close=dataframe['close'], 
                           window=timeperiod).average_true_range()

def BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Bollinger Bands"""
    from ta.volatility import BollingerBands
    bollinger = BollingerBands(close=dataframe['close'], window=timeperiod, window_dev=nbdevup)
    return {
        'upper': bollinger.bollinger_hband(),
        'mid': bollinger.bollinger_mavg(),
        'lower': bollinger.bollinger_lband()
    }

def MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    from ta.trend import MACD as ta_MACD
    macd_indicator = ta_MACD(close=dataframe['close'], 
                             window_slow=slowperiod, 
                             window_fast=fastperiod, 
                             window_sign=signalperiod)
    return {
        'macd': macd_indicator.macd(),
        'signal': macd_indicator.macd_signal(),
        'histogram': macd_indicator.macd_diff()
    }

def STOCH(dataframe, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    from ta.momentum import StochasticOscillator
    stoch = StochasticOscillator(high=dataframe['high'], 
                                low=dataframe['low'], 
                                close=dataframe['close'], 
                                window=fastk_period, 
                                smooth_window=slowk_period)
    return {
        'fastk': stoch.stoch(),
        'fastd': stoch.stoch_signal()
    }

def MFI(dataframe, timeperiod=14):
    """Money Flow Index"""
    from ta.volume import MFIIndicator
    return MFIIndicator(high=dataframe['high'], 
                        low=dataframe['low'], 
                        close=dataframe['close'], 
                        volume=dataframe['volume'], 
                        window=timeperiod).money_flow_index()

def generate_mock_data(symbol='BTC/USDT', timeframe='5m', days=7):
    """
    Tạo dữ liệu giả lập cho việc kiểm tra chiến lược
    """
    logger.info(f"Tạo dữ liệu giả lập cho {symbol} trong {days} ngày với khung thời gian {timeframe}")
    
    # Xác định số lượng nến dựa trên timeframe và số ngày
    if timeframe == '5m':
        candles_per_day = 24 * 12  # 12 nến mỗi giờ, 24 giờ mỗi ngày
    elif timeframe == '15m':
        candles_per_day = 24 * 4
    elif timeframe == '1h':
        candles_per_day = 24
    elif timeframe == '4h':
        candles_per_day = 6
    elif timeframe == '1d':
        candles_per_day = 1
    else:
        candles_per_day = 24 * 12  # Mặc định 5m
    
    num_candles = days * candles_per_day
    
    # Xác định giá ban đầu dựa trên symbol
    if 'BTC' in symbol:
        initial_price = 40000.0
        volatility = 0.015  # 1.5% biến động
    elif 'ETH' in symbol:
        initial_price = 2000.0
        volatility = 0.02  # 2% biến động
    elif 'SOL' in symbol:
        initial_price = 100.0
        volatility = 0.025  # 2.5% biến động
    else:
        initial_price = 100.0
        volatility = 0.02  # 2% biến động
    
    # Tạo chuỗi thời gian
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    # Tạo chuỗi thời gian với số lượng điểm tương ứng
    timestamps = []
    current_time = start_time
    
    for _ in range(num_candles):
        timestamps.append(current_time)
        if timeframe == '5m':
            current_time += timedelta(minutes=5)
        elif timeframe == '15m':
            current_time += timedelta(minutes=15)
        elif timeframe == '1h':
            current_time += timedelta(hours=1)
        elif timeframe == '4h':
            current_time += timedelta(hours=4)
        elif timeframe == '1d':
            current_time += timedelta(days=1)
    
    # Tạo mô hình giá theo phương pháp Random Walk với xu hướng
    # Sử dụng các mô hình giá thực tế hơn, với xu hướng và chu kỳ
    
    # Tham số mô hình
    trend_strength = 0.0003  # Giá trị dương cho xu hướng tăng, âm cho xu hướng giảm
    cycle_period = num_candles // 3  # Chu kỳ khoảng 1/3 tổng số nến
    cycle_amplitude = 0.05  # Biên độ của chu kỳ (5%)
    
    # Tạo mô hình giá
    prices = [initial_price]
    
    for i in range(1, num_candles):
        # Thành phần xu hướng
        trend_component = prices[-1] * trend_strength
        
        # Thành phần chu kỳ
        cycle_component = prices[-1] * cycle_amplitude * np.sin(2 * np.pi * i / cycle_period)
        
        # Thành phần ngẫu nhiên
        random_component = prices[-1] * volatility * np.random.normal()
        
        # Tính giá mới
        new_price = prices[-1] + trend_component + cycle_component + random_component
        
        # Đảm bảo giá không âm
        new_price = max(new_price, 0.01 * initial_price)  # Giá không giảm quá 99%
        
        prices.append(new_price)
    
    # Tạo dữ liệu OHLCV (Open, High, Low, Close, Volume)
    data = []
    
    for i in range(num_candles):
        close_price = prices[i]
        
        # Tính open, high, low
        if i > 0:
            open_price = prices[i-1]
        else:
            open_price = close_price * (1 - 0.005 * np.random.random())
        
        high_price = max(close_price, open_price) * (1 + 0.01 * np.random.random())
        low_price = min(close_price, open_price) * (1 - 0.01 * np.random.random())
        
        # Tính volume (tương quan với biến động giá)
        price_change = abs(close_price - open_price) / open_price
        volume = initial_price * 10 * (1 + 10 * price_change) * (0.5 + np.random.random())
        
        data.append([timestamps[i], open_price, high_price, low_price, close_price, volume])
    
    # Tạo DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Thêm symbol vào DataFrame
    df['symbol'] = symbol
    
    logger.info(f"Đã tạo {len(df)} dòng dữ liệu giả lập")
    
    return df

def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame
    """
    logger.info("Thêm các chỉ báo kỹ thuật...")
    
    # Đảm bảo rằng df được sắp xếp theo thời gian
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 1. RSI
    df['rsi'] = RSI(df['close'], timeperiod=14)
    
    # 2. MACD
    macd = MACD(df)
    df['macd'] = macd['macd']
    df['macdsignal'] = macd['signal']
    df['macdhist'] = macd['histogram']
    
    # 3. Bollinger Bands
    bollinger = BBANDS(df)
    df['bb_upper'] = bollinger['upper']
    df['bb_mid'] = bollinger['mid']
    df['bb_lower'] = bollinger['lower']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    
    # 4. Average True Range (ATR)
    df['atr'] = ATR(df)
    
    # 5. EMA
    df['ema_9'] = EMA(df['close'], 9)
    df['ema_21'] = EMA(df['close'], 21)
    df['ema_50'] = EMA(df['close'], 50)
    df['ema_200'] = EMA(df['close'], 200)
    
    # 6. SMA
    df['sma_50'] = SMA(df['close'], 50)
    df['sma_200'] = SMA(df['close'], 200)
    
    # 7. Stochastic Oscillator
    stoch = STOCH(df)
    df['stoch_k'] = stoch['fastk']
    df['stoch_d'] = stoch['fastd']
    
    # 8. Money Flow Index
    df['mfi'] = MFI(df)
    
    # 9. Price distance from EMA
    df['distance_ema_200'] = ((df['close'] - df['ema_200']) / df['close']) * 100
    
    # 10. Volume Rate of Change
    df['volume_mean'] = df['volume'].rolling(window=20).mean()
    df['volume_roc'] = (df['volume'] / df['volume_mean'])
    
    # 11. Trend Direction (1=Uptrend, -1=Downtrend, 0=Sideways)
    df['trend'] = 0
    df.loc[(df['ema_50'] > df['sma_200']) & 
           (df['close'] > df['ema_50']), 'trend'] = 1
    df.loc[(df['ema_50'] < df['sma_200']) & 
           (df['close'] < df['ema_50']), 'trend'] = -1
    
    # 12. Volatility
    df['volatility'] = df['atr'] / df['close'] * 100
    
    # 13. Candle Patterns
    df['body'] = abs(df['open'] - df['close'])
    df['wick_high'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['wick_low'] = df[['open', 'close']].min(axis=1) - df['low']
    
    df['is_hammer'] = 0
    df.loc[(df['body'] > 0) &
           (df['wick_low'] >= 2 * df['body']) &
           (df['wick_high'] <= 0.2 * df['wick_low']), 'is_hammer'] = 1
    
    df['is_shooting_star'] = 0
    df.loc[(df['body'] > 0) &
           (df['wick_high'] >= 2 * df['body']) &
           (df['wick_low'] <= 0.2 * df['wick_high']), 'is_shooting_star'] = 1
    
    # 14. Thêm phân tích tâm lý thị trường
    try:
        symbol = df['symbol'].iloc[0].split('/')[0]
        sentiment_score = get_sentiment_signal(symbol)
        df['sentiment_score'] = sentiment_score
        logger.info(f"Đã thêm điểm tâm lý thị trường: {sentiment_score}")
    except Exception as e:
        logger.warning(f"Không thể thêm phân tích tâm lý thị trường: {e}")
        df['sentiment_score'] = 0
    
    # Xử lý các giá trị NaN và vô cùng
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    logger.info(f"Đã thêm các chỉ báo kỹ thuật. Kích thước DataFrame: {df.shape}")
    
    return df

def simulate_strategy(df, initial_capital=10000.0, leverage=3.0):
    """
    Mô phỏng chiến lược giao dịch trên dữ liệu giả lập
    """
    logger.info(f"Bắt đầu mô phỏng chiến lược với vốn ban đầu {initial_capital}$ và đòn bẩy {leverage}x")
    
    # Thiết lập các tham số chiến lược
    buy_rsi_threshold = 30  # Mua khi RSI dưới ngưỡng này
    sell_rsi_threshold = 70  # Bán khi RSI trên ngưỡng này
    stoploss_percentage = 0.02  # Cắt lỗ ở 2%
    risk_per_trade = 0.05  # Rủi ro 5% vốn cho mỗi giao dịch
    
    # Khởi tạo danh mục đầu tư
    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trade_history = []
    equity_curve = [initial_capital]
    
    # Mô phỏng giao dịch
    for i in range(1, len(df)):
        current_row = df.iloc[i]
        previous_row = df.iloc[i-1]
        
        # Tính toán các tín hiệu giao dịch
        buy_signal = False
        sell_signal = False
        
        # Tín hiệu Long dựa trên chiến lược
        if position == 0:
            # RSI quá bán
            rsi_buy = current_row['rsi'] < buy_rsi_threshold and previous_row['rsi'] < buy_rsi_threshold
            
            # Giá dưới Bollinger Band Dưới
            bb_buy = current_row['close'] < current_row['bb_lower']
            
            # Xu hướng tăng
            trend_buy = current_row['ema_50'] > current_row['sma_200']
            
            # MACD Histogram đổi chiều
            macd_buy = (current_row['macdhist'] > 0 and previous_row['macdhist'] < 0)
            
            # Tâm lý thị trường tích cực
            sentiment_buy = current_row['sentiment_score'] > 0.3
            
            # Kết hợp các tín hiệu
            buy_signal = (rsi_buy or bb_buy) and (trend_buy or macd_buy or sentiment_buy)
        
        # Tín hiệu thoát lệnh Long
        if position > 0:
            # RSI quá mua
            rsi_sell = current_row['rsi'] > sell_rsi_threshold
            
            # Giá trên Bollinger Band Trên
            bb_sell = current_row['close'] > current_row['bb_upper']
            
            # MACD Histogram đảo chiều giảm
            macd_sell = (current_row['macdhist'] < 0 and previous_row['macdhist'] > 0)
            
            # Giá giảm quá mức cắt lỗ
            stoploss_hit = current_row['low'] < entry_price * (1 - stoploss_percentage)
            
            # Tâm lý thị trường trở nên tiêu cực
            sentiment_sell = current_row['sentiment_score'] < -0.3
            
            # Kết hợp các tín hiệu
            sell_signal = rsi_sell or bb_sell or macd_sell or stoploss_hit or sentiment_sell
        
        # Thực hiện giao dịch
        if buy_signal and position == 0:
            # Tính toán kích thước vị thế dựa trên rủi ro
            position_size = (capital * risk_per_trade) / (current_row['close'] * (stoploss_percentage / leverage))
            position = position_size * leverage
            entry_price = current_row['close']
            
            trade_history.append({
                'timestamp': current_row['timestamp'],
                'action': 'BUY',
                'price': entry_price,
                'position': position,
                'capital': capital
            })
            
            logger.info(f"MUA tại {entry_price}$ với vị thế {position} và vốn {capital}$")
        
        elif sell_signal and position > 0:
            # Tính lợi nhuận/lỗ
            exit_price = current_row['close']
            pnl = position * (exit_price - entry_price)
            capital += pnl
            
            trade_history.append({
                'timestamp': current_row['timestamp'],
                'action': 'SELL',
                'price': exit_price,
                'position': position,
                'pnl': pnl,
                'capital': capital
            })
            
            logger.info(f"BÁN tại {exit_price}$ với lợi nhuận {pnl}$ và vốn {capital}$")
            
            # Reset vị thế
            position = 0.0
            entry_price = 0.0
        
        # Cập nhật equity curve
        if position > 0:
            current_value = capital + position * (current_row['close'] - entry_price)
        else:
            current_value = capital
        
        equity_curve.append(current_value)
    
    # Tính toán các chỉ số hiệu suất
    returns = (capital - initial_capital) / initial_capital * 100
    winning_trades = len([t for t in trade_history if t.get('pnl', 0) > 0])
    losing_trades = len([t for t in trade_history if t.get('pnl', 0) < 0])
    total_trades = winning_trades + losing_trades
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # Tính max drawdown
    peak = equity_curve[0]
    max_drawdown = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    logger.info(f"Mô phỏng hoàn thành!")
    logger.info(f"Vốn ban đầu: {initial_capital}$")
    logger.info(f"Vốn cuối: {capital}$")
    logger.info(f"Lợi nhuận: {returns:.2f}%")
    logger.info(f"Số giao dịch: {total_trades}")
    logger.info(f"Thắng: {winning_trades}, Thua: {losing_trades}")
    logger.info(f"Tỷ lệ thắng: {win_rate:.2f}%")
    logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Kết quả
    results = {
        'initial_capital': initial_capital,
        'final_capital': capital,
        'returns': returns,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'equity_curve': equity_curve,
        'trade_history': trade_history
    }
    
    return results

def visualize_results(df, results):
    """
    Tạo hình ảnh kết quả mô phỏng
    """
    logger.info("Tạo hình ảnh kết quả mô phỏng...")
    
    # Tạo bố cục figure
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Biểu đồ giá và giao dịch
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=1)
    ax1.set_title('Biểu đồ giá và giao dịch')
    ax1.plot(df['timestamp'], df['close'], label='Giá đóng cửa')
    ax1.plot(df['timestamp'], df['ema_50'], label='EMA 50', linestyle='--', alpha=0.7)
    ax1.plot(df['timestamp'], df['sma_200'], label='SMA 200', linestyle='--', alpha=0.7)
    
    # Thêm các điểm giao dịch
    for trade in results['trade_history']:
        if trade['action'] == 'BUY':
            ax1.scatter(trade['timestamp'], trade['price'], color='green', s=100, marker='^')
        else:
            ax1.scatter(trade['timestamp'], trade['price'], color='red', s=100, marker='v')
    
    ax1.set_ylabel('Giá')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Equity Curve
    ax2 = plt.subplot2grid((3, 1), (1, 0), rowspan=1)
    ax2.set_title('Equity Curve')
    ax2.plot(range(len(results['equity_curve'])), results['equity_curve'], label='Equity')
    ax2.set_xlabel('Số nến')
    ax2.set_ylabel('Vốn ($)')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Chỉ báo
    ax3 = plt.subplot2grid((3, 1), (2, 0), rowspan=1)
    ax3.set_title('RSI và MACD')
    ax3.plot(df['timestamp'], df['rsi'], label='RSI', color='purple')
    ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7)
    ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7)
    ax3.set_ylabel('RSI')
    ax3.legend(loc='upper left')
    ax3.grid(True)
    
    # MACD trên trục y thứ hai
    ax3b = ax3.twinx()
    ax3b.plot(df['timestamp'], df['macd'], label='MACD', color='blue', alpha=0.7)
    ax3b.plot(df['timestamp'], df['macdsignal'], label='Signal', color='orange', alpha=0.7)
    ax3b.bar(df['timestamp'], df['macdhist'], label='Histogram', color='gray', alpha=0.3)
    ax3b.set_ylabel('MACD')
    ax3b.legend(loc='upper right')
    
    # Thông tin tổng quan
    plt.figtext(0.01, 0.01, f"Vốn ban đầu: {results['initial_capital']}$, Vốn cuối: {results['final_capital']:.2f}$\n"
               f"Lợi nhuận: {results['returns']:.2f}%, Max Drawdown: {results['max_drawdown']:.2f}%\n"
               f"Tổng giao dịch: {results['total_trades']}, Tỷ lệ thắng: {results['win_rate']:.2f}%",
               fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    
    # Lưu hình
    output_file = "simulation_results.png"
    plt.savefig(output_file, dpi=100)
    logger.info(f"Đã lưu kết quả mô phỏng vào {output_file}")
    
    # Hiển thị hình (bỏ comment dòng dưới nếu muốn hiển thị)
    # plt.show()
    
    return output_file

def run_strategy_simulation():
    """
    Giả lập việc chạy chiến lược trên dữ liệu mô phỏng
    """
    logger.info("Bắt đầu giả lập chiến lược giao dịch...")
    
    try:
        # Tạo dữ liệu giả lập
        df = generate_mock_data(symbol='BTC/USDT', timeframe='5m', days=14)
        
        # Thêm các chỉ báo kỹ thuật
        df = add_technical_indicators(df)
        
        # Mô phỏng chiến lược
        results = simulate_strategy(df, initial_capital=10000.0, leverage=3.0)
        
        # Tạo hình ảnh kết quả
        output_file = visualize_results(df, results)
        
        return {
            'success': True,
            'output_file': output_file,
            'results': results
        }
    
    except Exception as e:
        logger.error(f"Lỗi khi chạy giả lập: {e}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == "__main__":
    # Chạy giả lập
    result = run_strategy_simulation()
    
    if result['success']:
        print(f"Giả lập thành công. Kết quả được lưu tại: {result['output_file']}")
        print(f"Lợi nhuận: {result['results']['returns']:.2f}%")
        print(f"Tỷ lệ thắng: {result['results']['win_rate']:.2f}%")
    else:
        print(f"Giả lập thất bại: {result['error']}")
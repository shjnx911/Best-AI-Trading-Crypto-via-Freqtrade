import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from user_data.strategies.enhanced_crypto_strategy import EnhancedCryptoStrategy

# Hàm giả lập tạo dữ liệu thay vì sử dụng API Binance
def generate_mock_data(symbol='BTC/USDT', timeframe='5m', days=7):
    """
    Tạo dữ liệu giả lập cho việc kiểm tra chiến lược
    """
    # Số lượng dòng dữ liệu (nến)
    n_points = days * 24 * 60 // 5  # 5 phút mỗi nến

    # Tạo ngày giờ
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start=start_time, end=end_time, periods=n_points)

    # Tạo dữ liệu giá ngẫu nhiên nhưng có xu hướng
    # Bắt đầu với giá cơ bản
    base_price = 50000.0 if 'BTC' in symbol else 2000.0 if 'ETH' in symbol else 100.0
    
    # Thêm xu hướng và biến động
    trend = np.cumsum(np.random.normal(0, 1, n_points)) * 0.01 * base_price
    volatility = np.random.normal(0, 1, n_points) * 0.005 * base_price
    
    # Tạo giá đóng cửa
    close = base_price + trend + volatility
    
    # Tạo giá cao hơn và thấp hơn dựa trên đóng cửa
    high = close + np.abs(np.random.normal(0, 1, n_points)) * 0.003 * base_price
    low = close - np.abs(np.random.normal(0, 1, n_points)) * 0.003 * base_price
    
    # Tạo giá mở cửa dựa trên giá đóng cửa trước đó
    open_price = np.zeros(n_points)
    open_price[0] = close[0] * 0.999  # Giá mở cửa đầu tiên
    open_price[1:] = close[:-1]

    # Tạo khối lượng
    volume = np.random.normal(500, 100, n_points) * (1 + np.abs(trend/base_price))
    
    # Tạo DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    })
    
    return df

def run_strategy_simulation():
    """
    Giả lập việc chạy chiến lược trên dữ liệu mô phỏng
    """
    print("Bắt đầu giả lập chiến lược giao dịch...")
    
    # Tạo các thư mục cần thiết
    os.makedirs("user_data/models", exist_ok=True)
    
    # Danh sách các cặp tiền
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Giả lập cấu hình
    config = {
        'timeframe': '5m',
        'stake_currency': 'USDT',
        'dry_run': True,
        'max_open_trades': 3,
        'stake_amount': '100',
        'tradable_balance_ratio': 0.99,
    }
    
    # Tạo instance của chiến lược
    strategy = EnhancedCryptoStrategy(config)
    
    results = {}
    
    for symbol in symbols:
        print(f"Giả lập dữ liệu cho {symbol}...")
        df = generate_mock_data(symbol=symbol, days=7)
        
        print(f"Áp dụng chiến lược cho {symbol}...")
        # Áp dụng các chỉ báo
        df = strategy.populate_indicators(df, {'pair': symbol})
        
        # Thêm tín hiệu giao dịch
        df = strategy.populate_entry_trend(df, {'pair': symbol})
        df = strategy.populate_exit_trend(df, {'pair': symbol})
        
        # Giả lập giao dịch
        balance = 1000.0  # USDT
        position_open = False
        entry_price = 0.0
        trades = []
        trade_count = 0
        
        print(f"Giả lập giao dịch cho {symbol}...")
        for i in range(1, len(df)):
            # Kiểm tra tín hiệu vào lệnh (Long)
            if not position_open and 'enter_long' in df.columns and df['enter_long'].iloc[i]:
                position_open = True
                entry_price = df['close'].iloc[i]
                trade = {
                    'pair': symbol,
                    'entry_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'type': 'long'
                }
                print(f"MUA {symbol} tại {entry_price} vào {df['timestamp'].iloc[i]}")
            
            # Kiểm tra tín hiệu vào lệnh (Short)
            elif not position_open and 'enter_short' in df.columns and df['enter_short'].iloc[i]:
                position_open = True
                entry_price = df['close'].iloc[i]
                trade = {
                    'pair': symbol,
                    'entry_time': df['timestamp'].iloc[i],
                    'entry_price': entry_price,
                    'type': 'short'
                }
                print(f"BÁN KHỐNG {symbol} tại {entry_price} vào {df['timestamp'].iloc[i]}")
            
            # Kiểm tra tín hiệu thoát lệnh (Long)
            elif position_open and trade['type'] == 'long' and 'exit_long' in df.columns and df['exit_long'].iloc[i]:
                exit_price = df['close'].iloc[i]
                profit_pct = (exit_price - entry_price) / entry_price * 100
                balance = balance * (1 + profit_pct/100)
                position_open = False
                
                trade['exit_time'] = df['timestamp'].iloc[i]
                trade['exit_price'] = exit_price
                trade['profit_pct'] = profit_pct
                trades.append(trade.copy())
                trade_count += 1
                print(f"ĐÓNG LONG {symbol} tại {exit_price} vào {df['timestamp'].iloc[i]}, lợi nhuận: {profit_pct:.2f}%")
            
            # Kiểm tra tín hiệu thoát lệnh (Short)
            elif position_open and trade['type'] == 'short' and 'exit_short' in df.columns and df['exit_short'].iloc[i]:
                exit_price = df['close'].iloc[i]
                profit_pct = (entry_price - exit_price) / entry_price * 100
                balance = balance * (1 + profit_pct/100)
                position_open = False
                
                trade['exit_time'] = df['timestamp'].iloc[i]
                trade['exit_price'] = exit_price
                trade['profit_pct'] = profit_pct
                trades.append(trade.copy())
                trade_count += 1
                print(f"ĐÓNG SHORT {symbol} tại {exit_price} vào {df['timestamp'].iloc[i]}, lợi nhuận: {profit_pct:.2f}%")
        
        # Tính toán kết quả
        total_profit = sum(trade['profit_pct'] for trade in trades) if trades else 0
        avg_profit = total_profit / len(trades) if trades else 0
        
        results[symbol] = {
            'trades': trades,
            'trade_count': trade_count,
            'total_profit_pct': total_profit,
            'avg_profit_pct': avg_profit,
            'final_balance': balance
        }
        
        print(f"Kết quả {symbol}: {trade_count} giao dịch, lợi nhuận {total_profit:.2f}%, số dư cuối {balance:.2f} USDT")
    
    # Tạo biểu đồ kết quả
    plt.figure(figsize=(12, 6))
    
    # Biểu đồ lợi nhuận trên từng cặp
    plt.subplot(1, 2, 1)
    pairs = list(results.keys())
    profits = [results[pair]['total_profit_pct'] for pair in pairs]
    plt.bar(pairs, profits)
    plt.title('Tổng lợi nhuận (%)')
    plt.ylabel('Lợi nhuận (%)')
    
    # Biểu đồ tăng trưởng vốn
    plt.subplot(1, 2, 2)
    balances = [results[pair]['final_balance'] for pair in pairs]
    plt.bar(pairs, balances)
    plt.axhline(y=1000, color='r', linestyle='--')
    plt.title('Số dư cuối (USDT)')
    plt.ylabel('USDT')
    
    plt.tight_layout()
    plt.savefig('simulation_results.png')
    
    # Tóm tắt kết quả
    total_trades = sum(results[pair]['trade_count'] for pair in pairs)
    avg_total_profit = sum(profits) / len(profits) if profits else 0
    
    print("\n=== TÓM TẮT KẾT QUẢ GIẢ LẬP ===")
    print(f"Tổng số giao dịch: {total_trades}")
    print(f"Lợi nhuận trung bình mỗi cặp: {avg_total_profit:.2f}%")
    print(f"Biểu đồ kết quả đã được lưu vào 'simulation_results.png'")

if __name__ == "__main__":
    run_strategy_simulation()
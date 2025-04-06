import os

# Tạo file .replit để chạy simulation.py
with open('.replit', 'w') as f:
    f.write('run = "python simulation.py"\n')
    f.write('language = "python"\n')
    
print("Đã cấu hình workflow để chạy simulation.py")
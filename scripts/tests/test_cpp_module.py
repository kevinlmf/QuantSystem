import sys
import os

# 添加 .so 文件路径（即 build/ 目录）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../cpp_core/build")))

import cpp_trading  # 导入 pybind11 模块

# 初始化并加载数据
feed = cpp_trading.DataFeed()
success = feed.load("data/SPY_1d.csv")

if not success:
    print("❌ Failed to load data.")
    exit()

# 遍历前 5 条数据看看
print("✅ Loaded. Displaying first 5 rows:")
for _ in range(5):
    if not feed.next():
        break
    row = feed.current()
    print(f"{row.date}: Open={row.open}, High={row.high}, Low={row.low}, Close={row.close}, Vol={row.volume}")

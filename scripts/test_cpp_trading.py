import sys
import os

# Add cpp_core/build directory to sys.path
sys.path.append(os.path.abspath("cpp_core/build"))

from cpp_trading import DataFeed

# Initialize and load data
feed = DataFeed()
if not feed.load("data/SPY_1d.csv"):
    print("❌ Failed to load data")
    exit(1)

print("✅ Data loaded successfully")

while feed.next():
    row = feed.current()
    print(f"{row.date}: Close={row.close}")

# Compute moving average
ma = feed.moving_average(5)
print(f"5-period moving average (first 5): {ma[:5]}")

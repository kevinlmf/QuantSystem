import sys
import os

sys.path.append(os.path.abspath("cpp_core/build/lib.macosx-14.5-arm64-cpython-310"))

from cpp_trading2 import DataFeed, Order, OrderExecutor, OrderType





# Initialize modules
feed = DataFeed()
executor = OrderExecutor()

if not feed.load("data/SPY_1d.csv"):
    print("❌ Failed to load data")
    exit(1)

print("✅ Data loaded successfully")

# Simple strategy: Buy every 10th row
step = 0
while feed.next():
    row = feed.current()
    print(f"{row.date}: Close={row.close}")

    if step % 10 == 0:
        order = Order()
        order.symbol = "SPY"
        order.type = OrderType.BUY
        order.price = row.close
        order.quantity = 1
        order.timestamp = step
        executor.submit_order(order)
        print(f"📥 Submitted order: BUY 1 SPY @ {row.close}")

    step += 1

# Simulate execution
executor.simulate_execution()

# Print filled orders
print("\n✅ Filled Orders:")
for filled in executor.get_filled_orders():
    print(f"BUY {filled.quantity} {filled.symbol} @ ${filled.price}")

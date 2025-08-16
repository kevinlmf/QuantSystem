# scripts/test_order_executor.py
import sys
import platform
from pathlib import Path
import pytest

# ========== 路径 & 动态库检测 ==========
project_root = Path(__file__).resolve().parents[1]
build_dir = project_root / "cpp_core" / "build"

if platform.system() == "Windows":
    lib_ext = ".pyd"
else:  # macOS / Linux
    lib_ext = ".so"

# 搜索动态库
found_lib = None
for file in build_dir.rglob(f"*{lib_ext}"):
    found_lib = file
    break

if not found_lib:
    pytest.skip(f"未找到动态库文件（{lib_ext}），跳过 OrderExecutor 测试", allow_module_level=True)

# 把动态库目录加到 sys.path
sys.path.insert(0, str(found_lib.parent))

# ========== 导入 C++ 模块 ==========
try:
    from cpp_trading import Order, OrderExecutor, OrderType
except ImportError:
    from cpp_trading2 import Order, OrderExecutor, OrderType

# ========== 测试用例 ==========
def test_order_executor_basic_flow():
    """测试下单与执行流程"""
    executor = OrderExecutor()

    # 创建订单
    order = Order()
    order.symbol = "AAPL"
    order.type = OrderType.BUY
    order.price = 180.5
    order.quantity = 10
    order.timestamp = 1680000000

    # 提交订单
    executor.submit_order(order)

    # 模拟执行
    executor.simulate_execution()

    # 检查是否有成交订单
    filled_orders = executor.get_filled_orders()
    assert len(filled_orders) > 0, "执行后应有成交订单"

    first_filled = filled_orders[0]
    assert first_filled.symbol == "AAPL"
    assert first_filled.type == OrderType.BUY
    assert first_filled.quantity == 10
    assert abs(first_filled.price - 180.5) < 1e-6

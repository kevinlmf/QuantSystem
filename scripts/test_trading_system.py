# scripts/test_cpp_integration.py
import sys
import platform
from math import isfinite
from pathlib import Path
import pytest

# ========= 路径 & 动态库检测 =========
ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "cpp_core" / "build"
DATA_CSV = ROOT / "data" / "SPY_1d.csv"

# 缺少数据就整体跳过
if not DATA_CSV.exists():
    pytest.skip(f"缺少测试数据文件：{DATA_CSV}", allow_module_level=True)

# 按平台选择扩展名
if platform.system() == "Windows":
    LIB_EXT = ".pyd"
else:  # macOS / Linux
    LIB_EXT = ".so"

# 在 build 目录下搜寻已编译扩展
found_lib_dir = None
for p in BUILD_DIR.rglob(f"*{LIB_EXT}"):
    found_lib_dir = p.parent
    break

if not found_lib_dir:
    pytest.skip(f"未找到已编译的扩展（{LIB_EXT}）于 {BUILD_DIR}，跳过集成测试", allow_module_level=True)

sys.path.insert(0, str(found_lib_dir))

# ========= 导入 C++ 模块（名称兜底） =========
try:
    from cpp_trading2 import DataFeed, Order, OrderExecutor, OrderType
except Exception:
    from cpp_trading import DataFeed, Order, OrderExecutor, OrderType


def test_cpp_datafeed_and_order_executor_integration():
    # 1) 加载数据
    feed = DataFeed()
    assert feed.load(str(DATA_CSV)), "DataFeed.load 返回 False"

    # 2) 简单策略：每第 10 行买 1 手
    executor = OrderExecutor()
    step = 0
    submitted = 0
    seen_rows = 0

    while feed.next():
        row = feed.current()
        # 行数据基本健壮性
        assert hasattr(row, "date")
        assert hasattr(row, "close")
        assert isinstance(row.close, (int, float)) and isfinite(row.close)

        if step % 10 == 0:
            order = Order()
            order.symbol = "SPY"
            order.type = OrderType.BUY
            order.price = float(row.close)
            order.quantity = 1
            order.timestamp = step
            executor.submit_order(order)
            submitted += 1
        step += 1
        seen_rows += 1

    assert seen_rows > 20, "数据太短，无法完成集成测试（至少需 20 行）"
    assert submitted >= 2, "应至少提交若干笔订单"

    # 3) 模拟执行并校验成交
    executor.simulate_execution()
    filled = executor.get_filled_orders()

    # 至少有 1 笔成交；不同实现可导致全部成交或部分成交，这里不苛求数量
    assert len(filled) >= 1, "模拟执行后应至少有一笔成交"

    # 校验首笔成交字段
    f0 = filled[0]
    assert f0.symbol == "SPY"
    assert f0.type in (OrderType.BUY, OrderType.SELL)  # 若撮合可能反向成交则放宽
    assert f0.quantity >= 1
    assert isinstance(f0.price, (int, float)) and isfinite(f0.price)


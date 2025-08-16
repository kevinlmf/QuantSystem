# scripts/test_cpp_trading.py
import sys
import os
import math
import csv
from pathlib import Path
import pytest

# ============== 路径设置 ==============
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]   # <repo>/
CPP_CORE = PROJECT_ROOT / "cpp_core"
DATA_CSV = PROJECT_ROOT / "data" / "SPY_1d.csv"

# 确保能 import 到 C++ 扩展模块
sys.path.insert(0, str(CPP_CORE))

# 如果数据文件不存在，就跳过整文件测试（更友好）
pytestmark = pytest.mark.skipif(
    not DATA_CSV.exists(),
    reason=f"数据文件不存在：{DATA_CSV}"
)

# ============== 兼容导入 ==============
def _import_datafeed():
    try:
        from cpp_trading2 import DataFeed  # 首选
        return DataFeed
    except Exception:
        from cpp_trading import DataFeed   # 兜底
        return DataFeed

# ============== 工具函数 ==============
def read_first_n_closes(csv_path: Path, n: int):
    """
    读取 CSV 的前 n 个【数值型】收盘价。
    - 自动跳过非数值行（如 Symbol,SPY）
    - 兼容列名：Close/close/CLOSE/Adj Close/adj_close/ADJ_CLOSE
    """
    def try_parse_float(x):
        try:
            return float(str(x).replace(",", "").strip())
        except Exception:
            return None

    close_keys = ("Close", "close", "CLOSE", "Adj Close", "adj_close", "ADJ_CLOSE")
    closes = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = None
            for k in close_keys:
                if k in row and row[k] not in (None, ""):
                    val = try_parse_float(row[k])
                    if val is not None:
                        break
            if val is None:
                # 本行不是有效数值（可能是副表头/说明行），跳过
                continue

            closes.append(val)
            if len(closes) >= n:
                break

    return closes

# ============== 测试用例 ==============
def test_load_and_iterate_basic():
    DataFeed = _import_datafeed()

    feed = DataFeed()
    assert feed.load(str(DATA_CSV)), "DataFeed.load 返回 False"

    # 遍历前几行，校验字段存在与类型正确
    seen = 0
    while feed.next() and seen < 5:
        row = feed.current()
        assert hasattr(row, "date"), "row 缺少 date 字段"
        assert hasattr(row, "close"), "row 缺少 close 字段"
        assert isinstance(row.close, (int, float)) and math.isfinite(row.close), "close 不是有限数"
        seen += 1

    assert seen > 0, "未能遍历到任何一行数据"


def test_moving_average_window5_matches_python_ref_with_tolerance():
    DataFeed = _import_datafeed()

    # 用 Python 计算前 5 根 close 的均值作为参考
    first5 = read_first_n_closes(DATA_CSV, 5)
    if len(first5) < 5:
        pytest.skip(f"无法从 {DATA_CSV} 读取到 5 个数值型收盘价，可能是文件表头/格式异常")
    ref_avg = sum(first5) / 5.0

    feed = DataFeed()
    assert feed.load(str(DATA_CSV)), "DataFeed.load 返回 False"

    ma = feed.moving_average(5)

    # 要求返回可迭代、非空
    assert hasattr(ma, "__iter__"), "moving_average(5) 应返回可迭代对象"
    ma_list = list(ma)
    assert len(ma_list) >= 1, "moving_average(5) 返回长度为 0"

    # 数值合理性：在前若干项里找到一个接近 ref_avg 的值（容差考虑实现对齐差异）
    head = ma_list[:10]
    head = [x for x in head if isinstance(x, (int, float)) and math.isfinite(x)]
    assert head, "moving_average(5) 前几项不是有效数值"

    tol = 1e-6
    matched = any(abs(x - ref_avg) <= tol for x in head)

    # 如果你的实现是严格对齐（第 5 项等于首个均值），可以把上面改成：
    # matched = (len(ma_list) >= 5 and isinstance(ma_list[4], (int, float)) and math.isfinite(ma_list[4]) and abs(ma_list[4] - ref_avg) <= tol)

    assert matched, f"5 日均线前几项未出现参考均值（{ref_avg:.6f}）±{tol}"



# scripts/test_trading_env.py
import sys
import os
from pathlib import Path
import pytest
import numpy as np

# ========== 路径设置 ==========
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "env"))
sys.path.insert(0, str(ROOT / "data"))

from trading_env import TradingEnv
from data_loader import load_csv_data

# ========== 数据准备 ==========
DATA_PATH = ROOT / "data" / "SPY_1d.csv"
if not DATA_PATH.exists():
    pytest.skip(f"缺少测试数据文件：{DATA_PATH}", allow_module_level=True)

data = load_csv_data(str(DATA_PATH))

# ========== 测试用例 ==========
def test_env_reset_and_initial_obs():
    env = TradingEnv(data, window_size=10, initial_balance=1000)
    obs, info = env.reset()
    assert isinstance(obs, (np.ndarray, list)), "reset 应返回 ndarray/list"
    assert env.balance == 1000, "初始资金应为 1000"
    assert "step" in info, "info 应包含 step 字段"

def test_env_step_flow():
    env = TradingEnv(data, window_size=10, initial_balance=1000)
    env.reset()

    actions = [1, 0, 2, 1, 2]  # Buy → Hold → Sell → Buy → Sell
    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        # 基础检查
        assert isinstance(obs, (np.ndarray, list))
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        # 资金要合理
        assert env.balance >= 0
        if done:
            break

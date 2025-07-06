import os
import sys
import numpy as np

# Add env/ and data/ directories to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../env")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data")))

from trading_env import TradingEnv
from data_loader import load_csv_data

# Load market data
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/SPY_1d.csv"))
data = load_csv_data(data_path)

# Initialize environment
env = TradingEnv(data, window_size=10, initial_balance=1000)

# Reset environment
obs, info = env.reset()
print("✅ Initial Observation:")
print(obs)

# Simulate a few steps
actions = [1, 0, 2, 1, 2]  # Example sequence: Buy → Hold → Sell → Buy → Sell
print("\n🏃‍♂️ Simulating environment steps:")
for step, action in enumerate(actions):
    obs, reward, done, truncated, info = env.step(action)
    print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Balance={env.balance:.2f}, Done={done}")

    if done:
        print("💥 Episode finished.")
        break

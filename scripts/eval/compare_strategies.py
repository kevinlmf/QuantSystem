import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from env.trading_env import TradingEnv
from env.data_loader import load_csv_data

# 加载数据
data = load_csv_data("data/SPY_1d.csv")

# 定义通用运行函数
def run_strategy(env, policy_fn):
    obs, _ = env.reset()
    done = False
    balances = []
    total_reward = 0
    while not done:
        action = policy_fn(env, obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        balances.append(env.balance)
        total_reward += reward
    return balances, total_reward

# 策略函数定义
def dqn_policy(env, obs):
    return dqn_model.predict(obs, deterministic=True)[0]

def ppo_policy(env, obs):
    return ppo_model.predict(obs, deterministic=True)[0]

def random_policy(env, obs):
    return env.action_space.sample()

# 加载模型
dqn_model = DQN.load("models/dqn_trading")
ppo_model = PPO.load("models/ppo_trading")

# 跑三种策略
env = TradingEnv(data=data)
dqn_balances, dqn_reward = run_strategy(env, dqn_policy)

env = TradingEnv(data=data)
ppo_balances, ppo_reward = run_strategy(env, ppo_policy)

env = TradingEnv(data=data)
random_balances, random_reward = run_strategy(env, random_policy)

# 画图对比
plt.figure(figsize=(10, 6))
plt.plot(dqn_balances, label=f"DQN (${dqn_reward:.2f})")
plt.plot(ppo_balances, label=f"PPO (${ppo_reward:.2f})")
plt.plot(random_balances, label=f"Random (${random_reward:.2f})")
plt.title("Strategy Comparison: Balance Over Time")
plt.xlabel("Step")
plt.ylabel("Balance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧠 Quant_trading_system 📈  
A modular and extensible quantitative trading system designed for **realistic financial market simulation**, **reinforcement learning strategies**, and **high-performance execution**.

---

## ⚙️ Key Highlights
- 🤖 **RL Algorithms**: PPO & DQN (Stable-Baselines3)
- ⚡ **High-performance C++ core**: Fast data loading & order execution (via PyBind11)
- 🌍 **Custom Gym Environment**: Multi-asset market simulator with discrete & continuous actions
- 📊 **Built-in Strategy Evaluation**: Compare PPO / DQN / Random baseline
- 🎛️ **CLI-friendly Automation**: Train & test in one command
- 📡 **TensorBoard Live Monitoring**
- 🧪 **Unit test support** for Python & C++ modules

---

## 🚀 Quick Start (Recommended)
```bash
git clone https://github.com/kevinlmf/Quant_trading_system.git
cd Quant_trading_system
bash scripts/set_up.sh
```

### This script will:
- 🔧 Create a virtual environment
- 📦 Install all dependencies
- ⚙️ Compile C++ module (`cpp_trading.so`)
- ✅ Verify everything works

> ⚠️ Default uses `python3.10`, replace with `python3` if needed.

---

## ✅ Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash scripts/build_cpp_module.sh
python scripts/test_cpp_module.py
```

---

## 🧠 Usage Guide

```bash
# Run baseline strategy
python scripts/test_random.py

# Train PPO / DQN agents
python train_ppo.py
python train_dqn.py

# Compare results
python scripts/compare_strategies.py
```

---

## 🗂️ Project Structure

```
Quant_trading_system/
├── train_dqn.py / train_ppo.py         ← RL entrypoints
├── scripts/
│   ├── run_training.sh                 ← Single-command trainer
│   ├── compare_strategies.py           ← Evaluation visualizer
│   ├── test_model.py / test_random.py  ← Model & baseline tests
│   ├── test_cpp_module.py              ← C++ sanity test
│   └── set_up.sh                       ← Setup all modules
├── env/
│   ├── trading_env.py                  ← Custom gym.Env
│   └── data_loader.py                  ← OHLCV loader (Py/C++)
├── cpp_core/
│   ├── include/ src/ bindings/         ← C++ code + PyBind11 interface
│   └── build/ CMakeLists.txt           ← Build configs
├── models/                             ← Saved agents
├── data/                               ← Market data (real/simulated)
├── tensorboard/                        ← Logging directory
└── README.md                           ← This file
```

---

## 📈 Finance Modules (Planned / Ongoing)

| Module                   | Description |
|--------------------------|-------------|
| 💹 **Financial Indicators**   | Sharpe Ratio, Sortino, Max Drawdown, Win Rate |
| 💼 **Asset Types**           | Crypto, Stocks, ETFs, FX, Index Futures |
| 🧠 **Risk Control Modules**  | CVaR, VaR, drawdown penalties, Copula modeling |
| 📉 **Market Models**         | GARCH, Heston, Black-Scholes, jump diffusion |
| 🧩 **High-Dimensional Support** | Factor-based & copula-based joint asset modeling |
| 🎯 **Alpha Signals**         | Feature engineering for technical/fundamental signals |
| 🔍 **Exploration Techniques** | Latent bonus, MBIE-EB, ensemble value uncertainty |
| 🧪 **Portfolio Optimizer**   | Mean-Variance, CVaR, Copula stress test integration |

---

## 🧱 Requirements

All listed in `requirements.txt`, including:

- `stable-baselines3==1.8.0`
- `gymnasium==0.29.1`
- `pybind11>=2.11`
- `numpy`, `pandas`, `matplotlib`, `tensorboard`, `scikit-learn`

---

## 🪄 Upcoming Features
- 📈 Portfolio metrics (Sharpe, CVaR, drawdown)
- 🧠 Real-time alpha/risk modules (via `strategy/` & `risk_control/`)
- 📁 Export trade logs (CSV)
- ⚙️ Hyperparameter tuning (Optuna)
- 🧩 Exploration strategy modules
- 🔍 Visual explainability tools for policy behavior

---

## 📄 License
MIT License © 2025 [Mengfan Long](https://github.com/kevinlmf)

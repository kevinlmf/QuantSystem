# 🧠 Quant_trading_system 📈

A full-featured modular quantitative trading system powered by:

- 🤖 Reinforcement Learning with PPO / DQN (Stable-Baselines3)
- 🧠 High-performance C++ data loader + execution module via PyBind11
- 🌿 Custom OpenAI Gym-style trading environment
- 🖥️ Linux-friendly CLI automation for training & testing
- 📊 Strategy comparison (DQN vs PPO vs Random baseline)
- 🧪 Integrated test framework for C++ modules and Python models
- 📦 TensorBoard logging support for live training monitoring

---

## 🚀 Quick Start (One Command Setup)

Clone this repo and run the setup script to get started:

```bash
git clone https://github.com/kevinlmf/Quant_trading_system.git
cd Quant_trading_system
bash scripts/set_up.sh
```

This script will:

- 🔧 Create and activate a Python virtual environment
- 📦 Install all required Python dependencies
- ⚙️ Build the C++ module with PyBind11
- ✅ Run a test to ensure `cpp_trading.so` loads properly

⚠️ Note: This script uses python3.10 by default.
If you don’t have it installed, you may:

Replace python3.10 with python3 in scripts/set_up.sh, or

Install Python 3.10 via pyenv.
---

## ✅ Manual Installation (Advanced)

```bash
# 1. Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Build the C++ backend module
bash scripts/build_cpp_module.sh

# 4. Run test to verify module works
python scripts/test_cpp_module.py
```

---

## 🧠 How to Use

```bash
# Run the random baseline
python scripts/test_random.py

# Train PPO agent
python train_ppo.py

# Train DQN agent
python train_dqn.py

# Compare strategies
python scripts/compare_strategies.py
```

---

## 📁 Project Structure

```
Quant_trading_system/
├── train_dqn.py / train_ppo.py         ← RL training entrypoints
├── scripts/                            ← CLI tools
│   ├── run_training.sh                 ← One-command training runner
│   ├── compare_strategies.py           ← Visualize DQN / PPO / Random
│   ├── test_model.py                   ← Evaluate saved models
│   ├── test_random.py                  ← Run random baseline
│   ├── test_cpp_module.py              ← Sanity test for cpp_trading.so
│   └── set_up.sh                       ← 🧠 One-shot full setup script
├── env/                                ← Custom Gym environment
│   ├── trading_env.py
│   └── data_loader.py
├── cpp_core/                           ← C++ backend
│   ├── include/                        ← C++ headers
│   │   ├── data_feed.h
│   │   └── order_executor.hpp
│   ├── src/                            ← C++ implementations
│   │   ├── data_feed.cpp
│   │   └── order_executor.cpp
│   ├── bindings/                       ← PyBind11 Python interface
│   │   ├── data_bindings.cpp
│   │   ├── order_bindings.cpp
│   │   └── main_bindings.cpp
│   ├── build/                          ← Output: cpp_trading.so
│   └── CMakeLists.txt                  ← Build config
├── models/                             ← Trained RL agents
├── data/                               ← Historical OHLCV data
├── tensorboard/                        ← Training logs
└── README.md                           ← You're here!
```

---

## 📦 Requirements

Listed in `requirements.txt`, including:

- `stable-baselines3==1.8.0`
- `gymnasium==0.29.1`
- `pybind11>=2.11`
- `numpy`, `pandas`, `matplotlib`, `tensorboard`, `scikit-learn`

---

## 🛠️ Features

- ✅ PPO / DQN reinforcement learning agents
- ✅ Modular training using custom `gym.Env`
- ✅ C++ backend with PyBind11 integration
- ✅ One-line setup and training script
- ✅ Strategy comparison plots
- ✅ TensorBoard integration

---

## 🔮 Future Work

- 📈 Add portfolio metrics (Sharpe, WinRate, Drawdown)
- 🧩 Integrate real-time execution engine
- 🧠 Add alpha signal & risk control modules
- 📁 Export trade logs as CSV
- 🎯 Hyperparameter tuning (Optuna)

---

## 📄 License

MIT License © 2025 Mengfan Long

---



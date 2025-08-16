# 📈 Quantitative Trading System

A modular **quantitative trading system** that integrates **trading environments**, **order execution**, **data infrastructure**, and multiple **trading strategies**.  
It supports **backtesting**, **strategy comparison**, and is extendable to **reinforcement learning (RL)** and **high-frequency trading (HFT)**.

---

## ⚙️ Features

- 🏗️ **Modular architecture**: strategies, data, and execution are decoupled for flexibility  
- 📊 **Strategy comparison**: includes Buy & Hold, Momentum, Pairs Trading, Random, and more  
- 🔧 **Testing coverage**: unit tests ensure robustness and reproducibility  
- 📈 **Extensibility**: easy integration of RL agents, Copula-based risk control, and market microstructure modeling  

---

## 🚀 Quick Start

### 1. Clone and navigate into the project
```bash
git clone https://github.com/yourname/Quantitive-Trading-System.git
cd Quantitive-Trading-System
```

### 2. Install dependencies (recommended: virtual environment)
```bash
pip install -r requirements.txt
```

### 3. Run tests to verify the system
```bash
# Run all tests
pytest -q scripts

# Or run individual test modules
pytest -q scripts/test_order_executor.py
pytest -q scripts/test_trading_env.py
pytest -q scripts/test_cpp_trading.py
pytest -q scripts/test_data_infrastructure.py
pytest -q scripts/test_trading_system.py
```

### 4. Run strategy comparison
```bash
python scripts/strategy_comparison.py
```

This will:
- Generate or load market data  
- Execute multiple trading strategies  
- Output performance metrics and comparison charts  

---

## 📊 Example Output

When running `python scripts/strategy_comparison.py`, you will obtain results such as:  
- **Net PnL** (profit & loss after transaction costs)  
- **Sharpe Ratio** (risk-adjusted returns)  
- **Max Drawdown** (risk of loss)  
- Strategy comparison plots for performance evaluation  

---

## 📂 Project Structure

```
Quantitive-Trading-System/
│── scripts/                  # Test scripts and strategy runner
│   ├── test_order_executor.py
│   ├── test_trading_env.py
│   ├── test_cpp_trading.py
│   ├── test_data_infrastructure.py
│   ├── test_trading_system.py
│   └── strategy_comparison.py
│
│── strategy/                 # Strategy implementations
│   ├── momentum_strategy.py
│   ├── pairs_trading.py
│   └── ...
│
│── data/                     # Market data (real/simulated)
│── cpp_core/                 # C++ trading core (PyBind11 bindings)
│── env/                      # Trading environments
│── evaluation/               # Backtesting and evaluation modules
│── requirements.txt          # Python dependencies
│── README.md                 # Project documentation
```

---

## 📌 Notes

- **NetPnL** = profit/loss after transaction fees → the most realistic measure of profitability  
- **Lookback period** is a key hyperparameter in momentum and pairs trading strategies  
- Can be extended with **RL agents (DQN, PPO, SAC)** for advanced trading experiments  

---

## 🛠️ Next Steps

- ✅ Add more benchmark strategies (mean reversion, factor-based, copula risk models)  
- ✅ Extend evaluation with CVaR and tail risk metrics  
- ✅ Integrate RL-based agents for adaptive trading  
- ✅ Scale simulations with parallel execution  

---

## 📜 License

MIT License © 2025 Your Name

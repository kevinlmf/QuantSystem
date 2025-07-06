A modular quantitative trading system integrating a **high‑performance C++ core** with a clean Python interface. The system focuses on **risk‑aware portfolio optimization**, realistic **order execution**, and **market‑grade back‑testing** via a custom Gymnasium environment.

---

## ⚙️ Current Highlights
- 📊 **Risk Metrics:** CVaR computation in Python  
- ⚡ **High‑performance C++ Core:** Data feed and order‑execution modules wrapped with PyBind11  
- 🌍 **Trading Environment:** Custom Gymnasium environment for back‑testing and strategy simulation  
- 🧪 **Testing Scripts:** Python tests covering environment and order‑execution integration  

---

## 🚀 Quick Start

```bash
git clone git@github.com:kevinlmf/Risk-Aware-Trading-System.git
cd Risk-Aware-Trading-System

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# build the C++ extension
cd cpp_core
python setup.py build_ext --inplace
cd ..
```

---

## 🧠 Usage

```bash
# 1️⃣ Run the trading‑environment smoke test
python scripts/test_trading_env.py

# 2️⃣ Run the C++ order‑execution test
python scripts/test_order_executor.py

# 3️⃣ Run the full end‑to‑end system test
python scripts/test_trading_system.py
```

---

## 🗂️ Project Structure

```text
Risk-Aware-Trading-System/
├── cpp_core/                 # C++ source code, pybind11 bindings, and build artifacts
│   ├── include/              # C++ headers
│   ├── src/                  # C++ implementation files
│   ├── bindings/             # PyBind11 binding files
│   └── build/                # Build output directory
├── data/                     # Market data CSV files
├── env/                      # Python trading environment and data loader
│   ├── trading_env.py        # Custom Gymnasium environment
│   └── data_loader.py        # CSV data loader utility
├── risk_control/             # Python CVaR risk‑metric implementation
│   └── cvar.py
├── scripts/                  # Test and utility scripts
│   ├── test_trading_env.py
│   ├── test_order_executor.py
│   └── test_trading_system.py
├── strategy/                 # Example trading strategies (work in progress)
│   └── mean_variance.py
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 📈 Modules Implemented Today

| Module | Description |
|--------|-------------|
| **Trading Environment** | Gym‑compatible environment supporting *Buy / Sell / Hold* actions with OHLCV observations |
| **Data Feed (C++)** | High‑performance CSV loader and iterator exposed to Python |
| **Order Execution (C++)** | Order and executor classes simulating realistic submission & fills |
| **Testing Scripts** | Stand‑alone tests verifying environment steps, order execution, and integration |

---

## 🧱 Requirements
- Python ≥ 3.10  
- Packages listed in `requirements.txt` (NumPy, Pandas, Gymnasium, PyBind11, …)  

---

## 📜 License
MIT License

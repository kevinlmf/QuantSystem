# 🧠 Risk-Aware Trading System 📈  
A modular and extensible quantitative trading system designed for **risk-sensitive portfolio optimization**, **financial mathematics models**, and **high-performance execution**.

---

## ⚙️ Key Highlights
- 📊 **Risk Metrics**: CVaR, VaR, Expected Shortfall
- 🧮 **Financial Mathematics Models**: Mean-Variance Optimization, Copula-Based Allocation, Option Pricing (Black-Scholes, Heston)
- ⚡ **High-performance C++ core**: Fast data loading & order execution (via PyBind11)
- 🌍 **Custom Market Simulator**: Multi-asset market simulation with stochastic processes (GBM, Jump Diffusion)
- 🏦 **Portfolio Optimizer**: Minimize CVaR / maximize Sharpe Ratio
- 🧪 **Stress Testing**: Tail risk and systemic risk analysis via Copula models

---

## 🚀 Quick Start (Recommended)
```bash
git clone git@github.com:kevinlmf/Risk-Aware-Trading-System.git
cd Risk-Aware-Trading-System
pip install -r requirements.txt
```

---

## ✅ Manual Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash cpp_core/build_cpp.sh
```

---

## 🧠 Usage Guide

```bash
# Run CVaR portfolio optimization
python scripts/run_cvar_optimization.py

# Simulate market and test risk metrics
python scripts/test_risk_metrics.py
```

---

## 🗂️ Project Structure

```
Risk-Aware-Trading-System/
├── strategy/
│   ├── mean_variance.py            ← Markowitz optimizer
│   ├── cvar_optimization.py        ← CVaR minimization
│   ├── copula_allocation.py        ← Copula-based allocation
│   ├── option_hedging.py           ← Option pricing & hedging
├── risk_control/
│   ├── cvar.py                     ← CVaR, VaR computation
│   ├── stress_testing.py           ← Tail risk & stress tests
├── env/
│   ├── trading_env.py              ← Market environment
│   ├── market_simulator.py         ← Price path generators
├── cpp_core/
│   ├── include/ src/ bindings/     ← C++ core logic
├── data/                           ← Market data
├── scripts/                        ← Run & test scripts
└── README.md                       ← This file
```

---

## 📈 Financial Modules

| Module                   | Description |
|--------------------------|-------------|
| 💹 **Portfolio Optimization** | Mean-Variance, CVaR minimization |
| 📊 **Risk Metrics**           | VaR, CVaR, Expected Shortfall |
| 🧠 **Copula Modeling**        | Multi-asset dependence modeling |
| 🧮 **Option Pricing**         | Black-Scholes, Heston model |
| 🧪 **Stress Testing**         | Systemic risk & tail dependence |
| 🎯 **Factor Models**          | High-dimensional copula factor models |

---

## 🧱 Requirements

All dependencies are listed in `requirements.txt`, including:

- `numpy`, `pandas`, `matplotlib`, `cvxpy`
- `scipy`, `pybind11`
- `scikit-learn`, `seaborn`

---

## 🪄 Upcoming Features
- 📈 Portfolio metrics (Sharpe, Sortino, Max Drawdown)
- 🧠 Copula risk modules for multi-asset systems
- 📁 Export allocation & trade logs (CSV)
- 🧮 Advanced stochastic control models
- ⚙️ Hyperparameter tuning for optimizers

---

## 📜 License
MIT License


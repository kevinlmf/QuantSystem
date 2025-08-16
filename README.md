## 📈 Quantitative Trading System

A modular **quantitative trading system** that integrates **trading environments**, **order execution**, **data infrastructure**, and multiple **trading strategies**.  
It supports **backtesting**, **strategy comparison**, and can be extended to **reinforcement learning (RL)** and **high‑frequency trading (HFT)** scenarios.

---

## ⚙️ Features

- 🏗️ **Modular architecture**: strategies, data, and execution are decoupled for flexibility  
- ⚡ **C++ core (optional)**: PyBind11 bindings for high‑performance data feed & order execution  
- 🌍 **Gymnasium environments**: single‑asset and multi‑asset sims with costs, slippage, market impact  
- 📊 **Strategy suite**: Momentum, Pairs Trading, Mean‑Variance (MPT) + comparison tools  
- 🛡️ **Risk analytics**: Sharpe/Sortino/Calmar, VaR/CVaR, drawdown analysis  
- 🧪 **Robust testing**: `pytest` tests for env, data, strategy, and C++ integration

---

## 🚀 Quick Start

```bash
git clone https://github.com/kevinlmf/Quantitive-Trading-System.git
cd Quantitive-Trading-System
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# (optional) build C++ extension
cd cpp_core && python setup.py build_ext --inplace && cd ..

# run tests
pytest -q scripts

# run strategy comparison demo
python scripts/strategy_comparison.py
```

---

## 🗂️ Repository Tree (Top‑Level)

```text
Quantitive-Trading-System/
├─ cpp_core/                  # C++ engine & bindings (optional, for speed)
├─ data/                      # Small sample data + data pipeline helpers
├─ env/                       # Trading environments & loaders
├─ execution_engine/          # Python market / execution simulators
├─ risk_control/              # Risk metrics & portfolio risk utils
├─ strategy/                  # Trading strategies
├─ scripts/                   # Tests, demos, utilities
├─ requirements.txt
├─ README.md
└─ setup.py                   # (optional) package build/install config
```

> **Note**: Large datasets, venv, and compiled artifacts are ignored via `.gitignore` to keep the repo lightweight.

---

## 🧩 Section Trees with File‑by‑File Notes

### 1) `cpp_core/` — High‑Performance C++ Core (Optional)
```text
cpp_core/
├─ CMakeLists.txt                 # CMake build script (alternative to setup.py)
├─ setup.py                       # Build PyBind11 extension in place
├─ bindings/
│  └─ all_bindings.cpp            # PyBind11 glue: exposes C++ classes to Python
├─ include/
│  ├─ data_feed.h                 # DataFeed class interface (CSV loader/iterator)
│  ├─ order.hpp                   # Order struct + enums (side/type)
│  └─ order_executor.hpp          # OrderExecutor interface (submit/fill/report)
├─ src/
│  ├─ data_feed.cpp               # Fast CSV parsing, rolling features, MA etc.
│  ├─ order.cpp                   # Order lifecycle helpers
│  └─ order_executor.cpp          # Matching/fill logic, fees & slippage model
└─ build/                         # (ignored) compiled artifacts output
```
**Purpose**: Provide low‑latency data iteration and realistic order execution primitives for heavy backtests. Use Python fallbacks if you skip building this.

---

### 2) `data/` — Data Infrastructure
```text
data/
├─ README.md                      # Explains sample files & how to fetch full data
├─ download_data.py               # CLI: fetch symbols (Yahoo/CSV), batch updates
├─ database.py                    # SQLite helper (connect, create tables, write)
├─ pipeline.py                    # ETL: clean, resample, align, enrich indicators
├─ validators.py                  # QA checks: missingness, outliers, schema
└─ multi_asset_provider.py        # Load many symbols, align calendar, join OHLCV
```
**Purpose**: Make reproducible, validated datasets. Use `download_data.py` for quick pulls; store big data outside the repo.

---

### 3) `env/` — Trading Environments & Loaders
```text
env/
├─ trading_env.py                 # Minimal Gymnasium env (Hold/Buy/Sell)
├─ advanced_trading_env.py        # Industrial env: fees, slippage, impact
├─ data_loader.py                 # Load CSV/DB to DataFrame(s), feature prep
├─ env_monitor.py                 # Runtime metrics (episode PnL, winrate, etc.)
└─ system_monitor.py              # CPU/RAM/IO monitors for long runs
```
**Purpose**: Standardize the simulation interface for both rule‑based and RL agents.

---

### 4) `execution_engine/` — Market / Execution Simulators
```text
execution_engine/
└─ market_simulator.py            # Python execution sim: spreads, latency, fills
```
**Purpose**: A Python alternative to the C++ executor; easier to tinker, slower than C++.

---

### 5) `risk_control/` — Risk Analytics & Limits
```text
risk_control/
├─ cvar.py                        # VaR/CVaR estimators (historical / Cornish-Fisher)
└─ portfolio_manager.py           # Position caps, exposure, leverage, DD limits
```
**Purpose**: Compute portfolio risk and enforce limits at strategy or portfolio level.

---

### 6) `strategy/` — Trading Strategies
```text
strategy/
├─ momentum_strategy.py           # Multi-factor momentum (price/tech/cross-sectional)
├─ pairs_trading.py               # Statistical arbitrage: pair selection + signals
└─ mean_variance.py               # Markowitz MPT weights with constraints
```
**Purpose**: Pluggable strategies with clear `generate_signals(...)` and sizing APIs.

**Key Notes**
- `momentum_strategy.py`: price returns across lookbacks; RSI/MACD/MA-cross; optional vol scaling.  
- `pairs_trading.py`: find highly correlated pairs; compute spread z‑score; enter/exit on thresholds.  
- `mean_variance.py`: compute covariance & expected returns windows; solve for constrained weights.

---

### 7) `scripts/` — Tests, Demos & Utilities
```text
scripts/
├─ strategy_comparison.py         # Simulate & compare Momentum / Pairs / MPT + benchmark
├─ test_cpp_trading.py            # C++ DataFeed & OrderExecutor smoke tests
├─ test_order_executor.py         # Submit/execute/get_filled_orders flow
├─ test_trading_env.py            # Env reset/step signatures, reward & info checks
├─ test_trading_system.py         # End-to-end: data→signals→orders→pnl
├─ test_data_infrastructure.py    # Pipeline & validators correctness
└─ set_up.sh                      # First-time project setup (env vars / folders)
```
**Purpose**: Reproducible entry points for testing and demonstrations.

---

## 📊 Metrics & Analytics (Built‑in)
- **Performance**: Annualized return, volatility, Sharpe/Sortino/Calmar, profit factor  
- **Risk**: Max drawdown, VaR/CVaR, tail risk statistics  
- **Trading**: Win rate, turnover, average trade duration  
- **Benchmarking**: Strategy vs. equal‑weight buy‑and‑hold; rolling Sharpe

---

## 🧪 Testing

```bash
pytest -q scripts                       # run all tests
pytest -q scripts/test_trading_env.py   # run a specific module
python scripts/strategy_comparison.py   # run the strategy comparison demo
```

---

## 🧱 Conventions & Tips

- Keep large data outside the repo; include tiny samples only (`data/README.md` explains how to fetch full data).  
- Prefer `--force-with-lease` when rewriting history; protect `main` branch in GitHub settings.  
- Use `.gitignore` to exclude `venv/`, caches, compiled artifacts, and large data.  
- For speed‑critical loops, prefer C++ modules (PyBind11) or vectorize with NumPy.

---

## 📜 License

MIT License © 2025 Mengfan Long

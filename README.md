#  HFT_Trading_System: High-Frequency Trading & Strategy Research Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

> **A high-frequency trading (HFT) platform built for ultra-low latency execution and intelligent strategy discovery.**

---

## Project Motivation



HFT_Trading_System is designed to answer two fundamental questions:

1. **How can we make trading faster?**  
   → Hardware-efficient C++ execution, lock-free data pipelines, and optimized memory design.

2. **What are the best strategies to trade?**  
   → Data-driven discovery using reinforcement learning, deep learning, and statistical inference.

---


##  Key Features

###  **Ultra-Low-Latency Core**
- **C++17 Execution Engine**: Sub-microsecond order processing via Pybind11  
- **Lock-Free Queues**: Concurrent order book and market data handling  
- **Hardware Optimization**: SIMD vectorization, cache alignment, NUMA-aware design  

###  **Strategy Research Layer**
- **Classical Strategies**: Momentum, Mean Reversion, Statistical Arbitrage  
- **ML-Based Models**: LSTM, Transformer, and Gradient Boosting  
- **RL & Imitation Learning**: PPO, DQN, Soft Behavior Cloning  
- **Ensemble Framework**: Combine multiple models for regime-robust performance  

###  **Evaluation & Risk Control**
- **Real-Time Analytics**: Sharpe, Sortino, and drawdown tracking  
- **Backtesting Engine**: Vectorized tick-level simulation  
- **Risk Constraints**: Position sizing, stop-loss, CVaR bounds  

---

##  Architecture Overview

```
HFT_Trading_System/
│
├── cpp_core/              # Ultra-low-latency execution engine (C++)
│   ├── order.cpp
│   ├── data_feed.cpp
│   └── pybind_bindings.cpp
│
├── strategy/              # Trading strategies
│   ├── momentum_strategy.py
│   ├── ml_strategy.py
│   └── ensemble_strategy.py
│
├── execution_engine/      # Order routing and market interface
├── risk_control/          # Real-time risk checks
├── evaluation/            # Backtests and performance metrics
│
├── benchmarks/            # Latency and throughput testing
├── examples/              # Usage demos
└── scripts/               # Comparative and diagnostic scripts
```

---

##  Quick Start

### Setup Environment
```bash
git clone https://github.com/kevinlmf/HFT_Trading_System
cd HFT_Trading_System
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

###  Build C++ Core
```bash
./build_cpp.sh  # Linux/macOS
# or
python cpp_core/setup.py build_ext --inplace  # Windows
```

### 3️ Verify Installation
```bash
python test_build.py
# ✓ All tests passed!
```

---

##  Run Examples

```bash
# Compare basic strategies
python scripts/strategy_comparison.py

# Evaluate ML-based strategies
python scripts/ml_strategy_comparison.py

# Full benchmark suite
python scripts/evaluate_strategies.py
```

---

##  Performance Highlights

| Metric | Value |
|--------|-------|
| **Order Processing** | < 1 μs (C++ core) |
| **Backtesting Speed** | ~1M ticks/sec |
| **Python Strategy Latency** | 10–50 μs |
| **Memory Footprint** | < 100 MB typical |

---


##  Documentation

- **[Quick Start Guide](QUICK_START.md)**
- **[Build Guide](BUILD_GUIDE.md)**
- **Examples & Benchmarks** in `/examples/` and `/benchmarks/`

---

##  Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**  
This software is **not intended for live trading** without extensive testing.  
Trading in financial markets involves substantial risk.

---
## Future Work

Future development of will focus on two research-driven directions:

1. **Statistical Computing Optimization**  
   Designing algorithms with **lower space and time complexity**, leveraging numerical linear algebra, memory-efficient data structures, and distributed computing to further reduce computational overhead in ultra-high-frequency settings.

2. **Microstructure-Aware Strategy Adaptation**  
   Developing **adaptive HFT strategies** that respond to evolving market microstructures — such as order flow imbalance, spread dynamics, and latency arbitrage — using online learning and reinforcement frameworks optimized for ultra-low latency execution.
---
##  Acknowledgments

Built using:
- [Pybind11](https://github.com/pybind/pybind11)
- [PyTorch](https://pytorch.org/)
- [JAX](https://github.com/google/jax)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

---

<div align="center">
Worry less — every microsecond counts😊

</div>


# QuantSystem: Advanced Quantitative Trading Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)

> **A high-performance quantitative trading platform combining ultra-low-latency C++ execution with modern ML/AI-driven strategies.**

---
##  Project Motivation

Modern quantitative trading requires:
1. **Ultra-low latency** for high-frequency opportunities
2. **Advanced ML/AI** for pattern recognition and prediction
3. **Robust risk management** for capital preservation
4. **Flexible architecture** for rapid strategy iteration

---

##  Vision

QuantSystem began as a **High-Frequency Trading (HFT)** system but is evolving into a comprehensive **Quantitative Trading & Financial AI Platform** that encompasses:

- **High-Frequency Trading** (microsecond execution)
- **Machine Learning** for adaptive strategies
- **Portfolio Optimization** (multi-asset allocation)
- **Options Pricing & Greeks** (derivatives trading)
- **Extreme-condition risk management** via dependency structure modeling
- **Factor Analysis** (systematic alpha generation)
- **AI Agents** (autonomous trading systems)

**Current Status**: Production-ready HFT core with expanding ML/AI capabilities.

---

## Key Features

###  **High-Performance Core**
- **C++ Execution Engine**: Sub-microsecond order processing via Pybind11
- **Lock-Free Data Structures**: Concurrent order book management
- **Hardware Optimization**: SIMD vectorization, cache-friendly algorithms

###  **Multi-Strategy Framework**
- **Traditional Strategies**: Momentum, Mean Reversion, Statistical Arbitrage
- **ML-Enhanced Strategies**: LSTM, Transformer-based prediction models
- **Ensemble Methods**: Multi-model aggregation for robust signals

###  **Machine Learning Stack**
- **Deep Learning**: PyTorch, TensorFlow for neural network strategies
- **Gradient Boosting**: XGBoost, LightGBM for feature-based models
- **JAX Integration**: Hardware-accelerated numerical computing

###  **Production-Ready Infrastructure**
- **Backtesting Engine**: Vectorized historical simulation
- **Risk Management**: Real-time position limits and drawdown controls
- **Performance Analytics**: Sharpe ratio, max drawdown, factor attribution

---

##  Architecture

```
QuantSystem/
│
├── cpp_core/              # C++ ultra-low-latency execution
│   ├── order.cpp          # Order management
│   ├── data_feed.cpp      # Market data processing
│   └── pybind_bindings.cpp
│
├── strategy/              # Trading strategies
│   ├── momentum_strategy.py
│   ├── ml_strategy.py     # ML-based strategies
│   └── ensemble_strategy.py
│
├── execution_engine/      # Order execution logic
├── risk_control/          # Risk management
├── evaluation/            # Performance metrics
│
├── benchmarks/            # Performance testing
├── examples/              # Usage examples
└── scripts/               # Comparison & analysis tools
```

---

##  Quick Start (5 Minutes)

### Prerequisites
- Python 3.10+
- C++ compiler (clang/gcc)
- Git

### Installation

```bash
# 1. Clone repository
git clone https://github.com/kevinlmf/QuantSystem.git
cd QuantSystem

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Build C++ module
./build_cpp.sh  # Windows: python cpp_core/setup.py build_ext --inplace

# 5. Verify installation
python test_build.py
```

Expected output:
```
✓ All tests passed!
```

---

##  Usage Examples

### Run Strategy Comparison
```bash
# Compare traditional strategies
python scripts/strategy_comparison.py

# Compare ML strategies
python scripts/ml_strategy_comparison.py

# Comprehensive evaluation
python scripts/evaluate_strategies.py
```

### Use in Your Code

```python
import sys
sys.path.insert(0, 'cpp_core')
import cpp_trading2

# Ultra-fast C++ execution
order = cpp_trading2.Order()
order.symbol = "AAPL"
order.quantity = 100
order.price = 150.0

feed = cpp_trading2.DataFeed()
feed.add_tick("AAPL", 150.5, 1000)

# Python strategies
from strategy.momentum_strategy import MomentumStrategy
from strategy.ml_strategy import MLStrategy

strategy = MomentumStrategy(lookback=20)
signal = strategy.generate_signal(market_data)

ml_strategy = MLStrategy(model_type='lstm')
ml_strategy.train(historical_data)
prediction = ml_strategy.predict(current_data)
```

---

## Performance Benchmarks

| Metric | Value |
|--------|-------|
| **Order Processing** | < 1 μs (C++ core) |
| **Backtesting Speed** | 1M ticks/sec |
| **Strategy Latency** | 10-50 μs (Python) |
| **Memory Footprint** | < 100 MB (typical) |

---

##  Roadmap & Future Development

### Phase 1: HFT Foundation (✅ Complete)
- [x] C++ execution engine
- [x] Basic strategies (momentum, mean reversion)
- [x] Backtesting framework
- [x] ML integration (PyTorch, TensorFlow, JAX)
---

## Contributing

Contributions are welcome! This is an evolving research project. Areas of interest:

- **Strategy Development**: New alpha signals and models
- **Performance Optimization**: C++ engine improvements
- **ML/AI Research**: Novel deep learning architectures
- **Documentation**: Examples and tutorials
- **Testing**: Edge cases and platform compatibility

Please open issues for bugs or feature requests.

---

##  Documentation

- **[Quick Start Guide](QUICK_START.md)**: Get running in 5 minutes
- **[Build Guide](BUILD_GUIDE.md)**: Detailed compilation instructions
- **Examples**: See `examples/` directory
- **Benchmarks**: See `benchmarks/` for performance tests

---

##  Current Limitations & Known Issues

This is an active research project with some rough edges:

- **Cross-Platform Builds**: C++ compilation can be tricky on Windows
- **Dependency Conflicts**: ML libraries (PyTorch, TensorFlow, JAX) may clash
- **Performance Tuning**: Hardware-specific optimization needed
- **Limited Documentation**: More examples and tutorials coming
- **Test Coverage**: Expanding test suite continuously

**Status**: Core functionality stable, advanced features under development.

---


##  License

MIT License © 2025 Mengfan Long

See [LICENSE](LICENSE) file for details.

---

## Disclaimer

**FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY**

This software is provided for educational and research purposes. It is not intended to be used for live trading without extensive testing and risk management. Trading carries significant financial risk.

---


## 🌟 Acknowledgments

Built with:
- [Pybind11](https://github.com/pybind/pybind11) - C++/Python bindings
- [PyTorch](https://pytorch.org/) - Deep learning
- [JAX](https://github.com/google/jax) - High-performance numerical computing
- [NumPy](https://numpy.org/) & [Pandas](https://pandas.pydata.org/) - Data processing

---

<div align="center">



*Code a lot, Worry less* 😄

</div>

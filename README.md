# HFT C++ JAX Trading System

High-performance high-frequency trading system combining ultra-low-latency C++ execution with Python/JAX machine learning strategies.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![C++17](https://img.shields.io/badge/C++-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

## Key Features

- **Ultra-Low Latency**: Sub-microsecond orderbook operations (0.46µs P99)
- **High Throughput**: 2.11M orderbook ops/sec, 592K execution ops/sec
- **6 Traditional HFT Strategies**: Market making, stat arb, order flow, momentum, pairs, mean-variance
- **9 ML/DL Strategies**: Random Forest, XGBoost, LightGBM, LSTM, Transformer, CNN, DQN, PPO, SAC
- **Risk Management**: Position limits, stop-loss, drawdown controls, CVaR
- **JAX Acceleration**: 22.18M feature calculations/sec with XLA compilation

## Installation

### Prerequisites
- Python 3.8+, C++ Compiler (GCC 7+/Clang 5+/MSVC 2019+), CMake 3.15+

### Quick Setup

```bash
git clone https://github.com/kevinlmf/HFT_CPP_JAX_System.git
cd HFT_CPP_JAX_System

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Build C++ extension
cd cpp_core/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ../..

# Verify
python -c "import cpp_trading2; print('Success')"
```

## Quick Start

### Traditional Strategy Comparison

```bash
python scripts/strategy_comparison.py
```

### Train ML Strategy

```python
from strategy.ml_traditional import XGBoostStrategy
from strategy.ml_base import PredictionType

strategy = XGBoostStrategy(
    prediction_type=PredictionType.CLASSIFICATION,
    n_estimators=100,
    max_depth=6
)

# Prepare data with automatic feature engineering
X_train, X_val, X_test, y_train, y_val, y_test = strategy.prepare_data(
    price_data, train_split=0.7, val_split=0.15
)

# Train and generate signals
result = strategy.train(X_train, y_train, X_val, y_val)
signals = strategy.generate_signals(price_data)
```

### Use C++ Core from Python

```python
import cpp_trading2

# Create orderbook
orderbook = cpp_trading2.OrderBook()
orderbook.add_order(cpp_trading2.Order(
    order_id="ORDER001",
    symbol="AAPL",
    side=cpp_trading2.OrderSide.BUY,
    order_type=cpp_trading2.OrderType.LIMIT,
    price=150.50,
    quantity=100
))

# Get market data
spread = orderbook.get_best_ask() - orderbook.get_best_bid()
```

### Direct C++ Usage

```cpp
#include "orderbook.hpp"

trading::OrderBook orderbook("AAPL");

trading::Order order{
    .order_id = "ORDER001",
    .symbol = "AAPL",
    .side = trading::OrderSide::BUY,
    .type = trading::OrderType::LIMIT,
    .price = 150.50,
    .quantity = 100
};
orderbook.add_order(order);

auto best_bid = orderbook.get_best_bid();
auto best_ask = orderbook.get_best_ask();
```

Build standalone:
```bash
g++ -std=c++17 -O3 -I cpp_core/include/ your_app.cpp -o your_app
```

## Performance Benchmarks

Apple M1 Pro (10-core):

| Component | Throughput | Latency (P99) |
|-----------|------------|---------------|
| OrderBook | 2.11M ops/sec | 0.46µs |
| Execution | 592K ops/sec | 4.25µs |
| JAX Features | 22.18M ops/sec | 5.74ms |

```bash
python benchmarks/benchmark_throughput.py
```

## Strategies

**Traditional**: Market Making, Statistical Arbitrage, Order Flow Imbalance, Momentum, Pairs Trading, Mean-Variance

**ML/DL**: Random Forest, XGBoost, LightGBM, LSTM, Transformer, CNN

**RL**: DQN, PPO, SAC

See [strategy/ML_STRATEGIES_README.md](strategy/ML_STRATEGIES_README.md) for details.

## Project Structure

```
HFT_JAX_System/
├── cpp_core/            # C++ core engine
├── strategy/            # Trading strategies
├── env/                 # Trading environments
├── execution_engine/    # Order execution
├── risk_control/        # Risk management
├── evaluation/          # Performance metrics
├── scripts/             # Comparison scripts
├── examples/            # Example code
└── benchmarks/          # Performance tests
```

## Documentation

- [ML Strategies Guide](strategy/ML_STRATEGIES_README.md)
- Examples: `examples/` directory
- Scripts: `scripts/` directory

## License

MIT License © 2025 Mengfan Long

**Disclaimer**: Educational and research purposes only. Not financial advice. Trading carries significant risk.

---

**Happy Coding, Happy Life!** 🚀


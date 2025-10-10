# HFT C++ JAX Trading System

High-performance high-frequency trading system combining ultra-low-latency C++ execution with Python/JAX machine learning strategies.

**Key Highlights:**
- **Ultra-Low Latency**: Sub-microsecond orderbook operations (0.46µs P99)
- **High Throughput**: 2.11M orderbook ops/sec, 592K execution ops/sec
- **6 Traditional HFT Strategies**: Market making, stat arb, order flow, momentum, pairs, mean-variance
- **9 ML/DL Strategies**: Random Forest, XGBoost, LightGBM, LSTM, Transformer, CNN, DQN, PPO, SAC
- **Risk Management**: Position limits, stop-loss, drawdown controls, CVaR
- **JAX Acceleration**: 22.18M feature calculations/sec with XLA compilation

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/kevinlmf/HFT_Trading_System.git
cd HFT_Trading_System

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build C++ extension
cd cpp_core
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)  # Linux
# make -j$(sysctl -n hw.ncpu)  # macOS
cd ../..

# Verify installation
python -c "import cpp_trading2; print('✓ C++ module loaded successfully')"

# Run demos
python scripts/strategy_comparison.py    # Traditional strategy comparison
python scripts/evaluate_strategies.py    # Comprehensive evaluation

# Run benchmarks
python scripts/benchmark_latency.py      # Python component speed
./scripts/benchmark_cpp.sh               # C++ core performance
```

---

## Project Structure

```
HFT_CPP_JAX_System/
├── cpp_core/                 # C++ high-performance core
│   ├── include/              # Header files
│   │   ├── orderbook.hpp         # Ultra-fast order book
│   │   ├── execution_engine.hpp  # Order matching engine
│   │   └── market_data.hpp       # Market data feed
│   ├── src/                  # C++ implementations
│   ├── bindings/             # PyBind11 bindings
│   └── benchmark/            # Performance tests
│
├── strategy/                 # Trading strategies
│   ├── market_making.py           # Market making
│   ├── statistical_arbitrage.py   # Statistical arbitrage
│   ├── order_flow_imbalance.py    # Order flow imbalance
│   ├── momentum_strategy.py       # Momentum
│   ├── pairs_trading.py           # Pairs trading
│   ├── mean_variance.py           # Mean-variance optimization
│   ├── ml_traditional.py          # Random Forest, XGBoost, LightGBM
│   ├── ml_deep.py                 # LSTM, Transformer, CNN
│   └── ml_rl.py                   # DQN, PPO, SAC
│
├── env/                      # Trading environments
│   ├── trading_env.py             # Basic Gymnasium environment
│   ├── advanced_trading_env.py    # Environment with costs/slippage
│   └── data_loader.py             # Historical data loader
│
├── execution_engine/         # Order execution
│   └── market_simulator.py        # Market simulation engine
│
├── risk_control/             # Risk management
│   ├── cvar.py                    # VaR/CVaR calculation
│   └── portfolio_manager.py       # Position & risk limits
│
├── evaluation/               # Performance analysis
│   ├── strategy_evaluator.py     # Strategy evaluator
│   ├── performance_metrics.py    # Performance metrics
│   └── pnl_analyzer.py           # P&L attribution
│
├── scripts/                  # Demo & benchmark scripts
│   ├── evaluate_strategies.py     # Strategy evaluation
│   ├── strategy_comparison.py     # Strategy comparison
│   ├── benchmark_latency.py       # Python benchmark
│   └── benchmark_cpp.sh           # C++ benchmark
│
├── data/                     # Market data
└── results/                  # Evaluation results
```

---

## Core Features

### 1. Traditional HFT Strategies
- **Market Making**: Bid-ask spread capture with inventory management
- **Statistical Arbitrage**: Mean-reversion pairs trading
- **Order Flow Imbalance**: Trade on order book imbalances
- **Momentum**: Trend-following based on price momentum
- **Pairs Trading**: Cointegration-based relative value
- **Mean-Variance**: Modern portfolio theory optimization

### 2. ML/DL Strategies
**Traditional ML:**
- Random Forest, XGBoost, LightGBM

**Deep Learning:**
- LSTM (Long Short-Term Memory)
- Transformer (Attention mechanism)
- CNN (Convolutional Neural Network)

**Reinforcement Learning:**
- DQN (Deep Q-Network)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

*See [strategy/ML_STRATEGIES_README.md](strategy/ML_STRATEGIES_README.md) for detailed documentation*

### 3. Ultra-Low Latency C++ Core
- **OrderBook**: Sub-microsecond order matching
- **Execution Engine**: High-throughput order processing
- **Market Data**: Real-time tick processing
- **PyBind11 Integration**: Zero-copy Python/C++ interop

### 4. Risk Management
- Position limits & leverage controls
- Stop-loss & take-profit automation
- Maximum drawdown protection
- VaR/CVaR risk metrics
- Real-time P&L monitoring

### 5. JAX Acceleration
- GPU-accelerated feature engineering
- XLA compilation for computational graphs
- Vectorized backtesting
- High-throughput inference (22.18M ops/sec)

---

## Performance Benchmarks

**Tested on Apple M1 Pro (10-core):**

| Component | Throughput | Latency (P99) | Status |
|-----------|------------|---------------|--------|
| OrderBook Operations | 2.11M ops/sec | 0.46µs | ⭐⭐⭐⭐⭐ |
| Execution Engine | 592K ops/sec | 4.25µs | ⭐⭐⭐⭐ |
| JAX Feature Calc | 22.18M ops/sec | 5.74ms | ⭐⭐⭐⭐⭐ |
| Risk Checks | 2.69M checks/sec | 0.46µs | ⭐⭐⭐⭐⭐ |


---

**License**: MIT License © 2025 Mengfan Long

**Disclaimer**: Educational and research purposes only. Not financial advice. Trading carries significant risk.

---

Code a lot, Worry less 😄

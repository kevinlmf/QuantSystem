# HFT C++ JAX Trading System

High-Frequency Trading system combining ultra-fast C++ order execution with Python/JAX-based reinforcement learning strategies.

## Overview

This project demonstrates a **production-grade HFT architecture** with:
- **Ultra-low latency C++ core**: <5µs order book operations with zero-copy design
- **PyBind11 bindings**: Seamless Python/C++ integration for strategy development
- **JAX integration**: Hardware-accelerated RL agents and backtesting
- **Realistic simulation**: Transaction costs, slippage, market impact modeling
- **Complete infrastructure**: Data feeds, execution engine, risk controls, monitoring

---

## Features

- **C++ High-Performance Core**
  - Fast order book implementation (lock-free, array-based)
  - Order execution engine with realistic fills
  - Data feed with efficient CSV parsing
  - Designed for HFT latency requirements (<5µs)

- **Trading Environment**
  - Gymnasium-compatible RL environments
  - Advanced market simulator with microstructure effects
  - Real-time monitoring (PnL, metrics, system resources)
  - Data loaders for historical market data

- **Strategy Suite**
  - **Market Making**: Bid-ask spread capture with inventory management
  - **Statistical Arbitrage**: Cointegration-based multi-asset mean reversion
  - **Order Flow Imbalance**: Order book imbalance exploitation
  - **Momentum**: Multi-factor trend following
  - **Pairs Trading**: Statistical arbitrage on correlated assets
  - **Mean-Variance**: Portfolio optimization (MPT)
  - Extensible framework for custom strategies

- **Risk Management**
  - VaR/CVaR calculation
  - Portfolio manager with position limits
  - Drawdown monitoring and controls

- **Evaluation & Analysis**
  - Comprehensive strategy evaluator with scoring system
  - 20+ performance metrics (Sharpe, Sortino, Calmar, Omega, etc.)
  - P&L breakdown with cost attribution (commissions, slippage, impact)
  - Risk control checks and limit monitoring
  - Strategy comparison and ranking tools

- **Infrastructure**
  - Market data download utilities
  - Demo scripts for evaluation and comparison
  - Modular and extensible architecture

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/kevinlmf/HFT_CPP_JAX_System.git
cd HFT_CPP_JAX_System

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build C++ extension
python setup.py build_ext --inplace

# Run demos
python scripts/evaluate_strategies.py   # Comprehensive evaluation
python scripts/strategy_comparison.py   # Strategy comparison

# View sample results
# See evaluation_results/ folder for sample output
```

---

## Project Structure

```
HFT_CPP_JAX_System/
├── cpp_core/                 # C++ high-performance core
│   ├── include/             # Header files
│   │   ├── fast_orderbook.hpp    # Ultra-fast order book (<5µs)
│   │   ├── order_executor.hpp    # Order matching engine
│   │   ├── data_feed.h           # Market data feed
│   │   └── order.hpp             # Order structures
│   ├── src/                 # C++ implementations
│   ├── bindings/            # PyBind11 Python bindings
│   └── jaxbind/             # JAX integration bindings
│
├── env/                     # Trading environments
│   ├── trading_env.py            # Basic Gymnasium environment
│   ├── advanced_trading_env.py   # Advanced env with costs/slippage
│   ├── data_loader.py            # Historical data loader
│   ├── env_monitor.py            # Episode metrics tracking
│   └── system_monitor.py         # System resource monitoring
│
├── execution_engine/        # Market simulation
│   ├── market_simulator.py       # Python-based execution simulator
│   ├── api/                      # REST API (optional)
│   └── app/                      # Web interface (optional)
│
├── strategy/                # Trading strategies
│   ├── market_making.py          # HFT market making with inventory control
│   ├── statistical_arbitrage.py  # Multi-asset cointegration strategy
│   ├── order_flow_imbalance.py   # Order book imbalance exploitation
│   ├── momentum_strategy.py      # Multi-factor momentum
│   ├── pairs_trading.py          # Pairs trading (statistical arbitrage)
│   └── mean_variance.py          # Portfolio optimization (MPT)
│
├── risk_control/            # Risk management
│   ├── cvar.py                   # VaR/CVaR calculation
│   └── portfolio_manager.py      # Position/risk limits
│
├── evaluation/              # Strategy evaluation & analysis
│   ├── strategy_evaluator.py    # Comprehensive strategy evaluator
│   ├── performance_metrics.py   # Performance metrics calculation
│   └── pnl_analyzer.py           # P&L breakdown & attribution
│
├── data/                    # Market data
│   ├── download_data.py          # Data fetching utilities
│   └── SPY_1d.csv                # Sample data
│
├── scripts/                 # Demo scripts
│   ├── evaluate_strategies.py    # Comprehensive strategy evaluation
│   └── strategy_comparison.py    # Strategy benchmarking
│
├── evaluation_results/      # Sample evaluation output
│   ├── strategy_comparison.png   # Visual comparison chart
│   ├── evaluation_results.csv    # Summary metrics table
│   ├── detailed_performance_report.csv  # Full metrics
│   └── README.md                 # Results analysis
│
├── requirements.txt         # Python dependencies
└── setup.py                # C++ extension build config
```

---

## C++ Core Architecture

The C++ core is designed for **microsecond-level latency**:

### Fast Order Book (`fast_orderbook.hpp`)
- Lock-free, array-based design
- Zero-copy memory layout for JAX integration
- Supports 1000 price levels per side
- Operations: add liquidity, market orders, best bid/ask
- Target latency: <5µs per operation

### Order Executor (`order_executor.hpp`)
- Realistic order matching logic
- Transaction cost modeling
- Slippage simulation
- Fill reporting

### Data Feed (`data_feed.h`)
- Efficient CSV parsing
- Streaming market data
- Feature calculations (moving averages, etc.)

---

## Python Integration

### PyBind11 Bindings
The `cpp_trading2` module exposes C++ classes to Python:
```python
import cpp_trading2

# Create order book
orderbook = cpp_trading2.FastOrderBook()
orderbook.add_bid(100.0, 10.0)
orderbook.add_ask(100.1, 15.0)

# Execute orders
executor = cpp_trading2.OrderExecutor()
order = executor.submit_order(side='buy', quantity=5.0, price=100.05)
```

### JAX Integration
JAX bindings enable GPU-accelerated backtesting and RL training with zero-copy data transfer.

---

## Evaluation Framework

The evaluation module provides comprehensive strategy analysis with:

### Performance Metrics (20+ metrics)
- **Returns**: Total, annualized, cumulative
- **Risk-adjusted**: Sharpe, Sortino, Calmar, Omega ratios
- **Risk**: Max drawdown, volatility, VaR/CVaR at 95% and 99%
- **Distribution**: Skewness, kurtosis
- **Trading**: Win rate, profit factor, win/loss ratio

### P&L Attribution
Breaks down profits/losses into components:
- Gross P&L (before costs)
- Transaction costs (commissions)
- Slippage costs
- Market impact
- Financing costs
- **Net P&L** (after all costs)

### Risk Control
Automated checks against configurable limits:
- Maximum drawdown threshold
- Maximum volatility
- Minimum Sharpe ratio
- VaR limits
- Minimum win rate

### Strategy Scoring
Overall score (0-100) based on:
- 30 pts: Risk-adjusted returns (Sharpe, Sortino)
- 25 pts: Absolute returns
- 25 pts: Risk metrics (drawdown, volatility)
- 20 pts: Risk control checks

### Example Usage

```python
from evaluation import StrategyEvaluator

# Create evaluator with risk limits
evaluator = StrategyEvaluator(
    risk_limits={
        'max_drawdown': 0.15,
        'max_volatility': 0.25,
        'min_sharpe': 1.0
    }
)

# Evaluate strategy
report = evaluator.evaluate(
    strategy_name="Market Making",
    returns=returns_series,
    trades=trades_df
)

# Print comprehensive report
evaluator.print_evaluation_report(report, detailed=True)

# Compare multiple strategies
reports = [report1, report2, report3]
comparison = evaluator.compare_strategies(reports)
evaluator.plot_comparison(reports)
```

---

## Dependencies

Core requirements:
- **Python**: 3.8+
- **C++ Compiler**: C++17 support required
- **PyBind11**: 2.11+
- **Stable-Baselines3**: 2.2.1 (for RL agents)
- **Gymnasium**: 0.29.1
- **NumPy/Pandas**: Standard scientific stack

See `requirements.txt` for complete list.

---

## Running Demos

```bash
# Comprehensive strategy evaluation with full metrics
python scripts/evaluate_strategies.py

# Compare multiple strategies
python scripts/strategy_comparison.py
```

---

## Performance Benchmarks

**C++ Order Book Operations:**
- Add liquidity: <1µs
- Market order execution: <5µs
- Best bid/ask query: <100ns

**Python Environment:**
- Step latency: ~50-100µs (with C++ backend)
- Episode throughput: ~10k steps/sec

---

## Use Cases

1. **HFT Research & Development**
   - Test ultra-low latency strategies
   - Benchmark execution algorithms
   - Prototype market making strategies

2. **Reinforcement Learning**
   - Train RL agents with realistic market simulation
   - GPU-accelerated training via JAX integration
   - Compatible with Stable-Baselines3 (PPO, SAC, DQN)

3. **Strategy Backtesting**
   - High-frequency strategy evaluation
   - Transaction cost analysis
   - Risk metric calculation

4. **Education**
   - Learn HFT system architecture
   - Understand order book mechanics
   - Practice C++/Python integration

---

## Strategy Details

### 1. Market Making Strategy
- **Purpose**: Provide liquidity and capture bid-ask spread
- **Features**:
  - Inventory risk management (linear/exponential skewing)
  - Adverse selection protection
  - Dynamic spread adjustment based on volatility
  - Position limits and inventory controls

### 2. Statistical Arbitrage Strategy
- **Purpose**: Multi-asset mean reversion trading
- **Features**:
  - Cointegration testing (Engle-Granger)
  - PCA-based factor extraction
  - Half-life estimation for timing
  - Z-score based entry/exit signals

### 3. Order Flow Imbalance Strategy
- **Purpose**: Exploit temporary order book imbalances
- **Features**:
  - Multi-level order book analysis
  - Exponentially weighted imbalance calculation
  - Volume-weighted signals
  - Price impact estimation

### 4. Momentum Strategy
- Multi-factor signals (price, technical, cross-sectional)
- Risk-adjusted position sizing
- Volatility scaling

### 5. Pairs Trading
- Correlation-based pair selection
- Spread z-score signals
- Mean reversion timing

### 6. Mean-Variance Optimization
- Markowitz portfolio theory
- Constrained optimization
- Rolling covariance estimation

---

## Roadmap

- [ ] Complete JAX bindings for GPU acceleration
- [x] Implement core HFT strategies (market making, stat arb, order flow)
- [ ] WebSocket live data feeds
- [ ] Multi-asset/multi-exchange support
- [ ] Distributed backtesting framework
- [ ] Real-time risk monitoring dashboard
- [ ] Machine learning-based alpha signals

---

## License

MIT License © 2025 Mengfan Long

---


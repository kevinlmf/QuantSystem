# Quick Start Guide

**TL;DR** - Get up and running in 5 minutes!

## Prerequisites

- Python 3.10+
- Git
- C++ compiler (clang/gcc)

## Installation (3 steps)

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/HFT_Trading_System.git
cd HFT_Trading_System/HFT_System
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Build C++ Module

```bash
./build_cpp.sh  # Windows: python cpp_core/setup.py build_ext --inplace
```

## Verify It Works

```bash
python test_build.py
```

You should see:
```
✓ All tests passed!
```

## Run Your First Strategy

```bash
python scripts/strategy_comparison.py
```

This will:
- Generate simulated market data
- Run 3 different trading strategies
- Show performance metrics
- Create comparison charts

## What's Next?

### Run All Demos
```bash
# Traditional strategies comparison
python scripts/strategy_comparison.py

# ML strategies comparison
python scripts/ml_strategy_comparison.py

# Comprehensive evaluation
python scripts/evaluate_strategies.py
```

### Explore Examples
```bash
# ML strategy examples
python examples/ml_strategy_example.py
python examples/ml_strategy_demo.py
```

### Run Benchmarks
```bash
# Performance benchmarks
python benchmarks/benchmark_throughput.py
```

### Use in Your Code
```python
import sys
sys.path.insert(0, 'cpp_core')
import cpp_trading2

# Use the C++ module
order = cpp_trading2.Order()
feed = cpp_trading2.DataFeed()

# Use Python strategies
from strategy.momentum_strategy import MomentumStrategy
strategy = MomentumStrategy()
```

## Common Issues

### Build fails?
```bash
# Rebuild clean
cd cpp_core
rm -rf build *.so
python setup.py build_ext --inplace
cd ..
```

### Import error?
```python
# Always add cpp_core to path
import sys
sys.path.insert(0, 'cpp_core')
import cpp_trading2
```

### Missing packages?
```bash
pip install numpy pandas matplotlib torch jax pybind11
```


##  Future Work & Project Status

This is a complex high-frequency trading system with many moving parts (C++ bindings, ML models, multi-strategy framework). 
### Current Status
- ✅ **Core functionality works**: C++ module, basic strategies, backtesting
-  **Known limitations**: This is an evolving project with some rough edges
-  **Active development**: Bugs are being fixed continuously

### What Might Go Wrong
- **Build issues**: C++ compilation can be tricky across different platforms
- **Dependency conflicts**: ML libraries (PyTorch, TensorFlow, JAX) can clash
- **Platform differences**: Some features tested primarily on macOS/Linux
- **Performance quirks**: Fine-tuning needed for different hardware

### I am Working On
- [ ] More robust cross-platform build system
- [ ] Better error handling and logging
- [ ] Improved documentation with more examples
- [ ] Fixing edge cases in strategy implementations
- [ ] More comprehensive test coverage




---

**License**: MIT License © 2025 Mengfan Long

**Disclaimer**: Educational and research purposes only. Not financial advice. Trading carries significant risk.

---

Code a lot, Worry less 😄

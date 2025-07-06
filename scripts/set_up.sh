#!/bin/bash

set -e
echo "📦 Setting up Risk-Aware Trading System..."

# 1. Create Python virtual environment (if it does not exist)
if [ ! -d "venv" ]; then
    echo "🧪 Creating virtual environment..."
    python3.10 -m venv venv
fi

# 2. Activate environment
echo "🔁 Activating environment..."
source venv/bin/activate

# 3. Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Build C++ PyBind11 module
echo "🔧 Building C++ PyBind11 module..."
mkdir -p cpp_core/build
cd cpp_core/build
cmake ..
make
cd ../../

# 5. Test if C++ module can be loaded
echo "🧪 Testing cpp_trading module..."
python scripts/test_order_executor.py

echo "✅ Setup complete. You can now run risk control or strategy scripts!"


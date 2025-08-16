#!/usr/bin/env bash
set -euo pipefail

echo "📦 Setting up Risk-Aware Trading System..."

# 1) venv
if [ ! -d "venv" ]; then
  echo "🧪 Creating virtual environment..."
  python3.10 -m venv venv
fi

echo "🔁 Activating environment..."
source venv/bin/activate

# 2) deps
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt || true
pip install pybind11

# 3) build C++ (CMake)
echo "🔧 Building C++ PyBind11 module (CMake)..."
mkdir -p cpp_core/build
pushd cpp_core/build >/dev/null
cmake ..
make -j
popd >/dev/null

# 4) 把 .so 放到 import 能找到的位置（优先放到 cpp_core/）
echo "📎 Locating built .so/.pyd..."
SO_FILE="$(find cpp_core -maxdepth 2 -type f \( -name 'cpp_trading2*.so' -o -name 'cpp_trading2*.pyd' \) | head -n1 || true)"
if [ -z "${SO_FILE}" ]; then
  echo "❌ Did not find cpp_trading2 module. Check CMakeLists and build output."
  exit 1
fi

# 确保 cpp_core/ 下有目标扩展（复制或软链）
TARGET_DIR="cpp_core"
BASENAME="$(basename "$SO_FILE")"
if [ ! -f "${TARGET_DIR}/${BASENAME}" ]; then
  # 优先用软链（可改为 cp）
  ln -sf "$(realpath "$SO_FILE")" "${TARGET_DIR}/${BASENAME}"
fi

# 5) 兼容包：让 import cpp_trading 也能用
mkdir -p cpp_trading
cat > cpp_trading/__init__.py <<'PY'
try:
    # 优先导入顶层模块名
    from cpp_trading2 import *
except Exception:
    # 回退：部分构建/运行路径下要带包前缀
    from cpp_core.cpp_trading2 import *
PY

# 6) 运行时搜索路径（根目录 + cpp_core + build）
export PYTHONPATH="$(pwd):$(pwd)/cpp_core:$(pwd)/cpp_core/build:${PYTHONPATH:-}"

# 7) quick test
echo "🧪 Quick test: importing cpp_trading2..."
python - <<'PY'
import sys
print("sys.path sample:", sys.path[:3])
import cpp_trading2 as m
print("✅ Loaded:", m.__name__)
PY

# 8) script self-test
echo "🧪 Running test_order_executor.py..."
python scripts/test_order_executor.py || (echo "⚠️ test_order_executor failed"; exit 1)

# 9) pytest suite
echo "🧪 Running pytest..."
pytest -q scripts || (echo "⚠️ pytest failed"; exit 1)

echo "✅ Setup complete. All good!"



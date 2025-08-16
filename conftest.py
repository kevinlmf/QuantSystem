# conftest.py (at project root)
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CPP_CORE = ROOT / "cpp_core"
BUILD = CPP_CORE / "build"

for p in (ROOT, CPP_CORE, BUILD):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)



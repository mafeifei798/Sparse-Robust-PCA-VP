from __future__ import annotations

import os
from pathlib import Path
import sys


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vp_rspca_mpl")

# Tests exercise both the installable ``src`` package and repository-local
# experiment/compatibility entry points.  Keep collection independent of
# whether pytest was launched as ``pytest`` or ``python -m pytest``.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

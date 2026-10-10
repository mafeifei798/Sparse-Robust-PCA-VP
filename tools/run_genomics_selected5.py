"""Launch the source-locked SNP-selection experiment with its import path configured.

Use ``python -m tools.run_genomics_selected5 --help`` from the checkout or
invoke this file by absolute path from another directory. Arguments and the
exit code are passed through; no protocol, solver or scientific budget is
changed. The original script and its frozen source hashes remain untouched.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    environment = dict(os.environ)
    python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + python_path if python_path else ""
    )
    # Script execution puts experiments/ on the child's import path, exactly
    # as the source-bound historical runner expects. No global sys.path or
    # sys.modules mutation is made in a host notebook/test process.
    completed = subprocess.run(
        [sys.executable, str(ROOT / "experiments/genomics_randomized_vp_selected5.py"), *arguments],
        env=environment,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

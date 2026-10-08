"""Check that the non-Fed test/CLI source closure is present in Git's index.

This is a read-only source inventory, not an artifact or scientific validator.
Downloaded inputs and sealed result payloads deliberately remain separate.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def required_sources(root: Path = ROOT) -> set[str]:
    modules: dict[str, str] = {}
    for directory, prefix in ((root, ""), (root / "src/rspca", "rspca"),
                              (root / "experiments", "experiments"), (root / "tools", "tools")):
        for path in directory.glob("*.py"):
            name = prefix if path.stem == "__init__" else ".".join(filter(None, (prefix, path.stem)))
            relative = path.relative_to(root).as_posix()
            modules[name] = relative
            if prefix in {"experiments", "tools"}:
                modules.setdefault(path.stem, relative)  # historical script imports

    makefile = (root / "Makefile").read_text(encoding="utf-8")
    required = set(re.findall(r"tests/test_[\w]+\.py", makefile.split("NONFED_SMOKE_ROOT", 1)[0]))
    required.update({
        ".gitignore", "pyproject.toml", "Makefile", ".github/workflows/tests.yml",
        "README.md", "LICENSE", "tools/check_nonfed_source_tracking.py",
        "tools/run_genomics_selected5.py", "experiments/genomics_randomized_vp_selected5.py",
        "tools/rebuild_notebooks.py", "tools/execute_nonfed_notebooks.py",
        "tools/generate_genomics_smoke_data.py", "experiments/spca_benchmark.py",
        "experiments/video_spca.py", "experiments/video_spca_benchmark.py", "experiments/genomics_1kg.py",
        "tests/conftest.py",
        "notebooks/spca_benchmark.ipynb", "notebooks/video_spca.ipynb", "notebooks/genomics_1kg.ipynb",
    })
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    required.update(name + ".py" for name in metadata["tool"]["setuptools"]["py-modules"])

    pending = list(required)
    visited: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in visited or not relative.endswith(".py"):
            continue
        visited.add(relative)
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source is missing: {relative}")
        parts = list(path.relative_to(root).with_suffix("").parts)
        if parts[0] == "src":
            parts.pop(0)
        package = parts[:-1]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=relative)):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(package[:len(package) - node.level + 1] + ([base] if base else []))
                names = [base, *(base + "." + alias.name for alias in node.names)]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value]  # explicit importlib.import_module literals
            for name in names:
                dependency = modules.get(name)
                if dependency is not None and dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
        # Parent package initialization is also part of the import contract.
        for length in range(1, len(package) + 1):
            dependency = modules.get(".".join(package[:length]))
            if dependency is not None and dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    return required


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print missing paths as JSON")
    arguments = parser.parse_args(argv)
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    missing = sorted(required_sources() - tracked)
    print(json.dumps(missing) if arguments.json else
          ("Non-Fed source closure is indexed." if not missing else "Missing from Git index:\n" + "\n".join(missing)))
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())

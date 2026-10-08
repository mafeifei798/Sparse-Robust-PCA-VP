#!/usr/bin/env python3
"""Execute the canonical non-federated notebooks as headless smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOKS = (
    ROOT / "notebooks" / "spca_benchmark.ipynb",
    ROOT / "notebooks" / "video_spca.ipynb",
    ROOT / "notebooks" / "genomics_1kg.ipynb",
)


def execute_notebooks(
    notebooks: Sequence[Path],
    *,
    output_dir: Path,
    timeout: int = 600,
) -> tuple[Path, ...]:
    """Execute notebooks from the repository root and save disposable copies."""

    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer")
    if not notebooks:
        raise ValueError("at least one notebook is required")

    resolved: list[Path] = []
    for notebook in notebooks:
        path = Path(notebook).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".ipynb":
            raise ValueError(f"not a notebook: {path}")
        resolved.append(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for path in resolved:
        document = nbformat.read(path, as_version=4)
        client = NotebookClient(
            document,
            timeout=timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
        destination = output_dir / f"{path.stem}.executed.ipynb"
        nbformat.write(document, destination)
        outputs.append(destination)
    return tuple(outputs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "smoke" / "nonfed" / "notebooks",
    )
    parser.add_argument("--timeout", type=int, default=600)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    selected = tuple(arguments.notebooks) or DEFAULT_NOTEBOOKS
    outputs = execute_notebooks(
        selected,
        output_dir=arguments.output_dir,
        timeout=arguments.timeout,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


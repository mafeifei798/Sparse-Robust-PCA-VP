#!/usr/bin/env python3
"""Reproducible solver-only and alpha-grid timing for randomized VP-SPCA.

The output deliberately keeps a single closed-form PCA fit separate from the
iterative alpha-grid totals.  It never presents one PCA fit as a matched timing
baseline for several randomized fits and optional full-data refinement phases.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy
import rspca.randomized as randomized_module
import rspca.vp as vp_module

from rspca.randomized import (
    fit_randomized_vp_spca,
    fit_randomized_vp_spca_from_workspace,
    prepare_randomized_vp_spca_workspace,
)
from rspca.vp import fit_vp_spca


_TOOL_MODULE = "tools.benchmark_randomized_runtime"
_TOOL_PATH = Path(__file__).resolve(strict=True)
_REPOSITORY_ROOT = _TOOL_PATH.parents[1]


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _alpha_grid(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "alphas must be comma-separated floats"
        ) from error
    if not values or any(not np.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("alphas must be finite and non-negative")
    return values


def _matrix(n_samples: int, n_features: int, rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(n_samples, rank))
    right = rng.normal(size=(n_features, rank))
    matrix = left @ right.T / np.sqrt(float(rank))
    matrix += 0.05 * rng.normal(size=matrix.shape)
    matrix -= matrix.mean(axis=0, keepdims=True)
    return np.asarray(matrix, dtype=np.float64)


def _sha256_file(path: Path) -> str:
    """Hash the exact bytes loaded from a source file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matrix_provenance(matrix: np.ndarray) -> dict[str, object]:
    """Bind a matrix to dtype, shape, canonical content, and a combined hash."""

    array = np.asarray(matrix)
    if array.dtype.hasobject:
        raise TypeError("benchmark matrix must not have object dtype")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    content = canonical.tobytes(order="C")
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    combined_digest = hashlib.sha256()
    combined_digest.update(header)
    combined_digest.update(b"\0")
    combined_digest.update(content)
    return {
        "generator": "deterministic_gaussian_low_rank_plus_noise_v1",
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "array_sha256": combined_digest.hexdigest(),
        "array_sha256_contract": (
            "sha256(canonical_sorted_compact_dtype_shape_json + NUL + "
            "little_endian_c_order_content)"
        ),
    }


def _source_path(module: ModuleType) -> Path:
    source = inspect.getsourcefile(module)
    if source is None:
        raise RuntimeError(f"cannot resolve Python source for {module.__name__}")
    return Path(source).resolve(strict=True)


def _relative_source_record(path: Path, *, module: str) -> dict[str, str]:
    """Describe a source without emitting an absolute, user-specific path."""

    try:
        relative_path = path.relative_to(_REPOSITORY_ROOT).as_posix()
        relative_to = "repository_root"
    except ValueError:
        # An installed copy can live outside this checkout.  Its module-relative
        # identity and exact byte hash are still useful, while its absolute path
        # could expose a user name or environment layout.
        relative_path = f"{module.replace('.', '/')}{path.suffix}"
        relative_to = "python_import_root"
    return {
        "module": module,
        "relative_path": relative_path,
        "relative_to": relative_to,
        "sha256": _sha256_file(path),
    }


def _source_provenance() -> dict[str, dict[str, str]]:
    return {
        "randomized": _relative_source_record(
            _source_path(randomized_module), module=randomized_module.__name__
        ),
        "vp": _relative_source_record(_source_path(vp_module), module=vp_module.__name__),
        "benchmark_tool": _relative_source_record(_TOOL_PATH, module=_TOOL_MODULE),
    }


def _thread_control_environment() -> dict[str, str | None]:
    """Record useful thread settings without copying arbitrary env contents."""

    controls: dict[str, str | None] = {}
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        value = os.environ.get(name)
        if value is None:
            controls[name] = None
        elif value.isdecimal():
            controls[name] = value
        else:
            controls[name] = "set_non_numeric_value_redacted"
    return controls


def _strict_json_compatible(value: Any, *, location: str = "root") -> Any:
    """Return JSON-native data and reject nested NaN/Inf with its location."""

    if isinstance(value, np.generic):
        return _strict_json_compatible(value.item(), location=location)
    if isinstance(value, np.ndarray):
        return _strict_json_compatible(value.tolist(), location=location)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key must be a string at {location}")
            normalized[key] = _strict_json_compatible(
                item, location=f"{location}.{key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_compatible(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"number must be finite at {location}")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(
        f"value of type {type(value).__name__} is not JSON serializable at {location}"
    )


def _serialize_report(report: Mapping[str, Any]) -> str:
    compatible = _strict_json_compatible(report)
    return json.dumps(
        compatible,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )


def _preflight_output_json(path: Path | None) -> None:
    if path is not None and os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _write_text_exclusive(path: Path, text: str) -> None:
    """Create ``path`` atomically with respect to competing creators."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    file_identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            path_identity = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if (
                path_identity.st_dev == file_identity.st_dev
                and path_identity.st_ino == file_identity.st_ino
            ):
                path.unlink()
        raise


def _samples(
    operation: Callable[[], object],
    *,
    warmups: int,
    repeats: int,
) -> list[float]:
    for _ in range(warmups):
        operation()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        operation()
        samples.append(time.perf_counter() - start)
    return samples


def _timing_record(samples: list[float]) -> dict[str, object]:
    return {
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples_seconds": samples,
    }


def run_benchmark(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.rank > min(arguments.n_samples, arguments.n_features):
        raise ValueError("rank must not exceed min(n_samples, n_features)")
    matrix = _matrix(
        arguments.n_samples,
        arguments.n_features,
        arguments.rank,
        arguments.seed,
    )
    matrix_provenance = _matrix_provenance(matrix)
    first_alpha = arguments.alphas[0]
    randomized_common = {
        "oversample": arguments.oversample,
        "n_power_iter": arguments.n_power_iter,
        "max_iter": arguments.max_iter,
        "tol": arguments.tol,
        "random_state": arguments.seed + 1,
    }

    pca_samples = _samples(
        lambda: np.linalg.svd(matrix, full_matrices=False),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    vp_samples = _samples(
        lambda: fit_vp_spca(
            matrix,
            arguments.rank,
            alpha=first_alpha,
            max_iter=arguments.max_iter,
            tol=arguments.tol,
            backtracking=True,
        ),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    vp_probe = fit_vp_spca(
        matrix,
        arguments.rank,
        alpha=first_alpha,
        max_iter=arguments.max_iter,
        tol=arguments.tol,
        backtracking=True,
    )
    randomized_cold_samples = _samples(
        lambda: fit_randomized_vp_spca(
            matrix,
            arguments.rank,
            alpha=first_alpha,
            finalize_full_iters=0,
            **randomized_common,
        ),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    randomized_probe = fit_randomized_vp_spca(
        matrix,
        arguments.rank,
        alpha=first_alpha,
        finalize_full_iters=0,
        **randomized_common,
    )

    workspace = prepare_randomized_vp_spca_workspace(
        matrix,
        arguments.rank,
        oversample=arguments.oversample,
        n_power_iter=arguments.n_power_iter,
        random_state=arguments.seed + 1,
    )
    randomized_reused_samples = _samples(
        lambda: fit_randomized_vp_spca_from_workspace(
            workspace,
            alpha=first_alpha,
            max_iter=arguments.max_iter,
            tol=arguments.tol,
            finalize_full_iters=0,
        ),
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )

    cold_results = [
        fit_randomized_vp_spca(
            matrix,
            arguments.rank,
            alpha=alpha,
            finalize_full_iters=arguments.finalize_full_iters,
            **randomized_common,
        )
        for alpha in arguments.alphas
    ]
    shared_results = [
        fit_randomized_vp_spca_from_workspace(
            workspace,
            alpha=alpha,
            max_iter=arguments.max_iter,
            tol=arguments.tol,
            finalize_full_iters=arguments.finalize_full_iters,
        )
        for alpha in arguments.alphas
    ]
    max_abs_differences = {
        "A": max(
            float(np.max(np.abs(cold.A - shared.A)))
            for cold, shared in zip(cold_results, shared_results, strict=True)
        ),
        "B": max(
            float(np.max(np.abs(cold.B - shared.B)))
            for cold, shared in zip(cold_results, shared_results, strict=True)
        ),
        "objectives": max(
            float(
                np.max(
                    np.abs(np.asarray(cold.objectives) - np.asarray(shared.objectives))
                )
            )
            for cold, shared in zip(cold_results, shared_results, strict=True)
        ),
        "stationarity": max(
            float(
                np.max(
                    np.abs(
                        np.asarray(cold.stationarity) - np.asarray(shared.stationarity)
                    )
                )
            )
            for cold, shared in zip(cold_results, shared_results, strict=True)
        ),
        "step_sizes": max(
            float(
                np.max(
                    np.abs(np.asarray(cold.step_sizes) - np.asarray(shared.step_sizes))
                )
            )
            for cold, shared in zip(cold_results, shared_results, strict=True)
        ),
    }

    def cold_grid() -> None:
        for alpha in arguments.alphas:
            fit_randomized_vp_spca(
                matrix,
                arguments.rank,
                alpha=alpha,
                finalize_full_iters=arguments.finalize_full_iters,
                **randomized_common,
            )

    def shared_grid() -> None:
        prepared = prepare_randomized_vp_spca_workspace(
            matrix,
            arguments.rank,
            oversample=arguments.oversample,
            n_power_iter=arguments.n_power_iter,
            random_state=arguments.seed + 1,
        )
        for alpha in arguments.alphas:
            fit_randomized_vp_spca_from_workspace(
                prepared,
                alpha=alpha,
                max_iter=arguments.max_iter,
                tol=arguments.tol,
                finalize_full_iters=arguments.finalize_full_iters,
            )

    cold_grid_samples = _samples(
        cold_grid,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    shared_grid_samples = _samples(
        shared_grid,
        warmups=arguments.warmups,
        repeats=arguments.repeats,
    )
    cold_median = statistics.median(cold_grid_samples)
    shared_median = statistics.median(shared_grid_samples)

    vp_final_objective = float(vp_probe.objectives[-1])
    vp_final_stationarity = float(vp_probe.stationarity[-1])
    randomized_full_objective = float(randomized_probe.full_objective)
    randomized_full_stationarity = float(randomized_probe.full_stationarity)
    if vp_final_objective == 0.0:
        raise ValueError("VP final objective must be nonzero for a relative gap")
    randomized_minus_vp_objective = randomized_full_objective - vp_final_objective

    return {
        "schema_version": 2,
        "scope": {
            "pca": "one closed-form fit; not matched to an iterative alpha grid",
            "solver_only": (
                "one alpha; VP and randomized share max_iter/tol, while randomized "
                "also includes range preparation and mandatory full diagnostics"
            ),
            "alpha_grid": (
                "all declared alphas plus optional full refinement; shared timing "
                "charges randomized workspace preparation exactly once"
            ),
            "runtime_inference": "descriptive local wall clock; no uncertainty claim",
        },
        "config": {
            "n_samples": arguments.n_samples,
            "n_features": arguments.n_features,
            "rank": arguments.rank,
            "alphas": list(arguments.alphas),
            "oversample": arguments.oversample,
            "n_power_iter": arguments.n_power_iter,
            "max_iter": arguments.max_iter,
            "finalize_full_iters": arguments.finalize_full_iters,
            "tol": arguments.tol,
            "seed": arguments.seed,
            "warmups": arguments.warmups,
            "repeats": arguments.repeats,
        },
        "deterministic_matrix": matrix_provenance,
        "source_files": _source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.system(),
            "machine": platform.machine(),
            "executable": Path(sys.executable).name,
            "executable_path_policy": "basename_only_to_avoid_sensitive_paths",
            "thread_controls": _thread_control_environment(),
        },
        "pca_single_fit": _timing_record(pca_samples),
        "solver_only_one_alpha": {
            "alpha": first_alpha,
            "vp": {
                **_timing_record(vp_samples),
                "converged": vp_probe.converged,
                "n_iter": vp_probe.n_iter,
                "status": vp_probe.status,
                "final_objective": vp_final_objective,
                "final_stationarity": vp_final_stationarity,
            },
            "randomized_cold": {
                **_timing_record(randomized_cold_samples),
                "converged": randomized_probe.converged,
                "n_iter": randomized_probe.n_iter,
                "status": randomized_probe.status,
                "full_objective": randomized_full_objective,
                "full_stationarity": randomized_full_stationarity,
            },
            "randomized_reused_workspace_fit_only": _timing_record(
                randomized_reused_samples
            ),
            "workspace_prepare_seconds_single_observation": workspace.timings["total"],
        },
        "fixed_budget_quality_one_alpha": {
            "interpretation": (
                "descriptive full-matrix diagnostics at the declared fixed iteration "
                "budget; not a convergence or quality-matched claim"
            ),
            "alpha": first_alpha,
            "declared_max_iter": arguments.max_iter,
            "randomized_full_refinement_iters": 0,
            "vp": {
                "final_objective": vp_final_objective,
                "final_stationarity": vp_final_stationarity,
                "converged": vp_probe.converged,
                "status": vp_probe.status,
            },
            "randomized": {
                "full_objective": randomized_full_objective,
                "full_stationarity": randomized_full_stationarity,
                "converged": randomized_probe.converged,
                "status": randomized_probe.status,
            },
            "randomized_minus_vp_objective": randomized_minus_vp_objective,
            "randomized_minus_vp_objective_relative_to_abs_vp": (
                randomized_minus_vp_objective / abs(vp_final_objective)
            ),
        },
        "randomized_alpha_grid": {
            "cold_reprepare_every_alpha": _timing_record(cold_grid_samples),
            "shared_prepare_once": _timing_record(shared_grid_samples),
            "median_speedup_ratio_cold_over_shared": cold_median / shared_median,
            "median_fraction_saved": 1.0 - shared_median / cold_median,
            "cold_vs_shared_max_abs_differences": max_abs_differences,
            "per_alpha_diagnostics": [
                {
                    "alpha": alpha,
                    "converged": result.converged,
                    "n_iter": result.n_iter,
                    "status": result.status,
                    "full_stationarity": result.full_stationarity,
                }
                for alpha, result in zip(
                    arguments.alphas,
                    cold_results,
                    strict=True,
                )
            ],
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=_positive_integer, default=400)
    parser.add_argument("--n-features", type=_positive_integer, default=200)
    parser.add_argument("--rank", type=_positive_integer, default=5)
    parser.add_argument("--alphas", type=_alpha_grid, default=(0.01, 0.03, 0.1))
    parser.add_argument("--oversample", type=_nonnegative_integer, default=10)
    parser.add_argument("--n-power-iter", type=_nonnegative_integer, default=2)
    parser.add_argument("--max-iter", type=_positive_integer, default=100)
    parser.add_argument("--finalize-full-iters", type=_nonnegative_integer, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--seed", type=_nonnegative_integer, default=260829)
    parser.add_argument("--warmups", type=_nonnegative_integer, default=1)
    parser.add_argument("--repeats", type=_positive_integer, default=5)
    parser.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    _preflight_output_json(arguments.output_json)
    if not np.isfinite(arguments.tol) or arguments.tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    report = run_benchmark(arguments)
    serialized = _serialize_report(report)
    if arguments.output_json is not None:
        _write_text_exclusive(arguments.output_json, serialized + "\n")
    sys.stdout.write(serialized + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

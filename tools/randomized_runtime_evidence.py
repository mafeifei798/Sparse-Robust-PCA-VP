#!/usr/bin/env python3
"""Generate and validate the paired Randomized VP-SPCA runtime evidence.

The committed evidence executes two immutable source snapshots against the
same deterministic matrices.  It is deliberately separate from the general
runtime benchmark: this tool answers the narrower before/after optimization
question and preserves the exact pre-optimization execution source.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter
from types import ModuleType
from typing import Any, Mapping, Sequence

import numpy as np
import scipy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE = (
    REPOSITORY_ROOT
    / "results"
    / "randomized_runtime_optimization_2026-08-30"
    / "evidence.json"
)
EVIDENCE_PACKAGE_MANIFEST_NAME = "manifest.json"
ASSET_ROOT = DEFAULT_EVIDENCE.parent / "provenance" / "assets"
BEFORE_SNAPSHOT = ASSET_ROOT / "randomized_before.py"
AFTER_SNAPSHOT = ASSET_ROOT / "randomized_after.py"
VP_SNAPSHOT = ASSET_ROOT / "vp_dependency.py"
BEFORE_SOURCE_SHA256 = (
    "37b68488d25abf775f97ec7ed755a1f5b00aceaa6fd08e29d044fe01e39fe0c5"
)
AFTER_SOURCE_SHA256 = "706a05cc1f29dcabd34e63f1968fcda378984838d85bace15d6d5539d2a2a046"
VP_SOURCE_SHA256 = "cae985e15e87d5ebf88c9c68c7570ed10832df30ec1245b7b8031def2bd6b9c8"
EXPECTED_EVIDENCE_PACKAGE_PATHS = frozenset(
    {
        "evidence.json",
        "provenance/assets/randomized_before.py",
        "provenance/assets/randomized_after.py",
        "provenance/assets/vp_dependency.py",
    }
)
FORMAL_SOURCE_MANIFEST = (
    REPOSITORY_ROOT
    / "results"
    / "nonfed_2026-08-29"
    / "provenance"
    / "source_manifest.json"
)

THREAD_CONTROL_NAMES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
NUMERIC_RESULT_FIELDS = (
    "A",
    "B",
    "Q",
    "objectives",
    "stationarity",
    "step_sizes",
    "full_objective",
    "full_stationarity",
    "full_diagnostic_step_size",
    "sketch_residual",
    "sketch_residual_relative",
)
DISCRETE_RESULT_FIELDS = (
    "converged",
    "sketch_converged",
    "full_converged",
    "n_iter",
    "status",
    "full_diagnostic_step_policy",
    "used_full_gradient",
    "sketch_iterations",
    "full_refinement_iterations",
)
MAX_ABS_TOLERANCES = {
    "A": 1e-12,
    "B": 1e-12,
    "Q": 1e-12,
    "objectives": 1e-9,
    "stationarity": 1e-10,
    "step_sizes": 1e-15,
    # The optimized diagnostics use equivalent matrix identities whose final
    # summation order differs.  On the 957x5000 case 1.4e-8 absolute is only
    # 2.4e-12 relative, so this remains a conservative numerical tolerance.
    "full_objective": 1e-7,
    "full_stationarity": 1e-10,
    "full_diagnostic_step_size": 1e-15,
    "sketch_residual": 1e-8,
    "sketch_residual_relative": 1e-11,
}

CASE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "case_id": "formal_400x200_full_refine",
        "description": (
            "8/29 formal synthetic shape with 300 full-data refinement steps"
        ),
        "matrix": {
            "n_samples": 400,
            "n_features": 200,
            "rank": 5,
            "seed": 101,
        },
        "solver": {
            "n_components": 5,
            "alpha": 0.02,
            "oversample": 20,
            "n_power_iter": 3,
            "max_iter": 500,
            "tol": 1e-5,
            "use_full_gradient": False,
            "finalize_full_iters": 300,
            "random_state": 123,
            "verbose": False,
        },
    },
    {
        "case_id": "video_wide_180x1024",
        "description": "video-like wide matrix, sketch solve plus full diagnostics",
        "matrix": {
            "n_samples": 180,
            "n_features": 1024,
            "rank": 3,
            "seed": 102,
        },
        "solver": {
            "n_components": 3,
            "alpha": 0.08,
            "oversample": 6,
            "n_power_iter": 2,
            "max_iter": 1000,
            "tol": 1e-5,
            "use_full_gradient": False,
            "finalize_full_iters": 0,
            "random_state": 123,
            "verbose": False,
        },
    },
    {
        "case_id": "genomics_wide_957x5000",
        "description": "genomics-like very-wide matrix under a fixed 20-step budget",
        "matrix": {
            "n_samples": 957,
            "n_features": 5000,
            "rank": 2,
            "seed": 103,
        },
        "solver": {
            "n_components": 2,
            "alpha": 0.08,
            "oversample": 8,
            "n_power_iter": 1,
            "max_iter": 20,
            "tol": 1e-6,
            "use_full_gradient": False,
            "finalize_full_iters": 0,
            "random_state": 123,
            "verbose": False,
        },
    },
)


class EvidenceValidationError(ValueError):
    """Raised when the evidence package violates its strict contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_relative(path: Path) -> str:
    return path.resolve(strict=True).relative_to(REPOSITORY_ROOT).as_posix()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise EvidenceValidationError(f"non-finite JSON constant: {value}")


def _strict_json_load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot load strict JSON: {error}") from error
    if not isinstance(payload, dict):
        raise EvidenceValidationError("evidence root must be a JSON object")
    _assert_finite_json(payload)
    return payload


def _assert_finite_json(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceValidationError(
                    f"non-string JSON key at {location}: {key!r}"
                )
            _assert_finite_json(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise EvidenceValidationError(f"non-finite number at {location}")
    elif not isinstance(value, (str, int, bool, float, type(None))):
        raise EvidenceValidationError(
            f"unsupported JSON value {type(value).__name__} at {location}"
        )


def _strict_json_native(value: Any, *, location: str = "root") -> Any:
    if isinstance(value, np.generic):
        return _strict_json_native(value.item(), location=location)
    if isinstance(value, np.ndarray):
        return _strict_json_native(value.tolist(), location=location)
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON key must be a string at {location}")
            normalized[key] = _strict_json_native(item, location=f"{location}.{key}")
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _strict_json_native(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"number must be finite at {location}")
    if isinstance(value, (str, int, bool, float)) or value is None:
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__} at {location}")


def _canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body.pop("canonical_payload_sha256", None)
    serialized = json.dumps(
        _strict_json_native(body),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    serialized = json.dumps(
        _strict_json_native(payload),
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            descriptor = -1
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        raise


def _matrix(config: Mapping[str, Any]) -> np.ndarray:
    rng = np.random.default_rng(int(config["seed"]))
    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    rank = int(config["rank"])
    left = rng.normal(size=(n_samples, rank))
    right = rng.normal(size=(n_features, rank))
    result = left @ right.T / np.sqrt(float(rank))
    result += 0.05 * rng.normal(size=result.shape)
    result -= result.mean(axis=0, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("result arrays must not have object dtype")
    if not np.all(np.isfinite(array)):
        raise ValueError("result arrays must be finite")
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    content = canonical.tobytes(order="C")
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    combined = hashlib.sha256(header + b"\0" + content).hexdigest()
    return {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "array_sha256": combined,
    }


def _read_bound_source(path: Path, expected_sha256: str) -> bytes:
    """Read source bytes and reject anything outside the declared identity."""

    try:
        source = path.read_bytes()
    except OSError as error:
        raise EvidenceValidationError(
            f"cannot read bound source {path}: {error}"
        ) from error
    actual_sha256 = hashlib.sha256(source).hexdigest()
    if actual_sha256 != expected_sha256:
        raise EvidenceValidationError(
            f"bound source SHA-256 mismatch for {path}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return source


def _exec_source_module(
    module_name: str,
    path: Path,
    expected_sha256: str,
    *,
    package_name: str,
) -> ModuleType:
    """Compile exact source bytes without consulting or writing bytecode caches."""

    source = _read_bound_source(path, expected_sha256)
    code = compile(source, str(path), "exec", dont_inherit=True)
    module = ModuleType(module_name)
    module.__file__ = str(path)
    module.__cached__ = None
    module.__loader__ = None
    module.__package__ = package_name
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_snapshot(
    label: str,
    randomized_path: Path,
    *,
    randomized_sha256: str,
    vp_path: Path | None = None,
    vp_sha256: str = VP_SOURCE_SHA256,
    asset_root: Path | None = None,
) -> ModuleType:
    """Load bound sources into an isolated package without any ``.pyc`` I/O."""

    resolved_vp_path = VP_SNAPSHOT if vp_path is None else vp_path
    resolved_asset_root = ASSET_ROOT if asset_root is None else asset_root
    package_name = f"_rspca_runtime_evidence_{label}"
    package = ModuleType(package_name)
    package.__path__ = [str(resolved_asset_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    package.__file__ = None
    package.__cached__ = None
    package.__loader__ = None
    package.__spec__ = None
    sys.modules[package_name] = package

    try:
        _exec_source_module(
            f"{package_name}.vp",
            resolved_vp_path,
            vp_sha256,
            package_name=package_name,
        )
        return _exec_source_module(
            f"{package_name}.randomized",
            randomized_path,
            randomized_sha256,
            package_name=package_name,
        )
    except BaseException:
        sys.modules.pop(f"{package_name}.vp", None)
        sys.modules.pop(f"{package_name}.randomized", None)
        sys.modules.pop(package_name, None)
        raise


def _fit(module: ModuleType, matrix: np.ndarray, solver: Mapping[str, Any]) -> Any:
    options = dict(solver)
    n_components = int(options.pop("n_components"))
    return module.fit_randomized_vp_spca(matrix, n_components, **options)


def _equivalence_record(before: Any, after: Any) -> dict[str, Any]:
    numeric: dict[str, Any] = {}
    for field in NUMERIC_RESULT_FIELDS:
        before_array = np.asarray(getattr(before, field))
        after_array = np.asarray(getattr(after, field))
        if before_array.shape != after_array.shape:
            raise ValueError(f"result shape mismatch for {field}")
        difference = np.asarray(after_array - before_array, dtype=np.float64)
        max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
        denominator = max(
            float(np.linalg.norm(before_array.ravel())), np.finfo(np.float64).tiny
        )
        relative_l2 = float(np.linalg.norm(difference.ravel()) / denominator)
        tolerance = float(MAX_ABS_TOLERANCES[field])
        numeric[field] = {
            "before": _array_identity(before_array),
            "after": _array_identity(after_array),
            "max_abs_difference": max_abs,
            "relative_l2_difference": relative_l2,
            "max_abs_tolerance": tolerance,
            "within_tolerance": bool(max_abs <= tolerance),
        }

    discrete: dict[str, Any] = {}
    for field in DISCRETE_RESULT_FIELDS:
        before_value = _strict_json_native(getattr(before, field))
        after_value = _strict_json_native(getattr(after, field))
        discrete[field] = {
            "before": before_value,
            "after": after_value,
            "equal": bool(before_value == after_value),
        }

    return {
        "numeric_fields": numeric,
        "discrete_fields": discrete,
        "all_numeric_within_tolerance": all(
            record["within_tolerance"] for record in numeric.values()
        ),
        "all_discrete_equal": all(record["equal"] for record in discrete.values()),
        "timing_fields_intentionally_excluded": True,
    }


def _timed_fit(
    module: ModuleType, matrix: np.ndarray, solver: Mapping[str, Any]
) -> float:
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        start = perf_counter()
        _fit(module, matrix, solver)
        elapsed = perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise RuntimeError(f"invalid elapsed time: {elapsed}")
    return float(elapsed)


def _thread_controls() -> dict[str, str]:
    controls: dict[str, str] = {}
    for name in THREAD_CONTROL_NAMES:
        value = os.environ.get(name)
        if value != "1":
            raise RuntimeError(
                f"{name}=1 is required before Python starts; got {value!r}"
            )
        controls[name] = value
    return controls


def _source_record(path: Path) -> dict[str, Any]:
    return {
        "path": _repository_relative(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _formal_manifest_binding() -> dict[str, Any]:
    manifest = _strict_json_load(FORMAL_SOURCE_MANIFEST)
    execution_files = manifest.get("execution_files")
    if not isinstance(execution_files, list):
        raise EvidenceValidationError("formal source manifest lacks execution_files")
    entries = {
        record.get("path"): record
        for record in execution_files
        if isinstance(record, dict)
        and record.get("path") in {"src/rspca/randomized.py", "src/rspca/vp.py"}
    }
    if set(entries) != {"src/rspca/randomized.py", "src/rspca/vp.py"}:
        raise EvidenceValidationError(
            "formal manifest must contain randomized.py and vp.py exactly once"
        )
    return {
        "source_manifest": _source_record(FORMAL_SOURCE_MANIFEST),
        "manifest_entries": {
            name: {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
            }
            for name, record in sorted(entries.items())
        },
        "randomized_execution_snapshot_recovered": True,
        "randomized_closure_scope": "randomized.py plus its exercised vp.py dependency",
        "full_release_source_closure_claimed": False,
        "before_snapshot_is_origin_main": False,
        "before_snapshot_provenance": {
            "kind": "recovered_formal_autodl_execution_snapshot",
            "remote_execution_path": (
                "/root/autodl-tmp/dhy/Robust-and-Sparse-PCA-nonfed-20260829/"
                "source/src/rspca/randomized.py"
            ),
            "retrieved_read_only": True,
        },
    }


def generate_evidence(*, warmups: int = 1, repeats: int = 7) -> dict[str, Any]:
    if warmups < 0:
        raise ValueError("warmups must be non-negative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    controls = _thread_controls()
    before_module = _load_snapshot(
        "before", BEFORE_SNAPSHOT, randomized_sha256=BEFORE_SOURCE_SHA256
    )
    after_module = _load_snapshot(
        "after", AFTER_SNAPSHOT, randomized_sha256=AFTER_SOURCE_SHA256
    )
    formal_binding = _formal_manifest_binding()
    source_records = {
        "before": {
            **_source_record(BEFORE_SNAPSHOT),
            "role": "pre_optimization_formal_execution_snapshot",
        },
        "after": {
            **_source_record(AFTER_SNAPSHOT),
            "role": "post_optimization_local_execution_snapshot",
        },
        "vp_dependency": {
            **_source_record(VP_SNAPSHOT),
            "role": "shared_exercised_execution_dependency",
        },
    }
    fixed_source_hashes = {
        "before": BEFORE_SOURCE_SHA256,
        "after": AFTER_SOURCE_SHA256,
        "vp_dependency": VP_SOURCE_SHA256,
    }
    for label, expected_sha256 in fixed_source_hashes.items():
        if source_records[label]["sha256"] != expected_sha256:
            raise EvidenceValidationError(
                f"fixed execution source SHA-256 mismatch for {label}"
            )

    manifest_entries = formal_binding["manifest_entries"]
    if (
        source_records["before"]["sha256"]
        != manifest_entries["src/rspca/randomized.py"]["sha256"]
    ):
        raise EvidenceValidationError(
            "before snapshot does not match formal randomized.py manifest entry"
        )
    if (
        source_records["vp_dependency"]["sha256"]
        != manifest_entries["src/rspca/vp.py"]["sha256"]
    ):
        raise EvidenceValidationError(
            "VP snapshot does not match formal vp.py manifest entry"
        )

    case_records: list[dict[str, Any]] = []
    for specification in CASE_SPECS:
        matrix = _matrix(specification["matrix"])
        before_result = _fit(before_module, matrix, specification["solver"])
        after_result = _fit(after_module, matrix, specification["solver"])
        equivalence = _equivalence_record(before_result, after_result)
        if not equivalence["all_numeric_within_tolerance"]:
            raise RuntimeError(
                f"numeric equivalence failed for {specification['case_id']}"
            )
        if not equivalence["all_discrete_equal"]:
            raise RuntimeError(
                f"discrete equivalence failed for {specification['case_id']}"
            )

        for _ in range(warmups):
            _fit(before_module, matrix, specification["solver"])
            _fit(after_module, matrix, specification["solver"])

        before_samples: list[float] = []
        after_samples: list[float] = []
        execution_order: list[list[str]] = []
        for repeat in range(repeats):
            ordered = (
                (("before", before_module), ("after", after_module))
                if repeat % 2 == 0
                else (("after", after_module), ("before", before_module))
            )
            execution_order.append([label for label, _ in ordered])
            for label, module in ordered:
                elapsed = _timed_fit(module, matrix, specification["solver"])
                if label == "before":
                    before_samples.append(elapsed)
                else:
                    after_samples.append(elapsed)

        before_median = float(statistics.median(before_samples))
        after_median = float(statistics.median(after_samples))
        reduction = float(100.0 * (before_median - after_median) / before_median)
        speedup = float(before_median / after_median)
        case_records.append(
            {
                **specification,
                "matrix_identity": _array_identity(matrix),
                "runtime": {
                    "before": {
                        "samples_seconds": before_samples,
                        "median_seconds": before_median,
                    },
                    "after": {
                        "samples_seconds": after_samples,
                        "median_seconds": after_median,
                    },
                    "median_reduction_percent": reduction,
                    "median_speedup_ratio": speedup,
                    "execution_order_by_repeat": execution_order,
                },
                "equivalence": equivalence,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "randomized_vp_spca_paired_runtime_optimization_evidence",
        "scope": (
            "local paired before/after end-to-end single-fit runtime and result "
            "equivalence; not the formal server quality evaluation"
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable_policy": "basename_only",
            "executable": Path(sys.executable).name,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_controls": controls,
        },
        "benchmark_protocol": {
            "clock": "time.perf_counter",
            "unit": "seconds",
            "warmups_per_source_per_case": warmups,
            "untimed_equivalence_runs_per_source_per_case": 1,
            "repeats_per_source_per_case": repeats,
            "garbage_collection_disabled_during_each_timed_call": True,
            "paired_order": "alternating before/after then after/before",
            "snapshot_import_time_included": False,
            "matrix_generation_time_included": False,
            "fit_scope": (
                "full fit_randomized_vp_spca call including range finder, "
                "optimization, requested refinement, and full-data diagnostics"
            ),
        },
        "execution_sources": source_records,
        "evidence_tool_source": _source_record(Path(__file__)),
        "formal_release_binding": formal_binding,
        "equivalence_contract": {
            "numeric_fields": list(NUMERIC_RESULT_FIELDS),
            "discrete_fields": list(DISCRETE_RESULT_FIELDS),
            "max_abs_tolerances": MAX_ABS_TOLERANCES,
            "array_hash_contract": (
                "sha256(compact sorted dtype-and-shape JSON + NUL + "
                "little-endian C-order bytes)"
            ),
            "internal_timing_fields_excluded": True,
        },
        "cases": case_records,
        "canonical_payload_sha256_contract": (
            "sha256 of compact sorted UTF-8 JSON after removing only the "
            "canonical_payload_sha256 field"
        ),
    }
    payload["canonical_payload_sha256"] = _canonical_payload_sha256(payload)
    _assert_finite_json(_strict_json_native(payload))
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _assert_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, location: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceValidationError(
            f"unexpected keys at {location}; missing={missing}, extra={extra}"
        )


def _validate_evidence_package_manifest(evidence_path: Path) -> str | None:
    """Validate the optional self-excluding manifest around a copied package."""

    package_root = evidence_path.parent
    manifest_path = package_root / EVIDENCE_PACKAGE_MANIFEST_NAME
    is_committed_evidence = evidence_path.resolve() == DEFAULT_EVIDENCE.resolve()
    if not manifest_path.is_file():
        if is_committed_evidence:
            raise EvidenceValidationError(
                f"missing committed evidence package manifest: {manifest_path}"
            )
        return None
    if package_root.is_symlink() or manifest_path.is_symlink():
        raise EvidenceValidationError(
            "evidence package and manifest cannot be symlinks"
        )

    manifest = _strict_json_load(manifest_path)
    _assert_exact_keys(
        manifest,
        {"schema_version", "artifact", "files"},
        location="evidence package manifest",
    )
    _require(manifest["schema_version"] == 1, "unsupported package manifest schema")
    _require(
        manifest["artifact"] == "randomized_vp_spca_runtime_evidence_package",
        "unexpected evidence package manifest artifact",
    )
    entries = manifest["files"]
    _require(isinstance(entries, list), "package manifest files must be a list")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    for index, entry in enumerate(entries):
        location = f"evidence package manifest.files[{index}]"
        _require(isinstance(entry, dict), f"{location} must be an object")
        _assert_exact_keys(
            entry,
            {"path", "sha256", "bytes", "role"},
            location=location,
        )
        relative = entry["path"]
        _require(
            isinstance(relative, str)
            and relative
            and "\\" not in relative
            and not relative.startswith("/")
            and all(part not in {"", ".", ".."} for part in relative.split("/")),
            f"invalid package-relative path at {location}",
        )
        _require(relative not in seen_paths, f"duplicate package path: {relative}")
        seen_paths.add(relative)
        digest = entry["sha256"]
        _require(
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            f"invalid package SHA-256 at {location}",
        )
        _require(digest not in seen_hashes, f"duplicate package SHA-256: {digest}")
        seen_hashes.add(digest)
        byte_count = entry["bytes"]
        _require(
            isinstance(byte_count, int)
            and not isinstance(byte_count, bool)
            and byte_count >= 0,
            f"invalid package byte count at {location}",
        )
        expected_role = (
            "evidence" if relative == "evidence.json" else "execution_source"
        )
        _require(entry["role"] == expected_role, f"invalid package role for {relative}")

        payload_path = package_root.joinpath(*relative.split("/"))
        current = package_root
        for part in relative.split("/"):
            current = current / part
            _require(
                not current.is_symlink(), f"package symlink is forbidden: {relative}"
            )
        _require(payload_path.is_file(), f"missing package payload: {relative}")
        _require(
            payload_path.stat().st_size == byte_count,
            f"package byte-count mismatch for {relative}",
        )
        _require(
            _sha256_file(payload_path) == digest,
            f"package SHA-256 mismatch for {relative}",
        )

    _require(
        seen_paths == set(EXPECTED_EVIDENCE_PACKAGE_PATHS),
        "evidence package manifest path set mismatch",
    )
    actual_paths: set[str] = set()
    for candidate in package_root.rglob("*"):
        relative = candidate.relative_to(package_root).as_posix()
        _require(
            not candidate.is_symlink(), f"package symlink is forbidden: {relative}"
        )
        if candidate.is_file() and candidate != manifest_path:
            actual_paths.add(relative)
    _require(
        actual_paths == seen_paths,
        "evidence package contains undeclared or missing files",
    )
    return _sha256_file(manifest_path)


def _validate_runtime(case: Mapping[str, Any]) -> None:
    runtime = case["runtime"]
    _require(isinstance(runtime, dict), "case runtime must be an object")
    protocol_repeats = None
    for label in ("before", "after"):
        record = runtime.get(label)
        _require(isinstance(record, dict), f"runtime.{label} must be an object")
        samples = record.get("samples_seconds")
        _require(
            isinstance(samples, list) and len(samples) > 0,
            f"runtime.{label} samples must be non-empty",
        )
        _require(
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in samples
            ),
            f"runtime.{label} samples must be positive finite numbers",
        )
        computed_median = float(statistics.median(float(value) for value in samples))
        _require(
            math.isclose(
                float(record.get("median_seconds")),
                computed_median,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            f"runtime.{label} median does not match samples",
        )
        if protocol_repeats is None:
            protocol_repeats = len(samples)
        _require(len(samples) == protocol_repeats, "before/after sample count mismatch")

    before_median = float(runtime["before"]["median_seconds"])
    after_median = float(runtime["after"]["median_seconds"])
    reduction = 100.0 * (before_median - after_median) / before_median
    speedup = before_median / after_median
    _require(
        math.isclose(
            float(runtime.get("median_reduction_percent")),
            reduction,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "median reduction does not match medians",
    )
    _require(
        math.isclose(
            float(runtime.get("median_speedup_ratio")),
            speedup,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "median speedup does not match medians",
    )
    orders = runtime.get("execution_order_by_repeat")
    expected_orders = [
        ["before", "after"] if index % 2 == 0 else ["after", "before"]
        for index in range(int(protocol_repeats))
    ]
    _require(orders == expected_orders, "paired execution order contract mismatch")


def _validate_equivalence_structure(case: Mapping[str, Any]) -> None:
    equivalence = case["equivalence"]
    _require(isinstance(equivalence, dict), "equivalence must be an object")
    numeric = equivalence.get("numeric_fields")
    discrete = equivalence.get("discrete_fields")
    _require(isinstance(numeric, dict), "numeric equivalence must be an object")
    _require(isinstance(discrete, dict), "discrete equivalence must be an object")
    _require(
        set(numeric) == set(NUMERIC_RESULT_FIELDS),
        "numeric equivalence fields mismatch",
    )
    _require(
        set(discrete) == set(DISCRETE_RESULT_FIELDS),
        "discrete equivalence fields mismatch",
    )
    for field, record in numeric.items():
        _require(isinstance(record, dict), f"numeric {field} record must be an object")
        tolerance = float(record.get("max_abs_tolerance"))
        difference = float(record.get("max_abs_difference"))
        _require(
            tolerance == float(MAX_ABS_TOLERANCES[field]),
            f"numeric tolerance mismatch for {field}",
        )
        _require(
            bool(record.get("within_tolerance")) == (difference <= tolerance),
            f"numeric pass flag mismatch for {field}",
        )
        _require(
            record.get("within_tolerance") is True,
            f"numeric equivalence failed for {field}",
        )
        for label in ("before", "after"):
            identity = record.get(label)
            _require(
                isinstance(identity, dict), f"numeric {field}.{label} must be an object"
            )
            for hash_name in ("content_sha256", "array_sha256"):
                digest = identity.get(hash_name)
                _require(
                    isinstance(digest, str)
                    and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest),
                    f"invalid {field}.{label}.{hash_name}",
                )
    for field, record in discrete.items():
        _require(isinstance(record, dict), f"discrete {field} must be an object")
        equality = record.get("before") == record.get("after")
        _require(record.get("equal") is equality, f"discrete flag mismatch for {field}")
        _require(equality, f"discrete equivalence failed for {field}")
    _require(
        equivalence.get("all_numeric_within_tolerance") is True,
        "aggregate numeric equivalence flag must be true",
    )
    _require(
        equivalence.get("all_discrete_equal") is True,
        "aggregate discrete equivalence flag must be true",
    )


def _validate_recomputed_equivalence(
    stored: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    case_id: str,
) -> None:
    """Require an exact fieldwise match to the newly executed snapshots."""

    stored_numeric = stored["numeric_fields"]
    recomputed_numeric = recomputed["numeric_fields"]
    for field in NUMERIC_RESULT_FIELDS:
        for source_label in ("before", "after"):
            _require(
                recomputed_numeric[field][source_label]
                == stored_numeric[field][source_label],
                "recomputed array identity mismatch for "
                f"{case_id}.{field}.{source_label}",
            )
        for diagnostic in (
            "max_abs_difference",
            "relative_l2_difference",
            "max_abs_tolerance",
            "within_tolerance",
        ):
            _require(
                recomputed_numeric[field][diagnostic]
                == stored_numeric[field][diagnostic],
                f"recomputed {diagnostic} mismatch for {case_id}.{field}",
            )

    stored_discrete = stored["discrete_fields"]
    recomputed_discrete = recomputed["discrete_fields"]
    for field in DISCRETE_RESULT_FIELDS:
        _require(
            recomputed_discrete[field] == stored_discrete[field],
            f"recomputed discrete field mismatch for {case_id}.{field}",
        )

    for aggregate in (
        "all_numeric_within_tolerance",
        "all_discrete_equal",
        "timing_fields_intentionally_excluded",
    ):
        _require(
            recomputed[aggregate] == stored[aggregate],
            f"recomputed aggregate mismatch for {case_id}.{aggregate}",
        )
    _require(
        dict(recomputed) == dict(stored),
        f"recomputed equivalence record has an exact-schema mismatch for {case_id}",
    )


def _recompute_equivalence_records(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    before_module = _load_snapshot(
        "recompute_before",
        BEFORE_SNAPSHOT,
        randomized_sha256=BEFORE_SOURCE_SHA256,
    )
    after_module = _load_snapshot(
        "recompute_after",
        AFTER_SNAPSHOT,
        randomized_sha256=AFTER_SOURCE_SHA256,
    )
    records: list[dict[str, Any]] = []
    for case in payload["cases"]:
        matrix = _matrix(case["matrix"])
        before_result = _fit(before_module, matrix, case["solver"])
        after_result = _fit(after_module, matrix, case["solver"])
        records.append(
            {
                "case_id": case["case_id"],
                "equivalence": _equivalence_record(before_result, after_result),
            }
        )
    return records


def _parse_strict_json_text(text: str, *, label: str) -> Any:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, EvidenceValidationError) as error:
        raise EvidenceValidationError(
            f"invalid strict JSON from {label}: {error}"
        ) from error
    _assert_finite_json(value)
    return value


def _recompute_in_recorded_environment(evidence_path: Path) -> list[dict[str, Any]]:
    """Replay with recorded thread limits set before the child imports NumPy."""

    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_CONTROL_NAMES})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "_recompute",
            "--evidence",
            str(evidence_path.resolve()),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise EvidenceValidationError(
            "isolated output recomputation failed: " + diagnostic[-2000:]
        )
    records = _parse_strict_json_text(
        completed.stdout, label="isolated output recomputation"
    )
    _require(isinstance(records, list), "isolated recomputation must return a list")
    return records


def _validate_array_identity(
    value: Any, expected: Mapping[str, Any], label: str
) -> None:
    actual = _array_identity(value)
    _require(actual == expected, f"array identity mismatch for {label}")


def validate_evidence(
    evidence_path: Path = DEFAULT_EVIDENCE, *, recompute_outputs: bool = False
) -> dict[str, Any]:
    outer_manifest_sha256 = _validate_evidence_package_manifest(evidence_path)
    payload = _strict_json_load(evidence_path)
    expected_top_keys = {
        "schema_version",
        "artifact",
        "scope",
        "generated_at_utc",
        "environment",
        "benchmark_protocol",
        "execution_sources",
        "evidence_tool_source",
        "formal_release_binding",
        "equivalence_contract",
        "cases",
        "canonical_payload_sha256_contract",
        "canonical_payload_sha256",
    }
    _assert_exact_keys(payload, expected_top_keys, location="root")
    _require(payload["schema_version"] == 1, "unsupported evidence schema")
    _require(
        payload["artifact"]
        == "randomized_vp_spca_paired_runtime_optimization_evidence",
        "unexpected evidence artifact",
    )
    expected_payload_hash = _canonical_payload_sha256(payload)
    _require(
        payload["canonical_payload_sha256"] == expected_payload_hash,
        "canonical payload SHA-256 mismatch",
    )

    protocol = payload["benchmark_protocol"]
    _require(isinstance(protocol, dict), "benchmark protocol must be an object")
    repeats = protocol.get("repeats_per_source_per_case")
    _require(
        isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0,
        "benchmark repeats must be a positive integer",
    )
    controls = payload["environment"].get("thread_controls")
    _require(
        controls == {name: "1" for name in THREAD_CONTROL_NAMES},
        "thread controls must all be recorded as one",
    )

    source_records = payload["execution_sources"]
    _require(
        isinstance(source_records, dict)
        and set(source_records) == {"before", "after", "vp_dependency"},
        "execution source roles mismatch",
    )
    for label, record in source_records.items():
        path = REPOSITORY_ROOT / record["path"]
        _require(path.is_file(), f"missing execution source for {label}")
        _require(
            _sha256_file(path) == record["sha256"],
            f"execution source SHA-256 mismatch for {label}",
        )
        _require(
            path.stat().st_size == record["bytes"], f"source size mismatch: {label}"
        )
    fixed_source_hashes = {
        "before": BEFORE_SOURCE_SHA256,
        "after": AFTER_SOURCE_SHA256,
        "vp_dependency": VP_SOURCE_SHA256,
    }
    for label, expected_sha256 in fixed_source_hashes.items():
        _require(
            source_records[label]["sha256"] == expected_sha256,
            f"fixed execution source SHA-256 mismatch for {label}",
        )

    tool_record = payload["evidence_tool_source"]
    tool_path = REPOSITORY_ROOT / tool_record["path"]
    _require(tool_path.is_file(), "missing evidence generator/validator source")
    _require(
        _sha256_file(tool_path) == tool_record["sha256"],
        "evidence tool source SHA-256 mismatch",
    )

    binding = payload["formal_release_binding"]
    _require(
        binding.get("randomized_execution_snapshot_recovered") is True,
        "snapshot recovery flag missing",
    )
    _require(
        binding.get("full_release_source_closure_claimed") is False,
        "evidence must not claim full release closure",
    )
    _require(
        binding.get("before_snapshot_is_origin_main") is False,
        "before snapshot must not be labelled origin/main",
    )
    manifest_record = binding["source_manifest"]
    manifest_path = REPOSITORY_ROOT / manifest_record["path"]
    _require(manifest_path.is_file(), "formal source manifest missing")
    _require(
        _sha256_file(manifest_path) == manifest_record["sha256"],
        "formal source manifest SHA-256 mismatch",
    )
    manifest = _strict_json_load(manifest_path)
    manifest_entries = {
        record["path"]: record["sha256"]
        for record in manifest["execution_files"]
        if record["path"] in {"src/rspca/randomized.py", "src/rspca/vp.py"}
    }
    recorded_entries = {
        path: record["sha256"] for path, record in binding["manifest_entries"].items()
    }
    _require(
        manifest_entries == recorded_entries, "formal manifest entry binding mismatch"
    )
    _require(
        source_records["before"]["sha256"]
        == manifest_entries["src/rspca/randomized.py"],
        "before snapshot is not the formal randomized.py execution source",
    )
    _require(
        source_records["vp_dependency"]["sha256"]
        == manifest_entries["src/rspca/vp.py"],
        "VP dependency is not the formal vp.py execution source",
    )

    cases = payload["cases"]
    _require(isinstance(cases, list), "cases must be a list")
    _require(len(cases) == len(CASE_SPECS), "case count mismatch")
    expected_specs = {
        specification["case_id"]: specification for specification in CASE_SPECS
    }
    _require(
        len({case.get("case_id") for case in cases}) == len(cases),
        "case IDs must be unique",
    )
    for case in cases:
        case_id = case.get("case_id")
        _require(case_id in expected_specs, f"unexpected case ID: {case_id}")
        specification = expected_specs[str(case_id)]
        for key in ("description", "matrix", "solver"):
            _require(
                case.get(key) == specification[key],
                f"case config mismatch: {case_id}.{key}",
            )
        matrix = _matrix(case["matrix"])
        _validate_array_identity(matrix, case["matrix_identity"], f"{case_id}.matrix")
        _validate_runtime(case)
        _require(
            len(case["runtime"]["before"]["samples_seconds"]) == repeats,
            f"sample count differs from protocol for {case_id}",
        )
        _validate_equivalence_structure(case)

    if recompute_outputs:
        recomputed_records = _recompute_in_recorded_environment(evidence_path)
        _require(
            len(recomputed_records) == len(cases),
            "isolated recomputation case count mismatch",
        )
        recomputed_by_case: dict[str, Mapping[str, Any]] = {}
        for record in recomputed_records:
            _require(
                isinstance(record, dict)
                and set(record) == {"case_id", "equivalence"}
                and isinstance(record["case_id"], str)
                and isinstance(record["equivalence"], dict),
                "invalid isolated recomputation record",
            )
            _require(
                record["case_id"] not in recomputed_by_case,
                "duplicate isolated recomputation case ID",
            )
            recomputed_by_case[record["case_id"]] = record["equivalence"]
        _require(
            set(recomputed_by_case) == {str(case["case_id"]) for case in cases},
            "isolated recomputation case IDs mismatch",
        )
        for case in cases:
            recomputed = recomputed_by_case[str(case["case_id"])]
            _require(
                recomputed["all_numeric_within_tolerance"] is True,
                f"recomputed numeric equivalence failed for {case['case_id']}",
            )
            _require(
                recomputed["all_discrete_equal"] is True,
                f"recomputed discrete equivalence failed for {case['case_id']}",
            )
            _validate_recomputed_equivalence(
                case["equivalence"],
                recomputed,
                case_id=str(case["case_id"]),
            )

    return {
        "valid": True,
        "evidence_path": (
            _repository_relative(evidence_path)
            if evidence_path.is_relative_to(REPOSITORY_ROOT)
            else evidence_path.name
        ),
        "canonical_payload_sha256": payload["canonical_payload_sha256"],
        "case_count": len(cases),
        "outputs_recomputed": recompute_outputs,
        "outer_manifest_sha256": outer_manifest_sha256,
        "median_reduction_percent": {
            case["case_id"]: case["runtime"]["median_reduction_percent"]
            for case in cases
        },
    }


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate", help="run paired A/B evidence")
    generate.add_argument("--output", type=Path, default=DEFAULT_EVIDENCE)
    generate.add_argument("--warmups", type=_nonnegative_integer, default=1)
    generate.add_argument("--repeats", type=_positive_integer, default=7)
    validate = subparsers.add_parser("validate", help="strictly validate evidence")
    validate.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    validate.add_argument(
        "--recompute-outputs",
        action="store_true",
        help="execute both snapshots and recheck numerical equivalence",
    )
    recompute = subparsers.add_parser("_recompute", help=argparse.SUPPRESS)
    recompute.add_argument("--evidence", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "_recompute":
        payload = _strict_json_load(arguments.evidence)
        records = _recompute_equivalence_records(payload)
        print(
            json.dumps(
                _strict_json_native(records),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
        return 0
    if arguments.command == "generate":
        if os.path.lexists(arguments.output):
            raise FileExistsError(
                f"refusing to overwrite existing evidence: {arguments.output}"
            )
        payload = generate_evidence(
            warmups=arguments.warmups, repeats=arguments.repeats
        )
        _write_json_exclusive(arguments.output, payload)
        print(
            json.dumps(
                {
                    "output": str(arguments.output),
                    "canonical_payload_sha256": payload["canonical_payload_sha256"],
                    "case_reductions_percent": {
                        case["case_id"]: case["runtime"]["median_reduction_percent"]
                        for case in payload["cases"]
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    report = validate_evidence(
        arguments.evidence, recompute_outputs=arguments.recompute_outputs
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

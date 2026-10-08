#!/usr/bin/env python3
"""Locked scale-aware VP-SPCA confirmation for the five-population cohort.

The formal profile is deliberately difficult to execute accidentally.  It
requires a reviewed shared-source anchor, an input/freshness-bound protocol
lock, an external SHA-256 anchor for that lock, an exclusive output path, and
an explicit execution flag.  The cheap smoke profile exercises the same
serialization and validation path but is not scientific evidence.

This module does not change the VP-SPCA objective.  It parameterizes its
existing L1 coefficient by the reference proximal threshold
``rho = alpha / sigma_max(X_train_panel)**2`` and reports both the historical
raw proximal-gradient mapping and a dimensionless KKT RMS diagnostic.
"""

from __future__ import annotations

# The speed contract must be installed before NumPy (and therefore any BLAS
# runtime) is imported.  Formal/development CLI execution also rejects a
# process that imported NumPy before this module, because changing these
# variables afterwards would only look pinned on paper.
import os
import sys

THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
NUMPY_PREIMPORTED = "numpy" in sys.modules
for _thread_name, _thread_value in THREAD_ENVIRONMENT.items():
    os.environ[_thread_name] = _thread_value


def _pin_pre_numpy_cpu_affinity() -> dict[str, Any]:
    """Pin one logical CPU before NumPy import when the OS supports it."""

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return {
            "policy": "single_logical_cpu_before_numpy_import",
            "status": "unsupported_by_os",
            "available_before": None,
            "pinned_after": None,
        }
    before = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    if not before:
        raise RuntimeError("CPU affinity mask is empty before NumPy import")
    selected = before[0]
    os.sched_setaffinity(0, {selected})
    after = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    if after != [selected]:
        raise RuntimeError("failed to pin one CPU before NumPy import")
    return {
        "policy": "single_logical_cpu_before_numpy_import",
        "status": "pinned",
        "available_before": before,
        "pinned_after": after,
    }


# ``Any`` is only needed by the annotation above and importing typing is safe
# before NumPy/BLAS initialization.
from typing import Any, Iterable, Mapping, Sequence


PRE_NUMPY_CPU_AFFINITY = _pin_pre_numpy_cpu_affinity()

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from time import perf_counter
import zipfile

import numpy as np
from numpy.typing import NDArray

if __package__:
    from . import genomics_pca_loading_grid as grid
else:
    import genomics_pca_loading_grid as grid  # type: ignore[no-redef]

from rspca.datasets import array_sha256, runtime_versions
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    stratified_split,
    transform_smartpca,
)
from rspca.randomized import (
    fit_randomized_vp_spca_from_workspace,
    prepare_randomized_vp_spca_workspace,
)
from rspca.vp import (
    fit_vp_spca,
    proximal_gradient_mapping,
    procrustes_factor,
    vp_objective,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

SCHEMA_VERSION = "genomics-scale-aware-vp-confirmation-v2"
LOCK_SCHEMA_VERSION = "genomics-scale-aware-vp-lock-v2"
MANIFEST_SCHEMA_VERSION = "genomics-scale-aware-vp-manifest-v2"
DEVELOPMENT_BUNDLE_SCHEMA_VERSION = "genomics-scale-aware-vp-development-bundle-v1"
EXECUTION_SPEND_SCHEMA_VERSION = "genomics-scale-aware-vp-execution-spend-v1"

FORMAL_SEEDS = tuple(range(4001, 4021))
DEVELOPMENT_SEEDS = tuple(range(2025, 2030))
RANK = 4
PANEL_SIZE = 400
RHO = 0.0025
TOTAL_BUDGET = 3000
RANDOMIZED_SKETCH_BUDGET = 1500
RANDOMIZED_FULL_BUDGET = 1500
RANDOMIZED_OVERSAMPLE = 30
RANDOMIZED_POWER_ITERATIONS = 4
RAW_TOL_FLOOR = 1e-300
SUPPORT_THRESHOLD = 1e-8
NORMALIZED_KKT_TOLERANCE = 1e-3
NATIVE_LAST_WINDOW = 20
MIN_COMPONENT_SUPPORT = 10
MAX_ENTRY_SUPPORT_FRACTION = 0.30
MIN_MEAN_ROW_SUPPORT_FRACTION = 0.75
MAX_MEAN_ROW_SUPPORT_FRACTION = 0.90
MIN_SPLIT_ROW_SUPPORT_FRACTION = 0.70
MAX_SPLIT_ROW_SUPPORT_FRACTION = 0.95
MEAN_BA_NONINFERIORITY_MARGIN = 0.005
WORST_BA_NONINFERIORITY_MARGIN = 0.03
RANDOMIZED_OBJECTIVE_EXCESS_MAX = 1e-4
RANDOMIZED_ETA_FACTOR_MAX = 1.05
RANDOMIZED_SPEEDUP_MIN = 1.05
RANDOMIZED_FASTER_FRACTION_MIN = 0.80
SPEED_BOOTSTRAP_SEED = 6_421_937
SPEED_BOOTSTRAP_DRAWS = 20_000
SPEED_BOOTSTRAP_CONFIDENCE = 0.95
FORMAL_WARMUP_REPETITIONS = 1
FORMAL_TIMED_REPETITIONS = 7
DEVELOPMENT_WARMUP_REPETITIONS = 0
DEVELOPMENT_TIMED_REPETITIONS = 1

SMOKE_SEEDS = (71,)
SMOKE_PANEL_SIZE = 40
SMOKE_TOTAL_BUDGET = 12
SMOKE_RANDOMIZED_SKETCH_BUDGET = 6
SMOKE_RANDOMIZED_FULL_BUDGET = 6
SMOKE_WARMUP_REPETITIONS = 1
SMOKE_TIMED_REPETITIONS = 2

ANALYSIS_FILENAME = "scale_aware_analysis.json"
PREDICTIONS_FILENAME = "scale_aware_predictions.json"
ARRAYS_FILENAME = "scale_aware_solver_arrays.npz"
LOCK_COPY_FILENAME = "protocol_lock.json"
SMOKE_PROTOCOL_FILENAME = "smoke_protocol.json"
MANIFEST_FILENAME = "scale_aware_output_manifest.json"
DEVELOPMENT_PROTOCOL_FILENAME = "development_protocol.json"
DEVELOPMENT_BUNDLE_MANIFEST_FILENAME = "development_evidence_manifest.json"
DEVELOPMENT_3000_DIRECTORY = "rho_0p0025_budget_3000"
DEVELOPMENT_4000_DIRECTORY = "rho_0p0025_budget_4000"
EXECUTION_SPEND_RECEIPT_FILENAME = "formal_execution_spend_receipt.json"
CANONICAL_FRESHNESS_RELATIVE_ROOTS = (
    "results/genomics_pca_loading_grid_2026-08-31",
    "results/genomics_pca_loading_confirmation_2026-08-31",
    "results/genomics_two_stage_solver_feasibility_2026-08-31",
)
FRESHNESS_SCOPE_SCHEMA_VERSION = "genomics-scale-aware-freshness-scope-v1"

SOURCE_FILE_ROLES = {
    "experiments/genomics_scale_aware_vp_confirmation.py": "locked_confirmation_runner",
    "experiments/genomics_pca_loading_grid.py": "train_only_pca_screen_and_metrics",
    "src/rspca/genomics_aim.py": "smartpca_preprocessing_and_stratified_split",
    "src/rspca/vp.py": "vp_solver_and_common_kkt_objective",
    "src/rspca/randomized.py": "randomized_vp_solver",
    "src/rspca/datasets.py": "array_hashing_and_runtime_identity",
    "src/rspca/__init__.py": "package_initialization_only",
}

# This list is the reviewed result of the isolated fresh-import discovery.  A
# source receipt must contain this exact set: the seven direct files are not a
# sufficient closure because importing rspca eagerly executes the other twelve.
FROZEN_DISCOVERED_SOURCE_PATHS = (
    "experiments/__init__.py",
    "experiments/genomics_pca_loading_grid.py",
    "experiments/genomics_scale_aware_vp_confirmation.py",
    "src/rspca/__init__.py",
    "src/rspca/accelerated.py",
    "src/rspca/cross_modal_gates.py",
    "src/rspca/datasets.py",
    "src/rspca/federated.py",
    "src/rspca/fedrpca_audit.py",
    "src/rspca/gauge.py",
    "src/rspca/gauge_stress.py",
    "src/rspca/genomics_aim.py",
    "src/rspca/metrics.py",
    "src/rspca/product_aggregation.py",
    "src/rspca/randomized.py",
    "src/rspca/robust.py",
    "src/rspca/statistical_gates.py",
    "src/rspca/trust_audit.py",
    "src/rspca/vp.py",
)

FROZEN_SHARED_CORE_HASHES = {
    "src/rspca/vp.py": "88a84aa99825357f3a16650bf3e56bfef54a850aa17880342334ba7ad402b953",
    "src/rspca/randomized.py": "101833335c80b178f35b696bdc28dd8a4c0eb6a9fc5a565af802d83088d0823b",
}
EXPECTED_POPULATIONS = grid.EXPECTED_POPULATIONS
METHODS = ("pca", "vp", "randomized")
VIEWS = ("native", "common_pca_probe")


@dataclass(frozen=True)
class RunConfig:
    """One immutable execution profile."""

    profile: str
    seeds: tuple[int, ...]
    rank: int
    panel_size: int
    rho: float
    total_budget: int
    randomized_sketch_budget: int
    randomized_full_budget: int
    randomized_oversample: int
    randomized_power_iterations: int
    support_threshold: float
    normalized_kkt_tolerance: float
    native_last_window: int
    minimum_component_support: int
    maximum_entry_support_fraction: float
    minimum_mean_row_support_fraction: float
    maximum_mean_row_support_fraction: float
    minimum_split_row_support_fraction: float
    maximum_split_row_support_fraction: float
    warmup_repetitions: int
    timed_repetitions: int
    test_fraction: float = 0.2


def formal_config() -> RunConfig:
    return RunConfig(
        profile="formal",
        seeds=FORMAL_SEEDS,
        rank=RANK,
        panel_size=PANEL_SIZE,
        rho=RHO,
        total_budget=TOTAL_BUDGET,
        randomized_sketch_budget=RANDOMIZED_SKETCH_BUDGET,
        randomized_full_budget=RANDOMIZED_FULL_BUDGET,
        randomized_oversample=RANDOMIZED_OVERSAMPLE,
        randomized_power_iterations=RANDOMIZED_POWER_ITERATIONS,
        support_threshold=SUPPORT_THRESHOLD,
        normalized_kkt_tolerance=NORMALIZED_KKT_TOLERANCE,
        native_last_window=NATIVE_LAST_WINDOW,
        minimum_component_support=MIN_COMPONENT_SUPPORT,
        maximum_entry_support_fraction=MAX_ENTRY_SUPPORT_FRACTION,
        minimum_mean_row_support_fraction=MIN_MEAN_ROW_SUPPORT_FRACTION,
        maximum_mean_row_support_fraction=MAX_MEAN_ROW_SUPPORT_FRACTION,
        minimum_split_row_support_fraction=MIN_SPLIT_ROW_SUPPORT_FRACTION,
        maximum_split_row_support_fraction=MAX_SPLIT_ROW_SUPPORT_FRACTION,
        warmup_repetitions=FORMAL_WARMUP_REPETITIONS,
        timed_repetitions=FORMAL_TIMED_REPETITIONS,
    )


def development_config(budget: int = TOTAL_BUDGET) -> RunConfig:
    if budget not in (TOTAL_BUDGET, 4000):
        raise ValueError("development budget must be exactly 3000 or 4000")
    half = budget // 2
    return RunConfig(
        profile=f"development_{budget}",
        seeds=DEVELOPMENT_SEEDS,
        rank=RANK,
        panel_size=PANEL_SIZE,
        rho=RHO,
        total_budget=budget,
        randomized_sketch_budget=half,
        randomized_full_budget=budget - half,
        randomized_oversample=RANDOMIZED_OVERSAMPLE,
        randomized_power_iterations=RANDOMIZED_POWER_ITERATIONS,
        support_threshold=SUPPORT_THRESHOLD,
        normalized_kkt_tolerance=NORMALIZED_KKT_TOLERANCE,
        native_last_window=NATIVE_LAST_WINDOW,
        minimum_component_support=MIN_COMPONENT_SUPPORT,
        maximum_entry_support_fraction=MAX_ENTRY_SUPPORT_FRACTION,
        minimum_mean_row_support_fraction=MIN_MEAN_ROW_SUPPORT_FRACTION,
        maximum_mean_row_support_fraction=MAX_MEAN_ROW_SUPPORT_FRACTION,
        minimum_split_row_support_fraction=MIN_SPLIT_ROW_SUPPORT_FRACTION,
        maximum_split_row_support_fraction=MAX_SPLIT_ROW_SUPPORT_FRACTION,
        warmup_repetitions=DEVELOPMENT_WARMUP_REPETITIONS,
        timed_repetitions=DEVELOPMENT_TIMED_REPETITIONS,
    )


def smoke_config() -> RunConfig:
    return RunConfig(
        profile="smoke",
        seeds=SMOKE_SEEDS,
        rank=RANK,
        panel_size=SMOKE_PANEL_SIZE,
        rho=RHO,
        total_budget=SMOKE_TOTAL_BUDGET,
        randomized_sketch_budget=SMOKE_RANDOMIZED_SKETCH_BUDGET,
        randomized_full_budget=SMOKE_RANDOMIZED_FULL_BUDGET,
        randomized_oversample=8,
        randomized_power_iterations=1,
        support_threshold=SUPPORT_THRESHOLD,
        normalized_kkt_tolerance=NORMALIZED_KKT_TOLERANCE,
        native_last_window=min(NATIVE_LAST_WINDOW, SMOKE_RANDOMIZED_FULL_BUDGET),
        minimum_component_support=1,
        maximum_entry_support_fraction=MAX_ENTRY_SUPPORT_FRACTION,
        minimum_mean_row_support_fraction=0.0,
        maximum_mean_row_support_fraction=1.0,
        minimum_split_row_support_fraction=0.0,
        maximum_split_row_support_fraction=1.0,
        warmup_repetitions=SMOKE_WARMUP_REPETITIONS,
        timed_repetitions=SMOKE_TIMED_REPETITIONS,
    )


def config_for_profile(profile: str) -> RunConfig:
    if profile == "formal":
        return formal_config()
    if profile == "development_3000":
        return development_config(3000)
    if profile == "development_4000":
        return development_config(4000)
    if profile == "smoke":
        return smoke_config()
    raise ValueError("unknown execution profile")


def _protocol(config: RunConfig) -> dict[str, Any]:
    return {
        "profile": config.profile,
        "scientific_role": (
            "fresh_locked_confirmation"
            if config.profile == "formal"
            else (
                "post_hoc_real_development_non_evidence"
                if config.profile.startswith("development_")
                else "synthetic_non_evidence_smoke"
            )
        ),
        "development_status": (
            "external_real_3000_and_4000_packages_required_before_formal_freeze"
        ),
        "fresh_seed_schedule": list(config.seeds),
        "rank": config.rank,
        "panel_size": config.panel_size,
        "shared_screen": "outer_train_only_rank4_pca_loading_top400",
        "loading_score": "descending_sum_abs_rank4_pca_loadings",
        "rho": config.rho,
        "alpha_rule": "alpha_equals_rho_times_panel_sigma_max_squared",
        "reference_step_rule": "gamma_ref_equals_inverse_panel_sigma_max_squared",
        "total_proximal_budget": config.total_budget,
        "vp_budget": config.total_budget,
        "randomized_sketch_budget": config.randomized_sketch_budget,
        "randomized_full_budget": config.randomized_full_budget,
        "randomized_total_budget": (
            config.randomized_sketch_budget + config.randomized_full_budget
        ),
        "randomized_oversample": config.randomized_oversample,
        "randomized_power_iterations": config.randomized_power_iterations,
        "randomized_seed_rule": "outer_seed_plus_730000",
        "raw_solver_tolerance": RAW_TOL_FLOOR,
        "native_last_window": config.native_last_window,
        "normalized_kkt_definition": "raw_frobenius_mapping_divided_by_alpha_sqrt_p_times_r",
        "normalized_kkt_tolerance": config.normalized_kkt_tolerance,
        "support_threshold": config.support_threshold,
        "minimum_component_support": config.minimum_component_support,
        "maximum_entry_support_fraction": config.maximum_entry_support_fraction,
        "minimum_mean_row_support_fraction": config.minimum_mean_row_support_fraction,
        "maximum_mean_row_support_fraction": config.maximum_mean_row_support_fraction,
        "minimum_split_row_support_fraction": config.minimum_split_row_support_fraction,
        "maximum_split_row_support_fraction": config.maximum_split_row_support_fraction,
        "mean_native_ba_noninferiority_margin": MEAN_BA_NONINFERIORITY_MARGIN,
        "worst_native_ba_noninferiority_margin": WORST_BA_NONINFERIORITY_MARGIN,
        "randomized_objective_relative_excess_max": (RANDOMIZED_OBJECTIVE_EXCESS_MAX),
        "randomized_common_eta_factor_max": RANDOMIZED_ETA_FACTOR_MAX,
        "randomized_solver_speedup_min": RANDOMIZED_SPEEDUP_MIN,
        "randomized_faster_fraction_min": RANDOMIZED_FASTER_FRACTION_MIN,
        "speed_bootstrap": {
            "unit": "outer_seed_cluster_of_paired_repetitions",
            "statistic": "geometric_mean_of_per_seed_median_speedup",
            "seed": SPEED_BOOTSTRAP_SEED,
            "draws": SPEED_BOOTSTRAP_DRAWS,
            "confidence": SPEED_BOOTSTRAP_CONFIDENCE,
            "gate": "percentile_lower_bound_at_least_randomized_solver_speedup_min",
        },
        "runtime_warmup_repetitions": config.warmup_repetitions,
        "runtime_timed_repetitions": config.timed_repetitions,
        "runtime_order": "alternating_vp_randomized_by_seed_and_repetition",
        "common_probe": "closed_form_rank4_pca_on_the_same_train_only_top400_panel",
        "test_fraction": config.test_fraction,
        "population_design": "five_superpopulation_metadata_labels_not_race",
        "genome_scope": (
            "anchored_chr20_22_only"
            if config.profile == "formal"
            else "synthetic_smoke"
        ),
        "fed_robust_or_fedrpca_called": False,
    }


def _reject_nonfinite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")


def _canonical_bytes(payload: Any) -> bytes:
    _reject_nonfinite(payload)
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _strict_json_bytes(payload: Any) -> bytes:
    _reject_nonfinite(payload)
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json_load_bytes(payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _strict_json_load(path: Path) -> Any:
    return _strict_json_load_bytes(path.read_bytes())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_regular_file(path: Path, *, role: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")


def _write_new_file(path: Path, payload: bytes) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _cpu_model() -> str:
    model = ""
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file() and not cpuinfo.is_symlink():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                model = line.split(":", 1)[1].strip()
                break
    if not model:
        model = platform.processor().strip()
    return model or "unknown"


def runtime_execution_contract() -> dict[str, Any]:
    """Record the timing environment after the pre-import contract is active."""

    try:
        from threadpoolctl import threadpool_info
    except ImportError as error:  # pragma: no cover - formal images include it
        raise RuntimeError("threadpoolctl is required for timing provenance") from error
    pools = []
    for item in threadpool_info():
        pools.append(
            {
                "architecture": str(item.get("architecture") or "unknown"),
                "internal_api": str(item.get("internal_api") or "unknown"),
                "num_threads": int(item.get("num_threads") or 0),
                "prefix": str(item.get("prefix") or "unknown"),
                "user_api": str(item.get("user_api") or "unknown"),
                "version": str(item.get("version") or "unknown"),
            }
        )
    pools.sort(
        key=lambda item: (
            item["user_api"],
            item["internal_api"],
            item["prefix"],
            item["version"],
        )
    )
    current_affinity = (
        sorted(int(cpu) for cpu in os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None
    )
    load = list(os.getloadavg()) if hasattr(os, "getloadavg") else None
    return {
        "thread_environment": {
            name: os.environ.get(name) for name in sorted(THREAD_ENVIRONMENT)
        },
        "numpy_preimported_before_contract": NUMPY_PREIMPORTED,
        "pre_numpy_cpu_affinity": dict(PRE_NUMPY_CPU_AFFINITY),
        "current_cpu_affinity": current_affinity,
        "logical_cpu_count": int(os.cpu_count() or 0),
        "cpu_model": _cpu_model(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
        "load_average_1_5_15": load,
        "threadpools": pools,
    }


def validate_runtime_execution_contract(
    receipt: Mapping[str, Any], *, require_speed_qualification: bool
) -> None:
    expected_keys = {
        "thread_environment",
        "numpy_preimported_before_contract",
        "pre_numpy_cpu_affinity",
        "current_cpu_affinity",
        "logical_cpu_count",
        "cpu_model",
        "platform",
        "python_implementation",
        "load_average_1_5_15",
        "threadpools",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != expected_keys:
        raise ValueError("runtime execution contract schema mismatch")
    if receipt["thread_environment"] != {
        name: "1" for name in sorted(THREAD_ENVIRONMENT)
    }:
        raise ValueError("BLAS/OpenMP thread environment is not pinned to one")
    pools = receipt["threadpools"]
    if not isinstance(pools, list) or any(
        not isinstance(item, Mapping)
        or set(item)
        != {
            "architecture",
            "internal_api",
            "num_threads",
            "prefix",
            "user_api",
            "version",
        }
        or isinstance(item["num_threads"], bool)
        or not isinstance(item["num_threads"], int)
        or item["num_threads"] < 1
        for item in pools
    ):
        raise ValueError("loaded numerical threadpool receipt is invalid")
    load = receipt["load_average_1_5_15"]
    if load is not None and (
        not isinstance(load, list)
        or len(load) != 3
        or any(not math.isfinite(float(value)) or float(value) < 0 for value in load)
    ):
        raise ValueError("load-average receipt is invalid")
    affinity = receipt["pre_numpy_cpu_affinity"]
    if not isinstance(affinity, Mapping) or set(affinity) != {
        "policy",
        "status",
        "available_before",
        "pinned_after",
    }:
        raise ValueError("CPU-affinity receipt schema mismatch")
    if require_speed_qualification:
        if (
            receipt["numpy_preimported_before_contract"] is not False
            or not pools
            or any(item["num_threads"] != 1 for item in pools)
            or affinity["status"] != "pinned"
            or not isinstance(affinity["pinned_after"], list)
            or len(affinity["pinned_after"]) != 1
            or receipt["current_cpu_affinity"] != affinity["pinned_after"]
        ):
            raise ValueError(
                "formal/development speed qualification requires pre-NumPy CPU pinning"
            )


def _deterministic_npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return an NPZ whose entry order and ZIP metadata are deterministic."""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            if not re.fullmatch(r"[a-z0-9_]+", name):
                raise ValueError(f"invalid deterministic NPZ key: {name}")
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asarray(arrays[name]),
                allow_pickle=False,
            )
            information = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            information.compress_type = zipfile.ZIP_STORED
            information.external_attr = 0o600 << 16
            archive.writestr(information, array_buffer.getvalue())
    return buffer.getvalue()


def _load_npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
        information = archive.infolist()
        names = [item.filename for item in information]
        if (
            not names
            or len(names) != len(set(names))
            or any(not re.fullmatch(r"[a-z0-9_]+\.npy", name) for name in names)
            or any(item.compress_type != zipfile.ZIP_STORED for item in information)
        ):
            raise ValueError("solver NPZ ZIP closure/encoding mismatch")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _fresh_import_repo_module_paths(repo_root: Path) -> list[str]:
    """Discover the real repo-local import closure in an isolated interpreter."""

    root = repo_root.resolve()
    discovery = r"""
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
importlib.import_module("experiments.genomics_scale_aware_vp_confirmation")
paths = set()
for module in tuple(sys.modules.values()):
    candidate = getattr(module, "__file__", None)
    if not candidate:
        continue
    path = Path(candidate).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        continue
    if path.is_file():
        paths.add(relative.as_posix())
print(json.dumps(sorted(paths), separators=(",", ":")))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-c", discovery, str(root)],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh import closure discovery failed: " + completed.stderr.strip()[:2000]
        )
    try:
        paths = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("fresh import closure emitted invalid JSON") from error
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, str) for path in paths)
        or paths != sorted(set(paths))
    ):
        raise RuntimeError("fresh import closure is empty, duplicated, or unordered")
    return paths


def collect_source_identity(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    imported_paths = _fresh_import_repo_module_paths(root)
    if tuple(imported_paths) != FROZEN_DISCOVERED_SOURCE_PATHS:
        raise ValueError(
            "fresh import closure differs from reviewed exact set; "
            f"missing={sorted(set(FROZEN_DISCOVERED_SOURCE_PATHS) - set(imported_paths))}, "
            f"extra={sorted(set(imported_paths) - set(FROZEN_DISCOVERED_SOURCE_PATHS))}"
        )
    files: dict[str, dict[str, Any]] = {}
    for relative in imported_paths:
        path = root / relative
        role = SOURCE_FILE_ROLES.get(relative, "runtime_import_closure")
        _require_regular_file(path, role=role)
        files[relative] = {
            "role": role,
            "sha256": _file_sha256(path),
            "bytes": int(path.stat().st_size),
        }
    for relative, expected_sha256 in FROZEN_SHARED_CORE_HASHES.items():
        if files[relative]["sha256"] != expected_sha256:
            raise ValueError(
                f"shared core changed after freeze: {relative}; "
                f"expected {expected_sha256}, observed {files[relative]['sha256']}"
            )
    unsigned = {
        "discovery": "isolated_fresh_import_sys_modules_repo_file_closure",
        "exact_discovered_paths": list(imported_paths),
        "files": files,
    }
    return {**unsigned, "aggregate_sha256": canonical_sha256(unsigned)}


def validate_source_identity(identity: Mapping[str, Any]) -> None:
    if set(identity) != {
        "discovery",
        "exact_discovered_paths",
        "files",
        "aggregate_sha256",
    }:
        raise ValueError("source identity schema mismatch")
    if identity["discovery"] != "isolated_fresh_import_sys_modules_repo_file_closure":
        raise ValueError("source identity was not collected by a fresh import")
    files = identity["files"]
    if (
        not isinstance(files, Mapping)
        or tuple(identity["exact_discovered_paths"]) != FROZEN_DISCOVERED_SOURCE_PATHS
        or tuple(files) != FROZEN_DISCOVERED_SOURCE_PATHS
    ):
        raise ValueError("source identity does not bind the exact discovered set")
    for relative, receipt in files.items():
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(receipt, Mapping)
            or set(receipt) != {"role", "sha256", "bytes"}
            or receipt["role"]
            not in set(SOURCE_FILE_ROLES.values()) | {"runtime_import_closure"}
            or not _is_sha256(receipt["sha256"])
            or isinstance(receipt["bytes"], bool)
            or not isinstance(receipt["bytes"], int)
            or receipt["bytes"] < 1
        ):
            raise ValueError(f"invalid source identity receipt: {relative!r}")
    unsigned = {
        "discovery": identity["discovery"],
        "exact_discovered_paths": identity["exact_discovered_paths"],
        "files": files,
    }
    if (
        not _is_sha256(identity["aggregate_sha256"])
        or canonical_sha256(unsigned) != identity["aggregate_sha256"]
    ):
        raise ValueError("source identity aggregate digest mismatch")


def _input_identity(data: grid.GenotypeInput) -> dict[str, Any]:
    return {
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "orientation": "samples_by_snps",
        "genotypes_sha256": array_sha256(np.asarray(data.genotypes, dtype=float)),
        "sample_ids_sha256": array_sha256(np.asarray(data.sample_ids, dtype=str)),
        "snp_ids_sha256": array_sha256(np.asarray(data.snp_ids, dtype=str)),
        "chromosomes_sha256": array_sha256(np.asarray(data.chromosomes, dtype=str)),
        "population_labels_sha256": array_sha256(
            np.asarray(data.population_labels, dtype=str)
        ),
    }


_FORMAL_INPUT_IDENTITY_KEYS = {
    "samples",
    "snps",
    "orientation",
    "genotypes_sha256",
    "sample_ids_sha256",
    "snp_ids_sha256",
    "chromosomes_sha256",
    "population_labels_sha256",
    "external_source_manifest_sha256",
    "expected_external_source_manifest_sha256",
    "external_source_manifest_anchor_verified",
    "external_source_manifest_bytes",
    "genotype_npz_sha256",
    "genotype_npz_bytes",
    "manifest_population_design",
    "manifest_input_scope",
    "manifest_ld_pruned_receipts_verified",
    "population_design",
    "input_scope",
    "ld_pruned",
    "population_label_is_race",
}


def validate_formal_input_identity(identity: Mapping[str, Any]) -> None:
    if set(identity) != _FORMAL_INPUT_IDENTITY_KEYS:
        raise ValueError("formal prepared-input identity schema mismatch")
    for field in (
        "genotypes_sha256",
        "sample_ids_sha256",
        "snp_ids_sha256",
        "chromosomes_sha256",
        "population_labels_sha256",
        "external_source_manifest_sha256",
        "expected_external_source_manifest_sha256",
        "genotype_npz_sha256",
    ):
        if not _is_sha256(identity[field]):
            raise ValueError(f"formal prepared-input digest invalid: {field}")
    if (
        isinstance(identity["samples"], bool)
        or not isinstance(identity["samples"], int)
        or identity["samples"] < 10
        or isinstance(identity["snps"], bool)
        or not isinstance(identity["snps"], int)
        or identity["snps"] < PANEL_SIZE
        or identity["orientation"] != "samples_by_snps"
        or identity["external_source_manifest_sha256"]
        != identity["expected_external_source_manifest_sha256"]
        or identity["external_source_manifest_anchor_verified"] is not True
        or identity["manifest_ld_pruned_receipts_verified"] is not True
        or identity["manifest_population_design"] != "five_superpopulation"
        or identity["population_design"] != "five_superpopulation"
        or identity["manifest_input_scope"] != "legacy_chr20_22"
        or identity["input_scope"] != "legacy_chr20_22"
        or identity["ld_pruned"] is not True
        or identity["population_label_is_race"] is not False
    ):
        raise ValueError("formal prepared-input policy mismatch")
    for field in ("external_source_manifest_bytes", "genotype_npz_bytes"):
        if (
            isinstance(identity[field], bool)
            or not isinstance(identity[field], int)
            or identity[field] <= 0
        ):
                raise ValueError(f"formal prepared-input size invalid: {field}")


def _formal_execution_semantics(
    prepared_input_identity: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, Any]:
    """The global spend key deliberately excludes lock/output file identity."""

    return {
        "seed_bank": list(FORMAL_SEEDS),
        "prepared_input_identity": dict(prepared_input_identity),
        "protocol": dict(protocol),
    }


_CAPABILITY_ISSUER_GUARD = object()


class _FormalExecutionCapability:
    """Private, one-shot permission minted only after an atomic spend receipt."""

    __slots__ = (
        "_execution_key",
        "_input_identity",
        "_protocol_sha256",
        "_receipt",
        "_spent_path",
        "_used",
    )

    def __init__(
        self,
        issuer_guard: object,
        *,
        execution_key: str,
        input_identity: Mapping[str, Any],
        protocol_sha256: str,
        receipt: Mapping[str, Any],
        spent_path: Path,
    ) -> None:
        if issuer_guard is not _CAPABILITY_ISSUER_GUARD:
            raise PermissionError("formal capabilities cannot be constructed publicly")
        self._execution_key = execution_key
        self._input_identity = dict(input_identity)
        self._protocol_sha256 = protocol_sha256
        self._receipt = dict(receipt)
        self._spent_path = spent_path.resolve()
        self._used = False

    def claim(self, data: grid.GenotypeInput, config: RunConfig) -> dict[str, Any]:
        if self._used:
            raise PermissionError("formal execution capability is single-use")
        # Mark used before any validation or fit: an exception never restores it.
        self._used = True
        if config != formal_config() or canonical_sha256(
            _protocol(config)
        ) != self._protocol_sha256:
            raise PermissionError("formal capability does not match this protocol")
        observed = _input_identity(data)
        if any(self._input_identity.get(key) != value for key, value in observed.items()):
            raise PermissionError("formal capability does not match this input")
        _require_regular_file(self._spent_path, role="global execution spend receipt")
        loaded = _strict_json_load(self._spent_path)
        if loaded != self._receipt or self._receipt[
            "execution_key_sha256"
        ] != self._execution_key:
            raise PermissionError("global execution spend receipt changed before fit")
        validate_execution_spend_receipt(loaded)
        return dict(self._receipt)


def _consume_global_formal_execution(
    *,
    registry_directory: Path,
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    output_directory: Path,
) -> _FormalExecutionCapability:
    """Atomically spend one semantic formal run before returning a capability."""

    registry = Path(registry_directory)
    if (
        not registry.is_absolute()
        or registry.is_symlink()
        or not registry.is_dir()
    ):
        raise ValueError("global formal execution registry must be an existing real directory")
    semantics = _formal_execution_semantics(
        lock["prepared_input_identity"], lock["protocol"]
    )
    execution_key = canonical_sha256(semantics)
    path = registry / f"{execution_key}.spent.json"
    if path.parent.resolve() != registry.resolve():
        raise ValueError("invalid global execution spend path")
    unsigned = {
        "schema_version": EXECUTION_SPEND_SCHEMA_VERSION,
        "status": "spent_before_any_fit_failure_remains_spent",
        "execution_key_sha256": execution_key,
        "semantic_tuple": semantics,
        "protocol_lock_file_sha256": lock_file_sha256,
        "protocol_lock_payload_sha256": lock["lock_payload_sha256"],
        "exclusive_output_directory": str(Path(output_directory).resolve()),
        "registry_directory": str(registry.resolve()),
        "spent_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_id": int(os.getpid()),
        "host": platform.node() or "unknown",
        "fit_started_when_written": False,
    }
    provisional = {
        **unsigned,
        "receipt_payload_sha256": canonical_sha256(unsigned),
        "receipt_zero_field_sha256": "0" * 64,
    }
    # The receipt contains its own final file hash.  A literal file cannot
    # self-hash, so this field binds the canonical receipt with that field
    # replaced by zeroes; validators repeat the same transform.
    final = dict(provisional)
    final["receipt_zero_field_sha256"] = _bytes_sha256(
        _strict_json_bytes(provisional)
    )
    _write_new_file(path, _strict_json_bytes(final))
    # The stable receipt hash is the zero-field canonical file hash above,
    # not the impossible recursive SHA of the final bytes.
    return _FormalExecutionCapability(
        _CAPABILITY_ISSUER_GUARD,
        execution_key=execution_key,
        input_identity=lock["prepared_input_identity"],
        protocol_sha256=canonical_sha256(lock["protocol"]),
        receipt=final,
        spent_path=path,
    )


def validate_execution_spend_receipt(receipt: Mapping[str, Any]) -> None:
    keys = {
        "schema_version",
        "status",
        "execution_key_sha256",
        "semantic_tuple",
        "protocol_lock_file_sha256",
        "protocol_lock_payload_sha256",
        "exclusive_output_directory",
        "registry_directory",
        "spent_at_utc",
        "process_id",
        "host",
        "fit_started_when_written",
        "receipt_payload_sha256",
        "receipt_zero_field_sha256",
    }
    if not isinstance(receipt, Mapping) or set(receipt) != keys:
        raise ValueError("execution spend receipt schema mismatch")
    unsigned = {
        key: receipt[key]
        for key in keys
        if key not in {"receipt_payload_sha256", "receipt_zero_field_sha256"}
    }
    provisional = {
        **unsigned,
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "receipt_zero_field_sha256": "0" * 64,
    }
    if (
        receipt["schema_version"] != EXECUTION_SPEND_SCHEMA_VERSION
        or receipt["status"] != "spent_before_any_fit_failure_remains_spent"
        or receipt["fit_started_when_written"] is not False
        or canonical_sha256(unsigned) != receipt["receipt_payload_sha256"]
        or _bytes_sha256(_strict_json_bytes(provisional))
        != receipt["receipt_zero_field_sha256"]
        or canonical_sha256(receipt["semantic_tuple"])
        != receipt["execution_key_sha256"]
        or not _is_sha256(receipt["protocol_lock_file_sha256"])
        or not _is_sha256(receipt["protocol_lock_payload_sha256"])
        or not Path(receipt["registry_directory"]).is_absolute()
        or not Path(receipt["exclusive_output_directory"]).is_absolute()
    ):
        raise ValueError("execution spend receipt identity mismatch")


def make_smoke_data() -> grid.GenotypeInput:
    """Return a deterministic five-group genotype matrix for non-evidence smoke."""

    rng = np.random.default_rng(881_701)
    populations = np.asarray(EXPECTED_POPULATIONS, dtype=str)
    samples_per_population = 10
    snps = 60
    labels = np.repeat(populations, samples_per_population)
    base = np.linspace(0.16, 0.44, snps)
    frequencies = np.tile(base, (populations.size, 1))
    for population in range(populations.size):
        start = population * 10
        stop = start + 10
        frequencies[population, start:stop] = np.clip(
            frequencies[population, start:stop] + 0.24,
            0.05,
            0.90,
        )
    genotypes = np.vstack(
        [
            rng.binomial(2, frequencies[index], size=(samples_per_population, snps))
            for index in range(populations.size)
        ]
    ).astype(float)
    data = grid.GenotypeInput(
        genotypes=np.asarray(genotypes, dtype=float),
        sample_ids=np.asarray(
            [f"smoke_sample_{index:03d}" for index in range(labels.size)]
        ),
        snp_ids=np.asarray([f"smoke_snp_{index:03d}" for index in range(snps)]),
        chromosomes=np.asarray([str(20 + index % 3) for index in range(snps)]),
        population_labels=labels,
    )
    grid._validate_input(data)
    return data


def _seed_like_key(value: str) -> bool:
    return bool(re.search(r"(?:^|_)seeds?(?:_|$)", value.lower()))


def _coerce_seed(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return int(number) if math.isfinite(number) and number.is_integer() else None
    if isinstance(value, str):
        stripped = value.strip()
        if re.fullmatch(r"[+-]?\d+", stripped):
            return int(stripped)
        try:
            number = float(stripped)
        except ValueError:
            return None
        return int(number) if math.isfinite(number) and number.is_integer() else None
    return None


_EXECUTION_CONTAINER_KEYS = {
    "cases",
    "case_records",
    "records",
    "seed_records",
    "seed_results",
    "results",
    "heldout_predictions",
    "predictions",
    "coordinates",
    "runs",
    "splits",
    "evaluations",
    "metrics",
    "rows",
    "per_seed_results",
}
_ALWAYS_EXECUTION_SEED_KEYS = {
    "completed_seeds",
    "confirmation_seeds_executed",
    "executed_seeds",
    "outer_seeds_executed",
}


def canonical_freshness_roots(repo_root: Path) -> tuple[Path, ...]:
    supplied = Path(repo_root)
    if supplied.is_symlink() or not supplied.exists() or not supplied.is_dir():
        raise ValueError("repository root must be a real directory")
    root = supplied.resolve()
    return tuple(root / relative for relative in CANONICAL_FRESHNESS_RELATIVE_ROOTS)


def _validate_canonical_freshness_root_strings(roots: Any) -> Path:
    if not isinstance(roots, list) or len(roots) != len(
        CANONICAL_FRESHNESS_RELATIVE_ROOTS
    ):
        raise ValueError("freshness audit must bind the two canonical roots")
    paths = [Path(value) for value in roots if isinstance(value, str)]
    if len(paths) != len(roots) or any(not path.is_absolute() for path in paths):
        raise ValueError("freshness roots must be absolute strings")
    first_relative = Path(CANONICAL_FRESHNESS_RELATIVE_ROOTS[0])
    repository_root = paths[0]
    for _ in first_relative.parts:
        repository_root = repository_root.parent
    expected = [
        repository_root / relative for relative in CANONICAL_FRESHNESS_RELATIVE_ROOTS
    ]
    if paths != expected:
        raise ValueError("freshness roots are not the canonical genomics evidence set")
    return repository_root


def build_freshness_scope_inventory(
    repo_root: Path, exclusive_output_directory: Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(exclusive_output_directory).resolve()
    included = [
        {
            "relative_path": relative,
            "absolute_path": str(root / relative),
            "role": role,
        }
        for relative, role in zip(
            CANONICAL_FRESHNESS_RELATIVE_ROOTS,
            (
                "post_hoc_pca_loading_grid_seed_registry",
                "locked_pca_confirmation_seed_registry",
                "two_stage_solver_feasibility_seed_registry",
            ),
            strict=True,
        )
    ]
    unsigned = {
        "schema_version": FRESHNESS_SCOPE_SCHEMA_VERSION,
        "included_roots": included,
        "excluded_broad_roots": [
            {
                "relative_path": "results",
                "reason": (
                    "contains unrelated video, legacy, and non-genomics results; "
                    "only the three enumerated genomics seed registries are relevant"
                ),
            },
            {
                "relative_path": "artifacts",
                "reason": (
                    "contains unrelated source snapshots, AppleDouble metadata, and "
                    "legacy symlinks; no scale-aware execution registry is stored there"
                ),
            },
        ],
        "reserved_scale_aware_output": {
            "absolute_path": str(output),
            "required_state_at_freeze_and_execution": "path_absent",
            "observed_exists": bool(os.path.lexists(output)),
            "reason": "exclusive formal package is written only after fresh evaluation",
        },
        "scope_rule": (
            "strict_recursive_audit_inside_each_included_root_no_silent_skip"
        ),
    }
    return {**unsigned, "inventory_sha256": canonical_sha256(unsigned)}


def validate_freshness_scope_inventory(
    inventory: Mapping[str, Any],
    *,
    freshness_roots: Sequence[str],
    exclusive_output_directory: str,
) -> None:
    if set(inventory) != {
        "schema_version",
        "included_roots",
        "excluded_broad_roots",
        "reserved_scale_aware_output",
        "scope_rule",
        "inventory_sha256",
    }:
        raise ValueError("freshness scope inventory schema mismatch")
    unsigned = dict(inventory)
    observed = unsigned.pop("inventory_sha256")
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError("freshness scope inventory digest mismatch")
    included = inventory["included_roots"]
    excluded = inventory["excluded_broad_roots"]
    if (
        not isinstance(included, list)
        or any(not isinstance(item, Mapping) for item in included)
        or any(
            set(item) != {"relative_path", "absolute_path", "role"} for item in included
        )
        or not isinstance(excluded, list)
        or any(not isinstance(item, Mapping) for item in excluded)
        or any(set(item) != {"relative_path", "reason"} for item in excluded)
    ):
        raise ValueError("freshness scope inventory entries are invalid")
    if (
        inventory["schema_version"] != FRESHNESS_SCOPE_SCHEMA_VERSION
        or inventory["scope_rule"]
        != "strict_recursive_audit_inside_each_included_root_no_silent_skip"
        or [item.get("absolute_path") for item in included] != list(freshness_roots)
        or [item.get("relative_path") for item in included]
        != list(CANONICAL_FRESHNESS_RELATIVE_ROOTS)
        or [item.get("role") for item in included]
        != [
            "post_hoc_pca_loading_grid_seed_registry",
            "locked_pca_confirmation_seed_registry",
            "two_stage_solver_feasibility_seed_registry",
        ]
        or [item.get("relative_path") for item in excluded] != ["results", "artifacts"]
        or [item.get("reason") for item in excluded]
        != [
            (
                "contains unrelated video, legacy, and non-genomics results; "
                "only the three enumerated genomics seed registries are relevant"
            ),
            (
                "contains unrelated source snapshots, AppleDouble metadata, and "
                "legacy symlinks; no scale-aware execution registry is stored there"
            ),
        ]
    ):
        raise ValueError("freshness scope inventory policy mismatch")
    reserved = inventory["reserved_scale_aware_output"]
    if (
        not isinstance(reserved, Mapping)
        or set(reserved)
        != {
            "absolute_path",
            "required_state_at_freeze_and_execution",
            "observed_exists",
            "reason",
        }
        or reserved.get("absolute_path") != exclusive_output_directory
        or reserved.get("required_state_at_freeze_and_execution") != "path_absent"
        or reserved.get("observed_exists") is not False
        or reserved.get("reason")
        != "exclusive formal package is written only after fresh evaluation"
    ):
        raise ValueError("freshness scope reserved output mismatch")


def _execution_seed_hits(
    value: Any,
    targets: set[int],
    *,
    path: str = "root",
    execution_context: bool = False,
) -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = str(key).lower()
            child_context = (
                execution_context
                or normalized_key in _EXECUTION_CONTAINER_KEYS
                or normalized_key in _ALWAYS_EXECUTION_SEED_KEYS
            )
            if child_context and _seed_like_key(str(key)):
                items = item if isinstance(item, list) else [item]
                for index, candidate in enumerate(items):
                    parsed = _coerce_seed(candidate)
                    if parsed in targets:
                        hits.append(f"{path}.{key}[{index}]={candidate!r}")
            hits.extend(
                _execution_seed_hits(
                    item,
                    targets,
                    path=f"{path}.{key}",
                    execution_context=child_context,
                )
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            hits.extend(
                _execution_seed_hits(
                    item,
                    targets,
                    path=f"{path}[{index}]",
                    execution_context=execution_context,
                )
            )
    return hits


def audit_fresh_seed_execution(
    roots: Sequence[Path],
    seeds: Sequence[int] = FORMAL_SEEDS,
) -> dict[str, Any]:
    """Fail if result-like files contain evidence that a fresh seed executed.

    Source and documentation files are intentionally outside this audit: the
    seed schedule was declared previously.  JSON execution containers, CSV
    seed columns, JSONL events, log seed tokens, and seed-bearing paths are
    inspected.  Numeric strings, leading zeroes, and integral floats are
    normalized before comparison.
    """

    targets = {int(seed) for seed in seeds}
    scanned: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    path_pattern = re.compile(
        r"(?:seed|split)[_-]?0*("
        + "|".join(str(seed) for seed in sorted(targets))
        + r")(?!\d)",
        re.IGNORECASE,
    )
    log_pattern = re.compile(
        r"(?:seed|split_seed|outer_seed)\s*[=:]\s*[\"']?([+\-0-9.eE]+)",
        re.IGNORECASE,
    )
    if not roots:
        raise ValueError("at least one freshness root is required")
    canonical_roots: list[str] = []
    root_snapshots: list[dict[str, Any]] = []
    for supplied_root in roots:
        candidate = Path(supplied_root)
        if candidate.is_symlink() or not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"freshness root must be a real directory: {candidate}")
        root = candidate.resolve()
        canonical_roots.append(str(root))
        root_scanned_start = len(scanned)
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"freshness audit refuses symlink: {path}")
            relative = path.relative_to(root).as_posix()
            if path_pattern.search(relative):
                hits.append(
                    {
                        "file": None,
                        "root": str(root),
                        "location": "path",
                        "detail": relative,
                    }
                )
            if not path.is_file():
                continue
            receipt = {
                "root": str(root),
                "relative_path": relative,
                "sha256": _file_sha256(path),
                "bytes": int(path.stat().st_size),
            }
            scanned.append(receipt)
            suffix = path.suffix.lower()
            if suffix == ".json":
                payload = _strict_json_load(path)
                for detail in _execution_seed_hits(payload, targets):
                    hits.append({"file": receipt, "location": "json", "detail": detail})
            elif suffix == ".csv":
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if reader.fieldnames is None:
                        raise ValueError(f"CSV lacks a header: {path}")
                    seed_fields = [
                        field for field in reader.fieldnames if _seed_like_key(field)
                    ]
                    for row_number, row in enumerate(reader, start=2):
                        for field in seed_fields:
                            parsed = _coerce_seed(row.get(field))
                            if parsed in targets:
                                hits.append(
                                    {
                                        "file": receipt,
                                        "location": "csv",
                                        "detail": f"row={row_number},field={field},value={row[field]!r}",
                                    }
                                )
            elif suffix == ".jsonl":
                with path.open("r", encoding="utf-8") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if not line.strip():
                            continue
                        payload = _strict_json_load_bytes(line.encode("utf-8"))
                        for detail in _execution_seed_hits(
                            {"records": [payload]}, targets
                        ):
                            hits.append(
                                {
                                    "file": receipt,
                                    "location": "jsonl",
                                    "detail": f"line={line_number}:{detail}",
                                }
                            )
            elif suffix in {".log", ".txt", ".out", ".err"}:
                # ASCII seed tokens remain searchable even in legacy logs whose
                # extension says text but whose surrounding bytes are not UTF-8.
                text = path.read_text(encoding="utf-8", errors="surrogateescape")
                for match in log_pattern.finditer(text):
                    parsed = _coerce_seed(match.group(1))
                    if parsed in targets:
                        hits.append(
                            {
                                "file": receipt,
                                "location": "log",
                                "detail": match.group(0),
                            }
                        )
            elif suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    for name in archive.files:
                        if path_pattern.search(name):
                            hits.append(
                                {
                                    "file": receipt,
                                    "location": "npz_key",
                                    "detail": name,
                                }
                            )
                        normalized_name = name.lower()
                        if _seed_like_key(normalized_name) or normalized_name in (
                            _ALWAYS_EXECUTION_SEED_KEYS
                        ):
                            for index, candidate in enumerate(
                                np.asarray(archive[name]).reshape(-1).tolist()
                            ):
                                if _coerce_seed(candidate) in targets:
                                    hits.append(
                                        {
                                            "file": receipt,
                                            "location": "npz_value",
                                            "detail": f"{name}[{index}]={candidate!r}",
                                        }
                                    )
            elif suffix == ".npy" and (
                _seed_like_key(path.stem)
                or path.stem.lower() in _ALWAYS_EXECUTION_SEED_KEYS
            ):
                values = np.load(path, allow_pickle=False)
                for index, candidate in enumerate(
                    np.asarray(values).reshape(-1).tolist()
                ):
                    if _coerce_seed(candidate) in targets:
                        hits.append(
                            {
                                "file": receipt,
                                "location": "npy_value",
                                "detail": f"index={index},value={candidate!r}",
                            }
                        )
        root_files = scanned[root_scanned_start:]
        if not root_files:
            raise ValueError(
                f"freshness root has no auditable result/log files: {root}"
            )
        root_snapshot_unsigned = {
            "root": str(root),
            "file_count": len(root_files),
            "files": root_files,
        }
        root_snapshots.append(
            {
                "root": str(root),
                "file_count": len(root_files),
                "snapshot_sha256": canonical_sha256(root_snapshot_unsigned),
            }
        )
    if len(set(canonical_roots)) != len(canonical_roots):
        raise ValueError("freshness roots must be unique")
    unsigned = {
        "roots": canonical_roots,
        "seeds": [int(seed) for seed in seeds],
        "files": scanned,
        "root_snapshots": root_snapshots,
        "snapshot_aggregate_sha256": canonical_sha256(root_snapshots),
        "hits": hits,
        "passed": not hits,
        "scope": (
            "all_regular_file_inventory_plus_structured_json_csv_jsonl_npz_npy_"
            "log_and_seed_bearing_paths"
        ),
    }
    result = {**unsigned, "audit_sha256": canonical_sha256(unsigned)}
    if hits:
        raise ValueError(f"fresh-seed execution evidence found: {hits[:3]}")
    return result


def build_formal_lock(
    *,
    repo_root: Path,
    data: grid.GenotypeInput,
    preparation_identity: Mapping[str, Any],
    freshness_roots: Sequence[Path],
    output_directory: Path,
    expected_shared_source_sha256: str,
    development_package_identity: Mapping[str, Any],
    execution_registry_directory: Path,
) -> dict[str, Any]:
    """Construct, but do not write, the fresh-confirmation protocol lock."""

    config = formal_config()
    grid._validate_input(data)
    expected_freshness_roots = canonical_freshness_roots(repo_root)
    observed_freshness_roots = tuple(Path(path).resolve() for path in freshness_roots)
    if observed_freshness_roots != expected_freshness_roots:
        raise ValueError(
            "formal lock freshness roots must be repo_root/results then repo_root/artifacts"
        )
    source_identity = collect_source_identity(repo_root)
    validate_source_identity(source_identity)
    if (
        not _is_sha256(expected_shared_source_sha256)
        or source_identity["aggregate_sha256"] != expected_shared_source_sha256
    ):
        raise ValueError("reviewed shared-source anchor does not match current source")
    output = output_directory.resolve()
    if os.path.lexists(output):
        raise FileExistsError("formal output directory must not exist at lock time")
    scope_inventory = build_freshness_scope_inventory(repo_root, output)
    freshness = audit_fresh_seed_execution(freshness_roots, config.seeds)
    input_identity = {
        **_input_identity(data),
        **dict(preparation_identity),
        "population_design": "five_superpopulation",
        "input_scope": "legacy_chr20_22",
        "ld_pruned": True,
        "population_label_is_race": False,
    }
    validate_formal_input_identity(input_identity)
    validate_development_package_identity(development_package_identity)
    if development_package_identity["prepared_input_identity"] != input_identity:
        raise ValueError("development bundle and formal lock use different input")
    if (
        development_package_identity["source_identity_aggregate_sha256"]
        != source_identity["aggregate_sha256"]
    ):
        raise ValueError("development bundle and formal lock use different source")
    registry = Path(execution_registry_directory)
    if (
        not registry.is_absolute()
        or registry.is_symlink()
        or not registry.is_dir()
    ):
        raise ValueError("formal execution registry must be an existing absolute directory")
    unsigned = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "frozen_before_fresh_confirmation_no_seed_executed",
        "protocol": _protocol(config),
        "development_package_identity": dict(development_package_identity),
        "source_identity": source_identity,
        "shared_core_sha256": {
            relative: source_identity["files"][relative]["sha256"]
            for relative in sorted(FROZEN_SHARED_CORE_HASHES)
        },
        "prepared_input_identity": input_identity,
        "freshness_audit": freshness,
        "freshness_scope_inventory": scope_inventory,
        "exclusive_output_directory": str(output),
        "global_execution_registry_directory": str(registry.resolve()),
        "execution_requires_explicit_authorization_flag": True,
        "post_hoc_development_values_are_frozen": True,
    }
    return {**unsigned, "lock_payload_sha256": canonical_sha256(unsigned)}


_LOCK_KEYS = {
    "schema_version",
    "status",
    "protocol",
    "development_package_identity",
    "source_identity",
    "shared_core_sha256",
    "prepared_input_identity",
    "freshness_audit",
    "freshness_scope_inventory",
    "exclusive_output_directory",
    "global_execution_registry_directory",
    "execution_requires_explicit_authorization_flag",
    "post_hoc_development_values_are_frozen",
    "lock_payload_sha256",
}


def validate_formal_lock(lock: Mapping[str, Any]) -> None:
    if set(lock) != _LOCK_KEYS:
        raise ValueError("formal lock has missing or unknown fields")
    if (
        lock["schema_version"] != LOCK_SCHEMA_VERSION
        or lock["status"] != "frozen_before_fresh_confirmation_no_seed_executed"
        or lock["protocol"] != _protocol(formal_config())
        or lock["execution_requires_explicit_authorization_flag"] is not True
        or lock["post_hoc_development_values_are_frozen"] is not True
    ):
        raise ValueError("formal lock protocol/status differs from the frozen contract")
    development = lock["development_package_identity"]
    if not isinstance(development, Mapping):
        raise ValueError("formal lock development package identity is invalid")
    validate_development_package_identity(development)
    source = lock["source_identity"]
    if not isinstance(source, Mapping):
        raise ValueError("formal lock source identity is invalid")
    validate_source_identity(source)
    expected_core = {
        relative: source["files"][relative]["sha256"]
        for relative in sorted(FROZEN_SHARED_CORE_HASHES)
    }
    if lock["shared_core_sha256"] != expected_core or expected_core != {
        relative: FROZEN_SHARED_CORE_HASHES[relative]
        for relative in sorted(FROZEN_SHARED_CORE_HASHES)
    }:
        raise ValueError("formal lock does not bind the frozen shared core")
    prepared_input = lock["prepared_input_identity"]
    if not isinstance(prepared_input, Mapping):
        raise ValueError("formal lock prepared-input identity is invalid")
    validate_formal_input_identity(prepared_input)
    if (
        development["prepared_input_identity"] != prepared_input
        or development["source_identity_aggregate_sha256"]
        != source["aggregate_sha256"]
    ):
        raise ValueError("formal lock development/input/source cross-anchor mismatch")
    freshness = lock["freshness_audit"]
    if (
        not isinstance(freshness, Mapping)
        or set(freshness)
        != {
            "roots",
            "seeds",
            "files",
            "root_snapshots",
            "snapshot_aggregate_sha256",
            "hits",
            "passed",
            "scope",
            "audit_sha256",
        }
        or freshness.get("passed") is not True
        or freshness.get("hits") != []
        or freshness.get("seeds") != list(FORMAL_SEEDS)
        or freshness.get("scope")
        != (
            "all_regular_file_inventory_plus_structured_json_csv_jsonl_npz_npy_"
            "log_and_seed_bearing_paths"
        )
    ):
        raise ValueError("formal lock freshness audit is not a clean exact schedule")
    _validate_canonical_freshness_root_strings(freshness.get("roots"))
    scope_inventory = lock["freshness_scope_inventory"]
    if not isinstance(scope_inventory, Mapping):
        raise ValueError("formal lock freshness scope inventory is invalid")
    validate_freshness_scope_inventory(
        scope_inventory,
        freshness_roots=freshness["roots"],
        exclusive_output_directory=lock["exclusive_output_directory"],
    )
    files = freshness.get("files")
    snapshots = freshness.get("root_snapshots")
    if not isinstance(files, list) or not files or not isinstance(snapshots, list):
        raise ValueError("formal lock freshness snapshot is empty")
    if any(not isinstance(receipt, Mapping) for receipt in files):
        raise ValueError("formal lock freshness file list is invalid")
    expected_snapshots: list[dict[str, Any]] = []
    for root in freshness["roots"]:
        root_files = [receipt for receipt in files if receipt.get("root") == root]
        if not root_files:
            raise ValueError("each canonical freshness root must contain audited files")
        for receipt in root_files:
            if (
                not isinstance(receipt, Mapping)
                or set(receipt) != {"root", "relative_path", "sha256", "bytes"}
                or not isinstance(receipt["relative_path"], str)
                or Path(receipt["relative_path"]).is_absolute()
                or ".." in Path(receipt["relative_path"]).parts
                or not _is_sha256(receipt["sha256"])
                or isinstance(receipt["bytes"], bool)
                or not isinstance(receipt["bytes"], int)
                or receipt["bytes"] < 0
            ):
                raise ValueError("formal lock freshness file receipt is invalid")
        root_snapshot_unsigned = {
            "root": root,
            "file_count": len(root_files),
            "files": root_files,
        }
        expected_snapshots.append(
            {
                "root": root,
                "file_count": len(root_files),
                "snapshot_sha256": canonical_sha256(root_snapshot_unsigned),
            }
        )
    if snapshots != expected_snapshots or freshness.get(
        "snapshot_aggregate_sha256"
    ) != canonical_sha256(expected_snapshots):
        raise ValueError("formal lock freshness root snapshot mismatch")
    freshness_unsigned = dict(freshness)
    observed_freshness_digest = freshness_unsigned.pop("audit_sha256", None)
    if (
        not _is_sha256(observed_freshness_digest)
        or canonical_sha256(freshness_unsigned) != observed_freshness_digest
    ):
        raise ValueError("formal lock freshness digest mismatch")
    if (
        not isinstance(lock["exclusive_output_directory"], str)
        or not Path(lock["exclusive_output_directory"]).is_absolute()
        or not isinstance(lock["global_execution_registry_directory"], str)
        or not Path(lock["global_execution_registry_directory"]).is_absolute()
    ):
        raise ValueError("formal lock output/registry directories must be absolute")
    unsigned = dict(lock)
    observed = unsigned.pop("lock_payload_sha256")
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError("formal lock payload digest mismatch")


def load_anchored_formal_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    _require_regular_file(path, role="formal protocol lock")
    if not _is_sha256(expected_sha256) or _file_sha256(path) != expected_sha256:
        raise ValueError("formal lock external SHA-256 anchor mismatch")
    lock = _strict_json_load(path)
    if not isinstance(lock, Mapping):
        raise ValueError("formal lock must be a JSON object")
    validate_formal_lock(lock)
    return dict(lock)


def _normalized_stationarity(raw: float, alpha: float, size: int) -> dict[str, float]:
    raw = float(raw)
    alpha = float(alpha)
    if not math.isfinite(raw) or raw < 0.0:
        raise ValueError("raw stationarity must be finite and nonnegative")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("scale-aware stationarity requires positive alpha")
    if isinstance(size, bool) or int(size) <= 0:
        raise ValueError("stationarity size must be positive")
    rms = raw / math.sqrt(int(size))
    return {
        "raw_frobenius": raw,
        "kkt_rms": rms,
        "eta": rms / alpha,
    }


def _support_receipt(loadings: FloatArray, config: RunConfig) -> dict[str, Any]:
    matrix = np.asarray(loadings, dtype=float)
    support = np.abs(matrix) > config.support_threshold
    entry_count = int(np.count_nonzero(support))
    row_support = np.any(support, axis=1)
    row_count = int(np.count_nonzero(row_support))
    component_counts = support.sum(axis=0).astype(np.int64)
    return {
        "threshold": config.support_threshold,
        "entry_count": entry_count,
        "entry_fraction": float(entry_count / support.size),
        "row_count": row_count,
        "row_fraction": float(row_count / matrix.shape[0]),
        "component_counts": component_counts.tolist(),
        "matrix_rank": int(np.linalg.matrix_rank(matrix)),
    }


def _common_diagnostics(
    matrix: FloatArray,
    loadings: FloatArray,
    *,
    alpha: float,
    reference_step: float,
) -> tuple[FloatArray, float, dict[str, float]]:
    factor = procrustes_factor(matrix, loadings)
    objective = float(vp_objective(matrix, factor, loadings, alpha))
    mapping = proximal_gradient_mapping(
        matrix,
        factor,
        loadings,
        alpha=alpha,
        step_size=reference_step,
    )
    raw = float(np.linalg.norm(mapping, ord="fro"))
    return factor, objective, _normalized_stationarity(raw, alpha, loadings.size)


def _metrics_and_predictions(
    train_scores: FloatArray,
    train_labels: StringArray,
    test_scores: FloatArray,
    test_labels: StringArray,
) -> tuple[dict[str, Any], StringArray, FloatArray]:
    predictions, classes, centroids = grid._nearest_centroid(
        np.asarray(train_scores, dtype=float),
        np.asarray(train_labels, dtype=str),
        np.asarray(test_scores, dtype=float),
    )
    metrics = grid._classification_metrics(
        np.asarray(test_labels, dtype=str),
        predictions,
        classes,
    )
    return metrics, predictions, centroids


def _fit_sparse_once(
    method: str,
    panel_train: FloatArray,
    *,
    alpha: float,
    pca_initial_loadings: FloatArray,
    reference_step: float,
    seed: int,
    config: RunConfig,
) -> tuple[FloatArray, dict[str, Any], float, FloatArray]:
    start = perf_counter()
    if method == "vp":
        result = fit_vp_spca(
            panel_train,
            config.rank,
            alpha=alpha,
            max_iter=config.total_budget,
            tol=RAW_TOL_FLOOR,
            step_size=reference_step,
            initial_B=pca_initial_loadings,
            backtracking=True,
            verbose=False,
        )
        seconds = perf_counter() - start
        trace = np.asarray(
            result.stationarity[-config.native_last_window :], dtype=float
        )
        metadata = {
            "native_status": str(result.status),
            "native_raw_converged": bool(result.converged),
            "native_iterations": int(result.n_iter),
            "sketch_iterations": None,
            "full_refinement_iterations": None,
            "randomized_seed": None,
            "sketch_residual_relative": None,
        }
        return np.asarray(result.B, dtype=float), metadata, float(seconds), trace
    if method != "randomized":
        raise ValueError(f"unsupported sparse method: {method}")
    randomized_seed = int(seed) + 730_000
    workspace = prepare_randomized_vp_spca_workspace(
        panel_train,
        config.rank,
        oversample=config.randomized_oversample,
        n_power_iter=config.randomized_power_iterations,
        random_state=randomized_seed,
    )
    result = fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=config.randomized_sketch_budget,
        tol=RAW_TOL_FLOOR,
        use_full_gradient=False,
        finalize_full_iters=config.randomized_full_budget,
        verbose=False,
    )
    seconds = perf_counter() - start
    trace = np.asarray(result.stationarity[-config.native_last_window :], dtype=float)
    metadata = {
        "native_status": str(result.status),
        "native_raw_converged": bool(result.full_converged),
        "native_iterations": int(result.n_iter),
        "sketch_iterations": int(result.sketch_iterations),
        "full_refinement_iterations": int(result.full_refinement_iterations),
        "randomized_seed": randomized_seed,
        "sketch_residual_relative": float(result.sketch_residual_relative),
    }
    return np.asarray(result.B, dtype=float), metadata, float(seconds), trace


def _deterministic_fit_identity(
    loadings: FloatArray,
    metadata: Mapping[str, Any],
    trace: FloatArray,
) -> str:
    return canonical_sha256(
        {
            "loadings_sha256": array_sha256(np.asarray(loadings, dtype=float)),
            "trace_sha256": array_sha256(np.asarray(trace, dtype=float)),
            "metadata": dict(metadata),
        }
    )


def _timed_sparse_fits(
    panel_train: FloatArray,
    *,
    alpha: float,
    pca_initial_loadings: FloatArray,
    reference_step: float,
    seed: int,
    config: RunConfig,
) -> dict[str, dict[str, Any]]:
    base_order = ("vp", "randomized") if seed % 2 else ("randomized", "vp")
    for _ in range(config.warmup_repetitions):
        for method in base_order:
            _fit_sparse_once(
                method,
                panel_train,
                alpha=alpha,
                pca_initial_loadings=pca_initial_loadings,
                reference_step=reference_step,
                seed=seed,
                config=config,
            )
    fits: dict[str, list[tuple[FloatArray, dict[str, Any], float, FloatArray, int]]] = {
        "vp": [],
        "randomized": [],
    }
    for repetition in range(config.timed_repetitions):
        order = base_order if repetition % 2 == 0 else tuple(reversed(base_order))
        for position, method in enumerate(order):
            loadings, metadata, seconds, trace = _fit_sparse_once(
                method,
                panel_train,
                alpha=alpha,
                pca_initial_loadings=pca_initial_loadings,
                reference_step=reference_step,
                seed=seed,
                config=config,
            )
            fits[method].append((loadings, metadata, seconds, trace, position))
    output: dict[str, dict[str, Any]] = {}
    for method in ("vp", "randomized"):
        repetitions = fits[method]
        identities = [
            _deterministic_fit_identity(loadings, metadata, trace)
            for loadings, metadata, _, trace, _ in repetitions
        ]
        if len(set(identities)) != 1:
            raise RuntimeError(
                f"{method} timed repetitions changed deterministic output"
            )
        output[method] = {
            "loadings": repetitions[0][0],
            "metadata": repetitions[0][1],
            "native_last_raw": repetitions[0][3],
            "solver_seconds_repetitions": [float(item[2]) for item in repetitions],
            "timing_positions": [int(item[4]) for item in repetitions],
            "solver_seconds_median": float(
                np.median([float(item[2]) for item in repetitions])
            ),
            "deterministic_output_sha256": identities[0],
            "deterministic_repetitions_identical": True,
        }
    return output


def _prediction_rows(
    *,
    seed: int,
    method: str,
    view: str,
    test_indices: IntArray,
    sample_ids: StringArray,
    truth: StringArray,
    predictions: StringArray,
) -> list[dict[str, Any]]:
    return [
        {
            "seed": int(seed),
            "method": method,
            "view": view,
            "sample_index": int(sample_index),
            "sample_id": str(sample_ids[sample_index]),
            "truth": str(truth[local_index]),
            "prediction": str(predictions[local_index]),
        }
        for local_index, sample_index in enumerate(test_indices.tolist())
    ]


def _record_metrics_block(
    metrics: Mapping[str, Any],
    predictions: StringArray,
    centroids: FloatArray,
) -> dict[str, Any]:
    return {
        "metrics": dict(metrics),
        "predictions_sha256": array_sha256(np.asarray(predictions, dtype=str)),
        "centroids_sha256": array_sha256(np.asarray(centroids, dtype=float)),
    }


def evaluate(
    data: grid.GenotypeInput,
    config: RunConfig,
    *,
    _formal_capability: _FormalExecutionCapability | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, np.ndarray],
    list[dict[str, Any]],
]:
    """Run one profile; formal seeds require a spent, private capability."""

    if config.profile == "formal":
        if not isinstance(_formal_capability, _FormalExecutionCapability):
            raise PermissionError(
                "public evaluate refuses formal seeds; use run_formal after atomic registry spend"
            )
        _formal_capability.claim(data, config)
    elif _formal_capability is not None:
        raise PermissionError("formal capability cannot be used for a non-formal profile")

    grid._validate_input(data)
    if (
        tuple(sorted(np.unique(data.population_labels).tolist()))
        != EXPECTED_POPULATIONS
    ):
        raise ValueError("the experiment requires exactly AFR/AMR/EAS/EUR/SAS")
    if (
        config.randomized_sketch_budget + config.randomized_full_budget
        != config.total_budget
    ):
        raise ValueError("randomized and VP total proximal budgets differ")
    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    split_receipts: list[dict[str, Any]] = []

    for seed in config.seeds:
        shared_start = perf_counter()
        train_indices, test_indices = stratified_split(
            labels, config.test_fraction, seed
        )
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        preprocessor = fit_smartpca_preprocessor(raw[train_indices])
        train_matrix = transform_smartpca(raw[train_indices], preprocessor)
        test_matrix = transform_smartpca(raw[test_indices], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        if retained_original.size < config.panel_size:
            raise ValueError(
                "training-retained SNP count is below the fixed panel size"
            )
        full_loadings, _ = grid.fit_top_pca(train_matrix, config.rank)
        ranking, importance = grid.rank_by_pca_loadings(full_loadings, config.rank)
        selected_retained = np.sort(ranking[: config.panel_size]).astype(np.int64)
        selected_original = np.sort(retained_original[selected_retained]).astype(
            np.int64
        )
        panel_train = np.ascontiguousarray(train_matrix[:, selected_retained])
        panel_test = np.ascontiguousarray(test_matrix[:, selected_retained])
        shared_screen_seconds = perf_counter() - shared_start

        pca_start = perf_counter()
        pca_loadings, singular_values = grid.fit_top_pca(panel_train, config.rank)
        pca_seconds = perf_counter() - pca_start
        sigma_max = float(singular_values[0])
        lipschitz = sigma_max * sigma_max
        if not math.isfinite(lipschitz) or lipschitz <= 0.0:
            raise RuntimeError("panel spectral scale must be finite and positive")
        reference_step = 1.0 / lipschitz
        alpha = config.rho * lipschitz
        if not math.isclose(reference_step * alpha, config.rho, rel_tol=1e-14):
            raise RuntimeError("scale-aware alpha does not reproduce the locked rho")

        sparse_fits = _timed_sparse_fits(
            panel_train,
            alpha=alpha,
            pca_initial_loadings=pca_loadings,
            reference_step=reference_step,
            seed=seed,
            config=config,
        )
        common_probe_metrics, common_probe_predictions, common_probe_centroids = (
            _metrics_and_predictions(
                panel_train @ pca_loadings,
                labels[train_indices],
                panel_test @ pca_loadings,
                labels[test_indices],
            )
        )

        prefix = f"seed_{seed}"
        arrays[f"{prefix}_train_indices"] = train_indices
        arrays[f"{prefix}_test_indices"] = test_indices
        arrays[f"{prefix}_retained_original_indices"] = retained_original
        arrays[f"{prefix}_full_width_pca_loadings"] = full_loadings
        arrays[f"{prefix}_loading_importance"] = importance
        arrays[f"{prefix}_selected_retained_indices"] = selected_retained
        arrays[f"{prefix}_selected_original_indices"] = selected_original
        arrays[f"{prefix}_selected_snp_ids"] = np.asarray(
            data.snp_ids[selected_original], dtype=str
        )
        arrays[f"{prefix}_pca_loadings"] = pca_loadings
        arrays[f"{prefix}_vp_loadings"] = sparse_fits["vp"]["loadings"]
        arrays[f"{prefix}_randomized_loadings"] = sparse_fits["randomized"]["loadings"]
        arrays[f"{prefix}_vp_native_last_raw"] = sparse_fits["vp"]["native_last_raw"]
        arrays[f"{prefix}_randomized_native_last_raw"] = sparse_fits["randomized"][
            "native_last_raw"
        ]

        split_hashes = {
            "train_indices_sha256": array_sha256(train_indices),
            "test_indices_sha256": array_sha256(test_indices),
            "train_sample_ids_sha256": array_sha256(sample_ids[train_indices]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test_indices]),
            "test_labels_sha256": array_sha256(labels[test_indices]),
            "retained_original_indices_sha256": array_sha256(retained_original),
            "selected_retained_indices_sha256": array_sha256(selected_retained),
            "selected_original_indices_sha256": array_sha256(selected_original),
            "selected_snp_ids_sha256": array_sha256(
                np.asarray(data.snp_ids[selected_original], dtype=str)
            ),
            "panel_train_sha256": array_sha256(panel_train),
            "panel_test_sha256": array_sha256(panel_test),
            "preprocessor_mean_sha256": array_sha256(preprocessor.mean_dosage),
            "preprocessor_scale_sha256": array_sha256(preprocessor.scale),
        }
        split_receipts.append(
            {
                "seed": int(seed),
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                "retained_snps": int(retained_original.size),
                "selected_original_indices": selected_original.tolist(),
                "selected_snp_ids": np.asarray(
                    data.snp_ids[selected_original], dtype=str
                ).tolist(),
                "ordered_original_ranking_sha256": array_sha256(
                    retained_original[ranking]
                ),
                "loading_importance_sha256": array_sha256(importance),
                "full_width_pca_loadings_sha256": array_sha256(full_loadings),
                "population_labels_used_for_screen": False,
                "test_rows_used_for_screen": False,
                **split_hashes,
            }
        )

        method_loadings = {
            "pca": pca_loadings,
            "vp": sparse_fits["vp"]["loadings"],
            "randomized": sparse_fits["randomized"]["loadings"],
        }
        for method in METHODS:
            evaluation_start = perf_counter()
            loadings = np.asarray(method_loadings[method], dtype=float)
            factor, objective, common_stationarity = _common_diagnostics(
                panel_train,
                loadings,
                alpha=alpha,
                reference_step=reference_step,
            )
            native_metrics, native_predictions, native_centroids = (
                _metrics_and_predictions(
                    panel_train @ loadings,
                    labels[train_indices],
                    panel_test @ loadings,
                    labels[test_indices],
                )
            )
            arrays[f"{prefix}_{method}_procrustes_factor"] = factor
            arrays[f"{prefix}_{method}_native_centroids"] = native_centroids
            arrays[f"{prefix}_{method}_common_pca_probe_centroids"] = (
                common_probe_centroids
            )
            support = _support_receipt(loadings, config)
            if method == "pca":
                metadata = {
                    "native_status": "closed_form_pca",
                    "native_raw_converged": True,
                    "native_iterations": 1,
                    "sketch_iterations": None,
                    "full_refinement_iterations": None,
                    "randomized_seed": None,
                    "sketch_residual_relative": None,
                }
                native_stationarity = None
                repetitions = [float(pca_seconds)]
                positions = [0]
                solver_seconds = float(pca_seconds)
                deterministic_output_sha256 = array_sha256(loadings)
                repetitions_identical = True
                common_reference_charge = 0.0
            else:
                fit = sparse_fits[method]
                metadata = dict(fit["metadata"])
                trace = np.asarray(fit["native_last_raw"], dtype=float)
                eta_trace = trace / (alpha * math.sqrt(loadings.size))
                rms_trace = trace / math.sqrt(loadings.size)
                native_stationarity = {
                    "last_window_count": int(trace.size),
                    "last_raw_frobenius": trace.tolist(),
                    "last_kkt_rms": rms_trace.tolist(),
                    "last_eta": eta_trace.tolist(),
                    "last_raw_sha256": array_sha256(trace),
                    "last_kkt_rms_sha256": array_sha256(rms_trace),
                    "last_eta_sha256": array_sha256(eta_trace),
                    "final_raw_frobenius": float(trace[-1]),
                    "final_kkt_rms": float(rms_trace[-1]),
                    "final_eta": float(eta_trace[-1]),
                    "last_window_max_eta": float(np.max(eta_trace)),
                    "last_window_all_qualified": bool(
                        np.all(eta_trace <= config.normalized_kkt_tolerance)
                    ),
                }
                repetitions = [
                    float(value) for value in fit["solver_seconds_repetitions"]
                ]
                positions = [int(value) for value in fit["timing_positions"]]
                solver_seconds = float(fit["solver_seconds_median"])
                deterministic_output_sha256 = str(fit["deterministic_output_sha256"])
                repetitions_identical = bool(fit["deterministic_repetitions_identical"])
                common_reference_charge = float(pca_seconds)
            evaluation_seconds = perf_counter() - evaluation_start
            support_qualified = bool(
                support["matrix_rank"] == config.rank
                and min(support["component_counts"]) >= config.minimum_component_support
                and support["entry_fraction"] <= config.maximum_entry_support_fraction
                and support["row_fraction"] >= config.minimum_split_row_support_fraction
                and support["row_fraction"] <= config.maximum_split_row_support_fraction
            )
            normalized_qualified = (
                None
                if method == "pca"
                else bool(
                    native_stationarity["last_window_all_qualified"]
                    and common_stationarity["eta"] <= config.normalized_kkt_tolerance
                )
            )
            record = {
                "seed": int(seed),
                "method": method,
                "rank": config.rank,
                "panel_size": config.panel_size,
                "rho": config.rho,
                "sigma_max": sigma_max,
                "lipschitz_sigma_max_squared": lipschitz,
                "reference_step_size": reference_step,
                "alpha": alpha,
                "rho_recomputed": reference_step * alpha,
                "loading_matrix_sha256": array_sha256(loadings),
                "procrustes_factor_sha256": array_sha256(factor),
                "common_objective": objective,
                "common_stationarity": common_stationarity,
                "native_stationarity": native_stationarity,
                "normalized_kkt_qualified": normalized_qualified,
                "support": support,
                "support_qualified": None if method == "pca" else support_qualified,
                "native": _record_metrics_block(
                    native_metrics, native_predictions, native_centroids
                ),
                "common_pca_probe": _record_metrics_block(
                    common_probe_metrics,
                    common_probe_predictions,
                    common_probe_centroids,
                ),
                "runtime_seconds": {
                    "shared_train_only_screen": float(shared_screen_seconds),
                    "common_pca_reference": float(pca_seconds),
                    "common_pca_reference_charged_to_end_to_end": (
                        common_reference_charge
                    ),
                    "solver_only_repetitions": repetitions,
                    "solver_only_median": solver_seconds,
                    "solver_only_scope": (
                        "closed_form_panel_pca"
                        if method == "pca"
                        else (
                            "proximal_iterations_from_frozen_exact_pca_initialization_"
                            "excluding_pca_svd"
                            if method == "vp"
                            else "randomized_workspace_plus_sketch_and_full_refinement"
                        )
                    ),
                    "standalone_exact_pca_initialization_seconds": float(pca_seconds),
                    "timing_positions": positions,
                    "warmup_repetitions": config.warmup_repetitions,
                    "timed_repetitions": (
                        1 if method == "pca" else config.timed_repetitions
                    ),
                    "common_diagnostic_and_scoring": float(evaluation_seconds),
                    "incremental_after_shared_screen": float(
                        common_reference_charge + solver_seconds + evaluation_seconds
                    ),
                    "end_to_end": float(
                        shared_screen_seconds
                        + common_reference_charge
                        + solver_seconds
                        + evaluation_seconds
                    ),
                },
                "deterministic_output_sha256": deterministic_output_sha256,
                "deterministic_repetitions_identical": repetitions_identical,
                "panel_selected_by_train_only_pca_loadings": True,
                "same_panel_for_all_methods_within_split": True,
                "test_used_for_preprocessing_screen_or_solver": False,
                **metadata,
                **split_hashes,
            }
            records.append(record)
            predictions.extend(
                _prediction_rows(
                    seed=seed,
                    method=method,
                    view="native",
                    test_indices=test_indices,
                    sample_ids=sample_ids,
                    truth=labels[test_indices],
                    predictions=native_predictions,
                )
            )
            predictions.extend(
                _prediction_rows(
                    seed=seed,
                    method=method,
                    view="common_pca_probe",
                    test_indices=test_indices,
                    sample_ids=sample_ids,
                    truth=labels[test_indices],
                    predictions=common_probe_predictions,
                )
            )
    order = {method: index for index, method in enumerate(METHODS)}
    view_order = {view: index for index, view in enumerate(VIEWS)}
    records.sort(key=lambda row: (int(row["seed"]), order[str(row["method"])]))
    predictions.sort(
        key=lambda row: (
            int(row["seed"]),
            order[str(row["method"])],
            view_order[str(row["view"])],
            int(row["sample_index"]),
        )
    )
    split_receipts.sort(key=lambda row: int(row["seed"]))
    return records, predictions, arrays, split_receipts


def _population_recall(record: Mapping[str, Any], view: str, population: str) -> float:
    return float(
        next(
            item["recall"]
            for item in record[view]["metrics"]["per_population_metrics"]
            if item["population"] == population
        )
    )


def summarize_records(
    records: Sequence[Mapping[str, Any]], config: RunConfig
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        if len(rows) != len(config.seeds):
            raise ValueError(f"record count mismatch for {method}")
        summary: dict[str, Any] = {
            "method": method,
            "seed_count": len(rows),
            "native_balanced_accuracy_mean": float(
                np.mean([row["native"]["metrics"]["balanced_accuracy"] for row in rows])
            ),
            "native_macro_f1_mean": float(
                np.mean([row["native"]["metrics"]["macro_f1"] for row in rows])
            ),
            "common_probe_balanced_accuracy_mean": float(
                np.mean(
                    [
                        row["common_pca_probe"]["metrics"]["balanced_accuracy"]
                        for row in rows
                    ]
                )
            ),
            "common_probe_macro_f1_mean": float(
                np.mean(
                    [row["common_pca_probe"]["metrics"]["macro_f1"] for row in rows]
                )
            ),
            "native_per_population_recall_mean": {
                population: float(
                    np.mean(
                        [_population_recall(row, "native", population) for row in rows]
                    )
                )
                for population in EXPECTED_POPULATIONS
            },
            "common_probe_per_population_recall_mean": {
                population: float(
                    np.mean(
                        [
                            _population_recall(row, "common_pca_probe", population)
                            for row in rows
                        ]
                    )
                )
                for population in EXPECTED_POPULATIONS
            },
            "entry_support_fraction_mean": float(
                np.mean([row["support"]["entry_fraction"] for row in rows])
            ),
            "row_support_fraction_mean": float(
                np.mean([row["support"]["row_fraction"] for row in rows])
            ),
            "row_support_count_mean": float(
                np.mean([row["support"]["row_count"] for row in rows])
            ),
            "solver_only_seconds_median": float(
                np.median(
                    [row["runtime_seconds"]["solver_only_median"] for row in rows]
                )
            ),
            "end_to_end_seconds_median": float(
                np.median([row["runtime_seconds"]["end_to_end"] for row in rows])
            ),
            "normalized_kkt_qualified_count": (
                None
                if method == "pca"
                else int(sum(bool(row["normalized_kkt_qualified"]) for row in rows))
            ),
            "support_qualified_count": (
                None
                if method == "pca"
                else int(sum(bool(row["support_qualified"]) for row in rows))
            ),
            "common_eta_max": (
                None
                if method == "pca"
                else float(max(row["common_stationarity"]["eta"] for row in rows))
            ),
        }
        summaries.append(summary)
    return summaries


def _paired_speed_bootstrap(
    vp_rows: Sequence[Mapping[str, Any]],
    randomized_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(vp_rows) != len(randomized_rows) or not vp_rows:
        raise ValueError("paired speed bootstrap requires matched nonempty seeds")
    per_seed: list[dict[str, Any]] = []
    log_statistics: list[float] = []
    for vp_row, randomized_row in zip(vp_rows, randomized_rows, strict=True):
        if vp_row["seed"] != randomized_row["seed"]:
            raise ValueError("paired speed rows changed seed order")
        vp_times = np.asarray(
            vp_row["runtime_seconds"]["solver_only_repetitions"], dtype=float
        )
        randomized_times = np.asarray(
            randomized_row["runtime_seconds"]["solver_only_repetitions"],
            dtype=float,
        )
        if (
            vp_times.shape != randomized_times.shape
            or vp_times.ndim != 1
            or vp_times.size < 1
            or not np.all(np.isfinite(vp_times))
            or not np.all(np.isfinite(randomized_times))
            or np.any(vp_times <= 0)
            or np.any(randomized_times <= 0)
        ):
            raise ValueError("paired speed repetitions are invalid")
        ratios = vp_times / randomized_times
        seed_log = float(np.median(np.log(ratios)))
        log_statistics.append(seed_log)
        per_seed.append(
            {
                "seed": int(vp_row["seed"]),
                "paired_repetitions": int(ratios.size),
                "median_speedup": float(math.exp(seed_log)),
                "speedups_sha256": array_sha256(np.asarray(ratios, dtype=float)),
            }
        )
    values = np.asarray(log_statistics, dtype=float)
    point = float(math.exp(float(np.mean(values))))
    rng = np.random.default_rng(SPEED_BOOTSTRAP_SEED)
    indices = rng.integers(
        0, values.size, size=(SPEED_BOOTSTRAP_DRAWS, values.size), endpoint=False
    )
    bootstrap = np.exp(np.mean(values[indices], axis=1))
    tail = (1.0 - SPEED_BOOTSTRAP_CONFIDENCE) / 2.0
    lower, upper = np.quantile(bootstrap, [tail, 1.0 - tail], method="linear")
    return {
        "unit": "outer_seed_cluster_of_paired_repetitions",
        "statistic": "geometric_mean_of_per_seed_median_speedup",
        "seed": SPEED_BOOTSTRAP_SEED,
        "draws": SPEED_BOOTSTRAP_DRAWS,
        "confidence": SPEED_BOOTSTRAP_CONFIDENCE,
        "seed_count": int(values.size),
        "point_estimate": point,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "per_seed": per_seed,
        "bootstrap_distribution_sha256": array_sha256(
            np.asarray(bootstrap, dtype=float)
        ),
    }


def compute_gates(
    records: Sequence[Mapping[str, Any]], config: RunConfig
) -> dict[str, Any]:
    index = {(int(record["seed"]), str(record["method"])): record for record in records}
    pca = [index[(seed, "pca")] for seed in config.seeds]
    vp = [index[(seed, "vp")] for seed in config.seeds]
    randomized = [index[(seed, "randomized")] for seed in config.seeds]

    def ba_delta(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        return np.asarray(
            [
                row["native"]["metrics"]["balanced_accuracy"]
                - baseline["native"]["metrics"]["balanced_accuracy"]
                for row, baseline in zip(rows, pca, strict=True)
            ],
            dtype=float,
        )

    vp_delta = ba_delta(vp)
    randomized_delta = ba_delta(randomized)
    quality_rows: list[dict[str, Any]] = []
    for seed in config.seeds:
        vp_row = index[(seed, "vp")]
        randomized_row = index[(seed, "randomized")]
        vp_objective = float(vp_row["common_objective"])
        randomized_objective = float(randomized_row["common_objective"])
        objective_excess = (randomized_objective - vp_objective) / max(
            1.0, abs(vp_objective)
        )
        eta_limit = max(
            config.normalized_kkt_tolerance,
            RANDOMIZED_ETA_FACTOR_MAX * float(vp_row["common_stationarity"]["eta"]),
        )
        vp_seconds = float(vp_row["runtime_seconds"]["solver_only_median"])
        randomized_seconds = float(
            randomized_row["runtime_seconds"]["solver_only_median"]
        )
        quality_rows.append(
            {
                "seed": int(seed),
                "randomized_objective_relative_excess": float(objective_excess),
                "randomized_common_eta_limit": float(eta_limit),
                "randomized_common_eta": float(
                    randomized_row["common_stationarity"]["eta"]
                ),
                "matched_objective_and_eta": bool(
                    objective_excess <= RANDOMIZED_OBJECTIVE_EXCESS_MAX
                    and randomized_row["common_stationarity"]["eta"] <= eta_limit
                ),
                "randomized_solver_speedup_vs_vp": float(
                    vp_seconds / randomized_seconds
                ),
                "randomized_faster": bool(randomized_seconds < vp_seconds),
            }
        )
    speedups = np.asarray(
        [row["randomized_solver_speedup_vs_vp"] for row in quality_rows], dtype=float
    )
    speed_bootstrap = _paired_speed_bootstrap(vp, randomized)
    faster_fraction = float(
        np.mean([bool(row["randomized_faster"]) for row in quality_rows])
    )

    def method_gate(
        rows: Sequence[Mapping[str, Any]], deltas: np.ndarray
    ) -> dict[str, Any]:
        row_support_mean = float(
            np.mean([row["support"]["row_fraction"] for row in rows])
        )
        split_support = all(bool(row["support_qualified"]) for row in rows)
        mean_row_support = bool(
            config.minimum_mean_row_support_fraction
            <= row_support_mean
            <= config.maximum_mean_row_support_fraction
        )
        return {
            "normalized_kkt_all_splits": all(
                bool(row["normalized_kkt_qualified"]) for row in rows
            ),
            "normalized_kkt_qualified_count": int(
                sum(bool(row["normalized_kkt_qualified"]) for row in rows)
            ),
            "support_per_split_bounds_all_splits": split_support,
            "row_support_fraction_mean": row_support_mean,
            "row_support_mean_in_locked_range": mean_row_support,
            "support_qualified": bool(split_support and mean_row_support),
            "native_ba_mean_delta_vs_pca": float(np.mean(deltas)),
            "native_ba_worst_delta_vs_pca": float(np.min(deltas)),
            "native_ba_noninferior": bool(
                np.mean(deltas) >= -MEAN_BA_NONINFERIORITY_MARGIN
                and np.min(deltas) >= -WORST_BA_NONINFERIORITY_MARGIN
            ),
        }

    vp_gate = method_gate(vp, vp_delta)
    randomized_gate = method_gate(randomized, randomized_delta)
    vp_passed = bool(
        vp_gate["normalized_kkt_all_splits"]
        and vp_gate["support_qualified"]
        and vp_gate["native_ba_noninferior"]
    )
    randomized_quality = all(
        bool(row["matched_objective_and_eta"]) for row in quality_rows
    )
    speed_gate = bool(
        float(speed_bootstrap["ci_lower"]) >= RANDOMIZED_SPEEDUP_MIN
        and faster_fraction >= RANDOMIZED_FASTER_FRACTION_MIN
    )
    randomized_passed = bool(
        randomized_gate["normalized_kkt_all_splits"]
        and randomized_gate["support_qualified"]
        and randomized_gate["native_ba_noninferior"]
        and randomized_quality
        and speed_gate
    )
    return {
        "vp": {**vp_gate, "passed": vp_passed},
        "randomized": {
            **randomized_gate,
            "matched_objective_and_eta_all_splits": randomized_quality,
            "solver_speedup_median": float(np.median(speedups)),
            "paired_solver_speed_bootstrap": speed_bootstrap,
            "faster_fraction": faster_fraction,
            "solver_speed_gate_passed": speed_gate,
            "passed": randomized_passed,
        },
        "matched_quality": quality_rows,
        "confirmation_passed": bool(vp_passed and randomized_passed),
        "claim_boundary": (
            "scale_aware_vp_and_randomized_confirmation"
            if config.profile == "formal"
            else "smoke_only_no_scientific_claim"
        ),
    }


def _build_smoke_protocol(
    config: RunConfig,
    source_identity: Mapping[str, Any],
    input_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if config.profile != "smoke":
        raise ValueError("smoke protocol requires the smoke profile")
    validate_source_identity(source_identity)
    unsigned = {
        "schema_version": "genomics-scale-aware-vp-smoke-protocol-v1",
        "status": "synthetic_smoke_only_not_scientific_evidence",
        "protocol": _protocol(config),
        "source_identity_sha256": source_identity["aggregate_sha256"],
        "input_identity": dict(input_identity),
        "formal_seeds_executed": [],
    }
    return {**unsigned, "protocol_payload_sha256": canonical_sha256(unsigned)}


def _build_development_protocol(
    config: RunConfig,
    source_identity: Mapping[str, Any],
    prepared_input_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not config.profile.startswith("development_"):
        raise ValueError("development protocol requires a development profile")
    validate_source_identity(source_identity)
    validate_formal_input_identity(prepared_input_identity)
    unsigned = {
        "schema_version": "genomics-scale-aware-vp-development-protocol-v1",
        "status": "real_post_hoc_development_non_confirmatory",
        "protocol": _protocol(config),
        "source_identity": dict(source_identity),
        "prepared_input_identity": dict(prepared_input_identity),
        "formal_seeds_executed": [],
    }
    return {**unsigned, "protocol_payload_sha256": canonical_sha256(unsigned)}


def _build_analysis_payload(
    *,
    config: RunConfig,
    data: grid.GenotypeInput,
    source_identity: Mapping[str, Any],
    protocol_receipt: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    arrays_bytes: bytes,
    split_receipts: Sequence[Mapping[str, Any]],
    execution_environment: Mapping[str, Any],
    execution_spend_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_source_identity(source_identity)
    summaries = summarize_records(records, config)
    gates = compute_gates(records, config)
    input_identity = _input_identity(data)
    validate_runtime_execution_contract(
        execution_environment,
        require_speed_qualification=config.profile != "smoke",
    )
    if config.profile == "formal":
        validate_formal_lock(protocol_receipt)
        receipt_sha256 = str(protocol_receipt["lock_payload_sha256"])
        prepared = dict(protocol_receipt["prepared_input_identity"])
        for key, value in input_identity.items():
            if prepared.get(key) != value:
                raise ValueError(f"formal lock input mismatch: {key}")
        input_identity = prepared
        if execution_spend_receipt is None:
            raise ValueError("formal analysis requires the pre-fit spend receipt")
        validate_execution_spend_receipt(execution_spend_receipt)
        semantics = _formal_execution_semantics(prepared, _protocol(config))
        if (
            execution_spend_receipt["semantic_tuple"] != semantics
            or execution_spend_receipt["execution_key_sha256"]
            != canonical_sha256(semantics)
            or execution_spend_receipt["protocol_lock_payload_sha256"]
            != protocol_receipt["lock_payload_sha256"]
        ):
            raise ValueError("formal spend receipt does not match lock semantics")
    elif config.profile.startswith("development_"):
        expected_development = _build_development_protocol(
            config,
            source_identity,
            protocol_receipt["prepared_input_identity"],
        )
        if protocol_receipt != expected_development:
            raise ValueError("development protocol receipt mismatch")
        receipt_sha256 = str(protocol_receipt["protocol_payload_sha256"])
        input_identity = dict(protocol_receipt["prepared_input_identity"])
        if execution_spend_receipt is not None:
            raise ValueError("development cannot carry a formal spend receipt")
    else:
        if protocol_receipt != _build_smoke_protocol(
            config, source_identity, input_identity
        ):
            raise ValueError("smoke protocol receipt mismatch")
        receipt_sha256 = str(protocol_receipt["protocol_payload_sha256"])
        if execution_spend_receipt is not None:
            raise ValueError("smoke cannot carry a formal spend receipt")
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "profile": config.profile,
        "status": (
            "formal_complete_qualified"
            if config.profile == "formal" and gates["confirmation_passed"]
            else (
                "formal_complete_negative"
                if config.profile == "formal"
                else (
                    "development_complete_post_hoc_non_evidence"
                    if config.profile.startswith("development_")
                    else "smoke_complete_non_evidence"
                )
            )
        ),
        "scientific_claim_allowed": bool(
            config.profile == "formal" and gates["confirmation_passed"]
        ),
        "protocol": _protocol(config),
        "protocol_receipt_payload_sha256": receipt_sha256,
        "development_package_identity": (
            dict(protocol_receipt["development_package_identity"])
            if config.profile == "formal"
            else None
        ),
        "source_identity": dict(source_identity),
        "input_identity": input_identity,
        "runtime_versions": runtime_versions(),
        "execution_environment": dict(execution_environment),
        "records": [dict(record) for record in records],
        "split_receipts": [dict(receipt) for receipt in split_receipts],
        "summaries": summaries,
        "gates": gates,
        "predictions_records_sha256": canonical_sha256(list(predictions)),
        "solver_arrays_sha256": _bytes_sha256(arrays_bytes),
        "formal_fresh_seeds_executed": (
            list(config.seeds) if config.profile == "formal" else []
        ),
        "smoke_seed_schedule": (
            list(config.seeds) if config.profile == "smoke" else []
        ),
        "development_seed_schedule": (
            list(config.seeds) if config.profile.startswith("development_") else []
        ),
        "formal_execution_spend_receipt_sha256": (
            canonical_sha256(execution_spend_receipt)
            if execution_spend_receipt is not None
            else None
        ),
        "fed_robust_or_fedrpca_called": False,
    }
    return {**unsigned, "analysis_sha256": canonical_sha256(unsigned)}


def _predictions_payload(
    config: RunConfig, predictions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [dict(row) for row in predictions]
    unsigned = {
        "schema_version": "genomics-scale-aware-vp-predictions-v1",
        "profile": config.profile,
        "records": rows,
        "records_sha256": canonical_sha256(rows),
    }
    return {**unsigned, "payload_sha256": canonical_sha256(unsigned)}


def _package_artifact_filenames(profile: str) -> tuple[str, ...]:
    if profile == "formal":
        protocol_filename = LOCK_COPY_FILENAME
    elif profile.startswith("development_"):
        protocol_filename = DEVELOPMENT_PROTOCOL_FILENAME
    else:
        protocol_filename = SMOKE_PROTOCOL_FILENAME
    files = (
        ANALYSIS_FILENAME,
        PREDICTIONS_FILENAME,
        ARRAYS_FILENAME,
        protocol_filename,
    )
    if profile == "formal":
        return (*files, EXECUTION_SPEND_RECEIPT_FILENAME)
    return files


def write_output_package(
    *,
    output_directory: Path,
    config: RunConfig,
    data: grid.GenotypeInput,
    repo_root: Path,
    protocol_receipt: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    split_receipts: Sequence[Mapping[str, Any]],
    execution_environment: Mapping[str, Any] | None = None,
    execution_spend_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write an exclusive package; the completion manifest is published last."""

    output = Path(output_directory)
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("output parent must be a real directory")
    source_identity = collect_source_identity(repo_root)
    if config.profile == "formal" and source_identity != protocol_receipt.get(
        "source_identity"
    ):
        raise ValueError("formal execution source differs from the locked closure")
    arrays_bytes = _deterministic_npz_bytes(arrays)
    environment = (
        runtime_execution_contract()
        if execution_environment is None
        else dict(execution_environment)
    )
    analysis = _build_analysis_payload(
        config=config,
        data=data,
        source_identity=source_identity,
        protocol_receipt=protocol_receipt,
        records=records,
        predictions=predictions,
        arrays_bytes=arrays_bytes,
        split_receipts=split_receipts,
        execution_environment=environment,
        execution_spend_receipt=execution_spend_receipt,
    )
    prediction_payload = _predictions_payload(config, predictions)
    payloads = {
        ANALYSIS_FILENAME: _strict_json_bytes(analysis),
        PREDICTIONS_FILENAME: _strict_json_bytes(prediction_payload),
        ARRAYS_FILENAME: arrays_bytes,
        (
            LOCK_COPY_FILENAME
            if config.profile == "formal"
            else (
                DEVELOPMENT_PROTOCOL_FILENAME
                if config.profile.startswith("development_")
                else SMOKE_PROTOCOL_FILENAME
            )
        ): _strict_json_bytes(protocol_receipt),
    }
    if config.profile == "formal":
        if execution_spend_receipt is None:
            raise ValueError("formal package requires execution spend receipt")
        payloads[EXECUTION_SPEND_RECEIPT_FILENAME] = _strict_json_bytes(
            execution_spend_receipt
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for filename in _package_artifact_filenames(config.profile):
            _write_new_file(temporary / filename, payloads[filename])
        artifact_receipts = {
            filename: {
                "bytes": len(payloads[filename]),
                "sha256": _bytes_sha256(payloads[filename]),
            }
            for filename in _package_artifact_filenames(config.profile)
        }
        manifest_unsigned = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "profile": config.profile,
            "manifest_written_last": True,
            "scientific_claim_allowed": analysis["scientific_claim_allowed"],
            "analysis_sha256": analysis["analysis_sha256"],
            "protocol_receipt_payload_sha256": analysis[
                "protocol_receipt_payload_sha256"
            ],
            "artifact_count_excluding_manifest": len(artifact_receipts),
            "exact_package_file_count": len(artifact_receipts) + 1,
            "artifacts": artifact_receipts,
            "package_anchor_sha256": canonical_sha256(
                {
                    "analysis_sha256": analysis["analysis_sha256"],
                    "artifacts": artifact_receipts,
                }
            ),
        }
        manifest = {
            **manifest_unsigned,
            "manifest_payload_sha256": canonical_sha256(manifest_unsigned),
        }
        _write_new_file(temporary / MANIFEST_FILENAME, _strict_json_bytes(manifest))
        temporary.rename(output)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    if config.profile == "formal":
        return {
            "profile": "formal",
            "analysis_sha256": analysis["analysis_sha256"],
            "package_anchor_sha256": manifest["package_anchor_sha256"],
            "output_manifest_file_sha256": _file_sha256(
                output / MANIFEST_FILENAME
            ),
            "strict_external_validation_pending": True,
        }
    return validate_output_package(output, repo_root=repo_root)


_ANALYSIS_KEYS = {
    "schema_version",
    "profile",
    "status",
    "scientific_claim_allowed",
    "protocol",
    "protocol_receipt_payload_sha256",
    "development_package_identity",
    "source_identity",
    "input_identity",
    "runtime_versions",
    "execution_environment",
    "records",
    "split_receipts",
    "summaries",
    "gates",
    "predictions_records_sha256",
    "solver_arrays_sha256",
    "formal_fresh_seeds_executed",
    "smoke_seed_schedule",
    "development_seed_schedule",
    "formal_execution_spend_receipt_sha256",
    "fed_robust_or_fedrpca_called",
    "analysis_sha256",
}

_RECORD_KEYS = {
    "seed",
    "method",
    "rank",
    "panel_size",
    "rho",
    "sigma_max",
    "lipschitz_sigma_max_squared",
    "reference_step_size",
    "alpha",
    "rho_recomputed",
    "loading_matrix_sha256",
    "procrustes_factor_sha256",
    "common_objective",
    "common_stationarity",
    "native_stationarity",
    "normalized_kkt_qualified",
    "support",
    "support_qualified",
    "native",
    "common_pca_probe",
    "runtime_seconds",
    "deterministic_output_sha256",
    "deterministic_repetitions_identical",
    "panel_selected_by_train_only_pca_loadings",
    "same_panel_for_all_methods_within_split",
    "test_used_for_preprocessing_screen_or_solver",
    "native_status",
    "native_raw_converged",
    "native_iterations",
    "sketch_iterations",
    "full_refinement_iterations",
    "randomized_seed",
    "sketch_residual_relative",
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "retained_original_indices_sha256",
    "selected_retained_indices_sha256",
    "selected_original_indices_sha256",
    "selected_snp_ids_sha256",
    "panel_train_sha256",
    "panel_test_sha256",
    "preprocessor_mean_sha256",
    "preprocessor_scale_sha256",
}

_SPLIT_KEYS = {
    "seed",
    "train_indices",
    "test_indices",
    "retained_snps",
    "selected_original_indices",
    "selected_snp_ids",
    "ordered_original_ranking_sha256",
    "loading_importance_sha256",
    "full_width_pca_loadings_sha256",
    "population_labels_used_for_screen",
    "test_rows_used_for_screen",
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "retained_original_indices_sha256",
    "selected_retained_indices_sha256",
    "selected_original_indices_sha256",
    "selected_snp_ids_sha256",
    "panel_train_sha256",
    "panel_test_sha256",
    "preprocessor_mean_sha256",
    "preprocessor_scale_sha256",
}


def _validate_digest_field(payload: Mapping[str, Any], field: str) -> None:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError(f"{field} mismatch")


def _validate_metric_block(block: Any) -> None:
    if not isinstance(block, Mapping) or set(block) != {
        "metrics",
        "predictions_sha256",
        "centroids_sha256",
    }:
        raise ValueError("metric block schema mismatch")
    if not _is_sha256(block["predictions_sha256"]) or not _is_sha256(
        block["centroids_sha256"]
    ):
        raise ValueError("metric block digest invalid")
    metrics = block["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "class_order",
        "confusion_matrix_counts",
        "per_population_metrics",
    }:
        raise ValueError("classification metric schema mismatch")
    if metrics["class_order"] != list(EXPECTED_POPULATIONS):
        raise ValueError("classification class order mismatch")
    confusion = np.asarray(metrics["confusion_matrix_counts"])
    if (
        confusion.shape != (len(EXPECTED_POPULATIONS),) * 2
        or not np.issubdtype(confusion.dtype, np.integer)
        or np.any(confusion < 0)
        or int(confusion.sum()) <= 0
    ):
        raise ValueError("classification confusion matrix invalid")
    per_population = metrics["per_population_metrics"]
    if not isinstance(per_population, list) or len(per_population) != len(
        EXPECTED_POPULATIONS
    ):
        raise ValueError("per-population metrics invalid")
    for population, item in zip(EXPECTED_POPULATIONS, per_population, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "population",
            "support",
            "predicted_count",
            "true_positive",
            "precision",
            "recall",
            "f1",
        }:
            raise ValueError("per-population metric schema mismatch")
        if item["population"] != population:
            raise ValueError("per-population metric order mismatch")
    for name in ("accuracy", "balanced_accuracy", "macro_f1"):
        value = metrics[name]
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"invalid classification metric: {name}")


def _validate_record(record: Any, config: RunConfig) -> None:
    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise ValueError("analysis record schema mismatch")
    method = record["method"]
    if (
        method not in METHODS
        or isinstance(record["seed"], bool)
        or not isinstance(record["seed"], int)
        or record["seed"] not in config.seeds
        or isinstance(record["rank"], bool)
        or not isinstance(record["rank"], int)
        or isinstance(record["panel_size"], bool)
        or not isinstance(record["panel_size"], int)
        or record["rank"] != config.rank
        or record["panel_size"] != config.panel_size
        or record["rho"] != config.rho
    ):
        raise ValueError("analysis record identity mismatch")
    for field in (
        "loading_matrix_sha256",
        "procrustes_factor_sha256",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "retained_original_indices_sha256",
        "selected_retained_indices_sha256",
        "selected_original_indices_sha256",
        "panel_train_sha256",
        "panel_test_sha256",
        "preprocessor_mean_sha256",
        "preprocessor_scale_sha256",
        "deterministic_output_sha256",
    ):
        if not _is_sha256(record[field]):
            raise ValueError(f"invalid record digest: {field}")
    lipschitz = float(record["lipschitz_sigma_max_squared"])
    alpha = float(record["alpha"])
    step = float(record["reference_step_size"])
    if (
        not math.isfinite(lipschitz)
        or lipschitz <= 0.0
        or not math.isclose(float(record["sigma_max"]) ** 2, lipschitz, rel_tol=2e-14)
        or not math.isclose(step, 1.0 / lipschitz, rel_tol=2e-14)
        or not math.isclose(alpha, config.rho * lipschitz, rel_tol=2e-14)
        or not math.isclose(float(record["rho_recomputed"]), config.rho, rel_tol=2e-14)
    ):
        raise ValueError("record scale-aware alpha/step identity mismatch")
    common = record["common_stationarity"]
    if not isinstance(common, Mapping) or set(common) != {
        "raw_frobenius",
        "kkt_rms",
        "eta",
    }:
        raise ValueError("common stationarity schema mismatch")
    expected = _normalized_stationarity(
        float(common["raw_frobenius"]), alpha, config.panel_size * config.rank
    )
    if any(
        not math.isclose(float(common[key]), value, rel_tol=2e-14)
        for key, value in expected.items()
    ):
        raise ValueError("common stationarity normalization mismatch")
    support = record["support"]
    if not isinstance(support, Mapping) or set(support) != {
        "threshold",
        "entry_count",
        "entry_fraction",
        "row_count",
        "row_fraction",
        "component_counts",
        "matrix_rank",
    }:
        raise ValueError("support receipt schema mismatch")
    if (
        support["threshold"] != config.support_threshold
        or any(
            isinstance(support[field], bool) or not isinstance(support[field], int)
            for field in ("entry_count", "row_count", "matrix_rank")
        )
        or not isinstance(support["component_counts"], list)
        or len(support["component_counts"]) != config.rank
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in support["component_counts"]
        )
        or sum(support["component_counts"]) != support["entry_count"]
        or not math.isclose(
            float(support["entry_fraction"]),
            int(support["entry_count"]) / (config.panel_size * config.rank),
            rel_tol=1e-14,
        )
        or not math.isclose(
            float(support["row_fraction"]),
            int(support["row_count"]) / config.panel_size,
            rel_tol=1e-14,
        )
    ):
        raise ValueError("support receipt arithmetic mismatch")
    _validate_metric_block(record["native"])
    _validate_metric_block(record["common_pca_probe"])
    if (
        not isinstance(record["native_status"], str)
        or not isinstance(record["native_raw_converged"], bool)
        or isinstance(record["native_iterations"], bool)
        or not isinstance(record["native_iterations"], int)
        or not 1 <= record["native_iterations"] <= config.total_budget
    ):
        raise ValueError("native solver metadata invalid")
    if method == "pca":
        if any(
            record[field] is not None
            for field in (
                "sketch_iterations",
                "full_refinement_iterations",
                "randomized_seed",
                "sketch_residual_relative",
            )
        ):
            raise ValueError("PCA solver metadata contains sparse fields")
    elif method == "vp":
        if any(
            record[field] is not None
            for field in (
                "sketch_iterations",
                "full_refinement_iterations",
                "randomized_seed",
                "sketch_residual_relative",
            )
        ):
            raise ValueError("VP solver metadata contains randomized fields")
    else:
        if (
            isinstance(record["sketch_iterations"], bool)
            or not isinstance(record["sketch_iterations"], int)
            or isinstance(record["full_refinement_iterations"], bool)
            or not isinstance(record["full_refinement_iterations"], int)
            or record["sketch_iterations"] < 1
            or record["sketch_iterations"] > config.randomized_sketch_budget
            or record["full_refinement_iterations"] < 1
            or record["full_refinement_iterations"] > config.randomized_full_budget
            or record["native_iterations"]
            != record["sketch_iterations"] + record["full_refinement_iterations"]
            or record["randomized_seed"] != record["seed"] + 730_000
            or not math.isfinite(float(record["sketch_residual_relative"]))
            or float(record["sketch_residual_relative"]) < 0.0
        ):
            raise ValueError("randomized solver metadata invalid")
    runtime = record["runtime_seconds"]
    runtime_keys = {
        "shared_train_only_screen",
        "common_pca_reference",
        "common_pca_reference_charged_to_end_to_end",
        "solver_only_repetitions",
        "solver_only_median",
        "solver_only_scope",
        "standalone_exact_pca_initialization_seconds",
        "timing_positions",
        "warmup_repetitions",
        "timed_repetitions",
        "common_diagnostic_and_scoring",
        "incremental_after_shared_screen",
        "end_to_end",
    }
    if not isinstance(runtime, Mapping) or set(runtime) != runtime_keys:
        raise ValueError("runtime receipt schema mismatch")
    repetitions = runtime["solver_only_repetitions"]
    expected_repetitions = 1 if method == "pca" else config.timed_repetitions
    if method == "pca":
        expected_positions = [0]
        expected_scope = "closed_form_panel_pca"
    else:
        base_order = (
            ("vp", "randomized")
            if record["seed"] % 2
            else (
                "randomized",
                "vp",
            )
        )
        expected_positions = []
        for repetition in range(config.timed_repetitions):
            order = base_order if repetition % 2 == 0 else tuple(reversed(base_order))
            expected_positions.append(order.index(method))
        expected_scope = (
            "proximal_iterations_from_frozen_exact_pca_initialization_excluding_pca_svd"
            if method == "vp"
            else "randomized_workspace_plus_sketch_and_full_refinement"
        )
    if (
        not isinstance(repetitions, list)
        or len(repetitions) != expected_repetitions
        or any(
            not math.isfinite(float(value)) or float(value) <= 0
            for value in repetitions
        )
        or not math.isclose(
            float(runtime["solver_only_median"]),
            float(np.median(np.asarray(repetitions, dtype=float))),
            rel_tol=1e-14,
        )
        or runtime["warmup_repetitions"] != config.warmup_repetitions
        or runtime["timed_repetitions"] != expected_repetitions
        or isinstance(runtime["warmup_repetitions"], bool)
        or not isinstance(runtime["warmup_repetitions"], int)
        or isinstance(runtime["timed_repetitions"], bool)
        or not isinstance(runtime["timed_repetitions"], int)
        or runtime["timing_positions"] != expected_positions
        or runtime["solver_only_scope"] != expected_scope
        or runtime["standalone_exact_pca_initialization_seconds"]
        != runtime["common_pca_reference"]
        or (
            method == "pca"
            and (
                runtime["common_pca_reference_charged_to_end_to_end"] != 0.0
                or runtime["solver_only_median"] != runtime["common_pca_reference"]
            )
        )
        or (
            method != "pca"
            and runtime["common_pca_reference_charged_to_end_to_end"]
            != runtime["common_pca_reference"]
        )
    ):
        raise ValueError("runtime repetition receipt mismatch")
    incremental = (
        float(runtime["common_pca_reference_charged_to_end_to_end"])
        + float(runtime["solver_only_median"])
        + float(runtime["common_diagnostic_and_scoring"])
    )
    if not math.isclose(
        float(runtime["incremental_after_shared_screen"]), incremental, rel_tol=2e-14
    ) or not math.isclose(
        float(runtime["end_to_end"]),
        float(runtime["shared_train_only_screen"]) + incremental,
        rel_tol=2e-14,
    ):
        raise ValueError("runtime incremental/end-to-end arithmetic mismatch")
    if method == "pca":
        if (
            record["native_stationarity"] is not None
            or record["normalized_kkt_qualified"] is not None
            or record["support_qualified"] is not None
            or record["native_status"] != "closed_form_pca"
        ):
            raise ValueError("PCA qualification semantics mismatch")
    else:
        native = record["native_stationarity"]
        if not isinstance(native, Mapping) or set(native) != {
            "last_window_count",
            "last_raw_frobenius",
            "last_kkt_rms",
            "last_eta",
            "last_raw_sha256",
            "last_kkt_rms_sha256",
            "last_eta_sha256",
            "final_raw_frobenius",
            "final_kkt_rms",
            "final_eta",
            "last_window_max_eta",
            "last_window_all_qualified",
        }:
            raise ValueError("native stationarity schema mismatch")
        if (
            isinstance(native["last_window_count"], bool)
            or not isinstance(native["last_window_count"], int)
            or native["last_window_count"] != config.native_last_window
        ):
            raise ValueError("native stationarity window mismatch")
        if any(
            not isinstance(native[field], list)
            or len(native[field]) != config.native_last_window
            or any(not math.isfinite(float(value)) for value in native[field])
            for field in ("last_raw_frobenius", "last_kkt_rms", "last_eta")
        ):
            raise ValueError("native stationarity values are invalid")
        for field in ("last_raw_sha256", "last_kkt_rms_sha256", "last_eta_sha256"):
            if not _is_sha256(native[field]):
                raise ValueError("native stationarity trace digest invalid")
        expected_qualified = bool(
            native["last_window_all_qualified"]
            and float(common["eta"]) <= config.normalized_kkt_tolerance
        )
        expected_support = bool(
            support["matrix_rank"] == config.rank
            and min(support["component_counts"]) >= config.minimum_component_support
            and support["entry_fraction"] <= config.maximum_entry_support_fraction
            and config.minimum_split_row_support_fraction
            <= support["row_fraction"]
            <= config.maximum_split_row_support_fraction
        )
        if (
            record["normalized_kkt_qualified"] is not expected_qualified
            or record["support_qualified"] is not expected_support
        ):
            raise ValueError("sparse qualification flag mismatch")
    if (
        record["deterministic_repetitions_identical"] is not True
        or record["panel_selected_by_train_only_pca_loadings"] is not True
        or record["same_panel_for_all_methods_within_split"] is not True
        or record["test_used_for_preprocessing_screen_or_solver"] is not False
    ):
        raise ValueError("record leakage/determinism invariant mismatch")


def _validate_split_receipts(
    receipts: Any, config: RunConfig, arrays: Mapping[str, np.ndarray]
) -> dict[int, Mapping[str, Any]]:
    if not isinstance(receipts, list) or len(receipts) != len(config.seeds):
        raise ValueError("split receipt count mismatch")
    by_seed: dict[int, Mapping[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, Mapping) or set(receipt) != _SPLIT_KEYS:
            raise ValueError("split receipt schema mismatch")
        seed = receipt["seed"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in config.seeds
            or seed in by_seed
        ):
            raise ValueError("split receipt seed mismatch")
        if (
            receipt["population_labels_used_for_screen"] is not False
            or receipt["test_rows_used_for_screen"] is not False
            or isinstance(receipt["retained_snps"], bool)
            or not isinstance(receipt["retained_snps"], int)
            or receipt["retained_snps"] < config.panel_size
            or any(
                not isinstance(values, list)
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in values
                )
                for values in (receipt["train_indices"], receipt["test_indices"])
            )
            or len(receipt["selected_original_indices"]) != config.panel_size
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in receipt["selected_original_indices"]
            )
            or not isinstance(receipt["selected_snp_ids"], list)
            or any(not isinstance(value, str) for value in receipt["selected_snp_ids"])
        ):
            raise ValueError("split screen policy mismatch")
        prefix = f"seed_{seed}"
        train = np.asarray(arrays[f"{prefix}_train_indices"], dtype=np.int64)
        test = np.asarray(arrays[f"{prefix}_test_indices"], dtype=np.int64)
        retained_original = np.asarray(
            arrays[f"{prefix}_retained_original_indices"], dtype=np.int64
        )
        full_loadings = np.asarray(
            arrays[f"{prefix}_full_width_pca_loadings"], dtype=float
        )
        stored_importance = np.asarray(
            arrays[f"{prefix}_loading_importance"], dtype=float
        )
        selected_retained = np.asarray(
            arrays[f"{prefix}_selected_retained_indices"], dtype=np.int64
        )
        selected_original = np.asarray(
            arrays[f"{prefix}_selected_original_indices"], dtype=np.int64
        )
        selected_snp_ids = np.asarray(arrays[f"{prefix}_selected_snp_ids"], dtype=str)
        ranking, recomputed_importance = grid.rank_by_pca_loadings(
            full_loadings, config.rank
        )
        expected_selected_retained = np.sort(ranking[: config.panel_size])
        expected_selected_original = np.sort(
            retained_original[expected_selected_retained]
        )
        if (
            train.tolist() != receipt["train_indices"]
            or test.tolist() != receipt["test_indices"]
            or selected_original.tolist() != receipt["selected_original_indices"]
            or selected_snp_ids.tolist() != receipt["selected_snp_ids"]
            or retained_original.shape != (receipt["retained_snps"],)
            or full_loadings.shape != (retained_original.size, config.rank)
            or stored_importance.shape != (retained_original.size,)
            or np.any(np.diff(retained_original) <= 0)
            or not np.array_equal(stored_importance, recomputed_importance)
            or not np.array_equal(selected_retained, expected_selected_retained)
            or not np.array_equal(selected_original, expected_selected_original)
            or not np.allclose(
                full_loadings.T @ full_loadings,
                np.eye(config.rank),
                atol=2e-7,
            )
            or np.unique(train).size != train.size
            or np.unique(test).size != test.size
            or np.intersect1d(train, test).size
            or selected_retained.shape != (config.panel_size,)
            or selected_original.shape != (config.panel_size,)
            or selected_snp_ids.shape != (config.panel_size,)
            or np.unique(selected_snp_ids).size != config.panel_size
            or np.any(np.diff(selected_retained) <= 0)
            or np.any(np.diff(selected_original) <= 0)
        ):
            raise ValueError("split receipt arrays mismatch")
        for field, array in (
            ("train_indices_sha256", train),
            ("test_indices_sha256", test),
            ("retained_original_indices_sha256", retained_original),
            ("selected_retained_indices_sha256", selected_retained),
            ("selected_original_indices_sha256", selected_original),
            ("selected_snp_ids_sha256", selected_snp_ids),
        ):
            if receipt[field] != array_sha256(array):
                raise ValueError(f"split array digest mismatch: {field}")
        if (
            receipt["ordered_original_ranking_sha256"]
            != array_sha256(retained_original[ranking])
            or receipt["loading_importance_sha256"] != array_sha256(stored_importance)
            or receipt["full_width_pca_loadings_sha256"] != array_sha256(full_loadings)
        ):
            raise ValueError("train-only PCA screen receipt mismatch")
        for field in _SPLIT_KEYS - {
            "seed",
            "train_indices",
            "test_indices",
            "retained_snps",
            "selected_original_indices",
            "selected_snp_ids",
            "population_labels_used_for_screen",
            "test_rows_used_for_screen",
        }:
            if not _is_sha256(receipt[field]):
                raise ValueError(f"split digest invalid: {field}")
        by_seed[int(seed)] = receipt
    if list(by_seed) != list(config.seeds):
        raise ValueError("split receipts must follow the locked seed order")
    return by_seed


def _expected_array_keys(config: RunConfig) -> set[str]:
    expected: set[str] = set()
    for seed in config.seeds:
        prefix = f"seed_{seed}"
        expected.update(
            {
                f"{prefix}_train_indices",
                f"{prefix}_test_indices",
                f"{prefix}_retained_original_indices",
                f"{prefix}_full_width_pca_loadings",
                f"{prefix}_loading_importance",
                f"{prefix}_selected_retained_indices",
                f"{prefix}_selected_original_indices",
                f"{prefix}_selected_snp_ids",
                f"{prefix}_pca_loadings",
                f"{prefix}_vp_loadings",
                f"{prefix}_randomized_loadings",
                f"{prefix}_vp_native_last_raw",
                f"{prefix}_randomized_native_last_raw",
            }
        )
        for method in METHODS:
            expected.update(
                {
                    f"{prefix}_{method}_procrustes_factor",
                    f"{prefix}_{method}_native_centroids",
                    f"{prefix}_{method}_common_pca_probe_centroids",
                }
            )
    return expected


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    config: RunConfig,
) -> None:
    if set(arrays) != _expected_array_keys(config):
        raise ValueError("solver NPZ array closure mismatch")
    index = {(int(row["seed"]), str(row["method"])): row for row in records}
    for name, array in arrays.items():
        values = np.asarray(array)
        if values.dtype.hasobject or (
            np.issubdtype(values.dtype, np.number) and not np.isfinite(values).all()
        ):
            raise ValueError(f"invalid solver array: {name}")
    for seed in config.seeds:
        prefix = f"seed_{seed}"
        for method in METHODS:
            record = index[(seed, method)]
            loadings = np.asarray(
                arrays[
                    f"{prefix}_{'randomized' if method == 'randomized' else method}_loadings"
                ],
                dtype=float,
            )
            factor = np.asarray(
                arrays[f"{prefix}_{method}_procrustes_factor"], dtype=float
            )
            native_centroids = np.asarray(
                arrays[f"{prefix}_{method}_native_centroids"], dtype=float
            )
            common_centroids = np.asarray(
                arrays[f"{prefix}_{method}_common_pca_probe_centroids"], dtype=float
            )
            if (
                loadings.shape != (config.panel_size, config.rank)
                or factor.shape != (config.panel_size, config.rank)
                or native_centroids.shape != (len(EXPECTED_POPULATIONS), config.rank)
                or common_centroids.shape != (len(EXPECTED_POPULATIONS), config.rank)
                or array_sha256(loadings) != record["loading_matrix_sha256"]
                or array_sha256(factor) != record["procrustes_factor_sha256"]
                or array_sha256(native_centroids)
                != record["native"]["centroids_sha256"]
                or array_sha256(common_centroids)
                != record["common_pca_probe"]["centroids_sha256"]
            ):
                raise ValueError("solver loading/factor/centroid array mismatch")
            if method != "pca":
                trace = np.asarray(
                    arrays[f"{prefix}_{method}_native_last_raw"], dtype=float
                )
                native = record["native_stationarity"]
                scale = math.sqrt(config.panel_size * config.rank)
                rms = trace / scale
                eta = trace / (float(record["alpha"]) * scale)
                if (
                    trace.shape != (config.native_last_window,)
                    or trace.tolist() != native["last_raw_frobenius"]
                    or rms.tolist() != native["last_kkt_rms"]
                    or eta.tolist() != native["last_eta"]
                    or array_sha256(trace) != native["last_raw_sha256"]
                    or array_sha256(rms) != native["last_kkt_rms_sha256"]
                    or array_sha256(eta) != native["last_eta_sha256"]
                    or trace[-1] != native["final_raw_frobenius"]
                    or rms[-1] != native["final_kkt_rms"]
                    or eta[-1] != native["final_eta"]
                    or float(np.max(eta)) != native["last_window_max_eta"]
                    or bool(np.all(eta <= config.normalized_kkt_tolerance))
                    is not native["last_window_all_qualified"]
                ):
                    raise ValueError("native stationarity trace mismatch")


def _validate_predictions(
    payload: Any,
    analysis: Mapping[str, Any],
    split_by_seed: Mapping[int, Mapping[str, Any]],
    config: RunConfig,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "profile",
        "records",
        "records_sha256",
        "payload_sha256",
    }:
        raise ValueError("predictions payload schema mismatch")
    _validate_digest_field(payload, "payload_sha256")
    rows = payload["records"]
    if (
        payload["schema_version"] != "genomics-scale-aware-vp-predictions-v1"
        or payload["profile"] != config.profile
        or not isinstance(rows, list)
        or payload["records_sha256"] != canonical_sha256(rows)
        or analysis["predictions_records_sha256"] != canonical_sha256(rows)
    ):
        raise ValueError("predictions payload identity mismatch")
    expected_count = sum(
        len(split_by_seed[seed]["test_indices"]) * len(METHODS) * len(VIEWS)
        for seed in config.seeds
    )
    if len(rows) != expected_count:
        raise ValueError("prediction row count mismatch")
    order = {method: index for index, method in enumerate(METHODS)}
    view_order = {view: index for index, view in enumerate(VIEWS)}
    expected_sort: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, str, str, int]] = set()
    truth_by_sample: dict[tuple[int, int], str] = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "seed",
            "method",
            "view",
            "sample_index",
            "sample_id",
            "truth",
            "prediction",
        }:
            raise ValueError("prediction row schema mismatch")
        seed = row["seed"]
        method = row["method"]
        view = row["view"]
        sample_index = row["sample_index"]
        key = (seed, method, view, sample_index)
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed not in config.seeds
            or method not in METHODS
            or view not in VIEWS
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or sample_index not in split_by_seed[seed]["test_indices"]
            or row["truth"] not in EXPECTED_POPULATIONS
            or row["prediction"] not in EXPECTED_POPULATIONS
            or not isinstance(row["sample_id"], str)
            or key in seen
        ):
            raise ValueError("prediction row identity mismatch")
        seen.add(key)
        truth_key = (seed, sample_index)
        if truth_key in truth_by_sample and truth_by_sample[truth_key] != row["truth"]:
            raise ValueError("prediction truth changed across methods/views")
        truth_by_sample[truth_key] = row["truth"]
        expected_sort.append((seed, order[method], view_order[view], sample_index))
    if expected_sort != sorted(expected_sort):
        raise ValueError("prediction rows are not canonically ordered")
    record_index = {
        (int(record["seed"]), str(record["method"])): record
        for record in analysis["records"]
    }
    for seed in config.seeds:
        test_order = split_by_seed[seed]["test_indices"]
        for method in METHODS:
            for view in VIEWS:
                selected = {
                    int(row["sample_index"]): row
                    for row in rows
                    if row["seed"] == seed
                    and row["method"] == method
                    and row["view"] == view
                }
                ordered = [selected[index] for index in test_order]
                truth = np.asarray([row["truth"] for row in ordered], dtype=str)
                predicted = np.asarray(
                    [row["prediction"] for row in ordered], dtype=str
                )
                record = record_index[(seed, method)]
                block = record[view]
                if array_sha256(predicted) != block["predictions_sha256"]:
                    raise ValueError("prediction array digest mismatch")
                recomputed = grid._classification_metrics(
                    truth, predicted, np.asarray(EXPECTED_POPULATIONS, dtype=str)
                )
                if recomputed != block["metrics"]:
                    raise ValueError("classification metrics do not match predictions")
        common_blocks = [
            record_index[(seed, method)]["common_pca_probe"] for method in METHODS
        ]
        if any(block != common_blocks[0] for block in common_blocks[1:]):
            raise ValueError("common PCA probe changed across methods")
        if record_index[(seed, "pca")]["native"] != common_blocks[0]:
            raise ValueError("PCA native and common-probe evaluations disagree")


def validate_analysis_payload(
    analysis: Any,
    predictions_payload: Any,
    arrays: Mapping[str, np.ndarray],
    protocol_receipt: Mapping[str, Any],
    execution_spend_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(analysis, Mapping) or set(analysis) != _ANALYSIS_KEYS:
        raise ValueError("analysis payload schema mismatch")
    _validate_digest_field(analysis, "analysis_sha256")
    if analysis["schema_version"] != SCHEMA_VERSION:
        raise ValueError("analysis schema version mismatch")
    config = config_for_profile(str(analysis["profile"]))
    if analysis["protocol"] != _protocol(config):
        raise ValueError("analysis protocol differs from locked profile")
    if analysis["fed_robust_or_fedrpca_called"] is not False:
        raise ValueError("out-of-scope Fed/robust path was marked as called")
    source = analysis["source_identity"]
    if not isinstance(source, Mapping):
        raise ValueError("analysis source identity invalid")
    validate_source_identity(source)
    validate_runtime_execution_contract(
        analysis["execution_environment"],
        require_speed_qualification=config.profile != "smoke",
    )
    records = analysis["records"]
    if not isinstance(records, list) or len(records) != len(config.seeds) * len(
        METHODS
    ):
        raise ValueError("analysis record count mismatch")
    for record in records:
        _validate_record(record, config)
    expected_record_order = [
        (seed, method) for seed in config.seeds for method in METHODS
    ]
    if [(row["seed"], row["method"]) for row in records] != expected_record_order:
        raise ValueError("analysis records are not in locked order")
    for seed in config.seeds:
        seed_rows = [record for record in records if record["seed"] == seed]
        for field in (
            "shared_train_only_screen",
            "common_pca_reference",
            "standalone_exact_pca_initialization_seconds",
        ):
            if len({row["runtime_seconds"][field] for row in seed_rows}) != 1:
                raise ValueError(f"shared timing changed across methods: {field}")
    if config.profile == "formal":
        validate_formal_lock(protocol_receipt)
        if execution_spend_receipt is None:
            raise ValueError("formal validation requires execution spend receipt")
        validate_execution_spend_receipt(execution_spend_receipt)
        semantics = _formal_execution_semantics(
            protocol_receipt["prepared_input_identity"], _protocol(config)
        )
        if (
            analysis["protocol_receipt_payload_sha256"]
            != protocol_receipt["lock_payload_sha256"]
            or analysis["input_identity"] != protocol_receipt["prepared_input_identity"]
            or analysis["formal_fresh_seeds_executed"] != list(config.seeds)
            or analysis["smoke_seed_schedule"] != []
            or analysis["development_seed_schedule"] != []
            or analysis["development_package_identity"]
            != protocol_receipt["development_package_identity"]
            or analysis["formal_execution_spend_receipt_sha256"]
            != canonical_sha256(execution_spend_receipt)
            or execution_spend_receipt["semantic_tuple"] != semantics
            or execution_spend_receipt["execution_key_sha256"]
            != canonical_sha256(semantics)
            or execution_spend_receipt["protocol_lock_payload_sha256"]
            != protocol_receipt["lock_payload_sha256"]
        ):
            raise ValueError("formal analysis does not match its protocol lock")
    elif config.profile.startswith("development_"):
        expected_development = _build_development_protocol(
            config,
            source,
            analysis["input_identity"],
        )
        if (
            protocol_receipt != expected_development
            or analysis["protocol_receipt_payload_sha256"]
            != expected_development["protocol_payload_sha256"]
            or analysis["formal_fresh_seeds_executed"] != []
            or analysis["smoke_seed_schedule"] != []
            or analysis["development_seed_schedule"] != list(config.seeds)
            or analysis["development_package_identity"] is not None
            or analysis["formal_execution_spend_receipt_sha256"] is not None
            or execution_spend_receipt is not None
        ):
            raise ValueError("development analysis protocol receipt mismatch")
    else:
        expected_smoke = _build_smoke_protocol(
            config, source, analysis["input_identity"]
        )
        if (
            protocol_receipt != expected_smoke
            or analysis["protocol_receipt_payload_sha256"]
            != expected_smoke["protocol_payload_sha256"]
            or analysis["formal_fresh_seeds_executed"] != []
            or analysis["smoke_seed_schedule"] != list(config.seeds)
            or analysis["development_seed_schedule"] != []
            or analysis["development_package_identity"] is not None
            or analysis["formal_execution_spend_receipt_sha256"] is not None
            or execution_spend_receipt is not None
        ):
            raise ValueError("smoke analysis protocol receipt mismatch")
    split_by_seed = _validate_split_receipts(analysis["split_receipts"], config, arrays)
    split_hash_fields = _SPLIT_KEYS - {
        "seed",
        "train_indices",
        "test_indices",
        "retained_snps",
        "selected_original_indices",
        "selected_snp_ids",
        "ordered_original_ranking_sha256",
        "loading_importance_sha256",
        "full_width_pca_loadings_sha256",
        "population_labels_used_for_screen",
        "test_rows_used_for_screen",
    }
    for record in records:
        split = split_by_seed[int(record["seed"])]
        for field in split_hash_fields:
            if record[field] != split[field]:
                raise ValueError(f"record/split digest mismatch: {field}")
    _validate_arrays(arrays, records, config)
    _validate_predictions(predictions_payload, analysis, split_by_seed, config)
    if analysis["summaries"] != summarize_records(records, config):
        raise ValueError("analysis summaries are not reproducible from records")
    gates = compute_gates(records, config)
    if analysis["gates"] != gates:
        raise ValueError("analysis gates are not reproducible from records")
    expected_claim = bool(config.profile == "formal" and gates["confirmation_passed"])
    expected_status = (
        "formal_complete_qualified"
        if expected_claim
        else (
            "formal_complete_negative"
            if config.profile == "formal"
            else (
                "development_complete_post_hoc_non_evidence"
                if config.profile.startswith("development_")
                else "smoke_complete_non_evidence"
            )
        )
    )
    if (
        analysis["scientific_claim_allowed"] is not expected_claim
        or analysis["status"] != expected_status
    ):
        raise ValueError("analysis claim/status does not follow its gates")
    return {
        "profile": config.profile,
        "analysis_sha256": analysis["analysis_sha256"],
        "record_count": len(records),
        "prediction_count": len(predictions_payload["records"]),
        "confirmation_passed": bool(gates["confirmation_passed"]),
        "scientific_claim_allowed": expected_claim,
    }


def validate_output_package(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
    genotype_npz: Path | None = None,
    source_manifest: Path | None = None,
    original_lock_path: Path | None = None,
    expected_original_lock_sha256: str | None = None,
    expected_output_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("output package must be a real directory")
    manifest_path = output / MANIFEST_FILENAME
    _require_regular_file(manifest_path, role="completion manifest")
    manifest = _strict_json_load(manifest_path)
    manifest_keys = {
        "schema_version",
        "status",
        "profile",
        "manifest_written_last",
        "scientific_claim_allowed",
        "analysis_sha256",
        "protocol_receipt_payload_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != manifest_keys:
        raise ValueError("completion manifest schema mismatch")
    _validate_digest_field(manifest, "manifest_payload_sha256")
    profile = str(manifest["profile"])
    config = config_for_profile(profile)
    if profile == "formal":
        required = {
            "repo_root": repo_root,
            "genotype_npz": genotype_npz,
            "source_manifest": source_manifest,
            "original_lock_path": original_lock_path,
            "expected_original_lock_sha256": expected_original_lock_sha256,
            "expected_output_manifest_sha256": expected_output_manifest_sha256,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "formal validation requires original inputs/lock/repo/output anchor: "
                + ",".join(missing)
            )
        if (
            not _is_sha256(expected_output_manifest_sha256)
            or _file_sha256(manifest_path) != expected_output_manifest_sha256
        ):
            raise ValueError("formal output manifest external SHA-256 mismatch")
    artifacts = _package_artifact_filenames(profile)
    expected_files = set(artifacts) | {MANIFEST_FILENAME}
    actual_files = {path.name for path in output.iterdir()}
    if actual_files != expected_files:
        raise ValueError("output package has missing or extra files")
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["status"] != "complete"
        or manifest["manifest_written_last"] is not True
        or manifest["artifact_count_excluding_manifest"] != len(artifacts)
        or manifest["exact_package_file_count"] != len(expected_files)
        or set(manifest["artifacts"]) != set(artifacts)
    ):
        raise ValueError("completion manifest policy mismatch")
    for filename in artifacts:
        path = output / filename
        _require_regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"bytes", "sha256"}
            or receipt["bytes"] != path.stat().st_size
            or receipt["sha256"] != _file_sha256(path)
        ):
            raise ValueError(f"artifact receipt mismatch: {filename}")
    expected_package_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest["package_anchor_sha256"] != expected_package_anchor:
        raise ValueError("package anchor mismatch")
    analysis = _strict_json_load(output / ANALYSIS_FILENAME)
    predictions = _strict_json_load(output / PREDICTIONS_FILENAME)
    arrays_bytes = (output / ARRAYS_FILENAME).read_bytes()
    arrays = _load_npz_bytes(arrays_bytes)
    protocol_path = output / (
        LOCK_COPY_FILENAME
        if profile == "formal"
        else (
            DEVELOPMENT_PROTOCOL_FILENAME
            if profile.startswith("development_")
            else SMOKE_PROTOCOL_FILENAME
        )
    )
    protocol_receipt = _strict_json_load(protocol_path)
    execution_spend_receipt = (
        _strict_json_load(output / EXECUTION_SPEND_RECEIPT_FILENAME)
        if profile == "formal"
        else None
    )
    result = validate_analysis_payload(
        analysis,
        predictions,
        arrays,
        protocol_receipt,
        execution_spend_receipt,
    )
    if (
        analysis["analysis_sha256"] != manifest["analysis_sha256"]
        or analysis["scientific_claim_allowed"]
        is not manifest["scientific_claim_allowed"]
        or analysis["protocol_receipt_payload_sha256"]
        != manifest["protocol_receipt_payload_sha256"]
        or analysis["solver_arrays_sha256"] != _bytes_sha256(arrays_bytes)
    ):
        raise ValueError("analysis/manifest/array cross-anchor mismatch")
    if (expected_analysis_sha256 is None) != (expected_package_anchor_sha256 is None):
        raise ValueError("external validation requires both package anchors")
    if expected_analysis_sha256 is not None:
        if (
            not _is_sha256(expected_analysis_sha256)
            or not _is_sha256(expected_package_anchor_sha256)
            or analysis["analysis_sha256"] != expected_analysis_sha256
            or manifest["package_anchor_sha256"] != expected_package_anchor_sha256
        ):
            raise ValueError("external package anchor mismatch")
    if repo_root is not None:
        current_source = collect_source_identity(repo_root)
        if current_source != analysis["source_identity"]:
            raise ValueError("current repo differs from packaged import closure")
    recomputation_verified = False
    original_lock_verified = False
    registry_spend_verified = False
    if profile == "smoke":
        _recompute_scientific_outputs_from_original_input(
            data_path=None,
            source_manifest=None,
            expected_source_manifest_sha256=None,
            analysis=analysis,
            predictions_payload=predictions,
            arrays=arrays,
            protocol_receipt=protocol_receipt,
            data_override=make_smoke_data(),
        )
        recomputation_verified = True
    if profile == "formal":
        assert repo_root is not None
        assert genotype_npz is not None
        assert source_manifest is not None
        assert original_lock_path is not None
        assert expected_original_lock_sha256 is not None
        _require_regular_file(original_lock_path, role="original formal lock")
        if (
            not _is_sha256(expected_original_lock_sha256)
            or _file_sha256(original_lock_path) != expected_original_lock_sha256
            or original_lock_path.read_bytes() != protocol_path.read_bytes()
        ):
            raise ValueError("original formal lock file/SHA differs from package copy")
        original_lock_verified = True
        assert execution_spend_receipt is not None
        if (
            execution_spend_receipt["exclusive_output_directory"]
            != str(output.resolve())
            or execution_spend_receipt["registry_directory"]
            != protocol_receipt["global_execution_registry_directory"]
            or execution_spend_receipt["protocol_lock_file_sha256"]
            != expected_original_lock_sha256
        ):
            raise ValueError("formal spend receipt output/registry/lock mismatch")
        spent_path = Path(execution_spend_receipt["registry_directory"]) / (
            execution_spend_receipt["execution_key_sha256"] + ".spent.json"
        )
        _require_regular_file(spent_path, role="global formal execution spend receipt")
        if spent_path.read_bytes() != (
            output / EXECUTION_SPEND_RECEIPT_FILENAME
        ).read_bytes():
            raise ValueError("global spend registry differs from packaged receipt")
        registry_spend_verified = True
        _recompute_scientific_outputs_from_original_input(
            data_path=genotype_npz,
            source_manifest=source_manifest,
            expected_source_manifest_sha256=protocol_receipt[
                "prepared_input_identity"
            ]["expected_external_source_manifest_sha256"],
            analysis=analysis,
            predictions_payload=predictions,
            arrays=arrays,
            protocol_receipt=protocol_receipt,
        )
        recomputation_verified = True
    return {
        **result,
        "artifact_integrity_verified": True,
        "source_closure_verified": repo_root is not None,
        "external_anchors_verified": expected_analysis_sha256 is not None,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "package_file_count": len(expected_files),
        "original_input_recomputation_verified": recomputation_verified,
        "original_lock_file_verified": original_lock_verified,
        "global_execution_spend_verified": registry_spend_verified,
    }


def _load_anchored_formal_input(
    genotype_npz: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
) -> tuple[grid.GenotypeInput, dict[str, Any]]:
    data = grid.load_genotype_npz(genotype_npz)
    preparation_identity = grid.validate_external_source_manifest(
        source_manifest,
        genotype_npz,
        data,
        expected_manifest_sha256=expected_source_manifest_sha256,
    )
    return data, preparation_identity


def _column_sign_aligned_equal(
    observed: FloatArray, expected: FloatArray, *, atol: float = 2e-10
) -> bool:
    observed = np.asarray(observed, dtype=float)
    expected = np.asarray(expected, dtype=float)
    if observed.shape != expected.shape or observed.ndim != 2:
        return False
    aligned = observed.copy()
    for column in range(observed.shape[1]):
        if float(np.dot(aligned[:, column], expected[:, column])) < 0.0:
            aligned[:, column] *= -1.0
    return bool(np.allclose(aligned, expected, rtol=2e-10, atol=atol))


def _recompute_scientific_outputs_from_original_input(
    *,
    data_path: Path | None,
    source_manifest: Path | None,
    expected_source_manifest_sha256: str | None,
    analysis: Mapping[str, Any],
    predictions_payload: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    protocol_receipt: Mapping[str, Any],
    data_override: grid.GenotypeInput | None = None,
) -> None:
    """Independently rebuild every scientific field from the original input."""

    config = config_for_profile(str(analysis["profile"]))
    if data_override is None:
        if (
            data_path is None
            or source_manifest is None
            or expected_source_manifest_sha256 is None
        ):
            raise ValueError("original scientific input paths/anchor are required")
        data, preparation = _load_anchored_formal_input(
            data_path, source_manifest, expected_source_manifest_sha256
        )
        expected_input = {
            **_input_identity(data),
            **preparation,
            "population_design": "five_superpopulation",
            "input_scope": "legacy_chr20_22",
            "ld_pruned": True,
            "population_label_is_race": False,
        }
    else:
        if config.profile != "smoke":
            raise ValueError("data override is allowed only for deterministic smoke")
        data = data_override
        expected_input = _input_identity(data)
    if analysis["input_identity"] != expected_input:
        raise ValueError("original genotype/preparation identity differs from analysis")
    if config.profile == "formal" and protocol_receipt[
        "prepared_input_identity"
    ] != expected_input:
        raise ValueError("original genotype/preparation identity differs from lock")
    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    records = {
        (int(record["seed"]), str(record["method"])): record
        for record in analysis["records"]
    }
    splits = {int(split["seed"]): split for split in analysis["split_receipts"]}
    prediction_rows = predictions_payload["records"]
    for seed in config.seeds:
        prefix = f"seed_{seed}"
        train_indices, test_indices = stratified_split(labels, config.test_fraction, seed)
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        preprocessor = fit_smartpca_preprocessor(raw[train_indices])
        train_matrix = transform_smartpca(raw[train_indices], preprocessor)
        test_matrix = transform_smartpca(raw[test_indices], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        full_loadings, _ = grid.fit_top_pca(train_matrix, config.rank)
        ranking, importance = grid.rank_by_pca_loadings(full_loadings, config.rank)
        selected_retained = np.sort(ranking[: config.panel_size]).astype(np.int64)
        selected_original = np.sort(retained_original[selected_retained]).astype(
            np.int64
        )
        selected_snp_ids = np.asarray(data.snp_ids[selected_original], dtype=str)
        panel_train = np.ascontiguousarray(train_matrix[:, selected_retained])
        panel_test = np.ascontiguousarray(test_matrix[:, selected_retained])
        pca_loadings, singular_values = grid.fit_top_pca(panel_train, config.rank)
        lipschitz = float(singular_values[0] ** 2)
        step = 1.0 / lipschitz
        alpha = config.rho * lipschitz
        split = splits[seed]
        if (
            not np.array_equal(arrays[f"{prefix}_train_indices"], train_indices)
            or not np.array_equal(arrays[f"{prefix}_test_indices"], test_indices)
            or not np.array_equal(
                arrays[f"{prefix}_retained_original_indices"], retained_original
            )
            or not _column_sign_aligned_equal(
                arrays[f"{prefix}_full_width_pca_loadings"], full_loadings
            )
            or not np.allclose(
                arrays[f"{prefix}_loading_importance"],
                importance,
                rtol=2e-10,
                atol=2e-10,
            )
            or not np.array_equal(
                arrays[f"{prefix}_selected_retained_indices"], selected_retained
            )
            or not np.array_equal(
                arrays[f"{prefix}_selected_original_indices"], selected_original
            )
            or not np.array_equal(
                arrays[f"{prefix}_selected_snp_ids"], selected_snp_ids
            )
            or not _column_sign_aligned_equal(
                arrays[f"{prefix}_pca_loadings"], pca_loadings
            )
            or split["selected_original_indices"] != selected_original.tolist()
            or split["selected_snp_ids"] != selected_snp_ids.tolist()
            or split["ordered_original_ranking_sha256"]
            != array_sha256(retained_original[ranking])
            or split["preprocessor_mean_sha256"]
            != array_sha256(preprocessor.mean_dosage)
            or split["preprocessor_scale_sha256"] != array_sha256(preprocessor.scale)
            or split["panel_train_sha256"] != array_sha256(panel_train)
            or split["panel_test_sha256"] != array_sha256(panel_test)
            or split["train_sample_ids_sha256"]
            != array_sha256(sample_ids[train_indices])
            or split["test_sample_ids_sha256"]
            != array_sha256(sample_ids[test_indices])
            or split["test_labels_sha256"] != array_sha256(labels[test_indices])
        ):
            raise ValueError(
                "original-input split/preprocess/PCA/top-panel reconstruction mismatch"
            )
        common_metrics, common_predictions, common_centroids = (
            _metrics_and_predictions(
                panel_train @ pca_loadings,
                labels[train_indices],
                panel_test @ pca_loadings,
                labels[test_indices],
            )
        )
        for method in METHODS:
            record = records[(seed, method)]
            loadings = np.asarray(
                arrays[f"{prefix}_{method}_loadings"], dtype=float
            )
            factor = np.asarray(
                arrays[f"{prefix}_{method}_procrustes_factor"], dtype=float
            )
            expected_factor = procrustes_factor(panel_train, loadings)
            objective = float(vp_objective(panel_train, factor, loadings, alpha))
            mapping = proximal_gradient_mapping(
                panel_train,
                factor,
                loadings,
                alpha=alpha,
                step_size=step,
            )
            stationarity = _normalized_stationarity(
                float(np.linalg.norm(mapping, ord="fro")), alpha, loadings.size
            )
            support = _support_receipt(loadings, config)
            native_metrics, native_predictions, native_centroids = (
                _metrics_and_predictions(
                    panel_train @ loadings,
                    labels[train_indices],
                    panel_test @ loadings,
                    labels[test_indices],
                )
            )
            if (
                not _column_sign_aligned_equal(factor, expected_factor, atol=3e-9)
                or not math.isclose(
                    float(record["common_objective"]),
                    objective,
                    rel_tol=2e-10,
                    abs_tol=2e-8,
                )
                or any(
                    not math.isclose(
                        float(record["common_stationarity"][key]),
                        float(stationarity[key]),
                        rel_tol=2e-9,
                        abs_tol=2e-10,
                    )
                    for key in stationarity
                )
                or record["support"] != support
                or not math.isclose(
                    float(record["lipschitz_sigma_max_squared"]),
                    lipschitz,
                    rel_tol=2e-10,
                )
                or not math.isclose(
                    float(record["reference_step_size"]), step, rel_tol=2e-10
                )
                or not math.isclose(float(record["alpha"]), alpha, rel_tol=2e-10)
                or record["native"]["metrics"] != native_metrics
                or record["common_pca_probe"]["metrics"] != common_metrics
                or not np.allclose(
                    arrays[f"{prefix}_{method}_native_centroids"],
                    native_centroids,
                    rtol=2e-10,
                    atol=2e-10,
                )
                or not np.allclose(
                    arrays[f"{prefix}_{method}_common_pca_probe_centroids"],
                    common_centroids,
                    rtol=2e-10,
                    atol=2e-10,
                )
            ):
                raise ValueError(
                    "saved A/B/step objective/KKT/support/centroid reconstruction mismatch"
                )
            for view, expected_predictions in (
                ("native", native_predictions),
                ("common_pca_probe", common_predictions),
            ):
                selected_rows = {
                    int(row["sample_index"]): row
                    for row in prediction_rows
                    if row["seed"] == seed
                    and row["method"] == method
                    and row["view"] == view
                }
                rows = [selected_rows[int(index)] for index in test_indices]
                if (
                    [int(row["sample_index"]) for row in rows]
                    != test_indices.tolist()
                    or [row["sample_id"] for row in rows]
                    != sample_ids[test_indices].tolist()
                    or [row["truth"] for row in rows]
                    != labels[test_indices].tolist()
                    or [row["prediction"] for row in rows]
                    != expected_predictions.tolist()
                ):
                    raise ValueError(
                        "original-input prediction/label/sample mapping mismatch"
                    )
    if analysis["summaries"] != summarize_records(analysis["records"], config) or analysis[
        "gates"
    ] != compute_gates(analysis["records"], config):
        raise ValueError("recomputed formal summaries/gates mismatch")


def _development_selection_summary(
    analyses_by_budget: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(analyses_by_budget) != {3000, 4000}:
        raise ValueError("development selection requires exact 3000/4000 evidence")
    candidates = []
    passing: list[int] = []
    for budget in (3000, 4000):
        analysis = analyses_by_budget[budget]
        config = development_config(budget)
        records = analysis["records"]
        gates = compute_gates(records, config)
        vp_rows = [record for record in records if record["method"] == "vp"]
        vp_passed = bool(
            gates["vp"]["normalized_kkt_all_splits"]
            and gates["vp"]["support_qualified"]
            and gates["vp"]["native_ba_noninferior"]
        )
        if vp_passed:
            passing.append(budget)
        candidates.append(
            {
                "budget": budget,
                "vp_passed_locked_development_gates": vp_passed,
                "vp_common_eta_max": float(
                    max(row["common_stationarity"]["eta"] for row in vp_rows)
                ),
                "vp_native_last20_eta_max": float(
                    max(
                        row["native_stationarity"]["last_window_max_eta"]
                        for row in vp_rows
                    )
                ),
                "vp_row_support_fraction_mean": float(
                    np.mean([row["support"]["row_fraction"] for row in vp_rows])
                ),
                "vp_entry_support_fraction_mean": float(
                    np.mean([row["support"]["entry_fraction"] for row in vp_rows])
                ),
                "vp_native_ba_mean_delta_vs_pca": float(
                    gates["vp"]["native_ba_mean_delta_vs_pca"]
                ),
                "vp_native_ba_worst_delta_vs_pca": float(
                    gates["vp"]["native_ba_worst_delta_vs_pca"]
                ),
            }
        )
    if passing != [3000, 4000]:
        raise ValueError(
            "real development rerun did not reproduce both passing budget candidates"
        )
    return {
        "status": "post_hoc_real_development_not_confirmation",
        "tested_budgets_in_predeclared_order": [3000, 4000],
        "selection_rule": "smallest_tested_budget_passing_all_locked_vp_gates",
        "candidates": candidates,
        "selected_rho": RHO,
        "selected_budget": 3000,
        "formal_seeds_executed": [],
    }


def validate_development_package_identity(identity: Mapping[str, Any]) -> None:
    keys = {
        "schema_version",
        "status",
        "bundle_manifest_file_sha256",
        "bundle_manifest_payload_sha256",
        "prepared_input_identity",
        "source_identity_aggregate_sha256",
        "selection_summary_sha256",
        "selected_rho",
        "selected_budget",
        "package_anchors_by_budget",
        "formal_seeds_executed",
        "identity_sha256",
    }
    if not isinstance(identity, Mapping) or set(identity) != keys:
        raise ValueError("development package identity schema mismatch")
    unsigned = dict(identity)
    observed = unsigned.pop("identity_sha256", None)
    validate_formal_input_identity(identity["prepared_input_identity"])
    anchors = identity["package_anchors_by_budget"]
    if (
        identity["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or identity["status"] != "validated_real_post_hoc_development_bundle"
        or any(
            not _is_sha256(identity[field])
            for field in (
                "bundle_manifest_file_sha256",
                "bundle_manifest_payload_sha256",
                "source_identity_aggregate_sha256",
                "selection_summary_sha256",
            )
        )
        or identity["selected_rho"] != RHO
        or identity["selected_budget"] != TOTAL_BUDGET
        or identity["formal_seeds_executed"] != []
        or not isinstance(anchors, Mapping)
        or set(anchors) != {"3000", "4000"}
        or any(
            not isinstance(value, Mapping)
            or set(value) != {"analysis_sha256", "package_anchor_sha256"}
            or not _is_sha256(value["analysis_sha256"])
            or not _is_sha256(value["package_anchor_sha256"])
            for value in anchors.values()
        )
        or not _is_sha256(observed)
        or canonical_sha256(unsigned) != observed
    ):
        raise ValueError("development package identity policy/digest mismatch")


def run_development_bundle(
    *,
    repo_root: Path,
    genotype_npz: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    output_directory: Path,
) -> dict[str, Any]:
    """Execute and package real 2025..2029 post-hoc development evidence."""

    output = Path(output_directory)
    if os.path.lexists(output):
        raise FileExistsError("development bundle output must not exist")
    data, preparation = _load_anchored_formal_input(
        genotype_npz, source_manifest, expected_source_manifest_sha256
    )
    prepared_input = {
        **_input_identity(data),
        **preparation,
        "population_design": "five_superpopulation",
        "input_scope": "legacy_chr20_22",
        "ld_pruned": True,
        "population_label_is_race": False,
    }
    validate_formal_input_identity(prepared_input)
    source = collect_source_identity(repo_root)
    environment = runtime_execution_contract()
    validate_runtime_execution_contract(environment, require_speed_qualification=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        package_results: dict[int, dict[str, Any]] = {}
        analyses: dict[int, Mapping[str, Any]] = {}
        artifact_receipts: dict[str, dict[str, Any]] = {}
        for budget, directory_name in (
            (3000, DEVELOPMENT_3000_DIRECTORY),
            (4000, DEVELOPMENT_4000_DIRECTORY),
        ):
            config = development_config(budget)
            protocol = _build_development_protocol(config, source, prepared_input)
            records, predictions, arrays, split_receipts = evaluate(data, config)
            package_results[budget] = write_output_package(
                output_directory=temporary / directory_name,
                config=config,
                data=data,
                repo_root=repo_root,
                protocol_receipt=protocol,
                records=records,
                predictions=predictions,
                arrays=arrays,
                split_receipts=split_receipts,
                execution_environment=environment,
            )
            analyses[budget] = _strict_json_load(
                temporary / directory_name / ANALYSIS_FILENAME
            )
        selection = _development_selection_summary(analyses)
        for path in sorted(temporary.rglob("*")):
            if path.is_symlink():
                raise ValueError("development bundle refuses symlinks")
            if path.is_file():
                relative = path.relative_to(temporary).as_posix()
                artifact_receipts[relative] = {
                    "bytes": int(path.stat().st_size),
                    "sha256": _file_sha256(path),
                }
        unsigned = {
            "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
            "status": "complete_real_post_hoc_development_non_confirmatory",
            "manifest_written_last": True,
            "prepared_input_identity": prepared_input,
            "source_identity": source,
            "execution_environment": environment,
            "selection_summary": selection,
            "packages": {
                str(budget): {
                    "relative_directory": directory,
                    "analysis_sha256": package_results[budget]["analysis_sha256"],
                    "package_anchor_sha256": package_results[budget][
                        "package_anchor_sha256"
                    ],
                }
                for budget, directory in (
                    (3000, DEVELOPMENT_3000_DIRECTORY),
                    (4000, DEVELOPMENT_4000_DIRECTORY),
                )
            },
            "artifact_count_excluding_bundle_manifest": len(artifact_receipts),
            "artifacts": artifact_receipts,
            "formal_seeds_executed": [],
        }
        manifest = {
            **unsigned,
            "manifest_payload_sha256": canonical_sha256(unsigned),
        }
        _write_new_file(
            temporary / DEVELOPMENT_BUNDLE_MANIFEST_FILENAME,
            _strict_json_bytes(manifest),
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    manifest_file_sha = _file_sha256(output / DEVELOPMENT_BUNDLE_MANIFEST_FILENAME)
    return validate_development_bundle(
        output,
        repo_root=repo_root,
        genotype_npz=genotype_npz,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_bundle_manifest_file_sha256=manifest_file_sha,
    )


def validate_development_bundle(
    bundle_directory: Path,
    *,
    repo_root: Path,
    genotype_npz: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    expected_bundle_manifest_file_sha256: str,
) -> dict[str, Any]:
    root = Path(bundle_directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("development bundle must be a real directory")
    manifest_path = root / DEVELOPMENT_BUNDLE_MANIFEST_FILENAME
    _require_regular_file(manifest_path, role="development bundle manifest")
    if (
        not _is_sha256(expected_bundle_manifest_file_sha256)
        or _file_sha256(manifest_path) != expected_bundle_manifest_file_sha256
    ):
        raise ValueError("development bundle external manifest SHA mismatch")
    manifest = _strict_json_load(manifest_path)
    keys = {
        "schema_version",
        "status",
        "manifest_written_last",
        "prepared_input_identity",
        "source_identity",
        "execution_environment",
        "selection_summary",
        "packages",
        "artifact_count_excluding_bundle_manifest",
        "artifacts",
        "formal_seeds_executed",
        "manifest_payload_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != keys:
        raise ValueError("development bundle manifest schema mismatch")
    _validate_digest_field(manifest, "manifest_payload_sha256")
    if (
        manifest["schema_version"] != DEVELOPMENT_BUNDLE_SCHEMA_VERSION
        or manifest["status"]
        != "complete_real_post_hoc_development_non_confirmatory"
        or manifest["manifest_written_last"] is not True
        or manifest["formal_seeds_executed"] != []
        or set(manifest["packages"]) != {"3000", "4000"}
    ):
        raise ValueError("development bundle policy mismatch")
    validate_source_identity(manifest["source_identity"])
    current_source = collect_source_identity(repo_root)
    if current_source != manifest["source_identity"]:
        raise ValueError("development bundle source differs from current repo")
    validate_runtime_execution_contract(
        manifest["execution_environment"], require_speed_qualification=True
    )
    data, preparation = _load_anchored_formal_input(
        genotype_npz, source_manifest, expected_source_manifest_sha256
    )
    expected_input = {
        **_input_identity(data),
        **preparation,
        "population_design": "five_superpopulation",
        "input_scope": "legacy_chr20_22",
        "ld_pruned": True,
        "population_label_is_race": False,
    }
    if manifest["prepared_input_identity"] != expected_input:
        raise ValueError("development bundle original input identity mismatch")
    actual_artifacts: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("development bundle refuses symlinks")
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(root).as_posix()
            actual_artifacts[relative] = {
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
    if (
        manifest["artifacts"] != actual_artifacts
        or manifest["artifact_count_excluding_bundle_manifest"]
        != len(actual_artifacts)
        or {path.name for path in root.iterdir()}
        != {
            DEVELOPMENT_3000_DIRECTORY,
            DEVELOPMENT_4000_DIRECTORY,
            DEVELOPMENT_BUNDLE_MANIFEST_FILENAME,
        }
    ):
        raise ValueError("development bundle file closure mismatch")
    analyses: dict[int, Mapping[str, Any]] = {}
    package_anchors: dict[str, dict[str, str]] = {}
    for budget, directory_name in (
        (3000, DEVELOPMENT_3000_DIRECTORY),
        (4000, DEVELOPMENT_4000_DIRECTORY),
    ):
        package = root / directory_name
        package_receipt = manifest["packages"][str(budget)]
        if package_receipt["relative_directory"] != directory_name:
            raise ValueError("development package directory mismatch")
        validation = validate_output_package(
            package,
            repo_root=repo_root,
            expected_analysis_sha256=package_receipt["analysis_sha256"],
            expected_package_anchor_sha256=package_receipt["package_anchor_sha256"],
        )
        analysis = _strict_json_load(package / ANALYSIS_FILENAME)
        predictions = _strict_json_load(package / PREDICTIONS_FILENAME)
        arrays = _load_npz_bytes((package / ARRAYS_FILENAME).read_bytes())
        protocol = _strict_json_load(package / DEVELOPMENT_PROTOCOL_FILENAME)
        _recompute_scientific_outputs_from_original_input(
            data_path=genotype_npz,
            source_manifest=source_manifest,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            analysis=analysis,
            predictions_payload=predictions,
            arrays=arrays,
            protocol_receipt=protocol,
        )
        if validation["profile"] != f"development_{budget}":
            raise ValueError("development package profile mismatch")
        analyses[budget] = analysis
        package_anchors[str(budget)] = {
            "analysis_sha256": validation["analysis_sha256"],
            "package_anchor_sha256": validation["package_anchor_sha256"],
        }
    selection = _development_selection_summary(analyses)
    if manifest["selection_summary"] != selection:
        raise ValueError("development budget selection evidence mismatch")
    identity_unsigned = {
        "schema_version": DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        "status": "validated_real_post_hoc_development_bundle",
        "bundle_manifest_file_sha256": _file_sha256(manifest_path),
        "bundle_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "prepared_input_identity": expected_input,
        "source_identity_aggregate_sha256": current_source["aggregate_sha256"],
        "selection_summary_sha256": canonical_sha256(selection),
        "selected_rho": selection["selected_rho"],
        "selected_budget": selection["selected_budget"],
        "package_anchors_by_budget": package_anchors,
        "formal_seeds_executed": [],
    }
    identity = {
        **identity_unsigned,
        "identity_sha256": canonical_sha256(identity_unsigned),
    }
    validate_development_package_identity(identity)
    return identity


def freeze_formal_protocol(
    *,
    repo_root: Path,
    genotype_npz: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    expected_shared_source_sha256: str,
    development_bundle_directory: Path,
    expected_development_bundle_manifest_sha256: str,
    execution_registry_directory: Path,
    exclusive_output_directory: Path,
    lock_output: Path,
) -> dict[str, Any]:
    """Freeze the contract only; this function never evaluates a formal seed."""

    data, preparation_identity = _load_anchored_formal_input(
        genotype_npz, source_manifest, expected_source_manifest_sha256
    )
    development_identity = validate_development_bundle(
        development_bundle_directory,
        repo_root=repo_root,
        genotype_npz=genotype_npz,
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_bundle_manifest_file_sha256=(
            expected_development_bundle_manifest_sha256
        ),
    )
    lock = build_formal_lock(
        repo_root=repo_root,
        data=data,
        preparation_identity=preparation_identity,
        freshness_roots=canonical_freshness_roots(repo_root),
        output_directory=exclusive_output_directory,
        expected_shared_source_sha256=expected_shared_source_sha256,
        development_package_identity=development_identity,
        execution_registry_directory=execution_registry_directory,
    )
    validate_formal_lock(lock)
    _write_new_file(lock_output, _strict_json_bytes(lock))
    return {
        "status": "formal_protocol_frozen_no_fresh_seed_executed",
        "lock_path": str(Path(lock_output).resolve()),
        "lock_file_sha256": _file_sha256(lock_output),
        "lock_payload_sha256": lock["lock_payload_sha256"],
        "source_identity_sha256": lock["source_identity"]["aggregate_sha256"],
        "development_bundle_manifest_file_sha256": (
            development_identity["bundle_manifest_file_sha256"]
        ),
        "formal_seeds_executed": [],
    }


def run_smoke(output_directory: Path, repo_root: Path) -> dict[str, Any]:
    config = smoke_config()
    data = make_smoke_data()
    source = collect_source_identity(repo_root)
    protocol = _build_smoke_protocol(config, source, _input_identity(data))
    records, predictions, arrays, split_receipts = evaluate(data, config)
    return write_output_package(
        output_directory=output_directory,
        config=config,
        data=data,
        repo_root=repo_root,
        protocol_receipt=protocol,
        records=records,
        predictions=predictions,
        arrays=arrays,
        split_receipts=split_receipts,
    )


def run_formal_confirmation(
    *,
    repo_root: Path,
    genotype_npz: Path,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    lock_path: Path,
    expected_lock_file_sha256: str,
    output_directory: Path,
    execute_fresh_confirmation: bool,
) -> dict[str, Any]:
    if execute_fresh_confirmation is not True:
        raise PermissionError("formal run requires --execute-fresh-confirmation")
    config = formal_config()
    lock = load_anchored_formal_lock(lock_path, expected_lock_file_sha256)
    output = Path(output_directory).resolve()
    if str(output) != lock["exclusive_output_directory"] or os.path.lexists(output):
        raise ValueError("formal output is not the locked new exclusive directory")
    current_source = collect_source_identity(repo_root)
    if current_source != lock["source_identity"]:
        raise ValueError("current runtime import closure differs from the formal lock")
    data, preparation_identity = _load_anchored_formal_input(
        genotype_npz, source_manifest, expected_source_manifest_sha256
    )
    expected_input = {
        **_input_identity(data),
        **preparation_identity,
        "population_design": "five_superpopulation",
        "input_scope": "legacy_chr20_22",
        "ld_pruned": True,
        "population_label_is_race": False,
    }
    if expected_input != lock["prepared_input_identity"]:
        raise ValueError("current prepared input differs from the formal lock")
    current_freshness = audit_fresh_seed_execution(
        canonical_freshness_roots(repo_root), config.seeds
    )
    current_scope_inventory = build_freshness_scope_inventory(repo_root, output)
    if (
        current_freshness != lock["freshness_audit"]
        or current_scope_inventory != lock["freshness_scope_inventory"]
    ):
        raise ValueError("freshness snapshot changed after formal protocol freeze")
    environment = runtime_execution_contract()
    validate_runtime_execution_contract(environment, require_speed_qualification=True)
    capability = _consume_global_formal_execution(
        registry_directory=Path(lock["global_execution_registry_directory"]),
        lock=lock,
        lock_file_sha256=expected_lock_file_sha256,
        output_directory=output,
    )
    records, predictions, arrays, split_receipts = evaluate(
        data, config, _formal_capability=capability
    )
    prepublish_freshness = audit_fresh_seed_execution(
        canonical_freshness_roots(repo_root), config.seeds
    )
    if prepublish_freshness != lock["freshness_audit"]:
        raise ValueError("freshness snapshot changed during formal evaluation")
    written = write_output_package(
        output_directory=output,
        config=config,
        data=data,
        repo_root=repo_root,
        protocol_receipt=lock,
        records=records,
        predictions=predictions,
        arrays=arrays,
        split_receipts=split_receipts,
        execution_environment=environment,
        execution_spend_receipt=capability._receipt,
    )
    return validate_output_package(
        output,
        repo_root=repo_root,
        expected_analysis_sha256=written["analysis_sha256"],
        expected_package_anchor_sha256=written["package_anchor_sha256"],
        genotype_npz=genotype_npz,
        source_manifest=source_manifest,
        original_lock_path=lock_path,
        expected_original_lock_sha256=expected_lock_file_sha256,
        expected_output_manifest_sha256=written[
            "output_manifest_file_sha256"
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    source = subparsers.add_parser(
        "source-identity", help="print the fresh-import repo source closure"
    )
    source.add_argument("--repo-root", type=Path, default=Path("."))

    smoke = subparsers.add_parser("smoke", help="run synthetic non-evidence smoke")
    smoke.add_argument("--repo-root", type=Path, default=Path("."))
    smoke.add_argument("--output-dir", type=Path, required=True)

    development = subparsers.add_parser(
        "run-development",
        help="run/package real post-hoc seeds 2025..2029 at budgets 3000/4000",
    )
    development.add_argument("--repo-root", type=Path, default=Path("."))
    development.add_argument("--genotypes", type=Path, required=True)
    development.add_argument("--source-manifest", type=Path, required=True)
    development.add_argument("--expected-source-manifest-sha256", required=True)
    development.add_argument("--output-dir", type=Path, required=True)

    validate_development = subparsers.add_parser(
        "validate-development", help="strictly validate the real development bundle"
    )
    validate_development.add_argument("output_dir", type=Path)
    validate_development.add_argument("--repo-root", type=Path, required=True)
    validate_development.add_argument("--genotypes", type=Path, required=True)
    validate_development.add_argument(
        "--source-manifest", type=Path, required=True
    )
    validate_development.add_argument(
        "--expected-source-manifest-sha256", required=True
    )
    validate_development.add_argument(
        "--expected-bundle-manifest-sha256", required=True
    )

    freeze = subparsers.add_parser(
        "freeze-formal", help="freeze a formal lock without executing fresh seeds"
    )
    freeze.add_argument("--repo-root", type=Path, default=Path("."))
    freeze.add_argument("--genotypes", type=Path, required=True)
    freeze.add_argument("--source-manifest", type=Path, required=True)
    freeze.add_argument("--expected-source-manifest-sha256", required=True)
    freeze.add_argument("--expected-shared-source-sha256", required=True)
    freeze.add_argument("--development-bundle", type=Path, required=True)
    freeze.add_argument(
        "--expected-development-bundle-manifest-sha256", required=True
    )
    freeze.add_argument("--execution-registry-dir", type=Path, required=True)
    freeze.add_argument("--exclusive-output-dir", type=Path, required=True)
    freeze.add_argument("--lock-output", type=Path, required=True)

    formal = subparsers.add_parser(
        "run-formal", help="execute the externally anchored fresh confirmation"
    )
    formal.add_argument("--repo-root", type=Path, default=Path("."))
    formal.add_argument("--genotypes", type=Path, required=True)
    formal.add_argument("--source-manifest", type=Path, required=True)
    formal.add_argument("--expected-source-manifest-sha256", required=True)
    formal.add_argument("--protocol-lock", type=Path, required=True)
    formal.add_argument("--expected-protocol-lock-sha256", required=True)
    formal.add_argument("--output-dir", type=Path, required=True)
    formal.add_argument("--execute-fresh-confirmation", action="store_true")

    validate = subparsers.add_parser("validate", help="strictly validate a package")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path)
    validate.add_argument("--expected-analysis-sha256")
    validate.add_argument("--expected-package-anchor-sha256")
    validate.add_argument("--genotypes", type=Path)
    validate.add_argument("--source-manifest", type=Path)
    validate.add_argument("--original-protocol-lock", type=Path)
    validate.add_argument("--expected-original-protocol-lock-sha256")
    validate.add_argument("--expected-output-manifest-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "source-identity":
        result = collect_source_identity(args.repo_root)
    elif args.command == "smoke":
        result = run_smoke(args.output_dir, args.repo_root)
    elif args.command == "run-development":
        result = run_development_bundle(
            repo_root=args.repo_root,
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            output_directory=args.output_dir,
        )
    elif args.command == "validate-development":
        result = validate_development_bundle(
            args.output_dir,
            repo_root=args.repo_root,
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_bundle_manifest_file_sha256=(
                args.expected_bundle_manifest_sha256
            ),
        )
    elif args.command == "freeze-formal":
        result = freeze_formal_protocol(
            repo_root=args.repo_root,
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            expected_shared_source_sha256=args.expected_shared_source_sha256,
            development_bundle_directory=args.development_bundle,
            expected_development_bundle_manifest_sha256=(
                args.expected_development_bundle_manifest_sha256
            ),
            execution_registry_directory=args.execution_registry_dir,
            exclusive_output_directory=args.exclusive_output_dir,
            lock_output=args.lock_output,
        )
    elif args.command == "run-formal":
        result = run_formal_confirmation(
            repo_root=args.repo_root,
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            expected_source_manifest_sha256=args.expected_source_manifest_sha256,
            lock_path=args.protocol_lock,
            expected_lock_file_sha256=args.expected_protocol_lock_sha256,
            output_directory=args.output_dir,
            execute_fresh_confirmation=args.execute_fresh_confirmation,
        )
    else:
        result = validate_output_package(
            args.output_dir,
            repo_root=args.repo_root,
            expected_analysis_sha256=args.expected_analysis_sha256,
            expected_package_anchor_sha256=args.expected_package_anchor_sha256,
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            original_lock_path=args.original_protocol_lock,
            expected_original_lock_sha256=(
                args.expected_original_protocol_lock_sha256
            ),
            expected_output_manifest_sha256=args.expected_output_manifest_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

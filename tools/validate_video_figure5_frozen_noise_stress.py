"""Independent semantic validator for frozen Figure 5 noise-stress output.

This tool intentionally does not import the experiment runner or ``rspca``.
Its protocol constants, exact-closure checks, canonical JSON hashing, CSV
cardinality checks, and current-source anchors form an external validation
boundary for downloaded smoke or formal output.
"""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from io import BytesIO
from itertools import permutations
import json
import math
from pathlib import Path
import stat
from typing import Any, Mapping

import numpy as np
from PIL import Image


REPORT_NAME = "frozen_noise_stress_report.json"
CASES_NAME = "frozen_noise_stress_cases.csv"
FRAMES_NAME = "frozen_noise_stress_full_frame_errors.csv"
FIGURE_NAME = "frozen_noise_stress_summary.png"
LOCK_NAME = "frozen_noise_stress_protocol_lock.json"
SOURCE_MANIFEST_NAME = "frozen_noise_stress_source_manifest.json"
DATA_MANIFEST_NAME = "frozen_noise_stress_data_manifest.json"
OUTPUT_MANIFEST_NAME = "frozen_noise_stress_output_manifest.json"
EXECUTION_RECEIPT_NAME = "frozen_noise_stress_execution_started_receipt.json"
SEED_REGISTRY_NAME = "frozen_noise_stress_seed_bank_registry.json"
FACTORS_NAME = "frozen_noise_stress_factors.npz"
PAYLOAD_NAMES = frozenset(
    {
        REPORT_NAME,
        CASES_NAME,
        FRAMES_NAME,
        FIGURE_NAME,
        LOCK_NAME,
        SOURCE_MANIFEST_NAME,
        DATA_MANIFEST_NAME,
        EXECUTION_RECEIPT_NAME,
        SEED_REGISTRY_NAME,
        FACTORS_NAME,
    }
)
OUTPUT_NAMES = PAYLOAD_NAMES | {OUTPUT_MANIFEST_NAME}
METHODS = {"PCA", "VP-SPCA", "RND-VP-SPCA"}
EXPECTED_SOURCE_ROLE_SCHEMA = {
    "experiments/video_figure5_frozen_noise_stress.py": "stress_runner",
    "experiments/video_spca_paper_reproduction.py": (
        "original_figure5_generator_alignment_and_fit_adapters"
    ),
    "src/rspca/__init__.py": "package_initialization",
    "src/rspca/accelerated.py": "package_initialization_only",
    "src/rspca/cross_modal_gates.py": "package_initialization_only",
    "src/rspca/datasets.py": "array_and_canonical_hashing",
    "src/rspca/federated.py": "package_initialization_only",
    "src/rspca/fedrpca_audit.py": "package_initialization_only",
    "src/rspca/gauge.py": "package_initialization_only",
    "src/rspca/gauge_stress.py": "package_initialization_only",
    "src/rspca/metrics.py": "reconstruction_metric",
    "src/rspca/product_aggregation.py": "package_initialization_only",
    "src/rspca/randomized.py": "randomized_vp_spca_solver",
    "src/rspca/robust.py": "package_initialization_only",
    "src/rspca/statistical_gates.py": "package_initialization_only",
    "src/rspca/trust_audit.py": "package_initialization_only",
    "src/rspca/vp.py": "deterministic_vp_spca_solver",
    "tools/validate_video_figure5_frozen_noise_stress.py": (
        "independent_external_anchor_validator"
    ),
}
FROZEN_SHARED_CORE_SHA256 = {
    "src/rspca/vp.py": (
        "88a84aa99825357f3a16650bf3e56bfef54a850aa17880342334ba7ad402b953"
    ),
    "src/rspca/randomized.py": (
        "101833335c80b178f35b696bdc28dd8a4c0eb6a9fc5a565af802d83088d0823b"
    ),
}
FORMAL_BANK = [
    [20.0, list(range(83001, 83021))],
    [10.0, list(range(83101, 83121))],
    [0.0, list(range(83201, 83221))],
]
SMOKE_BANK = [[20.0, [83901]], [10.0, [83902]], [0.0, [83903]]]
THRESHOLDS = {
    "clean_relative_error_ratio": ("upper", "<=", 1.02),
    "max_frame_rmse_ratio": ("upper", "<=", 1.02),
    "spatial_mean_cosine_loss": ("upper", "<=", 0.002),
    "spatial_worst_cosine_loss": ("upper", "<=", 0.002),
    "temporal_mean_cosine_loss": ("upper", "<=", 0.002),
    "temporal_worst_cosine_loss": ("upper", "<=", 0.002),
    "off_support_absolute_increment": ("upper", "<=", 0.01),
    "cross_talk_absolute_increment": ("upper", "<=", 0.01),
    "loading_sparsity_absolute_difference": ("upper", "<=", 0.01),
    "relative_full_objective_excess": ("upper", "<=", 1e-4),
    "complete_method_runtime_speedup": ("lower", ">", 10.0),
}
DATA_MANIFEST_KEYS = {
    "schema_version",
    "manifest_kind",
    "profile",
    "orientation",
    "clean_array_sha256",
    "clean_shape",
    "cases",
    "precomputed_before_first_fit",
    "regenerated_and_hash_checked_before_each_fit",
    "data_manifest_body_sha256",
}
DATA_CASE_KEYS = {
    "snr_db",
    "noise_seed",
    "shape",
    "distribution",
    "target_snr_db",
    "achieved_snr_db",
    "noise_nonzero_fraction",
    "full_column_rank_witness",
    "array_sha256",
}
CASE_RECORD_KEYS = {
    "snr_db",
    "noise_seed",
    "method",
    "raw_solver_fit_call_runtime_seconds",
    "complete_method_runtime_seconds",
    "runtime_components",
    "fit_execution_order",
    "fit_execution_position",
    "factor_keys",
    "factor_step_size",
    "converged",
    "full_converged",
    "n_iter",
    "status",
    "final_full_stationarity",
    "tolerance",
    "stationarity_tolerance",
    "stationarity_met",
    "final_full_objective_recomputed",
    "last_solver_objective",
    "clean_relative_reconstruction_error",
    "observed_relative_reconstruction_error",
    "full_frame_normalized_rmse",
    "loading_sparsity_at_threshold",
    "loading_zero_threshold",
    "mode_and_dynamics_recovery",
    "localization",
    "randomized_diagnostics",
}
CASE_CSV_FIELDS = {
    "snr_db",
    "noise_seed",
    "method",
    "converged",
    "full_converged",
    "n_iter",
    "status",
    "final_full_stationarity",
    "tolerance",
    "stationarity_tolerance",
    "stationarity_met",
    "factor_step_size",
    "factor_A_key",
    "factor_B_key",
    "factor_step_key",
    "final_full_objective_recomputed",
    "last_solver_objective",
    "raw_solver_fit_call_runtime_seconds",
    "complete_method_runtime_seconds",
    "clean_relative_reconstruction_error",
    "observed_relative_reconstruction_error",
    "frame_rmse_median",
    "frame_rmse_p95",
    "frame_rmse_max",
    "worst_frame_index",
    "worst_frame_time",
    "loading_sparsity_at_threshold",
    "spatial_mode_mean_cosine",
    "spatial_mode_worst_cosine",
    "temporal_dynamics_mean_cosine",
    "temporal_dynamics_worst_cosine",
    "off_support_energy_fraction_mean",
    "cross_talk_energy_fraction_mean",
    "sketch_residual",
    "sketch_residual_relative",
}
FRAME_CSV_FIELDS = {
    "snr_db", "noise_seed", "method", "frame_index", "time", "normalized_rmse"
}
FORMAL_SEED_REGISTRY_KEYS = {
    "schema_version",
    "registry_kind",
    "canonical_formal_root_resolved",
    "canonical_anchor_dir_resolved",
    "canonical_output_dir_resolved",
    "formal_snr_seed_bank",
    "freshness_inventory_sha256",
    "freshness_audit_body_sha256",
    "timing_environment_body_sha256",
    "created_with_exclusive_open_before_anchor_and_any_fit",
    "reuse_or_reissue_forbidden",
    "seed_registry_body_sha256",
}
SMOKE_SEED_REGISTRY_KEYS = {
    "schema_version",
    "registry_kind",
    "formal_snr_seed_bank",
    "created_with_exclusive_open_before_anchor_and_any_fit",
    "reuse_or_reissue_forbidden",
    "seed_registry_body_sha256",
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} exact schema mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _array_sha256(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays are forbidden")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _independent_figure5_generator(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_size = int(config["grid_size"])
    coordinate = np.linspace(0.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="xy")

    def blob(center_x: float, center_y: float, sigma: float) -> np.ndarray:
        image = np.exp(
            -((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)
            / (2.0 * sigma**2)
        )
        return np.asarray(image / max(float(image.max()), 1e-12), dtype=np.float64)

    raw_modes = np.ascontiguousarray(
        np.column_stack(
            [
                blob(0.55, 0.55, 0.06).reshape(-1),
                blob(0.30, 0.20, 0.025).reshape(-1),
                blob(0.78, 0.80, 0.025).reshape(-1),
            ]
        ),
        dtype=np.float64,
    )
    time = np.linspace(0.0, float(config["total_time"]), int(config["n_timesteps"]))

    def window(start: float, stop: float, ramp: float) -> np.ndarray:
        left = 1.0 / (1.0 + np.exp(-(time - start) / ramp))
        right = 1.0 / (1.0 + np.exp(-(time - stop) / ramp))
        return np.clip(left - right, 0.0, 1.0)

    raw_scores = np.zeros((time.size, 3), dtype=np.float64)
    raw_scores[:, 0] = (
        2.0
        * np.sin(2.0 * np.pi * 0.08 * time)
        * window(0.0, float(config["total_time"]), 6.0)
    )
    raw_scores[:, 1] = 0.8 * window(45.0, 65.0, 2.0) + 1.8 * np.sin(
        2.0 * np.pi * 0.055 * time + 0.35 * np.pi
    ) * window(65.0, 105.0, 2.0)
    raw_scores[:, 2] = 2.0 * np.sin(
        2.0 * np.pi * 0.12 * time + 0.1 * np.pi
    ) * window(100.0, float(config["total_time"]), 2.0)
    centered = raw_scores - raw_scores.mean(axis=0, keepdims=True)
    clean = np.ascontiguousarray(raw_modes @ centered.T)
    norms = np.linalg.norm(centered, axis=0)
    truth_modes = np.ascontiguousarray(raw_modes * norms)
    truth_dynamics = np.ascontiguousarray(centered / norms)
    return clean, truth_modes, truth_dynamics, np.asarray(time, dtype=np.float64)


def _independent_noisy_case(
    clean: np.ndarray, snr_db: float, noise_seed: int
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    rng = np.random.default_rng(int(noise_seed))
    noise = np.asarray(rng.standard_normal(clean.shape), dtype=np.float64)
    noise *= (
        float(np.linalg.norm(clean)) / (10.0 ** (float(snr_db) / 20.0))
    ) / float(np.linalg.norm(noise))
    observed = np.ascontiguousarray(clean + noise)
    achieved = float(20.0 * np.log10(np.linalg.norm(clean) / np.linalg.norm(noise)))
    witness = np.ascontiguousarray(noise[: clean.shape[1], :])
    singular_values = np.linalg.svd(witness, compute_uv=False)
    tolerance = singular_values[0] * max(witness.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    witness_record = {
        "leading_square_shape": [clean.shape[1], clean.shape[1]],
        "rank": rank,
        "minimum_singular_value": float(singular_values[-1]),
        "passed": rank == clean.shape[1],
    }
    return noise, observed, achieved, witness_record


def _soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _align_factors(
    truth_modes: np.ndarray,
    estimated_modes: np.ndarray,
    estimated_dynamics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    denominator = np.linalg.norm(truth_modes, axis=0)[:, None] * np.linalg.norm(
        estimated_modes, axis=0
    )[None, :]
    correlations = np.divide(
        truth_modes.T @ estimated_modes,
        denominator,
        out=np.zeros((3, 3), dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )
    candidates = list(permutations(range(3)))
    permutation = max(
        candidates,
        key=lambda candidate: sum(abs(correlations[index, candidate[index]]) for index in range(3)),
    )
    signs = [
        1 if correlations[index, permutation[index]] >= 0.0 else -1
        for index in range(3)
    ]
    aligned_modes = np.column_stack(
        [signs[index] * estimated_modes[:, permutation[index]] for index in range(3)]
    )
    aligned_dynamics = np.column_stack(
        [
            signs[index] * estimated_dynamics[:, permutation[index]]
            for index in range(3)
        ]
    )
    return aligned_modes, aligned_dynamics, list(permutation), signs


def _paired_cosines(reference: np.ndarray, estimate: np.ndarray) -> list[float]:
    numerator = np.sum(reference * estimate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(estimate, axis=0)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )
    return [float(value) for value in np.clip(values, -1.0, 1.0)]


def _localization(
    truth_modes: np.ndarray, aligned_modes: np.ndarray, threshold: float
) -> dict[str, Any]:
    maxima = np.max(np.abs(truth_modes), axis=0)
    support = np.abs(truth_modes) >= threshold * maxima[None, :]
    off_support: list[float] = []
    cross_talk: list[float] = []
    for component in range(3):
        energy = aligned_modes[:, component] ** 2
        total = max(float(np.sum(energy)), np.finfo(np.float64).eps)
        off_support.append(float(np.sum(energy[~support[:, component]]) / total))
        other_support = np.any(np.delete(support, component, axis=1), axis=1) & ~support[
            :, component
        ]
        cross_talk.append(float(np.sum(energy[other_support]) / total))
    return {
        "relative_truth_support_threshold": float(threshold),
        "truth_support_pixels_per_component": [
            int(value) for value in np.sum(support, axis=0)
        ],
        "off_support_energy_fraction_per_component": off_support,
        "off_support_energy_fraction_mean": float(np.mean(off_support)),
        "cross_talk_energy_fraction_per_component": cross_talk,
        "cross_talk_energy_fraction_mean": float(np.mean(cross_talk)),
    }


def _reorthogonalize(matrix: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(matrix, mode="reduced")
    q, _ = np.linalg.qr(q, mode="reduced")
    return np.asarray(q, dtype=np.float64)


def _independent_sketch_residual(
    observed: np.ndarray, config: Mapping[str, Any]
) -> tuple[float, float]:
    sketch_size = min(
        min(observed.shape),
        int(config["n_components"]) + int(config["randomized_oversample"]),
    )
    rng = np.random.default_rng(int(config["randomized_seed"]))
    omega = rng.standard_normal((observed.shape[1], sketch_size))
    q = _reorthogonalize(observed @ omega)
    for _ in range(int(config["randomized_power_iters"])):
        q = _reorthogonalize(observed @ _reorthogonalize(observed.T @ q))
    residual = observed - q @ (q.T @ observed)
    absolute = float(np.linalg.norm(residual))
    relative = absolute / max(float(np.linalg.norm(observed)), np.finfo(float).eps)
    return absolute, relative


def _strict_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )


def _verify_body_digest(
    payload: Mapping[str, Any], field: str, *, label: str
) -> None:
    body = dict(payload)
    recorded = body.pop(field, None)
    if not isinstance(recorded, str) or recorded != _canonical_sha256(body):
        raise ValueError(f"{label} canonical digest mismatch")


def _validate_expected_formal_file_hashes(
    profile: str,
    *,
    actual_lock_sha256: str,
    actual_output_manifest_sha256: str,
    expected_lock_sha256: str | None,
    expected_output_manifest_sha256: str | None,
) -> None:
    if (
        profile != "formal_confirmation"
        and expected_lock_sha256 is None
        and expected_output_manifest_sha256 is None
    ):
        return
    for value, label in (
        (expected_lock_sha256, "--expected-lock-sha256"),
        (expected_output_manifest_sha256, "--expected-output-manifest-sha256"),
    ):
        if not isinstance(value, str) or len(value) != 64 or any(
            character not in "0123456789abcdef" for character in value
        ):
            raise ValueError(f"formal validation requires {label} as 64 lowercase hex")
    if expected_lock_sha256 != actual_lock_sha256:
        raise ValueError("formal lock differs from the out-of-band expected SHA-256")
    if expected_output_manifest_sha256 != actual_output_manifest_sha256:
        raise ValueError(
            "formal output manifest differs from the out-of-band expected SHA-256"
        )


def _expected_config(profile: str) -> dict[str, Any]:
    common = {
        "alpha": 9.0,
        "loading_zero_threshold": 1e-10,
        "max_iter": 500,
        "n_components": 3,
        "randomized_full_refine_iters": 0,
        "randomized_oversample": 100,
        "randomized_power_iters": 4,
        "randomized_seed": 42,
        "support_threshold": 0.05,
        "tol": 1e-6,
        "total_time": 150.0,
    }
    if profile == "formal_confirmation":
        return {
            **common,
            "profile": profile,
            "grid_size": 200,
            "n_timesteps": 300,
            "snr_seed_bank": FORMAL_BANK,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 84001,
            "dpi": 150,
        }
    if profile == "smoke_not_evidence":
        return {
            **common,
            "profile": profile,
            "grid_size": 10,
            "n_timesteps": 36,
            "snr_seed_bank": SMOKE_BANK,
            "bootstrap_draws": 200,
            "bootstrap_seed": 84901,
            "dpi": 72,
        }
    raise ValueError("unknown output profile")


def _validate_gate_specification(specification: Mapping[str, Any]) -> None:
    if specification.get("formal_pairs_per_snr") != 20:
        raise ValueError("formal paired-seed count changed")
    if specification.get("stratification") != "each_SNR_evaluated_separately":
        raise ValueError("SNR stratification changed")
    bootstrap = specification.get("paired_bootstrap", {})
    if bootstrap.get("confidence_level") != 0.95 or bootstrap.get("draws") != 10000:
        raise ValueError("formal bootstrap contract changed")
    actual = specification.get("thresholds", {})
    if set(actual) != set(THRESHOLDS):
        raise ValueError("predeclared gate metric set changed")
    for name, (bound, operator, value) in THRESHOLDS.items():
        rule = actual[name]
        if (
            rule.get("ci_bound") != bound
            or rule.get("operator") != operator
            or rule.get("value") != value
        ):
            raise ValueError(f"predeclared gate changed: {name}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _assert_finite_tree(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number in {label}")


def _close(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=2e-13, abs_tol=2e-14):
        raise ValueError(f"numeric mismatch for {label}: {actual} != {expected}")


def _close_recomputed(actual: float, expected: float, *, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=2e-10, abs_tol=2e-11):
        raise ValueError(
            f"independent factor recomputation mismatch for {label}: "
            f"{actual} != {expected}"
        )


def _independent_factor_recomputation(
    row: Mapping[str, Any],
    factors: Mapping[str, np.ndarray],
    *,
    clean: np.ndarray,
    observed: np.ndarray,
    truth_modes: np.ndarray,
    truth_dynamics: np.ndarray,
    time: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    keys = row["factor_keys"]
    _require_exact_keys(keys, {"A", "B", "step"}, label="factor key record")
    a = np.asarray(factors[keys["A"]], dtype=np.float64)
    b = np.asarray(factors[keys["B"]], dtype=np.float64)
    step_array = np.asarray(factors[keys["step"]], dtype=np.float64)
    expected_shape = (int(config["n_timesteps"]), int(config["n_components"]))
    if a.shape != expected_shape or b.shape != expected_shape or step_array.shape != (1,):
        raise ValueError("factor NPZ A/B/step shape mismatch")
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(step_array).all():
        raise ValueError("factor NPZ contains non-finite values")
    step = float(step_array[0])
    if step <= 0.0:
        raise ValueError("factor NPZ step must be positive")
    _close(step, row["factor_step_size"], label="factor step")
    if not np.allclose(a.T @ a, np.eye(3), rtol=2e-10, atol=2e-10):
        raise ValueError("factor A is not orthonormal")

    phi = observed @ b
    reconstruction = phi @ a.T
    residual = observed - reconstruction
    objective = 0.5 * float(np.sum(residual * residual)) + float(
        config["alpha"]
    ) * float(np.sum(np.abs(b)))
    _close_recomputed(
        objective,
        row["final_full_objective_recomputed"],
        label="full objective",
    )
    gradient = observed.T @ (observed @ (b - a))
    prox = _soft_threshold(
        b - step * gradient, step * float(config["alpha"])
    )
    stationarity = float(np.linalg.norm((b - prox) / step))
    if row["method"] != "PCA":
        _close_recomputed(
            stationarity,
            row["final_full_stationarity"],
            label="full stationarity",
        )

    clean_error = float(np.linalg.norm(clean - reconstruction) / np.linalg.norm(clean))
    observed_error = float(
        np.linalg.norm(observed - reconstruction) / np.linalg.norm(observed)
    )
    _close_recomputed(
        clean_error,
        row["clean_relative_reconstruction_error"],
        label="clean reconstruction error",
    )
    _close_recomputed(
        observed_error,
        row["observed_relative_reconstruction_error"],
        label="observed reconstruction error",
    )
    frame_errors = np.sqrt(np.mean((reconstruction - clean) ** 2, axis=0)) / float(
        np.sqrt(np.mean(clean * clean))
    )
    worst = int(np.argmax(frame_errors))
    expected_frame = {
        "n_frames": int(frame_errors.size),
        "median": float(np.median(frame_errors)),
        "p95": float(np.percentile(frame_errors, 95.0)),
        "max": float(frame_errors[worst]),
        "worst_frame_index": worst,
        "worst_frame_time": float(time[worst]),
    }
    actual_frame = row["full_frame_normalized_rmse"]
    _require_exact_keys(
        actual_frame,
        {"n_frames", "median", "p95", "max", "worst_frame_index", "worst_frame_time"},
        label="full-frame summary",
    )
    if int(actual_frame["n_frames"]) != expected_frame["n_frames"] or int(
        actual_frame["worst_frame_index"]
    ) != worst:
        raise ValueError("independent factor frame index/count mismatch")
    for field in ("median", "p95", "max", "worst_frame_time"):
        _close_recomputed(actual_frame[field], expected_frame[field], label=f"frame {field}")

    aligned_modes, aligned_dynamics, permutation, signs = _align_factors(
        truth_modes, phi, a
    )
    spatial = _paired_cosines(truth_modes, aligned_modes)
    temporal = _paired_cosines(truth_dynamics, aligned_dynamics)
    recovery = row["mode_and_dynamics_recovery"]
    _require_exact_keys(
        recovery,
        {
            "spatial_mode_cosine_per_component",
            "spatial_mode_mean_cosine",
            "spatial_mode_worst_cosine",
            "temporal_dynamics_cosine_per_component",
            "temporal_dynamics_mean_cosine",
            "temporal_dynamics_worst_cosine",
            "component_permutation",
            "component_signs",
            "amplitude_refit",
        },
        label="mode/dynamics recovery",
    )
    if recovery["component_permutation"] != permutation or recovery["component_signs"] != signs:
        raise ValueError("independent factor alignment differs from report")
    for actual_values, expected_values, label in (
        (recovery["spatial_mode_cosine_per_component"], spatial, "spatial cosine"),
        (recovery["temporal_dynamics_cosine_per_component"], temporal, "temporal cosine"),
    ):
        if not np.allclose(actual_values, expected_values, rtol=2e-10, atol=2e-11):
            raise ValueError(f"independent factor recomputation mismatch for {label}")
    for field, expected in (
        ("spatial_mode_mean_cosine", float(np.mean(spatial))),
        ("spatial_mode_worst_cosine", float(np.min(spatial))),
        ("temporal_dynamics_mean_cosine", float(np.mean(temporal))),
        ("temporal_dynamics_worst_cosine", float(np.min(temporal))),
    ):
        _close_recomputed(recovery[field], expected, label=field)

    localization = _localization(
        truth_modes, aligned_modes, float(config["support_threshold"])
    )
    if set(row["localization"]) != set(localization):
        raise ValueError("localization exact schema mismatch")
    for field in (
        "relative_truth_support_threshold",
        "off_support_energy_fraction_mean",
        "cross_talk_energy_fraction_mean",
    ):
        _close_recomputed(row["localization"][field], localization[field], label=field)
    if row["localization"]["truth_support_pixels_per_component"] != localization[
        "truth_support_pixels_per_component"
    ]:
        raise ValueError("independent truth support size mismatch")
    for field in (
        "off_support_energy_fraction_per_component",
        "cross_talk_energy_fraction_per_component",
    ):
        if not np.allclose(
            row["localization"][field], localization[field], rtol=2e-10, atol=2e-11
        ):
            raise ValueError(f"independent factor recomputation mismatch for {field}")
    sparsity = float(np.mean(np.abs(b) <= float(config["loading_zero_threshold"])))
    _close_recomputed(
        row["loading_sparsity_at_threshold"], sparsity, label="loading sparsity"
    )
    return np.asarray(frame_errors, dtype=np.float64)


def _bootstrap_interval(
    values: np.ndarray,
    *,
    statistic: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    observed = float(np.mean(values) if statistic == "mean" else np.median(values))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    samples = values[indices]
    estimates = (
        np.mean(samples, axis=1)
        if statistic == "mean"
        else np.median(samples, axis=1)
    )
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return observed, float(lower), float(upper)


def _independent_gate_recomputation(
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    by_key = {
        (float(row["snr_db"]), int(row["noise_seed"]), str(row["method"])): row
        for row in records
    }
    strata: dict[str, Any] = {}
    epsilon = np.finfo(np.float64).eps
    for stratum_index, (snr_db, seeds) in enumerate(config["snr_seed_bank"]):
        pairs = [
            (
                by_key[(float(snr_db), int(seed), "VP-SPCA")],
                by_key[(float(snr_db), int(seed), "RND-VP-SPCA")],
            )
            for seed in seeds
        ]
        values = {
            "clean_relative_error_ratio": np.asarray(
                [
                    rnd["clean_relative_reconstruction_error"]
                    / max(vp["clean_relative_reconstruction_error"], epsilon)
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "max_frame_rmse_ratio": np.asarray(
                [
                    rnd["full_frame_normalized_rmse"]["max"]
                    / max(vp["full_frame_normalized_rmse"]["max"], epsilon)
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "spatial_mean_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"]["spatial_mode_mean_cosine"]
                    - rnd["mode_and_dynamics_recovery"]["spatial_mode_mean_cosine"]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "spatial_worst_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"]["spatial_mode_worst_cosine"]
                    - rnd["mode_and_dynamics_recovery"]["spatial_mode_worst_cosine"]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "temporal_mean_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"][
                        "temporal_dynamics_mean_cosine"
                    ]
                    - rnd["mode_and_dynamics_recovery"][
                        "temporal_dynamics_mean_cosine"
                    ]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "temporal_worst_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"][
                        "temporal_dynamics_worst_cosine"
                    ]
                    - rnd["mode_and_dynamics_recovery"][
                        "temporal_dynamics_worst_cosine"
                    ]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "off_support_absolute_increment": np.asarray(
                [
                    rnd["localization"]["off_support_energy_fraction_mean"]
                    - vp["localization"]["off_support_energy_fraction_mean"]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "cross_talk_absolute_increment": np.asarray(
                [
                    rnd["localization"]["cross_talk_energy_fraction_mean"]
                    - vp["localization"]["cross_talk_energy_fraction_mean"]
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "loading_sparsity_absolute_difference": np.asarray(
                [
                    abs(
                        rnd["loading_sparsity_at_threshold"]
                        - vp["loading_sparsity_at_threshold"]
                    )
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "relative_full_objective_excess": np.asarray(
                [
                    (
                        rnd["final_full_objective_recomputed"]
                        - vp["final_full_objective_recomputed"]
                    )
                    / max(abs(vp["final_full_objective_recomputed"]), epsilon)
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
            "complete_method_runtime_speedup": np.asarray(
                [
                    vp["complete_method_runtime_seconds"]
                    / max(rnd["complete_method_runtime_seconds"], epsilon)
                    for vp, rnd in pairs
                ],
                dtype=float,
            ),
        }
        metrics: dict[str, Any] = {}
        for metric_index, (name, vector) in enumerate(values.items()):
            statistic = (
                "median" if name == "complete_method_runtime_speedup" else "mean"
            )
            estimate, lower, upper = _bootstrap_interval(
                vector,
                statistic=statistic,
                draws=int(config["bootstrap_draws"]),
                seed=int(config["bootstrap_seed"]) + 100 * stratum_index + metric_index,
            )
            bound, operator, threshold = THRESHOLDS[name]
            evaluated = upper if bound == "upper" else lower
            passed = evaluated <= threshold if operator == "<=" else evaluated > threshold
            metrics[name] = {
                "paired_values": vector,
                "statistic": statistic,
                "estimate": estimate,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "evaluated_ci_bound": evaluated,
                "required_operator": operator,
                "threshold": threshold,
                "passed": bool(passed),
            }
        solver_converged = all(
            bool(vp["converged"])
            and bool(vp["stationarity_met"])
            and bool(rnd["converged"])
            and bool(rnd["stationarity_met"])
            for vp, rnd in pairs
        )
        eligible = config["profile"] == "formal_confirmation" and len(pairs) == 20
        thresholds_passed = all(metric["passed"] for metric in metrics.values())
        strict = bool(eligible and thresholds_passed and solver_converged)
        status = (
            "smoke_not_evidence"
            if not eligible
            else (
                "fixed_budget_gate_failed"
                if not thresholds_passed
                else (
                    "fixed_budget_quality_matched_nonconverged"
                    if not solver_converged
                    else "strict_converged_quality_matched_and_speedup"
                )
            )
        )
        key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        strata[key] = {
            "n_paired_seeds": len(pairs),
            "formal_gate_eligible": eligible,
            "metrics": metrics,
            "all_predeclared_thresholds_passed": thresholds_passed,
            "all_vp_and_randomized_solvers_converged": solver_converged,
            "strict_claim_qualified": strict,
            "claim_status": status,
        }
    overall = bool(
        config["profile"] == "formal_confirmation"
        and all(item["strict_claim_qualified"] for item in strata.values())
    )
    return {"strata": strata, "overall_strict_claim_qualified": overall}


def _validate_method_summaries(
    report: Mapping[str, Any],
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    extractors = {
        "clean_relative_reconstruction_error": lambda row: row[
            "clean_relative_reconstruction_error"
        ],
        "max_frame_normalized_rmse": lambda row: row[
            "full_frame_normalized_rmse"
        ]["max"],
        "spatial_mode_mean_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["spatial_mode_mean_cosine"],
        "spatial_mode_worst_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["spatial_mode_worst_cosine"],
        "temporal_dynamics_mean_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["temporal_dynamics_mean_cosine"],
        "temporal_dynamics_worst_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["temporal_dynamics_worst_cosine"],
        "off_support_energy_fraction_mean": lambda row: row["localization"][
            "off_support_energy_fraction_mean"
        ],
        "cross_talk_energy_fraction_mean": lambda row: row["localization"][
            "cross_talk_energy_fraction_mean"
        ],
        "loading_sparsity_at_threshold": lambda row: row[
            "loading_sparsity_at_threshold"
        ],
        "complete_method_runtime_seconds": lambda row: row[
            "complete_method_runtime_seconds"
        ],
        "final_full_objective_recomputed": lambda row: row[
            "final_full_objective_recomputed"
        ],
    }
    summaries = report.get("method_summaries_by_snr", {})
    for snr_db, seeds in config["snr_seed_bank"]:
        snr_key = str(int(snr_db) if float(snr_db).is_integer() else snr_db)
        for method in METHODS:
            selected = [
                row
                for row in records
                if float(row["snr_db"]) == float(snr_db) and row["method"] == method
            ]
            summary = summaries.get(snr_key, {}).get(method, {})
            if summary.get("n_cases") != len(seeds):
                raise ValueError("method-summary case count mismatch")
            expected_convergence = float(
                np.mean([bool(row["converged"]) for row in selected])
            )
            _close(
                summary.get("converged_fraction"),
                expected_convergence,
                label="method-summary convergence fraction",
            )
            for name, extractor in extractors.items():
                values = np.asarray([extractor(row) for row in selected], dtype=float)
                expected = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p05": float(np.percentile(values, 5.0)),
                    "p95": float(np.percentile(values, 95.0)),
                }
                actual = summary.get(name, {})
                for field, value in expected.items():
                    _close(
                        actual.get(field),
                        value,
                        label=f"method-summary {snr_key}/{method}/{name}/{field}",
                    )


def _render_summary_pixels(
    records: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("PCA", "VP-SPCA", "RND-VP-SPCA")
    colors = {"PCA": "#6b7280", "VP-SPCA": "#2563eb", "RND-VP-SPCA": "#15803d"}
    extractors = (
        ("Clean relative error ↓", lambda row: row["clean_relative_reconstruction_error"], False),
        ("Worst-frame normalized RMSE ↓", lambda row: row["full_frame_normalized_rmse"]["max"], False),
        ("Worst spatial-mode cosine ↑", lambda row: row["mode_and_dynamics_recovery"]["spatial_mode_worst_cosine"], False),
        ("Worst temporal-dynamics cosine ↑", lambda row: row["mode_and_dynamics_recovery"]["temporal_dynamics_worst_cosine"], False),
        ("Mean off-support energy ↓", lambda row: row["localization"]["off_support_energy_fraction_mean"], False),
        ("Mean cross-talk energy ↓", lambda row: row["localization"]["cross_talk_energy_fraction_mean"], False),
        (f"Loading sparsity (|B|≤{float(config['loading_zero_threshold']):g})", lambda row: row["loading_sparsity_at_threshold"], False),
        ("Complete method runtime (s, log) ↓", lambda row: row["complete_method_runtime_seconds"], True),
    )
    snrs = [float(snr) for snr, _ in config["snr_seed_bank"]]
    x = np.arange(len(snrs), dtype=float)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        for axis, (title, extractor, log_scale) in zip(axes.ravel(), extractors):
            for method_index, method in enumerate(methods):
                means: list[float] = []
                lower: list[float] = []
                upper: list[float] = []
                offset = (method_index - 1) * 0.08
                for snr_index, snr_db in enumerate(snrs):
                    values = np.asarray(
                        [
                            extractor(row)
                            for row in records
                            if row["method"] == method and float(row["snr_db"]) == snr_db
                        ],
                        dtype=float,
                    )
                    means.append(float(np.mean(values)))
                    lower.append(float(np.percentile(values, 5.0)))
                    upper.append(float(np.percentile(values, 95.0)))
                    axis.scatter(
                        np.full(values.size, x[snr_index] + offset),
                        values,
                        s=15,
                        alpha=0.28,
                        color=colors[method],
                    )
                mean_array = np.asarray(means)
                axis.errorbar(
                    x + offset,
                    mean_array,
                    yerr=np.vstack(
                        [mean_array - np.asarray(lower), np.asarray(upper) - mean_array]
                    ),
                    marker="o",
                    linewidth=1.6,
                    capsize=3,
                    color=colors[method],
                    label=method,
                )
            axis.set_title(title)
            axis.set_xticks(x, [f"{snr:g}" for snr in snrs])
            axis.set_xlabel("SNR (dB)")
            if log_scale:
                axis.set_yscale("log")
        axes[0, 0].legend(loc="best", fontsize=8)
        convergence = all(
            bool(row["converged"])
            for row in records
            if row["method"] in {"VP-SPCA", "RND-VP-SPCA"}
        )
        qualification = (
            "converged strict evaluation"
            if convergence
            else "nonconverged solver present: fixed-budget evidence only"
        )
        figure.suptitle(
            "Original Figure 5 under locked dense full-rank noise\n"
            f"alpha=9, rank=3, no truth tuning; {qualification}",
            fontsize=14,
        )
        buffer = BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=int(config["dpi"]),
            metadata={"Software": "vp-rspca frozen Figure 5 stress"},
        )
        plt.close(figure)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image.load()
        return np.asarray(image).copy()


def _validate_summary_png(
    path: Path,
    records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        expected_size = (20 * int(config["dpi"]), 10 * int(config["dpi"]))
        if image.size != expected_size or image.mode != "RGBA":
            raise ValueError("summary PNG dimensions or mode differ from the lock")
        actual_pixels = np.asarray(image).copy()
    expected_pixels = _render_summary_pixels(records, config)
    if actual_pixels.shape != expected_pixels.shape or not np.array_equal(
        actual_pixels, expected_pixels
    ):
        raise ValueError("summary PNG pixels differ from independent record redraw")


def validate_frozen_noise_stress_output(
    output_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    allow_relocated_copy: bool = False,
    expected_lock_sha256: str | None = None,
    expected_output_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate closure, protocol semantics, rows, hashes, and source anchors."""

    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("output path must be a real directory, not a symlink")
    entries = list(output.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"output contains a non-regular entry: {entry.name}")
    if {entry.name for entry in entries} != OUTPUT_NAMES:
        raise ValueError("output directory does not have exact frozen closure")

    actual_output_manifest_sha256 = _file_sha256(output / OUTPUT_MANIFEST_NAME)
    manifest = _strict_json(output / OUTPUT_MANIFEST_NAME)
    _verify_body_digest(
        manifest, "output_manifest_body_sha256", label="output manifest"
    )
    if manifest.get("self_excluding") is not True:
        raise ValueError("output manifest must be self-excluding")
    if set(manifest.get("files", {})) != PAYLOAD_NAMES:
        raise ValueError("output manifest payload set mismatch")
    for name, record in manifest["files"].items():
        path = output / name
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"size mismatch for {name}")

    report = _strict_json(output / REPORT_NAME)
    lock = _strict_json(output / LOCK_NAME)
    source = _strict_json(output / SOURCE_MANIFEST_NAME)
    data = _strict_json(output / DATA_MANIFEST_NAME)
    receipt = _strict_json(output / EXECUTION_RECEIPT_NAME)
    seed_registry = _strict_json(output / SEED_REGISTRY_NAME)
    _verify_body_digest(report, "report_body_sha256", label="report")
    _verify_body_digest(lock, "protocol_lock_body_sha256", label="protocol lock")
    _verify_body_digest(
        source, "source_manifest_body_sha256", label="source manifest"
    )
    _verify_body_digest(data, "data_manifest_body_sha256", label="data manifest")
    _verify_body_digest(
        receipt, "execution_receipt_body_sha256", label="execution receipt"
    )
    _verify_body_digest(
        seed_registry, "seed_registry_body_sha256", label="seed registry"
    )
    _assert_finite_tree(report, "report")
    _assert_finite_tree(lock, "lock")
    _assert_finite_tree(data, "data")
    _assert_finite_tree(receipt, "receipt")
    _assert_finite_tree(seed_registry, "seed registry")

    profile = lock.get("profile")
    hash_profile = (
        "formal_confirmation"
        if (
            profile == "formal_confirmation"
            or seed_registry.get("registry_kind")
            == "create_once_global_formal_noise_seed_bank_claim"
            or receipt.get("receipt_kind")
            == "formal_lock_consumed_before_any_solver_fit"
        )
        else str(profile)
    )
    _validate_expected_formal_file_hashes(
        hash_profile,
        actual_lock_sha256=_file_sha256(output / LOCK_NAME),
        actual_output_manifest_sha256=actual_output_manifest_sha256,
        expected_lock_sha256=expected_lock_sha256,
        expected_output_manifest_sha256=expected_output_manifest_sha256,
    )
    expected_config = _expected_config(profile)
    if lock.get("config") != expected_config or report.get("config") != expected_config:
        raise ValueError("profile configuration differs from the external anchor")
    if report.get("profile") != profile or data.get("profile") != profile:
        raise ValueError("profile identity mismatch across artifacts")
    _validate_gate_specification(lock.get("gate_specification", {}))
    report_spec = report.get("stratified_randomized_vs_vp_gates", {}).get(
        "gate_specification", {}
    )
    _validate_gate_specification(report_spec)
    if lock.get("protocol", {}).get("truth_used_during_fit") is not False:
        raise ValueError("truth-isolation contract changed")
    if lock.get("protocol", {}).get("parameter_selection") != "none":
        raise ValueError("parameter selection is forbidden")
    if lock.get("protocol", {}).get("formal_cli_override_surface") != "none":
        raise ValueError("formal override surface is forbidden")
    if report.get("protocol") != lock.get("protocol"):
        raise ValueError("report protocol differs from its lock")
    if report.get("execution_authorization") != lock.get("execution_authorization"):
        raise ValueError("report execution authorization differs from its lock")

    links = {
        "protocol_lock_body_sha256": lock["protocol_lock_body_sha256"],
        "source_manifest_body_sha256": source["source_manifest_body_sha256"],
        "data_manifest_body_sha256": data["data_manifest_body_sha256"],
    }
    for field, digest in links.items():
        if report.get(field) != digest:
            raise ValueError(f"report {field} link mismatch")
    if report.get("protocol_lock_file_sha256") != _file_sha256(output / LOCK_NAME):
        raise ValueError("report protocol-lock file SHA-256 mismatch")
    if report.get("execution_receipt_body_sha256") != receipt.get(
        "execution_receipt_body_sha256"
    ):
        raise ValueError("report execution-receipt link mismatch")
    if report.get("seed_registry_body_sha256") != seed_registry.get(
        "seed_registry_body_sha256"
    ):
        raise ValueError("report seed-registry link mismatch")
    if receipt.get("protocol_lock_file_sha256") != _file_sha256(output / LOCK_NAME):
        raise ValueError("execution receipt lock-file link mismatch")
    if receipt.get("protocol_lock_body_sha256") != lock.get(
        "protocol_lock_body_sha256"
    ):
        raise ValueError("execution receipt lock-body link mismatch")
    if lock.get("source_manifest_body_sha256") != source[
        "source_manifest_body_sha256"
    ]:
        raise ValueError("lock-to-source link mismatch")
    if lock.get("data_manifest_body_sha256") != data["data_manifest_body_sha256"]:
        raise ValueError("lock-to-data link mismatch")
    if source.get("frozen_shared_core_sha256") != FROZEN_SHARED_CORE_SHA256:
        raise ValueError("shared-core freeze declaration mismatch")
    source_files = source.get("files", {})
    if set(source_files) != set(EXPECTED_SOURCE_ROLE_SCHEMA):
        raise ValueError("source manifest exact path schema mismatch")
    if {
        relative: record.get("role") for relative, record in source_files.items()
    } != EXPECTED_SOURCE_ROLE_SCHEMA:
        raise ValueError("source manifest exact path/role schema mismatch")
    for relative, record in source_files.items():
        _require_exact_keys(
            record,
            {"sha256", "size_bytes", "role", "module_names"},
            label=f"source record {relative}",
        )
    for relative, digest in FROZEN_SHARED_CORE_SHA256.items():
        if source.get("files", {}).get(relative, {}).get("sha256") != digest:
            raise ValueError(f"shared-core source record mismatch: {relative}")
    if source.get("federated_execution_path_used") is not False:
        raise ValueError("federated execution is outside this protocol")

    bank = expected_config["snr_seed_bank"]
    expected_pairs = {
        (float(snr), int(seed)) for snr, seeds in bank for seed in seeds
    }
    expected_case_count = len(expected_pairs)
    _require_exact_keys(data, DATA_MANIFEST_KEYS, label="data manifest")
    clean, truth_modes, truth_dynamics, truth_time = _independent_figure5_generator(
        expected_config
    )
    if data.get("clean_shape") != list(clean.shape):
        raise ValueError("independently regenerated clean shape differs from data lock")
    if data.get("clean_array_sha256") != _array_sha256(clean):
        raise ValueError("independently regenerated clean hash differs from data lock")
    data_cases = data.get("cases", [])
    if len(data_cases) != expected_case_count:
        raise ValueError("data manifest case count mismatch")
    if {
        (float(row.get("snr_db")), int(row.get("noise_seed"))) for row in data_cases
    } != expected_pairs:
        raise ValueError("data manifest seed bank mismatch")
    for row in data_cases:
        _require_exact_keys(row, DATA_CASE_KEYS, label="data case")
        _require_exact_keys(
            row["full_column_rank_witness"],
            {"leading_square_shape", "rank", "minimum_singular_value", "passed"},
            label="data rank witness",
        )
        _require_exact_keys(
            row["array_sha256"],
            {"clean", "noise", "observed"},
            label="data array hashes",
        )
        snr_db = float(row["snr_db"])
        noise_seed = int(row["noise_seed"])
        noise, observed, achieved, witness = _independent_noisy_case(
            clean, snr_db, noise_seed
        )
        if row["shape"] != list(observed.shape):
            raise ValueError("independently regenerated observed shape mismatch")
        if row["distribution"] != "iid_standard_normal_scaled_to_exact_frobenius_snr":
            raise ValueError("data noise distribution declaration changed")
        _close_recomputed(row["target_snr_db"], snr_db, label="target SNR")
        _close_recomputed(row["achieved_snr_db"], achieved, label="achieved SNR")
        if float(row["noise_nonzero_fraction"]) != float(np.mean(noise != 0.0)):
            raise ValueError("independently regenerated dense-noise fraction mismatch")
        if row["array_sha256"] != {
            "clean": _array_sha256(clean),
            "noise": _array_sha256(noise),
            "observed": _array_sha256(observed),
        }:
            raise ValueError("independently regenerated data array hash mismatch")
        if row["full_column_rank_witness"]["leading_square_shape"] != witness[
            "leading_square_shape"
        ] or row["full_column_rank_witness"]["rank"] != witness["rank"] or row[
            "full_column_rank_witness"
        ]["passed"] is not witness["passed"]:
            raise ValueError("independently regenerated full-rank witness mismatch")
        _close_recomputed(
            row["full_column_rank_witness"]["minimum_singular_value"],
            witness["minimum_singular_value"],
            label="rank witness minimum singular value",
        )
    if data_cases:
        del noise, observed

    records = report.get("case_records", [])
    if len(records) != expected_case_count * len(METHODS):
        raise ValueError("report method-case count mismatch")
    with np.load(output / FACTORS_NAME, allow_pickle=False) as factor_archive:
        factor_arrays = {
            key: np.asarray(factor_archive[key]).copy() for key in factor_archive.files
        }
    expected_factor_keys = {
        key for row in records for key in row.get("factor_keys", {}).values()
    }
    if set(factor_arrays) != expected_factor_keys:
        raise ValueError("factor NPZ exact key closure mismatch")
    if any(array.dtype != np.dtype("float64") for array in factor_arrays.values()):
        raise ValueError("factor NPZ arrays must use exact float64 dtype")
    expected_factor_nbytes = len(records) * (
        2 * int(expected_config["n_timesteps"]) * int(expected_config["n_components"])
        + 1
    ) * np.dtype("float64").itemsize
    if sum(array.nbytes for array in factor_arrays.values()) != expected_factor_nbytes:
        raise ValueError("factor NPZ uncompressed payload size differs from exact contract")
    independently_recomputed_frames: dict[tuple[float, int, str], np.ndarray] = {}
    sketch_residuals: dict[tuple[float, int], tuple[float, float]] = {}
    cached_pair: tuple[float, int] | None = None
    cached_observed: np.ndarray | None = None
    grouped: dict[tuple[float, int], set[str]] = {}
    for row in records:
        _require_exact_keys(row, CASE_RECORD_KEYS, label="report case record")
        pair = (float(row["snr_db"]), int(row["noise_seed"]))
        grouped.setdefault(pair, set()).add(str(row["method"]))
        if row.get("mode_and_dynamics_recovery", {}).get("amplitude_refit") is not False:
            raise ValueError("post-hoc amplitude fitting is forbidden")
        if float(row.get("tolerance")) != 1e-6 or float(
            row.get("stationarity_tolerance")
        ) != 1e-6:
            raise ValueError("every method row tolerance must equal locked 1e-6")
        if row.get("full_frame_normalized_rmse", {}).get("n_frames") != expected_config[
            "n_timesteps"
        ]:
            raise ValueError("full-frame summary is incomplete")
        if not 0.0 <= float(row["loading_sparsity_at_threshold"]) <= 1.0:
            raise ValueError("invalid loading sparsity")
        if float(row.get("loading_zero_threshold")) != float(
            expected_config["loading_zero_threshold"]
        ):
            raise ValueError("loading-zero threshold differs from the lock")
        recovery = row.get("mode_and_dynamics_recovery", {})
        spatial_values = np.asarray(
            recovery.get("spatial_mode_cosine_per_component", []), dtype=float
        )
        temporal_values = np.asarray(
            recovery.get("temporal_dynamics_cosine_per_component", []), dtype=float
        )
        if spatial_values.shape != (3,) or temporal_values.shape != (3,):
            raise ValueError("mode/dynamics component cosine count changed")
        _close(
            recovery["spatial_mode_mean_cosine"],
            float(np.mean(spatial_values)),
            label="spatial cosine mean",
        )
        _close(
            recovery["spatial_mode_worst_cosine"],
            float(np.min(spatial_values)),
            label="spatial cosine worst",
        )
        _close(
            recovery["temporal_dynamics_mean_cosine"],
            float(np.mean(temporal_values)),
            label="temporal cosine mean",
        )
        _close(
            recovery["temporal_dynamics_worst_cosine"],
            float(np.min(temporal_values)),
            label="temporal cosine worst",
        )
        localization = row.get("localization", {})
        off_support = np.asarray(
            localization.get("off_support_energy_fraction_per_component", []),
            dtype=float,
        )
        cross_talk = np.asarray(
            localization.get("cross_talk_energy_fraction_per_component", []),
            dtype=float,
        )
        if off_support.shape != (3,) or cross_talk.shape != (3,):
            raise ValueError("localization component count changed")
        _close(
            localization["off_support_energy_fraction_mean"],
            float(np.mean(off_support)),
            label="off-support mean",
        )
        _close(
            localization["cross_talk_energy_fraction_mean"],
            float(np.mean(cross_talk)),
            label="cross-talk mean",
        )
        final_stationarity = row.get("final_full_stationarity")
        expected_stationarity_met = (
            True
            if final_stationarity is None
            else float(final_stationarity) <= float(row["stationarity_tolerance"])
        )
        if bool(row.get("stationarity_met")) is not expected_stationarity_met:
            raise ValueError("stationarity_met disagrees with final stationarity")
        if row["method"] != "PCA" and final_stationarity is None:
            raise ValueError("sparse solver final stationarity is missing")
        if bool(row["full_converged"]) is not bool(row["converged"]):
            raise ValueError("full_converged and converged disagree")
        if row["method"] != "PCA" and bool(row["converged"]) is not bool(
            row["stationarity_met"]
        ):
            raise ValueError("sparse convergence disagrees with locked full stationarity")
        raw_runtime = float(row["raw_solver_fit_call_runtime_seconds"])
        complete_runtime = float(row["complete_method_runtime_seconds"])
        if raw_runtime <= 0.0 or complete_runtime <= 0.0:
            raise ValueError("method runtimes must be positive")
        expected_order = (
            ["PCA", "VP-SPCA", "RND-VP-SPCA"]
            if int(row["noise_seed"]) % 2
            else ["PCA", "RND-VP-SPCA", "VP-SPCA"]
        )
        if row.get("fit_execution_order") != expected_order:
            raise ValueError("locked alternating fit order changed")
        if row.get("fit_execution_position") != expected_order.index(row["method"]):
            raise ValueError("fit execution position is inconsistent")
        _require_exact_keys(
            row["runtime_components"],
            {
                "pca_and_sigma_preparation_seconds",
                "solver_fit_call_seconds",
                "randomized_fit_call_includes_range_compress_optimization_and_diagnostics",
                "used_by_speed_gate",
            },
            label="runtime components",
        )
        _require_exact_keys(
            row["factor_keys"], {"A", "B", "step"}, label="factor keys"
        )
        randomized = row.get("randomized_diagnostics", {})
        _require_exact_keys(
            randomized,
            {
                "sketch_residual",
                "sketch_residual_relative",
                "reported_full_objective",
                "reported_full_stationarity",
                "sketch_converged",
                "full_converged",
                "used_full_gradient",
                "internal_timing_diagnostics_not_used_for_speed_gate",
            },
            label="randomized diagnostics",
        )
        if row["method"] == "RND-VP-SPCA":
            if randomized.get("sketch_residual_relative") is None:
                raise ValueError("randomized sketch residual is missing")
            if randomized.get("reported_full_objective") is None:
                raise ValueError("randomized full objective is missing")
            _close(
                randomized["reported_full_objective"],
                row["final_full_objective_recomputed"],
                label="randomized full objective",
            )
            _close(
                randomized["reported_full_stationarity"],
                row["final_full_stationarity"],
                label="randomized full stationarity",
            )
            if bool(randomized["full_converged"]) is not bool(row["full_converged"]):
                raise ValueError("randomized nested full_converged disagrees with row")
            if randomized["used_full_gradient"] is not False:
                raise ValueError("locked randomized method unexpectedly used full gradient")
        elif any(value is not None for value in randomized.values()):
            raise ValueError("non-randomized row contains randomized diagnostics")
        if row["method"] == "PCA":
            if not (
                row["status"] == "closed-form truncated SVD"
                and int(row["n_iter"]) == 1
                and bool(row["converged"])
                and bool(row["full_converged"])
            ):
                raise ValueError("PCA status/convergence/n_iter semantics mismatch")
        elif row["method"] == "VP-SPCA":
            expected_status = (
                "converged" if bool(row["full_converged"]) else "maximum iterations reached"
            )
            if row["status"] != expected_status or not 1 <= int(row["n_iter"]) <= 500:
                raise ValueError("VP status/convergence/n_iter semantics mismatch")
            if not bool(row["converged"]) and int(row["n_iter"]) != 500:
                raise ValueError("nonconverged VP did not exhaust the locked budget")
        else:
            sketch_converged = bool(randomized["sketch_converged"])
            expected_status = (
                "sketch solve: "
                + ("converged" if sketch_converged else "maximum iterations reached")
                + "; full stationarity diagnostic: "
                + ("passed" if bool(row["full_converged"]) else "failed")
            )
            if row["status"] != expected_status or not 1 <= int(row["n_iter"]) <= 500:
                raise ValueError("randomized status/convergence/n_iter semantics mismatch")
            if not sketch_converged and int(row["n_iter"]) != 500:
                raise ValueError("randomized sketch did not exhaust the locked budget")

        if pair != cached_pair:
            _, cached_observed, _, _ = _independent_noisy_case(
                clean, pair[0], pair[1]
            )
            cached_pair = pair
        if cached_observed is None:  # pragma: no cover - guarded by assignment
            raise RuntimeError("independent observed-data cache is unavailable")
        observed = cached_observed
        key3 = (pair[0], pair[1], str(row["method"]))
        independently_recomputed_frames[key3] = _independent_factor_recomputation(
            row,
            factor_arrays,
            clean=clean,
            observed=observed,
            truth_modes=truth_modes,
            truth_dynamics=truth_dynamics,
            time=truth_time,
            config=expected_config,
        )
        if row["method"] == "RND-VP-SPCA":
            sketch = sketch_residuals.setdefault(
                pair, _independent_sketch_residual(observed, expected_config)
            )
            if not math.isclose(
                float(randomized["sketch_residual"]),
                sketch[0],
                rel_tol=2e-8,
                abs_tol=2e-9 * max(1.0, float(np.linalg.norm(observed))),
            ) or not math.isclose(
                float(randomized["sketch_residual_relative"]),
                sketch[1],
                rel_tol=2e-8,
                abs_tol=2e-9,
            ):
                raise ValueError("independently recomputed randomized sketch residual mismatch")
    if set(grouped) != expected_pairs or any(value != METHODS for value in grouped.values()):
        raise ValueError("method pairing is incomplete")
    json_by_key = {
        (float(row["snr_db"]), int(row["noise_seed"]), row["method"]): row
        for row in records
    }
    for snr_db, noise_seed in expected_pairs:
        pca_runtime = float(
            json_by_key[(snr_db, noise_seed, "PCA")][
                "raw_solver_fit_call_runtime_seconds"
            ]
        )
        for method in METHODS:
            row = json_by_key[(snr_db, noise_seed, method)]
            raw = float(row["raw_solver_fit_call_runtime_seconds"])
            expected_complete = raw + pca_runtime if method == "VP-SPCA" else raw
            _close(
                row["complete_method_runtime_seconds"],
                expected_complete,
                label=f"complete method runtime {snr_db}/{noise_seed}/{method}",
            )
    _validate_method_summaries(report, records, expected_config)

    case_rows = _read_csv(output / CASES_NAME)
    if len(case_rows) != expected_case_count * len(METHODS):
        raise ValueError("case CSV cardinality mismatch")
    if not case_rows or set(case_rows[0]) != CASE_CSV_FIELDS:
        raise ValueError("case CSV exact column schema mismatch")
    csv_groups: dict[tuple[float, int], set[str]] = {}
    for row in case_rows:
        key = (float(row["snr_db"]), int(row["noise_seed"]))
        csv_groups.setdefault(key, set()).add(row["method"])
    if set(csv_groups) != expected_pairs or any(value != METHODS for value in csv_groups.values()):
        raise ValueError("case CSV pairing is incomplete")
    if len(
        {
            (float(row["snr_db"]), int(row["noise_seed"]), row["method"])
            for row in case_rows
        }
    ) != len(case_rows):
        raise ValueError("case CSV contains duplicate method rows")
    for csv_row in case_rows:
        key = (
            float(csv_row["snr_db"]),
            int(csv_row["noise_seed"]),
            csv_row["method"],
        )
        row = json_by_key[key]
        if csv_row["status"] != row["status"]:
            raise ValueError("case CSV status differs from report JSON")
        if (csv_row["converged"] == "true") is not bool(row["converged"]):
            raise ValueError("case CSV convergence differs from report JSON")
        if (csv_row["full_converged"] == "true") is not bool(
            row["full_converged"]
        ):
            raise ValueError("case CSV full convergence differs from report JSON")
        if (csv_row["stationarity_met"] == "true") is not bool(
            row["stationarity_met"]
        ):
            raise ValueError("case CSV stationarity flag differs from report JSON")
        if int(csv_row["n_iter"]) != int(row["n_iter"]):
            raise ValueError("case CSV iteration count differs from report JSON")
        stationarity_text = csv_row["final_full_stationarity"]
        if row["final_full_stationarity"] is None:
            if stationarity_text != "":
                raise ValueError("case CSV PCA stationarity must be blank")
        else:
            _close(
                float(stationarity_text),
                row["final_full_stationarity"],
                label="case CSV final stationarity",
            )
        frame = row["full_frame_normalized_rmse"]
        recovery = row["mode_and_dynamics_recovery"]
        localization = row["localization"]
        randomized = row["randomized_diagnostics"]
        numeric_pairs = {
            "tolerance": row["tolerance"],
            "stationarity_tolerance": row["stationarity_tolerance"],
            "factor_step_size": row["factor_step_size"],
            "final_full_objective_recomputed": row[
                "final_full_objective_recomputed"
            ],
            "raw_solver_fit_call_runtime_seconds": row[
                "raw_solver_fit_call_runtime_seconds"
            ],
            "complete_method_runtime_seconds": row[
                "complete_method_runtime_seconds"
            ],
            "clean_relative_reconstruction_error": row[
                "clean_relative_reconstruction_error"
            ],
            "observed_relative_reconstruction_error": row[
                "observed_relative_reconstruction_error"
            ],
            "frame_rmse_median": frame["median"],
            "frame_rmse_p95": frame["p95"],
            "frame_rmse_max": frame["max"],
            "worst_frame_time": frame["worst_frame_time"],
            "loading_sparsity_at_threshold": row[
                "loading_sparsity_at_threshold"
            ],
            "spatial_mode_mean_cosine": recovery["spatial_mode_mean_cosine"],
            "spatial_mode_worst_cosine": recovery["spatial_mode_worst_cosine"],
            "temporal_dynamics_mean_cosine": recovery[
                "temporal_dynamics_mean_cosine"
            ],
            "temporal_dynamics_worst_cosine": recovery[
                "temporal_dynamics_worst_cosine"
            ],
            "off_support_energy_fraction_mean": localization[
                "off_support_energy_fraction_mean"
            ],
            "cross_talk_energy_fraction_mean": localization[
                "cross_talk_energy_fraction_mean"
            ],
        }
        for field, expected in numeric_pairs.items():
            _close(float(csv_row[field]), expected, label=f"case CSV {field}")
        for csv_field, factor_field in (
            ("factor_A_key", "A"),
            ("factor_B_key", "B"),
            ("factor_step_key", "step"),
        ):
            if csv_row[csv_field] != row["factor_keys"][factor_field]:
                raise ValueError("case CSV factor key differs from report JSON")
        if int(csv_row["worst_frame_index"]) != int(frame["worst_frame_index"]):
            raise ValueError("case CSV worst-frame index differs from report JSON")
        for field, expected in (
            ("last_solver_objective", row["last_solver_objective"]),
            ("sketch_residual", randomized["sketch_residual"]),
            ("sketch_residual_relative", randomized["sketch_residual_relative"]),
        ):
            if expected is None:
                if csv_row[field] != "":
                    raise ValueError(f"case CSV {field} must be blank")
            else:
                _close(float(csv_row[field]), expected, label=f"case CSV {field}")

    frame_rows = _read_csv(output / FRAMES_NAME)
    expected_frame_count = expected_case_count * len(METHODS) * expected_config[
        "n_timesteps"
    ]
    if len(frame_rows) != expected_frame_count:
        raise ValueError("full-frame CSV cardinality mismatch")
    if not frame_rows or set(frame_rows[0]) != FRAME_CSV_FIELDS:
        raise ValueError("full-frame CSV exact column schema mismatch")
    frame_groups: dict[tuple[float, int, str], set[int]] = {}
    frame_values: dict[tuple[float, int, str], dict[int, tuple[float, float]]] = {}
    for row in frame_rows:
        key = (float(row["snr_db"]), int(row["noise_seed"]), row["method"])
        frame_index = int(row["frame_index"])
        if frame_index in frame_values.setdefault(key, {}):
            raise ValueError("full-frame CSV contains a duplicate frame")
        timestamp = float(row["time"])
        error = float(row["normalized_rmse"])
        frame_values[key][frame_index] = (timestamp, error)
        frame_groups.setdefault(key, set()).add(frame_index)
        if not math.isfinite(float(row["normalized_rmse"])):
            raise ValueError("non-finite frame error")
    expected_group_count = expected_case_count * len(METHODS)
    if len(frame_groups) != expected_group_count:
        raise ValueError("full-frame CSV group count mismatch")
    expected_indices = set(range(expected_config["n_timesteps"]))
    if any(indices != expected_indices for indices in frame_groups.values()):
        raise ValueError("full-frame CSV indices are incomplete")
    expected_time = np.linspace(
        0.0, float(expected_config["total_time"]), expected_config["n_timesteps"]
    )
    for key, by_index in frame_values.items():
        ordered = [by_index[index] for index in range(expected_config["n_timesteps"])]
        times = np.asarray([item[0] for item in ordered], dtype=float)
        errors = np.asarray([item[1] for item in ordered], dtype=float)
        if not np.allclose(times, expected_time, rtol=2e-13, atol=2e-14):
            raise ValueError("full-frame CSV time grid changed")
        if not np.allclose(
            errors,
            independently_recomputed_frames[key],
            rtol=2e-10,
            atol=2e-11,
        ):
            raise ValueError("frame CSV differs from independent factor reconstruction")
        json_summary = json_by_key[key]["full_frame_normalized_rmse"]
        worst = int(np.argmax(errors))
        expected_summary = {
            "n_frames": errors.size,
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95.0)),
            "max": float(errors[worst]),
            "worst_frame_index": worst,
            "worst_frame_time": float(times[worst]),
        }
        if int(json_summary["n_frames"]) != expected_summary["n_frames"]:
            raise ValueError("report full-frame count differs from frame CSV")
        if int(json_summary["worst_frame_index"]) != worst:
            raise ValueError("report worst-frame index differs from frame CSV")
        for field in ("median", "p95", "max", "worst_frame_time"):
            _close(
                json_summary[field],
                expected_summary[field],
                label=f"frame summary {field}",
            )

    gates = report["stratified_randomized_vs_vp_gates"]
    if gates.get("pooling_across_snr_for_gate_forbidden") is not True:
        raise ValueError("SNR pooling prohibition changed")
    recomputed = _independent_gate_recomputation(records, expected_config)
    for snr, seeds in bank:
        key = str(int(snr) if float(snr).is_integer() else snr)
        stratum = gates.get("strata", {}).get(key)
        if not isinstance(stratum, Mapping):
            raise ValueError("missing SNR gate stratum")
        if stratum.get("n_paired_seeds") != len(seeds):
            raise ValueError("paired gate count mismatch")
        if set(stratum.get("metrics", {})) != set(THRESHOLDS):
            raise ValueError("gate result metric set mismatch")
        should_be_eligible = profile == "formal_confirmation"
        if stratum.get("formal_gate_eligible") is not should_be_eligible:
            raise ValueError("formal gate eligibility mismatch")
        expected_stratum = recomputed["strata"][key]
        for field in (
            "n_paired_seeds",
            "formal_gate_eligible",
            "all_predeclared_thresholds_passed",
            "all_vp_and_randomized_solvers_converged",
            "strict_claim_qualified",
            "claim_status",
        ):
            if stratum.get(field) != expected_stratum[field]:
                raise ValueError(f"independent gate recomputation mismatch: {key}/{field}")
        for name, metric in stratum["metrics"].items():
            expected_metric = expected_stratum["metrics"][name]
            actual_vector = np.asarray(metric.get("paired_values", []), dtype=float)
            if actual_vector.shape != expected_metric["paired_values"].shape or not np.allclose(
                actual_vector,
                expected_metric["paired_values"],
                rtol=2e-13,
                atol=2e-14,
            ):
                raise ValueError(f"paired gate vector mismatch: {key}/{name}")
            for field in (
                "estimate",
                "ci95_lower",
                "ci95_upper",
                "evaluated_ci_bound",
                "threshold",
            ):
                _close(
                    metric[field],
                    expected_metric[field],
                    label=f"gate {key}/{name}/{field}",
                )
            for field in ("statistic", "required_operator", "passed"):
                if metric.get(field) != expected_metric[field]:
                    raise ValueError(f"gate semantic mismatch: {key}/{name}/{field}")
    if gates.get("overall_strict_claim_qualified") is not recomputed[
        "overall_strict_claim_qualified"
    ]:
        raise ValueError("overall strict gate qualification mismatch")
    expected_overall_gate_status = (
        "strict_all_snr"
        if recomputed["overall_strict_claim_qualified"]
        else (
            "smoke_not_evidence"
            if profile != "formal_confirmation"
            else "report_each_failed_snr_boundary"
        )
    )
    if gates.get("overall_claim_status") != expected_overall_gate_status:
        raise ValueError("overall gate claim status mismatch")

    sparse_rows = [row for row in records if row["method"] != "PCA"]
    any_nonconverged = any(
        not bool(row["converged"]) or not bool(row["stationarity_met"])
        for row in sparse_rows
    )
    qualification = report.get("strict_qualification", {})
    all_sparse_converged = not any_nonconverged
    if qualification.get("all_vp_and_randomized_solvers_converged") is not all_sparse_converged:
        raise ValueError("report sparse-solver convergence summary mismatch")
    if qualification.get("overall_strict_claim_qualified") is not recomputed[
        "overall_strict_claim_qualified"
    ]:
        raise ValueError("report strict qualification mismatch")
    expected_claim_status = (
        "smoke_not_evidence"
        if profile != "formal_confirmation"
        else (
            "strict_converged_quality_matched_and_speedup_all_snr"
            if recomputed["overall_strict_claim_qualified"]
            else (
                "fixed_budget_only_due_to_solver_nonconvergence"
                if any_nonconverged
                else "predeclared_gate_failed_report_stratum_boundaries"
            )
        )
    )
    if qualification.get("claim_status") != expected_claim_status:
        raise ValueError("report claim status disagrees with independent recomputation")
    if profile == "smoke_not_evidence":
        if report.get("formal_confirmation") is not False:
            raise ValueError("smoke cannot be a formal confirmation")
        if qualification.get("claim_status") != "smoke_not_evidence":
            raise ValueError("smoke claim status is invalid")
        if receipt.get("receipt_kind") != "smoke_not_formal_lock_consumption":
            raise ValueError("smoke execution receipt kind is invalid")
        if receipt.get("created_with_exclusive_open_before_any_fit") is not False:
            raise ValueError("smoke receipt cannot claim formal exclusive consumption")
        _require_exact_keys(
            seed_registry, SMOKE_SEED_REGISTRY_KEYS, label="smoke seed registry"
        )
        if seed_registry != {
            "schema_version": 1,
            "registry_kind": "smoke_not_global_seed_bank_claim",
            "formal_snr_seed_bank": None,
            "created_with_exclusive_open_before_anchor_and_any_fit": False,
            "reuse_or_reissue_forbidden": False,
            "seed_registry_body_sha256": seed_registry.get(
                "seed_registry_body_sha256"
            ),
        }:
            raise ValueError("smoke seed-registry disclosure schema changed")
    else:
        if report.get("formal_confirmation") is not True or report.get("paper_scale") is not True:
            raise ValueError("formal output must be paper-scale confirmation")
        seed_audit = lock.get("fresh_seed_history_audit")
        if not isinstance(seed_audit, Mapping):
            raise ValueError("formal fresh-seed history audit is missing")
        seed_audit_body = dict(seed_audit)
        recorded_seed_audit_digest = seed_audit_body.pop(
            "seed_history_audit_body_sha256", None
        )
        if recorded_seed_audit_digest != _canonical_sha256(seed_audit_body):
            raise ValueError("formal fresh-seed history audit digest mismatch")
        if seed_audit.get("matches") != [] or seed_audit.get(
            "no_prior_outcome_artifact_matches"
        ) is not True:
            raise ValueError("formal seed bank was not fresh at freeze time")
        if seed_audit.get("searched_roots") != [
            "results",
            "assets/evaluation",
            "artifacts",
        ]:
            raise ValueError("formal freshness semantic roots changed")
        if seed_audit.get("inventory_sha256") != _canonical_sha256(
            {"files": seed_audit.get("inventory")}
        ):
            raise ValueError("formal freshness inventory digest mismatch")
        expected_seed_list = sorted(seed for _, seeds in FORMAL_BANK for seed in seeds)
        if seed_audit.get("locked_formal_seeds") != expected_seed_list:
            raise ValueError("formal fresh-seed history bank mismatch")
        if any_nonconverged and not str(qualification.get("claim_status", "")).startswith(
            "fixed_budget_only"
        ):
            raise ValueError("nonconverged formal output must be fixed-budget only")
        authorization = lock.get("execution_authorization", {})
        authorized_output = authorization.get("exclusive_output_dir_resolved")
        if not isinstance(authorized_output, str):
            raise ValueError("formal lock lacks its uniquely bound output path")
        if not allow_relocated_copy and authorized_output != str(output.resolve()):
            raise ValueError("formal output differs from the uniquely bound output path")
        if receipt.get("receipt_kind") != "formal_lock_consumed_before_any_solver_fit":
            raise ValueError("formal execution receipt kind is invalid")
        if receipt.get("created_with_exclusive_open_before_any_fit") is not True:
            raise ValueError("formal receipt lacks pre-fit exclusive-consumption proof")
        if receipt.get("lock_reuse_forbidden") is not True:
            raise ValueError("formal receipt does not forbid lock reuse")
        if receipt.get("exclusive_output_dir_resolved") != authorized_output:
            raise ValueError("formal receipt and lock output paths disagree")
        execution_seed_audit = receipt.get("pre_execution_fresh_seed_history_audit")
        if not isinstance(execution_seed_audit, Mapping):
            raise ValueError("formal receipt lacks its execution-time seed audit")
        execution_audit_body = dict(execution_seed_audit)
        execution_audit_digest = execution_audit_body.pop(
            "seed_history_audit_body_sha256", None
        )
        if execution_audit_digest != _canonical_sha256(execution_audit_body):
            raise ValueError("execution-time seed audit digest mismatch")
        if execution_seed_audit.get("matches") != [] or execution_seed_audit.get(
            "no_prior_outcome_artifact_matches"
        ) is not True:
            raise ValueError("formal seeds were not fresh immediately before execution")
        if execution_seed_audit.get("locked_formal_seeds") != expected_seed_list:
            raise ValueError("execution-time seed audit bank mismatch")
        if execution_seed_audit.get("inventory_sha256") != seed_audit.get(
            "inventory_sha256"
        ):
            raise ValueError("freshness inventory changed between freeze and execute")
        _require_exact_keys(
            seed_registry, FORMAL_SEED_REGISTRY_KEYS, label="formal seed registry"
        )
        if seed_registry.get("registry_kind") != (
            "create_once_global_formal_noise_seed_bank_claim"
        ):
            raise ValueError("formal create-once seed registry kind changed")
        if seed_registry.get("formal_snr_seed_bank") != FORMAL_BANK:
            raise ValueError("formal seed registry bank mismatch")
        if seed_registry.get("created_with_exclusive_open_before_anchor_and_any_fit") is not True:
            raise ValueError("formal seed registry lacks atomic pre-anchor claim")
        if seed_registry.get("reuse_or_reissue_forbidden") is not True:
            raise ValueError("formal seed registry permits reuse")
        if seed_registry.get("freshness_inventory_sha256") != seed_audit.get(
            "inventory_sha256"
        ):
            raise ValueError("seed registry freshness inventory link mismatch")
        if seed_registry.get("freshness_audit_body_sha256") != seed_audit.get(
            "seed_history_audit_body_sha256"
        ):
            raise ValueError("seed registry freshness-audit link mismatch")
        authorization = lock["execution_authorization"]
        if seed_registry.get("canonical_anchor_dir_resolved") != authorization.get(
            "formal_anchor_dir_resolved"
        ) or seed_registry.get("canonical_output_dir_resolved") != authorization.get(
            "exclusive_output_dir_resolved"
        ):
            raise ValueError("seed registry canonical paths disagree with formal lock")
        canonical_root = Path(str(seed_registry.get("canonical_formal_root_resolved")))
        if (
            Path(str(seed_registry.get("canonical_anchor_dir_resolved"))).parent
            != canonical_root
            or Path(str(seed_registry.get("canonical_output_dir_resolved"))).parent
            != canonical_root
        ):
            raise ValueError("seed registry anchor/output are outside canonical root")
        registry_file_sha = _file_sha256(output / SEED_REGISTRY_NAME)
        if lock.get("seed_registry_body_sha256") != seed_registry.get(
            "seed_registry_body_sha256"
        ) or receipt.get("seed_registry_body_sha256") != seed_registry.get(
            "seed_registry_body_sha256"
        ):
            raise ValueError("formal seed-registry body links disagree")
        if lock.get("seed_registry_file_sha256") != registry_file_sha or receipt.get(
            "seed_registry_file_sha256"
        ) != registry_file_sha:
            raise ValueError("formal seed-registry file links disagree")

        frozen_timing = lock.get("timing_environment")
        execution_timing = receipt.get("execution_timing_environment")
        if not isinstance(frozen_timing, Mapping) or not isinstance(
            execution_timing, Mapping
        ):
            raise ValueError("formal timing environment records are missing")
        if seed_registry.get("timing_environment_body_sha256") != frozen_timing.get(
            "timing_environment_body_sha256"
        ):
            raise ValueError("seed registry timing-environment link mismatch")
        for label, snapshot in (
            ("freeze", frozen_timing),
            ("execution", execution_timing),
        ):
            snapshot_body = dict(snapshot)
            snapshot_digest = snapshot_body.pop(
                "timing_environment_body_sha256", None
            )
            if snapshot_digest != _canonical_sha256(snapshot_body):
                raise ValueError(f"formal {label} timing environment digest mismatch")
            if snapshot.get("formal_timing_qualified") is not True or snapshot.get(
                "qualification_failures"
            ) != []:
                raise ValueError(f"formal {label} timing environment is unqualified")
            if snapshot.get("platform_system") != "Linux":
                raise ValueError("formal timing is not Linux AutoDL")
            if snapshot.get("autodl_timing_marker") != "1":
                raise ValueError("formal AutoDL timing marker is missing")
            repo_text = str(snapshot.get("repo_root_resolved", ""))
            if not repo_text.startswith("/root/autodl-tmp/dhy/"):
                raise ValueError("formal timing repository is outside AutoDL data disk")
            if snapshot.get("matplotlib_backend_environment", "").lower() != "agg":
                raise ValueError("formal Matplotlib backend was not fixed to Agg")
            thread_environment = snapshot.get("thread_environment", {})
            if set(thread_environment) != {
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            } or any(value != "1" for value in thread_environment.values()):
                raise ValueError("formal BLAS/OpenMP environment was not fixed to one")
            if len(snapshot.get("cpu_affinity", [])) != 1:
                raise ValueError("formal CPU affinity is not exactly one logical CPU")
            pools = snapshot.get("threadpool_info", [])
            if not pools or any(int(pool.get("num_threads", -1)) != 1 for pool in pools):
                raise ValueError("formal threadpoolctl backend is not single-threaded")
            if not snapshot.get("cpu_model") or not snapshot.get(
                "numpy_backend_configuration"
            ):
                raise ValueError("formal CPU/backend provenance is incomplete")
        if execution_timing != frozen_timing:
            raise ValueError("formal timing environment changed between freeze and execute")

    root = (
        Path(source_root).resolve()
        if source_root is not None
        else Path(__file__).resolve().parents[1]
    )
    for relative, record in source.get("files", {}).items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"anchored source missing or symlinked: {relative}")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"current source hash mismatch: {relative}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"current source size mismatch: {relative}")

    _validate_summary_png(output / FIGURE_NAME, records, expected_config)
    return {
        "valid": True,
        "profile": profile,
        "n_cases": expected_case_count,
        "n_method_records": len(records),
        "n_frame_rows": len(frame_rows),
        "claim_status": qualification.get("claim_status"),
        "output_manifest_sha256": actual_output_manifest_sha256,
        "relocated_copy": bool(
            profile == "formal_confirmation"
            and str(output.resolve())
            != lock.get("execution_authorization", {}).get(
                "exclusive_output_dir_resolved"
            )
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--allow-relocated-copy", action="store_true")
    parser.add_argument("--expected-lock-sha256", default=None)
    parser.add_argument("--expected-output-manifest-sha256", default=None)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = validate_frozen_noise_stress_output(
        arguments.output_dir,
        source_root=arguments.source_root,
        allow_relocated_copy=arguments.allow_relocated_copy,
        expected_lock_sha256=arguments.expected_lock_sha256,
        expected_output_manifest_sha256=arguments.expected_output_manifest_sha256,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

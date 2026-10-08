#!/usr/bin/env python3
"""Strict paired-seed summary for clean and corrupted SPCA benchmarks.

The input artifacts are the raw JSON files emitted by
``experiments/spca_benchmark.py`` in ``matched-sparsity`` mode.  This tool
validates the complete seed-by-method matrix and its data provenance before it
computes any statistics.  Scientific metrics use paired, seed-level percentile
bootstrap intervals for ``method - thresholded PCA`` contrasts.  Runtime is
reported descriptively and is deliberately excluded from the bootstrap.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from rspca.datasets import canonical_sha256


BASELINE_METHOD = "thresholded_pca_matched_sparsity"
EXPECTED_METHODS = (
    BASELINE_METHOD,
    "accelerated_vp_spca",
    "certified_randomized_vp_spca",
    "randomized_vp_spca",
    "robust_vp_spca",
    "vp_spca",
)
GENERAL_METRICS = (
    "relative_reconstruction_error",
    "explained_variance_ratio",
    "subspace_error",
    "component_mean_abs_correlation",
    "component_min_abs_correlation",
    "achieved_sparsity",
    "support_precision",
    "support_recall",
    "support_f1",
)
CORRUPTION_METRICS = (
    "outlier_precision",
    "outlier_recall",
    "outlier_f1",
    "outlier_roc_auc",
    "outlier_average_precision",
)
METRIC_DIRECTIONS = {
    "relative_reconstruction_error": "lower_is_better",
    "explained_variance_ratio": "higher_is_better",
    "subspace_error": "lower_is_better",
    "component_mean_abs_correlation": "higher_is_better",
    "component_min_abs_correlation": "higher_is_better",
    "achieved_sparsity": "descriptive_target_matching_quantity",
    "support_precision": "higher_is_better",
    "support_recall": "higher_is_better",
    "support_f1": "higher_is_better",
    "outlier_precision": "higher_is_better",
    "outlier_recall": "higher_is_better",
    "outlier_f1": "higher_is_better",
    "outlier_roc_auc": "higher_is_better",
    "outlier_average_precision": "higher_is_better",
}
CONFIG_FIELDS = frozenset(
    {
        "n_samples",
        "n_features",
        "rank",
        "support_fraction",
        "noise_std",
        "corruption_fraction",
        "corruption_scale",
        "corruption_distribution",
        "alphas",
        "selection",
        "target_sparsity",
        "repeats",
        "base_seed",
        "max_iter",
        "tol",
        "oversample",
        "n_power_iter",
        "randomized_full_gradient",
        "randomized_finalize_full_iters",
        "certified_residual_tolerance",
        "certified_full_refine_iters",
    }
)
DATA_HASH_FIELDS = (
    "observed_sha256",
    "clean_uncorrupted_sha256",
    "low_rank_truth_sha256",
    "true_loadings_sha256",
    "feature_support_sha256",
    "corruption_support_sha256",
    "corruption_values_sha256",
)
SHARED_REGIME_HASH_FIELDS = (
    "clean_uncorrupted_sha256",
    "low_rank_truth_sha256",
    "true_loadings_sha256",
    "feature_support_sha256",
)
CSV_FIELDS = (
    "analysis_manifest_sha256",
    "regime",
    "input_json_sha256",
    "config_sha256",
    "seed_manifest_sha256",
    "method",
    "metric",
    "metric_direction",
    "n_seeds",
    "absolute_mean",
    "absolute_sample_std",
    "converged_count",
    "convergence_rate",
    "inference_qualified",
    "qualified_pairs",
    "unqualified_pairs",
    "bootstrap_ci95_status",
    "summary_population",
    "runtime_n",
    "runtime_mean_seconds",
    "runtime_sample_std_seconds",
    "runtime_min_seconds",
    "runtime_max_seconds",
    "runtime_in_bootstrap",
    "baseline_method",
    "difference_direction",
    "n_pairs",
    "paired_mean_difference",
    "paired_sample_std_difference",
    "bootstrap_ci95_lower",
    "bootstrap_ci95_upper",
    "bootstrap_resamples",
    "bootstrap_seed",
    "bootstrap_unit",
    "bootstrap_plan_sha256",
)
_HEX_DIGITS = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _ValidatedRegime:
    name: str
    path: Path
    input_json_sha256: str
    config: dict[str, Any]
    config_sha256: str
    seed_manifest_sha256: str
    seeds: tuple[int, ...]
    run_provenance: dict[int, dict[str, Any]]
    records: dict[tuple[int, str], dict[str, Any]]


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant is forbidden: {value}")


def _reject_nonfinite_numbers(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {location}")


def _load_strict_json(path_value: Path) -> tuple[dict[str, Any], str]:
    path = Path(path_value)
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"input JSON is not UTF-8: {path}") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_no_duplicate_object,
            parse_constant=_invalid_json_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"input JSON must contain one top-level object: {path}")
    _reject_nonfinite_numbers(payload)
    return payload, hashlib.sha256(raw).hexdigest()


def _require(mapping: Mapping[str, Any], key: str, *, location: str) -> Any:
    if key not in mapping:
        raise ValueError(f"missing required field {location}.{key}")
    return mapping[key]


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest")
    return value


def _sample_summary(values: Sequence[float], *, name: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} requires at least two finite seed-level values")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array, dtype=np.float64)),
        "sample_std": float(np.std(array, ddof=1)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _validate_config(config_value: Any, *, path: Path) -> dict[str, Any]:
    if not isinstance(config_value, dict):
        raise ValueError(f"config must be an object: {path}")
    if set(config_value) != CONFIG_FIELDS:
        missing = sorted(CONFIG_FIELDS - set(config_value))
        extra = sorted(set(config_value) - CONFIG_FIELDS)
        raise ValueError(
            f"config schema mismatch in {path}; missing={missing}, extra={extra}"
        )
    config = dict(config_value)
    if config["selection"] != "matched-sparsity":
        raise ValueError(f"SPCA summary requires matched-sparsity selection: {path}")
    target = _finite(config["target_sparsity"], name=f"{path}:target_sparsity")
    if not 0.0 <= target <= 1.0:
        raise ValueError(f"target_sparsity must lie in [0, 1]: {path}")
    repeats = _integer(config["repeats"], name=f"{path}:repeats", minimum=2)
    _integer(config["base_seed"], name=f"{path}:base_seed")
    for key in ("n_samples", "n_features", "rank", "max_iter"):
        _integer(config[key], name=f"{path}:{key}", minimum=1)
    for key in (
        "oversample",
        "n_power_iter",
        "randomized_finalize_full_iters",
        "certified_full_refine_iters",
    ):
        _integer(config[key], name=f"{path}:{key}")
    if not isinstance(config["randomized_full_gradient"], bool):
        raise TypeError(f"randomized_full_gradient must be boolean: {path}")
    for key in (
        "support_fraction",
        "noise_std",
        "corruption_fraction",
        "corruption_scale",
        "tol",
        "certified_residual_tolerance",
    ):
        _finite(config[key], name=f"{path}:{key}")
    if not 0.0 < float(config["support_fraction"]) <= 1.0:
        raise ValueError(f"support_fraction must lie in (0, 1]: {path}")
    if float(config["noise_std"]) < 0.0:
        raise ValueError(f"noise_std must be non-negative: {path}")
    corruption_fraction = float(config["corruption_fraction"])
    if not 0.0 <= corruption_fraction <= 1.0:
        raise ValueError(f"corruption_fraction must lie in [0, 1]: {path}")
    corruption_scale = float(config["corruption_scale"])
    if corruption_scale < 0.0 or (
        corruption_fraction > 0.0 and corruption_scale == 0.0
    ):
        raise ValueError(f"invalid corruption_scale: {path}")
    if float(config["tol"]) <= 0.0:
        raise ValueError(f"tol must be positive: {path}")
    if float(config["certified_residual_tolerance"]) < 0.0:
        raise ValueError(f"certified_residual_tolerance must be non-negative: {path}")
    if config["rank"] >= config["n_samples"] or config["rank"] > config["n_features"]:
        raise ValueError(f"rank is incompatible with the matrix dimensions: {path}")
    support_size = max(
        1, int(round(float(config["support_fraction"]) * config["n_features"]))
    )
    if config["rank"] * support_size > config["n_features"]:
        raise ValueError(f"disjoint component supports do not fit: {path}")
    if config["corruption_distribution"] not in {
        "gaussian",
        "laplace",
        "rademacher",
    }:
        raise ValueError(f"unexpected corruption_distribution: {path}")
    alphas = config["alphas"]
    if not isinstance(alphas, list) or not alphas:
        raise ValueError(f"alphas must be a non-empty list: {path}")
    alpha_values = [_finite(value, name=f"{path}:alphas") for value in alphas]
    if any(value < 0.0 for value in alpha_values) or len(set(alpha_values)) != len(
        alpha_values
    ):
        raise ValueError(f"alphas must be unique and non-negative: {path}")
    return config


def _validate_regime(name: str, path: Path) -> _ValidatedRegime:
    payload, input_digest = _load_strict_json(path)
    expected_top_level = {
        "schema_version",
        "config",
        "provenance",
        "seed_manifest",
        "records",
    }
    if set(payload) != expected_top_level:
        missing = sorted(expected_top_level - set(payload))
        extra = sorted(set(payload) - expected_top_level)
        raise ValueError(
            f"top-level schema mismatch in {path}; missing={missing}, extra={extra}"
        )
    if payload["schema_version"] != 1:
        raise ValueError(f"unsupported SPCA artifact schema_version in {path}")
    config = _validate_config(payload["config"], path=path)
    manifest = payload["seed_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError(f"seed_manifest must be an object: {path}")
    claimed_manifest_digest = _sha256(
        _require(manifest, "manifest_sha256", location="seed_manifest"),
        name=f"{path}:seed_manifest.manifest_sha256",
    )
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256")
    if canonical_sha256(manifest_body) != claimed_manifest_digest:
        raise ValueError(f"seed_manifest.manifest_sha256 does not verify: {path}")
    if manifest.get("manifest_version") != 1:
        raise ValueError(f"unsupported seed manifest version: {path}")
    if manifest.get("namespace") != "spca-benchmark-v2":
        raise ValueError(f"unexpected seed manifest namespace: {path}")
    if manifest.get("base_seed") != config["base_seed"]:
        raise ValueError(f"seed manifest base_seed disagrees with config: {path}")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("config") != config:
        raise ValueError(f"seed manifest metadata does not bind the config: {path}")
    software_versions = manifest.get("software_versions")
    provenance = payload["provenance"]
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("software_versions"), Mapping)
        or provenance.get("software_versions") != software_versions
        or metadata.get("software_versions") != software_versions
    ):
        raise ValueError(f"software-version provenance is inconsistent: {path}")

    seed_values = manifest.get("seeds")
    if not isinstance(seed_values, list):
        raise ValueError(f"seed manifest seeds must be a list: {path}")
    seeds = tuple(
        _integer(seed, name=f"{path}:seed_manifest.seeds[{index}]")
        for index, seed in enumerate(seed_values)
    )
    if len(seeds) != config["repeats"] or len(set(seeds)) != len(seeds):
        raise ValueError(f"seed manifest must contain repeats unique seeds: {path}")

    provenance_values = manifest.get("run_provenance")
    if not isinstance(provenance_values, list) or len(provenance_values) != len(seeds):
        raise ValueError(f"run_provenance must contain one entry per seed: {path}")
    run_provenance: dict[int, dict[str, Any]] = {}
    expected_provenance_fields = {
        "run",
        "seed",
        "run_data_sha256",
        *DATA_HASH_FIELDS,
    }
    for expected_run, (expected_seed, value) in enumerate(
        zip(seeds, provenance_values, strict=True)
    ):
        if not isinstance(value, dict) or set(value) != expected_provenance_fields:
            raise ValueError(
                f"run_provenance[{expected_run}] has an incomplete hash contract: {path}"
            )
        if value["run"] != expected_run or value["seed"] != expected_seed:
            raise ValueError(f"run_provenance order/identity mismatch: {path}")
        hashes = {
            field: _sha256(
                value[field], name=f"{path}:run_provenance[{expected_run}].{field}"
            )
            for field in DATA_HASH_FIELDS
        }
        run_digest = _sha256(
            value["run_data_sha256"],
            name=f"{path}:run_provenance[{expected_run}].run_data_sha256",
        )
        if canonical_sha256(hashes) != run_digest:
            raise ValueError(
                f"run_data_sha256 does not verify for seed {expected_seed}: {path}"
            )
        run_provenance[expected_seed] = dict(value)
    run_digests = [value["run_data_sha256"] for value in run_provenance.values()]
    if len(set(run_digests)) != len(run_digests):
        raise ValueError(f"run_data_sha256 must be unique across seeds: {path}")

    record_values = payload["records"]
    if not isinstance(record_values, list):
        raise ValueError(f"records must be a list: {path}")
    records: dict[tuple[int, str], dict[str, Any]] = {}
    alphas = tuple(float(value) for value in config["alphas"])
    metrics = GENERAL_METRICS + (
        CORRUPTION_METRICS if float(config["corruption_fraction"]) > 0.0 else ()
    )
    for index, value in enumerate(record_values):
        location = f"{path}:records[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"record must be an object: {location}")
        run = _integer(
            _require(value, "run", location=location), name=f"{location}.run"
        )
        seed = _integer(
            _require(value, "seed", location=location), name=f"{location}.seed"
        )
        method = _require(value, "method", location=location)
        if not isinstance(method, str) or method not in EXPECTED_METHODS:
            raise ValueError(f"unexpected SPCA method at {location}: {method!r}")
        if run >= len(seeds) or seed != seeds[run]:
            raise ValueError(
                f"record run/seed identity disagrees with manifest: {location}"
            )
        identity = (seed, method)
        if identity in records:
            raise ValueError(
                f"duplicate method record for seed/method {identity}: {path}"
            )
        if value.get("selection") != "matched-sparsity":
            raise ValueError(f"record selection mismatch: {location}")
        target = _finite(
            _require(value, "target_sparsity", location=location),
            name=f"{location}.target_sparsity",
        )
        if target != float(config["target_sparsity"]):
            raise ValueError(
                f"record target_sparsity disagrees with config: {location}"
            )
        alpha = _require(value, "alpha", location=location)
        if method == BASELINE_METHOD:
            if alpha is not None:
                raise ValueError(f"thresholded PCA alpha must be null: {location}")
        else:
            selected_alpha = _finite(alpha, name=f"{location}.alpha")
            if selected_alpha not in alphas:
                raise ValueError(
                    f"selected alpha is not in the configured grid: {location}"
                )
        for field in DATA_HASH_FIELDS:
            row_hash = _sha256(
                _require(value, field, location=location), name=f"{location}.{field}"
            )
            if row_hash != run_provenance[seed][field]:
                raise ValueError(f"record has unpaired {field}: {location}")
        row_run_digest = _sha256(
            _require(value, "run_data_sha256", location=location),
            name=f"{location}.run_data_sha256",
        )
        if row_run_digest != run_provenance[seed]["run_data_sha256"]:
            raise ValueError(f"record has unpaired run_data_sha256: {location}")
        for metric in metrics:
            metric_value = _finite(
                _require(value, metric, location=location),
                name=f"{location}.{metric}",
            )
            if (
                metric
                in {
                    "subspace_error",
                    "component_mean_abs_correlation",
                    "component_min_abs_correlation",
                    "achieved_sparsity",
                    "support_precision",
                    "support_recall",
                    "support_f1",
                    *CORRUPTION_METRICS,
                }
                and not 0.0 <= metric_value <= 1.0
            ):
                raise ValueError(f"{location}.{metric} must lie in [0, 1]")
            if metric == "relative_reconstruction_error" and metric_value < 0.0:
                raise ValueError(f"{location}.{metric} must be non-negative")
        runtime = _finite(
            _require(value, "runtime_seconds", location=location),
            name=f"{location}.runtime_seconds",
        )
        if runtime < 0.0:
            raise ValueError(f"runtime_seconds must be non-negative: {location}")
        if not isinstance(_require(value, "converged", location=location), bool):
            raise TypeError(f"converged must be boolean: {location}")
        n_iter = _integer(
            _require(value, "n_iter", location=location),
            name=f"{location}.n_iter",
            minimum=1,
        )
        status = _require(value, "status", location=location)
        if not isinstance(status, str) or not status:
            raise ValueError(f"status must be a non-empty string: {location}")
        expected_budget = int(
            round(
                float(config["corruption_fraction"])
                * int(config["n_samples"])
                * int(config["n_features"])
            )
        )
        if value.get("outlier_budget_fraction") != config["corruption_fraction"]:
            raise ValueError(
                f"outlier_budget_fraction disagrees with config: {location}"
            )
        if value.get("outlier_budget_k") != expected_budget:
            raise ValueError(f"outlier_budget_k disagrees with config: {location}")
        fixed_policies = {
            "outlier_selection_policy": "predeclared_corruption_fraction_top_k",
            "centering_policy": (
                "none_dgp_components_column_centered_before_corruption"
            ),
            "runtime_comparison_scope": ("exploratory_not_matvec_or_phase_matched"),
        }
        for field, expected_value in fixed_policies.items():
            if value.get(field) != expected_value:
                raise ValueError(f"record has unexpected {field}: {location}")
        expected_postprocessing = (
            "hard_threshold_to_predeclared_target_sparsity"
            if method == BASELINE_METHOD
            else "none"
        )
        if value.get("loading_postprocessing") != expected_postprocessing:
            raise ValueError(
                f"record has unexpected loading_postprocessing: {location}"
            )
        if method == BASELINE_METHOD and (
            not value["converged"] or n_iter != 1 or status != "closed_form"
        ):
            raise ValueError(f"thresholded PCA baseline health is invalid: {location}")
        records[identity] = dict(value)

    expected_identities = {
        (seed, method) for seed in seeds for method in EXPECTED_METHODS
    }
    if set(records) != expected_identities:
        missing = sorted(expected_identities - set(records))
        extra = sorted(set(records) - expected_identities)
        raise ValueError(
            f"seed-by-method record matrix is incomplete in {path}; "
            f"missing={missing}, extra={extra}"
        )
    if len(record_values) != len(expected_identities):
        raise ValueError(
            f"record count disagrees with the complete pair matrix: {path}"
        )

    return _ValidatedRegime(
        name=name,
        path=Path(path),
        input_json_sha256=input_digest,
        config=config,
        config_sha256=canonical_sha256(config),
        seed_manifest_sha256=claimed_manifest_digest,
        seeds=seeds,
        run_provenance=run_provenance,
        records=records,
    )


def _validate_cross_regime_pairing(
    clean: _ValidatedRegime, corrupt: _ValidatedRegime
) -> str:
    clean_fraction = float(clean.config["corruption_fraction"])
    corrupt_fraction = float(corrupt.config["corruption_fraction"])
    if clean_fraction != 0.0:
        raise ValueError("the clean artifact must have corruption_fraction == 0")
    if not 0.0 < corrupt_fraction <= 1.0:
        raise ValueError("the corrupt artifact must have corruption_fraction in (0, 1]")
    clean_design = dict(clean.config)
    corrupt_design = dict(corrupt.config)
    clean_alphas = list(clean_design.pop("alphas"))
    corrupt_alphas = list(corrupt_design.pop("alphas"))
    clean_design.pop("corruption_fraction")
    corrupt_design.pop("corruption_fraction")
    if clean_design != corrupt_design:
        raise ValueError(
            "clean and corrupt artifacts must share one config except "
            "corruption_fraction and their predeclared alpha grids"
        )
    if clean.seeds != corrupt.seeds:
        raise ValueError(
            "clean and corrupt artifacts do not share one ordered seed bank"
        )
    for seed in clean.seeds:
        clean_provenance = clean.run_provenance[seed]
        corrupt_provenance = corrupt.run_provenance[seed]
        if (
            clean_provenance["observed_sha256"]
            != clean_provenance["clean_uncorrupted_sha256"]
        ):
            raise ValueError(f"clean seed {seed} observed data are not uncorrupted")
        if (
            corrupt_provenance["observed_sha256"]
            == corrupt_provenance["clean_uncorrupted_sha256"]
        ):
            raise ValueError(f"corrupt seed {seed} has no observed corruption")
        for field in SHARED_REGIME_HASH_FIELDS:
            if clean_provenance[field] != corrupt_provenance[field]:
                raise ValueError(
                    f"clean/corrupt seed {seed} has different paired {field}"
                )
    # The two corruption regimes can require different alpha scales to hit one
    # predeclared achieved-sparsity target.  Bind both grids while requiring
    # every other design choice, the target, and the paired seed bank to match.
    return canonical_sha256(
        {
            "shared_design": clean_design,
            "regime_alpha_grids": {
                "clean": clean_alphas,
                "corrupt": corrupt_alphas,
            },
        }
    )


def _bootstrap_plan(
    seeds: Sequence[int], *, resamples: int, seed: int
) -> tuple[NDArray[np.int64], str]:
    digest = canonical_sha256(
        {
            "namespace": "spca-benchmark-paired-seed-bootstrap-v1",
            "bootstrap_seed": seed,
            "ordered_seeds": list(seeds),
            "resamples": resamples,
        }
    )
    digest_words = np.frombuffer(bytes.fromhex(digest)[:16], dtype="<u4")
    seed_words = (seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF)
    sequence = np.random.SeedSequence([*seed_words, *map(int, digest_words)])
    rng = np.random.default_rng(sequence)
    indices = rng.integers(
        0,
        len(seeds),
        size=(resamples, len(seeds)),
        endpoint=False,
        dtype=np.int64,
    )
    plan_digest = hashlib.sha256(
        indices.astype("<i8", copy=False).tobytes()
    ).hexdigest()
    return indices, plan_digest


def _analysis_manifest(
    regimes: Sequence[_ValidatedRegime],
    *,
    paired_design_config_sha256: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    bootstrap_plan_sha256: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "artifact_type": "spca_clean_corrupt_paired_seed_summary",
        "inputs": [
            {
                "regime": regime.name,
                "input_json_sha256": regime.input_json_sha256,
                "config": regime.config,
                "config_sha256": regime.config_sha256,
                "seed_manifest_sha256": regime.seed_manifest_sha256,
                "ordered_seeds": list(regime.seeds),
                "record_count": len(regime.records),
            }
            for regime in regimes
        ],
        "paired_design_and_regime_alpha_grids_sha256": (
            paired_design_config_sha256
        ),
        "methods": list(EXPECTED_METHODS),
        "baseline_method": BASELINE_METHOD,
        "metrics_by_regime": {
            "clean": list(GENERAL_METRICS),
            "corrupt": list(GENERAL_METRICS + CORRUPTION_METRICS),
        },
        "metric_directions": METRIC_DIRECTIONS,
        "pairing_policy": {
            "unit": "seed",
            "required_matrix": "every_method_once_per_seed_per_regime",
            "cross_regime": (
                "same_ordered_seed_bank_and_identical_clean_uncorrupted_"
                "low_rank_loadings_and_feature_support_hashes"
            ),
            "difference": "method_minus_thresholded_pca_matched_sparsity",
        },
        "matched_sparsity_policy": {
            "target": regimes[0].config["target_sparsity"],
            "alpha_grid_scope": "predeclared_per_corruption_regime",
            "grid_selection_evidence": "achieved_sparsity_only",
            "truth_metrics_used_for_grid_selection": False,
            "grids_bound_in_analysis_manifest": True,
        },
        "bootstrap_policy": {
            "unit": "seed",
            "interval": "two_sided_percentile_95",
            "quantile_method": "numpy_linear",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "shared_plan_across_regimes_methods_and_metrics": True,
            "plan_sha256": bootstrap_plan_sha256,
            "runtime_included": False,
            "unqualified_runs_retained": True,
            "inference_requires_every_pair_converged": True,
            "scope": (
                "variation_over_the_fixed_paired_seed_bank_not_population_"
                "sampling_uncertainty"
            ),
        },
        "runtime_policy": {
            "statistics": ["mean", "sample_std", "min", "max"],
            "comparison_scope": "descriptive_only",
            "bootstrap_included": False,
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def summarize_spca_benchmarks(
    clean_json: Path,
    corrupt_json: Path,
    *,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 260829,
) -> dict[str, Any]:
    """Validate and summarize paired clean/corrupt SPCA result artifacts."""

    resamples = _integer(bootstrap_resamples, name="bootstrap_resamples", minimum=2)
    seed = _integer(bootstrap_seed, name="bootstrap_seed")
    clean = _validate_regime("clean", Path(clean_json))
    corrupt = _validate_regime("corrupt", Path(corrupt_json))
    design_digest = _validate_cross_regime_pairing(clean, corrupt)
    indices, plan_digest = _bootstrap_plan(clean.seeds, resamples=resamples, seed=seed)
    regimes = (clean, corrupt)
    manifest = _analysis_manifest(
        regimes,
        paired_design_config_sha256=design_digest,
        bootstrap_resamples=resamples,
        bootstrap_seed=seed,
        bootstrap_plan_sha256=plan_digest,
    )

    records: list[dict[str, Any]] = []
    for regime in regimes:
        metrics = GENERAL_METRICS + (
            CORRUPTION_METRICS if regime.name == "corrupt" else ()
        )
        for method in EXPECTED_METHODS:
            method_rows = [
                regime.records[(seed_value, method)] for seed_value in regime.seeds
            ]
            baseline_rows = [
                regime.records[(seed_value, BASELINE_METHOD)]
                for seed_value in regime.seeds
            ]
            converged_count = sum(bool(row["converged"]) for row in method_rows)
            pair_qualified = [
                bool(method_row["converged"]) and bool(baseline_row["converged"])
                for method_row, baseline_row in zip(
                    method_rows, baseline_rows, strict=True
                )
            ]
            qualified_pairs = int(sum(pair_qualified))
            inference_qualified = qualified_pairs == len(regime.seeds)
            runtime = _sample_summary(
                [float(row["runtime_seconds"]) for row in method_rows],
                name=f"{regime.name}:{method}:runtime",
            )
            for metric in metrics:
                absolute = _sample_summary(
                    [float(row[metric]) for row in method_rows],
                    name=f"{regime.name}:{method}:{metric}",
                )
                differences = np.asarray(
                    [
                        float(method_row[metric]) - float(baseline_row[metric])
                        for method_row, baseline_row in zip(
                            method_rows, baseline_rows, strict=True
                        )
                    ],
                    dtype=np.float64,
                )
                if not np.isfinite(differences).all():
                    raise ValueError(
                        f"non-finite paired difference for {regime.name}:{method}:{metric}"
                    )
                bootstrap_means = np.mean(differences[indices], axis=1)
                lower, upper = np.quantile(
                    bootstrap_means, (0.025, 0.975), method="linear"
                )
                records.append(
                    {
                        "analysis_manifest_sha256": manifest["manifest_sha256"],
                        "regime": regime.name,
                        "input_json_sha256": regime.input_json_sha256,
                        "config_sha256": regime.config_sha256,
                        "seed_manifest_sha256": regime.seed_manifest_sha256,
                        "method": method,
                        "metric": metric,
                        "metric_direction": METRIC_DIRECTIONS[metric],
                        "n_seeds": len(regime.seeds),
                        "absolute_mean": absolute["mean"],
                        "absolute_sample_std": absolute["sample_std"],
                        "converged_count": converged_count,
                        "convergence_rate": converged_count / len(regime.seeds),
                        "inference_qualified": inference_qualified,
                        "qualified_pairs": qualified_pairs,
                        "unqualified_pairs": len(regime.seeds) - qualified_pairs,
                        "bootstrap_ci95_status": (
                            "ok"
                            if inference_qualified
                            else "unqualified_fixed_budget_runs_included"
                        ),
                        "summary_population": (
                            "all_preregistered_pairs_including_unqualified_fixed_budget_runs"
                        ),
                        "runtime_n": runtime["n"],
                        "runtime_mean_seconds": runtime["mean"],
                        "runtime_sample_std_seconds": runtime["sample_std"],
                        "runtime_min_seconds": runtime["min"],
                        "runtime_max_seconds": runtime["max"],
                        "runtime_in_bootstrap": False,
                        "baseline_method": BASELINE_METHOD,
                        "difference_direction": "method_minus_baseline",
                        "n_pairs": len(regime.seeds),
                        "paired_mean_difference": float(np.mean(differences)),
                        "paired_sample_std_difference": float(
                            np.std(differences, ddof=1)
                        ),
                        "bootstrap_ci95_lower": float(lower),
                        "bootstrap_ci95_upper": float(upper),
                        "bootstrap_resamples": resamples,
                        "bootstrap_seed": seed,
                        "bootstrap_unit": "seed",
                        "bootstrap_plan_sha256": plan_digest,
                    }
                )

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "spca_clean_corrupt_paired_seed_summary",
        "analysis_manifest": manifest,
        "records": records,
    }
    result["summary_sha256"] = canonical_sha256(result)
    return result


def _csv_text(records: Sequence[Mapping[str, Any]]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for record in records:
        if set(record) != set(CSV_FIELDS):
            missing = sorted(set(CSV_FIELDS) - set(record))
            extra = sorted(set(record) - set(CSV_FIELDS))
            raise ValueError(
                f"CSV summary schema mismatch; missing={missing}, extra={extra}"
            )
        writer.writerow(record)
    return stream.getvalue()


def write_summary_outputs(
    summary: Mapping[str, Any], *, output_json: Path, output_csv: Path
) -> None:
    """Exclusively create JSON and CSV outputs; existing files are never replaced."""

    json_path = Path(output_json)
    csv_path = Path(output_csv)
    if json_path.absolute() == csv_path.absolute():
        raise ValueError("JSON and CSV output paths must be different")
    records = summary.get("records")
    if not isinstance(records, list):
        raise ValueError("summary records must be a list")
    json_text = json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    csv_text = _csv_text(records)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    created: list[tuple[Path, int]] = []
    try:
        for path in (json_path, csv_path):
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
            created.append((path, descriptor))
        for (_, descriptor), text in zip(created, (json_text, csv_text), strict=True):
            encoded = text.encode("utf-8")
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        for _, descriptor in created:
            os.close(descriptor)
        created = [(path, -1) for path, _ in created]
    except Exception:
        for path, descriptor in created:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            path.unlink(missing_ok=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-json", required=True, type=Path)
    parser.add_argument("--corrupt-json", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260829)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_spca_benchmarks(
        args.clean_json,
        args.corrupt_json,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_summary_outputs(
        summary, output_json=args.output_json, output_csv=args.output_csv
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "records": len(summary["records"]),
                "summary_sha256": summary["summary_sha256"],
                "analysis_manifest_sha256": summary["analysis_manifest"][
                    "manifest_sha256"
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

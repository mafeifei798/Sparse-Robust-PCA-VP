#!/usr/bin/env python3
"""Strict multi-seed benchmark for held-out 1000 Genomes classification.

The benchmark reuses :mod:`experiments.genomics_1kg` for every fit.  Explicit
parent seeds define paired outer splits: all methods evaluated for one parent
seed receive the same derived split seed and must report identical train/test
hashes.  Sparse methods select alpha only on the inner training/validation
partition implemented by the single-run module.  The outer test partition is
therefore used exactly once for the primary analysis.

The primary outcomes are held-out accuracy, balanced accuracy, and macro-F1.
Runtime is retained as descriptive provenance but is deliberately excluded
from the paired seed bootstrap.  Output files are opt-in, exclusive-create
artifacts: existing or conflicting paths are rejected rather than overwritten.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import io
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from .genomics_1kg import (
        GenomicsConfig,
        GenotypeDataset,
        load_genotypes,
        load_population_labels,
        run_genomics,
    )
else:  # Support execution from the repository root.
    from genomics_1kg import (  # type: ignore[no-redef]
        GenomicsConfig,
        GenotypeDataset,
        load_genotypes,
        load_population_labels,
        run_genomics,
    )
from rspca.datasets import (
    array_sha256,
    canonical_sha256,
    make_seed_manifest,
    runtime_versions,
    verify_seed_manifest,
)


METHODS = ("pca", "vp", "randomized", "robust")
CORE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")

# These values must agree within one requested seed before a paired difference
# is scientifically meaningful.  They prove that methods saw the same outer
# split and the same labels, not merely the same integer seed.
PAIR_PROVENANCE_FIELDS = (
    "genotype_sha256",
    "metadata_sha256",
    "normalized_genotypes_sha256",
    "sample_ids_sha256",
    "snp_ids_sha256",
    "population_labels_sha256",
    "split_seed",
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
)

# Schema-v1 formal runs write exactly this provenance surface.  Keeping the
# read-side schema closed prevents fields that participate in data identity,
# array shape, or runtime provenance from being silently added, removed, or
# given a different type after execution.
FORMAL_PROVENANCE_FIELDS = frozenset(
    {
        "expected_genotype_sha256",
        "expected_metadata_sha256",
        "genotype_path",
        "genotype_sha256",
        "input_orientation",
        "internal_orientation",
        "metadata_path",
        "metadata_sample_id_column",
        "metadata_sha256",
        "missing_value",
        "normalized_genotypes_sha256",
        "population_column",
        "population_labels_sha256",
        "runtime_versions_at_start",
        "sample_ids_sha256",
        "samples",
        "snp_ids_sha256",
        "snps",
    }
)
RUNTIME_VERSION_FIELDS = frozenset(
    {"machine", "numpy", "platform", "python", "scipy", "vp_rspca"}
)


@dataclass(frozen=True)
class GenomicsBenchmarkConfig:
    """Pre-registered solver, split, and paired-bootstrap policy."""

    seeds: tuple[int, ...] = tuple(range(2025, 2045))
    methods: tuple[str, ...] = METHODS
    rank: int = 5
    alpha: float = 0.1
    alpha_candidates: tuple[float, ...] = (
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        30.0,
        100.0,
    )
    validation_fraction: float = 0.2
    max_iter: int = 1000
    randomized_max_iter: int = 500
    tol: float = 1e-5
    selection_threshold: float = 1e-8
    test_fraction: float = 0.2
    standardize: bool = True
    oversample: int = 30
    n_power_iter: int = 4
    randomized_full_gradient: bool = False
    randomized_finalize_full_iters: int = 500
    missing_value: float | None = -9.0
    bootstrap_resamples: int = 100_000
    bootstrap_seed: int = 2025


@dataclass(frozen=True)
class GenomicsBenchmarkResult:
    """Complete in-memory result, including every underlying single run."""

    config: GenomicsBenchmarkConfig
    manifest: Mapping[str, Any]
    provenance: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    full_runs: tuple[Mapping[str, Any], ...]
    method_summaries: tuple[Mapping[str, Any], ...]
    paired_summaries: tuple[Mapping[str, Any], ...]


def _integer(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if (positive and integer <= 0) or (not positive and integer < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return integer


def _finite_scalar(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not np.isfinite(scalar) or (nonnegative and scalar < 0.0):
        qualifier = "finite non-negative" if nonnegative else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return scalar


def _validate_config(config: GenomicsBenchmarkConfig) -> None:
    if not isinstance(config.seeds, tuple) or len(config.seeds) < 2:
        raise ValueError("seeds must be a tuple containing at least two seeds")
    seeds = tuple(_integer(seed, "seed") for seed in config.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    if not isinstance(config.methods, tuple) or not config.methods:
        raise ValueError("methods must be a non-empty tuple")
    if len(set(config.methods)) != len(config.methods):
        raise ValueError("methods must not contain duplicates")
    unknown = sorted(set(config.methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown genomics methods: {unknown}")
    if "pca" not in config.methods:
        raise ValueError(
            "methods must include pca for paired method-minus-PCA analysis"
        )

    _integer(config.rank, "rank", positive=True)
    _integer(config.max_iter, "max_iter", positive=True)
    _integer(config.randomized_max_iter, "randomized_max_iter", positive=True)
    _integer(config.oversample, "oversample")
    _integer(config.n_power_iter, "n_power_iter")
    _integer(
        config.randomized_finalize_full_iters,
        "randomized_finalize_full_iters",
    )
    if _integer(config.bootstrap_resamples, "bootstrap_resamples", positive=True) < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    _integer(config.bootstrap_seed, "bootstrap_seed")
    _finite_scalar(config.alpha, "alpha", nonnegative=True)
    _finite_scalar(config.tol, "tol")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive")
    _finite_scalar(config.selection_threshold, "selection_threshold", nonnegative=True)
    for name, fraction in (
        ("validation_fraction", config.validation_fraction),
        ("test_fraction", config.test_fraction),
    ):
        _finite_scalar(fraction, name)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"{name} must lie strictly between zero and one")
    if not isinstance(config.standardize, (bool, np.bool_)):
        raise TypeError("standardize must be boolean")
    if not isinstance(config.randomized_full_gradient, (bool, np.bool_)):
        raise TypeError("randomized_full_gradient must be boolean")
    candidates = tuple(
        _finite_scalar(alpha, "alpha candidate", nonnegative=True)
        for alpha in config.alpha_candidates
    )
    if any(method != "pca" for method in config.methods) and not candidates:
        raise ValueError(
            "sparse methods require a non-empty inner-validation alpha grid"
        )
    if len(set(candidates)) != len(candidates):
        raise ValueError("alpha_candidates must not contain duplicates")
    if config.missing_value is not None:
        _finite_scalar(config.missing_value, "missing_value")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _single_run_config(
    benchmark: GenomicsBenchmarkConfig,
    *,
    method: str,
    seed: int,
) -> GenomicsConfig:
    return GenomicsConfig(
        method=method,
        rank=benchmark.rank,
        alpha=benchmark.alpha,
        alpha_candidates=(() if method == "pca" else tuple(benchmark.alpha_candidates)),
        validation_fraction=benchmark.validation_fraction,
        max_iter=(
            benchmark.randomized_max_iter
            if method == "randomized"
            else benchmark.max_iter
        ),
        tol=benchmark.tol,
        # The formal multi-seed analysis bootstraps paired outer-split units.
        # The single-run full-cohort SNP bootstrap is a different estimand and
        # is intentionally disabled here.
        bootstrap_repeats=0,
        selection_threshold=benchmark.selection_threshold,
        test_fraction=benchmark.test_fraction,
        base_seed=int(seed),
        standardize=benchmark.standardize,
        oversample=benchmark.oversample,
        n_power_iter=benchmark.n_power_iter,
        randomized_full_gradient=benchmark.randomized_full_gradient,
        randomized_finalize_full_iters=benchmark.randomized_finalize_full_iters,
        missing_value=benchmark.missing_value,
    )


def _expected_single_run_manifest(
    benchmark: GenomicsBenchmarkConfig,
    provenance: Mapping[str, Any],
    *,
    method: str,
    seed: int,
) -> dict[str, Any]:
    """Rebuild the seed manifest bound to one formal method/seed request."""

    single_config = _single_run_config(benchmark, method=method, seed=seed)
    return make_seed_manifest(
        int(seed),
        single_config.bootstrap_repeats + 2,
        namespace="genomics-1kg-v2",
        metadata={
            **asdict(single_config),
            "n_samples": provenance["samples"],
            "n_snps": provenance["snps"],
            "orientation": "samples-by-snps",
            "loader_schema_version": 2,
            "normalized_genotypes_sha256": provenance[
                "normalized_genotypes_sha256"
            ],
            "sample_ids_sha256": provenance["sample_ids_sha256"],
            "snp_ids_sha256": provenance["snp_ids_sha256"],
            "seed_roles": {
                "index_0": "full_cohort_solver",
                "index_1": "held_out_split_and_selection_parent",
                "index_2_onward": "bootstrap_solver_parent_for_resample_seed",
            },
        },
    )


def _manifest(
    config: GenomicsBenchmarkConfig,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "manifest_version": 1,
        "namespace": "genomics-1kg-multiseed-benchmark-v1",
        "explicit_parent_seeds": [int(seed) for seed in config.seeds],
        "methods": list(config.methods),
        "configuration": asdict(config),
        "input_identity": {
            key: provenance[key]
            for key in (
                "genotype_sha256",
                "metadata_sha256",
                "normalized_genotypes_sha256",
                "sample_ids_sha256",
                "snp_ids_sha256",
                "population_labels_sha256",
            )
        },
        "analysis_policy": {
            "primary_record_type": "held_out_population_classification",
            "primary_metrics": list(CORE_METRICS),
            "outer_pairing_unit": "explicit_parent_seed",
            "outer_test_usage": "evaluate_once_after_inner_validation_selection",
            "sparse_alpha_selection": "inner_train_validation_only",
            "single_run_full_cohort_bootstrap_repeats": 0,
            "best_seed_selection": False,
        },
        "bootstrap_policy": {
            "unit": "explicit_parent_seed",
            "difference": "method_minus_pca",
            "interval": "percentile_95",
            "resamples": int(config.bootstrap_resamples),
            "seed": int(config.bootstrap_seed),
            "metrics": list(CORE_METRICS),
            "runtime_included": False,
        },
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def _one_record(
    *,
    method: str,
    run_index: int,
    requested_seed: int,
    runtime_seconds: float,
    classification: Mapping[str, Any],
    single_manifest: Mapping[str, Any],
    benchmark_manifest_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    if classification.get("record_type") != "held_out_population_classification":
        raise ValueError("single run is missing its held-out classification record")
    if classification.get("method") != method:
        raise ValueError("single-run classification method does not match its request")
    if not verify_seed_manifest(single_manifest):
        raise ValueError("single-run seed manifest hash mismatch")
    if int(single_manifest.get("base_seed", -1)) != requested_seed:
        raise ValueError("single-run seed manifest does not bind the requested seed")
    seeds = single_manifest.get("seeds")
    if not isinstance(seeds, list) or len(seeds) < 2:
        raise ValueError("single-run seed manifest is missing the outer split seed")
    if int(classification.get("split_seed", -1)) != int(seeds[1]):
        raise ValueError("classification split seed disagrees with its seed manifest")
    if classification.get("test_used_for_model_or_feature_selection") is not False:
        raise ValueError("outer test data was used for model or feature selection")
    if method == "pca":
        if classification.get("alpha_selection_enabled") is not False:
            raise ValueError("PCA must not run alpha selection")
    else:
        if classification.get("alpha_selection_enabled") is not True:
            raise ValueError("sparse methods must run inner-validation alpha selection")
        selected_alpha = _finite_scalar(
            classification.get("selected_alpha"), "selected_alpha", nonnegative=True
        )
        metadata = single_manifest.get("metadata")
        if not isinstance(metadata, Mapping) or "alpha_candidates" not in metadata:
            raise ValueError("single-run manifest is missing the registered alpha grid")
        if selected_alpha not in {float(item) for item in metadata["alpha_candidates"]}:
            raise ValueError("selected alpha is absent from the pre-registered grid")

    runtime = _finite_scalar(runtime_seconds, "runtime_seconds", nonnegative=True)
    record: dict[str, Any] = {
        "benchmark_manifest_sha256": benchmark_manifest_sha256,
        "single_run_manifest_sha256": single_manifest["manifest_sha256"],
        "run": int(run_index),
        "run_index": int(run_index),
        "requested_seed": int(requested_seed),
        "method": method,
        "runtime_seconds": runtime,
        **{
            key: provenance[key]
            for key in (
                "genotype_sha256",
                "metadata_sha256",
                "normalized_genotypes_sha256",
                "sample_ids_sha256",
                "snp_ids_sha256",
                "population_labels_sha256",
            )
        },
        **dict(classification),
        "classification_record": dict(classification),
    }
    # The method and pair identity above are authoritative wrapper fields.
    record["method"] = method
    record["run"] = record["run_index"] = int(run_index)
    record["requested_seed"] = int(requested_seed)
    for metric in CORE_METRICS:
        _finite_metric(record, metric, method=method, seed=requested_seed)
    for field in PAIR_PROVENANCE_FIELDS:
        if field not in record:
            raise ValueError(
                f"classification record is missing paired provenance {field!r}"
            )
        if field != "split_seed":
            _require_sha256(record[field], field)
    return record


def _record_key(record: Mapping[str, Any]) -> tuple[str, int]:
    try:
        return str(record["method"]), int(record["requested_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("record is missing method/requested_seed identity") from error


def _finite_metric(
    record: Mapping[str, Any], metric: str, *, method: str, seed: int
) -> float:
    if metric not in record:
        raise ValueError(
            f"missing core metric {metric!r} for method {method!r}, seed {seed}"
        )
    try:
        value = float(record[metric])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"core metric {metric!r} must be finite for method {method!r}, seed {seed}"
        ) from error
    if not np.isfinite(value):
        raise ValueError(
            f"core metric {metric!r} must be finite for method {method!r}, seed {seed}"
        )
    return value


def _validated_records(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    benchmark_manifest_sha256: str,
) -> dict[str, dict[int, Mapping[str, Any]]]:
    grouped: dict[str, dict[int, Mapping[str, Any]]] = {}
    for record in records:
        method, seed = _record_key(record)
        by_seed = grouped.setdefault(method, {})
        if seed in by_seed:
            raise ValueError(f"duplicate record for method {method!r}, seed {seed}")
        by_seed[seed] = record

    expected_methods = set(methods)
    if set(grouped) != expected_methods:
        raise ValueError(
            "method records are missing or unexpected: "
            f"expected={sorted(expected_methods)}, actual={sorted(grouped)}"
        )
    expected_seeds = set(int(seed) for seed in seeds)
    for method in methods:
        actual_seeds = set(grouped[method])
        if actual_seeds != expected_seeds:
            raise ValueError(
                f"method {method!r} has missing or unexpected seeds: "
                f"expected={sorted(expected_seeds)}, actual={sorted(actual_seeds)}"
            )
        for seed in expected_seeds:
            record = grouped[method][seed]
            if record.get("benchmark_manifest_sha256") != benchmark_manifest_sha256:
                raise ValueError(
                    f"benchmark manifest hash mismatch for method {method!r}, seed {seed}"
                )
            for metric in CORE_METRICS:
                _finite_metric(record, metric, method=method, seed=seed)
            _finite_metric(record, "runtime_seconds", method=method, seed=seed)
            for field in ("solver_converged", "alpha_selection_qualified"):
                if not isinstance(record.get(field), (bool, np.bool_)):
                    raise ValueError(
                        f"{field} must be boolean for method {method!r}, seed {seed}"
                    )
            for field in PAIR_PROVENANCE_FIELDS:
                if field not in record:
                    raise ValueError(
                        f"missing paired provenance {field!r} for method "
                        f"{method!r}, seed {seed}"
                    )
                if field != "split_seed":
                    _require_sha256(record[field], field)

    baseline = grouped["pca"]
    for method in methods:
        if method == "pca":
            continue
        for seed in expected_seeds:
            for field in PAIR_PROVENANCE_FIELDS:
                if field not in baseline[seed] or field not in grouped[method][seed]:
                    raise ValueError(
                        f"missing paired provenance {field!r} for method {method!r}, "
                        f"seed {seed}"
                    )
                if grouped[method][seed][field] != baseline[seed][field]:
                    raise ValueError(
                        f"paired provenance hash mismatch in {field!r} for method "
                        f"{method!r}, seed {seed}"
                    )
    return grouped


def summarize_method_metrics(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    benchmark_manifest_sha256: str,
) -> list[dict[str, Any]]:
    grouped = _validated_records(
        records,
        methods=methods,
        seeds=seeds,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
    )
    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    summaries: list[dict[str, Any]] = []
    for method in methods:
        method_records = grouped[method]
        converged = [
            bool(method_records[seed]["solver_converged"]) for seed in ordered_seeds
        ]
        qualified = [
            bool(method_records[seed]["alpha_selection_qualified"])
            for seed in ordered_seeds
        ]
        inference_qualified = all(
            solver_ok and selection_ok
            for solver_ok, selection_ok in zip(converged, qualified, strict=True)
        )
        for metric in (*CORE_METRICS, "runtime_seconds"):
            values = np.asarray(
                [
                    _finite_metric(
                        method_records[seed], metric, method=method, seed=seed
                    )
                    for seed in ordered_seeds
                ],
                dtype=np.float64,
            )
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_seeds": len(ordered_seeds),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "solver_converged_count": int(sum(converged)),
                    "solver_convergence_rate": float(np.mean(converged)),
                    "alpha_selection_qualified_count": int(sum(qualified)),
                    "alpha_selection_qualified_rate": float(np.mean(qualified)),
                    "inference_qualified": inference_qualified,
                    "qualification_status": (
                        "ok"
                        if inference_qualified
                        else "unqualified_fixed_budget_runs_included"
                    ),
                    "summary_population": (
                        "all_preregistered_runs_including_unqualified_fixed_budget_runs"
                    ),
                    "metric_scope": (
                        "descriptive_wall_clock_only"
                        if metric == "runtime_seconds"
                        else "primary_held_out_population_classification"
                    ),
                }
            )
    return summaries


def _bootstrap_indices(
    seeds: Sequence[int], *, resamples: int, bootstrap_seed: int
) -> np.ndarray:
    ordered = tuple(sorted(int(seed) for seed in seeds))
    digest = canonical_sha256(
        {
            "namespace": "genomics-1kg-seed-paired-bootstrap-v1",
            "seeds": list(ordered),
            "bootstrap_seed": int(bootstrap_seed),
        }
    )
    digest_words = np.frombuffer(bytes.fromhex(digest)[:16], dtype="<u4")
    seed_words = (
        int(bootstrap_seed) & 0xFFFFFFFF,
        (int(bootstrap_seed) >> 32) & 0xFFFFFFFF,
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([*seed_words, *map(int, digest_words)])
    )
    return rng.integers(
        0,
        len(ordered),
        size=(int(resamples), len(ordered)),
        endpoint=False,
    )


def summarize_paired_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    seeds: Sequence[int],
    benchmark_manifest_sha256: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    grouped = _validated_records(
        records,
        methods=methods,
        seeds=seeds,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
    )
    resamples = _integer(bootstrap_resamples, "bootstrap_resamples", positive=True)
    if resamples < 2:
        raise ValueError("bootstrap_resamples must be at least 2")
    bootstrap_seed = _integer(bootstrap_seed, "bootstrap_seed")
    ordered_seeds = tuple(sorted(int(seed) for seed in seeds))
    indices = _bootstrap_indices(
        ordered_seeds,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summaries: list[dict[str, Any]] = []
    for method in methods:
        if method == "pca":
            continue
        pair_qualified = [
            bool(grouped[method][seed]["solver_converged"])
            and bool(grouped[method][seed]["alpha_selection_qualified"])
            and bool(grouped["pca"][seed]["solver_converged"])
            and bool(grouped["pca"][seed]["alpha_selection_qualified"])
            for seed in ordered_seeds
        ]
        qualified_pairs = int(sum(pair_qualified))
        inference_qualified = qualified_pairs == len(ordered_seeds)
        for metric in CORE_METRICS:
            differences = np.asarray(
                [
                    _finite_metric(
                        grouped[method][seed], metric, method=method, seed=seed
                    )
                    - _finite_metric(
                        grouped["pca"][seed], metric, method="pca", seed=seed
                    )
                    for seed in ordered_seeds
                ],
                dtype=np.float64,
            )
            bootstrap_means = np.mean(differences[indices], axis=1)
            low, high = np.quantile(bootstrap_means, (0.025, 0.975))
            summaries.append(
                {
                    "method": method,
                    "metric": metric,
                    "baseline_method": "pca",
                    "difference_direction": "method_minus_pca",
                    "n_pairs": len(ordered_seeds),
                    "mean_difference": float(np.mean(differences)),
                    "std_difference": float(np.std(differences, ddof=1)),
                    "bootstrap_ci95_low": float(low),
                    "bootstrap_ci95_high": float(high),
                    "bootstrap_ci95_status": (
                        "ok"
                        if inference_qualified
                        else "unqualified_fixed_budget_runs_included"
                    ),
                    "inference_qualified": inference_qualified,
                    "qualified_pairs": qualified_pairs,
                    "unqualified_pairs": len(ordered_seeds) - qualified_pairs,
                    "summary_population": (
                        "all_preregistered_pairs_including_unqualified_fixed_budget_runs"
                    ),
                    "bootstrap_resamples": resamples,
                    "bootstrap_seed": bootstrap_seed,
                    "bootstrap_unit": "explicit_parent_seed",
                    "runtime_included": False,
                }
            )
    return summaries


def run_genomics_benchmark(
    dataset: GenotypeDataset,
    population_labels: np.ndarray,
    config: GenomicsBenchmarkConfig,
    *,
    provenance: Mapping[str, Any],
) -> GenomicsBenchmarkResult:
    """Run every method on every explicit paired parent seed."""

    _validate_config(config)
    required_provenance = {
        "genotype_sha256",
        "metadata_sha256",
        "normalized_genotypes_sha256",
        "sample_ids_sha256",
        "snp_ids_sha256",
        "population_labels_sha256",
    }
    missing_provenance = sorted(required_provenance - set(provenance))
    if missing_provenance:
        raise ValueError(f"provenance is missing required fields: {missing_provenance}")
    for field in required_provenance:
        _require_sha256(provenance[field], field)
    labels = np.asarray(population_labels, dtype=str)
    if labels.shape != (dataset.genotypes.shape[0],):
        raise ValueError("population_labels must contain one label per sample")
    if array_sha256(dataset.genotypes) != provenance["normalized_genotypes_sha256"]:
        raise ValueError("normalized genotype hash mismatch")
    if array_sha256(dataset.sample_ids) != provenance["sample_ids_sha256"]:
        raise ValueError("sample ID hash mismatch")
    if array_sha256(dataset.snp_ids) != provenance["snp_ids_sha256"]:
        raise ValueError("SNP ID hash mismatch")
    if array_sha256(labels) != provenance["population_labels_sha256"]:
        raise ValueError("population-label hash mismatch")

    manifest = _manifest(config, provenance)
    benchmark_manifest_sha256 = str(manifest["manifest_sha256"])
    records: list[dict[str, Any]] = []
    full_runs: list[dict[str, Any]] = []
    for run_index, requested_seed in enumerate(config.seeds):
        for method in config.methods:
            single_config = _single_run_config(
                config,
                method=method,
                seed=int(requested_seed),
            )
            started = time.perf_counter()
            rows, single_manifest = run_genomics(
                dataset,
                single_config,
                population_labels=labels,
            )
            elapsed = time.perf_counter() - started
            classifications = [
                row
                for row in rows
                if row.get("record_type") == "held_out_population_classification"
            ]
            if len(classifications) != 1:
                raise ValueError(
                    f"method {method!r}, seed {requested_seed} returned "
                    f"{len(classifications)} held-out records; expected exactly one"
                )
            record = _one_record(
                method=method,
                run_index=run_index,
                requested_seed=int(requested_seed),
                runtime_seconds=elapsed,
                classification=classifications[0],
                single_manifest=single_manifest,
                benchmark_manifest_sha256=benchmark_manifest_sha256,
                provenance=provenance,
            )
            full_run_index = len(full_runs)
            record["full_run_index"] = full_run_index
            records.append(record)
            full_runs.append(
                {
                    "full_run_index": full_run_index,
                    "run": run_index,
                    "run_index": run_index,
                    "requested_seed": int(requested_seed),
                    "method": method,
                    "runtime_seconds": float(elapsed),
                    "config": asdict(single_config),
                    "seed_manifest": single_manifest,
                    "records": rows,
                }
            )

    method_summaries = summarize_method_metrics(
        records,
        methods=config.methods,
        seeds=config.seeds,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
    )
    paired_summaries = summarize_paired_differences(
        records,
        methods=config.methods,
        seeds=config.seeds,
        benchmark_manifest_sha256=benchmark_manifest_sha256,
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
    )
    return GenomicsBenchmarkResult(
        config=config,
        manifest=manifest,
        provenance=dict(provenance),
        records=tuple(records),
        full_runs=tuple(full_runs),
        method_summaries=tuple(method_summaries),
        paired_summaries=tuple(paired_summaries),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite numbers are forbidden in benchmark artifacts")
    return value


def _reject_nonfinite_numbers(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, location=f"{location}[{index}]")
    elif isinstance(value, np.ndarray):
        _reject_nonfinite_numbers(value.tolist(), location=location)
    elif isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        raise ValueError(f"number must be finite at {location}")


def _json_payload(result: GenomicsBenchmarkResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config": asdict(result.config),
        "runtime_versions": runtime_versions(),
        "manifest": result.manifest,
        "provenance": result.provenance,
        "evaluation_protocol": {
            "primary_record_type": "held_out_population_classification",
            "primary_metrics": list(CORE_METRICS),
            "pairing_unit": "explicit_parent_seed",
            "paired_difference": "method_minus_pca_on_identical_outer_split",
            "sparse_alpha_selection": "inner_train_validation_only",
            "outer_test_used_for_selection": False,
            "bootstrap_unit": "explicit_parent_seed",
            "bootstrap_interval": "percentile_95",
            "runtime_bootstrapped": False,
            "best_seed_selection": False,
            "unqualified_runs_retained": True,
            "bootstrap_inference_requires_every_pair_qualified": True,
        },
        "records": list(result.records),
        "full_runs": list(result.full_runs),
        "method_summaries": list(result.method_summaries),
        "paired_summaries": list(result.paired_summaries),
    }


def _validate_result(result: GenomicsBenchmarkResult) -> None:
    """Recheck artifact integrity at the write boundary."""

    _validate_config(result.config)
    _reject_nonfinite_numbers(asdict(result.config), location="config")
    _reject_nonfinite_numbers(result.manifest, location="manifest")
    _reject_nonfinite_numbers(result.provenance, location="provenance")
    _reject_nonfinite_numbers(result.records, location="records")
    _reject_nonfinite_numbers(result.full_runs, location="full_runs")
    _reject_nonfinite_numbers(result.method_summaries, location="method_summaries")
    _reject_nonfinite_numbers(result.paired_summaries, location="paired_summaries")
    if not verify_seed_manifest(result.manifest):
        raise ValueError("benchmark manifest hash mismatch")
    if dict(result.manifest) != _manifest(result.config, result.provenance):
        raise ValueError("benchmark manifest does not bind this config/provenance")
    _validated_records(
        result.records,
        methods=result.config.methods,
        seeds=result.config.seeds,
        benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
    )
    expected_runs = len(result.config.methods) * len(result.config.seeds)
    if len(result.full_runs) != expected_runs or len(result.records) != expected_runs:
        raise ValueError("full runs are missing or duplicated")
    full_run_fields = {
        "full_run_index",
        "run",
        "run_index",
        "requested_seed",
        "method",
        "runtime_seconds",
        "config",
        "seed_manifest",
        "records",
    }
    grouped = _validated_records(
        result.records,
        methods=result.config.methods,
        seeds=result.config.seeds,
        benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
    )
    seen: set[int] = set()
    for run_index, requested_seed in enumerate(result.config.seeds):
        for method_index, method in enumerate(result.config.methods):
            record = grouped[method][int(requested_seed)]
            index = record.get("full_run_index")
            expected_index = run_index * len(result.config.methods) + method_index
            if (
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or int(index) != expected_index
            ):
                raise ValueError(
                    "full_run_index does not match canonical method/seed ordering"
                )
            index = int(index)
            if index in seen or not 0 <= index < len(result.full_runs):
                raise ValueError(
                    "full_run_index values are duplicated or out of range"
                )
            seen.add(index)
            full_run = result.full_runs[index]
            if set(full_run) != full_run_fields:
                missing = sorted(full_run_fields - set(full_run))
                extra = sorted(set(full_run) - full_run_fields)
                raise ValueError(
                    "full run schema mismatch; "
                    f"missing={missing}, extra={extra}"
                )
            expected_identity = {
                "full_run_index": index,
                "run": run_index,
                "run_index": run_index,
                "requested_seed": int(requested_seed),
                "method": method,
            }
            for field_name, expected in expected_identity.items():
                observed = full_run.get(field_name)
                if isinstance(observed, (bool, np.bool_)) or observed != expected:
                    raise ValueError(
                        f"full run {field_name} does not match canonical ordering"
                    )
            for field_name in ("run", "run_index"):
                observed = record.get(field_name)
                if isinstance(observed, (bool, np.bool_)) or observed != run_index:
                    raise ValueError(
                        f"primary record {field_name} does not match canonical ordering"
                    )
            rows = full_run.get("records")
            if not isinstance(rows, list) or any(
                not isinstance(row, Mapping) for row in rows
            ):
                raise ValueError("full run records must be an array of objects")
            classifications = [
                row
                for row in rows
                if row.get("record_type") == "held_out_population_classification"
            ]
            if len(classifications) != 1:
                raise ValueError(
                    "full run must contain exactly one held-out classification record"
                )
            expected_config = asdict(
                _single_run_config(
                    result.config,
                    method=method,
                    seed=int(requested_seed),
                )
            )
            if full_run.get("config") != expected_config:
                raise ValueError(
                    "full run config does not match the benchmark request"
                )
            expected_record = _one_record(
                method=method,
                run_index=run_index,
                requested_seed=int(requested_seed),
                runtime_seconds=full_run["runtime_seconds"],
                classification=classifications[0],
                single_manifest=full_run["seed_manifest"],
                benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
                provenance=result.provenance,
            )
            expected_record["full_run_index"] = index
            if dict(record) != expected_record:
                raise ValueError(
                    "primary record does not exactly match its full run"
                )

    expected_method_summaries = tuple(
        summarize_method_metrics(
            result.records,
            methods=result.config.methods,
            seeds=result.config.seeds,
            benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
        )
    )
    if tuple(result.method_summaries) != expected_method_summaries:
        raise ValueError("method summaries do not match the per-seed records")
    expected_paired_summaries = tuple(
        summarize_paired_differences(
            result.records,
            methods=result.config.methods,
            seeds=result.config.seeds,
            benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
            bootstrap_resamples=result.config.bootstrap_resamples,
            bootstrap_seed=result.config.bootstrap_seed,
        )
    )
    if tuple(result.paired_summaries) != expected_paired_summaries:
        raise ValueError("paired summaries do not match the per-seed records")


def validate_result_payload(
    payload: Mapping[str, Any],
    *,
    label: str = "Genomics benchmark payload",
    expected_provenance: Mapping[str, Any] | None = None,
) -> GenomicsBenchmarkResult:
    """Rehydrate and fully validate one schema-v1 benchmark JSON object.

    This is the read-side counterpart of :func:`write_results`.  It validates
    the exact top-level/config/full-run schemas, JSON tuple round trips,
    manifest binding, canonical record indices, record/full-run duplication,
    and recomputed summaries before returning an in-memory result.  When
    ``expected_provenance`` is supplied, it is an external trust input rather
    than a value derived from the payload; the complete formal provenance and
    both runtime copies must match it exactly.
    """

    if not isinstance(payload, Mapping):
        raise ValueError(f"{label}: root must be an object")
    top_level_fields = {
        "schema_version",
        "config",
        "runtime_versions",
        "manifest",
        "provenance",
        "evaluation_protocol",
        "records",
        "full_runs",
        "method_summaries",
        "paired_summaries",
    }
    if set(payload) != top_level_fields:
        missing = sorted(top_level_fields - set(payload))
        extra = sorted(set(payload) - top_level_fields)
        raise ValueError(
            f"{label}: top-level schema mismatch; missing={missing}, extra={extra}"
        )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, (bool, np.bool_))
        or not isinstance(schema_version, (int, np.integer))
        or int(schema_version) != 1
    ):
        raise ValueError(f"{label}: schema_version must be integer 1")
    runtime = payload["runtime_versions"]
    if not isinstance(runtime, Mapping) or set(runtime) != RUNTIME_VERSION_FIELDS:
        actual = set(runtime) if isinstance(runtime, Mapping) else set()
        raise ValueError(
            f"{label}: runtime_versions schema mismatch; "
            f"missing={sorted(RUNTIME_VERSION_FIELDS - actual)}, "
            f"extra={sorted(actual - RUNTIME_VERSION_FIELDS)}"
        )
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError(
            f"{label}: runtime_versions values must be non-empty strings"
        )
    _reject_nonfinite_numbers(payload, location=label)

    config_fields = {field.name for field in fields(GenomicsBenchmarkConfig)}
    raw_config = payload["config"]
    if not isinstance(raw_config, Mapping) or set(raw_config) != config_fields:
        actual = set(raw_config) if isinstance(raw_config, Mapping) else set()
        raise ValueError(
            f"{label}: config schema mismatch; "
            f"missing={sorted(config_fields - actual)}, "
            f"extra={sorted(actual - config_fields)}"
        )
    config_values = dict(raw_config)
    for field_name in ("seeds", "methods", "alpha_candidates"):
        if not isinstance(config_values[field_name], list):
            raise ValueError(f"{label}: config.{field_name} must be an array")
        config_values[field_name] = tuple(config_values[field_name])
    config = GenomicsBenchmarkConfig(**config_values)
    _validate_config(config)

    manifest_source = payload["manifest"]
    if not isinstance(manifest_source, Mapping):
        raise ValueError(f"{label}: manifest must be an object")
    manifest = dict(manifest_source)
    embedded = manifest.get("configuration")
    if not isinstance(embedded, Mapping):
        raise ValueError(f"{label}: manifest.configuration must be an object")
    embedded_values = dict(embedded)
    if set(embedded_values) != config_fields:
        raise ValueError(f"{label}: manifest configuration schema mismatch")
    for field_name in ("seeds", "methods", "alpha_candidates"):
        if not isinstance(embedded_values[field_name], list):
            raise ValueError(
                f"{label}: manifest.configuration.{field_name} must be an array"
            )
        embedded_values[field_name] = tuple(embedded_values[field_name])
    embedded_config = GenomicsBenchmarkConfig(**embedded_values)
    _validate_config(embedded_config)
    manifest["configuration"] = asdict(embedded_config)

    def mapping_tuple(value: Any, *, location: str) -> tuple[Mapping[str, Any], ...]:
        if not isinstance(value, list):
            raise ValueError(f"{label}: {location} must be an array")
        rows: list[Mapping[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                raise ValueError(f"{label}: {location}[{index}] must be an object")
            rows.append(dict(item))
        return tuple(rows)

    single_config_fields = {field.name for field in fields(GenomicsConfig)}
    normalized_full_runs: list[Mapping[str, Any]] = []
    for index, source in enumerate(
        mapping_tuple(payload["full_runs"], location="full_runs")
    ):
        full_run = dict(source)
        single_config = full_run.get("config")
        if not isinstance(single_config, Mapping) or set(single_config) != single_config_fields:
            raise ValueError(
                f"{label}: full_runs[{index}].config schema mismatch"
            )
        normalized_single_config = dict(single_config)
        if not isinstance(normalized_single_config["alpha_candidates"], list):
            raise ValueError(
                f"{label}: full_runs[{index}].config.alpha_candidates must be an array"
            )
        normalized_single_config["alpha_candidates"] = tuple(
            normalized_single_config["alpha_candidates"]
        )
        full_run["config"] = normalized_single_config
        normalized_full_runs.append(full_run)

    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{label}: provenance must be an object")
    if set(provenance) != FORMAL_PROVENANCE_FIELDS:
        actual = set(provenance)
        raise ValueError(
            f"{label}: provenance schema mismatch; "
            f"missing={sorted(FORMAL_PROVENANCE_FIELDS - actual)}, "
            f"extra={sorted(actual - FORMAL_PROVENANCE_FIELDS)}"
        )
    provenance_runtime = provenance["runtime_versions_at_start"]
    if (
        not isinstance(provenance_runtime, Mapping)
        or set(provenance_runtime) != RUNTIME_VERSION_FIELDS
        or any(
            not isinstance(value, str) or not value
            for value in provenance_runtime.values()
        )
    ):
        raise ValueError(
            f"{label}: provenance.runtime_versions_at_start schema mismatch"
        )
    if dict(runtime) != dict(provenance_runtime):
        raise ValueError(
            f"{label}: runtime_versions does not exactly match "
            "provenance.runtime_versions_at_start"
        )
    hash_fields = (
        "expected_genotype_sha256",
        "expected_metadata_sha256",
        "genotype_sha256",
        "metadata_sha256",
        "normalized_genotypes_sha256",
        "population_labels_sha256",
        "sample_ids_sha256",
        "snp_ids_sha256",
    )
    for field_name in hash_fields:
        _require_sha256(
            provenance[field_name],
            f"{label}: provenance.{field_name}",
        )
    if provenance["expected_genotype_sha256"] != provenance["genotype_sha256"]:
        raise ValueError(f"{label}: expected/observed genotype hashes differ")
    if provenance["expected_metadata_sha256"] != provenance["metadata_sha256"]:
        raise ValueError(f"{label}: expected/observed metadata hashes differ")
    for field_name in (
        "genotype_path",
        "metadata_path",
        "metadata_sample_id_column",
        "population_column",
    ):
        if not isinstance(provenance[field_name], str) or not provenance[field_name]:
            raise ValueError(
                f"{label}: provenance.{field_name} must be a non-empty string"
            )
    if provenance["input_orientation"] not in {
        "samples-by-snps",
        "snps-by-samples",
    }:
        raise ValueError(f"{label}: provenance.input_orientation is invalid")
    if provenance["internal_orientation"] != "samples-by-snps":
        raise ValueError(
            f"{label}: provenance.internal_orientation must be samples-by-snps"
        )
    for field_name in ("samples", "snps"):
        value = provenance[field_name]
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) <= 0
        ):
            raise ValueError(
                f"{label}: provenance.{field_name} must be a positive integer"
            )
    missing_value = provenance["missing_value"]
    if missing_value is not None:
        if isinstance(missing_value, (bool, np.bool_)) or not isinstance(
            missing_value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"{label}: provenance.missing_value must be finite or null"
            )
        _finite_scalar(float(missing_value), f"{label}: provenance.missing_value")
    if missing_value != config.missing_value:
        raise ValueError(
            f"{label}: provenance.missing_value does not match config.missing_value"
        )
    if expected_provenance is not None:
        if not isinstance(expected_provenance, Mapping):
            raise TypeError(f"{label}: expected_provenance must be an object")
        if set(expected_provenance) != FORMAL_PROVENANCE_FIELDS:
            actual = set(expected_provenance)
            raise ValueError(
                f"{label}: expected_provenance schema mismatch; "
                f"missing={sorted(FORMAL_PROVENANCE_FIELDS - actual)}, "
                f"extra={sorted(actual - FORMAL_PROVENANCE_FIELDS)}"
            )
        if dict(provenance) != dict(expected_provenance):
            raise ValueError(
                f"{label}: provenance does not match external expected_provenance"
            )
        if dict(runtime) != dict(expected_provenance["runtime_versions_at_start"]):
            raise ValueError(
                f"{label}: runtime_versions do not match external "
                "expected_provenance"
            )
    result = GenomicsBenchmarkResult(
        config=config,
        manifest=manifest,
        provenance=dict(provenance),
        records=mapping_tuple(payload["records"], location="records"),
        full_runs=tuple(normalized_full_runs),
        method_summaries=mapping_tuple(
            payload["method_summaries"], location="method_summaries"
        ),
        paired_summaries=mapping_tuple(
            payload["paired_summaries"], location="paired_summaries"
        ),
    )
    expected_protocol = _json_payload(result)["evaluation_protocol"]
    if payload["evaluation_protocol"] != expected_protocol:
        raise ValueError(f"{label}: evaluation protocol mismatch")
    canonical = lambda value: json.dumps(  # noqa: E731
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if canonical(payload["config"]) != canonical(asdict(result.config)):
        raise ValueError(f"{label}: config failed JSON round-trip validation")
    if canonical(payload["manifest"]) != canonical(result.manifest):
        raise ValueError(f"{label}: manifest failed JSON round-trip validation")
    _validate_result(result)
    for index, full_run in enumerate(result.full_runs):
        method = full_run["method"]
        requested_seed = full_run["requested_seed"]
        run_records = full_run["records"]
        full_cohort_records = [
            row
            for row in run_records
            if row.get("record_type") == "full_cohort_unsupervised_fit"
        ]
        held_out_records = [
            row
            for row in run_records
            if row.get("record_type") == "held_out_population_classification"
        ]
        if len(full_cohort_records) != 1:
            raise ValueError(
                f"{label}: full_runs[{index}] must contain exactly one "
                "full_cohort_unsupervised_fit record"
            )
        if len(held_out_records) != 1:  # Also checked by _validate_result.
            raise ValueError(
                f"{label}: full_runs[{index}] must contain exactly one "
                "held_out_population_classification record"
            )
        full_cohort = full_cohort_records[0]
        held_out = held_out_records[0]

        def positive_integer(record: Mapping[str, Any], field_name: str) -> int:
            value = record.get(field_name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) <= 0
            ):
                raise ValueError(
                    f"{label}: full_runs[{index}].{field_name} must be a "
                    "positive integer"
                )
            return int(value)

        if full_cohort.get("method") != method or held_out.get("method") != method:
            raise ValueError(
                f"{label}: full_runs[{index}] record methods do not match the run"
            )
        full_samples = positive_integer(full_cohort, "samples")
        input_snps = positive_integer(full_cohort, "input_snps")
        full_retained = positive_integer(full_cohort, "retained_snps")
        held_retained = positive_integer(held_out, "retained_snps")
        train_samples = positive_integer(held_out, "train_samples")
        test_samples = positive_integer(held_out, "test_samples")
        inner_fit_samples = positive_integer(held_out, "inner_fit_samples")
        validation_samples = held_out.get("validation_samples")
        if (
            isinstance(validation_samples, (bool, np.bool_))
            or not isinstance(validation_samples, (int, np.integer))
            or int(validation_samples) < 0
        ):
            raise ValueError(
                f"{label}: full_runs[{index}].validation_samples must be a "
                "non-negative integer"
            )
        if full_samples != int(result.provenance["samples"]):
            raise ValueError(
                f"{label}: full_runs[{index}] full-cohort samples do not match "
                "provenance.samples"
            )
        if input_snps != int(result.provenance["snps"]):
            raise ValueError(
                f"{label}: full_runs[{index}] input_snps do not match "
                "provenance.snps"
            )
        if train_samples + test_samples != full_samples:
            raise ValueError(
                f"{label}: full_runs[{index}] train_samples + test_samples does "
                "not match the full cohort"
            )
        if inner_fit_samples + int(validation_samples) != train_samples:
            raise ValueError(
                f"{label}: full_runs[{index}] inner_fit_samples + "
                "validation_samples does not match train_samples"
            )
        if full_retained > input_snps or held_retained > input_snps:
            raise ValueError(
                f"{label}: full_runs[{index}] retained SNP count exceeds "
                "provenance.snps"
            )
        if full_cohort.get("preprocessing_scope") != "full_cohort_unsupervised":
            raise ValueError(
                f"{label}: full_runs[{index}] full-cohort preprocessing scope "
                "is invalid"
            )
        if held_out.get("preprocessing_fit_scope") != "train_only":
            raise ValueError(
                f"{label}: full_runs[{index}] held-out preprocessing scope is invalid"
            )
        for record_name, record in (
            ("full_cohort", full_cohort),
            ("held_out", held_out),
        ):
            for field_name in (
                "retained_snp_mask_sha256",
                "retained_snp_ids_sha256",
            ):
                digest = _require_sha256(
                    record.get(field_name),
                    f"{label}: full_runs[{index}].{record_name}.{field_name}",
                )
                if digest != record[field_name]:
                    raise ValueError(
                        f"{label}: full_runs[{index}].{record_name}."
                        f"{field_name} must be lowercase"
                    )

        # These two masks intentionally need not be equal: full-cohort
        # preprocessing is used only for unsupervised diagnostics, whereas the
        # held-out classifier fits preprocessing on the outer training split.
        observed_manifest = full_run["seed_manifest"]
        if not isinstance(observed_manifest, Mapping):
            raise ValueError(
                f"{label}: full_runs[{index}].seed_manifest must be an object"
            )
        expected_manifest = _expected_single_run_manifest(
            result.config,
            result.provenance,
            method=method,
            seed=int(requested_seed),
        )
        if canonical(observed_manifest) != canonical(expected_manifest):
            raise ValueError(
                f"{label}: full_runs[{index}].seed_manifest does not exactly "
                "bind its config, dimensions, hashes, and seed roles"
            )
    return result


def _canonical_json_cell(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _csv_rows(
    result: GenomicsBenchmarkResult,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    for source in result.records:
        record = dict(source)
        classification = record.pop("classification_record")
        record["classification_record"] = _canonical_json_cell(classification)
        record["alpha_selection_candidates"] = _canonical_json_cell(
            record.get("alpha_selection_candidates")
        )
        record["class_order"] = _canonical_json_cell(record.get("class_order"))
        rows.append({"row_type": "per_seed", **record})
    rows.extend(
        {
            "row_type": "method_summary",
            "benchmark_manifest_sha256": result.manifest["manifest_sha256"],
            **dict(summary),
        }
        for summary in result.method_summaries
    )
    rows.extend(
        {
            "row_type": "paired_summary",
            "benchmark_manifest_sha256": result.manifest["manifest_sha256"],
            **dict(summary),
        }
        for summary in result.paired_summaries
    )
    preferred = (
        "row_type",
        "benchmark_manifest_sha256",
        "single_run_manifest_sha256",
        "run",
        "run_index",
        "requested_seed",
        "method",
        "metric",
        *CORE_METRICS,
        "runtime_seconds",
        "selected_alpha",
        "alpha_selection_status",
        "alpha_selection_qualified",
        "solver_converged",
        "solver_status",
        "solver_n_iter",
        "solver_final_stationarity",
        "n_seeds",
        "mean",
        "std",
        "baseline_method",
        "difference_direction",
        "n_pairs",
        "mean_difference",
        "std_difference",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "bootstrap_resamples",
        "bootstrap_seed",
        "bootstrap_unit",
    )
    all_fields = set().union(*(row.keys() for row in rows)) if rows else set()
    fieldnames = list(
        dict.fromkeys(field for field in preferred if field in all_fields)
    )
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    return rows, fieldnames


def _resolved_output_paths(
    output_json: str | Path | None,
    output_csv: str | Path | None,
) -> dict[str, Path]:
    requested = {
        name: Path(path)
        for name, path in (("json", output_json), ("csv", output_csv))
        if path is not None
    }
    resolved = [path.resolve(strict=False) for path in requested.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output_json and output_csv must be different paths")
    existing = [str(path) for path in requested.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing}")
    return requested


def validate_output_paths(
    *, output_json: str | Path | None, output_csv: str | Path | None
) -> None:
    """Public preflight used by the CLI before expensive computation starts."""

    _resolved_output_paths(output_json, output_csv)


def _csv_text(result: GenomicsBenchmarkResult) -> str:
    rows, fieldnames = _csv_rows(result)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    if fieldnames:
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    return buffer.getvalue()


def _exclusive_write_texts(contents: Mapping[str, tuple[Path, str]]) -> None:
    """Reserve every output before writing and roll back this call on failure."""

    reserved: dict[str, tuple[Path, int]] = {}
    created_paths: list[Path] = []
    try:
        for name, (path, _) in contents.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o666,
            )
            reserved[name] = (path, descriptor)
            created_paths.append(path)
        for name, (_, text) in contents.items():
            path, descriptor = reserved[name]
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                reserved[name] = (path, -1)
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
    except BaseException:
        for _, descriptor in reserved.values():
            if descriptor >= 0:
                os.close(descriptor)
        for path in created_paths:
            path.unlink(missing_ok=True)
        raise


def write_results(
    result: GenomicsBenchmarkResult,
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> dict[str, Path]:
    """Exclusive-create strict JSON and tidy CSV artifacts."""

    _validate_result(result)
    paths = _resolved_output_paths(output_json, output_csv)
    contents: dict[str, tuple[Path, str]] = {}
    if "json" in paths:
        contents["json"] = (
            paths["json"],
            json.dumps(
                _json_safe(_json_payload(result)),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )
    if "csv" in paths:
        contents["csv"] = (paths["csv"], _csv_text(result))
    _exclusive_write_texts(contents)
    return paths


def _decode_delimiter(value: str | None) -> str | None:
    if value is None:
        return None
    decoded = bytes(value, "utf-8").decode("unicode_escape")
    if len(decoded) != 1:
        raise argparse.ArgumentTypeError("delimiter must decode to one character")
    return decoded


def _parse_missing_value(value: str) -> float | None:
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "missing value must be a finite number or 'none'"
        ) from error
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            "missing value must be a finite number or 'none'"
        )
    return parsed


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    try:
        for token in value.split(","):
            token = token.strip()
            if not token:
                raise ValueError
            if ".." in token:
                endpoints = token.split("..")
                if len(endpoints) != 2:
                    raise ValueError
                start, stop = (int(endpoint.strip()) for endpoint in endpoints)
                if start > stop:
                    raise ValueError
                seeds.extend(range(start, stop + 1))
            else:
                seeds.append(int(token))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be non-negative comma-separated integers or inclusive ranges "
            "such as 2025..2044"
        ) from error
    if any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique and non-negative")
    return tuple(seeds)


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(
        token.strip().lower() for token in value.split(",") if token.strip()
    )
    if not methods or any(method not in METHODS for method in methods):
        raise argparse.ArgumentTypeError(
            "methods must be comma-separated pca,vp,randomized,robust"
        )
    return methods


def _parse_alpha_candidates(value: str) -> tuple[float, ...]:
    try:
        candidates = tuple(
            float(token.strip()) for token in value.split(",") if token.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "alpha candidates must be comma-separated numbers"
        ) from error
    if (
        not candidates
        or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise argparse.ArgumentTypeError(
            "alpha candidates must be unique finite non-negative numbers"
        )
    return candidates


def _parse_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected a 64-character SHA-256 hex digest")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genotypes", type=Path)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--expected-genotype-sha256", type=_parse_sha256, required=True)
    parser.add_argument("--expected-metadata-sha256", type=_parse_sha256, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=tuple(range(2025, 2045)))
    parser.add_argument("--methods", type=_parse_methods, default=METHODS)
    parser.add_argument(
        "--orientation",
        choices=("samples-by-snps", "snps-by-samples"),
        default="samples-by-snps",
    )
    parser.add_argument("--delimiter", type=_decode_delimiter)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--row-id-column")
    parser.add_argument("--missing-value", type=_parse_missing_value, default=-9.0)
    parser.add_argument("--metadata-delimiter", type=_decode_delimiter, default=",")
    parser.add_argument("--metadata-sample-id-column", default="sample_id")
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--alpha-candidates",
        type=_parse_alpha_candidates,
        default=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--randomized-max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--selection-threshold", type=float, default=1e-8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--oversample", type=int, default=30)
    parser.add_argument("--n-power-iter", type=int, default=4)
    parser.add_argument("--randomized-full-gradient", action="store_true")
    parser.add_argument("--randomized-finalize-full-iters", type=int, default=500)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2025)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    validate_output_paths(output_json=args.output_json, output_csv=args.output_csv)
    genotype_sha256 = _file_sha256(args.genotypes)
    metadata_sha256 = _file_sha256(args.metadata)
    if genotype_sha256 != args.expected_genotype_sha256:
        raise ValueError(
            "genotype SHA-256 mismatch: "
            f"expected {args.expected_genotype_sha256}, observed {genotype_sha256}"
        )
    if metadata_sha256 != args.expected_metadata_sha256:
        raise ValueError(
            "metadata SHA-256 mismatch: "
            f"expected {args.expected_metadata_sha256}, observed {metadata_sha256}"
        )

    dataset = load_genotypes(
        args.genotypes,
        orientation=args.orientation,
        delimiter=args.delimiter,
        has_header=not args.no_header,
        row_id_column=args.row_id_column,
        missing_value=args.missing_value,
    )
    labels = load_population_labels(
        args.metadata,
        dataset.sample_ids,
        sample_id_column=args.metadata_sample_id_column,
        population_column=args.population_column,
        delimiter=args.metadata_delimiter,
    )
    config = GenomicsBenchmarkConfig(
        seeds=args.seeds,
        methods=args.methods,
        rank=args.rank,
        alpha=args.alpha,
        alpha_candidates=args.alpha_candidates,
        validation_fraction=args.validation_fraction,
        max_iter=args.max_iter,
        randomized_max_iter=args.randomized_max_iter,
        tol=args.tol,
        selection_threshold=args.selection_threshold,
        test_fraction=args.test_fraction,
        standardize=not args.no_standardize,
        oversample=args.oversample,
        n_power_iter=args.n_power_iter,
        randomized_full_gradient=args.randomized_full_gradient,
        randomized_finalize_full_iters=args.randomized_finalize_full_iters,
        missing_value=args.missing_value,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    provenance = {
        "genotype_path": str(args.genotypes.resolve()),
        "metadata_path": str(args.metadata.resolve()),
        "genotype_sha256": genotype_sha256,
        "metadata_sha256": metadata_sha256,
        "expected_genotype_sha256": args.expected_genotype_sha256,
        "expected_metadata_sha256": args.expected_metadata_sha256,
        "input_orientation": args.orientation,
        "internal_orientation": "samples-by-snps",
        "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
        "sample_ids_sha256": array_sha256(dataset.sample_ids),
        "snp_ids_sha256": array_sha256(dataset.snp_ids),
        "population_labels_sha256": array_sha256(labels),
        "samples": int(dataset.genotypes.shape[0]),
        "snps": int(dataset.genotypes.shape[1]),
        "population_column": args.population_column,
        "metadata_sample_id_column": args.metadata_sample_id_column,
        "missing_value": args.missing_value,
        "runtime_versions_at_start": runtime_versions(),
    }
    result = run_genomics_benchmark(
        dataset,
        labels,
        config,
        provenance=provenance,
    )
    paths = write_results(
        result,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": result.manifest["manifest_sha256"],
                "records": len(result.records),
                "full_runs": len(result.full_runs),
                "method_summaries": len(result.method_summaries),
                "paired_summaries": len(result.paired_summaries),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CORE_METRICS",
    "METHODS",
    "GenomicsBenchmarkConfig",
    "GenomicsBenchmarkResult",
    "main",
    "run_genomics_benchmark",
    "summarize_method_metrics",
    "summarize_paired_differences",
    "validate_result_payload",
    "validate_output_paths",
    "write_results",
]

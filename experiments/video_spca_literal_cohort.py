#!/usr/bin/env python3
"""Current-pipeline video benchmark for literal seeds 91001 through 91020.

This runner exists specifically to reproduce the historical *literal* seed
cohort without relabelling it as the content-derived seed-manifest cohort used
by :mod:`experiments.video_spca_benchmark`.  The seed list is a module constant,
cannot be changed through the CLI, and is recorded as ``literal_predeclared`` in
the config, every per-seed record, the cohort manifest, and the protocol.

JSON and CSV contain every :class:`experiments.video_spca.MethodReport` field,
full per-dataset content-addressed provenance, deterministic method/paired
summaries, and summaries of ``corruption_detection_at_budget``.  Outputs are
written only after validation into one atomically claimed new directory.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

if __package__:
    from .video_spca import (
        DEFAULT_METHODS,
        MethodReport,
        VideoExperimentConfig,
        _claim_and_write_output_directory,
        _json_safe,
        _normalize_undefined_zero_corruption_metrics,
        _sha256_bytes,
        _strict_json_load,
        _video_dataset_manifest,
        run_video_experiment,
        validate_video_output_dir,
    )
    from .video_spca_benchmark import (
        CORE_METRICS,
        VideoBenchmarkConfig,
        _fixed_video_config,
        _validate_config as _validate_benchmark_config,
        summarize_method_metrics,
        summarize_paired_differences as _summarize_manifest_bound_paired_differences,
    )
else:  # Support direct execution from the repository root.
    from video_spca import (  # type: ignore[no-redef]
        DEFAULT_METHODS,
        MethodReport,
        VideoExperimentConfig,
        _claim_and_write_output_directory,
        _json_safe,
        _normalize_undefined_zero_corruption_metrics,
        _sha256_bytes,
        _strict_json_load,
        _video_dataset_manifest,
        run_video_experiment,
        validate_video_output_dir,
    )
    from video_spca_benchmark import (  # type: ignore[no-redef]
        CORE_METRICS,
        VideoBenchmarkConfig,
        _fixed_video_config,
        _validate_config as _validate_benchmark_config,
        summarize_method_metrics,
        summarize_paired_differences as _summarize_manifest_bound_paired_differences,
    )

from rspca.datasets import canonical_sha256, runtime_versions


LITERAL_SEEDS = tuple(range(91001, 91021))
SEED_SOURCE = "literal_predeclared"
COHORT_ID = "video_spca_current_pipeline_literal_91001_through_91020"
_OUTPUT_MANIFEST_NAME = "literal_video_cohort_output_manifest.json"

_EXECUTION_SOURCE_PATHS = (
    "experiments/video_spca_literal_cohort.py",
    "experiments/video_spca.py",
    "experiments/video_spca_benchmark.py",
)

_SCIENTIFIC_RSPCA_PATHS = (
    "src/rspca/accelerated.py",
    "src/rspca/datasets.py",
    "src/rspca/metrics.py",
    "src/rspca/randomized.py",
    "src/rspca/robust.py",
    "src/rspca/vp.py",
)

_PACKAGE_INITIALIZER_PATH = "src/rspca/__init__.py"
_PACKAGE_INITIALIZATION_ONLY_PATHS = (
    "src/rspca/cross_modal_gates.py",
    "src/rspca/federated.py",
    "src/rspca/fedrpca_audit.py",
    "src/rspca/gauge.py",
    "src/rspca/gauge_stress.py",
    "src/rspca/product_aggregation.py",
    "src/rspca/statistical_gates.py",
    "src/rspca/trust_audit.py",
)
_EXPECTED_EAGER_RSPCA_MODULES = (
    "accelerated",
    "cross_modal_gates",
    "datasets",
    "federated",
    "fedrpca_audit",
    "gauge",
    "gauge_stress",
    "metrics",
    "product_aggregation",
    "randomized",
    "robust",
    "statistical_gates",
    "trust_audit",
    "vp",
)

_RUNTIME_ENVIRONMENT_KEYS = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_BUDGET_SUMMARY_METRICS = (
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "true_positives",
    "false_positives",
    "false_negatives",
    "true_negatives",
    "budget_top_k",
    "budget_fraction",
)


def _formal_video_config() -> VideoExperimentConfig:
    """Return the exact current formal-video settings used for cohort comparison."""

    return VideoExperimentConfig(
        grid_size=32,
        n_timesteps=180,
        total_time=45.0,
        n_components=3,
        noise_std=0.03,
        corruption_fraction=0.01,
        corruption_scale=1.5,
        seed=LITERAL_SEEDS[0],
        alpha=0.08,
        max_iter=1000,
        tol=1e-5,
        randomized_oversample=15,
        randomized_power_iters=3,
        randomized_full_refine_iters=500,
        support_threshold=1e-10,
        outlier_support_threshold=1e-10,
    )


@dataclass(frozen=True)
class LiteralVideoCohortConfig:
    """Fixed solver settings for the immutable literal seed cohort."""

    video_config: VideoExperimentConfig = field(default_factory=_formal_video_config)
    methods: tuple[str, ...] = DEFAULT_METHODS
    bootstrap_resamples: int = 100_000
    bootstrap_seed: int = 260_826


@dataclass(frozen=True)
class LiteralVideoCohortResult:
    """Complete in-memory literal-cohort evidence before artifact creation."""

    config: LiteralVideoCohortConfig
    cohort_manifest: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    method_summaries: tuple[Mapping[str, Any], ...]
    paired_summaries: tuple[Mapping[str, Any], ...]
    corruption_budget_summaries: tuple[Mapping[str, Any], ...]


def _formal_default_contract(config: LiteralVideoCohortConfig) -> dict[str, Any]:
    reference = LiteralVideoCohortConfig()
    reference_fixed = _fixed_video_config(reference.video_config)
    exact = (
        _fixed_video_config(config.video_config) == reference_fixed
        and config.methods == reference.methods
        and config.bootstrap_resamples == reference.bootstrap_resamples
        and config.bootstrap_seed == reference.bootstrap_seed
    )
    return {
        "exact": exact,
        "reference_fixed_video_config_sha256": canonical_sha256(reference_fixed),
        "reference_methods": list(reference.methods),
        "reference_bootstrap_resamples": reference.bootstrap_resamples,
        "reference_bootstrap_seed": reference.bootstrap_seed,
    }


def _validate_config(config: LiteralVideoCohortConfig) -> None:
    if not isinstance(config, LiteralVideoCohortConfig):
        raise TypeError("config must be a LiteralVideoCohortConfig")
    # Reuse the formal benchmark's validation without reusing its derived seeds.
    _validate_benchmark_config(
        VideoBenchmarkConfig(
            video_config=config.video_config,
            repeats=len(LITERAL_SEEDS),
            base_seed=LITERAL_SEEDS[0],
            methods=config.methods,
            bootstrap_resamples=config.bootstrap_resamples,
            bootstrap_seed=config.bootstrap_seed,
        )
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_import_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module:
            modules.add(node.module.split(".", maxsplit=1)[0])
    return modules


def _source_closure_contract(source_root: str | Path | None = None) -> dict[str, Any]:
    """Build the source-closure contract from a live or frozen source tree."""

    root = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else Path(source_root)
    )
    initializer = root / _PACKAGE_INITIALIZER_PATH
    eager_modules = _relative_import_modules(initializer)
    expected = set(_EXPECTED_EAGER_RSPCA_MODULES)
    if eager_modules != expected:
        missing = sorted(eager_modules - expected)
        stale = sorted(expected - eager_modules)
        raise RuntimeError(
            "rspca package eager-import closure changed; "
            f"unbound_modules={missing}, no_longer_imported={stale}"
        )

    bound_rspca_modules = {
        Path(relative).stem
        for relative in (
            *_SCIENTIFIC_RSPCA_PATHS,
            *_PACKAGE_INITIALIZATION_ONLY_PATHS,
        )
    }
    transitive_edges: list[dict[str, str]] = []
    for module in sorted(eager_modules):
        path = root / "src" / "rspca" / f"{module}.py"
        for dependency in sorted(_relative_import_modules(path)):
            if dependency not in bound_rspca_modules:
                raise RuntimeError(
                    f"rspca eager module {module!r} imports unbound dependency "
                    f"{dependency!r}"
                )
            transitive_edges.append({"from": module, "to": dependency})

    return {
        "policy": (
            "AST-verified rspca package initializer and transitive relative-import "
            "closure; initialization-only modules are provenance, not video methods"
        ),
        "package_initializer": _PACKAGE_INITIALIZER_PATH,
        "eager_import_modules": sorted(eager_modules),
        "transitive_relative_import_edges": transitive_edges,
        "execution_source_paths": list(_EXECUTION_SOURCE_PATHS),
        "scientific_rspca_paths": list(_SCIENTIFIC_RSPCA_PATHS),
        "package_initialization_only_paths": list(
            _PACKAGE_INITIALIZATION_ONLY_PATHS
        ),
    }


def _source_snapshot(
    source_root: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Hash the declared execution closure from a live or frozen source tree."""

    root = (
        Path(__file__).resolve().parents[1]
        if source_root is None
        else Path(source_root)
    )
    closure = _source_closure_contract(root)
    roles = {
        **{relative: "execution" for relative in _EXECUTION_SOURCE_PATHS},
        **{
            relative: "scientific_or_provenance_dependency"
            for relative in _SCIENTIFIC_RSPCA_PATHS
        },
        _PACKAGE_INITIALIZER_PATH: "package_initializer",
        **{
            relative: "package_initialization_only"
            for relative in _PACKAGE_INITIALIZATION_ONLY_PATHS
        },
    }
    source_paths = (
        *_EXECUTION_SOURCE_PATHS,
        *_SCIENTIFIC_RSPCA_PATHS,
        _PACKAGE_INITIALIZER_PATH,
        *_PACKAGE_INITIALIZATION_ONLY_PATHS,
    )
    expected_paths = {
        *closure["execution_source_paths"],
        *closure["scientific_rspca_paths"],
        closure["package_initializer"],
        *closure["package_initialization_only_paths"],
    }
    if set(source_paths) != expected_paths or set(roles) != expected_paths:
        raise RuntimeError("source closure paths and source roles disagree")
    records: list[dict[str, Any]] = []
    for relative in source_paths:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required source file is missing or a symlink: {path}")
        records.append(
            {
                "path": relative,
                "role": roles[relative],
                "scientific_method_dependency": relative
                in _SCIENTIFIC_RSPCA_PATHS,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return tuple(records)


def _validate_frozen_source_snapshot(source_root: str | Path) -> None:
    """Require an exact, regular-file-only historical execution snapshot."""

    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("literal cohort source snapshot must be a real directory")
    expected = {record["path"] for record in _source_snapshot(root)}
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("literal cohort source snapshot contains a symlink")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                raise ValueError("literal cohort source snapshot contains bytecode")
            actual.add(relative)
        elif not path.is_dir():
            raise ValueError("literal cohort source snapshot contains a special file")
    if actual != expected:
        raise ValueError("literal cohort source snapshot file closure mismatch")


def _execution_environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    bytecode_paths = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*.pyc")
        if path.is_file()
    )
    if os.environ.get("PYTHONDONTWRITEBYTECODE") == "1" and bytecode_paths:
        preview = bytecode_paths[:5]
        raise RuntimeError(
            "PYTHONDONTWRITEBYTECODE=1 requires a bytecode-free isolated source "
            f"tree; found={preview}, total={len(bytecode_paths)}"
        )
    return {
        "variables": {key: os.environ.get(key) for key in _RUNTIME_ENVIRONMENT_KEYS},
        "bytecode_policy": {
            "required_for_formal_remote_command": "PYTHONDONTWRITEBYTECODE=1",
            "source_tree_bytecode_free": not bytecode_paths,
            "pyc_file_count_at_start": len(bytecode_paths),
        },
    }


def _normalized_method_report(
    report: MethodReport, *, zero_corruption: bool
) -> dict[str, Any]:
    payload = {"methods": {report.method: asdict(report)}}
    _normalize_undefined_zero_corruption_metrics(
        payload,
        zero_corruption=zero_corruption,
    )
    normalized = _json_safe(payload)["methods"][report.method]
    if not isinstance(normalized, dict):
        raise TypeError("normalized MethodReport must be a mapping")
    return normalized


def _cohort_manifest_body(
    config: LiteralVideoCohortConfig,
    *,
    sources: Sequence[Mapping[str, Any]],
    versions: Mapping[str, Any],
    environment: Mapping[str, Any],
    run_provenance: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed = _fixed_video_config(config.video_config)
    return {
        "schema_version": 1,
        "artifact": "current_pipeline_literal_video_seed_cohort",
        "cohort_id": COHORT_ID,
        "seed_source": SEED_SOURCE,
        "seeds": list(LITERAL_SEEDS),
        "seed_count": len(LITERAL_SEEDS),
        "manifest_derived_seed_generation": False,
        "distinct_from_manifest_derived_base_91001_cohort": True,
        "formal_default_config": _formal_default_contract(config),
        "fixed_video_config_excluding_seed": fixed,
        "fixed_video_config_sha256": canonical_sha256(fixed),
        "methods": list(config.methods),
        "bootstrap_policy": {
            "unit": "literal_seed",
            "difference": "method_minus_pca",
            "interval": "percentile_95",
            "resamples": int(config.bootstrap_resamples),
            "seed": int(config.bootstrap_seed),
            "runtime_included": False,
            "unqualified_runs_retained": True,
            "inference_requires_every_pair_converged": True,
        },
        "hyperparameter_selection": "none_fixed_predeclared_configuration",
        "runtime_versions": dict(versions),
        "execution_environment": dict(environment),
        "source_closure": _source_closure_contract(),
        "source_files": [dict(record) for record in sources],
        "run_provenance": [dict(record) for record in run_provenance],
    }


def summarize_corruption_budget_metrics(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate every declared top-k corruption-budget metric by method."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        detection = record.get("corruption_detection_at_budget")
        if detection is None:
            continue
        if not isinstance(detection, Mapping):
            raise TypeError("corruption_detection_at_budget must be a mapping or null")
        grouped.setdefault(str(record["method"]), []).append(record)

    summaries: list[dict[str, Any]] = []
    for method in sorted(grouped):
        method_records = sorted(
            grouped[method], key=lambda record: (int(record["run"]), int(record["seed"]))
        )
        if len(method_records) != len(LITERAL_SEEDS):
            raise ValueError(
                f"method {method!r} lacks a corruption-budget record for every literal seed"
            )
        converged_count = sum(bool(record["converged"]) for record in method_records)
        for metric in _BUDGET_SUMMARY_METRICS:
            raw_values = [
                record["corruption_detection_at_budget"].get(metric)
                for record in method_records
            ]
            defined = [value for value in raw_values if value is not None]
            if len(defined) not in {0, len(raw_values)}:
                raise ValueError(
                    f"corruption-budget metric {metric!r} is only partially defined"
                )
            if not defined:
                if metric not in {"roc_auc", "average_precision"}:
                    raise ValueError(
                        f"corruption-budget metric {metric!r} cannot be null"
                    )
                mean = std = minimum = maximum = None
                status = "undefined_zero_corruption_truth"
            else:
                values = np.asarray(defined, dtype=np.float64)
                if not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"corruption-budget metric {metric!r} must be finite"
                    )
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1))
                minimum = float(np.min(values))
                maximum = float(np.max(values))
                status = "defined"
            summaries.append(
                {
                    "method": method,
                    "metric": f"corruption_detection_at_budget.{metric}",
                    "n_seeds": len(method_records),
                    "n_defined": len(defined),
                    "mean": mean,
                    "std": std,
                    "min": minimum,
                    "max": maximum,
                    "status": status,
                    "converged_count": int(converged_count),
                    "convergence_rate": converged_count / len(method_records),
                    "metric_scope": (
                        "deterministic_fixed_literal_seed_corruption_budget_metric"
                    ),
                }
            )
    return summaries


def _summarize_literal_paired_differences(
    records: Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Reuse the portable paired statistics without emitting a seed manifest.

    The established summarizer verifies a field named ``seed_manifest_sha256``.
    A transient in-memory adapter supplies the literal cohort digest under that
    compatibility name; neither the persisted records nor the cohort manifest
    claims that the literal seeds were generated by a seed manifest.
    """

    adapted = [
        {
            **dict(record),
            "seed_manifest_sha256": record["cohort_manifest_sha256"],
        }
        for record in records
    ]
    return _summarize_manifest_bound_paired_differences(
        adapted,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )


def run_literal_video_cohort(
    config: LiteralVideoCohortConfig | None = None,
    *,
    progress: bool = False,
) -> LiteralVideoCohortResult:
    """Run the current pipeline once on every immutable literal seed."""

    resolved = LiteralVideoCohortConfig() if config is None else config
    _validate_config(resolved)
    fixed = _fixed_video_config(resolved.video_config)
    fixed_sha256 = canonical_sha256(fixed)
    sources = _source_snapshot()
    versions = runtime_versions()
    environment = _execution_environment()

    records: list[dict[str, Any]] = []
    run_provenance: list[dict[str, Any]] = []
    for run_index, seed in enumerate(LITERAL_SEEDS):
        run_config = replace(resolved.video_config, seed=seed)
        if _fixed_video_config(run_config) != fixed:
            raise RuntimeError("video configuration changed across literal seeds")
        experiment = run_video_experiment(run_config, methods=resolved.methods)
        if tuple(experiment.reports) != resolved.methods:
            raise RuntimeError("video runner returned methods in an unexpected order")
        dataset_manifest = _video_dataset_manifest(experiment.dataset)
        dataset_manifest_sha256 = str(dataset_manifest["manifest_sha256"])
        if experiment.dataset.digest != dataset_manifest["sha256"]:
            raise RuntimeError("dataset digest and dataset manifest disagree")
        if {
            report.data_sha256 for report in experiment.reports.values()
        } != {experiment.dataset.digest}:
            raise RuntimeError("methods were not evaluated on one literal-seed dataset")
        if {
            report.dataset_manifest_sha256 for report in experiment.reports.values()
        } != {dataset_manifest_sha256}:
            raise RuntimeError("method reports do not bind the dataset manifest")

        run_provenance.append(
            {
                "run": run_index,
                "run_index": run_index,
                "seed": seed,
                "data_sha256": experiment.dataset.digest,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "dataset_manifest": dataset_manifest,
            }
        )
        zero_corruption = not bool(np.any(experiment.dataset.corruption_support))
        for method in resolved.methods:
            method_report = _normalized_method_report(
                experiment.reports[method],
                zero_corruption=zero_corruption,
            )
            records.append(
                {
                    "seed_source": SEED_SOURCE,
                    "fixed_video_config_sha256": fixed_sha256,
                    "run": run_index,
                    "run_index": run_index,
                    "seed": seed,
                    **method_report,
                    "method_report_sha256": canonical_sha256(method_report),
                }
            )
        if progress:
            print(
                json.dumps(
                    {
                        "completed": run_index + 1,
                        "total": len(LITERAL_SEEDS),
                        "seed": seed,
                        "data_sha256": experiment.dataset.digest,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )

    manifest = _cohort_manifest_body(
        resolved,
        sources=sources,
        versions=versions,
        environment=environment,
        run_provenance=run_provenance,
    )
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_sha256 = str(manifest["manifest_sha256"])
    for record in records:
        record["cohort_manifest_sha256"] = manifest_sha256

    method_summaries = summarize_method_metrics(records)
    paired_summaries = _summarize_literal_paired_differences(
        records,
        bootstrap_resamples=resolved.bootstrap_resamples,
        bootstrap_seed=resolved.bootstrap_seed,
    )
    corruption_budget_summaries = summarize_corruption_budget_metrics(records)
    return LiteralVideoCohortResult(
        config=resolved,
        cohort_manifest=manifest,
        records=tuple(records),
        method_summaries=tuple(method_summaries),
        paired_summaries=tuple(paired_summaries),
        corruption_budget_summaries=tuple(corruption_budget_summaries),
    )


def _config_payload(config: LiteralVideoCohortConfig) -> dict[str, Any]:
    return {
        "cohort_id": COHORT_ID,
        "seed_source": SEED_SOURCE,
        "seeds": list(LITERAL_SEEDS),
        "formal_default_config": _formal_default_contract(config),
        "fixed_video_config_excluding_seed": _fixed_video_config(config.video_config),
        "methods": list(config.methods),
        "bootstrap_resamples": int(config.bootstrap_resamples),
        "bootstrap_seed": int(config.bootstrap_seed),
    }


def _json_payload(result: LiteralVideoCohortResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact": "current_pipeline_literal_video_seed_cohort",
        "config": _config_payload(result.config),
        "runtime_versions": result.cohort_manifest["runtime_versions"],
        "execution_environment": result.cohort_manifest["execution_environment"],
        "cohort_manifest": result.cohort_manifest,
        "evaluation_protocol": {
            "seed_source": SEED_SOURCE,
            "seed_list_is_literal_and_predeclared": True,
            "manifest_derived_seed_generation": False,
            "distinct_from_manifest_derived_base_91001_cohort": True,
            "formal_default_config_exact": _formal_default_contract(result.config)[
                "exact"
            ],
            "pairing_unit": "literal_seed",
            "paired_difference": "method_minus_pca_on_identical_dataset",
            "shared_dataset_across_methods": True,
            "bootstrap_unit": "literal_seed",
            "bootstrap_interval": "percentile_95",
            "runtime_bootstrapped": False,
            "hyperparameter_policy": "fixed_predeclared",
            "ground_truth_used_for_selection": False,
            "ground_truth_usage": "post_fit_scoring_and_alignment_only",
            "post_hoc_amplitude_fit": False,
            "outlier_budget_source": "video_config.corruption_fraction",
            "best_seed_selection": False,
            "unqualified_runs_retained": True,
            "bootstrap_inference_requires_every_pair_converged": True,
        },
        "records": list(result.records),
        "method_summaries": list(result.method_summaries),
        "paired_summaries": list(result.paired_summaries),
        "corruption_budget_summaries": list(result.corruption_budget_summaries),
    }


def _validate_result(result: LiteralVideoCohortResult) -> None:
    """Rebuild the literal manifest, record matrix, and every summary."""

    _validate_config(result.config)
    _json_safe(_json_payload(result))
    method_fields = set(MethodReport.__dataclass_fields__)
    expected_fields = method_fields | {
        "seed_source",
        "fixed_video_config_sha256",
        "run",
        "run_index",
        "seed",
        "method_report_sha256",
        "cohort_manifest_sha256",
    }
    if len(result.records) != len(LITERAL_SEEDS) * len(result.config.methods):
        raise ValueError("literal-cohort record matrix is incomplete")

    records_by_identity: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    fixed_sha256 = canonical_sha256(_fixed_video_config(result.config.video_config))
    for index, record in enumerate(result.records):
        location = f"records[{index}]"
        if set(record) != expected_fields:
            raise ValueError(f"{location} schema does not preserve one complete MethodReport")
        run = int(record["run"])
        run_index = int(record["run_index"])
        seed = int(record["seed"])
        method = str(record["method"])
        if run != run_index or run < 0 or run >= len(LITERAL_SEEDS):
            raise ValueError(f"{location} has an invalid run identity")
        if seed != LITERAL_SEEDS[run]:
            raise ValueError(f"{location} is not an exact literal-seed record")
        if method not in result.config.methods:
            raise ValueError(f"{location}.method was not predeclared")
        identity = (run, seed, method)
        if identity in records_by_identity:
            raise ValueError(f"duplicate literal-cohort record: {identity}")
        records_by_identity[identity] = record
        if record["seed_source"] != SEED_SOURCE:
            raise ValueError(f"{location}.seed_source must equal {SEED_SOURCE!r}")
        if record["fixed_video_config_sha256"] != fixed_sha256:
            raise ValueError(f"{location} has a mismatched fixed video config")
        report_payload = {field_name: record[field_name] for field_name in method_fields}
        if record["method_report_sha256"] != canonical_sha256(report_payload):
            raise ValueError(f"{location} MethodReport SHA-256 mismatch")

    expected_identities = {
        (run, seed, method)
        for run, seed in enumerate(LITERAL_SEEDS)
        for method in result.config.methods
    }
    if set(records_by_identity) != expected_identities:
        raise ValueError("literal-cohort record identities are missing or duplicated")

    raw_provenance = result.cohort_manifest.get("run_provenance")
    if not isinstance(raw_provenance, list) or len(raw_provenance) != len(LITERAL_SEEDS):
        raise ValueError("cohort manifest lacks complete run provenance")
    run_provenance: list[dict[str, Any]] = []
    for run, seed in enumerate(LITERAL_SEEDS):
        provenance = raw_provenance[run]
        if not isinstance(provenance, Mapping):
            raise TypeError("run provenance entries must be mappings")
        if (
            provenance.get("run") != run
            or provenance.get("run_index") != run
            or provenance.get("seed") != seed
        ):
            raise ValueError("run provenance does not match the literal seed list")
        dataset_manifest = provenance.get("dataset_manifest")
        if not isinstance(dataset_manifest, Mapping):
            raise TypeError("run provenance dataset_manifest must be a mapping")
        dataset_body = dict(dataset_manifest)
        dataset_sha = dataset_body.pop("manifest_sha256", None)
        if dataset_sha != canonical_sha256(dataset_body):
            raise ValueError("dataset manifest SHA-256 mismatch")
        method_rows = [
            records_by_identity[(run, seed, method)] for method in result.config.methods
        ]
        if {row["data_sha256"] for row in method_rows} != {provenance["data_sha256"]}:
            raise ValueError("method reports and run provenance disagree on data SHA-256")
        if {row["dataset_manifest_sha256"] for row in method_rows} != {
            provenance["dataset_manifest_sha256"]
        }:
            raise ValueError("method reports and run provenance disagree on dataset manifest")
        if provenance["data_sha256"] != dataset_manifest.get("sha256"):
            raise ValueError("run provenance data SHA-256 is not dataset-manifest bound")
        if provenance["dataset_manifest_sha256"] != dataset_sha:
            raise ValueError("run provenance records the wrong dataset-manifest digest")
        run_provenance.append(dict(provenance))

    sources = _source_snapshot()
    versions = runtime_versions()
    environment = _execution_environment()
    expected_manifest = _cohort_manifest_body(
        result.config,
        sources=sources,
        versions=versions,
        environment=environment,
        run_provenance=run_provenance,
    )
    expected_manifest["manifest_sha256"] = canonical_sha256(expected_manifest)
    if dict(result.cohort_manifest) != expected_manifest:
        raise ValueError("cohort manifest does not bind current sources/config/provenance")
    manifest_sha256 = expected_manifest["manifest_sha256"]
    if any(
        record["cohort_manifest_sha256"] != manifest_sha256
        for record in result.records
    ):
        raise ValueError("per-seed records do not bind the literal cohort manifest")

    expected_methods = tuple(summarize_method_metrics(result.records))
    if result.method_summaries != expected_methods:
        raise ValueError("method summaries do not match literal-cohort records")
    expected_paired = tuple(
        _summarize_literal_paired_differences(
            result.records,
            bootstrap_resamples=result.config.bootstrap_resamples,
            bootstrap_seed=result.config.bootstrap_seed,
        )
    )
    if result.paired_summaries != expected_paired:
        raise ValueError("paired summaries do not match literal-cohort records")
    expected_budget = tuple(summarize_corruption_budget_metrics(result.records))
    if result.corruption_budget_summaries != expected_budget:
        raise ValueError("corruption-budget summaries do not match literal-cohort records")


def _canonical_json_cell(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _csv_record(record: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(record)
    for field_name in (
        "component_permutation",
        "component_signs",
        "corruption_detection",
        "corruption_detection_at_budget",
        "extra",
    ):
        value = record[field_name]
        row[field_name] = _canonical_json_cell(value)
        if isinstance(value, Mapping):
            for key in sorted(value):
                item = value[key]
                row[f"{field_name}_{key}"] = (
                    _canonical_json_cell(item)
                    if isinstance(item, (Mapping, list, tuple))
                    else item
                )
    return row


def _csv_text(result: LiteralVideoCohortResult) -> str:
    manifest_sha256 = str(result.cohort_manifest["manifest_sha256"])
    rows: list[dict[str, Any]] = [
        {"row_type": "per_seed", **_csv_record(record)} for record in result.records
    ]
    rows.extend(
        {
            "row_type": "method_summary",
            "cohort_manifest_sha256": manifest_sha256,
            **dict(summary),
        }
        for summary in result.method_summaries
    )
    rows.extend(
        {
            "row_type": "paired_summary",
            "cohort_manifest_sha256": manifest_sha256,
            **dict(summary),
        }
        for summary in result.paired_summaries
    )
    rows.extend(
        {
            "row_type": "corruption_budget_summary",
            "cohort_manifest_sha256": manifest_sha256,
            **dict(summary),
        }
        for summary in result.corruption_budget_summaries
    )
    fieldnames = [
        "row_type",
        "seed_source",
        "cohort_manifest_sha256",
        "fixed_video_config_sha256",
        "method_report_sha256",
        "run",
        "run_index",
        "seed",
        "method",
        "metric",
    ]
    all_fields = set().union(*(row.keys() for row in rows))
    fieldnames = list(dict.fromkeys(name for name in fieldnames if name in all_fields))
    fieldnames.extend(sorted(all_fields - set(fieldnames)))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(_json_safe(rows))
    return buffer.getvalue()


def _output_completion_manifest(
    result: LiteralVideoCohortResult,
    artifacts: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    artifact_records = [
        {
            "name": name,
            "path": filename,
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }
        for name, (filename, content) in artifacts.items()
    ]
    formal_contract = _formal_default_contract(result.config)
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "literal_video_cohort_output",
        "self_excluding_manifest": True,
        "completion_contract": "manifest_written_last_after_all_declared_artifacts",
        "cohort_manifest_sha256": result.cohort_manifest["manifest_sha256"],
        "config_sha256": canonical_sha256(_config_payload(result.config)),
        "source_files_sha256": canonical_sha256(
            result.cohort_manifest["source_files"]
        ),
        "execution_environment_sha256": canonical_sha256(
            result.cohort_manifest["execution_environment"]
        ),
        "formal_default_config_exact": formal_contract["exact"],
        "artifacts": artifact_records,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def validate_literal_video_cohort_output_directory(
    output_dir: str | Path,
    *,
    source_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate completion, exact closure, hashes, sources, and config bindings.

    ``source_root=None`` validates a newly written result against the current
    checkout.  A historical sealed result must instead pass its packaged
    execution snapshot explicitly, so later maintenance does not invalidate an
    otherwise immutable artifact.
    """

    destination = Path(output_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("literal cohort output must be a real directory")
    manifest_path = destination / _OUTPUT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("literal cohort output is incomplete: completion manifest missing")
    manifest = _strict_json_load(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("literal cohort output manifest must be a mapping")
    expected_fields = {
        "schema_version",
        "artifact",
        "self_excluding_manifest",
        "completion_contract",
        "cohort_manifest_sha256",
        "config_sha256",
        "source_files_sha256",
        "execution_environment_sha256",
        "formal_default_config_exact",
        "artifacts",
        "manifest_sha256",
    }
    if set(manifest) != expected_fields:
        raise ValueError("literal cohort output manifest schema mismatch")
    if manifest["schema_version"] != 1 or isinstance(
        manifest["schema_version"], bool
    ):
        raise ValueError("unsupported literal cohort output manifest schema")
    if manifest["artifact"] != "literal_video_cohort_output":
        raise ValueError("unexpected literal cohort output artifact")
    if manifest["self_excluding_manifest"] is not True:
        raise ValueError("literal cohort output manifest must be self-excluding")
    if (
        manifest["completion_contract"]
        != "manifest_written_last_after_all_declared_artifacts"
    ):
        raise ValueError("literal cohort completion contract mismatch")
    manifest_body = dict(manifest)
    claimed_manifest_sha256 = manifest_body.pop("manifest_sha256")
    if claimed_manifest_sha256 != canonical_sha256(manifest_body):
        raise ValueError("literal cohort output manifest SHA-256 mismatch")

    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise ValueError("literal cohort output manifest must declare artifacts")
    declared_paths: set[str] = set()
    declared_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "name",
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(f"literal cohort artifact[{index}] schema mismatch")
        name = record["name"]
        relative = record["path"]
        if not isinstance(name, str) or not name or name in declared_names:
            raise ValueError("literal cohort artifact names must be unique strings")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or relative == _OUTPUT_MANIFEST_NAME
            or relative in declared_paths
        ):
            raise ValueError("literal cohort artifact paths must be unique filenames")
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise ValueError("literal cohort artifact byte counts are invalid")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("literal cohort artifact SHA-256 is invalid")
        path = destination / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"declared literal cohort artifact is missing: {relative}")
        content = path.read_bytes()
        if len(content) != record["bytes"] or _sha256_bytes(content) != digest:
            raise ValueError(f"literal cohort artifact hash/size mismatch: {relative}")
        declared_names.add(name)
        declared_paths.add(relative)

    actual_paths = {
        path.name
        for path in destination.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != declared_paths | {_OUTPUT_MANIFEST_NAME}:
        raise ValueError("literal cohort output directory file closure mismatch")
    if any(path.is_symlink() or not path.is_file() for path in destination.iterdir()):
        raise ValueError("literal cohort output contains non-regular or nested artifacts")
    if declared_names != {"json", "csv"}:
        raise ValueError("literal cohort output must contain exactly JSON and CSV")

    json_record = next(record for record in records if record["name"] == "json")
    payload = _strict_json_load(destination / json_record["path"])
    if not isinstance(payload, Mapping):
        raise TypeError("literal cohort JSON must be a mapping")
    cohort_manifest = payload.get("cohort_manifest")
    config = payload.get("config")
    if not isinstance(cohort_manifest, Mapping) or not isinstance(config, Mapping):
        raise ValueError("literal cohort JSON lacks manifest/config provenance")
    cohort_body = dict(cohort_manifest)
    cohort_sha256 = cohort_body.pop("manifest_sha256", None)
    if cohort_sha256 != canonical_sha256(cohort_body):
        raise ValueError("embedded literal cohort manifest SHA-256 mismatch")
    if cohort_sha256 != manifest["cohort_manifest_sha256"]:
        raise ValueError("literal cohort completion-to-cohort binding mismatch")
    if canonical_sha256(config) != manifest["config_sha256"]:
        raise ValueError("literal cohort output config binding mismatch")
    if (
        canonical_sha256(cohort_manifest.get("source_files"))
        != manifest["source_files_sha256"]
    ):
        raise ValueError("literal cohort output source binding mismatch")
    if (
        canonical_sha256(cohort_manifest.get("execution_environment"))
        != manifest["execution_environment_sha256"]
    ):
        raise ValueError("literal cohort output environment binding mismatch")
    if payload.get("execution_environment") != cohort_manifest.get(
        "execution_environment"
    ):
        raise ValueError("literal cohort top-level environment binding mismatch")
    if source_root is not None:
        _validate_frozen_source_snapshot(source_root)
    validating_sources = _source_snapshot(source_root)
    if list(cohort_manifest.get("source_files", [])) != [
        dict(record) for record in validating_sources
    ]:
        qualifier = (
            "packaged execution snapshot"
            if source_root is not None
            else "current source tree"
        )
        raise ValueError(
            f"literal cohort sources differ from the validating {qualifier}"
        )
    if cohort_manifest.get("source_closure") != _source_closure_contract(source_root):
        raise ValueError("literal cohort source-closure contract mismatch")
    formal = config.get("formal_default_config")
    if not isinstance(formal, Mapping) or not isinstance(formal.get("exact"), bool):
        raise ValueError("literal cohort config lacks formal-default qualification")
    if formal["exact"] is not manifest["formal_default_config_exact"]:
        raise ValueError("literal cohort formal-default qualification mismatch")
    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, Mapping) or (
        protocol.get("formal_default_config_exact") is not formal["exact"]
    ):
        raise ValueError("literal cohort protocol qualification mismatch")
    return manifest


def write_literal_video_cohort(
    result: LiteralVideoCohortResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Validate and exclusive-create strict JSON/CSV in a new output directory."""

    _validate_result(result)
    payload = _json_safe(_json_payload(result))
    json_bytes = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    artifacts: dict[str, tuple[str, bytes]] = {
        "json": ("literal_video_cohort.json", json_bytes),
        "csv": ("literal_video_cohort.csv", _csv_text(result).encode("utf-8")),
    }
    completion = _output_completion_manifest(result, artifacts)
    artifacts["manifest"] = (
        _OUTPUT_MANIFEST_NAME,
        (
            json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    paths = _claim_and_write_output_directory(output_dir, artifacts)
    validate_literal_video_cohort_output_directory(output_dir)
    return paths


def _parse_methods(value: str) -> tuple[str, ...]:
    aliases = {
        "pca": "PCA",
        "vp": "VP-SPCA",
        "vp-spca": "VP-SPCA",
        "rnd": "RND-VP-SPCA",
        "rnd-vp-spca": "RND-VP-SPCA",
        "robust": "Robust-VP-SPCA",
        "robust-vp-spca": "Robust-VP-SPCA",
    }
    selected: list[str] = []
    for token in value.split(","):
        normalized = token.strip().lower()
        if normalized not in aliases:
            raise argparse.ArgumentTypeError(f"unknown method alias: {token}")
        selected.append(aliases[normalized])
    return tuple(selected)


def _parser() -> argparse.ArgumentParser:
    defaults = _formal_video_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=defaults.grid_size)
    parser.add_argument("--n-timesteps", type=int, default=defaults.n_timesteps)
    parser.add_argument("--total-time", type=float, default=defaults.total_time)
    parser.add_argument("--n-components", type=int, default=defaults.n_components)
    parser.add_argument("--noise-std", type=float, default=defaults.noise_std)
    parser.add_argument(
        "--corruption-fraction", type=float, default=defaults.corruption_fraction
    )
    parser.add_argument("--corruption-scale", type=float, default=defaults.corruption_scale)
    parser.add_argument("--alpha", type=float, default=defaults.alpha)
    parser.add_argument("--max-iter", type=int, default=defaults.max_iter)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument(
        "--randomized-oversample", type=int, default=defaults.randomized_oversample
    )
    parser.add_argument(
        "--randomized-power-iters", type=int, default=defaults.randomized_power_iters
    )
    parser.add_argument(
        "--randomized-full-refine-iters",
        type=int,
        default=defaults.randomized_full_refine_iters,
    )
    parser.add_argument(
        "--support-threshold", type=float, default=defaults.support_threshold
    )
    parser.add_argument(
        "--outlier-support-threshold",
        type=float,
        default=defaults.outlier_support_threshold,
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=260_826)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=DEFAULT_METHODS,
        help="Comma-separated: pca,vp,rnd,robust",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/video_spca_literal_91001_91020"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    video_config = VideoExperimentConfig(
        grid_size=args.grid_size,
        n_timesteps=args.n_timesteps,
        total_time=args.total_time,
        n_components=args.n_components,
        noise_std=args.noise_std,
        corruption_fraction=args.corruption_fraction,
        corruption_scale=args.corruption_scale,
        seed=LITERAL_SEEDS[0],
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        randomized_oversample=args.randomized_oversample,
        randomized_power_iters=args.randomized_power_iters,
        randomized_full_refine_iters=args.randomized_full_refine_iters,
        support_threshold=args.support_threshold,
        outlier_support_threshold=args.outlier_support_threshold,
    )
    config = LiteralVideoCohortConfig(
        video_config=video_config,
        methods=args.methods,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    _validate_config(config)
    validate_video_output_dir(args.output_dir)
    result = run_literal_video_cohort(config, progress=True)
    paths = write_literal_video_cohort(result, args.output_dir)
    print(
        json.dumps(
            {
                "cohort_id": COHORT_ID,
                "seed_source": SEED_SOURCE,
                "seeds": list(LITERAL_SEEDS),
                "records": len(result.records),
                "cohort_manifest_sha256": result.cohort_manifest["manifest_sha256"],
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
    "COHORT_ID",
    "LITERAL_SEEDS",
    "SEED_SOURCE",
    "LiteralVideoCohortConfig",
    "LiteralVideoCohortResult",
    "main",
    "run_literal_video_cohort",
    "summarize_corruption_budget_metrics",
    "validate_literal_video_cohort_output_directory",
    "write_literal_video_cohort",
]

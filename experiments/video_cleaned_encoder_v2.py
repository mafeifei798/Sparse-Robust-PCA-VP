"""Paired synthetic-video runner for Cleaned-Encoder Robust VP-SPCA v2.

The runner has exactly two robust arms on every seed:

* the already locked fixed baseline, called through ``fit_robust_vp_spca``;
* the experimental cleaned-encoder v2 implementation.

There is no hyperparameter search, multistart, continuation, or result-based
winner selection.  Formal execution is fail-closed behind a source-bound fresh
seed protocol, a separate independent-audit receipt, and an explicit launch
acknowledgement.  Smoke output is plumbing evidence only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.video_spca_paper_reproduction import (
    PaperVideoConfig,
    RobustDecompositionDataset,
    generate_figure6_dataset,
    normalized_full_frame_errors,
    summarize_full_frame_errors,
)
from experiments.video_spca_paper_robust_study import (
    ObservedPreparedCase,
    prepare_observed_case,
)
from rspca.cleaned_encoder_v2 import (
    CleanedEncoderV2Result,
    fit_cleaned_encoder_robust_vp_spca_v2,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.metrics import outlier_detection_metrics, relative_reconstruction_error
from rspca.robust import RobustVPSPCAResult, fit_robust_vp_spca


PROTOCOL_KIND = "cleaned_encoder_robust_vp_spca_v2_fresh_seed_protocol"
AUDIT_KIND = "cleaned_encoder_robust_vp_spca_v2_independent_audit"
REPORT_NAME = "paired_report.json"
CASES_NAME = "paired_cases.csv"
PROTOCOL_COPY_NAME = "protocol_copy.json"
MANIFEST_NAME = "output_manifest.json"
REQUIRED_OUTPUTS = frozenset(
    {REPORT_NAME, CASES_NAME, PROTOCOL_COPY_NAME, MANIFEST_NAME}
)
FORMAL_ACKNOWLEDGEMENT = "RUN_LOCKED_FRESH_SEEDS"
FROZEN_LOCAL_SOURCE_CLOSURE = frozenset(
    {
        "src/rspca/__init__.py",
        "src/rspca/accelerated.py",
        "src/rspca/cleaned_encoder_v2.py",
        "src/rspca/cross_modal_gates.py",
        "src/rspca/datasets.py",
        "src/rspca/federated.py",
        "src/rspca/fedrpca_audit.py",
        "src/rspca/gauge.py",
        "src/rspca/gauge_stress.py",
        "src/rspca/metrics.py",
        "src/rspca/product_aggregation.py",
        "src/rspca/randomized.py",
        "src/rspca/robust.py",
        "src/rspca/statistical_gates.py",
        "src/rspca/trust_audit.py",
        "src/rspca/vp.py",
        "experiments/__init__.py",
        "experiments/video_cleaned_encoder_v2.py",
        "experiments/video_spca_paper_reproduction.py",
        "experiments/video_spca_paper_robust_study.py",
    }
)


@dataclass(frozen=True)
class PairedVideoV2Config:
    """Fixed paired-video configuration; no grids or alternative starts."""

    grid_size: int
    n_timesteps: int
    total_time: float
    n_components: int
    corruption_fraction: float
    corruption_scale: float
    seeds: tuple[int, ...]
    alpha: float
    kappa: float
    max_iter: int
    tol: float
    catastrophic_frame_threshold: float
    bootstrap_draws: int
    bootstrap_seed: int
    loading_nonzero_threshold: float = 1e-10
    minimum_loading_nonzero_fraction: float = 1e-4
    minimum_reconstruction_to_observed_norm: float = 1e-3


@dataclass(frozen=True)
class FixedBaselineFit:
    result: RobustVPSPCAResult
    reconstruction: np.ndarray
    outliers: np.ndarray
    runtime_seconds: float


@dataclass(frozen=True)
class CleanedEncoderFit:
    result: CleanedEncoderV2Result
    reconstruction: np.ndarray
    outliers: np.ndarray
    runtime_seconds: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("strict JSON forbids non-finite values")
    if isinstance(value, Path):
        return str(value)
    return value


def _strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_body(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in protocol.items()
        if str(key) != "protocol_body_sha256"
    }


def validate_config(config: PairedVideoV2Config) -> None:
    integers = {
        "grid_size": config.grid_size,
        "n_timesteps": config.n_timesteps,
        "n_components": config.n_components,
        "max_iter": config.max_iter,
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
    }
    for name, value in integers.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if config.n_timesteps <= config.n_components:
        raise ValueError("n_timesteps must exceed n_components")
    if config.n_components != 3:
        raise ValueError("the frozen video construction has rank 3")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a nonempty unique tuple")
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
        for seed in config.seeds
    ):
        raise ValueError("seeds must be nonnegative integers")
    finite = {
        "total_time": config.total_time,
        "corruption_fraction": config.corruption_fraction,
        "corruption_scale": config.corruption_scale,
        "alpha": config.alpha,
        "kappa": config.kappa,
        "tol": config.tol,
        "catastrophic_frame_threshold": config.catastrophic_frame_threshold,
        "loading_nonzero_threshold": config.loading_nonzero_threshold,
        "minimum_loading_nonzero_fraction": config.minimum_loading_nonzero_fraction,
        "minimum_reconstruction_to_observed_norm": (
            config.minimum_reconstruction_to_observed_norm
        ),
    }
    for name, value in finite.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.total_time <= 0.0 or config.kappa <= 0.0 or config.tol <= 0.0:
        raise ValueError("total_time, kappa, and tol must be positive")
    if config.alpha < 0.0 or config.corruption_scale < 0.0:
        raise ValueError("alpha and corruption_scale must be nonnegative")
    if not 0.0 <= config.corruption_fraction < 1.0:
        raise ValueError("corruption_fraction must lie in [0,1)")
    if not 0.0 <= config.minimum_loading_nonzero_fraction <= 1.0:
        raise ValueError("minimum_loading_nonzero_fraction must lie in [0,1]")
    if (
        min(
            config.catastrophic_frame_threshold,
            config.loading_nonzero_threshold,
            config.minimum_reconstruction_to_observed_norm,
        )
        < 0.0
    ):
        raise ValueError("diagnostic thresholds must be nonnegative")


def _config_from_mapping(value: Mapping[str, Any]) -> PairedVideoV2Config:
    normalized = dict(value)
    normalized["seeds"] = tuple(int(seed) for seed in normalized["seeds"])
    config = PairedVideoV2Config(**normalized)
    validate_config(config)
    return config


def smoke_config() -> PairedVideoV2Config:
    return PairedVideoV2Config(
        grid_size=12,
        n_timesteps=24,
        total_time=12.0,
        n_components=3,
        corruption_fraction=0.05,
        corruption_scale=3.0,
        seeds=(123,),
        alpha=0.0015,
        kappa=0.001,
        max_iter=20,
        tol=1e-6,
        catastrophic_frame_threshold=2.0,
        bootstrap_draws=100,
        bootstrap_seed=73001,
    )


def _paper_config(config: PairedVideoV2Config, seed: int) -> PaperVideoConfig:
    return PaperVideoConfig(
        grid_size=config.grid_size,
        n_timesteps=config.n_timesteps,
        total_time=config.total_time,
        n_components=config.n_components,
        robust_alpha=config.alpha,
        robust_kappa=config.kappa,
        corruption_fraction=config.corruption_fraction,
        corruption_scale=config.corruption_scale,
        corruption_seed=int(seed),
        max_iter=config.max_iter,
        tol=config.tol,
        figure6_frame_index=min(10, config.n_timesteps - 1),
        dpi=72,
    )


def fit_fixed_baseline_observed(
    observed: ObservedPreparedCase,
    config: PairedVideoV2Config,
) -> FixedBaselineFit:
    """Call the old locked fixed baseline with its exact formal arguments."""

    validate_config(config)
    started = perf_counter()
    result = fit_robust_vp_spca(
        observed.normalized_observed,
        config.n_components,
        alpha=config.alpha,
        kappa=config.kappa,
        max_iter=config.max_iter,
        tol=config.tol,
        step_size=1.0,
        initial_B=observed.pca_B,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - started
    reconstruction = observed.normalization_sigma_max * (
        (observed.normalized_observed @ result.B) @ result.A.T
    )
    outliers = observed.normalization_sigma_max * result.S
    return FixedBaselineFit(
        result=result,
        reconstruction=np.asarray(reconstruction, dtype=np.float64),
        outliers=np.asarray(outliers, dtype=np.float64),
        runtime_seconds=float(runtime),
    )


def fit_cleaned_encoder_v2_observed(
    observed: ObservedPreparedCase,
    config: PairedVideoV2Config,
) -> CleanedEncoderFit:
    validate_config(config)
    started = perf_counter()
    result = fit_cleaned_encoder_robust_vp_spca_v2(
        observed.normalized_observed,
        config.n_components,
        alpha=config.alpha,
        kappa=config.kappa,
        max_iter=config.max_iter,
        tol=config.tol,
        B_step_size=1.0,
        initial_B=observed.pca_B,
        initial_S=None,
        verbose=False,
    )
    runtime = perf_counter() - started
    return CleanedEncoderFit(
        result=result,
        reconstruction=np.asarray(
            observed.normalization_sigma_max * result.reconstruction,
            dtype=np.float64,
        ),
        outliers=np.asarray(
            observed.normalization_sigma_max * result.S,
            dtype=np.float64,
        ),
        runtime_seconds=float(runtime),
    )


def _objective_monotone(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    array = np.asarray(values, dtype=np.float64)
    tolerance = 1e-10 * np.maximum(1.0, np.abs(array[:-1]))
    return bool(np.all(np.diff(array) <= tolerance))


def _nontrivial_gate(
    observed: ObservedPreparedCase,
    reconstruction: np.ndarray,
    B: np.ndarray,
    config: PairedVideoV2Config,
) -> dict[str, Any]:
    loading_nonzero_fraction = float(
        np.mean(np.abs(B) > config.loading_nonzero_threshold)
    )
    reconstruction_ratio = float(
        np.linalg.norm(reconstruction, ord="fro")
        / max(
            float(np.linalg.norm(observed.observed, ord="fro")),
            np.finfo(np.float64).eps,
        )
    )
    loading_rank = int(np.linalg.matrix_rank(B))
    qualified = bool(
        loading_nonzero_fraction >= config.minimum_loading_nonzero_fraction
        and reconstruction_ratio >= config.minimum_reconstruction_to_observed_norm
        and loading_rank == config.n_components
    )
    return {
        "qualified": qualified,
        "selection_filter": False,
        "failure_policy": "retain paired result and fail diagnostic",
        "loading_nonzero_threshold": config.loading_nonzero_threshold,
        "loading_nonzero_fraction": loading_nonzero_fraction,
        "minimum_loading_nonzero_fraction": config.minimum_loading_nonzero_fraction,
        "reconstruction_to_observed_frobenius_norm": reconstruction_ratio,
        "minimum_reconstruction_to_observed_norm": (
            config.minimum_reconstruction_to_observed_norm
        ),
        "loading_rank": loading_rank,
        "required_loading_rank": config.n_components,
    }


def _truth_metrics(
    dataset: RobustDecompositionDataset,
    reconstruction: np.ndarray,
    outliers: np.ndarray,
    config: PairedVideoV2Config,
) -> dict[str, Any]:
    frame_errors = normalized_full_frame_errors(dataset.clean, reconstruction)
    frame_summary = summarize_full_frame_errors(frame_errors, dataset.time)
    nominal_budget = int(round(config.corruption_fraction * dataset.observed.size))
    detection = outlier_detection_metrics(
        dataset.corruption_support,
        scores=np.abs(outliers),
        top_k=nominal_budget,
    )
    return {
        "clean_relative_reconstruction_error": relative_reconstruction_error(
            dataset.clean,
            reconstruction,
        ),
        "full_frame_normalized_rmse": frame_summary,
        "catastrophic_frame_threshold": config.catastrophic_frame_threshold,
        "catastrophic_frame_count": int(
            np.count_nonzero(frame_errors > config.catastrophic_frame_threshold)
        ),
        "outlier_detection": {
            **detection,
            "outlier_nominal_budget_f1": float(detection["f1"]),
            "budget_top_k": nominal_budget,
            "budget_source": "round(protocol corruption_fraction * observed.size)",
            "budget_semantics": "nominal protocol budget, not realized support size",
            "realized_truth_fraction": float(np.mean(dataset.corruption_support)),
        },
    }


def run_paired_seed(config: PairedVideoV2Config, seed: int) -> dict[str, Any]:
    """Fit both observed-only arms, freeze hashes, then consult truth."""

    validate_config(config)
    if int(seed) not in config.seeds:
        raise ValueError("seed is not in the declared paired bank")
    dataset = generate_figure6_dataset(_paper_config(config, int(seed)))
    observed = prepare_observed_case(dataset)
    baseline = fit_fixed_baseline_observed(observed, config)
    v2 = fit_cleaned_encoder_v2_observed(observed, config)

    frozen_hashes = {
        "fixed_baseline": {
            "A": array_sha256(baseline.result.A),
            "B": array_sha256(baseline.result.B),
            "S_normalized": array_sha256(baseline.result.S),
            "reconstruction_raw": array_sha256(baseline.reconstruction),
            "outliers_raw": array_sha256(baseline.outliers),
        },
        "cleaned_encoder_v2": {
            "A": array_sha256(v2.result.A),
            "B": array_sha256(v2.result.B),
            "S_normalized": array_sha256(v2.result.S),
            "reconstruction_raw": array_sha256(v2.reconstruction),
            "outliers_raw": array_sha256(v2.outliers),
        },
    }
    baseline_metrics = _truth_metrics(
        dataset,
        baseline.reconstruction,
        baseline.outliers,
        config,
    )
    v2_metrics = _truth_metrics(
        dataset,
        v2.reconstruction,
        v2.outliers,
        config,
    )
    final_v2_blocks = (
        asdict(v2.result.block_stationarity[-1])
        if v2.result.block_stationarity
        else None
    )
    return _json_safe(
        {
            "seed": int(seed),
            "data": {
                "orientation": "spatial_pixels_x_time_snapshots",
                "observed_sha256": observed.data_sha256,
                "clean_sha256": array_sha256(dataset.clean),
                "corruption_sha256": array_sha256(dataset.corruption),
                "corruption_support_sha256": array_sha256(dataset.corruption_support),
                "normalization_sigma_max": observed.normalization_sigma_max,
                "normalization_policy": "divide observed X by largest singular value",
                "pca_initial_B_sha256": observed.pca_B_sha256,
                "preparation_runtime_seconds": observed.preparation_runtime_seconds,
            },
            "locked_parameters": {
                "n_components": config.n_components,
                "alpha": config.alpha,
                "kappa": config.kappa,
                "max_iter": config.max_iter,
                "tol": config.tol,
            },
            "fit_hashes_frozen_before_truth_metrics": frozen_hashes,
            "methods": {
                "fixed_baseline": {
                    "method_id": "fixed_baseline",
                    "objective": "0.5||X-XBA^T-S||_F^2+alpha||B||_1+kappa||S||_1",
                    "encoder": "observed X",
                    "reconstruction": "XBA^T",
                    "solver": {
                        "implementation": "rspca.robust.fit_robust_vp_spca",
                        "exact_formal_call_contract": {
                            "step_size": 1.0,
                            "initial_B": "observed PCA right singular vectors",
                            "backtracking": True,
                            "fixed_alpha_kappa": True,
                        },
                        "converged": baseline.result.converged,
                        "status": baseline.result.status,
                        "n_iter": baseline.result.n_iter,
                        "final_stationarity": (
                            baseline.result.stationarity[-1]
                            if baseline.result.stationarity
                            else None
                        ),
                        "final_objective": baseline.result.objectives[-1],
                        "objective_trace_monotone": _objective_monotone(
                            baseline.result.objectives
                        ),
                        "runtime_seconds": baseline.runtime_seconds,
                    },
                    "nontrivial_solution_gate": _nontrivial_gate(
                        observed,
                        baseline.reconstruction,
                        baseline.result.B,
                        config,
                    ),
                    "truth_metrics_after_fit_hash_freeze": baseline_metrics,
                },
                "cleaned_encoder_v2": {
                    "method_id": "cleaned_encoder_v2",
                    "objective": (
                        "0.5||(X-S)-(X-S)BA^T||_F^2+alpha||B||_1+kappa||S||_1"
                    ),
                    "encoder": "cleaned X-S",
                    "reconstruction": "(X-S)BA^T",
                    "solver": {
                        "implementation": (
                            "rspca.cleaned_encoder_v2."
                            "fit_cleaned_encoder_robust_vp_spca_v2"
                        ),
                        "initial_B": "same observed PCA right singular vectors",
                        "initial_S": "all zeros",
                        "converged": v2.result.converged,
                        "status": v2.result.status,
                        "n_iter": v2.result.n_iter,
                        "final_stationarity": (
                            v2.result.stationarity[-1]
                            if v2.result.stationarity
                            else None
                        ),
                        "final_block_stationarity": final_v2_blocks,
                        "final_objective": v2.result.objectives[-1],
                        "objective_trace_monotone": _objective_monotone(
                            v2.result.objectives
                        ),
                        "runtime_seconds": v2.runtime_seconds,
                    },
                    "nontrivial_solution_gate": _nontrivial_gate(
                        observed,
                        v2.reconstruction,
                        v2.result.B,
                        config,
                    ),
                    "truth_metrics_after_fit_hash_freeze": v2_metrics,
                },
            },
            "truth_isolation": {
                "fit_function_input": "ObservedPreparedCase only",
                "clean_truth_used_for_fit_or_selection": False,
                "support_used_for_fit_or_selection": False,
                "within_method_candidate_selection": False,
                "paired_output_hashes_frozen_before_truth_metrics": True,
            },
        }
    )


def _method_metric(case: Mapping[str, Any], method: str, metric: str) -> float:
    truth = case["methods"][method]["truth_metrics_after_fit_hash_freeze"]
    if metric == "clean_relative_reconstruction_error":
        return float(truth[metric])
    if metric in {"frame_median", "frame_p95", "frame_max"}:
        field = metric.removeprefix("frame_")
        return float(truth["full_frame_normalized_rmse"][field])
    if metric == "outlier_nominal_budget_f1":
        return float(truth["outlier_detection"][metric])
    raise ValueError(f"unknown metric: {metric}")


def _bootstrap_mean_interval(
    differences: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(draws, differences.size))
    means = np.mean(differences[indices], axis=1)
    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def _paired_difference_summary(
    cases: Sequence[Mapping[str, Any]],
    metric: str,
    config: PairedVideoV2Config,
) -> dict[str, Any]:
    differences = np.asarray(
        [
            _method_metric(case, "cleaned_encoder_v2", metric)
            - _method_metric(case, "fixed_baseline", metric)
            for case in cases
        ],
        dtype=np.float64,
    )
    lower, upper = _bootstrap_mean_interval(
        differences,
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed,
    )
    return {
        "difference_definition": "cleaned_encoder_v2 minus fixed_baseline",
        "values_in_locked_seed_order": differences.tolist(),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "strict_negative_differences": int(np.count_nonzero(differences < 0.0)),
        "strict_positive_differences": int(np.count_nonzero(differences > 0.0)),
        "ties": int(np.count_nonzero(differences == 0.0)),
        "paired_bootstrap_mean_difference_95_interval": [lower, upper],
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
    }


def _method_summary(
    cases: Sequence[Mapping[str, Any]],
    method: str,
) -> dict[str, Any]:
    metrics = (
        "clean_relative_reconstruction_error",
        "frame_median",
        "frame_p95",
        "frame_max",
        "outlier_nominal_budget_f1",
    )
    values = {
        metric: np.asarray(
            [_method_metric(case, method, metric) for case in cases],
            dtype=np.float64,
        )
        for metric in metrics
    }
    runtimes = np.asarray(
        [case["methods"][method]["solver"]["runtime_seconds"] for case in cases],
        dtype=np.float64,
    )
    stationarity = np.asarray(
        [case["methods"][method]["solver"]["final_stationarity"] for case in cases],
        dtype=np.float64,
    )
    return {
        "case_count": len(cases),
        "metrics": {
            metric: {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "values_in_locked_seed_order": array.tolist(),
            }
            for metric, array in values.items()
        },
        "converged_count": int(
            sum(bool(case["methods"][method]["solver"]["converged"]) for case in cases)
        ),
        "nontrivial_qualified_count": int(
            sum(
                bool(case["methods"][method]["nontrivial_solution_gate"]["qualified"])
                for case in cases
            )
        ),
        "final_stationarity": {
            "mean": float(np.mean(stationarity)),
            "max": float(np.max(stationarity)),
            "values_in_locked_seed_order": stationarity.tolist(),
        },
        "runtime_seconds": {
            "mean": float(np.mean(runtimes)),
            "total": float(np.sum(runtimes)),
            "values_in_locked_seed_order": runtimes.tolist(),
            "scope": "solver call only; excludes common SVD preparation and serialization",
        },
    }


def summarize_cases(
    cases: Sequence[Mapping[str, Any]],
    config: PairedVideoV2Config,
    *,
    run_kind: str,
) -> dict[str, Any]:
    if [int(case["seed"]) for case in cases] != list(config.seeds):
        raise ValueError("cases are not in the locked seed order")
    comparisons = {
        metric: _paired_difference_summary(cases, metric, config)
        for metric in (
            "outlier_nominal_budget_f1",
            "clean_relative_reconstruction_error",
            "frame_median",
            "frame_p95",
            "frame_max",
        )
    }
    if run_kind == "formal":
        if len(cases) != 20:
            raise ValueError("formal success gate requires exactly 20 cases")
        f1 = comparisons["outlier_nominal_budget_f1"]
        clean = comparisons["clean_relative_reconstruction_error"]
        maximum = comparisons["frame_max"]
        f1_lower = f1["paired_bootstrap_mean_difference_95_interval"][0]
        clean_upper = clean["paired_bootstrap_mean_difference_95_interval"][1]
        maximum_upper = maximum["paired_bootstrap_mean_difference_95_interval"][1]
        clauses = {
            "f1_mean_strictly_positive": f1["mean_difference"] > 0.0,
            "f1_median_strictly_positive": f1["median_difference"] > 0.0,
            "f1_at_least_16_of_20_strict_improvement_wins": (
                f1["strict_positive_differences"] >= 16
            ),
            "f1_paired_bootstrap_lower_strictly_positive": f1_lower > 0.0,
            "clean_zero_margin_mean_noninferiority": clean["mean_difference"] <= 0.0,
            "clean_zero_margin_median_noninferiority": (
                clean["median_difference"] <= 0.0
            ),
            "clean_zero_margin_bootstrap_upper_noninferiority": clean_upper <= 0.0,
            "max_frame_zero_margin_mean_noninferiority": (
                maximum["mean_difference"] <= 0.0
            ),
            "max_frame_zero_margin_median_noninferiority": (
                maximum["median_difference"] <= 0.0
            ),
            "max_frame_zero_margin_bootstrap_upper_noninferiority": (
                maximum_upper <= 0.0
            ),
        }
        gate = {
            "evaluated": True,
            "passed": bool(all(clauses.values())),
            "clauses": clauses,
            "failure_policy": (
                "retain as negative result and stop this v2 target; no retuning, "
                "seed replacement, winner selection, or target change"
            ),
        }
        claim_status = (
            "predeclared paired success gate passed"
            if gate["passed"]
            else "retained negative result; predeclared paired success gate failed"
        )
    else:
        gate = {
            "evaluated": False,
            "passed": None,
            "reason": "smoke dimensions, seed bank, and budget are non-scientific",
        }
        claim_status = "SMOKE_PLUMBING_ONLY_NO_SCIENTIFIC_CLAIM"
    return {
        "run_kind": run_kind,
        "claim_status": claim_status,
        "method_summaries": {
            method: _method_summary(cases, method)
            for method in ("fixed_baseline", "cleaned_encoder_v2")
        },
        "paired_comparisons": comparisons,
        "predeclared_success_gate": gate,
    }


def _run_seed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _config_from_mapping(payload["config"])
    return run_paired_seed(config, int(payload["seed"]))


def run_cases(
    config: PairedVideoV2Config,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    validate_config(config)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    payloads = [{"config": asdict(config), "seed": int(seed)} for seed in config.seeds]
    if workers == 1:
        return [_run_seed_payload(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        cases = list(executor.map(_run_seed_payload, payloads))
    if [case["seed"] for case in cases] != list(config.seeds):
        raise RuntimeError("parallel execution changed the locked seed order")
    return cases


def _case_csv_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        for method in ("fixed_baseline", "cleaned_encoder_v2"):
            record = case["methods"][method]
            truth = record["truth_metrics_after_fit_hash_freeze"]
            frame = truth["full_frame_normalized_rmse"]
            solver = record["solver"]
            rows.append(
                {
                    "seed": int(case["seed"]),
                    "method": method,
                    "clean_relative_reconstruction_error": truth[
                        "clean_relative_reconstruction_error"
                    ],
                    "frame_median_nrmse": frame["median"],
                    "frame_p95_nrmse": frame["p95"],
                    "frame_max_nrmse": frame["max"],
                    "outlier_nominal_budget_f1": truth["outlier_detection"][
                        "outlier_nominal_budget_f1"
                    ],
                    "converged": solver["converged"],
                    "final_stationarity": solver["final_stationarity"],
                    "nontrivial_qualified": record["nontrivial_solution_gate"][
                        "qualified"
                    ],
                    "runtime_seconds": solver["runtime_seconds"],
                }
            )
    return rows


def write_closed_output(
    output_dir: Path,
    *,
    config: PairedVideoV2Config,
    cases: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    run_kind: str,
    protocol_copy: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "study_kind": "cleaned_encoder_robust_vp_spca_v2_paired_synthetic_video",
        "run_kind": run_kind,
        "config": asdict(config),
        "summary": summary,
        "cases": list(cases),
        "claim_boundary": (
            "synthetic-only experimental evaluation; no real-video, new-algorithm, "
            "runtime-speedup, or generalization claim"
        ),
        "runtime_versions": runtime_versions(),
    }
    (output_dir / REPORT_NAME).write_bytes(_strict_json_bytes(report))
    (output_dir / PROTOCOL_COPY_NAME).write_bytes(_strict_json_bytes(protocol_copy))
    rows = _case_csv_rows(cases)
    fieldnames = list(rows[0]) if rows else []
    with (output_dir / CASES_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload_names = (REPORT_NAME, CASES_NAME, PROTOCOL_COPY_NAME)
    manifest = {
        "schema_version": 1,
        "manifest_kind": "self_excluding_sha256_output_manifest",
        "required_output_names": sorted(REQUIRED_OUTPUTS),
        "files": {
            name: {
                "sha256": _file_sha256(output_dir / name),
                "size_bytes": (output_dir / name).stat().st_size,
            }
            for name in payload_names
        },
    }
    manifest["manifest_body_sha256"] = canonical_sha256(manifest)
    (output_dir / MANIFEST_NAME).write_bytes(_strict_json_bytes(manifest))
    validate_closed_output(output_dir)
    return manifest


def validate_closed_output(output_dir: Path) -> dict[str, Any]:
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError("output must be a real directory")
    entries = {path.name: path for path in output_dir.iterdir()}
    if set(entries) != set(REQUIRED_OUTPUTS):
        raise ValueError("output has missing or extra entries")
    for path in entries.values():
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISREG(mode):
            raise ValueError("output entries must be regular non-symlink files")
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    body_hash = manifest.pop("manifest_body_sha256")
    if canonical_sha256(manifest) != body_hash:
        raise ValueError("manifest body digest mismatch")
    for name, expected in manifest["files"].items():
        path = output_dir / name
        if _file_sha256(path) != expected["sha256"]:
            raise ValueError(f"payload digest mismatch: {name}")
        if path.stat().st_size != int(expected["size_bytes"]):
            raise ValueError(f"payload size mismatch: {name}")
    return {**manifest, "manifest_body_sha256": body_hash}


def preflight_new_output_directory(output_dir: Path) -> None:
    """Fail before expensive work when an output path is already occupied."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing occupied formal output: {output_dir}")


def audit_closed_output(output_dir: Path) -> dict[str, Any]:
    """Recompute smoke/formal aggregates and the CSV from retained cases."""

    manifest = validate_closed_output(output_dir)
    report = json.loads((output_dir / REPORT_NAME).read_text(encoding="utf-8"))
    config = _config_from_mapping(report["config"])
    run_kind = str(report["run_kind"])
    if run_kind not in {"smoke", "formal"}:
        raise ValueError("report has an unknown run kind")
    cases = report["cases"]
    recomputed_summary = summarize_cases(cases, config, run_kind=run_kind)
    if canonical_sha256(recomputed_summary) != canonical_sha256(report["summary"]):
        raise ValueError("retained cases do not reproduce the reported summary")

    expected_rows = [
        {str(key): str(value) for key, value in row.items()}
        for row in _case_csv_rows(cases)
    ]
    with (output_dir / CASES_NAME).open(newline="", encoding="utf-8") as handle:
        actual_rows = list(csv.DictReader(handle))
    if actual_rows != expected_rows:
        raise ValueError("retained cases do not reproduce the CSV")

    protocol_copy = json.loads(
        (output_dir / PROTOCOL_COPY_NAME).read_text(encoding="utf-8")
    )
    if run_kind == "smoke":
        if protocol_copy.get("formal_protocol_used") is not False:
            raise ValueError("smoke output incorrectly claims a formal protocol")
        if protocol_copy.get("formal_seeds_used") is not False:
            raise ValueError("smoke output incorrectly claims formal seeds")
    else:
        if protocol_copy.get("protocol_kind") != PROTOCOL_KIND:
            raise ValueError("formal output lacks the frozen protocol copy")
        if protocol_copy.get("protocol_body_sha256") != canonical_sha256(
            _protocol_body(protocol_copy)
        ):
            raise ValueError("formal protocol copy digest mismatch")
    audit = {
        "audit_kind": "cleaned_encoder_v2_closed_output_semantic_audit",
        "run_kind": run_kind,
        "seed_count": len(config.seeds),
        "summary_recomputed": True,
        "csv_recomputed": True,
        "manifest_body_sha256": manifest["manifest_body_sha256"],
        "claim_status": recomputed_summary["claim_status"],
    }
    audit["audit_body_sha256"] = canonical_sha256(audit)
    return audit


def load_and_validate_protocol(
    path: Path,
) -> tuple[dict[str, Any], PairedVideoV2Config]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("unexpected protocol kind")
    reported_body_hash = str(protocol.get("protocol_body_sha256", ""))
    recomputed_body_hash = canonical_sha256(_protocol_body(protocol))
    if reported_body_hash != recomputed_body_hash:
        raise ValueError("protocol body digest mismatch")
    if protocol.get("formal_launch_state") != "FROZEN_NOT_LAUNCHED":
        raise ValueError("protocol is not in FROZEN_NOT_LAUNCHED state")
    if protocol.get("formal_launch_authorized_by_this_file") is not False:
        raise ValueError("protocol must not self-authorize formal launch")
    config = _config_from_mapping(protocol["config"])
    if (config.n_components, config.alpha, config.kappa) != (3, 0.0015, 0.001):
        raise ValueError("locked baseline rank/alpha/kappa changed")
    if len(config.seeds) != 20:
        raise ValueError("fresh formal bank must contain exactly 20 seeds")
    excluded = {int(seed) for seed in protocol["fresh_seed_contract"]["excluded_seeds"]}
    if excluded.intersection(config.seeds):
        raise ValueError("fresh seed bank overlaps excluded/viewed seeds")
    repo_root = Path(__file__).resolve().parents[1]
    expected_sources = protocol["source_sha256"]
    if not expected_sources:
        raise ValueError("source closure must not be empty")
    if set(expected_sources) != set(FROZEN_LOCAL_SOURCE_CLOSURE):
        raise ValueError("source closure is not the complete frozen local import set")
    for relative, expected in expected_sources.items():
        source = repo_root / relative
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"source closure entry is unavailable: {relative}")
        if _file_sha256(source) != expected:
            raise ValueError(f"source closure digest mismatch: {relative}")
    scan_contract = protocol["fresh_seed_contract"]
    scan_path = repo_root / scan_contract["seed_history_scan_path"]
    if not scan_path.is_file() or scan_path.is_symlink():
        raise ValueError("seed-history scan record is unavailable")
    if _file_sha256(scan_path) != scan_contract["seed_history_scan_sha256"]:
        raise ValueError("seed-history scan record digest mismatch")
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    if tuple(int(seed) for seed in scan["candidate_bank"]) != config.seeds:
        raise ValueError("seed-history scan is bound to a different bank")
    if int(scan["semantic_seed_hits_before_freeze"]) != 0:
        raise ValueError("seed-history scan did not establish zero prior hits")
    if int(scan["candidate_outcomes_generated_before_freeze"]) != 0:
        raise ValueError("seed-history scan followed candidate outcomes")
    if int(scan["formal_v2_launches_before_scan"]) != 0:
        raise ValueError("formal v2 execution preceded the seed-history scan")
    anchor = protocol["historical_fixed_baseline_anchor"]
    anchor_root = repo_root / anchor["source_artifact_root"]
    anchor_files = {
        "formal_development_selection_lock_file_sha256": (
            anchor_root / "formal_dev_v1/selection_lock.json"
        ),
        "formal_confirmation_manifest_file_sha256": (
            anchor_root / "formal_confirm_v1/confirmation_output_manifest.json"
        ),
    }
    for digest_field, anchor_path in anchor_files.items():
        if not anchor_path.is_file() or anchor_path.is_symlink():
            raise ValueError(f"historical anchor is unavailable: {anchor_path}")
        if _file_sha256(anchor_path) != anchor[digest_field]:
            raise ValueError(f"historical anchor digest mismatch: {anchor_path}")
    return protocol, config


def validate_audit_receipt(
    receipt_path: Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("audit_kind") != AUDIT_KIND or receipt.get("status") != "passed":
        raise ValueError("independent audit receipt is not a pass")
    if receipt.get("auditor_independent") is not True:
        raise ValueError("audit receipt does not attest independent review")
    if receipt.get("protocol_body_sha256") != protocol["protocol_body_sha256"]:
        raise ValueError("audit receipt is bound to a different protocol")
    if receipt.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("audit receipt is bound to a different source closure")
    return receipt


def _smoke_protocol_copy(config: PairedVideoV2Config) -> dict[str, Any]:
    return {
        "protocol_kind": "SMOKE_ONLY_NOT_FORMAL_PROTOCOL",
        "formal_protocol_used": False,
        "formal_seeds_used": False,
        "scientific_gate_evaluated": False,
        "config": asdict(config),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    validate_protocol = subparsers.add_parser("validate-protocol")
    validate_protocol.add_argument("protocol", type=Path)
    validate_output = subparsers.add_parser("validate-output")
    validate_output.add_argument("output_dir", type=Path)
    formal = subparsers.add_parser("formal")
    formal.add_argument("--protocol", type=Path, required=True)
    formal.add_argument("--audit-receipt", type=Path, required=True)
    formal.add_argument("--output-dir", type=Path, required=True)
    formal.add_argument("--workers", type=int, default=1)
    formal.add_argument("--acknowledge-formal-launch", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "validate-protocol":
        protocol, config = load_and_validate_protocol(args.protocol)
        print(
            json.dumps(
                {
                    "status": "VALID_FROZEN_NOT_LAUNCHED",
                    "protocol_body_sha256": protocol["protocol_body_sha256"],
                    "seed_count": len(config.seeds),
                    "formal_launches": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "validate-output":
        audit = audit_closed_output(args.output_dir)
        print(json.dumps({"status": "VALID_SEMANTIC", **audit}, sort_keys=True))
        return 0
    if args.command == "smoke":
        config = smoke_config()
        cases = run_cases(config, workers=1)
        summary = summarize_cases(cases, config, run_kind="smoke")
        manifest = write_closed_output(
            args.output_dir,
            config=config,
            cases=cases,
            summary=summary,
            run_kind="smoke",
            protocol_copy=_smoke_protocol_copy(config),
        )
        print(
            json.dumps(
                {
                    "status": "SMOKE_PASS_NO_SCIENTIFIC_CLAIM",
                    "output_dir": str(args.output_dir),
                    "manifest_body_sha256": manifest["manifest_body_sha256"],
                    "formal_launches": 0,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.acknowledge_formal_launch != FORMAL_ACKNOWLEDGEMENT:
        raise ValueError("formal launch acknowledgement is missing or incorrect")
    protocol, config = load_and_validate_protocol(args.protocol)
    validate_audit_receipt(args.audit_receipt, protocol)
    repo_root = Path(__file__).resolve().parents[1]
    expected_output = (repo_root / protocol["formal_output_dir"]).resolve()
    if args.output_dir.resolve() != expected_output:
        raise ValueError("formal output directory differs from the frozen protocol")
    preflight_new_output_directory(args.output_dir)
    cases = run_cases(config, workers=args.workers)
    summary = summarize_cases(cases, config, run_kind="formal")
    manifest = write_closed_output(
        args.output_dir,
        config=config,
        cases=cases,
        summary=summary,
        run_kind="formal",
        protocol_copy=protocol,
    )
    print(
        json.dumps(
            {
                "status": summary["claim_status"],
                "output_dir": str(args.output_dir),
                "manifest_body_sha256": manifest["manifest_body_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

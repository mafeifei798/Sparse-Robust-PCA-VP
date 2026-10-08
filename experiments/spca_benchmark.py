#!/usr/bin/env python3
"""Paired, multi-seed benchmark for PCA and the canonical RSPCA solvers.

No artifact is written unless ``--output-json`` and/or ``--output-csv`` is
provided.  Hyperparameters can either be retained as a full alpha grid or
selected by distance to a pre-declared loading sparsity target.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import rspca
import scipy
from scipy.stats import t as student_t

from rspca.datasets import (
    array_sha256,
    canonical_sha256,
    generate_sparse_low_rank,
    make_seed_manifest,
)
from rspca.metrics import (
    component_correlation,
    explained_variance_ratio,
    outlier_detection_metrics,
    relative_reconstruction_error,
    sparsity_ratio,
    subspace_error,
    support_precision_recall_f1,
)


@dataclass(frozen=True)
class BenchmarkConfig:
    n_samples: int = 240
    n_features: int = 120
    rank: int = 3
    support_fraction: float = 0.15
    noise_std: float = 0.05
    corruption_fraction: float = 0.05
    corruption_scale: float = 8.0
    corruption_distribution: str = "gaussian"
    alphas: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
    selection: str = "alpha-grid"
    target_sparsity: float | None = None
    repeats: int = 3
    base_seed: int = 2025
    max_iter: int = 300
    tol: float = 1e-6
    oversample: int = 10
    n_power_iter: int = 2
    randomized_full_gradient: bool = False
    randomized_finalize_full_iters: int = 5
    certified_residual_tolerance: float = 0.05
    certified_full_refine_iters: int = 100


@dataclass
class _Candidate:
    method: str
    alpha: float | None
    A: NDArray[np.float64]
    B: NDArray[np.float64]
    S: NDArray[np.float64] | None
    runtime_seconds: float
    converged: bool
    n_iter: int
    objective: float | None
    stationarity: float | None
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "rspca": str(rspca.__version__),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
    }


def _validate_config(config: BenchmarkConfig) -> None:
    if config.selection not in {"alpha-grid", "matched-sparsity"}:
        raise ValueError("selection must be alpha-grid or matched-sparsity")
    positive_integers = {
        "n_samples": config.n_samples,
        "n_features": config.n_features,
        "rank": config.rank,
        "repeats": config.repeats,
        "max_iter": config.max_iter,
        "certified_full_refine_iters": config.certified_full_refine_iters,
    }
    nonnegative_integers = {
        "base_seed": config.base_seed,
        "oversample": config.oversample,
        "n_power_iter": config.n_power_iter,
        "randomized_finalize_full_iters": config.randomized_finalize_full_iters,
    }
    for name, value in {**positive_integers, **nonnegative_integers}.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if any(value <= 0 for value in positive_integers.values()):
        raise ValueError("sample, feature, rank, repeat, and iteration counts must be positive")
    if any(value < 0 for value in nonnegative_integers.values()):
        raise ValueError("seed, sketch, and optional refinement counts must be non-negative")
    if config.rank >= config.n_samples or config.rank > config.n_features:
        raise ValueError(
            "rank must be smaller than n_samples and no larger than n_features"
        )
    finite_scalars = {
        "support_fraction": config.support_fraction,
        "noise_std": config.noise_std,
        "corruption_fraction": config.corruption_fraction,
        "corruption_scale": config.corruption_scale,
        "tol": config.tol,
        "certified_residual_tolerance": config.certified_residual_tolerance,
    }
    for name, value in finite_scalars.items():
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            finite = False
        if isinstance(value, bool) or not finite:
            raise ValueError(f"{name} must be a finite real scalar")
    if not 0.0 < config.support_fraction <= 1.0:
        raise ValueError("support_fraction must lie in (0, 1]")
    support_size = max(1, int(round(config.support_fraction * config.n_features)))
    if config.rank * support_size > config.n_features:
        raise ValueError("disjoint component supports do not fit in n_features")
    if config.noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= config.corruption_fraction <= 1.0:
        raise ValueError("corruption_fraction must lie in [0, 1]")
    if config.corruption_scale < 0.0 or (
        config.corruption_fraction > 0.0 and config.corruption_scale == 0.0
    ):
        raise ValueError(
            "corruption_scale must be positive when corruption_fraction is positive"
        )
    try:
        alphas = tuple(float(alpha) for alpha in config.alphas)
    except (TypeError, ValueError) as error:
        raise ValueError("alphas must contain finite non-negative values") from error
    if (
        not alphas
        or any(isinstance(alpha, bool) for alpha in config.alphas)
        or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in alphas)
    ):
        raise ValueError("alphas must contain finite non-negative values")
    if len(set(alphas)) != len(alphas):
        raise ValueError("alphas must not contain duplicates")
    if config.target_sparsity is not None and (
        not np.isfinite(config.target_sparsity)
        or not 0 <= config.target_sparsity <= 1
    ):
        raise ValueError("target_sparsity must lie in [0, 1]")
    if config.selection == "matched-sparsity" and config.target_sparsity is None:
        raise ValueError(
            "matched-sparsity requires an explicit pre-declared target_sparsity"
        )
    if config.tol <= 0.0:
        raise ValueError("tol must be positive")
    if config.certified_residual_tolerance < 0:
        raise ValueError("certified_residual_tolerance must be non-negative")
    if not isinstance(config.randomized_full_gradient, (bool, np.bool_)):
        raise TypeError("randomized_full_gradient must be boolean")


def _last_finite(values: Sequence[float] | NDArray[np.float64]) -> float | None:
    if len(values) == 0:
        return None
    value = float(values[-1])
    return value if np.isfinite(value) else None


def _pca_candidate(matrix: NDArray[np.float64], rank: int) -> _Candidate:
    start = time.perf_counter()
    _, _, right_transpose = np.linalg.svd(matrix, full_matrices=False)
    loadings = right_transpose[:rank].T.copy()
    elapsed = time.perf_counter() - start
    return _Candidate(
        method="pca",
        alpha=None,
        A=loadings.copy(),
        B=loadings,
        S=None,
        runtime_seconds=elapsed,
        converged=True,
        n_iter=1,
        objective=None,
        stationarity=None,
        status="closed_form",
    )


def _fit_candidates(
    matrix: NDArray[np.float64], config: BenchmarkConfig, seed: int
) -> list[_Candidate]:
    # Delayed imports keep the metric/data modules independently usable while
    # still calling only the canonical package APIs (there is no local fallback).
    from rspca.accelerated import fit_accelerated_vp_spca
    from rspca.randomized import (
        certified_randomized_vp_spca,
        fit_randomized_vp_spca,
    )
    from rspca.robust import fit_robust_vp_spca
    from rspca.vp import fit_vp_spca

    candidates: list[_Candidate] = []
    for alpha in config.alphas:
        start = time.perf_counter()
        vp = fit_vp_spca(
            matrix,
            config.rank,
            alpha=alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            backtracking=True,
            verbose=False,
        )
        candidates.append(
            _Candidate(
                method="vp_spca",
                alpha=alpha,
                A=vp.A,
                B=vp.B,
                S=None,
                runtime_seconds=time.perf_counter() - start,
                converged=bool(vp.converged),
                n_iter=int(vp.n_iter),
                objective=_last_finite(vp.objectives),
                stationarity=_last_finite(vp.stationarity),
                status=str(vp.status),
            )
        )

        start = time.perf_counter()
        accelerated = fit_accelerated_vp_spca(
            matrix,
            config.rank,
            alpha=alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            acceleration=True,
            backtracking=True,
            monotone=True,
            adaptive_restart=True,
            verbose=False,
        )
        candidates.append(
            _Candidate(
                method="accelerated_vp_spca",
                alpha=alpha,
                A=accelerated.A,
                B=accelerated.B,
                S=None,
                runtime_seconds=time.perf_counter() - start,
                converged=bool(accelerated.converged),
                n_iter=int(accelerated.n_iter),
                objective=_last_finite(accelerated.objectives),
                stationarity=_last_finite(accelerated.stationarity),
                status=str(accelerated.status),
                diagnostics={
                    "restart_count": int(accelerated.restart_count),
                    "monotone_restarts": int(accelerated.monotone_restarts),
                    "adaptive_restarts": int(accelerated.adaptive_restarts),
                },
            )
        )

        start = time.perf_counter()
        randomized = fit_randomized_vp_spca(
            matrix,
            config.rank,
            alpha=alpha,
            oversample=config.oversample,
            n_power_iter=config.n_power_iter,
            max_iter=config.max_iter,
            tol=config.tol,
            use_full_gradient=config.randomized_full_gradient,
            finalize_full_iters=config.randomized_finalize_full_iters,
            random_state=seed,
            verbose=False,
        )
        candidates.append(
            _Candidate(
                method="randomized_vp_spca",
                alpha=alpha,
                A=randomized.A,
                B=randomized.B,
                S=None,
                runtime_seconds=time.perf_counter() - start,
                converged=bool(randomized.converged),
                n_iter=int(randomized.n_iter),
                objective=(
                    float(randomized.full_objective)
                    if np.isfinite(randomized.full_objective)
                    else _last_finite(randomized.objectives)
                ),
                stationarity=(
                    float(randomized.full_stationarity)
                    if np.isfinite(randomized.full_stationarity)
                    else _last_finite(randomized.stationarity)
                ),
                status=str(randomized.status),
                diagnostics={
                    "sketch_final_objective": _last_finite(
                        randomized.sketch_objectives
                    ),
                    "full_objective_diagnostic": float(
                        randomized.full_objective
                    ),
                    "sketch_converged": randomized.sketch_converged,
                    "full_converged": bool(randomized.full_converged),
                    "sketch_residual": float(randomized.sketch_residual),
                    "sketch_residual_relative": float(
                        randomized.sketch_residual_relative
                    ),
                    "full_stationarity_diagnostic": float(
                        randomized.full_stationarity
                    ),
                    "full_diagnostic_step_size": float(
                        randomized.full_diagnostic_step_size
                    ),
                    "full_diagnostic_step_policy": str(
                        randomized.full_diagnostic_step_policy
                    ),
                    "full_refinement_iterations": int(
                        randomized.full_refinement_iterations
                    ),
                },
            )
        )

        start = time.perf_counter()
        certified = certified_randomized_vp_spca(
            matrix,
            config.rank,
            alpha=alpha,
            oversample=config.oversample,
            n_power_iter=config.n_power_iter,
            max_iter=config.max_iter,
            tol=config.tol,
            residual_tolerance=config.certified_residual_tolerance,
            full_stationarity_tolerance=config.tol,
            full_refine_iters=config.certified_full_refine_iters,
            random_state=seed,
            verbose=False,
        )
        certificate = certified.certificate
        if certificate is None:
            raise RuntimeError("certified randomized VP-SPCA returned no certificate")
        candidates.append(
            _Candidate(
                method="certified_randomized_vp_spca",
                alpha=alpha,
                A=certified.A,
                B=certified.B,
                S=None,
                runtime_seconds=time.perf_counter() - start,
                converged=bool(certified.converged),
                n_iter=int(certified.n_iter),
                objective=float(certified.full_objective),
                stationarity=float(certified.full_stationarity),
                status=str(certified.status),
                diagnostics={
                    "restart_count": int(certificate.restart_count),
                    "monotone_restarts": int(certificate.monotone_restarts),
                    "adaptive_restarts": int(certificate.adaptive_restarts),
                    "sketch_final_objective": _last_finite(
                        certified.sketch_objectives
                    ),
                    "full_objective_diagnostic": float(
                        certified.full_objective
                    ),
                    "sketch_converged": certified.sketch_converged,
                    "full_converged": bool(certified.full_converged),
                    "sketch_residual": float(certified.sketch_residual),
                    "sketch_residual_relative": float(
                        certified.sketch_residual_relative
                    ),
                    "full_stationarity_diagnostic": float(
                        certified.full_stationarity
                    ),
                    "full_diagnostic_step_size": float(
                        certified.full_diagnostic_step_size
                    ),
                    "full_diagnostic_step_policy": str(
                        certified.full_diagnostic_step_policy
                    ),
                    "full_refinement_iterations": int(
                        certified.full_refinement_iterations
                    ),
                    "refinement_triggered": bool(certificate.refinement_triggered),
                    "certificate_passed": bool(certificate.certified),
                    "certificate_trigger_reason": str(certificate.trigger_reason),
                    "certificate_loss_gap_upper_bound": float(
                        certificate.loss_gap_upper_bound
                    ),
                },
            )
        )

        start = time.perf_counter()
        robust = fit_robust_vp_spca(
            matrix,
            config.rank,
            alpha=alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=False,
        )
        candidates.append(
            _Candidate(
                method="robust_vp_spca",
                alpha=alpha,
                A=robust.A,
                B=robust.B,
                S=robust.S,
                runtime_seconds=time.perf_counter() - start,
                converged=bool(robust.converged),
                n_iter=int(robust.n_iter),
                objective=_last_finite(robust.objectives),
                stationarity=_last_finite(robust.stationarity),
                status=str(robust.status),
            )
        )
    return candidates


def _threshold_to_sparsity(
    loadings: NDArray[np.float64], target_sparsity: float
) -> NDArray[np.float64]:
    if not 0.0 <= target_sparsity <= 1.0:
        raise ValueError("target_sparsity must lie in [0, 1]")
    flattened = np.abs(loadings).reshape(-1)
    keep = int(round((1.0 - target_sparsity) * flattened.size))
    thresholded = np.zeros_like(loadings)
    if keep == 0:
        return thresholded
    if keep >= flattened.size:
        return loadings.copy()
    selected = np.argpartition(flattened, flattened.size - keep)[-keep:]
    thresholded.reshape(-1)[selected] = loadings.reshape(-1)[selected]
    return thresholded


def _matched_candidates(
    matrix: NDArray[np.float64],
    pca: _Candidate,
    candidates: Iterable[_Candidate],
    target_sparsity: float,
) -> list[_Candidate]:
    pca.B = _threshold_to_sparsity(pca.B, target_sparsity)
    pca.method = "thresholded_pca_matched_sparsity"
    pca.diagnostics["loading_postprocessing"] = (
        "hard_threshold_to_predeclared_target_sparsity"
    )
    cross_covariance = matrix.T @ (matrix @ pca.B)
    left, _, right_transpose = np.linalg.svd(cross_covariance, full_matrices=False)
    pca.A = left @ right_transpose
    chosen = [pca]
    by_method: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_method.setdefault(candidate.method, []).append(candidate)
    for method in sorted(by_method):
        selected = min(
            by_method[method],
            key=lambda candidate: (
                abs(sparsity_ratio(candidate.B) - target_sparsity),
                float(candidate.alpha or 0.0),
            ),
        )
        chosen.append(selected)
    return chosen


def _aligned_loadings(
    truth: NDArray[np.float64], estimate: NDArray[np.float64]
) -> tuple[NDArray[np.float64], dict[str, Any]]:
    correlations = component_correlation(truth, estimate)
    aligned = np.zeros_like(truth)
    for reference_index, estimated_index, sign in zip(
        correlations["reference_indices"],
        correlations["estimated_indices"],
        correlations["signs"],
    ):
        aligned[:, reference_index] = sign * estimate[:, estimated_index]
    return aligned, correlations


def _evaluate_candidate(
    candidate: _Candidate,
    *,
    matrix: NDArray[np.float64],
    target: NDArray[np.float64],
    true_loadings: NDArray[np.float64],
    true_feature_support: NDArray[np.bool_],
    true_corruption_support: NDArray[np.bool_],
    outlier_budget_fraction: float,
    outlier_budget_k: int,
    run_index: int,
    seed: int,
    selection: str,
    target_sparsity: float | None,
    data_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    reconstruction = matrix @ candidate.B @ candidate.A.T
    aligned, correlations = _aligned_loadings(true_loadings, candidate.B)
    support = support_precision_recall_f1(true_feature_support, aligned)
    if candidate.S is None:
        outlier_scores = np.abs(matrix - reconstruction)
    else:
        outlier_scores = np.abs(candidate.S)
    outlier = outlier_detection_metrics(
        true_corruption_support,
        scores=outlier_scores,
        top_k=outlier_budget_k,
    )
    row: dict[str, Any] = {
        "run": run_index,
        "seed": seed,
        "method": candidate.method,
        "selection": selection,
        "alpha": candidate.alpha,
        "target_sparsity": target_sparsity,
        "achieved_sparsity": sparsity_ratio(candidate.B),
        "relative_reconstruction_error": relative_reconstruction_error(
            target, reconstruction
        ),
        "explained_variance_ratio": explained_variance_ratio(
            target, reconstruction, center=False
        ),
        "subspace_error": subspace_error(true_loadings, candidate.B),
        "component_mean_abs_correlation": correlations["mean_abs_correlation"],
        "component_min_abs_correlation": correlations["min_abs_correlation"],
        "support_precision": support["precision"],
        "support_recall": support["recall"],
        "support_f1": support["f1"],
        "outlier_precision": outlier["precision"],
        "outlier_recall": outlier["recall"],
        "outlier_f1": outlier["f1"],
        "outlier_roc_auc": outlier["roc_auc"],
        "outlier_average_precision": outlier["average_precision"],
        "outlier_selection_policy": "predeclared_corruption_fraction_top_k",
        "outlier_budget_fraction": outlier_budget_fraction,
        "outlier_budget_k": outlier_budget_k,
        "centering_policy": "none_dgp_components_column_centered_before_corruption",
        "peak_memory_scope": "not_measured",
        "runtime_comparison_scope": "exploratory_not_matvec_or_phase_matched",
        "iteration_budget_scope": (
            "method_specific_max_iter_randomized_and_certified_have_distinct_phases"
        ),
        "loading_postprocessing": "none",
        "runtime_seconds": candidate.runtime_seconds,
        "converged": candidate.converged,
        "n_iter": candidate.n_iter,
        "final_objective": candidate.objective,
        "final_stationarity": candidate.stationarity,
        "status": candidate.status,
        "restart_count": None,
        "monotone_restarts": None,
        "adaptive_restarts": None,
        "sketch_residual": None,
        "sketch_residual_relative": None,
        "sketch_final_objective": None,
        "full_objective_diagnostic": None,
        "sketch_converged": None,
        "full_converged": None,
        "full_stationarity_diagnostic": None,
        "full_diagnostic_step_size": None,
        "full_diagnostic_step_policy": None,
        "full_refinement_iterations": None,
        "refinement_triggered": None,
        "certificate_passed": None,
        "certificate_trigger_reason": None,
        "certificate_loss_gap_upper_bound": None,
        **data_provenance,
    }
    row.update(candidate.diagnostics)
    return row


def run_benchmark(config: BenchmarkConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run paired repetitions and return flat result rows plus the seed manifest."""

    _validate_config(config)
    software_versions = _software_versions()
    manifest = make_seed_manifest(
        config.base_seed,
        config.repeats,
        namespace="spca-benchmark-v2",
        metadata={
            "config": asdict(config),
            "software_versions": software_versions,
        },
    )
    rows: list[dict[str, Any]] = []
    run_provenance: list[dict[str, Any]] = []
    for run_index, seed in enumerate(manifest["seeds"]):
        dataset = generate_sparse_low_rank(
            n_samples=config.n_samples,
            n_features=config.n_features,
            rank=config.rank,
            support_fraction=config.support_fraction,
            noise_std=config.noise_std,
            corruption_fraction=config.corruption_fraction,
            corruption_scale=config.corruption_scale,
            corruption_distribution=config.corruption_distribution,
            seed=seed,
        )
        # Do not empirically center after corruptions are injected.  The latent
        # signal and Gaussian noise are already column-centered by the DGP;
        # centering the observed matrix here would turn sparse cellwise
        # corruption into a dense column offset and invalidate its support mask.
        matrix = dataset.observed
        target = dataset.low_rank

        actual_corruption = matrix - dataset.uncorrupted
        actual_corruption_support = actual_corruption != 0.0
        if not np.array_equal(actual_corruption_support, dataset.corruption_support):
            raise RuntimeError("solver-input corruption support differs from ground truth")
        outlier_budget_k = int(round(config.corruption_fraction * matrix.size))
        data_provenance: dict[str, Any] = {
            "observed_sha256": array_sha256(matrix),
            "clean_uncorrupted_sha256": array_sha256(dataset.uncorrupted),
            "low_rank_truth_sha256": array_sha256(dataset.low_rank),
            "true_loadings_sha256": array_sha256(dataset.loadings),
            "feature_support_sha256": array_sha256(dataset.feature_support),
            "corruption_support_sha256": array_sha256(
                dataset.corruption_support
            ),
            "corruption_values_sha256": array_sha256(dataset.corruption),
        }
        data_provenance["run_data_sha256"] = canonical_sha256(data_provenance)
        run_provenance.append(
            {
                "run": run_index,
                "seed": seed,
                **data_provenance,
            }
        )

        pca = _pca_candidate(matrix, config.rank)
        fitted = _fit_candidates(matrix, config, seed)
        if config.selection == "matched-sparsity":
            # Validation guarantees this is explicitly supplied rather than
            # inferred from the synthetic ground-truth support.
            assert config.target_sparsity is not None
            selected = _matched_candidates(
                matrix, pca, fitted, config.target_sparsity
            )
            row_target: float | None = config.target_sparsity
        else:
            selected = [pca, *fitted]
            row_target = None
        for candidate in selected:
            rows.append(
                _evaluate_candidate(
                    candidate,
                    matrix=matrix,
                    target=target,
                    true_loadings=dataset.loadings,
                    true_feature_support=dataset.feature_support,
                    true_corruption_support=dataset.corruption_support,
                    outlier_budget_fraction=config.corruption_fraction,
                    outlier_budget_k=outlier_budget_k,
                    run_index=run_index,
                    seed=seed,
                    selection=config.selection,
                    target_sparsity=row_target,
                    data_provenance=data_provenance,
                )
            )
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256", None)
    manifest_body["run_provenance"] = run_provenance
    manifest_body["software_versions"] = software_versions
    manifest_body["manifest_sha256"] = canonical_sha256(manifest_body)
    manifest = manifest_body
    return rows, manifest


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_results(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    config: BenchmarkConfig,
    *,
    output_json: Path | None = None,
    output_csv: Path | None = None,
) -> None:
    """Write explicitly requested artifacts; passing no path performs no I/O."""

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "config": asdict(config),
            "provenance": {"software_versions": _software_versions()},
            "seed_manifest": manifest,
            "records": rows,
        }
        output_json.write_text(
            json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_context = {
            "manifest_sha256": manifest["manifest_sha256"],
            "n_samples": config.n_samples,
            "n_features": config.n_features,
            "rank": config.rank,
            "support_fraction": config.support_fraction,
            "noise_std": config.noise_std,
            "corruption_fraction": config.corruption_fraction,
            "corruption_scale": config.corruption_scale,
            "corruption_distribution": config.corruption_distribution,
            **{
                f"version_{name}": version
                for name, version in _software_versions().items()
            },
        }
        csv_rows = [{**csv_context, **row} for row in rows]
        fieldnames = list(csv_rows[0]) if csv_rows else []
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(_json_safe(csv_rows))


def summarize_reconstruction_errors(
    rows: Sequence[dict[str, Any]], selection: str
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    """Summarize without pooling distinct alpha-grid configurations."""

    if selection == "alpha-grid":
        group_fields = ("method", "alpha")
    elif selection == "matched-sparsity":
        group_fields = ("method",)
    else:
        raise ValueError("selection must be alpha-grid or matched-sparsity")
    baseline_method = (
        "pca"
        if selection == "alpha-grid"
        else "thresholded_pca_matched_sparsity"
    )
    baseline_by_pair: dict[tuple[int, int], float] = {}
    for row in rows:
        if row["method"] != baseline_method:
            continue
        pair = (int(row["run"]), int(row["seed"]))
        if pair in baseline_by_pair:
            raise ValueError(f"duplicate PCA baseline for run/seed pair {pair}")
        baseline_by_pair[pair] = float(row["relative_reconstruction_error"])
    groups = sorted(
        {tuple(row[field] for field in group_fields) for row in rows},
        key=lambda group: tuple(
            "" if value is None else str(value) for value in group
        ),
    )
    summary: list[dict[str, Any]] = []
    for group in groups:
        group_rows = [
            row
            for row in rows
            if tuple(row[field] for field in group_fields) == group
        ]
        values = np.asarray(
            [row["relative_reconstruction_error"] for row in group_rows],
            dtype=float,
        )
        group_by_pair: dict[tuple[int, int], float] = {}
        for row in group_rows:
            pair = (int(row["run"]), int(row["seed"]))
            if pair in group_by_pair:
                raise ValueError(f"duplicate result for group {group} and pair {pair}")
            group_by_pair[pair] = float(row["relative_reconstruction_error"])
        paired_keys = sorted(set(group_by_pair) & set(baseline_by_pair))
        paired_differences = np.asarray(
            [
                group_by_pair[pair] - baseline_by_pair[pair]
                for pair in paired_keys
            ],
            dtype=float,
        )
        n_pairs = int(paired_differences.size)
        if n_pairs:
            paired_mean: float | None = float(np.mean(paired_differences))
            paired_std: float | None = float(
                np.std(paired_differences, ddof=1) if n_pairs > 1 else 0.0
            )
        else:
            paired_mean = None
            paired_std = None
        if n_pairs >= 2:
            half_width = float(
                student_t.ppf(0.975, df=n_pairs - 1)
                * float(paired_std)
                / np.sqrt(n_pairs)
            )
            paired_ci_low = float(paired_mean - half_width)
            paired_ci_high = float(paired_mean + half_width)
            paired_ci_status = "ok"
        else:
            paired_ci_low = None
            paired_ci_high = None
            paired_ci_status = "insufficient_pairs"
        summary.append(
            {
                **dict(zip(group_fields, group)),
                "n": int(values.size),
                "mean_relative_reconstruction_error": float(np.mean(values)),
                "std_relative_reconstruction_error": float(
                    np.std(values, ddof=1) if values.size > 1 else 0.0
                ),
                "paired_baseline_method": baseline_method,
                "n_pairs": n_pairs,
                "mean_paired_error_difference_vs_pca": paired_mean,
                "std_paired_error_difference_vs_pca": paired_std,
                "paired_error_difference_ci95_low": paired_ci_low,
                "paired_error_difference_ci95_high": paired_ci_high,
                "paired_error_difference_ci95_status": paired_ci_status,
            }
        )
    return group_fields, summary


def _parse_alphas(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from error
    if (
        not alphas
        or any(not np.isfinite(alpha) or alpha < 0 for alpha in alphas)
        or len(set(alphas)) != len(alphas)
    ):
        raise argparse.ArgumentTypeError(
            "alphas must contain unique finite non-negative numbers"
        )
    return alphas


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=240)
    parser.add_argument("--n-features", type=int, default=120)
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--support-fraction", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--corruption-fraction", type=float, default=0.05)
    parser.add_argument("--corruption-scale", type=float, default=8.0)
    parser.add_argument(
        "--corruption-distribution",
        choices=("gaussian", "laplace", "rademacher"),
        default="gaussian",
    )
    parser.add_argument(
        "--alphas",
        type=_parse_alphas,
        default=(0.01, 0.03, 0.1, 0.3, 1.0, 3.0),
    )
    parser.add_argument(
        "--selection", choices=("alpha-grid", "matched-sparsity"), default="alpha-grid"
    )
    parser.add_argument("--target-sparsity", type=float)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--n-power-iter", type=int, default=2)
    parser.add_argument("--randomized-full-gradient", action="store_true")
    parser.add_argument("--randomized-finalize-full-iters", type=int, default=5)
    parser.add_argument("--certified-residual-tolerance", type=float, default=0.05)
    parser.add_argument("--certified-full-refine-iters", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    config = BenchmarkConfig(
        n_samples=arguments.n_samples,
        n_features=arguments.n_features,
        rank=arguments.rank,
        support_fraction=arguments.support_fraction,
        noise_std=arguments.noise_std,
        corruption_fraction=arguments.corruption_fraction,
        corruption_scale=arguments.corruption_scale,
        corruption_distribution=arguments.corruption_distribution,
        alphas=arguments.alphas,
        selection=arguments.selection,
        target_sparsity=arguments.target_sparsity,
        repeats=arguments.repeats,
        base_seed=arguments.base_seed,
        max_iter=arguments.max_iter,
        tol=arguments.tol,
        oversample=arguments.oversample,
        n_power_iter=arguments.n_power_iter,
        randomized_full_gradient=arguments.randomized_full_gradient,
        randomized_finalize_full_iters=arguments.randomized_finalize_full_iters,
        certified_residual_tolerance=arguments.certified_residual_tolerance,
        certified_full_refine_iters=arguments.certified_full_refine_iters,
    )
    rows, manifest = run_benchmark(config)
    write_results(
        rows,
        manifest,
        config,
        output_json=arguments.output_json,
        output_csv=arguments.output_csv,
    )
    group_fields, summary = summarize_reconstruction_errors(rows, config.selection)
    print(
        json.dumps(
            {
                "records": len(rows),
                "manifest_sha256": manifest["manifest_sha256"],
                "summary_grouping": list(group_fields),
                "relative_reconstruction_error_summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

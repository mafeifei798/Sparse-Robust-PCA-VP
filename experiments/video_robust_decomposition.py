"""Development and confirmation study of robust synthetic-video decomposition.

This study is separate from ``video_decomposition_reference.py`` and preserves
the reference experiment's saved outputs.
It retains the historical GitHub construction and orientation
``X = spatial pixels x time snapshots`` with ``Phi = X @ B``.  The solver is
the repository's existing :func:`rspca.robust.fit_robust_vp_spca`;
deterministic multistart and B-only descending-kappa continuation are studied
as initialization/solve strategies, not presented as new PCA algorithms.

The workflow has two fail-closed phases:

1. ``develop`` may use simulation truth on declared development corruption
   seeds to select exactly one ``(alpha, kappa)`` pair and write a source-bound
   selection lock.
2. ``confirm`` reads that lock, permits no tuning overrides, and evaluates the
   locked protocol on a disjoint, previously untouched seed bank.

Within every fixed-parameter arm, candidate starts are selected solely by the
independently recomputed robust objective on observed data at the common target
kappa.  Clean truth, corruption support, convergence, runtime, and stationarity
are not selection inputs.  If the objective winner is degenerate, it remains
the winner and fails the separately reported nontrivial-solution gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from experiments.video_decomposition_reference import (
    ARXIV_ID,
    ARXIV_PDF_SHA256,
    ARXIV_VERSION,
    LEGACY_SOURCE_COMMIT,
    LEGACY_SOURCE_SHA256,
    PaperVideoConfig,
    RobustDecompositionDataset,
    generate_figure6_dataset,
    normalized_full_frame_errors,
    summarize_full_frame_errors,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.metrics import outlier_detection_metrics, relative_reconstruction_error
from rspca.robust import RobustVPSPCAResult, fit_robust_vp_spca, robust_objective


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

STUDY_PROTOCOL_VERSION = 1
LEGACY_ROBUST_SOURCE_SHA256 = (
    "eac1b41bd1fb27d5542cf42d3cb05e1498ebbc3be57616efe1fa112133ac4692"
)
DEVELOPMENT_RESULTS_NAME = "development_results.json"
DEVELOPMENT_GRID_NAME = "development_grid.csv"
SELECTION_LOCK_NAME = "selection_lock.json"
DEVELOPMENT_MANIFEST_NAME = "development_output_manifest.json"
CONFIRMATION_REPORT_NAME = "confirmation_report.json"
CONFIRMATION_CASES_NAME = "confirmation_cases.csv"
CONFIRMATION_FRAMES_NAME = "confirmation_full_frame_errors.csv"
CONFIRMATION_FIGURE_NAME = "confirmation_representative_and_worst_frames.png"
CONFIRMATION_MANIFEST_NAME = "confirmation_output_manifest.json"

# Version-1 server artifacts used the ``true_budget`` label below even though
# the implementation has always ranked exactly
# ``round(protocol corruption_fraction * observed.size)`` cells.  That is a
# nominal protocol budget, not the realized Bernoulli support count.  Keep the
# legacy spelling readable, but use the scientifically accurate name in all
# newly generated summaries and CSV files.
OUTLIER_NOMINAL_BUDGET_F1 = "outlier_nominal_budget_f1"
LEGACY_OUTLIER_TRUE_BUDGET_F1 = "outlier_true_budget_f1"
SEMANTIC_AUDIT_VERSION = 1

_DEVELOPMENT_REQUIRED_PAYLOADS = frozenset(
    {DEVELOPMENT_RESULTS_NAME, DEVELOPMENT_GRID_NAME, SELECTION_LOCK_NAME}
)
_DEVELOPMENT_REQUIRED_OUTPUTS = _DEVELOPMENT_REQUIRED_PAYLOADS | {
    DEVELOPMENT_MANIFEST_NAME
}
_CONFIRMATION_REQUIRED_PAYLOADS = frozenset(
    {
        CONFIRMATION_REPORT_NAME,
        CONFIRMATION_CASES_NAME,
        CONFIRMATION_FRAMES_NAME,
        CONFIRMATION_FIGURE_NAME,
        SELECTION_LOCK_NAME,
    }
)
_CONFIRMATION_REQUIRED_OUTPUTS = _CONFIRMATION_REQUIRED_PAYLOADS | {
    CONFIRMATION_MANIFEST_NAME
}

# These seeds have already appeared in viewed exploratory output.  They may be
# used for development but can never enter a confirmation bank.
KNOWN_VIEWED_OR_DEVELOPMENT_SEEDS = frozenset(
    {120, 123, *range(91001, 91021)}
)
DEFAULT_DEVELOPMENT_SEEDS = (123, 1123, 2123)
DEFAULT_CONFIRMATION_SEEDS = tuple(range(51001, 51021))


@dataclass(frozen=True)
class StartSpec:
    """Observed-only deterministic loading initialization."""

    start_id: str
    kind: str
    random_seed: int | None = None
    perturbation_scale: float | None = None


@dataclass(frozen=True)
class RobustStudyConfig:
    """Frozen simulation and solver protocol used to create a selection lock."""

    grid_size: int = 200
    n_timesteps: int = 300
    total_time: float = 150.0
    n_components: int = 3
    corruption_fraction: float = 0.05
    corruption_scale: float = 3.0
    development_seeds: tuple[int, ...] = DEFAULT_DEVELOPMENT_SEEDS
    confirmation_seeds: tuple[int, ...] = DEFAULT_CONFIRMATION_SEEDS
    alpha_grid: tuple[float, ...] = (0.0015, 0.003, 0.006)
    kappa_grid: tuple[float, ...] = (0.001, 0.002, 0.005)
    max_iter_per_start: int = 200
    tol: float = 1e-6
    legacy_dynamic_alpha: float = 0.012
    legacy_dynamic_max_iter: int = 500
    legacy_dynamic_tol: float = 1e-6
    legacy_huber_tau: float = 1.0
    legacy_kappa_min: float = 0.005
    legacy_kappa_max: float = 0.04
    continuation_kappa_multipliers: tuple[float, ...] = (4.0, 1.0)
    continuation_stage_max_iters: tuple[int, ...] = (40, 160)
    random_orthogonal_seed: int = 61001
    perturbation_scales: tuple[float, ...] = (0.05, 0.20)
    perturbation_seeds: tuple[int, ...] = (62001, 62002)
    loading_nonzero_threshold: float = 1e-10
    minimum_loading_nonzero_fraction: float = 1e-4
    minimum_reconstruction_to_observed_norm: float = 1e-3
    catastrophic_frame_threshold: float = 2.0
    representative_frame_index: int = 100
    bootstrap_draws: int = 10000
    bootstrap_seed: int = 63001
    minimum_confirmation_convergence_fraction: float = 0.95
    minimum_primary_win_fraction: float = 0.8
    maximum_true_budget_f1_decrease: float = 0.02
    dpi: int = 150


@dataclass(frozen=True)
class ObservedPreparedCase:
    """Observed-only state passed to fitting and candidate selection."""

    observed: FloatArray
    normalized_observed: FloatArray
    normalization_sigma_max: float
    pca_B: FloatArray
    pca_reconstruction: FloatArray
    data_sha256: str
    pca_B_sha256: str
    preparation_runtime_seconds: float


@dataclass(frozen=True)
class CandidateFit:
    """Internal final candidate plus its auditable observed-only record."""

    start_id: str
    arm_id: str
    result: RobustVPSPCAResult
    record: Mapping[str, Any]


@dataclass(frozen=True)
class SelectedArm:
    """All candidates and the stable objective-only winner for one arm."""

    arm_id: str
    selected: CandidateFit
    candidate_records: tuple[Mapping[str, Any], ...]
    selected_index: int
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ConfirmationCaseInternal:
    """One confirmation seed, including small frame snapshots for plotting."""

    seed: int
    record: Mapping[str, Any]
    frame_errors: Mapping[str, FloatArray]
    snapshots: Mapping[int, Mapping[str, FloatArray]]


@dataclass(frozen=True)
class LegacyDynamicKappaFit:
    """Instrumented exact port of the origin/main changing-kappa baseline."""

    reconstruction: FloatArray
    outliers: FloatArray
    record: Mapping[str, Any]


def default_start_specs(config: RobustStudyConfig) -> tuple[StartSpec, ...]:
    specs: list[StartSpec] = [StartSpec("pca", "pca")]
    specs.append(
        StartSpec(
            "random_orthogonal",
            "fixed_random_orthogonal",
            random_seed=config.random_orthogonal_seed,
        )
    )
    for index, (scale, seed) in enumerate(
        zip(config.perturbation_scales, config.perturbation_seeds)
    ):
        specs.append(
            StartSpec(
                f"pca_perturb_{index + 1}",
                "pca_perturbation",
                random_seed=seed,
                perturbation_scale=scale,
            )
        )
    return tuple(specs)


def smoke_study_config() -> RobustStudyConfig:
    """Tiny two-phase plumbing protocol; never paper-scale evidence."""

    return RobustStudyConfig(
        grid_size=10,
        n_timesteps=36,
        total_time=150.0,
        development_seeds=(123, 1123),
        confirmation_seeds=(71001, 71002),
        alpha_grid=(0.003, 0.006),
        kappa_grid=(0.001, 0.003),
        max_iter_per_start=4,
        legacy_dynamic_max_iter=4,
        continuation_kappa_multipliers=(2.0, 1.0),
        continuation_stage_max_iters=(1, 3),
        perturbation_scales=(0.10,),
        perturbation_seeds=(62001,),
        representative_frame_index=12,
        bootstrap_draws=200,
        dpi=72,
    )


def _validate_unique_seed_tuple(values: tuple[int, ...], *, name: str) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate seeds")
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must contain integers")
        if int(value) < 0:
            raise ValueError(f"{name} seeds must be non-negative")


def validate_study_config(config: RobustStudyConfig) -> None:
    """Fail closed on leakage, unequal budgets, or ambiguous schedules."""

    _validate_unique_seed_tuple(config.development_seeds, name="development_seeds")
    _validate_unique_seed_tuple(config.confirmation_seeds, name="confirmation_seeds")
    overlap = set(config.development_seeds) & set(config.confirmation_seeds)
    if overlap:
        raise ValueError(f"development and confirmation seeds overlap: {sorted(overlap)}")
    viewed_confirmation = set(config.confirmation_seeds) & set(
        KNOWN_VIEWED_OR_DEVELOPMENT_SEEDS
    )
    if viewed_confirmation:
        raise ValueError(
            "confirmation contains previously viewed/development seeds: "
            f"{sorted(viewed_confirmation)}"
        )
    integer_fields = {
        "grid_size": config.grid_size,
        "n_timesteps": config.n_timesteps,
        "n_components": config.n_components,
        "max_iter_per_start": config.max_iter_per_start,
        "legacy_dynamic_max_iter": config.legacy_dynamic_max_iter,
        "random_orthogonal_seed": config.random_orthogonal_seed,
        "representative_frame_index": config.representative_frame_index,
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
        "dpi": config.dpi,
    }
    for name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 8 or config.n_timesteps <= config.n_components:
        raise ValueError("invalid video dimensions")
    if config.n_components != 3:
        raise ValueError("the historical paper video has exactly 3 components")
    if (
        config.max_iter_per_start <= 0
        or config.legacy_dynamic_max_iter <= 0
        or config.bootstrap_draws <= 0
        or config.dpi <= 0
    ):
        raise ValueError("iteration, bootstrap, and dpi budgets must be positive")
    if not 0 <= config.representative_frame_index < config.n_timesteps:
        raise ValueError("representative_frame_index is outside the video")
    positive_scalars = {
        "total_time": config.total_time,
        "corruption_scale": config.corruption_scale,
        "tol": config.tol,
        "legacy_dynamic_alpha": config.legacy_dynamic_alpha,
        "legacy_dynamic_tol": config.legacy_dynamic_tol,
        "legacy_huber_tau": config.legacy_huber_tau,
        "legacy_kappa_min": config.legacy_kappa_min,
        "legacy_kappa_max": config.legacy_kappa_max,
        "loading_nonzero_threshold": config.loading_nonzero_threshold,
        "minimum_loading_nonzero_fraction": config.minimum_loading_nonzero_fraction,
        "minimum_reconstruction_to_observed_norm": (
            config.minimum_reconstruction_to_observed_norm
        ),
        "catastrophic_frame_threshold": config.catastrophic_frame_threshold,
    }
    for name, value in positive_scalars.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if config.legacy_kappa_min > config.legacy_kappa_max:
        raise ValueError("legacy kappa bounds are reversed")
    if not 0.0 < config.corruption_fraction < 1.0:
        raise ValueError("corruption_fraction must lie in (0, 1)")
    if not 0.0 < config.minimum_confirmation_convergence_fraction <= 1.0:
        raise ValueError("minimum confirmation convergence fraction must lie in (0, 1]")
    if not 0.0 < config.minimum_primary_win_fraction <= 1.0:
        raise ValueError("minimum primary win fraction must lie in (0, 1]")
    if not 0.0 <= config.maximum_true_budget_f1_decrease <= 1.0:
        raise ValueError("maximum true-budget F1 decrease must lie in [0, 1]")
    for name, grid in (("alpha_grid", config.alpha_grid), ("kappa_grid", config.kappa_grid)):
        if not grid or len(set(grid)) != len(grid):
            raise ValueError(f"{name} must be nonempty and duplicate-free")
        if any(not np.isfinite(value) or value <= 0.0 for value in grid):
            raise ValueError(f"{name} values must be finite and positive")
    multipliers = config.continuation_kappa_multipliers
    budgets = config.continuation_stage_max_iters
    if len(multipliers) < 2 or len(multipliers) != len(budgets):
        raise ValueError("continuation schedule and stage budgets have incompatible lengths")
    if any(
        not np.isfinite(value) or value <= 0.0 for value in multipliers
    ) or any(left <= right for left, right in zip(multipliers, multipliers[1:])):
        raise ValueError("continuation kappa multipliers must be strictly descending")
    if multipliers[-1] != 1.0:
        raise ValueError("continuation schedule must end exactly at target kappa")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in budgets
    ):
        raise ValueError("continuation stage budgets must be positive integers")
    if sum(budgets) != config.max_iter_per_start:
        raise ValueError("direct and continuation per-start budgets must be equal")
    if len(config.perturbation_scales) != len(config.perturbation_seeds):
        raise ValueError("perturbation scales and seeds must pair exactly")
    if not config.perturbation_scales:
        raise ValueError("at least one PCA perturbation is required")
    if any(not np.isfinite(value) or value <= 0.0 for value in config.perturbation_scales):
        raise ValueError("perturbation scales must be finite and positive")
    _validate_unique_seed_tuple(config.perturbation_seeds, name="perturbation_seeds")
    specs = default_start_specs(config)
    if len({spec.start_id for spec in specs}) != len(specs):
        raise ValueError("start IDs must be unique")


def _paper_video_config(config: RobustStudyConfig, seed: int) -> PaperVideoConfig:
    return PaperVideoConfig(
        grid_size=config.grid_size,
        n_timesteps=config.n_timesteps,
        total_time=config.total_time,
        n_components=config.n_components,
        corruption_fraction=config.corruption_fraction,
        corruption_scale=config.corruption_scale,
        corruption_seed=int(seed),
        max_iter=config.max_iter_per_start,
        tol=config.tol,
        figure6_frame_index=config.representative_frame_index,
        dpi=config.dpi,
    )


def prepare_observed_case(dataset: RobustDecompositionDataset) -> ObservedPreparedCase:
    """Prepare SVD normalization without accepting clean truth or support."""

    observed = np.asarray(dataset.observed, dtype=np.float64)
    started = perf_counter()
    _, singular_values, right_t = linalg.svd(
        observed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    scale = max(sigma_max, np.finfo(np.float64).eps)
    pca_B = np.asarray(right_t[:3].T, dtype=np.float64)
    spatial_modes = observed @ pca_B
    pca_reconstruction = spatial_modes @ pca_B.T
    runtime_seconds = perf_counter() - started
    return ObservedPreparedCase(
        observed=observed,
        normalized_observed=np.asarray(observed / scale, dtype=np.float64),
        normalization_sigma_max=scale,
        pca_B=pca_B,
        pca_reconstruction=np.asarray(pca_reconstruction, dtype=np.float64),
        data_sha256=array_sha256(observed),
        pca_B_sha256=array_sha256(pca_B),
        preparation_runtime_seconds=float(runtime_seconds),
    )


def build_initializations(
    observed: ObservedPreparedCase,
    specs: Sequence[StartSpec],
) -> dict[str, FloatArray]:
    """Build all starts from observed PCA or fixed RNGs, never from truth."""

    output: dict[str, FloatArray] = {}
    for spec in specs:
        if spec.kind == "pca":
            values = observed.pca_B.copy()
        elif spec.kind == "fixed_random_orthogonal":
            if spec.random_seed is None:
                raise ValueError("random orthogonal start requires a seed")
            rng = np.random.default_rng(spec.random_seed)
            raw = rng.standard_normal(observed.pca_B.shape)
            values, _ = np.linalg.qr(raw, mode="reduced")
        elif spec.kind == "pca_perturbation":
            if spec.random_seed is None or spec.perturbation_scale is None:
                raise ValueError("PCA perturbation requires a seed and scale")
            rng = np.random.default_rng(spec.random_seed)
            perturbation = rng.standard_normal(observed.pca_B.shape)
            norms = np.linalg.norm(perturbation, axis=0, keepdims=True)
            perturbation /= np.maximum(norms, np.finfo(np.float64).eps)
            values = observed.pca_B + spec.perturbation_scale * perturbation
        else:
            raise ValueError(f"unknown start kind: {spec.kind}")
        output[spec.start_id] = np.ascontiguousarray(values, dtype=np.float64)
    if list(output) != [spec.start_id for spec in specs]:
        raise RuntimeError("start construction order changed")
    return output


def _objective_trace_monotone(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    array = np.asarray(values, dtype=np.float64)
    tolerance = 1e-10 * np.maximum(1.0, np.abs(array[:-1]))
    return bool(np.all(np.diff(array) <= tolerance))


def _nontrivial_arrays_gate(
    observed: ObservedPreparedCase,
    *,
    A: FloatArray,
    B: FloatArray,
    config: RobustStudyConfig,
) -> dict[str, Any]:
    loading_nonzero_fraction = float(
        np.mean(np.abs(B) > config.loading_nonzero_threshold)
    )
    normalized_reconstruction = (
        observed.normalized_observed @ B @ A.T
    )
    reconstruction_ratio = float(
        np.linalg.norm(normalized_reconstruction, ord="fro")
        / max(
            float(np.linalg.norm(observed.normalized_observed, ord="fro")),
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
        "loading_nonzero_threshold": config.loading_nonzero_threshold,
        "loading_nonzero_fraction": loading_nonzero_fraction,
        "minimum_loading_nonzero_fraction": config.minimum_loading_nonzero_fraction,
        "reconstruction_to_observed_frobenius_norm": reconstruction_ratio,
        "minimum_reconstruction_to_observed_norm": (
            config.minimum_reconstruction_to_observed_norm
        ),
        "loading_rank": loading_rank,
        "required_loading_rank": config.n_components,
        "selection_filter": False,
        "failure_policy": "retain objective winner and fail qualification",
    }


def _nontrivial_solution_gate(
    observed: ObservedPreparedCase,
    result: RobustVPSPCAResult,
    config: RobustStudyConfig,
) -> dict[str, Any]:
    return _nontrivial_arrays_gate(
        observed,
        A=np.asarray(result.A, dtype=np.float64),
        B=np.asarray(result.B, dtype=np.float64),
        config=config,
    )


def _legacy_soft_threshold(values: FloatArray, threshold: float) -> FloatArray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _fit_legacy_dynamic_kappa_exact(
    observed: ObservedPreparedCase,
    config: RobustStudyConfig,
) -> LegacyDynamicKappaFit:
    """Exact-operation port of origin/main ``robust_spca_vp`` plus audit trace.

    The objective changes whenever the residual-MAD kappa changes.  Its final
    diagnostic objective is therefore never compared with fixed-kappa
    candidate objectives and never enters their objective-only selection.
    """

    started = perf_counter()
    _, singular_values, right_t = linalg.svd(
        observed.observed,
        full_matrices=False,
    )
    sigma_max = float(singular_values[0]) if singular_values.size else 1.0
    normalized = observed.observed / sigma_max
    right = right_t.T
    A = np.asarray(right[:, : config.n_components].copy(), dtype=np.float64)
    B = np.asarray(right[:, : config.n_components].copy(), dtype=np.float64)
    sparse = np.zeros_like(normalized)
    kappa_trace: list[float] = []
    B_change_trace: list[float] = []
    stopped = False
    iterations = 0
    for iteration in range(config.legacy_dynamic_max_iter):
        residual_for_gradient = normalized @ B @ A.T - normalized + sparse
        gradient = normalized.T @ (residual_for_gradient @ A)
        next_B = _legacy_soft_threshold(
            B - gradient, config.legacy_dynamic_alpha
        )
        change = float(linalg.norm(next_B - B, "fro"))
        B_change_trace.append(change)
        iterations = iteration + 1
        if change <= config.legacy_dynamic_tol:
            B = next_B
            stopped = True
            break
        B = next_B
        procrustes_matrix = (normalized - sparse).T @ (normalized @ B)
        left, _, right_transpose = linalg.svd(
            procrustes_matrix, full_matrices=False
        )
        A = left @ right_transpose
        residual = normalized - normalized @ B @ A.T
        median = float(np.median(residual))
        sigma_hat = (
            float(np.median(np.abs(residual - median))) / 0.6745 + 1e-12
        )
        kappa_effective = float(
            np.clip(
                config.legacy_huber_tau * sigma_hat,
                config.legacy_kappa_min,
                config.legacy_kappa_max,
            )
        )
        kappa_trace.append(kappa_effective)
        sparse = _legacy_soft_threshold(residual, kappa_effective)
    runtime = perf_counter() - started
    normalized_reconstruction = normalized @ B @ A.T
    reconstruction = np.asarray(
        sigma_max * normalized_reconstruction, dtype=np.float64
    )
    outliers = np.asarray(sigma_max * sparse, dtype=np.float64)
    final_kappa = kappa_trace[-1] if kappa_trace else config.legacy_kappa_min
    diagnostic_objective = robust_objective(
        normalized,
        A,
        B,
        sparse,
        alpha=config.legacy_dynamic_alpha,
        kappa=final_kappa,
    )
    nontrivial = _nontrivial_arrays_gate(
        observed, A=A, B=B, config=config
    )
    record: dict[str, Any] = {
        "method_id": "legacy_dynamic_kappa",
        "historical_source": {
            "repository": "mafeifei798/Robust-and-Sparse-PCA",
            "commit": LEGACY_SOURCE_COMMIT,
            "path": "robust_spca_vp.py",
            "sha256": LEGACY_ROBUST_SOURCE_SHA256,
        },
        "port_policy": (
            "same SVD normalization, PCA A/B initialization, update order, "
            "unit B step, residual-MAD kappa clipping, soft thresholds, and "
            "B-change stopping; added diagnostics do not alter iterates"
        ),
        "alpha": config.legacy_dynamic_alpha,
        "max_iter": config.legacy_dynamic_max_iter,
        "tol": config.legacy_dynamic_tol,
        "huber_tau": config.legacy_huber_tau,
        "kappa_min": config.legacy_kappa_min,
        "kappa_max": config.legacy_kappa_max,
        "changing_objective": True,
        "eligible_for_fixed_target_objective_selection": False,
        "convergence_semantics": "historical B-change stopping only",
        "converged": stopped,
        "status": (
            "historical B-change tolerance reached"
            if stopped
            else "historical maximum iterations reached"
        ),
        "actual_iterations": iterations,
        "allocated_iterations": config.legacy_dynamic_max_iter,
        "solver_calls": 1,
        "B_change_trace": B_change_trace,
        "kappa_trace": kappa_trace,
        "final_dynamic_kappa": final_kappa,
        "final_diagnostic_objective_at_final_kappa": float(
            diagnostic_objective
        ),
        "diagnostic_objective_comparable_to_fixed_target_arms": False,
        "normalization_sigma_max": sigma_max,
        "A_sha256": array_sha256(A),
        "B_sha256": array_sha256(B),
        "outliers_sha256_raw_units": array_sha256(outliers),
        "reconstruction_sha256": array_sha256(reconstruction),
        "nontrivial_solution_gate": nontrivial,
        "runtime_seconds": float(runtime),
    }
    record["fit_sha256"] = canonical_sha256(
        {
            "A": record["A_sha256"],
            "B": record["B_sha256"],
            "S_raw": record["outliers_sha256_raw_units"],
        }
    )
    return LegacyDynamicKappaFit(
        reconstruction=reconstruction,
        outliers=outliers,
        record=record,
    )


def _fit_candidate(
    observed: ObservedPreparedCase,
    *,
    start_id: str,
    arm_id: str,
    initial_B: FloatArray,
    alpha: float,
    target_kappa: float,
    kappa_schedule: Sequence[float],
    stage_budgets: Sequence[int],
    config: RobustStudyConfig,
) -> CandidateFit:
    if len(kappa_schedule) != len(stage_budgets) or not kappa_schedule:
        raise ValueError("candidate schedule is empty or inconsistent")
    if float(kappa_schedule[-1]) != float(target_kappa):
        raise ValueError("candidate schedule must end at the common target kappa")
    current_B = np.ascontiguousarray(initial_B, dtype=np.float64)
    initial_hash = array_sha256(current_B)
    previous_final_hash: str | None = None
    stages: list[dict[str, Any]] = []
    total_runtime = 0.0
    final_result: RobustVPSPCAResult | None = None
    for stage_index, (kappa, max_iter) in enumerate(
        zip(kappa_schedule, stage_budgets)
    ):
        stage_initial_hash = array_sha256(current_B)
        if previous_final_hash is not None and stage_initial_hash != previous_final_hash:
            raise RuntimeError("B-only continuation handoff hash mismatch")
        started = perf_counter()
        result = fit_robust_vp_spca(
            observed.normalized_observed,
            config.n_components,
            alpha=alpha,
            kappa=float(kappa),
            max_iter=int(max_iter),
            tol=config.tol,
            step_size=1.0,
            initial_B=current_B,
            backtracking=True,
            verbose=False,
        )
        stage_runtime = perf_counter() - started
        total_runtime += stage_runtime
        final_B_hash = array_sha256(result.B)
        stages.append(
            {
                "stage_index": int(stage_index),
                "kappa": float(kappa),
                "allocated_max_iter": int(max_iter),
                "actual_iterations": int(result.n_iter),
                "converged": bool(result.converged),
                "status": result.status,
                "final_stationarity": (
                    float(result.stationarity[-1]) if result.stationarity else None
                ),
                "objective_trace": [float(value) for value in result.objectives],
                "objective_trace_monotone": _objective_trace_monotone(
                    result.objectives
                ),
                "initial_B_sha256": stage_initial_hash,
                "final_B_sha256": final_B_hash,
                "handoff_from_previous_final_B": (
                    None
                    if previous_final_hash is None
                    else stage_initial_hash == previous_final_hash
                ),
                "A_and_S_reinitialized_for_stage": True,
                "runtime_seconds": float(stage_runtime),
            }
        )
        current_B = result.B
        previous_final_hash = final_B_hash
        final_result = result
    if final_result is None:  # pragma: no cover - guarded above
        raise RuntimeError("candidate produced no final result")
    recomputed = robust_objective(
        observed.normalized_observed,
        final_result.A,
        final_result.B,
        final_result.S,
        alpha=alpha,
        kappa=target_kappa,
    )
    reported = float(final_result.objectives[-1])
    consistency = abs(float(recomputed) - reported)
    tolerance = 1e-10 * max(1.0, abs(reported))
    if not np.isfinite(recomputed) or not np.isfinite(reported):
        raise RuntimeError("candidate objective is non-finite")
    if consistency > tolerance:
        raise RuntimeError("reported and recomputed target objectives disagree")
    nontrivial = _nontrivial_solution_gate(observed, final_result, config)
    record = {
        "start_id": start_id,
        "arm_id": arm_id,
        "initial_B_sha256": initial_hash,
        "final_A_sha256": array_sha256(final_result.A),
        "final_B_sha256": array_sha256(final_result.B),
        "final_S_sha256_normalized": array_sha256(final_result.S),
        "alpha": float(alpha),
        "target_kappa": float(target_kappa),
        "kappa_schedule": [float(value) for value in kappa_schedule],
        "B_only_continuation": bool(len(kappa_schedule) > 1),
        "A_and_S_reinitialized_each_stage": True,
        "allocated_iterations": int(sum(stage_budgets)),
        "actual_iterations": int(sum(stage["actual_iterations"] for stage in stages)),
        "solver_calls": len(stages),
        "runtime_seconds": float(total_runtime),
        "final_target_objective_reported": reported,
        "final_target_objective_recomputed": float(recomputed),
        "objective_consistency_absolute_error": float(consistency),
        "converged_at_target": bool(final_result.converged),
        "final_stationarity": (
            float(final_result.stationarity[-1])
            if final_result.stationarity
            else None
        ),
        "status": final_result.status,
        "nontrivial_solution_gate": nontrivial,
        "stages": stages,
    }
    return CandidateFit(
        start_id=start_id,
        arm_id=arm_id,
        result=final_result,
        record=record,
    )


def select_objective_winner_index(
    candidate_records: Sequence[Mapping[str, Any]],
) -> int:
    """Stable observed-objective-only selection with no silent filtering."""

    if not candidate_records:
        raise ValueError("candidate_records must not be empty")
    objectives: list[float] = []
    for index, record in enumerate(candidate_records):
        if "final_target_objective_recomputed" not in record:
            raise ValueError(f"candidate {index} lacks a recomputed target objective")
        value = float(record["final_target_objective_recomputed"])
        if not np.isfinite(value):
            raise ValueError(f"candidate {index} has a non-finite objective")
        objectives.append(value)
    # np.argmin returns the first occurrence, implementing the predeclared
    # start-order tie break without consulting any other diagnostic.
    return int(np.argmin(np.asarray(objectives, dtype=np.float64)))


def _run_multistart_arm(
    observed: ObservedPreparedCase,
    *,
    arm_id: str,
    initializations: Mapping[str, FloatArray],
    specs: Sequence[StartSpec],
    alpha: float,
    target_kappa: float,
    kappa_schedule: Sequence[float],
    stage_budgets: Sequence[int],
    config: RobustStudyConfig,
    retain_start_ids: Iterable[str] = (),
) -> tuple[SelectedArm, dict[str, CandidateFit]]:
    records: list[Mapping[str, Any]] = []
    best: CandidateFit | None = None
    best_index = -1
    retained: dict[str, CandidateFit] = {}
    retained_ids = set(retain_start_ids)
    for index, spec in enumerate(specs):
        candidate = _fit_candidate(
            observed,
            start_id=spec.start_id,
            arm_id=arm_id,
            initial_B=initializations[spec.start_id],
            alpha=alpha,
            target_kappa=target_kappa,
            kappa_schedule=kappa_schedule,
            stage_budgets=stage_budgets,
            config=config,
        )
        records.append(candidate.record)
        if spec.start_id in retained_ids:
            retained[spec.start_id] = candidate
        if best is None or float(
            candidate.record["final_target_objective_recomputed"]
        ) < float(best.record["final_target_objective_recomputed"]):
            best = candidate
            best_index = index
    selected_index = select_objective_winner_index(records)
    if best is None or selected_index != best_index:
        raise RuntimeError("streaming and canonical objective selectors disagree")
    selected_record = records[selected_index]
    arm_record = {
        "arm_id": arm_id,
        "selection_rule": (
            "minimum independently recomputed observed robust objective at common "
            "target kappa; ties use declared start order"
        ),
        "truth_inputs_available_to_selector": False,
        "convergence_used_for_selection": False,
        "runtime_used_for_selection": False,
        "nontrivial_gate_used_for_selection": False,
        "selected_index": int(selected_index),
        "selected_start_id": str(selected_record["start_id"]),
        "selected_fit_sha256": canonical_sha256(
            {
                "A": selected_record["final_A_sha256"],
                "B": selected_record["final_B_sha256"],
                "S": selected_record["final_S_sha256_normalized"],
            }
        ),
        "selected_objective": float(
            selected_record["final_target_objective_recomputed"]
        ),
        "selected_nontrivial_qualified": bool(
            selected_record["nontrivial_solution_gate"]["qualified"]
        ),
        "candidate_count": len(records),
        "all_candidates_retained_in_report": True,
    }
    return (
        SelectedArm(
            arm_id=arm_id,
            selected=best,
            candidate_records=tuple(records),
            selected_index=selected_index,
            record=arm_record,
        ),
        retained,
    )


def _candidate_raw_arrays(
    observed: ObservedPreparedCase,
    candidate: CandidateFit,
) -> tuple[FloatArray, FloatArray]:
    result = candidate.result
    normalized_modes = observed.normalized_observed @ result.B
    reconstruction = observed.normalization_sigma_max * (
        normalized_modes @ result.A.T
    )
    outliers = observed.normalization_sigma_max * result.S
    return (
        np.asarray(reconstruction, dtype=np.float64),
        np.asarray(outliers, dtype=np.float64),
    )


def _truth_metrics(
    dataset: RobustDecompositionDataset,
    reconstruction: FloatArray,
    outlier_scores: FloatArray,
    *,
    config: RobustStudyConfig,
) -> tuple[dict[str, Any], FloatArray]:
    frame_errors = normalized_full_frame_errors(dataset.clean, reconstruction)
    summary = summarize_full_frame_errors(frame_errors, dataset.time)
    nominal_budget = int(round(config.corruption_fraction * dataset.observed.size))
    detection = outlier_detection_metrics(
        dataset.corruption_support,
        scores=np.abs(outlier_scores),
        top_k=nominal_budget,
    )
    return (
        {
            "clean_relative_reconstruction_error": relative_reconstruction_error(
                dataset.clean, reconstruction
            ),
            "full_frame_normalized_rmse": summary,
            "catastrophic_frame_threshold": config.catastrophic_frame_threshold,
            "catastrophic_frame_count": int(
                np.count_nonzero(frame_errors > config.catastrophic_frame_threshold)
            ),
            "outlier_detection": {
                **detection,
                "budget_source": "round(protocol corruption_fraction * observed.size)",
                "budget_semantics": (
                    "nominal protocol-fraction budget; not the realized truth-support "
                    "count"
                ),
                "canonical_f1_metric_name": OUTLIER_NOMINAL_BUDGET_F1,
                "budget_top_k": nominal_budget,
                "nominal_budget_fraction": config.corruption_fraction,
                "realized_truth_fraction": float(
                    np.mean(dataset.corruption_support)
                ),
            },
        },
        frame_errors,
    )


def _development_parameter_fit(
    dataset: RobustDecompositionDataset,
    observed: ObservedPreparedCase,
    *,
    alpha: float,
    kappa: float,
    config: RobustStudyConfig,
) -> dict[str, Any]:
    candidate = _fit_candidate(
        observed,
        start_id="pca",
        arm_id="development_direct_pca_start",
        initial_B=observed.pca_B,
        alpha=alpha,
        target_kappa=kappa,
        kappa_schedule=(kappa,),
        stage_budgets=(config.max_iter_per_start,),
        config=config,
    )
    reconstruction, outliers = _candidate_raw_arrays(observed, candidate)
    # Development intentionally uses simulation truth. Detection ranking is
    # omitted from the grid because it is not part of parameter selection and
    # is disproportionately expensive at paper scale.
    frame_errors = normalized_full_frame_errors(dataset.clean, reconstruction)
    frame_summary = summarize_full_frame_errors(frame_errors, dataset.time)
    return {
        "alpha": float(alpha),
        "kappa": float(kappa),
        "candidate": candidate.record,
        "truth_metrics_used_for_development_only": {
            "clean_relative_reconstruction_error": (
                relative_reconstruction_error(dataset.clean, reconstruction)
            ),
            "full_frame_normalized_rmse": frame_summary,
            "catastrophic_frame_count": int(
                np.count_nonzero(frame_errors > config.catastrophic_frame_threshold)
            ),
        },
        "outlier_sha256_raw_units": array_sha256(outliers),
    }


def _run_development_seed_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _config_from_mapping(payload["config"])
    seed = int(payload["seed"])
    validate_study_config(config)
    dataset = generate_figure6_dataset(_paper_video_config(config, seed))
    observed = prepare_observed_case(dataset)
    legacy = _fit_legacy_dynamic_kappa_exact(observed, config)
    legacy_frame_errors = normalized_full_frame_errors(
        dataset.clean, legacy.reconstruction
    )
    pca_frame_errors = normalized_full_frame_errors(
        dataset.clean, observed.pca_reconstruction
    )
    historical_before_anchor = {
        "method_id": "legacy_dynamic_kappa",
        "solver": legacy.record,
        "truth_metrics_used_for_development_description_only": {
            "clean_relative_reconstruction_error": relative_reconstruction_error(
                dataset.clean, legacy.reconstruction
            ),
            "full_frame_normalized_rmse": summarize_full_frame_errors(
                legacy_frame_errors, dataset.time
            ),
            "catastrophic_frame_count": int(
                np.count_nonzero(
                    legacy_frame_errors > config.catastrophic_frame_threshold
                )
            ),
        },
        "parameter_selection_input": False,
    }
    pca_anchor = {
        "method_id": "pca",
        "truth_metrics_used_for_development_description_only": {
            "clean_relative_reconstruction_error": relative_reconstruction_error(
                dataset.clean, observed.pca_reconstruction
            ),
            "full_frame_normalized_rmse": summarize_full_frame_errors(
                pca_frame_errors, dataset.time
            ),
            "catastrophic_frame_count": int(
                np.count_nonzero(
                    pca_frame_errors > config.catastrophic_frame_threshold
                )
            ),
        },
        "parameter_selection_input": False,
    }
    records: list[dict[str, Any]] = []
    for alpha in config.alpha_grid:
        for kappa in config.kappa_grid:
            record = _development_parameter_fit(
                dataset,
                observed,
                alpha=alpha,
                kappa=kappa,
                config=config,
            )
            record["seed"] = seed
            records.append(record)
    return {
        "seed": seed,
        "observed_sha256": observed.data_sha256,
        "clean_sha256": array_sha256(dataset.clean),
        "corruption_sha256": array_sha256(dataset.corruption),
        "historical_before_anchor": historical_before_anchor,
        "pca_anchor": pca_anchor,
        "records": records,
    }


def select_development_parameters(
    seed_results: Sequence[Mapping[str, Any]],
    config: RobustStudyConfig,
) -> dict[str, Any]:
    """Select alpha/kappa using the predeclared development-truth ordering."""

    expected_pairs = [
        (float(alpha), float(kappa))
        for alpha in config.alpha_grid
        for kappa in config.kappa_grid
    ]
    grouped: dict[tuple[float, float], list[Mapping[str, Any]]] = {
        pair: [] for pair in expected_pairs
    }
    for seed_result in seed_results:
        for record in seed_result["records"]:
            key = (float(record["alpha"]), float(record["kappa"]))
            if key not in grouped:
                raise ValueError("development result contains an undeclared parameter")
            grouped[key].append(record)
    aggregates: list[dict[str, Any]] = []
    expected_seed_count = len(config.development_seeds)
    for order_index, pair in enumerate(expected_pairs):
        records = grouped[pair]
        if len(records) != expected_seed_count:
            raise ValueError("development parameter grid is not exact-cover")
        max_frames = [
            float(
                record["truth_metrics_used_for_development_only"]
                ["full_frame_normalized_rmse"]["max"]
            )
            for record in records
        ]
        clean_errors = [
            float(
                record["truth_metrics_used_for_development_only"]
                ["clean_relative_reconstruction_error"]
            )
            for record in records
        ]
        nontrivial = [
            bool(record["candidate"]["nontrivial_solution_gate"]["qualified"])
            for record in records
        ]
        aggregates.append(
            {
                "parameter_order_index": order_index,
                "alpha": pair[0],
                "kappa": pair[1],
                "all_development_nontrivial": bool(all(nontrivial)),
                "median_max_frame_normalized_rmse": float(np.median(max_frames)),
                "mean_max_frame_normalized_rmse": float(np.mean(max_frames)),
                "median_clean_relative_reconstruction_error": float(
                    np.median(clean_errors)
                ),
                "mean_clean_relative_reconstruction_error": float(
                    np.mean(clean_errors)
                ),
                "per_seed_max_frame_normalized_rmse": max_frames,
                "per_seed_clean_relative_reconstruction_error": clean_errors,
            }
        )
    ranked = sorted(
        aggregates,
        key=lambda record: (
            not bool(record["all_development_nontrivial"]),
            float(record["median_max_frame_normalized_rmse"]),
            float(record["median_clean_relative_reconstruction_error"]),
            int(record["parameter_order_index"]),
        ),
    )
    selected = ranked[0]
    if not bool(selected["all_development_nontrivial"]):
        raise RuntimeError(
            "no development alpha/kappa pair passed the predeclared "
            "nontrivial-solution gate on every development seed"
        )
    return {
        "selection_rule": (
            "require all dev fits nontrivial; minimize median per-seed maximum "
            "frame normalized RMSE; then median global clean relative error; "
            "then declared alpha-major/kappa-minor grid order"
        ),
        "truth_use": "simulation truth permitted only in development",
        "convergence_used_for_parameter_selection": False,
        "selected_alpha": float(selected["alpha"]),
        "selected_kappa": float(selected["kappa"]),
        "selected_parameter_order_index": int(selected["parameter_order_index"]),
        "aggregates": aggregates,
    }


_CONFIG_TUPLE_FIELDS = (
    "development_seeds",
    "confirmation_seeds",
    "alpha_grid",
    "kappa_grid",
    "continuation_kappa_multipliers",
    "continuation_stage_max_iters",
    "perturbation_scales",
    "perturbation_seeds",
)


def _config_from_mapping(value: Mapping[str, Any]) -> RobustStudyConfig:
    normalized = dict(value)
    for name in _CONFIG_TUPLE_FIELDS:
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    return RobustStudyConfig(**normalized)


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(repo_root: Path, arguments: Sequence[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def _git_text(repo_root: Path, arguments: Sequence[str]) -> str | None:
    value = _git_bytes(repo_root, arguments)
    return None if value is None else value.decode("utf-8", errors="replace").strip()


def _executed_source_provenance(repo_root: Path) -> dict[str, Any]:
    """Bind the runner, its imported reproduction helper, and eager rspca tree."""

    direct_roles = {
        "experiments/video_robust_decomposition.py": "robust_study_runner",
        "experiments/video_decomposition_reference.py": (
            "historical_video_generator_and_shared_metrics"
        ),
        "src/rspca/datasets.py": "hashing_and_runtime_provenance",
        "src/rspca/metrics.py": "truth_evaluation_after_selection",
        "src/rspca/robust.py": "existing_repository_robust_vp_solver",
        "src/rspca/vp.py": "solver_dependency_and_package_api",
    }
    allowed_experiment_modules = {
        "experiments.video_decomposition_reference",
        "experiments.video_robust_decomposition",
        "__main__",
    }
    by_path: dict[str, dict[str, Any]] = {}
    study_path = Path(__file__).resolve()
    reproduction_path = (
        repo_root / "experiments/video_decomposition_reference.py"
    ).resolve()
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            continue
        include = module_name == "rspca" or module_name.startswith("rspca.")
        include = include or (
            module_name in allowed_experiment_modules
            and path in {study_path, reproduction_path}
        )
        if not include or not path.is_file() or path.suffix != ".py":
            continue
        role = direct_roles.get(
            relative,
            (
                "package_initialization"
                if relative == "src/rspca/__init__.py"
                else "package_initialization_only"
            ),
        )
        entry = by_path.setdefault(
            relative,
            {"sha256": _file_sha256(path), "role": role, "module_names": []},
        )
        entry["module_names"].append(module_name)
    required = set(direct_roles) | {"src/rspca/__init__.py"}
    missing = sorted(required - set(by_path))
    if missing:
        raise RuntimeError(f"study source closure is incomplete: {missing}")
    for entry in by_path.values():
        entry["module_names"] = sorted(set(entry["module_names"]))
    legacy_verified: dict[str, bool | None] = {}
    for relative, expected in LEGACY_SOURCE_SHA256.items():
        content = _git_bytes(repo_root, ["show", f"{LEGACY_SOURCE_COMMIT}:{relative}"])
        legacy_verified[relative] = (
            None if content is None else sha256(content).hexdigest() == expected
        )
    robust_legacy_content = _git_bytes(
        repo_root, ["show", f"{LEGACY_SOURCE_COMMIT}:robust_spca_vp.py"]
    )
    legacy_verified["robust_spca_vp.py"] = (
        None
        if robust_legacy_content is None
        else sha256(robust_legacy_content).hexdigest()
        == LEGACY_ROBUST_SOURCE_SHA256
    )
    status = _git_text(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    loaded = {relative: by_path[relative] for relative in sorted(by_path)}
    source_hashes = {
        relative: record["sha256"] for relative, record in loaded.items()
    }
    return {
        "lineage": {
            "github_repository": "mafeifei798/Robust-and-Sparse-PCA",
            "historical_commit": LEGACY_SOURCE_COMMIT,
            "historical_files_sha256": dict(LEGACY_SOURCE_SHA256),
            "historical_robust_source_sha256": LEGACY_ROBUST_SOURCE_SHA256,
            "historical_files_verified_from_local_git": legacy_verified,
            "historical_orientation": "spatial_pixels_x_time_snapshots",
            "historical_spatial_mode_definition": "Phi = X @ B",
            "historical_generator": "spca_multiscale_video.py classic video",
            "historical_robust_solver": "robust_spca_vp.py",
            "paper": {
                "arxiv_id": ARXIV_ID,
                "arxiv_version": ARXIV_VERSION,
                "official_pdf_sha256": ARXIV_PDF_SHA256,
                "section": "6.1 Multiscale Video Example",
            },
            "current_repository_calls": {
                "exact_historical_before_port": (
                    "experiments.video_robust_decomposition."
                    "_fit_legacy_dynamic_kappa_exact"
                ),
                "development_and_confirmation_solver": (
                    "rspca.robust.fit_robust_vp_spca"
                ),
                "objective_recomputation": "rspca.robust.robust_objective",
                "generator": (
                    "experiments.video_decomposition_reference."
                    "generate_figure6_dataset"
                ),
            },
            "strategy_claim_boundary": (
                "multistart and B-only descending-kappa continuation are "
                "initialization/solve strategies around the existing repository "
                "algorithm, not new PCA algorithms"
            ),
        },
        "loaded_repo_modules": loaded,
        "source_sha256": source_hashes,
        "package_eager_imports_bound": True,
        "git_head": _git_text(repo_root, ["rev-parse", "HEAD"]),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "runtime_versions": runtime_versions(),
    }


def _without_runtime(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_runtime(item)
            for key, item in value.items()
            if str(key) not in {"runtime_seconds", "runtime_versions"}
        }
    if isinstance(value, (list, tuple)):
        return [_without_runtime(item) for item in value]
    return _json_safe(value)


def _study_protocol_body(config: RobustStudyConfig) -> dict[str, Any]:
    return {
        "protocol_version": STUDY_PROTOCOL_VERSION,
        "config": asdict(config),
        "start_specs": [asdict(spec) for spec in default_start_specs(config)],
        "phase_boundary": {
            "development_truth_use": "alpha/kappa selection only",
            "confirmation_tuning": False,
            "confirmation_seed_bank_untouched_when_lock_created": True,
        },
        "within_arm_selection": {
            "input": "observed normalized X and final target robust objective only",
            "tie_break": "declared start order",
            "truth_used": False,
            "convergence_used": False,
            "nontrivial_gate_used": False,
        },
        "continuation": {
            "state_handoff": "B only",
            "A_and_S_reinitialized_each_stage": True,
            "intermediate_objectives_used_for_winner_selection": False,
            "direct_and_continuation_equal_allocated_iterations_per_start": True,
        },
        "comparators": [
            "PCA",
            "origin_main_legacy_dynamic_kappa_changing_objective",
            "locked_single_start",
            "direct_multistart_equal_budget",
            "B_only_descending_kappa_multistart",
        ],
        "historical_v1_anchor": {
            "alpha": 0.012,
            "kappa": 0.005,
            "max_iter": 500,
            "corruption_seed": 123,
            "role": "viewed development anchor; not rerun as confirmation",
        },
        "exact_origin_main_before": {
            "source_commit": LEGACY_SOURCE_COMMIT,
            "source_path": "robust_spca_vp.py",
            "source_sha256": LEGACY_ROBUST_SOURCE_SHA256,
            "alpha": config.legacy_dynamic_alpha,
            "kappa_policy": (
                "per-iteration residual MAD times huber_tau, clipped to "
                "[legacy_kappa_min, legacy_kappa_max]"
            ),
            "stopping": "B-change Frobenius norm",
            "changing_objective": True,
            "eligible_for_fixed_target_objective_selection": False,
            "role": "same-seed historical before baseline",
        },
    }


def _parallel_map_by_seed(
    worker: Any,
    payloads: Sequence[Mapping[str, Any]],
    *,
    workers: int,
) -> list[Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("workers must be a positive integer")
    if workers == 1:
        return [worker(payload) for payload in payloads]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(worker, payloads))


def _thread_environment(workers: int) -> dict[str, Any]:
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    values = {name: os.environ.get(name) for name in names}
    if workers > 1:
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise RuntimeError(
                "parallel seed execution requires explicit thread environment: "
                f"{missing}"
            )
        for name, value in values.items():
            try:
                integer = int(str(value))
            except ValueError as error:
                raise RuntimeError(f"{name} must be a positive integer") from error
            if integer <= 0:
                raise RuntimeError(f"{name} must be a positive integer")
    return {"workers": workers, "thread_environment": values}


def run_development(
    config: RobustStudyConfig,
    *,
    workers: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the exact development grid and return report plus frozen lock."""

    validate_study_config(config)
    execution = _thread_environment(workers)
    payloads = [
        {"config": asdict(config), "seed": int(seed)}
        for seed in config.development_seeds
    ]
    started = perf_counter()
    seed_results = _parallel_map_by_seed(
        _run_development_seed_worker, payloads, workers=workers
    )
    runtime = perf_counter() - started
    by_seed = {int(result["seed"]): result for result in seed_results}
    if set(by_seed) != set(config.development_seeds) or len(by_seed) != len(
        config.development_seeds
    ):
        raise RuntimeError("development seed shards do not form an exact cover")
    ordered = [by_seed[int(seed)] for seed in config.development_seeds]
    expected_grid_size = len(config.alpha_grid) * len(config.kappa_grid)
    if any(len(result["records"]) != expected_grid_size for result in ordered):
        raise RuntimeError("development parameter shards do not cover the exact grid")
    selection = select_development_parameters(ordered, config)
    repo_root = Path(__file__).resolve().parents[1]
    provenance = _executed_source_provenance(repo_root)
    protocol = _study_protocol_body(config)
    protocol_sha256 = canonical_sha256(protocol)
    evidence_projection = _without_runtime(
        {"seed_results": ordered, "selection": selection}
    )
    evidence_sha256 = canonical_sha256(evidence_projection)
    lock_body: dict[str, Any] = {
        "schema_version": 1,
        "lock_kind": "paper_video_robust_development_selection_lock",
        "study_protocol": protocol,
        "study_protocol_sha256": protocol_sha256,
        "source_sha256": provenance["source_sha256"],
        "development_evidence_sha256_without_runtime": evidence_sha256,
        "development_seed_manifest_sha256": canonical_sha256(
            {"seeds": list(config.development_seeds)}
        ),
        "confirmation_seed_manifest_sha256": canonical_sha256(
            {"seeds": list(config.confirmation_seeds)}
        ),
        "selected_parameters": {
            "alpha": selection["selected_alpha"],
            "kappa": selection["selected_kappa"],
        },
        "confirmation_parameter_overrides_allowed": False,
        "confirmation_truth_used_for_tuning": False,
        "paper_scale": bool(
            config.grid_size == 200
            and config.n_timesteps == 300
            and config.total_time == 150.0
        ),
    }
    lock_body["selection_lock_sha256"] = canonical_sha256(lock_body)
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "development",
        "paper_scale": lock_body["paper_scale"],
        "study_protocol": protocol,
        "study_protocol_sha256": protocol_sha256,
        "selection": selection,
        "seed_results": ordered,
        "selection_lock_sha256": lock_body["selection_lock_sha256"],
        "development_evidence_sha256_without_runtime": evidence_sha256,
        "provenance": provenance,
        "execution": {**execution, "runtime_seconds": float(runtime)},
        "scientific_boundary": (
            "development truth selects alpha/kappa for this synthetic DGP; it "
            "is not confirmation evidence"
        ),
    }
    report["report_body_sha256"] = canonical_sha256(report)
    return _json_safe(report), _json_safe(lock_body)


def _claim_new_output_directory(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing path: {destination}") from error
    return destination


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(_json_safe(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_exact_manifest(
    directory: Path,
    *,
    manifest_name: str,
    manifest_kind: str,
    required_payloads: frozenset[str],
) -> Path:
    actual = {path.name for path in directory.iterdir()}
    if actual != required_payloads:
        raise RuntimeError("payload set is incomplete before manifest creation")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": manifest_kind,
        "self_excluding": True,
        "required_payloads": sorted(required_payloads),
        "files": {
            name: {
                "sha256": _file_sha256(directory / name),
                "size_bytes": int((directory / name).stat().st_size),
            }
            for name in sorted(required_payloads)
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    path = directory / manifest_name
    _write_json_exclusive(path, manifest)
    return path


def _write_development_grid_csv(report: Mapping[str, Any], path: Path) -> None:
    fields = [
        "seed",
        "alpha",
        "kappa",
        "clean_relative_reconstruction_error",
        "frame_median",
        "frame_p95",
        "frame_max",
        "worst_frame_index",
        "final_target_objective",
        "converged",
        "final_stationarity",
        "nontrivial_qualified",
        "runtime_seconds",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed_result in report["seed_results"]:
            for record in seed_result["records"]:
                truth = record["truth_metrics_used_for_development_only"]
                frame = truth["full_frame_normalized_rmse"]
                candidate = record["candidate"]
                writer.writerow(
                    {
                        "seed": record["seed"],
                        "alpha": f"{record['alpha']:.17g}",
                        "kappa": f"{record['kappa']:.17g}",
                        "clean_relative_reconstruction_error": (
                            f"{truth['clean_relative_reconstruction_error']:.17g}"
                        ),
                        "frame_median": f"{frame['median']:.17g}",
                        "frame_p95": f"{frame['p95']:.17g}",
                        "frame_max": f"{frame['max']:.17g}",
                        "worst_frame_index": frame["worst_frame_index"],
                        "final_target_objective": (
                            f"{candidate['final_target_objective_recomputed']:.17g}"
                        ),
                        "converged": candidate["converged_at_target"],
                        "final_stationarity": (
                            f"{candidate['final_stationarity']:.17g}"
                        ),
                        "nontrivial_qualified": candidate[
                            "nontrivial_solution_gate"
                        ]["qualified"],
                        "runtime_seconds": f"{candidate['runtime_seconds']:.17g}",
                    }
                )


def save_development_output(
    report: Mapping[str, Any],
    selection_lock: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = _claim_new_output_directory(output_dir)
    paths = {
        "report": destination / DEVELOPMENT_RESULTS_NAME,
        "grid": destination / DEVELOPMENT_GRID_NAME,
        "lock": destination / SELECTION_LOCK_NAME,
        "manifest": destination / DEVELOPMENT_MANIFEST_NAME,
    }
    _write_json_exclusive(paths["report"], report)
    _write_development_grid_csv(report, paths["grid"])
    _write_json_exclusive(paths["lock"], selection_lock)
    paths["manifest"] = _write_exact_manifest(
        destination,
        manifest_name=DEVELOPMENT_MANIFEST_NAME,
        manifest_kind="paper_video_robust_development_exact_closure",
        required_payloads=_DEVELOPMENT_REQUIRED_PAYLOADS,
    )
    return paths


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _validate_exact_output_directory(
    directory: str | Path,
    *,
    required_outputs: frozenset[str],
    required_payloads: frozenset[str],
    manifest_name: str,
) -> dict[str, Any]:
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("output path must be a real directory")
    entries = list(root.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"output contains non-regular entry: {entry.name}")
    if {entry.name for entry in entries} != required_outputs:
        raise ValueError("output directory does not match its fixed file contract")
    manifest = _strict_json(root / manifest_name)
    if set(manifest.get("required_payloads", [])) != required_payloads:
        raise ValueError("manifest required payload contract mismatch")
    if set(manifest.get("files", {})) != required_payloads:
        raise ValueError("manifest file set mismatch")
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("manifest canonical digest mismatch")
    for name, record in manifest["files"].items():
        path = root / name
        if _file_sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch for {name}")
    return manifest


def validate_development_output_directory(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = _validate_exact_output_directory(
        root,
        required_outputs=_DEVELOPMENT_REQUIRED_OUTPUTS,
        required_payloads=_DEVELOPMENT_REQUIRED_PAYLOADS,
        manifest_name=DEVELOPMENT_MANIFEST_NAME,
    )
    lock = load_selection_lock(root / SELECTION_LOCK_NAME, verify_source=False)
    report = _strict_json(root / DEVELOPMENT_RESULTS_NAME)
    report_body = dict(report)
    digest = report_body.pop("report_body_sha256", None)
    if digest != canonical_sha256(report_body):
        raise ValueError("development report digest mismatch")
    if report["selection_lock_sha256"] != lock["selection_lock_sha256"]:
        raise ValueError("development report and lock disagree")
    if report.get("phase") != "development":
        raise ValueError("development report phase mismatch")
    config = _config_from_mapping(lock["study_protocol"]["config"])
    seed_results = report.get("seed_results")
    if not isinstance(seed_results, list) or [
        int(result["seed"]) for result in seed_results
    ] != list(config.development_seeds):
        raise ValueError("development report seed order/cover mismatch")
    expected_pairs = [
        (float(alpha), float(kappa))
        for alpha in config.alpha_grid
        for kappa in config.kappa_grid
    ]
    for result in seed_results:
        records = result.get("records")
        if not isinstance(records, list) or [
            (float(record["alpha"]), float(record["kappa"]))
            for record in records
        ] != expected_pairs:
            raise ValueError("development report parameter grid mismatch")
    evidence = canonical_sha256(
        _without_runtime(
            {"seed_results": seed_results, "selection": report["selection"]}
        )
    )
    if evidence != report.get("development_evidence_sha256_without_runtime"):
        raise ValueError("development report evidence digest mismatch")
    if evidence != lock.get("development_evidence_sha256_without_runtime"):
        raise ValueError("development report evidence does not match lock")
    if report.get("study_protocol") != lock.get("study_protocol"):
        raise ValueError("development report and lock protocol mismatch")
    selected = report["selection"]
    recomputed_selection = select_development_parameters(seed_results, config)
    if canonical_sha256(selected) != canonical_sha256(recomputed_selection):
        raise ValueError(
            "development selection is not the implemented ranking of seed results"
        )
    if {
        "alpha": selected["selected_alpha"],
        "kappa": selected["selected_kappa"],
    } != lock["selected_parameters"]:
        raise ValueError("development selection and lock parameters mismatch")
    if report.get("provenance", {}).get("source_sha256") != lock.get(
        "source_sha256"
    ):
        raise ValueError("development source closure and lock mismatch")
    return manifest


def load_selection_lock(
    path: str | Path,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    lock = _strict_json(Path(path))
    body = dict(lock)
    digest = body.pop("selection_lock_sha256", None)
    if digest != canonical_sha256(body):
        raise ValueError("selection lock canonical digest mismatch")
    protocol = lock.get("study_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("selection lock lacks a study protocol")
    if lock.get("study_protocol_sha256") != canonical_sha256(protocol):
        raise ValueError("selection lock protocol digest mismatch")
    config = _config_from_mapping(protocol["config"])
    validate_study_config(config)
    if protocol != _json_safe(_study_protocol_body(config)):
        raise ValueError("selection lock protocol does not match the implemented contract")
    if lock.get("schema_version") != 1 or lock.get("lock_kind") != (
        "paper_video_robust_development_selection_lock"
    ):
        raise ValueError("selection lock schema or kind mismatch")
    if lock.get("development_seed_manifest_sha256") != canonical_sha256(
        {"seeds": list(config.development_seeds)}
    ):
        raise ValueError("development seed manifest digest mismatch")
    if lock.get("confirmation_seed_manifest_sha256") != canonical_sha256(
        {"seeds": list(config.confirmation_seeds)}
    ):
        raise ValueError("confirmation seed manifest digest mismatch")
    selected = lock.get("selected_parameters")
    if not isinstance(selected, Mapping):
        raise ValueError("selection lock lacks selected parameters")
    if float(selected["alpha"]) not in config.alpha_grid or float(
        selected["kappa"]
    ) not in config.kappa_grid:
        raise ValueError("selection lock parameters are outside the development grid")
    if lock.get("confirmation_parameter_overrides_allowed") is not False:
        raise ValueError("selection lock must forbid confirmation overrides")
    expected_paper_scale = bool(
        config.grid_size == 200
        and config.n_timesteps == 300
        and config.total_time == 150.0
    )
    if lock.get("paper_scale") is not expected_paper_scale:
        raise ValueError("selection lock paper-scale declaration mismatch")
    if verify_source:
        repo_root = Path(__file__).resolve().parents[1]
        current = _executed_source_provenance(repo_root)["source_sha256"]
        if current != lock.get("source_sha256"):
            raise ValueError("selection lock source closure mismatch")
    return lock


def _candidate_fit_sha256(candidate: CandidateFit) -> str:
    record = candidate.record
    return canonical_sha256(
        {
            "A": record["final_A_sha256"],
            "B": record["final_B_sha256"],
            "S": record["final_S_sha256_normalized"],
        }
    )


def _fit_locked_observed_arms(
    observed: ObservedPreparedCase,
    *,
    alpha: float,
    target_kappa: float,
    config: RobustStudyConfig,
) -> dict[str, Any]:
    """Fit locked arms using only observed data and fixed RNG start specs."""

    specs = default_start_specs(config)
    initializations = build_initializations(observed, specs)
    direct, retained = _run_multistart_arm(
        observed,
        arm_id="direct_multistart_equal_budget",
        initializations=initializations,
        specs=specs,
        alpha=alpha,
        target_kappa=target_kappa,
        kappa_schedule=(target_kappa,),
        stage_budgets=(config.max_iter_per_start,),
        config=config,
        retain_start_ids=("pca",),
    )
    if "pca" not in retained:
        raise RuntimeError("direct arm did not retain its locked PCA start")
    continuation_schedule = tuple(
        float(target_kappa * multiplier)
        for multiplier in config.continuation_kappa_multipliers
    )
    if continuation_schedule[-1] != float(target_kappa):
        raise RuntimeError("floating-point continuation target changed")
    continuation, _ = _run_multistart_arm(
        observed,
        arm_id="B_only_descending_kappa_multistart",
        initializations=initializations,
        specs=specs,
        alpha=alpha,
        target_kappa=target_kappa,
        kappa_schedule=continuation_schedule,
        stage_budgets=config.continuation_stage_max_iters,
        config=config,
    )
    direct_initial_hashes = [
        str(record["initial_B_sha256"]) for record in direct.candidate_records
    ]
    continuation_initial_hashes = [
        str(record["initial_B_sha256"])
        for record in continuation.candidate_records
    ]
    declared_hashes = [
        array_sha256(initializations[spec.start_id]) for spec in specs
    ]
    if direct_initial_hashes != declared_hashes:
        raise RuntimeError("direct arm initializations changed from declared starts")
    if continuation_initial_hashes != declared_hashes:
        raise RuntimeError(
            "continuation arm initializations changed from declared starts"
        )
    direct_budgets = {
        int(record["allocated_iterations"]) for record in direct.candidate_records
    }
    continuation_budgets = {
        int(record["allocated_iterations"])
        for record in continuation.candidate_records
    }
    if direct_budgets != {config.max_iter_per_start} or continuation_budgets != {
        config.max_iter_per_start
    }:
        raise RuntimeError("direct and continuation allocated budgets differ")
    return {
        "specs": specs,
        "initial_B_sha256_by_start": {
            spec.start_id: declared_hashes[index]
            for index, spec in enumerate(specs)
        },
        "single_start": retained["pca"],
        "direct": direct,
        "continuation": continuation,
        "fairness": {
            "start_ids": [spec.start_id for spec in specs],
            "direct_initial_B_sha256": direct_initial_hashes,
            "continuation_initial_B_sha256": continuation_initial_hashes,
            "initializations_byte_for_byte_paired": True,
            "allocated_iterations_per_start": config.max_iter_per_start,
            "direct_solver_calls_per_start": 1,
            "continuation_solver_calls_per_start": len(continuation_schedule),
            "continuation_schedule": list(continuation_schedule),
            "B_only_handoff": True,
            "A_and_S_reinitialized_each_stage": True,
        },
    }


def _method_solver_summary(candidate: CandidateFit) -> dict[str, Any]:
    record = candidate.record
    return {
        "candidate_fit_sha256": _candidate_fit_sha256(candidate),
        "source_arm_id": candidate.arm_id,
        "start_id": candidate.start_id,
        "converged": bool(record["converged_at_target"]),
        "status": record["status"],
        "final_stationarity": record["final_stationarity"],
        "final_target_objective_recomputed": record[
            "final_target_objective_recomputed"
        ],
        "nontrivial_solution_gate": record["nontrivial_solution_gate"],
        "allocated_iterations": record["allocated_iterations"],
        "actual_iterations": record["actual_iterations"],
        "solver_calls": record["solver_calls"],
        "runtime_seconds": record["runtime_seconds"],
        "runtime_scope": (
            "selected objective-winning fit only; use the method summary's "
            "strategy_candidate_fit_cost for multistart search cost"
        ),
    }


def _run_confirmation_seed_worker(
    payload: Mapping[str, Any],
) -> ConfirmationCaseInternal:
    """Run one locked confirmation seed; truth enters only after fit hashes freeze."""

    config = _config_from_mapping(payload["config"])
    validate_study_config(config)
    seed = int(payload["seed"])
    if seed not in config.confirmation_seeds:
        raise ValueError("worker seed is not in the locked confirmation bank")
    alpha = float(payload["alpha"])
    target_kappa = float(payload["kappa"])
    if alpha not in config.alpha_grid or target_kappa not in config.kappa_grid:
        raise ValueError("worker parameters are outside the locked development grid")

    dataset = generate_figure6_dataset(_paper_video_config(config, seed))
    observed = prepare_observed_case(dataset)
    # This function accepts only ObservedPreparedCase.  The truth arrays remain
    # outside the fitting and objective-only selection boundary.
    fitted = _fit_locked_observed_arms(
        observed,
        alpha=alpha,
        target_kappa=target_kappa,
        config=config,
    )
    legacy = _fit_legacy_dynamic_kappa_exact(observed, config)
    single: CandidateFit = fitted["single_start"]
    direct: SelectedArm = fitted["direct"]
    continuation: SelectedArm = fitted["continuation"]

    pca_reconstruction = observed.pca_reconstruction
    pca_outliers = observed.observed - pca_reconstruction
    robust_candidates = {
        "single_start": single,
        "direct_multistart": direct.selected,
        "continuation_multistart": continuation.selected,
    }
    arrays_by_fit: dict[str, tuple[FloatArray, FloatArray]] = {}
    method_arrays: dict[str, tuple[FloatArray, FloatArray]] = {
        "pca": (pca_reconstruction, pca_outliers),
        "legacy_dynamic_kappa": (legacy.reconstruction, legacy.outliers),
    }
    for method, candidate in robust_candidates.items():
        fit_sha = _candidate_fit_sha256(candidate)
        if fit_sha not in arrays_by_fit:
            arrays_by_fit[fit_sha] = _candidate_raw_arrays(observed, candidate)
        method_arrays[method] = arrays_by_fit[fit_sha]

    metrics_by_array_hash: dict[str, tuple[dict[str, Any], FloatArray]] = {}
    method_records: dict[str, dict[str, Any]] = {}
    method_frame_errors: dict[str, FloatArray] = {}
    method_labels = {
        "pca": "Corrupted-PCA",
        "legacy_dynamic_kappa": "Origin/main legacy dynamic-kappa Robust",
        "single_start": "Locked single-start (PCA init)",
        "direct_multistart": "Direct multistart (equal budget)",
        "continuation_multistart": "B-only continuation multistart",
    }
    for method in _CONFIRMATION_METHOD_ORDER:
        reconstruction, outliers = method_arrays[method]
        evaluation_key = canonical_sha256(
            {
                "reconstruction": array_sha256(reconstruction),
                "outliers": array_sha256(outliers),
            }
        )
        if evaluation_key not in metrics_by_array_hash:
            metrics_by_array_hash[evaluation_key] = _truth_metrics(
                dataset,
                reconstruction,
                outliers,
                config=config,
            )
        truth_metrics, frame_errors = metrics_by_array_hash[evaluation_key]
        method_frame_errors[method] = frame_errors.copy()
        if method == "pca":
            solver = {
                "candidate_fit_sha256": canonical_sha256(
                    {
                        "pca_B_sha256": observed.pca_B_sha256,
                        "reconstruction_sha256": array_sha256(reconstruction),
                    }
                ),
                "source_arm_id": "closed_form_corrupted_pca",
                "start_id": None,
                "converged": True,
                "status": "closed-form truncated SVD",
                "final_stationarity": None,
                "final_target_objective_recomputed": None,
                "nontrivial_solution_gate": {
                    "qualified": True,
                    "not_applicable_to_closed_form_pca": True,
                },
                "allocated_iterations": 1,
                "actual_iterations": 1,
                "solver_calls": 1,
                "runtime_seconds": observed.preparation_runtime_seconds,
                "outlier_score_definition": "observed minus PCA reconstruction",
            }
        elif method == "legacy_dynamic_kappa":
            solver = {
                "candidate_fit_sha256": legacy.record["fit_sha256"],
                "source_arm_id": "origin_main_legacy_dynamic_kappa",
                "start_id": "historical_internal_pca",
                "converged": legacy.record["converged"],
                "status": legacy.record["status"],
                "final_stationarity": None,
                "final_target_objective_recomputed": None,
                "final_diagnostic_objective_at_final_dynamic_kappa": (
                    legacy.record[
                        "final_diagnostic_objective_at_final_kappa"
                    ]
                ),
                "changing_objective": True,
                "eligible_for_fixed_target_objective_selection": False,
                "nontrivial_solution_gate": legacy.record[
                    "nontrivial_solution_gate"
                ],
                "allocated_iterations": legacy.record["allocated_iterations"],
                "actual_iterations": legacy.record["actual_iterations"],
                "solver_calls": 1,
                "runtime_seconds": legacy.record["runtime_seconds"],
                "historical_record": legacy.record,
            }
        else:
            solver = _method_solver_summary(robust_candidates[method])
        method_records[method] = {
            "method_id": method,
            "label": method_labels[method],
            "reconstruction_sha256": array_sha256(reconstruction),
            "outlier_scores_sha256_raw_units": array_sha256(outliers),
            "truth_metrics_after_fit_selection_lock": truth_metrics,
            "solver": solver,
        }

    snapshot_indices = {
        config.representative_frame_index,
        int(
            method_records["legacy_dynamic_kappa"]
            ["truth_metrics_after_fit_selection_lock"]
            ["full_frame_normalized_rmse"]["worst_frame_index"]
        ),
        int(
            method_records["single_start"]
            ["truth_metrics_after_fit_selection_lock"]
            ["full_frame_normalized_rmse"]["worst_frame_index"]
        ),
        int(
            method_records["continuation_multistart"]
            ["truth_metrics_after_fit_selection_lock"]
            ["full_frame_normalized_rmse"]["worst_frame_index"]
        ),
    }
    snapshots: dict[int, dict[str, FloatArray]] = {}
    for frame_index in sorted(snapshot_indices):
        clean_frame = np.asarray(dataset.clean[:, frame_index], dtype=np.float64)
        snapshot: dict[str, FloatArray] = {
            "clean": clean_frame.copy(),
            "observed": np.asarray(
                dataset.observed[:, frame_index], dtype=np.float64
            ).copy(),
            "true_corruption": np.asarray(
                dataset.corruption[:, frame_index], dtype=np.float64
            ).copy(),
        }
        for method, (reconstruction, outliers) in method_arrays.items():
            reconstruction_frame = np.asarray(
                reconstruction[:, frame_index], dtype=np.float64
            )
            snapshot[f"{method}_reconstruction"] = reconstruction_frame.copy()
            snapshot[f"{method}_outliers"] = np.asarray(
                outliers[:, frame_index], dtype=np.float64
            ).copy()
            snapshot[f"{method}_abs_error"] = np.abs(
                reconstruction_frame - clean_frame
            )
        snapshots[frame_index] = snapshot

    record: dict[str, Any] = {
        "seed": seed,
        "data": {
            "orientation": "spatial_pixels_x_time_snapshots",
            "observed_sha256": observed.data_sha256,
            "clean_sha256": array_sha256(dataset.clean),
            "corruption_sha256": array_sha256(dataset.corruption),
            "corruption_support_sha256": array_sha256(
                dataset.corruption_support
            ),
            "normalization_sigma_max": observed.normalization_sigma_max,
            "normalization_policy": "divide observed X by its largest singular value",
        },
        "selected_parameters": {"alpha": alpha, "kappa": target_kappa},
        "start_specs": [asdict(spec) for spec in fitted["specs"]],
        "initial_B_sha256_by_start": fitted["initial_B_sha256_by_start"],
        "fairness": fitted["fairness"],
        "arms": {
            "direct_multistart_equal_budget": {
                **direct.record,
                "candidate_records": list(direct.candidate_records),
            },
            "B_only_descending_kappa_multistart": {
                **continuation.record,
                "candidate_records": list(continuation.candidate_records),
            },
        },
        "methods": method_records,
        "truth_isolation": {
            "fit_function_input": "ObservedPreparedCase only",
            "selected_fit_hashes_frozen_before_truth_metrics": True,
            "clean_truth_used_for_within_arm_selection": False,
            "support_used_for_within_arm_selection": False,
            "historical_before_baseline_used_for_fixed_arm_selection": False,
        },
    }
    record["case_body_sha256_without_runtime"] = canonical_sha256(
        _without_runtime(record)
    )
    return ConfirmationCaseInternal(
        seed=seed,
        record=_json_safe(record),
        frame_errors=method_frame_errors,
        snapshots=snapshots,
    )


_CONFIRMATION_METHOD_ORDER = (
    "pca",
    "legacy_dynamic_kappa",
    "single_start",
    "direct_multistart",
    "continuation_multistart",
)

_CONFIRMATION_METRICS: tuple[tuple[str, str], ...] = (
    ("clean_relative_reconstruction_error", "lower"),
    ("frame_median", "lower"),
    ("frame_p95", "lower"),
    ("frame_max", "lower"),
    ("catastrophic_frame_count", "lower"),
    ("outlier_roc_auc", "higher"),
    ("outlier_average_precision", "higher"),
    (OUTLIER_NOMINAL_BUDGET_F1, "higher"),
)


def _case_metric(case: ConfirmationCaseInternal, method: str, metric: str) -> float:
    truth = case.record["methods"][method][
        "truth_metrics_after_fit_selection_lock"
    ]
    if metric == "clean_relative_reconstruction_error":
        return float(truth[metric])
    if metric == "catastrophic_frame_count":
        return float(truth[metric])
    if metric.startswith("frame_"):
        return float(truth["full_frame_normalized_rmse"][metric.removeprefix("frame_")])
    detection_key = {
        "outlier_roc_auc": "roc_auc",
        "outlier_average_precision": "average_precision",
        OUTLIER_NOMINAL_BUDGET_F1: "f1",
        LEGACY_OUTLIER_TRUE_BUDGET_F1: "f1",
    }[metric]
    return float(truth["outlier_detection"][detection_key])


def _sample_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be a finite nonempty vector")
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "values_in_locked_seed_order": [float(value) for value in array],
    }


def _strategy_candidate_fit_cost(
    case_record: Mapping[str, Any], method: str
) -> dict[str, Any]:
    """Account for every recorded candidate fit used by one strategy.

    The version-1 report's flat ``runtime_seconds`` field refers only to the
    selected objective winner.  For multistart methods that omits the other
    candidates which had to be fitted before selection.  This helper derives a
    backward-compatible strategy cost directly from the retained case records.
    Robust multistart cost excludes common SVD preparation; the PCA branch
    records that preparation call itself.  All branches exclude process
    scheduling, plotting, and serialization, so none is end-to-end wall-clock
    time.
    """

    if method not in _CONFIRMATION_METHOD_ORDER:
        raise ValueError(f"unknown confirmation method: {method}")
    if method == "direct_multistart":
        candidates = case_record["arms"]["direct_multistart_equal_budget"][
            "candidate_records"
        ]
    elif method == "continuation_multistart":
        candidates = case_record["arms"]["B_only_descending_kappa_multistart"][
            "candidate_records"
        ]
    else:
        solver = case_record["methods"][method]["solver"]
        if method == "pca":
            cost_scope = (
                "recorded truncated-SVD/PCA preparation call; excludes process "
                "scheduling, plotting, and serialization"
            )
        else:
            cost_scope = (
                "recorded solver calls for the evaluated strategy; excludes common "
                "preparation, scheduling, selection, plotting, and serialization"
            )
        return {
            "cost_scope": cost_scope,
            "evaluated_candidate_count": 1,
            "candidate_fit_runtime_seconds": float(solver["runtime_seconds"]),
            "allocated_iterations": int(solver["allocated_iterations"]),
            "actual_iterations": int(solver["actual_iterations"]),
            "solver_calls": int(solver["solver_calls"]),
        }
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("multistart strategy lacks retained candidate records")
    return {
        "cost_scope": (
            "sum of every retained candidate solver call used before objective "
            "selection; excludes common preparation, scheduling, selection, "
            "plotting, and serialization"
        ),
        "evaluated_candidate_count": len(candidates),
        "candidate_fit_runtime_seconds": float(
            sum(float(record["runtime_seconds"]) for record in candidates)
        ),
        "allocated_iterations": int(
            sum(int(record["allocated_iterations"]) for record in candidates)
        ),
        "actual_iterations": int(
            sum(int(record["actual_iterations"]) for record in candidates)
        ),
        "solver_calls": int(sum(int(record["solver_calls"]) for record in candidates)),
    }


def _normalize_legacy_budget_metric_keys(value: Any) -> Any:
    """Map version-1 misnomers to the canonical nominal-budget vocabulary."""

    key_map = {
        LEGACY_OUTLIER_TRUE_BUDGET_F1: OUTLIER_NOMINAL_BUDGET_F1,
        "mean_true_budget_f1_difference": "mean_nominal_budget_f1_difference",
        "maximum_allowed_true_budget_f1_decrease": (
            "maximum_allowed_nominal_budget_f1_decrease"
        ),
        "true_budget_f1_noninferiority_pass": (
            "nominal_budget_f1_noninferiority_pass"
        ),
    }
    if isinstance(value, Mapping):
        return {
            key_map.get(str(key), str(key)): _normalize_legacy_budget_metric_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_normalize_legacy_budget_metric_keys(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_legacy_budget_metric_keys(item) for item in value]
    return value


def _confirmation_metric_semantics() -> dict[str, str]:
    return {
        OUTLIER_NOMINAL_BUDGET_F1: (
            "F1 after selecting round(protocol corruption_fraction * observed.size) "
            "largest absolute outlier scores; this is a nominal budget, not the "
            "realized support count"
        ),
        "legacy_version_1_field_alias": LEGACY_OUTLIER_TRUE_BUDGET_F1,
        "runtime": (
            "multistart cost sums every retained candidate solver call and excludes "
            "common preparation and orchestration overhead"
        ),
    }


def _bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(differences, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("paired differences must be finite and nonempty")
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, array.size, size=(draws, array.size))
    means = np.mean(array[samples], axis=1)
    lower, upper = np.percentile(means, [2.5, 97.5])
    return {
        "method": "paired percentile bootstrap of the mean",
        "confidence_level": 0.95,
        "draws": int(draws),
        "seed": int(seed),
        "lower": float(lower),
        "upper": float(upper),
    }


def _aggregate_confirmation(
    cases: Sequence[ConfirmationCaseInternal],
    config: RobustStudyConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    method_summaries: dict[str, Any] = {}
    for method in _CONFIRMATION_METHOD_ORDER:
        metric_summaries = {
            metric: _sample_summary(
                [_case_metric(case, method, metric) for case in cases]
            )
            for metric, _ in _CONFIRMATION_METRICS
        }
        solvers = [case.record["methods"][method]["solver"] for case in cases]
        strategy_costs = [
            _strategy_candidate_fit_cost(case.record, method) for case in cases
        ]
        method_summaries[method] = {
            "metrics": metric_summaries,
            "convergence_fraction": float(
                np.mean([bool(solver["converged"]) for solver in solvers])
            ),
            "nontrivial_qualification_fraction": float(
                np.mean(
                    [
                        bool(solver["nontrivial_solution_gate"]["qualified"])
                        for solver in solvers
                    ]
                )
            ),
            "strategy_candidate_fit_cost": {
                "scope": strategy_costs[0]["cost_scope"],
                "total_runtime_seconds": float(
                    sum(
                        float(cost["candidate_fit_runtime_seconds"])
                        for cost in strategy_costs
                    )
                ),
                "total_allocated_iterations": int(
                    sum(int(cost["allocated_iterations"]) for cost in strategy_costs)
                ),
                "total_actual_iterations": int(
                    sum(int(cost["actual_iterations"]) for cost in strategy_costs)
                ),
                "total_solver_calls": int(
                    sum(int(cost["solver_calls"]) for cost in strategy_costs)
                ),
                "evaluated_candidate_count_per_seed": [
                    int(cost["evaluated_candidate_count"]) for cost in strategy_costs
                ],
                "runtime_seconds_in_locked_seed_order": [
                    float(cost["candidate_fit_runtime_seconds"])
                    for cost in strategy_costs
                ],
            },
            "selected_fit_only_cost": {
                "scope": "objective-winning fit only; not multistart strategy cost",
                "total_runtime_seconds": float(
                    sum(float(solver["runtime_seconds"]) for solver in solvers)
                ),
                "total_actual_iterations": int(
                    sum(int(solver["actual_iterations"]) for solver in solvers)
                ),
                "total_solver_calls": int(
                    sum(int(solver["solver_calls"]) for solver in solvers)
                ),
            },
        }

    paired: dict[str, Any] = {}
    bootstrap_index = 0
    proposed = "continuation_multistart"
    for baseline in (
        "pca",
        "legacy_dynamic_kappa",
        "single_start",
        "direct_multistart",
    ):
        comparison: dict[str, Any] = {}
        for metric, direction in _CONFIRMATION_METRICS:
            differences = [
                _case_metric(case, proposed, metric)
                - _case_metric(case, baseline, metric)
                for case in cases
            ]
            bootstrap_seed = config.bootstrap_seed + bootstrap_index
            bootstrap_index += 1
            wins = [value < 0.0 for value in differences]
            if direction == "higher":
                wins = [value > 0.0 for value in differences]
            comparison[metric] = {
                "difference_definition": "continuation_multistart minus baseline",
                "preferred_direction": direction,
                "paired_differences_in_locked_seed_order": [
                    float(value) for value in differences
                ],
                "mean_difference": float(np.mean(differences)),
                "median_difference": float(np.median(differences)),
                "win_fraction": float(np.mean(wins)),
                "bootstrap_mean_difference_ci": _bootstrap_mean_ci(
                    differences,
                    draws=config.bootstrap_draws,
                    seed=bootstrap_seed,
                ),
            }
        paired[f"continuation_minus_{baseline}"] = comparison

    continuation_summary = method_summaries[proposed]
    convergence_pass = bool(
        continuation_summary["convergence_fraction"]
        >= config.minimum_confirmation_convergence_fraction
    )
    nontrivial_pass = bool(
        continuation_summary["nontrivial_qualification_fraction"] == 1.0
    )
    comparator_gates: dict[str, Any] = {}
    for baseline in (
        "pca",
        "legacy_dynamic_kappa",
        "single_start",
        "direct_multistart",
    ):
        comparison = paired[f"continuation_minus_{baseline}"]
        primary = comparison["frame_max"]
        clean = comparison["clean_relative_reconstruction_error"]
        f1 = comparison[OUTLIER_NOMINAL_BUDGET_F1]
        gate = {
            "mean_max_frame_difference_below_zero": bool(
                primary["mean_difference"] < 0.0
            ),
            "median_max_frame_difference_below_zero": bool(
                primary["median_difference"] < 0.0
            ),
            "max_frame_win_fraction": primary["win_fraction"],
            "minimum_max_frame_win_fraction": config.minimum_primary_win_fraction,
            "max_frame_win_fraction_pass": bool(
                primary["win_fraction"] >= config.minimum_primary_win_fraction
            ),
            "max_frame_bootstrap_upper_below_zero": bool(
                primary["bootstrap_mean_difference_ci"]["upper"] < 0.0
            ),
            "mean_clean_relative_error_difference_below_zero": bool(
                clean["mean_difference"] < 0.0
            ),
            "mean_nominal_budget_f1_difference": f1["mean_difference"],
            "maximum_allowed_nominal_budget_f1_decrease": (
                config.maximum_true_budget_f1_decrease
            ),
            "nominal_budget_f1_noninferiority_pass": bool(
                f1["mean_difference"]
                >= -config.maximum_true_budget_f1_decrease
            ),
        }
        gate["passed"] = bool(
            gate["mean_max_frame_difference_below_zero"]
            and gate["median_max_frame_difference_below_zero"]
            and gate["max_frame_win_fraction_pass"]
            and gate["max_frame_bootstrap_upper_below_zero"]
            and gate["mean_clean_relative_error_difference_below_zero"]
            and gate["nominal_budget_f1_noninferiority_pass"]
        )
        comparator_gates[baseline] = gate
    strict_pass = bool(
        convergence_pass
        and nontrivial_pass
        and all(record["passed"] for record in comparator_gates.values())
    )
    qualification = {
        "primary_metric": "per-seed maximum full-frame normalized RMSE",
        "convergence_fraction": continuation_summary["convergence_fraction"],
        "minimum_convergence_fraction": (
            config.minimum_confirmation_convergence_fraction
        ),
        "convergence_gate_passed": convergence_pass,
        "all_selected_continuation_fits_nontrivial": nontrivial_pass,
        "comparator_effect_gates": comparator_gates,
        "historical_before_after_comparison": (
            "truth-metric paired comparison only; legacy objectives change with "
            "per-iteration kappa and are never compared to fixed-target objectives"
        ),
        "strict_confirmatory_effect_gate_passed": strict_pass,
        "claim_status": (
            "strict_confirmatory_support"
            if strict_pass
            else (
                "fixed-budget descriptive only: convergence gate failed"
                if not convergence_pass
                else "confirmatory effect gate failed"
            )
        ),
        "failure_policy": (
            "retain and report every locked confirmation result; do not retune, "
            "replace objective winners, or hide failed seeds"
        ),
    }
    return method_summaries, {"paired": paired, "qualification": qualification}


def _cohort_worst_frame(
    cases: Sequence[ConfirmationCaseInternal],
    method: str,
) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    for case in cases:
        values = np.asarray(case.frame_errors[method], dtype=np.float64)
        index = int(np.argmax(values))
        candidates.append((float(values[index]), int(case.seed), index))
    worst_error, seed, frame_index = sorted(
        candidates, key=lambda value: (-value[0], value[1], value[2])
    )[0]
    return {
        "seed": seed,
        "frame_index": frame_index,
        "normalized_rmse": worst_error,
        "tie_break": "largest error, then lowest seed, then lowest frame index",
    }


def _figure_display_spec(
    cases: Sequence[ConfirmationCaseInternal],
    config: RobustStudyConfig,
) -> dict[str, Any]:
    by_seed = {case.seed: case for case in cases}
    representative = {
        "row_id": "representative_locked_frame",
        "seed": int(config.confirmation_seeds[0]),
        "frame_index": int(config.representative_frame_index),
        "selection_rule": "first locked confirmation seed and predeclared frame",
    }
    legacy_worst = {
        "row_id": "origin_main_legacy_cohort_worst",
        **_cohort_worst_frame(cases, "legacy_dynamic_kappa"),
        "selection_rule": "global origin/main dynamic-kappa full-frame argmax",
    }
    single_worst = {
        "row_id": "single_start_cohort_worst",
        **_cohort_worst_frame(cases, "single_start"),
        "selection_rule": "global locked-single-start full-frame argmax",
    }
    continuation_worst = {
        "row_id": "continuation_cohort_worst",
        **_cohort_worst_frame(cases, "continuation_multistart"),
        "selection_rule": "global continuation full-frame argmax",
    }
    rows = [representative, legacy_worst, single_worst, continuation_worst]
    scene_values: list[FloatArray] = []
    sparse_values: list[FloatArray] = []
    for row in rows:
        snapshot = by_seed[int(row["seed"])].snapshots[int(row["frame_index"])]
        scene_values.extend([snapshot["clean"], snapshot["observed"]])
        sparse_values.append(snapshot["true_corruption"])
    scene_limit = max(float(np.max(np.abs(value))) for value in scene_values)
    sparse_limit = max(float(np.max(np.abs(value))) for value in sparse_values)
    epsilon = float(np.finfo(np.float64).eps)
    return {
        "rows": rows,
        "all_method_panels_in_a_row_share_seed_and_frame": True,
        "scene_symmetric_abs_limit": max(scene_limit, epsilon),
        "scene_limit_source": (
            "maximum absolute clean/observed value over the selected audit rows"
        ),
        "sparse_symmetric_abs_limit": max(sparse_limit, epsilon),
        "sparse_limit_source": (
            "maximum absolute true corruption over the selected audit rows"
        ),
        "error_abs_limit": max(scene_limit, epsilon),
        "error_limit_source": "same locked data-derived limit as scene panels",
        "per_method_autoscale": False,
        "sparse_display_threshold": None,
    }


def run_confirmation(
    selection_lock_path: str | Path,
    *,
    workers: int = 1,
) -> tuple[dict[str, Any], tuple[ConfirmationCaseInternal, ...]]:
    """Run the locked confirmation bank with no parameter override surface."""

    lock = load_selection_lock(selection_lock_path, verify_source=True)
    config = _config_from_mapping(lock["study_protocol"]["config"])
    execution = _thread_environment(workers)
    alpha = float(lock["selected_parameters"]["alpha"])
    kappa = float(lock["selected_parameters"]["kappa"])
    payloads = [
        {
            "config": asdict(config),
            "seed": int(seed),
            "alpha": alpha,
            "kappa": kappa,
        }
        for seed in config.confirmation_seeds
    ]
    started = perf_counter()
    raw_cases = _parallel_map_by_seed(
        _run_confirmation_seed_worker, payloads, workers=workers
    )
    runtime = perf_counter() - started
    by_seed = {int(case.seed): case for case in raw_cases}
    if set(by_seed) != set(config.confirmation_seeds) or len(by_seed) != len(
        config.confirmation_seeds
    ):
        raise RuntimeError("confirmation seed shards do not form an exact cover")
    cases = tuple(by_seed[int(seed)] for seed in config.confirmation_seeds)
    for case in cases:
        if tuple(case.frame_errors) != _CONFIRMATION_METHOD_ORDER:
            raise RuntimeError("confirmation method grid is not exact or ordered")
        if any(
            np.asarray(case.frame_errors[method]).shape != (config.n_timesteps,)
            for method in _CONFIRMATION_METHOD_ORDER
        ):
            raise RuntimeError("confirmation frame grid is incomplete")
    method_summaries, paired_and_gate = _aggregate_confirmation(cases, config)
    figure_spec = _figure_display_spec(cases, config)
    repo_root = Path(__file__).resolve().parents[1]
    provenance = _executed_source_provenance(repo_root)
    if provenance["source_sha256"] != lock["source_sha256"]:
        raise RuntimeError("source closure changed after lock verification")
    evidence_projection = _without_runtime(
        {
            "selection_lock_sha256": lock["selection_lock_sha256"],
            "cases": [case.record for case in cases],
            "method_summaries": method_summaries,
            **paired_and_gate,
            "figure_display": figure_spec,
            "metric_semantics": _confirmation_metric_semantics(),
        }
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "phase": "confirmation",
        "paper_scale": bool(lock["paper_scale"]),
        "selection_lock_sha256": lock["selection_lock_sha256"],
        "study_protocol_sha256": lock["study_protocol_sha256"],
        "locked_selected_parameters": {"alpha": alpha, "kappa": kappa},
        "confirmation_parameter_overrides_allowed": False,
        "confirmation_truth_used_for_tuning": False,
        "confirmation_seeds": list(config.confirmation_seeds),
        "metric_semantics": _confirmation_metric_semantics(),
        "case_records": [case.record for case in cases],
        "method_summaries": method_summaries,
        "paired_confirm_summaries": paired_and_gate["paired"],
        "strict_qualification": paired_and_gate["qualification"],
        "figure_display": figure_spec,
        "confirmation_evidence_sha256_without_runtime": canonical_sha256(
            evidence_projection
        ),
        "provenance": provenance,
        "execution": {**execution, "runtime_seconds": float(runtime)},
        "scientific_boundary": (
            "locked synthetic-DGP confirmation of initialization/solve strategies "
            "around the existing robust VP-SPCA implementation; no new PCA "
            "algorithm and no real-video generalization claim"
        ),
    }
    report["report_body_sha256"] = canonical_sha256(report)
    return _json_safe(report), cases


def _write_confirmation_cases_csv(
    report: Mapping[str, Any], path: Path
) -> None:
    fields = [
        "seed",
        "method",
        "clean_relative_reconstruction_error",
        "frame_median",
        "frame_p95",
        "frame_max",
        "worst_frame_index",
        "worst_frame_time",
        "catastrophic_frame_count",
        "outlier_roc_auc",
        "outlier_average_precision",
        OUTLIER_NOMINAL_BUDGET_F1,
        "converged",
        "final_stationarity",
        "final_target_objective",
        "nontrivial_qualified",
        "selected_start_id",
        "actual_iterations",
        "solver_calls",
        "runtime_seconds",
        "candidate_fit_sha256",
    ]
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["case_records"]:
            for method in _CONFIRMATION_METHOD_ORDER:
                record = case["methods"][method]
                truth = record["truth_metrics_after_fit_selection_lock"]
                frame = truth["full_frame_normalized_rmse"]
                detection = truth["outlier_detection"]
                solver = record["solver"]
                stationarity = solver["final_stationarity"]
                objective = solver["final_target_objective_recomputed"]
                writer.writerow(
                    {
                        "seed": case["seed"],
                        "method": method,
                        "clean_relative_reconstruction_error": (
                            f"{truth['clean_relative_reconstruction_error']:.17g}"
                        ),
                        "frame_median": f"{frame['median']:.17g}",
                        "frame_p95": f"{frame['p95']:.17g}",
                        "frame_max": f"{frame['max']:.17g}",
                        "worst_frame_index": frame["worst_frame_index"],
                        "worst_frame_time": f"{frame['worst_frame_time']:.17g}",
                        "catastrophic_frame_count": truth[
                            "catastrophic_frame_count"
                        ],
                        "outlier_roc_auc": f"{detection['roc_auc']:.17g}",
                        "outlier_average_precision": (
                            f"{detection['average_precision']:.17g}"
                        ),
                        OUTLIER_NOMINAL_BUDGET_F1: f"{detection['f1']:.17g}",
                        "converged": solver["converged"],
                        "final_stationarity": (
                            "" if stationarity is None else f"{stationarity:.17g}"
                        ),
                        "final_target_objective": (
                            "" if objective is None else f"{objective:.17g}"
                        ),
                        "nontrivial_qualified": solver[
                            "nontrivial_solution_gate"
                        ]["qualified"],
                        "selected_start_id": (
                            "" if solver["start_id"] is None else solver["start_id"]
                        ),
                        "actual_iterations": solver["actual_iterations"],
                        "solver_calls": solver["solver_calls"],
                        "runtime_seconds": f"{solver['runtime_seconds']:.17g}",
                        "candidate_fit_sha256": solver["candidate_fit_sha256"],
                    }
                )


def _write_confirmation_frames_csv(
    cases: Sequence[ConfirmationCaseInternal],
    config: RobustStudyConfig,
    path: Path,
) -> None:
    fields = ["seed", "method", "frame_index", "time", "normalized_rmse"]
    time = np.linspace(0.0, config.total_time, config.n_timesteps)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            for method in _CONFIRMATION_METHOD_ORDER:
                values = np.asarray(case.frame_errors[method], dtype=np.float64)
                for frame_index, value in enumerate(values):
                    writer.writerow(
                        {
                            "seed": case.seed,
                            "method": method,
                            "frame_index": frame_index,
                            "time": f"{time[frame_index]:.17g}",
                            "normalized_rmse": f"{value:.17g}",
                        }
                    )


def _render_confirmation_figure(
    report: Mapping[str, Any],
    cases: Sequence[ConfirmationCaseInternal],
    config: RobustStudyConfig,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    by_seed = {case.seed: case for case in cases}
    display = report["figure_display"]
    rows = display["rows"]
    scene_limit = float(display["scene_symmetric_abs_limit"])
    sparse_limit = float(display["sparse_symmetric_abs_limit"])
    error_limit = float(display["error_abs_limit"])
    panels: tuple[tuple[str, str, str], ...] = (
        ("clean", "Clean truth", "scene"),
        ("observed", "Observed", "scene"),
        ("pca_reconstruction", "PCA low-rank", "scene"),
        ("legacy_dynamic_kappa_reconstruction", "Legacy-dynamic low-rank", "scene"),
        ("single_start_reconstruction", "Single-start low-rank", "scene"),
        ("direct_multistart_reconstruction", "Direct-MS low-rank", "scene"),
        (
            "continuation_multistart_reconstruction",
            "Continuation-MS low-rank",
            "scene",
        ),
        ("true_corruption", "True corruption", "sparse"),
        ("legacy_dynamic_kappa_outliers", "Legacy-dynamic S", "sparse"),
        ("single_start_outliers", "Single-start S", "sparse"),
        ("direct_multistart_outliers", "Direct-MS S", "sparse"),
        ("continuation_multistart_outliers", "Continuation-MS S", "sparse"),
        ("legacy_dynamic_kappa_abs_error", "|Legacy - clean|", "error"),
        ("single_start_abs_error", "|Single - clean|", "error"),
        ("continuation_multistart_abs_error", "|Continuation - clean|", "error"),
    )
    figure, axes = plt.subplots(
        len(rows), len(panels), figsize=(35.0, 10.8), squeeze=False
    )
    time = np.linspace(0.0, config.total_time, config.n_timesteps)
    for row_index, row in enumerate(rows):
        seed = int(row["seed"])
        frame_index = int(row["frame_index"])
        case = by_seed[seed]
        snapshot = case.snapshots[frame_index]
        for column_index, (key, title, kind) in enumerate(panels):
            axis = axes[row_index, column_index]
            values = np.asarray(snapshot[key], dtype=np.float64).reshape(
                config.grid_size, config.grid_size
            )
            if kind == "scene":
                image = axis.imshow(
                    values,
                    cmap="RdBu_r",
                    vmin=-scene_limit,
                    vmax=scene_limit,
                    interpolation="nearest",
                )
            elif kind == "sparse":
                image = axis.imshow(
                    values,
                    cmap="RdBu_r",
                    vmin=-sparse_limit,
                    vmax=sparse_limit,
                    interpolation="nearest",
                )
            else:
                image = axis.imshow(
                    values,
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_limit,
                    interpolation="nearest",
                )
            del image
            if row_index == 0:
                axis.set_title(title, fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])
        frame_errors = {
            method: float(case.frame_errors[method][frame_index])
            for method in _CONFIRMATION_METHOD_ORDER
        }
        axes[row_index, 0].set_ylabel(
            f"{row['row_id']}\nseed={seed}, frame={frame_index}, "
            f"t={time[frame_index]:.3g}\n"
            f"NRMSE P/L/S/D/C={frame_errors['pca']:.3g}/"
            f"{frame_errors['legacy_dynamic_kappa']:.3g}/"
            f"{frame_errors['single_start']:.3g}/"
            f"{frame_errors['direct_multistart']:.3g}/"
            f"{frame_errors['continuation_multistart']:.3g}",
            fontsize=7,
        )
    figure.suptitle(
        "Locked confirmation: representative plus historical/fixed/proposed cohort worst "
        "frames; shared data-derived scales; native S (no display threshold)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95), pad=0.35)
    figure.savefig(path, dpi=config.dpi, bbox_inches="tight")
    plt.close(figure)


def save_confirmation_output(
    report: Mapping[str, Any],
    cases: Sequence[ConfirmationCaseInternal],
    selection_lock_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    lock_path = Path(selection_lock_path)
    lock = load_selection_lock(lock_path, verify_source=True)
    if report["selection_lock_sha256"] != lock["selection_lock_sha256"]:
        raise ValueError("confirmation report and selection lock disagree")
    config = _config_from_mapping(lock["study_protocol"]["config"])
    destination = _claim_new_output_directory(output_dir)
    paths = {
        "report": destination / CONFIRMATION_REPORT_NAME,
        "cases": destination / CONFIRMATION_CASES_NAME,
        "frames": destination / CONFIRMATION_FRAMES_NAME,
        "figure": destination / CONFIRMATION_FIGURE_NAME,
        "lock": destination / SELECTION_LOCK_NAME,
        "manifest": destination / CONFIRMATION_MANIFEST_NAME,
    }
    _write_json_exclusive(paths["report"], report)
    _write_confirmation_cases_csv(report, paths["cases"])
    _write_confirmation_frames_csv(cases, config, paths["frames"])
    _render_confirmation_figure(report, cases, config, paths["figure"])
    with lock_path.open("rb") as source, paths["lock"].open("xb") as target:
        shutil.copyfileobj(source, target)
    paths["manifest"] = _write_exact_manifest(
        destination,
        manifest_name=CONFIRMATION_MANIFEST_NAME,
        manifest_kind="paper_video_robust_confirmation_exact_closure",
        required_payloads=_CONFIRMATION_REQUIRED_PAYLOADS,
    )
    return paths


def _parse_csv_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid CSV Boolean: {value!r}")


def _close_float(left: Any, right: Any) -> bool:
    return bool(
        np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12)
    )


def _validate_confirmation_case_contract(
    case: Mapping[str, Any], config: RobustStudyConfig
) -> None:
    if set(case.get("methods", {})) != set(_CONFIRMATION_METHOD_ORDER):
        raise ValueError("confirmation case method set mismatch")
    specs = default_start_specs(config)
    expected_start_ids = [spec.start_id for spec in specs]
    if [record["start_id"] for record in case.get("start_specs", [])] != (
        expected_start_ids
    ):
        raise ValueError("confirmation case start order mismatch")
    fairness = case.get("fairness", {})
    if fairness.get("start_ids") != expected_start_ids:
        raise ValueError("confirmation fairness start order mismatch")
    if fairness.get("direct_initial_B_sha256") != fairness.get(
        "continuation_initial_B_sha256"
    ):
        raise ValueError("confirmation direct/continuation starts are unpaired")
    if fairness.get("allocated_iterations_per_start") != config.max_iter_per_start:
        raise ValueError("confirmation fairness budget mismatch")
    if fairness.get("continuation_schedule") != [
        float(case["selected_parameters"]["kappa"] * multiplier)
        for multiplier in config.continuation_kappa_multipliers
    ]:
        raise ValueError("confirmation continuation schedule mismatch")
    expected_arms = {
        "direct_multistart_equal_budget",
        "B_only_descending_kappa_multistart",
    }
    if set(case.get("arms", {})) != expected_arms:
        raise ValueError("confirmation arm set mismatch")
    for arm_id in (
        "direct_multistart_equal_budget",
        "B_only_descending_kappa_multistart",
    ):
        arm = case["arms"][arm_id]
        candidates = arm.get("candidate_records")
        if not isinstance(candidates, list) or [
            record["start_id"] for record in candidates
        ] != expected_start_ids:
            raise ValueError("confirmation candidate start grid mismatch")
        selected_index = select_objective_winner_index(candidates)
        if selected_index != int(arm["selected_index"]):
            raise ValueError("confirmation selected candidate is not objective minimum")
        if arm["selected_start_id"] != candidates[selected_index]["start_id"]:
            raise ValueError("confirmation selected start ID mismatch")
        for candidate in candidates:
            if int(candidate["allocated_iterations"]) != config.max_iter_per_start:
                raise ValueError("confirmation candidate budget mismatch")
            stages = candidate.get("stages")
            if not isinstance(stages, list) or not stages:
                raise ValueError("confirmation candidate stage records missing")
            if not _close_float(stages[-1]["kappa"], case["selected_parameters"]["kappa"]):
                raise ValueError("confirmation candidate does not end at target kappa")
            for stage_index, stage in enumerate(stages):
                trace = stage.get("objective_trace")
                if not isinstance(trace, list) or not _objective_trace_monotone(trace):
                    raise ValueError("confirmation stage objective trace is not monotone")
                if bool(stage["objective_trace_monotone"]) is not True:
                    raise ValueError("confirmation stage monotonicity declaration mismatch")
                if stage_index:
                    if stage["initial_B_sha256"] != stages[stage_index - 1][
                        "final_B_sha256"
                    ]:
                        raise ValueError("confirmation B-only handoff hash mismatch")
                    if stage["handoff_from_previous_final_B"] is not True:
                        raise ValueError("confirmation B-only handoff flag mismatch")
            if len(stages) > 1 and candidate["B_only_continuation"] is not True:
                raise ValueError("confirmation continuation semantics mismatch")
    body = dict(case)
    digest = body.pop("case_body_sha256_without_runtime", None)
    if digest != canonical_sha256(_without_runtime(body)):
        raise ValueError("confirmation case evidence digest mismatch")


def _lightweight_confirmation_cases(
    case_records: Sequence[Mapping[str, Any]],
) -> tuple[ConfirmationCaseInternal, ...]:
    return tuple(
        ConfirmationCaseInternal(
            seed=int(case["seed"]),
            record=case,
            frame_errors={},
            snapshots={},
        )
        for case in case_records
    )


def _scientific_method_summary_projection(
    summaries: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop timing-accounting fields while retaining every scientific value."""

    projected: dict[str, Any] = {}
    for method in _CONFIRMATION_METHOD_ORDER:
        summary = summaries.get(method)
        if not isinstance(summary, Mapping):
            raise ValueError(f"confirmation method summary missing for {method}")
        projected[method] = {
            "metrics": _normalize_legacy_budget_metric_keys(summary.get("metrics")),
            "convergence_fraction": summary.get("convergence_fraction"),
            "nontrivial_qualification_fraction": summary.get(
                "nontrivial_qualification_fraction"
            ),
        }
    return projected


def _reported_cost_accounting_mode(
    report: Mapping[str, Any],
    cases: Sequence[ConfirmationCaseInternal],
) -> dict[str, str]:
    """Validate new strategy cost fields or identify the version-1 undercount."""

    modes: dict[str, str] = {}
    summaries = report["method_summaries"]
    for method in _CONFIRMATION_METHOD_ORDER:
        summary = summaries[method]
        costs = [_strategy_candidate_fit_cost(case.record, method) for case in cases]
        expected_strategy = {
            "scope": costs[0]["cost_scope"],
            "total_runtime_seconds": float(
                sum(float(cost["candidate_fit_runtime_seconds"]) for cost in costs)
            ),
            "total_allocated_iterations": int(
                sum(int(cost["allocated_iterations"]) for cost in costs)
            ),
            "total_actual_iterations": int(
                sum(int(cost["actual_iterations"]) for cost in costs)
            ),
            "total_solver_calls": int(sum(int(cost["solver_calls"]) for cost in costs)),
            "evaluated_candidate_count_per_seed": [
                int(cost["evaluated_candidate_count"]) for cost in costs
            ],
            "runtime_seconds_in_locked_seed_order": [
                float(cost["candidate_fit_runtime_seconds"]) for cost in costs
            ],
        }
        if "strategy_candidate_fit_cost" in summary:
            if canonical_sha256(summary["strategy_candidate_fit_cost"]) != canonical_sha256(
                expected_strategy
            ):
                raise ValueError(f"strategy cost summary mismatch for {method}")
            modes[method] = "all_candidate_strategy_cost"
            continue

        # Frozen version-1 artifacts retained only the selected winner's cost.
        # Accept that historical schema after verifying exactly what it means;
        # the strict audit exposes corrected all-candidate totals separately.
        solvers = [case.record["methods"][method]["solver"] for case in cases]
        selected_runtime = float(
            sum(float(solver["runtime_seconds"]) for solver in solvers)
        )
        selected_iterations = int(
            sum(int(solver["actual_iterations"]) for solver in solvers)
        )
        if not _close_float(summary.get("total_runtime_seconds"), selected_runtime):
            raise ValueError(f"legacy selected-fit runtime mismatch for {method}")
        if int(summary.get("total_actual_iterations")) != selected_iterations:
            raise ValueError(f"legacy selected-fit iteration mismatch for {method}")
        modes[method] = "legacy_selected_fit_only"
    return modes


def _validate_recomputed_confirmation_semantics(
    report: Mapping[str, Any],
    config: RobustStudyConfig,
    case_records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[ConfirmationCaseInternal, ...], dict[str, str]]:
    cases = _lightweight_confirmation_cases(case_records)
    recomputed_summaries, recomputed_pair_and_gate = _aggregate_confirmation(
        cases, config
    )
    reported_science = _scientific_method_summary_projection(
        report.get("method_summaries", {})
    )
    recomputed_science = _scientific_method_summary_projection(recomputed_summaries)
    if canonical_sha256(reported_science) != canonical_sha256(recomputed_science):
        raise ValueError("confirmation method summaries do not recompute from cases")
    reported_paired = _normalize_legacy_budget_metric_keys(
        report.get("paired_confirm_summaries")
    )
    if canonical_sha256(reported_paired) != canonical_sha256(
        recomputed_pair_and_gate["paired"]
    ):
        raise ValueError("confirmation paired summaries do not recompute from cases")
    reported_qualification = _normalize_legacy_budget_metric_keys(
        report.get("strict_qualification")
    )
    if canonical_sha256(reported_qualification) != canonical_sha256(
        recomputed_pair_and_gate["qualification"]
    ):
        raise ValueError("confirmation qualification gate does not recompute from cases")
    cost_modes = _reported_cost_accounting_mode(report, cases)
    return recomputed_pair_and_gate["qualification"], cases, cost_modes


def validate_confirmation_output_directory(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    manifest = _validate_exact_output_directory(
        root,
        required_outputs=_CONFIRMATION_REQUIRED_OUTPUTS,
        required_payloads=_CONFIRMATION_REQUIRED_PAYLOADS,
        manifest_name=CONFIRMATION_MANIFEST_NAME,
    )
    lock = load_selection_lock(root / SELECTION_LOCK_NAME, verify_source=False)
    config = _config_from_mapping(lock["study_protocol"]["config"])
    report = _strict_json(root / CONFIRMATION_REPORT_NAME)
    report_body = dict(report)
    digest = report_body.pop("report_body_sha256", None)
    if digest != canonical_sha256(report_body):
        raise ValueError("confirmation report digest mismatch")
    if report.get("phase") != "confirmation":
        raise ValueError("confirmation report phase mismatch")
    if report.get("selection_lock_sha256") != lock["selection_lock_sha256"]:
        raise ValueError("confirmation report and copied lock disagree")
    if report.get("study_protocol_sha256") != lock["study_protocol_sha256"]:
        raise ValueError("confirmation report protocol digest mismatch")
    if report.get("provenance", {}).get("source_sha256") != lock.get(
        "source_sha256"
    ):
        raise ValueError("confirmation report source closure mismatch")
    if report.get("confirmation_seeds") != list(config.confirmation_seeds):
        raise ValueError("confirmation report seed bank mismatch")
    if report.get("locked_selected_parameters") != lock["selected_parameters"]:
        raise ValueError("confirmation report selected parameter mismatch")
    cases = report.get("case_records")
    if not isinstance(cases, list) or [int(case["seed"]) for case in cases] != list(
        config.confirmation_seeds
    ):
        raise ValueError("confirmation report cases are not an exact seed cover")
    for case in cases:
        _validate_confirmation_case_contract(case, config)

    _validate_recomputed_confirmation_semantics(report, config, cases)

    evidence_body = {
        "selection_lock_sha256": report["selection_lock_sha256"],
        "cases": cases,
        "method_summaries": report["method_summaries"],
        "paired": report["paired_confirm_summaries"],
        "qualification": report["strict_qualification"],
        "figure_display": report["figure_display"],
    }
    if "metric_semantics" in report:
        if report["metric_semantics"] != _confirmation_metric_semantics():
            raise ValueError("confirmation metric semantics mismatch")
        evidence_body["metric_semantics"] = report["metric_semantics"]
    evidence_projection = _without_runtime(evidence_body)
    if report.get("confirmation_evidence_sha256_without_runtime") != canonical_sha256(
        evidence_projection
    ):
        raise ValueError("confirmation evidence digest mismatch")

    case_rows: list[dict[str, str]]
    with (root / CONFIRMATION_CASES_NAME).open(
        encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        case_rows = list(reader)
    expected_pairs = [
        (int(seed), method)
        for seed in config.confirmation_seeds
        for method in _CONFIRMATION_METHOD_ORDER
    ]
    if [(int(row["seed"]), row["method"]) for row in case_rows] != expected_pairs:
        raise ValueError("confirmation cases CSV is not an exact ordered grid")
    case_by_seed = {int(case["seed"]): case for case in cases}
    for row in case_rows:
        case = case_by_seed[int(row["seed"])]
        method = row["method"]
        method_record = case["methods"][method]
        truth = method_record["truth_metrics_after_fit_selection_lock"]
        frame = truth["full_frame_normalized_rmse"]
        detection = truth["outlier_detection"]
        solver = method_record["solver"]
        expected_numbers = {
            "clean_relative_reconstruction_error": truth[
                "clean_relative_reconstruction_error"
            ],
            "frame_median": frame["median"],
            "frame_p95": frame["p95"],
            "frame_max": frame["max"],
            "worst_frame_time": frame["worst_frame_time"],
            "outlier_roc_auc": detection["roc_auc"],
            "outlier_average_precision": detection["average_precision"],
            OUTLIER_NOMINAL_BUDGET_F1: detection["f1"],
            "runtime_seconds": solver["runtime_seconds"],
        }
        budget_columns = {
            OUTLIER_NOMINAL_BUDGET_F1,
            LEGACY_OUTLIER_TRUE_BUDGET_F1,
        } & set(row)
        if len(budget_columns) != 1:
            raise ValueError(
                "confirmation cases CSV must contain exactly one nominal-budget "
                "F1 field"
            )
        budget_column = next(iter(budget_columns))
        expected_numbers[budget_column] = expected_numbers.pop(
            OUTLIER_NOMINAL_BUDGET_F1
        )
        if any(
            not _close_float(row[name], value)
            for name, value in expected_numbers.items()
        ):
            raise ValueError("confirmation cases CSV numeric value mismatch")
        if int(row["worst_frame_index"]) != int(frame["worst_frame_index"]):
            raise ValueError("confirmation cases CSV worst frame mismatch")
        if int(row["catastrophic_frame_count"]) != int(
            truth["catastrophic_frame_count"]
        ):
            raise ValueError("confirmation cases CSV catastrophic count mismatch")
        if _parse_csv_bool(row["converged"]) is not bool(solver["converged"]):
            raise ValueError("confirmation cases CSV convergence mismatch")
        if _parse_csv_bool(row["nontrivial_qualified"]) is not bool(
            solver["nontrivial_solution_gate"]["qualified"]
        ):
            raise ValueError("confirmation cases CSV qualification mismatch")
        if row["candidate_fit_sha256"] != solver["candidate_fit_sha256"]:
            raise ValueError("confirmation cases CSV fit hash mismatch")

    with (root / CONFIRMATION_FRAMES_NAME).open(
        encoding="utf-8", newline=""
    ) as stream:
        frame_rows = list(csv.DictReader(stream))
    expected_frame_keys = [
        (int(seed), method, frame_index)
        for seed in config.confirmation_seeds
        for method in _CONFIRMATION_METHOD_ORDER
        for frame_index in range(config.n_timesteps)
    ]
    observed_frame_keys = [
        (int(row["seed"]), row["method"], int(row["frame_index"]))
        for row in frame_rows
    ]
    if observed_frame_keys != expected_frame_keys:
        raise ValueError("confirmation full-frame CSV is not an exact ordered grid")
    expected_time = np.linspace(0.0, config.total_time, config.n_timesteps)
    values_by_pair: dict[tuple[int, str], list[float]] = {}
    for row in frame_rows:
        frame_index = int(row["frame_index"])
        if not _close_float(row["time"], expected_time[frame_index]):
            raise ValueError("confirmation full-frame CSV time mismatch")
        value = float(row["normalized_rmse"])
        if not np.isfinite(value):
            raise ValueError("confirmation full-frame CSV has non-finite error")
        values_by_pair.setdefault((int(row["seed"]), row["method"]), []).append(
            value
        )
    for (seed, method), values in values_by_pair.items():
        summary = summarize_full_frame_errors(
            np.asarray(values, dtype=np.float64), expected_time
        )
        reported = case_by_seed[seed]["methods"][method][
            "truth_metrics_after_fit_selection_lock"
        ]["full_frame_normalized_rmse"]
        for name in ("median", "p95", "max", "worst_frame_time"):
            if not _close_float(summary[name], reported[name]):
                raise ValueError("confirmation full-frame summary mismatch")
        if summary["worst_frame_index"] != reported["worst_frame_index"]:
            raise ValueError("confirmation full-frame argmax mismatch")

    lightweight_cases = tuple(
        ConfirmationCaseInternal(
            seed=int(seed),
            record={},
            frame_errors={
                method: np.asarray(values_by_pair[(int(seed), method)])
                for method in _CONFIRMATION_METHOD_ORDER
            },
            snapshots={},
        )
        for seed in config.confirmation_seeds
    )
    rows = report.get("figure_display", {}).get("rows", [])
    if len(rows) != 4:
        raise ValueError("confirmation figure must contain exactly four audit rows")
    expected_legacy = _cohort_worst_frame(
        lightweight_cases, "legacy_dynamic_kappa"
    )
    expected_single = _cohort_worst_frame(lightweight_cases, "single_start")
    expected_continuation = _cohort_worst_frame(
        lightweight_cases, "continuation_multistart"
    )
    for row, expected in zip(
        rows[1:], (expected_legacy, expected_single, expected_continuation)
    ):
        if int(row["seed"]) != expected["seed"] or int(
            row["frame_index"]
        ) != expected["frame_index"] or not _close_float(
            row["normalized_rmse"], expected["normalized_rmse"]
        ):
            raise ValueError("confirmation figure worst-row selection mismatch")
    if (
        int(rows[0]["seed"]) != config.confirmation_seeds[0]
        or int(rows[0]["frame_index"]) != config.representative_frame_index
    ):
        raise ValueError("confirmation representative row selection mismatch")
    return manifest


def audit_confirmation_output_directory(output_dir: str | Path) -> dict[str, Any]:
    """Strictly rederive claim gates and corrected costs from a frozen package.

    This is a read-only compatibility path for both the original version-1
    server artifacts and newly generated artifacts.  It does not trust the
    report's aggregate tables or claim-status string: they are recomputed from
    the retained per-seed case records after the structural validator passes.
    """

    root = Path(output_dir)
    manifest = validate_confirmation_output_directory(root)
    lock = load_selection_lock(root / SELECTION_LOCK_NAME, verify_source=False)
    config = _config_from_mapping(lock["study_protocol"]["config"])
    report = _strict_json(root / CONFIRMATION_REPORT_NAME)
    qualification, cases, cost_modes = _validate_recomputed_confirmation_semantics(
        report, config, report["case_records"]
    )
    recomputed_summaries, _ = _aggregate_confirmation(cases, config)
    continuation_truth = [
        case.record["methods"]["continuation_multistart"]
        ["truth_metrics_after_fit_selection_lock"]
        for case in cases
    ]
    catastrophic_total = int(
        sum(int(record["catastrophic_frame_count"]) for record in continuation_truth)
    )
    worst_case = max(
        (
            (
                float(record["full_frame_normalized_rmse"]["max"]),
                int(case.seed),
                int(record["full_frame_normalized_rmse"]["worst_frame_index"]),
            )
            for case, record in zip(cases, continuation_truth)
        ),
        key=lambda item: (item[0], -item[1], -item[2]),
    )
    audit: dict[str, Any] = {
        "schema_version": SEMANTIC_AUDIT_VERSION,
        "audit_kind": "paper_video_robust_confirmation_read_only_semantic_audit",
        "artifact_manifest_sha256": manifest["manifest_sha256"],
        "selection_lock_sha256": lock["selection_lock_sha256"],
        "paper_scale": bool(report["paper_scale"]),
        "confirmation_seed_count": len(cases),
        "reported_claim_status": report["strict_qualification"]["claim_status"],
        "recomputed_claim_status": qualification["claim_status"],
        "recomputed_strict_confirmatory_effect_gate_passed": qualification[
            "strict_confirmatory_effect_gate_passed"
        ],
        "aggregate_and_gate_recomputation_passed": True,
        "metric_semantics": _confirmation_metric_semantics(),
        "reported_cost_accounting_mode_by_method": cost_modes,
        "recomputed_strategy_candidate_fit_cost_by_method": {
            method: recomputed_summaries[method]["strategy_candidate_fit_cost"]
            for method in _CONFIRMATION_METHOD_ORDER
        },
        "clean_sequence_editorial_audit": {
            "predeclared_catastrophic_frame_threshold": (
                config.catastrophic_frame_threshold
            ),
            "continuation_catastrophic_frame_count_across_all_seeds": (
                catastrophic_total
            ),
            "zero_catastrophic_frames": catastrophic_total == 0,
            "continuation_cohort_worst_normalized_rmse": worst_case[0],
            "continuation_cohort_worst_seed": worst_case[1],
            "continuation_cohort_worst_frame_index": worst_case[2],
            "role": (
                "absolute post-run editorial check; not part of the frozen strict "
                "confirmatory effect gate"
            ),
        },
        "claim_boundary": (
            "synthetic-DGP solver-strategy evidence only; no new PCA algorithm, "
            "real-video generalization, whole-video visual-inspection, or runtime "
            "speedup claim"
        ),
    }
    audit["audit_body_sha256"] = canonical_sha256(audit)
    return _json_safe(audit)


def audit_video_study_output_directories(
    development_dir: str | Path,
    confirmation_dir: str | Path,
) -> dict[str, Any]:
    """Read-only two-phase audit including development-selection recomputation."""

    development_root = Path(development_dir)
    confirmation_root = Path(confirmation_dir)
    development_manifest = validate_development_output_directory(development_root)
    confirmation_audit = audit_confirmation_output_directory(confirmation_root)
    development_lock = development_root / SELECTION_LOCK_NAME
    confirmation_lock = confirmation_root / SELECTION_LOCK_NAME
    if _file_sha256(development_lock) != _file_sha256(confirmation_lock):
        raise ValueError("development and confirmation copied locks differ")
    development_report = _strict_json(development_root / DEVELOPMENT_RESULTS_NAME)
    audit: dict[str, Any] = {
        "schema_version": SEMANTIC_AUDIT_VERSION,
        "audit_kind": "paper_video_robust_two_phase_read_only_semantic_audit",
        "development_manifest_sha256": development_manifest["manifest_sha256"],
        "development_selection_recomputed": True,
        "selected_parameters": {
            "alpha": development_report["selection"]["selected_alpha"],
            "kappa": development_report["selection"]["selected_kappa"],
            "parameter_order_index": development_report["selection"][
                "selected_parameter_order_index"
            ],
        },
        "development_selection_sha256": canonical_sha256(
            development_report["selection"]
        ),
        "copied_selection_lock_byte_identical": True,
        "confirmation": confirmation_audit,
    }
    audit["audit_body_sha256"] = canonical_sha256(audit)
    return _json_safe(audit)


def _preflight_new_output_path(path: str | Path) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    develop = subparsers.add_parser(
        "develop", help="run the exact development grid and freeze a lock"
    )
    develop.add_argument("--output-dir", type=Path, required=True)
    develop.add_argument("--workers", type=int, default=1)
    develop.add_argument(
        "--smoke",
        action="store_true",
        help="tiny plumbing-only protocol; never paper-scale evidence",
    )
    confirm = subparsers.add_parser(
        "confirm", help="run the copied locked protocol with no tuning options"
    )
    confirm.add_argument("--selection-lock", type=Path, required=True)
    confirm.add_argument("--output-dir", type=Path, required=True)
    confirm.add_argument("--workers", type=int, default=1)
    validate_dev = subparsers.add_parser(
        "validate-development", help="validate a complete development directory"
    )
    validate_dev.add_argument("output_dir", type=Path)
    validate_confirm = subparsers.add_parser(
        "validate-confirmation", help="validate a complete confirmation directory"
    )
    validate_confirm.add_argument("output_dir", type=Path)
    audit_confirm = subparsers.add_parser(
        "audit-confirmation",
        help="read-only recomputation of confirmation aggregates, gates, and costs",
    )
    audit_confirm.add_argument("output_dir", type=Path)
    audit_study = subparsers.add_parser(
        "audit-study",
        help="read-only two-phase audit including development selection",
    )
    audit_study.add_argument("--development-dir", type=Path, required=True)
    audit_study.add_argument("--confirmation-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_cli_parser().parse_args(argv)
    if arguments.command == "develop":
        _preflight_new_output_path(arguments.output_dir)
        config = smoke_study_config() if arguments.smoke else RobustStudyConfig()
        report, lock = run_development(config, workers=arguments.workers)
        paths = save_development_output(report, lock, arguments.output_dir)
        validate_development_output_directory(arguments.output_dir)
        print(
            json.dumps(
                {
                    "phase": "development",
                    "paper_scale": report["paper_scale"],
                    "selected_parameters": lock["selected_parameters"],
                    "selection_lock_sha256": lock["selection_lock_sha256"],
                    "output_dir": str(arguments.output_dir.resolve()),
                    "manifest": str(paths["manifest"].resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "confirm":
        _preflight_new_output_path(arguments.output_dir)
        report, cases = run_confirmation(
            arguments.selection_lock, workers=arguments.workers
        )
        paths = save_confirmation_output(
            report,
            cases,
            arguments.selection_lock,
            arguments.output_dir,
        )
        validate_confirmation_output_directory(arguments.output_dir)
        print(
            json.dumps(
                {
                    "phase": "confirmation",
                    "paper_scale": report["paper_scale"],
                    "claim_status": report["strict_qualification"]["claim_status"],
                    "confirmation_evidence_sha256_without_runtime": report[
                        "confirmation_evidence_sha256_without_runtime"
                    ],
                    "output_dir": str(arguments.output_dir.resolve()),
                    "manifest": str(paths["manifest"].resolve()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "validate-development":
        validate_development_output_directory(arguments.output_dir)
        print("VALID development")
        return 0
    if arguments.command == "validate-confirmation":
        validate_confirmation_output_directory(arguments.output_dir)
        print("VALID confirmation")
        return 0
    if arguments.command == "audit-confirmation":
        print(
            json.dumps(
                audit_confirmation_output_directory(arguments.output_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if arguments.command == "audit-study":
        print(
            json.dumps(
                audit_video_study_output_directories(
                    arguments.development_dir, arguments.confirmation_dir
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    raise RuntimeError("unreachable command")


if __name__ == "__main__":
    raise SystemExit(main())

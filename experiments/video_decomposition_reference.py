"""Reference multiscale-video experiments with source provenance.

This runner ports the 200 x 200 x 300 construction from the historical
``spca_multiscale_video.py`` demo to the current :mod:`rspca` APIs.  It keeps
the two experiments from Section 6.1 of Erichson et al. separate:

* Figure 5: clean spatial modes and their time dynamics (PCA versus VP-SPCA);
* Figure 6: clean/corrupted/reconstruction/native-outlier decomposition.

The visual alignment uses a Hungarian permutation and signs only.  It never
fits a recovered amplitude to ground truth.  The robust fit receives one
explicit, predeclared Huber threshold, and the plotted outlier layer is the
native ``S`` returned by the solver without display thresholding.

The default is intentionally the paper-scale 200 x 200 grid and 300 frames.
Use ``--smoke`` for a small source-level check; smoke output is not
paper-scale evidence.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.metrics import outlier_detection_metrics, relative_reconstruction_error
from rspca.randomized import RandomizedVPSPCAResult, fit_randomized_vp_spca
from rspca.robust import RobustVPSPCAResult, fit_robust_vp_spca
from rspca.vp import VPSPCAResult, fit_vp_spca


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

ARXIV_ID = "1804.00341"
ARXIV_VERSION = 3
ARXIV_PDF_SHA256 = (
    "e35b37476492b589099f4997fdeaa486d19249fb4bc2c85339ab2fe1c02101e0"
)
LEGACY_SOURCE_COMMIT = "e01c02f7cbe1eaedc8f5262a4ca9ec4f54db84fa"
LEGACY_SOURCE_SHA256 = {
    "spca_multiscale_video.py": (
        "e6222e19b2423a63f7ff4fce8eb370b4037372d40668c09b25c3702ca686bc56"
    ),
    "spca_video_comparison.py": (
        "d06fc8c78562baa5b4adeda0014d1983990c920d615f9faba172802af638c2e7"
    ),
}
REPORT_NAME = "paper_video_report.json"
FRAME_ERRORS_NAME = "paper_video_full_frame_errors.csv"
FIGURE5_NAME = "figure5_clean_modes_and_dynamics.png"
FIGURE6_NAME = "figure6_robust_decomposition.png"
MANIFEST_NAME = "paper_video_output_manifest.json"
_REQUIRED_PAYLOAD_NAMES = frozenset(
    {FIGURE5_NAME, FIGURE6_NAME, FRAME_ERRORS_NAME, REPORT_NAME}
)
_REQUIRED_OUTPUT_NAMES = _REQUIRED_PAYLOAD_NAMES | {MANIFEST_NAME}


@dataclass(frozen=True)
class PaperVideoConfig:
    """Frozen Section 6.1 reproduction configuration.

    ``robust_kappa`` is expressed after division by the largest singular value,
    matching the normalization used by the historical robust demo.  It is
    passed explicitly to the current solver and never re-estimated.
    """

    grid_size: int = 200
    n_timesteps: int = 300
    total_time: float = 150.0
    n_components: int = 3
    figure5_alpha: float = 9.0
    robust_alpha: float = 0.012
    robust_kappa: float = 0.005
    randomized_oversample: int = 100
    randomized_power_iters: int = 4
    randomized_full_refine_iters: int = 0
    randomized_seed: int = 42
    corruption_fraction: float = 0.05
    corruption_scale: float = 3.0
    corruption_seed: int = 123
    max_iter: int = 500
    tol: float = 1e-6
    figure6_frame_index: int = 100
    dpi: int = 150


@dataclass(frozen=True)
class CleanModeTimeDataset:
    """Clean Figure 5 data in the paper's pixels-by-snapshots orientation."""

    clean: FloatArray
    spatial_modes: FloatArray
    temporal_scores: FloatArray
    raw_spatial_modes: FloatArray
    raw_temporal_scores: FloatArray
    time: FloatArray


@dataclass(frozen=True)
class RobustDecompositionDataset:
    """Figure 6 clean signal plus the historical sparse gross corruption."""

    clean: FloatArray
    observed: FloatArray
    corruption: FloatArray
    corruption_support: BoolArray
    time: FloatArray


@dataclass(frozen=True)
class NativeFactorFit:
    """A native factorization; no field has been fitted to evaluation truth."""

    method: str
    axes: FloatArray
    loadings: FloatArray
    spatial_modes: FloatArray
    temporal_dynamics: FloatArray
    reconstruction: FloatArray
    runtime_seconds: float
    converged: bool
    n_iter: int
    final_stationarity: float | None
    status: str
    objectives: tuple[float, ...]
    outliers: FloatArray | None = None
    extra: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PaperVideoResult:
    """In-memory result used by the writer and focused tests."""

    config: PaperVideoConfig
    figure5_data: CleanModeTimeDataset
    figure6_data: RobustDecompositionDataset
    figure5_fits: Mapping[str, NativeFactorFit]
    figure6_fits: Mapping[str, NativeFactorFit]


def _validate_config(config: PaperVideoConfig) -> None:
    integer_values = {
        "grid_size": config.grid_size,
        "n_timesteps": config.n_timesteps,
        "n_components": config.n_components,
        "corruption_seed": config.corruption_seed,
        "randomized_oversample": config.randomized_oversample,
        "randomized_power_iters": config.randomized_power_iters,
        "randomized_full_refine_iters": config.randomized_full_refine_iters,
        "randomized_seed": config.randomized_seed,
        "max_iter": config.max_iter,
        "figure6_frame_index": config.figure6_frame_index,
        "dpi": config.dpi,
    }
    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if config.n_timesteps <= config.n_components:
        raise ValueError("n_timesteps must exceed n_components")
    if config.n_components != 3:
        raise ValueError("the Section 6.1 construction has exactly 3 components")
    if min(
        config.corruption_seed,
        config.randomized_oversample,
        config.randomized_power_iters,
        config.randomized_full_refine_iters,
        config.randomized_seed,
    ) < 0:
        raise ValueError(
            "seeds, oversampling, and randomized iteration values must be non-negative"
        )
    if config.max_iter <= 0 or config.dpi <= 0:
        raise ValueError("max_iter and dpi must be positive")
    if not 0 <= config.figure6_frame_index < config.n_timesteps:
        raise ValueError("figure6_frame_index is outside the video")
    finite = {
        "total_time": config.total_time,
        "figure5_alpha": config.figure5_alpha,
        "robust_alpha": config.robust_alpha,
        "robust_kappa": config.robust_kappa,
        "corruption_fraction": config.corruption_fraction,
        "corruption_scale": config.corruption_scale,
        "tol": config.tol,
    }
    for name, value in finite.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if config.figure5_alpha < 0.0 or config.robust_alpha < 0.0:
        raise ValueError("regularization parameters must be non-negative")
    if config.robust_kappa <= 0.0 or config.tol <= 0.0:
        raise ValueError("robust_kappa and tol must be positive")
    if not 0.0 <= config.corruption_fraction < 1.0:
        raise ValueError("corruption_fraction must lie in [0, 1)")
    if config.corruption_scale < 0.0:
        raise ValueError("corruption_scale must be non-negative")


def _smooth_window(
    time: FloatArray, start: float, stop: float, *, ramp: float
) -> FloatArray:
    left = 1.0 / (1.0 + np.exp(-(time - start) / ramp))
    right = 1.0 / (1.0 + np.exp(-(time - stop) / ramp))
    return np.clip(left - right, 0.0, 1.0)


def _legacy_spatial_modes(grid_size: int) -> FloatArray:
    coordinate = np.linspace(0.0, 1.0, grid_size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="xy")

    def gaussian_blob(center_x: float, center_y: float, sigma: float) -> FloatArray:
        blob = np.exp(
            -(
                (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            )
            / (2.0 * sigma**2)
        )
        return np.asarray(blob / max(float(blob.max()), 1e-12), dtype=np.float64)

    images = (
        gaussian_blob(0.55, 0.55, 0.06),
        gaussian_blob(0.30, 0.20, 0.025),
        gaussian_blob(0.78, 0.80, 0.025),
    )
    return np.ascontiguousarray(
        np.column_stack([image.reshape(-1) for image in images]),
        dtype=np.float64,
    )


def _canonicalize_truth_factor(
    raw_modes: FloatArray,
    raw_scores: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Choose the solver's unit-temporal-dynamics gauge from truth alone.

    The paper-oriented factorization is ``X = Phi @ A.T`` with orthonormal
    temporal columns ``A``.  Normalizing the truth dynamics and moving their
    norms into ``Phi`` makes native solver amplitudes directly comparable
    without fitting a scale from an estimate to truth.
    """

    norms = np.linalg.norm(raw_scores, axis=0)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise RuntimeError("truth contains a zero temporal dynamic")
    return raw_modes * norms, raw_scores / norms


def generate_figure5_dataset(config: PaperVideoConfig) -> CleanModeTimeDataset:
    """Port the historical smooth-window Figure 5 construction exactly."""

    _validate_config(config)
    raw_modes = _legacy_spatial_modes(config.grid_size)
    time = np.linspace(0.0, config.total_time, config.n_timesteps)
    raw_scores = np.zeros((config.n_timesteps, 3), dtype=np.float64)
    window1 = _smooth_window(time, 0.0, config.total_time, ramp=6.0)
    raw_scores[:, 0] = 2.0 * np.sin(2.0 * np.pi * 0.08 * time) * window1
    plateau = _smooth_window(time, 45.0, 65.0, ramp=2.0)
    burst2 = _smooth_window(time, 65.0, 105.0, ramp=2.0)
    raw_scores[:, 1] = 0.8 * plateau + 1.8 * np.sin(
        2.0 * np.pi * 0.055 * time + 0.35 * np.pi
    ) * burst2
    burst3 = _smooth_window(time, 100.0, config.total_time, ramp=2.0)
    raw_scores[:, 2] = 2.0 * np.sin(
        2.0 * np.pi * 0.12 * time + 0.1 * np.pi
    ) * burst3

    centered_scores = raw_scores - raw_scores.mean(axis=0, keepdims=True)
    clean = np.ascontiguousarray(raw_modes @ centered_scores.T)
    modes, scores = _canonicalize_truth_factor(raw_modes, centered_scores)
    return CleanModeTimeDataset(
        clean=clean,
        spatial_modes=np.ascontiguousarray(modes),
        temporal_scores=np.ascontiguousarray(scores),
        raw_spatial_modes=raw_modes,
        raw_temporal_scores=np.ascontiguousarray(raw_scores),
        time=np.asarray(time, dtype=np.float64),
    )


def generate_figure6_dataset(config: PaperVideoConfig) -> RobustDecompositionDataset:
    """Port the classic video and one-shot sparse corruption from the old demo."""

    _validate_config(config)
    modes = _legacy_spatial_modes(config.grid_size)
    time = np.linspace(0.0, config.total_time, config.n_timesteps)
    raw_scores = np.zeros((config.n_timesteps, 3), dtype=np.float64)
    raw_scores[:, 0] = 2.0 * np.sin(2.0 * np.pi * 0.20 * time)
    mask2 = (time >= 50.0) & (time <= 100.0)
    raw_scores[mask2, 1] = 2.0 * np.sin(2.0 * np.pi * 0.05 * time[mask2])
    mask3 = (time >= 80.0) & (time <= 120.0)
    raw_scores[mask3, 2] = 2.0 * np.sin(2.0 * np.pi * 0.30 * time[mask3])
    scores = raw_scores - raw_scores.mean(axis=0, keepdims=True)
    clean = np.ascontiguousarray(modes @ scores.T)

    rng = np.random.default_rng(config.corruption_seed)
    support = rng.random(clean.shape) < config.corruption_fraction
    corruption = np.zeros_like(clean)
    corruption[support] = rng.normal(
        0.0,
        config.corruption_scale,
        size=int(np.count_nonzero(support)),
    )
    # A continuous draw can theoretically be exactly zero.  Score the actual
    # increment that entered X rather than an RNG pre-mask.
    support = corruption != 0.0
    observed = np.ascontiguousarray(clean + corruption)
    return RobustDecompositionDataset(
        clean=clean,
        observed=observed,
        corruption=np.ascontiguousarray(corruption),
        corruption_support=np.ascontiguousarray(support),
        time=np.asarray(time, dtype=np.float64),
    )


def _pca_fit(data: FloatArray, *, method: str) -> tuple[NativeFactorFit, float]:
    start = perf_counter()
    _, singular_values, right_t = linalg.svd(
        data,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    runtime = perf_counter() - start
    axes = np.asarray(right_t[:3].T, dtype=np.float64)
    spatial_modes = data @ axes
    reconstruction = spatial_modes @ axes.T
    sigma_max = float(singular_values[0]) if singular_values.size else 1.0
    return (
        NativeFactorFit(
            method=method,
            axes=axes,
            loadings=axes.copy(),
            spatial_modes=np.asarray(spatial_modes, dtype=np.float64),
            temporal_dynamics=axes.copy(),
            reconstruction=np.asarray(reconstruction, dtype=np.float64),
            runtime_seconds=float(runtime),
            converged=True,
            n_iter=1,
            final_stationarity=None,
            status="closed-form truncated SVD",
            objectives=(),
            extra={"largest_singular_value": sigma_max},
        ),
        sigma_max,
    )


def _vp_fit(
    data: FloatArray,
    pca: NativeFactorFit,
    sigma_max: float,
    config: PaperVideoConfig,
) -> NativeFactorFit:
    step_size = 1.0 if sigma_max == 0.0 else 1.0 / sigma_max**2
    start = perf_counter()
    result: VPSPCAResult = fit_vp_spca(
        data,
        config.n_components,
        alpha=config.figure5_alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        step_size=step_size,
        initial_B=pca.axes,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - start
    spatial_modes = data @ result.B
    return NativeFactorFit(
        method="VP-SPCA",
        axes=result.A,
        loadings=result.B,
        spatial_modes=np.asarray(spatial_modes, dtype=np.float64),
        temporal_dynamics=np.asarray(result.A, dtype=np.float64),
        reconstruction=np.asarray(spatial_modes @ result.A.T, dtype=np.float64),
        runtime_seconds=float(runtime),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        final_stationarity=(
            float(result.stationarity[-1]) if result.stationarity else None
        ),
        status=result.status,
        objectives=tuple(float(value) for value in result.objectives),
        extra={"alpha": float(result.alpha), "step_size_policy": "1/sigma_max^2"},
    )


def _randomized_fit(
    data: FloatArray,
    config: PaperVideoConfig,
) -> NativeFactorFit:
    start = perf_counter()
    result: RandomizedVPSPCAResult = fit_randomized_vp_spca(
        data,
        config.n_components,
        alpha=config.figure5_alpha,
        oversample=config.randomized_oversample,
        n_power_iter=config.randomized_power_iters,
        max_iter=config.max_iter,
        tol=config.tol,
        finalize_full_iters=config.randomized_full_refine_iters,
        random_state=config.randomized_seed,
        verbose=False,
    )
    runtime = perf_counter() - start
    spatial_modes = data @ result.B
    certificate = None if result.certificate is None else asdict(result.certificate)
    return NativeFactorFit(
        method="RND-VP-SPCA",
        axes=result.A,
        loadings=result.B,
        spatial_modes=np.asarray(spatial_modes, dtype=np.float64),
        temporal_dynamics=np.asarray(result.A, dtype=np.float64),
        reconstruction=np.asarray(spatial_modes @ result.A.T, dtype=np.float64),
        runtime_seconds=float(runtime),
        converged=bool(result.full_converged),
        n_iter=int(result.n_iter),
        final_stationarity=float(result.full_stationarity),
        status=result.status,
        # Sketch and full objective traces have different scopes. Do not pool
        # them into a monotonicity claim; retain the certified diagnostics.
        objectives=(),
        extra={
            "alpha": float(result.alpha),
            "sketch_converged": result.sketch_converged,
            "full_converged": bool(result.full_converged),
            "sketch_residual": float(result.sketch_residual),
            "sketch_residual_relative": float(result.sketch_residual_relative),
            "full_objective": float(result.full_objective),
            "full_stationarity": float(result.full_stationarity),
            "full_diagnostic_step_size": float(result.full_diagnostic_step_size),
            "full_diagnostic_step_policy": result.full_diagnostic_step_policy,
            "used_full_gradient": bool(result.used_full_gradient),
            "sketch_iterations": int(result.sketch_iterations),
            "full_refinement_iterations": int(result.full_refinement_iterations),
            "timings": dict(result.timings),
            "certificate": certificate,
        },
    )


def _robust_fit(
    data: RobustDecompositionDataset,
    config: PaperVideoConfig,
) -> tuple[NativeFactorFit, NativeFactorFit]:
    # Normalize once as the old robust demo did, then pass both its PCA start
    # and the exact normalized step size to avoid a redundant decomposition.
    normalized_pca, sigma_max = _pca_fit(data.observed, method="Corrupted-PCA")
    scale = max(sigma_max, np.finfo(np.float64).eps)
    normalized = data.observed / scale

    start = perf_counter()
    result: RobustVPSPCAResult = fit_robust_vp_spca(
        normalized,
        config.n_components,
        alpha=config.robust_alpha,
        kappa=config.robust_kappa,
        max_iter=config.max_iter,
        tol=config.tol,
        step_size=1.0,
        initial_B=normalized_pca.axes,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - start
    normalized_spatial_modes = normalized @ result.B
    reconstruction = scale * (normalized_spatial_modes @ result.A.T)
    outliers = scale * result.S
    robust = NativeFactorFit(
        method="Robust-VP-SPCA",
        axes=result.A,
        loadings=result.B,
        spatial_modes=np.asarray(scale * normalized_spatial_modes, dtype=np.float64),
        temporal_dynamics=np.asarray(result.A, dtype=np.float64),
        reconstruction=np.asarray(reconstruction, dtype=np.float64),
        runtime_seconds=float(runtime),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        final_stationarity=(
            float(result.stationarity[-1]) if result.stationarity else None
        ),
        status=result.status,
        objectives=tuple(float(value) for value in result.objectives),
        outliers=np.asarray(outliers, dtype=np.float64),
        extra={
            "alpha_normalized": float(result.alpha),
            "kappa_normalized": float(result.kappa),
            "kappa_raw_units": float(scale * result.kappa),
            "normalization_sigma_max": float(scale),
            "kappa_policy": "explicit_fixed",
            "outlier_display": "native_S_without_threshold",
        },
    )
    return normalized_pca, robust


def run_paper_video_reproduction(
    config: PaperVideoConfig | None = None,
) -> PaperVideoResult:
    """Fit the clean and robust paper panels on separately generated data."""

    resolved = PaperVideoConfig() if config is None else config
    _validate_config(resolved)
    figure5_data = generate_figure5_dataset(resolved)
    pca, sigma_max = _pca_fit(figure5_data.clean, method="PCA")
    vp = _vp_fit(figure5_data.clean, pca, sigma_max, resolved)
    randomized = _randomized_fit(figure5_data.clean, resolved)

    figure6_data = generate_figure6_dataset(resolved)
    corrupt_pca, robust = _robust_fit(figure6_data, resolved)
    return PaperVideoResult(
        config=resolved,
        figure5_data=figure5_data,
        figure6_data=figure6_data,
        figure5_fits={
            "VP-SPCA": vp,
            "RND-VP-SPCA": randomized,
            "PCA": pca,
        },
        figure6_fits={"Robust-VP-SPCA": robust, "Corrupted-PCA": corrupt_pca},
    )


def align_factor_pair_sign_permutation_only(
    truth_modes: FloatArray,
    estimated_modes: FloatArray,
    estimated_scores: FloatArray,
) -> tuple[FloatArray, FloatArray, tuple[int, ...], tuple[int, ...]]:
    """Align columns without changing any estimated magnitude."""

    truth = np.asarray(truth_modes, dtype=np.float64)
    modes = np.asarray(estimated_modes, dtype=np.float64)
    scores = np.asarray(estimated_scores, dtype=np.float64)
    if truth.ndim != 2 or modes.shape != truth.shape:
        raise ValueError("truth_modes and estimated_modes must have equal 2-D shape")
    if scores.ndim != 2 or scores.shape[1] != modes.shape[1]:
        raise ValueError("estimated_scores has an incompatible component count")
    truth_norm = np.linalg.norm(truth, axis=0)
    mode_norm = np.linalg.norm(modes, axis=0)
    denominator = truth_norm[:, None] * mode_norm[None, :]
    correlations = np.divide(
        truth.T @ modes,
        denominator,
        out=np.zeros((truth.shape[1], modes.shape[1]), dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )
    reference_indices, estimated_indices = linear_sum_assignment(
        -np.abs(correlations)
    )
    aligned_modes = np.zeros_like(modes)
    aligned_scores = np.zeros_like(scores)
    permutation = [-1] * truth.shape[1]
    signs = [1] * truth.shape[1]
    for reference, estimated in zip(reference_indices, estimated_indices):
        sign = 1 if correlations[reference, estimated] >= 0.0 else -1
        aligned_modes[:, reference] = sign * modes[:, estimated]
        aligned_scores[:, reference] = sign * scores[:, estimated]
        permutation[reference] = int(estimated)
        signs[reference] = int(sign)
    return aligned_modes, aligned_scores, tuple(permutation), tuple(signs)


def normalized_full_frame_errors(
    clean: FloatArray,
    reconstruction: FloatArray,
) -> FloatArray:
    """Return one normalized RMSE per column-frame, with no frame selection."""

    clean_array = np.asarray(clean, dtype=np.float64)
    estimate = np.asarray(reconstruction, dtype=np.float64)
    if clean_array.ndim != 2 or estimate.shape != clean_array.shape:
        raise ValueError("clean and reconstruction must have equal 2-D shape")
    if not np.isfinite(clean_array).all() or not np.isfinite(estimate).all():
        raise ValueError("frame-error inputs must be finite")
    denominator = float(np.sqrt(np.mean(clean_array * clean_array)))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("clean video RMS must be positive")
    return np.sqrt(np.mean((estimate - clean_array) ** 2, axis=0)) / denominator


def summarize_full_frame_errors(
    values: FloatArray,
    time: FloatArray,
) -> dict[str, float | int]:
    """Summarize all frames and retain the first maximum as the worst frame."""

    errors = np.asarray(values, dtype=np.float64)
    time_array = np.asarray(time, dtype=np.float64)
    if errors.ndim != 1 or errors.shape != time_array.shape or errors.size == 0:
        raise ValueError("errors and time must be equal nonempty vectors")
    if not np.isfinite(errors).all() or not np.isfinite(time_array).all():
        raise ValueError("errors and time must be finite")
    worst = int(np.argmax(errors))
    return {
        "n_frames": int(errors.size),
        "median": float(np.median(errors)),
        "p95": float(np.percentile(errors, 95.0)),
        "max": float(errors[worst]),
        "worst_frame_index": worst,
        "worst_frame_time": float(time_array[worst]),
    }


def _paired_cosines(reference: FloatArray, estimate: FloatArray) -> list[float]:
    if reference.shape != estimate.shape:
        raise ValueError("paired cosine inputs must have equal shape")
    numerator = np.sum(reference * estimate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(
        estimate, axis=0
    )
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )
    return [float(value) for value in np.clip(values, -1.0, 1.0)]


def _figure5_recovery_metrics(
    result: PaperVideoResult,
    fit: NativeFactorFit,
) -> dict[str, Any]:
    aligned_modes, aligned_dynamics, permutation, signs = (
        align_factor_pair_sign_permutation_only(
            result.figure5_data.spatial_modes,
            fit.spatial_modes,
            fit.temporal_dynamics,
        )
    )
    spatial = _paired_cosines(result.figure5_data.spatial_modes, aligned_modes)
    temporal = _paired_cosines(
        result.figure5_data.temporal_scores, aligned_dynamics
    )
    return {
        "spatial_mode_cosine_per_component": spatial,
        "spatial_mode_mean_cosine": float(np.mean(spatial)),
        "temporal_dynamics_cosine_per_component": temporal,
        "temporal_dynamics_mean_cosine": float(np.mean(temporal)),
        "component_permutation": list(permutation),
        "component_signs": list(signs),
        "amplitude_refit": False,
    }


def _localization_metrics(
    truth_modes: FloatArray,
    aligned_modes: FloatArray,
    *,
    relative_support_threshold: float,
) -> dict[str, Any]:
    """Measure energy leakage outside each generating mode's declared region."""

    truth = np.asarray(truth_modes, dtype=np.float64)
    estimate = np.asarray(aligned_modes, dtype=np.float64)
    if truth.shape != estimate.shape or truth.ndim != 2:
        raise ValueError("localization inputs must have equal 2-D shape")
    if not 0.0 < relative_support_threshold < 1.0:
        raise ValueError("relative_support_threshold must lie in (0, 1)")
    maxima = np.max(np.abs(truth), axis=0)
    support = np.abs(truth) >= relative_support_threshold * maxima[None, :]
    off_support: list[float] = []
    cross_talk: list[float] = []
    for component in range(truth.shape[1]):
        energy = estimate[:, component] ** 2
        total = max(float(np.sum(energy)), np.finfo(np.float64).eps)
        off_support.append(float(np.sum(energy[~support[:, component]]) / total))
        other_support = np.any(
            np.delete(support, component, axis=1), axis=1
        ) & ~support[:, component]
        cross_talk.append(float(np.sum(energy[other_support]) / total))
    return {
        "relative_truth_support_threshold": float(relative_support_threshold),
        "truth_support_pixels_per_component": [
            int(value) for value in np.sum(support, axis=0)
        ],
        "off_support_energy_fraction_per_component": off_support,
        "off_support_energy_fraction_mean": float(np.mean(off_support)),
        "cross_talk_energy_fraction_per_component": cross_talk,
        "cross_talk_energy_fraction_mean": float(np.mean(cross_talk)),
    }


def _figure5_effect_gate(result: PaperVideoResult) -> dict[str, Any]:
    """Gate a localization claim against PCA instead of global cosine alone."""

    support_threshold = 0.05
    required_absolute_improvement = 0.01
    maximum_clean_error_increase = 0.01
    metrics: dict[str, Any] = {}
    for method, fit in result.figure5_fits.items():
        aligned_modes, _, _, _ = align_factor_pair_sign_permutation_only(
            result.figure5_data.spatial_modes,
            fit.spatial_modes,
            fit.temporal_dynamics,
        )
        metrics[method] = {
            **_localization_metrics(
                result.figure5_data.spatial_modes,
                aligned_modes,
                relative_support_threshold=support_threshold,
            ),
            "clean_relative_reconstruction_error": relative_reconstruction_error(
                result.figure5_data.clean, fit.reconstruction
            ),
        }
    baseline = metrics["PCA"]
    comparisons: dict[str, Any] = {}
    for method in ("VP-SPCA", "RND-VP-SPCA"):
        candidate = metrics[method]
        off_support_improvement = float(
            baseline["off_support_energy_fraction_mean"]
            - candidate["off_support_energy_fraction_mean"]
        )
        cross_talk_improvement = float(
            baseline["cross_talk_energy_fraction_mean"]
            - candidate["cross_talk_energy_fraction_mean"]
        )
        clean_error_increase = float(
            candidate["clean_relative_reconstruction_error"]
            - baseline["clean_relative_reconstruction_error"]
        )
        comparisons[method] = {
            "off_support_absolute_improvement_vs_pca": off_support_improvement,
            "cross_talk_absolute_improvement_vs_pca": cross_talk_improvement,
            "clean_error_increase_vs_pca": clean_error_increase,
            "qualified": bool(
                off_support_improvement >= required_absolute_improvement
                and cross_talk_improvement >= required_absolute_improvement
                and clean_error_increase <= maximum_clean_error_increase
            ),
        }
    return {
        "gate_version": 1,
        "claim": "sparse modes improve localization over PCA",
        "global_cosine_alone_is_sufficient": False,
        "relative_truth_support_threshold": support_threshold,
        "required_absolute_improvement": required_absolute_improvement,
        "maximum_clean_error_increase": maximum_clean_error_increase,
        "methods": metrics,
        "comparisons": comparisons,
        "primary_vp_claim_qualified": bool(comparisons["VP-SPCA"]["qualified"]),
    }


def _objective_is_monotone(values: Sequence[float]) -> bool:
    if len(values) < 2:
        return True
    array = np.asarray(values, dtype=np.float64)
    tolerance = 1e-10 * np.maximum(1.0, np.abs(array[:-1]))
    return bool(np.all(np.diff(array) <= tolerance))


def _strict_fit_record(fit: NativeFactorFit, tol: float) -> dict[str, Any]:
    stationarity_met = (
        fit.final_stationarity is None
        or float(fit.final_stationarity) <= float(tol)
    )
    monotone = _objective_is_monotone(fit.objectives)
    qualified = bool(fit.converged and stationarity_met and monotone)
    return {
        "qualified": qualified,
        "converged": bool(fit.converged),
        "status": fit.status,
        "n_iter": int(fit.n_iter),
        "final_stationarity": fit.final_stationarity,
        "stationarity_tolerance": float(tol),
        "stationarity_met": bool(stationarity_met),
        "objective_monotone_with_roundoff_tolerance": monotone,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


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


def _loaded_repo_module_provenance(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Bind every loaded local ``rspca`` module, including eager imports."""

    direct_roles = {
        "src/rspca/datasets.py": "data_hashing_and_runtime_provenance",
        "src/rspca/metrics.py": "evaluation_metrics",
        "src/rspca/randomized.py": "randomized_figure5_solver",
        "src/rspca/robust.py": "robust_figure6_solver",
        "src/rspca/vp.py": "deterministic_figure5_solver",
        "experiments/video_decomposition_reference.py": "experiment_runner",
    }
    by_path: dict[str, dict[str, Any]] = {}
    runner_path = Path(__file__).resolve()
    for module_name, module in sorted(sys.modules.items()):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            continue
        if path != runner_path and not (
            module_name == "rspca" or module_name.startswith("rspca.")
        ):
            continue
        if not path.is_file() or path.suffix != ".py":
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
            {
                "sha256": _file_sha256(path),
                "role": role,
                "module_names": [],
            },
        )
        entry["module_names"].append(module_name)
    required_direct = set(direct_roles) | {"src/rspca/__init__.py"}
    missing = sorted(required_direct - set(by_path))
    if missing:
        raise RuntimeError(f"executed source provenance is incomplete: {missing}")
    for entry in by_path.values():
        entry["module_names"] = sorted(set(entry["module_names"]))
    return {relative: by_path[relative] for relative in sorted(by_path)}


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    loaded_modules = _loaded_repo_module_provenance(repo_root)
    source_hashes = {
        relative: record["sha256"] for relative, record in loaded_modules.items()
    }
    legacy_verified: dict[str, bool | None] = {}
    for relative, expected in LEGACY_SOURCE_SHA256.items():
        content = _git_bytes(repo_root, ["show", f"{LEGACY_SOURCE_COMMIT}:{relative}"])
        legacy_verified[relative] = (
            None if content is None else sha256(content).hexdigest() == expected
        )
    status = _git_output(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    return {
        "paper": {
            "title": "Sparse Principal Component Analysis via Variable Projection",
            "arxiv_id": ARXIV_ID,
            "arxiv_version": ARXIV_VERSION,
            "official_pdf_sha256": ARXIV_PDF_SHA256,
            "section": "6.1 Multiscale Video Example",
        },
        "legacy_implementation": {
            "commit": LEGACY_SOURCE_COMMIT,
            "sha256": dict(LEGACY_SOURCE_SHA256),
            "verified_from_local_git_objects": legacy_verified,
        },
        "current_source_sha256": source_hashes,
        "loaded_repo_modules": loaded_modules,
        "package_eager_imports_bound": True,
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "runtime_versions": runtime_versions(),
    }


def _data_provenance(result: PaperVideoResult) -> dict[str, Any]:
    fig5 = result.figure5_data
    fig6 = result.figure6_data
    return {
        "orientation": "spatial_pixels_x_time_snapshots",
        "protocol_identity": (
            "legacy-derived Section 6.1 paper-layout reproduction; distinct from "
            "the current time_samples_x_pixel_features benchmark"
        ),
        "figure5": {
            "shape": list(fig5.clean.shape),
            "array_sha256": {
                "clean": array_sha256(fig5.clean),
                "spatial_modes": array_sha256(fig5.spatial_modes),
                "temporal_scores": array_sha256(fig5.temporal_scores),
                "raw_spatial_modes": array_sha256(fig5.raw_spatial_modes),
                "raw_temporal_scores": array_sha256(fig5.raw_temporal_scores),
                "time": array_sha256(fig5.time),
            },
        },
        "figure6": {
            "shape": list(fig6.observed.shape),
            "corruption_distribution": (
                "Gaussian-valued impulses on an iid Bernoulli mask, matching "
                "legacy Git code; not asserted byte-identical to paper salt-and-pepper data"
            ),
            "injected_corruption_cells": int(
                np.count_nonzero(fig6.corruption_support)
            ),
            "realized_corruption_fraction": float(
                np.mean(fig6.corruption_support)
            ),
            "array_sha256": {
                "clean": array_sha256(fig6.clean),
                "observed": array_sha256(fig6.observed),
                "corruption": array_sha256(fig6.corruption),
                "corruption_support": array_sha256(fig6.corruption_support),
                "time": array_sha256(fig6.time),
            },
        },
    }


def _all_frame_errors(
    result: PaperVideoResult,
) -> dict[str, tuple[FloatArray, FloatArray]]:
    output: dict[str, tuple[FloatArray, FloatArray]] = {}
    for method, fit in result.figure5_fits.items():
        output[f"figure5/{method}"] = (
            result.figure5_data.time,
            normalized_full_frame_errors(
                result.figure5_data.clean, fit.reconstruction
            ),
        )
    for method, fit in result.figure6_fits.items():
        output[f"figure6/{method}"] = (
            result.figure6_data.time,
            normalized_full_frame_errors(
                result.figure6_data.clean, fit.reconstruction
            ),
        )
    return output


def _write_frame_errors(
    errors: Mapping[str, tuple[FloatArray, FloatArray]], path: Path
) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["panel", "method", "frame_index", "time", "normalized_rmse"]
        )
        for key in sorted(errors):
            panel, method = key.split("/", 1)
            time, values = errors[key]
            for index, (timestamp, error) in enumerate(zip(time, values)):
                writer.writerow(
                    [panel, method, index, f"{timestamp:.17g}", f"{error:.17g}"]
                )


def _write_figure5(result: PaperVideoResult, path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = result.figure5_data
    aligned: dict[str, tuple[FloatArray, FloatArray, tuple[int, ...], tuple[int, ...]]] = {}
    for method, fit in result.figure5_fits.items():
        aligned[method] = align_factor_pair_sign_permutation_only(
            data.spatial_modes, fit.spatial_modes, fit.temporal_dynamics
        )

    row_items = (
        ("Truth", data.spatial_modes, data.temporal_scores),
        ("VP-SPCA", aligned["VP-SPCA"][0], aligned["VP-SPCA"][1]),
        (
            "RND-VP-SPCA",
            aligned["RND-VP-SPCA"][0],
            aligned["RND-VP-SPCA"][1],
        ),
        ("PCA", aligned["PCA"][0], aligned["PCA"][1]),
    )
    spatial_limits = []
    temporal_limits = []
    for component in range(3):
        spatial_limits.append(
            max(
                float(np.max(np.abs(modes[:, component])))
                for _, modes, _ in row_items
            )
        )
        temporal_limits.append(
            max(
                float(np.max(np.abs(scores[:, component])))
                for _, _, scores in row_items
            )
        )

    figure, axes = plt.subplots(4, 6, figsize=(20, 13), constrained_layout=True)
    for row, (label, modes, scores) in enumerate(row_items):
        for component in range(3):
            limit = max(spatial_limits[component], np.finfo(float).eps)
            axis = axes[row, component]
            axis.imshow(
                modes[:, component].reshape(result.config.grid_size, -1),
                cmap="magma",
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
            )
            axis.set_title(f"{label} spatial mode {component + 1}")
            axis.axis("off")

            time_axis = axes[row, component + 3]
            if label != "Truth":
                time_axis.plot(
                    data.time,
                    data.temporal_scores[:, component],
                    "--",
                    color="black",
                    linewidth=1.0,
                    label="truth",
                )
            time_axis.plot(
                data.time,
                scores[:, component],
                color={
                    "Truth": "#b45309",
                    "VP-SPCA": "#2563eb",
                    "RND-VP-SPCA": "#15803d",
                    "PCA": "#6b7280",
                }[label],
                linewidth=1.2,
                label=label.lower(),
            )
            time_limit = max(temporal_limits[component], np.finfo(float).eps)
            time_axis.set_ylim(-1.05 * time_limit, 1.05 * time_limit)
            time_axis.set_xlim(float(data.time[0]), float(data.time[-1]))
            time_axis.set_title(f"{label} dynamics {component + 1}")
            time_axis.grid(alpha=0.25)
            if row == 3:
                time_axis.set_xlabel("time (s)")
            if component == 0:
                time_axis.set_ylabel("native coefficient")
            if label != "Truth":
                time_axis.legend(fontsize=7, loc="upper right")
    figure.suptitle(
        "Figure 5 reproduction: sign/permutation alignment only; no truth-amplitude fit",
        fontsize=14,
    )
    figure.savefig(path, dpi=result.config.dpi)
    plt.close(figure)
    return {
        "alignment": {
            method: {
                "permutation": list(values[2]),
                "signs": list(values[3]),
            }
            for method, values in aligned.items()
        },
        "spatial_limits": spatial_limits,
        "temporal_limits": temporal_limits,
        "post_hoc_amplitude_fit": False,
    }


def _write_figure6(result: PaperVideoResult, path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    data = result.figure6_data
    robust = result.figure6_fits["Robust-VP-SPCA"]
    if robust.outliers is None:
        raise RuntimeError("robust fit did not return its native outlier layer")
    index = result.config.figure6_frame_index
    clean_limit = max(float(np.max(np.abs(data.clean))), np.finfo(float).eps)
    outlier_limit = max(
        float(np.max(np.abs(data.corruption))),
        float(np.max(np.abs(robust.outliers))),
        np.finfo(float).eps,
    )
    panels = (
        ("(a) Truth", data.clean[:, index], "magma", -clean_limit, clean_limit),
        (
            "(b) Corrupted",
            data.observed[:, index],
            "magma",
            -clean_limit,
            clean_limit,
        ),
        (
            "(c) Reconstruction",
            robust.reconstruction[:, index],
            "magma",
            -clean_limit,
            clean_limit,
        ),
        (
            "(d) Native outliers S",
            robust.outliers[:, index],
            "seismic",
            -outlier_limit,
            outlier_limit,
        ),
    )
    figure, axes = plt.subplots(1, 4, figsize=(18, 4.8), constrained_layout=True)
    for axis, (title, values, cmap, vmin, vmax) in zip(axes, panels):
        image = axis.imshow(
            values.reshape(result.config.grid_size, -1),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        axis.set_title(title)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(
        f"Figure 6 reproduction, frame {index}: fixed kappa, native S",
        fontsize=14,
    )
    figure.savefig(path, dpi=result.config.dpi)
    plt.close(figure)
    return {
        "frame_index": int(index),
        "frame_time": float(data.time[index]),
        "clean_shared_limit": clean_limit,
        "outlier_shared_limit": outlier_limit,
        "outlier_post_threshold": False,
        "outlier_array_sha256": array_sha256(robust.outliers),
    }


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
    return value


def _report_payload(
    result: PaperVideoResult,
    *,
    repo_root: Path,
    errors: Mapping[str, tuple[FloatArray, FloatArray]],
    figure5_display: Mapping[str, Any],
    figure6_display: Mapping[str, Any],
) -> dict[str, Any]:
    strict = {
        key: _strict_fit_record(fit, result.config.tol)
        for key, fit in {
            "figure5/VP-SPCA": result.figure5_fits["VP-SPCA"],
            "figure5/RND-VP-SPCA": result.figure5_fits["RND-VP-SPCA"],
            "figure5/PCA": result.figure5_fits["PCA"],
            "figure6/Robust-VP-SPCA": result.figure6_fits["Robust-VP-SPCA"],
            "figure6/Corrupted-PCA": result.figure6_fits["Corrupted-PCA"],
        }.items()
    }
    robust = result.figure6_fits["Robust-VP-SPCA"]
    if robust.outliers is None:
        raise RuntimeError("robust fit lacks an outlier layer")
    corruption_budget = int(np.count_nonzero(result.figure6_data.corruption_support))
    detection = outlier_detection_metrics(
        result.figure6_data.corruption_support,
        scores=np.abs(robust.outliers),
        top_k=corruption_budget,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": asdict(result.config),
        "paper_scale": bool(
            result.config.grid_size == 200
            and result.config.n_timesteps == 300
            and result.config.total_time == 150.0
        ),
        "provenance": _source_provenance(repo_root),
        "data": _data_provenance(result),
        "protocol": {
            "figure5_and_figure6_fitted_separately": True,
            "orientation": "spatial_pixels_x_time_snapshots",
            "spatial_mode_definition": "Phi = X @ B",
            "not_the_current_video_benchmark_protocol": True,
            "ground_truth_used_during_fit": False,
            "alignment_transform": "Hungarian permutation and signs only",
            "post_hoc_amplitude_fit": False,
            "robust_kappa_policy": "explicit_fixed_in_normalized_units",
            "outlier_layer": "native solver S",
            "outlier_post_threshold": False,
            "frame_error_scope": "all frames",
            "frame_error_definition": (
                "per-frame RMSE against clean target divided by global clean-video RMS"
            ),
        },
        "strict_qualification": {
            "overall": bool(all(item["qualified"] for item in strict.values())),
            "methods": strict,
            "failure_policy": (
                "nonconverged fits remain fixed-budget descriptive output and are "
                "not strict-qualified"
            ),
        },
        "paper_result_effect_gate": _figure5_effect_gate(result),
        "full_frame_error_summary": {
            key: summarize_full_frame_errors(values, time)
            for key, (time, values) in errors.items()
        },
        "fit_metrics": {
            "figure5": {
                method: {
                    "clean_relative_reconstruction_error": (
                        relative_reconstruction_error(
                            result.figure5_data.clean, fit.reconstruction
                        )
                    ),
                    "runtime_seconds": float(fit.runtime_seconds),
                    "loading_sparsity_at_1e-10": float(
                        np.mean(np.abs(fit.loadings) <= 1e-10)
                    ),
                    "mode_and_dynamics_recovery": _figure5_recovery_metrics(
                        result, fit
                    ),
                    "extra": fit.extra,
                }
                for method, fit in result.figure5_fits.items()
            },
            "figure6": {
                method: {
                    "clean_relative_reconstruction_error": (
                        relative_reconstruction_error(
                            result.figure6_data.clean, fit.reconstruction
                        )
                    ),
                    "observed_relative_reconstruction_error": (
                        relative_reconstruction_error(
                            result.figure6_data.observed, fit.reconstruction
                        )
                    ),
                    "runtime_seconds": float(fit.runtime_seconds),
                    "extra": fit.extra,
                }
                for method, fit in result.figure6_fits.items()
            },
            "figure6_robust_corruption_detection_at_true_budget": detection,
        },
        "display": {
            "figure5": figure5_display,
            "figure6": figure6_display,
        },
        "artifacts": {
            "figure5": FIGURE5_NAME,
            "figure6": FIGURE6_NAME,
            "full_frame_errors": FRAME_ERRORS_NAME,
        },
    }
    payload["report_body_sha256"] = canonical_sha256(payload)
    return _json_safe(payload)


def save_paper_video_reproduction(
    result: PaperVideoResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a new exact-closure directory; existing paths are never replaced."""

    destination = Path(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing path: {destination}") from error

    repo_root = Path(__file__).resolve().parents[1]
    paths = {
        "figure5": destination / FIGURE5_NAME,
        "figure6": destination / FIGURE6_NAME,
        "full_frame_errors": destination / FRAME_ERRORS_NAME,
        "report": destination / REPORT_NAME,
        "manifest": destination / MANIFEST_NAME,
    }
    errors = _all_frame_errors(result)
    _write_frame_errors(errors, paths["full_frame_errors"])
    figure5_display = _write_figure5(result, paths["figure5"])
    figure6_display = _write_figure6(result, paths["figure6"])
    report = _report_payload(
        result,
        repo_root=repo_root,
        errors=errors,
        figure5_display=figure5_display,
        figure6_display=figure6_display,
    )
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_body: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "paper_video_exact_output_closure",
        "self_excluding": True,
        "files": {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for key, path in paths.items()
            if key != "manifest"
        },
    }
    manifest_body["manifest_sha256"] = canonical_sha256(manifest_body)
    paths["manifest"].write_text(
        json.dumps(manifest_body, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return paths


def validate_paper_video_output_directory(output_dir: str | Path) -> dict[str, Any]:
    """Validate strict JSON, the self-excluding digest, and exact file closure."""

    destination = Path(output_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("output path must be a real directory, not a symlink")
    entries = list(destination.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"output closure contains a non-regular entry: {entry.name}"
            )
    actual_names = {entry.name for entry in entries}
    if actual_names != _REQUIRED_OUTPUT_NAMES:
        raise ValueError(
            "output directory must contain exactly the four required payloads "
            "and the manifest"
        )
    manifest_path = destination / MANIFEST_NAME
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ))
    body = dict(manifest)
    recorded_digest = body.pop("manifest_sha256")
    if recorded_digest != canonical_sha256(body):
        raise ValueError("output manifest canonical digest mismatch")
    if not isinstance(manifest.get("files"), Mapping):
        raise ValueError("output manifest files must be a mapping")
    if set(manifest["files"]) != _REQUIRED_PAYLOAD_NAMES:
        raise ValueError("output manifest does not declare the fixed payload set")
    for name, record in manifest["files"].items():
        path = destination / name
        if _file_sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch for {name}")
    with (destination / REPORT_NAME).open(encoding="utf-8") as stream:
        report = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-standard JSON constant: {value}")
        ))
    report_body = dict(report)
    report_digest = report_body.pop("report_body_sha256")
    if report_digest != canonical_sha256(report_body):
        raise ValueError("report canonical digest mismatch")
    if set(report.get("artifacts", {}).values()) != {
        FIGURE5_NAME,
        FIGURE6_NAME,
        FRAME_ERRORS_NAME,
    }:
        raise ValueError("report artifact declarations are incomplete")
    return manifest


def smoke_config() -> PaperVideoConfig:
    """Cheap deterministic plumbing configuration, explicitly not evidence."""

    return PaperVideoConfig(
        grid_size=12,
        n_timesteps=48,
        total_time=150.0,
        figure5_alpha=0.2,
        randomized_oversample=2,
        randomized_power_iters=1,
        randomized_full_refine_iters=1,
        max_iter=3,
        tol=1e-4,
        figure6_frame_index=16,
        dpi=72,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run a 12x12x48 plumbing check instead of the paper-scale default",
    )
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--tol", type=float, default=None)
    parser.add_argument("--figure5-alpha", type=float, default=None)
    parser.add_argument("--robust-alpha", type=float, default=None)
    parser.add_argument("--robust-kappa", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    base = smoke_config() if arguments.smoke else PaperVideoConfig()
    values = asdict(base)
    for field_name in (
        "max_iter",
        "tol",
        "figure5_alpha",
        "robust_alpha",
        "robust_kappa",
    ):
        value = getattr(arguments, field_name)
        if value is not None:
            values[field_name] = value
    config = PaperVideoConfig(**values)
    started = perf_counter()
    result = run_paper_video_reproduction(config)
    paths = save_paper_video_reproduction(result, arguments.output_dir)
    validate_paper_video_output_directory(arguments.output_dir)
    elapsed = perf_counter() - started
    print(
        json.dumps(
            {
                "elapsed_seconds": elapsed,
                "output_dir": str(arguments.output_dir),
                "paper_scale": bool(
                    config.grid_size == 200 and config.n_timesteps == 300
                ),
                "files": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

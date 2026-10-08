"""Paired synthetic-video benchmark for PCA and VP-SPCA variants.

The data contract is explicit: rows of ``X`` are time samples (video frames)
and columns are spatial pixel features.  Consequently the VP-SPCA ``B`` matrix
has one row per pixel and its L1 penalty creates *spatial* sparsity.

The default data include Gaussian sensor noise and sparse gross corruptions, so
the rank-three PCA baseline cannot obtain a mathematically automatic zero
error.  Every method is fitted to the same immutable generated array.  Ground
truth is used only after fitting for permutation/sign alignment and recovery
metrics; no component or score is post-hoc amplitude-fitted.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import csv
import hashlib
import io
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.metrics import (
    component_correlation,
    explained_variance_ratio,
    outlier_detection_metrics,
    relative_reconstruction_error,
    sparsity_ratio,
    subspace_error,
    support_precision_recall_f1,
)
from rspca.randomized import fit_randomized_vp_spca
from rspca.robust import fit_robust_vp_spca
from rspca.vp import fit_vp_spca


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

_OUTPUT_MANIFEST_NAME = "video_spca_output_manifest.json"


@dataclass(frozen=True)
class VideoExperimentConfig:
    """Configuration for a deterministic, smoke-sized video experiment."""

    grid_size: int = 20
    n_timesteps: int = 100
    total_time: float = 30.0
    n_components: int = 3
    noise_std: float = 0.03
    corruption_fraction: float = 0.01
    corruption_scale: float = 1.5
    seed: int = 2025
    alpha: float = 0.08
    max_iter: int = 80
    tol: float = 1e-5
    randomized_oversample: int = 6
    randomized_power_iters: int = 2
    randomized_full_refine_iters: int = 10
    support_threshold: float = 1e-10
    outlier_support_threshold: float = 1e-10


@dataclass(frozen=True)
class VideoDataset:
    """Generated video and the latent quantities used only for scoring."""

    observed: FloatArray
    clean: FloatArray
    spatial_components: FloatArray
    temporal_scores: FloatArray
    spatial_support: BoolArray
    gaussian_noise: FloatArray
    injected_corruption: FloatArray
    corruption_support: BoolArray
    time: FloatArray
    feature_mean_removed: FloatArray
    grid_size: int
    digest: str


@dataclass(frozen=True)
class FittedMethod:
    """Native fitted quantities before any ground-truth scoring."""

    name: str
    axes: FloatArray
    loadings: FloatArray
    scores: FloatArray
    reconstruction: FloatArray
    runtime_seconds: float
    converged: bool
    n_iter: int
    final_stationarity: float | None
    status: str
    outliers: FloatArray | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MethodReport:
    """Comparable metrics for one method on the shared video matrix."""

    method: str
    data_sha256: str
    dataset_manifest_sha256: str
    observed_reconstruction_error: float
    clean_reconstruction_error: float
    observed_explained_variance: float
    achieved_sparsity: float
    subspace_error: float
    axis_component_mean_abs_correlation: float
    loading_component_mean_abs_correlation: float
    temporal_score_mean_abs_correlation: float
    support_precision: float
    support_recall: float
    support_f1: float
    runtime_seconds: float
    converged: bool
    n_iter: int
    final_stationarity: float | None
    status: str
    component_permutation: tuple[int, ...]
    component_signs: tuple[int, ...]
    corruption_detection: Mapping[str, float | int] | None
    corruption_detection_at_budget: Mapping[str, float | int] | None
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class VideoExperimentResult:
    """Dataset, native fits, and reports from one paired experiment."""

    config: VideoExperimentConfig
    dataset: VideoDataset
    fits: Mapping[str, FittedMethod]
    reports: Mapping[str, MethodReport]


DEFAULT_METHODS = ("PCA", "VP-SPCA", "RND-VP-SPCA", "Robust-VP-SPCA")


def _validate_config(config: VideoExperimentConfig) -> None:
    integer_fields = {
        "grid_size": config.grid_size,
        "n_timesteps": config.n_timesteps,
        "n_components": config.n_components,
        "seed": config.seed,
        "max_iter": config.max_iter,
        "randomized_oversample": config.randomized_oversample,
        "randomized_power_iters": config.randomized_power_iters,
        "randomized_full_refine_iters": config.randomized_full_refine_iters,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if config.n_timesteps <= config.n_components:
        raise ValueError("n_timesteps must exceed n_components")
    if not 1 <= config.n_components <= 3:
        raise ValueError("the synthetic generator supports one to three components")
    if config.seed < 0:
        raise ValueError("seed must be non-negative")
    if config.max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if (
        min(
            config.randomized_oversample,
            config.randomized_power_iters,
            config.randomized_full_refine_iters,
        )
        < 0
    ):
        raise ValueError(
            "randomized iteration and oversampling values must be non-negative"
        )
    finite_scalars = {
        "total_time": config.total_time,
        "noise_std": config.noise_std,
        "corruption_fraction": config.corruption_fraction,
        "corruption_scale": config.corruption_scale,
        "alpha": config.alpha,
        "tol": config.tol,
        "support_threshold": config.support_threshold,
        "outlier_support_threshold": config.outlier_support_threshold,
    }
    for name, value in finite_scalars.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if config.noise_std < 0.0 or config.corruption_scale < 0.0:
        raise ValueError("noise_std and corruption_scale must be non-negative")
    if not 0.0 <= config.corruption_fraction < 1.0:
        raise ValueError("corruption_fraction must lie in [0, 1)")
    if config.alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive")
    if config.support_threshold < 0.0 or config.outlier_support_threshold < 0.0:
        raise ValueError("support thresholds must be non-negative")


def _smooth_window(
    time: FloatArray, start: float, stop: float, ramp: float
) -> FloatArray:
    left = 1.0 / (1.0 + np.exp(-(time - start) / ramp))
    right = 1.0 / (1.0 + np.exp(-(time - stop) / ramp))
    return left - right


def _compact_blob(
    x_grid: FloatArray,
    y_grid: FloatArray,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    sigma: float,
) -> FloatArray:
    distance_squared = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
    values = np.exp(-distance_squared / (2.0 * sigma**2))
    values[distance_squared > radius**2] = 0.0
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).eps:
        raise RuntimeError("grid is too coarse for the requested compact component")
    return values / norm


def generate_video_dataset(config: VideoExperimentConfig) -> VideoDataset:
    """Generate one centred ``time x pixels`` common dataset for all methods."""

    _validate_config(config)
    coordinate = np.linspace(0.0, 1.0, config.grid_size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="xy")
    specifications = (
        (0.24, 0.30, 0.18, 0.075),
        (0.73, 0.28, 0.14, 0.055),
        (0.61, 0.76, 0.11, 0.043),
    )[: config.n_components]
    component_images = [
        _compact_blob(
            x_grid,
            y_grid,
            center_x=center_x,
            center_y=center_y,
            radius=radius,
            sigma=sigma,
        )
        for center_x, center_y, radius, sigma in specifications
    ]
    spatial_components = np.column_stack(
        [component.reshape(-1) for component in component_images]
    )
    spatial_support = spatial_components != 0.0

    time = np.linspace(0.0, config.total_time, config.n_timesteps)
    scaled_time = time / config.total_time
    candidates = (
        2.8 * np.sin(2.0 * np.pi * (1.5 * scaled_time + 0.15 * scaled_time**2)),
        2.3
        * np.cos(2.0 * np.pi * 4.0 * scaled_time + 0.35)
        * _smooth_window(scaled_time, 0.10, 0.78, 0.035),
        2.0
        * np.sin(2.0 * np.pi * 8.0 * scaled_time + 0.55)
        * _smooth_window(scaled_time, 0.42, 0.96, 0.025),
    )[: config.n_components]
    temporal_scores = np.column_stack(candidates)
    temporal_scores -= temporal_scores.mean(axis=0, keepdims=True)
    # An explicit sum of rank-one signals makes the synthetic construction
    # transparent and avoids delegating tiny inner dimensions to platform BLAS.
    clean = np.zeros((config.n_timesteps, config.grid_size**2), dtype=np.float64)
    for component_index in range(config.n_components):
        clean += (
            temporal_scores[:, component_index, None]
            * spatial_components[None, :, component_index]
        )
    clean -= clean.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(config.seed)
    gaussian_noise = rng.normal(0.0, config.noise_std, size=clean.shape)
    gaussian_feature_mean = gaussian_noise.mean(axis=0)
    gaussian_noise -= gaussian_feature_mean[None, :]
    injected_corruption = np.zeros_like(clean)
    corruption_support = np.zeros_like(clean, dtype=bool)
    # Corruptions are generated in opposite-valued temporal pairs within each
    # pixel.  They are therefore exactly sparse *and* have zero feature mean;
    # centering cannot smear one injected cell across every frame.
    available_time_pairs = config.n_timesteps // 2
    requested_pairs = int(round(config.corruption_fraction * clean.size / 2.0))
    if config.corruption_fraction > 0.0 and config.corruption_scale > 0.0:
        requested_pairs = max(1, requested_pairs)
    else:
        requested_pairs = 0
    maximum_pairs = clean.shape[1] * available_time_pairs
    requested_pairs = min(requested_pairs, maximum_pairs)
    if requested_pairs:
        time_permutation = rng.permutation(config.n_timesteps)
        time_pairs = time_permutation[: 2 * available_time_pairs].reshape(-1, 2)
        pair_slots = rng.choice(maximum_pairs, size=requested_pairs, replace=False)
        feature_indices = pair_slots // available_time_pairs
        temporal_pair_indices = pair_slots % available_time_pairs
        magnitudes = config.corruption_scale * (
            0.5 + np.abs(rng.standard_normal(requested_pairs))
        )
        signs = rng.choice(np.array([-1.0, 1.0]), size=requested_pairs)
        signed_magnitudes = signs * magnitudes
        first_times = time_pairs[temporal_pair_indices, 0]
        second_times = time_pairs[temporal_pair_indices, 1]
        injected_corruption[first_times, feature_indices] = signed_magnitudes
        injected_corruption[second_times, feature_indices] = -signed_magnitudes
        corruption_support[first_times, feature_indices] = True
        corruption_support[second_times, feature_indices] = True

    # Derive the scoring mask from the final increment itself.  This guards the
    # zero-scale edge case and makes the sparse ground truth exactly the term
    # that enters the fitted matrix, rather than a pre-centering proxy mask.
    corruption_support = injected_corruption != 0.0

    observed = clean + gaussian_noise + injected_corruption
    observed = np.ascontiguousarray(observed, dtype=np.float64)
    return VideoDataset(
        observed=observed,
        clean=np.ascontiguousarray(clean, dtype=np.float64),
        spatial_components=np.ascontiguousarray(spatial_components),
        temporal_scores=np.ascontiguousarray(temporal_scores),
        spatial_support=np.ascontiguousarray(spatial_support),
        gaussian_noise=np.ascontiguousarray(gaussian_noise),
        injected_corruption=np.ascontiguousarray(injected_corruption),
        corruption_support=np.ascontiguousarray(corruption_support),
        time=np.asarray(time, dtype=np.float64),
        feature_mean_removed=np.asarray(gaussian_feature_mean, dtype=np.float64),
        grid_size=config.grid_size,
        # This is content-addressed data provenance. Solver-only settings such
        # as max_iter and tol must not change a data digest.
        digest=array_sha256(observed),
    )


def _fit_pca(data: FloatArray, n_components: int) -> FittedMethod:
    start = perf_counter()
    _, _, right_t = np.linalg.svd(data, full_matrices=False)
    axes = right_t[:n_components].T
    scores = data @ axes
    reconstruction = scores @ axes.T
    runtime = perf_counter() - start
    return FittedMethod(
        name="PCA",
        axes=axes,
        loadings=axes.copy(),
        scores=scores,
        reconstruction=reconstruction,
        runtime_seconds=runtime,
        converged=True,
        n_iter=1,
        final_stationarity=None,
        status="closed-form truncated SVD",
    )


def _fit_vp(data: FloatArray, config: VideoExperimentConfig) -> FittedMethod:
    start = perf_counter()
    result = fit_vp_spca(
        data,
        config.n_components,
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        backtracking=True,
    )
    runtime = perf_counter() - start
    scores = data @ result.B
    return FittedMethod(
        name="VP-SPCA",
        axes=result.A,
        loadings=result.B,
        scores=scores,
        reconstruction=scores @ result.A.T,
        runtime_seconds=runtime,
        converged=result.converged,
        n_iter=result.n_iter,
        final_stationarity=(result.stationarity[-1] if result.stationarity else None),
        status=result.status,
        extra={
            "alpha": result.alpha,
            "final_objective": result.objectives[-1] if result.objectives else None,
        },
    )


def _fit_randomized(data: FloatArray, config: VideoExperimentConfig) -> FittedMethod:
    start = perf_counter()
    result = fit_randomized_vp_spca(
        data,
        config.n_components,
        alpha=config.alpha,
        oversample=config.randomized_oversample,
        n_power_iter=config.randomized_power_iters,
        max_iter=config.max_iter,
        tol=config.tol,
        finalize_full_iters=config.randomized_full_refine_iters,
        random_state=config.seed,
        verbose=False,
    )
    runtime = perf_counter() - start
    scores = data @ result.B
    return FittedMethod(
        name="RND-VP-SPCA",
        axes=result.A,
        loadings=result.B,
        scores=scores,
        reconstruction=scores @ result.A.T,
        runtime_seconds=runtime,
        converged=result.converged,
        n_iter=result.n_iter,
        final_stationarity=result.full_stationarity,
        status=result.status,
        extra={
            "alpha": result.alpha,
            "full_objective": result.full_objective,
            "full_diagnostic_step_size": result.full_diagnostic_step_size,
            "full_diagnostic_step_policy": result.full_diagnostic_step_policy,
            "sketch_residual_relative": result.sketch_residual_relative,
            "sketch_iterations": result.sketch_iterations,
            "full_refinement_iterations": result.full_refinement_iterations,
            "timings": dict(result.timings),
        },
    )


def _fit_robust(data: FloatArray, config: VideoExperimentConfig) -> FittedMethod:
    start = perf_counter()
    result = fit_robust_vp_spca(
        data,
        config.n_components,
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - start
    scores = data @ result.B
    return FittedMethod(
        name="Robust-VP-SPCA",
        axes=result.A,
        loadings=result.B,
        scores=scores,
        reconstruction=scores @ result.A.T,
        runtime_seconds=runtime,
        converged=result.converged,
        n_iter=result.n_iter,
        final_stationarity=(result.stationarity[-1] if result.stationarity else None),
        status=result.status,
        outliers=result.S,
        extra={
            "alpha": result.alpha,
            "kappa": result.kappa,
            "residual_scale": result.residual_scale,
            "estimated_outlier_fraction": result.outlier_fraction,
            "final_objective": result.objectives[-1] if result.objectives else None,
        },
    )


def fit_video_methods(
    dataset: VideoDataset,
    config: VideoExperimentConfig,
    *,
    methods: Iterable[str] = DEFAULT_METHODS,
) -> dict[str, FittedMethod]:
    """Fit selected methods to the exact same observed array."""

    dispatch = {
        "PCA": lambda: _fit_pca(dataset.observed, config.n_components),
        "VP-SPCA": lambda: _fit_vp(dataset.observed, config),
        "RND-VP-SPCA": lambda: _fit_randomized(dataset.observed, config),
        "Robust-VP-SPCA": lambda: _fit_robust(dataset.observed, config),
    }
    selected = tuple(methods)
    if not selected:
        raise ValueError("methods must contain at least one method")
    if len(set(selected)) != len(selected):
        raise ValueError("methods contains duplicates")
    unknown = sorted(set(selected) - set(dispatch))
    if unknown:
        raise ValueError(f"unknown video methods: {unknown}")
    return {name: dispatch[name]() for name in selected}


def _aligned_columns_without_rescaling(
    truth: FloatArray,
    estimate: FloatArray,
    matching: Mapping[str, Any],
) -> tuple[FloatArray, tuple[int, ...], tuple[int, ...]]:
    """Apply only the Hungarian permutation and signs selected for scoring."""

    aligned = np.zeros_like(truth)
    permutation = [-1] * truth.shape[1]
    signs = [1] * truth.shape[1]
    for reference_index, estimated_index, sign in zip(
        matching["reference_indices"],
        matching["estimated_indices"],
        matching["signs"],
    ):
        aligned[:, reference_index] = sign * estimate[:, estimated_index]
        permutation[reference_index] = int(estimated_index)
        signs[reference_index] = int(sign)
    return aligned, tuple(permutation), tuple(signs)


def _paired_mean_absolute_cosine(reference: FloatArray, estimate: FloatArray) -> float:
    """Score corresponding columns without selecting a second assignment."""

    if reference.shape != estimate.shape:
        raise ValueError("paired component matrices must have identical shapes")
    numerator = np.sum(reference * estimate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(estimate, axis=0)
    cosine = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > np.finfo(np.float64).eps,
    )
    return float(np.mean(np.abs(np.clip(cosine, -1.0, 1.0))))


def _video_dataset_manifest(dataset: VideoDataset) -> dict[str, Any]:
    """Content-address every array that defines or scores the video dataset."""

    scoring_inputs = {
        "observed": dataset.observed,
        "clean": dataset.clean,
        "spatial_components": dataset.spatial_components,
        "temporal_scores": dataset.temporal_scores,
        "spatial_support": dataset.spatial_support,
        "corruption_support": dataset.corruption_support,
    }
    provenance_arrays = {
        "gaussian_noise": dataset.gaussian_noise,
        "injected_corruption": dataset.injected_corruption,
        "time": dataset.time,
        "feature_mean_removed": dataset.feature_mean_removed,
    }
    body: dict[str, Any] = {
        "manifest_version": 1,
        "orientation": "time_samples_x_pixel_features",
        "grid_size": int(dataset.grid_size),
        "shape": list(dataset.observed.shape),
        "sha256": dataset.digest,
        "scoring_input_sha256": {
            name: array_sha256(values) for name, values in scoring_inputs.items()
        },
        "provenance_array_sha256": {
            name: array_sha256(values) for name, values in provenance_arrays.items()
        },
        "clean_rank": int(np.linalg.matrix_rank(dataset.clean)),
        "observed_rank": int(np.linalg.matrix_rank(dataset.observed)),
        "injected_corruptions": int(np.count_nonzero(dataset.corruption_support)),
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def evaluate_fitted_method(
    dataset: VideoDataset,
    fitted: FittedMethod,
    config: VideoExperimentConfig,
) -> MethodReport:
    """Score a native fit without fitting amplitudes to the ground truth."""

    expected_component_shape = dataset.spatial_components.shape
    if fitted.axes.shape != expected_component_shape:
        raise ValueError("fitted axes have an incompatible shape")
    if fitted.loadings.shape != expected_component_shape:
        raise ValueError("fitted loadings have an incompatible shape")
    if fitted.reconstruction.shape != dataset.observed.shape:
        raise ValueError("fitted reconstruction has an incompatible shape")

    loading_match = component_correlation(dataset.spatial_components, fitted.loadings)
    aligned_loadings, permutation, signs = _aligned_columns_without_rescaling(
        dataset.spatial_components, fitted.loadings, loading_match
    )
    estimated_support = np.abs(aligned_loadings) > config.support_threshold
    support = support_precision_recall_f1(
        dataset.spatial_support,
        estimated_support,
        threshold=0.0,
    )
    axis_match = component_correlation(dataset.spatial_components, fitted.axes)

    aligned_scores = np.zeros_like(dataset.temporal_scores)
    for reference_index, estimated_index, sign in zip(
        range(len(permutation)), permutation, signs
    ):
        if estimated_index >= 0:
            aligned_scores[:, reference_index] = (
                sign * fitted.scores[:, estimated_index]
            )
    # Temporal recovery must retain the permutation/sign chosen by the spatial
    # loading match.  A second Hungarian solve would hide factor-pairing errors.
    temporal_correlation = _paired_mean_absolute_cosine(
        dataset.temporal_scores, aligned_scores
    )

    corruption_detection: Mapping[str, float | int] | None = None
    corruption_detection_at_budget: Mapping[str, float | int] | None = None
    if fitted.outliers is not None:
        outlier_support = np.abs(fitted.outliers) > config.outlier_support_threshold
        corruption_detection = outlier_detection_metrics(
            dataset.corruption_support,
            estimated_support=outlier_support,
            scores=np.abs(fitted.outliers),
            threshold=0.0,
        )
        # This top-k budget is declared by the data-generating configuration,
        # never inferred from the realized ground-truth support.
        budget_top_k = int(round(config.corruption_fraction * dataset.observed.size))
        budget_metrics = outlier_detection_metrics(
            dataset.corruption_support,
            scores=np.abs(fitted.outliers),
            top_k=budget_top_k,
        )
        corruption_detection_at_budget = {
            **budget_metrics,
            "budget_fraction": float(config.corruption_fraction),
            "budget_top_k": budget_top_k,
        }

    dataset_manifest = _video_dataset_manifest(dataset)

    return MethodReport(
        method=fitted.name,
        data_sha256=dataset.digest,
        dataset_manifest_sha256=str(dataset_manifest["manifest_sha256"]),
        observed_reconstruction_error=relative_reconstruction_error(
            dataset.observed, fitted.reconstruction
        ),
        clean_reconstruction_error=relative_reconstruction_error(
            dataset.clean, fitted.reconstruction
        ),
        observed_explained_variance=explained_variance_ratio(
            dataset.observed, fitted.reconstruction, center=False
        ),
        achieved_sparsity=sparsity_ratio(
            fitted.loadings, threshold=config.support_threshold
        ),
        subspace_error=subspace_error(dataset.spatial_components, fitted.axes),
        axis_component_mean_abs_correlation=float(axis_match["mean_abs_correlation"]),
        loading_component_mean_abs_correlation=float(
            loading_match["mean_abs_correlation"]
        ),
        temporal_score_mean_abs_correlation=temporal_correlation,
        support_precision=float(support["precision"]),
        support_recall=float(support["recall"]),
        support_f1=float(support["f1"]),
        runtime_seconds=float(fitted.runtime_seconds),
        converged=bool(fitted.converged),
        n_iter=int(fitted.n_iter),
        final_stationarity=(
            None
            if fitted.final_stationarity is None
            else float(fitted.final_stationarity)
        ),
        status=fitted.status,
        component_permutation=permutation,
        component_signs=signs,
        corruption_detection=corruption_detection,
        corruption_detection_at_budget=corruption_detection_at_budget,
        extra=dict(fitted.extra),
    )


def run_video_experiment(
    config: VideoExperimentConfig | None = None,
    *,
    methods: Iterable[str] = DEFAULT_METHODS,
) -> VideoExperimentResult:
    """Generate once, fit all methods, and return paired reports."""

    resolved_config = VideoExperimentConfig() if config is None else config
    dataset = generate_video_dataset(resolved_config)
    fits = fit_video_methods(dataset, resolved_config, methods=methods)
    reports = {
        name: evaluate_fitted_method(dataset, fitted, resolved_config)
        for name, fitted in fits.items()
    }
    return VideoExperimentResult(
        config=resolved_config,
        dataset=dataset,
        fits=fits,
        reports=reports,
    )


def _jsonable_report(result: VideoExperimentResult) -> dict[str, Any]:
    dataset_manifest = _video_dataset_manifest(result.dataset)
    return {
        "schema_version": 2,
        "config": asdict(result.config),
        "runtime_versions": runtime_versions(),
        "data": dataset_manifest,
        "methods": {name: asdict(report) for name, report in result.reports.items()},
        "evaluation_protocol": {
            "shared_data_and_seed": True,
            "corruption_ground_truth": (
                "exact zero-feature-mean sparse increment used in observed data"
            ),
            "ground_truth_alignment": (
                "one spatial-loading Hungarian permutation/sign reused for "
                "loading, support, and temporal-score recovery"
            ),
            "independent_temporal_matching": False,
            "post_hoc_amplitude_fit": False,
            "support_threshold": result.config.support_threshold,
            "outlier_support_threshold": result.config.outlier_support_threshold,
            "outlier_budget_source": "config.corruption_fraction",
            "outlier_budget_top_k_rule": (
                "round(config.corruption_fraction * observed_cell_count)"
            ),
        },
    }


def _normalize_undefined_zero_corruption_metrics(
    payload: dict[str, Any], *, zero_corruption: bool
) -> None:
    """Normalize only ranking metrics undefined when the truth has no positives.

    The metric implementation deliberately returns NaN for AUROC and average
    precision when no corruption cell exists.  Those two values have an exact
    semantic null representation.  Every other non-finite diagnostic is an
    artifact error and is rejected by :func:`_json_safe`.
    """

    if not zero_corruption:
        return
    methods = payload.get("methods")
    if not isinstance(methods, Mapping):
        return
    for method in methods.values():
        if not isinstance(method, Mapping):
            continue
        for field_name in (
            "corruption_detection",
            "corruption_detection_at_budget",
        ):
            detection = method.get(field_name)
            if not isinstance(detection, dict):
                continue
            for metric_name in ("roc_auc", "average_precision"):
                value = detection.get(metric_name)
                if isinstance(value, (float, np.floating)) and np.isnan(float(value)):
                    detection[metric_name] = None


def _json_safe(value: Any, *, location: str = "root") -> Any:
    """Return a strict-JSON-compatible copy and reject non-finite numbers."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), location=location)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError(f"number must be finite at {location}")
    return value


def _strict_jsonable_report(result: VideoExperimentResult) -> dict[str, Any]:
    """Build the complete single-run payload under its strict null contract."""

    _validate_config(result.config)
    payload = _jsonable_report(result)
    _normalize_undefined_zero_corruption_metrics(
        payload,
        zero_corruption=not bool(np.any(result.dataset.corruption_support)),
    )
    converted = _json_safe(payload)
    if not isinstance(converted, dict):  # Defensive: the payload is always a mapping.
        raise TypeError("video report payload must be a mapping")
    return converted


def validate_video_output_dir(output_dir: str | Path) -> None:
    """Fail fast when a requested output directory already exists in any form."""

    destination = Path(output_dir)
    # Path.exists() follows symlinks and therefore misses dangling symlinks.
    if os.path.lexists(destination):
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {destination}"
        )


def _claim_and_write_output_directory(
    output_dir: str | Path,
    artifacts: Mapping[str, tuple[str, bytes]],
) -> dict[str, Path]:
    """Atomically claim a new directory and exclusive-create every artifact.

    Files are opened relative to a descriptor for the directory inode that this
    call claimed.  If another process renames or replaces the visible path, this
    writer neither follows that replacement nor deletes anything.  On an I/O
    failure the claimed directory is intentionally left in place as an
    incomplete, non-reusable run; recursive cleanup would be unsafe under races.
    """

    destination = Path(output_dir)
    filenames = [filename for filename, _ in artifacts.values()]
    if not artifacts:
        raise ValueError("artifacts must contain at least one output")
    if len(set(filenames)) != len(filenames):
        raise ValueError("artifact filenames must be unique")
    if any(Path(filename).name != filename or filename in {"", ".", ".."} for filename in filenames):
        raise ValueError("artifact filenames must be plain relative filenames")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {destination}"
        ) from error

    claimed_stat = destination.stat(follow_symlinks=False)
    directory_flags = os.O_RDONLY
    directory_flags |= getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(destination, directory_flags)
    try:
        opened_stat = os.fstat(directory_fd)
        if (opened_stat.st_dev, opened_stat.st_ino) != (
            claimed_stat.st_dev,
            claimed_stat.st_ino,
        ):
            raise RuntimeError("claimed output directory was replaced before writing")
        for _, (filename, content) in artifacts.items():
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            file_flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(filename, file_flags, 0o666, dir_fd=directory_fd)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

    return {
        name: destination / filename for name, (filename, _) in artifacts.items()
    }


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _single_run_output_manifest(
    result: VideoExperimentResult,
    artifacts: Mapping[str, tuple[str, bytes]],
) -> dict[str, Any]:
    dataset_manifest = _video_dataset_manifest(result.dataset)
    artifact_records = []
    for name, (filename, content) in artifacts.items():
        artifact_records.append(
            {
                "name": name,
                "path": filename,
                "bytes": len(content),
                "sha256": _sha256_bytes(content),
            }
        )
    body: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "video_spca_single_run_output",
        "self_excluding_manifest": True,
        "completion_contract": "manifest_written_last_after_all_declared_artifacts",
        "config_sha256": canonical_sha256(asdict(result.config)),
        "data_sha256": result.dataset.digest,
        "dataset_manifest_sha256": dataset_manifest["manifest_sha256"],
        "artifacts": artifact_records,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def _strict_json_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-standard JSON constant in {path}: {token}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def validate_video_output_directory(output_dir: str | Path) -> dict[str, Any]:
    """Validate a completed single-run directory and its exact file closure."""

    destination = Path(output_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("video output must be a real directory")
    manifest_path = destination / _OUTPUT_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("video output is incomplete: completion manifest is missing")
    manifest = _strict_json_load(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError("video output manifest must be a mapping")
    expected_manifest_fields = {
        "schema_version",
        "artifact",
        "self_excluding_manifest",
        "completion_contract",
        "config_sha256",
        "data_sha256",
        "dataset_manifest_sha256",
        "artifacts",
        "manifest_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("video output manifest schema mismatch")
    if manifest["schema_version"] != 1 or isinstance(
        manifest["schema_version"], bool
    ):
        raise ValueError("unsupported video output manifest schema")
    if manifest["artifact"] != "video_spca_single_run_output":
        raise ValueError("unexpected video output manifest artifact")
    if manifest["self_excluding_manifest"] is not True:
        raise ValueError("video output manifest must be self-excluding")
    if (
        manifest["completion_contract"]
        != "manifest_written_last_after_all_declared_artifacts"
    ):
        raise ValueError("video output completion contract mismatch")
    manifest_body = dict(manifest)
    claimed_manifest_sha256 = manifest_body.pop("manifest_sha256")
    if claimed_manifest_sha256 != canonical_sha256(manifest_body):
        raise ValueError("video output manifest SHA-256 mismatch")

    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise ValueError("video output manifest must declare artifacts")
    declared_paths: set[str] = set()
    declared_names: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "name",
            "path",
            "bytes",
            "sha256",
        }:
            raise ValueError(f"video output artifact[{index}] schema mismatch")
        name = record["name"]
        relative = record["path"]
        if not isinstance(name, str) or not name or name in declared_names:
            raise ValueError("video output artifact names must be unique strings")
        if (
            not isinstance(relative, str)
            or Path(relative).name != relative
            or relative == _OUTPUT_MANIFEST_NAME
            or relative in declared_paths
        ):
            raise ValueError("video output artifact paths must be unique plain filenames")
        if (
            isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] < 0
        ):
            raise ValueError("video output artifact byte counts must be non-negative")
        digest = record["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("video output artifact SHA-256 is invalid")
        path = destination / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"declared video output artifact is missing: {relative}")
        content = path.read_bytes()
        if len(content) != record["bytes"] or _sha256_bytes(content) != digest:
            raise ValueError(f"video output artifact hash/size mismatch: {relative}")
        declared_names.add(name)
        declared_paths.add(relative)

    actual_paths = {
        path.name
        for path in destination.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != declared_paths | {_OUTPUT_MANIFEST_NAME}:
        raise ValueError("video output directory file closure mismatch")
    if any(path.is_symlink() or not path.is_file() for path in destination.iterdir()):
        raise ValueError("video output directory contains non-regular or nested artifacts")

    metrics_records = [record for record in records if record["name"] == "json"]
    if len(metrics_records) != 1:
        raise ValueError("video output manifest must declare exactly one metrics JSON")
    metrics = _strict_json_load(destination / metrics_records[0]["path"])
    if not isinstance(metrics, Mapping):
        raise TypeError("video metrics JSON must be a mapping")
    if canonical_sha256(metrics.get("config")) != manifest["config_sha256"]:
        raise ValueError("video output config binding mismatch")
    data = metrics.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("video metrics JSON lacks data provenance")
    if data.get("sha256") != manifest["data_sha256"]:
        raise ValueError("video output data SHA-256 binding mismatch")
    if data.get("manifest_sha256") != manifest["dataset_manifest_sha256"]:
        raise ValueError("video output dataset-manifest binding mismatch")
    return manifest


def save_video_experiment(
    result: VideoExperimentResult,
    output_dir: str | Path,
    *,
    make_plot: bool = False,
) -> dict[str, Path]:
    """Strictly serialize a run into one atomically claimed new directory."""

    payload = _strict_jsonable_report(result)
    json_bytes = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flat_fields = (
        "method",
        "data_sha256",
        "dataset_manifest_sha256",
        "observed_reconstruction_error",
        "clean_reconstruction_error",
        "observed_explained_variance",
        "achieved_sparsity",
        "subspace_error",
        "axis_component_mean_abs_correlation",
        "loading_component_mean_abs_correlation",
        "temporal_score_mean_abs_correlation",
        "support_precision",
        "support_recall",
        "support_f1",
        "runtime_seconds",
        "converged",
        "n_iter",
        "final_stationarity",
        "status",
    )
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=flat_fields)
    writer.writeheader()
    payload_methods = payload["methods"]
    for method_name in result.reports:
        serialised = payload_methods[method_name]
        writer.writerow({field_name: serialised[field_name] for field_name in flat_fields})

    artifacts: dict[str, tuple[str, bytes]] = {
        "json": ("video_spca_metrics.json", json_bytes),
        "csv": ("video_spca_metrics.csv", csv_buffer.getvalue().encode("utf-8")),
    }
    if make_plot:
        # Render before claiming the destination so rendering failures cannot
        # leave a directory that looks like a completed experiment.
        artifacts["plot"] = (
            "video_spca_summary.png",
            _summary_plot_png_bytes(result),
        )
    manifest = _single_run_output_manifest(result, artifacts)
    artifacts["manifest"] = (
        _OUTPUT_MANIFEST_NAME,
        (
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    paths = _claim_and_write_output_directory(output_dir, artifacts)
    validate_video_output_directory(output_dir)
    return paths


def _summary_plot_png_bytes(result: VideoExperimentResult) -> bytes:
    import matplotlib.pyplot as plt

    names = list(result.reports)
    metrics = (
        ("clean_reconstruction_error", "Clean reconstruction error"),
        ("achieved_sparsity", "Achieved loading sparsity"),
        ("support_f1", "Spatial support F1"),
        ("runtime_seconds", "Runtime (seconds)"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (field_name, title) in zip(axes.ravel(), metrics):
        values = [float(getattr(result.reports[name], field_name)) for name in names]
        axis.bar(names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Paired noisy/corrupted video benchmark")
    figure.tight_layout()
    buffer = io.BytesIO()
    try:
        figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
    return buffer.getvalue()


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
        normalised = token.strip().lower()
        if normalised not in aliases:
            raise argparse.ArgumentTypeError(f"unknown method alias: {token}")
        selected.append(aliases[normalised])
    return tuple(selected)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--n-timesteps", type=int, default=100)
    parser.add_argument("--total-time", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--noise-std", type=float, default=0.03)
    parser.add_argument("--corruption-fraction", type=float, default=0.01)
    parser.add_argument("--corruption-scale", type=float, default=1.5)
    parser.add_argument("--alpha", type=float, default=0.08)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--randomized-oversample", type=int, default=6)
    parser.add_argument("--randomized-power-iters", type=int, default=2)
    parser.add_argument("--randomized-full-refine-iters", type=int, default=10)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=DEFAULT_METHODS,
        help="Comma-separated: pca,vp,rnd,robust",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/video_spca"))
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> VideoExperimentResult:
    args = parse_args(argv)
    config = VideoExperimentConfig(
        grid_size=args.grid_size,
        n_timesteps=args.n_timesteps,
        total_time=args.total_time,
        noise_std=args.noise_std,
        corruption_fraction=args.corruption_fraction,
        corruption_scale=args.corruption_scale,
        seed=args.seed,
        alpha=args.alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        randomized_oversample=args.randomized_oversample,
        randomized_power_iters=args.randomized_power_iters,
        randomized_full_refine_iters=args.randomized_full_refine_iters,
    )
    _validate_config(config)
    validate_video_output_dir(args.output_dir)
    result = run_video_experiment(config, methods=args.methods)
    paths = save_video_experiment(result, args.output_dir, make_plot=args.plot)
    print(
        json.dumps(
            _json_safe(
                {
                    "data_sha256": result.dataset.digest,
                    "dataset_manifest_sha256": _video_dataset_manifest(result.dataset)[
                        "manifest_sha256"
                    ],
                    "orientation": "time_samples_x_pixel_features",
                    "outputs": {name: str(path) for name, path in paths.items()},
                    "methods": {
                        name: {
                            "clean_reconstruction_error": report.clean_reconstruction_error,
                            "achieved_sparsity": report.achieved_sparsity,
                            "subspace_error": report.subspace_error,
                            "support_f1": report.support_f1,
                            "runtime_seconds": report.runtime_seconds,
                            "converged": report.converged,
                            "n_iter": report.n_iter,
                        }
                        for name, report in result.reports.items()
                    },
                }
            ),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return result


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_METHODS",
    "FittedMethod",
    "MethodReport",
    "VideoDataset",
    "VideoExperimentConfig",
    "VideoExperimentResult",
    "evaluate_fitted_method",
    "fit_video_methods",
    "generate_video_dataset",
    "main",
    "run_video_experiment",
    "save_video_experiment",
    "validate_video_output_directory",
    "validate_video_output_dir",
]

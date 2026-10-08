"""Current VP-SPCA reconstruction of the paper's Figure-5 visual task.

This is a clean, rank-three, 200 x 200 x 300 synthetic-video experiment.  It
exists to answer a narrow visual question: what does the repository's current
VP-SPCA implementation look like on the same *kind* of mode-and-dynamics panel
shown in Figure 5 of Erichson et al.?

The paper fixes the dimensions (200 x 200 pixels, 300 snapshots, dt=.5 s) but
does not publish its generating arrays or complete formulas.  Therefore the
data below are explicitly described as "closest paper-style inferred" data:
they combine the paper dimensions with the classic generator in the historical
GitHub commit.  They are not claimed to be the exact author data.

The fit and display rules are deliberately fixed:

* current :func:`rspca.vp.fit_vp_spca`, alpha=10, 500 iterations, tol=1e-6;
* PCA initialization, spectral step size, and objective backtracking;
* truth may choose a component permutation and sign only;
* each estimated spatial column is divided by its own max absolute value and
  the same factor is transferred to the temporal column, exactly preserving
  the fitted reconstruction;
* no truth-dependent least-squares amplitude fit and no result-driven tuning.

The writer refuses to overwrite a destination and emits an exact-closure
manifest together with raw arrays, metrics, and two presentation figures.
This output must not be pooled with the separate corrupted-video robust study.
"""

from __future__ import annotations

import argparse
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
from rspca.metrics import relative_reconstruction_error
from rspca.vp import VPSPCAResult, fit_vp_spca


FloatArray = NDArray[np.float64]

ARXIV_ID = "1804.00341"
ARXIV_VERSION = 3
ARXIV_PDF_SHA256 = (
    "e35b37476492b589099f4997fdeaa486d19249fb4bc2c85339ab2fe1c02101e0"
)
PAPER_SOURCE_PLOT_SHA256 = {
    "plots/toy_spca.pdf": (
        "239eeab7da69c2dac7a4924fbe4777c70fbd00a49eb16aa47d6296dcd06a0668"
    ),
    "plots/toy_pca.pdf": (
        "19b882561b10459e262553bed87994095f130a9715c7d5e5f489b5fa12b8e944"
    ),
}
LEGACY_SOURCE_COMMIT = "e01c02f7cbe1eaedc8f5262a4ca9ec4f54db84fa"
LEGACY_CLASSIC_SOURCE_SHA256 = (
    "e6222e19b2423a63f7ff4fce8eb370b4037372d40668c09b25c3702ca686bc56"
)

COMBINED_FIGURE_NAME = "figure5_current_vp_spca_vs_pca_paper_style.png"
GITHUB_FIGURE_NAME = "figure5_current_vp_spca_github_style.png"
TRUTH_FIGURE_NAME = "figure5_inferred_truth_reference.png"
RAW_NAME = "figure5_current_fits_and_data.npz"
REPORT_NAME = "figure5_paper_style_report.json"
README_NAME = "README.md"
MANIFEST_NAME = "figure5_paper_style_manifest.json"
_PAYLOAD_NAMES = frozenset(
    {
        COMBINED_FIGURE_NAME,
        GITHUB_FIGURE_NAME,
        TRUTH_FIGURE_NAME,
        RAW_NAME,
        REPORT_NAME,
        README_NAME,
    }
)
_OUTPUT_NAMES = _PAYLOAD_NAMES | {MANIFEST_NAME}

PROTOCOL_IDENTITY = (
    "closest paper-style inferred from the paper dimensions and historical "
    "classic GitHub generator; not exact paper data"
)


@dataclass(frozen=True)
class Figure5PaperStyleConfig:
    """Frozen current-method Figure-5-style protocol."""

    grid_size: int = 200
    n_snapshots: int = 300
    dt: float = 0.5
    rank: int = 3
    alpha: float = 10.0
    max_iter: int = 500
    tol: float = 1e-6
    dpi: int = 200


@dataclass(frozen=True)
class Figure5Dataset:
    """Clean synthetic video in pixels-by-snapshots orientation."""

    data: FloatArray
    uncentered_data: FloatArray
    spatial_modes: FloatArray
    temporal_raw: FloatArray
    temporal_centered: FloatArray
    time: FloatArray


@dataclass(frozen=True)
class FactorFit:
    """One native fitted factorization before truth-based display alignment."""

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
    extra: Mapping[str, Any]


@dataclass(frozen=True)
class DisplayFactor:
    """Sign/permutation aligned, estimate-only gauge for plotting."""

    spatial_modes: FloatArray
    temporal_dynamics: FloatArray
    permutation: tuple[int, ...]
    signs: tuple[int, ...]
    estimate_only_scales: FloatArray


@dataclass(frozen=True)
class Figure5PaperStyleResult:
    config: Figure5PaperStyleConfig
    dataset: Figure5Dataset
    fits: Mapping[str, FactorFit]
    displays: Mapping[str, DisplayFactor]


def _validate_config(config: Figure5PaperStyleConfig) -> None:
    integer_values = {
        "grid_size": config.grid_size,
        "n_snapshots": config.n_snapshots,
        "rank": config.rank,
        "max_iter": config.max_iter,
        "dpi": config.dpi,
    }
    for name, value in integer_values.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 16:
        raise ValueError("grid_size must be at least 16")
    if config.n_snapshots != 300:
        raise ValueError("the frozen Figure-5-style protocol has 300 snapshots")
    if config.rank != 3:
        raise ValueError("the frozen Figure-5-style protocol has rank 3")
    if config.max_iter != 500:
        raise ValueError("the frozen VP-SPCA budget is 500 iterations")
    if config.dpi <= 0:
        raise ValueError("dpi must be positive")
    for name, value in {"dt": config.dt, "alpha": config.alpha, "tol": config.tol}.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if config.dt != 0.5:
        raise ValueError("the frozen paper timestep is exactly 0.5 seconds")
    if config.alpha != 10.0:
        raise ValueError("the historical GitHub comparison fixes alpha=10")
    if config.tol != 1e-6:
        raise ValueError("the frozen VP-SPCA tolerance is 1e-6")


def _classic_spatial_modes(grid_size: int) -> FloatArray:
    coordinate = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate, indexing="xy")

    def gaussian(center_x: float, center_y: float, sigma: float) -> FloatArray:
        values = np.exp(
            -(
                (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            )
            / (2.0 * sigma**2)
        )
        return np.asarray(values / max(float(values.max()), 1e-12), dtype=np.float64)

    images = (
        gaussian(0.55, 0.55, 0.06),
        gaussian(0.30, 0.20, 0.025),
        gaussian(0.78, 0.80, 0.025),
    )
    return np.ascontiguousarray(
        np.column_stack([image.reshape(-1) for image in images]),
        dtype=np.float64,
    )


def generate_figure5_paper_style_dataset(
    config: Figure5PaperStyleConfig | None = None,
) -> Figure5Dataset:
    """Generate the frozen classic clean video with exact half-second times."""

    resolved = Figure5PaperStyleConfig() if config is None else config
    _validate_config(resolved)
    time = np.arange(resolved.n_snapshots, dtype=np.float64) * resolved.dt
    modes = _classic_spatial_modes(resolved.grid_size)
    temporal = np.zeros((resolved.n_snapshots, resolved.rank), dtype=np.float64)
    temporal[:, 0] = 2.0 * np.sin(2.0 * np.pi * 0.20 * time)
    active2 = (time >= 50.0) & (time <= 100.0)
    temporal[active2, 1] = 2.0 * np.sin(2.0 * np.pi * 0.05 * time[active2])
    active3 = (time >= 80.0) & (time <= 120.0)
    temporal[active3, 2] = 2.0 * np.sin(2.0 * np.pi * 0.30 * time[active3])
    uncentered = np.ascontiguousarray(modes @ temporal.T)
    centered_temporal = temporal - temporal.mean(axis=0, keepdims=True)
    data = np.ascontiguousarray(modes @ centered_temporal.T)
    return Figure5Dataset(
        data=data,
        uncentered_data=uncentered,
        spatial_modes=modes,
        temporal_raw=np.ascontiguousarray(temporal),
        temporal_centered=np.ascontiguousarray(centered_temporal),
        time=time,
    )


def _pca_fit(data: FloatArray, rank: int) -> tuple[FactorFit, float]:
    """Fit rank-r PCA using the exact snapshots Gram matrix."""

    start = perf_counter()
    gram = np.asarray(data.T @ data, dtype=np.float64)
    eigenvalues, eigenvectors = linalg.eigh(
        gram,
        subset_by_index=[gram.shape[0] - rank, gram.shape[0] - 1],
        check_finite=False,
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axes = np.ascontiguousarray(eigenvectors[:, order], dtype=np.float64)
    spatial = np.ascontiguousarray(data @ axes)
    reconstruction = np.ascontiguousarray(spatial @ axes.T)
    runtime = perf_counter() - start
    sigma_max = float(np.sqrt(eigenvalues[0]))
    return (
        FactorFit(
            method="PCA",
            axes=axes,
            loadings=axes.copy(),
            spatial_modes=spatial,
            temporal_dynamics=axes.copy(),
            reconstruction=reconstruction,
            runtime_seconds=float(runtime),
            converged=True,
            n_iter=1,
            final_stationarity=None,
            status="closed-form rank-3 PCA from X.T @ X",
            objectives=(),
            extra={
                "largest_singular_value": sigma_max,
                "solver": "scipy.linalg.eigh on exact snapshots Gram matrix",
            },
        ),
        sigma_max,
    )


def _vp_fit(
    data: FloatArray,
    pca: FactorFit,
    sigma_max: float,
    config: Figure5PaperStyleConfig,
) -> FactorFit:
    step_size = 1.0 if sigma_max == 0.0 else 1.0 / sigma_max**2
    start = perf_counter()
    result: VPSPCAResult = fit_vp_spca(
        data,
        config.rank,
        alpha=config.alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        step_size=step_size,
        initial_B=pca.axes,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - start
    spatial = np.ascontiguousarray(data @ result.B)
    reconstruction = np.ascontiguousarray(spatial @ result.A.T)
    return FactorFit(
        method="VP-SPCA",
        axes=np.ascontiguousarray(result.A),
        loadings=np.ascontiguousarray(result.B),
        spatial_modes=spatial,
        temporal_dynamics=np.ascontiguousarray(result.A),
        reconstruction=reconstruction,
        runtime_seconds=float(runtime),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        final_stationarity=(
            float(result.stationarity[-1]) if result.stationarity else None
        ),
        status=result.status,
        objectives=tuple(float(value) for value in result.objectives),
        extra={
            "alpha": float(result.alpha),
            "initialization": "PCA",
            "step_size": float(step_size),
            "step_size_policy": "1 / largest_singular_value**2",
            "backtracking": True,
        },
    )


def align_and_apply_estimate_only_gauge(
    truth_modes: FloatArray,
    estimated_modes: FloatArray,
    estimated_dynamics: FloatArray,
) -> DisplayFactor:
    """Align by truth sign/order, then normalize using estimates only.

    The max-absolute spatial scale is transferred exactly to the matching
    dynamics column, so the fitted reconstruction is invariant.  Truth never
    enters the amplitude calculation.
    """

    truth = np.asarray(truth_modes, dtype=np.float64)
    modes = np.asarray(estimated_modes, dtype=np.float64)
    dynamics = np.asarray(estimated_dynamics, dtype=np.float64)
    if truth.ndim != 2 or modes.shape != truth.shape:
        raise ValueError("truth_modes and estimated_modes must have equal 2-D shape")
    if dynamics.ndim != 2 or dynamics.shape[1] != modes.shape[1]:
        raise ValueError("estimated_dynamics has an incompatible component count")
    truth_norm = np.linalg.norm(truth, axis=0)
    estimate_norm = np.linalg.norm(modes, axis=0)
    denominator = truth_norm[:, None] * estimate_norm[None, :]
    cosine = np.divide(
        truth.T @ modes,
        denominator,
        out=np.zeros((truth.shape[1], modes.shape[1]), dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )
    truth_index, estimate_index = linear_sum_assignment(-np.abs(cosine))
    order = np.argsort(truth_index)
    permutation = tuple(int(value) for value in estimate_index[order])
    signs = tuple(
        1 if cosine[truth, estimate] >= 0.0 else -1
        for truth, estimate in zip(truth_index[order], estimate_index[order])
    )
    aligned_modes = modes[:, permutation] * np.asarray(signs)[None, :]
    aligned_dynamics = dynamics[:, permutation] * np.asarray(signs)[None, :]
    scales = np.max(np.abs(aligned_modes), axis=0)
    if np.any(scales <= np.finfo(np.float64).eps):
        raise RuntimeError("cannot display a zero estimated spatial mode")
    display_modes = aligned_modes / scales[None, :]
    display_dynamics = aligned_dynamics * scales[None, :]
    return DisplayFactor(
        spatial_modes=np.ascontiguousarray(display_modes),
        temporal_dynamics=np.ascontiguousarray(display_dynamics),
        permutation=permutation,
        signs=signs,
        estimate_only_scales=np.ascontiguousarray(scales),
    )


def run_figure5_paper_style(
    config: Figure5PaperStyleConfig | None = None,
) -> Figure5PaperStyleResult:
    resolved = Figure5PaperStyleConfig() if config is None else config
    _validate_config(resolved)
    dataset = generate_figure5_paper_style_dataset(resolved)
    pca, sigma_max = _pca_fit(dataset.data, resolved.rank)
    vp = _vp_fit(dataset.data, pca, sigma_max, resolved)
    fits = {"VP-SPCA": vp, "PCA": pca}
    displays = {
        method: align_and_apply_estimate_only_gauge(
            dataset.spatial_modes, fit.spatial_modes, fit.temporal_dynamics
        )
        for method, fit in fits.items()
    }
    return Figure5PaperStyleResult(
        config=resolved,
        dataset=dataset,
        fits=fits,
        displays=displays,
    )


def _paired_cosines(reference: FloatArray, estimate: FloatArray) -> list[float]:
    numerator = np.sum(reference * estimate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(estimate, axis=0)
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros(reference.shape[1], dtype=np.float64),
        where=denominator > np.finfo(np.float64).eps,
    )
    return [float(value) for value in np.clip(values, -1.0, 1.0)]


def _truth_display(dataset: Figure5Dataset) -> tuple[FloatArray, FloatArray]:
    scales = np.max(np.abs(dataset.spatial_modes), axis=0)
    return (
        dataset.spatial_modes / scales[None, :],
        dataset.temporal_centered * scales[None, :],
    )


def _method_metrics(result: Figure5PaperStyleResult, method: str) -> dict[str, Any]:
    fit = result.fits[method]
    display = result.displays[method]
    spatial = _paired_cosines(result.dataset.spatial_modes, display.spatial_modes)
    temporal = _paired_cosines(
        result.dataset.temporal_centered, display.temporal_dynamics
    )
    reconstructed_from_display = display.spatial_modes @ display.temporal_dynamics.T
    native_aligned = (
        fit.spatial_modes[:, display.permutation]
        * np.asarray(display.signs)[None, :]
    ) @ (
        fit.temporal_dynamics[:, display.permutation]
        * np.asarray(display.signs)[None, :]
    ).T
    gauge_error = relative_reconstruction_error(
        fit.reconstruction, reconstructed_from_display
    )
    aligned_error = relative_reconstruction_error(fit.reconstruction, native_aligned)
    final_stationarity = fit.final_stationarity
    objective_monotone = True
    if len(fit.objectives) >= 2:
        values = np.asarray(fit.objectives)
        tolerance = 1e-10 * np.maximum(1.0, np.abs(values[:-1]))
        objective_monotone = bool(np.all(np.diff(values) <= tolerance))
    return {
        "clean_relative_reconstruction_error": relative_reconstruction_error(
            result.dataset.data, fit.reconstruction
        ),
        "spatial_cosine_per_component": spatial,
        "spatial_cosine_mean": float(np.mean(spatial)),
        "temporal_cosine_per_component": temporal,
        "temporal_cosine_mean": float(np.mean(temporal)),
        "runtime_seconds": float(fit.runtime_seconds),
        "converged": bool(fit.converged),
        "n_iter": int(fit.n_iter),
        "final_stationarity": final_stationarity,
        "status": fit.status,
        "objective_monotone_with_roundoff_tolerance": objective_monotone,
        "loading_zero_fraction_at_1e-10": float(
            np.mean(np.abs(fit.loadings) <= 1e-10)
        ),
        "component_permutation": list(display.permutation),
        "component_signs": list(display.signs),
        "estimate_only_spatial_maxabs_scales": [
            float(value) for value in display.estimate_only_scales
        ],
        "display_gauge_reconstruction_relative_error": float(gauge_error),
        "alignment_reconstruction_relative_error": float(aligned_error),
        "truth_dependent_amplitude_refit": False,
        "extra": dict(fit.extra),
    }


def _configure_image_axis(axis: Any) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#2b2b2b")
        spine.set_linewidth(1.0)


def _configure_dynamics_axis(axis: Any, time: FloatArray, limit: float) -> None:
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_ylim(-1.06 * limit, 1.06 * limit)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_color("#2b2b2b")
        spine.set_linewidth(1.0)


def _write_paper_style_combined(
    result: Figure5PaperStyleResult, path: Path
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("VP-SPCA", "PCA")
    temporal_limit = max(
        float(np.max(np.abs(result.displays[method].temporal_dynamics)))
        for method in methods
    )
    temporal_limit = max(temporal_limit, np.finfo(np.float64).eps)
    figure = plt.figure(figsize=(18, 6.6), facecolor="white")
    grid = figure.add_gridspec(
        2,
        7,
        width_ratios=(1, 1, 1, 0.10, 1, 1, 1),
        left=0.018,
        right=0.992,
        bottom=0.055,
        top=0.91,
        wspace=0.035,
        hspace=0.035,
    )
    columns = (("VP-SPCA", 0), ("PCA", 4))
    for method, offset in columns:
        display = result.displays[method]
        for component in range(3):
            image_axis = figure.add_subplot(grid[0, offset + component])
            image_axis.imshow(
                np.clip(display.spatial_modes[:, component], 0.0, 1.0).reshape(
                    result.config.grid_size, result.config.grid_size
                ),
                cmap="cubehelix_r",
                vmin=0.0,
                vmax=1.0,
                interpolation="bilinear",
            )
            _configure_image_axis(image_axis)
            time_axis = figure.add_subplot(grid[1, offset + component])
            time_axis.plot(
                result.dataset.time,
                display.temporal_dynamics[:, component],
                color="#202020",
                linewidth=1.35,
            )
            _configure_dynamics_axis(time_axis, result.dataset.time, temporal_limit)
    figure.text(0.244, 0.955, "Current VP-SPCA", ha="center", va="center", fontsize=16)
    figure.text(0.759, 0.955, "PCA", ha="center", va="center", fontsize=16)
    figure.text(
        0.5,
        0.012,
        "Closest paper-style inferred clean video; not the unpublished exact paper data",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#565656",
    )
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "two side-by-side 2x3 method blocks; spatial top, dynamics bottom",
        "methods": list(methods),
        "spatial_colormap": "cubehelix_r",
        "spatial_display": "clip estimate-only maxabs-gauged modes to [0, 1]",
        "temporal_shared_abs_limit": float(temporal_limit),
        "truth_overlay": False,
        "titles_legends_colorbars_inside_panels": False,
    }


def _write_github_style_vp(
    result: Figure5PaperStyleResult, path: Path
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    display = result.displays["VP-SPCA"]
    truth_modes, truth_dynamics = _truth_display(result.dataset)
    limit = max(
        float(np.max(np.abs(display.temporal_dynamics))),
        float(np.max(np.abs(truth_dynamics))),
        np.finfo(np.float64).eps,
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 8.5), facecolor="white")
    for component in range(3):
        image_axis = axes[0, component]
        image = image_axis.imshow(
            np.clip(display.spatial_modes[:, component], 0.0, 1.0).reshape(
                result.config.grid_size, result.config.grid_size
            ),
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        image_axis.set_title(f"Current VP-SPCA  $\\Phi_{component + 1}$", fontsize=13)
        image_axis.axis("off")
        figure.colorbar(image, ax=image_axis, fraction=0.046, pad=0.035)
        time_axis = axes[1, component]
        time_axis.plot(
            result.dataset.time,
            display.temporal_dynamics[:, component],
            color="#2563eb",
            linewidth=1.55,
            label="current VP-SPCA",
        )
        time_axis.plot(
            result.dataset.time,
            truth_dynamics[:, component],
            color="#111827",
            linestyle="--",
            linewidth=1.05,
            alpha=0.78,
            label="inferred generator truth",
        )
        time_axis.set_xlim(float(result.dataset.time[0]), float(result.dataset.time[-1]))
        time_axis.set_ylim(-1.06 * limit, 1.06 * limit)
        time_axis.set_xlabel("time (s)")
        if component == 0:
            time_axis.set_ylabel("coefficient in reconstruction-preserving gauge")
            time_axis.legend(fontsize=8, loc="upper right")
        time_axis.grid(alpha=0.22)
    figure.suptitle(
        "Current VP-SPCA on the classic clean multiscale video",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.006,
        "Estimate-only max-|spatial mode| gauge; truth selects sign/order only; no amplitude refit",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.01, 0.025, 0.99, 0.97))
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "GitHub-style 2x3; current VP-SPCA only",
        "spatial_colormap": "magma",
        "spatial_display": "clip estimate-only maxabs-gauged modes to [0, 1]",
        "temporal_shared_abs_limit": float(limit),
        "truth_overlay": True,
        "truth_role": "sign/permutation reference and dashed visual reference only",
        "truth_dependent_amplitude_refit": False,
    }


def _write_truth_reference(result: Figure5PaperStyleResult, path: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    modes, dynamics = _truth_display(result.dataset)
    limit = max(float(np.max(np.abs(dynamics))), np.finfo(np.float64).eps)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor="white")
    for component in range(3):
        axes[0, component].imshow(
            modes[:, component].reshape(result.config.grid_size, result.config.grid_size),
            cmap="cubehelix_r",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
        )
        _configure_image_axis(axes[0, component])
        axes[0, component].set_title(f"inferred truth  $\\Phi_{component + 1}$")
        axes[1, component].plot(
            result.dataset.time,
            dynamics[:, component],
            color="#202020",
            linewidth=1.45,
        )
        _configure_dynamics_axis(axes[1, component], result.dataset.time, limit)
    figure.suptitle(
        "Reference: historical classic generator at the paper dimensions",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.018,
        "Inferred reference only — the paper does not publish the exact generating arrays/formulas",
        ha="center",
        fontsize=9,
        color="#565656",
    )
    figure.tight_layout(rect=(0.01, 0.035, 0.99, 0.95), h_pad=0.25, w_pad=0.25)
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "2x3; inferred generating spatial modes and centered dynamics",
        "spatial_colormap": "cubehelix_r",
        "temporal_shared_abs_limit": float(limit),
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


def _source_provenance(repo_root: Path) -> dict[str, Any]:
    roles = {
        "experiments/video_figure5_paper_style.py": "experiment_runner",
        "src/rspca/vp.py": "current_vp_spca_solver",
        "src/rspca/datasets.py": "array_and_runtime_provenance",
        "src/rspca/metrics.py": "reconstruction_metric",
        "src/rspca/__init__.py": "package_initialization",
    }
    current: dict[str, dict[str, str]] = {}
    for relative, role in roles.items():
        path = repo_root / relative
        if not path.is_file():
            raise RuntimeError(f"required executed source is missing: {relative}")
        current[relative] = {"sha256": _file_sha256(path), "role": role}
    historical = _git_bytes(
        repo_root, ["show", f"{LEGACY_SOURCE_COMMIT}:spca_multiscale_video.py"]
    )
    historical_hash = None if historical is None else sha256(historical).hexdigest()
    status = _git_output(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    return {
        "paper": {
            "title": "Sparse Principal Component Analysis via Variable Projection",
            "arxiv_id": ARXIV_ID,
            "arxiv_version": ARXIV_VERSION,
            "paper_pdf_sha256": ARXIV_PDF_SHA256,
            "section": "6.1 Multiscale Video Example",
            "published_constraints": {
                "spatial_grid": "200 x 200",
                "duration_seconds": 150.0,
                "dt_seconds": 0.5,
                "snapshots": 300,
                "rank": 3,
            },
            "official_source_plot_sha256": dict(PAPER_SOURCE_PLOT_SHA256),
            "paper_does_not_publish_complete_generator_formulas_or_arrays": True,
        },
        "historical_classic_generator": {
            "commit": LEGACY_SOURCE_COMMIT,
            "path": "spca_multiscale_video.py",
            "expected_source_sha256": LEGACY_CLASSIC_SOURCE_SHA256,
            "local_git_object_sha256": historical_hash,
            "verified": historical_hash == LEGACY_CLASSIC_SOURCE_SHA256,
        },
        "current_executed_source": current,
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "runtime_versions": runtime_versions(),
    }


def _array_hashes(result: Figure5PaperStyleResult) -> dict[str, str]:
    values: dict[str, FloatArray] = {
        "data": result.dataset.data,
        "uncentered_data": result.dataset.uncentered_data,
        "truth_spatial_modes": result.dataset.spatial_modes,
        "truth_temporal_raw": result.dataset.temporal_raw,
        "truth_temporal_centered": result.dataset.temporal_centered,
        "time": result.dataset.time,
    }
    for method, prefix in (("VP-SPCA", "vp"), ("PCA", "pca")):
        fit = result.fits[method]
        display = result.displays[method]
        values.update(
            {
                f"{prefix}_axes": fit.axes,
                f"{prefix}_loadings": fit.loadings,
                f"{prefix}_spatial_modes": fit.spatial_modes,
                f"{prefix}_temporal_dynamics": fit.temporal_dynamics,
                f"{prefix}_reconstruction": fit.reconstruction,
                f"{prefix}_display_spatial_modes": display.spatial_modes,
                f"{prefix}_display_temporal_dynamics": display.temporal_dynamics,
                f"{prefix}_display_scales": display.estimate_only_scales,
            }
        )
    return {name: array_sha256(value) for name, value in values.items()}


def _write_raw_npz(result: Figure5PaperStyleResult, path: Path) -> None:
    vp = result.fits["VP-SPCA"]
    pca = result.fits["PCA"]
    vp_display = result.displays["VP-SPCA"]
    pca_display = result.displays["PCA"]
    np.savez_compressed(
        path,
        data=result.dataset.data,
        uncentered_data=result.dataset.uncentered_data,
        truth_spatial_modes=result.dataset.spatial_modes,
        truth_temporal_raw=result.dataset.temporal_raw,
        truth_temporal_centered=result.dataset.temporal_centered,
        time=result.dataset.time,
        vp_axes=vp.axes,
        vp_loadings=vp.loadings,
        vp_spatial_modes=vp.spatial_modes,
        vp_temporal_dynamics=vp.temporal_dynamics,
        vp_reconstruction=vp.reconstruction,
        vp_display_spatial_modes=vp_display.spatial_modes,
        vp_display_temporal_dynamics=vp_display.temporal_dynamics,
        vp_display_scales=vp_display.estimate_only_scales,
        vp_permutation=np.asarray(vp_display.permutation, dtype=np.int64),
        vp_signs=np.asarray(vp_display.signs, dtype=np.int64),
        pca_axes=pca.axes,
        pca_loadings=pca.loadings,
        pca_spatial_modes=pca.spatial_modes,
        pca_temporal_dynamics=pca.temporal_dynamics,
        pca_reconstruction=pca.reconstruction,
        pca_display_spatial_modes=pca_display.spatial_modes,
        pca_display_temporal_dynamics=pca_display.temporal_dynamics,
        pca_display_scales=pca_display.estimate_only_scales,
        pca_permutation=np.asarray(pca_display.permutation, dtype=np.int64),
        pca_signs=np.asarray(pca_display.signs, dtype=np.int64),
    )


def _report(
    result: Figure5PaperStyleResult,
    display: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    config = result.config
    hashes = _array_hashes(result)
    vp_metrics = _method_metrics(result, "VP-SPCA")
    pca_metrics = _method_metrics(result, "PCA")
    normalized_temporal = result.dataset.temporal_centered / np.linalg.norm(
        result.dataset.temporal_centered, axis=0, keepdims=True
    )
    normalized_spatial = result.dataset.spatial_modes / np.linalg.norm(
        result.dataset.spatial_modes, axis=0, keepdims=True
    )
    temporal_gram = normalized_temporal.T @ normalized_temporal
    spatial_gram = normalized_spatial.T @ normalized_spatial
    pca_matches_truth_to_numerical_precision = bool(
        min(pca_metrics["spatial_cosine_per_component"]) >= 1.0 - 1e-10
        and min(pca_metrics["temporal_cosine_per_component"]) >= 1.0 - 1e-10
    )
    return {
        "schema_version": 1,
        "experiment": "current VP-SPCA Figure-5 paper-style reconstruction",
        "protocol_identity": PROTOCOL_IDENTITY,
        "paper_scale": bool(config.grid_size == 200),
        "config": asdict(config),
        "protocol": {
            "orientation": "spatial_pixels_x_time_snapshots",
            "clean_only": True,
            "rank": 3,
            "time_formula": "t = np.arange(300) * 0.5",
            "time_first": float(result.dataset.time[0]),
            "time_last": float(result.dataset.time[-1]),
            "time_step": float(result.dataset.time[1] - result.dataset.time[0]),
            "spatial_gaussians": [
                {"center": [0.55, 0.55], "sigma": 0.06},
                {"center": [0.30, 0.20], "sigma": 0.025},
                {"center": [0.78, 0.80], "sigma": 0.025},
            ],
            "temporal_formulas": [
                "2 sin(2 pi 0.20 t), all snapshots",
                "2 sin(2 pi 0.05 t), active on inclusive [50, 100] seconds",
                "2 sin(2 pi 0.30 t), active on inclusive [80, 120] seconds",
            ],
            "centering": "subtract each pixel's temporal mean before fitting",
            "methods": ["VP-SPCA", "PCA"],
            "vp_solver": "current rspca.vp.fit_vp_spca",
            "vp_parameter_source": "historical GitHub comparison, fixed before this run",
            "vp_initialization": "PCA",
            "vp_backtracking": True,
            "result_driven_tuning": False,
            "alignment": "truth-based Hungarian permutation and sign only",
            "display_gauge": (
                "divide each estimated spatial column by its own max absolute "
                "value and multiply the matching temporal column by that value"
            ),
            "display_gauge_uses_truth_amplitude": False,
            "truth_dependent_amplitude_refit": False,
            "separate_from_corrupted_robust_study": True,
        },
        "data": {
            "shape": list(result.dataset.data.shape),
            "array_sha256": hashes,
            "normalized_truth_temporal_gram": temporal_gram.tolist(),
            "normalized_truth_spatial_gram": spatial_gram.tolist(),
            "maximum_absolute_off_diagonal_temporal_gram": float(
                np.max(np.abs(temporal_gram - np.eye(config.rank)))
            ),
            "maximum_absolute_off_diagonal_spatial_gram": float(
                np.max(np.abs(spatial_gram - np.eye(config.rank)))
            ),
        },
        "fit_metrics": {
            "VP-SPCA": vp_metrics,
            "PCA": pca_metrics,
        },
        "paper_qualitative_contrast_audit": {
            "paper_claim": "SPCA disambiguates modes 2 and 3 while PCA mixes them",
            "pca_matches_aligned_truth_to_numerical_precision": (
                pca_matches_truth_to_numerical_precision
            ),
            "paper_pca_mixing_reproduced_by_this_inferred_protocol": bool(
                not pca_matches_truth_to_numerical_precision
            ),
            "interpretation": (
                "The frozen classic formulas with exact dt=.5 make the truth "
                "spatial and temporal components essentially orthogonal. If PCA "
                "matches truth, the unpublished paper generator must contain "
                "additional differences needed for its displayed PCA mixing."
            ),
        },
        "display": dict(display),
        "artifacts": {
            "paper_style_combined": COMBINED_FIGURE_NAME,
            "github_style_current_vp": GITHUB_FIGURE_NAME,
            "inferred_truth_reference": TRUTH_FIGURE_NAME,
            "raw_arrays": RAW_NAME,
        },
        "provenance": _source_provenance(repo_root),
    }


def _readme_text(report: Mapping[str, Any]) -> str:
    vp = report["fit_metrics"]["VP-SPCA"]
    pca = report["fit_metrics"]["PCA"]
    return f"""# Current VP-SPCA Figure-5 paper-style reconstruction

This package answers the visual question posed by the original paper/GitHub
mode-and-dynamics figure. It is a **clean rank-3 video** with 200 x 200 pixels,
300 snapshots, and exact half-second times from 0.0 through 149.5 seconds.
It is not a corrupted-frame or robust-decomposition experiment.

## What is shown

- `{COMBINED_FIGURE_NAME}`: the paper's 2 x 3 visual grammar, with current
  VP-SPCA on the left and PCA on the right. The top row is spatial structure;
  the bottom row is the associated temporal dynamics.
- `{GITHUB_FIGURE_NAME}`: current VP-SPCA alone in the historical GitHub-style
  2 x 3 layout, with the inferred generator truth overlaid as a dashed line.
- `{TRUTH_FIGURE_NAME}`: the inferred classic-generator reference.
- `{RAW_NAME}`: data, truth, native fits, reconstructions, aligned display
  factors, time, permutations, signs, and estimate-only scales.

## Result from this one frozen run

| Method | clean relative reconstruction error | mean spatial cosine | mean temporal cosine | converged | iterations |
|---|---:|---:|---:|---:|---:|
| Current VP-SPCA | {vp['clean_relative_reconstruction_error']:.9g} | {vp['spatial_cosine_mean']:.9f} | {vp['temporal_cosine_mean']:.9f} | {vp['converged']} | {vp['n_iter']} |
| PCA | {pca['clean_relative_reconstruction_error']:.9g} | {pca['spatial_cosine_mean']:.9f} | {pca['temporal_cosine_mean']:.9f} | {pca['converged']} | {pca['n_iter']} |

PCA reconstructs this exactly rank-3 clean matrix almost perfectly, but its
orthogonal basis is not generally identifiable when component energies or
correlations create rotations. In this particular frozen inferred dataset,
however, PCA's aligned spatial and temporal cosines are both effectively 1.0.
It therefore **does not reproduce the paper's displayed PCA mixing of modes 2
and 3**. The exact half-second classic temporal columns are mutually orthogonal
to numerical precision, and the separated Gaussian modes are also essentially
orthogonal. This is evidence that the unpublished paper generator contains
additional differences; it is not a solver or plotting failure and must not be
hidden.

## Critical provenance boundary

The paper specifies a 200 x 200 grid, 150 seconds, dt=.5 s, 300 snapshots, and
three degrees of freedom. It does **not** publish a byte-addressable dataset or
the complete generating formulas. The arrays here are therefore the **closest
paper-style data inferred from those paper constraints plus the historical
classic GitHub generator**. They must not be described as the exact paper data.

The VP settings were fixed from the historical GitHub comparison before this
run: alpha=10, max_iter=500, tol=1e-6, PCA initialization, spectral step size,
and backtracking. There was no result-driven parameter adjustment.

For display, truth chooses component order and sign only. Spatial columns are
normalized using their own fitted max-absolute value, and that scale is moved
to the matching time column, preserving the reconstruction. There is no
truth-dependent amplitude fit. The report records the reconstruction-invariance
error and all array/source hashes.

This package is independent of the corrupted-video robust confirmation results;
their reconstruction errors and claims must not be pooled.
"""


def save_figure5_paper_style(
    result: Figure5PaperStyleResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a new exact-closure output directory; never overwrite."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    combined = destination / COMBINED_FIGURE_NAME
    github = destination / GITHUB_FIGURE_NAME
    truth = destination / TRUTH_FIGURE_NAME
    raw = destination / RAW_NAME
    report_path = destination / REPORT_NAME
    readme = destination / README_NAME
    manifest_path = destination / MANIFEST_NAME
    combined_display = _write_paper_style_combined(result, combined)
    github_display = _write_github_style_vp(result, github)
    truth_display = _write_truth_reference(result, truth)
    _write_raw_npz(result, raw)
    repo_root = Path(__file__).resolve().parents[1]
    report = _report(
        result,
        {
            "paper_style_combined": combined_display,
            "github_style_current_vp": github_display,
            "inferred_truth_reference": truth_display,
        },
        repo_root,
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme.write_text(_readme_text(report), encoding="utf-8")
    files = {
        path.name: {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}
        for path in (combined, github, truth, raw, report_path, readme)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "figure5_paper_style_exact_output_closure",
        "self_excluding": True,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_figure5_paper_style_output(destination)
    return {
        "combined": combined,
        "github": github,
        "truth": truth,
        "raw": raw,
        "report": report_path,
        "readme": readme,
        "manifest": manifest_path,
    }


def validate_figure5_paper_style_output(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("output must be a real directory")
    entries = list(directory.iterdir())
    nonregular = [
        path.name
        for path in entries
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode)
    ]
    if nonregular:
        raise ValueError(f"output contains non-regular entries: {sorted(nonregular)}")
    names = {path.name for path in entries}
    if names != _OUTPUT_NAMES:
        raise ValueError(
            f"output must contain exactly the seven required files: {sorted(_OUTPUT_NAMES)}"
        )
    manifest_path = directory / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("files", {})) != _PAYLOAD_NAMES:
        raise ValueError("manifest must name exactly the six required payloads")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("manifest self-hash mismatch")
    for name, record in manifest["files"].items():
        path = directory / name
        if record.get("sha256") != _file_sha256(path):
            raise ValueError(f"payload hash mismatch: {name}")
        if record.get("size_bytes") != path.stat().st_size:
            raise ValueError(f"payload size mismatch: {name}")
    report = json.loads((directory / REPORT_NAME).read_text(encoding="utf-8"))
    if report.get("protocol_identity") != PROTOCOL_IDENTITY:
        raise ValueError("report protocol identity mismatch")
    config = report.get("config", {})
    if config.get("alpha") != 10.0 or config.get("max_iter") != 500:
        raise ValueError("report VP-SPCA protocol mismatch")
    if config.get("dt") != 0.5 or config.get("n_snapshots") != 300:
        raise ValueError("report paper-time protocol mismatch")
    if report.get("protocol", {}).get("truth_dependent_amplitude_refit") is not False:
        raise ValueError("report must prohibit truth-dependent amplitude refit")
    with np.load(directory / RAW_NAME, allow_pickle=False) as arrays:
        expected_hashes = report["data"]["array_sha256"]
        for name, expected in expected_hashes.items():
            if name not in arrays.files:
                raise ValueError(f"raw NPZ missing array: {name}")
            if array_sha256(arrays[name]) != expected:
                raise ValueError(f"raw NPZ array hash mismatch: {name}")
        for prefix in ("vp", "pca"):
            display_reconstruction = (
                arrays[f"{prefix}_display_spatial_modes"]
                @ arrays[f"{prefix}_display_temporal_dynamics"].T
            )
            if not np.allclose(
                display_reconstruction,
                arrays[f"{prefix}_reconstruction"],
                rtol=2e-12,
                atol=2e-12,
            ):
                raise ValueError(f"{prefix} display gauge changed reconstruction")
    return manifest


def smoke_config() -> Figure5PaperStyleConfig:
    return Figure5PaperStyleConfig(grid_size=24, dpi=80)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a 24x24 grid but retain the exact time/solver protocol.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    config = smoke_config() if args.smoke else Figure5PaperStyleConfig()
    result = run_figure5_paper_style(config)
    paths = save_figure5_paper_style(result, args.output_dir)
    vp = _method_metrics(result, "VP-SPCA")
    pca = _method_metrics(result, "PCA")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "paper_scale": config.grid_size == 200,
                "vp_spca": vp,
                "pca": pca,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()

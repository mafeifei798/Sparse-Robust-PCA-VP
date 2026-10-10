"""Evaluate VP-SPCA mode recovery on the source-locked smooth-video dataset.

The generator is a source-faithful port of ``generate_multiscale_video`` at
commit ``e01c02f``: 200 x 200 pixels, 300 endpoint-inclusive snapshots over
150 seconds, the three smooth temporal programs, frame-by-frame assembly, and
per-pixel temporal centering.  The default X/Phi/a/t hashes are locked below.

Only the current repository VP-SPCA and a rank-three PCA reference are fit.
VP-SPCA uses the historical comparison's fixed alpha=10, max_iter=500, and
tol=1e-6, with the current solver's PCA initialization and backtracking.  The
native fit and integrity view use truth only for permutation/sign; their
estimate-only gauge preserves reconstruction and performs no amplitude refit.
The two primary GitHub-faithful plots separately reproduce the old
truth-dependent least-squares amplitude fit on a display-only temporal copy.
That display transform never enters the native reconstruction or metrics.

This is a GitHub-derived clean Figure-5-style task, not the unpublished exact
paper data and not the separate corrupted-video robust study.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions

from experiments.video_mode_recovery_reference import (
    DisplayFactor,
    FactorFit,
    Figure5Dataset,
    Figure5PaperStyleConfig,
    _method_metrics,
    _pca_fit,
    _vp_fit,
    align_and_apply_estimate_only_gauge,
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
EXPECTED_DEFAULT_ARRAY_SHA256 = {
    "X": "8ed4ba2a3cd690ba8ce93c6955678e2025ad220fe8e10f413f2bacc1263a76b7",
    "phi_true": "f5637a993df66cc98aac574870ecd3cb8675594dc13628e920ce21eb1e54be8c",
    "a_true": "b800afa694202cf6b5a33afd309e84d6ffa4ebe193268a85ededf3d920d4c6f9",
    "time": "476110f34b2e2250a681517b7696e2737181873ef6b10ea1c85d81ffb0faf01d",
}

VP_FIGURE_NAME = "github_style_current_vp_spca.png"
PCA_FIGURE_NAME = "github_style_current_pca.png"
INTEGRITY_FIGURE_NAME = "integrity_current_vp_spca_no_refit.png"
COMBINED_FIGURE_NAME = "paper_style_current_vp_spca_vs_pca.png"
RAW_NAME = "historical_github_current_fits_and_data.npz"
REPORT_NAME = "historical_github_current_report.json"
README_NAME = "README.md"
MANIFEST_NAME = "historical_github_current_manifest.json"
_PAYLOAD_NAMES = frozenset(
    {
        VP_FIGURE_NAME,
        PCA_FIGURE_NAME,
        INTEGRITY_FIGURE_NAME,
        COMBINED_FIGURE_NAME,
        RAW_NAME,
        REPORT_NAME,
        README_NAME,
    }
)
_OUTPUT_NAMES = _PAYLOAD_NAMES | {MANIFEST_NAME}

PROTOCOL_IDENTITY = (
    "exact commit e01c02f default smooth GitHub generator with current "
    "VP-SPCA and PCA; clean Figure-5-style task; not exact paper data"
)


@dataclass(frozen=True)
class HistoricalGitHubFigure5Config:
    grid_size: int = 200
    n_snapshots: int = 300
    total_time: float = 150.0
    rank: int = 3
    alpha: float = 10.0
    max_iter: int = 500
    tol: float = 1e-6
    dpi: int = 200


@dataclass(frozen=True)
class HistoricalGitHubFigure5Result:
    config: HistoricalGitHubFigure5Config
    dataset: Figure5Dataset
    phi_true_images: np.ndarray
    a_true_raw: np.ndarray
    fits: Mapping[str, FactorFit]
    displays: Mapping[str, DisplayFactor]


@dataclass(frozen=True)
class LegacyGitHubDisplayFactor:
    """Historical display-only gauge and truth-amplitude plotting copy."""

    spatial_modes: np.ndarray
    temporal_gauge: np.ndarray
    temporal_plot: np.ndarray
    permutation: tuple[int, ...]
    signs: tuple[int, ...]
    spatial_l2_norms: np.ndarray
    truth_ls_scales: np.ndarray


def _validate_config(config: HistoricalGitHubFigure5Config) -> None:
    integers = {
        "grid_size": config.grid_size,
        "n_snapshots": config.n_snapshots,
        "rank": config.rank,
        "max_iter": config.max_iter,
        "dpi": config.dpi,
    }
    for name, value in integers.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 16:
        raise ValueError("grid_size must be at least 16")
    if config.n_snapshots != 300 or config.total_time != 150.0:
        raise ValueError("the historical default uses 300 snapshots over 150 seconds")
    if config.rank != 3:
        raise ValueError("the historical video has rank 3")
    if config.alpha != 10.0 or config.max_iter != 500 or config.tol != 1e-6:
        raise ValueError("the frozen historical comparison uses alpha=10, 500, 1e-6")
    if config.dpi <= 0:
        raise ValueError("dpi must be positive")


def _smooth_window(
    time: np.ndarray, start: float, stop: float, *, ramp: float
) -> np.ndarray:
    left = 1.0 / (1.0 + np.exp(-(time - start) / ramp))
    right = 1.0 / (1.0 + np.exp(-(time - stop) / ramp))
    return np.clip(left - right, 0.0, 1.0)


def generate_exact_historical_github_dataset(
    config: HistoricalGitHubFigure5Config | None = None,
) -> tuple[Figure5Dataset, np.ndarray, np.ndarray]:
    """Port the commit-e01c02f default generator operation-for-operation."""

    resolved = HistoricalGitHubFigure5Config() if config is None else config
    _validate_config(resolved)
    coordinate = np.linspace(0.0, 1.0, resolved.grid_size)
    x_grid, y_grid = np.meshgrid(coordinate, coordinate)

    def gaussian_blob(center_x: float, center_y: float, sigma: float) -> np.ndarray:
        blob = np.exp(
            -(
                (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
            )
            / (2.0 * sigma**2)
        )
        return blob / np.maximum(blob.max(), 1e-12)

    phi_true = np.zeros((3, resolved.grid_size, resolved.grid_size))
    phi_true[0] = gaussian_blob(0.55, 0.55, 0.06)
    phi_true[1] = gaussian_blob(0.30, 0.20, 0.025)
    phi_true[2] = gaussian_blob(0.78, 0.80, 0.025)

    time = np.linspace(0.0, resolved.total_time, resolved.n_snapshots)
    a_true = np.zeros((3, resolved.n_snapshots))
    win1 = _smooth_window(time, 0.0, resolved.total_time, ramp=6.0)
    a_true[0] = 2.0 * np.sin(2.0 * np.pi * 0.08 * time) * win1
    plateau = _smooth_window(time, 45.0, 65.0, ramp=2.0)
    burst2 = _smooth_window(time, 65.0, 105.0, ramp=2.0)
    a_true[1] = 0.8 * plateau + 1.8 * np.sin(
        2.0 * np.pi * 0.055 * time + 0.35 * np.pi
    ) * burst2
    burst3 = _smooth_window(time, 100.0, resolved.total_time, ramp=2.0)
    a_true[2] = 2.0 * np.sin(
        2.0 * np.pi * 0.12 * time + 0.1 * np.pi
    ) * burst3

    # Preserve the historical frame loop and reshape order.  Vectorizing this
    # multiplication changes a few final bits and therefore changes X's hash.
    video = np.zeros(
        (resolved.n_snapshots, resolved.grid_size, resolved.grid_size)
    )
    for index in range(resolved.n_snapshots):
        frame = (
            a_true[0, index] * phi_true[0]
            + a_true[1, index] * phi_true[1]
            + a_true[2, index] * phi_true[2]
        )
        video[index] = frame
    data = video.reshape(resolved.n_snapshots, -1).T
    data = data - data.mean(axis=1, keepdims=True)

    flat_modes = phi_true.reshape(3, -1).T
    centered_temporal = a_true.T - a_true.T.mean(axis=0, keepdims=True)
    dataset = Figure5Dataset(
        data=data,
        uncentered_data=video.reshape(resolved.n_snapshots, -1).T,
        spatial_modes=flat_modes,
        temporal_raw=a_true.T,
        temporal_centered=centered_temporal,
        time=time,
    )
    return dataset, phi_true, a_true


def run_historical_github_current(
    config: HistoricalGitHubFigure5Config | None = None,
) -> HistoricalGitHubFigure5Result:
    resolved = HistoricalGitHubFigure5Config() if config is None else config
    _validate_config(resolved)
    dataset, phi_true, a_true = generate_exact_historical_github_dataset(resolved)
    pca, sigma_max = _pca_fit(dataset.data, resolved.rank)
    vp_config = Figure5PaperStyleConfig(
        grid_size=resolved.grid_size,
        n_snapshots=resolved.n_snapshots,
        dt=0.5,
        rank=resolved.rank,
        alpha=resolved.alpha,
        max_iter=resolved.max_iter,
        tol=resolved.tol,
        dpi=resolved.dpi,
    )
    vp = _vp_fit(dataset.data, pca, sigma_max, vp_config)
    fits = {"VP-SPCA": vp, "PCA": pca}
    displays = {
        method: align_and_apply_estimate_only_gauge(
            dataset.spatial_modes, fit.spatial_modes, fit.temporal_dynamics
        )
        for method, fit in fits.items()
    }
    return HistoricalGitHubFigure5Result(
        config=resolved,
        dataset=dataset,
        phi_true_images=phi_true,
        a_true_raw=a_true,
        fits=fits,
        displays=displays,
    )


def _legacy_github_display(
    result: HistoricalGitHubFigure5Result,
    method: str,
) -> LegacyGitHubDisplayFactor:
    """Reproduce the old GitHub display transformation, explicitly display-only."""

    fit = result.fits[method]
    alignment = result.displays[method]
    norms = np.maximum(
        np.linalg.norm(fit.spatial_modes, axis=0), np.finfo(np.float64).eps
    )
    modes = fit.spatial_modes / norms[None, :]
    dynamics = fit.temporal_dynamics * norms[None, :]
    signs = np.asarray(alignment.signs, dtype=np.float64)
    aligned_modes = modes[:, alignment.permutation] * signs[None, :]
    aligned_dynamics = dynamics[:, alignment.permutation] * signs[None, :]
    plot_dynamics = aligned_dynamics.copy()
    ls_scales = np.ones(result.config.rank, dtype=np.float64)
    for component in range(result.config.rank):
        denominator = float(plot_dynamics[:, component] @ plot_dynamics[:, component])
        denominator += 1e-12
        numerator = float(
            plot_dynamics[:, component] @ result.a_true_raw[component]
        )
        ls_scales[component] = numerator / denominator
        plot_dynamics[:, component] *= ls_scales[component]
    return LegacyGitHubDisplayFactor(
        spatial_modes=np.ascontiguousarray(aligned_modes),
        temporal_gauge=np.ascontiguousarray(aligned_dynamics),
        temporal_plot=np.ascontiguousarray(plot_dynamics),
        permutation=alignment.permutation,
        signs=alignment.signs,
        spatial_l2_norms=np.ascontiguousarray(norms),
        truth_ls_scales=np.ascontiguousarray(ls_scales),
    )


def _plot_github_method(
    result: HistoricalGitHubFigure5Result,
    method: str,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    display = _legacy_github_display(result, method)
    truth = result.a_true_raw.T
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    method_label = "Current VP-SPCA" if method == "VP-SPCA" else "PCA"
    line_color = "#2563eb" if method == "VP-SPCA" else "#4b5563"
    for component in range(3):
        image_axis = axes[0, component]
        image = image_axis.imshow(
            display.spatial_modes[:, component].reshape(
                result.config.grid_size, result.config.grid_size
            ),
            cmap="magma",
            interpolation="bilinear",
        )
        image_axis.set_title(f"{method_label}  $\\Phi_{component + 1}$", fontsize=13)
        image_axis.axis("off")
        figure.colorbar(image, ax=image_axis, fraction=0.046, pad=0.035)

        time_axis = axes[1, component]
        time_axis.plot(
            result.dataset.time,
            display.temporal_plot[:, component],
            color=line_color,
            linewidth=1.55,
            label=method_label,
        )
        time_axis.plot(
            result.dataset.time,
            truth[:, component],
            color="#f97316",
            linestyle="--",
            linewidth=1.15,
            alpha=0.92,
            label="raw generator truth",
        )
        time_axis.set_xlim(float(result.dataset.time[0]), float(result.dataset.time[-1]))
        time_axis.set_xlabel("time (s)")
        if component == 0:
            time_axis.set_ylabel("Activation")
            time_axis.legend(fontsize=8, loc="best")
        time_axis.grid(alpha=0.22)
    figure.suptitle(
        f"{method_label} — exact historical GitHub smooth-video DGP",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.007,
        "Historical display protocol: L2 spatial gauge + display-only truth LS amplitude fit",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.01, 0.028, 0.99, 0.97))
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "GitHub-style 2x3; spatial modes top, dynamics bottom",
        "method": method,
        "spatial_colormap": "magma",
        "spatial_per_component_autoscale": True,
        "truth_overlay": "raw generator truth, dashed",
        "spatial_gauge": "L2 norm transferred to temporal factor; reconstruction preserving",
        "truth_ls_amplitude_scales": [float(value) for value in display.truth_ls_scales],
        "truth_dependent_amplitude_refit": True,
        "amplitude_refit_scope": "display-only temporal plotting copy",
        "native_fit_or_metrics_changed": False,
    }


def _plot_integrity_vp(
    result: HistoricalGitHubFigure5Result,
    path: Path,
) -> dict[str, Any]:
    """Plot the no-refit, reconstruction-preserving current VP factors."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    display = result.displays["VP-SPCA"]
    truth = result.dataset.temporal_centered
    temporal_limit = max(
        float(np.max(np.abs(display.temporal_dynamics))),
        float(np.max(np.abs(truth))),
        np.finfo(np.float64).eps,
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    for component in range(3):
        image = axes[0, component].imshow(
            display.spatial_modes[:, component].reshape(
                result.config.grid_size, result.config.grid_size
            ),
            cmap="magma",
            interpolation="bilinear",
        )
        axes[0, component].set_title(f"Current VP-SPCA  $\\Phi_{component + 1}$")
        axes[0, component].axis("off")
        figure.colorbar(image, ax=axes[0, component], fraction=0.046, pad=0.035)
        axes[1, component].plot(
            result.dataset.time,
            display.temporal_dynamics[:, component],
            color="#2563eb",
            linewidth=1.55,
            label="current VP-SPCA",
        )
        axes[1, component].plot(
            result.dataset.time,
            truth[:, component],
            color="#111827",
            linestyle="--",
            linewidth=1.05,
            label="centered generator truth",
        )
        axes[1, component].set_xlim(
            float(result.dataset.time[0]), float(result.dataset.time[-1])
        )
        axes[1, component].set_ylim(-1.06 * temporal_limit, 1.06 * temporal_limit)
        axes[1, component].set_xlabel("time (s)")
        axes[1, component].grid(alpha=0.22)
        if component == 0:
            axes[1, component].set_ylabel("reconstruction-preserving coefficient")
            axes[1, component].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Integrity view — current VP-SPCA without truth-amplitude refit",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.007,
        "Estimate-only max-|spatial mode| gauge; truth selects sign/order only",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.tight_layout(rect=(0.01, 0.028, 0.99, 0.97))
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "2x3 integrity view",
        "spatial_colormap": "magma",
        "spatial_per_component_autoscale": True,
        "truth_overlay": "centered generator truth, dashed",
        "temporal_shared_abs_limit": float(temporal_limit),
        "truth_dependent_amplitude_refit": False,
        "display_gauge_preserves_reconstruction": True,
    }


def _paper_colormap() -> Any:
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "paper_purple",
        ["#ffffff", "#ead7d9", "#d49aa5", "#8a4d75", "#28152f"],
    )


def _plot_combined(
    result: HistoricalGitHubFigure5Result, path: Path
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("VP-SPCA", "PCA")
    temporal_limit = max(
        float(np.max(np.abs(result.displays[method].temporal_dynamics)))
        for method in methods
    )
    cmap = _paper_colormap()
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
    for method, offset in (("VP-SPCA", 0), ("PCA", 4)):
        display = result.displays[method]
        for component in range(3):
            image_axis = figure.add_subplot(grid[0, offset + component])
            image_axis.imshow(
                display.spatial_modes[:, component].reshape(
                    result.config.grid_size, result.config.grid_size
                ),
                cmap=cmap,
                interpolation="bilinear",
            )
            image_axis.set_xticks([])
            image_axis.set_yticks([])
            dynamics_axis = figure.add_subplot(grid[1, offset + component])
            dynamics_axis.plot(
                result.dataset.time,
                display.temporal_dynamics[:, component],
                color="#202020",
                linewidth=1.3,
            )
            dynamics_axis.set_xlim(
                float(result.dataset.time[0]), float(result.dataset.time[-1])
            )
            dynamics_axis.set_ylim(-1.06 * temporal_limit, 1.06 * temporal_limit)
            dynamics_axis.set_xticks([])
            dynamics_axis.set_yticks([])
            for axis in (image_axis, dynamics_axis):
                for spine in axis.spines.values():
                    spine.set_color("#2b2b2b")
                    spine.set_linewidth(1.0)
    figure.text(0.244, 0.955, "Current VP-SPCA", ha="center", va="center", fontsize=16)
    figure.text(0.759, 0.955, "PCA", ha="center", va="center", fontsize=16)
    figure.text(
        0.5,
        0.012,
        "Exact historical GitHub smooth-video DGP; GitHub-derived, not exact paper data",
        ha="center",
        fontsize=8,
        color="#565656",
    )
    figure.savefig(path, dpi=result.config.dpi, facecolor="white")
    plt.close(figure)
    return {
        "layout": "paper-style side-by-side 2x3 method blocks",
        "methods": list(methods),
        "spatial_colormap": "custom white-pink-purple",
        "spatial_per_component_autoscale": True,
        "truth_overlay": False,
        "temporal_shared_abs_limit": float(temporal_limit),
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_bytes(repo_root: Path, args: Sequence[str]) -> bytes | None:
    try:
        complete = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return complete.stdout


def _git_text(repo_root: Path, args: Sequence[str]) -> str | None:
    value = _git_bytes(repo_root, args)
    return None if value is None else value.decode("utf-8", errors="replace").strip()


def _provenance(repo_root: Path) -> dict[str, Any]:
    current_paths = {
        "experiments/video_clean_mode_recovery.py": "runner",
        "experiments/video_mode_recovery_reference.py": "shared_fit_and_gauge_helpers",
        "src/rspca/vp.py": "current_vp_spca_solver",
        "src/rspca/datasets.py": "array_hash_and_runtime_provenance",
        "src/rspca/metrics.py": "reconstruction_metric",
    }
    current = {
        relative: {"sha256": _file_sha256(repo_root / relative), "role": role}
        for relative, role in current_paths.items()
    }
    legacy: dict[str, Any] = {}
    for relative, expected in LEGACY_SOURCE_SHA256.items():
        content = _git_bytes(repo_root, ["show", f"{LEGACY_SOURCE_COMMIT}:{relative}"])
        actual = None if content is None else sha256(content).hexdigest()
        legacy[relative] = {
            "expected_sha256": expected,
            "local_git_object_sha256": actual,
            "verified": actual == expected,
        }
    return {
        "historical_git_commit": LEGACY_SOURCE_COMMIT,
        "historical_sources": legacy,
        "current_executed_source": current,
        "git_head": _git_text(repo_root, ["rev-parse", "HEAD"]),
        "runtime_versions": runtime_versions(),
    }


def _raw_arrays(result: HistoricalGitHubFigure5Result) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "X": result.dataset.data,
        "X_uncentered": result.dataset.uncentered_data,
        "phi_true": result.phi_true_images,
        "phi_true_flat": result.dataset.spatial_modes,
        "a_true": result.a_true_raw,
        "a_true_centered": result.dataset.temporal_centered.T,
        "time": result.dataset.time,
    }
    for method, prefix in (("VP-SPCA", "vp"), ("PCA", "pca")):
        fit = result.fits[method]
        display = result.displays[method]
        legacy_display = _legacy_github_display(result, method)
        arrays.update(
            {
                f"{prefix}_axes": fit.axes,
                f"{prefix}_loadings": fit.loadings,
                f"{prefix}_spatial_modes": fit.spatial_modes,
                f"{prefix}_temporal_dynamics": fit.temporal_dynamics,
                f"{prefix}_reconstruction": fit.reconstruction,
                f"{prefix}_display_spatial_modes": display.spatial_modes,
                f"{prefix}_display_temporal_dynamics": display.temporal_dynamics,
                f"{prefix}_display_scales": display.estimate_only_scales,
                f"{prefix}_permutation": np.asarray(display.permutation, dtype=np.int64),
                f"{prefix}_signs": np.asarray(display.signs, dtype=np.int64),
                f"{prefix}_legacy_display_spatial_modes": (
                    legacy_display.spatial_modes
                ),
                f"{prefix}_legacy_display_temporal_gauge": (
                    legacy_display.temporal_gauge
                ),
                f"{prefix}_legacy_display_temporal_plot": (
                    legacy_display.temporal_plot
                ),
                f"{prefix}_legacy_display_spatial_l2_norms": (
                    legacy_display.spatial_l2_norms
                ),
                f"{prefix}_legacy_display_truth_ls_scales": (
                    legacy_display.truth_ls_scales
                ),
            }
        )
    return arrays


def _write_raw(result: HistoricalGitHubFigure5Result, path: Path) -> dict[str, str]:
    arrays = _raw_arrays(result)
    np.savez_compressed(path, **arrays)
    return {name: array_sha256(value) for name, value in arrays.items()}


def _build_report(
    result: HistoricalGitHubFigure5Result,
    array_hashes: Mapping[str, str],
    display: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    actual_default = {name: array_hashes[name] for name in EXPECTED_DEFAULT_ARRAY_SHA256}
    default_hash_contract_applies = result.config.grid_size == 200
    default_hash_contract_verified = (
        actual_default == EXPECTED_DEFAULT_ARRAY_SHA256
        if default_hash_contract_applies
        else None
    )
    return {
        "schema_version": 1,
        "experiment": "current methods on exact historical GitHub smooth-video DGP",
        "protocol_identity": PROTOCOL_IDENTITY,
        "paper_scale_dimensions": result.config.grid_size == 200,
        "config": asdict(result.config),
        "protocol": {
            "orientation": "spatial_pixels_x_time_snapshots",
            "clean_only": True,
            "generator": "operation-for-operation port of commit e01c02f default generate_multiscale_video",
            "time_formula": "np.linspace(0.0, 150.0, 300)",
            "realized_time_step_seconds": float(
                result.dataset.time[1] - result.dataset.time[0]
            ),
            "paper_text_says_dt_seconds": 0.5,
            "endpoint_convention_differs_from_paper_text": True,
            "methods": ["VP-SPCA", "PCA"],
            "vp_solver": "current rspca.vp.fit_vp_spca",
            "vp_parameters_frozen_from_historical_comparison": True,
            "result_driven_tuning": False,
            "truth_alignment": "Hungarian permutation and sign only",
            "estimate_only_display_gauge": "spatial maxabs scale transferred to temporal factor",
            "native_fit_truth_dependent_amplitude_refit": False,
            "primary_historical_display_truth_dependent_amplitude_refit": True,
            "primary_historical_display_amplitude_refit_scope": (
                "display-only temporal plotting copy; never native fit, "
                "reconstruction, or metrics"
            ),
            "not_exact_paper_data": True,
            "separate_from_corrupted_robust_study": True,
        },
        "historical_default_hash_contract": {
            "expected": dict(EXPECTED_DEFAULT_ARRAY_SHA256),
            "actual": actual_default,
            "applies_to_this_grid": default_hash_contract_applies,
            "verified": default_hash_contract_verified,
        },
        "data": {
            "shape": list(result.dataset.data.shape),
            "array_sha256": dict(array_hashes),
        },
        "fit_metrics": {
            "VP-SPCA": _method_metrics(result, "VP-SPCA"),
            "PCA": _method_metrics(result, "PCA"),
        },
        "display": dict(display),
        "artifacts": {
            "github_style_current_vp_spca": VP_FIGURE_NAME,
            "github_style_current_pca": PCA_FIGURE_NAME,
            "no_refit_integrity_current_vp_spca": INTEGRITY_FIGURE_NAME,
            "paper_style_combined": COMBINED_FIGURE_NAME,
            "raw_arrays": RAW_NAME,
        },
        "provenance": _provenance(repo_root),
    }


def _readme(report: Mapping[str, Any]) -> str:
    vp = report["fit_metrics"]["VP-SPCA"]
    pca = report["fit_metrics"]["PCA"]
    hashes = report["historical_default_hash_contract"]
    return f"""# Spatial-mode recovery and temporal dynamics

This experiment evaluates spatial-mode recovery and temporal dynamics in a
clean 200 x 200 x 300 synthetic video. It is separate from corrupted-video
decomposition.

## Figures

- `{VP_FIGURE_NAME}` — VP-SPCA using the source reference's 2 x 3 layout.
- `{PCA_FIGURE_NAME}` — PCA on the identical data and display contract.
- `{INTEGRITY_FIGURE_NAME}` — current VP-SPCA with no truth-amplitude refit.
- `{COMBINED_FIGURE_NAME}` — compact paper-style side-by-side comparison.

The top row shows three spatial modes. The bottom row shows their associated
time dynamics. In the two primary GitHub-style figures, dashed orange curves
are the **raw** generator truth and the solid curves use the historical
display-only LS amplitude fit. In the integrity figure, the dashed reference is
the **centered** generator truth and the solid curve has no amplitude refit.

## Frozen current-method result

| Method | clean relative error | mean spatial cosine | mean temporal cosine | converged | iterations |
|---|---:|---:|---:|---:|---:|
| Current VP-SPCA | {vp['clean_relative_reconstruction_error']:.9g} | {vp['spatial_cosine_mean']:.9f} | {vp['temporal_cosine_mean']:.9f} | {vp['converged']} | {vp['n_iter']} |
| PCA | {pca['clean_relative_reconstruction_error']:.9g} | {pca['spatial_cosine_mean']:.9f} | {pca['temporal_cosine_mean']:.9f} | {pca['converged']} | {pca['n_iter']} |

VP-SPCA uses alpha=10, max_iter=500, tol=1e-6, PCA initialization, the current
spectral step policy, and backtracking. These values came from the historical
GitHub comparison and were fixed before this run; there was no result-driven
tuning.

## Display and evidence boundary

The two primary GitHub-style plots faithfully reproduce the historical display
protocol: L2-normalize each fitted spatial column, transfer the norm to its
temporal column, align by Hungarian permutation/sign, and then apply a
truth-dependent least-squares scale to a **display-only temporal plotting
copy**. Every spatial image is independently autoscaled with `magma`. These
choices affect visualization only; they do not alter the native fit,
reconstruction, or reported metrics.

The separate integrity figure uses only an estimate-derived max-absolute
spatial gauge with the scale transferred to time, so it preserves the
reconstruction and performs no truth-amplitude refit. Both contracts and their
arrays/scales are recorded in the JSON/NPZ.

`{RAW_NAME}` stores the historical inputs, both native fits and
reconstructions, display factors/scales, and alignment decisions. The exact
historical X/Phi/a/time hash contract passed: `{hashes['verified']}`.

The generator is exact for commit `{LEGACY_SOURCE_COMMIT[:8]}`'s default smooth
GitHub demo, including `np.linspace(0,150,300)` and its realized step of about
0.501672 seconds. The paper states dt=.5 seconds and does not publish complete
formulas or raw arrays, so this result is **GitHub-derived and Figure-5-style,
not the exact unpublished paper dataset**. It must not be pooled with formal
corrupted-video Robust VP-SPCA metrics.
"""


def save_historical_github_current(
    result: HistoricalGitHubFigure5Result,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    vp_path = destination / VP_FIGURE_NAME
    pca_path = destination / PCA_FIGURE_NAME
    integrity_path = destination / INTEGRITY_FIGURE_NAME
    combined_path = destination / COMBINED_FIGURE_NAME
    raw_path = destination / RAW_NAME
    report_path = destination / REPORT_NAME
    readme_path = destination / README_NAME
    manifest_path = destination / MANIFEST_NAME

    vp_display = _plot_github_method(result, "VP-SPCA", vp_path)
    pca_display = _plot_github_method(result, "PCA", pca_path)
    integrity_display = _plot_integrity_vp(result, integrity_path)
    combined_display = _plot_combined(result, combined_path)
    array_hashes = _write_raw(result, raw_path)
    repo_root = Path(__file__).resolve().parents[1]
    report = _build_report(
        result,
        array_hashes,
        {
            "github_style_current_vp_spca": vp_display,
            "github_style_current_pca": pca_display,
            "no_refit_integrity_current_vp_spca": integrity_display,
            "paper_style_combined": combined_display,
        },
        repo_root,
    )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme_path.write_text(_readme(report), encoding="utf-8")
    payload_paths = (
        vp_path,
        pca_path,
        integrity_path,
        combined_path,
        raw_path,
        report_path,
        readme_path,
    )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "historical_github_current_exact_output_closure",
        "self_excluding": True,
        "files": {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in payload_paths
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validate_historical_github_current_output(destination)
    return {
        "vp": vp_path,
        "pca": pca_path,
        "integrity": integrity_path,
        "combined": combined_path,
        "raw": raw_path,
        "report": report_path,
        "readme": readme_path,
        "manifest": manifest_path,
    }


def validate_historical_github_current_output(
    output_dir: str | Path,
) -> dict[str, Any]:
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
    if {path.name for path in entries} != _OUTPUT_NAMES:
        raise ValueError("output must contain exactly the eight required files")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if set(manifest.get("files", {})) != _PAYLOAD_NAMES:
        raise ValueError("manifest must name exactly the seven required payloads")
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
        raise ValueError("protocol identity mismatch")
    hash_contract = report.get("historical_default_hash_contract", {})
    if report.get("paper_scale_dimensions") is True:
        if hash_contract.get("applies_to_this_grid") is not True:
            raise ValueError("historical default hash contract applicability mismatch")
        if hash_contract.get("verified") is not True:
            raise ValueError("historical default input hashes are not verified")
    protocol = report.get("protocol", {})
    if protocol.get("native_fit_truth_dependent_amplitude_refit") is not False:
        raise ValueError("native fit must not use truth-dependent amplitude refit")
    if protocol.get("primary_historical_display_truth_dependent_amplitude_refit") is not True:
        raise ValueError("primary display must disclose historical truth amplitude fit")
    with np.load(directory / RAW_NAME, allow_pickle=False) as arrays:
        expected = report["data"]["array_sha256"]
        for name, digest in expected.items():
            if name not in arrays.files or array_sha256(arrays[name]) != digest:
                raise ValueError(f"raw array mismatch: {name}")
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
            legacy_gauge_reconstruction = (
                arrays[f"{prefix}_legacy_display_spatial_modes"]
                @ arrays[f"{prefix}_legacy_display_temporal_gauge"].T
            )
            if not np.allclose(
                legacy_gauge_reconstruction,
                arrays[f"{prefix}_reconstruction"],
                rtol=2e-12,
                atol=2e-12,
            ):
                raise ValueError(f"{prefix} historical L2 gauge changed reconstruction")
    return manifest


def smoke_config() -> HistoricalGitHubFigure5Config:
    return HistoricalGitHubFigure5Config(grid_size=24, dpi=80)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    config = smoke_config() if args.smoke else HistoricalGitHubFigure5Config()
    result = run_historical_github_current(config)
    paths = save_historical_github_current(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "historical_input_hashes_verified": (
                    config.grid_size != 200
                    or {
                        "X": array_sha256(result.dataset.data),
                        "phi_true": array_sha256(result.phi_true_images),
                        "a_true": array_sha256(result.a_true_raw),
                        "time": array_sha256(result.dataset.time),
                    }
                    == EXPECTED_DEFAULT_ARRAY_SHA256
                ),
                "vp_spca": _method_metrics(result, "VP-SPCA"),
                "pca": _method_metrics(result, "PCA"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()

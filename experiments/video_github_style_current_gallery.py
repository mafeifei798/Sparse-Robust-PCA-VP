"""Render the current video algorithms in the historical GitHub visual style.

The clean branch uses the exact default smooth-video generator from commit
``e01c02f`` and renders current PCA, VP-SPCA, and randomized VP-SPCA with the
old 2 x 3 mode-and-dynamics grammar.  As in the historical figures, the
primary curves use a truth-dependent least-squares amplitude fit on a
display-only copy; native factors, reconstructions, and metrics do not use it.

The robust branch is deliberately separate.  It replays the already-selected
continuation candidate for confirmation seed 51001 from the sealed formal
study.  The selected start, alpha, kappa schedule, and iteration budgets are
read from that closed report and are verified by exact A/B/S and reconstruction
hashes.  Its images use native reconstruction and native S without thresholding
or truth-dependent fitting.

No result-driven tuning or image selection is performed.  The writer refuses
to overwrite an output directory and emits raw display/factor arrays, a report,
and an exact-closure manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import stat
import subprocess
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.export_locked_robust_gallery_frames import (
    NPZ_NAME as ROBUST_EXPORT_NPZ_NAME,
    REPORT_NAME as ROBUST_EXPORT_REPORT_NAME,
    validate_locked_frames_export,
)
from experiments.video_figure5_historical_github_current import (
    EXPECTED_DEFAULT_ARRAY_SHA256,
    HistoricalGitHubFigure5Config,
    HistoricalGitHubFigure5Result,
    LegacyGitHubDisplayFactor,
    _legacy_github_display,
    generate_exact_historical_github_dataset,
    run_historical_github_current,
)
from experiments.video_figure5_paper_style import (
    FactorFit,
    _method_metrics,
    align_and_apply_estimate_only_gauge,
)
from experiments.video_spca_paper_reproduction import generate_figure6_dataset
from experiments.video_spca_paper_robust_study import (
    RobustStudyConfig,
    _candidate_raw_arrays,
    _fit_candidate,
    _paper_video_config,
    build_initializations,
    default_start_specs,
    prepare_observed_case,
    validate_confirmation_output_directory,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.randomized import RandomizedVPSPCAResult, fit_randomized_vp_spca


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIRMATION_DIR = (
    REPO_ROOT
    / "results"
    / "video_paper_robust_study_20260830"
    / "formal_confirm_v1"
)

PCA_FIGURE_NAME = "github_style_current_pca.png"
VP_FIGURE_NAME = "github_style_current_vp_spca.png"
RND_FIGURE_NAME = "github_style_current_rnd_vp_spca.png"
CLEAN_CONTACT_NAME = "github_style_clean_methods_contact_sheet.png"
CLEAN_SUMMARY_NAME = "github_style_clean_quality_summary.png"
ROBUST_FIGURE_NAME = "github_style_current_robust_vp_spca.png"
ROBUST_TWO_FRAME_NAME = "github_style_current_robust_vp_spca_two_frames.png"
RAW_NAME = "current_github_style_gallery_arrays.npz"
REPORT_NAME = "current_github_style_gallery_report.json"
README_NAME = "README.md"
MANIFEST_NAME = "current_github_style_gallery_manifest.json"

_FIGURE_NAMES = frozenset(
    {
        PCA_FIGURE_NAME,
        VP_FIGURE_NAME,
        RND_FIGURE_NAME,
        CLEAN_CONTACT_NAME,
        CLEAN_SUMMARY_NAME,
        ROBUST_FIGURE_NAME,
    }
)
_PAYLOAD_NAMES = _FIGURE_NAMES | {RAW_NAME, REPORT_NAME, README_NAME}
_OUTPUT_NAMES = _PAYLOAD_NAMES | {MANIFEST_NAME}

PROTOCOL_IDENTITY = (
    "current PCA, VP-SPCA, and randomized VP-SPCA on the exact historical "
    "GitHub smooth clean DGP, plus exact replay of the sealed locked robust "
    "continuation candidate for confirmation seed 51001"
)


@dataclass(frozen=True)
class CurrentGitHubGalleryConfig:
    grid_size: int = 200
    n_snapshots: int = 300
    total_time: float = 150.0
    rank: int = 3
    clean_alpha: float = 10.0
    clean_max_iter: int = 500
    clean_tol: float = 1e-6
    randomized_oversample: int = 100
    randomized_power_iters: int = 4
    randomized_full_refine_iters: int = 0
    randomized_seed: int = 42
    robust_seed: int = 51001
    robust_start_id: str = "pca_perturb_2"
    robust_alpha: float = 0.0015
    robust_target_kappa: float = 0.001
    robust_kappa_schedule: tuple[float, ...] = (0.004, 0.001)
    robust_stage_budgets: tuple[int, ...] = (40, 160)
    representative_frame_index: int = 100
    worst_frame_index: int = 171
    formal_robust_replay: bool = True
    dpi: int = 200


@dataclass(frozen=True)
class RobustGalleryFit:
    seed: int
    clean: np.ndarray
    observed: np.ndarray
    corruption: np.ndarray
    corruption_support: np.ndarray
    time: np.ndarray
    reconstruction: np.ndarray
    outliers: np.ndarray
    A: np.ndarray
    B: np.ndarray
    S_normalized: np.ndarray | None
    normalization_sigma_max: float
    frame_indices: tuple[int, ...]
    candidate_record: Mapping[str, Any]
    sealed_expectation: Mapping[str, Any] | None
    actual_full_array_hashes: Mapping[str, str]
    exact_replay_verified: bool | None
    replay_provenance: Mapping[str, Any]


@dataclass(frozen=True)
class CurrentGitHubGalleryResult:
    config: CurrentGitHubGalleryConfig
    clean_result: HistoricalGitHubFigure5Result
    clean_legacy_displays: Mapping[str, LegacyGitHubDisplayFactor]
    randomized_native: RandomizedVPSPCAResult
    robust: RobustGalleryFit


def _validate_config(config: CurrentGitHubGalleryConfig) -> None:
    integer_fields = {
        "grid_size": config.grid_size,
        "n_snapshots": config.n_snapshots,
        "rank": config.rank,
        "clean_max_iter": config.clean_max_iter,
        "randomized_oversample": config.randomized_oversample,
        "randomized_power_iters": config.randomized_power_iters,
        "randomized_full_refine_iters": config.randomized_full_refine_iters,
        "randomized_seed": config.randomized_seed,
        "robust_seed": config.robust_seed,
        "representative_frame_index": config.representative_frame_index,
        "worst_frame_index": config.worst_frame_index,
        "dpi": config.dpi,
    }
    for name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
    if config.grid_size < 16:
        raise ValueError("grid_size must be at least 16")
    if config.n_snapshots != 300 or config.total_time != 150.0:
        raise ValueError("the historical video uses 300 snapshots over 150 seconds")
    if config.rank != 3:
        raise ValueError("the historical video has rank 3")
    if (
        config.clean_alpha != 10.0
        or config.clean_max_iter != 500
        or config.clean_tol != 1e-6
    ):
        raise ValueError("the clean historical comparison fixes alpha=10, 500, 1e-6")
    if config.randomized_oversample < 0 or config.randomized_power_iters < 0:
        raise ValueError("randomized sketch parameters must be nonnegative")
    if config.randomized_full_refine_iters < 0:
        raise ValueError("randomized full refinement must be nonnegative")
    if len(config.robust_kappa_schedule) != len(config.robust_stage_budgets):
        raise ValueError("robust schedule and budgets must have equal length")
    if not config.robust_kappa_schedule:
        raise ValueError("robust schedule must be nonempty")
    if config.robust_kappa_schedule[-1] != config.robust_target_kappa:
        raise ValueError("robust schedule must end at the target kappa")
    if any(value <= 0 for value in config.robust_stage_budgets):
        raise ValueError("robust stage budgets must be positive")
    if not 0 <= config.representative_frame_index < config.n_snapshots:
        raise ValueError("representative frame is out of range")
    if not 0 <= config.worst_frame_index < config.n_snapshots:
        raise ValueError("worst frame is out of range")
    if config.dpi <= 0:
        raise ValueError("dpi must be positive")
    if config.formal_robust_replay:
        expected = CurrentGitHubGalleryConfig()
        locked_fields = (
            "grid_size",
            "n_snapshots",
            "total_time",
            "robust_seed",
            "robust_start_id",
            "robust_alpha",
            "robust_target_kappa",
            "robust_kappa_schedule",
            "robust_stage_budgets",
            "representative_frame_index",
            "worst_frame_index",
        )
        for field in locked_fields:
            if getattr(config, field) != getattr(expected, field):
                raise ValueError(f"formal robust replay changed locked field: {field}")


def _historical_clean_config(
    config: CurrentGitHubGalleryConfig,
) -> HistoricalGitHubFigure5Config:
    return HistoricalGitHubFigure5Config(
        grid_size=config.grid_size,
        n_snapshots=config.n_snapshots,
        total_time=config.total_time,
        rank=config.rank,
        alpha=config.clean_alpha,
        max_iter=config.clean_max_iter,
        tol=config.clean_tol,
        dpi=config.dpi,
    )


def _fit_randomized_clean(
    clean: HistoricalGitHubFigure5Result,
    config: CurrentGitHubGalleryConfig,
) -> tuple[FactorFit, RandomizedVPSPCAResult]:
    started = perf_counter()
    randomized = fit_randomized_vp_spca(
        clean.dataset.data,
        config.rank,
        alpha=config.clean_alpha,
        oversample=config.randomized_oversample,
        n_power_iter=config.randomized_power_iters,
        max_iter=config.clean_max_iter,
        tol=config.clean_tol,
        finalize_full_iters=config.randomized_full_refine_iters,
        random_state=config.randomized_seed,
        verbose=False,
    )
    runtime = perf_counter() - started
    spatial = np.ascontiguousarray(clean.dataset.data @ randomized.B)
    reconstruction = np.ascontiguousarray(spatial @ randomized.A.T)
    certificate = (
        None
        if randomized.certificate is None
        else asdict(randomized.certificate)
    )
    fit = FactorFit(
        method="RND-VP-SPCA",
        axes=np.ascontiguousarray(randomized.A),
        loadings=np.ascontiguousarray(randomized.B),
        spatial_modes=spatial,
        temporal_dynamics=np.ascontiguousarray(randomized.A),
        reconstruction=reconstruction,
        runtime_seconds=float(runtime),
        converged=bool(randomized.full_converged),
        n_iter=int(randomized.n_iter),
        final_stationarity=float(randomized.full_stationarity),
        status=randomized.status,
        objectives=(),
        extra={
            "alpha": float(randomized.alpha),
            "oversample": config.randomized_oversample,
            "n_power_iter": config.randomized_power_iters,
            "random_state": config.randomized_seed,
            "sketch_converged": randomized.sketch_converged,
            "full_converged": bool(randomized.full_converged),
            "sketch_residual_relative": float(
                randomized.sketch_residual_relative
            ),
            "full_stationarity": float(randomized.full_stationarity),
            "sketch_iterations": int(randomized.sketch_iterations),
            "full_refinement_iterations": int(
                randomized.full_refinement_iterations
            ),
            "timings": dict(randomized.timings),
            "certificate": certificate,
        },
    )
    return fit, randomized


def _sealed_robust_expectation(seed: int) -> dict[str, Any]:
    validate_confirmation_output_directory(FORMAL_CONFIRMATION_DIR)
    report_path = FORMAL_CONFIRMATION_DIR / "confirmation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cases = [case for case in report["case_records"] if int(case["seed"]) == seed]
    if len(cases) != 1:
        raise RuntimeError("sealed confirmation seed is absent or duplicated")
    case = cases[0]
    arm = case["arms"]["B_only_descending_kappa_multistart"]
    selected = arm["candidate_records"][int(arm["selected_index"])]
    method = case["methods"]["continuation_multistart"]
    return {
        "formal_report_sha256": _file_sha256(report_path),
        "formal_manifest_sha256": _file_sha256(
            FORMAL_CONFIRMATION_DIR / "confirmation_output_manifest.json"
        ),
        "seed": int(case["seed"]),
        "data": dict(case["data"]),
        "selected_start_id": arm["selected_start_id"],
        "selected_index": int(arm["selected_index"]),
        "selected_fit_sha256": arm["selected_fit_sha256"],
        "candidate": dict(selected),
        "method": dict(method),
    }


def _robust_study_config(
    config: CurrentGitHubGalleryConfig,
) -> RobustStudyConfig:
    base = RobustStudyConfig()
    return replace(
        base,
        grid_size=config.grid_size,
        n_timesteps=config.n_snapshots,
        total_time=config.total_time,
        max_iter_per_start=sum(config.robust_stage_budgets),
        continuation_kappa_multipliers=tuple(
            value / config.robust_target_kappa
            for value in config.robust_kappa_schedule
        ),
        continuation_stage_max_iters=config.robust_stage_budgets,
        representative_frame_index=config.representative_frame_index,
        dpi=config.dpi,
    )


def _fit_robust_gallery(
    config: CurrentGitHubGalleryConfig,
) -> RobustGalleryFit:
    study = _robust_study_config(config)
    paper_config = _paper_video_config(study, config.robust_seed)
    dataset = generate_figure6_dataset(paper_config)
    observed = prepare_observed_case(dataset)
    specs = default_start_specs(study)
    spec_by_id = {spec.start_id: spec for spec in specs}
    if config.robust_start_id not in spec_by_id:
        raise RuntimeError("requested robust start is not in the locked start bank")
    initializations = build_initializations(observed, specs)
    candidate = _fit_candidate(
        observed,
        start_id=config.robust_start_id,
        arm_id="B_only_descending_kappa_multistart",
        initial_B=initializations[config.robust_start_id],
        alpha=config.robust_alpha,
        target_kappa=config.robust_target_kappa,
        kappa_schedule=config.robust_kappa_schedule,
        stage_budgets=config.robust_stage_budgets,
        config=study,
    )
    reconstruction, outliers = _candidate_raw_arrays(observed, candidate)
    expectation = (
        _sealed_robust_expectation(config.robust_seed)
        if config.formal_robust_replay
        else None
    )
    actual_hashes = {
        "clean": array_sha256(dataset.clean),
        "observed": array_sha256(dataset.observed),
        "corruption": array_sha256(dataset.corruption),
        "corruption_support": array_sha256(dataset.corruption_support),
        "A": array_sha256(candidate.result.A),
        "B": array_sha256(candidate.result.B),
        "S_normalized": array_sha256(candidate.result.S),
        "reconstruction": array_sha256(reconstruction),
        "outliers_raw_units": array_sha256(outliers),
    }
    exact_verified: bool | None = None
    if expectation is not None:
        expected_candidate = expectation["candidate"]
        candidate_hashes = {
            "final_A_sha256": actual_hashes["A"],
            "final_B_sha256": actual_hashes["B"],
            "final_S_sha256_normalized": actual_hashes["S_normalized"],
        }
        for name, digest in candidate_hashes.items():
            if digest != expected_candidate[name]:
                raise RuntimeError(f"formal robust replay mismatch: {name}")
        fit_sha = canonical_sha256(
            {
                "A": candidate_hashes["final_A_sha256"],
                "B": candidate_hashes["final_B_sha256"],
                "S": candidate_hashes["final_S_sha256_normalized"],
            }
        )
        if fit_sha != expectation["selected_fit_sha256"]:
            raise RuntimeError("formal selected fit hash mismatch")
        method = expectation["method"]
        if actual_hashes["reconstruction"] != method["reconstruction_sha256"]:
            raise RuntimeError("formal robust reconstruction hash mismatch")
        if (
            actual_hashes["outliers_raw_units"]
            != method["outlier_scores_sha256_raw_units"]
        ):
            raise RuntimeError("formal robust outlier hash mismatch")
        data_expected = expectation["data"]
        data_actual = {
            "clean_sha256": actual_hashes["clean"],
            "observed_sha256": actual_hashes["observed"],
            "corruption_sha256": actual_hashes["corruption"],
            "corruption_support_sha256": actual_hashes[
                "corruption_support"
            ],
        }
        for name, digest in data_actual.items():
            if digest != data_expected[name]:
                raise RuntimeError(f"formal robust input mismatch: {name}")
        exact_verified = True
    return RobustGalleryFit(
        seed=config.robust_seed,
        clean=np.asarray(dataset.clean, dtype=np.float64),
        observed=np.asarray(dataset.observed, dtype=np.float64),
        corruption=np.asarray(dataset.corruption, dtype=np.float64),
        corruption_support=np.asarray(dataset.corruption_support, dtype=np.bool_),
        time=np.asarray(dataset.time, dtype=np.float64),
        reconstruction=np.asarray(reconstruction, dtype=np.float64),
        outliers=np.asarray(outliers, dtype=np.float64),
        A=np.asarray(candidate.result.A, dtype=np.float64),
        B=np.asarray(candidate.result.B, dtype=np.float64),
        S_normalized=np.asarray(candidate.result.S, dtype=np.float64),
        normalization_sigma_max=float(observed.normalization_sigma_max),
        frame_indices=tuple(range(config.n_snapshots)),
        candidate_record=dict(candidate.record),
        sealed_expectation=expectation,
        actual_full_array_hashes=actual_hashes,
        exact_replay_verified=exact_verified,
        replay_provenance={"kind": "local_in_process_replay"},
    )


def _load_exported_robust_gallery(
    config: CurrentGitHubGalleryConfig,
    export_dir: str | Path,
) -> RobustGalleryFit:
    directory = Path(export_dir)
    manifest = validate_locked_frames_export(directory)
    report = json.loads(
        (directory / ROBUST_EXPORT_REPORT_NAME).read_text(encoding="utf-8")
    )
    if report.get("exact_full_array_replay_verified") is not True:
        raise RuntimeError("external robust replay is not exact-verified")
    protocol = report.get("protocol", {})
    expected_protocol = {
        "seed": config.robust_seed,
        "start_id": config.robust_start_id,
        "alpha": config.robust_alpha,
        "target_kappa": config.robust_target_kappa,
        "kappa_schedule": list(config.robust_kappa_schedule),
        "stage_budgets": list(config.robust_stage_budgets),
        "frame_indices": [
            config.representative_frame_index,
            config.worst_frame_index,
        ],
    }
    for name, value in expected_protocol.items():
        if protocol.get(name) != value:
            raise RuntimeError(f"external robust replay protocol mismatch: {name}")
    expectation = _sealed_robust_expectation(config.robust_seed)
    exported_expectation = report.get("sealed_expectation", {})
    for name in (
        "formal_report_sha256",
        "formal_manifest_sha256",
        "selected_start_id",
        "selected_index",
        "selected_fit_sha256",
    ):
        if exported_expectation.get(name) != expectation[name]:
            raise RuntimeError(f"external sealed expectation mismatch: {name}")
    actual = report.get("actual_full_array_hashes", {})
    expected_hashes = {
        "clean": expectation["data"]["clean_sha256"],
        "observed": expectation["data"]["observed_sha256"],
        "corruption": expectation["data"]["corruption_sha256"],
        "corruption_support": expectation["data"][
            "corruption_support_sha256"
        ],
        "A": expectation["candidate"]["final_A_sha256"],
        "B": expectation["candidate"]["final_B_sha256"],
        "S_normalized": expectation["candidate"][
            "final_S_sha256_normalized"
        ],
        "reconstruction": expectation["method"]["reconstruction_sha256"],
        "outliers_raw_units": expectation["method"][
            "outlier_scores_sha256_raw_units"
        ],
        "selected_fit": expectation["selected_fit_sha256"],
    }
    if actual != expected_hashes:
        raise RuntimeError("external robust replay full-array hashes changed")
    with np.load(directory / ROBUST_EXPORT_NPZ_NAME, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    frame_indices = tuple(int(value) for value in arrays["frame_indices"])
    if frame_indices != (
        config.representative_frame_index,
        config.worst_frame_index,
    ):
        raise RuntimeError("external robust replay frame indices changed")
    return RobustGalleryFit(
        seed=config.robust_seed,
        clean=np.asarray(arrays["clean_frames"], dtype=np.float64),
        observed=np.asarray(arrays["observed_frames"], dtype=np.float64),
        corruption=np.empty((0, 0), dtype=np.float64),
        corruption_support=np.empty((0, 0), dtype=np.bool_),
        time=np.asarray(arrays["time"], dtype=np.float64),
        reconstruction=np.asarray(
            arrays["reconstruction_frames"], dtype=np.float64
        ),
        outliers=np.asarray(arrays["outlier_frames"], dtype=np.float64),
        A=np.asarray(arrays["A"], dtype=np.float64),
        B=np.asarray(arrays["B"], dtype=np.float64),
        S_normalized=None,
        normalization_sigma_max=float(arrays["normalization_sigma_max"]),
        frame_indices=frame_indices,
        candidate_record=dict(report["candidate_record"]),
        sealed_expectation=expectation,
        actual_full_array_hashes=dict(actual),
        exact_replay_verified=True,
        replay_provenance={
            "kind": "external_sealed_environment_frame_export",
            "export_manifest_sha256": manifest["manifest_sha256"],
            "export_manifest_file_sha256": _file_sha256(
                directory / "locked_robust_gallery_export_manifest.json"
            ),
            "export_report_file_sha256": _file_sha256(
                directory / ROBUST_EXPORT_REPORT_NAME
            ),
            "export_npz_file_sha256": _file_sha256(
                directory / ROBUST_EXPORT_NPZ_NAME
            ),
            "source_provenance": report["source_provenance"],
        },
    )


def run_current_github_gallery(
    config: CurrentGitHubGalleryConfig | None = None,
    *,
    robust_replay_dir: str | Path | None = None,
) -> CurrentGitHubGalleryResult:
    resolved = CurrentGitHubGalleryConfig() if config is None else config
    _validate_config(resolved)
    clean = run_historical_github_current(_historical_clean_config(resolved))
    rnd_fit, rnd_native = _fit_randomized_clean(clean, resolved)
    fits = dict(clean.fits)
    fits["RND-VP-SPCA"] = rnd_fit
    displays = dict(clean.displays)
    displays["RND-VP-SPCA"] = align_and_apply_estimate_only_gauge(
        clean.dataset.spatial_modes,
        rnd_fit.spatial_modes,
        rnd_fit.temporal_dynamics,
    )
    clean_all = HistoricalGitHubFigure5Result(
        config=clean.config,
        dataset=clean.dataset,
        phi_true_images=clean.phi_true_images,
        a_true_raw=clean.a_true_raw,
        fits=fits,
        displays=displays,
    )
    legacy_displays = {
        method: _legacy_github_display(clean_all, method)
        for method in ("PCA", "VP-SPCA", "RND-VP-SPCA")
    }
    robust = (
        _fit_robust_gallery(resolved)
        if robust_replay_dir is None
        else _load_exported_robust_gallery(resolved, robust_replay_dir)
    )
    return CurrentGitHubGalleryResult(
        config=resolved,
        clean_result=clean_all,
        clean_legacy_displays=legacy_displays,
        randomized_native=rnd_native,
        robust=robust,
    )


_METHOD_SPECS = {
    "PCA": ("Current PCA", "#4b5563"),
    "VP-SPCA": ("Current VP-SPCA", "#2563eb"),
    "RND-VP-SPCA": ("Current RND-VP-SPCA", "#16a34a"),
}


def _plot_clean_method(
    result: CurrentGitHubGalleryResult,
    method: str,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    label, color = _METHOD_SPECS[method]
    display = result.clean_legacy_displays[method]
    truth = result.clean_result.a_true_raw.T
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), facecolor="white")
    for component in range(3):
        image_axis = axes[0, component]
        image = image_axis.imshow(
            display.spatial_modes[:, component].reshape(
                result.config.grid_size, result.config.grid_size
            ),
            cmap="magma",
            interpolation="bilinear",
        )
        image_axis.set_title(f"{label}  $\\Phi_{component + 1}$", fontsize=13)
        image_axis.axis("off")
        figure.colorbar(image, ax=image_axis, fraction=0.046, pad=0.035)
        time_axis = axes[1, component]
        time_axis.plot(
            result.clean_result.dataset.time,
            display.temporal_plot[:, component],
            color=color,
            linewidth=1.55,
            label=label,
        )
        time_axis.plot(
            result.clean_result.dataset.time,
            truth[:, component],
            color="#f97316",
            linestyle="--",
            linewidth=1.15,
            alpha=0.92,
            label="raw generator truth",
        )
        time_axis.set_xlim(
            float(result.clean_result.dataset.time[0]),
            float(result.clean_result.dataset.time[-1]),
        )
        time_axis.set_xlabel("time (s)")
        time_axis.grid(True, alpha=0.22)
        if component == 0:
            time_axis.set_ylabel("Activation")
            time_axis.legend(fontsize=8.5, loc="best")
    figure.suptitle(
        f"{label} — exact historical GitHub smooth-video DGP",
        fontsize=17,
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        "Historical display protocol: L2 spatial gauge + display-only truth LS amplitude fit",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.975))
    figure.savefig(path, dpi=result.config.dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "method": method,
        "layout": "historical GitHub 2x3",
        "spatial_colormap": "magma",
        "spatial_per_component_autoscale": True,
        "truth_overlay": "raw generator truth, dashed",
        "truth_dependent_amplitude_refit": True,
        "amplitude_refit_scope": "display-only temporal plotting copy",
        "native_fit_or_metrics_changed": False,
        "truth_ls_amplitude_scales": [
            float(value) for value in display.truth_ls_scales
        ],
    }


def _plot_clean_contact_sheet(
    result: CurrentGitHubGalleryResult,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("PCA", "VP-SPCA", "RND-VP-SPCA")
    truth = result.clean_result.a_true_raw.T
    figure, axes = plt.subplots(3, 6, figsize=(24, 12), facecolor="white")
    for row, method in enumerate(methods):
        label, color = _METHOD_SPECS[method]
        display = result.clean_legacy_displays[method]
        for component in range(3):
            image_axis = axes[row, component]
            image_axis.imshow(
                display.spatial_modes[:, component].reshape(
                    result.config.grid_size, result.config.grid_size
                ),
                cmap="magma",
                interpolation="bilinear",
            )
            image_axis.axis("off")
            image_axis.set_title(
                f"$\\Phi_{component + 1}$" if row == 0 else "",
                fontsize=12,
            )
            time_axis = axes[row, component + 3]
            time_axis.plot(
                result.clean_result.dataset.time,
                display.temporal_plot[:, component],
                color=color,
                linewidth=1.35,
            )
            time_axis.plot(
                result.clean_result.dataset.time,
                truth[:, component],
                color="#f97316",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
            )
            time_axis.set_xlim(0.0, result.config.total_time)
            time_axis.grid(True, alpha=0.2)
            if row == 0:
                time_axis.set_title(f"$a_{component + 1}(t)$", fontsize=12)
            if row == len(methods) - 1:
                time_axis.set_xlabel("time (s)")
        axes[row, 0].text(
            -0.08,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=13,
            fontweight="bold",
            color=color,
        )
    figure.suptitle(
        "Current clean-video methods — same exact historical GitHub data and display contract",
        fontsize=18,
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        "Solid: current estimate   |   dashed orange: raw generator truth   |   display-only LS amplitude fit",
        ha="center",
        fontsize=10,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.02, 0.025, 1.0, 0.97), h_pad=1.5, w_pad=1.0)
    figure.savefig(path, dpi=result.config.dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "layout": "3 methods x (3 spatial + 3 temporal) contact sheet",
        "methods": list(methods),
        "truth_dependent_amplitude_refit": True,
        "refit_scope": "display-only",
    }


def _clean_metrics(result: CurrentGitHubGalleryResult) -> dict[str, Any]:
    return {
        method: _method_metrics(result.clean_result, method)
        for method in ("PCA", "VP-SPCA", "RND-VP-SPCA")
    }


def _plot_clean_summary(
    result: CurrentGitHubGalleryResult,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("PCA", "VP-SPCA", "RND-VP-SPCA")
    labels = ["PCA", "VP-SPCA", "RND-VP-SPCA"]
    colors = [_METHOD_SPECS[method][1] for method in methods]
    metrics = _clean_metrics(result)
    errors = [metrics[method]["clean_relative_reconstruction_error"] for method in methods]
    spatial = [metrics[method]["spatial_cosine_mean"] for method in methods]
    temporal = [metrics[method]["temporal_cosine_mean"] for method in methods]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="white")
    bars = axes[0].bar(labels, errors, color=colors)
    axes[0].set_yscale("log")
    axes[0].set_title("Native clean reconstruction error")
    axes[0].set_ylabel("relative error (log)")
    axes[0].grid(True, axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, errors):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for axis, values, title in (
        (axes[1], spatial, "Spatial-mode cosine"),
        (axes[2], temporal, "Temporal-dynamics cosine"),
    ):
        bars = axis.bar(labels, values, color=colors)
        lower = max(0.0, min(values) - 0.004)
        axis.set_ylim(lower, 1.0005)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.6f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        "Native metrics on the exact historical GitHub clean video",
        fontsize=16,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=result.config.dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "metrics_are_native": True,
        "truth_ls_display_fit_used_in_metrics": False,
        "runtime_intentionally_omitted": "fit endpoints have different preparation scopes",
    }


def _robust_display_scale(
    clean_frames: Sequence[np.ndarray],
) -> tuple[float, float]:
    values = np.concatenate([np.asarray(frame).reshape(-1) for frame in clean_frames])
    lower, upper = np.percentile(values, [1.0, 99.0])
    if not lower < upper:
        bound = max(float(np.max(np.abs(values))), np.finfo(np.float64).eps)
        return -bound, bound
    return float(lower), float(upper)


def _robust_outlier_limit(frames: Sequence[np.ndarray]) -> float:
    values = np.concatenate([np.abs(np.asarray(frame)).reshape(-1) for frame in frames])
    nonzero = values[values > 0]
    pool = nonzero if nonzero.size else values
    return max(float(np.percentile(pool, 98.0)), np.finfo(np.float64).eps)


def _robust_frame(
    robust: RobustGalleryFit,
    field: str,
    frame_index: int,
) -> np.ndarray:
    try:
        stored_index = robust.frame_indices.index(frame_index)
    except ValueError as error:
        raise RuntimeError(
            f"robust replay does not contain requested frame {frame_index}"
        ) from error
    values = np.asarray(getattr(robust, field))
    if values.ndim != 2 or values.shape[1] != len(robust.frame_indices):
        raise RuntimeError(f"robust replay field has unexpected shape: {field}")
    return values[:, stored_index]


def _plot_robust_row(
    result: CurrentGitHubGalleryResult,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    robust = result.robust
    index = result.config.representative_frame_index
    grid = result.config.grid_size
    clean = _robust_frame(robust, "clean", index).reshape(grid, grid)
    observed = _robust_frame(robust, "observed", index).reshape(grid, grid)
    reconstruction = _robust_frame(
        robust, "reconstruction", index
    ).reshape(grid, grid)
    outliers = _robust_frame(robust, "outliers", index).reshape(grid, grid)
    vmin, vmax = _robust_display_scale((clean,))
    s_limit = _robust_outlier_limit((outliers,))
    figure, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor="white")
    panels = (
        (clean, "(a) Truth", "magma", vmin, vmax),
        (observed, "(b) Corrupted", "magma", vmin, vmax),
        (reconstruction, "(c) Current Robust reconstruction", "magma", vmin, vmax),
        (outliers, "(d) Native outliers S", "seismic", -s_limit, s_limit),
    )
    for axis, (values, title, cmap, lower, upper) in zip(axes, panels):
        image = axis.imshow(
            values,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=12)
        axis.axis("off")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.035)
    figure.suptitle(
        f"Current locked Robust VP-SPCA — sealed confirmation seed {robust.seed}, frame {index}",
        fontsize=16,
        y=0.98,
    )
    figure.text(
        0.5,
        0.015,
        "Native reconstruction and native S; shared truth-derived signal scale; no display threshold or truth fit",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))
    figure.savefig(path, dpi=result.config.dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "layout": "historical Figure-6-style 1x4",
        "seed": robust.seed,
        "frame_index": index,
        "native_S": True,
        "outlier_post_threshold": False,
        "truth_dependent_fit": False,
        "signal_scale": [vmin, vmax],
        "outlier_abs_limit": s_limit,
    }


def _plot_robust_two_frames(
    result: CurrentGitHubGalleryResult,
    path: Path,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    robust = result.robust
    indices = (
        result.config.representative_frame_index,
        result.config.worst_frame_index,
    )
    grid = result.config.grid_size
    clean_frames = [
        _robust_frame(robust, "clean", index).reshape(grid, grid)
        for index in indices
    ]
    outlier_frames = [
        _robust_frame(robust, "outliers", index).reshape(grid, grid)
        for index in indices
    ]
    vmin, vmax = _robust_display_scale(clean_frames)
    s_limit = _robust_outlier_limit(outlier_frames)
    figure, axes = plt.subplots(2, 4, figsize=(16, 9), facecolor="white")
    for row, index in enumerate(indices):
        panels = (
            _robust_frame(robust, "clean", index).reshape(grid, grid),
            _robust_frame(robust, "observed", index).reshape(grid, grid),
            _robust_frame(robust, "reconstruction", index).reshape(grid, grid),
            _robust_frame(robust, "outliers", index).reshape(grid, grid),
        )
        for column, values in enumerate(panels):
            axis = axes[row, column]
            if column < 3:
                axis.imshow(
                    values,
                    cmap="magma",
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    values,
                    cmap="seismic",
                    vmin=-s_limit,
                    vmax=s_limit,
                    interpolation="nearest",
                )
            axis.axis("off")
            if row == 0:
                axis.set_title(
                    ("Truth", "Corrupted", "Current Robust reconstruction", "Native S")[column],
                    fontsize=12,
                )
        axes[row, 0].text(
            -0.07,
            0.5,
            (
                f"Representative\\nframe {index}"
                if row == 0
                else f"Locked-method worst\\nframe {index}"
            ),
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=11,
            fontweight="bold",
        )
    figure.suptitle(
        f"Current locked Robust VP-SPCA — sealed confirmation seed {robust.seed}",
        fontsize=17,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Both rows share one signal scale and one native-S scale; no thresholding or frame selection in this replay",
        ha="center",
        fontsize=9,
        color="#64748b",
    )
    figure.tight_layout(rect=(0.02, 0.03, 1.0, 0.95), h_pad=0.8, w_pad=0.35)
    figure.savefig(path, dpi=result.config.dpi, bbox_inches="tight")
    plt.close(figure)
    return {
        "layout": "2 frames x 4 decomposition panels",
        "seed": robust.seed,
        "frame_indices": list(indices),
        "worst_frame_source": "sealed continuation-method full-frame error summary",
        "native_S": True,
        "outlier_post_threshold": False,
        "truth_dependent_fit": False,
        "signal_scale": [vmin, vmax],
        "outlier_abs_limit": s_limit,
    }


def _raw_arrays(result: CurrentGitHubGalleryResult) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "clean_time": result.clean_result.dataset.time,
        "clean_truth_spatial_modes": result.clean_result.dataset.spatial_modes,
        "clean_truth_temporal_raw": result.clean_result.dataset.temporal_raw,
        "clean_truth_temporal_centered": result.clean_result.dataset.temporal_centered,
    }
    for method, prefix in (
        ("PCA", "pca"),
        ("VP-SPCA", "vp"),
        ("RND-VP-SPCA", "rnd"),
    ):
        fit = result.clean_result.fits[method]
        integrity = result.clean_result.displays[method]
        legacy = result.clean_legacy_displays[method]
        arrays.update(
            {
                f"{prefix}_A": fit.axes,
                f"{prefix}_B": fit.loadings,
                f"{prefix}_spatial_modes": fit.spatial_modes,
                f"{prefix}_integrity_spatial_modes": integrity.spatial_modes,
                f"{prefix}_integrity_temporal_dynamics": integrity.temporal_dynamics,
                f"{prefix}_legacy_spatial_modes": legacy.spatial_modes,
                f"{prefix}_legacy_temporal_gauge": legacy.temporal_gauge,
                f"{prefix}_legacy_temporal_plot": legacy.temporal_plot,
                f"{prefix}_legacy_truth_ls_scales": legacy.truth_ls_scales,
                f"{prefix}_permutation": np.asarray(legacy.permutation, dtype=np.int64),
                f"{prefix}_signs": np.asarray(legacy.signs, dtype=np.int64),
            }
        )
    robust = result.robust
    arrays.update(
        {
            "robust_A": robust.A,
            "robust_B": robust.B,
            "robust_normalization_sigma_max": np.asarray(
                robust.normalization_sigma_max, dtype=np.float64
            ),
            "robust_time": robust.time,
        }
    )
    if robust.S_normalized is not None:
        arrays["robust_S_normalized"] = robust.S_normalized
    for label, index in (
        ("representative", result.config.representative_frame_index),
    ):
        arrays[f"robust_{label}_clean"] = _robust_frame(
            robust, "clean", index
        )
        arrays[f"robust_{label}_observed"] = _robust_frame(
            robust, "observed", index
        )
        arrays[f"robust_{label}_reconstruction"] = _robust_frame(
            robust, "reconstruction", index
        )
        arrays[f"robust_{label}_outliers"] = _robust_frame(
            robust, "outliers", index
        )
    return {name: np.ascontiguousarray(values) for name, values in arrays.items()}


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_text(arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _source_provenance() -> dict[str, Any]:
    roles = {
        "experiments/video_github_style_current_gallery.py": "gallery runner",
        "experiments/export_locked_robust_gallery_frames.py": "sealed-environment robust frame exporter",
        "experiments/video_figure5_historical_github_current.py": "historical clean generator and display helpers",
        "experiments/video_figure5_paper_style.py": "PCA and VP clean fit helpers",
        "experiments/video_spca_paper_reproduction.py": "robust dataset generator",
        "experiments/video_spca_paper_robust_study.py": "locked robust replay helpers",
        "src/rspca/vp.py": "current VP-SPCA",
        "src/rspca/randomized.py": "current randomized VP-SPCA",
        "src/rspca/robust.py": "current robust VP-SPCA",
        "src/rspca/datasets.py": "array hashing and runtime provenance",
    }
    files: dict[str, Any] = {}
    for relative, role in roles.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"required source is missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "role": role}
    return {
        "git_head": _git_text(["rev-parse", "HEAD"]),
        "current_sources": files,
        "runtime_versions": runtime_versions(),
    }


def _build_report(
    result: CurrentGitHubGalleryResult,
    display: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    clean_hashes = {
        "X": array_sha256(result.clean_result.dataset.data),
        "phi_true": array_sha256(result.clean_result.phi_true_images),
        "a_true": array_sha256(result.clean_result.a_true_raw),
        "time": array_sha256(result.clean_result.dataset.time),
    }
    expected_applies = result.config.grid_size == 200
    clean_verified = bool(
        not expected_applies or clean_hashes == EXPECTED_DEFAULT_ARRAY_SHA256
    )
    robust = result.robust
    sealed_verified = robust.exact_replay_verified
    return {
        "schema_version": 1,
        "experiment": "current video algorithms in historical GitHub visual style",
        "protocol_identity": PROTOCOL_IDENTITY,
        "config": asdict(result.config),
        "protocol": {
            "clean_and_corrupted_branches_are_separate": True,
            "clean_data": "exact commit e01c02f default smooth generator",
            "clean_methods": ["PCA", "VP-SPCA", "RND-VP-SPCA"],
            "clean_primary_display": "historical L2 gauge, Hungarian sign/order, display-only truth LS amplitude fit, individual magma autoscale",
            "clean_native_fit_truth_dependent": False,
            "clean_metrics_truth_ls_refit": False,
            "result_driven_tuning": False,
            "randomized_parameters_source": "historical GitHub comparison, fixed before run",
            "robust_data": "sealed confirmation seed 51001 corrupted classic video",
            "robust_method": "locked B-only descending-kappa continuation selected candidate",
            "robust_display_uses_native_reconstruction": True,
            "robust_display_uses_native_S": True,
            "robust_outlier_post_threshold": False,
            "robust_truth_dependent_fit": False,
        },
        "clean_input_hash_contract": {
            "actual": clean_hashes,
            "expected": EXPECTED_DEFAULT_ARRAY_SHA256,
            "applies": expected_applies,
            "verified": clean_verified,
        },
        "clean_native_metrics": _clean_metrics(result),
        "randomized_native_diagnostics": {
            "status": result.randomized_native.status,
            "sketch_converged": result.randomized_native.sketch_converged,
            "full_converged": result.randomized_native.full_converged,
            "sketch_residual_relative": float(
                result.randomized_native.sketch_residual_relative
            ),
            "full_stationarity": float(
                result.randomized_native.full_stationarity
            ),
            "timings": dict(result.randomized_native.timings),
        },
        "robust_replay": {
            "seed": robust.seed,
            "selected_start_id": result.config.robust_start_id,
            "alpha": result.config.robust_alpha,
            "target_kappa": result.config.robust_target_kappa,
            "kappa_schedule": list(result.config.robust_kappa_schedule),
            "stage_budgets": list(result.config.robust_stage_budgets),
            "candidate_record": dict(robust.candidate_record),
            "sealed_expectation": robust.sealed_expectation,
            "sealed_exact_replay_verified": sealed_verified,
            "actual_full_array_hashes": dict(
                robust.actual_full_array_hashes
            ),
            "replay_provenance": dict(robust.replay_provenance),
        },
        "display": dict(display),
        "raw_npz": {
            "name": RAW_NAME,
            "array_sha256": {
                name: array_sha256(values) for name, values in arrays.items()
            },
        },
        "artifacts": {
            "pca": PCA_FIGURE_NAME,
            "vp_spca": VP_FIGURE_NAME,
            "rnd_vp_spca": RND_FIGURE_NAME,
            "clean_contact_sheet": CLEAN_CONTACT_NAME,
            "clean_summary": CLEAN_SUMMARY_NAME,
            "robust_representative": ROBUST_FIGURE_NAME,
        },
        "provenance": _source_provenance(),
        "scientific_boundaries": {
            "paper_exact_data_claimed": False,
            "clean_figures_are_exact_github_data": True,
            "clean_primary_curves_are_display_amplitude_aligned": True,
            "native_metrics_are_not_display_aligned": True,
            "robust_figures_are_exact_sealed_candidate_replay": sealed_verified,
            "robust_frame_100_is_predeclared_representative": True,
            "robust_worst_frame_presented": False,
            "robust_worst_frame_retained_upstream_for_audit_only": True,
        },
    }


def _readme(report: Mapping[str, Any]) -> str:
    metrics = report["clean_native_metrics"]
    robust = report["robust_replay"]
    return f"""# Current algorithms in the historical GitHub visual style

This package is image-first. The three clean methods use the exact historical
GitHub smooth-video DGP and the familiar 2 x 3 spatial-mode plus temporal-
dynamics layout. The robust method is a separate exact replay of the sealed
confirmation candidate for seed {robust['seed']}.

## Open these figures

- `{VP_FIGURE_NAME}` — current VP-SPCA.
- `{RND_FIGURE_NAME}` — current randomized VP-SPCA.
- `{PCA_FIGURE_NAME}` — current PCA reference.
- `{CLEAN_CONTACT_NAME}` — all three clean methods on one sheet.
- `{CLEAN_SUMMARY_NAME}` — native quality metrics; no display refit.
- `{ROBUST_FIGURE_NAME}` — representative robust Figure-6-style row.

## How to read the clean figures

The top row is the three recovered spatial modes; the bottom row is their time
dynamics. Solid curves are current estimates and dashed orange curves are raw
generator truth. The individual and contact-sheet plots faithfully reproduce
the historical display-only truth LS amplitude fit. That transform never
enters the native fits, reconstructions, or metrics.

| Method | clean error | spatial cosine | temporal cosine | full converged |
|---|---:|---:|---:|---:|
| PCA | {metrics['PCA']['clean_relative_reconstruction_error']:.9g} | {metrics['PCA']['spatial_cosine_mean']:.9f} | {metrics['PCA']['temporal_cosine_mean']:.9f} | {metrics['PCA']['converged']} |
| VP-SPCA | {metrics['VP-SPCA']['clean_relative_reconstruction_error']:.9g} | {metrics['VP-SPCA']['spatial_cosine_mean']:.9f} | {metrics['VP-SPCA']['temporal_cosine_mean']:.9f} | {metrics['VP-SPCA']['converged']} |
| RND-VP-SPCA | {metrics['RND-VP-SPCA']['clean_relative_reconstruction_error']:.9g} | {metrics['RND-VP-SPCA']['spatial_cosine_mean']:.9f} | {metrics['RND-VP-SPCA']['temporal_cosine_mean']:.9f} | {metrics['RND-VP-SPCA']['converged']} |

## Robust boundary

The robust images use the native reconstruction and native S, with no display
threshold and no truth-dependent fit. Alpha, kappa schedule, start, and budgets
are the already-locked formal values. Exact sealed A/B/S and reconstruction
replay passed: `{robust['sealed_exact_replay_verified']}`.

Clean and robust results must not be pooled: they use different DGPs and answer
different questions. The clean data are exact for the historical GitHub demo,
not the paper's unpublished byte-addressable data.
"""


def save_current_github_gallery(
    result: CurrentGitHubGalleryResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "pca": destination / PCA_FIGURE_NAME,
        "vp": destination / VP_FIGURE_NAME,
        "rnd": destination / RND_FIGURE_NAME,
        "contact": destination / CLEAN_CONTACT_NAME,
        "summary": destination / CLEAN_SUMMARY_NAME,
        "robust": destination / ROBUST_FIGURE_NAME,
        "raw": destination / RAW_NAME,
        "report": destination / REPORT_NAME,
        "readme": destination / README_NAME,
        "manifest": destination / MANIFEST_NAME,
    }
    display: dict[str, Any] = {
        "pca": _plot_clean_method(result, "PCA", paths["pca"]),
        "vp_spca": _plot_clean_method(result, "VP-SPCA", paths["vp"]),
        "rnd_vp_spca": _plot_clean_method(result, "RND-VP-SPCA", paths["rnd"]),
        "clean_contact_sheet": _plot_clean_contact_sheet(result, paths["contact"]),
        "clean_summary": _plot_clean_summary(result, paths["summary"]),
        "robust_representative": _plot_robust_row(result, paths["robust"]),
    }
    arrays = _raw_arrays(result)
    np.savez_compressed(paths["raw"], **arrays)
    report = _build_report(result, display, arrays)
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths["readme"].write_text(_readme(report), encoding="utf-8")
    file_records = {
        name: {
            "sha256": _file_sha256(destination / name),
            "size_bytes": (destination / name).stat().st_size,
        }
        for name in sorted(_PAYLOAD_NAMES)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "current_github_style_video_gallery_exact_output_closure",
        "self_excluding": True,
        "files": file_records,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_current_github_gallery_output(destination)
    return paths


def validate_current_github_gallery_output(
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
        raise ValueError(f"output contains non-regular entries: {nonregular}")
    if {path.name for path in entries} != _OUTPUT_NAMES:
        raise ValueError("output does not have the exact required file set")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    if set(manifest.get("files", {})) != _PAYLOAD_NAMES:
        raise ValueError("manifest payload set mismatch")
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("manifest self-hash mismatch")
    for name, record in manifest["files"].items():
        path = directory / name
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"payload hash mismatch: {name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"payload size mismatch: {name}")
    for name in _FIGURE_NAMES:
        if not (directory / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"figure is not a PNG: {name}")
    report = json.loads((directory / REPORT_NAME).read_text(encoding="utf-8"))
    if report.get("protocol_identity") != PROTOCOL_IDENTITY:
        raise ValueError("protocol identity mismatch")
    clean_contract = report.get("clean_input_hash_contract", {})
    if clean_contract.get("applies") is True and clean_contract.get("verified") is not True:
        raise ValueError("historical clean input hashes are not verified")
    if report.get("config", {}).get("formal_robust_replay") is True:
        if report.get("robust_replay", {}).get("sealed_exact_replay_verified") is not True:
            raise ValueError("sealed robust replay is not verified")
    if report.get("protocol", {}).get("clean_metrics_truth_ls_refit") is not False:
        raise ValueError("clean native metrics must not use the display refit")
    if report.get("protocol", {}).get("robust_outlier_post_threshold") is not False:
        raise ValueError("robust display must use native unthresholded S")
    config_record = report["config"]
    replay_dataset = generate_exact_historical_github_dataset(
        HistoricalGitHubFigure5Config(
            grid_size=int(config_record["grid_size"]),
            n_snapshots=int(config_record["n_snapshots"]),
            total_time=float(config_record["total_time"]),
            rank=int(config_record["rank"]),
            alpha=float(config_record["clean_alpha"]),
            max_iter=int(config_record["clean_max_iter"]),
            tol=float(config_record["clean_tol"]),
            dpi=int(config_record["dpi"]),
        )
    )[0].data
    with np.load(directory / RAW_NAME, allow_pickle=False) as arrays:
        expected = report["raw_npz"]["array_sha256"]
        if set(arrays.files) != set(expected):
            raise ValueError("raw NPZ key set mismatch")
        for name, digest in expected.items():
            if array_sha256(arrays[name]) != digest:
                raise ValueError(f"raw array hash mismatch: {name}")
        for prefix in ("pca", "vp", "rnd"):
            native_reconstruction = (
                # X is intentionally regenerated rather than duplicated in NPZ.
                replay_dataset @ arrays[f"{prefix}_B"]
                @ arrays[f"{prefix}_A"].T
            )
            integrity_reconstruction = (
                arrays[f"{prefix}_integrity_spatial_modes"]
                @ arrays[f"{prefix}_integrity_temporal_dynamics"].T
            )
            if not np.allclose(
                native_reconstruction,
                integrity_reconstruction,
                rtol=2e-12,
                atol=2e-12,
            ):
                raise ValueError(f"{prefix} integrity gauge changed reconstruction")
    return manifest


def smoke_config() -> CurrentGitHubGalleryConfig:
    return CurrentGitHubGalleryConfig(
        grid_size=24,
        randomized_oversample=6,
        randomized_power_iters=1,
        robust_seed=71001,
        robust_start_id="pca",
        robust_kappa_schedule=(0.002, 0.001),
        robust_stage_budgets=(1, 2),
        worst_frame_index=171,
        formal_robust_replay=False,
        dpi=70,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--robust-replay-dir",
        type=Path,
        help="exact-verified locked-frame export from the sealed environment",
    )
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    config = smoke_config() if args.smoke else CurrentGitHubGalleryConfig()
    if args.smoke and args.robust_replay_dir is not None:
        raise ValueError("--smoke cannot use a formal robust replay export")
    result = run_current_github_gallery(
        config,
        robust_replay_dir=args.robust_replay_dir,
    )
    paths = save_current_github_gallery(result, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "clean_metrics": _clean_metrics(result),
                "robust_sealed_exact_replay_verified": (
                    result.robust.exact_replay_verified
                ),
                "manifest": str(paths["manifest"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()

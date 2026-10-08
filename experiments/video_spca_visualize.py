"""Reproducible visual audit for the synthetic-video decomposition benchmark.

This module deliberately lives beside, rather than inside, the metric runner.
It consumes the native fitted arrays returned by :mod:`experiments.video_spca`
and writes a new, self-contained visualization directory.  The destination
must not already exist, which prevents a visual audit from silently replacing
formal metric artifacts.

The visual contract contains three static figures, machine-readable per-frame
errors, a content-addressed manifest, and optional GIF/MP4 animations:

* fixed-frame clean/observed/low-rank reconstructions;
* per-frame normalized RMSE for every fitted method;
* injected corruption versus the Robust-VP-SPCA sparse layer and support; and
* an animated version of the same decomposition when a requested writer is
  available.

Ground truth is used only after fitting.  Plot limits are fixed from declared
reference arrays and are recorded in the manifest; no frame or color scale is
selected to make a method look better.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

try:
    from experiments.video_spca import (
        DEFAULT_METHODS,
        FittedMethod,
        MethodReport,
        VideoDataset,
        VideoExperimentConfig,
        VideoExperimentResult,
        _validate_config as _validate_video_config,
        _video_dataset_manifest,
        evaluate_fitted_method,
        generate_video_dataset,
        run_video_experiment,
    )
except ModuleNotFoundError as error:  # support ``python experiments/...py``
    if error.name != "experiments":
        raise
    from video_spca import (  # type: ignore[no-redef]
        DEFAULT_METHODS,
        FittedMethod,
        MethodReport,
        VideoDataset,
        VideoExperimentConfig,
        VideoExperimentResult,
        _validate_config as _validate_video_config,
        _video_dataset_manifest,
        evaluate_fitted_method,
        generate_video_dataset,
        run_video_experiment,
    )
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

VISUALIZATION_SCHEMA_VERSION = 2
METHOD_LABELS = {
    "PCA": "PCA low-rank",
    "VP-SPCA": "VP-SPCA low-rank",
    "RND-VP-SPCA": "Randomized VP-SPCA low-rank",
    "Robust-VP-SPCA": "Robust VP-SPCA low-rank",
}
METHOD_COLORS = {
    "PCA": "#4C78A8",
    "VP-SPCA": "#F58518",
    "RND-VP-SPCA": "#54A24B",
    "Robust-VP-SPCA": "#E45756",
}


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
        raise ValueError("non-finite numbers are forbidden in visual artifacts")
    return value


def _jsonable_detection(
    value: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Serialize the two mathematically nullable zero-positive metrics only."""

    if value is None:
        return None
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if (
            key in {"roc_auc", "average_precision"}
            and isinstance(item, (np.floating, float))
            and np.isnan(float(item))
        ):
            normalized[key] = None
        else:
            normalized[key] = _json_safe(item)
    return normalized


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class _OwnedOutputDirectory:
    """An atomically claimed destination guarded by an inode-bound marker."""

    path: Path
    device: int
    inode: int
    marker_name: str
    marker_payload: bytes
    directory_fd: int

    @classmethod
    def claim(cls, destination: Path) -> "_OwnedOutputDirectory":
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"visualization output directory already exists: {destination}"
            ) from error

        created = destination.lstat()
        if not stat.S_ISDIR(created.st_mode):  # defensive: mkdir must make a dir
            raise RuntimeError("claimed visualization output is not a directory")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        directory_fd = os.open(destination, flags)
        opened = os.fstat(directory_fd)
        if (opened.st_dev, opened.st_ino) != (created.st_dev, created.st_ino):
            os.close(directory_fd)
            raise RuntimeError("visualization destination changed while claiming it")

        token = uuid4().hex
        marker_name = "video_visualization_ownership.json"
        marker_payload = (
            json.dumps(
                {
                    "schema_version": 1,
                    "ownership_token": token,
                    "device": int(created.st_dev),
                    "inode": int(created.st_ino),
                    "failure_policy": "preserve_for_manual_review",
                },
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        marker_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        marker_flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            marker_fd = os.open(
                marker_name,
                marker_flags,
                0o600,
                dir_fd=directory_fd,
            )
            with os.fdopen(marker_fd, "wb") as stream:
                stream.write(marker_payload)
                stream.flush()
                os.fsync(stream.fileno())
        except BaseException:
            os.close(directory_fd)
            raise
        return cls(
            path=destination,
            device=created.st_dev,
            inode=created.st_ino,
            marker_name=marker_name,
            marker_payload=marker_payload,
            directory_fd=directory_fd,
        )

    def _path_matches_inode(self) -> bool:
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            return False
        return stat.S_ISDIR(current.st_mode) and (
            current.st_dev,
            current.st_ino,
        ) == (self.device, self.inode)

    def _marker_matches(self) -> bool:
        marker_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            marker_fd = os.open(
                self.marker_name,
                marker_flags,
                dir_fd=self.directory_fd,
            )
        except OSError:
            return False
        try:
            marker_stat = os.fstat(marker_fd)
            return stat.S_ISREG(marker_stat.st_mode) and (
                os.read(marker_fd, len(self.marker_payload) + 1) == self.marker_payload
            )
        finally:
            os.close(marker_fd)

    def still_owned(self) -> bool:
        return self._path_matches_inode() and self._marker_matches()

    def release(self) -> None:
        if not self.still_owned():
            raise RuntimeError("visualization destination ownership was lost")

    def cleanup_after_failure(self) -> None:
        # There is no portable race-free recursive deletion primitive anchored
        # to an open directory fd.  Preserve partial output and its marker for
        # manual review instead of risking deletion of a replacement path.
        return None

    def close(self) -> None:
        try:
            os.close(self.directory_fd)
        except OSError:
            pass


def _reject_nonfinite_numbers(value: Any, *, location: str) -> None:
    """Reject non-finite numeric values in arbitrarily nested containers."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, location=f"{location}[{index}]")
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"object arrays are forbidden at {location}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"array must be finite at {location}")
    elif isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        raise ValueError(f"number must be finite at {location}")
    elif isinstance(value, (np.complexfloating, complex)) and not np.isfinite(value):
        raise ValueError(f"number must be finite at {location}")


def _require_sha256(value: Any, *, location: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{location} must be a lowercase hexadecimal SHA-256")
    return value


def _array_digest(values: Any, *, location: str) -> str:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{location} must be a NumPy array")
    _reject_nonfinite_numbers(values, location=location)
    return array_sha256(values)


def _values_equivalent(left: Any, right: Any) -> bool:
    """Deep equality that treats two intentionally undefined NaNs as equal."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _values_equivalent(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _values_equivalent(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, np.generic):
        left = left.item()
    if isinstance(right, np.generic):
        right = right.item()
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
    return bool(left == right)


def _validate_detection_mapping(
    value: Mapping[str, Any] | None,
    *,
    location: str,
    allow_undefined_ranking: bool,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"{location} must be a mapping or null")
    for key, item in value.items():
        if isinstance(item, (np.floating, float)) and not np.isfinite(float(item)):
            if (
                allow_undefined_ranking
                and key in {"roc_auc", "average_precision"}
                and np.isnan(float(item))
            ):
                continue
            raise ValueError(f"number must be finite at {location}.{key}")
        _reject_nonfinite_numbers(item, location=f"{location}.{key}")


def _validate_dataset_closure(
    dataset: VideoDataset,
    config: VideoExperimentConfig,
) -> dict[str, Any]:
    if not isinstance(dataset, VideoDataset):
        raise TypeError("result.dataset must be a VideoDataset")
    if dataset.grid_size != config.grid_size:
        raise ValueError("dataset grid size disagrees with the video config")

    expected = generate_video_dataset(config)
    for item in fields(VideoDataset):
        name = item.name
        # The digest is checked independently against the observed array below;
        # do not let a generated string comparison mask that closure check.
        if name == "digest":
            continue
        actual_value = getattr(dataset, name)
        expected_value = getattr(expected, name)
        if isinstance(expected_value, np.ndarray):
            actual_digest = _array_digest(actual_value, location=f"dataset.{name}")
            expected_digest = array_sha256(expected_value)
            if actual_digest != expected_digest:
                raise ValueError(f"dataset/config closure mismatch at dataset.{name}")
        elif actual_value != expected_value:
            raise ValueError(f"dataset/config closure mismatch at dataset.{name}")

    observed_digest = _array_digest(dataset.observed, location="dataset.observed")
    if _require_sha256(dataset.digest, location="dataset.digest") != observed_digest:
        raise ValueError("dataset digest does not match the observed video")
    manifest = _video_dataset_manifest(dataset)
    expected_manifest = _video_dataset_manifest(expected)
    if manifest != expected_manifest:
        raise ValueError("dataset manifest does not close over the configured dataset")
    return manifest


def _validate_fit_closure(
    name: str,
    fitted: FittedMethod,
    dataset: VideoDataset,
    config: VideoExperimentConfig,
) -> None:
    if not isinstance(fitted, FittedMethod):
        raise TypeError(f"fits[{name!r}] must be a FittedMethod")
    if fitted.name != name:
        raise ValueError(f"fits[{name!r}] carries method name {fitted.name!r}")
    expected_component_shape = (config.grid_size**2, config.n_components)
    expected_score_shape = (config.n_timesteps, config.n_components)
    expected_movie_shape = dataset.observed.shape
    array_contracts = {
        "axes": (fitted.axes, expected_component_shape),
        "loadings": (fitted.loadings, expected_component_shape),
        "scores": (fitted.scores, expected_score_shape),
        "reconstruction": (fitted.reconstruction, expected_movie_shape),
    }
    for field_name, (values, expected_shape) in array_contracts.items():
        _array_digest(values, location=f"fits.{name}.{field_name}")
        if values.shape != expected_shape:
            raise ValueError(
                f"fits.{name}.{field_name} has shape {values.shape}, "
                f"expected {expected_shape}"
            )
    if fitted.outliers is not None:
        _array_digest(fitted.outliers, location=f"fits.{name}.outliers")
        if fitted.outliers.shape != expected_movie_shape:
            raise ValueError(f"fits.{name}.outliers has the wrong shape")
    if name == "Robust-VP-SPCA" and fitted.outliers is None:
        raise ValueError("Robust-VP-SPCA must include its native outlier layer")
    if name != "Robust-VP-SPCA" and fitted.outliers is not None:
        raise ValueError(f"{name} unexpectedly includes an outlier layer")

    if not np.allclose(
        fitted.scores,
        dataset.observed @ fitted.loadings,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(f"fits.{name}.scores do not close over data and loadings")
    if not np.allclose(
        fitted.reconstruction,
        fitted.scores @ fitted.axes.T,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError(
            f"fits.{name}.reconstruction does not close over scores and axes"
        )

    if not np.isfinite(fitted.runtime_seconds) or fitted.runtime_seconds < 0.0:
        raise ValueError(f"fits.{name}.runtime_seconds must be finite and non-negative")
    if not isinstance(fitted.converged, (bool, np.bool_)):
        raise TypeError(f"fits.{name}.converged must be boolean")
    if (
        isinstance(fitted.n_iter, bool)
        or not isinstance(fitted.n_iter, (int, np.integer))
        or fitted.n_iter <= 0
    ):
        raise ValueError(f"fits.{name}.n_iter must be a positive integer")
    if fitted.final_stationarity is not None and (
        not np.isfinite(fitted.final_stationarity) or fitted.final_stationarity < 0.0
    ):
        raise ValueError(
            f"fits.{name}.final_stationarity must be finite and non-negative"
        )
    if not isinstance(fitted.status, str) or not fitted.status:
        raise ValueError(f"fits.{name}.status must be a non-empty string")
    if not isinstance(fitted.extra, Mapping):
        raise TypeError(f"fits.{name}.extra must be a mapping")
    _reject_nonfinite_numbers(fitted.extra, location=f"fits.{name}.extra")


def _validate_report_closure(
    name: str,
    report: MethodReport,
    fitted: FittedMethod,
    dataset: VideoDataset,
    config: VideoExperimentConfig,
    dataset_manifest: Mapping[str, Any],
) -> None:
    if not isinstance(report, MethodReport):
        raise TypeError(f"reports[{name!r}] must be a MethodReport")
    if report.method != name:
        raise ValueError(f"reports[{name!r}] carries method name {report.method!r}")
    if report.data_sha256 != dataset.digest:
        raise ValueError(f"reports.{name}.data_sha256 disagrees with the dataset")
    if report.dataset_manifest_sha256 != dataset_manifest["manifest_sha256"]:
        raise ValueError(
            f"reports.{name}.dataset_manifest_sha256 disagrees with the dataset"
        )

    finite_fields = (
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
    )
    for field_name in finite_fields:
        value = getattr(report, field_name)
        if not np.isfinite(value):
            raise ValueError(f"reports.{name}.{field_name} must be finite")
    if report.final_stationarity is not None and not np.isfinite(
        report.final_stationarity
    ):
        raise ValueError(f"reports.{name}.final_stationarity must be finite")
    _reject_nonfinite_numbers(report.extra, location=f"reports.{name}.extra")
    allow_undefined = not bool(np.any(dataset.corruption_support))
    _validate_detection_mapping(
        report.corruption_detection,
        location=f"reports.{name}.corruption_detection",
        allow_undefined_ranking=allow_undefined,
    )
    _validate_detection_mapping(
        report.corruption_detection_at_budget,
        location=f"reports.{name}.corruption_detection_at_budget",
        allow_undefined_ranking=allow_undefined,
    )

    expected_report = evaluate_fitted_method(dataset, fitted, config)
    if not _values_equivalent(asdict(report), asdict(expected_report)):
        raise ValueError(f"reports.{name} does not close over dataset/config/fit")


def validate_video_visualization_result(result: VideoExperimentResult) -> None:
    """Validate the complete dataset/config/fit/report closure before I/O."""

    if not isinstance(result, VideoExperimentResult):
        raise TypeError("result must be a VideoExperimentResult")
    if not isinstance(result.config, VideoExperimentConfig):
        raise TypeError("result.config must be a VideoExperimentConfig")
    _validate_video_config(result.config)
    _reject_nonfinite_numbers(asdict(result.config), location="config")
    dataset_manifest = _validate_dataset_closure(result.dataset, result.config)

    if not isinstance(result.fits, Mapping) or not isinstance(result.reports, Mapping):
        raise TypeError("result fits and reports must be mappings")
    fit_names = tuple(result.fits)
    report_names = tuple(result.reports)
    if not fit_names or fit_names != report_names:
        raise ValueError(
            "fits and reports must have one identical ordered method closure"
        )
    if len(set(fit_names)) != len(fit_names):
        raise ValueError("fits contain duplicate methods")
    unknown = sorted(set(fit_names) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"unknown fitted video methods: {unknown}")

    for name in fit_names:
        fitted = result.fits[name]
        report = result.reports[name]
        _validate_fit_closure(name, fitted, result.dataset, result.config)
        _validate_report_closure(
            name,
            report,
            fitted,
            result.dataset,
            result.config,
            dataset_manifest,
        )


def default_frame_indices(n_timesteps: int, *, count: int = 5) -> tuple[int, ...]:
    """Return deterministic, evenly spaced frame indices including endpoints."""

    if isinstance(n_timesteps, bool) or not isinstance(n_timesteps, int):
        raise TypeError("n_timesteps must be an integer")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if n_timesteps <= 0 or count <= 0:
        raise ValueError("n_timesteps and count must be positive")
    count = min(count, n_timesteps)
    # np.rint has an explicit ties-to-even rule and therefore stays stable.
    indices = np.rint(np.linspace(0, n_timesteps - 1, count)).astype(int)
    return tuple(dict.fromkeys(int(index) for index in indices))


def validate_frame_indices(
    frame_indices: Iterable[int], n_timesteps: int
) -> tuple[int, ...]:
    """Validate fixed frame indices without reordering a declared selection."""

    indices = tuple(frame_indices)
    if not indices:
        raise ValueError("frame_indices must contain at least one frame")
    if any(
        isinstance(index, bool) or not isinstance(index, (int, np.integer))
        for index in indices
    ):
        raise TypeError("every frame index must be an integer")
    resolved = tuple(int(index) for index in indices)
    if len(set(resolved)) != len(resolved):
        raise ValueError("frame_indices must not contain duplicates")
    invalid = [index for index in resolved if index < 0 or index >= n_timesteps]
    if invalid:
        raise ValueError(f"frame indices outside [0, {n_timesteps - 1}]: {invalid}")
    return resolved


def normalized_frame_rmse(result: VideoExperimentResult) -> dict[str, FloatArray]:
    """Return frame RMSE normalized by the movie-wide clean-signal RMS.

    A per-frame clean norm becomes unstable when the latent signal crosses
    zero.  The single movie-wide denominator is fixed before any method is
    scored, remains comparable across time, and is recorded in the manifest.
    """

    clean = np.asarray(result.dataset.clean, dtype=np.float64)
    clean_rms = float(np.sqrt(np.mean(np.square(clean))))
    if not np.isfinite(clean_rms) or clean_rms <= np.finfo(np.float64).eps:
        raise ValueError("clean video must have a positive finite RMS")
    errors: dict[str, FloatArray] = {}
    for name, fitted in result.fits.items():
        reconstruction = np.asarray(fitted.reconstruction, dtype=np.float64)
        if reconstruction.shape != clean.shape:
            raise ValueError(f"{name} reconstruction shape does not match clean video")
        values = np.sqrt(np.mean(np.square(reconstruction - clean), axis=1)) / clean_rms
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} produced non-finite per-frame errors")
        errors[name] = np.asarray(values, dtype=np.float64)
    return errors


def _stable_budget_support(scores: FloatArray, fraction: float) -> BoolArray:
    """Select the declared global top-k budget with stable C-order tie breaks."""

    values = np.abs(np.asarray(scores, dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError("outlier scores must be finite")
    top_k = int(round(float(fraction) * values.size))
    top_k = min(max(top_k, 0), values.size)
    flat_mask = np.zeros(values.size, dtype=bool)
    if top_k:
        order = np.argsort(-values.ravel(order="C"), kind="stable")
        flat_mask[order[:top_k]] = True
    return flat_mask.reshape(values.shape, order="C")


def _positive_quantile_scale(values: FloatArray, quantile: float) -> float:
    scale = float(np.quantile(np.abs(values), quantile))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        scale = float(np.max(np.abs(values), initial=1.0))
    return max(scale, np.finfo(np.float64).eps)


def _scene_rows(result: VideoExperimentResult) -> list[tuple[str, FloatArray]]:
    rows: list[tuple[str, FloatArray]] = [
        ("Clean target", result.dataset.clean),
        ("Observed input", result.dataset.observed),
    ]
    rows.extend(
        (METHOD_LABELS.get(name, name), fitted.reconstruction)
        for name, fitted in result.fits.items()
    )
    return rows


def _save_fixed_frame_figure(
    result: VideoExperimentResult,
    frame_indices: tuple[int, ...],
    path: Path,
    *,
    dpi: int,
    scene_limit: float,
) -> None:
    import matplotlib.pyplot as plt

    rows = _scene_rows(result)
    figure, axes = plt.subplots(
        len(rows),
        len(frame_indices),
        figsize=(2.65 * len(frame_indices) + 1.4, 2.25 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    last_image = None
    for row_index, (label, matrix) in enumerate(rows):
        for column_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, column_index]
            last_image = axis.imshow(
                matrix[frame_index].reshape(result.dataset.grid_size, -1),
                cmap="coolwarm",
                vmin=-scene_limit,
                vmax=scene_limit,
                interpolation="nearest",
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(
                    f"frame {frame_index}\nt={result.dataset.time[frame_index]:.2f}"
                )
            if column_index == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center")
    if last_image is not None:
        figure.colorbar(
            last_image,
            ax=axes.ravel().tolist(),
            fraction=0.015,
            pad=0.01,
            label="signal (fixed clean-reference limits)",
        )
    figure.suptitle(
        "Fixed-frame video decomposition: native low-rank outputs\n"
        f"common display range = ±{scene_limit:.4g}; observed corruptions may clip",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_error_figure(
    result: VideoExperimentResult,
    frame_errors: Mapping[str, FloatArray],
    frame_indices: tuple[int, ...],
    path: Path,
    *,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11.5, 5.2), layout="constrained")
    for name, values in frame_errors.items():
        axis.plot(
            result.dataset.time,
            values,
            label=name,
            color=METHOD_COLORS.get(name),
            linewidth=1.8,
        )
    for position, frame_index in enumerate(frame_indices):
        axis.axvline(
            result.dataset.time[frame_index],
            color="#8A8A8A",
            linewidth=0.8,
            alpha=0.35,
            label="fixed displayed frames" if position == 0 else None,
        )
    axis.set_xlabel("time")
    axis.set_ylabel("frame RMSE / global clean RMS")
    axis.set_title("Per-frame clean-target error (ground truth used only for scoring)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_corruption_figure(
    result: VideoExperimentResult,
    frame_indices: tuple[int, ...],
    path: Path,
    *,
    dpi: int,
    corruption_limit: float,
    estimated_support: BoolArray,
) -> None:
    import matplotlib.pyplot as plt

    robust = result.fits["Robust-VP-SPCA"]
    if robust.outliers is None:  # guarded by the public writer as well
        raise ValueError("Robust-VP-SPCA did not return an outlier layer")
    rows: tuple[tuple[str, NDArray[Any], str], ...] = (
        ("Injected corruption", result.dataset.injected_corruption, "coolwarm"),
        ("Robust sparse layer", robust.outliers, "coolwarm"),
        ("True support", result.dataset.corruption_support, "gray_r"),
        ("Estimated support @ budget", estimated_support, "gray_r"),
    )
    figure, axes = plt.subplots(
        len(rows),
        len(frame_indices),
        figsize=(2.65 * len(frame_indices) + 1.4, 2.25 * len(rows)),
        squeeze=False,
        layout="constrained",
    )
    magnitude_image = None
    for row_index, (label, matrix, cmap) in enumerate(rows):
        for column_index, frame_index in enumerate(frame_indices):
            axis = axes[row_index, column_index]
            if row_index < 2:
                magnitude_image = axis.imshow(
                    matrix[frame_index].reshape(result.dataset.grid_size, -1),
                    cmap=cmap,
                    vmin=-corruption_limit,
                    vmax=corruption_limit,
                    interpolation="nearest",
                )
            else:
                axis.imshow(
                    matrix[frame_index].reshape(result.dataset.grid_size, -1),
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                )
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(f"frame {frame_index}")
            if column_index == 0:
                axis.set_ylabel(label, rotation=0, ha="right", va="center")
    if magnitude_image is not None:
        figure.colorbar(
            magnitude_image,
            ax=axes[:2].ravel().tolist(),
            fraction=0.015,
            pad=0.01,
            label="corruption amplitude (shared limits)",
        )
    budget = int(
        round(result.config.corruption_fraction * result.dataset.observed.size)
    )
    figure.suptitle(
        "Robust sparse-layer audit against injected truth\n"
        f"estimated support uses declared global top-{budget} budget with stable ties",
        fontsize=13,
    )
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_animation(
    result: VideoExperimentResult,
    paths: Mapping[str, Path],
    *,
    stride: int,
    fps: int,
    dpi: int,
    scene_limit: float,
    corruption_limit: float,
) -> tuple[int, ...]:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    requested = tuple(paths)
    unavailable = [
        name for name in requested if not animation.writers.is_available(name)
    ]
    if unavailable:
        raise RuntimeError(
            "requested Matplotlib animation writer(s) unavailable: "
            + ", ".join(unavailable)
        )
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("animation stride must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("animation fps must be a positive integer")

    frames = list(range(0, result.config.n_timesteps, stride))
    if frames[-1] != result.config.n_timesteps - 1:
        frames.append(result.config.n_timesteps - 1)
    rows = _scene_rows(result)
    robust = result.fits["Robust-VP-SPCA"]
    assert robust.outliers is not None
    panels: list[tuple[str, FloatArray, float]] = [
        (label, matrix, scene_limit) for label, matrix in rows
    ]
    panels.extend(
        [
            (
                "Injected corruption",
                result.dataset.injected_corruption,
                corruption_limit,
            ),
            ("Robust sparse layer", robust.outliers, corruption_limit),
        ]
    )
    n_columns = 4
    n_rows = int(np.ceil(len(panels) / n_columns))
    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(10.4, 2.75 * n_rows),
        squeeze=False,
        layout="constrained",
    )
    images = []
    for axis, (label, matrix, limit) in zip(axes.ravel(), panels):
        image = axis.imshow(
            matrix[frames[0]].reshape(result.dataset.grid_size, -1),
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            animated=False,
        )
        axis.set_title(label, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
        images.append(image)
    for axis in axes.ravel()[len(panels) :]:
        axis.set_visible(False)
    title = figure.suptitle("")

    def update(frame_index: int) -> list[Any]:
        for image, (_, matrix, _) in zip(images, panels):
            image.set_data(matrix[frame_index].reshape(result.dataset.grid_size, -1))
        title.set_text(
            f"frame {frame_index}/{result.config.n_timesteps - 1} · "
            f"t={result.dataset.time[frame_index]:.2f}"
        )
        return [*images, title]

    movie = animation.FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )
    if "pillow" in paths:
        movie.save(paths["pillow"], writer=animation.PillowWriter(fps=fps), dpi=dpi)
    if "ffmpeg" in paths:
        movie.save(
            paths["ffmpeg"],
            writer=animation.FFMpegWriter(
                fps=fps,
                bitrate=1800,
                metadata={"artist": "vp-rspca", "comment": "native decomposition"},
            ),
            dpi=dpi,
        )
    plt.close(figure)
    return tuple(frames)


def _write_frame_error_csv(
    result: VideoExperimentResult,
    frame_errors: Mapping[str, FloatArray],
    path: Path,
) -> None:
    fieldnames = ["frame_index", "time", *frame_errors]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for frame_index in range(result.config.n_timesteps):
            writer.writerow(
                {
                    "frame_index": frame_index,
                    "time": float(result.dataset.time[frame_index]),
                    **{
                        name: float(values[frame_index])
                        for name, values in frame_errors.items()
                    },
                }
            )


def _frame_error_summaries(
    result: VideoExperimentResult,
    frame_errors: Mapping[str, FloatArray],
) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for name, values in frame_errors.items():
        worst = int(np.argmax(values))
        summaries[name] = {
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "maximum": float(values[worst]),
            "worst_frame_index": worst,
            "worst_frame_time": float(result.dataset.time[worst]),
            "global_clean_reconstruction_error": float(
                result.reports[name].clean_reconstruction_error
            ),
        }
    return summaries


def write_video_visualizations(
    result: VideoExperimentResult,
    output_dir: str | Path,
    *,
    frame_indices: Iterable[int] | None = None,
    animation_formats: Iterable[str] = (),
    animation_stride: int = 3,
    fps: int = 6,
    dpi: int = 150,
) -> dict[str, Path]:
    """Write a complete visual audit to a new directory and return its paths.

    After all inputs validate, ``output_dir`` is claimed with one atomic mkdir.
    Files are written directly into that inode.  A failure deliberately retains
    the partial directory and its unguessable ownership receipt for manual review;
    it never recursively deletes a path that a concurrent process could replace.
    """

    destination = Path(output_dir)
    validate_video_visualization_result(result)
    if not result.fits:
        raise ValueError("the video result contains no fitted methods")
    robust = result.fits.get("Robust-VP-SPCA")
    if robust is None or robust.outliers is None:
        raise ValueError(
            "Robust-VP-SPCA with a native outlier layer is required for the "
            "corruption audit"
        )
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if (
        isinstance(animation_stride, bool)
        or not isinstance(animation_stride, int)
        or animation_stride <= 0
    ):
        raise ValueError("animation_stride must be a positive integer")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps <= 0:
        raise ValueError("fps must be a positive integer")
    frames = validate_frame_indices(
        (
            default_frame_indices(result.config.n_timesteps)
            if frame_indices is None
            else frame_indices
        ),
        result.config.n_timesteps,
    )
    formats = tuple(str(value).strip().lower() for value in animation_formats)
    if len(set(formats)) != len(formats):
        raise ValueError("animation_formats must not contain duplicates")
    aliases = {"gif": "pillow", "mp4": "ffmpeg", "pillow": "pillow", "ffmpeg": "ffmpeg"}
    unknown = sorted(set(formats) - set(aliases))
    if unknown:
        raise ValueError(f"unknown animation format(s): {unknown}")
    writers = tuple(aliases[value] for value in formats)
    if len(set(writers)) != len(writers):
        raise ValueError("animation formats resolve to duplicate writers")

    frame_errors = normalized_frame_rmse(result)
    selected_clean = result.dataset.clean[np.asarray(frames)]
    scene_limit = _positive_quantile_scale(selected_clean, 0.995)
    combined_corruption = np.concatenate(
        [
            result.dataset.injected_corruption[np.asarray(frames)].ravel(),
            robust.outliers[np.asarray(frames)].ravel(),
        ]
    )
    corruption_limit = _positive_quantile_scale(combined_corruption, 1.0)
    estimated_support = _stable_budget_support(
        robust.outliers, result.config.corruption_fraction
    )

    ownership = _OwnedOutputDirectory.claim(destination)
    try:
        paths = {
            "ownership_receipt": destination / ownership.marker_name,
            "fixed_frames": destination / "video_decomposition_fixed_frames.png",
            "per_frame_error_plot": destination / "video_per_frame_error.png",
            "corruption_layers": destination / "video_corruption_layers.png",
            "per_frame_error_csv": destination / "video_per_frame_error.csv",
        }
        _save_fixed_frame_figure(
            result,
            frames,
            paths["fixed_frames"],
            dpi=dpi,
            scene_limit=scene_limit,
        )
        _save_error_figure(
            result,
            frame_errors,
            frames,
            paths["per_frame_error_plot"],
            dpi=dpi,
        )
        _save_corruption_figure(
            result,
            frames,
            paths["corruption_layers"],
            dpi=dpi,
            corruption_limit=corruption_limit,
            estimated_support=estimated_support,
        )
        _write_frame_error_csv(result, frame_errors, paths["per_frame_error_csv"])

        animation_paths: dict[str, Path] = {}
        if "pillow" in writers:
            animation_paths["pillow"] = destination / "video_decomposition.gif"
            paths["animation_gif"] = animation_paths["pillow"]
        if "ffmpeg" in writers:
            animation_paths["ffmpeg"] = destination / "video_decomposition.mp4"
            paths["animation_mp4"] = animation_paths["ffmpeg"]
        animation_frames: tuple[int, ...] = ()
        if animation_paths:
            animation_frames = _save_animation(
                result,
                animation_paths,
                stride=animation_stride,
                fps=fps,
                dpi=min(dpi, 120),
                scene_limit=scene_limit,
                corruption_limit=corruption_limit,
            )

        report_manifest_hashes = {
            report.dataset_manifest_sha256 for report in result.reports.values()
        }
        if len(report_manifest_hashes) != 1:
            raise ValueError("fitted reports do not share one dataset manifest hash")
        artifact_rows = {
            name: {
                "path": path.name,
                "sha256": _file_sha256(path),
                "size_bytes": path.stat().st_size,
                "availability": "local_output",
                "local_copy_included": True,
            }
            for name, path in paths.items()
        }
        nonconverged_methods = [
            name for name, report in result.reports.items() if not report.converged
        ]
        method_diagnostics = {
            name: {
                "converged": bool(report.converged),
                "status": report.status,
                "n_iter": int(report.n_iter),
                "final_stationarity": (
                    None
                    if report.final_stationarity is None
                    else float(report.final_stationarity)
                ),
                "corruption_detection": _jsonable_detection(
                    report.corruption_detection
                ),
                "corruption_detection_at_budget": _jsonable_detection(
                    report.corruption_detection_at_budget
                ),
            }
            for name, report in result.reports.items()
        }
        clean_rms = float(np.sqrt(np.mean(np.square(result.dataset.clean))))
        manifest_body: dict[str, Any] = {
            "schema_version": VISUALIZATION_SCHEMA_VERSION,
            "config": asdict(result.config),
            "runtime_versions": runtime_versions(),
            "data_sha256": result.dataset.digest,
            "dataset_manifest_sha256": next(iter(report_manifest_hashes)),
            "orientation": "time_samples_x_pixel_features",
            "methods": list(result.fits),
            "method_diagnostics": method_diagnostics,
            "visual_inference_qualified": False,
            "visual_inference_qualification": {
                "reason": "single_seed_visual_diagnostic_only",
                "all_methods_converged": not nonconverged_methods,
                "nonconverged_methods": nonconverged_methods,
                "fixed_budget_descriptive_only": bool(nonconverged_methods),
                "multi_seed_inference_required": True,
            },
            "frame_indices": list(frames),
            "frame_times": [float(result.dataset.time[index]) for index in frames],
            "frame_error_definition": {
                "metric": "frame_rmse_divided_by_global_clean_rms",
                "global_clean_rms": clean_rms,
                "ground_truth_used_only_after_fitting": True,
            },
            "frame_error_summaries": _frame_error_summaries(result, frame_errors),
            "display_contract": {
                "scene_limit": scene_limit,
                "scene_limit_rule": (
                    "99.5th percentile of abs(clean) over predeclared fixed frames"
                ),
                "scene_values_outside_limits_are_clipped_for_display": True,
                "corruption_limit": corruption_limit,
                "corruption_limit_rule": (
                    "maximum absolute injected/estimated corruption over "
                    "predeclared fixed frames"
                ),
                "corruption_truth_and_estimate_share_limits": True,
            },
            "support_contract": {
                "source": "abs(Robust-VP-SPCA native outlier layer)",
                "budget_fraction": float(result.config.corruption_fraction),
                "budget_top_k": int(
                    round(
                        result.config.corruption_fraction * result.dataset.observed.size
                    )
                ),
                "selection": "global stable descending score; C-order tie break",
                "selected_cells": int(np.count_nonzero(estimated_support)),
                "ground_truth_used_for_selection": False,
            },
            "animation": {
                "requested_formats": list(formats),
                "writers": list(writers),
                "stride": animation_stride,
                "fps": fps,
                "frame_indices": list(animation_frames),
            },
            "artifact_copy_scope": {
                "complete_local_payload": True,
                "external_server_only_artifacts": [],
            },
            "artifacts": artifact_rows,
        }
        manifest_body["manifest_sha256"] = canonical_sha256(manifest_body)
        manifest_path = destination / "video_visualization_manifest.json"
        manifest_path.write_text(
            json.dumps(
                _json_safe(manifest_body),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        paths["manifest"] = manifest_path
        validate_visualization_manifest_local_copy(manifest_body, destination)
        ownership.release()
        return paths
    except BaseException:
        ownership.cleanup_after_failure()
        raise
    finally:
        ownership.close()


def _safe_artifact_path(root: Path, value: Any, *, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{location} must stay inside the artifact directory")
    return root / relative


def _validate_manifest_digest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("visualization manifest must be a mapping")
    body = _json_safe(manifest)
    claimed = _require_sha256(
        body.pop("manifest_sha256", None),
        location="manifest.manifest_sha256",
    )
    if canonical_sha256(body) != claimed:
        raise ValueError("visualization manifest SHA-256 mismatch")
    body["manifest_sha256"] = claimed
    return body


def _validate_manifest_scientific_contract(manifest: Mapping[str, Any]) -> None:
    _require_sha256(manifest.get("data_sha256"), location="manifest.data_sha256")
    _require_sha256(
        manifest.get("dataset_manifest_sha256"),
        location="manifest.dataset_manifest_sha256",
    )
    methods = manifest.get("methods")
    diagnostics = manifest.get("method_diagnostics")
    if (
        not isinstance(methods, list)
        or not methods
        or any(not isinstance(name, str) or not name for name in methods)
        or len(set(methods)) != len(methods)
    ):
        raise ValueError("manifest.methods must be a non-empty unique string list")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != set(methods):
        raise ValueError("manifest method diagnostics must match the method closure")
    if "Robust-VP-SPCA" not in methods:
        raise ValueError("manifest must include the robust sparse-layer method")

    diagnostic_fields = {
        "converged",
        "status",
        "n_iter",
        "final_stationarity",
        "corruption_detection",
        "corruption_detection_at_budget",
    }
    detection_fields = {
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "roc_auc",
        "average_precision",
    }
    count_fields = {
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "budget_top_k",
    }
    unit_interval_fields = {
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "budget_fraction",
    }
    nonconverged: list[str] = []
    for name in methods:
        record = diagnostics[name]
        location = f"manifest.method_diagnostics.{name}"
        if not isinstance(record, Mapping) or set(record) != diagnostic_fields:
            raise ValueError(f"{location} does not have the complete schema")
        if not isinstance(record["converged"], bool):
            raise TypeError(f"{location}.converged must be boolean")
        if not record["converged"]:
            nonconverged.append(name)
        if not isinstance(record["status"], str) or not record["status"]:
            raise ValueError(f"{location}.status must be a non-empty string")
        n_iter = record["n_iter"]
        if isinstance(n_iter, bool) or not isinstance(n_iter, int) or n_iter <= 0:
            raise ValueError(f"{location}.n_iter must be a positive integer")
        stationarity = record["final_stationarity"]
        if stationarity is not None and (
            not isinstance(stationarity, (int, float))
            or isinstance(stationarity, bool)
            or not np.isfinite(stationarity)
            or stationarity < 0.0
        ):
            raise ValueError(
                f"{location}.final_stationarity must be null or finite/non-negative"
            )

        detection = record["corruption_detection"]
        at_budget = record["corruption_detection_at_budget"]
        if name == "Robust-VP-SPCA":
            if not isinstance(detection, Mapping) or set(detection) != detection_fields:
                raise ValueError(f"{location}.corruption_detection is incomplete")
            if not isinstance(at_budget, Mapping) or set(at_budget) != {
                *detection_fields,
                "budget_fraction",
                "budget_top_k",
            }:
                raise ValueError(
                    f"{location}.corruption_detection_at_budget is incomplete"
                )
        elif detection is not None or at_budget is not None:
            raise ValueError(f"{location} unexpectedly declares corruption metrics")
        for field_name, value in (
            ("corruption_detection", detection),
            ("corruption_detection_at_budget", at_budget),
        ):
            if value is not None:
                for key, item in value.items():
                    if item is None and key in {"roc_auc", "average_precision"}:
                        continue
                    item_location = f"{location}.{field_name}.{key}"
                    if key in count_fields:
                        if (
                            isinstance(item, bool)
                            or not isinstance(item, int)
                            or item < 0
                        ):
                            raise ValueError(
                                f"{item_location} must be a non-negative integer"
                            )
                        continue
                    if (
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not np.isfinite(item)
                    ):
                        raise ValueError(f"{item_location} must be finite numeric")
                    if key in unit_interval_fields and not 0.0 <= item <= 1.0:
                        raise ValueError(f"{item_location} must lie in [0, 1]")

    robust_budget = diagnostics["Robust-VP-SPCA"]["corruption_detection_at_budget"]
    support_contract = manifest.get("support_contract")
    if not isinstance(support_contract, Mapping):
        raise ValueError("manifest.support_contract must be a mapping")
    if (
        support_contract.get("ground_truth_used_for_selection") is not False
        or support_contract.get("budget_top_k") != robust_budget["budget_top_k"]
        or support_contract.get("selected_cells") != robust_budget["budget_top_k"]
        or support_contract.get("budget_fraction") != robust_budget["budget_fraction"]
    ):
        raise ValueError("manifest support and corruption-budget contracts disagree")

    animation = manifest.get("animation")
    if not isinstance(animation, Mapping):
        raise ValueError("manifest.animation must be a mapping")
    for field_name in ("stride", "fps"):
        value = animation.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"manifest.animation.{field_name} must be a positive integer"
            )
    requested_formats = animation.get("requested_formats")
    writers = animation.get("writers")
    animation_frames = animation.get("frame_indices")
    if (
        not isinstance(requested_formats, list)
        or len(set(requested_formats)) != len(requested_formats)
        or any(
            value not in {"gif", "mp4", "pillow", "ffmpeg"}
            for value in requested_formats
        )
    ):
        raise ValueError("manifest animation formats are invalid")
    writer_aliases = {
        "gif": "pillow",
        "mp4": "ffmpeg",
        "pillow": "pillow",
        "ffmpeg": "ffmpeg",
    }
    expected_writers = [writer_aliases[value] for value in requested_formats]
    if writers != expected_writers:
        raise ValueError("manifest animation writers disagree with requested formats")
    if (
        not isinstance(animation_frames, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in animation_frames
        )
        or (not requested_formats and animation_frames)
        or (requested_formats and not animation_frames)
    ):
        raise ValueError("manifest animation frame indices are invalid")
    if requested_formats:
        config = manifest.get("config")
        n_timesteps = config.get("n_timesteps") if isinstance(config, Mapping) else None
        if (
            isinstance(n_timesteps, bool)
            or not isinstance(n_timesteps, int)
            or n_timesteps <= 0
        ):
            raise ValueError("manifest config has an invalid timestep count")
        expected_frames = list(range(0, n_timesteps, animation["stride"]))
        if expected_frames[-1] != n_timesteps - 1:
            expected_frames.append(n_timesteps - 1)
        if animation_frames != expected_frames:
            raise ValueError("manifest animation frames disagree with stride/config")

    if manifest.get("visual_inference_qualified") is not False:
        raise ValueError("a single-seed visual manifest cannot qualify inference")
    qualification = manifest.get("visual_inference_qualification")
    expected_qualification = {
        "reason": "single_seed_visual_diagnostic_only",
        "all_methods_converged": not nonconverged,
        "nonconverged_methods": nonconverged,
        "fixed_budget_descriptive_only": bool(nonconverged),
        "multi_seed_inference_required": True,
    }
    if qualification != expected_qualification:
        raise ValueError("manifest visual-inference qualification is inconsistent")


def validate_visualization_manifest_local_copy(
    manifest: Mapping[str, Any],
    local_dir: str | Path,
    *,
    manifest_filename: str = "video_visualization_manifest.json",
) -> None:
    """Validate a complete or explicitly projected local artifact closure."""

    checked = _validate_manifest_digest(manifest)
    if checked.get("schema_version") != VISUALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported visualization manifest schema version")
    _validate_manifest_scientific_contract(checked)
    artifacts = checked.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("visualization manifest must declare artifacts")
    root = Path(local_dir)
    expected_local_paths: set[str] = set()
    declared_paths: set[str] = set()
    external_names: list[str] = []
    for name, raw_record in artifacts.items():
        location = f"manifest.artifacts.{name}"
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"{location} must be a mapping")
        path = _safe_artifact_path(root, raw_record.get("path"), location=location)
        relative = path.relative_to(root).as_posix()
        if relative == manifest_filename or relative in declared_paths:
            raise ValueError(f"{location} duplicates a reserved/declared path")
        declared_paths.add(relative)
        digest = _require_sha256(
            raw_record.get("sha256"), location=f"{location}.sha256"
        )
        size = raw_record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{location}.size_bytes must be a non-negative integer")
        included = raw_record.get("local_copy_included")
        availability = raw_record.get("availability")
        if not isinstance(included, bool):
            raise TypeError(f"{location}.local_copy_included must be boolean")
        if included:
            if availability not in {
                "local_output",
                "local_copy_included",
                "local_and_server",
            }:
                raise ValueError(
                    f"{location} overstates or misstates local availability"
                )
            if not path.is_file():
                raise ValueError(f"declared local artifact is missing: {relative}")
            if path.stat().st_size != size or _file_sha256(path) != digest:
                raise ValueError(
                    f"declared local artifact hash/size mismatch: {relative}"
                )
            expected_local_paths.add(relative)
        else:
            if availability != "external_server_only":
                raise ValueError(f"{location} must identify its external-only status")
            if path.exists():
                raise ValueError(
                    f"external-only artifact unexpectedly exists locally: {relative}"
                )
            external_names.append(str(name))

    actual_local_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != manifest_filename
    }
    if actual_local_paths != expected_local_paths:
        missing = sorted(expected_local_paths - actual_local_paths)
        extra = sorted(actual_local_paths - expected_local_paths)
        raise ValueError(
            f"local visualization artifact closure mismatch; missing={missing}, extra={extra}"
        )
    scope = checked.get("artifact_copy_scope")
    if not isinstance(scope, Mapping):
        raise ValueError("manifest.artifact_copy_scope must be a mapping")
    if scope.get("complete_local_payload") != (not external_names):
        raise ValueError("artifact copy scope misstates local completeness")
    if scope.get("external_server_only_artifacts") != external_names:
        raise ValueError("artifact copy scope has the wrong external artifact list")


def project_visualization_manifest_for_local_copy(
    manifest: Mapping[str, Any],
    local_dir: str | Path,
    *,
    manifest_filename: str = "video_visualization_manifest.json",
) -> dict[str, Any]:
    """Project a complete output manifest onto an intentionally partial copy."""

    checked = _validate_manifest_digest(manifest)
    if checked.get("schema_version") != VISUALIZATION_SCHEMA_VERSION:
        raise ValueError("unsupported visualization manifest schema version")
    _validate_manifest_scientific_contract(checked)
    body = dict(checked)
    body.pop("manifest_sha256")
    root = Path(local_dir)
    artifacts = body.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("visualization manifest must declare artifacts")
    projected_artifacts: dict[str, dict[str, Any]] = {}
    external_names: list[str] = []
    for name, raw_record in artifacts.items():
        if not isinstance(raw_record, Mapping):
            raise TypeError(f"manifest.artifacts.{name} must be a mapping")
        record = dict(raw_record)
        path = _safe_artifact_path(
            root,
            record.get("path"),
            location=f"manifest.artifacts.{name}.path",
        )
        if path.is_file():
            if path.stat().st_size != record.get("size_bytes") or _file_sha256(
                path
            ) != record.get("sha256"):
                raise ValueError(f"local artifact differs from manifest: {path.name}")
            record["availability"] = "local_and_server"
            record["local_copy_included"] = True
        else:
            record["availability"] = "external_server_only"
            record["local_copy_included"] = False
            external_names.append(str(name))
        projected_artifacts[str(name)] = record
    body["artifacts"] = projected_artifacts
    body["artifact_copy_scope"] = {
        "complete_local_payload": not external_names,
        "external_server_only_artifacts": external_names,
    }
    body["manifest_sha256"] = canonical_sha256(body)
    validate_visualization_manifest_local_copy(
        body,
        root,
        manifest_filename=manifest_filename,
    )
    return body


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
    if "Robust-VP-SPCA" not in selected:
        raise argparse.ArgumentTypeError(
            "robust must be included because the visual audit compares sparse layers"
        )
    if len(set(selected)) != len(selected):
        raise argparse.ArgumentTypeError("methods must not contain duplicates")
    return tuple(selected)


def _parse_frame_indices(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(token.strip()) for token in value.split(",") if token.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "frame indices must be comma-separated integers"
        ) from error


def _parse_animation_formats(value: str) -> tuple[str, ...]:
    normalized = value.strip().lower()
    if normalized in {"", "none"}:
        return ()
    formats = tuple(token.strip() for token in normalized.split(",") if token.strip())
    unknown = sorted(set(formats) - {"gif", "mp4"})
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown animation formats: {unknown}")
    return formats


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
        help="Comma-separated: pca,vp,rnd,robust (robust is required)",
    )
    parser.add_argument(
        "--frame-indices",
        type=_parse_frame_indices,
        default=None,
        help="Comma-separated fixed frames; default is five evenly spaced frames",
    )
    parser.add_argument(
        "--animation-formats",
        type=_parse_animation_formats,
        default=("gif",),
        help="Comma-separated gif,mp4 or none (default: gif)",
    )
    parser.add_argument("--animation-stride", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Must name a new directory; existing directories are never overwritten",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = parse_args(argv)
    # Avoid spending solver time when the requested destination is already
    # occupied.  ``lexists`` is intentional: a dangling symlink is occupied,
    # too.  The later atomic mkdir in ``write_video_visualizations`` remains
    # authoritative for races after this advisory fail-fast check.
    if os.path.lexists(args.output_dir):
        raise FileExistsError(
            f"visualization output directory already exists: {args.output_dir}"
        )
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
    result = run_video_experiment(config, methods=args.methods)
    paths = write_video_visualizations(
        result,
        args.output_dir,
        frame_indices=args.frame_indices,
        animation_formats=args.animation_formats,
        animation_stride=args.animation_stride,
        fps=args.fps,
        dpi=args.dpi,
    )
    print(
        json.dumps(
            {
                "data_sha256": result.dataset.digest,
                "frame_error_summaries": _frame_error_summaries(
                    result, normalized_frame_rmse(result)
                ),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()


__all__ = [
    "VISUALIZATION_SCHEMA_VERSION",
    "default_frame_indices",
    "main",
    "normalized_frame_rmse",
    "parse_args",
    "project_visualization_manifest_for_local_copy",
    "validate_frame_indices",
    "validate_video_visualization_result",
    "validate_visualization_manifest_local_copy",
    "write_video_visualizations",
]

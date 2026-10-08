"""Build readable, audit-preserving views of a frozen video confirmation run.

The solver is never called.  Scene and sparse panels are copied pixel-for-pixel
from the closed confirmation PNG, while the time-series figures are redrawn
only from the closed full-frame-error CSV.  The source package is hash-checked
before a new, exclusive-create presentation directory is claimed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FORMAL_MANIFEST_NAME = "confirmation_output_manifest.json"
FORMAL_REPORT_NAME = "confirmation_report.json"
FORMAL_FRAMES_NAME = "confirmation_full_frame_errors.csv"
FORMAL_FIGURE_NAME = "confirmation_representative_and_worst_frames.png"

MAIN_FIGURE_NAME = "reconstruction_selected_frames.png"
SPARSE_FIGURE_NAME = "outlier_and_error_selected_frames.png"
CURVES_FIGURE_NAME = "full_frame_error_curves.png"
SUMMARY_FIGURE_NAME = "full_frame_error_summary.png"
RECEIPT_NAME = "presentation_receipt.json"
README_NAME = "README.md"
OUTPUT_MANIFEST_NAME = "presentation_output_manifest.json"

_SOURCE_PANEL_COUNT = 15
_SOURCE_ROW_COUNT = 4
_ROW_IDS = (
    "representative_locked_frame",
    "origin_main_legacy_cohort_worst",
    "single_start_cohort_worst",
)
_ROW_LABELS = {
    "representative_locked_frame": "Representative (locked)",
    "origin_main_legacy_cohort_worst": "Legacy cohort worst",
    "single_start_cohort_worst": "Fixed cohort worst",
}
_METHODS = ("pca", "legacy_dynamic_kappa", "single_start")
_METHOD_LABELS = {
    "pca": "PCA",
    "legacy_dynamic_kappa": "Exact legacy dynamic kappa",
    "single_start": "Optimized fixed single start",
}
_METHOD_COLORS = {
    "pca": "#1f77b4",
    "legacy_dynamic_kappa": "#d62728",
    "single_start": "#2ca02c",
}
_MAIN_COLUMNS = (
    (0, "Clean truth"),
    (1, "Observed"),
    (2, "PCA"),
    (3, "Exact legacy dynamic kappa"),
    (4, "Optimized fixed single start"),
)
_SPARSE_COLUMNS = (
    (7, "True S"),
    (8, "Legacy S"),
    (9, "Fixed S"),
    (12, "|Legacy - clean|"),
    (13, "|Fixed - clean|"),
)
_CSV_FIELDS = (
    "seed",
    "method",
    "frame_index",
    "time",
    "normalized_rmse",
)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant in {path.name}: {value}")

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def validate_frozen_confirmation(
    confirmation_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the closed file contract and hashes needed for presentation."""

    root = Path(confirmation_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("confirmation path must be a real directory")
    entries = list(root.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"confirmation contains non-regular entry: {entry.name}")

    manifest = _strict_json(root / FORMAL_MANIFEST_NAME)
    payloads = set(manifest.get("required_payloads", []))
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != payloads:
        raise ValueError("confirmation manifest payload/file set mismatch")
    if {entry.name for entry in entries} != payloads | {FORMAL_MANIFEST_NAME}:
        raise ValueError("confirmation directory does not match its closed contract")
    body = dict(manifest)
    declared_manifest_digest = body.pop("manifest_sha256", None)
    if declared_manifest_digest != _canonical_sha256(body):
        raise ValueError("confirmation manifest canonical digest mismatch")
    for name, record in files.items():
        if not isinstance(record, dict):
            raise ValueError(f"invalid manifest record for {name}")
        path = root / name
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"SHA-256 mismatch for {name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"size mismatch for {name}")
    required = {FORMAL_REPORT_NAME, FORMAL_FRAMES_NAME, FORMAL_FIGURE_NAME}
    if not required.issubset(payloads):
        raise ValueError("confirmation package lacks a presentation input")

    report = _strict_json(root / FORMAL_REPORT_NAME)
    report_body = dict(report)
    report_digest = report_body.pop("report_body_sha256", None)
    if report_digest != _canonical_sha256(report_body):
        raise ValueError("confirmation report canonical digest mismatch")
    return manifest, report


def _dark_runs(values: np.ndarray, minimum_length: int) -> list[tuple[int, int]]:
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for raw in indices[1:]:
        current = int(raw)
        if current != previous + 1:
            if previous - start + 1 >= minimum_length:
                runs.append((start, previous))
            start = current
        previous = current
    if previous - start + 1 >= minimum_length:
        runs.append((start, previous))
    return runs


def detect_formal_panel_boxes(image: Image.Image) -> list[list[tuple[int, int, int, int]]]:
    """Recover the 4x15 axes rectangles from the frozen Matplotlib raster."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = rgb.shape
    dark = np.max(rgb, axis=2) < 160

    # A row border spans almost the complete axes grid.  Text and scene pixels
    # cannot reach this many dark pixels on one scanline.
    row_threshold = max(1, int(width * 0.65))
    horizontal_runs = _dark_runs(dark.sum(axis=1) >= row_threshold, 1)
    if len(horizontal_runs) != 2 * _SOURCE_ROW_COUNT:
        raise ValueError(
            "could not uniquely detect the four formal figure row borders"
        )
    row_bounds: list[tuple[int, int]] = []
    for index in range(0, len(horizontal_runs), 2):
        top_run = horizontal_runs[index]
        bottom_run = horizontal_runs[index + 1]
        top = top_run[0]
        bottom = bottom_run[1]
        if bottom <= top:
            raise ValueError("invalid formal figure row bounds")
        row_bounds.append((top, bottom))

    # On an axes top border each panel is a long dark run separated by the
    # white subplot gap.  This remains reliable for black error-panel interiors.
    first_top = row_bounds[0][0]
    minimum_panel_width = max(20, width // (_SOURCE_PANEL_COUNT * 3))
    column_runs = _dark_runs(dark[first_top], minimum_panel_width)
    if len(column_runs) != _SOURCE_PANEL_COUNT:
        raise ValueError("could not uniquely detect the fifteen formal panels")

    widths = [right - left + 1 for left, right in column_runs]
    heights = [bottom - top + 1 for top, bottom in row_bounds]
    if max(widths) - min(widths) > 2 or max(heights) - min(heights) > 2:
        raise ValueError("formal panel geometry is not a regular grid")
    if min(widths + heights) < 50:
        raise ValueError("formal panels are implausibly small")

    # Matplotlib's integer rasterization can make a few axes one pixel wider.
    # Normalize to the common minimum by dropping only the redundant outermost
    # border pixel; never resample the panel interior.
    panel_width = min(widths)
    panel_height = min(heights)
    return [
        [
            (left, top, left + panel_width, top + panel_height)
            for left, _ in column_runs
        ]
        for top, _ in row_bounds
    ]


def _load_frame_errors(
    path: Path,
) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    grouped: dict[tuple[int, str], list[tuple[int, float, float]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
            raise ValueError("full-frame CSV schema mismatch")
        for row in reader:
            seed = int(row["seed"])
            method = row["method"]
            frame_index = int(row["frame_index"])
            time = float(row["time"])
            error = float(row["normalized_rmse"])
            if seed < 0 or frame_index < 0 or not math.isfinite(time) or not math.isfinite(error):
                raise ValueError("full-frame CSV contains invalid values")
            if error < 0.0:
                raise ValueError("normalized RMSE cannot be negative")
            grouped[(seed, method)].append((frame_index, time, error))

    if not grouped:
        raise ValueError("full-frame CSV is empty")
    parsed: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    frame_counts: set[int] = set()
    for key, rows in grouped.items():
        ordered = sorted(rows)
        indices = [row[0] for row in ordered]
        if indices != list(range(len(ordered))):
            raise ValueError(f"non-exact frame cover for seed/method {key}")
        times = np.asarray([row[1] for row in ordered], dtype=np.float64)
        values = np.asarray([row[2] for row in ordered], dtype=np.float64)
        if np.any(np.diff(times) <= 0.0):
            raise ValueError(f"time must be strictly increasing for {key}")
        parsed[key] = (times, values)
        frame_counts.add(len(ordered))
    if len(frame_counts) != 1:
        raise ValueError("inconsistent frame counts across seed/method groups")

    seeds = {seed for seed, _ in parsed}
    methods = {method for _, method in parsed}
    for seed in seeds:
        if {method for candidate_seed, method in parsed if candidate_seed == seed} != methods:
            raise ValueError("incomplete method grid in full-frame CSV")
    return parsed


def _selected_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    display = report.get("figure_display")
    if not isinstance(display, dict) or not isinstance(display.get("rows"), list):
        raise ValueError("confirmation report lacks figure-display rows")
    by_id = {row.get("row_id"): row for row in display["rows"] if isinstance(row, dict)}
    if len(by_id) != len(display["rows"]):
        raise ValueError("figure-display row IDs must be unique")
    try:
        rows = [dict(by_id[row_id]) for row_id in _ROW_IDS]
    except KeyError as error:
        raise ValueError(f"missing selected figure row: {error.args[0]}") from error
    for row in rows:
        if int(row["seed"]) < 0 or int(row["frame_index"]) < 0:
            raise ValueError("invalid selected row seed/frame")
    return rows


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        family = "DejaVu Sans"
        properties = font_manager.FontProperties(family=family, weight="bold" if bold else "normal")
        return ImageFont.truetype(font_manager.findfont(properties), size=size)
    except Exception:
        return ImageFont.load_default()


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=4)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    draw.multiline_text(
        (left + (right - left - text_width) / 2, top + (bottom - top - text_height) / 2),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=4,
    )


def _row_metadata(
    row: Mapping[str, Any],
    errors: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
) -> str:
    seed = int(row["seed"])
    frame = int(row["frame_index"])
    time = errors[(seed, "pca")][0][frame]
    values = {method: errors[(seed, method)][1][frame] for method in _METHODS}
    return (
        f"{_ROW_LABELS[str(row['row_id'])]}\n"
        f"seed {seed} | frame {frame} | t={time:.2f}\n"
        f"NRMSE P/L/F = {values['pca']:.3f} / "
        f"{values['legacy_dynamic_kappa']:.3f} / {values['single_start']:.3f}"
    )


def _render_crop_grid(
    source: Image.Image,
    boxes: Sequence[Sequence[tuple[int, int, int, int]]],
    rows: Sequence[Mapping[str, Any]],
    errors: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    columns: Sequence[tuple[int, str]],
    *,
    heading: str,
    note: str,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    panel_width = boxes[0][0][2] - boxes[0][0][0]
    panel_height = boxes[0][0][3] - boxes[0][0][1]
    if any(
        box[2] - box[0] != panel_width or box[3] - box[1] != panel_height
        for row_boxes in boxes
        for box in row_boxes
    ):
        raise ValueError("source panel boxes do not have an identical shape")

    left_label = 390
    right_margin = 24
    column_gap = 14
    row_gap = 78
    heading_height = 126
    footer_height = 58
    width = left_label + len(columns) * panel_width + (len(columns) - 1) * column_gap + right_margin
    height = heading_height + len(rows) * panel_height + (len(rows) - 1) * row_gap + footer_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    _center_text(draw, (10, 4, width - 10, 52), heading, font=_font(30, bold=True))
    _center_text(draw, (10, 53, width - 10, 91), note, font=_font(17), fill="#333333")

    for column_offset, (_, title) in enumerate(columns):
        x = left_label + column_offset * (panel_width + column_gap)
        _center_text(
            draw,
            (x, 90, x + panel_width, heading_height - 4),
            title,
            font=_font(18, bold=True),
        )

    audit: list[dict[str, Any]] = []
    source_row_by_id = {
        "representative_locked_frame": 0,
        "origin_main_legacy_cohort_worst": 1,
        "single_start_cohort_worst": 2,
        "continuation_cohort_worst": 3,
    }
    for row_offset, row in enumerate(rows):
        y = heading_height + row_offset * (panel_height + row_gap)
        _center_text(
            draw,
            (10, y, left_label - 18, y + panel_height),
            _row_metadata(row, errors),
            font=_font(18, bold=True),
        )
        source_row = source_row_by_id[str(row["row_id"])]
        for column_offset, (source_column, title) in enumerate(columns):
            x = left_label + column_offset * (panel_width + column_gap)
            source_box = tuple(boxes[source_row][source_column])
            crop = source.crop(source_box)
            destination_box = (x, y, x + panel_width, y + panel_height)
            canvas.paste(crop, (x, y))
            if np.any(
                np.asarray(canvas.crop(destination_box), dtype=np.uint8)
                != np.asarray(crop, dtype=np.uint8)
            ):
                raise RuntimeError("pixel-preserving panel paste verification failed")
            audit.append(
                {
                    "row_id": row["row_id"],
                    "panel": title,
                    "source_box_left_top_right_bottom": list(source_box),
                    "destination_box_left_top_right_bottom": list(destination_box),
                    "source_panel_pixel_sha256": sha256(crop.tobytes()).hexdigest(),
                    "destination_panel_pixel_sha256": sha256(
                        canvas.crop(destination_box).tobytes()
                    ).hexdigest(),
                    "pixels_equal": True,
                }
            )
    _center_text(
        draw,
        (20, height - footer_height + 4, width - 20, height - 5),
        "P/L/F = PCA / legacy dynamic kappa / fixed single start. No panel autoscaling or display threshold was introduced.",
        font=_font(16),
        fill="#333333",
    )
    return canvas, audit


def _image_png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _write_text_exclusive(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(text)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def _save_matplotlib_png_exclusive(figure: Any, path: Path, *, dpi: int) -> None:
    stream = BytesIO()
    figure.savefig(
        stream,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "render_video_confirmation_presentation.py"},
    )
    _write_bytes_exclusive(path, stream.getvalue())


def _render_curves(
    errors: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    global_max = max(float(values.max()) for (_, method), (_, values) in errors.items() if method in _METHODS)
    upper = max(1.0, math.ceil(global_max * 10.0) / 10.0 + 0.1)
    figure, axes = plt.subplots(3, 1, figsize=(16.0, 12.5), sharex=True, sharey=True)
    for axis, row in zip(axes, rows):
        seed = int(row["seed"])
        frame = int(row["frame_index"])
        for method in _METHODS:
            time, values = errors[(seed, method)]
            axis.plot(
                time,
                values,
                color=_METHOD_COLORS[method],
                linewidth=2.0,
                label=_METHOD_LABELS[method],
                zorder=2,
            )
            axis.scatter(
                [time[frame]],
                [values[frame]],
                color=_METHOD_COLORS[method],
                edgecolor="white",
                linewidth=0.8,
                s=52,
                zorder=4,
            )
        axis.axvline(time[frame], color="#222222", linestyle="--", linewidth=1.2, alpha=0.8)
        axis.axhline(2.0, color="#7f3c8d", linestyle=":", linewidth=1.5, label="Catastrophic threshold = 2.0")
        axis.set_ylim(0.0, upper)
        axis.grid(True, color="#dddddd", linewidth=0.7, alpha=0.8)
        axis.set_ylabel("Normalized frame RMSE", fontsize=13)
        axis.set_title(_row_metadata(row, errors), fontsize=14, fontweight="bold", loc="left")
        axis.tick_params(labelsize=12)
    axes[0].legend(ncol=4, fontsize=11, loc="upper left", frameon=True)
    axes[-1].set_xlabel("Time", fontsize=13)
    figure.suptitle(
        "All 300 frame errors for the three displayed locked cases",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.005,
        "Common y-axis; selected-frame markers correspond exactly to the crop-grid rows. No smoothing or clipping.",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0.02, 0.025, 1.0, 0.975))
    _save_matplotlib_png_exclusive(figure, path, dpi=180)
    plt.close(figure)


def _per_seed_maxima(
    errors: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[list[int], dict[str, np.ndarray]]:
    seeds = sorted({seed for seed, method in errors if method in _METHODS})
    for seed in seeds:
        missing = [method for method in _METHODS if (seed, method) not in errors]
        if missing:
            raise ValueError(f"seed {seed} is missing presentation methods: {missing}")
    maxima = {
        method: np.asarray([errors[(seed, method)][1].max() for seed in seeds], dtype=np.float64)
        for method in _METHODS
    }
    return seeds, maxima


def _render_summary(
    errors: Mapping[tuple[int, str], tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt

    seeds, maxima = _per_seed_maxima(errors)
    positions = np.arange(len(_METHODS), dtype=np.float64)
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(15.0, 10.0), gridspec_kw={"height_ratios": [1.7, 1.0]})
    for index, seed in enumerate(seeds):
        values = np.asarray([maxima[method][index] for method in _METHODS])
        top.plot(positions, values, color="#aaaaaa", linewidth=0.9, alpha=0.55, zorder=1)
    for position, method in zip(positions, _METHODS):
        values = maxima[method]
        top.scatter(
            np.full(values.shape, position),
            values,
            color=_METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            s=55,
            alpha=0.9,
            zorder=3,
        )
        mean = float(values.mean())
        top.scatter([position], [mean], marker="D", color="black", s=75, zorder=4)
        top.annotate(
            f"mean {mean:.3f}",
            (position, mean),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
        )
    top.axhline(2.0, color="#7f3c8d", linestyle=":", linewidth=1.5, label="Catastrophic threshold = 2.0")
    top.set_xticks(positions, [_METHOD_LABELS[method] for method in _METHODS])
    top.set_ylabel("Per-seed maximum frame RMSE", fontsize=13)
    top.set_title("Every locked seed is retained (gray lines pair the same seed)", fontsize=15, fontweight="bold", loc="left")
    top.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    top.tick_params(labelsize=12)
    top.legend(loc="upper left", fontsize=11)

    differences = maxima["single_start"] - maxima["pca"]
    colors = ["#2ca02c" if value < 0.0 else "#d62728" for value in differences]
    bottom.bar(np.arange(len(seeds)), differences, color=colors, width=0.78)
    bottom.axhline(0.0, color="black", linewidth=1.2)
    bottom.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds], rotation=55, ha="right")
    bottom.set_ylabel("Fixed max RMSE - PCA max RMSE", fontsize=13)
    bottom.set_xlabel("Locked confirmation seed", fontsize=13)
    bottom.set_title(
        f"Fixed is lower on {int(np.sum(differences < 0.0))}/{len(seeds)} paired seeds",
        fontsize=15,
        fontweight="bold",
        loc="left",
    )
    bottom.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    bottom.tick_params(labelsize=11)

    figure.suptitle(
        f"Full-sequence summary from {len(seeds)} seeds x {len(next(iter(errors.values()))[1])} frames",
        fontsize=19,
        fontweight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0.02, 0.02, 1.0, 0.965))
    _save_matplotlib_png_exclusive(figure, path, dpi=190)
    plt.close(figure)

    catastrophic_counts = {
        method: int(
            sum(np.count_nonzero(errors[(seed, method)][1] > 2.0) for seed in seeds)
        )
        for method in _METHODS
    }
    return {
        "seed_count": len(seeds),
        "frames_per_seed": len(next(iter(errors.values()))[1]),
        "mean_per_seed_max_frame_rmse": {
            method: float(maxima[method].mean()) for method in _METHODS
        },
        "fixed_lower_max_error_than_pca_seed_count": int(np.sum(differences < 0.0)),
        "catastrophic_frame_threshold": 2.0,
        "catastrophic_frame_count_over_all_seed_frames": catastrophic_counts,
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "size_bytes": int(path.stat().st_size)}


def _presentation_readme() -> str:
    return """# Readable locked-confirmation views

These figures are presentation-only views of the closed formal confirmation
package. No solver was rerun and no model output was changed.

- `reconstruction_selected_frames.png`: clean truth, observed, PCA, exact
  historical dynamic-kappa, and optimized fixed-single-start reconstruction for
  the locked representative, legacy-worst, and fixed-worst rows.
- `outlier_and_error_selected_frames.png`: the matching true/estimated sparse
  layers and absolute reconstruction-error panels.
- `full_frame_error_curves.png`: every one of the 300 frame errors for those
  three displayed cases, on a common y-axis without smoothing or clipping.
- `full_frame_error_summary.png`: all 20 locked seeds, including the legacy
  failures, plus the paired fixed-versus-PCA max-frame differences.

The first two PNGs copy their panel pixels exactly from the formal PNG without
resampling. Titles and row annotations were added outside those copied pixels.
The receipt records every source/destination crop box and pixel digest. The two
curve figures are redrawn deterministically from the formal full-frame CSV.
`presentation_output_manifest.json` hashes every presentation payload.
"""


def render_presentation(
    confirmation_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    confirmation_root = Path(confirmation_dir).resolve()
    output = Path(output_dir).resolve()
    if output == confirmation_root or confirmation_root in output.parents:
        raise ValueError("presentation output must be outside the closed confirmation directory")
    manifest, report = validate_frozen_confirmation(confirmation_root)
    errors = _load_frame_errors(confirmation_root / FORMAL_FRAMES_NAME)
    rows = _selected_rows(report)
    for row in rows:
        seed = int(row["seed"])
        frame = int(row["frame_index"])
        for method in _METHODS:
            key = (seed, method)
            if key not in errors or frame >= len(errors[key][1]):
                raise ValueError(f"selected row is not covered by full-frame CSV: {row['row_id']}")

    with Image.open(confirmation_root / FORMAL_FIGURE_NAME) as loaded:
        source = loaded.convert("RGB")
    boxes = detect_formal_panel_boxes(source)
    main_image, main_crop_audit = _render_crop_grid(
        source,
        boxes,
        rows,
        errors,
        _MAIN_COLUMNS,
        heading="Locked video decomposition: readable reconstruction panels",
        note="Formal panel pixels copied exactly; three prespecified audit rows; one shared formal scene scale.",
    )
    sparse_image, sparse_crop_audit = _render_crop_grid(
        source,
        boxes,
        rows,
        errors,
        _SPARSE_COLUMNS,
        heading="Locked video decomposition: sparse layer and absolute error",
        note="Formal panel pixels copied exactly; native S; no threshold, autoscale, or failure masking.",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing path: {output}") from error

    paths = {
        "main": output / MAIN_FIGURE_NAME,
        "sparse": output / SPARSE_FIGURE_NAME,
        "curves": output / CURVES_FIGURE_NAME,
        "summary": output / SUMMARY_FIGURE_NAME,
        "receipt": output / RECEIPT_NAME,
        "readme": output / README_NAME,
        "manifest": output / OUTPUT_MANIFEST_NAME,
    }
    _write_bytes_exclusive(paths["main"], _image_png_bytes(main_image))
    _write_bytes_exclusive(paths["sparse"], _image_png_bytes(sparse_image))
    _render_curves(errors, rows, paths["curves"])
    full_frame_summary = _render_summary(errors, paths["summary"])

    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "video_confirmation_presentation_receipt",
        "scientific_boundary": (
            "presentation-only; no solver rerun, parameter change, candidate selection, "
            "retuning, panel autoscale, or display threshold"
        ),
        "source_confirmation_directory_name": confirmation_root.name,
        "source_confirmation_manifest_digest": manifest["manifest_sha256"],
        "source_inputs": {
            name: _artifact_record(confirmation_root / name)
            for name in (FORMAL_REPORT_NAME, FORMAL_FRAMES_NAME, FORMAL_FIGURE_NAME)
        },
        "selected_rows": rows,
        "source_panel_grid": {
            "row_count": len(boxes),
            "column_count": len(boxes[0]),
            "boxes_left_top_right_bottom": boxes,
        },
        "pixel_copy_policy": {
            "resampling": "none",
            "source_panels_recolored": False,
            "source_panels_autoscaled": False,
            "source_panels_thresholded": False,
            "labels_added_outside_source_panel_pixels": True,
        },
        "main_figure_panel_audit": main_crop_audit,
        "sparse_figure_panel_audit": sparse_crop_audit,
        "full_frame_summary_recomputed_from_csv": full_frame_summary,
    }
    _write_json_exclusive(paths["receipt"], receipt)
    _write_text_exclusive(paths["readme"], _presentation_readme())

    payload_names = sorted(path.name for key, path in paths.items() if key != "manifest")
    output_manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "video_confirmation_presentation_exact_closure",
        "self_excluding": True,
        "required_payloads": payload_names,
        "files": {name: _artifact_record(output / name) for name in payload_names},
    }
    output_manifest["manifest_sha256"] = _canonical_sha256(output_manifest)
    _write_json_exclusive(paths["manifest"], output_manifest)
    return paths


def validate_presentation_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("presentation path must be a real directory")
    entries = list(root.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"presentation contains non-regular entry: {entry.name}")
    manifest = _strict_json(root / OUTPUT_MANIFEST_NAME)
    payloads = set(manifest.get("required_payloads", []))
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != payloads:
        raise ValueError("presentation manifest payload/file set mismatch")
    if {entry.name for entry in entries} != payloads | {OUTPUT_MANIFEST_NAME}:
        raise ValueError("presentation directory does not match its closed contract")
    body = dict(manifest)
    digest = body.pop("manifest_sha256", None)
    if digest != _canonical_sha256(body):
        raise ValueError("presentation manifest canonical digest mismatch")
    for name, record in files.items():
        if _artifact_record(root / name) != record:
            raise ValueError(f"presentation hash/size mismatch for {name}")
    receipt = _strict_json(root / RECEIPT_NAME)
    audits: Iterable[Mapping[str, Any]] = (
        list(receipt.get("main_figure_panel_audit", []))
        + list(receipt.get("sparse_figure_panel_audit", []))
    )
    if not audits or not all(
        record.get("pixels_equal") is True
        and record.get("source_panel_pixel_sha256")
        == record.get("destination_panel_pixel_sha256")
        for record in audits
    ):
        raise ValueError("presentation pixel-copy audit failed")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render", help="create a new presentation package")
    render.add_argument("confirmation_dir", type=Path)
    render.add_argument("output_dir", type=Path)
    validate = subparsers.add_parser("validate", help="validate a presentation package")
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        paths = render_presentation(args.confirmation_dir, args.output_dir)
        validate_presentation_output(args.output_dir)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
        return 0
    manifest = validate_presentation_output(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

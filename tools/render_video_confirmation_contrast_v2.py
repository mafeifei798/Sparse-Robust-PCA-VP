"""Render an honest high-contrast view of the sealed video confirmation PNG.

This tool never calls a generator or solver.  It validates the closed formal
package, decodes only the quantized ``RdBu_r`` colour indices already present
in the sealed raster, and deterministically remaps those indices for display.
The result is a presentation transform, not recovered floating-point data.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.render_video_confirmation_presentation import (
    FORMAL_FIGURE_NAME,
    FORMAL_REPORT_NAME,
    _canonical_sha256,
    detect_formal_panel_boxes,
    validate_frozen_confirmation,
)


FIGURE_NAME = "reconstruction_selected_frames_contrast_v2.png"
RECEIPT_NAME = "presentation_v2_receipt.json"
README_NAME = "README.md"
MANIFEST_NAME = "presentation_v2_output_manifest.json"

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
_COLUMNS = (
    (0, "Clean truth", "clean"),
    (1, "Observed", "observed"),
    (2, "PCA", "scene"),
    (3, "Exact legacy", "scene"),
    (4, "Fixed robust", "scene"),
)
_SOURCE_ROW_BY_ID = {
    "representative_locked_frame": 0,
    "origin_main_legacy_cohort_worst": 1,
    "single_start_cohort_worst": 2,
    "continuation_cohort_worst": 3,
}
_FORMAL_PANEL_LIMIT = 12.621548138375921
_OBSERVED_QUANTILE = 0.995
_TRIM = 2


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "size_bytes": int(path.stat().st_size)}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant in {path.name}: {value}")

    with path.open(encoding="utf-8") as stream:
        value = json.load(stream, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _source_lut() -> np.ndarray:
    from matplotlib import colormaps

    rgba = colormaps["RdBu_r"](np.linspace(0.0, 1.0, 256))[:, :3]
    return np.rint(255.0 * rgba).astype(np.uint8)


def _target_lut() -> np.ndarray:
    """A signed palette with a dark zero so low-amplitude scenes stay visible."""

    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "purple_dark_amber",
        ("#5b21b6", "#111827", "#f59e0b"),
        N=256,
    )
    return np.rint(255.0 * cmap(np.linspace(0.0, 1.0, 256))[:, :3]).astype(
        np.uint8
    )


def decode_rd_bu_r_indices(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Map raster RGB values to nearest source-colormap indices.

    The operation intentionally returns quantized colour-bin centres, never a
    claim of recovering the original floating-point matrix.
    """

    values = np.asarray(rgb, dtype=np.uint8)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("rgb must have shape (height, width, 3)")
    # int32 is required: three squared 8-bit channel differences can exceed
    # int16 and would otherwise wrap before the nearest-LUT search.
    lut = _source_lut().astype(np.int32)
    flat = values.reshape(-1, 3).astype(np.int32)
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    distances = np.sum((unique[:, None, :] - lut[None, :, :]) ** 2, axis=2)
    selected = np.argmin(distances, axis=1).astype(np.uint8)
    nearest = np.sqrt(np.min(distances, axis=1).astype(np.float64))
    per_pixel = nearest[inverse]
    return selected[inverse].reshape(values.shape[:2]), {
        "maximum_nearest_lut_rgb_l2": float(np.max(per_pixel)),
        "median_nearest_lut_rgb_l2": float(np.median(per_pixel)),
        "mean_nearest_lut_rgb_l2": float(np.mean(per_pixel)),
    }


def quantized_values_from_indices(indices: np.ndarray, formal_limit: float) -> np.ndarray:
    if not np.isfinite(formal_limit) or formal_limit <= 0.0:
        raise ValueError("formal_limit must be finite and positive")
    normalized = np.asarray(indices, dtype=np.float64) / 255.0
    return (2.0 * normalized - 1.0) * float(formal_limit)


def display_scale_and_clipping(
    clean_values: np.ndarray,
    observed_values: np.ndarray,
    scene_values: Sequence[np.ndarray],
    *,
    observed_quantile: float = _OBSERVED_QUANTILE,
) -> dict[str, Any]:
    """Return the predeclared row scales and transparent saturation rates."""

    if not 0.0 < observed_quantile <= 1.0:
        raise ValueError("observed_quantile must lie in (0, 1]")
    clean = np.asarray(clean_values, dtype=np.float64)
    observed = np.asarray(observed_values, dtype=np.float64)
    scenes = [np.asarray(value, dtype=np.float64) for value in scene_values]
    if clean.size == 0 or observed.shape != clean.shape or any(
        value.shape != clean.shape for value in scenes
    ):
        raise ValueError("all display panels must have one equal nonempty shape")
    if not all(np.isfinite(value).all() for value in [clean, observed, *scenes]):
        raise ValueError("display panels must be finite")
    epsilon = float(np.finfo(np.float64).eps)
    clean_limit = max(float(np.max(np.abs(clean))), epsilon)
    observed_limit = max(
        float(np.quantile(np.abs(observed), observed_quantile)), epsilon
    )

    def clipped_fraction(values: np.ndarray, limit: float) -> float:
        return float(np.mean(np.abs(values) > limit))

    return {
        "scene_limit": clean_limit,
        "scene_limit_rule": "row clean-panel maximum absolute decoded colour-bin centre",
        "observed_limit": observed_limit,
        "observed_limit_rule": (
            f"row observed-panel absolute decoded colour-bin {observed_quantile:.3f} quantile"
        ),
        "observed_quantile": float(observed_quantile),
        "scene_clipped_fraction": [
            clipped_fraction(value, clean_limit) for value in scenes
        ],
        "observed_clipped_fraction": clipped_fraction(observed, observed_limit),
    }


def _encode_values(values: np.ndarray, limit: float) -> np.ndarray:
    normalized = np.clip((np.asarray(values) / limit + 1.0) / 2.0, 0.0, 1.0)
    indices = np.rint(255.0 * normalized).astype(np.uint8)
    return _target_lut()[indices]


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        properties = font_manager.FontProperties(
            family="DejaVu Sans", weight="bold" if bold else "normal"
        )
        return ImageFont.truetype(font_manager.findfont(properties), size=size)
    except Exception:
        return ImageFont.load_default()


def _center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    left, top, right, bottom = box
    bounds = draw.multiline_textbbox((0, 0), value, font=font, align="center", spacing=4)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.multiline_text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2),
        value,
        font=font,
        fill=fill,
        align="center",
        spacing=4,
    )


def _selected_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    display = report.get("figure_display")
    if not isinstance(display, dict) or not isinstance(display.get("rows"), list):
        raise ValueError("confirmation report lacks figure-display rows")
    by_id = {row.get("row_id"): row for row in display["rows"] if isinstance(row, dict)}
    if set(_ROW_IDS) - set(by_id):
        raise ValueError("confirmation report lacks a required v2 row")
    formal_limit = float(display.get("scene_symmetric_abs_limit", float("nan")))
    if formal_limit != _FORMAL_PANEL_LIMIT:
        raise ValueError("unexpected sealed formal scene limit")
    return [dict(by_id[row_id]) for row_id in _ROW_IDS]


def _render_contrast(
    source: Image.Image,
    boxes: Sequence[Sequence[tuple[int, int, int, int]]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    panel_width = boxes[0][0][2] - boxes[0][0][0] - 2 * _TRIM
    panel_height = boxes[0][0][3] - boxes[0][0][1] - 2 * _TRIM
    left_label = 320
    gap = 18
    row_gap = 86
    heading = 150
    footer = 96
    width = left_label + len(_COLUMNS) * panel_width + (len(_COLUMNS) - 1) * gap + 24
    height = heading + len(rows) * panel_height + (len(rows) - 1) * row_gap + footer
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    _center_text(
        draw,
        (8, 4, width - 8, 55),
        "Locked confirmation: clean-scale contrast view",
        font=_font(31, bold=True),
    )
    _center_text(
        draw,
        (20, 56, width - 20, 104),
        "Quantized recolour of sealed raster pixels; no solver rerun and no recovered numeric arrays",
        font=_font(18),
        fill="#374151",
    )
    for column, (_, title, _) in enumerate(_COLUMNS):
        x = left_label + column * (panel_width + gap)
        _center_text(draw, (x, 105, x + panel_width, heading), title, font=_font(19, bold=True))

    audits: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        source_row = _SOURCE_ROW_BY_ID[str(row["row_id"])]
        decoded: list[np.ndarray] = []
        decode_diagnostics: list[dict[str, float]] = []
        crop_hashes: list[str] = []
        for source_column, _, _ in _COLUMNS:
            left, top, right, bottom = boxes[source_row][source_column]
            crop = np.asarray(
                source.crop(
                    (left + _TRIM, top + _TRIM, right - _TRIM, bottom - _TRIM)
                ),
                dtype=np.uint8,
            )
            indices, diagnostic = decode_rd_bu_r_indices(crop)
            decoded.append(quantized_values_from_indices(indices, _FORMAL_PANEL_LIMIT))
            decode_diagnostics.append(diagnostic)
            crop_hashes.append(sha256(crop.tobytes()).hexdigest())
        scales = display_scale_and_clipping(decoded[0], decoded[1], [decoded[0], *decoded[2:]])
        y = heading + row_index * (panel_height + row_gap)
        label = (
            f"{_ROW_LABELS[str(row['row_id'])]}\n"
            f"seed {int(row['seed'])} | frame {int(row['frame_index'])}\n"
            f"scene +/-{scales['scene_limit']:.3g}\n"
            f"observed +/-{scales['observed_limit']:.3g}"
        )
        _center_text(draw, (8, y, left_label - 15, y + panel_height), label, font=_font(18, bold=True))
        scene_clip = list(scales["scene_clipped_fraction"])
        scene_clip_by_column = (scene_clip[0], None, scene_clip[1], scene_clip[2], scene_clip[3])
        for column, (_, title, kind) in enumerate(_COLUMNS):
            limit = scales["observed_limit"] if kind == "observed" else scales["scene_limit"]
            recolored = _encode_values(decoded[column], float(limit))
            x = left_label + column * (panel_width + gap)
            canvas.paste(Image.fromarray(recolored, mode="RGB"), (x, y))
            clip_fraction = (
                float(scales["observed_clipped_fraction"])
                if kind == "observed"
                else float(scene_clip_by_column[column] or 0.0)
            )
            annotation = f"clip {100.0 * clip_fraction:.2f}%"
            draw.rounded_rectangle((x + 7, y + 7, x + 118, y + 35), radius=5, fill="#ffffffdd")
            draw.text((x + 13, y + 9), annotation, font=_font(14, bold=True), fill="#111827")
            audits.append(
                {
                    "row_id": row["row_id"],
                    "seed": int(row["seed"]),
                    "frame_index": int(row["frame_index"]),
                    "panel": title,
                    "source_column": int(_COLUMNS[column][0]),
                    "source_trimmed_panel_pixel_sha256": crop_hashes[column],
                    "source_formal_limit": _FORMAL_PANEL_LIMIT,
                    "display_limit": float(limit),
                    "clipped_fraction": clip_fraction,
                    "decode_diagnostic": decode_diagnostics[column],
                    "decode_semantics": "nearest sealed RdBu_r RGB bin; quantized presentation value only",
                }
            )
    _center_text(
        draw,
        (20, height - footer + 3, width - 20, height - 8),
        "Clean/PCA/legacy/fixed share each row's clean-derived scale. Observed uses its own q99.5 scale.\n"
        "Dark centre means zero. Saturation is annotated; consult the sealed full-range audit for original pixels.",
        font=_font(16),
        fill="#374151",
    )
    return canvas, audits


def _png_bytes(image: Image.Image) -> bytes:
    stream = BytesIO()
    image.save(stream, format="PNG", optimize=False, compress_level=6)
    return stream.getvalue()


def _write_bytes_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)


def _write_text_exclusive(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(payload)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _write_text_exclusive(
        path,
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def _readme(source_relative: str) -> str:
    return f"""# Locked video presentation v2

This directory is a presentation-only contrast transform of the closed formal
confirmation raster. It does not rerun a generator or solver, change any
metric, recover floating-point arrays, or create a new scientific result.

- `{FIGURE_NAME}` keeps the three prespecified rows and the exact clean,
  observed, PCA, legacy, and fixed panel identities from the sealed raster.
- Clean/PCA/legacy/fixed share a row-specific scale derived only from the
  sealed clean-panel colour bins. Observed uses a separate predeclared q99.5
  colour-bin scale. Every clipped percentage is printed on the figure.
- The transform first maps each source RGB pixel to the nearest one of the 256
  sealed `RdBu_r` bins, then recolours the resulting **quantized bin centre**
  with a dark-zero palette. It is not numerical data recovery.

The original full-range audit remains authoritative:
[`{source_relative}`]({source_relative}).

`{RECEIPT_NAME}` records source/input hashes, crop hashes, display rules,
clipping rates, source-code drift, and the failed source-identical replay gate.
`{MANIFEST_NAME}` closes every payload by SHA-256 and byte size.
"""


def render_contrast_v2(confirmation_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    confirmation = Path(confirmation_dir).resolve()
    output = Path(output_dir).resolve()
    if output == confirmation or confirmation in output.parents:
        raise ValueError("v2 output must be outside the sealed confirmation directory")
    formal_manifest, report = validate_frozen_confirmation(confirmation)
    rows = _selected_rows(report)
    with Image.open(confirmation / FORMAL_FIGURE_NAME) as loaded:
        source = loaded.convert("RGB")
    boxes = detect_formal_panel_boxes(source)
    figure, audits = _render_contrast(source, boxes, rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing path: {output}") from error
    paths = {
        "figure": output / FIGURE_NAME,
        "receipt": output / RECEIPT_NAME,
        "readme": output / README_NAME,
        "manifest": output / MANIFEST_NAME,
    }
    _write_bytes_exclusive(paths["figure"], _png_bytes(figure))
    current_runner = Path(__file__).resolve().parents[1] / "experiments" / "video_spca_paper_robust_study.py"
    formal_runner_hash = report["provenance"]["source_sha256"]["experiments/video_spca_paper_robust_study.py"]
    current_runner_hash = _file_sha256(current_runner)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "kind": "video_confirmation_quantized_raster_contrast_v2_receipt",
        "scientific_boundary": (
            "presentation-only quantized recolour of sealed raster; no solver rerun, "
            "no floating-point data recovery, no metric change, and no new scientific result"
        ),
        "source_confirmation_directory_name": confirmation.name,
        "source_confirmation_manifest_sha256": formal_manifest["manifest_sha256"],
        "source_inputs": {
            FORMAL_REPORT_NAME: _artifact_record(confirmation / FORMAL_REPORT_NAME),
            FORMAL_FIGURE_NAME: _artifact_record(confirmation / FORMAL_FIGURE_NAME),
        },
        "selected_rows": rows,
        "display_transform": {
            "input": "sealed PNG panel RGB pixels after a fixed two-pixel spine trim",
            "source_colormap": "Matplotlib RdBu_r with 256 bins",
            "source_formal_symmetric_limit": _FORMAL_PANEL_LIMIT,
            "decode": "nearest RGB lookup bin; decoded values are quantized bin centres only",
            "target_palette": "purple-dark-zero-amber",
            "scene_scale_rule": "per-row maximum absolute decoded clean-panel colour-bin centre, shared by clean/PCA/legacy/fixed",
            "observed_scale_rule": "per-row q99.5 absolute decoded observed-panel colour-bin centre",
            "observed_quantile": _OBSERVED_QUANTILE,
            "clipping_disclosed_on_figure": True,
            "panel_selection_changed": False,
            "metric_values_changed": False,
        },
        "panel_audit": audits,
        "source_identical_factor_replay_gate": {
            "status": "failed_closed_not_used",
            "reason": "current-platform seed-51001 data, legacy A/B/reconstruction/S, and fixed A/B/S/reconstruction hashes did not match the sealed Linux run",
            "replay_arrays_used_in_v2": False,
            "factor_modes_figure_emitted": False,
        },
        "source_code_audit": {
            "this_renderer_sha256": _file_sha256(Path(__file__)),
            "formal_runner_sha256": formal_runner_hash,
            "current_runner_sha256": current_runner_hash,
            "formal_runner_matches_current": formal_runner_hash == current_runner_hash,
        },
        "authoritative_full_range_audit": f"../formal_confirm_v1/{FORMAL_FIGURE_NAME}",
    }
    _write_json_exclusive(paths["receipt"], receipt)
    _write_text_exclusive(paths["readme"], _readme(f"../formal_confirm_v1/{FORMAL_FIGURE_NAME}"))
    payloads = sorted(path.name for key, path in paths.items() if key != "manifest")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "video_confirmation_quantized_raster_contrast_v2_exact_closure",
        "self_excluding": True,
        "required_payloads": payloads,
        "files": {name: _artifact_record(output / name) for name in payloads},
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    _write_json_exclusive(paths["manifest"], manifest)
    return paths


def validate_contrast_v2_output(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("v2 presentation path must be a real directory")
    entries = list(root.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"v2 presentation contains non-regular entry: {entry.name}")
    manifest = _strict_json(root / MANIFEST_NAME)
    payloads = set(manifest.get("required_payloads", []))
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != payloads:
        raise ValueError("v2 manifest payload/file set mismatch")
    if {entry.name for entry in entries} != payloads | {MANIFEST_NAME}:
        raise ValueError("v2 directory does not match its closed contract")
    body = dict(manifest)
    declared = body.pop("manifest_sha256", None)
    if declared != _canonical_sha256(body):
        raise ValueError("v2 manifest canonical digest mismatch")
    for name, record in files.items():
        if _artifact_record(root / name) != record:
            raise ValueError(f"v2 hash/size mismatch for {name}")
    receipt = _strict_json(root / RECEIPT_NAME)
    if receipt.get("kind") != "video_confirmation_quantized_raster_contrast_v2_receipt":
        raise ValueError("unexpected v2 receipt kind")
    gate = receipt.get("source_identical_factor_replay_gate", {})
    if gate.get("status") != "failed_closed_not_used" or gate.get("replay_arrays_used_in_v2") is not False:
        raise ValueError("v2 replay boundary is not fail closed")
    transform = receipt.get("display_transform", {})
    if transform.get("metric_values_changed") is not False or transform.get("clipping_disclosed_on_figure") is not True:
        raise ValueError("v2 display-transform boundary is invalid")
    audits = receipt.get("panel_audit")
    if not isinstance(audits, list) or len(audits) != len(_ROW_IDS) * len(_COLUMNS):
        raise ValueError("v2 panel audit is incomplete")
    with Image.open(root / FIGURE_NAME) as image:
        image.verify()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    render = subparsers.add_parser("render")
    render.add_argument("confirmation_dir", type=Path)
    render.add_argument("output_dir", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "render":
        paths = render_contrast_v2(args.confirmation_dir, args.output_dir)
        validate_contrast_v2_output(args.output_dir)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
        return 0
    print(json.dumps(validate_contrast_v2_output(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

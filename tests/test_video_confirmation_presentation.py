from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from tools.render_video_confirmation_presentation import (
    CURVES_FIGURE_NAME,
    FORMAL_FIGURE_NAME,
    FORMAL_FRAMES_NAME,
    FORMAL_MANIFEST_NAME,
    FORMAL_REPORT_NAME,
    MAIN_FIGURE_NAME,
    OUTPUT_MANIFEST_NAME,
    RECEIPT_NAME,
    README_NAME,
    SPARSE_FIGURE_NAME,
    SUMMARY_FIGURE_NAME,
    _canonical_sha256,
    detect_formal_panel_boxes,
    render_presentation,
    validate_frozen_confirmation,
    validate_presentation_output,
)


def _file_record(path: Path) -> dict[str, object]:
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_synthetic_formal_package(root: Path) -> None:
    root.mkdir()
    panel = 60
    gap = 5
    left = 10
    top = 20
    width = left * 2 + 15 * panel + 14 * gap
    height = top * 2 + 4 * panel + 3 * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    for row in range(4):
        y = top + row * (panel + gap)
        for column in range(15):
            x = left + column * (panel + gap)
            color = (
                20 + (row * 37 + column * 11) % 210,
                20 + (row * 17 + column * 23) % 210,
                20 + (row * 29 + column * 7) % 210,
            )
            draw.rectangle(
                (x, y, x + panel - 1, y + panel - 1),
                fill=color,
                outline="black",
                width=2,
            )
    image.save(root / FORMAL_FIGURE_NAME)

    rows = [
        {
            "row_id": "representative_locked_frame",
            "seed": 11,
            "frame_index": 2,
            "selection_rule": "test representative",
        },
        {
            "row_id": "origin_main_legacy_cohort_worst",
            "seed": 12,
            "frame_index": 3,
            "selection_rule": "test legacy worst",
        },
        {
            "row_id": "single_start_cohort_worst",
            "seed": 13,
            "frame_index": 4,
            "selection_rule": "test fixed worst",
        },
        {
            "row_id": "continuation_cohort_worst",
            "seed": 13,
            "frame_index": 4,
            "selection_rule": "test continuation worst",
        },
    ]
    report: dict[str, object] = {"phase": "confirmation", "figure_display": {"rows": rows}}
    report["report_body_sha256"] = _canonical_sha256(report)
    (root / FORMAL_REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with (root / FORMAL_FRAMES_NAME).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["seed", "method", "frame_index", "time", "normalized_rmse"],
        )
        writer.writeheader()
        for seed in (11, 12, 13):
            for method, offset in (
                ("pca", 0.4),
                ("legacy_dynamic_kappa", 1.0),
                ("single_start", 0.2),
            ):
                for frame in range(8):
                    value = offset + 0.02 * frame + 0.001 * (seed - 11)
                    if method == "legacy_dynamic_kappa" and frame == 3:
                        value = 2.5 + 0.01 * (seed - 11)
                    writer.writerow(
                        {
                            "seed": seed,
                            "method": method,
                            "frame_index": frame,
                            "time": float(frame),
                            "normalized_rmse": value,
                        }
                    )

    payloads = sorted((FORMAL_FIGURE_NAME, FORMAL_FRAMES_NAME, FORMAL_REPORT_NAME))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "synthetic_test_confirmation",
        "self_excluding": True,
        "required_payloads": payloads,
        "files": {name: _file_record(root / name) for name in payloads},
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    (root / FORMAL_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_panel_detection_recovers_exact_regular_grid(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    _write_synthetic_formal_package(formal)
    with Image.open(formal / FORMAL_FIGURE_NAME) as image:
        boxes = detect_formal_panel_boxes(image)
    assert len(boxes) == 4
    assert all(len(row) == 15 for row in boxes)
    assert {box[2] - box[0] for row in boxes for box in row} == {60}
    assert {box[3] - box[1] for row in boxes for box in row} == {60}


def test_render_is_closed_pixel_audited_and_exclusive(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    output = tmp_path / "presentation"
    _write_synthetic_formal_package(formal)
    before = {path.name: path.read_bytes() for path in formal.iterdir()}

    render_presentation(formal, output)
    manifest = validate_presentation_output(output)
    assert set(manifest["required_payloads"]) == {
        CURVES_FIGURE_NAME,
        MAIN_FIGURE_NAME,
        README_NAME,
        RECEIPT_NAME,
        SPARSE_FIGURE_NAME,
        SUMMARY_FIGURE_NAME,
    }
    assert {path.name for path in output.iterdir()} == set(
        manifest["required_payloads"]
    ) | {OUTPUT_MANIFEST_NAME}
    receipt = json.loads((output / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert len(receipt["main_figure_panel_audit"]) == 15
    assert len(receipt["sparse_figure_panel_audit"]) == 15
    assert all(
        row["pixels_equal"]
        and row["source_panel_pixel_sha256"]
        == row["destination_panel_pixel_sha256"]
        for key in ("main_figure_panel_audit", "sparse_figure_panel_audit")
        for row in receipt[key]
    )
    assert receipt["full_frame_summary_recomputed_from_csv"][
        "fixed_lower_max_error_than_pca_seed_count"
    ] == 3
    assert {path.name: path.read_bytes() for path in formal.iterdir()} == before

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_presentation(formal, output)


def test_formal_hash_tampering_is_rejected(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    _write_synthetic_formal_package(formal)
    with (formal / FORMAL_FRAMES_NAME).open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_frozen_confirmation(formal)

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools.render_video_confirmation_contrast_v2 import (
    FIGURE_NAME,
    MANIFEST_NAME,
    README_NAME,
    RECEIPT_NAME,
    _canonical_sha256,
    _source_lut,
    decode_rd_bu_r_indices,
    display_scale_and_clipping,
    quantized_values_from_indices,
    validate_contrast_v2_output,
)


def _record(path: Path) -> dict[str, object]:
    return {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _write_synthetic_v2(root: Path) -> None:
    root.mkdir()
    Image.new("RGB", (12, 12), "black").save(root / FIGURE_NAME)
    (root / README_NAME).write_text("presentation only\n", encoding="utf-8")
    receipt = {
        "kind": "video_confirmation_quantized_raster_contrast_v2_receipt",
        "display_transform": {
            "metric_values_changed": False,
            "clipping_disclosed_on_figure": True,
        },
        "source_identical_factor_replay_gate": {
            "status": "failed_closed_not_used",
            "replay_arrays_used_in_v2": False,
        },
        "panel_audit": [{"panel": index} for index in range(15)],
    }
    (root / RECEIPT_NAME).write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    payloads = sorted((FIGURE_NAME, README_NAME, RECEIPT_NAME))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "manifest_kind": "video_confirmation_quantized_raster_contrast_v2_exact_closure",
        "self_excluding": True,
        "required_payloads": payloads,
        "files": {name: _record(root / name) for name in payloads},
    }
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_rd_bu_decode_is_exact_for_source_lut_and_does_not_overflow() -> None:
    indices = np.asarray([[0, 31, 127], [128, 224, 255]], dtype=np.uint8)
    rgb = _source_lut()[indices]
    decoded, diagnostics = decode_rd_bu_r_indices(rgb)
    assert np.array_equal(decoded, indices)
    assert diagnostics["maximum_nearest_lut_rgb_l2"] == 0.0
    values = quantized_values_from_indices(decoded, 12.0)
    assert values[0, 0] == -12.0
    assert values[-1, -1] == 12.0


def test_clean_scale_is_shared_and_observed_clipping_is_disclosed() -> None:
    clean = np.asarray([[-2.0, 0.0], [0.5, 2.0]])
    observed = np.asarray([[-8.0, -1.0], [1.0, 8.0]])
    pca = np.asarray([[-2.5, 0.0], [0.0, 1.0]])
    legacy = np.asarray([[-5.0, 0.0], [0.0, 5.0]])
    fixed = np.asarray([[-1.8, 0.0], [0.0, 1.9]])
    value = display_scale_and_clipping(
        clean, observed, [clean, pca, legacy, fixed], observed_quantile=0.75
    )
    assert value["scene_limit"] == 2.0
    assert value["scene_clipped_fraction"] == [0.0, 0.25, 0.5, 0.0]
    assert value["observed_clipped_fraction"] == 0.0
    assert value["observed_limit"] == 8.0


def test_v2_output_is_closed_and_hash_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "v2"
    _write_synthetic_v2(root)
    validate_contrast_v2_output(root)
    with (root / README_NAME).open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        validate_contrast_v2_output(root)

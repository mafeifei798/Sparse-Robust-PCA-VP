from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from experiments.video_mode_recovery_reference import (
    COMBINED_FIGURE_NAME,
    GITHUB_FIGURE_NAME,
    MANIFEST_NAME,
    RAW_NAME,
    README_NAME,
    REPORT_NAME,
    TRUTH_FIGURE_NAME,
    Figure5PaperStyleConfig,
    align_and_apply_estimate_only_gauge,
    generate_figure5_paper_style_dataset,
    run_figure5_paper_style,
    save_figure5_paper_style,
    smoke_config,
    validate_figure5_paper_style_output,
)


def test_defaults_lock_paper_dimensions_and_historical_vp_settings() -> None:
    config = Figure5PaperStyleConfig()

    assert (config.grid_size, config.n_snapshots, config.dt) == (200, 300, 0.5)
    assert (config.rank, config.alpha, config.max_iter, config.tol) == (
        3,
        10.0,
        500,
        1e-6,
    )


def test_generator_uses_exact_half_second_classic_clean_protocol() -> None:
    config = smoke_config()
    dataset = generate_figure5_paper_style_dataset(config)

    np.testing.assert_array_equal(
        dataset.time, np.arange(300, dtype=np.float64) * 0.5
    )
    assert dataset.time[0] == 0.0
    assert dataset.time[-1] == 149.5
    assert dataset.data.shape == (config.grid_size**2, 300)
    assert dataset.spatial_modes.shape == (config.grid_size**2, 3)
    assert dataset.temporal_raw.shape == (300, 3)
    inactive2 = (dataset.time < 50.0) | (dataset.time > 100.0)
    inactive3 = (dataset.time < 80.0) | (dataset.time > 120.0)
    np.testing.assert_array_equal(dataset.temporal_raw[inactive2, 1], 0.0)
    np.testing.assert_array_equal(dataset.temporal_raw[inactive3, 2], 0.0)
    np.testing.assert_allclose(dataset.data.mean(axis=1), 0.0, atol=2e-15)
    np.testing.assert_allclose(
        dataset.data,
        dataset.spatial_modes @ dataset.temporal_centered.T,
        rtol=1e-14,
        atol=1e-14,
    )


def test_estimate_only_gauge_preserves_reconstruction_without_amplitude_fit() -> None:
    rng = np.random.default_rng(11)
    truth = rng.normal(size=(80, 3))
    modes = np.column_stack((4.0 * truth[:, 2], -2.0 * truth[:, 0], truth[:, 1]))
    dynamics = rng.normal(size=(50, 3))
    native = modes @ dynamics.T

    display = align_and_apply_estimate_only_gauge(truth, modes, dynamics)

    np.testing.assert_allclose(
        display.spatial_modes @ display.temporal_dynamics.T,
        native,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(np.max(np.abs(display.spatial_modes), axis=0), 1.0)
    assert display.permutation == (1, 2, 0)
    assert display.signs == (-1, 1, 1)


def test_smoke_writer_has_exact_closure_raw_replay_and_truthful_boundary(tmp_path) -> None:
    result = run_figure5_paper_style(smoke_config())
    output = tmp_path / "paper-style-smoke"

    paths = save_figure5_paper_style(result, output)
    manifest = validate_figure5_paper_style_output(output)

    assert set(paths) == {
        "combined",
        "github",
        "truth",
        "raw",
        "report",
        "readme",
        "manifest",
    }
    assert {path.name for path in output.iterdir()} == {
        COMBINED_FIGURE_NAME,
        GITHUB_FIGURE_NAME,
        TRUTH_FIGURE_NAME,
        RAW_NAME,
        REPORT_NAME,
        README_NAME,
        MANIFEST_NAME,
    }
    assert paths["combined"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["github"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["truth"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert manifest["self_excluding"] is True
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["paper_scale"] is False
    assert "not exact paper data" in report["protocol_identity"]
    assert report["protocol"]["time_formula"] == "t = np.arange(300) * 0.5"
    assert report["protocol"]["methods"] == ["VP-SPCA", "PCA"]
    assert report["protocol"]["result_driven_tuning"] is False
    assert report["protocol"]["truth_dependent_amplitude_refit"] is False
    record = report["provenance"]["historical_classic_generator"]
    root = Path(__file__).resolve().parents[1]
    historical = subprocess.run(
        ["git", "show", f"{record['commit']}:{record['path']}"],
        cwd=root,
        capture_output=True,
        timeout=10,
        check=False,
    )
    actual_hash = (
        sha256(historical.stdout).hexdigest()
        if historical.returncode == 0
        else None
    )
    if actual_hash is not None:
        assert actual_hash == record["expected_source_sha256"]
    assert record["local_git_object_sha256"] == actual_hash
    assert record["verified"] is (actual_hash == record["expected_source_sha256"])
    contrast = report["paper_qualitative_contrast_audit"]
    assert contrast["pca_matches_aligned_truth_to_numerical_precision"] is True
    assert contrast["paper_pca_mixing_reproduced_by_this_inferred_protocol"] is False
    with np.load(paths["raw"], allow_pickle=False) as arrays:
        assert arrays["data"].shape == (smoke_config().grid_size**2, 300)
        for prefix in ("vp", "pca"):
            np.testing.assert_allclose(
                arrays[f"{prefix}_display_spatial_modes"]
                @ arrays[f"{prefix}_display_temporal_dynamics"].T,
                arrays[f"{prefix}_reconstruction"],
                rtol=2e-12,
                atol=2e-12,
            )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_figure5_paper_style(result, output)


def test_validator_rejects_tampering(tmp_path) -> None:
    source = tmp_path / "source"
    save_figure5_paper_style(run_figure5_paper_style(smoke_config()), source)
    target = tmp_path / "tampered"
    shutil.copytree(source, target)
    with (target / README_NAME).open("a", encoding="utf-8") as stream:
        stream.write("tampered\n")

    with pytest.raises(ValueError, match="payload hash mismatch"):
        validate_figure5_paper_style_output(target)

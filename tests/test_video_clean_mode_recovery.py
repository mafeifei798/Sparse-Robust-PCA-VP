from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from experiments.video_clean_mode_recovery import (
    COMBINED_FIGURE_NAME,
    EXPECTED_DEFAULT_ARRAY_SHA256,
    INTEGRITY_FIGURE_NAME,
    MANIFEST_NAME,
    PCA_FIGURE_NAME,
    RAW_NAME,
    README_NAME,
    REPORT_NAME,
    VP_FIGURE_NAME,
    HistoricalGitHubFigure5Config,
    generate_exact_historical_github_dataset,
    run_historical_github_current,
    save_historical_github_current,
    smoke_config,
    validate_historical_github_current_output,
)
from rspca.datasets import array_sha256


def test_defaults_lock_exact_historical_github_protocol() -> None:
    config = HistoricalGitHubFigure5Config()
    assert (config.grid_size, config.n_snapshots, config.total_time) == (
        200,
        300,
        150.0,
    )
    assert (config.rank, config.alpha, config.max_iter, config.tol) == (
        3,
        10.0,
        500,
        1e-6,
    )


@pytest.mark.historical_environment
def test_default_generator_matches_commit_e01c02f_array_hashes() -> None:
    dataset, phi_true, a_true = generate_exact_historical_github_dataset()
    actual = {
        "X": array_sha256(dataset.data),
        "phi_true": array_sha256(phi_true),
        "a_true": array_sha256(a_true),
        "time": array_sha256(dataset.time),
    }

    assert actual == EXPECTED_DEFAULT_ARRAY_SHA256


def test_default_generator_preserves_shapes_centering_and_time_step() -> None:
    dataset, phi_true, a_true = generate_exact_historical_github_dataset()
    assert dataset.data.shape == (40000, 300)
    assert phi_true.shape == (3, 200, 200)
    assert a_true.shape == (3, 300)
    np.testing.assert_allclose(dataset.data.mean(axis=1), 0.0, atol=2e-15)
    assert dataset.time[1] - dataset.time[0] == pytest.approx(150.0 / 299.0)
    assert dataset.time[1] - dataset.time[0] != 0.5


def test_smoke_output_is_closed_replayable_and_has_no_amplitude_refit(tmp_path) -> None:
    result = run_historical_github_current(smoke_config())
    output = tmp_path / "historical-github-smoke"
    paths = save_historical_github_current(result, output)
    manifest = validate_historical_github_current_output(output)

    assert set(paths) == {
        "vp",
        "pca",
        "integrity",
        "combined",
        "raw",
        "report",
        "readme",
        "manifest",
    }
    assert {path.name for path in output.iterdir()} == {
        VP_FIGURE_NAME,
        PCA_FIGURE_NAME,
        INTEGRITY_FIGURE_NAME,
        COMBINED_FIGURE_NAME,
        RAW_NAME,
        REPORT_NAME,
        README_NAME,
        MANIFEST_NAME,
    }
    assert manifest["self_excluding"] is True
    for key in ("vp", "pca", "integrity", "combined"):
        assert paths[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["protocol"]["methods"] == ["VP-SPCA", "PCA"]
    assert report["protocol"]["result_driven_tuning"] is False
    assert report["protocol"]["native_fit_truth_dependent_amplitude_refit"] is False
    assert (
        report["protocol"][
            "primary_historical_display_truth_dependent_amplitude_refit"
        ]
        is True
    )
    primary_display = report["display"]["github_style_current_vp_spca"]
    integrity_display = report["display"]["no_refit_integrity_current_vp_spca"]
    assert primary_display["truth_overlay"] == "raw generator truth, dashed"
    assert primary_display["truth_dependent_amplitude_refit"] is True
    assert integrity_display["truth_overlay"] == "centered generator truth, dashed"
    assert integrity_display["truth_dependent_amplitude_refit"] is False
    readme = paths["readme"].read_text(encoding="utf-8")
    assert "dashed orange curves\nare the **raw** generator truth" in readme
    assert "the **centered** generator truth" in readme
    assert report["protocol"]["not_exact_paper_data"] is True
    provenance = report["provenance"]
    root = Path(__file__).resolve().parents[1]
    for relative, record in provenance["historical_sources"].items():
        historical = subprocess.run(
            ["git", "show", f"{provenance['historical_git_commit']}:{relative}"],
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
            assert actual_hash == record["expected_sha256"]
        assert record["local_git_object_sha256"] == actual_hash
        assert record["verified"] is (actual_hash == record["expected_sha256"])
    with np.load(paths["raw"], allow_pickle=False) as arrays:
        for prefix in ("vp", "pca"):
            np.testing.assert_allclose(
                arrays[f"{prefix}_display_spatial_modes"]
                @ arrays[f"{prefix}_display_temporal_dynamics"].T,
                arrays[f"{prefix}_reconstruction"],
                rtol=2e-12,
                atol=2e-12,
            )
            np.testing.assert_allclose(
                arrays[f"{prefix}_legacy_display_spatial_modes"]
                @ arrays[f"{prefix}_legacy_display_temporal_gauge"].T,
                arrays[f"{prefix}_reconstruction"],
                rtol=2e-12,
                atol=2e-12,
            )
            assert not np.array_equal(
                arrays[f"{prefix}_legacy_display_temporal_plot"],
                arrays[f"{prefix}_legacy_display_temporal_gauge"],
            )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_historical_github_current(result, output)

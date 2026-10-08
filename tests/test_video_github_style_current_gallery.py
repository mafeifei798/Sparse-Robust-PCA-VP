from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.video_github_style_current_gallery import (
    CLEAN_CONTACT_NAME,
    CLEAN_SUMMARY_NAME,
    MANIFEST_NAME,
    PCA_FIGURE_NAME,
    RAW_NAME,
    README_NAME,
    REPORT_NAME,
    RND_FIGURE_NAME,
    ROBUST_FIGURE_NAME,
    VP_FIGURE_NAME,
    CurrentGitHubGalleryConfig,
    _sealed_robust_expectation,
    run_current_github_gallery,
    save_current_github_gallery,
    smoke_config,
    validate_current_github_gallery_output,
)


def test_defaults_lock_historical_clean_and_sealed_robust_protocol() -> None:
    config = CurrentGitHubGalleryConfig()
    assert (
        config.grid_size,
        config.n_snapshots,
        config.total_time,
        config.rank,
    ) == (200, 300, 150.0, 3)
    assert (config.clean_alpha, config.clean_max_iter, config.clean_tol) == (
        10.0,
        500,
        1e-6,
    )
    assert (
        config.randomized_oversample,
        config.randomized_power_iters,
        config.randomized_full_refine_iters,
        config.randomized_seed,
    ) == (100, 4, 0, 42)
    assert (
        config.robust_seed,
        config.robust_start_id,
        config.robust_alpha,
        config.robust_target_kappa,
    ) == (51001, "pca_perturb_2", 0.0015, 0.001)
    assert config.robust_kappa_schedule == (0.004, 0.001)
    assert config.robust_stage_budgets == (40, 160)
    assert (config.representative_frame_index, config.worst_frame_index) == (
        100,
        171,
    )
    assert config.formal_robust_replay is True


def test_sealed_expectation_names_the_preselected_confirmation_candidate() -> None:
    expectation = _sealed_robust_expectation(51001)
    candidate = expectation["candidate"]
    assert expectation["selected_start_id"] == "pca_perturb_2"
    assert expectation["selected_fit_sha256"] == (
        "b159ac1b0153270b511aa557c3457edb692f8bacbb80dd094cc45b5664422c82"
    )
    assert candidate["kappa_schedule"] == [0.004, 0.001]
    assert candidate["allocated_iterations"] == 200
    assert candidate["converged_at_target"] is True
    assert expectation["method"]["truth_metrics_after_fit_selection_lock"][
        "full_frame_normalized_rmse"
    ]["worst_frame_index"] == 171


def test_smoke_gallery_closes_all_figures_and_replay_arrays(tmp_path) -> None:
    result = run_current_github_gallery(smoke_config())
    output = tmp_path / "current-gallery-smoke"
    paths = save_current_github_gallery(result, output)
    manifest = validate_current_github_gallery_output(output)

    assert set(paths) == {
        "pca",
        "vp",
        "rnd",
        "contact",
        "summary",
        "robust",
        "raw",
        "report",
        "readme",
        "manifest",
    }
    assert {path.name for path in output.iterdir()} == {
        PCA_FIGURE_NAME,
        VP_FIGURE_NAME,
        RND_FIGURE_NAME,
        CLEAN_CONTACT_NAME,
        CLEAN_SUMMARY_NAME,
        ROBUST_FIGURE_NAME,
        RAW_NAME,
        REPORT_NAME,
        README_NAME,
        MANIFEST_NAME,
    }
    assert manifest["self_excluding"] is True
    for key in (
        "pca",
        "vp",
        "rnd",
        "contact",
        "summary",
        "robust",
    ):
        assert paths[key].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["protocol"]["clean_methods"] == [
        "PCA",
        "VP-SPCA",
        "RND-VP-SPCA",
    ]
    assert report["protocol"]["result_driven_tuning"] is False
    assert report["protocol"]["clean_metrics_truth_ls_refit"] is False
    assert report["protocol"]["robust_outlier_post_threshold"] is False
    assert report["robust_replay"]["sealed_exact_replay_verified"] is None
    assert report["display"]["rnd_vp_spca"][
        "truth_dependent_amplitude_refit"
    ] is True
    assert report["display"]["robust_representative"][
        "truth_dependent_fit"
    ] is False

    with np.load(paths["raw"], allow_pickle=False) as arrays:
        assert "rnd_A" in arrays.files
        assert "rnd_B" in arrays.files
        assert "robust_S_normalized" in arrays.files
        assert arrays["robust_representative_clean"].shape == (24 * 24,)
        for prefix in ("pca", "vp", "rnd"):
            np.testing.assert_allclose(
                arrays[f"{prefix}_legacy_spatial_modes"]
                @ arrays[f"{prefix}_legacy_temporal_gauge"].T,
                arrays[f"{prefix}_integrity_spatial_modes"]
                @ arrays[f"{prefix}_integrity_temporal_dynamics"].T,
                rtol=2e-12,
                atol=2e-12,
            )
            assert not np.array_equal(
                arrays[f"{prefix}_legacy_temporal_plot"],
                arrays[f"{prefix}_legacy_temporal_gauge"],
            )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_current_github_gallery(result, output)

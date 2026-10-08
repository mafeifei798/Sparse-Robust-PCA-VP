from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path

import numpy as np
import pytest

import experiments.video_spca as video_experiment
from rspca.datasets import canonical_sha256
from experiments.video_spca import (
    FittedMethod,
    VideoExperimentConfig,
    evaluate_fitted_method,
    generate_video_dataset,
    run_video_experiment,
    save_video_experiment,
    validate_video_output_directory,
    validate_video_output_dir,
)
from spca_multiscale_video import align_components, generate_multiscale_video


def _smoke_config(**overrides: object) -> VideoExperimentConfig:
    values: dict[str, object] = {
        "grid_size": 10,
        "n_timesteps": 36,
        "total_time": 12.0,
        "noise_std": 0.03,
        "corruption_fraction": 0.02,
        "corruption_scale": 1.0,
        "seed": 2025,
        "alpha": 0.08,
        "max_iter": 3,
        "tol": 1e-4,
        "randomized_oversample": 2,
        "randomized_power_iters": 1,
        "randomized_full_refine_iters": 1,
    }
    values.update(overrides)
    return VideoExperimentConfig(**values)  # type: ignore[arg-type]


def test_generator_is_time_by_pixels_noisy_corrupted_and_deterministic() -> None:
    config = _smoke_config()
    first = generate_video_dataset(config)
    repeated = generate_video_dataset(config)
    different_seed = generate_video_dataset(_smoke_config(seed=2026))

    assert first.observed.shape == (config.n_timesteps, config.grid_size**2)
    assert first.spatial_components.shape == (config.grid_size**2, 3)
    assert first.temporal_scores.shape == (config.n_timesteps, 3)
    assert first.spatial_support.shape == first.spatial_components.shape
    assert first.corruption_support.shape == first.observed.shape
    assert np.linalg.matrix_rank(first.clean) == 3
    # Noise/corruptions make the centred observed data more than exact rank 3,
    # so the PCA reconstruction baseline is not guaranteed a zero residual.
    assert np.linalg.matrix_rank(first.observed) > 3
    assert np.linalg.norm(first.observed - first.clean) > 0.0
    np.testing.assert_allclose(first.observed.mean(axis=0), 0.0, atol=1e-14)

    expected_corruptions = 2 * round(
        config.corruption_fraction * first.observed.size / 2.0
    )
    assert np.count_nonzero(first.corruption_support) == expected_corruptions
    np.testing.assert_array_equal(
        first.corruption_support, first.injected_corruption != 0.0
    )
    np.testing.assert_allclose(first.injected_corruption.sum(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(first.gaussian_noise.mean(axis=0), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        first.observed,
        first.clean + first.gaussian_noise + first.injected_corruption,
        atol=0.0,
        rtol=0.0,
    )
    actual_sparse_increment = first.observed - (first.clean + first.gaussian_noise)
    np.testing.assert_array_equal(
        actual_sparse_increment != 0.0, first.corruption_support
    )
    np.testing.assert_allclose(
        actual_sparse_increment,
        first.injected_corruption,
        rtol=1e-15,
        atol=1e-15,
    )
    np.testing.assert_array_equal(first.observed, repeated.observed)
    assert first.digest == repeated.digest
    assert first.digest != different_seed.digest
    assert not np.array_equal(first.observed, different_seed.observed)


def test_data_digest_ignores_solver_only_configuration() -> None:
    baseline = generate_video_dataset(
        _smoke_config(seed=19, max_iter=10, tol=1e-3)
    )
    different_solver_budget = generate_video_dataset(
        _smoke_config(seed=19, max_iter=900, tol=1e-9)
    )

    np.testing.assert_array_equal(
        baseline.observed, different_solver_budget.observed
    )
    assert baseline.digest == different_solver_budget.digest


def test_all_methods_use_one_data_digest_and_report_required_metrics() -> None:
    result = run_video_experiment(_smoke_config())

    assert set(result.reports) == {
        "PCA",
        "VP-SPCA",
        "RND-VP-SPCA",
        "Robust-VP-SPCA",
    }
    assert {report.data_sha256 for report in result.reports.values()} == {
        result.dataset.digest
    }
    assert result.reports["PCA"].observed_reconstruction_error > 0.0
    for name, report in result.reports.items():
        assert np.isfinite(report.observed_reconstruction_error)
        assert np.isfinite(report.clean_reconstruction_error)
        assert 0.0 <= report.achieved_sparsity <= 1.0
        assert 0.0 <= report.subspace_error <= 1.0
        assert 0.0 <= report.support_precision <= 1.0
        assert 0.0 <= report.support_recall <= 1.0
        assert 0.0 <= report.support_f1 <= 1.0
        assert report.runtime_seconds >= 0.0
        assert report.n_iter >= 1
        assert len(report.component_permutation) == 3
        assert len(report.component_signs) == 3
        assert result.fits[name].loadings.shape[0] == _smoke_config().grid_size ** 2
    assert result.reports["Robust-VP-SPCA"].corruption_detection is not None
    budget_detection = result.reports[
        "Robust-VP-SPCA"
    ].corruption_detection_at_budget
    assert budget_detection is not None
    expected_budget = round(
        result.config.corruption_fraction * result.dataset.observed.size
    )
    assert budget_detection["budget_fraction"] == pytest.approx(
        result.config.corruption_fraction
    )
    assert budget_detection["budget_top_k"] == expected_budget
    assert (
        budget_detection["true_positives"] + budget_detection["false_positives"]
        == expected_budget
    )
    assert result.reports["PCA"].corruption_detection is None
    assert result.reports["PCA"].corruption_detection_at_budget is None


def test_scoring_aligns_permutation_and_sign_but_never_refits_amplitude() -> None:
    config = _smoke_config()
    dataset = generate_video_dataset(config)
    permutation = np.array([2, 0, 1])
    signs = np.array([-1.0, 1.0, -1.0])
    axes = dataset.spatial_components[:, permutation] * signs
    loadings = 7.0 * axes
    scores = 3.0 * dataset.temporal_scores[:, permutation] * signs
    fitted = FittedMethod(
        name="permuted-scaled-test",
        axes=axes,
        loadings=loadings,
        scores=scores,
        # A post-hoc amplitude fit could make this perfect; the fixed scorer
        # must retain the native factor of two and therefore report error one.
        reconstruction=2.0 * dataset.clean,
        runtime_seconds=0.0,
        converged=True,
        n_iter=1,
        final_stationarity=0.0,
        status="test",
    )

    report = evaluate_fitted_method(dataset, fitted, config)

    assert report.clean_reconstruction_error == pytest.approx(1.0)
    assert report.loading_component_mean_abs_correlation == pytest.approx(1.0)
    assert report.axis_component_mean_abs_correlation == pytest.approx(1.0)
    assert report.temporal_score_mean_abs_correlation == pytest.approx(1.0)
    assert report.support_precision == pytest.approx(1.0)
    assert report.support_recall == pytest.approx(1.0)
    assert report.support_f1 == pytest.approx(1.0)


def test_temporal_score_uses_spatial_assignment_without_second_matching() -> None:
    config = _smoke_config()
    dataset = generate_video_dataset(config)
    # Spatial factors are already in the correct order, while temporal scores
    # are cyclically wrong.  An independent second Hungarian match would hide
    # this pairing failure and incorrectly report perfect temporal recovery.
    mismatched_scores = dataset.temporal_scores[:, [1, 2, 0]]
    fitted = FittedMethod(
        name="temporal-pairing-counterexample",
        axes=dataset.spatial_components,
        loadings=dataset.spatial_components,
        scores=mismatched_scores,
        reconstruction=np.zeros_like(dataset.observed),
        runtime_seconds=0.0,
        converged=True,
        n_iter=1,
        final_stationarity=0.0,
        status="test",
    )

    report = evaluate_fitted_method(dataset, fitted, config)

    assert report.component_permutation == (0, 1, 2)
    assert report.temporal_score_mean_abs_correlation < 0.1


def test_saved_report_records_orientation_and_no_amplitude_fit(tmp_path) -> None:
    result = run_video_experiment(_smoke_config(), methods=("PCA", "VP-SPCA"))
    paths = save_video_experiment(result, tmp_path / "saved")

    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert payload["data"]["orientation"] == "time_samples_x_pixel_features"
    assert payload["schema_version"] == 2
    assert payload["runtime_versions"]["python"]
    assert payload["runtime_versions"]["numpy"] == np.__version__
    assert payload["data"]["observed_rank"] > payload["data"]["clean_rank"]
    assert payload["evaluation_protocol"]["shared_data_and_seed"] is True
    assert payload["evaluation_protocol"]["independent_temporal_matching"] is False
    assert payload["evaluation_protocol"]["post_hoc_amplitude_fit"] is False
    assert "exact" in payload["evaluation_protocol"]["corruption_ground_truth"]
    assert "reused" in payload["evaluation_protocol"]["ground_truth_alignment"]
    assert payload["evaluation_protocol"]["outlier_budget_source"] == (
        "config.corruption_fraction"
    )
    scoring_hashes = payload["data"]["scoring_input_sha256"]
    assert set(scoring_hashes) == {
        "observed",
        "clean",
        "spatial_components",
        "temporal_scores",
        "spatial_support",
        "corruption_support",
    }
    assert set(payload["data"]["provenance_array_sha256"]) == {
        "gaussian_noise",
        "injected_corruption",
        "time",
        "feature_mean_removed",
    }
    assert scoring_hashes["observed"] == result.dataset.digest
    manifest_body = dict(payload["data"])
    manifest_sha256 = manifest_body.pop("manifest_sha256")
    assert manifest_sha256 == canonical_sha256(manifest_body)
    assert {
        report["dataset_manifest_sha256"] for report in payload["methods"].values()
    } == {manifest_sha256}
    assert set(payload["methods"]) == {"PCA", "VP-SPCA"}


def test_zero_corruption_report_is_strict_json_and_uses_zero_budget(tmp_path) -> None:
    result = run_video_experiment(
        _smoke_config(corruption_fraction=0.0, max_iter=1),
        methods=("Robust-VP-SPCA",),
    )
    json_path = save_video_experiment(result, tmp_path / "zero-corruption")["json"]
    raw = json_path.read_text(encoding="utf-8")

    def reject_nonstandard_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant: {value}")

    payload = json.loads(raw, parse_constant=reject_nonstandard_constant)
    report = payload["methods"]["Robust-VP-SPCA"]
    assert report["corruption_detection"]["roc_auc"] is None
    assert report["corruption_detection"]["average_precision"] is None
    assert report["corruption_detection_at_budget"]["budget_top_k"] == 0
    assert report["corruption_detection_at_budget"]["true_positives"] == 0
    assert report["corruption_detection_at_budget"]["false_positives"] == 0


def test_save_atomically_claims_new_directory_and_never_overwrites(tmp_path: Path) -> None:
    result = run_video_experiment(_smoke_config(), methods=("PCA",))
    output_dir = tmp_path / "exclusive"
    paths = save_video_experiment(result, output_dir, make_plot=True)
    original = {name: path.read_bytes() for name, path in paths.items()}

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_video_experiment(result, output_dir, make_plot=True)

    assert {name: path.read_bytes() for name, path in paths.items()} == original


def test_completion_manifest_is_last_self_excluding_and_validates_exact_closure(
    tmp_path: Path,
) -> None:
    result = run_video_experiment(_smoke_config(), methods=("PCA",))
    output_dir = tmp_path / "manifested"
    paths = save_video_experiment(result, output_dir)

    assert set(paths) == {"json", "csv", "manifest"}
    manifest = validate_video_output_directory(output_dir)
    assert manifest["self_excluding_manifest"] is True
    assert {record["path"] for record in manifest["artifacts"]} == {
        paths["json"].name,
        paths["csv"].name,
    }
    assert paths["manifest"].name not in {
        record["path"] for record in manifest["artifacts"]
    }

    (output_dir / "undeclared.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure mismatch"):
        validate_video_output_directory(output_dir)


def test_partial_write_never_emits_completion_manifest_or_deletes_claimed_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_video_experiment(_smoke_config(), methods=("PCA",))
    output_dir = tmp_path / "partial"
    real_open = video_experiment.os.open

    def fail_on_csv(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "video_spca_metrics.csv":
            raise OSError("synthetic CSV write failure")
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(video_experiment.os, "open", fail_on_csv)
    with pytest.raises(OSError, match="synthetic CSV write failure"):
        save_video_experiment(result, output_dir)

    assert output_dir.is_dir()
    assert (output_dir / "video_spca_metrics.json").is_file()
    assert not (output_dir / "video_spca_output_manifest.json").exists()
    with pytest.raises(ValueError, match="completion manifest is missing"):
        validate_video_output_directory(output_dir)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_save_rejects_nonfinite_nested_diagnostics_before_claim(
    tmp_path: Path,
    invalid: float,
) -> None:
    result = run_video_experiment(_smoke_config(), methods=("VP-SPCA",))
    report = result.reports["VP-SPCA"]
    tampered_report = replace(
        report,
        extra={**report.extra, "nested": {"invalid": invalid}},
    )
    tampered = replace(result, reports={"VP-SPCA": tampered_report})
    output_dir = tmp_path / "nonfinite"

    with pytest.raises(ValueError, match="number must be finite at root.methods"):
        save_video_experiment(tampered, output_dir)

    assert not os.path.lexists(output_dir)


def test_nonfinite_ranking_metric_is_nullable_only_for_zero_corruption(
    tmp_path: Path,
) -> None:
    result = run_video_experiment(_smoke_config(), methods=("Robust-VP-SPCA",))
    report = result.reports["Robust-VP-SPCA"]
    assert report.corruption_detection is not None
    tampered_detection = dict(report.corruption_detection)
    tampered_detection["roc_auc"] = float("nan")
    tampered_report = replace(report, corruption_detection=tampered_detection)
    tampered = replace(result, reports={"Robust-VP-SPCA": tampered_report})
    output_dir = tmp_path / "invalid-ranking"

    with pytest.raises(ValueError, match="corruption_detection.roc_auc"):
        save_video_experiment(tampered, output_dir)

    assert not os.path.lexists(output_dir)


def test_plot_render_failure_happens_before_output_directory_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = run_video_experiment(_smoke_config(), methods=("PCA",))
    output_dir = tmp_path / "plot-failure"

    def fail_before_claim(_: object) -> bytes:
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(video_experiment, "_summary_plot_png_bytes", fail_before_claim)
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        save_video_experiment(result, output_dir, make_plot=True)
    assert not os.path.lexists(output_dir)


def test_cli_preflights_existing_and_dangling_output_before_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = (tmp_path / "existing", tmp_path / "dangling")
    targets[0].mkdir()
    targets[1].symlink_to(tmp_path / "missing-target", target_is_directory=True)

    def solver_must_not_run(*_: object, **__: object) -> object:
        raise AssertionError("solver ran before output preflight")

    monkeypatch.setattr(video_experiment, "run_video_experiment", solver_must_not_run)
    for target in targets:
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            video_experiment.main(["--output-dir", str(target)])
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            validate_video_output_dir(target)


def test_legacy_generator_retains_four_outputs_with_corrected_orientation() -> None:
    observed, modes, scores, time = generate_multiscale_video(
        n_grid=10,
        n_timesteps=36,
        total_time=12.0,
        noise_std=0.03,
        corruption_fraction=0.02,
        seed=2025,
    )

    assert observed.shape == (36, 100)
    assert modes.shape == (3, 10, 10)
    assert scores.shape == (3, 36)
    assert time.shape == (36,)


def test_legacy_alignment_uses_only_permutation_and_sign() -> None:
    truth = np.eye(3).reshape(3, 1, 3)
    recovered = 5.0 * truth[[2, 0, 1]] * np.array([-1.0, 1.0, -1.0])[:, None, None]

    permutation, signs = align_components(recovered, truth)

    np.testing.assert_array_equal(permutation, [1, 2, 0])
    np.testing.assert_array_equal(signs, [1.0, -1.0, -1.0])


def test_generator_validation_rejects_exactly_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        generate_video_dataset(_smoke_config(grid_size=7))
    with pytest.raises(ValueError, match="corruption_fraction"):
        generate_video_dataset(_smoke_config(corruption_fraction=1.0))


def test_zero_corruption_scale_has_no_false_truth_support() -> None:
    dataset = generate_video_dataset(_smoke_config(corruption_scale=0.0))

    assert not np.any(dataset.corruption_support)
    assert not np.any(dataset.injected_corruption)
    np.testing.assert_array_equal(
        dataset.observed, dataset.clean + dataset.gaussian_noise
    )

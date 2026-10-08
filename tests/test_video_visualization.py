from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

from matplotlib import animation
import numpy as np
import pytest

import experiments.video_spca_visualize as visualization
from experiments.video_spca import (
    VideoExperimentConfig,
    generate_video_dataset,
    run_video_experiment,
)
from experiments.video_spca_visualize import (
    default_frame_indices,
    normalized_frame_rmse,
    project_visualization_manifest_for_local_copy,
    validate_frame_indices,
    validate_visualization_manifest_local_copy,
    write_video_visualizations,
)
from rspca.datasets import canonical_sha256


def _result():
    config = VideoExperimentConfig(
        grid_size=8,
        n_timesteps=12,
        total_time=4.0,
        noise_std=0.02,
        corruption_fraction=0.02,
        corruption_scale=0.5,
        seed=71,
        alpha=0.08,
        max_iter=2,
        tol=1e-4,
        randomized_oversample=1,
        randomized_power_iters=0,
        randomized_full_refine_iters=0,
    )
    return run_video_experiment(
        config,
        methods=("PCA", "RND-VP-SPCA", "Robust-VP-SPCA"),
    )


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _rehash_manifest(payload: dict) -> dict:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def _assert_rejected_before_output(
    result,
    tmp_path: Path,
    *,
    match: str,
) -> None:
    output_dir = tmp_path / "must-not-be-created" / "visual-audit"
    assert not output_dir.parent.exists()
    with pytest.raises((TypeError, ValueError), match=match):
        write_video_visualizations(result, output_dir, animation_formats=())
    assert not output_dir.parent.exists()


def test_frame_selection_and_normalized_errors_are_explicit() -> None:
    assert default_frame_indices(10, count=5) == (0, 2, 4, 7, 9)
    assert validate_frame_indices((9, 0, 4), 10) == (9, 0, 4)
    with pytest.raises(ValueError, match="duplicates"):
        validate_frame_indices((0, 0), 10)
    with pytest.raises(ValueError, match="outside"):
        validate_frame_indices((10,), 10)

    result = _result()
    errors = normalized_frame_rmse(result)
    clean_rms = np.sqrt(np.mean(np.square(result.dataset.clean)))
    assert set(errors) == {"PCA", "RND-VP-SPCA", "Robust-VP-SPCA"}
    for name, values in errors.items():
        expected = (
            np.sqrt(
                np.mean(
                    np.square(result.fits[name].reconstruction - result.dataset.clean),
                    axis=1,
                )
            )
            / clean_rms
        )
        np.testing.assert_allclose(values, expected, rtol=0.0, atol=0.0)


def test_static_visual_audit_is_complete_hashed_and_non_overwriting(
    tmp_path: Path,
) -> None:
    result = _result()
    output_dir = tmp_path / "visual-audit"
    paths = write_video_visualizations(
        result,
        output_dir,
        frame_indices=(0, 5, 11),
        animation_formats=(),
        dpi=72,
    )

    assert set(paths) == {
        "ownership_receipt",
        "fixed_frames",
        "per_frame_error_plot",
        "corruption_layers",
        "per_frame_error_csv",
        "manifest",
    }
    assert all(path.is_file() for path in paths.values())
    assert paths["fixed_frames"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["corruption_layers"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["data_sha256"] == result.dataset.digest
    assert payload["orientation"] == "time_samples_x_pixel_features"
    assert payload["frame_indices"] == [0, 5, 11]
    assert payload["frame_error_definition"]["ground_truth_used_only_after_fitting"]
    assert payload["support_contract"]["ground_truth_used_for_selection"] is False
    assert payload["support_contract"]["selected_cells"] == round(
        result.config.corruption_fraction * result.dataset.observed.size
    )
    assert (
        payload["display_contract"]["corruption_truth_and_estimate_share_limits"]
        is True
    )
    assert payload["visual_inference_qualified"] is False
    assert (
        payload["visual_inference_qualification"]["multi_seed_inference_required"]
        is True
    )
    assert set(payload["method_diagnostics"]) == set(result.reports)
    for name, report in result.reports.items():
        diagnostics = payload["method_diagnostics"][name]
        assert set(diagnostics) == {
            "converged",
            "status",
            "n_iter",
            "final_stationarity",
            "corruption_detection",
            "corruption_detection_at_budget",
        }
        assert diagnostics["converged"] is report.converged
        assert diagnostics["status"] == report.status
        assert diagnostics["n_iter"] == report.n_iter
        assert diagnostics["final_stationarity"] == report.final_stationarity
        assert diagnostics["corruption_detection"] == report.corruption_detection
        assert (
            diagnostics["corruption_detection_at_budget"]
            == report.corruption_detection_at_budget
        )
    robust_diagnostics = payload["method_diagnostics"]["Robust-VP-SPCA"]
    assert robust_diagnostics["corruption_detection"] is not None
    assert robust_diagnostics["corruption_detection_at_budget"] is not None
    manifest_body = dict(payload)
    manifest_hash = manifest_body.pop("manifest_sha256")
    assert manifest_hash == canonical_sha256(manifest_body)
    for artifact_name, artifact in payload["artifacts"].items():
        assert artifact_name in paths
        artifact_path = output_dir / artifact["path"]
        assert artifact["sha256"] == _sha256(artifact_path)
        assert artifact["size_bytes"] == artifact_path.stat().st_size
        assert artifact["availability"] == "local_output"
        assert artifact["local_copy_included"] is True
    validate_visualization_manifest_local_copy(payload, output_dir)

    incomplete_detection = json.loads(json.dumps(payload))
    incomplete_detection["method_diagnostics"]["Robust-VP-SPCA"][
        "corruption_detection"
    ].pop("f1")
    incomplete_detection = _rehash_manifest(incomplete_detection)
    with pytest.raises(ValueError, match="corruption_detection is incomplete"):
        validate_visualization_manifest_local_copy(
            incomplete_detection,
            output_dir,
        )

    overstated_inference = json.loads(json.dumps(payload))
    overstated_inference["visual_inference_qualified"] = True
    overstated_inference = _rehash_manifest(overstated_inference)
    with pytest.raises(ValueError, match="cannot qualify inference"):
        validate_visualization_manifest_local_copy(overstated_inference, output_dir)

    invalid_animation = json.loads(json.dumps(payload))
    invalid_animation["animation"]["stride"] = 0
    invalid_animation = _rehash_manifest(invalid_animation)
    with pytest.raises(ValueError, match="animation.stride"):
        validate_visualization_manifest_local_copy(invalid_animation, output_dir)

    with paths["per_frame_error_csv"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == result.config.n_timesteps
    assert list(rows[0]) == [
        "frame_index",
        "time",
        "PCA",
        "RND-VP-SPCA",
        "Robust-VP-SPCA",
    ]
    assert rows[0]["frame_index"] == "0"

    with pytest.raises(FileExistsError, match="already exists"):
        write_video_visualizations(result, output_dir, animation_formats=())


def test_mixed_seed_dataset_fits_and_reports_are_rejected_before_io(
    tmp_path: Path,
) -> None:
    seed_71_result = _result()
    seed_72_dataset = generate_video_dataset(replace(seed_71_result.config, seed=72))
    mixed_result = replace(seed_71_result, dataset=seed_72_dataset)

    _assert_rejected_before_output(
        mixed_result,
        tmp_path,
        match="dataset/config closure mismatch",
    )


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("animation_stride", 0),
        ("animation_stride", -1),
        ("animation_stride", 1.5),
        ("animation_stride", True),
        ("fps", 0),
        ("fps", -1),
        ("fps", 1.5),
        ("fps", True),
    ],
)
def test_static_writer_rejects_invalid_animation_integers_before_io(
    keyword: str,
    value: object,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "must-not-be-created" / "visual-audit"
    with pytest.raises(ValueError, match="positive integer"):
        write_video_visualizations(
            _result(),
            output_dir,
            animation_formats=(),
            **{keyword: value},
        )
    assert not output_dir.parent.exists()


def test_atomic_claim_never_replaces_concurrent_empty_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "concurrent-empty"
    original_claim = visualization._OwnedOutputDirectory.claim

    def racing_claim(cls, destination: Path):
        destination.mkdir()
        return original_claim(destination)

    monkeypatch.setattr(
        visualization._OwnedOutputDirectory,
        "claim",
        classmethod(racing_claim),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        write_video_visualizations(_result(), output_dir, animation_formats=())
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


@pytest.mark.parametrize("existing_kind", ["empty_directory", "file"])
def test_writer_never_replaces_existing_output(
    existing_kind: str,
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "existing-output"
    if existing_kind == "empty_directory":
        output_dir.mkdir()
        expected_bytes = None
    else:
        expected_bytes = b"owned by another process"
        output_dir.write_bytes(expected_bytes)

    with pytest.raises(FileExistsError, match="already exists"):
        write_video_visualizations(_result(), output_dir, animation_formats=())
    if expected_bytes is None:
        assert output_dir.is_dir() and list(output_dir.iterdir()) == []
    else:
        assert output_dir.read_bytes() == expected_bytes


def test_writer_never_follows_or_replaces_dangling_output_symlink(
    tmp_path: Path,
) -> None:
    dangling_target = tmp_path / "absent-target"
    output_dir = tmp_path / "dangling-output"
    output_dir.symlink_to(dangling_target, target_is_directory=True)
    assert os.path.lexists(output_dir)
    assert not output_dir.exists()

    with pytest.raises(FileExistsError, match="already exists"):
        write_video_visualizations(_result(), output_dir, animation_formats=())
    assert output_dir.is_symlink()
    assert output_dir.readlink() == dangling_target
    assert not dangling_target.exists()


@pytest.mark.parametrize("existing_kind", ["directory", "dangling_symlink"])
def test_cli_existing_output_fails_before_running_solver(
    existing_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "occupied-cli-output"
    if existing_kind == "directory":
        output_dir.mkdir()
    else:
        output_dir.symlink_to(tmp_path / "absent-target", target_is_directory=True)
        assert os.path.lexists(output_dir) and not output_dir.exists()

    solver_called = False

    def unexpected_solver_call(*args, **kwargs):
        nonlocal solver_called
        solver_called = True
        raise AssertionError("solver must not run for an occupied output path")

    monkeypatch.setattr(visualization, "run_video_experiment", unexpected_solver_call)
    with pytest.raises(FileExistsError, match="already exists"):
        visualization.main(["--output-dir", str(output_dir)])
    assert not solver_called


def test_partial_render_failure_preserves_owned_output_for_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "new-parent" / "partial-output"

    def fail_after_partial_write(*args, **kwargs):
        path = args[2]
        path.write_bytes(b"partial")
        raise RuntimeError("injected render failure")

    monkeypatch.setattr(
        visualization,
        "_save_fixed_frame_figure",
        fail_after_partial_write,
    )
    with pytest.raises(RuntimeError, match="injected render failure"):
        write_video_visualizations(_result(), output_dir, animation_formats=())
    assert output_dir.is_dir()
    assert (output_dir / "video_visualization_ownership.json").is_file()
    assert (output_dir / "video_decomposition_fixed_frames.png").read_bytes() == (
        b"partial"
    )


def test_failure_preserves_replacement_when_owned_inode_moves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "raced-output"
    moved_owned_dir = tmp_path / "moved-owned-output"
    sentinel = b"replacement belongs to another process"

    def replace_destination_then_fail(*args, **kwargs):
        path = args[2]
        path.parent.rename(moved_owned_dir)
        path.parent.mkdir()
        (path.parent / "sentinel.bin").write_bytes(sentinel)
        raise RuntimeError("injected ownership race")

    monkeypatch.setattr(
        visualization,
        "_save_fixed_frame_figure",
        replace_destination_then_fail,
    )
    with pytest.raises(RuntimeError, match="injected ownership race"):
        write_video_visualizations(_result(), output_dir, animation_formats=())

    assert (output_dir / "sentinel.bin").read_bytes() == sentinel
    assert moved_owned_dir.is_dir()
    assert (moved_owned_dir / "video_visualization_ownership.json").is_file()


def test_check_then_path_swap_cannot_delete_replacement(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "owned-output"
    moved_owned_dir = tmp_path / "moved-after-check"
    replacement_bytes = b"must survive cleanup"
    ownership = visualization._OwnedOutputDirectory.claim(output_dir)
    try:
        assert ownership.still_owned()
        output_dir.rename(moved_owned_dir)
        output_dir.mkdir()
        (output_dir / "replacement.bin").write_bytes(replacement_bytes)

        ownership.cleanup_after_failure()

        assert (output_dir / "replacement.bin").read_bytes() == replacement_bytes
        assert (moved_owned_dir / "video_visualization_ownership.json").is_file()
    finally:
        ownership.close()


@pytest.mark.parametrize("field_name", ["observed", "time", "clean"])
def test_nonfinite_dataset_plot_inputs_are_rejected_before_io(
    field_name: str,
    tmp_path: Path,
) -> None:
    result = _result()
    values = getattr(result.dataset, field_name).copy()
    values.flat[0] = np.nan
    bad_dataset = replace(result.dataset, **{field_name: values})

    _assert_rejected_before_output(
        replace(result, dataset=bad_dataset),
        tmp_path,
        match=rf"dataset\.{field_name}",
    )


@pytest.mark.parametrize(
    ("method_name", "field_name"),
    [
        ("RND-VP-SPCA", "axes"),
        ("RND-VP-SPCA", "loadings"),
        ("RND-VP-SPCA", "scores"),
        ("RND-VP-SPCA", "reconstruction"),
        ("Robust-VP-SPCA", "outliers"),
    ],
)
def test_nonfinite_fitted_plot_arrays_are_rejected_before_io(
    method_name: str,
    field_name: str,
    tmp_path: Path,
) -> None:
    result = _result()
    fitted = result.fits[method_name]
    values = getattr(fitted, field_name).copy()
    values.flat[-1] = np.inf
    fits = dict(result.fits)
    fits[method_name] = replace(fitted, **{field_name: values})

    _assert_rejected_before_output(
        replace(result, fits=fits),
        tmp_path,
        match=rf"fits\.{method_name}\.{field_name}",
    )


def test_tampered_dataset_digest_and_report_are_rejected_before_io(
    tmp_path: Path,
) -> None:
    result = _result()
    bad_digest_dataset = replace(result.dataset, digest="0" * 64)
    _assert_rejected_before_output(
        replace(result, dataset=bad_digest_dataset),
        tmp_path,
        match="dataset digest",
    )

    report = result.reports["Robust-VP-SPCA"]
    reports = dict(result.reports)
    reports["Robust-VP-SPCA"] = replace(
        report,
        clean_reconstruction_error=report.clean_reconstruction_error + 0.01,
    )
    _assert_rejected_before_output(
        replace(result, reports=reports),
        tmp_path / "report-tamper",
        match="does not close over dataset/config/fit",
    )


def test_nested_nonfinite_report_metadata_is_rejected_before_io(
    tmp_path: Path,
) -> None:
    result = _result()
    report = result.reports["PCA"]
    reports = dict(result.reports)
    reports["PCA"] = replace(
        report,
        extra={**report.extra, "nested": {"invalid": [np.nan]}},
    )

    _assert_rejected_before_output(
        replace(result, reports=reports),
        tmp_path,
        match=r"reports\.PCA\.extra\.nested\.invalid\[0\]",
    )


def test_visual_audit_requires_native_robust_sparse_layer(tmp_path: Path) -> None:
    config = VideoExperimentConfig(grid_size=8, n_timesteps=12, max_iter=1)
    pca_only = run_video_experiment(config, methods=("PCA",))

    with pytest.raises(ValueError, match="native outlier layer"):
        write_video_visualizations(
            pca_only,
            tmp_path / "pca-only",
            animation_formats=(),
        )


@pytest.mark.skipif(
    not animation.writers.is_available("pillow"),
    reason="Matplotlib Pillow animation writer is unavailable",
)
def test_requested_gif_is_rendered_and_hashed(tmp_path: Path) -> None:
    result = _result()
    paths = write_video_visualizations(
        result,
        tmp_path / "gif-audit",
        frame_indices=(0, 11),
        animation_formats=("gif",),
        animation_stride=6,
        fps=2,
        dpi=50,
    )

    gif = paths["animation_gif"]
    assert gif.read_bytes().startswith((b"GIF87a", b"GIF89a"))
    payload = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert payload["animation"]["requested_formats"] == ["gif"]
    assert payload["animation"]["writers"] == ["pillow"]
    assert payload["animation"]["frame_indices"] == [0, 6, 11]
    assert payload["artifacts"]["animation_gif"]["sha256"] == _sha256(gif)
    assert payload["artifacts"]["animation_gif"]["availability"] == "local_output"
    assert payload["artifacts"]["animation_gif"]["local_copy_included"] is True
    validate_visualization_manifest_local_copy(payload, gif.parent)


@pytest.mark.skipif(
    not animation.writers.is_available("pillow"),
    reason="Matplotlib Pillow animation writer is unavailable",
)
def test_manifest_projection_marks_omitted_gif_as_external_only(
    tmp_path: Path,
) -> None:
    result = _result()
    complete_paths = write_video_visualizations(
        result,
        tmp_path / "complete-gif-audit",
        frame_indices=(0, 11),
        animation_formats=("gif",),
        animation_stride=6,
        fps=2,
        dpi=50,
    )
    complete = json.loads(complete_paths["manifest"].read_text(encoding="utf-8"))
    validate_visualization_manifest_local_copy(
        complete,
        complete_paths["manifest"].parent,
    )

    partial_dir = tmp_path / "partial-local-copy"
    partial_dir.mkdir()
    for name, record in complete["artifacts"].items():
        if name != "animation_gif":
            shutil.copy2(
                complete_paths["manifest"].parent / record["path"],
                partial_dir / record["path"],
            )
    projected = project_visualization_manifest_for_local_copy(
        complete,
        partial_dir,
    )
    gif_record = projected["artifacts"]["animation_gif"]
    assert gif_record["availability"] == "external_server_only"
    assert gif_record["local_copy_included"] is False
    assert projected["artifact_copy_scope"] == {
        "complete_local_payload": False,
        "external_server_only_artifacts": ["animation_gif"],
    }
    projected_path = partial_dir / "video_visualization_manifest.json"
    projected_path.write_text(
        json.dumps(projected, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_visualization_manifest_local_copy(projected, partial_dir)

    overstated = json.loads(json.dumps(projected))
    overstated["artifacts"]["animation_gif"]["availability"] = "local_copy_included"
    overstated["artifacts"]["animation_gif"]["local_copy_included"] = True
    overstated["artifact_copy_scope"] = {
        "complete_local_payload": True,
        "external_server_only_artifacts": [],
    }
    overstated = _rehash_manifest(overstated)
    with pytest.raises(ValueError, match="declared local artifact is missing"):
        validate_visualization_manifest_local_copy(overstated, partial_dir)


def test_checked_in_video_manifests_are_complete_local_and_server_closures() -> None:
    asset_root = Path(__file__).resolve().parents[1] / "assets" / "evaluation" / "video"
    for seed in (91001, 91007):
        local_dir = asset_root / f"seed_{seed}"
        payload = json.loads(
            (local_dir / "video_visualization_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        validate_visualization_manifest_local_copy(payload, local_dir)
        assert payload["schema_version"] == 2
        assert payload["visual_inference_qualified"] is False
        assert payload["artifact_copy_scope"] == {
            "complete_local_payload": True,
            "external_server_only_artifacts": [],
        }
        gif = payload["artifacts"]["animation_gif"]
        assert gif["availability"] == "local_and_server"
        assert gif["local_copy_included"] is True
        assert (local_dir / gif["path"]).is_file()

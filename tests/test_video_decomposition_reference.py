from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from experiments.video_decomposition_reference import (
    ARXIV_ID,
    ARXIV_VERSION,
    FIGURE5_NAME,
    FIGURE6_NAME,
    FRAME_ERRORS_NAME,
    LEGACY_SOURCE_COMMIT,
    LEGACY_SOURCE_SHA256,
    MANIFEST_NAME,
    REPORT_NAME,
    PaperVideoConfig,
    align_factor_pair_sign_permutation_only,
    generate_figure5_dataset,
    generate_figure6_dataset,
    normalized_full_frame_errors,
    run_paper_video_reproduction,
    save_paper_video_reproduction,
    smoke_config,
    summarize_full_frame_errors,
    validate_paper_video_output_directory,
)
from rspca.datasets import array_sha256, canonical_sha256


def test_defaults_preserve_paper_scale_and_declared_source_lineage() -> None:
    config = PaperVideoConfig()

    assert (config.grid_size, config.n_timesteps, config.total_time) == (
        200,
        300,
        150.0,
    )
    assert config.n_components == 3
    assert config.robust_kappa > 0.0
    assert config.randomized_oversample == 100
    assert config.randomized_power_iters == 4
    assert ARXIV_ID == "1804.00341"
    assert ARXIV_VERSION == 3
    assert LEGACY_SOURCE_COMMIT.startswith("e01c02f")


def test_generators_use_paper_pixels_by_snapshots_orientation_and_are_deterministic() -> (
    None
):
    config = smoke_config()
    figure5 = generate_figure5_dataset(config)
    figure6 = generate_figure6_dataset(config)
    figure6_again = generate_figure6_dataset(config)

    expected_shape = (config.grid_size**2, config.n_timesteps)
    assert figure5.clean.shape == expected_shape
    assert figure6.clean.shape == expected_shape
    assert figure6.observed.shape == expected_shape
    assert figure5.spatial_modes.shape == (config.grid_size**2, 3)
    assert figure5.temporal_scores.shape == (config.n_timesteps, 3)
    np.testing.assert_allclose(figure5.clean.mean(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        figure5.clean,
        figure5.spatial_modes @ figure5.temporal_scores.T,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        np.linalg.norm(figure5.temporal_scores, axis=0),
        np.ones(3),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        figure6.observed - figure6.clean,
        figure6.corruption,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_array_equal(
        figure6.corruption_support, figure6.corruption != 0.0
    )
    np.testing.assert_array_equal(figure6.observed, figure6_again.observed)


def test_alignment_changes_only_column_order_and_sign_not_amplitude() -> None:
    rng = np.random.default_rng(7)
    truth_modes = rng.normal(size=(20, 3))
    truth_dynamics = rng.normal(size=(15, 3))
    estimate_modes = 5.0 * truth_modes
    estimate_dynamics = 7.0 * truth_dynamics

    aligned_modes, aligned_dynamics, permutation, signs = (
        align_factor_pair_sign_permutation_only(
            truth_modes, estimate_modes, estimate_dynamics
        )
    )

    assert permutation == (0, 1, 2)
    assert signs == (1, 1, 1)
    np.testing.assert_allclose(aligned_modes, 5.0 * truth_modes)
    np.testing.assert_allclose(aligned_dynamics, 7.0 * truth_dynamics)
    assert not np.allclose(aligned_modes, truth_modes)
    assert not np.allclose(aligned_dynamics, truth_dynamics)


def test_full_frame_summary_uses_every_column_and_keeps_worst_frame() -> None:
    clean = np.ones((2, 4), dtype=np.float64)
    reconstruction = clean + np.array([0.0, 1.0, -2.0, 0.5])[None, :]
    time = np.array([0.0, 0.5, 1.0, 1.5])

    errors = normalized_full_frame_errors(clean, reconstruction)
    summary = summarize_full_frame_errors(errors, time)

    np.testing.assert_allclose(errors, [0.0, 1.0, 2.0, 0.5])
    assert summary["n_frames"] == 4
    assert summary["median"] == pytest.approx(0.75)
    assert summary["p95"] == pytest.approx(np.percentile(errors, 95.0))
    assert summary["max"] == pytest.approx(2.0)
    assert summary["worst_frame_index"] == 2
    assert summary["worst_frame_time"] == pytest.approx(1.0)


def test_smoke_output_is_strict_native_fixed_kappa_and_exactly_closed(
    tmp_path,
) -> None:
    config = smoke_config()
    result = run_paper_video_reproduction(config)
    output = tmp_path / "paper-video-smoke"

    paths = save_paper_video_reproduction(result, output)
    manifest = validate_paper_video_output_directory(output)

    assert set(paths) == {
        "figure5",
        "figure6",
        "full_frame_errors",
        "report",
        "manifest",
    }
    assert {path.name for path in output.iterdir()} == {
        FIGURE5_NAME,
        FIGURE6_NAME,
        FRAME_ERRORS_NAME,
        REPORT_NAME,
        MANIFEST_NAME,
    }
    assert paths["figure5"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["figure6"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert manifest["self_excluding"] is True

    payload = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert payload["paper_scale"] is False
    assert payload["data"]["orientation"] == "spatial_pixels_x_time_snapshots"
    assert "distinct" in payload["data"]["protocol_identity"]
    assert payload["protocol"]["spatial_mode_definition"] == "Phi = X @ B"
    assert payload["protocol"]["post_hoc_amplitude_fit"] is False
    assert payload["protocol"]["outlier_post_threshold"] is False
    assert payload["protocol"]["robust_kappa_policy"] == (
        "explicit_fixed_in_normalized_units"
    )
    robust_extra = payload["fit_metrics"]["figure6"]["Robust-VP-SPCA"][
        "extra"
    ]
    assert robust_extra["kappa_normalized"] == pytest.approx(config.robust_kappa)
    assert robust_extra["kappa_policy"] == "explicit_fixed"
    robust_outliers = result.figure6_fits["Robust-VP-SPCA"].outliers
    assert robust_outliers is not None
    assert payload["display"]["figure6"]["outlier_array_sha256"] == (
        array_sha256(robust_outliers)
    )
    assert isinstance(payload["strict_qualification"]["overall"], bool)
    assert payload["strict_qualification"]["failure_policy"].startswith(
        "nonconverged"
    )
    # A source archive has no Git objects. Absence must be reported as None,
    # never as verified; an available historical blob must match its anchor.
    expected_verification = {}
    root = Path(__file__).resolve().parents[1]
    for relative, expected_hash in LEGACY_SOURCE_SHA256.items():
        historical = subprocess.run(
            ["git", "show", f"{LEGACY_SOURCE_COMMIT}:{relative}"],
            cwd=root, capture_output=True, timeout=10, check=False,
        )
        if historical.returncode == 0:
            assert sha256(historical.stdout).hexdigest() == expected_hash
            expected_verification[relative] = True
        else:
            expected_verification[relative] = None
    assert payload["provenance"]["legacy_implementation"][
        "verified_from_local_git_objects"
    ] == expected_verification
    loaded_modules = payload["provenance"]["loaded_repo_modules"]
    assert loaded_modules["src/rspca/__init__.py"]["role"] == (
        "package_initialization"
    )
    assert any(
        record["role"] == "package_initialization_only"
        for record in loaded_modules.values()
    )
    assert payload["provenance"]["package_eager_imports_bound"] is True
    randomized_extra = payload["fit_metrics"]["figure5"]["RND-VP-SPCA"][
        "extra"
    ]
    assert set(randomized_extra["timings"]) >= {
        "range_finder",
        "compress",
        "optimization",
        "diagnostics",
    }
    assert randomized_extra["sketch_residual_relative"] >= 0.0
    assert np.isfinite(randomized_extra["full_stationarity"])
    for method in ("PCA", "VP-SPCA", "RND-VP-SPCA"):
        recovery = payload["fit_metrics"]["figure5"][method][
            "mode_and_dynamics_recovery"
        ]
        assert recovery["amplitude_refit"] is False
        assert len(recovery["spatial_mode_cosine_per_component"]) == 3
        assert len(recovery["temporal_dynamics_cosine_per_component"]) == 3
    effect_gate = payload["paper_result_effect_gate"]
    assert effect_gate["global_cosine_alone_is_sufficient"] is False
    assert set(effect_gate["comparisons"]) == {"VP-SPCA", "RND-VP-SPCA"}
    for method in ("PCA", "VP-SPCA", "RND-VP-SPCA"):
        localization = effect_gate["methods"][method]
        assert len(localization["off_support_energy_fraction_per_component"]) == 3
        assert len(localization["cross_talk_energy_fraction_per_component"]) == 3

    with paths["full_frame_errors"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5 * config.n_timesteps
    assert {row["panel"] for row in rows} == {"figure5", "figure6"}
    assert {row["method"] for row in rows} == {
        "PCA",
        "VP-SPCA",
        "RND-VP-SPCA",
        "Corrupted-PCA",
        "Robust-VP-SPCA",
    }

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_paper_video_reproduction(result, output)


def _saved_smoke_copy(tmp_path):
    source = tmp_path / "source-output"
    save_paper_video_reproduction(run_paper_video_reproduction(smoke_config()), source)
    target = tmp_path / "tampered-output"
    shutil.copytree(source, target)
    return target


def test_validator_rejects_deleted_payload_even_after_manifest_resign(tmp_path) -> None:
    output = _saved_smoke_copy(tmp_path)
    (output / FIGURE5_NAME).unlink()
    manifest_path = output / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop(FIGURE5_NAME)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="exactly the four required payloads"):
        validate_paper_video_output_directory(output)


@pytest.mark.parametrize("attack", ["extra_directory", "payload_symlink"])
def test_validator_rejects_nonregular_or_symlink_entries(tmp_path, attack: str) -> None:
    output = _saved_smoke_copy(tmp_path)
    if attack == "extra_directory":
        (output / "untracked-subdirectory").mkdir()
    else:
        figure = output / FIGURE5_NAME
        figure.unlink()
        figure.symlink_to(output / FIGURE6_NAME)

    with pytest.raises(ValueError, match="non-regular"):
        validate_paper_video_output_directory(output)

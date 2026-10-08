from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_pca_loading_all_autosomes as allauto
from experiments import genomics_pca_loading_amr_whitening as amr
from experiments import genomics_pca_loading_grid as base
from rspca.datasets import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def _small_payload(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(amr, "BOOTSTRAP_RESAMPLES", 2_000)
    data = allauto._synthetic_smoke_data()
    return amr.evaluate(
        data,
        amr.WhiteningConfig(
            seeds=(73,),
            analysis_mode="synthetic_smoke",
            analysis_status="engineering_smoke_only",
            snp_chunk_size=512,
        ),
        input_identity=allauto._synthetic_input_identity(data),
        source_identity=amr._source_identity(REPO_ROOT),
    )


def test_formal_protocol_is_fixed_and_fresh() -> None:
    amr._validate_config(amr.WhiteningConfig())
    assert amr.FORMAL_SEEDS == tuple(range(731001, 731021))
    assert amr.COMPONENT_GRID == (4, 6, 8)
    assert amr.PANEL_FRACTIONS == (0.05, 0.10, 0.25, 0.50, None)
    assert amr.DISTANCE_MODES == (
        "raw_euclidean",
        "train_singular_whitened_euclidean",
    )
    with pytest.raises(ValueError, match="frozen fresh bank"):
        amr._validate_config(
            amr.WhiteningConfig(
                seeds=(71001,),
                analysis_mode="fresh_registered_shard",
            )
        )
    with pytest.raises(ValueError, match="exact fresh seed cover"):
        amr._validate_config(
            amr.WhiteningConfig(
                seeds=(731001,),
                analysis_mode="fresh_fixed_protocol",
            )
        )


def test_whitening_is_train_only_unit_variance_geometry() -> None:
    train_scores = np.asarray(
        [
            [-10.0, -1.0],
            [-8.0, 1.0],
            [-6.0, -1.0],
            [-4.0, 1.0],
            [-2.0, -1.0],
            [2.0, 1.0],
            [4.0, -1.0],
            [6.0, 1.0],
            [8.0, -1.0],
            [10.0, 1.0],
        ]
    )
    singular = np.sqrt(np.sum(train_scores * train_scores, axis=0))
    fit = allauto.ChunkedPCAFit(
        mean_dosage=np.zeros(2),
        scale=np.ones(2),
        retained_mask=np.ones(2, dtype=bool),
        retained_original=np.arange(2, dtype=np.int64),
        loadings=np.eye(2),
        singular_values=singular,
        explained_variance_ratio=singular**2 / np.sum(singular**2),
        train_scores=train_scores,
        test_scores=np.asarray(
            [[-9.0, 0.0], [-5.0, 0.0], [0.0, 0.0], [5.0, 0.0], [9.0, 0.0]]
        ),
        sample_gram_sha256="0" * 64,
    )
    labels = np.repeat(np.asarray(amr.EXPECTED_POPULATIONS), 2)
    truth = np.asarray(amr.EXPECTED_POPULATIONS)
    raw = amr._score_cell(
        fit,
        n_components=2,
        train_labels=labels,
        test_labels=truth,
        distance_mode="raw_euclidean",
    )
    white = amr._score_cell(
        fit,
        n_components=2,
        train_labels=labels,
        test_labels=truth,
        distance_mode="train_singular_whitened_euclidean",
    )
    assert np.array_equal(raw[-1], np.ones(2))
    assert np.array_equal(white[-1], singular / np.sqrt(9.0))
    whitened_train = train_scores / white[-1][None, :]
    assert np.allclose(np.std(whitened_train, axis=0, ddof=1), 1.0)


def test_smoke_grid_closes_and_raw_white_share_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _small_payload(monkeypatch)
    validation = amr.validate_result_payload(payload)
    assert validation["valid"]
    assert validation["records"] == 30
    assert validation["predictions"] == 600
    assert payload["analysis_policy"]["best_configuration_selected"] is False
    assert (
        payload["method_scope"]["robust_vp_randomized_federated_lpca_alpca_fisher_run"]
        is False
    )
    keyed = {
        (
            row["seed"],
            row["n_components"],
            row["panel_label"],
            row["distance_mode"],
        ): row
        for row in payload["records"]
    }
    for rank in amr.COMPONENT_GRID:
        for fraction in amr.PANEL_FRACTIONS:
            panel = amr._panel_label(fraction)
            raw = keyed[(73, rank, panel, amr.DISTANCE_MODES[0])]
            white = keyed[(73, rank, panel, amr.DISTANCE_MODES[1])]
            for field in (
                "selected_original_indices_sha256",
                "sample_gram_sha256",
                "loadings_prefix_sha256",
                "singular_values",
                "train_indices_sha256",
                "test_indices_sha256",
            ):
                assert raw[field] == white[field]
            expected = np.asarray(white["singular_values"]) / np.sqrt(
                white["train_sample_count"] - 1
            )
            assert np.array_equal(np.asarray(white["distance_scale"]), expected)


def test_result_validator_rejects_prediction_and_scale_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _small_payload(monkeypatch)
    changed = copy.deepcopy(payload)
    changed["predictions"][0]["prediction"] = "EUR"
    changed["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "analysis_sha256"}
    )
    with pytest.raises(
        ValueError,
        match="correct flag mismatch|prediction hash mismatch|metric mismatch",
    ):
        amr.validate_result_payload(changed)

    changed = copy.deepcopy(payload)
    changed["records"][0]["distance_scale"][0] = 999.0
    changed["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "analysis_sha256"}
    )
    with pytest.raises(ValueError, match="distance scale"):
        amr.validate_result_payload(changed)

    changed = copy.deepcopy(payload)
    changed["predictions"][0]["sample_id"] = "CORRUPTED"
    changed["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "analysis_sha256"}
    )
    with pytest.raises(
        ValueError, match="sample ID identity|raw/whitened prediction identity"
    ):
        amr.validate_result_payload(changed)


def test_writer_and_external_anchor_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _small_payload(monkeypatch)
    output = tmp_path / "package"
    written = amr.write_output_package(payload, output)
    assert written["package_self_consistency_verified"]
    validated = amr.validate_output_package(
        output,
        repo_root=REPO_ROOT,
        expected_analysis_sha256=written["analysis_sha256"],
        expected_package_anchor_sha256=written["package_anchor_sha256"],
    )
    assert validated["source_bytes_verified"]
    assert {path.name for path in output.iterdir()} == set(amr.ARTIFACT_FILENAMES) | {
        amr.MANIFEST_FILENAME
    }
    with pytest.raises(FileExistsError):
        amr.write_output_package(payload, output)


def test_package_rejects_artifact_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _small_payload(monkeypatch)
    output = tmp_path / "package"
    amr.write_output_package(payload, output)
    (output / amr.PAIRED_FILENAME).write_text("forged\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact receipt mismatch"):
        amr.validate_output_package(output)


def test_paired_summary_uses_seed_as_unit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(amr, "BOOTSTRAP_RESAMPLES", 1_000)
    records = []
    classes = np.asarray(amr.EXPECTED_POPULATIONS)
    truth = np.repeat(classes, 2)
    for seed, raw_correct, white_correct in ((1, 7, 9), (2, 8, 8)):
        for mode, correct in (
            (amr.DISTANCE_MODES[0], raw_correct),
            (amr.DISTANCE_MODES[1], white_correct),
        ):
            observed = truth.copy()
            for index in range(correct, truth.size):
                observed[index] = classes[(list(classes).index(truth[index]) + 1) % 5]
            metrics = base._classification_metrics(truth, observed, classes)
            records.append(
                {
                    "seed": seed,
                    "n_components": 4,
                    "panel_label": "5pct",
                    "distance_mode": mode,
                    **metrics,
                }
            )
    # Pad other cells out of this focused helper; the function summarizes only present cells.
    result = amr._paired_summaries(records)
    assert len(result) == 1
    delta = result[0]["deltas"]["balanced_accuracy"]
    assert delta["positive_seeds"] == 1
    assert delta["equal_seeds"] == 1
    assert delta["negative_seeds"] == 0


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        amr._strict_json_load(duplicate)
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        amr._strict_json_load(nonfinite)

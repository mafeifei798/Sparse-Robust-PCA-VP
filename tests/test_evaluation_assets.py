from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image
import pytest

from experiments.video_spca_visualize import (
    validate_visualization_manifest_local_copy,
)
from rspca.datasets import canonical_sha256


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "assets" / "evaluation"
LEGACY_GENOMICS = ROOT / "assets" / "legacy" / "genomics"
RANDOMIZED_EVIDENCE = EVALUATION / "randomized" / "autodl_wide_957x5000.json"

VIDEO_LOCAL_ARTIFACTS = {
    "animation_gif": "video_decomposition.gif",
    "corruption_layers": "video_corruption_layers.png",
    "fixed_frames": "video_decomposition_fixed_frames.png",
    "ownership_receipt": "video_visualization_ownership.json",
    "per_frame_error_csv": "video_per_frame_error.csv",
    "per_frame_error_plot": "video_per_frame_error.png",
}
VIDEO_EXPECTED = {
    91001: {
        "manifest_sha256": (
            "7354a65ea9e3b23b4e4fbe610bacfa7e282334ffabf7a6c9c533ea0a6a725cc2"
        ),
        "manifest_file_sha256": (
            "6e064d1e7a954d1a3e0654d6ee5e1b3ef89e2785f9fe5c27576cc512b406c795"
        ),
        "gif_sha256": "527be88827b86090f0072cf35ad35892aaa0788dc483f6df4c3a9ffac0ec9df2",
        "gif_first_frame_sha256": (
            "89d6a7a20c466cf32dd16ba57817955fe3de5911a5d8e82b275849989758527c"
        ),
        "gif_last_frame_sha256": (
            "0698d4baa4a77122c711ecc256c1e153b112bcf4bae2f441cdd2ea149cf06060"
        ),
    },
    91007: {
        "manifest_sha256": (
            "5267f16dd06689fb820fccc30bc7b94e029d429a23037a96e2ec85c83551c2a3"
        ),
        "manifest_file_sha256": (
            "78115af9c0f1fde91b46d35d30dd6178644f63c81be121dfa3d30c37ac9ce4d0"
        ),
        "gif_sha256": "e52b2fc48da3cb57a6b9d58d2705725a715c1bc713e0f0ee2c39bd5c35305c5b",
        "gif_first_frame_sha256": (
            "49eb28ef143aaeb034ca5dd87356675f2fdd192f52d1521ce962d5d1a1368fe1"
        ),
        "gif_last_frame_sha256": (
            "3fe494f651cfb91b9b9f1fe6a586814351b4c295bf03633f27cc7d0e06b94617"
        ),
    },
}

GENOMICS_EXPECTED_SHA256 = {
    "population_diagnostics.png": (
        "ed9a0cb649ca089cf0779eeefcb498c0ab2a16f5fdb43c3c91dfeab2b5441f9d"
    ),
    "confusion_matrix.csv": (
        "2f7a40c4a4533618d0c4567443ed62c59dd86843b372b0ca288d139b481d7aad"
    ),
    "per_population_metrics.csv": (
        "137066b691345d97f7caf21b368367be1bf6244a535bad4dfe4eb98d7b59b74e"
    ),
    "method_summaries.csv": (
        "e3ac1daa86b1745b4ab90e7c836fea2d2804a5a2b196465ca7c2b95701b4d01d"
    ),
}
SPARSE_GENOMICS_EXPECTED_SHA256 = {
    "population_diagnostics.png": (
        "4d1b22f3c75e0c9f44dad24d9500e0b9518de5defc95b1e399212afe450f8f23"
    ),
    "confusion_matrix.csv": (
        "b731e11847b35e012f11ccb7c9d781db6263f812571e147851c53cf32000cd7e"
    ),
    "method_summaries.csv": (
        "7ff2ba6a1cc3fecf6b25f9c22c88becc0ad8ca31b1031658973c8901f4917e24"
    ),
    "per_population_metrics.csv": (
        "eabc1b694f457763fa093e556d3eec4085c46c78888143da2edea6f41f4fca99"
    ),
    "provenance.json": (
        "99fd5c1bb163a0e0f1e9bd13ae00763aaa78ab2027f0bb8491a136e39435d53b"
    ),
    "manifest.json": (
        "bc779484e95e9821052aa43e9ff421a46c402f4a749de3b862a48c86ff13bcb0"
    ),
}
SPARSE_GENOMICS_MANIFEST_SHA256 = (
    "dd2474d61ddc5a137dd85cf86e61c8053f392ba18950bd968ec4a67283d15441"
)
LEGACY_GENOMICS_EXPECTED_SHA256 = {
    "gene_957x14389_all_variants.png": (
        "6dade3589b650debfb2aa1163f16c581884aa2ebc85f74b555ba58baed8ccebf"
    ),
    "gene_957x14389_top200.png": (
        "3fb85fd1049e12b97b38b0fc7a4dcce20cf418148983e3de95ae94ff7f8bc04b"
    ),
    "gene_2504x80000_all_variants.png": (
        "587a17b169837b15e734b61dc54dd5eb3673606ab39b3aa4d2010e299a1d7afa"
    ),
    "gene_2504x80000_top200.png": (
        "0939f7540e33d3fcc7879d2f2f03344a715d47f10f68965d55b76285f139ccd4"
    ),
}
POPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
EXPECTED_RECALL = {
    "AFR": 0.98875,
    "AMR": 0.67125,
    "EAS": 1.0,
    "EUR": 0.96625,
    "SAS": 0.99375,
}
EXPECTED_CONFUSION = {
    "AFR": [791, 2, 0, 7, 0],
    "AMR": [11, 537, 0, 252, 0],
    "EAS": [0, 0, 800, 0, 0],
    "EUR": [0, 27, 0, 773, 0],
    "SAS": [0, 3, 0, 2, 795],
}
SPARSE_EXPECTED_RECALL = {
    "vp": {
        "AFR": 0.98875,
        "AMR": 0.66875,
        "EAS": 1.0,
        "EUR": 0.9625,
        "SAS": 0.99,
    },
    "randomized": {
        "AFR": 0.98875,
        "AMR": 0.6675,
        "EAS": 1.0,
        "EUR": 0.9625,
        "SAS": 0.9925,
    },
    "robust": {
        "AFR": 0.98625,
        "AMR": 0.655,
        "EAS": 1.0,
        "EUR": 0.9475,
        "SAS": 0.9825,
    },
}
SPARSE_EXPECTED_CONFUSION = {
    "vp": {
        "AFR": [791, 2, 0, 7, 0],
        "AMR": [7, 535, 0, 258, 0],
        "EAS": [0, 0, 800, 0, 0],
        "EUR": [0, 30, 0, 770, 0],
        "SAS": [0, 5, 0, 3, 792],
    },
    "randomized": {
        "AFR": [791, 2, 0, 7, 0],
        "AMR": [6, 534, 0, 260, 0],
        "EAS": [0, 0, 800, 0, 0],
        "EUR": [0, 30, 0, 770, 0],
        "SAS": [0, 4, 0, 2, 794],
    },
    "robust": {
        "AFR": [789, 3, 0, 8, 0],
        "AMR": [8, 524, 0, 268, 0],
        "EAS": [0, 0, 800, 0, 0],
        "EUR": [0, 42, 0, 758, 0],
        "SAS": [0, 5, 1, 8, 786],
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def _assert_png(path: Path) -> None:
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def _assert_nested_finite(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_nested_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_nested_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


@pytest.mark.parametrize("seed", sorted(VIDEO_EXPECTED))
def test_current_video_assets_match_their_strict_manifest(seed: int) -> None:
    directory = EVALUATION / "video" / f"seed_{seed}"
    manifest_path = directory / "video_visualization_manifest.json"
    expected_files = {
        "video_visualization_manifest.json",
        *VIDEO_LOCAL_ARTIFACTS.values(),
    }
    actual_files = {path.name for path in directory.iterdir() if path.is_file()}
    assert actual_files == expected_files

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["config"]["seed"] == seed
    assert payload["config"]["n_timesteps"] == 180
    assert payload["methods"] == [
        "PCA",
        "VP-SPCA",
        "RND-VP-SPCA",
        "Robust-VP-SPCA",
    ]

    manifest_body = dict(payload)
    recorded_manifest_sha256 = manifest_body.pop("manifest_sha256")
    expected = VIDEO_EXPECTED[seed]
    assert recorded_manifest_sha256 == expected["manifest_sha256"]
    assert canonical_sha256(manifest_body) == recorded_manifest_sha256
    assert _sha256(manifest_path) == expected["manifest_file_sha256"]
    validate_visualization_manifest_local_copy(payload, directory)

    diagnostics = payload["method_diagnostics"]
    assert set(diagnostics) == set(payload["methods"])
    diagnostic_fields = {
        "converged",
        "status",
        "n_iter",
        "final_stationarity",
        "corruption_detection",
        "corruption_detection_at_budget",
    }
    for method in payload["methods"]:
        assert set(diagnostics[method]) == diagnostic_fields
        assert isinstance(diagnostics[method]["converged"], bool)
        assert diagnostics[method]["status"]
        assert diagnostics[method]["n_iter"] > 0
    assert diagnostics["PCA"]["converged"] is True
    assert diagnostics["PCA"]["final_stationarity"] is None
    for method in ("VP-SPCA", "RND-VP-SPCA", "Robust-VP-SPCA"):
        assert diagnostics[method]["converged"] is False
        assert math.isfinite(diagnostics[method]["final_stationarity"])
    detection_fields = {
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
        "true_negatives",
        "roc_auc",
        "average_precision",
    }
    robust = diagnostics["Robust-VP-SPCA"]
    assert set(robust["corruption_detection"]) == detection_fields
    assert set(robust["corruption_detection_at_budget"]) == {
        *detection_fields,
        "budget_fraction",
        "budget_top_k",
    }
    assert payload["visual_inference_qualified"] is False
    assert payload["visual_inference_qualification"] == {
        "all_methods_converged": False,
        "fixed_budget_descriptive_only": True,
        "multi_seed_inference_required": True,
        "nonconverged_methods": [
            "VP-SPCA",
            "RND-VP-SPCA",
            "Robust-VP-SPCA",
        ],
        "reason": "single_seed_visual_diagnostic_only",
    }
    assert payload["artifact_copy_scope"] == {
        "complete_local_payload": True,
        "external_server_only_artifacts": [],
    }

    assert set(payload["artifacts"]) == set(VIDEO_LOCAL_ARTIFACTS)
    for artifact_name, filename in VIDEO_LOCAL_ARTIFACTS.items():
        artifact = payload["artifacts"][artifact_name]
        assert artifact["path"] == filename
        path = directory / filename
        assert path.is_file()
        assert artifact["size_bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha256(path)
        assert artifact["availability"] == "local_and_server"
        assert artifact["local_copy_included"] is True

    for filename in VIDEO_LOCAL_ARTIFACTS.values():
        if filename.endswith(".png"):
            _assert_png(directory / filename)

    ownership = json.loads(
        (directory / VIDEO_LOCAL_ARTIFACTS["ownership_receipt"]).read_text(
            encoding="utf-8"
        )
    )
    assert ownership["schema_version"] == 1
    assert ownership["failure_policy"] == "preserve_for_manual_review"
    assert re.fullmatch(r"[0-9a-f]{32}", ownership["ownership_token"])

    gif = payload["artifacts"]["animation_gif"]
    assert gif["path"] == "video_decomposition.gif"
    assert gif["size_bytes"] > 4_000_000
    assert gif["sha256"] == expected["gif_sha256"]
    assert gif["availability"] == "local_and_server"
    assert gif["local_copy_included"] is True
    gif_path = directory / gif["path"]
    assert gif_path.is_file()
    with Image.open(gif_path) as animation:
        assert animation.n_frames == 61
        assert animation.size == (1144, 605)
        first_frame_sha256 = hashlib.sha256(
            animation.convert("RGB").tobytes()
        ).hexdigest()
        animation.seek(animation.n_frames - 1)
        last_frame_sha256 = hashlib.sha256(
            animation.convert("RGB").tobytes()
        ).hexdigest()
    assert first_frame_sha256 == expected["gif_first_frame_sha256"]
    assert last_frame_sha256 == expected["gif_last_frame_sha256"]
    assert first_frame_sha256 != last_frame_sha256
    assert payload["animation"]["requested_formats"] == ["gif"]
    assert payload["animation"]["writers"] == ["pillow"]
    assert len(payload["animation"]["frame_indices"]) == 61
    assert payload["animation"]["frame_indices"][:1] == [0]
    assert payload["animation"]["frame_indices"][-1:] == [179]


@pytest.mark.parametrize("seed", sorted(VIDEO_EXPECTED))
def test_current_video_error_csv_is_a_complete_180_frame_curve(seed: int) -> None:
    directory = EVALUATION / "video" / f"seed_{seed}"
    payload = json.loads(
        (directory / "video_visualization_manifest.json").read_text(encoding="utf-8")
    )
    fields, rows = _read_csv(directory / "video_per_frame_error.csv")
    assert fields == ["frame_index", "time", *payload["methods"]]
    assert len(rows) == payload["config"]["n_timesteps"] == 180
    assert [int(row["frame_index"]) for row in rows] == list(range(180))

    times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    assert np.all(np.isfinite(times))
    assert np.all(np.diff(times) > 0.0)
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(payload["config"]["total_time"])

    for method in payload["methods"]:
        values = np.asarray([float(row[method]) for row in rows], dtype=np.float64)
        assert np.all(np.isfinite(values))
        assert np.all(values >= 0.0)
        summary = payload["frame_error_summaries"][method]
        worst = int(np.argmax(values))
        assert math.sqrt(float(np.mean(values**2))) == pytest.approx(
            summary["global_clean_reconstruction_error"]
        )
        assert float(np.median(values)) == pytest.approx(summary["median"])
        assert float(np.percentile(values, 95)) == pytest.approx(summary["p95"])
        assert float(values[worst]) == pytest.approx(summary["maximum"])
        assert worst == summary["worst_frame_index"]
        assert float(times[worst]) == pytest.approx(summary["worst_frame_time"])


def test_current_genomics_pca_assets_have_fixed_content_and_schema() -> None:
    directory = EVALUATION / "genomics" / "pca_20seeds"
    assert {path.name for path in directory.iterdir() if path.is_file()} == {
        "README.md",
        *GENOMICS_EXPECTED_SHA256,
    }
    for filename, expected_sha256 in GENOMICS_EXPECTED_SHA256.items():
        path = directory / filename
        assert path.is_file()
        assert _sha256(path) == expected_sha256
    _assert_png(directory / "population_diagnostics.png")

    fields, summaries = _read_csv(directory / "method_summaries.csv")
    assert fields == [
        "alpha_selection_qualified_splits",
        "final_loading_support_coordinate_system",
        "final_loading_support_threshold",
        "held_out_decisions",
        "max_final_selected_snp_count",
        "mean_final_loading_entry_sparsity",
        "mean_final_selected_snp_count",
        "mean_split_accuracy",
        "mean_split_balanced_accuracy",
        "mean_split_macro_f1",
        "method",
        "min_final_selected_snp_count",
        "outer_splits",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_macro_f1",
        "solver_converged_splits",
    ]
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["method"] == "pca"
    assert int(summary["outer_splits"]) == 20
    assert int(summary["held_out_decisions"]) == 4_000
    assert int(summary["solver_converged_splits"]) == 20
    assert int(summary["alpha_selection_qualified_splits"]) == 20
    assert float(summary["mean_split_balanced_accuracy"]) == pytest.approx(0.924)
    assert float(summary["pooled_balanced_accuracy"]) == pytest.approx(0.924)

    fields, metrics = _read_csv(directory / "per_population_metrics.csv")
    assert fields == [
        "f1",
        "max_split_recall",
        "mean_split_recall",
        "method",
        "min_split_recall",
        "outer_splits",
        "population",
        "precision",
        "predicted_count",
        "recall",
        "std_split_recall",
        "support",
        "true_positive",
    ]
    assert [row["population"] for row in metrics] == POPULATIONS
    for row in metrics:
        population = row["population"]
        assert row["method"] == "pca"
        assert int(row["outer_splits"]) == 20
        assert int(row["support"]) == 800
        assert float(row["recall"]) == pytest.approx(EXPECTED_RECALL[population])
        assert (
            int(row["true_positive"])
            == EXPECTED_CONFUSION[population][POPULATIONS.index(population)]
        )


def test_current_genomics_confusion_semantics_match_the_reported_recalls() -> None:
    directory = EVALUATION / "genomics" / "pca_20seeds"
    fields, rows = _read_csv(directory / "confusion_matrix.csv")
    assert fields == [
        "count",
        "method",
        "predicted_population",
        "true_population",
        "true_row_fraction",
    ]
    assert len(rows) == 25

    by_pair = {
        (row["true_population"], row["predicted_population"]): row for row in rows
    }
    assert set(by_pair) == {
        (truth, predicted) for truth in POPULATIONS for predicted in POPULATIONS
    }
    assert {row["method"] for row in rows} == {"pca"}

    total = 0
    for truth in POPULATIONS:
        counts = [
            int(by_pair[(truth, predicted)]["count"]) for predicted in POPULATIONS
        ]
        assert counts == EXPECTED_CONFUSION[truth]
        assert sum(counts) == 800
        total += sum(counts)
        for predicted, count in zip(POPULATIONS, counts, strict=True):
            fraction = float(by_pair[(truth, predicted)]["true_row_fraction"])
            assert fraction == pytest.approx(count / 800)
        diagonal = counts[POPULATIONS.index(truth)] / 800
        assert diagonal == pytest.approx(EXPECTED_RECALL[truth])
    assert total == 4_000
    assert int(by_pair[("AMR", "EUR")]["count"]) == 252
    assert float(by_pair[("AMR", "EUR")]["true_row_fraction"]) == pytest.approx(0.315)


def test_current_genomics_sparse_assets_are_hash_bound_and_fixed_budget_only() -> None:
    directory = EVALUATION / "genomics" / "sparse_20seeds"
    assert {path.name for path in directory.iterdir() if path.is_file()} == {
        "README.md",
        *SPARSE_GENOMICS_EXPECTED_SHA256,
    }
    for filename, expected_sha256 in SPARSE_GENOMICS_EXPECTED_SHA256.items():
        path = directory / filename
        assert path.is_file()
        assert _sha256(path) == expected_sha256

    image_path = directory / "population_diagnostics.png"
    _assert_png(image_path)
    with Image.open(image_path) as figure:
        assert figure.size == (2982, 1782)
        assert figure.mode == "RGBA"

    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _assert_nested_finite(manifest)
    assert manifest["schema_version"] == 1
    assert manifest["artifact_type"] == (
        "genomics_sparse_population_diagnostics_assets"
    )
    manifest_body = dict(manifest)
    recorded_manifest_sha256 = manifest_body.pop("manifest_sha256")
    assert recorded_manifest_sha256 == SPARSE_GENOMICS_MANIFEST_SHA256
    assert canonical_sha256(manifest_body) == recorded_manifest_sha256
    assert manifest["manifest_sha256_contract"] == (
        "sha256 of compact sorted UTF-8 JSON after removing only manifest_sha256"
    )

    listed_files = {entry["path"]: entry for entry in manifest["files"]}
    assert set(listed_files) == {
        "population_diagnostics.png",
        "confusion_matrix.csv",
        "method_summaries.csv",
        "per_population_metrics.csv",
        "provenance.json",
    }
    for filename, entry in listed_files.items():
        path = directory / filename
        assert entry["sha256"] == _sha256(path)
        assert entry["bytes"] == path.stat().st_size
        assert entry["role"]

    provenance = json.loads(
        (directory / "provenance.json").read_text(encoding="utf-8")
    )
    _assert_nested_finite(provenance)
    assert provenance["schema_version"] == 1
    assert provenance["execution"]["methods"] == ["vp", "randomized", "robust"]
    assert provenance["execution"]["outer_seeds"] == list(range(2025, 2045))
    assert provenance["execution"]["records"] == 60
    assert provenance["figure_policy"]["best_seed_selection"] is False
    assert provenance["figure_policy"]["embedding_seed"] == 2025
    assert provenance["figure_policy"]["solver_rerun_for_figure"] is False
    assert provenance["figure_policy"]["two_render_byte_identity_verified"] is True
    assert provenance["interpretation"]["label_is_race"] is False
    assert provenance["interpretation"]["independent_external_cohort"] is False
    assert provenance["interpretation"]["label_semantics"] == (
        "1000_Genomes_metadata_super_population"
    )
    assert provenance["qualification"]["claim_boundary"] == (
        "fixed_budget_descriptive_only"
    )
    for field in ("solver_converged_splits", "alpha_selection_qualified_splits"):
        assert provenance["qualification"][field] == {
            "vp": 0,
            "randomized": 0,
            "robust": 0,
        }
    assert provenance["formal_source_binding"]["formal_results_sha256"] == (
        "e5a20a09da4b525b8d1f337033bcf74055a22b3bed6e221ed3aeb213b92f516b"
    )
    assert provenance["formal_source_binding"]["external_trust_anchor_sha256"] == (
        "b561bef832c9376c80cb00cc59c2de722ade0aed2625fb1320630e4a6ceaf14b"
    )
    assert provenance["large_row_level_artifacts"]["included_in_tracked_assets"] is False

    fields, summaries = _read_csv(directory / "method_summaries.csv")
    assert fields == [
        "alpha_selection_qualified_splits",
        "final_loading_support_coordinate_system",
        "final_loading_support_threshold",
        "held_out_decisions",
        "max_final_selected_snp_count",
        "mean_final_loading_entry_sparsity",
        "mean_final_selected_snp_count",
        "mean_split_accuracy",
        "mean_split_balanced_accuracy",
        "mean_split_macro_f1",
        "method",
        "min_final_selected_snp_count",
        "outer_splits",
        "pooled_accuracy",
        "pooled_balanced_accuracy",
        "pooled_macro_f1",
        "solver_converged_splits",
    ]
    expected_summaries = {
        "vp": (0.922, 0.9203462516788037, 0.2513265306122449, 682.3, 635, 686),
        "randomized": (
            0.92225,
            0.9205450836955646,
            0.24056851311953356,
            682.55,
            636,
            686,
        ),
        "robust": (
            0.91425,
            0.9126042489999777,
            0.31142857142857144,
            667.8,
            551,
            686,
        ),
    }
    assert [row["method"] for row in summaries] == ["vp", "randomized", "robust"]
    for row in summaries:
        method = row["method"]
        balanced_accuracy, macro_f1, entry_sparsity, selected, minimum, maximum = (
            expected_summaries[method]
        )
        assert int(row["outer_splits"]) == 20
        assert int(row["held_out_decisions"]) == 4_000
        assert int(row["solver_converged_splits"]) == 0
        assert int(row["alpha_selection_qualified_splits"]) == 0
        assert row["final_loading_support_coordinate_system"] == (
            "original_snp_raw_dosage_coefficients"
        )
        assert float(row["final_loading_support_threshold"]) == pytest.approx(1e-8)
        assert float(row["mean_split_balanced_accuracy"]) == pytest.approx(
            balanced_accuracy
        )
        assert float(row["mean_split_macro_f1"]) == pytest.approx(macro_f1)
        assert float(row["mean_final_loading_entry_sparsity"]) == pytest.approx(
            entry_sparsity
        )
        assert float(row["mean_final_selected_snp_count"]) == pytest.approx(selected)
        assert int(row["min_final_selected_snp_count"]) == minimum
        assert int(row["max_final_selected_snp_count"]) == maximum


def test_current_genomics_sparse_confusion_and_per_population_metrics_are_exact() -> None:
    directory = EVALUATION / "genomics" / "sparse_20seeds"
    fields, metrics = _read_csv(directory / "per_population_metrics.csv")
    assert fields == [
        "f1",
        "max_split_recall",
        "mean_split_recall",
        "method",
        "min_split_recall",
        "outer_splits",
        "population",
        "precision",
        "predicted_count",
        "recall",
        "std_split_recall",
        "support",
        "true_positive",
    ]
    assert len(metrics) == 15
    by_method_population = {
        (row["method"], row["population"]): row for row in metrics
    }
    assert set(by_method_population) == {
        (method, population)
        for method in SPARSE_EXPECTED_RECALL
        for population in POPULATIONS
    }
    for method, recalls in SPARSE_EXPECTED_RECALL.items():
        for population, expected_recall in recalls.items():
            row = by_method_population[(method, population)]
            assert int(row["outer_splits"]) == 20
            assert int(row["support"]) == 800
            assert float(row["recall"]) == pytest.approx(expected_recall)
            diagonal = SPARSE_EXPECTED_CONFUSION[method][population][
                POPULATIONS.index(population)
            ]
            assert int(row["true_positive"]) == diagonal

    fields, rows = _read_csv(directory / "confusion_matrix.csv")
    assert fields == [
        "count",
        "method",
        "predicted_population",
        "true_population",
        "true_row_fraction",
    ]
    assert len(rows) == 75
    by_pair = {
        (row["method"], row["true_population"], row["predicted_population"]): row
        for row in rows
    }
    assert len(by_pair) == 75
    for method, expected_matrix in SPARSE_EXPECTED_CONFUSION.items():
        total = 0
        for truth in POPULATIONS:
            counts = [
                int(by_pair[(method, truth, predicted)]["count"])
                for predicted in POPULATIONS
            ]
            assert counts == expected_matrix[truth]
            assert sum(counts) == 800
            total += sum(counts)
            for predicted, count in zip(POPULATIONS, counts, strict=True):
                fraction = float(
                    by_pair[(method, truth, predicted)]["true_row_fraction"]
                )
                assert fraction == pytest.approx(count / 800)
            assert counts[POPULATIONS.index(truth)] / 800 == pytest.approx(
                SPARSE_EXPECTED_RECALL[method][truth]
            )
        assert total == 4_000
    assert int(by_pair[("vp", "AMR", "EUR")]["count"]) == 258
    assert int(by_pair[("randomized", "AMR", "EUR")]["count"]) == 260
    assert int(by_pair[("robust", "AMR", "EUR")]["count"]) == 268


def test_original_github_genomics_figures_are_preserved_byte_for_byte() -> None:
    actual_files = {path.name for path in LEGACY_GENOMICS.iterdir() if path.is_file()}
    assert actual_files == {"README.md", *LEGACY_GENOMICS_EXPECTED_SHA256}
    for filename, expected_sha256 in LEGACY_GENOMICS_EXPECTED_SHA256.items():
        path = LEGACY_GENOMICS / filename
        _assert_png(path)
        assert _sha256(path) == expected_sha256


def test_autodl_randomized_runtime_evidence_is_source_bound_and_recomputable() -> None:
    assert _sha256(RANDOMIZED_EVIDENCE) == (
        "5e4fa6c687be2b93b8d27bf4650d78ac7d634591e24ee20137693a1826102280"
    )
    payload = json.loads(RANDOMIZED_EVIDENCE.read_text(encoding="utf-8"))
    _assert_nested_finite(payload)
    assert payload["schema_version"] == 2
    matrix = payload["deterministic_matrix"]
    assert matrix["dtype"] == "<f8"
    assert matrix["shape"] == [957, 5000]
    assert matrix["array_sha256"] == (
        "11fc5e474dd27626346f5636f22e87b6d2bef79bb45b1a0158565da03b0533df"
    )

    execution_snapshots = {
        "benchmark_tool": ROOT / "tools" / "benchmark_randomized_runtime.py",
        "randomized": ROOT
        / "results"
        / "randomized_runtime_optimization_2026-08-30"
        / "provenance"
        / "assets"
        / "randomized_after.py",
        "vp": ROOT
        / "results"
        / "randomized_runtime_optimization_2026-08-30"
        / "provenance"
        / "assets"
        / "vp_dependency.py",
    }
    assert set(payload["source_files"]) == set(execution_snapshots)
    for name, source in payload["source_files"].items():
        path = execution_snapshots[name]
        assert source["relative_to"] == "repository_root"
        assert path.is_file()
        assert _sha256(path) == source["sha256"]
    assert set(payload["environment"]["thread_controls"].values()) == {"1"}

    solver = payload["solver_only_one_alpha"]
    grid = payload["randomized_alpha_grid"]
    for record in (
        payload["pca_single_fit"],
        solver["vp"],
        solver["randomized_cold"],
        grid["cold_reprepare_every_alpha"],
        grid["shared_prepare_once"],
    ):
        assert len(record["samples_seconds"]) == 7
        assert float(np.median(record["samples_seconds"])) == pytest.approx(
            record["median_seconds"]
        )

    cold = grid["cold_reprepare_every_alpha"]["median_seconds"]
    shared = grid["shared_prepare_once"]["median_seconds"]
    assert grid["median_speedup_ratio_cold_over_shared"] == pytest.approx(cold / shared)
    assert grid["median_fraction_saved"] == pytest.approx(1.0 - shared / cold)
    assert set(grid["cold_vs_shared_max_abs_differences"].values()) == {0.0}

    quality = payload["fixed_budget_quality_one_alpha"]
    assert quality["vp"]["converged"] is False
    assert quality["randomized"]["converged"] is False
    objective_gap = (
        quality["randomized"]["full_objective"] - quality["vp"]["final_objective"]
    )
    assert quality["randomized_minus_vp_objective"] == pytest.approx(objective_gap)
    assert abs(quality["randomized_minus_vp_objective_relative_to_abs_vp"]) < 1e-12


def _assert_local_markdown_links_exist(readme: Path) -> None:
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", readme.read_text(encoding="utf-8"))
    assert targets
    for target in targets:
        target = target.strip().strip("<>").split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        assert (
            (readme.parent / target).resolve().exists()
        ), f"broken link in {readme}: {target}"


def test_current_evaluation_readmes_only_link_to_existing_local_targets() -> None:
    _assert_local_markdown_links_exist(EVALUATION / "video" / "README.md")
    _assert_local_markdown_links_exist(
        EVALUATION / "genomics" / "pca_20seeds" / "README.md"
    )
    _assert_local_markdown_links_exist(
        EVALUATION / "genomics" / "sparse_20seeds" / "README.md"
    )
    _assert_local_markdown_links_exist(EVALUATION / "randomized" / "README.md")

    main_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    required_targets = [
        "assets/evaluation/video/seed_91001/video_decomposition.gif",
        "assets/evaluation/video/README.md",
        "docs/video_decomposition_visual_audit.md",
        "assets/evaluation/genomics/pca_20seeds/population_diagnostics.png",
        "assets/evaluation/genomics/pca_20seeds/README.md",
        "assets/evaluation/genomics/sparse_20seeds/README.md",
        "docs/genomics_population_diagnostics.md",
        "docs/randomized_runtime_optimization_2026-08-30.md",
        "assets/evaluation/randomized/README.md",
    ]
    for target in required_targets:
        assert target in main_readme
        assert (ROOT / target).exists()

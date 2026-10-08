from __future__ import annotations

import csv
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
import stat

import pytest

from experiments.genomics_aim_protocol import OUTPUT_FILE_NAMES, validate_output_package


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = (
    REPOSITORY_ROOT
    / "results"
    / "genomics_aim_chr20_22_formal_20seeds_20260830"
)
SEEDS = list(range(2025, 2045))
METHODS = ["pca", "vp", "randomized", "robust"]

COMMON_CONFIG = {
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 1701,
    "loading_score": "l1",
    "max_iter": 500,
    "methods": METHODS,
    "n_components": 2,
    "native_support_threshold": 1e-8,
    "near_dense_fraction": 0.95,
    "panel_size": 200,
    "randomized_alpha": 0.02,
    "randomized_finalize_full_iters": 5,
    "randomized_oversample": 10,
    "randomized_power_iterations": 2,
    "robust_alpha": 0.02,
    "robust_huber_delta": 1.345,
    "seeds": SEEDS,
    "test_fraction": 0.2,
    "tol": 1e-6,
    "vp_alpha": 0.02,
}

PACKAGES = {
    "formal_3continent_20seeds_allmethods_alpha002_v1": {
        "population_design": "three_continent",
        "samples": 600,
        "snps": 30457,
        "prep_manifest": (
            "provenance/three_continent/"
            "1kg_chr20_22_3continent_200each_ld.manifest.json"
        ),
        "prep_manifest_sha256": (
            "62f378719d54595965447a0c7fdf60cb7a1086a8fbe22c363b56a8804a607c06"
        ),
        "file_sha256": {
            "aim_protocol_method_summaries.csv": "0f235dc685dc33f53f9ccfbfd4eb45e0ef90bf096aef02992db984816cc89e2f",
            "aim_protocol_output_manifest.json": "a1d033731f622f8f76fa75574ce894ee33875314f46b4e6e05bb59994aa4adfe",
            "aim_protocol_panel_stability_and_delta.png": "28b7aa377949b8d15228e053a5dce811fb6fdce3586944c3954769f58a057375",
            "aim_protocol_population_metrics.csv": "79dfac5329e8e1d469f95f41d352ab4f45f76acfdc6670b1932cfa313c1a5147",
            "aim_protocol_registered_coordinates.csv": "e722c538cec241ad8969fe267fb410bbbcad5c24c535e83f341aea716824978b",
            "aim_protocol_registered_stratification.png": "6efbe042649802aa9fb63034e864bc37daeae333f889dcb190358ed96503453c",
            "aim_protocol_results.json": "9bca01b195d6340f29c1a6ada1582f6f74c1b75e7ec77d996b733c59977beb4e",
            "aim_protocol_selection_frequencies.csv": "30f428bbe32851edefa85ab7b21e4676f6542951412f48883d9b45d67d3608a6",
            "aim_protocol_split_metrics.csv": "d554da8ed73665c955da9107cb6733f33a5bd6e2bf238f0c4a36c722839cd305",
        },
        "summary": {
            "pca": (0.9954166666666667, 0.9962500000000001, 0.9962500000000001, 0.7809942173537835, 1.0, 20),
            "vp": (0.9954166666666667, 0.9962500000000001, 0.9962500000000001, 0.7809942173537835, 1.0, 0),
            "randomized": (0.9954166666666667, 0.9962500000000001, 0.9962500000000001, 0.7817833724231066, 0.9985074626865671, 0),
            "robust": (0.9954166666666667, 0.9954166666666667, 0.9962500000000001, 0.7529548594565721, 0.7960553401405884, 0),
        },
        "common_probe_recall": {
            "pca": {"AFR": 0.98875, "EAS": 1.0, "EUR": 1.0},
            "vp": {"AFR": 0.98875, "EAS": 1.0, "EUR": 1.0},
            "randomized": {"AFR": 0.98875, "EAS": 1.0, "EUR": 1.0},
            "robust": {"AFR": 0.98875, "EAS": 1.0, "EUR": 1.0},
        },
    },
    "formal_5superpop_20seeds_allmethods_alpha002_v1": {
        "population_design": "five_superpopulation",
        "samples": 1000,
        "snps": 27346,
        "prep_manifest": (
            "provenance/five_superpopulation/"
            "1kg_chr20_22_5superpop_200each_ld.manifest.json"
        ),
        "prep_manifest_sha256": (
            "9378814177e0721e711ce2e6c3695effc5ff766d2529c17cc5c4e0de84f30342"
        ),
        "file_sha256": {
            "aim_protocol_method_summaries.csv": "8a7c6aeead38ce8b9a272445807f3eb78758dc82936f7913243c764acf0990d7",
            "aim_protocol_output_manifest.json": "71008887511794071de779d6265ca90b645a09f4deb437277559dfa555dde931",
            "aim_protocol_panel_stability_and_delta.png": "d60331e6f940b7d5b494f73f0275376fd1517e6d8c708ecb1b142c5358bc0bc8",
            "aim_protocol_population_metrics.csv": "88c3823fb2d0ce3a46caf06be2dd20423f91edcf305401da31cc4d17b502a801",
            "aim_protocol_registered_coordinates.csv": "cc53144240068c19f590b1f7bb4434f68adef107c1cf9a149df544164ff1cf06",
            "aim_protocol_registered_stratification.png": "3d7fc02b1ee2683e1f277993aa12ebb48eeb49afd94664605e01b0bdfa5ee8a6",
            "aim_protocol_results.json": "5d7a040cfccb7db48f388942807c916030d60a28246a76b5fa9549518e48a117",
            "aim_protocol_selection_frequencies.csv": "180976574d709944813f35b61d75f8b92c6757e05a149ccaf383226401737c4b",
            "aim_protocol_split_metrics.csv": "8960fa1a92064ee87e3e52f72d81ef86d866f7c33c342d9ab26a7b68380c3a51",
        },
        "summary": {
            "pca": (0.83125, 0.783, 0.783, 0.7908736501130382, 1.0, 20),
            "vp": (0.83125, 0.783, 0.783, 0.7914592706727914, 0.9990049751243781, 0),
            "randomized": (0.83125, 0.78325, 0.78325, 0.7912031226328547, 0.9985074626865671, 0),
            "robust": (0.8145, 0.80775, 0.8075, 0.6095249445122729, 0.5026329121338737, 0),
        },
        "common_probe_recall": {
            "pca": {"AFR": 0.98625, "AMR": 0.3375, "EAS": 1.0, "EUR": 0.97125, "SAS": 0.62},
            "vp": {"AFR": 0.98625, "AMR": 0.33625, "EAS": 1.0, "EUR": 0.97125, "SAS": 0.62125},
            "randomized": {"AFR": 0.98625, "AMR": 0.33625, "EAS": 1.0, "EUR": 0.97125, "SAS": 0.6225},
            "robust": {"AFR": 0.98625, "AMR": 0.41875, "EAS": 1.0, "EUR": 0.96875, "SAS": 0.66375},
        },
    },
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("package_name", tuple(PACKAGES))
def test_formal_package_is_immutable_valid_and_exact(package_name: str) -> None:
    expected = PACKAGES[package_name]
    package = RESULT_ROOT / package_name
    entries = {entry.name: entry for entry in package.iterdir()}

    assert set(entries) == set(OUTPUT_FILE_NAMES.values())
    assert len(entries) == 9
    for filename, expected_digest in expected["file_sha256"].items():
        mode = entries[filename].lstat().st_mode
        assert stat.S_ISREG(mode)
        assert not stat.S_ISLNK(mode)
        assert _digest(entries[filename]) == expected_digest

    manifest = validate_output_package(package)
    payload = json.loads(
        (package / OUTPUT_FILE_NAMES["results_json"]).read_text(encoding="utf-8")
    )
    expected_config = dict(COMMON_CONFIG)
    expected_config["population_design"] = expected["population_design"]
    assert payload["config"] == expected_config
    assert payload["registered_seed_policy"] == "first_config_seed_not_best"
    assert manifest["registered_seed"] == SEEDS[0]
    assert manifest["registered_seed_policy"] == "first_config_seed_not_best"
    assert {row["seed"] for row in payload["registered_coordinates"]} == {SEEDS[0]}
    assert {
        row["registered_seed_policy"] for row in payload["registered_coordinates"]
    } == {"first_config_seed_not_best"}
    assert {
        (row["seed"], row["method"]) for row in payload["split_records"]
    } == {(seed, method) for seed in SEEDS for method in METHODS}

    identity = payload["input_identity"]
    assert identity["samples"] == expected["samples"]
    assert identity["snps"] == expected["snps"]
    assert identity["source_manifest_sha256"] == expected["prep_manifest_sha256"]
    prep_manifest = RESULT_ROOT / expected["prep_manifest"]
    assert prep_manifest.is_file() and not prep_manifest.is_symlink()
    assert _digest(prep_manifest) == expected["prep_manifest_sha256"]

    summaries = {row["method"]: row for row in payload["method_summaries"]}
    assert list(summaries) == METHODS
    for method, values in expected["summary"].items():
        full_width, native, common_probe, stability, pca_jaccard, qualified = values
        summary = summaries[method]
        assert summary["mean_full_width_native_balanced_accuracy"] == pytest.approx(full_width)
        assert summary["mean_native_embedding_balanced_accuracy"] == pytest.approx(native)
        assert summary["mean_common_pca_probe_balanced_accuracy"] == pytest.approx(common_probe)
        assert summary["mean_pairwise_panel_jaccard_across_splits"] == pytest.approx(stability)
        assert summary["mean_panel_jaccard_with_equal_budget_pca"] == pytest.approx(pca_jaccard)
        assert summary["qualified_splits"] == qualified

    recall_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with (package / OUTPUT_FILE_NAMES["population_metrics_csv"]).open(
        encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            if row["evaluation"] == "common_pca_probe":
                recall_values[(row["method"], row["population"])].append(
                    float(row["recall"])
                )
    for method, population_values in expected["common_probe_recall"].items():
        for population, value in population_values.items():
            observed = recall_values[(method, population)]
            assert len(observed) == len(SEEDS)
            assert sum(observed) / len(observed) == pytest.approx(value)


def test_formal_result_documentation_is_linked_and_numerically_bound() -> None:
    card_path = RESULT_ROOT / "README.md"
    card = card_path.read_text(encoding="utf-8")
    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    protocol = (REPOSITORY_ROOT / "docs/genomics_vp_aim_protocol.md").read_text(
        encoding="utf-8"
    )

    required_card_lines = {
        "| PCA | 0.995417 | 0.996250 | 0.996250 | 0.988750 | 1.000000 | 1.000000 |",
        "| Robust VP-SPCA (supplementary) | 0.995417 | 0.995417 | 0.996250 | 0.988750 | 1.000000 | 1.000000 |",
        "| Randomized VP-SPCA | 0.781783 | 0.998507 | 170 / 129 | 0/20 |",
        "| PCA | 0.831250 | 0.783000 | 0.783000 | 0.986250 | 0.337500 | 1.000000 | 0.971250 | 0.620000 |",
        "| Robust VP-SPCA (supplementary) | 0.814500 | 0.807750 | 0.807500 | 0.986250 | 0.418750 | 1.000000 | 0.968750 | 0.663750 |",
        "| Randomized VP-SPCA | 0.791203 | 0.998507 | 170 / 129 | 0/20 |",
        "| Robust VP-SPCA (supplementary) | 0.609525 | 0.502633 | 133 / 84 | 0/20 |",
    }
    assert all(line in card for line in required_card_lines)
    assert "not one deployable AIM" in card
    assert "not race" in card
    assert "not provide independent replication" in card
    assert "Common-probe AMR recall" in card

    for target in re.findall(r"!?(?:\[[^]]*\])\(([^)#]+)(?:#[^)]+)?\)", card):
        assert (card_path.parent / target).resolve().exists(), target

    result_card_link = (
        "results/genomics_aim_chr20_22_formal_20seeds_20260830/README.md"
    )
    assert result_card_link in root_readme
    assert "0.996250" in root_readme and "0.783250" in root_readme
    assert "Robust VP-SPCA is retained only as a" in root_readme
    assert (
        "formal_5superpop_20seeds_allmethods_alpha002_v1/"
        "aim_protocol_registered_stratification.png"
    ) in root_readme

    for flag in (
        "--vp-alpha 0.02",
        "--randomized-alpha 0.02",
        "--robust-alpha 0.02",
    ):
        assert flag in protocol
    for expected in PACKAGES.values():
        assert expected["prep_manifest_sha256"] in protocol
    assert "[`formal 20-seed result card`]" in protocol
    assert "supplementary Robust VP-SPCA diagnostic" in protocol
    assert "Population recalls in the formal result card use the fixed-200" in protocol

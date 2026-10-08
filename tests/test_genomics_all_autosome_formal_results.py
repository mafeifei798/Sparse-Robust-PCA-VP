from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "results"
    / "genomics_all_autosome_stratification_2026-08-31"
    / "remote_v2_clean"
    / "registered_20seeds_merged"
)
RESULT_ROOT = PACKAGE.parents[1]
PROVENANCE = RESULT_ROOT / "provenance"

EXPECTED_FILE_SHA256 = {
    "all_autosome_stratification_confusion.png": "fc5361f405bc9d2c6d06f7203203b698059270cc97c81de6d9796452853f2c1e",
    "all_autosome_stratification_grid_metrics.csv": "61fd28b8600ed96710dd363e8f90b7318ea1b1ac659f9d078f36df3dd35ceeda",
    "all_autosome_stratification_output_manifest.json": "1bfde1ee122f8e1d2472a825127271f9cd9ab2e966e00b279af4030871d6140d",
    "all_autosome_stratification_panel_stability.csv": "e7bdb7e92823a64bf4e79dc2b1403593e2c37439a8d8cdbaa69d2795d492f2a0",
    "all_autosome_stratification_pc12.png": "c0631debf6520e08d5f822a123c7a63970e8581ae54741b40791217b36c37843",
    "all_autosome_stratification_pc34.png": "11f908bd9e1da04c313daa75b013f8ff2cbe8fdf02a19f9523fca553bdc65606",
    "all_autosome_stratification_population_recall.csv": "a3c29218a8dddb3314f9cfdcd64c33118f577f960ba369953237f5d528ffca62",
    "all_autosome_stratification_registered_coordinates.csv": "cdf0c66fbfefbef2decd125560386c5eb395b11dd21ce21c384018765d215fe7",
    "all_autosome_stratification_results.json": "7b25edd08c35f8e23bf92a25d178d01e28ba0a3b176a9812a9994519906ecbaf",
    "all_autosome_stratification_sample_predictions.csv": "b04af811ef853a833cf3e8d68a11129dce1985173c1bc8fba74288861b488c3c",
}

EXPECTED_PROVENANCE_FILE_SHA256 = {
    "genomics_all_autosome_stratification_intent_v2_2026-08-31.json": "65f1b6a5dbb51c78ca9e9e5a8b7c92b9f882020f86bd84ea3146188acd463fc8",
    "genomics_all_autosome_stratification_intent_v2_clean_2026-08-31.json": "4d86d9238b1800fe844befc6b07211ede75d0e668a3ac7d144a3cbbe57a8c22e",
    "rspca-all-autosome-stratification-v2-clean-final.tgz": "e56c391cb469d2f3d7e72d3068d2a16844c6c1561517964429aad1607787fed9",
}

EXPECTED_REMOTE_LOG_SHA256 = {
    "merge.log": "253933b722497af0751d0cf3bcf3fb5045bdc47f365c64237b54bd425de53c13",
    "shard_2025_2029.log": "9ad01781c3a2277269ce68deebdad07e5f8f6a614177920d6cd671dffae2103c",
    "shard_2030_2034.log": "d8139efa63721ff6177c6592ce642d34666933bd696227e324a443c8ae109a3e",
    "shard_2035_2039.log": "56c57d5ea22cae6a24fa12c00afe640353027315007d4cdf8cf33aae0af176a3",
    "shard_2040_2044.log": "a77c373d26efedce16fab1992d24d72a1a3c633479d752483fb0ef536636fbde",
    "smoke_preflight.log": "231aef00b45f1fbcbd330eb638a5828156d9735b22513eee3bc138744dfe8fe0",
    "smoke_validate_self.log": "e2d55a8769233c6a60c3c693f9b28f00805fd26f60aa64aca883f88dea5cbae9",
    "validate_merged_trusted.log": "a2cf0827283859272bfb48c3073e2476f7e72d2aafe68f18599cd225a52f822d",
    "validate_self_2025_2029.log": "21bf6eb0bb964bba3df68c49f6bd0940ea97546ed6b03a358f0acc46254cf3c5",
    "validate_self_2030_2034.log": "7667c59b429ae2ac7bbbbc6297d5110b094f2cfd43c394b073aa9b83b6bc45fb",
    "validate_self_2035_2039.log": "590851c7a6731fed7e9987ba7e2e41799f93a5a662e7e8532a349fe09fbd452c",
    "validate_self_2040_2044.log": "a294035645c4590fd952327206d810a4a341b0e8a29882476fa6bc44c0848b05",
    "validate_self_merged.log": "c5d24d9cb32691d29679c903ed31254ad4de58b379ba98eeb1b15c26b69698cb",
    "validate_shard_2025_2029.log": "86f4cfa26696494fa128caa4cade353a9586fd5a18ad2476e54bc1e23d915e0e",
    "validate_shard_2030_2034.log": "f80629191730678d9f0a19db5ad05cf6e1c676a99e9db54b568f87309c2decae",
    "validate_shard_2035_2039.log": "92fa5f4f5ab8386b7b5ed5c7667e65afab1e64f09061e828a803a64f04fc0a6e",
    "validate_shard_2040_2044.log": "d1e1436e0ac232a6fd2c0d8b28c8870fea96aee0de0ccc0ade011542a0e47738",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )


def test_downloaded_all_autosome_package_is_exact_and_closed() -> None:
    assert PACKAGE.is_dir()
    assert not PACKAGE.is_symlink()
    actual = {path.name for path in PACKAGE.iterdir()}
    assert actual == set(EXPECTED_FILE_SHA256)
    for name, expected in EXPECTED_FILE_SHA256.items():
        path = PACKAGE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert _sha256(path) == expected

    manifest = _strict_json(PACKAGE / "all_autosome_stratification_output_manifest.json")
    assert manifest["analysis_sha256"] == (
        "46190a1e12c9c0213b70b8e9778579cebcf8374915874436d26cea1805402981"
    )
    assert manifest["package_anchor_sha256"] == (
        "4afd97e46570a3cec4d32be83c1abc878d6b1ee04bb7a6efc0d5c66757040d9a"
    )
    assert manifest["status"] == "complete_protocol"
    assert manifest["protocol_seed_cover_complete"] is True
    assert manifest["manifest_written_last"] is True
    assert manifest["confirmatory"] is False
    assert manifest["target_metric_or_quality_gate_used"] is False
    assert manifest["best_configuration_selected"] is False
    assert manifest["included_seeds"] == list(range(2025, 2045))


def test_downloaded_all_autosome_grid_reports_every_fixed_cell() -> None:
    with (PACKAGE / "all_autosome_stratification_grid_metrics.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    index = {(int(row["n_components"]), row["panel_label"]): row for row in rows}
    assert set(index) == {
        (rank, panel)
        for rank in (4, 6, 8)
        for panel in ("5pct", "10pct", "25pct", "50pct", "all")
    }

    expected = {
        (4, "5pct"): (22755, 0.93875, 0.937217395960824, -0.0125),
        (4, "all"): (455102, 0.95125, 0.9504999265202743, 0.0),
        (6, "5pct"): (22755, 0.9435, 0.9422636662636247, -0.01275),
        (6, "all"): (455102, 0.95625, 0.9556748030383458, 0.0),
        (8, "5pct"): (22755, 0.94875, 0.9478686804598897, -0.0075),
        (8, "all"): (455102, 0.95625, 0.9556975701175097, 0.0),
    }
    for key, (snps, ba, macro_f1, delta) in expected.items():
        row = index[key]
        assert int(row["seeds"]) == 20
        assert int(row["minimum_selected_snp_count"]) == snps
        assert int(row["maximum_selected_snp_count"]) == snps
        assert abs(float(row["mean_balanced_accuracy"]) - ba) < 1e-14
        assert abs(float(row["mean_macro_f1"]) - macro_f1) < 1e-14
        assert (
            abs(float(row["mean_paired_balanced_accuracy_delta_vs_all"]) - delta)
            < 1e-14
        )
        assert row["descriptive_only_no_qualification_gate"] == "True"


def test_rank8_population_result_keeps_admixed_boundary_visible() -> None:
    with (PACKAGE / "all_autosome_stratification_population_recall.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    index = {
        (int(row["n_components"]), row["panel_label"], row["population"]): row
        for row in rows
    }
    expected = {
        ("5pct", "AFR"): 0.98875,
        ("5pct", "AMR"): 0.755,
        ("5pct", "EAS"): 1.0,
        ("5pct", "EUR"): 1.0,
        ("5pct", "SAS"): 1.0,
        ("all", "AFR"): 0.98875,
        ("all", "AMR"): 0.7925,
        ("all", "EAS"): 1.0,
        ("all", "EUR"): 1.0,
        ("all", "SAS"): 1.0,
    }
    for (panel, population), recall in expected.items():
        row = index[(8, panel, population)]
        assert int(row["seeds"]) == 20
        assert int(row["pooled_support"]) == 800
        assert abs(float(row["mean_recall"]) - recall) < 1e-14


def test_downloaded_execution_provenance_is_exact_and_complete() -> None:
    assert PROVENANCE.is_dir()
    assert not PROVENANCE.is_symlink()
    assert {path.name for path in PROVENANCE.iterdir()} == {
        *EXPECTED_PROVENANCE_FILE_SHA256,
        "remote_logs",
    }
    for name, expected in EXPECTED_PROVENANCE_FILE_SHA256.items():
        path = PROVENANCE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert _sha256(path) == expected

    logs = PROVENANCE / "remote_logs"
    assert logs.is_dir()
    assert not logs.is_symlink()
    assert {path.name for path in logs.iterdir()} == set(EXPECTED_REMOTE_LOG_SHA256)
    for name, expected in EXPECTED_REMOTE_LOG_SHA256.items():
        path = logs / name
        assert path.is_file()
        assert not path.is_symlink()
        assert _sha256(path) == expected

    merged_replay = (logs / "validate_merged_trusted.log").read_text(
        encoding="utf-8"
    )
    assert '"protocol_seed_cover_complete": true' in merged_replay
    assert '"package_self_consistency_verified": true' in merged_replay
    assert '"trusted_scientific_replay_verified": true' in merged_replay
    assert '"trusted_input_replay_verified": false' in merged_replay
    assert '"source_shard_provenance_replay_verified": false' in merged_replay

    result_card = (RESULT_ROOT / "README.md").read_text(encoding="utf-8")
    for figure in (
        "all_autosome_stratification_pc12.png",
        "all_autosome_stratification_pc34.png",
        "all_autosome_stratification_confusion.png",
    ):
        assert figure in result_card

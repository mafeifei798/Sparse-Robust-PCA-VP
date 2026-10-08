from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments import genomics_pca_loading_amr_whitening as amr


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = (
    ROOT
    / "results"
    / "genomics_amr_whitening_2026-08-31"
    / "registered_20seeds_raw_vs_whitened"
)
PROVENANCE = PACKAGE.parent / "provenance"

ANALYSIS_SHA256 = "b2aced797ab06b6402ae7d7c1bb377b924696096b1e8c22b0d8280f5b6a3ecee"
PACKAGE_ANCHOR_SHA256 = (
    "9d33ffe4b5260a2c13c3813a654b38250e991a37d6e450856408c18f6a949248"
)

EXPECTED_FILE_SHA256 = {
    "amr_whitening_full_grid.png": (
        "a855f91ee84f85f3bb26565cc2eb602cd3dd68cc31b69cc18528e349710209bc"
    ),
    "amr_whitening_grid.csv": (
        "74053eca698909933038e99b2fca2097256381d99ba99be54a19ef35c327b4ec"
    ),
    "amr_whitening_output_manifest.json": (
        "3d3d6e9c15928e412300fae98e73bcc7f77af759a3a6c14f8078a848b29395c9"
    ),
    "amr_whitening_paired_deltas.csv": (
        "056a4f1cc9c6368e84af6ce662a0f69b39b5db6f7cf17f979ca8a75769020c30"
    ),
    "amr_whitening_predictions.csv": (
        "12b0b442a6c29e02bdc46ba6aaabcc45f75d7e5ebb0b7df8f8f2798014e3833e"
    ),
    "amr_whitening_rank8_all_tradeoff.png": (
        "a87c2ef914ab5161cfc713cc1618d76a3106c4dea883cabe06be24c91763dba4"
    ),
    "amr_whitening_results.json": (
        "23db7d0c5fe452c11d8408fd5006933fc8822323c0da3b2ce920c6254f81bf0a"
    ),
}

EXPECTED_PROVENANCE_SHA256 = {
    "genomics_amr_whitening_intent_2026-08-31.json": (
        "4167c0c1de646b20e6020abadc2c876e8881bdcc0cd2077c26e207f318046a06"
    ),
    "remote_logs/formal_run.log": (
        "a294c12a3f024f81b3d699df0ae11c2938476c3e9ccd2e1d67aa4e9944ef0fda"
    ),
    "remote_logs/formal_trusted_replay.log": (
        "bbb4ce463ae722746101b29e98c44bf83c68ce93b1cf4cd27df6776a4ee92f54"
    ),
    "remote_logs/formal_validate.log": (
        "497433c9b3756f58a0b17d51da3e26eb6c5ecd910c9fe41f918fc00c43b63998"
    ),
    "remote_logs/legacy_raw_replay.log": (
        "74012324ae7eb5663b463f6c2065dfc89bd7c132adc0e06de3d83a06145ebd3e"
    ),
    "source/rspca-amr-whitening-source-20260831-v1.tgz": (
        "667ef39740d682e0807d7968e14f28736880e17c4eecb31439aea13e0feadca5"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_text(text: str) -> dict[str, object]:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(
        text,
        object_pairs_hook=pairs_hook,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON token: {token}")
        ),
    )


def _strict_json(path: Path) -> dict[str, object]:
    return _strict_json_text(path.read_text(encoding="utf-8"))


def _strict_last_json_line(path: Path) -> dict[str, object]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines
    return _strict_json_text(lines[-1])


def test_formal_package_is_exact_closed_and_source_bound() -> None:
    assert PACKAGE.is_dir()
    assert not PACKAGE.is_symlink()
    assert {path.name for path in PACKAGE.iterdir()} == set(EXPECTED_FILE_SHA256)
    for name, expected in EXPECTED_FILE_SHA256.items():
        path = PACKAGE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert _sha256(path) == expected

    validation = amr.validate_output_package(
        PACKAGE,
        repo_root=ROOT,
        expected_analysis_sha256=ANALYSIS_SHA256,
        expected_package_anchor_sha256=PACKAGE_ANCHOR_SHA256,
    )
    assert validation["valid"]
    assert validation["source_bytes_verified"]
    assert validation["full_grid_closed"]
    assert validation["paired_fit_identity_verified"]
    assert validation["records"] == 600
    assert validation["predictions"] == 120_000
    assert validation["seeds"] == 20


def test_formal_provenance_is_separate_and_exact() -> None:
    actual = {
        path.relative_to(PROVENANCE).as_posix()
        for path in PROVENANCE.rglob("*")
        if path.is_file()
    }
    assert actual == set(EXPECTED_PROVENANCE_SHA256)
    for relative, expected in EXPECTED_PROVENANCE_SHA256.items():
        path = PROVENANCE / relative
        assert not path.is_symlink()
        assert _sha256(path) == expected

    trusted = _strict_last_json_line(
        PROVENANCE / "remote_logs/formal_trusted_replay.log"
    )
    assert trusted["analysis_sha256"] == ANALYSIS_SHA256
    assert trusted["package_anchor_sha256"] == PACKAGE_ANCHOR_SHA256
    assert trusted["trusted_scientific_replay_verified"] is True
    assert trusted["source_bytes_verified"] is True
    assert trusted["records"] == 600
    assert trusted["predictions"] == 120_000

    legacy = _strict_last_json_line(PROVENANCE / "remote_logs/legacy_raw_replay.log")
    assert legacy["valid"] is True
    assert legacy["raw_grid_cells_exactly_reproduced"] == 15
    assert legacy["prediction_and_metric_equivalence"] is True


def test_rank8_all_whitening_improves_amr_without_eur_tradeoff() -> None:
    payload = _strict_json(PACKAGE / amr.RESULT_FILENAME)
    summaries = {
        (row["n_components"], row["panel_label"], row["distance_mode"]): row
        for row in payload["grid_summaries"]
    }
    raw = summaries[(8, "all", amr.DISTANCE_MODES[0])]
    white = summaries[(8, "all", amr.DISTANCE_MODES[1])]

    assert raw["balanced_accuracy"]["mean"] == 0.95275
    assert white["balanced_accuracy"]["mean"] == 0.97875
    assert raw["macro_f1"]["mean"] == 0.9518492357406163
    assert white["macro_f1"]["mean"] == 0.9786140951610584
    assert raw["population_metrics"]["AMR"]["recall"]["mean"] == (0.7687499999999999)
    assert white["population_metrics"]["AMR"]["recall"]["mean"] == (0.8987499999999999)
    assert (
        white["population_metrics"]["AMR"]["precision"]["mean"]
        > raw["population_metrics"]["AMR"]["precision"]["mean"]
    )
    assert raw["population_metrics"]["EUR"]["recall"]["mean"] == 1.0
    assert white["population_metrics"]["EUR"]["recall"]["mean"] == 1.0

    paired = next(
        row
        for row in payload["paired_summaries"]
        if row["n_components"] == 8 and row["panel_label"] == "all"
    )
    ba = paired["deltas"]["balanced_accuracy"]
    amr_recall = paired["deltas"]["AMR_recall"]
    assert ba["candidate_minus_raw_mean"] == 0.026000000000000023
    assert ba["paired_percentile_95_ci"] == [
        0.02175000000000002,
        0.030750000000000038,
    ]
    assert (ba["positive_seeds"], ba["equal_seeds"], ba["negative_seeds"]) == (
        20,
        0,
        0,
    )
    assert amr_recall["candidate_minus_raw_mean"] == 0.13
    assert amr_recall["paired_percentile_95_ci"] == [
        0.10875000000000004,
        0.15375,
    ]
    assert (
        amr_recall["positive_seeds"],
        amr_recall["equal_seeds"],
        amr_recall["negative_seeds"],
    ) == (20, 0, 0)

    ledger = payload["headline_transition_ledger"]
    assert ledger["selection_reason"] == (
        "historical_formal_headline_alignment_not_new_result_selection"
    )
    assert ledger["raw_AMR_to_EUR"] == 181
    assert ledger["whitened_AMR_to_EUR"] == 77
    assert ledger["raw_EUR_to_AMR"] == 0
    assert ledger["whitened_EUR_to_AMR"] == 0
    assert ledger["population_transitions"]["AMR"] == {
        "support": 800,
        "raw_wrong_to_whitened_correct": 104,
        "raw_correct_to_whitened_wrong": 0,
        "both_correct": 615,
        "both_wrong": 81,
    }


def test_full_grid_reports_rank_dependent_boundary_without_winner_selection() -> None:
    payload = _strict_json(PACKAGE / amr.RESULT_FILENAME)
    assert payload["config"]["seeds"] == list(range(731001, 731021))
    assert payload["analysis_policy"]["best_configuration_selected"] is False
    assert payload["analysis_policy"]["all_30_cells_reported"] is True
    assert payload["analysis_policy"]["headline_reason"] == (
        "historical_formal_alignment_not_new_result_selection"
    )
    assert len(payload["grid_summaries"]) == 30
    assert len(payload["paired_summaries"]) == 15

    paired = {
        (row["n_components"], row["panel_label"]): row
        for row in payload["paired_summaries"]
    }
    for rank in (6, 8):
        for panel in ("5pct", "10pct", "25pct", "50pct", "all"):
            delta = paired[(rank, panel)]["deltas"]["balanced_accuracy"]
            assert delta["positive_seeds"] == 20
            assert delta["negative_seeds"] == 0
    rank4_all = paired[(4, "all")]["deltas"]
    assert rank4_all["balanced_accuracy"]["candidate_minus_raw_mean"] == (
        -0.008999999999999985
    )
    assert rank4_all["AMR_recall"]["candidate_minus_raw_mean"] == (-0.04500000000000002)


def test_post_prediction_subpopulation_audit_localizes_amr_gain() -> None:
    counts: dict[tuple[str, str], list[int]] = {}
    with (PACKAGE / amr.PREDICTION_FILENAME).open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            if (
                int(row["n_components"]) != 8
                or row["panel_label"] != "all"
                or row["truth"] != "AMR"
            ):
                continue
            key = (row["distance_mode"], row["subpopulation"])
            counts.setdefault(key, [0, 0])
            counts[key][1] += 1
            counts[key][0] += row["correct"] == "True"

    expected = {
        (amr.DISTANCE_MODES[0], "CLM"): (176, 217),
        (amr.DISTANCE_MODES[1], "CLM"): (200, 217),
        (amr.DISTANCE_MODES[0], "MXL"): (142, 146),
        (amr.DISTANCE_MODES[1], "MXL"): (142, 146),
        (amr.DISTANCE_MODES[0], "PEL"): (199, 199),
        (amr.DISTANCE_MODES[1], "PEL"): (199, 199),
        (amr.DISTANCE_MODES[0], "PUR"): (98, 238),
        (amr.DISTANCE_MODES[1], "PUR"): (178, 238),
    }
    assert {key: tuple(value) for key, value in counts.items()} == expected

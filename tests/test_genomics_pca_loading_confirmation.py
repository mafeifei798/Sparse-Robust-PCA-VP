from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_pca_loading_confirmation as confirmation
from experiments.genomics_pca_loading_grid import GenotypeInput
from rspca.datasets import array_sha256, canonical_sha256
from rspca.genomics_aim import stratified_split


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_RESULT_ROOT = (
    REPO_ROOT / "results" / "genomics_pca_loading_confirmation_2026-08-31"
)
FORMAL_LOCK_SHA256 = (
    "8c8b007fded84353247718d3238434aed66d44845547997bd83ed4d2663e0f63"
)
FORMAL_ANALYSIS_SHA256 = (
    "687c90b455e9509393de448566f0ba23257865cdad0d2951cb62be36d3333264"
)
FORMAL_PACKAGE_ANCHOR_SHA256 = (
    "1baa0efd49c3d6617c150169cec543f3bdcae4496ee45310bed01e89ae1ab881"
)


def _dataset() -> GenotypeInput:
    rng = np.random.default_rng(8831)
    labels = np.asarray(
        [
            population
            for population in confirmation.EXPECTED_POPULATIONS
            for _ in range(10)
        ],
        dtype=str,
    )
    probabilities = np.full((labels.size, 440), 0.28)
    for population_index, population in enumerate(confirmation.EXPECTED_POPULATIONS):
        rows = labels == population
        start = 40 * population_index
        probabilities[rows, start : start + 32] = 0.84
        probabilities[rows, 220 + 20 * population_index : 230 + 20 * population_index] = (
            0.10 + 0.15 * population_index
        )
    genotypes = rng.binomial(2, probabilities).astype(float)
    return GenotypeInput(
        genotypes=genotypes,
        sample_ids=np.asarray(
            [f"sample_{index:03d}" for index in range(labels.size)], dtype=str
        ),
        snp_ids=np.asarray(
            [f"22:{index + 1}:A:G" for index in range(genotypes.shape[1])],
            dtype=str,
        ),
        chromosomes=np.asarray(["22"] * genotypes.shape[1], dtype=str),
        population_labels=labels,
    )


def _prepared_identity(data: GenotypeInput) -> dict[str, object]:
    manifest_sha = "1" * 64
    return {
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "expected_external_source_manifest_sha256": manifest_sha,
        "external_source_manifest_anchor_verified": True,
        "external_source_manifest_bytes": 1234,
        "external_source_manifest_sha256": manifest_sha,
        "genotype_npz_bytes": 4321,
        "genotype_npz_sha256": "2" * 64,
        "genotypes_sha256": array_sha256(data.genotypes),
        "ld_pruned_input_validated_from_external_manifest": True,
        "manifest_input_scope": "legacy_chr20_22",
        "manifest_ld_pruned_receipts_verified": True,
        "manifest_population_design": "five_superpopulation",
        "orientation": "samples_by_snps",
        "population_labels_sha256": array_sha256(data.population_labels),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "samples": int(data.genotypes.shape[0]),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "snps": int(data.genotypes.shape[1]),
    }


def _lock(data: GenotypeInput) -> dict[str, object]:
    snapshot = [
        {
            "n_components": n_components,
            "panel_label": panel_label,
            "mean_balanced_accuracy": float(
                0.50 + 0.01 * n_components + 0.001 * (0 if panel_label == "all" else int(panel_label) // 200)
            ),
        }
        for n_components in (2, 3, 4)
        for panel_label in ("200", "400", "800", "all")
    ]
    snapshot_map = {
        (row["n_components"], row["panel_label"]): row["mean_balanced_accuracy"]
        for row in snapshot
    }
    source_identity = confirmation._source_identity(REPO_ROOT)
    lock: dict[str, object] = {
        "schema_version": confirmation.LOCK_SCHEMA_VERSION,
        "status": "locked_before_confirmation_execution",
        "development_evidence": {
            "analysis_sha256": "3" * 64,
            "package_anchor_sha256": "4" * 64,
            "source_closure_verified": True,
            "external_anchors_verified": True,
            "development_seeds": list(confirmation.grid.DEFAULT_SEEDS),
            "prepared_input_identity": _prepared_identity(data),
            "complete_grid_balanced_accuracy_snapshot": snapshot,
            "baseline_development_mean_balanced_accuracy": snapshot_map[(2, "200")],
            "locked_development_mean_balanced_accuracy": snapshot_map[(4, "400")],
        },
        "selection": {
            "baseline": dict(confirmation.BASELINE),
            "locked_candidate": dict(confirmation.LOCKED),
            "reason": confirmation.SELECTION_REASON,
            "selection_is_post_hoc_to_development_grid": True,
            "confirmation_outcomes_seen_before_lock": False,
            "confirmation_split_rows_accessed_before_lock": False,
            "no_best_seed_or_confirmation_seed_selected": True,
            "pre_outcome_boundary": confirmation.PRE_OUTCOME_BOUNDARY,
        },
        "confirmation_protocol": dict(confirmation.FROZEN_CONFIRMATION_PROTOCOL),
        "claim_gates": dict(confirmation.FROZEN_CLAIM_GATES),
        "method_scope": dict(confirmation.FROZEN_METHOD_SCOPE),
        "seed_novelty_audit": {
            "scope": "existing_genomics_result_evidence_semantic_seed_fields",
            "scope_limitation": (
                "absence in the supplied repository is not proof that no external or "
                "unrecorded outcome has ever been viewed"
            ),
            "requested_confirmation_seeds": list(confirmation.CONFIRMATION_SEEDS),
            "files_scanned": 0,
            "semantic_seed_fields_seen": 0,
            "scanned_file_receipts_sha256": canonical_sha256([]),
            "hits": [],
            "passed": True,
            "cross_domain_numeric_reuse_is_out_of_scope": True,
        },
        "execution_source_identity": source_identity,
        "lock_builder_sha256": source_identity["files"][
            "experiments/genomics_pca_loading_confirmation.py"
        ],
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    return lock


def _input_identity(data: GenotypeInput) -> dict[str, object]:
    identity = _prepared_identity(data)
    identity["ld_pruned_input_bound_to_anchored_receipts"] = True
    return identity


def _payload(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, object], dict[str, object], GenotypeInput]:
    data = _dataset()
    monkeypatch.setattr(
        confirmation, "EXPECTED_PREPARED_SAMPLES", int(data.genotypes.shape[0])
    )
    lock = _lock(data)
    confirmation.validate_confirmation_lock(lock)
    payload = confirmation.run_locked_confirmation(
        data,
        lock,
        lock_file_sha256="a" * 64,
        input_identity=_input_identity(data),
        source_identity=lock["execution_source_identity"],
    )
    return payload, lock, data


def _resign_payload(payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(unsigned)


def test_seed_audit_catches_prior_confirmation_and_ignores_numeric_nonseed(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "results/genomics_prior"
    evidence.mkdir(parents=True)
    (evidence / "unrelated.json").write_text(json.dumps({"coordinate": 3001}))
    clean = confirmation.audit_confirmation_seed_novelty(tmp_path)
    assert clean["passed"] is True
    (evidence / "pca_loading_confirmation_results.json").write_text(
        json.dumps({"protocol": {"seeds": [3001, 3002]}})
    )
    audited = confirmation.audit_confirmation_seed_novelty(tmp_path)
    assert audited["passed"] is False
    assert {hit["seed"] for hit in audited["hits"]} == {3001, 3002}


def test_lock_exactly_freezes_protocol_input_source_and_claim_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset()
    monkeypatch.setattr(confirmation, "EXPECTED_PREPARED_SAMPLES", 50)
    lock = _lock(data)
    assert confirmation.validate_confirmation_lock(lock)["status"] == "valid"
    for path, value, message in (
        (("claim_gates", "primary_balanced_accuracy"), "changed", "claim gates"),
        (("confirmation_protocol", "held_out_use"), "training", "frozen plan"),
        (("development_evidence", "prepared_input_identity", "genotype_npz_sha256"), "bad", "prepared"),
        (("execution_source_identity", "formal_genomics_runner_modified"), True, "source"),
    ):
        tampered = json.loads(json.dumps(lock))
        target = tampered
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        tampered.pop("lock_payload_sha256")
        tampered["lock_payload_sha256"] = canonical_sha256(tampered)
        with pytest.raises(ValueError, match=message):
            confirmation.validate_confirmation_lock(tampered)


def test_anchored_lock_file_rejects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = _dataset()
    monkeypatch.setattr(confirmation, "EXPECTED_PREPARED_SAMPLES", 50)
    lock = _lock(data)
    path = tmp_path / confirmation.LOCK_FILENAME
    confirmation.write_confirmation_lock(lock, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = confirmation.load_anchored_lock(path, digest)
    assert loaded["lock_payload_sha256"] == lock["lock_payload_sha256"]
    with pytest.raises(ValueError, match="external file-hash anchor"):
        confirmation.load_anchored_lock(path, "0" * 64)


def test_confirmation_exact_cover_pairing_and_coordinate_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, lock, _ = _payload(monkeypatch)
    validation = confirmation.validate_confirmation_payload(
        payload, lock, expected_lock_file_sha256="a" * 64
    )
    assert validation["records"] == 40
    assert validation["paired_seeds"] == 20
    assert len(payload["registered_coordinates"]) == 30
    assert payload["method_scope"] == confirmation.FROZEN_METHOD_SCOPE
    assert payload["protocol"]["cohort_or_population_generalization_interval"] is False


def test_confirmation_rankings_are_heldout_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _dataset()
    monkeypatch.setattr(confirmation, "EXPECTED_PREPARED_SAMPLES", 50)
    train, heldout = stratified_split(
        data.population_labels, 0.2, confirmation.CONFIRMATION_SEEDS[0]
    )
    assert not set(train.tolist()) & set(heldout.tolist())
    mutated_matrix = data.genotypes.copy()
    mutated_matrix[heldout] = 2.0 - mutated_matrix[heldout]
    mutated = replace(data, genotypes=mutated_matrix)
    first_lock = _lock(data)
    second_lock = _lock(mutated)
    first = confirmation.run_locked_confirmation(
        data,
        first_lock,
        lock_file_sha256="a" * 64,
        input_identity=_input_identity(data),
        source_identity=first_lock["execution_source_identity"],
    )
    second = confirmation.run_locked_confirmation(
        mutated,
        second_lock,
        lock_file_sha256="b" * 64,
        input_identity=_input_identity(mutated),
        source_identity=second_lock["execution_source_identity"],
    )
    first_rankings = [
        row["ordered_original_indices_sha256"]
        for row in first["ranking_records"]
        if row["seed"] == confirmation.CONFIRMATION_SEEDS[0]
    ]
    second_rankings = [
        row["ordered_original_indices_sha256"]
        for row in second["ranking_records"]
        if row["seed"] == confirmation.CONFIRMATION_SEEDS[0]
    ]
    assert first_rankings == second_rankings


def test_adversarial_confusion_metric_input_and_coordinate_tampering_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, lock, _ = _payload(monkeypatch)
    negative = json.loads(json.dumps(payload))
    negative["records"][0]["confusion_matrix_counts"][0][0] = -1
    _resign_payload(negative)
    with pytest.raises(ValueError, match="non-negative integer 5x5"):
        confirmation.validate_confirmation_payload(negative, lock)

    epsilon = json.loads(json.dumps(payload))
    epsilon["records"][0]["balanced_accuracy"] += 1e-15
    _resign_payload(epsilon)
    with pytest.raises(ValueError, match="confusion counts"):
        confirmation.validate_confirmation_payload(epsilon, lock)

    coordinate = json.loads(json.dumps(payload))
    coordinate["registered_coordinates"][0]["component_1"] += 1_000_000.0
    _resign_payload(coordinate)
    with pytest.raises(ValueError, match="coordinates do not match"):
        confirmation.validate_confirmation_payload(coordinate, lock)

    wrong_input = json.loads(json.dumps(payload))
    wrong_input["input_identity"]["population_labels_sha256"] = "f" * 64
    _resign_payload(wrong_input)
    with pytest.raises(ValueError, match="locked prepared input"):
        confirmation.validate_confirmation_payload(wrong_input, lock)


def test_package_requires_lock_source_and_dual_result_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, lock, _ = _payload(monkeypatch)
    output = tmp_path / "confirmation_package"
    confirmation.write_confirmation_package(payload, lock, output)
    manifest = json.loads((output / confirmation.MANIFEST_FILENAME).read_text())
    validated = confirmation.validate_confirmation_package(
        output,
        lock=lock,
        repo_root=REPO_ROOT,
        expected_lock_file_sha256="a" * 64,
        expected_analysis_sha256=payload["analysis_sha256"],
        expected_package_anchor_sha256=manifest["package_anchor_sha256"],
    )
    assert validated["lock_file_anchor_verified"] is True
    assert validated["source_closure_verified"] is True
    assert validated["external_anchors_verified"] is True
    assert validated["package_files"] == 7
    with pytest.raises(ValueError, match="trusted lock-file anchor"):
        confirmation.validate_confirmation_package(
            output,
            lock=lock,
            expected_lock_file_sha256="b" * 64,
        )


def test_package_rejects_manifest_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, lock, _ = _payload(monkeypatch)
    output = tmp_path / "confirmation_package"
    confirmation.write_confirmation_package(payload, lock, output)
    manifest = output / confirmation.MANIFEST_FILENAME
    outside = tmp_path / "outside_manifest.json"
    manifest.rename(outside)
    manifest.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        confirmation.validate_confirmation_package(output, lock=lock)


def test_checked_in_locked_confirmation_is_exact_and_claim_bounded() -> None:
    lock_path = FORMAL_RESULT_ROOT / "lock" / confirmation.LOCK_FILENAME
    lock = confirmation.load_anchored_lock(lock_path, FORMAL_LOCK_SHA256)
    validation = confirmation.validate_confirmation_package(
        FORMAL_RESULT_ROOT / "locked_pca_loading_confirmation_20seeds_v1",
        lock=lock,
        repo_root=(
            FORMAL_RESULT_ROOT
            / "provenance"
            / "confirmation_source_fd120a02"
        ),
        expected_lock_file_sha256=FORMAL_LOCK_SHA256,
        expected_analysis_sha256=FORMAL_ANALYSIS_SHA256,
        expected_package_anchor_sha256=FORMAL_PACKAGE_ANCHOR_SHA256,
    )
    assert validation["overall_gate_passed"] is True
    assert validation["paired_seeds"] == 20
    payload = confirmation._strict_json_load(
        FORMAL_RESULT_ROOT
        / "locked_pca_loading_confirmation_20seeds_v1"
        / confirmation.RESULT_FILENAME
    )
    summaries = {row["metric"]: row for row in payload["metric_summaries"]}
    assert summaries["balanced_accuracy"]["baseline_mean"] == pytest.approx(0.78925)
    assert summaries["balanced_accuracy"]["locked_mean"] == pytest.approx(0.92475)
    assert summaries["AMR_recall"]["locked_mean"] == pytest.approx(0.635)
    assert summaries["SAS_recall"]["locked_mean"] == pytest.approx(0.99875)
    assert payload["protocol"]["cohort_or_population_generalization_interval"] is False

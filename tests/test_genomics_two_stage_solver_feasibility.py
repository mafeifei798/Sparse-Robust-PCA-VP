from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from experiments import genomics_two_stage_solver_feasibility as companion
from experiments import genomics_pca_loading_grid as grid
from rspca.datasets import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_RESULT_ROOT = (
    REPO_ROOT / "results" / "genomics_two_stage_solver_feasibility_2026-08-31"
)
FORMAL_LOCK_SHA256 = (
    "933591628d828eb9b1a9d400c71b0c5e7ee177ab1dcaef1a295d07575769d913"
)
FORMAL_ANALYSIS_SHA256 = (
    "41aef10b9cd80e351e22efe8502c9ae40182e0768e19326dcbded66be605e3f1"
)
FORMAL_PACKAGE_ANCHOR_SHA256 = (
    "638c76d0016e59437c2874b82013af12e908ec0da4dad79c65d002f78797e31f"
)
FORMAL_FAILURE_RECEIPT_SHA256 = (
    "7f45dbb5ae2ee1c91aee5c56ddef72a158753c38dbf18e94f889eefaf9da3adb"
)


def _synthetic_data() -> grid.GenotypeInput:
    rng = np.random.default_rng(9917)
    populations = np.repeat(np.asarray(companion.EXPECTED_POPULATIONS, dtype=str), 200)
    genotypes = rng.binomial(2, 0.35, size=(1000, 400)).astype(float)
    # Give the first four SNPs stable but imperfect population structure.
    for population_index in range(5):
        rows = slice(population_index * 200, (population_index + 1) * 200)
        genotypes[rows, :4] = rng.binomial(
            2,
            np.clip(0.1 + 0.16 * population_index, 0.05, 0.9),
            size=(200, 4),
        )
    return grid.GenotypeInput(
        genotypes=genotypes,
        sample_ids=np.asarray([f"sample_{index:04d}" for index in range(1000)], dtype=str),
        snp_ids=np.asarray([f"snp_{index:04d}" for index in range(400)], dtype=str),
        chromosomes=np.asarray(["20"] * 400, dtype=str),
        population_labels=populations,
    )


def _identity_and_lock() -> tuple[dict[str, object], dict[str, object]]:
    source = companion._source_identity(REPO_ROOT)
    prepared = {
        "samples": 1000,
        "snps": 400,
        "orientation": "samples_by_snps",
        "manifest_population_design": "five_superpopulation",
        "ld_pruned_input_validated_from_external_manifest": True,
        "genotype_npz_sha256": "1" * 64,
    }
    input_identity = {
        **prepared,
        "ld_pruned_input_bound_to_anchored_receipts": True,
    }
    lock = {
        "lock_payload_sha256": "2" * 64,
        "prepared_input_identity": prepared,
        "execution_source_identity": source,
    }
    return input_identity, lock


def _fake_vp(matrix, rank, **kwargs):
    loadings = np.eye(matrix.shape[1], rank, dtype=float)
    return SimpleNamespace(
        A=loadings.copy(),
        B=loadings,
        objectives=[1.0],
        stationarity=[1.0],
        converged=False,
        n_iter=min(7, int(kwargs["max_iter"])),
        status="synthetic fixed-budget",
    )


def _fake_workspace(matrix, rank, **kwargs):
    return SimpleNamespace(X=np.asarray(matrix), rank=rank, kwargs=kwargs)


def _fake_randomized(workspace, **kwargs):
    loadings = np.eye(workspace.X.shape[1], workspace.rank, dtype=float)
    sketch_iterations = min(5, int(kwargs["max_iter"]))
    full_iterations = min(3, int(kwargs["finalize_full_iters"]))
    return SimpleNamespace(
        A=loadings.copy(),
        B=loadings,
        full_objective=1.0,
        full_stationarity=1.0,
        full_converged=False,
        n_iter=sketch_iterations + full_iterations,
        status="synthetic fixed-budget",
        sketch_iterations=sketch_iterations,
        full_refinement_iterations=full_iterations,
        timings={"optimization": 0.0},
        sketch_residual_relative=0.01,
    )


@pytest.fixture(scope="module")
def development_bundle(tmp_path_factory):
    old = {name: os.environ.get(name) for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )}
    for name in old:
        os.environ[name] = "1"
    data = _synthetic_data()
    input_identity, lock = _identity_and_lock()
    with (
        patch.object(companion, "fit_vp_spca", side_effect=_fake_vp),
        patch.object(companion, "prepare_randomized_vp_spca_workspace", side_effect=_fake_workspace),
        patch.object(companion, "fit_randomized_vp_spca_from_workspace", side_effect=_fake_randomized),
    ):
        payload = companion.evaluate_solver_experiment(
            data,
            phase="feasibility",
            seeds=companion.DEVELOPMENT_SEEDS,
            budgets=companion.BUDGETS,
            lock=lock,
            lock_file_sha256="3" * 64,
            input_identity=input_identity,
            source_identity=lock["execution_source_identity"],
        )
    output = tmp_path_factory.mktemp("two_stage_negative") / "package"
    validation = companion.write_solver_package(
        payload,
        output,
        lock=lock,
        expected_lock_file_sha256="3" * 64,
        repo_root=REPO_ROOT,
    )
    yield payload, lock, output, validation
    for name, value in old.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_protocol_freezes_controlled_two_stage_semantics() -> None:
    protocol = companion.FROZEN_PROTOCOL
    assert protocol["stage_one"] == "outer_train_only_rank4_pca_loading_top400"
    assert protocol["randomized_budget_allocation"] == "half_sketch_half_full_refinement"
    assert protocol["budget_semantics"] == "maximum_total_accepted_proximal_steps_sketch_plus_full"
    assert protocol["heldout_truth_use"] == "scoring_and_predeclared_development_ba_gate_only"
    assert protocol["analysis_role"] == "two_stage_controlled_solver_feasibility_not_aim_discovery"
    assert companion.DEVELOPMENT_SEEDS == tuple(range(2025, 2030))
    assert companion.CONFIRMATION_SEEDS == tuple(range(4001, 4021))


def test_development_exact_grid_and_panel_closure(development_bundle) -> None:
    payload, lock, _, _ = development_bundle
    validation = companion.validate_solver_payload(
        payload,
        lock=lock,
        expected_lock_file_sha256="3" * 64,
    )
    assert validation["record_count"] == 25
    assert validation["prediction_row_count"] == 5000
    assert validation["confirmation_permitted"] is False
    for seed in companion.DEVELOPMENT_SEEDS:
        records = [record for record in payload["records"] if record["seed"] == seed]
        assert len(records) == 5
        assert len({record["selected_original_indices_sha256"] for record in records}) == 1
        assert {(record["method"], record["budget"]) for record in records} == {
            ("pca", None),
            ("vp", 500),
            ("vp", 1000),
            ("randomized", 500),
            ("randomized", 1000),
        }


def test_resigned_prediction_tamper_is_rejected(development_bundle) -> None:
    payload, lock, _, _ = development_bundle
    tampered = copy.deepcopy(payload)
    row = tampered["heldout_predictions"][0]
    row["prediction"] = next(
        population for population in companion.EXPECTED_POPULATIONS if population != row["prediction"]
    )
    unsigned = dict(tampered)
    unsigned.pop("analysis_sha256")
    tampered["analysis_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="prediction hash|metrics"):
        companion.validate_solver_payload(
            tampered,
            lock=lock,
            expected_lock_file_sha256="3" * 64,
        )


def test_package_is_externally_anchored_and_exact_cover(development_bundle) -> None:
    payload, lock, output, validation = development_bundle
    assert validation["status"] == "structurally_valid_unanchored"
    anchored = companion.validate_solver_package(
        output,
        lock=lock,
        expected_lock_file_sha256="3" * 64,
        expected_analysis_sha256=payload["analysis_sha256"],
        expected_package_anchor_sha256=validation["package_anchor_sha256"],
        repo_root=REPO_ROOT,
    )
    assert anchored["status"] == "valid_and_externally_anchored"
    assert anchored["source_snapshot_exact_cover_verified"] is True


def test_gate_failure_forbids_confirmation_lock(development_bundle) -> None:
    payload, lock, output, validation = development_bundle
    with pytest.raises(ValueError, match="confirmation lock is forbidden"):
        companion.build_confirmation_decision_lock(
            feasibility_package=output,
            feasibility_lock=lock,
            feasibility_lock_file_sha256="3" * 64,
            expected_feasibility_analysis_sha256=payload["analysis_sha256"],
            expected_feasibility_package_anchor_sha256=validation["package_anchor_sha256"],
            repo_root=REPO_ROOT,
        )


def test_candidate_bound_pass_creates_fresh_seed_decision_lock(
    development_bundle, tmp_path
) -> None:
    payload, lock, _, _ = development_bundle
    passing = copy.deepcopy(payload)
    for record in passing["records"]:
        if record["method"] == "pca":
            continue
        record["full_stationarity_common"] = 0.0
        record["common_stationarity_qualified"] = True
        record["native_full_objective"] = 1.0
        record["native_full_stationarity"] = 0.0
        record["full_objective_common"] = 1.0
        runtime = record["runtime_seconds"]
        runtime["solver_only_seconds"] = 2.0 if record["method"] == "vp" else 1.0
        record["solver_timed_repetitions_seconds"] = [
            runtime["solver_only_seconds"]
        ] * companion.SPARSE_TIMED_REPETITIONS
        runtime["end_to_end_seconds"] = (
            runtime["shared_train_only_screen_seconds"]
            + runtime["common_reference_charged_to_end_to_end_seconds"]
            + runtime["solver_only_seconds"]
            + runtime["common_diagnostic_and_scoring_seconds"]
        )
    passing["summaries"] = companion._summarize_records(
        passing["records"],
        seeds=companion.DEVELOPMENT_SEEDS,
        budgets=companion.BUDGETS,
    )
    passing["matched_quality"] = companion._matched_quality_rows(
        passing["records"],
        seeds=companion.DEVELOPMENT_SEEDS,
        budgets=companion.BUDGETS,
    )
    passing["gate_results"] = companion._eligibility_gates(
        passing["records"],
        passing["matched_quality"],
        seeds=companion.DEVELOPMENT_SEEDS,
        budgets=companion.BUDGETS,
    )
    unsigned = dict(passing)
    unsigned.pop("analysis_sha256")
    passing["analysis_sha256"] = canonical_sha256(unsigned)
    assert passing["gate_results"]["selected_budget"] == 500
    output = tmp_path / "passing_package"
    package_validation = companion.write_solver_package(
        passing,
        output,
        lock=lock,
        expected_lock_file_sha256="3" * 64,
        repo_root=REPO_ROOT,
    )
    decision = companion.build_confirmation_decision_lock(
        feasibility_package=output,
        feasibility_lock=lock,
        feasibility_lock_file_sha256="3" * 64,
        expected_feasibility_analysis_sha256=passing["analysis_sha256"],
        expected_feasibility_package_anchor_sha256=package_validation["package_anchor_sha256"],
        repo_root=REPO_ROOT,
    )
    companion.validate_confirmation_decision_lock(decision)
    assert decision["selected_budget"] == 500
    assert decision["confirmation_seeds"] == list(range(4001, 4021))
    assert decision["selection_receipt"]["confirmation_outcomes_seen_before_lock"] is False


def test_manifest_symlink_is_rejected(development_bundle, tmp_path) -> None:
    _, lock, output, validation = development_bundle
    linked = tmp_path / "linked"
    linked.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        companion.validate_solver_package(
            linked,
            lock=lock,
            expected_lock_file_sha256="3" * 64,
            expected_package_anchor_sha256=validation["package_anchor_sha256"],
        )


def test_confirmation_writer_consumes_actual_decision_lock(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(companion, "CONFIRMATION_SEEDS", (4001,))
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    data = _synthetic_data()
    input_identity, base_lock = _identity_and_lock()
    decision_lock = {
        **base_lock,
        "selected_budget": 500,
    }
    with (
        patch.object(companion, "fit_vp_spca", side_effect=_fake_vp),
        patch.object(companion, "prepare_randomized_vp_spca_workspace", side_effect=_fake_workspace),
        patch.object(companion, "fit_randomized_vp_spca_from_workspace", side_effect=_fake_randomized),
    ):
        payload = companion.evaluate_solver_experiment(
            data,
            phase="confirmation",
            seeds=(4001,),
            budgets=(500,),
            lock=decision_lock,
            lock_file_sha256="4" * 64,
            input_identity=input_identity,
            source_identity=decision_lock["execution_source_identity"],
            decision_lock_file_sha256="4" * 64,
        )
    output = tmp_path / "confirmation_package"
    validation = companion.write_solver_package(
        payload,
        output,
        lock=decision_lock,
        expected_lock_file_sha256="4" * 64,
        repo_root=REPO_ROOT,
    )
    assert validation["phase"] == "confirmation"
    assert validation["record_count"] == 3


def test_checked_in_negative_feasibility_is_anchored_and_stopped() -> None:
    lock = companion.load_anchored_feasibility_lock(
        FORMAL_RESULT_ROOT / "lock" / "two_stage_solver_feasibility_lock_v2.json",
        FORMAL_LOCK_SHA256,
    )
    package = FORMAL_RESULT_ROOT / "dev_5seeds_v1"
    validation = companion.validate_solver_package(
        package,
        lock=lock,
        expected_lock_file_sha256=FORMAL_LOCK_SHA256,
        expected_analysis_sha256=FORMAL_ANALYSIS_SHA256,
        expected_package_anchor_sha256=FORMAL_PACKAGE_ANCHOR_SHA256,
    )
    assert validation["status"] == "valid_and_externally_anchored"
    payload = companion._strict_json_load(package / companion.RESULT_FILENAME)
    assert payload["gate_results"]["selected_budget"] is None
    assert payload["gate_results"]["confirmation_permitted"] is False
    summaries = {
        (row["method"], row["budget"]): row for row in payload["summaries"]
    }
    assert summaries[("pca", None)]["balanced_accuracy_mean"] == pytest.approx(0.929)
    assert summaries[("randomized", 500)]["solver_only_seconds_median"] < summaries[("vp", 500)]["solver_only_seconds_median"]
    assert summaries[("randomized", 1000)]["solver_only_seconds_median"] < summaries[("vp", 1000)]["solver_only_seconds_median"]
    for budget_row in payload["gate_results"]["by_budget"]:
        assert budget_row["candidate_bound_gate_passed"] is False
        assert budget_row["randomized_full_quality_matched_all_splits"] is True
        assert budget_row["at_least_one_sparse_solver_all_splits_qualified"] is False

    receipt_path = FORMAL_RESULT_ROOT / "confirmation_forbidden_receipt.json"
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == FORMAL_FAILURE_RECEIPT_SHA256
    receipt = companion._strict_json_load(receipt_path)
    assert receipt["status"] == "development_gate_failed_confirmation_forbidden"
    assert receipt["confirmation_lock_created"] is False
    assert receipt["confirmation_seeds_executed"] == []

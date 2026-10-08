"""Closed-package and independently anchored replay receipts for own VP panels."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
verify = importlib.import_module("verify_genomics_randomized_vp_selected5")
run = verify.run
PACKAGE = run.BASE / "paired_20seeds"
# Filled only from the successfully completed run and separate trusted replay.
COMPLETION_SHA = "273516a02b9ceb905eab43a055ca0302f98e33e7dd4a7bf23f64e639fcca5ba4"
REPLAY_SHA = "231435721602ff0713f3de38bad28009cf8df69383e3ab4e223de42e99e1a668"


@pytest.fixture(scope="module")
def evidence():
    return verify.validate_structure(PACKAGE, COMPLETION_SHA)


def test_closed_package_full_grid_and_all_runtime_costs(evidence):
    assert len(evidence["records"]) == 80
    assert len(evidence["summary"]["cells"]) == 4
    assert len(evidence["summary"]["contrasts"]) == 3
    assert evidence["summary"]["independent_population_validation"] is False


def test_all_vp_owned_selections_and_stored_native_prediction_readouts(evidence):
    mapping = {(r["seed"], r["method"]): r for r in evidence["records"]}
    cohort = verify.load_arrays(PACKAGE / "cohort.npz")
    for seed in run.SEEDS:
        directory = PACKAGE / f"seed_{seed}"
        pre = verify.load_arrays(directory / "preprocessing.npz")
        train, test = pre["train_indices"], pre["test_indices"]
        for name, penalty in run.METHODS:
            row = mapping[seed, name]
            final = verify.load_arrays(directory / f"{name}-final.npz")
            np.testing.assert_allclose(final["A"].T @ final["A"], np.eye(8), atol=1e-8, rtol=0)
            assert run.shared.readouts(final["train_scores"], final["test_scores"], cohort["labels"], train, test) == row["readouts"]
            if penalty is None:
                assert final["B"].shape == (455102, 8)
                continue
            selection = verify.load_arrays(directory / f"{name}-selection.npz")
            selector = verify.load_arrays(directory / f"{name}-selector.npz")
            importance = np.abs(selector["B"]).sum(axis=1)
            selected = np.sort(np.lexsort((pre["retained_original_indices"], -importance))[:22755])
            np.testing.assert_array_equal(selection["importance"], importance)
            np.testing.assert_array_equal(selection["selected_retained_indices"], selected)
            np.testing.assert_array_equal(selection["selected_original_indices"], pre["retained_original_indices"][selected])
            assert row["selection"]["selector_B_sha256"] == run.array_sha256(selector["B"])
            assert row["selection"]["importance_sha256"] == run.array_sha256(importance)
            assert run.shared.readouts(selector["train_scores"], selector["test_scores"], cohort["labels"], train, test) == row["full_vp_context"]
            for role, model in (("selector", selector), ("final", final)):
                assert row[f"{role}_audit"]["nonzero_entry_fraction"] == float(np.mean(model["B"] != 0))
                assert row[f"{role}_audit"]["nonzero_snp_fraction"] == float(np.mean(np.any(model["B"] != 0, axis=1)))


def test_separate_original_input_replay_receipt(evidence):
    path = run.BASE / "original_input_replay.json"
    assert run.digest(path) == REPLAY_SHA
    receipt = run.shared.formal._strict_json_load(path)
    assert receipt["status"] == "PASS_ORIGINAL_INPUT_SELECTION_AND_PROJECTION_REPLAY"
    assert receipt["completion_sha256"] == COMPLETION_SHA
    assert receipt["verifier_sha256"] == run.digest(ROOT / "experiments/verify_genomics_randomized_vp_selected5.py")
    assert receipt["trusted_original_input_sha256"] == run.shared.formal.FROZEN_FORMAL_NPZ_SHA256
    assert receipt["full_model_projection_replays"] == 80
    assert receipt["own_vp_panel_replays"] == 60
    assert receipt["heldout_prediction_replays_both_readouts"] == 56000
    assert receipt["optimizer_refit_verified"] is False
    assert run.digest(run.shared.INPUT) == receipt["trusted_original_input_sha256"]


def test_bad_external_anchor_fails_before_self_signed_claims():
    with pytest.raises(ValueError, match="external completion anchor mismatch"):
        verify.validate_structure(PACKAGE, "0" * 64)

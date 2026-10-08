"""Pre-fit tests for actual randomized VP-owned feature selection."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
run = importlib.import_module("genomics_randomized_vp_selected5")


def toy():
    rng = np.random.default_rng(916)
    X = rng.normal(size=(40, 80))
    X -= X.mean(axis=0)
    XT = rng.normal(size=(10, 80))
    X.setflags(write=False)
    XT.setflags(write=False)
    return X, XT


def test_fixed_grid_and_source_identity():
    run.shared.verify_sources()
    assert run.SEEDS == tuple(range(2025, 2045))
    assert run.RANK == 8 and run.PANEL_SIZE == 22755
    assert run.PENALTIES == (.0002, .001, .005)
    assert run.REPEATS == 3
    assert run.METHODS[0] == ("PCA-default-all", None)
    source = (ROOT / "experiments/genomics_randomized_vp_selected5.py").read_text()
    for forbidden in ("_selected_mask(", "fit_chunked_top_pca(", "_rank_select_numeric(", "full_fit_from_pca"):
        assert forbidden not in source


def test_vp_owns_both_fits_and_topk_without_pca(monkeypatch):
    X, XT = toy()
    calls = []
    actual = run.shared.fit_method

    def checked(matrix, seed, penalty):
        assert penalty is not None, "PCA cannot be called by a VP arm"
        calls.append((matrix.shape, seed, penalty))
        return actual(matrix, seed, penalty)

    monkeypatch.setattr(run.shared, "fit_method", checked)
    result = run.fit_select_refit(X, XT, 2025, .001, panel_size=16)
    assert calls == [((40, 80), 2025, .001), ((40, 16), 2025, .001)]
    B = result["selector"]["B"]
    importance = np.abs(B).sum(axis=1)
    expected = np.sort(np.lexsort((np.arange(80), -importance))[:16])
    np.testing.assert_array_equal(result["selected"], expected)
    np.testing.assert_array_equal(result["importance"], importance)
    assert result["final"]["B"].shape == (16, 8)
    np.testing.assert_allclose(result["final"]["train_scores"], X[:, expected] @ result["final"]["B"], atol=1e-12)
    np.testing.assert_allclose(result["final"]["test_scores"], XT[:, expected] @ result["final"]["B"], atol=1e-12)
    assert all(result["stage_seconds"][key] > 0 for key in run.STAGES)
    assert result["pipeline_seconds"] >= sum(result["stage_seconds"].values())


def test_heldout_values_cannot_change_selection_or_either_fit():
    X, XT = toy()
    first = run.fit_select_refit(X, XT, 2025, .005, panel_size=16)
    changed = run.fit_select_refit(X, XT * 19 + 5, 2025, .005, panel_size=16)
    np.testing.assert_array_equal(first["selected"], changed["selected"])
    np.testing.assert_array_equal(first["importance"], changed["importance"])
    for role in ("selector", "final"):
        for key in ("A", "B", "mean"):
            np.testing.assert_array_equal(first[role][key], changed[role][key])
    np.testing.assert_array_equal(first["final"]["train_scores"], changed["final"]["train_scores"])
    assert not np.allclose(first["final"]["test_scores"], changed["final"]["test_scores"])


def test_default_pca_uses_full_input_and_untouched_defaults():
    X, XT = toy()
    result = run.fit_select_refit(X, XT, 2025, None, panel_size=16)
    fit = result["final"]
    model = PCA(n_components=8, random_state=2025).fit(X)
    assert fit["solver"]["parameters"] == model.get_params()
    assert result["selector"] is result["selected"] is result["importance"] is None
    assert fit["B"].shape == (80, 8)
    np.testing.assert_allclose(fit["B"], model.components_.T, atol=1e-12)
    np.testing.assert_allclose(fit["test_scores"], model.transform(XT), atol=1e-12)
    assert result["stage_seconds"]["ranking"] == result["stage_seconds"]["panel_refit"] == 0


def test_ties_are_selected_in_original_index_order():
    B = np.ones((12, 8))
    selected, _ = run.select_loading_panel(B, 8, score="l1")
    np.testing.assert_array_equal(selected, np.arange(8))


def test_complete_toy_seed_and_cost_accounting(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "PANEL_SIZE", 16)
    X, XT = toy()
    labels = np.tile(run.POPULATIONS, 10)
    train, test = np.arange(40), np.arange(40, 50)
    fits, records = run.run_seed(X, XT, labels, train, test, 2025, tmp_path, np.arange(80), 1.25)
    assert len(records) == 4
    for record in records:
        assert len(record["timings"]) == 3
        expected = np.median([t["model_computation"] for t in record["timings"]])
        assert record["discovery_workflow_seconds"] == 1.25 + expected
        assert record["audited_discovery_seconds"] >= record["discovery_workflow_seconds"]
        if record["penalty_fraction"] is not None:
            assert record["selection"]["count"] == record["final_input_snps"] == 16
            assert record["selector_audit"]["full_eta"] is not None
            assert record["final_audit"]["full_eta"] is not None
            assert record["full_vp_context"] is not None
            with np.load(tmp_path / f'{record["method"]}-selection.npz', allow_pickle=False) as selected:
                np.testing.assert_array_equal(selected["selected_retained_indices"], fits[record["method"]]["selected"])
    schedule = run.shared.formal._strict_json_load(tmp_path / "timing_schedule.json")
    assert len(schedule) == 4 and schedule[0]["warmup"]
    run.render_native(fits, labels, train, test, 2025, tmp_path)
    from PIL import Image
    for pair in (12, 34, 56, 78):
        with Image.open(tmp_path / f"pc{pair}.png") as picture:
            assert picture.size == (3080, 2332)
            picture.verify()


def test_incomplete_grid_is_not_summarized_and_bootstrap_is_fixed():
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        run.summarize([])
    assert run.interval([2., 2., 2.], resamples=20) == {"estimate": 2., "ci95": [2., 2.]}
    assert run.interval([1., 3., 7.], resamples=100) == run.interval([1., 3., 7.], resamples=100)

"""Pre-fit contracts for the additive, frozen 5%-panel comparison."""

import copy
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
run = importlib.import_module("genomics_randomized_vp_5pct")


def test_fixed_grid_and_previous_source_hashes():
    run.verify_sources()
    assert run.SEEDS == tuple(range(2025, 2045))
    assert run.RANK == 8 and run.PANEL_SIZE == 22755
    assert run.PENALTIES == (.0002, .001, .005)
    assert len(run.METHODS) == 4 and run.REPEATS == 3
    assert run.REFERENCE.name == "registered_20seeds_merged"


def test_default_pca_parameters_and_mean_projection():
    rng = np.random.default_rng(916)
    X = rng.normal(size=(30, 40)) + np.arange(40)
    XT = rng.normal(size=(5, 40)) + np.arange(40)
    before = X.copy()
    model, A, B, mean = run.fit_method(X, 2025, None)
    assert model.get_params() == PCA(n_components=8, random_state=2025).get_params()
    np.testing.assert_array_equal(A, B)
    np.testing.assert_array_equal(X, before)
    train, test = run.project(X, XT, B, mean)
    np.testing.assert_allclose(train, model.transform(X), atol=1e-12)
    np.testing.assert_allclose(test, model.transform(XT), atol=1e-12)
    assert np.count_nonzero(B) == B.size


def test_randomized_settings_not_adapted_to_panel_scores(monkeypatch):
    seen = {}

    class Fake:
        A = np.ones((40, 8))
        B = np.full((40, 8), .5)

    def fit(X, rank, **kwargs):
        seen.update(rank=rank, shape=X.shape, **kwargs)
        return Fake()

    monkeypatch.setattr(run, "fit_fast_randomized_vp_spca", fit)
    _, A, B, mean = run.fit_method(np.ones((20, 40)), 2025, .001)
    assert seen == {"rank": 8, "shape": (20, 40), "penalty_fraction": .001,
        "oversample": 10, "n_power_iter": 4, "max_iter": 50,
        "relative_tolerance": .001, "check_every": 10, "random_state": 2025}
    assert A.shape == B.shape == (40, 8)
    np.testing.assert_array_equal(mean, np.zeros(40))


def test_panel_preprocessing_uses_only_training_rows():
    rng = np.random.default_rng(791)
    G = rng.integers(0, 3, size=(50, 40)).astype(float)
    train, test = np.arange(40), np.arange(40, 50)
    selected = np.array([2, 3, 5, 7, 11, 14, 21, 31, 37])
    original = run.prepare_panel(G, train, test, selected)
    changed = G.copy()
    changed[test] = 2
    perturbed = run.prepare_panel(changed, train, test, selected)
    np.testing.assert_array_equal(original[1], perturbed[1])
    np.testing.assert_array_equal(original[3].mean_dosage, perturbed[3].mean_dosage)
    np.testing.assert_array_equal(original[0], G[:, selected])
    assert not original[1].flags.writeable and not original[2].flags.writeable


def test_sealed_panel_mask_rejects_mutation():
    payload = run.formal._strict_json_load(run.REFERENCE / run.formal.RESULT_FILENAME)
    case = run.historical._reference_case(payload, seed=2025, width=455102)
    mask = run.historical._selected_mask(case)
    assert mask.sum() == 22755
    bad = copy.deepcopy(case)
    bad["selected_original_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="panel mask"):
        run.historical._selected_mask(bad)


def test_bootstrap_is_deterministic_and_grid_fail_closed():
    assert run.interval([1., 1., 1.], resamples=64) == {"estimate": 1., "ci95": [1., 1.]}
    assert run.interval([1., 3., 7.], statistic="median", resamples=64) == run.interval([1., 3., 7.], statistic="median", resamples=64)
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        run.summarize([])


def test_toy_paired_runtime_repeats_and_saved_scores(tmp_path):
    rng = np.random.default_rng(2321)
    G = rng.integers(0, 3, size=(50, 40)).astype(float)
    labels = np.repeat(run.POPULATIONS, 10)
    train = np.concatenate([np.arange(i*10, i*10+8) for i in range(5)])
    test = np.setdiff1d(np.arange(50), train)
    _, X, XT, _, _, preprocessing = run.prepare_panel(G, train, test, np.arange(40))
    fits, records = run.run_panel(X, XT, labels, train, test, 2025, tmp_path,
        {"preprocessing_seconds": preprocessing, "screen_seconds": 0., "materialization_seconds": 0.})
    assert len(records) == 4
    for row in records:
        assert len(row["fit_seconds"]) == len(row["fit_cpu_seconds"]) == 3
        assert row["discovery_workflow_seconds"] >= row["fit_seconds_median"]
        with np.load(tmp_path / f'{row["method"]}.npz', allow_pickle=False) as saved:
            train_scores, test_scores = run.project(X, XT, saved["B"], saved["mean"])
            np.testing.assert_allclose(train_scores, saved["train_scores"], atol=1e-12)
            np.testing.assert_allclose(test_scores, saved["test_scores"], atol=1e-12)
            assert run.readouts(train_scores, test_scores, labels, train, test) == row["readouts"]
    assert list(fits) != []

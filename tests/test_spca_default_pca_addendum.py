"""Unit checks for the ordinary-PCA addendum, not a timing benchmark."""

import importlib
from pathlib import Path
import sys
import unittest

import numpy as np
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
runner = importlib.import_module("spca_default_pca_addendum")


class OrdinaryPCAUnitTests(unittest.TestCase):
    def test_historical_sources_and_alpha_locks_are_exact_and_isolated(self):
        original = sys.modules.get("rspca.vp")
        parents = {str((ROOT/relative).parent) for relative, _ in runner.SOURCE_BINDINGS.values()}
        before = {p for parent in parents for p in Path(parent).rglob("*.pyc")}
        with runner.historical_modules() as modules:
            self.assertIn("video_literal_2026-08-30_execution_source", modules.vp.__file__)
            self.assertTrue(modules.accelerated.__name__.startswith("_ordinary_pca_historical_rspca."))
            self.assertIs(sys.modules.get("rspca.vp"), original)
            history = runner.load_history()
            for regime in ("clean", "corrupt"):
                rows = history[regime]["records"]
                for seed in history[regime]["seed_manifest"]["seeds"]:
                    for method in runner.SPARSE:
                        found = [r for r in rows if r["seed"] == seed and r["method"] == method]
                        self.assertEqual(len(found), 1)
                        self.assertIn(found[0]["alpha"], history[regime]["config"]["alphas"])
        self.assertFalse(any(n.startswith("_ordinary_pca_historical_rspca") for n in sys.modules))
        after = {p for parent in parents for p in Path(parent).rglob("*.pyc")}
        self.assertEqual(before, after, "Historical imports must not create bytecode in sealed snapshots")

    def test_default_pca_is_unthresholded_and_reconstruction_includes_mean(self):
        X = np.random.default_rng(13).normal(size=(75, 30)) + np.arange(30)
        original = X.copy()
        X.setflags(write=False)
        fit = runner.fit_method("pca_default", X, {"rank": 5}, None, 13, None)
        reference = PCA(n_components=5).fit(X)
        np.testing.assert_array_equal(fit["B"], reference.components_.T)
        np.testing.assert_allclose(fit["reconstruction"], reference.inverse_transform(reference.transform(X)), atol=1e-12)
        np.testing.assert_allclose(fit["mean"], X.mean(axis=0), atol=1e-12)
        self.assertEqual(fit["diagnostic"]["parameters"], reference.get_params())
        self.assertEqual(np.count_nonzero(fit["B"]), fit["B"].size)
        np.testing.assert_array_equal(X, original)

    def test_supplementary_threshold_changes_only_its_own_control(self):
        B = np.arange(1, 101, dtype=float).reshape(20, 5)
        original = B.copy()
        thresholded = runner.threshold_loadings(B, .85)
        self.assertEqual(np.count_nonzero(thresholded), 15)
        np.testing.assert_array_equal(B, original)
        X = np.random.default_rng(7).normal(size=(80, 30))
        config = {"rank": 5, "target_sparsity": .85}
        with runner.historical_modules() as legacy:
            for method in runner.METHODS[1:3]:
                fit = runner.fit_method(method, X, config, None, 7, legacy)
                np.testing.assert_allclose(fit["mean"], 0)
                np.testing.assert_allclose(fit["reconstruction"], X @ fit["B"] @ fit["A"].T)
                np.testing.assert_allclose(fit["A"].T @ fit["A"], np.eye(5), atol=1e-12)

    def test_reproduction_guard_preserves_failures(self):
        record = dict(regime="corrupt", seed=7, method="accelerated_vp_spca", alpha=.2,
            subspace_error=.2, relative_reconstruction_error=.4, achieved_sparsity=.85,
            diagnostic={"converged": True})
        original = {**record, "converged": True}
        self.assertEqual(runner.reproduction_check(record, original)["status"], "passed")
        changed = {**record, "subspace_error": .201}
        self.assertEqual(runner.reproduction_check(changed, original)["status"], "failed")
        changed = {**record, "alpha": .3}
        self.assertEqual(runner.reproduction_check(changed, original)["status"], "failed")

    def test_all_legacy_methods_accept_original_fixed_options_and_metrics(self):
        config = dict(runner.load_history()["corrupt"]["config"])
        config.update(n_samples=60, n_features=25, rank=3, max_iter=4,
                      randomized_finalize_full_iters=2, certified_full_refine_iters=4)
        with runner.historical_modules() as legacy:
            data = legacy.datasets.generate_sparse_low_rank(**{k: config[k] for k in runner.DATA_KEYS}, seed=27)
            for method in runner.METHODS:
                fit = runner.fit_method(method, data.observed, config, .2 if method in runner.SPARSE else None, 27, legacy)
                values = runner.evaluate(data, fit, legacy)
                self.assertEqual(set(values), set(runner.METRICS))
                self.assertTrue(all(np.isfinite(v) for v in values.values()))
                self.assertEqual(fit["B"].shape, (25, 3))


if __name__ == "__main__":
    unittest.main()

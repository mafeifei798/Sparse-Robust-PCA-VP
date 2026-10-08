"""Replay saved paired ordinary-PCA evidence; never regenerate historical numerators."""

import hashlib
import importlib
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
runner = importlib.import_module("spca_default_pca_addendum")
BASE = ROOT/"results/spca_default_pca_addendum_2026-09-06"
PACKAGE = BASE/"paired_replay"


class OrdinaryPCAArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = runner.strict_json(PACKAGE/"records.json")
        cls.summary = runner.strict_json(PACKAGE/"summary.json")
        cls.spec = runner.strict_json(PACKAGE/"specification_before_run.json")

    def test_complete_artifact_hashes_strict_json_and_exact_source_archive(self):
        self.assertEqual(runner.digest(PACKAGE/"completion.json"),
                         "0de0e590e83ac8c50e4d993dbc43af18777d4dbd62dd19404da9aa11116ff095")
        receipt = runner.strict_json(PACKAGE/"completion.json")
        self.assertEqual((receipt["records"], receipt["datasets"], receipt["reproduction_checks"]), (168, 24, 120))
        actual = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*")
                  if p.is_file() and p != PACKAGE/"completion.json"}
        self.assertEqual(set(receipt["files"]), actual)
        self.assertNotIn("FAILED.json", actual)
        for relative, expected in receipt["files"].items():
            self.assertEqual(runner.digest(PACKAGE/relative), expected)
            if relative.endswith(".json"):
                runner.strict_json(PACKAGE/relative)
        self.assertEqual(self.spec["protocol_sha256"], runner.digest(BASE/"PROTOCOL.md"))
        with zipfile.ZipFile(PACKAGE/"source_snapshot.zip") as archive:
            for name, (_, expected) in runner.SOURCE_BINDINGS.items():
                self.assertEqual(hashlib.sha256(archive.read(f"historical_rspca/{name}.py")).hexdigest(), expected)
                self.assertEqual(self.spec["source_sha256"][name], expected)
            for regime, expected in runner.HISTORICAL_RESULT_SHA256.items():
                self.assertEqual(hashlib.sha256(archive.read(f"history/{regime}.json")).hexdigest(), expected)
            self.assertEqual(hashlib.sha256(archive.read("spca_default_pca_addendum.py")).hexdigest(), self.spec["runner_sha256"])
            self.assertEqual(hashlib.sha256(archive.read("PROTOCOL.md")).hexdigest(), self.spec["protocol_sha256"])

    def test_complete_fixed_grid_and_alpha_choices(self):
        history = runner.load_history()
        expected = {(regime, seed, method) for regime, h in history.items()
                    for seed in h["seed_manifest"]["seeds"] for method in runner.METHODS}
        self.assertEqual(len(self.records), len(expected))
        self.assertEqual({(r["regime"], r["seed"], r["method"]) for r in self.records}, expected)
        for regime, h in history.items():
            self.assertEqual(self.spec["regimes"][regime]["config"], h["config"])
            locks = {(r["seed"], r["method"]): r["alpha"] for r in self.spec["regimes"][regime]["alpha_locks"]}
            old = {(r["seed"], r["method"]): r["alpha"] for r in h["records"]}
            self.assertEqual(len(locks), 48)
            for record in self.records:
                if record["regime"] != regime:
                    continue
                key = record["seed"], record["method"]
                if record["method"] in runner.SPARSE:
                    self.assertEqual(record["alpha"], locks[key])
                    self.assertEqual(record["alpha"], old[key])
                else:
                    self.assertIsNone(record["alpha"])

    def test_every_input_fit_score_reconstruction_and_metric_replays(self):
        with runner.historical_modules() as legacy:
            for regime, config in self.spec["regimes"].items():
                for seed in config["seeds"]:
                    directory = PACKAGE/regime/f"seed_{seed}"
                    with np.load(directory/"input.npz", allow_pickle=False) as saved:
                        inputs = {name: saved[name] for name in saved.files}
                    data = SimpleNamespace(**inputs)
                    identity = runner.strict_json(directory/"input_identity.json")
                    self.assertEqual(data.observed.shape, (400, 200))
                    self.assertEqual(data.loadings.shape, (200, 5))
                    for name, value in inputs.items():
                        self.assertEqual(legacy.datasets.array_sha256(value), identity["arrays_sha256"][name])
                    np.testing.assert_allclose(data.low_rank+data.gaussian_noise+data.corruption,
                                               data.observed, atol=1e-12, rtol=1e-12)
                    for r in self.records:
                        if (r["regime"], r["seed"]) != (regime, seed):
                            continue
                        self.assertEqual(r["input_sha256"], identity["arrays_sha256"]["observed"])
                        with np.load(directory/f"{r['method']}.npz", allow_pickle=False) as saved:
                            fit = {k: saved[k] for k in saved.files}
                        self.assertEqual(fit["A"].shape, (200, 5))
                        self.assertEqual(fit["B"].shape, (200, 5))
                        np.testing.assert_allclose(fit["A"].T @ fit["A"], np.eye(5), atol=1e-10, rtol=1e-10)
                        scores = (data.observed-fit["mean"]) @ fit["B"]
                        np.testing.assert_allclose(scores, fit["scores"], atol=1e-10, rtol=1e-10)
                        fit["reconstruction"] = scores @ fit["A"].T + fit["mean"]
                        replay = runner.evaluate(data, fit, legacy)
                        for metric, value in replay.items():
                            self.assertAlmostEqual(value, r[metric], delta=1e-10)

    def test_primary_control_is_literal_default_pca_and_not_thresholded(self):
        for r in self.records:
            if r["method"] != "pca_default":
                continue
            self.assertEqual(r["diagnostic"]["parameters"], dict(copy=True, iterated_power="auto",
                n_components=5, n_oversamples=10, power_iteration_normalizer="auto", random_state=None,
                svd_solver="auto", tol=0.0, whiten=False))
            self.assertEqual(r["diagnostic"]["resolved_solver"], "full")
            self.assertEqual(r["achieved_sparsity"], 0)
            directory = PACKAGE/r["regime"]/f"seed_{r['seed']}"
            with np.load(directory/"input.npz", allow_pickle=False) as data:
                mean = data["observed"].mean(axis=0)
            with np.load(directory/"pca_default.npz", allow_pickle=False) as fit:
                np.testing.assert_allclose(fit["mean"], mean, atol=1e-12, rtol=1e-12)
                self.assertEqual(np.count_nonzero(fit["B"]), 1000)

    def test_reproduction_and_historical_byte_boundary_are_explicit(self):
        self.assertFalse(self.spec["historical_byte_identity_claim"])
        identities = runner.strict_json(PACKAGE/"input_identities.json")
        self.assertEqual(len(identities), 24)
        receipt = runner.strict_json(PACKAGE/"completion.json")
        self.assertEqual(sum(i["observed_matches_historical"] for i in identities), receipt["historical_byte_matches"])
        history = runner.load_history()
        index = {(r["regime"], r["seed"], r["method"]): r for r in self.records}
        old = {(regime, r["seed"], r["method"]): r for regime, h in history.items() for r in h["records"]}
        guards = runner.strict_json(PACKAGE/"reproduction.json")
        self.assertEqual(len(guards), 120)
        for guard in guards:
            key = guard["regime"], guard["seed"], guard["method"]
            self.assertEqual(runner.reproduction_check(index[key], old[key]), guard)
            self.assertEqual(guard["status"], "passed")

    def test_every_paired_mean_and_convergence_claim_replays(self):
        self.assertEqual(len(self.summary["cells"]), 14)
        self.assertEqual(len(self.summary["contrasts"]), 36)
        self.assertEqual(self.summary["bootstrap_resamples"], 100000)
        self.assertEqual(self.summary["bootstrap_seed"], 260826)
        self.assertFalse(self.summary["runtime_in_bootstrap"])
        index = {(r["regime"], r["seed"], r["method"]): r for r in self.records}
        for contrast in self.summary["contrasts"]:
            regime, method, baseline = contrast["regime"], contrast["method"], contrast["baseline"]
            a = [index[regime, s, method] for s in contrast["seeds"]]
            b = [index[regime, s, baseline] for s in contrast["seeds"]]
            self.assertEqual(contrast["all_pairs_converged"], all(r["diagnostic"]["converged"] for r in a+b))
            for metric, values in contrast["metrics"].items():
                lhs, rhs = np.array([r[metric] for r in a]), np.array([r[metric] for r in b])
                np.testing.assert_allclose(lhs-rhs, values["paired_differences"], atol=1e-12, rtol=1e-12)
                self.assertAlmostEqual(lhs.mean(), values["method_mean"], delta=1e-12)
                self.assertAlmostEqual(rhs.mean(), values["baseline_mean"], delta=1e-12)
                if metric != "support_f1":
                    self.assertAlmostEqual(100*(1-lhs.mean()/rhs.mean()), values["reduction_percent"], delta=1e-10)
        for c in self.summary["cells"]:
            expected = 0 if c["method"] in ("vp_spca", "randomized_vp_spca") else 12
            self.assertEqual(c["converged_count"], expected)

    def test_all_bootstrap_intervals_replay(self):
        replay = runner.summarize(self.records)
        for new, old in zip(replay["contrasts"], self.summary["contrasts"], strict=True):
            for metric in new["metrics"]:
                for field in ("paired_difference_ci95", "reduction_percent_ci95"):
                    if field in new["metrics"][metric]:
                        np.testing.assert_allclose(new["metrics"][metric][field], old["metrics"][metric][field],
                                                   atol=1e-10, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()

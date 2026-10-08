"""Read-only replay of the paired development package, including failures."""

import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results/genomics_randomized_vp_matched_2026-09-05/development_rank4_budget50"
spec = importlib.util.spec_from_file_location("matched_runner", ROOT / "experiments/genomics_randomized_vp_matched.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MatchedGenomicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads((PACKAGE / "results.json").read_text())
        cls.protocol = json.loads((PACKAGE / "protocol_before_run.json").read_text())

    def test_completion_binds_every_artifact(self):
        completion = json.loads((PACKAGE / "completion.json").read_text())
        self.assertEqual(completion["status"], "development_complete")
        self.assertEqual(completion["records"], 30)
        for name, expected in completion["files"].items():
            with (PACKAGE / name).open("rb") as handle:
                observed = hashlib.file_digest(handle, "sha256").hexdigest()
            self.assertEqual(observed, expected, name)
        self.assertEqual(self.payload["protocol_sha256"], runner.digest(PACKAGE / "protocol_before_run.json"))

    def test_archived_sources_match_pre_run_hashes(self):
        with zipfile.ZipFile(PACKAGE / "source_snapshot.zip") as source:
            self.assertEqual(set(source.namelist()), set(self.protocol["source_sha256"]))
            for name, expected in self.protocol["source_sha256"].items():
                self.assertEqual(hashlib.sha256(source.read(name)).hexdigest(), expected)

    def test_grid_and_no_convergence_overclaim(self):
        self.assertEqual(self.protocol["status"], "exposed_seed_development_not_confirmation")
        self.assertEqual(self.protocol["penalty_fractions"], [.0002, .001, .005])
        self.assertEqual(len(self.payload["records"]), 30)
        identities = {(r["seed"], r["method"]) for r in self.payload["records"]}
        self.assertEqual(len(identities), 30)
        for record in self.payload["records"]:
            self.assertEqual(record["input_snps"], 27346)
            self.assertEqual(record["training_retained_snps"], 27346)
            self.assertEqual(len(record["fit_seconds"]), 3)
            if record["method"].startswith("RVP"):
                self.assertGreater(record["solver"]["alpha"], 0)
                self.assertFalse(record["solver"]["sketch_converged"])
                self.assertFalse(record["full_stationarity_qualified"])
                self.assertEqual(record["solver"]["iterations"], 50)
                self.assertTrue(np.all(np.diff(record["solver"]["sketch_objectives_scaled"]) <= 1e-10))

    def test_all_splits_and_native_score_readouts_replay(self):
        split_by_seed = {}
        for record in self.payload["records"]:
            seed, method = record["seed"], record["method"]
            with np.load(PACKAGE / f"seed_{seed}_{method}_scores.npz", allow_pickle=False) as arrays:
                train, test = arrays["train_indices"], arrays["test_indices"]
                self.assertEqual(len(np.intersect1d(train, test)), 0)
                self.assertEqual((len(train), len(test)), (800, 200))
                self.assertEqual(len(np.unique(np.r_[train, test])), 1000)
                self.assertEqual(arrays["B"].shape, (27346, 4))
                for name in ("B", "A", "train_scores", "test_scores"):
                    self.assertTrue(np.isfinite(arrays[name]).all())
                if seed in split_by_seed:
                    np.testing.assert_array_equal(train, split_by_seed[seed][0])
                    np.testing.assert_array_equal(test, split_by_seed[seed][1])
                else:
                    split_by_seed[seed] = train.copy(), test.copy()
                for standardize, key in ((False, "raw_centroid"), (True, "std_centroid")):
                    replay = runner.metrics(arrays["train_scores"], arrays["test_scores"],
                                            arrays["labels"][train], arrays["labels"][test],
                                            standardize=standardize)
                    self.assertEqual(replay, record[key])
                self.assertEqual(float(np.mean(arrays["B"] != 0)), record["nonzero_entry_fraction"])

    def test_failed_first_pilot_not_promoted(self):
        failed = PACKAGE.parent / "pilot_rank4"
        self.assertTrue((failed / "INVALID.md").is_file())
        self.assertFalse((failed / "completion.json").exists())


if __name__ == "__main__":
    unittest.main()

"""Artifact, split, readout and claim-boundary replay; no model refitting."""

import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
runner = importlib.import_module("genomics_randomized_vp_matched")
grid = importlib.import_module("genomics_randomized_vp_amr_grid")
BASE = ROOT / "results/genomics_randomized_vp_amr_rankgrid_2026-09-05"
PACKAGES = [
    ("openblas_dense_grid", 2, 27346, 5, 3),
    ("openblas_dense_grid_q4", 4, 27346, 5, 3),
    ("all_autosome_blocked_seed2025", 2, 455102, 1, 1),
    ("all_autosome_blocked_q4_seed2025", 4, 455102, 1, 1),
]


def load(path):
    def reject(value):
        raise ValueError(f"nonstandard JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


class AMRDevelopmentResultTests(unittest.TestCase):
    def test_complete_hash_manifests_and_strict_json(self):
        for name, *_ in PACKAGES:
            package = BASE / name
            manifest = load(package / "completion.json")
            actual = {str(p.relative_to(package)) for p in package.rglob("*")
                      if p.is_file() and p != package / "completion.json"}
            self.assertEqual(set(manifest["files"]), actual)
            self.assertEqual(manifest["cells"], 36)
            for relative, expected in manifest["files"].items():
                self.assertEqual(runner.digest(package / relative), expected, (name, relative))
                if relative.endswith(".json"):
                    load(package / relative)

    def test_source_snapshots_and_protocols_are_bound(self):
        for name, power, width, _, _ in PACKAGES:
            package = BASE / name
            specification = load(package / "specification_before_run.json")
            if "power_iterations" in specification:
                self.assertEqual(specification["power_iterations"], power)
            else:
                # The first sealed orchestrator predates this convenience
                # field; its child pre-fit protocols bind q explicitly below.
                self.assertEqual(name, "openblas_dense_grid")
                self.assertIn("PCA-randomized-q2", specification["methods"])
            self.assertIn("not_confirmation", specification["status"])
            protocol_file = BASE / ("PROTOCOL.md" if power == 2 else "PROTOCOL_Q4_EXTENSION.md")
            self.assertEqual(specification["protocol_sha256"], runner.digest(protocol_file))
            if width == 455102:
                locations = [(package, specification)]
            else:
                locations = [(package / f"rank_{rank}",
                              load(package / f"rank_{rank}/protocol_before_run.json"))
                             for rank in (4, 6, 8)]
            for location, protocol in locations:
                self.assertEqual(protocol["power_iterations"], power)
                with zipfile.ZipFile(location / "source_snapshot.zip") as archive:
                    for relative, expected in protocol["source_sha256"].items():
                        self.assertEqual(hashlib.sha256(archive.read(relative)).hexdigest(), expected)
                result = load(location / "results.json")
                self.assertEqual(result["runtime_warnings"], [])
                pools = result["threadpools"]
                self.assertTrue(pools)
                self.assertTrue(all(pool["num_threads"] == 1 for pool in pools))

    def test_complete_grid_timing_and_aggregate_replay(self):
        for name, power, width, nseeds, repeats in PACKAGES:
            summary = load(BASE / name / "summary.json")
            records = summary["records"]
            self.assertEqual(len(records), 18 * nseeds)
            self.assertEqual(len({(r["rank"], r["seed"], r["method"]) for r in records}), len(records))
            self.assertEqual({r["rank"] for r in records}, {4, 6, 8})
            self.assertEqual({r["seed"] for r in records}, set(range(2025, 2025+nseeds)))
            qlabel = f"PCA-skSVD-q{power}" if width == 455102 else f"PCA-randomized-q{power}"
            self.assertIn(qlabel, {r["method"] for r in records})
            self.assertEqual(grid.summarize(records), summary["cells"])
            for record in records:
                self.assertEqual(record["input_snps"], width)
                self.assertEqual(record["training_retained_snps"], width)
                self.assertEqual(len(record["fit_seconds"]), repeats)
                self.assertTrue(all(np.isfinite(t) and t > 0 for t in record["fit_seconds"]))
                self.assertEqual(float(np.median(record["fit_seconds"])), record["fit_seconds_median"])
                self.assertAlmostEqual(record["workflow_seconds"], record["preprocessing_seconds"]
                                       + record["fit_seconds_median"] + record["projection_seconds_median"])
                self.assertGreaterEqual(record["audited_workflow_seconds"], record["workflow_seconds"])

    def test_every_native_score_readout_and_split_replays(self):
        split_identity = {}
        for name, _, width, _, _ in PACKAGES:
            for record in load(BASE / name / "summary.json")["records"]:
                rank, seed, method = record["rank"], record["seed"], record["method"]
                path = BASE / name / f"rank_{rank}/seed_{seed}_{method}_scores.npz"
                with np.load(path, allow_pickle=False) as arrays:
                    train, test = arrays["train_indices"], arrays["test_indices"]
                    labels = arrays["labels"]
                    self.assertEqual((len(train), len(test)), (800, 200))
                    self.assertEqual(len(np.intersect1d(train, test)), 0)
                    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(1000))
                    key = width, seed
                    identity = arrays["sample_ids"], train, test, labels
                    if key not in split_identity:
                        split_identity[key] = tuple(a.copy() for a in identity)
                    for left, right in zip(identity, split_identity[key], strict=True):
                        np.testing.assert_array_equal(left, right)
                    for population in runner.POPULATIONS:
                        self.assertEqual(int(np.sum(labels[train] == population)), 160)
                        self.assertEqual(int(np.sum(labels[test] == population)), 40)
                    B = arrays["B"]
                    self.assertEqual(B.shape, (width, rank))
                    for field in ("B", "A", "train_scores", "test_scores"):
                        self.assertTrue(np.isfinite(arrays[field]).all())
                    for standardize, readout in ((False, "raw_centroid"), (True, "std_centroid")):
                        replay = runner.metrics(arrays["train_scores"], arrays["test_scores"],
                                                labels[train], labels[test], standardize=standardize)
                        self.assertEqual(replay, record[readout])
                    self.assertEqual(float(np.mean(B != 0)), record["nonzero_entry_fraction"])
                    self.assertEqual(float(np.mean(np.any(B != 0, axis=1))), record["nonzero_snp_fraction"])

    def test_stationarity_and_fixed_budget_boundaries(self):
        for name, power, width, _, _ in PACKAGES:
            for record in load(BASE / name / "summary.json")["records"]:
                if not record["method"].startswith("RVP"):
                    self.assertIsNone(record["full_stationarity_qualified"])
                    continue
                solver = record["solver"]
                self.assertGreater(solver["alpha"], 0)
                self.assertLessEqual(solver["iterations"], 50)
                self.assertTrue(np.all(np.diff(solver["sketch_objectives_scaled"]) <= 1e-10))
                self.assertEqual(record["full_stationarity_qualified"], record["full_eta"] <= .001)
                self.assertEqual(solver["sketch_converged"], solver["sketch_eta"] <= .001)
                if width == 455102:
                    self.assertEqual(solver["blocked_fit_data_passes"], 2 * power + 2)

    def test_original_full_input_training_statistics_and_sampled_projections(self):
        from rspca.genomics_aim import transform_smartpca
        from rspca.genomics_blocked import fit_block_preprocessor
        source = importlib.import_module("genomics_pca_loading_all_autosomes")
        input_dir = ROOT / "results/genomics_all_autosome_preparation_2026-08-31/package"
        with tempfile.TemporaryDirectory(prefix="rspca-amr-projection-replay-") as scratch:
            with source.load_formal_input(
                input_dir / "genomics_aim_all_autosomes.npz",
                input_dir / "genomics_aim_all_autosomes.manifest.json",
                expected_source_manifest_sha256=source.FROZEN_FORMAL_MANIFEST_SHA256,
                scratch_root=Path(scratch),
            ) as (data, _):
                first = BASE / "all_autosome_blocked_seed2025"
                with np.load(first / "seed_2025_preprocessing.npz", allow_pickle=False) as frozen:
                    train, test = frozen["train_indices"], frozen["test_indices"]
                    pre = fit_block_preprocessor(data.genotypes, train)
                    for field in ("mean_dosage", "scale", "retained_snps"):
                        np.testing.assert_array_equal(getattr(pre, field), frozen[field])
                # Four deterministic rows per split, never selected by outcome.
                # This verifies the original-input -> stored-score path without
                # pretending to refit the full scientific model.
                sampled_train = transform_smartpca(data.genotypes[train[:4]], pre)
                sampled_test = transform_smartpca(data.genotypes[test[:4]], pre)
                for name, _, width, _, _ in PACKAGES:
                    if width != 455102:
                        continue
                    for record in load(BASE / name / "summary.json")["records"]:
                        path = BASE / name / f"rank_{record['rank']}/seed_2025_{record['method']}_scores.npz"
                        with np.load(path, allow_pickle=False) as arrays:
                            np.testing.assert_array_equal(arrays["sample_ids"], data.sample_ids)
                            np.testing.assert_array_equal(arrays["labels"], data.population_labels)
                            B = arrays["B"]
                            np.testing.assert_allclose(sampled_train @ B, arrays["train_scores"][:4],
                                                       atol=1e-8, rtol=1e-10)
                            np.testing.assert_allclose(sampled_test @ B, arrays["test_scores"][:4],
                                                       atol=1e-8, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()

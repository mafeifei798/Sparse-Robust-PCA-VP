"""Replay fixed default-PCA runtime evidence without refitting timed models."""

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "tools"))
runtime = importlib.import_module("genomics_randomized_vp_default_runtime")
report = importlib.import_module("summarize_genomics_default_runtime")
BASE = ROOT / "results/genomics_randomized_vp_runtime_2026-09-06"


class DefaultRuntimeEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.packages = {name: report.verify_package(BASE/name, width, nseeds)
                        for name, width, nseeds in report.PACKAGES}

    def test_complete_grid_sources_default_parameters_and_threads(self):
        for name, _, nseeds in report.PACKAGES:
            package = BASE/name
            spec = report.strict_json(package/"specification_before_run.json")
            self.assertEqual(spec["protocol_sha256"], runtime.digest(BASE/"PROTOCOL.md"))
            self.assertIn("not_confirmation", spec["status"])
            self.assertEqual(spec["environment"], dict(python="3.11.13", numpy="1.26.4",
                                                     scipy="1.11.4", sklearn="1.7.0"))
            self.assertEqual(spec["penalty_fractions"], [.0002, .001, .005])
            self.assertEqual([m[0] for m in spec["methods"]], report.METHODS)
            result = report.strict_json(package/"results.json")
            self.assertEqual(result["runtime_warnings"], [])
            pools = result["threadpools"]
            self.assertTrue(pools)
            self.assertTrue(all(p["num_threads"] == 1 for p in pools))
            self.assertEqual(sum(p["internal_api"] == "openblas" for p in pools), 2)
            self.assertEqual(len(self.packages[name]["cells"]), 42)
            self.assertEqual(len(self.packages[name]["records"]), nseeds*21)

    def test_every_native_readout_split_and_sparsity_replays(self):
        identities = {}
        for name, width, _ in report.PACKAGES:
            for record in self.packages[name]["records"]:
                seed, rank, method = record["seed"], record["rank"], record["method"]
                path = BASE/name/f"rank_{rank}/seed_{seed}_{method}_scores.npz"
                with np.load(path, allow_pickle=False) as arrays:
                    train, test, labels = arrays["train_indices"], arrays["test_indices"], arrays["labels"]
                    self.assertEqual((len(train), len(test)), (800, 200))
                    self.assertEqual(len(np.intersect1d(train, test)), 0)
                    np.testing.assert_array_equal(np.sort(np.r_[train, test]), np.arange(1000))
                    identity = arrays["sample_ids"], train, test, labels
                    key = width, seed
                    if key not in identities:
                        identities[key] = tuple(a.copy() for a in identity)
                    for left, right in zip(identity, identities[key], strict=True):
                        np.testing.assert_array_equal(left, right)
                    for pop in runtime.POPULATIONS:
                        self.assertEqual(int(np.sum(labels[train] == pop)), 160)
                        self.assertEqual(int(np.sum(labels[test] == pop)), 40)
                    for field, shape in (("B", (width, rank)), ("A", (width, rank)),
                                         ("train_scores", (800, rank)), ("test_scores", (200, rank))):
                        array = arrays[field]
                        self.assertEqual(array.shape, shape)
                        self.assertEqual(array.dtype, np.float64)
                        self.assertTrue(np.isfinite(array).all())
                    for standardize, view in ((False, "raw_centroid"), (True, "std_centroid")):
                        replay = runtime.metrics(arrays["train_scores"], arrays["test_scores"],
                            labels[train], labels[test], standardize=standardize)
                        self.assertEqual(replay, record[view])
                    B = arrays["B"]
                    self.assertEqual(float(np.mean(B != 0)), record["nonzero_entry_fraction"])
                    self.assertEqual(float(np.mean(np.any(B != 0, axis=1))), record["nonzero_snp_fraction"])

    def test_all_implementation_equivalence_guards_replay(self):
        for name, _, nseeds in report.PACKAGES:
            records = self.packages[name]["records"]
            index = {(r["seed"], r["rank"], r["method"]): r for r in records}
            stored = report.strict_json(BASE/name/"equivalence.json")
            self.assertEqual(len(stored), 9*nseeds)
            for guard in stored:
                fits = []
                for which in ("reference", "optimized"):
                    method = guard[which]
                    path = BASE/name/f"rank_{guard['rank']}/seed_{guard['seed']}_{method}_scores.npz"
                    with np.load(path, allow_pickle=False) as arrays:
                        fit = {field: arrays[field] for field in ("A", "B", "train_scores", "test_scores")}
                    fit["record"] = index[guard["seed"], guard["rank"], method]
                    fits.append(fit)
                replay = runtime.equivalence(*fits)
                for key in ("status", "all_readout_predictions_identical", "nonzero_mask_identical",
                            "seed", "rank", "reference", "optimized"):
                    self.assertEqual(replay[key], guard[key])
                # BLAS versions can change the last bit of norm evaluation,
                # never the frozen numerical tolerance or support/predictions.
                for field in ("A", "B", "train_scores", "test_scores"):
                    key = field + "_relative_frobenius_difference"
                    self.assertAlmostEqual(replay[key], guard[key], delta=1e-12)

    def test_timing_charges_and_fixed_budget_convergence_are_explicit(self):
        for name, _, _ in report.PACKAGES:
            records = self.packages[name]["records"]
            prep = {}
            for r in records:
                for field in ("fit_seconds_median", "projection_seconds_median", "readout_seconds",
                              "preprocessing_seconds", "audit_seconds"):
                    self.assertTrue(np.isfinite(r[field]) and r[field] > 0)
                self.assertEqual(r["audited_workflow_seconds"], r["workflow_seconds"]+r["audit_seconds"])
                prep.setdefault(r["seed"], r["preprocessing_seconds"])
                self.assertEqual(prep[r["seed"]], r["preprocessing_seconds"])
                if r["method"] == "PCA-default":
                    self.assertIsNone(r["full_stationarity_qualified"])
                    continue
                solver = r["solver"]
                self.assertEqual(solver["iterations"], 50)
                self.assertFalse(solver["sketch_converged"])
                self.assertFalse(r["full_stationarity_qualified"])
                self.assertGreater(solver["sketch_eta"], .001)
                self.assertGreater(r["full_eta"], .001)
                self.assertTrue(np.all(np.diff(solver["sketch_objectives_scaled"]) <= 1e-10))
            for seed, seconds in prep.items():
                receipt = report.strict_json(BASE/name/f"seed_{seed}_preparation.json")
                self.assertEqual(receipt["seconds"], seconds)

    def test_every_fixed_native_figure_decodes(self):
        count = 0
        for name, _, nseeds in report.PACKAGES:
            package = BASE/name
            expected = {f"rank_{rank}/seed_{seed}_pc{a}{a+1}.png" for rank in (4, 6, 8)
                        for seed in range(2025, 2025+nseeds) for a in range(1, rank, 2)}
            actual = {str(p.relative_to(package)) for p in package.rglob("*.png")}
            self.assertEqual(actual, expected)
            for relative in expected:
                with Image.open(package/relative) as im:
                    im.load()
                    self.assertEqual(im.size, (3600, 1692))
                    self.assertIsNotNone(im.getbbox())
                count += 1
        self.assertEqual(count, 54)

    def test_complete_paired_report_replays_and_does_not_select_a_winner(self):
        output = BASE/"complete_report"
        receipt = report.strict_json(output/"completion.json")
        self.assertEqual(set(receipt["files"]), {"README.md", "analysis.json"})
        for name, expected in receipt["files"].items():
            self.assertEqual(runtime.digest(output/name), expected)
        analysis = report.strict_json(output/"analysis.json")
        self.assertEqual(analysis["status"], "complete_artifact_only_descriptive_report_not_confirmation")
        for name, _, _ in report.PACKAGES:
            expected = self.packages[name]
            self.assertEqual(analysis["packages"][name]["cells"], expected["cells"])
            self.assertEqual(analysis["packages"][name]["paired_comparisons"], report.comparisons(expected["records"]))
            self.assertEqual(receipt["source_completion_sha256"][name], runtime.digest(BASE/name/"completion.json"))

    def test_full_input_statistics_and_fixed_rows_replay(self):
        from rspca.genomics_aim import transform_smartpca
        from rspca.genomics_blocked import fit_block_preprocessor
        source = importlib.import_module("genomics_pca_loading_all_autosomes")
        original = ROOT/"results/genomics_all_autosome_preparation_2026-08-31/package"
        package = BASE/"full_cached_matched"
        with tempfile.TemporaryDirectory(prefix="rspca-default-runtime-replay-") as scratch:
            with source.load_formal_input(original/"genomics_aim_all_autosomes.npz",
                original/"genomics_aim_all_autosomes.manifest.json",
                expected_source_manifest_sha256=source.FROZEN_FORMAL_MANIFEST_SHA256,
                scratch_root=Path(scratch)) as (data, _):
                with np.load(package/"seed_2025_preprocessing.npz", allow_pickle=False) as frozen:
                    train, test = frozen["train_indices"], frozen["test_indices"]
                    pre = fit_block_preprocessor(data.genotypes, train)
                    for field in ("mean_dosage", "scale", "retained_snps"):
                        np.testing.assert_array_equal(getattr(pre, field), frozen[field])
                train_rows = transform_smartpca(data.genotypes[train[:4]], pre)
                test_rows = transform_smartpca(data.genotypes[test[:4]], pre)
                for record in self.packages["full_cached_matched"]["records"]:
                    path = package/f"rank_{record['rank']}/seed_2025_{record['method']}_scores.npz"
                    with np.load(path, allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays["sample_ids"], data.sample_ids)
                        np.testing.assert_array_equal(arrays["labels"], data.population_labels)
                        B = arrays["B"]
                        np.testing.assert_allclose(train_rows @ B, arrays["train_scores"][:4], atol=1e-8, rtol=1e-10)
                        np.testing.assert_allclose(test_rows @ B, arrays["test_scores"][:4], atol=1e-8, rtol=1e-10)


if __name__ == "__main__":
    unittest.main()

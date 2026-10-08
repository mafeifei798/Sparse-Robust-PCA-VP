"""Independent artifact and input replay; no refitting of timed SNP models."""

import importlib
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
run = importlib.import_module("genomics_randomized_vp_5pct")
PACKAGE = run.BASE / "paired_20seeds"
PROTOCOL_SHA = "88ca8544c8e5c1a585367226b415aa3781e1a154adc9e451508c57041418e81a"
COMPLETION_SHA = "ad42b4ae580c9e47126a92df7fa514603330de89966397a0a3978efaee2f7534"


@pytest.fixture(scope="module")
def evidence():
    load = run.formal._strict_json_load
    return {"completion": load(PACKAGE / "completion.json"),
        "spec": load(PACKAGE / "specification_before_run.json"),
        "records": load(PACKAGE / "results.json"),
        "summary": load(PACKAGE / "summary.json"),
        "preparation": load(PACKAGE / "preparation.json")}


def test_closed_package_hashes_and_source_snapshot(evidence):
    assert run.digest(PACKAGE / "completion.json") == COMPLETION_SHA
    completion = evidence["completion"]
    assert completion["status"] == "complete_frozen_5pct_panel_development"
    assert completion["records"] == 80 and completion["timed_fits"] == 240
    assert completion["excluded_warmup_fits"] == 80 and completion["exact_panel_replays"] == 20
    files = {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*") if p.is_file()}
    assert files == set(completion["files"]) | {"completion.json"}
    assert not any(p.is_symlink() for p in PACKAGE.rglob("*"))
    for name, expected in completion["files"].items():
        assert run.digest(PACKAGE / name) == expected
        if name.endswith(".json"):
            run.formal._strict_json_load(PACKAGE / name)
    assert not (PACKAGE / "FAILED.json").exists()
    spec = evidence["spec"]
    assert spec["protocol_sha256"] == PROTOCOL_SHA == run.digest(run.BASE / "PROTOCOL.md")
    assert spec["source_sha256"]["experiments/genomics_randomized_vp_5pct.py"] == "9aa0c2c651416936c93f4b4bd9f28006a80cbf67f36b89d0c7ec7211039d26e5"
    for name, expected in run.SOURCE_LOCKS.items():
        assert spec["source_sha256"][name] == expected
    assert all("fed" not in name.lower() for name in spec["source_sha256"])
    with zipfile.ZipFile(PACKAGE / "source_snapshot.zip") as archive:
        import hashlib
        assert set(archive.namelist()) == set(spec["source_sha256"]) | {"PROTOCOL.md"}
        for name, expected in spec["source_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert hashlib.sha256(archive.read("PROTOCOL.md")).hexdigest() == PROTOCOL_SHA


def test_all_cells_default_settings_stationarity_and_timing_costs(evidence):
    records, spec = evidence["records"], evidence["spec"]
    assert spec["seeds"] == list(run.SEEDS)
    assert spec["rank"] == 8 and spec["panel_size"] == 22755
    assert spec["power_iterations"] == 4 and spec["max_iter"] == 50
    assert spec["methods"] == [list(item) for item in run.METHODS]
    assert {(r["seed"], r["method"]) for r in records} == {(seed, name) for seed in run.SEEDS for name, _ in run.METHODS}
    environment = run.formal._strict_json_load(PACKAGE / "input_and_environment.json")
    assert all(p["num_threads"] == 1 for p in environment["threadpools"])
    assert sum(p["internal_api"] == "openblas" for p in environment["threadpools"]) == 2
    assert environment["input"]["genotype_npz_sha256"] == run.formal.FROZEN_FORMAL_NPZ_SHA256
    for row in records:
        assert row["input_snps"] == 22755 and row["original_snps"] == 455102
        assert row["rank"] == 8
        assert len(row["fit_seconds"]) == len(row["fit_cpu_seconds"]) == 3
        assert len(row["projection_seconds"]) == 3
        assert row["fit_seconds_median"] == float(np.median(row["fit_seconds"]))
        assert row["projection_seconds_median"] == float(np.median(row["projection_seconds"]))
        assert all(t > 0 for t in row["fit_seconds"] + row["fit_cpu_seconds"])
        costs = row["preparation"]
        frozen = costs["preprocessing_seconds"] + row["fit_seconds_median"] + row["projection_seconds_median"] + row["readout_seconds"]
        assert row["frozen_panel_workflow_seconds"] == frozen
        assert row["discovery_workflow_seconds"] == costs["screen_seconds"] + costs["materialization_seconds"] + frozen
        assert row["audited_discovery_seconds"] == row["discovery_workflow_seconds"] + row["audit_seconds"]
        if row["method"] == "PCA-default":
            # The exact parameters are also bound by the executed source and sklearn version.
            from sklearn.decomposition import PCA
            defaults = PCA(n_components=8, random_state=row["seed"]).get_params()
            assert row["solver"]["parameters"] == defaults
            assert row["solver"]["resolved_solver"] == "randomized"
            assert row["full_stationarity_qualified"] is None
            assert row["nonzero_snp_fraction"] == row["nonzero_entry_fraction"] == 1.
        else:
            assert 1 <= row["solver"]["iterations"] <= 50
            assert row["solver"]["sketch_converged"] == (row["solver"]["sketch_eta"] <= .001)
            assert row["full_stationarity_qualified"] == (row["full_eta"] <= .001)
            assert np.all(np.diff(row["solver"]["sketch_objectives_scaled"]) <= 1e-10)


def test_every_shared_panel_original_binding_and_native_prediction_replays(evidence):
    index = {(r["seed"], r["method"]): r for r in evidence["records"]}
    reference = run.formal._strict_json_load(run.REFERENCE / run.formal.RESULT_FILENAME)
    with run.formal.load_formal_input(run.INPUT, run.MANIFEST,
            expected_source_manifest_sha256=run.formal.FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=Path("/private/tmp")) as (data, _):
        for seed in run.SEEDS:
            directory = PACKAGE / f"seed_{seed}"
            case = run.historical._reference_case(reference, seed=seed, width=455102)
            expected_train, expected_test = run.historical._split_from_reference(data.population_labels, data.sample_ids, case)
            selected = np.flatnonzero(run.historical._selected_mask(case)).astype(np.int64)
            with np.load(directory / "shared_panel.npz", allow_pickle=False) as saved:
                G = saved["genotypes"]
                train, test, labels = saved["train_indices"], saved["test_indices"], saved["labels"]
                np.testing.assert_array_equal(train, expected_train)
                np.testing.assert_array_equal(test, expected_test)
                np.testing.assert_array_equal(saved["selected_original_indices"], selected)
                np.testing.assert_array_equal(saved["sample_ids"], data.sample_ids)
                np.testing.assert_array_equal(labels, data.population_labels)
                np.testing.assert_array_equal(saved["snp_ids"], data.snp_ids[selected])
                np.testing.assert_array_equal(G, np.asarray(data.genotypes[:, selected]))
                pre = run.fit_smartpca_preprocessor(G[train])
                np.testing.assert_array_equal(pre.mean_dosage, saved["mean_dosage"])
                np.testing.assert_array_equal(pre.scale, saved["scale"])
                np.testing.assert_array_equal(pre.retained_snps, saved["retained_snps"])
                X = np.ascontiguousarray(run.transform_smartpca(G[train], pre))
                XT = np.ascontiguousarray(run.transform_smartpca(G[test], pre))
                input_hashes = {"train": run.array_sha256(X), "test": run.array_sha256(XT)}
                assert X.shape == (800, 22755) and XT.shape == (200, 22755)
                for name, _ in run.METHODS:
                    row = index[seed, name]
                    assert row["shared_input_hashes"] == input_hashes
                    assert row["preparation"]["raw_panel_sha256"] == run.array_sha256(G)
                    assert row["preparation"]["selected_indices_sha256"] == case["selected_original_indices_sha256"]
                    with np.load(directory / f"{name}.npz", allow_pickle=False) as model:
                        A, B, mean = model["A"], model["B"], model["mean"]
                        assert A.shape == B.shape == (22755, 8)
                        assert A.dtype == B.dtype == np.float64
                        np.testing.assert_allclose(A.T @ A, np.eye(8), atol=1e-8, rtol=0)
                        scores, test_scores = run.project(X, XT, B, mean)
                        np.testing.assert_allclose(scores, model["train_scores"], atol=1e-9, rtol=1e-10)
                        np.testing.assert_allclose(test_scores, model["test_scores"], atol=1e-9, rtol=1e-10)
                        assert run.readouts(scores, test_scores, labels, train, test) == row["readouts"]
                        assert row["nonzero_entry_fraction"] == float(np.mean(B != 0))
                        assert row["nonzero_snp_fraction"] == float(np.mean(np.any(B != 0, axis=1)))
                with np.load(directory / "full_reference_scores.npz", allow_pickle=False) as full:
                    views = run.readouts(full["train_scores"], full["test_scores"], labels, train, test)
                    assert views == index[seed, "PCA-default"]["preparation"]["full_exact_pca_context"]
            del G, X, XT


def test_bootstrap_and_summary_replay_including_all_penalties(evidence):
    def compare(left, right):
        if isinstance(left, dict):
            assert set(left) == set(right)
            for key in left:
                compare(left[key], right[key])
        elif isinstance(left, list):
            assert len(left) == len(right)
            for a, b in zip(left, right, strict=True):
                compare(a, b)
        elif isinstance(left, float):
            # Different supported BLAS/NumPy versions can change only final bits.
            assert left == pytest.approx(right, abs=1e-12, rel=0)
        else:
            assert left == right
    compare(run.summarize(evidence["records"]), evidence["summary"])
    assert len(evidence["summary"]["cells"]) == 4
    assert len(evidence["summary"]["contrasts"]) == 3
    assert evidence["summary"]["independent_population_validation"] is False


def test_every_fixed_split_figure_decodes_and_schedule_is_complete(evidence):
    expected = {f"seed_{seed}/pc{pair}.png" for seed in run.SEEDS for pair in (12, 34, 56, 78)}
    assert {str(p.relative_to(PACKAGE)) for p in PACKAGE.rglob("*.png")} == expected
    for name in expected:
        with Image.open(PACKAGE / name) as image:
            image.load()
            assert image.size == (3080, 2332)
            assert image.getbbox() is not None
    for seed in run.SEEDS:
        schedule = run.formal._strict_json_load(PACKAGE / f"seed_{seed}/timing_schedule.json")
        assert [r["repeat"] for r in schedule] == [-1, 0, 1, 2]
        for row in schedule:
            assert sorted(row["order"]) == sorted(name for name, _ in run.METHODS)
            assert row["warmup"] == (row["repeat"] == -1)

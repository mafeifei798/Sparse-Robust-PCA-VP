"""Read-only, original-input replay of the own-VP 5% selection package.

This validates saved models and all original-input projections, not optimizer
refitting or a second measurement of runtime. Reports must stay outside the
sealed package.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
from pathlib import Path
import sys
import types
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __name__ == "__main__" and "rspca" not in sys.modules:
    package = types.ModuleType("rspca")
    package.__path__ = [str(ROOT / "src/rspca")]
    package.__package__ = "rspca"
    package.__spec__ = importlib.machinery.ModuleSpec("rspca", None, is_package=True)
    sys.modules["rspca"] = package

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

import genomics_randomized_vp_selected5 as run
from rspca.genomics_aim import fit_smartpca_preprocessor, transform_smartpca
from rspca.genomics_blocked import GenotypeOperator, fit_block_preprocessor

RUNNER_SHA = "f3c605ec8e343ad3b0408c72b54c41a49537df8a207a22dfa73b671bafb452be"
PROTOCOL_SHA = "92d7cfbe866ce8e3cbdf01ea93dca8223aff12400090667a2c607553239192c7"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_nested(left, right):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), "dictionary fields differ")
        for key in left:
            equal_nested(left[key], right[key])
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), "list shape differs")
        for a, b in zip(left, right, strict=True):
            equal_nested(a, b)
    elif isinstance(left, float):
        require(np.isclose(left, right, atol=1e-12, rtol=0), "numeric summary differs")
    else:
        require(left == right, "value differs")


def validate_structure(package, expected_completion_sha256):
    package = Path(package)
    load = run.shared.formal._strict_json_load
    require(run.digest(package / "completion.json") == expected_completion_sha256, "external completion anchor mismatch")
    complete = load(package / "completion.json")
    require(complete["status"] == "complete_own_rvp_selection_development", "not a successful complete package")
    require(complete["records"] == 80 and complete["timed_pipelines"] == 240 and complete["timed_solver_fits"] == 420, "wrong complete-grid counts")
    require(complete["excluded_warmup_pipelines"] == 4 and complete["excluded_warmup_solver_fits"] == 7, "wrong warmup count")
    paths = list(package.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "symlink inside package")
    require({str(p.relative_to(package)) for p in paths if p.is_file()} == set(complete["files"]) | {"completion.json"}, "package is not closed")
    require(not (package / "FAILED.json").exists(), "failed package cannot pass")
    for name, expected in complete["files"].items():
        require(run.digest(package / name) == expected, f"file hash differs: {name}")
        if name.endswith(".json"):
            load(package / name)
    spec, records = load(package / "specification_before_run.json"), load(package / "results.json")
    preparation, summary = load(package / "preparation.json"), load(package / "summary.json")
    environment = load(package / "input_and_environment.json")
    require(spec["source_sha256"]["experiments/genomics_randomized_vp_selected5.py"] == RUNNER_SHA, "runner differs from pre-fit anchor")
    require(spec["protocol_sha256"] == PROTOCOL_SHA == run.digest(run.BASE / "PROTOCOL.md"), "protocol differs from pre-fit anchor")
    require(all("fed" not in name.lower() for name in spec["source_sha256"]), "Fed source entered the executed closure")
    for name, expected in spec["source_sha256"].items():
        require(run.digest(ROOT / name) == expected, f"current replay helper differs from executed source: {name}")
    for name, expected in run.shared.SOURCE_LOCKS.items():
        require(spec["source_sha256"][name] == expected, f"historical solver changed: {name}")
    with zipfile.ZipFile(package / "source_snapshot.zip") as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate snapshot names")
        require(set(archive.namelist()) == set(spec["source_sha256"]) | {"PROTOCOL.md"}, "wrong source closure")
        for name, expected in spec["source_sha256"].items():
            require(hashlib.sha256(archive.read(name)).hexdigest() == expected, f"source snapshot differs: {name}")
        require(hashlib.sha256(archive.read("PROTOCOL.md")).hexdigest() == PROTOCOL_SHA, "protocol snapshot differs")
    require(spec["seeds"] == list(run.SEEDS) and spec["methods"] == [list(m) for m in run.METHODS], "grid differs")
    require((spec["rank"], spec["panel_size"], spec["power_iterations"], spec["oversample"], spec["max_iter"], spec["timed_repeats"]) == (8, 22755, 4, 10, 50, 3), "settings differ")
    require(all(p["num_threads"] == 1 for p in environment["threadpools"]), "uncontrolled thread pool")
    require(sum(p["internal_api"] == "openblas" for p in environment["threadpools"]) == 2, "unexpected BLAS environment")
    require(environment["input"]["genotype_npz_sha256"] == run.shared.formal.FROZEN_FORMAL_NPZ_SHA256, "input binding differs")
    require({(r["seed"], r["method"]) for r in records} == {(seed, name) for seed in run.SEEDS for name, _ in run.METHODS} and len(records) == 80, "missing or duplicate cell")
    prep = {p["seed"]: p for p in preparation}
    require(len(preparation) == len(prep) == 20, "wrong preparation count")
    for row in records:
        require(row["preparation_seconds"] == prep[row["seed"]]["seconds"], "preparation not charged equally")
        require(len(row["timings"]) == 3, "wrong timed repetitions")
        for key, median in row["timing_medians"].items():
            require(median == float(np.median([t[key] for t in row["timings"]])), "wrong timing median")
        for t in row["timings"]:
            require(t["model_computation"] >= sum(t[key] for key in run.STAGES) + t["readout"], "missing computational cost")
            require(t["full_fit"] > 0 and t["model_cpu"] > 0 and t["projection"] > 0, "nonpositive timing")
            if row["penalty_fraction"] is not None:
                require(all(t[key] > 0 for key in run.STAGES), "missing VP selection/refit stage")
        require(row["discovery_workflow_seconds"] == row["preparation_seconds"] + row["timing_medians"]["model_computation"], "wrong discovery cost")
        require(row["audited_discovery_seconds"] == row["discovery_workflow_seconds"] + row["audit_seconds"], "wrong audited cost")
        if row["penalty_fraction"] is None:
            require(row["final_solver"]["parameters"] == PCA(n_components=8, random_state=row["seed"]).get_params(), "PCA defaults changed")
            require(row["final_solver"]["resolved_solver"] == "randomized", "unexpected default solver")
            require(row["selector_solver"] is row["selection"] is None, "PCA is not a selection arm")
            require(row["final_input_snps"] == row["training_retained_snps"], "PCA did not use full input")
        else:
            require(row["final_input_snps"] == row["selection"]["count"] == 22755, "wrong own-panel size")
            for role in ("selector", "final"):
                solver, audit = row[f"{role}_solver"], row[f"{role}_audit"]
                require(1 <= solver["iterations"] <= 50, "iteration budget changed")
                require(solver["sketch_converged"] == (solver["sketch_eta"] <= .001), "wrong sketch gate")
                require(audit["full_stationarity_qualified"] == (audit["full_eta"] <= .001), "wrong full-data gate")
                require(np.all(np.diff(solver["sketch_objectives_scaled"]) <= 1e-10), "descent trace failed")
    equal_nested(run.summarize(records), summary)
    expected_figures = {f"seed_{seed}/pc{pair}.png" for seed in run.SEEDS for pair in (12, 34, 56, 78)}
    require({str(p.relative_to(package)) for p in package.rglob("*.png")} == expected_figures, "figure set incomplete")
    for name in expected_figures:
        with Image.open(package / name) as picture:
            picture.load()
            require(picture.size == (3080, 2332) and picture.getbbox() is not None, "invalid figure")
    for seed in run.SEEDS:
        schedule = load(package / f"seed_{seed}/timing_schedule.json")
        require([s["repeat"] for s in schedule] == ([-1, 0, 1, 2] if seed == run.SEEDS[0] else [0, 1, 2]), "wrong warmup/timing schedule")
        for s in schedule:
            order = np.random.default_rng(seed + (s["repeat"] + 1) * 10000).permutation(len(run.METHODS))
            require(s["order"] == [run.METHODS[i][0] for i in order] and s["warmup"] == (s["repeat"] == -1), "timing order not pre-fixed")
    return {"records": records, "preparation": preparation, "summary": summary, "specification": spec,
            "file_count": len(complete["files"]) + 1}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_original_input(package, evidence):
    """Replay full and selected native projections without recreating giant caches."""
    package = Path(package)
    records = {(r["seed"], r["method"]): r for r in evidence["records"]}
    preparations = {p["seed"]: p for p in evidence["preparation"]}
    reference = run.shared.formal._strict_json_load(run.shared.REFERENCE / run.shared.formal.RESULT_FILENAME)
    full_models, panel_models, predictions = 0, 0, 0
    with threadpool_limits(1), run.shared.formal.load_formal_input(run.shared.INPUT, run.shared.MANIFEST,
            expected_source_manifest_sha256=run.shared.formal.FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=Path("/private/tmp")) as (data, identity):
        cohort = load_arrays(package / "cohort.npz")
        for key, expected in (("labels", data.population_labels), ("sample_ids", data.sample_ids),
                              ("snp_ids", data.snp_ids), ("chromosomes", data.chromosomes)):
            np.testing.assert_array_equal(cohort[key], expected)
        for seed in run.SEEDS:
            directory = package / f"seed_{seed}"
            split = [r for r in reference["split_receipts"] if r["seed"] == seed][0]
            train, test = run.shared.historical._split_from_reference(data.population_labels, data.sample_ids, split)
            saved = load_arrays(directory / "preprocessing.npz")
            pre = fit_block_preprocessor(data.genotypes, train, chunk_size=2048)
            retained = np.flatnonzero(pre.retained_snps).astype(np.int64)
            for key, expected in (("train_indices", train), ("test_indices", test),
                                  ("retained_original_indices", retained), ("mean_dosage", pre.mean_dosage),
                                  ("scale", pre.scale), ("retained_snps", pre.retained_snps)):
                np.testing.assert_array_equal(saved[key], expected)
            models = {}
            for name, penalty in run.METHODS:
                role = "final" if penalty is None else "selector"
                model = load_arrays(directory / f"{name}-{role}.npz")
                require(model["B"].shape == model["A"].shape == (len(retained), 8), "wrong full model shape")
                np.testing.assert_allclose(model["A"].T @ model["A"], np.eye(8), atol=1e-8, rtol=0)
                models[name] = model
            combined_B = np.ascontiguousarray(np.concatenate([models[name]["B"] for name, _ in run.METHODS], axis=1))
            scores_by_split = {}
            for label, rows in (("train", train), ("test", test)):
                scores = np.zeros((len(rows), 32))
                stream_digest = hashlib.sha256()
                for start, stop, block in GenotypeOperator(data.genotypes, rows, pre, chunk_size=2048).blocks():
                    stream_digest.update(np.asarray([start, stop], dtype="<i8").tobytes())
                    stream_digest.update(np.ascontiguousarray(block, dtype="<f8").tobytes())
                    scores += block @ combined_B[start:stop]
                require(stream_digest.hexdigest() == preparations[seed]["identity"][label]["standardized_block_stream_sha256"], "original-input standardized cache binding differs")
                scores_by_split[label] = scores
            del combined_B
            for i, (name, penalty) in enumerate(run.METHODS):
                model, row = models[name], records[seed, name]
                views = {}
                for label in ("train", "test"):
                    values = scores_by_split[label][:, i*8:(i+1)*8] - model["mean"] @ model["B"]
                    np.testing.assert_allclose(values, model[f"{label}_scores"], atol=1e-8, rtol=1e-10)
                    views[label] = values
                computed = run.shared.readouts(views["train"], views["test"], data.population_labels, train, test)
                require(computed == (row["readouts"] if penalty is None else row["full_vp_context"]), "full model readout differs")
                full_models += 1
                predictions += 400
                if penalty is None:
                    continue
                saved_selection = load_arrays(directory / f"{name}-selection.npz")
                # Independently spell out the ranking, without calling the run's helper.
                importance = np.abs(model["B"]).sum(axis=1)
                selected = np.sort(np.lexsort((retained, -importance))[:22755]).astype(np.int64)
                original = retained[selected]
                for key, expected in (("importance", importance), ("selected_retained_indices", selected), ("selected_original_indices", original)):
                    np.testing.assert_array_equal(saved_selection[key], expected)
                binding = row["selection"]
                for key, value in (("selector_B_sha256", model["B"]), ("importance_sha256", importance),
                                   ("selected_original_indices_sha256", original), ("selected_retained_indices_sha256", selected)):
                    require(run.array_sha256(value) == binding[key], "selection receipt binding differs")
                G = np.ascontiguousarray(data.genotypes[:, original])
                panel_pre = fit_smartpca_preprocessor(G[train])
                np.testing.assert_array_equal(panel_pre.mean_dosage, pre.mean_dosage[original])
                np.testing.assert_array_equal(panel_pre.scale, pre.scale[original])
                XP = np.ascontiguousarray(transform_smartpca(G[train], panel_pre))
                XTP = np.ascontiguousarray(transform_smartpca(G[test], panel_pre))
                require(run.array_sha256(XP) == binding["panel_train_sha256"] and run.array_sha256(XTP) == binding["panel_test_sha256"], "own panel original-input binding differs")
                final = load_arrays(directory / f"{name}-final.npz")
                require(final["A"].shape == final["B"].shape == (22755, 8), "wrong panel fit shape")
                np.testing.assert_allclose(final["A"].T @ final["A"], np.eye(8), atol=1e-8, rtol=0)
                panel_scores = run.shared.project(XP, XTP, final["B"], final["mean"])
                np.testing.assert_allclose(panel_scores[0], final["train_scores"], atol=1e-9, rtol=1e-10)
                np.testing.assert_allclose(panel_scores[1], final["test_scores"], atol=1e-9, rtol=1e-10)
                require(run.shared.readouts(*panel_scores, data.population_labels, train, test) == row["readouts"], "panel prediction replay differs")
                for role, fit in (("selector", model), ("final", final)):
                    require(row[f"{role}_audit"]["nonzero_entry_fraction"] == float(np.mean(fit["B"] != 0)), "coefficient support differs")
                    require(row[f"{role}_audit"]["nonzero_snp_fraction"] == float(np.mean(np.any(fit["B"] != 0, axis=1))), "row support differs")
                panel_models += 1
                predictions += 400
                del G, XP, XTP, final
            del models, scores_by_split
            print(f"verified seed={seed}: original data, VP-owned selections, full/panel scores and predictions", flush=True)
    return {"trusted_original_input_sha256": identity["genotype_npz_sha256"],
        "full_model_projection_replays": full_models, "own_vp_panel_replays": panel_models,
        "heldout_prediction_replays_both_readouts": predictions,
        "optimizer_refit_verified": False,
        "boundary": "original-input projection and selection replay; optimizer execution bound by pre-fit source snapshot, not refitted here"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=run.BASE / "paired_20seeds")
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    require(not args.receipt.resolve().is_relative_to(args.package.resolve()), "receipt must stay outside sealed package")
    require(not args.receipt.exists(), "refusing to overwrite a verification receipt")
    evidence = validate_structure(args.package, args.completion_sha256)
    replay = validate_original_input(args.package, evidence)
    run.write_json(args.receipt, {"status": "PASS_ORIGINAL_INPUT_SELECTION_AND_PROJECTION_REPLAY",
        "completion_sha256": args.completion_sha256, "verifier_sha256": run.digest(Path(__file__)),
        "files": evidence["file_count"], **replay})
    print(f"VERIFIED: {args.receipt}", flush=True)

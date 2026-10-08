"""Frozen rank-8 / 5% panel comparison; no selection or tuning after outcomes."""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter, process_time
import traceback
import types
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[1]
# Fresh scientific execution needs the non-Fed modules, not eager package exports.
if __name__ == "__main__" and "rspca" not in sys.modules:
    package = types.ModuleType("rspca")
    package.__path__ = [str(ROOT / "src/rspca")]
    package.__package__ = "rspca"
    package.__spec__ = importlib.machinery.ModuleSpec("rspca", None, is_package=True)
    sys.modules["rspca"] = package

import numpy as np
import scipy
from scipy import linalg
import sklearn
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_info, threadpool_limits

import genomics_all_autosome_runtime_benchmark as historical
import genomics_pca_loading_all_autosomes as formal
from genomics_randomized_vp_matched import COLORS, POPULATIONS, digest, metrics, write_json
from rspca.datasets import array_sha256
from rspca.genomics_aim import fit_smartpca_preprocessor, transform_smartpca
from rspca.randomized import _full_diagnostics
from rspca.randomized_reduced_fast import fit_fast_randomized_vp_spca

BASE = ROOT / "results/genomics_randomized_vp_5pct_2026-09-07"
REFERENCE = ROOT / "results/genomics_all_autosome_stratification_2026-08-31/remote_v2_clean/registered_20seeds_merged"
INPUT = ROOT / "results/genomics_all_autosome_preparation_2026-08-31/package/genomics_aim_all_autosomes.npz"
MANIFEST = INPUT.with_suffix(".manifest.json")
SEEDS = tuple(range(2025, 2045))
RANK, PANEL_SIZE, REPEATS, BOOTSTRAP_SEED = 8, 22755, 3, 9072026
PENALTIES = (.0002, .001, .005)
METHODS = (("PCA-default", None),) + tuple((f"RVP-c{c:g}", c) for c in PENALTIES)
SOURCE_LOCKS = {
    "src/rspca/randomized_reduced_fast.py": "eeb2ff1752e1b470d704be2f14c41c905c948542766fb97bfe09e9e3d137b48d",
    "src/rspca/randomized_reduced.py": "6f9a1bfd60cd90a4a8bc9ffdf7753f01f1dcaac90bf4383c9f769c2967b73e70",
    "src/rspca/randomized.py": "101833335c80b178f35b696bdc28dd8a4c0eb6a9fc5a565af802d83088d0823b",
    "src/rspca/vp.py": "88a84aa99825357f3a16650bf3e56bfef54a850aa17880342334ba7ad402b953",
    "src/rspca/genomics_aim.py": "41d0d80802a41526f3121e147a1018580424e6a4f88c6237db4c118c4ef63b27",
    "experiments/genomics_randomized_vp_matched.py": "62bb043a4cf97ab471a6e3669b851a6801dbbadfc3b7f912660c8903701811fa",
    "experiments/genomics_pca_loading_all_autosomes.py": "97087a965c7e4137698fadfd6910171606c55b8448a8e6550a91c6363769885d",
}


def verify_sources():
    for name, expected in SOURCE_LOCKS.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"previous solver/source identity changed: {name}")


def load_reference():
    verification = formal.validate_output_package(
        REFERENCE, expected_analysis_sha256=historical.FORMAL_ANALYSIS_SHA256,
        expected_package_anchor_sha256=historical.FORMAL_PACKAGE_ANCHOR_SHA256)
    payload = formal._strict_json_load(REFERENCE / formal.RESULT_FILENAME)
    historical._validate_reference_payload(payload, historical.FORMAL_MODE)
    return payload, verification


def source_closure():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.suffix == ".py" and path.is_relative_to(ROOT):
                paths.add(path)
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def fit_method(X, seed, penalty):
    if penalty is None:
        model = PCA(n_components=RANK, random_state=seed).fit(X)
        return model, model.components_.T, model.components_.T, model.mean_
    model = fit_fast_randomized_vp_spca(
        X, RANK, penalty_fraction=penalty, oversample=10, n_power_iter=4,
        max_iter=50, relative_tolerance=.001, check_every=10, random_state=seed)
    return model, model.A, model.B, np.zeros(X.shape[1])


def project(X, XT, B, mean):
    offset = mean @ B
    return X @ B - offset, XT @ B - offset


def readouts(train_scores, test_scores, labels, train, test):
    result = {}
    for name, standardize in (("raw", False), ("standardized", True)):
        value = metrics(train_scores, test_scores, labels[train], labels[test], standardize=standardize)
        confusion = np.asarray(value["confusion"])
        value["population_precision"] = {
            pop: float(confusion[i, i] / max(1, confusion[:, i].sum()))
            for i, pop in enumerate(POPULATIONS)}
        result[name] = value
    return result


def prepare_panel(raw, train, test, selected):
    started = perf_counter()
    panel = np.ascontiguousarray(raw[:, selected])
    materialization = perf_counter() - started
    started = perf_counter()
    pre = fit_smartpca_preprocessor(panel[train])
    X = np.ascontiguousarray(transform_smartpca(panel[train], pre))
    XT = np.ascontiguousarray(transform_smartpca(panel[test], pre))
    preprocessing = perf_counter() - started
    if not np.all(pre.retained_snps):
        raise ValueError("a sealed panel SNP was dropped by training preprocessing")
    X.setflags(write=False)
    XT.setflags(write=False)
    return panel, X, XT, pre, materialization, preprocessing


def run_panel(X, XT, labels, train, test, seed, output, costs):
    input_hashes = {"train": array_sha256(X), "test": array_sha256(XT)}
    fits, schedule = {}, []
    for repeat in range(-1, REPEATS):
        order = np.random.default_rng(seed + (repeat + 1) * 10000).permutation(len(METHODS))
        schedule.append({"repeat": repeat, "warmup": repeat == -1,
                         "order": [METHODS[i][0] for i in order]})
        for index in order:
            name, penalty = METHODS[index]
            gc.collect()
            wall, cpu = perf_counter(), process_time()
            model, A, B, mean = fit_method(X, seed, penalty)
            elapsed, cpu_elapsed = perf_counter() - wall, process_time() - cpu
            wall = perf_counter()
            scores, test_scores = project(X, XT, B, mean)
            projection_seconds = perf_counter() - wall
            if not all(np.isfinite(a).all() for a in (A, B, mean, scores, test_scores)):
                raise RuntimeError("nonfinite fit/projection")
            if repeat == -1:
                del model, A, B, mean, scores, test_scores
                continue
            if name not in fits:
                if penalty is None:
                    diagnostic = {"parameters": model.get_params(), "resolved_solver": model._fit_svd_solver}
                else:
                    diagnostic = {"alpha": model.alpha, "iterations": model.n_iter,
                        "sketch_converged": model.sketch_converged,
                        "sketch_eta": model.sketch_relative_stationarity,
                        "sketch_objectives_scaled": model.sketch_objectives_scaled,
                        "restarts": model.restarts, "timings": model.timings,
                        "preparation_diagnostics": model.preparation_diagnostics}
                fits[name] = {"A": A.copy(), "B": B.copy(), "mean": mean.copy(),
                    "train_scores": scores, "test_scores": test_scores,
                    "diagnostic": diagnostic, "wall": [], "cpu": [], "projection": []}
            else:
                for key, value in (("A", A), ("B", B), ("mean", mean), ("train_scores", scores), ("test_scores", test_scores)):
                    np.testing.assert_allclose(value, fits[name][key], rtol=1e-10, atol=1e-10)
            fits[name]["wall"].append(elapsed)
            fits[name]["cpu"].append(cpu_elapsed)
            fits[name]["projection"].append(projection_seconds)
            print(f"seed={seed} repeat={repeat} {name} fit={elapsed:.4f}s cpu={cpu_elapsed:.4f}s", flush=True)
            del model, A, B, mean, scores, test_scores
    if input_hashes != {"train": array_sha256(X), "test": array_sha256(XT)}:
        raise RuntimeError("shared model input changed during fitting")
    started = perf_counter()
    norm = float(linalg.norm(X))
    norm_seconds = perf_counter() - started
    records = []
    for name, penalty in METHODS:
        fit = fits[name]
        started = perf_counter()
        views = readouts(fit["train_scores"], fit["test_scores"], labels, train, test)
        readout_seconds = perf_counter() - started
        started = perf_counter()
        if penalty is None:
            residual2 = max(0., norm**2 - X.shape[0] * float(fit["mean"] @ fit["mean"]) - float(np.sum(fit["train_scores"]**2)))
            eta = None
        else:
            alpha = fit["diagnostic"]["alpha"]
            _, objective, stationarity = _full_diagnostics(X, fit["A"], fit["B"], alpha, 1 / norm**2, norm)
            residual2 = max(0., 2 * (objective - alpha * float(np.abs(fit["B"]).sum())))
            eta = float(stationarity / (alpha * np.sqrt(fit["B"].size)))
        audit_seconds = perf_counter() - started + norm_seconds
        fit_seconds = float(np.median(fit["wall"]))
        projection_seconds = float(np.median(fit["projection"]))
        frozen_workflow = costs["preprocessing_seconds"] + fit_seconds + projection_seconds + readout_seconds
        discovery = costs["screen_seconds"] + costs["materialization_seconds"] + frozen_workflow
        record = {"seed": seed, "rank": RANK, "method": name, "penalty_fraction": penalty,
            "input_snps": X.shape[1], "original_snps": 455102, "shared_input_hashes": input_hashes,
            "fit_seconds": fit["wall"], "fit_cpu_seconds": fit["cpu"],
            "fit_seconds_median": fit_seconds, "projection_seconds": fit["projection"],
            "projection_seconds_median": projection_seconds, "readout_seconds": readout_seconds,
            "frozen_panel_workflow_seconds": frozen_workflow, "discovery_workflow_seconds": discovery,
            "audited_discovery_seconds": discovery + audit_seconds, "audit_seconds": audit_seconds,
            "preparation": costs, "readouts": views, "solver": fit["diagnostic"],
            "full_reconstruction_error": float(np.sqrt(residual2) / norm), "full_eta": eta,
            "full_stationarity_qualified": bool(eta <= .001) if eta is not None else None,
            "nonzero_entry_fraction": float(np.mean(fit["B"] != 0)),
            "nonzero_snp_fraction": float(np.mean(np.any(fit["B"] != 0, axis=1)))}
        fit["record"] = record
        records.append(record)
        np.savez_compressed(output / f"{name}.npz", **{key: fit[key] for key in ("A", "B", "mean", "train_scores", "test_scores")})
    write_json(output / "timing_schedule.json", schedule)
    write_json(output / "records.json", records)
    return fits, records


def interval(values, *, statistic="mean", resamples=100000):
    values = np.asarray(values, dtype=float)
    reducer = np.mean if statistic == "mean" else np.median
    index = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values), size=(resamples, len(values)))
    distribution = reducer(values[index], axis=1)
    return {"estimate": float(reducer(values)), "ci95": np.quantile(distribution, [.025, .975]).tolist()}


def summarize(records):
    expected = {(seed, name) for seed in SEEDS for name, _ in METHODS}
    mapping = {(r["seed"], r["method"]): r for r in records}
    if len(records) != len(expected) or set(mapping) != expected:
        raise ValueError("incomplete or duplicated 20-seed method grid")
    cells, contrasts = [], []
    for name, penalty in METHODS:
        rows = [mapping[seed, name] for seed in SEEDS]
        cell = {"method": name, "n_seeds": len(rows), "penalty_fraction": penalty,
            "fit_seconds_median": float(np.median([r["fit_seconds_median"] for r in rows])),
            "frozen_panel_workflow_median": float(np.median([r["frozen_panel_workflow_seconds"] for r in rows])),
            "discovery_workflow_median": float(np.median([r["discovery_workflow_seconds"] for r in rows])),
            "full_qualified": sum(r["full_stationarity_qualified"] is True for r in rows),
            "sketch_qualified": sum(r["solver"].get("sketch_converged") is True for r in rows),
            "nonzero_entry_fraction": float(np.mean([r["nonzero_entry_fraction"] for r in rows])),
            "nonzero_snp_fraction": float(np.mean([r["nonzero_snp_fraction"] for r in rows]))}
        for view in ("raw", "standardized"):
            confusion = np.sum([r["readouts"][view]["confusion"] for r in rows], axis=0)
            cell[view] = {key: float(np.mean([r["readouts"][view][key] for r in rows])) for key in ("balanced_accuracy", "macro_f1")}
            cell[view].update({"confusion": confusion.tolist(),
                "population_recall": {pop: float(np.mean([r["readouts"][view]["population_recall"][pop] for r in rows])) for pop in POPULATIONS},
                "population_precision": {pop: float(confusion[i, i] / max(1, confusion[:, i].sum())) for i, pop in enumerate(POPULATIONS)}})
        cells.append(cell)
        if penalty is not None:
            baseline = [mapping[seed, "PCA-default"] for seed in SEEDS]
            contrast = {"method": name, "baseline": "PCA-default", "quality": {}, "runtime": {}}
            for scope in ("fit_seconds_median", "frozen_panel_workflow_seconds", "discovery_workflow_seconds", "audited_discovery_seconds"):
                reductions = [100 * (1 - r[scope] / b[scope]) for r, b in zip(rows, baseline, strict=True)]
                contrast["runtime"][scope] = {**interval(reductions, statistic="median"), "per_seed_reduction_percent": reductions}
            for view in ("raw", "standardized"):
                q = {}
                for key in ("balanced_accuracy", "macro_f1"):
                    differences = [r["readouts"][view][key] - b["readouts"][view][key] for r, b in zip(rows, baseline, strict=True)]
                    q[key] = interval(differences)
                for pop in POPULATIONS:
                    differences = [r["readouts"][view]["population_recall"][pop] - b["readouts"][view]["population_recall"][pop] for r, b in zip(rows, baseline, strict=True)]
                    q[f"recall_{pop}"] = interval(differences)
                q["prediction_disagreements"] = int(sum(np.count_nonzero(np.asarray(r["readouts"][view]["predictions"]) != np.asarray(b["readouts"][view]["predictions"])) for r, b in zip(rows, baseline, strict=True)))
                contrast["quality"][view] = q
            contrasts.append(contrast)
    return {"cells": cells, "contrasts": contrasts, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": 100000, "independent_population_validation": False}


def render_native(fits, labels, train, test, seed, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for first in range(0, RANK, 2):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10.6), constrained_layout=True)
        all_points = []
        for ax, (name, _) in zip(axes.flat, METHODS, strict=True):
            fit = fits[name]
            B = fit["B"]
            signs = np.sign(B[np.abs(B).argmax(axis=0), np.arange(RANK)])
            signs[signs == 0] = 1
            for split, indices in (("train", train), ("test", test)):
                scores = fit[f"{split}_scores"] * signs
                all_points.append(scores[:, first:first+2])
                for pop, color in zip(POPULATIONS, COLORS, strict=True):
                    selected = labels[indices] == pop
                    ax.scatter(scores[selected, first], scores[selected, first+1], color=color,
                        s=16 if split == "train" else 29, alpha=.55 if split == "train" else 1,
                        edgecolors="none" if split == "train" else "black",
                        linewidths=0 if split == "train" else .4, label=pop if split == "test" else None)
            value = fit["record"]["readouts"]["standardized"]
            ax.set_title(f'{name} | 8D standardized readout: BA {value["balanced_accuracy"]:.1%}, AMR {value["population_recall"]["AMR"]:.1%}', fontsize=11)
            ax.set_xlabel(f"Native component {first+1}")
            ax.set_ylabel(f"Native component {first+2}")
            ax.spines[["top", "right"]].set_visible(False)
        points = np.vstack(all_points)
        for dim in (0, 1):
            lo, hi = points[:, dim].min(), points[:, dim].max()
            margin = .05 * max(1, hi-lo)
            for ax in axes.flat:
                (ax.set_xlim if dim == 0 else ax.set_ylim)(lo-margin, hi+margin)
        handles, legend = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend, loc="outside lower center", ncol=5)
        fig.suptitle(f"Same frozen 5% SNP panel (22,755 / 455,102) | seed {seed}\nNative sample coordinates; signs only; black outlines: held-out samples", fontsize=14)
        fig.savefig(directory / f"pc{first+1}{first+2}.png", dpi=220)
        plt.close(fig)


def write_report(output, summary, preparation):
    lines = ["# Randomized VP-SPCA on the same frozen 5% SNP panels", "",
        "All 20 historical splits, rank 8, 22,755 SNPs per split; three fixed VP penalties.",
        "The panel is PCA-screened. VP did not independently discover a 5% SNP support.", "",
        "| Method | Raw BA | Standardized BA | Standardized AMR recall | Fit median s | Frozen-panel workflow s | Discovery workflow s | Nonzero SNP rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["cells"]:
        lines.append(f'| {c["method"]} | {c["raw"]["balanced_accuracy"]:.3%} | {c["standardized"]["balanced_accuracy"]:.3%} | {c["standardized"]["population_recall"]["AMR"]:.3%} | {c["fit_seconds_median"]:.6f} | {c["frozen_panel_workflow_median"]:.6f} | {c["discovery_workflow_median"]:.6f} | {c["nonzero_snp_fraction"]:.3%} |')
    lines += ["", "## Paired contrasts versus literal default PCA", "",
        "Intervals resample 20 overlapping split IDs, not independent people. Descriptive, no multiplicity correction.", "",
        "| Method | Standardized BA delta (pp), 95% interval | Standardized AMR delta (pp), 95% interval | Median paired fit reduction, 95% interval |",
        "| --- | --- | --- | --- |"]
    for c in summary["contrasts"]:
        ba, amr = c["quality"]["standardized"]["balanced_accuracy"], c["quality"]["standardized"]["recall_AMR"]
        time = c["runtime"]["fit_seconds_median"]
        lines.append(f'| {c["method"]} | {100*ba["estimate"]:+.3f} [{100*ba["ci95"][0]:+.3f}, {100*ba["ci95"][1]:+.3f}] | {100*amr["estimate"]:+.3f} [{100*amr["ci95"][0]:+.3f}, {100*amr["ci95"][1]:+.3f}] | {time["estimate"]:.3f}% [{time["ci95"][0]:.3f}, {time["ci95"][1]:.3f}] |')
    lines += ["", "## All populations, both readouts", "",
        "| Method / readout | AFR recall | AMR recall | EAS recall | EUR recall | SAS recall | Macro-F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["cells"]:
        for view in ("raw", "standardized"):
            values = " | ".join(f'{c[view]["population_recall"][pop]:.3%}' for pop in POPULATIONS)
            lines.append(f'| {c["method"]} / {view} | {values} | {c[view]["macro_f1"]:.6f} |')
    screen = float(np.median([p["screen_seconds"] for p in preparation]))
    lines += ["", "## Costs and boundaries", "",
        f"Median full-width screen cost: {screen:.6f} s (one observation per split, including full-width reference projection).",
        "Screen/ranking, materialization and panel preprocessing are charged equally to all compared methods.",
        "Fit excludes projection, preprocessing, independent audits, input authentication/decompression, rendering/export, download and upstream QC/LD.",
        "No timing is mixed with the historical AutoDL audit or the 455,102-SNP Mac experiment.",
        "Raw and standardized predictions use all eight components; 2D plots are not 2D classification claims.",
        "All sparse fits keep the original 50-step budget. Stationarity counts below are not quality-based selection gates.", "",
        "| Method | Sketch qualified | Full-data qualified | Nonzero coefficients |",
        "| --- | ---: | ---: | ---: |"]
    for c in summary["cells"][1:]:
        lines.append(f'| {c["method"]} | {c["sketch_qualified"]}/20 | {c["full_qualified"]}/20 | {c["nonzero_entry_fraction"]:.3%} |')
    lines += ["", "## Every split's native stratification figures", "",
        "| Split | PC1/2 | PC3/4 | PC5/6 | PC7/8 |", "| --- | --- | --- | --- | --- |"]
    for seed in SEEDS:
        links = " | ".join(f"[PNG](seed_{seed}/pc{pair}.png)" for pair in (12, 34, 56, 78))
        lines.append(f"| {seed} | {links} |")
    lines += ["", "Machine-readable: [all records](results.json), [all paired statistics](summary.json),",
        "[screen and panel provenance](preparation.json), [completion and hashes](completion.json).",
        "The run records source snapshots, input bindings, repeated fits, PCA means and native scores.",
        "No result was used to change a seed, panel, penalty or iteration budget.", ""]
    (output / "README.md").write_text("\n".join(lines))


def run(output, protocol):
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing experiment")
    verify_sources()
    if sklearn.__version__ != "1.7.0":
        raise RuntimeError("use the pinned sklearn 1.7.0 environment")
    environment = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    if set(environment.values()) != {"1"}:
        raise RuntimeError("all four BLAS thread environment variables must be 1")
    reference, verified = load_reference()
    output.mkdir(parents=True)
    try:
        closure = source_closure()
        spec = {"status": "fixed_panel_exposed_split_development_not_confirmation",
            "seeds": list(SEEDS), "rank": RANK, "panel_size": PANEL_SIZE,
            "methods": list(METHODS), "power_iterations": 4, "oversample": 10,
            "max_iter": 50, "relative_tolerance": .001, "timed_repeats": REPEATS,
            "warmups_per_seed_method": 1, "protocol_sha256": digest(protocol),
            "reference_analysis_sha256": historical.FORMAL_ANALYSIS_SHA256,
            "reference_package_anchor_sha256": historical.FORMAL_PACKAGE_ANCHOR_SHA256,
            "reference_validation": verified, "source_sha256": closure,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "sklearn": sklearn.__version__, "platform": platform.platform()},
            "thread_environment": environment}
        write_json(output / "specification_before_run.json", spec)
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in closure:
                archive.write(ROOT / name, name)
            archive.write(protocol, "PROTOCOL.md")
        warnings.simplefilter("error", RuntimeWarning)
        records, preparation = [], []
        with threadpool_limits(1), formal.load_formal_input(INPUT, MANIFEST,
                expected_source_manifest_sha256=formal.FROZEN_FORMAL_MANIFEST_SHA256,
                scratch_root=Path("/private/tmp")) as (data, identity):
            pools = threadpool_info()
            if any(p["num_threads"] != 1 for p in pools):
                raise RuntimeError("uncontrolled computational thread pool")
            write_json(output / "input_and_environment.json", {"input": identity, "threadpools": pools})
            G, labels, sample_ids = data.genotypes, data.population_labels, data.sample_ids
            for seed in SEEDS:
                print(f"seed={seed}: authenticating split and replaying full-width panel screen", flush=True)
                case = historical._reference_case(reference, seed=seed, width=G.shape[1])
                train, test = historical._split_from_reference(labels, sample_ids, case)
                selected = np.flatnonzero(historical._selected_mask(case)).astype(np.int64)
                if len(selected) != PANEL_SIZE:
                    raise ValueError("wrong frozen panel size")
                directory = output / f"seed_{seed}"
                directory.mkdir()
                start = perf_counter()
                full_fit = formal.fit_chunked_top_pca(G, train, test, n_components=RANK, chunk_size=2048)
                replayed, ranking, importance = historical._rank_select_numeric(full_fit, PANEL_SIZE)
                screen_seconds = perf_counter() - start
                _, screen_receipt = historical._verify_selected_indices(replayed, ranking, importance, case)
                full_views = readouts(full_fit.train_scores, full_fit.test_scores, labels, train, test)
                np.savez_compressed(directory / "full_reference_scores.npz", train_scores=full_fit.train_scores,
                    test_scores=full_fit.test_scores, singular_values=full_fit.singular_values)
                panel, X, XT, pre, materialization, preprocessing = prepare_panel(G, train, test, selected)
                np.testing.assert_allclose(pre.mean_dosage, full_fit.mean_dosage[selected], atol=0, rtol=0)
                np.testing.assert_allclose(pre.scale, full_fit.scale[selected], atol=0, rtol=0)
                del full_fit, ranking, importance
                costs = {"seed": seed, "screen_seconds": screen_seconds,
                    "materialization_seconds": materialization, "preprocessing_seconds": preprocessing,
                    "screen_replayed_exact_panel": True, "screen_receipt": screen_receipt,
                    "case": case, "full_exact_pca_context": full_views,
                    "raw_panel_sha256": array_sha256(panel), "selected_indices_sha256": array_sha256(selected)}
                preparation.append(costs)
                np.savez_compressed(directory / "shared_panel.npz", genotypes=panel, labels=labels,
                    sample_ids=sample_ids, train_indices=train, test_indices=test,
                    selected_original_indices=selected, snp_ids=data.snp_ids[selected],
                    mean_dosage=pre.mean_dosage, scale=pre.scale, retained_snps=pre.retained_snps)
                del panel
                print(f"seed={seed}: exact panel replay passed; screen={screen_seconds:.3f}s; fitting {X.shape}", flush=True)
                fits, new_records = run_panel(X, XT, labels, train, test, seed, directory, costs)
                records.extend(new_records)
                write_json(output / "results.json", records)
                write_json(output / "preparation.json", preparation)
                render_native(fits, labels, train, test, seed, directory)
                del fits, X, XT, pre
                gc.collect()
        summary = summarize(records)
        write_json(output / "summary.json", summary)
        write_report(output, summary, preparation)
        if any(digest(ROOT / name) != expected for name, expected in closure.items()):
            raise RuntimeError("source changed during execution")
        if digest(protocol) != spec["protocol_sha256"]:
            raise RuntimeError("frozen protocol changed during execution")
        write_json(output / "completion.json", {"status": "complete_frozen_5pct_panel_development",
            "records": len(records), "seeds": list(SEEDS), "timed_fits": len(records) * REPEATS,
            "excluded_warmup_fits": len(records), "exact_panel_replays": len(preparation),
            "files": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        print(f"COMPLETE: {output}; {len(records)} configurations, {len(records)*REPEATS} timed fits", flush=True)
    except BaseException as error:
        write_json(output / "FAILED.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "partial_records": len(locals().get("records", [])), "policy": "preserved; no automatic rerun"})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "paired_20seeds")
    parser.add_argument("--protocol", type=Path, default=BASE / "PROTOCOL.md")
    args = parser.parse_args()
    run(args.output, args.protocol)

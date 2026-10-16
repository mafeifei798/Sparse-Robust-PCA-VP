"""Own-loading randomized VP-SPCA 5% discovery versus literal full default PCA."""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
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
from threadpoolctl import threadpool_info, threadpool_limits

import genomics_randomized_vp_5pct as shared
from genomics_randomized_vp_matched import COLORS, POPULATIONS, digest, write_json
from rspca.datasets import array_sha256
from rspca.genomics_aim import select_loading_panel
from rspca.genomics_cached import standardized_genotype_cache
from rspca.randomized import _full_diagnostics

BASE = ROOT / "results/genomics/selected5_replay"
SEEDS = tuple(range(2025, 2045))
RANK, PANEL_SIZE, REPEATS, BOOTSTRAP_SEED = 8, 22755, 3, 9082026
PENALTIES = (.0002, .001, .005)
METHODS = (("PCA-default-all", None),) + tuple((f"RVP-select5-c{c:g}", c) for c in PENALTIES)
STAGES = ("full_fit", "ranking", "materialization", "panel_refit", "projection")


def pack_fit(model, A, B, mean, penalty):
    if penalty is None:
        diagnostics = {"parameters": model.get_params(), "resolved_solver": model._fit_svd_solver}
    else:
        diagnostics = {"alpha": model.alpha, "iterations": model.n_iter,
            "sketch_converged": model.sketch_converged,
            "sketch_eta": model.sketch_relative_stationarity,
            "sketch_objectives_scaled": model.sketch_objectives_scaled,
            "restarts": model.restarts, "timings": model.timings,
            "preparation_diagnostics": model.preparation_diagnostics}
    return {"A": A, "B": B, "mean": mean, "solver": diagnostics}


def fit_select_refit(X, XT, seed, penalty, *, panel_size=PANEL_SIZE):
    """Labels cannot enter either unsupervised fit or the own-B ranking.

    XT is not accessed until the training-only mask is fixed. No PCA is called
    by a VP arm; each stage invokes the same independently seeded VP solver.
    """
    start, cpu = perf_counter(), process_time()
    stage = {key: 0. for key in STAGES}
    tick = perf_counter()
    full = pack_fit(*shared.fit_method(X, seed, penalty), penalty)
    stage["full_fit"] = perf_counter() - tick
    selected, importance = None, None
    if penalty is not None:
        tick = perf_counter()
        selected, importance = select_loading_panel(full["B"], panel_size, score="l1")
        stage["ranking"] = perf_counter() - tick
        tick = perf_counter()
        XP = np.ascontiguousarray(X[:, selected])
        XTP = np.ascontiguousarray(XT[:, selected])
        XP.setflags(write=False)
        XTP.setflags(write=False)
        stage["materialization"] = perf_counter() - tick
        tick = perf_counter()
        final = pack_fit(*shared.fit_method(XP, seed, penalty), penalty)
        stage["panel_refit"] = perf_counter() - tick
    else:
        XP, XTP, final = X, XT, full
    tick = perf_counter()
    final["train_scores"], final["test_scores"] = shared.project(XP, XTP, final["B"], final["mean"])
    stage["projection"] = perf_counter() - tick
    wall, used_cpu = perf_counter() - start, process_time() - cpu
    return {"final": final, "selector": full if penalty is not None else None,
            "selected": selected, "importance": importance,
            "stage_seconds": stage, "pipeline_seconds": wall, "pipeline_cpu_seconds": used_cpu}


def compare_repetition(first, current):
    for role in ("selector", "final"):
        if first[role] is None:
            if current[role] is not None:
                raise ValueError("changed selector role")
            continue
        for key in ("A", "B", "mean", "train_scores", "test_scores"):
            if key in first[role]:
                np.testing.assert_allclose(current[role][key], first[role][key], rtol=1e-10, atol=1e-10)
    if first["selected"] is not None:
        np.testing.assert_array_equal(current["selected"], first["selected"])
        np.testing.assert_allclose(current["importance"], first["importance"], rtol=1e-12, atol=1e-12)


def audit_fit(X, fit, penalty):
    start = perf_counter()
    norm = float(linalg.norm(X))
    if penalty is None:
        residual2 = max(0., norm**2 - X.shape[0] * float(fit["mean"] @ fit["mean"])
                        - float(np.sum(fit["train_scores"]**2)))
        eta = None
    else:
        alpha = fit["solver"]["alpha"]
        _, objective, stationarity = _full_diagnostics(X, fit["A"], fit["B"], alpha, 1 / norm**2, norm)
        residual2 = max(0., 2 * (objective - alpha * float(np.abs(fit["B"]).sum())))
        eta = float(stationarity / (alpha * np.sqrt(fit["B"].size)))
    return {"full_eta": eta, "full_stationarity_qualified": bool(eta <= .001) if eta is not None else None,
        "relative_reconstruction_error_on_fitted_features": float(np.sqrt(residual2) / norm),
        "nonzero_entry_fraction": float(np.mean(fit["B"] != 0)),
        "nonzero_snp_fraction": float(np.mean(np.any(fit["B"] != 0, axis=1))),
        "audit_seconds": perf_counter() - start}


def run_seed(X, XT, labels, train, test, seed, output, retained_original, preparation_seconds):
    fits, times, schedule = {}, {}, []
    repeats = range(-1 if seed == SEEDS[0] else 0, REPEATS)
    for repeat in repeats:
        order = np.random.default_rng(seed + (repeat + 1) * 10000).permutation(len(METHODS))
        schedule.append({"repeat": repeat, "warmup": repeat == -1, "order": [METHODS[i][0] for i in order]})
        for index in order:
            name, penalty = METHODS[index]
            gc.collect()
            result = fit_select_refit(X, XT, seed, penalty, panel_size=PANEL_SIZE)
            tick, cpu = perf_counter(), process_time()
            views = shared.readouts(result["final"]["train_scores"], result["final"]["test_scores"], labels, train, test)
            readout_seconds, readout_cpu = perf_counter() - tick, process_time() - cpu
            timing = {**result["stage_seconds"], "readout": readout_seconds,
                "model_computation": result["pipeline_seconds"] + readout_seconds,
                "model_cpu": result["pipeline_cpu_seconds"] + readout_cpu}
            print(f"seed={seed} repeat={repeat} {name} total={timing['model_computation']:.3f}s "
                  f"full_fit={timing['full_fit']:.3f}s refit={timing['panel_refit']:.3f}s", flush=True)
            if repeat == -1:
                del result
                continue
            if name not in fits:
                fits[name] = result
                fits[name]["readouts"] = views
                times[name] = []
            else:
                compare_repetition(fits[name], result)
                if views != fits[name]["readouts"]:
                    raise RuntimeError("repeated readouts changed")
            times[name].append(timing)
            del result
    records = []
    for name, penalty in METHODS:
        fit = fits[name]
        start = perf_counter()
        final = fit["final"]
        if penalty is None:
            final_audit = audit_fit(X, final, None)
            selector_audit, full_views, selection = None, None, None
        else:
            selector_audit = audit_fit(X, fit["selector"], penalty)
            full_scores = shared.project(X, XT, fit["selector"]["B"], fit["selector"]["mean"])
            full_views = shared.readouts(*full_scores, labels, train, test)
            fit["selector"]["train_scores"], fit["selector"]["test_scores"] = full_scores
            selected, importance = select_loading_panel(fit["selector"]["B"], PANEL_SIZE, score="l1")
            np.testing.assert_array_equal(selected, fit["selected"])
            np.testing.assert_array_equal(importance, fit["importance"])
            panel_train = np.ascontiguousarray(X[:, selected])
            panel_test = np.ascontiguousarray(XT[:, selected])
            replay = shared.project(panel_train, panel_test, final["B"], final["mean"])
            np.testing.assert_allclose(replay[0], final["train_scores"], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(replay[1], final["test_scores"], rtol=1e-12, atol=1e-12)
            final_audit = audit_fit(panel_train, final, penalty)
            original = retained_original[selected]
            selection = {"source": "own full-training randomized VP-SPCA encoder B",
                "rule": "sum(abs(B), axis=1); ties by original SNP index", "count": len(selected),
                "selected_retained_indices_sha256": array_sha256(selected),
                "selected_original_indices_sha256": array_sha256(original),
                "selector_B_sha256": array_sha256(fit["selector"]["B"]),
                "importance_sha256": array_sha256(importance),
                "panel_train_sha256": array_sha256(panel_train), "panel_test_sha256": array_sha256(panel_test)}
            del panel_train, panel_test
        audit_seconds = perf_counter() - start
        if penalty is not None:
            np.savez_compressed(output / f"{name}-selection.npz", importance=importance,
                selected_retained_indices=selected, selected_original_indices=original)
        medians = {key: float(np.median([t[key] for t in times[name]])) for key in times[name][0]}
        discovery = preparation_seconds + medians["model_computation"]
        record = {"seed": seed, "method": name, "rank": RANK, "penalty_fraction": penalty,
            "original_snps": 455102, "training_retained_snps": X.shape[1], "final_input_snps": final["B"].shape[0],
            "timings": times[name], "timing_medians": medians,
            "preparation_seconds": preparation_seconds, "discovery_workflow_seconds": discovery,
            "audit_seconds": audit_seconds, "audited_discovery_seconds": discovery + audit_seconds,
            "selection": selection, "readouts": fit["readouts"], "full_vp_context": full_views,
            "final_solver": final["solver"], "final_audit": final_audit,
            "selector_solver": fit["selector"]["solver"] if penalty is not None else None,
            "selector_audit": selector_audit}
        for role in ("final", "selector"):
            if fit[role] is not None:
                arrays = {key: fit[role][key] for key in ("A", "B", "mean", "train_scores", "test_scores")}
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise RuntimeError("nonfinite saved model")
                np.savez_compressed(output / f"{name}-{role}.npz", **arrays)
        fit["record"] = record
        records.append(record)
    write_json(output / "timing_schedule.json", schedule)
    write_json(output / "records.json", records)
    return fits, records


def interval(values, *, statistic="mean", resamples=100000):
    values = np.asarray(values, dtype=float)
    reducer = np.mean if statistic == "mean" else np.median
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(values), size=(resamples, len(values)))
    distribution = reducer(values[indices], axis=1)
    return {"estimate": float(reducer(values)), "ci95": np.quantile(distribution, [.025, .975]).tolist()}


def summarize(records):
    mapping = {(r["seed"], r["method"]): r for r in records}
    expected = {(seed, name) for seed in SEEDS for name, _ in METHODS}
    if len(records) != len(expected) or set(mapping) != expected:
        raise ValueError("incomplete or duplicated own-selection grid")
    cells, contrasts = [], []
    for name, penalty in METHODS:
        rows = [mapping[seed, name] for seed in SEEDS]
        cell = {"method": name, "penalty_fraction": penalty, "n_seeds": len(rows),
            "final_input_snps": rows[0]["final_input_snps"],
            "stage_medians": {key: float(np.median([r["timing_medians"][key] for r in rows])) for key in rows[0]["timing_medians"]},
            "discovery_workflow_median": float(np.median([r["discovery_workflow_seconds"] for r in rows])),
            "audited_discovery_median": float(np.median([r["audited_discovery_seconds"] for r in rows]))}
        for view in ("raw", "standardized"):
            matrix = np.sum([r["readouts"][view]["confusion"] for r in rows], axis=0)
            cell[view] = {key: float(np.mean([r["readouts"][view][key] for r in rows])) for key in ("balanced_accuracy", "macro_f1")}
            cell[view].update({"confusion": matrix.tolist(),
                "population_recall": {pop: float(np.mean([r["readouts"][view]["population_recall"][pop] for r in rows])) for pop in POPULATIONS},
                "population_precision": {pop: float(matrix[i, i] / max(1, matrix[:, i].sum())) for i, pop in enumerate(POPULATIONS)}})
        for role in ("selector", "final"):
            if rows[0][f"{role}_audit"] is not None:
                cell[f"{role}_audit"] = {"full_qualified": sum(r[f"{role}_audit"]["full_stationarity_qualified"] is True for r in rows),
                    "sketch_qualified": sum(r[f"{role}_solver"].get("sketch_converged") is True for r in rows),
                    **{key: float(np.mean([r[f"{role}_audit"][key] for r in rows])) for key in ("nonzero_entry_fraction", "nonzero_snp_fraction")}}
        cells.append(cell)
        if penalty is not None:
            baseline = [mapping[seed, "PCA-default-all"] for seed in SEEDS]
            contrast = {"method": name, "baseline": "PCA-default-all", "runtime": {}, "quality": {}, "compression_quality_delta": {}}
            for scope in ("model_computation", "discovery_workflow_seconds", "audited_discovery_seconds"):
                extract = (lambda row: row["timing_medians"][scope]) if scope == "model_computation" else (lambda row: row[scope])
                reductions = [100 * (1 - extract(r) / extract(b)) for r, b in zip(rows, baseline, strict=True)]
                contrast["runtime"][scope] = {**interval(reductions, statistic="median"), "per_seed_reduction_percent": reductions}
            for view in ("raw", "standardized"):
                quality = {}
                for key in ("balanced_accuracy", "macro_f1"):
                    quality[key] = interval([r["readouts"][view][key] - b["readouts"][view][key] for r, b in zip(rows, baseline, strict=True)])
                for pop in POPULATIONS:
                    quality[f"recall_{pop}"] = interval([r["readouts"][view]["population_recall"][pop] - b["readouts"][view]["population_recall"][pop] for r, b in zip(rows, baseline, strict=True)])
                quality["prediction_disagreements"] = int(sum(np.count_nonzero(np.asarray(r["readouts"][view]["predictions"]) != np.asarray(b["readouts"][view]["predictions"])) for r, b in zip(rows, baseline, strict=True)))
                contrast["quality"][view] = quality
                contrast["compression_quality_delta"][view] = {key: interval([r["readouts"][view][key] - r["full_vp_context"][view][key] for r in rows]) for key in ("balanced_accuracy", "macro_f1")}
            contrasts.append(contrast)
    return {"cells": cells, "contrasts": contrasts, "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": 100000, "independent_population_validation": False}


def render_native(fits, labels, train, test, seed, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for first in range(0, RANK, 2):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10.6), constrained_layout=True)
        for ax, (name, penalty) in zip(axes.flat, METHODS, strict=True):
            fit = fits[name]["final"]
            B = fit["B"]
            signs = np.sign(B[np.abs(B).argmax(axis=0), np.arange(RANK)])
            signs[signs == 0] = 1
            for split, indices in (("train", train), ("test", test)):
                scores = fit[f"{split}_scores"] * signs
                for pop, color in zip(POPULATIONS, COLORS, strict=True):
                    mask = labels[indices] == pop
                    ax.scatter(scores[mask, first], scores[mask, first+1], color=color,
                        s=16 if split == "train" else 29, alpha=.55 if split == "train" else 1,
                        edgecolors="none" if split == "train" else "black",
                        linewidths=0 if split == "train" else .4, label=pop if split == "test" else None)
            view = fits[name]["record"]["readouts"]["standardized"]
            title = "Default PCA | all SNPs" if penalty is None else f"RVP own 5% selection + RVP refit | c={penalty:g}"
            ax.set_title(f'{title}\n8D standardized: BA {view["balanced_accuracy"]:.1%} | AMR recall {view["population_recall"]["AMR"]:.1%}', fontsize=11)
            ax.set_xlabel(f"Native component {first+1}")
            ax.set_ylabel(f"Native component {first+2}")
            ax.spines[["top", "right"]].set_visible(False)
        handles, legends = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legends, loc="outside lower center", ncol=5)
        fig.suptitle(f"Randomized VP-SPCA discovers 22,755 / 455,102 SNPs | seed {seed}\nNative coordinates, per-panel scales; signs only; outlines: held-out samples", fontsize=14)
        fig.savefig(directory / f"pc{first+1}{first+2}.png", dpi=220)
        plt.close(fig)


def write_report(output, summary):
    lines = ["# Randomized VP-SPCA own 5% SNP selection and refit", "",
        "All 20 exposed splits, all three frozen penalties, rank 8. PCA uses all SNPs; VP selects its own 22,755 SNPs from full-training B, then independently refits.", "",
        "| Method | Raw BA | Standardized BA | Standardized AMR recall | Model computation s | Prepared-input discovery s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["cells"]:
        lines.append(f'| {c["method"]} | {c["raw"]["balanced_accuracy"]:.3%} | {c["standardized"]["balanced_accuracy"]:.3%} | {c["standardized"]["population_recall"]["AMR"]:.3%} | {c["stage_medians"]["model_computation"]:.6f} | {c["discovery_workflow_median"]:.6f} |')
    lines += ["", "## Paired comparisons with full-input default PCA", "",
        "Descriptive 95% bootstrap intervals over overlapping split IDs; not independent people or a non-inferiority test.", "",
        "| Method | Standardized BA delta pp | AMR recall delta pp | Model computation reduction % | Prepared-input discovery reduction % |",
        "| --- | --- | --- | --- | --- |"]
    for c in summary["contrasts"]:
        columns = []
        for value, multiplier in ((c["quality"]["standardized"]["balanced_accuracy"], 100),
                                  (c["quality"]["standardized"]["recall_AMR"], 100),
                                  (c["runtime"]["model_computation"], 1),
                                  (c["runtime"]["discovery_workflow_seconds"], 1)):
            columns.append(f'{multiplier*value["estimate"]:+.3f} [{multiplier*value["ci95"][0]:+.3f}, {multiplier*value["ci95"][1]:+.3f}]')
        lines.append(f'| {c["method"]} | ' + " | ".join(columns) + " |")
    lines += ["", "## All populations, both 8D readouts", "",
        "| Method / readout | AFR | AMR | EAS | EUR | SAS | Macro-F1 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for c in summary["cells"]:
        for view in ("raw", "standardized"):
            values = " | ".join(f'{c[view]["population_recall"][pop]:.3%}' for pop in POPULATIONS)
            lines.append(f'| {c["method"]} / {view} | {values} | {c[view]["macro_f1"]:.6f} |')
    lines += ["", "## Boundaries and every figure", "",
        "The 5% feature budget is top-K of VP's own loadings, not intrinsic 95% row sparsity. Neither stage uses PCA for selection or initialization.",
        "Times include full selector fitting, ranking, matrix materialization, panel refitting, projection and readouts. Discovery additionally charges common full-width training preprocessing/cache construction.",
        "Authentication/decompression, QC/LD/download, exports and figures are excluded. Independent scientific audits are separately recorded. The 50-step budget is not time-to-convergence; both stage diagnostics are in results.json.",
        "Only deterministic signs change in plots; each panel has native axis limits. 8D classification is not a claim of separation in any one 2D view.",
        "No penalty, split, loading score or budget was selected after inspecting outcomes.", "",
        "| Seed | PC1/2 | PC3/4 | PC5/6 | PC7/8 |", "| --- | --- | --- | --- | --- |"]
    for seed in SEEDS:
        lines.append(f"| {seed} | " + " | ".join(f"[PNG](seed_{seed}/pc{pair}.png)" for pair in (12, 34, 56, 78)) + " |")
    lines += ["", "[All records](results.json) · [All paired statistics](summary.json) · [Input preparation](preparation.json) · [Completion and hashes](completion.json)", ""]
    (output / "README.md").write_text("\n".join(lines))


def run(output, protocol, *, input_path, reference, manifest=None):
    input_path = Path(input_path)
    manifest = input_path.with_suffix(".manifest.json") if manifest is None else Path(manifest)
    if output.exists():
        raise FileExistsError("refusing to overwrite an existing experiment")
    shared.verify_sources()
    if sklearn.__version__ != "1.7.0":
        raise RuntimeError("use pinned sklearn 1.7.0")
    environment = {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    if set(environment.values()) != {"1"}:
        raise RuntimeError("all four thread environment variables must equal 1")
    reference, verified = shared.load_reference(reference)
    output.mkdir(parents=True)
    try:
        closure = shared.source_closure()
        closure[str(Path(__file__).resolve().relative_to(ROOT))] = digest(Path(__file__))
        spec = {"status": "own_rvp_selection_exposed_split_development_not_confirmation",
            "seeds": list(SEEDS), "rank": RANK, "panel_size": PANEL_SIZE,
            "methods": list(METHODS), "power_iterations": 4, "oversample": 10,
            "max_iter": 50, "relative_tolerance": .001, "check_every": 10,
            "timed_repeats": REPEATS, "excluded_warmup_pipelines": len(METHODS),
            "warmup_split": SEEDS[0], "selection": "own full-training B L1 row score, exact top-K",
            "protocol_sha256": digest(protocol), "reference_validation": verified,
            "source_sha256": closure, "thread_environment": environment,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                        "sklearn": sklearn.__version__, "platform": platform.platform()}}
        write_json(output / "specification_before_run.json", spec)
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in closure:
                archive.write(ROOT / name, name)
            archive.write(protocol, "PROTOCOL.md")
        warnings.simplefilter("error", RuntimeWarning)
        records, preparation = [], []
        with threadpool_limits(1), shared.formal.load_formal_input(input_path, manifest,
                expected_source_manifest_sha256=shared.formal.FROZEN_FORMAL_MANIFEST_SHA256,
                scratch_root=Path("/private/tmp")) as (data, identity):
            pools = threadpool_info()
            if any(p["num_threads"] != 1 for p in pools):
                raise RuntimeError("uncontrolled computational thread pool")
            write_json(output / "input_and_environment.json", {"input": identity, "threadpools": pools})
            G, labels, sample_ids = data.genotypes, data.population_labels, data.sample_ids
            if G.shape != (1000, 455102):
                raise ValueError("wrong frozen input dimensions")
            np.savez_compressed(output / "cohort.npz", labels=labels, sample_ids=sample_ids,
                                snp_ids=data.snp_ids, chromosomes=data.chromosomes)
            for seed in SEEDS:
                split = [r for r in reference["split_receipts"] if r["seed"] == seed]
                if len(split) != 1:
                    raise ValueError("missing or duplicate authenticated split")
                train, test = shared.historical._split_from_reference(labels, sample_ids, split[0])
                print(f"seed={seed}: preparing common full-width float64 input; no PCA panel is loaded", flush=True)
                directory = output / f"seed_{seed}"
                directory.mkdir()
                with standardized_genotype_cache(G, train, test, scratch_root="/private/tmp") as cache:
                    pre = cache.preprocessing
                    retained = np.flatnonzero(pre.retained_snps).astype(np.int64)
                    if array_sha256(retained) != split[0]["retained_original_indices_sha256"]:
                        raise ValueError("training preprocessing differs from frozen split")
                    np.savez_compressed(directory / "preprocessing.npz", train_indices=train, test_indices=test,
                        retained_original_indices=retained, mean_dosage=pre.mean_dosage,
                        scale=pre.scale, retained_snps=pre.retained_snps)
                    preparation.append({"seed": seed, "seconds": cache.preparation_seconds,
                        "timings": cache.timings, "identity": cache.identity, "split_receipt": split[0]})
                    print(f"seed={seed}: cache ready {cache.train.shape}, preparation={cache.preparation_seconds:.3f}s", flush=True)
                    fits, new_records = run_seed(cache.train, cache.test, labels, train, test,
                        seed, directory, retained, cache.preparation_seconds)
                    records.extend(new_records)
                    write_json(output / "results.json", records)
                    write_json(output / "preparation.json", preparation)
                    render_native(fits, labels, train, test, seed, directory)
                    del fits
                gc.collect()
                print(f"seed={seed}: completed, independently checked and saved", flush=True)
        summary = summarize(records)
        write_json(output / "summary.json", summary)
        write_report(output, summary)
        if any(digest(ROOT / name) != expected for name, expected in closure.items()):
            raise RuntimeError("source changed during execution")
        if digest(protocol) != spec["protocol_sha256"]:
            raise RuntimeError("protocol changed during execution")
        write_json(output / "completion.json", {"status": "complete_own_rvp_selection_development",
            "records": len(records), "seeds": list(SEEDS), "timed_pipelines": len(records) * REPEATS,
            "timed_solver_fits": len(SEEDS) * REPEATS * (1 + 2 * len(PENALTIES)),
            "excluded_warmup_pipelines": len(METHODS), "excluded_warmup_solver_fits": 1 + 2 * len(PENALTIES),
            "files": {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}})
        print(f"COMPLETE: {output}; {len(records)} configurations, {len(records)*REPEATS} timed pipelines", flush=True)
    except BaseException as error:
        write_json(output / "FAILED.json", {"error": repr(error), "traceback": traceback.format_exc(),
            "partial_records": len(locals().get("records", [])), "policy": "preserved; no automatic rerun"})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE / "paired_20seeds")
    parser.add_argument("--protocol", type=Path, required=True, help="Original protocol file")
    parser.add_argument("--input", dest="input_path", type=Path, required=True, help="Authenticated genotype NPZ archive")
    parser.add_argument("--manifest", type=Path, help="Input manifest; defaults to the NPZ path with suffix .manifest.json")
    parser.add_argument("--reference", type=Path, required=True, help="Authenticated full-SNP reference package")
    args = parser.parse_args(argv)
    return run(args.output, args.protocol, input_path=args.input_path,
               reference=args.reference, manifest=args.manifest)


if __name__ == "__main__":
    main()

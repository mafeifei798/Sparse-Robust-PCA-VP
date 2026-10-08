"""Complete same-quality runtime grid with literal default sklearn PCA."""

import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import platform
import resource
from time import perf_counter
import warnings
import zipfile

import numpy as np
from scipy import linalg
import scipy
import sklearn
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_info, threadpool_limits

import genomics_pca_loading_all_autosomes as formal_input
from genomics_randomized_vp_matched import COLORS, POPULATIONS, digest, metrics, write_json
from genomics_randomized_vp_amr_grid import summarize
from rspca.genomics_aim import stratified_split, fit_smartpca_preprocessor, transform_smartpca
from rspca.genomics_cached import standardized_genotype_cache
from rspca.randomized import _full_diagnostics
from rspca.randomized_reduced import fit_reduced_randomized_vp_spca
from rspca.randomized_reduced_fast import fit_fast_randomized_vp_spca


PENALTIES = [.0002, .001, .005]
METHODS = [("PCA-default", "pca", None)] + [
    (f"RVP-{implementation}-c{penalty:g}", implementation, penalty)
    for implementation in ("reference", "optimized") for penalty in PENALTIES]


def equivalence(reference, optimized):
    signs = np.sign(np.sum(reference["B"] * optimized["B"], axis=0))
    signs[signs == 0] = 1
    report = {}
    for field in ("B", "A", "train_scores", "test_scores"):
        left, right = reference[field], optimized[field] * signs
        atol = 1e-8 if field in ("B", "A") else 1e-6
        np.testing.assert_allclose(right, left, atol=atol, rtol=1e-6, err_msg=field)
        report[f"{field}_max_absolute_difference"] = float(np.max(np.abs(right-left)))
        relative = float(linalg.norm(right-left) / max(linalg.norm(left), np.finfo(float).tiny))
        report[f"{field}_relative_frobenius_difference"] = relative
        if field.endswith("scores") and relative > 1e-8:
            raise AssertionError(f"{field} relative error exceeds 1e-8")
    np.testing.assert_array_equal(reference["B"] != 0, optimized["B"] != 0)
    before, after = reference["record"], optimized["record"]
    for view in ("raw_centroid", "std_centroid"):
        if before[view] != after[view]:
            raise AssertionError(f"{view} predictions or metrics changed")
    old, new = before["solver"], after["solver"]
    np.testing.assert_allclose(new["sketch_objectives_scaled"], old["sketch_objectives_scaled"], atol=1e-9, rtol=1e-9)
    if abs(new["alpha"]/old["alpha"] - 1) > 1e-10:
        raise AssertionError("alpha changed")
    np.testing.assert_allclose(new["sketch_eta"], old["sketch_eta"], atol=1e-8, rtol=1e-6)
    np.testing.assert_allclose(after["full_eta"], before["full_eta"], atol=1e-8, rtol=1e-6)
    if (new["iterations"], new["sketch_converged"], after["full_stationarity_qualified"]) != (
        old["iterations"], old["sketch_converged"], before["full_stationarity_qualified"]):
        raise AssertionError("budget/convergence semantics changed")
    report.update({"status": "passed", "all_readout_predictions_identical": True,
                   "nonzero_mask_identical": True, "seed": before["seed"],
                   "rank": before["rank"], "reference": before["method"],
                   "optimized": after["method"], "reference_restarts": old["restarts"],
                   "optimized_restarts": new["restarts"]})
    return report


def plot_pairs(fits, labels, train, test, rank, seed, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    locations = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    for first in range(0, rank, 2):
        fig, axes = plt.subplots(2, 4, figsize=(20, 9.4), constrained_layout=True)
        axes[1, 0].axis("off")
        axes[1, 0].text(.02, .65, "Main comparator:\ndefault sklearn PCA\n\nTop: unchanged randomized VP\nBottom: optimized randomized VP\n\nSame float64 SNP input\nq = 4, 50-step budget\nOutlined points are held out",
                        transform=axes[1, 0].transAxes, fontsize=12, va="top")
        all_points = []
        for (name, _, _), location in zip(METHODS, locations, strict=True):
            ax, fit = axes[location], fits[name]
            B = fit["B"]
            signs = np.sign(B[np.argmax(np.abs(B), axis=0), np.arange(rank)])
            signs[signs == 0] = 1
            for split, rows in (("train", train), ("test", test)):
                scores = fit[f"{split}_scores"] * signs
                all_points.append(scores[:, first:first+2])
                for pop, color in zip(POPULATIONS, COLORS, strict=True):
                    mask = labels[rows] == pop
                    ax.scatter(scores[mask, first], scores[mask, first+1], c=color,
                               s=13 if split == "train" else 24, alpha=.5 if split == "train" else 1,
                               edgecolors="none" if split == "train" else "black",
                               linewidths=0 if split == "train" else .35,
                               label=pop if split == "test" else None)
            r = fit["record"]
            title = name.replace("RVP-reference-", "Randomized VP/reference\n").replace("RVP-optimized-", "Randomized VP/optimized\n")
            ax.set_title(f'{title}\nAMR raw/std {r["raw_centroid"]["population_recall"]["AMR"]:.3f}/{r["std_centroid"]["population_recall"]["AMR"]:.3f}; fit {r["fit_seconds_median"]:.3f}s', fontsize=10)
            ax.set_xlabel(f"Native component {first+1}")
            ax.set_ylabel(f"Native component {first+2}")
            ax.spines[["top", "right"]].set_visible(False)
        combined = np.vstack(all_points)
        for column in (0, 1):
            lo, hi = combined[:, column].min(), combined[:, column].max()
            margin = .06 * max(hi-lo, 1)
            for location in locations:
                (axes[location].set_xlim if column == 0 else axes[location].set_ylim)(lo-margin, hi+margin)
        handles, legend_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, legend_labels, loc="outside lower center", ncol=5)
        fig.suptitle(f"Default PCA versus randomized VP-SPCA | rank {rank} | seed {seed}\nNative X @ B coordinates; signs only, shared limits, no sample selection")
        fig.savefig(directory/f"seed_{seed}_pc{first+1}{first+2}.png", dpi=180)
        plt.close(fig)


@contextmanager
def load_input(args):
    if args.kind == "full":
        with formal_input.load_formal_input(args.input, args.manifest,
            expected_source_manifest_sha256=formal_input.FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=Path("/private/tmp")) as (data, identity):
            yield data.genotypes, data.population_labels, data.sample_ids, identity
    else:
        with np.load(args.input, allow_pickle=False) as data:
            yield data["genotypes"], data["population_labels"].astype(str), data["sample_ids"].astype(str), {"source": "frozen chr20-22 NPZ"}


@contextmanager
def prepare(raw, train, test, kind):
    if kind == "full":
        with standardized_genotype_cache(raw, train, test, scratch_root=Path("/private/tmp")) as cache:
            yield cache.train, cache.test, cache.preprocessing, cache.preparation_seconds, {
                "cache": cache.identity, "timings": cache.timings,
                "scope": "training statistics + materialization + block integrity hashing + flush; charged equally to all methods"}
    else:
        start = perf_counter()
        pre = fit_smartpca_preprocessor(raw[train])
        X = np.ascontiguousarray(transform_smartpca(raw[train], pre))
        XT = np.ascontiguousarray(transform_smartpca(raw[test], pre))
        X.setflags(write=False)
        XT.setflags(write=False)
        yield X, XT, pre, perf_counter()-start, {"scope": "same training-only preprocessing as previous dense grid"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("dense", "full"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    expected_seeds = list(range(2025, 2030)) if args.kind == "dense" else [2025]
    if args.seeds != expected_seeds or args.repeats != 3:
        parser.error("this fixed grid requires dense seeds 2025..2029 or full seed 2025, with three repeats")
    if args.kind == "full" and args.manifest is None:
        parser.error("full input requires its authenticated manifest")
    if sklearn.__version__ != "1.7.0":
        parser.error("the recorded timing environment is sklearn 1.7.0")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ["src/rspca/randomized_reduced_fast.py", "src/rspca/randomized_reduced.py",
               "src/rspca/randomized.py", "src/rspca/vp.py", "src/rspca/genomics_aim.py",
               "src/rspca/genomics_blocked.py", "src/rspca/genomics_cached.py",
               "experiments/genomics_randomized_vp_default_runtime.py",
               "experiments/genomics_randomized_vp_matched.py", "experiments/genomics_randomized_vp_amr_grid.py",
               "experiments/genomics_pca_loading_all_autosomes.py", "experiments/genomics_pca_loading_grid.py"]
    expected_sha = formal_input.FROZEN_FORMAL_NPZ_SHA256 if args.kind == "full" else "b89d1557689b9ee98cefd2ae99f23b1f41b0cb0f61c3d0ab4cd3b8667429089e"
    if digest(args.input) != expected_sha:
        raise ValueError("input identity mismatch")
    spec = {"status": "implementation_only_exposed_seed_development_not_confirmation",
            "kind": args.kind, "input_sha256": expected_sha, "seeds": args.seeds,
            "ranks": [4, 6, 8], "repeats": 3, "methods": METHODS, "power_iterations": 4,
            "penalty_fractions": PENALTIES, "oversample": 10, "max_iter": 50,
            "primary_baseline": "literal PCA(n_components=rank, random_state=seed), all other sklearn defaults unchanged",
            "precision": "float64 input and arithmetic for every method",
            "workflow": "preprocessing/cache + independent fit + native projection + common raw/std readouts; data authentication, extraction, rendering and export excluded",
            "audit": "separate full-data stationarity with independently charged Frobenius norm; no PCA spectral norm borrowed",
            "protocol_sha256": digest(args.protocol), "source_sha256": {p: digest(root/p) for p in sources},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    write_json(args.output/"specification_before_run.json", spec)
    with zipfile.ZipFile(args.output/"source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(root/p, p)
        archive.write(args.protocol, "PROTOCOL.md")
    warnings.simplefilter("error", RuntimeWarning)
    records, guards = [], []
    with threadpool_limits(1), load_input(args) as (G, labels, sample_ids, identity):
        pools = threadpool_info()
        if any(p["num_threads"] != 1 for p in pools):
            raise RuntimeError("uncontrolled thread pool")
        write_json(args.output/"input_and_environment.json", {"input": identity, "threadpools": pools})
        for seed in args.seeds:
            train, test = stratified_split(labels, .2, seed)
            with prepare(G, train, test, args.kind) as (X, XT, pre, prep_seconds, prep_identity):
                write_json(args.output/f"seed_{seed}_preparation.json", {"seconds": prep_seconds, **prep_identity})
                np.savez_compressed(args.output/f"seed_{seed}_preprocessing.npz", train_indices=train,
                    test_indices=test, mean_dosage=pre.mean_dosage, scale=pre.scale, retained_snps=pre.retained_snps)
                started = perf_counter()
                norm = float(linalg.norm(X))
                norm_seconds = perf_counter()-started
                for rank in spec["ranks"]:
                    directory = args.output/f"rank_{rank}"
                    directory.mkdir(exist_ok=True)
                    fits = {}
                    for repeat in range(3):
                        order = np.random.default_rng(seed+rank*100+repeat*10000).permutation(len(METHODS))
                        for index in order:
                            name, implementation, penalty = METHODS[index]
                            gc.collect()
                            print(f"seed={seed} rank={rank} repeat={repeat} method={name} shape={X.shape}", flush=True)
                            start = perf_counter()
                            if implementation == "pca":
                                model = PCA(n_components=rank, random_state=seed)
                                model.fit(X)
                                B, A = model.components_.T, model.components_.T
                            else:
                                fit_function = fit_reduced_randomized_vp_spca if implementation == "reference" else fit_fast_randomized_vp_spca
                                model = fit_function(X, rank, penalty_fraction=penalty,
                                                     n_power_iter=4, max_iter=50, random_state=seed)
                                B, A = model.B, model.A
                            seconds = perf_counter()-start
                            start = perf_counter()
                            train_scores, test_scores = X @ B, XT @ B
                            projection = perf_counter()-start
                            if not all(np.isfinite(a).all() for a in (A, B, train_scores, test_scores)):
                                raise RuntimeError("nonfinite fit/projection")
                            if name not in fits:
                                diagnostic = {"resolved_solver": model._fit_svd_solver,
                                    "parameters": model.get_params(), "largest_singular_value": float(model.singular_values_[0])} if implementation == "pca" else {
                                    "alpha": model.alpha, "sketch_converged": model.sketch_converged,
                                    "sketch_eta": model.sketch_relative_stationarity, "iterations": model.n_iter,
                                    "restarts": model.restarts, "solver_timings": model.timings,
                                    "sketch_objectives_scaled": model.sketch_objectives_scaled,
                                    "preparation_diagnostics": getattr(model, "preparation_diagnostics", None)}
                                fits[name] = {"B": B, "A": A, "train_scores": train_scores, "test_scores": test_scores,
                                              "diagnostic": diagnostic, "times": [], "projection_times": []}
                            else:
                                np.testing.assert_allclose(B, fits[name]["B"], atol=1e-10, rtol=1e-10)
                            fits[name]["times"].append(seconds)
                            fits[name]["projection_times"].append(projection)
                            print(f"fit={seconds:.4f}s projection={projection:.4f}s peak_rss_bytes={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}", flush=True)
                    for name, implementation, _ in METHODS:
                        fit = fits[name]
                        A, B = fit["A"], fit["B"]
                        start = perf_counter()
                        raw = metrics(fit["train_scores"], fit["test_scores"], labels[train], labels[test], standardize=False)
                        std = metrics(fit["train_scores"], fit["test_scores"], labels[train], labels[test], standardize=True)
                        readout_seconds = perf_counter()-start
                        start = perf_counter()
                        if implementation == "pca":
                            residual2, eta = norm**2-float(np.sum(fit["train_scores"]**2)), None
                        else:
                            alpha = fit["diagnostic"]["alpha"]
                            _, objective, stationarity = _full_diagnostics(X, A, B, alpha, 1/norm**2, norm)
                            residual2 = 2*(objective-alpha*float(np.abs(B).sum()))
                            eta = float(stationarity/(alpha*np.sqrt(B.size)))
                        audit_seconds = perf_counter()-start+norm_seconds
                        record = {"seed": seed, "rank": rank, "method": name, "input_snps": G.shape[1],
                            "training_retained_snps": X.shape[1], "preprocessing_seconds": prep_seconds,
                            "fit_seconds": fit["times"], "fit_seconds_median": float(np.median(fit["times"])),
                            "projection_seconds_median": float(np.median(fit["projection_times"])),
                            "readout_seconds": readout_seconds, "audit_seconds": audit_seconds,
                            "full_eta": eta, "full_stationarity_qualified": bool(eta <= .001) if eta is not None else None,
                            "full_reconstruction_error": float(np.sqrt(max(0, residual2))/norm),
                            "nonzero_entry_fraction": float(np.mean(B != 0)),
                            "nonzero_snp_fraction": float(np.mean(np.any(B != 0, axis=1))),
                            "raw_centroid": raw, "std_centroid": std, "solver": fit["diagnostic"]}
                        record["workflow_seconds"] = prep_seconds+record["fit_seconds_median"]+record["projection_seconds_median"]+readout_seconds
                        record["audited_workflow_seconds"] = record["workflow_seconds"]+audit_seconds
                        fits[name]["record"] = record
                        records.append(record)
                        np.savez(directory/f"seed_{seed}_{name}_scores.npz", B=B, A=A,
                            train_scores=fit["train_scores"], test_scores=fit["test_scores"],
                            train_indices=train, test_indices=test, labels=labels, sample_ids=sample_ids)
                    # Save all actual outputs before evaluating the strict guard.
                    write_json(args.output/"results.json", {"records": records, "runtime_warnings": [], "threadpools": pools})
                    try:
                        for penalty in PENALTIES:
                            guards.append(equivalence(fits[f"RVP-reference-c{penalty:g}"], fits[f"RVP-optimized-c{penalty:g}"]))
                    except AssertionError as error:
                        write_json(args.output/"FAILED_EQUIVALENCE.json", {"seed": seed, "rank": rank, "error": str(error)})
                        raise
                    write_json(args.output/"equivalence.json", guards)
                    plot_pairs(fits, labels, train, test, rank, seed, directory)
                    print(f"Complete rank={rank} seed={seed}; all equivalence guards passed", flush=True)
                    del fits, model, A, B, train_scores, test_scores
                    gc.collect()
    cells = summarize(records)
    write_json(args.output/"summary.json", {"cells": cells, "records": records,
        "equivalence_passed": len(guards), "specification_sha256": digest(args.output/"specification_before_run.json")})
    write_json(args.output/"completion.json", {"status": "complete_default_pca_runtime_development",
        "cells": len(cells), "records": len(records), "timed_fits": 3*len(records),
        "equivalence_passed": len(guards),
        "files": {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*")) if p.is_file()}})
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

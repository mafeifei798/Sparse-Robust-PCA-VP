#!/usr/bin/env python3
"""Paired, full-width *development* comparison, without PCA SNP prescreening.

Run with PYTHONPATH=src and all BLAS thread environment variables set to 1.
Every requested configuration is reported. This is not a formal confirmation
runner and does not consume the reserved fresh-seed protocol.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from time import perf_counter
import warnings
import zipfile

import numpy as np
import scipy
from scipy import linalg
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from threadpoolctl import threadpool_info, threadpool_limits

from rspca.genomics_aim import fit_smartpca_preprocessor, stratified_split, transform_smartpca
from rspca.randomized import _full_diagnostics
from rspca.randomized_reduced import fit_reduced_randomized_vp_spca


POPULATIONS = ["AFR", "AMR", "EAS", "EUR", "SAS"]
COLORS = ["#E69F00", "#CC79A7", "#009E73", "#0072B2", "#D55E00"]


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def metrics(train_scores, test_scores, train_labels, test_labels, *, standardize):
    scale = train_scores.std(axis=0, ddof=1) if standardize else np.ones(train_scores.shape[1])
    scale = np.maximum(scale, np.finfo(float).eps)
    centroids = np.stack([train_scores[train_labels == label].mean(axis=0) for label in POPULATIONS]) / scale
    distances = ((test_scores[:, None, :] / scale - centroids[None, :, :]) ** 2).sum(axis=2)
    predicted = np.asarray(POPULATIONS)[np.argmin(distances, axis=1)]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, predicted)),
        "macro_f1": float(f1_score(test_labels, predicted, average="macro")),
        "population_recall": {label: float(np.mean(predicted[test_labels == label] == label)) for label in POPULATIONS},
        "confusion": confusion_matrix(test_labels, predicted, labels=POPULATIONS).tolist(),
        "predictions": predicted.tolist(),
    }


def draw_panels(fits, train_labels, test_labels, output, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for x, y in ((0, 1), (2, 3)):
        fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
        for ax, fit in zip(axes.flat, fits, strict=True):
            B = fit["B"]
            signs = np.sign(B[np.abs(B).argmax(axis=0), np.arange(B.shape[1])])
            signs[signs == 0] = 1
            train = fit["train_scores"] * signs
            test = fit["test_scores"] * signs
            for label, color in zip(POPULATIONS, COLORS, strict=True):
                mask = train_labels == label
                ax.scatter(train[mask, x], train[mask, y], s=14, c=color, alpha=.55, linewidths=0)
                mask = test_labels == label
                ax.scatter(test[mask, x], test[mask, y], s=24, c=color, edgecolors="black", linewidths=.35, label=label)
            record = fit["record"]
            ax.set_title(f'{record["method"]}\nBA {record["raw_centroid"]["balanced_accuracy"]:.3f}; fit {record["fit_seconds_median"]:.3f}s', fontsize=11)
            ax.set_xlabel(f"Native component {x+1}")
            ax.set_ylabel(f"Native component {y+1}")
            ax.spines[["top", "right"]].set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=5)
        fig.suptitle(f"Same 1,000 people / all train-retained SNPs / seed {seed}\nNative X @ B scores; signs only, no Procrustes alignment. Outlined points are held out.", fontsize=14)
        fig.savefig(output / f"seed_{seed}_pc{x+1}{y+1}.png", dpi=190)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2025])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--penalty-fractions", type=float, nargs=3, default=[.02, .05, .1])
    parser.add_argument("--fail-on-runtime-warning", action="store_true")
    parser.add_argument("--power-iterations", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    if not set(args.seeds).issubset(set(range(2025, 2030))):
        parser.error("this development runner allows exposed seeds 2025..2029 only")
    if args.rank < 4 or args.max_iter < 1 or args.repeats < 1:
        parser.error("rank >= 4 and positive iteration/repeat counts required")
    if len(set(args.penalty_fractions)) != 3 or any(not np.isfinite(c) or c <= 0 for c in args.penalty_fractions):
        parser.error("three distinct, finite positive penalty fractions required")
    # Exclusive creation prevents overwriting an earlier experiment or failure.
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    source_names = ["experiments/genomics_randomized_vp_matched.py", "src/rspca/randomized_reduced.py",
                    "src/rspca/vp.py", "src/rspca/randomized.py", "src/rspca/genomics_aim.py"]
    protocol = {
        "status": "exposed_seed_development_not_confirmation",
        "purpose": "randomized VP-SPCA versus PCA on the identical full SNP matrix",
        "input": str(args.input.resolve()), "input_sha256": digest(args.input),
        "seeds": args.seeds, "rank": args.rank, "max_iter": args.max_iter, "repeats": args.repeats,
        "penalty_fractions": args.penalty_fractions, "oversample": 10, "power_iterations": args.power_iterations,
        "sketch_relative_tolerance": .001, "check_every": 10,
        "alpha_rule": "penalty_fraction * sketch_sigma_max_squared / sqrt(n_train_retained_snps)",
        "readout": "same nearest training centroid; raw primary, train-only per-component std secondary",
        "feature_policy": "all train-polymorphic input SNPs, no loading ranking or prescreen",
        "fit_timing": "validation, own initialization/sketch and solver included; no PCA initialization shared",
        "workflow_timing": "shared training-only preprocessing + fit + native train/test projection",
        "audit_timing": "full-data reconstruction/stationarity reported separately, and also added to workflow",
        "selection": "full fixed grid reported; no winner or noninferiority claim from these exposed seeds",
        "runtime_warning_policy": "error" if args.fail_on_runtime_warning else "record_all_seeds",
        "source_sha256": {name: digest(repo / name) for name in source_names},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "machine": platform.machine()},
        "thread_environment": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    }
    if set(protocol["thread_environment"].values()) != {"1"}:
        raise RuntimeError("set all four thread environment variables to 1 before Python starts")
    write_json(args.output / "protocol_before_run.json", protocol)
    with zipfile.ZipFile(args.output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in source_names:
            archive.write(repo / name, name)
    with np.load(args.input, allow_pickle=False) as package:
        G = package["genotypes"]
        labels = package["population_labels"].astype(str)
        sample_ids = package["sample_ids"].astype(str)
    results = []
    runtime_warnings = []
    for seed in args.seeds:
        prep_start = perf_counter()
        train, test = stratified_split(labels, .2, seed)
        preprocessing = fit_smartpca_preprocessor(G[train])
        X = np.ascontiguousarray(transform_smartpca(G[train], preprocessing))
        Xtest = np.ascontiguousarray(transform_smartpca(G[test], preprocessing))
        preparation_seconds = perf_counter() - prep_start
        specifications = [("PCA-full", "full", None), ("PCA-auto", "auto", None),
                          (f"PCA-randomized-q{args.power_iterations}", "randomized", None)]
        specifications += [(f"RVP-SPCA-c{c:g}", None, c) for c in protocol["penalty_fractions"]]
        fits = {}
        # Rotate ordering over seeds/repeats; no warmed/shared sketch among SPCA settings.
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught_warnings:
            # Keep environment warnings as evidence; do not drop or flood the
            # terminal with them. Explicit finite and independent-dot checks
            # below remain mandatory before writing any scientific record.
            warnings.simplefilter("error" if args.fail_on_runtime_warning else "default", RuntimeWarning)
            pool_identity = threadpool_info()
            for repeat in range(args.repeats):
                order = np.random.default_rng(seed + 10000 * repeat).permutation(len(specifications))
                for index in order:
                    name, solver, c = specifications[index]
                    print(f"seed={seed} repeat={repeat} method={name} shape={X.shape}", flush=True)
                    started = perf_counter()
                    if solver:
                        model = PCA(n_components=args.rank, svd_solver=solver, random_state=seed,
                                    **({"iterated_power": args.power_iterations, "n_oversamples": 10} if solver == "randomized" else {}))
                        model.fit(X)
                        B, A = model.components_.T, model.components_.T
                        diagnostic = {"resolved_solver": model._fit_svd_solver,
                                      "largest_singular_value": float(model.singular_values_[0])}
                    else:
                        model = fit_reduced_randomized_vp_spca(X, args.rank, penalty_fraction=c,
                                                             max_iter=args.max_iter, random_state=seed,
                                                             n_power_iter=args.power_iterations)
                        B, A = model.B, model.A
                        diagnostic = {"alpha": model.alpha, "sketch_converged": model.sketch_converged,
                                      "sketch_eta": model.sketch_relative_stationarity, "iterations": model.n_iter,
                                      "restarts": model.restarts, "solver_timings": model.timings,
                                      "sketch_objectives_scaled": model.sketch_objectives_scaled}
                    fit_seconds = perf_counter() - started
                    projection_start = perf_counter()
                    train_scores, test_scores = X @ B, Xtest @ B
                    projection_seconds = perf_counter() - projection_start
                    if not all(np.isfinite(array).all() for array in (B, A, train_scores, test_scores)):
                        raise RuntimeError("nonfinite fit or projection")
                    np.testing.assert_allclose(train_scores[:3], np.einsum("ij,jk->ik", X[:3], B, optimize=False), atol=1e-9, rtol=1e-10)
                    np.testing.assert_allclose(test_scores[:3], np.einsum("ij,jk->ik", Xtest[:3], B, optimize=False), atol=1e-9, rtol=1e-10)
                    if name not in fits:
                        fits[name] = {"B": B, "A": A, "train_scores": train_scores, "test_scores": test_scores,
                                      "diagnostic": diagnostic, "fit_times": [], "projection_times": []}
                    else:
                        np.testing.assert_allclose(B, fits[name]["B"], atol=1e-12, rtol=1e-12)
                    fits[name]["fit_times"].append(fit_seconds)
                    fits[name]["projection_times"].append(projection_seconds)
                    print(f"completed fit={fit_seconds:.3f}s" + (f" sketch_eta={diagnostic['sketch_eta']:.4g}" if not solver else ""), flush=True)
            Lfull = fits["PCA-full"]["diagnostic"]["largest_singular_value"] ** 2
            norm = float(linalg.norm(X))
            for name, solver, c in specifications:
                fit = fits[name]
                B, A = fit["B"], fit["A"]
                audit_start = perf_counter()
                if solver:
                    residual_squared = norm**2 - float(np.sum(fit["train_scores"]**2))
                    full_eta = None
                else:
                    alpha = fit["diagnostic"]["alpha"]
                    A_full, objective, stationarity = _full_diagnostics(X, A, B, alpha, 1.0/Lfull, norm)
                    residual_squared = 2 * (objective - alpha * float(np.abs(B).sum()))
                    full_eta = stationarity / (alpha * np.sqrt(B.size))
                audit_seconds = perf_counter() - audit_start
                record = {"seed": seed, "method": name, "rank": args.rank, "input_snps": G.shape[1],
                          "training_retained_snps": X.shape[1], "preprocessing_seconds": preparation_seconds,
                          "fit_seconds": fit["fit_times"], "fit_seconds_median": float(np.median(fit["fit_times"])),
                          "projection_seconds_median": float(np.median(fit["projection_times"])),
                          "audit_seconds": audit_seconds, "full_eta": full_eta,
                          "full_stationarity_qualified": bool(full_eta <= .001) if full_eta is not None else None,
                          "full_reconstruction_error": float(np.sqrt(max(0, residual_squared)) / norm),
                          "nonzero_entry_fraction": float(np.mean(B != 0)),
                          "nonzero_snp_fraction": float(np.mean(np.any(B != 0, axis=1))),
                          "raw_centroid": metrics(fit["train_scores"], fit["test_scores"], labels[train], labels[test], standardize=False),
                          "std_centroid": metrics(fit["train_scores"], fit["test_scores"], labels[train], labels[test], standardize=True),
                          "solver": fit["diagnostic"]}
                record["workflow_seconds"] = preparation_seconds + record["fit_seconds_median"] + record["projection_seconds_median"]
                record["audited_workflow_seconds"] = record["workflow_seconds"] + audit_seconds
                fit["record"] = record
                results.append(record)
                np.savez_compressed(args.output / f"seed_{seed}_{name}_scores.npz", B=B, A=A,
                                    train_scores=fit["train_scores"], test_scores=fit["test_scores"],
                                    train_indices=train, test_indices=test, sample_ids=sample_ids,
                                    labels=labels, retained_snps=preprocessing.retained_snps)
        runtime_warnings.extend({"seed": seed, "message": str(w.message), "file": w.filename, "line": w.lineno} for w in caught_warnings)
        write_json(args.output / "results.json", {"protocol_sha256": digest(args.output / "protocol_before_run.json"),
                                                  "records": results, "threadpools": pool_identity,
                                                  "runtime_warnings": runtime_warnings})
        draw_panels([fits[name] for name, _, _ in specifications], labels[train], labels[test], args.output, seed)
    write_json(args.output / "completion.json", {"status": "development_complete", "records": len(results),
                                                 "files": {path.name: digest(path) for path in sorted(args.output.iterdir()) if path.is_file()}})
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

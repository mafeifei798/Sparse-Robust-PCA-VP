"""Run and summarize the complete exposed-seed AMR rank/readout grid.

The solver and fit settings from the earlier development turn are unchanged.
The fit runner additionally supports fail-on-warning and all-seed warning receipts.
This orchestrator adds a pre-fit specification, complete aggregates, and
native higher-component figures. It never chooses a winner.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import zipfile

import numpy as np

from genomics_randomized_vp_matched import COLORS, POPULATIONS, digest, write_json


METHODS = ["PCA-full", "PCA-auto", "PCA-randomized-q2", "RVP-SPCA-c0.0002",
           "RVP-SPCA-c0.001", "RVP-SPCA-c0.005"]


def summarize(records):
    summary = []
    methods = list(dict.fromkeys(r["method"] for r in records))
    for rank in sorted({r["rank"] for r in records}):
        for method in methods:
            rows = [r for r in records if r["rank"] == rank and r["method"] == method]
            if not rows:
                continue
            for view in ("raw_centroid", "std_centroid"):
                cm = np.sum([r[view]["confusion"] for r in rows], axis=0)
                by_population = {}
                for index, label in enumerate(POPULATIONS):
                    denominator = int(cm[:, index].sum())
                    by_population[label] = {
                        "recall": float(cm[index, index] / cm[index].sum()),
                        "precision": float(cm[index, index] / denominator) if denominator else 0.0,
                        "repeated_test_observations": int(cm[index].sum()),
                    }
                record = {
                    "rank": rank, "method": method, "readout": view,
                    "seeds": [r["seed"] for r in rows], "confusion": cm.tolist(),
                    "population": by_population,
                    "balanced_accuracy_mean": statistics.mean(r[view]["balanced_accuracy"] for r in rows),
                    "macro_f1_mean": statistics.mean(r[view]["macro_f1"] for r in rows),
                    "fit_seconds_median": statistics.median(r["fit_seconds_median"] for r in rows),
                    "workflow_seconds_median": statistics.median(r["workflow_seconds"] for r in rows),
                    "audited_workflow_seconds_median": statistics.median(r["audited_workflow_seconds"] for r in rows),
                    "nonzero_entry_fraction_mean": statistics.mean(r["nonzero_entry_fraction"] for r in rows),
                    "nonzero_snp_fraction_mean": statistics.mean(r["nonzero_snp_fraction"] for r in rows),
                    "sketch_converged_count": sum(r["solver"].get("sketch_converged", False) for r in rows),
                    "full_converged_count": sum(r["full_stationarity_qualified"] is True for r in rows),
                }
                if method.startswith("RVP"):
                    record["sketch_eta_range"] = [min(r["solver"]["sketch_eta"] for r in rows), max(r["solver"]["sketch_eta"] for r in rows)]
                    record["full_eta_range"] = [min(r["full_eta"] for r in rows), max(r["full_eta"] for r in rows)]
                summary.append(record)
    return summary


def plot_higher_components(package, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rank = records[0]["rank"]
    methods = list(dict.fromkeys(r["method"] for r in records))
    for seed in sorted({r["seed"] for r in records}):
        for first in range(4, rank, 2):
            fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
            for ax, method in zip(axes.flat, methods, strict=True):
                r = next(r for r in records if r["seed"] == seed and r["method"] == method)
                with np.load(package / f"seed_{seed}_{method}_scores.npz", allow_pickle=False) as d:
                    B = d["B"]
                    signs = np.sign(B[np.argmax(abs(B), axis=0), np.arange(rank)])
                    signs[signs == 0] = 1
                    for label, color in zip(POPULATIONS, COLORS, strict=True):
                        for split in ("train", "test"):
                            mask = d["labels"][d[f"{split}_indices"]] == label
                            points = d[f"{split}_scores"][mask] * signs
                            ax.scatter(points[:, first], points[:, first + 1], c=color,
                                       s=14 if split == "train" else 24,
                                       alpha=.55 if split == "train" else 1,
                                       linewidths=0 if split == "train" else .35,
                                       edgecolors="none" if split == "train" else "black",
                                       label=label if split == "test" else None)
                ax.set_title(f'{method}\nRaw / standardized AMR recall: {r["raw_centroid"]["population_recall"]["AMR"]:.3f} / {r["std_centroid"]["population_recall"]["AMR"]:.3f}', fontsize=10)
                ax.set_xlabel(f"Native component {first + 1}")
                ax.set_ylabel(f"Native component {first + 2}")
                ax.spines[["top", "right"]].set_visible(False)
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=5)
            fig.suptitle(f"All methods / seed {seed} / rank {rank}\nNative X @ B coordinates, signs only; outlined points are held out.")
            # Outside the sealed child package: never change its completion manifest.
            target = package.parent / "higher_component_figures"
            target.mkdir(exist_ok=True)
            fig.savefig(target / f"rank_{rank}_seed_{seed}_pc{first+1}{first+2}.png", dpi=190)
            plt.close(fig)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = ["#333333", "#0072B2", "#56B4E9", "#009E73", "#E69F00", "#CC79A7"]
    methods = list(dict.fromkeys(r["method"] for r in summary))
    for column, view in enumerate(("raw_centroid", "std_centroid")):
        for method, color in zip(methods, colors, strict=True):
            rows = [r for r in summary if r["method"] == method and r["readout"] == view]
            ranks = [r["rank"] for r in rows]
            for row, name in enumerate(("balanced_accuracy_mean", "AMR")):
                values = [100 * (r[name] if name != "AMR" else r["population"]["AMR"]["recall"]) for r in rows]
                axes[row, column].plot(ranks, values, marker="o", color=color,
                                      linestyle="--" if method.startswith("PCA") else "-", label=method, alpha=.85)
        axes[0, column].set_title("Raw centroid" if view == "raw_centroid" else "Train-standardized centroid")
        for ax in axes[:, column]:
            ax.set_xticks([4, 6, 8])
            ax.set_xlabel("Components")
            ax.grid(alpha=.15)
    axes[0, 0].set_ylabel("Balanced accuracy (%)")
    axes[1, 0].set_ylabel("AMR recall (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3)
    fig.suptitle(f"All {len(summary)} development cells — same SNP input, splits and readouts\n{len(summary[0]['seeds'])} exposed split(s); fixed-budget VP-SPCA, no selected winner")
    fig.savefig(output / "amr_rank_readout_full_grid.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--power-iterations", type=int, choices=(2, 4), default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = {"status": "exposed_seed_development_not_confirmation", "ranks": [4, 6, 8],
            "seeds": list(range(2025, 2030)), "repeats": 3, "iterations": 50,
            "penalty_fractions": [.0002, .001, .005],
            "methods": [m.replace("q2", f"q{args.power_iterations}") for m in METHODS],
            "power_iterations": args.power_iterations,
            "readouts": ["raw_centroid", "std_centroid"],
            "input_sha256": digest(args.input), "protocol_sha256": digest(args.protocol),
            "orchestrator_sha256": digest(__file__), "python": sys.executable}
    if spec["input_sha256"] != "b89d1557689b9ee98cefd2ae99f23b1f41b0cb0f61c3d0ab4cd3b8667429089e":
        raise ValueError("unexpected development input")
    write_json(args.output / "specification_before_run.json", spec)
    with zipfile.ZipFile(args.output / "orchestrator_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(__file__, Path(__file__).name)
        z.write(args.protocol, "PROTOCOL.md")
    runner = Path(__file__).with_name("genomics_randomized_vp_matched.py")
    all_records = []
    child_manifests = {}
    for rank in spec["ranks"]:
        directory = args.output / f"rank_{rank}"
        command = [sys.executable, "-W", "error::RuntimeWarning", str(runner),
                   "--input", str(args.input), "--output", str(directory),
                   "--rank", str(rank), "--seeds", *map(str, spec["seeds"]),
                   "--repeats", "3", "--max-iter", "50",
                   "--power-iterations", str(args.power_iterations),
                   "--penalty-fractions", ".0002", ".001", ".005", "--fail-on-runtime-warning"]
        print(f"Starting complete rank {rank} grid", flush=True)
        with (args.output / f"rank_{rank}.log").open("x") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"rank {rank} failed; log retained, no automatic retry")
        payload = json.loads((directory / "results.json").read_text())
        if payload["runtime_warnings"]:
            raise RuntimeError("runtime warnings remain; do not promote this environment")
        all_records.extend(payload["records"])
        child_manifests[f"rank_{rank}"] = digest(directory / "completion.json")
        plot_higher_components(directory, payload["records"])
        write_json(args.output / "progress.json", {"completed_ranks": list(child_manifests), "records": len(all_records)})
        print(f"Completed rank {rank}: 30 fit records, both readouts", flush=True)
    summary = summarize(all_records)
    write_json(args.output / "summary.json", {"specification_sha256": digest(args.output / "specification_before_run.json"),
                                             "cells": summary, "records": all_records,
                                             "child_completion_sha256": child_manifests})
    plot_summary(summary, args.output)
    write_json(args.output / "completion.json", {"status": "development_complete", "cells": len(summary),
        "files": {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*")) if p.is_file()}})
    print(f"Complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()

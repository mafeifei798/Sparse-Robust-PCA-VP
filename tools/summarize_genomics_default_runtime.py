"""Replay and report every cell of the default-PCA implementation benchmark.

This is an artifact-only report: no model fitting, parameter selection, source
package mutation, or new scientific confirmation claim.
"""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from genomics_randomized_vp_amr_grid import summarize
from genomics_randomized_vp_matched import digest, write_json


PACKAGES = (("dense_matched", 27346, 5), ("full_cached_matched", 455102, 1))
POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
METHODS = ["PCA-default"] + [
    f"RVP-{implementation}-c{c}"
    for implementation in ("reference", "optimized")
    for c in ("0.0002", "0.001", "0.005")
]


def strict_json(path):
    def reject(value):
        raise ValueError(f"Nonstandard JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def verify_package(package, width, nseeds):
    completion = strict_json(package / "completion.json")
    actual = {str(p.relative_to(package)) for p in package.rglob("*")
              if p.is_file() and p.name != "completion.json"}
    if actual != set(completion["files"]):
        raise ValueError(f"File-set mismatch: {package}")
    for relative, expected in completion["files"].items():
        if digest(package / relative) != expected:
            raise ValueError(f"Hash mismatch: {package / relative}")
        if relative.endswith(".json"):
            strict_json(package / relative)
    if (completion["cells"], completion["records"], completion["timed_fits"],
        completion["equivalence_passed"]) != (42, 21*nseeds, 63*nseeds, 9*nseeds):
        raise ValueError("Incomplete fixed grid")
    spec = strict_json(package / "specification_before_run.json")
    if spec["power_iterations"] != 4 or spec["max_iter"] != 50 or spec["oversample"] != 10:
        raise ValueError("Solver settings changed")
    if spec["seeds"] != list(range(2025, 2025+nseeds)):
        raise ValueError("Seed bank changed")
    with zipfile.ZipFile(package / "source_snapshot.zip") as archive:
        for relative, expected in spec["source_sha256"].items():
            if hashlib.sha256(archive.read(relative)).hexdigest() != expected:
                raise ValueError(f"Archived source mismatch: {relative}")
        if hashlib.sha256(archive.read("PROTOCOL.md")).hexdigest() != spec["protocol_sha256"]:
            raise ValueError("Archived protocol mismatch")
    summary = strict_json(package / "summary.json")
    records = summary["records"]
    expected = {(seed, rank, method) for seed in spec["seeds"]
                for rank in (4, 6, 8) for method in METHODS}
    if len(records) != len(expected) or {(r["seed"], r["rank"], r["method"]) for r in records} != expected:
        raise ValueError("Missing or duplicate record")
    if summarize(records) != summary["cells"]:
        raise ValueError("Aggregate replay mismatch")
    if strict_json(package / "results.json")["records"] != records:
        raise ValueError("Raw records differ from summary")
    for r in records:
        if r["input_snps"] != width or r["training_retained_snps"] != width:
            raise ValueError("SNP input changed")
        if len(r["fit_seconds"]) != 3 or statistics.median(r["fit_seconds"]) != r["fit_seconds_median"]:
            raise ValueError("Timing repeats/median mismatch")
        expected_workflow = (r["preprocessing_seconds"] + r["fit_seconds_median"]
                             + r["projection_seconds_median"] + r["readout_seconds"])
        if abs(expected_workflow-r["workflow_seconds"]) > 1e-12:
            raise ValueError("Workflow charge mismatch")
        if r["method"] == "PCA-default":
            expected_params = dict(copy=True, iterated_power="auto", n_components=r["rank"],
                n_oversamples=10, power_iteration_normalizer="auto", random_state=r["seed"],
                svd_solver="auto", tol=0.0, whiten=False)
            if r["solver"]["parameters"] != expected_params or r["solver"]["resolved_solver"] != "randomized":
                raise ValueError("Main comparator is not literal default PCA")
    return summary


def comparisons(records):
    index = {(r["seed"], r["rank"], r["method"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    result = []
    for rank in (4, 6, 8):
        for method in METHODS[1:]:
            baselines = ["PCA-default"]
            if "optimized" in method:
                baselines.append(method.replace("optimized", "reference"))
            for baseline in baselines:
                row = dict(rank=rank, method=method, baseline=baseline, seeds=seeds)
                for scope in ("fit_seconds_median", "workflow_seconds", "audited_workflow_seconds"):
                    gains = [100*(1-index[s, rank, method][scope]/index[s, rank, baseline][scope])
                             for s in seeds]
                    ratios = [index[s, rank, baseline][scope]/index[s, rank, method][scope] for s in seeds]
                    row[scope] = dict(paired_reduction_percent=gains,
                        median_reduction_percent=statistics.median(gains),
                        range_reduction_percent=[min(gains), max(gains)],
                        median_speedup=statistics.median(ratios))
                for view in ("raw_centroid", "std_centroid"):
                    row[view] = {}
                    for metric in ("balanced_accuracy", "macro_f1"):
                        delta = [index[s, rank, method][view][metric]-index[s, rank, baseline][view][metric]
                                 for s in seeds]
                        row[view][metric] = dict(mean_delta=statistics.mean(delta),
                            paired_delta=delta, range_delta=[min(delta), max(delta)])
                    amr = [index[s, rank, method][view]["population_recall"]["AMR"]
                           - index[s, rank, baseline][view]["population_recall"]["AMR"] for s in seeds]
                    row[view]["AMR_recall"] = dict(mean_delta=statistics.mean(amr),
                        paired_delta=amr, range_delta=[min(amr), max(amr)])
                result.append(row)
    return result


def pct(value):
    return f"{100*value:.2f}%"


def build_report(base, output):
    # Read/verify both completed inputs before creating any report output.
    packages = {name: verify_package(base/name, width, nseeds)
                for name, width, nseeds in PACKAGES}
    output.mkdir(parents=True, exist_ok=False)
    analysis = {"status": "complete_artifact_only_descriptive_report_not_confirmation",
        "packages": {}, "source_completion_sha256": {
            name: digest(base/name/"completion.json") for name, _, _ in PACKAGES}}
    lines = ["# Randomized VP-SPCA: same-quality runtime optimization", "",
        "The primary comparator is literal `PCA(n_components=rank, random_state=seed)` "
        "with every other scikit-learn 1.7.0 default unchanged. Its resolved solver is "
        "`randomized` in every cell. No full-SVD or private-kernel time is used as the main baseline.", "",
        "Both unchanged and optimized VP-SPCA use q=4, oversampling 10, all input SNPs, "
        "the same L1 penalties and at most 50 iterations. This is implementation-only, "
        "exposed-seed development, not a newly confirmed biological or algorithmic claim.", "",
        "## Interpretation and timing scope", "",
        "- Every implementation-equivalence pair must preserve all held-out predictions "
        "in both readouts, every nonzero-loading mask, and native coordinates within the "
        "predeclared tolerances. This does not mean every prediction equals PCA.",
        "- Each fit is independently initialized and includes its own sketch and VP solver. "
        "The same per-split preprocessing/cache cost is charged to every standalone workflow.",
        "- Workflow here is preprocessing/cache + median fit + median native projection + "
        "both common readouts. The pre-fit prose identified readout separately; the frozen "
        "runner's specification and these totals explicitly include it. Audited workflow "
        "additionally includes the full-data numerical audit, including an independently "
        "charged Frobenius norm. No PCA initialization or spectral norm is free to VP.",
        "- NPZ authentication/extraction, plotting and export are excluded. This is not "
        "download-to-answer end-to-end timing. Full-input fits share a read-only standardized "
        "float64 mmap cache; its preparation, integrity hashing and flush are included in workflow.",
        "- Runtime is the median of three serial fits per split. Cross-split times are medians "
        "of those medians. Gains are computed per paired split and then summarized; ranges "
        "are observed ranges, not confidence intervals. Five exposed overlapping splits are "
        "not five independent cohorts; three timing repeats are not biological replicates.",
        "- The full-input comparison uses one exposed split on a memory-constrained local Mac. "
        "Caching/page residency is part of this execution regime. Do not equate it to the "
        "previous blocked-SVD-kernel benchmark or assume timings generalize to AutoDL.",
        "- No setting is selected as a winner. Raw and train-standardized nearest-centroid "
        "readouts are both reported; population labels are used for split/readout/evaluation, "
        "not the unsupervised fitted projection. Cohort-level upstream QC/LD remains transductive.", ""]
    for name, width, nseeds in PACKAGES:
        summary = packages[name]
        rows = summary["records"]
        paired = comparisons(rows)
        guards = strict_json(base/name/"equivalence.json")
        analysis["packages"][name] = {"cells": summary["cells"], "paired_comparisons": paired,
            "equivalence_count": len(guards),
            "max_test_score_relative_frobenius_difference": max(g["test_scores_relative_frobenius_difference"] for g in guards)}
        lines += [f"## {width:,} SNPs — {nseeds} exposed split(s)", "",
            f"{len(rows)} configurations, {3*len(rows)} timed fits, {len(summary['cells'])} "
            f"method/rank/readout cells, {len(guards)} implementation-equivalence checks passed.", "",
            "### Complete timing table", "",
            "| Rank | Method | Fit s | Workflow s | Audited workflow s | Nonzero SNPs | Nonzero coefficients |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for c in summary["cells"]:
            if c["readout"] == "raw_centroid":
                lines.append(f"| {c['rank']} | {c['method']} | {c['fit_seconds_median']:.6f} | "
                    f"{c['workflow_seconds_median']:.6f} | {c['audited_workflow_seconds_median']:.6f} | "
                    f"{pct(c['nonzero_snp_fraction_mean'])} | {pct(c['nonzero_entry_fraction_mean'])} |")
        lines += ["", "### Complete paired speed comparison", "",
            "Positive reduction means faster; negative means slower. Brackets contain paired-split ranges.", "",
            "| Rank | Method | Compared with | Fit reduction | Workflow reduction | Audited reduction |",
            "| --- | --- | --- | ---: | ---: | ---: |"]
        for row in paired:
            cells = []
            for scope in ("fit_seconds_median", "workflow_seconds", "audited_workflow_seconds"):
                val = row[scope]
                lo, hi = val["range_reduction_percent"]
                cells.append(f"{val['median_reduction_percent']:.2f}% [{lo:.2f}, {hi:.2f}]")
            lines.append(f"| {row['rank']} | {row['method']} | {row['baseline']} | " + " | ".join(cells) + " |")
        for readout in ("raw_centroid", "std_centroid"):
            lines += ["", f"### Complete quality table: {readout}", "",
                "Recalls are pooled repeated held-out observations (40 per population per split), "
                "not independent new people. BA and macro-F1 are split means.", "",
                "| Rank | Method | BA | Macro-F1 | AFR recall | AMR recall | EAS recall | EUR recall | SAS recall |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for c in summary["cells"]:
                if c["readout"] == readout:
                    vals = [pct(c["balanced_accuracy_mean"]), pct(c["macro_f1_mean"])]
                    vals += [pct(c["population"][pop]["recall"]) for pop in POPULATIONS]
                    lines.append(f"| {c['rank']} | {c['method']} | " + " | ".join(vals) + " |")
        lines += ["", "### Fixed-budget solver status", "",
            "Qualification requires relative stationarity ≤ 0.001. A useful stratification "
            "at 50 steps is not a claim of converged sparse optimization.", "",
            "| Rank | Method | Sketch qualified | Full qualified | Sketch eta range | Full eta range |",
            "| --- | --- | ---: | ---: | --- | --- |"]
        for c in summary["cells"]:
            if c["readout"] == "raw_centroid" and c["method"].startswith("RVP"):
                a, b = c["sketch_eta_range"], c["full_eta_range"]
                lines.append(f"| {c['rank']} | {c['method']} | {c['sketch_converged_count']}/{nseeds} | "
                    f"{c['full_converged_count']}/{nseeds} | {a[0]:.6g}–{a[1]:.6g} | {b[0]:.6g}–{b[1]:.6g} |")
        lines += ["", "### Every native stratification figure", "",
            "All seeds, all ranks, all coordinate pairs, all seven methods. Same axis limits; "
            "only coordinate signs are normalized. No Procrustes rotation or outcome-selected subset.", ""]
        for seed in range(2025, 2025+nseeds):
            for rank in (4, 6, 8):
                links = [f"[PC{a}{a+1}](../{name}/rank_{rank}/seed_{seed}_pc{a}{a+1}.png)" for a in range(1, rank, 2)]
                lines.append(f"- Seed {seed}, rank {rank}: " + " · ".join(links))
        lines += ["", f"Source completion SHA-256: `{analysis['source_completion_sha256'][name]}`.", ""]
    lines += ["## What changed technically", "",
        "The optimized implementation reuses proximal work buffers, uses the equivalent "
        "clipping form of L1 shrinkage, replaces unconditional duplicate QR with a checked "
        "orthogonality/rank-repair pass, and factors the small sketch row Gram only when "
        "conditioning and right-basis orthogonality checks pass (otherwise thin SVD). "
        "It preserves the same randomized VP-SPCA objective, q, penalty grid, and iteration "
        "budget. These are verified implementation optimizations, not a new VP-SPCA theory.", "",
        "The genome-wide cache eliminates repeated genotype standardization inside matrix "
        "products, but is shared and charged equally to PCA and both VP variants. Current "
        "cached-to-cached comparisons support attribution; old blocked-to-cached runtimes "
        "do not isolate the buffer/QR/factorization change.", "",
        "## Evidence files", "",
        "`analysis.json` contains all 84 quality cells and all paired runtime/quality deltas. "
        "Original packages retain every timing repeat, held-out prediction, A/B matrix, native "
        "score array, source snapshot, input identity, thread-pool receipt, numerical diagnostic "
        "and file hash. This report does not overwrite any sealed source package.", ""]
    write_json(output/"analysis.json", analysis)
    (output/"README.md").write_text("\n".join(lines))
    write_json(output/"completion.json", {"status": "complete_artifact_only_report",
        "source_completion_sha256": analysis["source_completion_sha256"],
        "report_builder_sha256": digest(Path(__file__)),
        "files": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}})
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_report(args.base, args.output)
    print(json.dumps({"status": result["status"], "packages": list(result["packages"])}))

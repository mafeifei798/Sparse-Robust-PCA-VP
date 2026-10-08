"""Render readable, audit-preserving views of two frozen genomics AIM runs.

This is a presentation-only layer.  It never calls a solver and never writes
inside either formal nine-file package.  Both source packages are hash-checked
before and after rendering, and the destination is claimed with exclusive
creation.  The output manifest is written last.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import stat
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


FORMAL_MANIFEST = "aim_protocol_output_manifest.json"
FORMAL_COORDINATES = "aim_protocol_registered_coordinates.csv"
FORMAL_POPULATION_METRICS = "aim_protocol_population_metrics.csv"
FORMAL_METHOD_SUMMARIES = "aim_protocol_method_summaries.csv"

SCATTER_FIGURE = "five_superpop_registered_common_probe.png"
RECALL_FIGURE = "five_superpop_common_probe_recall.png"
SUMMARY_FIGURE = "three_vs_five_primary_summary.png"
ROBUST_FIGURE = "supplementary_robust_context.png"
RECEIPT = "presentation_receipt.json"
README = "README.md"
OUTPUT_MANIFEST = "presentation_output_manifest.json"

PRIMARY_METHODS = ("pca", "vp", "randomized")
ALL_METHODS = PRIMARY_METHODS + ("robust",)
METHOD_LABELS = {
    "pca": "PCA",
    "vp": "VP-SPCA",
    "randomized": "Randomized VP-SPCA",
    "robust": "Robust VP-SPCA",
}
METHOD_COLORS = {
    "pca": "#264653",
    "vp": "#E76F51",
    "randomized": "#2A9D8F",
    "robust": "#7B2CBF",
}
POPULATION_COLORS = {
    "AFR": "#0072B2",
    "AMR": "#E69F00",
    "EAS": "#009E73",
    "EUR": "#CC79A7",
    "SAS": "#D55E00",
}

FORMAL_FILENAMES = {
    "method_summaries_csv": "aim_protocol_method_summaries.csv",
    "panel_stability_png": "aim_protocol_panel_stability_and_delta.png",
    "population_metrics_csv": "aim_protocol_population_metrics.csv",
    "registered_coordinates_csv": "aim_protocol_registered_coordinates.csv",
    "registered_stratification_png": "aim_protocol_registered_stratification.png",
    "results_json": "aim_protocol_results.json",
    "selection_frequencies_csv": "aim_protocol_selection_frequencies.csv",
    "split_metrics_csv": "aim_protocol_split_metrics.csv",
}

_COORDINATE_FIELDS = {
    "registered_seed_policy",
    "seed",
    "method",
    "evaluation",
    "sample_index",
    "sample_id",
    "truth",
    "prediction",
    "component_1",
    "component_2",
    "qualification_status",
    "overall_qualified",
}
_POPULATION_FIELDS = {
    "seed",
    "method",
    "evaluation",
    "population",
    "support",
    "predicted_count",
    "true_positive",
    "precision",
    "recall",
    "f1",
}
_SUMMARY_FIELDS = {
    "method",
    "splits",
    "panel_size",
    "mean_full_width_native_balanced_accuracy",
    "mean_common_pca_probe_balanced_accuracy",
    "mean_panel_jaccard_with_equal_budget_pca",
    "mean_pairwise_panel_jaccard_across_splits",
    "qualified_splits",
    "qualification_status",
}


@dataclass(frozen=True)
class CohortData:
    name: str
    path: Path
    result_sha256: str
    registered_seed: int
    populations: tuple[str, ...]
    coordinates: tuple[dict[str, Any], ...]
    population_recall: Mapping[tuple[str, str], tuple[float, ...]]
    summaries: Mapping[str, Mapping[str, Any]]
    source_snapshot: Mapping[str, Mapping[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"non-standard JSON constant in {path.name}: {value}")

    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream, parse_constant=reject)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _real_regular_file(path: Path) -> os.stat_result:
    metadata = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"source entry must be a regular file: {path.name}")
    return metadata


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_formal_package(path: str | Path) -> dict[str, Any]:
    """Check the frozen nine-file contract and every declared payload hash."""

    root = Path(path)
    metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("formal package root must be a real directory")
    expected_names = set(FORMAL_FILENAMES.values()) | {FORMAL_MANIFEST}
    actual_names = {entry.name for entry in root.iterdir()}
    if actual_names != expected_names:
        raise ValueError("formal package does not match its nine-file contract")
    stats = {name: _real_regular_file(root / name) for name in expected_names}

    manifest = _strict_json(root / FORMAL_MANIFEST)
    required_keys = {
        "schema_version",
        "package_contract",
        "completion_status",
        "result_sha256",
        "registered_seed",
        "registered_seed_policy",
        "solver_lineage",
        "expected_filenames",
        "files",
    }
    if set(manifest) != required_keys:
        raise ValueError("formal manifest schema mismatch")
    if (
        manifest["schema_version"] != 2
        or manifest["package_contract"] != "fixed_file_set_v1"
        or manifest["completion_status"] != "complete"
        or manifest["registered_seed_policy"] != "first_config_seed_not_best"
        or manifest["expected_filenames"] != sorted(expected_names)
        or not _valid_sha256(manifest["result_sha256"])
    ):
        raise ValueError("formal manifest contract declaration is invalid")
    if isinstance(manifest["registered_seed"], bool) or not isinstance(
        manifest["registered_seed"], int
    ):
        raise ValueError("formal registered seed must be an integer")

    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(FORMAL_FILENAMES):
        raise ValueError("formal manifest payload roles mismatch")
    for role, filename in FORMAL_FILENAMES.items():
        record = files[role]
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise ValueError(f"invalid formal manifest record for {role}")
        if (
            record["path"] != filename
            or not _valid_sha256(record["sha256"])
            or isinstance(record["bytes"], bool)
            or not isinstance(record["bytes"], int)
            or record["bytes"] <= 0
        ):
            raise ValueError(f"invalid formal payload declaration for {role}")
        if stats[filename].st_size != record["bytes"]:
            raise ValueError(f"formal byte-count mismatch for {filename}")
        if _file_sha256(root / filename) != record["sha256"]:
            raise ValueError(f"formal SHA-256 mismatch for {filename}")
    manifest_mtime = stats[FORMAL_MANIFEST].st_mtime_ns
    if any(
        stats[filename].st_mtime_ns > manifest_mtime
        for filename in FORMAL_FILENAMES.values()
    ):
        raise ValueError("formal completion manifest is not the final write")
    return manifest


def _source_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    return {
        entry.name: {
            "bytes": entry.stat().st_size,
            "sha256": _file_sha256(entry),
        }
        for entry in sorted(path.iterdir(), key=lambda item: item.name)
    }


def _csv_rows(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        if not required_fields.issubset(fields):
            raise ValueError(f"CSV schema mismatch in {path.name}")
        rows = list(reader)
    if not rows or any(
        None in row or any(value is None for value in row.values()) for row in rows
    ):
        raise ValueError(f"empty or malformed CSV in {path.name}")
    return rows


def _finite_float(value: str, label: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite {label}")
    return parsed


def _nonnegative_int(value: str, label: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"negative {label}")
    return parsed


def load_cohort(path: str | Path, *, expected_populations: Sequence[str]) -> CohortData:
    root = Path(path)
    manifest = validate_formal_package(root)
    snapshot = _source_snapshot(root)
    registered_seed = int(manifest["registered_seed"])

    raw_coordinates = _csv_rows(root / FORMAL_COORDINATES, _COORDINATE_FIELDS)
    coordinates: list[dict[str, Any]] = []
    by_method_samples: dict[str, set[str]] = {}
    for row in raw_coordinates:
        if row["evaluation"] != "top_k_common_pca_probe":
            continue
        method = row["method"]
        if method not in ALL_METHODS:
            raise ValueError(f"unexpected coordinate method: {method}")
        seed = int(row["seed"])
        if (
            seed != registered_seed
            or row["registered_seed_policy"] != "first_config_seed_not_best"
        ):
            raise ValueError("registered coordinate seed policy mismatch")
        truth = row["truth"]
        prediction = row["prediction"]
        if truth not in expected_populations or prediction not in expected_populations:
            raise ValueError("unexpected population label in registered coordinates")
        parsed = {
            "seed": seed,
            "method": method,
            "sample_id": row["sample_id"],
            "truth": truth,
            "prediction": prediction,
            "component_1": _finite_float(row["component_1"], "component_1"),
            "component_2": _finite_float(row["component_2"], "component_2"),
        }
        coordinates.append(parsed)
        by_method_samples.setdefault(method, set()).add(row["sample_id"])
    if set(by_method_samples) != set(ALL_METHODS):
        raise ValueError("registered common-probe coordinates lack a method")
    if len({frozenset(values) for values in by_method_samples.values()}) != 1:
        raise ValueError("methods do not share the same registered held-out samples")
    if any(len(values) != len(by_method_samples["pca"]) for values in by_method_samples.values()):
        raise ValueError("registered method sample counts differ")

    summary_rows = _csv_rows(root / FORMAL_METHOD_SUMMARIES, _SUMMARY_FIELDS)
    summaries: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        method = row["method"]
        if method not in ALL_METHODS or method in summaries:
            raise ValueError("duplicate or unexpected method summary")
        summary = {
            "splits": _nonnegative_int(row["splits"], "split count"),
            "panel_size": _nonnegative_int(row["panel_size"], "panel size"),
            "full_width_ba": _finite_float(
                row["mean_full_width_native_balanced_accuracy"], "full-width BA"
            ),
            "fixed_200_common_ba": _finite_float(
                row["mean_common_pca_probe_balanced_accuracy"], "common-probe BA"
            ),
            "pca_overlap": _finite_float(
                row["mean_panel_jaccard_with_equal_budget_pca"], "PCA overlap"
            ),
            "panel_stability": _finite_float(
                row["mean_pairwise_panel_jaccard_across_splits"], "panel stability"
            ),
            "qualified_splits": _nonnegative_int(
                row["qualified_splits"], "qualified split count"
            ),
            "qualification_status": row["qualification_status"],
        }
        if summary["panel_size"] != 200 or summary["splits"] != 20:
            raise ValueError("presentation expects the frozen 20-split, fixed-200 protocol")
        for key in ("full_width_ba", "fixed_200_common_ba", "pca_overlap", "panel_stability"):
            if not 0.0 <= summary[key] <= 1.0:
                raise ValueError(f"summary metric outside [0,1]: {key}")
        if summary["qualified_splits"] > summary["splits"]:
            raise ValueError("qualified split count exceeds total splits")
        summaries[method] = summary
    if set(summaries) != set(ALL_METHODS):
        raise ValueError("formal method summaries lack a required method")

    population_rows = _csv_rows(
        root / FORMAL_POPULATION_METRICS, _POPULATION_FIELDS
    )
    recall_lists: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in population_rows:
        if row["evaluation"] != "common_pca_probe":
            continue
        method = row["method"]
        population = row["population"]
        if method not in ALL_METHODS or population not in expected_populations:
            raise ValueError("unexpected method or population in recall table")
        recall = _finite_float(row["recall"], "recall")
        if not 0.0 <= recall <= 1.0:
            raise ValueError("recall outside [0,1]")
        recall_lists[(method, population)].append((int(row["seed"]), recall))
    expected_pairs = {
        (method, population)
        for method in ALL_METHODS
        for population in expected_populations
    }
    if set(recall_lists) != expected_pairs:
        raise ValueError("common-probe recall table is incomplete")
    population_recall: dict[tuple[str, str], tuple[float, ...]] = {}
    for pair, values in recall_lists.items():
        ordered = sorted(values)
        seeds = [seed for seed, _ in ordered]
        if len(seeds) != 20 or len(set(seeds)) != 20:
            raise ValueError("recall table must contain 20 unique split seeds")
        population_recall[pair] = tuple(value for _, value in ordered)

    return CohortData(
        name=root.name,
        path=root,
        result_sha256=manifest["result_sha256"],
        registered_seed=registered_seed,
        populations=tuple(expected_populations),
        coordinates=tuple(coordinates),
        population_recall=population_recall,
        summaries=summaries,
        source_snapshot=snapshot,
    )


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(
        path,
        dpi=240,
        facecolor="white",
        bbox_inches="tight",
        metadata={"Software": "render_genomics_aim_presentation.py"},
    )
    plt.close(fig)


def _balanced_accuracy(rows: Sequence[Mapping[str, Any]], populations: Sequence[str]) -> float:
    recalls = []
    for population in populations:
        selected = [row for row in rows if row["truth"] == population]
        if not selected:
            raise ValueError("population absent from registered coordinates")
        recalls.append(sum(row["truth"] == row["prediction"] for row in selected) / len(selected))
    return float(np.mean(recalls))


def render_registered_scatter(five: CohortData, path: Path) -> dict[str, Any]:
    _configure_style()
    primary_rows = [row for row in five.coordinates if row["method"] in PRIMARY_METHODS]
    x_values = np.asarray([row["component_1"] for row in primary_rows], dtype=float)
    y_values = np.asarray([row["component_2"] for row in primary_rows], dtype=float)
    x_pad = max(1e-9, 0.06 * float(np.ptp(x_values)))
    y_pad = max(1e-9, 0.08 * float(np.ptp(y_values)))
    x_limits = (float(x_values.min() - x_pad), float(x_values.max() + x_pad))
    y_limits = (float(y_values.min() - y_pad), float(y_values.max() + y_pad))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.9), sharex=True, sharey=True)
    audit: dict[str, Any] = {}
    for axis, method in zip(axes, PRIMARY_METHODS):
        rows = [row for row in five.coordinates if row["method"] == method]
        for population in five.populations:
            population_rows = [row for row in rows if row["truth"] == population]
            correct = [row for row in population_rows if row["truth"] == row["prediction"]]
            incorrect = [row for row in population_rows if row["truth"] != row["prediction"]]
            axis.scatter(
                [row["component_1"] for row in correct],
                [row["component_2"] for row in correct],
                s=34,
                c=POPULATION_COLORS[population],
                alpha=0.78,
                edgecolors="white",
                linewidths=0.35,
                zorder=2,
            )
            axis.scatter(
                [row["component_1"] for row in incorrect],
                [row["component_2"] for row in incorrect],
                s=78,
                c=POPULATION_COLORS[population],
                marker="X",
                edgecolors="black",
                linewidths=1.0,
                zorder=4,
            )
        misclassified = sum(row["truth"] != row["prediction"] for row in rows)
        ba = _balanced_accuracy(rows, five.populations)
        axis.set_title(
            f"{METHOD_LABELS[method]}\nBA = {ba:.3f}  |  {misclassified}/{len(rows)} misclassified"
        )
        axis.set_xlim(*x_limits)
        axis.set_ylim(*y_limits)
        axis.set_xlabel("Common-probe component 1")
        axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.35)
        axis.axvline(0.0, color="#777777", linewidth=0.6, alpha=0.35)
        audit[method] = {
            "balanced_accuracy": ba,
            "held_out_samples": len(rows),
            "misclassified": misclassified,
        }
    axes[0].set_ylabel("Common-probe component 2")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=POPULATION_COLORS[population],
            markeredgecolor="white",
            markersize=8,
            label=population,
        )
        for population in five.populations
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="X",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=9,
            label="X = misclassified",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
    )
    fig.suptitle(
        f"Five-super-population held-out structure — registered seed {five.registered_seed}",
        fontsize=17,
        fontweight="bold",
        y=1.02,
    )
    fig.text(
        0.5,
        0.955,
        "Fixed 200-SNP panel, common PCA probe; first configured seed (not best-seed selection)",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.018,
        "VP and randomized views nearly coincide with PCA because their selected panels overlap PCA by >99.8%.",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.985, top=0.84, bottom=0.23, wspace=0.08)
    _save_figure(fig, path)
    return {
        "registered_seed": five.registered_seed,
        "registered_seed_policy": "first_config_seed_not_best",
        "common_axis_limits": {"component_1": list(x_limits), "component_2": list(y_limits)},
        "methods": audit,
    }


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def render_recall_bars(five: CohortData, path: Path) -> dict[str, Any]:
    _configure_style()
    fig, axis = plt.subplots(figsize=(14.5, 7.2))
    x = np.arange(len(five.populations), dtype=float)
    width = 0.23
    audit: dict[str, Any] = {}
    for population in ("AMR", "SAS"):
        index = five.populations.index(population)
        axis.axvspan(index - 0.48, index + 0.48, color="#F4E3C1", alpha=0.45, zorder=0)
    for method_index, method in enumerate(PRIMARY_METHODS):
        means: list[float] = []
        deviations: list[float] = []
        audit[method] = {}
        for population in five.populations:
            mean, deviation = _mean_sd(five.population_recall[(method, population)])
            means.append(mean)
            deviations.append(deviation)
            audit[method][population] = {"mean": mean, "split_sd": deviation}
        positions = x + (method_index - 1) * width
        bars = axis.bar(
            positions,
            means,
            width,
            yerr=deviations,
            capsize=3,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for bar, mean in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.075, mean + 0.025),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.3,
                rotation=90 if mean > 0.92 else 0,
            )
    axis.set_xticks(x, five.populations)
    axis.set_ylim(0.0, 1.12)
    axis.set_ylabel("Held-out recall")
    axis.set_title(
        "Five-super-population recall by reference group\n"
        "20-split mean ± split SD; fixed 200 SNPs, common PCA probe",
        fontweight="bold",
        pad=14,
    )
    axis.legend(loc="lower left", frameon=True, framealpha=0.95, ncol=3)
    axis.text(
        0.985,
        0.055,
        "Main bottlenecks: AMR and SAS\nPCA: 20/20 qualified\nVP / randomized: 0/20 qualified (descriptive)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color="#8B1E1E",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#8B1E1E", "alpha": 0.96},
    )
    fig.text(
        0.5,
        0.01,
        "Reference-super-population labels are metadata categories, not race; each split selects its own panel.",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.14)
    _save_figure(fig, path)
    return audit


def _grouped_bars(
    axis: plt.Axes,
    categories: Sequence[str],
    values: Mapping[str, Sequence[float]],
    *,
    ylim: tuple[float, float],
    ylabel: str,
    value_format: str = ".3f",
) -> None:
    x = np.arange(len(categories), dtype=float)
    width = 0.23
    for method_index, method in enumerate(PRIMARY_METHODS):
        positions = x + (method_index - 1) * width
        bars = axis.bar(
            positions,
            values[method],
            width,
            color=METHOD_COLORS[method],
            edgecolor="white",
            linewidth=0.7,
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for bar, value in zip(bars, values[method]):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.015 * (ylim[1] - ylim[0]),
                format(value, value_format),
                ha="center",
                va="bottom",
                fontsize=8.2,
                rotation=90 if len(categories) >= 4 else 0,
            )
    axis.set_xticks(x, categories)
    axis.set_ylim(*ylim)
    axis.set_ylabel(ylabel)


def render_primary_summary(three: CohortData, five: CohortData, path: Path) -> dict[str, Any]:
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    cohorts = (("3 groups", three), ("5 groups", five))

    ba_categories = ("3: full", "3: fixed 200", "5: full", "5: fixed 200")
    ba_values = {
        method: (
            three.summaries[method]["full_width_ba"],
            three.summaries[method]["fixed_200_common_ba"],
            five.summaries[method]["full_width_ba"],
            five.summaries[method]["fixed_200_common_ba"],
        )
        for method in PRIMARY_METHODS
    }
    _grouped_bars(
        axes[0, 0],
        ba_categories,
        ba_values,
        ylim=(0.72, 1.025),
        ylabel="Mean balanced accuracy",
    )
    axes[0, 0].set_title("A. Full width vs fixed-200 common-probe BA", fontweight="bold")

    stability_values = {
        method: tuple(cohort.summaries[method]["panel_stability"] for _, cohort in cohorts)
        for method in PRIMARY_METHODS
    }
    _grouped_bars(
        axes[0, 1],
        [label for label, _ in cohorts],
        stability_values,
        ylim=(0.68, 0.83),
        ylabel="Mean pairwise panel Jaccard",
    )
    axes[0, 1].set_title("B. Panel stability across 20 split-specific panels", fontweight="bold")

    overlap_axis = axes[1, 0]
    y_positions = np.arange(len(PRIMARY_METHODS), dtype=float)
    offsets = {"3 groups": -0.08, "5 groups": 0.08}
    marker_styles = {"3 groups": "o", "5 groups": "s"}
    overlap_audit: dict[str, dict[str, float]] = {}
    for label, cohort in cohorts:
        values = [cohort.summaries[method]["pca_overlap"] for method in PRIMARY_METHODS]
        overlap_axis.scatter(
            values,
            y_positions + offsets[label],
            s=85,
            marker=marker_styles[label],
            c=[METHOD_COLORS[method] for method in PRIMARY_METHODS],
            edgecolors="black",
            linewidths=0.55,
            label=label,
            zorder=3,
        )
        for y, value in zip(y_positions + offsets[label], values):
            overlap_axis.text(value - 0.00008, y + 0.04, f"{value:.6f}", ha="right", va="bottom", fontsize=8.5)
        overlap_audit[label] = dict(zip(PRIMARY_METHODS, values))
    overlap_axis.set_xlim(0.9979, 1.00015)
    overlap_axis.set_yticks(y_positions, [METHOD_LABELS[method] for method in PRIMARY_METHODS])
    overlap_axis.set_xlabel("Mean panel Jaccard with equal-budget PCA")
    overlap_axis.set_title("C. Panel overlap with PCA (expanded x-axis)", fontweight="bold")
    overlap_axis.legend(loc="lower left", frameon=False)

    qualification_axis = axes[1, 1]
    x = np.arange(len(PRIMARY_METHODS), dtype=float)
    width = 0.32
    for cohort_index, (label, cohort) in enumerate(cohorts):
        positions = x + (cohort_index - 0.5) * width
        values = [cohort.summaries[method]["qualified_splits"] for method in PRIMARY_METHODS]
        bars = qualification_axis.bar(
            positions,
            values,
            width,
            color="#4C78A8" if cohort_index == 0 else "#F58518",
            edgecolor="white",
            label=label,
            zorder=3,
        )
        qualification_axis.scatter(
            positions,
            values,
            marker="x" if cohort_index else "o",
            s=55,
            color="black",
            zorder=4,
        )
        for position, value in zip(positions, values):
            qualification_axis.text(position, value + 0.65, f"{value}/20", ha="center", fontsize=9, fontweight="bold")
    qualification_axis.set_xticks(x, [METHOD_LABELS[method] for method in PRIMARY_METHODS])
    qualification_axis.tick_params(axis="x", rotation=9)
    qualification_axis.set_ylim(0, 22.5)
    qualification_axis.set_ylabel("Qualified splits")
    qualification_axis.set_title("D. Solver qualification is not optional", fontweight="bold")
    qualification_axis.legend(loc="upper right", frameon=False)
    qualification_axis.text(
        0.5,
        0.20,
        "VP and randomized: 0/20 in both cohorts\nmetrics above are descriptive failure analysis",
        transform=qualification_axis.transAxes,
        ha="center",
        color="#8B1E1E",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#8B1E1E"},
    )

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=8, label=METHOD_LABELS[method])
        for method in PRIMARY_METHODS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("Three-group vs five-group fixed-budget AIM summary", fontsize=18, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.015,
        "Same 20 seeds; chr20–22 only; each split selects a new 200-SNP panel. This is not one fixed panel or genome-wide replication.",
        ha="center",
        color="#444444",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.89, bottom=0.075, hspace=0.32, wspace=0.25)
    _save_figure(fig, path)
    return {
        "balanced_accuracy": ba_values,
        "panel_stability": stability_values,
        "pca_panel_overlap": overlap_audit,
        "qualified_splits": {
            label: {method: cohort.summaries[method]["qualified_splits"] for method in PRIMARY_METHODS}
            for label, cohort in cohorts
        },
    }


def render_robust_supplement(three: CohortData, five: CohortData, path: Path) -> dict[str, Any]:
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.7))

    x = np.arange(len(five.populations), dtype=float)
    width = 0.34
    recall_audit: dict[str, dict[str, float]] = {}
    for method_index, method in enumerate(("pca", "robust")):
        means = [_mean_sd(five.population_recall[(method, population)])[0] for population in five.populations]
        recall_audit[method] = dict(zip(five.populations, means))
        axes[0, 0].bar(
            x + (method_index - 0.5) * width,
            means,
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )
    axes[0, 0].set_xticks(x, five.populations)
    axes[0, 0].set_ylim(0, 1.06)
    axes[0, 0].set_ylabel("Mean held-out recall")
    axes[0, 0].set_title("A. Five-group common-probe recall", fontweight="bold")
    axes[0, 0].legend(frameon=False)

    categories = ("3: full", "3: fixed 200", "5: full", "5: fixed 200")
    for method_index, method in enumerate(("pca", "robust")):
        values = (
            three.summaries[method]["full_width_ba"],
            three.summaries[method]["fixed_200_common_ba"],
            five.summaries[method]["full_width_ba"],
            five.summaries[method]["fixed_200_common_ba"],
        )
        axes[0, 1].bar(
            np.arange(4) + (method_index - 0.5) * width,
            values,
            width,
            color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
            zorder=3,
        )
        for position, value in zip(np.arange(4) + (method_index - 0.5) * width, values):
            axes[0, 1].text(position, value + 0.009, f"{value:.3f}", ha="center", fontsize=8.5, rotation=90)
    axes[0, 1].set_xticks(np.arange(4), categories)
    axes[0, 1].set_ylim(0.72, 1.03)
    axes[0, 1].set_ylabel("Mean balanced accuracy")
    axes[0, 1].set_title("B. Positive-looking five-group delta is descriptive", fontweight="bold")
    axes[0, 1].legend(frameon=False)

    labels = ("3: stability", "3: overlap", "5: stability", "5: overlap")
    robust_values = (
        three.summaries["robust"]["panel_stability"],
        three.summaries["robust"]["pca_overlap"],
        five.summaries["robust"]["panel_stability"],
        five.summaries["robust"]["pca_overlap"],
    )
    bars = axes[1, 0].bar(np.arange(4), robust_values, color=METHOD_COLORS["robust"], zorder=3)
    for bar, value in zip(bars, robust_values):
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", fontsize=9)
    axes[1, 0].set_xticks(np.arange(4), labels, rotation=10)
    axes[1, 0].set_ylim(0, 1.08)
    axes[1, 0].set_ylabel("Jaccard")
    axes[1, 0].set_title("C. Robust panels are less stable / less PCA-like", fontweight="bold")

    q_values = [
        three.summaries["pca"]["qualified_splits"],
        three.summaries["robust"]["qualified_splits"],
        five.summaries["pca"]["qualified_splits"],
        five.summaries["robust"]["qualified_splits"],
    ]
    q_labels = ("3: PCA", "3: Robust", "5: PCA", "5: Robust")
    q_colors = [METHOD_COLORS["pca"], METHOD_COLORS["robust"]] * 2
    axes[1, 1].bar(np.arange(4), q_values, color=q_colors, zorder=3)
    axes[1, 1].scatter(np.arange(4), q_values, marker="x", color="black", s=60, zorder=4)
    for position, value in enumerate(q_values):
        axes[1, 1].text(position, value + 0.7, f"{value}/20", ha="center", fontweight="bold")
    axes[1, 1].set_xticks(np.arange(4), q_labels, rotation=8)
    axes[1, 1].set_ylim(0, 22.5)
    axes[1, 1].set_ylabel("Qualified splits")
    axes[1, 1].set_title("D. Robust is unqualified in both cohorts", fontweight="bold")

    fig.suptitle(
        "Supplementary Robust VP-SPCA diagnostic — not the primary genomics model",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.932,
        "It was not used in the original genomics notebooks. All Robust metrics below are descriptive because qualification is 0/20.",
        ha="center",
        color="#8B1E1E",
        fontsize=10.5,
    )
    fig.subplots_adjust(left=0.08, right=0.985, top=0.87, bottom=0.08, hspace=0.32, wspace=0.24)
    _save_figure(fig, path)
    return {
        "five_group_common_probe_recall": recall_audit,
        "robust_panel_stability_and_overlap": dict(zip(labels, robust_values)),
        "qualified_splits": dict(zip(q_labels, q_values)),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _relative_source(path: Path, output: Path) -> str:
    return os.path.relpath(path, output)


def _write_readme(path: Path, three: CohortData, five: CohortData) -> None:
    text = f"""# Readable genomics AIM presentation figures

These figures are redrawn only from the two frozen formal CSV packages. No
solver was rerun, and the formal directories were hash-checked before and after
rendering. Seed `{five.registered_seed}` is shown because it is the first
configured seed, not because it looked best.

## Primary PCA / VP / randomized storyline

![Registered five-group common-probe scatter]({SCATTER_FIGURE})

Black `X` markers are held-out misclassifications. All panels share the same
axis limits. The near-identical three panels are not a plotting mistake: the VP
and randomized panels overlap the equal-budget PCA panel by more than 99.8%.

![Five-group per-population recall]({RECALL_FIGURE})

The bars are 20-split means and the error bars are split standard deviations.
AMR and SAS are the visible bottlenecks; PCA is `20/20` qualified, while VP and
randomized are both `0/20`, so their values are descriptive failure analysis.

![Three-vs-five primary summary]({SUMMARY_FIGURE})

This page shows full-width versus fixed-200 balanced accuracy, across-split
panel stability, panel overlap with PCA, and the qualification counts together.
The overlap panel explicitly uses an expanded x-axis and prints exact values.

## Supplementary robust diagnostic

![Supplementary robust context]({ROBUST_FIGURE})

Robust VP-SPCA is separated because it was not part of the original genomics
notebooks and is not needed by the primary PCA/VP/randomized question. Its
positive-looking five-group result remains visible, together with its `0/20`
qualification and weaker panel stability/overlap.

## Interpretation limits

- The inputs cover chromosomes 20--22, not the whole genome.
- Every outer split selects a different 200-SNP panel; this is not one fixed
  deployable AIM panel.
- The five-group labels are 1000 Genomes reference-super-population metadata,
  not race, disease, or causal biology.
- The three- and five-group tracks share raw inputs and are not independent
  replication.
- VP, randomized, and robust results are unqualified (`0/20`) because the
  500-iteration sparse fits did not converge; no sparse-method superiority is
  claimed.

## Audit trail

Source package identities, plotted values, source snapshots, and figure hashes
are recorded in [`{RECEIPT}`]({RECEIPT}). The output directory was created
exclusively and is closed by [`{OUTPUT_MANIFEST}`]({OUTPUT_MANIFEST}). The two
unchanged formal sources are:

- [`{three.name}`]({_relative_source(three.path, path.parent)}/)
- [`{five.name}`]({_relative_source(five.path, path.parent)}/)
"""
    path.write_text(text, encoding="utf-8")


def validate_presentation_output(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    metadata = root.lstat()
    if root.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("presentation root must be a real directory")
    manifest = _strict_json(root / OUTPUT_MANIFEST)
    required_keys = {
        "schema_version",
        "package_contract",
        "completion_status",
        "required_payloads",
        "files",
        "source_result_sha256",
        "manifest_sha256",
    }
    if set(manifest) != required_keys:
        raise ValueError("presentation manifest schema mismatch")
    body = dict(manifest)
    digest = body.pop("manifest_sha256")
    if digest != _canonical_sha256(body):
        raise ValueError("presentation manifest canonical digest mismatch")
    payloads = manifest["required_payloads"]
    files = manifest["files"]
    if (
        manifest["schema_version"] != 1
        or manifest["package_contract"] != "exclusive_fixed_file_set_v1"
        or manifest["completion_status"] != "complete"
        or not isinstance(payloads, list)
        or payloads != sorted(payloads)
        or not isinstance(files, dict)
        or set(files) != set(payloads)
    ):
        raise ValueError("presentation contract declaration is invalid")
    if {entry.name for entry in root.iterdir()} != set(payloads) | {OUTPUT_MANIFEST}:
        raise ValueError("presentation directory file set mismatch")
    for name in payloads:
        file_metadata = _real_regular_file(root / name)
        record = files[name]
        if not isinstance(record, dict) or set(record) != {"bytes", "sha256"}:
            raise ValueError(f"invalid presentation file record for {name}")
        if (
            file_metadata.st_size != record["bytes"]
            or _file_sha256(root / name) != record["sha256"]
        ):
            raise ValueError(f"presentation hash or size mismatch for {name}")
    return manifest


def render_presentation(
    three_formal: str | Path,
    five_formal: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Create the closed presentation package without mutating formal inputs."""

    three_path = Path(three_formal)
    five_path = Path(five_formal)
    output = Path(output_directory)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    parent = output.parent
    parent_metadata = parent.lstat()
    if parent.is_symlink() or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ValueError("output parent must be a real existing directory")
    source_roots = (three_path.resolve(), five_path.resolve())
    output_resolved = output.resolve(strict=False)
    if any(output_resolved.is_relative_to(source) for source in source_roots):
        raise ValueError("presentation output cannot be inside a formal package")

    three = load_cohort(three_path, expected_populations=("AFR", "EAS", "EUR"))
    five = load_cohort(
        five_path, expected_populations=("AFR", "AMR", "EAS", "EUR", "SAS")
    )
    output.mkdir(mode=0o755, exist_ok=False)
    try:
        plotted = {
            "registered_scatter": render_registered_scatter(five, output / SCATTER_FIGURE),
            "five_group_recall": render_recall_bars(five, output / RECALL_FIGURE),
            "three_vs_five_summary": render_primary_summary(three, five, output / SUMMARY_FIGURE),
            "robust_supplement": render_robust_supplement(three, five, output / ROBUST_FIGURE),
        }
        _write_readme(output / README, three, five)

        # Revalidate every source byte before sealing the presentation output.
        validate_formal_package(three.path)
        validate_formal_package(five.path)
        after_snapshots = {
            "three_group": _source_snapshot(three.path),
            "five_group": _source_snapshot(five.path),
        }
        if after_snapshots["three_group"] != three.source_snapshot or after_snapshots["five_group"] != five.source_snapshot:
            raise RuntimeError("a formal source changed during presentation rendering")

        receipt: dict[str, Any] = {
            "schema_version": 1,
            "presentation_kind": "genomics_aim_readable_views",
            "solver_called": False,
            "formal_sources_unchanged_during_render": True,
            "primary_methods": list(PRIMARY_METHODS),
            "supplementary_methods": ["robust"],
            "source_packages": {
                "three_group": {
                    "path": _relative_source(three.path, output),
                    "result_sha256": three.result_sha256,
                    "snapshot": three.source_snapshot,
                },
                "five_group": {
                    "path": _relative_source(five.path, output),
                    "result_sha256": five.result_sha256,
                    "snapshot": five.source_snapshot,
                },
            },
            "plot_data_recomputed_from_formal_csv": True,
            "plotted_values": plotted,
            "scope_guards": {
                "chromosome_scope": "chr20-22_not_genomewide",
                "panel_scope": "split_specific_fixed_200_not_one_panel",
                "labels": "1000_genomes_reference_superpopulation_not_race",
                "replication": "shared_raw_inputs_not_independent_replication",
                "sparse_solver_claim": "vp_randomized_robust_zero_of_twenty_qualified_descriptive_only",
            },
        }
        receipt["receipt_body_sha256"] = _canonical_sha256(receipt)
        _write_json(output / RECEIPT, receipt)

        payloads = sorted((SCATTER_FIGURE, RECALL_FIGURE, SUMMARY_FIGURE, ROBUST_FIGURE, RECEIPT, README))
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "package_contract": "exclusive_fixed_file_set_v1",
            "completion_status": "complete",
            "required_payloads": payloads,
            "files": {
                name: {
                    "bytes": (output / name).stat().st_size,
                    "sha256": _file_sha256(output / name),
                }
                for name in payloads
            },
            "source_result_sha256": {
                "three_group": three.result_sha256,
                "five_group": five.result_sha256,
            },
        }
        manifest["manifest_sha256"] = _canonical_sha256(manifest)
        _write_json(output / OUTPUT_MANIFEST, manifest)
        return validate_presentation_output(output)
    except BaseException:
        shutil.rmtree(output)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--three-formal", type=Path, required=True)
    parser.add_argument("--five-formal", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = render_presentation(
        args.three_formal,
        args.five_formal,
        args.output_directory,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""CLI for the leakage-safe, loading-only multi-autosome AIM protocol."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from functools import wraps
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import sys
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

import numpy as np

import rspca.genomics_aim as genomics_aim_module
from rspca.datasets import array_sha256, canonical_sha256
from rspca.genomics_aim import (
    AIMProtocolConfig,
    AIMProtocolResult,
    INPUT_SCOPES,
    METHODS,
    MultiAutosomeGenotypeData,
    POPULATION_DESIGNS,
    run_aim_protocol,
    validate_aim_protocol_result,
)


OUTPUT_PACKAGE_SCHEMA_VERSION = 2
OUTPUT_FILE_NAMES = {
    "results_json": "aim_protocol_results.json",
    "method_summaries_csv": "aim_protocol_method_summaries.csv",
    "split_metrics_csv": "aim_protocol_split_metrics.csv",
    "population_metrics_csv": "aim_protocol_population_metrics.csv",
    "selection_frequencies_csv": "aim_protocol_selection_frequencies.csv",
    "registered_coordinates_csv": "aim_protocol_registered_coordinates.csv",
    "registered_stratification_png": "aim_protocol_registered_stratification.png",
    "panel_stability_png": "aim_protocol_panel_stability_and_delta.png",
    "output_manifest_json": "aim_protocol_output_manifest.json",
}
DATA_OUTPUT_ROLES = tuple(
    role for role in OUTPUT_FILE_NAMES if role != "output_manifest_json"
)
METHOD_DISPLAY_NAMES = {
    "pca": "PCA",
    "vp": "VP-SPCA",
    "randomized": "Randomized VP-SPCA",
    "robust": "Robust VP-SPCA",
}


class _ProgressReporter:
    """Write machine-readable, non-scientific run progress to a text stream."""

    def __init__(
        self,
        config: AIMProtocolConfig,
        *,
        stream: TextIO | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._stream = sys.stderr if stream is None else stream
        self._clock = clock
        self._started_at = float(clock())
        self._completed = 0
        self._total = len(config.methods) * len(config.seeds)

    def _emit(
        self,
        phase: str,
        *,
        method: str | None,
        seed: int | None,
        status: str,
    ) -> None:
        elapsed = max(0.0, float(self._clock()) - self._started_at)
        payload = {
            "completed": self._completed,
            "elapsed_seconds_descriptive": round(elapsed, 3),
            "event": "genomics_aim_progress",
            "method": method,
            "phase": phase,
            "seed": seed,
            "status": status,
            "total": self._total,
        }
        self._stream.write(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._stream.flush()

    def start(self) -> None:
        self._emit("start", method=None, seed=None, status="running")

    def method_seed_complete(self, *, method: str, seed: int) -> None:
        self._completed += 1
        self._emit(
            "method_seed_complete",
            method=method,
            seed=seed,
            status="running",
        )

    def end(self, *, succeeded: bool) -> None:
        self._emit(
            "end",
            method=None,
            seed=None,
            status="completed" if succeeded else "failed",
        )


def _run_aim_protocol_with_progress(
    data: MultiAutosomeGenotypeData,
    config: AIMProtocolConfig,
    reporter: _ProgressReporter,
) -> AIMProtocolResult:
    """Observe completed split records without changing the protocol result.

    The scientific runner's smallest externally visible safe unit is one full
    seed.  Its private split evaluator returns all method records for that seed,
    so the events are emitted immediately after that return.  Temporarily
    wrapping that evaluator keeps progress out of the result schema and out of
    every RNG, solver, metric, and artifact byte path.
    """

    original_evaluator = genomics_aim_module._evaluate_aim_split_prevalidated

    def reporting_evaluator(*args: Any, **kwargs: Any) -> Any:
        evaluated = original_evaluator(*args, **kwargs)
        for record in evaluated[0]:
            reporter.method_seed_complete(
                method=str(record["method"]),
                seed=int(record["seed"]),
            )
        return evaluated

    genomics_aim_module._evaluate_aim_split_prevalidated = reporting_evaluator
    try:
        return run_aim_protocol(data, config)
    finally:
        genomics_aim_module._evaluate_aim_split_prevalidated = original_evaluator


def _delimiter(path: Path, explicit: str | None) -> str:
    if explicit is not None:
        if len(explicit) != 1:
            raise ValueError("delimiter must be exactly one character")
        return explicit
    return "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","


def _read_unique_mapping(
    path: Path,
    *,
    key_column: str,
    value_column: str,
    delimiter: str | None,
) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=_delimiter(path, delimiter))
        if reader.fieldnames is None:
            raise ValueError(f"metadata file has no header: {path}")
        missing = {key_column, value_column} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"metadata file is missing columns: {sorted(missing)}")
        mapping: dict[str, str] = {}
        for row in reader:
            key = str(row[key_column]).strip()
            value = str(row[value_column]).strip()
            if not key or not value:
                raise ValueError(f"metadata contains an empty key/value: {path}")
            if key in mapping:
                raise ValueError(f"duplicate metadata key {key!r}: {path}")
            mapping[key] = value
    if not mapping:
        raise ValueError(f"metadata file has no data rows: {path}")
    return mapping


def _npz_array(
    archive: Any,
    candidates: Sequence[str],
    *,
    role: str,
) -> np.ndarray:
    for candidate in candidates:
        if candidate in archive:
            return np.asarray(archive[candidate])
    raise ValueError(f"NPZ is missing {role}; expected one of {tuple(candidates)}")


def _chromosomes_from_canonical_snp_ids(snp_ids: np.ndarray) -> np.ndarray:
    chromosomes: list[str] = []
    for snp_id in np.asarray(snp_ids, dtype=str):
        prefix = str(snp_id).split(":", 1)[0].lower()
        if prefix.startswith("chr"):
            prefix = prefix[3:]
        try:
            chromosome = int(prefix)
        except ValueError as error:
            raise ValueError(
                f"cannot infer chromosome from canonical SNP ID {snp_id!r}"
            ) from error
        if not 1 <= chromosome <= 22:
            raise ValueError(f"non-autosomal SNP ID {snp_id!r}")
        chromosomes.append(str(chromosome))
    return np.asarray(chromosomes, dtype=str)


def load_protocol_input(
    genotype_npz: Path,
    population_metadata: Path | None,
    *,
    source_manifest_sha256: str,
    input_scope: str,
    ld_pruned: bool,
    sample_id_column: str = "sample_id",
    population_column: str = "population",
    sample_metadata_delimiter: str | None = None,
    variant_metadata: Path | None = None,
    infer_chromosomes_from_snp_ids: bool = False,
    variant_id_column: str = "snp_id",
    chromosome_column: str = "chromosome",
    variant_metadata_delimiter: str | None = None,
) -> MultiAutosomeGenotypeData:
    """Load a strict NPZ and align sample/variant metadata by identifier."""

    with np.load(genotype_npz, allow_pickle=False) as archive:
        matrix = np.asarray(
            _npz_array(archive, ("genotypes", "X"), role="genotype matrix"),
            dtype=float,
        )
        sample_ids = np.asarray(
            _npz_array(archive, ("sample_ids", "row_ids"), role="sample IDs"),
            dtype=str,
        )
        snp_ids = np.asarray(
            _npz_array(archive, ("snp_ids", "column_ids"), role="SNP IDs"),
            dtype=str,
        )
        embedded_chromosomes = (
            np.asarray(archive["chromosomes"], dtype=str)
            if "chromosomes" in archive
            else None
        )
        embedded_population_labels = (
            np.asarray(archive["population_labels"], dtype=str)
            if "population_labels" in archive
            else None
        )

    if embedded_population_labels is not None:
        if population_metadata is not None:
            raise ValueError(
                "provide population labels in exactly one place: NPZ or metadata"
            )
        labels = embedded_population_labels
    else:
        if population_metadata is None:
            raise ValueError(
                "NPZ has no population_labels array; provide --population-metadata"
            )
        population_mapping = _read_unique_mapping(
            population_metadata,
            key_column=sample_id_column,
            value_column=population_column,
            delimiter=sample_metadata_delimiter,
        )
        missing_samples = [
            sample for sample in sample_ids if sample not in population_mapping
        ]
        if missing_samples:
            raise ValueError(
                f"population metadata is missing {len(missing_samples)} samples; "
                f"first={missing_samples[0]!r}"
            )
        labels = np.asarray(
            [population_mapping[sample] for sample in sample_ids], dtype=str
        )

    if embedded_chromosomes is not None:
        if variant_metadata is not None or infer_chromosomes_from_snp_ids:
            raise ValueError(
                "provide chromosomes in exactly one place: NPZ, variant metadata, "
                "or canonical SNP IDs"
            )
        chromosomes = embedded_chromosomes
    else:
        if variant_metadata is not None and infer_chromosomes_from_snp_ids:
            raise ValueError(
                "choose either variant metadata or canonical SNP-ID inference"
            )
        if infer_chromosomes_from_snp_ids:
            chromosomes = _chromosomes_from_canonical_snp_ids(snp_ids)
        elif variant_metadata is None:
            raise ValueError(
                "NPZ has no chromosomes array; provide --variant-metadata or "
                "--chromosomes-from-snp-ids"
            )
        else:
            chromosome_mapping = _read_unique_mapping(
                variant_metadata,
                key_column=variant_id_column,
                value_column=chromosome_column,
                delimiter=variant_metadata_delimiter,
            )
            missing_snps = [snp for snp in snp_ids if snp not in chromosome_mapping]
            if missing_snps:
                raise ValueError(
                    f"variant metadata is missing {len(missing_snps)} SNPs; "
                    f"first={missing_snps[0]!r}"
                )
            chromosomes = np.asarray(
                [chromosome_mapping[snp] for snp in snp_ids], dtype=str
            )

    return MultiAutosomeGenotypeData(
        genotypes=matrix,
        sample_ids=sample_ids,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
        population_labels=labels,
        source_manifest_sha256=source_manifest_sha256,
        input_scope=input_scope,
        ld_pruned=ld_pruned,
    )


def _parse_csv_values(value: str, *, role: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError(f"{role} must not be empty")
    return values


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = _parse_csv_values(value, role="methods")
    invalid = [method for method in methods if method not in METHODS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown methods {invalid}; expected values from {METHODS}"
        )
    return methods


def _parse_seeds(value: str) -> tuple[int, ...]:
    values = _parse_csv_values(value, role="seeds")
    try:
        seeds = tuple(int(item) for item in values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must be non-negative")
    return seeds


def preflight_output_directory(output_directory: Path) -> None:
    """Reject every pre-existing output target, including dangling symlinks."""

    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        kind = "symlink" if output_directory.is_symlink() else "filesystem entry"
        raise FileExistsError(
            f"output path already exists as a {kind}: {output_directory}"
        )


def _reject_nonfinite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite numeric value at {path}")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    _reject_nonfinite(payload, path=path.name)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        handle.write("\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path.name}")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"CSV rows do not share one schema: {path.name}")
    _reject_nonfinite(rows, path=path.name)
    for row_index, row in enumerate(rows):
        for field, value in row.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                raise ValueError(
                    f"nested CSV value at {path.name}[{row_index}].{field}"
                )
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, figure: Any, *, description: str) -> None:
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        metadata={
            "Software": "vp-rspca genomics_aim_protocol",
            "Description": description,
        },
    )
    with path.open("xb") as handle:
        handle.write(buffer.getvalue())


def _with_default_matplotlib_style(
    render: Callable[..., None],
) -> Callable[..., None]:
    """Isolate protocol figures from caller-owned global Matplotlib styles."""

    @wraps(render)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        import matplotlib

        matplotlib.use("Agg", force=True)
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            render(*args, **kwargs)

    return wrapped


@_with_default_matplotlib_style
def _render_registered_stratification(
    result: AIMProtocolResult,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    evaluations = (
        ("full_width_native", "Full-width native embedding"),
        ("top_k_native_embedding", "Top-K native embedding"),
        ("top_k_common_pca_probe", "Top-K common PCA probe"),
    )
    methods = tuple(result.config.methods)
    populations = sorted({str(row["truth"]) for row in result.registered_coordinates})
    palette = plt.get_cmap("tab10")
    colors = {
        population: palette(index % 10) for index, population in enumerate(populations)
    }
    figure, axes = plt.subplots(
        len(methods),
        len(evaluations),
        figsize=(13.5, max(3.2, 2.85 * len(methods))),
        squeeze=False,
    )
    for row_index, method in enumerate(methods):
        for column_index, (evaluation, title) in enumerate(evaluations):
            axis = axes[row_index, column_index]
            rows = sorted(
                (
                    row
                    for row in result.registered_coordinates
                    if row["method"] == method and row["evaluation"] == evaluation
                ),
                key=lambda row: int(row["sample_index"]),
            )
            for population in populations:
                population_rows = [row for row in rows if row["truth"] == population]
                correct = [
                    row for row in population_rows if row["prediction"] == row["truth"]
                ]
                incorrect = [
                    row for row in population_rows if row["prediction"] != row["truth"]
                ]
                if correct:
                    axis.scatter(
                        [float(row["component_1"]) for row in correct],
                        [float(row["component_2"]) for row in correct],
                        s=22,
                        alpha=0.82,
                        color=colors[population],
                        label=population,
                    )
                if incorrect:
                    axis.scatter(
                        [float(row["component_1"]) for row in incorrect],
                        [float(row["component_2"]) for row in incorrect],
                        s=36,
                        marker="x",
                        linewidths=1.2,
                        color=colors[population],
                    )
            qualification = str(rows[0]["qualification_status"])
            qualification_color = (
                "#20732d" if qualification == "qualified" else "#a12b2b"
            )
            axis.set_title(title, fontsize=9)
            axis.text(
                0.02,
                0.98,
                qualification,
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=7.5,
                color=qualification_color,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none"},
            )
            axis.axhline(0.0, color="#d4d4d4", linewidth=0.6, zorder=0)
            axis.axvline(0.0, color="#d4d4d4", linewidth=0.6, zorder=0)
            axis.set_xlabel("Component 1")
            axis.set_ylabel("Component 2")
            if column_index == 0:
                axis.text(
                    -0.28,
                    0.5,
                    METHOD_DISPLAY_NAMES[method],
                    transform=axis.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=10,
                    weight="bold",
                )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=len(populations),
            frameon=False,
            bbox_to_anchor=(0.5, 0.965),
        )
    registered_seed = result.config.seeds[0]
    figure.suptitle(
        "Held-out population stratification — registered first outer seed "
        f"{registered_seed} (never best-seed selected)",
        fontsize=13,
        y=0.995,
    )
    figure.text(
        0.5,
        0.002,
        "Population labels are used only for train centroids and held-out scoring. "
        "Preprocessing and loading-only panels are train-only; each panel/solver "
        "has its own coordinate system. Cross-seed coordinates are not pooled. "
        "x = held-out misclassification.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.035, 0.045, 1.0, 0.93))
    _write_figure(
        path,
        figure,
        description=(
            "Held-out full-width and fixed-budget AIM coordinates from the first "
            "configured outer seed; not selected by outcome."
        ),
    )
    plt.close(figure)


@_with_default_matplotlib_style
def _render_panel_stability_and_delta(
    result: AIMProtocolResult,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    summaries = list(result.method_summaries)
    labels = [METHOD_DISPLAY_NAMES[str(summary["method"])] for summary in summaries]
    positions = np.arange(len(summaries), dtype=float)
    means = np.asarray(
        [summary["mean_pairwise_panel_jaccard_across_splits"] for summary in summaries],
        dtype=float,
    )
    minima = np.asarray(
        [
            summary["minimum_pairwise_panel_jaccard_across_splits"]
            for summary in summaries
        ],
        dtype=float,
    )
    deltas = np.asarray(
        [
            summary["mean_paired_delta_vs_pca_common_pca_probe_balanced_accuracy"]
            for summary in summaries
        ],
        dtype=float,
    )
    lows = np.asarray(
        [
            summary[
                "paired_delta_vs_pca_common_pca_probe_balanced_accuracy_bootstrap_ci95_low"
            ]
            for summary in summaries
        ],
        dtype=float,
    )
    highs = np.asarray(
        [
            summary[
                "paired_delta_vs_pca_common_pca_probe_balanced_accuracy_bootstrap_ci95_high"
            ]
            for summary in summaries
        ],
        dtype=float,
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), squeeze=False)
    stability_axis, delta_axis = axes[0]
    width = 0.36
    stability_axis.bar(
        positions - width / 2,
        means,
        width,
        label="Mean pairwise Jaccard",
        color="#4c78a8",
    )
    stability_axis.bar(
        positions + width / 2,
        minima,
        width,
        label="Minimum pairwise Jaccard",
        color="#9ecae9",
    )
    stability_axis.set_ylim(0.0, 1.02)
    stability_axis.set_ylabel("Panel overlap")
    stability_axis.set_title("Top-K panel stability across registered splits")
    stability_axis.legend(frameon=False, fontsize=8)

    lower_errors = np.maximum(0.0, deltas - lows)
    upper_errors = np.maximum(0.0, highs - deltas)
    bar_colors = [
        (
            "#2f855a"
            if summary["qualification_status"] == "all_splits_qualified"
            else "#c05621"
        )
        for summary in summaries
    ]
    delta_axis.bar(positions, deltas, color=bar_colors, alpha=0.86)
    delta_axis.errorbar(
        positions,
        deltas,
        yerr=np.vstack([lower_errors, upper_errors]),
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.0,
    )
    delta_axis.axhline(0.0, color="black", linewidth=0.8)
    delta_axis.set_ylabel("Balanced-accuracy delta vs PCA panel")
    delta_axis.set_title("Common PCA probe: paired mean and percentile 95% CI")
    for index, summary in enumerate(summaries):
        delta_axis.text(
            index,
            deltas[index],
            str(summary["qualification_status"]),
            rotation=90,
            va="bottom" if deltas[index] >= 0.0 else "top",
            ha="center",
            fontsize=7,
        )
    for axis in (stability_axis, delta_axis):
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.2)
    figure.suptitle(
        "Fixed-budget loading-panel evidence (same K for every method)",
        fontsize=13,
    )
    figure.text(
        0.5,
        0.005,
        f"Paired bootstrap: {result.config.bootstrap_resamples} resamples, "
        f"seed {result.config.bootstrap_seed}. Runtime is not bootstrapped. "
        "Orange bars indicate solver-qualified claims are not available.",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.93))
    _write_figure(
        path,
        figure,
        description=(
            "Cross-split fixed-budget panel stability and paired common-PCA-probe "
            "balanced-accuracy deltas with qualification status."
        ),
    )
    plt.close(figure)


def write_protocol_outputs(
    result: AIMProtocolResult,
    data: MultiAutosomeGenotypeData,
    output_directory: Path,
) -> dict[str, Path]:
    """Write and then validate one fixed, non-overwriting result package."""

    validate_aim_protocol_result(result, data)
    preflight_output_directory(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    paths = {
        role: output_directory / filename
        for role, filename in OUTPUT_FILE_NAMES.items()
    }

    payload = {
        "schema_version": OUTPUT_PACKAGE_SCHEMA_VERSION,
        "result_sha256": result.result_sha256,
        "config": asdict(result.config),
        "input_identity": result.input_identity,
        "method_summaries": list(result.method_summaries),
        "split_records": list(result.split_records),
        "selection_frequency_hashes": {
            method: array_sha256(frequency)
            for method, frequency in result.selection_frequencies.items()
        },
        "registered_seed_policy": "first_config_seed_not_best",
        "registered_coordinates": list(result.registered_coordinates),
        "registered_coordinates_sha256": canonical_sha256(
            list(result.registered_coordinates)
        ),
        "solver_lineage": result.solver_lineage,
    }
    _write_json(paths["results_json"], payload)
    _write_csv(
        paths["method_summaries_csv"],
        [dict(row) for row in result.method_summaries],
    )

    split_fields = (
        "seed",
        "method",
        "input_scope",
        "population_design",
        "train_samples",
        "test_samples",
        "input_snps",
        "training_retained_snps",
        "panel_size",
        "prepanel_solver_converged",
        "prepanel_native_support_count",
        "prepanel_native_support_fraction",
        "prepanel_native_support_sufficient_for_panel",
        "prepanel_native_support_degenerate",
        "prepanel_qualified",
        "prepanel_qualification_status",
        "full_width_native_balanced_accuracy",
        "native_panel_refit_solver_converged",
        "native_panel_refit_qualification_status",
        "overall_qualified",
        "qualification_status",
        "native_embedding_accuracy",
        "native_embedding_balanced_accuracy",
        "native_embedding_macro_f1",
        "common_pca_probe_accuracy",
        "common_pca_probe_balanced_accuracy",
        "common_pca_probe_macro_f1",
        "paired_delta_vs_pca_native_embedding_balanced_accuracy",
        "paired_delta_vs_pca_common_pca_probe_balanced_accuracy",
        "native_panel_retention_delta_balanced_accuracy",
        "common_pca_probe_retention_delta_vs_full_width_pca_balanced_accuracy",
        "panel_jaccard_with_equal_budget_pca",
        "selected_snp_ids_sha256",
        "train_indices_sha256",
        "test_indices_sha256",
    )
    _write_csv(
        paths["split_metrics_csv"],
        [
            {field: record[field] for field in split_fields}
            for record in result.split_records
        ],
    )

    population_rows: list[dict[str, Any]] = []
    for record in result.split_records:
        for evaluation in (
            "full_width_native",
            "native_embedding",
            "common_pca_probe",
        ):
            for metrics in record[f"{evaluation}_per_population_metrics"]:
                population_rows.append(
                    {
                        "seed": record["seed"],
                        "method": record["method"],
                        "evaluation": evaluation,
                        **metrics,
                    }
                )
    _write_csv(paths["population_metrics_csv"], population_rows)

    snp_ids = np.asarray(data.snp_ids, dtype=str)
    chromosomes = np.asarray(data.chromosomes, dtype=str)
    frequency_rows = [
        {
            "method": method,
            "snp_index": index,
            "snp_id": str(snp_ids[index]),
            "chromosome": str(chromosomes[index]),
            "selection_frequency": float(frequency[index]),
        }
        for method, frequency in result.selection_frequencies.items()
        for index in range(snp_ids.size)
    ]
    _write_csv(paths["selection_frequencies_csv"], frequency_rows)
    _write_csv(
        paths["registered_coordinates_csv"],
        [dict(row) for row in result.registered_coordinates],
    )
    _render_registered_stratification(
        result,
        paths["registered_stratification_png"],
    )
    _render_panel_stability_and_delta(result, paths["panel_stability_png"])

    manifest = {
        "schema_version": OUTPUT_PACKAGE_SCHEMA_VERSION,
        "package_contract": "fixed_file_set_v1",
        "completion_status": "complete",
        "result_sha256": result.result_sha256,
        "registered_seed": result.config.seeds[0],
        "registered_seed_policy": "first_config_seed_not_best",
        "solver_lineage": result.solver_lineage,
        "expected_filenames": sorted(OUTPUT_FILE_NAMES.values()),
        "files": {
            role: {
                "path": paths[role].name,
                "sha256": _file_sha256(paths[role]),
                "bytes": paths[role].stat().st_size,
            }
            for role in DATA_OUTPUT_ROLES
        },
    }
    # This is intentionally the final write. Its presence means all fixed files
    # were produced and their bytes were bound into one completion receipt.
    _write_json(paths["output_manifest_json"], manifest)
    validate_output_package(output_directory)
    return paths


def _strict_json_load(path: Path) -> Any:
    def reject_constant(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token!r} in {path.name}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    _reject_nonfinite(payload, path=path.name)
    return payload


def _validate_regular_file(path: Path, *, role: str) -> os.stat_result:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"missing package file for {role}: {path.name}") from error
    if stat.S_ISLNK(file_stat.st_mode):
        raise ValueError(f"package file for {role} must not be a symlink")
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError(f"package file for {role} must be a regular file")
    return file_stat


def _valid_lower_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {path.name}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"malformed CSV row in {path.name}")
    return list(reader.fieldnames), rows


def validate_output_package(output_directory: Path) -> dict[str, Any]:
    """Validate the fixed file set, completion receipt, hashes, and CSV payloads."""

    output_directory = Path(output_directory)
    try:
        directory_stat = output_directory.lstat()
    except FileNotFoundError as error:
        raise ValueError(
            f"output package does not exist: {output_directory}"
        ) from error
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise ValueError("output package root must be a real directory, not a symlink")

    expected_names = set(OUTPUT_FILE_NAMES.values())
    actual_names = {entry.name for entry in output_directory.iterdir()}
    if actual_names != expected_names:
        raise ValueError(
            "output package file set mismatch; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unknown={sorted(actual_names - expected_names)}"
        )
    paths = {
        role: output_directory / filename
        for role, filename in OUTPUT_FILE_NAMES.items()
    }
    stats = {
        role: _validate_regular_file(path, role=role) for role, path in paths.items()
    }

    manifest = _strict_json_load(paths["output_manifest_json"])
    expected_manifest_keys = {
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
    if not isinstance(manifest, dict) or set(manifest) != expected_manifest_keys:
        raise ValueError("output manifest schema or field set is invalid")
    if (
        manifest["schema_version"] != OUTPUT_PACKAGE_SCHEMA_VERSION
        or manifest["package_contract"] != "fixed_file_set_v1"
        or manifest["completion_status"] != "complete"
        or manifest["registered_seed_policy"] != "first_config_seed_not_best"
        or manifest["expected_filenames"] != sorted(expected_names)
    ):
        raise ValueError("output manifest contract declaration is invalid")
    if not _valid_lower_sha256(manifest["result_sha256"]):
        raise ValueError("output manifest result_sha256 is invalid")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(DATA_OUTPUT_ROLES):
        raise ValueError("output manifest file roles do not match the fixed contract")
    for role in DATA_OUTPUT_ROLES:
        entry = files[role]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise ValueError(f"invalid manifest entry schema for {role}")
        if entry["path"] != OUTPUT_FILE_NAMES[role]:
            raise ValueError(f"manifest path mismatch for {role}")
        if not _valid_lower_sha256(entry["sha256"]):
            raise ValueError(f"invalid manifest SHA-256 for {role}")
        if (
            isinstance(entry["bytes"], bool)
            or not isinstance(entry["bytes"], int)
            or entry["bytes"] <= 0
        ):
            raise ValueError(f"invalid manifest byte count for {role}")
        if stats[role].st_size != entry["bytes"]:
            raise ValueError(f"byte-count mismatch for {role}")
        if _file_sha256(paths[role]) != entry["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {role}")

    manifest_mtime = stats["output_manifest_json"].st_mtime_ns
    if any(stats[role].st_mtime_ns > manifest_mtime for role in DATA_OUTPUT_ROLES):
        raise ValueError("completion manifest is not the final package write")

    payload = _strict_json_load(paths["results_json"])
    expected_payload_keys = {
        "schema_version",
        "result_sha256",
        "config",
        "input_identity",
        "method_summaries",
        "split_records",
        "selection_frequency_hashes",
        "registered_seed_policy",
        "registered_coordinates",
        "registered_coordinates_sha256",
        "solver_lineage",
    }
    if not isinstance(payload, dict) or set(payload) != expected_payload_keys:
        raise ValueError("results JSON schema or field set is invalid")
    if (
        payload["schema_version"] != OUTPUT_PACKAGE_SCHEMA_VERSION
        or payload["result_sha256"] != manifest["result_sha256"]
        or payload["registered_seed_policy"] != "first_config_seed_not_best"
        or payload["config"].get("seeds", [None])[0] != manifest["registered_seed"]
        or payload["solver_lineage"] != manifest["solver_lineage"]
    ):
        raise ValueError("results JSON identity disagrees with completion manifest")
    methods = tuple(payload["config"]["methods"])
    if set(payload["solver_lineage"]) != set(methods):
        raise ValueError("solver lineage does not match configured methods")
    for method in methods:
        lineage = payload["solver_lineage"][method]
        if (
            not isinstance(lineage, dict)
            or not _valid_lower_sha256(lineage.get("source_sha256"))
            or not _valid_lower_sha256(lineage.get("panel_ranking_source_sha256"))
            or not isinstance(lineage.get("qualification_role"), str)
            or not lineage["qualification_role"]
            or lineage.get("panel_selection_uses_population_labels") is not False
            or lineage.get("supervised_feature_ranking") is not False
        ):
            raise ValueError(f"invalid solver lineage for {method}")
    coordinate_hash = canonical_sha256(payload["registered_coordinates"])
    if coordinate_hash != payload["registered_coordinates_sha256"]:
        raise ValueError("registered coordinate hash mismatch")
    recomputed_result_hash = canonical_sha256(
        {
            "config": payload["config"],
            "input_identity": payload["input_identity"],
            "split_records": payload["split_records"],
            "method_summaries": payload["method_summaries"],
            "selection_frequency_hashes": payload["selection_frequency_hashes"],
            "registered_coordinates": payload["registered_coordinates"],
            "solver_lineage": payload["solver_lineage"],
        }
    )
    if recomputed_result_hash != payload["result_sha256"]:
        raise ValueError("results JSON does not reproduce result_sha256")

    coordinate_fields, coordinate_rows = _csv_rows(paths["registered_coordinates_csv"])
    expected_coordinate_fields = [
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
    ]
    if coordinate_fields != expected_coordinate_fields:
        raise ValueError("registered coordinate CSV schema is invalid")
    expected_coordinate_rows = [
        {field: str(row[field]) for field in expected_coordinate_fields}
        for row in payload["registered_coordinates"]
    ]
    if coordinate_rows != expected_coordinate_rows:
        raise ValueError("registered coordinate CSV disagrees with results JSON")
    for row_index, row in enumerate(coordinate_rows):
        for field in ("component_1", "component_2"):
            try:
                value = float(row[field])
            except ValueError as error:
                raise ValueError(
                    f"coordinate CSV has a nonnumeric {field} at row {row_index}"
                ) from error
            if not math.isfinite(value):
                raise ValueError(
                    f"coordinate CSV has a non-finite {field} at row {row_index}"
                )

    frequency_fields, frequency_rows = _csv_rows(paths["selection_frequencies_csv"])
    if frequency_fields != [
        "method",
        "snp_index",
        "snp_id",
        "chromosome",
        "selection_frequency",
    ]:
        raise ValueError("selection frequency CSV schema is invalid")
    snps = int(payload["input_identity"]["snps"])
    if len(frequency_rows) != len(methods) * snps:
        raise ValueError("selection frequency CSV has the wrong row count")
    frequency_arrays: dict[str, np.ndarray] = {
        method: np.empty(snps, dtype=float) for method in methods
    }
    seen_frequency_keys: set[tuple[str, int]] = set()
    for row in frequency_rows:
        method = row["method"]
        try:
            snp_index = int(row["snp_index"])
            value = float(row["selection_frequency"])
        except ValueError as error:
            raise ValueError(
                "selection frequency CSV has a nonnumeric value"
            ) from error
        key = (method, snp_index)
        if (
            method not in frequency_arrays
            or not 0 <= snp_index < snps
            or key in seen_frequency_keys
        ):
            raise ValueError("selection frequency CSV key is invalid or duplicated")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                "selection frequency CSV values must be finite and in [0, 1]"
            )
        seen_frequency_keys.add(key)
        frequency_arrays[method][snp_index] = value
    if set(payload["selection_frequency_hashes"]) != set(methods):
        raise ValueError("selection frequency hashes do not match configured methods")
    for method, frequency in frequency_arrays.items():
        if array_sha256(frequency) != payload["selection_frequency_hashes"][method]:
            raise ValueError(f"selection frequency payload mismatch for {method}")

    for role in ("registered_stratification_png", "panel_stability_png"):
        with paths[role].open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"invalid PNG signature for {role}")
    return manifest


def audit_output_package_claim_gates(output_directory: Path) -> dict[str, Any]:
    """Read-only native-support claim audit for old and new result packages.

    Frozen server packages created before the native-support hardening do not
    contain ``prepanel_native_support_sufficient_for_panel``.  Their retained
    support and solver-status fields are sufficient to rederive all
    qualification gates without rerunning a solver or changing panel
    membership.
    """

    output_directory = Path(output_directory)
    manifest = validate_output_package(output_directory)
    payload = _strict_json_load(output_directory / OUTPUT_FILE_NAMES["results_json"])
    config = payload["config"]
    panel_size = int(config["panel_size"])
    methods = tuple(str(method) for method in config["methods"])
    seeds = tuple(int(seed) for seed in config["seeds"])
    expected_keys = {(seed, method) for seed in seeds for method in methods}
    records_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    qualification_failures: list[dict[str, Any]] = []
    read_side_qualified_by_key: dict[tuple[int, str], bool] = {}
    for record in payload["split_records"]:
        key = (int(record["seed"]), str(record["method"]))
        if key in records_by_key:
            raise ValueError(f"duplicate split record in claim audit: {key}")
        records_by_key[key] = record
        count = int(record["prepanel_native_support_count"])
        retained_count = int(record["training_retained_snps"])
        if retained_count <= 0 or not 0 <= count <= retained_count:
            raise ValueError("retained native-support count is out of range")
        expected_fraction = count / retained_count
        if not math.isclose(
            float(record["prepanel_native_support_fraction"]),
            expected_fraction,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("retained native-support count/fraction disagree")
        sufficient = bool(count >= panel_size)
        declared = record.get("prepanel_native_support_sufficient_for_panel")
        if declared is not None and bool(declared) != sufficient:
            raise ValueError(
                "declared native-support panel-size gate disagrees with retained count"
            )
        if not sufficient:
            failures.append(
                {
                    "seed": key[0],
                    "method": key[1],
                    "prepanel_native_support_count": count,
                    "panel_size": panel_size,
                    "risk": "zero-loading SNPs can be tie-filled into exact top-K",
                }
            )
        reasons: list[str] = []
        if not bool(record["prepanel_solver_converged"]):
            reasons.append("prepanel_solver_not_converged")
        if not sufficient:
            reasons.append("prepanel_native_support_below_panel_size")
        if key[1] != "pca" and expected_fraction >= float(
            config["near_dense_fraction"]
        ):
            reasons.append("prepanel_native_support_near_dense")
        if not bool(record["native_panel_refit_solver_converged"]):
            reasons.append("native_panel_refit_solver_not_converged")
        if not bool(record["common_pca_probe_solver_converged"]):
            reasons.append("common_pca_probe_not_converged")
        read_side_qualified_by_key[key] = not reasons
        if reasons:
            qualification_failures.append(
                {"seed": key[0], "method": key[1], "reasons": reasons}
            )
    if set(records_by_key) != expected_keys:
        raise ValueError("claim audit split records are not an exact seed/method cover")

    method_gates: dict[str, Any] = {}
    for method in methods:
        records = [records_by_key[(seed, method)] for seed in seeds]
        support_counts = [
            int(record["prepanel_native_support_count"]) for record in records
        ]
        original_qualified = [bool(record["overall_qualified"]) for record in records]
        read_side_qualified = [
            read_side_qualified_by_key[(seed, method)] for seed in seeds
        ]
        method_gates[method] = {
            "minimum_prepanel_native_support_count": min(support_counts),
            "all_splits_native_support_at_least_panel_size": bool(
                min(support_counts) >= panel_size
            ),
            "original_qualified_splits": int(np.count_nonzero(original_qualified)),
            "read_side_qualified_splits": int(
                np.count_nonzero(read_side_qualified)
            ),
            "all_splits_read_side_qualified": bool(all(read_side_qualified)),
        }
    all_support_sufficient = not failures
    all_read_side_qualified = bool(
        all(record["all_splits_read_side_qualified"] for record in method_gates.values())
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "audit_kind": "genomics_aim_native_support_read_only_claim_audit",
        "artifact_result_sha256": payload["result_sha256"],
        "artifact_manifest_result_sha256": manifest["result_sha256"],
        "input_identity": payload["input_identity"],
        "population_design": config["population_design"],
        "methods": list(methods),
        "seeds": list(seeds),
        "panel_size": panel_size,
        "all_split_methods_native_support_at_least_panel_size": (
            all_support_sufficient
        ),
        "native_support_failures": failures,
        "read_side_qualification_failures": qualification_failures,
        "method_gates": method_gates,
        "all_methods_all_splits_read_side_qualified": all_read_side_qualified,
        "claim_status": (
            "all_methods_all_splits_read_side_qualified"
            if all_read_side_qualified
            else (
                "unqualified_zero_loading_tie_fill_risk"
                if not all_support_sufficient
                else "native_support_gate_passed_but_other_qualification_gates_failed"
            )
        ),
        "panel_semantics": (
            "per-split loading-ranked fixed-K panels; not a single frozen consensus "
            "panel and not intrinsic sparse support"
        ),
    }
    audit["audit_body_sha256"] = canonical_sha256(audit)
    return audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genotypes", type=Path, required=True)
    parser.add_argument(
        "--population-metadata",
        type=Path,
        help="Required only when the NPZ has no population_labels array.",
    )
    parser.add_argument("--variant-metadata", type=Path)
    parser.add_argument(
        "--chromosomes-from-snp-ids",
        action="store_true",
        help="Infer autosome from the prefix of canonical CHR:POS:REF:ALT IDs.",
    )
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--input-scope", choices=INPUT_SCOPES, required=True)
    parser.add_argument(
        "--population-design", choices=POPULATION_DESIGNS, required=True
    )
    parser.add_argument(
        "--ld-pruned",
        action="store_true",
        help="Declare that the supplied variants passed an external LD-pruning receipt.",
    )
    parser.add_argument("--sample-id-column", default="sample_id")
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--sample-metadata-delimiter")
    parser.add_argument("--variant-id-column", default="snp_id")
    parser.add_argument("--chromosome-column", default="chromosome")
    parser.add_argument("--variant-metadata-delimiter")
    parser.add_argument("--methods", type=_parse_methods, default=METHODS)
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--panel-size", type=int, default=200)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--seeds", type=_parse_seeds, default=(2025, 2026, 2027, 2028, 2029)
    )
    parser.add_argument("--loading-score", choices=("l1", "l2", "max"), default="l1")
    parser.add_argument("--native-support-threshold", type=float, default=1e-8)
    parser.add_argument("--near-dense-fraction", type=float, default=0.95)
    parser.add_argument("--vp-alpha", type=float, default=0.1)
    parser.add_argument("--randomized-alpha", type=float, default=0.1)
    parser.add_argument("--robust-alpha", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--randomized-oversample", type=int, default=10)
    parser.add_argument("--randomized-power-iterations", type=int, default=2)
    parser.add_argument("--randomized-finalize-full-iters", type=int, default=5)
    parser.add_argument("--robust-huber-delta", type=float, default=1.345)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1701)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # This must precede input loading and every solver call. It catches files,
    # directories, valid symlinks, and dangling symlinks via lexists().
    preflight_output_directory(args.output_dir)
    if not args.ld_pruned:
        raise SystemExit(
            "--ld-pruned is required; provide it only with an external LD-pruning receipt"
        )
    data = load_protocol_input(
        args.genotypes,
        args.population_metadata,
        source_manifest_sha256=args.source_manifest_sha256,
        input_scope=args.input_scope,
        ld_pruned=args.ld_pruned,
        sample_id_column=args.sample_id_column,
        population_column=args.population_column,
        sample_metadata_delimiter=args.sample_metadata_delimiter,
        variant_metadata=args.variant_metadata,
        infer_chromosomes_from_snp_ids=args.chromosomes_from_snp_ids,
        variant_id_column=args.variant_id_column,
        chromosome_column=args.chromosome_column,
        variant_metadata_delimiter=args.variant_metadata_delimiter,
    )
    config = AIMProtocolConfig(
        population_design=args.population_design,
        methods=tuple(args.methods),
        n_components=args.n_components,
        panel_size=args.panel_size,
        test_fraction=args.test_fraction,
        seeds=tuple(args.seeds),
        loading_score=args.loading_score,
        native_support_threshold=args.native_support_threshold,
        near_dense_fraction=args.near_dense_fraction,
        vp_alpha=args.vp_alpha,
        randomized_alpha=args.randomized_alpha,
        robust_alpha=args.robust_alpha,
        max_iter=args.max_iter,
        tol=args.tol,
        randomized_oversample=args.randomized_oversample,
        randomized_power_iterations=args.randomized_power_iterations,
        randomized_finalize_full_iters=args.randomized_finalize_full_iters,
        robust_huber_delta=args.robust_huber_delta,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    progress = _ProgressReporter(config)
    progress.start()
    try:
        result = _run_aim_protocol_with_progress(data, config, progress)
        paths = write_protocol_outputs(result, data, args.output_dir)
    except BaseException:
        progress.end(succeeded=False)
        raise
    progress.end(succeeded=True)
    print(
        json.dumps(
            {
                "result_sha256": result.result_sha256,
                "outputs": {role: str(path) for role, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rebuild auditable population diagnostics from a sealed Genomics result.

The source benchmark remains read-only.  Each requested method/seed is rerun
with the source configuration and inputs, and its split/prediction hashes and
primary metrics must reproduce before any report is published.  The new
directory contains per-split held-out predictions, pooled confusion matrices,
per-population metrics, and a deterministic figure.

These labels are metadata population/ancestry groups (for the formal 1000
Genomes run, five ``super_pop`` groups).  They are not biological races.  The
evaluation measures repeated held-out splits of one fixed cohort, not
generalization to an independent cohort or individual ancestry estimation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

if __package__ in {None, ""}:  # Direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.genomics_1kg import (
    _classification_diagnostics,
    load_genotypes,
    load_population_labels,
    run_genomics,
)
from experiments.genomics_1kg_benchmark import (
    GenomicsBenchmarkConfig,
    METHODS,
    _single_run_config,
    validate_result_payload,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from tools.validate_nonfed_release import (
    GENOMICS_TRUST_ANCHOR_PATH,
    load_trusted_genomics_provenance,
)


REPORT_SCHEMA_VERSION = 1
PRIMARY_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
REPRODUCTION_HASH_FIELDS = (
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "test_predictions_sha256",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> Mapping[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON object key {key!r} in {path}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON number {value!r} in {path}")

    with path.open(encoding="utf-8") as handle:
        payload = json.load(
            handle,
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    if not isinstance(payload, Mapping):
        raise ValueError("source benchmark JSON must contain an object")
    return payload


def _benchmark_config(payload: Mapping[str, Any]) -> GenomicsBenchmarkConfig:
    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ValueError("source benchmark is missing its configuration")
    values = dict(raw)
    for field in ("seeds", "methods", "alpha_candidates"):
        if field in values:
            values[field] = tuple(values[field])
    try:
        return GenomicsBenchmarkConfig(**values)
    except TypeError as error:
        raise ValueError("source benchmark configuration is unsupported") from error


def _source_records(payload: Mapping[str, Any]) -> dict[tuple[str, int], Mapping[str, Any]]:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("source benchmark has no per-seed records")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("source per-seed records must be objects")
        try:
            key = (str(record["method"]), int(record["requested_seed"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("source record is missing method/seed identity") from error
        if key in indexed:
            raise ValueError(f"duplicate source record for method/seed {key}")
        indexed[key] = record
    return indexed


def _assert_reproduced(
    source: Mapping[str, Any],
    reproduced: Mapping[str, Any],
    *,
    method: str,
    seed: int,
    reproduced_manifest_sha256: str,
) -> None:
    if source.get("single_run_manifest_sha256") != reproduced_manifest_sha256:
        raise ValueError(
            f"{method} seed {seed} did not reproduce source single-run manifest"
        )
    for field in REPRODUCTION_HASH_FIELDS:
        if reproduced.get(field) != source.get(field):
            raise ValueError(
                f"{method} seed {seed} did not reproduce source {field}"
            )
    for metric in PRIMARY_METRICS:
        try:
            source_value = float(source[metric])
            reproduced_value = float(reproduced[metric])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{method} seed {seed} is missing primary metric {metric}"
            ) from error
        if not np.isclose(source_value, reproduced_value, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"{method} seed {seed} did not reproduce source {metric}: "
                f"{reproduced_value} versus {source_value}"
            )
    for field in ("split_seed", "selected_alpha", "solver_converged"):
        if reproduced.get(field) != source.get(field):
            raise ValueError(
                f"{method} seed {seed} did not reproduce source {field}"
            )


def reproduce_population_records(
    payload: Mapping[str, Any],
    dataset: Any,
    population_labels: np.ndarray,
    *,
    methods: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    expected_provenance: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rerun selected source cells and retain only outer-held-out diagnostics."""

    validate_result_payload(
        payload,
        label="source Genomics benchmark",
        expected_provenance=expected_provenance,
    )
    benchmark_config = _benchmark_config(payload)
    source = _source_records(payload)
    selected_methods = tuple(methods) if methods is not None else benchmark_config.methods
    selected_seeds = tuple(int(seed) for seed in seeds) if seeds is not None else benchmark_config.seeds
    if not selected_methods or not selected_seeds:
        raise ValueError("at least one method and seed are required")
    if len(set(selected_methods)) != len(selected_methods) or any(
        method not in benchmark_config.methods for method in selected_methods
    ):
        raise ValueError("requested methods must be unique members of the source run")
    if len(set(selected_seeds)) != len(selected_seeds) or any(
        seed not in benchmark_config.seeds for seed in selected_seeds
    ):
        raise ValueError("requested seeds must be unique members of the source run")

    enriched: list[dict[str, Any]] = []
    for seed in selected_seeds:
        for method in selected_methods:
            source_record = source.get((method, seed))
            if source_record is None:
                raise ValueError(f"source is missing method {method!r}, seed {seed}")
            single_config = _single_run_config(
                benchmark_config,
                method=method,
                seed=seed,
            )
            rows, single_manifest = run_genomics(
                dataset,
                single_config,
                population_labels=np.asarray(population_labels, dtype=str),
                include_population_diagnostics=True,
            )
            classifications = [
                row
                for row in rows
                if row.get("record_type") == "held_out_population_classification"
            ]
            if len(classifications) != 1:
                raise ValueError(
                    f"{method} seed {seed} did not return exactly one held-out record"
                )
            classification = classifications[0]
            _assert_reproduced(
                source_record,
                classification,
                method=method,
                seed=seed,
                reproduced_manifest_sha256=str(
                    single_manifest["manifest_sha256"]
                ),
            )
            enriched.append(
                {
                    "method": method,
                    "requested_seed": seed,
                    "source_run_index": int(source_record.get("run_index", -1)),
                    "source_single_run_manifest_sha256": source_record.get(
                        "single_run_manifest_sha256"
                    ),
                    "reproduced_single_run_manifest_sha256": single_manifest[
                        "manifest_sha256"
                    ],
                    **classification,
                }
            )
    return enriched


def _validate_prediction_record(record: Mapping[str, Any]) -> None:
    if int(record.get("population_diagnostics_schema_version", -1)) != 1:
        raise ValueError("record is missing population diagnostics schema version 1")
    class_order = record.get("class_order")
    predictions = record.get("held_out_prediction_records")
    if not isinstance(class_order, list) or not class_order:
        raise ValueError("population record is missing class_order")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("population record is missing held-out predictions")
    truth = np.asarray([row["true_population"] for row in predictions], dtype=str)
    predicted = np.asarray(
        [row["predicted_population"] for row in predictions], dtype=str
    )
    diagnostics = _classification_diagnostics(
        truth,
        predicted,
        class_order=[str(label) for label in class_order],
    )
    for field in (
        "confusion_matrix_counts",
        "confusion_matrix_true_normalized",
        "per_population_metrics",
    ):
        if diagnostics[field] != record.get(field):
            raise ValueError(f"population diagnostic {field} is internally inconsistent")
    for metric in PRIMARY_METRICS:
        if not np.isclose(
            float(diagnostics[metric]),
            float(record[metric]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"population diagnostic {metric} is internally inconsistent")
    if len(predictions) != int(record.get("test_samples", -1)):
        raise ValueError("prediction count does not match test_samples")


def aggregate_population_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate repeated held-out decisions without calling them independent OOF."""

    if not records:
        raise ValueError("at least one population record is required")
    by_method: dict[str, list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for record in records:
        _validate_prediction_record(record)
        key = (str(record["method"]), int(record["requested_seed"]))
        if key in seen:
            raise ValueError(f"duplicate population record {key}")
        seen.add(key)
        by_method.setdefault(key[0], []).append(record)

    method_summaries: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []
    per_population_rows: list[dict[str, Any]] = []
    for method, method_records in by_method.items():
        class_order = [str(label) for label in method_records[0]["class_order"]]
        if any(record["class_order"] != class_order for record in method_records):
            raise ValueError(f"class order changed across {method} splits")
        counts = np.sum(
            [np.asarray(record["confusion_matrix_counts"], dtype=np.int64) for record in method_records],
            axis=0,
        )
        supports = counts.sum(axis=1)
        normalized = counts / supports[:, None]
        pooled_truth: list[str] = []
        pooled_prediction: list[str] = []
        for record in method_records:
            for prediction in record["held_out_prediction_records"]:
                pooled_truth.append(str(prediction["true_population"]))
                pooled_prediction.append(str(prediction["predicted_population"]))
        pooled = _classification_diagnostics(
            np.asarray(pooled_truth, dtype=str),
            np.asarray(pooled_prediction, dtype=str),
            class_order=class_order,
        )
        method_summaries.append(
            {
                "method": method,
                "outer_splits": len(method_records),
                "held_out_decisions": len(pooled_truth),
                **{
                    f"mean_split_{metric}": float(
                        np.mean([float(record[metric]) for record in method_records])
                    )
                    for metric in PRIMARY_METRICS
                },
                **{
                    f"pooled_{metric}": float(pooled[metric])
                    for metric in PRIMARY_METRICS
                },
                "solver_converged_splits": int(
                    sum(bool(record["solver_converged"]) for record in method_records)
                ),
                "alpha_selection_qualified_splits": int(
                    sum(
                        bool(record["alpha_selection_qualified"])
                        for record in method_records
                    )
                ),
                "mean_final_loading_entry_sparsity": float(
                    np.mean(
                        [
                            float(record["final_loading_entry_sparsity"])
                            for record in method_records
                        ]
                    )
                ),
                "mean_final_selected_snp_count": float(
                    np.mean(
                        [
                            int(record["final_selected_snp_count"])
                            for record in method_records
                        ]
                    )
                ),
                "min_final_selected_snp_count": int(
                    min(int(record["final_selected_snp_count"]) for record in method_records)
                ),
                "max_final_selected_snp_count": int(
                    max(int(record["final_selected_snp_count"]) for record in method_records)
                ),
                "final_loading_support_threshold": float(
                    method_records[0]["final_loading_support_threshold"]
                ),
                "final_loading_support_coordinate_system": method_records[0][
                    "final_loading_support_coordinate_system"
                ],
            }
        )
        for true_index, true_label in enumerate(class_order):
            for predicted_index, predicted_label in enumerate(class_order):
                confusion_rows.append(
                    {
                        "method": method,
                        "true_population": true_label,
                        "predicted_population": predicted_label,
                        "count": int(counts[true_index, predicted_index]),
                        "true_row_fraction": float(
                            normalized[true_index, predicted_index]
                        ),
                    }
                )
        for population in pooled["per_population_metrics"]:
            label = str(population["population"])
            split_recalls = [
                next(
                    float(item["recall"])
                    for item in record["per_population_metrics"]
                    if item["population"] == label
                )
                for record in method_records
            ]
            per_population_rows.append(
                {
                    "method": method,
                    **population,
                    "outer_splits": len(method_records),
                    "mean_split_recall": float(np.mean(split_recalls)),
                    "std_split_recall": float(np.std(split_recalls, ddof=0)),
                    "min_split_recall": float(np.min(split_recalls)),
                    "max_split_recall": float(np.max(split_recalls)),
                }
            )
    return {
        "method_summaries": method_summaries,
        "confusion_rows": confusion_rows,
        "per_population_rows": per_population_rows,
    }


def _prediction_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        class_order = [str(label) for label in record["class_order"]]
        for prediction in record["held_out_prediction_records"]:
            embedding = [float(value) for value in prediction["embedding"]]
            distances = [
                float(value)
                for value in prediction["squared_distances_to_centroids"]
            ]
            row: dict[str, Any] = {
                "prediction_unit": "one_outer_split_held_out_decision",
                "method": str(record["method"]),
                "requested_seed": int(record["requested_seed"]),
                "split_seed": int(record["split_seed"]),
                "sample_index": int(prediction["sample_index"]),
                "sample_id": str(prediction["sample_id"]),
                "sample_id_utf8_sha256": hashlib.sha256(
                    str(prediction["sample_id"]).encode("utf-8")
                ).hexdigest(),
                "test_sample_ids_sha256": str(record["test_sample_ids_sha256"]),
                "test_indices_sha256": str(record["test_indices_sha256"]),
                "test_predictions_sha256": str(record["test_predictions_sha256"]),
                "single_run_manifest_sha256": str(
                    record["source_single_run_manifest_sha256"]
                ),
                "true_population": str(prediction["true_population"]),
                "predicted_population": str(prediction["predicted_population"]),
                "correct": bool(
                    prediction["true_population"] == prediction["predicted_population"]
                ),
            }
            row.update(
                {
                    f"embedding_component_{index + 1}": value
                    for index, value in enumerate(embedding)
                }
            )
            row.update(
                {
                    f"squared_distance_to_{label}": distances[index]
                    for index, label in enumerate(class_order)
                }
            )
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _population_figure(
    path: Path,
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(str(record["method"]) for record in records))
    first_by_method = {
        method: min(
            (record for record in records if record["method"] == method),
            key=lambda record: int(record["requested_seed"]),
        )
        for method in methods
    }
    all_labels = list(
        dict.fromkeys(
            str(label)
            for record in records
            for label in record["class_order"]
        )
    )
    palette = plt.get_cmap("tab10")
    colors = {label: palette(index % 10) for index, label in enumerate(all_labels)}
    figure, axes = plt.subplots(
        2,
        len(methods),
        figsize=(4.2 * len(methods), 8.2),
        squeeze=False,
        constrained_layout=True,
    )
    confusion_rows = aggregate["confusion_rows"]
    image = None
    for column, method in enumerate(methods):
        record = first_by_method[method]
        predictions = record["held_out_prediction_records"]
        embedding = np.asarray([row["embedding"] for row in predictions], dtype=float)
        truth = np.asarray([row["true_population"] for row in predictions], dtype=str)
        predicted = np.asarray(
            [row["predicted_population"] for row in predictions], dtype=str
        )
        x = embedding[:, 0]
        y = embedding[:, 1] if embedding.shape[1] > 1 else np.zeros(embedding.shape[0])
        embedding_axis = axes[0, column]
        for label in record["class_order"]:
            selected = truth == label
            embedding_axis.scatter(
                x[selected],
                y[selected],
                s=18,
                alpha=0.72,
                color=colors[str(label)],
                edgecolors="none",
            )
            if np.any(selected):
                embedding_axis.text(
                    float(np.median(x[selected])),
                    float(np.median(y[selected])),
                    str(label),
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=colors[str(label)],
                    bbox={
                        "boxstyle": "round,pad=0.18",
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.78,
                    },
                )
        incorrect = truth != predicted
        if np.any(incorrect):
            embedding_axis.scatter(
                x[incorrect],
                y[incorrect],
                s=48,
                marker="x",
                linewidths=1.2,
                color="black",
            )
        embedding_axis.set_title(
            f"{method}: first registered split {record['requested_seed']} "
            "(black x = error)"
        )
        embedding_axis.set_xlabel("component 1")
        embedding_axis.set_ylabel("component 2")
        embedding_axis.grid(alpha=0.2)

        class_order = [str(label) for label in record["class_order"]]
        matrix = np.asarray(
            [
                [
                    next(
                        row["true_row_fraction"]
                        for row in confusion_rows
                        if row["method"] == method
                        and row["true_population"] == true_label
                        and row["predicted_population"] == predicted_label
                    )
                    for predicted_label in class_order
                ]
                for true_label in class_order
            ],
            dtype=float,
        )
        confusion_axis = axes[1, column]
        image = confusion_axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        confusion_axis.set_xticks(range(len(class_order)), class_order, rotation=45)
        confusion_axis.set_yticks(range(len(class_order)), class_order)
        confusion_axis.set_xlabel("predicted population")
        confusion_axis.set_ylabel("true population")
        confusion_axis.set_title(f"{method}: pooled held-out confusion")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                confusion_axis.text(
                    column_index,
                    row_index,
                    f"{100.0 * matrix[row_index, column_index]:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if matrix[row_index, column_index] > 0.55 else "black",
                )
    if image is not None:
        figure.colorbar(image, ax=axes[1, :].tolist(), shrink=0.75, label="row fraction")
    figure.suptitle(
        "Held-out 1000 Genomes population diagnostics\n"
        "Metadata population groups, not race; fixed-cohort repeated splits",
        fontsize=14,
    )
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_population_report(
    records: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    source_identity: Mapping[str, Any],
    include_figure: bool = True,
) -> Path:
    """Atomically create a new report directory; never modify the source release."""

    destination = output_dir.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        aggregate = aggregate_population_records(records)
        prediction_rows = _prediction_rows(records)
        with (temporary / "reproduced_population_records.json").open(
            "x", encoding="utf-8"
        ) as handle:
            json.dump(
                list(records),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        _write_csv(temporary / "held_out_predictions.csv", prediction_rows)
        _write_csv(temporary / "confusion_matrix.csv", aggregate["confusion_rows"])
        _write_csv(
            temporary / "per_population_metrics.csv",
            aggregate["per_population_rows"],
        )
        _write_csv(temporary / "method_summaries.csv", aggregate["method_summaries"])
        if include_figure:
            _population_figure(
                temporary / "population_diagnostics.png",
                records,
                aggregate,
            )
        artifacts = []
        for path in sorted(temporary.iterdir()):
            artifacts.append(
                {
                    "path": path.name,
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "source_identity": dict(source_identity),
            "runtime_versions": runtime_versions(),
            "evaluation_semantics": {
                "label": "metadata_population_or_ancestry_group",
                "label_is_race": False,
                "prediction_unit": "one_outer_split_held_out_decision",
                "repeated_sample_note": (
                    "a sample may appear in multiple requested-seed test splits; "
                    "rows are not one-unique-prediction-per-person cross-validation"
                ),
                "generalization_scope": "repeated_splits_of_one_fixed_cohort",
                "independent_external_cohort": False,
                "model_selection_scope": "training_partition_only",
                "figure_embedding_policy": "first_registered_seed_not_best_seed",
                "figure_generated": bool(include_figure),
            },
            "method_summaries": aggregate["method_summaries"],
            "per_population_metrics": aggregate["per_population_rows"],
            "artifacts": artifacts,
        }
        report["report_manifest_sha256"] = canonical_sha256(report)
        with (temporary / "report.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def _parse_methods(value: str) -> tuple[str, ...]:
    methods = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not methods or len(set(methods)) != len(methods) or any(
        method not in METHODS for method in methods
    ):
        raise argparse.ArgumentTypeError(
            "methods must be unique comma-separated pca,vp,randomized,robust"
        )
    return methods


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    try:
        for token in value.split(","):
            token = token.strip()
            if ".." in token:
                start_text, stop_text = token.split("..", maxsplit=1)
                start, stop = int(start_text), int(stop_text)
                if start > stop:
                    raise ValueError
                seeds.extend(range(start, stop + 1))
            elif token:
                seeds.append(int(token))
            else:
                raise ValueError
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers or inclusive ranges"
        ) from error
    if not seeds or any(seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique and non-negative")
    return tuple(seeds)


def _decode_delimiter(value: str) -> str:
    decoded = bytes(value, "utf-8").decode("unicode_escape")
    if len(decoded) != 1:
        raise argparse.ArgumentTypeError("delimiter must decode to one character")
    return decoded


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_results", type=Path)
    parser.add_argument("--genotypes", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", type=_parse_methods)
    parser.add_argument("--seeds", type=_parse_seeds)
    parser.add_argument("--metadata-delimiter", type=_decode_delimiter, default="\t")
    parser.add_argument("--genotype-delimiter", type=_decode_delimiter)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--row-id-column")
    parser.add_argument(
        "--trusted-provenance",
        type=Path,
        help=(
            "frozen external Genomics provenance sidecar; formal release-layout "
            "sources require and auto-discover provenance/"
            "genomics_trusted_provenance.json"
        ),
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help=(
            "write verified JSON/CSV only; reproduced records can be rendered "
            "later in an environment with matplotlib"
        ),
    )
    return parser


def _expected_provenance_for_source(
    source_results: Path,
    explicit_anchor: Path | None,
) -> tuple[Mapping[str, Any] | None, Path | None]:
    """Resolve the external trust input for one source result.

    A canonical ``<release>/genomics/results.json`` source cannot silently
    fall back to self-consistency when its sibling trust anchor is absent.
    Synthetic or ad-hoc payloads outside that layout retain the opt-in API.
    """

    anchor = explicit_anchor
    formal_layout = (
        source_results.name == "results.json"
        and source_results.parent.name == "genomics"
    )
    if anchor is None and formal_layout:
        anchor = source_results.parent.parent / GENOMICS_TRUST_ANCHOR_PATH
        if not anchor.is_file():
            raise ValueError(
                "formal Genomics release source is missing its frozen trusted "
                f"provenance sidecar: {anchor}"
            )
    if anchor is None:
        return None, None
    return (
        load_trusted_genomics_provenance(
            anchor,
            result_path=source_results,
        ),
        anchor,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.output_dir.resolve(strict=False).exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output_dir}")
    payload = _strict_json(args.source_results)
    expected_provenance, trust_anchor = _expected_provenance_for_source(
        args.source_results,
        args.trusted_provenance,
    )
    validate_result_payload(
        payload,
        label=str(args.source_results),
        expected_provenance=expected_provenance,
    )
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("source benchmark is missing provenance")
    observed_genotype_sha256 = _file_sha256(args.genotypes)
    observed_metadata_sha256 = _file_sha256(args.metadata)
    if observed_genotype_sha256 != provenance.get("genotype_sha256"):
        raise ValueError("genotype SHA-256 does not match the sealed source result")
    if observed_metadata_sha256 != provenance.get("metadata_sha256"):
        raise ValueError("metadata SHA-256 does not match the sealed source result")
    dataset = load_genotypes(
        args.genotypes,
        orientation=str(provenance.get("input_orientation", "samples-by-snps")),
        delimiter=args.genotype_delimiter,
        has_header=not args.no_header,
        row_id_column=args.row_id_column,
        missing_value=provenance.get("missing_value", -9.0),
    )
    labels = load_population_labels(
        args.metadata,
        dataset.sample_ids,
        sample_id_column=str(
            provenance.get("metadata_sample_id_column", "sample_id")
        ),
        population_column=str(provenance.get("population_column", "population")),
        delimiter=args.metadata_delimiter,
    )
    for observed, expected, label in (
        (array_sha256(dataset.genotypes), provenance.get("normalized_genotypes_sha256"), "normalized genotypes"),
        (array_sha256(dataset.sample_ids), provenance.get("sample_ids_sha256"), "sample IDs"),
        (array_sha256(dataset.snp_ids), provenance.get("snp_ids_sha256"), "SNP IDs"),
        (array_sha256(labels), provenance.get("population_labels_sha256"), "population labels"),
    ):
        if observed != expected:
            raise ValueError(f"{label} do not reproduce the sealed source hash")
    records = reproduce_population_records(
        payload,
        dataset,
        labels,
        methods=args.methods,
        seeds=args.seeds,
        expected_provenance=expected_provenance,
    )
    source_identity = {
        "source_results_path": str(args.source_results.resolve()),
        "source_results_sha256": _file_sha256(args.source_results),
        "source_benchmark_manifest_sha256": payload.get("manifest", {}).get(
            "manifest_sha256"
        ),
        "genotype_sha256": observed_genotype_sha256,
        "metadata_sha256": observed_metadata_sha256,
        "population_column": provenance.get("population_column"),
        "tool_sha256": _file_sha256(Path(__file__)),
    }
    if trust_anchor is not None:
        source_identity["trusted_provenance_path"] = str(trust_anchor.resolve())
        source_identity["trusted_provenance_sha256"] = _file_sha256(trust_anchor)
    destination = write_population_report(
        records,
        output_dir=args.output_dir,
        source_identity=source_identity,
        include_figure=not args.no_figure,
    )
    print(
        json.dumps(
            {
                "output_dir": str(destination),
                "records": len(records),
                "methods": sorted({record["method"] for record in records}),
                "seeds": sorted({record["requested_seed"] for record in records}),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

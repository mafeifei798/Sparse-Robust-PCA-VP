#!/usr/bin/env python3
"""Post-hoc PCA loading-panel diagnostic for five 1000 Genomes groups.

This is deliberately an additive *development diagnostic*, not a replacement
for the frozen Genomics protocol.  It stays on the repository's historical
PCA path: smartpca-like scaling, unsupervised loading-magnitude ranking, a PCA
refit on every frozen panel, and nearest-training-centroid scoring on held-out
rows.  Population labels are never passed to preprocessing, PCA, or loading
ranking.

The expensive full-width PCA is fit once at the largest requested component
count in each outer split.  Its prefixes are reused for the 2/3/4-component
grid.  Finite panels are then refit at the largest component count; the
requested prefix is used for scoring.  The ``all`` panel is an explicit replay
of the full-width fit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray
from scipy import linalg

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    stratified_split,
    transform_smartpca,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

DEFAULT_SEEDS = (2025, 2026, 2027, 2028, 2029)
DEFAULT_COMPONENTS = (2, 3, 4)
DEFAULT_PANELS: tuple[int | None, ...] = (200, 400, 800, None)
EXPECTED_POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
RESULT_FILENAME = "pca_loading_grid_results.json"
METRICS_FILENAME = "pca_loading_grid_metrics.csv"
POPULATION_FILENAME = "pca_loading_grid_population_recall.csv"
COORDINATES_FILENAME = "pca_loading_grid_registered_coordinates.csv"
PC12_FILENAME = "pca_loading_grid_pc12.png"
PC34_FILENAME = "pca_loading_grid_pc34.png"
MANIFEST_FILENAME = "pca_loading_grid_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    METRICS_FILENAME,
    POPULATION_FILENAME,
    COORDINATES_FILENAME,
    PC12_FILENAME,
    PC34_FILENAME,
)
SOURCE_FILE_ROLES = {
    "experiments/genomics_pca_loading_grid.py": "diagnostic_runner",
    "src/rspca/genomics_aim.py": "historical_genomics_preprocessing_and_split",
    "src/rspca/datasets.py": "hashing_and_runtime_identity",
    "src/rspca/__init__.py": "package_initialization_only",
}


@dataclass(frozen=True)
class PCALoadingGridConfig:
    """Fixed grid; ``None`` means every train-retained SNP."""

    seeds: tuple[int, ...] = DEFAULT_SEEDS
    component_grid: tuple[int, ...] = DEFAULT_COMPONENTS
    panel_sizes: tuple[int | None, ...] = DEFAULT_PANELS
    test_fraction: float = 0.2
    loading_score: str = "l1_abs_sum"
    population_design: str = "five_superpopulation"
    analysis_status: str = "post_hoc_development_diagnostic"


@dataclass(frozen=True)
class GenotypeInput:
    genotypes: FloatArray
    sample_ids: StringArray
    snp_ids: StringArray
    chromosomes: StringArray
    population_labels: StringArray


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_array_sha256(array: np.ndarray) -> str:
    """Match the preparation receipt's NumPy ``.npy`` byte hash."""

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return _bytes_sha256(buffer.getvalue())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_bytes(payload: Any) -> bytes:
    _reject_nonfinite(payload)
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )


def _reject_nonfinite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")


def _validate_config(config: PCALoadingGridConfig) -> None:
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(isinstance(seed, bool) or int(seed) < 0 for seed in config.seeds):
        raise ValueError("seeds must be non-negative integers")
    if config.component_grid != (2, 3, 4):
        raise ValueError("component_grid is fixed to (2, 3, 4)")
    finite_panels = tuple(panel for panel in config.panel_sizes if panel is not None)
    if (
        config.panel_sizes.count(None) != 1
        or tuple(sorted(finite_panels)) != finite_panels
        or len(set(finite_panels)) != len(finite_panels)
        or any(isinstance(panel, bool) or int(panel) < max(config.component_grid) for panel in finite_panels)
    ):
        raise ValueError("panel_sizes must be increasing positive sizes plus one all")
    if config.panel_sizes[-1] is not None:
        raise ValueError("the all-SNP panel must be last")
    if not 0.0 < float(config.test_fraction) < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if config.loading_score != "l1_abs_sum":
        raise ValueError("this diagnostic fixes loading_score to l1_abs_sum")
    if config.population_design != "five_superpopulation":
        raise ValueError("this diagnostic is restricted to five_superpopulation")
    if config.analysis_status != "post_hoc_development_diagnostic":
        raise ValueError("analysis_status must remain post_hoc_development_diagnostic")


def _validate_input(data: GenotypeInput) -> None:
    matrix = np.asarray(data.genotypes, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 5:
        raise ValueError("genotypes must be a samples x SNPs matrix")
    arrays = {
        "sample_ids": (data.sample_ids, matrix.shape[0]),
        "population_labels": (data.population_labels, matrix.shape[0]),
        "snp_ids": (data.snp_ids, matrix.shape[1]),
        "chromosomes": (data.chromosomes, matrix.shape[1]),
    }
    for name, (values, expected) in arrays.items():
        array = np.asarray(values, dtype=str)
        if array.shape != (expected,) or np.any(np.char.strip(array) == ""):
            raise ValueError(f"{name} must contain {expected} non-empty entries")
    if np.unique(data.sample_ids).size != matrix.shape[0]:
        raise ValueError("sample_ids must be unique")
    if np.unique(data.snp_ids).size != matrix.shape[1]:
        raise ValueError("snp_ids must be unique")
    if np.any(np.isinf(matrix)):
        raise ValueError("genotypes may contain NaN missingness but not infinity")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0 or np.any((finite < 0.0) | (finite > 2.0)):
        raise ValueError("finite genotype dosages must lie in [0, 2]")
    labels = np.asarray(data.population_labels, dtype=str)
    if tuple(sorted(np.unique(labels).tolist())) != EXPECTED_POPULATIONS:
        raise ValueError(f"population labels must be exactly {EXPECTED_POPULATIONS}")
    for population in EXPECTED_POPULATIONS:
        if np.count_nonzero(labels == population) < 2:
            raise ValueError(f"population {population} requires at least two samples")


def load_genotype_npz(path: Path) -> GenotypeInput:
    """Load the prepared NPZ without permitting pickle-backed arrays."""

    with np.load(path, allow_pickle=False) as archive:
        required = {
            "genotypes",
            "sample_ids",
            "snp_ids",
            "chromosomes",
            "population_labels",
        }
        if set(archive.files) != required:
            raise ValueError(
                f"NPZ keys must be exactly {sorted(required)}; got {sorted(archive.files)}"
            )
        data = GenotypeInput(
            genotypes=np.asarray(archive["genotypes"], dtype=float),
            sample_ids=np.asarray(archive["sample_ids"], dtype=str),
            snp_ids=np.asarray(archive["snp_ids"], dtype=str),
            chromosomes=np.asarray(archive["chromosomes"], dtype=str),
            population_labels=np.asarray(archive["population_labels"], dtype=str),
        )
    _validate_input(data)
    return data


def validate_external_source_manifest(
    path: Path,
    genotype_npz: Path,
    data: GenotypeInput,
    *,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Bind the diagnostic to the prepared LD-pruned five-group NPZ."""

    _regular_file(path, role="external source manifest")
    _regular_file(genotype_npz, role="genotype NPZ")
    actual_manifest_sha256 = _file_sha256(path)
    if not _is_sha256(expected_manifest_sha256):
        raise ValueError("expected source-manifest SHA-256 is invalid")
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("external source manifest does not match its trusted anchor")
    manifest = _strict_json_load(path)
    if not isinstance(manifest, Mapping):
        raise ValueError("external source manifest must be a JSON object")
    dataset = manifest.get("dataset")
    output = manifest.get("output")
    if not isinstance(dataset, Mapping) or not isinstance(output, Mapping):
        raise ValueError("external source manifest is missing dataset/output receipts")
    if (
        manifest.get("schema_version")
        != "rspca-1kg-phase3-v5b-chr20-22-preparation-v1"
        or dataset.get("population_design") != "five_superpopulation"
        or dataset.get("orientation") != "samples_by_snps"
        or dataset.get("ld_pruned") is not True
        or dataset.get("input_scope") != "legacy_chr20_22"
        or dataset.get("population_label_is_race") is not False
    ):
        raise ValueError("external source manifest does not describe the required input")
    expected_file_hash = _file_sha256(genotype_npz)
    expected_shape = [int(data.genotypes.shape[0]), int(data.genotypes.shape[1])]
    if output.get("sha256") != expected_file_hash or output.get("shape") != expected_shape:
        raise ValueError("external source manifest does not bind the supplied NPZ")
    if int(output.get("bytes", -1)) != genotype_npz.stat().st_size:
        raise ValueError("external source manifest NPZ size mismatch")
    if output.get("npz_keys") != [
        "chromosomes",
        "genotypes",
        "population_labels",
        "sample_ids",
        "snp_ids",
    ]:
        raise ValueError("external source manifest NPZ-key receipt mismatch")
    array_receipts = output.get("array_sha256")
    if not isinstance(array_receipts, Mapping) or set(array_receipts) != {
        "chromosomes",
        "genotypes",
        "population_labels",
        "sample_ids",
        "snp_ids",
    }:
        raise ValueError("external source manifest array receipts are incomplete")
    with np.load(genotype_npz, allow_pickle=False) as archive:
        for key, expected_hash in array_receipts.items():
            if not _is_sha256(expected_hash) or _npy_array_sha256(archive[key]) != expected_hash:
                raise ValueError(f"external source manifest array hash mismatch: {key}")
    preparation = manifest.get("preparation_tool")
    variant_processing = manifest.get("variant_processing")
    plink2 = manifest.get("plink2")
    downstream_boundary = manifest.get("downstream_label_boundary")
    if not all(
        isinstance(value, Mapping)
        for value in (preparation, variant_processing, plink2, downstream_boundary)
    ):
        raise ValueError("external source manifest lacks preparation receipts")
    configuration = preparation.get("configuration")
    if (
        not _is_sha256(preparation.get("sha256"))
        or not isinstance(configuration, Mapping)
        or configuration.get("population_design") != "five_superpopulation"
        or configuration.get("ld_window") != 50
        or configuration.get("ld_step") != 5
        or configuration.get("ld_r2") != 0.2
        or configuration.get("maf") != 0.05
        or configuration.get("geno") != 0.05
    ):
        raise ValueError("external source manifest preparation-tool receipt mismatch")
    ld_pruning = variant_processing.get("ld_pruning")
    per_chromosome = variant_processing.get("per_chromosome")
    if (
        variant_processing.get("padding_with_unpruned_variants") is not False
        or variant_processing.get("population_labels_used_for_variant_qc_or_ld") is not False
        or not isinstance(ld_pruning, Mapping)
        or ld_pruning.get("method") != "plink2_indep_pairwise"
        or ld_pruning.get("window") != 50
        or ld_pruning.get("step_variants") != 5
        or ld_pruning.get("r2_threshold") != 0.2
        or ld_pruning.get("indep_order") != 2
        or not isinstance(per_chromosome, Mapping)
        or set(per_chromosome) != {"20", "21", "22"}
    ):
        raise ValueError("external source manifest LD-pruning receipt mismatch")
    for chromosome, chromosome_receipt in per_chromosome.items():
        commands = chromosome_receipt.get("commands")
        if not isinstance(commands, list) or len(commands) < 3:
            raise ValueError(f"chromosome {chromosome} lacks PLINK command receipts")
        roles = {command.get("role") for command in commands}
        required_roles = {
            f"chr{chromosome}_import_qc",
            f"chr{chromosome}_ld_prune",
            f"chr{chromosome}_export_pruned",
        }
        if not required_roles.issubset(roles):
            raise ValueError(f"chromosome {chromosome} PLINK roles are incomplete")
        for command in commands:
            if (
                command.get("returncode") != 0
                or not _is_sha256(command.get("stdout_sha256"))
                or not _is_sha256(command.get("stderr_sha256"))
                or not isinstance(command.get("argv"), list)
            ):
                raise ValueError(f"chromosome {chromosome} has an invalid command receipt")
        prune = next(
            command
            for command in commands
            if command.get("role") == f"chr{chromosome}_ld_prune"
        )
        export = next(
            command
            for command in commands
            if command.get("role") == f"chr{chromosome}_export_pruned"
        )
        if "--indep-pairwise" not in prune["argv"] or "--extract" not in export["argv"]:
            raise ValueError(f"chromosome {chromosome} commands do not enact LD pruning")
    if (
        not _is_sha256(plink2.get("sha256"))
        or not isinstance(plink2.get("version_command"), Mapping)
        or plink2["version_command"].get("returncode") != 0
        or downstream_boundary.get("panel_selection_used_population_labels") is not False
    ):
        raise ValueError("external source manifest PLINK/label-boundary receipt mismatch")
    return {
        "external_source_manifest_sha256": actual_manifest_sha256,
        "expected_external_source_manifest_sha256": expected_manifest_sha256,
        "external_source_manifest_anchor_verified": True,
        "external_source_manifest_bytes": int(path.stat().st_size),
        "genotype_npz_sha256": expected_file_hash,
        "genotype_npz_bytes": int(genotype_npz.stat().st_size),
        "manifest_population_design": dataset["population_design"],
        "manifest_input_scope": dataset["input_scope"],
        "manifest_ld_pruned_receipts_verified": True,
    }


def fit_top_pca(matrix: FloatArray, n_components: int) -> tuple[FloatArray, FloatArray]:
    """Return exact leading PCA loadings through the smaller Gram matrix.

    This is algebraically the leading part of the SVD used by the historical
    notebooks, while avoiding a full 800-by-27k SVD when only four components
    are requested.  Signs are canonicalized for deterministic figures.
    """

    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or min(values.shape) < n_components:
        raise ValueError("matrix dimensions must be at least n_components")
    if not np.isfinite(values).all():
        raise ValueError("PCA matrix must be finite")
    if isinstance(n_components, bool) or int(n_components) < 1:
        raise ValueError("n_components must be positive")
    n_components = int(n_components)
    rows, columns = values.shape
    if rows <= columns:
        gram = values @ values.T
        eigenvalues, left = linalg.eigh(
            gram,
            subset_by_index=(rows - n_components, rows - 1),
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        left = left[:, order]
        singular_values = np.sqrt(eigenvalues)
        threshold = np.finfo(float).eps * max(values.shape) * singular_values[0]
        if np.any(singular_values <= threshold):
            raise ValueError("training matrix has insufficient numerical PCA rank")
        loadings = values.T @ left
        loadings /= singular_values[None, :]
    else:
        gram = values.T @ values
        eigenvalues, loadings = linalg.eigh(
            gram,
            subset_by_index=(columns - n_components, columns - 1),
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        loadings = loadings[:, order]
        singular_values = np.sqrt(eigenvalues)
        threshold = np.finfo(float).eps * max(values.shape) * singular_values[0]
        if np.any(singular_values <= threshold):
            raise ValueError("training matrix has insufficient numerical PCA rank")
    loadings = np.asarray(loadings, dtype=float)
    norms = np.linalg.norm(loadings, axis=0)
    if np.any(norms <= 0.0):
        raise RuntimeError("PCA returned a zero loading")
    loadings /= norms[None, :]
    for component in range(loadings.shape[1]):
        pivot = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[pivot, component] < 0.0:
            loadings[:, component] *= -1.0
    if not np.allclose(loadings.T @ loadings, np.eye(n_components), atol=2e-7):
        raise RuntimeError("PCA loadings failed the orthonormality check")
    return np.ascontiguousarray(loadings), np.asarray(singular_values, dtype=float)


def rank_by_pca_loadings(loadings: FloatArray, n_components: int) -> tuple[IntArray, FloatArray]:
    """Rank SNPs using training PCA loadings only; this function has no labels."""

    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("loadings must be a finite two-dimensional matrix")
    if not 1 <= int(n_components) <= matrix.shape[1]:
        raise ValueError("n_components exceeds available loading columns")
    importance = np.sum(np.abs(matrix[:, : int(n_components)]), axis=1)
    if not np.any(importance > 0.0):
        raise ValueError("all loading scores are zero")
    indices = np.arange(matrix.shape[0], dtype=np.int64)
    ranking = np.lexsort((indices, -importance)).astype(np.int64)
    return ranking, np.asarray(importance, dtype=float)


def _nearest_centroid(
    train_scores: FloatArray,
    train_labels: StringArray,
    test_scores: FloatArray,
) -> tuple[StringArray, StringArray, FloatArray]:
    classes = np.unique(np.asarray(train_labels, dtype=str))
    centroids = np.stack(
        [train_scores[train_labels == population].mean(axis=0) for population in classes],
        axis=0,
    )
    distances = np.sum(
        (test_scores[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return (
        np.asarray(classes[np.argmin(distances, axis=1)], dtype=str),
        np.asarray(classes, dtype=str),
        np.asarray(centroids, dtype=float),
    )


def _classification_metrics(
    truth: StringArray,
    predictions: StringArray,
    classes: StringArray,
) -> dict[str, Any]:
    positions = {population: index for index, population in enumerate(classes.tolist())}
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    for expected, observed in zip(truth.tolist(), predictions.tolist(), strict=True):
        confusion[positions[expected], positions[observed]] += 1
    per_population: list[dict[str, Any]] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for index, population in enumerate(classes.tolist()):
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        recall = float(true_positive / support)
        precision = float(true_positive / predicted_count) if predicted_count else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
        recalls.append(recall)
        f1s.append(f1)
        per_population.append(
            {
                "population": population,
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(np.trace(confusion) / confusion.sum()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "class_order": classes.tolist(),
        "confusion_matrix_counts": confusion.tolist(),
        "per_population_metrics": per_population,
    }


def _panel_label(panel_size: int | None) -> str:
    return "all" if panel_size is None else str(int(panel_size))


def _record(
    *,
    seed: int,
    split_hashes: Mapping[str, str],
    evaluation: str,
    n_components: int,
    panel_size: int | None,
    selected_original: IntArray,
    retained_original: IntArray,
    loadings: FloatArray,
    train_scores: FloatArray,
    test_scores: FloatArray,
    train_labels: StringArray,
    test_labels: StringArray,
    reused_full_width_fit: bool,
) -> tuple[dict[str, Any], StringArray, FloatArray]:
    predictions, classes, centroids = _nearest_centroid(
        train_scores[:, :n_components],
        train_labels,
        test_scores[:, :n_components],
    )
    metrics = _classification_metrics(test_labels, predictions, classes)
    record = {
        "seed": int(seed),
        "evaluation": evaluation,
        "n_components": int(n_components),
        "panel_label": "full_width" if evaluation == "full_width" else _panel_label(panel_size),
        "requested_panel_size": None if evaluation == "full_width" else panel_size,
        "available_train_retained_snps": int(retained_original.size),
        "selected_snp_count": int(selected_original.size),
        "selected_original_indices_sha256": array_sha256(selected_original),
        "selected_snp_set_is_loading_only": True,
        "test_used_for_preprocessing_pca_or_ranking": False,
        "population_labels_used_for_panel_ranking": False,
        "preprocessing_scope": "outer_train_only",
        "pca_fit_scope": "outer_train_only",
        "panel_refit_scope": "outer_train_only",
        "classifier_fit_scope": "outer_train_centroids_after_panel_freeze",
        "solver": "closed_form_pca_smaller_gram_eigh",
        "solver_converged": True,
        "solver_n_iter": 1,
        "full_width_fit_reused": bool(reused_full_width_fit),
        "loadings_sha256": array_sha256(loadings[:, :n_components]),
        "test_predictions_sha256": array_sha256(predictions),
        **dict(split_hashes),
        **metrics,
    }
    return record, predictions, centroids


def evaluate_loading_grid(
    data: GenotypeInput,
    config: PCALoadingGridConfig,
    *,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate every fixed grid cell without selecting a winner."""

    _validate_input(data)
    _validate_config(config)
    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    max_components = max(config.component_grid)
    records: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []

    for seed in config.seeds:
        print(
            json.dumps({"event": "pca_loading_grid_seed_start", "seed": seed}),
            file=sys.stderr,
            flush=True,
        )
        train, test = stratified_split(labels, config.test_fraction, seed)
        preprocessor = fit_smartpca_preprocessor(raw[train])
        train_matrix = transform_smartpca(raw[train], preprocessor)
        test_matrix = transform_smartpca(raw[test], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        if retained_original.size < max(
            panel for panel in config.panel_sizes if panel is not None
        ):
            raise ValueError("a requested panel exceeds the train-retained SNP count")
        split_hashes = {
            "train_indices_sha256": array_sha256(train),
            "test_indices_sha256": array_sha256(test),
            "train_sample_ids_sha256": array_sha256(sample_ids[train]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test]),
            "test_labels_sha256": array_sha256(labels[test]),
        }
        split_receipts.append(
            {
                "seed": int(seed),
                **split_hashes,
                "retained_original_indices_sha256": array_sha256(retained_original),
                "retained_snp_ids_sha256": array_sha256(data.snp_ids[retained_original]),
                "retained_snps": int(retained_original.size),
                "preprocessor_mean_sha256": array_sha256(preprocessor.mean_dosage),
                "preprocessor_scale_sha256": array_sha256(preprocessor.scale),
            }
        )

        # This is the only full-width PCA fit in this outer split.
        full_loadings, _ = fit_top_pca(train_matrix, max_components)
        full_train_scores = train_matrix @ full_loadings
        full_test_scores = test_matrix @ full_loadings
        panel_fit_cache: dict[
            str, tuple[IntArray, FloatArray, FloatArray, FloatArray]
        ] = {}

        for n_components in config.component_grid:
            ranking_retained, importance = rank_by_pca_loadings(
                full_loadings, n_components
            )
            ranked_original = retained_original[ranking_retained]
            rankings.append(
                {
                    "seed": int(seed),
                    "n_components": int(n_components),
                    "available_train_retained_snps": int(retained_original.size),
                    "ordered_original_indices": ranked_original.tolist(),
                    "ordered_original_indices_sha256": array_sha256(ranked_original),
                    "loading_importance_sha256": array_sha256(importance),
                    "full_width_loading_prefix_sha256": array_sha256(
                        full_loadings[:, :n_components]
                    ),
                    "selection_rule": "descending_sum_abs_training_pca_loadings",
                    "tie_break": "retained_column_index_ascending",
                    "population_labels_used": False,
                    "test_rows_used": False,
                }
            )

            full_record, _, _ = _record(
                seed=seed,
                split_hashes=split_hashes,
                evaluation="full_width",
                n_components=n_components,
                panel_size=None,
                selected_original=np.sort(retained_original),
                retained_original=retained_original,
                loadings=full_loadings,
                train_scores=full_train_scores,
                test_scores=full_test_scores,
                train_labels=labels[train],
                test_labels=labels[test],
                reused_full_width_fit=True,
            )
            records.append(full_record)

            for panel_size in config.panel_sizes:
                if panel_size is None:
                    selected_retained = np.arange(retained_original.size, dtype=np.int64)
                    selected_original = np.sort(retained_original)
                    panel_loadings = full_loadings
                    panel_train_scores = full_train_scores
                    panel_test_scores = full_test_scores
                    reused = True
                else:
                    selected_retained = np.sort(
                        ranking_retained[: int(panel_size)]
                    ).astype(np.int64)
                    selected_original = np.sort(
                        retained_original[selected_retained]
                    ).astype(np.int64)
                    selected_hash = array_sha256(selected_original)
                    # A refit depends only on the selected columns, not on how
                    # many leading components will later be used for scoring.
                    cache_key = selected_hash
                    if cache_key not in panel_fit_cache:
                        panel_train = train_matrix[:, selected_retained]
                        panel_test = test_matrix[:, selected_retained]
                        panel_loadings, _ = fit_top_pca(
                            panel_train, max_components
                        )
                        panel_fit_cache[cache_key] = (
                            selected_original,
                            panel_loadings,
                            panel_train @ panel_loadings,
                            panel_test @ panel_loadings,
                        )
                    (
                        cached_original,
                        panel_loadings,
                        panel_train_scores,
                        panel_test_scores,
                    ) = panel_fit_cache[cache_key]
                    if not np.array_equal(cached_original, selected_original):
                        raise RuntimeError("panel-fit cache identity mismatch")
                    reused = False

                panel_record, predictions, _ = _record(
                    seed=seed,
                    split_hashes=split_hashes,
                    evaluation="loading_panel",
                    n_components=n_components,
                    panel_size=panel_size,
                    selected_original=selected_original,
                    retained_original=retained_original,
                    loadings=panel_loadings,
                    train_scores=panel_train_scores,
                    test_scores=panel_test_scores,
                    train_labels=labels[train],
                    test_labels=labels[test],
                    reused_full_width_fit=reused,
                )
                records.append(panel_record)

                if seed == config.seeds[0] and n_components in (2, max_components):
                    panel_label = _panel_label(panel_size)
                    for local_index, sample_index in enumerate(test.tolist()):
                        coordinates.append(
                            {
                                "row_type": "held_out_sample",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": panel_label,
                                "requested_panel_size": panel_size,
                                "selected_snp_count": int(selected_original.size),
                                "sample_index": int(sample_index),
                                "sample_id": str(sample_ids[sample_index]),
                                "population": str(labels[sample_index]),
                                "truth": str(labels[sample_index]),
                                "prediction": str(predictions[local_index]),
                                "correct": bool(predictions[local_index] == labels[sample_index]),
                                **{
                                    f"component_{component + 1}": float(
                                        panel_test_scores[local_index, component]
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )
                    classes = np.unique(labels[train])
                    plotted_centroids = np.stack(
                        [
                            panel_train_scores[labels[train] == population].mean(axis=0)
                            for population in classes
                        ],
                        axis=0,
                    )
                    for class_index, population in enumerate(classes.tolist()):
                        coordinates.append(
                            {
                                "row_type": "train_centroid",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": panel_label,
                                "requested_panel_size": panel_size,
                                "selected_snp_count": int(selected_original.size),
                                "sample_index": None,
                                "sample_id": None,
                                "population": population,
                                "truth": None,
                                "prediction": None,
                                "correct": None,
                                **{
                                    f"component_{component + 1}": float(
                                        plotted_centroids[class_index, component]
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )

        print(
            json.dumps(
                {
                    "event": "pca_loading_grid_seed_complete",
                    "seed": seed,
                    "full_width_pca_fits": 1,
                    "finite_panel_pca_fits": len(panel_fit_cache),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    summaries: list[dict[str, Any]] = []
    population_summaries: list[dict[str, Any]] = []
    grid_order = [("full_width", None)] + [
        ("loading_panel", panel) for panel in config.panel_sizes
    ]
    for n_components in config.component_grid:
        full_rows = [
            row
            for row in records
            if row["evaluation"] == "full_width"
            and row["n_components"] == n_components
        ]
        full_by_seed = {int(row["seed"]): row for row in full_rows}
        for evaluation, panel_size in grid_order:
            panel_label = "full_width" if evaluation == "full_width" else _panel_label(panel_size)
            rows = [
                row
                for row in records
                if row["evaluation"] == evaluation
                and row["n_components"] == n_components
                and row["panel_label"] == panel_label
            ]
            if len(rows) != len(config.seeds):
                raise RuntimeError("incomplete fixed grid")
            balanced = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
            accuracies = np.asarray([row["accuracy"] for row in rows], dtype=float)
            macro_f1 = np.asarray([row["macro_f1"] for row in rows], dtype=float)
            full_delta = np.asarray(
                [
                    row["balanced_accuracy"]
                    - full_by_seed[int(row["seed"])]["balanced_accuracy"]
                    for row in rows
                ],
                dtype=float,
            )
            summaries.append(
                {
                    "evaluation": evaluation,
                    "n_components": int(n_components),
                    "panel_label": panel_label,
                    "requested_panel_size": None if evaluation == "full_width" else panel_size,
                    "seeds": len(rows),
                    "minimum_selected_snp_count": int(min(row["selected_snp_count"] for row in rows)),
                    "maximum_selected_snp_count": int(max(row["selected_snp_count"] for row in rows)),
                    "mean_accuracy": float(np.mean(accuracies)),
                    "mean_balanced_accuracy": float(np.mean(balanced)),
                    "sample_std_balanced_accuracy": float(np.std(balanced, ddof=1)) if len(rows) > 1 else 0.0,
                    "minimum_balanced_accuracy": float(np.min(balanced)),
                    "maximum_balanced_accuracy": float(np.max(balanced)),
                    "mean_macro_f1": float(np.mean(macro_f1)),
                    "mean_paired_balanced_accuracy_delta_vs_full_width": float(np.mean(full_delta)),
                    "qualification_status": "closed_form_pca_all_splits",
                }
            )
            for population in EXPECTED_POPULATIONS:
                metrics = [
                    next(
                        item
                        for item in row["per_population_metrics"]
                        if item["population"] == population
                    )
                    for row in rows
                ]
                recalls = np.asarray([item["recall"] for item in metrics], dtype=float)
                population_summaries.append(
                    {
                        "evaluation": evaluation,
                        "n_components": int(n_components),
                        "panel_label": panel_label,
                        "requested_panel_size": None if evaluation == "full_width" else panel_size,
                        "population": population,
                        "seeds": len(rows),
                        "pooled_support": int(sum(item["support"] for item in metrics)),
                        "pooled_true_positive": int(sum(item["true_positive"] for item in metrics)),
                        "mean_recall": float(np.mean(recalls)),
                        "minimum_recall": float(np.min(recalls)),
                        "maximum_recall": float(np.max(recalls)),
                    }
                )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_policy": {
            "status": "post_hoc_development_diagnostic",
            "confirmatory": False,
            "best_configuration_selected": False,
            "headline_configuration_selected": False,
            "all_grid_cells_reported": True,
            "population_labels_are_race": False,
            "population_labels_used_for_split": True,
            "population_labels_used_for_training_centroids": True,
            "population_labels_used_for_held_out_scoring": True,
            "population_labels_used_for_preprocessing_pca_or_loading_ranking": False,
            "test_rows_used_for_preprocessing_pca_or_loading_ranking": False,
            "selection_rule": "training_only_unsupervised_pca_loading_l1",
            "interpretation": "diagnose whether more PCs or a larger loading-only panel reduces visible AMR/SAS/EUR overlap",
        },
        "config": asdict(config),
        "input_identity": dict(input_identity),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "split_receipts": split_receipts,
        "ranking_records": rankings,
        "records": records,
        "grid_summaries": summaries,
        "population_summaries": population_summaries,
        "registered_seed_policy": {
            "seed": int(config.seeds[0]),
            "selection": "first_configured_seed_never_best_seed",
            "plotted_component_pairs": {
                "pc1_pc2": 2,
                "pc3_pc4": int(max_components),
            },
            "panels": [_panel_label(panel) for panel in config.panel_sizes],
        },
        "registered_coordinates": coordinates,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_result_payload(payload)
    return payload


def _derived_metrics(record: Mapping[str, Any]) -> tuple[float, float, float]:
    confusion = np.asarray(record["confusion_matrix_counts"], dtype=np.int64)
    supports = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    diagonal = np.diag(confusion)
    recalls = diagonal / supports
    precision = np.divide(
        diagonal,
        predicted,
        out=np.zeros(diagonal.size, dtype=float),
        where=predicted > 0,
    )
    f1 = np.divide(
        2.0 * precision * recalls,
        precision + recalls,
        out=np.zeros(diagonal.size, dtype=float),
        where=(precision + recalls) > 0,
    )
    return (
        float(diagonal.sum() / confusion.sum()),
        float(np.mean(recalls)),
        float(np.mean(f1)),
    )


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on the fixed grid, leakage declarations, and metrics."""

    required_top = {
        "schema_version",
        "analysis_policy",
        "config",
        "input_identity",
        "source_identity",
        "runtime_versions",
        "split_receipts",
        "ranking_records",
        "records",
        "grid_summaries",
        "population_summaries",
        "registered_seed_policy",
        "registered_coordinates",
        "analysis_sha256",
    }
    if set(payload) != required_top or payload["schema_version"] != 1:
        raise ValueError("result payload has an unexpected top-level schema")
    _reject_nonfinite(payload)
    policy = payload["analysis_policy"]
    if (
        policy.get("status") != "post_hoc_development_diagnostic"
        or policy.get("confirmatory") is not False
        or policy.get("best_configuration_selected") is not False
        or policy.get("headline_configuration_selected") is not False
        or policy.get("all_grid_cells_reported") is not True
        or policy.get("population_labels_used_for_preprocessing_pca_or_loading_ranking") is not False
        or policy.get("test_rows_used_for_preprocessing_pca_or_loading_ranking") is not False
        or policy.get("selection_rule") != "training_only_unsupervised_pca_loading_l1"
    ):
        raise ValueError("analysis-policy boundary is invalid")
    config = PCALoadingGridConfig(
        seeds=tuple(int(seed) for seed in payload["config"]["seeds"]),
        component_grid=tuple(int(value) for value in payload["config"]["component_grid"]),
        panel_sizes=tuple(
            None if value is None else int(value)
            for value in payload["config"]["panel_sizes"]
        ),
        test_fraction=float(payload["config"]["test_fraction"]),
        loading_score=str(payload["config"]["loading_score"]),
        population_design=str(payload["config"]["population_design"]),
        analysis_status=str(payload["config"]["analysis_status"]),
    )
    if set(payload["config"]) != set(asdict(config)):
        raise ValueError("config contains unknown or missing fields")
    _validate_config(config)
    receipt_fields = {
        "seed",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "retained_original_indices_sha256",
        "retained_snp_ids_sha256",
        "retained_snps",
        "preprocessor_mean_sha256",
        "preprocessor_scale_sha256",
    }
    receipt_map: dict[int, Mapping[str, Any]] = {}
    for receipt in payload["split_receipts"]:
        if set(receipt) != receipt_fields:
            raise ValueError("split receipt schema mismatch")
        seed = int(receipt["seed"])
        if seed in receipt_map:
            raise ValueError("duplicate split receipt")
        if int(receipt["retained_snps"]) < max(
            panel for panel in config.panel_sizes if panel is not None
        ):
            raise ValueError("split receipt has insufficient retained SNPs")
        for field in receipt_fields - {"seed", "retained_snps"}:
            if not _is_sha256(receipt[field]):
                raise ValueError(f"split receipt contains invalid {field}")
        receipt_map[seed] = receipt
    if set(receipt_map) != set(config.seeds):
        raise ValueError("split receipts do not exactly cover configured seeds")
    rankings = payload["ranking_records"]
    expected_ranking_keys = {
        (seed, components)
        for seed in config.seeds
        for components in config.component_grid
    }
    ranking_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for ranking in rankings:
        key = (int(ranking["seed"]), int(ranking["n_components"]))
        if key in ranking_map:
            raise ValueError("duplicate ranking record")
        if (
            ranking.get("population_labels_used") is not False
            or ranking.get("test_rows_used") is not False
            or ranking.get("selection_rule") != "descending_sum_abs_training_pca_loadings"
            or ranking.get("tie_break") != "retained_column_index_ascending"
        ):
            raise ValueError("ranking boundary is invalid")
        ordered = np.asarray(ranking["ordered_original_indices"], dtype=np.int64)
        receipt = receipt_map[key[0]]
        if (
            ordered.ndim != 1
            or ordered.size != int(ranking["available_train_retained_snps"])
            or ordered.size != int(receipt["retained_snps"])
            or np.unique(ordered).size != ordered.size
            or np.any(ordered < 0)
            or array_sha256(ordered) != ranking["ordered_original_indices_sha256"]
            or array_sha256(np.sort(ordered))
            != receipt["retained_original_indices_sha256"]
        ):
            raise ValueError("ranked original-index permutation is invalid")
        for field in (
            "loading_importance_sha256",
            "full_width_loading_prefix_sha256",
        ):
            if not _is_sha256(ranking[field]):
                raise ValueError(f"invalid {field}")
        ranking_map[key] = ranking
    if set(ranking_map) != expected_ranking_keys:
        raise ValueError("ranking records do not cover the fixed component grid")

    grid = [("full_width", "full_width", None)] + [
        ("loading_panel", _panel_label(panel), panel) for panel in config.panel_sizes
    ]
    expected_record_keys = {
        (seed, components, evaluation, label)
        for seed in config.seeds
        for components in config.component_grid
        for evaluation, label, _ in grid
    }
    record_map: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    support_signature_by_seed: dict[int, tuple[int, ...]] = {}
    for record in payload["records"]:
        key = (
            int(record["seed"]),
            int(record["n_components"]),
            str(record["evaluation"]),
            str(record["panel_label"]),
        )
        if key in record_map:
            raise ValueError("duplicate grid record")
        if (
            record.get("population_labels_used_for_panel_ranking") is not False
            or record.get("test_used_for_preprocessing_pca_or_ranking") is not False
            or record.get("selected_snp_set_is_loading_only") is not True
            or record.get("solver_converged") is not True
            or record.get("solver") != "closed_form_pca_smaller_gram_eigh"
            or record.get("preprocessing_scope") != "outer_train_only"
            or record.get("pca_fit_scope") != "outer_train_only"
            or record.get("panel_refit_scope") != "outer_train_only"
            or record.get("classifier_fit_scope")
            != "outer_train_centroids_after_panel_freeze"
        ):
            raise ValueError("record violates the fitting boundary")
        receipt = receipt_map[key[0]]
        for field in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if record.get(field) != receipt[field]:
                raise ValueError(f"record {field} disagrees with split receipt")
        ranking = ranking_map[(key[0], key[1])]
        ordered = np.asarray(ranking["ordered_original_indices"], dtype=np.int64)
        if key[2] == "full_width" or key[3] == "all":
            expected_selected = np.sort(ordered)
        else:
            requested = int(record["requested_panel_size"])
            if key[3] != str(requested):
                raise ValueError("panel label and requested size disagree")
            expected_selected = np.sort(ordered[:requested])
        if (
            int(record["selected_snp_count"]) != expected_selected.size
            or record["selected_original_indices_sha256"]
            != array_sha256(expected_selected)
        ):
            raise ValueError("selected panel does not match loading ranking")
        if record["class_order"] != list(EXPECTED_POPULATIONS):
            raise ValueError("class order is not the fixed five-population order")
        raw_confusion = record.get("confusion_matrix_counts")
        if (
            not isinstance(raw_confusion, list)
            or len(raw_confusion) != len(EXPECTED_POPULATIONS)
            or any(
                not isinstance(row, list)
                or len(row) != len(EXPECTED_POPULATIONS)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    for value in row
                )
                for row in raw_confusion
            )
        ):
            raise ValueError("confusion counts must be a non-negative integer 5x5 matrix")
        confusion = np.asarray(raw_confusion, dtype=np.int64)
        support_signature = tuple(int(value) for value in confusion.sum(axis=1))
        if any(value <= 0 for value in support_signature):
            raise ValueError("every population must have positive held-out support")
        prior_signature = support_signature_by_seed.setdefault(key[0], support_signature)
        if prior_signature != support_signature:
            raise ValueError("held-out supports disagree within an outer split")
        for field in (
            "selected_original_indices_sha256",
            "loadings_sha256",
            "test_predictions_sha256",
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if not _is_sha256(record.get(field)):
                raise ValueError(f"record contains invalid {field}")
        derived = _derived_metrics(record)
        reported = (
            float(record["accuracy"]),
            float(record["balanced_accuracy"]),
            float(record["macro_f1"]),
        )
        if not np.allclose(derived, reported, atol=1e-14, rtol=0.0):
            raise ValueError("record metrics disagree with confusion counts")
        populations = record["per_population_metrics"]
        if [row["population"] for row in populations] != list(EXPECTED_POPULATIONS):
            raise ValueError("per-population rows are incomplete or reordered")
        for population_index, population_row in enumerate(populations):
            support = int(confusion[population_index].sum())
            predicted_count = int(confusion[:, population_index].sum())
            true_positive = int(confusion[population_index, population_index])
            recall = float(true_positive / support)
            precision = (
                float(true_positive / predicted_count) if predicted_count else 0.0
            )
            f1 = (
                float(2.0 * precision * recall / (precision + recall))
                if precision + recall
                else 0.0
            )
            if (
                population_row.get("support") != support
                or population_row.get("predicted_count") != predicted_count
                or population_row.get("true_positive") != true_positive
                or not math.isclose(float(population_row.get("recall")), recall, abs_tol=1e-14)
                or not math.isclose(float(population_row.get("precision")), precision, abs_tol=1e-14)
                or not math.isclose(float(population_row.get("f1")), f1, abs_tol=1e-14)
            ):
                raise ValueError("per-population metrics disagree with confusion counts")
        record_map[key] = record
    if set(record_map) != expected_record_keys:
        raise ValueError("records do not cover the complete fixed grid")
    for seed in config.seeds:
        for components in config.component_grid:
            full = record_map[(seed, components, "full_width", "full_width")]
            all_panel = record_map[(seed, components, "loading_panel", "all")]
            for field in (
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "confusion_matrix_counts",
                "test_predictions_sha256",
                "loadings_sha256",
                "selected_original_indices_sha256",
            ):
                if full[field] != all_panel[field]:
                    raise ValueError("all-panel replay does not equal full width")

    expected_summary_keys = {
        (components, evaluation, label)
        for components in config.component_grid
        for evaluation, label, _ in grid
    }
    summary_keys = {
        (int(row["n_components"]), row["evaluation"], row["panel_label"])
        for row in payload["grid_summaries"]
    }
    if summary_keys != expected_summary_keys or len(summary_keys) != len(payload["grid_summaries"]):
        raise ValueError("grid summaries are incomplete or duplicated")
    for summary in payload["grid_summaries"]:
        rows = [
            record_map[(seed, int(summary["n_components"]), summary["evaluation"], summary["panel_label"])]
            for seed in config.seeds
        ]
        balanced = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        full_rows = [
            record_map[(seed, int(summary["n_components"]), "full_width", "full_width")]
            for seed in config.seeds
        ]
        derived_summary = {
            "seeds": len(rows),
            "minimum_selected_snp_count": int(min(row["selected_snp_count"] for row in rows)),
            "maximum_selected_snp_count": int(max(row["selected_snp_count"] for row in rows)),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "mean_balanced_accuracy": float(np.mean(balanced)),
            "sample_std_balanced_accuracy": float(np.std(balanced, ddof=1)) if len(rows) > 1 else 0.0,
            "minimum_balanced_accuracy": float(np.min(balanced)),
            "maximum_balanced_accuracy": float(np.max(balanced)),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
            "mean_paired_balanced_accuracy_delta_vs_full_width": float(
                np.mean(
                    [
                        row["balanced_accuracy"] - full["balanced_accuracy"]
                        for row, full in zip(rows, full_rows, strict=True)
                    ]
                )
            ),
        }
        for field, expected_value in derived_summary.items():
            observed = summary.get(field)
            if isinstance(expected_value, float):
                if not math.isclose(float(observed), expected_value, abs_tol=1e-14):
                    raise ValueError(f"summary {field} is not record-derived")
            elif observed != expected_value:
                raise ValueError(f"summary {field} is not record-derived")
        if summary.get("qualification_status") != "closed_form_pca_all_splits":
            raise ValueError("summary qualification status is invalid")

    expected_population_keys = {
        (components, evaluation, label, population)
        for components in config.component_grid
        for evaluation, label, _ in grid
        for population in EXPECTED_POPULATIONS
    }
    population_keys = {
        (
            int(row["n_components"]),
            row["evaluation"],
            row["panel_label"],
            row["population"],
        )
        for row in payload["population_summaries"]
    }
    if population_keys != expected_population_keys or len(population_keys) != len(payload["population_summaries"]):
        raise ValueError("population summaries are incomplete or duplicated")
    for summary in payload["population_summaries"]:
        records = [
            record_map[
                (
                    seed,
                    int(summary["n_components"]),
                    summary["evaluation"],
                    summary["panel_label"],
                )
            ]
            for seed in config.seeds
        ]
        metrics = [
            next(
                item
                for item in record["per_population_metrics"]
                if item["population"] == summary["population"]
            )
            for record in records
        ]
        recalls = np.asarray([item["recall"] for item in metrics], dtype=float)
        expected_values = {
            "seeds": len(records),
            "pooled_support": int(sum(item["support"] for item in metrics)),
            "pooled_true_positive": int(sum(item["true_positive"] for item in metrics)),
            "mean_recall": float(np.mean(recalls)),
            "minimum_recall": float(np.min(recalls)),
            "maximum_recall": float(np.max(recalls)),
        }
        for field, expected_value in expected_values.items():
            observed = summary.get(field)
            if isinstance(expected_value, float):
                if not math.isclose(float(observed), expected_value, abs_tol=1e-14):
                    raise ValueError(f"population summary {field} is not record-derived")
            elif observed != expected_value:
                raise ValueError(f"population summary {field} is not record-derived")

    registered = payload["registered_seed_policy"]
    if (
        registered.get("seed") != config.seeds[0]
        or registered.get("selection") != "first_configured_seed_never_best_seed"
        or registered.get("plotted_component_pairs")
        != {"pc1_pc2": 2, "pc3_pc4": max(config.component_grid)}
        or registered.get("panels") != [_panel_label(panel) for panel in config.panel_sizes]
    ):
        raise ValueError("registered-coordinate policy is invalid")
    coordinates = payload["registered_coordinates"]
    if not coordinates or len(coordinates) % len(config.panel_sizes) != 0:
        raise ValueError("registered coordinates have an invalid panel row count")
    coordinate_panels = {row.get("panel_label") for row in coordinates}
    expected_coordinate_panels = {
        _panel_label(value) for value in config.panel_sizes
    }
    if coordinate_panels != expected_coordinate_panels:
        raise ValueError("registered coordinates contain an unexpected panel")
    registered_sample_ids: set[str] | None = None
    for plotted_components in (2, max(config.component_grid)):
        for panel in [_panel_label(value) for value in config.panel_sizes]:
            panel_rows = [
                row
                for row in coordinates
                if row["panel_label"] == panel
                and row["n_components"] == plotted_components
            ]
            centroids = [row for row in panel_rows if row["row_type"] == "train_centroid"]
            samples = [row for row in panel_rows if row["row_type"] == "held_out_sample"]
            if (
                len(centroids) != len(EXPECTED_POPULATIONS)
                or not samples
                or {row["population"] for row in centroids} != set(EXPECTED_POPULATIONS)
                or any(row["seed"] != config.seeds[0] for row in panel_rows)
            ):
                raise ValueError("registered coordinates do not cover a plotted panel")
            sample_ids = {str(row["sample_id"]) for row in samples}
            if len(sample_ids) != len(samples):
                raise ValueError("registered held-out sample IDs must be unique per panel")
            if registered_sample_ids is None:
                registered_sample_ids = sample_ids
            elif sample_ids != registered_sample_ids:
                raise ValueError("registered panels do not contain the same held-out samples")
            if any(
                row.get("truth") not in EXPECTED_POPULATIONS
                or row.get("prediction") not in EXPECTED_POPULATIONS
                or row.get("population") != row.get("truth")
                or row.get("correct") is not (row.get("truth") == row.get("prediction"))
                for row in samples
            ):
                raise ValueError("registered held-out labels/predictions are inconsistent")
            registered_record = record_map[
                (
                    config.seeds[0],
                    plotted_components,
                    "loading_panel",
                    panel,
                )
            ]
            positions = {
                population: index
                for index, population in enumerate(EXPECTED_POPULATIONS)
            }
            coordinate_confusion = np.zeros(
                (len(EXPECTED_POPULATIONS), len(EXPECTED_POPULATIONS)), dtype=np.int64
            )
            for row in samples:
                coordinate_confusion[
                    positions[str(row["truth"])], positions[str(row["prediction"])]
                ] += 1
            if coordinate_confusion.tolist() != registered_record["confusion_matrix_counts"]:
                raise ValueError("registered coordinate predictions disagree with record")
            if any(
                not math.isfinite(float(row[f"component_{index}"]))
                for row in panel_rows
                for index in range(1, max(config.component_grid) + 1)
            ):
                raise ValueError("registered coordinates contain non-finite values")
    without_digest = dict(payload)
    digest = without_digest.pop("analysis_sha256")
    if not _is_sha256(digest) or canonical_sha256(without_digest) != digest:
        raise ValueError("analysis_sha256 mismatch")
    return {
        "status": "valid",
        "analysis_sha256": digest,
        "records": len(payload["records"]),
        "grid_summaries": len(payload["grid_summaries"]),
        "registered_coordinates": len(coordinates),
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    ""
                    if row.get(field) is None
                    else json.dumps(row[field], separators=(",", ":"))
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field)
                )
                for field in fieldnames
            }
        )
    return output.getvalue().encode("utf-8")


def _metrics_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "evaluation",
        "n_components",
        "panel_label",
        "requested_panel_size",
        "seeds",
        "minimum_selected_snp_count",
        "maximum_selected_snp_count",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "sample_std_balanced_accuracy",
        "minimum_balanced_accuracy",
        "maximum_balanced_accuracy",
        "mean_macro_f1",
        "mean_paired_balanced_accuracy_delta_vs_full_width",
        "qualification_status",
    )
    return _csv_bytes(payload["grid_summaries"], fields)


def _population_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "evaluation",
        "n_components",
        "panel_label",
        "requested_panel_size",
        "population",
        "seeds",
        "pooled_support",
        "pooled_true_positive",
        "mean_recall",
        "minimum_recall",
        "maximum_recall",
    )
    return _csv_bytes(payload["population_summaries"], fields)


def _coordinates_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    max_components = max(payload["config"]["component_grid"])
    fields = (
        "row_type",
        "seed",
        "n_components",
        "panel_label",
        "requested_panel_size",
        "selected_snp_count",
        "sample_index",
        "sample_id",
        "population",
        "truth",
        "prediction",
        "correct",
        *(f"component_{index}" for index in range(1, max_components + 1)),
    )
    return _csv_bytes(payload["registered_coordinates"], fields)


def _render_registered_figure(
    payload: Mapping[str, Any],
    path: Path,
    *,
    x_component: int,
    y_component: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    config = payload["config"]
    plotted_components = 2 if (x_component, y_component) == (1, 2) else max(
        config["component_grid"]
    )
    panels = [_panel_label(value) for value in config["panel_sizes"]]
    palette = plt.get_cmap("tab10")
    colors = {
        population: palette(index)
        for index, population in enumerate(EXPECTED_POPULATIONS)
    }
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(17.2, 4.9),
        squeeze=False,
        dpi=150,
    )
    summary_map = {
        (row["n_components"], row["panel_label"]): row
        for row in payload["grid_summaries"]
        if row["evaluation"] == "loading_panel"
    }
    for panel_index, panel in enumerate(panels):
        axis = axes[0, panel_index]
        rows = [
            row
            for row in payload["registered_coordinates"]
            if row["panel_label"] == panel
            and row["n_components"] == plotted_components
            and row["row_type"] == "held_out_sample"
        ]
        centroids = [
            row
            for row in payload["registered_coordinates"]
            if row["panel_label"] == panel
            and row["n_components"] == plotted_components
            and row["row_type"] == "train_centroid"
        ]
        for population in EXPECTED_POPULATIONS:
            population_rows = [row for row in rows if row["truth"] == population]
            correct = [row for row in population_rows if row["correct"]]
            incorrect = [row for row in population_rows if not row["correct"]]
            if correct:
                axis.scatter(
                    [row[f"component_{x_component}"] for row in correct],
                    [row[f"component_{y_component}"] for row in correct],
                    color=colors[population],
                    s=29,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.35,
                    label=population,
                )
            if incorrect:
                axis.scatter(
                    [row[f"component_{x_component}"] for row in incorrect],
                    [row[f"component_{y_component}"] for row in incorrect],
                    color=colors[population],
                    s=43,
                    marker="x",
                    linewidths=1.3,
                )
            centroid = next(row for row in centroids if row["population"] == population)
            axis.scatter(
                [centroid[f"component_{x_component}"]],
                [centroid[f"component_{y_component}"]],
                color=colors[population],
                s=105,
                marker="*",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )
        registered_record = next(
            row
            for row in payload["records"]
            if row["seed"] == config["seeds"][0]
            and row["n_components"] == plotted_components
            and row["evaluation"] == "loading_panel"
            and row["panel_label"] == panel
        )
        mean_summary = summary_map[(plotted_components, panel)]
        title_label = "All retained" if panel == "all" else f"Top {panel} loadings"
        axis.set_title(
            f"{title_label}\nseed BA={registered_record['balanced_accuracy']:.3f}; "
            f"{mean_summary['seeds']}-seed mean={mean_summary['mean_balanced_accuracy']:.3f}",
            fontsize=10,
        )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
        axis.set_xlabel(f"PC{x_component}")
        axis.set_ylabel(f"PC{y_component}")
        axis.grid(alpha=0.12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.95),
    )
    figure.suptitle(
        "PCA-only loading-panel diagnostic — held-out samples, first fixed seed "
        f"{config['seeds'][0]} (not best-seed selected)",
        fontsize=13,
        y=1.015,
    )
    figure.text(
        0.5,
        0.01,
        "Post-hoc development diagnostic. Ranking and PCA refits use outer-training genotypes only; "
        "stars are training centroids, x marks held-out errors, and each panel has its own PCA axes.",
        ha="center",
        fontsize=8.4,
    )
    figure.tight_layout(rect=(0.0, 0.065, 1.0, 0.9), w_pad=1.6)
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "vp-rspca", "Creation Time": None},
    )
    plt.close(figure)


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular file: {path}")
    return status


def write_output_package(
    payload: Mapping[str, Any], output_directory: Path
) -> dict[str, Path]:
    """Write six artifacts and then atomically publish a completion manifest."""

    validate_result_payload(payload)
    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        raise FileExistsError(f"output directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_directory.parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    if os.path.lexists(temporary):  # practically impossible, still fail closed
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / METRICS_FILENAME).write_bytes(_metrics_csv_bytes(payload))
        (temporary / POPULATION_FILENAME).write_bytes(_population_csv_bytes(payload))
        (temporary / COORDINATES_FILENAME).write_bytes(_coordinates_csv_bytes(payload))
        _render_registered_figure(
            payload,
            temporary / PC12_FILENAME,
            x_component=1,
            y_component=2,
        )
        _render_registered_figure(
            payload,
            temporary / PC34_FILENAME,
            x_component=3,
            y_component=4,
        )
        artifacts: dict[str, dict[str, Any]] = {}
        for filename in ARTIFACT_FILENAMES:
            path = temporary / filename
            status = _regular_file(path, role=filename)
            artifacts[filename] = {
                "bytes": int(status.st_size),
                "sha256": _file_sha256(path),
            }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "manifest_written_last": True,
            "analysis_status": "post_hoc_development_diagnostic",
            "confirmatory": False,
            "best_configuration_selected": False,
            "analysis_sha256": payload["analysis_sha256"],
            "artifact_count_excluding_manifest": len(ARTIFACT_FILENAMES),
            "exact_package_file_count": len(ARTIFACT_FILENAMES) + 1,
            "artifacts": artifacts,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "artifacts": artifacts,
            }
        )
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        temporary.rename(output_directory)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    validation = validate_output_package(output_directory)
    return {
        filename: output_directory / filename
        for filename in (*ARTIFACT_FILENAMES, MANIFEST_FILENAME)
    } | {"validation": validation}  # type: ignore[dict-item]


def validate_output_package(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("output package must be a real directory")
    expected = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    actual = {entry.name for entry in output_directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"output package files must be exact; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    manifest_path = output_directory / MANIFEST_FILENAME
    _regular_file(manifest_path, role="completion manifest")
    manifest = _strict_json_load(manifest_path)
    required_manifest = {
        "schema_version",
        "status",
        "manifest_written_last",
        "analysis_status",
        "confirmatory",
        "best_configuration_selected",
        "analysis_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if set(manifest) != required_manifest:
        raise ValueError("completion manifest schema mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["status"] != "complete"
        or manifest["manifest_written_last"] is not True
        or manifest["analysis_status"] != "post_hoc_development_diagnostic"
        or manifest["confirmatory"] is not False
        or manifest["best_configuration_selected"] is not False
        or manifest["artifact_count_excluding_manifest"] != len(ARTIFACT_FILENAMES)
        or manifest["exact_package_file_count"] != len(expected)
        or set(manifest["artifacts"]) != set(ARTIFACT_FILENAMES)
    ):
        raise ValueError("completion manifest policy or closure mismatch")
    unsigned = dict(manifest)
    manifest_digest = unsigned.pop("manifest_payload_sha256")
    if not _is_sha256(manifest_digest) or canonical_sha256(unsigned) != manifest_digest:
        raise ValueError("manifest_payload_sha256 mismatch")
    derived_package_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    if (
        not _is_sha256(manifest["package_anchor_sha256"])
        or manifest["package_anchor_sha256"] != derived_package_anchor
    ):
        raise ValueError("package_anchor_sha256 mismatch")
    for filename in ARTIFACT_FILENAMES:
        path = output_directory / filename
        status = _regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if set(receipt) != {"bytes", "sha256"}:
            raise ValueError(f"artifact receipt schema mismatch: {filename}")
        if receipt["bytes"] != status.st_size or receipt["sha256"] != _file_sha256(path):
            raise ValueError(f"artifact hash/size mismatch: {filename}")
    payload = _strict_json_load(output_directory / RESULT_FILENAME)
    result_validation = validate_result_payload(payload)
    if payload["analysis_sha256"] != manifest["analysis_sha256"]:
        raise ValueError("result and completion-manifest analysis hashes disagree")
    if (expected_analysis_sha256 is None) != (
        expected_package_anchor_sha256 is None
    ):
        raise ValueError(
            "external verification requires both analysis and package anchors"
        )
    if expected_analysis_sha256 is not None:
        if not _is_sha256(expected_analysis_sha256):
            raise ValueError("expected_analysis_sha256 is not a valid digest")
        if payload["analysis_sha256"] != expected_analysis_sha256:
            raise ValueError("package does not match the external analysis-hash anchor")
        if not _is_sha256(expected_package_anchor_sha256):
            raise ValueError("expected_package_anchor_sha256 is not a valid digest")
        if manifest["package_anchor_sha256"] != expected_package_anchor_sha256:
            raise ValueError("package does not match the external package-hash anchor")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        source_files = payload["source_identity"].get("files")
        if not isinstance(source_files, Mapping) or not source_files:
            raise ValueError("result has no source-file closure for repo verification")
        for relative, expected_hash in source_files.items():
            source_path = root / str(relative)
            _regular_file(source_path, role=f"source closure {relative}")
            if not _is_sha256(expected_hash) or _file_sha256(source_path) != expected_hash:
                raise ValueError(f"source closure hash mismatch: {relative}")
    if (output_directory / METRICS_FILENAME).read_bytes() != _metrics_csv_bytes(payload):
        raise ValueError("metrics CSV is not an exact serialization of result summaries")
    if (output_directory / POPULATION_FILENAME).read_bytes() != _population_csv_bytes(payload):
        raise ValueError("population CSV is not an exact serialization of result summaries")
    if (output_directory / COORDINATES_FILENAME).read_bytes() != _coordinates_csv_bytes(payload):
        raise ValueError("coordinate CSV is not an exact serialization of registered rows")
    for filename in (PC12_FILENAME, PC34_FILENAME):
        figure_path = output_directory / filename
        if not figure_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"invalid PNG signature: {filename}")
        from PIL import Image

        with Image.open(figure_path) as figure:
            figure.verify()
        with Image.open(figure_path) as figure:
            if figure.format != "PNG" or figure.width < 2500 or figure.height < 700:
                raise ValueError(f"PNG is unreadable or below the clarity floor: {filename}")
    return {
        **result_validation,
        "artifact_integrity_verified": True,
        "external_anchors_verified": expected_analysis_sha256 is not None,
        "source_closure_verified": repo_root is not None,
        "manifest_payload_sha256": manifest_digest,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "package_files": len(expected),
    }


def _source_identity(repo_root: Path) -> dict[str, Any]:
    files: dict[str, str] = {}
    for relative in SOURCE_FILE_ROLES:
        path = repo_root / relative
        _regular_file(path, role=relative)
        files[relative] = _file_sha256(path)
    return {
        "files": files,
        "file_roles": dict(SOURCE_FILE_ROLES),
        "source_manifest_sha256": canonical_sha256(
            {"files": files, "file_roles": SOURCE_FILE_ROLES}
        ),
        "formal_runner_modified": False,
        "fed_or_fedrpca_scientific_code_used": False,
    }


def _parse_int_tuple(value: str, *, role: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{role} must be comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError(f"{role} must not be empty")
    return parsed


def _parse_panels(value: str) -> tuple[int | None, ...]:
    parsed: list[int | None] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized == "all":
            parsed.append(None)
        else:
            try:
                parsed.append(int(normalized))
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    "panels must be comma-separated integers ending in all"
                ) from error
    if not parsed:
        raise argparse.ArgumentTypeError("panels must not be empty")
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="execute the fixed post-hoc grid")
    run.add_argument("--genotypes", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--expected-source-manifest-sha256", required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument(
        "--seeds",
        type=lambda value: _parse_int_tuple(value, role="seeds"),
        default=DEFAULT_SEEDS,
    )
    run.add_argument(
        "--components",
        type=lambda value: _parse_int_tuple(value, role="components"),
        default=DEFAULT_COMPONENTS,
    )
    run.add_argument("--panels", type=_parse_panels, default=DEFAULT_PANELS)
    run.add_argument("--test-fraction", type=float, default=0.2)
    validate = subparsers.add_parser("validate", help="validate a completed package")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path)
    validate.add_argument("--expected-analysis-sha256")
    validate.add_argument("--expected-package-anchor-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(
            json.dumps(
                validate_output_package(
                    args.output_dir,
                    repo_root=args.repo_root,
                    expected_analysis_sha256=args.expected_analysis_sha256,
                    expected_package_anchor_sha256=args.expected_package_anchor_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if os.path.lexists(args.output_dir):
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    data = load_genotype_npz(args.genotypes)
    preparation_receipt = validate_external_source_manifest(
        args.source_manifest,
        args.genotypes,
        data,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    config = PCALoadingGridConfig(
        seeds=tuple(args.seeds),
        component_grid=tuple(args.components),
        panel_sizes=tuple(args.panels),
        test_fraction=args.test_fraction,
    )
    repo_root = args.repo_root.resolve()
    input_identity = {
        **preparation_receipt,
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "orientation": "samples_by_snps",
        "ld_pruned_input_validated_from_external_manifest": True,
    }
    payload = evaluate_loading_grid(
        data,
        config,
        input_identity=input_identity,
        source_identity=_source_identity(repo_root),
    )
    paths = write_output_package(payload, args.output_dir)
    validation = paths.pop("validation")
    print(
        json.dumps(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "output_directory": str(args.output_dir),
                "validation": validation,
                "files": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

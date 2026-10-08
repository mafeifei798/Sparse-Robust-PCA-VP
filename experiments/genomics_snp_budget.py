#!/usr/bin/env python3
"""Leakage-safe PCA diagnostic for a fixed, fully reported SNP-budget grid.

This additive diagnostic asks a narrow question: how much of the held-out
1000 Genomes metadata super-population separation is retained when PCA is fit
with fewer variants?  It reuses the exact outer splits from a validated formal
Genomics benchmark.  Within each outer split, variants are ranked with a
multiclass Fisher score computed from outer-training rows and labels only.
Preprocessing, PCA, and nearest-centroid classification are then refit from
scratch on each selected outer-training subset.  Outer-test rows and labels
never participate in feature ranking or fitting.

All requested budgets are reported.  The tool deliberately performs no
post-hoc "best budget" selection and does not claim that selected variants are
biological markers or externally validated ancestry predictors.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

if __package__:
    from . import genomics_1kg as genomics_module
    from . import genomics_1kg_benchmark as benchmark_module
else:  # Support ``python experiments/genomics_snp_budget.py``.
    import genomics_1kg as genomics_module  # type: ignore[no-redef]
    import genomics_1kg_benchmark as benchmark_module  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions


ARTIFACT_FILENAMES = (
    "snp_budget_diagnostic.json",
    "snp_budget_metrics.csv",
    "snp_budget_diagnostic.png",
)
COMPLETION_MANIFEST = "snp_budget_output_manifest.json"
DEFAULT_BUDGETS: tuple[int | None, ...] = (50, 100, 200, 400, None)
CORE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
RECORD_FIELDS = frozenset(
    {
        "record_type",
        "feature_selection_scope",
        "preprocessing_fit_scope",
        "pca_fit_scope",
        "classifier_fit_scope",
        "test_used_for_model_or_feature_selection",
        "selection_rule",
        "selection_tie_break",
        "ranking_retained_mask_sha256",
        "ranking_original_indices_sha256",
        "ranking_scores_sha256",
        "ranking_snp_ids_sha256",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "requested_seed",
        "split_seed",
        "budget_label",
        "requested_snp_budget",
        "available_train_retained_snps",
        "selected_snp_count",
        "selected_snp_ids",
        "selected_snp_ids_sha256",
        "applicable",
        "applicability_status",
        "solver",
        "solver_converged",
        "solver_status",
        "solver_n_iter",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "class_order",
        "confusion_matrix_counts",
        "per_population_metrics",
        "centroids_sha256",
        "test_predictions_sha256",
        "formal_all_snp_replay_exact",
    }
)
PER_POPULATION_FIELDS = frozenset(
    {
        "population",
        "support",
        "predicted_count",
        "true_positive",
        "precision",
        "recall",
        "f1",
    }
)
RANKING_RECORD_FIELDS = frozenset(
    {
        "record_type",
        "requested_seed",
        "split_seed",
        "available_train_retained_snps",
        "ranking_snp_ids",
        "ranking_snp_ids_sha256",
        "ranking_original_indices_sha256",
        "ranking_scores_sha256",
        "ranking_retained_mask_sha256",
        "feature_selection_scope",
        "selection_rule",
        "selection_tie_break",
    }
)
SPLIT_REFERENCE_FIELDS = frozenset(
    {
        "requested_seed",
        "split_seed",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "expected_all_accuracy",
        "expected_all_balanced_accuracy",
        "expected_all_macro_f1",
        "expected_all_retained_snps",
        "expected_all_retained_snp_mask_sha256",
        "expected_all_retained_snp_ids_sha256",
        "expected_all_test_predictions_sha256",
    }
)
SOURCE_FILE_ROLES = {
    "experiments/__init__.py": "package_initialization_only",
    "experiments/genomics_snp_budget.py": "diagnostic_runner",
    "experiments/genomics_1kg.py": "direct_scientific_dependency",
    "experiments/genomics_1kg_benchmark.py": "benchmark_validation_dependency",
    "src/rspca/datasets.py": "direct_scientific_dependency",
    "src/rspca/metrics.py": "direct_scientific_dependency",
    "src/rspca/__init__.py": "package_initialization_only",
    "src/rspca/accelerated.py": "package_initialization_only",
    "src/rspca/cross_modal_gates.py": "package_initialization_only",
    "src/rspca/federated.py": "package_initialization_only",
    "src/rspca/fedrpca_audit.py": "package_initialization_only",
    "src/rspca/gauge.py": "package_initialization_only",
    "src/rspca/gauge_stress.py": "package_initialization_only",
    "src/rspca/product_aggregation.py": "package_initialization_only",
    "src/rspca/randomized.py": "package_initialization_only",
    "src/rspca/robust.py": "package_initialization_only",
    "src/rspca/statistical_gates.py": "package_initialization_only",
    "src/rspca/trust_audit.py": "package_initialization_only",
    "src/rspca/vp.py": "package_initialization_only",
}


@dataclass(frozen=True)
class SNPBudgetConfig:
    """Fixed analysis policy; ``None`` denotes all train-retained variants."""

    budgets: tuple[int | None, ...] = DEFAULT_BUDGETS
    rank: int = 5
    test_fraction: float = 0.2
    standardize: bool = True
    selection_rule: str = "outer_train_multiclass_fisher_score"


@dataclass(frozen=True)
class FormalSplitReference:
    requested_seed: int
    split_seed: int
    train_indices_sha256: str
    test_indices_sha256: str
    train_sample_ids_sha256: str
    test_sample_ids_sha256: str
    test_labels_sha256: str
    expected_all_accuracy: float
    expected_all_balanced_accuracy: float
    expected_all_macro_f1: float
    expected_all_retained_snps: int
    expected_all_retained_snp_mask_sha256: str
    expected_all_retained_snp_ids_sha256: str
    expected_all_test_predictions_sha256: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    return normalized


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {value}")

    with path.open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_config(config: SNPBudgetConfig) -> None:
    if config.selection_rule != "outer_train_multiclass_fisher_score":
        raise ValueError("selection_rule is fixed for this diagnostic")
    if isinstance(config.rank, bool) or not isinstance(config.rank, int):
        raise TypeError("rank must be an integer")
    if config.rank <= 0:
        raise ValueError("rank must be positive")
    if not np.isfinite(config.test_fraction) or not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if not isinstance(config.standardize, (bool, np.bool_)):
        raise TypeError("standardize must be boolean")
    if not isinstance(config.budgets, tuple) or not config.budgets:
        raise ValueError("budgets must be a non-empty tuple")
    normalized: list[int | None] = []
    for budget in config.budgets:
        if budget is None:
            normalized.append(None)
            continue
        if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
            raise ValueError("numeric SNP budgets must be positive integers")
        normalized.append(int(budget))
    if len(set(normalized)) != len(normalized):
        raise ValueError("SNP budgets must not contain duplicates")
    if normalized[-1] is not None or normalized.count(None) != 1:
        raise ValueError("the final and only unbounded SNP budget must be 'all'")
    numeric = [budget for budget in normalized if budget is not None]
    if numeric != sorted(numeric):
        raise ValueError("numeric SNP budgets must be strictly increasing")


def _budget_label(budget: int | None) -> str:
    return "all" if budget is None else str(int(budget))


def parse_budgets(value: str) -> tuple[int | None, ...]:
    tokens = [token.strip().lower() for token in value.split(",")]
    if not tokens or any(not token for token in tokens):
        raise argparse.ArgumentTypeError("budgets must be comma-separated")
    budgets: list[int | None] = []
    for token in tokens:
        if token == "all":
            budgets.append(None)
            continue
        try:
            budget = int(token)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "budgets must contain positive integers and final token 'all'"
            ) from error
        budgets.append(budget)
    try:
        _validate_config(SNPBudgetConfig(budgets=tuple(budgets)))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return tuple(budgets)


def rank_snps_training_only(
    train_genotypes: NDArray[np.float64],
    train_labels: NDArray[np.str_],
    *,
    standardize: bool = True,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Rank train-retained SNPs by deterministic multiclass Fisher score.

    No evaluation rows are accepted by this API.  Ties are broken by original
    SNP column index, making the selection independent of sort stability.
    """

    matrix = np.asarray(train_genotypes, dtype=float)
    labels = np.asarray(train_labels, dtype=str)
    if matrix.ndim != 2 or labels.shape != (matrix.shape[0],):
        raise ValueError("training genotypes and labels have incompatible shapes")
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError("feature ranking requires at least two populations")
    preprocessor = genomics_module.fit_genotype_preprocessor(
        matrix,
        standardize=standardize,
    )
    transformed = genomics_module.transform_genotypes(matrix, preprocessor)
    retained_indices = np.flatnonzero(preprocessor.retained_features).astype(np.int64)
    grand_mean = transformed.mean(axis=0)
    between = np.zeros(transformed.shape[1], dtype=float)
    within = np.zeros(transformed.shape[1], dtype=float)
    for label in classes:
        group = transformed[labels == label]
        group_mean = group.mean(axis=0)
        between += group.shape[0] * np.square(group_mean - grand_mean)
        within += np.square(group - group_mean).sum(axis=0)
    total = between + within
    denominator_floor = np.finfo(float).eps * np.maximum(total, 1.0)
    scores = between / np.maximum(within, denominator_floor)
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("training-only Fisher ranking produced non-finite scores")
    order = np.lexsort((retained_indices, -scores))
    return (
        retained_indices[order],
        np.asarray(scores[order], dtype=float),
        np.asarray(preprocessor.retained_features, dtype=bool),
    )


def _classification_diagnostics(
    truth: NDArray[np.str_],
    prediction: NDArray[np.str_],
    *,
    class_order: NDArray[np.str_],
) -> dict[str, Any]:
    labels = np.asarray(truth, dtype=str)
    predictions = np.asarray(prediction, dtype=str)
    classes = np.asarray(class_order, dtype=str)
    if labels.shape != predictions.shape or labels.ndim != 1 or labels.size == 0:
        raise ValueError("truth and predictions must be aligned non-empty vectors")
    if classes.ndim != 1 or classes.size < 2 or np.unique(classes).size != classes.size:
        raise ValueError("class_order must contain unique population labels")
    if not set(np.unique(np.concatenate((labels, predictions)))).issubset(set(classes)):
        raise ValueError("class_order does not cover observed labels")
    index = {label: position for position, label in enumerate(classes.tolist())}
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        confusion[index[actual], index[predicted]] += 1
    support = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    if np.any(support == 0):
        raise ValueError("each population requires held-out support")
    per_population: list[dict[str, Any]] = []
    for position, label in enumerate(classes.tolist()):
        true_positive = int(confusion[position, position])
        recall = float(true_positive / support[position])
        precision = (
            float(true_positive / predicted_counts[position])
            if predicted_counts[position]
            else 0.0
        )
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        per_population.append(
            {
                "population": label,
                "support": int(support[position]),
                "predicted_count": int(predicted_counts[position]),
                "true_positive": true_positive,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    recalls = [float(item["recall"]) for item in per_population]
    f1_values = [float(item["f1"]) for item in per_population]
    return {
        "accuracy": float(np.trace(confusion) / labels.size),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "class_order": classes.tolist(),
        "confusion_matrix_counts": confusion.tolist(),
        "per_population_metrics": per_population,
    }


def _nearest_centroid_predict(
    train_scores: NDArray[np.float64],
    train_labels: NDArray[np.str_],
    test_scores: NDArray[np.float64],
) -> tuple[NDArray[np.str_], NDArray[np.str_], NDArray[np.float64]]:
    classes = np.unique(train_labels)
    centroids = np.stack(
        [train_scores[train_labels == label].mean(axis=0) for label in classes]
    )
    distances = np.sum(
        np.square(test_scores[:, None, :] - centroids[None, :, :]),
        axis=2,
    )
    return classes[np.argmin(distances, axis=1)], classes, centroids


def _inapplicable_record(
    *,
    reference: FormalSplitReference,
    budget: int | None,
    available: int,
    common: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        **common,
        "requested_seed": reference.requested_seed,
        "split_seed": reference.split_seed,
        "budget_label": _budget_label(budget),
        "requested_snp_budget": budget,
        "available_train_retained_snps": available,
        "selected_snp_count": 0,
        "selected_snp_ids": [],
        "selected_snp_ids_sha256": None,
        "applicable": False,
        "applicability_status": reason,
        "solver": "closed_form_pca_svd",
        "solver_converged": None,
        "solver_status": "not_run_inapplicable",
        "solver_n_iter": 0,
        "accuracy": None,
        "balanced_accuracy": None,
        "macro_f1": None,
        "class_order": [],
        "confusion_matrix_counts": [],
        "per_population_metrics": [],
        "centroids_sha256": None,
        "test_predictions_sha256": None,
        "formal_all_snp_replay_exact": False,
    }


def evaluate_snp_budgets(
    dataset: genomics_module.GenotypeDataset,
    population_labels: NDArray[np.str_],
    config: SNPBudgetConfig,
    split_references: Sequence[FormalSplitReference],
    *,
    input_identity: Mapping[str, Any],
    formal_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate the fixed budget grid on every formal outer split."""

    _validate_config(config)
    matrix = np.asarray(dataset.genotypes, dtype=float)
    labels = np.asarray(population_labels, dtype=str)
    sample_ids = np.asarray(dataset.sample_ids, dtype=str)
    snp_ids = np.asarray(dataset.snp_ids, dtype=str)
    if matrix.ndim != 2 or labels.shape != (matrix.shape[0],):
        raise ValueError("population labels must align with genotype rows")
    if sample_ids.shape != (matrix.shape[0],) or snp_ids.shape != (matrix.shape[1],):
        raise ValueError("dataset identifiers do not align with genotype dimensions")
    if len(split_references) < 2:
        raise ValueError("at least two formal outer splits are required")
    seeds = [reference.requested_seed for reference in split_references]
    if len(set(seeds)) != len(seeds):
        raise ValueError("outer split requested seeds must be unique")

    records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    for reference in split_references:
        train_indices, test_indices = genomics_module.stratified_train_test_split(
            labels,
            config.test_fraction,
            reference.split_seed,
        )
        hashes = {
            "train_indices_sha256": array_sha256(train_indices),
            "test_indices_sha256": array_sha256(test_indices),
            "train_sample_ids_sha256": array_sha256(sample_ids[train_indices]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test_indices]),
            "test_labels_sha256": array_sha256(labels[test_indices]),
        }
        for field, actual in hashes.items():
            expected = getattr(reference, field)
            if actual != expected:
                raise ValueError(
                    f"formal outer split mismatch for seed {reference.requested_seed}: "
                    f"{field}"
                )

        raw_train = matrix[train_indices]
        raw_test = matrix[test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        ranked_indices, ranked_scores, retained_mask = rank_snps_training_only(
            raw_train,
            train_labels,
            standardize=config.standardize,
        )
        available = int(ranked_indices.size)
        ranked_snp_ids = snp_ids[ranked_indices]
        ranking_records.append(
            {
                "record_type": "outer_train_snp_ranking",
                "requested_seed": reference.requested_seed,
                "split_seed": reference.split_seed,
                "available_train_retained_snps": available,
                "ranking_snp_ids": ranked_snp_ids.tolist(),
                "ranking_snp_ids_sha256": array_sha256(ranked_snp_ids),
                "ranking_original_indices_sha256": array_sha256(ranked_indices),
                "ranking_scores_sha256": array_sha256(ranked_scores),
                "ranking_retained_mask_sha256": array_sha256(retained_mask),
                "feature_selection_scope": "outer_train_rows_and_labels_only",
                "selection_rule": config.selection_rule,
                "selection_tie_break": "ascending_original_snp_column_index",
            }
        )
        ranking_common = {
            "record_type": "outer_split_snp_budget_evaluation",
            "feature_selection_scope": "outer_train_rows_and_labels_only",
            "preprocessing_fit_scope": "outer_train_selected_snps_only",
            "pca_fit_scope": "outer_train_selected_snps_only",
            "classifier_fit_scope": "outer_train_component_scores_only",
            "test_used_for_model_or_feature_selection": False,
            "selection_rule": config.selection_rule,
            "selection_tie_break": "ascending_original_snp_column_index",
            "ranking_retained_mask_sha256": array_sha256(retained_mask),
            "ranking_original_indices_sha256": array_sha256(ranked_indices),
            "ranking_scores_sha256": array_sha256(ranked_scores),
            "ranking_snp_ids_sha256": array_sha256(snp_ids[ranked_indices]),
            **hashes,
        }
        for budget in config.budgets:
            if budget is not None and budget > available:
                records.append(
                    _inapplicable_record(
                        reference=reference,
                        budget=budget,
                        available=available,
                        common=ranking_common,
                        reason="requested_budget_exceeds_train_retained_snps",
                    )
                )
                continue
            requested_count = available if budget is None else int(budget)
            selected_indices = np.sort(ranked_indices[:requested_count])
            selected_ids = snp_ids[selected_indices]
            selected_train = raw_train[:, selected_indices]
            selected_test = raw_test[:, selected_indices]
            preprocessor = genomics_module.fit_genotype_preprocessor(
                selected_train,
                standardize=config.standardize,
            )
            if not np.all(preprocessor.retained_features):
                raise RuntimeError(
                    "a training-retained selected SNP became unavailable during refit"
                )
            train = genomics_module.transform_genotypes(selected_train, preprocessor)
            test = genomics_module.transform_genotypes(selected_test, preprocessor)
            if config.rank > min(train.shape):
                records.append(
                    _inapplicable_record(
                        reference=reference,
                        budget=budget,
                        available=available,
                        common=ranking_common,
                        reason="rank_exceeds_selected_training_matrix_dimensions",
                    )
                )
                continue
            _, _, right_transpose = np.linalg.svd(train, full_matrices=False)
            loadings = np.ascontiguousarray(right_transpose[: config.rank].T)
            train_scores = train @ loadings
            test_scores = test @ loadings
            prediction, classes, centroids = _nearest_centroid_predict(
                train_scores,
                train_labels,
                test_scores,
            )
            diagnostics = _classification_diagnostics(
                test_labels,
                prediction,
                class_order=classes,
            )
            selected_ids_sha256 = array_sha256(selected_ids)
            prediction_sha256 = array_sha256(prediction)
            if budget is None:
                replay_checks: tuple[tuple[str, Any, Any], ...] = (
                    ("accuracy", diagnostics["accuracy"], reference.expected_all_accuracy),
                    (
                        "balanced_accuracy",
                        diagnostics["balanced_accuracy"],
                        reference.expected_all_balanced_accuracy,
                    ),
                    ("macro_f1", diagnostics["macro_f1"], reference.expected_all_macro_f1),
                    (
                        "retained_snps",
                        int(selected_indices.size),
                        reference.expected_all_retained_snps,
                    ),
                    (
                        "retained_snp_mask_sha256",
                        array_sha256(retained_mask),
                        reference.expected_all_retained_snp_mask_sha256,
                    ),
                    (
                        "retained_snp_ids_sha256",
                        selected_ids_sha256,
                        reference.expected_all_retained_snp_ids_sha256,
                    ),
                    (
                        "test_predictions_sha256",
                        prediction_sha256,
                        reference.expected_all_test_predictions_sha256,
                    ),
                )
                for field, actual, expected in replay_checks:
                    if actual != expected:
                        raise ValueError(
                            "all-SNP replay does not exactly reproduce formal PCA "
                            f"for seed {reference.requested_seed}: {field}"
                        )
            records.append(
                {
                    **ranking_common,
                    "requested_seed": reference.requested_seed,
                    "split_seed": reference.split_seed,
                    "budget_label": _budget_label(budget),
                    "requested_snp_budget": budget,
                    "available_train_retained_snps": available,
                    "selected_snp_count": int(selected_indices.size),
                    "selected_snp_ids": selected_ids.tolist(),
                    "selected_snp_ids_sha256": selected_ids_sha256,
                    "applicable": True,
                    "applicability_status": "applicable",
                    "solver": "closed_form_pca_svd",
                    "solver_converged": True,
                    "solver_status": "closed_form_pca",
                    "solver_n_iter": 1,
                    "accuracy": diagnostics["accuracy"],
                    "balanced_accuracy": diagnostics["balanced_accuracy"],
                    "macro_f1": diagnostics["macro_f1"],
                    "class_order": diagnostics["class_order"],
                    "confusion_matrix_counts": diagnostics[
                        "confusion_matrix_counts"
                    ],
                    "per_population_metrics": diagnostics[
                        "per_population_metrics"
                    ],
                    "centroids_sha256": array_sha256(centroids),
                    "test_predictions_sha256": prediction_sha256,
                    "formal_all_snp_replay_exact": budget is None,
                }
            )

    summaries, population_summaries = summarize_records(
        records,
        budgets=config.budgets,
        requested_seeds=tuple(seeds),
    )
    analysis_policy = {
        "primary_question": (
            "retained held-out metadata super-population separation under "
            "the fixed, fully reported SNP-budget grid"
        ),
        "feature_selection": "supervised_multiclass_fisher_score_outer_train_only",
        "outer_test_usage": "final_evaluation_only",
        "budget_selection": "none_report_every_fixed_grid_budget",
        "best_budget_selected": False,
        "population_label_semantics": "1000_genomes_metadata_super_population",
        "population_label_is_race": False,
        "external_cohort_validation": False,
        "selected_variants_are_validated_biological_markers": False,
        "evidence_boundary": (
            "Repeated held-out splits of one fixed cohort; no independent-cohort "
            "generalization or individual ancestry inference."
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": asdict(config),
        "analysis_policy": analysis_policy,
        "input_identity": dict(input_identity),
        "formal_benchmark_identity": dict(formal_identity),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "ranking_records": ranking_records,
        "records": records,
        "budget_summaries": summaries,
        "population_summaries": population_summaries,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_result_payload(payload)
    return payload


def summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int | None],
    requested_seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = [_budget_label(budget) for budget in budgets]
    expected = {(seed, label) for seed in requested_seeds for label in labels}
    keyed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in records:
        key = (int(record["requested_seed"]), str(record["budget_label"]))
        if key in keyed:
            raise ValueError(f"duplicate SNP-budget record: {key}")
        keyed[key] = record
    if set(keyed) != expected:
        raise ValueError("SNP-budget records do not cover the declared seed-budget grid")
    summaries: list[dict[str, Any]] = []
    population_summaries: list[dict[str, Any]] = []
    for budget, label in zip(budgets, labels, strict=True):
        subset = [keyed[(seed, label)] for seed in requested_seeds]
        applicable = [record for record in subset if bool(record["applicable"])]
        summary: dict[str, Any] = {
            "budget_label": label,
            "requested_snp_budget": budget,
            "requested_splits": len(requested_seeds),
            "applicable_splits": len(applicable),
            "inapplicable_splits": len(requested_seeds) - len(applicable),
            "solver_converged_splits": sum(
                record["solver_converged"] is True for record in subset
            ),
            "qualification_status": (
                "closed_form_pca_all_splits"
                if len(applicable) == len(requested_seeds)
                else "inapplicable_split_present"
            ),
            "best_budget_selected": False,
        }
        for metric in CORE_METRICS:
            values = np.asarray([record[metric] for record in applicable], dtype=float)
            summary[f"mean_{metric}"] = float(np.mean(values)) if values.size else None
            summary[f"std_{metric}"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else None
            )
            summary[f"min_{metric}"] = float(np.min(values)) if values.size else None
            summary[f"max_{metric}"] = float(np.max(values)) if values.size else None
        summaries.append(summary)

        if not applicable:
            continue
        class_order = list(applicable[0]["class_order"])
        if any(list(record["class_order"]) != class_order for record in applicable):
            raise ValueError("population class order changed across outer splits")
        pooled = np.sum(
            np.asarray(
                [record["confusion_matrix_counts"] for record in applicable],
                dtype=np.int64,
            ),
            axis=0,
        )
        for position, population in enumerate(class_order):
            recalls: list[float] = []
            f1_values: list[float] = []
            for record in applicable:
                row = next(
                    item
                    for item in record["per_population_metrics"]
                    if item["population"] == population
                )
                recalls.append(float(row["recall"]))
                f1_values.append(float(row["f1"]))
            true_positive = int(pooled[position, position])
            support = int(pooled[position].sum())
            predicted_count = int(pooled[:, position].sum())
            pooled_recall = float(true_positive / support)
            pooled_precision = (
                float(true_positive / predicted_count) if predicted_count else 0.0
            )
            pooled_f1 = (
                float(
                    2.0
                    * pooled_precision
                    * pooled_recall
                    / (pooled_precision + pooled_recall)
                )
                if pooled_precision + pooled_recall
                else 0.0
            )
            population_summaries.append(
                {
                    "budget_label": label,
                    "requested_snp_budget": budget,
                    "population": population,
                    "applicable_splits": len(applicable),
                    "pooled_support": support,
                    "pooled_predicted_count": predicted_count,
                    "pooled_true_positive": true_positive,
                    "pooled_precision": pooled_precision,
                    "pooled_recall": pooled_recall,
                    "pooled_f1": pooled_f1,
                    "mean_split_recall": float(np.mean(recalls)),
                    "std_split_recall": (
                        float(np.std(recalls, ddof=1))
                        if len(recalls) > 1
                        else None
                    ),
                    "mean_split_f1": float(np.mean(f1_values)),
                    "std_split_f1": (
                        float(np.std(f1_values, ddof=1))
                        if len(f1_values) > 1
                        else None
                    ),
                    "best_budget_selected": False,
                }
            )
    return summaries, population_summaries


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "config",
        "analysis_policy",
        "input_identity",
        "formal_benchmark_identity",
        "source_identity",
        "runtime_versions",
        "ranking_records",
        "records",
        "budget_summaries",
        "population_summaries",
        "analysis_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("SNP-budget result top-level schema mismatch")
    if payload["schema_version"] != 1:
        raise ValueError("SNP-budget result schema_version must be 1")
    body = dict(payload)
    observed_sha = _require_sha256(body.pop("analysis_sha256"), "analysis_sha256")
    if canonical_sha256(body) != observed_sha:
        raise ValueError("SNP-budget analysis hash mismatch")
    config_payload = payload["config"]
    if not isinstance(config_payload, Mapping):
        raise ValueError("config must be an object")
    try:
        config = SNPBudgetConfig(
            budgets=tuple(config_payload["budgets"]),
            rank=config_payload["rank"],
            test_fraction=config_payload["test_fraction"],
            standardize=config_payload["standardize"],
            selection_rule=config_payload["selection_rule"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("invalid SNP-budget config schema") from error
    if set(config_payload) != set(asdict(config)):
        raise ValueError("invalid SNP-budget config fields")
    _validate_config(config)
    policy = payload["analysis_policy"]
    if not isinstance(policy, Mapping) or policy.get("best_budget_selected") is not False:
        raise ValueError("analysis must explicitly prohibit best-budget selection")
    if policy.get("outer_test_usage") != "final_evaluation_only":
        raise ValueError("outer-test usage policy mismatch")
    input_identity = payload["input_identity"]
    expected_input_fields = {
        "genotype_sha256",
        "metadata_sha256",
        "normalized_genotypes_sha256",
        "sample_ids_sha256",
        "snp_ids_sha256",
        "population_labels_sha256",
        "samples",
        "snps",
    }
    if not isinstance(input_identity, Mapping) or set(input_identity) != expected_input_fields:
        raise ValueError("input_identity schema mismatch")
    for field in expected_input_fields - {"samples", "snps"}:
        _require_sha256(input_identity[field], f"input_identity.{field}")
    for field in ("samples", "snps"):
        value = input_identity[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"input_identity.{field} must be a positive integer")
    formal = payload["formal_benchmark_identity"]
    expected_formal_fields = {
        "formal_results_sha256",
        "benchmark_manifest_sha256",
        "explicit_parent_seeds",
        "source_primary_method",
        "same_outer_split_hashes_required",
        "formal_test_fraction",
        "formal_rank",
        "formal_standardize",
        "all_snp_exact_replay_required",
        "outer_split_reference_sha256",
        "outer_split_references",
    }
    if not isinstance(formal, Mapping) or set(formal) != expected_formal_fields:
        raise ValueError("formal_benchmark_identity schema mismatch")
    for field in (
        "formal_results_sha256",
        "benchmark_manifest_sha256",
        "outer_split_reference_sha256",
    ):
        _require_sha256(formal[field], f"formal_benchmark_identity.{field}")
    if (
        formal["source_primary_method"] != "pca"
        or formal["same_outer_split_hashes_required"] is not True
        or formal["all_snp_exact_replay_required"] is not True
        or formal["formal_rank"] != config.rank
        or formal["formal_test_fraction"] != config.test_fraction
        or formal["formal_standardize"] != config.standardize
    ):
        raise ValueError("formal benchmark policy/config binding mismatch")
    requested_seeds = formal.get("explicit_parent_seeds")
    if (
        not isinstance(requested_seeds, list)
        or len(requested_seeds) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in requested_seeds)
        or len(set(requested_seeds)) != len(requested_seeds)
    ):
        raise ValueError("formal explicit parent seeds are invalid")
    reference_payloads = formal["outer_split_references"]
    if (
        not isinstance(reference_payloads, list)
        or len(reference_payloads) != len(requested_seeds)
        or canonical_sha256(reference_payloads)
        != formal["outer_split_reference_sha256"]
    ):
        raise ValueError("formal outer-split reference closure mismatch")
    references: dict[int, Mapping[str, Any]] = {}
    for expected_seed, reference in zip(
        requested_seeds, reference_payloads, strict=True
    ):
        if not isinstance(reference, Mapping) or set(reference) != SPLIT_REFERENCE_FIELDS:
            raise ValueError("formal outer-split reference schema mismatch")
        seed = reference["requested_seed"]
        split_seed = reference["split_seed"]
        retained = reference["expected_all_retained_snps"]
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed != expected_seed
            or isinstance(split_seed, bool)
            or not isinstance(split_seed, int)
            or split_seed < 0
            or isinstance(retained, bool)
            or not isinstance(retained, int)
            or retained <= 0
        ):
            raise ValueError("formal outer-split reference integer field mismatch")
        for field in SPLIT_REFERENCE_FIELDS:
            if field.endswith("_sha256"):
                _require_sha256(reference[field], f"outer_split_reference.{field}")
        for field in (
            "expected_all_accuracy",
            "expected_all_balanced_accuracy",
            "expected_all_macro_f1",
        ):
            value = reference[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"formal outer-split reference has invalid {field}")
        references[seed] = reference
    ranking_payloads = payload["ranking_records"]
    if not isinstance(ranking_payloads, list) or len(ranking_payloads) != len(
        requested_seeds
    ):
        raise ValueError("ranking_records must contain one record per formal seed")
    rankings: dict[int, Mapping[str, Any]] = {}
    for expected_seed, ranking in zip(
        requested_seeds, ranking_payloads, strict=True
    ):
        if not isinstance(ranking, Mapping) or set(ranking) != RANKING_RECORD_FIELDS:
            raise ValueError("training-only ranking record schema mismatch")
        if (
            ranking["record_type"] != "outer_train_snp_ranking"
            or ranking["requested_seed"] != expected_seed
            or ranking["split_seed"] != references[expected_seed]["split_seed"]
            or ranking["feature_selection_scope"]
            != "outer_train_rows_and_labels_only"
            or ranking["selection_rule"] != config.selection_rule
            or ranking["selection_tie_break"]
            != "ascending_original_snp_column_index"
        ):
            raise ValueError("training-only ranking policy/seed mismatch")
        available = ranking["available_train_retained_snps"]
        ranked_ids = ranking["ranking_snp_ids"]
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or not 0 < available <= input_identity["snps"]
            or not isinstance(ranked_ids, list)
            or len(ranked_ids) != available
            or any(not isinstance(value, str) or not value for value in ranked_ids)
            or len(set(ranked_ids)) != len(ranked_ids)
        ):
            raise ValueError("training-only ordered ranking IDs/count are invalid")
        for field in (
            "ranking_snp_ids_sha256",
            "ranking_original_indices_sha256",
            "ranking_scores_sha256",
            "ranking_retained_mask_sha256",
        ):
            _require_sha256(ranking[field], f"ranking_record.{field}")
        if array_sha256(np.asarray(ranked_ids, dtype=str)) != ranking[
            "ranking_snp_ids_sha256"
        ]:
            raise ValueError("ordered training-only SNP ranking hash mismatch")
        rankings[expected_seed] = ranking
    summaries, populations = summarize_records(
        payload["records"],
        budgets=config.budgets,
        requested_seeds=requested_seeds,
    )
    if payload["budget_summaries"] != summaries:
        raise ValueError("budget summaries do not match per-split records")
    if payload["population_summaries"] != populations:
        raise ValueError("population summaries do not match per-split records")
    budget_by_label = {
        _budget_label(budget): budget for budget in config.budgets
    }
    records_by_seed: dict[int, list[Mapping[str, Any]]] = {
        seed: [] for seed in requested_seeds
    }
    for record in payload["records"]:
        if not isinstance(record, Mapping) or set(record) != RECORD_FIELDS:
            raise ValueError("SNP-budget per-record schema mismatch")
        if (
            record["record_type"] != "outer_split_snp_budget_evaluation"
            or record["test_used_for_model_or_feature_selection"] is not False
        ):
            raise ValueError("record permits outer-test leakage")
        expected_scope = {
            "feature_selection_scope": "outer_train_rows_and_labels_only",
            "preprocessing_fit_scope": "outer_train_selected_snps_only",
            "pca_fit_scope": "outer_train_selected_snps_only",
            "classifier_fit_scope": "outer_train_component_scores_only",
            "selection_rule": config.selection_rule,
            "selection_tie_break": "ascending_original_snp_column_index",
            "solver": "closed_form_pca_svd",
        }
        if any(record[field] != value for field, value in expected_scope.items()):
            raise ValueError("record scientific scope/policy mismatch")
        requested_seed = record["requested_seed"]
        split_seed = record["split_seed"]
        if (
            isinstance(requested_seed, bool)
            or not isinstance(requested_seed, int)
            or requested_seed not in references
            or isinstance(split_seed, bool)
            or not isinstance(split_seed, int)
            or split_seed < 0
        ):
            raise ValueError("record seed is invalid")
        reference = references[requested_seed]
        ranking = rankings[requested_seed]
        if split_seed != reference["split_seed"]:
            raise ValueError("record split seed does not match formal reference")
        for field in (
            "ranking_retained_mask_sha256",
            "ranking_original_indices_sha256",
            "ranking_scores_sha256",
            "ranking_snp_ids_sha256",
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            _require_sha256(record[field], f"record.{field}")
        for field in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if record[field] != reference[field]:
                raise ValueError(f"record formal split hash mismatch: {field}")
        for field in (
            "available_train_retained_snps",
            "ranking_retained_mask_sha256",
            "ranking_original_indices_sha256",
            "ranking_scores_sha256",
            "ranking_snp_ids_sha256",
        ):
            if record[field] != ranking[field]:
                raise ValueError(f"record does not match ordered ranking: {field}")
        budget_label = record["budget_label"]
        if not isinstance(budget_label, str) or budget_label not in budget_by_label:
            raise ValueError("record budget label is invalid")
        expected_budget = budget_by_label[budget_label]
        if record["requested_snp_budget"] != expected_budget:
            raise ValueError("record requested budget does not match fixed grid")
        available = record["available_train_retained_snps"]
        selected_count = record["selected_snp_count"]
        if (
            isinstance(available, bool)
            or not isinstance(available, int)
            or not 0 < available <= input_identity["snps"]
            or isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or selected_count < 0
        ):
            raise ValueError("record SNP counts are invalid")
        if not isinstance(record["applicable"], bool):
            raise ValueError("record applicable must be boolean")
        if record["applicable"] is True:
            if (
                record["applicability_status"] != "applicable"
                or record["solver_converged"] is not True
                or record["solver_status"] != "closed_form_pca"
                or record["solver_n_iter"] != 1
            ):
                raise ValueError("applicable closed-form PCA status mismatch")
            expected_count = available if expected_budget is None else expected_budget
            if selected_count != expected_count or selected_count > available:
                raise ValueError("selected SNP count does not match requested budget")
            selected_ids = record["selected_snp_ids"]
            if (
                not isinstance(selected_ids, list)
                or len(selected_ids) != selected_count
                or any(not isinstance(value, str) or not value for value in selected_ids)
                or len(set(selected_ids)) != len(selected_ids)
            ):
                raise ValueError("selected SNP IDs do not match selected count")
            selected_digest = _require_sha256(
                record["selected_snp_ids_sha256"],
                "record.selected_snp_ids_sha256",
            )
            if array_sha256(np.asarray(selected_ids, dtype=str)) != selected_digest:
                raise ValueError("selected SNP ID hash mismatch")
            expected_selected = set(ranking["ranking_snp_ids"][:selected_count])
            if set(selected_ids) != expected_selected:
                raise ValueError(
                    "selected SNP IDs are not the exact Fisher-ranking prefix"
                )
            _require_sha256(record["centroids_sha256"], "record.centroids_sha256")
            _require_sha256(
                record["test_predictions_sha256"],
                "record.test_predictions_sha256",
            )
            for metric in CORE_METRICS:
                value = record[metric]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    or not 0.0 <= value <= 1.0
                ):
                    raise ValueError(f"applicable record has invalid {metric}")
            classes = record["class_order"]
            confusion = np.asarray(record["confusion_matrix_counts"])
            per_population = record["per_population_metrics"]
            if (
                not isinstance(classes, list)
                or len(classes) < 2
                or any(not isinstance(value, str) or not value for value in classes)
                or len(set(classes)) != len(classes)
                or confusion.shape != (len(classes), len(classes))
                or confusion.dtype.kind not in {"i", "u"}
                or np.any(confusion < 0)
                or int(confusion.sum()) <= 0
                or not isinstance(per_population, list)
                or len(per_population) != len(classes)
            ):
                raise ValueError("record confusion/per-population shape is invalid")
            recomputed_population: list[dict[str, Any]] = []
            for index, (population, item) in enumerate(
                zip(classes, per_population, strict=True)
            ):
                if not isinstance(item, Mapping) or set(item) != PER_POPULATION_FIELDS:
                    raise ValueError("per-population record schema mismatch")
                support = int(confusion[index].sum())
                predicted_count = int(confusion[:, index].sum())
                true_positive = int(confusion[index, index])
                if support <= 0:
                    raise ValueError("each population requires held-out support")
                recall = float(true_positive / support)
                precision = (
                    float(true_positive / predicted_count) if predicted_count else 0.0
                )
                f1 = (
                    float(2.0 * precision * recall / (precision + recall))
                    if precision + recall
                    else 0.0
                )
                recomputed_population.append(
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
            if per_population != recomputed_population:
                raise ValueError("per-population metrics do not match confusion matrix")
            recomputed_metrics = {
                "accuracy": float(np.trace(confusion) / confusion.sum()),
                "balanced_accuracy": float(
                    np.mean([row["recall"] for row in recomputed_population])
                ),
                "macro_f1": float(
                    np.mean([row["f1"] for row in recomputed_population])
                ),
            }
            if any(record[field] != value for field, value in recomputed_metrics.items()):
                raise ValueError("record metrics do not match confusion matrix")
        else:
            if (
                record["applicability_status"]
                not in {
                    "requested_budget_exceeds_train_retained_snps",
                    "rank_exceeds_selected_training_matrix_dimensions",
                }
                or record["solver_converged"] is not None
                or record["solver_status"] != "not_run_inapplicable"
                or record["solver_n_iter"] != 0
                or selected_count != 0
                or record["selected_snp_ids"] != []
                or record["selected_snp_ids_sha256"] is not None
                or any(record[metric] is not None for metric in CORE_METRICS)
                or record["class_order"] != []
                or record["confusion_matrix_counts"] != []
                or record["per_population_metrics"] != []
                or record["centroids_sha256"] is not None
                or record["test_predictions_sha256"] is not None
            ):
                raise ValueError("inapplicable record payload/status mismatch")
        if budget_label == "all":
            if record["formal_all_snp_replay_exact"] is not True:
                raise ValueError("all-SNP record lacks exact formal PCA replay")
            exact_replay = {
                "accuracy": reference["expected_all_accuracy"],
                "balanced_accuracy": reference["expected_all_balanced_accuracy"],
                "macro_f1": reference["expected_all_macro_f1"],
                "available_train_retained_snps": reference[
                    "expected_all_retained_snps"
                ],
                "selected_snp_count": reference["expected_all_retained_snps"],
                "ranking_retained_mask_sha256": reference[
                    "expected_all_retained_snp_mask_sha256"
                ],
                "selected_snp_ids_sha256": reference[
                    "expected_all_retained_snp_ids_sha256"
                ],
                "test_predictions_sha256": reference[
                    "expected_all_test_predictions_sha256"
                ],
            }
            if any(record[field] != value for field, value in exact_replay.items()):
                raise ValueError("all-SNP record does not replay formal PCA exactly")
        elif record["formal_all_snp_replay_exact"] is not False:
            raise ValueError("finite-budget record cannot claim formal all-SNP replay")
        records_by_seed[requested_seed].append(record)
    ranking_fields = (
        "available_train_retained_snps",
        "ranking_retained_mask_sha256",
        "ranking_original_indices_sha256",
        "ranking_scores_sha256",
        "ranking_snp_ids_sha256",
    )
    for seed, seed_records in records_by_seed.items():
        first = seed_records[0]
        if any(
            any(record[field] != first[field] for field in ranking_fields)
            for record in seed_records[1:]
        ):
            raise ValueError(f"training-only ranking changed across budgets for seed {seed}")
        applicable_sets = {
            record["budget_label"]: set(record["selected_snp_ids"])
            for record in seed_records
            if record["applicable"]
        }
        ordered_labels = [
            _budget_label(budget)
            for budget in config.budgets
            if _budget_label(budget) in applicable_sets
        ]
        for smaller, larger in zip(ordered_labels, ordered_labels[1:]):
            if not applicable_sets[smaller].issubset(applicable_sets[larger]):
                raise ValueError("selected SNP sets are not nested across fixed budgets")
    source = payload["source_identity"]
    if not isinstance(source, Mapping) or set(source) != {
        "files",
        "file_roles",
        "source_manifest_sha256",
    }:
        raise ValueError("source_identity schema mismatch")
    if (
        not isinstance(source["files"], Mapping)
        or not isinstance(source["file_roles"], Mapping)
        or set(source["files"]) != set(source["file_roles"])
        or set(source["files"]) != set(SOURCE_FILE_ROLES)
        or dict(source["file_roles"]) != SOURCE_FILE_ROLES
    ):
        raise ValueError("source_identity executed-file closure mismatch")
    for relative_path, digest in source["files"].items():
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
        ):
            raise ValueError("source_identity contains an unsafe path")
        _require_sha256(digest, f"source_identity.{relative_path}")
        if source["file_roles"][relative_path] not in {
            "diagnostic_runner",
            "direct_scientific_dependency",
            "benchmark_validation_dependency",
            "package_initialization_only",
        }:
            raise ValueError("source_identity contains an unknown file role")
    source_body = {
        "files": dict(source["files"]),
        "file_roles": dict(source["file_roles"]),
    }
    if canonical_sha256(source_body) != _require_sha256(
        source["source_manifest_sha256"], "source_manifest_sha256"
    ):
        raise ValueError("source manifest hash mismatch")


def _csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "row_type",
        "requested_seed",
        "split_seed",
        "budget_label",
        "requested_snp_budget",
        "applicable",
        "applicability_status",
        "selected_snp_count",
        "solver_converged",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "population",
        "pooled_support",
        "pooled_precision",
        "pooled_recall",
        "pooled_f1",
        "mean_split_recall",
        "std_split_recall",
        "requested_splits",
        "applicable_splits",
        "qualification_status",
        "best_budget_selected",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for record in payload["records"]:
        writer.writerow(
            {
                "row_type": "per_split",
                **{field: record.get(field) for field in fields if field != "row_type"},
            }
        )
    for summary in payload["budget_summaries"]:
        row = {
            "row_type": "budget_summary",
            **summary,
            "accuracy": summary["mean_accuracy"],
            "balanced_accuracy": summary["mean_balanced_accuracy"],
            "macro_f1": summary["mean_macro_f1"],
        }
        writer.writerow({field: row.get(field) for field in fields})
    for summary in payload["population_summaries"]:
        writer.writerow(
            {
                "row_type": "population_summary",
                **{field: summary.get(field) for field in fields if field != "row_type"},
            }
        )
    return buffer.getvalue().encode("utf-8")


def _figure_bytes(payload: Mapping[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    summaries = payload["budget_summaries"]
    population_summaries = payload["population_summaries"]
    labels = [str(summary["budget_label"]) for summary in summaries]
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for metric, display, color in (
        ("balanced_accuracy", "Balanced accuracy", "#1f77b4"),
        ("macro_f1", "Macro-F1", "#ff7f0e"),
    ):
        means = np.asarray(
            [summary[f"mean_{metric}"] for summary in summaries], dtype=float
        )
        standard_deviations = np.asarray(
            [summary[f"std_{metric}"] for summary in summaries], dtype=float
        )
        axes[0].errorbar(
            x,
            means,
            yerr=standard_deviations,
            marker="o",
            capsize=3,
            linewidth=2,
            label=display,
            color=color,
        )
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Fixed-grid SNP budget")
    axes[0].set_ylabel("Held-out score (mean ± split SD)")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="lower right")
    axes[0].set_title("All budgets reported; no best-budget selection")

    populations = sorted({row["population"] for row in population_summaries})
    for population in populations:
        by_budget = {
            row["budget_label"]: row
            for row in population_summaries
            if row["population"] == population
        }
        axes[1].plot(
            x,
            [by_budget[label]["pooled_recall"] for label in labels],
            marker="o",
            linewidth=1.7,
            label=population,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Fixed-grid SNP budget")
    axes[1].set_ylabel("Pooled held-out recall")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, loc="lower right")
    axes[1].set_title("Per-super-population diagnostic")
    figure.suptitle(
        "Training-only Fisher ranking → refit preprocessing/PCA/centroids",
        fontsize=12,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160, metadata={"Software": "matplotlib"})
    plt.close(figure)
    return buffer.getvalue()


def _exclusive_write(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def write_output_package(payload: Mapping[str, Any], output_dir: Path) -> Path:
    """Publish a completion-manifest-last package without overwriting paths."""

    validate_result_payload(payload)
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    parent = output_dir.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")
    artifacts = {
        ARTIFACT_FILENAMES[0]: _json_bytes(payload),
        ARTIFACT_FILENAMES[1]: _csv_bytes(payload),
        ARTIFACT_FILENAMES[2]: _figure_bytes(payload),
    }
    output_dir.mkdir(mode=0o700)
    entries: dict[str, Any] = {}
    for filename in ARTIFACT_FILENAMES:
        data = artifacts[filename]
        _exclusive_write(output_dir / filename, data)
        entries[filename] = {
            "sha256": _bytes_sha256(data),
            "size_bytes": len(data),
        }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "completion_manifest_written_last": True,
        "self_excluding_manifest": True,
        "analysis_sha256": payload["analysis_sha256"],
        "files": entries,
    }
    _exclusive_write(output_dir / COMPLETION_MANIFEST, _json_bytes(manifest))
    validate_output_package(output_dir)
    return output_dir


def validate_output_package(
    output_dir: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(output_dir)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("output package must be a non-symlink directory")
    expected = set(ARTIFACT_FILENAMES) | {COMPLETION_MANIFEST}
    actual = {entry.name for entry in root.iterdir()}
    if actual != expected:
        raise ValueError(
            f"output package file set mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError(f"output artifact must be a regular file: {entry.name}")
    manifest = _strict_json_load(root / COMPLETION_MANIFEST)
    if not isinstance(manifest, Mapping) or set(manifest) != {
        "schema_version",
        "complete",
        "completion_manifest_written_last",
        "self_excluding_manifest",
        "analysis_sha256",
        "files",
    }:
        raise ValueError("completion manifest schema mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["complete"] is not True
        or manifest["completion_manifest_written_last"] is not True
        or manifest["self_excluding_manifest"] is not True
        or set(manifest["files"]) != set(ARTIFACT_FILENAMES)
    ):
        raise ValueError("completion manifest policy mismatch")
    for filename in ARTIFACT_FILENAMES:
        entry = manifest["files"][filename]
        if not isinstance(entry, Mapping) or set(entry) != {"sha256", "size_bytes"}:
            raise ValueError(f"invalid manifest entry for {filename}")
        expected_hash = _require_sha256(entry["sha256"], f"{filename} sha256")
        path = root / filename
        if _file_sha256(path) != expected_hash or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"artifact hash/size mismatch: {filename}")
    payload = _strict_json_load(root / ARTIFACT_FILENAMES[0])
    validate_result_payload(payload)
    if payload["analysis_sha256"] != manifest["analysis_sha256"]:
        raise ValueError("completion manifest analysis hash mismatch")
    if not (root / ARTIFACT_FILENAMES[2]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("diagnostic figure is not a PNG")
    with (root / ARTIFACT_FILENAMES[1]).open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_rows = (
        len(payload["records"])
        + len(payload["budget_summaries"])
        + len(payload["population_summaries"])
    )
    if len(csv_rows) != expected_rows:
        raise ValueError("CSV row count does not match JSON payload")
    if repo_root is not None:
        source = payload["source_identity"]
        for relative_path, expected_hash in source["files"].items():
            candidate = Path(repo_root) / relative_path
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError(f"source closure file is missing: {relative_path}")
            if _file_sha256(candidate) != expected_hash:
                raise ValueError(f"source closure hash mismatch: {relative_path}")
    return {
        "valid": True,
        "analysis_sha256": payload["analysis_sha256"],
        "artifact_count": len(ARTIFACT_FILENAMES),
        "records": len(payload["records"]),
    }


def _load_formal_context(
    formal_results_path: Path,
    *,
    expected_formal_results_sha256: str,
) -> tuple[
    benchmark_module.GenomicsBenchmarkResult,
    list[FormalSplitReference],
    dict[str, Any],
]:
    expected = _require_sha256(
        expected_formal_results_sha256,
        "expected_formal_results_sha256",
    )
    actual = _file_sha256(formal_results_path)
    if actual != expected:
        raise ValueError("formal benchmark file hash mismatch")
    raw_payload = _strict_json_load(formal_results_path)
    result = benchmark_module.validate_result_payload(
        raw_payload,
        label="formal Genomics benchmark",
    )
    pca_records = {
        int(record["requested_seed"]): record
        for record in result.records
        if record["method"] == "pca"
    }
    seeds = tuple(int(seed) for seed in result.config.seeds)
    if set(pca_records) != set(seeds):
        raise ValueError("formal benchmark lacks one PCA record per declared seed")
    references = [
        FormalSplitReference(
            requested_seed=seed,
            split_seed=int(pca_records[seed]["split_seed"]),
            train_indices_sha256=_require_sha256(
                pca_records[seed]["train_indices_sha256"], "train_indices_sha256"
            ),
            test_indices_sha256=_require_sha256(
                pca_records[seed]["test_indices_sha256"], "test_indices_sha256"
            ),
            train_sample_ids_sha256=_require_sha256(
                pca_records[seed]["train_sample_ids_sha256"],
                "train_sample_ids_sha256",
            ),
            test_sample_ids_sha256=_require_sha256(
                pca_records[seed]["test_sample_ids_sha256"],
                "test_sample_ids_sha256",
            ),
            test_labels_sha256=_require_sha256(
                pca_records[seed]["test_labels_sha256"], "test_labels_sha256"
            ),
            expected_all_accuracy=float(pca_records[seed]["accuracy"]),
            expected_all_balanced_accuracy=float(
                pca_records[seed]["balanced_accuracy"]
            ),
            expected_all_macro_f1=float(pca_records[seed]["macro_f1"]),
            expected_all_retained_snps=int(pca_records[seed]["retained_snps"]),
            expected_all_retained_snp_mask_sha256=_require_sha256(
                pca_records[seed]["retained_snp_mask_sha256"],
                "retained_snp_mask_sha256",
            ),
            expected_all_retained_snp_ids_sha256=_require_sha256(
                pca_records[seed]["retained_snp_ids_sha256"],
                "retained_snp_ids_sha256",
            ),
            expected_all_test_predictions_sha256=_require_sha256(
                pca_records[seed]["test_predictions_sha256"],
                "test_predictions_sha256",
            ),
        )
        for seed in seeds
    ]
    identity = {
        "formal_results_sha256": actual,
        "benchmark_manifest_sha256": result.manifest["manifest_sha256"],
        "explicit_parent_seeds": list(seeds),
        "source_primary_method": "pca",
        "same_outer_split_hashes_required": True,
        "formal_test_fraction": result.config.test_fraction,
        "formal_rank": result.config.rank,
        "formal_standardize": result.config.standardize,
        "all_snp_exact_replay_required": True,
        "outer_split_reference_sha256": canonical_sha256(
            [asdict(reference) for reference in references]
        ),
        "outer_split_references": [asdict(reference) for reference in references],
    }
    return result, references, identity


def _source_identity(repo_root: Path) -> dict[str, Any]:
    file_roles = dict(SOURCE_FILE_ROLES)
    files = {relative: _file_sha256(repo_root / relative) for relative in file_roles}
    return {
        "files": files,
        "file_roles": file_roles,
        "source_manifest_sha256": canonical_sha256(
            {"files": files, "file_roles": file_roles}
        ),
    }


def run_from_paths(arguments: argparse.Namespace) -> Path:
    output_dir = Path(arguments.output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    genotype_path = Path(arguments.genotypes)
    metadata_path = Path(arguments.metadata)
    formal_path = Path(arguments.formal_results)
    expected_genotype = _require_sha256(
        arguments.expected_genotype_sha256,
        "expected_genotype_sha256",
    )
    expected_metadata = _require_sha256(
        arguments.expected_metadata_sha256,
        "expected_metadata_sha256",
    )
    if _file_sha256(genotype_path) != expected_genotype:
        raise ValueError("genotype file hash mismatch")
    if _file_sha256(metadata_path) != expected_metadata:
        raise ValueError("metadata file hash mismatch")
    formal, references, formal_identity = _load_formal_context(
        formal_path,
        expected_formal_results_sha256=arguments.expected_formal_results_sha256,
    )
    provenance = formal.provenance
    if provenance["genotype_sha256"] != expected_genotype:
        raise ValueError("formal benchmark is not bound to this genotype file")
    if provenance["metadata_sha256"] != expected_metadata:
        raise ValueError("formal benchmark is not bound to this metadata file")
    dataset = genomics_module.load_genotypes(
        genotype_path,
        orientation=str(provenance["input_orientation"]),
        missing_value=provenance["missing_value"],
    )
    labels = genomics_module.load_population_labels(
        metadata_path,
        dataset.sample_ids,
        sample_id_column=str(provenance["metadata_sample_id_column"]),
        population_column=str(provenance["population_column"]),
        delimiter=arguments.metadata_delimiter,
    )
    identities = {
        "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
        "sample_ids_sha256": array_sha256(dataset.sample_ids),
        "snp_ids_sha256": array_sha256(dataset.snp_ids),
        "population_labels_sha256": array_sha256(labels),
    }
    for field, actual in identities.items():
        if actual != provenance[field]:
            raise ValueError(f"loaded cohort does not match formal provenance: {field}")
    config = SNPBudgetConfig(
        budgets=arguments.budgets,
        rank=formal.config.rank,
        test_fraction=formal.config.test_fraction,
        standardize=formal.config.standardize,
    )
    repo_root = Path(arguments.repo_root).resolve()
    input_identity = {
        "genotype_sha256": expected_genotype,
        "metadata_sha256": expected_metadata,
        **identities,
        "samples": int(dataset.genotypes.shape[0]),
        "snps": int(dataset.genotypes.shape[1]),
    }
    payload = evaluate_snp_budgets(
        dataset,
        labels,
        config,
        references,
        input_identity=input_identity,
        formal_identity=formal_identity,
        source_identity=_source_identity(repo_root),
    )
    return write_output_package(payload, output_dir)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run and publish a new diagnostic package")
    run.add_argument("genotypes", type=Path)
    run.add_argument("metadata", type=Path)
    run.add_argument("formal_results", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--expected-genotype-sha256", required=True)
    run.add_argument("--expected-metadata-sha256", required=True)
    run.add_argument("--expected-formal-results-sha256", required=True)
    run.add_argument("--metadata-delimiter", default="\t")
    run.add_argument("--budgets", type=parse_budgets, default=DEFAULT_BUDGETS)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    validate = subparsers.add_parser("validate", help="validate an existing package")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        path = run_from_paths(arguments)
        result = validate_output_package(path, repo_root=arguments.repo_root)
    else:
        result = validate_output_package(
            arguments.output_dir,
            repo_root=arguments.repo_root,
        )
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    raise SystemExit(main())

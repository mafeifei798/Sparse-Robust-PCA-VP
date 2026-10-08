from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import genomics_snp_budget as diagnostic
from experiments.genomics_1kg import GenotypeDataset, stratified_train_test_split
from experiments.genomics_snp_budget import (
    FormalSplitReference,
    SNPBudgetConfig,
    evaluate_snp_budgets,
    rank_snps_training_only,
    validate_output_package,
    validate_result_payload,
    write_output_package,
)
from rspca.datasets import array_sha256, canonical_sha256


def _dataset() -> tuple[GenotypeDataset, np.ndarray]:
    rng = np.random.default_rng(702)
    labels = np.asarray(["AFR"] * 18 + ["EAS"] * 18 + ["EUR"] * 18, dtype=str)
    matrix = rng.normal(scale=0.7, size=(labels.size, 12))
    matrix[labels == "AFR", :3] -= 2.0
    matrix[labels == "EAS", :3] += 2.0
    matrix[labels == "EUR", 3:5] += 1.6
    dataset = GenotypeDataset(
        genotypes=matrix,
        sample_ids=np.asarray([f"sample_{index:03d}" for index in range(labels.size)]),
        snp_ids=np.asarray([f"snp_{index:03d}" for index in range(matrix.shape[1])]),
    )
    return dataset, labels


def _references(
    dataset: GenotypeDataset,
    labels: np.ndarray,
    seeds: tuple[int, ...] = (7201, 7202),
    *,
    test_fraction: float = 0.25,
) -> list[FormalSplitReference]:
    references: list[FormalSplitReference] = []
    for offset, split_seed in enumerate(seeds):
        train, test = stratified_train_test_split(labels, test_fraction, split_seed)
        ranked_indices, _, retained_mask = rank_snps_training_only(
            dataset.genotypes[train],
            labels[train],
        )
        selected = np.sort(ranked_indices)
        preprocessor = diagnostic.genomics_module.fit_genotype_preprocessor(
            dataset.genotypes[train][:, selected]
        )
        train_matrix = diagnostic.genomics_module.transform_genotypes(
            dataset.genotypes[train][:, selected], preprocessor
        )
        test_matrix = diagnostic.genomics_module.transform_genotypes(
            dataset.genotypes[test][:, selected], preprocessor
        )
        _, _, vt = np.linalg.svd(train_matrix, full_matrices=False)
        loadings = np.ascontiguousarray(vt[:2].T)
        predictions, classes, _ = diagnostic._nearest_centroid_predict(
            train_matrix @ loadings,
            labels[train],
            test_matrix @ loadings,
        )
        metrics = diagnostic._classification_diagnostics(
            labels[test], predictions, class_order=classes
        )
        references.append(
            FormalSplitReference(
                requested_seed=2025 + offset,
                split_seed=split_seed,
                train_indices_sha256=array_sha256(train),
                test_indices_sha256=array_sha256(test),
                train_sample_ids_sha256=array_sha256(dataset.sample_ids[train]),
                test_sample_ids_sha256=array_sha256(dataset.sample_ids[test]),
                test_labels_sha256=array_sha256(labels[test]),
                expected_all_accuracy=metrics["accuracy"],
                expected_all_balanced_accuracy=metrics["balanced_accuracy"],
                expected_all_macro_f1=metrics["macro_f1"],
                expected_all_retained_snps=int(selected.size),
                expected_all_retained_snp_mask_sha256=array_sha256(retained_mask),
                expected_all_retained_snp_ids_sha256=array_sha256(
                    dataset.snp_ids[selected]
                ),
                expected_all_test_predictions_sha256=array_sha256(predictions),
            )
        )
    return references


def _payload(
    *,
    config: SNPBudgetConfig | None = None,
    dataset: GenotypeDataset | None = None,
    labels: np.ndarray | None = None,
) -> dict[str, object]:
    if dataset is None or labels is None:
        dataset, labels = _dataset()
    config = config or SNPBudgetConfig(
        budgets=(2, 4, None),
        rank=2,
        test_fraction=0.25,
    )
    references = _references(
        dataset,
        labels,
        test_fraction=config.test_fraction,
    )
    source_files = {
        path: "e" * 64 for path in diagnostic.SOURCE_FILE_ROLES
    }
    source_roles = dict(diagnostic.SOURCE_FILE_ROLES)
    return evaluate_snp_budgets(
        dataset,
        labels,
        config,
        references,
        input_identity={
            "genotype_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
            "sample_ids_sha256": array_sha256(dataset.sample_ids),
            "snp_ids_sha256": array_sha256(dataset.snp_ids),
            "population_labels_sha256": array_sha256(labels),
            "samples": dataset.genotypes.shape[0],
            "snps": dataset.genotypes.shape[1],
        },
        formal_identity={
            "formal_results_sha256": "c" * 64,
            "benchmark_manifest_sha256": "d" * 64,
            "explicit_parent_seeds": [reference.requested_seed for reference in references],
            "source_primary_method": "pca",
            "same_outer_split_hashes_required": True,
            "formal_test_fraction": config.test_fraction,
            "formal_rank": config.rank,
            "formal_standardize": config.standardize,
            "all_snp_exact_replay_required": True,
            "outer_split_reference_sha256": canonical_sha256(
                [
                    {
                        key: value
                        for key, value in reference.__dict__.items()
                    }
                    for reference in references
                ]
            ),
            "outer_split_references": [
                dict(reference.__dict__) for reference in references
            ],
        },
        source_identity={
            "files": source_files,
            "file_roles": source_roles,
            "source_manifest_sha256": canonical_sha256(
                {
                    "files": source_files,
                    "file_roles": source_roles,
                }
            ),
        },
    )


def test_training_only_fisher_ranking_is_deterministic_and_informative() -> None:
    dataset, labels = _dataset()
    train, test = stratified_train_test_split(labels, 0.25, 81)
    first_indices, first_scores, first_retained = rank_snps_training_only(
        dataset.genotypes[train],
        labels[train],
    )
    mutated = dataset.genotypes.copy()
    mutated[test] = 10_000.0
    second_indices, second_scores, second_retained = rank_snps_training_only(
        mutated[train],
        labels[train],
    )
    np.testing.assert_array_equal(first_indices, second_indices)
    np.testing.assert_array_equal(first_scores, second_scores)
    np.testing.assert_array_equal(first_retained, second_retained)
    assert set(first_indices[:5]).issuperset({0, 1, 2})
    assert np.all(np.diff(first_scores) <= 0.0)


def test_fisher_ties_break_by_original_column_index() -> None:
    labels = np.asarray(["A"] * 4 + ["B"] * 4)
    base = np.asarray([-2.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 2.0])
    matrix = np.column_stack((base, base, -base, np.arange(8, dtype=float)))
    indices, scores, _ = rank_snps_training_only(matrix, labels)
    tied = [int(index) for index, score in zip(indices, scores, strict=True) if score == scores[0]]
    assert tied == sorted(tied)


def test_evaluation_reports_full_grid_and_no_best_budget() -> None:
    payload = _payload()
    assert len(payload["records"]) == 2 * 3
    assert [row["budget_label"] for row in payload["budget_summaries"]] == [
        "2",
        "4",
        "all",
    ]
    assert all(row["applicable"] for row in payload["records"])
    assert all(row["solver_converged"] for row in payload["records"])
    assert all(
        row["test_used_for_model_or_feature_selection"] is False
        for row in payload["records"]
    )
    assert payload["analysis_policy"]["best_budget_selected"] is False
    assert len(payload["population_summaries"]) == 3 * 3
    assert all(
        summary["qualification_status"] == "closed_form_pca_all_splits"
        for summary in payload["budget_summaries"]
    )


def test_unavailable_budget_is_retained_and_explicitly_inapplicable() -> None:
    payload = _payload(
        config=SNPBudgetConfig(
            budgets=(2, 50, None),
            rank=2,
            test_fraction=0.25,
        )
    )
    unavailable = [record for record in payload["records"] if record["budget_label"] == "50"]
    assert len(unavailable) == 2
    assert all(record["applicable"] is False for record in unavailable)
    assert all(record["balanced_accuracy"] is None for record in unavailable)
    summary = next(
        row for row in payload["budget_summaries"] if row["budget_label"] == "50"
    )
    assert summary["applicable_splits"] == 0
    assert summary["mean_balanced_accuracy"] is None
    assert summary["qualification_status"] == "inapplicable_split_present"


def test_formal_split_hash_mismatch_is_rejected_before_scoring() -> None:
    dataset, labels = _dataset()
    references = _references(dataset, labels)
    references[0] = replace(references[0], train_indices_sha256="0" * 64)
    with pytest.raises(ValueError, match="formal outer split mismatch"):
        evaluate_snp_budgets(
            dataset,
            labels,
            SNPBudgetConfig(budgets=(2, None), rank=2, test_fraction=0.25),
            references,
            input_identity={},
            formal_identity={
                "formal_results_sha256": "c" * 64,
                "benchmark_manifest_sha256": "d" * 64,
                "explicit_parent_seeds": [reference.requested_seed for reference in references],
                "source_primary_method": "pca",
                "same_outer_split_hashes_required": True,
                "formal_test_fraction": 0.25,
                "formal_rank": 2,
                "formal_standardize": True,
                "all_snp_exact_replay_required": True,
                "outer_split_reference_sha256": "e" * 64,
                "outer_split_references": [
                    dict(reference.__dict__) for reference in references
                ],
            },
            source_identity=_payload()["source_identity"],
        )


def test_all_snp_budget_must_exactly_replay_formal_pca() -> None:
    dataset, labels = _dataset()
    references = _references(dataset, labels)
    references[0] = replace(
        references[0],
        expected_all_balanced_accuracy=(
            references[0].expected_all_balanced_accuracy - 0.01
        ),
    )
    with pytest.raises(ValueError, match="all-SNP replay"):
        evaluate_snp_budgets(
            dataset,
            labels,
            SNPBudgetConfig(budgets=(2, None), rank=2, test_fraction=0.25),
            references,
            input_identity={},
            formal_identity={
                "explicit_parent_seeds": [reference.requested_seed for reference in references]
            },
            source_identity={},
        )


@pytest.mark.parametrize(
    "value",
    ("50,100,200,400,all", "2,all"),
)
def test_budget_parser_accepts_ordered_fixed_grid(value: str) -> None:
    parsed = diagnostic.parse_budgets(value)
    assert parsed[-1] is None


@pytest.mark.parametrize(
    "value",
    ("50,50,all", "all,50", "50", "0,all", "50,,all", "100,50,all"),
)
def test_budget_parser_rejects_ambiguous_or_posthoc_shapes(value: str) -> None:
    with pytest.raises(Exception):
        diagnostic.parse_budgets(value)


def test_result_validator_recomputes_summaries_after_coherent_rehash() -> None:
    payload = _payload()
    payload["records"][0]["balanced_accuracy"] = 0.0
    body = dict(payload)
    body.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(body)
    with pytest.raises(ValueError, match="summaries do not match"):
        validate_result_payload(payload)


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_field",
        "forged_selected_ids",
        "wrong_budget",
        "negative_available",
        "invalid_split_hash",
        "confusion_metric_mismatch",
        "per_population_mismatch",
        "non_nested_selected_sets",
        "alternate_nested_prefixes",
    ),
)
def test_record_validator_rejects_coherently_resigned_scientific_attacks(
    mutation: str,
) -> None:
    payload = _payload()
    record = payload["records"][0]
    if mutation == "unknown_field":
        record["forged_scientific_field"] = True
    elif mutation == "forged_selected_ids":
        forged = [f"FORGED_{index}" for index in range(record["selected_snp_count"])]
        record["selected_snp_ids"] = forged
        record["selected_snp_ids_sha256"] = array_sha256(np.asarray(forged, dtype=str))
    elif mutation == "wrong_budget":
        record["requested_snp_budget"] = 999
    elif mutation == "negative_available":
        record["available_train_retained_snps"] = -3
    elif mutation == "invalid_split_hash":
        record["train_indices_sha256"] = "not-a-digest"
    elif mutation == "confusion_metric_mismatch":
        record["confusion_matrix_counts"][0][0] += 1
    elif mutation == "per_population_mismatch":
        record["per_population_metrics"][0]["recall"] = 0.0
    elif mutation == "non_nested_selected_sets":
        seed = record["requested_seed"]
        records = [item for item in payload["records"] if item["requested_seed"] == seed]
        small = next(item for item in records if item["budget_label"] == "2")
        medium = next(item for item in records if item["budget_label"] == "4")
        outside = next(
            value
            for value in next(item for item in records if item["budget_label"] == "all")[
                "selected_snp_ids"
            ]
            if value not in medium["selected_snp_ids"]
        )
        small["selected_snp_ids"][0] = outside
        small["selected_snp_ids_sha256"] = array_sha256(
            np.asarray(small["selected_snp_ids"], dtype=str)
        )
    elif mutation == "alternate_nested_prefixes":
        seed = record["requested_seed"]
        ranking = next(
            item for item in payload["ranking_records"] if item["requested_seed"] == seed
        )["ranking_snp_ids"]
        for item in payload["records"]:
            if item["requested_seed"] != seed or item["budget_label"] == "all":
                continue
            count = item["selected_snp_count"]
            forged = ranking[-count:]
            item["selected_snp_ids"] = forged
            item["selected_snp_ids_sha256"] = array_sha256(
                np.asarray(forged, dtype=str)
            )
    body = dict(payload)
    body.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(body)
    with pytest.raises(ValueError):
        validate_result_payload(payload)


def test_output_package_is_complete_exclusive_and_tamper_evident(tmp_path: Path) -> None:
    payload = _payload()
    output = tmp_path / "budget_report"
    assert write_output_package(payload, output) == output
    result = validate_output_package(output)
    assert result["valid"] is True
    assert result["records"] == 6
    assert set(path.name for path in output.iterdir()) == {
        *diagnostic.ARTIFACT_FILENAMES,
        diagnostic.COMPLETION_MANIFEST,
    }
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_output_package(payload, output)

    csv_path = output / "snp_budget_metrics.csv"
    csv_path.write_bytes(csv_path.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        validate_output_package(output)


def test_output_validator_rejects_extra_files(tmp_path: Path) -> None:
    output = tmp_path / "budget_report"
    write_output_package(_payload(), output)
    (output / "extra.txt").write_text("unsealed", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        validate_output_package(output)


def test_output_json_is_strict_and_csv_has_all_row_types(tmp_path: Path) -> None:
    output = tmp_path / "budget_report"
    write_output_package(_payload(), output)
    text = (output / "snp_budget_diagnostic.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert parsed["analysis_policy"]["population_label_is_race"] is False
    csv_text = (output / "snp_budget_metrics.csv").read_text(encoding="utf-8")
    assert "per_split" in csv_text
    assert "budget_summary" in csv_text
    assert "population_summary" in csv_text


def test_source_closure_validation_binds_current_files(tmp_path: Path) -> None:
    for relative in diagnostic.SOURCE_FILE_ROLES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative}\nvalue = 1\n", encoding="utf-8")
    files = {
        relative: diagnostic._file_sha256(tmp_path / relative)
        for relative in diagnostic.SOURCE_FILE_ROLES
    }
    roles = dict(diagnostic.SOURCE_FILE_ROLES)
    payload = _payload()
    payload["source_identity"] = {
        "files": files,
        "file_roles": roles,
        "source_manifest_sha256": canonical_sha256(
            {
                "files": files,
                "file_roles": roles,
            }
        ),
    }
    body = dict(payload)
    body.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(body)
    output = tmp_path / "budget_report"
    write_output_package(payload, output)
    assert validate_output_package(output, repo_root=tmp_path)["valid"] is True
    source = tmp_path / "src/rspca/federated.py"
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source closure hash mismatch"):
        validate_output_package(output, repo_root=tmp_path)


def test_declared_source_closure_matches_fresh_import_execution() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = """
import json
from pathlib import Path
import sys
from experiments import genomics_snp_budget
root = Path.cwd().resolve()
loaded = set()
for module in tuple(sys.modules.values()):
    filename = getattr(module, "__file__", None)
    if not filename:
        continue
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.startswith("experiments/") or relative.startswith("src/rspca/"):
        loaded.add(relative)
print(json.dumps(sorted(loaded)))
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = "src"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    loaded = set(json.loads(completed.stdout))
    assert loaded == set(diagnostic.SOURCE_FILE_ROLES)

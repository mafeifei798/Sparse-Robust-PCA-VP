from __future__ import annotations

from dataclasses import asdict, replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_1kg_benchmark as benchmark
from experiments.genomics_1kg import GenotypeDataset
from experiments.genomics_1kg_benchmark import (
    CORE_METRICS,
    GenomicsBenchmarkConfig,
    run_genomics_benchmark,
    summarize_method_metrics,
    summarize_paired_differences,
    write_results,
)
from rspca.datasets import array_sha256, make_seed_manifest


def _dataset() -> tuple[GenotypeDataset, np.ndarray]:
    rng = np.random.default_rng(73)
    labels = np.asarray(["AFR"] * 12 + ["EUR"] * 12, dtype=str)
    matrix = rng.normal(size=(labels.size, 8))
    matrix[labels == "EUR", :3] += 1.5
    dataset = GenotypeDataset(
        genotypes=matrix,
        sample_ids=np.asarray([f"sample_{index}" for index in range(labels.size)]),
        snp_ids=np.asarray([f"snp_{index}" for index in range(matrix.shape[1])]),
    )
    return dataset, labels


def _provenance(dataset: GenotypeDataset, labels: np.ndarray) -> dict[str, object]:
    return {
        "genotype_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
        "sample_ids_sha256": array_sha256(dataset.sample_ids),
        "snp_ids_sha256": array_sha256(dataset.snp_ids),
        "population_labels_sha256": array_sha256(labels),
    }


def _digest(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fake_single_run(
    dataset: GenotypeDataset,
    config: object,
    *,
    population_labels: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    del dataset, population_labels
    method = str(config.method)  # type: ignore[attr-defined]
    requested_seed = int(config.base_seed)  # type: ignore[attr-defined]
    manifest = make_seed_manifest(
        requested_seed,
        2,
        namespace="genomics-1kg-v2",
        metadata=asdict(config),  # type: ignore[arg-type]
    )
    method_offset = {
        "pca": 0.0,
        "vp": 0.01,
        "randomized": 0.02,
        "robust": -0.01,
    }[method]
    seed_offset = (requested_seed % 10) * 0.001
    sparse = method != "pca"
    split_seed = int(manifest["seeds"][1])  # type: ignore[index]
    classification: dict[str, object] = {
        "record_type": "held_out_population_classification",
        "method": method,
        "accuracy": 0.80 + method_offset + seed_offset,
        "balanced_accuracy": 0.79 + method_offset + seed_offset,
        "macro_f1": 0.78 + method_offset + seed_offset,
        "selected_alpha": 0.1 if sparse else None,
        "alpha_selection_enabled": sparse,
        "alpha_selection_status": (
            "selected_from_converged_candidates" if sparse else "disabled"
        ),
        "alpha_selection_qualified": True,
        "alpha_selection_candidates": (
            [{"alpha": 0.1, "balanced_accuracy": 0.8}] if sparse else []
        ),
        "solver_converged": True,
        "solver_status": "test fixture",
        "solver_n_iter": 2,
        "solver_final_stationarity": 1e-7,
        "test_used_for_model_or_feature_selection": False,
        "split_seed": split_seed,
        "train_indices_sha256": _digest(f"train-{requested_seed}"),
        "test_indices_sha256": _digest(f"test-{requested_seed}"),
        "train_sample_ids_sha256": _digest(f"train-samples-{requested_seed}"),
        "test_sample_ids_sha256": _digest(f"test-samples-{requested_seed}"),
        "test_labels_sha256": _digest(f"test-labels-{requested_seed}"),
    }
    return [
        {
            "record_type": "full_cohort_unsupervised_fit",
            "method": method,
            "converged": True,
            "status": "fixture full run retained",
        },
        classification,
    ], manifest


def _fake_result(monkeypatch: pytest.MonkeyPatch):
    dataset, labels = _dataset()
    monkeypatch.setattr(benchmark, "run_genomics", _fake_single_run)
    config = GenomicsBenchmarkConfig(
        seeds=(2025, 2026),
        methods=("pca", "vp", "randomized", "robust"),
        alpha_candidates=(0.1, 1.0),
        max_iter=2,
        randomized_finalize_full_iters=0,
        bootstrap_resamples=32,
        bootstrap_seed=9,
    )
    return run_genomics_benchmark(
        dataset,
        labels,
        config,
        provenance=_provenance(dataset, labels),
    )


def test_explicit_range_parser_and_complete_paired_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert benchmark._parse_seeds("2025..2044") == tuple(range(2025, 2045))
    with pytest.raises(Exception, match="unique"):
        benchmark._parse_seeds("2025,2025")

    result = _fake_result(monkeypatch)
    assert result.manifest["explicit_parent_seeds"] == [2025, 2026]
    assert result.manifest["bootstrap_policy"] == {
        "unit": "explicit_parent_seed",
        "difference": "method_minus_pca",
        "interval": "percentile_95",
        "resamples": 32,
        "seed": 9,
        "metrics": list(CORE_METRICS),
        "runtime_included": False,
    }
    assert len(result.records) == len(result.full_runs) == 8
    assert len(result.method_summaries) == 4 * (len(CORE_METRICS) + 1)
    assert len(result.paired_summaries) == 3 * len(CORE_METRICS)
    assert {row["n_pairs"] for row in result.paired_summaries} == {2}
    assert all(row["metric"] != "runtime_seconds" for row in result.paired_summaries)
    assert all(len(run["records"]) == 2 for run in result.full_runs)
    assert all(row["bootstrap_ci95_status"] == "ok" for row in result.paired_summaries)
    assert all(row["inference_qualified"] is True for row in result.paired_summaries)

    vp_accuracy = next(
        row
        for row in result.paired_summaries
        if row["method"] == "vp" and row["metric"] == "accuracy"
    )
    assert vp_accuracy["mean_difference"] == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("duplicate", "duplicate record"),
        ("missing", "missing or unexpected seeds"),
        ("hash", "hash mismatch"),
        ("nonfinite", "must be finite"),
    ),
)
def test_aggregation_rejects_duplicate_missing_hash_mismatch_and_nonfinite_core(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    result = _fake_result(monkeypatch)
    records = [dict(record) for record in result.records]
    if mutation == "duplicate":
        records.append(dict(records[-1]))
    elif mutation == "missing":
        records = records[:-1]
    elif mutation == "hash":
        target = next(
            record
            for record in records
            if record["method"] == "vp" and record["requested_seed"] == 2025
        )
        target["test_indices_sha256"] = _digest("not-the-pca-split")
    elif mutation == "nonfinite":
        records[0]["accuracy"] = float("nan")

    with pytest.raises(ValueError, match=message):
        summarize_method_metrics(
            records,
            methods=result.config.methods,
            seeds=result.config.seeds,
            benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
        )


def test_strict_json_tidy_csv_and_no_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _fake_result(monkeypatch)
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "metrics.csv"
    paths = write_results(result, output_json=json_path, output_csv=csv_path)
    assert paths == {"json": json_path, "csv": csv_path}
    text = json_path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    assert len(payload["full_runs"]) == 8
    assert payload["evaluation_protocol"]["runtime_bootstrapped"] is False

    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["row_type"] for row in rows} == {
        "per_seed",
        "method_summary",
        "paired_summary",
    }
    assert all(
        row["metric"] != "runtime_seconds"
        for row in rows
        if row["row_type"] == "paired_summary"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_results(result, output_json=json_path, output_csv=tmp_path / "new.csv")
    with pytest.raises(ValueError, match="different paths"):
        write_results(result, output_json=tmp_path / "same", output_csv=tmp_path / "same")
    invalid_records = [dict(record) for record in result.records]
    invalid_records[0]["macro_f1"] = float("inf")
    invalid_result = replace(result, records=tuple(invalid_records))
    with pytest.raises(ValueError, match="must be finite"):
        write_results(
            invalid_result,
            output_json=tmp_path / "invalid.json",
            output_csv=tmp_path / "invalid.csv",
        )
    assert not (tmp_path / "invalid.json").exists()


def test_unqualified_fixed_budget_runs_are_retained_but_not_marked_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _fake_result(monkeypatch)
    records = [dict(record) for record in result.records]
    for record in records:
        if record["method"] == "vp":
            record["solver_converged"] = False
            record["alpha_selection_qualified"] = False
    method_summaries = summarize_method_metrics(
        records,
        methods=result.config.methods,
        seeds=result.config.seeds,
        benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
    )
    paired_summaries = summarize_paired_differences(
        records,
        methods=result.config.methods,
        seeds=result.config.seeds,
        benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
        bootstrap_resamples=result.config.bootstrap_resamples,
        bootstrap_seed=result.config.bootstrap_seed,
    )
    vp_methods = [row for row in method_summaries if row["method"] == "vp"]
    assert vp_methods
    assert {row["inference_qualified"] for row in vp_methods} == {False}
    vp_pairs = [row for row in paired_summaries if row["method"] == "vp"]
    assert {row["bootstrap_ci95_status"] for row in vp_pairs} == {
        "unqualified_fixed_budget_runs_included"
    }
    assert {row["qualified_pairs"] for row in vp_pairs} == {0}
    assert all(row["bootstrap_ci95_low"] is not None for row in vp_pairs)


def test_write_rejects_nonfinite_diagnostics_and_tampered_summaries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result = _fake_result(monkeypatch)
    records = [dict(record) for record in result.records]
    records[0]["solver_final_stationarity"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        write_results(
            replace(result, records=tuple(records)),
            output_json=tmp_path / "diagnostic.json",
            output_csv=tmp_path / "diagnostic.csv",
        )

    method_summaries = [dict(row) for row in result.method_summaries]
    method_summaries[0]["mean"] = 999.0
    with pytest.raises(ValueError, match="method summaries do not match"):
        write_results(
            replace(result, method_summaries=tuple(method_summaries)),
            output_json=tmp_path / "method.json",
            output_csv=tmp_path / "method.csv",
        )

    paired_summaries = [dict(row) for row in result.paired_summaries]
    paired_summaries[0]["mean_difference"] = 999.0
    with pytest.raises(ValueError, match="paired summaries do not match"):
        write_results(
            replace(result, paired_summaries=tuple(paired_summaries)),
            output_json=tmp_path / "paired.json",
            output_csv=tmp_path / "paired.csv",
        )

    full_runs = [dict(run) for run in result.full_runs]
    full_runs[0]["seed_manifest"] = dict(full_runs[0]["seed_manifest"])
    full_runs[0]["seed_manifest"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="seed manifest"):
        write_results(
            replace(result, full_runs=tuple(full_runs)),
            output_json=tmp_path / "manifest.json",
            output_csv=tmp_path / "manifest.csv",
        )


def test_real_runner_uses_same_outer_split_and_inner_alpha_selection() -> None:
    dataset, labels = _dataset()
    config = GenomicsBenchmarkConfig(
        seeds=(2025, 2026),
        methods=("pca", "vp"),
        rank=2,
        alpha=0.01,
        alpha_candidates=(0.01, 0.1),
        max_iter=2,
        tol=1e-4,
        oversample=1,
        n_power_iter=0,
        randomized_finalize_full_iters=0,
        bootstrap_resamples=16,
        bootstrap_seed=11,
    )
    result = run_genomics_benchmark(
        dataset,
        labels,
        config,
        provenance=_provenance(dataset, labels),
    )
    assert len(result.records) == 4
    for seed in config.seeds:
        paired = [row for row in result.records if row["requested_seed"] == seed]
        assert {row["method"] for row in paired} == {"pca", "vp"}
        assert len({row["test_indices_sha256"] for row in paired}) == 1
        vp = next(row for row in paired if row["method"] == "vp")
        assert vp["alpha_selection_enabled"] is True
        assert vp["alpha_selection_scope"] == "inner_train_validation_only"
        assert vp["selected_alpha"] in config.alpha_candidates


def test_cli_rejects_input_hash_mismatch_before_creating_outputs(tmp_path: Path) -> None:
    genotype = tmp_path / "genotypes.npz"
    metadata = tmp_path / "metadata.csv"
    genotype.write_bytes(b"not loaded because the hash gate runs first")
    metadata.write_text("sample_id,population\n", encoding="utf-8")
    output_json = tmp_path / "out.json"
    output_csv = tmp_path / "out.csv"
    with pytest.raises(ValueError, match="genotype SHA-256 mismatch"):
        benchmark.main(
            [
                str(genotype),
                "--metadata",
                str(metadata),
                "--expected-genotype-sha256",
                "0" * 64,
                "--expected-metadata-sha256",
                "0" * 64,
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
        )
    assert not output_json.exists()
    assert not output_csv.exists()

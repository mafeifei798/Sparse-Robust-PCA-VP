from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_1kg_benchmark as benchmark
from experiments.genomics_1kg import (
    GenomicsConfig,
    GenotypeDataset,
    run_genomics,
)
from experiments.genomics_1kg_benchmark import (
    GenomicsBenchmarkConfig,
    run_genomics_benchmark,
    validate_result_payload,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from tools import build_genomics_population_report as population_report
from tools.build_genomics_population_report import (
    aggregate_population_records,
    reproduce_population_records,
    write_population_report,
)


def _dataset() -> tuple[GenotypeDataset, np.ndarray]:
    rng = np.random.default_rng(517)
    labels = np.asarray(["AFR"] * 12 + ["EAS"] * 12 + ["EUR"] * 12, dtype=str)
    matrix = rng.normal(size=(labels.size, 10))
    matrix[labels == "EAS", :3] += 2.0
    matrix[labels == "EUR", 3:6] -= 2.0
    return (
        GenotypeDataset(
            genotypes=matrix,
            sample_ids=np.asarray([f"sample_{index:02d}" for index in range(labels.size)]),
            snp_ids=np.asarray([f"snp_{index:02d}" for index in range(matrix.shape[1])]),
        ),
        labels,
    )


def _provenance(dataset: GenotypeDataset, labels: np.ndarray) -> dict[str, object]:
    return {
        "genotype_path": "/formal/input/genotypes.npz",
        "metadata_path": "/formal/input/populations.csv",
        "genotype_sha256": "a" * 64,
        "metadata_sha256": "b" * 64,
        "expected_genotype_sha256": "a" * 64,
        "expected_metadata_sha256": "b" * 64,
        "input_orientation": "samples-by-snps",
        "internal_orientation": "samples-by-snps",
        "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
        "sample_ids_sha256": array_sha256(dataset.sample_ids),
        "snp_ids_sha256": array_sha256(dataset.snp_ids),
        "population_labels_sha256": array_sha256(labels),
        "samples": int(dataset.genotypes.shape[0]),
        "snps": int(dataset.genotypes.shape[1]),
        "population_column": "population",
        "metadata_sample_id_column": "sample_id",
        "missing_value": -9.0,
        "runtime_versions_at_start": runtime_versions(),
    }


def _classification(rows: list[dict[str, object]]) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["record_type"] == "held_out_population_classification"
    )


def test_population_diagnostics_are_opt_in_and_self_consistent() -> None:
    dataset, labels = _dataset()
    config = GenomicsConfig(
        method="pca",
        rank=2,
        bootstrap_repeats=0,
        test_fraction=0.25,
        base_seed=2025,
    )
    default_rows, default_manifest = run_genomics(
        dataset,
        config,
        population_labels=labels,
    )
    rich_rows, rich_manifest = run_genomics(
        dataset,
        config,
        population_labels=labels,
        include_population_diagnostics=True,
    )
    default = _classification(default_rows)
    rich = _classification(rich_rows)
    assert default_manifest == rich_manifest
    assert "held_out_prediction_records" not in default
    assert rich["population_label_is_race"] is False
    assert rich["embedding_fit_scope"] == "outer_train_only"
    assert rich["final_loading_support_fit_scope"] == "outer_train_only"
    assert rich["final_loading_support_coordinate_system"] == (
        "original_snp_raw_dosage_coefficients"
    )
    assert 0 <= int(rich["final_selected_snp_count"]) <= dataset.genotypes.shape[1]
    assert len(rich["final_selected_snp_ids"]) == rich["final_selected_snp_count"]
    predictions = rich["held_out_prediction_records"]
    assert isinstance(predictions, list)
    assert len(predictions) == rich["test_samples"]
    assert sum(sum(row) for row in rich["confusion_matrix_counts"]) == len(predictions)
    for prediction in predictions:
        predicted_index = int(
            np.argmin(prediction["squared_distances_to_centroids"])
        )
        assert rich["class_order"][predicted_index] == prediction[
            "predicted_population"
        ]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert rich[metric] == pytest.approx(default[metric])


def test_rebuild_reproduces_sealed_primary_records_and_writes_new_directory(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    dataset, labels = _dataset()
    config = GenomicsBenchmarkConfig(
        seeds=(2025, 2026),
        methods=("pca",),
        rank=2,
        max_iter=2,
        randomized_max_iter=2,
        alpha_candidates=(0.1,),
        bootstrap_resamples=8,
        bootstrap_seed=19,
    )
    sealed = run_genomics_benchmark(
        dataset,
        labels,
        config,
        provenance=_provenance(dataset, labels),
    )
    payload = json.loads(json.dumps(benchmark._json_payload(sealed)))
    records = reproduce_population_records(
        payload,
        dataset,
        labels,
        methods=("pca",),
        seeds=(2025, 2026),
    )
    assert len(records) == 2
    tampered_payload = copy.deepcopy(payload)
    tampered_payload["records"][0]["single_run_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="primary record"):
        reproduce_population_records(
            tampered_payload,
            dataset,
            labels,
            methods=("pca",),
            seeds=(2025,),
        )
    aggregate = aggregate_population_records(records)
    assert len(aggregate["method_summaries"]) == 1
    summary = aggregate["method_summaries"][0]
    assert summary["outer_splits"] == 2
    assert summary["held_out_decisions"] == sum(
        int(record["test_samples"]) for record in records
    )
    assert summary["mean_final_selected_snp_count"] == pytest.approx(10.0)

    output_dir = tmp_path / "population-report"
    written = write_population_report(
        records,
        output_dir=output_dir,
        source_identity={"source_results_sha256": "c" * 64},
    )
    assert written == output_dir.resolve()
    expected = {
        "confusion_matrix.csv",
        "held_out_predictions.csv",
        "method_summaries.csv",
        "per_population_metrics.csv",
        "population_diagnostics.png",
        "reproduced_population_records.json",
        "report.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["evaluation_semantics"]["label_is_race"] is False
    assert report["evaluation_semantics"]["independent_external_cohort"] is False
    assert report["method_summaries"] == aggregate["method_summaries"]
    with (output_dir / "held_out_predictions.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == summary["held_out_decisions"]
    assert {row["prediction_unit"] for row in predictions} == {
        "one_outer_split_held_out_decision"
    }
    assert (output_dir / "population_diagnostics.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_population_report(
            records,
            output_dir=output_dir,
            source_identity={},
        )


def _benchmark_payload() -> tuple[dict[str, object], GenotypeDataset, np.ndarray]:
    dataset, labels = _dataset()
    result = run_genomics_benchmark(
        dataset,
        labels,
        GenomicsBenchmarkConfig(
            seeds=(2025, 2026),
            methods=("pca",),
            rank=2,
            max_iter=2,
            randomized_max_iter=2,
            alpha_candidates=(0.1,),
            bootstrap_resamples=8,
            bootstrap_seed=19,
        ),
        provenance=_provenance(dataset, labels),
    )
    payload = json.loads(json.dumps(benchmark._json_payload(result)))
    return payload, dataset, labels


def _coherently_resign_payload(payload: dict[str, object]) -> None:
    """Rebuild every payload-authored identity layer after a provenance edit."""

    config_values = dict(payload["config"])  # type: ignore[arg-type]
    for field_name in ("seeds", "methods", "alpha_candidates"):
        config_values[field_name] = tuple(config_values[field_name])
    config = GenomicsBenchmarkConfig(**config_values)
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    rebuilt_benchmark = benchmark._manifest(config, provenance)
    payload["manifest"] = json.loads(json.dumps(rebuilt_benchmark))
    full_runs = payload["full_runs"]
    records = payload["records"]
    assert isinstance(full_runs, list) and isinstance(records, list)
    for full_run in full_runs:
        assert isinstance(full_run, dict)
        index = int(full_run["full_run_index"])
        method = str(full_run["method"])
        requested_seed = int(full_run["requested_seed"])
        rebuilt_child = benchmark._expected_single_run_manifest(
            config,
            provenance,
            method=method,
            seed=requested_seed,
        )
        full_run["seed_manifest"] = json.loads(json.dumps(rebuilt_child))
        run_records = full_run["records"]
        assert isinstance(run_records, list)
        classification = next(
            row
            for row in run_records
            if isinstance(row, dict)
            and row.get("record_type") == "held_out_population_classification"
        )
        rebuilt_record = benchmark._one_record(
            method=method,
            run_index=int(full_run["run_index"]),
            requested_seed=requested_seed,
            runtime_seconds=float(full_run["runtime_seconds"]),
            classification=classification,
            single_manifest=rebuilt_child,
            benchmark_manifest_sha256=str(rebuilt_benchmark["manifest_sha256"]),
            provenance=provenance,
        )
        rebuilt_record["full_run_index"] = index
        records[index] = json.loads(json.dumps(rebuilt_record))
    payload["method_summaries"] = json.loads(
        json.dumps(
            benchmark.summarize_method_metrics(
                records,
                methods=config.methods,
                seeds=config.seeds,
                benchmark_manifest_sha256=str(
                    rebuilt_benchmark["manifest_sha256"]
                ),
            )
        )
    )
    payload["paired_summaries"] = json.loads(
        json.dumps(
            benchmark.summarize_paired_differences(
                records,
                methods=config.methods,
                seeds=config.seeds,
                benchmark_manifest_sha256=str(
                    rebuilt_benchmark["manifest_sha256"]
                ),
                bootstrap_resamples=config.bootstrap_resamples,
                bootstrap_seed=config.bootstrap_seed,
            )
        )
    )


@pytest.mark.parametrize(
    "attack",
    (
        "both_runtime_dicts",
        "paths_and_metadata_columns",
        "raw_input_hashes",
        "fully_coherent_shape",
    ),
)
def test_external_expected_provenance_rejects_coherently_resigned_identity(
    attack: str,
) -> None:
    payload, _, _ = _benchmark_payload()
    expected = copy.deepcopy(payload["provenance"])
    provenance = payload["provenance"]
    assert isinstance(provenance, dict)
    if attack == "both_runtime_dicts":
        runtime = payload["runtime_versions"]
        provenance_runtime = provenance["runtime_versions_at_start"]
        assert isinstance(runtime, dict) and isinstance(provenance_runtime, dict)
        runtime["numpy"] = "fabricated-runtime"
        provenance_runtime["numpy"] = "fabricated-runtime"
    elif attack == "paths_and_metadata_columns":
        provenance["genotype_path"] = "/fabricated/genotypes.npz"
        provenance["metadata_path"] = "/fabricated/metadata.tsv"
        provenance["metadata_sample_id_column"] = "fabricated_sample"
        provenance["population_column"] = "fabricated_population"
    elif attack == "raw_input_hashes":
        provenance["genotype_sha256"] = "c" * 64
        provenance["expected_genotype_sha256"] = "c" * 64
        provenance["metadata_sha256"] = "d" * 64
        provenance["expected_metadata_sha256"] = "d" * 64
    elif attack == "fully_coherent_shape":
        provenance["samples"] = int(provenance["samples"]) + 1
        provenance["snps"] = int(provenance["snps"]) + 1
        provenance["normalized_genotypes_sha256"] = "e" * 64
        full_runs = payload["full_runs"]
        assert isinstance(full_runs, list)
        for full_run in full_runs:
            assert isinstance(full_run, dict)
            run_records = full_run["records"]
            assert isinstance(run_records, list)
            full_cohort = next(
                row
                for row in run_records
                if isinstance(row, dict)
                and row.get("record_type") == "full_cohort_unsupervised_fit"
            )
            held_out = next(
                row
                for row in run_records
                if isinstance(row, dict)
                and row.get("record_type")
                == "held_out_population_classification"
            )
            full_cohort["samples"] = int(full_cohort["samples"]) + 1
            full_cohort["input_snps"] = int(full_cohort["input_snps"]) + 1
            held_out["train_samples"] = int(held_out["train_samples"]) + 1
            held_out["inner_fit_samples"] = int(held_out["inner_fit_samples"]) + 1
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(attack)

    _coherently_resign_payload(payload)
    # Every payload-authored schema, record copy, digest, and summary agrees.
    # Only the independently supplied expected provenance can distinguish it
    # from the sealed formal identity.
    validate_result_payload(payload, label=f"self-consistent-{attack}")
    with pytest.raises(ValueError, match="external expected_provenance"):
        validate_result_payload(
            payload,
            label=f"externally-checked-{attack}",
            expected_provenance=expected,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest",
        "summary",
        "record",
        "full_run",
        "hash",
        "runtime_schema",
        "provenance_runtime",
        "provenance_samples",
        "single_manifest_shape",
    ),
)
def test_full_source_validator_rejects_every_integrity_layer_before_solver(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    payload, dataset, labels = _benchmark_payload()
    if mutation == "manifest":
        payload["manifest"]["manifest_sha256"] = "0" * 64
    elif mutation == "summary":
        payload["method_summaries"][0]["mean"] = -123.0
    elif mutation == "record":
        payload["records"][0]["accuracy"] = 0.123
    elif mutation == "full_run":
        payload["full_runs"][0]["runtime_seconds"] += 1.0
    elif mutation == "hash":
        payload["provenance"]["genotype_sha256"] = "0" * 64
    elif mutation == "runtime_schema":
        payload["runtime_versions"] = {"fabricated": "0"}
    elif mutation == "provenance_runtime":
        payload["provenance"]["runtime_versions_at_start"]["numpy"] = (
            "fabricated"
        )
    elif mutation == "provenance_samples":
        payload["provenance"]["samples"] = 999_999
    elif mutation == "single_manifest_shape":
        single_manifest = payload["full_runs"][0]["seed_manifest"]
        single_manifest["metadata"]["n_samples"] = 999_999
        body = dict(single_manifest)
        body.pop("manifest_sha256")
        single_manifest["manifest_sha256"] = canonical_sha256(body)
        payload["records"][0]["single_run_manifest_sha256"] = single_manifest[
            "manifest_sha256"
        ]
    else:  # pragma: no cover - guarded by parametrization
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        validate_result_payload(payload, label=f"tampered-{mutation}")

    def unexpected_solver(*args: object, **kwargs: object) -> object:
        raise AssertionError("solver must not run for an invalid source artifact")

    monkeypatch.setattr(population_report, "run_genomics", unexpected_solver)
    with pytest.raises(ValueError):
        reproduce_population_records(payload, dataset, labels)


def test_validator_rejects_fully_rehashed_provenance_shape_attack() -> None:
    payload, _, _ = _benchmark_payload()
    original_benchmark_hash = payload["manifest"]["manifest_sha256"]
    config_values = dict(payload["config"])
    for field_name in ("seeds", "methods", "alpha_candidates"):
        config_values[field_name] = tuple(config_values[field_name])
    config = GenomicsBenchmarkConfig(**config_values)

    # Simulate an attacker changing both a shape and a data-identity hash, then
    # consistently rebuilding the benchmark manifest, every child seed
    # manifest, and both hashes duplicated into every primary record.
    payload["provenance"]["samples"] = 999_999
    payload["provenance"]["normalized_genotypes_sha256"] = "c" * 64
    rebuilt_benchmark = benchmark._manifest(config, payload["provenance"])
    payload["manifest"] = json.loads(json.dumps(rebuilt_benchmark))
    assert payload["manifest"]["manifest_sha256"] != original_benchmark_hash
    for full_run in payload["full_runs"]:
        rebuilt_child = benchmark._expected_single_run_manifest(
            config,
            payload["provenance"],
            method=full_run["method"],
            seed=int(full_run["requested_seed"]),
        )
        full_run["seed_manifest"] = json.loads(json.dumps(rebuilt_child))
        record = payload["records"][int(full_run["full_run_index"])]
        record["benchmark_manifest_sha256"] = rebuilt_benchmark[
            "manifest_sha256"
        ]
        record["single_run_manifest_sha256"] = rebuilt_child[
            "manifest_sha256"
        ]
        record["normalized_genotypes_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="full-cohort samples"):
        validate_result_payload(payload, label="fully-rehashed-shape-attack")


def test_cli_validates_source_before_data_loading_or_output_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload, _, _ = _benchmark_payload()
    payload["manifest"]["manifest_sha256"] = "0" * 64
    source = tmp_path / "tampered.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    output_dir = tmp_path / "must-not-exist"

    def unexpected_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("data must not load for an invalid source artifact")

    monkeypatch.setattr(population_report, "load_genotypes", unexpected_load)
    with pytest.raises(ValueError, match="manifest"):
        population_report.main(
            [
                str(source),
                "--genotypes",
                str(tmp_path / "not-read.npz"),
                "--metadata",
                str(tmp_path / "not-read.tsv"),
                "--output-dir",
                str(output_dir),
            ]
        )
    assert not output_dir.exists()


def test_current_sealed_genomics_result_passes_full_payload_validator() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "results/nonfed_2026-08-29/genomics/results.json"
    )
    payload = json.loads(source.read_text(encoding="utf-8"))
    expected_provenance, trust_anchor = (
        population_report._expected_provenance_for_source(source, None)
    )
    assert trust_anchor == (
        source.parent.parent
        / "provenance/genomics_trusted_provenance.json"
    )
    result = validate_result_payload(
        payload,
        label=str(source),
        expected_provenance=expected_provenance,
    )
    assert len(result.records) == len(result.full_runs) == 80
    assert result.manifest["manifest_sha256"] == (
        "83e7380cd3bd0c9b09ea763773f5a0384cc8cfcea2679e43e17bf8bcbf26e7ec"
    )

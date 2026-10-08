from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_1kg_benchmark as benchmark
from experiments.genomics_1kg import GenotypeDataset
from rspca.datasets import array_sha256, canonical_sha256, make_seed_manifest
from tools import combine_genomics_benchmark_shards as combiner


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _dataset() -> tuple[GenotypeDataset, np.ndarray]:
    matrix = np.arange(48, dtype=float).reshape(8, 6) / 10.0
    dataset = GenotypeDataset(
        genotypes=matrix,
        sample_ids=np.asarray([f"sample-{index}" for index in range(8)]),
        snp_ids=np.asarray([f"snp-{index}" for index in range(6)]),
    )
    labels = np.asarray(["AFR"] * 4 + ["EUR"] * 4)
    return dataset, labels


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
        "runtime_versions_at_start": {
            "python": "fixture",
            "numpy": "fixture",
        },
    }


def _fake_single_run(
    dataset: GenotypeDataset,
    config: object,
    *,
    population_labels: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    method = str(config.method)  # type: ignore[attr-defined]
    requested_seed = int(config.base_seed)  # type: ignore[attr-defined]
    manifest = make_seed_manifest(
        requested_seed,
        int(config.bootstrap_repeats) + 2,  # type: ignore[attr-defined]
        namespace="genomics-1kg-v2",
        metadata={
            **asdict(config),  # type: ignore[arg-type]
            "n_samples": int(dataset.genotypes.shape[0]),
            "n_snps": int(dataset.genotypes.shape[1]),
            "orientation": "samples-by-snps",
            "loader_schema_version": 2,
            "normalized_genotypes_sha256": array_sha256(dataset.genotypes),
            "sample_ids_sha256": array_sha256(dataset.sample_ids),
            "snp_ids_sha256": array_sha256(dataset.snp_ids),
            "seed_roles": {
                "index_0": "full_cohort_solver",
                "index_1": "held_out_split_and_selection_parent",
                "index_2_onward": "bootstrap_solver_parent_for_resample_seed",
            },
        },
    )
    sparse = method != "pca"
    method_offset = {"pca": 0.0, "vp": 0.02}[method]
    seed_offset = (requested_seed % 10) * 0.001
    classification: dict[str, object] = {
        "record_type": "held_out_population_classification",
        "method": method,
        "accuracy": 0.80 + method_offset + seed_offset,
        "balanced_accuracy": 0.79 + method_offset + seed_offset,
        "macro_f1": 0.78 + method_offset + seed_offset,
        "selected_alpha": 0.1 if sparse else None,
        "alpha_selection_enabled": sparse,
        "alpha_selection_scope": (
            "inner_train_validation_only" if sparse else "disabled"
        ),
        "alpha_selection_status": (
            "selected_from_converged_candidates" if sparse else "disabled"
        ),
        "alpha_selection_qualified": True,
        "alpha_selection_candidates": (
            [{"alpha": 0.1, "balanced_accuracy": 0.8}] if sparse else []
        ),
        "solver_converged": True,
        "solver_status": "fixture",
        "solver_n_iter": 2,
        "solver_final_stationarity": 1e-7,
        "class_order": ["AFR", "EUR"],
        "test_used_for_model_or_feature_selection": False,
        "split_seed": int(manifest["seeds"][1]),
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
            "status": "fixture diagnostics retained",
        },
        classification,
    ], manifest


def _write_shard(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    name: str,
    seeds: tuple[int, ...],
    *,
    rank: int = 2,
) -> tuple[Path, benchmark.GenomicsBenchmarkResult]:
    dataset, labels = _dataset()
    monkeypatch.setattr(benchmark, "run_genomics", _fake_single_run)
    config = benchmark.GenomicsBenchmarkConfig(
        seeds=seeds,
        methods=("pca", "vp"),
        rank=rank,
        alpha_candidates=(0.1, 1.0),
        max_iter=2,
        randomized_max_iter=2,
        randomized_finalize_full_iters=0,
        bootstrap_resamples=16,
        bootstrap_seed=11,
    )
    result = benchmark.run_genomics_benchmark(
        dataset,
        labels,
        config,
        provenance=_provenance(dataset, labels),
    )
    path = tmp_path / f"{name}.json"
    benchmark.write_results(result, output_json=path)
    return path, result


def test_combines_disjoint_shards_in_canonical_order_and_recomputes_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_path, first = _write_shard(monkeypatch, tmp_path, "first", (2025, 2027))
    second_path, second = _write_shard(monkeypatch, tmp_path, "second", (2026, 2028))

    expected_seeds = (2025, 2026, 2027, 2028)
    forward = combiner.combine_shard_paths(
        [first_path, second_path], expected_seeds=expected_seeds
    )
    reverse = combiner.combine_shard_paths(
        [second_path, first_path], expected_seeds=expected_seeds
    )
    assert forward == reverse
    assert forward.config.seeds == (2025, 2026, 2027, 2028)
    assert forward.provenance == first.provenance == second.provenance
    assert forward.manifest == benchmark._manifest(forward.config, forward.provenance)
    assert [record["requested_seed"] for record in forward.records] == [
        2025,
        2025,
        2026,
        2026,
        2027,
        2027,
        2028,
        2028,
    ]
    assert [record["method"] for record in forward.records] == ["pca", "vp"] * 4
    assert [record["full_run_index"] for record in forward.records] == list(range(8))
    assert {record["benchmark_manifest_sha256"] for record in forward.records} == {
        forward.manifest["manifest_sha256"]
    }
    assert {summary["n_seeds"] for summary in forward.method_summaries} == {4}
    assert {summary["n_pairs"] for summary in forward.paired_summaries} == {4}

    output_json = tmp_path / "combined" / "results.json"
    output_csv = tmp_path / "combined" / "metrics.csv"
    assert (
        combiner.main(
            [
                str(second_path),
                str(first_path),
                "--expected-seeds",
                "2025..2028",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
        )
        == 0
    )
    assert output_json.is_file() and output_csv.is_file()
    persisted = combiner.load_genomics_shard(output_json)
    assert persisted == forward


def test_rejects_rehashed_single_run_manifest_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_path, _ = _write_shard(monkeypatch, tmp_path, "first", (2025, 2026))
    second_path, _ = _write_shard(monkeypatch, tmp_path, "second", (2027, 2028))
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    single_manifest = payload["full_runs"][0]["seed_manifest"]
    single_manifest["metadata"]["n_samples"] += 1
    manifest_body = dict(single_manifest)
    manifest_body.pop("manifest_sha256")
    single_manifest["manifest_sha256"] = canonical_sha256(manifest_body)
    payload["records"][0]["single_run_manifest_sha256"] = single_manifest[
        "manifest_sha256"
    ]
    first_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="seed manifest does not bind its config/data"):
        combiner.combine_shard_paths(
            [first_path, second_path],
            expected_seeds=(2025, 2026, 2027, 2028),
        )


def test_rejects_overlapping_seeds_and_config_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_path, _ = _write_shard(monkeypatch, tmp_path, "first", (2025, 2026))
    overlap_path, _ = _write_shard(monkeypatch, tmp_path, "overlap", (2026, 2027))
    with pytest.raises(ValueError, match="seed sets overlap"):
        combiner.combine_shard_paths(
            [first_path, overlap_path],
            expected_seeds=(2025, 2026, 2027),
        )

    drift_path, _ = _write_shard(
        monkeypatch,
        tmp_path,
        "drift",
        (2027, 2028),
        rank=3,
    )
    with pytest.raises(ValueError, match="config differs outside the seeds field"):
        combiner.combine_shard_paths(
            [first_path, drift_path],
            expected_seeds=(2025, 2026, 2027, 2028),
        )

    disjoint_path, _ = _write_shard(monkeypatch, tmp_path, "disjoint", (2027, 2028))
    with pytest.raises(ValueError, match="does not match the expected seed plan"):
        combiner.combine_shard_paths(
            [first_path, disjoint_path],
            expected_seeds=(2025, 2026, 2027, 2028, 2029),
        )


def test_rejects_record_tampering_and_cli_preflights_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_path, _ = _write_shard(monkeypatch, tmp_path, "first", (2025, 2026))
    second_path, _ = _write_shard(monkeypatch, tmp_path, "second", (2027, 2028))
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    payload["records"][0]["accuracy"] = 0.123
    first_path.write_text(
        json.dumps(payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid Genomics benchmark shard"):
        combiner.combine_shard_paths(
            [first_path, second_path],
            expected_seeds=(2025, 2026, 2027, 2028),
        )

    existing_json = tmp_path / "existing.json"
    existing_json.write_text("sentinel", encoding="utf-8")
    csv_path = tmp_path / "must-not-exist.csv"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        combiner.main(
            [
                str(tmp_path / "missing-one.json"),
                str(tmp_path / "missing-two.json"),
                "--expected-seeds",
                "2025..2028",
                "--output-json",
                str(existing_json),
                "--output-csv",
                str(csv_path),
            ]
        )
    assert existing_json.read_text(encoding="utf-8") == "sentinel"
    assert not csv_path.exists()


def test_genomics_writer_rolls_back_both_outputs_if_second_reservation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, result = _write_shard(monkeypatch, tmp_path, "source", (2025, 2026))
    output_json = tmp_path / "transaction" / "results.json"
    output_csv = tmp_path / "transaction" / "metrics.csv"
    original_open = benchmark.os.open
    calls = 0

    def fail_second_reservation(path: object, flags: int, mode: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise FileExistsError("simulated reservation race")
        return original_open(path, flags, mode)

    monkeypatch.setattr(benchmark.os, "open", fail_second_reservation)
    with pytest.raises(FileExistsError, match="simulated reservation race"):
        benchmark.write_results(
            result,
            output_json=output_json,
            output_csv=output_csv,
        )

    assert not output_json.exists()
    assert not output_csv.exists()

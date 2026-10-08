from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from experiments.video_spca import MethodReport, VideoExperimentConfig
from experiments.video_spca_benchmark import (
    CORE_METRICS,
    VideoBenchmarkConfig,
    VideoBenchmarkResult,
    run_video_benchmark,
    summarize_paired_differences,
    write_results,
)
from rspca.datasets import make_seed_manifest, verify_seed_manifest


def _smoke_config(**overrides: object) -> VideoBenchmarkConfig:
    video = VideoExperimentConfig(
        grid_size=8,
        n_timesteps=16,
        total_time=5.0,
        noise_std=0.02,
        corruption_fraction=0.01,
        corruption_scale=0.5,
        seed=0,
        alpha=0.08,
        max_iter=2,
        tol=1e-4,
        randomized_oversample=1,
        randomized_power_iters=0,
        randomized_full_refine_iters=0,
    )
    values: dict[str, object] = {
        "video_config": video,
        "repeats": 2,
        "base_seed": 2025,
        "methods": ("PCA", "VP-SPCA"),
        "bootstrap_resamples": 64,
        "bootstrap_seed": 91,
    }
    values.update(overrides)
    return VideoBenchmarkConfig(**values)  # type: ignore[arg-type]


def _synthetic_records(n_pairs: int = 3) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for run_index in range(n_pairs):
        seed = 100 + run_index
        common = {
            "run_index": run_index,
            "seed": seed,
            "data_sha256": f"data-{seed}",
            "dataset_manifest_sha256": f"manifest-{seed}",
            "fixed_video_config_sha256": "fixed-config",
            "seed_manifest_sha256": "seed-manifest",
            "converged": True,
        }
        baseline = {metric: float(run_index + 1) for metric in CORE_METRICS}
        method = {
            metric: float(run_index + 1) + (run_index - 1) * 0.25
            for metric in CORE_METRICS
        }
        records.append({**common, "method": "PCA", **baseline})
        records.append({**common, "method": "VP-SPCA", **method})
    return records


def test_runner_uses_reproducible_seed_manifest_and_shared_seed_data() -> None:
    first = run_video_benchmark(_smoke_config())
    repeated = run_video_benchmark(_smoke_config())

    assert verify_seed_manifest(first.seed_manifest)
    assert first.seed_manifest == repeated.seed_manifest
    assert first.seed_manifest["seeds"] == repeated.seed_manifest["seeds"]
    assert len(set(first.seed_manifest["seeds"])) == 2
    assert (
        first.seed_manifest["seeds"]
        != make_seed_manifest(2026, 2, namespace="video-spca-benchmark-v1")["seeds"]
    )
    assert len(first.records) == 4
    assert first.paired_summaries == repeated.paired_summaries

    manifest_sha256 = first.seed_manifest["manifest_sha256"]
    for run_index in range(2):
        rows = [row for row in first.records if row["run_index"] == run_index]
        assert {row["run"] for row in rows} == {run_index}
        assert [row["method"] for row in rows] == ["PCA", "VP-SPCA"]
        assert {row["data_sha256"] for row in rows} == {
            first.seed_manifest["run_provenance"][run_index]["data_sha256"]
        }
        assert {row["dataset_manifest_sha256"] for row in rows} == {
            first.seed_manifest["run_provenance"][run_index]["dataset_manifest_sha256"]
        }
        assert {row["seed_manifest_sha256"] for row in rows} == {manifest_sha256}
        assert {row["fixed_video_config_sha256"] for row in rows} == {
            first.seed_manifest["metadata"]["fixed_video_config_sha256"]
        }
        for row in rows:
            assert all(metric in row for metric in CORE_METRICS)
            assert isinstance(row["converged"], bool)
            assert int(row["n_iter"]) >= 1
            assert isinstance(row["status"], str)

    assert len(first.paired_summaries) == len(CORE_METRICS)
    assert len(first.method_summaries) == 2 * (len(CORE_METRICS) + 1)
    assert {summary["method"] for summary in first.method_summaries} == {
        "PCA",
        "VP-SPCA",
    }
    pca_clean = next(
        summary
        for summary in first.method_summaries
        if summary["method"] == "PCA"
        and summary["metric"] == "clean_reconstruction_error"
    )
    assert pca_clean["n_seeds"] == 2
    assert pca_clean["std"] >= 0.0
    runtime_summary = next(
        summary
        for summary in first.method_summaries
        if summary["method"] == "PCA" and summary["metric"] == "runtime_seconds"
    )
    assert runtime_summary["metric_scope"] == "descriptive_wall_clock_only"
    assert {summary["method"] for summary in first.paired_summaries} == {"VP-SPCA"}
    expected_status = (
        "ok"
        if all(
            bool(record["converged"])
            for record in first.records
            if record["method"] == "VP-SPCA"
        )
        else "unqualified_fixed_budget_runs_included"
    )
    assert {summary["bootstrap_ci95_status"] for summary in first.paired_summaries} == {
        expected_status
    }
    assert {summary["inference_qualified"] for summary in first.paired_summaries} == {
        expected_status == "ok"
    }
    assert {summary["bootstrap_unit"] for summary in first.paired_summaries} == {"seed"}
    assert all(
        summary["metric"] != "runtime_seconds" for summary in first.paired_summaries
    )


def test_complete_method_reports_and_nested_diagnostics_survive_json_and_csv(
    tmp_path: Path,
) -> None:
    zero_corruption_video = replace(
        _smoke_config().video_config,
        corruption_fraction=0.0,
    )
    result = run_video_benchmark(
        _smoke_config(
            video_config=zero_corruption_video,
            repeats=1,
            methods=("PCA", "RND-VP-SPCA", "Robust-VP-SPCA"),
            bootstrap_resamples=8,
        )
    )
    expected_report_fields = set(MethodReport.__dataclass_fields__)
    for record in result.records:
        assert expected_report_fields <= set(record)
        assert record["run"] == record["run_index"] == 0
        assert len(record["component_permutation"]) == 3
        assert len(record["component_signs"]) == 3

    randomized = next(
        record for record in result.records if record["method"] == "RND-VP-SPCA"
    )
    assert "sketch_residual_relative" in randomized["extra"]
    assert "full_diagnostic_step_policy" in randomized["extra"]
    assert "timings" in randomized["extra"]

    robust = next(
        record for record in result.records if record["method"] == "Robust-VP-SPCA"
    )
    assert robust["corruption_detection"] is not None
    assert robust["corruption_detection"]["roc_auc"] is None
    assert robust["corruption_detection"]["average_precision"] is None
    budget = robust["corruption_detection_at_budget"]
    assert budget is not None
    assert budget["roc_auc"] is None
    assert budget["average_precision"] is None
    expected_budget = round(
        result.config.video_config.corruption_fraction
        * result.config.video_config.n_timesteps
        * result.config.video_config.grid_size**2
    )
    assert budget["budget_top_k"] == expected_budget
    assert "estimated_outlier_fraction" in robust["extra"]

    json_path = tmp_path / "complete.json"
    csv_path = tmp_path / "complete.csv"
    write_results(result, output_json=json_path, output_csv=csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    saved_randomized = next(
        record for record in payload["records"] if record["method"] == "RND-VP-SPCA"
    )
    assert "timings" in saved_randomized["extra"]

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    csv_randomized = next(
        row
        for row in csv_rows
        if row["row_type"] == "per_seed" and row["method"] == "RND-VP-SPCA"
    )
    csv_robust = next(
        row
        for row in csv_rows
        if row["row_type"] == "per_seed" and row["method"] == "Robust-VP-SPCA"
    )
    assert "timings" in json.loads(csv_randomized["extra"])
    assert (
        json.loads(csv_robust["corruption_detection_at_budget"])["budget_top_k"]
        == expected_budget
    )
    assert csv_robust["corruption_detection_at_budget_budget_top_k"] == str(
        expected_budget
    )


def test_paired_bootstrap_is_row_order_invariant_and_reports_insufficient_pairs() -> (
    None
):
    records = _synthetic_records()
    first = summarize_paired_differences(
        records,
        bootstrap_resamples=101,
        bootstrap_seed=7,
    )
    shuffled = summarize_paired_differences(
        list(reversed(records)),
        bootstrap_resamples=101,
        bootstrap_seed=7,
    )
    assert first == shuffled

    clean = next(
        summary
        for summary in first
        if summary["metric"] == "clean_reconstruction_error"
    )
    assert clean["n_pairs"] == 3
    assert clean["mean_difference"] == pytest.approx(0.0)
    assert clean["std_difference"] == pytest.approx(0.25)
    assert clean["difference_direction"] == "method_minus_pca"
    assert clean["bootstrap_ci95_status"] == "ok"
    assert clean["inference_qualified"] is True
    assert clean["qualified_pairs"] == 3
    assert clean["bootstrap_ci95_low"] <= clean["mean_difference"]
    assert clean["bootstrap_ci95_high"] >= clean["mean_difference"]

    one_pair = summarize_paired_differences(
        _synthetic_records(n_pairs=1),
        bootstrap_resamples=25,
        bootstrap_seed=7,
    )
    assert {summary["bootstrap_ci95_status"] for summary in one_pair} == {
        "insufficient_pairs"
    }
    assert all(summary["bootstrap_ci95_low"] is None for summary in one_pair)
    assert all(summary["bootstrap_ci95_high"] is None for summary in one_pair)


def test_paired_bootstrap_retains_nonconverged_pairs_but_marks_unqualified() -> None:
    records = _synthetic_records()
    target = next(record for record in records if record["method"] == "VP-SPCA")
    target["converged"] = False
    summaries = summarize_paired_differences(
        records,
        bootstrap_resamples=101,
        bootstrap_seed=7,
    )
    assert {row["bootstrap_ci95_status"] for row in summaries} == {
        "unqualified_fixed_budget_runs_included"
    }
    assert {row["inference_qualified"] for row in summaries} == {False}
    assert {row["qualified_pairs"] for row in summaries} == {2}
    assert all(row["bootstrap_ci95_low"] is not None for row in summaries)


def test_pairing_rejects_duplicate_missing_and_mismatched_dataset_rows() -> None:
    records = _synthetic_records(n_pairs=2)
    with pytest.raises(ValueError, match="duplicate record"):
        summarize_paired_differences(
            [*records, dict(records[-1])],
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )

    with pytest.raises(ValueError, match="not paired with PCA"):
        summarize_paired_differences(
            records[:-1],
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )

    mismatched = [dict(record) for record in records]
    mismatched[-1]["dataset_manifest_sha256"] = "wrong-dataset"
    with pytest.raises(ValueError, match="different data"):
        summarize_paired_differences(
            mismatched,
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )

    for provenance_field in (
        "fixed_video_config_sha256",
        "seed_manifest_sha256",
    ):
        mixed_config = [dict(record) for record in records]
        for record in mixed_config:
            record[provenance_field] = "shared"
        mixed_config[-1][provenance_field] = "different"
        with pytest.raises(ValueError, match=provenance_field):
            summarize_paired_differences(
                mixed_config,
                bootstrap_resamples=10,
                bootstrap_seed=1,
            )

        missing_provenance = [dict(record) for record in records]
        missing_provenance[-1].pop(provenance_field)
        with pytest.raises(ValueError, match=f"contain {provenance_field}"):
            summarize_paired_differences(
                missing_provenance,
                bootstrap_resamples=10,
                bootstrap_seed=1,
            )

    missing_metric = [dict(record) for record in records]
    missing_metric[-1].pop("clean_reconstruction_error")
    with pytest.raises(ValueError, match="missing metric"):
        summarize_paired_differences(
            missing_metric,
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )

    nonfinite_metric = [dict(record) for record in records]
    nonfinite_metric[-1]["clean_reconstruction_error"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        summarize_paired_differences(
            nonfinite_metric,
            bootstrap_resamples=10,
            bootstrap_seed=1,
        )

    with pytest.raises(ValueError, match="at least 2"):
        summarize_paired_differences(
            records,
            bootstrap_resamples=1,
            bootstrap_seed=1,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid", "message"),
    [
        ("repeats", True, "repeats must be an integer"),
        ("repeats", 1.5, "repeats must be an integer"),
        ("base_seed", -1, "base_seed must be non-negative"),
        ("bootstrap_resamples", 0, "bootstrap_resamples must be positive"),
        ("bootstrap_resamples", 1, "bootstrap_resamples must be at least 2"),
        ("bootstrap_seed", True, "bootstrap_seed must be an integer"),
    ],
)
def test_runner_rejects_invalid_seed_and_bootstrap_controls(
    field_name: str,
    invalid: object,
    message: str,
) -> None:
    config = replace(_smoke_config(), **{field_name: invalid})
    with pytest.raises((TypeError, ValueError), match=message):
        run_video_benchmark(config)


def test_manifest_digest_binds_bootstrap_policy() -> None:
    first = run_video_benchmark(_smoke_config(repeats=1, bootstrap_resamples=8))
    changed = run_video_benchmark(
        _smoke_config(
            repeats=1,
            bootstrap_resamples=9,
            bootstrap_seed=92,
        )
    )

    assert first.seed_manifest["seeds"] == changed.seed_manifest["seeds"]
    assert (
        first.seed_manifest["run_provenance"] == changed.seed_manifest["run_provenance"]
    )
    assert (
        first.seed_manifest["manifest_sha256"]
        != changed.seed_manifest["manifest_sha256"]
    )
    policy = first.seed_manifest["metadata"]["bootstrap_policy"]
    assert policy["resamples"] == 8
    assert policy["seed"] == 91
    assert policy["runtime_included"] is False


def test_writer_is_opt_in_and_emits_strict_json_and_long_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = run_video_benchmark(_smoke_config(repeats=1, bootstrap_resamples=8))

    assert write_results(result) == {}
    assert list(tmp_path.iterdir()) == []

    json_path = tmp_path / "requested" / "video.json"
    paths = write_results(result, output_json=json_path)
    assert paths == {"json": json_path}
    assert json_path.is_file()
    assert not (json_path.parent / "video.csv").exists()
    payload = json.loads(
        json_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(f"non-standard JSON constant: {token}")
        ),
    )
    assert payload["evaluation_protocol"]["best_seed_selection"] is False
    assert payload["evaluation_protocol"]["ground_truth_usage"] == (
        "post_fit_scoring_and_alignment_only"
    )
    assert "none" in payload["evaluation_protocol"]["hyperparameter_selection"]
    assert payload["records"][0]["dataset_manifest_sha256"]
    assert {summary["method"] for summary in payload["method_summaries"]} == {
        "PCA",
        "VP-SPCA",
    }
    assert payload["paired_summaries"][0]["bootstrap_ci95_status"] == (
        "insufficient_pairs"
    )

    csv_path = tmp_path / "requested" / "video.csv"
    write_results(result, output_csv=csv_path)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["row_type"] for row in rows} == {
        "per_seed",
        "method_summary",
        "paired_summary",
    }
    assert {row["seed_manifest_sha256"] for row in rows} == {
        result.seed_manifest["manifest_sha256"]
    }
    forbidden_numeric_tokens = {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
    assert all(
        value.strip().lower() not in forbidden_numeric_tokens
        for row in rows
        for value in row.values()
    )

    with pytest.raises(ValueError, match="different paths"):
        write_results(
            result,
            output_json=tmp_path / "same-output",
            output_csv=tmp_path / "same-output",
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_writer_recursively_rejects_nonfinite_diagnostics_without_partial_outputs(
    tmp_path: Path,
    invalid: float,
) -> None:
    result = run_video_benchmark(_smoke_config(repeats=1, bootstrap_resamples=8))
    records = deepcopy(result.records)
    records[0]["extra"]["nested_nonfinite"] = {"value": invalid}
    tampered = replace(result, records=records)
    json_path = tmp_path / "nonfinite.json"
    csv_path = tmp_path / "nonfinite.csv"

    with pytest.raises(ValueError, match="number must be finite at records"):
        write_results(
            tampered,
            output_json=json_path,
            output_csv=csv_path,
        )

    assert not json_path.exists()
    assert not csv_path.exists()


def test_writer_rejects_nonfinite_config_manifest_records_and_summaries(
    tmp_path: Path,
) -> None:
    result = run_video_benchmark(_smoke_config(repeats=2, bootstrap_resamples=8))
    invalid_results: list[tuple[VideoBenchmarkResult, str]] = []

    invalid_config = replace(
        result.config,
        video_config=replace(result.config.video_config, alpha=float("nan")),
    )
    invalid_results.append(
        (replace(result, config=invalid_config), "alpha must be finite")
    )

    invalid_manifest = deepcopy(result.seed_manifest)
    invalid_manifest["metadata"]["nested_nonfinite"] = [float("inf")]
    invalid_results.append(
        (replace(result, seed_manifest=invalid_manifest), "number must be finite")
    )

    invalid_records = deepcopy(result.records)
    invalid_records[0]["clean_reconstruction_error"] = float("-inf")
    invalid_results.append(
        (replace(result, records=invalid_records), "number must be finite")
    )

    invalid_method_summaries = deepcopy(result.method_summaries)
    invalid_method_summaries[0]["mean"] = float("nan")
    invalid_results.append(
        (
            replace(result, method_summaries=invalid_method_summaries),
            "number must be finite",
        )
    )

    invalid_paired_summaries = deepcopy(result.paired_summaries)
    invalid_paired_summaries[0]["bootstrap_ci95_low"] = float("inf")
    invalid_results.append(
        (
            replace(result, paired_summaries=invalid_paired_summaries),
            "number must be finite",
        )
    )

    for index, (invalid_result, message) in enumerate(invalid_results):
        json_path = tmp_path / f"invalid-{index}.json"
        with pytest.raises(ValueError, match=message):
            write_results(invalid_result, output_json=json_path)
        assert not json_path.exists()


def test_writer_rejects_tampered_manifest_and_recomputed_summaries(
    tmp_path: Path,
) -> None:
    result = run_video_benchmark(_smoke_config(repeats=2, bootstrap_resamples=8))

    manifest = deepcopy(result.seed_manifest)
    manifest["metadata"]["bootstrap_policy"]["seed"] += 1
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        write_results(
            replace(result, seed_manifest=manifest),
            output_json=tmp_path / "manifest.json",
        )

    method_summaries = deepcopy(result.method_summaries)
    method_summaries[0]["mean"] += 1.0
    with pytest.raises(ValueError, match="method summaries do not match"):
        write_results(
            replace(result, method_summaries=method_summaries),
            output_json=tmp_path / "method.json",
        )

    paired_summaries = deepcopy(result.paired_summaries)
    paired_summaries[0]["mean_difference"] += 1.0
    with pytest.raises(ValueError, match="paired summaries do not match"):
        write_results(
            replace(result, paired_summaries=paired_summaries),
            output_json=tmp_path / "paired.json",
        )

    assert list(tmp_path.iterdir()) == []


def test_writer_exclusive_create_preserves_existing_outputs(
    tmp_path: Path,
) -> None:
    result = run_video_benchmark(_smoke_config(repeats=1, bootstrap_resamples=8))
    json_path = tmp_path / "exclusive.json"
    csv_path = tmp_path / "exclusive.csv"
    write_results(result, output_json=json_path, output_csv=csv_path)
    original_json = json_path.read_bytes()
    original_csv = csv_path.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_results(result, output_json=json_path, output_csv=csv_path)

    assert json_path.read_bytes() == original_json
    assert csv_path.read_bytes() == original_csv

    new_csv_path = tmp_path / "must-not-be-created.csv"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_results(result, output_json=json_path, output_csv=new_csv_path)
    assert json_path.read_bytes() == original_json
    assert not new_csv_path.exists()

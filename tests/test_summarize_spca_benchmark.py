from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rspca.datasets import canonical_sha256
from tools.summarize_spca_benchmark import (
    BASELINE_METHOD,
    CORRUPTION_METRICS,
    DATA_HASH_FIELDS,
    EXPECTED_METHODS,
    GENERAL_METRICS,
    summarize_spca_benchmarks,
    write_summary_outputs,
)


SEEDS = (101, 202, 303, 404)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _config(corruption_fraction: float) -> dict[str, Any]:
    return {
        "n_samples": 40,
        "n_features": 24,
        "rank": 3,
        "support_fraction": 0.1,
        "noise_std": 0.05,
        "corruption_fraction": corruption_fraction,
        "corruption_scale": 8.0,
        "corruption_distribution": "gaussian",
        "alphas": [0.1, 0.3],
        "selection": "matched-sparsity",
        "target_sparsity": 0.85,
        "repeats": len(SEEDS),
        "base_seed": 73001,
        "max_iter": 500,
        "tol": 1e-5,
        "oversample": 20,
        "n_power_iter": 3,
        "randomized_full_gradient": False,
        "randomized_finalize_full_iters": 300,
        "certified_residual_tolerance": 0.05,
        "certified_full_refine_iters": 500,
    }


def _metric_value(metric: str, *, method_index: int, seed_index: int) -> float:
    method_delta = 0.01 * method_index
    seed_delta = 0.002 * seed_index
    if metric == "relative_reconstruction_error":
        return 0.30 + method_delta + seed_delta
    if metric == "explained_variance_ratio":
        return 0.60 + method_delta + seed_delta
    if metric == "subspace_error":
        return 0.20 + method_delta + seed_delta
    if metric == "achieved_sparsity":
        return 0.85 + 0.002 * method_index
    if metric.startswith("component_"):
        return 0.70 + method_delta + seed_delta
    if metric.startswith("support_"):
        return 0.55 + method_delta + seed_delta
    if metric.startswith("outlier_"):
        return 0.50 + method_delta + seed_delta
    raise AssertionError(metric)


def _artifact(corruption_fraction: float) -> dict[str, Any]:
    regime = "clean" if corruption_fraction == 0.0 else "corrupt"
    config = _config(corruption_fraction)
    software = {
        "python": "3.10.12",
        "rspca": "0.2.0",
        "numpy": "2.2.6",
        "scipy": "1.15.3",
    }
    run_provenance: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    metrics = GENERAL_METRICS + (
        CORRUPTION_METRICS if corruption_fraction > 0.0 else ()
    )
    for run, seed in enumerate(SEEDS):
        clean_hash = _digest(f"shared-clean-{seed}")
        hashes = {
            "observed_sha256": (
                clean_hash if regime == "clean" else _digest(f"corrupt-observed-{seed}")
            ),
            "clean_uncorrupted_sha256": clean_hash,
            "low_rank_truth_sha256": _digest(f"shared-low-rank-{seed}"),
            "true_loadings_sha256": _digest(f"shared-loadings-{seed}"),
            "feature_support_sha256": _digest(f"shared-feature-support-{seed}"),
            "corruption_support_sha256": _digest(f"{regime}-corruption-support-{seed}"),
            "corruption_values_sha256": _digest(f"{regime}-corruption-values-{seed}"),
        }
        assert set(hashes) == set(DATA_HASH_FIELDS)
        run_data_sha256 = canonical_sha256(hashes)
        run_provenance.append(
            {
                "run": run,
                "seed": seed,
                **hashes,
                "run_data_sha256": run_data_sha256,
            }
        )
        for method_index, method in enumerate(EXPECTED_METHODS):
            record: dict[str, Any] = {
                "run": run,
                "seed": seed,
                "method": method,
                "selection": "matched-sparsity",
                "alpha": (
                    None
                    if method == BASELINE_METHOD
                    else config["alphas"][method_index % 2]
                ),
                "target_sparsity": config["target_sparsity"],
                "runtime_seconds": 0.2 + method_index + 0.1 * run,
                "converged": not (method == "vp_spca" and run == 0),
                "n_iter": 1 if method == BASELINE_METHOD else 12,
                "status": (
                    "closed_form" if method == BASELINE_METHOD else "fixture_status"
                ),
                "outlier_budget_fraction": corruption_fraction,
                "outlier_budget_k": int(
                    round(
                        corruption_fraction * config["n_samples"] * config["n_features"]
                    )
                ),
                "outlier_selection_policy": "predeclared_corruption_fraction_top_k",
                "centering_policy": (
                    "none_dgp_components_column_centered_before_corruption"
                ),
                "runtime_comparison_scope": ("exploratory_not_matvec_or_phase_matched"),
                "loading_postprocessing": (
                    "hard_threshold_to_predeclared_target_sparsity"
                    if method == BASELINE_METHOD
                    else "none"
                ),
                **hashes,
                "run_data_sha256": run_data_sha256,
            }
            for metric in metrics:
                record[metric] = _metric_value(
                    metric, method_index=method_index, seed_index=run
                )
            if regime == "clean":
                record.update(
                    {
                        "outlier_precision": 1.0,
                        "outlier_recall": 1.0,
                        "outlier_f1": 1.0,
                        "outlier_roc_auc": None,
                        "outlier_average_precision": None,
                    }
                )
            records.append(record)

    manifest: dict[str, Any] = {
        "manifest_version": 1,
        "namespace": "spca-benchmark-v2",
        "base_seed": config["base_seed"],
        "seeds": list(SEEDS),
        "metadata": {"config": config, "software_versions": software},
        "run_provenance": run_provenance,
        "software_versions": software,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return {
        "schema_version": 1,
        "config": config,
        "provenance": {"software_versions": software},
        "seed_manifest": manifest,
        "records": records,
    }


def _write_inputs(root: Path) -> tuple[Path, Path]:
    clean = root / "clean.json"
    corrupt = root / "corrupt.json"
    clean.write_text(
        json.dumps(_artifact(0.0), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    corrupt.write_text(
        json.dumps(_artifact(0.05), sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return clean, corrupt


def test_summary_is_paired_deterministic_and_binds_inputs(tmp_path: Path) -> None:
    clean, corrupt = _write_inputs(tmp_path)
    corrupt_payload = json.loads(corrupt.read_text(encoding="utf-8"))
    corrupt_payload["config"]["alphas"] = [0.2, 0.25, 0.3]
    corrupt_payload["seed_manifest"]["metadata"]["config"]["alphas"] = [
        0.2,
        0.25,
        0.3,
    ]
    for record in corrupt_payload["records"]:
        if record["method"] != BASELINE_METHOD:
            record["alpha"] = 0.25
    manifest = corrupt_payload["seed_manifest"]
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    corrupt.write_text(
        json.dumps(corrupt_payload, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary = summarize_spca_benchmarks(
        clean, corrupt, bootstrap_resamples=256, bootstrap_seed=19
    )
    repeated = summarize_spca_benchmarks(
        clean, corrupt, bootstrap_resamples=256, bootstrap_seed=19
    )
    assert summary == repeated
    assert summary["summary_sha256"] == canonical_sha256(
        {key: value for key, value in summary.items() if key != "summary_sha256"}
    )

    manifest = summary["analysis_manifest"]
    assert manifest["manifest_sha256"] == canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    assert manifest["bootstrap_policy"]["resamples"] == 256
    assert manifest["bootstrap_policy"]["runtime_included"] is False
    assert manifest["matched_sparsity_policy"] == {
        "target": 0.85,
        "alpha_grid_scope": "predeclared_per_corruption_regime",
        "grid_selection_evidence": "achieved_sparsity_only",
        "truth_metrics_used_for_grid_selection": False,
        "grids_bound_in_analysis_manifest": True,
    }
    assert (
        manifest["inputs"][0]["input_json_sha256"]
        == hashlib.sha256(clean.read_bytes()).hexdigest()
    )

    rows = summary["records"]
    assert len(rows) == len(EXPECTED_METHODS) * (
        len(GENERAL_METRICS) + len(GENERAL_METRICS + CORRUPTION_METRICS)
    )
    assert not any(
        row["regime"] == "clean" and row["metric"].startswith("outlier_")
        for row in rows
    )
    vp_relative = next(
        row
        for row in rows
        if row["regime"] == "clean"
        and row["method"] == "accelerated_vp_spca"
        and row["metric"] == "relative_reconstruction_error"
    )
    assert vp_relative["absolute_mean"] == pytest.approx(0.313)
    assert vp_relative["paired_mean_difference"] == pytest.approx(0.01)
    assert vp_relative["bootstrap_ci95_lower"] == pytest.approx(0.01)
    assert vp_relative["bootstrap_ci95_upper"] == pytest.approx(0.01)
    assert vp_relative["runtime_in_bootstrap"] is False
    assert vp_relative["bootstrap_ci95_status"] == "ok"
    assert vp_relative["inference_qualified"] is True

    vp_support = next(
        row
        for row in rows
        if row["regime"] == "corrupt"
        and row["method"] == "vp_spca"
        and row["metric"] == "support_f1"
    )
    assert vp_support["converged_count"] == 3
    assert vp_support["convergence_rate"] == pytest.approx(0.75)
    assert vp_support["inference_qualified"] is False
    assert vp_support["qualified_pairs"] == 3
    assert vp_support["unqualified_pairs"] == 1
    assert (
        vp_support["bootstrap_ci95_status"]
        == "unqualified_fixed_budget_runs_included"
    )


def test_outputs_are_strict_complete_and_never_overwritten(tmp_path: Path) -> None:
    clean, corrupt = _write_inputs(tmp_path)
    summary = summarize_spca_benchmarks(
        clean, corrupt, bootstrap_resamples=32, bootstrap_seed=7
    )
    output_json = tmp_path / "summary" / "summary.json"
    output_csv = tmp_path / "summary" / "summary.csv"
    write_summary_outputs(summary, output_json=output_json, output_csv=output_csv)

    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded == summary
    with output_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(summary["records"])
    assert all(value != "" for row in rows for value in row.values())
    original_json = output_json.read_bytes()
    original_csv = output_csv.read_bytes()
    with pytest.raises(FileExistsError):
        write_summary_outputs(summary, output_json=output_json, output_csv=output_csv)
    assert output_json.read_bytes() == original_json
    assert output_csv.read_bytes() == original_csv


@pytest.mark.parametrize(
    "failure", ["missing_method", "unpaired_hash", "missing_metric"]
)
def test_summary_rejects_incomplete_or_unpaired_records(
    tmp_path: Path, failure: str
) -> None:
    clean, corrupt = _write_inputs(tmp_path)
    payload = json.loads(corrupt.read_text(encoding="utf-8"))
    if failure == "missing_method":
        payload["records"].pop()
        match = "record matrix is incomplete"
    elif failure == "unpaired_hash":
        payload["records"][0]["run_data_sha256"] = _digest("wrong-run-data")
        match = "unpaired run_data_sha256"
    else:
        payload["records"][0].pop("support_f1")
        match = "missing required field"
    corrupt.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        summarize_spca_benchmarks(clean, corrupt, bootstrap_resamples=16)


def test_summary_rejects_nonstandard_nan_before_statistics(tmp_path: Path) -> None:
    clean, corrupt = _write_inputs(tmp_path)
    payload = json.loads(corrupt.read_text(encoding="utf-8"))
    payload["records"][0]["support_f1"] = float("nan")
    corrupt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="numeric constant is forbidden"):
        summarize_spca_benchmarks(clean, corrupt, bootstrap_resamples=16)

from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy import linalg

from experiments.video_decomposition_reference import (
    RobustDecompositionDataset,
    generate_figure6_dataset,
)
from experiments.video_robust_decomposition import (
    CONFIRMATION_CASES_NAME,
    CONFIRMATION_FIGURE_NAME,
    CONFIRMATION_FRAMES_NAME,
    CONFIRMATION_MANIFEST_NAME,
    CONFIRMATION_REPORT_NAME,
    DEVELOPMENT_RESULTS_NAME,
    DEVELOPMENT_MANIFEST_NAME,
    LEGACY_OUTLIER_TRUE_BUDGET_F1,
    LEGACY_ROBUST_SOURCE_SHA256,
    OUTLIER_NOMINAL_BUDGET_F1,
    SELECTION_LOCK_NAME,
    _without_runtime,
    audit_confirmation_output_directory,
    audit_video_study_output_directories,
    _build_cli_parser,
    _candidate_fit_sha256,
    _fit_legacy_dynamic_kappa_exact,
    _fit_locked_observed_arms,
    _paper_video_config,
    load_selection_lock,
    prepare_observed_case,
    run_confirmation,
    run_development,
    save_confirmation_output,
    save_development_output,
    select_objective_winner_index,
    smoke_study_config,
    validate_confirmation_output_directory,
    validate_development_output_directory,
    validate_study_config,
)
from rspca.datasets import canonical_sha256


def test_config_rejects_leakage_bad_schedule_and_unequal_budget() -> None:
    config = smoke_study_config()
    with pytest.raises(ValueError, match="overlap"):
        validate_study_config(
            replace(config, confirmation_seeds=(config.development_seeds[0],))
        )
    with pytest.raises(ValueError, match="previously viewed"):
        validate_study_config(replace(config, confirmation_seeds=(91007,)))
    with pytest.raises(ValueError, match="strictly descending"):
        validate_study_config(
            replace(config, continuation_kappa_multipliers=(1.0, 2.0))
        )
    with pytest.raises(ValueError, match="budgets must be equal"):
        validate_study_config(
            replace(config, continuation_stage_max_iters=(1, 2))
        )


def test_objective_selector_uses_only_target_objective_and_stable_tie() -> None:
    records = [
        {
            "final_target_objective_recomputed": 4.0,
            "clean_error": 0.0,
            "converged": True,
            "runtime_seconds": 0.0,
            "nontrivial": True,
        },
        {
            "final_target_objective_recomputed": 3.0,
            "clean_error": 999.0,
            "converged": False,
            "runtime_seconds": 999.0,
            "nontrivial": False,
        },
        {"final_target_objective_recomputed": 3.0, "clean_error": -1.0},
    ]
    assert select_objective_winner_index(records) == 1
    with pytest.raises(ValueError, match="non-finite"):
        select_objective_winner_index(
            [{"final_target_objective_recomputed": float("nan")}]
        )


def _literal_origin_main_legacy(
    observed: np.ndarray, config
) -> tuple[np.ndarray, np.ndarray]:
    _, singular_values, right_t = linalg.svd(observed, full_matrices=False)
    sigma_max = singular_values[0] if singular_values.size else 1.0
    normalized = observed / sigma_max
    right = right_t.T
    A = right[:, : config.n_components].copy()
    B = right[:, : config.n_components].copy()
    sparse = np.zeros_like(normalized)
    for _ in range(config.legacy_dynamic_max_iter):
        residual_for_gradient = normalized @ B @ A.T - normalized + sparse
        gradient = normalized.T @ (residual_for_gradient @ A)
        next_B = np.sign(B - gradient) * np.maximum(
            np.abs(B - gradient) - config.legacy_dynamic_alpha, 0.0
        )
        if linalg.norm(next_B - B, "fro") <= config.legacy_dynamic_tol:
            B = next_B
            break
        B = next_B
        matrix = (normalized - sparse).T @ (normalized @ B)
        left, _, right_transpose = linalg.svd(matrix, full_matrices=False)
        A = left @ right_transpose
        residual = normalized - normalized @ B @ A.T
        median = np.median(residual)
        sigma_hat = np.median(np.abs(residual - median)) / 0.6745 + 1e-12
        kappa = float(
            np.clip(
                config.legacy_huber_tau * sigma_hat,
                config.legacy_kappa_min,
                config.legacy_kappa_max,
            )
        )
        sparse = np.sign(residual) * np.maximum(np.abs(residual) - kappa, 0.0)
    return sigma_max * (normalized @ B @ A.T), sigma_max * sparse


def test_legacy_dynamic_baseline_exactly_matches_origin_main_operations() -> None:
    config = smoke_study_config()
    dataset = generate_figure6_dataset(_paper_video_config(config, 123))
    observed = prepare_observed_case(dataset)
    fit = _fit_legacy_dynamic_kappa_exact(observed, config)
    expected_reconstruction, expected_outliers = _literal_origin_main_legacy(
        dataset.observed, config
    )

    np.testing.assert_array_equal(fit.reconstruction, expected_reconstruction)
    np.testing.assert_array_equal(fit.outliers, expected_outliers)
    assert fit.record["historical_source"]["sha256"] == (
        LEGACY_ROBUST_SOURCE_SHA256
    )
    assert fit.record["changing_objective"] is True
    assert fit.record["eligible_for_fixed_target_objective_selection"] is False
    assert all(
        config.legacy_kappa_min <= value <= config.legacy_kappa_max
        for value in fit.record["kappa_trace"]
    )


def test_truth_isolation_and_equal_budget_B_only_handoffs() -> None:
    config = smoke_study_config()
    dataset = generate_figure6_dataset(_paper_video_config(config, 71001))
    changed_truth = RobustDecompositionDataset(
        clean=np.flip(dataset.clean, axis=1).copy(),
        observed=dataset.observed.copy(),
        corruption=np.zeros_like(dataset.corruption),
        corruption_support=np.zeros_like(dataset.corruption_support),
        time=dataset.time.copy(),
    )
    observed_a = prepare_observed_case(dataset)
    observed_b = prepare_observed_case(changed_truth)
    fits_a = _fit_locked_observed_arms(
        observed_a, alpha=0.003, target_kappa=0.001, config=config
    )
    fits_b = _fit_locked_observed_arms(
        observed_b, alpha=0.003, target_kappa=0.001, config=config
    )

    for name in ("direct", "continuation"):
        assert _candidate_fit_sha256(fits_a[name].selected) == (
            _candidate_fit_sha256(fits_b[name].selected)
        )
        assert fits_a[name].record["selected_start_id"] == (
            fits_b[name].record["selected_start_id"]
        )
    fairness = fits_a["fairness"]
    assert fairness["direct_initial_B_sha256"] == (
        fairness["continuation_initial_B_sha256"]
    )
    assert fairness["allocated_iterations_per_start"] == (
        config.max_iter_per_start
    )
    for record in fits_a["continuation"].candidate_records:
        assert record["allocated_iterations"] == config.max_iter_per_start
        for previous, current in zip(record["stages"], record["stages"][1:]):
            assert previous["final_B_sha256"] == current["initial_B_sha256"]
            assert current["A_and_S_reinitialized_for_stage"] is True


@pytest.fixture(scope="module")
def saved_two_phase_smoke(tmp_path_factory):
    os.environ.setdefault("MPLBACKEND", "Agg")
    root = tmp_path_factory.mktemp("paper-robust-study")
    development_dir = root / "development"
    confirmation_dir = root / "confirmation"
    config = smoke_study_config()
    development, lock = run_development(config, workers=1)
    save_development_output(development, lock, development_dir)
    validate_development_output_directory(development_dir)
    confirmation, cases = run_confirmation(
        development_dir / SELECTION_LOCK_NAME, workers=1
    )
    save_confirmation_output(
        confirmation,
        cases,
        development_dir / SELECTION_LOCK_NAME,
        confirmation_dir,
    )
    validate_confirmation_output_directory(confirmation_dir)
    return config, development_dir, confirmation_dir, confirmation


def test_two_phase_smoke_is_exact_closed_full_frame_and_worst_audited(
    saved_two_phase_smoke,
) -> None:
    config, development_dir, confirmation_dir, report = saved_two_phase_smoke
    assert (development_dir / DEVELOPMENT_MANIFEST_NAME).is_file()
    assert (confirmation_dir / CONFIRMATION_MANIFEST_NAME).is_file()
    assert (confirmation_dir / CONFIRMATION_FIGURE_NAME).read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    with (confirmation_dir / CONFIRMATION_FRAMES_NAME).open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len(config.confirmation_seeds) * 5 * config.n_timesteps
    assert {row["method"] for row in rows} == {
        "pca",
        "legacy_dynamic_kappa",
        "single_start",
        "direct_multistart",
        "continuation_multistart",
    }
    for display_row in report["figure_display"]["rows"][1:]:
        method = {
            "origin_main_legacy_cohort_worst": "legacy_dynamic_kappa",
            "single_start_cohort_worst": "single_start",
            "continuation_cohort_worst": "continuation_multistart",
        }[display_row["row_id"]]
        candidates = [row for row in rows if row["method"] == method]
        expected = sorted(
            candidates,
            key=lambda row: (
                -float(row["normalized_rmse"]),
                int(row["seed"]),
                int(row["frame_index"]),
            ),
        )[0]
        assert int(display_row["seed"]) == int(expected["seed"])
        assert int(display_row["frame_index"]) == int(expected["frame_index"])
    assert report["paper_scale"] is False
    assert "fixed-budget descriptive" in report["strict_qualification"][
        "claim_status"
    ]
    assert "continuation_minus_legacy_dynamic_kappa" in report[
        "paired_confirm_summaries"
    ]


def _refresh_confirmation_manifest(directory: Path, *names: str) -> None:
    manifest_path = directory / CONFIRMATION_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        path = directory / name
        manifest["files"][name] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh_development_manifest(directory: Path, *names: str) -> None:
    manifest_path = directory / DEVELOPMENT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in names:
        path = directory / name
        manifest["files"][name] = {
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _refresh_confirmation_report(report: dict, path: Path) -> None:
    evidence = {
        "selection_lock_sha256": report["selection_lock_sha256"],
        "cases": report["case_records"],
        "method_summaries": report["method_summaries"],
        "paired": report["paired_confirm_summaries"],
        "qualification": report["strict_qualification"],
        "figure_display": report["figure_display"],
    }
    if "metric_semantics" in report:
        evidence["metric_semantics"] = report["metric_semantics"]
    report["confirmation_evidence_sha256_without_runtime"] = canonical_sha256(
        _without_runtime(evidence)
    )
    report.pop("report_body_sha256", None)
    report["report_body_sha256"] = canonical_sha256(report)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _to_legacy_budget_keys(value):
    key_map = {
        OUTLIER_NOMINAL_BUDGET_F1: LEGACY_OUTLIER_TRUE_BUDGET_F1,
        "mean_nominal_budget_f1_difference": "mean_true_budget_f1_difference",
        "maximum_allowed_nominal_budget_f1_decrease": (
            "maximum_allowed_true_budget_f1_decrease"
        ),
        "nominal_budget_f1_noninferiority_pass": (
            "true_budget_f1_noninferiority_pass"
        ),
    }
    if isinstance(value, dict):
        return {
            key_map.get(key, key): _to_legacy_budget_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_to_legacy_budget_keys(item) for item in value]
    return value


def test_strict_audit_uses_nominal_budget_and_all_candidate_cost(
    saved_two_phase_smoke,
) -> None:
    _, development_dir, confirmation_dir, report = saved_two_phase_smoke
    audit = audit_video_study_output_directories(development_dir, confirmation_dir)
    confirmation = audit["confirmation"]
    assert confirmation["aggregate_and_gate_recomputation_passed"] is True
    assert confirmation["reported_claim_status"] == confirmation[
        "recomputed_claim_status"
    ]
    assert set(confirmation["reported_cost_accounting_mode_by_method"].values()) == {
        "all_candidate_strategy_cost"
    }
    for method in ("direct_multistart", "continuation_multistart"):
        summary = report["method_summaries"][method]
        assert summary["strategy_candidate_fit_cost"]["total_runtime_seconds"] > (
            summary["selected_fit_only_cost"]["total_runtime_seconds"]
        )
        assert set(
            summary["strategy_candidate_fit_cost"][
                "evaluated_candidate_count_per_seed"
            ]
        ) == {3}
    with (confirmation_dir / CONFIRMATION_CASES_NAME).open(
        encoding="utf-8", newline=""
    ) as stream:
        fieldnames = csv.DictReader(stream).fieldnames
    assert fieldnames is not None
    assert OUTLIER_NOMINAL_BUDGET_F1 in fieldnames
    assert LEGACY_OUTLIER_TRUE_BUDGET_F1 not in fieldnames


def test_version1_confirmation_shape_remains_strictly_auditable(
    saved_two_phase_smoke, tmp_path
) -> None:
    _, _, confirmation_dir, _ = saved_two_phase_smoke
    output = tmp_path / "legacy-v1-shape"
    shutil.copytree(confirmation_dir, output)
    report_path = output / CONFIRMATION_REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("metric_semantics")
    for summary in report["method_summaries"].values():
        selected = summary.pop("selected_fit_only_cost")
        summary.pop("strategy_candidate_fit_cost")
        summary["total_runtime_seconds"] = selected["total_runtime_seconds"]
        summary["total_actual_iterations"] = selected["total_actual_iterations"]
    report = _to_legacy_budget_keys(report)
    _refresh_confirmation_report(report, report_path)

    csv_path = output / CONFIRMATION_CASES_NAME
    with csv_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames is not None
        fields = [
            LEGACY_OUTLIER_TRUE_BUDGET_F1
            if name == OUTLIER_NOMINAL_BUDGET_F1
            else name
            for name in reader.fieldnames
        ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row[LEGACY_OUTLIER_TRUE_BUDGET_F1] = row.pop(
                OUTLIER_NOMINAL_BUDGET_F1
            )
            writer.writerow(row)
    _refresh_confirmation_manifest(
        output, CONFIRMATION_REPORT_NAME, CONFIRMATION_CASES_NAME
    )

    validate_confirmation_output_directory(output)
    audit = audit_confirmation_output_directory(output)
    assert set(audit["reported_cost_accounting_mode_by_method"].values()) == {
        "legacy_selected_fit_only"
    }
    assert audit["aggregate_and_gate_recomputation_passed"] is True


def test_resigned_false_confirmation_claim_is_rejected(
    saved_two_phase_smoke, tmp_path
) -> None:
    _, _, confirmation_dir, _ = saved_two_phase_smoke
    output = tmp_path / "false-claim"
    shutil.copytree(confirmation_dir, output)
    report_path = output / CONFIRMATION_REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["strict_qualification"]["strict_confirmatory_effect_gate_passed"] = True
    report["strict_qualification"]["claim_status"] = "strict_confirmatory_support"
    _refresh_confirmation_report(report, report_path)
    _refresh_confirmation_manifest(output, CONFIRMATION_REPORT_NAME)
    with pytest.raises(ValueError, match="qualification gate does not recompute"):
        validate_confirmation_output_directory(output)


def test_resigned_false_development_selection_is_rejected(
    saved_two_phase_smoke, tmp_path
) -> None:
    _, development_dir, _, _ = saved_two_phase_smoke
    output = tmp_path / "false-development-selection"
    shutil.copytree(development_dir, output)
    report_path = output / DEVELOPMENT_RESULTS_NAME
    lock_path = output / SELECTION_LOCK_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    lock = json.loads(lock_path.read_text(encoding="utf-8"))

    selected_index = int(report["selection"]["selected_parameter_order_index"])
    alternatives = [
        record
        for record in report["selection"]["aggregates"]
        if int(record["parameter_order_index"]) != selected_index
    ]
    assert alternatives
    false_winner = alternatives[0]
    report["selection"]["selected_alpha"] = false_winner["alpha"]
    report["selection"]["selected_kappa"] = false_winner["kappa"]
    report["selection"]["selected_parameter_order_index"] = false_winner[
        "parameter_order_index"
    ]

    evidence = canonical_sha256(
        _without_runtime(
            {
                "seed_results": report["seed_results"],
                "selection": report["selection"],
            }
        )
    )
    report["development_evidence_sha256_without_runtime"] = evidence
    lock["development_evidence_sha256_without_runtime"] = evidence
    lock["selected_parameters"] = {
        "alpha": false_winner["alpha"],
        "kappa": false_winner["kappa"],
    }
    lock.pop("selection_lock_sha256", None)
    lock["selection_lock_sha256"] = canonical_sha256(lock)
    report["selection_lock_sha256"] = lock["selection_lock_sha256"]
    report.pop("report_body_sha256", None)
    report["report_body_sha256"] = canonical_sha256(report)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_development_manifest(
        output, DEVELOPMENT_RESULTS_NAME, SELECTION_LOCK_NAME
    )

    with pytest.raises(ValueError, match="selection is not the implemented ranking"):
        validate_development_output_directory(output)


def test_lock_tamper_and_confirmation_override_surface_fail_closed(
    saved_two_phase_smoke, tmp_path
) -> None:
    _, development_dir, _, _ = saved_two_phase_smoke
    original = json.loads(
        (development_dir / SELECTION_LOCK_NAME).read_text(encoding="utf-8")
    )
    tampered = json.loads(json.dumps(original))
    first_source = next(iter(tampered["source_sha256"]))
    tampered["source_sha256"][first_source] = "0" * 64
    tampered.pop("selection_lock_sha256")
    tampered["selection_lock_sha256"] = canonical_sha256(tampered)
    tampered_path = tmp_path / "tampered-lock.json"
    tampered_path.write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source closure mismatch"):
        load_selection_lock(tampered_path, verify_source=True)

    parser = _build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "confirm",
                "--selection-lock",
                str(development_dir / SELECTION_LOCK_NAME),
                "--output-dir",
                str(tmp_path / "out"),
                "--alpha",
                "0.1",
            ]
        )


@pytest.mark.parametrize("attack", ["extra_directory", "deleted_payload"])
def test_confirmation_exact_closure_rejects_resigned_attacks(
    saved_two_phase_smoke, tmp_path, attack
) -> None:
    _, _, confirmation_dir, _ = saved_two_phase_smoke
    output = tmp_path / f"tampered-{attack}"
    shutil.copytree(confirmation_dir, output)
    manifest_path = output / CONFIRMATION_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if attack == "extra_directory":
        (output / "extra").mkdir()
    else:
        (output / CONFIRMATION_FIGURE_NAME).unlink()
        manifest["files"].pop(CONFIRMATION_FIGURE_NAME)
        manifest["required_payloads"].remove(CONFIRMATION_FIGURE_NAME)
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError):
        validate_confirmation_output_directory(output)


def test_saved_robust_results_preserve_comparisons_and_failed_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "results/video/robust_decomposition/confirmation"
    report = json.loads(
        (directory / "results.json").read_text(encoding="utf-8")
    )
    lock = json.loads(
        (directory / "selection_lock.json").read_text(encoding="utf-8")
    )
    seeds = report["confirmation_seeds"]
    assert len(seeds) == len(set(seeds)) == 20
    assert [case["seed"] for case in report["case_records"]] == seeds
    assert report["confirmation_truth_used_for_tuning"] is False
    assert report["confirmation_parameter_overrides_allowed"] is False
    assert report["locked_selected_parameters"] == lock["selected_parameters"]
    assert report["selection_lock_sha256"] == lock["selection_lock_sha256"]

    summaries = report["method_summaries"]
    assert set(summaries) == {
        "continuation_multistart", "direct_multistart", "legacy_dynamic_kappa",
        "pca", "single_start",
    }
    robust_metrics = summaries["continuation_multistart"]["metrics"]
    pca_metrics = summaries["pca"]["metrics"]
    robust_error = robust_metrics["clean_relative_reconstruction_error"]["mean"]
    pca_error = pca_metrics["clean_relative_reconstruction_error"]["mean"]
    assert robust_error == pytest.approx(0.3373443365552934, abs=1e-12)
    assert pca_error == pytest.approx(0.4332992716224669, abs=1e-12)
    assert (pca_error - robust_error) / pca_error == pytest.approx(
        0.22145187253113807, abs=1e-12
    )

    # Version-1 artifacts use true_budget for the nominal protocol budget.
    robust_f1 = robust_metrics[LEGACY_OUTLIER_TRUE_BUDGET_F1]["mean"]
    pca_f1 = pca_metrics[LEGACY_OUTLIER_TRUE_BUDGET_F1]["mean"]
    assert robust_f1 == pytest.approx(0.8584540686275662, abs=1e-12)
    assert pca_f1 == pytest.approx(0.9228835714567744, abs=1e-12)
    assert robust_f1 < pca_f1

    qualification = report["strict_qualification"]
    assert qualification["strict_confirmatory_effect_gate_passed"] is False
    assert qualification["claim_status"] == "confirmatory effect gate failed"
    gates = qualification["comparator_effect_gates"]
    assert gates["pca"]["passed"] is False
    assert gates["direct_multistart"]["passed"] is False
    assert gates["single_start"]["passed"] is False
    assert gates["legacy_dynamic_kappa"]["passed"] is True

    comparison = report["paired_confirm_summaries"][
        "continuation_minus_direct_multistart"
    ]["clean_relative_reconstruction_error"]
    assert comparison["mean_difference"] == pytest.approx(
        1.3371185128563389e-08, abs=1e-14
    )
    interval = comparison["bootstrap_mean_difference_ci"]
    assert interval["lower"] <= 0.0 <= interval["upper"]
    boundary = report["scientific_boundary"].lower()
    assert "no new pca algorithm" in boundary
    assert "no real-video generalization claim" in boundary

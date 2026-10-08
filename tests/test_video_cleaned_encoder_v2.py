from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments.video_cleaned_encoder_v2 import (
    AUDIT_KIND,
    FORMAL_ACKNOWLEDGEMENT,
    FROZEN_LOCAL_SOURCE_CLOSURE,
    _paper_config,
    audit_closed_output,
    fit_fixed_baseline_observed,
    load_and_validate_protocol,
    main,
    preflight_new_output_directory,
    run_paired_seed,
    smoke_config,
    summarize_cases,
    validate_audit_receipt,
    validate_closed_output,
    write_closed_output,
)
from experiments.video_spca_paper_reproduction import generate_figure6_dataset
from rspca.datasets import canonical_sha256
from experiments.video_spca_paper_robust_study import (
    RobustStudyConfig,
    _fit_candidate,
    prepare_observed_case,
)


def test_new_runner_reproduces_old_fixed_candidate_exactly() -> None:
    config = smoke_config()
    dataset = generate_figure6_dataset(_paper_config(config, config.seeds[0]))
    observed = prepare_observed_case(dataset)
    new_fit = fit_fixed_baseline_observed(observed, config)
    old_config = RobustStudyConfig(
        grid_size=config.grid_size,
        n_timesteps=config.n_timesteps,
        total_time=config.total_time,
        n_components=config.n_components,
        corruption_fraction=config.corruption_fraction,
        corruption_scale=config.corruption_scale,
        max_iter_per_start=config.max_iter,
        tol=config.tol,
        alpha_grid=(config.alpha,),
        kappa_grid=(config.kappa,),
        continuation_stage_max_iters=(4, config.max_iter - 4),
    )
    old_fit = _fit_candidate(
        observed,
        start_id="pca",
        arm_id="exact_reproduction_test",
        initial_B=observed.pca_B,
        alpha=config.alpha,
        target_kappa=config.kappa,
        kappa_schedule=(config.kappa,),
        stage_budgets=(config.max_iter,),
        config=old_config,
    )

    np.testing.assert_array_equal(new_fit.result.A, old_fit.result.A)
    np.testing.assert_array_equal(new_fit.result.B, old_fit.result.B)
    np.testing.assert_array_equal(new_fit.result.S, old_fit.result.S)
    assert new_fit.result.objectives == old_fit.result.objectives
    assert new_fit.result.stationarity == old_fit.result.stationarity
    assert new_fit.result.step_sizes == old_fit.result.step_sizes
    assert new_fit.result.converged == old_fit.result.converged


def test_paired_smoke_case_reports_all_required_metrics_and_stationarity() -> None:
    config = smoke_config()
    case = run_paired_seed(config, config.seeds[0])
    assert list(case["methods"]) == ["fixed_baseline", "cleaned_encoder_v2"]
    assert case["truth_isolation"]["paired_output_hashes_frozen_before_truth_metrics"]
    for method in case["methods"].values():
        truth = method["truth_metrics_after_fit_hash_freeze"]
        assert set(truth["full_frame_normalized_rmse"]) == {
            "n_frames",
            "median",
            "p95",
            "max",
            "worst_frame_index",
            "worst_frame_time",
        }
        assert "outlier_nominal_budget_f1" in truth["outlier_detection"]
        assert np.isfinite(method["solver"]["runtime_seconds"])
        assert np.isfinite(method["solver"]["final_stationarity"])
    v2_blocks = case["methods"]["cleaned_encoder_v2"]["solver"][
        "final_block_stationarity"
    ]
    assert set(v2_blocks) == {
        "B_prox_mapping",
        "B_prox_mapping_raw",
        "A_riemannian_kkt",
        "A_global_optimality_gap",
        "A_orthogonality",
        "S_prox_mapping",
        "S_prox_mapping_raw",
        "total",
    }
    summary = summarize_cases([case], config, run_kind="smoke")
    assert summary["claim_status"] == "SMOKE_PLUMBING_ONLY_NO_SCIENTIFIC_CLAIM"
    assert not summary["predeclared_success_gate"]["evaluated"]


def test_smoke_output_is_closed_and_refuses_overwrite(tmp_path) -> None:
    config = smoke_config()
    case = run_paired_seed(config, config.seeds[0])
    summary = summarize_cases([case], config, run_kind="smoke")
    output = tmp_path / "smoke"
    protocol_copy = {
        "protocol_kind": "SMOKE_ONLY_NOT_FORMAL_PROTOCOL",
        "formal_protocol_used": False,
        "formal_seeds_used": False,
    }
    manifest = write_closed_output(
        output,
        config=config,
        cases=[case],
        summary=summary,
        run_kind="smoke",
        protocol_copy=protocol_copy,
    )
    assert (
        validate_closed_output(output)["manifest_body_sha256"]
        == manifest["manifest_body_sha256"]
    )
    audit = audit_closed_output(output)
    assert audit["summary_recomputed"]
    assert audit["csv_recomputed"]
    with pytest.raises(FileExistsError, match="overwrite"):
        write_closed_output(
            output,
            config=config,
            cases=[case],
            summary=summary,
            run_kind="smoke",
            protocol_copy=protocol_copy,
        )


def test_formal_audit_receipt_is_bound_to_protocol_and_source(tmp_path) -> None:
    protocol = {
        "protocol_body_sha256": "protocol-digest",
        "source_sha256": {"solver.py": "source-digest"},
    }
    receipt = {
        "audit_kind": AUDIT_KIND,
        "status": "passed",
        "auditor_independent": True,
        "protocol_body_sha256": "wrong-protocol",
        "source_sha256": protocol["source_sha256"],
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="different protocol"):
        validate_audit_receipt(path, protocol)

    receipt["protocol_body_sha256"] = protocol["protocol_body_sha256"]
    receipt["source_sha256"] = {"solver.py": "wrong-source"}
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="different source"):
        validate_audit_receipt(path, protocol)


def test_fresh_seed_protocol_is_source_bound_and_not_self_authorizing() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root / "experiments/configs/video_cleaned_encoder_v2_fresh_seed_protocol.json"
    )
    # This preregistration is immutable historical evidence, not a promise
    # that today's source tree can launch the already completed experiment.
    protocol = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in protocol.items() if key != "protocol_body_sha256"}
    assert canonical_sha256(body) == protocol["protocol_body_sha256"]
    config = protocol["config"]
    assert tuple(config["seeds"]) == tuple(range(2026090101, 2026090121))
    assert (config["n_components"], config["alpha"], config["kappa"]) == (3, 0.0015, 0.001)
    assert protocol["formal_launch_state"] == "FROZEN_NOT_LAUNCHED"
    assert protocol["formal_launch_authorized_by_this_file"] is False
    assert protocol["audit_gate"]["current_receipt"] is None
    assert set(protocol["source_sha256"]) == set(FROZEN_LOCAL_SOURCE_CLOSURE)


def test_live_launch_gate_still_rejects_source_drift(tmp_path) -> None:
    from hashlib import sha256

    root = Path(__file__).resolve().parents[1]
    original = root / "experiments/configs/video_cleaned_encoder_v2_fresh_seed_protocol.json"
    protocol = json.loads(original.read_text(encoding="utf-8"))
    # A synthetic protocol, never a rewritten preregistration or launch.
    protocol["source_sha256"] = {
        name: sha256((root / name).read_bytes()).hexdigest()
        for name in FROZEN_LOCAL_SOURCE_CLOSURE
    }
    protocol["source_sha256"]["src/rspca/vp.py"] = "0" * 64
    protocol["protocol_body_sha256"] = canonical_sha256(
        {key: value for key, value in protocol.items() if key != "protocol_body_sha256"}
    )
    path = tmp_path / "synthetic-protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    with pytest.raises(ValueError, match="source closure digest mismatch: src/rspca/vp.py"):
        load_and_validate_protocol(path)


def test_completed_v2_artifact_preserves_the_frozen_protocol_and_negative_result() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "results/video_cleaned_encoder_robust_vp_spca_v2_20260901/formal_fresh_v1"
    manifest = validate_closed_output(output)
    # Independently retained result-audit anchor; do not re-sign old output.
    assert manifest["manifest_body_sha256"] == "8d5883b6a095d0f86a57b1383e5334d4890ba50469381c1fe92ea397f3c35100"
    original = json.loads((root / "experiments/configs/video_cleaned_encoder_v2_fresh_seed_protocol.json").read_text())
    assert json.loads((output / "protocol_copy.json").read_text()) == original
    audit = audit_closed_output(output)
    assert audit["seed_count"] == 20
    assert "negative" in audit["claim_status"]


def test_frozen_source_closure_matches_fresh_runtime_import_trace() -> None:
    root = Path(__file__).resolve().parents[1]
    probe = """
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
for path in (root, root / "src"):
    sys.path.insert(0, str(path))
importlib.import_module("experiments.video_cleaned_encoder_v2")
loaded = set()
for module in tuple(sys.modules.values()):
    source = getattr(module, "__file__", None)
    if source is None:
        continue
    try:
        relative = Path(source).resolve().relative_to(root)
    except ValueError:
        continue
    if relative.suffix == ".py":
        loaded.add(relative.as_posix())
print(json.dumps(sorted(loaded)))
"""
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(root)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    runtime_closure = set(json.loads(completed.stdout))
    assert runtime_closure == set(FROZEN_LOCAL_SOURCE_CLOSURE)
    assert "experiments/__init__.py" in runtime_closure
    assert "src/rspca/__init__.py" in runtime_closure


def test_formal_existing_output_fails_before_run_cases(tmp_path, monkeypatch) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    protocol = {
        "formal_output_dir": str(occupied),
        "protocol_body_sha256": "test-only",
        "source_sha256": {},
    }
    config = smoke_config()
    called = False

    def forbidden_run_cases(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("run_cases must not be called")

    monkeypatch.setattr(
        "experiments.video_cleaned_encoder_v2.load_and_validate_protocol",
        lambda path: (protocol, config),
    )
    monkeypatch.setattr(
        "experiments.video_cleaned_encoder_v2.validate_audit_receipt",
        lambda path, value: {},
    )
    monkeypatch.setattr(
        "experiments.video_cleaned_encoder_v2.run_cases", forbidden_run_cases
    )
    with pytest.raises(FileExistsError, match="occupied formal output"):
        main(
            [
                "formal",
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--audit-receipt",
                str(tmp_path / "receipt.json"),
                "--output-dir",
                str(occupied),
                "--acknowledge-formal-launch",
                FORMAL_ACKNOWLEDGEMENT,
            ]
        )
    assert not called
    with pytest.raises(FileExistsError, match="occupied formal output"):
        preflight_new_output_directory(occupied)
    audit_closed_output,

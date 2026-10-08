from __future__ import annotations

import ast
from hashlib import sha256
import inspect
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import experiments.video_cleaned_input_solver_truth_blind_smoke as smoke_module
from experiments.video_cleaned_input_solver_truth_blind_smoke import (
    AUDIT_KIND,
    OUTPUT_MANIFEST_NAME,
    PACK_KIND,
    PACK_MANIFEST_COPY_NAME,
    PACK_MANIFEST_NAME,
    PROTOCOL_COPY_NAME,
    PROTOCOL_KIND,
    PROVENANCE_COMMITMENT_KIND,
    PROVENANCE_VERIFICATION_STATUS,
    REPORT_NAME,
    RUN_ACKNOWLEDGEMENT,
    STATIC_AUDIT_KIND,
    MINIMUM_SOURCE_PATHS,
    _PROVENANCE_BOUNDARY,
    _EXPECTED_PROVENANCE_CHECKS,
    _STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS,
    _array_sha256,
    _canonical_sha256,
    _file_sha256,
    run_truth_blind_smoke,
)


def _strict_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _build_fixture(
    root: Path,
    *,
    extra_npz_key: bool = False,
    extra_case_field: bool = False,
    flip_exclusion: bool = False,
    wrong_shape: bool = False,
    zero_initial_B: bool = False,
    unnormalized_scale: bool = False,
    mismatched_provenance_case: bool = False,
    extra_provenance_check: bool = False,
    npz_archive_comment: bool = False,
) -> Path:
    source_map: dict[str, str] = {}
    for index, relative in enumerate(sorted(MINIMUM_SOURCE_PATHS)):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"VALUE = {index}\n", encoding="utf-8")
        source_map[relative] = sha256(source.read_bytes()).hexdigest()

    seed_scan = {
        "schema_version": 1,
        "scan_kind": (
            "video_cleaned_input_solver_v3_pre_freeze_semantic_seed_history_scan"
        ),
        "scan_timestamp": "2026-09-08T00:00:00+08:00",
        "git_head": "0" * 40,
        "candidate_bank": [17],
        "scan_roots": ["tests"],
        "excluded_paths": ["seed_scan.json"],
        "exact_candidate_regex": "\\b17\\b",
        "exact_scan_command": "test fixture",
        "semantic_seed_hits_before_freeze": 0,
        "candidate_outcomes_generated_before_freeze": 0,
        "selection_rule": "test-only fixed seed",
        "use_boundary": "truth-blind solver smoke test only",
        "seed_replacement_allowed": False,
    }
    seed_scan_path = root / "operations" / "seed_scan.json"
    _strict_write(seed_scan_path, seed_scan)

    pack_dir = root / "results" / "pack"
    pack_dir.mkdir(parents=True)
    observed = np.zeros((8 if wrong_shape else 64, 4), dtype=np.float64)
    sigma1 = 0.0
    if unnormalized_scale:
        observed[:4] = 2.0 * np.eye(4, dtype=np.float64)
        sigma1 = 2.0
    initial_B = (
        np.zeros((4, 3), dtype=np.float64)
        if zero_initial_B
        else np.eye(4, 3, dtype=np.float64)
    )
    case_path = pack_dir / "observed_case_17.npz"
    with case_path.open("xb") as handle:
        if extra_npz_key:
            np.savez(
                handle,
                observed_normalized=observed,
                initial_B=initial_B,
                forbidden=np.zeros(1),
            )
        else:
            np.savez(
                handle,
                observed_normalized=observed,
                initial_B=initial_B,
            )
    if npz_archive_comment:
        with ZipFile(case_path, "a") as archive:
            archive.comment = b"clean_truth_forbidden=123"

    audit_path = "operations/audit.json"
    protocol = {
        "schema_version": 1,
        "protocol_kind": PROTOCOL_KIND,
        "smoke_launch_state": "FROZEN_NOT_RUN",
        "smoke_launch_authorized_by_this_file": False,
        "data_config": {
            "grid_size": 8,
            "n_timesteps": 4,
            "total_time": 4.0,
            "n_components": 3,
            "corruption_fraction": 0.05,
            "corruption_scale": 3.0,
            "seeds": [17],
        },
        "solver_config": {
            "alpha": 0.0015,
            "kappa": 0.001,
            "max_updates": 3,
            "tol": 1e-8,
            "max_block_delay": 5,
            "backtracking_expand": 1.2,
        },
        "blind_pack": {
            "output_dir": "results/pack",
            "manifest_name": PACK_MANIFEST_NAME,
            "exact_npz_keys": ["observed_normalized", "initial_B"],
        },
        "smoke_output": {
            "output_dir": "results/smoke",
            "report_name": REPORT_NAME,
            "manifest_name": OUTPUT_MANIFEST_NAME,
        },
        "solver_gate": {
            "all_cases_converged": True,
            "complete_stationarity_at_most": 1e-8,
            "orthogonality_at_most": 1e-10,
            "monotonicity_tolerance_factor": 1e-10,
            "certificate_rtol": 1e-9,
            "certificate_atol": 1e-12,
            "maximum_backtracking_trials": 61,
            "no_bounded_delay_violation": True,
            "all_values_finite": True,
        },
        "audit_gate": {
            "required": True,
            "receipt_path": audit_path,
            "status_required": "passed",
            "auditor_independent_required": True,
            "receipt_must_match_protocol_body_sha256": True,
            "receipt_must_match_source_sha256": True,
            "static_receipt_path": "operations/static_audit.json",
            "static_receipt_required_before_blind_pack": True,
        },
        "fresh_seed_contract": {
            "seed_history_scan_path": "operations/seed_scan.json",
            "seed_history_scan_sha256": _file_sha256(seed_scan_path),
            "semantic_hits_before_freeze": 0,
            "candidate_outcomes_before_freeze": 0,
            "seed_replacement_allowed": False,
            "use_boundary": "truth-blind solver smoke test only",
        },
        "source_sha256": source_map,
    }
    protocol["protocol_body_sha256"] = _canonical_sha256(protocol)
    protocol_path = root / "protocol.json"
    _strict_write(protocol_path, protocol)

    checked_nonruntime: dict[str, str] = {}
    for index, relative in enumerate(sorted(_STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS)):
        evidence_path = root / relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        if not evidence_path.exists():
            evidence_path.write_text(f"test audit evidence {index}\n", encoding="utf-8")
        checked_nonruntime[relative] = _file_sha256(evidence_path)
    checked_nonruntime["operations/seed_scan.json"] = _file_sha256(seed_scan_path)
    static_receipt = {
        "schema_version": 1,
        "audit_kind": STATIC_AUDIT_KIND,
        "status": "passed",
        "auditor_independent": True,
        "auditor_id": "codex-subagent:test-static-auditor",
        "audit_timestamp": "2026-09-08T00:01:00+08:00",
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_sha256": source_map,
        "checked_nonruntime_sha256": checked_nonruntime,
        "focused_test_command": "pytest focused test fixture",
        "focused_test_count": 1,
        "focused_tests_passed": True,
        "blind_pack_absent_at_audit": True,
        "candidate_pack_generations_before_audit": 0,
        "candidate_outcomes_observed_before_audit": 0,
        "solver_smoke_runs_before_audit": 0,
        "statistical_or_formal_run_authorized": False,
    }
    static_receipt["receipt_body_sha256"] = _canonical_sha256(static_receipt)
    static_receipt_path = root / "operations" / "static_audit.json"
    _strict_write(static_receipt_path, static_receipt)

    case = {
        "seed": 17,
        "npz_name": case_path.name,
        "shape": list(observed.shape),
        "sigma1": sigma1,
        "observed_sha256": "0" * 64,
        "observed_normalized_sha256": _array_sha256(observed),
        "initial_B_sha256": _array_sha256(initial_B),
        "npz_sha256": _file_sha256(case_path),
        "npz_size_bytes": case_path.stat().st_size,
    }
    if extra_case_field:
        case["clean_truth_forbidden"] = [[123.0]]
    pack_manifest = {
        "schema_version": 1,
        "manifest_kind": PACK_KIND,
        "protocol_kind": PROTOCOL_KIND,
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_sha256": source_map,
        "data_config": protocol["data_config"],
        "solver_config": protocol["solver_config"],
        "exact_npz_keys": ["observed_normalized", "initial_B"],
        "case_count": 1,
        "cases": [case],
        "truth_exclusion": {
            "clean_written": False,
            "corruption_written": False,
            "corruption_support_written": False,
            "time_written": False,
            "downstream_solver_inputs_are_observed_only": True,
        },
        "static_audit_receipt_sha256": _file_sha256(static_receipt_path),
    }
    if flip_exclusion:
        pack_manifest["truth_exclusion"]["clean_written"] = True
    pack_manifest["manifest_body_sha256"] = _canonical_sha256(pack_manifest)
    _strict_write(pack_dir / PACK_MANIFEST_NAME, pack_manifest)

    commitment = {
        "schema_version": 1,
        "commitment_kind": PROVENANCE_COMMITMENT_KIND,
        "verification_status": PROVENANCE_VERIFICATION_STATUS,
        "not_a_final_audit_receipt": True,
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_map_sha256": _canonical_sha256({"source_sha256": source_map}),
        "fresh_seed_scan_file_sha256": _file_sha256(seed_scan_path),
        "static_audit_receipt_sha256": _file_sha256(static_receipt_path),
        "blind_pack_manifest_body_sha256": pack_manifest["manifest_body_sha256"],
        "blind_pack_manifest_file_sha256": _file_sha256(pack_dir / PACK_MANIFEST_NAME),
        "verifier_source_sha256": source_map[
            "experiments/audit_video_cleaned_input_solver_blind_pack_provenance.py"
        ],
        "case_count": 1,
        "ordered_cases": [
            {
                "seed": 17,
                "observed_sha256": (
                    "f" * 64 if mismatched_provenance_case else case["observed_sha256"]
                ),
                "observed_normalized_sha256": case["observed_normalized_sha256"],
                "initial_B_sha256": case["initial_B_sha256"],
                "sigma1": case["sigma1"],
            }
        ],
        "checks": dict(_EXPECTED_PROVENANCE_CHECKS),
        "boundary": dict(_PROVENANCE_BOUNDARY),
    }
    if extra_provenance_check:
        commitment["checks"]["forbidden_truth_side_channel"] = True
    commitment["commitment_body_sha256"] = _canonical_sha256(commitment)
    receipt = {
        "schema_version": 1,
        "audit_kind": AUDIT_KIND,
        "status": "passed",
        "auditor_independent": True,
        "auditor_id": "codex-subagent:test-provenance-auditor",
        "audit_timestamp": "2026-09-08T00:02:00+08:00",
        "provenance_verifier_command": "python -m experiments.audit_video_cleaned_input_solver_blind_pack_provenance",
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_sha256": source_map,
        "static_audit_receipt_sha256": _file_sha256(static_receipt_path),
        "candidate_outcomes_observed_before_audit": False,
        "formal_or_statistical_run_authorized": False,
        "static_audit_completed_before_blind_pack": True,
        "blind_pack_generated_after_static_audit": True,
        "blind_pack_provenance_verified": True,
        "blind_pack_provenance_commitment": commitment,
    }
    receipt["receipt_body_sha256"] = _canonical_sha256(receipt)
    _strict_write(root / audit_path, receipt)
    return protocol_path


def test_truth_blind_runner_retains_auditable_solver_only_pass(tmp_path: Path) -> None:
    protocol_path = _build_fixture(tmp_path)
    report = run_truth_blind_smoke(
        protocol_path,
        acknowledgement=RUN_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )

    assert report["status"] == "PASS_SOLVER_ONLY_TRUTH_BLIND_SMOKE"
    assert report["passed"] is True
    assert report["case_count"] == 1
    case = report["cases"][0]
    assert case["checks"] == {
        "finite": True,
        "objective_monotone": True,
        "orthogonality": True,
        "complete_certificate_recomputed": True,
        "bounded_delay": True,
        "backtracking_trials": True,
        "converged": True,
    }
    assert case["n_updates"] == 3
    assert [update["block"] for update in case["updates"]] == ["B", "A", "S"]
    output = tmp_path / "results" / "smoke"
    assert {entry.name for entry in output.iterdir()} == {
        REPORT_NAME,
        PROTOCOL_COPY_NAME,
        PACK_MANIFEST_COPY_NAME,
        OUTPUT_MANIFEST_NAME,
    }

    with pytest.raises(FileExistsError, match="create-once"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )


def test_truth_blind_runner_fails_closed_on_ack_and_npz_inventory(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path, extra_npz_key=True)
    with pytest.raises(ValueError, match="acknowledgement"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement="wrong",
            repo_root=tmp_path,
        )
    with pytest.raises(ValueError, match="whitelist"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "fixture_kwargs,match",
    [
        ({"extra_case_field": True}, "case keys"),
        ({"flip_exclusion": True}, "exclusion declaration"),
    ],
)
def test_truth_blind_runner_rejects_manifest_side_channels(
    tmp_path: Path,
    fixture_kwargs: dict[str, bool],
    match: str,
) -> None:
    protocol_path = _build_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(ValueError, match=match):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )


@pytest.mark.parametrize(
    "fixture_kwargs,match",
    [
        ({"wrong_shape": True}, "shape is invalid"),
        ({"zero_initial_B": True}, "observed-PCA"),
        ({"unnormalized_scale": True}, "sigma1 scale contract"),
    ],
)
def test_truth_blind_runner_rejects_semantically_forged_pack(
    tmp_path: Path,
    fixture_kwargs: dict[str, bool],
    match: str,
) -> None:
    protocol_path = _build_fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(ValueError, match=match):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "results" / "smoke").exists()


def test_truth_blind_runner_requires_independent_generator_commitment(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path, mismatched_provenance_case=True)
    with pytest.raises(ValueError, match="observed_sha256 mismatch"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "results" / "smoke").exists()


def test_truth_blind_runner_rejects_provenance_check_side_channel(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path, extra_provenance_check=True)
    with pytest.raises(ValueError, match="commitment checks changed"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "results" / "smoke").exists()


def test_truth_blind_runner_rejects_npz_container_side_channel(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path, npz_archive_comment=True)
    with pytest.raises(ValueError, match="canonical two-array container"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "results" / "smoke").exists()


def test_truth_blind_runner_binds_fresh_seed_scan(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path)
    scan_path = tmp_path / "operations" / "seed_scan.json"
    scan_path.write_text(scan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="seed history scan digest mismatch"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )

    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["candidate_bank"] = [18]
    _strict_write(scan_path, scan)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["fresh_seed_contract"]["seed_history_scan_sha256"] = _file_sha256(
        scan_path
    )
    protocol.pop("protocol_body_sha256")
    protocol["protocol_body_sha256"] = _canonical_sha256(protocol)
    _strict_write(protocol_path, protocol)
    with pytest.raises(ValueError, match="candidate bank differs"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )


def test_truth_blind_runner_retains_solver_exception_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = _build_fixture(tmp_path)

    def fail_solver(*args, **kwargs):
        raise RuntimeError("deliberate test failure")

    monkeypatch.setattr(
        smoke_module, "fit_certificate_balanced_cleaned_encoder", fail_solver
    )
    report = run_truth_blind_smoke(
        protocol_path,
        acknowledgement=RUN_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    assert report["passed"] is False
    assert report["status"] == "FAIL_SOLVER_ONLY_TRUTH_BLIND_SMOKE_RETAINED"
    assert report["declared_case_count"] == 1
    assert report["case_count"] == 1
    assert report["cases"][0]["exception_type"] == "RuntimeError"
    assert report["cases"][0]["exception_message"] == "deliberate test failure"
    assert (tmp_path / "results" / "smoke" / REPORT_NAME).is_file()


def test_truth_blind_runner_retains_nonfinite_gate_failure_as_json_safe_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = _build_fixture(tmp_path)

    monkeypatch.setattr(
        smoke_module,
        "_run_case",
        lambda *args, **kwargs: {
            "seed": 17,
            "runtime_seconds": float("nan"),
            "checks": {"finite": False},
        },
    )
    report = run_truth_blind_smoke(
        protocol_path,
        acknowledgement=RUN_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    assert report["passed"] is False
    assert report["status"] == "FAIL_SOLVER_ONLY_TRUTH_BLIND_SMOKE_RETAINED"
    case = report["cases"][0]
    assert case["runtime_seconds"] is None
    assert case["nonfinite_fields"] == ["$.runtime_seconds"]
    assert case["status"] == "non-finite numerical result retained"
    assert (tmp_path / "results" / "smoke" / REPORT_NAME).is_file()


def test_truth_blind_runner_outputs_only_pre_solver_validated_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = _build_fixture(tmp_path)
    pack_manifest_path = tmp_path / "results" / "pack" / PACK_MANIFEST_NAME
    audit_path = tmp_path / "operations" / "audit.json"
    expected_protocol = protocol_path.read_bytes()
    expected_pack_manifest = pack_manifest_path.read_bytes()
    expected_audit = audit_path.read_bytes()
    original_run_case = smoke_module._run_case

    def mutate_inputs_after_snapshot(*args, **kwargs):
        protocol_path.write_text("{}\n", encoding="utf-8")
        pack_manifest_path.write_text("{}\n", encoding="utf-8")
        audit_path.write_text("{}\n", encoding="utf-8")
        return original_run_case(*args, **kwargs)

    monkeypatch.setattr(smoke_module, "_run_case", mutate_inputs_after_snapshot)
    report = run_truth_blind_smoke(
        protocol_path,
        acknowledgement=RUN_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    assert report["passed"] is True
    output = tmp_path / "results" / "smoke"
    assert (output / PROTOCOL_COPY_NAME).read_bytes() == expected_protocol
    assert (output / PACK_MANIFEST_COPY_NAME).read_bytes() == expected_pack_manifest
    assert (
        report["independent_audit_receipt_sha256"] == sha256(expected_audit).hexdigest()
    )


def test_truth_blind_runner_rejects_protocol_and_output_symlinks(
    tmp_path: Path,
) -> None:
    protocol_path = _build_fixture(tmp_path)
    protocol_link = tmp_path / "protocol-link.json"
    protocol_link.symlink_to(protocol_path)
    with pytest.raises(ValueError, match="protocol path must not be a symlink"):
        run_truth_blind_smoke(
            protocol_link,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )

    smoke_link = tmp_path / "results" / "smoke"
    smoke_link.symlink_to(tmp_path / "results" / "smoke-real")
    with pytest.raises(ValueError, match="symlink components"):
        run_truth_blind_smoke(
            protocol_path,
            acknowledgement=RUN_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "results" / "smoke-real").exists()


def test_truth_blind_cli_returns_nonzero_for_retained_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        smoke_module,
        "run_truth_blind_smoke",
        lambda *args, **kwargs: {
            "status": "FAIL_SOLVER_ONLY_TRUTH_BLIND_SMOKE_RETAINED",
            "passed": False,
            "case_count": 1,
        },
    )
    assert (
        smoke_module.main(
            [
                "--protocol",
                "ignored.json",
                "--acknowledge-solver-smoke",
                RUN_ACKNOWLEDGEMENT,
            ]
        )
        == 1
    )


def test_truth_blind_runner_has_no_direct_generator_or_evaluator_import() -> None:
    source = inspect.getsource(smoke_module)
    tree = ast.parse(source)
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.add(node.module)
    assert "experiments.video_spca_paper_reproduction" not in imported_modules
    assert "generate_figure6_dataset" not in source
    assert "rspca.metrics" not in source
    assert "outlier_detection_metrics" not in source
    assert 'archive["clean"]' not in source
    assert 'archive["corruption"]' not in source
    assert 'archive["corruption_support"]' not in source
    assert "dataset.clean" not in source
    assert "dataset.corruption" not in source

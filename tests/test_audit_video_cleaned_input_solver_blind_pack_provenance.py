from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
from scipy import linalg

import experiments.audit_video_cleaned_input_solver_blind_pack_provenance as auditor
from experiments.audit_video_cleaned_input_solver_blind_pack_provenance import (
    COMMITMENT_KIND,
    RUNTIME_SOURCE_PATHS,
    audit_blind_pack_provenance,
)
from experiments.prepare_video_cleaned_input_solver_blind_pack import (
    MANIFEST_NAME,
    PREPARATION_ACKNOWLEDGEMENT,
    load_and_validate_protocol,
    prepare_blind_pack,
    validate_blind_pack,
)


TEST_SEED = 17


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _strict_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _source_map(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    source_root = Path(auditor.__file__).resolve().parents[1]
    for relative in sorted(RUNTIME_SOURCE_PATHS):
        source = root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes((source_root / relative).read_bytes())
        result[relative] = _file_sha256(source)
    return result


def _checked_nonruntime_map(
    root: Path,
    *,
    seed_scan_relative: str,
) -> dict[str, str]:
    source_root = Path(auditor.__file__).resolve().parents[1]
    result: dict[str, str] = {}
    for relative in sorted(auditor.STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS):
        evidence = root / relative
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_bytes((source_root / relative).read_bytes())
        result[relative] = _file_sha256(evidence)
    result[seed_scan_relative] = _file_sha256(root / seed_scan_relative)
    return result


def _write_static_audit_receipt(
    root: Path,
    protocol: dict[str, object],
) -> Path:
    fresh_seed_contract = protocol["fresh_seed_contract"]
    assert isinstance(fresh_seed_contract, dict)
    seed_scan_relative = fresh_seed_contract["seed_history_scan_path"]
    assert isinstance(seed_scan_relative, str)
    receipt = {
        "schema_version": 1,
        "audit_kind": auditor.STATIC_AUDIT_KIND,
        "status": "passed",
        "auditor_independent": True,
        "auditor_id": "test-auditor",
        "audit_timestamp": "2026-09-08T00:01:00+08:00",
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_sha256": protocol["source_sha256"],
        "checked_nonruntime_sha256": _checked_nonruntime_map(
            root, seed_scan_relative=seed_scan_relative
        ),
        "focused_test_command": "pytest focused seed-17 verifier fixture",
        "focused_test_count": 1,
        "focused_tests_passed": True,
        "blind_pack_absent_at_audit": True,
        "candidate_pack_generations_before_audit": 0,
        "candidate_outcomes_observed_before_audit": 0,
        "solver_smoke_runs_before_audit": 0,
        "statistical_or_formal_run_authorized": False,
    }
    receipt["receipt_body_sha256"] = auditor._canonical_sha256(receipt)
    audit_gate = protocol["audit_gate"]
    assert isinstance(audit_gate, dict)
    receipt_relative = audit_gate["static_receipt_path"]
    assert isinstance(receipt_relative, str)
    receipt_path = root / receipt_relative
    _strict_write(receipt_path, receipt)
    return receipt_path


def _build_protocol(root: Path) -> Path:
    scan = {
        "schema_version": 1,
        "scan_kind": (
            "video_cleaned_input_solver_v3_pre_freeze_semantic_seed_history_scan"
        ),
        "scan_timestamp": "2026-09-08T00:00:00+08:00",
        "git_head": "0" * 40,
        "candidate_bank": [TEST_SEED],
        "scan_roots": ["tests"],
        "excluded_paths": ["operations/seed_scan.json"],
        "exact_candidate_regex": "\\b17\\b",
        "exact_scan_command": "test fixture only",
        "semantic_seed_hits_before_freeze": 0,
        "candidate_outcomes_generated_before_freeze": 0,
        "selection_rule": "test-only fixed seed",
        "use_boundary": "truth-blind solver smoke test only",
        "seed_replacement_allowed": False,
    }
    scan_path = root / "operations" / "seed_scan.json"
    _strict_write(scan_path, scan)
    protocol = {
        "schema_version": 1,
        "protocol_kind": auditor.PROTOCOL_KIND,
        "smoke_launch_state": auditor.PROTOCOL_STATE,
        "smoke_launch_authorized_by_this_file": False,
        "data_config": {
            "grid_size": 8,
            "n_timesteps": 12,
            "total_time": 6.0,
            "n_components": 3,
            "corruption_fraction": 0.05,
            "corruption_scale": 3.0,
            "seeds": [TEST_SEED],
        },
        "solver_config": {
            "alpha": 0.0015,
            "kappa": 0.001,
            "max_updates": 12,
            "tol": 1e-6,
            "max_block_delay": 5,
            "backtracking_expand": 1.2,
        },
        "blind_pack": {
            "output_dir": "results/blind-pack",
            "manifest_name": MANIFEST_NAME,
            "exact_npz_keys": list(auditor.EXACT_NPZ_KEYS),
        },
        "audit_gate": {
            "required": True,
            "receipt_path": "operations/final_audit.json",
            "status_required": "passed",
            "auditor_independent_required": True,
            "receipt_must_match_protocol_body_sha256": True,
            "receipt_must_match_source_sha256": True,
            "static_receipt_path": "operations/static_audit.json",
            "static_receipt_required_before_blind_pack": True,
        },
        "fresh_seed_contract": {
            "seed_history_scan_path": "operations/seed_scan.json",
            "seed_history_scan_sha256": _file_sha256(scan_path),
            "semantic_hits_before_freeze": 0,
            "candidate_outcomes_before_freeze": 0,
            "seed_replacement_allowed": False,
            "use_boundary": "truth-blind solver smoke test only",
        },
        "source_sha256": _source_map(root),
    }
    protocol["protocol_body_sha256"] = auditor._canonical_sha256(protocol)
    path = root / "protocol.json"
    _strict_write(path, protocol)
    _write_static_audit_receipt(root, protocol)
    return path


def _prepare(root: Path) -> Path:
    protocol_path = _build_protocol(root)
    prepare_blind_pack(
        protocol_path,
        acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
        repo_root=root,
    )
    return protocol_path


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generator_replay_returns_deterministic_unsigned_commitments_without_writes(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    before = _tree_hashes(root)
    first = audit_blind_pack_provenance(protocol_path, repo_root=root)
    second = audit_blind_pack_provenance(protocol_path, repo_root=root)
    after = _tree_hashes(root)

    assert first == second
    assert before == after
    assert first["commitment_kind"] == COMMITMENT_KIND
    assert first["verification_status"] == "PASS_GENERATOR_REPLAY_COMMITMENTS"
    assert first["not_a_final_audit_receipt"] is True
    assert first["case_count"] == 1
    assert [case["seed"] for case in first["ordered_cases"]] == [TEST_SEED]
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    static_receipt_path = root / protocol["audit_gate"]["static_receipt_path"]
    manifest_path = root / protocol["blind_pack"]["output_dir"] / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert first["static_audit_receipt_sha256"] == _file_sha256(static_receipt_path)
    assert (
        manifest["static_audit_receipt_sha256"] == first["static_audit_receipt_sha256"]
    )
    commitment_body = dict(first)
    reported_commitment_sha = commitment_body.pop("commitment_body_sha256")
    assert auditor._canonical_sha256(commitment_body) == reported_commitment_sha
    assert first["verifier_source_sha256"] == _file_sha256(
        Path(auditor.__file__).resolve()
    )
    assert len(RUNTIME_SOURCE_PATHS) == 22
    assert auditor.VERIFIER_SOURCE_PATH in RUNTIME_SOURCE_PATHS
    assert first["checks"]["runtime_source_closure_22_files"] is True
    assert first["checks"]["static_audit_receipt_bound"] is True
    assert first["checks"]["canonical_npz_whole_bytes"] is True
    assert set(first["ordered_cases"][0]) == {
        "seed",
        "observed_sha256",
        "observed_normalized_sha256",
        "initial_B_sha256",
        "sigma1",
    }
    assert first["boundary"] == {
        "solver_fits": 0,
        "statistical_metrics": 0,
        "truth_metrics": 0,
        "artifact_filesystem_writes": 0,
        "final_audit_receipts_issued": 0,
    }
    with pytest.raises(FileExistsError, match="create-once"):
        prepare_blind_pack(
            protocol_path,
            acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
            repo_root=root,
        )


def test_generator_replay_rejects_fully_self_digested_arbitrary_pack(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    validated = load_and_validate_protocol(protocol_path, repo_root=root)
    manifest_path = validated.output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    shape = tuple(case["shape"])

    forged = np.zeros(shape, dtype=np.float64)
    diagonal = np.linspace(1.0, 0.2, min(shape))
    forged[: len(diagonal), : len(diagonal)] = np.diag(diagonal)
    _, _, right_t = linalg.svd(
        forged,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    forged_B = np.ascontiguousarray(right_t[:3].T, dtype=np.float64)
    forged_sigma1 = 73.0
    forged_raw_observed = np.ascontiguousarray(forged * forged_sigma1, dtype=np.float64)
    case_path = validated.output_dir / case["npz_name"]
    with case_path.open("wb") as handle:
        np.savez(handle, observed_normalized=forged, initial_B=forged_B)
    case.update(
        {
            "sigma1": forged_sigma1,
            "observed_sha256": auditor._array_sha256(forged_raw_observed),
            "observed_normalized_sha256": auditor._array_sha256(forged),
            "initial_B_sha256": auditor._array_sha256(forged_B),
            "npz_sha256": _file_sha256(case_path),
            "npz_size_bytes": case_path.stat().st_size,
        }
    )
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = auditor._canonical_sha256(manifest)
    _strict_write(manifest_path, manifest)

    # The original internal validator proves only self-consistency.
    validate_blind_pack(validated.output_dir, protocol=validated)
    with pytest.raises(ValueError, match="generator replay observed hash mismatch"):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


@pytest.mark.parametrize(
    "target", ["protocol", "manifest", "seed_scan", "static_receipt"]
)
def test_replay_auditor_rejects_digest_chain_tampering_before_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if target == "protocol":
        protocol["data_config"]["total_time"] = 7.0
        _strict_write(protocol_path, protocol)
    elif target == "manifest":
        manifest_path = root / protocol["blind_pack"]["output_dir"] / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["case_count"] = 2
        _strict_write(manifest_path, manifest)
    elif target == "seed_scan":
        scan_path = root / protocol["fresh_seed_contract"]["seed_history_scan_path"]
        scan_path.write_text(
            scan_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
    else:
        receipt_path = root / protocol["audit_gate"]["static_receipt_path"]
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["auditor_id"] = "tampered-without-redigest"
        _strict_write(receipt_path, receipt)

    def forbidden_replay(*args, **kwargs):
        raise AssertionError("digest tamper must fail before generator replay")

    monkeypatch.setattr(auditor, "generate_figure6_dataset", forbidden_replay)
    expected = {
        "protocol": "protocol body digest mismatch",
        "manifest": "manifest body digest mismatch",
        "seed_scan": "seed history scan digest mismatch",
        "static_receipt": "static audit receipt body digest mismatch",
    }[target]
    with pytest.raises(ValueError, match=expected):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


def test_replay_auditor_rejects_missing_self_from_exact_22_file_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    del protocol["source_sha256"][auditor.VERIFIER_SOURCE_PATH]
    protocol["protocol_body_sha256"] = auditor._canonical_sha256(
        auditor._without_digest(protocol, "protocol_body_sha256")
    )
    _strict_write(protocol_path, protocol)

    monkeypatch.setattr(
        auditor,
        "generate_figure6_dataset",
        lambda *args, **kwargs: pytest.fail("source closure must fail before replay"),
    )
    with pytest.raises(ValueError, match="22-file runtime closure"):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


def test_contained_rejects_an_intermediate_symlink_even_when_target_stays_inside(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    (root / "real").mkdir()
    (root / "real" / "evidence.json").write_text("{}\n", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "link").symlink_to("../real", target_is_directory=True)

    with pytest.raises(ValueError, match="must not contain symlink components"):
        auditor._contained(
            root,
            Path("nested/link/evidence.json"),
            name="adversarial intermediate path",
        )


def test_replay_auditor_rejects_noncanonical_npz_with_valid_array_and_file_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = root / protocol["blind_pack"]["output_dir"] / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    case_path = manifest_path.parent / case["npz_name"]
    with np.load(case_path, allow_pickle=False) as archive:
        normalized = np.ascontiguousarray(archive["observed_normalized"])
        initial_B = np.ascontiguousarray(archive["initial_B"])
    with case_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            observed_normalized=normalized,
            initial_B=initial_B,
        )
    case["npz_sha256"] = _file_sha256(case_path)
    case["npz_size_bytes"] = case_path.stat().st_size
    manifest["manifest_body_sha256"] = auditor._canonical_sha256(
        auditor._without_digest(manifest, "manifest_body_sha256")
    )
    _strict_write(manifest_path, manifest)

    monkeypatch.setattr(
        auditor,
        "generate_figure6_dataset",
        lambda *args, **kwargs: pytest.fail("canonical NPZ check must precede replay"),
    )
    with pytest.raises(ValueError, match="canonical two-array container"):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("empty_auditor", "nonempty auditor_id"),
        ("zero_tests", "focused_test_count must be at least 1"),
        ("failed_tests", "focused tests did not pass"),
        ("pack_present", "did not establish blind-pack absence"),
        (
            "outcome_seen",
            "candidate_outcomes_observed_before_audit=0",
        ),
        ("protocol_unbound", "receipt protocol digest mismatch"),
        ("source_unbound", "receipt source map mismatch"),
        ("missing_evidence", "lacks required files"),
        ("bad_evidence_hash", "nonruntime digest mismatch"),
        ("extra_receipt_key", "static audit receipt keys changed"),
    ],
)
def test_replay_auditor_rejects_self_digested_but_invalid_static_audit_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected: str,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt_path = root / protocol["audit_gate"]["static_receipt_path"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "empty_auditor":
        receipt["auditor_id"] = "  "
    elif mutation == "zero_tests":
        receipt["focused_test_count"] = 0
    elif mutation == "failed_tests":
        receipt["focused_tests_passed"] = False
    elif mutation == "pack_present":
        receipt["blind_pack_absent_at_audit"] = False
    elif mutation == "outcome_seen":
        receipt["candidate_outcomes_observed_before_audit"] = 1
    elif mutation == "protocol_unbound":
        receipt["protocol_body_sha256"] = "0" * 64
    elif mutation == "source_unbound":
        receipt["source_sha256"] = {}
    elif mutation == "missing_evidence":
        missing = "docs/video_cleaned_input_solver_v3_novelty_audit.md"
        del receipt["checked_nonruntime_sha256"][missing]
    elif mutation == "bad_evidence_hash":
        target = "docs/video_cleaned_input_solver_v3_novelty_audit.md"
        receipt["checked_nonruntime_sha256"][target] = "0" * 64
    else:
        receipt["unexpected_outcome_field"] = 0
    receipt["receipt_body_sha256"] = auditor._canonical_sha256(
        auditor._without_digest(receipt, "receipt_body_sha256")
    )
    _strict_write(receipt_path, receipt)

    monkeypatch.setattr(
        auditor,
        "generate_figure6_dataset",
        lambda *args, **kwargs: pytest.fail("invalid static receipt must fail early"),
    )
    with pytest.raises((TypeError, ValueError), match=expected):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


def test_manifest_binds_exact_static_receipt_file_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    receipt_path = root / protocol["audit_gate"]["static_receipt_path"]
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )

    monkeypatch.setattr(
        auditor,
        "generate_figure6_dataset",
        lambda *args, **kwargs: pytest.fail("receipt binding must fail before replay"),
    )
    with pytest.raises(
        ValueError, match="blind-pack static audit receipt digest mismatch"
    ):
        audit_blind_pack_provenance(protocol_path, repo_root=root)


def test_protocol_cannot_disable_static_audit_before_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    protocol_path = _prepare(root)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["audit_gate"]["static_receipt_required_before_blind_pack"] = False
    protocol["protocol_body_sha256"] = auditor._canonical_sha256(
        auditor._without_digest(protocol, "protocol_body_sha256")
    )
    _strict_write(protocol_path, protocol)

    monkeypatch.setattr(
        auditor,
        "generate_figure6_dataset",
        lambda *args, **kwargs: pytest.fail("disabled static gate must fail early"),
    )
    with pytest.raises(ValueError, match="must remain required before blind pack"):
        audit_blind_pack_provenance(protocol_path, repo_root=root)

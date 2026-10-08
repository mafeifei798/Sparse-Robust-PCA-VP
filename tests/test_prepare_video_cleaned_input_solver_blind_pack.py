from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
from scipy import linalg

from experiments.prepare_video_cleaned_input_solver_blind_pack import (
    EXACT_NPZ_KEYS,
    MANIFEST_NAME,
    MINIMUM_SOURCE_PATHS,
    PREPARATION_ACKNOWLEDGEMENT,
    PROTOCOL_KIND,
    PROTOCOL_STATE,
    STATIC_AUDIT_KIND,
    STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS,
    _paper_config,
    load_and_validate_protocol,
    prepare_blind_pack,
    validate_blind_pack,
)
from experiments.video_spca_paper_reproduction import generate_figure6_dataset
from rspca.datasets import array_sha256, canonical_sha256


TEST_SEED = 17


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_source_closure(root: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for index, relative in enumerate(sorted(MINIMUM_SOURCE_PATHS)):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test source {index}\n", encoding="utf-8")
        output[relative] = _file_sha256(path)
    return output


def _protocol_value(root: Path, *, output_dir: str = "blind-output") -> dict:
    seed_scan = {
        "schema_version": 1,
        "scan_kind": (
            "video_cleaned_input_solver_v3_pre_freeze_semantic_seed_history_scan"
        ),
        "scan_timestamp": "2026-09-08T00:00:00+08:00",
        "git_head": "0" * 40,
        "candidate_bank": [TEST_SEED],
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
    seed_scan_path.parent.mkdir(parents=True, exist_ok=True)
    seed_scan_path.write_text(
        json.dumps(seed_scan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "schema_version": 1,
        "protocol_kind": PROTOCOL_KIND,
        "smoke_launch_state": PROTOCOL_STATE,
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
            "output_dir": output_dir,
            "manifest_name": MANIFEST_NAME,
            "exact_npz_keys": list(EXACT_NPZ_KEYS),
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
            "seed_history_scan_sha256": _file_sha256(seed_scan_path),
            "semantic_hits_before_freeze": 0,
            "candidate_outcomes_before_freeze": 0,
            "seed_replacement_allowed": False,
            "use_boundary": "truth-blind solver smoke test only",
        },
        "source_sha256": _write_source_closure(root),
        "truth_metrics_forbidden": [
            "clean",
            "corruption",
            "corruption_support",
            "time",
        ],
    }
    protocol["protocol_body_sha256"] = canonical_sha256(protocol)
    return protocol


def _write_protocol(
    root: Path,
    protocol: dict,
    name: str = "protocol.json",
    *,
    write_static_receipt: bool = True,
) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            protocol,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if write_static_receipt:
        checked_nonruntime: dict[str, str] = {}
        for index, relative in enumerate(sorted(STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS)):
            evidence_path = root / relative
            evidence_path.parent.mkdir(parents=True, exist_ok=True)
            if not evidence_path.exists():
                evidence_path.write_text(
                    f"test audit evidence {index}\n", encoding="utf-8"
                )
            checked_nonruntime[relative] = _file_sha256(evidence_path)
        seed_scan_relative = protocol["fresh_seed_contract"]["seed_history_scan_path"]
        checked_nonruntime[seed_scan_relative] = _file_sha256(root / seed_scan_relative)
        receipt = {
            "schema_version": 1,
            "audit_kind": STATIC_AUDIT_KIND,
            "status": "passed",
            "auditor_independent": True,
            "auditor_id": "codex-subagent:test-auditor",
            "audit_timestamp": "2026-09-08T00:01:00+08:00",
            "protocol_body_sha256": protocol["protocol_body_sha256"],
            "source_sha256": protocol["source_sha256"],
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
        receipt["receipt_body_sha256"] = canonical_sha256(receipt)
        receipt_path = root / protocol["audit_gate"]["static_receipt_path"]
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    return path


def test_pack_has_exact_observed_only_keys_and_reproducible_hashes(
    tmp_path: Path,
) -> None:
    protocol_value = _protocol_value(tmp_path)
    protocol_path = _write_protocol(tmp_path, protocol_value)
    manifest = prepare_blind_pack(
        protocol_path,
        acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    validated = load_and_validate_protocol(protocol_path, repo_root=tmp_path)
    audited = validate_blind_pack(validated.output_dir, protocol=validated)
    assert audited == manifest
    assert manifest["truth_exclusion"] == {
        "clean_written": False,
        "corruption_written": False,
        "corruption_support_written": False,
        "time_written": False,
        "downstream_solver_inputs_are_observed_only": True,
    }

    case = manifest["cases"][0]
    npz_path = validated.output_dir / case["npz_name"]
    with np.load(npz_path, allow_pickle=False) as archive:
        assert archive.files == list(EXACT_NPZ_KEYS)
        normalized = archive["observed_normalized"]
        initial_B = archive["initial_B"]
        assert not {
            "clean",
            "corruption",
            "corruption_support",
            "time",
        }.intersection(archive.files)

    dataset = generate_figure6_dataset(_paper_config(validated.data_config, TEST_SEED))
    observed = np.ascontiguousarray(dataset.observed, dtype=np.float64)
    _, singular_values, right_t = linalg.svd(
        observed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    expected_normalized = np.ascontiguousarray(
        observed / singular_values[0], dtype=np.float64
    )
    expected_B = np.ascontiguousarray(right_t[:3].T, dtype=np.float64)
    np.testing.assert_array_equal(normalized, expected_normalized)
    np.testing.assert_array_equal(initial_B, expected_B)
    assert case["shape"] == list(observed.shape)
    assert case["sigma1"] == float(singular_values[0])
    assert case["observed_sha256"] == array_sha256(observed)
    assert case["observed_normalized_sha256"] == array_sha256(normalized)
    assert case["initial_B_sha256"] == array_sha256(initial_B)
    assert case["npz_sha256"] == _file_sha256(npz_path)


def test_pack_is_create_once_and_checks_before_generating(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol_path = _write_protocol(tmp_path, _protocol_value(tmp_path))
    prepare_blind_pack(
        protocol_path,
        acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )

    def forbidden_generator(*args, **kwargs):
        raise AssertionError("occupied output must fail before data generation")

    monkeypatch.setattr(
        "experiments.prepare_video_cleaned_input_solver_blind_pack.generate_figure6_dataset",
        forbidden_generator,
    )
    with pytest.raises(FileExistsError, match="create-once"):
        prepare_blind_pack(
            protocol_path,
            acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )


def test_pack_requires_static_audit_before_generator_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = _protocol_value(tmp_path)
    protocol_path = _write_protocol(
        tmp_path,
        protocol,
        write_static_receipt=False,
    )

    def forbidden_generator(*args, **kwargs):
        raise AssertionError("generator must not run before static audit")

    monkeypatch.setattr(
        "experiments.prepare_video_cleaned_input_solver_blind_pack.generate_figure6_dataset",
        forbidden_generator,
    )
    with pytest.raises(ValueError, match="regular non-symlink file"):
        prepare_blind_pack(
            protocol_path,
            acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "blind-output").exists()


def test_bad_ack_and_bad_protocols_fail_without_creating_pack(
    tmp_path: Path,
) -> None:
    protocol = _protocol_value(tmp_path)
    protocol_path = _write_protocol(tmp_path, protocol)
    with pytest.raises(ValueError, match="acknowledgement"):
        prepare_blind_pack(
            protocol_path,
            acknowledgement="NOT_THE_ACK",
            repo_root=tmp_path,
        )
    assert not (tmp_path / "blind-output").exists()

    tampered = dict(protocol)
    tampered["smoke_launch_state"] = "RUN"
    tampered_path = _write_protocol(tmp_path, tampered, "tampered.json")
    with pytest.raises(ValueError, match="digest|FROZEN_NOT_RUN"):
        load_and_validate_protocol(tampered_path, repo_root=tmp_path)

    extra_config = json.loads(json.dumps(protocol))
    extra_config["data_config"]["clean"] = "forbidden"
    extra_config.pop("protocol_body_sha256")
    extra_config["protocol_body_sha256"] = canonical_sha256(extra_config)
    extra_path = _write_protocol(tmp_path, extra_config, "extra.json")
    with pytest.raises(ValueError, match="data_config keys differ"):
        load_and_validate_protocol(extra_path, repo_root=tmp_path)


def test_protocol_rejects_nonfinite_json_and_wrong_npz_whitelist(
    tmp_path: Path,
) -> None:
    protocol = _protocol_value(tmp_path)
    nonfinite_path = tmp_path / "nonfinite.json"
    text = json.dumps(protocol).replace('"total_time": 6.0', '"total_time": NaN')
    nonfinite_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_and_validate_protocol(nonfinite_path, repo_root=tmp_path)

    wrong_keys = json.loads(json.dumps(protocol))
    wrong_keys["blind_pack"]["exact_npz_keys"].append("clean")
    wrong_keys.pop("protocol_body_sha256")
    wrong_keys["protocol_body_sha256"] = canonical_sha256(wrong_keys)
    wrong_path = _write_protocol(tmp_path, wrong_keys, "wrong-keys.json")
    with pytest.raises(ValueError, match="exact_npz_keys"):
        load_and_validate_protocol(wrong_path, repo_root=tmp_path)


def test_protocol_binds_fresh_seed_scan_digest_and_candidate_bank(
    tmp_path: Path,
) -> None:
    protocol = _protocol_value(tmp_path)
    protocol_path = _write_protocol(tmp_path, protocol)
    scan_path = tmp_path / "operations" / "seed_scan.json"

    scan_path.write_text(scan_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="seed history scan digest mismatch"):
        load_and_validate_protocol(protocol_path, repo_root=tmp_path)

    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    scan["candidate_bank"] = [TEST_SEED + 1]
    scan_path.write_text(
        json.dumps(scan, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    protocol["fresh_seed_contract"]["seed_history_scan_sha256"] = _file_sha256(
        scan_path
    )
    protocol.pop("protocol_body_sha256")
    protocol["protocol_body_sha256"] = canonical_sha256(protocol)
    protocol_path = _write_protocol(tmp_path, protocol)
    with pytest.raises(ValueError, match="candidate bank differs"):
        load_and_validate_protocol(protocol_path, repo_root=tmp_path)


def test_preparer_rejects_protocol_and_output_symlinks(tmp_path: Path) -> None:
    protocol_path = _write_protocol(tmp_path, _protocol_value(tmp_path))
    protocol_link = tmp_path / "protocol-link.json"
    protocol_link.symlink_to(protocol_path)
    with pytest.raises(ValueError, match="protocol path must not be a symlink"):
        load_and_validate_protocol(protocol_link, repo_root=tmp_path)

    output_link = tmp_path / "blind-output"
    output_link.symlink_to(tmp_path / "blind-output-real")
    with pytest.raises(ValueError, match="symlink components"):
        prepare_blind_pack(
            protocol_path,
            acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
            repo_root=tmp_path,
        )
    assert not (tmp_path / "blind-output-real").exists()


def test_pack_validator_rejects_self_digested_non_pca_initialization(
    tmp_path: Path,
) -> None:
    protocol_path = _write_protocol(tmp_path, _protocol_value(tmp_path))
    prepare_blind_pack(
        protocol_path,
        acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    validated = load_and_validate_protocol(protocol_path, repo_root=tmp_path)
    manifest_path = validated.output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    npz_path = validated.output_dir / case["npz_name"]
    with np.load(npz_path, allow_pickle=False) as archive:
        observed = np.asarray(archive["observed_normalized"])
        forged_B = np.zeros_like(archive["initial_B"])
    with npz_path.open("wb") as handle:
        np.savez(handle, observed_normalized=observed, initial_B=forged_B)
    case["initial_B_sha256"] = array_sha256(forged_B)
    case["npz_sha256"] = _file_sha256(npz_path)
    case["npz_size_bytes"] = npz_path.stat().st_size
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="observed-PCA"):
        validate_blind_pack(validated.output_dir, protocol=validated)


def test_pack_validator_rejects_npz_archive_comment_side_channel(
    tmp_path: Path,
) -> None:
    protocol_path = _write_protocol(tmp_path, _protocol_value(tmp_path))
    prepare_blind_pack(
        protocol_path,
        acknowledgement=PREPARATION_ACKNOWLEDGEMENT,
        repo_root=tmp_path,
    )
    validated = load_and_validate_protocol(protocol_path, repo_root=tmp_path)
    manifest_path = validated.output_dir / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    case = manifest["cases"][0]
    npz_path = validated.output_dir / case["npz_name"]
    with ZipFile(npz_path, "a") as archive:
        archive.comment = b"clean_truth_forbidden=123"
    case["npz_sha256"] = _file_sha256(npz_path)
    case["npz_size_bytes"] = npz_path.stat().st_size
    manifest.pop("manifest_body_sha256")
    manifest["manifest_body_sha256"] = canonical_sha256(manifest)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical two-array container"):
        validate_blind_pack(validated.output_dir, protocol=validated)

"""Run the isolated cleaned-input solver smoke on observed-only packs.

This runner directly imports no data generator or statistical evaluator and
makes no computational use of either.  Importing an ``rspca`` submodule does
execute the package's existing eager ``__init__`` imports; those import-only
modules are therefore source-bound even though this runner calls none of their
Fed/FedRPCA or metric functions.  The runner accepts only a frozen protocol and
pack files whose NPZ inventory is exactly ``observed_normalized`` and
``initial_B``.  Its gate concerns numerical plumbing and stationarity only.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import stat
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg

from rspca.certificate_balanced_cleaned_encoder import (
    fit_certificate_balanced_cleaned_encoder,
)
from rspca.cleaned_encoder_v2 import cleaned_encoder_block_stationarity


PROTOCOL_KIND = "cleaned_input_solver_v3_truth_blind_smoke_protocol"
PROTOCOL_STATE = "FROZEN_NOT_RUN"
PACK_KIND = "cleaned_input_solver_v3_observed_only_blind_pack"
AUDIT_KIND = "cleaned_input_solver_v3_truth_blind_smoke_independent_audit"
STATIC_AUDIT_KIND = "cleaned_input_solver_v3_static_pre_pack_independent_audit"
PROVENANCE_COMMITMENT_KIND = (
    "cleaned_input_solver_v3_blind_pack_generator_replay_commitments"
)
PROVENANCE_VERIFICATION_STATUS = "PASS_GENERATOR_REPLAY_COMMITMENTS"
RUN_ACKNOWLEDGEMENT = "RUN_LOCKED_TRUTH_BLIND_SOLVER_SMOKE"
EXACT_NPZ_KEYS = ("observed_normalized", "initial_B")
PACK_MANIFEST_NAME = "blind_pack_manifest.json"
REPORT_NAME = "solver_smoke_report.json"
PROTOCOL_COPY_NAME = "protocol_copy.json"
PACK_MANIFEST_COPY_NAME = "blind_pack_manifest_copy.json"
OUTPUT_MANIFEST_NAME = "solver_smoke_output_manifest.json"
REQUIRED_OUTPUT_NAMES = frozenset(
    {
        REPORT_NAME,
        PROTOCOL_COPY_NAME,
        PACK_MANIFEST_COPY_NAME,
        OUTPUT_MANIFEST_NAME,
    }
)
MINIMUM_SOURCE_PATHS = frozenset(
    {
        "experiments/__init__.py",
        "experiments/audit_video_cleaned_input_solver_blind_pack_provenance.py",
        "experiments/prepare_video_cleaned_input_solver_blind_pack.py",
        "experiments/video_cleaned_input_solver_truth_blind_smoke.py",
        "experiments/video_spca_paper_reproduction.py",
        "src/rspca/__init__.py",
        "src/rspca/accelerated.py",
        "src/rspca/certificate_balanced_cleaned_encoder.py",
        "src/rspca/cleaned_encoder_v2.py",
        "src/rspca/cross_modal_gates.py",
        "src/rspca/datasets.py",
        "src/rspca/federated.py",
        "src/rspca/fedrpca_audit.py",
        "src/rspca/gauge.py",
        "src/rspca/gauge_stress.py",
        "src/rspca/metrics.py",
        "src/rspca/product_aggregation.py",
        "src/rspca/randomized.py",
        "src/rspca/robust.py",
        "src/rspca/statistical_gates.py",
        "src/rspca/trust_audit.py",
        "src/rspca/vp.py",
    }
)
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PACK_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "protocol_kind",
        "protocol_body_sha256",
        "source_sha256",
        "data_config",
        "solver_config",
        "exact_npz_keys",
        "case_count",
        "cases",
        "truth_exclusion",
        "static_audit_receipt_sha256",
        "manifest_body_sha256",
    }
)
_PACK_CASE_KEYS = frozenset(
    {
        "seed",
        "npz_name",
        "shape",
        "sigma1",
        "observed_sha256",
        "observed_normalized_sha256",
        "initial_B_sha256",
        "npz_sha256",
        "npz_size_bytes",
    }
)
_EXPECTED_EXCLUSION = {
    "clean_written": False,
    "corruption_written": False,
    "corruption_support_written": False,
    "time_written": False,
    "downstream_solver_inputs_are_observed_only": True,
}
_FRESH_SEED_CONTRACT_KEYS = frozenset(
    {
        "seed_history_scan_path",
        "seed_history_scan_sha256",
        "semantic_hits_before_freeze",
        "candidate_outcomes_before_freeze",
        "seed_replacement_allowed",
        "use_boundary",
    }
)
_SEED_HISTORY_SCAN_KEYS = frozenset(
    {
        "schema_version",
        "scan_kind",
        "scan_timestamp",
        "git_head",
        "candidate_bank",
        "scan_roots",
        "excluded_paths",
        "exact_candidate_regex",
        "exact_scan_command",
        "semantic_seed_hits_before_freeze",
        "candidate_outcomes_generated_before_freeze",
        "selection_rule",
        "use_boundary",
        "seed_replacement_allowed",
    }
)
_SEED_HISTORY_SCAN_KIND = (
    "video_cleaned_input_solver_v3_pre_freeze_semantic_seed_history_scan"
)
_PROVENANCE_CASE_KEYS = frozenset(
    {
        "seed",
        "observed_sha256",
        "observed_normalized_sha256",
        "initial_B_sha256",
        "sigma1",
    }
)
_AUDIT_GATE_KEYS = frozenset(
    {
        "required",
        "receipt_path",
        "status_required",
        "auditor_independent_required",
        "receipt_must_match_protocol_body_sha256",
        "receipt_must_match_source_sha256",
        "static_receipt_path",
        "static_receipt_required_before_blind_pack",
    }
)
_STATIC_AUDIT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "audit_kind",
        "status",
        "auditor_independent",
        "auditor_id",
        "audit_timestamp",
        "protocol_body_sha256",
        "source_sha256",
        "checked_nonruntime_sha256",
        "focused_test_command",
        "focused_test_count",
        "focused_tests_passed",
        "blind_pack_absent_at_audit",
        "candidate_pack_generations_before_audit",
        "candidate_outcomes_observed_before_audit",
        "solver_smoke_runs_before_audit",
        "statistical_or_formal_run_authorized",
        "receipt_body_sha256",
    }
)
_STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS = frozenset(
    {
        "docs/video_cleaned_input_solver_v3_novelty_audit.md",
        "operations/video_cleaned_input_solver_v3_scope_seal_20260908.json",
        "tests/test_certificate_balanced_cleaned_encoder.py",
        "tests/test_prepare_video_cleaned_input_solver_blind_pack.py",
        "tests/test_audit_video_cleaned_input_solver_blind_pack_provenance.py",
        "tests/test_video_cleaned_input_solver_truth_blind_smoke.py",
        "tests/test_cleaned_encoder_v2.py",
    }
)
_FINAL_AUDIT_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "audit_kind",
        "status",
        "auditor_independent",
        "auditor_id",
        "audit_timestamp",
        "provenance_verifier_command",
        "protocol_body_sha256",
        "source_sha256",
        "static_audit_receipt_sha256",
        "candidate_outcomes_observed_before_audit",
        "formal_or_statistical_run_authorized",
        "static_audit_completed_before_blind_pack",
        "blind_pack_generated_after_static_audit",
        "blind_pack_provenance_verified",
        "blind_pack_provenance_commitment",
        "receipt_body_sha256",
    }
)
_PROVENANCE_COMMITMENT_KEYS = frozenset(
    {
        "schema_version",
        "commitment_kind",
        "verification_status",
        "not_a_final_audit_receipt",
        "protocol_body_sha256",
        "source_map_sha256",
        "fresh_seed_scan_file_sha256",
        "static_audit_receipt_sha256",
        "blind_pack_manifest_body_sha256",
        "blind_pack_manifest_file_sha256",
        "verifier_source_sha256",
        "case_count",
        "ordered_cases",
        "checks",
        "boundary",
        "commitment_body_sha256",
    }
)
_PROVENANCE_BOUNDARY = {
    "solver_fits": 0,
    "statistical_metrics": 0,
    "truth_metrics": 0,
    "artifact_filesystem_writes": 0,
    "final_audit_receipts_issued": 0,
}
_EXPECTED_PROVENANCE_CHECKS = {
    "protocol_body_digest": True,
    "runtime_source_closure_22_files": True,
    "fresh_seed_receipt": True,
    "static_audit_receipt_bound": True,
    "manifest_body_digest": True,
    "closed_pack_inventory": True,
    "canonical_npz_whole_bytes": True,
    "generator_replay_raw_observed": True,
    "generator_replay_sigma1": True,
    "generator_replay_normalized": True,
    "generator_replay_pca_B": True,
}


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"strict JSON forbids non-finite constant {token}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"strict JSON forbids duplicate key {key!r}")
        result[key] = value
    return result


def _require_finite_json(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON keys must be strings")
            _require_finite_json(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_json(item)
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("strict JSON forbids non-finite values")
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _parse_strict_json_bytes(payload: bytes) -> dict[str, Any]:
    result = json.loads(
        payload.decode("utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(result, dict):
        raise TypeError("JSON document must contain an object")
    _require_finite_json(result)
    return result


def _load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path}")
    return _parse_strict_json_bytes(path.read_bytes())


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("strict JSON forbids non-finite values")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: Any) -> str:
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be content-hashed")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def _protocol_body(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in protocol.items()
        if str(key) != "protocol_body_sha256"
    }


def _manifest_body(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in manifest.items()
        if str(key) != "manifest_body_sha256"
    }


def _relative_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path


def _contained(root: Path, relative: Path, *, name: str) -> Path:
    lexical = root
    for component in relative.parts:
        lexical = lexical / component
        if lexical.is_symlink():
            raise ValueError(f"{name} must not contain symlink components")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the repository") from error
    return resolved


def _regular_file(path: Path, *, name: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise ValueError(f"{name} must be a regular non-symlink file")


def _snapshot_validated_json(
    path: Path,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> bytes:
    _regular_file(path, name=name)
    payload = path.read_bytes()
    if _parse_strict_json_bytes(payload) != expected:
        raise RuntimeError(f"{name} changed after validation")
    return payload


def _require_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return value


def _validate_fresh_seed_contract(protocol: Mapping[str, Any], repo_root: Path) -> None:
    contract = _require_mapping(
        protocol["fresh_seed_contract"], name="fresh_seed_contract"
    )
    if set(contract) != set(_FRESH_SEED_CONTRACT_KEYS):
        raise ValueError("fresh_seed_contract keys changed")
    relative = _relative_path(
        contract["seed_history_scan_path"], name="seed history scan path"
    )
    reported_digest = _require_sha(
        contract["seed_history_scan_sha256"], name="seed history scan digest"
    )
    unresolved = repo_root / relative
    if unresolved.is_symlink():
        raise ValueError("seed history scan must not be a symlink")
    scan_path = _contained(repo_root, relative, name="seed history scan path")
    _regular_file(scan_path, name="seed history scan")
    if _file_sha256(scan_path) != reported_digest:
        raise ValueError("seed history scan digest mismatch")
    scan = _load_strict_json(scan_path)
    if set(scan) != set(_SEED_HISTORY_SCAN_KEYS):
        raise ValueError("seed history scan keys changed")
    if scan["schema_version"] != 1 or scan["scan_kind"] != _SEED_HISTORY_SCAN_KIND:
        raise ValueError("unexpected seed history scan schema or kind")
    data = _require_mapping(protocol["data_config"], name="data_config")
    if scan["candidate_bank"] != data.get("seeds"):
        raise ValueError("seed history candidate bank differs from data seeds")
    if (
        isinstance(contract["semantic_hits_before_freeze"], bool)
        or contract["semantic_hits_before_freeze"] != 0
        or isinstance(scan["semantic_seed_hits_before_freeze"], bool)
        or scan["semantic_seed_hits_before_freeze"] != 0
    ):
        raise ValueError("fresh seeds must have zero semantic history hits")
    if (
        isinstance(contract["candidate_outcomes_before_freeze"], bool)
        or contract["candidate_outcomes_before_freeze"] != 0
        or isinstance(scan["candidate_outcomes_generated_before_freeze"], bool)
        or scan["candidate_outcomes_generated_before_freeze"] != 0
    ):
        raise ValueError("fresh seeds must have zero outcomes before freeze")
    if (
        contract["seed_replacement_allowed"] is not False
        or scan["seed_replacement_allowed"] is not False
    ):
        raise ValueError("fresh-seed replacement must remain forbidden")
    if (
        not isinstance(contract["use_boundary"], str)
        or not contract["use_boundary"]
        or scan["use_boundary"] != contract["use_boundary"]
    ):
        raise ValueError("fresh-seed use boundary differs from scan receipt")


def _validate_audit_gate(value: Any) -> Mapping[str, Any]:
    gate = _require_mapping(value, name="audit_gate")
    if set(gate) != set(_AUDIT_GATE_KEYS):
        raise ValueError("independent audit gate changed")
    if gate["required"] is not True:
        raise ValueError("final independent audit must remain required")
    if gate["status_required"] != "passed":
        raise ValueError("audit status requirement must remain passed")
    for field in (
        "auditor_independent_required",
        "receipt_must_match_protocol_body_sha256",
        "receipt_must_match_source_sha256",
        "static_receipt_required_before_blind_pack",
    ):
        if gate[field] is not True:
            raise ValueError(f"audit gate requires {field}=true")
    final_receipt = _relative_path(
        gate["receipt_path"], name="final audit receipt path"
    )
    static_receipt = _relative_path(
        gate["static_receipt_path"], name="static audit receipt path"
    )
    if final_receipt == static_receipt:
        raise ValueError("static and final audit receipt paths must be distinct")
    return gate


def _validate_static_audit_receipt(
    protocol: Mapping[str, Any], repo_root: Path
) -> tuple[dict[str, Any], Path, str]:
    gate = _validate_audit_gate(protocol["audit_gate"])
    receipt_path = _contained(
        repo_root,
        _relative_path(gate["static_receipt_path"], name="static audit receipt path"),
        name="static audit receipt path",
    )
    receipt = _load_strict_json(receipt_path)
    if set(receipt) != set(_STATIC_AUDIT_RECEIPT_KEYS):
        raise ValueError("static audit receipt keys changed")
    reported = _require_sha(
        receipt["receipt_body_sha256"], name="static audit receipt body digest"
    )
    body = {
        str(key): item
        for key, item in receipt.items()
        if str(key) != "receipt_body_sha256"
    }
    if _canonical_sha256(body) != reported:
        raise ValueError("static audit receipt body digest mismatch")
    if receipt["schema_version"] != 1 or receipt["audit_kind"] != STATIC_AUDIT_KIND:
        raise ValueError("unexpected static audit receipt schema or kind")
    if receipt["status"] != gate["status_required"]:
        raise ValueError("static independent audit did not pass")
    if receipt["auditor_independent"] is not True:
        raise ValueError("static audit receipt is not independent")
    for field in ("auditor_id", "audit_timestamp", "focused_test_command"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise ValueError(f"static audit receipt {field} must be nonempty")
    count = receipt["focused_test_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("static audit focused_test_count must be positive")
    if receipt["focused_tests_passed"] is not True:
        raise ValueError("static audit focused tests did not pass")
    if receipt["protocol_body_sha256"] != protocol["protocol_body_sha256"]:
        raise ValueError("static audit receipt protocol digest mismatch")
    if receipt["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("static audit receipt source map mismatch")
    checked = receipt["checked_nonruntime_sha256"]
    if not isinstance(checked, Mapping):
        raise TypeError("static checked_nonruntime_sha256 must be an object")
    required = set(_STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS)
    required.add(
        str(
            _relative_path(
                protocol["fresh_seed_contract"]["seed_history_scan_path"],
                name="seed history scan path",
            )
        )
    )
    if not required.issubset(checked):
        missing = sorted(required.difference(checked))
        raise ValueError(f"static audit lacks nonruntime evidence: {missing}")
    for raw_path, raw_digest in checked.items():
        relative = _relative_path(raw_path, name="static nonruntime evidence path")
        digest = _require_sha(raw_digest, name=f"static evidence digest {raw_path}")
        evidence_path = _contained(
            repo_root,
            relative,
            name=f"static nonruntime evidence {raw_path}",
        )
        _regular_file(evidence_path, name=f"static nonruntime evidence {raw_path}")
        if _file_sha256(evidence_path) != digest:
            raise ValueError(f"static nonruntime evidence digest mismatch: {raw_path}")
    if receipt["blind_pack_absent_at_audit"] is not True:
        raise ValueError("static audit did not establish blind-pack absence")
    for field in (
        "candidate_pack_generations_before_audit",
        "candidate_outcomes_observed_before_audit",
        "solver_smoke_runs_before_audit",
    ):
        value = receipt[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise ValueError(f"static audit receipt requires {field}=0")
    if receipt["statistical_or_formal_run_authorized"] is not False:
        raise ValueError("static audit receipt exceeds solver-smoke scope")
    return receipt, receipt_path, _file_sha256(receipt_path)


def _validate_protocol(path: Path, repo_root: Path) -> dict[str, Any]:
    protocol = _load_strict_json(path)
    required = {
        "schema_version",
        "protocol_kind",
        "smoke_launch_state",
        "data_config",
        "solver_config",
        "blind_pack",
        "smoke_output",
        "solver_gate",
        "audit_gate",
        "fresh_seed_contract",
        "source_sha256",
        "protocol_body_sha256",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"protocol is missing required fields: {missing}")
    if protocol["schema_version"] != 1:
        raise ValueError("unsupported protocol schema_version")
    if protocol["protocol_kind"] != PROTOCOL_KIND:
        raise ValueError("unexpected protocol kind")
    if protocol["smoke_launch_state"] != PROTOCOL_STATE:
        raise ValueError("protocol is not frozen and unrun")
    if protocol.get("smoke_launch_authorized_by_this_file") not in (None, False):
        raise ValueError("protocol must not self-authorize execution")
    reported = _require_sha(
        protocol["protocol_body_sha256"], name="protocol_body_sha256"
    )
    if _canonical_sha256(_protocol_body(protocol)) != reported:
        raise ValueError("protocol body digest mismatch")

    solver = _require_mapping(protocol["solver_config"], name="solver_config")
    if set(solver) != {
        "alpha",
        "kappa",
        "max_updates",
        "tol",
        "max_block_delay",
        "backtracking_expand",
    }:
        raise ValueError("solver_config keys changed")
    if (solver["alpha"], solver["kappa"]) != (0.0015, 0.001):
        raise ValueError("locked alpha/kappa changed")
    if (
        isinstance(solver["max_updates"], bool)
        or not isinstance(solver["max_updates"], int)
        or solver["max_updates"] < 3
    ):
        raise ValueError("max_updates must be an integer at least three")
    if (
        isinstance(solver["max_block_delay"], bool)
        or not isinstance(solver["max_block_delay"], int)
        or solver["max_block_delay"] < 3
    ):
        raise ValueError("max_block_delay must be an integer at least three")
    for name in ("alpha", "kappa", "tol", "backtracking_expand"):
        value = solver[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"solver_config.{name} must be numeric")
        if not np.isfinite(value):
            raise ValueError(f"solver_config.{name} must be finite")
    if solver["kappa"] <= 0.0 or solver["tol"] <= 0.0:
        raise ValueError("kappa and tol must be positive")
    if solver["alpha"] < 0.0 or solver["backtracking_expand"] < 1.0:
        raise ValueError("invalid alpha or backtracking expansion")

    pack = _require_mapping(protocol["blind_pack"], name="blind_pack")
    if set(pack) != {"output_dir", "manifest_name", "exact_npz_keys"}:
        raise ValueError("blind_pack keys changed")
    if pack["manifest_name"] != PACK_MANIFEST_NAME:
        raise ValueError("blind-pack manifest name changed")
    if pack["exact_npz_keys"] != list(EXACT_NPZ_KEYS):
        raise ValueError("blind-pack NPZ key contract changed")

    smoke = _require_mapping(protocol["smoke_output"], name="smoke_output")
    if set(smoke) != {"output_dir", "report_name", "manifest_name"}:
        raise ValueError("smoke_output keys changed")
    if (
        smoke["report_name"] != REPORT_NAME
        or smoke["manifest_name"] != OUTPUT_MANIFEST_NAME
    ):
        raise ValueError("smoke output names changed")

    gate = _require_mapping(protocol["solver_gate"], name="solver_gate")
    expected_gate = {
        "all_cases_converged",
        "complete_stationarity_at_most",
        "orthogonality_at_most",
        "monotonicity_tolerance_factor",
        "certificate_rtol",
        "certificate_atol",
        "maximum_backtracking_trials",
        "no_bounded_delay_violation",
        "all_values_finite",
    }
    if set(gate) != expected_gate:
        raise ValueError("solver_gate keys changed")
    if gate["all_cases_converged"] is not True:
        raise ValueError("smoke must require every case to converge")

    _validate_audit_gate(protocol["audit_gate"])

    sources = _require_mapping(protocol["source_sha256"], name="source_sha256")
    if not sources:
        raise ValueError("source_sha256 must be nonempty")
    if set(sources) != set(MINIMUM_SOURCE_PATHS):
        missing = sorted(MINIMUM_SOURCE_PATHS.difference(sources))
        extra = sorted(set(sources).difference(MINIMUM_SOURCE_PATHS))
        raise ValueError(
            "source_sha256 differs from audited 22-file closure; "
            f"missing={missing}, extra={extra}"
        )
    for raw_path, raw_digest in sources.items():
        relative = _relative_path(raw_path, name="source path")
        digest = _require_sha(raw_digest, name=f"source digest {raw_path}")
        source = _contained(repo_root, relative, name=f"source {raw_path}")
        _regular_file(source, name=f"source {raw_path}")
        if _file_sha256(source) != digest:
            raise ValueError(f"source digest mismatch: {raw_path}")
    _validate_fresh_seed_contract(protocol, repo_root)
    return protocol


def _validate_audit_receipt(
    protocol: Mapping[str, Any],
    repo_root: Path,
    *,
    static_audit_receipt_sha256: str,
) -> dict[str, Any]:
    gate = _validate_audit_gate(protocol["audit_gate"])
    receipt_path = _contained(
        repo_root,
        _relative_path(gate["receipt_path"], name="audit receipt path"),
        name="audit receipt path",
    )
    receipt = _load_strict_json(receipt_path)
    if set(receipt) != set(_FINAL_AUDIT_RECEIPT_KEYS):
        raise ValueError("final independent audit receipt keys changed")
    reported = _require_sha(
        receipt["receipt_body_sha256"], name="final audit receipt body digest"
    )
    body = {
        str(key): item
        for key, item in receipt.items()
        if str(key) != "receipt_body_sha256"
    }
    if _canonical_sha256(body) != reported:
        raise ValueError("final audit receipt body digest mismatch")
    if receipt.get("schema_version") != 1:
        raise ValueError("unexpected final audit receipt schema")
    if receipt.get("audit_kind") != AUDIT_KIND:
        raise ValueError("unexpected independent audit kind")
    if receipt.get("status") != gate["status_required"]:
        raise ValueError("independent audit did not pass")
    if (
        gate["auditor_independent_required"]
        and receipt.get("auditor_independent") is not True
    ):
        raise ValueError("audit receipt is not independent")
    for field in ("auditor_id", "audit_timestamp", "provenance_verifier_command"):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise ValueError(f"final audit receipt {field} must be nonempty")
    if (
        gate["receipt_must_match_protocol_body_sha256"]
        and receipt.get("protocol_body_sha256") != protocol["protocol_body_sha256"]
    ):
        raise ValueError("audit receipt protocol digest mismatch")
    if (
        gate["receipt_must_match_source_sha256"]
        and receipt.get("source_sha256") != protocol["source_sha256"]
    ):
        raise ValueError("audit receipt source map mismatch")
    if receipt.get("candidate_outcomes_observed_before_audit") is not False:
        raise ValueError("audit was not completed before candidate outcomes")
    if receipt.get("formal_or_statistical_run_authorized") is not False:
        raise ValueError("audit receipt exceeds solver-smoke scope")
    if receipt.get("static_audit_completed_before_blind_pack") is not True:
        raise ValueError("static audit was not completed before blind-pack preparation")
    if receipt.get("blind_pack_generated_after_static_audit") is not True:
        raise ValueError("blind pack was not generated after the static audit")
    if receipt.get("blind_pack_provenance_verified") is not True:
        raise ValueError("blind-pack provenance was not independently verified")
    if receipt.get("static_audit_receipt_sha256") != static_audit_receipt_sha256:
        raise ValueError("final audit receipt static-audit digest mismatch")
    if not isinstance(receipt.get("blind_pack_provenance_commitment"), Mapping):
        raise ValueError("audit receipt lacks generator-replay commitment")
    return receipt


def _validate_audit_pack_provenance(
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    manifest_file_sha256: str,
    static_audit_receipt_sha256: str,
) -> None:
    commitment = receipt["blind_pack_provenance_commitment"]
    if set(commitment) != set(_PROVENANCE_COMMITMENT_KEYS):
        raise ValueError("generator-replay commitment keys changed")
    reported = _require_sha(
        commitment["commitment_body_sha256"],
        name="generator-replay commitment body digest",
    )
    body = {
        str(key): item
        for key, item in commitment.items()
        if str(key) != "commitment_body_sha256"
    }
    if _canonical_sha256(body) != reported:
        raise ValueError("generator-replay commitment body digest mismatch")
    if (
        commitment["schema_version"] != 1
        or commitment["commitment_kind"] != PROVENANCE_COMMITMENT_KIND
        or commitment["verification_status"] != PROVENANCE_VERIFICATION_STATUS
        or commitment["not_a_final_audit_receipt"] is not True
    ):
        raise ValueError("unexpected generator-replay commitment disposition")
    if commitment["protocol_body_sha256"] != protocol["protocol_body_sha256"]:
        raise ValueError("generator-replay protocol digest mismatch")
    expected_source_map_sha = _canonical_sha256(
        {"source_sha256": protocol["source_sha256"]}
    )
    if commitment["source_map_sha256"] != expected_source_map_sha:
        raise ValueError("generator-replay source-map digest mismatch")
    if (
        commitment["fresh_seed_scan_file_sha256"]
        != protocol["fresh_seed_contract"]["seed_history_scan_sha256"]
    ):
        raise ValueError("generator-replay fresh-seed receipt digest mismatch")
    if commitment["static_audit_receipt_sha256"] != static_audit_receipt_sha256:
        raise ValueError("generator-replay static-audit digest mismatch")
    if (
        commitment["blind_pack_manifest_body_sha256"]
        != manifest["manifest_body_sha256"]
        or commitment["blind_pack_manifest_file_sha256"] != manifest_file_sha256
    ):
        raise ValueError("generator-replay blind-pack manifest digest mismatch")
    verifier_relative = (
        "experiments/audit_video_cleaned_input_solver_blind_pack_provenance.py"
    )
    if commitment["verifier_source_sha256"] != protocol["source_sha256"].get(
        verifier_relative
    ):
        raise ValueError("generator-replay verifier source digest mismatch")
    if commitment["checks"] != _EXPECTED_PROVENANCE_CHECKS:
        raise ValueError("generator-replay commitment checks changed or did not pass")
    if commitment["boundary"] != _PROVENANCE_BOUNDARY:
        raise ValueError("generator-replay commitment exceeded its boundary")
    commitments = commitment["ordered_cases"]
    if not isinstance(commitments, list) or not commitments:
        raise ValueError("generator-replay commitment lacks ordered cases")
    cases = manifest["cases"]
    count = commitment["case_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or count != len(commitments)
        or count != len(cases)
    ):
        raise ValueError("audited blind-pack commitment count mismatch")
    for commitment, case in zip(commitments, cases, strict=True):
        if not isinstance(commitment, Mapping) or set(commitment) != set(
            _PROVENANCE_CASE_KEYS
        ):
            raise ValueError("audited blind-pack case commitment keys changed")
        seed = commitment["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed != case["seed"]:
            raise ValueError("audited blind-pack seed order mismatch")
        for name in (
            "observed_sha256",
            "observed_normalized_sha256",
            "initial_B_sha256",
        ):
            _require_sha(commitment[name], name=f"audited case {name}")
            if commitment[name] != case[name]:
                raise ValueError(f"audited blind-pack case {name} mismatch")
        sigma1 = commitment["sigma1"]
        if (
            isinstance(sigma1, bool)
            or not isinstance(sigma1, (int, float))
            or not np.isfinite(sigma1)
            or float(sigma1).hex() != float(case["sigma1"]).hex()
        ):
            raise ValueError("audited blind-pack case sigma1 mismatch")


def _validate_pack(protocol: Mapping[str, Any], repo_root: Path) -> tuple[
    Path,
    dict[str, Any],
    list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]],
]:
    pack_config = protocol["blind_pack"]
    pack_dir = _contained(
        repo_root,
        _relative_path(pack_config["output_dir"], name="blind-pack output"),
        name="blind-pack output",
    )
    if pack_dir.is_symlink() or not pack_dir.is_dir():
        raise ValueError("blind-pack output must be a real directory")
    manifest_path = pack_dir / PACK_MANIFEST_NAME
    _regular_file(manifest_path, name="blind-pack manifest")
    manifest = _load_strict_json(manifest_path)
    if set(manifest) != set(_PACK_MANIFEST_KEYS):
        raise ValueError("blind-pack manifest keys changed")
    reported = _require_sha(
        manifest.get("manifest_body_sha256"), name="manifest_body_sha256"
    )
    if _canonical_sha256(_manifest_body(manifest)) != reported:
        raise ValueError("blind-pack manifest body digest mismatch")
    if manifest.get("manifest_kind") != PACK_KIND:
        raise ValueError("unexpected blind-pack kind")
    if manifest.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("blind pack protocol kind mismatch")
    if manifest.get("protocol_body_sha256") != protocol["protocol_body_sha256"]:
        raise ValueError("blind pack protocol digest mismatch")
    if manifest.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("blind pack source map mismatch")
    if manifest.get("exact_npz_keys") != list(EXACT_NPZ_KEYS):
        raise ValueError("blind pack NPZ whitelist changed")
    if manifest.get("data_config") != protocol["data_config"]:
        raise ValueError("blind pack data config mismatch")
    if manifest.get("solver_config") != protocol["solver_config"]:
        raise ValueError("blind pack solver config mismatch")
    if manifest.get("truth_exclusion") != _EXPECTED_EXCLUSION:
        raise ValueError("blind-pack exclusion declaration changed")
    _require_sha(
        manifest.get("static_audit_receipt_sha256"),
        name="blind-pack static audit receipt digest",
    )
    cases = manifest.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or manifest.get("case_count") != len(cases)
    ):
        raise ValueError("blind pack cases are invalid")
    if [case.get("seed") for case in cases] != protocol["data_config"]["seeds"]:
        raise ValueError("blind pack seed order changed")
    expected_entries = {PACK_MANIFEST_NAME}
    seen_seeds: set[int] = set()
    data_config = protocol["data_config"]
    expected_shape = (
        int(data_config["grid_size"]) ** 2,
        int(data_config["n_timesteps"]),
    )
    n_components = int(data_config["n_components"])
    loaded_cases: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray]] = []
    for case in cases:
        if not isinstance(case, Mapping) or set(case) != set(_PACK_CASE_KEYS):
            raise ValueError("blind-pack case keys changed")
        seed = case["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("blind-pack case seed is invalid")
        if seed in seen_seeds:
            raise ValueError("blind-pack case seeds are duplicated")
        seen_seeds.add(seed)
        name = case["npz_name"]
        if name != f"observed_case_{seed}.npz":
            raise ValueError("blind-pack case name is invalid")
        shape = case["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in shape
            )
            or tuple(shape) != expected_shape
        ):
            raise ValueError("blind-pack case shape is invalid")
        sigma1 = case["sigma1"]
        if (
            isinstance(sigma1, bool)
            or not isinstance(sigma1, (int, float))
            or not np.isfinite(sigma1)
            or sigma1 < 0.0
        ):
            raise ValueError("blind-pack sigma1 is invalid")
        for digest_name in (
            "observed_sha256",
            "observed_normalized_sha256",
            "initial_B_sha256",
            "npz_sha256",
        ):
            _require_sha(case[digest_name], name=f"case {digest_name}")
        size = case["npz_size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("blind-pack NPZ size is invalid")
        expected_entries.add(name)
        case_path = pack_dir / name
        _regular_file(case_path, name=f"blind-pack case {name}")
        if _file_sha256(case_path) != _require_sha(
            case.get("npz_sha256"), name=f"NPZ digest {name}"
        ):
            raise ValueError(f"blind-pack NPZ digest mismatch: {name}")
        if case_path.stat().st_size != case.get("npz_size_bytes"):
            raise ValueError(f"blind-pack NPZ size mismatch: {name}")
        observed, initial_B = _load_case(
            pack_dir,
            case,
            n_components,
            expected_shape=expected_shape,
            sigma1=float(sigma1),
        )
        loaded_cases.append((case, observed, initial_B))
    if {entry.name for entry in pack_dir.iterdir()} != expected_entries:
        raise ValueError("blind-pack directory inventory changed")
    return pack_dir, manifest, loaded_cases


def _load_case(
    pack_dir: Path,
    record: Mapping[str, Any],
    n_components: int,
    *,
    expected_shape: tuple[int, int],
    sigma1: float,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(pack_dir / record["npz_name"], allow_pickle=False) as archive:
        if archive.files != list(EXACT_NPZ_KEYS):
            raise ValueError("blind-pack NPZ key whitelist mismatch")
        observed = np.ascontiguousarray(archive["observed_normalized"])
        initial_B = np.ascontiguousarray(archive["initial_B"])
    buffer = BytesIO()
    np.savez(
        buffer,
        observed_normalized=observed,
        initial_B=initial_B,
    )
    if (pack_dir / record["npz_name"]).read_bytes() != buffer.getvalue():
        raise ValueError("blind-pack NPZ is not the canonical two-array container")
    if observed.dtype != np.float64 or initial_B.dtype != np.float64:
        raise ValueError("blind-pack arrays must be float64")
    if observed.shape != expected_shape:
        raise ValueError("blind-pack array shape differs from frozen data config")
    if initial_B.shape != (observed.shape[1], n_components):
        raise ValueError("blind-pack array shape mismatch")
    if list(observed.shape) != record.get("shape"):
        raise ValueError("blind-pack reported shape mismatch")
    if not np.isfinite(observed).all() or not np.isfinite(initial_B).all():
        raise ValueError("blind-pack arrays must be finite")
    if _array_sha256(observed) != record.get("observed_normalized_sha256"):
        raise ValueError("observed-normalized array digest mismatch")
    if _array_sha256(initial_B) != record.get("initial_B_sha256"):
        raise ValueError("initial-B array digest mismatch")
    _, singular_values, right_t = linalg.svd(
        observed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if singular_values.size < n_components:
        raise ValueError("observed-normalized array has too few singular vectors")
    normalized_sigma1 = float(singular_values[0]) if singular_values.size else 0.0
    expected_sigma1 = sigma1 / max(sigma1, np.finfo(np.float64).eps)
    if not np.isclose(
        normalized_sigma1,
        expected_sigma1,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("observed-normalized array violates the sigma1 scale contract")
    expected_B = np.ascontiguousarray(right_t[:n_components].T, dtype=np.float64)
    orthogonality = np.linalg.norm(
        initial_B.T @ initial_B - np.eye(n_components), ord="fro"
    )
    if orthogonality > 1e-12 or not np.allclose(
        initial_B,
        expected_B,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("initial_B is not the observed-PCA gesdd initialization")
    return observed, initial_B


def _block_delay_valid(choices: Sequence[str], horizon: int) -> bool:
    if len(choices) < 3 or tuple(choices[:3]) != ("B", "A", "S"):
        return False
    for block in ("B", "A", "S"):
        positions = [index for index, value in enumerate(choices) if value == block]
        if not positions or positions[0] >= 3:
            return False
        if len(positions) > 1 and max(np.diff(positions)) > horizon:
            return False
    return True


def _schedule_trace_valid(updates: Sequence[Any], horizon: int) -> bool:
    if len(updates) < 3:
        return False
    blocks = ("B", "A", "S")
    last_selected: dict[str, int | None] = {block: None for block in blocks}
    for index, update in enumerate(updates):
        if update.update_index != index or update.block not in blocks:
            return False
        scores = {
            "B": float(update.B_certificate_before),
            "A": float(update.A_certificate_before),
            "S": float(update.S_certificate_before),
        }
        if not all(np.isfinite(value) and value >= 0.0 for value in scores.values()):
            return False
        if float(update.total_certificate_before) != max(scores.values()):
            return False
        if index < 3:
            if update.block != blocks[index] or update.reason != "initial_coverage":
                return False
        else:
            overdue = tuple(
                block
                for block in blocks
                if index - int(last_selected[block]) >= horizon
            )
            if overdue:
                oldest = min(int(last_selected[block]) for block in overdue)
                candidates = tuple(
                    block for block in overdue if int(last_selected[block]) == oldest
                )
                expected_reason = "bounded_delay"
            else:
                candidates = blocks
                expected_reason = "greedy"
            expected_block = max(candidates, key=lambda block: scores[block])
            if update.reason != expected_reason or update.block != expected_block:
                return False
        last_selected[update.block] = index
    return _block_delay_valid(tuple(update.block for update in updates), horizon)


def _numeric_tree_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_numeric_tree_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_numeric_tree_finite(item) for item in value)
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all())
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    if isinstance(value, (int, np.integer, bool, np.bool_, str)) or value is None:
        return True
    return False


def _max_diagnostic_difference(first: Any, second: Any) -> float:
    left = asdict(first)
    right = asdict(second)
    if set(left) != set(right):
        return float("inf")
    return float(max(abs(float(left[key]) - float(right[key])) for key in left))


def _run_case(
    observed: np.ndarray,
    initial_B: np.ndarray,
    *,
    seed: int,
    solver: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    started = perf_counter()
    result = fit_certificate_balanced_cleaned_encoder(
        observed,
        initial_B.shape[1],
        alpha=float(solver["alpha"]),
        kappa=float(solver["kappa"]),
        max_updates=int(solver["max_updates"]),
        tol=float(solver["tol"]),
        max_block_delay=int(solver["max_block_delay"]),
        initial_B=initial_B,
        initial_S=np.zeros_like(observed),
        backtracking_expand=float(solver["backtracking_expand"]),
    )
    runtime_seconds = perf_counter() - started
    final = result.block_stationarity[-1]
    recomputed = cleaned_encoder_block_stationarity(
        observed,
        result.A,
        result.B,
        result.S,
        alpha=result.alpha,
        kappa=result.kappa,
        B_step_size=result.B_certificate_step_size,
        S_step_size=result.S_certificate_step_size,
    )
    diagnostic_difference = _max_diagnostic_difference(final, recomputed)
    objective_array = np.asarray(result.objectives, dtype=np.float64)
    monotonicity_tolerance = float(gate["monotonicity_tolerance_factor"]) * np.maximum(
        1.0, np.abs(objective_array[:-1])
    )
    maximum_objective_excess = float(
        np.max(np.diff(objective_array) - monotonicity_tolerance, initial=-np.inf)
    )
    orthogonality = float(
        np.linalg.norm(result.A.T @ result.A - np.eye(result.A.shape[1]), ord="fro")
    )
    finite = _numeric_tree_finite(
        {
            "A": result.A,
            "B": result.B,
            "S": result.S,
            "objectives": result.objectives,
            "stationarity": result.stationarity,
            "block_stationarity": [
                asdict(value) for value in result.block_stationarity
            ],
            "updates": [asdict(value) for value in result.updates],
            "block_counts": asdict(result.block_counts),
            "work": asdict(result.work),
            "B_certificate_step_size": result.B_certificate_step_size,
            "S_certificate_step_size": result.S_certificate_step_size,
            "recomputed": asdict(recomputed),
            "runtime_seconds": runtime_seconds,
        }
    )
    certificate_match = bool(
        np.isclose(
            final.total,
            recomputed.total,
            rtol=float(gate["certificate_rtol"]),
            atol=float(gate["certificate_atol"]),
        )
        and diagnostic_difference
        <= float(gate["certificate_atol"])
        + float(gate["certificate_rtol"]) * max(1.0, abs(float(recomputed.total)))
    )
    delay_valid = _schedule_trace_valid(result.updates, int(solver["max_block_delay"]))
    trials_valid = all(
        update.backtracking_trials <= int(gate["maximum_backtracking_trials"])
        and update.backtracking_trials >= 0
        for update in result.updates
    )
    checks = {
        "finite": finite,
        "objective_monotone": maximum_objective_excess <= 0.0,
        "orthogonality": orthogonality <= float(gate["orthogonality_at_most"]),
        "complete_certificate_recomputed": certificate_match,
        "bounded_delay": delay_valid,
        "backtracking_trials": trials_valid,
        "converged": bool(
            result.converged
            and final.total <= float(gate["complete_stationarity_at_most"])
        ),
    }
    update_trace = [asdict(update) for update in result.updates]
    return {
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
        "converged": bool(result.converged),
        "status": result.status,
        "n_updates": result.n_updates,
        "initial_objective": float(result.objectives[0]),
        "final_objective": float(result.objectives[-1]),
        "initial_complete_stationarity": float(result.stationarity[0]),
        "final_complete_stationarity": float(result.stationarity[-1]),
        "final_block_stationarity": asdict(final),
        "independent_recomputed_stationarity": asdict(recomputed),
        "maximum_diagnostic_difference": diagnostic_difference,
        "A_orthogonality": orthogonality,
        "maximum_objective_excess": maximum_objective_excess,
        "B_certificate_step_size": result.B_certificate_step_size,
        "S_certificate_step_size": result.S_certificate_step_size,
        "block_counts": asdict(result.block_counts),
        "work": asdict(result.work),
        "checks": checks,
        "objectives": list(result.objectives),
        "stationarity": list(result.stationarity),
        "updates": update_trace,
    }


def _solver_exception_case(
    *,
    seed: int,
    error: Exception,
    runtime_seconds: float,
) -> dict[str, Any]:
    checks = {
        "finite": False,
        "objective_monotone": False,
        "orthogonality": False,
        "complete_certificate_recomputed": False,
        "bounded_delay": False,
        "backtracking_trials": False,
        "converged": False,
    }
    return {
        "seed": int(seed),
        "runtime_seconds": float(runtime_seconds),
        "converged": False,
        "status": "solver exception retained",
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "checks": checks,
    }


def _replace_nonfinite_for_retention(
    value: Any,
    *,
    path: str = "$",
) -> tuple[Any, list[str]]:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        paths: list[str] = []
        for key, item in value.items():
            replaced, nested = _replace_nonfinite_for_retention(
                item, path=f"{path}.{key}"
            )
            output[str(key)] = replaced
            paths.extend(nested)
        return output, paths
    if isinstance(value, (list, tuple)):
        output_list: list[Any] = []
        paths = []
        for index, item in enumerate(value):
            replaced, nested = _replace_nonfinite_for_retention(
                item, path=f"{path}[{index}]"
            )
            output_list.append(replaced)
            paths.extend(nested)
        return output_list, paths
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None, [path]
    if isinstance(value, np.generic):
        return value.item(), []
    return value, []


def run_truth_blind_smoke(
    protocol_path: Path,
    *,
    acknowledgement: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run the frozen solver-only smoke and retain either disposition."""

    if acknowledgement != RUN_ACKNOWLEDGEMENT:
        raise ValueError("solver-smoke acknowledgement is missing or incorrect")
    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    raw_protocol_path = Path(protocol_path)
    if raw_protocol_path.is_symlink():
        raise ValueError("protocol path must not be a symlink")
    protocol_path = raw_protocol_path.resolve()
    try:
        protocol_path.relative_to(root)
    except ValueError as error:
        raise ValueError("protocol path must be contained in the repository") from error
    protocol = _validate_protocol(protocol_path, root)
    static_audit_receipt, static_audit_receipt_path, static_audit_receipt_sha256 = (
        _validate_static_audit_receipt(protocol, root)
    )
    audit_receipt = _validate_audit_receipt(
        protocol,
        root,
        static_audit_receipt_sha256=static_audit_receipt_sha256,
    )
    pack_dir, pack_manifest, loaded_cases = _validate_pack(protocol, root)
    if pack_manifest["static_audit_receipt_sha256"] != static_audit_receipt_sha256:
        raise ValueError("blind pack is bound to a different static audit receipt")
    pack_manifest_path = pack_dir / PACK_MANIFEST_NAME
    _validate_audit_pack_provenance(
        audit_receipt,
        pack_manifest,
        protocol=protocol,
        manifest_file_sha256=_file_sha256(pack_manifest_path),
        static_audit_receipt_sha256=static_audit_receipt_sha256,
    )

    audit_receipt_path = _contained(
        root,
        _relative_path(
            protocol["audit_gate"]["receipt_path"], name="audit receipt path"
        ),
        name="audit receipt path",
    )
    protocol_bytes = _snapshot_validated_json(
        protocol_path, protocol, name="protocol snapshot"
    )
    pack_manifest_bytes = _snapshot_validated_json(
        pack_manifest_path,
        pack_manifest,
        name="blind-pack manifest snapshot",
    )
    audit_receipt_bytes = _snapshot_validated_json(
        audit_receipt_path,
        audit_receipt,
        name="audit receipt snapshot",
    )
    static_audit_receipt_bytes = _snapshot_validated_json(
        static_audit_receipt_path,
        static_audit_receipt,
        name="static audit receipt snapshot",
    )
    audit_receipt_sha256 = sha256(audit_receipt_bytes).hexdigest()
    if sha256(static_audit_receipt_bytes).hexdigest() != static_audit_receipt_sha256:
        raise RuntimeError("static audit receipt changed after validation")

    smoke_config = protocol["smoke_output"]
    output_dir = _contained(
        root,
        _relative_path(smoke_config["output_dir"], name="smoke output"),
        name="smoke output",
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"solver-smoke output is create-once: {output_dir}")

    cases: list[dict[str, Any]] = []
    for record, observed, initial_B in loaded_cases:
        case_started = perf_counter()
        try:
            case = _run_case(
                observed,
                initial_B,
                seed=int(record["seed"]),
                solver=protocol["solver_config"],
                gate=protocol["solver_gate"],
            )
        except Exception as error:
            cases.append(
                _solver_exception_case(
                    seed=int(record["seed"]),
                    error=error,
                    runtime_seconds=perf_counter() - case_started,
                )
            )
            break
        retained_case, nonfinite_fields = _replace_nonfinite_for_retention(case)
        if nonfinite_fields:
            retained_case["status"] = "non-finite numerical result retained"
            retained_case["converged"] = False
            retained_case["checks"]["finite"] = False
            retained_case["nonfinite_fields"] = nonfinite_fields
        cases.append(retained_case)

    hard_checks = [bool(value) for case in cases for value in case["checks"].values()]
    declared_case_count = int(pack_manifest["case_count"])
    passed = bool(
        hard_checks and all(hard_checks) and len(cases) == declared_case_count
    )
    report = {
        "schema_version": 1,
        "report_kind": "cleaned_input_solver_v3_truth_blind_smoke",
        "status": (
            "PASS_SOLVER_ONLY_TRUTH_BLIND_SMOKE"
            if passed
            else "FAIL_SOLVER_ONLY_TRUTH_BLIND_SMOKE_RETAINED"
        ),
        "passed": passed,
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "blind_pack_manifest_body_sha256": pack_manifest["manifest_body_sha256"],
        "independent_audit_receipt_sha256": audit_receipt_sha256,
        "static_audit_receipt_sha256": static_audit_receipt_sha256,
        "source_sha256": protocol["source_sha256"],
        "solver_config": protocol["solver_config"],
        "solver_gate": protocol["solver_gate"],
        "declared_case_count": declared_case_count,
        "case_count": len(cases),
        "cases": cases,
        "boundary": {
            "observed_only_solver_inputs": True,
            "statistical_metrics_computed": False,
            "model_selection_performed": False,
            "new_algorithm_validated": False,
            "statistical_or_formal_run_authorized": False,
            "blind_pack_semantics_recomputed_before_output_creation": True,
            "failure_policy": "retain and stop without changing scheduler, seeds, parameters, tolerance, delay, or budget",
        },
        "audit_status": audit_receipt["status"],
    }
    report_bytes = _strict_json_bytes(report)

    manifest = {
        "schema_version": 1,
        "manifest_kind": "cleaned_input_solver_v3_truth_blind_smoke_output",
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "report_status": report["status"],
        "blind_pack_semantic_validation": "passed_before_output_creation",
        "payload_sha256": {
            REPORT_NAME: sha256(report_bytes).hexdigest(),
            PROTOCOL_COPY_NAME: sha256(protocol_bytes).hexdigest(),
            PACK_MANIFEST_COPY_NAME: sha256(pack_manifest_bytes).hexdigest(),
        },
    }
    manifest["manifest_body_sha256"] = _canonical_sha256(manifest)
    manifest_bytes = _strict_json_bytes(manifest)

    # Only claim the create-once output root after every payload has been
    # serialized and hashed successfully in memory.  A numerical failure is a
    # retained report; a serialization/preflight failure must not leave an
    # empty directory that looks like that report.
    output_dir.mkdir(parents=True, exist_ok=False)
    payloads = {
        REPORT_NAME: report_bytes,
        PROTOCOL_COPY_NAME: protocol_bytes,
        PACK_MANIFEST_COPY_NAME: pack_manifest_bytes,
        OUTPUT_MANIFEST_NAME: manifest_bytes,
    }
    for name, payload in payloads.items():
        with (output_dir / name).open("xb") as handle:
            handle.write(payload)
    if {entry.name for entry in output_dir.iterdir()} != set(REQUIRED_OUTPUT_NAMES):
        raise RuntimeError("solver-smoke output inventory changed")
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--acknowledge-solver-smoke", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    report = run_truth_blind_smoke(
        arguments.protocol,
        acknowledgement=arguments.acknowledge_solver_smoke,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "passed": report["passed"],
                "case_count": report["case_count"],
                "statistical_or_formal_runs": 0,
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Independently replay and commit the v3 observed-only blind pack.

The pack's internal hashes prove consistency, not that its matrices came from
the frozen generator and seeds.  This verifier closes that provenance gap by
replaying the deterministic Figure 6 generator, normalization, and GESDD PCA
for every protocol seed and comparing their byte-level array hashes with the
pack.  It does not fit a solver, compute a truth metric, or issue the final
independent-audit receipt.

The public function is pure with respect to the filesystem.  The command-line
entry point emits deterministic JSON to stdout and has no output-path option.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg

from experiments.video_spca_paper_reproduction import (
    PaperVideoConfig,
    generate_figure6_dataset,
)


PROTOCOL_KIND = "cleaned_input_solver_v3_truth_blind_smoke_protocol"
PROTOCOL_STATE = "FROZEN_NOT_RUN"
PACK_KIND = "cleaned_input_solver_v3_observed_only_blind_pack"
PACK_MANIFEST_NAME = "blind_pack_manifest.json"
EXACT_NPZ_KEYS = ("observed_normalized", "initial_B")
COMMITMENT_KIND = "cleaned_input_solver_v3_blind_pack_generator_replay_commitments"
STATIC_AUDIT_KIND = "cleaned_input_solver_v3_static_pre_pack_independent_audit"
VERIFIER_SOURCE_PATH = (
    "experiments/audit_video_cleaned_input_solver_blind_pack_provenance.py"
)

DATA_CONFIG_KEYS = frozenset(
    {
        "grid_size",
        "n_timesteps",
        "total_time",
        "n_components",
        "corruption_fraction",
        "corruption_scale",
        "seeds",
    }
)
SOLVER_CONFIG_KEYS = frozenset(
    {
        "alpha",
        "kappa",
        "max_updates",
        "tol",
        "max_block_delay",
        "backtracking_expand",
    }
)
BLIND_PACK_KEYS = frozenset({"output_dir", "manifest_name", "exact_npz_keys"})
AUDIT_GATE_KEYS = frozenset(
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
STATIC_AUDIT_RECEIPT_KEYS = frozenset(
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
STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS = frozenset(
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
FRESH_SEED_CONTRACT_KEYS = frozenset(
    {
        "seed_history_scan_path",
        "seed_history_scan_sha256",
        "semantic_hits_before_freeze",
        "candidate_outcomes_before_freeze",
        "seed_replacement_allowed",
        "use_boundary",
    }
)
SEED_HISTORY_SCAN_KEYS = frozenset(
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
SEED_HISTORY_SCAN_KIND = (
    "video_cleaned_input_solver_v3_pre_freeze_semantic_seed_history_scan"
)
RUNTIME_SOURCE_PATHS = frozenset(
    {
        "experiments/__init__.py",
        VERIFIER_SOURCE_PATH,
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
PACK_MANIFEST_KEYS = frozenset(
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
PACK_CASE_KEYS = frozenset(
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
EXPECTED_TRUTH_EXCLUSION = {
    "clean_written": False,
    "corruption_written": False,
    "corruption_support_written": False,
    "time_written": False,
    "downstream_solver_inputs_are_observed_only": True,
}

_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
                raise TypeError("JSON object keys must be strings")
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


def _load_strict_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON input must be a regular non-symlink file: {path}")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise TypeError("JSON document must contain an object")
    _require_finite_json(value)
    return value


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
    return sha256(_strict_json_bytes(value).rstrip(b"\n")).hexdigest()


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


def _without_digest(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {str(key): item for key, item in value.items() if str(key) != field}


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if set(value) != set(expected):
        missing = sorted(set(expected).difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(f"{name} keys changed; missing={missing}, extra={extra}")
    return value


def _require_sha(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _relative_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path


def _contained(root: Path, relative: Path, *, name: str) -> Path:
    unresolved = root
    for component in relative.parts:
        unresolved = unresolved / component
        if unresolved.is_symlink():
            raise ValueError(f"{name} must not contain symlink components")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the repository root") from error
    return resolved


def _canonical_npz_bytes(
    observed_normalized: np.ndarray,
    initial_B: np.ndarray,
) -> bytes:
    buffer = BytesIO()
    np.savez(
        buffer,
        observed_normalized=observed_normalized,
        initial_B=initial_B,
    )
    return buffer.getvalue()


def _regular_file(path: Path, *, name: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or not stat.S_ISREG(path.lstat().st_mode)
    ):
        raise ValueError(f"{name} must be a regular non-symlink file")


def _validate_data_config(value: Any) -> dict[str, Any]:
    data = _require_exact_keys(value, DATA_CONFIG_KEYS, name="data_config")
    grid_size = _require_integer(data["grid_size"], name="grid_size", minimum=8)
    n_timesteps = _require_integer(data["n_timesteps"], name="n_timesteps", minimum=1)
    n_components = _require_integer(
        data["n_components"], name="n_components", minimum=1
    )
    if n_components != 3 or n_timesteps <= n_components:
        raise ValueError("video data_config must have rank 3 and more than 3 frames")
    total_time = _require_float(data["total_time"], name="total_time")
    fraction = _require_float(data["corruption_fraction"], name="corruption_fraction")
    scale = _require_float(data["corruption_scale"], name="corruption_scale")
    if total_time <= 0.0 or not 0.0 <= fraction < 1.0 or scale < 0.0:
        raise ValueError("invalid video data_config scalar")
    raw_seeds = data["seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ValueError("seeds must be a nonempty JSON list")
    seeds = [_require_integer(seed, name="seed", minimum=0) for seed in raw_seeds]
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be duplicate-free")
    return {
        "grid_size": grid_size,
        "n_timesteps": n_timesteps,
        "total_time": total_time,
        "n_components": n_components,
        "corruption_fraction": fraction,
        "corruption_scale": scale,
        "seeds": seeds,
    }


def _validate_solver_config(value: Any) -> dict[str, Any]:
    solver = _require_exact_keys(value, SOLVER_CONFIG_KEYS, name="solver_config")
    alpha = _require_float(solver["alpha"], name="alpha")
    kappa = _require_float(solver["kappa"], name="kappa")
    tol = _require_float(solver["tol"], name="tol")
    expand = _require_float(solver["backtracking_expand"], name="backtracking_expand")
    updates = _require_integer(solver["max_updates"], name="max_updates", minimum=1)
    delay = _require_integer(
        solver["max_block_delay"], name="max_block_delay", minimum=3
    )
    if (alpha, kappa) != (0.0015, 0.001):
        raise ValueError("locked alpha/kappa changed")
    if tol <= 0.0 or expand < 1.0:
        raise ValueError("invalid solver tolerance or backtracking expansion")
    return {
        "alpha": alpha,
        "kappa": kappa,
        "max_updates": updates,
        "tol": tol,
        "max_block_delay": delay,
        "backtracking_expand": expand,
    }


def _validate_fresh_seed_receipt(
    protocol: Mapping[str, Any],
    *,
    seeds: list[int],
    repo_root: Path,
) -> str:
    contract = _require_exact_keys(
        protocol["fresh_seed_contract"],
        FRESH_SEED_CONTRACT_KEYS,
        name="fresh_seed_contract",
    )
    relative = _relative_path(
        contract["seed_history_scan_path"], name="seed history scan path"
    )
    scan_path = _contained(repo_root, relative, name="seed history scan")
    _regular_file(scan_path, name="seed history scan")
    expected_file_sha = _require_sha(
        contract["seed_history_scan_sha256"], name="seed history scan SHA-256"
    )
    if _file_sha256(scan_path) != expected_file_sha:
        raise ValueError("seed history scan digest mismatch")
    scan = _load_strict_json(scan_path)
    _require_exact_keys(scan, SEED_HISTORY_SCAN_KEYS, name="seed history scan")
    if scan["schema_version"] != 1 or scan["scan_kind"] != SEED_HISTORY_SCAN_KIND:
        raise ValueError("unexpected seed history scan schema or kind")
    if scan["candidate_bank"] != seeds:
        raise ValueError("seed history candidate bank differs from protocol")
    if (
        isinstance(contract["semantic_hits_before_freeze"], bool)
        or contract["semantic_hits_before_freeze"] != 0
        or isinstance(scan["semantic_seed_hits_before_freeze"], bool)
        or scan["semantic_seed_hits_before_freeze"] != 0
    ):
        raise ValueError("fresh-seed receipt does not establish zero semantic hits")
    if (
        isinstance(contract["candidate_outcomes_before_freeze"], bool)
        or contract["candidate_outcomes_before_freeze"] != 0
        or isinstance(scan["candidate_outcomes_generated_before_freeze"], bool)
        or scan["candidate_outcomes_generated_before_freeze"] != 0
    ):
        raise ValueError("fresh-seed receipt does not establish zero prior outcomes")
    if (
        contract["seed_replacement_allowed"] is not False
        or scan["seed_replacement_allowed"] is not False
        or scan["use_boundary"] != contract["use_boundary"]
    ):
        raise ValueError("fresh-seed receipt boundary changed")
    return expected_file_sha


def _validate_audit_gate(value: Any) -> Mapping[str, Any]:
    gate = _require_exact_keys(value, AUDIT_GATE_KEYS, name="audit_gate")
    if gate["required"] is not True:
        raise ValueError("final independent audit must remain required")
    if gate["status_required"] != "passed":
        raise ValueError("audit status requirement must remain passed")
    if gate["auditor_independent_required"] is not True:
        raise ValueError("independent auditor must remain required")
    if gate["receipt_must_match_protocol_body_sha256"] is not True:
        raise ValueError("audit receipt must remain bound to the protocol")
    if gate["receipt_must_match_source_sha256"] is not True:
        raise ValueError("audit receipt must remain bound to the source map")
    if gate["static_receipt_required_before_blind_pack"] is not True:
        raise ValueError("static audit receipt must remain required before blind pack")
    final_receipt = _relative_path(
        gate["receipt_path"], name="final audit receipt path"
    )
    static_receipt = _relative_path(
        gate["static_receipt_path"], name="static audit receipt path"
    )
    if final_receipt == static_receipt:
        raise ValueError("static and final audit receipt paths must be distinct")
    return gate


def _validate_protocol(
    path: Path, repo_root: Path
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = _load_strict_json(path)
    required = {
        "schema_version",
        "protocol_kind",
        "protocol_body_sha256",
        "smoke_launch_state",
        "data_config",
        "solver_config",
        "blind_pack",
        "audit_gate",
        "fresh_seed_contract",
        "source_sha256",
    }
    missing = sorted(required.difference(protocol))
    if missing:
        raise ValueError(f"protocol is missing required fields: {missing}")
    if protocol["schema_version"] != 1 or protocol["protocol_kind"] != PROTOCOL_KIND:
        raise ValueError("unexpected protocol schema or kind")
    if protocol["smoke_launch_state"] != PROTOCOL_STATE:
        raise ValueError("protocol is not frozen and unrun")
    if protocol.get("smoke_launch_authorized_by_this_file") not in (None, False):
        raise ValueError("protocol must not self-authorize smoke execution")
    body_sha = _require_sha(
        protocol["protocol_body_sha256"], name="protocol_body_sha256"
    )
    if _canonical_sha256(_without_digest(protocol, "protocol_body_sha256")) != body_sha:
        raise ValueError("protocol body digest mismatch")
    data = _validate_data_config(protocol["data_config"])
    solver = _validate_solver_config(protocol["solver_config"])
    pack = _require_exact_keys(
        protocol["blind_pack"], BLIND_PACK_KEYS, name="blind_pack"
    )
    if pack["manifest_name"] != PACK_MANIFEST_NAME:
        raise ValueError("blind-pack manifest name changed")
    if pack["exact_npz_keys"] != list(EXACT_NPZ_KEYS):
        raise ValueError("blind-pack NPZ key whitelist changed")
    _validate_audit_gate(protocol["audit_gate"])
    sources = protocol["source_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != set(RUNTIME_SOURCE_PATHS):
        raise ValueError("protocol source map differs from the 22-file runtime closure")
    for raw_relative, raw_sha in sources.items():
        relative = _relative_path(raw_relative, name="source path")
        source_path = _contained(repo_root, relative, name=f"source {raw_relative}")
        _regular_file(source_path, name=f"source {raw_relative}")
        if _file_sha256(source_path) != _require_sha(
            raw_sha, name=f"source SHA-256 {raw_relative}"
        ):
            raise ValueError(f"source digest mismatch: {raw_relative}")
    if sources[VERIFIER_SOURCE_PATH] != _file_sha256(Path(__file__).resolve()):
        raise ValueError(
            "protocol verifier source digest differs from executing verifier"
        )
    scan_sha = _validate_fresh_seed_receipt(
        protocol, seeds=data["seeds"], repo_root=repo_root
    )
    return protocol, {"data": data, "solver": solver}, scan_sha


def _validate_static_audit_receipt(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> str:
    """Validate and hash the independent, pre-pack static audit receipt."""

    gate = _validate_audit_gate(protocol["audit_gate"])
    relative = _relative_path(
        gate["static_receipt_path"], name="static audit receipt path"
    )
    receipt_path = _contained(repo_root, relative, name="static audit receipt path")
    _regular_file(receipt_path, name="static audit receipt")
    receipt = _load_strict_json(receipt_path)
    _require_exact_keys(receipt, STATIC_AUDIT_RECEIPT_KEYS, name="static audit receipt")
    body_sha = _require_sha(
        receipt["receipt_body_sha256"], name="static audit receipt body SHA-256"
    )
    if _canonical_sha256(_without_digest(receipt, "receipt_body_sha256")) != body_sha:
        raise ValueError("static audit receipt body digest mismatch")
    if receipt["schema_version"] != 1 or receipt["audit_kind"] != STATIC_AUDIT_KIND:
        raise ValueError("unexpected static audit receipt schema or kind")
    if receipt["status"] != gate["status_required"]:
        raise ValueError("static independent audit did not pass")
    if receipt["auditor_independent"] is not True:
        raise ValueError("static audit receipt is not independent")
    for field in ("auditor_id", "audit_timestamp", "focused_test_command"):
        value = receipt[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"static audit receipt requires nonempty {field}")
    if receipt["protocol_body_sha256"] != protocol["protocol_body_sha256"]:
        raise ValueError("static audit receipt protocol digest mismatch")
    if receipt["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("static audit receipt source map mismatch")
    checked_nonruntime = receipt["checked_nonruntime_sha256"]
    if not isinstance(checked_nonruntime, Mapping):
        raise TypeError("static audit checked_nonruntime_sha256 must be an object")
    dynamic_scan_path = str(
        _relative_path(
            protocol["fresh_seed_contract"]["seed_history_scan_path"],
            name="seed history scan path",
        )
    )
    required_nonruntime = set(STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS)
    required_nonruntime.add(dynamic_scan_path)
    if not required_nonruntime.issubset(checked_nonruntime):
        missing = sorted(required_nonruntime.difference(checked_nonruntime))
        raise ValueError(
            "static audit checked_nonruntime_sha256 lacks required files: " f"{missing}"
        )
    for raw_relative, raw_sha in checked_nonruntime.items():
        relative = _relative_path(raw_relative, name="static audit nonruntime path")
        checked_path = _contained(
            repo_root,
            relative,
            name=f"static audit nonruntime file {raw_relative}",
        )
        _regular_file(checked_path, name=f"static audit nonruntime file {raw_relative}")
        if _file_sha256(checked_path) != _require_sha(
            raw_sha, name=f"static audit nonruntime SHA-256 {raw_relative}"
        ):
            raise ValueError(f"static audit nonruntime digest mismatch: {raw_relative}")
    _require_integer(
        receipt["focused_test_count"], name="focused_test_count", minimum=1
    )
    if receipt["focused_tests_passed"] is not True:
        raise ValueError("static audit focused tests did not pass")
    if receipt["blind_pack_absent_at_audit"] is not True:
        raise ValueError("static audit did not establish blind-pack absence")
    for field in (
        "candidate_pack_generations_before_audit",
        "candidate_outcomes_observed_before_audit",
        "solver_smoke_runs_before_audit",
    ):
        count = receipt[field]
        if isinstance(count, bool) or not isinstance(count, int) or count != 0:
            raise ValueError(f"static audit receipt requires {field}=0")
    if receipt["statistical_or_formal_run_authorized"] is not False:
        raise ValueError("static audit receipt exceeds solver-smoke scope")
    return _file_sha256(receipt_path)


def _paper_config(
    data: Mapping[str, Any],
    solver: Mapping[str, Any],
    seed: int,
) -> PaperVideoConfig:
    return PaperVideoConfig(
        grid_size=int(data["grid_size"]),
        n_timesteps=int(data["n_timesteps"]),
        total_time=float(data["total_time"]),
        n_components=int(data["n_components"]),
        robust_alpha=float(solver["alpha"]),
        robust_kappa=float(solver["kappa"]),
        corruption_fraction=float(data["corruption_fraction"]),
        corruption_scale=float(data["corruption_scale"]),
        corruption_seed=int(seed),
        figure6_frame_index=min(10, int(data["n_timesteps"]) - 1),
        dpi=72,
    )


def _replay_observed_only(
    data: Mapping[str, Any],
    solver: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dataset = generate_figure6_dataset(_paper_config(data, solver, seed))
    observed = np.ascontiguousarray(dataset.observed, dtype=np.float64)
    _, singular_values, right_t = linalg.svd(
        observed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if singular_values.size < int(data["n_components"]):
        raise ValueError("replayed observed matrix has too few singular vectors")
    sigma1 = float(singular_values[0])
    normalized = np.ascontiguousarray(
        observed / max(sigma1, np.finfo(np.float64).eps), dtype=np.float64
    )
    initial_B = np.ascontiguousarray(
        right_t[: int(data["n_components"])].T, dtype=np.float64
    )
    return observed, normalized, initial_B, sigma1


def _validate_case_record(
    case: Any,
    *,
    expected_seed: int,
    expected_shape: tuple[int, int],
) -> Mapping[str, Any]:
    record = _require_exact_keys(case, PACK_CASE_KEYS, name="pack case")
    seed = _require_integer(record["seed"], name="pack case seed", minimum=0)
    if seed != expected_seed or record["npz_name"] != f"observed_case_{seed}.npz":
        raise ValueError("pack case seed or filename differs from protocol order")
    shape = record["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or tuple(shape) != expected_shape
    ):
        raise ValueError("pack case shape differs from frozen data config")
    sigma1 = _require_float(record["sigma1"], name="pack case sigma1")
    if sigma1 < 0.0:
        raise ValueError("pack case sigma1 must be nonnegative")
    size = _require_integer(
        record["npz_size_bytes"], name="pack case NPZ size", minimum=1
    )
    if size <= 0:
        raise ValueError("pack case NPZ size must be positive")
    for field in (
        "observed_sha256",
        "observed_normalized_sha256",
        "initial_B_sha256",
        "npz_sha256",
    ):
        _require_sha(record[field], name=f"pack case {field}")
    return record


def audit_blind_pack_provenance(
    protocol_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Replay every seed and return deterministic, unsigned commitments.

    The function performs no writes.  A mismatch raises before any commitment
    object is returned.
    """

    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("repo_root must be a real directory")
    protocol_path = Path(protocol_path)
    protocol, validated, seed_scan_sha = _validate_protocol(protocol_path, root)
    static_audit_receipt_sha = _validate_static_audit_receipt(protocol, repo_root=root)
    data = validated["data"]
    solver = validated["solver"]
    pack_relative = _relative_path(
        protocol["blind_pack"]["output_dir"], name="blind-pack output"
    )
    pack_dir = _contained(root, pack_relative, name="blind-pack output")
    if pack_dir.is_symlink() or not pack_dir.is_dir():
        raise ValueError("blind-pack output must be a real directory")
    manifest_path = pack_dir / PACK_MANIFEST_NAME
    _regular_file(manifest_path, name="blind-pack manifest")
    manifest = _load_strict_json(manifest_path)
    _require_exact_keys(manifest, PACK_MANIFEST_KEYS, name="blind-pack manifest")
    manifest_body_sha = _require_sha(
        manifest["manifest_body_sha256"], name="blind-pack manifest body SHA-256"
    )
    if (
        _canonical_sha256(_without_digest(manifest, "manifest_body_sha256"))
        != manifest_body_sha
    ):
        raise ValueError("blind-pack manifest body digest mismatch")
    if manifest["schema_version"] != 1 or manifest["manifest_kind"] != PACK_KIND:
        raise ValueError("unexpected blind-pack manifest schema or kind")
    if manifest["protocol_kind"] != PROTOCOL_KIND:
        raise ValueError("blind-pack protocol kind mismatch")
    if manifest["protocol_body_sha256"] != protocol["protocol_body_sha256"]:
        raise ValueError("blind-pack protocol digest mismatch")
    if manifest["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("blind-pack source map mismatch")
    if manifest["data_config"] != protocol["data_config"]:
        raise ValueError("blind-pack data config mismatch")
    if manifest["solver_config"] != protocol["solver_config"]:
        raise ValueError("blind-pack solver config mismatch")
    if manifest["exact_npz_keys"] != list(EXACT_NPZ_KEYS):
        raise ValueError("blind-pack NPZ whitelist changed")
    if manifest["truth_exclusion"] != EXPECTED_TRUTH_EXCLUSION:
        raise ValueError("blind-pack truth exclusion declaration changed")
    manifest_static_receipt_sha = _require_sha(
        manifest["static_audit_receipt_sha256"],
        name="blind-pack static audit receipt SHA-256",
    )
    if manifest_static_receipt_sha != static_audit_receipt_sha:
        raise ValueError("blind-pack static audit receipt digest mismatch")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("blind-pack cases must be a nonempty list")
    case_count = _require_integer(
        manifest["case_count"], name="blind-pack case count", minimum=1
    )
    if case_count != len(cases) or case_count != len(data["seeds"]):
        raise ValueError("blind-pack case count differs from protocol")

    expected_shape = (
        int(data["grid_size"]) ** 2,
        int(data["n_timesteps"]),
    )
    expected_entries = {PACK_MANIFEST_NAME}
    commitments: list[dict[str, Any]] = []
    for case, seed in zip(cases, data["seeds"], strict=True):
        record = _validate_case_record(
            case,
            expected_seed=int(seed),
            expected_shape=expected_shape,
        )
        case_path = pack_dir / str(record["npz_name"])
        expected_entries.add(case_path.name)
        _regular_file(case_path, name=f"blind-pack case {seed}")
        if case_path.stat().st_size != record["npz_size_bytes"]:
            raise ValueError(f"blind-pack NPZ size mismatch for seed {seed}")
        if _file_sha256(case_path) != record["npz_sha256"]:
            raise ValueError(f"blind-pack NPZ digest mismatch for seed {seed}")
        with np.load(case_path, allow_pickle=False) as archive:
            if archive.files != list(EXACT_NPZ_KEYS):
                raise ValueError(
                    f"blind-pack NPZ key whitelist mismatch for seed {seed}"
                )
            packed_normalized = np.ascontiguousarray(archive["observed_normalized"])
            packed_B = np.ascontiguousarray(archive["initial_B"])
        if case_path.read_bytes() != _canonical_npz_bytes(packed_normalized, packed_B):
            raise ValueError(
                f"blind-pack NPZ is not the canonical two-array container "
                f"for seed {seed}"
            )
        if packed_normalized.dtype != np.float64 or packed_B.dtype != np.float64:
            raise ValueError(f"blind-pack dtype mismatch for seed {seed}")
        if (
            packed_normalized.shape != expected_shape
            or packed_B.shape != (expected_shape[1], int(data["n_components"]))
            or not np.isfinite(packed_normalized).all()
            or not np.isfinite(packed_B).all()
        ):
            raise ValueError(
                f"blind-pack shape or finite-value mismatch for seed {seed}"
            )

        observed, replay_normalized, replay_B, replay_sigma1 = _replay_observed_only(
            data, solver, int(seed)
        )
        replay_hashes = {
            "observed_sha256": _array_sha256(observed),
            "observed_normalized_sha256": _array_sha256(replay_normalized),
            "initial_B_sha256": _array_sha256(replay_B),
        }
        if tuple(observed.shape) != expected_shape:
            raise ValueError(f"generator replay shape mismatch for seed {seed}")
        if record["observed_sha256"] != replay_hashes["observed_sha256"]:
            raise ValueError(f"generator replay observed hash mismatch for seed {seed}")
        if (
            record["observed_normalized_sha256"]
            != replay_hashes["observed_normalized_sha256"]
            or _array_sha256(packed_normalized)
            != replay_hashes["observed_normalized_sha256"]
        ):
            raise ValueError(
                f"generator replay normalized hash mismatch for seed {seed}"
            )
        if (
            record["initial_B_sha256"] != replay_hashes["initial_B_sha256"]
            or _array_sha256(packed_B) != replay_hashes["initial_B_sha256"]
        ):
            raise ValueError(f"generator replay PCA B hash mismatch for seed {seed}")
        if float(record["sigma1"]).hex() != replay_sigma1.hex():
            raise ValueError(f"generator replay sigma1 mismatch for seed {seed}")
        commitments.append(
            {
                "seed": int(seed),
                "observed_sha256": replay_hashes["observed_sha256"],
                "observed_normalized_sha256": replay_hashes[
                    "observed_normalized_sha256"
                ],
                "initial_B_sha256": replay_hashes["initial_B_sha256"],
                "sigma1": replay_sigma1,
            }
        )

    if {entry.name for entry in pack_dir.iterdir()} != expected_entries:
        raise ValueError("blind-pack directory inventory changed")
    verifier_path = Path(__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "commitment_kind": COMMITMENT_KIND,
        "verification_status": "PASS_GENERATOR_REPLAY_COMMITMENTS",
        "not_a_final_audit_receipt": True,
        "protocol_body_sha256": protocol["protocol_body_sha256"],
        "source_map_sha256": _canonical_sha256(
            {"source_sha256": protocol["source_sha256"]}
        ),
        "fresh_seed_scan_file_sha256": seed_scan_sha,
        "static_audit_receipt_sha256": static_audit_receipt_sha,
        "blind_pack_manifest_body_sha256": manifest_body_sha,
        "blind_pack_manifest_file_sha256": _file_sha256(manifest_path),
        "verifier_source_sha256": _file_sha256(verifier_path),
        "case_count": len(commitments),
        "ordered_cases": commitments,
        "checks": {
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
        },
        "boundary": {
            "solver_fits": 0,
            "statistical_metrics": 0,
            "truth_metrics": 0,
            "artifact_filesystem_writes": 0,
            "final_audit_receipts_issued": 0,
        },
    }
    result["commitment_body_sha256"] = _canonical_sha256(result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    result = audit_blind_pack_provenance(arguments.protocol)
    sys.stdout.buffer.write(_strict_json_bytes(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

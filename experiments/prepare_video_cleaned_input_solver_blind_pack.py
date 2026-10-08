"""Create an observed-only input pack for the v3 solver smoke.

This preparer is deliberately separated from the solver runner.  It is the
only stage that calls the truth-bearing synthetic video generator.  Each
retained ``.npz`` contains exactly the normalized observed matrix and the
shared observed-PCA loading initialization; clean signal, corruption,
support, and time arrays are never written.

Preparation is fail-closed behind a self-digested frozen protocol and an
explicit acknowledgement.  The output directory is create-once.  A partial
directory is intentionally retained after a write failure rather than being
silently removed and reused.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import stat
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import linalg

from experiments.video_spca_paper_reproduction import (
    PaperVideoConfig,
    generate_figure6_dataset,
)
from rspca.datasets import array_sha256, canonical_sha256


PROTOCOL_KIND = "cleaned_input_solver_v3_truth_blind_smoke_protocol"
PROTOCOL_STATE = "FROZEN_NOT_RUN"
PREPARATION_ACKNOWLEDGEMENT = "PREPARE_LOCKED_OBSERVED_ONLY_PACK"
MANIFEST_KIND = "cleaned_input_solver_v3_observed_only_blind_pack"
MANIFEST_NAME = "blind_pack_manifest.json"
EXACT_NPZ_KEYS = ("observed_normalized", "initial_B")
STATIC_AUDIT_KIND = "cleaned_input_solver_v3_static_pre_pack_independent_audit"

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
_CASE_RECORD_KEYS = frozenset(
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
_MANIFEST_KEYS = frozenset(
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


@dataclass(frozen=True)
class BlindDataConfig:
    """Synthetic input fields allowed to affect the observed-only pack."""

    grid_size: int
    n_timesteps: int
    total_time: float
    n_components: int
    corruption_fraction: float
    corruption_scale: float
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class BlindSolverConfig:
    """Frozen solver settings retained in the protocol but unused here."""

    alpha: float
    kappa: float
    max_updates: int
    tol: float
    max_block_delay: int
    backtracking_expand: float


@dataclass(frozen=True)
class ValidatedBlindPackProtocol:
    """Validated protocol plus its resolved, repository-contained output."""

    raw: Mapping[str, Any]
    data_config: BlindDataConfig
    solver_config: BlindSolverConfig
    output_dir: Path
    repo_root: Path


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"strict JSON forbids non-finite constant {token}")


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"strict JSON forbids duplicate key {key!r}")
        output[key] = value
    return output


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


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protocol_body(protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in protocol.items()
        if str(key) != "protocol_body_sha256"
    }


def _require_exact_keys(
    value: Any,
    expected: frozenset[str],
    *,
    label: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    if set(value) != set(expected):
        missing = sorted(set(expected).difference(value))
        extra = sorted(set(value).difference(expected))
        raise ValueError(f"{label} keys differ; missing={missing}, extra={extra}")
    return value


def _integer(value: Any, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _data_config(value: Any) -> BlindDataConfig:
    mapping = _require_exact_keys(value, DATA_CONFIG_KEYS, label="data_config")
    grid_size = _integer(mapping["grid_size"], name="grid_size", minimum=8)
    n_timesteps = _integer(mapping["n_timesteps"], name="n_timesteps", minimum=1)
    n_components = _integer(mapping["n_components"], name="n_components", minimum=1)
    if n_components != 3:
        raise ValueError("the frozen video construction requires n_components=3")
    if n_timesteps <= n_components:
        raise ValueError("n_timesteps must exceed n_components")
    total_time = _finite_float(mapping["total_time"], name="total_time")
    corruption_fraction = _finite_float(
        mapping["corruption_fraction"], name="corruption_fraction"
    )
    corruption_scale = _finite_float(
        mapping["corruption_scale"], name="corruption_scale"
    )
    if total_time <= 0.0:
        raise ValueError("total_time must be positive")
    if not 0.0 <= corruption_fraction < 1.0:
        raise ValueError("corruption_fraction must lie in [0,1)")
    if corruption_scale < 0.0:
        raise ValueError("corruption_scale must be nonnegative")
    seed_values = mapping["seeds"]
    if not isinstance(seed_values, list) or not seed_values:
        raise ValueError("seeds must be a nonempty JSON list")
    seeds = tuple(_integer(seed, name="seed", minimum=0) for seed in seed_values)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be duplicate-free")
    return BlindDataConfig(
        grid_size=grid_size,
        n_timesteps=n_timesteps,
        total_time=total_time,
        n_components=n_components,
        corruption_fraction=corruption_fraction,
        corruption_scale=corruption_scale,
        seeds=seeds,
    )


def _solver_config(value: Any) -> BlindSolverConfig:
    mapping = _require_exact_keys(value, SOLVER_CONFIG_KEYS, label="solver_config")
    alpha = _finite_float(mapping["alpha"], name="alpha")
    kappa = _finite_float(mapping["kappa"], name="kappa")
    tol = _finite_float(mapping["tol"], name="tol")
    backtracking_expand = _finite_float(
        mapping["backtracking_expand"], name="backtracking_expand"
    )
    max_updates = _integer(mapping["max_updates"], name="max_updates", minimum=1)
    max_block_delay = _integer(
        mapping["max_block_delay"], name="max_block_delay", minimum=3
    )
    if (alpha, kappa) != (0.0015, 0.001):
        raise ValueError("locked alpha/kappa must remain 0.0015/0.001")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if backtracking_expand < 1.0:
        raise ValueError("backtracking_expand must be at least 1")
    return BlindSolverConfig(
        alpha=alpha,
        kappa=kappa,
        max_updates=max_updates,
        tol=tol,
        max_block_delay=max_block_delay,
        backtracking_expand=backtracking_expand,
    )


def _relative_path(raw: Any, *, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{name} must be a nonempty string")
    path = Path(raw)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ValueError(f"{name} must be a contained relative path")
    return path


def _contained_path(root: Path, relative: Path, *, name: str) -> Path:
    lexical = root
    for component in relative.parts:
        lexical = lexical / component
        if lexical.is_symlink():
            raise ValueError(f"{name} must not contain symlink components")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes the repository root") from error
    return resolved


def _validate_source_sha256(
    value: Any,
    *,
    repo_root: Path,
) -> dict[str, str]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source_sha256 must be a nonempty object")
    if set(value) != set(MINIMUM_SOURCE_PATHS):
        missing = sorted(MINIMUM_SOURCE_PATHS.difference(value))
        extra = sorted(set(value).difference(MINIMUM_SOURCE_PATHS))
        raise ValueError(
            "source_sha256 differs from audited 22-file closure; "
            f"missing={missing}, extra={extra}"
        )
    output: dict[str, str] = {}
    for raw_relative, raw_digest in value.items():
        relative = _relative_path(raw_relative, name="source_sha256 path")
        if not isinstance(raw_digest, str) or _HEX_SHA256.fullmatch(raw_digest) is None:
            raise ValueError(f"source_sha256 has an invalid digest: {raw_relative}")
        unresolved_source = repo_root / relative
        if unresolved_source.is_symlink():
            raise ValueError(f"source must not be a symlink: {raw_relative}")
        source = _contained_path(
            repo_root, relative, name=f"source_sha256 path {raw_relative}"
        )
        if source.is_symlink() or not source.is_file():
            raise ValueError(
                f"source is not a regular non-symlink file: {raw_relative}"
            )
        if not stat.S_ISREG(source.lstat().st_mode):
            raise ValueError(f"source is not a regular file: {raw_relative}")
        if _file_sha256(source) != raw_digest:
            raise ValueError(f"source digest mismatch: {raw_relative}")
        output[str(relative)] = raw_digest
    return output


def _validate_fresh_seed_contract(
    value: Any,
    *,
    data_seeds: tuple[int, ...],
    repo_root: Path,
) -> None:
    contract = _require_exact_keys(
        value, FRESH_SEED_CONTRACT_KEYS, label="fresh_seed_contract"
    )
    relative = _relative_path(
        contract["seed_history_scan_path"], name="seed history scan path"
    )
    reported_digest = contract["seed_history_scan_sha256"]
    if (
        not isinstance(reported_digest, str)
        or _HEX_SHA256.fullmatch(reported_digest) is None
    ):
        raise ValueError("seed history scan digest must be a lowercase SHA-256")
    unresolved = repo_root / relative
    if unresolved.is_symlink():
        raise ValueError("seed history scan must not be a symlink")
    scan_path = _contained_path(repo_root, relative, name="seed history scan path")
    scan = _load_strict_json(scan_path)
    if _file_sha256(scan_path) != reported_digest:
        raise ValueError("seed history scan digest mismatch")
    _require_exact_keys(scan, SEED_HISTORY_SCAN_KEYS, label="seed history scan")
    if scan["schema_version"] != 1 or scan["scan_kind"] != SEED_HISTORY_SCAN_KIND:
        raise ValueError("unexpected seed history scan schema or kind")
    if scan["candidate_bank"] != list(data_seeds):
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


def _validated_audit_gate(value: Any) -> Mapping[str, Any]:
    gate = _require_exact_keys(value, AUDIT_GATE_KEYS, label="audit_gate")
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


def _validate_static_audit_receipt(
    protocol: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[Path, str]:
    """Validate the independent pre-pack receipt without generating any data."""

    gate = _validated_audit_gate(protocol["audit_gate"])
    receipt_path = _contained_path(
        repo_root,
        _relative_path(gate["static_receipt_path"], name="static audit receipt path"),
        name="static audit receipt path",
    )
    receipt = _load_strict_json(receipt_path)
    _require_exact_keys(
        receipt,
        STATIC_AUDIT_RECEIPT_KEYS,
        label="static audit receipt",
    )
    reported_body_sha = receipt["receipt_body_sha256"]
    if (
        not isinstance(reported_body_sha, str)
        or _HEX_SHA256.fullmatch(reported_body_sha) is None
    ):
        raise ValueError("static audit receipt body digest must be a lowercase SHA-256")
    receipt_body = {
        str(key): item
        for key, item in receipt.items()
        if str(key) != "receipt_body_sha256"
    }
    if canonical_sha256(receipt_body) != reported_body_sha:
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
    if receipt["focused_tests_passed"] is not True:
        raise ValueError("static focused tests did not pass")
    focused_test_count = receipt["focused_test_count"]
    if (
        isinstance(focused_test_count, bool)
        or not isinstance(focused_test_count, int)
        or focused_test_count < 1
    ):
        raise ValueError("static audit focused_test_count must be positive")
    if receipt["protocol_body_sha256"] != protocol["protocol_body_sha256"]:
        raise ValueError("static audit receipt protocol digest mismatch")
    if receipt["source_sha256"] != protocol["source_sha256"]:
        raise ValueError("static audit receipt source map mismatch")
    checked_nonruntime = receipt["checked_nonruntime_sha256"]
    if not isinstance(checked_nonruntime, Mapping):
        raise TypeError("static checked_nonruntime_sha256 must be an object")
    required_nonruntime = set(STATIC_AUDIT_MINIMUM_NONRUNTIME_PATHS)
    required_nonruntime.add(
        str(
            _relative_path(
                protocol["fresh_seed_contract"]["seed_history_scan_path"],
                name="seed history scan path",
            )
        )
    )
    if not required_nonruntime.issubset(checked_nonruntime):
        missing = sorted(required_nonruntime.difference(checked_nonruntime))
        raise ValueError(f"static audit lacks nonruntime evidence: {missing}")
    for raw_path, raw_digest in checked_nonruntime.items():
        relative = _relative_path(raw_path, name="static nonruntime evidence path")
        expected_digest = raw_digest
        if (
            not isinstance(expected_digest, str)
            or _HEX_SHA256.fullmatch(expected_digest) is None
        ):
            raise ValueError("static nonruntime evidence digest is invalid")
        evidence_path = _contained_path(
            repo_root,
            relative,
            name=f"static nonruntime evidence {raw_path}",
        )
        if evidence_path.is_symlink() or not evidence_path.is_file():
            raise ValueError("static nonruntime evidence must be a regular file")
        if _file_sha256(evidence_path) != expected_digest:
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
    return receipt_path, _file_sha256(receipt_path)


def load_and_validate_protocol(
    path: Path,
    *,
    repo_root: Path | None = None,
) -> ValidatedBlindPackProtocol:
    """Load the self-digested frozen protocol and validate every pack input."""

    root = (
        Path(__file__).resolve().parents[1]
        if repo_root is None
        else Path(repo_root).resolve()
    )
    if root.is_symlink() or not root.is_dir():
        raise ValueError("repo_root must be a real directory")
    raw_protocol_path = Path(path)
    if raw_protocol_path.is_symlink():
        raise ValueError("protocol path must not be a symlink")
    resolved_protocol_path = raw_protocol_path.resolve()
    try:
        resolved_protocol_path.relative_to(root)
    except ValueError as error:
        raise ValueError("protocol path must be contained in the repository") from error
    protocol = _load_strict_json(resolved_protocol_path)
    required_top_level = {
        "schema_version",
        "protocol_kind",
        "smoke_launch_state",
        "data_config",
        "solver_config",
        "blind_pack",
        "audit_gate",
        "fresh_seed_contract",
        "source_sha256",
        "protocol_body_sha256",
    }
    missing = sorted(required_top_level.difference(protocol))
    if missing:
        raise ValueError(f"protocol is missing required fields: {missing}")
    if _integer(protocol["schema_version"], name="schema_version", minimum=1) != 1:
        raise ValueError("unsupported protocol schema_version")
    if protocol["protocol_kind"] != PROTOCOL_KIND:
        raise ValueError("unexpected protocol kind")
    if protocol["smoke_launch_state"] != PROTOCOL_STATE:
        raise ValueError("protocol is not in FROZEN_NOT_RUN state")
    if protocol.get("smoke_launch_authorized_by_this_file") not in (None, False):
        raise ValueError("protocol must not self-authorize smoke launch")
    reported_digest = protocol["protocol_body_sha256"]
    if (
        not isinstance(reported_digest, str)
        or _HEX_SHA256.fullmatch(reported_digest) is None
    ):
        raise ValueError("protocol_body_sha256 must be a lowercase SHA-256")
    if canonical_sha256(_protocol_body(protocol)) != reported_digest:
        raise ValueError("protocol body digest mismatch")

    data_config = _data_config(protocol["data_config"])
    solver_config = _solver_config(protocol["solver_config"])
    blind_pack = _require_exact_keys(
        protocol["blind_pack"], BLIND_PACK_KEYS, label="blind_pack"
    )
    if blind_pack["manifest_name"] != MANIFEST_NAME:
        raise ValueError(f"blind_pack manifest_name must be {MANIFEST_NAME}")
    if blind_pack["exact_npz_keys"] != list(EXACT_NPZ_KEYS):
        raise ValueError("blind_pack exact_npz_keys changed")
    relative_output = _relative_path(
        blind_pack["output_dir"], name="blind_pack.output_dir"
    )
    output_dir = _contained_path(root, relative_output, name="blind_pack.output_dir")
    _validated_audit_gate(protocol["audit_gate"])
    _validate_source_sha256(protocol["source_sha256"], repo_root=root)
    _validate_fresh_seed_contract(
        protocol["fresh_seed_contract"],
        data_seeds=data_config.seeds,
        repo_root=root,
    )
    return ValidatedBlindPackProtocol(
        raw=protocol,
        data_config=data_config,
        solver_config=solver_config,
        output_dir=output_dir,
        repo_root=root,
    )


def _paper_config(config: BlindDataConfig, seed: int) -> PaperVideoConfig:
    return PaperVideoConfig(
        grid_size=config.grid_size,
        n_timesteps=config.n_timesteps,
        total_time=config.total_time,
        n_components=config.n_components,
        robust_alpha=0.0015,
        robust_kappa=0.001,
        corruption_fraction=config.corruption_fraction,
        corruption_scale=config.corruption_scale,
        corruption_seed=seed,
        figure6_frame_index=min(10, config.n_timesteps - 1),
        dpi=72,
    )


def _prepare_observed_only(
    observed_input: Any,
    *,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    observed = np.ascontiguousarray(observed_input, dtype=np.float64)
    if observed.ndim != 2 or not np.isfinite(observed).all():
        raise ValueError("generated observed matrix must be finite and two-dimensional")
    _, singular_values, right_t = linalg.svd(
        observed,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if singular_values.size < n_components:
        raise ValueError("observed matrix has too few singular vectors")
    sigma1 = float(singular_values[0]) if singular_values.size else 0.0
    if not np.isfinite(sigma1) or sigma1 < 0.0:
        raise ValueError("observed PCA produced an invalid leading singular value")
    scale = max(sigma1, np.finfo(np.float64).eps)
    normalized = np.ascontiguousarray(observed / scale, dtype=np.float64)
    initial_B = np.ascontiguousarray(right_t[:n_components].T, dtype=np.float64)
    if not np.isfinite(normalized).all() or not np.isfinite(initial_B).all():
        raise ValueError("observed-only preparation produced non-finite arrays")
    return normalized, initial_B, sigma1


def _validate_observed_only_semantics(
    normalized: np.ndarray,
    initial_B: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    n_components: int,
    sigma1: float,
) -> None:
    if normalized.shape != expected_shape:
        raise ValueError("observed_normalized shape differs from frozen data config")
    if initial_B.shape != (expected_shape[1], n_components):
        raise ValueError("initial_B shape differs from frozen data config")
    _, singular_values, right_t = linalg.svd(
        normalized,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    if singular_values.size < n_components:
        raise ValueError("observed_normalized has too few singular vectors")
    normalized_sigma1 = float(singular_values[0]) if singular_values.size else 0.0
    expected_sigma1 = sigma1 / max(sigma1, np.finfo(np.float64).eps)
    if not np.isclose(
        normalized_sigma1,
        expected_sigma1,
        rtol=1e-12,
        atol=1e-14,
    ):
        raise ValueError("observed_normalized does not match the sigma1 scale contract")
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


def _case_file_name(seed: int) -> str:
    return f"observed_case_{seed}.npz"


def _write_npz_create_once(
    path: Path,
    *,
    observed_normalized: np.ndarray,
    initial_B: np.ndarray,
) -> None:
    with path.open("xb") as handle:
        np.savez(
            handle,
            observed_normalized=observed_normalized,
            initial_B=initial_B,
        )


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


def prepare_blind_pack(
    protocol_path: Path,
    *,
    acknowledgement: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize the frozen observed-only pack exactly once."""

    if acknowledgement != PREPARATION_ACKNOWLEDGEMENT:
        raise ValueError("pack preparation acknowledgement is missing or incorrect")
    validated = load_and_validate_protocol(protocol_path, repo_root=repo_root)
    _, static_audit_receipt_sha256 = _validate_static_audit_receipt(
        validated.raw,
        repo_root=validated.repo_root,
    )
    output_dir = validated.output_dir
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"blind-pack output is create-once: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    cases: list[dict[str, Any]] = []
    config = validated.data_config
    expected_shape = (config.grid_size * config.grid_size, config.n_timesteps)
    for seed in config.seeds:
        # This is the sole contact with the truth-bearing generator.  Only its
        # observed field crosses the following line.
        dataset = generate_figure6_dataset(_paper_config(config, seed))
        observed = np.ascontiguousarray(dataset.observed, dtype=np.float64)
        if observed.shape != expected_shape:
            raise ValueError("generated observed matrix has an unexpected shape")
        observed_hash = array_sha256(observed)
        normalized, initial_B, sigma1 = _prepare_observed_only(
            observed,
            n_components=config.n_components,
        )
        npz_name = _case_file_name(seed)
        npz_path = output_dir / npz_name
        _write_npz_create_once(
            npz_path,
            observed_normalized=normalized,
            initial_B=initial_B,
        )
        cases.append(
            {
                "seed": seed,
                "npz_name": npz_name,
                "shape": list(expected_shape),
                "sigma1": sigma1,
                "observed_sha256": observed_hash,
                "observed_normalized_sha256": array_sha256(normalized),
                "initial_B_sha256": array_sha256(initial_B),
                "npz_sha256": _file_sha256(npz_path),
                "npz_size_bytes": npz_path.stat().st_size,
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": MANIFEST_KIND,
        "protocol_kind": PROTOCOL_KIND,
        "protocol_body_sha256": validated.raw["protocol_body_sha256"],
        "source_sha256": dict(validated.raw["source_sha256"]),
        "data_config": _json_safe(asdict(config)),
        "solver_config": _json_safe(asdict(validated.solver_config)),
        "exact_npz_keys": list(EXACT_NPZ_KEYS),
        "case_count": len(cases),
        "cases": cases,
        "truth_exclusion": {
            "clean_written": False,
            "corruption_written": False,
            "corruption_support_written": False,
            "time_written": False,
            "downstream_solver_inputs_are_observed_only": True,
        },
        "static_audit_receipt_sha256": static_audit_receipt_sha256,
    }
    manifest["manifest_body_sha256"] = canonical_sha256(manifest)
    manifest_path = output_dir / MANIFEST_NAME
    with manifest_path.open("xb") as handle:
        handle.write(_strict_json_bytes(manifest))
    validate_blind_pack(output_dir, protocol=validated)
    return manifest


def _require_regular_output_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"pack entry must be a regular non-symlink file: {path}")
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"pack entry must be a regular file: {path}")


def validate_blind_pack(
    output_dir: Path,
    *,
    protocol: ValidatedBlindPackProtocol | None = None,
) -> dict[str, Any]:
    """Validate the self-digest, inventory, and observed-only array hashes."""

    output = Path(output_dir)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("blind-pack output must be a real directory")
    manifest_path = output / MANIFEST_NAME
    _require_regular_output_file(manifest_path)
    manifest = _load_strict_json(manifest_path)
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="blind-pack manifest")
    if _integer(manifest["schema_version"], name="schema_version", minimum=1) != 1:
        raise ValueError("unsupported blind-pack manifest schema_version")
    reported_body_digest = manifest.get("manifest_body_sha256")
    if (
        not isinstance(reported_body_digest, str)
        or _HEX_SHA256.fullmatch(reported_body_digest) is None
    ):
        raise ValueError("manifest_body_sha256 is invalid")
    manifest_body = {
        key: value for key, value in manifest.items() if key != "manifest_body_sha256"
    }
    if canonical_sha256(manifest_body) != reported_body_digest:
        raise ValueError("blind-pack manifest body digest mismatch")
    if manifest.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError("unexpected blind-pack manifest kind")
    if manifest.get("protocol_kind") != PROTOCOL_KIND:
        raise ValueError("blind pack is bound to an unexpected protocol kind")
    if manifest.get("exact_npz_keys") != list(EXACT_NPZ_KEYS):
        raise ValueError("blind-pack exact NPZ keys changed")
    static_receipt_digest = manifest.get("static_audit_receipt_sha256")
    if (
        not isinstance(static_receipt_digest, str)
        or _HEX_SHA256.fullmatch(static_receipt_digest) is None
    ):
        raise ValueError("static_audit_receipt_sha256 is invalid")
    manifest_data_config = _data_config(manifest.get("data_config"))
    _solver_config(manifest.get("solver_config"))
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("blind-pack cases must be a nonempty list")
    case_count = _integer(manifest.get("case_count"), name="case_count", minimum=1)
    if case_count != len(cases):
        raise ValueError("blind-pack case count mismatch")
    case_names: list[str] = []
    seen_seeds: set[int] = set()
    expected_shape = (
        manifest_data_config.grid_size * manifest_data_config.grid_size,
        manifest_data_config.n_timesteps,
    )
    for case in cases:
        record = _require_exact_keys(case, _CASE_RECORD_KEYS, label="case record")
        seed = _integer(record["seed"], name="case seed", minimum=0)
        if seed in seen_seeds:
            raise ValueError("blind-pack case seeds are duplicated")
        seen_seeds.add(seed)
        expected_name = _case_file_name(seed)
        if record["npz_name"] != expected_name:
            raise ValueError("blind-pack case file name changed")
        case_names.append(expected_name)
        case_path = output / expected_name
        _require_regular_output_file(case_path)
        if _file_sha256(case_path) != record["npz_sha256"]:
            raise ValueError(f"NPZ file digest mismatch: {expected_name}")
        if case_path.stat().st_size != record["npz_size_bytes"]:
            raise ValueError(f"NPZ file size mismatch: {expected_name}")
        with np.load(case_path, allow_pickle=False) as archive:
            if archive.files != list(EXACT_NPZ_KEYS):
                raise ValueError(f"NPZ key whitelist mismatch: {expected_name}")
            normalized = np.asarray(archive["observed_normalized"])
            initial_B = np.asarray(archive["initial_B"])
        if case_path.read_bytes() != _canonical_npz_bytes(normalized, initial_B):
            raise ValueError("blind-pack NPZ is not the canonical two-array container")
        if normalized.dtype != np.dtype(np.float64):
            raise ValueError("observed_normalized dtype must be float64")
        if initial_B.dtype != np.dtype(np.float64):
            raise ValueError("initial_B dtype must be float64")
        shape = record["shape"]
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in shape
            )
            or tuple(shape) != normalized.shape
            or tuple(shape) != expected_shape
        ):
            raise ValueError("blind-pack case shape mismatch")
        if initial_B.shape != (
            normalized.shape[1],
            manifest_data_config.n_components,
        ):
            raise ValueError("blind-pack initial_B shape mismatch")
        if not np.isfinite(normalized).all() or not np.isfinite(initial_B).all():
            raise ValueError("blind-pack arrays must be finite")
        if array_sha256(normalized) != record["observed_normalized_sha256"]:
            raise ValueError("observed_normalized array digest mismatch")
        if array_sha256(initial_B) != record["initial_B_sha256"]:
            raise ValueError("initial_B array digest mismatch")
        if (
            isinstance(record["sigma1"], bool)
            or not isinstance(record["sigma1"], (int, float))
            or not np.isfinite(record["sigma1"])
            or record["sigma1"] < 0.0
        ):
            raise ValueError("sigma1 must be finite")
        _validate_observed_only_semantics(
            normalized,
            initial_B,
            expected_shape=expected_shape,
            n_components=manifest_data_config.n_components,
            sigma1=float(record["sigma1"]),
        )
        for name in (
            "observed_sha256",
            "observed_normalized_sha256",
            "initial_B_sha256",
            "npz_sha256",
        ):
            if (
                not isinstance(record[name], str)
                or _HEX_SHA256.fullmatch(record[name]) is None
            ):
                raise ValueError(f"case {name} must be a lowercase SHA-256")

    expected_entries = set(case_names) | {MANIFEST_NAME}
    if {path.name for path in output.iterdir()} != expected_entries:
        raise ValueError("blind-pack output has missing or extra entries")
    truth_exclusion = manifest.get("truth_exclusion")
    expected_exclusion = {
        "clean_written": False,
        "corruption_written": False,
        "corruption_support_written": False,
        "time_written": False,
        "downstream_solver_inputs_are_observed_only": True,
    }
    if truth_exclusion != expected_exclusion:
        raise ValueError("blind-pack truth exclusion declaration changed")
    if protocol is not None:
        if output.resolve() != protocol.output_dir:
            raise ValueError("blind-pack output differs from the frozen protocol")
        if manifest["protocol_body_sha256"] != protocol.raw["protocol_body_sha256"]:
            raise ValueError("blind pack is bound to a different protocol digest")
        if manifest["source_sha256"] != protocol.raw["source_sha256"]:
            raise ValueError("blind pack is bound to a different source map")
        if [case["seed"] for case in cases] != list(protocol.data_config.seeds):
            raise ValueError("blind pack is bound to a different seed order")
        if manifest["data_config"] != _json_safe(asdict(protocol.data_config)):
            raise ValueError("blind pack data_config changed")
        if manifest["solver_config"] != _json_safe(asdict(protocol.solver_config)):
            raise ValueError("blind pack solver_config changed")
        _, static_audit_receipt_sha256 = _validate_static_audit_receipt(
            protocol.raw,
            repo_root=protocol.repo_root,
        )
        if manifest["static_audit_receipt_sha256"] != static_audit_receipt_sha256:
            raise ValueError("blind pack static audit receipt digest changed")
    elif [case["seed"] for case in cases] != list(manifest_data_config.seeds):
        raise ValueError("blind-pack cases differ from manifest data_config")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_protocol = subparsers.add_parser("validate-protocol")
    validate_protocol.add_argument("protocol", type=Path)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--acknowledge-pack-preparation", required=True)
    validate_pack = subparsers.add_parser("validate-pack")
    validate_pack.add_argument("--protocol", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    if arguments.command == "validate-protocol":
        validated = load_and_validate_protocol(arguments.protocol)
        print(
            json.dumps(
                {
                    "status": "VALID_FROZEN_NOT_RUN",
                    "protocol_body_sha256": validated.raw["protocol_body_sha256"],
                    "seed_count": len(validated.data_config.seeds),
                    "pack_preparations": 0,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    if arguments.command == "prepare":
        manifest = prepare_blind_pack(
            arguments.protocol,
            acknowledgement=arguments.acknowledge_pack_preparation,
        )
        validated = load_and_validate_protocol(arguments.protocol)
        print(
            json.dumps(
                {
                    "status": "OBSERVED_ONLY_BLIND_PACK_PREPARED",
                    "output_dir": str(validated.output_dir),
                    "manifest_body_sha256": manifest["manifest_body_sha256"],
                    "case_count": manifest["case_count"],
                    "smoke_launches": 0,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    validated = load_and_validate_protocol(arguments.protocol)
    manifest = validate_blind_pack(validated.output_dir, protocol=validated)
    print(
        json.dumps(
            {
                "status": "VALID_OBSERVED_ONLY_BLIND_PACK",
                "manifest_body_sha256": manifest["manifest_body_sha256"],
                "case_count": manifest["case_count"],
            },
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

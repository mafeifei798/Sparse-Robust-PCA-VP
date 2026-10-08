"""Build and verify the frozen v3 video-visualization execution receipt.

The server package has two nested integrity layers:

* ``run_receipt.json`` binds frozen source, command/environment documents,
  full server output manifests, and checked-in Git projections; and
* the self-excluding top-level ``manifest.json`` hashes every other file in the
  isolated root.

The checked-in receipt copy can be verified without the large server root.  It
binds the current source snapshot and Git assets to the same receipt while the
copied server-root manifest proves which hashes were present remotely.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import socket
import sys
from typing import Any, Mapping, Sequence

# Validation must be read-only even if the caller forgot the environment flag.
# Set this before importing any repository-owned module.
sys.dont_write_bytecode = True

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
for _import_root in (_SOURCE_ROOT, _SOURCE_ROOT / "src"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from experiments.video_spca_visualize import (
    validate_visualization_manifest_local_copy,
)
from rspca.datasets import canonical_sha256


RECEIPT_SCHEMA_VERSION = 1
COMMAND_SCHEMA_VERSION = 1
ROOT_MANIFEST_SCHEMA_VERSION = 1
ROOT_MANIFEST_NAME = "manifest.json"
RECEIPT_NAME = "run_receipt.json"
COMMAND_NAME = "run_commands.json"
ENVIRONMENT_NAME = "environment.json"
LOCAL_PACKAGE_MANIFEST_NAME = "local_manifest.json"
SERVER_ROOT_MANIFEST_COPY_NAME = "server_root_manifest.json"
ENVIRONMENT_SCHEMA_VERSION = 1

_REQUIRED_SOURCE_PATHS = {
    "experiments/video_spca.py",
    "experiments/video_spca_visualize.py",
    "tools/verify_video_visual_v3_receipt.py",
}
_REQUIRED_ENVIRONMENT_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "MKL_NUM_THREADS",
    "MPLBACKEND",
    "MPLCONFIGDIR",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONPATH",
}
_REQUIRED_PACKAGE_KEYS = {"Pillow", "matplotlib", "numpy", "scipy"}


def _expected_argv(seed: int) -> list[str]:
    return [
        "experiments/video_spca_visualize.py",
        "--grid-size",
        "32",
        "--n-timesteps",
        "180",
        "--total-time",
        "45",
        "--seed",
        str(seed),
        "--noise-std",
        "0.03",
        "--corruption-fraction",
        "0.01",
        "--corruption-scale",
        "1.5",
        "--alpha",
        "0.08",
        "--max-iter",
        "1000",
        "--tol",
        "1e-5",
        "--randomized-oversample",
        "15",
        "--randomized-power-iters",
        "3",
        "--randomized-full-refine-iters",
        "500",
        "--methods",
        "pca,vp,rnd,robust",
        "--frame-indices",
        "0,45,90,135,179",
        "--animation-formats",
        "gif",
        "--animation-stride",
        "3",
        "--fps",
        "6",
        "--dpi",
        "110",
        "--output-dir",
        f"../results/seed_{seed}",
    ]


class ReceiptValidationError(ValueError):
    """Raised when any receipt or file-closure assertion fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptValidationError(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "size_bytes": path.stat().st_size}


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ReceiptValidationError(f"non-finite JSON constant in {path}: {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReceiptValidationError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    _require(not path.is_symlink(), f"strict JSON path must not be a symlink: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ReceiptValidationError(
            f"cannot read strict JSON {path}: {error}"
        ) from error
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise ReceiptValidationError(
            f"refusing to overwrite receipt file: {path}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Preserve an exclusive partial file for manual review; never unlink a
        # path that a concurrent process could have exchanged.
        raise


def _validate_sha(value: Any, location: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{location} must be a lowercase SHA-256",
    )
    return value


def _safe_relative_path(value: Any, *, location: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{location} must be a path")
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and all(part not in {"", ".", ".."} for part in relative.parts),
        f"{location} must be a normalized relative POSIX path",
    )
    return value


def _validate_self_digest(
    payload: Mapping[str, Any],
    *,
    digest_field: str,
    location: str,
) -> str:
    body = dict(payload)
    claimed = _validate_sha(body.pop(digest_field, None), f"{location}.{digest_field}")
    _require(canonical_sha256(body) == claimed, f"{location} digest mismatch")
    return claimed


def _validate_file_record(
    record: Any,
    path: Path,
    *,
    location: str,
) -> None:
    _validate_file_record_shape(record, location=location)
    expected_sha = record["sha256"]
    expected_size = record["size_bytes"]
    _require(not path.is_symlink(), f"receipt-bound path is a symlink: {path}")
    _require(path.is_file(), f"missing receipt-bound file: {path}")
    _require(path.stat().st_size == expected_size, f"size mismatch for {path}")
    _require(_file_sha256(path) == expected_sha, f"SHA-256 mismatch for {path}")


def _validate_file_record_shape(record: Any, *, location: str) -> None:
    _require(isinstance(record, Mapping), f"{location} must be a mapping")
    _require(
        set(record) >= {"sha256", "size_bytes"},
        f"{location} must contain SHA-256 and size",
    )
    _validate_sha(record.get("sha256"), f"{location}.sha256")
    expected_size = record.get("size_bytes")
    _require(
        isinstance(expected_size, int)
        and not isinstance(expected_size, bool)
        and expected_size >= 0,
        f"{location}.size_bytes must be a non-negative integer",
    )


def _relative_file_records(root: Path, *, excluded: set[str]) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), f"invalid closure root: {root}")
    records: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative, location=f"closure path under {root}")
        _require(not path.is_symlink(), f"symlinks are forbidden in closure: {path}")
        _require(
            "__pycache__" not in path.parts and path.suffix != ".pyc",
            f"Python bytecode/cache is forbidden in closure: {path}",
        )
        if path.is_file() and relative not in excluded:
            records[relative] = _file_record(path)
        elif not path.is_dir() and relative not in excluded:
            raise ReceiptValidationError(f"special files are forbidden: {path}")
    return records


def write_self_excluding_manifest(
    root: str | Path,
    *,
    manifest_name: str,
    manifest_kind: str,
) -> Path:
    directory = Path(root)
    body: dict[str, Any] = {
        "schema_version": ROOT_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": manifest_kind,
        "excluded_paths": [manifest_name],
        "files": _relative_file_records(directory, excluded={manifest_name}),
    }
    body["manifest_sha256"] = canonical_sha256(body)
    path = directory / manifest_name
    _write_json(path, body)
    return path


def validate_self_excluding_manifest(
    root: str | Path,
    *,
    manifest_name: str,
    require_exact_files: bool,
) -> dict[str, Any]:
    directory = Path(root)
    manifest = _strict_json(directory / manifest_name)
    _require(
        manifest.get("schema_version") == ROOT_MANIFEST_SCHEMA_VERSION,
        "unsupported self-excluding manifest schema",
    )
    _require(
        manifest.get("excluded_paths") == [manifest_name],
        "self-excluding manifest has the wrong exclusion",
    )
    _validate_self_digest(
        manifest,
        digest_field="manifest_sha256",
        location=manifest_name,
    )
    records = manifest.get("files")
    _require(isinstance(records, Mapping), f"{manifest_name}.files must be a mapping")
    if require_exact_files:
        actual = _relative_file_records(directory, excluded={manifest_name})
        _require(set(actual) == set(records), f"{manifest_name} file closure mismatch")
        for relative, record in records.items():
            _validate_file_record(
                record,
                directory / relative,
                location=f"{manifest_name}.files.{relative}",
            )
    return manifest


def _validate_command_spec(commands: Mapping[str, Any]) -> str:
    expected_fields = {
        "schema_version",
        "execution_root",
        "working_directory",
        "python_executable",
        "environment",
        "source_sha256",
        "environment_file_sha256",
        "runs",
        "command_spec_sha256",
    }
    _require(set(commands) == expected_fields, "command specification schema mismatch")
    _require(
        commands.get("schema_version") == COMMAND_SCHEMA_VERSION,
        "unsupported command specification schema",
    )
    digest = _validate_self_digest(
        commands,
        digest_field="command_spec_sha256",
        location=COMMAND_NAME,
    )
    execution_root = commands.get("execution_root")
    _require(
        isinstance(execution_root, str) and Path(execution_root).is_absolute(),
        "command execution_root must be absolute",
    )
    _require(
        commands.get("working_directory") == "source",
        "commands must execute from the frozen source directory",
    )
    python_executable = commands.get("python_executable")
    _require(
        isinstance(python_executable, str) and Path(python_executable).is_absolute(),
        "command python_executable must be absolute",
    )
    environment = commands.get("environment")
    _require(
        isinstance(environment, Mapping)
        and set(environment) == _REQUIRED_ENVIRONMENT_KEYS
        and all(isinstance(value, str) for value in environment.values()),
        "command environment closure is invalid",
    )
    expected_pythonpath = f"{execution_root}/source/src:{execution_root}/source"
    _require(
        environment["PYTHONPATH"] == expected_pythonpath
        and environment["PYTHONHASHSEED"] == "0"
        and environment["PYTHONDONTWRITEBYTECODE"] == "1"
        and environment["CUDA_VISIBLE_DEVICES"] == ""
        and environment["MPLBACKEND"] == "Agg"
        and Path(environment["MPLCONFIGDIR"]).is_absolute(),
        "command environment has the wrong deterministic execution settings",
    )
    _require(
        all(
            environment[key] == "8"
            for key in {
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            }
        ),
        "command thread-count settings must all equal eight",
    )
    sources = commands.get("source_sha256")
    _require(
        isinstance(sources, Mapping) and set(sources) == _REQUIRED_SOURCE_PATHS,
        "command source hash closure is invalid",
    )
    for relative, value in sources.items():
        _safe_relative_path(relative, location="commands.source_sha256 key")
        _validate_sha(value, f"commands.source_sha256.{relative}")
    _validate_sha(
        commands.get("environment_file_sha256"),
        "commands.environment_file_sha256",
    )

    runs = commands.get("runs")
    _require(
        isinstance(runs, list) and len(runs) == 2, "command spec must have two runs"
    )
    _require(
        all(isinstance(run, Mapping) for run in runs),
        "each command run must be a mapping",
    )
    _require(
        [run.get("seed") for run in runs] == [91001, 91007],
        "command spec must contain ordered seeds 91001 and 91007",
    )
    for index, run in enumerate(runs):
        _require(
            set(run) == {"seed", "output_dir", "argv", "command_sha256"},
            f"commands.runs[{index}] schema mismatch",
        )
        seed = run["seed"]
        expected_output = f"results/seed_{seed}"
        _require(
            run.get("output_dir") == expected_output,
            f"commands.runs[{index}] output directory mismatch",
        )
        _safe_relative_path(run["output_dir"], location=f"commands.runs[{index}]")
        argv = run.get("argv")
        _require(
            isinstance(argv, list)
            and argv
            and all(isinstance(value, str) and value for value in argv),
            f"commands.runs[{index}].argv must be a non-empty string list",
        )
        _require(
            argv == _expected_argv(seed),
            f"commands.runs[{index}] does not match the frozen formal argv",
        )
        _require(
            run.get("command_sha256")
            == canonical_sha256(
                {key: value for key, value in run.items() if key != "command_sha256"}
            ),
            f"commands.runs[{index}] digest mismatch",
        )
    return digest


def _validate_environment_spec(environment: Mapping[str, Any]) -> str:
    expected_fields = {
        "schema_version",
        "captured_at_utc",
        "hostname",
        "platform",
        "machine",
        "python_executable",
        "python_version",
        "python_implementation",
        "libc",
        "packages",
        "environment_variables",
        "environment_sha256",
    }
    _require(set(environment) == expected_fields, "environment schema mismatch")
    _require(
        environment.get("schema_version") == ENVIRONMENT_SCHEMA_VERSION,
        "unsupported environment schema",
    )
    digest = _validate_self_digest(
        environment,
        digest_field="environment_sha256",
        location=ENVIRONMENT_NAME,
    )
    for field in (
        "captured_at_utc",
        "hostname",
        "platform",
        "machine",
        "python_executable",
        "python_version",
        "python_implementation",
        "libc",
    ):
        _require(
            isinstance(environment.get(field), str) and bool(environment[field]),
            f"environment.{field} must be a non-empty string",
        )
    _require(
        Path(environment["python_executable"]).is_absolute(),
        "environment python executable must be absolute",
    )
    packages = environment.get("packages")
    _require(
        isinstance(packages, Mapping)
        and set(packages) == _REQUIRED_PACKAGE_KEYS
        and all(isinstance(value, str) and value for value in packages.values()),
        "environment package closure is invalid",
    )
    variables = environment.get("environment_variables")
    _require(
        isinstance(variables, Mapping)
        and set(variables) == _REQUIRED_ENVIRONMENT_KEYS
        and all(isinstance(value, str) for value in variables.values()),
        "environment variable closure is invalid",
    )
    return digest


def _manifest_projection_invariant(manifest: Mapping[str, Any]) -> str:
    body = deepcopy(dict(manifest))
    body.pop("manifest_sha256", None)
    body.pop("artifact_copy_scope", None)
    for record in body.get("artifacts", {}).values():
        record.pop("availability", None)
        record.pop("local_copy_included", None)
    return canonical_sha256(body)


def _load_visual_manifest(directory: Path) -> tuple[dict[str, Any], Path]:
    _require(
        directory.is_dir() and not directory.is_symlink(),
        f"visual output directory is invalid: {directory}",
    )
    # Reuse the generic closure walker so even a hash-matching symlinked payload
    # is rejected before the visualization-specific validator reads it.
    _relative_file_records(directory, excluded=set())
    path = directory / "video_visualization_manifest.json"
    manifest = _strict_json(path)
    validate_visualization_manifest_local_copy(manifest, directory)
    return manifest, path


def prepare_execution_documents(root: str | Path) -> dict[str, Path]:
    """Exclusively capture the frozen v3 command and runtime environment."""

    requested_root = Path(root)
    _require(
        requested_root.is_dir() and not requested_root.is_symlink(),
        f"invalid isolated execution root: {requested_root}",
    )
    directory = requested_root.resolve()
    environment_path = directory / ENVIRONMENT_NAME
    commands_path = directory / COMMAND_NAME
    for reserved_path in (environment_path, commands_path):
        _require(
            not os.path.lexists(reserved_path),
            f"execution document already exists: {reserved_path}",
        )

    source_root = directory / "source"
    source_files = _relative_file_records(source_root, excluded=set())
    _require(
        _REQUIRED_SOURCE_PATHS <= set(source_files),
        "frozen source snapshot is incomplete",
    )
    missing_environment = sorted(_REQUIRED_ENVIRONMENT_KEYS - set(os.environ))
    _require(
        not missing_environment,
        f"required execution variables are missing: {missing_environment}",
    )
    environment_variables = {
        key: os.environ[key] for key in sorted(_REQUIRED_ENVIRONMENT_KEYS)
    }
    libc_name, libc_version = platform.libc_ver()
    environment: dict[str, Any] = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine() or "unknown",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "libc": f"{libc_name or 'unknown'} {libc_version or 'unknown'}",
        "packages": {
            name: package_version(name) for name in sorted(_REQUIRED_PACKAGE_KEYS)
        },
        "environment_variables": environment_variables,
    }
    environment["environment_sha256"] = canonical_sha256(environment)
    _validate_environment_spec(environment)
    _write_json(environment_path, environment)

    runs: list[dict[str, Any]] = []
    for seed in (91001, 91007):
        run: dict[str, Any] = {
            "seed": seed,
            "output_dir": f"results/seed_{seed}",
            "argv": _expected_argv(seed),
        }
        run["command_sha256"] = canonical_sha256(run)
        runs.append(run)
    commands: dict[str, Any] = {
        "schema_version": COMMAND_SCHEMA_VERSION,
        "execution_root": directory.as_posix(),
        "working_directory": "source",
        "python_executable": environment["python_executable"],
        "environment": environment_variables,
        "source_sha256": {
            relative: source_files[relative]["sha256"]
            for relative in sorted(_REQUIRED_SOURCE_PATHS)
        },
        "environment_file_sha256": _file_sha256(environment_path),
        "runs": runs,
    }
    commands["command_spec_sha256"] = canonical_sha256(commands)
    _validate_command_spec(commands)
    _write_json(commands_path, commands)
    return {"environment": environment_path, "commands": commands_path}


def build_receipt(
    root: str | Path,
    *,
    repo_root: str | Path,
    checked_asset_root: str | Path,
) -> Path:
    directory = Path(root).resolve()
    repository = Path(repo_root).resolve()
    asset_root = Path(checked_asset_root).resolve()
    receipt_path = directory / RECEIPT_NAME
    root_manifest_path = directory / ROOT_MANIFEST_NAME
    for reserved_path in (receipt_path, root_manifest_path):
        _require(
            not os.path.lexists(reserved_path),
            f"receipt build target already exists: {reserved_path}",
        )
    commands_path = directory / COMMAND_NAME
    environment_path = directory / ENVIRONMENT_NAME
    commands = _strict_json(commands_path)
    command_internal_sha = _validate_command_spec(commands)
    environment = _strict_json(environment_path)
    environment_internal_sha = _validate_environment_spec(environment)
    _require(
        commands["environment"] == environment["environment_variables"],
        "command/environment variables disagree",
    )
    _require(
        commands["python_executable"] == environment["python_executable"],
        "command/environment Python executable disagrees",
    )

    source_root = directory / "source"
    source_files = _relative_file_records(source_root, excluded=set())
    _require(source_files, "frozen source snapshot is empty")
    _require(
        _REQUIRED_SOURCE_PATHS <= set(source_files),
        "frozen source snapshot is incomplete",
    )
    declared_source_sha = commands.get("source_sha256")
    for relative in _REQUIRED_SOURCE_PATHS:
        _require(
            declared_source_sha.get(relative) == source_files[relative]["sha256"],
            f"command/source SHA mismatch for {relative}",
        )
    _require(
        commands.get("environment_file_sha256") == _file_sha256(environment_path),
        "command/environment SHA mismatch",
    )

    run_records: list[dict[str, Any]] = []
    for run in commands["runs"]:
        seed = int(run["seed"])
        output_relative = _safe_relative_path(
            run["output_dir"], location=f"seed {seed} output directory"
        )
        output_directory = directory / output_relative
        server_manifest, server_manifest_path = _load_visual_manifest(output_directory)
        checked_directory = asset_root / f"seed_{seed}"
        projected_manifest, projected_manifest_path = _load_visual_manifest(
            checked_directory
        )
        invariant = _manifest_projection_invariant(server_manifest)
        _require(
            _manifest_projection_invariant(projected_manifest) == invariant,
            f"checked-in seed {seed} is not a projection of the server manifest",
        )
        _require(
            server_manifest["data_sha256"] == projected_manifest["data_sha256"],
            f"seed {seed} data SHA differs between server and Git",
        )
        run_records.append(
            {
                "seed": seed,
                "command_sha256": run["command_sha256"],
                "data_sha256": server_manifest["data_sha256"],
                "projection_invariant_sha256": invariant,
                "server_output_manifest": {
                    "path": server_manifest_path.relative_to(directory).as_posix(),
                    **_file_record(server_manifest_path),
                    "internal_sha256": server_manifest["manifest_sha256"],
                },
                "server_artifacts": server_manifest["artifacts"],
                "checked_in_projection": {
                    "path": projected_manifest_path.relative_to(repository).as_posix(),
                    **_file_record(projected_manifest_path),
                    "internal_sha256": projected_manifest["manifest_sha256"],
                },
            }
        )

    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "receipt_kind": "video_visualization_v3_frozen_execution",
        "execution_root": commands.get("execution_root"),
        "source_snapshot": {
            "path": "source",
            "files": source_files,
            "tree_sha256": canonical_sha256(source_files),
        },
        "commands": {
            "path": COMMAND_NAME,
            **_file_record(commands_path),
            "internal_sha256": command_internal_sha,
        },
        "environment": {
            "path": ENVIRONMENT_NAME,
            **_file_record(environment_path),
            "internal_sha256": environment_internal_sha,
        },
        "runs": run_records,
        "scientific_boundary": {
            "single_seed_visual_inference_qualified": False,
            "fixed_budget_nonconverged_runs_are_descriptive_only": True,
        },
    }
    body["receipt_sha256"] = canonical_sha256(body)
    _write_json(receipt_path, body)
    write_self_excluding_manifest(
        directory,
        manifest_name=ROOT_MANIFEST_NAME,
        manifest_kind="video_visual_v3_full_server_root",
    )
    validate_full_root(directory)
    return receipt_path


def _validate_receipt_core(
    receipt: Mapping[str, Any],
    *,
    command_path: Path,
    environment_path: Path,
) -> None:
    _require(
        set(receipt)
        == {
            "schema_version",
            "receipt_kind",
            "execution_root",
            "source_snapshot",
            "commands",
            "environment",
            "runs",
            "scientific_boundary",
            "receipt_sha256",
        },
        "receipt schema mismatch",
    )
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION,
        "unsupported v3 receipt schema",
    )
    _validate_self_digest(
        receipt,
        digest_field="receipt_sha256",
        location=RECEIPT_NAME,
    )
    _require(
        receipt.get("receipt_kind") == "video_visualization_v3_frozen_execution",
        "wrong receipt kind",
    )
    execution_root = receipt.get("execution_root")
    _require(
        isinstance(execution_root, str) and Path(execution_root).is_absolute(),
        "receipt execution root must be absolute",
    )
    command_record = receipt.get("commands")
    _require(isinstance(command_record, Mapping), "receipt command record is invalid")
    _require(
        set(command_record) == {"path", "sha256", "size_bytes", "internal_sha256"},
        "receipt command record schema mismatch",
    )
    _require(
        command_record.get("path") == COMMAND_NAME, "receipt command path mismatch"
    )
    _validate_file_record(
        receipt.get("commands"), command_path, location="receipt.commands"
    )
    commands = _strict_json(command_path)
    _require(
        _validate_command_spec(commands) == command_record.get("internal_sha256"),
        "receipt command internal SHA mismatch",
    )
    _require(
        commands["execution_root"] == execution_root,
        "receipt/command execution roots disagree",
    )
    environment_record = receipt.get("environment")
    _require(
        isinstance(environment_record, Mapping),
        "receipt environment record is invalid",
    )
    _require(
        set(environment_record) == {"path", "sha256", "size_bytes", "internal_sha256"},
        "receipt environment record schema mismatch",
    )
    _require(
        environment_record.get("path") == ENVIRONMENT_NAME,
        "receipt environment path mismatch",
    )
    _validate_file_record(
        receipt.get("environment"),
        environment_path,
        location="receipt.environment",
    )
    environment = _strict_json(environment_path)
    _require(
        _validate_environment_spec(environment)
        == environment_record.get("internal_sha256"),
        "receipt environment internal SHA mismatch",
    )
    _require(
        commands["environment"] == environment["environment_variables"]
        and commands["python_executable"] == environment["python_executable"],
        "command/environment execution contract mismatch",
    )
    source = receipt.get("source_snapshot")
    _require(isinstance(source, Mapping), "receipt source snapshot is missing")
    _require(
        set(source) == {"path", "files", "tree_sha256"},
        "receipt source snapshot schema mismatch",
    )
    _require(source.get("path") == "source", "receipt source path mismatch")
    files = source.get("files")
    _require(isinstance(files, Mapping) and files, "receipt source file map is empty")
    for relative, record in files.items():
        _safe_relative_path(relative, location="receipt source path")
        _validate_file_record_shape(
            record, location=f"receipt.source_snapshot.files.{relative}"
        )
    _require(
        source.get("tree_sha256") == canonical_sha256(files),
        "receipt source tree digest mismatch",
    )
    _require(
        _REQUIRED_SOURCE_PATHS <= set(files),
        "receipt source closure is incomplete",
    )
    _require(
        all(
            commands["source_sha256"][relative] == files[relative]["sha256"]
            for relative in _REQUIRED_SOURCE_PATHS
        ),
        "receipt source records disagree with command source hashes",
    )
    runs = receipt.get("runs")
    _require(
        isinstance(runs, list)
        and all(isinstance(run, Mapping) for run in runs)
        and [run.get("seed") for run in runs] == [91001, 91007],
        "receipt run closure must contain seeds 91001 and 91007",
    )
    for run in runs:
        _require(
            set(run)
            == {
                "seed",
                "command_sha256",
                "data_sha256",
                "projection_invariant_sha256",
                "server_output_manifest",
                "server_artifacts",
                "checked_in_projection",
            },
            f"receipt seed {run['seed']} schema mismatch",
        )
        _validate_sha(run.get("data_sha256"), f"receipt seed {run['seed']} data")
        _validate_sha(run.get("command_sha256"), f"receipt seed {run['seed']} command")
        _validate_sha(
            run.get("projection_invariant_sha256"),
            f"receipt seed {run['seed']} projection invariant",
        )
        for record_name in ("server_output_manifest", "checked_in_projection"):
            record = run.get(record_name)
            _require(
                isinstance(record, Mapping)
                and set(record) == {"path", "sha256", "size_bytes", "internal_sha256"},
                f"receipt seed {run['seed']} {record_name} schema mismatch",
            )
            _validate_file_record_shape(
                record,
                location=f"receipt seed {run['seed']} {record_name}",
            )
            _validate_sha(
                record.get("internal_sha256"),
                f"receipt seed {run['seed']} {record_name} internal SHA",
            )
        _require(
            isinstance(run.get("server_artifacts"), Mapping)
            and bool(run["server_artifacts"]),
            f"receipt seed {run['seed']} server artifacts are invalid",
        )
    _require(
        receipt.get("scientific_boundary")
        == {
            "single_seed_visual_inference_qualified": False,
            "fixed_budget_nonconverged_runs_are_descriptive_only": True,
        },
        "receipt scientific boundary mismatch",
    )
    _require(
        len(commands["runs"]) == len(runs)
        and all(
            receipt_run.get("command_sha256") == command_run["command_sha256"]
            for receipt_run, command_run in zip(runs, commands["runs"], strict=True)
        ),
        "receipt run commands disagree with command specification",
    )


def validate_full_root(root: str | Path) -> dict[str, Any]:
    directory = Path(root).resolve()
    root_manifest = validate_self_excluding_manifest(
        directory,
        manifest_name=ROOT_MANIFEST_NAME,
        require_exact_files=True,
    )
    receipt_path = directory / RECEIPT_NAME
    receipt = _strict_json(receipt_path)
    _validate_receipt_core(
        receipt,
        command_path=directory / COMMAND_NAME,
        environment_path=directory / ENVIRONMENT_NAME,
    )
    source_root = directory / str(receipt["source_snapshot"]["path"])
    for relative, record in receipt["source_snapshot"]["files"].items():
        _validate_file_record(
            record,
            source_root / relative,
            location=f"receipt.source_snapshot.files.{relative}",
        )
    for run in receipt["runs"]:
        server_record = run["server_output_manifest"]
        manifest_relative = _safe_relative_path(
            server_record.get("path"),
            location=f"receipt seed {run['seed']} server manifest path",
        )
        expected_manifest_relative = (
            f"results/seed_{run['seed']}/video_visualization_manifest.json"
        )
        _require(
            manifest_relative == expected_manifest_relative,
            f"seed {run['seed']} server manifest path mismatch",
        )
        manifest_path = directory / manifest_relative
        _validate_file_record(
            server_record,
            manifest_path,
            location=f"receipt.runs.{run['seed']}.server_output_manifest",
        )
        manifest, _ = _load_visual_manifest(manifest_path.parent)
        _require(
            manifest["manifest_sha256"] == server_record["internal_sha256"],
            f"seed {run['seed']} output manifest internal SHA mismatch",
        )
        _require(
            manifest["artifacts"] == run["server_artifacts"],
            f"seed {run['seed']} artifact records disagree with receipt",
        )
        _require(
            _manifest_projection_invariant(manifest)
            == run["projection_invariant_sha256"],
            f"seed {run['seed']} projection invariant mismatch",
        )
        for artifact_name, artifact in manifest["artifacts"].items():
            artifact_relative = _safe_relative_path(
                artifact.get("path"),
                location=f"seed {run['seed']} artifact {artifact_name}",
            )
            root_relative = f"results/seed_{run['seed']}/{artifact_relative}"
            _require(
                root_manifest["files"].get(root_relative)
                == {
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                },
                f"server root does not bind seed {run['seed']} artifact {artifact_name}",
            )
    root_records = root_manifest["files"]
    _require(
        root_records[RECEIPT_NAME] == _file_record(receipt_path),
        "server root manifest does not bind the run receipt",
    )
    _require(
        root_records[COMMAND_NAME] == _file_record(directory / COMMAND_NAME)
        and root_records[ENVIRONMENT_NAME]
        == _file_record(directory / ENVIRONMENT_NAME),
        "server root manifest does not bind command/environment documents",
    )
    return {
        "receipt_sha256": receipt["receipt_sha256"],
        "root_manifest_sha256": root_manifest["manifest_sha256"],
        "seeds": [run["seed"] for run in receipt["runs"]],
    }


def validate_checked_copy(
    package_root: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    package = Path(package_root).resolve()
    repository = Path(repo_root).resolve()
    validate_self_excluding_manifest(
        package,
        manifest_name=LOCAL_PACKAGE_MANIFEST_NAME,
        require_exact_files=True,
    )
    receipt = _strict_json(package / RECEIPT_NAME)
    _validate_receipt_core(
        receipt,
        command_path=package / COMMAND_NAME,
        environment_path=package / ENVIRONMENT_NAME,
    )
    frozen_source_root = package / "source"
    _require(
        frozen_source_root.is_dir() and not frozen_source_root.is_symlink(),
        "checked receipt must include its frozen execution source snapshot",
    )
    frozen_records = _relative_file_records(frozen_source_root, excluded=set())
    _require(
        set(frozen_records) == set(receipt["source_snapshot"]["files"]),
        "checked receipt frozen-source file closure mismatch",
    )
    for relative, record in receipt["source_snapshot"]["files"].items():
        _safe_relative_path(relative, location="checked source path")
        _validate_file_record(
            record,
            frozen_source_root / relative,
            location=f"checked_execution_source.{relative}",
        )

    server_root_manifest = _strict_json(package / SERVER_ROOT_MANIFEST_COPY_NAME)
    _require(
        server_root_manifest.get("schema_version") == ROOT_MANIFEST_SCHEMA_VERSION
        and server_root_manifest.get("manifest_kind")
        == "video_visual_v3_full_server_root",
        "copied server root manifest schema/kind mismatch",
    )
    _require(
        server_root_manifest.get("excluded_paths") == [ROOT_MANIFEST_NAME],
        "copied server root manifest has the wrong exclusion",
    )
    _validate_self_digest(
        server_root_manifest,
        digest_field="manifest_sha256",
        location=SERVER_ROOT_MANIFEST_COPY_NAME,
    )
    server_records = server_root_manifest.get("files")
    _require(isinstance(server_records, Mapping), "server root file map is missing")
    for relative, record in server_records.items():
        _safe_relative_path(relative, location="copied server root path")
        _validate_file_record_shape(
            record, location=f"copied_server_root.files.{relative}"
        )
    _require(
        server_records.get(RECEIPT_NAME) == _file_record(package / RECEIPT_NAME),
        "copied server manifest does not bind the copied receipt",
    )
    _require(
        server_records.get(COMMAND_NAME) == _file_record(package / COMMAND_NAME)
        and server_records.get(ENVIRONMENT_NAME)
        == _file_record(package / ENVIRONMENT_NAME),
        "copied server manifest does not bind command/environment documents",
    )
    for relative, record in receipt["source_snapshot"]["files"].items():
        _require(
            server_records.get(f"source/{relative}") == record,
            f"server root/source receipt mismatch for {relative}",
        )

    for run in receipt["runs"]:
        seed = run["seed"]
        server_manifest = run["server_output_manifest"]
        server_manifest_path = _safe_relative_path(
            server_manifest.get("path"),
            location=f"receipt seed {seed} server manifest path",
        )
        _require(
            server_records.get(server_manifest_path)
            == {
                "sha256": server_manifest["sha256"],
                "size_bytes": server_manifest["size_bytes"],
            },
            f"server root does not bind seed {seed} manifest",
        )
        projected_record = run["checked_in_projection"]
        projected_relative = _safe_relative_path(
            projected_record.get("path"),
            location=f"receipt seed {seed} checked projection path",
        )
        expected_projection = (
            f"assets/evaluation/video/seed_{seed}/video_visualization_manifest.json"
        )
        _require(
            projected_relative == expected_projection,
            f"seed {seed} checked projection path mismatch",
        )
        projected_path = repository / projected_relative
        _validate_file_record(
            projected_record,
            projected_path,
            location=f"receipt.runs.{seed}.checked_in_projection",
        )
        projected, _ = _load_visual_manifest(projected_path.parent)
        _require(
            projected["manifest_sha256"] == projected_record["internal_sha256"],
            f"seed {seed} checked projection internal SHA mismatch",
        )
        _require(
            _manifest_projection_invariant(projected)
            == run["projection_invariant_sha256"],
            f"seed {seed} checked projection differs from server receipt",
        )
        server_gif = run["server_artifacts"]["animation_gif"]
        checked_gif = projected["artifacts"]["animation_gif"]
        _require(
            checked_gif["sha256"] == server_gif["sha256"]
            and checked_gif["size_bytes"] == server_gif["size_bytes"],
            f"seed {seed} Git GIF is not byte-identical to the server GIF",
        )
        for artifact_name, artifact in run["server_artifacts"].items():
            artifact_path = _safe_relative_path(
                artifact.get("path"),
                location=f"receipt seed {seed} artifact {artifact_name}",
            )
            root_relative = f"results/seed_{seed}/{artifact_path}"
            _require(
                server_records.get(root_relative)
                == {
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                },
                f"copied server root does not bind seed {seed} artifact {artifact_name}",
            )
    return {
        "receipt_sha256": receipt["receipt_sha256"],
        "server_root_manifest_sha256": server_root_manifest["manifest_sha256"],
        "seeds": [run["seed"] for run in receipt["runs"]],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("root", type=Path)
    build = subparsers.add_parser("build")
    build.add_argument("root", type=Path)
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--checked-assets", type=Path, required=True)
    full = subparsers.add_parser("validate-full")
    full.add_argument("root", type=Path)
    checked = subparsers.add_parser("validate-checked")
    checked.add_argument("package", type=Path)
    checked.add_argument("--repo-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        paths = prepare_execution_documents(args.root)
        report = {
            name: {"path": str(path), **_file_record(path)}
            for name, path in paths.items()
        }
    elif args.command == "build":
        build_receipt(
            args.root,
            repo_root=args.repo_root,
            checked_asset_root=args.checked_assets,
        )
        report = validate_full_root(args.root)
    elif args.command == "validate-full":
        report = validate_full_root(args.root)
    else:
        report = validate_checked_copy(args.package, repo_root=args.repo_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()


__all__ = [
    "ReceiptValidationError",
    "build_receipt",
    "main",
    "prepare_execution_documents",
    "validate_checked_copy",
    "validate_full_root",
    "validate_self_excluding_manifest",
    "write_self_excluding_manifest",
]

#!/usr/bin/env python3
"""Validate a closed, non-federated research-result release package.

The package root must contain ``manifest.json``.  The manifest binds every
payload file by relative POSIX path, byte count, SHA-256 digest, and evidence
role.  Validation is read-only: this module never repairs a package or writes a
validation marker into it.

Role grammar
------------

The three package-level roles are ``source``, ``config``, and ``environment``.
Result roles are scoped capabilities, for example
``video.result_json`` or ``genomics.tidy_csv``.  An entry normally uses the
singular ``role`` field.  A JSON file that deliberately contains both raw
records and their summary instead uses ``roles``, for example
``["video.result_json", "video.summary_json"]``.  ``role`` and ``roles`` are
mutually exclusive, so the same path never needs to be duplicated.

This validator checks package integrity and evidence coverage.  It does not
reinterpret scientific results or certify the claims made in them.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 1
EVIDENCE_SCOPE = "non_federated_summer_research"

PACKAGE_ROLES = frozenset({"source", "config", "environment", "supporting"})
ROLE_KINDS_BY_SCOPE: Mapping[str, frozenset[str]] = {
    "spca.clean": frozenset({"result_json"}),
    "spca.corrupt": frozenset({"result_json"}),
    "spca": frozenset({"summary_json"}),
    "video": frozenset({"result_json", "summary_json"}),
    "genomics": frozenset({"result_json", "summary_json", "tidy_csv"}),
}
REQUIRED_CAPABILITIES = frozenset(
    {
        "source",
        "config",
        "environment",
        "spca.clean.result_json",
        "spca.corrupt.result_json",
        "spca.summary_json",
        "video.result_json",
        "video.summary_json",
        "genomics.result_json",
        "genomics.summary_json",
        "genomics.tidy_csv",
    }
)

_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE = re.compile(r"[A-Za-z]:")

# This digest is the trust root for the formal Genomics provenance.  The
# release manifest is intentionally self-authored and can therefore be
# coherently re-signed together with a modified result.  Pinning the frozen
# sidecar here gives the read-side validator an expectation that is external
# to that re-signable package graph.
GENOMICS_TRUST_ANCHOR_PATH = (
    "provenance/genomics_trusted_provenance.json"
)
GENOMICS_TRUST_ANCHOR_SHA256 = (
    "b561bef832c9376c80cb00cc59c2de722ade0aed2625fb1320630e4a6ceaf14b"
)
GENOMICS_RESULT_PATH = "genomics/results.json"
GENOMICS_TRUST_ANCHOR_FIELDS = frozenset(
    {
        "artifact_type",
        "evidence_scope",
        "provenance",
        "result_path",
        "result_sha256",
        "runtime_versions",
        "schema_version",
    }
)
GENOMICS_PROVENANCE_FIELDS = frozenset(
    {
        "expected_genotype_sha256",
        "expected_metadata_sha256",
        "genotype_path",
        "genotype_sha256",
        "input_orientation",
        "internal_orientation",
        "metadata_path",
        "metadata_sample_id_column",
        "metadata_sha256",
        "missing_value",
        "normalized_genotypes_sha256",
        "population_column",
        "population_labels_sha256",
        "runtime_versions_at_start",
        "sample_ids_sha256",
        "samples",
        "snp_ids_sha256",
        "snps",
    }
)
GENOMICS_RUNTIME_VERSION_FIELDS = frozenset(
    {"machine", "numpy", "platform", "python", "scipy", "vp_rspca"}
)


class ReleaseValidationError(ValueError):
    """Raised when a release package violates the manifest contract."""


@dataclass(frozen=True)
class ValidatedRelease:
    """A compact report returned after every validation gate succeeds."""

    package_root: Path
    manifest_path: Path
    file_count: int
    payload_bytes: int
    capabilities: tuple[str, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseValidationError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ReleaseValidationError(
        f"non-standard JSON numeric constant is forbidden: {value}"
    )


def _reject_nonfinite(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, location=f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        # ``parse_constant`` catches NaN and Infinity.  This separate walk also
        # catches a finite-looking literal such as 1e400 that Python parses as
        # positive infinity.
        raise ReleaseValidationError(f"non-finite JSON number at {location}")


def _load_strict_json(path: Path, *, require_object: bool) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ReleaseValidationError(f"cannot read JSON file {path}: {error}") from error
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReleaseValidationError(f"JSON file is not UTF-8: {path}") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ReleaseValidationError:
        raise
    except json.JSONDecodeError as error:
        raise ReleaseValidationError(f"invalid JSON file {path}: {error}") from error
    _reject_nonfinite(payload)
    if require_object and not isinstance(payload, dict):
        raise ReleaseValidationError(f"JSON file must contain one top-level object: {path}")
    return payload


def _require_exact_keys(
    mapping: Mapping[str, Any], expected: frozenset[str], *, location: str
) -> None:
    actual = frozenset(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unknown {extra}")
        raise ReleaseValidationError(f"{location} has invalid fields: {', '.join(details)}")


def _portable_relative_path(value: Any, *, location: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{location} must be a non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ReleaseValidationError(f"{location} contains a forbidden control character")
    if "\\" in value:
        raise ReleaseValidationError(
            f"{location} must use POSIX separators and cannot contain backslashes"
        )
    if value.startswith("/") or _WINDOWS_DRIVE.match(value):
        raise ReleaseValidationError(f"{location} must be relative, not absolute: {value!r}")
    path = PurePosixPath(value)
    if value != path.as_posix():
        raise ReleaseValidationError(f"{location} is not a canonical POSIX path: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ReleaseValidationError(f"{location} contains forbidden traversal: {value!r}")
    if path.as_posix() == MANIFEST_NAME:
        raise ReleaseValidationError(
            f"{location} cannot list {MANIFEST_NAME!r}; a manifest cannot hash itself"
        )
    return path


def _parse_role(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseValidationError(f"{location} must be a non-empty string")
    if value in PACKAGE_ROLES:
        return value

    matching_scopes = [
        scope for scope in ROLE_KINDS_BY_SCOPE if value.startswith(f"{scope}.")
    ]
    if not matching_scopes:
        raise ReleaseValidationError(f"{location} has an unknown role: {value!r}")
    scope = max(matching_scopes, key=len)
    kind = value[len(scope) + 1 :]
    allowed = ROLE_KINDS_BY_SCOPE[scope]
    if kind not in allowed:
        raise ReleaseValidationError(
            f"{location} has a capability invalid for {scope!r}: {kind!r}"
        )
    return f"{scope}.{kind}"


def _entry_capabilities(entry: Mapping[str, Any], *, location: str) -> frozenset[str]:
    has_role = "role" in entry
    has_roles = "roles" in entry
    if has_role == has_roles:
        raise ReleaseValidationError(
            f"{location} must contain exactly one of 'role' or 'roles'"
        )
    if has_role:
        return frozenset(
            {_parse_role(entry["role"], location=f"{location}.role")}
        )

    values = entry["roles"]
    if not isinstance(values, list) or not values:
        raise ReleaseValidationError(f"{location}.roles must be a non-empty array")
    capabilities = [
        _parse_role(value, location=f"{location}.roles[{index}]")
        for index, value in enumerate(values)
    ]
    if len(capabilities) != len(set(capabilities)):
        raise ReleaseValidationError(f"{location}.roles contains duplicate roles")
    allowed_multi_role_sets = {
        frozenset({"video.result_json", "video.summary_json"}),
        frozenset({"genomics.result_json", "genomics.summary_json"}),
    }
    capability_set = frozenset(capabilities)
    if len(capabilities) > 1 and capability_set not in allowed_multi_role_sets:
        raise ReleaseValidationError(
            f"{location}.roles may combine only raw and summary JSON "
            "capabilities for one study"
        )
    has_csv = any(capability.endswith(".tidy_csv") for capability in capabilities)
    has_json = any(
        capability in {"config", "environment"}
        or capability.endswith(".result_json")
        or capability.endswith(".summary_json")
        for capability in capabilities
    )
    if has_csv and has_json:
        raise ReleaseValidationError(
            f"{location}.roles cannot combine CSV and JSON capabilities"
        )
    return capability_set


def _integer_bytes(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseValidationError(f"{location} must be a non-negative integer")
    if value < 0:
        raise ReleaseValidationError(f"{location} must be a non-negative integer")
    return value


def _sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or _LOWER_SHA256.fullmatch(value) is None:
        raise ReleaseValidationError(
            f"{location} must be a lowercase hexadecimal SHA-256 digest"
        )
    return value


def _ensure_no_symlink_components(root: Path, relative: PurePosixPath) -> Path:
    current = root
    for part in relative.parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as error:
            raise ReleaseValidationError(f"declared file is missing: {relative}") from error
        except OSError as error:
            raise ReleaseValidationError(
                f"cannot inspect declared path {relative}: {error}"
            ) from error
        if stat.S_ISLNK(mode):
            raise ReleaseValidationError(
                f"symlinks are forbidden in release payload paths: {relative}"
            )
    if not stat.S_ISREG(mode):
        raise ReleaseValidationError(f"declared payload is not a regular file: {relative}")
    return current


def _digest_and_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as error:
        raise ReleaseValidationError(f"cannot read declared payload {path}: {error}") from error
    return digest.hexdigest(), byte_count


def load_trusted_genomics_provenance(
    anchor_path: str | Path,
    *,
    result_path: str | Path | None = None,
) -> Mapping[str, Any]:
    """Load the frozen formal Genomics provenance after its external hash gate.

    The returned mapping is safe to pass as ``expected_provenance`` to the
    standalone Genomics result validator.  This function accepts a relocated
    release, but never a different sidecar body.
    """

    path = Path(anchor_path)
    if path.is_symlink() or not path.is_file():
        raise ReleaseValidationError(
            f"missing regular Genomics trust anchor: {path}"
        )
    digest, _ = _digest_and_count(path)
    if digest != GENOMICS_TRUST_ANCHOR_SHA256:
        raise ReleaseValidationError(
            "Genomics trust anchor does not match the frozen external SHA-256: "
            f"expected={GENOMICS_TRUST_ANCHOR_SHA256}, actual={digest}"
        )
    anchor = _load_strict_json(path, require_object=True)
    _require_exact_keys(
        anchor,
        GENOMICS_TRUST_ANCHOR_FIELDS,
        location="Genomics trust anchor",
    )
    if anchor["schema_version"] != 1 or isinstance(
        anchor["schema_version"], bool
    ):
        raise ReleaseValidationError(
            "Genomics trust anchor schema_version must equal integer 1"
        )
    if anchor["artifact_type"] != "genomics_formal_external_trust_anchor":
        raise ReleaseValidationError("Genomics trust anchor artifact_type is invalid")
    if anchor["evidence_scope"] != EVIDENCE_SCOPE:
        raise ReleaseValidationError("Genomics trust anchor evidence_scope is invalid")
    if anchor["result_path"] != GENOMICS_RESULT_PATH:
        raise ReleaseValidationError(
            f"Genomics trust anchor result_path must equal {GENOMICS_RESULT_PATH!r}"
        )
    expected_result_digest = _sha256(
        anchor["result_sha256"],
        location="Genomics trust anchor result_sha256",
    )
    if result_path is not None:
        candidate = Path(result_path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ReleaseValidationError(
                f"missing regular Genomics result bound by trust anchor: {candidate}"
            )
        actual_result_digest, _ = _digest_and_count(candidate)
        if actual_result_digest != expected_result_digest:
            raise ReleaseValidationError(
                "Genomics result does not match the externally frozen result "
                "SHA-256: "
                f"expected={expected_result_digest}, actual={actual_result_digest}"
            )
    runtime = anchor["runtime_versions"]
    if not isinstance(runtime, Mapping):
        raise ReleaseValidationError(
            "Genomics trust anchor runtime_versions must be an object"
        )
    _require_exact_keys(
        runtime,
        GENOMICS_RUNTIME_VERSION_FIELDS,
        location="Genomics trust anchor runtime_versions",
    )
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ReleaseValidationError(
            "Genomics trust anchor runtime version values must be non-empty strings"
        )
    provenance = anchor["provenance"]
    if not isinstance(provenance, Mapping):
        raise ReleaseValidationError(
            "Genomics trust anchor provenance must be an object"
        )
    _require_exact_keys(
        provenance,
        GENOMICS_PROVENANCE_FIELDS,
        location="Genomics trust anchor provenance",
    )
    if provenance["runtime_versions_at_start"] != runtime:
        raise ReleaseValidationError(
            "Genomics trust anchor runtime copies do not exactly match"
        )
    for field in ("samples", "snps"):
        value = provenance[field]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ReleaseValidationError(
                f"Genomics trust anchor provenance.{field} must be a positive integer"
            )
    return dict(provenance)


def _validate_tidy_csv(path: Path) -> None:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = csv.reader(handle, strict=True)
            try:
                header = next(rows)
            except StopIteration as error:
                raise ReleaseValidationError(f"tidy CSV is empty: {path}") from error
            if not header or any(not column for column in header):
                raise ReleaseValidationError(f"tidy CSV has an empty header field: {path}")
            if len(header) != len(set(header)):
                raise ReleaseValidationError(f"tidy CSV has duplicate header fields: {path}")
            row_count = 0
            for row_number, row in enumerate(rows, start=2):
                if len(row) != len(header):
                    raise ReleaseValidationError(
                        f"tidy CSV row {row_number} has {len(row)} fields; "
                        f"expected {len(header)}: {path}"
                    )
                row_count += 1
            if row_count == 0:
                raise ReleaseValidationError(f"tidy CSV has no data rows: {path}")
    except UnicodeDecodeError as error:
        raise ReleaseValidationError(f"tidy CSV is not UTF-8: {path}") from error
    except csv.Error as error:
        raise ReleaseValidationError(f"invalid tidy CSV {path}: {error}") from error
    except OSError as error:
        raise ReleaseValidationError(f"cannot read tidy CSV {path}: {error}") from error


def _declared_regular_files(root: Path) -> set[str]:
    """Return all payload files while rejecting every symlink in the package."""

    files: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in [*dirnames, *filenames]:
            candidate = directory_path / name
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                raise ReleaseValidationError(f"symlink is forbidden in release: {relative}")
        for name in filenames:
            candidate = directory_path / name
            if not candidate.is_file():
                relative = candidate.relative_to(root).as_posix()
                raise ReleaseValidationError(
                    f"non-regular filesystem entry is forbidden in release: {relative}"
                )
            relative = candidate.relative_to(root).as_posix()
            if relative != MANIFEST_NAME:
                files.add(relative)
    return files


def validate_nonfed_release(
    package_root: str | Path,
    *,
    require_genomics_semantics: bool = True,
) -> ValidatedRelease:
    """Validate ``package_root`` and return a read-only integrity report.

    Every payload file must be declared, and every declared file must exist.
    Consequently, a successful report binds a closed package rather than a
    hand-picked subset of a directory.
    """

    root = Path(package_root)
    if root.is_symlink():
        raise ReleaseValidationError(f"package root cannot be a symlink: {root}")
    if not root.is_dir():
        raise ReleaseValidationError(f"package root is not a directory: {root}")

    manifest_path = root / MANIFEST_NAME
    if manifest_path.is_symlink():
        raise ReleaseValidationError(f"manifest cannot be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise ReleaseValidationError(f"missing release manifest: {manifest_path}")
    manifest = _load_strict_json(manifest_path, require_object=True)
    _require_exact_keys(
        manifest,
        frozenset({"schema_version", "evidence_scope", "files"}),
        location="manifest",
    )
    if isinstance(manifest["schema_version"], bool) or manifest["schema_version"] != SCHEMA_VERSION:
        raise ReleaseValidationError(
            f"manifest.schema_version must equal integer {SCHEMA_VERSION}"
        )
    if manifest["evidence_scope"] != EVIDENCE_SCOPE:
        raise ReleaseValidationError(
            f"manifest.evidence_scope must equal {EVIDENCE_SCOPE!r}"
        )
    file_entries = manifest["files"]
    if not isinstance(file_entries, list) or not file_entries:
        raise ReleaseValidationError("manifest.files must be a non-empty array")

    seen_paths: set[str] = set()
    seen_hashes: set[str] = set()
    all_capabilities: set[str] = set()
    capabilities_by_path: dict[str, frozenset[str]] = {}
    digests_by_path: dict[str, str] = {}
    payload_bytes = 0
    for index, entry in enumerate(file_entries):
        location = f"manifest.files[{index}]"
        if not isinstance(entry, dict):
            raise ReleaseValidationError(f"{location} must be an object")
        expected_keys = (
            frozenset({"path", "sha256", "bytes", "role"})
            if "role" in entry and "roles" not in entry
            else frozenset({"path", "sha256", "bytes", "roles"})
        )
        _require_exact_keys(entry, expected_keys, location=location)
        relative = _portable_relative_path(entry["path"], location=f"{location}.path")
        relative_string = relative.as_posix()
        digest = _sha256(entry["sha256"], location=f"{location}.sha256")
        expected_bytes = _integer_bytes(entry["bytes"], location=f"{location}.bytes")
        capabilities = _entry_capabilities(entry, location=location)

        if relative_string in seen_paths:
            raise ReleaseValidationError(f"duplicate manifest path: {relative_string}")
        if digest in seen_hashes:
            raise ReleaseValidationError(f"duplicate manifest SHA-256 digest: {digest}")
        seen_paths.add(relative_string)
        seen_hashes.add(digest)
        all_capabilities.update(capabilities)
        capabilities_by_path[relative_string] = capabilities
        digests_by_path[relative_string] = digest

        payload_path = _ensure_no_symlink_components(root, relative)
        actual_digest, actual_bytes = _digest_and_count(payload_path)
        if actual_bytes != expected_bytes:
            raise ReleaseValidationError(
                f"byte-count mismatch for {relative_string}: "
                f"manifest={expected_bytes}, actual={actual_bytes}"
            )
        if actual_digest != digest:
            raise ReleaseValidationError(
                f"SHA-256 mismatch for {relative_string}: "
                f"manifest={digest}, actual={actual_digest}"
            )
        payload_bytes += actual_bytes

        suffix = payload_path.suffix.lower()
        json_capability = any(
            capability in {"config", "environment"}
            or capability.endswith(".result_json")
            or capability.endswith(".summary_json")
            for capability in capabilities
        )
        if json_capability and suffix != ".json":
            raise ReleaseValidationError(
                f"JSON role requires a .json path: {relative_string}"
            )
        # Parse every declared JSON file, not merely the minimum result and
        # summary roles.  A supporting role cannot be used to smuggle a
        # non-standard JSON artifact into a validated release.
        if suffix == ".json":
            _load_strict_json(payload_path, require_object=False)
        if any(capability.endswith(".tidy_csv") for capability in capabilities):
            if suffix != ".csv":
                raise ReleaseValidationError(
                    f"tidy_csv role requires a .csv path: {relative_string}"
                )
            _validate_tidy_csv(payload_path)

    missing_capabilities = sorted(REQUIRED_CAPABILITIES - all_capabilities)
    if missing_capabilities:
        raise ReleaseValidationError(
            f"manifest is missing required evidence roles: {missing_capabilities}"
        )

    actual_files = _declared_regular_files(root)
    untracked = sorted(actual_files - seen_paths)
    if untracked:
        raise ReleaseValidationError(f"release contains undeclared files: {untracked}")
    missing_from_tree = sorted(seen_paths - actual_files)
    if missing_from_tree:
        # Normally caught during entry validation; retained as a closed-package
        # consistency check in case the directory changes during validation.
        raise ReleaseValidationError(
            f"manifest declares files absent from final package scan: {missing_from_tree}"
        )

    anchor_capabilities = capabilities_by_path.get(GENOMICS_TRUST_ANCHOR_PATH)
    if anchor_capabilities != frozenset({"config"}):
        raise ReleaseValidationError(
            "release must declare the frozen Genomics trust anchor with exactly "
            "the config role"
        )
    declared_anchor_digest = digests_by_path[GENOMICS_TRUST_ANCHOR_PATH]
    if declared_anchor_digest != GENOMICS_TRUST_ANCHOR_SHA256:
        raise ReleaseValidationError(
            "release manifest does not bind the frozen external Genomics trust "
            "anchor SHA-256"
        )
    result_capabilities = capabilities_by_path.get(GENOMICS_RESULT_PATH)
    if result_capabilities is None or "genomics.result_json" not in result_capabilities:
        raise ReleaseValidationError(
            f"release must assign genomics.result_json to {GENOMICS_RESULT_PATH!r}"
        )
    trusted_provenance = load_trusted_genomics_provenance(
        root / GENOMICS_TRUST_ANCHOR_PATH,
        result_path=(root / GENOMICS_RESULT_PATH) if require_genomics_semantics else None,
    )
    result_payload = _load_strict_json(
        root / GENOMICS_RESULT_PATH,
        require_object=True,
    )
    if require_genomics_semantics:
        try:
            repository_root = str(Path(__file__).resolve().parents[1])
            if repository_root not in sys.path:
                sys.path.insert(0, repository_root)
            from experiments.genomics_1kg_benchmark import validate_result_payload

            validate_result_payload(
                result_payload,
                label=str(root / GENOMICS_RESULT_PATH),
                expected_provenance=trusted_provenance,
            )
        except (ImportError, TypeError, ValueError) as error:
            raise ReleaseValidationError(
                f"Genomics scientific payload validation failed: {error}"
            ) from error
    expected_runtime = trusted_provenance["runtime_versions_at_start"]
    if result_payload.get("runtime_versions") != expected_runtime:
        raise ReleaseValidationError(
            "Genomics result runtime_versions do not match the frozen external "
            "trust anchor"
        )
    if result_payload.get("provenance") != trusted_provenance:
        raise ReleaseValidationError(
            "Genomics result provenance does not match the frozen external trust "
            "anchor"
        )

    return ValidatedRelease(
        package_root=root.resolve(),
        manifest_path=manifest_path.resolve(),
        file_count=len(file_entries),
        payload_bytes=payload_bytes,
        capabilities=tuple(sorted(all_capabilities)),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a closed non-federated research release package."
    )
    parser.add_argument(
        "package_root",
        nargs="?",
        default="results/nonfed_2026-08-29",
        help="package directory containing manifest.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = validate_nonfed_release(args.package_root)
    except ReleaseValidationError as error:
        print(f"INVALID non-Fed release: {error}", file=sys.stderr)
        return 1
    print(
        "VALID non-Fed release: "
        f"{report.file_count} files, {report.payload_bytes} payload bytes, "
        f"manifest={report.manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

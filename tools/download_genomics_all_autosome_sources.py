#!/usr/bin/env python3
"""Download, seal, and verify the pinned 1000 Genomes autosome sources.

The repository manifest is an integrity description, not its own trust root.
Every command therefore requires two values supplied out of band: the expected
source-manifest SHA-256 and the expected downloader-source SHA-256.  ``validate``
also requires the externally recorded SHA-256 of the completion file.

Downloads are installed without overwriting an existing final file.  A complete
set of already-verified payloads can be sealed offline, including an explicit
replacement of a provisional completion file, without contacting the network.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import quote


SCHEMA_VERSION = "rspca-1kgp-all-autosome-source-manifest-v1"
COMPLETION_SCHEMA_VERSION = "rspca-1kgp-all-autosome-download-v2"
EXPECTED_DATASET_ID = "madhavajay/1kgp-bv-all"
EXPECTED_REVISION = "43ad04b827a977808c0ec1f80990ff79b60035a2"
EXPECTED_AUTOSOMES = tuple(str(value) for value in range(1, 23))
EXPECTED_FILE_COUNT = 45
EXPECTED_TOTAL_BYTES = 5_834_492_014
PINNED_SOURCE_MANIFEST_SHA256 = (
    "5e59af304e9d6b1e0a4099d139943fa1b9d3438a2588d9ee4f1f2f49206384c1"
)
SOURCE_MANIFEST_LOGICAL_PATH = (
    "operations/genomics_all_autosome_hf_source_manifest_2026-08-31.json"
)
DOWNLOADER_LOGICAL_PATH = "tools/download_genomics_all_autosome_sources.py"
COMPLETION_NAME = "all_autosome_download_manifest.json"
MAX_JSON_BYTES = 4 * 1024 * 1024
MAX_TOOL_BYTES = 32 * 1024 * 1024
MAX_EXECUTABLE_BYTES = 512 * 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_FORMAL_FRESH_CLI_PROCESS = __name__ == "__main__"


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    nlink: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    identity: FileIdentity
    sha256: str


@dataclass(frozen=True)
class InputAnchors:
    source_manifest: FileSnapshot
    downloader_source: FileSnapshot
    source_payload: dict[str, Any]
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CurlRuntime:
    path: Path
    snapshot: FileSnapshot
    name: str
    version: str


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute path without following a leaf symlink."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in _HEX for character in value)
    )


def _require_sha256(value: object, *, role: str) -> str:
    if not _is_sha256(value):
        raise ValueError(f"{role} must be a lowercase SHA-256")
    return str(value)


def _identity_from_stat(value: os.stat_result) -> FileIdentity:
    return FileIdentity(
        device=int(value.st_dev),
        inode=int(value.st_ino),
        mode=int(value.st_mode),
        nlink=int(value.st_nlink),
        size=int(value.st_size),
        mtime_ns=int(value.st_mtime_ns),
        ctime_ns=int(value.st_ctime_ns),
    )


def _open_regular_readonly(path: Path, *, role: str) -> tuple[int, FileIdentity]:
    if not os.path.lexists(path):
        raise FileNotFoundError(f"{role} does not exist: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(f"{role} must not be a symlink: {path}") from error
        raise
    try:
        identity = _identity_from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(identity.mode):
            raise ValueError(f"{role} must be a regular file: {path}")
        if identity.nlink != 1:
            raise ValueError(f"{role} must have exactly one hard link: {path}")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, identity


def _hash_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            return digest.hexdigest()
        digest.update(block)


def _snapshot_regular_file(
    path: Path,
    *,
    role: str,
    max_bytes: int | None = None,
) -> FileSnapshot:
    descriptor, before = _open_regular_readonly(path, role=role)
    try:
        if max_bytes is not None and before.size > max_bytes:
            raise ValueError(f"{role} exceeds the {max_bytes}-byte safety limit")
        digest = _hash_descriptor(descriptor)
        after = _identity_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if after != before:
        raise RuntimeError(f"{role} changed while it was being hashed: {path}")
    return FileSnapshot(path=path, identity=before, sha256=digest)


# Bind the imported module to the source bytes that were present during import.
# A long-lived importer cannot later point the in-memory implementation at a
# newly replaced on-disk file and have that new file misreported as executed
# code.  Fresh CLI subprocesses remain the required formal launch boundary.
_IMPORT_DOWNLOADER_SOURCE_SNAPSHOT = _snapshot_regular_file(
    _lexical_absolute(Path(__file__)),
    role="import-time downloader source",
    max_bytes=MAX_TOOL_BYTES,
)


def _read_regular_bytes(
    path: Path,
    *,
    role: str,
    max_bytes: int,
) -> tuple[bytes, FileSnapshot]:
    descriptor, before = _open_regular_readonly(path, role=role)
    try:
        if before.size > max_bytes:
            raise ValueError(f"{role} exceeds the {max_bytes}-byte safety limit")
        chunks: list[bytes] = []
        remaining = before.size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise RuntimeError(f"{role} became shorter while it was read")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1):
            raise RuntimeError(f"{role} became longer while it was read")
        after = _identity_from_stat(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if after != before:
        raise RuntimeError(f"{role} changed while it was read: {path}")
    raw = b"".join(chunks)
    return raw, FileSnapshot(
        path=path,
        identity=before,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _assert_snapshot_unchanged(snapshot: FileSnapshot, *, role: str) -> None:
    current = _snapshot_regular_file(snapshot.path, role=role, max_bytes=None)
    if current.identity != snapshot.identity or current.sha256 != snapshot.sha256:
        raise RuntimeError(f"{role} changed during the operation: {snapshot.path}")


def _assert_identity_unchanged(snapshot: FileSnapshot, *, role: str) -> None:
    current = _current_regular_identity(snapshot.path, role=role)
    if current != snapshot.identity:
        raise RuntimeError(f"{role} changed after verification: {snapshot.path}")


def _file_sha256(path: Path) -> str:
    return _snapshot_regular_file(path, role="file").sha256


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_float(token: str) -> float:
    raise ValueError(f"floating-point JSON numbers are not permitted: {token}")


def _reject_constant(token: str) -> object:
    raise ValueError(f"non-finite JSON token is not permitted: {token}")


def _parse_json_object(raw: bytes, *, role: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{role} is not valid UTF-8") from error
    payload = json.loads(
        text,
        object_pairs_hook=_reject_duplicate_pairs,
        parse_float=_reject_float,
        parse_constant=_reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return payload


def _load_json_snapshot(
    path: Path,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], FileSnapshot]:
    raw, snapshot = _read_regular_bytes(
        path,
        role=role,
        max_bytes=MAX_JSON_BYTES,
    )
    if expected_sha256 is not None and snapshot.sha256 != expected_sha256:
        raise ValueError(
            f"{role} SHA-256 does not match the external expectation: "
            f"{snapshot.sha256}"
        )
    return _parse_json_object(raw, role=role), snapshot


def validate_source_manifest(payload: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    if set(payload) != {"files", "schema_version", "scope", "source"}:
        raise ValueError("source manifest has an unexpected top-level schema")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected source-manifest schema version")

    source = payload["source"]
    expected_source_keys = {
        "authentication_boundary",
        "dataset_id",
        "provider",
        "repository_url",
        "revision",
        "upstream_declared_by_mirror",
    }
    if not isinstance(source, dict) or set(source) != expected_source_keys:
        raise ValueError("source has an unexpected schema")
    for key in expected_source_keys:
        if not isinstance(source[key], str) or not source[key]:
            raise ValueError(f"source.{key} must be a nonempty string")
    if source["dataset_id"] != EXPECTED_DATASET_ID:
        raise ValueError("source dataset_id is not the pinned mirror")
    if source["revision"] != EXPECTED_REVISION:
        raise ValueError("source revision is not the pinned mirror revision")

    scope = payload["scope"]
    expected_scope_keys = {
        "autosomes",
        "file_count",
        "mitochondrial_or_contig_files_included",
        "sex_chromosomes_included",
        "total_bytes",
    }
    if not isinstance(scope, dict) or set(scope) != expected_scope_keys:
        raise ValueError("scope has an unexpected schema")
    autosomes = scope["autosomes"]
    if not isinstance(autosomes, list) or tuple(autosomes) != EXPECTED_AUTOSOMES:
        raise ValueError("source manifest must cover chromosomes 1-22 exactly")
    if scope["sex_chromosomes_included"] is not False:
        raise ValueError("sex chromosomes must be excluded")
    if scope["mitochondrial_or_contig_files_included"] is not False:
        raise ValueError("mitochondrial and noncanonical contigs must be excluded")
    if (
        isinstance(scope["file_count"], bool)
        or not isinstance(scope["file_count"], int)
        or scope["file_count"] != EXPECTED_FILE_COUNT
    ):
        raise ValueError("source file_count is not the pinned integer")
    if (
        isinstance(scope["total_bytes"], bool)
        or not isinstance(scope["total_bytes"], int)
        or scope["total_bytes"] != EXPECTED_TOTAL_BYTES
    ):
        raise ValueError("source total_bytes does not match the integer pin")

    files = payload["files"]
    if not isinstance(files, list) or len(files) != EXPECTED_FILE_COUNT:
        raise ValueError(f"files must contain exactly {EXPECTED_FILE_COUNT} entries")
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    basenames: set[str] = set()
    seen_pairs: set[tuple[int, str]] = set()
    total_bytes = 0
    for index, raw in enumerate(files):
        if not isinstance(raw, dict):
            raise ValueError(f"files[{index}] must be an object")
        role = raw.get("role")
        expected_keys = (
            {"bytes", "git_blob_oid", "path", "role", "sha256"}
            if role == "sample_metadata"
            else {"bytes", "chromosome", "lfs_pointer_oid", "path", "role", "sha256"}
        )
        if set(raw) != expected_keys:
            raise ValueError(f"files[{index}] has an unexpected schema")
        path = raw.get("path")
        byte_count = raw.get("bytes")
        sha256 = raw.get("sha256")
        if not isinstance(path, str) or not path.startswith("data/"):
            raise ValueError(f"files[{index}].path is invalid")
        basename = Path(path).name
        if basename in basenames or path in paths:
            raise ValueError("source paths and basenames must be unique")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
            raise ValueError(f"files[{index}].bytes must be a positive integer")
        _require_sha256(sha256, role=f"files[{index}].sha256")
        paths.add(path)
        basenames.add(basename)
        total_bytes += byte_count

        if role == "sample_metadata":
            if path != "data/hg38_corrected.psam":
                raise ValueError("unexpected sample metadata path")
            blob = raw.get("git_blob_oid")
            if (
                blob != "4b0605075174eeb04e842ee2083a671c07379bfa"
                or not isinstance(blob, str)
                or len(blob) != 40
                or blob != blob.lower()
                or any(character not in _HEX for character in blob)
            ):
                raise ValueError("sample metadata Git blob OID changed")
        else:
            chromosome = raw.get("chromosome")
            if isinstance(chromosome, bool) or not isinstance(chromosome, int):
                raise ValueError("chromosome must be an integer")
            if not 1 <= chromosome <= 22:
                raise ValueError("chromosome must lie in 1..22")
            expected_role = (
                "genotype_matrix_zstd"
                if path.endswith(".pgen.zst")
                else "variant_metadata_zstd"
                if path.endswith("_rs.pvar.zst")
                else None
            )
            if role != expected_role:
                raise ValueError(f"file role/path mismatch for {path}")
            expected_path = (
                f"data/chr{chromosome}_hg38.pgen.zst"
                if role == "genotype_matrix_zstd"
                else f"data/chr{chromosome}_hg38_rs.pvar.zst"
            )
            if path != expected_path:
                raise ValueError(f"unexpected chromosome path: {path}")
            if (chromosome, role) in seen_pairs:
                raise ValueError("duplicate chromosome/role entry")
            seen_pairs.add((chromosome, role))
            pointer = raw.get("lfs_pointer_oid")
            if (
                not isinstance(pointer, str)
                or len(pointer) != 40
                or pointer != pointer.lower()
                or any(character not in _HEX for character in pointer)
            ):
                raise ValueError("lfs_pointer_oid must be a lowercase Git SHA-1")
        normalized.append(dict(raw))

    expected_pairs = {
        (int(chromosome), role)
        for chromosome in EXPECTED_AUTOSOMES
        for role in ("genotype_matrix_zstd", "variant_metadata_zstd")
    }
    if seen_pairs != expected_pairs:
        raise ValueError("source manifest does not contain both files for every autosome")
    if total_bytes != EXPECTED_TOTAL_BYTES:
        raise ValueError("file byte counts do not sum to pinned total_bytes")
    return tuple(normalized)


def _downloader_source_path() -> Path:
    return _lexical_absolute(Path(__file__))


def _capture_input_anchors(
    *,
    source_manifest: Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
) -> InputAnchors:
    expected_source = _require_sha256(
        expected_source_manifest_sha256,
        role="expected source-manifest SHA-256",
    )
    expected_tool = _require_sha256(
        expected_downloader_source_sha256,
        role="expected downloader-source SHA-256",
    )
    if expected_source != PINNED_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "external source-manifest SHA-256 disagrees with the repository safety pin"
        )
    source_payload, source_snapshot = _load_json_snapshot(
        source_manifest,
        role="source manifest",
        expected_sha256=expected_source,
    )
    records = validate_source_manifest(source_payload)
    import_snapshot = _IMPORT_DOWNLOADER_SOURCE_SNAPSHOT
    if _downloader_source_path() != import_snapshot.path:
        raise RuntimeError("downloader source path changed after module import")
    _assert_snapshot_unchanged(import_snapshot, role="import-time downloader source")
    tool_snapshot = import_snapshot
    if tool_snapshot.sha256 != expected_tool:
        raise ValueError(
            "downloader source SHA-256 does not match the external expectation: "
            f"{tool_snapshot.sha256}"
        )
    return InputAnchors(
        source_manifest=source_snapshot,
        downloader_source=tool_snapshot,
        source_payload=source_payload,
        records=records,
    )


def _assert_input_anchors_unchanged(anchors: InputAnchors) -> None:
    _assert_snapshot_unchanged(anchors.source_manifest, role="source manifest")
    _assert_snapshot_unchanged(anchors.downloader_source, role="downloader source")


def _require_fresh_cli_process() -> None:
    if not _FORMAL_FRESH_CLI_PROCESS:
        raise RuntimeError(
            "formal download, seal, and validation operations require a fresh CLI "
            "subprocess; imported in-process calls cannot attest mutable Python state"
        )


def _download_url(relative_path: str) -> str:
    encoded_path = "/".join(quote(part, safe="") for part in relative_path.split("/"))
    return (
        f"https://huggingface.co/datasets/{EXPECTED_DATASET_ID}/resolve/"
        f"{EXPECTED_REVISION}/{encoded_path}?download=true"
    )


def _verify_payload(path: Path, record: Mapping[str, Any]) -> FileSnapshot:
    snapshot = _snapshot_regular_file(
        path,
        role=f"downloaded payload {record['path']}",
    )
    if snapshot.identity.size != record["bytes"]:
        raise ValueError(
            f"byte-count mismatch for {path.name}: "
            f"{snapshot.identity.size} != {record['bytes']}"
        )
    if snapshot.sha256 != record["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {snapshot.sha256}")
    return snapshot


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_file(path: Path, *, role: str) -> None:
    descriptor, _ = _open_regular_readonly(path, role=role)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path, *, role: str) -> FileIdentity:
    if not os.path.lexists(path):
        raise FileNotFoundError(f"{role} does not exist: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(f"{role} must be a real non-symlink directory: {path}") from error
        raise
    try:
        identity = _identity_from_stat(os.fstat(descriptor))
        if not stat.S_ISDIR(identity.mode):
            raise ValueError(f"{role} must be a directory: {path}")
    finally:
        os.close(descriptor)
    return identity


def _assert_directory_unchanged(path: Path, expected: FileIdentity, *, role: str) -> None:
    current = _require_real_directory(path, role=role)
    if (current.device, current.inode) != (expected.device, expected.inode):
        raise RuntimeError(f"{role} was replaced during the operation: {path}")


def _capture_curl_runtime(curl_path: Path) -> CurlRuntime:
    snapshot = _snapshot_regular_file(
        curl_path,
        role="curl executable",
        max_bytes=MAX_EXECUTABLE_BYTES,
    )
    if snapshot.identity.mode & 0o111 == 0:
        raise PermissionError(f"curl is not executable: {curl_path}")
    if curl_path.name in {"", ".", ".."}:
        raise ValueError("curl executable name is invalid")
    completed = subprocess.run(
        [str(curl_path), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    version_lines = completed.stdout.splitlines()
    if not version_lines or not version_lines[0].strip():
        raise ValueError("curl --version did not report a version line")
    _assert_snapshot_unchanged(snapshot, role="curl executable")
    return CurlRuntime(
        path=curl_path,
        snapshot=snapshot,
        name=curl_path.name,
        version=version_lines[0].strip(),
    )


def _create_empty_partial(path: Path) -> FileIdentity:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        identity = _identity_from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(identity.mode) or identity.nlink != 1:
            raise ValueError(f"new partial is not an isolated regular file: {path}")
    finally:
        os.close(descriptor)
    return identity


def _current_regular_identity(path: Path, *, role: str) -> FileIdentity:
    descriptor, identity = _open_regular_readonly(path, role=role)
    os.close(descriptor)
    return identity


def _install_verified_partial(
    *,
    partial: Path,
    destination: Path,
    record: Mapping[str, Any],
    output_dir: Path,
) -> str:
    verified_partial = _verify_payload(partial, record)
    try:
        os.link(partial, destination, follow_symlinks=False)
    except FileExistsError:
        _verify_payload(destination, record)
        _assert_snapshot_unchanged(verified_partial, role="verified partial")
        partial.unlink()
        _fsync_directory(output_dir)
        return "raced_with_identical_verified_destination"

    linked_partial = os.lstat(partial)
    linked_destination = os.lstat(destination)
    if stat.S_ISLNK(linked_destination.st_mode) or not stat.S_ISREG(linked_destination.st_mode):
        raise RuntimeError("installed destination is not a regular file")
    if (
        linked_partial.st_dev,
        linked_partial.st_ino,
        linked_destination.st_dev,
        linked_destination.st_ino,
    ) != (
        verified_partial.identity.device,
        verified_partial.identity.inode,
        verified_partial.identity.device,
        verified_partial.identity.inode,
    ):
        raise RuntimeError("partial changed during no-overwrite installation")
    if linked_partial.st_nlink != 2 or linked_destination.st_nlink != 2:
        raise RuntimeError("unexpected hard-link count during installation")
    partial.unlink()
    _fsync_file(destination, role="installed payload")
    _fsync_directory(output_dir)
    _verify_payload(destination, record)
    return "downloaded_and_verified"


def _download_one(
    record: Mapping[str, Any],
    *,
    output_dir: Path,
    curl: CurlRuntime,
) -> dict[str, Any]:
    destination = output_dir / Path(record["path"]).name
    partial = output_dir / f".{destination.name}.partial"
    if os.path.lexists(destination):
        _verify_payload(destination, record)
        return {"path": destination.name, "status": "already_verified"}

    if os.path.lexists(partial):
        partial_identity = _current_regular_identity(partial, role="partial payload")
    else:
        partial_identity = _create_empty_partial(partial)
    if partial_identity.size > record["bytes"]:
        raise ValueError(
            f"partial payload is oversized for {destination.name}: "
            f"{partial_identity.size} > {record['bytes']}"
        )
    if partial_identity.size == record["bytes"]:
        status = _install_verified_partial(
            partial=partial,
            destination=destination,
            record=record,
            output_dir=output_dir,
        )
        return {"path": destination.name, "status": status}

    remaining = record["bytes"] - partial_identity.size
    argv = [
        str(curl.path),
        "-fL",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-filesize",
        str(remaining),
        "--continue-at",
        "-",
        "--output",
        str(partial),
        _download_url(str(record["path"])),
    ]
    _assert_snapshot_unchanged(curl.snapshot, role="curl executable")
    completed = subprocess.run(argv, check=False, capture_output=True, text=True)
    _assert_snapshot_unchanged(curl.snapshot, role="curl executable")
    current_partial = _current_regular_identity(partial, role="partial payload")
    if (current_partial.device, current_partial.inode) != (
        partial_identity.device,
        partial_identity.inode,
    ):
        raise RuntimeError(f"partial payload was replaced during curl: {partial}")
    if current_partial.size > record["bytes"]:
        raise ValueError(
            f"partial payload exceeded its pinned size for {destination.name}: "
            f"{current_partial.size} > {record['bytes']}"
        )
    if completed.returncode != 0:
        tail = completed.stderr[-2000:]
        raise RuntimeError(f"curl failed for {record['path']}: {tail}")
    status = _install_verified_partial(
        partial=partial,
        destination=destination,
        record=record,
        output_dir=output_dir,
    )
    return {"path": destination.name, "status": status}


def _payload_names(records: Sequence[Mapping[str, Any]]) -> set[str]:
    return {Path(record["path"]).name for record in records}


def _require_safe_download_preflight(
    *, output_dir: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    allowed = _payload_names(records) | {
        f".{Path(record['path']).name}.partial" for record in records
    }
    actual = {entry.name for entry in output_dir.iterdir()}
    unexpected = actual - allowed
    if unexpected:
        raise ValueError(
            "download output directory contains unrelated entries; refusing to write: "
            f"{sorted(unexpected)}"
        )
    for record in records:
        destination = output_dir / Path(record["path"]).name
        partial = output_dir / f".{destination.name}.partial"
        destination_exists = os.path.lexists(destination)
        partial_exists = os.path.lexists(partial)
        if destination_exists and partial_exists:
            raise ValueError(
                f"both final and partial payload exist for {destination.name}; "
                "refusing an ambiguous resume"
            )
        if destination_exists:
            _verify_payload(destination, record)
        elif partial_exists:
            identity = _current_regular_identity(partial, role="partial payload")
            if identity.size > record["bytes"]:
                raise ValueError(
                    f"partial payload is oversized for {destination.name}: "
                    f"{identity.size} > {record['bytes']}"
                )
            if identity.size == record["bytes"]:
                _verify_payload(partial, record)


def _require_directory_file_set(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    completion_allowed: bool,
) -> None:
    expected = _payload_names(records)
    if completion_allowed:
        expected.add(COMPLETION_NAME)
    actual = {entry.name for entry in output_dir.iterdir()}
    if actual != expected:
        raise ValueError(
            f"download directory file set mismatch; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )


def _build_completion_payload(
    *,
    anchors: InputAnchors,
    output_dir: Path,
    transfer: Mapping[str, Any],
) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    payload_snapshots: list[FileSnapshot] = []
    for record in anchors.records:
        path = output_dir / Path(record["path"]).name
        snapshot = _verify_payload(path, record)
        payload_snapshots.append(snapshot)
        files.append(
            {
                "bytes": snapshot.identity.size,
                "path": path.name,
                "sha256": snapshot.sha256,
                "source_path": record["path"],
            }
        )
    for snapshot in payload_snapshots:
        _assert_identity_unchanged(snapshot, role="payload used to build completion")
    payload: dict[str, Any] = {
        "schema_version": COMPLETION_SCHEMA_VERSION,
        "external_input_anchors": {
            "expected_downloader_source_sha256": anchors.downloader_source.sha256,
            "expected_source_manifest_sha256": anchors.source_manifest.sha256,
        },
        "source_manifest": {
            "logical_path": SOURCE_MANIFEST_LOGICAL_PATH,
            "payload_sha256": _canonical_sha256(anchors.source_payload),
            "sha256": anchors.source_manifest.sha256,
        },
        "pinned_source": {
            "dataset_id": EXPECTED_DATASET_ID,
            "revision": EXPECTED_REVISION,
        },
        "downloader": {
            "logical_path": DOWNLOADER_LOGICAL_PATH,
            "sha256": anchors.downloader_source.sha256,
        },
        "transfer": dict(transfer),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    payload["payload_sha256"] = _canonical_sha256(payload)
    return payload


def _validate_completion_schema(
    completion: Mapping[str, Any],
    *,
    anchors: InputAnchors,
) -> None:
    expected_top = {
        "schema_version",
        "external_input_anchors",
        "source_manifest",
        "pinned_source",
        "downloader",
        "transfer",
        "file_count",
        "total_bytes",
        "files",
        "payload_sha256",
    }
    if set(completion) != expected_top:
        raise ValueError("completion manifest has an unexpected top-level schema")
    if completion["schema_version"] != COMPLETION_SCHEMA_VERSION:
        raise ValueError("unexpected completion-manifest schema version")
    stored_hash = _require_sha256(
        completion["payload_sha256"], role="completion payload SHA-256"
    )
    unsigned = dict(completion)
    unsigned.pop("payload_sha256")
    if stored_hash != _canonical_sha256(unsigned):
        raise ValueError("download completion payload SHA-256 mismatch")

    external = completion["external_input_anchors"]
    if not isinstance(external, dict) or set(external) != {
        "expected_downloader_source_sha256",
        "expected_source_manifest_sha256",
    }:
        raise ValueError("completion external_input_anchors schema changed")
    if external != {
        "expected_downloader_source_sha256": anchors.downloader_source.sha256,
        "expected_source_manifest_sha256": anchors.source_manifest.sha256,
    }:
        raise ValueError("completion external input anchors do not match the command")

    source = completion["source_manifest"]
    if not isinstance(source, dict) or set(source) != {
        "logical_path",
        "payload_sha256",
        "sha256",
    }:
        raise ValueError("completion source_manifest schema changed")
    if source != {
        "logical_path": SOURCE_MANIFEST_LOGICAL_PATH,
        "payload_sha256": _canonical_sha256(anchors.source_payload),
        "sha256": anchors.source_manifest.sha256,
    }:
        raise ValueError("completion source-manifest identity changed")

    pinned = completion["pinned_source"]
    if pinned != {"dataset_id": EXPECTED_DATASET_ID, "revision": EXPECTED_REVISION}:
        raise ValueError("completion pinned-source identity changed")
    downloader = completion["downloader"]
    if downloader != {
        "logical_path": DOWNLOADER_LOGICAL_PATH,
        "sha256": anchors.downloader_source.sha256,
    }:
        raise ValueError("completion downloader identity changed")

    transfer = completion["transfer"]
    if not isinstance(transfer, dict) or set(transfer) != {"curl", "jobs", "mode"}:
        raise ValueError("completion transfer schema changed")
    mode = transfer["mode"]
    if mode == "offline_reseal":
        if transfer["curl"] is not None or transfer["jobs"] is not None:
            raise ValueError("offline reseal must not claim curl or worker metadata")
    elif mode == "download":
        jobs = transfer["jobs"]
        if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
            raise ValueError("completion download jobs must lie in 1..8")
        curl = transfer["curl"]
        if not isinstance(curl, dict) or set(curl) != {"name", "sha256", "version"}:
            raise ValueError("completion curl schema changed")
        if (
            not isinstance(curl["name"], str)
            or not curl["name"]
            or Path(curl["name"]).name != curl["name"]
        ):
            raise ValueError("completion curl name must be a basename")
        _require_sha256(curl["sha256"], role="completion curl SHA-256")
        if not isinstance(curl["version"], str) or not curl["version"].strip():
            raise ValueError("completion curl version must be a nonempty string")
    else:
        raise ValueError("completion transfer mode is invalid")

    if (
        isinstance(completion["file_count"], bool)
        or not isinstance(completion["file_count"], int)
        or completion["file_count"] != EXPECTED_FILE_COUNT
    ):
        raise ValueError("completion file_count changed")
    if (
        isinstance(completion["total_bytes"], bool)
        or not isinstance(completion["total_bytes"], int)
        or completion["total_bytes"] != EXPECTED_TOTAL_BYTES
    ):
        raise ValueError("completion total_bytes changed")
    files = completion["files"]
    if not isinstance(files, list) or len(files) != EXPECTED_FILE_COUNT:
        raise ValueError("completion files list length changed")
    for index, item in enumerate(files):
        if not isinstance(item, dict) or set(item) != {
            "bytes",
            "path",
            "sha256",
            "source_path",
        }:
            raise ValueError(f"completion files[{index}] schema changed")
        if isinstance(item["bytes"], bool) or not isinstance(item["bytes"], int):
            raise ValueError(f"completion files[{index}].bytes must be an integer")
        for key in ("path", "source_path"):
            if not isinstance(item[key], str) or not item[key]:
                raise ValueError(f"completion files[{index}].{key} must be a string")
        _require_sha256(item["sha256"], role=f"completion files[{index}].sha256")


def _encode_completion(completion: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(completion, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_completion_atomic(
    *,
    output_dir: Path,
    completion: Mapping[str, Any],
    replace_snapshot: FileSnapshot | None,
) -> FileSnapshot:
    completion_path = output_dir / COMPLETION_NAME
    temporary = output_dir / (
        f".{COMPLETION_NAME}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )
    encoded = _encode_completion(completion)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(temporary, flags, 0o600)
    installed = False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace_snapshot is None:
            if os.path.lexists(completion_path):
                raise FileExistsError(
                    f"completion already exists; use explicit offline reseal replacement: "
                    f"{completion_path}"
                )
            os.link(temporary, completion_path, follow_symlinks=False)
            temporary.unlink()
        else:
            _assert_snapshot_unchanged(
                replace_snapshot, role="existing completion manifest"
            )
            os.replace(temporary, completion_path)
        installed = True
        _fsync_directory(output_dir)
    finally:
        if not installed and os.path.lexists(temporary):
            temporary.unlink()
    return _snapshot_regular_file(
        completion_path,
        role="download completion manifest",
        max_bytes=MAX_JSON_BYTES,
    )


def _validate_download_directory_locked(
    *,
    anchors: InputAnchors,
    output_dir: Path,
    expected_completion_sha256: str,
) -> dict[str, Any]:
    expected_completion = _require_sha256(
        expected_completion_sha256,
        role="expected completion-file SHA-256",
    )
    output_identity = _require_real_directory(output_dir, role="output directory")
    completion_path = output_dir / COMPLETION_NAME
    completion, completion_snapshot = _load_json_snapshot(
        completion_path,
        role="download completion manifest",
        expected_sha256=expected_completion,
    )
    _validate_completion_schema(completion, anchors=anchors)

    expected_files: list[dict[str, Any]] = []
    payload_snapshots: list[FileSnapshot] = []
    for record in anchors.records:
        path = output_dir / Path(record["path"]).name
        snapshot = _verify_payload(path, record)
        payload_snapshots.append(snapshot)
        expected_files.append(
            {
                "bytes": snapshot.identity.size,
                "path": path.name,
                "sha256": snapshot.sha256,
                "source_path": record["path"],
            }
        )
    if completion["files"] != expected_files:
        raise ValueError("download completion file records do not match verified payloads")
    _require_directory_file_set(
        output_dir=output_dir,
        records=anchors.records,
        completion_allowed=True,
    )
    _assert_snapshot_unchanged(
        completion_snapshot, role="download completion manifest"
    )
    _assert_directory_unchanged(
        output_dir, output_identity, role="output directory"
    )
    _assert_input_anchors_unchanged(anchors)
    for snapshot in payload_snapshots:
        _assert_identity_unchanged(snapshot, role="verified payload")
    _require_directory_file_set(
        output_dir=output_dir,
        records=anchors.records,
        completion_allowed=True,
    )
    return {
        "status": "valid_and_externally_anchored",
        "completion_file_sha256": completion_snapshot.sha256,
        "completion_payload_sha256": completion["payload_sha256"],
        "downloader_source_sha256": anchors.downloader_source.sha256,
        "file_count": completion["file_count"],
        "source_manifest_sha256": anchors.source_manifest.sha256,
        "total_bytes": completion["total_bytes"],
        "transfer_mode": completion["transfer"]["mode"],
    }


def _lock_path(output_dir: Path) -> Path:
    return output_dir.parent / f".{output_dir.name}.download.lock"


@contextmanager
def _output_lock(output_dir: Path, *, exclusive: bool) -> Iterator[None]:
    if not output_dir.is_absolute() or output_dir.name in {"", ".", ".."}:
        raise ValueError("output directory must be an explicit non-root absolute path")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(output_dir.parent, role="output parent directory")
    lock_path = _lock_path(output_dir)
    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.EMLINK}:
            raise ValueError(f"output lock must not be a symlink: {lock_path}") from error
        raise
    try:
        locked_identity = _identity_from_stat(os.fstat(descriptor))
        if not stat.S_ISREG(locked_identity.mode) or locked_identity.nlink != 1:
            raise ValueError("output lock must be an isolated regular file")
        if stat.S_IMODE(locked_identity.mode) & 0o077:
            raise PermissionError("output lock must not be group/world accessible")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"another operation currently owns the output lock: {lock_path}"
            ) from error
        try:
            yield
        finally:
            current = os.lstat(lock_path)
            if stat.S_ISLNK(current.st_mode) or (
                current.st_dev,
                current.st_ino,
                current.st_nlink,
            ) != (
                locked_identity.device,
                locked_identity.inode,
                1,
            ):
                raise RuntimeError("output lock path was replaced during the operation")
    finally:
        os.close(descriptor)


def validate_download_directory(
    *,
    source_manifest: Path,
    output_dir: Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    expected_completion_sha256: str,
) -> dict[str, Any]:
    _require_fresh_cli_process()
    anchors = _capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
    )
    with _output_lock(output_dir, exclusive=False):
        result = _validate_download_directory_locked(
            anchors=anchors,
            output_dir=output_dir,
            expected_completion_sha256=expected_completion_sha256,
        )
    _assert_input_anchors_unchanged(anchors)
    return result


def _seal_locked(
    *,
    anchors: InputAnchors,
    output_dir: Path,
    replace_completion: bool,
    expected_existing_completion_sha256: str | None,
) -> dict[str, Any]:
    output_identity = _require_real_directory(output_dir, role="output directory")
    completion_path = output_dir / COMPLETION_NAME
    replace_snapshot: FileSnapshot | None = None
    if replace_completion:
        if expected_existing_completion_sha256 is None:
            raise ValueError(
                "--replace-completion requires --expected-existing-completion-sha256"
            )
        expected_existing = _require_sha256(
            expected_existing_completion_sha256,
            role="expected existing completion SHA-256",
        )
        replace_snapshot = _snapshot_regular_file(
            completion_path,
            role="existing completion manifest",
            max_bytes=MAX_JSON_BYTES,
        )
        if replace_snapshot.sha256 != expected_existing:
            raise ValueError("existing completion SHA-256 does not match the expectation")
    else:
        if expected_existing_completion_sha256 is not None:
            raise ValueError(
                "--expected-existing-completion-sha256 requires --replace-completion"
            )
        if os.path.lexists(completion_path):
            raise FileExistsError(
                "completion already exists; validate it or request an explicit offline replacement"
            )

    expected_before = _payload_names(anchors.records)
    if replace_completion:
        expected_before.add(COMPLETION_NAME)
    actual_before = {entry.name for entry in output_dir.iterdir()}
    if actual_before != expected_before:
        raise ValueError(
            "offline reseal requires the exact payload set and optional completion only; "
            f"missing={sorted(expected_before-actual_before)}, "
            f"extra={sorted(actual_before-expected_before)}"
        )
    _assert_input_anchors_unchanged(anchors)
    completion = _build_completion_payload(
        anchors=anchors,
        output_dir=output_dir,
        transfer={"curl": None, "jobs": None, "mode": "offline_reseal"},
    )
    _assert_input_anchors_unchanged(anchors)
    completion_snapshot = _write_completion_atomic(
        output_dir=output_dir,
        completion=completion,
        replace_snapshot=replace_snapshot,
    )
    _assert_input_anchors_unchanged(anchors)
    _assert_directory_unchanged(output_dir, output_identity, role="output directory")
    report = _validate_download_directory_locked(
        anchors=anchors,
        output_dir=output_dir,
        expected_completion_sha256=completion_snapshot.sha256,
    )
    return {
        **report,
        "status": "sealed_and_locally_verified__record_completion_sha256_externally",
        "replaced_existing_completion": replace_completion,
    }


def seal_download_directory(
    *,
    source_manifest: Path,
    output_dir: Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    replace_completion: bool = False,
    expected_existing_completion_sha256: str | None = None,
) -> dict[str, Any]:
    _require_fresh_cli_process()
    anchors = _capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
    )
    with _output_lock(output_dir, exclusive=True):
        result = _seal_locked(
            anchors=anchors,
            output_dir=output_dir,
            replace_completion=replace_completion,
            expected_existing_completion_sha256=expected_existing_completion_sha256,
        )
    _assert_input_anchors_unchanged(anchors)
    return result


def _download_sources_locked(
    *,
    anchors: InputAnchors,
    output_dir: Path,
    curl_path: Path,
    jobs: int,
) -> dict[str, Any]:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= 8:
        raise ValueError("jobs must be an integer in 1..8")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_identity = _require_real_directory(output_dir, role="output directory")
    completion_path = output_dir / COMPLETION_NAME
    if os.path.lexists(completion_path):
        raise FileExistsError(
            "completion already exists; use validate with its external SHA or seal "
            "with explicit replacement"
        )
    _require_safe_download_preflight(
        output_dir=output_dir,
        records=anchors.records,
    )

    missing = [
        record
        for record in anchors.records
        if not os.path.lexists(output_dir / Path(record["path"]).name)
    ]
    curl: CurlRuntime | None = None
    if missing:
        curl = _capture_curl_runtime(curl_path)
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    _download_one,
                    record,
                    output_dir=output_dir,
                    curl=curl,
                ): record
                for record in anchors.records
            }
            for future in as_completed(futures):
                record = futures[future]
                status = future.result()
                print(
                    json.dumps(
                        {"event": "source_ready", "source_path": record["path"], **status},
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        _assert_snapshot_unchanged(curl.snapshot, role="curl executable")
        transfer: dict[str, Any] = {
            "curl": {
                "name": curl.name,
                "sha256": curl.snapshot.sha256,
                "version": curl.version,
            },
            "jobs": jobs,
            "mode": "download",
        }
    else:
        transfer = {"curl": None, "jobs": None, "mode": "offline_reseal"}

    _require_directory_file_set(
        output_dir=output_dir,
        records=anchors.records,
        completion_allowed=False,
    )
    _assert_input_anchors_unchanged(anchors)
    completion = _build_completion_payload(
        anchors=anchors,
        output_dir=output_dir,
        transfer=transfer,
    )
    _assert_input_anchors_unchanged(anchors)
    completion_snapshot = _write_completion_atomic(
        output_dir=output_dir,
        completion=completion,
        replace_snapshot=None,
    )
    _assert_input_anchors_unchanged(anchors)
    if curl is not None:
        _assert_snapshot_unchanged(curl.snapshot, role="curl executable")
    _assert_directory_unchanged(output_dir, output_identity, role="output directory")
    report = _validate_download_directory_locked(
        anchors=anchors,
        output_dir=output_dir,
        expected_completion_sha256=completion_snapshot.sha256,
    )
    return {
        **report,
        "status": "downloaded_and_locally_verified__record_completion_sha256_externally",
    }


def download_sources(
    *,
    source_manifest: Path,
    output_dir: Path,
    curl_path: Path,
    jobs: int,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
) -> dict[str, Any]:
    _require_fresh_cli_process()
    anchors = _capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
    )
    with _output_lock(output_dir, exclusive=True):
        result = _download_sources_locked(
            anchors=anchors,
            output_dir=output_dir,
            curl_path=curl_path,
            jobs=jobs,
        )
    _assert_input_anchors_unchanged(anchors)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("download", "seal", "validate"):
        child = subparsers.add_parser(name)
        child.add_argument("--source-manifest", required=True, type=Path)
        child.add_argument("--output-dir", required=True, type=Path)
        child.add_argument("--expected-source-manifest-sha256", required=True)
        child.add_argument("--expected-downloader-source-sha256", required=True)
        if name == "download":
            child.add_argument("--jobs", type=int, default=4)
            child.add_argument(
                "--curl",
                type=Path,
                default=Path(shutil.which("curl") or "/usr/bin/curl"),
            )
        elif name == "seal":
            child.add_argument("--replace-completion", action="store_true")
            child.add_argument("--expected-existing-completion-sha256")
        else:
            child.add_argument("--expected-completion-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    source_manifest = _lexical_absolute(args.source_manifest)
    output_dir = _lexical_absolute(args.output_dir)
    try:
        if args.command == "download":
            result = download_sources(
                source_manifest=source_manifest,
                output_dir=output_dir,
                curl_path=_lexical_absolute(args.curl),
                jobs=args.jobs,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
                expected_downloader_source_sha256=args.expected_downloader_source_sha256,
            )
        elif args.command == "seal":
            result = seal_download_directory(
                source_manifest=source_manifest,
                output_dir=output_dir,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
                expected_downloader_source_sha256=args.expected_downloader_source_sha256,
                replace_completion=args.replace_completion,
                expected_existing_completion_sha256=(
                    args.expected_existing_completion_sha256
                ),
            )
        else:
            result = validate_download_directory(
                source_manifest=source_manifest,
                output_dir=output_dir,
                expected_source_manifest_sha256=args.expected_source_manifest_sha256,
                expected_downloader_source_sha256=args.expected_downloader_source_sha256,
                expected_completion_sha256=args.expected_completion_sha256,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        PermissionError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools import download_genomics_all_autosome_sources as downloader


ROOT = Path(__file__).resolve().parents[1]
PINNED_MANIFEST = (
    ROOT
    / "operations"
    / "genomics_all_autosome_hf_source_manifest_2026-08-31.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tool_sha() -> str:
    return downloader._file_sha256(Path(downloader.__file__).absolute())


def _tiny_source_payload(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict[str, bytes]]:
    payloads = {
        "data/chr1_hg38.pgen.zst": b"pgen",
        "data/chr1_hg38_rs.pvar.zst": b"pvar",
        "data/hg38_corrected.psam": b"psam",
    }
    monkeypatch.setattr(downloader, "EXPECTED_AUTOSOMES", ("1",))
    monkeypatch.setattr(downloader, "EXPECTED_FILE_COUNT", 3)
    monkeypatch.setattr(
        downloader, "EXPECTED_TOTAL_BYTES", sum(map(len, payloads.values()))
    )
    files = [
        {
            "bytes": len(payloads["data/chr1_hg38.pgen.zst"]),
            "chromosome": 1,
            "lfs_pointer_oid": "1" * 40,
            "path": "data/chr1_hg38.pgen.zst",
            "role": "genotype_matrix_zstd",
            "sha256": _sha(payloads["data/chr1_hg38.pgen.zst"]),
        },
        {
            "bytes": len(payloads["data/chr1_hg38_rs.pvar.zst"]),
            "chromosome": 1,
            "lfs_pointer_oid": "2" * 40,
            "path": "data/chr1_hg38_rs.pvar.zst",
            "role": "variant_metadata_zstd",
            "sha256": _sha(payloads["data/chr1_hg38_rs.pvar.zst"]),
        },
        {
            "bytes": len(payloads["data/hg38_corrected.psam"]),
            "git_blob_oid": "4b0605075174eeb04e842ee2083a671c07379bfa",
            "path": "data/hg38_corrected.psam",
            "role": "sample_metadata",
            "sha256": _sha(payloads["data/hg38_corrected.psam"]),
        },
    ]
    return (
        {
            "files": files,
            "schema_version": downloader.SCHEMA_VERSION,
            "scope": {
                "autosomes": ["1"],
                "file_count": 3,
                "mitochondrial_or_contig_files_included": False,
                "sex_chromosomes_included": False,
                "total_bytes": sum(map(len, payloads.values())),
            },
            "source": {
                "authentication_boundary": "test",
                "dataset_id": downloader.EXPECTED_DATASET_ID,
                "provider": "test",
                "repository_url": "https://example.invalid",
                "revision": downloader.EXPECTED_REVISION,
                "upstream_declared_by_mirror": "test",
            },
        },
        payloads,
    )


def _tiny_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    populate: bool = True,
) -> tuple[Path, Path, dict, dict[str, bytes], str, str]:
    source_payload, payloads = _tiny_source_payload(monkeypatch)
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(source_payload, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    source_sha = downloader._file_sha256(source_manifest)
    monkeypatch.setattr(downloader, "PINNED_SOURCE_MANIFEST_SHA256", source_sha)
    output_dir = tmp_path / "download"
    output_dir.mkdir()
    if populate:
        for source_path, payload in payloads.items():
            (output_dir / Path(source_path).name).write_bytes(payload)
    return source_manifest, output_dir, source_payload, payloads, source_sha, _tool_sha()


def _fake_curl(path: Path, body: str) -> Path:
    path.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = \"--version\" ]; then\n"
        "  echo 'curl fake 1.0'\n"
        "  exit 0\n"
        "fi\n"
        + body
        + "\n",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


def _seal(
    source_manifest: Path,
    output_dir: Path,
    source_sha: str,
    tool_sha: str,
    **kwargs,
) -> dict:
    anchors = downloader._capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_sha,
        expected_downloader_source_sha256=tool_sha,
    )
    with downloader._output_lock(output_dir, exclusive=True):
        return downloader._seal_locked(
            anchors=anchors,
            output_dir=output_dir,
            replace_completion=kwargs.pop("replace_completion", False),
            expected_existing_completion_sha256=kwargs.pop(
                "expected_existing_completion_sha256", None
            ),
        )


def _validate(
    source_manifest: Path,
    output_dir: Path,
    source_sha: str,
    tool_sha: str,
    completion_sha: str,
) -> dict:
    anchors = downloader._capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_sha,
        expected_downloader_source_sha256=tool_sha,
    )
    with downloader._output_lock(output_dir, exclusive=False):
        return downloader._validate_download_directory_locked(
            anchors=anchors,
            output_dir=output_dir,
            expected_completion_sha256=completion_sha,
        )


def test_repository_pinned_source_manifest_is_complete_and_raw_hash_matches() -> None:
    assert downloader._file_sha256(PINNED_MANIFEST) == (
        downloader.PINNED_SOURCE_MANIFEST_SHA256
    )
    payload, snapshot = downloader._load_json_snapshot(
        PINNED_MANIFEST,
        role="source manifest",
        expected_sha256=downloader.PINNED_SOURCE_MANIFEST_SHA256,
    )
    records = downloader.validate_source_manifest(payload)

    assert snapshot.sha256 == downloader.PINNED_SOURCE_MANIFEST_SHA256
    assert len(records) == 45
    assert sum(record["bytes"] for record in records) == 5_834_492_014
    assert {
        record["chromosome"]
        for record in records
        if record["role"] != "sample_metadata"
    } == set(range(1, 23))
    assert sum(record["role"] == "sample_metadata" for record in records) == 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["files"].pop(),
        lambda payload: payload["scope"].update({"total_bytes": 1}),
        lambda payload: payload["source"].update({"revision": "0" * 40}),
        lambda payload: payload["files"][0].update({"chromosome": 22}),
        lambda payload: payload["source"].update({"extra": "x"}),
        lambda payload: payload["scope"].update({"file_count": 45.0}),
    ],
)
def test_source_manifest_rejects_structural_tampering(mutation) -> None:
    payload = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
    mutation(payload)
    with pytest.raises(ValueError):
        downloader.validate_source_manifest(payload)


def test_external_source_anchor_defeats_coherent_internal_resign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_sha = downloader.PINNED_SOURCE_MANIFEST_SHA256
    payload = json.loads(PINNED_MANIFEST.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    resigned = tmp_path / "resigned.json"
    resigned.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    resigned_sha = downloader._file_sha256(resigned)
    monkeypatch.setattr(downloader, "PINNED_SOURCE_MANIFEST_SHA256", resigned_sha)

    with pytest.raises(ValueError, match="external.*disagrees"):
        downloader._capture_input_anchors(
            source_manifest=resigned,
            expected_source_manifest_sha256=original_sha,
            expected_downloader_source_sha256=_tool_sha(),
        )


@pytest.mark.parametrize(
    "raw, message",
    [
        (b'{"x":1,"x":2}', "duplicate JSON"),
        (b'{"x":NaN}', "non-finite"),
        (b'{"x":Infinity}', "non-finite"),
        (b'{"x":1e9999}', "floating-point"),
        (b'{"x":1.5}', "floating-point"),
    ],
)
def test_json_parser_rejects_duplicate_and_noninteger_numbers(
    raw: bytes, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        downloader._parse_json_object(raw, role="attack")


def test_download_one_installs_complete_partial_without_network(tmp_path: Path) -> None:
    payload = b"verified small payload\n"
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": len(payload),
        "sha256": _sha(payload),
    }
    partial = tmp_path / ".chr1_hg38.pgen.zst.partial"
    partial.write_bytes(payload)
    marker = tmp_path / "network-was-used"
    fake_curl = _fake_curl(
        tmp_path / "fake-curl", f'touch "{marker}"\nexit 90'
    )
    curl = downloader._capture_curl_runtime(fake_curl)

    status = downloader._download_one(record, output_dir=tmp_path, curl=curl)

    destination = tmp_path / "chr1_hg38.pgen.zst"
    assert status["status"] == "downloaded_and_verified"
    assert destination.read_bytes() == payload
    assert not partial.exists()
    assert not marker.exists()


@pytest.mark.parametrize("payload", [b"bad!", b"oversized"])
def test_download_one_rejects_bad_or_oversized_complete_partial_before_network(
    tmp_path: Path, payload: bytes
) -> None:
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": 4,
        "sha256": _sha(b"good"),
    }
    partial = tmp_path / ".chr1_hg38.pgen.zst.partial"
    partial.write_bytes(payload)
    marker = tmp_path / "network-was-used"
    curl = downloader._capture_curl_runtime(
        _fake_curl(tmp_path / "fake-curl", f'touch "{marker}"\nexit 0')
    )
    with pytest.raises(ValueError):
        downloader._download_one(record, output_dir=tmp_path, curl=curl)
    assert not marker.exists()
    assert not (tmp_path / "chr1_hg38.pgen.zst").exists()


def test_download_one_resumes_undersized_partial_and_enforces_final_hash(
    tmp_path: Path,
) -> None:
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": 4,
        "sha256": _sha(b"pgen"),
    }
    partial = tmp_path / ".chr1_hg38.pgen.zst.partial"
    partial.write_bytes(b"pg")
    curl = downloader._capture_curl_runtime(
        _fake_curl(
            tmp_path / "fake-curl",
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"--output\" ]; then shift; out=$1; fi\n"
            "  shift\n"
            "done\n"
            "printf 'en' >> \"$out\"\n",
        )
    )

    status = downloader._download_one(record, output_dir=tmp_path, curl=curl)

    assert status["status"] == "downloaded_and_verified"
    assert (tmp_path / "chr1_hg38.pgen.zst").read_bytes() == b"pgen"
    assert not partial.exists()


def test_download_one_rejects_curl_that_overshoots_size(tmp_path: Path) -> None:
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": 4,
        "sha256": _sha(b"pgen"),
    }
    partial = tmp_path / ".chr1_hg38.pgen.zst.partial"
    partial.write_bytes(b"pg")
    curl = downloader._capture_curl_runtime(
        _fake_curl(
            tmp_path / "fake-curl",
            "out=''\n"
            "while [ \"$#\" -gt 0 ]; do\n"
            "  if [ \"$1\" = \"--output\" ]; then shift; out=$1; fi\n"
            "  shift\n"
            "done\n"
            "printf 'enTOO-MUCH' >> \"$out\"\n",
        )
    )
    with pytest.raises(ValueError, match="exceeded"):
        downloader._download_one(record, output_dir=tmp_path, curl=curl)
    assert not (tmp_path / "chr1_hg38.pgen.zst").exists()


def test_existing_wrong_destination_is_never_overwritten(tmp_path: Path) -> None:
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": 4,
        "sha256": _sha(b"good"),
    }
    destination = tmp_path / "chr1_hg38.pgen.zst"
    destination.write_bytes(b"evil")
    (tmp_path / ".chr1_hg38.pgen.zst.partial").write_bytes(b"good")
    curl = downloader._capture_curl_runtime(
        _fake_curl(tmp_path / "fake-curl", "exit 0")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        downloader._download_one(record, output_dir=tmp_path, curl=curl)
    assert destination.read_bytes() == b"evil"


@pytest.mark.parametrize("kind", ["symlink", "dangling", "hardlink"])
def test_partial_rejects_symlink_dangling_and_hardlink(
    tmp_path: Path, kind: str
) -> None:
    record = {
        "path": "data/chr1_hg38.pgen.zst",
        "bytes": 4,
        "sha256": _sha(b"good"),
    }
    partial = tmp_path / ".chr1_hg38.pgen.zst.partial"
    if kind == "dangling":
        partial.symlink_to(tmp_path / "missing")
    else:
        target = tmp_path / "target"
        target.write_bytes(b"good")
        if kind == "symlink":
            partial.symlink_to(target)
        else:
            os.link(target, partial)
    curl = downloader._capture_curl_runtime(
        _fake_curl(tmp_path / "fake-curl", "exit 0")
    )
    with pytest.raises(ValueError):
        downloader._download_one(record, output_dir=tmp_path, curl=curl)


def test_output_lock_rejects_concurrent_owner_and_symlink(tmp_path: Path) -> None:
    output_dir = tmp_path / "sources"
    lock_path = tmp_path / ".sources.download.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another operation"):
            with downloader._output_lock(output_dir, exclusive=True):
                pass
    finally:
        os.close(descriptor)

    lock_path.unlink()
    target = tmp_path / "lock-target"
    target.write_text("x", encoding="utf-8")
    lock_path.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        with downloader._output_lock(output_dir, exclusive=True):
            pass


def test_offline_seal_and_validate_are_portable_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    sealed = _seal(source, output, source_sha, tool_sha)
    completion = output / downloader.COMPLETION_NAME
    raw = completion.read_text(encoding="utf-8")

    assert sealed["status"].startswith("sealed_and_locally_verified")
    assert sealed["transfer_mode"] == "offline_reseal"
    assert sealed["completion_file_sha256"] == downloader._file_sha256(completion)
    assert str(tmp_path) not in raw
    validated = _validate(
        source, output, source_sha, tool_sha, sealed["completion_file_sha256"]
    )
    assert validated["status"] == "valid_and_externally_anchored"

    portable_root = tmp_path / "portable-copy"
    portable_output = portable_root / "payloads"
    portable_root.mkdir()
    shutil.copytree(output, portable_output)
    portable_source = portable_root / "renamed-source.json"
    shutil.copy2(source, portable_source)
    moved = _validate(
        portable_source,
        portable_output,
        source_sha,
        tool_sha,
        sealed["completion_file_sha256"],
    )
    assert moved["status"] == "valid_and_externally_anchored"

    (portable_output / "undeclared.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        _validate(
            portable_source,
            portable_output,
            source_sha,
            tool_sha,
            sealed["completion_file_sha256"],
        )


def test_validate_never_executes_artifact_supplied_curl_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    sealed = _seal(source, output, source_sha, tool_sha)
    completion_path = output / downloader.COMPLETION_NAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    marker = tmp_path / "validator-executed-artifact"
    evil = _fake_curl(tmp_path / "evil", f'touch "{marker}"\nexit 0')
    completion["transfer"] = {
        "curl": {
            "name": "evil",
            "path": str(evil),
            "sha256": downloader._file_sha256(evil),
            "version": "evil",
        },
        "jobs": 1,
        "mode": "download",
    }
    unsigned = dict(completion)
    unsigned.pop("payload_sha256")
    completion["payload_sha256"] = downloader._canonical_sha256(unsigned)
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    attacked_sha = downloader._file_sha256(completion_path)

    with pytest.raises(ValueError, match="curl schema"):
        _validate(source, output, source_sha, tool_sha, attacked_sha)
    assert not marker.exists()
    assert attacked_sha != sealed["completion_file_sha256"]


@pytest.mark.parametrize("attack", ["duplicate", "overflow"])
def test_completion_raw_json_rejects_duplicate_and_nonfinite_before_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    sealed = _seal(source, output, source_sha, tool_sha)
    completion_path = output / downloader.COMPLETION_NAME
    raw = completion_path.read_text(encoding="utf-8")
    if attack == "duplicate":
        attacked = raw.replace(
            "{", '{"schema_version":"shadow",', 1
        )
        message = "duplicate JSON"
    else:
        attacked = raw.replace('"file_count": 3', '"file_count": 1e9999', 1)
        message = "floating-point"
    completion_path.write_text(attacked, encoding="utf-8")
    attacked_sha = downloader._file_sha256(completion_path)
    with pytest.raises(ValueError, match=message):
        _validate(source, output, source_sha, tool_sha, attacked_sha)
    assert attacked_sha != sealed["completion_file_sha256"]


def test_completion_external_file_hash_rejects_coherent_resign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    sealed = _seal(source, output, source_sha, tool_sha)
    completion_path = output / downloader.COMPLETION_NAME
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["transfer"]["mode"] = "download"
    completion["transfer"]["jobs"] = 1
    completion["transfer"]["curl"] = {
        "name": "curl",
        "sha256": "0" * 64,
        "version": "forged",
    }
    unsigned = dict(completion)
    unsigned.pop("payload_sha256")
    completion["payload_sha256"] = downloader._canonical_sha256(unsigned)
    completion_path.write_text(
        json.dumps(completion, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="external expectation"):
        _validate(
            source,
            output,
            source_sha,
            tool_sha,
            sealed["completion_file_sha256"],
        )


def test_explicit_offline_reseal_replaces_only_expected_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    legacy = output / downloader.COMPLETION_NAME
    legacy.write_text('{"legacy":true}\n', encoding="utf-8")
    legacy_sha = downloader._file_sha256(legacy)

    with pytest.raises(ValueError, match="does not match"):
        _seal(
            source,
            output,
            source_sha,
            tool_sha,
            replace_completion=True,
            expected_existing_completion_sha256="0" * 64,
        )
    assert downloader._file_sha256(legacy) == legacy_sha

    resealed = _seal(
        source,
        output,
        source_sha,
        tool_sha,
        replace_completion=True,
        expected_existing_completion_sha256=legacy_sha,
    )
    assert resealed["replaced_existing_completion"] is True
    assert resealed["completion_file_sha256"] != legacy_sha
    assert _validate(
        source,
        output,
        source_sha,
        tool_sha,
        resealed["completion_file_sha256"],
    )["status"] == "valid_and_externally_anchored"


def test_atomic_completion_failure_does_not_leave_final_or_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    anchors = downloader._capture_input_anchors(
        source_manifest=source,
        expected_source_manifest_sha256=source_sha,
        expected_downloader_source_sha256=tool_sha,
    )
    completion = downloader._build_completion_payload(
        anchors=anchors,
        output_dir=output,
        transfer={"curl": None, "jobs": None, "mode": "offline_reseal"},
    )
    real_link = downloader.os.link

    def fail_link(src, dst, **kwargs):
        if Path(dst).name == downloader.COMPLETION_NAME:
            raise OSError("injected atomic install failure")
        return real_link(src, dst, **kwargs)

    monkeypatch.setattr(downloader.os, "link", fail_link)
    with pytest.raises(OSError, match="injected"):
        downloader._write_completion_atomic(
            output_dir=output,
            completion=completion,
            replace_snapshot=None,
        )
    assert not (output / downloader.COMPLETION_NAME).exists()
    assert not list(output.glob(f".{downloader.COMPLETION_NAME}.*.tmp"))


def test_source_manifest_change_during_seal_aborts_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    real_verify = downloader._verify_payload
    changed = False

    def mutate_source(path, record):
        nonlocal changed
        result = real_verify(path, record)
        if not changed:
            source.write_bytes(source.read_bytes() + b"\n")
            changed = True
        return result

    monkeypatch.setattr(downloader, "_verify_payload", mutate_source)
    with pytest.raises(RuntimeError, match="source manifest changed"):
        _seal(source, output, source_sha, tool_sha)
    assert not (output / downloader.COMPLETION_NAME).exists()


def test_payload_change_during_completion_build_aborts_before_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    real_verify = downloader._verify_payload
    changed = False

    def mutate_verified_payload(path, record):
        nonlocal changed
        result = real_verify(path, record)
        if not changed:
            Path(path).write_bytes(b"evil")
            changed = True
        return result

    monkeypatch.setattr(downloader, "_verify_payload", mutate_verified_payload)
    with pytest.raises(RuntimeError, match="used to build completion"):
        _seal(source, output, source_sha, tool_sha)
    assert not (output / downloader.COMPLETION_NAME).exists()


def test_downloader_source_change_during_seal_aborts_before_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, _ = _tiny_environment(tmp_path, monkeypatch)
    tool_copy = tmp_path / "downloader-copy.py"
    shutil.copy2(Path(downloader.__file__), tool_copy)
    tool_sha = downloader._file_sha256(tool_copy)
    import_snapshot = downloader._snapshot_regular_file(
        tool_copy, role="test import-time downloader source"
    )
    monkeypatch.setattr(downloader, "_downloader_source_path", lambda: tool_copy)
    monkeypatch.setattr(
        downloader, "_IMPORT_DOWNLOADER_SOURCE_SNAPSHOT", import_snapshot
    )
    real_verify = downloader._verify_payload
    changed = False

    def mutate_tool(path, record):
        nonlocal changed
        result = real_verify(path, record)
        if not changed:
            tool_copy.write_bytes(tool_copy.read_bytes() + b"\n")
            changed = True
        return result

    monkeypatch.setattr(downloader, "_verify_payload", mutate_tool)
    with pytest.raises(RuntimeError, match="downloader source changed"):
        _seal(source, output, source_sha, tool_sha)
    assert not (output / downloader.COMPLETION_NAME).exists()


def test_imported_code_cannot_be_rebound_to_replaced_disk_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, _ = _tiny_environment(tmp_path, monkeypatch)
    tool_copy = tmp_path / "downloader-copy.py"
    tool_copy.write_bytes(b"old imported implementation\n")
    import_snapshot = downloader._snapshot_regular_file(
        tool_copy, role="test import-time downloader source"
    )
    monkeypatch.setattr(downloader, "_downloader_source_path", lambda: tool_copy)
    monkeypatch.setattr(
        downloader, "_IMPORT_DOWNLOADER_SOURCE_SNAPSHOT", import_snapshot
    )
    tool_copy.write_bytes(b"new disk implementation\n")

    with pytest.raises(RuntimeError, match="changed"):
        _seal(source, output, source_sha, import_snapshot.sha256)
    assert not (output / downloader.COMPLETION_NAME).exists()


def test_output_and_completion_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    real_output = tmp_path / "real-output"
    output.rename(real_output)
    output.symlink_to(real_output, target_is_directory=True)
    with pytest.raises(ValueError, match="non-symlink directory"):
        _seal(source, output, source_sha, tool_sha)

    output.unlink()
    real_output.rename(output)
    target = tmp_path / "completion-target"
    target.write_text("legacy", encoding="utf-8")
    (output / downloader.COMPLETION_NAME).symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _seal(
            source,
            output,
            source_sha,
            tool_sha,
            replace_completion=True,
            expected_existing_completion_sha256=_sha(b"legacy"),
        )


def test_source_manifest_and_curl_leaf_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    source_link = tmp_path / "source-link.json"
    source_link.symlink_to(source)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _seal(source_link, output, source_sha, tool_sha)

    real_curl = _fake_curl(tmp_path / "real-curl", "exit 0")
    curl_link = tmp_path / "curl-link"
    curl_link.symlink_to(real_curl)
    with pytest.raises(ValueError, match="must not be a symlink"):
        downloader._capture_curl_runtime(curl_link)


def test_payload_changed_after_hash_before_validation_return_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    sealed = _seal(source, output, source_sha, tool_sha)
    real_file_set = downloader._require_directory_file_set
    calls = 0

    def mutate_after_first_file_set(**kwargs):
        nonlocal calls
        real_file_set(**kwargs)
        calls += 1
        if calls == 1:
            (output / "chr1_hg38.pgen.zst").write_bytes(b"evil")

    monkeypatch.setattr(
        downloader, "_require_directory_file_set", mutate_after_first_file_set
    )
    with pytest.raises(RuntimeError, match="changed after verification"):
        _validate(
            source,
            output,
            source_sha,
            tool_sha,
            sealed["completion_file_sha256"],
        )


def test_download_command_seals_complete_payloads_without_curl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch
    )
    missing_curl = tmp_path / "curl-does-not-exist"
    anchors = downloader._capture_input_anchors(
        source_manifest=source,
        expected_source_manifest_sha256=source_sha,
        expected_downloader_source_sha256=tool_sha,
    )
    with downloader._output_lock(output, exclusive=True):
        result = downloader._download_sources_locked(
            anchors=anchors,
            output_dir=output,
            curl_path=missing_curl,
            jobs=4,
        )
    assert result["transfer_mode"] == "offline_reseal"
    assert result["completion_file_sha256"] == downloader._file_sha256(
        output / downloader.COMPLETION_NAME
    )


def test_download_preflight_rejects_unrelated_directory_before_curl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, tool_sha = _tiny_environment(
        tmp_path, monkeypatch, populate=False
    )
    (output / "unrelated.txt").write_text("do not touch", encoding="utf-8")
    marker = tmp_path / "curl-used"
    curl = _fake_curl(tmp_path / "fake-curl", f'touch "{marker}"\nexit 0')
    anchors = downloader._capture_input_anchors(
        source_manifest=source,
        expected_source_manifest_sha256=source_sha,
        expected_downloader_source_sha256=tool_sha,
    )
    with downloader._output_lock(output, exclusive=True):
        with pytest.raises(ValueError, match="unrelated entries"):
            downloader._download_sources_locked(
                anchors=anchors,
                output_dir=output,
                curl_path=curl,
                jobs=1,
            )
    assert not marker.exists()
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "do not touch"


def test_external_downloader_hash_is_required_and_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, output, _, _, source_sha, _ = _tiny_environment(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="downloader source SHA-256"):
        _seal(source, output, source_sha, "0" * 64)


def test_cli_requires_both_input_anchors() -> None:
    parser = downloader.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "download",
                "--source-manifest",
                "source.json",
                "--output-dir",
                "out",
            ]
        )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: downloader.download_sources(
            source_manifest=Path("source"),
            output_dir=Path("output"),
            curl_path=Path("curl"),
            jobs=1,
            expected_source_manifest_sha256="0" * 64,
            expected_downloader_source_sha256="0" * 64,
        ),
        lambda: downloader.seal_download_directory(
            source_manifest=Path("source"),
            output_dir=Path("output"),
            expected_source_manifest_sha256="0" * 64,
            expected_downloader_source_sha256="0" * 64,
        ),
        lambda: downloader.validate_download_directory(
            source_manifest=Path("source"),
            output_dir=Path("output"),
            expected_source_manifest_sha256="0" * 64,
            expected_downloader_source_sha256="0" * 64,
            expected_completion_sha256="0" * 64,
        ),
    ],
)
def test_imported_formal_operations_require_fresh_cli(operation) -> None:
    with pytest.raises(RuntimeError, match="fresh CLI subprocess"):
        operation()


def test_fresh_cli_reaches_validation_after_attesting_real_sources(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(Path(downloader.__file__).absolute()),
        "validate",
        "--source-manifest",
        str(PINNED_MANIFEST),
        "--output-dir",
        str(tmp_path / "missing-output"),
        "--expected-source-manifest-sha256",
        downloader.PINNED_SOURCE_MANIFEST_SHA256,
        "--expected-downloader-source-sha256",
        _tool_sha(),
        "--expected-completion-sha256",
        "0" * 64,
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "fresh CLI subprocess" not in completed.stderr
    assert "output directory does not exist" in completed.stderr

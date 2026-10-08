from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tools.validate_nonfed_release import (
    EVIDENCE_SCOPE,
    GENOMICS_TRUST_ANCHOR_PATH,
    REQUIRED_CAPABILITIES,
    ReleaseValidationError,
    main,
    validate_nonfed_release,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_GENOMICS_TRUST_ANCHOR = (
    REPOSITORY_ROOT
    / "results/nonfed_2026-08-29"
    / GENOMICS_TRUST_ANCHOR_PATH
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_payload(root: Path, relative: str, data: bytes) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": relative, "sha256": _sha256(data), "bytes": len(data)}


def _json_bytes(label: str) -> bytes:
    return (
        json.dumps({"artifact": label, "records": []}, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _valid_package(root: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []

    entry = _write_payload(root, "source/source_snapshot.txt", b"non-fed source\n")
    entries.append({**entry, "role": "source"})
    entry = _write_payload(root, "provenance/config.json", _json_bytes("config"))
    entries.append({**entry, "role": "config"})
    entry = _write_payload(
        root, "provenance/environment.json", _json_bytes("environment")
    )
    entries.append({**entry, "role": "environment"})

    for relative, role in (
        ("spca/clean/results.json", "spca.clean.result_json"),
        ("spca/corrupt/results.json", "spca.corrupt.result_json"),
        ("spca/summary.json", "spca.summary_json"),
    ):
        entry = _write_payload(root, relative, _json_bytes(role))
        entries.append({**entry, "role": role})

    entry = _write_payload(root, "video/results.json", _json_bytes("video"))
    entries.append(
        {
            **entry,
            "roles": ["video.result_json", "video.summary_json"],
        }
    )
    trust_anchor_bytes = FROZEN_GENOMICS_TRUST_ANCHOR.read_bytes()
    trust_anchor = json.loads(trust_anchor_bytes)
    genomics_bytes = (
        json.dumps(
            {
                "artifact": "genomics",
                "provenance": trust_anchor["provenance"],
                "runtime_versions": trust_anchor["runtime_versions"],
            },
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    entry = _write_payload(root, "genomics/results.json", genomics_bytes)
    entries.append(
        {
            **entry,
            "roles": ["genomics.result_json", "genomics.summary_json"],
        }
    )
    csv_bytes = b"seed,method,accuracy\n2025,pca,0.9\n"
    entry = _write_payload(root, "genomics/tidy.csv", csv_bytes)
    entries.append({**entry, "role": "genomics.tidy_csv"})
    entry = _write_payload(root, GENOMICS_TRUST_ANCHOR_PATH, trust_anchor_bytes)
    entries.append({**entry, "role": "config"})

    manifest = {
        "schema_version": 1,
        "evidence_scope": EVIDENCE_SCOPE,
        "files": entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_valid_closed_package_supports_one_file_with_multiple_json_roles(
    tmp_path: Path,
) -> None:
    _valid_package(tmp_path)

    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    report = validate_nonfed_release(tmp_path, require_genomics_semantics=False)
    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert report.file_count == 10
    assert set(report.capabilities) == set(REQUIRED_CAPABILITIES)
    assert before == after


@pytest.mark.parametrize("bad_path", ["/tmp/result.json", "../result.json", "a/../b"])
def test_rejects_absolute_or_traversing_manifest_paths(
    tmp_path: Path, bad_path: str
) -> None:
    manifest = _valid_package(tmp_path)
    manifest["files"][0]["path"] = bad_path
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match="relative|traversal|canonical"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


def test_rejects_duplicate_paths_before_they_can_supply_multiple_roles(
    tmp_path: Path,
) -> None:
    manifest = _valid_package(tmp_path)
    duplicate = dict(manifest["files"][0])
    duplicate["role"] = "supporting"
    manifest["files"].append(duplicate)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match="duplicate manifest path"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


def test_rejects_duplicate_hashes_even_for_different_paths(tmp_path: Path) -> None:
    manifest = _valid_package(tmp_path)
    original = manifest["files"][0]
    copied_path = tmp_path / "copy.txt"
    copied_path.write_bytes((tmp_path / original["path"]).read_bytes())
    manifest["files"].append(
        {
            "path": "copy.txt",
            "sha256": original["sha256"],
            "bytes": original["bytes"],
            "role": "supporting",
        }
    )
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match="duplicate manifest SHA-256"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


@pytest.mark.parametrize("mutation", ["missing", "bytes", "hash"])
def test_rejects_missing_or_content_mismatched_payloads(
    tmp_path: Path, mutation: str
) -> None:
    manifest = _valid_package(tmp_path)
    entry = manifest["files"][0]
    payload = tmp_path / entry["path"]
    if mutation == "missing":
        payload.unlink()
        match = "missing"
    elif mutation == "bytes":
        entry["bytes"] += 1
        match = "byte-count mismatch"
    else:
        entry["sha256"] = "0" * 64
        match = "SHA-256 mismatch"
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match=match):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


@pytest.mark.parametrize(
    "invalid_json,match",
    [
        ('{"schema_version":1,"schema_version":1,"evidence_scope":"x","files":[]}', "duplicate JSON"),
        ('{"schema_version":NaN,"evidence_scope":"x","files":[]}', "non-standard JSON"),
    ],
)
def test_manifest_uses_strict_json(
    tmp_path: Path, invalid_json: str, match: str
) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "manifest.json").write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match=match):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


@pytest.mark.parametrize(
    "invalid_json,match",
    [
        (b'{"records": NaN}\n', "non-standard JSON"),
        (b'{"records": [], "records": []}\n', "duplicate JSON"),
        (b'{"value": 1e400}\n', "non-finite JSON"),
    ],
)
def test_result_and_summary_artifacts_use_strict_json(
    tmp_path: Path, invalid_json: bytes, match: str
) -> None:
    manifest = _valid_package(tmp_path)
    entry = next(
        item for item in manifest["files"] if item["path"] == "video/results.json"
    )
    path = tmp_path / entry["path"]
    path.write_bytes(invalid_json)
    entry["bytes"] = len(invalid_json)
    entry["sha256"] = _sha256(invalid_json)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match=match):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


def test_rejects_missing_evidence_capability(tmp_path: Path) -> None:
    manifest = _valid_package(tmp_path)
    entry = next(item for item in manifest["files"] if "roles" in item)
    entry["roles"] = ["video.result_json"]
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match="video.summary_json"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


def test_rejects_undeclared_payload_and_malformed_tidy_csv(tmp_path: Path) -> None:
    _valid_package(tmp_path)
    (tmp_path / "untracked.log").write_text("not in manifest\n", encoding="utf-8")
    with pytest.raises(ReleaseValidationError, match="undeclared files"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)

    (tmp_path / "untracked.log").unlink()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["files"]
        if item.get("role") == "genomics.tidy_csv"
    )
    bad_csv = b"seed,seed\n2025,pca\n"
    (tmp_path / entry["path"]).write_bytes(bad_csv)
    entry["bytes"] = len(bad_csv)
    entry["sha256"] = _sha256(bad_csv)
    _rewrite_manifest(tmp_path, manifest)
    with pytest.raises(ReleaseValidationError, match="duplicate header"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)


def test_cli_returns_one_for_invalid_package(tmp_path: Path, capsys: Any) -> None:
    _valid_package(tmp_path)
    (tmp_path / "spca/clean/results.json").unlink()

    assert main([str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "INVALID non-Fed release" in captured.err


def test_rejects_coherently_resigned_anchor_result_and_outer_manifest(
    tmp_path: Path,
) -> None:
    manifest = _valid_package(tmp_path)
    anchor_path = tmp_path / GENOMICS_TRUST_ANCHOR_PATH
    result_path = tmp_path / "genomics/results.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["runtime_versions"]["numpy"] = "fabricated"
    anchor["provenance"]["runtime_versions_at_start"]["numpy"] = "fabricated"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["runtime_versions"] = anchor["runtime_versions"]
    result["provenance"] = anchor["provenance"]
    result_data = (
        json.dumps(result, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    anchor["result_sha256"] = _sha256(result_data)
    anchor_data = (
        json.dumps(anchor, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    for path, data in ((anchor_path, anchor_data), (result_path, result_data)):
        path.write_bytes(data)
        relative = path.relative_to(tmp_path).as_posix()
        entry = next(item for item in manifest["files"] if item["path"] == relative)
        entry["bytes"] = len(data)
        entry["sha256"] = _sha256(data)
    _rewrite_manifest(tmp_path, manifest)

    with pytest.raises(ReleaseValidationError, match="frozen external"):
        validate_nonfed_release(tmp_path, require_genomics_semantics=False)

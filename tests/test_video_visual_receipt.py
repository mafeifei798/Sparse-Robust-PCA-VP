from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

import tools.verify_video_visual_v3_receipt as receipt
from tools.verify_video_visual_v3_receipt import (
    ReceiptValidationError,
    build_receipt,
    prepare_execution_documents,
    validate_checked_copy,
    validate_self_excluding_manifest,
    write_self_excluding_manifest,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CHECKED_PACKAGE = REPOSITORY / "assets" / "evaluation" / "video" / "v4_receipt"
EXPECTED_RECEIPT_SHA256 = (
    "c0a1d0178bfd85d9d9f83c8ae06a62744ebcb46317fa9399a9d4453686990a85"
)
EXPECTED_SERVER_ROOT_MANIFEST_SHA256 = (
    "95b4dacdd9edfa61c947e14fa162ba4804000123d1e4265bebc374af860c7963"
)
EXPECTED_LOCAL_MANIFEST_FILE_SHA256 = (
    "4a539b28e55503397cd531249807894c71cb3dcf37ea61ffad52a90cbc390e67"
)


def _copy_checked_repository(tmp_path: Path) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    receipt_payload = json.loads(
        (CHECKED_PACKAGE / "run_receipt.json").read_text(encoding="utf-8")
    )
    for relative in receipt_payload["source_snapshot"]["files"]:
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / relative, destination)
    video_root = repository / "assets" / "evaluation" / "video"
    video_root.mkdir(parents=True, exist_ok=True)
    for seed in (91001, 91007):
        shutil.copytree(
            REPOSITORY / "assets" / "evaluation" / "video" / f"seed_{seed}",
            video_root / f"seed_{seed}",
        )
    package = video_root / "v4_receipt"
    shutil.copytree(CHECKED_PACKAGE, package)
    return repository, package


def test_real_checked_receipt_is_source_command_payload_and_server_bound_read_only() -> (
    None
):
    before = {
        path.name: (path.stat().st_mtime_ns, receipt._file_sha256(path))
        for path in CHECKED_PACKAGE.iterdir()
        if path.is_file()
    }

    report = validate_checked_copy(CHECKED_PACKAGE, repo_root=REPOSITORY)

    assert report == {
        "receipt_sha256": EXPECTED_RECEIPT_SHA256,
        "server_root_manifest_sha256": EXPECTED_SERVER_ROOT_MANIFEST_SHA256,
        "seeds": [91001, 91007],
    }
    assert (
        receipt._file_sha256(CHECKED_PACKAGE / "local_manifest.json")
        == EXPECTED_LOCAL_MANIFEST_FILE_SHA256
    )
    run_receipt = json.loads(
        (CHECKED_PACKAGE / "run_receipt.json").read_text(encoding="utf-8")
    )
    assert run_receipt["source_snapshot"]["files"][
        "tools/verify_video_visual_v3_receipt.py"
    ]["sha256"] == receipt._file_sha256(
        CHECKED_PACKAGE / "source" / "tools" / "verify_video_visual_v3_receipt.py"
    )
    commands = json.loads(
        (CHECKED_PACKAGE / "run_commands.json").read_text(encoding="utf-8")
    )
    assert commands["environment"]["PYTHONDONTWRITEBYTECODE"] == "1"
    server_root = json.loads(
        (CHECKED_PACKAGE / "server_root_manifest.json").read_text(encoding="utf-8")
    )
    assert not any(
        "__pycache__" in relative or relative.endswith(".pyc")
        for relative in server_root["files"]
    )
    after = {
        path.name: (path.stat().st_mtime_ns, receipt._file_sha256(path))
        for path in CHECKED_PACKAGE.iterdir()
        if path.is_file()
    }
    assert after == before


def test_real_checked_receipt_rejects_packaged_execution_source_tamper(
    tmp_path: Path,
) -> None:
    repository, package = _copy_checked_repository(tmp_path)
    source = package / "source" / "experiments" / "video_spca_visualize.py"
    source.write_bytes(source.read_bytes() + b"\n# tampered source\n")

    with pytest.raises(ReceiptValidationError, match="size mismatch|SHA-256 mismatch"):
        validate_checked_copy(package, repo_root=repository)


def test_checked_receipt_survives_later_repository_source_maintenance(
    tmp_path: Path,
) -> None:
    repository, package = _copy_checked_repository(tmp_path)
    maintained = repository / "src" / "rspca" / "randomized.py"
    maintained.write_bytes(maintained.read_bytes() + b"\n# later maintenance\n")

    report = validate_checked_copy(package, repo_root=repository)

    assert report["receipt_sha256"] == EXPECTED_RECEIPT_SHA256


@pytest.mark.parametrize("tamper_kind", ["projected_manifest", "gif_payload"])
def test_real_checked_receipt_rejects_manifest_or_payload_tamper(
    tamper_kind: str,
    tmp_path: Path,
) -> None:
    repository, package = _copy_checked_repository(tmp_path)
    seed_dir = repository / "assets" / "evaluation" / "video" / "seed_91001"
    if tamper_kind == "projected_manifest":
        path = seed_dir / "video_visualization_manifest.json"
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = seed_dir / "video_decomposition.gif"
        content = bytearray(path.read_bytes())
        content[-1] ^= 1
        path.write_bytes(content)

    with pytest.raises(
        (ReceiptValidationError, ValueError),
        match="size mismatch|SHA-256 mismatch|hash/size mismatch",
    ):
        validate_checked_copy(package, repo_root=repository)


def _prepare_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "isolated-root"
    for relative in sorted(receipt._REQUIRED_SOURCE_PATHS):
        path = root / "source" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen test source: {relative}\n", encoding="utf-8")
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "MKL_NUM_THREADS": "8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(tmp_path / "mpl-config"),
        "NUMEXPR_NUM_THREADS": "8",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": f"{root}/source/src:{root}/source",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    return root


def test_prepare_execution_documents_are_strict_and_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepare_root(tmp_path, monkeypatch)

    paths = prepare_execution_documents(root)

    environment = receipt._strict_json(paths["environment"])
    commands = receipt._strict_json(paths["commands"])
    assert (
        receipt._validate_environment_spec(environment)
        == environment["environment_sha256"]
    )
    assert receipt._validate_command_spec(commands) == commands["command_spec_sha256"]
    assert commands["execution_root"] == str(root)
    assert commands["source_sha256"] == {
        relative: receipt._file_sha256(root / "source" / relative)
        for relative in sorted(receipt._REQUIRED_SOURCE_PATHS)
    }


@pytest.mark.parametrize("tamper_kind", ["frozen_source", "declared_source_hash"])
def test_receipt_build_rejects_wrong_command_source_binding(
    tamper_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepare_root(tmp_path, monkeypatch)
    paths = prepare_execution_documents(root)
    source_path = root / "source" / "experiments" / "video_spca.py"
    if tamper_kind == "frozen_source":
        source_path.write_text("changed after command capture\n", encoding="utf-8")
    else:
        commands = json.loads(paths["commands"].read_text(encoding="utf-8"))
        commands["source_sha256"]["experiments/video_spca.py"] = "0" * 64
        commands.pop("command_spec_sha256")
        commands["command_spec_sha256"] = receipt.canonical_sha256(commands)
        paths["commands"].write_text(
            json.dumps(commands, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(ReceiptValidationError, match="command/source SHA mismatch"):
        build_receipt(root, repo_root=tmp_path, checked_asset_root=tmp_path)
    assert not (root / "run_receipt.json").exists()
    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("reserved_name", ["environment.json", "run_commands.json"])
def test_prepare_execution_documents_preflight_never_overwrites(
    reserved_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _prepare_root(tmp_path, monkeypatch)
    reserved = root / reserved_name
    reserved.write_bytes(b"belongs to another capture")

    with pytest.raises(ReceiptValidationError, match="already exists"):
        prepare_execution_documents(root)
    assert reserved.read_bytes() == b"belongs to another capture"
    other = root / ({"environment.json", "run_commands.json"} - {reserved_name}).pop()
    assert not os.path.lexists(other)


def test_self_excluding_manifest_validates_read_only_and_rejects_payload_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"sealed payload")
    manifest = write_self_excluding_manifest(
        root,
        manifest_name="manifest.json",
        manifest_kind="unit_test",
    )
    before = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.iterdir()
    }

    validated = validate_self_excluding_manifest(
        root,
        manifest_name="manifest.json",
        require_exact_files=True,
    )

    assert validated["manifest_kind"] == "unit_test"
    assert manifest.is_file()
    after = {
        path.name: (path.stat().st_mtime_ns, path.read_bytes())
        for path in root.iterdir()
    }
    assert after == before

    payload.write_bytes(b"tampered payload")
    with pytest.raises(ReceiptValidationError, match="size mismatch|SHA-256 mismatch"):
        validate_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            require_exact_files=True,
        )


def test_self_excluding_manifest_rejects_undeclared_file(tmp_path: Path) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    (root / "payload.bin").write_bytes(b"payload")
    write_self_excluding_manifest(
        root,
        manifest_name="manifest.json",
        manifest_kind="unit_test",
    )
    (root / "undeclared.bin").write_bytes(b"not in the manifest")

    with pytest.raises(ReceiptValidationError, match="file closure mismatch"):
        validate_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            require_exact_files=True,
        )


def test_manifest_closure_rejects_symlink(tmp_path: Path) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    target = tmp_path / "outside.bin"
    target.write_bytes(b"outside")
    (root / "linked.bin").symlink_to(target)

    with pytest.raises(ReceiptValidationError, match="symlinks are forbidden"):
        write_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            manifest_kind="unit_test",
        )
    assert target.read_bytes() == b"outside"


@pytest.mark.parametrize("cache_kind", ["directory", "pyc_file"])
def test_manifest_closure_rejects_python_bytecode_cache(
    cache_kind: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    if cache_kind == "directory":
        (root / "__pycache__").mkdir()
    else:
        (root / "module.pyc").write_bytes(b"compiled bytecode is not provenance")

    with pytest.raises(ReceiptValidationError, match="bytecode/cache is forbidden"):
        write_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            manifest_kind="unit_test",
        )


def test_verifier_imports_never_create_bytecode_without_environment_flag(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "frozen" / "source"
    for relative in (
        "experiments/__init__.py",
        "experiments/video_spca.py",
        "experiments/video_spca_visualize.py",
        "tools/verify_video_visual_v3_receipt.py",
        "pyproject.toml",
    ):
        destination = source / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository / relative, destination)
    (source / "src" / "rspca").mkdir(parents=True)
    for path in (repository / "src" / "rspca").glob("*.py"):
        shutil.copy2(path, source / "src" / "rspca" / path.name)

    environment = os.environ.copy()
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment["PYTHONPATH"] = f"{source / 'src'}:{source}"
    subprocess.run(
        [
            sys.executable,
            str(source / "tools" / "verify_video_visual_v3_receipt.py"),
            "--help",
        ],
        cwd=source,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not list(source.rglob("__pycache__"))
    assert not list(source.rglob("*.pyc"))


@pytest.mark.parametrize("invalid_json", ['{"key": 1, "key": 2}', '{"key": NaN}'])
def test_strict_json_rejects_duplicate_or_nonfinite_values(
    invalid_json: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ReceiptValidationError, match="duplicate JSON key|non-finite"):
        receipt._strict_json(path)


@pytest.mark.parametrize("existing_kind", ["file", "directory", "dangling_symlink"])
def test_self_excluding_manifest_never_overwrites_existing_target(
    existing_kind: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    path = root / "manifest.json"
    if existing_kind == "file":
        path.write_bytes(b"belongs to another writer")
    elif existing_kind == "directory":
        path.mkdir()
    else:
        path.symlink_to(tmp_path / "absent-manifest-target")
    before = path.read_bytes() if path.is_file() and not path.is_symlink() else None

    with pytest.raises(ReceiptValidationError):
        write_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            manifest_kind="unit_test",
        )

    assert os.path.lexists(path)
    if before is not None:
        assert path.read_bytes() == before
    elif existing_kind == "directory":
        assert path.is_dir()
    else:
        assert path.is_symlink()


def test_exclusive_manifest_write_loses_race_without_overwriting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "closure"
    root.mkdir()
    target = root / "manifest.json"
    replacement = b"concurrent writer owns this file"
    original_records = receipt._relative_file_records

    def race_after_scan(directory: Path, *, excluded: set[str]):
        records = original_records(directory, excluded=excluded)
        target.write_bytes(replacement)
        return records

    monkeypatch.setattr(receipt, "_relative_file_records", race_after_scan)
    with pytest.raises(ReceiptValidationError, match="refusing to overwrite"):
        write_self_excluding_manifest(
            root,
            manifest_name="manifest.json",
            manifest_kind="unit_test",
        )
    assert target.read_bytes() == replacement


@pytest.mark.parametrize("reserved_name", ["run_receipt.json", "manifest.json"])
@pytest.mark.parametrize("existing_kind", ["file", "dangling_symlink"])
def test_receipt_build_preflight_preserves_reserved_targets(
    reserved_name: str,
    existing_kind: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "isolated-root"
    root.mkdir()
    reserved = root / reserved_name
    if existing_kind == "file":
        reserved.write_bytes(b"existing receipt owner")
    else:
        reserved.symlink_to(tmp_path / "missing-receipt-target")

    with pytest.raises(ReceiptValidationError, match="already exists"):
        build_receipt(root, repo_root=tmp_path, checked_asset_root=tmp_path)

    assert os.path.lexists(reserved)
    if existing_kind == "file":
        assert reserved.read_bytes() == b"existing receipt owner"
    else:
        assert reserved.is_symlink()

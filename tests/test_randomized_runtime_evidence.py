from __future__ import annotations

import hashlib
import importlib._bootstrap_external
import json
from pathlib import Path
import shutil
import sys

import pytest

from tools import randomized_runtime_evidence as evidence


EVIDENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "randomized_runtime_optimization_2026-08-30"
    / "evidence.json"
)


def _payload() -> dict[str, object]:
    return json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))


def _write_rehashed(path: Path, payload: dict[str, object]) -> None:
    payload["canonical_payload_sha256"] = evidence._canonical_payload_sha256(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _matching_timestamp_pyc(source_path: Path, malicious_source: bytes) -> Path:
    source_stat = source_path.stat()
    code = compile(malicious_source, str(source_path), "exec")
    pyc = importlib._bootstrap_external._code_to_timestamp_pyc(
        code,
        int(source_stat.st_mtime),
        source_stat.st_size,
    )
    cache_path = (
        source_path.parent
        / "__pycache__"
        / f"{source_path.stem}.{sys.implementation.cache_tag}.pyc"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pyc)
    return cache_path


def test_committed_evidence_validates_and_recomputes_numerical_contract() -> None:
    report = evidence.validate_evidence(EVIDENCE_PATH, recompute_outputs=True)

    assert report["valid"] is True
    assert report["outputs_recomputed"] is True
    assert report["case_count"] == 3
    assert report["outer_manifest_sha256"]
    assert set(report["median_reduction_percent"]) == {
        "formal_400x200_full_refine",
        "video_wide_180x1024",
        "genomics_wide_957x5000",
    }


def test_rejects_rehashed_but_incorrect_median_and_reduction(tmp_path: Path) -> None:
    payload = _payload()
    first_case = payload["cases"][0]  # type: ignore[index]
    first_case["runtime"]["before"]["median_seconds"] += 0.25  # type: ignore[index,operator]
    first_case["runtime"]["median_reduction_percent"] = 99.0  # type: ignore[index]
    target = tmp_path / "wrong-median.json"
    _write_rehashed(target, payload)

    with pytest.raises(
        evidence.EvidenceValidationError,
        match="runtime.before median does not match samples",
    ):
        evidence.validate_evidence(target)


def test_rejects_rehashed_source_hash_tampering(tmp_path: Path) -> None:
    payload = _payload()
    payload["execution_sources"]["before"]["sha256"] = "0" * 64  # type: ignore[index]
    target = tmp_path / "wrong-source.json"
    _write_rehashed(target, payload)

    with pytest.raises(
        evidence.EvidenceValidationError,
        match="execution source SHA-256 mismatch for before",
    ):
        evidence.validate_evidence(target)


def test_recompute_rejects_resigned_array_identity_and_delta_tampering(
    tmp_path: Path,
) -> None:
    """Regression for a re-signed attack that the tolerance-only check accepted."""

    payload = _payload()
    case = payload["cases"][0]  # type: ignore[index]
    record = case["equivalence"]["numeric_fields"]["A"]  # type: ignore[index]
    record["before"]["content_sha256"] = "0" * 64  # type: ignore[index]
    record["before"]["array_sha256"] = "f" * 64  # type: ignore[index]
    record["max_abs_difference"] = 0.0  # type: ignore[index]
    record["relative_l2_difference"] = 0.0  # type: ignore[index]
    record["within_tolerance"] = True  # type: ignore[index]
    target = tmp_path / "resigned-array-identity-attack.json"
    _write_rehashed(target, payload)

    with pytest.raises(
        evidence.EvidenceValidationError,
        match=r"recomputed array identity mismatch for .*\.A\.before",
    ):
        evidence.validate_evidence(target, recompute_outputs=True)


def test_recompute_rejects_resigned_diagnostic_identity_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload()
    case = payload["cases"][0]  # type: ignore[index]
    record = case["equivalence"]["numeric_fields"]["full_objective"]  # type: ignore[index]
    record["before"]["content_sha256"] = "0" * 64  # type: ignore[index]
    record["before"]["array_sha256"] = "f" * 64  # type: ignore[index]
    target = tmp_path / "resigned-diagnostic-attack.json"
    _write_rehashed(target, payload)

    with pytest.raises(
        evidence.EvidenceValidationError,
        match=r"recomputed array identity mismatch for .*\.full_objective\.before",
    ):
        evidence.validate_evidence(target, recompute_outputs=True)


def test_recompute_rejects_resigned_discrete_value_tampering(tmp_path: Path) -> None:
    payload = _payload()
    case = payload["cases"][0]  # type: ignore[index]
    record = case["equivalence"]["discrete_fields"]["status"]  # type: ignore[index]
    record["before"] = "forged status"  # type: ignore[index]
    record["after"] = "forged status"  # type: ignore[index]
    record["equal"] = True  # type: ignore[index]
    target = tmp_path / "resigned-discrete-attack.json"
    _write_rehashed(target, payload)

    with pytest.raises(
        evidence.EvidenceValidationError,
        match=r"recomputed discrete field mismatch for .*\.status",
    ):
        evidence.validate_evidence(target, recompute_outputs=True)


def test_snapshot_loader_ignores_matching_malicious_timestamp_pyc(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    vp_path = assets / "vp.py"
    randomized_path = assets / "randomized.py"
    vp_source = b'VALUE = "source-vp"\n'
    randomized_source = b"from .vp import VALUE\n" b'ORIGIN = "source-randomized"\n'
    vp_path.write_bytes(vp_source)
    randomized_path.write_bytes(randomized_source)
    vp_pyc = _matching_timestamp_pyc(vp_path, b'VALUE = "malicious-vp"\n')
    randomized_pyc = _matching_timestamp_pyc(
        randomized_path,
        b'VALUE = "malicious-vp"\nORIGIN = "malicious-randomized"\n',
    )
    original_pyc_bytes = {
        vp_pyc: vp_pyc.read_bytes(),
        randomized_pyc: randomized_pyc.read_bytes(),
    }

    module = evidence._load_snapshot(
        "matching_pyc_attack",
        randomized_path,
        randomized_sha256=hashlib.sha256(randomized_source).hexdigest(),
        vp_path=vp_path,
        vp_sha256=hashlib.sha256(vp_source).hexdigest(),
        asset_root=assets,
    )

    assert module.ORIGIN == "source-randomized"
    assert module.VALUE == "source-vp"
    assert {
        path: path.read_bytes() for path in original_pyc_bytes
    } == original_pyc_bytes
    assert set((assets / "__pycache__").iterdir()) == set(original_pyc_bytes)


def test_snapshot_loader_does_not_create_pycache_and_checks_source_hash(
    tmp_path: Path,
) -> None:
    assets = tmp_path / "clean-assets"
    assets.mkdir()
    vp_path = assets / "vp.py"
    randomized_path = assets / "randomized.py"
    vp_source = b'VALUE = "source-vp"\n'
    randomized_source = b"from .vp import VALUE\n"
    vp_path.write_bytes(vp_source)
    randomized_path.write_bytes(randomized_source)

    module = evidence._load_snapshot(
        "no_cache",
        randomized_path,
        randomized_sha256=hashlib.sha256(randomized_source).hexdigest(),
        vp_path=vp_path,
        vp_sha256=hashlib.sha256(vp_source).hexdigest(),
        asset_root=assets,
    )
    assert module.VALUE == "source-vp"
    assert not (assets / "__pycache__").exists()

    with pytest.raises(evidence.EvidenceValidationError, match="bound source SHA-256"):
        evidence._load_snapshot(
            "wrong_hash",
            randomized_path,
            randomized_sha256="0" * 64,
            vp_path=vp_path,
            vp_sha256=hashlib.sha256(vp_source).hexdigest(),
            asset_root=assets,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("payload", "package SHA-256 mismatch for evidence.json"),
        ("undeclared", "contains undeclared or missing files"),
    ],
)
def test_outer_manifest_rejects_payload_tampering_and_undeclared_files(
    tmp_path: Path, mutation: str, message: str
) -> None:
    package = tmp_path / "package"
    shutil.copytree(EVIDENCE_PATH.parent, package)
    copied_evidence = package / "evidence.json"
    if mutation == "payload":
        copied_evidence.write_bytes(copied_evidence.read_bytes().replace(b"\n", b" ", 1))
    else:
        (package / "undeclared.txt").write_text("not declared\n", encoding="utf-8")

    with pytest.raises(evidence.EvidenceValidationError, match=message):
        evidence.validate_evidence(copied_evidence)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_rejects_nonfinite_runtime_sample(tmp_path: Path, constant: str) -> None:
    original = EVIDENCE_PATH.read_text(encoding="utf-8")
    marker = '"samples_seconds": ['
    prefix, suffix = original.split(marker, maxsplit=1)
    comma = suffix.index(",")
    tampered = prefix + marker + "\n          " + constant + suffix[comma:]
    target = tmp_path / "nonfinite.json"
    target.write_text(tampered, encoding="utf-8")

    with pytest.raises(
        evidence.EvidenceValidationError, match="non-finite JSON constant"
    ):
        evidence.validate_evidence(target)


def test_rejects_duplicate_json_key(tmp_path: Path) -> None:
    original = EVIDENCE_PATH.read_text(encoding="utf-8")
    tampered = original.replace("{\n", '{\n  "schema_version": 1,\n', 1)
    target = tmp_path / "duplicate-key.json"
    target.write_text(tampered, encoding="utf-8")

    with pytest.raises(evidence.EvidenceValidationError, match="duplicate JSON key"):
        evidence.validate_evidence(target)


def test_generator_refuses_to_overwrite_existing_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "owned.json"
    target.write_text("owner data\n", encoding="utf-8")

    def unexpected_generation(*, warmups: int, repeats: int) -> dict[str, object]:
        raise AssertionError("generation must not begin after output preflight")

    monkeypatch.setattr(evidence, "generate_evidence", unexpected_generation)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evidence.main(["generate", "--output", str(target)])

    assert target.read_text(encoding="utf-8") == "owner data\n"

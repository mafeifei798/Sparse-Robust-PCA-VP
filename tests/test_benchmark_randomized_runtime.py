from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from tools import benchmark_randomized_runtime as benchmark


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _assert_nested_finite(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_nested_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_nested_finite(item)
    elif isinstance(value, float):
        assert math.isfinite(value)


@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_cli_preflights_existing_output_before_benchmark_computation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_kind: str,
) -> None:
    target = tmp_path / "already-owned.json"
    if target_kind == "file":
        target.write_text("owner data\n", encoding="utf-8")
    else:
        target.mkdir()

    def unexpected_benchmark(arguments: object) -> dict[str, object]:
        raise AssertionError("benchmark computation must not start after preflight")

    monkeypatch.setattr(benchmark, "run_benchmark", unexpected_benchmark)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        benchmark.main(["--output-json", str(target)])

    if target_kind == "file":
        assert target.read_text(encoding="utf-8") == "owner data\n"
    else:
        assert target.is_dir()


def test_exclusive_create_preserves_target_created_after_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "racing-owner.json"

    def racing_benchmark(arguments: object) -> dict[str, object]:
        target.write_text("racing owner\n", encoding="utf-8")
        return {"nested": {"finite": 1.0}}

    monkeypatch.setattr(benchmark, "run_benchmark", racing_benchmark)
    with pytest.raises(FileExistsError):
        benchmark.main(["--output-json", str(target)])

    assert target.read_text(encoding="utf-8") == "racing owner\n"


def test_matrix_and_source_hashes_are_repeatable_and_tamper_sensitive(
    tmp_path: Path,
) -> None:
    first = benchmark._matrix(10, 7, 2, 2025)
    repeated = benchmark._matrix(10, 7, 2, 2025)
    first_record = benchmark._matrix_provenance(first)
    repeated_record = benchmark._matrix_provenance(repeated)

    assert np.array_equal(first, repeated)
    assert first_record == repeated_record
    assert first_record["dtype"] == "<f8"
    assert first_record["shape"] == [10, 7]

    tampered = first.copy()
    tampered[0, 0] = np.nextafter(tampered[0, 0], np.inf)
    tampered_record = benchmark._matrix_provenance(tampered)
    assert tampered_record["content_sha256"] != first_record["content_sha256"]
    assert tampered_record["array_sha256"] != first_record["array_sha256"]

    reshaped_record = benchmark._matrix_provenance(first.reshape(14, 5))
    assert reshaped_record["content_sha256"] == first_record["content_sha256"]
    assert reshaped_record["shape"] == [14, 5]
    assert reshaped_record["array_sha256"] != first_record["array_sha256"]

    source_records = benchmark._source_provenance()
    assert set(source_records) == {"randomized", "vp", "benchmark_tool"}
    for record in source_records.values():
        relative_path = Path(record["relative_path"])
        assert not relative_path.is_absolute()
        assert record["relative_to"] == "repository_root"
        source_path = REPOSITORY_ROOT / relative_path
        assert record["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()

    probe = tmp_path / "source.py"
    probe.write_bytes(b"value = 1\n")
    original_digest = benchmark._sha256_file(probe)
    probe.write_bytes(b"value = 2\n")
    assert benchmark._sha256_file(probe) != original_digest


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_nested_nonfinite_report_hard_fails_before_output_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bad_value: float,
) -> None:
    target = tmp_path / "new" / "result.json"
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda arguments: {"outer": [{"diagnostic": np.float64(bad_value)}]},
    )

    with pytest.raises(
        ValueError,
        match=r"number must be finite at root\.outer\[0\]\.diagnostic",
    ):
        benchmark.main(["--output-json", str(target)])

    assert not target.exists()
    assert not target.parent.exists()


def test_cli_smoke_emits_bound_strict_report_and_fixed_budget_quality(
    tmp_path: Path,
) -> None:
    target = tmp_path / "runtime.json"
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "tools" / "benchmark_randomized_runtime.py"),
        "--n-samples",
        "12",
        "--n-features",
        "8",
        "--rank",
        "2",
        "--alphas",
        "0.1",
        "--oversample",
        "2",
        "--n-power-iter",
        "0",
        "--max-iter",
        "2",
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--output-json",
        str(target),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert json.loads(completed.stdout) == payload
    assert payload["schema_version"] == 2
    assert payload["deterministic_matrix"]["dtype"] == "<f8"
    assert payload["deterministic_matrix"]["shape"] == [12, 8]
    assert len(payload["deterministic_matrix"]["content_sha256"]) == 64
    assert len(payload["deterministic_matrix"]["array_sha256"]) == 64

    environment_record = payload["environment"]
    assert environment_record["platform"]
    assert environment_record["machine"]
    assert Path(environment_record["executable"]).name == environment_record["executable"]
    assert environment_record["executable_path_policy"].startswith("basename_only")

    for record in payload["source_files"].values():
        assert not Path(record["relative_path"]).is_absolute()
        source_path = REPOSITORY_ROOT / record["relative_path"]
        assert record["sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()

    quality = payload["fixed_budget_quality_one_alpha"]
    vp_objective = quality["vp"]["final_objective"]
    randomized_objective = quality["randomized"]["full_objective"]
    gap = randomized_objective - vp_objective
    assert quality["randomized_minus_vp_objective"] == pytest.approx(gap)
    assert quality[
        "randomized_minus_vp_objective_relative_to_abs_vp"
    ] == pytest.approx(gap / abs(vp_objective))
    assert "not a convergence or quality-matched claim" in quality["interpretation"]
    _assert_nested_finite(payload)

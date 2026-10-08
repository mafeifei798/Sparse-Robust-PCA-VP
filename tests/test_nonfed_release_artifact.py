from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import pytest

from experiments import genomics_1kg_benchmark as benchmark
from experiments.genomics_1kg_benchmark import (
    GenomicsBenchmarkConfig,
    validate_result_payload,
)

from tools.validate_nonfed_release import (
    EVIDENCE_SCOPE,
    REQUIRED_CAPABILITIES,
    ReleaseValidationError,
    validate_nonfed_release,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPOSITORY_ROOT / "results" / "nonfed_2026-08-29"
MANIFEST_PATH = PACKAGE_ROOT / "manifest.json"

EXPECTED_CANONICAL_PAYLOADS = frozenset(
    {
        "README.md",
        "provenance/environment.json",
        "provenance/evaluation_config.json",
        "provenance/combine_genomics_benchmark_shards.py",
        "provenance/genomics_1kg_execution.py",
        "provenance/genomics_1kg_benchmark_execution.py",
        "provenance/genomics_1kg_benchmark_combination.py",
        "provenance/genomics_formal_config.json",
        "provenance/genomics_trusted_provenance.json",
        "provenance/randomized_execution.py",
        "provenance/run_genomics_combine.sh",
        "provenance/run_genomics_formal.sh",
        "provenance/run_genomics_shard.sh",
        "provenance/run_spca_refined.sh",
        "provenance/run_video_qualified_after_genomics.sh",
        "provenance/source_manifest.json",
        "provenance/spca_refined_config.json",
        "provenance/verification.json",
        "spca/clean/metrics.csv",
        "spca/clean/results.json",
        "spca/corrupt/metrics.csv",
        "spca/corrupt/results.json",
        "spca/summary.csv",
        "spca/summary.json",
        "video/metrics.csv",
        "video/results.json",
        "genomics/metrics.csv",
        "genomics/results.json",
    }
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    not MANIFEST_PATH.is_file(),
    reason="the dated release is still being assembled and has no manifest yet",
)
def test_checked_in_nonfed_release_is_closed_and_uses_canonical_layout() -> None:
    """Validate the actual dated artifact once its self-excluding manifest exists."""

    report = validate_nonfed_release(PACKAGE_ROOT)
    manifest = _load_json(MANIFEST_PATH)
    declared_paths = {entry["path"] for entry in manifest["files"]}
    payload_paths = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and path != MANIFEST_PATH
    }

    assert manifest["evidence_scope"] == EVIDENCE_SCOPE
    assert set(report.capabilities) == set(REQUIRED_CAPABILITIES) | {"supporting"}
    assert declared_paths == payload_paths
    assert EXPECTED_CANONICAL_PAYLOADS <= declared_paths
    assert report.file_count == len(declared_paths)


def test_packaged_execution_snapshots_match_source_provenance() -> None:
    """Bind packaged source, scripts, and configs to their provenance digests."""

    source_manifest = _load_json(PACKAGE_ROOT / "provenance/source_manifest.json")
    randomized_entries = [
        entry
        for entry in source_manifest["execution_files"]
        if entry["path"] == "src/rspca/randomized.py"
    ]
    assert randomized_entries == [
        {
            "packaged_execution_copy": "provenance/randomized_execution.py",
            "path": "src/rspca/randomized.py",
            "sha256": (
                "37b68488d25abf775f97ec7ed755a1f5" "b00aceaa6fd08e29d044fe01e39fe0c5"
            ),
        }
    ]

    packaged_entries = []
    for group in (
        "execution_files",
        "execution_scripts",
        "frozen_remote_configs",
        "genomics_combination_files",
    ):
        packaged_entries.extend(
            entry
            for entry in source_manifest[group]
            if "packaged_execution_copy" in entry
        )

    assert packaged_entries, "at least one execution-time source snapshot is required"
    for entry in packaged_entries:
        snapshot = PACKAGE_ROOT / entry["packaged_execution_copy"]
        assert snapshot.is_file()
        assert _sha256(snapshot) == entry["sha256"]


def test_frozen_result_hash_rejects_coherent_internal_and_outer_resign(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "nonfed-attack"
    shutil.copytree(PACKAGE_ROOT, copied)
    result_path = copied / "genomics/results.json"
    payload = _load_json(result_path)
    config_values = dict(payload["config"])
    for field_name in ("seeds", "methods", "alpha_candidates"):
        config_values[field_name] = tuple(config_values[field_name])
    config = GenomicsBenchmarkConfig(**config_values)
    full_run = payload["full_runs"][0]
    classification = next(
        row
        for row in full_run["records"]
        if row["record_type"] == "held_out_population_classification"
    )
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        classification[metric] = float(classification[metric]) - 0.01
    rebuilt = benchmark._one_record(
        method=full_run["method"],
        run_index=int(full_run["run_index"]),
        requested_seed=int(full_run["requested_seed"]),
        runtime_seconds=float(full_run["runtime_seconds"]),
        classification=classification,
        single_manifest=full_run["seed_manifest"],
        benchmark_manifest_sha256=payload["manifest"]["manifest_sha256"],
        provenance=payload["provenance"],
    )
    rebuilt["full_run_index"] = int(full_run["full_run_index"])
    payload["records"][int(full_run["full_run_index"])] = rebuilt
    payload["method_summaries"] = benchmark.summarize_method_metrics(
        payload["records"],
        methods=config.methods,
        seeds=config.seeds,
        benchmark_manifest_sha256=payload["manifest"]["manifest_sha256"],
    )
    payload["paired_summaries"] = benchmark.summarize_paired_differences(
        payload["records"],
        methods=config.methods,
        seeds=config.seeds,
        benchmark_manifest_sha256=payload["manifest"]["manifest_sha256"],
        bootstrap_resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
    )
    # All payload-authored full-run/record/summary layers still validate, and
    # the formal provenance remains unchanged.  The release-level frozen
    # result hash is the independent boundary that rejects this rewrite.
    validate_result_payload(
        payload,
        label="coherently resigned internal Genomics payload",
        expected_provenance=payload["provenance"],
    )
    _write_json(result_path, payload)
    outer = _load_json(copied / "manifest.json")
    result_entry = next(
        entry for entry in outer["files"] if entry["path"] == "genomics/results.json"
    )
    result_entry["bytes"] = result_path.stat().st_size
    result_entry["sha256"] = _sha256(result_path)
    _write_json(copied / "manifest.json", outer)

    with pytest.raises(ReleaseValidationError, match="externally frozen result"):
        validate_nonfed_release(copied)

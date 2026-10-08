from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import pytest

from experiments import genomics_all_autosome_runtime_benchmark as runtime
from experiments import genomics_pca_loading_all_autosomes as all_auto
from rspca.datasets import array_sha256, canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_PACKAGE = (
    REPO_ROOT
    / "results"
    / "genomics_all_autosome_stratification_2026-08-31"
    / "remote_v2_clean"
    / "registered_20seeds_merged"
)
FORMAL_RESULT = FORMAL_PACKAGE / all_auto.RESULT_FILENAME
FORMAL_RUNTIME_ROOT = REPO_ROOT / "results" / "genomics_all_autosome_runtime_2026-08-31"
FORMAL_RUNTIME_PACKAGE = FORMAL_RUNTIME_ROOT / "registered_20seeds_rank8_5pct"
FORMAL_RUNTIME_PROVENANCE = FORMAL_RUNTIME_ROOT / "provenance"
FORMAL_RUNTIME_ANALYSIS_SHA256 = (
    "da7eaaa209d0bca48538e12b0a586fbf18695a3d278f75a6ed7991cd48e0343a"
)
FORMAL_RUNTIME_PACKAGE_ANCHOR_SHA256 = (
    "1c31951ea39e9a0d766937b3d7467e8489c4ff1264d04610c06e230ceffff87a"
)
FORMAL_RUNTIME_FILE_SHA256 = {
    "genomics_all_autosome_runtime_output_manifest.json": (
        "4f0ed73026161591ee215ae880a64be15336828ac90dc0f279c0f147ab95bbcf"
    ),
    "genomics_all_autosome_runtime_paired.png": (
        "420b78570c900a50ad9ad4b2a06f07b555f40ddfae52af14843b4aa06829e839"
    ),
    "genomics_all_autosome_runtime_records.csv": (
        "f6d58f2d1e3cb1c970e33fdf7ed90f23fa63f87ad927d245309b8b3f5de6ee47"
    ),
    "genomics_all_autosome_runtime_results.json": (
        "6dfe306abe426a8fd88b904046d77f55a37614874d3704cfe1ad07d3e4224778"
    ),
    "genomics_all_autosome_runtime_summary.csv": (
        "74b29ffd9533380d3187640549bc10dfe01b85d731f812536120c27d70782bbf"
    ),
}
FORMAL_RUNTIME_PROVENANCE_SHA256 = {
    "formal.log": ("deba9b09c483e30ec17b42f9da7a0b0c85087aafb3ae9ecbf0ec679603e4d839"),
    "genomics_all_autosome_runtime_benchmark.md": (
        "30d3d70aa19bed672b110988e30439286cb6890b76f00d9d1f7643487dd18985"
    ),
    "genomics_all_autosome_runtime_benchmark.py": (
        "96e0ceabb8a9bad6c87a885573fb0acef67476fb4bf713088f21c7203442bf63"
    ),
    "test_genomics_all_autosome_runtime_benchmark.py": (
        "376a1591b73cf9c3b31ccc16ba4d97add6101f6ef9ce2dd80b096a11c1323d5d"
    ),
}


@pytest.fixture(scope="module")
def smoke_data():
    return all_auto._synthetic_smoke_data()


@pytest.fixture(scope="module")
def smoke_reference(smoke_data) -> dict[str, Any]:
    return all_auto.evaluate_all_autosome_stratification(
        smoke_data,
        all_auto.AllAutosomeConfig(
            seeds=runtime.SMOKE_MODE.seeds,
            analysis_mode="tiny_smoke",
            snp_chunk_size=runtime.SMOKE_MODE.snp_chunk_size,
        ),
        input_identity=all_auto._synthetic_input_identity(smoke_data),
        source_identity=all_auto._source_identity(REPO_ROOT),
    )


@pytest.fixture(scope="module")
def tiny_runtime_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("all-autosome-runtime-smoke")
    output = root / "package"
    validation = runtime.run_tiny_smoke(
        output_directory=output,
        scratch_root=root,
        repo_root=REPO_ROOT,
    )
    assert validation["package_self_consistency_verified"] is True
    return output


def _resign(value: dict[str, Any], digest_key: str) -> None:
    unsigned = dict(value)
    unsigned.pop(digest_key, None)
    value[digest_key] = canonical_sha256(unsigned)


def _timing(*, wall: float = 1.0, cpu: float = 0.8, rss: int = 1024) -> dict[str, Any]:
    return {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "ru_maxrss_before_bytes": max(0, rss - 128),
        "ru_maxrss_after_bytes": rss,
        "ru_maxrss_delta_bytes_lower_bound": min(128, rss),
        "ru_maxrss_raw_unit": "bytes",
    }


def _worker_output(
    job: Mapping[str, Any],
    *,
    contract_sha256: str,
    pid: int = 43210,
    stderr: bytes | None = None,
    case: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    kind = str(job["worker_kind"])
    case_payload = (
        {
            "seed": int(job["seed"]),
            "all_reference": {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "test_predictions_sha256": "a" * 64,
            },
            "panel_reference": {
                "accuracy": 1.0,
                "balanced_accuracy": 1.0,
                "macro_f1": 1.0,
                "test_predictions_sha256": "a" * 64,
            },
            "selected_snp_count": 8,
            "selected_original_indices_sha256": "b" * 64,
        }
        if case is None
        else dict(case)
    )
    science = {
        "case": case_payload,
        "panel_observed": dict(case_payload["panel_reference"]),
    }
    if kind == "discovery":
        science["all_observed"] = dict(case_payload["all_reference"])
        science["selection"] = {
            "selected_snp_count": int(case_payload["selected_snp_count"]),
            "selected_original_indices_sha256": str(
                case_payload["selected_original_indices_sha256"]
            ),
        }
        science["materialized_compact_panel"] = copy.deepcopy(science["selection"])
        stages = {
            "all_fit_projection_and_score": _timing(wall=2.0),
            "rank_and_select": _timing(wall=0.1),
            "compact_panel_copy_and_flush": _timing(wall=0.2),
            "compact_panel_fit_projection_and_score": _timing(wall=0.4),
        }
    else:
        science["compact_panel_receipt"] = {
            "selected_snp_count": int(case_payload["selected_snp_count"]),
            "selected_original_indices_sha256": str(
                case_payload["selected_original_indices_sha256"]
            ),
        }
        stages = {"compact_panel_fit_projection_and_score": _timing(wall=0.4)}
    output: dict[str, Any] = {
        "schema_version": 1,
        "job": dict(job),
        "worker_pid": pid,
        "fresh_child_worker": True,
        "runtime_environment": {
            "pid": pid,
            "affinity_count": 1,
            "blas_runtime_identity": {
                "threadpoolctl_version": "test",
                "blas_pools": [],
            },
        },
        "stages": stages,
        "total": _timing(wall=2.8 if kind == "discovery" else 0.4),
        "science": science,
        "science_sha256": canonical_sha256(science),
        "job_contract_sha256": contract_sha256,
    }
    if stderr is not None:
        output["subprocess_stderr_receipt"] = {
            "bytes": len(stderr),
            "sha256": hashlib.sha256(stderr).hexdigest(),
            "text_tail": stderr.decode("utf-8", errors="replace")[-2000:],
        }
    return output


def _build_smoke_contract(
    root: Path,
    smoke_data,
    smoke_reference: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    raw_path = root / "genotypes.npy"
    metadata_path = root / "metadata.npz"
    reference_path = root / all_auto.RESULT_FILENAME
    panel_path = root / "compact_panel_seed_73.npy"
    np.save(raw_path, np.asarray(smoke_data.genotypes), allow_pickle=False)
    runtime._write_worker_metadata(
        metadata_path,
        sample_ids=np.asarray(smoke_data.sample_ids),
        population_labels=np.asarray(smoke_data.population_labels),
    )
    reference_path.write_bytes(runtime._strict_json_bytes(smoke_reference))
    case = runtime._reference_case(
        smoke_reference,
        seed=runtime.SMOKE_MODE.seeds[0],
        width=int(smoke_data.genotypes.shape[1]),
    )
    compact = runtime.materialize_compact_panel(
        np.asarray(smoke_data.genotypes),
        runtime._selected_mask(case),
        panel_path,
        chunk_size=runtime.SMOKE_MODE.snp_chunk_size,
    )
    contract = runtime._build_job_contract(
        mode=runtime.SMOKE_MODE,
        source_identity=runtime._source_identity(REPO_ROOT),
        reference_receipt=runtime._reference_file_receipt(reference_path),
        raw_receipt=runtime._raw_npy_receipt(raw_path),
        metadata_receipt=runtime._metadata_receipt(metadata_path),
        compact_receipts={runtime.SMOKE_MODE.seeds[0]: compact},
        formal_package=None,
    )
    contract_path = root / "worker_job_contract.json"
    contract_path.write_bytes(runtime._strict_json_bytes(contract))
    return contract_path, contract


def _smoke_public_case(
    smoke_data, smoke_reference: Mapping[str, Any]
) -> dict[str, Any]:
    return runtime._public_case(
        runtime._reference_case(
            smoke_reference,
            seed=runtime.SMOKE_MODE.seeds[0],
            width=int(smoke_data.genotypes.shape[1]),
        )
    )


def test_runtime_protocol_constants_and_balanced_worker_schedule_are_locked() -> None:
    assert runtime.FORMAL_SEEDS == tuple(range(2025, 2045))
    assert runtime.FORMAL_RANK == 8
    assert runtime.FORMAL_PANEL_LABEL == "5pct"
    assert runtime.FORMAL_PANEL_FRACTION == 0.05
    assert runtime.EXCLUDED_WARMUPS == 1
    assert runtime.TIMED_REPETITIONS == 3
    assert runtime.BOOTSTRAP_RESAMPLES == 100_000
    assert runtime.BOOTSTRAP_SEED == 831_2026
    assert runtime.FORMAL_MODE.require_formal_anchors is True
    assert runtime.FORMAL_MODE.require_single_cpu is True
    assert runtime.FORMAL_MODE.require_blas_one is True
    assert runtime.FORMAL_PROTOCOL["status"] == ("retrospective_paired_runtime_audit")
    assert runtime.FORMAL_PROTOCOL["panel_selection"] == (
        "sealed_train_only_rank8_loading_mask_for_same_split"
    )
    assert runtime.FORMAL_PROTOCOL["runtime_used_for_panel_or_rank_selection"] is False
    assert (
        "verified-hot actual model input"
        in runtime.FORMAL_PROTOCOL["filesystem_cache_scope"]
    )

    smoke_jobs = runtime._worker_schedule(runtime.SMOKE_MODE)
    assert len(smoke_jobs) == 6
    assert [job["worker_kind"] for job in smoke_jobs] == [
        "discovery",
        "deployment",
        "discovery",
        "deployment",
        "deployment",
        "discovery",
    ]
    assert [job["phase"] for job in smoke_jobs] == [
        "warmup",
        "warmup",
        "timed",
        "timed",
        "timed",
        "timed",
    ]
    assert [job["repetition"] for job in smoke_jobs] == [-1, -1, 0, 0, 1, 1]

    with pytest.raises(ValueError, match="formal runtime protocol differs"):
        runtime._validate_mode(
            runtime.BenchmarkMode(
                **{
                    **runtime.FORMAL_MODE.__dict__,
                    "repeats": runtime.TIMED_REPETITIONS + 1,
                }
            )
        )


def test_strict_json_rejects_duplicate_keys_nonfinite_and_noncanonical_bytes() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key: value"):
        runtime._strict_json_decode(
            b'{"value": 1, "value": 2}\n', role="duplicate-test"
        )
    for constant in (b"NaN", b"Infinity", b"-Infinity"):
        with pytest.raises(ValueError, match="non-finite JSON constant"):
            runtime._strict_json_decode(
                b'{"value": ' + constant + b"}\n", role="nonfinite-test"
            )
    with pytest.raises(ValueError, match="non-finite JSON value"):
        runtime._strict_json_bytes({"nested": [float("inf")]})
    with pytest.raises(ValueError, match="not canonical strict JSON"):
        runtime._strict_json_decode(b'{"value":1}\n', role="spacing-test")
    canonical = runtime._strict_json_bytes({"value": [1, None, "x"]})
    assert runtime._strict_json_decode(canonical, role="roundtrip") == {
        "value": [1, None, "x"]
    }


def test_reference_rejects_a_stale_analysis_digest(
    smoke_reference: dict[str, Any],
) -> None:
    attacked = copy.deepcopy(smoke_reference)
    attacked["config"]["test_fraction"] = 0.3
    with pytest.raises(ValueError, match="analysis SHA is stale or invalid"):
        runtime._validate_reference_payload(attacked, runtime.SMOKE_MODE)


def test_committed_formal_reference_is_bound_to_exact_source_and_input() -> None:
    reference = runtime._strict_json_load(FORMAL_RESULT)
    runtime._validate_reference_payload(reference, runtime.FORMAL_MODE)
    assert reference["analysis_sha256"] == runtime.FORMAL_ANALYSIS_SHA256
    assert reference["input_identity"]["genotype_npz_sha256"] == (
        all_auto.FROZEN_FORMAL_NPZ_SHA256
    )
    assert (
        reference["source_identity"]["files"][
            "experiments/genomics_pca_loading_all_autosomes.py"
        ]
        == runtime.FORMAL_RUNNER_SHA256
    )


def test_formal_job_contract_rejects_stale_hash_and_package_anchor() -> None:
    placeholder_receipt = {"path": "/not-opened", "bytes": 1, "sha256": "a" * 64}
    placeholder_panel = {
        **placeholder_receipt,
        "shape": [1000, 22_755],
        "dtype": "<f4",
        "array_sha256": "b" * 64,
        "selected_snp_count": 22_755,
        "selected_original_indices_sha256": "c" * 64,
    }
    contract = runtime._build_job_contract(
        mode=runtime.FORMAL_MODE,
        source_identity={"test": True},
        reference_receipt=placeholder_receipt,
        raw_receipt=placeholder_receipt,
        metadata_receipt=placeholder_receipt,
        compact_receipts={
            seed: copy.deepcopy(placeholder_panel) for seed in runtime.FORMAL_SEEDS
        },
        formal_package={
            "analysis_sha256": runtime.FORMAL_ANALYSIS_SHA256,
            "package_anchor_sha256": runtime.FORMAL_PACKAGE_ANCHOR_SHA256,
            "package_self_consistency_verified": True,
            "protocol_seed_cover_complete": True,
        },
    )
    assert runtime._validate_job_contract(contract) == runtime.FORMAL_MODE

    stale = copy.deepcopy(contract)
    stale["nonce"] = "changed-without-resigning"
    with pytest.raises(ValueError, match="job-contract hash mismatch"):
        runtime._validate_job_contract(stale)

    wrong_package = copy.deepcopy(contract)
    wrong_package["formal_package"]["package_anchor_sha256"] = "0" * 64
    _resign(wrong_package, "contract_sha256")
    with pytest.raises(ValueError, match="lacks sealed-package validation"):
        runtime._validate_job_contract(wrong_package)


def test_split_mask_and_scientific_receipts_fail_closed(
    smoke_data,
    smoke_reference: dict[str, Any],
) -> None:
    seed = runtime.SMOKE_MODE.seeds[0]
    case = runtime._reference_case(
        smoke_reference,
        seed=seed,
        width=int(smoke_data.genotypes.shape[1]),
    )
    train, test = runtime._split_from_reference(
        np.asarray(smoke_data.population_labels),
        np.asarray(smoke_data.sample_ids),
        case,
    )
    assert train.size + test.size == smoke_data.population_labels.size
    assert array_sha256(test) == case["test_indices_sha256"]

    mask = runtime._selected_mask(case)
    selected = np.flatnonzero(mask).astype(np.int64)
    assert selected.size == case["selected_snp_count"]
    assert array_sha256(selected) == case["selected_original_indices_sha256"]

    bad_split = dict(case)
    bad_split["test_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="split differs from formal evidence"):
        runtime._split_from_reference(
            np.asarray(smoke_data.population_labels),
            np.asarray(smoke_data.sample_ids),
            bad_split,
        )

    bad_mask = dict(case)
    bad_mask["selected_original_indices_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="panel mask differs"):
        runtime._selected_mask(bad_mask)

    observed = dict(case["panel_reference"])
    runtime._verify_science(observed, case["panel_reference"], role="panel")
    observed["balanced_accuracy"] = float(observed["balanced_accuracy"]) - 0.01
    with pytest.raises(RuntimeError, match="differ from formal evidence"):
        runtime._verify_science(observed, case["panel_reference"], role="panel")


def test_compact_panel_is_bound_to_the_original_selected_columns(
    tmp_path: Path,
    smoke_data,
    smoke_reference: dict[str, Any],
) -> None:
    case = runtime._reference_case(
        smoke_reference,
        seed=runtime.SMOKE_MODE.seeds[0],
        width=int(smoke_data.genotypes.shape[1]),
    )
    mask = runtime._selected_mask(case)
    destination = tmp_path / "compact.npy"
    receipt = runtime.materialize_compact_panel(
        np.asarray(smoke_data.genotypes),
        mask,
        destination,
        chunk_size=runtime.SMOKE_MODE.snp_chunk_size,
    )
    runtime.verify_compact_panel_binding(
        np.asarray(smoke_data.genotypes),
        mask,
        destination,
        chunk_size=runtime.SMOKE_MODE.snp_chunk_size,
    )
    assert receipt["selected_snp_count"] == case["selected_snp_count"]
    assert receipt["selected_original_indices_sha256"] == (
        case["selected_original_indices_sha256"]
    )
    assert runtime._load_compact_panel(destination, receipt).shape == tuple(
        receipt["shape"]
    )

    compact = np.load(destination, mmap_mode="r+")
    compact[0, 0] = np.nextafter(compact[0, 0], np.float32(np.inf))
    compact.flush()
    del compact
    with pytest.raises(ValueError, match="differs from its parent receipt"):
        runtime._load_compact_panel(destination, receipt)


def test_worker_contract_rejects_unknown_job_and_output_schema(
    tmp_path: Path,
    smoke_data,
    smoke_reference: dict[str, Any],
) -> None:
    contract_path, contract = _build_smoke_contract(
        tmp_path, smoke_data, smoke_reference
    )
    assert runtime._validate_job_contract(contract) == runtime.SMOKE_MODE
    with pytest.raises(ValueError, match="expected exactly one authorized job"):
        runtime._load_authorized_job(contract_path, "not-in-contract")

    job = contract["jobs"][0]
    output = _worker_output(
        job,
        contract_sha256=contract["contract_sha256"],
        case=_smoke_public_case(smoke_data, smoke_reference),
    )
    runtime.validate_worker_output(
        output,
        expected_job=job,
        mode=runtime.SMOKE_MODE,
        expected_contract_sha256=contract["contract_sha256"],
    )
    partial_evidence = copy.deepcopy(output)
    partial_evidence["science"] = {
        "case": {"seed": int(job["seed"])},
        "panel_observed": copy.deepcopy(output["science"]["panel_observed"]),
    }
    partial_evidence["science_sha256"] = canonical_sha256(partial_evidence["science"])
    with pytest.raises(ValueError, match="evidence case closure mismatch"):
        runtime.validate_worker_output(
            partial_evidence,
            expected_job=job,
            mode=runtime.SMOKE_MODE,
            expected_contract_sha256=contract["contract_sha256"],
        )
    attacked = copy.deepcopy(output)
    attacked["science"]["case"]["seed"] = 999
    with pytest.raises(ValueError, match="science hash mismatch"):
        runtime.validate_worker_output(
            attacked,
            expected_job=job,
            mode=runtime.SMOKE_MODE,
            expected_contract_sha256=contract["contract_sha256"],
        )


def test_execute_worker_checks_popen_pid_and_records_smoke_stderr(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    smoke_data,
    smoke_reference: dict[str, Any],
) -> None:
    contract_path, contract = _build_smoke_contract(
        tmp_path, smoke_data, smoke_reference
    )
    job = contract["jobs"][0]

    class FakeProcess:
        returncode = 0

        def __init__(self, pid: int, stdout: bytes, stderr: bytes) -> None:
            self.pid = pid
            self._stdout = stdout
            self._stderr = stderr

        def communicate(self) -> tuple[bytes, bytes]:
            return self._stdout, self._stderr

    child_pid = 50123
    stderr = b"smoke-only diagnostic\n"
    valid_output = _worker_output(
        job,
        contract_sha256=contract["contract_sha256"],
        pid=child_pid,
        case=_smoke_public_case(smoke_data, smoke_reference),
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(
            child_pid, runtime._strict_json_bytes(valid_output), stderr
        ),
    )
    observed = runtime._execute_worker_job(
        contract_path=contract_path,
        job=job,
        scratch_root=tmp_path,
        mode=runtime.SMOKE_MODE,
    )
    assert observed["worker_pid"] == child_pid
    assert observed["subprocess_stderr_receipt"] == {
        "bytes": len(stderr),
        "sha256": hashlib.sha256(stderr).hexdigest(),
        "text_tail": stderr.decode(),
    }

    mismatched = _worker_output(
        job,
        contract_sha256=contract["contract_sha256"],
        pid=child_pid + 1,
        case=_smoke_public_case(smoke_data, smoke_reference),
    )
    monkeypatch.setattr(
        runtime.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(
            child_pid, runtime._strict_json_bytes(mismatched), b""
        ),
    )
    with pytest.raises(RuntimeError, match="Popen PID, worker PID"):
        runtime._execute_worker_job(
            contract_path=contract_path,
            job=job,
            scratch_root=tmp_path,
            mode=runtime.SMOKE_MODE,
        )


def test_formal_worker_output_forbids_any_stderr_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = runtime._worker_schedule(runtime.FORMAL_MODE)[0]
    output = _worker_output(
        job,
        contract_sha256="d" * 64,
        stderr=b"warning\n",
    )
    monkeypatch.setattr(
        all_auto,
        "_validate_blas_runtime_identity",
        lambda *args, **kwargs: None,
    )
    with pytest.raises(ValueError, match="formal worker stderr must remain empty"):
        runtime.validate_worker_output(
            output,
            expected_job=job,
            mode=runtime.FORMAL_MODE,
            expected_contract_sha256="d" * 64,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"wall_seconds": float("nan")}, "wall_seconds"),
        ({"cpu_seconds": -0.1}, "cpu_seconds"),
        ({"ru_maxrss_after_bytes": True}, "ru_maxrss_after_bytes"),
        ({"ru_maxrss_after_bytes": -1}, "ru_maxrss_after_bytes"),
    ],
)
def test_timing_schema_rejects_nonfinite_negative_and_boolean_values(
    mutation: dict[str, Any], message: str
) -> None:
    value = _timing()
    value.update(mutation)
    with pytest.raises(ValueError, match=message):
        runtime._validate_timing(value, role="probe")


def test_timing_schema_and_rss_scope_are_explicit() -> None:
    value, timing = runtime._timed_call(lambda: "done")
    assert value == "done"
    runtime._validate_timing(timing, role="timed_call")
    assert timing["wall_seconds"] >= 0.0
    assert timing["cpu_seconds"] >= 0.0
    assert timing["ru_maxrss_after_bytes"] >= timing["ru_maxrss_before_bytes"]
    assert timing["ru_maxrss_delta_bytes_lower_bound"] >= 0

    before = {
        "wall_clock_seconds": 10.0,
        "process_cpu_seconds": 4.0,
        "ru_maxrss_bytes": 1_000,
        "ru_maxrss_raw_unit": "KiB",
    }
    after = {
        "wall_clock_seconds": 12.5,
        "process_cpu_seconds": 5.5,
        "ru_maxrss_bytes": 1_800,
        "ru_maxrss_raw_unit": "KiB",
    }
    total = runtime._total_timing(before, after)
    assert total["wall_seconds"] == 2.5
    assert total["cpu_seconds"] == 1.5
    assert total["ru_maxrss_delta_bytes_lower_bound"] == 800
    assert "lower_bound" in total["rss_claim_scope"]


def test_paired_ratio_bootstrap_uses_seed_level_pairing_and_is_deterministic() -> None:
    deployment = np.asarray([2.0])
    all_reference = np.asarray([4.0])
    discovery = np.asarray([6.0])
    reduction = runtime._paired_ratio_bootstrap(
        deployment,
        all_reference,
        mode=runtime.SMOKE_MODE,
        estimand="deployment_reduction_fraction",
    )
    repeated = runtime._paired_ratio_bootstrap(
        deployment,
        all_reference,
        mode=runtime.SMOKE_MODE,
        estimand="deployment_reduction_fraction",
    )
    assert reduction == repeated
    assert reduction["estimate"] == 0.5
    assert reduction["ci95_low"] == 0.5
    assert reduction["ci95_high"] == 0.5
    assert reduction["resamples"] == runtime.SMOKE_MODE.bootstrap_resamples
    assert reduction["unit"] == ("registered_outer_split_seed_after_per_seed_median")

    speedup = runtime._paired_ratio_bootstrap(
        deployment,
        all_reference,
        mode=runtime.SMOKE_MODE,
        estimand="deployment_speedup",
    )
    overhead = runtime._paired_ratio_bootstrap(
        discovery,
        all_reference,
        mode=runtime.SMOKE_MODE,
        estimand="discovery_overhead_fraction",
    )
    assert speedup["estimate"] == 2.0
    assert overhead["estimate"] == 0.5
    with pytest.raises(ValueError, match="positive per-seed median vectors"):
        runtime._paired_ratio_bootstrap(
            np.asarray([0.0]),
            all_reference,
            mode=runtime.SMOKE_MODE,
            estimand="deployment_speedup",
        )


def test_tiny_smoke_runs_fresh_workers_and_emits_a_self_consistent_package(
    tiny_runtime_package: Path,
) -> None:
    validation = runtime.validate_output_package(
        tiny_runtime_package,
        repo_root=REPO_ROOT,
    )
    payload = runtime._strict_json_load(tiny_runtime_package / runtime.RESULT_FILENAME)
    manifest = runtime._strict_json_load(
        tiny_runtime_package / runtime.MANIFEST_FILENAME
    )
    assert validation["package_self_consistency_verified"] is True
    assert validation["formal_anchors_verified"] is False
    assert payload["status"] == "complete_tiny_smoke_engineering_only"
    assert payload["source_result_binding"]["package_anchor_sha256"] is None
    assert payload["fresh_child_worker_invocations"] == 6
    assert len(payload["records"]) == 2
    assert len(payload["timed_worker_outputs"]) == 4
    assert all(
        output["subprocess_stderr_receipt"]["sha256"]
        == hashlib.sha256(
            output["subprocess_stderr_receipt"]["text_tail"].encode()
        ).hexdigest()
        for output in payload["timed_worker_outputs"]
        if output["subprocess_stderr_receipt"]["bytes"]
        == len(output["subprocess_stderr_receipt"]["text_tail"].encode())
    )
    assert manifest["artifact_count_excluding_manifest"] == 4
    assert manifest["exact_package_file_count"] == 5
    assert set(manifest["artifacts"]) == set(runtime.ARTIFACT_FILENAMES)


def test_runtime_result_rejects_a_stale_analysis_digest(
    tiny_runtime_package: Path,
) -> None:
    payload = runtime._strict_json_load(tiny_runtime_package / runtime.RESULT_FILENAME)
    payload["analysis_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="runtime analysis SHA mismatch"):
        runtime.validate_result_payload(payload)


def test_output_is_exclusive_and_manifest_closure_rejects_tampering(
    tmp_path: Path,
    tiny_runtime_package: Path,
) -> None:
    payload = runtime._strict_json_load(tiny_runtime_package / runtime.RESULT_FILENAME)
    owner = tmp_path / "owner"
    owner.mkdir()
    marker = owner / "keep.txt"
    marker.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output directory already exists"):
        runtime.write_output_package(payload, owner)
    assert marker.read_text(encoding="utf-8") == "owned\n"

    copied = tmp_path / "copied"
    shutil.copytree(tiny_runtime_package, copied)
    (copied / "undeclared.txt").write_text("not in manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure mismatch"):
        runtime.validate_output_package(copied)

    copied = tmp_path / "tampered"
    shutil.copytree(tiny_runtime_package, copied)
    records = copied / runtime.RECORDS_FILENAME
    records.write_bytes(records.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="artifact receipt mismatch"):
        runtime.validate_output_package(copied)


def test_external_runtime_anchor_validation_requires_an_exact_pair(
    tiny_runtime_package: Path,
) -> None:
    with pytest.raises(ValueError, match="requires both runtime anchors"):
        runtime.validate_output_package(
            tiny_runtime_package,
            expected_analysis_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="differs from supplied external anchors"):
        runtime.validate_output_package(
            tiny_runtime_package,
            expected_analysis_sha256="0" * 64,
            expected_package_anchor_sha256="1" * 64,
        )


def test_no_nan_or_infinite_number_survives_the_smoke_payload(
    tiny_runtime_package: Path,
) -> None:
    payload = runtime._strict_json_load(tiny_runtime_package / runtime.RESULT_FILENAME)

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, float):
            assert math.isfinite(value)

    walk(payload)
    assert os.path.lexists(tiny_runtime_package)


def test_downloaded_formal_runtime_package_is_exact_and_externally_anchored() -> None:
    assert FORMAL_RUNTIME_PACKAGE.is_dir()
    assert not FORMAL_RUNTIME_PACKAGE.is_symlink()
    assert {path.name for path in FORMAL_RUNTIME_PACKAGE.iterdir()} == set(
        FORMAL_RUNTIME_FILE_SHA256
    )
    for name, expected_sha256 in FORMAL_RUNTIME_FILE_SHA256.items():
        path = FORMAL_RUNTIME_PACKAGE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert runtime._file_sha256(path) == expected_sha256

    validation = runtime.validate_output_package(
        FORMAL_RUNTIME_PACKAGE,
        repo_root=REPO_ROOT,
        expected_analysis_sha256=FORMAL_RUNTIME_ANALYSIS_SHA256,
        expected_package_anchor_sha256=FORMAL_RUNTIME_PACKAGE_ANCHOR_SHA256,
    )
    assert validation == {
        "analysis_sha256": FORMAL_RUNTIME_ANALYSIS_SHA256,
        "formal_anchors_verified": True,
        "fresh_child_worker_invocations": 160,
        "manifest_payload_sha256": (
            "75d3fe21ada06ace825469046117af8e6616157e129519f6bdb942a48ea680d0"
        ),
        "package_anchor_sha256": FORMAL_RUNTIME_PACKAGE_ANCHOR_SHA256,
        "package_self_consistency_verified": True,
        "seed_count": 20,
        "timed_record_count": 60,
    }


def test_downloaded_formal_runtime_result_locks_effects_and_claim_boundaries() -> None:
    payload = runtime._strict_json_load(
        FORMAL_RUNTIME_PACKAGE / runtime.RESULT_FILENAME
    )
    assert payload["status"] == "complete_formal_retrospective_runtime_audit"
    assert payload["analysis_sha256"] == FORMAL_RUNTIME_ANALYSIS_SHA256
    assert payload["fresh_child_worker_invocations"] == 160
    assert len(payload["excluded_warmup_receipts"]) == 40
    assert len(payload["timed_worker_outputs"]) == 120
    assert len(payload["records"]) == 60
    assert len(payload["per_seed_medians"]) == 20
    assert payload["mode"]["seeds"] == list(range(2025, 2045))
    assert payload["mode"]["bootstrap_resamples"] == 100_000
    assert payload["mode"]["bootstrap_seed"] == 831_2026

    summary = payload["aggregate_summary"]
    assert math.isclose(
        summary["all_reference_mean_of_seed_medians_seconds"],
        15.593407317483798,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        summary["frozen_panel_deployment_mean_of_seed_medians_seconds"],
        0.8386407867074013,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        summary["prepared_input_discovery_mean_of_seed_medians_seconds"],
        16.92077802415006,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    expected_intervals = {
        "deployment_reduction": (
            0.946218246619705,
            0.9457344921116707,
            0.9467092255273191,
        ),
        "deployment_speedup": (
            18.593666757736983,
            18.42791192626192,
            18.764974048419944,
        ),
        "discovery_overhead": (
            0.08512383981517457,
            0.08370457768483772,
            0.08659080304409618,
        ),
    }
    for key, (estimate, low, high) in expected_intervals.items():
        block = summary[key]
        assert math.isclose(block["estimate"], estimate, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(block["ci95_low"], low, rel_tol=0.0, abs_tol=1e-15)
        assert math.isclose(block["ci95_high"], high, rel_tol=0.0, abs_tol=1e-15)
        assert block["resamples"] == 100_000
        assert block["seed"] == 831_2026
        assert block["unit"] == "registered_outer_split_seed_after_per_seed_median"

    assert all(
        row["frozen_panel_deployment_wall_seconds"]
        < row["all_reference_wall_seconds"]
        < row["prepared_input_discovery_wall_seconds"]
        for row in payload["per_seed_medians"]
    )
    assert all(row["all_snp_count"] == 455_102 for row in payload["records"])
    assert all(row["selected_snp_count"] == 22_755 for row in payload["records"])
    assert payload["source_result_binding"]["analysis_sha256"] == (
        "46190a1e12c9c0213b70b8e9778579cebcf8374915874436d26cea1805402981"
    )
    assert payload["source_result_binding"]["package_anchor_sha256"] == (
        "4afd97e46570a3cec4d32be83c1abc878d6b1ee04bb7a6efc0d5c66757040d9a"
    )
    assert (
        payload["analysis_policy"][
            "prepared_input_discovery_is_not_entire_snp_pipeline"
        ]
        is True
    )
    assert (
        payload["analysis_policy"]["finite_panels_are_not_one_universal_aim_list"]
        is True
    )
    rss_values = {
        summary["all_reference_absolute_peak_rss_mean_of_seed_medians_bytes"],
        summary["frozen_panel_deployment_absolute_peak_rss_mean_of_seed_medians_bytes"],
        summary[
            "prepared_input_discovery_absolute_peak_rss_mean_of_seed_medians_bytes"
        ],
    }
    assert rss_values == {4_239_544_320.0}


def test_downloaded_formal_runtime_provenance_is_separate_and_exact() -> None:
    assert FORMAL_RUNTIME_PROVENANCE.is_dir()
    assert not FORMAL_RUNTIME_PROVENANCE.is_symlink()
    assert {path.name for path in FORMAL_RUNTIME_PROVENANCE.iterdir()} == {
        *FORMAL_RUNTIME_PROVENANCE_SHA256,
        "formal_execution_receipt.json",
    }
    for name, expected_sha256 in FORMAL_RUNTIME_PROVENANCE_SHA256.items():
        path = FORMAL_RUNTIME_PROVENANCE / name
        assert path.is_file()
        assert not path.is_symlink()
        assert runtime._file_sha256(path) == expected_sha256

    receipt = json.loads(
        (FORMAL_RUNTIME_PROVENANCE / "formal_execution_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "complete_and_downloaded"
    assert receipt["analysis_sha256"] == FORMAL_RUNTIME_ANALYSIS_SHA256
    assert receipt["package_anchor_sha256"] == (FORMAL_RUNTIME_PACKAGE_ANCHOR_SHA256)
    assert receipt["package_file_sha256"] == FORMAL_RUNTIME_FILE_SHA256
    assert receipt["package_self_consistency_verified_remote"] is True
    assert receipt["package_self_consistency_verified_after_download"] is True

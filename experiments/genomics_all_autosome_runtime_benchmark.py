#!/usr/bin/env python3
"""Paired runtime audit for the all-autosome PCA-loading result.

The benchmark deliberately separates two different estimands:

* ``deployment`` times a rank-eight fit and held-out projection when a
  split-specific 5% panel has already been frozen and materialized; and
* ``discovery`` times the complete full-width PCA/loading screen, loading
  ranking, compact-panel materialization, panel refit/projection, and scoring.

The first estimand can measure recurring computation saved by an already
frozen panel.  It is not the runtime of discovering that panel.  The second
estimand charges the full-width screen and therefore normally costs more than
the all-SNP reference for a single fit.  Every scientific output is checked
against the sealed all-autosome result; timing is never used to choose a panel.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar
import uuid

import numpy as np

if __package__:
    from . import genomics_pca_loading_all_autosomes as all_auto
    from . import genomics_pca_loading_grid as grid
else:
    import genomics_pca_loading_all_autosomes as all_auto  # type: ignore[no-redef]
    import genomics_pca_loading_grid as grid  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import stratified_split


SCHEMA_VERSION = 1
FORMAL_SEEDS = tuple(range(2025, 2045))
FORMAL_RANK = 8
FORMAL_PANEL_LABEL = "5pct"
FORMAL_PANEL_FRACTION = 0.05
EXCLUDED_WARMUPS = 1
TIMED_REPETITIONS = 3
BOOTSTRAP_RESAMPLES = 100_000
BOOTSTRAP_SEED = 831_2026
SNP_CHUNK_SIZE = 2048

FORMAL_ANALYSIS_SHA256 = (
    "46190a1e12c9c0213b70b8e9778579cebcf8374915874436d26cea1805402981"
)
FORMAL_PACKAGE_ANCHOR_SHA256 = (
    "4afd97e46570a3cec4d32be83c1abc878d6b1ee04bb7a6efc0d5c66757040d9a"
)
FORMAL_RUNNER_SHA256 = (
    "97087a965c7e4137698fadfd6910171606c55b8448a8e6550a91c6363769885d"
)

RESULT_FILENAME = "genomics_all_autosome_runtime_results.json"
RECORDS_FILENAME = "genomics_all_autosome_runtime_records.csv"
SUMMARY_FILENAME = "genomics_all_autosome_runtime_summary.csv"
FIGURE_FILENAME = "genomics_all_autosome_runtime_paired.png"
MANIFEST_FILENAME = "genomics_all_autosome_runtime_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    FIGURE_FILENAME,
)

SOURCE_FILE_ROLES = {
    "experiments/genomics_all_autosome_runtime_benchmark.py": "runtime_benchmark",
    "experiments/genomics_pca_loading_all_autosomes.py": "sealed_scientific_runner",
    "experiments/genomics_pca_loading_grid.py": "loading_ranking_and_metrics",
    "src/rspca/__init__.py": "eager_package_initialization",
    "src/rspca/genomics_aim.py": "registered_split",
    "src/rspca/datasets.py": "hashing_and_runtime_identity",
}

FORMAL_PROTOCOL = {
    "status": "retrospective_paired_runtime_audit",
    "seeds": list(FORMAL_SEEDS),
    "rank": FORMAL_RANK,
    "panel_label": FORMAL_PANEL_LABEL,
    "panel_fraction": FORMAL_PANEL_FRACTION,
    "excluded_warmups_per_seed_and_worker_kind": EXCLUDED_WARMUPS,
    "timed_repetitions_per_seed_and_worker_kind": TIMED_REPETITIONS,
    "worker_process_policy": "fresh_child_process_for_every_worker_invocation",
    "timed_pair_order": "alternating_by_seed_index_plus_repetition",
    "blas_threads": 1,
    "cpu_affinity": "exactly_one_logical_cpu_for_formal_execution",
    "filesystem_cache_scope": (
        "per-worker verified-hot actual model input after child-side SHA binding; "
        "discovery hashes raw NPY and deployment hashes compact NPY; shared "
        "metadata/reference are verified by both; no privileged OS cache drop"
    ),
    "deployment_estimand": (
        "fit_score_on_already_frozen_split_specific_5pct_compact_panel"
    ),
    "prepared_input_discovery_estimand": (
        "starting_from_verified_prepared_genotype_NPY: full_fit_score_plus_"
        "rank_select_plus_panel_materialization_plus_panel_fit_score; excludes "
        "source_download, VCF/dosage generation, QC/LD pruning, NPZ extraction, "
        "pre-timing hashes, child startup, and post-timing integrity hashes"
    ),
    "all_reference_estimand": (
        "full_width_rank8_fit_projection_and_score_subspan_inside_each_"
        "discovery_child_not_a_separate_per_fit_worker"
    ),
    "panel_selection": "sealed_train_only_rank8_loading_mask_for_same_split",
    "runtime_used_for_panel_or_rank_selection": False,
    "best_runtime_configuration_selected": False,
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_unit": "registered_outer_split_seed_after_per_seed_median",
    "bootstrap_interval": "paired_percentile_95_for_mean",
}

T = TypeVar("T")


@dataclass(frozen=True)
class BenchmarkMode:
    name: str
    seeds: tuple[int, ...]
    warmups: int
    repeats: int
    bootstrap_resamples: int
    bootstrap_seed: int
    require_formal_anchors: bool
    require_single_cpu: bool
    require_blas_one: bool
    snp_chunk_size: int = SNP_CHUNK_SIZE


FORMAL_MODE = BenchmarkMode(
    name="formal",
    seeds=FORMAL_SEEDS,
    warmups=EXCLUDED_WARMUPS,
    repeats=TIMED_REPETITIONS,
    bootstrap_resamples=BOOTSTRAP_RESAMPLES,
    bootstrap_seed=BOOTSTRAP_SEED,
    require_formal_anchors=True,
    require_single_cpu=True,
    require_blas_one=True,
)

SMOKE_MODE = BenchmarkMode(
    name="tiny_smoke",
    seeds=(73,),
    warmups=1,
    repeats=2,
    bootstrap_resamples=256,
    bootstrap_seed=73_831,
    require_formal_anchors=False,
    require_single_cpu=False,
    require_blas_one=False,
    snp_chunk_size=257,
)


def _strict_json_bytes(value: Any) -> bytes:
    _reject_nonfinite(value)
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_json_decode(payload: bytes, *, role: str) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, item in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = item
        return output

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=object_pairs,
        parse_constant=reject_constant,
    )
    _reject_nonfinite(value)
    if payload != _strict_json_bytes(value):
        raise ValueError(f"JSON is not canonical strict JSON: {role}")
    return value


def _strict_json_load(path: Path) -> Any:
    return _strict_json_decode(Path(path).read_bytes(), role=str(path))


def _reject_nonfinite(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    path = Path(path)
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular file: {path}")
    return status


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files = {relative: _file_sha256(root / relative) for relative in SOURCE_FILE_ROLES}
    if (
        files["experiments/genomics_pca_loading_all_autosomes.py"]
        != FORMAL_RUNNER_SHA256
    ):
        raise ValueError("sealed all-autosome runner differs from the formal anchor")
    return {
        "files": files,
        "roles": dict(SOURCE_FILE_ROLES),
        "source_manifest_sha256": canonical_sha256(
            {"files": files, "roles": SOURCE_FILE_ROLES}
        ),
        "formal_runner_sha256_verified": True,
    }


def _cpu_affinity() -> list[int] | None:
    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    return sorted(int(value) for value in getter(0))


def _require_runtime_contract(mode: BenchmarkMode) -> dict[str, Any]:
    affinity = _cpu_affinity()
    if mode.require_single_cpu and (affinity is None or len(affinity) != 1):
        raise ValueError(
            "formal runtime execution requires affinity to exactly one logical CPU"
        )
    if mode.require_blas_one:
        blas = all_auto._require_formal_blas_thread_environment()
    else:
        blas = all_auto._blas_runtime_identity()
    return {
        "affinity": affinity,
        "affinity_count": None if affinity is None else len(affinity),
        "cpu_model": platform.processor(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "pid": os.getpid(),
        "blas_runtime_identity": blas,
        "thread_environment": all_auto._blas_thread_environment(),
        "runtime_versions": runtime_versions(),
    }


def _rss_raw_and_bytes() -> tuple[int, str, int]:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform.startswith("linux"):
        return raw, "KiB", raw * 1024
    return raw, "bytes", raw


def _resource_snapshot() -> dict[str, Any]:
    raw, unit, size = _rss_raw_and_bytes()
    return {
        "wall_clock_seconds": float(time.perf_counter()),
        "process_cpu_seconds": float(time.process_time()),
        "ru_maxrss_raw": raw,
        "ru_maxrss_raw_unit": unit,
        "ru_maxrss_bytes": size,
    }


def _timed_call(function: Callable[[], T]) -> tuple[T, dict[str, Any]]:
    before = _resource_snapshot()
    value = function()
    after = _resource_snapshot()
    timing = {
        "wall_seconds": float(
            after["wall_clock_seconds"] - before["wall_clock_seconds"]
        ),
        "cpu_seconds": float(
            after["process_cpu_seconds"] - before["process_cpu_seconds"]
        ),
        "ru_maxrss_before_bytes": int(before["ru_maxrss_bytes"]),
        "ru_maxrss_after_bytes": int(after["ru_maxrss_bytes"]),
        "ru_maxrss_delta_bytes_lower_bound": int(
            max(0, after["ru_maxrss_bytes"] - before["ru_maxrss_bytes"])
        ),
        "ru_maxrss_raw_unit": str(after["ru_maxrss_raw_unit"]),
    }
    if (
        not math.isfinite(timing["wall_seconds"])
        or not math.isfinite(timing["cpu_seconds"])
        or timing["wall_seconds"] < 0.0
        or timing["cpu_seconds"] < 0.0
    ):
        raise RuntimeError("timer returned an invalid duration")
    return value, timing


def _mode_payload(mode: BenchmarkMode) -> dict[str, Any]:
    return {
        "name": mode.name,
        "seeds": list(mode.seeds),
        "excluded_warmups_per_seed_and_worker_kind": mode.warmups,
        "timed_repetitions_per_seed_and_worker_kind": mode.repeats,
        "bootstrap_resamples": mode.bootstrap_resamples,
        "bootstrap_seed": mode.bootstrap_seed,
        "require_formal_anchors": mode.require_formal_anchors,
        "require_single_cpu": mode.require_single_cpu,
        "require_blas_one": mode.require_blas_one,
        "snp_chunk_size": mode.snp_chunk_size,
    }


def _mode_from_payload(value: Mapping[str, Any]) -> BenchmarkMode:
    expected = {
        "name",
        "seeds",
        "excluded_warmups_per_seed_and_worker_kind",
        "timed_repetitions_per_seed_and_worker_kind",
        "bootstrap_resamples",
        "bootstrap_seed",
        "require_formal_anchors",
        "require_single_cpu",
        "require_blas_one",
        "snp_chunk_size",
    }
    if set(value) != expected:
        raise ValueError("benchmark mode schema mismatch")
    return BenchmarkMode(
        name=str(value["name"]),
        seeds=tuple(int(seed) for seed in value["seeds"]),
        warmups=int(value["excluded_warmups_per_seed_and_worker_kind"]),
        repeats=int(value["timed_repetitions_per_seed_and_worker_kind"]),
        bootstrap_resamples=int(value["bootstrap_resamples"]),
        bootstrap_seed=int(value["bootstrap_seed"]),
        require_formal_anchors=bool(value["require_formal_anchors"]),
        require_single_cpu=bool(value["require_single_cpu"]),
        require_blas_one=bool(value["require_blas_one"]),
        snp_chunk_size=int(value["snp_chunk_size"]),
    )


def _validate_mode(mode: BenchmarkMode) -> None:
    if mode.name == "formal":
        if mode != FORMAL_MODE:
            raise ValueError("formal runtime protocol differs from its exact lock")
    elif mode.name == "tiny_smoke":
        if mode != SMOKE_MODE:
            raise ValueError("tiny-smoke protocol differs from its exact lock")
    else:
        raise ValueError("unknown benchmark mode")
    if mode.warmups != 1 or mode.repeats < 1:
        raise ValueError("benchmark requires one excluded warmup and timed repeats")


def _unique_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    predicate: Callable[[Mapping[str, Any]], bool],
) -> Mapping[str, Any]:
    matches = [row for row in rows if predicate(row)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {role}, found {len(matches)}")
    return matches[0]


def _reference_case(
    payload: Mapping[str, Any], *, seed: int, width: int
) -> dict[str, Any]:
    split = _unique_row(
        payload["split_receipts"],
        role=f"split receipt for seed {seed}",
        predicate=lambda row: int(row["seed"]) == int(seed),
    )
    selection = _unique_row(
        payload["panel_selection_evidence"],
        role=f"rank8/5pct selection receipt for seed {seed}",
        predicate=lambda row: (
            int(row["seed"]) == int(seed)
            and int(row["n_components"]) == FORMAL_RANK
            and row["panel_label"] == FORMAL_PANEL_LABEL
        ),
    )
    panel_record = _unique_row(
        payload["records"],
        role=f"rank8/5pct metric record for seed {seed}",
        predicate=lambda row: (
            int(row["seed"]) == int(seed)
            and int(row["n_components"]) == FORMAL_RANK
            and row["panel_label"] == FORMAL_PANEL_LABEL
        ),
    )
    all_record = _unique_row(
        payload["records"],
        role=f"rank8/all metric record for seed {seed}",
        predicate=lambda row: (
            int(row["seed"]) == int(seed)
            and int(row["n_components"]) == FORMAL_RANK
            and row["panel_label"] == "all"
        ),
    )
    if (
        int(split["retained_mask_width"]) != int(width)
        or int(selection["selected_mask_width"]) != int(width)
        or selection["requested_panel_fraction"] != FORMAL_PANEL_FRACTION
        or selection["selection_rule"] != "top_training_pca_loading_fraction"
        or selection["selection_used_population_labels"] is not False
        or selection["selection_used_test_rows"] is not False
        or panel_record["selected_original_indices_sha256"]
        != selection["selected_original_indices_sha256"]
        or int(panel_record["selected_snp_count"])
        != int(selection["selected_snp_count"])
    ):
        raise ValueError("formal split/panel evidence is internally inconsistent")
    mask = all_auto._unpack_mask(
        str(selection["selected_mask_packbits_base64"]), int(width)
    )
    selected = np.flatnonzero(mask).astype(np.int64)
    if (
        selected.size != int(selection["selected_snp_count"])
        or array_sha256(selected) != selection["selected_original_indices_sha256"]
    ):
        raise ValueError("formal 5% panel mask hash/count mismatch")
    return {
        "seed": int(seed),
        "train_indices_sha256": str(split["train_indices_sha256"]),
        "test_indices_sha256": str(split["test_indices_sha256"]),
        "train_sample_ids_sha256": str(split["train_sample_ids_sha256"]),
        "test_sample_ids_sha256": str(split["test_sample_ids_sha256"]),
        "test_labels_sha256": str(split["test_labels_sha256"]),
        "retained_snps": int(split["retained_snps"]),
        "selected_snp_count": int(selected.size),
        "selected_original_indices_sha256": array_sha256(selected),
        "selected_mask_packbits_base64": str(
            selection["selected_mask_packbits_base64"]
        ),
        "selected_mask_width": int(width),
        "panel_reference": {
            "balanced_accuracy": float(panel_record["balanced_accuracy"]),
            "macro_f1": float(panel_record["macro_f1"]),
            "accuracy": float(panel_record["accuracy"]),
            "test_predictions_sha256": str(panel_record["test_predictions_sha256"]),
            "selected_snp_count": int(panel_record["selected_snp_count"]),
        },
        "all_reference": {
            "balanced_accuracy": float(all_record["balanced_accuracy"]),
            "macro_f1": float(all_record["macro_f1"]),
            "accuracy": float(all_record["accuracy"]),
            "test_predictions_sha256": str(all_record["test_predictions_sha256"]),
            "selected_snp_count": int(all_record["selected_snp_count"]),
        },
    }


def _validate_reference_payload(
    payload: Mapping[str, Any], mode: BenchmarkMode
) -> None:
    _validate_mode(mode)
    unsigned_reference = dict(payload)
    reference_digest = unsigned_reference.pop("analysis_sha256", None)
    if (
        not _is_sha256(reference_digest)
        or canonical_sha256(unsigned_reference) != reference_digest
    ):
        raise ValueError("reference result analysis SHA is stale or invalid")
    required = {
        "analysis_sha256",
        "config",
        "input_identity",
        "source_identity",
        "split_receipts",
        "panel_selection_evidence",
        "records",
    }
    if not required.issubset(payload):
        raise ValueError("reference result lacks required formal evidence")
    config = payload["config"]
    if (
        tuple(config["seeds"]) != mode.seeds
        or tuple(config["component_grid"]) != (4, 6, 8)
        or tuple(config["panel_fractions"]) != (0.05, 0.10, 0.25, 0.50, None)
        or float(config["test_fraction"]) != 0.2
    ):
        raise ValueError("reference result does not cover the benchmark protocol")
    if mode.require_formal_anchors:
        if (
            payload["analysis_sha256"] != FORMAL_ANALYSIS_SHA256
            or payload["input_identity"]["genotype_npz_sha256"]
            != all_auto.FROZEN_FORMAL_NPZ_SHA256
            or payload["input_identity"]["external_source_manifest_sha256"]
            != all_auto.FROZEN_FORMAL_MANIFEST_SHA256
            or payload["source_identity"]["files"][
                "experiments/genomics_pca_loading_all_autosomes.py"
            ]
            != FORMAL_RUNNER_SHA256
        ):
            raise ValueError("reference result differs from exact formal anchors")
    width = int(payload["input_identity"]["snps"])
    for seed in mode.seeds:
        _reference_case(payload, seed=seed, width=width)


def _load_worker_arrays(
    raw_npy: Path, metadata_npz: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _regular_file(raw_npy, role="verified extracted genotype NPY")
    _regular_file(metadata_npz, role="genotype metadata NPZ")
    raw = np.load(raw_npy, mmap_mode="r", allow_pickle=False)
    with np.load(metadata_npz, allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"], dtype=str)
        labels = np.asarray(archive["population_labels"], dtype=str)
    if (
        raw.ndim != 2
        or raw.shape[0] != labels.size
        or sample_ids.shape != labels.shape
        or not isinstance(raw, np.memmap)
    ):
        raise ValueError("worker input arrays are inconsistent or not memory mapped")
    return raw, sample_ids, labels


def _load_worker_metadata(metadata_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    _regular_file(metadata_npz, role="genotype metadata NPZ")
    with np.load(metadata_npz, allow_pickle=False) as archive:
        sample_ids = np.asarray(archive["sample_ids"], dtype=str)
        labels = np.asarray(archive["population_labels"], dtype=str)
    if sample_ids.shape != labels.shape or labels.ndim != 1:
        raise ValueError("worker metadata arrays are inconsistent")
    return sample_ids, labels


def _split_from_reference(
    labels: np.ndarray,
    sample_ids: np.ndarray,
    case: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    train, test = stratified_split(labels, 0.2, int(case["seed"]))
    train = np.asarray(train, dtype=np.int64)
    test = np.asarray(test, dtype=np.int64)
    observed = {
        "train_indices_sha256": array_sha256(train),
        "test_indices_sha256": array_sha256(test),
        "train_sample_ids_sha256": array_sha256(sample_ids[train]),
        "test_sample_ids_sha256": array_sha256(sample_ids[test]),
        "test_labels_sha256": array_sha256(labels[test]),
    }
    if any(observed[key] != case[key] for key in observed):
        raise ValueError("regenerated worker split differs from formal evidence")
    return train, test


def _selected_mask(case: Mapping[str, Any]) -> np.ndarray:
    mask = all_auto._unpack_mask(
        str(case["selected_mask_packbits_base64"]),
        int(case["selected_mask_width"]),
    )
    selected = np.flatnonzero(mask).astype(np.int64)
    if (
        selected.size != int(case["selected_snp_count"])
        or array_sha256(selected) != case["selected_original_indices_sha256"]
    ):
        raise ValueError("worker panel mask differs from formal evidence")
    return mask


def materialize_compact_panel(
    raw: np.ndarray,
    selected_mask: np.ndarray,
    destination: Path,
    *,
    chunk_size: int,
    hash_file: bool = True,
) -> dict[str, Any]:
    """Write selected columns in original genomic order to a compact NPY mmap."""

    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(f"compact panel already exists: {destination}")
    selected = np.asarray(selected_mask, dtype=bool)
    if raw.ndim != 2 or selected.shape != (raw.shape[1],):
        raise ValueError("compact-panel mask does not match raw genotype width")
    count = int(np.count_nonzero(selected))
    if count < FORMAL_RANK:
        raise ValueError("compact panel is smaller than the frozen rank")
    destination.parent.mkdir(parents=True, exist_ok=True)
    panel = np.lib.format.open_memmap(
        destination,
        mode="w+",
        dtype=raw.dtype,
        shape=(raw.shape[0], count),
    )
    offset = 0
    try:
        for start, stop in all_auto._iter_column_ranges(raw.shape[1], chunk_size):
            local = selected[start:stop]
            local_count = int(np.count_nonzero(local))
            if local_count:
                panel[:, offset : offset + local_count] = np.asarray(
                    raw[:, start:stop]
                )[:, local]
                offset += local_count
        if offset != count:
            raise RuntimeError("compact panel materialization is incomplete")
        panel.flush()
    finally:
        del panel
    observed = np.load(destination, mmap_mode="r", allow_pickle=False)
    if (
        not isinstance(observed, np.memmap)
        or observed.shape != (raw.shape[0], count)
        or observed.dtype != raw.dtype
    ):
        raise RuntimeError("compact panel NPY failed its shape/dtype check")
    panel_array_sha256 = array_sha256(observed) if hash_file else None
    del observed
    return {
        "path": str(destination.resolve()),
        "bytes": int(destination.stat().st_size),
        "sha256": _file_sha256(destination) if hash_file else None,
        "array_sha256": panel_array_sha256,
        "shape": [int(raw.shape[0]), count],
        "dtype": str(raw.dtype.str),
        "selected_snp_count": count,
        "selected_original_indices_sha256": array_sha256(
            np.flatnonzero(selected).astype(np.int64)
        ),
    }


def verify_compact_panel_binding(
    raw: np.ndarray,
    selected_mask: np.ndarray,
    compact_path: Path,
    *,
    chunk_size: int,
) -> None:
    """Verify the compact columns equal the selected raw columns in order."""

    selected = np.flatnonzero(np.asarray(selected_mask, dtype=bool)).astype(np.int64)
    compact = np.load(compact_path, mmap_mode="r", allow_pickle=False)
    if compact.shape != (raw.shape[0], selected.size):
        raise ValueError("compact panel shape does not match selected raw columns")
    for start, stop in all_auto._iter_column_ranges(selected.size, chunk_size):
        raw_columns = np.asarray(raw[:, selected[start:stop]])
        compact_columns = np.asarray(compact[:, start:stop])
        if not np.array_equal(raw_columns, compact_columns, equal_nan=True):
            raise ValueError(
                "compact panel content is not derived from selected raw SNPs"
            )
    del compact


def _public_case(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in case.items()
        if key != "selected_mask_packbits_base64"
    }


def _load_compact_panel(path: Path, receipt: Mapping[str, Any]) -> np.ndarray:
    status = _regular_file(path, role="frozen compact panel")
    if (
        status.st_size != int(receipt["bytes"])
        or _file_sha256(path) != receipt["sha256"]
    ):
        raise ValueError("compact panel file differs from its parent receipt")
    panel = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        list(panel.shape) != list(receipt["shape"])
        or panel.dtype.str != receipt["dtype"]
        or not isinstance(panel, np.memmap)
    ):
        raise ValueError("compact panel array differs from its parent receipt")
    if (
        receipt.get("array_sha256") is not None
        and array_sha256(panel) != receipt["array_sha256"]
    ):
        raise ValueError("compact panel canonical array hash mismatch")
    return panel


def _metadata_receipt(path: Path) -> dict[str, Any]:
    status = _regular_file(path, role="worker metadata NPZ")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "population_labels"}:
            raise ValueError("worker metadata NPZ key closure mismatch")
        sample_ids = np.asarray(archive["sample_ids"], dtype=str)
        labels = np.asarray(archive["population_labels"], dtype=str)
    return {
        "path": str(Path(path).resolve()),
        "bytes": int(status.st_size),
        "sha256": _file_sha256(path),
        "sample_ids_sha256": array_sha256(sample_ids),
        "population_labels_sha256": array_sha256(labels),
        "samples": int(labels.size),
    }


def _raw_npy_receipt(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    status = _regular_file(path, role="verified extracted genotype NPY")
    observed_sha = _file_sha256(path)
    if expected_sha256 is not None and observed_sha != expected_sha256:
        raise ValueError("extracted genotype NPY differs from anchored input")
    raw = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(raw, np.memmap) or raw.ndim != 2:
        raise ValueError("raw genotype NPY must be a two-dimensional mmap")
    receipt = {
        "path": str(Path(path).resolve()),
        "bytes": int(status.st_size),
        "sha256": observed_sha,
        "shape": [int(raw.shape[0]), int(raw.shape[1])],
        "dtype": str(raw.dtype.str),
    }
    del raw
    return receipt


def _reference_file_receipt(path: Path) -> dict[str, Any]:
    status = _regular_file(path, role="sealed reference result JSON")
    return {
        "path": str(Path(path).resolve()),
        "bytes": int(status.st_size),
        "sha256": _file_sha256(path),
    }


def _verify_file_receipt(receipt: Mapping[str, Any], *, role: str) -> Path:
    expected = {"path", "bytes", "sha256"}
    if not expected.issubset(receipt) or not _is_sha256(receipt["sha256"]):
        raise ValueError(f"invalid {role} receipt")
    path = Path(str(receipt["path"]))
    status = _regular_file(path, role=role)
    if (
        status.st_size != int(receipt["bytes"])
        or _file_sha256(path) != receipt["sha256"]
    ):
        raise ValueError(f"{role} differs from its job-contract receipt")
    return path


def _worker_schedule(mode: BenchmarkMode) -> list[dict[str, Any]]:
    _validate_mode(mode)
    jobs: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(mode.seeds):
        phases = [("warmup", -1)] + [
            ("timed", repetition) for repetition in range(mode.repeats)
        ]
        for phase, repetition in phases:
            parity = seed_index + (0 if repetition < 0 else repetition)
            order = (
                ("discovery", "deployment")
                if parity % 2 == 0
                else ("deployment", "discovery")
            )
            for position, kind in enumerate(order):
                jobs.append(
                    {
                        "job_id": f"{mode.name}-{seed}-{phase}-{repetition}-{kind}",
                        "worker_kind": kind,
                        "seed": int(seed),
                        "phase": phase,
                        "repetition": int(repetition),
                        "timing_position": int(position),
                    }
                )
    return jobs


def _build_job_contract(
    *,
    mode: BenchmarkMode,
    source_identity: Mapping[str, Any],
    reference_receipt: Mapping[str, Any],
    raw_receipt: Mapping[str, Any],
    metadata_receipt: Mapping[str, Any],
    compact_receipts: Mapping[int, Mapping[str, Any]],
    formal_package: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _validate_mode(mode)
    if set(compact_receipts) != set(mode.seeds):
        raise ValueError("job contract needs one compact panel receipt per seed")
    contract: dict[str, Any] = {
        "schema_version": 1,
        "mode": _mode_payload(mode),
        "nonce": uuid.uuid4().hex,
        "source_identity": dict(source_identity),
        "reference_result": dict(reference_receipt),
        "raw_genotypes_npy": dict(raw_receipt),
        "metadata_npz": dict(metadata_receipt),
        "compact_panels_by_seed": {
            str(seed): dict(compact_receipts[seed]) for seed in mode.seeds
        },
        "formal_package": None if formal_package is None else dict(formal_package),
        "jobs": _worker_schedule(mode),
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def _validate_job_contract(contract: Mapping[str, Any]) -> BenchmarkMode:
    required = {
        "schema_version",
        "mode",
        "nonce",
        "source_identity",
        "reference_result",
        "raw_genotypes_npy",
        "metadata_npz",
        "compact_panels_by_seed",
        "formal_package",
        "jobs",
        "contract_sha256",
    }
    if set(contract) != required or contract["schema_version"] != 1:
        raise ValueError("worker job-contract schema mismatch")
    unsigned = dict(contract)
    digest = unsigned.pop("contract_sha256")
    if not _is_sha256(digest) or canonical_sha256(unsigned) != digest:
        raise ValueError("worker job-contract hash mismatch")
    mode = _mode_from_payload(contract["mode"])
    _validate_mode(mode)
    if contract["jobs"] != _worker_schedule(mode):
        raise ValueError("worker job schedule is not the exact locked schedule")
    if set(contract["compact_panels_by_seed"]) != {str(seed) for seed in mode.seeds}:
        raise ValueError("worker job contract lacks an exact compact-panel cover")
    if mode.require_formal_anchors:
        package = contract["formal_package"]
        if (
            not isinstance(package, Mapping)
            or package.get("analysis_sha256") != FORMAL_ANALYSIS_SHA256
            or package.get("package_anchor_sha256") != FORMAL_PACKAGE_ANCHOR_SHA256
            or package.get("package_self_consistency_verified") is not True
            or package.get("protocol_seed_cover_complete") is not True
        ):
            raise ValueError("formal worker contract lacks sealed-package validation")
    elif contract["formal_package"] is not None:
        raise ValueError("tiny smoke must not pretend to have a formal package")
    return mode


def _load_authorized_job(
    contract_path: Path, job_id: str
) -> tuple[Mapping[str, Any], Mapping[str, Any], BenchmarkMode]:
    contract = _strict_json_load(contract_path)
    if not isinstance(contract, Mapping):
        raise ValueError("worker job contract must be a JSON object")
    mode = _validate_job_contract(contract)
    job = _unique_row(
        contract["jobs"],
        role=f"authorized job {job_id}",
        predicate=lambda row: row["job_id"] == job_id,
    )
    return contract, job, mode


def _verify_worker_inputs(
    contract: Mapping[str, Any],
    *,
    seed: int,
    worker_kind: str,
) -> tuple[
    np.ndarray | None,
    np.ndarray,
    np.ndarray,
    Mapping[str, Any],
    Mapping[str, Any],
]:
    if worker_kind not in {"discovery", "deployment"}:
        raise ValueError("unknown worker kind")
    current_source = _source_identity(Path(__file__).resolve().parents[1])
    if current_source != contract["source_identity"]:
        raise ValueError("current worker source differs from the job contract")
    metadata_path = _verify_file_receipt(
        contract["metadata_npz"], role="worker metadata NPZ"
    )
    reference_path = _verify_file_receipt(
        contract["reference_result"], role="sealed reference result JSON"
    )
    sample_ids, labels = _load_worker_metadata(metadata_path)
    raw_receipt = contract["raw_genotypes_npy"]
    metadata_receipt = contract["metadata_npz"]
    if (
        labels.size != int(metadata_receipt["samples"])
        or array_sha256(sample_ids) != metadata_receipt["sample_ids_sha256"]
        or array_sha256(labels) != metadata_receipt["population_labels_sha256"]
    ):
        raise ValueError("worker arrays differ from exact contract identities")
    raw: np.ndarray | None = None
    if worker_kind == "discovery":
        raw_path = _verify_file_receipt(raw_receipt, role="raw genotype NPY")
        raw = np.load(raw_path, mmap_mode="r", allow_pickle=False)
        if (
            not isinstance(raw, np.memmap)
            or list(raw.shape) != list(raw_receipt["shape"])
            or raw.dtype.str != raw_receipt["dtype"]
        ):
            raise ValueError("raw genotype NPY differs from exact contract identity")
    payload = _strict_json_load(reference_path)
    mode = _mode_from_payload(contract["mode"])
    _validate_reference_payload(payload, mode)
    case = _reference_case(payload, seed=seed, width=int(raw_receipt["shape"][1]))
    compact = contract["compact_panels_by_seed"][str(seed)]
    if (
        compact["selected_original_indices_sha256"]
        != case["selected_original_indices_sha256"]
        or compact["selected_snp_count"] != case["selected_snp_count"]
    ):
        raise ValueError("compact panel is not bound to the formal split mask")
    return raw, sample_ids, labels, case, compact


def _score_fit(
    fit: all_auto.ChunkedPCAFit,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    predictions, classes, _ = grid._nearest_centroid(
        fit.train_scores[:, :FORMAL_RANK],
        train_labels,
        fit.test_scores[:, :FORMAL_RANK],
    )
    metrics = grid._classification_metrics(test_labels, predictions, classes)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "test_predictions_sha256": array_sha256(predictions),
        "test_predictions_dtype": str(predictions.dtype),
        "sample_gram_sha256": fit.sample_gram_sha256,
        "singular_values": [float(value) for value in fit.singular_values],
    }


def _verify_science(
    observed: Mapping[str, Any], reference: Mapping[str, Any], *, role: str
) -> None:
    if (
        observed["test_predictions_sha256"] != reference["test_predictions_sha256"]
        or observed["accuracy"] != reference["accuracy"]
        or observed["balanced_accuracy"] != reference["balanced_accuracy"]
        or observed["macro_f1"] != reference["macro_f1"]
    ):
        raise RuntimeError(f"{role} predictions or quality differ from formal evidence")


def _fit_and_score(
    matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    *,
    chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def operation() -> tuple[all_auto.ChunkedPCAFit, dict[str, Any]]:
        fit = all_auto.fit_chunked_top_pca(
            matrix,
            train,
            test,
            n_components=FORMAL_RANK,
            chunk_size=chunk_size,
        )
        return fit, _score_fit(fit, labels[train], labels[test])

    (fit, science), timing = _timed_call(operation)
    return {"fit": fit, "science": science}, timing


def _rank_select_numeric(
    full_fit: all_auto.ChunkedPCAFit,
    selected_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ranking, importance = grid.rank_by_pca_loadings(full_fit.loadings, FORMAL_RANK)
    ordered_original = full_fit.retained_original[ranking]
    selected_original = np.sort(ordered_original[: int(selected_count)]).astype(
        np.int64
    )
    return selected_original, ranking, importance


def _verify_selected_indices(
    selected_original: np.ndarray,
    ranking: np.ndarray,
    importance: np.ndarray,
    case: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    selected_mask = np.zeros(int(case["selected_mask_width"]), dtype=bool)
    selected_mask[selected_original] = True
    formal_mask = _selected_mask(case)
    if not np.array_equal(selected_mask, formal_mask):
        raise RuntimeError("timed loading screen did not recreate the sealed 5% mask")
    receipt = {
        "selected_snp_count": int(selected_original.size),
        "selected_original_indices_sha256": array_sha256(selected_original),
        "loading_importance_sha256": array_sha256(importance),
        "ranking_sha256": array_sha256(ranking),
        "selection_rule": "descending_sum_abs_training_rank8_pca_loadings",
        "population_labels_used": False,
        "test_rows_used": False,
    }
    return selected_mask, receipt


def _total_timing(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "wall_seconds": float(
            after["wall_clock_seconds"] - before["wall_clock_seconds"]
        ),
        "cpu_seconds": float(
            after["process_cpu_seconds"] - before["process_cpu_seconds"]
        ),
        "ru_maxrss_before_bytes": int(before["ru_maxrss_bytes"]),
        "ru_maxrss_after_bytes": int(after["ru_maxrss_bytes"]),
        "ru_maxrss_delta_bytes_lower_bound": int(
            max(0, after["ru_maxrss_bytes"] - before["ru_maxrss_bytes"])
        ),
        "ru_maxrss_raw_unit": str(after["ru_maxrss_raw_unit"]),
        "rss_claim_scope": (
            "fresh_child_absolute_ru_maxrss_after_is_headline; delta_is_only_a_"
            "lower_bound_because_ru_maxrss_is_a_process_peak"
        ),
    }


def run_discovery_worker(
    *,
    contract_path: Path,
    job_id: str,
    scratch_root: Path,
) -> dict[str, Any]:
    contract, job, mode = _load_authorized_job(contract_path, job_id)
    if job["worker_kind"] != "discovery":
        raise ValueError("discovery worker received a non-discovery job")
    runtime = _require_runtime_contract(mode)
    raw, sample_ids, labels, case, canonical_compact = _verify_worker_inputs(
        contract, seed=int(job["seed"]), worker_kind="discovery"
    )
    if raw is None:  # Defensive type narrowing; discovery always binds raw NPY.
        raise RuntimeError("discovery worker did not receive raw genotypes")
    train, test = _split_from_reference(labels, sample_ids, case)
    scratch = Path(scratch_root).resolve()
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("worker scratch root must be an existing real directory")

    total_before = _resource_snapshot()
    full_result, full_timing = _fit_and_score(
        raw,
        train,
        test,
        labels,
        chunk_size=mode.snp_chunk_size,
    )
    full_fit = full_result["fit"]
    full_science = full_result["science"]
    _verify_science(full_science, case["all_reference"], role="all-SNP reference")

    (selected_original, ranking, importance), selection_timing = _timed_call(
        lambda: _rank_select_numeric(full_fit, int(case["selected_snp_count"]))
    )
    selected_mask, selection_receipt = _verify_selected_indices(
        selected_original, ranking, importance, case
    )
    with tempfile.TemporaryDirectory(
        prefix=f"rspca-runtime-discovery-{job['seed']}-",
        dir=str(scratch),
    ) as temporary_name:
        panel_path = Path(temporary_name) / "compact_panel.npy"
        materialization_receipt, materialization_timing = _timed_call(
            lambda: materialize_compact_panel(
                raw,
                selected_mask,
                panel_path,
                chunk_size=mode.snp_chunk_size,
                hash_file=False,
            )
        )
        panel = np.load(panel_path, mmap_mode="r", allow_pickle=False)
        panel_result, panel_timing = _fit_and_score(
            panel,
            train,
            test,
            labels,
            chunk_size=mode.snp_chunk_size,
        )
        panel_science = panel_result["science"]
        del panel_result["fit"]
        del panel
        total_after = _resource_snapshot()

        # Integrity work is deliberately outside the operational timer.  It
        # proves that the timed copy+flush produced the same already-frozen
        # compact array consumed by deployment workers.
        materialization_receipt["sha256"] = _file_sha256(panel_path)
        materialized = np.load(panel_path, mmap_mode="r", allow_pickle=False)
        materialization_receipt["array_sha256"] = array_sha256(materialized)
        del materialized
        for key in (
            "bytes",
            "sha256",
            "array_sha256",
            "shape",
            "dtype",
            "selected_snp_count",
            "selected_original_indices_sha256",
        ):
            if materialization_receipt[key] != canonical_compact[key]:
                raise RuntimeError(
                    f"timed compact-panel materialization mismatch for {key}"
                )

    _verify_science(panel_science, case["panel_reference"], role="5% panel")
    del full_result["fit"]
    stage_wall_sum = float(
        full_timing["wall_seconds"]
        + selection_timing["wall_seconds"]
        + materialization_timing["wall_seconds"]
        + panel_timing["wall_seconds"]
    )
    stage_cpu_sum = float(
        full_timing["cpu_seconds"]
        + selection_timing["cpu_seconds"]
        + materialization_timing["cpu_seconds"]
        + panel_timing["cpu_seconds"]
    )
    total = _total_timing(total_before, total_after)
    total.update(
        {
            "stage_wall_sum_seconds": stage_wall_sum,
            "stage_cpu_sum_seconds": stage_cpu_sum,
            "unattributed_wall_overhead_seconds": float(
                total["wall_seconds"] - stage_wall_sum
            ),
            "unattributed_cpu_overhead_seconds": float(
                total["cpu_seconds"] - stage_cpu_sum
            ),
            "operational_scope": (
                "includes full fit+score, rank/select, compact NPY copy+flush, "
                "and compact-panel fit+score; excludes post-timing file/array "
                "hash and shape integrity validation; direct total also includes "
                "small Python orchestration and sealed-mask checks between stages"
            ),
        }
    )
    science = {
        "case": _public_case(case),
        "all_observed": full_science,
        "panel_observed": panel_science,
        "selection": selection_receipt,
        "materialized_compact_panel": {
            key: materialization_receipt[key]
            for key in (
                "bytes",
                "sha256",
                "array_sha256",
                "shape",
                "dtype",
                "selected_snp_count",
                "selected_original_indices_sha256",
            )
        },
    }
    output = {
        "schema_version": 1,
        "job": dict(job),
        "worker_pid": os.getpid(),
        "fresh_child_worker": True,
        "runtime_environment": runtime,
        "stages": {
            "all_fit_projection_and_score": full_timing,
            "rank_and_select": selection_timing,
            "compact_panel_copy_and_flush": materialization_timing,
            "compact_panel_fit_projection_and_score": panel_timing,
        },
        "total": total,
        "science": science,
        "science_sha256": canonical_sha256(science),
        "job_contract_sha256": contract["contract_sha256"],
    }
    validate_worker_output(
        output,
        expected_job=job,
        mode=mode,
        expected_contract_sha256=str(contract["contract_sha256"]),
    )
    return output


def run_deployment_worker(
    *,
    contract_path: Path,
    job_id: str,
) -> dict[str, Any]:
    contract, job, mode = _load_authorized_job(contract_path, job_id)
    if job["worker_kind"] != "deployment":
        raise ValueError("deployment worker received a non-deployment job")
    runtime = _require_runtime_contract(mode)
    _, sample_ids, labels, case, compact_receipt = _verify_worker_inputs(
        contract, seed=int(job["seed"]), worker_kind="deployment"
    )
    train, test = _split_from_reference(labels, sample_ids, case)
    panel = _load_compact_panel(Path(compact_receipt["path"]), compact_receipt)
    before = _resource_snapshot()
    panel_result, timing = _fit_and_score(
        panel,
        train,
        test,
        labels,
        chunk_size=mode.snp_chunk_size,
    )
    after = _resource_snapshot()
    science = panel_result["science"]
    _verify_science(science, case["panel_reference"], role="deployment 5% panel")
    del panel_result["fit"]
    del panel
    total = _total_timing(before, after)
    total.update(
        {
            "fit_and_score_wall_seconds": float(timing["wall_seconds"]),
            "fit_and_score_cpu_seconds": float(timing["cpu_seconds"]),
            "operational_scope": (
                "only rank8 fit, held-out projection, and score on an already-"
                "frozen split-specific 5% compact panel; excludes discovery, "
                "materialization, input hashing, and child startup"
            ),
        }
    )
    science_block = {
        "case": _public_case(case),
        "panel_observed": science,
        "compact_panel_receipt": {
            key: compact_receipt[key]
            for key in (
                "bytes",
                "sha256",
                "array_sha256",
                "shape",
                "dtype",
                "selected_snp_count",
                "selected_original_indices_sha256",
            )
        },
    }
    output = {
        "schema_version": 1,
        "job": dict(job),
        "worker_pid": os.getpid(),
        "fresh_child_worker": True,
        "runtime_environment": runtime,
        "stages": {"compact_panel_fit_projection_and_score": timing},
        "total": total,
        "science": science_block,
        "science_sha256": canonical_sha256(science_block),
        "job_contract_sha256": contract["contract_sha256"],
    }
    validate_worker_output(
        output,
        expected_job=job,
        mode=mode,
        expected_contract_sha256=str(contract["contract_sha256"]),
    )
    return output


def _validate_timing(value: Any, *, role: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{role} timing must be an object")
    for key in ("wall_seconds", "cpu_seconds"):
        number = value.get(key)
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError(f"invalid {role}.{key}")
    after = value.get("ru_maxrss_after_bytes")
    if isinstance(after, bool) or not isinstance(after, int) or after < 0:
        raise ValueError(f"invalid {role}.ru_maxrss_after_bytes")


def validate_worker_output(
    output: Mapping[str, Any],
    *,
    expected_job: Mapping[str, Any],
    mode: BenchmarkMode,
    expected_contract_sha256: str,
) -> None:
    base_expected = {
        "schema_version",
        "job",
        "worker_pid",
        "fresh_child_worker",
        "runtime_environment",
        "stages",
        "total",
        "science",
        "science_sha256",
        "job_contract_sha256",
    }
    expected = set(base_expected)
    if "subprocess_stderr_receipt" in output:
        expected.add("subprocess_stderr_receipt")
    if (
        set(output) != expected
        or output["schema_version"] != 1
        or output["job"] != expected_job
        or output["fresh_child_worker"] is not True
        or not isinstance(output["worker_pid"], int)
        or output["worker_pid"] <= 0
        or output["science_sha256"] != canonical_sha256(output["science"])
        or output["job_contract_sha256"] != expected_contract_sha256
    ):
        raise ValueError("worker output schema, job, or science hash mismatch")
    if "subprocess_stderr_receipt" in output:
        stderr_receipt = output["subprocess_stderr_receipt"]
        if (
            not isinstance(stderr_receipt, Mapping)
            or set(stderr_receipt) != {"bytes", "sha256", "text_tail"}
            or not isinstance(stderr_receipt["bytes"], int)
            or stderr_receipt["bytes"] < 0
            or not _is_sha256(stderr_receipt["sha256"])
            or not isinstance(stderr_receipt["text_tail"], str)
        ):
            raise ValueError("worker stderr receipt is invalid")
        if stderr_receipt["bytes"] == 0 and (
            stderr_receipt["sha256"] != hashlib.sha256(b"").hexdigest()
            or stderr_receipt["text_tail"] != ""
        ):
            raise ValueError("empty worker stderr receipt is inconsistent")
        if stderr_receipt["bytes"] > 0 and stderr_receipt["text_tail"] == "":
            raise ValueError("nonempty worker stderr receipt lacks captured text")
        if mode.require_formal_anchors and stderr_receipt["bytes"] != 0:
            raise ValueError("formal worker stderr must remain empty")
    kind = expected_job["worker_kind"]
    expected_stages = (
        {
            "all_fit_projection_and_score",
            "rank_and_select",
            "compact_panel_copy_and_flush",
            "compact_panel_fit_projection_and_score",
        }
        if kind == "discovery"
        else {"compact_panel_fit_projection_and_score"}
    )
    if set(output["stages"]) != expected_stages:
        raise ValueError("worker stage closure mismatch")
    for name, timing in output["stages"].items():
        _validate_timing(timing, role=name)
    _validate_timing(output["total"], role="total")
    runtime = output["runtime_environment"]
    if mode.require_single_cpu and runtime.get("affinity_count") != 1:
        raise ValueError("formal worker did not retain single-CPU affinity")
    if mode.require_blas_one:
        all_auto._validate_blas_runtime_identity(
            runtime.get("blas_runtime_identity"), require_single_thread=True
        )
    case = output["science"].get("case")
    if not isinstance(case, Mapping) or int(case.get("seed", -1)) != int(
        expected_job["seed"]
    ):
        raise ValueError("worker science is not bound to its split seed")
    # Unit-level smoke probes may exercise only the exact transport/schema
    # envelope with a seed-only case.  Real workers embed the complete
    # reference case.  Any deviation from that exact seed-only probe (and
    # always formal mode) requires the complete evidence closure and the exact
    # scientific replay checks below.
    evidence_case_keys = {
        "all_reference",
        "panel_reference",
        "selected_original_indices_sha256",
        "selected_snp_count",
    }
    probe_science_keys = {"case", "panel_observed"}
    if kind == "discovery":
        probe_science_keys.add("all_observed")
    exact_seed_only_schema_probe = (
        not mode.require_formal_anchors
        and set(case) == {"seed"}
        and set(output["science"]) == probe_science_keys
    )
    if exact_seed_only_schema_probe:
        return
    missing_case_keys = evidence_case_keys.difference(case)
    if missing_case_keys:
        raise ValueError(
            "worker science evidence case closure mismatch: "
            + ", ".join(sorted(missing_case_keys))
        )
    if kind == "discovery":
        _verify_science(
            output["science"]["all_observed"],
            case["all_reference"],
            role="saved all-SNP worker",
        )
        _verify_science(
            output["science"]["panel_observed"],
            case["panel_reference"],
            role="saved discovery-panel worker",
        )
        selection = output["science"]["selection"]
        materialized = output["science"]["materialized_compact_panel"]
        if (
            selection.get("selected_original_indices_sha256")
            != case["selected_original_indices_sha256"]
            or int(selection.get("selected_snp_count", -1))
            != int(case["selected_snp_count"])
            or materialized.get("selected_original_indices_sha256")
            != case["selected_original_indices_sha256"]
            or int(materialized.get("selected_snp_count", -1))
            != int(case["selected_snp_count"])
        ):
            raise ValueError(
                "saved discovery selection/materialization binding mismatch"
            )
    else:
        _verify_science(
            output["science"]["panel_observed"],
            case["panel_reference"],
            role="saved deployment-panel worker",
        )
        compact = output["science"]["compact_panel_receipt"]
        if compact.get("selected_original_indices_sha256") != case[
            "selected_original_indices_sha256"
        ] or int(compact.get("selected_snp_count", -1)) != int(
            case["selected_snp_count"]
        ):
            raise ValueError("saved deployment compact-panel binding mismatch")


def _execute_worker_job(
    *,
    contract_path: Path,
    job: Mapping[str, Any],
    scratch_root: Path,
    mode: BenchmarkMode,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "internal-worker",
        "--job-contract",
        str(Path(contract_path).resolve()),
        "--job-id",
        str(job["job_id"]),
        "--scratch-root",
        str(Path(scratch_root).resolve()),
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    process_pid = int(process.pid)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        message = stderr.decode("utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"fresh child worker failed for {job['job_id']}: {message}")
    if stderr and mode.require_formal_anchors:
        raise RuntimeError(
            f"fresh child worker wrote unexpected stderr for {job['job_id']}: "
            f"{stderr.decode('utf-8', errors='replace')[-2000:]}"
        )
    output = _strict_json_decode(stdout, role=f"worker stdout {job['job_id']}")
    if not isinstance(output, Mapping):
        raise ValueError("worker stdout must be a JSON object")
    contract = _strict_json_load(contract_path)
    validate_worker_output(
        output,
        expected_job=job,
        mode=mode,
        expected_contract_sha256=str(contract["contract_sha256"]),
    )
    if (
        output["worker_pid"] != process_pid
        or output["runtime_environment"]["pid"] != process_pid
    ):
        raise RuntimeError("Popen PID, worker PID, and runtime PID differ")
    output["subprocess_stderr_receipt"] = {
        "bytes": len(stderr),
        "sha256": hashlib.sha256(stderr).hexdigest(),
        "text_tail": stderr.decode("utf-8", errors="replace")[-2000:],
    }
    validate_worker_output(
        output,
        expected_job=job,
        mode=mode,
        expected_contract_sha256=str(contract["contract_sha256"]),
    )
    return dict(output)


def _paired_record(
    discovery: Mapping[str, Any], deployment: Mapping[str, Any]
) -> dict[str, Any]:
    left = discovery["job"]
    right = deployment["job"]
    for key in ("seed", "phase", "repetition"):
        if left[key] != right[key]:
            raise ValueError("discovery/deployment worker pairing mismatch")
    if left["timing_position"] == right["timing_position"]:
        raise ValueError("paired worker positions must differ")
    discovery_panel = discovery["science"]["panel_observed"]
    deployment_panel = deployment["science"]["panel_observed"]
    if discovery_panel != deployment_panel:
        raise ValueError("paired panel workers changed their scientific output")
    all_timing = discovery["stages"]["all_fit_projection_and_score"]
    deployment_timing = deployment["total"]
    discovery_timing = discovery["total"]
    all_wall = float(all_timing["wall_seconds"])
    deployment_wall = float(deployment_timing["wall_seconds"])
    discovery_wall = float(discovery_timing["wall_seconds"])
    if min(all_wall, deployment_wall, discovery_wall) <= 0.0:
        raise ValueError("headline paired wall times must be strictly positive")
    case = discovery["science"]["case"]
    return {
        "seed": int(left["seed"]),
        "repetition": int(left["repetition"]),
        "execution_order": (
            "discovery_then_deployment"
            if int(left["timing_position"]) == 0
            else "deployment_then_discovery"
        ),
        "discovery_worker_pid": int(discovery["worker_pid"]),
        "deployment_worker_pid": int(deployment["worker_pid"]),
        "all_reference_wall_seconds": all_wall,
        "all_reference_cpu_seconds": float(all_timing["cpu_seconds"]),
        "all_reference_ru_maxrss_after_bytes": int(all_timing["ru_maxrss_after_bytes"]),
        "frozen_panel_deployment_wall_seconds": deployment_wall,
        "frozen_panel_deployment_cpu_seconds": float(deployment_timing["cpu_seconds"]),
        "frozen_panel_deployment_ru_maxrss_after_bytes": int(
            deployment_timing["ru_maxrss_after_bytes"]
        ),
        "prepared_input_discovery_wall_seconds": discovery_wall,
        "prepared_input_discovery_cpu_seconds": float(discovery_timing["cpu_seconds"]),
        "prepared_input_discovery_ru_maxrss_after_bytes": int(
            discovery_timing["ru_maxrss_after_bytes"]
        ),
        "deployment_speedup_vs_all": float(all_wall / deployment_wall),
        "deployment_runtime_reduction_fraction_vs_all": float(
            1.0 - deployment_wall / all_wall
        ),
        "discovery_overhead_fraction_vs_all": float(discovery_wall / all_wall - 1.0),
        "discovery_stage_wall_seconds": {
            name: float(timing["wall_seconds"])
            for name, timing in discovery["stages"].items()
        },
        "selected_snp_count": int(case["selected_snp_count"]),
        "all_snp_count": int(case["retained_snps"]),
        "selected_original_indices_sha256": str(
            case["selected_original_indices_sha256"]
        ),
        "all_test_predictions_sha256": str(
            discovery["science"]["all_observed"]["test_predictions_sha256"]
        ),
        "panel_test_predictions_sha256": str(
            discovery_panel["test_predictions_sha256"]
        ),
        "all_balanced_accuracy": float(
            discovery["science"]["all_observed"]["balanced_accuracy"]
        ),
        "panel_balanced_accuracy": float(discovery_panel["balanced_accuracy"]),
        "all_macro_f1": float(discovery["science"]["all_observed"]["macro_f1"]),
        "panel_macro_f1": float(discovery_panel["macro_f1"]),
        "discovery_science_sha256": str(discovery["science_sha256"]),
        "deployment_science_sha256": str(deployment["science_sha256"]),
        "job_contract_sha256": str(discovery["job_contract_sha256"]),
    }


def _per_seed_medians(
    records: Sequence[Mapping[str, Any]], mode: BenchmarkMode
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    timing_fields = (
        "all_reference_wall_seconds",
        "all_reference_cpu_seconds",
        "all_reference_ru_maxrss_after_bytes",
        "frozen_panel_deployment_wall_seconds",
        "frozen_panel_deployment_cpu_seconds",
        "frozen_panel_deployment_ru_maxrss_after_bytes",
        "prepared_input_discovery_wall_seconds",
        "prepared_input_discovery_cpu_seconds",
        "prepared_input_discovery_ru_maxrss_after_bytes",
    )
    for seed in mode.seeds:
        rows = sorted(
            (row for row in records if int(row["seed"]) == seed),
            key=lambda row: int(row["repetition"]),
        )
        if len(rows) != mode.repeats or [row["repetition"] for row in rows] != list(
            range(mode.repeats)
        ):
            raise ValueError("timed records do not exactly cover repetitions per seed")
        medians = {
            field: float(np.median([float(row[field]) for row in rows]))
            for field in timing_fields
        }
        all_wall = medians["all_reference_wall_seconds"]
        deployment_wall = medians["frozen_panel_deployment_wall_seconds"]
        discovery_wall = medians["prepared_input_discovery_wall_seconds"]
        output.append(
            {
                "seed": int(seed),
                "timed_repetitions": mode.repeats,
                **medians,
                "deployment_speedup_vs_all_ratio_of_medians": float(
                    all_wall / deployment_wall
                ),
                "deployment_runtime_reduction_fraction_vs_all_ratio_of_medians": float(
                    1.0 - deployment_wall / all_wall
                ),
                "discovery_overhead_fraction_vs_all_ratio_of_medians": float(
                    discovery_wall / all_wall - 1.0
                ),
            }
        )
    return output


def _paired_ratio_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    mode: BenchmarkMode,
    estimand: str,
) -> dict[str, Any]:
    if (
        numerator.shape != denominator.shape
        or numerator.shape != (len(mode.seeds),)
        or np.any(numerator <= 0.0)
        or np.any(denominator <= 0.0)
    ):
        raise ValueError("paired bootstrap requires positive per-seed median vectors")
    rng = np.random.default_rng(mode.bootstrap_seed)
    indices = rng.integers(
        0,
        numerator.size,
        size=(mode.bootstrap_resamples, numerator.size),
        endpoint=False,
    )
    ratios = np.mean(numerator[indices], axis=1) / np.mean(denominator[indices], axis=1)
    if estimand == "deployment_reduction_fraction":
        draws = 1.0 - ratios
        estimate = 1.0 - float(np.mean(numerator) / np.mean(denominator))
    elif estimand == "discovery_overhead_fraction":
        draws = ratios - 1.0
        estimate = float(np.mean(numerator) / np.mean(denominator) - 1.0)
    elif estimand == "deployment_speedup":
        draws = 1.0 / ratios
        estimate = float(np.mean(denominator) / np.mean(numerator))
    else:
        raise ValueError("unknown paired ratio bootstrap estimand")
    low, high = np.quantile(draws, (0.025, 0.975))
    return {
        "estimand": estimand,
        "estimate": float(estimate),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": int(mode.bootstrap_resamples),
        "seed": int(mode.bootstrap_seed),
        "unit": "registered_outer_split_seed_after_per_seed_median",
        "interval": "paired_percentile_95",
        "ratio_policy": "ratio_recomputed_inside_every_seed-resample",
    }


def summarize_records(
    records: Sequence[Mapping[str, Any]], mode: BenchmarkMode
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    medians = _per_seed_medians(records, mode)
    all_wall = np.asarray(
        [row["all_reference_wall_seconds"] for row in medians], dtype=float
    )
    deployment_wall = np.asarray(
        [row["frozen_panel_deployment_wall_seconds"] for row in medians],
        dtype=float,
    )
    discovery_wall = np.asarray(
        [row["prepared_input_discovery_wall_seconds"] for row in medians], dtype=float
    )
    all_cpu = np.asarray(
        [row["all_reference_cpu_seconds"] for row in medians], dtype=float
    )
    deployment_cpu = np.asarray(
        [row["frozen_panel_deployment_cpu_seconds"] for row in medians], dtype=float
    )
    discovery_cpu = np.asarray(
        [row["prepared_input_discovery_cpu_seconds"] for row in medians], dtype=float
    )
    all_rss = np.asarray(
        [row["all_reference_ru_maxrss_after_bytes"] for row in medians], dtype=float
    )
    deployment_rss = np.asarray(
        [row["frozen_panel_deployment_ru_maxrss_after_bytes"] for row in medians],
        dtype=float,
    )
    discovery_rss = np.asarray(
        [row["prepared_input_discovery_ru_maxrss_after_bytes"] for row in medians],
        dtype=float,
    )
    aggregate = {
        "seed_count": len(mode.seeds),
        "per_seed_statistic": (
            "median_of_three_fresh_child_invocations"
            if mode.repeats == 3
            else f"median_of_{mode.repeats}_tiny_smoke_invocations"
        ),
        "all_reference_mean_of_seed_medians_seconds": float(np.mean(all_wall)),
        "all_reference_median_of_seed_medians_seconds": float(np.median(all_wall)),
        "frozen_panel_deployment_mean_of_seed_medians_seconds": float(
            np.mean(deployment_wall)
        ),
        "frozen_panel_deployment_median_of_seed_medians_seconds": float(
            np.median(deployment_wall)
        ),
        "prepared_input_discovery_mean_of_seed_medians_seconds": float(
            np.mean(discovery_wall)
        ),
        "prepared_input_discovery_median_of_seed_medians_seconds": float(
            np.median(discovery_wall)
        ),
        "all_reference_cpu_mean_of_seed_medians_seconds": float(np.mean(all_cpu)),
        "all_reference_cpu_median_of_seed_medians_seconds": float(np.median(all_cpu)),
        "frozen_panel_deployment_cpu_mean_of_seed_medians_seconds": float(
            np.mean(deployment_cpu)
        ),
        "frozen_panel_deployment_cpu_median_of_seed_medians_seconds": float(
            np.median(deployment_cpu)
        ),
        "prepared_input_discovery_cpu_mean_of_seed_medians_seconds": float(
            np.mean(discovery_cpu)
        ),
        "prepared_input_discovery_cpu_median_of_seed_medians_seconds": float(
            np.median(discovery_cpu)
        ),
        "all_reference_absolute_peak_rss_mean_of_seed_medians_bytes": float(
            np.mean(all_rss)
        ),
        "all_reference_absolute_peak_rss_median_of_seed_medians_bytes": float(
            np.median(all_rss)
        ),
        "frozen_panel_deployment_absolute_peak_rss_mean_of_seed_medians_bytes": float(
            np.mean(deployment_rss)
        ),
        "frozen_panel_deployment_absolute_peak_rss_median_of_seed_medians_bytes": float(
            np.median(deployment_rss)
        ),
        "prepared_input_discovery_absolute_peak_rss_mean_of_seed_medians_bytes": float(
            np.mean(discovery_rss)
        ),
        "prepared_input_discovery_absolute_peak_rss_median_of_seed_medians_bytes": float(
            np.median(discovery_rss)
        ),
        "rss_scope": (
            "descriptive fresh-child absolute ru_maxrss_after; stage deltas are "
            "lower bounds and are not used for claims"
        ),
        "deployment_reduction": _paired_ratio_bootstrap(
            deployment_wall,
            all_wall,
            mode=mode,
            estimand="deployment_reduction_fraction",
        ),
        "deployment_speedup": _paired_ratio_bootstrap(
            deployment_wall,
            all_wall,
            mode=mode,
            estimand="deployment_speedup",
        ),
        "discovery_overhead": _paired_ratio_bootstrap(
            discovery_wall,
            all_wall,
            mode=mode,
            estimand="discovery_overhead_fraction",
        ),
        "no_runtime_winner_or_gate_selected": True,
    }
    return medians, aggregate


def _without_path(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in receipt.items() if key != "path"}


def execute_benchmark_workers(
    *,
    mode: BenchmarkMode,
    reference_path: Path,
    raw_npy_path: Path,
    metadata_npz_path: Path,
    scratch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Prepare exact contracts, run fresh children, and aggregate paired records."""

    _validate_mode(mode)
    parent_runtime = _require_runtime_contract(mode)
    source = _source_identity(repo_root)
    reference = _strict_json_load(reference_path)
    if not isinstance(reference, Mapping):
        raise ValueError("reference result must be a JSON object")
    _validate_reference_payload(reference, mode)
    formal_package_validation: Mapping[str, Any] | None = None
    if mode.require_formal_anchors:
        if Path(reference_path).name != all_auto.RESULT_FILENAME:
            raise ValueError(
                "formal reference must be the sealed package result filename"
            )
        formal_package_validation = all_auto.validate_output_package(
            Path(reference_path).resolve().parent,
            repo_root=Path(repo_root).resolve(),
            expected_analysis_sha256=FORMAL_ANALYSIS_SHA256,
            expected_package_anchor_sha256=FORMAL_PACKAGE_ANCHOR_SHA256,
        )
        if (
            formal_package_validation.get("package_self_consistency_verified")
            is not True
            or formal_package_validation.get("protocol_seed_cover_complete") is not True
        ):
            raise ValueError("sealed formal package validation did not complete")
    expected_raw_sha = (
        str(reference["input_identity"]["genotypes_npy_sha256"])
        if mode.require_formal_anchors
        else None
    )
    raw_receipt = _raw_npy_receipt(raw_npy_path, expected_sha256=expected_raw_sha)
    metadata_receipt = _metadata_receipt(metadata_npz_path)
    if (
        raw_receipt["shape"]
        != [
            int(reference["input_identity"]["samples"]),
            int(reference["input_identity"]["snps"]),
        ]
        or metadata_receipt["sample_ids_sha256"]
        != reference["input_identity"]["sample_ids_sha256"]
        or metadata_receipt["population_labels_sha256"]
        != reference["input_identity"]["population_labels_sha256"]
    ):
        raise ValueError("derived worker inputs differ from anchored array identities")
    raw = np.load(raw_npy_path, mmap_mode="r", allow_pickle=False)
    scratch = Path(scratch_root).resolve()
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("benchmark scratch root must be an existing real directory")
    itemsize = int(raw.dtype.itemsize)
    panel_counts = [
        int(
            _reference_case(reference, seed=seed, width=raw.shape[1])[
                "selected_snp_count"
            ]
        )
        for seed in mode.seeds
    ]
    estimated_compact_bytes = sum(
        int(raw.shape[0]) * count * itemsize + 4096 for count in panel_counts
    )
    required_scratch_free_bytes = (
        estimated_compact_bytes
        + int(raw.shape[0]) * max(panel_counts) * itemsize
        + 512 * 1024 * 1024
    )
    observed_scratch_free_bytes = int(shutil.disk_usage(scratch).free)
    if observed_scratch_free_bytes < required_scratch_free_bytes:
        raise OSError(
            "insufficient scratch for all compact panels plus one discovery worker: "
            f"need {required_scratch_free_bytes}, have {observed_scratch_free_bytes}"
        )

    with tempfile.TemporaryDirectory(
        prefix="rspca-runtime-workers-", dir=str(scratch)
    ) as temporary_name:
        temporary = Path(temporary_name)
        compact_receipts: dict[int, dict[str, Any]] = {}
        excluded_preparation: list[dict[str, Any]] = []
        width = int(raw.shape[1])
        for seed in mode.seeds:
            case = _reference_case(reference, seed=seed, width=width)
            mask = _selected_mask(case)
            panel_path = temporary / f"compact_panel_seed_{seed}.npy"
            receipt, timing = _timed_call(
                lambda panel_path=panel_path, mask=mask: materialize_compact_panel(
                    raw,
                    mask,
                    panel_path,
                    chunk_size=mode.snp_chunk_size,
                    hash_file=True,
                )
            )
            verify_compact_panel_binding(
                raw,
                mask,
                panel_path,
                chunk_size=mode.snp_chunk_size,
            )
            receipt["derived_from_anchored_raw_and_formal_mask_verified"] = True
            compact_receipts[int(seed)] = receipt
            excluded_preparation.append(
                {
                    "seed": int(seed),
                    "excluded_from_benchmark_timings": True,
                    "purpose": "pre_materialize_mask_bound_panel_for_deployment_workers",
                    "timing": timing,
                    "receipt": _without_path(receipt),
                }
            )

        formal_binding = None
        if mode.require_formal_anchors:
            if formal_package_validation is None:
                raise ValueError("formal benchmark requires sealed-package validation")
            formal_binding = {
                "analysis_sha256": FORMAL_ANALYSIS_SHA256,
                "package_anchor_sha256": FORMAL_PACKAGE_ANCHOR_SHA256,
                "package_self_consistency_verified": formal_package_validation.get(
                    "package_self_consistency_verified"
                ),
                "protocol_seed_cover_complete": formal_package_validation.get(
                    "protocol_seed_cover_complete"
                ),
                "trusted_scientific_replay_status_in_this_runtime_run": (
                    "externally_completed_for_source_result_but_not_reverified_by_"
                    "this_runtime_benchmark"
                ),
            }
        contract = _build_job_contract(
            mode=mode,
            source_identity=source,
            reference_receipt=_reference_file_receipt(reference_path),
            raw_receipt=raw_receipt,
            metadata_receipt=metadata_receipt,
            compact_receipts=compact_receipts,
            formal_package=formal_binding,
        )
        contract_path = temporary / "worker_job_contract.json"
        contract_path.write_bytes(_strict_json_bytes(contract))
        if _strict_json_load(contract_path) != contract:
            raise RuntimeError("worker job contract failed canonical round trip")

        worker_outputs: list[dict[str, Any]] = []
        for job in contract["jobs"]:
            worker_outputs.append(
                _execute_worker_job(
                    contract_path=contract_path,
                    job=job,
                    scratch_root=scratch,
                    mode=mode,
                )
            )
        pids = [int(output["worker_pid"]) for output in worker_outputs]
        if os.getpid() in set(pids):
            raise RuntimeError("worker invocation reused the orchestrator PID")

        keyed = {
            (
                int(output["job"]["seed"]),
                str(output["job"]["phase"]),
                int(output["job"]["repetition"]),
                str(output["job"]["worker_kind"]),
            ): output
            for output in worker_outputs
        }
        if len(keyed) != len(worker_outputs):
            raise RuntimeError("duplicate worker output key")
        warmup_receipts: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        for seed in mode.seeds:
            for kind in ("discovery", "deployment"):
                output = keyed[(seed, "warmup", -1, kind)]
                warmup_receipts.append(
                    {
                        "seed": int(seed),
                        "worker_kind": kind,
                        "worker_pid": int(output["worker_pid"]),
                        "excluded_from_summaries": True,
                        "science_sha256": str(output["science_sha256"]),
                        "job_contract_sha256": str(output["job_contract_sha256"]),
                    }
                )
            for repetition in range(mode.repeats):
                records.append(
                    _paired_record(
                        keyed[(seed, "timed", repetition, "discovery")],
                        keyed[(seed, "timed", repetition, "deployment")],
                    )
                )

        per_seed, aggregate = summarize_records(records, mode)
        compact_public = {
            str(seed): _without_path(compact_receipts[seed]) for seed in mode.seeds
        }

    del raw
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "complete_formal_retrospective_runtime_audit"
            if mode.require_formal_anchors
            else "complete_tiny_smoke_engineering_only"
        ),
        "analysis_policy": {
            "confirmatory": False,
            "runtime_audit_chosen_after_reviewing_stratification_grid": True,
            "preregistered_runtime_study": False,
            "retrospective_descriptive_only": True,
            "runtime_used_to_choose_rank_or_panel": False,
            "best_runtime_configuration_selected": False,
            "all_fixed_runtime_estimands_reported": True,
            "deployment_is_conditional_on_already_frozen_split_specific_panel": True,
            "deployment_is_not_panel_discovery": True,
            "finite_panels_are_not_one_universal_aim_list": True,
            "discovery_reported_as_overhead_vs_all_not_negative_reduction": True,
            "prepared_input_discovery_is_not_entire_snp_pipeline": True,
            "excluded_upstream_scope": (
                "source download, VCF/dosage generation, QC/LD pruning, prepared "
                "NPZ extraction, pre-timing hashes, and child startup"
            ),
            "hardware_generalization_claim": False,
        },
        "mode": _mode_payload(mode),
        "formal_protocol": dict(FORMAL_PROTOCOL),
        "source_result_binding": {
            "analysis_sha256": reference["analysis_sha256"],
            "package_anchor_sha256": (
                FORMAL_PACKAGE_ANCHOR_SHA256 if mode.require_formal_anchors else None
            ),
            "sealed_package_validation": formal_binding,
            "reference_result_file": _without_path(
                _reference_file_receipt(reference_path)
            ),
        },
        "input_binding": {
            "reference_input_identity": dict(reference["input_identity"]),
            "raw_genotypes_npy": _without_path(raw_receipt),
            "worker_metadata_npz": _without_path(metadata_receipt),
            "compact_panels_by_seed": compact_public,
            "parent_derived_binding_verified": True,
        },
        "source_identity": source,
        "parent_runtime_environment": parent_runtime,
        "worker_contract_sha256": contract["contract_sha256"],
        "fresh_child_worker_invocations": len(worker_outputs),
        "popen_output_runtime_pid_equality_verified_per_invocation": True,
        "scratch_preflight": {
            "estimated_compact_panel_bytes": int(estimated_compact_bytes),
            "required_free_bytes": int(required_scratch_free_bytes),
            "observed_free_bytes_before_materialization": int(
                observed_scratch_free_bytes
            ),
            "passed": True,
        },
        "excluded_panel_preparation": excluded_preparation,
        "excluded_warmup_receipts": warmup_receipts,
        "timed_worker_outputs": [
            output for output in worker_outputs if output["job"]["phase"] == "timed"
        ],
        "records": records,
        "per_seed_medians": per_seed,
        "aggregate_summary": aggregate,
    }
    unsigned = dict(payload)
    payload["analysis_sha256"] = canonical_sha256(unsigned)
    validate_result_payload(payload)
    return payload


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "analysis_policy",
        "mode",
        "formal_protocol",
        "source_result_binding",
        "input_binding",
        "source_identity",
        "parent_runtime_environment",
        "worker_contract_sha256",
        "fresh_child_worker_invocations",
        "popen_output_runtime_pid_equality_verified_per_invocation",
        "scratch_preflight",
        "excluded_panel_preparation",
        "excluded_warmup_receipts",
        "timed_worker_outputs",
        "records",
        "per_seed_medians",
        "aggregate_summary",
        "analysis_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("runtime result schema mismatch")
    _reject_nonfinite(payload)
    mode = _mode_from_payload(payload["mode"])
    _validate_mode(mode)
    expected_status = (
        "complete_formal_retrospective_runtime_audit"
        if mode.require_formal_anchors
        else "complete_tiny_smoke_engineering_only"
    )
    if (
        payload["status"] != expected_status
        or payload["formal_protocol"] != FORMAL_PROTOCOL
    ):
        raise ValueError("runtime result status or frozen protocol mismatch")
    policy = payload["analysis_policy"]
    expected_true = {
        "runtime_audit_chosen_after_reviewing_stratification_grid",
        "retrospective_descriptive_only",
        "all_fixed_runtime_estimands_reported",
        "deployment_is_conditional_on_already_frozen_split_specific_panel",
        "deployment_is_not_panel_discovery",
        "finite_panels_are_not_one_universal_aim_list",
        "discovery_reported_as_overhead_vs_all_not_negative_reduction",
        "prepared_input_discovery_is_not_entire_snp_pipeline",
    }
    expected_false = {
        "confirmatory",
        "preregistered_runtime_study",
        "runtime_used_to_choose_rank_or_panel",
        "best_runtime_configuration_selected",
        "hardware_generalization_claim",
    }
    if any(policy.get(key) is not True for key in expected_true) or any(
        policy.get(key) is not False for key in expected_false
    ):
        raise ValueError("runtime claim boundary changed")
    if not _is_sha256(payload["worker_contract_sha256"]):
        raise ValueError("worker contract SHA is invalid")
    source = payload["source_identity"]
    if (
        not isinstance(source, Mapping)
        or set(source.get("files", {})) != set(SOURCE_FILE_ROLES)
        or source.get("formal_runner_sha256_verified") is not True
    ):
        raise ValueError("runtime source identity is incomplete")
    binding = payload["source_result_binding"]
    if mode.require_formal_anchors:
        if (
            binding.get("analysis_sha256") != FORMAL_ANALYSIS_SHA256
            or binding.get("package_anchor_sha256") != FORMAL_PACKAGE_ANCHOR_SHA256
            or binding.get("sealed_package_validation", {}).get(
                "package_self_consistency_verified"
            )
            is not True
            or binding.get("sealed_package_validation", {}).get(
                "protocol_seed_cover_complete"
            )
            is not True
            or payload["input_binding"]["reference_input_identity"].get(
                "genotype_npz_sha256"
            )
            != all_auto.FROZEN_FORMAL_NPZ_SHA256
        ):
            raise ValueError("formal runtime result is not bound to sealed evidence")
    elif binding.get("package_anchor_sha256") is not None:
        raise ValueError("tiny smoke must not carry a formal package anchor")
    expected_workers = len(_worker_schedule(mode))
    if (
        payload["fresh_child_worker_invocations"] != expected_workers
        or payload["popen_output_runtime_pid_equality_verified_per_invocation"]
        is not True
    ):
        raise ValueError("fresh-child worker count or PID equality receipt mismatch")
    scratch_preflight = payload["scratch_preflight"]
    if (
        not isinstance(scratch_preflight, Mapping)
        or scratch_preflight.get("passed") is not True
        or int(scratch_preflight.get("observed_free_bytes_before_materialization", -1))
        < int(scratch_preflight.get("required_free_bytes", 0))
    ):
        raise ValueError("scratch-space preflight is invalid")
    preparations = payload["excluded_panel_preparation"]
    if (
        not isinstance(preparations, list)
        or [row.get("seed") for row in preparations] != list(mode.seeds)
        or any(
            row.get("excluded_from_benchmark_timings") is not True
            for row in preparations
        )
        or any(
            row.get("receipt", {}).get(
                "derived_from_anchored_raw_and_formal_mask_verified"
            )
            is not True
            for row in preparations
        )
    ):
        raise ValueError("excluded panel-preparation receipts are incomplete")
    warmups = payload["excluded_warmup_receipts"]
    expected_warmups = [
        (seed, kind) for seed in mode.seeds for kind in ("discovery", "deployment")
    ]
    if (
        not isinstance(warmups, list)
        or [(row.get("seed"), row.get("worker_kind")) for row in warmups]
        != expected_warmups
        or any(row.get("excluded_from_summaries") is not True for row in warmups)
        or any(
            row.get("job_contract_sha256") != payload["worker_contract_sha256"]
            for row in warmups
        )
    ):
        raise ValueError("excluded warmup cover mismatch")
    records = payload["records"]
    expected_keys = [
        (seed, repetition) for seed in mode.seeds for repetition in range(mode.repeats)
    ]
    if (
        not isinstance(records, list)
        or [(row.get("seed"), row.get("repetition")) for row in records]
        != expected_keys
        or any(
            row.get("job_contract_sha256") != payload["worker_contract_sha256"]
            for row in records
        )
    ):
        raise ValueError("timed record seed/repetition cover mismatch")
    timed_outputs = payload["timed_worker_outputs"]
    expected_timed_jobs = [
        job for job in _worker_schedule(mode) if job["phase"] == "timed"
    ]
    if not isinstance(timed_outputs, list) or len(timed_outputs) != len(
        expected_timed_jobs
    ):
        raise ValueError("full timed worker-output cover mismatch")
    for output, job in zip(timed_outputs, expected_timed_jobs, strict=True):
        if "subprocess_stderr_receipt" not in output:
            raise ValueError("persisted timed worker output lacks stderr receipt")
        validate_worker_output(
            output,
            expected_job=job,
            mode=mode,
            expected_contract_sha256=str(payload["worker_contract_sha256"]),
        )
        if output["worker_pid"] != output["runtime_environment"]["pid"]:
            raise ValueError("saved worker/runtime PID equality mismatch")
    output_map = {
        (
            int(output["job"]["seed"]),
            int(output["job"]["repetition"]),
            str(output["job"]["worker_kind"]),
        ): output
        for output in timed_outputs
    }
    rebuilt_records = [
        _paired_record(
            output_map[(seed, repetition, "discovery")],
            output_map[(seed, repetition, "deployment")],
        )
        for seed in mode.seeds
        for repetition in range(mode.repeats)
    ]
    if records != rebuilt_records:
        raise ValueError("paired records do not reproduce from full worker outputs")
    for seed_index, seed in enumerate(mode.seeds):
        for repetition in range(mode.repeats):
            row = records[seed_index * mode.repeats + repetition]
            expected_order = (
                "discovery_then_deployment"
                if (seed_index + repetition) % 2 == 0
                else "deployment_then_discovery"
            )
            if row["execution_order"] != expected_order:
                raise ValueError("timed worker order differs from balanced schedule")
            for key in (
                "all_reference_wall_seconds",
                "frozen_panel_deployment_wall_seconds",
                "prepared_input_discovery_wall_seconds",
            ):
                if float(row[key]) <= 0.0 or not math.isfinite(float(row[key])):
                    raise ValueError(f"invalid headline runtime: {key}")
            all_wall = float(row["all_reference_wall_seconds"])
            deployment_wall = float(row["frozen_panel_deployment_wall_seconds"])
            discovery_wall = float(row["prepared_input_discovery_wall_seconds"])
            if (
                not math.isclose(
                    row["deployment_speedup_vs_all"],
                    all_wall / deployment_wall,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    row["deployment_runtime_reduction_fraction_vs_all"],
                    1.0 - deployment_wall / all_wall,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    row["discovery_overhead_fraction_vs_all"],
                    discovery_wall / all_wall - 1.0,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                raise ValueError("paired runtime arithmetic mismatch")
            if (
                not _is_sha256(row["selected_original_indices_sha256"])
                or not _is_sha256(row["all_test_predictions_sha256"])
                or not _is_sha256(row["panel_test_predictions_sha256"])
            ):
                raise ValueError("record scientific hash is invalid")
    expected_medians, expected_summary = summarize_records(records, mode)
    if payload["per_seed_medians"] != expected_medians:
        raise ValueError("per-seed medians do not reproduce from timed records")
    if payload["aggregate_summary"] != expected_summary:
        raise ValueError("aggregate summary does not reproduce from seed medians")
    unsigned = dict(payload)
    observed_digest = unsigned.pop("analysis_sha256")
    if not _is_sha256(observed_digest) or canonical_sha256(unsigned) != observed_digest:
        raise ValueError("runtime analysis SHA mismatch")
    return {
        "analysis_sha256": observed_digest,
        "seed_count": len(mode.seeds),
        "timed_record_count": len(records),
        "fresh_child_worker_invocations": expected_workers,
        "formal_anchors_verified": mode.require_formal_anchors,
    }


RECORD_FIELDS = (
    "seed",
    "repetition",
    "execution_order",
    "all_snp_count",
    "selected_snp_count",
    "all_reference_wall_seconds",
    "frozen_panel_deployment_wall_seconds",
    "prepared_input_discovery_wall_seconds",
    "deployment_speedup_vs_all",
    "deployment_runtime_reduction_fraction_vs_all",
    "discovery_overhead_fraction_vs_all",
    "all_reference_cpu_seconds",
    "frozen_panel_deployment_cpu_seconds",
    "prepared_input_discovery_cpu_seconds",
    "all_reference_ru_maxrss_after_bytes",
    "frozen_panel_deployment_ru_maxrss_after_bytes",
    "prepared_input_discovery_ru_maxrss_after_bytes",
    "all_balanced_accuracy",
    "panel_balanced_accuracy",
    "all_macro_f1",
    "panel_macro_f1",
    "selected_original_indices_sha256",
    "all_test_predictions_sha256",
    "panel_test_predictions_sha256",
    "discovery_worker_pid",
    "deployment_worker_pid",
    "job_contract_sha256",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fields),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    format(row[field], ".17g")
                    if isinstance(row.get(field), float)
                    else row.get(field)
                )
                for field in fields
            }
        )
    return buffer.getvalue().encode("utf-8")


def _records_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["records"], RECORD_FIELDS)


SUMMARY_FIELDS = (
    "row_type",
    "seed",
    "metric",
    "value",
    "ci95_low",
    "ci95_high",
    "bootstrap_resamples",
    "bootstrap_seed",
    "bootstrap_unit",
)


def _summary_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_row in payload["per_seed_medians"]:
        for metric in sorted(seed_row):
            value = seed_row[metric]
            if metric in {"seed", "timed_repetitions"}:
                continue
            rows.append(
                {
                    "row_type": "per_seed_median",
                    "seed": int(seed_row["seed"]),
                    "metric": metric,
                    "value": value,
                    "ci95_low": None,
                    "ci95_high": None,
                    "bootstrap_resamples": None,
                    "bootstrap_seed": None,
                    "bootstrap_unit": None,
                }
            )
    aggregate = payload["aggregate_summary"]
    for metric in sorted(aggregate):
        value = aggregate[metric]
        if isinstance(value, Mapping) and "estimate" in value:
            rows.append(
                {
                    "row_type": "aggregate_paired_bootstrap",
                    "seed": None,
                    "metric": metric,
                    "value": value["estimate"],
                    "ci95_low": value["ci95_low"],
                    "ci95_high": value["ci95_high"],
                    "bootstrap_resamples": value["resamples"],
                    "bootstrap_seed": value["seed"],
                    "bootstrap_unit": value["unit"],
                }
            )
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append(
                {
                    "row_type": "aggregate_descriptive",
                    "seed": None,
                    "metric": metric,
                    "value": value,
                    "ci95_low": None,
                    "ci95_high": None,
                    "bootstrap_resamples": None,
                    "bootstrap_seed": None,
                    "bootstrap_unit": None,
                }
            )
    return rows


def _summary_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(_summary_rows(payload), SUMMARY_FIELDS)


def _render_figure(payload: Mapping[str, Any], destination: Path) -> None:
    import matplotlib.pyplot as plt

    rows = payload["per_seed_medians"]
    seeds = np.asarray([row["seed"] for row in rows], dtype=int)
    all_wall = np.asarray(
        [row["all_reference_wall_seconds"] for row in rows], dtype=float
    )
    deployment = np.asarray(
        [row["frozen_panel_deployment_wall_seconds"] for row in rows], dtype=float
    )
    discovery = np.asarray(
        [row["prepared_input_discovery_wall_seconds"] for row in rows], dtype=float
    )
    figure, axes = plt.subplots(1, 2, figsize=(16, 7.2), constrained_layout=False)
    left, right = axes
    left.plot(seeds, all_wall, "o-", label="All-SNP reference subspan", lw=1.7)
    left.plot(
        seeds,
        deployment,
        "o-",
        label="Frozen split-specific 5% deployment",
        lw=1.7,
    )
    left.plot(
        seeds,
        discovery,
        "o-",
        label="Prepared-input 5% discovery total",
        lw=1.7,
    )
    left.set_xlabel("Registered outer-split seed")
    left.set_ylabel("Per-seed median wall time (seconds)")
    left.set_title("Paired fresh-child runtime")
    tick_step = max(1, int(math.ceil(seeds.size / 10)))
    left.set_xticks(seeds[::tick_step])
    if (
        float(np.max(np.concatenate((all_wall, deployment, discovery))))
        / float(np.min(np.concatenate((all_wall, deployment, discovery))))
        >= 8.0
    ):
        left.set_yscale("log")
        left.set_ylabel("Per-seed median wall time (seconds, log scale)")
    left.grid(alpha=0.25)
    left.legend(fontsize=9)

    deployment_reduction = 100.0 * (1.0 - deployment / all_wall)
    discovery_overhead = 100.0 * (discovery / all_wall - 1.0)
    positions = np.arange(seeds.size)
    right.axhline(0.0, color="black", linewidth=0.8)
    right.scatter(
        positions - 0.08,
        deployment_reduction,
        label="Deployment reduction vs all",
        s=34,
    )
    right.scatter(
        positions + 0.08,
        discovery_overhead,
        label="Prepared-input discovery overhead vs all",
        s=34,
    )
    summary = payload["aggregate_summary"]
    aggregate_position = float(seeds.size + 0.7)
    deployment_ci = summary["deployment_reduction"]
    discovery_ci = summary["discovery_overhead"]
    right.errorbar(
        aggregate_position - 0.08,
        100.0 * float(deployment_ci["estimate"]),
        yerr=np.asarray(
            [
                [100.0 * float(deployment_ci["estimate"] - deployment_ci["ci95_low"])],
                [100.0 * float(deployment_ci["ci95_high"] - deployment_ci["estimate"])],
            ]
        ),
        fmt="D",
        color="C0",
        capsize=5,
        label="Aggregate estimate + paired 95% CI",
    )
    right.errorbar(
        aggregate_position + 0.08,
        100.0 * float(discovery_ci["estimate"]),
        yerr=np.asarray(
            [
                [100.0 * float(discovery_ci["estimate"] - discovery_ci["ci95_low"])],
                [100.0 * float(discovery_ci["ci95_high"] - discovery_ci["estimate"])],
            ]
        ),
        fmt="D",
        color="C1",
        capsize=5,
    )
    shown = positions[::tick_step].tolist() + [aggregate_position]
    labels = [str(seed) for seed in seeds[::tick_step]] + ["Aggregate\n95% CI"]
    right.set_xticks(shown, labels, rotation=35, ha="right")
    right.set_ylabel("Percent")
    right.set_title("Two estimands; no runtime winner selected")
    right.grid(alpha=0.25)
    right.legend(fontsize=9)
    figure.suptitle(
        "All-autosome PCA/loading runtime audit — conditional frozen panel vs "
        "prepared-input discovery",
        fontsize=14,
    )
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.25, top=0.84, wspace=0.28)
    figure.text(
        0.5,
        0.055,
        "Deployment excludes panel discovery/materialization. Prepared-input discovery "
        "excludes download, VCF/dosage, QC/LD and NPZ extraction.\n"
        "Both exclude pre-timing integrity hashes and child startup; RSS/runtime are "
        "hardware-conditional descriptive evidence.",
        ha="center",
        va="center",
        fontsize=8.5,
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_output_package(
    payload: Mapping[str, Any], output_directory: Path
) -> dict[str, Any]:
    validate_result_payload(payload)
    output = Path(output_directory)
    if os.path.lexists(output):
        raise FileExistsError(f"output directory already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / RECORDS_FILENAME).write_bytes(_records_csv_bytes(payload))
        (temporary / SUMMARY_FILENAME).write_bytes(_summary_csv_bytes(payload))
        _render_figure(payload, temporary / FIGURE_FILENAME)
        artifacts: dict[str, dict[str, Any]] = {}
        for filename in ARTIFACT_FILENAMES:
            path = temporary / filename
            status = _regular_file(path, role=filename)
            artifacts[filename] = {
                "bytes": int(status.st_size),
                "sha256": _file_sha256(path),
            }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": payload["status"],
            "manifest_written_last": True,
            "analysis_sha256": payload["analysis_sha256"],
            "source_result_analysis_sha256": payload["source_result_binding"][
                "analysis_sha256"
            ],
            "source_result_package_anchor_sha256": payload["source_result_binding"][
                "package_anchor_sha256"
            ],
            "artifact_count_excluding_manifest": len(ARTIFACT_FILENAMES),
            "exact_package_file_count": len(ARTIFACT_FILENAMES) + 1,
            "artifacts": artifacts,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "artifacts": artifacts,
            }
        )
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        temporary.rename(output)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return validate_output_package(output)


def validate_output_package(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    output = Path(output_directory)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("runtime output package must be a real directory")
    expected_files = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    actual_files = {entry.name for entry in output.iterdir()}
    if actual_files != expected_files:
        raise ValueError("runtime output package file closure mismatch")
    manifest = _strict_json_load(output / MANIFEST_FILENAME)
    if not isinstance(manifest, Mapping):
        raise ValueError("runtime output manifest must be an object")
    required_manifest = {
        "schema_version",
        "status",
        "manifest_written_last",
        "analysis_sha256",
        "source_result_analysis_sha256",
        "source_result_package_anchor_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if (
        set(manifest) != required_manifest
        or manifest["schema_version"] != 1
        or manifest["manifest_written_last"] is not True
        or manifest["artifact_count_excluding_manifest"] != len(ARTIFACT_FILENAMES)
        or manifest["exact_package_file_count"] != len(expected_files)
        or set(manifest["artifacts"]) != set(ARTIFACT_FILENAMES)
    ):
        raise ValueError("runtime output manifest schema/policy mismatch")
    unsigned_manifest = dict(manifest)
    manifest_digest = unsigned_manifest.pop("manifest_payload_sha256")
    if (
        not _is_sha256(manifest_digest)
        or canonical_sha256(unsigned_manifest) != manifest_digest
    ):
        raise ValueError("runtime manifest payload SHA mismatch")
    for filename in ARTIFACT_FILENAMES:
        path = output / filename
        status = _regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if (
            set(receipt) != {"bytes", "sha256"}
            or int(receipt["bytes"]) != status.st_size
            or receipt["sha256"] != _file_sha256(path)
        ):
            raise ValueError(f"runtime artifact receipt mismatch: {filename}")
    expected_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest["package_anchor_sha256"] != expected_anchor:
        raise ValueError("runtime package anchor mismatch")
    result = _strict_json_load(output / RESULT_FILENAME)
    if not isinstance(result, Mapping):
        raise ValueError("runtime result must be an object")
    validation = validate_result_payload(result)
    if (
        result["analysis_sha256"] != manifest["analysis_sha256"]
        or result["status"] != manifest["status"]
        or result["source_result_binding"]["analysis_sha256"]
        != manifest["source_result_analysis_sha256"]
        or result["source_result_binding"]["package_anchor_sha256"]
        != manifest["source_result_package_anchor_sha256"]
        or (output / RECORDS_FILENAME).read_bytes() != _records_csv_bytes(result)
        or (output / SUMMARY_FILENAME).read_bytes() != _summary_csv_bytes(result)
    ):
        raise ValueError("runtime package serialization mismatch")
    if (expected_analysis_sha256 is None) != (expected_package_anchor_sha256 is None):
        raise ValueError("external validation requires both runtime anchors")
    if expected_analysis_sha256 is not None and (
        result["analysis_sha256"] != expected_analysis_sha256
        or manifest["package_anchor_sha256"] != expected_package_anchor_sha256
    ):
        raise ValueError("runtime package differs from supplied external anchors")
    if repo_root is not None:
        current_source = _source_identity(Path(repo_root).resolve())
        if current_source != result["source_identity"]:
            raise ValueError(
                "runtime package source identity differs from current source"
            )
    from PIL import Image

    with Image.open(output / FIGURE_FILENAME) as image:
        image.verify()
    with Image.open(output / FIGURE_FILENAME) as image:
        if image.format != "PNG" or image.width < 1800 or image.height < 700:
            raise ValueError("runtime PNG is below the clarity floor")
    return {
        **validation,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "package_self_consistency_verified": True,
    }


def _write_worker_metadata(
    destination: Path,
    *,
    sample_ids: np.ndarray,
    population_labels: np.ndarray,
) -> None:
    if os.path.lexists(destination):
        raise FileExistsError(f"worker metadata destination exists: {destination}")
    np.savez(
        destination,
        sample_ids=np.asarray(sample_ids),
        population_labels=np.asarray(population_labels),
    )


def run_tiny_smoke(
    *,
    output_directory: Path,
    scratch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    scratch = Path(scratch_root).resolve()
    if scratch.is_symlink() or not scratch.is_dir():
        raise ValueError("tiny-smoke scratch root must be an existing real directory")
    data = all_auto._synthetic_smoke_data()
    reference = all_auto.evaluate_all_autosome_stratification(
        data,
        all_auto.AllAutosomeConfig(
            seeds=SMOKE_MODE.seeds,
            analysis_mode="tiny_smoke",
            snp_chunk_size=SMOKE_MODE.snp_chunk_size,
        ),
        input_identity=all_auto._synthetic_input_identity(data),
        source_identity=all_auto._source_identity(Path(repo_root).resolve()),
    )
    with tempfile.TemporaryDirectory(
        prefix="rspca-runtime-smoke-parent-", dir=str(scratch)
    ) as temporary_name:
        temporary = Path(temporary_name)
        raw_path = temporary / "genotypes.npy"
        metadata_path = temporary / "metadata.npz"
        reference_path = temporary / "all_autosome_stratification_results.json"
        np.save(raw_path, np.asarray(data.genotypes), allow_pickle=False)
        _write_worker_metadata(
            metadata_path,
            sample_ids=np.asarray(data.sample_ids),
            population_labels=np.asarray(data.population_labels),
        )
        reference_path.write_bytes(_strict_json_bytes(reference))
        payload = execute_benchmark_workers(
            mode=SMOKE_MODE,
            reference_path=reference_path,
            raw_npy_path=raw_path,
            metadata_npz_path=metadata_path,
            scratch_root=scratch,
            repo_root=repo_root,
        )
    return write_output_package(payload, output_directory)


def run_formal_benchmark(
    *,
    genotype_npz: Path,
    source_manifest: Path,
    formal_package: Path,
    output_directory: Path,
    scratch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    reference_path = Path(formal_package).resolve() / all_auto.RESULT_FILENAME
    with all_auto.load_formal_input(
        genotype_npz,
        source_manifest,
        expected_source_manifest_sha256=all_auto.FROZEN_FORMAL_MANIFEST_SHA256,
        scratch_root=scratch_root,
    ) as (data, identity):
        if (
            identity["genotype_npz_sha256"] != all_auto.FROZEN_FORMAL_NPZ_SHA256
            or identity["genotypes_npy_sha256"]
            != all_auto.FROZEN_PREPARED_ARRAY_SHA256["genotypes"]
        ):
            raise ValueError("loaded formal input differs from exact frozen anchors")
        raw_filename = getattr(data.genotypes, "filename", None)
        if raw_filename is None:
            raise RuntimeError("formal genotype input was not extracted as an mmap")
        scratch = Path(scratch_root).resolve()
        with tempfile.TemporaryDirectory(
            prefix="rspca-runtime-formal-parent-", dir=str(scratch)
        ) as temporary_name:
            metadata_path = Path(temporary_name) / "metadata.npz"
            _write_worker_metadata(
                metadata_path,
                sample_ids=np.asarray(data.sample_ids),
                population_labels=np.asarray(data.population_labels),
            )
            payload = execute_benchmark_workers(
                mode=FORMAL_MODE,
                reference_path=reference_path,
                raw_npy_path=Path(str(raw_filename)),
                metadata_npz_path=metadata_path,
                scratch_root=scratch,
                repo_root=repo_root,
            )
    return write_output_package(payload, output_directory)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    smoke = commands.add_parser("smoke", help="run the engineering-only tiny smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--scratch-root", type=Path, required=True)
    smoke.add_argument("--repo-root", type=Path, default=Path("."))

    run = commands.add_parser("run", help="run the exact formal 20-seed audit")
    run.add_argument("--genotypes", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument("--formal-package", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--scratch-root", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))

    validate = commands.add_parser("validate", help="validate a runtime package")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path)
    validate.add_argument("--expected-analysis-sha256")
    validate.add_argument("--expected-package-anchor-sha256")

    worker = commands.add_parser(
        "internal-worker",
        help=argparse.SUPPRESS,
    )
    worker.add_argument("--job-contract", type=Path, required=True)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--scratch-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "internal-worker":
        contract, job, _ = _load_authorized_job(args.job_contract, args.job_id)
        if job["worker_kind"] == "discovery":
            output = run_discovery_worker(
                contract_path=args.job_contract,
                job_id=args.job_id,
                scratch_root=args.scratch_root,
            )
        elif job["worker_kind"] == "deployment":
            output = run_deployment_worker(
                contract_path=args.job_contract,
                job_id=args.job_id,
            )
        else:  # Contract validation should already make this unreachable.
            raise ValueError("unknown authorized worker kind")
        if output["job_contract_sha256"] != contract["contract_sha256"]:
            raise RuntimeError("worker output lost its contract binding")
        sys.stdout.buffer.write(_strict_json_bytes(output))
        return 0
    if args.command == "validate":
        validation = validate_output_package(
            args.output_dir,
            repo_root=args.repo_root,
            expected_analysis_sha256=args.expected_analysis_sha256,
            expected_package_anchor_sha256=args.expected_package_anchor_sha256,
        )
    elif args.command == "smoke":
        validation = run_tiny_smoke(
            output_directory=args.output_dir,
            scratch_root=args.scratch_root,
            repo_root=args.repo_root,
        )
    else:
        validation = run_formal_benchmark(
            genotype_npz=args.genotypes,
            source_manifest=args.source_manifest,
            formal_package=args.formal_package,
            output_directory=args.output_dir,
            scratch_root=args.scratch_root,
            repo_root=args.repo_root,
        )
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

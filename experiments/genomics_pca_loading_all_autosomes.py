#!/usr/bin/env python3
"""Fixed PCA-loading stratification evaluation on autosomes 1--22.

This runner is an additive continuation of the repository's original Genomics
notebooks and the chromosome-20--22 PCA-loading diagnostic.  It uses only:

* outer-training-only smartpca-like allele-frequency scaling;
* exact PCA through a chunked sample Gram matrix;
* unsupervised ranking by the sum of absolute training PCA loadings;
* a training-only PCA refit for each frozen loading panel; and
* nearest-training-centroid classification on held-out samples.

The fixed descriptive grid is ranks 4/6/8 crossed with 5%/10%/25%/50%/all of
the train-retained SNPs over the registered seeds 2025..2044.  Counts are the
deterministic floor of each fraction (with duplicate effective counts removed
and disclosed). Every resolved cell is reported;
there is no winner selection, target metric, quality gate, or result-triggered
parameter change.  Population labels never enter preprocessing, PCA, or SNP
ranking.  No robust, sparse, randomized, federated, LPCA, ALPCA, or Fisher
solver is run.
"""

from __future__ import annotations

import argparse
from base64 import b64decode, b64encode
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterator, Mapping, Sequence
import uuid
import zipfile

import numpy as np
from numpy.typing import NDArray
from scipy import linalg
import threadpoolctl

if __package__:
    from . import genomics_pca_loading_grid as base
else:  # Support direct execution from the repository root.
    import genomics_pca_loading_grid as base  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import (
    LEGACY_GITHUB_BASELINE_COMMIT,
    LEGACY_SOURCE_SHA256,
    stratified_split,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

PROTOCOL_SEEDS = tuple(range(2025, 2045))
COMPONENT_GRID = (4, 6, 8)
PANEL_FRACTIONS: tuple[float | None, ...] = (0.05, 0.10, 0.25, 0.50, None)
EXPECTED_POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
EXPECTED_AUTOSOMES = tuple(str(value) for value in range(1, 23))
NPZ_KEYS = (
    "chromosomes",
    "genotypes",
    "population_labels",
    "sample_ids",
    "snp_ids",
)
FORMAL_MANIFEST_SCHEMA = "rspca-1kgp-all-autosome-preparation-v1"
FROZEN_FORMAL_MANIFEST_SHA256 = (
    "0abdf5f4b6c6724f30f539377f1244bb90d3932d7c3e1b9fd6510950bef61030"
)
FROZEN_FORMAL_NPZ_SHA256 = (
    "6ef9864f1d15b8e294434d84b07443ba0d66a19bcfb38401cd1fcd67581e80bc"
)
FROZEN_FORMAL_NPZ_BYTES = 157_174_730
FROZEN_FORMAL_SHAPE = (1000, 455_102)
FROZEN_FORMAL_TOTAL_QC_VARIANTS = 6_638_894
FROZEN_LEGACY_PREPARATION_MANIFEST_SHA256 = (
    "9378814177e0721e711ce2e6c3695effc5ff766d2529c17cc5c4e0de84f30342"
)
FROZEN_SELECTED_SAMPLE_IDS_SHA256 = (
    "fbb844e7d6277b15ebdd63d2a362f8254c0edf901bbbb82f6a7b81e7f6095497"
)
FROZEN_SELECTED_RECORDS_SHA256 = (
    "0616dd6392c601cc1c6d9c559d0d4bc74d636166af739ee9d0a0009c6c675253"
)
FROZEN_PREPARED_ARRAY_SHA256 = {
    "chromosomes": "aa105e30e76210eacdbd4fcd9c13512252203d0301cb462530aae9b074f2fb60",
    "genotypes": "0c37c18b2ca06429b0a8fb491531b2da4a105dfd5d40b8642a28cce94a9f2930",
    "population_labels": (
        "6ca09a20bd51f5ada6c18be707ed3b7596b5d31fa95746b1e9c3cfc642832277"
    ),
    "sample_ids": "9418acf4a1dc45982f70318f9f2109413280e6693c4641e94f649874156d53a4",
    "snp_ids": "527485ecfa1592527467b361c879ec0cf35a7da6391787cda66b511e91e3bdcb",
}
FROZEN_PREPARER_SOURCE_SHA256 = (
    "c35632846f2a31c14f54986545ab87837a53597e1f3f05e0696a5b3e403356e4"
)
FROZEN_DOWNLOADER_SOURCE_SHA256 = (
    "a93a6ce70a3e830841de9d5a720a72e7807cd15b94cf5a5fb0a01561ceb83a93"
)
FROZEN_LEGACY_PREPARER_SOURCE_SHA256 = (
    "0c1252f05b42f52797bb3f8536a11da623e33e347ee6d137943bb401c9ed19e8"
)
FROZEN_UPSTREAM_SOURCE_MANIFEST_SHA256 = (
    "5e59af304e9d6b1e0a4099d139943fa1b9d3438a2588d9ee4f1f2f49206384c1"
)
FROZEN_DOWNLOAD_COMPLETION_SHA256 = (
    "bfabf0af2cb5f2bffa4a4243db64bf96ef788a979f0d1a7872ee2f3553b5291d"
)
FROZEN_PLINK2_SHA256 = (
    "f3e0f57049651a04da0948b4bb8e2509b08e71d760f6959c3ec74ea4436c916b"
)
FROZEN_PLINK2_VERSION = "PLINK v2.0.0-a.7.4LM 64-bit Intel (18 Aug 2026)"
FROZEN_LEGACY_GITHUB_BASELINE_COMMIT = "03f4b432e5d552a9772cfb251fe06f33833b4197"
FROZEN_LEGACY_SOURCE_SHA256 = {
    "gene.ipynb": "60110c78b28da17042af55ee63087aceb4f786542e8cd478f783f21f710d29fa",
    "gene large scale.ipynb": (
        "37c6605617186f28a2af11a2ad9c2db7c657d13e4b78314ae05feeaacf9e4f4b"
    ),
    "spca_variable_projection.py": (
        "0611c5cf14701c6b54ac6624473a3e3b7738d1220bf1f48fc12a8c83bbf9e93d"
    ),
    "randomized_spca_vp.py": (
        "5ce6a21c86bdd88f23bebe689cdf6ac9178854838ee8329b82c67f02205d40dc"
    ),
    "robust_spca_vp.py": (
        "eac1b41bd1fb27d5542cf42d3cb05e1498ebbc3be57616efe1fa112133ac4692"
    ),
}

METHOD_SCOPE = {
    "methods": ["pca_loading_ranking"],
    "robust_solver_run": False,
    "vp_solver_run": False,
    "randomized_solver_run": False,
    "federated_solver_run": False,
    "lpca_solver_run": False,
    "alpca_solver_run": False,
    "fisher_or_supervised_panel_ranking_run": False,
}

RESULT_FILENAME = "all_autosome_stratification_results.json"
METRICS_FILENAME = "all_autosome_stratification_grid_metrics.csv"
POPULATION_FILENAME = "all_autosome_stratification_population_recall.csv"
STABILITY_FILENAME = "all_autosome_stratification_panel_stability.csv"
SAMPLE_FILENAME = "all_autosome_stratification_sample_predictions.csv"
COORDINATES_FILENAME = "all_autosome_stratification_registered_coordinates.csv"
PC12_FILENAME = "all_autosome_stratification_pc12.png"
PC34_FILENAME = "all_autosome_stratification_pc34.png"
CONFUSION_FILENAME = "all_autosome_stratification_confusion.png"
MANIFEST_FILENAME = "all_autosome_stratification_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    METRICS_FILENAME,
    POPULATION_FILENAME,
    STABILITY_FILENAME,
    SAMPLE_FILENAME,
    COORDINATES_FILENAME,
    PC12_FILENAME,
    PC34_FILENAME,
    CONFUSION_FILENAME,
)

SOURCE_FILE_ROLES = {
    "experiments/genomics_pca_loading_all_autosomes.py": (
        "fixed_all_autosome_pca_loading_runner"
    ),
    "experiments/genomics_pca_loading_grid.py": (
        "historical_pca_loading_metrics_and_small_panel_eigh_helpers"
    ),
    "src/rspca/genomics_aim.py": "historical_split_and_smartpca_lineage",
    "src/rspca/datasets.py": "array_and_canonical_hashing_runtime_identity",
    "src/rspca/__init__.py": "package_initialization_only",
}

_NESTED_GRAM_CAPABILITY = object()
_BYTE_POPCOUNT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(
    axis=1, dtype=np.uint16
)


@dataclass(frozen=True)
class AllAutosomeConfig:
    """Fixed descriptive protocol; ``None`` denotes all train-retained SNPs."""

    seeds: tuple[int, ...] = PROTOCOL_SEEDS
    component_grid: tuple[int, ...] = COMPONENT_GRID
    panel_fractions: tuple[float | None, ...] = PANEL_FRACTIONS
    test_fraction: float = 0.2
    loading_score: str = "l1_abs_sum"
    population_design: str = "five_superpopulation_200_each"
    analysis_status: str = "fixed_descriptive_development_evaluation"
    analysis_mode: str = "fixed_protocol"
    snp_chunk_size: int = 2048


@dataclass(frozen=True)
class ChunkedPCAFit:
    """Train-only preprocessing and exact PCA products without a dense X copy."""

    mean_dosage: FloatArray
    scale: FloatArray
    retained_mask: BoolArray
    retained_original: IntArray
    loadings: FloatArray
    singular_values: FloatArray
    explained_variance_ratio: FloatArray
    train_scores: FloatArray
    test_scores: FloatArray
    sample_gram_sha256: str


def _config_payload(config: AllAutosomeConfig) -> dict[str, Any]:
    """Return the config in its canonical JSON-domain representation."""

    payload = asdict(config)
    for field in ("seeds", "component_grid", "panel_fractions"):
        payload[field] = list(payload[field])
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _npy_array_sha256(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return _bytes_sha256(buffer.getvalue())


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_nonfinite_json(value: Any, *, path: str = "root") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_json(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_json(item, path=f"{path}[{index}]")


def _strict_json_bytes(value: Any) -> bytes:
    _reject_nonfinite_json(value)
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    with path.open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular file: {path}")
    return status


def _panel_label(panel_fraction: float | None) -> str:
    if panel_fraction is None:
        return "all"
    return f"{int(round(100.0 * float(panel_fraction)))}pct"


def _blas_thread_environment() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
    }


def _blas_build_identity() -> dict[str, Any]:
    config = getattr(np.__config__, "CONFIG", {})
    dependencies = (
        config.get("Build Dependencies", {}) if isinstance(config, Mapping) else {}
    )
    blas = dependencies.get("blas", {}) if isinstance(dependencies, Mapping) else {}
    if not isinstance(blas, Mapping):
        blas = {}
    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "openblas_configuration": blas.get("openblas configuration"),
    }


def _blas_runtime_identity() -> dict[str, Any]:
    """Report the BLAS libraries and thread counts loaded in this process."""

    pool_fields = (
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
    )
    pools: list[dict[str, Any]] = []
    for raw_pool in threadpoolctl.threadpool_info():
        if raw_pool.get("user_api") != "blas":
            continue
        threads = raw_pool.get("num_threads")
        if type(threads) is not int or threads < 1:
            raise ValueError("threadpoolctl reported an invalid BLAS thread count")
        pool = {field: raw_pool.get(field) for field in pool_fields}
        pool["num_threads"] = threads
        pools.append(pool)
    pools.sort(
        key=lambda pool: tuple(
            "" if pool.get(field) is None else str(pool[field])
            for field in (*pool_fields, "num_threads")
        )
    )
    return {
        "threadpoolctl_version": str(threadpoolctl.__version__),
        "blas_pools": pools,
    }


def _validate_blas_runtime_identity(
    identity: Any, *, require_single_thread: bool
) -> None:
    if not isinstance(identity, Mapping) or set(identity) != {
        "threadpoolctl_version",
        "blas_pools",
    }:
        raise ValueError("BLAS runtime identity schema mismatch")
    version = identity.get("threadpoolctl_version")
    pools = identity.get("blas_pools")
    if not isinstance(version, str) or not version or not isinstance(pools, list):
        raise ValueError("BLAS runtime identity is invalid")
    expected_pool_fields = {
        "user_api",
        "internal_api",
        "prefix",
        "version",
        "threading_layer",
        "architecture",
        "num_threads",
    }
    for pool in pools:
        if not isinstance(pool, Mapping) or set(pool) != expected_pool_fields:
            raise ValueError("BLAS runtime pool schema mismatch")
        threads = pool.get("num_threads")
        if (
            pool.get("user_api") != "blas"
            or not isinstance(pool.get("internal_api"), str)
            or not pool.get("internal_api")
            or type(threads) is not int
            or threads < 1
        ):
            raise ValueError("BLAS runtime pool identity is invalid")
    if require_single_thread and not pools:
        raise ValueError("formal execution requires a detected BLAS runtime pool")
    if require_single_thread and any(pool["num_threads"] != 1 for pool in pools):
        counts = [int(pool["num_threads"]) for pool in pools]
        raise ValueError(
            "formal execution requires every loaded BLAS pool to report "
            f"num_threads=1; observed={counts}"
        )


def _require_formal_blas_thread_environment() -> dict[str, Any]:
    if any(value != "1" for value in _blas_thread_environment().values()):
        raise ValueError(
            "formal execution requires OPENBLAS_NUM_THREADS=1, "
            "OMP_NUM_THREADS=1, and MKL_NUM_THREADS=1 before Python starts"
        )
    runtime_identity = _blas_runtime_identity()
    _validate_blas_runtime_identity(runtime_identity, require_single_thread=True)
    return runtime_identity


def _validate_config(config: AllAutosomeConfig) -> None:
    if (
        not config.seeds
        or any(type(seed) is not int for seed in config.seeds)
        or len(set(config.seeds)) != len(config.seeds)
    ):
        raise ValueError("seeds must be non-empty and unique")
    if any(seed < 0 for seed in config.seeds):
        raise ValueError("seeds must be non-negative integers")
    if any(type(value) is not int for value in config.component_grid):
        raise ValueError("component_grid entries must be exact integers")
    if config.component_grid != COMPONENT_GRID:
        raise ValueError(f"component_grid is fixed to {COMPONENT_GRID}")
    if config.panel_fractions != PANEL_FRACTIONS:
        raise ValueError(f"panel_fractions is fixed to {PANEL_FRACTIONS}")
    if not math.isclose(config.test_fraction, 0.2, abs_tol=0.0, rel_tol=0.0):
        raise ValueError("test_fraction is fixed to 0.2")
    if config.loading_score != "l1_abs_sum":
        raise ValueError("loading_score is fixed to l1_abs_sum")
    if config.population_design != "five_superpopulation_200_each":
        raise ValueError("population_design is fixed to five_superpopulation_200_each")
    if config.analysis_status != "fixed_descriptive_development_evaluation":
        raise ValueError("analysis_status must remain descriptive")
    if config.analysis_mode not in {"fixed_protocol", "registered_shard", "tiny_smoke"}:
        raise ValueError(
            "analysis_mode must be fixed_protocol, registered_shard, or tiny_smoke"
        )
    if config.analysis_mode == "fixed_protocol" and config.seeds != PROTOCOL_SEEDS:
        raise ValueError("fixed_protocol seeds are frozen to 2025..2044")
    if config.analysis_mode == "registered_shard" and not set(config.seeds).issubset(
        set(PROTOCOL_SEEDS)
    ):
        raise ValueError("registered_shard seeds must be a subset of 2025..2044")
    if type(config.snp_chunk_size) is not int or config.snp_chunk_size < 32:
        raise ValueError("snp_chunk_size must be an integer at least 32")
    if (
        config.analysis_mode in {"fixed_protocol", "registered_shard"}
        and config.snp_chunk_size != 2048
    ):
        raise ValueError("formal protocol snp_chunk_size is frozen to 2048")


def _normalize_autosome(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("chr"):
        text = text[3:]
    try:
        number = int(text)
    except ValueError as error:
        raise ValueError(f"non-autosomal chromosome label: {value!r}") from error
    if not 1 <= number <= 22:
        raise ValueError(f"non-autosomal chromosome label: {value!r}")
    return str(number)


def _validate_input(
    data: base.GenotypeInput,
    *,
    require_formal_cohort: bool,
    chunk_size: int,
) -> None:
    raw = np.asanyarray(data.genotypes)
    if raw.ndim != 2 or min(raw.shape) < max(COMPONENT_GRID):
        raise ValueError("genotypes must be a samples x SNPs matrix with rank >= 8")
    if raw.dtype.kind not in "iuf":
        raise ValueError("genotypes must have a real numeric dtype")
    arrays = {
        "sample_ids": (data.sample_ids, raw.shape[0]),
        "population_labels": (data.population_labels, raw.shape[0]),
        "snp_ids": (data.snp_ids, raw.shape[1]),
        "chromosomes": (data.chromosomes, raw.shape[1]),
    }
    for name, (values, expected) in arrays.items():
        array = np.asarray(values, dtype=str)
        if array.shape != (expected,) or np.any(np.char.strip(array) == ""):
            raise ValueError(f"{name} must contain {expected} non-empty entries")
    if np.unique(np.asarray(data.sample_ids, dtype=str)).size != raw.shape[0]:
        raise ValueError("sample_ids must be unique")
    if np.unique(np.asarray(data.snp_ids, dtype=str)).size != raw.shape[1]:
        raise ValueError("snp_ids must be unique")
    labels = np.asarray(data.population_labels, dtype=str)
    if tuple(sorted(np.unique(labels).tolist())) != EXPECTED_POPULATIONS:
        raise ValueError(f"population labels must be exactly {EXPECTED_POPULATIONS}")
    counts = {label: int(np.count_nonzero(labels == label)) for label in labels}
    if any(counts[label] < 2 for label in EXPECTED_POPULATIONS):
        raise ValueError("every population requires at least two samples")
    if require_formal_cohort and (
        raw.shape[0] != 1000
        or any(counts[label] != 200 for label in EXPECTED_POPULATIONS)
    ):
        raise ValueError("formal input must contain exactly 200 samples per group")
    chromosomes = {
        _normalize_autosome(value) for value in np.asarray(data.chromosomes, dtype=str)
    }
    if require_formal_cohort and chromosomes != set(EXPECTED_AUTOSOMES):
        raise ValueError("formal input must cover autosomes 1 through 22 exactly")
    if not require_formal_cohort and len(chromosomes) < 2:
        raise ValueError("smoke input must span at least two autosomes")
    for start in range(0, raw.shape[1], int(chunk_size)):
        block = np.asarray(raw[:, start : start + int(chunk_size)], dtype=float)
        if np.any(np.isinf(block)):
            raise ValueError("genotypes may contain NaN missingness but not infinity")
        finite = block[np.isfinite(block)]
        if finite.size and np.any((finite < 0.0) | (finite > 2.0)):
            raise ValueError("finite genotype dosages must lie in [0, 2]")


def _pack_mask(mask: BoolArray) -> str:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    packed = np.packbits(values, bitorder="little")
    return b64encode(packed.tobytes()).decode("ascii")


def _decode_packed_mask(payload: str, width: int) -> NDArray[np.uint8]:
    if not isinstance(payload, str) or isinstance(width, bool) or width < 1:
        raise ValueError("invalid packed-mask receipt")
    try:
        raw = b64decode(payload.encode("ascii"), validate=True)
    except Exception as error:
        raise ValueError("packed mask is not canonical base64") from error
    expected_bytes = (int(width) + 7) // 8
    if len(raw) != expected_bytes or b64encode(raw).decode("ascii") != payload:
        raise ValueError("packed mask byte length/base64 mismatch")
    remainder = int(width) % 8
    if remainder and raw[-1] & (~((1 << remainder) - 1) & 0xFF):
        raise ValueError("packed mask has non-zero padding bits")
    return np.frombuffer(raw, dtype=np.uint8)


def _packed_popcount(values: NDArray[np.uint8]) -> int:
    return int(np.sum(_BYTE_POPCOUNT[np.asarray(values, dtype=np.uint8)]))


def _unpack_mask(payload: str, width: int) -> BoolArray:
    packed = _decode_packed_mask(payload, width)
    unpacked = np.unpackbits(packed, bitorder="little")
    mask = np.asarray(unpacked[: int(width)], dtype=bool)
    return mask


def _iter_column_ranges(width: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, int(width), int(chunk_size)):
        yield start, min(int(width), start + int(chunk_size))


def _training_block(
    raw: np.ndarray,
    rows: IntArray,
    start: int,
    stop: int,
) -> FloatArray:
    return np.asarray(raw[rows, start:stop], dtype=float)


def _transform_block(
    raw: np.ndarray,
    rows: IntArray,
    start: int,
    stop: int,
    mean_dosage: FloatArray,
    scale: FloatArray,
    retained_mask: BoolArray,
) -> FloatArray:
    local_retained = retained_mask[start:stop]
    block = _training_block(raw, rows, start, stop)[:, local_retained]
    means = mean_dosage[start:stop][local_retained]
    scales = scale[start:stop][local_retained]
    block = np.where(np.isfinite(block), block, means[None, :])
    block -= means[None, :]
    block /= scales[None, :]
    if not np.isfinite(block).all():
        raise RuntimeError("chunked smartpca transform produced non-finite values")
    return np.asarray(block, dtype=float)


def fit_chunked_top_pca(
    raw_genotypes: np.ndarray,
    train_indices: IntArray,
    test_indices: IntArray,
    *,
    n_components: int = 8,
    chunk_size: int = 2048,
) -> ChunkedPCAFit:
    """Fit exact PCA using O(n_train^2 + p*r) working memory.

    This is the same right singular subspace as a full SVD of the train-only
    smartpca-scaled matrix.  The matrix is traversed in SNP chunks: once for
    train statistics, once for the sample Gram matrix, and once for loadings
    and held-out scores.  It never materializes a float64 ``n x p`` copy.
    """

    raw = np.asanyarray(raw_genotypes)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if raw.ndim != 2 or min(raw.shape) < int(n_components):
        raise ValueError("raw_genotypes is too small for requested PCA rank")
    if train.ndim != 1 or test.ndim != 1 or not train.size or not test.size:
        raise ValueError("train_indices and test_indices must be non-empty vectors")
    if np.intersect1d(train, test).size:
        raise ValueError("train/test indices must be disjoint")
    if (
        np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= raw.shape[0])
        or np.any(test >= raw.shape[0])
    ):
        raise ValueError("train/test index outside genotype rows")
    if isinstance(n_components, bool) or not 1 <= int(n_components) <= train.size:
        raise ValueError("invalid n_components")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")

    width = int(raw.shape[1])
    means = np.empty(width, dtype=float)
    scales = np.empty(width, dtype=float)
    retained = np.zeros(width, dtype=bool)
    for start, stop in _iter_column_ranges(width, int(chunk_size)):
        block = _training_block(raw, train, start, stop)
        if np.any(np.isinf(block)):
            raise ValueError("train genotypes may contain NaN but not infinity")
        observed = np.isfinite(block)
        counts = observed.sum(axis=0)
        sums = np.where(observed, block, 0.0).sum(axis=0)
        local_means = np.divide(
            sums,
            counts,
            out=np.zeros(stop - start, dtype=float),
            where=counts > 0,
        )
        frequencies = local_means / 2.0
        local_scales = np.sqrt(2.0 * frequencies * (1.0 - frequencies))
        local_retained = (
            (counts > 0) & np.isfinite(local_scales) & (local_scales > 1e-12)
        )
        means[start:stop] = local_means
        scales[start:stop] = np.where(local_retained, local_scales, 1.0)
        retained[start:stop] = local_retained
    retained_original = np.flatnonzero(retained).astype(np.int64)
    if retained_original.size < int(n_components):
        raise ValueError("too few train-polymorphic SNPs for requested PCA rank")

    gram = np.zeros((train.size, train.size), dtype=float)
    for start, stop in _iter_column_ranges(width, int(chunk_size)):
        if not np.any(retained[start:stop]):
            continue
        block = _transform_block(raw, train, start, stop, means, scales, retained)
        gram += block @ block.T
    gram = (gram + gram.T) * 0.5
    gram_sha256 = array_sha256(gram)
    eigenvalues, left = linalg.eigh(
        gram,
        subset_by_index=(train.size - int(n_components), train.size - 1),
        check_finite=False,
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(np.asarray(eigenvalues[order], dtype=float), 0.0)
    left = np.asarray(left[:, order], dtype=float)
    singular_values = np.sqrt(eigenvalues)
    threshold = (
        np.finfo(float).eps
        * max(train.size, retained_original.size)
        * singular_values[0]
    )
    if np.any(singular_values <= threshold):
        raise ValueError("training matrix has insufficient numerical PCA rank")

    loadings = np.empty((retained_original.size, int(n_components)), dtype=float)
    test_scores = np.zeros((test.size, int(n_components)), dtype=float)
    retained_offset = 0
    for start, stop in _iter_column_ranges(width, int(chunk_size)):
        local_count = int(np.count_nonzero(retained[start:stop]))
        if local_count == 0:
            continue
        train_block = _transform_block(raw, train, start, stop, means, scales, retained)
        local_loadings = train_block.T @ left
        local_loadings /= singular_values[None, :]
        loadings[retained_offset : retained_offset + local_count] = local_loadings
        test_block = _transform_block(raw, test, start, stop, means, scales, retained)
        test_scores += test_block @ local_loadings
        retained_offset += local_count
    if retained_offset != retained_original.size:
        raise RuntimeError("retained loading assembly is incomplete")

    norms = np.linalg.norm(loadings, axis=0)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise RuntimeError("chunked PCA returned an invalid loading norm")
    loadings /= norms[None, :]
    train_scores = left * singular_values[None, :]
    train_scores /= norms[None, :]
    test_scores /= norms[None, :]
    for component in range(int(n_components)):
        pivot = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[pivot, component] < 0.0:
            loadings[:, component] *= -1.0
            train_scores[:, component] *= -1.0
            test_scores[:, component] *= -1.0
    if not np.allclose(
        loadings.T @ loadings,
        np.eye(int(n_components)),
        atol=3e-7,
        rtol=0.0,
    ):
        raise RuntimeError("chunked PCA loadings failed orthonormality")
    return ChunkedPCAFit(
        mean_dosage=np.asarray(means, dtype=float),
        scale=np.asarray(scales, dtype=float),
        retained_mask=np.asarray(retained, dtype=bool),
        retained_original=retained_original,
        loadings=np.ascontiguousarray(loadings),
        singular_values=np.asarray(singular_values, dtype=float),
        explained_variance_ratio=np.asarray(
            singular_values**2 / float(np.trace(gram)), dtype=float
        ),
        train_scores=np.asarray(train_scores, dtype=float),
        test_scores=np.asarray(test_scores, dtype=float),
        sample_gram_sha256=gram_sha256,
    )


def _resolved_panels(retained_count: int) -> list[dict[str, Any]]:
    """Resolve the frozen fractions before any labels or outcomes are scored."""

    if isinstance(retained_count, bool) or int(retained_count) < max(COMPONENT_GRID):
        raise ValueError("retained_count is too small for the fixed rank grid")
    resolved: list[dict[str, Any]] = []
    seen_counts: dict[int, str] = {}
    for fraction in PANEL_FRACTIONS:
        label = _panel_label(fraction)
        if fraction is None:
            count = int(retained_count)
            duplicate_of = None
            active = True
            selection_rule = "all_train_retained_no_loading_selection"
        else:
            count = max(
                max(COMPONENT_GRID),
                int(math.floor(float(fraction) * int(retained_count))),
            )
            count = min(int(retained_count), count)
            duplicate_of = seen_counts.get(count)
            active = duplicate_of is None
            if active:
                seen_counts[count] = label
            selection_rule = "top_training_pca_loading_fraction"
        resolved.append(
            {
                "panel_label": label,
                "requested_fraction": fraction,
                "effective_snp_count": count,
                "effective_fraction_of_train_retained": float(
                    count / int(retained_count)
                ),
                "active_grid_cell": active,
                "duplicate_effective_count_of": duplicate_of,
                "selection_rule": selection_rule,
                "resolution_uses_labels_or_outcomes": False,
            }
        )
    return resolved


def fit_chunked_panel_pca(
    raw_genotypes: np.ndarray,
    train_indices: IntArray,
    test_indices: IntArray,
    *,
    selected_mask: BoolArray,
    mean_dosage: FloatArray,
    scale: FloatArray,
    n_components: int = 8,
    chunk_size: int = 2048,
    _precomputed_gram: FloatArray | None = None,
    _capability: object | None = None,
) -> ChunkedPCAFit:
    """Refit PCA on one frozen panel without materializing its full matrix."""

    raw = np.asanyarray(raw_genotypes)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    selected = np.asarray(selected_mask, dtype=bool)
    means = np.asarray(mean_dosage, dtype=float)
    scales = np.asarray(scale, dtype=float)
    if raw.ndim != 2 or selected.shape != (raw.shape[1],):
        raise ValueError("selected_mask does not match genotype width")
    if means.shape != selected.shape or scales.shape != selected.shape:
        raise ValueError("preprocessing arrays do not match genotype width")
    selected_original = np.flatnonzero(selected).astype(np.int64)
    if selected_original.size < int(n_components):
        raise ValueError("selected panel is smaller than requested PCA rank")
    if np.any(scales[selected] <= 0.0) or not np.isfinite(scales[selected]).all():
        raise ValueError("selected panel contains an invalid frozen scale")

    if _precomputed_gram is None:
        gram = np.zeros((train.size, train.size), dtype=float)
        for start, stop in _iter_column_ranges(raw.shape[1], int(chunk_size)):
            local = selected[start:stop]
            if not np.any(local):
                continue
            train_block = _training_block(raw, train, start, stop)[:, local]
            local_means = means[start:stop][local]
            train_block = np.where(
                np.isfinite(train_block), train_block, local_means[None, :]
            )
            train_block -= local_means[None, :]
            train_block /= scales[start:stop][local][None, :]
            gram += train_block @ train_block.T
    else:
        if _capability is not _NESTED_GRAM_CAPABILITY:
            raise PermissionError("precomputed Gram is private to nested-panel fitting")
        gram = np.asarray(_precomputed_gram, dtype=float)
        if (
            gram.shape != (train.size, train.size)
            or not np.isfinite(gram).all()
            or not np.allclose(gram, gram.T, atol=1e-8, rtol=1e-12)
        ):
            raise ValueError("invalid precomputed nested-panel Gram")
    gram = (gram + gram.T) * 0.5
    total_sum_squares = float(np.trace(gram))
    if not math.isfinite(total_sum_squares) or total_sum_squares <= 0.0:
        raise ValueError("selected panel has zero or invalid total variance")
    eigenvalues, left = linalg.eigh(
        gram,
        subset_by_index=(train.size - int(n_components), train.size - 1),
        check_finite=False,
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(np.asarray(eigenvalues[order], dtype=float), 0.0)
    left = np.asarray(left[:, order], dtype=float)
    singular_values = np.sqrt(eigenvalues)
    threshold = (
        np.finfo(float).eps
        * max(train.size, selected_original.size)
        * singular_values[0]
    )
    if np.any(singular_values <= threshold):
        raise ValueError("selected panel has insufficient numerical PCA rank")

    loadings = np.empty((selected_original.size, int(n_components)), dtype=float)
    test_scores = np.zeros((test.size, int(n_components)), dtype=float)
    selected_offset = 0
    for start, stop in _iter_column_ranges(raw.shape[1], int(chunk_size)):
        local = selected[start:stop]
        local_count = int(np.count_nonzero(local))
        if local_count == 0:
            continue
        local_means = means[start:stop][local]
        local_scales = scales[start:stop][local]
        train_block = _training_block(raw, train, start, stop)[:, local]
        train_block = np.where(
            np.isfinite(train_block), train_block, local_means[None, :]
        )
        train_block -= local_means[None, :]
        train_block /= local_scales[None, :]
        local_loadings = train_block.T @ left
        local_loadings /= singular_values[None, :]
        loadings[selected_offset : selected_offset + local_count] = local_loadings
        test_block = _training_block(raw, test, start, stop)[:, local]
        test_block = np.where(np.isfinite(test_block), test_block, local_means[None, :])
        test_block -= local_means[None, :]
        test_block /= local_scales[None, :]
        test_scores += test_block @ local_loadings
        selected_offset += local_count
    if selected_offset != selected_original.size:
        raise RuntimeError("selected-panel loading assembly is incomplete")

    norms = np.linalg.norm(loadings, axis=0)
    if np.any(norms <= 0.0) or not np.isfinite(norms).all():
        raise RuntimeError("selected-panel PCA returned invalid loading norms")
    loadings /= norms[None, :]
    train_scores = left * singular_values[None, :]
    train_scores /= norms[None, :]
    test_scores /= norms[None, :]
    for component in range(int(n_components)):
        pivot = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[pivot, component] < 0.0:
            loadings[:, component] *= -1.0
            train_scores[:, component] *= -1.0
            test_scores[:, component] *= -1.0
    if not np.allclose(
        loadings.T @ loadings,
        np.eye(int(n_components)),
        atol=3e-7,
        rtol=0.0,
    ):
        raise RuntimeError("selected-panel PCA loadings failed orthonormality")
    return ChunkedPCAFit(
        mean_dosage=means,
        scale=scales,
        retained_mask=selected,
        retained_original=selected_original,
        loadings=np.ascontiguousarray(loadings),
        singular_values=np.asarray(singular_values, dtype=float),
        explained_variance_ratio=np.asarray(
            singular_values**2 / total_sum_squares, dtype=float
        ),
        train_scores=np.asarray(train_scores, dtype=float),
        test_scores=np.asarray(test_scores, dtype=float),
        sample_gram_sha256=array_sha256(gram),
    )


def fit_nested_chunked_panels(
    raw_genotypes: np.ndarray,
    train_indices: IntArray,
    test_indices: IntArray,
    *,
    ordered_original_indices: IntArray,
    panel_counts: Sequence[int],
    mean_dosage: FloatArray,
    scale: FloatArray,
    n_components: int = 8,
    chunk_size: int = 2048,
) -> dict[int, ChunkedPCAFit]:
    """Fit all nested panels in two genomic-order mmap passes.

    The first pass assigns each selected SNP to one of the disjoint increments
    between frozen thresholds and accumulates one delta Gram per increment.
    Cumulative Grams then provide the four exact panel eigensystems.  A second
    genomic-order pass recovers all four loading matrices and held-out scores
    together.  This keeps the arithmetic proportional to the selected widths
    while avoiding random ranked-column mmap access and twelve full-width
    recovery scans per seed.
    """

    raw = np.asanyarray(raw_genotypes)
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    ordered = np.asarray(ordered_original_indices, dtype=np.int64)
    counts = tuple(sorted({int(value) for value in panel_counts}))
    if (
        ordered.ndim != 1
        or np.unique(ordered).size != ordered.size
        or np.any(ordered < 0)
        or np.any(ordered >= raw.shape[1])
    ):
        raise ValueError("ordered_original_indices must be unique valid columns")
    if not counts or counts[-1] > ordered.size or counts[0] < int(n_components):
        raise ValueError("invalid nested panel counts")
    if isinstance(chunk_size, bool) or int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive")
    means = np.asarray(mean_dosage, dtype=float)
    scales = np.asarray(scale, dtype=float)
    if means.shape != (raw.shape[1],) or scales.shape != means.shape:
        raise ValueError("preprocessing arrays do not match genotype width")

    # group_for_column[g] identifies the newly admitted ranked prefix.  The
    # array is tiny (one signed byte per SNP) even at the formal 455,102 width.
    group_for_column = np.full(raw.shape[1], -1, dtype=np.int8)
    previous = 0
    for group_index, count in enumerate(counts):
        group_for_column[ordered[previous:count]] = group_index
        previous = count
    if np.any(group_for_column[ordered[: counts[-1]]] < 0):
        raise RuntimeError("nested group assignment is incomplete")

    delta_grams = [
        np.zeros((train.size, train.size), dtype=float) for _ in range(len(counts))
    ]
    for start, stop in _iter_column_ranges(raw.shape[1], int(chunk_size)):
        local_groups = group_for_column[start:stop]
        selected = local_groups >= 0
        if not np.any(selected):
            continue
        block = _training_block(raw, train, start, stop)[:, selected]
        local_means = means[start:stop][selected]
        local_scales = scales[start:stop][selected]
        if np.any(local_scales <= 0.0) or not np.isfinite(local_scales).all():
            raise ValueError("nested selection contains an invalid frozen scale")
        block = np.where(np.isfinite(block), block, local_means[None, :])
        block -= local_means[None, :]
        block /= local_scales[None, :]
        selected_groups = local_groups[selected]
        for group_index, delta in enumerate(delta_grams):
            in_group = selected_groups == group_index
            if np.any(in_group):
                local = block[:, in_group]
                delta += local @ local.T

    states: dict[int, dict[str, Any]] = {}
    cumulative_gram = np.zeros((train.size, train.size), dtype=float)
    for group_index, count in enumerate(counts):
        cumulative_gram += delta_grams[group_index]
        gram = (cumulative_gram + cumulative_gram.T) * 0.5
        total_sum_squares = float(np.trace(gram))
        if not math.isfinite(total_sum_squares) or total_sum_squares <= 0.0:
            raise ValueError("nested panel has zero or invalid total variance")
        eigenvalues, left = linalg.eigh(
            gram,
            subset_by_index=(train.size - int(n_components), train.size - 1),
            check_finite=False,
            driver="evr",
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(np.asarray(eigenvalues[order], dtype=float), 0.0)
        left = np.asarray(left[:, order], dtype=float)
        singular_values = np.sqrt(eigenvalues)
        threshold = np.finfo(float).eps * max(train.size, count) * singular_values[0]
        if np.any(singular_values <= threshold):
            raise ValueError("nested panel has insufficient numerical PCA rank")
        states[count] = {
            "group_index": group_index,
            "left": left,
            "singular_values": singular_values,
            "total_sum_squares": total_sum_squares,
            "sample_gram_sha256": array_sha256(gram),
            "selected_original": np.sort(ordered[:count]).astype(np.int64),
            "loadings": np.empty((count, int(n_components)), dtype=float),
            "test_scores": np.zeros((test.size, int(n_components)), dtype=float),
            "offset": 0,
        }

    # Recover every panel together so each row-major mmap chunk is read only
    # once for train and once for test, regardless of the number of thresholds.
    for start, stop in _iter_column_ranges(raw.shape[1], int(chunk_size)):
        local_groups = group_for_column[start:stop]
        selected = local_groups >= 0
        if not np.any(selected):
            continue
        local_means = means[start:stop][selected]
        local_scales = scales[start:stop][selected]
        train_block = _training_block(raw, train, start, stop)[:, selected]
        train_block = np.where(
            np.isfinite(train_block), train_block, local_means[None, :]
        )
        train_block -= local_means[None, :]
        train_block /= local_scales[None, :]
        test_block = _training_block(raw, test, start, stop)[:, selected]
        test_block = np.where(np.isfinite(test_block), test_block, local_means[None, :])
        test_block -= local_means[None, :]
        test_block /= local_scales[None, :]
        selected_groups = local_groups[selected]
        for count in counts:
            state = states[count]
            in_panel = selected_groups <= int(state["group_index"])
            local_count = int(np.count_nonzero(in_panel))
            if local_count == 0:
                continue
            local_loadings = train_block[:, in_panel].T @ state["left"]
            local_loadings /= state["singular_values"][None, :]
            offset = int(state["offset"])
            state["loadings"][offset : offset + local_count] = local_loadings
            state["test_scores"] += test_block[:, in_panel] @ local_loadings
            state["offset"] = offset + local_count

    output: dict[int, ChunkedPCAFit] = {}
    for count in counts:
        state = states[count]
        if int(state["offset"]) != count:
            raise RuntimeError("nested loading assembly is incomplete")
        loadings = np.asarray(state["loadings"], dtype=float)
        singular_values = np.asarray(state["singular_values"], dtype=float)
        norms = np.linalg.norm(loadings, axis=0)
        if np.any(norms <= 0.0) or not np.isfinite(norms).all():
            raise RuntimeError("nested-panel PCA returned invalid loading norms")
        loadings /= norms[None, :]
        train_scores = np.asarray(state["left"], dtype=float) * singular_values[None, :]
        train_scores /= norms[None, :]
        test_scores = np.asarray(state["test_scores"], dtype=float)
        test_scores /= norms[None, :]
        for component in range(int(n_components)):
            pivot = int(np.argmax(np.abs(loadings[:, component])))
            if loadings[pivot, component] < 0.0:
                loadings[:, component] *= -1.0
                train_scores[:, component] *= -1.0
                test_scores[:, component] *= -1.0
        if not np.allclose(
            loadings.T @ loadings,
            np.eye(int(n_components)),
            atol=3e-7,
            rtol=0.0,
        ):
            raise RuntimeError("nested-panel PCA loadings failed orthonormality")
        selected_mask = np.zeros(raw.shape[1], dtype=bool)
        selected_original = np.asarray(state["selected_original"], dtype=np.int64)
        selected_mask[selected_original] = True
        output[count] = ChunkedPCAFit(
            mean_dosage=means,
            scale=scales,
            retained_mask=selected_mask,
            retained_original=selected_original,
            loadings=np.ascontiguousarray(loadings),
            singular_values=singular_values,
            explained_variance_ratio=np.asarray(
                singular_values**2 / float(state["total_sum_squares"]), dtype=float
            ),
            train_scores=np.asarray(train_scores, dtype=float),
            test_scores=np.asarray(test_scores, dtype=float),
            sample_gram_sha256=str(state["sample_gram_sha256"]),
        )
    return output


def _exact_git_toplevel(repo_root: Path) -> Path | None:
    """Return the exact Git top-level, or ``None`` for a plain source snapshot."""

    root = Path(repo_root).resolve()
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    top = Path(completed.stdout.strip()).resolve()
    return top if top == root else None


def _git_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if _exact_git_toplevel(root) is None:
        return {
            "git_metadata_available": False,
            "head_commit": None,
            "head_tree": None,
            "tracked_dirty": None,
            "untracked": None,
            "repo_root_binding": "resolved_runner_parent_plain_source_snapshot",
        }
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    porcelain = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_metadata_available": True,
        "head_commit": head,
        "head_tree": tree,
        "tracked_dirty": any(not line.startswith("??") for line in porcelain),
        "untracked": any(line.startswith("??") for line in porcelain),
        "repo_root_binding": "resolved_runner_parent_equals_git_toplevel",
    }


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    expected_root = Path(__file__).resolve().parents[1]
    if root != expected_root:
        raise ValueError("repo_root must contain the executing runner")
    files: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for relative in SOURCE_FILE_ROLES:
        path = root / relative
        status = _regular_file(path, role=f"source file {relative}")
        files[relative] = _file_sha256(path)
        sizes[relative] = int(status.st_size)
    if (
        LEGACY_GITHUB_BASELINE_COMMIT != FROZEN_LEGACY_GITHUB_BASELINE_COMMIT
        or LEGACY_SOURCE_SHA256 != FROZEN_LEGACY_SOURCE_SHA256
    ):
        raise RuntimeError(
            "runtime genomics lineage constants changed from runner lock"
        )
    git_objects_available = _exact_git_toplevel(root) is not None
    historical_files: dict[str, dict[str, Any]] = {}
    for legacy_path, expected_hash in FROZEN_LEGACY_SOURCE_SHA256.items():
        observed_hash = expected_hash
        if git_objects_available:
            completed = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "show",
                    f"{FROZEN_LEGACY_GITHUB_BASELINE_COMMIT}:{legacy_path}",
                ],
                check=False,
                capture_output=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"cannot read historical Git object {legacy_path}: "
                    f"{completed.stderr[-300:].decode('utf-8', errors='replace')}"
                )
            observed_hash = _bytes_sha256(completed.stdout)
            if observed_hash != expected_hash:
                raise RuntimeError(
                    f"historical Git source hash mismatch: {legacy_path}"
                )
        historical_files[legacy_path] = {
            "sha256": observed_hash,
            "git_object": f"{FROZEN_LEGACY_GITHUB_BASELINE_COMMIT}:{legacy_path}",
            "git_object_bytes_verified_in_this_environment": git_objects_available,
        }
    return {
        "scope": "direct_scientific_and_serialization_dependencies",
        "import_graph_exact_cover_claimed": False,
        "files": files,
        "file_bytes": sizes,
        "roles": dict(SOURCE_FILE_ROLES),
        "historical_github_lineage": {
            "baseline_commit": FROZEN_LEGACY_GITHUB_BASELINE_COMMIT,
            "files": historical_files,
            "verification_mode": (
                "git_object_bytes_and_embedded_lock"
                if git_objects_available
                else "embedded_lock_only_plain_source_snapshot"
            ),
            "git_objects_and_hashes_verified": git_objects_available,
            "embedded_lineage_lock_verified": True,
            "plain_snapshot_does_not_reauthenticate_historical_git_objects": (
                not git_objects_available
            ),
            "extended_model_branch": [
                "smartpca_like_scaling",
                "ordinary_pca",
                "sum_absolute_pca_loading_ranking",
                "pca_refit_on_loading_panel",
            ],
            "historical_sparse_robust_randomized_branches_executed": False,
        },
        "git_identity": _git_identity(root),
    }


def _synthetic_input_identity(data: base.GenotypeInput) -> dict[str, Any]:
    labels = np.asarray(data.population_labels, dtype=str)
    return {
        "fixture": "synthetic_multi_autosome_smoke_only",
        "input_scope": "synthetic_multi_autosome_smoke",
        "orientation": "samples_by_snps",
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "genotypes_sha256": array_sha256(np.asarray(data.genotypes)),
        "sample_ids_sha256": array_sha256(np.asarray(data.sample_ids)),
        "snp_ids_sha256": array_sha256(np.asarray(data.snp_ids)),
        "chromosomes_sha256": array_sha256(np.asarray(data.chromosomes)),
        "population_labels_sha256": array_sha256(labels),
        "population_counts": {
            population: int(np.count_nonzero(labels == population))
            for population in EXPECTED_POPULATIONS
        },
        "ld_pruned": False,
        "external_source_manifest_anchor_verified": False,
    }


def _exact_mapping_keys(value: Any, keys: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"source manifest {role} schema mismatch")
    return value


def _validate_inert_file_receipt(
    value: Any,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    receipt = _exact_mapping_keys(value, {"bytes", "path", "sha256"}, role=role)
    if (
        not isinstance(receipt["path"], str)
        or not receipt["path"]
        or isinstance(receipt["bytes"], bool)
        or not isinstance(receipt["bytes"], int)
        or receipt["bytes"] < 1
        or not _is_sha256(receipt["sha256"])
        or (expected_sha256 is not None and receipt["sha256"] != expected_sha256)
    ):
        raise ValueError(f"source manifest {role} receipt mismatch")
    return receipt


def _validate_formal_manifest_structure(manifest: Mapping[str, Any]) -> None:
    required_top = {
        "cohort_anchor",
        "dataset",
        "downstream_label_boundary",
        "output",
        "plink2",
        "preparation",
        "schema_version",
        "sources",
        "variant_processing",
    }
    if set(manifest) != required_top:
        raise ValueError("all-autosome source manifest top-level schema mismatch")
    if manifest.get("schema_version") != FORMAL_MANIFEST_SCHEMA:
        raise ValueError("unexpected all-autosome source-manifest schema")
    preparation = _exact_mapping_keys(
        manifest.get("preparation"),
        {
            "disk_preflight_policy",
            "large_intermediate_policy",
            "memory_strategy",
            "retained_intermediates",
            "runtime_dependencies",
            "tool",
        },
        role="preparation",
    )
    _validate_inert_file_receipt(
        preparation["tool"],
        role="preparation.tool",
        expected_sha256=FROZEN_PREPARER_SOURCE_SHA256,
    )
    dependencies = _exact_mapping_keys(
        preparation["runtime_dependencies"],
        {"legacy_preparer", "source_downloader"},
        role="preparation.runtime_dependencies",
    )
    _validate_inert_file_receipt(
        dependencies["legacy_preparer"],
        role="preparation.runtime_dependencies.legacy_preparer",
        expected_sha256=FROZEN_LEGACY_PREPARER_SOURCE_SHA256,
    )
    _validate_inert_file_receipt(
        dependencies["source_downloader"],
        role="preparation.runtime_dependencies.source_downloader",
        expected_sha256=FROZEN_DOWNLOADER_SOURCE_SHA256,
    )
    if {
        "large_intermediate_policy": preparation["large_intermediate_policy"],
        "memory_strategy": preparation["memory_strategy"],
        "disk_preflight_policy": preparation["disk_preflight_policy"],
        "retained_intermediates": preparation["retained_intermediates"],
    } != {
        "large_intermediate_policy": (
            "one_chromosome_at_a_time; decompressed PGEN, QC fileset, RAW, "
            "prune.out, and logs deleted before the next chromosome"
        ),
        "memory_strategy": (
            "one chromosome float32 NPY memmap at a time; final genotype NPY "
            "member written row-wise across 22 disk blocks; validator extracts "
            "that member to a temporary memmap and checks it in row chunks"
        ),
        "disk_preflight_policy": (
            "per chromosome require 8*pgen_zst_bytes + 4*pvar_zst_bytes + "
            "8GiB free; before NPZ require 2*uncompressed genotype bytes + 3GiB; "
            "before validation extraction require member bytes + 1GiB"
        ),
        "retained_intermediates": "exact prune.in files only",
    }:
        raise ValueError("source manifest preparation policy mismatch")

    sources = _exact_mapping_keys(
        manifest.get("sources"),
        {
            "authentication_boundary",
            "download_directory",
            "external_input_anchors",
            "source_manifest",
        },
        role="sources",
    )
    anchors = _exact_mapping_keys(
        sources["external_input_anchors"],
        {
            "expected_download_completion_sha256",
            "expected_downloader_source_sha256",
            "expected_source_manifest_sha256",
        },
        role="sources.external_input_anchors",
    )
    if anchors != {
        "expected_download_completion_sha256": FROZEN_DOWNLOAD_COMPLETION_SHA256,
        "expected_downloader_source_sha256": FROZEN_DOWNLOADER_SOURCE_SHA256,
        "expected_source_manifest_sha256": FROZEN_UPSTREAM_SOURCE_MANIFEST_SHA256,
    }:
        raise ValueError("source manifest upstream anchors mismatch")
    source_manifest_receipt = _exact_mapping_keys(
        sources["source_manifest"],
        {"path", "payload_sha256", "sha256"},
        role="sources.source_manifest",
    )
    if (
        not isinstance(source_manifest_receipt["path"], str)
        or not source_manifest_receipt["path"]
        or source_manifest_receipt["sha256"] != FROZEN_UPSTREAM_SOURCE_MANIFEST_SHA256
        or not _is_sha256(source_manifest_receipt["payload_sha256"])
    ):
        raise ValueError("source manifest upstream source receipt mismatch")
    download = _exact_mapping_keys(
        sources["download_directory"],
        {
            "completion_manifest_sha256",
            "completion_payload_sha256",
            "file_count",
            "path",
            "total_bytes",
            "validation_status",
        },
        role="sources.download_directory",
    )
    if (
        download["completion_manifest_sha256"] != FROZEN_DOWNLOAD_COMPLETION_SHA256
        or not _is_sha256(download["completion_payload_sha256"])
        or download["validation_status"] != "valid_and_externally_anchored"
        or download["file_count"] != 45
        or not isinstance(download["total_bytes"], int)
        or download["total_bytes"] < 1
        or not isinstance(download["path"], str)
        or not download["path"]
        or not isinstance(sources["authentication_boundary"], str)
        or not sources["authentication_boundary"]
    ):
        raise ValueError("source manifest download receipt mismatch")

    plink = _exact_mapping_keys(
        manifest.get("plink2"), {"bytes", "path", "sha256", "version"}, role="plink2"
    )
    if (
        not isinstance(plink["path"], str)
        or not plink["path"]
        or plink["sha256"] != FROZEN_PLINK2_SHA256
        or plink["version"] != FROZEN_PLINK2_VERSION
        or not isinstance(plink["bytes"], int)
        or plink["bytes"] < 1
    ):
        raise ValueError("source manifest inert PLINK 2 receipt mismatch")
    dataset = manifest.get("dataset")
    expected_dataset = {
        "input_scope": "all_autosomes",
        "autosomes_present": list(EXPECTED_AUTOSOMES),
        "orientation": "samples_by_snps",
        "population_design": "five_superpopulation",
        "population_label_semantics": "1000_Genomes_superpopulation",
        "population_label_is_race": False,
        "ld_pruned": True,
    }
    if dataset != expected_dataset:
        raise ValueError("source manifest does not describe the required dataset")
    cohort = _exact_mapping_keys(
        manifest.get("cohort_anchor"),
        {
            "legacy_preparation_manifest",
            "psam_check",
            "selected_count",
            "selected_counts_by_group",
            "selected_records_sha256",
            "selected_sample_ids_sha256",
        },
        role="cohort_anchor",
    )
    legacy_cohort = _exact_mapping_keys(
        cohort["legacy_preparation_manifest"],
        {"path", "sha256"},
        role="cohort_anchor.legacy_preparation_manifest",
    )
    psam_check = _exact_mapping_keys(
        cohort["psam_check"],
        {
            "rows",
            "selected_sample_ids_sha256",
            "selected_samples_present",
            "superpopulation_column_checked",
        },
        role="cohort_anchor.psam_check",
    )
    if (
        not isinstance(legacy_cohort["path"], str)
        or not legacy_cohort["path"]
        or legacy_cohort["sha256"] != FROZEN_LEGACY_PREPARATION_MANIFEST_SHA256
        or cohort.get("selected_count") != 1000
        or cohort.get("selected_counts_by_group")
        != {population: 200 for population in EXPECTED_POPULATIONS}
        or cohort.get("selected_sample_ids_sha256") != FROZEN_SELECTED_SAMPLE_IDS_SHA256
        or cohort.get("selected_records_sha256") != FROZEN_SELECTED_RECORDS_SHA256
        or psam_check["selected_samples_present"] != 1000
        or psam_check["selected_sample_ids_sha256"] != FROZEN_SELECTED_SAMPLE_IDS_SHA256
        or psam_check["superpopulation_column_checked"] is not True
        or not isinstance(psam_check["rows"], int)
        or psam_check["rows"] < 1000
    ):
        raise ValueError("source manifest frozen-cohort anchor mismatch")
    processing = _exact_mapping_keys(
        manifest.get("variant_processing"),
        {
            "biallelic_snv_only",
            "cohort_scope",
            "duplicate_variant_policy",
            "hwe_filter",
            "ld_pruning",
            "maf",
            "maximum_variant_missingness",
            "padding_with_unpruned_variants",
            "per_chromosome",
            "policy",
            "population_labels_used_for_variant_qc_or_ld",
            "variant_target",
        },
        role="variant_processing",
    )
    ld = processing.get("ld_pruning")
    per_chromosome = processing.get("per_chromosome")
    if (
        processing.get("policy") != "genuine_qc_then_ld_prune_exact_prune_in_only"
        or processing.get("cohort_scope")
        != "same_frozen_1000_reference_samples_before_outer_split"
        or processing.get("population_labels_used_for_variant_qc_or_ld") is not False
        or processing.get("variant_target") is not None
        or processing.get("padding_with_unpruned_variants") is not False
        or processing.get("hwe_filter") is not None
        or processing.get("maf") != 0.05
        or processing.get("maximum_variant_missingness") != 0.05
        or processing.get("biallelic_snv_only") is not True
        or processing.get("duplicate_variant_policy") != "plink2_rm_dup_exclude_all"
        or not isinstance(ld, Mapping)
        or ld
        != {
            "method": "plink2_indep_pairwise",
            "window_variants": 50,
            "step_variants": 5,
            "r2_threshold": 0.2,
            "indep_order": 2,
        }
        or not isinstance(per_chromosome, Mapping)
        or set(per_chromosome) != set(EXPECTED_AUTOSOMES)
    ):
        raise ValueError("source manifest QC/LD protocol mismatch")
    for chromosome in EXPECTED_AUTOSOMES:
        receipt = _exact_mapping_keys(
            per_chromosome[chromosome],
            {
                "commands",
                "ephemeral_artifacts",
                "ld_pruned_variant_count",
                "qc_variant_count",
                "retained_prune_in",
                "source_inputs",
            },
            role=f"variant_processing.per_chromosome[{chromosome}]",
        )
        retained = _exact_mapping_keys(
            receipt["retained_prune_in"],
            {"bytes", "path", "sha256", "variant_count", "variant_ids_sha256"},
            role=f"chromosome {chromosome} retained_prune_in",
        )
        if (
            retained.get("path") != f"prune_in/chr{chromosome}.prune.in"
            or not isinstance(retained.get("bytes"), int)
            or retained["bytes"] < 1
            or not isinstance(retained.get("variant_count"), int)
            or retained["variant_count"] < 1
            or not _is_sha256(retained.get("sha256"))
            or not _is_sha256(retained.get("variant_ids_sha256"))
            or receipt["ld_pruned_variant_count"] != retained["variant_count"]
            or not isinstance(receipt["qc_variant_count"], int)
            or receipt["qc_variant_count"] < retained["variant_count"]
        ):
            raise ValueError(f"chromosome {chromosome} prune.in receipt mismatch")
        sources_for_chromosome = _exact_mapping_keys(
            receipt["source_inputs"],
            {"pgen_zst", "psam", "pvar_zst"},
            role=f"chromosome {chromosome} source_inputs",
        )
        for source_role, source_receipt in sources_for_chromosome.items():
            _validate_inert_file_receipt(
                source_receipt,
                role=f"chromosome {chromosome} source_inputs.{source_role}",
            )
        ephemeral = _exact_mapping_keys(
            receipt["ephemeral_artifacts"],
            {"decompressed_pgen", "export_raw", "qc_pvar"},
            role=f"chromosome {chromosome} ephemeral_artifacts",
        )
        for artifact_role, artifact in ephemeral.items():
            artifact_receipt = _exact_mapping_keys(
                artifact,
                {"bytes", "sha256"},
                role=f"chromosome {chromosome} {artifact_role}",
            )
            if (
                not isinstance(artifact_receipt["bytes"], int)
                or artifact_receipt["bytes"] < 1
                or not _is_sha256(artifact_receipt["sha256"])
            ):
                raise ValueError(f"chromosome {chromosome} ephemeral receipt mismatch")
        commands = receipt["commands"]
        if not isinstance(commands, list) or len(commands) != 4:
            raise ValueError(f"chromosome {chromosome} command receipts mismatch")
        for command in commands:
            command_receipt = _exact_mapping_keys(
                command,
                {
                    "argv",
                    "cwd",
                    "environment_overrides",
                    "returncode",
                    "role",
                    "stderr_sha256",
                    "stdout_sha256",
                },
                role=f"chromosome {chromosome} inert command",
            )
            if (
                not isinstance(command_receipt["argv"], list)
                or not command_receipt["argv"]
                or not all(
                    isinstance(value, str) and value
                    for value in command_receipt["argv"]
                )
                or not isinstance(command_receipt["cwd"], str)
                or not command_receipt["cwd"]
                or command_receipt["environment_overrides"]
                != {"LANG": "C", "LC_ALL": "C", "TZ": "UTC"}
                or command_receipt["returncode"] != 0
                or not isinstance(command_receipt["role"], str)
                or not command_receipt["role"]
                or not _is_sha256(command_receipt["stderr_sha256"])
                or not _is_sha256(command_receipt["stdout_sha256"])
            ):
                raise ValueError(
                    f"chromosome {chromosome} inert command receipt mismatch"
                )
    boundary = manifest.get("downstream_label_boundary")
    if not isinstance(boundary, Mapping) or (
        boundary.get("population_labels_used_for_variant_qc_or_ld") is not False
        or boundary.get("panel_selection_used_population_labels") is not False
        or set(boundary.get("forbidden", ()))
        != {
            "preprocessing_fit",
            "component_fit",
            "loading_based_panel_ranking",
            "panel_refit",
        }
    ):
        raise ValueError("source manifest label-boundary receipt mismatch")
    output = manifest.get("output")
    required_output = {
        "array_receipts",
        "array_sha256",
        "bytes",
        "dtype",
        "missing_count",
        "npz_keys",
        "npz_container",
        "path",
        "sha256",
        "shape",
        "total_qc_variants",
    }
    if not isinstance(output, Mapping) or set(output) != required_output:
        raise ValueError("source manifest output schema mismatch")
    arrays = output.get("array_sha256")
    array_receipts = output.get("array_receipts")
    expected_container = {
        "format": "zip_npz",
        "compression": "ZIP_DEFLATED",
        "compresslevel": 6,
        "member_order": [f"{key}.npy" for key in NPZ_KEYS],
        "member_timestamp": [1980, 1, 1, 0, 0, 0],
        "member_mode_octal": "0600",
        "genotypes_force_zip64": True,
    }
    if (
        output.get("path") != "genomics_aim_all_autosomes.npz"
        or output.get("npz_keys") != list(NPZ_KEYS)
        or output.get("npz_container") != expected_container
        or output.get("shape") != list(FROZEN_FORMAL_SHAPE)
        or output.get("dtype") != np.dtype("float32").str
        or output.get("missing_count") != 0
        or output.get("total_qc_variants") != FROZEN_FORMAL_TOTAL_QC_VARIANTS
        or output.get("sha256") != FROZEN_FORMAL_NPZ_SHA256
        or output.get("bytes") != FROZEN_FORMAL_NPZ_BYTES
        or not isinstance(arrays, Mapping)
        or set(arrays) != set(NPZ_KEYS)
        or dict(arrays) != FROZEN_PREPARED_ARRAY_SHA256
        or not isinstance(array_receipts, Mapping)
        or set(array_receipts) != set(NPZ_KEYS)
    ):
        raise ValueError("source manifest output receipt mismatch")
    expected_shapes = {
        "genotypes": output["shape"],
        "sample_ids": [1000],
        "population_labels": [1000],
        "snp_ids": [output["shape"][1]],
        "chromosomes": [output["shape"][1]],
    }
    expected_dtypes = {
        "chromosomes": "<U2",
        "genotypes": "<f4",
        "population_labels": "<U3",
        "sample_ids": "<U7",
        "snp_ids": "<U16",
    }
    for key in NPZ_KEYS:
        receipt = array_receipts[key]
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"sha256", "shape", "dtype"}
            or receipt.get("sha256") != arrays[key]
            or receipt.get("shape") != expected_shapes[key]
            or receipt.get("dtype") != expected_dtypes[key]
        ):
            raise ValueError(f"source manifest array receipt mismatch: {key}")


def _zip_members(npz_path: Path) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(npz_path, "r") as archive:
        infos = archive.infolist()
    expected_names = [f"{key}.npy" for key in NPZ_KEYS]
    names = [info.filename for info in infos]
    if names != expected_names or len(set(names)) != len(names):
        raise ValueError(
            f"NPZ members must be exactly ordered as {expected_names}; got {names}"
        )
    for info in infos:
        if (
            info.is_dir()
            or info.flag_bits & 0x1
            or info.compress_type != zipfile.ZIP_DEFLATED
            or info.date_time != (1980, 1, 1, 0, 0, 0)
            or ((info.external_attr >> 16) & 0o777) != 0o600
            or info.create_system != 3
            or (info.filename == "genotypes.npy" and info.extract_version != 45)
        ):
            raise ValueError(f"NPZ member is not canonical: {info.filename}")
    return {info.filename: info for info in infos}


@contextmanager
def load_formal_input(
    genotype_npz: Path,
    source_manifest: Path,
    *,
    expected_source_manifest_sha256: str,
    scratch_root: Path | None = None,
) -> Iterator[tuple[base.GenotypeInput, dict[str, Any]]]:
    """Load canonical NPZ with only its genotype member extracted/memory-mapped."""

    npz_path = Path(genotype_npz).resolve()
    manifest_path = Path(source_manifest).resolve()
    _regular_file(npz_path, role="all-autosome genotype NPZ")
    _regular_file(manifest_path, role="all-autosome source manifest")
    if expected_source_manifest_sha256 != FROZEN_FORMAL_MANIFEST_SHA256:
        raise ValueError(
            "expected_source_manifest_sha256 must equal the frozen formal anchor"
        )
    observed_manifest_sha = _file_sha256(manifest_path)
    if observed_manifest_sha != expected_source_manifest_sha256:
        raise ValueError("source manifest does not match the supplied external anchor")
    manifest = _strict_json_load(manifest_path)
    if not isinstance(manifest, Mapping):
        raise ValueError("source manifest must be a JSON object")
    _validate_formal_manifest_structure(manifest)
    output = manifest["output"]
    if (
        _file_sha256(npz_path) != output["sha256"]
        or npz_path.stat().st_size != output["bytes"]
        or output["sha256"] != FROZEN_FORMAL_NPZ_SHA256
    ):
        raise ValueError("source manifest does not bind the supplied NPZ")
    members = _zip_members(npz_path)

    scratch_parent = None if scratch_root is None else Path(scratch_root).resolve()
    if scratch_parent is not None:
        if scratch_parent.is_symlink() or not scratch_parent.is_dir():
            raise ValueError("scratch_root must be an existing real directory")
    extraction_parent = (
        Path(tempfile.gettempdir()).resolve()
        if scratch_parent is None
        else scratch_parent
    )
    required_scratch_bytes = int(members["genotypes.npy"].file_size) + 2**30
    if shutil.disk_usage(extraction_parent).free < required_scratch_bytes:
        raise OSError(
            "insufficient scratch for verified genotype extraction: need at least "
            f"{required_scratch_bytes} free bytes"
        )
    with tempfile.TemporaryDirectory(
        prefix="rspca-all-autosome-npz-",
        dir=None if scratch_parent is None else str(scratch_parent),
    ) as temporary_name:
        extracted = Path(temporary_name) / "genotypes.npy"
        with zipfile.ZipFile(npz_path, "r") as archive:
            with (
                archive.open("genotypes.npy", "r") as source,
                extracted.open("xb") as target,
            ):
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if (
            extracted.stat().st_size != members["genotypes.npy"].file_size
            or _file_sha256(extracted) != output["array_sha256"]["genotypes"]
        ):
            raise ValueError("extracted genotype NPY receipt mismatch")
        genotypes = np.load(extracted, mmap_mode="r", allow_pickle=False)
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {
                key: np.asarray(archive[key]) for key in NPZ_KEYS if key != "genotypes"
            }
        if (
            genotypes.shape != tuple(output["shape"])
            or genotypes.dtype.str != output["dtype"]
        ):
            raise ValueError("genotype member shape/dtype mismatch")
        for key, array in arrays.items():
            if _npy_array_sha256(array) != output["array_sha256"][key]:
                raise ValueError(f"NPZ array hash mismatch: {key}")
        data = base.GenotypeInput(
            genotypes=genotypes,
            sample_ids=np.asarray(arrays["sample_ids"], dtype=str),
            snp_ids=np.asarray(arrays["snp_ids"], dtype=str),
            chromosomes=np.asarray(arrays["chromosomes"], dtype=str),
            population_labels=np.asarray(arrays["population_labels"], dtype=str),
        )
        _validate_input(data, require_formal_cohort=True, chunk_size=2048)
        labels = np.asarray(data.population_labels, dtype=str)
        identity = {
            "input_scope": "all_autosomes",
            "orientation": "samples_by_snps",
            "samples": int(genotypes.shape[0]),
            "snps": int(genotypes.shape[1]),
            "genotype_npz_sha256": output["sha256"],
            "genotype_npz_bytes": int(output["bytes"]),
            "genotype_dtype": str(genotypes.dtype.str),
            "genotypes_npy_sha256": output["array_sha256"]["genotypes"],
            "sample_ids_sha256": array_sha256(data.sample_ids),
            "snp_ids_sha256": array_sha256(data.snp_ids),
            "chromosomes_sha256": array_sha256(data.chromosomes),
            "population_labels_sha256": array_sha256(labels),
            "population_counts": {
                population: int(np.count_nonzero(labels == population))
                for population in EXPECTED_POPULATIONS
            },
            "autosomes_present": list(EXPECTED_AUTOSOMES),
            "ld_pruned": True,
            "external_source_manifest_sha256": observed_manifest_sha,
            "expected_external_source_manifest_sha256": (
                expected_source_manifest_sha256
            ),
            "external_source_manifest_anchor_verified": True,
            "prepared_array_sha256": dict(output["array_sha256"]),
            "raw_input_memory_strategy": (
                "ZIP genotype NPY member extracted to temporary disk and read by mmap"
            ),
            "temporary_genotype_npy_bytes": int(members["genotypes.npy"].file_size),
            "minimum_scratch_free_bytes_per_worker": required_scratch_bytes,
            "upstream_qc_ld_scope": (
                "transductive_unsupervised_same_frozen_1000_before_outer_split"
            ),
            "runner_scaling_pca_loading_scope": "outer_train_only",
            "upstream_full_preparer_validation_replayed_by_runner": False,
            "frozen_manifest_schema_and_npz_binding_verified_by_runner": True,
        }
        yield data, identity


def _make_record(
    *,
    seed: int,
    n_components: int,
    panel_spec: Mapping[str, Any],
    split_receipt: Mapping[str, Any],
    selected_original: IntArray,
    retained_count: int,
    fit: ChunkedPCAFit,
    train_labels: StringArray,
    test_labels: StringArray,
) -> tuple[dict[str, Any], StringArray, StringArray, FloatArray]:
    predictions, classes, centroids = base._nearest_centroid(
        fit.train_scores[:, : int(n_components)],
        train_labels,
        fit.test_scores[:, : int(n_components)],
    )
    metrics = base._classification_metrics(test_labels, predictions, classes)
    requested_fraction = panel_spec["requested_fraction"]
    record = {
        "seed": int(seed),
        "evaluation": "pca_loading_fraction_panel",
        "n_components": int(n_components),
        "panel_label": str(panel_spec["panel_label"]),
        "requested_panel_fraction": requested_fraction,
        "available_train_retained_snps": int(retained_count),
        "selected_snp_count": int(selected_original.size),
        "effective_fraction_of_train_retained": float(
            selected_original.size / int(retained_count)
        ),
        "selected_original_indices_sha256": array_sha256(selected_original),
        "panel_feature_set_is_split_specific": True,
        "loading_selection_is_split_specific": requested_fraction is not None,
        "all_panel_uses_loading_selection": False,
        "selection_rule": str(panel_spec["selection_rule"]),
        "selected_snp_set_is_loading_only": requested_fraction is not None,
        "test_rows_used_for_runner_scaling_pca_or_ranking": False,
        "population_labels_used_for_panel_ranking": False,
        "runner_scaling_imputation_scope": "outer_train_only",
        "upstream_qc_ld_scope": (
            "transductive_unsupervised_same_frozen_1000_before_outer_split"
        ),
        "pca_fit_scope": "outer_train_only",
        "panel_refit_scope": (
            "outer_train_only" if requested_fraction is not None else "full_width_reuse"
        ),
        "classifier_fit_scope": "outer_train_centroids_after_panel_freeze",
        "solver": "exact_chunked_sample_gram_eigh",
        "solver_converged": True,
        "solver_n_iter": 1,
        "full_width_fit_reused": requested_fraction is None,
        "sample_gram_sha256": fit.sample_gram_sha256,
        "loadings_sha256": array_sha256(
            np.asarray(fit.loadings[:, : int(n_components)])
        ),
        "singular_values": [
            float(value) for value in fit.singular_values[: int(n_components)]
        ],
        "explained_variance_ratio": [
            float(value) for value in fit.explained_variance_ratio[: int(n_components)]
        ],
        "cumulative_explained_variance_ratio": float(
            np.sum(fit.explained_variance_ratio[: int(n_components)])
        ),
        "test_predictions_sha256": array_sha256(predictions),
        "test_predictions_dtype": str(predictions.dtype),
        **{
            key: split_receipt[key]
            for key in (
                "train_indices_sha256",
                "test_indices_sha256",
                "train_sample_ids_sha256",
                "test_sample_ids_sha256",
                "test_labels_sha256",
            )
        },
        **metrics,
    }
    return record, predictions, classes, centroids


def _summaries(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid: list[dict[str, Any]] = []
    populations: list[dict[str, Any]] = []
    for n_components in COMPONENT_GRID:
        all_by_seed = {
            int(row["seed"]): row
            for row in records
            if row["n_components"] == n_components and row["panel_label"] == "all"
        }
        for fraction in PANEL_FRACTIONS:
            panel_label = _panel_label(fraction)
            rows = sorted(
                (
                    row
                    for row in records
                    if row["n_components"] == n_components
                    and row["panel_label"] == panel_label
                ),
                key=lambda row: int(row["seed"]),
            )
            if not rows:
                continue
            balanced = np.asarray(
                [row["balanced_accuracy"] for row in rows], dtype=float
            )
            accuracies = np.asarray([row["accuracy"] for row in rows], dtype=float)
            macro_f1 = np.asarray([row["macro_f1"] for row in rows], dtype=float)
            deltas = np.asarray(
                [
                    row["balanced_accuracy"]
                    - all_by_seed[int(row["seed"])]["balanced_accuracy"]
                    for row in rows
                ],
                dtype=float,
            )
            grid.append(
                {
                    "n_components": int(n_components),
                    "panel_label": panel_label,
                    "requested_panel_fraction": fraction,
                    "evaluated_seeds": [int(row["seed"]) for row in rows],
                    "seeds": len(rows),
                    "minimum_selected_snp_count": int(
                        min(row["selected_snp_count"] for row in rows)
                    ),
                    "maximum_selected_snp_count": int(
                        max(row["selected_snp_count"] for row in rows)
                    ),
                    "mean_effective_fraction_of_train_retained": float(
                        np.mean(
                            [
                                row["effective_fraction_of_train_retained"]
                                for row in rows
                            ]
                        )
                    ),
                    "mean_accuracy": float(np.mean(accuracies)),
                    "mean_balanced_accuracy": float(np.mean(balanced)),
                    "sample_std_balanced_accuracy": float(
                        np.std(balanced, ddof=1) if len(rows) > 1 else 0.0
                    ),
                    "minimum_balanced_accuracy": float(np.min(balanced)),
                    "maximum_balanced_accuracy": float(np.max(balanced)),
                    "mean_macro_f1": float(np.mean(macro_f1)),
                    "mean_paired_balanced_accuracy_delta_vs_all": float(
                        np.mean(deltas)
                    ),
                    "sample_std_paired_balanced_accuracy_delta_vs_all": float(
                        np.std(deltas, ddof=1) if len(rows) > 1 else 0.0
                    ),
                    "descriptive_only_no_qualification_gate": True,
                }
            )
            for population in EXPECTED_POPULATIONS:
                metrics = [
                    next(
                        item
                        for item in row["per_population_metrics"]
                        if item["population"] == population
                    )
                    for row in rows
                ]
                recalls = np.asarray([item["recall"] for item in metrics], dtype=float)
                populations.append(
                    {
                        "n_components": int(n_components),
                        "panel_label": panel_label,
                        "requested_panel_fraction": fraction,
                        "population": population,
                        "seeds": len(rows),
                        "pooled_support": int(
                            sum(int(item["support"]) for item in metrics)
                        ),
                        "pooled_true_positive": int(
                            sum(int(item["true_positive"]) for item in metrics)
                        ),
                        "mean_recall": float(np.mean(recalls)),
                        "minimum_recall": float(np.min(recalls)),
                        "maximum_recall": float(np.max(recalls)),
                    }
                )
    return grid, populations


def _selection_masks_from_evidence(
    evidence: Sequence[Mapping[str, Any]],
    *,
    width: int,
) -> dict[tuple[int, int, str], NDArray[np.uint8]]:
    """Validate each selection transiently and retain only packed bitsets."""

    output: dict[tuple[int, int, str], NDArray[np.uint8]] = {}
    for row in evidence:
        key = (int(row["seed"]), int(row["n_components"]), str(row["panel_label"]))
        if key in output:
            raise ValueError(f"duplicate panel-selection evidence: {key}")
        packed = _decode_packed_mask(
            str(row["selected_mask_packbits_base64"]), int(width)
        )
        mask = np.unpackbits(packed, bitorder="little", count=int(width)).astype(
            bool, copy=False
        )
        indices = np.flatnonzero(mask).astype(np.int64)
        if (
            indices.size != int(row["selected_snp_count"])
            or array_sha256(indices) != row["selected_original_indices_sha256"]
        ):
            raise ValueError(f"panel-selection evidence hash/count mismatch: {key}")
        output[key] = packed
    return output


def _stability_summaries(
    evidence: Sequence[Mapping[str, Any]],
    *,
    width: int,
    predecoded_selections: (
        Mapping[tuple[int, int, str], NDArray[np.uint8]] | None
    ) = None,
) -> list[dict[str, Any]]:
    selections = (
        _selection_masks_from_evidence(evidence, width=width)
        if predecoded_selections is None
        else predecoded_selections
    )
    summaries: list[dict[str, Any]] = []
    for n_components in COMPONENT_GRID:
        for fraction in PANEL_FRACTIONS:
            panel_label = _panel_label(fraction)
            keyed = sorted(
                (
                    (seed, packed)
                    for (seed, rank, label), packed in selections.items()
                    if rank == n_components and label == panel_label
                ),
                key=lambda item: item[0],
            )
            if not keyed:
                continue
            seeds = [seed for seed, _ in keyed]
            arrays = [packed for _, packed in keyed]
            jaccards: list[float] = []
            for left_index in range(len(arrays)):
                for right_index in range(left_index + 1, len(arrays)):
                    intersection = _packed_popcount(
                        np.bitwise_and(arrays[left_index], arrays[right_index])
                    )
                    union = _packed_popcount(
                        np.bitwise_or(arrays[left_index], arrays[right_index])
                    )
                    jaccards.append(float(intersection / union) if union else 1.0)
            intersection_all = arrays[0].copy()
            union_all = arrays[0].copy()
            frequencies = np.zeros(int(width), dtype=np.uint16)
            for index, packed in enumerate(arrays):
                if index:
                    np.bitwise_and(intersection_all, packed, out=intersection_all)
                    np.bitwise_or(union_all, packed, out=union_all)
                frequencies += np.unpackbits(
                    packed, bitorder="little", count=int(width)
                )
            threshold = int(math.ceil(0.8 * len(arrays)))
            summaries.append(
                {
                    "n_components": int(n_components),
                    "panel_label": panel_label,
                    "requested_panel_fraction": fraction,
                    "evaluated_seeds": seeds,
                    "seeds": len(arrays),
                    "pairwise_comparisons": len(jaccards),
                    "mean_pairwise_jaccard": float(
                        np.mean(jaccards) if jaccards else 1.0
                    ),
                    "minimum_pairwise_jaccard": float(
                        np.min(jaccards) if jaccards else 1.0
                    ),
                    "maximum_pairwise_jaccard": float(
                        np.max(jaccards) if jaccards else 1.0
                    ),
                    "all_seed_intersection_size": _packed_popcount(intersection_all),
                    "all_seed_union_size": _packed_popcount(union_all),
                    "selected_in_at_least_80pct_splits": int(
                        np.count_nonzero(frequencies >= threshold)
                    ),
                    "stability_is_across_split_specific_panels": True,
                }
            )
    return summaries


def evaluate_all_autosome_stratification(
    data: base.GenotypeInput,
    config: AllAutosomeConfig,
    *,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute the fixed complete grid without selecting or qualifying a cell."""

    _validate_config(config)
    formal = config.analysis_mode in {"fixed_protocol", "registered_shard"}
    if formal:
        blas_runtime_identity = _require_formal_blas_thread_environment()
    else:
        blas_runtime_identity = _blas_runtime_identity()
    _validate_input(
        data,
        require_formal_cohort=formal,
        chunk_size=int(config.snp_chunk_size),
    )
    if formal and (
        input_identity.get("input_scope") != "all_autosomes"
        or input_identity.get("external_source_manifest_anchor_verified") is not True
    ):
        raise ValueError(
            "fixed protocol requires externally anchored all-autosome input"
        )
    raw = np.asanyarray(data.genotypes)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    max_components = max(config.component_grid)
    records: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    panel_evidence: list[dict[str, Any]] = []
    sample_evidence: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []

    for seed in config.seeds:
        print(
            json.dumps({"event": "all_autosome_seed_start", "seed": seed}),
            file=sys.stderr,
            flush=True,
        )
        train, test = stratified_split(labels, config.test_fraction, int(seed))
        full_fit = fit_chunked_top_pca(
            raw,
            train,
            test,
            n_components=max_components,
            chunk_size=int(config.snp_chunk_size),
        )
        retained = full_fit.retained_original
        resolutions = _resolved_panels(int(retained.size))
        active_resolutions = [row for row in resolutions if row["active_grid_cell"]]
        split_receipt = {
            "seed": int(seed),
            "train_indices_sha256": array_sha256(train),
            "train_indices_dtype": str(train.dtype),
            "test_indices_sha256": array_sha256(test),
            "test_indices_dtype": str(test.dtype),
            "train_sample_ids_sha256": array_sha256(sample_ids[train]),
            "train_sample_ids_dtype": str(sample_ids[train].dtype),
            "test_sample_ids_sha256": array_sha256(sample_ids[test]),
            "test_sample_ids_dtype": str(sample_ids[test].dtype),
            "test_labels_sha256": array_sha256(labels[test]),
            "test_labels_dtype": str(labels[test].dtype),
            "retained_snps": int(retained.size),
            "retained_mask_width": int(raw.shape[1]),
            "retained_mask_packbits_base64": _pack_mask(full_fit.retained_mask),
            "retained_original_indices_sha256": array_sha256(retained),
            "retained_snp_ids_sha256": array_sha256(data.snp_ids[retained]),
            "preprocessor_mean_sha256": array_sha256(full_fit.mean_dosage),
            "preprocessor_scale_sha256": array_sha256(full_fit.scale),
            "full_sample_gram_sha256": full_fit.sample_gram_sha256,
            "full_singular_values": [
                float(value) for value in full_fit.singular_values
            ],
            "full_explained_variance_ratio": [
                float(value) for value in full_fit.explained_variance_ratio
            ],
            "panel_resolution": resolutions,
            "panel_resolution_precedes_label_scoring": True,
        }
        split_receipts.append(split_receipt)

        for n_components in config.component_grid:
            ranking_retained, importance = base.rank_by_pca_loadings(
                full_fit.loadings, int(n_components)
            )
            ordered_original = retained[ranking_retained]
            finite_specs = [
                row
                for row in active_resolutions
                if row["requested_fraction"] is not None
            ]
            finite_counts = [int(row["effective_snp_count"]) for row in finite_specs]
            nested_fits = fit_nested_chunked_panels(
                raw,
                train,
                test,
                ordered_original_indices=ordered_original,
                panel_counts=finite_counts,
                mean_dosage=full_fit.mean_dosage,
                scale=full_fit.scale,
                n_components=max_components,
                chunk_size=int(config.snp_chunk_size),
            )
            ranking_record: dict[str, Any] = {
                "seed": int(seed),
                "n_components": int(n_components),
                "available_train_retained_snps": int(retained.size),
                "loading_importance_sha256": array_sha256(importance),
                "full_width_loading_prefix_sha256": array_sha256(
                    full_fit.loadings[:, : int(n_components)]
                ),
                "selection_rule": "descending_sum_abs_training_pca_loadings",
                "tie_break": "retained_column_index_ascending",
                "population_labels_used": False,
                "test_rows_used": False,
                "fraction_counts_resolved_before_scoring": True,
                "panel_labels": [row["panel_label"] for row in active_resolutions],
            }
            ranking_records.append(ranking_record)

            for panel_spec in active_resolutions:
                fraction = panel_spec["requested_fraction"]
                if fraction is None:
                    selected_original = retained
                    selected_mask = full_fit.retained_mask
                    fit = full_fit
                else:
                    count = int(panel_spec["effective_snp_count"])
                    selected_original = np.sort(ordered_original[:count]).astype(
                        np.int64
                    )
                    selected_mask = np.zeros(raw.shape[1], dtype=bool)
                    selected_mask[selected_original] = True
                    fit = nested_fits[count]
                    if not np.array_equal(fit.retained_original, selected_original):
                        raise RuntimeError("nested panel fit identity mismatch")
                selection_receipt = {
                    "seed": int(seed),
                    "n_components": int(n_components),
                    "panel_label": str(panel_spec["panel_label"]),
                    "requested_panel_fraction": fraction,
                    "available_train_retained_snps": int(retained.size),
                    "selected_snp_count": int(selected_original.size),
                    "effective_fraction_of_train_retained": float(
                        selected_original.size / retained.size
                    ),
                    "selected_mask_width": int(raw.shape[1]),
                    "selected_mask_packbits_base64": _pack_mask(selected_mask),
                    "selected_original_indices_sha256": array_sha256(selected_original),
                    "selection_rule": str(panel_spec["selection_rule"]),
                    "selection_used_population_labels": False,
                    "selection_used_test_rows": False,
                    "panel_feature_set_is_split_specific": True,
                    "loading_selection_is_split_specific": fraction is not None,
                }
                panel_evidence.append(selection_receipt)
                record, predictions, classes, centroids = _make_record(
                    seed=int(seed),
                    n_components=int(n_components),
                    panel_spec=panel_spec,
                    split_receipt=split_receipt,
                    selected_original=selected_original,
                    retained_count=int(retained.size),
                    fit=fit,
                    train_labels=labels[train],
                    test_labels=labels[test],
                )
                records.append(record)
                for position, sample_index in enumerate(test.tolist()):
                    sample_evidence.append(
                        {
                            "seed": int(seed),
                            "n_components": int(n_components),
                            "panel_label": str(panel_spec["panel_label"]),
                            "requested_panel_fraction": fraction,
                            "selected_snp_count": int(selected_original.size),
                            "selected_original_indices_sha256": record[
                                "selected_original_indices_sha256"
                            ],
                            "sample_position": int(position),
                            "sample_index": int(sample_index),
                            "sample_id": str(sample_ids[sample_index]),
                            "truth": str(labels[sample_index]),
                            "prediction": str(predictions[position]),
                            "correct": bool(
                                predictions[position] == labels[sample_index]
                            ),
                            "test_indices_sha256": split_receipt["test_indices_sha256"],
                            "test_indices_dtype": split_receipt["test_indices_dtype"],
                            "test_sample_ids_sha256": split_receipt[
                                "test_sample_ids_sha256"
                            ],
                            "test_sample_ids_dtype": split_receipt[
                                "test_sample_ids_dtype"
                            ],
                            "test_labels_sha256": split_receipt["test_labels_sha256"],
                            "test_labels_dtype": split_receipt["test_labels_dtype"],
                            "test_predictions_sha256": record[
                                "test_predictions_sha256"
                            ],
                            "test_predictions_dtype": record["test_predictions_dtype"],
                        }
                    )

                if seed == config.seeds[0] and n_components == max_components:
                    for position, sample_index in enumerate(test.tolist()):
                        coordinates.append(
                            {
                                "row_type": "held_out_sample",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": str(panel_spec["panel_label"]),
                                "requested_panel_fraction": fraction,
                                "selected_snp_count": int(selected_original.size),
                                "effective_fraction_of_train_retained": float(
                                    selected_original.size / retained.size
                                ),
                                "panel_feature_set_is_split_specific": True,
                                "loading_selection_is_split_specific": (
                                    fraction is not None
                                ),
                                "sample_index": int(sample_index),
                                "sample_id": str(sample_ids[sample_index]),
                                "population": str(labels[sample_index]),
                                "truth": str(labels[sample_index]),
                                "prediction": str(predictions[position]),
                                "correct": bool(
                                    predictions[position] == labels[sample_index]
                                ),
                                **{
                                    f"component_{component + 1}": float(
                                        fit.test_scores[position, component]
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )
                    if tuple(classes.tolist()) != EXPECTED_POPULATIONS:
                        raise RuntimeError("unexpected centroid population order")
                    for class_index, population in enumerate(classes.tolist()):
                        coordinates.append(
                            {
                                "row_type": "train_centroid",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": str(panel_spec["panel_label"]),
                                "requested_panel_fraction": fraction,
                                "selected_snp_count": int(selected_original.size),
                                "effective_fraction_of_train_retained": float(
                                    selected_original.size / retained.size
                                ),
                                "panel_feature_set_is_split_specific": True,
                                "loading_selection_is_split_specific": (
                                    fraction is not None
                                ),
                                "sample_index": None,
                                "sample_id": None,
                                "population": population,
                                "truth": None,
                                "prediction": None,
                                "correct": None,
                                **{
                                    f"component_{component + 1}": float(
                                        centroids[class_index, component]
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )
        print(
            json.dumps(
                {
                    "event": "all_autosome_seed_complete",
                    "seed": seed,
                    "retained_snps": int(retained.size),
                    "full_width_pca_fits": 1,
                    "nested_fraction_panel_groups": len(COMPONENT_GRID),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    grid_summaries, population_summaries = _summaries(records)
    stability = _stability_summaries(panel_evidence, width=int(raw.shape[1]))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_policy": {
            "status": "fixed_descriptive_development_evaluation",
            "confirmatory": False,
            "best_configuration_selected": False,
            "headline_configuration_selected": False,
            "target_metric_or_quality_gate_used": False,
            "result_triggered_parameter_change_allowed": False,
            "all_resolved_grid_cells_reported": True,
            "population_labels_are_race": False,
            "population_labels_used_for_split": True,
            "population_labels_used_for_training_centroids": True,
            "population_labels_used_for_held_out_scoring": True,
            "population_labels_used_for_runner_scaling_pca_or_loading_ranking": False,
            "test_rows_used_for_runner_scaling_pca_or_loading_ranking": False,
            "upstream_qc_ld_uses_all_frozen_cohort_genotypes": True,
            "upstream_qc_ld_scope": (
                "transductive_unsupervised_same_frozen_1000_before_outer_split"
            ),
            "all_panel_uses_loading_selection": False,
            "finite_loading_selections_are_split_specific": True,
            "all_panel_feature_set_uses_split_train_polymorphic_mask": True,
            "panel_grid_interpretation": (
                "fixed train-retained fractions test compression sensitivity; "
                "they are not a single fixed AIM list"
            ),
            "multi_pc_interpretation": (
                "classification uses all requested PCs; PC1/2 and PC3/4 are "
                "registered two-dimensional views, not the full decision space"
            ),
        },
        "config": _config_payload(config),
        "method_scope": dict(METHOD_SCOPE),
        "input_identity": dict(input_identity),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "execution_provenance": {
            "execution_mode": config.analysis_mode,
            "registered_seed_bank": list(PROTOCOL_SEEDS),
            "included_seeds": list(config.seeds),
            "exact_registered_seed_cover": (
                config.analysis_mode == "fixed_protocol"
                and config.seeds == PROTOCOL_SEEDS
            ),
            "merged_from_shards": False,
            "source_shard_analysis_sha256": [],
        },
        "memory_and_solver_policy": {
            "raw_npz_strategy": input_identity.get(
                "raw_input_memory_strategy", "in_memory_synthetic_smoke"
            ),
            "train_preprocessing": "SNP chunks; no full standardized matrix",
            "full_pca": "chunked sample Gram plus V equals X transpose U over s",
            "finite_panel_pca": (
                "two genomic-order passes per rank: delta Grams then joint loading "
                "and held-out-score recovery for all fixed fractions"
            ),
            "maximum_resident_dense_panel_matrix": None,
            "seed_parallelism": 1,
            "reason": "bound per-process RSS and preserve deterministic BLAS order",
            "blas_thread_environment": _blas_thread_environment(),
            "blas_build_identity": _blas_build_identity(),
            "blas_runtime_identity": blas_runtime_identity,
            "formal_blas_thread_policy": (
                "all three recorded variables and every detected BLAS runtime "
                "pool must equal 1"
            ),
        },
        "split_receipts": split_receipts,
        "ranking_records": ranking_records,
        "panel_selection_evidence": panel_evidence,
        "records": records,
        "grid_summaries": grid_summaries,
        "population_summaries": population_summaries,
        "panel_stability_summaries": stability,
        "sample_prediction_evidence": sample_evidence,
        "registered_seed_policy": {
            "seed": int(config.seeds[0]),
            "selection": "first_configured_seed_never_best_seed",
            "rank": int(max_components),
            "rank_policy": "fixed_maximum_grid_rank_never_result_selected",
            "plotted_component_pairs": [[1, 2], [3, 4]],
            "confusion_rank": int(max_components),
            "panels": [_panel_label(value) for value in PANEL_FRACTIONS],
            "finite_panels_are_independently_loading_selected_within_each_split": True,
            "all_panel_uses_no_loading_selection": True,
            "every_panel_feature_set_uses_split_train_retained_mask": True,
        },
        "registered_coordinates": coordinates,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_result_payload(payload)
    return payload


def _config_from_payload(value: Mapping[str, Any]) -> AllAutosomeConfig:
    if set(value) != {
        "seeds",
        "component_grid",
        "panel_fractions",
        "test_fraction",
        "loading_score",
        "population_design",
        "analysis_status",
        "analysis_mode",
        "snp_chunk_size",
    }:
        raise ValueError("result config schema mismatch")
    config = AllAutosomeConfig(
        seeds=tuple(value["seeds"]),
        component_grid=tuple(value["component_grid"]),
        panel_fractions=tuple(value["panel_fractions"]),
        test_fraction=value["test_fraction"],
        loading_score=value["loading_score"],
        population_design=value["population_design"],
        analysis_status=value["analysis_status"],
        analysis_mode=value["analysis_mode"],
        snp_chunk_size=value["snp_chunk_size"],
    )
    _validate_config(config)
    return config


def _record_metric_subset(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "class_order",
            "confusion_matrix_counts",
            "per_population_metrics",
        )
    }


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Strict self-consistency validation from row and bit-mask evidence."""

    required = {
        "schema_version",
        "analysis_policy",
        "config",
        "method_scope",
        "input_identity",
        "source_identity",
        "runtime_versions",
        "execution_provenance",
        "memory_and_solver_policy",
        "split_receipts",
        "ranking_records",
        "panel_selection_evidence",
        "records",
        "grid_summaries",
        "population_summaries",
        "panel_stability_summaries",
        "sample_prediction_evidence",
        "registered_seed_policy",
        "registered_coordinates",
        "analysis_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("all-autosome result schema mismatch")
    config_value = payload.get("config")
    if not isinstance(config_value, Mapping):
        raise ValueError("result config must be an object")
    config = _config_from_payload(config_value)
    policy = payload.get("analysis_policy")
    if not isinstance(policy, Mapping) or (
        policy.get("status") != "fixed_descriptive_development_evaluation"
        or policy.get("confirmatory") is not False
        or policy.get("best_configuration_selected") is not False
        or policy.get("headline_configuration_selected") is not False
        or policy.get("target_metric_or_quality_gate_used") is not False
        or policy.get("result_triggered_parameter_change_allowed") is not False
        or policy.get("all_resolved_grid_cells_reported") is not True
        or policy.get(
            "population_labels_used_for_runner_scaling_pca_or_loading_ranking"
        )
        is not False
        or policy.get("test_rows_used_for_runner_scaling_pca_or_loading_ranking")
        is not False
        or policy.get("upstream_qc_ld_uses_all_frozen_cohort_genotypes") is not True
        or policy.get("upstream_qc_ld_scope")
        != "transductive_unsupervised_same_frozen_1000_before_outer_split"
        or policy.get("all_panel_uses_loading_selection") is not False
        or policy.get("finite_loading_selections_are_split_specific") is not True
        or policy.get("all_panel_feature_set_uses_split_train_polymorphic_mask")
        is not True
    ):
        raise ValueError("analysis policy permits selection, leakage, or overclaiming")
    if payload.get("method_scope") != METHOD_SCOPE:
        raise ValueError("method scope changed from PCA-loading-only")
    memory_policy = payload.get("memory_and_solver_policy")
    if not isinstance(memory_policy, Mapping) or (
        memory_policy.get("maximum_resident_dense_panel_matrix") is not None
        or memory_policy.get("seed_parallelism") != 1
        or memory_policy.get("formal_blas_thread_policy")
        != (
            "all three recorded variables and every detected BLAS runtime "
            "pool must equal 1"
        )
        or set(memory_policy.get("blas_thread_environment", {}))
        != {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"}
        or set(memory_policy.get("blas_build_identity", {}))
        != {"name", "version", "openblas_configuration"}
        or (
            config.analysis_mode in {"fixed_protocol", "registered_shard"}
            and set(memory_policy["blas_thread_environment"].values()) != {"1"}
        )
    ):
        raise ValueError("memory/solver/BLAS policy mismatch")
    try:
        _validate_blas_runtime_identity(
            memory_policy.get("blas_runtime_identity"),
            require_single_thread=(
                config.analysis_mode in {"fixed_protocol", "registered_shard"}
            ),
        )
    except ValueError as error:
        raise ValueError("memory/solver/BLAS policy mismatch") from error
    execution = payload.get("execution_provenance")
    exact_cover = (
        config.analysis_mode == "fixed_protocol" and config.seeds == PROTOCOL_SEEDS
    )
    if not isinstance(execution, Mapping) or (
        execution.get("execution_mode") != config.analysis_mode
        or execution.get("registered_seed_bank") != list(PROTOCOL_SEEDS)
        or execution.get("included_seeds") != list(config.seeds)
        or execution.get("exact_registered_seed_cover") is not exact_cover
        or not isinstance(execution.get("merged_from_shards"), bool)
        or not isinstance(execution.get("source_shard_analysis_sha256"), list)
        or not all(
            _is_sha256(value)
            for value in execution.get("source_shard_analysis_sha256", [])
        )
        or (
            execution.get("merged_from_shards")
            and (
                not exact_cover
                or len(execution.get("source_shard_analysis_sha256", [])) < 2
            )
        )
        or (
            not execution.get("merged_from_shards")
            and execution.get("source_shard_analysis_sha256") != []
        )
    ):
        raise ValueError("execution/shard provenance mismatch")
    identity = payload.get("input_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("input identity must be an object")
    samples = identity.get("samples")
    width = identity.get("snps")
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or samples < 10
        or isinstance(width, bool)
        or not isinstance(width, int)
        or width < max(COMPONENT_GRID)
    ):
        raise ValueError("input shape receipt is invalid")
    if config.analysis_mode in {"fixed_protocol", "registered_shard"} and (
        identity.get("input_scope") != "all_autosomes"
        or identity.get("samples") != 1000
        or identity.get("snps") != FROZEN_FORMAL_SHAPE[1]
        or identity.get("population_counts")
        != {population: 200 for population in EXPECTED_POPULATIONS}
        or identity.get("external_source_manifest_anchor_verified") is not True
        or identity.get("external_source_manifest_sha256")
        != FROZEN_FORMAL_MANIFEST_SHA256
        or identity.get("expected_external_source_manifest_sha256")
        != FROZEN_FORMAL_MANIFEST_SHA256
        or identity.get("genotype_npz_sha256") != FROZEN_FORMAL_NPZ_SHA256
        or identity.get("genotype_npz_bytes") != FROZEN_FORMAL_NPZ_BYTES
        or identity.get("prepared_array_sha256") != FROZEN_PREPARED_ARRAY_SHA256
        or identity.get("sample_ids_sha256")
        != "5184e0cea16cebd4180d0532b6eb8c63da130f49257c6cc8bd592c206eebc497"
        or identity.get("population_labels_sha256")
        != "88f7b16a8fb78093f05bf0ba107485b587a51836edf24eec4039c97e98925c25"
        or identity.get("upstream_qc_ld_scope")
        != "transductive_unsupervised_same_frozen_1000_before_outer_split"
        or identity.get("runner_scaling_pca_loading_scope") != "outer_train_only"
    ):
        raise ValueError("formal result lacks the anchored 1000-person cohort")

    split_receipts = payload.get("split_receipts")
    if not isinstance(split_receipts, list) or len(split_receipts) != len(config.seeds):
        raise ValueError("split-receipt count mismatch")
    split_by_seed: dict[int, Mapping[str, Any]] = {}
    retained_masks_by_seed: dict[int, NDArray[np.uint8]] = {}
    retained_counts_by_seed: dict[int, int] = {}
    active_labels_by_seed: dict[int, tuple[str, ...]] = {}
    for receipt in split_receipts:
        if not isinstance(receipt, Mapping):
            raise ValueError("split receipt must be an object")
        seed = int(receipt.get("seed", -1))
        if seed in split_by_seed or seed not in config.seeds:
            raise ValueError("split receipts do not exactly cover configured seeds")
        if receipt.get("retained_mask_width") != width:
            raise ValueError("retained-mask width mismatch")
        packed_mask = _decode_packed_mask(
            receipt.get("retained_mask_packbits_base64"), width
        )
        mask = np.unpackbits(packed_mask, bitorder="little", count=width).astype(
            bool, copy=False
        )
        retained_indices = np.flatnonzero(mask).astype(np.int64)
        if (
            retained_indices.size != receipt.get("retained_snps")
            or array_sha256(retained_indices)
            != receipt.get("retained_original_indices_sha256")
            or retained_indices.size < max(COMPONENT_GRID)
        ):
            raise ValueError("retained-SNP receipt mismatch")
        for key in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
            "retained_snp_ids_sha256",
            "preprocessor_mean_sha256",
            "preprocessor_scale_sha256",
            "full_sample_gram_sha256",
        ):
            if not _is_sha256(receipt.get(key)):
                raise ValueError(f"invalid split hash receipt: {key}")
        expected_resolution = _resolved_panels(int(retained_indices.size))
        if receipt.get("panel_resolution") != expected_resolution:
            raise ValueError("panel fractions were not resolved by the fixed rule")
        if receipt.get("panel_resolution_precedes_label_scoring") is not True:
            raise ValueError("panel resolution timing receipt is false")
        active_labels = tuple(
            row["panel_label"] for row in expected_resolution if row["active_grid_cell"]
        )
        split_by_seed[seed] = receipt
        retained_masks_by_seed[seed] = packed_mask
        retained_counts_by_seed[seed] = int(retained_indices.size)
        active_labels_by_seed[seed] = active_labels
    if tuple(split_by_seed) != config.seeds:
        raise ValueError("split receipts are not in configured seed order")

    ranking_records = payload.get("ranking_records")
    expected_rank_keys = {
        (int(seed), int(rank)) for seed in config.seeds for rank in COMPONENT_GRID
    }
    if not isinstance(ranking_records, list):
        raise ValueError("ranking_records must be a list")
    rank_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    for row in ranking_records:
        if not isinstance(row, Mapping):
            raise ValueError("ranking record must be an object")
        key = (int(row.get("seed", -1)), int(row.get("n_components", -1)))
        if key in rank_map or key not in expected_rank_keys:
            raise ValueError("ranking records do not exactly cover seed/rank grid")
        if (
            row.get("available_train_retained_snps") != retained_counts_by_seed[key[0]]
            or not _is_sha256(row.get("loading_importance_sha256"))
            or not _is_sha256(row.get("full_width_loading_prefix_sha256"))
            or row.get("selection_rule") != "descending_sum_abs_training_pca_loadings"
            or row.get("population_labels_used") is not False
            or row.get("test_rows_used") is not False
            or row.get("fraction_counts_resolved_before_scoring") is not True
            or tuple(row.get("panel_labels", ())) != active_labels_by_seed[key[0]]
        ):
            raise ValueError("ranking record policy/receipt mismatch")
        rank_map[key] = row
    if set(rank_map) != expected_rank_keys:
        raise ValueError("incomplete ranking grid")

    panel_evidence = payload.get("panel_selection_evidence")
    if not isinstance(panel_evidence, list):
        raise ValueError("panel_selection_evidence must be a list")
    selection_masks = _selection_masks_from_evidence(panel_evidence, width=width)
    expected_cells = {
        (int(seed), int(rank), label)
        for seed in config.seeds
        for rank in COMPONENT_GRID
        for label in active_labels_by_seed[int(seed)]
    }
    if set(selection_masks) != expected_cells:
        raise ValueError("panel-selection evidence does not cover resolved grid")
    evidence_map = {
        (int(row["seed"]), int(row["n_components"]), str(row["panel_label"])): row
        for row in panel_evidence
    }
    for seed in config.seeds:
        resolution_map = {
            row["panel_label"]: row
            for row in split_by_seed[int(seed)]["panel_resolution"]
            if row["active_grid_cell"]
        }
        retained_mask = retained_masks_by_seed[int(seed)]
        retained_count = retained_counts_by_seed[int(seed)]
        for rank in COMPONENT_GRID:
            previous: NDArray[np.uint8] | None = None
            for fraction in PANEL_FRACTIONS:
                label = _panel_label(fraction)
                key = (int(seed), int(rank), label)
                if key not in selection_masks:
                    continue
                packed = selection_masks[key]
                receipt = evidence_map[key]
                spec = resolution_map[label]
                if (
                    receipt.get("selected_mask_width") != width
                    or receipt.get("requested_panel_fraction") != fraction
                    or receipt.get("available_train_retained_snps") != retained_count
                    or receipt.get("selected_snp_count") != spec["effective_snp_count"]
                    or not math.isclose(
                        float(receipt.get("effective_fraction_of_train_retained")),
                        float(int(receipt.get("selected_snp_count")) / retained_count),
                        rel_tol=0.0,
                        abs_tol=0.0,
                    )
                    or receipt.get("selection_used_population_labels") is not False
                    or receipt.get("selection_used_test_rows") is not False
                    or receipt.get("panel_feature_set_is_split_specific") is not True
                    or receipt.get("loading_selection_is_split_specific")
                    is not (fraction is not None)
                    or np.any(np.bitwise_and(packed, np.bitwise_not(retained_mask)))
                ):
                    raise ValueError(f"invalid panel-selection receipt: {key}")
                if fraction is None:
                    if (
                        not np.array_equal(packed, retained_mask)
                        or receipt.get("selection_rule")
                        != "all_train_retained_no_loading_selection"
                    ):
                        raise ValueError("all panel must be the exact retained set")
                else:
                    if (
                        receipt.get("selection_rule")
                        != "top_training_pca_loading_fraction"
                    ):
                        raise ValueError("finite panel selection rule mismatch")
                    if previous is not None and np.any(
                        np.bitwise_and(previous, np.bitwise_not(packed))
                    ):
                        raise ValueError("fraction panels are not nested")
                    previous = packed

    records = payload.get("records")
    sample_evidence = payload.get("sample_prediction_evidence")
    if not isinstance(records, list) or not isinstance(sample_evidence, list):
        raise ValueError("records/sample evidence must be lists")
    record_map: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("record must be an object")
        key = (
            int(record.get("seed", -1)),
            int(record.get("n_components", -1)),
            str(record.get("panel_label")),
        )
        if key in record_map or key not in expected_cells:
            raise ValueError("records do not exactly cover the resolved grid")
        selection = evidence_map[key]
        split = split_by_seed[key[0]]
        if (
            record.get("evaluation") != "pca_loading_fraction_panel"
            or record.get("requested_panel_fraction")
            != selection["requested_panel_fraction"]
            or record.get("selected_snp_count") != selection["selected_snp_count"]
            or record.get("selected_original_indices_sha256")
            != selection["selected_original_indices_sha256"]
            or record.get("population_labels_used_for_panel_ranking") is not False
            or record.get("test_rows_used_for_runner_scaling_pca_or_ranking")
            is not False
            or record.get("solver") != "exact_chunked_sample_gram_eigh"
            or record.get("solver_converged") is not True
            or record.get("all_panel_uses_loading_selection") is not False
            or record.get("panel_feature_set_is_split_specific") is not True
            or record.get("loading_selection_is_split_specific")
            is not (selection["requested_panel_fraction"] is not None)
            or not _is_sha256(record.get("loadings_sha256"))
            or not _is_sha256(record.get("sample_gram_sha256"))
            or not _is_sha256(record.get("test_predictions_sha256"))
        ):
            raise ValueError(f"record policy or selection mismatch: {key}")
        for hash_key in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if record.get(hash_key) != split.get(hash_key):
                raise ValueError(f"record split binding mismatch: {key}/{hash_key}")
        evr = np.asarray(record.get("explained_variance_ratio"), dtype=float)
        singular = np.asarray(record.get("singular_values"), dtype=float)
        if (
            evr.shape != (key[1],)
            or singular.shape != (key[1],)
            or not np.isfinite(evr).all()
            or not np.isfinite(singular).all()
            or np.any(evr <= 0.0)
            or np.any(singular <= 0.0)
            or np.any(np.diff(singular) > 1e-8)
            or not math.isclose(
                float(np.sum(evr)),
                float(record.get("cumulative_explained_variance_ratio")),
                rel_tol=1e-12,
                abs_tol=1e-14,
            )
        ):
            raise ValueError(f"record variance receipt mismatch: {key}")
        record_map[key] = record
    if set(record_map) != expected_cells:
        raise ValueError("incomplete result record grid")

    evidence_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {
        key: [] for key in expected_cells
    }
    for row in sample_evidence:
        if not isinstance(row, Mapping):
            raise ValueError("sample evidence row must be an object")
        key = (int(row["seed"]), int(row["n_components"]), str(row["panel_label"]))
        if key not in evidence_groups:
            raise ValueError("sample evidence references an unknown grid cell")
        evidence_groups[key].append(row)
    classes = np.asarray(EXPECTED_POPULATIONS, dtype=str)
    for key, rows in evidence_groups.items():
        rows.sort(key=lambda row: int(row["sample_position"]))
        if [int(row["sample_position"]) for row in rows] != list(range(len(rows))):
            raise ValueError(f"sample positions are not contiguous: {key}")
        if not rows:
            raise ValueError(f"grid cell lacks sample evidence: {key}")
        record = record_map[key]
        split = split_by_seed[key[0]]
        test_indices = np.asarray(
            [int(row["sample_index"]) for row in rows],
            dtype=np.dtype(split["test_indices_dtype"]),
        )
        test_ids = np.asarray(
            [str(row["sample_id"]) for row in rows],
            dtype=np.dtype(split["test_sample_ids_dtype"]),
        )
        truth = np.asarray(
            [str(row["truth"]) for row in rows],
            dtype=np.dtype(split["test_labels_dtype"]),
        )
        predictions = np.asarray(
            [str(row["prediction"]) for row in rows],
            dtype=np.dtype(record["test_predictions_dtype"]),
        )
        if (
            array_sha256(test_indices) != split["test_indices_sha256"]
            or array_sha256(test_ids) != split["test_sample_ids_sha256"]
            or array_sha256(truth) != split["test_labels_sha256"]
            or array_sha256(predictions) != record["test_predictions_sha256"]
            or any(
                bool(row["correct"]) != (row["truth"] == row["prediction"])
                for row in rows
            )
            or any(
                row.get("requested_panel_fraction")
                != record["requested_panel_fraction"]
                or row.get("selected_snp_count") != record["selected_snp_count"]
                or row.get("test_indices_sha256") != split["test_indices_sha256"]
                or row.get("test_indices_dtype") != split["test_indices_dtype"]
                or row.get("test_sample_ids_sha256") != split["test_sample_ids_sha256"]
                or row.get("test_sample_ids_dtype") != split["test_sample_ids_dtype"]
                or row.get("test_labels_sha256") != split["test_labels_sha256"]
                or row.get("test_labels_dtype") != split["test_labels_dtype"]
                or row.get("test_predictions_sha256")
                != record["test_predictions_sha256"]
                or row.get("test_predictions_dtype") != record["test_predictions_dtype"]
                for row in rows
            )
            or any(
                row["selected_original_indices_sha256"]
                != record["selected_original_indices_sha256"]
                for row in rows
            )
        ):
            raise ValueError(f"sample evidence hash or correctness mismatch: {key}")
        derived_metrics = base._classification_metrics(truth, predictions, classes)
        if _record_metric_subset(record) != derived_metrics:
            raise ValueError(f"record metrics do not match sample evidence: {key}")

    expected_grid, expected_population = _summaries(records)
    if payload.get("grid_summaries") != expected_grid:
        raise ValueError("grid summaries do not match split records")
    if payload.get("population_summaries") != expected_population:
        raise ValueError("population summaries do not match split records")
    expected_stability = _stability_summaries(
        panel_evidence,
        width=width,
        predecoded_selections=selection_masks,
    )
    if payload.get("panel_stability_summaries") != expected_stability:
        raise ValueError("panel stability summaries do not match selection masks")

    registered = payload.get("registered_seed_policy")
    if not isinstance(registered, Mapping) or (
        registered.get("seed") != config.seeds[0]
        or registered.get("selection") != "first_configured_seed_never_best_seed"
        or registered.get("rank") != max(COMPONENT_GRID)
        or registered.get("rank_policy")
        != "fixed_maximum_grid_rank_never_result_selected"
        or registered.get("plotted_component_pairs") != [[1, 2], [3, 4]]
        or registered.get("confusion_rank") != max(COMPONENT_GRID)
        or registered.get(
            "finite_panels_are_independently_loading_selected_within_each_split"
        )
        is not True
        or registered.get("all_panel_uses_no_loading_selection") is not True
        or registered.get("every_panel_feature_set_uses_split_train_retained_mask")
        is not True
    ):
        raise ValueError("registered figure policy mismatch")
    coordinates = payload.get("registered_coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise ValueError("registered coordinates are missing")
    coordinate_cells = {
        (int(row["seed"]), int(row["n_components"]), str(row["panel_label"]))
        for row in coordinates
    }
    expected_coordinate_cells = {
        (int(config.seeds[0]), max(COMPONENT_GRID), label)
        for label in active_labels_by_seed[int(config.seeds[0])]
    }
    if coordinate_cells != expected_coordinate_cells:
        raise ValueError("registered coordinates use an unregistered seed/rank/panel")
    for row in coordinates:
        if row.get("row_type") not in {"held_out_sample", "train_centroid"}:
            raise ValueError("unknown coordinate row type")
        for component in range(1, max(COMPONENT_GRID) + 1):
            value = row.get(f"component_{component}")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("registered coordinate is non-finite")
    coordinate_groups: dict[tuple[int, int, str], list[Mapping[str, Any]]] = {
        key: [] for key in expected_coordinate_cells
    }
    for row in coordinates:
        key = (int(row["seed"]), int(row["n_components"]), str(row["panel_label"]))
        coordinate_groups[key].append(row)
    for key, rows in coordinate_groups.items():
        record = record_map[key]
        held_out = [row for row in rows if row["row_type"] == "held_out_sample"]
        centroids = [row for row in rows if row["row_type"] == "train_centroid"]
        evidence_by_index = {
            int(row["sample_index"]): row for row in evidence_groups[key]
        }
        if (
            len(held_out) != len(evidence_by_index)
            or len({int(row["sample_index"]) for row in held_out}) != len(held_out)
            or [row["population"] for row in centroids] != list(EXPECTED_POPULATIONS)
        ):
            raise ValueError(f"registered coordinate row cover mismatch: {key}")
        for row in held_out:
            evidence = evidence_by_index[int(row["sample_index"])]
            if (
                row.get("sample_id") != evidence["sample_id"]
                or row.get("population") != evidence["truth"]
                or row.get("truth") != evidence["truth"]
                or row.get("prediction") != evidence["prediction"]
                or row.get("correct") != evidence["correct"]
                or row.get("selected_snp_count") != record["selected_snp_count"]
                or row.get("requested_panel_fraction")
                != record["requested_panel_fraction"]
                or row.get("panel_feature_set_is_split_specific")
                is not record["panel_feature_set_is_split_specific"]
                or row.get("loading_selection_is_split_specific")
                is not record["loading_selection_is_split_specific"]
            ):
                raise ValueError(f"registered coordinate/evidence mismatch: {key}")
        for row in centroids:
            if (
                row.get("sample_index") is not None
                or row.get("sample_id") is not None
                or row.get("truth") is not None
                or row.get("prediction") is not None
                or row.get("correct") is not None
                or row.get("selected_snp_count") != record["selected_snp_count"]
                or row.get("requested_panel_fraction")
                != record["requested_panel_fraction"]
            ):
                raise ValueError(f"registered centroid receipt mismatch: {key}")

    analysis_sha = payload.get("analysis_sha256")
    if not _is_sha256(analysis_sha):
        raise ValueError("analysis_sha256 is invalid")
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    if canonical_sha256(unsigned) != analysis_sha:
        raise ValueError("analysis_sha256 mismatch")
    return {
        "schema_version": 1,
        "seeds": len(config.seeds),
        "records": len(records),
        "sample_prediction_rows": len(sample_evidence),
        "selection_masks": len(panel_evidence),
        "analysis_sha256": analysis_sha,
        "self_consistency_verified": True,
        "trusted_input_replay_verified": False,
    }


def merge_registered_shard_payloads(
    shard_payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge only an exact, disjoint cover of the frozen 20-seed bank."""

    if len(shard_payloads) < 2:
        raise ValueError("merge requires at least two registered shard payloads")
    shards = list(shard_payloads)
    for payload in shards:
        validate_result_payload(payload)
        if payload["config"]["analysis_mode"] != "registered_shard":
            raise ValueError("merge inputs must be registered_shard payloads")
    reference = shards[0]
    for payload in shards[1:]:
        for key in (
            "analysis_policy",
            "method_scope",
            "input_identity",
            "source_identity",
            "runtime_versions",
            "memory_and_solver_policy",
        ):
            if payload[key] != reference[key]:
                raise ValueError(f"shards disagree on immutable field: {key}")
        reference_config = dict(reference["config"])
        candidate_config = dict(payload["config"])
        for config_value in (reference_config, candidate_config):
            config_value.pop("seeds")
        if candidate_config != reference_config:
            raise ValueError("shards disagree on non-seed configuration")
    seed_to_shard: dict[int, Mapping[str, Any]] = {}
    for payload in shards:
        seeds = tuple(int(seed) for seed in payload["config"]["seeds"])
        if seeds != tuple(sorted(seeds)):
            raise ValueError(
                "each registered shard must store seeds in ascending order"
            )
        for seed in seeds:
            if seed in seed_to_shard:
                raise ValueError(f"duplicate registered seed across shards: {seed}")
            seed_to_shard[seed] = payload
    if set(seed_to_shard) != set(PROTOCOL_SEEDS):
        missing = sorted(set(PROTOCOL_SEEDS) - set(seed_to_shard))
        extra = sorted(set(seed_to_shard) - set(PROTOCOL_SEEDS))
        raise ValueError(
            f"shards must exactly cover seed bank; missing={missing}, extra={extra}"
        )
    panel_order = {
        _panel_label(value): index for index, value in enumerate(PANEL_FRACTIONS)
    }

    def combined_list(name: str, sort_key: Any) -> list[dict[str, Any]]:
        rows = [dict(row) for payload in shards for row in payload[name]]
        return sorted(rows, key=sort_key)

    split_receipts = combined_list("split_receipts", lambda row: int(row["seed"]))
    rankings = combined_list(
        "ranking_records", lambda row: (int(row["seed"]), int(row["n_components"]))
    )
    panel_evidence = combined_list(
        "panel_selection_evidence",
        lambda row: (
            int(row["seed"]),
            int(row["n_components"]),
            panel_order[str(row["panel_label"])],
        ),
    )
    records = combined_list(
        "records",
        lambda row: (
            int(row["seed"]),
            int(row["n_components"]),
            panel_order[str(row["panel_label"])],
        ),
    )
    sample_evidence = combined_list(
        "sample_prediction_evidence",
        lambda row: (
            int(row["seed"]),
            int(row["n_components"]),
            panel_order[str(row["panel_label"])],
            int(row["sample_position"]),
        ),
    )
    seed_2025_payload = seed_to_shard[PROTOCOL_SEEDS[0]]
    if int(seed_2025_payload["registered_seed_policy"]["seed"]) != PROTOCOL_SEEDS[0]:
        raise ValueError(
            "the shard containing seed 2025 did not register its coordinates"
        )
    coordinates = [dict(row) for row in seed_2025_payload["registered_coordinates"]]
    grid, populations = _summaries(records)
    width = int(reference["input_identity"]["snps"])
    stability = _stability_summaries(panel_evidence, width=width)
    chunk_size = int(reference["config"]["snp_chunk_size"])
    merged_config = AllAutosomeConfig(snp_chunk_size=chunk_size)
    merged: dict[str, Any] = {
        "schema_version": 1,
        "analysis_policy": dict(reference["analysis_policy"]),
        "config": _config_payload(merged_config),
        "method_scope": dict(reference["method_scope"]),
        "input_identity": dict(reference["input_identity"]),
        "source_identity": dict(reference["source_identity"]),
        "runtime_versions": dict(reference["runtime_versions"]),
        "execution_provenance": {
            "execution_mode": "fixed_protocol",
            "registered_seed_bank": list(PROTOCOL_SEEDS),
            "included_seeds": list(PROTOCOL_SEEDS),
            "exact_registered_seed_cover": True,
            "merged_from_shards": True,
            "source_shard_analysis_sha256": sorted(
                str(payload["analysis_sha256"]) for payload in shards
            ),
        },
        "memory_and_solver_policy": dict(reference["memory_and_solver_policy"]),
        "split_receipts": split_receipts,
        "ranking_records": rankings,
        "panel_selection_evidence": panel_evidence,
        "records": records,
        "grid_summaries": grid,
        "population_summaries": populations,
        "panel_stability_summaries": stability,
        "sample_prediction_evidence": sample_evidence,
        "registered_seed_policy": {
            "seed": PROTOCOL_SEEDS[0],
            "selection": "first_configured_seed_never_best_seed",
            "rank": max(COMPONENT_GRID),
            "rank_policy": "fixed_maximum_grid_rank_never_result_selected",
            "plotted_component_pairs": [[1, 2], [3, 4]],
            "confusion_rank": max(COMPONENT_GRID),
            "panels": [_panel_label(value) for value in PANEL_FRACTIONS],
            "finite_panels_are_independently_loading_selected_within_each_split": True,
            "all_panel_uses_no_loading_selection": True,
            "every_panel_feature_set_uses_split_train_retained_mask": True,
        },
        "registered_coordinates": coordinates,
    }
    merged["analysis_sha256"] = canonical_sha256(merged)
    validate_result_payload(merged)
    return merged


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue().encode("utf-8")


GRID_FIELDS = (
    "n_components",
    "panel_label",
    "requested_panel_fraction",
    "seeds",
    "minimum_selected_snp_count",
    "maximum_selected_snp_count",
    "mean_effective_fraction_of_train_retained",
    "mean_accuracy",
    "mean_balanced_accuracy",
    "sample_std_balanced_accuracy",
    "minimum_balanced_accuracy",
    "maximum_balanced_accuracy",
    "mean_macro_f1",
    "mean_paired_balanced_accuracy_delta_vs_all",
    "sample_std_paired_balanced_accuracy_delta_vs_all",
    "descriptive_only_no_qualification_gate",
)
POPULATION_FIELDS = (
    "n_components",
    "panel_label",
    "requested_panel_fraction",
    "population",
    "seeds",
    "pooled_support",
    "pooled_true_positive",
    "mean_recall",
    "minimum_recall",
    "maximum_recall",
)
STABILITY_FIELDS = (
    "n_components",
    "panel_label",
    "requested_panel_fraction",
    "seeds",
    "pairwise_comparisons",
    "mean_pairwise_jaccard",
    "minimum_pairwise_jaccard",
    "maximum_pairwise_jaccard",
    "all_seed_intersection_size",
    "all_seed_union_size",
    "selected_in_at_least_80pct_splits",
    "stability_is_across_split_specific_panels",
)
SAMPLE_FIELDS = (
    "seed",
    "n_components",
    "panel_label",
    "requested_panel_fraction",
    "selected_snp_count",
    "selected_original_indices_sha256",
    "sample_position",
    "sample_index",
    "sample_id",
    "truth",
    "prediction",
    "correct",
    "test_indices_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "test_predictions_sha256",
)
COORDINATE_FIELDS = (
    "row_type",
    "seed",
    "n_components",
    "panel_label",
    "requested_panel_fraction",
    "selected_snp_count",
    "effective_fraction_of_train_retained",
    "panel_feature_set_is_split_specific",
    "loading_selection_is_split_specific",
    "sample_index",
    "sample_id",
    "population",
    "truth",
    "prediction",
    "correct",
    *(f"component_{index}" for index in range(1, max(COMPONENT_GRID) + 1)),
)


def _metrics_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["grid_summaries"], GRID_FIELDS)


def _population_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["population_summaries"], POPULATION_FIELDS)


def _stability_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["panel_stability_summaries"], STABILITY_FIELDS)


def _sample_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["sample_prediction_evidence"], SAMPLE_FIELDS)


def _coordinates_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(payload["registered_coordinates"], COORDINATE_FIELDS)


def _active_registered_panels(payload: Mapping[str, Any]) -> list[str]:
    seed = int(payload["registered_seed_policy"]["seed"])
    present = {
        row["panel_label"]
        for row in payload["registered_coordinates"]
        if row["seed"] == seed
    }
    return [
        _panel_label(fraction)
        for fraction in PANEL_FRACTIONS
        if _panel_label(fraction) in present
    ]


def _render_pc_figure(
    payload: Mapping[str, Any],
    path: Path,
    *,
    x_component: int,
    y_component: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        _render_pc_figure_default_style(
            payload,
            path,
            x_component=x_component,
            y_component=y_component,
        )


def _render_pc_figure_default_style(
    payload: Mapping[str, Any],
    path: Path,
    *,
    x_component: int,
    y_component: int,
) -> None:
    import matplotlib.pyplot as plt

    panels = _active_registered_panels(payload)
    seed = int(payload["registered_seed_policy"]["seed"])
    rank = int(payload["registered_seed_policy"]["rank"])
    palette = plt.get_cmap("tab10")
    colors = {
        population: palette(index)
        for index, population in enumerate(EXPECTED_POPULATIONS)
    }
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 10.4), squeeze=False, dpi=150)
    flat_axes = axes.ravel()
    summary_map = {
        (row["n_components"], row["panel_label"]): row
        for row in payload["grid_summaries"]
    }
    for panel_index, panel in enumerate(panels):
        axis = flat_axes[panel_index]
        rows = [
            row
            for row in payload["registered_coordinates"]
            if row["seed"] == seed
            and row["n_components"] == rank
            and row["panel_label"] == panel
            and row["row_type"] == "held_out_sample"
        ]
        centroids = [
            row
            for row in payload["registered_coordinates"]
            if row["seed"] == seed
            and row["n_components"] == rank
            and row["panel_label"] == panel
            and row["row_type"] == "train_centroid"
        ]
        record = next(
            row
            for row in payload["records"]
            if row["seed"] == seed
            and row["n_components"] == rank
            and row["panel_label"] == panel
        )
        for population in EXPECTED_POPULATIONS:
            population_rows = [row for row in rows if row["truth"] == population]
            correct = [row for row in population_rows if row["correct"]]
            incorrect = [row for row in population_rows if not row["correct"]]
            if correct:
                axis.scatter(
                    [row[f"component_{x_component}"] for row in correct],
                    [row[f"component_{y_component}"] for row in correct],
                    color=colors[population],
                    s=28,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.35,
                    label=population,
                )
            if incorrect:
                axis.scatter(
                    [row[f"component_{x_component}"] for row in incorrect],
                    [row[f"component_{y_component}"] for row in incorrect],
                    color=colors[population],
                    s=44,
                    marker="x",
                    linewidths=1.3,
                )
            centroid = next(row for row in centroids if row["population"] == population)
            axis.scatter(
                [centroid[f"component_{x_component}"]],
                [centroid[f"component_{y_component}"]],
                color=colors[population],
                s=105,
                marker="*",
                edgecolors="black",
                linewidths=0.7,
                zorder=5,
            )
        selected = int(record["selected_snp_count"])
        fraction = float(record["effective_fraction_of_train_retained"])
        scope = (
            "split train-retained; no loading selection"
            if panel == "all"
            else "split-specific loading panel"
        )
        mean_ba = summary_map[(rank, panel)]["mean_balanced_accuracy"]
        axis.set_title(
            f"{panel}: {selected:,} SNPs ({100*fraction:.2f}%)\n"
            f"{scope}\nseed BA={record['balanced_accuracy']:.3f}; "
            f"{len(payload['config']['seeds'])}-seed mean={mean_ba:.3f}",
            fontsize=9.2,
        )
        evr = record["explained_variance_ratio"]
        axis.set_xlabel(
            f"PC{x_component} ({100*evr[x_component-1]:.2f}% train variance)"
        )
        axis.set_ylabel(
            f"PC{y_component} ({100*evr[y_component-1]:.2f}% train variance)"
        )
        axis.axhline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
        axis.axvline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
        axis.grid(alpha=0.12)
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    figure.suptitle(
        "All-autosome PCA-loading compression sensitivity — held-out samples, "
        f"fixed seed {seed}, rank {rank} (not best-seed/config selected)",
        fontsize=13,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Each finite panel is reselected inside this split using training genotypes only; "
        "panels are not one fixed AIM list. Upstream QC/LD used the fixed 1,000-person "
        "cohort before splitting; in-run scaling/PCA/ranking are train-only. Stars: train "
        "centroids; x: held-out errors. Each panel has independently refit PCA axes.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.07, 0.98, 0.90), w_pad=1.6, h_pad=1.8)
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "vp-rspca", "Creation Time": None},
    )
    plt.close(figure)


def _aggregate_confusion(
    payload: Mapping[str, Any], panel_label: str, rank: int
) -> NDArray[np.int64]:
    matrices = [
        np.asarray(row["confusion_matrix_counts"], dtype=np.int64)
        for row in payload["records"]
        if row["panel_label"] == panel_label and row["n_components"] == rank
    ]
    if not matrices:
        raise ValueError("missing confusion rows for panel")
    return np.sum(np.stack(matrices, axis=0), axis=0)


def _render_confusion(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    with matplotlib.rc_context(rc=matplotlib.rcParamsDefault):
        _render_confusion_default_style(payload, path)


def _render_confusion_default_style(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    panels = _active_registered_panels(payload)
    rank = int(payload["registered_seed_policy"]["confusion_rank"])
    figure, axes = plt.subplots(2, 3, figsize=(15.6, 10.4), squeeze=False, dpi=150)
    flat_axes = axes.ravel()
    image = None
    for index, panel in enumerate(panels):
        axis = flat_axes[index]
        counts = _aggregate_confusion(payload, panel, rank)
        totals = counts.sum(axis=1, keepdims=True)
        normalized = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=float),
            where=totals > 0,
        )
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(len(EXPECTED_POPULATIONS)):
            for column in range(len(EXPECTED_POPULATIONS)):
                value = normalized[row, column]
                axis.text(
                    column,
                    row,
                    f"{100*value:.0f}%\n({counts[row, column]})",
                    ha="center",
                    va="center",
                    fontsize=8.2,
                    color="white" if value >= 0.55 else "#111827",
                )
        summary = next(
            row
            for row in payload["grid_summaries"]
            if row["panel_label"] == panel and row["n_components"] == rank
        )
        axis.set_title(
            f"{panel} | {summary['minimum_selected_snp_count']:,}–"
            f"{summary['maximum_selected_snp_count']:,} SNPs\n"
            f"mean BA={summary['mean_balanced_accuracy']:.3f}",
            fontsize=10,
        )
        axis.set_xticks(range(len(EXPECTED_POPULATIONS)), EXPECTED_POPULATIONS)
        axis.set_yticks(range(len(EXPECTED_POPULATIONS)), EXPECTED_POPULATIONS)
        axis.set_xlabel("Predicted superpopulation")
        if index in {0, 3}:
            axis.set_ylabel("True superpopulation")
    spare_axis = flat_axes[-1]
    spare_axis.axis("off")
    if image is not None:
        color_axis = spare_axis.inset_axes([0.43, 0.18, 0.08, 0.64])
        figure.colorbar(image, cax=color_axis)
        color_axis.set_title("Recall\nshare", fontsize=9, pad=8)
    protocol_complete = (
        payload["config"]["analysis_mode"] == "fixed_protocol"
        and tuple(payload["config"]["seeds"]) == PROTOCOL_SEEDS
    )
    seed_scope = (
        "All 20 registered seeds pooled"
        if protocol_complete
        else f"{len(payload['config']['seeds'])} included seeds pooled "
        f"({payload['config']['analysis_mode']})"
    )
    figure.suptitle(
        f"{seed_scope} — fixed rank {rank}; no panel selected as winner",
        fontsize=13,
        y=0.985,
    )
    figure.text(
        0.5,
        0.012,
        "Rows are normalized within true superpopulation; parentheses show held-out "
        "split-observations, so participants repeat across seeds. Finite loading panels "
        "are selected independently within every train/test split.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.075, 0.98, 0.92), w_pad=1.5, h_pad=1.8)
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "vp-rspca", "Creation Time": None},
    )
    plt.close(figure)


def write_output_package(
    payload: Mapping[str, Any], output_directory: Path
) -> dict[str, Any]:
    """Exclusively publish an exact artifact set with manifest written last."""

    validation = validate_result_payload(payload)
    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        raise FileExistsError(f"output directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = (
        output_directory.parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    )
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / METRICS_FILENAME).write_bytes(_metrics_csv_bytes(payload))
        (temporary / POPULATION_FILENAME).write_bytes(_population_csv_bytes(payload))
        (temporary / STABILITY_FILENAME).write_bytes(_stability_csv_bytes(payload))
        (temporary / SAMPLE_FILENAME).write_bytes(_sample_csv_bytes(payload))
        (temporary / COORDINATES_FILENAME).write_bytes(_coordinates_csv_bytes(payload))
        _render_pc_figure(
            payload, temporary / PC12_FILENAME, x_component=1, y_component=2
        )
        _render_pc_figure(
            payload, temporary / PC34_FILENAME, x_component=3, y_component=4
        )
        _render_confusion(payload, temporary / CONFUSION_FILENAME)
        artifacts: dict[str, dict[str, Any]] = {}
        for filename in ARTIFACT_FILENAMES:
            path = temporary / filename
            status = _regular_file(path, role=filename)
            artifacts[filename] = {
                "bytes": int(status.st_size),
                "sha256": _file_sha256(path),
            }
        seeds = tuple(payload["config"]["seeds"])
        protocol_complete = (
            payload["config"]["analysis_mode"] == "fixed_protocol"
            and seeds == PROTOCOL_SEEDS
        )
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": (
                "complete_protocol" if protocol_complete else "complete_shard_or_smoke"
            ),
            "manifest_written_last": True,
            "analysis_status": "fixed_descriptive_development_evaluation",
            "confirmatory": False,
            "best_configuration_selected": False,
            "target_metric_or_quality_gate_used": False,
            "protocol_seed_cover_complete": protocol_complete,
            "registered_seed_bank": list(PROTOCOL_SEEDS),
            "included_seeds": list(seeds),
            "analysis_sha256": payload["analysis_sha256"],
            "artifact_count_excluding_manifest": len(ARTIFACT_FILENAMES),
            "exact_package_file_count": len(ARTIFACT_FILENAMES) + 1,
            "artifacts": artifacts,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(
            {"analysis_sha256": payload["analysis_sha256"], "artifacts": artifacts}
        )
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        temporary.rename(output_directory)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    package_validation = validate_output_package(output_directory)
    return {
        **validation,
        **package_validation,
        "writer_self_check_only": True,
    }


def _first_replay_mismatch(expected: Any, observed: Any, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return f"{path}: type {type(expected).__name__} != {type(observed).__name__}"
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return f"{path}: mapping keys differ"
        for key in expected:
            mismatch = _first_replay_mismatch(
                expected[key], observed[key], f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            mismatch = _first_replay_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    if expected != observed:
        return f"{path}: {expected!r} != {observed!r}"
    return None


def _scientific_replay_mismatch(
    saved: Mapping[str, Any], replayed: Mapping[str, Any]
) -> str | None:
    """Compare computation while excluding non-computational merge provenance."""

    saved_science = dict(saved)
    replayed_science = dict(replayed)
    for value in (saved_science, replayed_science):
        value.pop("analysis_sha256", None)
        value.pop("execution_provenance", None)
    return _first_replay_mismatch(saved_science, replayed_science)


def validate_output_package(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
    genotypes_path: Path | None = None,
    source_manifest_path: Path | None = None,
    expected_source_manifest_sha256: str | None = None,
    scratch_root: Path | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("output package must be a real directory")
    expected = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    actual = {entry.name for entry in output_directory.iterdir()}
    if actual != expected:
        raise ValueError(
            f"output package files must be exact; missing={sorted(expected-actual)}, "
            f"extra={sorted(actual-expected)}"
        )
    manifest_path = output_directory / MANIFEST_FILENAME
    _regular_file(manifest_path, role="completion manifest")
    manifest = _strict_json_load(manifest_path)
    required_manifest = {
        "schema_version",
        "status",
        "manifest_written_last",
        "analysis_status",
        "confirmatory",
        "best_configuration_selected",
        "target_metric_or_quality_gate_used",
        "protocol_seed_cover_complete",
        "registered_seed_bank",
        "included_seeds",
        "analysis_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != required_manifest:
        raise ValueError("completion manifest schema mismatch")
    if manifest_path.read_bytes() != _strict_json_bytes(manifest):
        raise ValueError("completion manifest JSON bytes are not canonical")
    if (
        manifest["schema_version"] != 1
        or manifest["status"] not in {"complete_protocol", "complete_shard_or_smoke"}
        or manifest["manifest_written_last"] is not True
        or manifest["analysis_status"] != "fixed_descriptive_development_evaluation"
        or manifest["confirmatory"] is not False
        or manifest["best_configuration_selected"] is not False
        or manifest["target_metric_or_quality_gate_used"] is not False
        or manifest["registered_seed_bank"] != list(PROTOCOL_SEEDS)
        or manifest["artifact_count_excluding_manifest"] != len(ARTIFACT_FILENAMES)
        or manifest["exact_package_file_count"] != len(expected)
        or set(manifest["artifacts"]) != set(ARTIFACT_FILENAMES)
    ):
        raise ValueError("completion manifest policy or closure mismatch")
    unsigned_manifest = dict(manifest)
    manifest_digest = unsigned_manifest.pop("manifest_payload_sha256")
    if (
        not _is_sha256(manifest_digest)
        or canonical_sha256(unsigned_manifest) != manifest_digest
    ):
        raise ValueError("manifest_payload_sha256 mismatch")
    expected_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest["package_anchor_sha256"] != expected_anchor:
        raise ValueError("package anchor mismatch")
    for filename in ARTIFACT_FILENAMES:
        path = output_directory / filename
        status = _regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"bytes", "sha256"}
            or receipt["bytes"] != status.st_size
            or receipt["sha256"] != _file_sha256(path)
        ):
            raise ValueError(f"artifact receipt mismatch: {filename}")
    result_path = output_directory / RESULT_FILENAME
    payload = _strict_json_load(result_path)
    if result_path.read_bytes() != _strict_json_bytes(payload):
        raise ValueError("result JSON bytes are not canonical")
    result_validation = validate_result_payload(payload)
    if payload["analysis_sha256"] != manifest["analysis_sha256"]:
        raise ValueError("result/manifest analysis hash mismatch")
    protocol_complete = (
        payload["config"]["analysis_mode"] == "fixed_protocol"
        and tuple(payload["config"]["seeds"]) == PROTOCOL_SEEDS
    )
    if (
        manifest["included_seeds"] != payload["config"]["seeds"]
        or manifest["protocol_seed_cover_complete"] is not protocol_complete
        or manifest["status"]
        != ("complete_protocol" if protocol_complete else "complete_shard_or_smoke")
    ):
        raise ValueError("manifest seed-cover status mismatch")
    if (expected_analysis_sha256 is None) != (expected_package_anchor_sha256 is None):
        raise ValueError("external verification requires both package anchors")
    if expected_analysis_sha256 is not None:
        if (
            not _is_sha256(expected_analysis_sha256)
            or not _is_sha256(expected_package_anchor_sha256)
            or payload["analysis_sha256"] != expected_analysis_sha256
            or manifest["package_anchor_sha256"] != expected_package_anchor_sha256
        ):
            raise ValueError("package does not match supplied external anchors")
    if repo_root is not None:
        root = Path(repo_root).resolve()
        files = payload["source_identity"].get("files")
        if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILE_ROLES):
            raise ValueError("source-file closure mismatch")
        for relative, expected_hash in files.items():
            path = root / relative
            _regular_file(path, role=f"source closure {relative}")
            if not _is_sha256(expected_hash) or _file_sha256(path) != expected_hash:
                raise ValueError(f"source hash mismatch: {relative}")
    expected_csv = {
        METRICS_FILENAME: _metrics_csv_bytes(payload),
        POPULATION_FILENAME: _population_csv_bytes(payload),
        STABILITY_FILENAME: _stability_csv_bytes(payload),
        SAMPLE_FILENAME: _sample_csv_bytes(payload),
        COORDINATES_FILENAME: _coordinates_csv_bytes(payload),
    }
    for filename, expected_bytes in expected_csv.items():
        if (output_directory / filename).read_bytes() != expected_bytes:
            raise ValueError(f"CSV is not an exact result serialization: {filename}")
    from PIL import Image

    for filename in (PC12_FILENAME, PC34_FILENAME, CONFUSION_FILENAME):
        path = output_directory / filename
        if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError(f"invalid PNG signature: {filename}")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG" or image.width < 3000 or image.height < 750:
                raise ValueError(f"PNG is below the clarity floor: {filename}")
    replay_arguments = (
        genotypes_path,
        source_manifest_path,
        expected_source_manifest_sha256,
    )
    replay_requested = any(value is not None for value in replay_arguments)
    trusted_full_replay = False
    trusted_scientific_replay = False
    shard_provenance_replay = False
    if replay_requested:
        if (
            not all(value is not None for value in replay_arguments)
            or repo_root is None
        ):
            raise ValueError(
                "trusted replay requires genotypes, source manifest, its SHA, and repo_root"
            )
        if payload["config"]["analysis_mode"] == "tiny_smoke":
            raise ValueError(
                "trusted external-input replay does not apply to tiny smoke"
            )
        with load_formal_input(
            Path(genotypes_path),
            Path(source_manifest_path),
            expected_source_manifest_sha256=str(expected_source_manifest_sha256),
            scratch_root=scratch_root,
        ) as (replay_data, replay_identity):
            if replay_identity != payload["input_identity"]:
                raise ValueError(
                    "trusted replay input identity differs from saved result"
                )
            replay_source = _source_identity(Path(repo_root))
            if replay_source != payload["source_identity"]:
                raise ValueError(
                    "trusted replay source identity differs from saved result"
                )
            replay_config = _config_from_payload(payload["config"])
            replay_payload = evaluate_all_autosome_stratification(
                replay_data,
                replay_config,
                input_identity=replay_identity,
                source_identity=replay_source,
            )
        merged = payload["execution_provenance"]["merged_from_shards"] is True
        mismatch = (
            _scientific_replay_mismatch(payload, replay_payload)
            if merged
            else _first_replay_mismatch(payload, replay_payload)
        )
        if mismatch is not None:
            raise ValueError(f"trusted-input replay mismatch: {mismatch}")
        trusted_scientific_replay = True
        trusted_full_replay = not merged
    return {
        **result_validation,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "protocol_seed_cover_complete": protocol_complete,
        "package_self_consistency_verified": True,
        "trusted_input_replay_verified": trusted_full_replay,
        "trusted_scientific_replay_verified": trusted_scientific_replay,
        "source_shard_provenance_replay_verified": shard_provenance_replay,
        "source_shard_provenance_boundary": (
            "externally package-anchored but not recreated by direct scientific replay"
        ),
    }


def _synthetic_smoke_data() -> base.GenotypeInput:
    rng = np.random.default_rng(8317)
    labels = np.asarray(
        [population for population in EXPECTED_POPULATIONS for _ in range(20)],
        dtype=str,
    )
    snps = 7200
    probabilities = np.full((labels.size, snps), 0.24, dtype=float)
    for population_index, population in enumerate(EXPECTED_POPULATIONS):
        rows = labels == population
        start = 500 * population_index
        probabilities[rows, start : start + 360] = 0.76
        probabilities[
            rows, 3000 + 250 * population_index : 3125 + 250 * population_index
        ] = (0.08 + 0.16 * population_index)
    probabilities[:, 5000:5600] += np.linspace(-0.08, 0.08, labels.size)[:, None]
    probabilities = np.clip(probabilities, 0.03, 0.94)
    genotypes = rng.binomial(2, probabilities).astype(np.float32)
    chromosomes = np.asarray(
        [str(1 + (index * 22) // snps) for index in range(snps)], dtype=str
    )
    return base.GenotypeInput(
        genotypes=genotypes,
        sample_ids=np.asarray([f"smoke_{index:03d}" for index in range(labels.size)]),
        snp_ids=np.asarray(
            [f"{chromosomes[index]}:{index + 1}:A:G" for index in range(snps)]
        ),
        chromosomes=chromosomes,
        population_labels=labels,
    )


def _parse_registered_seeds(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds or seeds != tuple(sorted(set(seeds))):
        raise argparse.ArgumentTypeError("shard seeds must be unique and ascending")
    if not set(seeds).issubset(set(PROTOCOL_SEEDS)):
        raise argparse.ArgumentTypeError("shard seeds must be drawn from 2025..2044")
    return seeds


def _add_formal_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--genotypes", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        required=True,
        help="existing large-disk directory for temporary genotypes.npy extraction",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke", help="run engineering-only synthetic smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--repo-root", type=Path, default=Path("."))
    run = commands.add_parser("run", help="run the exact registered 20-seed protocol")
    _add_formal_input_arguments(run)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    shard = commands.add_parser(
        "run-shard", help="run an explicit subset of the registered seed bank"
    )
    _add_formal_input_arguments(shard)
    shard.add_argument("--seeds", type=_parse_registered_seeds, required=True)
    shard.add_argument("--output-dir", type=Path, required=True)
    shard.add_argument("--repo-root", type=Path, default=Path("."))
    merge = commands.add_parser(
        "merge", help="merge an exact disjoint cover of registered shard packages"
    )
    merge.add_argument("--shard-packages", type=Path, nargs="+", required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--repo-root", type=Path, default=Path("."))
    validate_self = commands.add_parser(
        "validate-self", help="validate package closure and internal evidence"
    )
    validate_self.add_argument("output_dir", type=Path)
    validate_self.add_argument("--repo-root", type=Path)
    validate = commands.add_parser(
        "validate", help="strict package validation plus full trusted-input replay"
    )
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--expected-analysis-sha256", required=True)
    validate.add_argument("--expected-package-anchor-sha256", required=True)
    _add_formal_input_arguments(validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate-self":
        validation = validate_output_package(
            args.output_dir,
            repo_root=args.repo_root,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if args.command == "validate":
        validation = validate_output_package(
            args.output_dir,
            repo_root=args.repo_root,
            expected_analysis_sha256=args.expected_analysis_sha256,
            expected_package_anchor_sha256=args.expected_package_anchor_sha256,
            genotypes_path=args.genotypes,
            source_manifest_path=args.source_manifest,
            expected_source_manifest_sha256=FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=args.scratch_root,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if args.command == "merge":
        payloads: list[Mapping[str, Any]] = []
        for package in args.shard_packages:
            validate_output_package(package, repo_root=args.repo_root)
            payload = _strict_json_load(Path(package) / RESULT_FILENAME)
            if not isinstance(payload, Mapping):
                raise ValueError("shard result must be a JSON object")
            payloads.append(payload)
        merged = merge_registered_shard_payloads(payloads)
        validation = write_output_package(merged, args.output_dir)
        print(
            json.dumps(
                {
                    "analysis_sha256": merged["analysis_sha256"],
                    "output_directory": str(args.output_dir),
                    "registered_seed_cover": list(merged["config"]["seeds"]),
                    "validation": validation,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    repo_root = Path(args.repo_root).resolve()
    if args.command == "smoke":
        data = _synthetic_smoke_data()
        config = AllAutosomeConfig(
            seeds=(73, 74),
            analysis_mode="tiny_smoke",
            snp_chunk_size=257,
        )
        identity = _synthetic_input_identity(data)
        payload = evaluate_all_autosome_stratification(
            data,
            config,
            input_identity=identity,
            source_identity=_source_identity(repo_root),
        )
        validation = write_output_package(payload, args.output_dir)
    else:
        config = (
            AllAutosomeConfig()
            if args.command == "run"
            else AllAutosomeConfig(
                seeds=tuple(args.seeds), analysis_mode="registered_shard"
            )
        )
        with load_formal_input(
            args.genotypes,
            args.source_manifest,
            expected_source_manifest_sha256=FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=args.scratch_root,
        ) as (data, identity):
            payload = evaluate_all_autosome_stratification(
                data,
                config,
                input_identity=identity,
                source_identity=_source_identity(repo_root),
            )
            validation = write_output_package(payload, args.output_dir)
    print(
        json.dumps(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "output_directory": str(args.output_dir),
                "included_seeds": list(payload["config"]["seeds"]),
                "grid_cells_per_seed": len(COMPONENT_GRID) * len(PANEL_FRACTIONS),
                "best_configuration_selected": False,
                "target_metric_or_quality_gate_used": False,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

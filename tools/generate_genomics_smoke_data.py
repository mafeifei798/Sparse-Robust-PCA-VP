#!/usr/bin/env python3
"""Generate a tiny, deterministic genomics dataset for local smoke tests.

The construction uses integer arithmetic rather than a random-number generator,
so repeated runs produce byte-identical NPZ, CSV, and manifest files.  The data
are synthetic and are not evidence about 1000 Genomes or biological variants.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np

from rspca.datasets import array_sha256


SCHEMA_VERSION = "rspca-genomics-smoke-v1"
POPULATIONS = ("AFR", "EUR", "EAS")
SAMPLES_PER_POPULATION = 10
N_VARIANTS = 18


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = BytesIO()
    np.lib.format.write_array(stream, np.asarray(array), allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    stream = BytesIO()
    with ZipFile(stream, mode="w", compression=ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            archive.writestr(info, _npy_bytes(np.asarray(arrays[name])))
    return stream.getvalue()


def _build_arrays() -> dict[str, np.ndarray]:
    population_index = np.repeat(
        np.arange(len(POPULATIONS), dtype=np.int8),
        SAMPLES_PER_POPULATION,
    )
    within_population = np.tile(
        np.arange(SAMPLES_PER_POPULATION, dtype=np.int16),
        len(POPULATIONS),
    )
    variant_index = np.arange(N_VARIANTS, dtype=np.int16)
    block_owner = variant_index // (N_VARIANTS // len(POPULATIONS))

    # Every SNP varies within each population, while six-SNP blocks retain a
    # clear population-specific signal for the held-out PCA smoke workflow.
    genotypes = (
        (within_population[:, None] + 2 * variant_index[None, :]) % 2
    ).astype(np.int8)
    matching_block = population_index[:, None] == block_owner[None, :]
    within_block = 1 + (
        (within_population[:, None] + variant_index[None, :]) % 2
    ).astype(np.int8)
    genotypes[matching_block] = within_block[matching_block]

    # Exercise the documented -9 missing-genotype path without making any SNP
    # or population unusable.
    row_index = np.arange(genotypes.shape[0], dtype=np.int16)[:, None]
    missing = (11 * row_index + 7 * variant_index[None, :]) % 53 == 0
    genotypes[missing] = -9

    sample_ids = np.asarray(
        [f"HGSMOKE{index:03d}" for index in range(genotypes.shape[0])],
        dtype=str,
    )
    variant_ids = np.asarray(
        [f"rs_smoke_{index + 1:03d}" for index in range(N_VARIANTS)],
        dtype=str,
    )
    return {
        "X": genotypes,
        "column_ids": variant_ids,
        "row_ids": sample_ids,
        "variant_ids": variant_ids,
    }


def _population_csv(sample_ids: np.ndarray) -> bytes:
    stream = StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["sample_id", "population"])
    populations = np.repeat(POPULATIONS, SAMPLES_PER_POPULATION)
    writer.writerows(zip(sample_ids.tolist(), populations, strict=True))
    return stream.getvalue().encode("utf-8")


def generate_smoke_dataset(output_dir: str | Path) -> dict[str, Any]:
    """Write the deterministic smoke dataset and return its manifest."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    arrays = _build_arrays()
    genotype_path = destination / "synthetic_genotypes.npz"
    metadata_path = destination / "synthetic_populations.csv"
    manifest_path = destination / "manifest.json"

    _atomic_write_bytes(genotype_path, _deterministic_npz(arrays))
    _atomic_write_bytes(metadata_path, _population_csv(arrays["row_ids"]))

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "synthetic_smoke_only",
        "construction": "closed_form_integer_arithmetic_without_rng",
        "matrix": {
            "orientation": "samples_by_variants",
            "shape": list(arrays["X"].shape),
            "dtype": arrays["X"].dtype.str,
            "missing_sentinel": -9,
            "missing_count": int(np.count_nonzero(arrays["X"] == -9)),
            "content_sha256": array_sha256(arrays["X"]),
        },
        "identifiers": {
            "sample_ids_sha256": array_sha256(arrays["row_ids"]),
            "variant_ids_sha256": array_sha256(arrays["variant_ids"]),
        },
        "populations": list(POPULATIONS),
        "samples_per_population": SAMPLES_PER_POPULATION,
        "artifacts": {
            "genotypes": {
                "path": genotype_path.name,
                "bytes": genotype_path.stat().st_size,
                "sha256": _sha256_file(genotype_path),
            },
            "metadata": {
                "path": metadata_path.name,
                "bytes": metadata_path.stat().st_size,
                "sha256": _sha256_file(metadata_path),
            },
        },
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _atomic_write_bytes(manifest_path, manifest_bytes)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/smoke/nonfed/genomics/input"),
        help="directory for the deterministic NPZ, population CSV, and manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    manifest = generate_smoke_dataset(arguments.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["generate_smoke_dataset", "main"]

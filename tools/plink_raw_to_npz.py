#!/usr/bin/env python3
"""Convert a PLINK 2 ``--export A`` .raw file to the canonical NPZ layout.

The input is scanned twice: the first pass validates its schema and sample IDs,
and the second pass fills one preallocated ``samples x variants`` matrix.  Raw
rows are never accumulated in memory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


METADATA_COLUMNS = ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")
SCHEMA_VERSION = "plink2-export-a-to-rspca-npz-v1"


@dataclass(frozen=True)
class _RawLayout:
    n_samples: int
    variant_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    input_sha256: str
    input_bytes: int


def _decode_line(raw_line: bytes, line_number: int, path: Path) -> str:
    try:
        # A UTF-8 BOM is harmless on the first line but invalid anywhere else.
        encoding = "utf-8-sig" if line_number == 1 else "utf-8"
        return raw_line.decode(encoding)
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{path}: line {line_number} is not valid UTF-8"
        ) from error


def _tokens(raw_line: bytes, line_number: int, path: Path) -> list[str]:
    values = _decode_line(raw_line, line_number, path).split()
    if not values:
        raise ValueError(f"{path}: blank line at line {line_number}")
    return values


def _scan_raw(path: Path) -> _RawLayout:
    digest = hashlib.sha256()
    input_bytes = 0
    row_ids: list[str] = []
    seen_iids: set[str] = set()
    variant_ids: tuple[str, ...] | None = None
    expected_width = 0

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            input_bytes += len(raw_line)
            values = _tokens(raw_line, line_number, path)
            if line_number == 1:
                if tuple(values[: len(METADATA_COLUMNS)]) != METADATA_COLUMNS:
                    observed = values[: len(METADATA_COLUMNS)]
                    raise ValueError(
                        "PLINK .raw header must begin with exactly "
                        f"{' '.join(METADATA_COLUMNS)}; observed {observed!r}"
                    )
                if len(values) == len(METADATA_COLUMNS):
                    raise ValueError("PLINK .raw input contains no variant columns")
                variant_ids = tuple(values[len(METADATA_COLUMNS) :])
                if len(set(variant_ids)) != len(variant_ids):
                    duplicates = sorted(
                        variant
                        for variant, count in _counts(variant_ids).items()
                        if count > 1
                    )
                    raise ValueError(
                        f"variant identifiers must be unique; duplicates: {duplicates!r}"
                    )
                expected_width = len(values)
                continue

            if len(values) != expected_width:
                raise ValueError(
                    f"{path}: line {line_number} has {len(values)} fields; "
                    f"expected {expected_width}"
                )
            iid = values[1]
            if iid in seen_iids:
                raise ValueError(f"duplicate IID {iid!r} at line {line_number}")
            seen_iids.add(iid)
            row_ids.append(iid)

    if variant_ids is None:
        raise ValueError(f"PLINK .raw input is empty: {path}")
    if not row_ids:
        raise ValueError("PLINK .raw input contains no sample rows")
    return _RawLayout(
        n_samples=len(row_ids),
        variant_ids=variant_ids,
        row_ids=tuple(row_ids),
        input_sha256=digest.hexdigest(),
        input_bytes=input_bytes,
    )


def _counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _parse_genotype(token: str, *, line_number: int, variant_id: str) -> float:
    if token.upper() == "NA":
        return float("nan")
    try:
        value = float(token)
    except ValueError as error:
        raise ValueError(
            f"invalid genotype {token!r} at line {line_number}, variant {variant_id!r}; "
            "only finite numbers, -9, and NA are accepted"
        ) from error
    if value == -9.0:
        return float("nan")
    if not np.isfinite(value):
        raise ValueError(
            f"non-finite genotype {token!r} at line {line_number}, "
            f"variant {variant_id!r}"
        )
    return value


def _load_matrix(path: Path, layout: _RawLayout) -> tuple[NDArray[np.float64], int]:
    matrix = np.empty(
        (layout.n_samples, len(layout.variant_ids)), dtype=np.float64
    )
    digest = hashlib.sha256()
    missing_count = 0
    expected_width = len(METADATA_COLUMNS) + len(layout.variant_ids)
    last_line_number = 0

    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            last_line_number = line_number
            digest.update(raw_line)
            values = _tokens(raw_line, line_number, path)
            if line_number == 1:
                if tuple(values) != METADATA_COLUMNS + layout.variant_ids:
                    raise ValueError("PLINK .raw input changed between validation passes")
                continue
            row_index = line_number - 2
            if row_index >= layout.n_samples or len(values) != expected_width:
                raise ValueError("PLINK .raw input changed between validation passes")
            if values[1] != layout.row_ids[row_index]:
                raise ValueError("PLINK .raw input changed between validation passes")
            for column_index, (variant_id, token) in enumerate(
                zip(layout.variant_ids, values[len(METADATA_COLUMNS) :], strict=True)
            ):
                value = _parse_genotype(
                    token, line_number=line_number, variant_id=variant_id
                )
                matrix[row_index, column_index] = value
                missing_count += int(np.isnan(value))

    if last_line_number - 1 != layout.n_samples:
        raise ValueError("PLINK .raw input changed between validation passes")
    if digest.hexdigest() != layout.input_sha256:
        raise ValueError("PLINK .raw input changed between validation passes")
    return matrix, missing_count


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _temporary_path(destination: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=suffix,
    )
    os.close(descriptor)
    return Path(name)


def _check_destination(path: Path, *, force: bool, label: str) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{label} already exists: {path}; pass force=True to replace it")
    if path.exists() and not path.is_file():
        raise ValueError(f"{label} is not a regular file: {path}")


def convert_plink_raw(
    input_path: str | Path,
    output_path: str | Path,
    *,
    provenance_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Convert a PLINK 2 export-A file and return its provenance record.

    Genotype tokens equal to numeric ``-9`` or case-insensitive ``NA`` become
    ``NaN``.  Metadata fields, including a ``PHENOTYPE`` value of ``-9``, are
    never parsed into ``X``.
    """

    source = Path(input_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    provenance = (
        Path(provenance_path).expanduser().resolve()
        if provenance_path is not None
        else None
    )
    if not source.is_file():
        raise FileNotFoundError(f"input .raw file does not exist: {source}")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_path must end in .npz")
    if source == destination or (provenance is not None and source == provenance):
        raise ValueError("input and output/provenance paths must be distinct")
    if provenance is not None and provenance == destination:
        raise ValueError("output and provenance paths must be distinct")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if provenance is not None:
        provenance.parent.mkdir(parents=True, exist_ok=True)
    _check_destination(destination, force=force, label="output NPZ")
    if provenance is not None:
        _check_destination(provenance, force=force, label="provenance JSON")

    layout = _scan_raw(source)
    matrix, missing_count = _load_matrix(source, layout)
    row_ids = np.asarray(layout.row_ids, dtype=str)
    variant_ids = np.asarray(layout.variant_ids, dtype=str)

    temporary_npz = _temporary_path(destination, ".npz")
    temporary_provenance: Path | None = None
    try:
        with temporary_npz.open("wb") as handle:
            np.savez(
                handle,
                X=matrix,
                row_ids=row_ids,
                variant_ids=variant_ids,
                # Compatibility alias used by the canonical genomics loader.
                column_ids=variant_ids,
            )
        output_sha256 = _file_sha256(temporary_npz)
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "source_format": "plink2_--export_A_raw",
            "metadata_columns": list(METADATA_COLUMNS),
            "input": {
                "path": str(source),
                "sha256": layout.input_sha256,
                "bytes": layout.input_bytes,
            },
            "output": {
                "path": str(destination),
                "sha256": output_sha256,
                "bytes": temporary_npz.stat().st_size,
                "npz_keys": ["X", "row_ids", "variant_ids", "column_ids"],
            },
            "matrix": {
                "orientation": "samples_by_variants",
                "shape": [layout.n_samples, len(layout.variant_ids)],
                "dtype": matrix.dtype.str,
                "missing_count": missing_count,
            },
            "identifiers": {
                "row_ids_source": "IID",
                "variant_ids_source": "header_fields_after_six_metadata_columns",
                "column_ids_alias": "identical_to_variant_ids_for_genomics_loader_compatibility",
            },
            "missing_genotype_policy": {
                "accepted_missing_tokens": ["numeric_-9", "case_insensitive_NA"],
                "stored_as": "NaN",
                "scope": "genotype_columns_only",
            },
        }
        if provenance is not None:
            temporary_provenance = _temporary_path(provenance, ".json")
            temporary_provenance.write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        os.replace(temporary_npz, destination)
        if provenance is not None and temporary_provenance is not None:
            os.replace(temporary_provenance, provenance)
        return record
    finally:
        temporary_npz.unlink(missing_ok=True)
        if temporary_provenance is not None:
            temporary_provenance.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a PLINK 2 --export A .raw file to samples-by-variants NPZ."
    )
    parser.add_argument("input_raw", type=Path, help="PLINK 2 --export A .raw file")
    parser.add_argument("output_npz", type=Path, help="destination .npz file")
    parser.add_argument(
        "--provenance",
        type=Path,
        help="optional JSON path for input/output hashes and shape metadata",
    )
    parser.add_argument(
        "--force", action="store_true", help="replace existing output files"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        record = convert_plink_raw(
            arguments.input_raw,
            arguments.output_npz,
            provenance_path=arguments.provenance,
            force=arguments.force,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

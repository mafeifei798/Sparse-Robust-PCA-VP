#!/usr/bin/env python3
"""Prepare a deterministic all-autosome 1000 Genomes PCA/AIM input.

This is a data-scope extension of ``prepare_genomics_aim_1kg.py``.  It keeps
the exact 1,000 ordered sample/label pairs from the pinned five-superpopulation
chr20--22 preparation manifest, while replacing the three-chromosome source by
the pinned 22-autosome PLINK 2 mirror package.  Population labels never enter
variant QC or LD pruning.

Each compressed PGEN is decompressed and processed in its own temporary
directory.  The large decompressed PGEN, QC fileset, export, and logs are
discarded before the next chromosome is processed.  The final package retains
only the deterministic NPZ, its provenance manifest, and the exact 22
``prune.in`` lists used to define the NPZ columns.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, BinaryIO, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from numpy.lib.format import open_memmap
from numpy.typing import NDArray

try:
    from tools import prepare_genomics_aim_1kg as legacy_preparer
except ModuleNotFoundError as error:
    # ``python tools/prepare_genomics_aim_all_autosomes.py`` puts ``tools/``
    # rather than the repository root on sys.path.  Keep the documented direct
    # invocation working without weakening any source hash in the manifest.
    if error.name != "tools":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tools import prepare_genomics_aim_1kg as legacy_preparer


SCHEMA_VERSION = "rspca-1kgp-all-autosome-preparation-v1"
AUTOSOMES = tuple(str(value) for value in range(1, 23))
POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
PINNED_SOURCE_MANIFEST_SHA256 = (
    "5e59af304e9d6b1e0a4099d139943fa1b9d3438a2588d9ee4f1f2f49206384c1"
)
PINNED_LEGACY_PREPARATION_MANIFEST_SHA256 = (
    "9378814177e0721e711ce2e6c3695effc5ff766d2529c17cc5c4e0de84f30342"
)
PINNED_SELECTED_SAMPLE_IDS_SHA256 = (
    "fbb844e7d6277b15ebdd63d2a362f8254c0edf901bbbb82f6a7b81e7f6095497"
)
EXPECTED_SAMPLE_COUNT = 1000
EXPECTED_PER_POPULATION = 200
MAF = 0.05
GENO = 0.05
LD_WINDOW = 50
LD_STEP = 5
LD_R2 = 0.2
INDEP_ORDER = 2
NPZ_NAME = "genomics_aim_all_autosomes.npz"
MANIFEST_NAME = "genomics_aim_all_autosomes.manifest.json"
PRUNE_DIRECTORY = "prune_in"
DOWNLOAD_COMPLETION_NAME = "all_autosome_download_manifest.json"
DOWNLOADER_SOURCE_PATH = (
    Path(__file__).resolve().with_name("download_genomics_all_autosome_sources.py")
)
NPZ_KEYS = (
    "chromosomes",
    "genotypes",
    "population_labels",
    "sample_ids",
    "snp_ids",
)
GIB = 1024**3
MAX_JSON_BYTES = 64 * 1024 * 1024
GENOTYPE_DTYPE = np.dtype("<f4")
NPZ_COMPRESSLEVEL = 6

StringArray = NDArray[np.str_]


def _capture_import_source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    raw = resolved.read_bytes()
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


_IMPORT_PREPARER_SOURCE = _capture_import_source(Path(__file__))
_IMPORT_LEGACY_PREPARER_SOURCE = _capture_import_source(Path(legacy_preparer.__file__))


@dataclass(frozen=True)
class _LegacyCohort:
    sample_ids: StringArray
    population_labels: StringArray
    selected_records: tuple[Mapping[str, str], ...]
    manifest_payload: Mapping[str, Any]


@dataclass(frozen=True)
class _ChromosomeOutput:
    chromosome: str
    genotype_block: Path
    snp_ids: tuple[str, ...]
    qc_variant_count: int
    retained_variant_count: int
    missing_count: int
    command_records: tuple[Mapping[str, Any], ...]
    ephemeral_artifacts: Mapping[str, Mapping[str, Any]]
    retained_prune_in: Path


class _HashingWriter:
    """Small write-only adapter used to hash exact NPY member bytes."""

    def __init__(self, destination: BinaryIO | None = None) -> None:
        self.destination = destination
        self.digest = hashlib.sha256()
        self.bytes_written = 0

    def write(self, payload: bytes) -> int:
        data = bytes(payload)
        self.digest.update(data)
        self.bytes_written += len(data)
        if self.destination is not None:
            written = self.destination.write(data)
            if written is not None and written != len(data):
                raise OSError("short write while emitting NPY member")
        return len(data)

    def flush(self) -> None:
        if self.destination is not None and hasattr(self.destination, "flush"):
            self.destination.flush()

    def hexdigest(self) -> str:
        return self.digest.hexdigest()


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _lexical_absolute(path: str | Path) -> Path:
    """Make a path absolute without following its leaf symlink."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _is_within(path: Path, directory: Path) -> bool:
    """Return whether an already-canonical path is equal to/below a directory."""

    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _require_regular(path: Path, *, role: str, executable: bool = False) -> None:
    if not _lexists(path):
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be a regular non-symlink file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise PermissionError(f"{role} is not executable: {path}")


def _require_directory(path: Path, *, role: str) -> None:
    if not path.exists() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{role} must be a real directory: {path}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text_list_sha256(values: Iterable[str]) -> str:
    encoded = "".join(f"{value}\n" for value in values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    _require_regular(path, role=role)
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError(f"{role} exceeds the {MAX_JSON_BYTES}-byte safety limit")

    def reject_constant(token: str) -> Any:
        raise ValueError(f"non-finite JSON token in {role}: {token}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key in {role}: {key}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{role} must contain a JSON object")
    return payload


def _exact_keys(value: Any, expected: set[str], *, role: str) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{role} has an unexpected schema: {observed}")
    return value


def _source_code_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": _file_sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _assert_import_bound_sources_unchanged() -> None:
    current_preparer = _source_code_record(Path(__file__))
    current_legacy = _source_code_record(Path(legacy_preparer.__file__))
    if current_preparer != _IMPORT_PREPARER_SOURCE:
        raise RuntimeError("all-autosome preparer source changed after module import")
    if current_legacy != _IMPORT_LEGACY_PREPARER_SOURCE:
        raise RuntimeError("legacy preparation helper changed after module import")


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "bytes": path.stat().st_size}


def _preflight_disk_space(path: Path, *, required_bytes: int, role: str) -> None:
    if required_bytes <= 0:
        raise ValueError("internal disk-space requirement must be positive")
    free = shutil.disk_usage(path).free
    if free < required_bytes:
        raise OSError(
            f"insufficient scratch space for {role}: need at least "
            f"{required_bytes} free bytes, observed {free}"
        )


def _fresh_downloader_validation(
    *,
    downloader_source: Path,
    source_manifest: Path,
    download_dir: Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    expected_download_completion_sha256: str,
) -> dict[str, Any]:
    """Run the on-disk downloader bytes in a fresh interpreter."""

    argv = [
        sys.executable,
        str(downloader_source),
        "validate",
        "--source-manifest",
        str(source_manifest),
        "--output-dir",
        str(download_dir),
        "--expected-source-manifest-sha256",
        expected_source_manifest_sha256,
        "--expected-downloader-source-sha256",
        expected_downloader_source_sha256,
        "--expected-completion-sha256",
        expected_download_completion_sha256,
    ]
    completed = subprocess.run(
        argv,
        check=False,
        capture_output=True,
        env=_command_environment(),
    )
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"fresh downloader validation subprocess failed: {tail}")
    try:
        report = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("fresh downloader validator returned invalid JSON") from error
    if not isinstance(report, dict):
        raise ValueError("fresh downloader validator report must be a JSON object")
    return report


def _load_pinned_sources(
    source_manifest: Path,
    download_dir: Path,
    *,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    expected_download_completion_sha256: str,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], dict[str, Any]]:
    for value, role in (
        (expected_source_manifest_sha256, "expected source-manifest SHA-256"),
        (expected_downloader_source_sha256, "expected downloader-source SHA-256"),
        (expected_download_completion_sha256, "expected download-completion SHA-256"),
    ):
        _validate_sha256_token(value, role=role)
    if expected_source_manifest_sha256 != PINNED_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "external source-manifest SHA-256 disagrees with the preparer safety pin"
        )
    _require_regular(source_manifest, role="all-autosome source manifest")
    observed = _file_sha256(source_manifest)
    if observed != PINNED_SOURCE_MANIFEST_SHA256:
        raise ValueError(
            "all-autosome source manifest SHA-256 does not match the repository "
            f"pin: {observed}"
        )
    downloader_source = DOWNLOADER_SOURCE_PATH
    observed_downloader_hash = _file_sha256(downloader_source)
    if observed_downloader_hash != expected_downloader_source_sha256:
        raise ValueError(
            "on-disk downloader source does not match the external expectation: "
            f"{observed_downloader_hash}"
        )
    completion_path = download_dir / DOWNLOAD_COMPLETION_NAME
    _require_regular(completion_path, role="download completion manifest")
    if _file_sha256(completion_path) != expected_download_completion_sha256:
        raise ValueError("download completion file does not match its external SHA-256")
    payload = _load_json(source_manifest, role="all-autosome source manifest")
    if set(payload) != {"files", "schema_version", "scope", "source"} or not isinstance(
        payload["files"], list
    ):
        raise ValueError("all-autosome source manifest has an unexpected schema")
    report = _fresh_downloader_validation(
        downloader_source=downloader_source,
        source_manifest=source_manifest,
        download_dir=download_dir,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
        expected_download_completion_sha256=expected_download_completion_sha256,
    )
    if (
        not isinstance(report, dict)
        or report.get("status") != "valid_and_externally_anchored"
    ):
        raise ValueError(
            "fresh downloader validator did not return anchored-valid status"
        )
    if report.get("source_manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("fresh downloader report source anchor changed")
    if report.get("downloader_source_sha256") != expected_downloader_source_sha256:
        raise ValueError("fresh downloader report tool anchor changed")
    if report.get("completion_file_sha256") != expected_download_completion_sha256:
        raise ValueError("fresh downloader report completion anchor changed")
    if _file_sha256(downloader_source) != expected_downloader_source_sha256:
        raise RuntimeError("downloader source changed across fresh validation")
    if _file_sha256(source_manifest) != expected_source_manifest_sha256:
        raise RuntimeError("source manifest changed across fresh validation")
    if _file_sha256(completion_path) != expected_download_completion_sha256:
        raise RuntimeError("completion manifest changed across fresh validation")
    completion_payload = _load_json(
        completion_path, role="download completion manifest"
    )
    if completion_payload.get("payload_sha256") != report.get(
        "completion_payload_sha256"
    ):
        raise ValueError(
            "completion payload SHA disagrees with fresh validation report"
        )
    records = tuple(dict(record) for record in payload["files"])
    return payload, records, report


def _load_legacy_cohort(path: Path) -> _LegacyCohort:
    _require_regular(path, role="legacy five-superpopulation preparation manifest")
    observed = _file_sha256(path)
    if observed != PINNED_LEGACY_PREPARATION_MANIFEST_SHA256:
        raise ValueError(
            "legacy preparation manifest SHA-256 does not match the frozen "
            f"five-superpopulation cohort pin: {observed}"
        )
    payload = _load_json(path, role="legacy five-superpopulation preparation manifest")
    if payload.get("schema_version") != legacy_preparer.SCHEMA_VERSION:
        raise ValueError("legacy preparation manifest schema version changed")
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or not (
        dataset.get("input_scope") == "legacy_chr20_22"
        and dataset.get("population_design") == "five_superpopulation"
        and dataset.get("population_label_semantics") == "1000_Genomes_superpopulation"
        and dataset.get("population_label_is_race") is False
        and dataset.get("ld_pruned") is True
    ):
        raise ValueError(
            "legacy manifest does not describe the frozen five-group input"
        )
    selection = payload.get("sample_selection")
    if not isinstance(selection, dict):
        raise ValueError("legacy manifest lacks sample_selection")
    records = selection.get("selected_records")
    if not isinstance(records, list) or len(records) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"legacy cohort must contain exactly {EXPECTED_SAMPLE_COUNT} selected records"
        )
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, dict) or set(raw) != {
            "sample_id",
            "subpopulation",
            "population_label",
        }:
            raise ValueError(f"legacy selected_records[{index}] has an invalid schema")
        if not all(isinstance(raw[key], str) and raw[key] for key in raw):
            raise ValueError(
                f"legacy selected_records[{index}] contains an empty field"
            )
        if raw["population_label"] not in POPULATIONS:
            raise ValueError(f"unexpected population label in legacy record {index}")
        normalized.append(dict(raw))
    sample_ids = [record["sample_id"] for record in normalized]
    labels = [record["population_label"] for record in normalized]
    if len(set(sample_ids)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("legacy cohort sample IDs are not unique")
    if normalized != sorted(
        normalized, key=lambda record: (record["population_label"], record["sample_id"])
    ):
        raise ValueError("legacy cohort ordering is not group then sample ID")
    counts = Counter(labels)
    if counts != Counter(
        {population: EXPECTED_PER_POPULATION for population in POPULATIONS}
    ):
        raise ValueError("legacy cohort is not exactly 200 samples per superpopulation")
    sample_hash = _text_list_sha256(sample_ids)
    if sample_hash != PINNED_SELECTED_SAMPLE_IDS_SHA256:
        raise ValueError(
            "legacy selected-sample list does not match its independent pin"
        )
    if selection.get("selected_sample_ids_sha256") != sample_hash:
        raise ValueError("legacy selected_sample_ids_sha256 is inconsistent")
    if selection.get("selected_count") != EXPECTED_SAMPLE_COUNT:
        raise ValueError("legacy selected_count is inconsistent")
    if selection.get("selected_counts_by_group") != dict(counts):
        raise ValueError("legacy selected_counts_by_group is inconsistent")
    return _LegacyCohort(
        sample_ids=np.asarray(sample_ids, dtype=str),
        population_labels=np.asarray(labels, dtype=str),
        selected_records=tuple(normalized),
        manifest_payload=payload,
    )


def _read_psam_and_check_cohort(path: Path, cohort: _LegacyCohort) -> dict[str, Any]:
    _require_regular(path, role="downloaded PSAM")
    header: list[str] | None = None
    iid_index = -1
    label_index: int | None = None
    rows: dict[str, tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            fields = tuple(line.split())
            if header is None:
                header = [field.lstrip("#") for field in fields]
                if "IID" not in header:
                    raise ValueError("PSAM header lacks IID")
                iid_index = header.index("IID")
                normalized = [field.lower().replace("_", "") for field in header]
                for candidate in ("superpop", "superpopulation"):
                    if candidate in normalized:
                        label_index = normalized.index(candidate)
                        break
                continue
            if len(fields) != len(header):
                raise ValueError(f"ragged PSAM row at line {line_number}")
            iid = fields[iid_index]
            if not iid or iid == "0":
                raise ValueError(f"invalid IID in PSAM row {line_number}")
            if iid in rows:
                raise ValueError(f"duplicate IID in PSAM: {iid}")
            rows[iid] = fields
    if header is None or not rows:
        raise ValueError("PSAM contains no sample rows")
    expected = cohort.sample_ids.tolist()
    missing = sorted(set(expected) - set(rows))
    if missing:
        raise ValueError(f"PSAM lacks frozen cohort samples: {missing[:5]}")
    labels_checked = label_index is not None
    if label_index is not None:
        expected_labels = dict(
            zip(
                cohort.sample_ids.tolist(),
                cohort.population_labels.tolist(),
                strict=True,
            )
        )
        mismatches = [
            sample
            for sample in expected
            if rows[sample][label_index] != expected_labels[sample]
        ]
        if mismatches:
            raise ValueError(
                "PSAM superpopulation disagrees with frozen legacy labels for "
                f"{mismatches[:5]}"
            )
    return {
        "rows": len(rows),
        "selected_samples_present": len(expected),
        "superpopulation_column_checked": labels_checked,
        "selected_sample_ids_sha256": _text_list_sha256(expected),
    }


def _write_keep_file(path: Path, sample_ids: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        # IID-only keep files avoid assuming a particular FID convention in the
        # corrected phase-3 PSAM while still selecting the exact frozen IIDs.
        handle.write("#IID\n")
        for sample_id in sample_ids:
            handle.write(f"{sample_id}\n")


def _command_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC"})
    return environment


def _run_zst_decompress(
    *, plink2: Path, source: Path, destination: Path, cwd: Path, chromosome: str
) -> dict[str, Any]:
    argv = [str(plink2), "--zst-decompress", str(source)]
    stderr_path = cwd / f"chr{chromosome}_decompress.stderr.log"
    try:
        with destination.open("xb") as output, stderr_path.open("xb") as stderr:
            completed = subprocess.run(
                argv,
                cwd=cwd,
                env=_command_environment(),
                check=False,
                stdout=output,
                stderr=stderr,
            )
    except BaseException:
        destination.unlink(missing_ok=True)
        stderr_path.unlink(missing_ok=True)
        raise
    stderr_bytes = stderr_path.read_bytes()
    record = {
        "role": f"chr{chromosome}_decompress_pgen",
        "argv": argv,
        "cwd": str(cwd),
        "environment_overrides": {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
        "returncode": completed.returncode,
        "stdout_sha256": _file_sha256(destination),
        "stderr_sha256": hashlib.sha256(stderr_bytes).hexdigest(),
    }
    if completed.returncode != 0:
        tail = stderr_bytes.decode("utf-8", errors="replace")[-2000:]
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"PLINK PGEN decompression for chromosome {chromosome} failed with "
            f"exit code {completed.returncode}: {tail}"
        )
    if destination.stat().st_size <= 0:
        raise ValueError(f"chromosome {chromosome} decompressed PGEN is empty")
    return record


def _parse_raw_to_block(
    *,
    raw_path: Path,
    block_path: Path,
    expected_sample_ids: Sequence[str],
    prune_ids: Sequence[str],
) -> tuple[int, int]:
    """Parse export-A directly into an ordered on-disk float32 block."""

    _require_regular(raw_path, role="PLINK export-A RAW")
    expected_index = {sample: index for index, sample in enumerate(expected_sample_ids)}
    if len(expected_index) != len(expected_sample_ids):
        raise ValueError("expected sample IDs are not unique")
    metadata = ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")
    matrix: np.memmap | None = None
    seen: set[str] = set()
    missing_count = 0
    with raw_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.split()
            if not fields:
                raise ValueError(f"blank line in PLINK RAW at line {line_number}")
            if line_number == 1:
                if tuple(value.lstrip("#") for value in fields[:6]) != metadata:
                    raise ValueError(
                        "PLINK RAW header lacks canonical metadata columns"
                    )
                raw_variant_ids = fields[6:]
                if len(raw_variant_ids) != len(prune_ids) or not raw_variant_ids:
                    raise ValueError("PLINK RAW column count does not match prune.in")
                for raw_id, prune_id in zip(raw_variant_ids, prune_ids, strict=True):
                    if raw_id != prune_id and not raw_id.startswith(prune_id + "_"):
                        raise ValueError(
                            f"PLINK RAW column {raw_id!r} does not match ordered "
                            f"prune.in ID {prune_id!r}"
                        )
                matrix = open_memmap(
                    block_path,
                    mode="w+",
                    dtype=GENOTYPE_DTYPE,
                    shape=(len(expected_sample_ids), len(prune_ids)),
                    fortran_order=False,
                )
                continue
            if matrix is None:
                raise AssertionError("RAW data row preceded header")
            if len(fields) != 6 + len(prune_ids):
                raise ValueError(
                    f"PLINK RAW line {line_number} has {len(fields)} fields; "
                    f"expected {6 + len(prune_ids)}"
                )
            sample_id = fields[1]
            if sample_id not in expected_index:
                raise ValueError(f"PLINK RAW contains unexpected sample {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"PLINK RAW contains duplicate sample {sample_id!r}")
            seen.add(sample_id)
            values = np.empty(len(prune_ids), dtype=GENOTYPE_DTYPE)
            for column, token in enumerate(fields[6:]):
                if token.upper() == "NA" or token == "-9":
                    values[column] = np.nan
                    missing_count += 1
                    continue
                try:
                    value = float(token)
                except ValueError as error:
                    raise ValueError(
                        f"invalid dosage {token!r} at RAW line {line_number}"
                    ) from error
                if not np.isfinite(value) or not 0.0 <= value <= 2.0:
                    raise ValueError(
                        f"dosage at RAW line {line_number} must lie in [0, 2]"
                    )
                values[column] = value
            matrix[expected_index[sample_id], :] = values
    if matrix is None:
        raise ValueError("PLINK RAW is empty")
    missing_samples = sorted(set(expected_sample_ids) - seen)
    if missing_samples:
        raise ValueError(
            f"PLINK RAW lacks frozen cohort samples: {missing_samples[:5]}"
        )
    matrix.flush()
    del matrix
    return len(prune_ids), missing_count


def _run_chromosome(
    *,
    chromosome: str,
    pgen_zst: Path,
    pvar_zst: Path,
    psam: Path,
    keep_path: Path,
    plink2: Path,
    run_work: Path,
    retained_prune_dir: Path,
) -> _ChromosomeOutput:
    block_path = run_work / f"chr{chromosome}.genotypes.npy"
    # PGEN compression ratios vary, and export-A is temporarily text.  This
    # deliberately conservative bound covers an 8x PGEN expansion, a 4x PVAR
    # working allowance, and an 8 GiB fixed RAW/QC margin.  Failure happens
    # before opening the decompression target, so no publishable package can be
    # created from a partial chromosome.
    required_free = 8 * pgen_zst.stat().st_size + 4 * pvar_zst.stat().st_size + 8 * GIB
    _preflight_disk_space(
        run_work,
        required_bytes=required_free,
        role=f"chromosome {chromosome} decompression/QC/export",
    )
    with tempfile.TemporaryDirectory(
        prefix=f"chr{chromosome}_", dir=run_work
    ) as chromosome_temp:
        work = Path(chromosome_temp)
        decompressed = work / f"chr{chromosome}.source.pgen"
        qc_prefix = work / f"chr{chromosome}.qc"
        ld_prefix = work / f"chr{chromosome}.ld"
        export_prefix = work / f"chr{chromosome}.pruned"
        decompression_record = _run_zst_decompress(
            plink2=plink2,
            source=pgen_zst,
            destination=decompressed,
            cwd=work,
            chromosome=chromosome,
        )
        common_tail = ["--threads", "1"]
        qc_argv = [
            str(plink2),
            "--pgen",
            str(decompressed),
            "--pvar",
            str(pvar_zst),
            "--psam",
            str(psam),
            "--keep",
            str(keep_path),
            "--chr",
            chromosome,
            "--snps-only",
            "just-acgt",
            "--max-alleles",
            "2",
            "--rm-dup",
            "exclude-all",
            "--maf",
            format(MAF, ".17g"),
            "--geno",
            format(GENO, ".17g"),
            "--make-pgen",
            *common_tail,
            "--out",
            str(qc_prefix),
        ]
        prune_argv = [
            str(plink2),
            "--pfile",
            str(qc_prefix),
            "--indep-pairwise",
            str(LD_WINDOW),
            str(LD_STEP),
            format(LD_R2, ".17g"),
            "--indep-order",
            str(INDEP_ORDER),
            *common_tail,
            "--out",
            str(ld_prefix),
        ]
        export_argv = [
            str(plink2),
            "--pfile",
            str(qc_prefix),
            "--extract",
            str(ld_prefix) + ".prune.in",
            "--export",
            "A",
            *common_tail,
            "--out",
            str(export_prefix),
        ]
        command_records = (
            decompression_record,
            legacy_preparer._run_command(
                qc_argv, cwd=work, role=f"chr{chromosome}_pgen_qc"
            ),
            legacy_preparer._run_command(
                prune_argv, cwd=work, role=f"chr{chromosome}_ld_prune"
            ),
            legacy_preparer._run_command(
                export_argv, cwd=work, role=f"chr{chromosome}_export_pruned"
            ),
        )
        qc_pvar = Path(str(qc_prefix) + ".pvar")
        prune_in = Path(str(ld_prefix) + ".prune.in")
        raw_path = Path(str(export_prefix) + ".raw")
        qc_ids = legacy_preparer._read_pvar_ids(qc_pvar, expected_chromosome=chromosome)
        prune_ids = legacy_preparer._read_id_file(prune_in, role="prune.in")
        if not prune_ids:
            raise ValueError(f"chromosome {chromosome} LD pruning retained no variants")
        if not set(prune_ids) <= set(qc_ids):
            extras = sorted(set(prune_ids) - set(qc_ids))[:5]
            raise ValueError(
                f"chromosome {chromosome} prune.in is not a QC-PVAR subset: {extras}"
            )
        _, missing_count = _parse_raw_to_block(
            raw_path=raw_path,
            block_path=block_path,
            expected_sample_ids=tuple(_read_keep_iids(keep_path)),
            prune_ids=prune_ids,
        )
        retained = retained_prune_dir / f"chr{chromosome}.prune.in"
        with prune_in.open("rb") as source, retained.open("xb") as destination:
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.chmod(retained, 0o600)
        ephemeral = {
            "decompressed_pgen": _artifact_record(decompressed),
            "qc_pvar": _artifact_record(qc_pvar),
            "export_raw": _artifact_record(raw_path),
        }
        return _ChromosomeOutput(
            chromosome=chromosome,
            genotype_block=block_path,
            snp_ids=tuple(prune_ids),
            qc_variant_count=len(qc_ids),
            retained_variant_count=len(prune_ids),
            missing_count=missing_count,
            command_records=command_records,
            ephemeral_artifacts=ephemeral,
            retained_prune_in=retained,
        )


def _read_keep_iids(path: Path) -> tuple[str, ...]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].lstrip("#") != "IID":
        raise ValueError("internal keep file has an invalid header")
    values = tuple(value for value in lines[1:] if value)
    if len(set(values)) != len(values):
        raise ValueError("internal keep file has duplicate IIDs")
    return values


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    # Passing a ZipInfo to ZipFile.open bypasses the archive-level default.
    # Set this explicitly so the claimed level is also the level actually used.
    info._compresslevel = NPZ_COMPRESSLEVEL
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _write_regular_npy_member(
    archive: zipfile.ZipFile, *, name: str, array: np.ndarray
) -> str:
    with archive.open(_zip_info(f"{name}.npy"), mode="w") as member:
        writer = _HashingWriter(member)
        np.lib.format.write_array(writer, np.asarray(array), allow_pickle=False)
    return writer.hexdigest()


def _write_genotype_npy_member(
    archive: zipfile.ZipFile,
    *,
    outputs: Sequence[_ChromosomeOutput],
    sample_count: int,
) -> str:
    total_snps = sum(output.retained_variant_count for output in outputs)
    # The all-autosome matrix can exceed the classic 2 GiB ZIP member limit.
    # Force ZIP64 up front; zipfile cannot retrofit the larger local header.
    with archive.open(_zip_info("genotypes.npy"), mode="w", force_zip64=True) as member:
        writer = _HashingWriter(member)
        header = {
            "descr": np.lib.format.dtype_to_descr(GENOTYPE_DTYPE),
            "fortran_order": False,
            "shape": (sample_count, total_snps),
        }
        np.lib.format.write_array_header_1_0(writer, header)
        blocks = [np.load(output.genotype_block, mmap_mode="r") for output in outputs]
        try:
            for row in range(sample_count):
                for block in blocks:
                    writer.write(
                        np.asarray(block[row, :], dtype=GENOTYPE_DTYPE).tobytes(
                            order="C"
                        )
                    )
        finally:
            del blocks
    return writer.hexdigest()


def _write_deterministic_npz(
    *,
    path: Path,
    outputs: Sequence[_ChromosomeOutput],
    cohort: _LegacyCohort,
) -> tuple[dict[str, str], int]:
    snp_ids = np.asarray(
        [identifier for output in outputs for identifier in output.snp_ids], dtype=str
    )
    if np.unique(snp_ids).size != snp_ids.size:
        counts = Counter(snp_ids.tolist())
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        raise ValueError(f"pruned SNP IDs are not globally unique: {duplicates[:5]}")
    chromosomes = np.asarray(
        [
            output.chromosome
            for output in outputs
            for _ in range(output.retained_variant_count)
        ],
        dtype=str,
    )
    arrays = {
        "chromosomes": chromosomes,
        "population_labels": cohort.population_labels,
        "sample_ids": cohort.sample_ids,
        "snp_ids": snp_ids,
    }
    hashes: dict[str, str] = {}
    with path.open("xb") as handle:
        with zipfile.ZipFile(
            handle,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=NPZ_COMPRESSLEVEL,
        ) as archive:
            for name in NPZ_KEYS:
                if name == "genotypes":
                    hashes[name] = _write_genotype_npy_member(
                        archive,
                        outputs=outputs,
                        sample_count=cohort.sample_ids.size,
                    )
                else:
                    hashes[name] = _write_regular_npy_member(
                        archive, name=name, array=arrays[name]
                    )
    return hashes, int(snp_ids.size)


def _copy_or_link_exclusive(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
        return
    except OSError as error:
        if error.errno != errno.EXDEV:
            raise
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with (
            source.open("rb") as input_handle,
            os.fdopen(descriptor, "wb", closefd=True) as output_handle,
        ):
            descriptor = -1
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        destination.unlink(missing_ok=True)
        raise


def _install_package(
    *,
    staging_npz: Path,
    staging_manifest: Path,
    outputs: Sequence[_ChromosomeOutput],
    destination: Path,
) -> None:
    parent = destination.parent
    _require_directory(parent, role="output parent")
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite output package: {destination}")
    installed: list[Path] = []
    destination_created = False
    prune_destination_created = False
    prune_destination = destination / PRUNE_DIRECTORY
    try:
        os.mkdir(destination, 0o700)
        destination_created = True
        os.mkdir(prune_destination, 0o700)
        prune_destination_created = True
        for source, target in (
            (staging_npz, destination / NPZ_NAME),
            (staging_manifest, destination / MANIFEST_NAME),
        ):
            _copy_or_link_exclusive(source, target)
            installed.append(target)
        for output in outputs:
            target = prune_destination / output.retained_prune_in.name
            _copy_or_link_exclusive(output.retained_prune_in, target)
            installed.append(target)
    except BaseException:
        for path in reversed(installed):
            path.unlink(missing_ok=True)
        if prune_destination_created:
            try:
                prune_destination.rmdir()
            except FileNotFoundError:
                pass
        if destination_created:
            try:
                destination.rmdir()
            except FileNotFoundError:
                pass
        raise


def _validate_sha256_token(value: str, *, role: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{role} must be a lowercase SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{role} must be a lowercase SHA-256") from error


def _npy_array_sha256(array: np.ndarray) -> str:
    writer = _HashingWriter()
    value = np.asarray(array)
    header = np.lib.format.header_data_from_array_1_0(value)
    np.lib.format.write_array_header_1_0(writer, header)
    if value.flags.f_contiguous and not value.flags.c_contiguous:
        order = "F"
    else:
        order = "C"
    if value.ndim == 0:
        writer.write(value.tobytes(order=order))
    elif order == "C":
        row_count = value.shape[0]
        for start in range(0, row_count, 64):
            writer.write(np.ascontiguousarray(value[start : start + 64]).tobytes())
    else:
        writer.write(np.asfortranarray(value).tobytes(order="F"))
    return writer.hexdigest()


def _expected_package_files() -> set[str]:
    return {
        NPZ_NAME,
        MANIFEST_NAME,
        *(f"{PRUNE_DIRECTORY}/chr{chromosome}.prune.in" for chromosome in AUTOSOMES),
    }


def _validate_exact_package_closure(package_dir: Path) -> None:
    _require_directory(package_dir, role="prepared package")
    actual: set[str] = set()
    for path in package_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"prepared package contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(package_dir).as_posix())
        elif not path.is_dir():
            raise ValueError(f"prepared package contains a non-file entry: {path}")
    expected = _expected_package_files()
    if actual != expected:
        raise ValueError(
            "prepared package file closure mismatch; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    directories = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_dir()
    }
    if directories != {PRUNE_DIRECTORY}:
        raise ValueError("prepared package directory closure mismatch")


def _extract_genotype_member_to_memmap(
    *, archive: zipfile.ZipFile, temporary_directory: Path, expected_data_bytes: int
) -> tuple[np.memmap, str, Path]:
    info = archive.getinfo("genotypes.npy")
    if (
        info.file_size < expected_data_bytes
        or info.file_size > expected_data_bytes + 1024 * 1024
    ):
        raise ValueError(
            "genotypes.npy uncompressed size is inconsistent with prune.in"
        )
    _preflight_disk_space(
        temporary_directory,
        required_bytes=info.file_size + GIB,
        role="streaming genotype validation extraction",
    )
    extracted = temporary_directory / "genotypes.npy"
    digest = hashlib.sha256()
    descriptor = os.open(extracted, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with (
            archive.open(info, mode="r") as source,
            os.fdopen(descriptor, "wb", closefd=True) as destination,
        ):
            descriptor = -1
            while True:
                block = source.read(4 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                destination.write(block)
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        extracted.unlink(missing_ok=True)
        raise
    matrix = np.load(extracted, mmap_mode="r", allow_pickle=False)
    if not isinstance(matrix, np.memmap):
        raise AssertionError("streamed genotype validation did not produce a memmap")
    return matrix, digest.hexdigest(), extracted


def _validate_command_record(
    record: Any, *, expected_role: str, plink2: Path
) -> Sequence[str]:
    value = _exact_keys(
        record,
        {
            "role",
            "argv",
            "cwd",
            "environment_overrides",
            "returncode",
            "stdout_sha256",
            "stderr_sha256",
        },
        role=f"command record {expected_role}",
    )
    if value["role"] != expected_role or value["returncode"] != 0:
        raise ValueError(
            f"command record {expected_role} did not complete successfully"
        )
    if value["environment_overrides"] != {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"}:
        raise ValueError(f"command record {expected_role} environment changed")
    argv = value["argv"]
    if not isinstance(argv, list) or not all(isinstance(token, str) for token in argv):
        raise ValueError(f"command record {expected_role} argv is invalid")
    if not argv or argv[0] != str(plink2):
        raise ValueError(f"command record {expected_role} used another executable")
    if not isinstance(value["cwd"], str) or not value["cwd"]:
        raise ValueError(f"command record {expected_role} cwd is invalid")
    for name in ("stdout_sha256", "stderr_sha256"):
        _validate_sha256_token(value[name], role=f"{expected_role}.{name}")
    return argv


def _validate_fixed_commands(
    records: Any,
    *,
    chromosome: str,
    plink2: Path,
    pgen_zst: Path,
    pvar_zst: Path,
    psam: Path,
) -> None:
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError(f"chromosome {chromosome} must record exactly four commands")
    decompress = _validate_command_record(
        records[0], expected_role=f"chr{chromosome}_decompress_pgen", plink2=plink2
    )
    if decompress != [str(plink2), "--zst-decompress", str(pgen_zst)]:
        raise ValueError(f"chromosome {chromosome} decompression command changed")
    qc = _validate_command_record(
        records[1], expected_role=f"chr{chromosome}_pgen_qc", plink2=plink2
    )
    fixed_qc_pairs = {
        "--pvar": str(pvar_zst),
        "--psam": str(psam),
        "--chr": chromosome,
        "--maf": format(MAF, ".17g"),
        "--geno": format(GENO, ".17g"),
        "--max-alleles": "2",
        "--rm-dup": "exclude-all",
        "--threads": "1",
    }
    for flag, expected in fixed_qc_pairs.items():
        if qc.count(flag) != 1 or qc[qc.index(flag) + 1] != expected:
            raise ValueError(f"chromosome {chromosome} QC flag {flag} changed")
    for required in ("--pgen", "--keep", "--snps-only", "--make-pgen", "--out"):
        if qc.count(required) != 1:
            raise ValueError(f"chromosome {chromosome} QC lacks {required}")
    if qc[qc.index("--snps-only") + 1] != "just-acgt":
        raise ValueError("biallelic A/C/G/T SNP policy changed")
    if not fixed_qc_pairs["--pvar"].endswith(".pvar.zst") or "--hwe" in qc:
        raise ValueError("compressed-PVAR or no-HWE QC policy changed")
    prune = _validate_command_record(
        records[2], expected_role=f"chr{chromosome}_ld_prune", plink2=plink2
    )
    marker = prune.index("--indep-pairwise") if "--indep-pairwise" in prune else -1
    if marker < 0 or prune[marker + 1 : marker + 4] != [
        str(LD_WINDOW),
        str(LD_STEP),
        format(LD_R2, ".17g"),
    ]:
        raise ValueError(f"chromosome {chromosome} LD-pruning parameters changed")
    if prune.count("--indep-order") != 1 or prune[
        prune.index("--indep-order") + 1
    ] != str(INDEP_ORDER):
        raise ValueError(f"chromosome {chromosome} indep-order changed")
    export = _validate_command_record(
        records[3], expected_role=f"chr{chromosome}_export_pruned", plink2=plink2
    )
    if export.count("--extract") != 1 or export.count("--export") != 1:
        raise ValueError(f"chromosome {chromosome} export is not prune.in-only")
    if export[export.index("--export") + 1] != "A":
        raise ValueError(f"chromosome {chromosome} export format changed")
    if not export[export.index("--extract") + 1].endswith(".prune.in"):
        raise ValueError(f"chromosome {chromosome} export did not use prune.in")


def _plink_version_text(plink2: Path) -> str:
    completed = subprocess.run(
        [str(plink2), "--version"],
        check=False,
        capture_output=True,
        env=_command_environment(),
    )
    if completed.returncode != 0:
        raise RuntimeError("plink2 --version failed during validation")
    text = completed.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        text = completed.stderr.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("plink2 --version produced no version text")
    return text


def validate_all_autosome_package(
    *,
    package_dir: str | Path,
    source_manifest: str | Path,
    download_dir: str | Path,
    legacy_prep_manifest: str | Path,
    plink2_path: str | Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    expected_download_completion_sha256: str,
    expected_package_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly validate a package against independent input and manifest pins."""

    package = _lexical_absolute(package_dir)
    source_manifest_path = _lexical_absolute(source_manifest)
    source_directory = _lexical_absolute(download_dir)
    legacy_manifest_path = _lexical_absolute(legacy_prep_manifest)
    plink2 = _lexical_absolute(plink2_path)
    _assert_import_bound_sources_unchanged()
    _validate_sha256_token(
        expected_package_manifest_sha256,
        role="expected_package_manifest_sha256",
    )
    _require_regular(plink2, role="PLINK 2 executable", executable=True)
    _validate_exact_package_closure(package)
    manifest_path = package / MANIFEST_NAME
    observed_manifest_hash = _file_sha256(manifest_path)
    if observed_manifest_hash != expected_package_manifest_sha256:
        raise ValueError(
            "package manifest does not match the independently supplied SHA-256: "
            f"{observed_manifest_hash}"
        )
    source_payload, source_records, completion = _load_pinned_sources(
        source_manifest_path,
        source_directory,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
        expected_download_completion_sha256=expected_download_completion_sha256,
    )
    cohort = _load_legacy_cohort(legacy_manifest_path)
    record_by_role = {
        (str(record.get("chromosome")), record["role"]): record
        for record in source_records
        if record["role"] != "sample_metadata"
    }
    psam_record = next(
        record for record in source_records if record["role"] == "sample_metadata"
    )
    psam = source_directory / Path(psam_record["path"]).name
    psam_check = _read_psam_and_check_cohort(psam, cohort)
    manifest = _load_json(manifest_path, role="all-autosome preparation manifest")
    _exact_keys(
        manifest,
        {
            "cohort_anchor",
            "dataset",
            "downstream_label_boundary",
            "output",
            "plink2",
            "preparation",
            "schema_version",
            "sources",
            "variant_processing",
        },
        role="all-autosome preparation manifest",
    )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected all-autosome preparation schema version")
    expected_preparation = {
        "tool": dict(_IMPORT_PREPARER_SOURCE),
        "runtime_dependencies": {
            "legacy_preparer": dict(_IMPORT_LEGACY_PREPARER_SOURCE),
            "source_downloader": _source_code_record(DOWNLOADER_SOURCE_PATH),
        },
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
    }
    if manifest["preparation"] != expected_preparation:
        raise ValueError("preparation source/dependency closure changed")
    expected_dataset = {
        "input_scope": "all_autosomes",
        "autosomes_present": list(AUTOSOMES),
        "orientation": "samples_by_snps",
        "population_design": "five_superpopulation",
        "population_label_semantics": "1000_Genomes_superpopulation",
        "population_label_is_race": False,
        "ld_pruned": True,
    }
    if manifest["dataset"] != expected_dataset:
        raise ValueError("dataset scope/semantics changed")
    expected_cohort_anchor = {
        "legacy_preparation_manifest": {
            "path": str(legacy_manifest_path),
            "sha256": PINNED_LEGACY_PREPARATION_MANIFEST_SHA256,
        },
        "selected_count": EXPECTED_SAMPLE_COUNT,
        "selected_counts_by_group": {
            population: EXPECTED_PER_POPULATION for population in POPULATIONS
        },
        "selected_sample_ids_sha256": PINNED_SELECTED_SAMPLE_IDS_SHA256,
        "selected_records_sha256": _canonical_sha256(list(cohort.selected_records)),
        "psam_check": psam_check,
    }
    if manifest["cohort_anchor"] != expected_cohort_anchor:
        raise ValueError("frozen cohort anchor changed")
    expected_sources = {
        "external_input_anchors": {
            "expected_source_manifest_sha256": expected_source_manifest_sha256,
            "expected_downloader_source_sha256": expected_downloader_source_sha256,
            "expected_download_completion_sha256": expected_download_completion_sha256,
        },
        "source_manifest": {
            "path": str(source_manifest_path),
            "sha256": PINNED_SOURCE_MANIFEST_SHA256,
            "payload_sha256": _canonical_sha256(source_payload),
        },
        "download_directory": {
            "path": str(source_directory),
            "completion_manifest_sha256": expected_download_completion_sha256,
            "completion_payload_sha256": completion["completion_payload_sha256"],
            "validation_status": completion["status"],
            "file_count": len(source_records),
            "total_bytes": sum(record["bytes"] for record in source_records),
        },
        "authentication_boundary": source_payload["source"]["authentication_boundary"],
    }
    if manifest["sources"] != expected_sources:
        raise ValueError("source/download anchors changed")
    expected_plink = {
        "path": str(plink2),
        "sha256": _file_sha256(plink2),
        "bytes": plink2.stat().st_size,
        "version": _plink_version_text(plink2),
    }
    if manifest["plink2"] != expected_plink:
        raise ValueError("PLINK 2 identity changed")
    boundary = {
        "allowed": [
            "stratified_train_test_split",
            "training_centroids_after_panel_freeze",
            "held_out_scoring",
        ],
        "forbidden": [
            "preprocessing_fit",
            "component_fit",
            "loading_based_panel_ranking",
            "panel_refit",
        ],
        "population_labels_used_for_variant_qc_or_ld": False,
        "panel_selection_used_population_labels": False,
    }
    if manifest["downstream_label_boundary"] != boundary:
        raise ValueError("population-label boundary changed")
    processing = _exact_keys(
        manifest["variant_processing"],
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
    expected_processing_static = {
        "policy": "genuine_qc_then_ld_prune_exact_prune_in_only",
        "cohort_scope": "same_frozen_1000_reference_samples_before_outer_split",
        "population_labels_used_for_variant_qc_or_ld": False,
        "variant_target": None,
        "padding_with_unpruned_variants": False,
        "hwe_filter": None,
        "maf": MAF,
        "maximum_variant_missingness": GENO,
        "biallelic_snv_only": True,
        "duplicate_variant_policy": "plink2_rm_dup_exclude_all",
        "ld_pruning": {
            "method": "plink2_indep_pairwise",
            "window_variants": LD_WINDOW,
            "step_variants": LD_STEP,
            "r2_threshold": LD_R2,
            "indep_order": INDEP_ORDER,
        },
    }
    for key, expected in expected_processing_static.items():
        if processing[key] != expected:
            raise ValueError(f"variant-processing field {key} changed")
    per_chromosome = processing["per_chromosome"]
    if not isinstance(per_chromosome, dict) or set(per_chromosome) != set(AUTOSOMES):
        raise ValueError("per_chromosome coverage is not exactly chromosomes 1..22")
    all_snp_ids: list[str] = []
    chromosome_values: list[str] = []
    total_missing = 0
    total_qc = 0
    for chromosome in AUTOSOMES:
        entry = _exact_keys(
            per_chromosome[chromosome],
            {
                "commands",
                "ephemeral_artifacts",
                "ld_pruned_variant_count",
                "qc_variant_count",
                "retained_prune_in",
                "source_inputs",
            },
            role=f"per_chromosome[{chromosome}]",
        )
        pgen_record = record_by_role[(chromosome, "genotype_matrix_zstd")]
        pvar_record = record_by_role[(chromosome, "variant_metadata_zstd")]
        pgen = source_directory / Path(pgen_record["path"]).name
        pvar = source_directory / Path(pvar_record["path"]).name
        expected_input_records = {
            "pgen_zst": {
                "path": str(pgen),
                "bytes": pgen_record["bytes"],
                "sha256": pgen_record["sha256"],
            },
            "pvar_zst": {
                "path": str(pvar),
                "bytes": pvar_record["bytes"],
                "sha256": pvar_record["sha256"],
            },
            "psam": {
                "path": str(psam),
                "bytes": psam_record["bytes"],
                "sha256": psam_record["sha256"],
            },
        }
        if entry["source_inputs"] != expected_input_records:
            raise ValueError(f"chromosome {chromosome} source binding changed")
        _validate_fixed_commands(
            entry["commands"],
            chromosome=chromosome,
            plink2=plink2,
            pgen_zst=pgen,
            pvar_zst=pvar,
            psam=psam,
        )
        if (
            not isinstance(entry["qc_variant_count"], int)
            or entry["qc_variant_count"] <= 0
        ):
            raise ValueError(f"chromosome {chromosome} QC count is invalid")
        prune_path = package / PRUNE_DIRECTORY / f"chr{chromosome}.prune.in"
        prune_ids = legacy_preparer._read_id_file(prune_path, role="retained prune.in")
        if not prune_ids:
            raise ValueError(f"chromosome {chromosome} retained prune.in is empty")
        expected_prune = {
            "path": f"{PRUNE_DIRECTORY}/chr{chromosome}.prune.in",
            "bytes": prune_path.stat().st_size,
            "sha256": _file_sha256(prune_path),
            "variant_count": len(prune_ids),
            "variant_ids_sha256": _text_list_sha256(prune_ids),
        }
        if entry["retained_prune_in"] != expected_prune:
            raise ValueError(
                f"chromosome {chromosome} retained prune.in record changed"
            )
        if entry["ld_pruned_variant_count"] != len(prune_ids):
            raise ValueError(f"chromosome {chromosome} prune count is inconsistent")
        ephemeral = _exact_keys(
            entry["ephemeral_artifacts"],
            {"decompressed_pgen", "export_raw", "qc_pvar"},
            role=f"chromosome {chromosome} ephemeral artifacts",
        )
        for role, artifact in ephemeral.items():
            value = _exact_keys(
                artifact, {"bytes", "sha256"}, role=f"{chromosome}.{role}"
            )
            if not isinstance(value["bytes"], int) or value["bytes"] <= 0:
                raise ValueError(
                    f"chromosome {chromosome} {role} byte count is invalid"
                )
            _validate_sha256_token(value["sha256"], role=f"{chromosome}.{role}.sha256")
        all_snp_ids.extend(prune_ids)
        chromosome_values.extend([chromosome] * len(prune_ids))
        total_qc += entry["qc_variant_count"]
    if len(set(all_snp_ids)) != len(all_snp_ids):
        raise ValueError("retained prune.in files contain globally duplicate SNP IDs")
    npz_path = package / NPZ_NAME
    expected_shape = (EXPECTED_SAMPLE_COUNT, len(all_snp_ids))
    genotype_member_hash = ""
    total_missing = 0
    with zipfile.ZipFile(npz_path, mode="r") as archive:
        names = archive.namelist()
        expected_members = [f"{name}.npy" for name in NPZ_KEYS]
        if names != expected_members or len(set(names)) != len(names):
            raise ValueError("NPZ member order/closure changed")
        for info in archive.infolist():
            if (
                info.date_time != (1980, 1, 1, 0, 0, 0)
                or info.compress_type != zipfile.ZIP_DEFLATED
                or info.create_system != 3
                or ((info.external_attr >> 16) & 0o777) != 0o600
                or info.flag_bits & 0x1
            ):
                raise ValueError("NPZ ZIP metadata is not deterministic/nonencrypted")
            if info.filename == "genotypes.npy" and info.extract_version != 45:
                raise ValueError("genotypes.npy was not emitted with forced ZIP64")
        with tempfile.TemporaryDirectory(
            prefix="rspca_validate_genotypes_", dir=package.parent
        ) as temporary_value:
            genotypes, genotype_member_hash, _ = _extract_genotype_member_to_memmap(
                archive=archive,
                temporary_directory=Path(temporary_value),
                expected_data_bytes=(
                    EXPECTED_SAMPLE_COUNT * len(all_snp_ids) * GENOTYPE_DTYPE.itemsize
                ),
            )
            if genotypes.shape != expected_shape or genotypes.dtype != GENOTYPE_DTYPE:
                raise ValueError("genotype shape/dtype changed")
            for start in range(0, EXPECTED_SAMPLE_COUNT, 32):
                block = np.asarray(genotypes[start : start + 32])
                if np.any(np.isinf(block)):
                    raise ValueError("genotypes contain infinity")
                finite = np.isfinite(block)
                if np.any(block[finite] < 0.0) or np.any(block[finite] > 2.0):
                    raise ValueError("genotypes contain dosage outside [0, 2]")
                total_missing += int(np.count_nonzero(np.isnan(block)))
            del genotypes
    with np.load(npz_path, allow_pickle=False) as archive:
        if tuple(archive.files) != NPZ_KEYS:
            raise ValueError("NPZ key closure/order changed")
        arrays = {name: archive[name] for name in NPZ_KEYS if name != "genotypes"}
    if not np.array_equal(arrays["sample_ids"].astype(str), cohort.sample_ids):
        raise ValueError("NPZ sample IDs differ from the frozen cohort")
    if not np.array_equal(
        arrays["population_labels"].astype(str), cohort.population_labels
    ):
        raise ValueError("NPZ population labels differ from the frozen cohort")
    if arrays["snp_ids"].astype(str).tolist() != all_snp_ids:
        raise ValueError("NPZ SNP IDs are not the exact concatenated prune.in lists")
    if arrays["chromosomes"].astype(str).tolist() != chromosome_values:
        raise ValueError("NPZ chromosome labels do not match prune.in blocks")
    output = _exact_keys(
        manifest["output"],
        {
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
        },
        role="output",
    )
    expected_output = {
        "path": NPZ_NAME,
        "sha256": _file_sha256(npz_path),
        "bytes": npz_path.stat().st_size,
        "npz_keys": list(NPZ_KEYS),
        "npz_container": {
            "format": "zip_npz",
            "compression": "ZIP_DEFLATED",
            "compresslevel": NPZ_COMPRESSLEVEL,
            "genotypes_force_zip64": True,
            "member_order": [f"{name}.npy" for name in NPZ_KEYS],
            "member_timestamp": [1980, 1, 1, 0, 0, 0],
            "member_mode_octal": "0600",
        },
        "shape": [EXPECTED_SAMPLE_COUNT, len(all_snp_ids)],
        "dtype": GENOTYPE_DTYPE.str,
        "missing_count": total_missing,
        "total_qc_variants": total_qc,
        "array_sha256": {
            name: (
                genotype_member_hash
                if name == "genotypes"
                else _npy_array_sha256(arrays[name])
            )
            for name in NPZ_KEYS
        },
    }
    expected_output["array_receipts"] = {
        name: {
            "sha256": expected_output["array_sha256"][name],
            "shape": list(
                expected_shape if name == "genotypes" else arrays[name].shape
            ),
            "dtype": (
                GENOTYPE_DTYPE.str if name == "genotypes" else arrays[name].dtype.str
            ),
        }
        for name in NPZ_KEYS
    }
    if output != expected_output:
        raise ValueError("NPZ output record is inconsistent")
    return {
        "schema_version": SCHEMA_VERSION,
        "package_dir": str(package),
        "package_manifest_sha256": observed_manifest_hash,
        "output_npz_sha256": expected_output["sha256"],
        "samples": EXPECTED_SAMPLE_COUNT,
        "ld_pruned_snps": len(all_snp_ids),
        "autosomes": len(AUTOSOMES),
        "self_consistency_verified": True,
    }


def prepare_all_autosome_dataset(
    *,
    source_manifest: str | Path,
    download_dir: str | Path,
    legacy_prep_manifest: str | Path,
    plink2_path: str | Path,
    output_dir: str | Path,
    expected_source_manifest_sha256: str,
    expected_downloader_source_sha256: str,
    expected_download_completion_sha256: str,
    scratch_parent: str | Path | None = None,
) -> dict[str, Any]:
    """Run the frozen all-autosome preparation and install one closed package."""

    source_manifest_path = _lexical_absolute(source_manifest)
    source_directory = _lexical_absolute(download_dir)
    legacy_manifest_path = _lexical_absolute(legacy_prep_manifest)
    plink2 = _lexical_absolute(plink2_path)
    destination = _lexical_absolute(output_dir)
    _assert_import_bound_sources_unchanged()
    _require_regular(plink2, role="PLINK 2 executable", executable=True)
    if _lexists(destination):
        raise FileExistsError(f"refusing to overwrite output package: {destination}")
    source_payload, source_records, completion = _load_pinned_sources(
        source_manifest_path,
        source_directory,
        expected_source_manifest_sha256=expected_source_manifest_sha256,
        expected_downloader_source_sha256=expected_downloader_source_sha256,
        expected_download_completion_sha256=expected_download_completion_sha256,
    )
    cohort = _load_legacy_cohort(legacy_manifest_path)
    record_by_role = {
        (str(record.get("chromosome")), record["role"]): record
        for record in source_records
        if record["role"] != "sample_metadata"
    }
    psam_record = next(
        record for record in source_records if record["role"] == "sample_metadata"
    )
    psam = source_directory / Path(psam_record["path"]).name
    psam_check = _read_psam_and_check_cohort(psam, cohort)
    scratch_root = (
        _lexical_absolute(scratch_parent)
        if scratch_parent is not None
        else destination.parent
    )
    _require_directory(destination.parent, role="output parent")
    _require_directory(scratch_root, role="scratch parent")
    canonical_source_directory = source_directory.resolve(strict=True)
    canonical_destination = destination.parent.resolve(strict=True) / destination.name
    canonical_scratch_root = scratch_root.resolve(strict=True)
    if _is_within(canonical_destination, canonical_source_directory):
        raise ValueError("output package must be outside the sealed download directory")
    if _is_within(canonical_scratch_root, canonical_source_directory):
        raise ValueError("scratch parent must be outside the sealed download directory")
    with tempfile.TemporaryDirectory(
        prefix="rspca_all_autosomes_", dir=scratch_root
    ) as value:
        run_work = Path(value)
        keep_path = run_work / "frozen_1000_samples.keep"
        _write_keep_file(keep_path, cohort.sample_ids.tolist())
        retained_prune_dir = run_work / PRUNE_DIRECTORY
        retained_prune_dir.mkdir(mode=0o700)
        plink_version, _ = legacy_preparer._plink_version(plink2, cwd=run_work)
        outputs: list[_ChromosomeOutput] = []
        for chromosome in AUTOSOMES:
            pgen_record = record_by_role[(chromosome, "genotype_matrix_zstd")]
            pvar_record = record_by_role[(chromosome, "variant_metadata_zstd")]
            output = _run_chromosome(
                chromosome=chromosome,
                pgen_zst=source_directory / Path(pgen_record["path"]).name,
                pvar_zst=source_directory / Path(pvar_record["path"]).name,
                psam=psam,
                keep_path=keep_path,
                plink2=plink2,
                run_work=run_work,
                retained_prune_dir=retained_prune_dir,
            )
            outputs.append(output)
        if tuple(output.chromosome for output in outputs) != AUTOSOMES:
            raise AssertionError("all-autosome preparation lost chromosome coverage")
        staging_npz = run_work / NPZ_NAME
        projected_genotype_bytes = (
            cohort.sample_ids.size
            * sum(output.retained_variant_count for output in outputs)
            * GENOTYPE_DTYPE.itemsize
        )
        _preflight_disk_space(
            run_work,
            required_bytes=2 * projected_genotype_bytes + 3 * GIB,
            role="final deterministic NPZ write plus prepublication validation",
        )
        array_hashes, snp_count = _write_deterministic_npz(
            path=staging_npz, outputs=outputs, cohort=cohort
        )
        if snp_count <= 0:
            raise ValueError("all-autosome preparation retained no SNPs")
        total_missing = sum(output.missing_count for output in outputs)
        snp_id_array = np.asarray(
            [identifier for output in outputs for identifier in output.snp_ids],
            dtype=str,
        )
        chromosome_array = np.asarray(
            [
                output.chromosome
                for output in outputs
                for _ in range(output.retained_variant_count)
            ],
            dtype=str,
        )
        array_shapes_and_dtypes = {
            "chromosomes": (list(chromosome_array.shape), chromosome_array.dtype.str),
            "genotypes": ([EXPECTED_SAMPLE_COUNT, snp_count], GENOTYPE_DTYPE.str),
            "population_labels": (
                list(cohort.population_labels.shape),
                cohort.population_labels.dtype.str,
            ),
            "sample_ids": (list(cohort.sample_ids.shape), cohort.sample_ids.dtype.str),
            "snp_ids": (list(snp_id_array.shape), snp_id_array.dtype.str),
        }
        per_chromosome: dict[str, Any] = {}
        for output in outputs:
            pgen_record = record_by_role[(output.chromosome, "genotype_matrix_zstd")]
            pvar_record = record_by_role[(output.chromosome, "variant_metadata_zstd")]
            pgen = source_directory / Path(pgen_record["path"]).name
            pvar = source_directory / Path(pvar_record["path"]).name
            per_chromosome[output.chromosome] = {
                "source_inputs": {
                    "pgen_zst": {
                        "path": str(pgen),
                        "bytes": pgen_record["bytes"],
                        "sha256": pgen_record["sha256"],
                    },
                    "pvar_zst": {
                        "path": str(pvar),
                        "bytes": pvar_record["bytes"],
                        "sha256": pvar_record["sha256"],
                    },
                    "psam": {
                        "path": str(psam),
                        "bytes": psam_record["bytes"],
                        "sha256": psam_record["sha256"],
                    },
                },
                "qc_variant_count": output.qc_variant_count,
                "ld_pruned_variant_count": output.retained_variant_count,
                "commands": list(output.command_records),
                "ephemeral_artifacts": output.ephemeral_artifacts,
                "retained_prune_in": {
                    "path": f"{PRUNE_DIRECTORY}/{output.retained_prune_in.name}",
                    "bytes": output.retained_prune_in.stat().st_size,
                    "sha256": _file_sha256(output.retained_prune_in),
                    "variant_count": output.retained_variant_count,
                    "variant_ids_sha256": _text_list_sha256(output.snp_ids),
                },
            }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "preparation": {
                "tool": dict(_IMPORT_PREPARER_SOURCE),
                "runtime_dependencies": {
                    "legacy_preparer": dict(_IMPORT_LEGACY_PREPARER_SOURCE),
                    "source_downloader": _source_code_record(DOWNLOADER_SOURCE_PATH),
                },
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
            },
            "dataset": {
                "input_scope": "all_autosomes",
                "autosomes_present": list(AUTOSOMES),
                "orientation": "samples_by_snps",
                "population_design": "five_superpopulation",
                "population_label_semantics": "1000_Genomes_superpopulation",
                "population_label_is_race": False,
                "ld_pruned": True,
            },
            "sources": {
                "external_input_anchors": {
                    "expected_source_manifest_sha256": expected_source_manifest_sha256,
                    "expected_downloader_source_sha256": expected_downloader_source_sha256,
                    "expected_download_completion_sha256": expected_download_completion_sha256,
                },
                "source_manifest": {
                    "path": str(source_manifest_path),
                    "sha256": PINNED_SOURCE_MANIFEST_SHA256,
                    "payload_sha256": _canonical_sha256(source_payload),
                },
                "download_directory": {
                    "path": str(source_directory),
                    "completion_manifest_sha256": expected_download_completion_sha256,
                    "completion_payload_sha256": completion[
                        "completion_payload_sha256"
                    ],
                    "validation_status": completion["status"],
                    "file_count": len(source_records),
                    "total_bytes": sum(record["bytes"] for record in source_records),
                },
                "authentication_boundary": source_payload["source"][
                    "authentication_boundary"
                ],
            },
            "cohort_anchor": {
                "legacy_preparation_manifest": {
                    "path": str(legacy_manifest_path),
                    "sha256": PINNED_LEGACY_PREPARATION_MANIFEST_SHA256,
                },
                "selected_count": EXPECTED_SAMPLE_COUNT,
                "selected_counts_by_group": {
                    population: EXPECTED_PER_POPULATION for population in POPULATIONS
                },
                "selected_sample_ids_sha256": PINNED_SELECTED_SAMPLE_IDS_SHA256,
                "selected_records_sha256": _canonical_sha256(
                    list(cohort.selected_records)
                ),
                "psam_check": psam_check,
            },
            "plink2": {
                "path": str(plink2),
                "sha256": _file_sha256(plink2),
                "bytes": plink2.stat().st_size,
                "version": plink_version,
            },
            "variant_processing": {
                "policy": "genuine_qc_then_ld_prune_exact_prune_in_only",
                "cohort_scope": "same_frozen_1000_reference_samples_before_outer_split",
                "population_labels_used_for_variant_qc_or_ld": False,
                "variant_target": None,
                "padding_with_unpruned_variants": False,
                "hwe_filter": None,
                "maf": MAF,
                "maximum_variant_missingness": GENO,
                "biallelic_snv_only": True,
                "duplicate_variant_policy": "plink2_rm_dup_exclude_all",
                "ld_pruning": {
                    "method": "plink2_indep_pairwise",
                    "window_variants": LD_WINDOW,
                    "step_variants": LD_STEP,
                    "r2_threshold": LD_R2,
                    "indep_order": INDEP_ORDER,
                },
                "per_chromosome": per_chromosome,
            },
            "downstream_label_boundary": {
                "allowed": [
                    "stratified_train_test_split",
                    "training_centroids_after_panel_freeze",
                    "held_out_scoring",
                ],
                "forbidden": [
                    "preprocessing_fit",
                    "component_fit",
                    "loading_based_panel_ranking",
                    "panel_refit",
                ],
                "population_labels_used_for_variant_qc_or_ld": False,
                "panel_selection_used_population_labels": False,
            },
            "output": {
                "path": NPZ_NAME,
                "sha256": _file_sha256(staging_npz),
                "bytes": staging_npz.stat().st_size,
                "npz_keys": list(NPZ_KEYS),
                "npz_container": {
                    "format": "zip_npz",
                    "compression": "ZIP_DEFLATED",
                    "compresslevel": NPZ_COMPRESSLEVEL,
                    "genotypes_force_zip64": True,
                    "member_order": [f"{name}.npy" for name in NPZ_KEYS],
                    "member_timestamp": [1980, 1, 1, 0, 0, 0],
                    "member_mode_octal": "0600",
                },
                "shape": [EXPECTED_SAMPLE_COUNT, snp_count],
                "dtype": GENOTYPE_DTYPE.str,
                "missing_count": total_missing,
                "total_qc_variants": sum(output.qc_variant_count for output in outputs),
                "array_sha256": array_hashes,
                "array_receipts": {
                    name: {
                        "sha256": array_hashes[name],
                        "shape": array_shapes_and_dtypes[name][0],
                        "dtype": array_shapes_and_dtypes[name][1],
                    }
                    for name in NPZ_KEYS
                },
            },
        }
        staging_manifest = run_work / MANIFEST_NAME
        _assert_import_bound_sources_unchanged()
        staging_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_hash = _file_sha256(staging_manifest)
        candidate_package = run_work / "candidate_package"
        _install_package(
            staging_npz=staging_npz,
            staging_manifest=staging_manifest,
            outputs=outputs,
            destination=candidate_package,
        )
        validated_receipt = validate_all_autosome_package(
            package_dir=candidate_package,
            source_manifest=source_manifest_path,
            download_dir=source_directory,
            legacy_prep_manifest=legacy_manifest_path,
            plink2_path=plink2,
            expected_source_manifest_sha256=expected_source_manifest_sha256,
            expected_downloader_source_sha256=expected_downloader_source_sha256,
            expected_download_completion_sha256=expected_download_completion_sha256,
            expected_package_manifest_sha256=manifest_hash,
        )
        # Publish only byte-identical hard links/copies of the already validated
        # candidate.  A failure before or during this exclusive installation
        # leaves no final output directory.
        _install_package(
            staging_npz=candidate_package / NPZ_NAME,
            staging_manifest=candidate_package / MANIFEST_NAME,
            outputs=outputs,
            destination=destination,
        )
    return {**validated_receipt, "package_dir": str(destination)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        child = subparsers.add_parser(command)
        child.add_argument("--source-manifest", required=True, type=Path)
        child.add_argument("--download-dir", required=True, type=Path)
        child.add_argument("--legacy-prep-manifest", required=True, type=Path)
        child.add_argument("--plink2", required=True, type=Path)
        child.add_argument("--output-dir", required=True, type=Path)
        child.add_argument("--expected-source-manifest-sha256", required=True)
        child.add_argument("--expected-downloader-source-sha256", required=True)
        child.add_argument("--expected-download-completion-sha256", required=True)
        if command == "run":
            child.add_argument("--scratch-parent", type=Path)
        else:
            child.add_argument("--expected-package-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = prepare_all_autosome_dataset(
                source_manifest=args.source_manifest,
                download_dir=args.download_dir,
                legacy_prep_manifest=args.legacy_prep_manifest,
                plink2_path=args.plink2,
                output_dir=args.output_dir,
                expected_source_manifest_sha256=(args.expected_source_manifest_sha256),
                expected_downloader_source_sha256=(
                    args.expected_downloader_source_sha256
                ),
                expected_download_completion_sha256=(
                    args.expected_download_completion_sha256
                ),
                scratch_parent=args.scratch_parent,
            )
        else:
            result = validate_all_autosome_package(
                package_dir=args.output_dir,
                source_manifest=args.source_manifest,
                download_dir=args.download_dir,
                legacy_prep_manifest=args.legacy_prep_manifest,
                plink2_path=args.plink2,
                expected_source_manifest_sha256=(args.expected_source_manifest_sha256),
                expected_downloader_source_sha256=(
                    args.expected_downloader_source_sha256
                ),
                expected_download_completion_sha256=(
                    args.expected_download_completion_sha256
                ),
                expected_package_manifest_sha256=(
                    args.expected_package_manifest_sha256
                ),
            )
    except (
        FileExistsError,
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
        PermissionError,
        RuntimeError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare a deterministic 1000 Genomes chr20--22 AIM input package.

This tool deliberately prepares only the historical GitHub data scope:
chromosomes 20, 21, and 22 from the 1000 Genomes phase-3 v5b release.  It
does not call the result genome-wide.  Variant QC and LD pruning are performed
by an explicitly supplied PLINK 2 executable.  Only variants in PLINK's
``.prune.in`` files are exported; there is no target-width padding path.

The output NPZ follows ``rspca.genomics_aim.MultiAutosomeGenotypeData``:
``samples x SNPs`` dosages plus sample, variant, chromosome, and population
metadata.  Population metadata is used here only to choose a declared
reference-population cohort.  Downstream AIM panel fitting/ranking must remain
label-free as enforced by ``rspca.genomics_aim``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import numpy as np
from numpy.typing import NDArray


SCHEMA_VERSION = "rspca-1kg-phase3-v5b-chr20-22-preparation-v1"
CHROMOSOMES = ("20", "21", "22")
DESIGN_GROUPS = {
    "three_continent": ("AFR", "EAS", "EUR"),
    "five_superpopulation": ("AFR", "AMR", "EAS", "EUR", "SAS"),
}
OFFICIAL_RELEASE_ROOT = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
)
OFFICIAL_VCF_BASENAMES = {
    chromosome: (
        f"ALL.chr{chromosome}.phase3_shapeit2_mvncall_integrated_"
        "v5b.20130502.genotypes.vcf.gz"
    )
    for chromosome in CHROMOSOMES
}
OFFICIAL_PANEL_BASENAME = "integrated_call_samples_v3.20130502.ALL.panel"
OFFICIAL_VCF_URLS = {
    chromosome: f"{OFFICIAL_RELEASE_ROOT}/{basename}"
    for chromosome, basename in OFFICIAL_VCF_BASENAMES.items()
}
OFFICIAL_PANEL_URL = f"{OFFICIAL_RELEASE_ROOT}/{OFFICIAL_PANEL_BASENAME}"

FloatArray = NDArray[np.float32]
StringArray = NDArray[np.str_]


@dataclass(frozen=True)
class PanelRecord:
    sample_id: str
    subpopulation: str
    superpopulation: str


@dataclass(frozen=True)
class _ExportedChromosome:
    chromosome: str
    genotypes: FloatArray
    sample_ids: StringArray
    snp_ids: StringArray
    qc_variant_count: int
    retained_variant_count: int
    command_records: tuple[Mapping[str, Any], ...]
    artifact_hashes: Mapping[str, Mapping[str, Any]]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _text_list_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode("utf-8")
    return _bytes_sha256(payload)


def _array_sha256(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return _bytes_sha256(buffer.getvalue())


def _source_record(path: Path, source_url: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "source_url": source_url,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _require_regular_file(path: Path, *, role: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{role} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{role} is not a regular file: {path}")


def _validate_official_names(vcfs: Mapping[str, Path], panel_path: Path) -> None:
    for chromosome in CHROMOSOMES:
        observed = vcfs[chromosome].name
        expected = OFFICIAL_VCF_BASENAMES[chromosome]
        if observed != expected:
            raise ValueError(
                f"chromosome {chromosome} VCF basename must be {expected!r}; "
                f"observed {observed!r}"
            )
    if panel_path.name != OFFICIAL_PANEL_BASENAME:
        raise ValueError(
            f"panel basename must be {OFFICIAL_PANEL_BASENAME!r}; "
            f"observed {panel_path.name!r}"
        )


def _read_panel(path: Path) -> tuple[PanelRecord, ...]:
    """Parse the official whitespace-delimited integrated sample panel."""

    rows: list[PanelRecord] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        header: list[str] | None = None
        indices: dict[str, int] | None = None
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            fields = line.split()
            if header is None:
                header = fields
                missing = {"sample", "pop", "super_pop"} - set(header)
                if missing:
                    raise ValueError(
                        f"panel header is missing columns {sorted(missing)}: {path}"
                    )
                indices = {name: header.index(name) for name in header}
                continue
            assert indices is not None
            if len(fields) != len(header):
                raise ValueError(
                    f"panel line {line_number} has {len(fields)} fields; "
                    f"expected {len(header)}"
                )
            sample_id = fields[indices["sample"]].strip()
            subpopulation = fields[indices["pop"]].strip()
            superpopulation = fields[indices["super_pop"]].strip()
            if not sample_id or not subpopulation or not superpopulation:
                raise ValueError(f"panel line {line_number} contains an empty field")
            rows.append(
                PanelRecord(
                    sample_id=sample_id,
                    subpopulation=subpopulation,
                    superpopulation=superpopulation,
                )
            )
    if header is None:
        raise ValueError(f"panel is empty: {path}")
    if not rows:
        raise ValueError(f"panel contains no sample rows: {path}")
    counts = Counter(row.sample_id for row in rows)
    duplicates = sorted(sample for sample, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"panel contains duplicate sample IDs: {duplicates[:5]}")
    return tuple(rows)


def _selection_rank(record: PanelRecord, seed: int) -> tuple[str, str]:
    token = (
        f"{SCHEMA_VERSION}\0{seed}\0{record.superpopulation}\0"
        f"{record.sample_id}"
    ).encode("utf-8")
    return hashlib.sha256(token).hexdigest(), record.sample_id


def select_panel_samples(
    records: Sequence[PanelRecord],
    *,
    population_design: str,
    cap_per_group: int | None,
    selection_seed: int,
) -> tuple[PanelRecord, ...]:
    """Select a transparent deterministic superpopulation-balanced cohort."""

    if population_design not in DESIGN_GROUPS:
        raise ValueError(
            f"population_design must be one of {tuple(DESIGN_GROUPS)}"
        )
    if isinstance(selection_seed, bool) or not isinstance(selection_seed, int):
        raise TypeError("selection_seed must be a non-negative integer")
    if selection_seed < 0:
        raise ValueError("selection_seed must be a non-negative integer")
    if cap_per_group is not None:
        if isinstance(cap_per_group, bool) or not isinstance(cap_per_group, int):
            raise TypeError("cap_per_group must be an integer or None")
        if cap_per_group < 2:
            raise ValueError("cap_per_group must be at least 2")

    selected: list[PanelRecord] = []
    for group in DESIGN_GROUPS[population_design]:
        candidates = [row for row in records if row.superpopulation == group]
        if not candidates:
            raise ValueError(f"panel contains no samples for superpopulation {group}")
        candidates.sort(key=lambda row: _selection_rank(row, selection_seed))
        if cap_per_group is not None:
            if len(candidates) < cap_per_group:
                raise ValueError(
                    f"superpopulation {group} has {len(candidates)} samples, fewer "
                    f"than cap_per_group={cap_per_group}"
                )
            candidates = candidates[:cap_per_group]
        selected.extend(candidates)

    # Output order never depends on PLINK's VCF traversal order.  Hash ranking
    # chooses capped membership; group/sample sorting fixes final row order.
    selected.sort(key=lambda row: (row.superpopulation, row.sample_id))
    return tuple(selected)


def _write_keep_file(path: Path, records: Sequence[PanelRecord]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            # --double-id assigns the VCF sample ID to both FID and IID.
            handle.write(f"{record.sample_id}\t{record.sample_id}\n")


def _run_command(
    argv: Sequence[str],
    *,
    cwd: Path,
    role: str,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment.update({"LC_ALL": "C", "LANG": "C", "TZ": "UTC"})
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=False,
    )
    stdout_path = cwd / f"{role}.stdout.log"
    stderr_path = cwd / f"{role}.stderr.log"
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    record = {
        "role": role,
        "argv": list(argv),
        "cwd": str(cwd),
        "environment_overrides": {"LC_ALL": "C", "LANG": "C", "TZ": "UTC"},
        "returncode": completed.returncode,
        "stdout_sha256": _bytes_sha256(completed.stdout),
        "stderr_sha256": _bytes_sha256(completed.stderr),
    }
    if completed.returncode != 0:
        tail = completed.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(
            f"PLINK command {role!r} failed with exit code "
            f"{completed.returncode}: {tail}"
        )
    return record


def _plink_version(plink2: Path, *, cwd: Path) -> tuple[str, Mapping[str, Any]]:
    record = _run_command([str(plink2), "--version"], cwd=cwd, role="plink2_version")
    stdout = (cwd / "plink2_version.stdout.log").read_text(
        encoding="utf-8", errors="replace"
    ).strip()
    if not stdout:
        stderr = (cwd / "plink2_version.stderr.log").read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        stdout = stderr
    if not stdout:
        raise ValueError("plink2 --version produced no version text")
    return stdout, record


def _read_id_file(path: Path, *, role: str) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"PLINK did not create {role}: {path}")
    values = tuple(
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    )
    if len(set(values)) != len(values):
        raise ValueError(f"{role} contains duplicate variant IDs")
    return values


def _read_pvar_ids(path: Path, *, expected_chromosome: str) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"PLINK did not create PVAR: {path}")
    identifiers: list[str] = []
    header: list[str] | None = None
    chrom_index = -1
    id_index = -1
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("##"):
                continue
            fields = line.split()
            if header is None:
                header = fields
                normalized = [value.lstrip("#") for value in header]
                if "CHROM" not in normalized or "ID" not in normalized:
                    raise ValueError(f"PVAR header lacks CHROM/ID columns: {path}")
                chrom_index = normalized.index("CHROM")
                id_index = normalized.index("ID")
                continue
            if max(chrom_index, id_index) >= len(fields):
                raise ValueError(f"ragged PVAR row at line {line_number}: {path}")
            chromosome = fields[chrom_index].lower().removeprefix("chr")
            if chromosome != expected_chromosome:
                raise ValueError(
                    f"PVAR contains chromosome {fields[chrom_index]!r}; "
                    f"expected only {expected_chromosome}"
                )
            identifier = fields[id_index]
            if not identifier or identifier == ".":
                raise ValueError(f"PVAR has an empty/missing ID at line {line_number}")
            identifiers.append(identifier)
    if header is None or not identifiers:
        raise ValueError(f"PVAR contains no variants: {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"PVAR variant IDs are not unique: {path}")
    return tuple(identifiers)


def _parse_raw(path: Path) -> tuple[FloatArray, StringArray, StringArray]:
    """Read a PLINK ``--export A`` file with strict dosage validation."""

    if not path.is_file():
        raise FileNotFoundError(f"PLINK did not create export-A RAW file: {path}")
    sample_ids: list[str] = []
    genotype_rows: list[list[float]] = []
    variant_ids: list[str] | None = None
    metadata = ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE")
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.split()
            if not fields:
                raise ValueError(f"blank line in PLINK RAW file at line {line_number}")
            if line_number == 1:
                observed = tuple(value.lstrip("#") for value in fields[:6])
                if observed != metadata or len(fields) == 6:
                    raise ValueError(
                        "PLINK RAW header must contain the six canonical metadata "
                        "columns followed by at least one variant"
                    )
                variant_ids = fields[6:]
                if len(set(variant_ids)) != len(variant_ids):
                    raise ValueError("PLINK RAW variant IDs are not unique")
                continue
            assert variant_ids is not None
            if len(fields) != 6 + len(variant_ids):
                raise ValueError(
                    f"PLINK RAW line {line_number} has {len(fields)} fields; "
                    f"expected {6 + len(variant_ids)}"
                )
            sample_id = fields[1]
            if not sample_id:
                raise ValueError(f"empty IID in PLINK RAW line {line_number}")
            values: list[float] = []
            for variant_id, token in zip(variant_ids, fields[6:], strict=True):
                if token.upper() == "NA" or token == "-9":
                    values.append(float("nan"))
                    continue
                try:
                    value = float(token)
                except ValueError as error:
                    raise ValueError(
                        f"invalid dosage {token!r} for {variant_id!r} at "
                        f"line {line_number}"
                    ) from error
                if not np.isfinite(value) or not 0.0 <= value <= 2.0:
                    raise ValueError(
                        f"dosage for {variant_id!r} at line {line_number} "
                        "must lie in [0, 2] or be NA/-9"
                    )
                values.append(value)
            sample_ids.append(sample_id)
            genotype_rows.append(values)
    if variant_ids is None or not sample_ids:
        raise ValueError(f"PLINK RAW file has no samples: {path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("PLINK RAW sample IDs are not unique")
    return (
        np.asarray(genotype_rows, dtype=np.float32),
        np.asarray(sample_ids, dtype=str),
        np.asarray(variant_ids, dtype=str),
    )


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "bytes": path.stat().st_size}


def _run_chromosome(
    *,
    chromosome: str,
    vcf_path: Path,
    keep_path: Path,
    plink2: Path,
    work_dir: Path,
    maf: float,
    geno: float,
    ld_window: int,
    ld_window_unit: str,
    ld_step: int,
    ld_r2: float,
) -> _ExportedChromosome:
    qc_prefix = work_dir / f"chr{chromosome}.qc"
    ld_prefix = work_dir / f"chr{chromosome}.ld"
    export_prefix = work_dir / f"chr{chromosome}.pruned"
    common_tail = ["--threads", "1"]

    import_argv = [
        str(plink2),
        "--vcf",
        str(vcf_path),
        "--double-id",
        "--keep",
        str(keep_path),
        "--set-all-var-ids",
        "@:#:$r:$a",
        "--rm-dup",
        "exclude-all",
        "--snps-only",
        "just-acgt",
        "--max-alleles",
        "2",
        "--maf",
        format(maf, ".17g"),
        "--geno",
        format(geno, ".17g"),
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
        f"{ld_window}kb" if ld_window_unit == "kb" else str(ld_window),
        str(ld_step),
        format(ld_r2, ".17g"),
        "--indep-order",
        "2",
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
        _run_command(
            import_argv, cwd=work_dir, role=f"chr{chromosome}_import_qc"
        ),
        _run_command(prune_argv, cwd=work_dir, role=f"chr{chromosome}_ld_prune"),
        _run_command(
            export_argv, cwd=work_dir, role=f"chr{chromosome}_export_pruned"
        ),
    )

    pvar_path = Path(str(qc_prefix) + ".pvar")
    prune_in_path = Path(str(ld_prefix) + ".prune.in")
    prune_out_path = Path(str(ld_prefix) + ".prune.out")
    raw_path = Path(str(export_prefix) + ".raw")
    qc_ids = _read_pvar_ids(pvar_path, expected_chromosome=chromosome)
    retained_ids = _read_id_file(prune_in_path, role="prune.in")
    excluded_ids = _read_id_file(prune_out_path, role="prune.out")
    retained_set = set(retained_ids)
    excluded_set = set(excluded_ids)
    if retained_set & excluded_set:
        raise ValueError(f"chromosome {chromosome} prune.in/prune.out overlap")
    if retained_set | excluded_set != set(qc_ids):
        missing = sorted(set(qc_ids) - retained_set - excluded_set)[:5]
        extra = sorted((retained_set | excluded_set) - set(qc_ids))[:5]
        raise ValueError(
            f"chromosome {chromosome} pruning files do not partition QC variants; "
            f"missing={missing}, extra={extra}"
        )
    if not retained_ids:
        raise ValueError(f"chromosome {chromosome} LD pruning retained no variants")

    genotypes, sample_ids, exported_ids = _parse_raw(raw_path)
    if exported_ids.size != len(retained_ids):
        raise ValueError(
            f"chromosome {chromosome} export has {exported_ids.size} variants but "
            f"prune.in has {len(retained_ids)}"
        )
    artifacts = {
        "qc_pvar": _artifact_record(pvar_path),
        "prune_in": _artifact_record(prune_in_path),
        "prune_out": _artifact_record(prune_out_path),
        "export_raw": _artifact_record(raw_path),
    }
    return _ExportedChromosome(
        chromosome=chromosome,
        genotypes=genotypes,
        sample_ids=sample_ids,
        snp_ids=exported_ids,
        qc_variant_count=len(qc_ids),
        retained_variant_count=len(retained_ids),
        command_records=command_records,
        artifact_hashes=artifacts,
    )


def _npy_payload(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
    return buffer.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an NPZ with fixed member order and ZIP metadata."""

    with path.open("xb") as handle:
        with zipfile.ZipFile(handle, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(arrays):
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_payload(np.asarray(arrays[name])))


def _exclusive_install(source: Path, destination: Path) -> None:
    """Install by exclusive hard-link creation without overwriting a target."""

    os.link(source, destination)
    source.unlink()


def _validate_probability(value: float, *, name: str, upper_inclusive: bool) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error
    upper_ok = number <= 1.0 if upper_inclusive else number < 1.0
    if not np.isfinite(number) or number < 0.0 or not upper_ok:
        bracket = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must lie in {bracket}")
    return number


def prepare_1kg_aim_dataset(
    *,
    vcfs: Mapping[str, str | Path],
    panel_path: str | Path,
    plink2_path: str | Path,
    output_npz: str | Path,
    manifest_path: str | Path,
    population_design: str,
    cap_per_group: int | None = None,
    selection_seed: int = 20260830,
    maf: float = 0.05,
    geno: float = 0.05,
    ld_window: int = 50,
    ld_window_unit: str = "variants",
    ld_step: int = 5,
    ld_r2: float = 0.2,
    work_dir: str | Path | None = None,
    require_official_basenames: bool = True,
) -> dict[str, Any]:
    """Build an exclusive-create NPZ and provenance manifest.

    The returned receipt includes the SHA-256 of the manifest file.  Pass that
    digest as ``source_manifest_sha256`` to the downstream AIM protocol.
    """

    if set(vcfs) != set(CHROMOSOMES):
        raise ValueError(f"vcfs must contain exactly chromosome keys {CHROMOSOMES}")
    resolved_vcfs = {
        chromosome: Path(vcfs[chromosome]).expanduser().resolve()
        for chromosome in CHROMOSOMES
    }
    panel = Path(panel_path).expanduser().resolve()
    plink2 = Path(plink2_path).expanduser().resolve()
    destination = Path(output_npz).expanduser().resolve()
    manifest_destination = Path(manifest_path).expanduser().resolve()
    if destination == manifest_destination:
        raise ValueError("output_npz and manifest_path must be distinct")
    if destination.suffix.lower() != ".npz":
        raise ValueError("output_npz must end in .npz")
    if manifest_destination.suffix.lower() != ".json":
        raise ValueError("manifest_path must end in .json")
    for chromosome, path in resolved_vcfs.items():
        _require_regular_file(path, role=f"chromosome {chromosome} VCF")
    _require_regular_file(panel, role="integrated sample panel")
    _require_regular_file(plink2, role="plink2 executable")
    if not os.access(plink2, os.X_OK):
        raise PermissionError(f"plink2 path is not executable: {plink2}")
    if require_official_basenames:
        _validate_official_names(resolved_vcfs, panel)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite output NPZ: {destination}")
    if manifest_destination.exists():
        raise FileExistsError(
            f"refusing to overwrite provenance manifest: {manifest_destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)

    maf_value = _validate_probability(maf, name="maf", upper_inclusive=False)
    if maf_value >= 0.5:
        raise ValueError("maf must be below 0.5")
    geno_value = _validate_probability(geno, name="geno", upper_inclusive=True)
    ld_r2_value = _validate_probability(ld_r2, name="ld_r2", upper_inclusive=True)
    if ld_window_unit not in {"variants", "kb"}:
        raise ValueError("ld_window_unit must be 'variants' or 'kb'")
    for name, value in (("ld_window", ld_window), ("ld_step", ld_step)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if ld_window_unit == "kb" and ld_step != 1:
        raise ValueError(
            "PLINK 2 requires ld_step=1 when --indep-pairwise uses a kb window"
        )

    panel_records = _read_panel(panel)
    selected = select_panel_samples(
        panel_records,
        population_design=population_design,
        cap_per_group=cap_per_group,
        selection_seed=selection_seed,
    )
    expected_sample_ids = np.asarray([row.sample_id for row in selected], dtype=str)
    labels = np.asarray([row.superpopulation for row in selected], dtype=str)

    if work_dir is None:
        manager: Any = tempfile.TemporaryDirectory(prefix="rspca_1kg_aim_")
    else:
        retained_work = Path(work_dir).expanduser().resolve()
        retained_work.mkdir(parents=True, exist_ok=False)
        manager = nullcontext(str(retained_work))

    with manager as work_value:
        work = Path(work_value).resolve()
        keep_path = work / "selected_samples.keep"
        _write_keep_file(keep_path, selected)
        version, version_command = _plink_version(plink2, cwd=work)
        chromosome_outputs = tuple(
            _run_chromosome(
                chromosome=chromosome,
                vcf_path=resolved_vcfs[chromosome],
                keep_path=keep_path,
                plink2=plink2,
                work_dir=work,
                maf=maf_value,
                geno=geno_value,
                    ld_window=ld_window,
                    ld_window_unit=ld_window_unit,
                    ld_step=ld_step,
                ld_r2=ld_r2_value,
            )
            for chromosome in CHROMOSOMES
        )

        aligned_blocks: list[FloatArray] = []
        snp_blocks: list[StringArray] = []
        chromosome_blocks: list[StringArray] = []
        expected_set = set(expected_sample_ids.tolist())
        for block in chromosome_outputs:
            observed = block.sample_ids.tolist()
            if set(observed) != expected_set or len(observed) != len(expected_set):
                missing = sorted(expected_set - set(observed))[:5]
                extra = sorted(set(observed) - expected_set)[:5]
                raise ValueError(
                    f"chromosome {block.chromosome} PLINK sample set differs from "
                    f"the selected panel cohort; missing={missing}, extra={extra}"
                )
            index = {sample: position for position, sample in enumerate(observed)}
            order = np.asarray([index[sample] for sample in expected_sample_ids], dtype=int)
            aligned_blocks.append(block.genotypes[order])
            snp_blocks.append(block.snp_ids)
            chromosome_blocks.append(
                np.full(
                    block.snp_ids.size,
                    block.chromosome,
                    dtype=f"<U{len(block.chromosome)}",
                )
            )

        genotypes = np.ascontiguousarray(np.concatenate(aligned_blocks, axis=1))
        snp_ids = np.concatenate(snp_blocks)
        chromosomes = np.concatenate(chromosome_blocks)
        if np.unique(snp_ids).size != snp_ids.size:
            counts = Counter(snp_ids.tolist())
            duplicates = sorted(key for key, count in counts.items() if count > 1)
            raise ValueError(f"exported SNP IDs are not globally unique: {duplicates[:5]}")
        if set(chromosomes.tolist()) != set(CHROMOSOMES):
            raise AssertionError("prepared legacy scope lost one or more chromosomes")

        arrays: dict[str, np.ndarray] = {
            "genotypes": genotypes,
            "sample_ids": expected_sample_ids,
            "snp_ids": snp_ids,
            "chromosomes": chromosomes,
            "population_labels": labels,
        }
        temporary_npz = destination.parent / (
            f".{destination.name}.{os.getpid()}.preparing"
        )
        temporary_manifest = manifest_destination.parent / (
            f".{manifest_destination.name}.{os.getpid()}.preparing"
        )
        if temporary_npz.exists() or temporary_manifest.exists():
            raise FileExistsError("deterministic temporary output path already exists")
        installed_npz = False
        try:
            _write_deterministic_npz(temporary_npz, arrays)
            npz_hash = _file_sha256(temporary_npz)
            source_inputs = {
                "vcfs": {
                    chromosome: _source_record(
                        resolved_vcfs[chromosome], OFFICIAL_VCF_URLS[chromosome]
                    )
                    for chromosome in CHROMOSOMES
                },
                "integrated_panel": _source_record(panel, OFFICIAL_PANEL_URL),
            }
            group_counts = Counter(labels.tolist())
            manifest: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "preparation_tool": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": _file_sha256(Path(__file__).resolve()),
                    "bytes": Path(__file__).resolve().stat().st_size,
                    "configuration": {
                        "population_design": population_design,
                        "cap_per_group": cap_per_group,
                        "selection_seed": selection_seed,
                        "maf": maf_value,
                        "geno": geno_value,
                        "ld_window": ld_window,
                        "ld_window_unit": ld_window_unit,
                        "ld_step": ld_step,
                        "ld_r2": ld_r2_value,
                    },
                },
                "dataset": {
                    "input_scope": "legacy_chr20_22",
                    "autosomes_present": list(CHROMOSOMES),
                    "orientation": "samples_by_snps",
                    "population_design": population_design,
                    "population_label_semantics": "1000_Genomes_superpopulation",
                    "population_label_is_race": False,
                    "ld_pruned": True,
                },
                "sources": source_inputs,
                "source_authentication_boundary": (
                    "canonical release URLs are declared and local bytes are SHA-256 "
                    "hashed; this receipt does not independently authenticate those "
                    "hashes against a publisher-signed checksum"
                ),
                "plink2": {
                    "path": str(plink2),
                    "sha256": _file_sha256(plink2),
                    "bytes": plink2.stat().st_size,
                    "version": version,
                    "version_command": version_command,
                },
                "sample_selection": {
                    "algorithm": "sha256_rank_within_superpopulation_then_group_sample_order",
                    "selection_seed": selection_seed,
                    "cap_per_group": cap_per_group,
                    "eligible_superpopulations": list(
                        DESIGN_GROUPS[population_design]
                    ),
                    "selected_count": len(selected),
                    "selected_counts_by_group": dict(sorted(group_counts.items())),
                    "selected_sample_ids_sha256": _text_list_sha256(
                        expected_sample_ids.tolist()
                    ),
                    "selected_records": [
                        {
                            "sample_id": row.sample_id,
                            "subpopulation": row.subpopulation,
                            "population_label": row.superpopulation,
                        }
                        for row in selected
                    ],
                },
                "variant_processing": {
                    "policy": "genuine_qc_then_ld_prune_only",
                    "cohort_scope": "full_selected_reference_cohort_before_outer_split",
                    "population_labels_used_for_variant_qc_or_ld": False,
                    "variant_target": None,
                    "padding_with_unpruned_variants": False,
                    "hwe_filter": None,
                    "maf": maf_value,
                    "maximum_variant_missingness": geno_value,
                    "biallelic_snv_only": True,
                    "duplicate_variant_policy": "plink2_rm_dup_exclude_all",
                    "ld_pruning": {
                        "method": "plink2_indep_pairwise",
                        "window": ld_window,
                        "window_unit": ld_window_unit,
                        "step_variants": ld_step,
                        "r2_threshold": ld_r2_value,
                        "indep_order": 2,
                    },
                    "per_chromosome": {
                        block.chromosome: {
                            "qc_variant_count": block.qc_variant_count,
                            "ld_pruned_variant_count": block.retained_variant_count,
                            "commands": list(block.command_records),
                            "intermediate_artifacts": block.artifact_hashes,
                        }
                        for block in chromosome_outputs
                    },
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
                    "panel_selection_used_population_labels": False,
                },
                "output": {
                    "path": str(destination),
                    "sha256": npz_hash,
                    "bytes": temporary_npz.stat().st_size,
                    "npz_keys": sorted(arrays),
                    "shape": [int(genotypes.shape[0]), int(genotypes.shape[1])],
                    "dtype": genotypes.dtype.str,
                    "missing_count": int(np.count_nonzero(np.isnan(genotypes))),
                    "array_sha256": {
                        name: _array_sha256(array) for name, array in arrays.items()
                    },
                },
            }
            temporary_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            _exclusive_install(temporary_npz, destination)
            installed_npz = True
            try:
                _exclusive_install(temporary_manifest, manifest_destination)
            except BaseException:
                # Roll back only the exact NPZ link created by this invocation.
                destination.unlink(missing_ok=True)
                installed_npz = False
                raise
        finally:
            temporary_npz.unlink(missing_ok=True)
            temporary_manifest.unlink(missing_ok=True)

    if not installed_npz:
        raise AssertionError("output installation did not complete")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "output_npz": str(destination),
        "output_npz_sha256": _file_sha256(destination),
        "manifest": str(manifest_destination),
        "source_manifest_sha256": _file_sha256(manifest_destination),
        "input_scope": "legacy_chr20_22",
        "population_design": population_design,
        "samples": int(expected_sample_ids.size),
        "ld_pruned_snps": int(snp_ids.size),
    }
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chr20-vcf", required=True, type=Path)
    parser.add_argument("--chr21-vcf", required=True, type=Path)
    parser.add_argument("--chr22-vcf", required=True, type=Path)
    parser.add_argument("--panel", required=True, type=Path)
    parser.add_argument("--plink2", required=True, type=Path)
    parser.add_argument(
        "--population-design", required=True, choices=tuple(DESIGN_GROUPS)
    )
    parser.add_argument("--cap-per-group", type=int)
    parser.add_argument("--selection-seed", type=int, default=20260830)
    parser.add_argument("--maf", type=float, default=0.05)
    parser.add_argument("--geno", type=float, default=0.05)
    parser.add_argument("--ld-window", type=int, default=50)
    parser.add_argument(
        "--ld-window-unit", choices=("variants", "kb"), default="variants"
    )
    parser.add_argument(
        "--ld-step",
        type=int,
        default=5,
        help="must be 1 when --ld-window-unit=kb (PLINK 2 contract)",
    )
    parser.add_argument("--ld-r2", type=float, default=0.2)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = prepare_1kg_aim_dataset(
            vcfs={
                "20": args.chr20_vcf,
                "21": args.chr21_vcf,
                "22": args.chr22_vcf,
            },
            panel_path=args.panel,
            plink2_path=args.plink2,
            output_npz=args.output_npz,
            manifest_path=args.manifest,
            population_design=args.population_design,
            cap_per_group=args.cap_per_group,
            selection_seed=args.selection_seed,
            maf=args.maf,
            geno=args.geno,
            ld_window=args.ld_window,
            ld_window_unit=args.ld_window_unit,
            ld_step=args.ld_step,
            ld_r2=args.ld_r2,
            work_dir=args.work_dir,
        )
    except (FileExistsError, FileNotFoundError, PermissionError, RuntimeError, ValueError) as error:
        build_parser().error(str(error))
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

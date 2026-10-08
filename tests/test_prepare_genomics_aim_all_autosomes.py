from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import textwrap
import zipfile

import numpy as np
import pytest

from tools import download_genomics_all_autosome_sources as downloader
from tools import prepare_genomics_aim_all_autosomes as preparer


ROOT = Path(__file__).resolve().parents[1]
PINNED_SOURCE_MANIFEST = (
    ROOT / "operations" / "genomics_all_autosome_hf_source_manifest_2026-08-31.json"
)
PINNED_LEGACY_MANIFEST = (
    ROOT
    / "results"
    / "genomics_aim_chr20_22_formal_20seeds_20260830"
    / "provenance"
    / "five_superpopulation"
    / "1kg_chr20_22_5superpop_200each_ld.manifest.json"
)


FAKE_PLINK = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import re
import sys

argv = sys.argv[1:]
log_path = os.environ.get("FAKE_PLINK_LOG")
if log_path:
    with Path(log_path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(argv) + "\n")

if argv == ["--version"]:
    print("PLINK v2.00a7.4LM fake all-autosome test build")
    raise SystemExit(0)

def value(flag):
    return argv[argv.index(flag) + 1]

if "--zst-decompress" in argv:
    source = Path(value("--zst-decompress"))
    sys.stdout.buffer.write(b"decompressed:" + source.read_bytes())
    raise SystemExit(0)

out = Path(value("--out"))
out.parent.mkdir(parents=True, exist_ok=True)

if "--pgen" in argv:
    pvar = Path(value("--pvar"))
    match = re.search(r"chr(\d+)_hg38", pvar.name)
    if not match:
        raise SystemExit("cannot infer chromosome from PVAR")
    chromosome = match.group(1)
    if os.environ.get("FAKE_PLINK_FAIL_CHR") == chromosome:
        raise SystemExit("injected chromosome failure")
    keep = Path(value("--keep"))
    samples = [
        line.strip()
        for line in keep.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    Path(str(out) + ".pgen").write_bytes(b"fake-qc-pgen")
    Path(str(out) + ".psam").write_text(
        "#FID\tIID\n" + "".join(f"0\t{sample}\n" for sample in samples),
        encoding="utf-8",
    )
    Path(str(out) + ".pvar").write_text(
        "##fileformat=PVARv1\n"
        "#CHROM\tPOS\tID\tREF\tALT\n"
        f"{chromosome}\t100\trs{chromosome}01\tA\tG\n"
        f"{chromosome}\t200\trs{chromosome}02\tC\tT\n"
        f"{chromosome}\t300\trs{chromosome}03\tG\tA\n",
        encoding="utf-8",
    )
elif "--indep-pairwise" in argv:
    pfile = Path(value("--pfile"))
    identifiers = [
        line.split()[2]
        for line in Path(str(pfile) + ".pvar").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    Path(str(out) + ".prune.in").write_text(
        identifiers[0] + "\n" + identifiers[2] + "\n", encoding="utf-8"
    )
    Path(str(out) + ".prune.out").write_text(
        identifiers[1] + "\n", encoding="utf-8"
    )
elif "--export" in argv:
    pfile = Path(value("--pfile"))
    variants = [
        line.strip()
        for line in Path(value("--extract")).read_text().splitlines()
        if line.strip()
    ]
    samples = [
        line.split()[1]
        for line in Path(str(pfile) + ".psam").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if os.environ.get("FAKE_PLINK_DROP_SAMPLE") == "1":
        samples = samples[:-1]
    chromosome = re.match(r"rs(\d+)", variants[0]).group(1)
    with Path(str(out) + ".raw").open("w", encoding="utf-8") as handle:
        handle.write(
            "#FID IID PAT MAT SEX PHENOTYPE "
            + " ".join(variant + "_ALT" for variant in variants)
            + "\n"
        )
        for position, sample in enumerate(reversed(samples)):
            first = "NA" if position == 0 else str((position + int(chromosome)) % 3)
            second = str((2 * position + int(chromosome)) % 3)
            handle.write(f"0 {sample} 0 0 0 -9 {first} {second}\n")
else:
    raise SystemExit(f"unrecognized fake PLINK command: {argv}")
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_fake_curl(path: Path) -> None:
    path.write_text(
        "#!/bin/sh\nprintf 'curl 8.0 fake test build\\n'\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _patch_tiny_protocol(monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    # Including chr10 reproduces lexicographic JSON key order (1, 10, 2),
    # while the protocol order remains numeric (1, 2, 10).
    chromosomes = ("1", "2", "10")
    monkeypatch.setattr(preparer, "AUTOSOMES", chromosomes)
    monkeypatch.setattr(preparer, "EXPECTED_SAMPLE_COUNT", 10)
    monkeypatch.setattr(preparer, "EXPECTED_PER_POPULATION", 2)
    monkeypatch.setattr(preparer, "GIB", 1)
    monkeypatch.setattr(downloader, "EXPECTED_AUTOSOMES", chromosomes)
    monkeypatch.setattr(downloader, "EXPECTED_FILE_COUNT", 2 * len(chromosomes) + 1)
    return chromosomes


def _selected_records() -> list[dict[str, str]]:
    records = []
    for population in preparer.POPULATIONS:
        for index in range(2):
            records.append(
                {
                    "sample_id": f"{population}_{index:02d}",
                    "subpopulation": f"{population}_SUB",
                    "population_label": population,
                }
            )
    return records


def _write_legacy_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, list[dict[str, str]]]:
    records = _selected_records()
    sample_ids = [record["sample_id"] for record in records]
    selected_hash = preparer._text_list_sha256(sample_ids)
    payload = {
        "schema_version": preparer.legacy_preparer.SCHEMA_VERSION,
        "dataset": {
            "input_scope": "legacy_chr20_22",
            "population_design": "five_superpopulation",
            "population_label_semantics": "1000_Genomes_superpopulation",
            "population_label_is_race": False,
            "ld_pruned": True,
        },
        "sample_selection": {
            "selected_count": 10,
            "selected_counts_by_group": {
                population: 2 for population in preparer.POPULATIONS
            },
            "selected_sample_ids_sha256": selected_hash,
            "selected_records": records,
        },
    }
    path = tmp_path / "legacy.manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        preparer, "PINNED_LEGACY_PREPARATION_MANIFEST_SHA256", _sha(path)
    )
    monkeypatch.setattr(preparer, "PINNED_SELECTED_SAMPLE_IDS_SHA256", selected_hash)
    return path, records


def _write_source_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    chromosomes: tuple[str, ...],
    records: list[dict[str, str]],
) -> tuple[Path, Path]:
    download = tmp_path / "download"
    download.mkdir()
    file_payloads: dict[str, bytes] = {}
    for chromosome in chromosomes:
        file_payloads[f"data/chr{chromosome}_hg38.pgen.zst"] = (
            f"fake compressed PGEN {chromosome}\n".encode()
        )
        file_payloads[f"data/chr{chromosome}_hg38_rs.pvar.zst"] = (
            f"fake compressed PVAR {chromosome}\n".encode()
        )
    psam_lines = ["#IID\tSuperPop"]
    psam_lines.extend(
        f"{record['sample_id']}\t{record['population_label']}" for record in records
    )
    psam_lines.extend(["EXTRA_1\tAFR", "EXTRA_2\tEUR"])
    file_payloads["data/hg38_corrected.psam"] = ("\n".join(psam_lines) + "\n").encode()
    for relative, payload in file_payloads.items():
        (download / Path(relative).name).write_bytes(payload)

    source_files: list[dict] = []
    for chromosome in chromosomes:
        for role, suffix, pointer in (
            ("genotype_matrix_zstd", ".pgen.zst", "1"),
            ("variant_metadata_zstd", "_rs.pvar.zst", "2"),
        ):
            relative = f"data/chr{chromosome}_hg38{suffix}"
            payload = file_payloads[relative]
            source_files.append(
                {
                    "bytes": len(payload),
                    "chromosome": int(chromosome),
                    "lfs_pointer_oid": pointer * 40,
                    "path": relative,
                    "role": role,
                    "sha256": _payload_sha(payload),
                }
            )
    psam_payload = file_payloads["data/hg38_corrected.psam"]
    source_files.append(
        {
            "bytes": len(psam_payload),
            "git_blob_oid": "4b0605075174eeb04e842ee2083a671c07379bfa",
            "path": "data/hg38_corrected.psam",
            "role": "sample_metadata",
            "sha256": _payload_sha(psam_payload),
        }
    )
    total_bytes = sum(record["bytes"] for record in source_files)
    monkeypatch.setattr(downloader, "EXPECTED_TOTAL_BYTES", total_bytes)
    source_payload = {
        "files": source_files,
        "schema_version": downloader.SCHEMA_VERSION,
        "scope": {
            "autosomes": list(chromosomes),
            "file_count": len(source_files),
            "mitochondrial_or_contig_files_included": False,
            "sex_chromosomes_included": False,
            "total_bytes": total_bytes,
        },
        "source": {
            "authentication_boundary": "fake test bytes with exact SHA pins",
            "dataset_id": downloader.EXPECTED_DATASET_ID,
            "provider": "offline fake",
            "repository_url": "https://example.invalid/fake",
            "revision": downloader.EXPECTED_REVISION,
            "upstream_declared_by_mirror": "offline fake",
        },
    }
    source_manifest = tmp_path / "source.manifest.json"
    source_manifest.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_hash = _sha(source_manifest)
    monkeypatch.setattr(preparer, "PINNED_SOURCE_MANIFEST_SHA256", source_hash)
    monkeypatch.setattr(downloader, "PINNED_SOURCE_MANIFEST_SHA256", source_hash)
    downloader_hash = _sha(Path(downloader.__file__))
    anchors = downloader._capture_input_anchors(
        source_manifest=source_manifest,
        expected_source_manifest_sha256=source_hash,
        expected_downloader_source_sha256=downloader_hash,
    )
    with downloader._output_lock(download, exclusive=True):
        downloader._seal_locked(
            anchors=anchors,
            output_dir=download,
            replace_completion=False,
            expected_existing_completion_sha256=None,
        )

    def in_process_fake_fresh_validation(**kwargs):
        test_anchors = downloader._capture_input_anchors(
            source_manifest=kwargs["source_manifest"],
            expected_source_manifest_sha256=kwargs["expected_source_manifest_sha256"],
            expected_downloader_source_sha256=kwargs[
                "expected_downloader_source_sha256"
            ],
        )
        with downloader._output_lock(kwargs["download_dir"], exclusive=False):
            return downloader._validate_download_directory_locked(
                anchors=test_anchors,
                output_dir=kwargs["download_dir"],
                expected_completion_sha256=kwargs[
                    "expected_download_completion_sha256"
                ],
            )

    monkeypatch.setattr(
        preparer,
        "_fresh_downloader_validation",
        in_process_fake_fresh_validation,
    )
    return source_manifest, download


def _write_fake_plink(tmp_path: Path) -> Path:
    path = tmp_path / "fake_plink2"
    path.write_text(textwrap.dedent(FAKE_PLINK), encoding="utf-8")
    path.chmod(0o700)
    return path


def _fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, Path]:
    chromosomes = _patch_tiny_protocol(monkeypatch)
    legacy, records = _write_legacy_manifest(tmp_path, monkeypatch)
    source_manifest, download = _write_source_package(
        tmp_path, monkeypatch, chromosomes, records
    )
    plink2 = _write_fake_plink(tmp_path)
    return source_manifest, download, legacy, plink2


def _run(
    *,
    source_manifest: Path,
    download: Path,
    legacy: Path,
    plink2: Path,
    output: Path,
    scratch_parent: Path | None = None,
) -> dict:
    return preparer.prepare_all_autosome_dataset(
        source_manifest=source_manifest,
        download_dir=download,
        legacy_prep_manifest=legacy,
        plink2_path=plink2,
        output_dir=output,
        expected_source_manifest_sha256=_sha(source_manifest),
        expected_downloader_source_sha256=_sha(Path(downloader.__file__)),
        expected_download_completion_sha256=_sha(download / downloader.COMPLETION_NAME),
        scratch_parent=scratch_parent,
    )


def test_repository_pins_cover_exact_real_sources_and_frozen_cohort() -> None:
    assert _sha(PINNED_SOURCE_MANIFEST) == preparer.PINNED_SOURCE_MANIFEST_SHA256
    source_payload = json.loads(PINNED_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    records = downloader.validate_source_manifest(source_payload)
    assert len(records) == 45
    assert {
        str(record["chromosome"]) for record in records if "chromosome" in record
    } == set(preparer.AUTOSOMES)
    assert _sha(PINNED_LEGACY_MANIFEST) == (
        preparer.PINNED_LEGACY_PREPARATION_MANIFEST_SHA256
    )
    cohort = preparer._load_legacy_cohort(PINNED_LEGACY_MANIFEST)
    assert cohort.sample_ids.shape == (1000,)
    assert Counter(cohort.population_labels.tolist()) == Counter(
        {population: 200 for population in preparer.POPULATIONS}
    )
    assert preparer.MAF == 0.05
    assert preparer.GENO == 0.05
    assert (preparer.LD_WINDOW, preparer.LD_STEP, preparer.LD_R2) == (50, 5, 0.2)


def test_fake_plink_smoke_is_deterministic_and_prune_in_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    command_log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("FAKE_PLINK_LOG", str(command_log))
    first = tmp_path / "first"
    second = tmp_path / "second"

    receipt_first = _run(
        source_manifest=source_manifest,
        download=download,
        legacy=legacy,
        plink2=plink2,
        output=first,
    )
    receipt_second = _run(
        source_manifest=source_manifest,
        download=download,
        legacy=legacy,
        plink2=plink2,
        output=second,
    )

    assert receipt_first["self_consistency_verified"] is True
    assert receipt_first["samples"] == 10
    assert receipt_first["ld_pruned_snps"] == 6
    assert receipt_first["autosomes"] == 3
    assert receipt_first["output_npz_sha256"] == receipt_second["output_npz_sha256"]
    assert _sha(first / preparer.NPZ_NAME) == _sha(second / preparer.NPZ_NAME)
    preparer._validate_exact_package_closure(first)
    assert sorted(
        path.name for path in (first / preparer.PRUNE_DIRECTORY).iterdir()
    ) == [
        "chr1.prune.in",
        "chr10.prune.in",
        "chr2.prune.in",
    ]
    with np.load(first / preparer.NPZ_NAME, allow_pickle=False) as archive:
        assert tuple(archive.files) == preparer.NPZ_KEYS
        assert archive["genotypes"].shape == (10, 6)
        assert archive["genotypes"].dtype == np.dtype("<f4")
        assert archive["sample_ids"].tolist() == [
            record["sample_id"] for record in _selected_records()
        ]
        assert archive["population_labels"].tolist() == [
            record["population_label"] for record in _selected_records()
        ]
        assert archive["chromosomes"].tolist() == ["1", "1", "2", "2", "10", "10"]
        assert archive["snp_ids"].tolist() == [
            "rs101",
            "rs103",
            "rs201",
            "rs203",
            "rs1001",
            "rs1003",
        ]
        assert int(np.count_nonzero(np.isnan(archive["genotypes"]))) == 3
    manifest = json.loads((first / preparer.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == preparer.SCHEMA_VERSION
    assert manifest["dataset"]["input_scope"] == "all_autosomes"
    assert manifest["cohort_anchor"]["selected_count"] == 10
    assert manifest["cohort_anchor"]["psam_check"] == {
        "rows": 12,
        "selected_samples_present": 10,
        "superpopulation_column_checked": True,
        "selected_sample_ids_sha256": preparer.PINNED_SELECTED_SAMPLE_IDS_SHA256,
    }
    assert manifest["output"]["npz_container"] == {
        "format": "zip_npz",
        "compression": "ZIP_DEFLATED",
        "compresslevel": preparer.NPZ_COMPRESSLEVEL,
        "genotypes_force_zip64": True,
        "member_order": [f"{name}.npy" for name in preparer.NPZ_KEYS],
        "member_timestamp": [1980, 1, 1, 0, 0, 0],
        "member_mode_octal": "0600",
    }
    assert preparer._zip_info("probe.npy")._compresslevel == (
        preparer.NPZ_COMPRESSLEVEL
    )
    with zipfile.ZipFile(first / preparer.NPZ_NAME) as archive:
        genotype_info = archive.getinfo("genotypes.npy")
        assert genotype_info.extract_version == 45
        assert ((genotype_info.external_attr >> 16) & 0o777) == 0o600
    assert set(manifest["output"]["array_receipts"]) == set(preparer.NPZ_KEYS)
    for name, record in manifest["output"]["array_receipts"].items():
        assert record["sha256"] == manifest["output"]["array_sha256"][name]
        assert set(record) == {"dtype", "shape", "sha256"}
    processing = manifest["variant_processing"]
    assert processing["padding_with_unpruned_variants"] is False
    assert processing["variant_target"] is None
    assert processing["hwe_filter"] is None
    assert processing["population_labels_used_for_variant_qc_or_ld"] is False
    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    flattened = [token for command in commands for token in command]
    assert "--hwe" not in flattened
    assert "--extract" in flattened
    assert flattened.count("--zst-decompress") == 6
    assert flattened.count("--indep-pairwise") == 6
    assert flattened.count("--maf") == 6
    assert flattened.count("--geno") == 6
    assert not list(tmp_path.glob("rspca_all_autosomes_*"))


def test_validator_rejects_extra_file_and_prune_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    package = tmp_path / "package"
    receipt = _run(
        source_manifest=source_manifest,
        download=download,
        legacy=legacy,
        plink2=plink2,
        output=package,
    )
    extra = package / "unexpected.txt"
    extra.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure mismatch"):
        preparer.validate_all_autosome_package(
            package_dir=package,
            source_manifest=source_manifest,
            download_dir=download,
            legacy_prep_manifest=legacy,
            plink2_path=plink2,
            expected_source_manifest_sha256=_sha(source_manifest),
            expected_downloader_source_sha256=_sha(Path(downloader.__file__)),
            expected_download_completion_sha256=_sha(
                download / downloader.COMPLETION_NAME
            ),
            expected_package_manifest_sha256=receipt["package_manifest_sha256"],
        )
    extra.unlink()
    prune = package / preparer.PRUNE_DIRECTORY / "chr1.prune.in"
    prune.write_text(prune.read_text(encoding="utf-8") + "rs199\n", encoding="utf-8")
    with pytest.raises(ValueError, match="retained prune.in record changed"):
        preparer.validate_all_autosome_package(
            package_dir=package,
            source_manifest=source_manifest,
            download_dir=download,
            legacy_prep_manifest=legacy,
            plink2_path=plink2,
            expected_source_manifest_sha256=_sha(source_manifest),
            expected_downloader_source_sha256=_sha(Path(downloader.__file__)),
            expected_download_completion_sha256=_sha(
                download / downloader.COMPLETION_NAME
            ),
            expected_package_manifest_sha256=receipt["package_manifest_sha256"],
        )


def test_validator_requires_external_manifest_hash_and_rejects_npz_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    package = tmp_path / "package"
    receipt = _run(
        source_manifest=source_manifest,
        download=download,
        legacy=legacy,
        plink2=plink2,
        output=package,
    )
    with pytest.raises(ValueError, match="independently supplied"):
        preparer.validate_all_autosome_package(
            package_dir=package,
            source_manifest=source_manifest,
            download_dir=download,
            legacy_prep_manifest=legacy,
            plink2_path=plink2,
            expected_source_manifest_sha256=_sha(source_manifest),
            expected_downloader_source_sha256=_sha(Path(downloader.__file__)),
            expected_download_completion_sha256=_sha(
                download / downloader.COMPLETION_NAME
            ),
            expected_package_manifest_sha256="0" * 64,
        )
    with (package / preparer.NPZ_NAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="NPZ output record is inconsistent"):
        preparer.validate_all_autosome_package(
            package_dir=package,
            source_manifest=source_manifest,
            download_dir=download,
            legacy_prep_manifest=legacy,
            plink2_path=plink2,
            expected_source_manifest_sha256=_sha(source_manifest),
            expected_downloader_source_sha256=_sha(Path(downloader.__file__)),
            expected_download_completion_sha256=_sha(
                download / downloader.COMPLETION_NAME
            ),
            expected_package_manifest_sha256=receipt["package_manifest_sha256"],
        )


def test_failure_rolls_back_unpublished_package_and_cleans_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_PLINK_FAIL_CHR", "2")
    package = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="chr2_pgen_qc"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=package,
        )
    assert not package.exists()
    assert not list(tmp_path.glob("rspca_all_autosomes_*"))


def test_missing_sample_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    monkeypatch.setenv("FAKE_PLINK_DROP_SAMPLE", "1")
    package = tmp_path / "failed"
    with pytest.raises(ValueError, match="lacks frozen cohort samples"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=package,
        )
    assert not package.exists()


def test_source_and_legacy_manifests_are_exact_hash_pins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_payload["source"]["provider"] = "coherently resigned attacker mirror"
    source_manifest.write_text(
        json.dumps(source_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="safety pin"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=tmp_path / "source_attack",
        )
    # Restore the source bytes so the second assertion isolates the old-cohort pin.
    (tmp_path / "second").mkdir()
    source_manifest, download, legacy, plink2 = _fixture(
        tmp_path / "second", monkeypatch
    )
    legacy_payload = json.loads(legacy.read_text(encoding="utf-8"))
    legacy_payload["sample_selection"]["selected_records"][0][
        "population_label"
    ] = "EUR"
    legacy.write_text(
        json.dumps(legacy_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cohort pin"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=tmp_path / "legacy_attack",
        )


def test_refuses_overwrite_and_preflight_failure_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    inside_download = download / "bad_package"
    with pytest.raises(ValueError, match="output package must be outside"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=inside_download,
        )
    assert not inside_download.exists()
    with pytest.raises(ValueError, match="scratch parent must be outside"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=tmp_path / "bad_scratch",
            scratch_parent=download,
        )
    assert not (tmp_path / "bad_scratch").exists()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=occupied,
        )
    broken_target = tmp_path / "missing_target"
    symlink_output = tmp_path / "symlink_output"
    symlink_output.symlink_to(broken_target)
    with pytest.raises(FileExistsError, match="overwrite"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=symlink_output,
        )
    assert symlink_output.is_symlink()
    assert not broken_target.exists()

    original = preparer._preflight_disk_space

    def fail_chromosome(path: Path, *, required_bytes: int, role: str) -> None:
        if role.startswith("chromosome"):
            raise OSError("injected insufficient scratch space")
        original(path, required_bytes=required_bytes, role=role)

    monkeypatch.setattr(preparer, "_preflight_disk_space", fail_chromosome)
    unpublished = tmp_path / "unpublished"
    with pytest.raises(OSError, match="insufficient scratch"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=unpublished,
        )
    assert not unpublished.exists()


def test_install_race_never_deletes_a_directory_it_did_not_create(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging_npz = tmp_path / "staging.npz"
    staging_manifest = tmp_path / "staging.json"
    staging_npz.write_bytes(b"npz")
    staging_manifest.write_bytes(b"manifest")
    destination = tmp_path / "raced_destination"
    original_mkdir = preparer.os.mkdir

    def inject_competing_creator(path: str | Path, mode: int = 0o777) -> None:
        if Path(path) == destination:
            original_mkdir(path, mode)
            raise FileExistsError("injected exclusive-create race")
        original_mkdir(path, mode)

    monkeypatch.setattr(preparer.os, "mkdir", inject_competing_creator)
    with pytest.raises(FileExistsError, match="exclusive-create race"):
        preparer._install_package(
            staging_npz=staging_npz,
            staging_manifest=staging_manifest,
            outputs=(),
            destination=destination,
        )
    assert destination.is_dir()
    assert not any(destination.iterdir())


def test_postcompute_validation_failure_is_not_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)

    def reject_candidate(**kwargs):
        raise ValueError("injected strict candidate rejection")

    monkeypatch.setattr(preparer, "validate_all_autosome_package", reject_candidate)
    package = tmp_path / "must_not_publish"
    with pytest.raises(ValueError, match="candidate rejection"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=package,
        )
    assert not package.exists()


def test_fresh_downloader_validation_executes_on_disk_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}
    report = {
        "status": "valid_and_externally_anchored",
        "source_manifest_sha256": "1" * 64,
        "downloader_source_sha256": "2" * 64,
        "completion_file_sha256": "3" * 64,
    }

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(report).encode("utf-8"),
            stderr=b"",
        )

    def forbidden_cached_validator(**kwargs):
        raise AssertionError("cached downloader API must not be used")

    monkeypatch.setattr(preparer.subprocess, "run", fake_run)
    monkeypatch.setattr(
        downloader, "validate_download_directory", forbidden_cached_validator
    )
    downloader_source = tmp_path / "downloader.py"
    source_manifest = tmp_path / "source.json"
    download_dir = tmp_path / "download"
    result = preparer._fresh_downloader_validation(
        downloader_source=downloader_source,
        source_manifest=source_manifest,
        download_dir=download_dir,
        expected_source_manifest_sha256="1" * 64,
        expected_downloader_source_sha256="2" * 64,
        expected_download_completion_sha256="3" * 64,
    )
    assert result == report
    argv = observed["argv"]
    assert argv[1:3] == [str(downloader_source), "validate"]
    assert argv[argv.index("--source-manifest") + 1] == str(source_manifest)
    assert argv[argv.index("--output-dir") + 1] == str(download_dir)
    assert argv[argv.index("--expected-source-manifest-sha256") + 1] == "1" * 64
    assert argv[argv.index("--expected-downloader-source-sha256") + 1] == "2" * 64
    assert argv[argv.index("--expected-completion-sha256") + 1] == "3" * 64


def test_import_bound_helper_source_mismatch_fails_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest, download, legacy, plink2 = _fixture(tmp_path, monkeypatch)
    forged = dict(preparer._IMPORT_LEGACY_PREPARER_SOURCE)
    forged["sha256"] = "0" * 64
    monkeypatch.setattr(preparer, "_IMPORT_LEGACY_PREPARER_SOURCE", forged)
    output = tmp_path / "unpublished"
    with pytest.raises(RuntimeError, match="legacy preparation helper changed"):
        _run(
            source_manifest=source_manifest,
            download=download,
            legacy=legacy,
            plink2=plink2,
            output=output,
        )
    assert not output.exists()

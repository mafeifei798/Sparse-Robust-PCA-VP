from __future__ import annotations

import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
import pytest

from rspca.genomics_aim import MultiAutosomeGenotypeData, validate_multiautosome_input
from tools.prepare_genomics_aim_1kg import (
    CHROMOSOMES,
    OFFICIAL_PANEL_BASENAME,
    OFFICIAL_PANEL_URL,
    OFFICIAL_VCF_BASENAMES,
    OFFICIAL_VCF_URLS,
    PanelRecord,
    prepare_1kg_aim_dataset,
    select_panel_samples,
)


FAKE_PLINK = r'''#!/usr/bin/env python3
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
    print("PLINK v2.00a6.9LM fake deterministic test build")
    raise SystemExit(0)

def value(flag):
    return argv[argv.index(flag) + 1]

out = Path(value("--out"))
out.parent.mkdir(parents=True, exist_ok=True)

if "--vcf" in argv:
    vcf = Path(value("--vcf"))
    match = re.search(r"chr(20|21|22)", vcf.name)
    if not match:
        raise SystemExit("cannot infer chromosome")
    chromosome = match.group(1)
    keep = Path(value("--keep"))
    samples = [line.split()[1] for line in keep.read_text().splitlines() if line.strip()]
    (Path(str(out) + ".pgen")).write_bytes(b"fake-pgen")
    (Path(str(out) + ".psam")).write_text(
        "#FID\tIID\n" + "".join(f"{sample}\t{sample}\n" for sample in samples),
        encoding="utf-8",
    )
    (Path(str(out) + ".pvar")).write_text(
        "##fileformat=PVARv1\n"
        "#CHROM\tPOS\tID\tREF\tALT\n"
        f"{chromosome}\t100\t{chromosome}:100:A:G\tA\tG\n"
        f"{chromosome}\t200\t{chromosome}:200:C:T\tC\tT\n"
        f"{chromosome}\t300\t{chromosome}:300:G:A\tG\tA\n",
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
    extract = Path(value("--extract"))
    variants = [line.strip() for line in extract.read_text().splitlines() if line.strip()]
    psam_rows = [
        line.split()
        for line in Path(str(pfile) + ".psam").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    samples = [row[1] for row in psam_rows]
    if os.environ.get("FAKE_PLINK_DROP_SAMPLE") == "1":
        samples = samples[:-1]
    chromosome = variants[0].split(":", 1)[0]
    with Path(str(out) + ".raw").open("w", encoding="utf-8") as handle:
        handle.write(
            "#FID IID PAT MAT SEX PHENOTYPE "
            + " ".join(variant + "_ALT" for variant in variants)
            + "\n"
        )
        for position, sample in enumerate(reversed(samples)):
            first = (position + int(chromosome)) % 3
            second = (position * 2 + int(chromosome)) % 3
            handle.write(
                f"{sample} {sample} 0 0 0 -9 {first} {second}\n"
            )
else:
    raise SystemExit(f"unrecognized fake PLINK command: {argv}")
'''


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    vcfs: dict[str, Path] = {}
    for chromosome in CHROMOSOMES:
        path = tmp_path / OFFICIAL_VCF_BASENAMES[chromosome]
        path.write_bytes(f"fake bgzip VCF chromosome {chromosome}\n".encode())
        vcfs[chromosome] = path

    panel = tmp_path / OFFICIAL_PANEL_BASENAME
    lines = ["sample\tpop\tsuper_pop\tgender"]
    subpopulations = {
        "AFR": "YRI",
        "AMR": "MXL",
        "EAS": "CHB",
        "EUR": "CEU",
        "SAS": "GIH",
    }
    for superpopulation, subpopulation in subpopulations.items():
        for index in range(4):
            lines.append(
                f"{superpopulation}_{index:02d}\t{subpopulation}\t"
                f"{superpopulation}\tunknown"
            )
    panel.write_text("\n".join(lines) + "\n", encoding="utf-8")

    plink2 = tmp_path / "fake_plink2"
    plink2.write_text(textwrap.dedent(FAKE_PLINK), encoding="utf-8")
    plink2.chmod(0o755)
    return vcfs, panel, plink2


def test_sample_selection_is_order_independent_and_seeded() -> None:
    records = tuple(
        PanelRecord(
            sample_id=f"{group}_{index}",
            subpopulation=f"{group}_SUB",
            superpopulation=group,
        )
        for group in ("AFR", "AMR", "EAS", "EUR", "SAS")
        for index in range(8)
    )
    selected = select_panel_samples(
        records,
        population_design="three_continent",
        cap_per_group=3,
        selection_seed=41,
    )
    reversed_selected = select_panel_samples(
        tuple(reversed(records)),
        population_design="three_continent",
        cap_per_group=3,
        selection_seed=41,
    )
    assert selected == reversed_selected
    assert len(selected) == 9
    assert {row.superpopulation for row in selected} == {"AFR", "EAS", "EUR"}
    assert [row.superpopulation for row in selected] == sorted(
        row.superpopulation for row in selected
    )
    assert select_panel_samples(
        records,
        population_design="three_continent",
        cap_per_group=3,
        selection_seed=42,
    ) != selected


def test_preparation_writes_canonical_npz_and_complete_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vcfs, panel, plink2 = _write_fixture(tmp_path)
    command_log = tmp_path / "commands.jsonl"
    monkeypatch.setenv("FAKE_PLINK_LOG", str(command_log))
    output = tmp_path / "prepared.npz"
    manifest_path = tmp_path / "prepared.manifest.json"

    receipt = prepare_1kg_aim_dataset(
        vcfs=vcfs,
        panel_path=panel,
        plink2_path=plink2,
        output_npz=output,
        manifest_path=manifest_path,
        population_design="three_continent",
        cap_per_group=2,
        selection_seed=991,
        maf=0.02,
        geno=0.04,
        ld_window=100,
        ld_window_unit="kb",
        ld_step=1,
        ld_r2=0.15,
    )

    assert receipt["source_manifest_sha256"] == _sha256(manifest_path)
    assert receipt["input_scope"] == "legacy_chr20_22"
    assert receipt["samples"] == 6
    assert receipt["ld_pruned_snps"] == 6
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "genotypes",
            "sample_ids",
            "snp_ids",
            "chromosomes",
            "population_labels",
        }
        assert archive["genotypes"].shape == (6, 6)
        assert archive["genotypes"].dtype == np.dtype("float32")
        assert archive["sample_ids"].tolist() == sorted(
            archive["sample_ids"].tolist(),
            key=lambda sample: (sample.split("_", 1)[0], sample),
        )
        assert set(archive["population_labels"].tolist()) == {"AFR", "EAS", "EUR"}
        assert archive["chromosomes"].tolist() == ["20", "20", "21", "21", "22", "22"]
        assert len(set(archive["snp_ids"].tolist())) == 6
        for sample, label in zip(
            archive["sample_ids"], archive["population_labels"], strict=True
        ):
            assert sample.startswith(label + "_")
        protocol_data = MultiAutosomeGenotypeData(
            genotypes=archive["genotypes"].astype(float),
            sample_ids=archive["sample_ids"].astype(str),
            snp_ids=archive["snp_ids"].astype(str),
            chromosomes=archive["chromosomes"].astype(str),
            population_labels=archive["population_labels"].astype(str),
            source_manifest_sha256=receipt["source_manifest_sha256"],
            input_scope="legacy_chr20_22",
            ld_pruned=True,
        )
        validate_multiautosome_input(protocol_data)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset"]["input_scope"] == "legacy_chr20_22"
    assert manifest["dataset"]["population_label_is_race"] is False
    assert manifest["sources"]["integrated_panel"]["source_url"] == OFFICIAL_PANEL_URL
    assert manifest["sources"]["integrated_panel"]["sha256"] == _sha256(panel)
    for chromosome in CHROMOSOMES:
        assert manifest["sources"]["vcfs"][chromosome]["source_url"] == OFFICIAL_VCF_URLS[chromosome]
        assert manifest["sources"]["vcfs"][chromosome]["sha256"] == _sha256(vcfs[chromosome])
        per_chromosome = manifest["variant_processing"]["per_chromosome"][chromosome]
        assert per_chromosome["qc_variant_count"] == 3
        assert per_chromosome["ld_pruned_variant_count"] == 2
        assert len(per_chromosome["commands"]) == 3
    assert manifest["variant_processing"]["variant_target"] is None
    assert manifest["variant_processing"]["padding_with_unpruned_variants"] is False
    assert (
        manifest["variant_processing"]["population_labels_used_for_variant_qc_or_ld"]
        is False
    )
    assert manifest["downstream_label_boundary"]["panel_selection_used_population_labels"] is False
    assert manifest["output"]["sha256"] == _sha256(output)
    assert manifest["output"]["npz_keys"] == sorted(
        ["genotypes", "sample_ids", "snp_ids", "chromosomes", "population_labels"]
    )

    commands = [json.loads(line) for line in command_log.read_text().splitlines()]
    assert len(commands) == 10  # version + three commands for each chromosome
    flattened = [token for command in commands for token in command]
    assert "shuf" not in flattened
    assert flattened.count("--threads") == 9
    assert flattened.count("--indep-pairwise") == 3
    assert flattened.count("--indep-order") == 3
    assert flattened.count("--extract") == 3
    assert flattened.count("--maf") == 3
    assert flattened.count("--geno") == 3
    assert flattened.count("--rm-dup") == 3


def test_five_superpopulation_extension_uses_every_group_without_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vcfs, panel, plink2 = _write_fixture(tmp_path)
    monkeypatch.delenv("FAKE_PLINK_DROP_SAMPLE", raising=False)
    output = tmp_path / "five.npz"
    manifest = tmp_path / "five.json"
    receipt = prepare_1kg_aim_dataset(
        vcfs=vcfs,
        panel_path=panel,
        plink2_path=plink2,
        output_npz=output,
        manifest_path=manifest,
        population_design="five_superpopulation",
    )
    assert receipt["samples"] == 20
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive["population_labels"].tolist()) == {
            "AFR",
            "AMR",
            "EAS",
            "EUR",
            "SAS",
        }


def test_repeated_preparation_produces_identical_npz_bytes(tmp_path: Path) -> None:
    vcfs, panel, plink2 = _write_fixture(tmp_path)
    outputs: list[Path] = []
    for index in range(2):
        output = tmp_path / f"repeat_{index}.npz"
        prepare_1kg_aim_dataset(
            vcfs=vcfs,
            panel_path=panel,
            plink2_path=plink2,
            output_npz=output,
            manifest_path=tmp_path / f"repeat_{index}.json",
            population_design="three_continent",
            cap_per_group=2,
            selection_seed=734,
        )
        outputs.append(output)
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_existing_destination_and_plink_sample_loss_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vcfs, panel, plink2 = _write_fixture(tmp_path)
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"owned by user")
    manifest = tmp_path / "unused.json"
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare_1kg_aim_dataset(
            vcfs=vcfs,
            panel_path=panel,
            plink2_path=plink2,
            output_npz=existing,
            manifest_path=manifest,
            population_design="three_continent",
        )
    assert existing.read_bytes() == b"owned by user"
    assert not manifest.exists()

    monkeypatch.setenv("FAKE_PLINK_DROP_SAMPLE", "1")
    failed_output = tmp_path / "failed.npz"
    failed_manifest = tmp_path / "failed.json"
    with pytest.raises(ValueError, match="sample set differs"):
        prepare_1kg_aim_dataset(
            vcfs=vcfs,
            panel_path=panel,
            plink2_path=plink2,
            output_npz=failed_output,
            manifest_path=failed_manifest,
            population_design="three_continent",
            cap_per_group=2,
        )
    assert not failed_output.exists()
    assert not failed_manifest.exists()


def test_official_source_names_are_required(tmp_path: Path) -> None:
    vcfs, panel, plink2 = _write_fixture(tmp_path)
    renamed = tmp_path / "chr20.vcf.gz"
    renamed.write_bytes(vcfs["20"].read_bytes())
    vcfs["20"] = renamed
    with pytest.raises(ValueError, match="VCF basename"):
        prepare_1kg_aim_dataset(
            vcfs=vcfs,
            panel_path=panel,
            plink2_path=plink2,
            output_npz=tmp_path / "out.npz",
            manifest_path=tmp_path / "out.json",
            population_design="three_continent",
        )

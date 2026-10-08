from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.genomics_1kg import load_genotypes
from tools.plink_raw_to_npz import convert_plink_raw, main


def _write_raw(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_convert_export_a_raw_preserves_ids_and_genotype_missingness(
    tmp_path: Path,
) -> None:
    raw = _write_raw(
        tmp_path / "tiny.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A rs2_G rs3_T\n"
        "F1 HG001 0 0 1 -9 0 1 2\n"
        "F2 HG002 0 0 2 2 2 NA 0.5\n"
        "F3 HG003 0 0 0 -9 -9 0 1\n",
    )
    output = tmp_path / "tiny.npz"
    provenance_path = tmp_path / "tiny.provenance.json"

    record = convert_plink_raw(raw, output, provenance_path=provenance_path)

    with np.load(output, allow_pickle=False) as archive:
        assert {"X", "row_ids", "variant_ids", "column_ids"} == set(archive.files)
        assert archive["row_ids"].tolist() == ["HG001", "HG002", "HG003"]
        assert archive["variant_ids"].tolist() == ["rs1_A", "rs2_G", "rs3_T"]
        assert np.array_equal(archive["column_ids"], archive["variant_ids"])
        assert archive["X"] == pytest.approx(
            np.asarray(
                [[0.0, 1.0, 2.0], [2.0, np.nan, 0.5], [np.nan, 0.0, 1.0]]
            ),
            nan_ok=True,
        )

    # Both PHENOTYPE=-9 rows retain their three genotype cells; only genotype
    # -9/NA tokens become missing.
    assert record["matrix"] == {
        "orientation": "samples_by_variants",
        "shape": [3, 3],
        "dtype": "<f8",
        "missing_count": 2,
    }
    assert record["missing_genotype_policy"]["scope"] == "genotype_columns_only"
    assert record["input"]["sha256"] == _sha256(raw)
    assert record["output"]["sha256"] == _sha256(output)
    assert json.loads(provenance_path.read_text(encoding="utf-8")) == record

    loaded = load_genotypes(output, missing_value=-9)
    assert loaded.genotypes == pytest.approx(
        np.asarray([[0.0, 1.0, 2.0], [2.0, np.nan, 0.5], [np.nan, 0.0, 1.0]]),
        nan_ok=True,
    )
    assert loaded.sample_ids.tolist() == ["HG001", "HG002", "HG003"]
    assert loaded.snp_ids.tolist() == ["rs1_A", "rs2_G", "rs3_T"]


@pytest.mark.parametrize(
    "header",
    [
        "#FID IID PAT MAT SEX PHENOTYPE rs1_A",
        "FID IID PAT MAT SEX PHENO1 rs1_A",
        "FID IID MAT PAT SEX PHENOTYPE rs1_A",
        "FID IID PAT MAT SEX PHENOTYPE",
    ],
)
def test_converter_rejects_noncanonical_export_a_header(
    tmp_path: Path, header: str
) -> None:
    raw = _write_raw(tmp_path / "bad.raw", f"{header}\nF1 I1 0 0 1 -9 1\n")

    with pytest.raises(ValueError, match="header|variant columns"):
        convert_plink_raw(raw, tmp_path / "bad.npz")


def test_converter_rejects_duplicate_iid_without_writing_output(tmp_path: Path) -> None:
    raw = _write_raw(
        tmp_path / "duplicate.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A rs2_G\n"
        "F1 SAME 0 0 1 -9 0 1\n"
        "F2 SAME 0 0 2 -9 1 2\n",
    )
    output = tmp_path / "duplicate.npz"

    with pytest.raises(ValueError, match="duplicate IID 'SAME'"):
        convert_plink_raw(raw, output)

    assert not output.exists()


@pytest.mark.parametrize("token", ["nan", "inf", "-inf", "Infinity", "."])
def test_converter_rejects_illegal_or_nonfinite_genotypes(
    tmp_path: Path, token: str
) -> None:
    raw = _write_raw(
        tmp_path / "invalid.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A\n"
        f"F1 I1 0 0 1 -9 {token}\n",
    )

    with pytest.raises(ValueError, match="genotype"):
        convert_plink_raw(raw, tmp_path / "invalid.npz")


def test_converter_rejects_ragged_rows_and_duplicate_variants(tmp_path: Path) -> None:
    ragged = _write_raw(
        tmp_path / "ragged.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A rs2_G\n"
        "F1 I1 0 0 1 -9 1\n",
    )
    with pytest.raises(ValueError, match="fields; expected"):
        convert_plink_raw(ragged, tmp_path / "ragged.npz")

    duplicate_variants = _write_raw(
        tmp_path / "variants.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A rs1_A\n"
        "F1 I1 0 0 1 -9 1 2\n",
    )
    with pytest.raises(ValueError, match="variant identifiers must be unique"):
        convert_plink_raw(duplicate_variants, tmp_path / "variants.npz")


def test_cli_writes_npz_and_optional_provenance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw = _write_raw(
        tmp_path / "cli.raw",
        "FID IID PAT MAT SEX PHENOTYPE rs1_A rs2_G\n"
        "F1 I1 0 0 1 -9 0 1\n"
        "F2 I2 0 0 2 -9 2 1\n",
    )
    output = tmp_path / "cli.npz"
    provenance = tmp_path / "cli.json"

    assert main([str(raw), str(output), "--provenance", str(provenance)]) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["matrix"]["shape"] == [2, 2]
    assert output.is_file()
    assert provenance.is_file()

    with pytest.raises(SystemExit) as error:
        main([str(raw), str(output)])
    assert error.value.code == 2

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_genomics_amr_loadings",
    ROOT / "tools" / "render_genomics_amr_loadings.py",
)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows_per_chromosome = 24
    width = 22 * rows_per_chromosome
    rng = np.random.default_rng(9301)
    loadings, _ = np.linalg.qr(rng.normal(size=(width, module.RANK)))
    chromosomes = np.repeat(
        np.asarray(module.allauto.EXPECTED_AUTOSOMES, dtype="<U2"),
        rows_per_chromosome,
    )
    retained = np.arange(width, dtype=np.int64)
    snp_ids = np.asarray([f"rs{index}" for index in range(width)], dtype=str)
    importance = np.sum(np.abs(loadings), axis=1)
    return retained, snp_ids, chromosomes, loadings, importance


def test_loading_summaries_are_deterministic_and_label_free() -> None:
    retained, snp_ids, chromosomes, loadings, importance = _fixture()
    blocks = module._chromosome_blocks(chromosomes)
    assert len(blocks) == 22
    assert blocks[0]["start_loading_row"] == 0
    assert blocks[-1]["stop_loading_row_exclusive"] == retained.size

    energy = module._chromosome_energy(loadings, chromosomes)
    matrix = np.asarray(
        [row["squared_loading_energy_fraction"] for row in energy], dtype=float
    ).reshape(22, module.RANK)
    np.testing.assert_allclose(matrix.sum(axis=0), np.ones(module.RANK), atol=1e-12)

    skyline = module._skyline_bin_maxima(
        loadings,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
    )
    assert len(skyline) == 22 * module.SKYLINE_BINS_PER_CHROMOSOME * module.RANK
    for row in skyline:
        chromosome_rows = np.flatnonzero(chromosomes == row["chromosome"])
        bin_rows = np.array_split(chromosome_rows, module.SKYLINE_BINS_PER_CHROMOSOME)[
            row["chromosome_bin"]
        ]
        component = row["pc"] - 1
        expected_row = int(
            bin_rows[int(np.argmax(np.abs(loadings[bin_rows, component])))]
        )
        assert row["loading_row"] == expected_row
        assert row["snp_id"] == snp_ids[expected_row]

    per_pc = module._top_per_pc(
        loadings,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
    )
    assert len(per_pc) == module.RANK * min(module.TOP_PER_PC_COUNT, retained.size)
    assert {row["pc"] for row in per_pc} == set(range(1, module.RANK + 1))
    assert "population" not in repr(skyline).lower()


def test_top_rows_and_csv_preserve_signed_loadings(tmp_path: Path) -> None:
    retained, snp_ids, chromosomes, loadings, importance = _fixture()
    index = np.arange(importance.size, dtype=np.int64)
    ranking = np.lexsort((index, -importance)).astype(np.int64)
    top = module._top_rows(
        ranking,
        module.TOP_CSV_COUNT,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
        loadings=loadings,
        importance=importance,
    )
    assert len(top) == module.TOP_CSV_COUNT
    assert top[0]["loading_row"] == int(ranking[0])
    assert top[0]["signed_loadings"] == loadings[ranking[0]].tolist()

    output = tmp_path / "top.csv"
    module._write_top_csv(output, top)
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == module.TOP_CSV_COUNT
    assert float(rows[0]["PC8_signed_loading"]) == float(loadings[ranking[0], 7])


def test_chromosome_blocks_reject_noncontiguous_input() -> None:
    _, _, chromosomes, _, _ = _fixture()
    corrupted = chromosomes.copy()
    corrupted[0], corrupted[-1] = corrupted[-1], corrupted[0]
    with pytest.raises(ValueError, match="contiguous chromosome order"):
        module._chromosome_blocks(corrupted)


def test_validator_rejects_non_package(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular directory"):
        module.validate(tmp_path / "missing")
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        module.validate(extra)

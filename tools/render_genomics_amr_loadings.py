#!/usr/bin/env python3
"""Render the exact formal PCA loadings used by the AMR readout study.

This is a presentation-only reconstruction.  It fixes the first fresh AMR
seed (731001), rank eight, and the all-train-retained-SNP cell.  It never uses
population labels to fit or rank SNPs, and it verifies the rebuilt arrays
against the hashes sealed in the formal AMR result before publishing output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Mapping, Sequence
import uuid

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
for candidate in (REPO_ROOT / "experiments", REPO_ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import genomics_pca_loading_all_autosomes as allauto  # noqa: E402
import genomics_pca_loading_amr_whitening as amr  # noqa: E402
import genomics_pca_loading_grid as grid  # noqa: E402
from rspca.datasets import array_sha256, canonical_sha256  # noqa: E402
from rspca.genomics_aim import stratified_split  # noqa: E402


SEED = 731001
RANK = 8
TEST_FRACTION = 0.2
TOP_CSV_COUNT = 200
TOP_HEATMAP_COUNT = 40
TOP_PER_PC_COUNT = 100
SKYLINE_BINS_PER_CHROMOSOME = 12

EXPECTED_AMR_ANALYSIS_SHA256 = (
    "b2aced797ab06b6402ae7d7c1bb377b924696096b1e8c22b0d8280f5b6a3ecee"
)
EXPECTED_AMR_PACKAGE_SHA256 = (
    "9d33ffe4b5260a2c13c3813a654b38250e991a37d6e450856408c18f6a949248"
)
EXPECTED_LOADINGS_SHA256 = (
    "46bd465c81eb2f59e6dfeecd3cfea80e4f91195f67b5c4e1eeeee95ebec9d05f"
)
EXPECTED_IMPORTANCE_SHA256 = (
    "81a785c31830ddfef8d6b421e1b0ff95622903e3bfaf405a9ea0a1b3a4544211"
)
EXPECTED_SAMPLE_GRAM_SHA256 = (
    "283dcadb6ce3e7ccc994e9ddd887f3bdd28ff50d0a0b8e36733c85079f51b565"
)
EXPECTED_RETAINED_ORIGINAL_SHA256 = (
    "24b035fe23bddeb8d68b1c663605959951fa212229e9adbaa1edb165468386cd"
)
EXPECTED_TRAIN_INDICES_SHA256 = (
    "40ead245905c0483d7e1061b1cef4e0a72c128fb6284eb7bb80a809707850060"
)
EXPECTED_TEST_INDICES_SHA256 = (
    "6c961747ebace697602d7a9ae6c9c6cfd24283c0e441a7fea457ffdeb3a7e626"
)
EXPECTED_SNP_IDS_SHA256 = (
    "e4384eceb9bea8854ee60ed2e6447a5bff841776bc98e6f838060872beb45e1f"
)
EXPECTED_CHROMOSOMES_SHA256 = (
    "77df1f225db02fd813de3c9daeb046bd6191f9013cd478f5c37166205a79b8ea"
)
EXPECTED_SINGULAR_VALUES_SHA256 = (
    "94f80f043f34384d8af1229a5436e4beb15a6f6f4f553c1bd6c264603eb25150"
)
EXPECTED_EXPLAINED_VARIANCE_RATIO_SHA256 = (
    "0af6f0f3397bcf75233a560e9d3e78e8472afa5d9b7c34a08ad630a2db853586"
)

NPZ_FILENAME = "pca_loadings_seed731001_rank8_all.npz"
CSV_FILENAME = "top200_snp_loadings_seed731001_rank8_all.csv"
PNG_FILENAME = "pca_loading_landscape_seed731001_rank8_all.png"
SUMMARY_FILENAME = "pca_loading_summary_seed731001_rank8_all.json"
MANIFEST_FILENAME = "pca_loading_output_manifest.json"
ARTIFACT_FILENAMES = (NPZ_FILENAME, CSV_FILENAME, PNG_FILENAME, SUMMARY_FILENAME)
NPZ_KEYS = (
    "train_indices",
    "test_indices",
    "retained_original_indices",
    "snp_ids",
    "chromosomes",
    "loadings",
    "importance_l1_abs_sum",
    "singular_values",
    "explained_variance_ratio",
)
FIXED_CELL = {
    "seed": SEED,
    "n_components": RANK,
    "panel_label": "all",
    "reason": "first_fresh_seed_and_historical_rank8_all_headline_alignment",
    "selected_after_viewing_loading_values": False,
}
METHOD = {
    "pca": "exact_chunked_outer_train_smartpca_scaled_ordinary_pca",
    "loading_importance": "sum_absolute_PC1_through_PC8_loadings",
    "population_labels_used_for_pca_or_loading_ranking": False,
    "forbidden_methods_used": [],
    "raw_and_whitened_loadings_are_identical": True,
    "whitening_changes_only_post_PCA_distance_scaling": True,
}
INTERPRETATION_BOUNDARY = (
    "SNP loading magnitude is an unsupervised axis contribution, not a "
    "population-specific effect size or a causal biological association."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token: {value}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )


def _receipt(path: Path) -> dict[str, Any]:
    return {"bytes": int(path.stat().st_size), "sha256": _sha256(path)}


def _chromosome_blocks(chromosomes: np.ndarray) -> list[dict[str, Any]]:
    values = np.asarray(chromosomes, dtype=str)
    blocks: list[dict[str, Any]] = []
    cursor = 0
    for chromosome in allauto.EXPECTED_AUTOSOMES:
        positions = np.flatnonzero(values == chromosome)
        if not positions.size:
            raise ValueError(f"missing chromosome {chromosome}")
        start, stop = int(positions[0]), int(positions[-1] + 1)
        if start != cursor or not np.all(values[start:stop] == chromosome):
            raise ValueError("retained SNPs are not in contiguous chromosome order")
        blocks.append(
            {
                "chromosome": chromosome,
                "start_loading_row": start,
                "stop_loading_row_exclusive": stop,
                "snp_count": stop - start,
                "midpoint_loading_row": (start + stop - 1) / 2.0,
            }
        )
        cursor = stop
    if cursor != values.size:
        raise ValueError("unexpected chromosome label after chromosome 22")
    return blocks


def _top_rows(
    ranking: np.ndarray,
    count: int,
    *,
    retained_original: np.ndarray,
    snp_ids: np.ndarray,
    chromosomes: np.ndarray,
    loadings: np.ndarray,
    importance: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_position, loading_row in enumerate(ranking[:count], start=1):
        row = int(loading_row)
        rows.append(
            {
                "importance_rank": rank_position,
                "loading_row": row,
                "original_snp_column": int(retained_original[row]),
                "chromosome": str(chromosomes[row]),
                "snp_id": str(snp_ids[row]),
                "importance_l1_abs_sum": float(importance[row]),
                "signed_loadings": [float(value) for value in loadings[row]],
            }
        )
    return rows


def _top_per_pc(
    loadings: np.ndarray,
    *,
    retained_original: np.ndarray,
    snp_ids: np.ndarray,
    chromosomes: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    index = np.arange(loadings.shape[0], dtype=np.int64)
    for component in range(loadings.shape[1]):
        ranking = np.lexsort((index, -np.abs(loadings[:, component])))
        for position, loading_row in enumerate(ranking[:TOP_PER_PC_COUNT], start=1):
            row = int(loading_row)
            records.append(
                {
                    "pc": component + 1,
                    "absolute_loading_rank_within_pc": position,
                    "loading_row": row,
                    "original_snp_column": int(retained_original[row]),
                    "chromosome": str(chromosomes[row]),
                    "snp_id": str(snp_ids[row]),
                    "signed_loading": float(loadings[row, component]),
                    "absolute_loading": float(abs(loadings[row, component])),
                }
            )
    return records


def _skyline_bin_maxima(
    loadings: np.ndarray,
    *,
    retained_original: np.ndarray,
    snp_ids: np.ndarray,
    chromosomes: np.ndarray,
) -> list[dict[str, Any]]:
    """Return deterministic max-absolute-loading points for inline plotting."""

    records: list[dict[str, Any]] = []
    for chromosome in allauto.EXPECTED_AUTOSOMES:
        chromosome_rows = np.flatnonzero(chromosomes == chromosome)
        for bin_index, bin_rows in enumerate(
            np.array_split(chromosome_rows, SKYLINE_BINS_PER_CHROMOSOME)
        ):
            if not bin_rows.size:
                continue
            for component in range(loadings.shape[1]):
                local = int(np.argmax(np.abs(loadings[bin_rows, component])))
                row = int(bin_rows[local])
                records.append(
                    {
                        "pc": component + 1,
                        "chromosome": chromosome,
                        "chromosome_bin": bin_index,
                        "loading_row": row,
                        "original_snp_column": int(retained_original[row]),
                        "snp_id": str(snp_ids[row]),
                        "signed_loading": float(loadings[row, component]),
                        "absolute_loading": float(abs(loadings[row, component])),
                    }
                )
    return records


def _chromosome_energy(
    loadings: np.ndarray,
    chromosomes: np.ndarray,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for chromosome in allauto.EXPECTED_AUTOSOMES:
        selected = np.asarray(chromosomes == chromosome, dtype=bool)
        energy = np.sum(np.square(loadings[selected]), axis=0)
        for component, value in enumerate(energy, start=1):
            records.append(
                {
                    "chromosome": chromosome,
                    "pc": component,
                    "squared_loading_energy_fraction": float(value),
                }
            )
    return records


def _write_top_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "importance_rank",
        "loading_row",
        "original_snp_column",
        "chromosome",
        "snp_id",
        "importance_l1_abs_sum",
        *[f"PC{component}_signed_loading" for component in range(1, RANK + 1)],
    ]
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            output = {key: row[key] for key in fields[:6]}
            output.update(
                {
                    f"PC{component}_signed_loading": row["signed_loadings"][
                        component - 1
                    ]
                    for component in range(1, RANK + 1)
                }
            )
            writer.writerow(output)


def _render_figure(
    path: Path,
    *,
    importance: np.ndarray,
    loadings: np.ndarray,
    chromosomes: np.ndarray,
    snp_ids: np.ndarray,
    blocks: Sequence[Mapping[str, Any]],
    top_rows: Sequence[Mapping[str, Any]],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
        }
    )
    figure = plt.figure(figsize=(24, 16), constrained_layout=False)
    layout = figure.add_gridspec(
        2,
        2,
        height_ratios=(1.05, 1.55),
        width_ratios=(0.72, 1.28),
        hspace=0.28,
        wspace=0.28,
    )

    skyline = figure.add_subplot(layout[0, :])
    x_values = np.arange(importance.size)
    palette = ("#2563eb", "#93c5fd")
    for block_index, block in enumerate(blocks):
        start = int(block["start_loading_row"])
        stop = int(block["stop_loading_row_exclusive"])
        skyline.scatter(
            x_values[start:stop],
            importance[start:stop],
            s=2.0,
            alpha=0.48,
            linewidths=0,
            color=palette[block_index % 2],
            rasterized=True,
        )
    skyline.set_xlim(-0.005 * importance.size, 1.005 * importance.size)
    skyline.set_xticks([float(block["midpoint_loading_row"]) for block in blocks])
    skyline.set_xticklabels([str(block["chromosome"]) for block in blocks])
    skyline.set_xlabel("Chromosome (x-position is retained-SNP order, not base pairs)")
    skyline.set_ylabel(r"SNP importance  $\sum_{j=1}^{8}|v_{ij}|$")
    skyline.set_title(
        "All-autosome PCA loading landscape — fixed seed 731001, rank 8, all 455,102 SNPs",
        loc="left",
        fontweight="bold",
    )
    skyline.grid(axis="y", alpha=0.18)
    for row in top_rows[:12]:
        point = int(row["loading_row"])
        skyline.annotate(
            str(row["snp_id"]),
            (point, importance[point]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=55,
            fontsize=8,
            color="#111827",
        )

    energy_axis = figure.add_subplot(layout[1, 0])
    energy = np.empty((22, RANK), dtype=float)
    for chromosome_index, chromosome in enumerate(allauto.EXPECTED_AUTOSOMES):
        energy[chromosome_index] = np.sum(
            np.square(loadings[np.asarray(chromosomes == chromosome, dtype=bool)]),
            axis=0,
        )
    energy_image = energy_axis.imshow(100.0 * energy, cmap="Blues", aspect="auto")
    energy_axis.set_xticks(np.arange(RANK))
    energy_axis.set_xticklabels([f"PC{value}" for value in range(1, RANK + 1)])
    energy_axis.set_yticks(np.arange(22))
    energy_axis.set_yticklabels([f"chr{value}" for value in range(1, 23)])
    energy_axis.set_title("Chromosome share of squared loading energy (%)", loc="left")
    for row in range(22):
        for column in range(RANK):
            value = 100.0 * energy[row, column]
            energy_axis.text(
                column,
                row,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 7.5 else "#0f172a",
            )
    colorbar = figure.colorbar(energy_image, ax=energy_axis, fraction=0.046, pad=0.04)
    colorbar.set_label("% of unit-norm PC loading")

    heatmap_axis = figure.add_subplot(layout[1, 1])
    top_indices = np.asarray(
        [int(row["loading_row"]) for row in top_rows[:TOP_HEATMAP_COUNT]],
        dtype=np.int64,
    )
    values = loadings[top_indices]
    limit = float(np.max(np.abs(values)))
    heatmap = heatmap_axis.imshow(
        values,
        cmap="RdBu_r",
        aspect="auto",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit),
    )
    heatmap_axis.set_xticks(np.arange(RANK))
    heatmap_axis.set_xticklabels([f"PC{value}" for value in range(1, RANK + 1)])
    heatmap_axis.set_yticks(np.arange(TOP_HEATMAP_COUNT))
    heatmap_axis.set_yticklabels(
        [
            f"{row['importance_rank']:02d}  chr{row['chromosome']} · {row['snp_id']}"
            for row in top_rows[:TOP_HEATMAP_COUNT]
        ]
    )
    heatmap_axis.set_title(
        "Top 40 SNPs by loading-only importance — signed PC loadings",
        loc="left",
    )
    heatmap_axis.tick_params(axis="y", pad=4)
    colorbar = figure.colorbar(heatmap, ax=heatmap_axis, fraction=0.032, pad=0.02)
    colorbar.set_label("Signed loading")

    figure.text(
        0.02,
        0.018,
        "Raw and singular-value-whitened readouts share this exact PCA fit and loading matrix. "
        "Population labels, subpopulation labels, ANOVA, Fisher scores, Robust PCA, VP, and RND are not used here.",
        fontsize=12,
        color="#334155",
    )
    figure.subplots_adjust(left=0.075, right=0.965, top=0.955, bottom=0.065)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def _build_summary(
    *,
    input_identity: Mapping[str, Any],
    retained_original: np.ndarray,
    snp_ids: np.ndarray,
    chromosomes: np.ndarray,
    loadings: np.ndarray,
    importance: np.ndarray,
    singular_values: np.ndarray,
    explained_variance_ratio: np.ndarray,
    sample_gram_sha256: str,
    top_rows: Sequence[Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
    runtime_identity: Mapping[str, Any],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "rspca-amr-loading-visualization-v1",
        "status": "complete_loading_only_presentation",
        "fixed_cell": dict(FIXED_CELL),
        "method": dict(METHOD),
        "renderer_source_sha256": _sha256(Path(__file__).resolve()),
        "source_amr_formal_result": {
            "analysis_sha256": EXPECTED_AMR_ANALYSIS_SHA256,
            "package_anchor_sha256": EXPECTED_AMR_PACKAGE_SHA256,
            "validated_before_loading_reconstruction": True,
        },
        "input_identity": dict(input_identity),
        "split": {
            "test_fraction": TEST_FRACTION,
            "train_samples": 800,
            "test_samples": 200,
            "train_indices_sha256": array_sha256(train_indices),
            "test_indices_sha256": array_sha256(test_indices),
        },
        "array_identity": {
            "retained_snp_count": int(retained_original.size),
            "retained_original_indices_sha256": array_sha256(retained_original),
            "snp_ids_sha256": array_sha256(snp_ids),
            "chromosomes_sha256": array_sha256(chromosomes),
            "loadings_sha256": array_sha256(loadings),
            "importance_sha256": array_sha256(importance),
            "sample_gram_sha256": sample_gram_sha256,
            "loadings_shape": list(loadings.shape),
        },
        "singular_values": [float(value) for value in singular_values],
        "explained_variance_ratio": [
            float(value) for value in explained_variance_ratio
        ],
        "chromosome_blocks": list(blocks),
        "chromosome_pc_loading_energy": _chromosome_energy(loadings, chromosomes),
        "top40_by_l1_abs_sum": list(top_rows[:TOP_HEATMAP_COUNT]),
        "top100_per_pc_by_absolute_loading": _top_per_pc(
            loadings,
            retained_original=retained_original,
            snp_ids=snp_ids,
            chromosomes=chromosomes,
        ),
        "skyline_bin_maxima": _skyline_bin_maxima(
            loadings,
            retained_original=retained_original,
            snp_ids=snp_ids,
            chromosomes=chromosomes,
        ),
        "skyline_bin_rule": {
            "bins_per_chromosome": SKYLINE_BINS_PER_CHROMOSOME,
            "point": "largest_absolute_loading_within_each_chromosome_bin_and_PC",
            "used_for_fit_or_selection": False,
        },
        "runtime_identity": dict(runtime_identity),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    return payload


def run(
    *,
    genotypes: Path,
    source_manifest: Path,
    amr_package: Path,
    output_dir: Path,
    scratch_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    runtime_identity = allauto._require_formal_blas_thread_environment()
    amr.validate_output_package(
        amr_package,
        repo_root=repo_root,
        expected_analysis_sha256=EXPECTED_AMR_ANALYSIS_SHA256,
        expected_package_anchor_sha256=EXPECTED_AMR_PACKAGE_SHA256,
    )
    target = Path(output_dir).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"output_dir already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(mode=0o700)
    try:
        with allauto.load_formal_input(
            genotypes,
            source_manifest,
            expected_source_manifest_sha256=allauto.FROZEN_FORMAL_MANIFEST_SHA256,
            scratch_root=scratch_root,
        ) as (data, input_identity):
            labels = np.asarray(data.population_labels, dtype=str)
            train, test = stratified_split(labels, TEST_FRACTION, SEED)
            if array_sha256(train) != EXPECTED_TRAIN_INDICES_SHA256:
                raise ValueError("fixed train split does not match formal hash")
            if array_sha256(test) != EXPECTED_TEST_INDICES_SHA256:
                raise ValueError("fixed test split does not match formal hash")
            fit = allauto.fit_chunked_top_pca(
                data.genotypes,
                train,
                test,
                n_components=RANK,
                chunk_size=2048,
            )
            ranking, importance = grid.rank_by_pca_loadings(fit.loadings, RANK)
            retained_original = np.asarray(fit.retained_original, dtype=np.int64)
            snp_ids = np.asarray(data.snp_ids[retained_original], dtype=str)
            chromosomes = np.asarray(data.chromosomes[retained_original], dtype=str)
            loadings = np.ascontiguousarray(fit.loadings[:, :RANK])
            importance = np.ascontiguousarray(importance)
            if array_sha256(snp_ids) != EXPECTED_SNP_IDS_SHA256:
                raise ValueError("retained SNP IDs do not match formal hash")
            if array_sha256(chromosomes) != EXPECTED_CHROMOSOMES_SHA256:
                raise ValueError("retained chromosomes do not match formal hash")
            if (
                array_sha256(np.asarray(fit.singular_values[:RANK], dtype=float))
                != EXPECTED_SINGULAR_VALUES_SHA256
            ):
                raise ValueError("singular values do not match formal hash")
            if array_sha256(loadings) != EXPECTED_LOADINGS_SHA256:
                raise ValueError("rebuilt loading matrix does not match formal hash")
            if array_sha256(importance) != EXPECTED_IMPORTANCE_SHA256:
                raise ValueError(
                    "rebuilt loading importance does not match formal hash"
                )
            if fit.sample_gram_sha256 != EXPECTED_SAMPLE_GRAM_SHA256:
                raise ValueError(
                    "rebuilt sample Gram matrix does not match formal hash"
                )
            if array_sha256(retained_original) != EXPECTED_RETAINED_ORIGINAL_SHA256:
                raise ValueError("retained SNP identity does not match formal hash")

            blocks = _chromosome_blocks(chromosomes)
            top_rows = _top_rows(
                ranking,
                TOP_CSV_COUNT,
                retained_original=retained_original,
                snp_ids=snp_ids,
                chromosomes=chromosomes,
                loadings=loadings,
                importance=importance,
            )
            np.savez_compressed(
                temporary / NPZ_FILENAME,
                train_indices=np.asarray(train, dtype=np.int64),
                test_indices=np.asarray(test, dtype=np.int64),
                retained_original_indices=retained_original,
                snp_ids=snp_ids,
                chromosomes=chromosomes,
                loadings=loadings,
                importance_l1_abs_sum=importance,
                singular_values=np.asarray(fit.singular_values[:RANK], dtype=float),
                explained_variance_ratio=np.asarray(
                    fit.explained_variance_ratio[:RANK], dtype=float
                ),
            )
            _write_top_csv(temporary / CSV_FILENAME, top_rows)
            _render_figure(
                temporary / PNG_FILENAME,
                importance=importance,
                loadings=loadings,
                chromosomes=chromosomes,
                snp_ids=snp_ids,
                blocks=blocks,
                top_rows=top_rows,
            )
            summary = _build_summary(
                input_identity=input_identity,
                retained_original=retained_original,
                snp_ids=snp_ids,
                chromosomes=chromosomes,
                loadings=loadings,
                importance=importance,
                singular_values=np.asarray(fit.singular_values[:RANK], dtype=float),
                explained_variance_ratio=np.asarray(
                    fit.explained_variance_ratio[:RANK], dtype=float
                ),
                sample_gram_sha256=fit.sample_gram_sha256,
                top_rows=top_rows,
                blocks=blocks,
                runtime_identity=runtime_identity,
                train_indices=np.asarray(train, dtype=np.int64),
                test_indices=np.asarray(test, dtype=np.int64),
            )
            (temporary / SUMMARY_FILENAME).write_bytes(_json_bytes(summary))

        artifacts = {name: _receipt(temporary / name) for name in ARTIFACT_FILENAMES}
        manifest: dict[str, Any] = {
            "schema_version": "rspca-amr-loading-visualization-manifest-v1",
            "status": "complete",
            "analysis_sha256": summary["analysis_sha256"],
            "artifacts": artifacts,
            "manifest_written_last": True,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_json_bytes(manifest))
        validate(temporary)
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate(target)


def validate(
    output_dir: Path,
    *,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    expected = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    if not root.is_dir() or root.is_symlink():
        raise ValueError("output package must be a regular directory")
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("output package file set mismatch")
    for path in root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"invalid artifact: {path.name}")

    manifest = _strict_json(root / MANIFEST_FILENAME)
    if set(manifest) != {
        "schema_version",
        "status",
        "analysis_sha256",
        "artifacts",
        "manifest_written_last",
        "package_anchor_sha256",
    }:
        raise ValueError("manifest schema mismatch")
    if (
        manifest["schema_version"] != "rspca-amr-loading-visualization-manifest-v1"
        or manifest["status"] != "complete"
        or manifest["manifest_written_last"] is not True
        or set(manifest["artifacts"]) != set(ARTIFACT_FILENAMES)
    ):
        raise ValueError("manifest semantic contract mismatch")
    unsigned_manifest = {
        key: value for key, value in manifest.items() if key != "package_anchor_sha256"
    }
    if canonical_sha256(unsigned_manifest) != manifest["package_anchor_sha256"]:
        raise ValueError("package anchor mismatch")
    if (
        expected_package_anchor_sha256 is not None
        and manifest["package_anchor_sha256"] != expected_package_anchor_sha256
    ):
        raise ValueError("external package anchor mismatch")
    for name in ARTIFACT_FILENAMES:
        if manifest["artifacts"].get(name) != _receipt(root / name):
            raise ValueError(f"artifact receipt mismatch: {name}")

    summary = _strict_json(root / SUMMARY_FILENAME)
    unsigned_summary = {
        key: value for key, value in summary.items() if key != "analysis_sha256"
    }
    if canonical_sha256(unsigned_summary) != summary.get("analysis_sha256"):
        raise ValueError("summary analysis hash mismatch")
    if manifest["analysis_sha256"] != summary["analysis_sha256"]:
        raise ValueError("manifest/summary analysis mismatch")
    if (
        expected_analysis_sha256 is not None
        and summary["analysis_sha256"] != expected_analysis_sha256
    ):
        raise ValueError("external analysis anchor mismatch")
    if (
        summary.get("schema_version") != "rspca-amr-loading-visualization-v1"
        or summary.get("status") != "complete_loading_only_presentation"
        or summary.get("fixed_cell") != FIXED_CELL
        or summary.get("method") != METHOD
        or summary.get("interpretation_boundary") != INTERPRETATION_BOUNDARY
        or summary.get("renderer_source_sha256") != _sha256(Path(__file__).resolve())
    ):
        raise ValueError("summary protocol/source identity mismatch")
    if summary.get("source_amr_formal_result") != {
        "analysis_sha256": EXPECTED_AMR_ANALYSIS_SHA256,
        "package_anchor_sha256": EXPECTED_AMR_PACKAGE_SHA256,
        "validated_before_loading_reconstruction": True,
    }:
        raise ValueError("source AMR formal-result binding mismatch")

    with np.load(root / NPZ_FILENAME, allow_pickle=False) as archive:
        if tuple(archive.files) != NPZ_KEYS:
            raise ValueError("loading NPZ schema mismatch")
        arrays = {key: np.asarray(archive[key]) for key in NPZ_KEYS}
    expected_shapes_and_dtypes = {
        "train_indices": ((800,), "<i8"),
        "test_indices": ((200,), "<i8"),
        "retained_original_indices": ((455_102,), "<i8"),
        "snp_ids": ((455_102,), "<U16"),
        "chromosomes": ((455_102,), "<U2"),
        "loadings": ((455_102, RANK), "<f8"),
        "importance_l1_abs_sum": ((455_102,), "<f8"),
        "singular_values": ((RANK,), "<f8"),
        "explained_variance_ratio": ((RANK,), "<f8"),
    }
    for key, (shape, dtype) in expected_shapes_and_dtypes.items():
        if arrays[key].shape != shape or arrays[key].dtype.str != dtype:
            raise ValueError(f"loading NPZ shape/dtype mismatch: {key}")
        if arrays[key].dtype.kind in "fc" and not np.isfinite(arrays[key]).all():
            raise ValueError(f"loading NPZ contains non-finite values: {key}")
    loadings = np.asarray(arrays["loadings"], dtype=float)
    importance = np.asarray(arrays["importance_l1_abs_sum"], dtype=float)
    retained = np.asarray(arrays["retained_original_indices"], dtype=np.int64)
    train_indices = np.asarray(arrays["train_indices"], dtype=np.int64)
    test_indices = np.asarray(arrays["test_indices"], dtype=np.int64)
    if loadings.shape != (allauto.FROZEN_FORMAL_SHAPE[1], RANK):
        raise ValueError("loading matrix shape mismatch")
    if not np.allclose(
        np.sum(np.abs(loadings), axis=1), importance, atol=0.0, rtol=0.0
    ):
        raise ValueError("importance is not exactly reconstructed from loadings")
    identity = summary.get("array_identity", {})
    expected_hashes = {
        "loadings_sha256": EXPECTED_LOADINGS_SHA256,
        "importance_sha256": EXPECTED_IMPORTANCE_SHA256,
        "retained_original_indices_sha256": EXPECTED_RETAINED_ORIGINAL_SHA256,
        "sample_gram_sha256": EXPECTED_SAMPLE_GRAM_SHA256,
    }
    for key, value in expected_hashes.items():
        if identity.get(key) != value:
            raise ValueError(f"formal array identity mismatch: {key}")
    if array_sha256(loadings) != EXPECTED_LOADINGS_SHA256:
        raise ValueError("NPZ loading hash mismatch")
    if array_sha256(importance) != EXPECTED_IMPORTANCE_SHA256:
        raise ValueError("NPZ importance hash mismatch")
    if array_sha256(retained) != EXPECTED_RETAINED_ORIGINAL_SHA256:
        raise ValueError("NPZ retained-index hash mismatch")
    if array_sha256(train_indices) != EXPECTED_TRAIN_INDICES_SHA256:
        raise ValueError("NPZ train-index hash mismatch")
    if array_sha256(test_indices) != EXPECTED_TEST_INDICES_SHA256:
        raise ValueError("NPZ test-index hash mismatch")
    if array_sha256(arrays["snp_ids"]) != EXPECTED_SNP_IDS_SHA256:
        raise ValueError("NPZ SNP-ID hash mismatch")
    if array_sha256(arrays["chromosomes"]) != EXPECTED_CHROMOSOMES_SHA256:
        raise ValueError("NPZ chromosome hash mismatch")
    if array_sha256(arrays["singular_values"]) != EXPECTED_SINGULAR_VALUES_SHA256:
        raise ValueError("NPZ singular-value hash mismatch")
    if (
        array_sha256(arrays["explained_variance_ratio"])
        != EXPECTED_EXPLAINED_VARIANCE_RATIO_SHA256
    ):
        raise ValueError("NPZ explained-variance hash mismatch")
    if summary.get("split") != {
        "test_fraction": TEST_FRACTION,
        "train_samples": 800,
        "test_samples": 200,
        "train_indices_sha256": EXPECTED_TRAIN_INDICES_SHA256,
        "test_indices_sha256": EXPECTED_TEST_INDICES_SHA256,
    }:
        raise ValueError("summary split identity mismatch")
    if not np.allclose(loadings.T @ loadings, np.eye(RANK), atol=3e-7, rtol=0.0):
        raise ValueError("loadings are not orthonormal")

    with Image.open(root / PNG_FILENAME) as image:
        image.verify()
    with Image.open(root / PNG_FILENAME) as image:
        if image.width < 4000 or image.height < 2500:
            raise ValueError("loading figure resolution is too small")

    with (root / CSV_FILENAME).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "importance_rank",
            "loading_row",
            "original_snp_column",
            "chromosome",
            "snp_id",
            "importance_l1_abs_sum",
            *[f"PC{component}_signed_loading" for component in range(1, RANK + 1)],
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError("top-loading CSV schema mismatch")
        rows = list(reader)
    if len(rows) != TOP_CSV_COUNT:
        raise ValueError("top-loading CSV row count mismatch")
    ranking = np.lexsort((np.arange(importance.size, dtype=np.int64), -importance))
    for position, row in enumerate(rows, start=1):
        loading_row = int(ranking[position - 1])
        if (
            int(row["importance_rank"]) != position
            or int(row["loading_row"]) != loading_row
            or row["snp_id"] != str(arrays["snp_ids"][loading_row])
            or row["chromosome"] != str(arrays["chromosomes"][loading_row])
            or int(row["original_snp_column"]) != int(retained[loading_row])
            or float(row["importance_l1_abs_sum"]) != float(importance[loading_row])
        ):
            raise ValueError(f"top-loading CSV mismatch at rank {position}")
        for component in range(1, RANK + 1):
            if float(row[f"PC{component}_signed_loading"]) != float(
                loadings[loading_row, component - 1]
            ):
                raise ValueError(
                    f"top-loading CSV PC{component} mismatch at rank {position}"
                )

    chromosomes = np.asarray(arrays["chromosomes"], dtype=str)
    snp_ids = np.asarray(arrays["snp_ids"], dtype=str)
    blocks = _chromosome_blocks(chromosomes)
    expected_top = _top_rows(
        ranking,
        TOP_CSV_COUNT,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
        loadings=loadings,
        importance=importance,
    )
    if summary.get("chromosome_blocks") != blocks:
        raise ValueError("summary chromosome blocks mismatch")
    if summary.get("chromosome_pc_loading_energy") != _chromosome_energy(
        loadings, chromosomes
    ):
        raise ValueError("summary chromosome energy mismatch")
    if summary.get("top40_by_l1_abs_sum") != expected_top[:TOP_HEATMAP_COUNT]:
        raise ValueError("summary top-40 loading rows mismatch")
    if summary.get("top100_per_pc_by_absolute_loading") != _top_per_pc(
        loadings,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
    ):
        raise ValueError("summary per-PC loading rows mismatch")
    if summary.get("skyline_bin_maxima") != _skyline_bin_maxima(
        loadings,
        retained_original=retained,
        snp_ids=snp_ids,
        chromosomes=chromosomes,
    ):
        raise ValueError("summary skyline-bin maxima mismatch")
    if summary.get("skyline_bin_rule") != {
        "bins_per_chromosome": SKYLINE_BINS_PER_CHROMOSOME,
        "point": "largest_absolute_loading_within_each_chromosome_bin_and_PC",
        "used_for_fit_or_selection": False,
    }:
        raise ValueError("summary skyline-bin rule mismatch")
    input_identity = summary.get("input_identity", {})
    if (
        input_identity.get("input_scope") != "all_autosomes"
        or input_identity.get("samples") != 1000
        or input_identity.get("snps") != 455_102
        or input_identity.get("genotype_dtype") != "<f4"
        or input_identity.get("genotype_npz_sha256") != allauto.FROZEN_FORMAL_NPZ_SHA256
        or input_identity.get("external_source_manifest_sha256")
        != allauto.FROZEN_FORMAL_MANIFEST_SHA256
    ):
        raise ValueError("summary frozen-input identity mismatch")

    return {
        "valid": True,
        "analysis_sha256": summary["analysis_sha256"],
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "seed": SEED,
        "n_components": RANK,
        "retained_snps": int(loadings.shape[0]),
        "formal_loading_hash_verified": True,
        "raw_and_whitened_loadings_are_identical": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--genotypes", type=Path, required=True)
    run_parser.add_argument("--source-manifest", type=Path, required=True)
    run_parser.add_argument("--amr-package", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--scratch-root", type=Path, required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("output_dir", type=Path)
    validate_parser.add_argument("--expected-analysis-sha256")
    validate_parser.add_argument("--expected-package-anchor-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = run(
            genotypes=arguments.genotypes,
            source_manifest=arguments.source_manifest,
            amr_package=arguments.amr_package,
            output_dir=arguments.output_dir,
            scratch_root=arguments.scratch_root,
            repo_root=arguments.repo_root,
        )
    else:
        result = validate(
            arguments.output_dir,
            expected_analysis_sha256=arguments.expected_analysis_sha256,
            expected_package_anchor_sha256=(arguments.expected_package_anchor_sha256),
        )
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

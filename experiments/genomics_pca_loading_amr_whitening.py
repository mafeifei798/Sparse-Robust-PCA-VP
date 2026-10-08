#!/usr/bin/env python3
"""Paired AMR readout audit for the all-autosome PCA/loading pipeline.

This additive runner leaves the repository's ordinary PCA, smartpca-like
training scaling, loading-only SNP ranking, and split-specific panel refits
unchanged.  It evaluates two nearest-centroid distance geometries on exactly
the same fitted scores:

* ``raw_euclidean``: the historical score-space Euclidean distance;
* ``train_singular_whitened_euclidean``: each PC is divided by its
  outer-training standard deviation, ``s / sqrt(n_train - 1)``.

The fixed evaluation is 20 fresh split seeds crossed with the complete
historical rank 4/6/8 by 5/10/25/50%/all panel grid.  Every raw and whitened
cell is reported.  There is no target pattern, winner selection, robust/SPCA,
federated method, Fisher/ANOVA ranking, or use of fine-population labels in
training.  The fine-population field is attached only after prediction for an
error audit.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

if __package__:
    from . import genomics_pca_loading_all_autosomes as allauto
    from . import genomics_pca_loading_grid as base
else:
    import genomics_pca_loading_all_autosomes as allauto  # type: ignore[no-redef]
    import genomics_pca_loading_grid as base  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import stratified_split


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

FORMAL_SEEDS = tuple(range(731001, 731021))
COMPONENT_GRID = (4, 6, 8)
PANEL_FRACTIONS: tuple[float | None, ...] = (0.05, 0.10, 0.25, 0.50, None)
DISTANCE_MODES = (
    "raw_euclidean",
    "train_singular_whitened_euclidean",
)
PAIRED_METRICS = (
    "balanced_accuracy",
    "macro_f1",
    "AMR_recall",
    "AMR_precision",
    "EUR_recall",
    "EUR_precision",
)
EXPECTED_POPULATIONS = ("AFR", "AMR", "EAS", "EUR", "SAS")
BOOTSTRAP_SEED = 8_312_027
BOOTSTRAP_RESAMPLES = 100_000
HEADLINE_RANK = 8
HEADLINE_PANEL = "all"

FROZEN_INPUT_NPZ_SHA256 = allauto.FROZEN_FORMAL_NPZ_SHA256
FROZEN_INPUT_MANIFEST_SHA256 = allauto.FROZEN_FORMAL_MANIFEST_SHA256
FROZEN_SUBPOP_MANIFEST_SHA256 = (
    "9378814177e0721e711ce2e6c3695effc5ff766d2529c17cc5c4e0de84f30342"
)
FROZEN_SELECTED_RECORDS_SHA256 = (
    "0616dd6392c601cc1c6d9c559d0d4bc74d636166af739ee9d0a0009c6c675253"
)
FROZEN_SUBPOPULATION_ARRAY_SHA256 = (
    "e4a6749390c74b2f3309feaf662c229eba16b83d4bf4ec0fc320874d17d1d67b"
)
FROZEN_LEGACY_ANALYSIS_SHA256 = (
    "46190a1e12c9c0213b70b8e9778579cebcf8374915874436d26cea1805402981"
)
FROZEN_LEGACY_PACKAGE_ANCHOR_SHA256 = (
    "4afd97e46570a3cec4d32be83c1abc878d6b1ee04bb7a6efc0d5c66757040d9a"
)

RESULT_FILENAME = "amr_whitening_results.json"
GRID_FILENAME = "amr_whitening_grid.csv"
PAIRED_FILENAME = "amr_whitening_paired_deltas.csv"
PREDICTION_FILENAME = "amr_whitening_predictions.csv"
HEATMAP_FILENAME = "amr_whitening_full_grid.png"
HEADLINE_FILENAME = "amr_whitening_rank8_all_tradeoff.png"
MANIFEST_FILENAME = "amr_whitening_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    GRID_FILENAME,
    PAIRED_FILENAME,
    PREDICTION_FILENAME,
    HEATMAP_FILENAME,
    HEADLINE_FILENAME,
)


@dataclass(frozen=True)
class WhiteningConfig:
    seeds: tuple[int, ...] = FORMAL_SEEDS
    component_grid: tuple[int, ...] = COMPONENT_GRID
    panel_fractions: tuple[float | None, ...] = PANEL_FRACTIONS
    distance_modes: tuple[str, ...] = DISTANCE_MODES
    test_fraction: float = 0.2
    snp_chunk_size: int = 2048
    analysis_mode: str = "fresh_fixed_protocol"
    analysis_status: str = "post_diagnosis_fresh_split_evaluation"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON token: {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON key: {key}")
            output[key] = value
        return output

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_pairs,
    )


def _panel_label(fraction: float | None) -> str:
    return "all" if fraction is None else f"{int(round(100.0 * fraction))}pct"


def _config_payload(config: WhiteningConfig) -> dict[str, Any]:
    output = asdict(config)
    for key in ("seeds", "component_grid", "panel_fractions", "distance_modes"):
        output[key] = list(output[key])
    return output


def _validate_config(config: WhiteningConfig) -> None:
    if tuple(config.component_grid) != COMPONENT_GRID:
        raise ValueError("component grid is frozen to 4/6/8")
    if tuple(config.panel_fractions) != PANEL_FRACTIONS:
        raise ValueError("panel grid is frozen to 5/10/25/50/all")
    if tuple(config.distance_modes) != DISTANCE_MODES:
        raise ValueError("distance modes are frozen")
    if config.test_fraction != 0.2 or config.snp_chunk_size < 1:
        raise ValueError("split fraction/chunk size changed")
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be a non-empty unique sequence")
    if config.analysis_mode in {"fresh_fixed_protocol", "fresh_registered_shard"}:
        if not set(config.seeds).issubset(FORMAL_SEEDS):
            raise ValueError("formal evaluation may use only the frozen fresh bank")
        if (
            config.analysis_mode == "fresh_fixed_protocol"
            and config.seeds != FORMAL_SEEDS
        ):
            raise ValueError("full formal evaluation requires exact fresh seed cover")


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if root != Path(__file__).resolve().parents[1]:
        raise ValueError("repo_root must contain the executing runner")
    inherited = allauto._source_identity(root)
    relative = "experiments/genomics_pca_loading_amr_whitening.py"
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError("AMR whitening runner must be a regular file")
    output = dict(inherited)
    files = dict(output["files"])
    sizes = dict(output["file_bytes"])
    roles = dict(output["roles"])
    files[relative] = _file_sha256(path)
    sizes[relative] = int(path.stat().st_size)
    roles[relative] = "paired_train_only_pc_whitening_readout"
    output.update(
        {
            "scope": "ordinary_pca_loading_plus_whitening_readout_dependencies",
            "files": files,
            "file_bytes": sizes,
            "roles": roles,
            "extension_source_sha256": files[relative],
        }
    )
    output["identity_sha256"] = canonical_sha256(output)
    return output


def _load_subpopulations(
    manifest_path: Path,
    sample_ids: StringArray,
    population_labels: StringArray,
) -> tuple[StringArray, dict[str, Any]]:
    path = Path(manifest_path).resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("subpopulation manifest must be a regular file")
    observed_hash = _file_sha256(path)
    if observed_hash != FROZEN_SUBPOP_MANIFEST_SHA256:
        raise ValueError("subpopulation manifest hash mismatch")
    payload = _strict_json_load(path)
    records = payload.get("sample_selection", {}).get("selected_records")
    if not isinstance(records, list) or len(records) != sample_ids.size:
        raise ValueError("subpopulation selected_records must contain 1000 rows")
    if (
        canonical_sha256({"selected_records": records})
        != FROZEN_SELECTED_RECORDS_SHA256
    ):
        # The historical receipt hashes the list itself rather than a wrapper.
        if canonical_sha256(records) != FROZEN_SELECTED_RECORDS_SHA256:
            raise ValueError("selected_records hash mismatch")
    keyed: dict[str, tuple[str, str]] = {}
    for row in records:
        if not isinstance(row, Mapping) or set(row) != {
            "sample_id",
            "population_label",
            "subpopulation",
        }:
            raise ValueError("invalid selected_records row")
        sample = str(row["sample_id"])
        if sample in keyed:
            raise ValueError("duplicate selected sample ID")
        keyed[sample] = (str(row["population_label"]), str(row["subpopulation"]))
    output: list[str] = []
    for sample, population in zip(
        sample_ids.tolist(), population_labels.tolist(), strict=True
    ):
        if sample not in keyed or keyed[sample][0] != population:
            raise ValueError(
                "ordered NPZ sample IDs/labels do not match subpopulation manifest"
            )
        output.append(keyed[sample][1])
    values = np.asarray(output, dtype="<U3")
    if array_sha256(values) != FROZEN_SUBPOPULATION_ARRAY_SHA256:
        raise ValueError("ordered subpopulation array hash mismatch")
    return values, {
        "manifest_sha256": observed_hash,
        "selected_records_sha256": FROZEN_SELECTED_RECORDS_SHA256,
        "ordered_subpopulation_array_sha256": array_sha256(values),
        "used_for_training_split_ranking_or_selection": False,
        "scope": "post_prediction_error_audit_only",
    }


def _score_cell(
    fit: allauto.ChunkedPCAFit,
    *,
    n_components: int,
    train_labels: StringArray,
    test_labels: StringArray,
    distance_mode: str,
) -> tuple[
    StringArray, StringArray, FloatArray, dict[str, Any], FloatArray, FloatArray
]:
    rank = int(n_components)
    singular = np.asarray(fit.singular_values[:rank], dtype=float)
    if (
        singular.shape != (rank,)
        or not np.isfinite(singular).all()
        or np.any(singular <= 0.0)
    ):
        raise ValueError("PCA singular values are invalid for whitening")
    if distance_mode == "raw_euclidean":
        scale = np.ones(rank, dtype=float)
    elif distance_mode == "train_singular_whitened_euclidean":
        scale = singular / math.sqrt(float(fit.train_scores.shape[0] - 1))
    else:
        raise ValueError("unknown distance mode")
    train_scores = np.asarray(fit.train_scores[:, :rank], dtype=float) / scale[None, :]
    test_scores = np.asarray(fit.test_scores[:, :rank], dtype=float) / scale[None, :]
    predictions, classes, centroids = base._nearest_centroid(
        train_scores,
        np.asarray(train_labels, dtype=str),
        test_scores,
    )
    if tuple(classes.tolist()) != EXPECTED_POPULATIONS:
        raise ValueError("unexpected population class order")
    metrics = base._classification_metrics(
        np.asarray(test_labels, dtype=str), predictions, classes
    )
    positions = {value: index for index, value in enumerate(classes.tolist())}
    amr_distance = np.sum(
        (test_scores - centroids[positions["AMR"]][None, :]) ** 2, axis=1
    )
    eur_distance = np.sum(
        (test_scores - centroids[positions["EUR"]][None, :]) ** 2, axis=1
    )
    margin = np.asarray(eur_distance - amr_distance, dtype=float)
    return predictions, classes, centroids, metrics, margin, scale


def _metric_for_population(
    record: Mapping[str, Any], population: str, key: str
) -> float:
    for row in record["per_population_metrics"]:
        if row["population"] == population:
            return float(row[key])
    raise ValueError(f"missing population metric: {population}")


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("summary values must be a finite vector")
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _grid_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for rank in COMPONENT_GRID:
        for fraction in PANEL_FRACTIONS:
            panel = _panel_label(fraction)
            for mode in DISTANCE_MODES:
                rows = sorted(
                    (
                        row
                        for row in records
                        if row["n_components"] == rank
                        and row["panel_label"] == panel
                        and row["distance_mode"] == mode
                    ),
                    key=lambda row: row["seed"],
                )
                if not rows:
                    continue
                item: dict[str, Any] = {
                    "n_components": rank,
                    "panel_label": panel,
                    "requested_panel_fraction": fraction,
                    "distance_mode": mode,
                    "seeds": [int(row["seed"]) for row in rows],
                    "evaluated_splits": len(rows),
                    "selected_snp_count_mean": float(
                        np.mean([row["selected_snp_count"] for row in rows])
                    ),
                }
                for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
                    item[metric] = _mean_std([float(row[metric]) for row in rows])
                item["population_metrics"] = {
                    population: {
                        key: _mean_std(
                            [
                                _metric_for_population(row, population, key)
                                for row in rows
                            ]
                        )
                        for key in ("recall", "precision", "f1")
                    }
                    for population in EXPECTED_POPULATIONS
                }
                output.append(item)
    return output


def _paired_summaries(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    seed_count = len({int(row["seed"]) for row in records})
    bootstrap_indices = rng.integers(
        0, seed_count, size=(BOOTSTRAP_RESAMPLES, seed_count), dtype=np.int64
    )
    output: list[dict[str, Any]] = []
    for rank in COMPONENT_GRID:
        for fraction in PANEL_FRACTIONS:
            panel = _panel_label(fraction)
            keyed = {
                (int(row["seed"]), str(row["distance_mode"])): row
                for row in records
                if row["n_components"] == rank and row["panel_label"] == panel
            }
            seeds = sorted({seed for seed, _ in keyed})
            if not seeds:
                continue
            if len(seeds) != seed_count:
                raise ValueError(
                    "paired cell seed cover differs from result seed cover"
                )
            item: dict[str, Any] = {
                "n_components": rank,
                "panel_label": panel,
                "requested_panel_fraction": fraction,
                "raw_mode": DISTANCE_MODES[0],
                "candidate_mode": DISTANCE_MODES[1],
                "seeds": seeds,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                "uncertainty_scope": (
                    "paired_split_randomization_on_one_fixed_transductively_prepared_cohort"
                ),
                "simultaneous_grid_interval": False,
                "deltas": {},
            }
            for metric in PAIRED_METRICS:
                values: list[float] = []
                for seed in seeds:
                    raw = keyed[(seed, DISTANCE_MODES[0])]
                    white = keyed[(seed, DISTANCE_MODES[1])]
                    if metric.startswith("AMR_") or metric.startswith("EUR_"):
                        population, key = metric.split("_", maxsplit=1)
                        left = _metric_for_population(raw, population, key)
                        right = _metric_for_population(white, population, key)
                    else:
                        left = float(raw[metric])
                        right = float(white[metric])
                    values.append(right - left)
                delta = np.asarray(values, dtype=float)
                draws = np.mean(delta[bootstrap_indices], axis=1)
                item["deltas"][metric] = {
                    "candidate_minus_raw_mean": float(np.mean(delta)),
                    "paired_percentile_95_ci": [
                        float(np.percentile(draws, 2.5)),
                        float(np.percentile(draws, 97.5)),
                    ],
                    "positive_seeds": int(np.count_nonzero(delta > 0.0)),
                    "equal_seeds": int(np.count_nonzero(delta == 0.0)),
                    "negative_seeds": int(np.count_nonzero(delta < 0.0)),
                    "per_seed": [float(value) for value in delta],
                }
            output.append(item)
    return output


def _transition_ledger(
    predictions: Sequence[Mapping[str, Any]],
    *,
    n_components: int = HEADLINE_RANK,
    panel_label: str = HEADLINE_PANEL,
) -> dict[str, Any]:
    rows = [
        row
        for row in predictions
        if row["n_components"] == n_components and row["panel_label"] == panel_label
    ]
    keyed = {
        (
            int(row["seed"]),
            int(row["sample_position"]),
            str(row["distance_mode"]),
        ): row
        for row in rows
    }
    pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for seed, position, mode in sorted(keyed):
        if mode != DISTANCE_MODES[0]:
            continue
        pairs.append(
            (
                keyed[(seed, position, DISTANCE_MODES[0])],
                keyed[(seed, position, DISTANCE_MODES[1])],
            )
        )
    if not pairs:
        raise ValueError("headline transition ledger has no paired predictions")
    transitions: dict[str, dict[str, int]] = {}
    for population in EXPECTED_POPULATIONS:
        population_pairs = [pair for pair in pairs if pair[0]["truth"] == population]
        transitions[population] = {
            "support": len(population_pairs),
            "raw_wrong_to_whitened_correct": sum(
                (not bool(raw["correct"])) and bool(white["correct"])
                for raw, white in population_pairs
            ),
            "raw_correct_to_whitened_wrong": sum(
                bool(raw["correct"]) and (not bool(white["correct"]))
                for raw, white in population_pairs
            ),
            "both_correct": sum(
                bool(raw["correct"]) and bool(white["correct"])
                for raw, white in population_pairs
            ),
            "both_wrong": sum(
                (not bool(raw["correct"])) and (not bool(white["correct"]))
                for raw, white in population_pairs
            ),
        }
    amr_pairs = [pair for pair in pairs if pair[0]["truth"] == "AMR"]
    eur_pairs = [pair for pair in pairs if pair[0]["truth"] == "EUR"]
    return {
        "n_components": n_components,
        "panel_label": panel_label,
        "selection_reason": "historical_formal_headline_alignment_not_new_result_selection",
        "population_transitions": transitions,
        "raw_AMR_to_EUR": sum(raw["prediction"] == "EUR" for raw, _ in amr_pairs),
        "whitened_AMR_to_EUR": sum(
            white["prediction"] == "EUR" for _, white in amr_pairs
        ),
        "raw_EUR_to_AMR": sum(raw["prediction"] == "AMR" for raw, _ in eur_pairs),
        "whitened_EUR_to_AMR": sum(
            white["prediction"] == "AMR" for _, white in eur_pairs
        ),
    }


def evaluate(
    data: base.GenotypeInput,
    config: WhiteningConfig,
    *,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    subpopulations: StringArray | None = None,
    subpopulation_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the fixed paired grid, sharing every PCA fit across distance modes."""

    _validate_config(config)
    formal = config.analysis_mode in {"fresh_fixed_protocol", "fresh_registered_shard"}
    if formal:
        blas_identity = allauto._require_formal_blas_thread_environment()
    else:
        blas_identity = allauto._blas_runtime_identity()
    allauto._validate_input(
        data,
        require_formal_cohort=formal,
        chunk_size=int(config.snp_chunk_size),
    )
    if formal and (
        input_identity.get("input_scope") != "all_autosomes"
        or input_identity.get("external_source_manifest_anchor_verified") is not True
    ):
        raise ValueError("formal evaluation requires the frozen all-autosome input")
    raw = np.asanyarray(data.genotypes)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    if subpopulations is None:
        subpopulations = np.full(sample_ids.size, "NA", dtype="<U3")
        subpopulation_identity = {
            "scope": "unavailable_in_synthetic_smoke",
            "used_for_training_split_ranking_or_selection": False,
        }
    subpopulations = np.asarray(subpopulations, dtype=str)
    if subpopulations.shape != sample_ids.shape:
        raise ValueError("subpopulation array must align with sample IDs")

    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []
    selection_receipts: list[dict[str, Any]] = []

    for seed in config.seeds:
        print(
            json.dumps({"event": "amr_whitening_seed_start", "seed": seed}),
            file=os.sys.stderr,
            flush=True,
        )
        train, test = stratified_split(labels, config.test_fraction, int(seed))
        full_fit = allauto.fit_chunked_top_pca(
            raw,
            train,
            test,
            n_components=max(COMPONENT_GRID),
            chunk_size=config.snp_chunk_size,
        )
        retained = full_fit.retained_original
        panel_specs = [
            row
            for row in allauto._resolved_panels(int(retained.size))
            if row["active_grid_cell"]
        ]
        split_receipt = {
            "seed": int(seed),
            "train_indices_sha256": array_sha256(train),
            "test_indices_sha256": array_sha256(test),
            "train_sample_ids_sha256": array_sha256(sample_ids[train]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test]),
            "test_labels_sha256": array_sha256(labels[test]),
            "retained_snps": int(retained.size),
            "retained_original_indices_sha256": array_sha256(retained),
            "full_sample_gram_sha256": full_fit.sample_gram_sha256,
            "full_singular_values": [
                float(value) for value in full_fit.singular_values
            ],
        }
        split_receipts.append(split_receipt)

        for rank in COMPONENT_GRID:
            ranking, importance = base.rank_by_pca_loadings(full_fit.loadings, rank)
            ordered_original = retained[ranking]
            finite_specs = [
                row for row in panel_specs if row["requested_fraction"] is not None
            ]
            finite_counts = [int(row["effective_snp_count"]) for row in finite_specs]
            nested = allauto.fit_nested_chunked_panels(
                raw,
                train,
                test,
                ordered_original_indices=ordered_original,
                panel_counts=finite_counts,
                mean_dosage=full_fit.mean_dosage,
                scale=full_fit.scale,
                n_components=max(COMPONENT_GRID),
                chunk_size=config.snp_chunk_size,
            )
            for panel_spec in panel_specs:
                fraction = panel_spec["requested_fraction"]
                if fraction is None:
                    fit = full_fit
                    selected_original = retained
                else:
                    count = int(panel_spec["effective_snp_count"])
                    fit = nested[count]
                    selected_original = np.sort(ordered_original[:count]).astype(
                        np.int64
                    )
                    if not np.array_equal(fit.retained_original, selected_original):
                        raise RuntimeError("nested panel identity mismatch")
                panel = str(panel_spec["panel_label"])
                panel_hash = array_sha256(selected_original)
                selection_receipts.append(
                    {
                        "seed": int(seed),
                        "n_components": rank,
                        "panel_label": panel,
                        "requested_panel_fraction": fraction,
                        "selected_snp_count": int(selected_original.size),
                        "selected_original_indices_sha256": panel_hash,
                        "loading_importance_sha256": array_sha256(importance),
                        "population_labels_used": False,
                        "test_rows_used": False,
                    }
                )
                shared = {
                    "seed": int(seed),
                    "n_components": rank,
                    "panel_label": panel,
                    "requested_panel_fraction": fraction,
                    "train_sample_count": int(train.size),
                    "test_sample_count": int(test.size),
                    "selected_snp_count": int(selected_original.size),
                    "selected_original_indices_sha256": panel_hash,
                    "sample_gram_sha256": fit.sample_gram_sha256,
                    "loadings_prefix_sha256": array_sha256(fit.loadings[:, :rank]),
                    "singular_values": [
                        float(value) for value in fit.singular_values[:rank]
                    ],
                    "train_indices_sha256": split_receipt["train_indices_sha256"],
                    "test_indices_sha256": split_receipt["test_indices_sha256"],
                    "test_sample_ids_sha256": split_receipt["test_sample_ids_sha256"],
                    "test_labels_sha256": split_receipt["test_labels_sha256"],
                    "pca_and_panel_shared_across_distance_modes": True,
                }
                for mode in DISTANCE_MODES:
                    observed, classes, centroids, metrics, margins, scales = (
                        _score_cell(
                            fit,
                            n_components=rank,
                            train_labels=labels[train],
                            test_labels=labels[test],
                            distance_mode=mode,
                        )
                    )
                    record = {
                        **shared,
                        "distance_mode": mode,
                        "distance_scale_sha256": array_sha256(scales),
                        "distance_scale": [float(value) for value in scales],
                        "classifier": "nearest_outer_train_population_centroid",
                        "whitening_uses_outer_train_singular_values_only": mode
                        == DISTANCE_MODES[1],
                        "test_predictions_sha256": array_sha256(observed),
                        "AMR_EUR_margin_sha256": array_sha256(margins),
                        **metrics,
                    }
                    records.append(record)
                    for position, sample_index in enumerate(test.tolist()):
                        predictions.append(
                            {
                                "seed": int(seed),
                                "n_components": rank,
                                "panel_label": panel,
                                "requested_panel_fraction": fraction,
                                "distance_mode": mode,
                                "sample_position": int(position),
                                "sample_index": int(sample_index),
                                "sample_id": str(sample_ids[sample_index]),
                                "truth": str(labels[sample_index]),
                                "subpopulation": str(subpopulations[sample_index]),
                                "prediction": str(observed[position]),
                                "correct": bool(
                                    observed[position] == labels[sample_index]
                                ),
                                "AMR_EUR_margin_positive_favors_AMR": float(
                                    margins[position]
                                ),
                                "selected_original_indices_sha256": panel_hash,
                                "test_indices_sha256": split_receipt[
                                    "test_indices_sha256"
                                ],
                                "test_predictions_sha256": record[
                                    "test_predictions_sha256"
                                ],
                            }
                        )
        print(
            json.dumps({"event": "amr_whitening_seed_complete", "seed": seed}),
            file=os.sys.stderr,
            flush=True,
        )

    grid = _grid_summaries(records)
    paired = _paired_summaries(records)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_policy": {
            "status": config.analysis_status,
            "fresh_seed_evaluation": formal,
            "best_configuration_selected": False,
            "all_30_cells_reported": True,
            "headline_rank_panel": [HEADLINE_RANK, HEADLINE_PANEL],
            "headline_reason": "historical_formal_alignment_not_new_result_selection",
            "development_observation": "seed_2025_supported_whitening_feasibility_only",
            "fine_population_labels_used_for_modeling": False,
            "upstream_qc_ld_scope": "transductive_unsupervised_same_frozen_1000_before_outer_split",
            "uncertainty_scope": "split_randomization_not_subject_or_external_cohort_sampling",
        },
        "config": _config_payload(config),
        "method_scope": {
            "ordinary_pca": True,
            "loading_only_panel_ranking": True,
            "train_only_score_whitening": True,
            "robust_vp_randomized_federated_lpca_alpca_fisher_run": False,
        },
        "input_identity": dict(input_identity),
        "subpopulation_identity": dict(subpopulation_identity or {}),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "blas_runtime_identity": blas_identity,
        "execution_provenance": {
            "execution_mode": config.analysis_mode,
            "frozen_fresh_seed_bank": list(FORMAL_SEEDS),
            "included_seeds": list(config.seeds),
            "exact_fresh_seed_cover": config.seeds == FORMAL_SEEDS,
        },
        "split_receipts": split_receipts,
        "selection_receipts": selection_receipts,
        "records": records,
        "predictions": predictions,
        "grid_summaries": grid,
        "paired_summaries": paired,
        "headline_transition_ledger": _transition_ledger(predictions),
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_result_payload(payload)
    return payload


def _first_mismatch(expected: Any, observed: Any, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return f"{path}: type {type(expected).__name__} != {type(observed).__name__}"
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return f"{path}: mapping keys differ"
        for key in sorted(expected):
            mismatch = _first_mismatch(expected[key], observed[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return f"{path}: list lengths differ"
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            mismatch = _first_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if isinstance(expected, float):
        if (
            not math.isfinite(expected)
            or not math.isfinite(observed)
            or expected != observed
        ):
            return f"{path}: float values differ"
        return None
    return None if expected == observed else f"{path}: values differ"


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild all metrics and paired summaries from stored prediction rows."""

    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ValueError("invalid AMR whitening result schema")
    reported_digest = payload.get("analysis_sha256")
    unsigned = {
        key: value for key, value in payload.items() if key != "analysis_sha256"
    }
    if (
        not isinstance(reported_digest, str)
        or canonical_sha256(unsigned) != reported_digest
    ):
        raise ValueError("analysis SHA mismatch")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("missing config")
    config = WhiteningConfig(
        seeds=tuple(int(value) for value in config_payload["seeds"]),
        component_grid=tuple(int(value) for value in config_payload["component_grid"]),
        panel_fractions=tuple(config_payload["panel_fractions"]),
        distance_modes=tuple(str(value) for value in config_payload["distance_modes"]),
        test_fraction=float(config_payload["test_fraction"]),
        snp_chunk_size=int(config_payload["snp_chunk_size"]),
        analysis_mode=str(config_payload["analysis_mode"]),
        analysis_status=str(config_payload["analysis_status"]),
    )
    _validate_config(config)
    if _config_payload(config) != dict(config_payload):
        raise ValueError("config canonical representation mismatch")
    formal = config.analysis_mode in {
        "fresh_fixed_protocol",
        "fresh_registered_shard",
    }
    if formal:
        input_identity = payload.get("input_identity")
        if not isinstance(input_identity, Mapping):
            raise ValueError("formal input identity is missing")
        required_input = {
            "input_scope": "all_autosomes",
            "samples": 1000,
            "snps": 455102,
            "genotype_npz_bytes": allauto.FROZEN_FORMAL_NPZ_BYTES,
            "genotype_npz_sha256": FROZEN_INPUT_NPZ_SHA256,
            "expected_external_source_manifest_sha256": (FROZEN_INPUT_MANIFEST_SHA256),
            "external_source_manifest_sha256": FROZEN_INPUT_MANIFEST_SHA256,
            "external_source_manifest_anchor_verified": True,
            "prepared_array_sha256": allauto.FROZEN_PREPARED_ARRAY_SHA256,
        }
        for key, value in required_input.items():
            if input_identity.get(key) != value:
                raise ValueError(f"formal input identity mismatch: {key}")
        if input_identity.get("population_counts") != {
            population: 200 for population in EXPECTED_POPULATIONS
        }:
            raise ValueError("formal population count identity mismatch")
        subpopulation_identity = payload.get("subpopulation_identity")
        required_subpopulation = {
            "manifest_sha256": FROZEN_SUBPOP_MANIFEST_SHA256,
            "selected_records_sha256": FROZEN_SELECTED_RECORDS_SHA256,
            "ordered_subpopulation_array_sha256": (FROZEN_SUBPOPULATION_ARRAY_SHA256),
            "used_for_training_split_ranking_or_selection": False,
            "scope": "post_prediction_error_audit_only",
        }
        if not isinstance(subpopulation_identity, Mapping):
            raise ValueError("formal subpopulation identity is missing")
        for key, value in required_subpopulation.items():
            if subpopulation_identity.get(key) != value:
                raise ValueError(f"formal subpopulation identity mismatch: {key}")
    records = payload.get("records")
    predictions = payload.get("predictions")
    if not isinstance(records, list) or not isinstance(predictions, list):
        raise ValueError("records/predictions must be lists")
    expected_records = (
        len(config.seeds)
        * len(COMPONENT_GRID)
        * len(PANEL_FRACTIONS)
        * len(DISTANCE_MODES)
    )
    if len(records) != expected_records:
        raise ValueError("record count does not close over fixed grid")
    split_receipts = payload.get("split_receipts")
    selection_receipts = payload.get("selection_receipts")
    if not isinstance(split_receipts, list) or not isinstance(selection_receipts, list):
        raise ValueError("split/selection receipts must be lists")
    if len(split_receipts) != len(config.seeds):
        raise ValueError("split receipt count mismatch")
    split_by_seed: dict[int, Mapping[str, Any]] = {}
    for receipt in split_receipts:
        seed = int(receipt["seed"])
        if seed in split_by_seed:
            raise ValueError("duplicate split receipt")
        split_by_seed[seed] = receipt
    if set(split_by_seed) != set(config.seeds):
        raise ValueError("split receipts do not cover configured seeds")
    expected_selection_count = (
        len(config.seeds) * len(COMPONENT_GRID) * len(PANEL_FRACTIONS)
    )
    if len(selection_receipts) != expected_selection_count:
        raise ValueError("selection receipt count mismatch")
    selection_by_key: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    for receipt in selection_receipts:
        key = (
            int(receipt["seed"]),
            int(receipt["n_components"]),
            str(receipt["panel_label"]),
        )
        if key in selection_by_key:
            raise ValueError("duplicate selection receipt")
        selection_by_key[key] = receipt
    record_keys: dict[tuple[int, int, str, str], Mapping[str, Any]] = {}
    prediction_groups: dict[tuple[int, int, str, str], list[Mapping[str, Any]]] = {}
    for row in records:
        key = (
            int(row["seed"]),
            int(row["n_components"]),
            str(row["panel_label"]),
            str(row["distance_mode"]),
        )
        if key in record_keys:
            raise ValueError(f"duplicate result record: {key}")
        record_keys[key] = row
    for row in predictions:
        key = (
            int(row["seed"]),
            int(row["n_components"]),
            str(row["panel_label"]),
            str(row["distance_mode"]),
        )
        prediction_groups.setdefault(key, []).append(row)
    if set(record_keys) != set(prediction_groups):
        raise ValueError("prediction groups do not exactly cover records")
    expected_selection_keys = {
        (seed, rank, _panel_label(fraction))
        for seed in config.seeds
        for rank in COMPONENT_GRID
        for fraction in PANEL_FRACTIONS
    }
    if set(selection_by_key) != expected_selection_keys:
        raise ValueError("selection receipts do not cover configured grid")
    for key, record in record_keys.items():
        rows = sorted(
            prediction_groups[key], key=lambda row: int(row["sample_position"])
        )
        if [int(row["sample_position"]) for row in rows] != list(
            range(int(record["test_sample_count"]))
        ):
            raise ValueError(f"sample positions are not an exact cover: {key}")
        truth = np.asarray([row["truth"] for row in rows], dtype=str)
        observed = np.asarray([row["prediction"] for row in rows], dtype=str)
        sample_indices = np.asarray(
            [row["sample_index"] for row in rows], dtype=np.int64
        )
        sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=str)
        margins = np.asarray(
            [row["AMR_EUR_margin_positive_favors_AMR"] for row in rows],
            dtype=float,
        )
        if not np.isfinite(margins).all():
            raise ValueError("non-finite AMR/EUR margin")
        split = split_by_seed[key[0]]
        selection = selection_by_key[(key[0], key[1], key[2])]
        if (
            array_sha256(sample_indices) != record["test_indices_sha256"]
            or record["test_indices_sha256"] != split["test_indices_sha256"]
        ):
            raise ValueError(f"test index identity mismatch: {key}")
        if (
            array_sha256(sample_ids) != record["test_sample_ids_sha256"]
            or record["test_sample_ids_sha256"] != split["test_sample_ids_sha256"]
        ):
            raise ValueError(f"test sample ID identity mismatch: {key}")
        if (
            array_sha256(truth) != record["test_labels_sha256"]
            or record["test_labels_sha256"] != split["test_labels_sha256"]
        ):
            raise ValueError(f"test truth identity mismatch: {key}")
        if (
            int(selection["selected_snp_count"]) != int(record["selected_snp_count"])
            or selection["selected_original_indices_sha256"]
            != record["selected_original_indices_sha256"]
        ):
            raise ValueError(f"selection/record identity mismatch: {key}")
        for row in rows:
            if bool(row["correct"]) != (row["truth"] == row["prediction"]):
                raise ValueError(f"correct flag mismatch: {key}")
            if (
                row["selected_original_indices_sha256"]
                != record["selected_original_indices_sha256"]
                or row["test_indices_sha256"] != record["test_indices_sha256"]
                or row["test_predictions_sha256"] != record["test_predictions_sha256"]
            ):
                raise ValueError(f"prediction-row receipt mismatch: {key}")
        if array_sha256(observed) != record["test_predictions_sha256"]:
            raise ValueError(f"prediction hash mismatch: {key}")
        if array_sha256(margins) != record["AMR_EUR_margin_sha256"]:
            raise ValueError(f"margin hash mismatch: {key}")
        classes = np.asarray(record["class_order"], dtype=str)
        metrics = base._classification_metrics(truth, observed, classes)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            if float(metrics[metric]) != float(record[metric]):
                raise ValueError(f"metric mismatch {metric}: {key}")
        if metrics["confusion_matrix_counts"] != record["confusion_matrix_counts"]:
            raise ValueError(f"confusion mismatch: {key}")
        if metrics["per_population_metrics"] != record["per_population_metrics"]:
            raise ValueError(f"population metric mismatch: {key}")
        singular = np.asarray(record["singular_values"], dtype=float)
        scale = np.asarray(record["distance_scale"], dtype=float)
        if singular.shape != (key[1],) or scale.shape != singular.shape:
            raise ValueError(f"singular/scale shape mismatch: {key}")
        if not np.isfinite(singular).all() or np.any(singular <= 0.0):
            raise ValueError(f"invalid singular values: {key}")
        expected_scale = (
            np.ones(key[1], dtype=float)
            if key[3] == DISTANCE_MODES[0]
            else singular / math.sqrt(float(record["train_sample_count"] - 1))
        )
        if not np.array_equal(scale, expected_scale):
            raise ValueError(f"distance scale is not protocol-defined: {key}")
        if array_sha256(scale) != record["distance_scale_sha256"]:
            raise ValueError(f"distance scale hash mismatch: {key}")
    expected_key_set = {
        (seed, rank, _panel_label(fraction), mode)
        for seed in config.seeds
        for rank in COMPONENT_GRID
        for fraction in PANEL_FRACTIONS
        for mode in DISTANCE_MODES
    }
    if set(record_keys) != expected_key_set:
        raise ValueError("records do not exactly cover seed/rank/panel/distance grid")
    shared_fields = (
        "selected_snp_count",
        "selected_original_indices_sha256",
        "sample_gram_sha256",
        "loadings_prefix_sha256",
        "singular_values",
        "train_indices_sha256",
        "test_indices_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
    )
    for seed in config.seeds:
        for rank in COMPONENT_GRID:
            for fraction in PANEL_FRACTIONS:
                panel = _panel_label(fraction)
                raw = record_keys[(seed, rank, panel, DISTANCE_MODES[0])]
                white = record_keys[(seed, rank, panel, DISTANCE_MODES[1])]
                for field in shared_fields:
                    if raw[field] != white[field]:
                        raise ValueError(
                            "raw/whitened scientific fit differs: "
                            f"{seed}/{rank}/{panel}/{field}"
                        )
                raw_rows = sorted(
                    prediction_groups[(seed, rank, panel, DISTANCE_MODES[0])],
                    key=lambda row: int(row["sample_position"]),
                )
                white_rows = sorted(
                    prediction_groups[(seed, rank, panel, DISTANCE_MODES[1])],
                    key=lambda row: int(row["sample_position"]),
                )
                for raw_row, white_row in zip(raw_rows, white_rows, strict=True):
                    for field in (
                        "sample_position",
                        "sample_index",
                        "sample_id",
                        "truth",
                        "subpopulation",
                        "selected_original_indices_sha256",
                        "test_indices_sha256",
                    ):
                        if raw_row[field] != white_row[field]:
                            raise ValueError(
                                "raw/whitened prediction identity differs: "
                                f"{seed}/{rank}/{panel}/{field}"
                            )
    rebuilt_grid = _grid_summaries(records)
    mismatch = _first_mismatch(
        rebuilt_grid, payload.get("grid_summaries"), "$.grid_summaries"
    )
    if mismatch:
        raise ValueError(mismatch)
    rebuilt_paired = _paired_summaries(records)
    mismatch = _first_mismatch(
        rebuilt_paired, payload.get("paired_summaries"), "$.paired_summaries"
    )
    if mismatch:
        raise ValueError(mismatch)
    rebuilt_ledger = _transition_ledger(predictions)
    mismatch = _first_mismatch(
        rebuilt_ledger,
        payload.get("headline_transition_ledger"),
        "$.headline_transition_ledger",
    )
    if mismatch:
        raise ValueError(mismatch)
    return {
        "valid": True,
        "analysis_sha256": reported_digest,
        "seeds": len(config.seeds),
        "records": len(records),
        "predictions": len(predictions),
        "full_grid_closed": True,
        "paired_fit_identity_verified": True,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fields})
    return buffer.getvalue().encode("utf-8")


def _grid_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    rows: list[dict[str, Any]] = []
    for item in payload["grid_summaries"]:
        row: dict[str, Any] = {
            "n_components": item["n_components"],
            "panel_label": item["panel_label"],
            "requested_panel_fraction": item["requested_panel_fraction"],
            "distance_mode": item["distance_mode"],
            "evaluated_splits": item["evaluated_splits"],
            "selected_snp_count_mean": item["selected_snp_count_mean"],
        }
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            for statistic in ("mean", "sample_std", "minimum", "maximum"):
                row[f"{metric}_{statistic}"] = item[metric][statistic]
        for population in EXPECTED_POPULATIONS:
            for metric in ("recall", "precision", "f1"):
                row[f"{population}_{metric}_mean"] = item["population_metrics"][
                    population
                ][metric]["mean"]
        rows.append(row)
    fields = list(rows[0]) if rows else []
    return _csv_bytes(rows, fields)


def _paired_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    rows: list[dict[str, Any]] = []
    for item in payload["paired_summaries"]:
        row: dict[str, Any] = {
            "n_components": item["n_components"],
            "panel_label": item["panel_label"],
            "requested_panel_fraction": item["requested_panel_fraction"],
            "evaluated_splits": len(item["seeds"]),
        }
        for metric in PAIRED_METRICS:
            result = item["deltas"][metric]
            row[f"{metric}_delta_mean"] = result["candidate_minus_raw_mean"]
            row[f"{metric}_ci_low"] = result["paired_percentile_95_ci"][0]
            row[f"{metric}_ci_high"] = result["paired_percentile_95_ci"][1]
            row[f"{metric}_positive_seeds"] = result["positive_seeds"]
            row[f"{metric}_equal_seeds"] = result["equal_seeds"]
            row[f"{metric}_negative_seeds"] = result["negative_seeds"]
        rows.append(row)
    fields = list(rows[0]) if rows else []
    return _csv_bytes(rows, fields)


def _prediction_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    rows = payload["predictions"]
    fields = (
        "seed",
        "n_components",
        "panel_label",
        "requested_panel_fraction",
        "distance_mode",
        "sample_position",
        "sample_index",
        "sample_id",
        "truth",
        "subpopulation",
        "prediction",
        "correct",
        "AMR_EUR_margin_positive_favors_AMR",
        "selected_original_indices_sha256",
        "test_indices_sha256",
        "test_predictions_sha256",
    )
    return _csv_bytes(rows, fields)


def _summary_lookup(
    payload: Mapping[str, Any], rank: int, panel: str, mode: str
) -> Mapping[str, Any]:
    for row in payload["grid_summaries"]:
        if (
            row["n_components"] == rank
            and row["panel_label"] == panel
            and row["distance_mode"] == mode
        ):
            return row
    raise ValueError("missing grid summary")


def _render_heatmap(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [_panel_label(value) for value in PANEL_FRACTIONS]
    ranks = list(COMPONENT_GRID)
    raw = np.asarray(
        [
            [
                _summary_lookup(payload, rank, panel, DISTANCE_MODES[0])[
                    "population_metrics"
                ]["AMR"]["recall"]["mean"]
                for panel in panels
            ]
            for rank in ranks
        ],
        dtype=float,
    )
    white = np.asarray(
        [
            [
                _summary_lookup(payload, rank, panel, DISTANCE_MODES[1])[
                    "population_metrics"
                ]["AMR"]["recall"]["mean"]
                for panel in panels
            ]
            for rank in ranks
        ],
        dtype=float,
    )
    raw_ba = np.asarray(
        [
            [
                _summary_lookup(payload, rank, panel, DISTANCE_MODES[0])[
                    "balanced_accuracy"
                ]["mean"]
                for panel in panels
            ]
            for rank in ranks
        ],
        dtype=float,
    )
    white_ba = np.asarray(
        [
            [
                _summary_lookup(payload, rank, panel, DISTANCE_MODES[1])[
                    "balanced_accuracy"
                ]["mean"]
                for panel in panels
            ]
            for rank in ranks
        ],
        dtype=float,
    )
    matrices = (raw, white, white - raw, white_ba - raw_ba)
    titles = (
        "Raw centroid: AMR recall",
        "Train-whitened centroid: AMR recall",
        "Whitened minus raw: AMR recall",
        "Whitened minus raw: balanced accuracy",
    )
    cmaps = ("viridis", "viridis", "RdBu_r", "RdBu_r")
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5), constrained_layout=True)
    for axis, matrix, title, cmap in zip(
        axes.ravel(), matrices, titles, cmaps, strict=True
    ):
        if "minus" in title:
            bound = max(0.005, float(np.max(np.abs(matrix))))
            image = axis.imshow(matrix, cmap=cmap, vmin=-bound, vmax=bound)
            fmt = "{:+.3f}"
        else:
            image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
            fmt = "{:.3f}"
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(
                    column,
                    row,
                    fmt.format(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=13,
                    color=(
                        "white"
                        if not ("minus" in title) and matrix[row, column] < 0.55
                        else "black"
                    ),
                )
        axis.set_title(title, fontsize=16)
        axis.set_xticks(range(len(panels)), ["5%", "10%", "25%", "50%", "All"])
        axis.set_yticks(range(len(ranks)), [f"Rank {rank}" for rank in ranks])
        axis.set_xlabel("Split-specific loading panel")
        fig.colorbar(image, ax=axis, shrink=0.82)
    fig.suptitle(
        "Full fixed grid — no cell selected as a winner",
        fontsize=20,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _pooled_confusion(payload: Mapping[str, Any], mode: str) -> NDArray[np.int64]:
    output = np.zeros((5, 5), dtype=np.int64)
    for row in payload["records"]:
        if (
            row["n_components"] == HEADLINE_RANK
            and row["panel_label"] == HEADLINE_PANEL
            and row["distance_mode"] == mode
        ):
            output += np.asarray(row["confusion_matrix_counts"], dtype=np.int64)
    return output


def _render_headline(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    raw = _pooled_confusion(payload, DISTANCE_MODES[0])
    white = _pooled_confusion(payload, DISTANCE_MODES[1])
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2), constrained_layout=True)
    for axis, confusion, title in (
        (axes[0], raw, "Historical raw distance"),
        (axes[1], white, "Train-whitened distance"),
    ):
        normalized = confusion / confusion.sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
        for row in range(5):
            for column in range(5):
                axis.text(
                    column,
                    row,
                    f"{confusion[row, column]}\n{normalized[row, column]:.1%}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if normalized[row, column] > 0.55 else "black",
                )
        axis.set_xticks(range(5), EXPECTED_POPULATIONS)
        axis.set_yticks(range(5), EXPECTED_POPULATIONS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(title, fontsize=16)
        fig.colorbar(image, ax=axis, shrink=0.75)
    ledger = payload["headline_transition_ledger"]
    raw_summary = _summary_lookup(
        payload, HEADLINE_RANK, HEADLINE_PANEL, DISTANCE_MODES[0]
    )
    white_summary = _summary_lookup(
        payload, HEADLINE_RANK, HEADLINE_PANEL, DISTANCE_MODES[1]
    )
    lines = [
        "Fixed companion: Rank 8 / All SNPs",
        "(historical alignment, not winner selection)",
        "",
        (
            f"Balanced accuracy: {raw_summary['balanced_accuracy']['mean']:.4f}"
            f" → {white_summary['balanced_accuracy']['mean']:.4f}"
        ),
        (
            f"Macro-F1: {raw_summary['macro_f1']['mean']:.4f}"
            f" → {white_summary['macro_f1']['mean']:.4f}"
        ),
        (
            "AMR recall: "
            f"{raw_summary['population_metrics']['AMR']['recall']['mean']:.4f}"
            " → "
            f"{white_summary['population_metrics']['AMR']['recall']['mean']:.4f}"
        ),
        (
            "AMR precision: "
            f"{raw_summary['population_metrics']['AMR']['precision']['mean']:.4f}"
            " → "
            f"{white_summary['population_metrics']['AMR']['precision']['mean']:.4f}"
        ),
        (
            "EUR recall: "
            f"{raw_summary['population_metrics']['EUR']['recall']['mean']:.4f}"
            " → "
            f"{white_summary['population_metrics']['EUR']['recall']['mean']:.4f}"
        ),
        "",
        f"AMR→EUR: {ledger['raw_AMR_to_EUR']} → {ledger['whitened_AMR_to_EUR']}",
        f"EUR→AMR: {ledger['raw_EUR_to_AMR']} → {ledger['whitened_EUR_to_AMR']}",
    ]
    axes[2].axis("off")
    axes[2].text(
        0.02,
        0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=14,
        linespacing=1.5,
        family="DejaVu Sans",
    )
    fig.suptitle(
        "AMR readout trade-off on 20 fresh fixed-cohort splits",
        fontsize=20,
    )
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _artifact_receipt(path: Path) -> dict[str, Any]:
    return {"sha256": _file_sha256(path), "bytes": int(path.stat().st_size)}


def write_output_package(
    payload: Mapping[str, Any], output_dir: Path
) -> dict[str, Any]:
    validate_result_payload(payload)
    target = Path(output_dir).resolve()
    if os.path.lexists(target):
        raise FileExistsError(f"output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent))
    )
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / GRID_FILENAME).write_bytes(_grid_csv_bytes(payload))
        (temporary / PAIRED_FILENAME).write_bytes(_paired_csv_bytes(payload))
        (temporary / PREDICTION_FILENAME).write_bytes(_prediction_csv_bytes(payload))
        _render_heatmap(payload, temporary / HEATMAP_FILENAME)
        _render_headline(payload, temporary / HEADLINE_FILENAME)
        artifacts = {
            name: _artifact_receipt(temporary / name) for name in ARTIFACT_FILENAMES
        }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "analysis_sha256": payload["analysis_sha256"],
            "manifest_written_last": True,
            "artifacts": artifacts,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        validate_output_package(temporary)
        os.rename(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_output_package(target)


def validate_output_package(
    output_dir: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    expected_files = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    if not root.is_dir() or root.is_symlink():
        raise ValueError("output package must be a regular directory")
    actual = {path.name for path in root.iterdir()}
    if actual != expected_files:
        raise ValueError("output package file set mismatch")
    for path in root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"non-regular package artifact: {path.name}")
    manifest = _strict_json_load(root / MANIFEST_FILENAME)
    if set(manifest) != {
        "schema_version",
        "status",
        "analysis_sha256",
        "manifest_written_last",
        "artifacts",
        "package_anchor_sha256",
    }:
        raise ValueError("manifest schema mismatch")
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
        if manifest["artifacts"].get(name) != _artifact_receipt(root / name):
            raise ValueError(f"artifact receipt mismatch: {name}")
    payload = _strict_json_load(root / RESULT_FILENAME)
    result_validation = validate_result_payload(payload)
    if manifest["analysis_sha256"] != payload["analysis_sha256"]:
        raise ValueError("manifest/result analysis mismatch")
    if (
        expected_analysis_sha256 is not None
        and payload["analysis_sha256"] != expected_analysis_sha256
    ):
        raise ValueError("external analysis anchor mismatch")
    if (root / GRID_FILENAME).read_bytes() != _grid_csv_bytes(payload):
        raise ValueError("grid CSV mismatch")
    if (root / PAIRED_FILENAME).read_bytes() != _paired_csv_bytes(payload):
        raise ValueError("paired CSV mismatch")
    if (root / PREDICTION_FILENAME).read_bytes() != _prediction_csv_bytes(payload):
        raise ValueError("prediction CSV mismatch")
    try:
        from PIL import Image

        for name in (HEATMAP_FILENAME, HEADLINE_FILENAME):
            with Image.open(root / name) as image:
                image.verify()
            with Image.open(root / name) as image:
                if image.width < 2000 or image.height < 1000:
                    raise ValueError(f"figure resolution is too small: {name}")
    except ImportError as error:
        raise RuntimeError("Pillow is required for figure validation") from error
    source_verified = False
    if repo_root is not None:
        current = _source_identity(Path(repo_root))
        reported = payload["source_identity"]
        if (
            current["files"] != reported["files"]
            or current["roles"] != reported["roles"]
            or current["extension_source_sha256"] != reported["extension_source_sha256"]
        ):
            raise ValueError("current source bytes differ from result source identity")
        source_verified = True
    return {
        **result_validation,
        "package_self_consistency_verified": True,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "source_bytes_verified": source_verified,
    }


def run_formal(
    *,
    genotypes: Path,
    source_manifest: Path,
    subpopulation_manifest: Path,
    output_dir: Path,
    repo_root: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    source_identity = _source_identity(repo_root)
    with allauto.load_formal_input(
        genotypes,
        source_manifest,
        expected_source_manifest_sha256=FROZEN_INPUT_MANIFEST_SHA256,
        scratch_root=scratch_root,
    ) as (
        data,
        input_identity,
    ):
        subpopulations, subpopulation_identity = _load_subpopulations(
            subpopulation_manifest,
            np.asarray(data.sample_ids, dtype=str),
            np.asarray(data.population_labels, dtype=str),
        )
        payload = evaluate(
            data,
            WhiteningConfig(),
            input_identity=input_identity,
            source_identity=source_identity,
            subpopulations=subpopulations,
            subpopulation_identity=subpopulation_identity,
        )
        return write_output_package(payload, output_dir)


def trusted_replay(
    output_dir: Path,
    *,
    genotypes: Path,
    source_manifest: Path,
    subpopulation_manifest: Path,
    repo_root: Path,
    expected_analysis_sha256: str,
    expected_package_anchor_sha256: str,
    scratch_root: Path,
) -> dict[str, Any]:
    initial = validate_output_package(
        output_dir,
        repo_root=repo_root,
        expected_analysis_sha256=expected_analysis_sha256,
        expected_package_anchor_sha256=expected_package_anchor_sha256,
    )
    observed = _strict_json_load(Path(output_dir) / RESULT_FILENAME)
    config_payload = observed["config"]
    config = WhiteningConfig(
        seeds=tuple(config_payload["seeds"]),
        component_grid=tuple(config_payload["component_grid"]),
        panel_fractions=tuple(config_payload["panel_fractions"]),
        distance_modes=tuple(config_payload["distance_modes"]),
        test_fraction=config_payload["test_fraction"],
        snp_chunk_size=config_payload["snp_chunk_size"],
        analysis_mode=config_payload["analysis_mode"],
        analysis_status=config_payload["analysis_status"],
    )
    with allauto.load_formal_input(
        genotypes,
        source_manifest,
        expected_source_manifest_sha256=FROZEN_INPUT_MANIFEST_SHA256,
        scratch_root=scratch_root,
    ) as (
        data,
        input_identity,
    ):
        subpopulations, subpopulation_identity = _load_subpopulations(
            subpopulation_manifest,
            np.asarray(data.sample_ids, dtype=str),
            np.asarray(data.population_labels, dtype=str),
        )
        replay = evaluate(
            data,
            config,
            input_identity=input_identity,
            source_identity=_source_identity(repo_root),
            subpopulations=subpopulations,
            subpopulation_identity=subpopulation_identity,
        )
    scientific_fields = (
        "config",
        "input_identity",
        "subpopulation_identity",
        "split_receipts",
        "selection_receipts",
        "records",
        "predictions",
        "grid_summaries",
        "paired_summaries",
        "headline_transition_ledger",
    )
    for field in scientific_fields:
        mismatch = _first_mismatch(replay[field], observed[field], f"$.{field}")
        if mismatch:
            raise ValueError(f"trusted replay mismatch: {mismatch}")
    return {
        **initial,
        "trusted_scientific_replay_verified": True,
        "trusted_replay_fields": list(scientific_fields),
    }


def verify_legacy_raw_replay(
    *,
    genotypes: Path,
    source_manifest: Path,
    subpopulation_manifest: Path,
    legacy_package: Path,
    repo_root: Path,
    scratch_root: Path,
) -> dict[str, Any]:
    allauto._require_formal_blas_thread_environment()
    legacy_validation = allauto.validate_output_package(
        legacy_package,
        repo_root=repo_root,
        expected_analysis_sha256=FROZEN_LEGACY_ANALYSIS_SHA256,
        expected_package_anchor_sha256=FROZEN_LEGACY_PACKAGE_ANCHOR_SHA256,
    )
    legacy = allauto._strict_json_load(Path(legacy_package) / allauto.RESULT_FILENAME)
    legacy_records = {
        (
            int(row["seed"]),
            int(row["n_components"]),
            str(row["panel_label"]),
        ): row
        for row in legacy["records"]
        if int(row["seed"]) == 2025
    }
    if len(legacy_records) != 15:
        raise ValueError("legacy package does not contain the complete seed-2025 grid")
    with allauto.load_formal_input(
        genotypes,
        source_manifest,
        expected_source_manifest_sha256=FROZEN_INPUT_MANIFEST_SHA256,
        scratch_root=scratch_root,
    ) as (
        data,
        input_identity,
    ):
        subpopulations, subpopulation_identity = _load_subpopulations(
            subpopulation_manifest,
            np.asarray(data.sample_ids, dtype=str),
            np.asarray(data.population_labels, dtype=str),
        )
        replay = evaluate(
            data,
            WhiteningConfig(
                seeds=(2025,),
                analysis_mode="legacy_engineering_replay",
                analysis_status="engineering_baseline_equivalence_only",
            ),
            input_identity=input_identity,
            source_identity=_source_identity(repo_root),
            subpopulations=subpopulations,
            subpopulation_identity=subpopulation_identity,
        )
    raw_records = {
        (
            int(row["seed"]),
            int(row["n_components"]),
            str(row["panel_label"]),
        ): row
        for row in replay["records"]
        if row["distance_mode"] == DISTANCE_MODES[0]
    }
    if set(raw_records) != set(legacy_records):
        raise ValueError("raw replay and legacy grid keys differ")
    checked: list[dict[str, Any]] = []
    for key in sorted(raw_records):
        raw = raw_records[key]
        old = legacy_records[key]
        for field in (
            "test_predictions_sha256",
            "balanced_accuracy",
            "macro_f1",
            "confusion_matrix_counts",
            "selected_original_indices_sha256",
        ):
            if raw[field] != old[field]:
                raise ValueError(f"legacy raw replay differs at {key}/{field}")
        checked.append(
            {
                "seed": key[0],
                "n_components": key[1],
                "panel_label": key[2],
                "test_predictions_sha256": raw["test_predictions_sha256"],
            }
        )
    return {
        "valid": True,
        "legacy_package_self_consistency_verified": legacy_validation[
            "package_self_consistency_verified"
        ],
        "seed": 2025,
        "raw_grid_cells_exactly_reproduced": len(checked),
        "prediction_and_metric_equivalence": True,
        "checked": checked,
    }


def run_tiny_smoke(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    data = allauto._synthetic_smoke_data()
    payload = evaluate(
        data,
        WhiteningConfig(
            seeds=(73,),
            analysis_mode="synthetic_smoke",
            analysis_status="engineering_smoke_only",
            snp_chunk_size=512,
        ),
        input_identity=allauto._synthetic_input_identity(data),
        source_identity=_source_identity(repo_root),
    )
    return write_output_package(payload, output_dir)


def _add_formal_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--genotypes", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--subpopulation-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--repo-root", type=Path, required=True)
    run = commands.add_parser("run")
    _add_formal_inputs(run)
    run.add_argument("--output-dir", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--expected-analysis-sha256", required=True)
    validate.add_argument("--expected-package-anchor-sha256", required=True)
    validate.add_argument("--trusted-replay", action="store_true")
    validate.add_argument("--genotypes", type=Path)
    validate.add_argument("--source-manifest", type=Path)
    validate.add_argument("--subpopulation-manifest", type=Path)
    validate.add_argument("--scratch-root", type=Path)
    legacy = commands.add_parser("verify-legacy-raw")
    _add_formal_inputs(legacy)
    legacy.add_argument("--legacy-package", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "smoke":
        result = run_tiny_smoke(args.output_dir, args.repo_root)
    elif args.command == "run":
        result = run_formal(
            genotypes=args.genotypes,
            source_manifest=args.source_manifest,
            subpopulation_manifest=args.subpopulation_manifest,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
            scratch_root=args.scratch_root,
        )
    elif args.command == "validate":
        if args.trusted_replay:
            if not all(
                value is not None
                for value in (
                    args.genotypes,
                    args.source_manifest,
                    args.subpopulation_manifest,
                    args.scratch_root,
                )
            ):
                raise ValueError(
                    "trusted replay requires genotype, manifest, subpopulation, "
                    "and scratch paths"
                )
            result = trusted_replay(
                args.output_dir,
                genotypes=args.genotypes,
                source_manifest=args.source_manifest,
                subpopulation_manifest=args.subpopulation_manifest,
                repo_root=args.repo_root,
                expected_analysis_sha256=args.expected_analysis_sha256,
                expected_package_anchor_sha256=args.expected_package_anchor_sha256,
                scratch_root=args.scratch_root,
            )
        else:
            result = validate_output_package(
                args.output_dir,
                repo_root=args.repo_root,
                expected_analysis_sha256=args.expected_analysis_sha256,
                expected_package_anchor_sha256=args.expected_package_anchor_sha256,
            )
    elif args.command == "verify-legacy-raw":
        result = verify_legacy_raw_replay(
            genotypes=args.genotypes,
            source_manifest=args.source_manifest,
            subpopulation_manifest=args.subpopulation_manifest,
            legacy_package=args.legacy_package,
            repo_root=args.repo_root,
            scratch_root=args.scratch_root,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

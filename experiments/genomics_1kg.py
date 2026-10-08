#!/usr/bin/env python3
"""Leakage-aware SPCA runner for 1000 Genomes-style genotype matrices.

The canonical internal orientation is always ``samples x SNPs``. Bootstrap
stability resamples full-cohort raw rows and refits preprocessing within every
resample. When population metadata is provided, the held-out classification
branch splits samples *before* fitting
imputation, scaling, monomorphic-SNP filtering, SPCA, and class centroids.
When ``--alpha-candidates`` is supplied, alpha is selected on a second,
training-only validation split and the outer test partition is evaluated once.

Supported genotype inputs are numeric ``.npy``, ``.npz`` (``X`` or
``genotypes`` key), and delimited text. Across all formats, the explicit
``missing_value`` sentinel (default ``-9``) is converted to NaN and infinities
are rejected. Convert VCF/PLINK data to one of these formats with a verified
variant-QC pipeline before using this runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import hypergeom

from rspca.datasets import array_sha256, make_seed_manifest, runtime_versions
from rspca.metrics import (
    component_correlation,
    explained_variance_ratio,
    relative_reconstruction_error,
    sparsity_ratio,
    subspace_error,
)


@dataclass(frozen=True)
class GenotypeDataset:
    genotypes: NDArray[np.float64]
    sample_ids: NDArray[np.str_]
    snp_ids: NDArray[np.str_]


@dataclass(frozen=True)
class GenotypePreprocessor:
    means: NDArray[np.float64]
    scales: NDArray[np.float64]
    retained_features: NDArray[np.bool_]
    standardize: bool


@dataclass(frozen=True)
class GenomicsConfig:
    method: str = "pca"
    rank: int = 5
    alpha: float = 0.1
    alpha_candidates: tuple[float, ...] = ()
    validation_fraction: float = 0.2
    max_iter: int = 500
    tol: float = 1e-6
    bootstrap_repeats: int = 20
    selection_threshold: float = 1e-8
    test_fraction: float = 0.2
    base_seed: int = 2025
    standardize: bool = True
    oversample: int = 10
    n_power_iter: int = 2
    randomized_full_gradient: bool = False
    randomized_finalize_full_iters: int = 5
    missing_value: float | None = -9.0


@dataclass(frozen=True)
class _PCAFitResult:
    """Minimal closed-form result matching the solver diagnostics contract."""

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    stationarity: tuple[float, ...]
    converged: bool
    status: str
    n_iter: int


def _validate_config(config: GenomicsConfig) -> None:
    if config.method not in {"pca", "vp", "randomized", "robust"}:
        raise ValueError("method must be pca, vp, randomized, or robust")
    integer_fields = {
        "rank": config.rank,
        "max_iter": config.max_iter,
        "bootstrap_repeats": config.bootstrap_repeats,
        "base_seed": config.base_seed,
        "oversample": config.oversample,
        "n_power_iter": config.n_power_iter,
        "randomized_finalize_full_iters": config.randomized_finalize_full_iters,
    }
    for name, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if config.rank <= 0 or config.max_iter <= 0:
        raise ValueError("rank and max_iter must be positive")
    if config.bootstrap_repeats < 0 or config.base_seed < 0:
        raise ValueError("bootstrap_repeats and base_seed must be non-negative")
    if (
        min(
            config.oversample,
            config.n_power_iter,
            config.randomized_finalize_full_iters,
        )
        < 0
    ):
        raise ValueError(
            "randomized sketch and refinement controls must be non-negative"
        )
    scalar_fields = {
        "alpha": config.alpha,
        "tol": config.tol,
        "selection_threshold": config.selection_threshold,
        "test_fraction": config.test_fraction,
        "validation_fraction": config.validation_fraction,
    }
    for name, value in scalar_fields.items():
        try:
            finite = bool(np.isfinite(float(value)))
        except (TypeError, ValueError):
            finite = False
        if isinstance(value, bool) or not finite:
            raise ValueError(f"{name} must be a finite real scalar")
    if config.alpha < 0.0 or config.selection_threshold < 0.0:
        raise ValueError("alpha and selection_threshold must be non-negative")
    if config.tol <= 0.0:
        raise ValueError("tol must be positive")
    if not 0.0 < config.test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if not 0.0 < config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    if not isinstance(config.standardize, (bool, np.bool_)):
        raise TypeError("standardize must be boolean")
    if not isinstance(config.randomized_full_gradient, (bool, np.bool_)):
        raise TypeError("randomized_full_gradient must be boolean")
    candidates = tuple(config.alpha_candidates)
    try:
        numeric_candidates = tuple(float(alpha) for alpha in candidates)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "alpha_candidates must contain finite non-negative scalars"
        ) from error
    if any(isinstance(alpha, bool) for alpha in candidates) or any(
        not np.isfinite(alpha) or alpha < 0.0 for alpha in numeric_candidates
    ):
        raise ValueError("alpha_candidates must contain finite non-negative scalars")
    if len(set(numeric_candidates)) != len(candidates):
        raise ValueError("alpha_candidates must not contain duplicates")
    if candidates and config.method == "pca":
        raise ValueError("alpha_candidates are not meaningful for closed-form PCA")
    if config.missing_value is not None and not np.isfinite(config.missing_value):
        raise ValueError("missing_value must be finite or None")


def _parse_numeric(value: str) -> float:
    normalized = value.strip()
    if normalized.upper() in {"", ".", "NA", "NAN", "NULL"}:
        return float("nan")
    return float(normalized)


def _normalize_missing_values(
    matrix: NDArray[np.float64], missing_value: float | None
) -> NDArray[np.float64]:
    normalized = np.asarray(matrix, dtype=float).copy()
    if missing_value is not None:
        sentinel = float(missing_value)
        if not np.isfinite(sentinel):
            raise ValueError("missing_value must be finite or None")
        normalized[normalized == sentinel] = np.nan
    if np.any(np.isinf(normalized)):
        raise ValueError("genotype values may use NaN for missingness but not infinity")
    return normalized


def _is_numeric_or_missing(value: str) -> bool:
    try:
        _parse_numeric(value)
    except ValueError:
        return False
    return True


def _delimited_matrix(
    path: Path,
    *,
    delimiter: str,
    has_header: bool,
    row_id_column: str | int | None,
) -> tuple[NDArray[np.float64], NDArray[np.str_], NDArray[np.str_]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    if not rows:
        raise ValueError(f"genotype file is empty: {path}")

    header = rows.pop(0) if has_header else None
    if not rows:
        raise ValueError("genotype file has no data rows")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError("genotype rows must all have the same non-zero width")
    if header is not None and len(header) != width:
        raise ValueError("header width does not match genotype rows")

    if isinstance(row_id_column, str):
        try:
            row_id_index = int(row_id_column)
        except ValueError:
            if header is None:
                raise ValueError("a named row ID column requires a header") from None
            if row_id_column not in header:
                raise ValueError(f"row ID column {row_id_column!r} is absent")
            row_id_index = header.index(row_id_column)
    elif row_id_column is not None:
        row_id_index = int(row_id_column)
    else:
        canonical_id_headers = {
            "fid",
            "iid",
            "id",
            "sample",
            "sample_id",
            "sampleid",
        }
        first_header = header[0].strip().lower() if header is not None else None
        row_id_index = (
            0
            if first_header in canonical_id_headers
            or not _is_numeric_or_missing(rows[0][0])
            else None
        )
    if row_id_index is not None and not 0 <= row_id_index < width:
        raise ValueError("row ID column index is outside the input width")

    numeric_indices = [index for index in range(width) if index != row_id_index]
    matrix = np.asarray(
        [[_parse_numeric(row[index]) for index in numeric_indices] for row in rows],
        dtype=float,
    )
    row_ids = np.asarray(
        (
            [row[row_id_index] for row in rows]
            if row_id_index is not None
            else [f"row_{index:06d}" for index in range(matrix.shape[0])]
        ),
        dtype=str,
    )
    column_ids = np.asarray(
        (
            [header[index] for index in numeric_indices]
            if header is not None
            else [f"column_{index:06d}" for index in range(matrix.shape[1])]
        ),
        dtype=str,
    )
    return matrix, row_ids, column_ids


def load_genotypes(
    path: Path | str,
    *,
    orientation: str = "samples-by-snps",
    delimiter: str | None = None,
    has_header: bool = True,
    row_id_column: str | int | None = None,
    missing_value: float | None = -9.0,
) -> GenotypeDataset:
    """Load genotypes and normalize orientation to ``samples x SNPs``."""

    path = Path(path)
    if orientation not in {"samples-by-snps", "snps-by-samples"}:
        raise ValueError("orientation must be samples-by-snps or snps-by-samples")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        matrix = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        if matrix.ndim != 2:
            raise ValueError("the .npy genotype array must be two-dimensional")
        row_ids = np.asarray([f"row_{index:06d}" for index in range(matrix.shape[0])])
        column_ids = np.asarray(
            [f"column_{index:06d}" for index in range(matrix.shape[1])]
        )
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            key = (
                "X"
                if "X" in archive
                else "genotypes" if "genotypes" in archive else None
            )
            if key is None:
                raise ValueError(".npz input must contain an X or genotypes array")
            matrix = np.asarray(archive[key], dtype=float)
            if matrix.ndim != 2:
                raise ValueError("the genotype array must be two-dimensional")
            row_ids = np.asarray(
                (
                    archive["row_ids"]
                    if "row_ids" in archive
                    else [f"row_{index:06d}" for index in range(matrix.shape[0])]
                ),
                dtype=str,
            )
            column_ids = np.asarray(
                (
                    archive["column_ids"]
                    if "column_ids" in archive
                    else [f"column_{index:06d}" for index in range(matrix.shape[1])]
                ),
                dtype=str,
            )
    else:
        actual_delimiter = delimiter or ("\t" if suffix in {".tsv", ".tab"} else ",")
        matrix, row_ids, column_ids = _delimited_matrix(
            path,
            delimiter=actual_delimiter,
            has_header=has_header,
            row_id_column=row_id_column,
        )

    if matrix.shape != (row_ids.size, column_ids.size):
        raise ValueError("row/column identifiers do not match the matrix dimensions")
    matrix = _normalize_missing_values(matrix, missing_value)
    if orientation == "samples-by-snps":
        genotypes, sample_ids, snp_ids = matrix, row_ids, column_ids
    else:
        genotypes, sample_ids, snp_ids = matrix.T, column_ids, row_ids
    if genotypes.shape[0] < 2 or genotypes.shape[1] < 2:
        raise ValueError("genotype data must contain at least two samples and two SNPs")
    if np.unique(sample_ids).size != sample_ids.size:
        raise ValueError("sample identifiers must be unique")
    if np.unique(snp_ids).size != snp_ids.size:
        raise ValueError("SNP identifiers must be unique")
    return GenotypeDataset(
        genotypes=np.asarray(genotypes, dtype=float),
        sample_ids=np.asarray(sample_ids, dtype=str),
        snp_ids=np.asarray(snp_ids, dtype=str),
    )


def fit_genotype_preprocessor(
    train_genotypes: NDArray[np.float64],
    *,
    standardize: bool = True,
    minimum_scale: float = 1e-12,
) -> GenotypePreprocessor:
    """Fit imputation/scaling/variant filtering on training samples only."""

    matrix = np.asarray(train_genotypes, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("train_genotypes must be two-dimensional")
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    sums = np.where(finite, matrix, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(matrix.shape[1]), where=counts > 0)
    imputed = np.where(finite, matrix, means)
    empirical_scales = np.sqrt(np.mean((imputed - means) ** 2, axis=0))
    retained = (
        (counts > 0)
        & np.isfinite(empirical_scales)
        & (empirical_scales > minimum_scale)
    )
    if not np.any(retained):
        raise ValueError("no non-monomorphic SNPs remain after training-only filtering")
    scales = empirical_scales if standardize else np.ones_like(empirical_scales)
    scales = np.where(retained, scales, 1.0)
    return GenotypePreprocessor(
        means=means.astype(float),
        scales=scales.astype(float),
        retained_features=retained.astype(bool),
        standardize=standardize,
    )


def transform_genotypes(
    genotypes: NDArray[np.float64], preprocessor: GenotypePreprocessor
) -> NDArray[np.float64]:
    """Apply a previously fitted preprocessor without recomputing statistics."""

    matrix = np.asarray(genotypes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != preprocessor.means.size:
        raise ValueError("genotypes do not match the fitted preprocessor width")
    imputed = np.where(np.isfinite(matrix), matrix, preprocessor.means)
    centered = imputed - preprocessor.means
    transformed = centered[:, preprocessor.retained_features]
    transformed /= preprocessor.scales[preprocessor.retained_features]
    return transformed


def stratified_train_test_split(
    labels: Sequence[str], test_fraction: float, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return deterministic stratified indices with every class in both sets."""

    labels_array = np.asarray(labels, dtype=str)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if np.unique(labels_array).size < 2:
        raise ValueError(
            "held-out population evaluation requires at least two populations"
        )
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == label)
        if indices.size < 2:
            raise ValueError(f"population {label!r} needs at least two samples")
        shuffled = rng.permutation(indices)
        n_test = min(indices.size - 1, max(1, int(round(test_fraction * indices.size))))
        test.extend(shuffled[:n_test].tolist())
        train.extend(shuffled[n_test:].tolist())
    return rng.permutation(train).astype(np.int64), rng.permutation(test).astype(
        np.int64
    )


def _fit_solver(
    matrix: NDArray[np.float64],
    config: GenomicsConfig,
    seed: int,
    *,
    randomized_workspace: Any | None = None,
) -> Any:
    # Canonical package entry points only; import failures are intentionally loud.
    if config.method == "pca":
        del seed
        _, _, right_transpose = np.linalg.svd(matrix, full_matrices=False)
        loadings = np.ascontiguousarray(right_transpose[: config.rank].T)
        return _PCAFitResult(
            A=loadings.copy(),
            B=loadings,
            stationarity=(0.0,),
            converged=True,
            status="closed-form PCA",
            n_iter=1,
        )
    if config.method == "vp":
        from rspca.vp import fit_vp_spca

        return fit_vp_spca(
            matrix,
            config.rank,
            alpha=config.alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            backtracking=True,
            verbose=False,
        )
    if config.method == "randomized":
        from rspca.randomized import (
            RandomizedVPSPCAWorkspace,
            fit_randomized_vp_spca,
            fit_randomized_vp_spca_from_workspace,
        )

        if randomized_workspace is not None:
            if not isinstance(randomized_workspace, RandomizedVPSPCAWorkspace):
                raise TypeError(
                    "randomized_workspace must be a RandomizedVPSPCAWorkspace"
                )
            if randomized_workspace.X is not matrix:
                raise ValueError(
                    "randomized_workspace must be prepared from this exact matrix"
                )
            return fit_randomized_vp_spca_from_workspace(
                randomized_workspace,
                alpha=config.alpha,
                max_iter=config.max_iter,
                tol=config.tol,
                use_full_gradient=config.randomized_full_gradient,
                finalize_full_iters=config.randomized_finalize_full_iters,
                verbose=False,
            )

        return fit_randomized_vp_spca(
            matrix,
            config.rank,
            alpha=config.alpha,
            oversample=config.oversample,
            n_power_iter=config.n_power_iter,
            max_iter=config.max_iter,
            tol=config.tol,
            use_full_gradient=config.randomized_full_gradient,
            finalize_full_iters=config.randomized_finalize_full_iters,
            random_state=seed,
            verbose=False,
        )
    if config.method == "robust":
        from rspca.robust import fit_robust_vp_spca

        return fit_robust_vp_spca(
            matrix,
            config.rank,
            alpha=config.alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=False,
        )
    raise ValueError("method must be pca, vp, randomized, or robust")


def _embed_loadings_in_original_snp_coordinates(
    loadings: NDArray[np.float64],
    retained_features: NDArray[np.bool_],
    scales: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map standardized-fit loadings to raw-dosage SNP coefficients."""

    retained = np.asarray(retained_features, dtype=bool)
    estimate = np.asarray(loadings, dtype=float)
    scale = np.asarray(scales, dtype=float)
    if estimate.ndim != 2 or estimate.shape[0] != int(np.count_nonzero(retained)):
        raise ValueError("loadings do not match the preprocessor's retained SNPs")
    if scale.shape != retained.shape or np.any(scale[retained] <= 0.0):
        raise ValueError("preprocessor scales must be positive and match SNP width")
    embedded = np.zeros((retained.size, estimate.shape[1]), dtype=float)
    embedded[retained] = estimate / scale[retained, None]
    return embedded


def _embed_support_in_original_snp_coordinates(
    loadings: NDArray[np.float64],
    retained_features: NDArray[np.bool_],
    threshold: float,
) -> NDArray[np.bool_]:
    retained = np.asarray(retained_features, dtype=bool)
    estimate = np.asarray(loadings, dtype=float)
    if estimate.ndim != 2 or estimate.shape[0] != int(np.count_nonzero(retained)):
        raise ValueError("loadings do not match the preprocessor's retained SNPs")
    support = np.zeros(retained.size, dtype=bool)
    support[retained] = np.any(np.abs(estimate) > threshold, axis=1)
    return support


def _fit_diagnostics(result: Any) -> dict[str, Any]:
    stationarity = list(result.stationarity)
    return {
        "converged": bool(result.converged),
        "status": str(result.status),
        "n_iter": int(result.n_iter),
        "final_stationarity": (float(stationarity[-1]) if stationarity else None),
        "full_diagnostic_step_size": (
            float(result.full_diagnostic_step_size)
            if hasattr(result, "full_diagnostic_step_size")
            else None
        ),
        "full_diagnostic_step_policy": (
            str(result.full_diagnostic_step_policy)
            if hasattr(result, "full_diagnostic_step_policy")
            else None
        ),
    }


def _support_jaccard(first: NDArray[np.bool_], second: NDArray[np.bool_]) -> float:
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def _fixed_support_expected_jaccard(
    universe_size: int, first_size: int, second_size: int
) -> float:
    """Exact E[Jaccard] for independent uniform supports of fixed sizes."""

    if universe_size <= 0:
        raise ValueError("universe_size must be positive")
    if not 0 <= first_size <= universe_size or not 0 <= second_size <= universe_size:
        raise ValueError("support sizes must lie between zero and universe_size")
    if first_size == 0 and second_size == 0:
        return 1.0
    lower = max(0, first_size + second_size - universe_size)
    upper = min(first_size, second_size)
    expected = 0.0
    probability_mass = 0.0
    # Chunking bounds memory for large SNP universes while summing the complete
    # hypergeometric support (no Bernoulli-density approximation is used).
    for start in range(lower, upper + 1, 100_000):
        intersection = np.arange(start, min(start + 100_000, upper + 1))
        probability = hypergeom.pmf(
            intersection,
            universe_size,
            first_size,
            second_size,
        )
        union = first_size + second_size - intersection
        expected += float(np.sum(probability * intersection / union))
        probability_mass += float(np.sum(probability))
    if probability_mass <= 0.0 or not np.isfinite(probability_mass):
        raise RuntimeError("failed to evaluate the fixed-support hypergeometric model")
    return float(expected / probability_mass)


def support_overlap_diagnostics(
    base_support: NDArray[np.bool_], bootstrap_support: NDArray[np.bool_]
) -> dict[str, float | bool | str | int]:
    """Report raw and chance-adjusted overlap for two feature supports.

    Conditional on the two observed support sizes, the chance model samples
    each support uniformly and independently from the same finite SNP universe.
    The expected Jaccard is summed over the exact hypergeometric intersection
    distribution. Empty and near-dense supports are explicitly marked
    degenerate because raw overlap is then not meaningful stability evidence.
    """

    base = np.asarray(base_support, dtype=bool).reshape(-1)
    bootstrap = np.asarray(bootstrap_support, dtype=bool).reshape(-1)
    if base.shape != bootstrap.shape or base.size == 0:
        raise ValueError("support arrays must have the same non-zero size")
    base_size = int(np.count_nonzero(base))
    bootstrap_size = int(np.count_nonzero(bootstrap))
    base_fraction = float(np.mean(base))
    bootstrap_fraction = float(np.mean(bootstrap))
    expected = _fixed_support_expected_jaccard(
        base.size,
        base_size,
        bootstrap_size,
    )
    observed = _support_jaccard(base, bootstrap)
    adjusted = (
        (observed - expected) / (1.0 - expected)
        if expected < 1.0 - np.finfo(float).eps
        else 0.0
    )
    degeneracy_reasons: list[str] = []
    if base_size == 0 or bootstrap_size == 0:
        degeneracy_reasons.append("empty_support")
    if base_fraction >= 0.95 or bootstrap_fraction >= 0.95:
        degeneracy_reasons.append("near_dense_support")
    return {
        "support_jaccard": observed,
        "support_universe_size": int(base.size),
        "base_selected_count": base_size,
        "bootstrap_selected_count": bootstrap_size,
        "base_selected_fraction": base_fraction,
        "bootstrap_selected_fraction": bootstrap_fraction,
        "fixed_support_expected_jaccard": float(expected),
        "chance_model": "independent_uniform_fixed_size_supports_hypergeometric",
        "chance_adjusted_jaccard": float(adjusted),
        "support_degenerate": bool(degeneracy_reasons),
        "support_degeneracy_reason": (
            ",".join(degeneracy_reasons) if degeneracy_reasons else "none"
        ),
    }


def _classification_metrics(
    truth: NDArray[np.str_], prediction: NDArray[np.str_]
) -> dict[str, float]:
    diagnostics = _classification_diagnostics(truth, prediction)
    return {
        "accuracy": float(diagnostics["accuracy"]),
        "balanced_accuracy": float(diagnostics["balanced_accuracy"]),
        "macro_f1": float(diagnostics["macro_f1"]),
    }


def _classification_diagnostics(
    truth: NDArray[np.str_],
    prediction: NDArray[np.str_],
    *,
    class_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return exact confusion and per-population metrics.

    Rows in the confusion matrix are true metadata population labels and
    columns are predicted labels.  This deliberately describes population
    classification; it must not be interpreted as race inference.
    """

    truth_array = np.asarray(truth, dtype=str)
    prediction_array = np.asarray(prediction, dtype=str)
    if (
        truth_array.ndim != 1
        or prediction_array.ndim != 1
        or truth_array.shape != prediction_array.shape
        or truth_array.size == 0
    ):
        raise ValueError(
            "truth and prediction must be non-empty one-dimensional arrays "
            "with identical shapes"
        )
    observed = np.unique(np.concatenate((truth_array, prediction_array)))
    classes = (
        np.asarray(class_order, dtype=str) if class_order is not None else observed
    )
    if (
        classes.ndim != 1
        or classes.size == 0
        or np.unique(classes).size != classes.size
    ):
        raise ValueError("class_order must contain unique population labels")
    if not set(observed.tolist()).issubset(set(classes.tolist())):
        raise ValueError("class_order does not cover every observed population label")

    class_index = {label: index for index, label in enumerate(classes.tolist())}
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    for true_label, predicted_label in zip(
        truth_array.tolist(), prediction_array.tolist(), strict=True
    ):
        confusion[class_index[true_label], class_index[predicted_label]] += 1

    supports = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    if np.any(supports == 0):
        raise ValueError("every class_order population needs held-out support")
    normalized = confusion / supports[:, None]
    per_population: list[dict[str, Any]] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for index, label in enumerate(classes.tolist()):
        true_positive = int(confusion[index, index])
        support = int(supports[index])
        predicted_count = int(predicted_counts[index])
        recall = float(true_positive / support)
        precision = float(true_positive / predicted_count) if predicted_count else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        recalls.append(recall)
        f1_scores.append(f1)
        per_population.append(
            {
                "population": str(label),
                "support": support,
                "predicted_count": predicted_count,
                "true_positive": true_positive,
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    return {
        "accuracy": float(np.trace(confusion) / truth_array.size),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_scores)),
        "class_order": classes.astype(str).tolist(),
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_true_normalized": normalized.tolist(),
        "per_population_metrics": per_population,
    }


def _nearest_centroid_predictions(
    fit_scores: NDArray[np.float64],
    fit_labels: NDArray[np.str_],
    evaluation_scores: NDArray[np.float64],
) -> tuple[NDArray[np.str_], NDArray[np.str_], NDArray[np.float64]]:
    classes = np.unique(fit_labels)
    centroids = np.stack(
        [fit_scores[fit_labels == label].mean(axis=0) for label in classes], axis=0
    )
    distances = np.sum(
        (evaluation_scores[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return classes[np.argmin(distances, axis=1)], classes, centroids


def _derived_seed(parent_seed: int, role: int) -> int:
    sequence = np.random.SeedSequence([int(parent_seed), int(role)])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _select_alpha_on_training_partition(
    raw_genotypes: NDArray[np.float64],
    labels: NDArray[np.str_],
    outer_train_indices: NDArray[np.int64],
    config: GenomicsConfig,
    seed: int,
    *,
    _reuse_randomized_workspace: bool = True,
) -> tuple[
    float,
    list[dict[str, Any]],
    NDArray[np.int64],
    NDArray[np.int64],
    str,
]:
    """Select alpha on an inner validation split without accessing outer test rows."""

    if not config.alpha_candidates:
        empty = np.empty(0, dtype=np.int64)
        return float(config.alpha), [], outer_train_indices.copy(), empty, "disabled"

    inner_split_seed = _derived_seed(seed, 1)
    candidate_solver_seed = _derived_seed(seed, 2)
    inner_fit_local, validation_local = stratified_train_test_split(
        labels[outer_train_indices],
        config.validation_fraction,
        inner_split_seed,
    )
    inner_fit_indices = outer_train_indices[inner_fit_local]
    validation_indices = outer_train_indices[validation_local]
    preprocessor = fit_genotype_preprocessor(
        raw_genotypes[inner_fit_indices],
        standardize=config.standardize,
    )
    inner_fit = transform_genotypes(raw_genotypes[inner_fit_indices], preprocessor)
    validation = transform_genotypes(raw_genotypes[validation_indices], preprocessor)
    if config.rank > min(inner_fit.shape):
        raise ValueError("rank exceeds the inner-training matrix dimensions")

    randomized_workspace = None
    if config.method == "randomized" and _reuse_randomized_workspace:
        from rspca.randomized import prepare_randomized_vp_spca_workspace

        randomized_workspace = prepare_randomized_vp_spca_workspace(
            inner_fit,
            config.rank,
            oversample=config.oversample,
            n_power_iter=config.n_power_iter,
            random_state=candidate_solver_seed,
        )

    records: list[dict[str, Any]] = []
    for alpha in config.alpha_candidates:
        candidate_config = replace(
            config,
            alpha=float(alpha),
            alpha_candidates=(),
        )
        result = _fit_solver(
            inner_fit,
            candidate_config,
            candidate_solver_seed,
            randomized_workspace=randomized_workspace,
        )
        fit_scores = inner_fit @ result.B
        validation_scores = validation @ result.B
        prediction, _, _ = _nearest_centroid_predictions(
            fit_scores,
            labels[inner_fit_indices],
            validation_scores,
        )
        metrics = _classification_metrics(labels[validation_indices], prediction)
        diagnostics = _fit_diagnostics(result)
        records.append(
            {
                "alpha": float(alpha),
                **metrics,
                "loading_sparsity": sparsity_ratio(
                    result.B,
                    threshold=config.selection_threshold,
                ),
                "solver_converged": diagnostics["converged"],
                "solver_status": diagnostics["status"],
                "solver_n_iter": diagnostics["n_iter"],
                "solver_final_stationarity": diagnostics["final_stationarity"],
                "solver_full_diagnostic_step_size": diagnostics[
                    "full_diagnostic_step_size"
                ],
                "solver_full_diagnostic_step_policy": diagnostics[
                    "full_diagnostic_step_policy"
                ],
                "candidate_solver_seed": candidate_solver_seed,
            }
        )

    converged_records = [record for record in records if record["solver_converged"]]
    eligible_records = converged_records or records
    selection_status = (
        "selected_from_converged_candidates"
        if converged_records
        else "no_candidate_converged_fixed_budget_fallback"
    )
    # Predeclared lexicographic rule among numerically eligible candidates:
    # downstream balanced accuracy, macro-F1, achieved sparsity, then smaller
    # alpha. Test labels never enter this choice. If no candidate converges, the
    # best fixed-budget iterate is retained but explicitly marked unqualified.
    selected = max(
        eligible_records,
        key=lambda record: (
            float(record["balanced_accuracy"]),
            float(record["macro_f1"]),
            float(record["loading_sparsity"]),
            -float(record["alpha"]),
        ),
    )
    return (
        float(selected["alpha"]),
        records,
        inner_fit_indices,
        validation_indices,
        selection_status,
    )


def _held_out_population_evaluation(
    raw_genotypes: NDArray[np.float64],
    labels: NDArray[np.str_],
    sample_ids: NDArray[np.str_],
    snp_ids: NDArray[np.str_],
    config: GenomicsConfig,
    seed: int,
    *,
    include_population_diagnostics: bool = False,
) -> dict[str, Any]:
    train_indices, test_indices = stratified_train_test_split(
        labels, config.test_fraction, seed
    )
    (
        selected_alpha,
        selection_records,
        inner_fit_indices,
        validation_indices,
        alpha_selection_status,
    ) = _select_alpha_on_training_partition(
        raw_genotypes,
        labels,
        train_indices,
        config,
        seed,
    )
    # Every learned quantity below is fitted without seeing test samples.
    preprocessor = fit_genotype_preprocessor(
        raw_genotypes[train_indices], standardize=config.standardize
    )
    train = transform_genotypes(raw_genotypes[train_indices], preprocessor)
    test = transform_genotypes(raw_genotypes[test_indices], preprocessor)
    if config.rank > min(train.shape):
        raise ValueError("rank exceeds the training matrix dimensions")
    final_solver_seed = _derived_seed(seed, 3) if config.alpha_candidates else seed
    final_config = replace(config, alpha=selected_alpha, alpha_candidates=())
    result = _fit_solver(train, final_config, final_solver_seed)
    fit_diagnostics = _fit_diagnostics(result)
    train_scores = train @ result.B
    test_scores = test @ result.B
    train_labels = labels[train_indices]
    test_labels = labels[test_indices]
    prediction, classes, centroids = _nearest_centroid_predictions(
        train_scores,
        train_labels,
        test_scores,
    )
    classification_diagnostics = _classification_diagnostics(
        test_labels,
        prediction,
        class_order=classes,
    )
    metrics = {
        metric: float(classification_diagnostics[metric])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    population_values, population_counts = np.unique(train_labels, return_counts=True)
    majority_class = population_values[int(np.argmax(population_counts))]
    majority_prediction = np.full(
        test_labels.shape, majority_class, dtype=test_labels.dtype
    )
    majority_metrics = _classification_metrics(test_labels, majority_prediction)
    record: dict[str, Any] = {
        "record_type": "held_out_population_classification",
        "method": config.method,
        "selected_alpha": selected_alpha if config.method != "pca" else None,
        "alpha_selection_enabled": bool(config.alpha_candidates),
        "alpha_selection_metric": (
            "validation_balanced_accuracy" if config.alpha_candidates else None
        ),
        "alpha_selection_tie_break": (
            "converged_eligibility,macro_f1,loading_sparsity,smaller_alpha"
            if config.alpha_candidates
            else None
        ),
        "alpha_selection_status": alpha_selection_status,
        "alpha_selection_qualified": alpha_selection_status
        in {"disabled", "selected_from_converged_candidates"},
        "alpha_selection_candidates": selection_records,
        "train_samples": int(train_indices.size),
        "inner_fit_samples": int(inner_fit_indices.size),
        "validation_samples": int(validation_indices.size),
        "test_samples": int(test_indices.size),
        "retained_snps": int(np.count_nonzero(preprocessor.retained_features)),
        "retained_snp_mask_sha256": array_sha256(preprocessor.retained_features),
        "retained_snp_ids_sha256": array_sha256(
            np.asarray(snp_ids)[preprocessor.retained_features]
        ),
        "populations": int(classes.size),
        **metrics,
        "majority_class": str(majority_class),
        "majority_baseline_accuracy": majority_metrics["accuracy"],
        "majority_baseline_balanced_accuracy": majority_metrics["balanced_accuracy"],
        "majority_baseline_macro_f1": majority_metrics["macro_f1"],
        "class_order": classes.astype(str).tolist(),
        "centroids_sha256": array_sha256(centroids),
        "test_predictions_sha256": array_sha256(prediction),
        "test_labels_sha256": array_sha256(test_labels),
        "solver_converged": fit_diagnostics["converged"],
        "solver_status": fit_diagnostics["status"],
        "solver_n_iter": fit_diagnostics["n_iter"],
        "solver_final_stationarity": fit_diagnostics["final_stationarity"],
        "solver_full_diagnostic_step_size": fit_diagnostics[
            "full_diagnostic_step_size"
        ],
        "solver_full_diagnostic_step_policy": fit_diagnostics[
            "full_diagnostic_step_policy"
        ],
        "preprocessing_fit_scope": "train_only",
        "spca_fit_scope": "train_only",
        "classifier_fit_scope": "train_only",
        "alpha_selection_scope": (
            "inner_train_validation_only" if config.alpha_candidates else "disabled"
        ),
        "test_used_for_model_or_feature_selection": False,
        "split_seed": seed,
        "inner_split_seed": (
            _derived_seed(seed, 1) if config.alpha_candidates else None
        ),
        "final_solver_seed": final_solver_seed,
        "train_indices_sha256": array_sha256(train_indices),
        "inner_fit_indices_sha256": array_sha256(inner_fit_indices),
        "validation_indices_sha256": (
            array_sha256(validation_indices) if validation_indices.size else None
        ),
        "test_indices_sha256": array_sha256(test_indices),
        "train_sample_ids_sha256": array_sha256(np.asarray(sample_ids)[train_indices]),
        "test_sample_ids_sha256": array_sha256(np.asarray(sample_ids)[test_indices]),
    }
    if include_population_diagnostics:
        squared_distances = np.sum(
            (test_scores[:, None, :] - centroids[None, :, :]) ** 2,
            axis=2,
        )
        final_loadings_original = _embed_loadings_in_original_snp_coordinates(
            result.B,
            preprocessor.retained_features,
            preprocessor.scales,
        )
        final_selected_support = np.any(
            np.abs(final_loadings_original) > config.selection_threshold,
            axis=1,
        )
        final_selected_snp_ids = np.asarray(snp_ids)[final_selected_support]
        record.update(
            {
                "population_diagnostics_schema_version": 1,
                "population_label_semantics": "metadata_population_group",
                "population_label_is_race": False,
                "classifier": "nearest_training_centroid",
                "embedding_fit_scope": "outer_train_only",
                "embedding_coordinate_system": "train_fitted_component_scores",
                "embedding_components": int(test_scores.shape[1]),
                "final_loading_entry_sparsity": sparsity_ratio(
                    final_loadings_original,
                    threshold=config.selection_threshold,
                ),
                "final_selected_snp_count": int(
                    np.count_nonzero(final_selected_support)
                ),
                "final_selected_snp_fraction": float(np.mean(final_selected_support)),
                "final_selected_snp_ids": final_selected_snp_ids.astype(str).tolist(),
                "final_selected_snp_ids_sha256": array_sha256(final_selected_snp_ids),
                "final_loading_support_threshold": float(config.selection_threshold),
                "final_loading_support_coordinate_system": (
                    "original_snp_raw_dosage_coefficients"
                ),
                "final_loading_support_fit_scope": "outer_train_only",
                "confusion_matrix_counts": classification_diagnostics[
                    "confusion_matrix_counts"
                ],
                "confusion_matrix_true_normalized": classification_diagnostics[
                    "confusion_matrix_true_normalized"
                ],
                "per_population_metrics": classification_diagnostics[
                    "per_population_metrics"
                ],
                "held_out_prediction_records": [
                    {
                        "sample_index": int(sample_index),
                        "sample_id": str(sample_ids[sample_index]),
                        "true_population": str(true_label),
                        "predicted_population": str(predicted_label),
                        "embedding": [float(value) for value in score],
                        "squared_distances_to_centroids": [
                            float(value) for value in distances
                        ],
                    }
                    for sample_index, true_label, predicted_label, score, distances in zip(
                        test_indices,
                        test_labels,
                        prediction,
                        test_scores,
                        squared_distances,
                        strict=True,
                    )
                ],
            }
        )
    return record


def load_population_labels(
    metadata_path: Path | str,
    sample_ids: Sequence[str],
    *,
    sample_id_column: str = "sample_id",
    population_column: str = "population",
    delimiter: str = ",",
) -> NDArray[np.str_]:
    """Align population labels to genotype sample IDs and reject missing IDs."""

    with Path(metadata_path).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError("population metadata needs a header")
        required = {sample_id_column, population_column}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"population metadata is missing columns: {sorted(required)}"
            )
        mapping: dict[str, str] = {}
        for row in reader:
            sample = row[sample_id_column]
            if sample in mapping:
                raise ValueError(f"duplicate population metadata for sample {sample!r}")
            mapping[sample] = row[population_column]
    missing = [sample for sample in sample_ids if sample not in mapping]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"population metadata is missing {len(missing)} samples: {preview}"
        )
    labels = np.asarray([mapping[sample] for sample in sample_ids], dtype=str)
    if np.any(labels == ""):
        raise ValueError("population labels must not be empty")
    return labels


def run_genomics(
    dataset: GenotypeDataset,
    config: GenomicsConfig,
    *,
    population_labels: NDArray[np.str_] | None = None,
    include_population_diagnostics: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run full-cohort stability and optional independent held-out evaluation."""

    _validate_config(config)
    raw_genotypes = _normalize_missing_values(
        dataset.genotypes,
        config.missing_value,
    )
    if raw_genotypes.ndim != 2:
        raise ValueError("dataset.genotypes must be two-dimensional")
    if np.asarray(dataset.sample_ids).shape != (raw_genotypes.shape[0],):
        raise ValueError("dataset.sample_ids must contain one ID per sample")
    if np.asarray(dataset.snp_ids).shape != (raw_genotypes.shape[1],):
        raise ValueError("dataset.snp_ids must contain one ID per SNP")
    labels_for_evaluation: NDArray[np.str_] | None = None
    if population_labels is not None:
        labels_for_evaluation = np.asarray(population_labels, dtype=str)
        if labels_for_evaluation.shape != (raw_genotypes.shape[0],):
            raise ValueError("population_labels must have one label per sample")
        if np.unique(labels_for_evaluation).size < 2:
            raise ValueError(
                "held-out population evaluation requires at least two populations"
            )
    if not isinstance(include_population_diagnostics, (bool, np.bool_)):
        raise TypeError("include_population_diagnostics must be boolean")
    seed_count = config.bootstrap_repeats + 2
    manifest = make_seed_manifest(
        config.base_seed,
        seed_count,
        namespace="genomics-1kg-v2",
        metadata={
            **asdict(config),
            "n_samples": int(raw_genotypes.shape[0]),
            "n_snps": int(raw_genotypes.shape[1]),
            "orientation": "samples-by-snps",
            "loader_schema_version": 2,
            "normalized_genotypes_sha256": array_sha256(raw_genotypes),
            "sample_ids_sha256": array_sha256(dataset.sample_ids),
            "snp_ids_sha256": array_sha256(dataset.snp_ids),
            "seed_roles": {
                "index_0": "full_cohort_solver",
                "index_1": "held_out_split_and_selection_parent",
                "index_2_onward": "bootstrap_solver_parent_for_resample_seed",
            },
        },
    )

    # This full-cohort preprocessing is used only for unsupervised stability,
    # never for the held-out population score below.
    full_preprocessor = fit_genotype_preprocessor(
        raw_genotypes, standardize=config.standardize
    )
    matrix = transform_genotypes(raw_genotypes, full_preprocessor)
    if config.rank > min(matrix.shape):
        raise ValueError("rank exceeds the preprocessed full-cohort matrix dimensions")
    base = _fit_solver(matrix, config, manifest["seeds"][0])
    base_fit_diagnostics = _fit_diagnostics(base)
    base_loadings_original = _embed_loadings_in_original_snp_coordinates(
        base.B,
        full_preprocessor.retained_features,
        full_preprocessor.scales,
    )
    reconstruction = matrix @ base.B @ base.A.T
    rows: list[dict[str, Any]] = [
        {
            "record_type": "full_cohort_unsupervised_fit",
            "method": config.method,
            "samples": int(matrix.shape[0]),
            "input_snps": int(raw_genotypes.shape[1]),
            "retained_snps": int(matrix.shape[1]),
            "retained_snp_mask_sha256": array_sha256(
                full_preprocessor.retained_features
            ),
            "retained_snp_ids_sha256": array_sha256(
                np.asarray(dataset.snp_ids)[full_preprocessor.retained_features]
            ),
            "rank": config.rank,
            "alpha": config.alpha,
            "relative_reconstruction_error": relative_reconstruction_error(
                matrix, reconstruction
            ),
            "explained_variance_ratio": explained_variance_ratio(
                matrix, reconstruction, center=False
            ),
            "loading_sparsity": sparsity_ratio(
                base_loadings_original, threshold=config.selection_threshold
            ),
            "loading_sparsity_retained_coordinate": sparsity_ratio(
                base.B, threshold=config.selection_threshold
            ),
            **base_fit_diagnostics,
            "analysis_scope": "unsupervised_full_cohort",
            "preprocessing_scope": "full_cohort_unsupervised",
            "loading_coordinate_system": "original_snp_raw_dosage_coefficients",
            "support_threshold_coordinate_system": (
                "original_snp_raw_dosage_coefficients"
            ),
        }
    ]

    base_feature_support = np.any(
        np.abs(base_loadings_original) > config.selection_threshold,
        axis=1,
    )
    selection_counts = np.zeros(raw_genotypes.shape[1], dtype=int)
    for bootstrap_index in range(config.bootstrap_repeats):
        solver_seed = int(manifest["seeds"][bootstrap_index + 2])
        resample_seed = _derived_seed(solver_seed, 11)
        resample_rng = np.random.default_rng(resample_seed)
        sampled_rows = resample_rng.integers(
            0,
            raw_genotypes.shape[0],
            size=raw_genotypes.shape[0],
        )
        bootstrap_raw = raw_genotypes[sampled_rows]
        bootstrap_preprocessor = fit_genotype_preprocessor(
            bootstrap_raw,
            standardize=config.standardize,
        )
        bootstrap_matrix = transform_genotypes(
            bootstrap_raw,
            bootstrap_preprocessor,
        )
        if config.rank > min(bootstrap_matrix.shape):
            raise ValueError(
                "rank exceeds a bootstrap matrix dimension after filtering"
            )
        bootstrap = _fit_solver(
            bootstrap_matrix,
            config,
            solver_seed,
        )
        bootstrap_loadings_original = _embed_loadings_in_original_snp_coordinates(
            bootstrap.B,
            bootstrap_preprocessor.retained_features,
            bootstrap_preprocessor.scales,
        )
        bootstrap_fit_diagnostics = _fit_diagnostics(bootstrap)
        bootstrap_support = np.any(
            np.abs(bootstrap_loadings_original) > config.selection_threshold,
            axis=1,
        )
        selection_counts += bootstrap_support
        correlations = component_correlation(
            base_loadings_original,
            bootstrap_loadings_original,
        )
        overlap = support_overlap_diagnostics(base_feature_support, bootstrap_support)
        rows.append(
            {
                "record_type": "bootstrap_stability",
                "method": config.method,
                "bootstrap": bootstrap_index,
                "seed": solver_seed,
                "solver_seed": solver_seed,
                "resample_seed": resample_seed,
                "sampled_rows": int(sampled_rows.size),
                "unique_sampled_rows": int(np.unique(sampled_rows).size),
                "sampled_row_indices_sha256": array_sha256(sampled_rows),
                "retained_snps": int(bootstrap_matrix.shape[1]),
                "retained_snp_mask_sha256": array_sha256(
                    bootstrap_preprocessor.retained_features
                ),
                "retained_snp_ids_sha256": array_sha256(
                    np.asarray(dataset.snp_ids)[
                        bootstrap_preprocessor.retained_features
                    ]
                ),
                **overlap,
                "subspace_error": subspace_error(
                    base_loadings_original,
                    bootstrap_loadings_original,
                ),
                "component_mean_abs_correlation": correlations["mean_abs_correlation"],
                "component_min_abs_correlation": correlations["min_abs_correlation"],
                "loading_sparsity": sparsity_ratio(
                    bootstrap_loadings_original,
                    threshold=config.selection_threshold,
                ),
                "loading_sparsity_retained_coordinate": sparsity_ratio(
                    bootstrap.B,
                    threshold=config.selection_threshold,
                ),
                **bootstrap_fit_diagnostics,
                "preprocessing_scope": "bootstrap_raw_resample_only",
                "imputation_fit_scope": "bootstrap_raw_resample_only",
                "filtering_fit_scope": "bootstrap_raw_resample_only",
                "scaling_fit_scope": "bootstrap_raw_resample_only",
                "loading_coordinate_system": ("original_snp_raw_dosage_coefficients"),
                "support_threshold_coordinate_system": (
                    "original_snp_raw_dosage_coefficients"
                ),
            }
        )

    retained_indices = np.flatnonzero(full_preprocessor.retained_features)
    retained_lookup = {
        int(original_index): int(retained_index)
        for retained_index, original_index in enumerate(retained_indices)
    }
    if config.bootstrap_repeats:
        frequencies = selection_counts / config.bootstrap_repeats
        for original_index in range(raw_genotypes.shape[1]):
            rows.append(
                {
                    "record_type": "snp_stability",
                    "method": config.method,
                    "snp_id": str(dataset.snp_ids[original_index]),
                    "original_snp_index": int(original_index),
                    "retained_in_base_fit": bool(
                        full_preprocessor.retained_features[original_index]
                    ),
                    "retained_snp_index": retained_lookup.get(original_index),
                    "selection_frequency": float(frequencies[original_index]),
                    "selected_in_base_fit": bool(base_feature_support[original_index]),
                    "bootstrap_repeats": config.bootstrap_repeats,
                    "support_threshold": config.selection_threshold,
                    "support_threshold_coordinate_system": (
                        "original_snp_raw_dosage_coefficients"
                    ),
                }
            )

    if labels_for_evaluation is not None:
        rows.append(
            _held_out_population_evaluation(
                raw_genotypes,
                labels_for_evaluation,
                np.asarray(dataset.sample_ids, dtype=str),
                np.asarray(dataset.snp_ids, dtype=str),
                config,
                manifest["seeds"][1],
                include_population_diagnostics=include_population_diagnostics,
            )
        )
    return rows, manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("non-finite numbers are forbidden in genomics artifacts")
    return value


def _reject_nonfinite_numbers(value: Any, *, location: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, location=f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, location=f"{location}[{index}]")
    elif isinstance(value, np.ndarray):
        _reject_nonfinite_numbers(value.tolist(), location=location)
    elif isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        raise ValueError(f"number must be finite at {location}")


def _resolved_output_paths(
    output_json: str | Path | None,
    output_csv: str | Path | None,
) -> dict[str, Path]:
    requested = {
        name: Path(path)
        for name, path in (("json", output_json), ("csv", output_csv))
        if path is not None
    }
    resolved = [path.resolve(strict=False) for path in requested.values()]
    if len(set(resolved)) != len(resolved):
        raise ValueError("output_json and output_csv must be different paths")
    existing = [str(path) for path in requested.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing}")
    return requested


def validate_output_paths(
    *, output_json: str | Path | None, output_csv: str | Path | None
) -> None:
    """Fail before data loading or fitting if requested outputs are unsafe."""

    _resolved_output_paths(output_json, output_csv)


def write_results(
    rows: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    config: GenomicsConfig,
    provenance: Mapping[str, Any],
    *,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> None:
    """Exclusive-create explicitly requested strict JSON/CSV result artifacts."""

    paths = _resolved_output_paths(output_json, output_csv)
    payload = {
        "schema_version": 2,
        "config": asdict(config),
        "provenance": provenance,
        "seed_manifest": manifest,
        "records": rows,
    }
    _reject_nonfinite_numbers(payload)
    json_text = (
        json.dumps(
            _json_safe(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )

    if "json" in paths:
        json_path = paths["json"]
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("x", encoding="utf-8") as handle:
            handle.write(json_text)
    if "csv" in paths:
        csv_path = paths["csv"]
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_context = {
            "manifest_sha256": manifest["manifest_sha256"],
            "loader_schema_version": provenance.get("loader_schema_version"),
            "genotype_sha256": provenance.get("genotype_sha256"),
            "metadata_sha256": provenance.get("metadata_sha256"),
            "sample_ids_sha256": provenance.get("sample_ids_sha256"),
            "snp_ids_sha256": provenance.get("snp_ids_sha256"),
            "input_orientation": provenance.get("input_orientation"),
            "internal_orientation": "samples-by-snps",
            "missing_value_policy": provenance.get("missing_value_policy"),
            "full_cohort_retained_snp_ids_sha256": provenance.get(
                "full_cohort_retained_snp_ids_sha256"
            ),
            "rank": config.rank,
            "alpha": config.alpha,
            "base_seed": config.base_seed,
            "missing_value": config.missing_value,
        }
        csv_rows = [{**csv_context, **row} for row in rows]
        fieldnames = sorted({key for row in csv_rows for key in row})
        with csv_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if fieldnames:
                writer.writeheader()
                writer.writerows(_json_safe(csv_rows))


def _decode_delimiter(value: str | None) -> str | None:
    if value is None:
        return None
    decoded = bytes(value, "utf-8").decode("unicode_escape")
    if len(decoded) != 1:
        raise argparse.ArgumentTypeError("delimiter must decode to one character")
    return decoded


def _parse_missing_value(value: str) -> float | None:
    if value.strip().lower() in {"none", "off", "disabled"}:
        return None
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "missing value must be a finite number or 'none'"
        ) from error
    if not np.isfinite(parsed):
        raise argparse.ArgumentTypeError(
            "missing value must be a finite number or 'none'"
        )
    return parsed


def _parse_alpha_candidates(value: str) -> tuple[float, ...]:
    try:
        candidates = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "alpha candidates must be comma-separated numbers"
        ) from error
    if (
        not candidates
        or any(not np.isfinite(alpha) or alpha < 0.0 for alpha in candidates)
        or len(set(candidates)) != len(candidates)
    ):
        raise argparse.ArgumentTypeError(
            "alpha candidates must be unique finite non-negative numbers"
        )
    return candidates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genotypes", type=Path)
    parser.add_argument(
        "--orientation",
        choices=("samples-by-snps", "snps-by-samples"),
        default="samples-by-snps",
    )
    parser.add_argument("--delimiter", type=_decode_delimiter)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--row-id-column")
    parser.add_argument(
        "--missing-value",
        type=_parse_missing_value,
        default=-9.0,
        help="numeric missing sentinel for every input format; use 'none' to disable",
    )
    parser.add_argument(
        "--method", choices=("pca", "vp", "randomized", "robust"), default="pca"
    )
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--alpha-candidates",
        type=_parse_alpha_candidates,
        default=(),
        help=(
            "optional comma-separated alpha grid selected on an inner training/"
            "validation split; the outer test split remains untouched"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--bootstrap-repeats", type=int, default=20)
    parser.add_argument("--selection-threshold", type=float, default=1e-8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--base-seed", type=int, default=2025)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--oversample", type=int, default=10)
    parser.add_argument("--n-power-iter", type=int, default=2)
    parser.add_argument("--randomized-full-gradient", action="store_true")
    parser.add_argument("--randomized-finalize-full-iters", type=int, default=5)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--metadata-delimiter", type=_decode_delimiter, default=",")
    parser.add_argument("--metadata-sample-id-column", default="sample_id")
    parser.add_argument("--population-column", default="population")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    validate_output_paths(
        output_json=arguments.output_json,
        output_csv=arguments.output_csv,
    )
    dataset = load_genotypes(
        arguments.genotypes,
        orientation=arguments.orientation,
        delimiter=arguments.delimiter,
        has_header=not arguments.no_header,
        row_id_column=arguments.row_id_column,
        missing_value=arguments.missing_value,
    )
    config = GenomicsConfig(
        method=arguments.method,
        rank=arguments.rank,
        alpha=arguments.alpha,
        alpha_candidates=arguments.alpha_candidates,
        validation_fraction=arguments.validation_fraction,
        max_iter=arguments.max_iter,
        tol=arguments.tol,
        bootstrap_repeats=arguments.bootstrap_repeats,
        selection_threshold=arguments.selection_threshold,
        test_fraction=arguments.test_fraction,
        base_seed=arguments.base_seed,
        standardize=not arguments.no_standardize,
        oversample=arguments.oversample,
        n_power_iter=arguments.n_power_iter,
        randomized_full_gradient=arguments.randomized_full_gradient,
        randomized_finalize_full_iters=arguments.randomized_finalize_full_iters,
        missing_value=arguments.missing_value,
    )
    labels = (
        load_population_labels(
            arguments.metadata,
            dataset.sample_ids,
            sample_id_column=arguments.metadata_sample_id_column,
            population_column=arguments.population_column,
            delimiter=arguments.metadata_delimiter,
        )
        if arguments.metadata is not None
        else None
    )
    rows, manifest = run_genomics(dataset, config, population_labels=labels)
    base = next(
        row for row in rows if row["record_type"] == "full_cohort_unsupervised_fit"
    )
    classification = next(
        (
            row
            for row in rows
            if row["record_type"] == "held_out_population_classification"
        ),
        None,
    )
    provenance = {
        "loader_schema_version": 2,
        "runtime_versions": runtime_versions(),
        "genotype_path": str(arguments.genotypes.resolve()),
        "genotype_sha256": _file_sha256(arguments.genotypes),
        "input_orientation": arguments.orientation,
        "internal_orientation": "samples-by-snps",
        "samples": int(dataset.genotypes.shape[0]),
        "snps": int(dataset.genotypes.shape[1]),
        "sample_ids_sha256": array_sha256(dataset.sample_ids),
        "snp_ids_sha256": array_sha256(dataset.snp_ids),
        "missing_value": arguments.missing_value,
        "missing_value_policy": (
            "disabled"
            if arguments.missing_value is None
            else "exact_numeric_sentinel_to_nan"
        ),
        "full_cohort_retained_snp_mask_sha256": base["retained_snp_mask_sha256"],
        "full_cohort_retained_snp_ids_sha256": base["retained_snp_ids_sha256"],
        "held_out_retained_snp_mask_sha256": (
            classification["retained_snp_mask_sha256"]
            if classification is not None
            else None
        ),
        "held_out_retained_snp_ids_sha256": (
            classification["retained_snp_ids_sha256"]
            if classification is not None
            else None
        ),
        "metadata_path": (
            str(arguments.metadata.resolve()) if arguments.metadata else None
        ),
        "metadata_sha256": (
            _file_sha256(arguments.metadata) if arguments.metadata else None
        ),
    }
    write_results(
        rows,
        manifest,
        config,
        provenance,
        output_json=arguments.output_json,
        output_csv=arguments.output_csv,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest["manifest_sha256"],
                "samples": base["samples"],
                "retained_snps": base["retained_snps"],
                "relative_reconstruction_error": base["relative_reconstruction_error"],
                "held_out_population_classification": classification,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage-safe, loading-only AIM evaluation for the repository's PCA methods.

This module deliberately does not import the sparse-PCA algorithms from Lee et
al. (2012). The paper motivates the ancestry-informative-marker (AIM) question;
the fitted methods here remain this repository's PCA, VP-SPCA, randomized
VP-SPCA, and robust VP-SPCA implementations.

Every matrix uses ``samples x SNPs`` orientation.  Population labels are used
only to construct a stratified split, fit training centroids after an AIM panel
has been frozen, and score held-out samples.  Preprocessing, component fitting,
loading-based ranking, and panel refitting never receive population labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .datasets import array_sha256, canonical_sha256


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

METHODS = ("pca", "vp", "randomized", "robust")
METHOD_SEED_ROLES = {
    "pca": 101,
    "vp": 211,
    "randomized": 307,
    "robust": 401,
}
POPULATION_DESIGNS = ("three_continent", "five_superpopulation")
INPUT_SCOPES = (
    "legacy_chr20_22",
    "all_autosomes",
    "synthetic_multi_autosome_smoke",
)
EXPECTED_SUPERPOPULATIONS = frozenset({"AFR", "AMR", "EAS", "EUR", "SAS"})
LEGACY_GITHUB_BASELINE_COMMIT = "03f4b432e5d552a9772cfb251fe06f33833b4197"
LEGACY_SOURCE_SHA256 = {
    "gene.ipynb": "60110c78b28da17042af55ee63087aceb4f786542e8cd478f783f21f710d29fa",
    "gene large scale.ipynb": "37c6605617186f28a2af11a2ad9c2db7c657d13e4b78314ae05feeaacf9e4f4b",
    "spca_variable_projection.py": "0611c5cf14701c6b54ac6624473a3e3b7738d1220bf1f48fc12a8c83bbf9e93d",
    "randomized_spca_vp.py": "5ce6a21c86bdd88f23bebe689cdf6ac9178854838ee8329b82c67f02205d40dc",
    "robust_spca_vp.py": "eac1b41bd1fb27d5542cf42d3cb05e1498ebbc3be57616efe1fa112133ac4692",
}


@dataclass(frozen=True)
class MultiAutosomeGenotypeData:
    """Validated input and provenance declarations for an AIM experiment.

    ``source_manifest_sha256`` identifies an external manifest; this module
    validates its syntax but cannot authenticate the files named by it.
    ``ld_pruned`` is likewise an explicit declaration that must be backed by a
    preprocessing receipt for a confirmatory run.
    """

    genotypes: FloatArray
    sample_ids: StringArray
    snp_ids: StringArray
    chromosomes: StringArray
    population_labels: StringArray
    source_manifest_sha256: str
    input_scope: str = "all_autosomes"
    ld_pruned: bool = True


@dataclass(frozen=True)
class SmartPCAPreprocessor:
    """Training-only allele-frequency imputation and smartpca-like scaling."""

    mean_dosage: FloatArray
    allele_frequency: FloatArray
    scale: FloatArray
    retained_snps: BoolArray


@dataclass(frozen=True)
class AIMProtocolConfig:
    """Frozen evaluation controls shared by all compared methods."""

    population_design: str
    methods: tuple[str, ...] = METHODS
    n_components: int = 2
    panel_size: int = 200
    test_fraction: float = 0.2
    seeds: tuple[int, ...] = (2025, 2026, 2027, 2028, 2029)
    loading_score: str = "l1"
    native_support_threshold: float = 1e-8
    near_dense_fraction: float = 0.95
    vp_alpha: float = 0.1
    randomized_alpha: float = 0.1
    robust_alpha: float = 0.1
    max_iter: int = 500
    tol: float = 1e-6
    randomized_oversample: int = 10
    randomized_power_iterations: int = 2
    randomized_finalize_full_iters: int = 5
    robust_huber_delta: float = 1.345
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 1701


@dataclass(frozen=True)
class _MethodFit:
    loadings: FloatArray
    converged: bool
    status: str
    n_iter: int
    final_stationarity: float


@dataclass(frozen=True)
class AIMProtocolResult:
    """Per-split evidence, aggregate summaries, and per-SNP selection rates."""

    config: AIMProtocolConfig
    input_identity: Mapping[str, Any]
    split_records: tuple[Mapping[str, Any], ...]
    method_summaries: tuple[Mapping[str, Any], ...]
    selection_frequencies: Mapping[str, FloatArray]
    registered_coordinates: tuple[Mapping[str, Any], ...]
    solver_lineage: Mapping[str, Mapping[str, Any]]
    result_sha256: str


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be positive")
    return integer


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def _finite_scalar(value: float, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real scalar")
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(scalar) or (positive and scalar <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return scalar


def _normalized_autosome(chromosome: str) -> str:
    value = str(chromosome).strip().lower()
    if value.startswith("chr"):
        value = value[3:]
    try:
        number = int(value)
    except ValueError as error:
        raise ValueError(f"non-autosomal chromosome label: {chromosome!r}") from error
    if not 1 <= number <= 22:
        raise ValueError(f"non-autosomal chromosome label: {chromosome!r}")
    return str(number)


def validate_multiautosome_input(data: MultiAutosomeGenotypeData) -> None:
    """Fail closed on orientation, identity, dosage, and genome-scope errors."""

    if not isinstance(data, MultiAutosomeGenotypeData):
        raise TypeError("data must be a MultiAutosomeGenotypeData")
    matrix = np.asarray(data.genotypes, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(
            "genotypes must be a samples x SNPs matrix with both axes >= 2"
        )
    arrays = {
        "sample_ids": (data.sample_ids, matrix.shape[0]),
        "population_labels": (data.population_labels, matrix.shape[0]),
        "snp_ids": (data.snp_ids, matrix.shape[1]),
        "chromosomes": (data.chromosomes, matrix.shape[1]),
    }
    for name, (values, expected) in arrays.items():
        array = np.asarray(values, dtype=str)
        if array.shape != (expected,):
            raise ValueError(f"{name} must contain exactly {expected} entries")
        if np.any(np.char.strip(array) == ""):
            raise ValueError(f"{name} must not contain empty values")
    if np.unique(np.asarray(data.sample_ids, dtype=str)).size != matrix.shape[0]:
        raise ValueError("sample_ids must be unique")
    if np.unique(np.asarray(data.snp_ids, dtype=str)).size != matrix.shape[1]:
        raise ValueError("snp_ids must be unique")
    if np.any(np.isinf(matrix)):
        raise ValueError("genotypes may contain NaN missingness but not infinity")
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0 or np.any((finite < 0.0) | (finite > 2.0)):
        raise ValueError("finite genotype dosages must lie in [0, 2]")
    if data.input_scope not in INPUT_SCOPES:
        raise ValueError(f"input_scope must be one of {INPUT_SCOPES}")
    chromosomes = {_normalized_autosome(value) for value in data.chromosomes}
    if data.input_scope == "all_autosomes":
        expected = {str(number) for number in range(1, 23)}
        if chromosomes != expected:
            missing = sorted(expected - chromosomes, key=int)
            extra = sorted(chromosomes - expected, key=int)
            raise ValueError(
                "all_autosomes input must cover chromosomes 1-22 exactly; "
                f"missing={missing}, extra={extra}"
            )
    elif data.input_scope == "legacy_chr20_22":
        if chromosomes != {"20", "21", "22"}:
            raise ValueError(
                "legacy_chr20_22 input must cover chromosomes 20, 21, and 22 exactly"
            )
    elif len(chromosomes) < 2:
        raise ValueError("synthetic smoke input must span at least two autosomes")
    digest = str(data.source_manifest_sha256)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("source_manifest_sha256 must be 64 lowercase hex characters")
    if not isinstance(data.ld_pruned, (bool, np.bool_)) or not data.ld_pruned:
        raise ValueError("AIM protocol requires an explicitly LD-pruned input")


def _validate_config(
    config: AIMProtocolConfig, data: MultiAutosomeGenotypeData
) -> None:
    if not isinstance(config, AIMProtocolConfig):
        raise TypeError("config must be an AIMProtocolConfig")
    if config.population_design not in POPULATION_DESIGNS:
        raise ValueError(f"population_design must be one of {POPULATION_DESIGNS}")
    if not config.methods or len(set(config.methods)) != len(config.methods):
        raise ValueError("methods must be non-empty and unique")
    if any(method not in METHODS for method in config.methods):
        raise ValueError(f"methods must be selected from {METHODS}")
    if "pca" not in config.methods:
        raise ValueError("PCA is required as the equal-budget reference")
    _positive_integer(config.n_components, name="n_components")
    if config.n_components < 2:
        raise ValueError("n_components must be at least 2 for registered 2D output")
    _positive_integer(config.panel_size, name="panel_size")
    if config.panel_size < config.n_components:
        raise ValueError("panel_size must be at least n_components")
    _positive_integer(config.max_iter, name="max_iter")
    _positive_integer(config.bootstrap_resamples, name="bootstrap_resamples")
    _nonnegative_integer(config.bootstrap_seed, name="bootstrap_seed")
    _nonnegative_integer(config.randomized_oversample, name="randomized_oversample")
    _nonnegative_integer(
        config.randomized_power_iterations,
        name="randomized_power_iterations",
    )
    _nonnegative_integer(
        config.randomized_finalize_full_iters,
        name="randomized_finalize_full_iters",
    )
    if config.loading_score not in {"l1", "l2", "max"}:
        raise ValueError("loading_score must be l1, l2, or max")
    for name in (
        "native_support_threshold",
        "vp_alpha",
        "randomized_alpha",
        "robust_alpha",
    ):
        if _finite_scalar(getattr(config, name), name=name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    _finite_scalar(config.tol, name="tol", positive=True)
    _finite_scalar(config.robust_huber_delta, name="robust_huber_delta", positive=True)
    test_fraction = _finite_scalar(config.test_fraction, name="test_fraction")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    near_dense = _finite_scalar(config.near_dense_fraction, name="near_dense_fraction")
    if not 0.0 < near_dense <= 1.0:
        raise ValueError("near_dense_fraction must lie in (0, 1]")
    if len(config.seeds) < 2 or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("at least two unique seeds are required for stability")
    for seed in config.seeds:
        _nonnegative_integer(seed, name="seed")
    labels = np.asarray(data.population_labels, dtype=str)
    populations = set(np.unique(labels).tolist())
    if config.population_design == "three_continent" and len(populations) != 3:
        raise ValueError("three_continent design requires exactly three labels")
    if (
        config.population_design == "five_superpopulation"
        and populations != EXPECTED_SUPERPOPULATIONS
    ):
        raise ValueError(
            "five_superpopulation design requires AFR, AMR, EAS, EUR, and SAS"
        )
    for population in populations:
        if np.count_nonzero(labels == population) < 2:
            raise ValueError(f"population {population!r} requires at least two samples")


def fit_smartpca_preprocessor(
    train_genotypes: ArrayLike,
    *,
    minimum_scale: float = 1e-12,
) -> SmartPCAPreprocessor:
    """Fit allele-frequency imputation/scaling on training samples only.

    This matches the historical notebooks' ``(G - 2p) / sqrt(2p(1-p))``
    convention while fixing their full-cohort leakage.
    """

    matrix = np.asarray(train_genotypes, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("train_genotypes must be a non-empty two-dimensional matrix")
    if np.any(np.isinf(matrix)):
        raise ValueError("train_genotypes may contain NaN but not infinity")
    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0 or np.any((finite_values < 0.0) | (finite_values > 2.0)):
        raise ValueError("finite genotype dosages must lie in [0, 2]")
    minimum_scale = _finite_scalar(
        minimum_scale,
        name="minimum_scale",
        positive=True,
    )
    observed = np.isfinite(matrix)
    counts = observed.sum(axis=0)
    sums = np.where(observed, matrix, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(matrix.shape[1]), where=counts > 0)
    frequencies = means / 2.0
    scales = np.sqrt(2.0 * frequencies * (1.0 - frequencies))
    retained = (counts > 0) & np.isfinite(scales) & (scales > minimum_scale)
    if not np.any(retained):
        raise ValueError("no polymorphic SNPs remain in the training partition")
    safe_scales = np.where(retained, scales, 1.0)
    return SmartPCAPreprocessor(
        mean_dosage=np.asarray(means, dtype=float),
        allele_frequency=np.asarray(frequencies, dtype=float),
        scale=np.asarray(safe_scales, dtype=float),
        retained_snps=np.asarray(retained, dtype=bool),
    )


def transform_smartpca(
    genotypes: ArrayLike,
    preprocessor: SmartPCAPreprocessor,
) -> FloatArray:
    """Apply frozen training statistics without inspecting evaluation rows."""

    if not isinstance(preprocessor, SmartPCAPreprocessor):
        raise TypeError("preprocessor must be a SmartPCAPreprocessor")
    matrix = np.asarray(genotypes, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != preprocessor.mean_dosage.size:
        raise ValueError("genotypes do not match the fitted preprocessor width")
    if np.any(np.isinf(matrix)):
        raise ValueError("genotypes may contain NaN but not infinity")
    finite_values = matrix[np.isfinite(matrix)]
    if np.any((finite_values < 0.0) | (finite_values > 2.0)):
        raise ValueError("finite genotype dosages must lie in [0, 2]")
    imputed = np.where(np.isfinite(matrix), matrix, preprocessor.mean_dosage)
    retained = preprocessor.retained_snps
    transformed = imputed[:, retained] - preprocessor.mean_dosage[retained]
    transformed /= preprocessor.scale[retained]
    if not np.isfinite(transformed).all():
        raise RuntimeError("preprocessing produced non-finite values")
    return np.asarray(transformed, dtype=float)


def stratified_split(
    labels: Sequence[str],
    test_fraction: float,
    seed: int,
) -> tuple[IntArray, IntArray]:
    """Return deterministic train/test indices with every population in both."""

    values = np.asarray(labels, dtype=str)
    if values.ndim != 1 or values.size < 4:
        raise ValueError("labels must be a one-dimensional array with at least 4 rows")
    fraction = _finite_scalar(test_fraction, name="test_fraction")
    if not 0.0 < fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    seed = _nonnegative_integer(seed, name="seed")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    test: list[int] = []
    for label in np.unique(values):
        indices = np.flatnonzero(values == label)
        if indices.size < 2:
            raise ValueError(f"population {label!r} requires at least two samples")
        shuffled = rng.permutation(indices)
        count = min(indices.size - 1, max(1, int(round(fraction * indices.size))))
        test.extend(shuffled[:count].tolist())
        train.extend(shuffled[count:].tolist())
    return (
        np.asarray(rng.permutation(train), dtype=np.int64),
        np.asarray(rng.permutation(test), dtype=np.int64),
    )


def loading_importance(loadings: ArrayLike, *, score: str = "l1") -> FloatArray:
    """Aggregate unsupervised component loadings into one score per SNP."""

    matrix = np.asarray(loadings, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1 or not np.isfinite(matrix).all():
        raise ValueError("loadings must be a non-empty finite two-dimensional matrix")
    magnitude = np.abs(matrix)
    if score == "l1":
        importance = np.sum(magnitude, axis=1)
    elif score == "l2":
        importance = np.sqrt(np.sum(magnitude * magnitude, axis=1))
    elif score == "max":
        importance = np.max(magnitude, axis=1)
    else:
        raise ValueError("score must be l1, l2, or max")
    if not np.any(importance > 0.0):
        raise ValueError("all loading importance scores are zero")
    return np.asarray(importance, dtype=float)


def select_loading_panel(
    loadings: ArrayLike,
    panel_size: int,
    *,
    score: str = "l1",
) -> tuple[IntArray, FloatArray]:
    """Select an exact top-K panel with ties broken by original column index."""

    importance = loading_importance(loadings, score=score)
    panel_size = _positive_integer(panel_size, name="panel_size")
    if panel_size > importance.size:
        raise ValueError("panel_size exceeds the number of fitted SNPs")
    indices = np.arange(importance.size, dtype=np.int64)
    ranking = np.lexsort((indices, -importance))
    selected = np.sort(ranking[:panel_size]).astype(np.int64)
    return selected, importance


def _fit_method(
    matrix: FloatArray,
    method: str,
    config: AIMProtocolConfig,
    seed: int,
) -> _MethodFit:
    if config.n_components > min(matrix.shape):
        raise ValueError("n_components exceeds a fitted matrix dimension")
    if method == "pca":
        _, _, right_t = np.linalg.svd(matrix, full_matrices=False)
        return _MethodFit(
            loadings=np.asarray(right_t[: config.n_components].T, dtype=float),
            converged=True,
            status="closed-form PCA",
            n_iter=1,
            final_stationarity=0.0,
        )
    if method == "vp":
        from .vp import fit_vp_spca

        result = fit_vp_spca(
            matrix,
            config.n_components,
            alpha=config.vp_alpha,
            max_iter=config.max_iter,
            tol=config.tol,
            backtracking=True,
            verbose=False,
        )
    elif method == "randomized":
        from .randomized import fit_randomized_vp_spca

        result = fit_randomized_vp_spca(
            matrix,
            config.n_components,
            alpha=config.randomized_alpha,
            oversample=config.randomized_oversample,
            n_power_iter=config.randomized_power_iterations,
            max_iter=config.max_iter,
            tol=config.tol,
            finalize_full_iters=config.randomized_finalize_full_iters,
            random_state=seed,
            verbose=False,
        )
    elif method == "robust":
        from .robust import fit_robust_vp_spca

        result = fit_robust_vp_spca(
            matrix,
            config.n_components,
            alpha=config.robust_alpha,
            huber_delta=config.robust_huber_delta,
            max_iter=config.max_iter,
            tol=config.tol,
            verbose=False,
        )
    else:  # pragma: no cover - protected by config validation
        raise ValueError(f"unknown method: {method}")
    loadings = np.asarray(result.B, dtype=float)
    stationarity = list(result.stationarity)
    if (
        loadings.shape != (matrix.shape[1], config.n_components)
        or not np.isfinite(loadings).all()
    ):
        raise RuntimeError(f"{method} returned invalid loadings")
    final_stationarity = float(stationarity[-1]) if stationarity else float("inf")
    if not np.isfinite(final_stationarity):
        raise RuntimeError(f"{method} returned a non-finite stationarity diagnostic")
    return _MethodFit(
        loadings=loadings,
        converged=bool(result.converged),
        status=str(result.status),
        n_iter=int(result.n_iter),
        final_stationarity=final_stationarity,
    )


def _classification_diagnostics(
    truth: StringArray,
    prediction: StringArray,
    classes: StringArray,
) -> dict[str, Any]:
    index = {label: position for position, label in enumerate(classes.tolist())}
    confusion = np.zeros((classes.size, classes.size), dtype=np.int64)
    for expected, observed in zip(truth.tolist(), prediction.tolist(), strict=True):
        confusion[index[expected], index[observed]] += 1
    supports = confusion.sum(axis=1)
    predicted_counts = confusion.sum(axis=0)
    recalls: list[float] = []
    f1s: list[float] = []
    per_population: list[dict[str, Any]] = []
    for position in range(classes.size):
        true_positive = int(confusion[position, position])
        recall = true_positive / int(supports[position])
        precision = (
            true_positive / int(predicted_counts[position])
            if predicted_counts[position]
            else 0.0
        )
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        recalls.append(float(recall))
        f1s.append(f1)
        per_population.append(
            {
                "population": str(classes[position]),
                "support": int(supports[position]),
                "predicted_count": int(predicted_counts[position]),
                "true_positive": true_positive,
                "precision": float(precision),
                "recall": float(recall),
                "f1": f1,
            }
        )
    return {
        "accuracy": float(np.trace(confusion) / truth.size),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "class_order": classes.tolist(),
        "confusion_matrix_counts": confusion.tolist(),
        "per_population_metrics": per_population,
    }


def _nearest_centroid(
    train_scores: FloatArray,
    train_labels: StringArray,
    test_scores: FloatArray,
) -> tuple[StringArray, StringArray]:
    classes = np.unique(train_labels)
    centroids = np.stack(
        [train_scores[train_labels == label].mean(axis=0) for label in classes],
        axis=0,
    )
    distances = np.sum(
        (test_scores[:, None, :] - centroids[None, :, :]) ** 2,
        axis=2,
    )
    return np.asarray(classes[np.argmin(distances, axis=1)], dtype=str), classes


def _support_jaccard(first: BoolArray, second: BoolArray) -> float:
    union = int(np.count_nonzero(first | second))
    return float(np.count_nonzero(first & second) / union) if union else 1.0


def _evaluate_aim_split_prevalidated(
    data: MultiAutosomeGenotypeData,
    config: AIMProtocolConfig,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    *,
    seed: int,
    capture_coordinates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, BoolArray], list[dict[str, Any]]]:
    """Implementation for inputs whose full data/config contract was checked."""

    seed = _nonnegative_integer(seed, name="seed")
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    n_samples = np.asarray(data.genotypes).shape[0]
    if (
        train.ndim != 1
        or test.ndim != 1
        or min(train.size, test.size) == 0
        or np.any(train < 0)
        or np.any(test < 0)
        or np.any(train >= n_samples)
        or np.any(test >= n_samples)
        or np.unique(train).size != train.size
        or np.unique(test).size != test.size
        or np.intersect1d(train, test).size
        or np.union1d(train, test).size != n_samples
    ):
        raise ValueError(
            "train_indices and test_indices must be an exact disjoint cover"
        )
    labels = np.asarray(data.population_labels, dtype=str)
    if set(np.unique(labels[train])) != set(np.unique(labels)) or set(
        np.unique(labels[test])
    ) != set(np.unique(labels)):
        raise ValueError("every population must appear in train and test")

    raw = np.asarray(data.genotypes, dtype=float)
    preprocessor = fit_smartpca_preprocessor(raw[train])
    train_matrix = transform_smartpca(raw[train], preprocessor)
    test_matrix = transform_smartpca(raw[test], preprocessor)
    retained_original = np.flatnonzero(preprocessor.retained_snps)
    if config.panel_size > retained_original.size:
        raise ValueError("panel_size exceeds training-retained SNP count")

    records: list[dict[str, Any]] = []
    supports: dict[str, BoolArray] = {}
    coordinate_rows: list[dict[str, Any]] = []
    for method in config.methods:
        method_seed = int(
            np.random.SeedSequence(
                [seed, METHOD_SEED_ROLES[method], 17]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        prepanel_fit = _fit_method(train_matrix, method, config, method_seed)
        selected_retained, importance = select_loading_panel(
            prepanel_fit.loadings,
            config.panel_size,
            score=config.loading_score,
        )
        selected_original = retained_original[selected_retained]
        support = np.zeros(raw.shape[1], dtype=bool)
        support[selected_original] = True
        supports[method] = support

        native_retained = np.any(
            np.abs(prepanel_fit.loadings) > config.native_support_threshold,
            axis=1,
        )
        native_count = int(np.count_nonzero(native_retained))
        native_fraction = float(native_count / retained_original.size)
        native_support_sufficient_for_panel = bool(
            native_count >= config.panel_size
        )
        native_degenerate = bool(
            native_count == 0 or native_fraction >= config.near_dense_fraction
        )
        prepanel_qualified = bool(
            prepanel_fit.converged
            and native_support_sufficient_for_panel
            and (method == "pca" or not native_degenerate)
        )

        # The loading-only panel is frozen above, before population labels are
        # used for any metric. This full-width score is the compression
        # reference for the later fixed-budget panel.
        full_train_scores = train_matrix @ prepanel_fit.loadings
        full_test_scores = test_matrix @ prepanel_fit.loadings
        full_predictions, full_classes = _nearest_centroid(
            full_train_scores,
            labels[train],
            full_test_scores,
        )
        full_metrics = _classification_diagnostics(
            labels[test], full_predictions, full_classes
        )

        panel_train = train_matrix[:, selected_retained]
        panel_test = test_matrix[:, selected_retained]
        panel_fit = _fit_method(panel_train, method, config, method_seed)
        train_scores = panel_train @ panel_fit.loadings
        test_scores = panel_test @ panel_fit.loadings
        predictions, classes = _nearest_centroid(
            train_scores,
            labels[train],
            test_scores,
        )
        native_metrics = _classification_diagnostics(labels[test], predictions, classes)

        # A shared closed-form PCA probe isolates panel quality from differences
        # among the native embedding solvers.  It is fitted only after this
        # method's loading-only panel has been frozen.
        probe_fit = _fit_method(panel_train, "pca", config, method_seed)
        probe_train_scores = panel_train @ probe_fit.loadings
        probe_test_scores = panel_test @ probe_fit.loadings
        probe_predictions, probe_classes = _nearest_centroid(
            probe_train_scores,
            labels[train],
            probe_test_scores,
        )
        probe_metrics = _classification_diagnostics(
            labels[test], probe_predictions, probe_classes
        )
        if not prepanel_fit.converged:
            prepanel_qualification_status = "prepanel_solver_not_converged"
        elif not native_support_sufficient_for_panel:
            prepanel_qualification_status = (
                "prepanel_native_support_below_panel_size"
            )
        elif method != "pca" and native_degenerate:
            prepanel_qualification_status = "prepanel_native_support_degenerate"
        else:
            prepanel_qualification_status = "qualified"
        native_panel_qualification_status = (
            "qualified"
            if panel_fit.converged
            else "native_panel_refit_solver_not_converged"
        )
        if prepanel_qualification_status != "qualified":
            qualification_status = prepanel_qualification_status
        elif native_panel_qualification_status != "qualified":
            qualification_status = native_panel_qualification_status
        elif not probe_fit.converged:  # closed-form PCA, retained as a fail-closed gate
            qualification_status = "common_pca_probe_not_converged"
        else:
            qualification_status = "qualified"
        selected_ids = np.asarray(data.snp_ids, dtype=str)[selected_original]
        records.append(
            {
                "seed": seed,
                "method": method,
                "population_design": config.population_design,
                "population_label_semantics": "metadata_population_group",
                "population_label_is_race": False,
                "input_scope": data.input_scope,
                "non_smoke_genome_scope": (
                    data.input_scope != "synthetic_multi_autosome_smoke"
                ),
                "train_samples": int(train.size),
                "test_samples": int(test.size),
                "input_snps": int(raw.shape[1]),
                "training_retained_snps": int(retained_original.size),
                "panel_size": int(config.panel_size),
                "panel_selection_rule": (
                    f"train_only_{config.loading_score}_across_"
                    f"{config.n_components}_model_loadings_top_k"
                ),
                "panel_selection_used_population_labels": False,
                "preprocessing_fit_scope": "train_only",
                "prepanel_component_fit_scope": "train_only",
                "native_panel_refit_scope": "train_only_selected_panel",
                "common_pca_probe_scope": "train_only_selected_panel",
                "classifier_fit_scope": "train_centroids_only_after_panel_freeze",
                "test_used_for_preprocessing_or_model_selection": False,
                "prepanel_solver_converged": prepanel_fit.converged,
                "prepanel_solver_status": prepanel_fit.status,
                "prepanel_solver_n_iter": prepanel_fit.n_iter,
                "prepanel_final_stationarity": prepanel_fit.final_stationarity,
                "prepanel_native_support_count": native_count,
                "prepanel_native_support_fraction": native_fraction,
                "prepanel_native_support_sufficient_for_panel": (
                    native_support_sufficient_for_panel
                ),
                "prepanel_native_support_degenerate": native_degenerate,
                "prepanel_qualified": prepanel_qualified,
                "prepanel_qualification_status": prepanel_qualification_status,
                "full_width_native_accuracy": full_metrics["accuracy"],
                "full_width_native_balanced_accuracy": full_metrics[
                    "balanced_accuracy"
                ],
                "full_width_native_macro_f1": full_metrics["macro_f1"],
                "full_width_native_per_population_metrics": full_metrics[
                    "per_population_metrics"
                ],
                "full_width_native_test_predictions_sha256": array_sha256(
                    full_predictions
                ),
                "native_embedding_method": method,
                "native_panel_refit_solver_converged": panel_fit.converged,
                "native_panel_refit_solver_status": panel_fit.status,
                "native_panel_refit_solver_n_iter": panel_fit.n_iter,
                "native_panel_refit_final_stationarity": panel_fit.final_stationarity,
                "native_panel_refit_qualified": bool(panel_fit.converged),
                "native_panel_refit_qualification_status": (
                    native_panel_qualification_status
                ),
                "common_pca_probe_solver_converged": probe_fit.converged,
                "common_pca_probe_solver_status": probe_fit.status,
                "overall_qualified": bool(
                    prepanel_qualified and panel_fit.converged and probe_fit.converged
                ),
                "qualification_status": qualification_status,
                "selected_snp_ids": selected_ids.tolist(),
                "selected_snp_ids_sha256": array_sha256(selected_ids),
                "selected_snp_indices_sha256": array_sha256(selected_original),
                "prepanel_loadings_sha256": array_sha256(prepanel_fit.loadings),
                "native_panel_loadings_sha256": array_sha256(panel_fit.loadings),
                "common_pca_probe_loadings_sha256": array_sha256(probe_fit.loadings),
                "loading_importance_sha256": array_sha256(importance),
                "train_indices_sha256": array_sha256(train),
                "test_indices_sha256": array_sha256(test),
                "native_test_predictions_sha256": array_sha256(predictions),
                "common_pca_probe_test_predictions_sha256": array_sha256(
                    probe_predictions
                ),
                "native_embedding_accuracy": native_metrics["accuracy"],
                "native_embedding_balanced_accuracy": native_metrics[
                    "balanced_accuracy"
                ],
                "native_embedding_macro_f1": native_metrics["macro_f1"],
                "native_embedding_class_order": native_metrics["class_order"],
                "native_embedding_confusion_matrix_counts": native_metrics[
                    "confusion_matrix_counts"
                ],
                "native_embedding_per_population_metrics": native_metrics[
                    "per_population_metrics"
                ],
                "common_pca_probe_accuracy": probe_metrics["accuracy"],
                "common_pca_probe_balanced_accuracy": probe_metrics[
                    "balanced_accuracy"
                ],
                "common_pca_probe_macro_f1": probe_metrics["macro_f1"],
                "common_pca_probe_class_order": probe_metrics["class_order"],
                "common_pca_probe_confusion_matrix_counts": probe_metrics[
                    "confusion_matrix_counts"
                ],
                "common_pca_probe_per_population_metrics": probe_metrics[
                    "per_population_metrics"
                ],
            }
        )
        if capture_coordinates:
            coordinate_evaluations = (
                ("full_width_native", full_test_scores, full_predictions),
                ("top_k_native_embedding", test_scores, predictions),
                ("top_k_common_pca_probe", probe_test_scores, probe_predictions),
            )
            for (
                evaluation,
                coordinates,
                evaluation_predictions,
            ) in coordinate_evaluations:
                for sample_index, truth, prediction, coordinate in zip(
                    test,
                    labels[test],
                    evaluation_predictions,
                    coordinates[:, :2],
                    strict=True,
                ):
                    coordinate_rows.append(
                        {
                            "registered_seed_policy": "first_config_seed_not_best",
                            "seed": seed,
                            "method": method,
                            "evaluation": evaluation,
                            "sample_index": int(sample_index),
                            "sample_id": str(data.sample_ids[sample_index]),
                            "truth": str(truth),
                            "prediction": str(prediction),
                            "component_1": float(coordinate[0]),
                            "component_2": float(coordinate[1]),
                            "qualification_status": qualification_status,
                            "overall_qualified": bool(
                                qualification_status == "qualified"
                            ),
                        }
                    )
    pca_support = supports["pca"]
    for record in records:
        record["panel_jaccard_with_equal_budget_pca"] = _support_jaccard(
            supports[str(record["method"])], pca_support
        )
    return records, supports, coordinate_rows


def evaluate_aim_split(
    data: MultiAutosomeGenotypeData,
    config: AIMProtocolConfig,
    train_indices: ArrayLike,
    test_indices: ArrayLike,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, BoolArray]]:
    """Fit panels on explicit training rows and evaluate untouched test rows."""

    validate_multiautosome_input(data)
    _validate_config(config, data)
    records, supports, _ = _evaluate_aim_split_prevalidated(
        data,
        config,
        train_indices,
        test_indices,
        seed=seed,
    )
    return records, supports


def _input_identity(data: MultiAutosomeGenotypeData) -> dict[str, Any]:
    chromosomes = np.asarray(
        [_normalized_autosome(value) for value in data.chromosomes], dtype=str
    )
    return {
        "source_manifest_sha256": data.source_manifest_sha256,
        "input_scope": data.input_scope,
        "ld_pruned_declared": bool(data.ld_pruned),
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "autosomes_present": sorted(np.unique(chromosomes).tolist(), key=int),
        "genotypes_sha256": array_sha256(np.asarray(data.genotypes, dtype=float)),
        "sample_ids_sha256": array_sha256(np.asarray(data.sample_ids, dtype=str)),
        "snp_ids_sha256": array_sha256(np.asarray(data.snp_ids, dtype=str)),
        "chromosomes_sha256": array_sha256(chromosomes),
        "population_labels_sha256": array_sha256(
            np.asarray(data.population_labels, dtype=str)
        ),
    }


def _source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _solver_lineage(config: AIMProtocolConfig) -> dict[str, dict[str, Any]]:
    package_directory = Path(__file__).resolve().parent
    repository_root = package_directory.parents[1]
    panel_source = package_directory / "genomics_aim.py"
    panel_source_sha256 = _source_file_sha256(panel_source)
    definitions = {
        "pca": {
            "maintained_callable": "numpy.linalg.svd via rspca.genomics_aim._fit_method",
            "source_path": "src/rspca/genomics_aim.py",
            "source_file": panel_source,
            "mathematical_objective": (
                "closed_form_rank_k_maximum_variance_PCA_without_sparsity_penalty"
            ),
            "qualification_role": (
                "mandatory_equal_budget_reference_and_common_probe_requires_at_"
                "least_panel_size_native_support"
            ),
            "legacy_sources": (
                "gene.ipynb",
                "gene large scale.ipynb",
            ),
        },
        "vp": {
            "maintained_callable": "rspca.vp.fit_vp_spca",
            "source_path": "src/rspca/vp.py",
            "source_file": package_directory / "vp.py",
            "mathematical_objective": (
                "0.5_frobenius_reconstruction_plus_alpha_l1_loading_penalty_"
                "with_orthonormal_A"
            ),
            "qualification_role": (
                "candidate_requires_prepanel_convergence_at_least_panel_size_"
                "native_support_nondegenerate_native_support_and_panel_refit_"
                "convergence"
            ),
            "legacy_sources": (
                "spca_variable_projection.py",
                "gene.ipynb",
                "gene large scale.ipynb",
            ),
        },
        "randomized": {
            "maintained_callable": "rspca.randomized.fit_randomized_vp_spca",
            "source_path": "src/rspca/randomized.py",
            "source_file": package_directory / "randomized.py",
            "mathematical_objective": (
                "same_VP_l1_objective_on_randomized_left_range_sketch_with_"
                "configured_full_data_refinement"
            ),
            "qualification_role": (
                "candidate_requires_prepanel_convergence_at_least_panel_size_"
                "native_support_nondegenerate_native_support_and_panel_refit_"
                "convergence"
            ),
            "legacy_sources": (
                "randomized_spca_vp.py",
                "gene.ipynb",
                "gene large scale.ipynb",
            ),
        },
        "robust": {
            "maintained_callable": "rspca.robust.fit_robust_vp_spca",
            "source_path": "src/rspca/robust.py",
            "source_file": package_directory / "robust.py",
            "mathematical_objective": (
                "split_Huber_0.5_clean_frobenius_plus_alpha_l1_loadings_plus_"
                "kappa_l1_sparse_corruption"
            ),
            "qualification_role": (
                "candidate_requires_prepanel_convergence_at_least_panel_size_"
                "native_support_nondegenerate_native_support_and_panel_refit_"
                "convergence"
            ),
            "legacy_sources": ("robust_spca_vp.py",),
        },
    }
    lineage: dict[str, dict[str, Any]] = {}
    for method in config.methods:
        definition = definitions[method]
        source_file = Path(definition["source_file"])
        try:
            source_file.relative_to(repository_root)
        except ValueError as error:  # pragma: no cover - installation corruption
            raise RuntimeError("solver source is outside the repository") from error
        legacy_sources = tuple(str(value) for value in definition["legacy_sources"])
        lineage[method] = {
            "origin_main_commit": LEGACY_GITHUB_BASELINE_COMMIT,
            "legacy_sources": [
                {
                    "path": source,
                    "sha256": LEGACY_SOURCE_SHA256[source],
                }
                for source in legacy_sources
            ],
            "maintained_callable": str(definition["maintained_callable"]),
            "source_path": str(definition["source_path"]),
            "source_sha256": _source_file_sha256(source_file),
            "mathematical_objective": str(definition["mathematical_objective"]),
            "qualification_role": str(definition["qualification_role"]),
            "panel_ranking_callable": "rspca.genomics_aim.select_loading_panel",
            "panel_ranking_source_path": "src/rspca/genomics_aim.py",
            "panel_ranking_source_sha256": panel_source_sha256,
            "panel_ranking_rule": (
                f"train_only_{config.loading_score}_importance_across_"
                f"{config.n_components}_own_solver_loadings_exact_top_{config.panel_size}"
            ),
            "panel_selection_uses_population_labels": False,
            "supervised_feature_ranking": False,
        }
    return lineage


def _sample_std(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("sample standard deviation requires at least two values")
    return float(np.std(array, ddof=1))


def _paired_bootstrap_interval(
    differences: Sequence[float],
    bootstrap_indices: IntArray,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap differences must be finite with size >= 2")
    if (
        bootstrap_indices.ndim != 2
        or bootstrap_indices.shape[1] != values.size
        or np.any(bootstrap_indices < 0)
        or np.any(bootstrap_indices >= values.size)
    ):
        raise ValueError("bootstrap_indices do not match the paired differences")
    bootstrap_means = np.mean(values[bootstrap_indices], axis=1)
    low, high = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(low), float(high)


def run_aim_protocol(
    data: MultiAutosomeGenotypeData,
    config: AIMProtocolConfig,
) -> AIMProtocolResult:
    """Run paired AIM evaluation with a common SNP-panel budget."""

    validate_multiautosome_input(data)
    _validate_config(config, data)
    records: list[dict[str, Any]] = []
    support_history: dict[str, list[BoolArray]] = {
        method: [] for method in config.methods
    }
    registered_coordinates: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(config.seeds):
        train, test = stratified_split(
            data.population_labels,
            config.test_fraction,
            seed,
        )
        split_records, split_supports, split_coordinates = (
            _evaluate_aim_split_prevalidated(
                data,
                config,
                train,
                test,
                seed=seed,
                capture_coordinates=seed_index == 0,
            )
        )
        records.extend(split_records)
        registered_coordinates.extend(split_coordinates)
        for method, support in split_supports.items():
            support_history[method].append(support)

    by_key = {
        (int(record["seed"]), str(record["method"])): record for record in records
    }
    for record in records:
        pca = by_key[(int(record["seed"]), "pca")]
        for embedding in ("native_embedding", "common_pca_probe"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                key = f"{embedding}_{metric}"
                record[f"paired_delta_vs_pca_{key}"] = float(record[key] - pca[key])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            record[f"native_panel_retention_delta_{metric}"] = float(
                record[f"native_embedding_{metric}"]
                - record[f"full_width_native_{metric}"]
            )
            record[f"common_pca_probe_retention_delta_vs_full_width_pca_{metric}"] = (
                float(
                    record[f"common_pca_probe_{metric}"]
                    - pca[f"full_width_native_{metric}"]
                )
            )

    bootstrap_rng = np.random.default_rng(config.bootstrap_seed)
    bootstrap_indices = np.asarray(
        bootstrap_rng.integers(
            0,
            len(config.seeds),
            size=(config.bootstrap_resamples, len(config.seeds)),
        ),
        dtype=np.int64,
    )

    summaries: list[dict[str, Any]] = []
    frequencies: dict[str, FloatArray] = {}
    for method in config.methods:
        method_records = [by_key[(seed, method)] for seed in config.seeds]
        stacked = np.stack(support_history[method], axis=0)
        frequency = np.mean(stacked, axis=0)
        frequencies[method] = np.asarray(frequency, dtype=float)
        pairwise = [
            _support_jaccard(stacked[first], stacked[second])
            for first, second in combinations(range(stacked.shape[0]), 2)
        ]
        qualified_splits = int(
            np.count_nonzero([record["overall_qualified"] for record in method_records])
        )
        if qualified_splits == len(method_records):
            qualification_status = "all_splits_qualified"
        elif qualified_splits == 0:
            qualification_status = "no_splits_qualified"
        else:
            qualification_status = "partially_qualified"
        summary: dict[str, Any] = {
            "method": method,
            "splits": len(method_records),
            "panel_size": config.panel_size,
            "mean_native_embedding_accuracy": float(
                np.mean(
                    [record["native_embedding_accuracy"] for record in method_records]
                )
            ),
            "mean_native_embedding_balanced_accuracy": float(
                np.mean(
                    [
                        record["native_embedding_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "mean_native_embedding_macro_f1": float(
                np.mean(
                    [record["native_embedding_macro_f1"] for record in method_records]
                )
            ),
            "mean_common_pca_probe_accuracy": float(
                np.mean(
                    [record["common_pca_probe_accuracy"] for record in method_records]
                )
            ),
            "mean_common_pca_probe_balanced_accuracy": float(
                np.mean(
                    [
                        record["common_pca_probe_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "mean_common_pca_probe_macro_f1": float(
                np.mean(
                    [record["common_pca_probe_macro_f1"] for record in method_records]
                )
            ),
            "mean_full_width_native_balanced_accuracy": float(
                np.mean(
                    [
                        record["full_width_native_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "mean_native_panel_retention_delta_balanced_accuracy": float(
                np.mean(
                    [
                        record["native_panel_retention_delta_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "mean_common_pca_probe_retention_delta_vs_full_width_pca_balanced_accuracy": float(
                np.mean(
                    [
                        record[
                            "common_pca_probe_retention_delta_vs_full_width_pca_balanced_accuracy"
                        ]
                        for record in method_records
                    ]
                )
            ),
            "mean_paired_delta_vs_pca_native_balanced_accuracy": float(
                np.mean(
                    [
                        record["paired_delta_vs_pca_native_embedding_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "mean_paired_delta_vs_pca_common_probe_balanced_accuracy": float(
                np.mean(
                    [
                        record["paired_delta_vs_pca_common_pca_probe_balanced_accuracy"]
                        for record in method_records
                    ]
                )
            ),
            "prepanel_convergence_fraction": float(
                np.mean(
                    [record["prepanel_solver_converged"] for record in method_records]
                )
            ),
            "native_panel_refit_convergence_fraction": float(
                np.mean(
                    [
                        record["native_panel_refit_solver_converged"]
                        for record in method_records
                    ]
                )
            ),
            "overall_qualification_fraction": float(
                np.mean([record["overall_qualified"] for record in method_records])
            ),
            "mean_prepanel_native_support_fraction": float(
                np.mean(
                    [
                        record["prepanel_native_support_fraction"]
                        for record in method_records
                    ]
                )
            ),
            "minimum_prepanel_native_support_count": int(
                min(
                    int(record["prepanel_native_support_count"])
                    for record in method_records
                )
            ),
            "all_splits_native_support_at_least_panel_size": bool(
                all(
                    int(record["prepanel_native_support_count"])
                    >= config.panel_size
                    for record in method_records
                )
            ),
            "mean_panel_jaccard_with_equal_budget_pca": float(
                np.mean(
                    [
                        record["panel_jaccard_with_equal_budget_pca"]
                        for record in method_records
                    ]
                )
            ),
            "mean_pairwise_panel_jaccard_across_splits": float(np.mean(pairwise)),
            "minimum_pairwise_panel_jaccard_across_splits": float(np.min(pairwise)),
            "snps_selected_in_every_split": int(np.count_nonzero(frequency == 1.0)),
            "snps_selected_in_at_least_80pct_splits": int(
                np.count_nonzero(frequency >= 0.8)
            ),
            "selection_frequency_sha256": array_sha256(frequency),
            "qualified_splits": qualified_splits,
            "qualification_status": qualification_status,
            "bootstrap_ci_qualification_status": (
                "qualified"
                if qualification_status == "all_splits_qualified"
                else "descriptive_only_unqualified_solver_splits"
            ),
            "bootstrap_interval": "paired_seed_percentile_95",
            "bootstrap_resamples": config.bootstrap_resamples,
            "bootstrap_seed": config.bootstrap_seed,
            "runtime_bootstrapped": False,
        }
        for evaluation in (
            "full_width_native",
            "native_embedding",
            "common_pca_probe",
        ):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                values = [
                    float(record[f"{evaluation}_{metric}"]) for record in method_records
                ]
                summary[f"std_{evaluation}_{metric}"] = _sample_std(values)
        for evaluation in ("native_embedding", "common_pca_probe"):
            for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
                key = f"paired_delta_vs_pca_{evaluation}_{metric}"
                differences = [float(record[key]) for record in method_records]
                low, high = _paired_bootstrap_interval(
                    differences,
                    bootstrap_indices,
                )
                summary[f"mean_{key}"] = float(np.mean(differences))
                summary[f"{key}_bootstrap_ci95_low"] = low
                summary[f"{key}_bootstrap_ci95_high"] = high
        summaries.append(summary)

    identity = _input_identity(data)
    solver_lineage = _solver_lineage(config)
    result = AIMProtocolResult(
        config=config,
        input_identity=identity,
        split_records=tuple(records),
        method_summaries=tuple(summaries),
        selection_frequencies=frequencies,
        registered_coordinates=tuple(registered_coordinates),
        solver_lineage=solver_lineage,
        result_sha256=_protocol_result_sha256(
            config=config,
            input_identity=identity,
            split_records=records,
            method_summaries=summaries,
            selection_frequencies=frequencies,
            registered_coordinates=registered_coordinates,
            solver_lineage=solver_lineage,
        ),
    )
    validate_aim_protocol_result(result, data)
    return result


def _reject_nonfinite(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"non-finite numeric value at {path}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")
        return
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        raise ValueError(f"non-finite numeric value at {path}")


def _protocol_result_sha256(
    *,
    config: AIMProtocolConfig,
    input_identity: Mapping[str, Any],
    split_records: Sequence[Mapping[str, Any]],
    method_summaries: Sequence[Mapping[str, Any]],
    selection_frequencies: Mapping[str, FloatArray],
    registered_coordinates: Sequence[Mapping[str, Any]],
    solver_lineage: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = {
        "config": asdict(config),
        "input_identity": dict(input_identity),
        "split_records": list(split_records),
        "method_summaries": list(method_summaries),
        "selection_frequency_hashes": {
            method: array_sha256(values)
            for method, values in selection_frequencies.items()
        },
        "registered_coordinates": list(registered_coordinates),
        "solver_lineage": dict(solver_lineage),
    }
    _reject_nonfinite(payload, path="result_digest_payload")
    return canonical_sha256(payload)


def validate_aim_protocol_result(
    result: AIMProtocolResult,
    data: MultiAutosomeGenotypeData,
) -> None:
    """Authenticate internal result consistency before any artifact is written."""

    if not isinstance(result, AIMProtocolResult):
        raise TypeError("result must be an AIMProtocolResult")
    validate_multiautosome_input(data)
    _validate_config(result.config, data)
    expected_identity = _input_identity(data)
    if dict(result.input_identity) != expected_identity:
        raise ValueError("result.input_identity does not exactly match supplied data")
    _reject_nonfinite(asdict(result.config), path="result.config")
    _reject_nonfinite(result.input_identity, path="result.input_identity")
    _reject_nonfinite(result.split_records, path="result.split_records")
    _reject_nonfinite(result.method_summaries, path="result.method_summaries")
    _reject_nonfinite(
        result.selection_frequencies,
        path="result.selection_frequencies",
    )
    _reject_nonfinite(
        result.registered_coordinates,
        path="result.registered_coordinates",
    )
    _reject_nonfinite(result.solver_lineage, path="result.solver_lineage")
    expected_lineage = _solver_lineage(result.config)
    if dict(result.solver_lineage) != expected_lineage:
        raise ValueError("solver_lineage does not match the current repository sources")

    expected_record_keys = {
        (int(seed), method)
        for seed in result.config.seeds
        for method in result.config.methods
    }
    records_by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    for record in result.split_records:
        key = (int(record.get("seed", -1)), str(record.get("method", "")))
        if key in records_by_key:
            raise ValueError(f"duplicate split record: {key}")
        records_by_key[key] = record
        method = key[1]
        native_support_count = int(record.get("prepanel_native_support_count", -1))
        native_support_sufficient = bool(
            native_support_count >= result.config.panel_size
        )
        if (
            "prepanel_native_support_sufficient_for_panel" not in record
            or bool(record["prepanel_native_support_sufficient_for_panel"])
            != native_support_sufficient
        ):
            raise ValueError(
                "prepanel native-support panel-size gate is inconsistent"
            )
        if not bool(record.get("prepanel_solver_converged")):
            expected_prepanel_status = "prepanel_solver_not_converged"
        elif not native_support_sufficient:
            expected_prepanel_status = "prepanel_native_support_below_panel_size"
        elif method != "pca" and bool(record.get("prepanel_native_support_degenerate")):
            expected_prepanel_status = "prepanel_native_support_degenerate"
        else:
            expected_prepanel_status = "qualified"
        if record.get("prepanel_qualification_status") != expected_prepanel_status:
            raise ValueError("prepanel qualification status is inconsistent")
        expected_prepanel_qualified = expected_prepanel_status == "qualified"
        if bool(record.get("prepanel_qualified")) != expected_prepanel_qualified:
            raise ValueError("prepanel qualification boolean is inconsistent")
        expected_panel_status = (
            "qualified"
            if bool(record.get("native_panel_refit_solver_converged"))
            else "native_panel_refit_solver_not_converged"
        )
        if (
            record.get("native_panel_refit_qualification_status")
            != expected_panel_status
        ):
            raise ValueError("native panel qualification status is inconsistent")
        if expected_prepanel_status != "qualified":
            expected_status = expected_prepanel_status
        elif expected_panel_status != "qualified":
            expected_status = expected_panel_status
        elif not bool(record.get("common_pca_probe_solver_converged")):
            expected_status = "common_pca_probe_not_converged"
        else:
            expected_status = "qualified"
        expected_overall = expected_status == "qualified"
        if (
            record.get("qualification_status") != expected_status
            or bool(record.get("overall_qualified")) != expected_overall
        ):
            raise ValueError("overall qualification status is inconsistent")
    if set(records_by_key) != expected_record_keys:
        raise ValueError("split records do not exactly cover configured seeds/methods")

    summary_methods = [
        str(summary.get("method", "")) for summary in result.method_summaries
    ]
    if summary_methods != list(result.config.methods):
        raise ValueError("method summaries do not match configured method order")

    if set(result.selection_frequencies) != set(result.config.methods):
        raise ValueError("selection frequency methods do not match config.methods")
    snp_ids = np.asarray(data.snp_ids, dtype=str)
    snp_index = {snp: index for index, snp in enumerate(snp_ids.tolist())}
    summary_by_method = {
        str(summary["method"]): summary for summary in result.method_summaries
    }
    for method in result.config.methods:
        frequency = np.asarray(result.selection_frequencies[method], dtype=float)
        if frequency.shape != (snp_ids.size,):
            raise ValueError(f"selection frequency shape mismatch for {method}")
        if not np.isfinite(frequency).all() or np.any(
            (frequency < 0.0) | (frequency > 1.0)
        ):
            raise ValueError(
                f"selection frequencies for {method} must be finite and in [0, 1]"
            )
        counts = np.zeros(snp_ids.size, dtype=np.int64)
        for seed in result.config.seeds:
            record = records_by_key[(seed, method)]
            selected_ids = [str(value) for value in record["selected_snp_ids"]]
            if len(selected_ids) != result.config.panel_size or len(
                set(selected_ids)
            ) != len(selected_ids):
                raise ValueError(f"{method} split {seed} has an invalid fixed panel")
            try:
                selected_indices = [snp_index[snp] for snp in selected_ids]
            except KeyError as error:
                raise ValueError(
                    f"unknown selected SNP ID for {method}: {error}"
                ) from error
            counts[selected_indices] += 1
        expected_frequency = counts.astype(float) / len(result.config.seeds)
        if not np.array_equal(frequency, expected_frequency):
            raise ValueError(
                f"selection frequencies disagree with split panels for {method}"
            )
        if summary_by_method[method].get("selection_frequency_sha256") != array_sha256(
            frequency
        ):
            raise ValueError(f"selection frequency hash mismatch for {method}")
        qualified_splits = int(
            np.count_nonzero(
                [
                    records_by_key[(seed, method)]["overall_qualified"]
                    for seed in result.config.seeds
                ]
            )
        )
        expected_summary_status = (
            "all_splits_qualified"
            if qualified_splits == len(result.config.seeds)
            else (
                "no_splits_qualified"
                if qualified_splits == 0
                else "partially_qualified"
            )
        )
        summary = summary_by_method[method]
        minimum_native_support = min(
            int(records_by_key[(seed, method)]["prepanel_native_support_count"])
            for seed in result.config.seeds
        )
        if (
            int(summary.get("qualified_splits", -1)) != qualified_splits
            or summary.get("qualification_status") != expected_summary_status
            or int(summary.get("minimum_prepanel_native_support_count", -1))
            != minimum_native_support
            or bool(
                summary.get("all_splits_native_support_at_least_panel_size")
            )
            != bool(minimum_native_support >= result.config.panel_size)
            or int(summary.get("bootstrap_resamples", -1))
            != result.config.bootstrap_resamples
            or int(summary.get("bootstrap_seed", -1)) != result.config.bootstrap_seed
            or summary.get("runtime_bootstrapped") is not False
        ):
            raise ValueError(
                f"summary qualification/bootstrap metadata mismatch for {method}"
            )

    registered_seed = result.config.seeds[0]
    expected_evaluations = {
        "full_width_native",
        "top_k_native_embedding",
        "top_k_common_pca_probe",
    }
    sample_lookup = {
        str(sample_id): (index, str(data.population_labels[index]))
        for index, sample_id in enumerate(data.sample_ids)
    }
    _, registered_test_indices = stratified_split(
        data.population_labels,
        result.config.test_fraction,
        registered_seed,
    )
    registered_test_ids = {
        str(data.sample_ids[index]) for index in registered_test_indices
    }
    known_populations = set(np.asarray(data.population_labels, dtype=str).tolist())
    coordinate_keys: set[tuple[str, str, str]] = set()
    for row in result.registered_coordinates:
        if int(row.get("seed", -1)) != registered_seed:
            raise ValueError(
                "registered coordinates must use the first configured seed"
            )
        method = str(row.get("method", ""))
        evaluation = str(row.get("evaluation", ""))
        sample_id = str(row.get("sample_id", ""))
        if (
            method not in result.config.methods
            or evaluation not in expected_evaluations
        ):
            raise ValueError("registered coordinate method/evaluation is invalid")
        if row.get("registered_seed_policy") != "first_config_seed_not_best":
            raise ValueError("registered coordinate seed policy is invalid")
        key = (method, evaluation, sample_id)
        if key in coordinate_keys:
            raise ValueError(f"duplicate registered coordinate row: {key}")
        coordinate_keys.add(key)
        if sample_id not in sample_lookup:
            raise ValueError(f"unknown coordinate sample ID: {sample_id}")
        if sample_id not in registered_test_ids:
            raise ValueError(
                "registered coordinates must contain held-out samples only"
            )
        expected_index, expected_truth = sample_lookup[sample_id]
        if int(row.get("sample_index", -1)) != expected_index:
            raise ValueError("coordinate sample index/ID mismatch")
        if str(row.get("truth", "")) != expected_truth:
            raise ValueError(
                "coordinate truth does not match supplied population metadata"
            )
        if str(row.get("prediction", "")) not in known_populations:
            raise ValueError("coordinate prediction is not a known population")
        record = records_by_key[(registered_seed, method)]
        if str(row.get("qualification_status")) != str(record["qualification_status"]):
            raise ValueError("coordinate qualification status mismatch")
        if bool(row.get("overall_qualified")) != bool(record["overall_qualified"]):
            raise ValueError("coordinate qualification boolean mismatch")
    registered_test_samples = int(
        records_by_key[(registered_seed, "pca")]["test_samples"]
    )
    expected_coordinate_count = (
        len(result.config.methods) * len(expected_evaluations) * registered_test_samples
    )
    if len(result.registered_coordinates) != expected_coordinate_count:
        raise ValueError("registered coordinate rows are incomplete")
    prediction_hash_fields = {
        "full_width_native": "full_width_native_test_predictions_sha256",
        "top_k_native_embedding": "native_test_predictions_sha256",
        "top_k_common_pca_probe": "common_pca_probe_test_predictions_sha256",
    }
    for method in result.config.methods:
        record = records_by_key[(registered_seed, method)]
        for evaluation, hash_field in prediction_hash_fields.items():
            rows = {
                int(row["sample_index"]): row
                for row in result.registered_coordinates
                if row["method"] == method and row["evaluation"] == evaluation
            }
            if set(rows) != set(registered_test_indices.tolist()):
                raise ValueError("registered coordinate test coverage is incomplete")
            predictions = np.asarray(
                [
                    str(rows[int(index)]["prediction"])
                    for index in registered_test_indices
                ],
                dtype=str,
            )
            if array_sha256(predictions) != record[hash_field]:
                raise ValueError("coordinate predictions disagree with split record")

    recomputed = _protocol_result_sha256(
        config=result.config,
        input_identity=result.input_identity,
        split_records=result.split_records,
        method_summaries=result.method_summaries,
        selection_frequencies=result.selection_frequencies,
        registered_coordinates=result.registered_coordinates,
        solver_lineage=result.solver_lineage,
    )
    if result.result_sha256 != recomputed:
        raise ValueError("result_sha256 does not match recomputed protocol result")


__all__ = [
    "AIMProtocolConfig",
    "AIMProtocolResult",
    "MultiAutosomeGenotypeData",
    "SmartPCAPreprocessor",
    "evaluate_aim_split",
    "fit_smartpca_preprocessor",
    "loading_importance",
    "run_aim_protocol",
    "select_loading_panel",
    "stratified_split",
    "transform_smartpca",
    "validate_aim_protocol_result",
    "validate_multiautosome_input",
]

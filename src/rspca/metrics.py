"""Metrics used by the reproducible SPCA experiments.

The functions in this module deliberately operate on plain NumPy arrays.  They
do not fit models and they never choose a threshold from the evaluation target;
callers must make that choice on training/validation data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment


_EPS = np.finfo(float).eps


def _matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _paired_matrices(
    reference: ArrayLike, estimate: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    reference_array = _matrix(reference, "reference")
    estimate_array = _matrix(estimate, "estimate")
    if reference_array.shape != estimate_array.shape:
        raise ValueError(
            "reference and estimate must have the same shape; got "
            f"{reference_array.shape} and {estimate_array.shape}"
        )
    return reference_array, estimate_array


def relative_reconstruction_error(reference: ArrayLike, estimate: ArrayLike) -> float:
    """Return ``||reference - estimate||_F / ||reference||_F``.

    A perfect reconstruction of an all-zero reference has error zero.  Any
    non-zero reconstruction of an all-zero reference has infinite relative
    error because the mathematical denominator is zero.
    """

    reference_array, estimate_array = _paired_matrices(reference, estimate)
    denominator = float(np.linalg.norm(reference_array, ord="fro"))
    numerator = float(np.linalg.norm(reference_array - estimate_array, ord="fro"))
    if denominator <= _EPS:
        return 0.0 if numerator <= _EPS else float("inf")
    return numerator / denominator


def explained_variance_ratio(
    reference: ArrayLike,
    estimate: ArrayLike,
    *,
    center: bool = True,
) -> float:
    """Return reconstruction-based explained variance.

    This is ``1 - SSE / SST``.  By default ``SST`` is computed after centering
    every feature of the reference data.  Negative values are retained because
    they reveal reconstructions that are worse than the mean-only baseline.
    """

    reference_array, estimate_array = _paired_matrices(reference, estimate)
    baseline = (
        reference_array - reference_array.mean(axis=0, keepdims=True)
        if center
        else reference_array
    )
    total = float(np.sum(baseline * baseline))
    residual = reference_array - estimate_array
    unexplained = float(np.sum(residual * residual))
    if total <= _EPS:
        return 1.0 if unexplained <= _EPS else 0.0
    return 1.0 - unexplained / total


def sparsity_ratio(values: ArrayLike, *, threshold: float = 1e-10) -> float:
    """Fraction of entries whose absolute magnitude does not exceed threshold."""

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("values contains NaN or infinite values")
    return float(np.mean(np.abs(array) <= threshold))


def _as_support(values: ArrayLike, threshold: float) -> NDArray[np.bool_]:
    array = np.asarray(values)
    if array.dtype == np.bool_:
        return array.astype(bool, copy=False)
    numeric = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("support values contain NaN or infinite values")
    return np.abs(numeric) > threshold


def support_precision_recall_f1(
    true_support: ArrayLike,
    estimated_support: ArrayLike,
    *,
    threshold: float = 1e-10,
) -> dict[str, float | int]:
    """Evaluate entrywise support recovery.

    Numeric arrays are converted to support masks using ``abs(x) > threshold``.
    For the empty/empty case precision, recall, and F1 are defined as one.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    truth = _as_support(true_support, threshold)
    estimate = _as_support(estimated_support, threshold)
    if truth.shape != estimate.shape:
        raise ValueError(
            f"support masks must have the same shape; got {truth.shape} and {estimate.shape}"
        )

    tp = int(np.count_nonzero(truth & estimate))
    fp = int(np.count_nonzero(~truth & estimate))
    fn = int(np.count_nonzero(truth & ~estimate))
    tn = int(np.count_nonzero(~truth & ~estimate))

    precision = tp / (tp + fp) if tp + fp else (1.0 if tp + fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def _orthonormal_basis(values: ArrayLike, name: str) -> NDArray[np.float64]:
    matrix = _matrix(values, name)
    if matrix.shape[1] == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.empty((matrix.shape[0], 0), dtype=float)
    tolerance = max(matrix.shape) * singular_values[0] * np.finfo(float).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    return left[:, :rank]


def principal_angles(
    reference_basis: ArrayLike,
    estimated_basis: ArrayLike,
    *,
    degrees: bool = False,
) -> NDArray[np.float64]:
    """Return canonical angles between two column spaces in ascending order."""

    reference = _orthonormal_basis(reference_basis, "reference_basis")
    estimate = _orthonormal_basis(estimated_basis, "estimated_basis")
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError("basis matrices must have the same ambient dimension")
    if min(reference.shape[1], estimate.shape[1]) == 0:
        result = np.empty(0, dtype=float)
    else:
        singular_values = np.linalg.svd(reference.T @ estimate, compute_uv=False)
        result = np.arccos(np.clip(singular_values, 0.0, 1.0))
    return np.degrees(result) if degrees else result


def subspace_error(reference_basis: ArrayLike, estimated_basis: ArrayLike) -> float:
    """Root-mean-square sine of the principal angles, in ``[0, 1]``.

    Rank mismatch is penalized by appending a right angle for every unmatched
    dimension.  Consequently, a rank-deficient zero estimate is not scored as a
    perfect subspace.
    """

    reference = _orthonormal_basis(reference_basis, "reference_basis")
    estimate = _orthonormal_basis(estimated_basis, "estimated_basis")
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError("basis matrices must have the same ambient dimension")
    maximum_rank = max(reference.shape[1], estimate.shape[1])
    if maximum_rank == 0:
        return 0.0
    angles = principal_angles(reference, estimate)
    missing = maximum_rank - angles.size
    sine_squared = np.sin(angles) ** 2
    if missing:
        sine_squared = np.concatenate([sine_squared, np.ones(missing)])
    return float(np.sqrt(np.mean(sine_squared)))


def component_correlation(
    reference_components: ArrayLike,
    estimated_components: ArrayLike,
) -> dict[str, Any]:
    """Match columns with the Hungarian algorithm and report cosine correlation.

    Correlations are sign-aligned after matching.  The assignment is one-to-one;
    if the component counts differ, unmatched columns are reported separately.
    """

    reference = _matrix(reference_components, "reference_components")
    estimate = _matrix(estimated_components, "estimated_components")
    if reference.shape[0] != estimate.shape[0]:
        raise ValueError("component matrices must have the same number of rows")
    if reference.shape[1] == 0 or estimate.shape[1] == 0:
        return {
            "mean_abs_correlation": 0.0,
            "min_abs_correlation": 0.0,
            "matched_correlations": [],
            "reference_indices": [],
            "estimated_indices": [],
            "signs": [],
            "unmatched_reference": list(range(reference.shape[1])),
            "unmatched_estimated": list(range(estimate.shape[1])),
        }

    reference_norms = np.linalg.norm(reference, axis=0)
    estimate_norms = np.linalg.norm(estimate, axis=0)
    denominator = np.outer(reference_norms, estimate_norms)
    similarities = np.divide(
        reference.T @ estimate,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator > _EPS,
    )
    reference_indices, estimated_indices = linear_sum_assignment(-np.abs(similarities))
    correlations = similarities[reference_indices, estimated_indices]
    signs = np.where(correlations < 0.0, -1, 1)
    absolute = np.abs(correlations)
    matched_reference = set(reference_indices.tolist())
    matched_estimate = set(estimated_indices.tolist())
    return {
        "mean_abs_correlation": float(np.mean(absolute)),
        "min_abs_correlation": float(np.min(absolute)),
        "matched_correlations": correlations.astype(float).tolist(),
        "reference_indices": reference_indices.astype(int).tolist(),
        "estimated_indices": estimated_indices.astype(int).tolist(),
        "signs": signs.astype(int).tolist(),
        "unmatched_reference": [
            index for index in range(reference.shape[1]) if index not in matched_reference
        ],
        "unmatched_estimated": [
            index for index in range(estimate.shape[1]) if index not in matched_estimate
        ],
    }


def _rank_auc(truth: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    positive_count = int(np.count_nonzero(truth))
    negative_count = truth.size - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[truth]))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _average_precision(truth: NDArray[np.bool_], scores: NDArray[np.float64]) -> float:
    positive_count = int(np.count_nonzero(truth))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    # AP is a threshold metric: observations with identical scores enter the
    # predicted set together.  Processing tied observations one by one would
    # make the result depend on their unrelated input order.
    cumulative_positives = 0
    average_precision = 0.0
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        positives_at_threshold = int(np.count_nonzero(sorted_truth[start:stop]))
        cumulative_positives += positives_at_threshold
        precision_at_threshold = cumulative_positives / stop
        recall_increment = positives_at_threshold / positive_count
        average_precision += recall_increment * precision_at_threshold
        start = stop
    return float(average_precision)


def outlier_detection_metrics(
    true_support: ArrayLike,
    estimated_support: ArrayLike | None = None,
    *,
    scores: ArrayLike | None = None,
    threshold: float = 1e-10,
    top_k: int | None = None,
) -> dict[str, float | int]:
    """Evaluate a cellwise corruption mask and optional continuous scores.

    ``top_k`` is useful when a corruption budget is chosen without consulting
    labels.  Scores are ranked by decreasing absolute magnitude; ties that
    cross the selection boundary are broken by increasing C-order flattened
    index.  This makes the selected mask independent of NumPy's partition
    implementation.  Passing both an explicit estimated mask and ``top_k`` is
    rejected.  AUROC and average precision are added when scores are supplied.
    """

    truth = _as_support(true_support, 0.0).reshape(-1)
    score_array: NDArray[np.float64] | None = None
    if scores is not None:
        score_array = np.abs(np.asarray(scores, dtype=float)).reshape(-1)
        if score_array.shape != truth.shape:
            raise ValueError("scores and true_support must have the same number of entries")
        if not np.all(np.isfinite(score_array)):
            raise ValueError("scores contains NaN or infinite values")

    if estimated_support is not None and top_k is not None:
        raise ValueError("pass either estimated_support or top_k, not both")
    if estimated_support is None:
        if score_array is None:
            raise ValueError("scores are required when estimated_support is omitted")
        if top_k is None:
            estimate = score_array > threshold
        else:
            if top_k < 0 or top_k > truth.size:
                raise ValueError("top_k must lie between zero and the number of cells")
            estimate = np.zeros(truth.size, dtype=bool)
            if top_k:
                # Locate the kth score in linear expected time, then make only
                # the boundary choice explicit. ``flatnonzero`` is already in
                # increasing C-order, so exact ties consume the lowest indices
                # without sorting every cell in a potentially large matrix.
                cutoff_position = truth.size - top_k
                cutoff = np.partition(score_array, cutoff_position)[
                    cutoff_position
                ]
                above_cutoff = np.flatnonzero(score_array > cutoff)
                remaining = top_k - above_cutoff.size
                boundary = np.flatnonzero(score_array == cutoff)[:remaining]
                selected = np.concatenate((above_cutoff, boundary))
                estimate[selected] = True
    else:
        estimate = _as_support(estimated_support, threshold).reshape(-1)
        if estimate.shape != truth.shape:
            raise ValueError(
                "estimated_support and true_support must have the same number of entries"
            )

    result = support_precision_recall_f1(truth, estimate, threshold=0.0)
    if score_array is not None:
        result["roc_auc"] = float(_rank_auc(truth, score_array))
        result["average_precision"] = float(_average_precision(truth, score_array))
    return result


__all__ = [
    "component_correlation",
    "explained_variance_ratio",
    "outlier_detection_metrics",
    "principal_angles",
    "relative_reconstruction_error",
    "sparsity_ratio",
    "subspace_error",
    "support_precision_recall_f1",
]

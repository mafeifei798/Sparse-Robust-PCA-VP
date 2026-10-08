from __future__ import annotations

import numpy as np
import pytest

from rspca.metrics import (
    component_correlation,
    explained_variance_ratio,
    outlier_detection_metrics,
    principal_angles,
    relative_reconstruction_error,
    sparsity_ratio,
    subspace_error,
    support_precision_recall_f1,
)


def test_reconstruction_metrics_have_expected_endpoints() -> None:
    matrix = np.array([[-1.0, 2.0], [1.0, -2.0]])

    assert relative_reconstruction_error(matrix, matrix) == pytest.approx(0.0)
    assert relative_reconstruction_error(matrix, np.zeros_like(matrix)) == pytest.approx(1.0)
    assert explained_variance_ratio(matrix, matrix) == pytest.approx(1.0)
    assert explained_variance_ratio(matrix, np.zeros_like(matrix)) == pytest.approx(0.0)


def test_sparsity_and_support_metrics() -> None:
    truth = np.array([[1, 0], [0, 1]], dtype=bool)
    estimate = np.array([[2.0, 0.0], [3.0, 0.0]])
    metrics = support_precision_recall_f1(truth, estimate)

    assert sparsity_ratio(estimate) == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["true_positives"] == 1


def test_empty_support_is_well_defined() -> None:
    empty = np.zeros((2, 3), dtype=bool)
    metrics = support_precision_recall_f1(empty, empty)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_principal_angles_and_subspace_error_are_rotation_invariant() -> None:
    reference = np.eye(5)[:, :2]
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    estimate = reference @ rotation

    assert principal_angles(reference, estimate) == pytest.approx(np.zeros(2), abs=1e-7)
    assert subspace_error(reference, estimate) == pytest.approx(0.0, abs=1e-7)
    assert subspace_error(reference, np.zeros_like(reference)) == pytest.approx(1.0)


def test_orthogonal_subspaces_have_unit_error() -> None:
    reference = np.eye(4)[:, :2]
    estimate = np.eye(4)[:, 2:]

    assert principal_angles(reference, estimate, degrees=True) == pytest.approx([90, 90])
    assert subspace_error(reference, estimate) == pytest.approx(1.0)


def test_component_correlation_recovers_permutation_and_sign() -> None:
    reference = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )
    estimate = reference[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    metrics = component_correlation(reference, estimate)

    assert metrics["mean_abs_correlation"] == pytest.approx(1.0)
    assert metrics["min_abs_correlation"] == pytest.approx(1.0)
    assert metrics["estimated_indices"] == [1, 2, 0]
    assert metrics["signs"] == [1, -1, -1]


def test_outlier_metrics_support_scores_and_fixed_budget() -> None:
    truth = np.array([[False, True], [False, True]])
    scores = np.array([[0.1, 0.9], [0.2, 0.8]])
    metrics = outlier_detection_metrics(truth, scores=scores, top_k=2)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)


def test_outlier_top_k_ties_use_lowest_c_order_flattened_indices() -> None:
    truth = np.array([[True, True, False], [False, False, False]])
    scores = np.ones_like(truth, dtype=float)

    metrics = outlier_detection_metrics(truth, scores=scores, top_k=2)

    # Every score is tied, so the documented tie-break must select flattened
    # indices 0 and 1, which are exactly the two positive cells here.
    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["true_negatives"] == 4
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_average_precision_is_invariant_to_order_within_score_ties() -> None:
    truth = np.array([True, False, False, True])
    scores = np.array([1.0, 1.0, 0.0, 0.0])
    permutation = np.array([1, 0, 3, 2])

    original = outlier_detection_metrics(truth, scores=scores)
    reordered = outlier_detection_metrics(
        truth[permutation], scores=scores[permutation]
    )

    assert original["average_precision"] == pytest.approx(0.5)
    assert reordered["average_precision"] == pytest.approx(0.5)


def test_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        relative_reconstruction_error(np.zeros((2, 2)), np.zeros((3, 2)))

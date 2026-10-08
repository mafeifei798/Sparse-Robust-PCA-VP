from __future__ import annotations

import inspect

import numpy as np
import pytest

import rspca.accelerated as accelerated_module
from rspca.accelerated import (
    fit_accelerated_vp_spca,
    spca_algorithm_4_1_fista_bt,
)
from rspca.vp import proximal_gradient_mapping


def _data(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(48, 15))
    X[:, :5] += rng.normal(size=(48, 2)) @ rng.normal(size=(2, 5))
    return X - X.mean(axis=0, keepdims=True)


def test_true_value_fista_is_monotone_and_reports_accepted_step() -> None:
    X = _data()
    result = fit_accelerated_vp_spca(
        X,
        2,
        alpha=0.15,
        max_iter=100,
        tol=1e-5,
        backtracking_expand=1.3,
        monotone=True,
        adaptive_restart=True,
    )

    assert np.max(np.diff(result.objectives), initial=-np.inf) <= 1e-8
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(2), atol=1e-11)
    assert result.restart_count == result.monotone_restarts + result.adaptive_restarts
    mapping = proximal_gradient_mapping(
        X,
        result.A,
        result.B,
        alpha=result.alpha,
        step_size=result.step_sizes[-1],
    )
    assert result.stationarity[-1] == pytest.approx(np.linalg.norm(mapping), rel=1e-12)


def test_accelerated_legacy_wrapper_and_parameter_validation() -> None:
    X = _data(4)
    A, B, objectives, criteria = spca_algorithm_4_1_fista_bt(
        X,
        2,
        alpha=0.1,
        max_iter=4,
        eps=1e-12,
        fista=True,
        backtracking=True,
    )
    assert A.shape == B.shape == (15, 2)
    assert len(objectives) == len(criteria) == 4

    with pytest.raises(ValueError, match="backtracking_shrink"):
        fit_accelerated_vp_spca(X, 2, backtracking_shrink=1.0)


def test_accelerated_source_is_matrix_free() -> None:
    source = inspect.getsource(accelerated_module)
    assert "X.T @ X" not in source
    assert "X_array.T @ X_array" not in source


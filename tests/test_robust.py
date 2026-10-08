from __future__ import annotations

import inspect

import numpy as np
import pytest

import rspca.robust as robust_module
from rspca.robust import (
    _procrustes_optimality_residual,
    _robust_block_stationarity,
    fit_robust_vp_spca,
    huber_loss,
    robust_objective,
    robust_spca_vp,
)


def _corrupted_data(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    low_rank = rng.normal(size=(45, 2)) @ rng.normal(size=(2, 14))
    X = low_rank + 0.04 * rng.normal(size=low_rank.shape)
    mask = np.zeros_like(X, dtype=bool)
    mask.flat[rng.choice(X.size, size=24, replace=False)] = True
    X[mask] += rng.choice((-8.0, 8.0), size=int(mask.sum()))
    return X, mask


def test_robust_full_objective_is_monotone_and_equals_huber_form() -> None:
    X, mask = _corrupted_data()
    result = fit_robust_vp_spca(
        X,
        2,
        alpha=0.05,
        kappa=0.5,
        max_iter=120,
        tol=1e-5,
    )

    assert np.max(np.diff(result.objectives), initial=-np.inf) <= 1e-8
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(2), atol=1e-11)
    direct = robust_objective(
        X,
        result.A,
        result.B,
        result.S,
        alpha=result.alpha,
        kappa=result.kappa,
    )
    assert result.objectives[-1] == pytest.approx(direct, rel=1e-11)
    raw_residual = X - X @ result.B @ result.A.T
    eliminated = huber_loss(raw_residual, result.kappa)
    eliminated += result.alpha * float(np.sum(np.abs(result.B)))
    assert direct == pytest.approx(eliminated, rel=1e-11)
    assert np.median(np.abs(result.S[mask])) > np.median(np.abs(result.S[~mask]))


def test_mad_threshold_is_fixed_and_legacy_order_is_preserved() -> None:
    X, _ = _corrupted_data(9)
    result = fit_robust_vp_spca(
        X, 2, alpha=0.04, kappa=None, huber_delta=1.25, max_iter=5, tol=1e-12
    )
    assert result.kappa == pytest.approx(1.25 * result.residual_scale)
    assert np.isfinite(result.objectives).all()

    B, A, S = robust_spca_vp(
        X,
        2,
        alpha_r=0.04,
        kappa=0.5,
        max_iter_r=3,
        tol_r=1e-12,
    )
    assert B.shape == A.shape == (X.shape[1], 2)
    assert S.shape == X.shape


def test_rank_deficient_procrustes_stationarity_uses_the_optimum_set() -> None:
    X = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    # The second component lies in the null space.  Both [e1,e2] and [e1,e3]
    # are exact Procrustes optima, even though a particular SVD returns only one.
    A = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    B = A.copy()
    S = np.zeros_like(X)

    assert _procrustes_optimality_residual(X, A, B, S) == 0.0
    assert _robust_block_stationarity(
        X,
        A,
        B,
        S,
        alpha=0.0,
        kappa=1.0,
        step_size=1.0 / 14.0,
    ) == pytest.approx(0.0, abs=1e-14)

    nonoptimal_A = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert _procrustes_optimality_residual(X, nonoptimal_A, B, S) > 1.0


def test_robust_validation_and_matrix_free_source() -> None:
    X, _ = _corrupted_data()
    with pytest.raises(ValueError, match="kappa"):
        fit_robust_vp_spca(X, 2, kappa=0.0)
    with pytest.raises(ValueError, match="initial_B"):
        fit_robust_vp_spca(X, 2, initial_B=np.ones((3, 2)))

    source = inspect.getsource(robust_module)
    assert "X.T @ X" not in source
    assert "X_array.T @ X_array" not in source

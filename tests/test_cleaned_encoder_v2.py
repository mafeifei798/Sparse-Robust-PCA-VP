from __future__ import annotations

import inspect

import numpy as np
import pytest

import rspca.cleaned_encoder_v2 as cleaned_module
from rspca.cleaned_encoder_v2 import (
    _B_gradient,
    _S_gradient,
    cleaned_encoder_block_stationarity,
    cleaned_encoder_objective,
    cleaned_encoder_reconstruction,
    fit_cleaned_encoder_robust_vp_spca_v2,
    update_A_cleaned_encoder_procrustes,
)


def _corrupted_data(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low_rank = rng.normal(size=(36, 2)) @ rng.normal(size=(2, 12))
    values = low_rank + 0.03 * rng.normal(size=low_rank.shape)
    indices = rng.choice(values.size, size=20, replace=False)
    values.flat[indices] += rng.choice((-6.0, 6.0), size=indices.size)
    return values


def test_cleaned_encoder_objective_matches_direct_residual() -> None:
    X = _corrupted_data()
    rng = np.random.default_rng(9)
    B = rng.normal(size=(X.shape[1], 2))
    S = 0.1 * rng.normal(size=X.shape)
    A = update_A_cleaned_encoder_procrustes(X, B, S)

    reconstruction = cleaned_encoder_reconstruction(X, A, B, S)
    direct = 0.5 * float(np.sum(((X - S) - reconstruction) ** 2))
    direct += 0.07 * float(np.sum(np.abs(B)))
    direct += 0.3 * float(np.sum(np.abs(S)))
    assert cleaned_encoder_objective(
        X,
        A,
        B,
        S,
        alpha=0.07,
        kappa=0.3,
    ) == pytest.approx(direct, rel=1e-12, abs=1e-12)


def test_cleaned_encoder_B_and_S_gradients_match_finite_differences() -> None:
    X = _corrupted_data(2)[:8, :6]
    rng = np.random.default_rng(3)
    raw_A = rng.normal(size=(X.shape[1], 2))
    A, _ = np.linalg.qr(raw_A, mode="reduced")
    B = rng.normal(size=A.shape)
    S = 0.05 * rng.normal(size=X.shape)
    cleaned = X - S
    direction_B = rng.normal(size=B.shape)
    direction_S = rng.normal(size=S.shape)

    def smooth(test_B: np.ndarray, test_S: np.ndarray) -> float:
        cleaned_test = X - test_S
        residual = cleaned_test - cleaned_test @ test_B @ A.T
        return 0.5 * float(np.sum(residual * residual))

    epsilon = 1e-6
    numerical_B = (
        smooth(B + epsilon * direction_B, S) - smooth(B - epsilon * direction_B, S)
    ) / (2.0 * epsilon)
    numerical_S = (
        smooth(B, S + epsilon * direction_S) - smooth(B, S - epsilon * direction_S)
    ) / (2.0 * epsilon)
    analytic_B = float(np.sum(_B_gradient(cleaned, A, B) * direction_B))
    analytic_S = float(np.sum(_S_gradient(cleaned, A, B) * direction_S))
    assert analytic_B == pytest.approx(numerical_B, rel=2e-7, abs=2e-7)
    assert analytic_S == pytest.approx(numerical_S, rel=2e-7, abs=2e-7)


def test_cleaned_encoder_palm_is_monotone_and_reports_every_block() -> None:
    X = _corrupted_data(4)
    result = fit_cleaned_encoder_robust_vp_spca_v2(
        X,
        2,
        alpha=0.05,
        kappa=0.4,
        max_iter=80,
        tol=1e-8,
    )

    assert np.diff(result.objectives).max(initial=-np.inf) <= 1e-9
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(2), atol=1e-11)
    assert result.n_iter == len(result.objectives)
    assert result.n_iter == len(result.block_stationarity)
    assert result.n_iter == len(result.B_step_sizes)
    assert result.n_iter == len(result.S_step_sizes)
    final = result.block_stationarity[-1]
    assert final.total == max(
        final.B_prox_mapping,
        final.A_riemannian_kkt,
        final.A_global_optimality_gap,
        final.A_orthogonality,
        final.S_prox_mapping,
    )
    recomputed = cleaned_encoder_block_stationarity(
        X,
        result.A,
        result.B,
        result.S,
        alpha=result.alpha,
        kappa=result.kappa,
        B_step_size=result.B_step_sizes[-1],
        S_step_size=result.S_step_sizes[-1],
    )
    assert recomputed == final
    assert result.objectives[-1] == pytest.approx(
        cleaned_encoder_objective(
            X,
            result.A,
            result.B,
            result.S,
            alpha=result.alpha,
            kappa=result.kappa,
        ),
        rel=1e-11,
    )
    np.testing.assert_array_equal(
        result.reconstruction,
        cleaned_encoder_reconstruction(X, result.A, result.B, result.S),
    )


def test_cleaned_encoder_validation_and_matrix_free_source() -> None:
    X = _corrupted_data()
    with pytest.raises(ValueError, match="kappa"):
        fit_cleaned_encoder_robust_vp_spca_v2(X, 2, kappa=0.0)
    with pytest.raises(ValueError, match="initial_S"):
        fit_cleaned_encoder_robust_vp_spca_v2(
            X,
            2,
            initial_S=np.zeros((2, 2)),
        )
    with pytest.raises(ValueError, match="backtracking_expand"):
        fit_cleaned_encoder_robust_vp_spca_v2(
            X,
            2,
            backtracking_expand=0.9,
        )

    source = inspect.getsource(cleaned_module)
    assert "X.T @ X" not in source
    assert "X_array.T @ X_array" not in source
    assert "cleaned.T @ cleaned" not in source

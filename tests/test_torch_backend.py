from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

import rspca.torch_backend as torch_backend  # noqa: E402
from rspca.torch_backend import (  # noqa: E402
    fit_torch_accelerated_vp_spca,
    spca_vp_fista_bt_torch,
)


def test_torch_solver_is_matrix_free_monotone_and_orthogonal() -> None:
    rng = np.random.default_rng(7)
    X = rng.normal(size=(36, 14))
    X -= X.mean(axis=0, keepdims=True)

    result = fit_torch_accelerated_vp_spca(
        X,
        3,
        alpha=0.15,
        max_iter=40,
        tol=1e-7,
        device="cpu",
    )

    assert result.A.shape == (14, 3)
    assert result.B.shape == (14, 3)
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(3), atol=1e-10)
    assert np.all(np.diff(result.objectives) <= 1e-8)
    assert np.all(np.asarray(result.step_sizes) > 0)
    assert np.isfinite(result.stationarity).all()


def test_torch_legacy_wrapper_and_invalid_adaptive_alpha() -> None:
    X = np.arange(80, dtype=float).reshape(10, 8)
    X -= X.mean(axis=0, keepdims=True)
    A, B, objectives, stationarity = spca_vp_fista_bt_torch(
        X,
        2,
        max_iter=3,
    )
    assert A.shape == B.shape == (8, 2)
    assert len(objectives) == len(stationarity)
    with pytest.raises(ValueError, match="adaptive_alpha"):
        spca_vp_fista_bt_torch(X, 2, adaptive_alpha=True)


def test_torch_monotone_restart_forces_line_search_when_backtracking_is_disabled() -> None:
    rng = np.random.default_rng(19)
    X = rng.normal(size=(28, 10))
    result = fit_torch_accelerated_vp_spca(
        X,
        2,
        alpha=0.1,
        max_iter=12,
        step_size=100.0,
        backtracking=False,
        monotone_restart=True,
    )
    assert np.all(np.diff(result.objectives) <= 1e-8)


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("alpha", np.nan),
        ("alpha", np.inf),
        ("tol", np.nan),
        ("tol", np.inf),
        ("step_size", np.nan),
        ("step_size", np.inf),
        ("backtracking_shrink", np.nan),
        ("backtracking_shrink", np.inf),
        ("step_growth", np.nan),
        ("step_growth", np.inf),
    ],
)
def test_torch_solver_rejects_non_finite_scalar_parameters(
    parameter: str,
    value: float,
) -> None:
    X = np.arange(24, dtype=float).reshape(6, 4)
    with pytest.raises(ValueError, match=parameter):
        fit_torch_accelerated_vp_spca(X, 2, **{parameter: value})


@pytest.mark.parametrize("max_iter", [np.nan, np.inf, 1.5, True])
def test_torch_solver_rejects_invalid_iteration_counts(max_iter: object) -> None:
    X = np.arange(24, dtype=float).reshape(6, 4)
    with pytest.raises(TypeError, match="max_iter"):
        fit_torch_accelerated_vp_spca(X, 2, max_iter=max_iter)  # type: ignore[arg-type]


def test_torch_spectral_start_cannot_miss_a_nonzero_null_sum_matrix() -> None:
    scale = 1_000.0
    X = torch.tensor([[scale, -scale]], dtype=torch.float64)
    ones = torch.ones((2, 1), dtype=X.dtype)
    assert torch.count_nonzero(X @ ones).item() == 0

    lipschitz = torch_backend._lipschitz_estimate(X)
    assert lipschitz == pytest.approx(2.0 * scale**2)

    result = fit_torch_accelerated_vp_spca(
        X,
        1,
        alpha=0.0,
        max_iter=1,
        initial_B=torch.tensor([[1.0], [0.0]], dtype=X.dtype),
        fista=False,
        backtracking=False,
        monotone_restart=False,
    )
    assert result.step_sizes == pytest.approx([1.0 / (2.0 * scale**2)])
    assert np.isfinite(result.objectives).all()


def test_unbacktracked_automatic_step_uses_safe_frobenius_upper_bound() -> None:
    X = np.asarray(
        [
            [3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result = fit_torch_accelerated_vp_spca(
        X,
        1,
        alpha=0.0,
        max_iter=1,
        fista=False,
        backtracking=False,
        monotone_restart=False,
    )

    assert result.step_sizes == pytest.approx([1.0 / np.sum(X * X)])
    assert np.isfinite(result.objectives).all()

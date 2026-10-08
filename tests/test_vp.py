from __future__ import annotations

import inspect

import numpy as np
import pytest
from scipy import linalg

import rspca.vp as vp_module
from rspca.vp import (
    fit_vp_spca,
    proximal_gradient_mapping,
    spca_vp,
    vp_objective,
)


def _data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(55, 3))
    loadings = rng.normal(size=(18, 3))
    X = scores @ loadings.T + 0.08 * rng.normal(size=(55, 18))
    return X - X.mean(axis=0, keepdims=True)


def _legacy_reference_fit(
    X: np.ndarray,
    n_components: int,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    step_size: float,
    initial_B: np.ndarray,
    backtracking: bool = True,
    verbose: bool = False,
) -> vp_module.VPSPCAResult:
    """Independent reconstruction of the pre-cache VP iteration loop."""

    del n_components, verbose
    B = initial_B.copy()
    gamma = step_size
    A = vp_module.procrustes_factor(X, B)
    smooth = vp_module.smooth_value(X, A, B)
    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    converged = False

    for _ in range(max_iter):
        gradient = vp_module.vp_gradient(X, A, B)
        gamma_trial = gamma
        for _ in range(61):
            B_trial = vp_module.soft_threshold(
                B - gamma_trial * gradient,
                gamma_trial * alpha,
            )
            A_trial = vp_module.procrustes_factor(X, B_trial, reference=A)
            smooth_trial = vp_module.smooth_value(X, A_trial, B_trial)
            difference = B_trial - B
            majorizer = smooth + float(np.sum(gradient * difference))
            majorizer += 0.5 * float(np.sum(difference * difference)) / gamma_trial
            if smooth_trial <= majorizer + 1e-12 * max(1.0, abs(smooth)):
                break
            if not backtracking:
                break
            gamma_trial *= 0.5
        else:  # pragma: no cover - same finite-loop guard as production
            raise RuntimeError("VP-SPCA backtracking failed to terminate")

        objective = smooth_trial + alpha * float(np.sum(np.abs(B_trial)))
        mapping = vp_module.proximal_gradient_mapping(
            X,
            A_trial,
            B_trial,
            alpha=alpha,
            step_size=gamma_trial,
        )
        criterion = float(linalg.norm(mapping, ord="fro", check_finite=False))
        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(gamma_trial)
        B, A, smooth = B_trial, A_trial, smooth_trial
        gamma = gamma_trial
        if criterion <= tol:
            converged = True
            break

    return vp_module.VPSPCAResult(
        A=A,
        B=B,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        status="converged" if converged else "maximum iterations reached",
    )


def _assert_results_array_equal(
    actual: vp_module.VPSPCAResult,
    expected: vp_module.VPSPCAResult,
) -> None:
    for field in ("A", "B", "objectives", "stationarity", "step_sizes"):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(expected, field)),
        )
    assert actual.converged is expected.converged
    assert actual.n_iter == expected.n_iter
    assert actual.alpha == expected.alpha
    assert actual.status == expected.status


def test_vp_has_fixed_monotone_objective_and_correct_stationarity() -> None:
    X = _data()
    result = fit_vp_spca(X, 3, alpha=0.12, max_iter=120, tol=1e-7)

    assert result.A.shape == result.B.shape == (X.shape[1], 3)
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(3), atol=1e-11)
    assert np.max(np.diff(result.objectives), initial=-np.inf) <= 1e-8
    assert result.n_iter == len(result.objectives) == len(result.stationarity)
    assert result.alpha == 0.12

    residual = X - X @ result.B @ result.A.T
    direct = 0.5 * np.sum(residual**2) + result.alpha * np.sum(np.abs(result.B))
    assert vp_objective(X, result.A, result.B, result.alpha) == pytest.approx(
        direct, rel=1e-11, abs=1e-11
    )
    mapping = proximal_gradient_mapping(
        X,
        result.A,
        result.B,
        alpha=result.alpha,
        step_size=result.step_sizes[-1],
    )
    assert result.stationarity[-1] == pytest.approx(np.linalg.norm(mapping), rel=1e-12)


def test_vp_zero_matrix_and_legacy_return_order() -> None:
    X = np.zeros((10, 7))
    result = fit_vp_spca(X, 2, alpha=0.5, max_iter=10, tol=1e-10)
    np.testing.assert_allclose(result.B, 0.0)
    np.testing.assert_allclose(result.A.T @ result.A, np.eye(2), atol=1e-12)
    assert result.objectives[-1] == 0.0

    A, B, objectives, criteria = spca_vp(
        _data(2), 2, alpha=0.1, max_iter=3, eps=1e-12
    )
    assert A.shape == B.shape == (18, 2)
    assert len(objectives) == len(criteria) == 3


def test_default_initialization_reuses_one_svd_for_step_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_second_svd(*args: object, **kwargs: object) -> np.ndarray:
        raise AssertionError("default initialization requested a second exact SVD")

    monkeypatch.setattr(vp_module.linalg, "svdvals", unexpected_second_svd)
    result = fit_vp_spca(_data(8), 2, alpha=0.1, max_iter=2)

    assert result.n_iter == 2
    assert result.step_sizes[0] > 0.0


@pytest.mark.parametrize(
    "case",
    [
        "forced_backtracking",
        "no_backtracking",
        "early_convergence",
        "maximum_iterations",
        "zero_matrix",
        "custom_initial_B",
    ],
)
def test_shared_product_cached_kernel_is_bitwise_legacy_equivalent(case: str) -> None:
    if case == "zero_matrix":
        X = np.zeros((12, 8), dtype=np.float64)
        initial_B, exact_step = vp_module._pca_initial_loadings_and_step_size(X, 2)
        common = dict(
            alpha=0.5,
            max_iter=10,
            tol=1e-12,
            step_size=exact_step,
            initial_B=initial_B,
            backtracking=True,
        )
    else:
        X = _data(101)
        initial_B, exact_step = vp_module._pca_initial_loadings_and_step_size(X, 3)
        common = dict(
            alpha=0.17,
            max_iter=6,
            tol=1e-14,
            step_size=exact_step,
            initial_B=initial_B,
            backtracking=True,
        )
        if case == "forced_backtracking":
            common["step_size"] = 32.0 * exact_step
        elif case == "no_backtracking":
            common["step_size"] = 32.0 * exact_step
            common["backtracking"] = False
        elif case == "early_convergence":
            common["tol"] = 1e12
            common["max_iter"] = 10
        elif case == "maximum_iterations":
            common["max_iter"] = 3
            common["tol"] = 1e-16
        elif case == "custom_initial_B":
            common["initial_B"] = np.random.default_rng(202).normal(
                size=initial_B.shape
            )

    actual = fit_vp_spca(X, 2 if case == "zero_matrix" else 3, **common)
    expected = _legacy_reference_fit(
        X,
        2 if case == "zero_matrix" else 3,
        **common,
    )
    _assert_results_array_equal(actual, expected)

    if case == "forced_backtracking":
        assert actual.step_sizes[0] < common["step_size"]
    elif case == "no_backtracking":
        assert actual.step_sizes[0] == common["step_size"]
    elif case == "early_convergence":
        assert actual.converged and actual.n_iter == 1
    elif case == "maximum_iterations":
        assert not actual.converged and actual.n_iter == common["max_iter"]
    elif case == "zero_matrix":
        assert actual.converged


def test_shared_kernel_reuses_accepted_mapping_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = vp_module._vp_gradient_prevalidated
    calls = 0

    def counted_gradient(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(vp_module, "_vp_gradient_prevalidated", counted_gradient)
    result = fit_vp_spca(_data(303), 3, alpha=0.11, max_iter=7, tol=1e-16)

    # One gradient initializes the loop; each accepted point supplies both its
    # stationarity mapping and the next iteration's gradient.
    assert result.n_iter == 7
    assert calls == result.n_iter + 1


@pytest.mark.parametrize(
    ("X", "components", "message"),
    [
        (np.ones(5), 1, "two-dimensional"),
        (np.ones((3, 4)), 4, "n_components"),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), 1, "finite"),
    ],
)
def test_vp_rejects_invalid_problems(
    X: np.ndarray, components: int, message: str
) -> None:
    with pytest.raises((ValueError, TypeError), match=message):
        fit_vp_spca(X, components)


def test_vp_source_does_not_materialize_feature_gram() -> None:
    source = inspect.getsource(vp_module)
    assert "X.T @ X" not in source
    assert "X_array.T @ X_array" not in source

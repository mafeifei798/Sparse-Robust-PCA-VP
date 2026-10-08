"""Robust VP-SPCA with a fixed MAD-calibrated Huber objective.

The implementation uses the equivalent outlier-splitting formulation

``0.5 ||X - X B A.T - S||_F^2 + alpha ||B||_1 + kappa ||S||_1``.

Eliminating ``S`` gives an elementwise Huber loss with threshold ``kappa``.
When ``kappa`` is not supplied, it is estimated once from the initial PCA
residual using the Gaussian-consistent MAD and then held fixed.  Consequently
the complete objective trace describes one optimization problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .vp import (
    FloatArray,
    _as_float_matrix,
    _initial_loadings,
    _pca_initial_loadings_and_step_size,
    _orthonormal_reference,
    _spectral_step_size,
    _thin_svd,
    _validate_n_components,
    _validate_nonnegative_scalar,
    _validate_positive_integer,
    _validate_positive_scalar,
    procrustes_factor,
    soft_threshold,
)


@dataclass(frozen=True)
class RobustVPSPCAResult:
    """Result returned by :func:`fit_robust_vp_spca`."""

    A: FloatArray
    B: FloatArray
    S: FloatArray
    objectives: list[float]
    stationarity: list[float]
    step_sizes: list[float]
    converged: bool
    n_iter: int
    alpha: float
    kappa: float
    residual_scale: float
    outlier_fraction: float
    status: str

    @property
    def criteria(self) -> list[float]:
        return self.stationarity

    def __iter__(self) -> Iterator[FloatArray]:
        # Preserve the historical robust routine's B, A, S unpacking order.
        yield self.B
        yield self.A
        yield self.S


def mad_scale(values: ArrayLike) -> float:
    """Gaussian-consistent median absolute deviation with safe fallbacks."""

    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a nonempty finite array")
    median = float(np.median(array))
    scale = 1.482602218505602 * float(np.median(np.abs(array - median)))
    if scale <= np.finfo(np.float64).eps:
        scale = 1.482602218505602 * float(np.median(np.abs(array)))
    if scale <= np.finfo(np.float64).eps:
        scale = float(np.sqrt(np.mean(array * array)))
    return max(scale, np.finfo(np.float64).eps)


def huber_loss(residual: ArrayLike, kappa: float) -> float:
    """Return the sum of scalar Huber losses with threshold ``kappa``."""

    kappa = _validate_positive_scalar(kappa, name="kappa")
    residual_array = np.asarray(residual, dtype=np.float64)
    if not np.isfinite(residual_array).all():
        raise ValueError("residual must contain only finite values")
    magnitude = np.abs(residual_array)
    quadratic = np.minimum(magnitude, kappa)
    linear = magnitude - quadratic
    return 0.5 * float(np.sum(quadratic * quadratic)) + kappa * float(
        np.sum(linear)
    )


def update_A_procrustes_with_S(
    X: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
    *,
    reference: ArrayLike | None = None,
) -> FloatArray:
    """Solve the robust Procrustes block without forming a feature Gram matrix."""

    X_array = _as_float_matrix(X)
    B_array = _as_float_matrix(B, name="B")
    S_array = _as_float_matrix(S, name="S")
    if S_array.shape != X_array.shape:
        raise ValueError("S must have the same shape as X")
    if B_array.shape[0] != X_array.shape[1]:
        raise ValueError("B must have shape (X.shape[1], n_components)")
    n_components = B_array.shape[1]
    if not 1 <= n_components <= min(X_array.shape):
        raise ValueError("B has an invalid component dimension")

    cross = (X_array - S_array).T @ (X_array @ B_array)
    cross_norm = float(linalg.norm(cross, ord="fro", check_finite=False))
    scale = max(1.0, float(linalg.norm(X_array, ord="fro", check_finite=False)) ** 2)
    if cross_norm <= np.finfo(np.float64).eps * scale:
        if reference is None:
            base = np.eye(X_array.shape[1], n_components, dtype=np.float64)
        else:
            base = _as_float_matrix(reference, name="reference")
            if base.shape != B_array.shape:
                raise ValueError("reference must have the same shape as B")
        return _orthonormal_reference(base, n_components)
    left, _, right_t = _thin_svd(cross)
    factor = left @ right_t
    left2, _, right_t2 = _thin_svd(factor)
    return np.asarray(left2 @ right_t2, dtype=np.float64)


def _smooth_fixed_blocks(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    S: FloatArray,
) -> float:
    """Squared loss for fixed ``A,S`` using only thin intermediate products."""

    target = X - S
    xb = X @ B
    target_a = target @ A
    value = 0.5 * float(np.sum(target * target))
    value += 0.5 * float(np.sum(xb * xb))
    value -= float(np.sum(target_a * xb))
    scale = max(1.0, float(np.sum(target * target)))
    if value < 0.0 and value >= -100.0 * np.finfo(float).eps * scale:
        value = 0.0
    return value


def robust_objective(
    X: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
    *,
    alpha: float,
    kappa: float,
) -> float:
    """Evaluate the complete fixed split-Huber objective."""

    X_array = _as_float_matrix(X)
    A_array = _as_float_matrix(A, name="A")
    B_array = _as_float_matrix(B, name="B")
    S_array = _as_float_matrix(S, name="S")
    if A_array.shape != B_array.shape or A_array.shape[0] != X_array.shape[1]:
        raise ValueError("A and B must have shape (X.shape[1], n_components)")
    if S_array.shape != X_array.shape:
        raise ValueError("S must have the same shape as X")
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    kappa = _validate_positive_scalar(kappa, name="kappa")
    clean_residual = X_array - X_array @ B_array @ A_array.T - S_array
    return (
        0.5 * float(np.sum(clean_residual * clean_residual))
        + alpha * float(np.sum(np.abs(B_array)))
        + kappa * float(np.sum(np.abs(S_array)))
    )


def _robust_gradient(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    S: FloatArray,
) -> FloatArray:
    # (XBA.T - (X-S))A = XB - (X-S)A because A.T A = I.
    return X.T @ ((X @ B) - (X - S) @ A)


def _procrustes_optimality_residual(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    S: FloatArray,
) -> float:
    """Return a normalized gap to the robust Procrustes optimum set.

    For ``C = (X-S).T @ (X@B)``, the best Stiefel-block value is
    ``||C||_*`` and the value attained by ``A`` is ``trace(A.T @ C)``.
    Their gap is zero for *every* optimal polar factor, including arbitrary
    null-space completions when ``C`` is rank deficient.  In contrast, the
    distance to one SVD-selected factor can be positive at an exact optimum.
    """

    cross = (X - S).T @ (X @ B)
    optimum = float(np.sum(linalg.svdvals(cross, check_finite=False)))
    attained = float(np.sum(A * cross))
    scale = max(1.0, optimum)
    # Subtract a conservative floating-point allowance before taking the square
    # root; a computed exact polar factor should not inherit a sqrt(eps) floor.
    gap = max(0.0, optimum - attained - 64.0 * np.finfo(float).eps * scale)
    return float(np.sqrt(2.0 * gap / scale))


def _robust_block_stationarity(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    S: FloatArray,
    *,
    alpha: float,
    kappa: float,
    step_size: float,
) -> float:
    gradient = _robust_gradient(X, A, B, S)
    prox_B = soft_threshold(B - step_size * gradient, step_size * alpha)
    B_mapping_norm = float(
        linalg.norm((B - prox_B) / step_size, ord="fro", check_finite=False)
    )
    A_gap = _procrustes_optimality_residual(X, A, B, S)
    raw_residual = X - X @ B @ A.T
    S_optimal = soft_threshold(raw_residual, kappa)
    S_gap = float(linalg.norm(S - S_optimal, ord="fro", check_finite=False))
    return max(B_mapping_norm, A_gap, S_gap)


def fit_robust_vp_spca(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.1,
    kappa: float | None = None,
    huber_delta: float = 1.345,
    max_iter: int = 500,
    tol: float = 1e-6,
    step_size: float | None = None,
    initial_B: ArrayLike | None = None,
    kappa_min: float | None = None,
    kappa_max: float | None = None,
    backtracking: bool = True,
    backtracking_expand: float = 1.2,
    verbose: bool = False,
) -> RobustVPSPCAResult:
    """Fit robust sparse PCA by monotone three-block minimization.

    ``B`` receives a backtracked proximal-gradient step, ``A`` an exact
    Procrustes update, and ``S`` its exact soft-threshold update.  The recorded
    objective includes all three terms.  A MAD threshold is calibrated once,
    avoiding the changing-objective issue in the historical implementation.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    huber_delta = _validate_positive_scalar(huber_delta, name="huber_delta")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_positive_scalar(tol, name="tol")
    expand = _validate_positive_scalar(backtracking_expand, name="backtracking_expand")
    if expand < 1.0:
        raise ValueError("backtracking_expand must be at least 1")
    if not isinstance(backtracking, (bool, np.bool_)):
        raise TypeError("backtracking must be boolean")

    if step_size is None and initial_B is None:
        B, gamma = _pca_initial_loadings_and_step_size(X_array, n_components)
    else:
        B = _initial_loadings(X_array, n_components, initial_B)
        gamma = (
            _spectral_step_size(X_array)
            if step_size is None
            else _validate_positive_scalar(step_size, name="step_size")
        )
    A = procrustes_factor(X_array, B)
    initial_residual = X_array - X_array @ B @ A.T
    residual_scale = mad_scale(initial_residual)

    if kappa is None:
        kappa_fixed = huber_delta * residual_scale
        if kappa_min is not None:
            kappa_fixed = max(
                kappa_fixed,
                _validate_positive_scalar(kappa_min, name="kappa_min"),
            )
        if kappa_max is not None:
            maximum = _validate_positive_scalar(kappa_max, name="kappa_max")
            if kappa_min is not None and maximum < float(kappa_min):
                raise ValueError("kappa_max must be at least kappa_min")
            kappa_fixed = min(kappa_fixed, maximum)
    else:
        kappa_fixed = _validate_positive_scalar(kappa, name="kappa")

    S = soft_threshold(initial_residual, kappa_fixed)

    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    converged = False

    for iteration in range(max_iter):
        gradient = _robust_gradient(X_array, A, B, S)
        smooth_current = _smooth_fixed_blocks(X_array, A, B, S)
        gamma_trial = gamma

        for _ in range(61):
            B_trial = soft_threshold(
                B - gamma_trial * gradient,
                gamma_trial * alpha,
            )
            smooth_trial_fixed = _smooth_fixed_blocks(X_array, A, B_trial, S)
            difference = B_trial - B
            upper = smooth_current + float(np.sum(gradient * difference))
            upper += 0.5 * float(np.sum(difference * difference)) / gamma_trial
            if smooth_trial_fixed <= upper + 1e-12 * max(1.0, abs(smooth_current)):
                break
            if not backtracking:
                break
            gamma_trial *= 0.5
        else:  # pragma: no cover
            raise RuntimeError("robust VP-SPCA backtracking failed to terminate")

        A_trial = update_A_procrustes_with_S(
            X_array,
            B_trial,
            S,
            reference=A,
        )
        raw_residual = X_array - X_array @ B_trial @ A_trial.T
        S_trial = soft_threshold(raw_residual, kappa_fixed)
        objective = robust_objective(
            X_array,
            A_trial,
            B_trial,
            S_trial,
            alpha=alpha,
            kappa=kappa_fixed,
        )
        criterion = _robust_block_stationarity(
            X_array,
            A_trial,
            B_trial,
            S_trial,
            alpha=alpha,
            kappa=kappa_fixed,
            step_size=gamma_trial,
        )

        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(gamma_trial)
        A, B, S = A_trial, B_trial, S_trial
        # Once outliers move into S, the local curvature can be far smaller
        # than the conservative global spectral bound.  Probe a larger step on
        # the next iteration and let the full-objective line search certify it.
        gamma = gamma_trial * expand if backtracking else gamma_trial

        if verbose and (iteration < 10 or (iteration + 1) % 50 == 0):
            print(
                f"Robust VP-SPCA iter={iteration + 1:4d} "
                f"objective={objective:.8e} stationarity={criterion:.3e} "
                f"kappa={kappa_fixed:.3e}"
            )

        if criterion <= tol:
            converged = True
            break

    outlier_fraction = float(np.mean(np.abs(S) > 10.0 * np.finfo(float).eps))
    status = "converged" if converged else "maximum iterations reached"
    return RobustVPSPCAResult(
        A=A,
        B=B,
        S=S,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        kappa=kappa_fixed,
        residual_scale=residual_scale,
        outlier_fraction=outlier_fraction,
        status=status,
    )


def robust_spca_vp(
    X_in: ArrayLike,
    r: int,
    alpha_r: float = 0.012,
    kappa: float | None = None,
    max_iter_r: int = 500,
    tol_r: float = 1e-6,
    huber_tau: float = 1.0,
    kappa_min: float = 0.005,
    kappa_max: float = 0.04,
    verbose: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Backward-compatible wrapper returning ``(B, A, S)``."""

    result = fit_robust_vp_spca(
        X_in,
        r,
        alpha=alpha_r,
        kappa=kappa,
        huber_delta=huber_tau,
        max_iter=max_iter_r,
        tol=tol_r,
        kappa_min=kappa_min if kappa is None else None,
        kappa_max=kappa_max if kappa is None else None,
        verbose=verbose,
    )
    return result.B, result.A, result.S


__all__ = [
    "RobustVPSPCAResult",
    "fit_robust_vp_spca",
    "huber_loss",
    "mad_scale",
    "robust_objective",
    "robust_spca_vp",
    "soft_threshold",
    "update_A_procrustes_with_S",
]

"""Accelerated, matrix-free optimization of the VP-SPCA value function."""

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
    _spectral_step_size,
    _validate_n_components,
    _validate_nonnegative_scalar,
    _validate_positive_integer,
    _validate_positive_scalar,
    procrustes_factor,
    proximal_gradient_mapping,
    smooth_value,
    soft_threshold,
    vp_gradient,
)


@dataclass(frozen=True)
class AcceleratedVPSPCAResult:
    """Result of true value-function FISTA with adaptive restart."""

    A: FloatArray
    B: FloatArray
    objectives: list[float]
    stationarity: list[float]
    step_sizes: list[float]
    converged: bool
    n_iter: int
    alpha: float
    status: str
    restart_count: int
    monotone_restarts: int
    adaptive_restarts: int

    @property
    def criteria(self) -> list[float]:
        return self.stationarity

    def __iter__(self) -> Iterator[object]:
        yield self.A
        yield self.B
        yield self.objectives
        yield self.stationarity


def fit_accelerated_vp_spca(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    step_size: float | None = None,
    initial_B: ArrayLike | None = None,
    acceleration: bool = True,
    backtracking: bool = True,
    backtracking_shrink: float = 0.5,
    backtracking_expand: float = 1.1,
    monotone: bool = True,
    adaptive_restart: bool = True,
    verbose: bool = False,
) -> AcceleratedVPSPCAResult:
    """Fit VP-SPCA with FISTA evaluated on the actual value function.

    The Procrustes factor ``A(Y)`` is recomputed at every extrapolated point
    before evaluating the gradient.  If a monotonicity check fails, the
    rejected accelerated proposal is discarded and a complete proximal step
    is recomputed from the current ``B``.  The stationarity mapping uses the
    accepted (pre-expansion) step size.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_positive_scalar(tol, name="tol")
    shrink = _validate_positive_scalar(backtracking_shrink, name="backtracking_shrink")
    expand = _validate_positive_scalar(backtracking_expand, name="backtracking_expand")
    if not 0.0 < shrink < 1.0:
        raise ValueError("backtracking_shrink must lie strictly between 0 and 1")
    if expand < 1.0:
        raise ValueError("backtracking_expand must be at least 1")
    for flag, name in (
        (acceleration, "acceleration"),
        (backtracking, "backtracking"),
        (monotone, "monotone"),
        (adaptive_restart, "adaptive_restart"),
    ):
        if not isinstance(flag, (bool, np.bool_)):
            raise TypeError(f"{name} must be boolean")

    if step_size is None and initial_B is None:
        B, gamma = _pca_initial_loadings_and_step_size(X_array, n_components)
    else:
        gamma = (
            _spectral_step_size(X_array)
            if step_size is None
            else _validate_positive_scalar(step_size, name="step_size")
        )
        B = _initial_loadings(X_array, n_components, initial_B)
    A = procrustes_factor(X_array, B)
    smooth = smooth_value(X_array, A, B)
    objective_previous = smooth + alpha * float(np.sum(np.abs(B)))

    Y = B.copy()
    momentum = 1.0
    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    monotone_restarts = 0
    adaptive_restarts = 0
    converged = False

    def take_value_function_step(
        point: FloatArray,
        trial_step: float,
        reference_A: FloatArray,
        *,
        force_backtracking: bool = False,
    ) -> tuple[FloatArray, FloatArray, float, float]:
        """Backtracked prox step, evaluating A(point) and A(candidate)."""

        A_point = procrustes_factor(X_array, point, reference=reference_A)
        smooth_point = smooth_value(X_array, A_point, point)
        gradient = vp_gradient(X_array, A_point, point)

        for _ in range(61):
            candidate = soft_threshold(
                point - trial_step * gradient,
                trial_step * alpha,
            )
            A_candidate = procrustes_factor(X_array, candidate, reference=A_point)
            smooth_candidate = smooth_value(X_array, A_candidate, candidate)
            difference = candidate - point
            upper = smooth_point + float(np.sum(gradient * difference))
            upper += 0.5 * float(np.sum(difference * difference)) / trial_step
            if smooth_candidate <= upper + 1e-12 * max(1.0, abs(smooth_point)):
                return candidate, A_candidate, smooth_candidate, trial_step
            if not (backtracking or force_backtracking):
                return candidate, A_candidate, smooth_candidate, trial_step
            trial_step *= shrink
        raise RuntimeError("accelerated VP-SPCA backtracking failed to terminate")

    for iteration in range(max_iter):
        B_trial, A_trial, smooth_trial, accepted_step = take_value_function_step(
            Y,
            gamma,
            A,
        )
        objective = smooth_trial + alpha * float(np.sum(np.abs(B_trial)))
        forced_restart = False

        if monotone and objective > objective_previous + 1e-12 * max(
            1.0, abs(objective_previous)
        ):
            # Reject the extrapolated proposal and recompute A(B), gradient,
            # line search, and prox from the last accepted loading.
            monotone_restarts += 1
            forced_restart = True
            momentum = 1.0
            Y = B.copy()
            B_trial, A_trial, smooth_trial, accepted_step = take_value_function_step(
                Y,
                accepted_step,
                A,
                force_backtracking=True,
            )
            objective = smooth_trial + alpha * float(np.sum(np.abs(B_trial)))

        mapping = proximal_gradient_mapping(
            X_array,
            A_trial,
            B_trial,
            alpha=alpha,
            step_size=accepted_step,
        )
        criterion = float(linalg.norm(mapping, ord="fro", check_finite=False))

        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(accepted_step)

        if verbose and (iteration < 10 or (iteration + 1) % 50 == 0):
            print(
                f"Accelerated VP-SPCA iter={iteration + 1:4d} "
                f"objective={objective:.8e} stationarity={criterion:.3e} "
                f"step={accepted_step:.3e}"
            )

        if criterion <= tol:
            B, A, smooth = B_trial, A_trial, smooth_trial
            converged = True
            break

        next_momentum = (
            0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            if acceleration
            else 1.0
        )

        gradient_restart = False
        if acceleration and adaptive_restart and not forced_restart:
            # O'Donoghue-Candes gradient restart.  This is evaluated using only
            # n_features-by-n_components arrays.
            alignment = float(np.sum((Y - B_trial) * (B_trial - B)))
            gradient_restart = alignment > 0.0
            if gradient_restart:
                adaptive_restarts += 1

        if forced_restart or gradient_restart or not acceleration:
            next_momentum = 1.0
            Y_next = B_trial.copy()
        else:
            coefficient = (momentum - 1.0) / next_momentum
            Y_next = B_trial + coefficient * (B_trial - B)

        B, A, smooth = B_trial, A_trial, smooth_trial
        Y = Y_next
        momentum = next_momentum
        objective_previous = objective
        gamma = accepted_step * expand if backtracking else accepted_step

    status = "converged" if converged else "maximum iterations reached"
    return AcceleratedVPSPCAResult(
        A=A,
        B=B,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        status=status,
        restart_count=monotone_restarts + adaptive_restarts,
        monotone_restarts=monotone_restarts,
        adaptive_restarts=adaptive_restarts,
    )


def spca_algorithm_4_1_fista_bt(
    X: ArrayLike,
    r: int,
    alpha: float = 0.1,
    max_iter: int = 500,
    eps: float = 1e-6,
    gamma: float | None = None,
    adaptive_alpha: bool = False,
    fista: bool = True,
    backtracking: bool = True,
    bt_shrink: float = 0.5,
    bt_expand: float = 1.2,
    restart_monotone: bool = True,
    verbose: bool = False,
) -> tuple[FloatArray, FloatArray, list[float], list[float]]:
    """Legacy wrapper preserving the historical four-array return order."""

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, r)
    alpha_fixed = _validate_nonnegative_scalar(alpha, name="alpha")
    if adaptive_alpha:
        B0 = _initial_loadings(X_array, n_components, None)
        A0 = procrustes_factor(X_array, B0)
        norm_x = float(linalg.norm(X_array, ord="fro", check_finite=False))
        residual = np.sqrt(max(0.0, 2.0 * smooth_value(X_array, A0, B0)))
        alpha_fixed *= float(
            np.clip(residual / max(norm_x, np.finfo(float).eps), 0.5, 1.0)
        )

    result = fit_accelerated_vp_spca(
        X_array,
        n_components,
        alpha=alpha_fixed,
        max_iter=max_iter,
        tol=eps,
        step_size=gamma,
        acceleration=fista,
        backtracking=backtracking,
        backtracking_shrink=bt_shrink,
        backtracking_expand=bt_expand,
        monotone=restart_monotone,
        adaptive_restart=True,
        verbose=verbose,
    )
    return result.A, result.B, result.objectives, result.stationarity


__all__ = [
    "AcceleratedVPSPCAResult",
    "fit_accelerated_vp_spca",
    "spca_algorithm_4_1_fista_bt",
]

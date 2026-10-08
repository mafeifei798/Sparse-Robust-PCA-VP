r"""Experimental Cleaned-Encoder Robust VP-SPCA v2.

This module is intentionally separate from :mod:`rspca.robust`.  It solves

.. math::

   \min_{A,B,S}\; \tfrac12\|(X-S)-(X-S)BA^T\|_F^2
       + \alpha\|B\|_1 + \kappa\|S\|_1,
   \qquad A^T A=I.

The implementation is an auditable PALM/block-coordinate prototype.  It is
not presented as a validated new algorithm.  Both nonsmooth blocks use
backtracked proximal-gradient steps, the Stiefel block is solved exactly by a
polar factor, and every stopping check recomputes all three block residuals at
the state produced by the complete cycle.
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
    _orthonormal_reference,
    _pca_initial_loadings_and_step_size,
    _thin_svd,
    _validate_n_components,
    _validate_nonnegative_scalar,
    _validate_positive_integer,
    _validate_positive_scalar,
    soft_threshold,
)


@dataclass(frozen=True)
class CleanedEncoderBlockStationarity:
    """Complete relative block-stationarity diagnostics for one iterate."""

    B_prox_mapping: float
    B_prox_mapping_raw: float
    A_riemannian_kkt: float
    A_global_optimality_gap: float
    A_orthogonality: float
    S_prox_mapping: float
    S_prox_mapping_raw: float
    total: float


@dataclass(frozen=True)
class CleanedEncoderV2Result:
    """Result of :func:`fit_cleaned_encoder_robust_vp_spca_v2`."""

    A: FloatArray
    B: FloatArray
    S: FloatArray
    reconstruction: FloatArray
    objectives: list[float]
    stationarity: list[float]
    block_stationarity: list[CleanedEncoderBlockStationarity]
    B_step_sizes: list[float]
    S_step_sizes: list[float]
    converged: bool
    n_iter: int
    alpha: float
    kappa: float
    outlier_fraction: float
    status: str

    @property
    def criteria(self) -> list[float]:
        return self.stationarity

    def __iter__(self) -> Iterator[FloatArray]:
        yield self.B
        yield self.A
        yield self.S


def _validate_factors(
    X: FloatArray,
    A: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    A_array = _as_float_matrix(A, name="A")
    B_array = _as_float_matrix(B, name="B")
    S_array = _as_float_matrix(S, name="S")
    if A_array.shape != B_array.shape or A_array.shape[0] != X.shape[1]:
        raise ValueError("A and B must have shape (X.shape[1], n_components)")
    if S_array.shape != X.shape:
        raise ValueError("S must have the same shape as X")
    gram = A_array.T @ A_array
    if not np.allclose(gram, np.eye(A_array.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("A must have orthonormal columns")
    return A_array, B_array, S_array


def _smooth_from_cleaned(
    cleaned: FloatArray,
    A: FloatArray,
    B: FloatArray,
) -> float:
    """Evaluate the smooth term using only thin products."""

    cleaned_B = cleaned @ B
    cleaned_A = cleaned @ A
    value = 0.5 * float(np.sum(cleaned * cleaned))
    value += 0.5 * float(np.sum(cleaned_B * cleaned_B))
    value -= float(np.sum(cleaned_A * cleaned_B))
    scale = max(1.0, float(np.sum(cleaned * cleaned)))
    if value < 0.0 and value >= -100.0 * np.finfo(float).eps * scale:
        value = 0.0
    return value


def cleaned_encoder_reconstruction(
    X: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
) -> FloatArray:
    """Return the native v2 low-rank layer ``(X-S) @ B @ A.T``."""

    X_array = _as_float_matrix(X)
    A_array, B_array, S_array = _validate_factors(X_array, A, B, S)
    return np.asarray(((X_array - S_array) @ B_array) @ A_array.T, dtype=np.float64)


def cleaned_encoder_objective(
    X: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
    *,
    alpha: float,
    kappa: float,
) -> float:
    """Evaluate the complete fixed v2 objective."""

    X_array = _as_float_matrix(X)
    A_array, B_array, S_array = _validate_factors(X_array, A, B, S)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    kappa = _validate_positive_scalar(kappa, name="kappa")
    cleaned = X_array - S_array
    return (
        _smooth_from_cleaned(cleaned, A_array, B_array)
        + alpha * float(np.sum(np.abs(B_array)))
        + kappa * float(np.sum(np.abs(S_array)))
    )


def _B_gradient(
    cleaned: FloatArray,
    A: FloatArray,
    B: FloatArray,
) -> FloatArray:
    return cleaned.T @ ((cleaned @ B) - (cleaned @ A))


def _S_gradient(
    cleaned: FloatArray,
    A: FloatArray,
    B: FloatArray,
) -> FloatArray:
    """Return ``-(Y M) M.T`` for ``Y=X-S`` and ``M=I-B A.T``.

    The low-rank identity avoids materializing either ``M`` or a feature Gram
    matrix.  Since ``A.T @ A = I``, ``(Y-YBA.T)A = YA-YB``.
    """

    cleaned_B = cleaned @ B
    cleaned_A = cleaned @ A
    residual = cleaned - cleaned_B @ A.T
    return -residual + (cleaned_A - cleaned_B) @ B.T


def update_A_cleaned_encoder_procrustes(
    X: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
    *,
    reference: ArrayLike | None = None,
) -> FloatArray:
    """Exactly minimize the v2 Stiefel block for fixed ``B,S``."""

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

    cleaned = X_array - S_array
    cross = cleaned.T @ (cleaned @ B_array)
    cross_norm = float(linalg.norm(cross, ord="fro", check_finite=False))
    scale = max(
        1.0,
        float(linalg.norm(cleaned, ord="fro", check_finite=False)) ** 2,
    )
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


def cleaned_encoder_block_stationarity(
    X: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    S: ArrayLike,
    *,
    alpha: float,
    kappa: float,
    B_step_size: float,
    S_step_size: float,
) -> CleanedEncoderBlockStationarity:
    """Recompute every block condition at one common complete-cycle state."""

    X_array = _as_float_matrix(X)
    A_array, B_array, S_array = _validate_factors(X_array, A, B, S)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    kappa = _validate_positive_scalar(kappa, name="kappa")
    B_step_size = _validate_positive_scalar(B_step_size, name="B_step_size")
    S_step_size = _validate_positive_scalar(S_step_size, name="S_step_size")
    cleaned = X_array - S_array

    gradient_B = _B_gradient(cleaned, A_array, B_array)
    prox_B = soft_threshold(
        B_array - B_step_size * gradient_B,
        B_step_size * alpha,
    )
    raw_B = float(
        linalg.norm(
            (B_array - prox_B) / B_step_size,
            ord="fro",
            check_finite=False,
        )
    )
    relative_B = raw_B / max(
        1.0,
        float(linalg.norm(B_array, ord="fro", check_finite=False)),
    )

    cross = cleaned.T @ (cleaned @ B_array)
    sym_multiplier = 0.5 * (A_array.T @ cross + cross.T @ A_array)
    tangent_residual = -cross + A_array @ sym_multiplier
    cross_scale = max(
        1.0,
        float(linalg.norm(cross, ord="fro", check_finite=False)),
    )
    A_kkt = float(
        linalg.norm(tangent_residual, ord="fro", check_finite=False) / cross_scale
    )
    optimum = float(np.sum(linalg.svdvals(cross, check_finite=False)))
    attained = float(np.sum(A_array * cross))
    gap_scale = max(1.0, optimum)
    gap = max(
        0.0,
        optimum - attained - 64.0 * np.finfo(float).eps * gap_scale,
    )
    A_gap = float(np.sqrt(2.0 * gap / gap_scale))
    A_orthogonality = float(
        linalg.norm(
            A_array.T @ A_array - np.eye(A_array.shape[1]),
            ord="fro",
            check_finite=False,
        )
    )

    gradient_S = _S_gradient(cleaned, A_array, B_array)
    prox_S = soft_threshold(
        S_array - S_step_size * gradient_S,
        S_step_size * kappa,
    )
    raw_S = float(
        linalg.norm(
            (S_array - prox_S) / S_step_size,
            ord="fro",
            check_finite=False,
        )
    )
    relative_S = raw_S / max(
        1.0,
        float(linalg.norm(X_array, ord="fro", check_finite=False)),
        float(linalg.norm(S_array, ord="fro", check_finite=False)),
    )
    total = max(relative_B, A_kkt, A_gap, A_orthogonality, relative_S)
    return CleanedEncoderBlockStationarity(
        B_prox_mapping=relative_B,
        B_prox_mapping_raw=raw_B,
        A_riemannian_kkt=A_kkt,
        A_global_optimality_gap=A_gap,
        A_orthogonality=A_orthogonality,
        S_prox_mapping=relative_S,
        S_prox_mapping_raw=raw_S,
        total=float(total),
    )


def _initial_S_step_size(B: FloatArray) -> float:
    # ||I-B A.T||_2 <= 1 + ||B||_2 because ||A||_2=1.
    bound = 1.0 + float(linalg.norm(B, ord=2, check_finite=False))
    return 1.0 / (bound * bound)


def fit_cleaned_encoder_robust_vp_spca_v2(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.0015,
    kappa: float = 0.001,
    max_iter: int = 200,
    tol: float = 1e-6,
    B_step_size: float | None = None,
    S_step_size: float | None = None,
    initial_B: ArrayLike | None = None,
    initial_S: ArrayLike | None = None,
    backtracking_expand: float = 1.2,
    verbose: bool = False,
) -> CleanedEncoderV2Result:
    """Fit the experimental cleaned-encoder objective by monotone PALM.

    ``alpha``, ``kappa`` and the objective remain fixed for the full run.  The
    method never selects an iterate or hyperparameter using evaluation truth.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    kappa = _validate_positive_scalar(kappa, name="kappa")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_positive_scalar(tol, name="tol")
    expand = _validate_positive_scalar(backtracking_expand, name="backtracking_expand")
    if expand < 1.0:
        raise ValueError("backtracking_expand must be at least 1")

    if B_step_size is None and initial_B is None:
        B, gamma_B = _pca_initial_loadings_and_step_size(X_array, n_components)
    else:
        B = _initial_loadings(X_array, n_components, initial_B)
        gamma_B = (
            1.0
            / max(
                float(linalg.norm(X_array, ord=2, check_finite=False)) ** 2,
                np.finfo(np.float64).eps,
            )
            if B_step_size is None
            else _validate_positive_scalar(B_step_size, name="B_step_size")
        )
    if initial_S is None:
        S = np.zeros_like(X_array)
    else:
        S = _as_float_matrix(initial_S, name="initial_S").copy()
        if S.shape != X_array.shape:
            raise ValueError("initial_S must have the same shape as X")
    A = update_A_cleaned_encoder_procrustes(X_array, B, S)
    gamma_S = (
        _initial_S_step_size(B)
        if S_step_size is None
        else _validate_positive_scalar(S_step_size, name="S_step_size")
    )

    objectives: list[float] = []
    stationarity: list[float] = []
    block_stationarity: list[CleanedEncoderBlockStationarity] = []
    B_step_sizes: list[float] = []
    S_step_sizes: list[float] = []
    converged = False
    previous_objective = cleaned_encoder_objective(
        X_array,
        A,
        B,
        S,
        alpha=alpha,
        kappa=kappa,
    )

    for iteration in range(max_iter):
        cleaned = X_array - S
        gradient_B = _B_gradient(cleaned, A, B)
        smooth_B = _smooth_from_cleaned(cleaned, A, B)
        gamma_B_trial = gamma_B
        objective_before_B = previous_objective
        for _ in range(61):
            B_trial = soft_threshold(
                B - gamma_B_trial * gradient_B,
                gamma_B_trial * alpha,
            )
            difference = B_trial - B
            smooth_trial = _smooth_from_cleaned(cleaned, A, B_trial)
            upper = smooth_B + float(np.sum(gradient_B * difference))
            upper += 0.5 * float(np.sum(difference * difference)) / gamma_B_trial
            objective_trial = (
                smooth_trial
                + alpha * float(np.sum(np.abs(B_trial)))
                + kappa * float(np.sum(np.abs(S)))
            )
            tolerance = 1e-12 * max(1.0, abs(objective_before_B))
            if smooth_trial <= upper + tolerance and objective_trial <= (
                objective_before_B + tolerance
            ):
                break
            gamma_B_trial *= 0.5
        else:  # pragma: no cover
            raise RuntimeError("cleaned-encoder B backtracking failed")

        objective_before_A = objective_trial
        A_trial = update_A_cleaned_encoder_procrustes(
            X_array,
            B_trial,
            S,
            reference=A,
        )
        objective_after_A = (
            _smooth_from_cleaned(cleaned, A_trial, B_trial)
            + alpha * float(np.sum(np.abs(B_trial)))
            + kappa * float(np.sum(np.abs(S)))
        )
        tolerance_A = 1e-11 * max(1.0, abs(objective_before_A))
        if objective_after_A > objective_before_A + tolerance_A:
            raise RuntimeError("cleaned-encoder A block increased the objective")

        gradient_S = _S_gradient(cleaned, A_trial, B_trial)
        smooth_S = _smooth_from_cleaned(cleaned, A_trial, B_trial)
        gamma_S_trial = gamma_S
        for _ in range(61):
            S_trial = soft_threshold(
                S - gamma_S_trial * gradient_S,
                gamma_S_trial * kappa,
            )
            difference_S = S_trial - S
            cleaned_trial = X_array - S_trial
            smooth_trial_S = _smooth_from_cleaned(
                cleaned_trial,
                A_trial,
                B_trial,
            )
            upper_S = smooth_S + float(np.sum(gradient_S * difference_S))
            upper_S += 0.5 * float(np.sum(difference_S * difference_S)) / gamma_S_trial
            objective = (
                smooth_trial_S
                + alpha * float(np.sum(np.abs(B_trial)))
                + kappa * float(np.sum(np.abs(S_trial)))
            )
            tolerance_S = 1e-12 * max(1.0, abs(objective_after_A))
            if smooth_trial_S <= upper_S + tolerance_S and objective <= (
                objective_after_A + tolerance_S
            ):
                break
            gamma_S_trial *= 0.5
        else:  # pragma: no cover
            raise RuntimeError("cleaned-encoder S backtracking failed")

        monotone_tolerance = 1e-10 * max(1.0, abs(previous_objective))
        if objective > previous_objective + monotone_tolerance:
            raise RuntimeError("cleaned-encoder full cycle increased the objective")

        A, B, S = A_trial, B_trial, S_trial
        diagnostics = cleaned_encoder_block_stationarity(
            X_array,
            A,
            B,
            S,
            alpha=alpha,
            kappa=kappa,
            B_step_size=gamma_B_trial,
            S_step_size=gamma_S_trial,
        )
        objectives.append(float(objective))
        stationarity.append(diagnostics.total)
        block_stationarity.append(diagnostics)
        B_step_sizes.append(gamma_B_trial)
        S_step_sizes.append(gamma_S_trial)
        previous_objective = float(objective)
        gamma_B = gamma_B_trial * expand
        gamma_S = gamma_S_trial * expand

        if verbose and (iteration < 10 or (iteration + 1) % 50 == 0):
            print(
                f"Cleaned-Encoder v2 iter={iteration + 1:4d} "
                f"objective={objective:.8e} stationarity={diagnostics.total:.3e}"
            )
        if diagnostics.total <= tol:
            converged = True
            break

    reconstruction = np.asarray(((X_array - S) @ B) @ A.T, dtype=np.float64)
    outlier_fraction = float(np.mean(np.abs(S) > 10.0 * np.finfo(float).eps))
    return CleanedEncoderV2Result(
        A=A,
        B=B,
        S=S,
        reconstruction=reconstruction,
        objectives=objectives,
        stationarity=stationarity,
        block_stationarity=block_stationarity,
        B_step_sizes=B_step_sizes,
        S_step_sizes=S_step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        kappa=kappa,
        outlier_fraction=outlier_fraction,
        status="converged" if converged else "maximum iterations reached",
    )


__all__ = [
    "CleanedEncoderBlockStationarity",
    "CleanedEncoderV2Result",
    "cleaned_encoder_block_stationarity",
    "cleaned_encoder_objective",
    "cleaned_encoder_reconstruction",
    "fit_cleaned_encoder_robust_vp_spca_v2",
    "update_A_cleaned_encoder_procrustes",
]

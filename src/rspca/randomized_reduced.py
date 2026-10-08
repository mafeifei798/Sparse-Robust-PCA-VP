"""Development implementation of randomized VP-SPCA with a small polar solve.

This solves the original L1-penalized VP objective on Z = Q.T @ X. It does
not select SNPs first, change the objective, or force B into the sketch row
space. Only the algebra used to evaluate the value function is different:
for Z = U diag(s) V.T, polar(Z.T Z B) = V polar(diag(s**2) V.T B).
At deficient rank the small SVD chooses one valid Procrustes minimizer;
the value function need not be differentiable there.

Inputs must already be centered. Scaling the *whole* sketch objective by
its spectral norm squared improves numerical conditioning without changing
its minimizers. Convergence below is explicitly for the sketch, not X.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import linalg

from .randomized import _randomized_range_finder_prevalidated
from .vp import (
    _as_float_matrix,
    _thin_svd,
    _validate_n_components,
    _validate_positive_integer,
    _validate_positive_scalar,
)


@dataclass(frozen=True)
class ReducedVPSPCAResult:
    A: np.ndarray
    B: np.ndarray
    Q: np.ndarray
    alpha: float
    lipschitz_sketch: float
    sketch_converged: bool
    sketch_relative_stationarity: float
    sketch_objectives_scaled: list[float]
    n_iter: int
    restarts: int
    timings: dict[str, float]


def reduced_value_polar(weights, coordinates):
    """Smooth value minus its constant, and a small Procrustes factor.

    ``weights = s**2 / L`` and ``coordinates = V.T @ B``. This also returns
    a valid orthogonal factor at zero/deficient cross-covariance; no claim
    of a unique derivative at those points is made.
    """
    cross = weights[:, None] * coordinates
    left, singular, right = _thin_svd(cross)
    polar = left @ right
    value = 0.5 * np.sum(coordinates * cross) - np.sum(singular)
    return float(value), polar


def _threshold(matrix, threshold):
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)


def solve_reduced_vp(
    right_basis,
    weights,
    rank,
    rho,
    *,
    max_iter=500,
    relative_tolerance=1e-3,
    check_every=10,
    acceleration=True,
):
    """Monotone, restarted FISTA on a prevalidated sketch SVD.

    The step is 1 in spectral-normalized units. For any minimizing A(Y),
    the fixed-A quadratic is a global upper model of the VP value function,
    so this step is safe even without assuming a global Lipschitz gradient
    of the reduced nonconvex value function. No expansive line search is
    used. Rejected momentum proposals are recomputed from accepted B.

    Relative stationarity is ||B-prox(B-grad)|| / (rho*sqrt(B.size)).
    Checking it less often can only delay the stop; the final iterate is
    always checked. The objective trace omits the fixed data-norm constant.
    """
    B = right_basis[:, :rank].copy()
    coordinates = right_basis.T @ B
    smooth, polar = reduced_value_polar(weights, coordinates)
    previous = smooth + rho * float(np.abs(B).sum())
    Y, y_coordinates = B.copy(), coordinates.copy()
    momentum = 1.0
    trace = []
    restarts = 0
    eta = float("inf")
    converged = False
    for iteration in range(1, max_iter + 1):
        _, y_polar = reduced_value_polar(weights, y_coordinates)
        gradient = right_basis @ (weights[:, None] * (y_coordinates - y_polar))
        candidate = _threshold(Y - gradient, rho)
        next_coordinates = right_basis.T @ candidate
        smooth, polar = reduced_value_polar(weights, next_coordinates)
        objective = smooth + rho * float(np.abs(candidate).sum())
        forced = objective > previous + 1e-12 * max(1.0, abs(previous))
        if forced:
            restarts += 1
            momentum = 1.0
            Y, y_coordinates = B, coordinates
            _, y_polar = reduced_value_polar(weights, coordinates)
            gradient = right_basis @ (weights[:, None] * (coordinates - y_polar))
            candidate = _threshold(B - gradient, rho)
            next_coordinates = right_basis.T @ candidate
            smooth, polar = reduced_value_polar(weights, next_coordinates)
            objective = smooth + rho * float(np.abs(candidate).sum())
        if objective > previous + 1e-10 * max(1.0, abs(previous)):
            raise RuntimeError("reduced VP descent check failed")
        trace.append(float(objective))
        if iteration % check_every == 0 or iteration == max_iter:
            grad_next = right_basis @ (weights[:, None] * (next_coordinates - polar))
            mapping = candidate - _threshold(candidate - grad_next, rho)
            eta = float(linalg.norm(mapping) / (rho * np.sqrt(candidate.size)))
            converged = bool(eta <= relative_tolerance)
        if converged:
            B, coordinates = candidate, next_coordinates
            break
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        restart = acceleration and np.sum((Y - candidate) * (candidate - B)) > 0
        if restart:
            restarts += 1
        coefficient = 0.0 if forced or restart or not acceleration else (momentum - 1.0) / next_momentum
        Y = candidate + coefficient * (candidate - B)
        y_coordinates = next_coordinates + coefficient * (next_coordinates - coordinates)
        B, coordinates = candidate, next_coordinates
        momentum = 1.0 if forced or restart or not acceleration else next_momentum
        previous = objective
    _, polar = reduced_value_polar(weights, coordinates)
    return B, right_basis @ polar, trace, eta, converged, iteration, restarts


def fit_reduced_randomized_vp_spca(
    X,
    n_components,
    *,
    penalty_fraction=0.05,
    oversample=10,
    n_power_iter=2,
    max_iter=500,
    relative_tolerance=1e-3,
    check_every=10,
    random_state=2025,
):
    """Fit genuine randomized VP-SPCA, without full-data refinement.

    alpha = penalty_fraction * ||Q.T X||_2**2 / sqrt(n_features).
    This scale rule is fixed using unlabeled training data only. A positive
    penalty is mandatory; alpha=0 cannot masquerade as sparse PCA here.
    Full-data stationarity/reconstruction must be audited separately, and
    must not be inferred from ``sketch_converged``.
    """
    begin = perf_counter()
    X = _as_float_matrix(X)
    rank = _validate_n_components(X, n_components)
    penalty_fraction = _validate_positive_scalar(penalty_fraction, name="penalty_fraction")
    relative_tolerance = _validate_positive_scalar(relative_tolerance, name="relative_tolerance")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    check_every = _validate_positive_integer(check_every, name="check_every")
    for value, name in ((oversample, "oversample"), (n_power_iter, "n_power_iter")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    Q = _randomized_range_finder_prevalidated(X, rank, oversample, n_power_iter, random_state)
    sketch = Q.T @ X
    _, singular, right_t = _thin_svd(sketch)
    scale = float(singular[0] ** 2)
    if scale <= np.finfo(float).tiny or not np.isfinite(scale):
        raise ValueError("X must have a finite positive spectral scale")
    weights = singular**2 / scale
    right_basis = np.ascontiguousarray(right_t.T)
    rho = penalty_fraction / np.sqrt(X.shape[1])
    prepared = perf_counter()
    B, A, trace, eta, converged, iterations, restarts = solve_reduced_vp(
        right_basis, weights, rank, rho, max_iter=max_iter,
        relative_tolerance=relative_tolerance, check_every=check_every,
    )
    finished = perf_counter()
    return ReducedVPSPCAResult(
        A=A, B=B, Q=Q, alpha=rho * scale, lipschitz_sketch=scale,
        sketch_converged=converged, sketch_relative_stationarity=eta,
        sketch_objectives_scaled=trace, n_iter=iterations, restarts=restarts,
        timings={"preparation_seconds": prepared - begin,
                 "iteration_seconds": finished - prepared,
                 "fit_seconds": finished - begin},
    )

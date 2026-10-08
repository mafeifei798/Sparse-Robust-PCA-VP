"""Checked, allocation-aware implementation of the same randomized VP fit.

No scientific setting is selected here. Float64, the original sketch rule,
L1 objective, 50/500-step budgets and convergence definition are unchanged.
The older implementation remains available as an independent reference.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import linalg

from .randomized_reduced import ReducedVPSPCAResult, reduced_value_polar
from .vp import (_as_float_matrix, _thin_svd, _validate_n_components,
                 _validate_positive_integer, _validate_positive_scalar)


@dataclass(frozen=True)
class FastReducedVPSPCAResult(ReducedVPSPCAResult):
    preparation_diagnostics: dict


def checked_qr(matrix, diagnostics):
    """Alternating Householder QR, with a checked second-QR repair."""
    q, r = linalg.qr(matrix, mode="economic", check_finite=False)
    error = float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))
    diagonal = np.abs(np.diag(r))
    deficient = not diagonal.size or diagonal.min() <= 1e-10 * diagonal.max()
    repair = deficient or not np.isfinite(error) or error > 1e-12
    diagnostics["qr_calls"] += 1
    diagnostics["qr_max_initial_orthogonality_error"] = max(
        diagnostics["qr_max_initial_orthogonality_error"], error)
    if repair:
        q, _ = linalg.qr(q, mode="economic", check_finite=False)
        diagnostics["qr_repairs"] += 1
        error = float(np.max(np.abs(q.T @ q - np.eye(q.shape[1]))))
    final_error = error
    if not np.isfinite(final_error) or final_error > 1e-10:
        raise RuntimeError("QR orthogonality validation failed")
    return q


def checked_range_finder(X, rank, oversample, n_power_iter, random_state):
    width = min(min(X.shape), rank + oversample)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    diagnostics = {"qr_calls": 0, "qr_repairs": 0,
                   "qr_max_initial_orthogonality_error": 0.0}
    omega = rng.standard_normal((X.shape[1], width))
    q = checked_qr(X @ omega, diagnostics)
    del omega
    for _ in range(n_power_iter):
        right = checked_qr(X.T @ q, diagnostics)
        q = checked_qr(X @ right, diagnostics)
    return q, diagnostics


def checked_sketch_factorization(sketch):
    """Small row-Gram factorization, with conditioning/orthogonality gates."""
    gram = sketch @ sketch.T
    eigenvalues, left = linalg.eigh((gram + gram.T) * .5, check_finite=False)
    ratio = float(eigenvalues[0] / eigenvalues[-1]) if eigenvalues[-1] > 0 else 0.0
    reason = "ill_conditioned_sketch"
    if np.isfinite(ratio) and ratio >= 1e-6:
        singular = np.sqrt(eigenvalues[::-1])
        basis = np.ascontiguousarray(sketch.T @ (left[:, ::-1] / singular))
        error = float(np.max(np.abs(basis.T @ basis - np.eye(len(singular)))))
        if np.isfinite(error) and error <= 1e-10:
            return singular, basis, {"factorization": "checked_small_row_gram",
                                     "sketch_eigenvalue_ratio": ratio,
                                     "basis_orthogonality_error": error}
        reason = "basis_orthogonality_guard"
    _, singular, right = _thin_svd(sketch)
    basis = np.ascontiguousarray(right.T)
    return singular, basis, {"factorization": "original_svd_fallback",
                             "fallback_reason": reason, "sketch_eigenvalue_ratio": ratio,
                             "basis_orthogonality_error": float(np.max(np.abs(basis.T @ basis - np.eye(len(singular)))))}


def solve_reduced_vp_buffered(V, weights, rank, rho, *, max_iter=500,
                              relative_tolerance=1e-3, check_every=10,
                              acceleration=True):
    """Same monotone restarted VP steps, using reusable feature-space buffers."""
    B = np.array(V[:, :rank], order="C", copy=True)
    coordinates = V.T @ B
    smooth, polar = reduced_value_polar(weights, coordinates)
    previous = smooth + rho * float(np.abs(B).sum())
    Y, y_coordinates = B.copy(), coordinates.copy()
    candidate, gradient, work = [np.empty_like(B) for _ in range(3)]
    momentum, restarts, eta, converged = 1.0, 0, float("inf"), False
    trace = []

    def proposal(point, coords):
        _, local_polar = reduced_value_polar(weights, coords)
        np.matmul(V, weights[:, None] * (coords - local_polar), out=gradient)
        np.subtract(point, gradient, out=candidate)
        # z - clip(z,-rho,rho) is exactly the same L1 proximal map.
        np.clip(candidate, -rho, rho, out=gradient)
        np.subtract(candidate, gradient, out=candidate)
        next_coords = V.T @ candidate
        local_value, next_polar = reduced_value_polar(weights, next_coords)
        np.abs(candidate, out=work)
        value = local_value + rho * float(work.sum())
        return next_coords, next_polar, value

    for iteration in range(1, max_iter + 1):
        next_coordinates, polar, objective = proposal(Y, y_coordinates)
        forced = objective > previous + 1e-12 * max(1.0, abs(previous))
        if forced:
            restarts += 1
            momentum = 1.0
            np.copyto(Y, B)
            y_coordinates = coordinates
            next_coordinates, polar, objective = proposal(B, coordinates)
        if objective > previous + 1e-10 * max(1.0, abs(previous)):
            raise RuntimeError("reduced VP descent check failed")
        trace.append(float(objective))
        if iteration % check_every == 0 or iteration == max_iter:
            np.matmul(V, weights[:, None] * (next_coordinates - polar), out=gradient)
            np.subtract(candidate, gradient, out=work)
            np.clip(work, -rho, rho, out=gradient)
            np.subtract(work, gradient, out=work)
            np.subtract(candidate, work, out=gradient)
            eta = float(linalg.norm(gradient) / (rho * np.sqrt(candidate.size)))
            converged = bool(eta <= relative_tolerance)
        if converged:
            B, coordinates = candidate, next_coordinates
            break
        next_momentum = (1.0 + np.sqrt(1.0 + 4.0 * momentum**2)) / 2.0
        np.subtract(Y, candidate, out=gradient)
        np.subtract(candidate, B, out=work)
        np.multiply(gradient, work, out=gradient)
        restart = acceleration and gradient.sum() > 0
        if restart:
            restarts += 1
        coefficient = 0.0 if forced or restart or not acceleration else (momentum - 1.0) / next_momentum
        np.multiply(work, coefficient, out=Y)
        np.add(candidate, Y, out=Y)
        y_coordinates = next_coordinates + coefficient * (next_coordinates - coordinates)
        B, candidate = candidate, B
        coordinates = next_coordinates
        momentum = 1.0 if forced or restart or not acceleration else next_momentum
        previous = objective
    _, polar = reduced_value_polar(weights, coordinates)
    return B, V @ polar, trace, eta, converged, iteration, restarts


def fit_fast_randomized_vp_spca(X, n_components, *, penalty_fraction=.05,
                               oversample=10, n_power_iter=2, max_iter=500,
                               relative_tolerance=1e-3, check_every=10,
                               random_state=2025):
    """Fit the unchanged float64 objective with checked implementation savings."""
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
    validated = perf_counter()
    Q, diagnostics = checked_range_finder(X, rank, oversample, n_power_iter, random_state)
    ranged = perf_counter()
    sketch = Q.T @ X
    singular, basis, factorization = checked_sketch_factorization(sketch)
    diagnostics.update(factorization)
    del sketch
    scale = float(singular[0]**2)
    if scale <= np.finfo(float).tiny or not np.isfinite(scale):
        raise ValueError("X must have a finite positive spectral scale")
    weights, rho = singular**2 / scale, penalty_fraction / np.sqrt(X.shape[1])
    prepared = perf_counter()
    B, A, trace, eta, converged, iterations, restarts = solve_reduced_vp_buffered(
        basis, weights, rank, rho, max_iter=max_iter,
        relative_tolerance=relative_tolerance, check_every=check_every)
    finished = perf_counter()
    return FastReducedVPSPCAResult(A=A, B=B, Q=Q, alpha=rho*scale, lipschitz_sketch=scale,
        sketch_converged=converged, sketch_relative_stationarity=eta,
        sketch_objectives_scaled=trace, n_iter=iterations, restarts=restarts,
        timings={"preparation_seconds": prepared-begin, "iteration_seconds": finished-prepared,
                 "fit_seconds": finished-begin, "validation_seconds": validated-begin,
                 "range_finder_seconds": ranged-validated, "factorization_seconds": prepared-ranged},
        preparation_diagnostics=diagnostics)

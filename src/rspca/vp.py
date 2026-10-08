r"""Matrix-free variable-projection sparse PCA.

The model solved here is

.. math::

   \min_{B,\;A^T A=I}\;\tfrac12\|X-XBA^T\|_F^2+\alpha\|B\|_1.

For a fixed ``B``, the optimal ``A`` is the polar (orthogonal Procrustes)
factor of ``X.T @ (X @ B)``.  Substituting that solution gives a value
function in ``B``.  All products below are evaluated as ``X.T @ (X @ M)``;
the feature Gram matrix is never formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VPSPCAResult:
    """Result returned by :func:`fit_vp_spca`.

    ``objectives``, ``stationarity``, and ``step_sizes`` contain one entry per
    accepted proximal step.  ``alpha`` is the single, fixed regularization
    value used for every entry in the trace.
    """

    A: FloatArray
    B: FloatArray
    objectives: list[float]
    stationarity: list[float]
    step_sizes: list[float]
    converged: bool
    n_iter: int
    alpha: float
    status: str

    @property
    def criteria(self) -> list[float]:
        """Legacy name for the proximal-gradient stationarity trace."""

        return self.stationarity

    def __iter__(self) -> Iterator[object]:
        """Allow legacy four-value unpacking without hiding result metadata."""

        yield self.A
        yield self.B
        yield self.objectives
        yield self.stationarity


def _as_float_matrix(X: ArrayLike, *, name: str = "X") -> FloatArray:
    """Return a finite, real, two-dimensional floating-point array."""

    array = np.asarray(X)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_n_components(X: FloatArray, n_components: int) -> int:
    if isinstance(n_components, bool) or not isinstance(n_components, (int, np.integer)):
        raise TypeError("n_components must be an integer")
    n_components = int(n_components)
    max_rank = min(X.shape)
    if not 1 <= n_components <= max_rank:
        raise ValueError(
            f"n_components must be between 1 and min(X.shape)={max_rank}"
        )
    return n_components


def _validate_nonnegative_scalar(value: float, *, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real scalar") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_positive_scalar(value: float, *, name: str) -> float:
    value = _validate_nonnegative_scalar(value, name=name)
    if value == 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def soft_threshold(values: ArrayLike, threshold: float) -> FloatArray:
    """Elementwise proximal operator of ``threshold * ||.||_1``."""

    threshold = _validate_nonnegative_scalar(threshold, name="threshold")
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)


def _thin_svd(matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute a thin SVD, retrying with the more conservative LAPACK driver."""

    try:
        return linalg.svd(
            matrix,
            full_matrices=False,
            check_finite=False,
            lapack_driver="gesdd",
        )
    except linalg.LinAlgError:
        return linalg.svd(
            matrix,
            full_matrices=False,
            check_finite=False,
            lapack_driver="gesvd",
        )


def _orthonormal_reference(reference: FloatArray, n_components: int) -> FloatArray:
    q, _ = linalg.qr(reference, mode="economic", check_finite=False)
    if q.shape[1] < n_components:
        raise ValueError("reference does not span enough independent columns")
    return np.asarray(q[:, :n_components], dtype=np.float64)


def procrustes_factor(
    X: ArrayLike,
    B: ArrayLike,
    *,
    reference: ArrayLike | None = None,
) -> FloatArray:
    """Return the stable polar factor solving the VP Procrustes subproblem.

    Parameters are checked explicitly because a silently broadcast loading
    matrix can otherwise produce a plausible-looking but invalid solution.
    When the cross-covariance is numerically zero, ``reference`` supplies a
    deterministic orthonormal factor; otherwise the first coordinate axes are
    used.  No feature-by-feature Gram matrix is formed.
    """

    X_array = _as_float_matrix(X)
    B_array = _as_float_matrix(B, name="B")
    n_features = X_array.shape[1]
    if B_array.shape[0] != n_features:
        raise ValueError("B must have shape (X.shape[1], n_components)")
    n_components = B_array.shape[1]
    if not 1 <= n_components <= min(X_array.shape):
        raise ValueError("B has an invalid component dimension")

    cross = X_array.T @ (X_array @ B_array)
    cross_norm = float(linalg.norm(cross, ord="fro", check_finite=False))
    scale = max(1.0, float(linalg.norm(X_array, ord="fro", check_finite=False)) ** 2)
    if cross_norm <= np.finfo(np.float64).eps * scale:
        if reference is None:
            base = np.eye(n_features, n_components, dtype=np.float64)
        else:
            base = _as_float_matrix(reference, name="reference")
            if base.shape != (n_features, n_components):
                raise ValueError("reference must have the same shape as B")
        return _orthonormal_reference(base, n_components)

    left, _, right_t = _thin_svd(cross)
    factor = left @ right_t
    # A final polar projection controls round-off when singular values cluster.
    left2, _, right_t2 = _thin_svd(factor)
    return np.asarray(left2 @ right_t2, dtype=np.float64)


def vp_gradient(X: ArrayLike, A: ArrayLike, B: ArrayLike) -> FloatArray:
    """Gradient of the VP smooth value at a consistent Procrustes factor."""

    X_array = _as_float_matrix(X)
    A_array = _as_float_matrix(A, name="A")
    B_array = _as_float_matrix(B, name="B")
    expected_rows = X_array.shape[1]
    if A_array.shape != B_array.shape or A_array.shape[0] != expected_rows:
        raise ValueError("A and B must both have shape (X.shape[1], n_components)")
    return X_array.T @ (X_array @ (B_array - A_array))


def smooth_value(X: ArrayLike, A: ArrayLike, B: ArrayLike) -> float:
    """Evaluate ``0.5 * ||X - X B A.T||_F**2`` without its dense residual."""

    X_array = _as_float_matrix(X)
    A_array = _as_float_matrix(A, name="A")
    B_array = _as_float_matrix(B, name="B")
    if A_array.shape != B_array.shape or A_array.shape[0] != X_array.shape[1]:
        raise ValueError("A and B must both have shape (X.shape[1], n_components)")
    gram_a = A_array.T @ A_array
    if not np.allclose(gram_a, np.eye(A_array.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("A must have orthonormal columns")

    xb = X_array @ B_array
    xa = X_array @ A_array
    value = 0.5 * float(np.sum(X_array * X_array))
    value += 0.5 * float(np.sum(xb * xb))
    value -= float(np.sum(xa * xb))
    # The exact value is nonnegative; remove only tiny cancellation noise.
    scale = max(1.0, float(np.sum(X_array * X_array)))
    if value < 0.0 and value >= -100.0 * np.finfo(np.float64).eps * scale:
        value = 0.0
    return value


def vp_objective(X: ArrayLike, A: ArrayLike, B: ArrayLike, alpha: float) -> float:
    """Evaluate the fixed VP-SPCA composite objective."""

    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    B_array = _as_float_matrix(B, name="B")
    return smooth_value(X, A, B_array) + alpha * float(np.sum(np.abs(B_array)))


def proximal_gradient_mapping(
    X: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    *,
    alpha: float,
    step_size: float,
) -> FloatArray:
    """Return the matrix-valued proximal-gradient stationarity mapping."""

    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    step_size = _validate_positive_scalar(step_size, name="step_size")
    B_array = _as_float_matrix(B, name="B")
    gradient = vp_gradient(X, A, B_array)
    prox = soft_threshold(
        B_array - step_size * gradient,
        step_size * alpha,
    )
    return (B_array - prox) / step_size


def _initial_loadings(
    X: FloatArray,
    n_components: int,
    initial_B: ArrayLike | None,
) -> FloatArray:
    if initial_B is not None:
        B = _as_float_matrix(initial_B, name="initial_B")
        expected = (X.shape[1], n_components)
        if B.shape != expected:
            raise ValueError(f"initial_B must have shape {expected}")
        return B.copy()

    _, _, right_t = _thin_svd(X)
    return np.asarray(right_t[:n_components].T, dtype=np.float64).copy()


def _spectral_step_size(X: FloatArray) -> float:
    """Safe exact spectral step computed directly from ``X`` (not a Gram matrix)."""

    singular_values = linalg.svdvals(X, check_finite=False)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    if sigma_max == 0.0:
        return 1.0
    return 1.0 / (sigma_max * sigma_max)


def _pca_initial_loadings_and_step_size(
    X: FloatArray,
    n_components: int,
) -> tuple[FloatArray, float]:
    """Compute the default PCA initialization and exact step in one thin SVD."""

    _, singular_values, right_t = _thin_svd(X)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    step_size = 1.0 if sigma_max == 0.0 else 1.0 / (sigma_max * sigma_max)
    loadings = np.asarray(right_t[:n_components].T, dtype=np.float64).copy()
    return loadings, step_size


def _soft_threshold_prevalidated(
    values: FloatArray,
    threshold: float,
) -> FloatArray:
    """Threshold arrays whose dtype and scalar inputs were already checked."""

    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def _procrustes_factor_from_product_prevalidated(
    X: FloatArray,
    B: FloatArray,
    product: FloatArray,
    *,
    reference: FloatArray | None,
    data_frobenius_norm_squared: float,
) -> FloatArray:
    """Internal Procrustes factor reusing an existing ``X @ B`` product."""

    cross = X.T @ product
    cross_norm = float(linalg.norm(cross, ord="fro", check_finite=False))
    scale = max(1.0, data_frobenius_norm_squared)
    if cross_norm <= np.finfo(np.float64).eps * scale:
        base = (
            np.eye(X.shape[1], B.shape[1], dtype=np.float64)
            if reference is None
            else reference
        )
        return _orthonormal_reference(base, B.shape[1])

    left, _, right_t = _thin_svd(cross)
    factor = left @ right_t
    left2, _, right_t2 = _thin_svd(factor)
    return np.asarray(left2 @ right_t2, dtype=np.float64)


def _smooth_value_from_product_prevalidated(
    X: FloatArray,
    A: FloatArray,
    product: FloatArray,
    *,
    data_sum_squares: float,
) -> float:
    """Internal smooth value reusing the Procrustes point's ``X @ B``."""

    XA = X @ A
    value = 0.5 * data_sum_squares
    value += 0.5 * float(np.sum(product * product))
    value -= float(np.sum(XA * product))
    scale = max(1.0, data_sum_squares)
    if value < 0.0 and value >= -100.0 * np.finfo(np.float64).eps * scale:
        value = 0.0
    return value


def _vp_gradient_prevalidated(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
) -> FloatArray:
    """VP gradient for inputs validated once at the public boundary."""

    return X.T @ (X @ (B - A))


def _fit_vp_spca_prevalidated(
    X: FloatArray,
    n_components: int,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    step_size: float,
    initial_B: FloatArray,
    backtracking: bool = True,
    verbose: bool = False,
) -> VPSPCAResult:
    """Shared product-cached VP loop behind public and randomized fitting.

    The caller owns the one-time validation boundary.  Within an accepted
    iterate, ``X @ B`` is shared by Procrustes and the smooth value, and the
    gradient used for the stationarity mapping is retained as the next
    iteration's gradient.  Trial products are deliberately local to a single
    backtracking proposal, so no rejected point or different alpha can leak
    into a later computation.
    """

    del n_components  # Shape consistency was established by the caller.
    B = initial_B.copy()
    gamma = step_size
    data_frobenius_norm = float(linalg.norm(X, ord="fro", check_finite=False))
    data_frobenius_norm_squared = data_frobenius_norm * data_frobenius_norm
    data_sum_squares = float(np.sum(X * X))

    XB = X @ B
    A = _procrustes_factor_from_product_prevalidated(
        X,
        B,
        XB,
        reference=None,
        data_frobenius_norm_squared=data_frobenius_norm_squared,
    )
    smooth = _smooth_value_from_product_prevalidated(
        X,
        A,
        XB,
        data_sum_squares=data_sum_squares,
    )
    # This gradient is also the mapping gradient at the initial accepted point.
    # Each later mapping gradient is cached into the next outer iteration.
    gradient = _vp_gradient_prevalidated(X, A, B)

    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    converged = False
    max_backtracking = 60

    for iteration in range(max_iter):
        gamma_trial = gamma

        for _ in range(max_backtracking + 1):
            B_trial = _soft_threshold_prevalidated(
                B - gamma_trial * gradient,
                gamma_trial * alpha,
            )
            XB_trial = X @ B_trial
            A_trial = _procrustes_factor_from_product_prevalidated(
                X,
                B_trial,
                XB_trial,
                reference=A,
                data_frobenius_norm_squared=data_frobenius_norm_squared,
            )
            smooth_trial = _smooth_value_from_product_prevalidated(
                X,
                A_trial,
                XB_trial,
                data_sum_squares=data_sum_squares,
            )
            difference = B_trial - B
            majorizer = smooth + float(np.sum(gradient * difference))
            majorizer += 0.5 * float(np.sum(difference * difference)) / gamma_trial
            if smooth_trial <= majorizer + 1e-12 * max(1.0, abs(smooth)):
                break
            if not backtracking:
                break
            gamma_trial *= 0.5
        else:  # pragma: no cover - protected by the explicit finite loop
            raise RuntimeError("VP-SPCA backtracking failed to terminate")

        objective = smooth_trial + alpha * float(np.sum(np.abs(B_trial)))
        mapping_gradient = _vp_gradient_prevalidated(X, A_trial, B_trial)
        mapping_prox = _soft_threshold_prevalidated(
            B_trial - gamma_trial * mapping_gradient,
            gamma_trial * alpha,
        )
        mapping = (B_trial - mapping_prox) / gamma_trial
        criterion = float(linalg.norm(mapping, ord="fro", check_finite=False))

        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(gamma_trial)
        B, A, smooth = B_trial, A_trial, smooth_trial
        gamma = gamma_trial
        gradient = mapping_gradient

        if verbose and (iteration < 10 or (iteration + 1) % 50 == 0):
            print(
                f"VP-SPCA iter={iteration + 1:4d} objective={objective:.8e} "
                f"stationarity={criterion:.3e} step={gamma_trial:.3e}"
            )

        if criterion <= tol:
            converged = True
            break

    status = "converged" if converged else "maximum iterations reached"
    return VPSPCAResult(
        A=A,
        B=B,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        status=status,
    )


def fit_vp_spca(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    step_size: float | None = None,
    initial_B: ArrayLike | None = None,
    backtracking: bool = True,
    verbose: bool = False,
) -> VPSPCAResult:
    """Fit sparse PCA using proximal gradient on the VP value function.

    Backtracking is performed on the *projected value function*: every trial
    loading gets its own Procrustes factor before the majorization test.  This
    keeps the objective trace and stationarity test aligned to one fixed
    objective.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_positive_scalar(tol, name="tol")
    if not isinstance(backtracking, (bool, np.bool_)):
        raise TypeError("backtracking must be boolean")

    if step_size is None and initial_B is None:
        B, gamma = _pca_initial_loadings_and_step_size(X_array, n_components)
    else:
        gamma = (
            _spectral_step_size(X_array)
            if step_size is None
            else _validate_positive_scalar(step_size, name="step_size")
        )
        B = _initial_loadings(X_array, n_components, initial_B)
    return _fit_vp_spca_prevalidated(
        X_array,
        n_components,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        step_size=gamma,
        initial_B=B,
        backtracking=bool(backtracking),
        verbose=verbose,
    )


def spca_vp(
    X: ArrayLike,
    r: int,
    alpha: float = 0.1,
    max_iter: int = 100,
    eps: float = 1e-6,
    gamma: float | None = None,
    adaptive_alpha: bool = False,
    verbose: bool = False,
) -> tuple[FloatArray, FloatArray, list[float], list[float]]:
    """Backward-compatible four-value wrapper around :func:`fit_vp_spca`.

    The historical ``adaptive_alpha`` option changed the optimization problem
    on every iteration.  For reproducible stationarity, this wrapper instead
    estimates its residual scale once at initialization and then holds the
    resulting regularization parameter fixed.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, r)
    alpha_fixed = _validate_nonnegative_scalar(alpha, name="alpha")
    if adaptive_alpha:
        initial_B = _initial_loadings(X_array, n_components, None)
        initial_A = procrustes_factor(X_array, initial_B)
        data_norm = float(linalg.norm(X_array, ord="fro", check_finite=False))
        residual_norm = np.sqrt(max(0.0, 2.0 * smooth_value(X_array, initial_A, initial_B)))
        scale = float(np.clip(residual_norm / max(data_norm, np.finfo(float).eps), 0.5, 1.0))
        alpha_fixed *= scale

    result = fit_vp_spca(
        X_array,
        n_components,
        alpha=alpha_fixed,
        max_iter=max_iter,
        tol=eps,
        step_size=gamma,
        backtracking=True,
        verbose=verbose,
    )
    return result.A, result.B, result.objectives, result.stationarity


__all__ = [
    "VPSPCAResult",
    "fit_vp_spca",
    "proximal_gradient_mapping",
    "procrustes_factor",
    "smooth_value",
    "soft_threshold",
    "spca_vp",
    "vp_gradient",
    "vp_objective",
]

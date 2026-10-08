"""Randomized VP-SPCA with residual diagnostics and certified refinement."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Iterator

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .vp import (
    FloatArray,
    _as_float_matrix,
    _fit_vp_spca_prevalidated,
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
class SketchRefinementCertificate:
    """A posteriori evidence used by certified sketch-to-full refinement.

    ``loss_gap_upper_bound`` bounds, at the sketched solution, the omitted
    squared reconstruction loss:

    ``0.5 * ||(I-QQ.T) X (I-B A.T)||_F**2``.

    The computable bound uses
    ``||(I-QQ.T)X||_F * (1 + ||B||_2)`` and does not build either identity or
    feature Gram matrices.
    """

    projection_residual: float
    relative_projection_residual: float
    residual_tolerance: float
    loss_gap_upper_bound: float
    full_stationarity_before: float
    full_stationarity_after: float
    full_stationarity_tolerance: float
    refinement_triggered: bool
    trigger_reason: str
    monotone_restarts: int
    adaptive_restarts: int
    restart_count: int
    certified: bool


@dataclass(frozen=True)
class RandomizedVPSPCAResult:
    """Result returned by randomized and certified VP-SPCA routines.

    ``converged`` and ``full_converged`` always refer to the full-data
    proximal-gradient diagnostic.  ``sketch_converged`` records the compressed
    solve separately (and is ``None`` when the sketch is used only to
    initialize a full-data solve).  This distinction prevents a successful
    compressed solve from being reported as convergence of the original
    problem.
    """

    A: FloatArray
    B: FloatArray
    objectives: list[float]
    stationarity: list[float]
    step_sizes: list[float]
    converged: bool
    sketch_converged: bool | None
    full_converged: bool
    n_iter: int
    alpha: float
    status: str
    Q: FloatArray
    sketch_residual: float
    sketch_residual_relative: float
    full_objective: float
    full_stationarity: float
    full_diagnostic_step_size: float
    full_diagnostic_step_policy: str
    used_full_gradient: bool
    sketch_iterations: int
    full_refinement_iterations: int
    timings: dict[str, float]
    sketch_objectives: list[float]
    certificate: SketchRefinementCertificate | None = None

    @property
    def criteria(self) -> list[float]:
        return self.stationarity

    def __iter__(self) -> Iterator[object]:
        # Historical randomized code returned B first.
        yield self.B
        yield self.A
        yield self.objectives


@dataclass(frozen=True)
class RandomizedVPSPCAWorkspace:
    """Reusable alpha-independent state for randomized VP-SPCA.

    Range finding, compression, the projection-residual diagnostic, and the
    PCA initialization do not depend on ``alpha``.  Preparing them once avoids
    repeating the expensive full-data passes in an alpha grid.  ``X`` is kept
    by reference rather than copied; callers must not mutate it while the
    workspace is in use.
    """

    X: FloatArray
    n_components: int
    oversample: int
    n_power_iter: int
    Q: FloatArray
    sketch: FloatArray
    projection_residual: float
    relative_projection_residual: float
    data_frobenius_norm: float
    initial_B: FloatArray
    sketch_step_size: float
    full_safe_step_size: float
    timings: dict[str, float]


def _validate_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _reorthogonalize(matrix: FloatArray) -> FloatArray:
    """Economic QR followed by a second QR to control loss of orthogonality."""

    q, _ = linalg.qr(matrix, mode="economic", check_finite=False)
    q, _ = linalg.qr(q, mode="economic", check_finite=False)
    return np.asarray(q, dtype=np.float64)


def _randomized_range_finder_prevalidated(
    X_array: FloatArray,
    target_rank: int,
    oversample: int,
    n_iter: int,
    random_state: int | np.random.Generator | None,
) -> FloatArray:
    sketch_size = min(min(X_array.shape), target_rank + oversample)
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    omega = rng.standard_normal((X_array.shape[1], sketch_size))
    Q = _reorthogonalize(X_array @ omega)

    # Alternating QR is essential for numerical stability when the spectrum is
    # steep; multiplying X X.T repeatedly before one final QR loses directions.
    for _ in range(n_iter):
        right_basis = _reorthogonalize(X_array.T @ Q)
        Q = _reorthogonalize(X_array @ right_basis)
    return Q


def randomized_range_finder(
    X: ArrayLike,
    target_rank: int,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: int | np.random.Generator | None = 42,
) -> FloatArray:
    """Find a left range basis using reorthogonalized power iterations."""

    X_array = _as_float_matrix(X)
    target_rank = _validate_n_components(X_array, target_rank)
    oversample = _validate_nonnegative_integer(oversample, name="oversample")
    n_iter = _validate_nonnegative_integer(n_iter, name="n_iter")
    return _randomized_range_finder_prevalidated(
        X_array,
        target_rank,
        oversample,
        n_iter,
        random_state,
    )


def _projection_residual_from_compressed(
    X: FloatArray,
    Q: FloatArray,
    compressed: FloatArray,
) -> tuple[float, float, float]:
    """Evaluate ``||(I-QQ.T)X||_F`` from an existing ``Q.T @ X``.

    The orthogonal norm identity is used when its relative gap is comfortably
    above round-off.  Near exact capture falls back to a direct blocked
    residual, retaining stability without charging that reconstruction pass
    to ordinary low-rank sketches.  Reusing ``compressed`` also avoids a
    redundant second multiplication by ``Q.T`` in every randomized fit.
    """

    data_norm = float(linalg.norm(X, ord="fro", check_finite=False))
    if data_norm == 0.0:
        return 0.0, 0.0, 0.0
    compressed_norm = float(linalg.norm(compressed, ord="fro", check_finite=False))
    norm_ratio = compressed_norm / data_norm
    relative_squared = (1.0 - norm_ratio) * (1.0 + norm_ratio)
    cancellation_guard = 64.0 * np.finfo(np.float64).eps
    if relative_squared > cancellation_guard:
        relative = float(np.sqrt(relative_squared))
        return float(data_norm * relative), relative, data_norm

    # Keep the two n_rows x block temporaries below roughly 8 MiB in total.
    bytes_per_column = 2 * X.shape[0] * X.dtype.itemsize
    block_columns = max(1, (8 * 1024 * 1024) // max(1, bytes_per_column))
    block_columns = min(X.shape[1], block_columns)
    residual = 0.0
    blocked_data_norm = 0.0
    for start in range(0, X.shape[1], block_columns):
        stop = min(X.shape[1], start + block_columns)
        block = X[:, start:stop]
        projected = Q @ compressed[:, start:stop]
        block_residual_norm = float(
            linalg.norm(block - projected, ord="fro", check_finite=False)
        )
        block_data_norm = float(linalg.norm(block, ord="fro", check_finite=False))
        residual = float(np.hypot(residual, block_residual_norm))
        blocked_data_norm = float(np.hypot(blocked_data_norm, block_data_norm))
    relative = residual / max(blocked_data_norm, np.finfo(float).eps)
    return float(residual), float(relative), float(blocked_data_norm)


def sketch_residual_diagnostic(X: ArrayLike, Q: ArrayLike) -> tuple[float, float]:
    """Return absolute and relative Frobenius projection residuals.

    An orthogonal norm identity handles ordinary residuals without an
    ``X``-sized allocation.  Near exact capture is evaluated directly in
    feature-column blocks to avoid catastrophic cancellation.
    """

    X_array = _as_float_matrix(X)
    Q_array = _as_float_matrix(Q, name="Q")
    if Q_array.shape[0] != X_array.shape[0]:
        raise ValueError("Q and X must have the same number of rows")
    gram = Q_array.T @ Q_array
    if not np.allclose(gram, np.eye(Q_array.shape[1]), atol=1e-8, rtol=1e-8):
        raise ValueError("Q must have orthonormal columns")

    compressed = Q_array.T @ X_array
    residual, relative, _ = _projection_residual_from_compressed(
        X_array,
        Q_array,
        compressed,
    )
    return residual, relative


def _full_diagnostics(
    X: FloatArray,
    A: FloatArray,
    B: FloatArray,
    alpha: float,
    step_size: float,
    data_frobenius_norm: float,
) -> tuple[FloatArray, float, float]:
    """Compute all full-data diagnostics with shared matrix products.

    The public VP helpers deliberately validate every standalone call.  Here
    all arrays were already validated, and sharing ``X @ B`` between the
    Procrustes factor and objective removes one full-data multiplication while
    retaining the mandatory full-data objective and stationarity checks.
    """

    XB = X @ B
    cross = X.T @ (XB)
    cross_norm = float(linalg.norm(cross, ord="fro", check_finite=False))
    scale = max(1.0, data_frobenius_norm * data_frobenius_norm)
    if cross_norm <= np.finfo(np.float64).eps * scale:
        A_full = _orthonormal_reference(A, B.shape[1])
    else:
        left, _, right_t = _thin_svd(cross)
        factor = left @ right_t
        left2, _, right_t2 = _thin_svd(factor)
        A_full = np.asarray(left2 @ right_t2, dtype=np.float64)

    XA = X @ A_full
    smooth = 0.5 * data_frobenius_norm * data_frobenius_norm
    smooth += 0.5 * float(np.sum(XB * XB))
    smooth -= float(np.sum(XA * XB))
    if smooth < 0.0 and smooth >= -100.0 * np.finfo(np.float64).eps * scale:
        smooth = 0.0
    objective = smooth + alpha * float(np.sum(np.abs(B)))

    gradient = X.T @ (X @ (B - A_full))
    prox = soft_threshold(B - step_size * gradient, step_size * alpha)
    mapping = (B - prox) / step_size
    return (
        A_full,
        objective,
        float(linalg.norm(mapping, ord="fro", check_finite=False)),
    )


def _sketch_residual_safe_step(
    sketch_step_size: float,
    sketch_is_zero: bool,
    projection_residual: float,
) -> float:
    """Return a safe full-data step without an exact SVD of the full matrix.

    With ``X = Q(Q.T X) + (I-QQ.T)X``, the spectral norm is at most the
    sketch spectral norm plus the residual Frobenius norm.  Squaring that bound
    gives a conservative Lipschitz constant for the full VP gradient.
    """

    # ``sketch_step_size`` and the PCA initialization came from the same thin
    # SVD.  Reusing it avoids immediately decomposing the sketch a second time.
    sketch_sigma = 0.0 if sketch_is_zero else 1.0 / np.sqrt(sketch_step_size)
    upper_sigma = sketch_sigma + float(projection_residual)
    return 1.0 if upper_sigma == 0.0 else 1.0 / (upper_sigma * upper_sigma)


def prepare_randomized_vp_spca_workspace(
    X: ArrayLike,
    n_components: int,
    *,
    oversample: int = 10,
    n_power_iter: int = 2,
    random_state: int | np.random.Generator | None = 42,
) -> RandomizedVPSPCAWorkspace:
    """Prepare alpha-independent randomized VP-SPCA state once.

    Reuse the returned workspace with
    :func:`fit_randomized_vp_spca_from_workspace` when evaluating several
    regularization values on the exact same matrix.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    oversample = _validate_nonnegative_integer(oversample, name="oversample")
    n_power_iter = _validate_nonnegative_integer(n_power_iter, name="n_power_iter")

    start = perf_counter()
    Q = _randomized_range_finder_prevalidated(
        X_array,
        n_components,
        oversample,
        n_power_iter,
        random_state,
    )
    after_range = perf_counter()
    sketch = Q.T @ X_array
    after_compression = perf_counter()
    residual, relative_residual, data_norm = _projection_residual_from_compressed(
        X_array,
        Q,
        sketch,
    )
    after_residual = perf_counter()
    initial_B, sketch_step_size = _pca_initial_loadings_and_step_size(
        sketch,
        n_components,
    )
    full_safe_step_size = _sketch_residual_safe_step(
        sketch_step_size,
        not bool(np.any(sketch)),
        residual,
    )
    after_initialization = perf_counter()

    return RandomizedVPSPCAWorkspace(
        X=X_array,
        n_components=n_components,
        oversample=oversample,
        n_power_iter=n_power_iter,
        Q=Q,
        sketch=sketch,
        projection_residual=residual,
        relative_projection_residual=relative_residual,
        data_frobenius_norm=data_norm,
        initial_B=initial_B,
        sketch_step_size=sketch_step_size,
        full_safe_step_size=full_safe_step_size,
        timings={
            "range_finder": after_range - start,
            "compress": after_compression - after_range,
            "residual_diagnostic": after_residual - after_compression,
            "initialization": after_initialization - after_residual,
            "total": after_initialization - start,
        },
    )


def _fit_randomized_vp_spca_from_workspace(
    workspace: RandomizedVPSPCAWorkspace,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    use_full_gradient: bool = False,
    finalize_full_iters: int = 0,
    verbose: bool = False,
    include_preparation_timings: bool = False,
) -> RandomizedVPSPCAResult:
    if not isinstance(workspace, RandomizedVPSPCAWorkspace):
        raise TypeError("workspace must be a RandomizedVPSPCAWorkspace")
    X_array = workspace.X
    n_components = workspace.n_components
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_positive_scalar(tol, name="tol")
    finalize_full_iters = _validate_nonnegative_integer(
        finalize_full_iters,
        name="finalize_full_iters",
    )
    if not isinstance(use_full_gradient, (bool, np.bool_)):
        raise TypeError("use_full_gradient must be boolean")

    sketch_objectives: list[float] = []
    sketch_iterations = 0
    full_iterations = 0
    sketch_converged: bool | None = None

    optimization_start = perf_counter()
    if use_full_gradient:
        phase_result = _fit_vp_spca_prevalidated(
            X_array,
            n_components,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            # The sketch spectral step is an informed proposal; full-data
            # backtracking certifies every accepted step without a full SVD.
            step_size=workspace.sketch_step_size,
            initial_B=workspace.initial_B,
            backtracking=True,
            verbose=verbose,
        )
    else:
        sketch_result = _fit_vp_spca_prevalidated(
            workspace.sketch,
            n_components,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            step_size=workspace.sketch_step_size,
            initial_B=workspace.initial_B,
            backtracking=True,
            verbose=verbose,
        )
        sketch_objectives = list(sketch_result.objectives)
        sketch_iterations = sketch_result.n_iter
        sketch_converged = bool(sketch_result.converged)
        phase_result = sketch_result

        if finalize_full_iters > 0:
            phase_result = _fit_vp_spca_prevalidated(
                X_array,
                n_components,
                alpha=alpha,
                max_iter=finalize_full_iters,
                tol=tol,
                # Backtracking makes this safe and avoids ``svdvals(X)``.
                step_size=workspace.sketch_step_size,
                initial_B=sketch_result.B,
                backtracking=True,
                verbose=verbose,
            )
            full_iterations = phase_result.n_iter

    after_iterations = perf_counter()
    if use_full_gradient or full_iterations:
        full_diagnostic_step_size = float(phase_result.step_sizes[-1])
        full_diagnostic_step_policy = "accepted_full_data_solver_step"
    else:
        full_diagnostic_step_size = workspace.full_safe_step_size
        full_diagnostic_step_policy = (
            "sketch_spectral_plus_projection_residual_upper_bound"
        )
    A_full, full_objective, full_stationarity = _full_diagnostics(
        X_array,
        phase_result.A,
        phase_result.B,
        alpha,
        full_diagnostic_step_size,
        workspace.data_frobenius_norm,
    )
    after_diagnostics = perf_counter()
    full_converged = bool(full_stationarity <= tol)

    preparation = workspace.timings if include_preparation_timings else {}
    optimization_time = after_iterations - optimization_start
    timings = {
        "range_finder": float(preparation.get("range_finder", 0.0)),
        "compress": float(preparation.get("compress", 0.0)),
        "residual_diagnostic": float(preparation.get("residual_diagnostic", 0.0)),
        "initialization": float(preparation.get("initialization", 0.0)),
        "optimization": optimization_time,
        "diagnostics": after_diagnostics - after_iterations,
        # Historical alias.
        "iter": optimization_time,
    }
    if use_full_gradient:
        status = phase_result.status
    elif finalize_full_iters > 0:
        status = (
            f"full refinement: {phase_result.status}; "
            f"sketch solve: {'converged' if sketch_converged else 'not converged'}"
        )
    else:
        status = (
            f"sketch solve: {phase_result.status}; full stationarity diagnostic: "
            f"{'passed' if full_converged else 'failed'}"
        )

    return RandomizedVPSPCAResult(
        A=A_full,
        B=phase_result.B,
        objectives=list(phase_result.objectives),
        stationarity=list(phase_result.stationarity),
        step_sizes=list(phase_result.step_sizes),
        converged=full_converged,
        sketch_converged=sketch_converged,
        full_converged=full_converged,
        n_iter=sketch_iterations
        + (phase_result.n_iter if use_full_gradient else full_iterations),
        alpha=alpha,
        status=status,
        Q=workspace.Q,
        sketch_residual=workspace.projection_residual,
        sketch_residual_relative=workspace.relative_projection_residual,
        full_objective=full_objective,
        full_stationarity=full_stationarity,
        full_diagnostic_step_size=full_diagnostic_step_size,
        full_diagnostic_step_policy=full_diagnostic_step_policy,
        used_full_gradient=bool(use_full_gradient),
        sketch_iterations=sketch_iterations,
        full_refinement_iterations=full_iterations,
        timings=timings,
        sketch_objectives=sketch_objectives,
    )


def fit_randomized_vp_spca_from_workspace(
    workspace: RandomizedVPSPCAWorkspace,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    use_full_gradient: bool = False,
    finalize_full_iters: int = 0,
    verbose: bool = False,
) -> RandomizedVPSPCAResult:
    """Fit one alpha using an existing randomized workspace.

    Preparation timings are intentionally zero in the returned per-alpha
    result.  Charge ``workspace.timings['total']`` once when reporting the
    end-to-end runtime of an alpha grid.
    """

    return _fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        use_full_gradient=use_full_gradient,
        finalize_full_iters=finalize_full_iters,
        verbose=verbose,
        include_preparation_timings=False,
    )


def fit_randomized_vp_spca(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.1,
    oversample: int = 10,
    n_power_iter: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
    use_full_gradient: bool = False,
    finalize_full_iters: int = 0,
    random_state: int | np.random.Generator | None = 42,
    verbose: bool = False,
) -> RandomizedVPSPCAResult:
    """Fit VP-SPCA on a randomized left sketch, optionally refining on ``X``.

    ``use_full_gradient=True`` uses the sketch only for initialization and
    every optimization step uses the full matrix. ``finalize_full_iters``
    instead runs the requested number of full-data iterations after a sketched
    solve.  Both full-data paths use objective backtracking rather than an
    exact full-matrix SVD to initialize their step size.
    """

    workspace = prepare_randomized_vp_spca_workspace(
        X,
        n_components,
        oversample=oversample,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    return _fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        use_full_gradient=use_full_gradient,
        finalize_full_iters=finalize_full_iters,
        verbose=verbose,
        include_preparation_timings=True,
    )


def _certified_randomized_vp_spca_from_workspace(
    workspace: RandomizedVPSPCAWorkspace,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    residual_tolerance: float = 0.05,
    full_stationarity_tolerance: float | None = None,
    full_refine_iters: int = 100,
    verbose: bool = False,
    include_preparation_timings: bool = False,
) -> RandomizedVPSPCAResult:
    """Residual-certified sketch solve with adaptive-restart full refinement.

    This 2025-style hybrid first solves the compressed problem.  It accepts
    that result only when both the range residual and a matrix-free full-data
    proximal-gradient mapping pass their thresholds.  Otherwise it switches
    to true value-function FISTA on the full matrix with monotone and gradient
    adaptive restarts.
    """

    if not isinstance(workspace, RandomizedVPSPCAWorkspace):
        raise TypeError("workspace must be a RandomizedVPSPCAWorkspace")
    X_array = workspace.X
    n_components = workspace.n_components
    residual_tolerance = _validate_nonnegative_scalar(
        residual_tolerance,
        name="residual_tolerance",
    )
    stationarity_tolerance = (
        _validate_positive_scalar(tol, name="tol")
        if full_stationarity_tolerance is None
        else _validate_positive_scalar(
            full_stationarity_tolerance,
            name="full_stationarity_tolerance",
        )
    )
    full_refine_iters = _validate_positive_integer(
        full_refine_iters,
        name="full_refine_iters",
    )

    result = _fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        use_full_gradient=False,
        finalize_full_iters=0,
        verbose=verbose,
        include_preparation_timings=include_preparation_timings,
    )

    b_spectral = float(linalg.svdvals(result.B, check_finite=False)[0])
    loss_gap_bound = 0.5 * result.sketch_residual**2 * (1.0 + b_spectral) ** 2
    residual_failed = result.sketch_residual_relative > residual_tolerance
    stationarity_failed = result.full_stationarity > stationarity_tolerance
    reasons: list[str] = []
    if residual_failed:
        reasons.append("projection residual")
    if stationarity_failed:
        reasons.append("full stationarity")
    refinement_triggered = bool(reasons)
    full_stationarity_before = result.full_stationarity
    monotone_restarts = 0
    adaptive_restarts = 0
    restart_count = 0

    if refinement_triggered:
        # Local import avoids a module cycle while making the extension's
        # adaptive-restart behavior explicit.
        from .accelerated import fit_accelerated_vp_spca

        refinement_start = perf_counter()
        refined = fit_accelerated_vp_spca(
            X_array,
            n_components,
            alpha=alpha,
            max_iter=full_refine_iters,
            tol=stationarity_tolerance,
            # Full-data backtracking validates this sketch-informed proposal.
            # Supplying it prevents an otherwise redundant full-matrix SVD.
            step_size=workspace.sketch_step_size,
            initial_B=result.B,
            acceleration=True,
            backtracking=True,
            monotone=True,
            adaptive_restart=True,
            verbose=verbose,
        )
        refinement_time = perf_counter() - refinement_start
        A_full, full_objective, full_stationarity_after = _full_diagnostics(
            X_array,
            refined.A,
            refined.B,
            float(alpha),
            float(refined.step_sizes[-1]),
            workspace.data_frobenius_norm,
        )
        monotone_restarts = refined.monotone_restarts
        adaptive_restarts = refined.adaptive_restarts
        restart_count = monotone_restarts + adaptive_restarts
        timings = dict(result.timings)
        timings["full_refinement"] = refinement_time
        result = replace(
            result,
            A=A_full,
            B=refined.B,
            objectives=list(refined.objectives),
            stationarity=list(refined.stationarity),
            step_sizes=list(refined.step_sizes),
            converged=full_stationarity_after <= stationarity_tolerance,
            full_converged=full_stationarity_after <= stationarity_tolerance,
            n_iter=result.sketch_iterations + refined.n_iter,
            status=(
                "certified after full refinement"
                if full_stationarity_after <= stationarity_tolerance
                else "full refinement stopped before stationarity certificate"
            ),
            full_objective=full_objective,
            full_stationarity=full_stationarity_after,
            full_diagnostic_step_size=float(refined.step_sizes[-1]),
            full_diagnostic_step_policy="accepted_full_data_solver_step",
            full_refinement_iterations=refined.n_iter,
            timings=timings,
        )
    else:
        full_stationarity_after = full_stationarity_before

    final_full_converged = bool(
        full_stationarity_after <= stationarity_tolerance
    )
    certified = final_full_converged and (
        refinement_triggered or result.sketch_residual_relative <= residual_tolerance
    )
    certificate = SketchRefinementCertificate(
        projection_residual=result.sketch_residual,
        relative_projection_residual=result.sketch_residual_relative,
        residual_tolerance=residual_tolerance,
        loss_gap_upper_bound=loss_gap_bound,
        full_stationarity_before=full_stationarity_before,
        full_stationarity_after=full_stationarity_after,
        full_stationarity_tolerance=stationarity_tolerance,
        refinement_triggered=refinement_triggered,
        trigger_reason=", ".join(reasons) if reasons else "none",
        monotone_restarts=monotone_restarts,
        adaptive_restarts=adaptive_restarts,
        restart_count=restart_count,
        certified=certified,
    )
    final_status = result.status
    if result.full_converged != final_full_converged:
        # The sketch-only fit initially reports against ``tol``.  When callers
        # explicitly select a different full-data certificate tolerance, keep
        # every public convergence field aligned with that one final decision.
        status_prefix = result.status.split(
            "; full stationarity diagnostic:",
            maxsplit=1,
        )[0]
        final_status = (
            f"{status_prefix}; full stationarity diagnostic: "
            f"{'passed' if final_full_converged else 'failed'} "
            "at certificate tolerance"
        )
    return replace(
        result,
        converged=final_full_converged,
        full_converged=final_full_converged,
        status=final_status,
        certificate=certificate,
    )


def certified_randomized_vp_spca_from_workspace(
    workspace: RandomizedVPSPCAWorkspace,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    residual_tolerance: float = 0.05,
    full_stationarity_tolerance: float | None = None,
    full_refine_iters: int = 100,
    verbose: bool = False,
) -> RandomizedVPSPCAResult:
    """Run certified randomized VP-SPCA from shared prepared state."""

    return _certified_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        residual_tolerance=residual_tolerance,
        full_stationarity_tolerance=full_stationarity_tolerance,
        full_refine_iters=full_refine_iters,
        verbose=verbose,
        include_preparation_timings=False,
    )


def certified_randomized_vp_spca(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.1,
    oversample: int = 10,
    n_power_iter: int = 2,
    max_iter: int = 500,
    tol: float = 1e-6,
    residual_tolerance: float = 0.05,
    full_stationarity_tolerance: float | None = None,
    full_refine_iters: int = 100,
    random_state: int | np.random.Generator | None = 42,
    verbose: bool = False,
) -> RandomizedVPSPCAResult:
    """Residual-certified sketch solve with optional full-data refinement."""

    workspace = prepare_randomized_vp_spca_workspace(
        X,
        n_components,
        oversample=oversample,
        n_power_iter=n_power_iter,
        random_state=random_state,
    )
    return _certified_randomized_vp_spca_from_workspace(
        workspace,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        residual_tolerance=residual_tolerance,
        full_stationarity_tolerance=full_stationarity_tolerance,
        full_refine_iters=full_refine_iters,
        verbose=verbose,
        include_preparation_timings=True,
    )


def randomized_spca_vp(
    X: ArrayLike,
    n_components: int,
    alpha: float = 0.1,
    oversample: int = 50,
    n_iter: int = 8,
    max_iter: int = 500,
    tol: float = 1e-6,
    use_full_gradient: bool = False,
    finalize_full_iters: int = 0,
    verbose: bool = True,
    return_timings: bool = False,
    random_state: int | np.random.Generator | None = 42,
) -> (
    tuple[FloatArray, FloatArray, list[float]]
    | tuple[
        FloatArray,
        FloatArray,
        list[float],
        dict[str, float],
    ]
):
    """Backward-compatible wrapper returning ``B, A, objectives``."""

    result = fit_randomized_vp_spca(
        X,
        n_components,
        alpha=alpha,
        oversample=oversample,
        n_power_iter=n_iter,
        max_iter=max_iter,
        tol=tol,
        use_full_gradient=use_full_gradient,
        finalize_full_iters=finalize_full_iters,
        random_state=random_state,
        verbose=verbose,
    )
    if return_timings:
        return result.B, result.A, result.objectives, result.timings
    return result.B, result.A, result.objectives


__all__ = [
    "RandomizedVPSPCAWorkspace",
    "RandomizedVPSPCAResult",
    "SketchRefinementCertificate",
    "certified_randomized_vp_spca",
    "certified_randomized_vp_spca_from_workspace",
    "fit_randomized_vp_spca",
    "fit_randomized_vp_spca_from_workspace",
    "prepare_randomized_vp_spca_workspace",
    "randomized_range_finder",
    "randomized_spca_vp",
    "sketch_residual_diagnostic",
]

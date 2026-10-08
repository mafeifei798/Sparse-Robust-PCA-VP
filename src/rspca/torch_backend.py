"""Optional matrix-free PyTorch backend for accelerated VP-SPCA.

PyTorch is deliberately an optional dependency.  Import this module only when
the ``torch`` extra is installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


def _validate_nonnegative_scalar(value: float, *, name: str) -> float:
    """Return a finite, non-negative floating-point parameter."""

    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real scalar") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_positive_scalar(value: float, *, name: str) -> float:
    result = _validate_nonnegative_scalar(value, name=name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


@dataclass(frozen=True)
class TorchVPSPCAResult:
    """Result returned by :func:`fit_torch_accelerated_vp_spca`."""

    A: Any
    B: Any
    objectives: list[float]
    stationarity: list[float]
    step_sizes: list[float]
    converged: bool
    n_iter: int
    alpha: float
    status: str
    device: str


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if resolved.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return torch.device("cpu")
    return resolved


def _as_tensor(
    X: np.ndarray | torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(X, torch.Tensor):
        dtype = torch.float32 if device.type == "mps" else X.dtype
        if not (dtype.is_floating_point if hasattr(dtype, "is_floating_point") else False):
            dtype = torch.float32
        result = X.detach().to(device=device, dtype=dtype)
    else:
        array = np.asarray(X)
        dtype = torch.float32 if device.type == "mps" else torch.float64
        result = torch.as_tensor(array, dtype=dtype, device=device)
    if result.ndim != 2:
        raise ValueError("X must be a two-dimensional matrix")
    if not bool(torch.isfinite(result).all().item()):
        raise ValueError("X must contain only finite values")
    return result


def soft_threshold_torch(value: torch.Tensor, threshold: float) -> torch.Tensor:
    """Elementwise proximal operator of ``threshold * ||.||_1``."""

    threshold = _validate_nonnegative_scalar(threshold, name="threshold")
    return torch.sign(value) * torch.clamp(torch.abs(value) - threshold, min=0.0)


def _xtx_matmul(X: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Return ``X.T @ X @ matrix`` without materialising the Gram matrix."""

    return X.mT @ (X @ matrix)


def update_A_procrustes_torch(X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Evaluate the variable-projection map ``A(B)`` matrix-free."""

    z = _xtx_matmul(X, B)
    try:
        u, _, vh = torch.linalg.svd(z, full_matrices=False)
        return u @ vh
    except (NotImplementedError, RuntimeError) as error:
        # Some MPS releases do not implement the small SVD used here.  Falling
        # back for this n-by-r matrix retains the algorithm without constructing
        # an n-by-n Gram matrix.
        if z.device.type != "mps":
            raise error
        z_cpu = z.detach().cpu()
        u, _, vh = torch.linalg.svd(z_cpu, full_matrices=False)
        return (u @ vh).to(device=z.device, dtype=z.dtype)


def _smooth_value(
    X: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
    half_norm_x_sq: torch.Tensor,
) -> torch.Tensor:
    """Compute ``0.5 ||X-XBA^T||_F^2`` without a dense residual."""

    xb = X @ B
    xa = X @ A
    return half_norm_x_sq + 0.5 * torch.sum(xb * xb) - torch.sum(xa * xb)


def _value_and_factor(
    X: torch.Tensor,
    B: torch.Tensor,
    half_norm_x_sq: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = update_A_procrustes_torch(X, B)
    return _smooth_value(X, B, A, half_norm_x_sq), A


def _lipschitz_estimate(X: torch.Tensor, iterations: int = 40) -> float:
    """Estimate ``||X||_2**2`` by deterministic matrix-free power iteration."""

    n_features = X.shape[1]
    # A fixed all-ones start can lie exactly in X's null space, causing a
    # nonzero matrix to receive the unsafe fallback estimate zero.  Starting
    # from a coordinate belonging to a nonzero column guarantees that the
    # first X.T @ X product is nonzero whenever X is nonzero.
    column_norms = torch.linalg.vector_norm(X, dim=0)
    start_index = int(torch.argmax(column_norms).item())
    if float(column_norms[start_index].item()) == 0.0:
        return 0.0
    vector = torch.zeros((n_features, 1), dtype=X.dtype, device=X.device)
    vector[start_index, 0] = 1.0
    eigenvalue = torch.zeros((), dtype=X.dtype, device=X.device)
    for _ in range(iterations):
        image = _xtx_matmul(X, vector)
        norm = torch.linalg.vector_norm(image)
        if float(norm.item()) == 0.0:
            return 0.0
        vector = image / norm
        eigenvalue = torch.sum(vector * _xtx_matmul(X, vector))
    return max(float(eigenvalue.item()), 0.0)


def _stationarity(
    X: torch.Tensor,
    B: torch.Tensor,
    A: torch.Tensor,
    step_size: float,
    alpha: float,
) -> float:
    gradient = _xtx_matmul(X, B - A)
    prox = soft_threshold_torch(B - step_size * gradient, step_size * alpha)
    mapping = (B - prox) / step_size
    return float(torch.linalg.matrix_norm(mapping, ord="fro").item())


@torch.no_grad()
def fit_torch_accelerated_vp_spca(
    X: np.ndarray | torch.Tensor,
    n_components: int,
    *,
    alpha: float = 0.1,
    max_iter: int = 500,
    tol: float = 1e-6,
    step_size: float | None = None,
    initial_B: np.ndarray | torch.Tensor | None = None,
    fista: bool = True,
    backtracking: bool = True,
    backtracking_shrink: float = 0.5,
    step_growth: float = 1.1,
    monotone_restart: bool = True,
    device: str | torch.device = "cpu",
    return_numpy: bool = True,
    verbose: bool = False,
) -> TorchVPSPCAResult:
    """Fit accelerated VP-SPCA using matrix-free PyTorch operations.

    The orthogonal factor is recomputed at every extrapolated point ``Y``.
    When a momentum proposal raises the accepted composite objective, a
    monotone restart recomputes the proposal from the last accepted ``B``.
    """

    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    max_iter = _validate_positive_integer(max_iter, name="max_iter")
    tol = _validate_nonnegative_scalar(tol, name="tol")
    backtracking_shrink = _validate_positive_scalar(
        backtracking_shrink,
        name="backtracking_shrink",
    )
    step_growth = _validate_positive_scalar(step_growth, name="step_growth")
    if backtracking_shrink >= 1.0:
        raise ValueError("backtracking_shrink must lie in (0, 1)")
    if step_growth < 1.0:
        raise ValueError("step_growth must be at least one")

    resolved_device = _resolve_device(device)
    matrix = _as_tensor(X, device=resolved_device)
    n_samples, n_features = matrix.shape
    n_components = _validate_positive_integer(n_components, name="n_components")
    if n_components > min(n_samples, n_features):
        raise ValueError("n_components must lie in [1, min(X.shape)]")

    if initial_B is None:
        try:
            _, _, vh = torch.linalg.svd(matrix, full_matrices=False)
        except (NotImplementedError, RuntimeError) as error:
            if matrix.device.type != "mps":
                raise error
            _, _, vh_cpu = torch.linalg.svd(matrix.detach().cpu(), full_matrices=False)
            vh = vh_cpu.to(device=matrix.device, dtype=matrix.dtype)
        B = vh.mT[:, :n_components].clone()
    else:
        B = _as_tensor(initial_B, device=resolved_device).to(dtype=matrix.dtype)
        if B.shape != (n_features, n_components):
            raise ValueError(
                f"initial_B must have shape {(n_features, n_components)}, got {tuple(B.shape)}"
            )

    A = update_A_procrustes_torch(matrix, B)
    half_norm_x_sq = 0.5 * torch.sum(matrix * matrix)
    smooth, A = _value_and_factor(matrix, B, half_norm_x_sq)
    objective = float((smooth + alpha * torch.sum(torch.abs(B))).item())

    if step_size is None:
        # Power iteration is an estimate from below.  Backtracking certifies it
        # before acceptance; without backtracking use the Frobenius-squared
        # upper bound so the automatic step cannot be unsafe.
        lipschitz = (
            _lipschitz_estimate(matrix)
            if backtracking
            else float(torch.sum(matrix * matrix).item())
        )
        step = 1.0 / lipschitz if lipschitz > 0 else 1.0
    else:
        step = _validate_positive_scalar(step_size, name="step_size")

    Y = B.clone()
    momentum = 1.0
    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    converged = False

    def proposal(
        origin: torch.Tensor,
        trial_step: float,
        *,
        force_backtracking: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        value_origin, A_origin = _value_and_factor(matrix, origin, half_norm_x_sq)
        gradient = _xtx_matmul(matrix, origin - A_origin)
        for _ in range(61):
            if not np.isfinite(trial_step) or trial_step <= 0.0:
                raise RuntimeError("backtracking encountered an invalid step size")
            candidate = soft_threshold_torch(
                origin - trial_step * gradient,
                trial_step * alpha,
            )
            value_candidate, A_candidate = _value_and_factor(
                matrix, candidate, half_norm_x_sq
            )
            difference = candidate - origin
            upper = (
                value_origin
                + torch.sum(gradient * difference)
                + 0.5 / trial_step * torch.sum(difference * difference)
            )
            candidate_objective = float(
                (value_candidate + alpha * torch.sum(torch.abs(candidate))).item()
            )
            finite_proposal = (
                bool(torch.isfinite(candidate).all().item())
                and bool(torch.isfinite(A_candidate).all().item())
                and bool(torch.isfinite(value_candidate).item())
                and bool(torch.isfinite(upper).item())
                and np.isfinite(candidate_objective)
            )
            majorized = finite_proposal and bool(
                (value_candidate <= upper + 1e-10).item()
            )
            if finite_proposal and (
                majorized or (not backtracking and not force_backtracking)
            ):
                return candidate, A_candidate, candidate_objective, trial_step
            if not backtracking and not force_backtracking:
                raise RuntimeError("proposal produced non-finite values")
            trial_step *= backtracking_shrink
            if not np.isfinite(trial_step) or trial_step < torch.finfo(matrix.dtype).eps:
                raise RuntimeError("backtracking step size underflow")
        raise RuntimeError("torch VP-SPCA backtracking failed to terminate")

    for iteration in range(1, max_iter + 1):
        B_previous = B
        candidate, A_candidate, candidate_objective, used_step = proposal(Y, step)

        if (
            fista
            and monotone_restart
            and candidate_objective > objective + 1e-12 * max(1.0, abs(objective))
        ):
            # Re-evaluate A and the gradient at the accepted point; simply
            # resetting Y after accepting the bad proposal is not a restart.
            momentum = 1.0
            Y = B_previous.clone()
            candidate, A_candidate, candidate_objective, used_step = proposal(
                Y,
                used_step,
                force_backtracking=True,
            )

        B = candidate
        A = A_candidate
        objective = candidate_objective
        criterion = _stationarity(matrix, B, A, used_step, alpha)
        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(used_step)

        if verbose and (iteration <= 10 or iteration % 50 == 0):
            print(
                f"iter={iteration:4d} objective={objective:.8e} "
                f"stationarity={criterion:.3e} step={used_step:.3e}"
            )
        if criterion <= tol:
            converged = True
            break

        if fista:
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum**2))
            Y = B + ((momentum - 1.0) / next_momentum) * (B - B_previous)
            momentum = float(next_momentum)
        else:
            Y = B.clone()
            momentum = 1.0
        step = used_step * step_growth if backtracking else used_step

    status = "converged" if converged else "maximum iterations reached"
    if return_numpy:
        A_output: Any = A.detach().cpu().numpy()
        B_output: Any = B.detach().cpu().numpy()
    else:
        A_output = A
        B_output = B
    return TorchVPSPCAResult(
        A=A_output,
        B=B_output,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=float(alpha),
        status=status,
        device=str(resolved_device),
    )


def spca_vp_fista_bt_torch(
    X_in: np.ndarray | torch.Tensor,
    r: int,
    alpha: float = 0.1,
    max_iter: int = 500,
    eps: float = 1e-6,
    gamma: float | None = None,
    adaptive_alpha: bool = False,
    fista: bool = True,
    backtracking: bool = True,
    bt_shrink: float = 0.5,
    bt_expand: float = 1.1,
    restart_monotone: bool = True,
    device: str | torch.device = "cpu",
    return_numpy: bool = True,
) -> tuple[Any, Any, list[float], list[float]]:
    """Compatibility wrapper for the historical top-level Torch function."""

    if adaptive_alpha:
        raise ValueError(
            "adaptive_alpha changed the optimized objective between iterations and "
            "is no longer supported; select alpha on validation data instead"
        )
    result = fit_torch_accelerated_vp_spca(
        X_in,
        r,
        alpha=alpha,
        max_iter=max_iter,
        tol=eps,
        step_size=gamma,
        fista=fista,
        backtracking=backtracking,
        backtracking_shrink=bt_shrink,
        step_growth=bt_expand,
        monotone_restart=restart_monotone,
        device=device,
        return_numpy=return_numpy,
    )
    return result.A, result.B, result.objectives, result.stationarity


# The original module accidentally exported a name that did not exist.  Keep
# this exact, tested public surface.
__all__ = [
    "TorchVPSPCAResult",
    "fit_torch_accelerated_vp_spca",
    "soft_threshold_torch",
    "spca_vp_fista_bt_torch",
    "update_A_procrustes_torch",
]

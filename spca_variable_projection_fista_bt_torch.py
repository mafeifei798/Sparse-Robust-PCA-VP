"""
PyTorch implementation of Improved SPCA (VP-SPCA variant) with:
1) FISTA acceleration (with optional monotone restart)
2) Backtracking line search for step size
3) Nonstationarity criterion aligned with effective gamma and alpha

API mirrors the NumPy version in `spca_variable_projection_fista_bt.py` and supports
NumPy input/outputs as well as running on CPU or CUDA.
"""

from __future__ import annotations

from typing import Tuple, List, Optional

import numpy as np
import torch


def _to_tensor(X, device: str | torch.device) -> torch.Tensor:
    """Convert input to float32 torch.Tensor on the target device.

    Ensures dtype=torch.float32 BEFORE moving to device to avoid MPS float64 errors.
    """
    if isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=torch.float32)
    Xnp: np.ndarray = np.asarray(X)
    if Xnp.dtype != np.float32:
        Xnp = Xnp.astype(np.float32, copy=False)
    return torch.from_numpy(Xnp).to(device)


def _to_numpy(T: torch.Tensor) -> np.ndarray:
    return T.detach().cpu().numpy()


def soft_threshold_torch(Z: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.sign(Z) * torch.clamp(torch.abs(Z) - tau, min=0.0)


def update_A_procrustes_torch(G: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Solve orthogonal Procrustes for A via SVD of Z = G @ B, where G = X^T X.
    Returns A = U @ V^T with A^T A = I.
    """
    Z = G @ B  # (n, r)
    # Fallback for MPS where SVD may be unsupported
    try:
        U, _, Vh = torch.linalg.svd(Z, full_matrices=False)
        return U @ Vh
    except NotImplementedError:
        Z_cpu = Z.detach().cpu()
        Uc, _, Vhc = torch.linalg.svd(Z_cpu, full_matrices=False)
        A_cpu = Uc @ Vhc
        return A_cpu.to(Z.device, dtype=Z.dtype)


def g_tilde_torch(B: torch.Tensor, A: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    """
    Smooth part g(B; A) up to constant 0.5||X||_F^2:
        g_tilde(B; A) = 0.5 * trace(B^T G B) - trace(A^T G B)
    """
    term1 = 0.5 * torch.sum(B * (G @ B))
    term2 = torch.sum((A.T @ G) * B.T)
    return term1 - term2


def objective_total_torch(
    B: torch.Tensor,
    A: torch.Tensor,
    G: torch.Tensor,
    normX2_half: float,
    alpha_eff: float,
) -> float:
    gtil = g_tilde_torch(B, A, G)
    return float(normX2_half + gtil.item() + alpha_eff * torch.sum(torch.abs(B)).item())


def compute_nonstationarity_effective_torch(
    A: torch.Tensor,
    B: torch.Tensor,
    G: torch.Tensor,
    gamma_eff: float,
    alpha_eff: float,
) -> float:
    grad = G @ (B - A)
    B_prox = soft_threshold_torch(B - gamma_eff * grad, gamma_eff * alpha_eff)
    mapping = (B - B_prox) / gamma_eff
    return float(torch.linalg.matrix_norm(mapping, ord='fro').item())


def spectral_norm_from_gram(G: torch.Tensor, iters: int = 30) -> float:
    """Approximate largest singular value of X using Gram matrix G = X^T X.

    Returns sigma_max ≈ sqrt(lambda_max(G)) using power iteration without SVD,
    which is friendly to MPS.
    """
    n = G.shape[0]
    v = torch.randn(n, 1, device=G.device, dtype=G.dtype)
    v = v / (torch.norm(v) + 1e-12)
    for _ in range(iters):
        v = G @ v
        v = v / (torch.norm(v) + 1e-12)
    lam = torch.squeeze(v.T @ (G @ v))
    lam = torch.clamp(lam, min=0.0)
    return float(torch.sqrt(lam).item())


@torch.no_grad()
def spca_vp_fista_bt_torch(
    X_in: np.ndarray | torch.Tensor,
    r: int,
    alpha: float = 0.1,
    max_iter: int = 500,
    eps: float = 1e-6,
    gamma: Optional[float] = None,
    adaptive_alpha: bool = False,
    fista: bool = True,
    backtracking: bool = True,
    bt_shrink: float = 0.5,
    bt_expand: float = 1.2,
    restart_monotone: bool = True,
    device: str | torch.device = 'cpu',
    return_numpy: bool = True,
) -> Tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor, List[float], List[float]]:
    """
    Improved SPCA with FISTA + backtracking in PyTorch.

    Returns (A, B, objectives, criteria). If return_numpy=True, A/B are numpy.
    """

    # Resolve device string with support for CUDA/MPS
    if isinstance(device, str):
        if device == 'cuda' and (not torch.cuda.is_available()):
            device = 'cpu'
        if device == 'mps':
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                device = 'cpu'
        device = torch.device(device)
    X = _to_tensor(X_in, device=device)
    # Ensure float32 dtype explicitly (important for MPS)
    X = X.to(torch.float32)

    m, n = X.shape
    G = X.T @ X
    froX = float(torch.linalg.matrix_norm(X, ord='fro').item())
    normX2_half = 0.5 * (froX * froX)

    # Initial gamma = 1 / ||X||_2^2
    if gamma is None:
        # Avoid SVD on MPS: use power iteration on Gram matrix
        sigma_max = spectral_norm_from_gram(G, iters=30)
        gamma_k = 1.0 / (sigma_max * sigma_max + 1e-12) if sigma_max > 0 else 1.0
    else:
        gamma_k = float(gamma)

    # Initialization from right singular vectors of X
    # Work with CPU SVD if CUDA lacks gesvdj support for certain sizes
    # but torch.linalg.svd works on both; we keep tensors on device.
    # SVD can be unavailable on MPS; fall back to CPU then move back
    try:
        _, _, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.mT
    except NotImplementedError:
        X_cpu = X.detach().cpu()
        _, _, Vh_cpu = torch.linalg.svd(X_cpu, full_matrices=False)
        V = Vh_cpu.mT.to(X.device, dtype=X.dtype)
    A = V[:, :r].clone()
    B = V[:, :r].clone()

    # FISTA vars
    Y = B.clone()
    t = 1.0

    def compute_alpha_eff(curr_A: torch.Tensor, curr_B: torch.Tensor) -> float:
        if not adaptive_alpha:
            return alpha
        XB = X @ curr_B
        residual = XB @ curr_A.T - X
        res_norm = float(torch.linalg.matrix_norm(residual, ord='fro').item()) / (froX + 1e-12)
        return alpha * float(np.clip(res_norm, 0.5, 1.0))

    alpha_eff = compute_alpha_eff(A, B)
    obj_prev = objective_total_torch(B, A, G, normX2_half, alpha_eff)

    objectives: List[float] = []
    criteria: List[float] = []

    for k in range(max_iter):
        grad = G @ (Y - A)
        alpha_eff = compute_alpha_eff(A, B)
        gY = g_tilde_torch(Y, A, G).item()

        accepted = False
        gamma_trial = gamma_k
        while True:
            B_trial = soft_threshold_torch(Y - gamma_trial * grad, gamma_trial * alpha_eff)
            diff = B_trial - Y
            lhs = g_tilde_torch(B_trial, A, G).item()
            rhs = gY + float(torch.sum(grad * diff).item()) + (0.5 / gamma_trial) * float(torch.sum(diff * diff).item())
            if lhs <= rhs + 1e-12:
                accepted = True
                break
            if not backtracking:
                break
            gamma_trial *= bt_shrink

        if not accepted:
            B_next = B_trial
            gamma_k = gamma_trial
        else:
            B_next = B_trial
            gamma_k = gamma_trial
            if backtracking:
                gamma_k *= bt_expand

        if fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Y_next = B_next + ((t - 1.0) / t_next) * (B_next - B)
        else:
            t_next = 1.0
            Y_next = B_next.clone()

        A_next = update_A_procrustes_torch(G, B_next)
        obj = objective_total_torch(B_next, A_next, G, normX2_half, alpha_eff)

        if restart_monotone and obj > obj_prev + 1e-14 and fista:
            t_next = 1.0
            Y_next = B_next.clone()

        crit = compute_nonstationarity_effective_torch(A_next, B_next, G, gamma_k, alpha_eff)
        objectives.append(obj)
        criteria.append(crit)

        # Convergence check
        if crit <= eps:
            B = B_next
            A = A_next
            break

        # Prepare next
        obj_prev = obj
        B = B_next
        A = A_next
        Y = Y_next
        t = t_next

    if return_numpy:
        return _to_numpy(A), _to_numpy(B), objectives, criteria
    return A, B, objectives, criteria


__all__ = [
    "spca_variable_projection_fista_bt_torch",
    "soft_threshold_torch",
    "update_A_procrustes_torch",
]



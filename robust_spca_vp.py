import numpy as np
from scipy import linalg


def soft_threshold(Z: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)


def update_A_procrustes_with_S(Xs: np.ndarray, B: np.ndarray, S: np.ndarray) -> np.ndarray:
    Z = (Xs - S).T @ (Xs @ B)
    U, _, Vt = linalg.svd(Z, full_matrices=False)
    return U @ Vt


def robust_spca_vp(
    X_in: np.ndarray,
    r: int,
    alpha_r: float = 0.012,
    kappa: float | None = None,
    max_iter_r: int = 500,
    tol_r: float = 1e-6,
    huber_tau: float = 1.0,
    kappa_min: float = 0.005,
    kappa_max: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Spectral normalization so gamma=1
    _, Ssvd, Vt = linalg.svd(X_in, full_matrices=False)
    sigma_max = Ssvd[0] if Ssvd.size > 0 else 1.0
    Xs = X_in / sigma_max
    V = Vt.T
    A_r = V[:, :r].copy()
    B_r = V[:, :r].copy()
    S_r = np.zeros_like(Xs)
    gamma = 1.0
    for k in range(max_iter_r):
        R = Xs @ B_r @ A_r.T - Xs + S_r
        grad_B = Xs.T @ (R @ A_r)
        B_next = soft_threshold(B_r - gamma * grad_B, gamma * alpha_r)
        if linalg.norm(B_next - B_r, 'fro') <= tol_r:
            B_r = B_next
            break
        B_r = B_next
        A_r = update_A_procrustes_with_S(Xs, B_r, S_r)
        residual = Xs - Xs @ B_r @ A_r.T
        if kappa is None:
            med = np.median(residual)
            sigma_hat = np.median(np.abs(residual - med)) / 0.6745 + 1e-12
            k_eff = float(np.clip(huber_tau * sigma_hat, kappa_min, kappa_max))
        else:
            k_eff = kappa
        S_r = soft_threshold(residual, k_eff)
    # Unscale S back
    return B_r, A_r, S_r * sigma_max


__all__ = [
    "robust_spca_vp",
    "soft_threshold",
    "update_A_procrustes_with_S",
]






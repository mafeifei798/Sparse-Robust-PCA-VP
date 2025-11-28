"""
Improved SPCA (VP-SPCA variant) with:
1) FISTA acceleration with optional monotone restart
2) Backtracking line search for proximal step size
3) Nonstationarity criterion aligned with effective gamma and alpha

This implementation uses a variable projection scheme identical in spirit to
`spca_variable_projection.py`, but replaces the B-update with a FISTA + backtracking
proximal gradient step on the smooth part g(B; A) = 0.5||X B A^T - X||_F^2
while keeping A fixed during the B-step. The gradient is simplified using the
Gram matrix G = X^T X: ∇g(B; A) = G @ (B - A).
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def soft_threshold(Z: np.ndarray, tau: float) -> np.ndarray:
    """Elementwise soft-thresholding for L1 proximal operator."""
    return np.sign(Z) * np.maximum(np.abs(Z) - tau, 0.0)


def update_A_procrustes(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve orthogonal Procrustes for A:
        A = argmin_A ||X - X B A^T||_F^2  s.t. A^T A = I
    Closed-form: if Z = X^T X B = U Σ V^T, then A = U V^T
    """
    Z = X.T @ X @ B
    U, _, Vt = linalg.svd(Z, full_matrices=False)
    return U @ Vt


def compute_nonstationarity_effective(
    A: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    gamma_eff: float,
    alpha_eff: float,
) -> float:
    """
    Nonstationarity based on proximal gradient mapping at (A, B) using the
    effective step size and regularization actually used in the B-step.

    mapping = (B - prox_{gamma_eff * alpha_eff}(B - gamma_eff * grad)) / gamma_eff
    with grad = G @ (B - A).
    """
    grad = G @ (B - A)
    B_prox = soft_threshold(B - gamma_eff * grad, gamma_eff * alpha_eff)
    mapping = (B - B_prox) / gamma_eff
    return float(linalg.norm(mapping, ord='fro'))


def _g_tilde(B: np.ndarray, A: np.ndarray, G: np.ndarray) -> float:
    """
    Smooth part g(B; A) up to an additive constant 0.5||X||_F^2, i.e.,
        g_tilde(B; A) = 0.5 * trace(B^T G B) - trace(A^T G B)
    Using this eliminates constants in backtracking conditions.
    """
    # term1 = 0.5 * ||X B||_F^2 = 0.5 * trace(B^T G B)
    term1 = 0.5 * float(np.sum(B * (G @ B)))
    # term2 = trace(A^T G B)
    term2 = float(np.sum((A.T @ G) * B.T))
    return term1 - term2


def _objective_total(
    B: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    normX2_half: float,
    alpha_eff: float,
) -> float:
    """
    Full objective: 0.5||X - XBA^T||_F^2 + alpha_eff * ||B||_1
    Using identities: 0.5||X - XBA^T||_F^2 = 0.5||X||_F^2 + g_tilde(B; A)
    """
    gtil = _g_tilde(B, A, G)
    return normX2_half + gtil + alpha_eff * float(np.sum(np.abs(B)))


def spca_algorithm_4_1_fista_bt(
    X: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """
    Variable-projection SPCA with FISTA acceleration and backtracking.

    Parameters
    ----------
    X : (m, n) array
        Centered data matrix
    r : int
        Number of components
    alpha : float
        L1 regularization weight
    max_iter : int
        Maximum iterations
    eps : float
        Convergence tolerance on nonstationarity mapping
    gamma : float | None
        Initial step size (if None, uses 1 / ||X||_2^2)
    adaptive_alpha : bool
        If True, scale alpha by residual magnitude each iteration (mild)
    fista : bool
        If True, use Nesterov acceleration
    backtracking : bool
        If True, use composite backtracking for step-size selection
    bt_shrink : float
        Factor in (0,1) to shrink step-size when backtracking fails
    bt_expand : float
        Factor >1 to optionally expand step-size after acceptance
    restart_monotone : bool
        If True, restart FISTA when objective increases

    Returns
    -------
    A : (n, r) array
        Orthogonal matrix A (A^T A = I)
    B : (n, r) array
        Sparse loading matrix
    objectives : list[float]
        Objective values per iteration
    criteria : list[float]
        Nonstationarity criterion values per iteration (aligned gamma/alpha)
    """

    m, n = X.shape
    # Precompute Gram matrix and norms
    G = X.T @ X
    froX = float(linalg.norm(X, ord='fro'))
    normX2_half = 0.5 * (froX * froX)

    # Spectral step-size default: 1 / ||X||_2^2
    if gamma is None:
        sigma_max = float(linalg.norm(X, ord=2))
        gamma = 1.0 / (sigma_max * sigma_max + 1e-12) if sigma_max > 0 else 1.0

    # Initialization from right singular vectors of X
    _, _, Vt = linalg.svd(X, full_matrices=False)
    V = Vt.T
    A = V[:, :r].copy()
    B = V[:, :r].copy()

    # FISTA variables
    Y = B.copy()
    t = 1.0
    gamma_k = float(gamma)

    print("Improved SPCA - FISTA + Backtracking")
    print(f"Data: {m}×{n}, Components: {r}, alpha={alpha}, gamma0={gamma_k:.3e}")
    print("-" * 60)

    objectives: list[float] = []
    criteria: list[float] = []

    # Helper to compute effective alpha when requested
    def compute_alpha_eff(curr_A: np.ndarray, curr_B: np.ndarray) -> float:
        if not adaptive_alpha:
            return alpha
        # Mild adaptation based on normalized residual magnitude
        XB = X @ curr_B
        residual = XB @ curr_A.T - X
        res_norm = float(linalg.norm(residual, ord='fro')) / (froX + 1e-12)
        return alpha * float(np.clip(res_norm, 0.5, 1.0))

    # Initial objective
    alpha_eff = compute_alpha_eff(A, B)
    obj_prev = _objective_total(B, A, G, normX2_half, alpha_eff)

    for k in range(max_iter):
        # 1) Compute gradient at Y with A fixed (variable projection scheme)
        grad = G @ (Y - A)

        # 2) Choose effective alpha for this iteration
        alpha_eff = compute_alpha_eff(A, B)

        # 3) Backtracking line search on composite function (without constants)
        gY = _g_tilde(Y, A, G)
        accepted = False
        gamma_trial = gamma_k
        while True:
            # Prox step at Y
            B_trial = soft_threshold(Y - gamma_trial * grad, gamma_trial * alpha_eff)
            # Check majorization: g(B_trial) <= g(Y) + <grad, B_trial - Y> + (1/(2γ))||B_trial - Y||^2
            diff = B_trial - Y
            lhs = _g_tilde(B_trial, A, G)
            rhs = gY + float(np.sum(grad * diff)) + (0.5 / gamma_trial) * float(np.sum(diff * diff))
            if lhs <= rhs + 1e-12:
                accepted = True
                break
            if not backtracking:
                break
            gamma_trial *= bt_shrink

        if not accepted:
            # Fall back to the trial even if no backtracking is used
            B_next = B_trial
            gamma_k = gamma_trial
        else:
            B_next = B_trial
            gamma_k = gamma_trial
            # Optional expansion for next iter to accelerate
            if backtracking:
                gamma_k *= bt_expand

        # 4) FISTA momentum update
        if fista:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            Y_next = B_next + ((t - 1.0) / t_next) * (B_next - B)
        else:
            t_next = 1.0
            Y_next = B_next.copy()

        # 5) Update A via Procrustes with new B
        A_next = update_A_procrustes(X, B_next)

        # 6) Objective with aligned alpha_eff
        obj = _objective_total(B_next, A_next, G, normX2_half, alpha_eff)

        # 7) Monotone restart if objective increases
        if restart_monotone and obj > obj_prev + 1e-14 and fista:
            t_next = 1.0
            Y_next = B_next.copy()

        # 8) Nonstationarity with effective gamma/alpha used for the accepted step
        crit = compute_nonstationarity_effective(A_next, B_next, G, gamma_k, alpha_eff)

        objectives.append(obj)
        criteria.append(crit)

        if k % 100 == 0 or k < 10:
            print(f"Iter {k:4d}: Obj = {obj:.6e}, T(A,B) = {crit:.6e}, gamma = {gamma_k:.3e}")

        # Convergence check
        if crit <= eps:
            print(f"\u2713 Converged at iteration {k}: T(A,B) = {crit:.6e} \u2264 {eps}")
            B = B_next
            A = A_next
            break

        # Prepare next iteration
        obj_prev = obj
        B = B_next
        A = A_next
        Y = Y_next
        t = t_next

    else:
        print(f"\u2717 Reached max iterations {max_iter}")

    # Final stats
    sparsity = float((np.abs(B) < 1e-10).mean())
    ortho_err = float(linalg.norm(A.T @ A - np.eye(r), ord='fro'))
    recon_err = float(linalg.norm(X - X @ B @ A.T, ord='fro') / (froX + 1e-12))

    print("-" * 60)
    print("Final Results (Improved SPCA):")
    print(f"  Sparsity ratio: {sparsity:.3f}")
    print(f"  Orthogonality error ||A^T A - I||_F: {ortho_err:.3e}")
    print(f"  Relative reconstruction error: {recon_err:.4f}")

    return A, B, objectives, criteria


def demo():
    """Small demo on synthetic low-rank + noise data."""
    np.random.seed(0)
    m, n, r_true = 120, 40, 3
    U = linalg.orth(np.random.randn(m, r_true))
    V = linalg.orth(np.random.randn(n, r_true))
    S = np.diag([7.5, 4.0, 2.5])
    X_clean = U @ S @ V.T
    X = X_clean + 0.10 * np.random.randn(m, n)
    X = X - X.mean(axis=0, keepdims=True)

    r = 3
    alpha = 0.02
    A, B, objs, crits = spca_algorithm_4_1_fista_bt(
        X, r=r, alpha=alpha, max_iter=800, eps=1e-8, adaptive_alpha=False,
        fista=True, backtracking=True, restart_monotone=True,
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(objs); axes[0].set_title('Objective (Improved)'); axes[0].set_xlabel('Iteration'); axes[0].grid(True, alpha=0.3)
    axes[1].semilogy(crits); axes[1].set_title('Nonstationarity T(A,B)'); axes[1].set_xlabel('Iteration'); axes[1].grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig('spca_alg41_fista_bt_convergence.png', dpi=150, bbox_inches='tight'); plt.show()

    return A, B, objs, crits


if __name__ == "__main__":
    demo()




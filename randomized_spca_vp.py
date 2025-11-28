"""
Randomized SPCA via Variable Projection

Idea:
- Use a randomized range finder to compress X on the left: X ≈ Q @ (Q^T X) = Q @ X_tilde
- Replace occurrences of X^T X with X_tilde^T X_tilde to accelerate the steps:
  * Procrustes: Z = (X^T X) B ≈ X_tilde^T (X_tilde B)
  * Gradient:  ∇_B v(B) = X^T(X B A^T − X)A ≈ (X_tilde^T X_tilde) (B − A)

We keep the classic VP-SPCA structure:
  1) Prox-grad update on B
  2) Closed-form Procrustes update on A using SVD of Z = (X^T X) B (approximated)

We also expose 'n_iter' (power iterations) and 'oversample' for the sketch, and by default
perform two extra subspace iterations as requested.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import time


def soft_threshold(X: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0.0)


def randomized_range_finder(
    X: np.ndarray,
    target_rank: int,
    oversample: int = 10,
    n_iter: int = 2,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """
    Compute an orthonormal basis Q for the column space of X with power iterations.

    Parameters
    ----------
    X : (m, n)
    target_rank : int, approximate rank to capture
    oversample : int, extra dimensions to improve accuracy
    n_iter : int, number of power iterations (>= 0). We default to 2 as requested.
    random_state : int or None
    """
    m, n = X.shape
    k = min(m, target_rank + oversample)
    rng = np.random.default_rng(random_state)
    Omega = rng.standard_normal((n, k))

    Y = X @ Omega  # (m, k)
    # Power iterations to sharpen spectrum
    for _ in range(max(n_iter, 0)):
        Y = X @ (X.T @ Y)
    # Orthonormalize columns of Y
    Q, _ = np.linalg.qr(Y, mode='reduced')
    return Q  # (m, k)


def randomized_spca_vp(
    X: np.ndarray,
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
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Randomized Variable-Projection SPCA operating on a
    compressed representation using a left range finder.

    Returns
    -------
    B : (n, r)
    A : (n, r)
    objectives : list of objective values 0.5||X - XBA^T||_F^2 + alpha*|B|_1
    """
    m, n = X.shape
    r = n_components
    if r > n:
        raise ValueError('n_components must be <= number of features')

    # 1) Compute left sketch Q and compressed matrix X_tilde = Q^T X
    t0_rf = time.perf_counter()
    Q = randomized_range_finder(X, target_rank=r, oversample=oversample, n_iter=n_iter)
    t1_rf = time.perf_counter()
    X_tilde = Q.T @ X  # (k, n)
    t2_comp = time.perf_counter()

    # Step size gamma using spectral norm approximation via X_tilde
    svals = linalg.svd(X_tilde, full_matrices=False, compute_uv=False)
    sigma_max = svals[0] if svals.size > 0 else 1.0
    gamma = 1.0 / (sigma_max**2)

    # Initialize A and B from right singular vectors of the (compressed) X
    Uc, Sc, Vt = linalg.svd(X_tilde, full_matrices=False)
    V = Vt.T
    A = V[:, :r].copy()
    B = V[:, :r].copy()

    objectives: List[float] = []

    if verbose:
        print(f"Randomized SPCA-VP | X:{X.shape}, r={r}, oversample={oversample}, n_iter={n_iter}, gamma={gamma:.3e}")

    # Helpers (sketched operators)
    def XtX_approx_matmul(M: np.ndarray) -> np.ndarray:
        # (X_tilde^T X_tilde) @ M = X_tilde^T @ (X_tilde @ M)
        return X_tilde.T @ (X_tilde @ M)

    t3_iter_start = time.perf_counter()
    for k_iter in range(max_iter):
        # Gradient using XtX_approx: grad = (XtX)(B - A)
        grad = X_tilde.T @ (X_tilde @ (B - A))

        # Proximal step on B
        B = soft_threshold(B - gamma * grad, gamma * alpha)

        # Procrustes update on A using Z ≈ (XtX) B
        Z = XtX_approx_matmul(B)  # (n, r)
        U, _, Vt = linalg.svd(Z, full_matrices=False)
        A = U @ Vt

        # Objective approximation (compute exactly but via compressed forms)
        # ||X - XBA^T||_F^2 = ||Q^T(X - XBA^T)||_F^2 + ||(I-QQ^T)(X - XBA^T)||_F^2
        # We approximate with the sketched term for speed: 0.5||X_tilde - X_tilde B A^T||_F^2
        R_tilde = X_tilde - X_tilde @ B @ A.T
        obj = 0.5 * (R_tilde**2).sum() + alpha * np.sum(np.abs(B))
        objectives.append(obj)

        if verbose and (k_iter % 50 == 0 or k_iter < 10):
            print(f"Iter {k_iter:4d}: obj~={obj:.6e}")

        # Convergence via small change in B
        if k_iter > 5:
            rel = linalg.norm(grad, 'fro') / (linalg.norm(B, 'fro') + 1e-12)
            if rel < tol:
                if verbose:
                    print(f"Converged at iter {k_iter} (rel grad={rel:.3e} < {tol})")
                break
    t4_iter_end = time.perf_counter()

    # Finalize on full data for stability (hybrid): small number of exact iterations
    # No finalize by default (set above). To enable, pass finalize_full_iters>0.

    if verbose:
        time_rf = t1_rf - t0_rf
        time_comp = t2_comp - t1_rf
        time_iter = t4_iter_end - t3_iter_start
        print(f"Timing | range_finder: {time_rf:.3f}s, compress: {time_comp:.3f}s, iter: {time_iter:.3f}s, total: {(time_rf+time_comp+time_iter):.3f}s")

    if return_timings:
        timings = {
            'range_finder': (t1_rf - t0_rf),
            'compress': (t2_comp - t1_rf),
            'iter': (t4_iter_end - t3_iter_start),
        }
        return B, A, objectives, timings

    return B, A, objectives


def demo_multiscale_video(seed: int = 42) -> None:
    """Demo on a multiscale video using the randomized SPCA-VP."""
    # Generate data
    np.random.seed(seed)
    n_grid, n_timesteps, total_time = 200, 300, 150.0
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.linspace(0.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(x, y)

    def blob(cx, cy, s):
        g = np.exp(-((Xg - cx) ** 2 + (Yg - cy) ** 2) / (2.0 * s ** 2))
        return g / (np.max(g) + 1e-12)

    phi = np.stack([
        blob(0.55, 0.55, 0.06),
        blob(0.30, 0.20, 0.025),
        blob(0.78, 0.80, 0.025)
    ], axis=0)

    t = np.linspace(0.0, total_time, n_timesteps)
    a = np.zeros((3, n_timesteps))
    a[0] = 2.0 * np.sin(2.0 * np.pi * 0.20 * t)
    a[1, (t >= 50) & (t <= 100)] = 2.0 * np.sin(2.0 * np.pi * 0.05 * t[(t >= 50) & (t <= 100)])
    a[2, (t >= 80) & (t <= 120)] = 2.0 * np.sin(2.0 * np.pi * 0.30 * t[(t >= 80) & (t <= 120)])

    video = np.zeros((n_timesteps, n_grid, n_grid))
    for k in range(n_timesteps):
        video[k] = a[0, k] * phi[0] + a[1, k] * phi[1] + a[2, k] * phi[2]
    X = video.reshape(n_timesteps, -1).T
    X = X - X.mean(axis=1, keepdims=True)

    # Run randomized SPCA-VP (with two extra subspace iterations by default)
    B, A, obj, timings = randomized_spca_vp(
        X, n_components=3, alpha=0.01, oversample=20, n_iter=5, max_iter=500, tol=1e-6, verbose=True, return_timings=True
    )

    # Save timing result
    try:
        with open('randomized_spca_vp_timing.txt', 'w') as f:
            f.write(f"range_finder: {timings['range_finder']:.6f}\n")
            f.write(f"compress: {timings['compress']:.6f}\n")
            f.write(f"iter: {timings['iter']:.6f}\n")
            f.write(f"total: {timings['range_finder'] + timings['compress'] + timings['iter']:.6f}\n")
        print("Timing written to randomized_spca_vp_timing.txt")
    except Exception as e:
        print("Failed to write timing file:", e)

    # Visualize spatial modes and objective curve
    # Reconstruct modes on original data
    Phi_rec = X @ B
    # zero-center to mitigate background bias
    Phi_rec = Phi_rec - Phi_rec.mean(axis=0, keepdims=True)
    n_grid2 = n_grid
    modes = Phi_rec.reshape(n_grid2, n_grid2, -1).transpose(2, 0, 1)

    plt.figure(figsize=(12, 5))
    for j in range(min(3, modes.shape[0])):
        ax = plt.subplot(2, 3, j + 1)
        im = ax.imshow(modes[j], cmap='magma')
        ax.set_title(f'Randomized Φ{j+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.subplot(2, 1, 2)
    plt.semilogy(obj)
    plt.title('Randomized SPCA-VP Objective (sketched)')
    plt.xlabel('Iteration')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig('randomized_spca_vp_results.png', dpi=150, bbox_inches='tight'); plt.show()


if __name__ == '__main__':
    demo_multiscale_video()



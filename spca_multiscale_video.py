"""
Multiscale video example with Variable-Projection SPCA

This script generates a synthetic multiscale video dataset on a 200×200 grid
over 300 time steps with pairwise-overlapping temporal activations, then applies a VP-SPCA implementation
(`spca_variable_projection.spca_vp`) to recover spatial modes and their
temporal activations. Results are visualized.
"""

import numpy as np
from scipy import linalg
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from typing import Tuple

from spca_variable_projection import spca_vp
from robust_spca_vp import robust_spca_vp


def generate_multiscale_video(
    n_grid: int = 200,
    n_timesteps: int = 300,
    total_time: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic multiscale video dataset.

    Returns
    -------
    X : ndarray, shape (p, n)
        Data matrix with p = n_grid^2 spatial pixels and n = n_timesteps frames.
    phi_true : ndarray, shape (3, n_grid, n_grid)
        Ground-truth spatial modes Φ₁..Φ₃.
    a_true : ndarray, shape (3, n_timesteps)
        Ground-truth temporal coefficients a₁(t)..a₃(t).
    t : ndarray, shape (n_timesteps,)
        Time vector.
    """

    # Spatial grid
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.linspace(0.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(x, y)

    def gaussian_blob(cx: float, cy: float, sigma: float) -> np.ndarray:
        blob = np.exp(-((Xg - cx) ** 2 + (Yg - cy) ** 2) / (2.0 * sigma ** 2))
        return blob / np.maximum(blob.max(), 1e-12)

    # Three localized spatial modes (Φ₁..Φ₃)
    phi_true = np.zeros((3, n_grid, n_grid))
    phi_true[0] = gaussian_blob(0.55, 0.55, 0.06)
    phi_true[1] = gaussian_blob(0.30, 0.20, 0.025)
    phi_true[2] = gaussian_blob(0.78, 0.80, 0.025)

    # Time vector and overlapping temporal activations a_j(t) (pairwise only)
    t = np.linspace(0.0, total_time, n_timesteps)
    a_true = np.zeros((3, n_timesteps))

    def smooth_window(t, t_start, t_end, ramp=5.0):
        """Smooth step window using logistic tapers."""
        left = 1.0 / (1.0 + np.exp(-(t - t_start) / ramp))
        right = 1.0 / (1.0 + np.exp(-(t - t_end) / ramp))
        return np.clip(left - right, 0.0, 1.0)

    def rect_window(t, start, end, ramp=2.0):
        return smooth_window(t, start, end, ramp=ramp)

    # a1(t): 全程近乎持续的正弦
    win1 = rect_window(t, 0.0, total_time, ramp=6.0)
    a_true[0] = 2.0 * np.sin(2.0 * np.pi * 0.08 * t) * win1

    # a2(t): 0 → 平台 → 局部振荡 → 回到 0（仅在 45~105 激活）
    plateau = rect_window(t, 45.0, 65.0, ramp=2.0)
    burst = rect_window(t, 65.0, 105.0, ramp=2.0)
    a_true[1] = 0.8 * plateau + 1.8 * np.sin(2.0 * np.pi * 0.055 * t + 0.35 * np.pi) * burst

    # a3(t): 后段高频振荡（仅在 100 之后激活）
    burst3 = rect_window(t, 100.0, total_time, ramp=2.0)
    a_true[2] = 2.0 * np.sin(2.0 * np.pi * 0.12 * t + 0.1 * np.pi) * burst3

    # Assemble video frames and build data matrix X (p × n)
    video = np.zeros((n_timesteps, n_grid, n_grid))
    for k in range(n_timesteps):
        frame = (
            a_true[0, k] * phi_true[0]
            + a_true[1, k] * phi_true[1]
            + a_true[2, k] * phi_true[2]
        )
        video[k] = frame

    X = video.reshape(n_timesteps, -1).T

    # Center snapshots in time (remove per-pixel mean)
    X = X - X.mean(axis=1, keepdims=True)

    return X, phi_true, a_true, t


def generate_multiscale_video_classic(
    n_grid: int = 200,
    n_timesteps: int = 300,
    total_time: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Classic multiscale video used in earlier demos (sinusoidal windows).
    """
    x = np.linspace(0.0, 1.0, n_grid)
    y = np.linspace(0.0, 1.0, n_grid)
    Xg, Yg = np.meshgrid(x, y)

    def gaussian_blob(cx: float, cy: float, sigma: float) -> np.ndarray:
        blob = np.exp(-((Xg - cx) ** 2 + (Yg - cy) ** 2) / (2.0 * sigma ** 2))
        return blob / np.maximum(blob.max(), 1e-12)

    phi_true = np.zeros((3, n_grid, n_grid))
    phi_true[0] = gaussian_blob(0.55, 0.55, 0.06)
    phi_true[1] = gaussian_blob(0.30, 0.20, 0.025)
    phi_true[2] = gaussian_blob(0.78, 0.80, 0.025)

    t = np.linspace(0.0, total_time, n_timesteps)
    a_true = np.zeros((3, n_timesteps))
    a_true[0] = 2.0 * np.sin(2.0 * np.pi * 0.20 * t)
    mask2 = (t >= 50.0) & (t <= 100.0)
    a_true[1, mask2] = 2.0 * np.sin(2.0 * np.pi * 0.05 * t[mask2])
    mask3 = (t >= 80.0) & (t <= 120.0)
    a_true[2, mask3] = 2.0 * np.sin(2.0 * np.pi * 0.30 * t[mask3])

    video = np.zeros((n_timesteps, n_grid, n_grid))
    for k in range(n_timesteps):
        frame = (
            a_true[0, k] * phi_true[0]
            + a_true[1, k] * phi_true[1]
            + a_true[2, k] * phi_true[2]
        )
        video[k] = frame

    X = video.reshape(n_timesteps, -1).T
    X = X - X.mean(axis=1, keepdims=True)
    return X, phi_true, a_true, t


def align_components(
    modes_rec: np.ndarray, modes_true: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align recovered spatial modes with ground truth by maximizing absolute
    correlation, using the Hungarian assignment. Also return signs.
    """
    r = modes_true.shape[0]
    # Flatten for correlation matrix [r_true × r_rec]
    T = modes_true.reshape(r, -1)
    R = modes_rec.reshape(r, -1)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    corr = Tn @ Rn.T  # shape (r, r)

    # Maximize |corr|
    cost = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)
    signs = np.sign(corr[row_ind, col_ind])
    return col_ind, signs


def main():
    # 1) Generate data
    X, phi_true, a_true, t = generate_multiscale_video()
    p, n = X.shape
    print(f"Generated multiscale video: p={p}, n={n}")

    # 2) Run VP-SPCA
    r = 3
    alpha = 9  # mild sparsity to promote localized modes
    A, B, obj, crit = spca_vp(X, r=r, alpha=alpha, max_iter=500, eps=1e-6)

    # 3) Recover spatial modes Φ = X B and temporal activations A(t)
    Phi_rec = X @ B  # shape (p, r)

    # Normalize spatial modes for visualization and rescale A accordingly
    Phi_rec_2d = Phi_rec.reshape(-1, r)
    norms = np.maximum(np.linalg.norm(Phi_rec_2d, axis=0, keepdims=False), 1e-12)
    Phi_rec_2d = Phi_rec_2d / norms
    A = A * norms  # preserve X ≈ Φ A^T

    # Reshape modes to images
    n_grid = int(np.sqrt(Phi_rec.shape[0]))
    modes_rec = Phi_rec_2d.reshape(n_grid, n_grid, r).transpose(2, 0, 1)

    # 4) Align recovered components with ground truth by correlation
    perm, signs = align_components(modes_rec, phi_true)
    modes_rec = modes_rec[perm] * signs[:, None, None]
    A_aligned = A[:, perm] * signs[None, :]

    # Least-squares rescale for visualization: align amplitude of recovered a_j
    # with the ground truth a_true[j] to minimize L2 error (only for plotting)
    A_plot = A_aligned.copy()
    for j in range(r):
        num = float(A_plot[:, j].T @ a_true[j])
        den = float(A_plot[:, j].T @ A_plot[:, j]) + 1e-12
        scale_j = num / den
        A_plot[:, j] *= scale_j

    # 5) Visualization of recovered spatial modes and temporal activations
    plt.figure(figsize=(16, 9))
    for j in range(r):
        # Spatial mode
        ax = plt.subplot(2, r, j + 1)
        im = ax.imshow(modes_rec[j], cmap='magma')
        ax.set_title(f'Recovered Φ{j+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Temporal activation (overlay truth if available)
        ax2 = plt.subplot(2, r, r + j + 1)
        ax2.plot(t, A_plot[:, j], label='Recovered a'+str(j+1))
        ax2.plot(t, a_true[j], '--', label='True a'+str(j+1), alpha=0.8)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        if j == 0:
            ax2.set_ylabel('Activation')
        ax2.legend()

    plt.tight_layout()
    plt.savefig('multiscale_spca_6_1_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 6) Objective and stationarity curves
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(obj)
    plt.title('Objective (VP-SPCA)')
    plt.xlabel('Iteration')
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.semilogy(crit)
    plt.title('Nonstationarity T(A,B)')
    plt.xlabel('Iteration')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('multiscale_spca_6_1_convergence.png', dpi=150, bbox_inches='tight')
    plt.show()

    # ===================== Robust SPCA via variable projection =====================
    # Classic video for robust SPCA, then add sparse gross corruptions
    X_classic, phi_classic, a_classic, t_classic = generate_multiscale_video_classic()
    rng = np.random.default_rng(123)
    X_cor = X_classic.copy()
    noise_mask = rng.random(X_cor.shape) < 0.05
    X_cor[noise_mask] += rng.normal(0.0, 3.0, size=noise_mask.sum())

    B_r, A_r, S_r = robust_spca_vp(X_cor, r=3)
    XBA = X_cor @ B_r @ A_r.T

    # Additional visualization for one representative frame
    n_grid = int(np.sqrt(X_classic.shape[0]))
    t_idx = 100
    frame_truth = (
        a_classic[0, t_idx] * phi_classic[0]
        + a_classic[1, t_idx] * phi_classic[1]
        + a_classic[2, t_idx] * phi_classic[2]
    )
    frame_cor = X_cor[:, t_idx].reshape(n_grid, n_grid)
    frame_recon = XBA[:, t_idx].reshape(n_grid, n_grid)

    # Post-threshold S for clearer display
    residual_final = X_cor - XBA
    medF = np.median(residual_final)
    sigmaF = np.median(np.abs(residual_final - medF)) / 0.6745 + 1e-12
    thr_show = 1.5 * sigmaF
    S_vis = S_r.copy()
    S_vis[np.abs(S_vis) < thr_show] = 0.0
    frame_outliers = S_vis[:, t_idx].reshape(n_grid, n_grid)

    vmin, vmax = np.percentile(frame_truth, 1), np.percentile(frame_truth, 99)
    nz = np.abs(frame_outliers)[np.abs(frame_outliers) > 0]
    s_lim = (np.percentile(nz, 98) if nz.size > 0 else np.percentile(np.abs(frame_outliers), 98))

    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 4, 1); im=ax.imshow(frame_truth, cmap='magma', vmin=vmin, vmax=vmax); ax.set_title('(a) Truth'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = plt.subplot(1, 4, 2); im=ax.imshow(frame_cor, cmap='magma', vmin=vmin, vmax=vmax); ax.set_title('(b) Corrupted'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = plt.subplot(1, 4, 3); im=ax.imshow(frame_recon, cmap='magma', vmin=vmin, vmax=vmax); ax.set_title('(c) Reconstruction'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax = plt.subplot(1, 4, 4); im=ax.imshow(frame_outliers, cmap='seismic', vmin=-s_lim, vmax=s_lim); ax.set_title('(d) Outliers'); ax.axis('off'); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig('robust_spca_vp_fig6.png', dpi=150, bbox_inches='tight'); plt.show()


if __name__ == "__main__":
    main()



"""
Compare VP-SPCA vs Improved SPCA (FISTA + Backtracking)
on the multiscale video from spca_multiscale_video.generate_multiscale_video.
"""

from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.optimize import linear_sum_assignment

from spca_variable_projection import spca_vp
from spca_variable_projection_fista_bt import spca_algorithm_4_1_fista_bt as spca_vp_fista_bt
from spca_multiscale_video import generate_multiscale_video
try:
    from spca_variable_projection_fista_bt_torch import spca_vp_fista_bt_torch
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    fro = lambda Z: linalg.norm(Z, 'fro')
    return float(fro(X - X_hat) / (fro(X) + 1e-12))


def align_components(modes_rec: np.ndarray, modes_true: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rloc = modes_true.shape[0]
    T = modes_true.reshape(rloc, -1)
    R = modes_rec.reshape(rloc, -1)
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
    Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-12)
    corr = Tn @ Rn.T
    cost = -np.abs(corr)
    row_ind, col_ind = linear_sum_assignment(cost)
    signs = np.sign(corr[row_ind, col_ind])
    return col_ind, signs


def main() -> None:
    # 1) Data identical to spca_multiscale_video_6_1.py
    X, phi_true, a_true, t = generate_multiscale_video()
    p, n = X.shape
    r = 3
    alpha = 10

    # Preview initial frame for consistent color scaling
    n_grid = int(np.sqrt(p))
    frame0 = X[:, 0].reshape(n_grid, n_grid)
    vmin_ref, vmax_ref = np.percentile(frame0, 1), np.percentile(frame0, 99)

    timings: dict[str, float] = {}
    errors: dict[str, float] = {}
    if TORCH_AVAILABLE:
        # Prefer CUDA, then Apple Metal (MPS), otherwise CPU
        if getattr(torch, 'cuda', None) and torch.cuda.is_available():  # type: ignore[attr-defined]
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = 'cpu'

    # 2) SPCA (VP-SPCA)
    t0 = time.time()
    A_spca, B_spca, obj_spca, crit_spca = spca_vp(
        X, r=r, alpha=alpha, max_iter=100, eps=1e-6, adaptive_alpha=True
    )
    timings['SPCA'] = time.time() - t0
    Xhat_spca = X @ B_spca @ A_spca.T
    errors['SPCA'] = reconstruction_error(X, Xhat_spca)

    # 3) Improved SPCA (FISTA + Backtracking)
    t0 = time.time()
    A_imp, B_imp, obj_imp, crit_imp = spca_vp_fista_bt(
        X, r=r, alpha=alpha, max_iter=100, eps=1e-6,
        adaptive_alpha=True, fista=True, backtracking=True, restart_monotone=True,
    )
    timings['SPCA(FISTA+BT)'] = time.time() - t0
    Xhat_imp = X @ B_imp @ A_imp.T
    errors['SPCA(FISTA+BT)'] = reconstruction_error(X, Xhat_imp)

    # 3b) Torch Improved SPCA (FISTA + Backtracking), optional
    if TORCH_AVAILABLE:
        t0 = time.time()
        A_t, B_t, obj_t, crit_t = spca_vp_fista_bt_torch(
            X, r=r, alpha=alpha, max_iter=100, eps=1e-6,
            adaptive_alpha=True, fista=True, backtracking=True, restart_monotone=True,
            device=device, return_numpy=True,
        )
        timings[f'SPCA(Torch:{device})'] = time.time() - t0
        Xhat_t = X @ B_t @ A_t.T
        errors[f'SPCA(Torch:{device})'] = reconstruction_error(X, Xhat_t)
    else:
        print("[Info] PyTorch not installed. Skipping Torch variant. To enable it: pip install torch")

    # 4) Report
    print("\nTiming (s):")
    for k, v in timings.items():
        print(f"  {k:18s}: {v:.3f}")
    print("\nReconstruction error:")
    for k, v in errors.items():
        print(f"  {k:18s}: {v:.4f}")

    # 5) Time bar (log)
    methods = list(timings.keys())
    tvals = [timings[m] for m in methods]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = ['steelblue', 'orange', 'green'][:len(methods)]
    bars = ax.bar(methods, tvals, color=colors)
    ax.set_title('Computation Time (s, log)')
    ax.set_ylabel('Seconds (log scale)')
    ax.set_yscale('log')
    for b, v in zip(bars, tvals):
        ax.text(b.get_x() + b.get_width()/2, v + 1e-3, f"{v:.3f}", ha='center', va='bottom', fontsize=9)
    ax.grid(True, axis='y', alpha=0.3, which='both')
    plt.tight_layout(); plt.savefig('alg41_vs_fista_time.png', dpi=150, bbox_inches='tight'); plt.show()

    # 6) Visualizations similar to spca_multiscale_video_6_1.py
    # ---- For SPCA ----
    Phi_rec = X @ B_spca
    Phi_rec_2d = Phi_rec.reshape(-1, r)
    norms = np.maximum(np.linalg.norm(Phi_rec_2d, axis=0, keepdims=False), 1e-12)
    Phi_rec_2d = Phi_rec_2d / norms
    A_tmp = A_spca * norms
    n_grid_loc = int(np.sqrt(Phi_rec.shape[0]))
    modes_rec = Phi_rec_2d.reshape(n_grid_loc, n_grid_loc, r).transpose(2, 0, 1)
    perm, signs = align_components(modes_rec, phi_true)
    modes_rec = modes_rec[perm] * signs[:, None, None]
    A_aligned = A_tmp[:, perm] * signs[None, :]
    A_plot = A_aligned.copy()
    for j in range(r):
        num = float(A_plot[:, j].T @ a_true[j])
        den = float(A_plot[:, j].T @ A_plot[:, j]) + 1e-12
        scale_j = num / den
        A_plot[:, j] *= scale_j
    plt.figure(figsize=(16, 9))
    for j in range(r):
        ax = plt.subplot(2, r, j + 1)
        im = ax.imshow(modes_rec[j], cmap='magma', vmin=vmin_ref, vmax=vmax_ref)
        ax.set_title(f'Recovered Φ{j+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax2 = plt.subplot(2, r, r + j + 1)
        ax2.plot(t, A_plot[:, j], label='Recovered a'+str(j+1))
        ax2.plot(t, a_true[j], '--', label='True a'+str(j+1), alpha=0.8)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        if j == 0:
            ax2.set_ylabel('Activation')
        ax2.legend()
    plt.tight_layout(); plt.savefig('alg41_vs_fista_components_SPCA.png', dpi=150, bbox_inches='tight'); plt.show()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(obj_spca); plt.title('Objective (VP-SPCA)'); plt.xlabel('Iteration'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2); plt.semilogy(crit_spca); plt.title('Nonstationarity T(A,B)'); plt.xlabel('Iteration'); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig('alg41_vs_fista_convergence_SPCA.png', dpi=150, bbox_inches='tight'); plt.show()

    # ---- For Improved SPCA (FISTA + BT) ----
    Phi_rec = X @ B_imp
    Phi_rec_2d = Phi_rec.reshape(-1, r)
    norms = np.maximum(np.linalg.norm(Phi_rec_2d, axis=0, keepdims=False), 1e-12)
    Phi_rec_2d = Phi_rec_2d / norms
    A_tmp = A_imp * norms
    n_grid_loc = int(np.sqrt(Phi_rec.shape[0]))
    modes_rec = Phi_rec_2d.reshape(n_grid_loc, n_grid_loc, r).transpose(2, 0, 1)
    perm, signs = align_components(modes_rec, phi_true)
    modes_rec = modes_rec[perm] * signs[:, None, None]
    A_aligned = A_tmp[:, perm] * signs[None, :]
    A_plot = A_aligned.copy()
    for j in range(r):
        num = float(A_plot[:, j].T @ a_true[j])
        den = float(A_plot[:, j].T @ A_plot[:, j]) + 1e-12
        scale_j = num / den
        A_plot[:, j] *= scale_j
    plt.figure(figsize=(16, 9))
    for j in range(r):
        ax = plt.subplot(2, r, j + 1)
        im = ax.imshow(modes_rec[j], cmap='magma', vmin=vmin_ref, vmax=vmax_ref)
        ax.set_title(f'Recovered Φ{j+1}-FISTA+BT')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax2 = plt.subplot(2, r, r + j + 1)
        ax2.plot(t, A_plot[:, j], label='Recovered a'+str(j+1)+'-FISTA+BT')
        ax2.plot(t, a_true[j], '--', label='True a'+str(j+1), alpha=0.8)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)
        if j == 0:
            ax2.set_ylabel('Activation')
        ax2.legend()
    plt.tight_layout(); plt.savefig('alg41_vs_fista_components_SPCA_FISTA_BT.png', dpi=150, bbox_inches='tight'); plt.show()
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(obj_imp); plt.title('Objective (Improved)'); plt.xlabel('Iteration'); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2); plt.semilogy(crit_imp); plt.title('Nonstationarity T(A,B)'); plt.xlabel('Iteration'); plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout(); plt.savefig('alg41_vs_fista_convergence_Improved.png', dpi=150, bbox_inches='tight'); plt.show()

    # ---- For Torch Improved SPCA ----
    if TORCH_AVAILABLE:
        Phi_rec = X @ B_t
        Phi_rec_2d = Phi_rec.reshape(-1, r)
        norms = np.maximum(np.linalg.norm(Phi_rec_2d, axis=0, keepdims=False), 1e-12)
        Phi_rec_2d = Phi_rec_2d / norms
        A_tmp = A_t * norms
        n_grid_loc = int(np.sqrt(Phi_rec.shape[0]))
        modes_rec = Phi_rec_2d.reshape(n_grid_loc, n_grid_loc, r).transpose(2, 0, 1)
        perm, signs = align_components(modes_rec, phi_true)
        modes_rec = modes_rec[perm] * signs[:, None, None]
        A_aligned = A_tmp[:, perm] * signs[None, :]
        A_plot = A_aligned.copy()
        for j in range(r):
            num = float(A_plot[:, j].T @ a_true[j])
            den = float(A_plot[:, j].T @ A_plot[:, j]) + 1e-12
            scale_j = num / den
            A_plot[:, j] *= scale_j
        plt.figure(figsize=(16, 9))
        for j in range(r):
            ax = plt.subplot(2, r, j + 1)
            im = ax.imshow(modes_rec[j], cmap='magma', vmin=vmin_ref, vmax=vmax_ref)
            ax.set_title(f'Recovered Φ{j+1}-Torch')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax2 = plt.subplot(2, r, r + j + 1)
            ax2.plot(t, A_plot[:, j], label='Recovered a'+str(j+1)+'-Torch')
            ax2.plot(t, a_true[j], '--', label='True a'+str(j+1), alpha=0.8)
            ax2.set_xlim(t[0], t[-1])
            ax2.set_xlabel('Time (s)')
            ax2.grid(True, alpha=0.3)
            if j == 0:
                ax2.set_ylabel('Activation')
            ax2.legend()
        plt.tight_layout(); plt.savefig('alg41_vs_fista_components_SPCA_Torch.png', dpi=150, bbox_inches='tight'); plt.show()
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1); plt.plot(obj_t); plt.title('Objective (Torch)'); plt.xlabel('Iteration'); plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2); plt.semilogy(crit_t); plt.title('Nonstationarity T(A,B) (Torch)'); plt.xlabel('Iteration'); plt.grid(True, which='both', alpha=0.3)
        plt.tight_layout(); plt.savefig('alg41_vs_fista_convergence_Torch.png', dpi=150, bbox_inches='tight'); plt.show()


if __name__ == '__main__':
    main()



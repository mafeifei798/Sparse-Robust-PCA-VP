"""
Compare PCA vs SPCA and Randomized SPCA on the multiscale video dataset.

Outputs
- Prints timing and reconstruction error for each method
- Saves a bar chart figure: spca_video_compare.png
- Saves coefficients comparison (top-3 components): spca_video_coeffs.png
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA

from spca_variable_projection import spca_vp
from randomized_spca_vp import randomized_spca_vp
from spca_multiscale_video import generate_multiscale_video, align_components


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    fro = lambda Z: linalg.norm(Z, 'fro')
    return float(fro(X - X_hat) / (fro(X) + 1e-12))


def plot_multiscale_method(
    name: str,
    Phi: np.ndarray,
    A: np.ndarray,
    phi_true: np.ndarray,
    a_true: np.ndarray,
    t: np.ndarray,
    n_grid: int,
) -> None:
    """
    Reproduce the multiscale visualization (spatial modes + temporal activations)
    for a given method.
    """
    r = Phi.shape[1]
    norms = np.maximum(np.linalg.norm(Phi, axis=0), 1e-12)
    Phi_norm = Phi / norms
    A_scaled = A * norms

    modes = Phi_norm.reshape(n_grid, n_grid, r).transpose(2, 0, 1)
    perm, signs = align_components(modes, phi_true)
    modes = modes[perm] * signs[:, None, None]
    A_aligned = A_scaled[:, perm] * signs[None, :]

    A_plot = A_aligned.copy()
    for j in range(r):
        denom = float(A_plot[:, j].T @ A_plot[:, j]) + 1e-12
        if denom > 0:
            num = float(A_plot[:, j].T @ a_true[j])
            A_plot[:, j] *= num / denom

    plt.figure(figsize=(16, 9))
    for j in range(r):
        ax = plt.subplot(2, r, j + 1)
        im = ax.imshow(modes[j], cmap='magma')
        ax.set_title(f'{name} Φ{j+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax2 = plt.subplot(2, r, r + j + 1)
        ax2.plot(t, A_plot[:, j], label=f'{name} a{j+1}')
        ax2.plot(t, a_true[j], '--', label='True', alpha=0.8)
        ax2.set_xlim(t[0], t[-1])
        ax2.set_xlabel('Time')
        ax2.grid(True, alpha=0.3)
        if j == 0:
            ax2.set_ylabel('Activation')
        if j == 0:
            ax2.legend(fontsize=9)

    plt.tight_layout()
    safe_name = name.replace(' ', '_')
    plt.savefig(f'spca_video_{safe_name}_multiscale.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # 1) Data (multiscale video)
    X, phi_true, a_true, t = generate_multiscale_video()
    p, n = X.shape
    r = 3
    alpha = 10
    n_grid = int(np.sqrt(p))
    print(f"Video matrix: p={p}, n={n}, r={r}")

    timings = {}
    errors = {}
    coefs = {}
    spatial_modes = {}

    # 2) PCA baseline
    t0 = time.time()
    pca = PCA(n_components=r, svd_solver='full')
    scores = pca.fit_transform(X.T)  # (n, r)
    comps = pca.components_.T       # (p, r)
    X_hat_pca = pca.inverse_transform(scores).T
    timings['PCA'] = time.time() - t0
    errors['PCA'] = reconstruction_error(X, X_hat_pca)
    coefs['PCA'] = scores
    spatial_modes['PCA'] = comps

    # 3) SPCA
    t0 = time.time()
    A_spca, B_spca, obj_spca, crit_spca = spca_vp(
        X, r=r, alpha=alpha, max_iter=500, eps=1e-6, adaptive_alpha=False
    )
    timings['SPCA'] = time.time() - t0
    X_hat_spca = X @ B_spca @ A_spca.T
    errors['SPCA'] = reconstruction_error(X, X_hat_spca)
    coefs['SPCA'] = A_spca
    spatial_modes['SPCA'] = X @ B_spca

    # 4) Randomized SPCA (sketched)
    t0 = time.time()
    B_rnd, A_rnd, obj_rnd, rnd_timings = randomized_spca_vp(
        X, n_components=r, alpha=alpha, oversample=100, n_iter=4, max_iter=500, tol=1e-6, verbose=False, return_timings=True
    )
    timings['RND-SPCA'] = time.time() - t0
    # Print fine-grained timings for randomized SPCA
    try:
        print(f"RND-SPCA details | range_finder={rnd_timings['range_finder']:.3f}s, iter={rnd_timings['iter']:.3f}s, compress={rnd_timings['compress']:.3f}s")
        with open('spca_video_comparison_rnd_timing.txt', 'w') as f:
            f.write(f"range_finder: {rnd_timings['range_finder']:.6f}\n")
            f.write(f"compress: {rnd_timings['compress']:.6f}\n")
            f.write(f"iter: {rnd_timings['iter']:.6f}\n")
            f.write(f"total: {rnd_timings['range_finder'] + rnd_timings['compress'] + rnd_timings['iter']:.6f}\n")
    except Exception as e:
        print('Failed to write RND-SPCA timing file:', e)
    X_hat_rnd = X @ B_rnd @ A_rnd.T
    errors['RND-SPCA'] = reconstruction_error(X, X_hat_rnd)
    coefs['RND-SPCA'] = A_rnd
    spatial_modes['RND-SPCA'] = X @ B_rnd

    # 7) Report
    print("\nTiming (s):")
    for k, v in timings.items():
        print(f"  {k:15s}: {v:.3f}")
    print("\nReconstruction error:")
    for k, v in errors.items():
        print(f"  {k:15s}: {v:.4f}")

    # 8) Bars
    methods = list(timings.keys())
    tvals = [timings[m] for m in methods]
    # add a tiny floor to visualize zero-error bars
    evals = [max(errors[m], 1e-9) for m in methods]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    color_map = {
        'PCA': 'gray',
        'SPCA': 'steelblue',
        'RND-SPCA': 'green',
    }
    bars = ax[0].bar(methods, tvals, color=[color_map.get(m, 'C0') for m in methods])
    ax[0].set_title('Computation Time (s, log)')
    ax[0].set_ylabel('Seconds (log scale)')
    ax[0].set_yscale('log')
    for b, v in zip(bars, tvals):
        ax[0].text(b.get_x() + b.get_width()/2, v + 1e-3, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    ax[0].grid(True, axis='y', alpha=0.3, which='both')

    bars_e = ax[1].bar(methods, evals, color=[color_map.get(m, 'C0') for m in methods])
    ax[1].set_title('Reconstruction Error')
    ax[1].grid(True, axis='y', alpha=0.3)
    for b, v in zip(bars_e, evals):
        ax[1].text(b.get_x() + b.get_width()/2, v + 1e-9, f"{v:.2e}", ha='center', va='bottom', fontsize=8)
    plt.tight_layout(); plt.savefig('spca_video_compare.png', dpi=150, bbox_inches='tight'); plt.show()

    # 9) Multiscale visualizations per method
    names = ['PCA', 'SPCA', 'RND-SPCA']
    for name in names:
        Phi = spatial_modes[name]
        A_mat = coefs[name]
        plot_multiscale_method(name, Phi, A_mat, phi_true, a_true, t, n_grid)


if __name__ == '__main__':
    main()



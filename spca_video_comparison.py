"""Backward-compatible entry point for :mod:`experiments.video_spca`.

The old script trained different routines around a ``pixels x time`` matrix and
post-hoc fit temporal amplitudes to ground truth.  The canonical experiment now
uses ``time x pixels`` for every method and ground truth only for permutation,
sign, and recovery scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.video_spca import main as _canonical_main
from rspca.metrics import relative_reconstruction_error
from spca_multiscale_video import align_components, generate_multiscale_video


def reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Compatibility wrapper for relative Frobenius reconstruction error."""

    return relative_reconstruction_error(X, X_hat)


def plot_multiscale_method(
    name: str,
    Phi: np.ndarray,
    A: np.ndarray,
    phi_true: np.ndarray,
    a_true: np.ndarray,
    t: np.ndarray,
    n_grid: int,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Legacy visualisation with permutation/sign alignment but no scale fit."""

    import matplotlib.pyplot as plt

    spatial = np.asarray(Phi, dtype=float)
    temporal = np.asarray(A, dtype=float)
    rank = spatial.shape[1]
    modes = spatial.T.reshape(rank, n_grid, n_grid)
    permutation, signs = align_components(modes, phi_true)
    modes = modes[permutation] * signs[:, None, None]
    aligned_temporal = temporal[:, permutation] * signs[None, :]

    figure, axes = plt.subplots(2, rank, figsize=(4.5 * rank, 7.0))
    axes = np.asarray(axes).reshape(2, rank)
    for component in range(rank):
        image = axes[0, component].imshow(modes[component], cmap="magma")
        axes[0, component].set_title(f"{name} spatial component {component + 1}")
        axes[0, component].axis("off")
        figure.colorbar(image, ax=axes[0, component], fraction=0.046, pad=0.04)
        axes[1, component].plot(t, aligned_temporal[:, component], label="recovered")
        axes[1, component].plot(t, a_true[component], "--", label="truth")
        axes[1, component].set_xlabel("time")
        axes[1, component].grid(alpha=0.25)
        if component == 0:
            axes[1, component].legend()
    figure.tight_layout()
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(f"spca_video_{name.replace(' ', '_')}_multiscale.png")
    )
    figure.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return destination


def main(argv: Sequence[str] | None = None):
    """Delegate to the canonical paired benchmark CLI."""

    return _canonical_main(argv)


if __name__ == "__main__":
    main()


__all__ = [
    "align_components",
    "generate_multiscale_video",
    "main",
    "plot_multiscale_method",
    "reconstruction_error",
]

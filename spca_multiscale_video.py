"""Compatibility entry point for the audited synthetic-video experiment.

New code should import :mod:`experiments.video_spca`.  The historical
``generate_multiscale_video`` function is retained, but its corrected data
orientation is now ``(time samples, pixel features)``.  This is required for
the VP-SPCA L1 penalty on ``B`` to act on spatial loadings.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from experiments.video_spca import (
    VideoExperimentConfig,
    generate_video_dataset,
    main as _canonical_main,
)


def generate_multiscale_video(
    n_grid: int = 20,
    n_timesteps: int = 100,
    total_time: float = 30.0,
    *,
    noise_std: float = 0.03,
    corruption_fraction: float = 0.01,
    corruption_scale: float = 1.5,
    seed: int = 2025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the historical four arrays with corrected ``time x pixels`` X.

    Returns
    -------
    X
        Centred observed video with shape
        ``(n_timesteps, n_grid ** 2)``.  Rows are frames/samples and columns are
        spatial pixel/features.
    phi_true
        Ground-truth compact spatial modes with shape ``(3, n_grid, n_grid)``.
    a_true
        Ground-truth temporal scores with shape ``(3, n_timesteps)``.
    t
        Time coordinates with shape ``(n_timesteps,)``.
    """

    dataset = generate_video_dataset(
        VideoExperimentConfig(
            grid_size=n_grid,
            n_timesteps=n_timesteps,
            total_time=total_time,
            noise_std=noise_std,
            corruption_fraction=corruption_fraction,
            corruption_scale=corruption_scale,
            seed=seed,
        )
    )
    phi_true = dataset.spatial_components.T.reshape(3, n_grid, n_grid)
    return (
        dataset.observed.copy(),
        phi_true,
        dataset.temporal_scores.T.copy(),
        dataset.time.copy(),
    )


def generate_multiscale_video_classic(
    n_grid: int = 20,
    n_timesteps: int = 100,
    total_time: float = 30.0,
    **kwargs: float | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compatibility alias using the audited noisy/corrupted generator."""

    return generate_multiscale_video(
        n_grid=n_grid,
        n_timesteps=n_timesteps,
        total_time=total_time,
        **kwargs,
    )


def align_components(
    modes_rec: np.ndarray, modes_true: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return Hungarian permutation and signs; never fit component amplitudes."""

    recovered = np.asarray(modes_rec, dtype=float)
    truth = np.asarray(modes_true, dtype=float)
    if recovered.ndim < 2 or truth.ndim < 2:
        raise ValueError("mode arrays must include component and feature dimensions")
    if recovered.shape[0] != truth.shape[0]:
        raise ValueError("recovered and true mode counts must match")
    recovered_flat = recovered.reshape(recovered.shape[0], -1)
    truth_flat = truth.reshape(truth.shape[0], -1)
    if recovered_flat.shape[1] != truth_flat.shape[1]:
        raise ValueError("recovered and true modes must have equal feature counts")
    recovered_norm = np.linalg.norm(recovered_flat, axis=1)
    truth_norm = np.linalg.norm(truth_flat, axis=1)
    denominator = truth_norm[:, None] * recovered_norm[None, :]
    correlation = np.divide(
        truth_flat @ recovered_flat.T,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > np.finfo(float).eps,
    )
    truth_indices, recovered_indices = linear_sum_assignment(-np.abs(correlation))
    signs = np.where(
        correlation[truth_indices, recovered_indices] < 0.0, -1.0, 1.0
    )
    order = np.argsort(truth_indices)
    return recovered_indices[order], signs[order]


def main(argv: Sequence[str] | None = None):
    """Delegate to the canonical audited CLI."""

    return _canonical_main(argv)


if __name__ == "__main__":
    main()


__all__ = [
    "align_components",
    "generate_multiscale_video",
    "generate_multiscale_video_classic",
    "main",
]

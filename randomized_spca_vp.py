"""Backward-compatible imports for randomized and certified VP-SPCA."""

from __future__ import annotations

import numpy as np

from rspca.randomized import (
    RandomizedVPSPCAResult,
    RandomizedVPSPCAWorkspace,
    SketchRefinementCertificate,
    certified_randomized_vp_spca,
    certified_randomized_vp_spca_from_workspace,
    fit_randomized_vp_spca,
    fit_randomized_vp_spca_from_workspace,
    prepare_randomized_vp_spca_workspace,
    randomized_range_finder,
    randomized_spca_vp,
    sketch_residual_diagnostic,
)


def demo_multiscale_video(seed: int = 42) -> RandomizedVPSPCAResult:
    """Run a noisy sparse-loading benchmark under the historical demo name."""

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(120, 80))
    X -= X.mean(axis=0, keepdims=True)
    result = certified_randomized_vp_spca(
        X,
        3,
        alpha=0.2,
        oversample=12,
        n_power_iter=2,
        max_iter=150,
        full_refine_iters=20,
        random_state=seed,
        verbose=True,
    )
    print(
        f"status={result.status}; sketch residual="
        f"{result.sketch_residual_relative:.3e}; "
        f"full stationarity={result.full_stationarity:.3e}"
    )
    return result


__all__ = [
    "RandomizedVPSPCAResult",
    "RandomizedVPSPCAWorkspace",
    "SketchRefinementCertificate",
    "certified_randomized_vp_spca",
    "certified_randomized_vp_spca_from_workspace",
    "demo_multiscale_video",
    "fit_randomized_vp_spca",
    "fit_randomized_vp_spca_from_workspace",
    "prepare_randomized_vp_spca_workspace",
    "randomized_range_finder",
    "randomized_spca_vp",
    "sketch_residual_diagnostic",
]


if __name__ == "__main__":
    demo_multiscale_video()

"""Backward-compatible entry point for matrix-free VP-SPCA.

The maintained implementation lives in :mod:`rspca.vp`.  This module keeps the
historical import path while ensuring notebooks, scripts, and package users all
execute the same solver.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from rspca.vp import (
    VPSPCAResult,
    fit_vp_spca,
    proximal_gradient_mapping,
    procrustes_factor,
    smooth_value,
    soft_threshold,
    spca_vp,
    vp_gradient,
    vp_objective,
)


def update_A_procrustes(X: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Historical name for :func:`rspca.vp.procrustes_factor`."""

    return procrustes_factor(X, B)


def compute_nonstationarity(
    X: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    gamma: float,
    alpha: float,
) -> float:
    """Return the Frobenius norm of the proximal-gradient mapping."""

    mapping = proximal_gradient_mapping(
        X,
        A,
        B,
        alpha=alpha,
        step_size=gamma,
    )
    return float(linalg.norm(mapping, ord="fro"))


def demo() -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
    """Run a deterministic smoke example without generating unverifiable plots."""

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 30))
    X -= X.mean(axis=0, keepdims=True)
    result = fit_vp_spca(X, 2, alpha=0.2, max_iter=200, verbose=True)
    print(
        f"status={result.status}; iterations={result.n_iter}; "
        f"loading sparsity={np.mean(result.B == 0.0):.3f}"
    )
    return result.A, result.B, result.objectives, result.stationarity


__all__ = [
    "VPSPCAResult",
    "compute_nonstationarity",
    "demo",
    "fit_vp_spca",
    "proximal_gradient_mapping",
    "procrustes_factor",
    "smooth_value",
    "soft_threshold",
    "spca_vp",
    "update_A_procrustes",
    "vp_gradient",
    "vp_objective",
]


if __name__ == "__main__":
    demo()

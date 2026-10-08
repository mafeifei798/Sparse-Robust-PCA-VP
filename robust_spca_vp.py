"""Backward-compatible imports for robust variable-projection SPCA."""

from rspca.robust import (
    RobustVPSPCAResult,
    fit_robust_vp_spca,
    huber_loss,
    mad_scale,
    robust_objective,
    robust_spca_vp,
    soft_threshold,
    update_A_procrustes_with_S,
)

__all__ = [
    "RobustVPSPCAResult",
    "fit_robust_vp_spca",
    "huber_loss",
    "mad_scale",
    "robust_objective",
    "robust_spca_vp",
    "soft_threshold",
    "update_A_procrustes_with_S",
]

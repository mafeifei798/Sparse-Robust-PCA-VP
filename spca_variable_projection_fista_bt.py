"""Backward-compatible imports for accelerated value-function VP-SPCA."""

from rspca.accelerated import (
    AcceleratedVPSPCAResult,
    fit_accelerated_vp_spca,
    spca_algorithm_4_1_fista_bt,
)
from rspca.vp import procrustes_factor as update_A_procrustes
from rspca.vp import soft_threshold

__all__ = [
    "AcceleratedVPSPCAResult",
    "fit_accelerated_vp_spca",
    "soft_threshold",
    "spca_algorithm_4_1_fista_bt",
    "update_A_procrustes",
]

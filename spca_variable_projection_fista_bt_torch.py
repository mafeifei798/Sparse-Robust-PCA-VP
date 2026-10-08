"""Backward-compatible imports for the optional PyTorch VP-SPCA backend.

New code should import from :mod:`rspca.torch_backend` after installing the
``torch`` extra: ``python -m pip install -e '.[torch]'``.
"""

import warnings

import torch

from rspca.torch_backend import (
    TorchVPSPCAResult,
    fit_torch_accelerated_vp_spca,
    soft_threshold_torch,
    spca_vp_fista_bt_torch,
)


def update_A_procrustes_torch(
    G: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Return the historical Gram-matrix update ``polar(G @ B)``.

    This top-level compatibility helper has always accepted ``G = X.T @ X``.
    The canonical :func:`rspca.torch_backend.update_A_procrustes_torch` instead
    accepts the original data matrix ``X`` so the maintained solver can remain
    matrix-free. Keeping the contracts separate prevents an old caller from
    silently evaluating ``polar(G.T @ G @ B)``.

    New code should pass ``X`` to the canonical helper. This legacy Gram-based
    entry point remains available for import compatibility only.
    """

    warnings.warn(
        "spca_variable_projection_fista_bt_torch.update_A_procrustes_torch "
        "uses the deprecated Gram-matrix contract; new code should call "
        "rspca.torch_backend.update_A_procrustes_torch(X, B)",
        DeprecationWarning,
        stacklevel=2,
    )
    cross = G @ B
    try:
        left, _, right_t = torch.linalg.svd(cross, full_matrices=False)
        return left @ right_t
    except (NotImplementedError, RuntimeError) as error:
        # Preserve the historical MPS fallback without changing the public
        # Gram-matrix semantics. Other devices should surface the real error.
        if cross.device.type != "mps":
            raise error
        left, _, right_t = torch.linalg.svd(cross.detach().cpu(), full_matrices=False)
        return (left @ right_t).to(device=cross.device, dtype=cross.dtype)

__all__ = [
    "TorchVPSPCAResult",
    "fit_torch_accelerated_vp_spca",
    "soft_threshold_torch",
    "spca_vp_fista_bt_torch",
    "update_A_procrustes_torch",
]

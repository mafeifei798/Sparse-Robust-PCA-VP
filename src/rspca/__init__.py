"""Sparse and robust PCA algorithms with variable projection."""

from .accelerated import (
    AcceleratedVPSPCAResult,
    fit_accelerated_vp_spca,
    spca_algorithm_4_1_fista_bt,
)
from .datasets import (
    SyntheticSparseLowRank,
    array_sha256,
    canonical_sha256,
    generate_sparse_low_rank,
    make_seed_manifest,
    runtime_versions,
    verify_seed_manifest,
)
from .metrics import (
    component_correlation,
    explained_variance_ratio,
    outlier_detection_metrics,
    principal_angles,
    relative_reconstruction_error,
    sparsity_ratio,
    subspace_error,
    support_precision_recall_f1,
)
from .randomized import (
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
)
from .robust import (
    RobustVPSPCAResult,
    fit_robust_vp_spca,
    robust_objective,
    robust_spca_vp,
)
from .vp import VPSPCAResult, fit_vp_spca, spca_vp, vp_objective

__version__ = "0.2.0"

__all__ = [
    "AcceleratedVPSPCAResult",
    "RandomizedVPSPCAResult",
    "RandomizedVPSPCAWorkspace",
    "RobustVPSPCAResult",
    "SketchRefinementCertificate",
    "SyntheticSparseLowRank",
    "VPSPCAResult",
    "array_sha256",
    "canonical_sha256",
    "certified_randomized_vp_spca",
    "certified_randomized_vp_spca_from_workspace",
    "component_correlation",
    "explained_variance_ratio",
    "fit_accelerated_vp_spca",
    "fit_randomized_vp_spca",
    "fit_randomized_vp_spca_from_workspace",
    "fit_robust_vp_spca",
    "fit_vp_spca",
    "generate_sparse_low_rank",
    "make_seed_manifest",
    "outlier_detection_metrics",
    "prepare_randomized_vp_spca_workspace",
    "principal_angles",
    "randomized_range_finder",
    "randomized_spca_vp",
    "relative_reconstruction_error",
    "robust_objective",
    "robust_spca_vp",
    "runtime_versions",
    "sparsity_ratio",
    "spca_algorithm_4_1_fista_bt",
    "spca_vp",
    "subspace_error",
    "support_precision_recall_f1",
    "verify_seed_manifest",
    "vp_objective",
]

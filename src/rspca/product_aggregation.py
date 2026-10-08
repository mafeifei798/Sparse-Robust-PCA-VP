"""Gauge-invariant means of low-rank classifier products.

LoRA-style factors are not identifiable: for any invertible matrix ``G``,
``(G @ A, B @ inv(G))`` represents the same product ``B @ A``.  Averaging
``A`` and ``B`` separately can therefore change solely because clients used
different factor gauges.  The routines here aggregate represented products
and return balanced rank-constrained factors for the next federated round.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
FactorPair = tuple[ArrayLike, ArrayLike]


@dataclass(frozen=True)
class ProductRankDiagnostics:
    """Loss and numerical checks for a best-rank product projection."""

    rank_tail: float
    per_class_rank_tails: tuple[float, ...]
    worst_class_tail: float
    rank_before_projection: int
    rank_after_projection: int
    output_factorization_residual: float


@dataclass(frozen=True)
class ProductMeanResult:
    """Rank-constrained mean of client product states.

    ``unprojected_product`` is the exact product-space target before enforcing
    the communicated rank. ``product == B @ A`` is its best rank-constrained
    approximation.
    """

    A: FloatArray
    B: FloatArray
    product: FloatArray
    unprojected_product: FloatArray
    rank_diagnostics: ProductRankDiagnostics

    @property
    def rank_tail(self) -> float:
        """Backward-compatible access to the global projection tail."""

        return self.rank_diagnostics.rank_tail

    @property
    def per_class_rank_tails(self) -> tuple[float, ...]:
        return self.rank_diagnostics.per_class_rank_tails

    @property
    def worst_class_tail(self) -> float:
        return self.rank_diagnostics.worst_class_tail


def factors_to_product(A: ArrayLike, B: ArrayLike) -> FloatArray:
    """Return the represented low-rank product ``B @ A``.

    ``A`` must have shape ``(rank, features)`` and ``B`` shape
    ``(classes, rank)``.
    """

    a = _as_nonempty_finite_matrix(A, name="A")
    b = _as_nonempty_finite_matrix(B, name="B")
    if b.shape[1] != a.shape[0]:
        raise ValueError("incompatible factors: B.shape[1] must equal A.shape[0]")
    product = b @ a
    if not np.isfinite(product).all():
        raise ValueError("B @ A produced non-finite values")
    return np.asarray(product, dtype=np.float64)


def balanced_factorization(Z: ArrayLike, rank: int) -> tuple[FloatArray, FloatArray]:
    """Return balanced factors of the best rank-``rank`` approximation to ``Z``.

    If ``Z = U diag(s) V.T``, the returned factors are
    ``B = U sqrt(diag(s))`` and ``A = sqrt(diag(s)) V.T``.  Thus both factors
    carry the same singular scale and ``B @ A`` is the truncated-SVD optimum.
    """

    matrix = _as_nonempty_finite_matrix(Z, name="Z")
    target_rank = _validate_rank(rank, matrix.shape)
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    roots = np.sqrt(np.maximum(singular_values[:target_rank], 0.0))
    b = left[:, :target_rank] * roots
    a = roots[:, None] * right_t[:target_rank, :]
    return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)


def exact_product_mean(
    local_factors: Sequence[FactorPair],
    weights: Sequence[float] | ArrayLike,
    rank: int,
    current_product: ArrayLike | None = None,
) -> ProductMeanResult:
    """Aggregate post-local factor states by their represented products.

    Each item is ``(A_i, B_i)``.  With a supplied server product ``P_t``, the
    target is evaluated as ``P_t + sum_i w_i (B_i A_i - P_t)``.  Normalized
    weights make this algebraically identical to the exact weighted product
    mean while retaining explicit federated-delta semantics.
    """

    _, products = _normalise_local_factors(local_factors)
    normalized_weights = _normalised_weights(weights, len(products), name="weights")
    output_rank = _validate_rank(rank, products[0].shape)

    if current_product is None:
        target = np.tensordot(normalized_weights, np.stack(products), axes=1)
    else:
        current = _as_nonempty_finite_matrix(current_product, name="current_product")
        if current.shape != products[0].shape:
            raise ValueError(
                "current_product must have the same shape as each client product"
            )
        deltas = np.stack([product - current for product in products])
        target = current + np.tensordot(normalized_weights, deltas, axes=1)
    return _product_mean_result(target, output_rank)


def _product_mean_result(target: ArrayLike, rank: int) -> ProductMeanResult:
    unprojected = _as_nonempty_finite_matrix(target, name="target")
    a, b = balanced_factorization(unprojected, rank)
    product = factors_to_product(a, b)
    return ProductMeanResult(
        A=a,
        B=b,
        product=product,
        unprojected_product=unprojected.copy(),
        rank_diagnostics=_rank_projection_diagnostics(unprojected, product, rank),
    )


def _rank_projection_diagnostics(
    unprojected: FloatArray, projected: FloatArray, rank: int
) -> ProductRankDiagnostics:
    unprojected_norm = float(np.linalg.norm(unprojected, ord="fro"))
    discarded = unprojected - projected
    rank_tail = float(
        np.linalg.norm(discarded, ord="fro")
        / max(unprojected_norm, np.finfo(float).tiny)
    )
    numerical_floor = np.finfo(float).eps * max(unprojected_norm, np.finfo(float).tiny)
    per_class_tails = tuple(
        float(np.linalg.norm(discarded[class_index]))
        / max(
            float(np.linalg.norm(unprojected[class_index])),
            numerical_floor,
        )
        for class_index in range(unprojected.shape[0])
    )
    left, singular_values, right_t = np.linalg.svd(unprojected, full_matrices=False)
    ideal_projection = (left[:, :rank] * singular_values[:rank]) @ right_t[:rank, :]
    factorization_residual = float(
        np.linalg.norm(projected - ideal_projection, ord="fro")
        / max(float(np.linalg.norm(ideal_projection, ord="fro")), np.finfo(float).tiny)
    )
    return ProductRankDiagnostics(
        rank_tail=rank_tail,
        per_class_rank_tails=per_class_tails,
        worst_class_tail=max(per_class_tails, default=0.0),
        rank_before_projection=_matrix_rank(unprojected),
        rank_after_projection=_matrix_rank(projected),
        output_factorization_residual=factorization_residual,
    )


def _normalise_local_factors(
    local_factors: Sequence[FactorPair],
) -> tuple[list[tuple[FloatArray, FloatArray]], list[FloatArray]]:
    if isinstance(local_factors, (str, bytes)) or not isinstance(
        local_factors, Sequence
    ):
        raise TypeError("local_factors must be a non-empty sequence of (A, B) pairs")
    if len(local_factors) == 0:
        raise ValueError("local_factors must not be empty")
    factors = [
        _validate_factor_pair(pair, name=f"local_factors[{index}]")
        for index, pair in enumerate(local_factors)
    ]
    reference_shapes = (factors[0][0].shape, factors[0][1].shape)
    for index, (a, b) in enumerate(factors[1:], start=1):
        if (a.shape, b.shape) != reference_shapes:
            raise ValueError(
                f"local_factors[{index}] must match the first A and B shapes"
            )
    products = [factors_to_product(a, b) for a, b in factors]
    return factors, products


def _validate_factor_pair(
    pair: FactorPair, *, name: str
) -> tuple[FloatArray, FloatArray]:
    if (
        isinstance(pair, (str, bytes))
        or not isinstance(pair, Sequence)
        or len(pair) != 2
    ):
        raise TypeError(f"{name} must be a two-element (A, B) sequence")
    a = _as_nonempty_finite_matrix(pair[0], name=f"{name}[0]")
    b = _as_nonempty_finite_matrix(pair[1], name=f"{name}[1]")
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"{name} has incompatible A and B inner dimensions")
    return a, b


def _normalised_weights(
    weights: Sequence[float] | ArrayLike, n_items: int, *, name: str
) -> FloatArray:
    array = _as_weight_vector(weights, n_items, name=name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    total = float(np.sum(array))
    if total <= 0.0:
        raise ValueError(f"{name} must contain at least one positive value")
    return array / total


def _as_weight_vector(
    values: Sequence[float] | ArrayLike, n_items: int, *, name: str
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != n_items:
        raise ValueError(f"{name} must be a length-{n_items} vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _matrix_rank(matrix: FloatArray) -> int:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    return int(np.count_nonzero(singular_values > tolerance))


def _validate_rank(rank: int, shape: tuple[int, int]) -> int:
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    value = int(rank)
    if not 1 <= value <= min(shape):
        raise ValueError("rank must be between 1 and min(matrix.shape)")
    return value


def _as_finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_nonempty_finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = _as_finite_matrix(values, name=name)
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    return array


__all__ = [
    "ProductMeanResult",
    "ProductRankDiagnostics",
    "balanced_factorization",
    "exact_product_mean",
    "factors_to_product",
]

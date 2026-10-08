"""Deterministic aggregation utilities for federated LoRA experiments.

The FedRPCA implementation follows the principal-component-pursuit (PCP)
formulation used by Jhunjhunwala et al. (2025)::

    min_{L, S} ||L||_* + lambda ||S||_1    subject to M = L + S.

For every LoRA factor *block*, client updates are vectorised as the columns of
``M``.  PCP is solved with an inexact augmented-Lagrangian/ADMM iteration: a
singular-value-thresholding update for ``L``, elementwise soft thresholding for
``S``, and a dual ascent step.  Importantly, A and B factors are separate
blocks.  Biases and other trainable parameters are never concatenated into an
adapter block; they use the same declared uniform or sample-weighted mean as
the adapter aggregation.

The module uses NumPy deliberately so the aggregation rules can be unit tested
without a particular deep-learning framework.  Callers can convert tensors to
and from NumPy at the server boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg as scipy_linalg
from scipy import special as scipy_special


FloatArray = NDArray[np.float64]
ClientUpdate = Mapping[str, ArrayLike]


@dataclass(frozen=True)
class RPCAResult:
    """Result of a principal-component-pursuit decomposition."""

    low_rank: FloatArray
    sparse: FloatArray
    residual: float
    iterations: int
    converged: bool
    rank: int
    sparsity: float


@dataclass(frozen=True)
class BlockDiagnostics:
    """Serializable diagnostics for one independently decomposed LoRA block.

    ``sparsity`` is the fraction of entries in ``S`` that are numerically zero.
    ``observed_cosine``, ``low_rank_cosine``, and ``sparse_cosine`` are mean
    pairwise client-column cosine similarities for ``M``, ``L``, and ``S``;
    ``low_rank_cosine - sparse_cosine`` is ``cosine_gap``.  ``beta_base`` is the
    fixed/adaptive value before reliability gating, and ``beta`` is the value
    actually applied to the sparse aggregate. ``beta_policy`` names both the
    weighting rule and any explicit safety clip so an extension cannot be
    serialized as the paper schedule.
    """

    residual: float
    rank: int
    sparsity: float
    observed_cosine: float
    low_rank_cosine: float
    sparse_cosine: float
    cosine_gap: float
    energy_ratio: float
    beta_policy: str
    beta_base: float
    gate: float
    beta: float
    iterations: int
    converged: bool
    primary_max_iter: int | None = None
    primary_residual: float | None = None
    primary_converged: bool | None = None
    fallback_triggered: bool = False
    final_max_iter: int | None = None


@dataclass(frozen=True, kw_only=True)
class CancellationAwareBlockDiagnostics(BlockDiagnostics):
    """Additional diagnostics for cancellation-aware RMS energy scaling.

    ``rms_energy_ratio`` is
    ``sqrt(sum_i w_i ||S_i||^2) / sqrt(sum_i w_i ||M_i||^2)`` and
    ``raw_rms_beta`` is its unbounded inverse.  ``observed_coherence`` and
    ``sparse_coherence`` compare the norm of each weighted column mean with
    its weighted RMS column norm.  Their ratio therefore exposes exactly how
    much more strongly the sparse columns cancel than the observed columns.
    The two zero guards distinguish an exactly zero ``S`` from a nonzero ``S``
    supported only on clients with zero aggregation weight.

    The separate subclass is intentional: canonical paper-FedRPCA continues
    to return :class:`BlockDiagnostics` with an unchanged serialized schema.
    """

    rms_energy_ratio: float | None
    raw_rms_beta: float | None
    observed_coherence: float
    sparse_coherence: float
    coherence_ratio: float | None
    sparse_exact_zero_guard: bool
    sparse_weighted_zero_guard: bool


@dataclass(frozen=True)
class AggregationResult:
    """Aggregated update and per-adapter-block diagnostics."""

    update: dict[str, FloatArray]
    diagnostics: dict[str, BlockDiagnostics]


@dataclass(frozen=True)
class RobustAggregationResult:
    """Aggregated values plus a JSON-serialisable robust-reducer audit trail.

    Unlike :class:`AggregationResult`, whose diagnostics are specifically PCP
    block diagnostics, this result is used by deterministic non-RPCA server
    baselines.  The input mapping may contain post-local states or deltas; the
    calling server is responsible for declaring that semantic scope.
    """

    update: dict[str, FloatArray]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class PairedRunPlan:
    """Immutable sources of randomness shared by every compared method.

    A plan fixes the non-IID client partition, model initialisation seed, and
    each client/round local-shuffle seed.  Reusing the same plan makes method
    comparisons paired rather than allowing each method to consume a different
    random stream.
    """

    seed: int
    client_indices: tuple[tuple[int, ...], ...]
    initialization_seed: int
    local_shuffle_seeds: tuple[tuple[int, ...], ...]
    alpha: float

    @property
    def n_clients(self) -> int:
        return len(self.client_indices)

    @property
    def rounds(self) -> int:
        return len(self.local_shuffle_seeds)

    def client_permutation(
        self, round_index: int, client_index: int
    ) -> NDArray[np.int64]:
        """Return the fixed local sample order for a client and round."""

        if not 0 <= round_index < self.rounds:
            raise IndexError("round_index is outside the paired plan")
        if not 0 <= client_index < self.n_clients:
            raise IndexError("client_index is outside the paired plan")
        indices = np.asarray(self.client_indices[client_index], dtype=np.int64)
        seed = self.local_shuffle_seeds[round_index][client_index]
        return np.random.default_rng(seed).permutation(indices)

    def manifest(self) -> dict[str, object]:
        """Return a JSON-serialisable experiment manifest."""

        return {
            "seed": self.seed,
            "alpha": self.alpha,
            "initialization_seed": self.initialization_seed,
            "client_indices": [list(indices) for indices in self.client_indices],
            "local_shuffle_seeds": [
                list(round_seeds) for round_seeds in self.local_shuffle_seeds
            ],
        }

    @property
    def digest(self) -> str:
        """SHA-256 digest suitable for attaching to result rows."""

        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class HsuBalancedRunPlan(PairedRunPlan):
    """Paired plan with auditable Hsu-partition provenance.

    The four matrix/vector fields use the deterministic ``np.unique`` class
    order recorded in ``class_labels``.  They retain both the sampled
    Dirichlet targets and their capacity-conditioned finite-data realization,
    so downstream runners never need to reconstruct or re-sample hidden RNG
    state merely to audit a partition.
    """

    class_labels: tuple[object, ...]
    population_distribution: tuple[float, ...]
    dirichlet_concentration: tuple[float, ...]
    sampled_client_label_distributions: tuple[tuple[float, ...], ...]
    realized_client_class_counts: tuple[tuple[int, ...], ...]

    @property
    def realized_client_label_distributions(
        self,
    ) -> tuple[tuple[float, ...], ...]:
        """Return empirical per-client label distributions."""

        counts = np.asarray(self.realized_client_class_counts, dtype=np.float64)
        sizes = np.sum(counts, axis=1, keepdims=True)
        distributions = counts / sizes
        return tuple(tuple(float(value) for value in row) for row in distributions)

    @property
    def client_total_variation(self) -> tuple[float, ...]:
        """Return TV distance between sampled and realized client priors."""

        sampled = np.asarray(self.sampled_client_label_distributions, dtype=np.float64)
        realized = np.asarray(
            self.realized_client_label_distributions, dtype=np.float64
        )
        distances = 0.5 * np.sum(np.abs(sampled - realized), axis=1)
        return tuple(float(value) for value in distances)

    @property
    def client_jensen_shannon_divergence(self) -> tuple[float, ...]:
        """Return natural-log Jensen-Shannon divergence for every client."""

        sampled = np.asarray(self.sampled_client_label_distributions, dtype=np.float64)
        realized = np.asarray(
            self.realized_client_label_distributions, dtype=np.float64
        )
        midpoint = 0.5 * (sampled + realized)
        sampled_terms = np.zeros_like(sampled)
        realized_terms = np.zeros_like(realized)
        sampled_mask = sampled > 0.0
        realized_mask = realized > 0.0
        sampled_terms[sampled_mask] = sampled[sampled_mask] * np.log(
            sampled[sampled_mask] / midpoint[sampled_mask]
        )
        realized_terms[realized_mask] = realized[realized_mask] * np.log(
            realized[realized_mask] / midpoint[realized_mask]
        )
        divergences = 0.5 * (
            np.sum(sampled_terms, axis=1) + np.sum(realized_terms, axis=1)
        )
        return tuple(float(value) for value in divergences)

    def manifest(self) -> dict[str, object]:
        """Return the paired RNG manifest plus partition diagnostics."""

        manifest = super().manifest()
        sampled = [list(row) for row in self.sampled_client_label_distributions]
        counts = [list(row) for row in self.realized_client_class_counts]
        realized = [list(row) for row in self.realized_client_label_distributions]
        total_variation = list(self.client_total_variation)
        jensen_shannon = list(self.client_jensen_shannon_divergence)
        manifest.update(
            {
                "partition_scheme": "hsu_balanced_per_client_dirichlet",
                "partition_scheme_version": 1,
                "dirichlet_parameterization": (
                    "alpha_times_empirical_population_prior"
                ),
                "finite_dataset_conditioning": (
                    "sinkhorn_transport_exact_balanced_without_replacement"
                ),
                "class_labels": list(self.class_labels),
                "population_distribution": list(self.population_distribution),
                "population_distribution_sha256": _json_sha256(
                    list(self.population_distribution)
                ),
                "dirichlet_concentration": list(self.dirichlet_concentration),
                "dirichlet_concentration_sha256": _json_sha256(
                    list(self.dirichlet_concentration)
                ),
                # ``alpha`` is the declared total concentration by
                # construction.  Re-summing a vector whose components are
                # near float64 max can overflow even though the original
                # finite alpha is valid.
                "dirichlet_total_concentration": float(self.alpha),
                "sampled_client_label_distributions": sampled,
                "sampled_client_label_distributions_sha256": _json_sha256(sampled),
                "realized_client_class_counts": counts,
                "realized_client_class_counts_sha256": _json_sha256(counts),
                "realized_client_label_distributions": realized,
                "client_total_variation": total_variation,
                "mean_client_total_variation": float(np.mean(total_variation)),
                "max_client_total_variation": float(np.max(total_variation)),
                "client_jensen_shannon_divergence": jensen_shannon,
                "mean_client_jensen_shannon_divergence": float(np.mean(jensen_shannon)),
                "max_client_jensen_shannon_divergence": float(np.max(jensen_shannon)),
            }
        )
        return manifest


def singular_value_threshold(matrix: ArrayLike, threshold: float) -> FloatArray:
    """Return the proximal operator of ``threshold * ||matrix||_*``."""

    x = _as_finite_float_array(matrix, name="matrix", ndim=2)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative number")
    if x.size == 0:
        return x.copy()
    u, singular_values, vt = _svd_with_sanity(x)
    shrunk = np.maximum(singular_values - threshold, 0.0)
    # Multiplying columns of U avoids constructing a dense diagonal matrix.
    result = (u * shrunk) @ vt
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(
            "singular-value thresholding produced non-finite values"
        )
    input_norm = float(np.linalg.norm(x, ord="fro"))
    result_norm = float(np.linalg.norm(result, ord="fro"))
    if not math.isfinite(result_norm) or result_norm > input_norm * (1.0 + 1e-8):
        raise FloatingPointError(
            "singular-value thresholding violated proximal norm contraction"
        )
    return result


def principal_component_pursuit_admm(
    matrix: ArrayLike,
    *,
    lam: float | None = None,
    mu: float | None = None,
    tol: float = 1e-7,
    max_iter: int = 1_000,
    rho: float | None = None,
) -> RPCAResult:
    """Solve PCP with SVT, soft thresholding, and dual ascent.

    The iteration is the inexact augmented-Lagrangian form commonly described
    as ADMM for PCP::

        rho = 1 / mu
        L+ = SVT_rho(M - S + rho Y)
        S+ = soft_{rho lambda}(M - L+ + rho Y)
        R  = M - L+ - S+
        Y+ = Y + mu R

    This deliberately matches Algorithm 2 in the FedRPCA paper, including its
    defaults ``mu = M.numel() / (4 * M.abs().sum())`` and
    ``lambda = 1 / sqrt(max(M.shape))``.  Convergence is tested with the
    scale-free primal residual ``||R||_F / ||M||_F``.

    The iteration is evaluated after dividing ``M`` by an exact power-of-two
    scale.  PCP is positively homogeneous, and the corresponding ADMM
    parameters transform as ``mu' = scale * mu`` and
    ``rho' = rho / scale``.  Thus the normalized iterates, after rescaling
    ``L`` and ``S``, are mathematically identical to the paper iteration while
    avoiding overflow in norms and underflow that would misclassify a tiny,
    non-zero matrix as zero.
    """

    observed = _as_finite_float_array(matrix, name="matrix", ndim=2)
    n_rows, n_cols = observed.shape
    if n_rows == 0 or n_cols == 0:
        raise ValueError("matrix must have at least one row and one column")
    if lam is None:
        lam = 1.0 / math.sqrt(max(n_rows, n_cols))
    if not math.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be a finite positive number")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a finite positive number")
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)):
        raise TypeError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    magnitude = float(np.max(np.abs(observed)))
    if magnitude == 0.0:
        zeros = np.zeros_like(observed)
        return RPCAResult(
            low_rank=zeros.copy(),
            sparse=zeros,
            residual=0.0,
            iterations=0,
            converged=True,
            rank=0,
            sparsity=1.0,
        )

    scale = _power_of_two_scale(observed)
    working = observed / scale
    frobenius = float(np.linalg.norm(working, ord="fro"))
    if not math.isfinite(frobenius) or frobenius <= 0.0:
        raise FloatingPointError("normalized PCP matrix has an invalid Frobenius norm")

    if mu is None:
        working_mu = working.size / (4.0 * float(np.sum(np.abs(working))))
    else:
        if not math.isfinite(mu) or mu <= 0:
            raise ValueError("mu must be a finite positive number")
        working_mu = mu * scale
    if not math.isfinite(working_mu) or working_mu <= 0:
        raise ValueError(
            "mu is not representable after scale-equivalent PCP normalization"
        )
    if rho is None:
        working_rho = 1.0 / working_mu
    else:
        if not math.isfinite(rho) or rho <= 0:
            raise ValueError("rho must be a finite positive number")
        working_rho = rho / scale
    if not math.isfinite(working_rho) or working_rho <= 0:
        raise ValueError(
            "rho is not representable after scale-equivalent PCP normalization"
        )
    if not math.isclose(working_rho * working_mu, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("FedRPCA Algorithm 2 requires rho = 1 / mu")

    dual = np.zeros_like(working)
    low_rank = np.zeros_like(working)
    sparse = np.zeros_like(working)
    residual = math.inf
    converged = False

    for iteration in range(1, int(max_iter) + 1):
        low_rank = singular_value_threshold(
            working - sparse + working_rho * dual, working_rho
        )
        sparse = _soft_threshold(
            working - low_rank + working_rho * dual, working_rho * lam
        )
        primal = working - low_rank - sparse
        residual = float(np.linalg.norm(primal, ord="fro") / frobenius)
        dual = dual + working_mu * primal
        if not math.isfinite(residual) or not np.all(np.isfinite(dual)):
            raise FloatingPointError(
                f"PCP produced non-finite iterates at iteration {iteration}"
            )
        dual_limit = lam + 64.0 * np.finfo(np.float64).eps * max(1.0, lam)
        if float(np.max(np.abs(dual))) > dual_limit:
            raise FloatingPointError(
                f"PCP dual feasibility was violated at iteration {iteration}"
            )
        if residual <= tol:
            converged = True
            break

    if converged:
        # L=0,S=M is feasible, so the optimal PCP objective cannot exceed
        # lambda*||M||_1.  The expression below is a lower bound on the current
        # objective because ||L||_F <= ||L||_*.  It rejects catastrophic L/S
        # cancellation that can otherwise report a tiny primal residual when
        # a broken SVD/BLAS backend returns enormous but finite factors.
        feasible_upper = lam * _stable_l1_norm(working)
        objective_lower = _stable_frobenius_norm(low_rank) + lam * _stable_l1_norm(
            sparse
        )
        objective_slack = max(1e-6, 10.0 * tol)
        if not math.isfinite(objective_lower) or objective_lower > feasible_upper * (
            1.0 + objective_slack
        ):
            raise FloatingPointError(
                "PCP rejected false convergence from explosive component "
                "cancellation"
            )

    _, singular_values, _ = _svd_with_sanity(low_rank)
    rank_tolerance = (
        max(low_rank.shape)
        * np.finfo(np.float64).eps
        * (float(singular_values[0]) if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    zero_tolerance = 10.0 * np.finfo(np.float64).eps * float(np.max(np.abs(working)))
    sparsity = float(np.mean(np.abs(sparse) <= zero_tolerance))
    try:
        with np.errstate(over="raise", invalid="raise"):
            output_low_rank = low_rank * scale
            output_sparse = sparse * scale
    except FloatingPointError as error:
        raise FloatingPointError(
            "PCP decomposition is not representable at the input scale"
        ) from error
    if not np.all(np.isfinite(output_low_rank)) or not np.all(
        np.isfinite(output_sparse)
    ):
        raise FloatingPointError(
            "PCP decomposition is not representable at the input scale"
        )
    return RPCAResult(
        low_rank=output_low_rank,
        sparse=output_sparse,
        residual=residual,
        iterations=iteration,
        converged=converged,
        rank=rank,
        sparsity=sparsity,
    )


def fedavg_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    sample_counts: Sequence[float] | None = None,
) -> dict[str, FloatArray]:
    """Return the weighted mean of client parameter dictionaries.

    ``sample_counts`` supplies the usual local-sample weights used by FedAvg;
    the values are normalised internally.  Omitting ``sample_counts`` selects
    an explicit uniform-client mean, which is useful for methods and ablations
    whose published aggregation rule weights task vectors equally.
    """

    updates, keys = _normalise_client_updates(client_updates)
    weights = _normalised_weights(sample_counts, len(updates))
    return {
        key: np.tensordot(
            weights, np.stack([update[key] for update in updates]), axes=1
        )
        for key in keys
    }


def sample_weighted_fedavg_aggregate(
    client_states: Sequence[ClientUpdate],
    *,
    sample_counts: Sequence[int],
) -> RobustAggregationResult:
    """Return strict sample-count-weighted FedAvg over post-local states.

    This named baseline is intentionally stricter than passing arbitrary
    floating weights to :func:`fedavg_aggregate`: every participating client
    must provide one positive, non-boolean integer training-sample count.
    Coordinate contributions are sorted before summation, making the returned
    state invariant to a permutation of client records (including their paired
    counts) at float64 precision.
    """

    states, keys = _normalise_client_updates(client_states)
    counts = _strict_positive_sample_counts(sample_counts, len(states))
    total = sum(counts)
    try:
        total_float = float(total)
    except OverflowError as error:
        raise ValueError(
            "sample_counts total is too large for float64 weights"
        ) from error
    if not math.isfinite(total_float):
        raise ValueError("sample_counts total is too large for float64 weights")
    weights = np.asarray(counts, dtype=np.float64) / total_float
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("sample_counts produce invalid float64 client weights")

    update = {
        key: _permutation_invariant_weighted_mean(
            np.stack([state[key] for state in states]), weights
        )
        for key in keys
    }
    return RobustAggregationResult(
        update=update,
        diagnostics={
            "schema_version": 1,
            "reducer": "sample_count_weighted_mean",
            "sample_counts": list(counts),
            "normalised_client_weights": [float(value) for value in weights],
        },
    )


def median_norm_clipped_fedavg_aggregate(
    client_updates: Sequence[ClientUpdate],
) -> RobustAggregationResult:
    """Clip each client's full mapped update at the median positive norm.

    All keys in one client mapping are treated as one conceptual vector.  Its
    global Frobenius norm is computed without materialising that vector.  The
    threshold is the ordinary median of the strictly positive client norms;
    zero-norm clients remain zero and are not allowed to collapse a positive
    threshold.  If every client update is exactly zero, the threshold is zero
    and every scale is one.  The clipped updates then use a uniform mean.
    """

    updates, keys = _normalise_client_updates(client_updates)
    norms = tuple(_mapping_frobenius_norm(update, keys) for update in updates)
    positive_norms = tuple(value for value in norms if value > 0.0)
    threshold = _finite_median(positive_norms) if positive_norms else 0.0
    scales = tuple(
        1.0 if norm == 0.0 or norm <= threshold else threshold / norm for norm in norms
    )
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in scales):
        raise FloatingPointError("median-norm clipping produced invalid scales")

    clipped = [
        {key: np.asarray(update[key] * scale, dtype=np.float64) for key in keys}
        for update, scale in zip(updates, scales, strict=True)
    ]
    merged = {
        key: _permutation_invariant_uniform_mean(
            np.stack([update[key] for update in clipped])
        )
        for key in keys
    }
    sorted_norms = sorted(float(value) for value in norms)
    sorted_scales = sorted(float(value) for value in scales)
    return RobustAggregationResult(
        update=merged,
        diagnostics={
            "schema_version": 1,
            "reducer": "median_positive_norm_clipped_uniform_mean",
            "norm_scope": "all_mapped_keys_one_global_frobenius_norm_per_client",
            "clip_threshold_policy": "median_strictly_positive_client_norms",
            "clip_threshold": float(threshold),
            "positive_norm_count": len(positive_norms),
            "zero_norm_count": len(norms) - len(positive_norms),
            "clipped_client_count": sum(scale < 1.0 for scale in scales),
            # Sorted summaries make the diagnostic record itself independent
            # of arbitrary client enumeration.  The clipping computation above
            # still pairs each norm with its own update before aggregation.
            "client_norms_sorted": sorted_norms,
            "client_clip_scales_sorted": sorted_scales,
        },
    )


def coordinate_trimmed_mean_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    trim_fraction: float = 0.1,
) -> RobustAggregationResult:
    """Return a per-coordinate, equal-tail trimmed mean.

    ``floor(n_clients * trim_fraction)`` values are removed independently from
    each tail of every coordinate.  Sorting each coordinate makes exact ties
    deterministic and makes the output invariant to client ordering.
    """

    updates, keys = _normalise_client_updates(client_updates)
    fraction = _validate_trim_fraction(trim_fraction)
    trim_count = int(math.floor(len(updates) * fraction))
    retained_count = len(updates) - 2 * trim_count
    if retained_count <= 0:
        raise ValueError("trim_fraction leaves no clients to aggregate")

    merged: dict[str, FloatArray] = {}
    for key in keys:
        ordered = np.sort(
            np.stack([update[key] for update in updates]), axis=0, kind="stable"
        )
        retained = ordered[trim_count : len(updates) - trim_count]
        merged[key] = _permutation_invariant_uniform_mean(retained)
    return RobustAggregationResult(
        update=merged,
        diagnostics={
            "schema_version": 1,
            "reducer": "coordinate_equal_tail_trimmed_mean",
            "trim_fraction_each_tail": fraction,
            "trim_count_each_tail": trim_count,
            "retained_client_count": retained_count,
        },
    )


def coordinate_median_aggregate(
    client_updates: Sequence[ClientUpdate],
) -> RobustAggregationResult:
    """Return the deterministic coordinate-wise median across clients."""

    updates, keys = _normalise_client_updates(client_updates)
    merged = {
        key: _coordinate_median(
            np.sort(
                np.stack([update[key] for update in updates]),
                axis=0,
                kind="stable",
            )
        )
        for key in keys
    }
    return RobustAggregationResult(
        update=merged,
        diagnostics={
            "schema_version": 1,
            "reducer": "coordinate_median",
            "even_client_policy": "arithmetic_midpoint_of_two_central_values",
        },
    )


def task_arithmetic_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    beta: float,
    sample_counts: Sequence[float] | None = None,
) -> dict[str, FloatArray]:
    """Scale the (possibly sample-weighted) mean client task vector."""

    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number")
    return {
        key: beta * value
        for key, value in fedavg_aggregate(
            client_updates, sample_counts=sample_counts
        ).items()
    }


def ties_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    density: float = 0.2,
    sample_counts: Sequence[float] | None = None,
    scale: float = 1.0,
) -> dict[str, FloatArray]:
    """TIES trim/elect/disjoint-mean aggregation.

    Trimming is global over each client's flattened update.  The elected sign
    is the sign of the (weighted) sum of trimmed values.  The disjoint mean is
    taken only over non-zero entries whose sign matches the elected sign;
    trimmed zeros are deliberately excluded from its denominator.
    """

    updates, keys = _normalise_client_updates(client_updates)
    if not math.isfinite(density) or not 0.0 < density <= 1.0:
        raise ValueError("density must be in (0, 1]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")
    weights = _normalised_weights(sample_counts, len(updates))
    shapes = {key: updates[0][key].shape for key in keys}
    sizes = {key: updates[0][key].size for key in keys}
    flat = np.stack(
        [np.concatenate([update[key].ravel() for key in keys]) for update in updates]
    )
    n_parameters = flat.shape[1]
    keep = min(n_parameters, max(1, int(math.ceil(density * n_parameters))))
    trimmed = np.zeros_like(flat)
    for client_index, vector in enumerate(flat):
        # Stable sorting makes exact-magnitude ties deterministic.
        selected = np.argsort(-np.abs(vector), kind="stable")[:keep]
        trimmed[client_index, selected] = vector[selected]

    elected = np.sign(np.sum(weights[:, None] * trimmed, axis=0))
    matching = ((trimmed > 0.0) & (elected[None, :] > 0.0)) | (
        (trimmed < 0.0) & (elected[None, :] < 0.0)
    )
    matching_weights = weights[:, None] * matching
    denominator = np.sum(matching_weights, axis=0)
    numerator = np.sum(matching_weights * trimmed, axis=0)
    merged = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    merged *= scale

    result: dict[str, FloatArray] = {}
    offset = 0
    for key in keys:
        next_offset = offset + sizes[key]
        result[key] = merged[offset:next_offset].reshape(shapes[key])
        offset = next_offset
    return result


def ties_reference_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    density: float = 0.1,
) -> dict[str, FloatArray]:
    """Uniform, unit-scale TIES with the upstream mass-sign/mean semantics.

    This opt-in reducer leaves :func:`ties_aggregate` and historical runs
    unchanged.  ``density`` is explicitly a *retained* fraction, not an
    independently authenticated interpretation of another paper's ``s``.
    Per-client trimming spans the complete flattened adapter delta.  It uses
    the upstream one-based kth threshold ``d - int(d * density)`` and retains
    every entry at or above it, including ties.  Zero elected signs receive
    the global majority sign; if that too is zero, the upstream positive/else
    selector chooses negative entries.  Trimmed zeros never count in the mean.

    The reference kth operation is undefined at density one, so that endpoint
    is rejected rather than silently substituting a different trim rule.
    Reference: ``prateeky2806/ties-merging``, ``src/utils/merge_utils.py`` at
    commit ``44e7891fc84f3de7e4caa52664cd864ca3715e91`` (mass sign, mean).
    """

    updates, keys = _normalise_client_updates(client_updates)
    if (
        isinstance(density, bool)
        or not math.isfinite(density)
        or not 0.0 < density < 1.0
    ):
        raise ValueError("reference TIES density must be in (0, 1)")
    shapes = {key: updates[0][key].shape for key in keys}
    sizes = {key: updates[0][key].size for key in keys}
    flat = np.stack(
        [np.concatenate([update[key].ravel() for key in keys]) for update in updates]
    )
    dimension = flat.shape[1]
    kth = dimension - int(dimension * density)
    threshold = np.partition(np.abs(flat), kth - 1, axis=1)[:, kth - 1]
    trimmed = flat * (np.abs(flat) >= threshold[:, None])
    elected = np.sign(np.sum(trimmed, axis=0))
    majority = np.sign(np.sum(elected))
    elected[elected == 0.0] = majority
    selected = np.where(elected[None, :] > 0.0, trimmed > 0.0, trimmed < 0.0)
    selected_values = trimmed * selected
    count = np.sum(selected_values != 0.0, axis=0)
    merged = np.sum(selected_values, axis=0) / np.maximum(count, 1)
    result: dict[str, FloatArray] = {}
    offset = 0
    for key in keys:
        next_offset = offset + sizes[key]
        result[key] = merged[offset:next_offset].reshape(shapes[key])
        offset = next_offset
    return result


def fedrpca_aggregate(
    client_updates: Sequence[ClientUpdate],
    *,
    adapter_keys: Iterable[str] | None = None,
    beta: float = 2.0,
    adaptive_beta: bool = False,
    cancellation_aware_rms: bool = False,
    beta_max: float | None = None,
    sample_counts: Sequence[float] | None = None,
    reliability_gated: bool = False,
    residual_gate_threshold: float = 1e-4,
    cosine_gap_target: float = 0.25,
    rpca_lambda: float | None = None,
    rpca_mu: float | None = None,
    rpca_tol: float = 1e-7,
    rpca_max_iter: int = 1_000,
    rpca_secondary_max_iter: int | None = None,
    rpca_rho: float | None = None,
) -> AggregationResult:
    """Aggregate LoRA updates with independent PCP decompositions.

    For block ``b`` with normalised aggregation weights ``w``, FedRPCA returns

    ``L_b @ w + beta_b * (S_b @ w)``.

    With ``adaptive_beta=True``, ``cancellation_aware_rms=False``, and omitted
    ``sample_counts``, ``w`` is uniform and the paper's schedule is used
    independently for every block:

    ``E_b = ||S_b @ w||_2 / ||M_b @ w||_2`` and
    ``beta_base = 1 / E_b``.  This is the unbounded schedule in Section B.3
    of the paper. Passing ``sample_counts`` evaluates the same formula with
    sample weights, which is explicitly diagnosed as an extension rather than
    the paper policy. Passing ``beta_max`` additionally enables a separately
    named numerical-safety policy ``clip(1 / E_b, 1, beta_max)``. When the
    aggregate sparse component is numerically zero, amplification is immaterial
    and is recorded as ``beta_base=1`` instead of an undefined infinity.

    Setting ``cancellation_aware_rms=True`` selects a parameter-free extension
    that keeps the same PCP decomposition but replaces the norm-of-sums energy
    estimate with weighted RMS column energy:

    ``beta_base = sqrt(sum_i w_i ||M_i||^2) /``
    ``sqrt(sum_i w_i ||S_i||^2))``.

    This distinguishes genuinely low sparse energy from cancellation among
    individually large client-specific columns.  An exactly zero sparse matrix
    uses ``beta_base=1``.  A ratio below one is retained rather than floored:
    this is a pure replacement of the paper energy estimator.  The extension
    cannot be combined with a beta clip or reliability gate, so there is no
    hidden tunable mixing coefficient.

    The optional reliability gate is a 2025-context extension.  It uses

    ``g_res = clip(1 - residual / residual_gate_threshold, 0, 1)``
    ``g_cos = clip((cos(L)-cos(S)) / cosine_gap_target, 0, 1)``
    ``g = g_res * g_cos`` and ``beta_eff = 1 + g*(beta_base - 1)``.

    Thus either a poor numerical decomposition or a failure to separate common
    from client-specific cosine structure smoothly recovers unamplified
    aggregation.  This is a testable heuristic, not a convergence claim. A
    50-seed synthetic evaluation found a practically negligible effect, so callers
    should treat it as an experimental ablation rather than an established
    improvement.

    ``adapter_keys`` is explicit by design.  If omitted, conservative LoRA A/B
    name inference is used.  Every selected key is decomposed independently;
    keys containing a ``bias`` token are rejected and all non-adapter keys use
    FedAvg.
    """

    updates, keys = _normalise_client_updates(client_updates)
    weights = _normalised_weights(sample_counts, len(updates))
    if not isinstance(cancellation_aware_rms, (bool, np.bool_)):
        raise TypeError("cancellation_aware_rms must be a boolean")
    cancellation_aware_rms = bool(cancellation_aware_rms)
    if cancellation_aware_rms:
        if not adaptive_beta:
            raise ValueError("cancellation_aware_rms requires adaptive_beta=True")
        if beta_max is not None:
            raise ValueError("cancellation_aware_rms does not permit beta_max")
        if reliability_gated:
            raise ValueError(
                "cancellation_aware_rms does not permit reliability gating"
            )
    if adaptive_beta:
        if beta_max is not None and (not math.isfinite(beta_max) or beta_max <= 1.0):
            raise ValueError("beta_max must be None or finite and greater than one")
    elif not math.isfinite(beta) or beta <= 1.0:
        raise ValueError("fixed FedRPCA beta must be finite and greater than one")
    if not math.isfinite(residual_gate_threshold) or residual_gate_threshold <= 0:
        raise ValueError("residual_gate_threshold must be finite and positive")
    if not math.isfinite(cosine_gap_target) or cosine_gap_target <= 0:
        raise ValueError("cosine_gap_target must be finite and positive")
    _validate_iteration_cap(rpca_max_iter, name="rpca_max_iter")
    if rpca_secondary_max_iter is not None:
        _validate_iteration_cap(rpca_secondary_max_iter, name="rpca_secondary_max_iter")
        if int(rpca_secondary_max_iter) <= int(rpca_max_iter):
            raise ValueError(
                "rpca_secondary_max_iter must be greater than rpca_max_iter"
            )

    if adapter_keys is None:
        selected = tuple(key for key in keys if _looks_like_lora_factor(key))
        if not selected:
            raise ValueError(
                "no LoRA A/B blocks were inferred; pass adapter_keys explicitly"
            )
    else:
        selected = tuple(adapter_keys)
        if not selected:
            raise ValueError("adapter_keys must contain at least one block")
    if len(set(selected)) != len(selected):
        raise ValueError("adapter_keys contains duplicate block names")
    unknown = sorted(set(selected) - set(keys))
    if unknown:
        raise KeyError(f"adapter_keys not present in all updates: {unknown}")
    biases = sorted(key for key in selected if _is_bias_key(key))
    if biases:
        raise ValueError(
            "bias parameters must use FedAvg and cannot be RPCA adapter blocks: "
            f"{biases}"
        )

    # Start with FedAvg so biases and any explicitly non-adapter trainables are
    # kept, but never enter an RPCA matrix.
    aggregate = fedavg_aggregate(updates, sample_counts=sample_counts)
    diagnostics: dict[str, BlockDiagnostics] = {}
    epsilon = np.finfo(np.float64).eps

    for key in selected:
        shape = updates[0][key].shape
        observed = np.column_stack([update[key].ravel() for update in updates])
        primary_decomposition = principal_component_pursuit_admm(
            observed,
            lam=rpca_lambda,
            mu=rpca_mu,
            tol=rpca_tol,
            max_iter=rpca_max_iter,
            rho=rpca_rho,
        )
        fallback_triggered = bool(
            rpca_secondary_max_iter is not None and not primary_decomposition.converged
        )
        if fallback_triggered:
            # This is a deterministic numerical-completion retry, not a warm
            # start or a modified PCP problem.  Calling the canonical solver
            # again reconstructs its zero initial L/S/Y state and reuses the
            # exact lambda, mu, rho and tolerance; only the iteration cap is
            # larger.  A primary success therefore follows the original code
            # path without a second solver invocation.
            decomposition = principal_component_pursuit_admm(
                observed,
                lam=rpca_lambda,
                mu=rpca_mu,
                tol=rpca_tol,
                max_iter=int(rpca_secondary_max_iter),
                rho=rpca_rho,
            )
            final_max_iter = int(rpca_secondary_max_iter)
        else:
            decomposition = primary_decomposition
            final_max_iter = int(rpca_max_iter)
        # Form means and the inverse-energy ratio in block-normalized
        # coordinates.  The final multiplication by the same exact power of
        # two leaves the aggregation rule unchanged, but prevents large finite
        # updates from overflowing their intermediate norms and makes the
        # numerical-zero guard relative to the block rather than to 1.0.
        block_scale = _power_of_two_scale(observed)
        low_mean = (decomposition.low_rank / block_scale) @ weights
        sparse_mean = (decomposition.sparse / block_scale) @ weights
        observed_mean = (observed / block_scale) @ weights
        sparse_norm = float(np.linalg.norm(sparse_mean))
        denominator = max(float(np.linalg.norm(observed_mean)), epsilon)
        energy_ratio = float(sparse_norm / denominator)
        if adaptive_beta:
            if cancellation_aware_rms:
                observed_rms = _weighted_rms_column_norm(
                    observed / block_scale, weights
                )
                sparse_rms = _weighted_rms_column_norm(
                    decomposition.sparse / block_scale, weights
                )
                sparse_exact_zero_guard = not bool(np.any(decomposition.sparse != 0.0))
                sparse_weighted_zero_guard = sparse_rms == 0.0
                observed_coherence = _column_mean_coherence(observed_mean, observed_rms)
                sparse_coherence = _column_mean_coherence(sparse_mean, sparse_rms)
                coherence_ratio = (
                    None
                    if observed_coherence == 0.0
                    else float(sparse_coherence / observed_coherence)
                )
                if sparse_weighted_zero_guard:
                    rms_energy_ratio = 0.0
                    raw_rms_beta = None
                    beta_base = 1.0
                else:
                    raw_rms_beta = float(observed_rms / sparse_rms)
                    rms_energy_ratio = (
                        None
                        if observed_rms == 0.0
                        else float(sparse_rms / observed_rms)
                    )
                    beta_base = raw_rms_beta
                beta_policy = (
                    "cancellation_aware_rms_energy"
                    if sample_counts is None
                    else "sample_weighted_cancellation_aware_rms_energy_extension"
                )
            elif sparse_norm <= epsilon:
                beta_base = 1.0
            elif beta_max is None:
                beta_base = float(1.0 / energy_ratio)
            else:
                beta_base = float(np.clip(1.0 / energy_ratio, 1.0, beta_max))
            if cancellation_aware_rms:
                pass
            elif sample_counts is None:
                beta_policy = (
                    "paper_inverse_energy"
                    if beta_max is None
                    else "safety_clipped_inverse_energy"
                )
            else:
                beta_policy = (
                    "sample_weighted_inverse_energy_extension"
                    if beta_max is None
                    else "sample_weighted_safety_clipped_inverse_energy_extension"
                )
        else:
            beta_base = float(beta)
            beta_policy = "fixed"

        observed_cosine = _mean_pairwise_column_cosine(observed)
        low_cosine = _mean_pairwise_column_cosine(decomposition.low_rank)
        sparse_cosine = _mean_pairwise_column_cosine(decomposition.sparse)
        cosine_gap = low_cosine - sparse_cosine
        if reliability_gated:
            residual_score = float(
                np.clip(
                    1.0 - decomposition.residual / residual_gate_threshold,
                    0.0,
                    1.0,
                )
            )
            cosine_score = float(np.clip(cosine_gap / cosine_gap_target, 0.0, 1.0))
            gate = residual_score * cosine_score
        else:
            gate = 1.0
        beta_effective = (
            beta_base if cancellation_aware_rms else 1.0 + gate * (beta_base - 1.0)
        )
        normalized_aggregate = low_mean + beta_effective * sparse_mean
        try:
            with np.errstate(over="raise", invalid="raise"):
                aggregate[key] = (normalized_aggregate * block_scale).reshape(shape)
        except FloatingPointError as error:
            raise FloatingPointError(
                f"FedRPCA aggregate for block {key!r} is not representable"
            ) from error
        if not np.all(np.isfinite(aggregate[key])):
            raise FloatingPointError(
                f"FedRPCA aggregate for block {key!r} is not representable"
            )
        diagnostic_kwargs = dict(
            residual=decomposition.residual,
            rank=decomposition.rank,
            sparsity=decomposition.sparsity,
            observed_cosine=observed_cosine,
            low_rank_cosine=low_cosine,
            sparse_cosine=sparse_cosine,
            cosine_gap=cosine_gap,
            energy_ratio=energy_ratio,
            beta_policy=beta_policy,
            beta_base=beta_base,
            gate=gate,
            beta=beta_effective,
            iterations=decomposition.iterations,
            converged=decomposition.converged,
            primary_max_iter=int(rpca_max_iter),
            primary_residual=primary_decomposition.residual,
            primary_converged=primary_decomposition.converged,
            fallback_triggered=fallback_triggered,
            final_max_iter=final_max_iter,
        )
        if cancellation_aware_rms:
            diagnostics[key] = CancellationAwareBlockDiagnostics(
                **diagnostic_kwargs,
                rms_energy_ratio=rms_energy_ratio,
                raw_rms_beta=raw_rms_beta,
                observed_coherence=observed_coherence,
                sparse_coherence=sparse_coherence,
                coherence_ratio=coherence_ratio,
                sparse_exact_zero_guard=sparse_exact_zero_guard,
                sparse_weighted_zero_guard=sparse_weighted_zero_guard,
            )
        else:
            diagnostics[key] = BlockDiagnostics(**diagnostic_kwargs)

    return AggregationResult(update=aggregate, diagnostics=diagnostics)


def reliability_gated_fedrpca(
    client_updates: Sequence[ClientUpdate],
    **kwargs: object,
) -> AggregationResult:
    """Convenience wrapper for reliability-gated FedRPCA.

    Keyword arguments are forwarded to :func:`fedrpca_aggregate`.  Passing an
    explicit false value for ``reliability_gated`` is rejected so the wrapper's
    name cannot silently misdescribe the experiment.
    """

    if "reliability_gated" in kwargs and kwargs["reliability_gated"] is not True:
        raise ValueError("reliability_gated_fedrpca always enables the gate")
    kwargs["reliability_gated"] = True
    return fedrpca_aggregate(client_updates, **kwargs)  # type: ignore[arg-type]


def fedrpca_aggregate_blocks(
    blocks: Mapping[str, Sequence[ArrayLike]],
    **kwargs: object,
) -> AggregationResult:
    """Generic block-oriented front end for independent FedRPCA decomposition.

    ``blocks`` maps each LoRA A or B block name to one array per client.  This
    orientation makes it impossible for the function itself to concatenate A,
    B, or bias parameters.  Bias-named blocks are still rejected defensively.
    """

    if not blocks:
        raise ValueError("blocks must not be empty")
    block_names = tuple(blocks)
    n_clients = len(blocks[block_names[0]])
    if n_clients == 0:
        raise ValueError("every block must contain at least one client array")
    if any(len(blocks[name]) != n_clients for name in block_names):
        raise ValueError("every block must contain the same number of clients")
    client_updates = [
        {name: blocks[name][client_index] for name in block_names}
        for client_index in range(n_clients)
    ]
    if "adapter_keys" in kwargs:
        raise TypeError("fedrpca_aggregate_blocks sets adapter_keys from blocks")
    return fedrpca_aggregate(
        client_updates, adapter_keys=block_names, **kwargs  # type: ignore[arg-type]
    )


def make_paired_run_plan(
    labels: ArrayLike,
    *,
    n_clients: int,
    alpha: float,
    rounds: int,
    seed: int,
    min_client_size: int = 1,
    max_attempts: int = 1_000,
) -> PairedRunPlan:
    """Create a deterministic Dirichlet partition and paired RNG manifest."""

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    for name, value in (
        ("n_clients", n_clients),
        ("rounds", rounds),
        ("min_client_size", min_client_size),
        ("max_attempts", max_attempts),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if n_clients <= 0 or rounds <= 0 or min_client_size < 0 or max_attempts <= 0:
        raise ValueError(
            "n_clients, rounds, and max_attempts must be positive; "
            "min_client_size must be non-negative"
        )
    if label_array.size < n_clients * min_client_size:
        raise ValueError("not enough samples to satisfy min_client_size")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite positive number")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    partition_sequence, initialization_sequence, shuffle_sequence = (
        np.random.SeedSequence(int(seed)).spawn(3)
    )
    partition_rng = np.random.default_rng(partition_sequence)
    client_indices: list[list[int]] | None = None
    classes = np.unique(label_array)
    for _ in range(int(max_attempts)):
        candidate: list[list[int]] = [[] for _ in range(int(n_clients))]
        for class_label in classes:
            indices = np.flatnonzero(label_array == class_label)
            indices = partition_rng.permutation(indices)
            proportions = partition_rng.dirichlet(
                np.full(int(n_clients), float(alpha), dtype=np.float64)
            )
            counts = partition_rng.multinomial(indices.size, proportions)
            cursor = 0
            for client_index, count in enumerate(counts):
                next_cursor = cursor + int(count)
                candidate[client_index].extend(
                    int(index) for index in indices[cursor:next_cursor]
                )
                cursor = next_cursor
        if min(map(len, candidate)) >= min_client_size:
            client_indices = candidate
            break
    if client_indices is None:
        raise RuntimeError(
            "could not draw a Dirichlet partition satisfying min_client_size; "
            "increase alpha, lower min_client_size, or increase max_attempts"
        )

    initialization_rng = np.random.default_rng(initialization_sequence)
    initialization_seed = int(
        initialization_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
    )
    shuffle_rng = np.random.default_rng(shuffle_sequence)
    local_seeds_array = shuffle_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=(int(rounds), int(n_clients)),
        dtype=np.uint32,
    )
    return PairedRunPlan(
        seed=int(seed),
        client_indices=tuple(tuple(sorted(indices)) for indices in client_indices),
        initialization_seed=initialization_seed,
        local_shuffle_seeds=tuple(
            tuple(int(value) for value in round_seeds)
            for round_seeds in local_seeds_array
        ),
        alpha=float(alpha),
    )


def make_hsu_balanced_run_plan(
    labels: ArrayLike,
    *,
    n_clients: int,
    alpha: float,
    rounds: int,
    seed: int,
) -> HsuBalancedRunPlan:
    """Create a balanced per-client Dirichlet partition and paired RNG plan.

    This follows the construction described by Hsu et al.: if ``p`` is the
    empirical population label distribution, each client's target label
    distribution is drawn independently as ``q_i ~ Dirichlet(alpha * p)``.
    The finite dataset is then assigned without replacement while enforcing
    balanced client capacities.  When the sample count is not divisible by the
    client count, a seeded random subset of clients receives one extra sample.

    Exact row (client-capacity) and column (observed-class-count) constraints
    necessarily condition the raw Dirichlet draws.  A Sinkhorn-scaled transport
    table is used to preserve their relative preferences before integer
    allocation.  Partition, model-initialisation, and local-shuffle randomness
    use independent child seed sequences, just as in :func:`make_paired_run_plan`.
    Consequently, the two constructors share initialisation and shuffle seeds
    for the same ``seed``, ``rounds``, and ``n_clients``.
    """

    label_array = np.asarray(labels)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if label_array.size == 0:
        raise ValueError("labels must not be empty")
    for name, value in (("n_clients", n_clients), ("rounds", rounds)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be an integer")
    if n_clients <= 0 or rounds <= 0:
        raise ValueError("n_clients and rounds must be positive")
    if label_array.size < n_clients:
        raise ValueError("not enough samples to give every balanced client one sample")
    if isinstance(alpha, bool) or not isinstance(
        alpha, (int, float, np.integer, np.floating)
    ):
        raise TypeError("alpha must be a real number")
    if not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise ValueError("alpha must be a finite positive number")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")

    try:
        unique_labels, inverse_labels, class_counts = np.unique(
            label_array, return_inverse=True, return_counts=True
        )
    except TypeError as error:
        raise TypeError("labels must contain mutually comparable values") from error
    canonical_labels = tuple(_canonical_class_label(value) for value in unique_labels)
    class_counts = np.asarray(class_counts, dtype=np.int64)
    population_distribution = class_counts.astype(np.float64) / label_array.size
    concentration = float(alpha) * population_distribution
    if not np.all(np.isfinite(concentration)) or np.any(concentration <= 0.0):
        raise ValueError(
            "alpha is too small or large for the observed class probabilities"
        )

    partition_sequence, initialization_sequence, shuffle_sequence = (
        np.random.SeedSequence(int(seed)).spawn(3)
    )
    partition_rng = np.random.default_rng(partition_sequence)
    client_preferences = partition_rng.dirichlet(concentration, size=int(n_clients))

    base_size, remainder = divmod(label_array.size, int(n_clients))
    client_sizes = np.full(int(n_clients), base_size, dtype=np.int64)
    if remainder:
        extra_clients = partition_rng.permutation(int(n_clients))[:remainder]
        client_sizes[extra_clients] += 1

    allocation = _balanced_dirichlet_allocation(
        client_preferences,
        client_sizes=client_sizes,
        class_counts=class_counts,
        rng=partition_rng,
    )
    client_indices: list[list[int]] = [[] for _ in range(int(n_clients))]
    for class_index in range(class_counts.size):
        indices = np.flatnonzero(inverse_labels == class_index)
        indices = partition_rng.permutation(indices)
        cursor = 0
        for client_index in range(int(n_clients)):
            count = int(allocation[client_index, class_index])
            next_cursor = cursor + count
            client_indices[client_index].extend(
                int(index) for index in indices[cursor:next_cursor]
            )
            cursor = next_cursor
        if cursor != indices.size:
            raise RuntimeError("balanced allocation did not consume every sample")

    initialization_rng = np.random.default_rng(initialization_sequence)
    initialization_seed = int(
        initialization_rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)
    )
    shuffle_rng = np.random.default_rng(shuffle_sequence)
    local_seeds_array = shuffle_rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=(int(rounds), int(n_clients)),
        dtype=np.uint32,
    )
    return HsuBalancedRunPlan(
        seed=int(seed),
        client_indices=tuple(tuple(sorted(indices)) for indices in client_indices),
        initialization_seed=initialization_seed,
        local_shuffle_seeds=tuple(
            tuple(int(value) for value in round_seeds)
            for round_seeds in local_seeds_array
        ),
        alpha=float(alpha),
        class_labels=canonical_labels,
        population_distribution=tuple(
            float(value) for value in population_distribution
        ),
        dirichlet_concentration=tuple(float(value) for value in concentration),
        sampled_client_label_distributions=tuple(
            tuple(float(value) for value in row) for row in client_preferences
        ),
        realized_client_class_counts=tuple(
            tuple(int(value) for value in row) for row in allocation
        ),
    )


def _balanced_dirichlet_allocation(
    client_preferences: FloatArray,
    *,
    client_sizes: NDArray[np.int64],
    class_counts: NDArray[np.int64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Integerise Dirichlet preferences under exact row/column margins."""

    # Dirichlet draws with concentration below one can legitimately underflow
    # to zero in float64.  A smallest-positive-value floor retains the intended
    # near-zero preference while keeping the transport problem fully supported.
    log_kernel = np.log(np.maximum(client_preferences, np.finfo(np.float64).tiny))
    log_rows = np.log(client_sizes.astype(np.float64))
    log_columns = np.log(class_counts.astype(np.float64))
    log_row_scale = np.zeros(client_sizes.size, dtype=np.float64)
    log_column_scale = np.zeros(class_counts.size, dtype=np.float64)

    # Log-domain Sinkhorn scaling avoids overflow for the highly concentrated
    # alpha values used in federated non-IID experiments.
    for iteration in range(20_000):
        log_row_scale = log_rows - scipy_special.logsumexp(
            log_kernel + log_column_scale[None, :], axis=1
        )
        log_column_scale = log_columns - scipy_special.logsumexp(
            log_kernel + log_row_scale[:, None], axis=0
        )
        if iteration % 25 == 24:
            target = np.exp(
                log_kernel + log_row_scale[:, None] + log_column_scale[None, :]
            )
            row_error = float(np.max(np.abs(np.sum(target, axis=1) - client_sizes)))
            column_error = float(np.max(np.abs(np.sum(target, axis=0) - class_counts)))
            if max(row_error, column_error) <= 1e-8:
                break

    target = np.exp(log_kernel + log_row_scale[:, None] + log_column_scale[None, :])
    # Guard integerisation even if an exceptionally ill-conditioned valid
    # preference matrix reaches the iteration cap before strict convergence.
    row_scale = np.minimum(
        1.0, client_sizes / np.maximum(np.sum(target, axis=1), 1e-300)
    )
    target *= row_scale[:, None]
    column_scale = np.minimum(
        1.0, class_counts / np.maximum(np.sum(target, axis=0), 1e-300)
    )
    target *= column_scale[None, :]

    allocation = np.floor(target).astype(np.int64)
    row_deficit = client_sizes - np.sum(allocation, axis=1)
    column_deficit = class_counts - np.sum(allocation, axis=0)
    if np.any(row_deficit < 0) or np.any(column_deficit < 0):
        raise RuntimeError("balanced allocation rounding exceeded a margin")
    if int(np.sum(row_deficit)) != int(np.sum(column_deficit)):
        raise RuntimeError("balanced allocation margins are inconsistent")

    fractional = target - np.floor(target)
    while np.any(row_deficit):
        active_rows = np.flatnonzero(row_deficit > 0)
        row_weights = row_deficit[active_rows].astype(np.float64)
        client_index = int(_partition_choice(rng, active_rows, weights=row_weights))
        active_columns = np.flatnonzero(column_deficit > 0)
        weights = fractional[client_index, active_columns]
        if not np.any(weights > 0.0):
            weights = client_preferences[client_index, active_columns]
        if not np.any(weights > 0.0):
            weights = np.ones(active_columns.size, dtype=np.float64)
        class_index = int(
            _partition_choice(
                rng,
                active_columns,
                weights=weights * column_deficit[active_columns],
            )
        )
        allocation[client_index, class_index] += 1
        fractional[client_index, class_index] = 0.0
        row_deficit[client_index] -= 1
        column_deficit[class_index] -= 1

    if not np.array_equal(np.sum(allocation, axis=1), client_sizes):
        raise RuntimeError("balanced allocation violated client capacities")
    if not np.array_equal(np.sum(allocation, axis=0), class_counts):
        raise RuntimeError("balanced allocation violated observed class counts")
    return allocation


def _partition_choice(
    rng: np.random.Generator,
    values: NDArray[np.int64],
    *,
    weights: FloatArray,
) -> np.int64:
    """Draw one integer value from non-negative unnormalised weights."""

    probabilities = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(probabilities))
    if not math.isfinite(total) or total <= 0.0:
        probabilities = np.full(values.size, 1.0 / values.size)
    else:
        probabilities = probabilities / total
    return np.int64(rng.choice(values, p=probabilities))


def _canonical_class_label(value: object) -> object:
    """Return a deterministic JSON scalar for a class label."""

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("class labels must be finite")
        return value
    raise TypeError("class labels must be JSON-serialisable scalar values")


def _json_sha256(value: object) -> str:
    """Hash a JSON-compatible value using the plan digest encoding."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _soft_threshold(matrix: FloatArray, threshold: float) -> FloatArray:
    return np.sign(matrix) * np.maximum(np.abs(matrix) - threshold, 0.0)


def _power_of_two_scale(matrix: FloatArray) -> float:
    """Return an exact finite scale that puts ``max(abs(matrix))`` in [1, 2).

    A zero matrix uses scale one.  Powers of two avoid an additional decimal
    rounding factor and keep the normalization defined for both subnormal
    values and values near float64 max.
    """

    magnitude = float(np.max(np.abs(matrix)))
    if magnitude == 0.0:
        return 1.0
    _, exponent = math.frexp(magnitude)
    return math.ldexp(1.0, exponent - 1)


def _stable_frobenius_norm(matrix: FloatArray) -> float:
    magnitude = float(np.max(np.abs(matrix)))
    if magnitude == 0.0:
        return 0.0
    normalized_norm = float(np.linalg.norm(matrix / magnitude, ord="fro"))
    return magnitude * normalized_norm


def _stable_l1_norm(matrix: FloatArray) -> float:
    magnitude = float(np.max(np.abs(matrix)))
    if magnitude == 0.0:
        return 0.0
    normalized_sum = float(np.sum(np.abs(matrix / magnitude)))
    return magnitude * normalized_sum


def _svd_with_sanity(
    matrix: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute a thin SVD, retrying with LAPACK ``gesvd`` if sanity fails.

    Some older NumPy/BLAS combinations have returned finite but corrupt
    ``gesdd`` factors for the tall, narrow matrices used by FedRPCA.  Merely
    checking finiteness does not detect that failure mode.  The checks below
    verify orthogonality, singular-value energy, and full reconstruction before
    accepting the factors.
    """

    try:
        numpy_factors = np.linalg.svd(matrix, full_matrices=False)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        numpy_factors = None
    if numpy_factors is not None and _svd_factorization_is_sane(matrix, *numpy_factors):
        return numpy_factors

    try:
        scipy_factors = scipy_linalg.svd(
            matrix,
            full_matrices=False,
            check_finite=False,
            lapack_driver="gesvd",
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError) as error:
        raise FloatingPointError(
            "both the default SVD and scipy LAPACK gesvd failed"
        ) from error
    if not _svd_factorization_is_sane(matrix, *scipy_factors):
        raise FloatingPointError(
            "both the default SVD and scipy LAPACK gesvd returned invalid factors"
        )
    return scipy_factors


def _svd_factorization_is_sane(
    matrix: FloatArray,
    u: FloatArray,
    singular_values: FloatArray,
    vt: FloatArray,
) -> bool:
    rows, columns = matrix.shape
    rank = min(rows, columns)
    if (
        u.shape != (rows, rank)
        or singular_values.shape != (rank,)
        or vt.shape != (rank, columns)
    ):
        return False
    if not (
        np.all(np.isfinite(u))
        and np.all(np.isfinite(singular_values))
        and np.all(np.isfinite(vt))
    ):
        return False
    if np.any(singular_values < 0.0):
        return False
    tolerance = 1e-8
    if float(np.max(np.abs(u))) > 1.0 + tolerance or float(np.max(np.abs(vt))) > (
        1.0 + tolerance
    ):
        return False
    identity = np.eye(rank, dtype=np.float64)
    u_orthogonality = float(np.linalg.norm(u.T @ u - identity, ord="fro"))
    vt_orthogonality = float(np.linalg.norm(vt @ vt.T - identity, ord="fro"))
    orthogonality_limit = tolerance * max(1.0, math.sqrt(rank))
    if (
        not math.isfinite(u_orthogonality)
        or not math.isfinite(vt_orthogonality)
        or u_orthogonality > orthogonality_limit
        or vt_orthogonality > orthogonality_limit
    ):
        return False

    matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
    singular_norm = float(np.linalg.norm(singular_values))
    if not math.isfinite(matrix_norm) or not math.isfinite(singular_norm):
        return False
    if not math.isclose(
        singular_norm, matrix_norm, rel_tol=tolerance, abs_tol=tolerance * matrix_norm
    ):
        return False

    reconstruction = (u * singular_values) @ vt
    reconstruction_error = float(np.linalg.norm(reconstruction - matrix, ord="fro"))
    reconstruction_limit = tolerance * matrix_norm
    return math.isfinite(reconstruction_error) and (
        reconstruction_error <= reconstruction_limit
    )


def _as_finite_float_array(
    value: ArrayLike, *, name: str, ndim: int | None = None
) -> FloatArray:
    array = np.asarray(value)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(array, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _normalise_client_updates(
    client_updates: Sequence[ClientUpdate],
) -> tuple[list[dict[str, FloatArray]], tuple[str, ...]]:
    if not client_updates:
        raise ValueError("client_updates must contain at least one client")
    if not isinstance(client_updates[0], Mapping) or not client_updates[0]:
        raise ValueError("each client update must be a non-empty mapping")
    keys = tuple(sorted(client_updates[0]))
    expected_keys = set(keys)
    normalised: list[dict[str, FloatArray]] = []
    expected_shapes: dict[str, tuple[int, ...]] = {}
    for client_index, update in enumerate(client_updates):
        if not isinstance(update, Mapping):
            raise TypeError(f"client update {client_index} is not a mapping")
        if set(update) != expected_keys:
            raise ValueError("all client updates must have identical parameter keys")
        converted: dict[str, FloatArray] = {}
        for key in keys:
            value = _as_finite_float_array(
                update[key], name=f"client_updates[{client_index}][{key!r}]"
            )
            if client_index == 0:
                expected_shapes[key] = value.shape
            elif value.shape != expected_shapes[key]:
                raise ValueError(f"parameter {key!r} has inconsistent client shapes")
            converted[key] = value
        normalised.append(converted)
    return normalised, keys


def _normalised_weights(
    sample_counts: Sequence[float] | None, n_clients: int
) -> FloatArray:
    if sample_counts is None:
        return np.full(n_clients, 1.0 / n_clients, dtype=np.float64)
    counts = _as_finite_float_array(sample_counts, name="sample_counts", ndim=1)
    if counts.size != n_clients:
        raise ValueError("sample_counts length must equal the number of clients")
    if np.any(counts < 0.0) or float(np.sum(counts)) <= 0.0:
        raise ValueError("sample_counts must be non-negative with a positive sum")
    return counts / np.sum(counts)


def _strict_positive_sample_counts(
    sample_counts: Sequence[int], n_clients: int
) -> tuple[int, ...]:
    if isinstance(sample_counts, (str, bytes)) or not isinstance(
        sample_counts, Sequence
    ):
        raise TypeError("sample_counts must be a sequence of positive integers")
    if len(sample_counts) != n_clients:
        raise ValueError("sample_counts length must equal the number of clients")
    counts: list[int] = []
    for client_index, value in enumerate(sample_counts):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"sample_counts[{client_index}] must be a positive integer")
        count = int(value)
        if count <= 0:
            raise ValueError(
                f"sample_counts[{client_index}] must be a positive integer"
            )
        counts.append(count)
    return tuple(counts)


def _validate_trim_fraction(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError("trim_fraction must be a real number")
    fraction = float(value)
    if not math.isfinite(fraction) or not 0.0 <= fraction < 0.5:
        raise ValueError("trim_fraction must be finite and in [0, 0.5)")
    return fraction


def _mapping_frobenius_norm(
    update: Mapping[str, FloatArray], keys: Sequence[str]
) -> float:
    magnitude = max(float(np.max(np.abs(update[key]))) for key in keys)
    if magnitude == 0.0:
        return 0.0
    squared_scaled_norm = math.fsum(
        float(np.sum(np.square(update[key] / magnitude))) for key in keys
    )
    norm = magnitude * math.sqrt(squared_scaled_norm)
    if not math.isfinite(norm):
        raise FloatingPointError(
            "client full-update Frobenius norm is not representable in float64"
        )
    return norm


def _finite_median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("median values must be finite")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    # Halving before addition avoids overflow for two equal near-max values.
    result = ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
    if not math.isfinite(result):
        raise FloatingPointError("median midpoint is not representable in float64")
    return result


def _permutation_invariant_weighted_mean(
    stacked: FloatArray, weights: FloatArray
) -> FloatArray:
    reshape = (weights.size,) + (1,) * (stacked.ndim - 1)
    contributions = stacked * weights.reshape(reshape)
    ordered = np.sort(contributions, axis=0, kind="stable")
    result = np.sum(ordered, axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("weighted aggregation produced non-finite values")
    return np.asarray(result, dtype=np.float64)


def _permutation_invariant_uniform_mean(stacked: FloatArray) -> FloatArray:
    ordered = np.sort(stacked, axis=0, kind="stable")
    # Divide before summation: a convex mean of finite values is finite even
    # when the unscaled same-sign sum would overflow.
    result = np.sum(ordered / stacked.shape[0], axis=0, dtype=np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("uniform aggregation produced non-finite values")
    return np.asarray(result, dtype=np.float64)


def _coordinate_median(ordered: FloatArray) -> FloatArray:
    middle = ordered.shape[0] // 2
    if ordered.shape[0] % 2:
        result = ordered[middle].copy()
    else:
        result = ordered[middle - 1] / 2.0 + ordered[middle] / 2.0
    if not np.isfinite(result).all():
        raise FloatingPointError("coordinate median produced non-finite values")
    return np.asarray(result, dtype=np.float64)


def _validate_iteration_cap(value: object, *, name: str) -> None:
    """Validate an explicit deterministic solver iteration cap."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _mean_pairwise_column_cosine(matrix: FloatArray) -> float:
    n_clients = matrix.shape[1]
    if n_clients < 2:
        return 0.0
    # Normalize every client column before taking the Gram matrix.  Computing
    # ``matrix.T @ matrix`` first can overflow even though every cosine is in
    # [-1, 1], and direct column norms can underflow for tiny finite updates.
    magnitudes = np.max(np.abs(matrix), axis=0)
    scaled = np.divide(
        matrix,
        magnitudes[None, :],
        out=np.zeros_like(matrix),
        where=magnitudes[None, :] > 0.0,
    )
    norms = np.linalg.norm(scaled, axis=0)
    unit = np.divide(
        scaled,
        norms[None, :],
        out=np.zeros_like(scaled),
        where=norms[None, :] > 0.0,
    )
    cosine = unit.T @ unit
    upper = np.triu_indices(n_clients, k=1)
    return float(np.mean(np.clip(cosine[upper], -1.0, 1.0)))


def _weighted_rms_column_norm(matrix: FloatArray, weights: FloatArray) -> float:
    """Return ``sqrt(sum_i w_i ||matrix[:, i]||_2^2)`` stably."""

    column_scales = np.max(np.abs(matrix), axis=0)
    scaled_columns = np.divide(
        matrix,
        column_scales[None, :],
        out=np.zeros_like(matrix),
        where=column_scales[None, :] > 0.0,
    )
    column_norms = column_scales * np.linalg.norm(scaled_columns, axis=0)
    norm_scale = float(np.max(column_norms))
    if norm_scale == 0.0:
        return 0.0
    scaled_norms = column_norms / norm_scale
    return float(norm_scale * math.sqrt(float(np.dot(weights, scaled_norms**2))))


def _column_mean_coherence(mean: FloatArray, rms_norm: float) -> float:
    """Return mean-column norm divided by weighted RMS column norm."""

    if rms_norm == 0.0:
        return 0.0
    coherence = float(np.linalg.norm(mean) / rms_norm)
    if coherence > 1.0 and math.isclose(coherence, 1.0, rel_tol=1e-12):
        return 1.0
    return coherence


def _is_bias_key(key: str) -> bool:
    tokens = [token for token in re.split(r"[./_]+", key.lower()) if token]
    return "bias" in tokens


def _looks_like_lora_factor(key: str) -> bool:
    normalised = re.sub(r"[./]+", "_", key.lower())
    return any(
        marker in normalised
        for marker in ("lora_a", "lora_b", "adapter_a", "adapter_b")
    ) and not _is_bias_key(key)


# Discoverable aliases for users who prefer verb-first naming.
aggregate_fedavg = fedavg_aggregate
aggregate_ties = ties_aggregate
aggregate_fedrpca = fedrpca_aggregate


__all__ = [
    "AggregationResult",
    "BlockDiagnostics",
    "CancellationAwareBlockDiagnostics",
    "HsuBalancedRunPlan",
    "PairedRunPlan",
    "RPCAResult",
    "RobustAggregationResult",
    "aggregate_fedavg",
    "aggregate_fedrpca",
    "aggregate_ties",
    "coordinate_median_aggregate",
    "coordinate_trimmed_mean_aggregate",
    "fedavg_aggregate",
    "fedrpca_aggregate",
    "fedrpca_aggregate_blocks",
    "make_hsu_balanced_run_plan",
    "make_paired_run_plan",
    "median_norm_clipped_fedavg_aggregate",
    "principal_component_pursuit_admm",
    "reliability_gated_fedrpca",
    "sample_weighted_fedavg_aggregate",
    "singular_value_threshold",
    "task_arithmetic_aggregate",
    "ties_aggregate",
    "ties_reference_aggregate",
]

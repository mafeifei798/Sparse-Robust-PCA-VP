"""Exact, offline reconstruction primitives for FedRPCA mechanism audits.

The production server intentionally returns only an aggregated update and
serialisable diagnostics.  A counterfactual component audit additionally
needs the actual ``L`` and ``S`` matrices.  This module deterministically
replays the *same* PCP problem from a saved client-delta snapshot, verifies
that ``Lw + beta Sw`` is byte-identical to the production aggregator, and
constructs same-parent intervention states without local retraining.

These functions are diagnostic only.  They never select a beta, inspect test
labels, or install an intervention into a training trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .federated import (
    BlockDiagnostics,
    fedrpca_aggregate_blocks,
    principal_component_pursuit_admm,
)


FloatArray = NDArray[np.float64]

BASE_COMPONENT_INTERVENTIONS = (
    "full_L_plus_beta_S",
    "L_only",
    "beta_equals_1",
)
SPARSE_OVERRIDE_INTERVENTIONS = (
    "blockwise_shared_signed_parameter_permutation_S",
    "norm_matched_random_sparse_S",
)


@dataclass(frozen=True)
class FedRPCAAuditDecomposition:
    """Read-only matrices and exact production lineage for one snapshot."""

    observed_blocks: Mapping[str, FloatArray]
    low_rank_blocks: Mapping[str, FloatArray]
    sparse_blocks: Mapping[str, FloatArray]
    beta_by_block: Mapping[str, float]
    full_update_blocks: Mapping[str, FloatArray]
    diagnostics: Mapping[str, Mapping[str, Any]]
    n_clients: int
    weighting: str
    production_reconstruction_verified: bool


def _readonly(array: ArrayLike) -> FloatArray:
    result = np.array(array, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _byte_equal(left: ArrayLike, right: ArrayLike) -> bool:
    first = np.asarray(left)
    second = np.asarray(right)
    return bool(
        first.shape == second.shape
        and first.dtype == second.dtype
        and np.ascontiguousarray(first).tobytes(order="C")
        == np.ascontiguousarray(second).tobytes(order="C")
    )


def _strict_positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _strict_positive_real(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be real")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validated_observed_blocks(
    observed_blocks: Mapping[str, ArrayLike],
) -> tuple[dict[str, FloatArray], int]:
    if not isinstance(observed_blocks, Mapping):
        raise TypeError("observed_blocks must be a mapping")
    if not observed_blocks:
        raise ValueError("observed_blocks must not be empty")
    if any(not isinstance(name, str) or not name for name in observed_blocks):
        raise TypeError("block names must be non-empty strings")
    if len(set(observed_blocks)) != len(observed_blocks):
        raise ValueError("block names must be unique")
    validated: dict[str, FloatArray] = {}
    n_clients: int | None = None
    for name, value in observed_blocks.items():
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise TypeError(f"observed block {name!r} must be real and non-boolean")
        array = np.asarray(raw, dtype=np.float64)
        if array.ndim != 2 or min(array.shape) <= 0:
            raise ValueError(f"observed block {name!r} must be a non-empty matrix")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"observed block {name!r} contains non-finite values")
        clients = int(array.shape[1])
        if n_clients is None:
            n_clients = clients
        elif clients != n_clients:
            raise ValueError("all observed blocks must share the client dimension")
        validated[name] = _readonly(array)
    assert n_clients is not None
    return validated, n_clients


def _power_of_two_scale(matrix: FloatArray) -> float:
    maximum = float(np.max(np.abs(matrix)))
    if maximum == 0.0:
        return 1.0
    _, exponent = math.frexp(maximum)
    return math.ldexp(1.0, exponent - 1)


def _mean_pairwise_column_cosine(matrix: FloatArray) -> float:
    if matrix.shape[1] < 2:
        return 0.0
    magnitudes = np.max(np.abs(matrix), axis=0)
    scaled = np.divide(
        matrix,
        magnitudes[None, :],
        out=np.zeros_like(matrix),
        where=magnitudes[None, :] > 0.0,
    )
    norms = np.linalg.norm(scaled, axis=0)
    normalised = np.divide(
        scaled,
        norms[None, :],
        out=np.zeros_like(scaled),
        where=norms[None, :] > 0.0,
    )
    gram = normalised.T @ normalised
    indices = np.triu_indices(matrix.shape[1], k=1)
    return float(np.mean(np.clip(gram[indices], -1.0, 1.0)))


def _mapping_proxy_arrays(
    values: Mapping[str, ArrayLike],
) -> Mapping[str, FloatArray]:
    return MappingProxyType({name: _readonly(value) for name, value in values.items()})


def decompose_fedrpca_snapshot(
    observed_blocks: Mapping[str, ArrayLike],
    *,
    rpca_tol: float = 1e-7,
    rpca_max_iter: int = 1_000,
    rpca_secondary_max_iter: int | None = 5_000,
    verify_production_reconstruction: bool = True,
) -> FedRPCAAuditDecomposition:
    """Replay canonical uniform-client FedRPCA and retain ``M``, ``L`` and ``S``.

    The block matrices must have flattened parameter coordinates in rows and
    ordered clients in columns.  The paper inverse-energy beta and the exact
    primary/secondary solver lineage are fixed here; sample-weighted or
    reliability-gated variants are intentionally outside this audit.
    """

    blocks, n_clients = _validated_observed_blocks(observed_blocks)
    tolerance = _strict_positive_real(rpca_tol, name="rpca_tol")
    primary_cap = _strict_positive_integer(rpca_max_iter, name="rpca_max_iter")
    if rpca_secondary_max_iter is None:
        secondary_cap = None
    else:
        secondary_cap = _strict_positive_integer(
            rpca_secondary_max_iter, name="rpca_secondary_max_iter"
        )
        if secondary_cap <= primary_cap:
            raise ValueError("rpca_secondary_max_iter must exceed rpca_max_iter")
    if not isinstance(verify_production_reconstruction, (bool, np.bool_)):
        raise TypeError("verify_production_reconstruction must be boolean")

    weights = np.full(n_clients, 1.0 / n_clients, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps
    low_rank: dict[str, FloatArray] = {}
    sparse: dict[str, FloatArray] = {}
    betas: dict[str, float] = {}
    full_updates: dict[str, FloatArray] = {}
    diagnostics: dict[str, Mapping[str, Any]] = {}

    for name, observed in blocks.items():
        primary = principal_component_pursuit_admm(
            observed,
            tol=tolerance,
            max_iter=primary_cap,
        )
        fallback = bool(secondary_cap is not None and not primary.converged)
        if fallback:
            final = principal_component_pursuit_admm(
                observed,
                tol=tolerance,
                max_iter=int(secondary_cap),
            )
            final_cap = int(secondary_cap)
        else:
            final = primary
            final_cap = primary_cap
        if not final.converged or not math.isfinite(final.residual):
            raise RuntimeError(f"FedRPCA audit block {name!r} did not converge")
        if final.residual > tolerance:
            raise RuntimeError(
                f"FedRPCA audit block {name!r} residual exceeds tolerance"
            )

        scale = _power_of_two_scale(observed)
        observed_mean = (observed / scale) @ weights
        low_mean = (final.low_rank / scale) @ weights
        sparse_mean = (final.sparse / scale) @ weights
        sparse_norm = float(np.linalg.norm(sparse_mean))
        denominator = max(float(np.linalg.norm(observed_mean)), epsilon)
        energy_ratio = float(sparse_norm / denominator)
        beta = 1.0 if sparse_norm <= epsilon else float(1.0 / energy_ratio)
        update = (low_mean + beta * sparse_mean) * scale
        if not np.all(np.isfinite(update)):
            raise FloatingPointError(
                f"FedRPCA audit update for block {name!r} is non-finite"
            )
        diagnostic = BlockDiagnostics(
            residual=float(final.residual),
            rank=int(final.rank),
            sparsity=float(final.sparsity),
            observed_cosine=_mean_pairwise_column_cosine(observed),
            low_rank_cosine=_mean_pairwise_column_cosine(final.low_rank),
            sparse_cosine=_mean_pairwise_column_cosine(final.sparse),
            cosine_gap=(
                _mean_pairwise_column_cosine(final.low_rank)
                - _mean_pairwise_column_cosine(final.sparse)
            ),
            energy_ratio=energy_ratio,
            beta_policy="paper_inverse_energy",
            beta_base=beta,
            gate=1.0,
            beta=beta,
            iterations=int(final.iterations),
            converged=bool(final.converged),
            primary_max_iter=primary_cap,
            primary_residual=float(primary.residual),
            primary_converged=bool(primary.converged),
            fallback_triggered=fallback,
            final_max_iter=final_cap,
        )
        low_rank[name] = _readonly(final.low_rank)
        sparse[name] = _readonly(final.sparse)
        betas[name] = beta
        full_updates[name] = _readonly(update)
        diagnostics[name] = MappingProxyType(asdict(diagnostic))

    verified = False
    if bool(verify_production_reconstruction):
        production_blocks = {
            name: tuple(blocks[name][:, client] for client in range(n_clients))
            for name in blocks
        }
        production = fedrpca_aggregate_blocks(
            production_blocks,
            adaptive_beta=True,
            beta_max=None,
            sample_counts=None,
            reliability_gated=False,
            rpca_tol=tolerance,
            rpca_max_iter=primary_cap,
            rpca_secondary_max_iter=secondary_cap,
        )
        for name in blocks:
            if not _byte_equal(production.update[name], full_updates[name]):
                raise RuntimeError(
                    f"audit reconstruction differs from production for {name!r}"
                )
            if asdict(production.diagnostics[name]) != dict(diagnostics[name]):
                raise RuntimeError(
                    f"audit diagnostics differ from production for {name!r}"
                )
        verified = True

    return FedRPCAAuditDecomposition(
        observed_blocks=_mapping_proxy_arrays(blocks),
        low_rank_blocks=_mapping_proxy_arrays(low_rank),
        sparse_blocks=_mapping_proxy_arrays(sparse),
        beta_by_block=MappingProxyType(dict(betas)),
        full_update_blocks=_mapping_proxy_arrays(full_updates),
        diagnostics=MappingProxyType(dict(diagnostics)),
        n_clients=n_clients,
        weighting="uniform_clients",
        production_reconstruction_verified=verified,
    )


def compose_component_intervention_updates(
    decomposition: FedRPCAAuditDecomposition,
    intervention: str,
    *,
    sparse_override: Mapping[str, ArrayLike] | None = None,
) -> dict[str, FloatArray]:
    """Compose flat adapter updates from one fixed decomposition.

    ``sparse_override`` is required only for a preregistered permutation or
    random-sparse draw.  It must retain the exact block shapes and client
    order.  The original per-block beta is always reused.
    """

    if not isinstance(decomposition, FedRPCAAuditDecomposition):
        raise TypeError("decomposition must be a FedRPCAAuditDecomposition")
    if intervention not in (
        *BASE_COMPONENT_INTERVENTIONS,
        *SPARSE_OVERRIDE_INTERVENTIONS,
    ):
        raise ValueError("unknown component intervention")
    if intervention in SPARSE_OVERRIDE_INTERVENTIONS and sparse_override is None:
        raise ValueError("a non-baseline intervention requires sparse_override")
    if intervention in BASE_COMPONENT_INTERVENTIONS and sparse_override is not None:
        raise ValueError("baseline component interventions reject sparse_override")
    if sparse_override is None:
        override: Mapping[str, FloatArray] | None = None
    else:
        override_values, clients = _validated_observed_blocks(sparse_override)
        if clients != decomposition.n_clients:
            raise ValueError("sparse_override client count changed")
        if set(override_values) != set(decomposition.sparse_blocks):
            raise ValueError("sparse_override block registry changed")
        for name, value in override_values.items():
            if value.shape != decomposition.sparse_blocks[name].shape:
                raise ValueError(f"sparse_override shape changed for {name!r}")
        override = override_values

    weights = np.full(
        decomposition.n_clients,
        1.0 / decomposition.n_clients,
        dtype=np.float64,
    )
    result: dict[str, FloatArray] = {}
    for name in decomposition.low_rank_blocks:
        observed = decomposition.observed_blocks[name]
        scale = _power_of_two_scale(observed)
        low_mean = (decomposition.low_rank_blocks[name] / scale) @ weights
        if intervention == "L_only":
            sparse_coefficient = 0.0
            sparse_matrix = decomposition.sparse_blocks[name]
        elif intervention == "beta_equals_1":
            sparse_coefficient = 1.0
            sparse_matrix = decomposition.sparse_blocks[name]
        elif intervention == "full_L_plus_beta_S":
            sparse_coefficient = decomposition.beta_by_block[name]
            sparse_matrix = decomposition.sparse_blocks[name]
        else:
            assert override is not None
            sparse_coefficient = decomposition.beta_by_block[name]
            sparse_matrix = override[name]
        sparse_mean = (sparse_matrix / scale) @ weights
        update = (low_mean + sparse_coefficient * sparse_mean) * scale
        if not np.all(np.isfinite(update)):
            raise FloatingPointError(
                f"component intervention update for {name!r} is non-finite"
            )
        result[name] = _readonly(update)
    return result


def compose_same_parent_state(
    parent_state: Mapping[str, ArrayLike],
    *,
    parameter_shapes: Mapping[str, Sequence[int]],
    adapter_updates: Mapping[str, ArrayLike],
    passthrough_server_values: Mapping[str, ArrayLike],
) -> dict[str, FloatArray]:
    """Apply adapter updates and attach exact saved classifier/head values.

    The two registries must exactly partition the parent state.  Passthrough
    values are server values, not deltas: recomputing ``parent +
    (server-parent)`` is not guaranteed to round-trip byte-for-byte in
    floating-point arithmetic.  Reusing the saved production value both keeps
    the classifier/head identical across interventions and permits an exact
    whole-state comparison with the production snapshot.
    """

    if not isinstance(parent_state, Mapping) or not parent_state:
        raise ValueError("parent_state must be a non-empty mapping")
    parent = {name: _readonly(value) for name, value in parent_state.items()}
    if not isinstance(parameter_shapes, Mapping):
        raise TypeError("parameter_shapes must be a mapping")
    adapter_names = tuple(adapter_updates)
    passthrough_names = tuple(passthrough_server_values)
    if set(adapter_names) & set(passthrough_names):
        raise ValueError("adapter and passthrough update registries overlap")
    if set(adapter_names) | set(passthrough_names) != set(parent):
        raise ValueError("updates do not exactly partition the parent state")
    if set(parameter_shapes) != set(adapter_names):
        raise ValueError("parameter_shapes must exactly cover adapter updates")

    result: dict[str, FloatArray] = {}
    for name in adapter_names:
        shape_values = parameter_shapes[name]
        if isinstance(shape_values, (str, bytes)) or not isinstance(
            shape_values, Sequence
        ):
            raise TypeError(f"parameter shape for {name!r} must be a sequence")
        shape: list[int] = []
        for dimension in shape_values:
            shape.append(_strict_positive_integer(dimension, name="shape dimension"))
        if tuple(shape) != parent[name].shape:
            raise ValueError(f"parameter shape disagrees with parent for {name!r}")
        flat = np.asarray(adapter_updates[name], dtype=np.float64)
        if flat.ndim != 1 or flat.size != int(np.prod(shape)):
            raise ValueError(f"adapter update has the wrong shape for {name!r}")
        result[name] = _readonly(parent[name] + flat.reshape(tuple(shape)))
    for name in passthrough_names:
        value = np.asarray(passthrough_server_values[name], dtype=np.float64)
        if value.shape != parent[name].shape or not np.all(np.isfinite(value)):
            raise ValueError(f"passthrough server value is invalid for {name!r}")
        result[name] = _readonly(value)
    return result


__all__ = [
    "BASE_COMPONENT_INTERVENTIONS",
    "SPARSE_OVERRIDE_INTERVENTIONS",
    "FedRPCAAuditDecomposition",
    "compose_component_intervention_updates",
    "compose_same_parent_state",
    "decompose_fedrpca_snapshot",
]

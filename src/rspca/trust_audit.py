"""Dependency-light metrics for causal trust audits of FedRPCA components.

The functions in this module are deliberately model- and dataset-agnostic.
They validate matched experimental records and compute descriptive quantities;
they do not select thresholds, tune interventions, or inspect test labels.

Undefined ranking quantities are represented by ``None`` rather than ``NaN``.
In particular, AUROC requires both a positive and a negative example, while
average precision (reported as AUPRC) requires at least one positive example.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

REQUIRED_COMPONENT_INTERVENTIONS = (
    "full_L_plus_beta_S",
    "L_only",
    "beta_equals_1",
    "blockwise_shared_signed_parameter_permutation_S",
    "norm_matched_random_sparse_S",
)

COMPONENT_CONTROL_DRAWS = 32
DEFAULT_COMPONENT_CONTROL_SEED = 20261202
DEFAULT_COMPONENT_INVARIANT_TOLERANCE = 1e-12

_BASELINE_COMPONENT_INTERVENTIONS = frozenset(REQUIRED_COMPONENT_INTERVENTIONS[:3])
_DRAW_COMPONENT_INTERVENTIONS = frozenset(REQUIRED_COMPONENT_INTERVENTIONS[3:])


def _strict_numeric_array(
    value: ArrayLike,
    *,
    name: str,
    min_ndim: int = 1,
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real, non-boolean numeric values")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim < min_ndim:
        raise ValueError(f"{name} must have at least {min_ndim} dimension(s)")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def _strict_binary_mask(value: ArrayLike, *, name: str) -> BoolArray:
    raw = np.asarray(value)
    if raw.size == 0:
        raise ValueError(f"{name} must not be empty")
    if raw.dtype.kind == "b":
        return np.asarray(raw, dtype=bool)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must be boolean or an integer 0/1 array")
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} must contain only 0 and 1")
    return np.asarray(raw, dtype=bool)


def _strict_positive_integer(value: int, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_finite_real(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _readonly_copy(array: FloatArray) -> FloatArray:
    copied = np.array(array, dtype=np.float64, copy=True)
    copied.setflags(write=False)
    return copied


def _strict_uint64(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 0 or result > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must lie in the uint64 range")
    return result


def _validate_control_settings(
    *, n_draws: int, seed: int, invariant_tolerance: float
) -> tuple[int, int, float]:
    draws = _strict_positive_integer(n_draws, name="n_draws")
    if draws != COMPONENT_CONTROL_DRAWS:
        raise ValueError(
            f"n_draws must equal the preregistered value {COMPONENT_CONTROL_DRAWS}"
        )
    random_seed = _strict_uint64(seed, name="seed")
    tolerance = _strict_finite_real(invariant_tolerance, name="invariant_tolerance")
    if tolerance < 0.0:
        raise ValueError("invariant_tolerance must be non-negative")
    return draws, random_seed, tolerance


def _validated_sparse_blocks(
    sparse_blocks: Mapping[str, ArrayLike],
) -> tuple[dict[str, FloatArray], int]:
    if not isinstance(sparse_blocks, Mapping):
        raise TypeError("sparse_blocks must be a mapping from block names to arrays")
    if not sparse_blocks:
        raise ValueError("sparse_blocks must not be empty")
    if any(not isinstance(name, str) or not name.strip() for name in sparse_blocks):
        raise TypeError("sparse block names must be non-empty strings")
    validated: dict[str, FloatArray] = {}
    n_clients: int | None = None
    for block_name in sorted(sparse_blocks):
        block = _strict_numeric_array(
            sparse_blocks[block_name], name=f"sparse block {block_name!r}", min_ndim=2
        )
        clients = int(block.shape[-1])
        if n_clients is None:
            n_clients = clients
        elif clients != n_clients:
            raise ValueError(
                "all sparse blocks must have the same final client dimension; "
                f"expected {n_clients}, got {clients} for {block_name!r}"
            )
        validated[block_name] = _readonly_copy(block)
    assert n_clients is not None
    return validated, n_clients


def _derive_control_seed(
    master_seed: int,
    *,
    namespace: str,
    draw_index: int,
    block_name: str = "",
    attempt: int = 0,
) -> int:
    payload = json.dumps(
        {
            "schema": "rspca-component-control-seed-v1",
            "master_seed": master_seed,
            "namespace": namespace,
            "draw_index": draw_index,
            "block_name": block_name,
            "attempt": attempt,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _stable_vector_norm(values: FloatArray) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    scale = float(np.max(np.abs(flat))) if flat.size else 0.0
    if scale == 0.0:
        return 0.0
    scaled_norm = float(np.linalg.norm(flat / scale))
    norm = scale * scaled_norm
    if not math.isfinite(norm):
        raise ValueError("control invariant norm is non-finite")
    return norm


def _column_norms(matrix: FloatArray) -> FloatArray:
    result = np.empty(matrix.shape[1], dtype=np.float64)
    for client_index in range(matrix.shape[1]):
        result[client_index] = _stable_vector_norm(matrix[:, client_index])
    return result


def _client_cosine_gram(matrix: FloatArray) -> FloatArray:
    norms = _column_norms(matrix)
    unit = np.zeros_like(matrix, dtype=np.float64)
    nonzero = norms > 0.0
    unit[:, nonzero] = matrix[:, nonzero] / norms[nonzero]
    gram = unit.T @ unit
    if not np.all(np.isfinite(gram)):
        raise ValueError("client cosine Gram matrix is non-finite")
    return gram


def _safe_mean_vector(matrix: FloatArray) -> FloatArray:
    # Dividing first avoids overflow when several same-signed finite values are
    # close to float64's maximum.
    mean = np.sum(matrix / matrix.shape[1], axis=1, dtype=np.float64)
    if not np.all(np.isfinite(mean)):
        raise ValueError("mean sparse update is non-finite")
    return mean


def _relative_scalar_error(reference: float, candidate: float) -> float:
    if reference == candidate:
        return 0.0
    denominator = max(abs(reference), np.finfo(np.float64).tiny)
    error = abs(candidate - reference) / denominator
    if not math.isfinite(error):
        raise ValueError("control invariant relative error is non-finite")
    return float(error)


def _max_relative_array_error(reference: FloatArray, candidate: FloatArray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError("control invariant arrays have different shapes")
    errors = np.empty(reference.size, dtype=np.float64)
    for index, (left, right) in enumerate(
        zip(reference.reshape(-1), candidate.reshape(-1), strict=True)
    ):
        errors[index] = _relative_scalar_error(float(left), float(right))
    return float(np.max(errors)) if errors.size else 0.0


def _digest_array(hasher: Any, *, block_name: str, array: FloatArray) -> None:
    name_bytes = block_name.encode("utf-8")
    hasher.update(len(name_bytes).to_bytes(8, "big"))
    hasher.update(name_bytes)
    hasher.update(len(array.shape).to_bytes(8, "big"))
    for dimension in array.shape:
        hasher.update(int(dimension).to_bytes(8, "big"))
    canonical = np.ascontiguousarray(array, dtype="<f8")
    hasher.update(canonical.tobytes(order="C"))


def _control_digest(
    *,
    intervention: str,
    draw_index: int,
    draw_seed: int,
    sparse_blocks: Mapping[str, FloatArray],
    block_metadata: Mapping[str, Mapping[str, Any]],
) -> str:
    hasher = hashlib.sha256()
    header = json.dumps(
        {
            "schema": "rspca-component-control-draw-v1",
            "intervention": intervention,
            "draw_index": draw_index,
            "draw_seed": draw_seed,
            "block_metadata": block_metadata,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    hasher.update(header)
    for block_name in sorted(sparse_blocks):
        _digest_array(hasher, block_name=block_name, array=sparse_blocks[block_name])
    return hasher.hexdigest()


def _signed_invariants(
    source: FloatArray,
    candidate: FloatArray,
) -> dict[str, Any]:
    source_support = np.count_nonzero(source, axis=0)
    candidate_support = np.count_nonzero(candidate, axis=0)
    source_norms = _column_norms(source)
    candidate_norms = _column_norms(candidate)
    source_gram = _client_cosine_gram(source)
    candidate_gram = _client_cosine_gram(candidate)
    source_mean_norm = _stable_vector_norm(_safe_mean_vector(source))
    candidate_mean_norm = _stable_vector_norm(_safe_mean_vector(candidate))
    return {
        "support_counts_preserved": bool(
            np.array_equal(source_support, candidate_support)
        ),
        "per_client_support_counts": source_support.astype(int).tolist(),
        "max_per_client_norm_relative_error": _max_relative_array_error(
            source_norms, candidate_norms
        ),
        "max_client_cosine_gram_absolute_error": float(
            np.max(np.abs(source_gram - candidate_gram))
        ),
        "mean_sparse_norm_relative_error": _relative_scalar_error(
            source_mean_norm, candidate_mean_norm
        ),
    }


def _random_sparse_invariants(
    source: FloatArray,
    candidate: FloatArray,
) -> dict[str, Any]:
    source_support = np.count_nonzero(source, axis=0)
    candidate_support = np.count_nonzero(candidate, axis=0)
    return {
        "support_counts_preserved": bool(
            np.array_equal(source_support, candidate_support)
        ),
        "per_client_support_counts": source_support.astype(int).tolist(),
        "max_per_client_norm_relative_error": _max_relative_array_error(
            _column_norms(source), _column_norms(candidate)
        ),
    }


def _concatenated_block_means(
    blocks: Mapping[str, FloatArray],
) -> FloatArray:
    return np.concatenate(
        [
            _safe_mean_vector(blocks[name].reshape(-1, blocks[name].shape[-1]))
            for name in blocks
        ]
    )


def _global_change_diagnostics(
    source_blocks: Mapping[str, FloatArray],
    candidate_blocks: Mapping[str, FloatArray],
) -> tuple[bool, bool, float]:
    changed = any(
        not np.array_equal(source_blocks[name], candidate_blocks[name])
        for name in source_blocks
    )
    source_means = _concatenated_block_means(source_blocks)
    candidate_means = _concatenated_block_means(candidate_blocks)
    mean_changed = not np.array_equal(source_means, candidate_means)
    mean_change_norm = _stable_vector_norm(candidate_means - source_means)
    return changed, mean_changed, mean_change_norm


def _global_mean_norm_relative_error(
    source_blocks: Mapping[str, FloatArray],
    candidate_blocks: Mapping[str, FloatArray],
) -> float:
    return _relative_scalar_error(
        _stable_vector_norm(_concatenated_block_means(source_blocks)),
        _stable_vector_norm(_concatenated_block_means(candidate_blocks)),
    )


def blockwise_shared_signed_parameter_permutation_controls(
    sparse_blocks: Mapping[str, ArrayLike],
    *,
    seed: int = DEFAULT_COMPONENT_CONTROL_SEED,
    n_draws: int = COMPONENT_CONTROL_DRAWS,
    invariant_tolerance: float = DEFAULT_COMPONENT_INVARIANT_TOLERANCE,
) -> tuple[dict[str, Any], ...]:
    """Create the preregistered block-local signed-permutation controls.

    Within block ``b``, parameter coordinates are flattened and a single
    non-identity signed permutation ``P_b`` is applied to every client column.
    Blocks never exchange coordinates.  Each returned draw contains read-only
    arrays plus JSON-serializable seeds, operators, digests, and independently
    recomputed invariant diagnostics.

    The procedure is deliberately fail-closed.  It raises an ``ineligible``
    error if 32 deterministic draws cannot both alter ``S`` and alter the mean
    sparse update.  In particular, a globally zero mean sparse update is
    mathematically invariant under every signed permutation.
    """

    draws, master_seed, tolerance = _validate_control_settings(
        n_draws=n_draws, seed=seed, invariant_tolerance=invariant_tolerance
    )
    blocks, _ = _validated_sparse_blocks(sparse_blocks)
    flattened = {
        name: block.reshape(-1, block.shape[-1]) for name, block in blocks.items()
    }
    concatenated_mean = np.concatenate(
        [_safe_mean_vector(matrix) for matrix in flattened.values()]
    )
    if not np.any(concatenated_mean != 0.0):
        raise ValueError(
            "signed-permutation control is ineligible: the complete mean sparse "
            "update is zero and cannot be changed by an orthogonal signed permutation"
        )

    results: list[dict[str, Any]] = []
    for draw_index in range(draws):
        draw_seed = _derive_control_seed(
            master_seed,
            namespace="blockwise_shared_signed_parameter_permutation_S",
            draw_index=draw_index,
        )
        accepted: dict[str, FloatArray] | None = None
        accepted_metadata: dict[str, dict[str, Any]] | None = None
        accepted_diagnostics: dict[str, dict[str, Any]] | None = None
        for attempt in range(4_096):
            candidate_blocks: dict[str, FloatArray] = {}
            block_metadata: dict[str, dict[str, Any]] = {}
            block_diagnostics: dict[str, dict[str, Any]] = {}
            invariant_ok = True
            for block_name, source in flattened.items():
                block_seed = _derive_control_seed(
                    draw_seed,
                    namespace="signed-permutation-block",
                    draw_index=draw_index,
                    block_name=block_name,
                    attempt=attempt,
                )
                rng = np.random.default_rng(block_seed)
                n_parameters = source.shape[0]
                permutation = np.asarray(rng.permutation(n_parameters), dtype=np.int64)
                signs = np.asarray(
                    rng.choice(np.array([-1.0, 1.0]), size=n_parameters),
                    dtype=np.float64,
                )
                identity = np.arange(n_parameters, dtype=np.int64)
                if np.array_equal(permutation, identity) and np.all(signs == 1.0):
                    signs[int(rng.integers(0, n_parameters))] = -1.0
                transformed_matrix = signs[:, None] * source[permutation, :]
                transformed = transformed_matrix.reshape(blocks[block_name].shape)
                candidate_blocks[block_name] = _readonly_copy(transformed)
                diagnostics = _signed_invariants(source, transformed_matrix)
                block_diagnostics[block_name] = diagnostics
                operator_payload = {
                    "block_seed": block_seed,
                    "permutation": permutation.astype(int).tolist(),
                    "signs": signs.astype(int).tolist(),
                }
                operator_digest = hashlib.sha256(
                    json.dumps(
                        operator_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                block_metadata[block_name] = {
                    **operator_payload,
                    "operator_digest": operator_digest,
                    "nonidentity_operator": bool(
                        not np.array_equal(permutation, identity)
                        or np.any(signs != 1.0)
                    ),
                }
                invariant_ok = invariant_ok and bool(
                    diagnostics["support_counts_preserved"]
                    and diagnostics["max_per_client_norm_relative_error"] <= tolerance
                    and diagnostics["max_client_cosine_gram_absolute_error"]
                    <= tolerance
                    and diagnostics["mean_sparse_norm_relative_error"] <= tolerance
                    and block_metadata[block_name]["nonidentity_operator"]
                )
            changed, mean_changed, _ = _global_change_diagnostics(
                blocks, candidate_blocks
            )
            global_mean_norm_error = _global_mean_norm_relative_error(
                blocks, candidate_blocks
            )
            if (
                invariant_ok
                and global_mean_norm_error <= tolerance
                and changed
                and mean_changed
            ):
                accepted = candidate_blocks
                accepted_metadata = block_metadata
                accepted_diagnostics = block_diagnostics
                break
        if (
            accepted is None
            or accepted_metadata is None
            or accepted_diagnostics is None
        ):
            raise ValueError(
                "signed-permutation control is ineligible: could not construct a "
                f"changed invariant-preserving draw at draw_index={draw_index}"
            )

        changed, mean_changed, mean_change_norm = _global_change_diagnostics(
            blocks, accepted
        )
        global_mean_norm_error = _global_mean_norm_relative_error(blocks, accepted)
        summary = {
            "eligible": True,
            "support_counts_preserved": all(
                item["support_counts_preserved"]
                for item in accepted_diagnostics.values()
            ),
            "max_per_client_norm_relative_error": max(
                item["max_per_client_norm_relative_error"]
                for item in accepted_diagnostics.values()
            ),
            "max_client_cosine_gram_absolute_error": max(
                item["max_client_cosine_gram_absolute_error"]
                for item in accepted_diagnostics.values()
            ),
            "max_mean_sparse_norm_relative_error": max(
                item["mean_sparse_norm_relative_error"]
                for item in accepted_diagnostics.values()
            ),
            "global_mean_sparse_norm_relative_error": global_mean_norm_error,
            "sparse_component_changed": changed,
            "mean_update_changed": mean_changed,
            "mean_update_change_norm": mean_change_norm,
            "maximum_allowed_error": tolerance,
            "per_block": accepted_diagnostics,
        }
        digest = _control_digest(
            intervention="blockwise_shared_signed_parameter_permutation_S",
            draw_index=draw_index,
            draw_seed=draw_seed,
            sparse_blocks=accepted,
            block_metadata=accepted_metadata,
        )
        results.append(
            {
                "intervention": "blockwise_shared_signed_parameter_permutation_S",
                "draw_index": draw_index,
                "seed": master_seed,
                "draw_seed": draw_seed,
                "digest": digest,
                "sparse_blocks": accepted,
                "block_metadata": accepted_metadata,
                "invariants": summary,
            }
        )
    if len({result["digest"] for result in results}) != draws:
        raise ValueError("signed-permutation control produced duplicate draw digests")
    return tuple(results)


def per_client_norm_support_matched_random_sparse_controls(
    sparse_blocks: Mapping[str, ArrayLike],
    *,
    seed: int = DEFAULT_COMPONENT_CONTROL_SEED,
    n_draws: int = COMPONENT_CONTROL_DRAWS,
    invariant_tolerance: float = DEFAULT_COMPONENT_INVARIANT_TOLERANCE,
) -> tuple[dict[str, Any], ...]:
    """Create 32 deterministic random sparse controls matched per client.

    Every block/client column retains its exact nonzero count and Euclidean
    norm.  Supports and directions are redrawn independently from seeds derived
    from ``seed``.  Draw arrays are read-only and all accompanying metadata is
    JSON serializable.
    """

    draws, master_seed, tolerance = _validate_control_settings(
        n_draws=n_draws, seed=seed, invariant_tolerance=invariant_tolerance
    )
    blocks, n_clients = _validated_sparse_blocks(sparse_blocks)
    flattened = {
        name: block.reshape(-1, block.shape[-1]) for name, block in blocks.items()
    }
    if not any(np.any(matrix != 0.0) for matrix in flattened.values()):
        raise ValueError(
            "random sparse control is ineligible: an all-zero sparse component "
            "cannot change while matching every client norm and support count"
        )

    results: list[dict[str, Any]] = []
    for draw_index in range(draws):
        draw_seed = _derive_control_seed(
            master_seed,
            namespace="norm_matched_random_sparse_S",
            draw_index=draw_index,
        )
        accepted: dict[str, FloatArray] | None = None
        accepted_metadata: dict[str, dict[str, Any]] | None = None
        accepted_diagnostics: dict[str, dict[str, Any]] | None = None
        for attempt in range(4_096):
            candidate_blocks: dict[str, FloatArray] = {}
            block_metadata: dict[str, dict[str, Any]] = {}
            block_diagnostics: dict[str, dict[str, Any]] = {}
            invariant_ok = True
            for block_name, source in flattened.items():
                block_seed = _derive_control_seed(
                    draw_seed,
                    namespace="random-sparse-block",
                    draw_index=draw_index,
                    block_name=block_name,
                    attempt=attempt,
                )
                rng = np.random.default_rng(block_seed)
                candidate = np.zeros_like(source, dtype=np.float64)
                support_indices: list[list[int]] = []
                target_norms = _column_norms(source)
                for client_index in range(n_clients):
                    support_count = int(np.count_nonzero(source[:, client_index]))
                    if support_count == 0:
                        support_indices.append([])
                        continue
                    support = np.asarray(
                        rng.choice(source.shape[0], size=support_count, replace=False),
                        dtype=np.int64,
                    )
                    values = np.asarray(
                        rng.standard_normal(support_count), dtype=np.float64
                    )
                    value_norm = _stable_vector_norm(values)
                    if value_norm == 0.0:
                        invariant_ok = False
                        break
                    values = values / value_norm * target_norms[client_index]
                    candidate[support, client_index] = values
                    support_indices.append(support.astype(int).tolist())
                if not invariant_ok:
                    break
                transformed = candidate.reshape(blocks[block_name].shape)
                candidate_blocks[block_name] = _readonly_copy(transformed)
                diagnostics = _random_sparse_invariants(source, candidate)
                block_diagnostics[block_name] = diagnostics
                realization_payload = {
                    "block_seed": block_seed,
                    "support_indices_by_client": support_indices,
                }
                realization_digest = hashlib.sha256(
                    json.dumps(
                        realization_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                block_metadata[block_name] = {
                    **realization_payload,
                    "realization_digest": realization_digest,
                }
                invariant_ok = invariant_ok and bool(
                    diagnostics["support_counts_preserved"]
                    and diagnostics["max_per_client_norm_relative_error"] <= tolerance
                )
            if not invariant_ok or len(candidate_blocks) != len(blocks):
                continue
            changed, mean_changed, _ = _global_change_diagnostics(
                blocks, candidate_blocks
            )
            if changed and mean_changed:
                accepted = candidate_blocks
                accepted_metadata = block_metadata
                accepted_diagnostics = block_diagnostics
                break
        if (
            accepted is None
            or accepted_metadata is None
            or accepted_diagnostics is None
        ):
            raise ValueError(
                "random sparse control is ineligible: could not construct a changed "
                f"norm/support-matched draw at draw_index={draw_index}"
            )

        changed, mean_changed, mean_change_norm = _global_change_diagnostics(
            blocks, accepted
        )
        summary = {
            "eligible": True,
            "support_counts_preserved": all(
                item["support_counts_preserved"]
                for item in accepted_diagnostics.values()
            ),
            "max_per_client_norm_relative_error": max(
                item["max_per_client_norm_relative_error"]
                for item in accepted_diagnostics.values()
            ),
            "sparse_component_changed": changed,
            "mean_update_changed": mean_changed,
            "mean_update_change_norm": mean_change_norm,
            "maximum_allowed_error": tolerance,
            "per_block": accepted_diagnostics,
        }
        digest = _control_digest(
            intervention="norm_matched_random_sparse_S",
            draw_index=draw_index,
            draw_seed=draw_seed,
            sparse_blocks=accepted,
            block_metadata=accepted_metadata,
        )
        results.append(
            {
                "intervention": "norm_matched_random_sparse_S",
                "draw_index": draw_index,
                "seed": master_seed,
                "draw_seed": draw_seed,
                "digest": digest,
                "sparse_blocks": accepted,
                "block_metadata": accepted_metadata,
                "invariants": summary,
            }
        )
    if len({result["digest"] for result in results}) != draws:
        raise ValueError("random sparse control produced duplicate draw digests")
    return tuple(results)


def _normalise_axis(axis: int, ndim: int, *, name: str) -> int:
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"{name} is out of bounds for an array with {ndim} dimensions")
    return normalized


def validate_matched_client_conditions(
    conditions: Mapping[str, ArrayLike],
    *,
    client_axis: int = 0,
    min_clients: int = 1,
) -> dict[str, FloatArray]:
    """Validate arrays measured on the same ordered clients.

    Every condition must have the same complete shape.  ``client_axis`` only
    identifies which dimension is counted as clients; this function cannot
    infer or repair client ordering, so callers must establish that provenance
    before calling it.  Defensive, read-only copies are returned.
    """

    if not isinstance(conditions, Mapping):
        raise TypeError("conditions must be a mapping from names to arrays")
    if len(conditions) < 2:
        raise ValueError("at least two matched conditions are required")
    minimum = _strict_positive_integer(min_clients, name="min_clients")

    validated: dict[str, FloatArray] = {}
    expected_shape: tuple[int, ...] | None = None
    normalized_axis: int | None = None
    for name, value in conditions.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("condition names must be non-empty strings")
        array = _strict_numeric_array(value, name=f"condition {name!r}")
        if expected_shape is None:
            expected_shape = array.shape
            normalized_axis = _normalise_axis(
                client_axis, array.ndim, name="client_axis"
            )
            if array.shape[normalized_axis] < minimum:
                raise ValueError(
                    f"conditions contain fewer than min_clients={minimum} clients"
                )
        elif array.shape != expected_shape:
            raise ValueError(
                "matched condition arrays must have the same shape; got "
                f"{expected_shape} and {array.shape} for {name!r}"
            )
        validated[name] = _readonly_copy(array)
    return validated


def client_sparse_energy(
    sparse_component: ArrayLike,
    *,
    client_axis: int = -1,
    normalize: bool = False,
) -> FloatArray:
    """Return squared sparse-component energy for each client.

    FedRPCA represents clients as matrix columns, hence the default final
    client axis.  All remaining dimensions are reduced.  With ``normalize``,
    energies sum to one unless the sparse component is identically zero, in
    which case the returned vector is all zeros.
    """

    sparse = _strict_numeric_array(sparse_component, name="sparse_component")
    axis = _normalise_axis(client_axis, sparse.ndim, name="client_axis")
    if not isinstance(normalize, (bool, np.bool_)):
        raise TypeError("normalize must be a boolean")
    reduce_axes = tuple(index for index in range(sparse.ndim) if index != axis)
    with np.errstate(over="ignore", invalid="ignore"):
        energy = np.sum(np.square(sparse), axis=reduce_axes, dtype=np.float64)
    energy = np.asarray(energy, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(energy)):
        raise ValueError("sparse-component energy overflowed to a non-finite value")
    if bool(normalize):
        total = float(np.sum(energy, dtype=np.float64))
        if not math.isfinite(total):
            raise ValueError("total sparse-component energy is non-finite")
        if total > 0.0:
            energy = energy / total
    return _readonly_copy(energy)


def _rank_auroc(truth: BoolArray, scores: FloatArray) -> float | None:
    positives = int(np.count_nonzero(truth))
    negatives = int(truth.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(np.sum(ranks[truth], dtype=np.float64))
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(np.clip(auc, 0.0, 1.0))


def _average_precision(truth: BoolArray, scores: FloatArray) -> float | None:
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    cumulative_positives = 0
    average_precision = 0.0
    start = 0
    while start < scores.size:
        stop = start + 1
        while stop < scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        positives_at_threshold = int(np.count_nonzero(sorted_truth[start:stop]))
        cumulative_positives += positives_at_threshold
        average_precision += (positives_at_threshold / positives) * (
            cumulative_positives / stop
        )
        start = stop
    return float(np.clip(average_precision, 0.0, 1.0))


def client_sparse_energy_fault_localization(
    sparse_component: ArrayLike,
    faulty_clients: ArrayLike,
    *,
    client_axis: int = -1,
) -> dict[str, Any]:
    """Rank planted faulty clients by their recovered sparse energy."""

    energies = client_sparse_energy(
        sparse_component, client_axis=client_axis, normalize=False
    )
    faults = _strict_binary_mask(faulty_clients, name="faulty_clients")
    if faults.ndim != 1:
        raise ValueError("faulty_clients must be one-dimensional")
    if faults.shape != energies.shape:
        raise ValueError(
            "faulty_clients must have one entry per client; got "
            f"{faults.shape} and {energies.shape}"
        )
    faulty_count = int(np.count_nonzero(faults))
    return {
        "n_clients": int(faults.size),
        "n_faulty": faulty_count,
        "n_clean": int(faults.size - faulty_count),
        "roc_auc": _rank_auroc(faults, energies),
        "auprc": _average_precision(faults, energies),
        "client_sparse_energy": energies.astype(float).tolist(),
    }


def planted_support_recovery_metrics(
    planted_support: ArrayLike,
    recovered_sparse: ArrayLike,
    *,
    threshold: float = 1e-10,
) -> dict[str, float | int | None]:
    """Evaluate planted entrywise support without choosing a threshold.

    ``threshold`` must be fixed independently of ``planted_support``.  AUPRC
    uses ``abs(recovered_sparse)`` as a continuous ranking score.  For an
    empty planted support, precision/recall/F1 follow the standard empty-set
    convention, while AUPRC is undefined and therefore returned as ``None``.
    """

    cutoff = _strict_finite_real(threshold, name="threshold")
    if cutoff < 0.0:
        raise ValueError("threshold must be non-negative")
    truth = _strict_binary_mask(planted_support, name="planted_support")
    scores = _strict_numeric_array(recovered_sparse, name="recovered_sparse")
    if truth.shape != scores.shape:
        raise ValueError(
            "planted_support and recovered_sparse must have the same shape; got "
            f"{truth.shape} and {scores.shape}"
        )
    scores = np.abs(scores).reshape(-1)
    truth = truth.reshape(-1)
    estimate = scores > cutoff

    true_positives = int(np.count_nonzero(truth & estimate))
    false_positives = int(np.count_nonzero(~truth & estimate))
    false_negatives = int(np.count_nonzero(truth & ~estimate))
    true_negatives = int(np.count_nonzero(~truth & ~estimate))
    predicted_count = true_positives + false_positives
    planted_count = true_positives + false_negatives
    precision = (
        true_positives / predicted_count
        if predicted_count
        else (1.0 if planted_count == 0 else 0.0)
    )
    recall = true_positives / planted_count if planted_count else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auprc": _average_precision(truth, scores),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_negatives": true_negatives,
    }


def normalized_direction_alignment(
    reference: ArrayLike,
    candidate: ArrayLike,
    *,
    zero_tolerance: float = 0.0,
) -> float:
    """Return cosine alignment after flattening and normalizing two directions.

    A zero or near-zero direction has no mathematically defined orientation and
    is rejected rather than assigned a misleading alignment.
    """

    tolerance = _strict_finite_real(zero_tolerance, name="zero_tolerance")
    if tolerance < 0.0:
        raise ValueError("zero_tolerance must be non-negative")
    left_array = _strict_numeric_array(reference, name="reference")
    right_array = _strict_numeric_array(candidate, name="candidate")
    if left_array.shape != right_array.shape:
        raise ValueError(
            "reference and candidate must have the same shape; got "
            f"{left_array.shape} and {right_array.shape}"
        )
    left = left_array.reshape(-1)
    right = right_array.reshape(-1)

    def unit_direction(values: FloatArray, name: str) -> FloatArray:
        scale = float(np.max(np.abs(values)))
        if scale == 0.0:
            raise ValueError(f"{name} has zero norm; direction is undefined")
        scaled = values / scale
        scaled_norm = float(np.linalg.norm(scaled))
        norm = scale * scaled_norm
        if not math.isfinite(norm):
            raise ValueError(f"{name} norm is non-finite")
        if norm <= tolerance:
            raise ValueError(
                f"{name} norm does not exceed zero_tolerance; direction is undefined"
            )
        return scaled / scaled_norm

    left_unit = unit_direction(left, "reference")
    right_unit = unit_direction(right, "candidate")
    alignment = float(np.dot(left_unit, right_unit))
    if not math.isfinite(alignment):
        raise ValueError("direction alignment is non-finite")
    return float(np.clip(alignment, -1.0, 1.0))


def paired_difference_in_differences(
    treatment_before: ArrayLike,
    treatment_after: ArrayLike,
    control_before: ArrayLike,
    control_after: ArrayLike,
) -> dict[str, Any]:
    """Compute a matched difference-in-differences descriptive estimate.

    Pair ``i`` contributes
    ``(treatment_after[i] - treatment_before[i]) -``
    ``(control_after[i] - control_before[i])``.  This function reports the
    arithmetic mean and its descriptive pair dispersion; it does not assert
    the causal parallel-trends assumption.
    """

    matched = validate_matched_client_conditions(
        {
            "treatment_before": treatment_before,
            "treatment_after": treatment_after,
            "control_before": control_before,
            "control_after": control_after,
        }
    )
    if matched["treatment_before"].ndim != 1:
        raise ValueError("difference-in-differences inputs must be one-dimensional")
    with np.errstate(over="ignore", invalid="ignore"):
        treatment_change = matched["treatment_after"] - matched["treatment_before"]
        control_change = matched["control_after"] - matched["control_before"]
        pair_effects = treatment_change - control_change
    if not (
        np.all(np.isfinite(treatment_change))
        and np.all(np.isfinite(control_change))
        and np.all(np.isfinite(pair_effects))
    ):
        raise ValueError(
            "difference-in-differences arithmetic produced non-finite values"
        )
    n_pairs = int(pair_effects.size)
    estimate = float(np.mean(pair_effects))
    standard_deviation = float(np.std(pair_effects, ddof=1)) if n_pairs > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(n_pairs)
    return {
        "n_pairs": n_pairs,
        "estimate": estimate,
        "pair_standard_deviation": standard_deviation,
        "pair_standard_error": float(standard_error),
        "mean_treatment_change": float(np.mean(treatment_change)),
        "mean_control_change": float(np.mean(control_change)),
        "pair_effects": pair_effects.astype(float).tolist(),
    }


def bootstrap_stability_summary(
    values: ArrayLike,
    *,
    n_bootstrap: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, float | int]:
    """Deterministically bootstrap the mean of paired audit effects.

    The percentile interval is descriptive.  ``same_sign_fraction`` is the
    fraction of bootstrap means with the same strict sign as the observed
    mean; for an exactly zero observed mean it is the fraction exactly zero.
    """

    array = _strict_numeric_array(values, name="values")
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    repeats = _strict_positive_integer(n_bootstrap, name="n_bootstrap", minimum=2)
    confidence = _strict_finite_real(confidence_level, name="confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    random_seed = int(seed)
    if random_seed < 0 or random_seed > np.iinfo(np.uint64).max:
        raise ValueError("seed must lie in the uint64 range")

    rng = np.random.default_rng(random_seed)
    bootstrap_means = np.empty(repeats, dtype=np.float64)
    # Cap temporary index arrays at roughly one million entries.
    chunk_size = max(1, min(repeats, 1_000_000 // array.size))
    start = 0
    while start < repeats:
        stop = min(start + chunk_size, repeats)
        indices = rng.integers(0, array.size, size=(stop - start, array.size))
        bootstrap_means[start:stop] = np.mean(array[indices], axis=1)
        start = stop
    if not np.all(np.isfinite(bootstrap_means)):
        raise ValueError("bootstrap arithmetic produced non-finite values")

    alpha = 1.0 - confidence
    lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    observed_mean = float(np.mean(array))
    positive_fraction = float(np.mean(bootstrap_means > 0.0))
    negative_fraction = float(np.mean(bootstrap_means < 0.0))
    zero_fraction = float(np.mean(bootstrap_means == 0.0))
    if observed_mean > 0.0:
        same_sign_fraction = positive_fraction
    elif observed_mean < 0.0:
        same_sign_fraction = negative_fraction
    else:
        same_sign_fraction = zero_fraction
    return {
        "n_observations": int(array.size),
        "n_bootstrap": repeats,
        "seed": random_seed,
        "confidence_level": confidence,
        "observed_mean": observed_mean,
        "bootstrap_mean": float(np.mean(bootstrap_means)),
        "bootstrap_standard_deviation": float(np.std(bootstrap_means, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "zero_fraction": zero_fraction,
        "same_sign_fraction": same_sign_fraction,
    }


def _validated_key_names(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of key names")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise TypeError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def validate_component_intervention_records(
    records: Sequence[Mapping[str, Any]],
    *,
    match_keys: Sequence[str] = ("seed",),
    numeric_keys: Sequence[str] = ("outcome",),
    intervention_key: str = "intervention",
    draw_index_key: str = "draw_index",
    digest_key: str = "digest",
) -> tuple[dict[str, Any], ...]:
    """Validate complete matched component-intervention audit blocks.

    Each unique tuple of ``match_keys`` is one complete matched stratum.  It
    must contain one record for each of the three deterministic baseline arms
    and exactly 32 records for each stochastic control arm.  Stochastic arms
    must use the complete draw-index set ``0..31`` and unique SHA-256 draw
    digests across the whole stratum.  Baseline records may omit draw fields or
    set them to ``None``.  Required numeric fields are converted to finite
    Python floats in defensive record copies.
    """

    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if not records:
        raise ValueError("records must not be empty")
    matches = _validated_key_names(match_keys, name="match_keys")
    numerics = _validated_key_names(numeric_keys, name="numeric_keys")
    if not isinstance(intervention_key, str) or not intervention_key.strip():
        raise TypeError("intervention_key must be a non-empty string")
    if not isinstance(draw_index_key, str) or not draw_index_key.strip():
        raise TypeError("draw_index_key must be a non-empty string")
    if not isinstance(digest_key, str) or not digest_key.strip():
        raise TypeError("digest_key must be a non-empty string")
    all_reserved = (
        matches
        + numerics
        + (
            intervention_key,
            draw_index_key,
            digest_key,
        )
    )
    if len(set(all_reserved)) != len(all_reserved):
        raise ValueError(
            "match, numeric, intervention, draw-index, and digest keys must be disjoint"
        )

    required = set(REQUIRED_COMPONENT_INTERVENTIONS)
    groups: dict[
        tuple[Any, ...], dict[str, list[tuple[int, int | None, str | None]]]
    ] = {}
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        always_required = matches + numerics + (intervention_key,)
        missing = [key for key in always_required if key not in record]
        if missing:
            raise ValueError(f"record {index} is missing required keys {missing}")
        intervention = record[intervention_key]
        if not isinstance(intervention, str) or intervention not in required:
            raise ValueError(
                f"record {index} has unknown intervention {intervention!r}"
            )
        group_values = tuple(record[key] for key in matches)
        for key, value in zip(matches, group_values, strict=True):
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"match key {key!r} in record {index} must not be boolean"
                )
            if isinstance(value, Real) and not math.isfinite(float(value)):
                raise ValueError(f"match key {key!r} in record {index} must be finite")
            if isinstance(
                value, (Sequence, Mapping, set, np.ndarray)
            ) and not isinstance(value, (str, bytes)):
                raise TypeError(f"match key {key!r} in record {index} must be scalar")
            try:
                hash(value)
            except TypeError as error:
                raise TypeError(
                    f"match key {key!r} in record {index} must be hashable"
                ) from error

        draw_index: int | None = None
        digest: str | None = None
        if intervention in _DRAW_COMPONENT_INTERVENTIONS:
            missing_draw_fields = [
                key for key in (draw_index_key, digest_key) if key not in record
            ]
            if missing_draw_fields:
                raise ValueError(
                    f"record {index} is missing stochastic draw keys "
                    f"{missing_draw_fields}"
                )
            draw_index = (
                _strict_positive_integer(
                    (
                        int(record[draw_index_key]) + 1
                        if isinstance(record[draw_index_key], Integral)
                        and not isinstance(record[draw_index_key], (bool, np.bool_))
                        else record[draw_index_key]
                    ),
                    name=f"record {index} field {draw_index_key!r}",
                )
                - 1
            )
            if draw_index >= COMPONENT_CONTROL_DRAWS:
                raise ValueError(
                    f"record {index} draw index must lie in "
                    f"0..{COMPONENT_CONTROL_DRAWS - 1}"
                )
            raw_digest = record[digest_key]
            if (
                not isinstance(raw_digest, str)
                or len(raw_digest) != 64
                or raw_digest != raw_digest.lower()
                or any(character not in "0123456789abcdef" for character in raw_digest)
            ):
                raise ValueError(
                    f"record {index} field {digest_key!r} must be a lowercase "
                    "64-character SHA-256 digest"
                )
            digest = raw_digest
        else:
            if (
                record.get(draw_index_key) is not None
                or record.get(digest_key) is not None
            ):
                raise ValueError(
                    f"baseline record {index} must not define stochastic draw fields"
                )

        intervention_records = groups.setdefault(group_values, {}).setdefault(
            intervention, []
        )
        if any(
            existing_draw == draw_index for _, existing_draw, _ in intervention_records
        ):
            if intervention in _BASELINE_COMPONENT_INTERVENTIONS:
                raise ValueError(
                    f"duplicate baseline intervention {intervention!r} for matched "
                    f"group {group_values!r}"
                )
            raise ValueError(
                f"duplicate draw_index={draw_index} for intervention {intervention!r} "
                f"and matched group {group_values!r}"
            )
        intervention_records.append((index, draw_index, digest))

        copied = dict(record)
        for key in numerics:
            copied[key] = _strict_finite_real(
                record[key], name=f"record {index} field {key!r}"
            )
        if draw_index is not None:
            copied[draw_index_key] = draw_index
        validated.append(copied)

    expected_draw_indices = set(range(COMPONENT_CONTROL_DRAWS))
    for group, intervention_records in groups.items():
        present = set(intervention_records)
        if present != required:
            missing_interventions = sorted(required - present)
            extra_interventions = sorted(present - required)
            raise ValueError(
                f"matched group {group!r} is not a complete intervention block; "
                f"missing={missing_interventions}, extra={extra_interventions}"
            )
        for intervention in _BASELINE_COMPONENT_INTERVENTIONS:
            count = len(intervention_records[intervention])
            if count != 1:
                raise ValueError(
                    f"matched group {group!r} must contain exactly one "
                    f"{intervention!r} baseline record; got {count}"
                )
        stochastic_digests: list[str] = []
        for intervention in _DRAW_COMPONENT_INTERVENTIONS:
            rows = intervention_records[intervention]
            if len(rows) != COMPONENT_CONTROL_DRAWS:
                raise ValueError(
                    f"matched group {group!r} must contain exactly "
                    f"{COMPONENT_CONTROL_DRAWS} {intervention!r} draws; got {len(rows)}"
                )
            draw_indices = {draw for _, draw, _ in rows}
            if draw_indices != expected_draw_indices:
                raise ValueError(
                    f"matched group {group!r} intervention {intervention!r} does "
                    f"not contain the complete draw-index set 0..{COMPONENT_CONTROL_DRAWS - 1}"
                )
            stochastic_digests.extend(
                digest for _, _, digest in rows if digest is not None
            )
        if len(set(stochastic_digests)) != 2 * COMPONENT_CONTROL_DRAWS:
            raise ValueError(
                f"matched group {group!r} contains duplicate stochastic draw digests"
            )
    return tuple(validated)


__all__ = [
    "COMPONENT_CONTROL_DRAWS",
    "DEFAULT_COMPONENT_CONTROL_SEED",
    "DEFAULT_COMPONENT_INVARIANT_TOLERANCE",
    "REQUIRED_COMPONENT_INTERVENTIONS",
    "blockwise_shared_signed_parameter_permutation_controls",
    "bootstrap_stability_summary",
    "client_sparse_energy",
    "client_sparse_energy_fault_localization",
    "normalized_direction_alignment",
    "paired_difference_in_differences",
    "per_client_norm_support_matched_random_sparse_controls",
    "planted_support_recovery_metrics",
    "validate_component_intervention_records",
    "validate_matched_client_conditions",
]

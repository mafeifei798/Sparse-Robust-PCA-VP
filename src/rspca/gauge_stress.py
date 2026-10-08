"""Deterministic product-preserving LoRA gauge stress diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
GAUGE_STRESS_DRAWS = 16
GAUGE_STRESS_CONDITION_NUMBERS = (1.0, 10.0, 100.0)


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _finite_array(value: ArrayLike, *, name: str, ndim: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must be real and non-boolean")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != ndim or result.size == 0:
        raise ValueError(f"{name} must be a non-empty {ndim}-dimensional array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def _uint64(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result <= np.iinfo(np.uint64).max:
        raise ValueError(f"{name} must lie in the uint64 range")
    return result


def _derived_seed(master: int, *parts: object) -> int:
    payload = json.dumps(
        [master, *parts], separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _haar_orthogonal(rank: int, rng: np.random.Generator) -> FloatArray:
    matrix = rng.standard_normal((rank, rank))
    q, r = np.linalg.qr(matrix)
    diagonal = np.diag(r)
    signs = np.where(diagonal < 0.0, -1.0, 1.0)
    return np.asarray(q * signs[None, :], dtype=np.float64)


def _gauge_matrix(
    rank: int,
    *,
    condition_number: float,
    rng: np.random.Generator,
) -> FloatArray:
    if condition_number == 1.0:
        return _haar_orthogonal(rank, rng)
    left = _haar_orthogonal(rank, rng)
    right = _haar_orthogonal(rank, rng)
    half_log = 0.5 * math.log(condition_number)
    singular_values = np.exp(np.linspace(-half_log, half_log, rank))
    return np.asarray(
        left @ np.diag(singular_values) @ right.T,
        dtype=np.float64,
    )


def _array_digest(values: Mapping[str, FloatArray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        encoded = name.encode("utf-8")
        array = np.ascontiguousarray(values[name], dtype="<f8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _stable_norm(value: FloatArray) -> float:
    maximum = float(np.max(np.abs(value)))
    if maximum == 0.0:
        return 0.0
    return float(maximum * np.linalg.norm(value / maximum))


def product_preserving_gauge_stress_draws(
    parent_state: Mapping[str, ArrayLike],
    observed_delta_blocks: Mapping[str, ArrayLike],
    adapter_pairs: Sequence[tuple[str, str]],
    *,
    seed: int = 20261205,
    draws_per_nonidentity_condition: int = GAUGE_STRESS_DRAWS,
    condition_numbers: Sequence[float] = GAUGE_STRESS_CONDITION_NUMBERS,
    maximum_product_relative_error: float = 1e-10,
) -> tuple[dict[str, Any], ...]:
    """Return identity, Haar, and condition-10/100 GL stress draws.

    Every client-local full factor pair is transformed as
    ``(B, A) -> (B G, G^{-1} A)``.  The parent representation is held fixed,
    so the returned matrices are transformed local-minus-parent deltas.  No
    layer, Q/V projection, or A/B pair exchanges coordinates with another.
    """

    master_seed = _uint64(seed, name="seed")
    if isinstance(draws_per_nonidentity_condition, bool) or not isinstance(
        draws_per_nonidentity_condition, Integral
    ):
        raise TypeError("draws_per_nonidentity_condition must be an integer")
    draws = int(draws_per_nonidentity_condition)
    if draws != GAUGE_STRESS_DRAWS:
        raise ValueError(f"draw count must equal {GAUGE_STRESS_DRAWS}")
    if tuple(float(value) for value in condition_numbers) != (
        GAUGE_STRESS_CONDITION_NUMBERS
    ):
        raise ValueError("condition-number grid must equal (1, 10, 100)")
    if isinstance(maximum_product_relative_error, bool) or not isinstance(
        maximum_product_relative_error, Real
    ):
        raise TypeError("maximum_product_relative_error must be real")
    tolerance = float(maximum_product_relative_error)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("maximum_product_relative_error must be non-negative")
    if not isinstance(parent_state, Mapping) or not isinstance(
        observed_delta_blocks, Mapping
    ):
        raise TypeError("parent_state and observed_delta_blocks must be mappings")
    if not adapter_pairs:
        raise ValueError("adapter_pairs must not be empty")
    pairs = tuple(adapter_pairs)
    if len(set(pairs)) != len(pairs) or any(
        len(pair) != 2 or pair[0] == pair[1] for pair in pairs
    ):
        raise ValueError("adapter_pairs must contain unique A/B key pairs")
    adapter_keys = tuple(key for pair in pairs for key in pair)
    if len(set(adapter_keys)) != len(adapter_keys):
        raise ValueError("adapter factor keys must be unique")
    if set(parent_state) != set(adapter_keys):
        raise ValueError("parent_state must exactly contain the adapter registry")
    if set(observed_delta_blocks) != set(adapter_keys):
        raise ValueError("observed blocks must exactly contain the adapter registry")

    parent = {
        name: _finite_array(parent_state[name], name=f"parent {name}", ndim=2)
        for name in adapter_keys
    }
    observed = {
        name: _finite_array(
            observed_delta_blocks[name], name=f"observed {name}", ndim=2
        )
        for name in adapter_keys
    }
    n_clients = observed[adapter_keys[0]].shape[1]
    if n_clients <= 0 or any(
        value.shape[1] != n_clients for value in observed.values()
    ):
        raise ValueError("observed blocks must share a non-empty client axis")
    for a_key, b_key in pairs:
        a = parent[a_key]
        b = parent[b_key]
        if b.shape[1] != a.shape[0]:
            raise ValueError(f"incompatible LoRA factor shapes for {a_key}/{b_key}")
        if observed[a_key].shape[0] != a.size or observed[b_key].shape[0] != b.size:
            raise ValueError(f"flattened observed shape changed for {a_key}/{b_key}")

    settings = [("identity_control", 1.0, 0)]
    settings.extend(("haar_orthogonal_rotation", 1.0, index) for index in range(draws))
    for condition_number in (10.0, 100.0):
        settings.extend(
            (
                "orthogonally_rotated_diagonal_GL_transform",
                condition_number,
                index,
            )
            for index in range(draws)
        )
    outputs: list[dict[str, Any]] = []
    for transform, condition_number, draw_index in settings:
        transformed = {
            name: np.array(value, copy=True) for name, value in observed.items()
        }
        operator_digests: dict[str, str] = {}
        maximum_error = 0.0
        maximum_realized_condition = 1.0
        for pair_index, (a_key, b_key) in enumerate(pairs):
            a_parent = parent[a_key]
            b_parent = parent[b_key]
            rank = a_parent.shape[0]
            for client_index in range(n_clients):
                local_a = a_parent + observed[a_key][:, client_index].reshape(
                    a_parent.shape
                )
                local_b = b_parent + observed[b_key][:, client_index].reshape(
                    b_parent.shape
                )
                if transform == "identity_control":
                    gauge = np.eye(rank, dtype=np.float64)
                else:
                    local_seed = _derived_seed(
                        master_seed,
                        "gauge-stress",
                        transform,
                        condition_number,
                        draw_index,
                        pair_index,
                        client_index,
                    )
                    gauge = _gauge_matrix(
                        rank,
                        condition_number=condition_number,
                        rng=np.random.default_rng(local_seed),
                    )
                inverse = np.linalg.inv(gauge)
                transformed_a = inverse @ local_a
                transformed_b = local_b @ gauge
                reference_product = local_b @ local_a
                candidate_product = transformed_b @ transformed_a
                absolute_error = _stable_norm(candidate_product - reference_product)
                relative_error = absolute_error / max(
                    _stable_norm(reference_product), 1e-12
                )
                maximum_error = max(maximum_error, relative_error)
                realized_condition = float(np.linalg.cond(gauge))
                maximum_realized_condition = max(
                    maximum_realized_condition, realized_condition
                )
                if transform != "identity_control":
                    transformed[a_key][:, client_index] = (
                        transformed_a - a_parent
                    ).reshape(-1)
                    transformed[b_key][:, client_index] = (
                        transformed_b - b_parent
                    ).reshape(-1)
                key = f"pair{pair_index}:client{client_index}"
                operator_digests[key] = hashlib.sha256(
                    np.ascontiguousarray(gauge, dtype="<f8").tobytes()
                ).hexdigest()
        if maximum_error > tolerance:
            raise RuntimeError(
                "gauge transform violated the product-preservation tolerance"
            )
        readonly_blocks = {
            name: _readonly(value) for name, value in transformed.items()
        }
        outputs.append(
            {
                "transform": transform,
                "condition_number": condition_number,
                "draw_index": draw_index,
                "seed": master_seed,
                "n_clients": n_clients,
                "adapter_pair_count": len(pairs),
                "maximum_product_relative_error": maximum_error,
                "maximum_realized_condition_number": maximum_realized_condition,
                "operator_digests": MappingProxyType(operator_digests),
                "transformed_observed_blocks": MappingProxyType(readonly_blocks),
                "transformed_blocks_sha256": _array_digest(readonly_blocks),
            }
        )
    if len(outputs) != 1 + 3 * draws:
        raise RuntimeError("gauge stress draw registry is incomplete")
    if len({row["transformed_blocks_sha256"] for row in outputs}) != len(outputs):
        raise RuntimeError("gauge stress produced duplicate transformed draws")
    return tuple(outputs)


def aggregate_lora_product_drift(
    reference_state: Mapping[str, ArrayLike],
    candidate_state: Mapping[str, ArrayLike],
    adapter_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Measure server LoRA-product drift for fixed A/B pair registries."""

    pairs = tuple(adapter_pairs)
    keys = {key for pair in pairs for key in pair}
    if not pairs or set(reference_state) != keys or set(candidate_state) != keys:
        raise ValueError("states must exactly match a non-empty adapter registry")
    per_pair: list[dict[str, Any]] = []
    absolute_squared = 0.0
    reference_squared = 0.0
    for a_key, b_key in pairs:
        ref_a = _finite_array(reference_state[a_key], name=a_key, ndim=2)
        ref_b = _finite_array(reference_state[b_key], name=b_key, ndim=2)
        cand_a = _finite_array(candidate_state[a_key], name=a_key, ndim=2)
        cand_b = _finite_array(candidate_state[b_key], name=b_key, ndim=2)
        if ref_a.shape != cand_a.shape or ref_b.shape != cand_b.shape:
            raise ValueError("reference and candidate factor shapes changed")
        reference = ref_b @ ref_a
        candidate = cand_b @ cand_a
        absolute = _stable_norm(candidate - reference)
        reference_norm = _stable_norm(reference)
        relative = absolute / max(reference_norm, 1e-12)
        absolute_squared += absolute * absolute
        reference_squared += reference_norm * reference_norm
        per_pair.append(
            {
                "a_key": a_key,
                "b_key": b_key,
                "absolute_frobenius_drift": absolute,
                "relative_frobenius_drift": relative,
            }
        )
    global_absolute = math.sqrt(absolute_squared)
    global_reference = math.sqrt(reference_squared)
    return {
        "pair_count": len(pairs),
        "per_pair": per_pair,
        "server_aggregate_product_absolute_frobenius_drift": global_absolute,
        "server_aggregate_product_relative_drift": global_absolute
        / max(global_reference, 1e-12),
    }


__all__ = [
    "GAUGE_STRESS_CONDITION_NUMBERS",
    "GAUGE_STRESS_DRAWS",
    "aggregate_lora_product_drift",
    "product_preserving_gauge_stress_draws",
]

"""Gauge-aware canonicalization for LoRA factor pairs.

A LoRA update is represented by two factors ``B @ A`` with shapes
``(d_out, rank)`` and ``(rank, d_in)``.  The factors are not identifiable:
``(G @ A, B @ inv(G))`` represents the same update for every invertible
``G``.  This module removes that arbitrary gauge before factor-space
aggregation.

The implementation never forms the generally large ``d_out x d_in``
product.  Thin QR factorizations reduce the problem to the SVD of an
``rank x rank`` core.  The returned factors are balanced, and an optional
orthogonal Procrustes step aligns their normalized left/right frames to a
reference pair without changing the represented product.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
FactorPair = tuple[ArrayLike, ArrayLike]


@dataclass(frozen=True)
class GaugeCanonicalizationResult:
    """A balanced, optionally reference-aligned LoRA factor pair.

    ``product_residual`` checks product preservation entirely in the
    QR coordinates of the input factors; computing it does not materialize
    ``B @ A``.  ``alignment_singular_values`` is empty when no reference was
    requested.
    """

    A: FloatArray | None
    B: FloatArray | None
    singular_values: FloatArray
    effective_rank: int
    product_residual: float | None
    balance_residual: float | None
    aligned: bool
    alignment_singular_values: FloatArray
    alignment_margin: float | None
    eligible: bool
    reason: str


@dataclass(frozen=True)
class GaugePairAlignmentResult:
    """Audit record for one named adapter pair across global/client states."""

    a_key: str
    b_key: str
    global_result: GaugeCanonicalizationResult
    local_results: tuple[GaugeCanonicalizationResult, ...]
    eligible: bool
    reason: str


@dataclass(frozen=True)
class GaugeAlignedStatesResult:
    """Fail-closed output from aligning complete trainable states."""

    global_state: dict[str, FloatArray] | None
    local_states: tuple[dict[str, FloatArray], ...] | None
    pair_results: tuple[GaugePairAlignmentResult, ...]
    eligible: bool
    reason: str


@dataclass(frozen=True)
class _UnalignedCanonicalization:
    A: FloatArray
    B: FloatArray
    left_frame: FloatArray
    right_frame: FloatArray
    singular_values: FloatArray
    input_left_basis: FloatArray
    input_right_basis: FloatArray
    input_core: FloatArray
    rank_rtol: float


class _IneligibleGaugeError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        singular_values: ArrayLike = (),
        effective_rank: int = 0,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.singular_values = np.asarray(singular_values, dtype=np.float64)
        self.effective_rank = int(effective_rank)


def canonicalize_lora_pair(
    A: ArrayLike,
    B: ArrayLike,
    reference: FactorPair | GaugeCanonicalizationResult | None = None,
    rank_rtol: float | None = None,
    alignment_rtol: float | None = None,
) -> GaugeCanonicalizationResult:
    """Balance a full-rank LoRA pair and optionally align it to a reference.

    Parameters
    ----------
    A, B:
        LoRA factors with shapes ``(rank, d_in)`` and ``(d_out, rank)``.
    reference:
        Optional ``(A, B)`` pair (or a previous result) whose normalized
        left/right frames define the orthogonal gauge.  The reference may
        represent a different update, but its factor shapes must match.
    rank_rtol:
        Relative singular-value tolerance used to reject zero or
        rank-deficient represented products.  The default is a
        dimension-aware floating-point tolerance.
    alignment_rtol:
        Relative tolerance used to reject a singular normalized-frame
        Procrustes problem.  The default is ``rank * eps``.

    Returns
    -------
    GaugeCanonicalizationResult
        Read-only, balanced factors representing exactly the same product up
        to floating-point roundoff.

    Notes
    -----
    Failing on a zero-initialized or rank-deficient product is deliberate.
    Such a pair has no complete rank-dimensional gauge to canonicalize, and
    silently inventing one would make factor aggregation initialization- and
    implementation-dependent.
    """

    try:
        unaligned = _canonicalize_unaligned(A, B, rank_rtol=rank_rtol)
    except _IneligibleGaugeError as error:
        return _ineligible_result(
            reason=error.reason,
            singular_values=error.singular_values,
            effective_rank=error.effective_rank,
        )
    output_a = unaligned.A
    output_b = unaligned.B
    alignment_singular_values = np.empty(0, dtype=np.float64)
    alignment_margin: float | None = None
    aligned = False

    if reference is not None:
        if (
            isinstance(reference, GaugeCanonicalizationResult)
            and not reference.eligible
        ):
            return _ineligible_result(
                reason=f"reference is ineligible: {reference.reason}"
            )
        reference_a, reference_b = _reference_factors(reference)
        if reference_a.shape != output_a.shape or reference_b.shape != output_b.shape:
            raise ValueError("reference A and B shapes must match the input factors")
        if isinstance(reference, GaugeCanonicalizationResult):
            reference_left_frame, reference_right_frame = _normalized_frames(
                reference_a,
                reference_b,
            )
        else:
            try:
                reference_state = _canonicalize_unaligned(
                    reference_a,
                    reference_b,
                    rank_rtol=rank_rtol,
                )
            except _IneligibleGaugeError as error:
                return _ineligible_result(
                    reason=f"reference is ineligible: {error.reason}",
                    singular_values=unaligned.singular_values,
                    effective_rank=unaligned.singular_values.size,
                )
            reference_left_frame = reference_state.left_frame
            reference_right_frame = reference_state.right_frame
        cross_frame = (
            unaligned.left_frame.T @ reference_left_frame
            + unaligned.right_frame.T @ reference_right_frame
        )
        left_rotation, alignment_singular_values, right_rotation_t = np.linalg.svd(
            cross_frame,
            full_matrices=False,
        )
        alignment_tolerance = _relative_tolerance(
            alignment_rtol,
            default=unaligned.singular_values.size * np.finfo(np.float64).eps,
            name="alignment_rtol",
        )
        largest_alignment = float(alignment_singular_values[0])
        alignment_margin = (
            float(alignment_singular_values[-1]) / largest_alignment
            if largest_alignment > 0.0
            else 0.0
        )
        if (
            largest_alignment <= 0.0
            or float(alignment_singular_values[-1])
            <= alignment_tolerance * largest_alignment
        ):
            return _ineligible_result(
                reason="normalized-frame alignment is singular at alignment_rtol",
                singular_values=unaligned.singular_values,
                effective_rank=unaligned.singular_values.size,
                alignment_singular_values=alignment_singular_values,
                alignment_margin=alignment_margin,
            )
        rotation = left_rotation @ right_rotation_t
        output_b = output_b @ rotation
        output_a = rotation.T @ output_a
        aligned = True

    balance_residual = _relative_balance_residual(output_a, output_b)
    product_residual = _relative_core_product_residual(
        output_a,
        output_b,
        unaligned,
    )
    return GaugeCanonicalizationResult(
        A=_readonly(output_a),
        B=_readonly(output_b),
        singular_values=_readonly(unaligned.singular_values),
        effective_rank=int(unaligned.singular_values.size),
        product_residual=product_residual,
        balance_residual=balance_residual,
        aligned=aligned,
        alignment_singular_values=_readonly(alignment_singular_values),
        alignment_margin=alignment_margin,
        eligible=True,
        reason="ok",
    )


def align_lora_states(
    global_state: Mapping[str, ArrayLike],
    local_states: Sequence[Mapping[str, ArrayLike]],
    adapter_pairs: Sequence[tuple[str, str]],
    rank_rtol: float | None = None,
    alignment_rtol: float | None = None,
) -> GaugeAlignedStatesResult:
    """Align every registered local adapter pair to its global pair.

    Non-adapter values are copied unchanged.  If any global pair is zero or
    rank deficient, or any local/reference alignment is ineligible, this
    wrapper returns ``global_state=None`` and ``local_states=None``.  It never
    falls back to a client-defined reference gauge.
    """

    global_arrays, local_arrays, pairs = _validate_states_and_pairs(
        global_state,
        local_states,
        adapter_pairs,
    )
    aligned_global = {key: value.copy() for key, value in global_arrays.items()}
    aligned_locals = [
        {key: value.copy() for key, value in state.items()} for state in local_arrays
    ]
    pair_results: list[GaugePairAlignmentResult] = []
    failures: list[str] = []

    for a_key, b_key in pairs:
        global_result = canonicalize_lora_pair(
            global_arrays[a_key],
            global_arrays[b_key],
            rank_rtol=rank_rtol,
            alignment_rtol=alignment_rtol,
        )
        local_results: list[GaugeCanonicalizationResult] = []
        if global_result.eligible:
            assert global_result.A is not None and global_result.B is not None
            aligned_global[a_key] = global_result.A.copy()
            aligned_global[b_key] = global_result.B.copy()
            for client_index, state in enumerate(local_arrays):
                local_result = canonicalize_lora_pair(
                    state[a_key],
                    state[b_key],
                    reference=global_result,
                    rank_rtol=rank_rtol,
                    alignment_rtol=alignment_rtol,
                )
                local_results.append(local_result)
                if local_result.eligible:
                    assert local_result.A is not None and local_result.B is not None
                    aligned_locals[client_index][a_key] = local_result.A.copy()
                    aligned_locals[client_index][b_key] = local_result.B.copy()
                else:
                    failures.append(
                        f"{a_key}/{b_key} client {client_index}: {local_result.reason}"
                    )
        else:
            failures.append(f"{a_key}/{b_key} global: {global_result.reason}")

        pair_eligible = (
            global_result.eligible
            and all(result.eligible for result in local_results)
            and len(local_results) == len(local_arrays)
        )
        pair_reason = "ok" if pair_eligible else failures[-1]
        pair_results.append(
            GaugePairAlignmentResult(
                a_key=a_key,
                b_key=b_key,
                global_result=global_result,
                local_results=tuple(local_results),
                eligible=pair_eligible,
                reason=pair_reason,
            )
        )

    if failures:
        return GaugeAlignedStatesResult(
            global_state=None,
            local_states=None,
            pair_results=tuple(pair_results),
            eligible=False,
            reason="; ".join(failures),
        )
    return GaugeAlignedStatesResult(
        global_state=aligned_global,
        local_states=tuple(aligned_locals),
        pair_results=tuple(pair_results),
        eligible=True,
        reason="ok",
    )


def _canonicalize_unaligned(
    A: ArrayLike,
    B: ArrayLike,
    *,
    rank_rtol: float | None,
) -> _UnalignedCanonicalization:
    a, b = _validate_factor_pair(A, B)
    rank = a.shape[0]
    if rank > min(a.shape[1], b.shape[0]):
        raise ValueError(
            "LoRA rank cannot exceed d_in or d_out for full-rank canonicalization"
        )

    left_basis, left_triangular = np.linalg.qr(b, mode="reduced")
    right_basis, right_triangular = np.linalg.qr(a.T, mode="reduced")
    core = left_triangular @ right_triangular.T
    core_left, singular_values, core_right_t = np.linalg.svd(
        core,
        full_matrices=False,
    )

    tolerance = _relative_tolerance(
        rank_rtol,
        default=max(a.shape[1], b.shape[0], rank) * np.finfo(np.float64).eps,
        name="rank_rtol",
    )
    largest = float(singular_values[0])
    if largest <= 0.0:
        raise _IneligibleGaugeError(
            "cannot canonicalize a zero-product LoRA pair",
            singular_values=singular_values,
            effective_rank=0,
        )
    if float(singular_values[-1]) <= tolerance * largest:
        effective_rank = int(np.count_nonzero(singular_values > tolerance * largest))
        raise _IneligibleGaugeError(
            "cannot canonicalize a rank-deficient LoRA product: "
            f"effective rank {effective_rank} < configured rank {rank}",
            singular_values=singular_values,
            effective_rank=effective_rank,
        )

    left_frame = left_basis @ core_left
    right_frame = right_basis @ core_right_t.T
    _fix_joint_column_signs(left_frame, right_frame)
    roots = np.sqrt(singular_values)
    output_b = left_frame * roots
    output_a = roots[:, None] * right_frame.T
    return _UnalignedCanonicalization(
        A=np.asarray(output_a, dtype=np.float64),
        B=np.asarray(output_b, dtype=np.float64),
        left_frame=np.asarray(left_frame, dtype=np.float64),
        right_frame=np.asarray(right_frame, dtype=np.float64),
        singular_values=np.asarray(singular_values, dtype=np.float64),
        input_left_basis=np.asarray(left_basis, dtype=np.float64),
        input_right_basis=np.asarray(right_basis, dtype=np.float64),
        input_core=np.asarray(core, dtype=np.float64),
        rank_rtol=tolerance,
    )


def _validate_factor_pair(A: ArrayLike, B: ArrayLike) -> tuple[FloatArray, FloatArray]:
    a = _as_nonempty_finite_matrix(A, name="A")
    b = _as_nonempty_finite_matrix(B, name="B")
    if b.shape[1] != a.shape[0]:
        raise ValueError("incompatible factors: B.shape[1] must equal A.shape[0]")
    return a, b


def _reference_factors(
    reference: FactorPair | GaugeCanonicalizationResult,
) -> tuple[FloatArray, FloatArray]:
    if isinstance(reference, GaugeCanonicalizationResult):
        if reference.A is None or reference.B is None:
            raise ValueError("ineligible reference has no canonical factors")
        return reference.A, reference.B
    if (
        isinstance(reference, (str, bytes))
        or not isinstance(reference, Sequence)
        or len(reference) != 2
    ):
        raise TypeError("reference/state must be a two-element (A, B) sequence")
    return _validate_factor_pair(reference[0], reference[1])


def _validate_states_and_pairs(
    global_state: Mapping[str, ArrayLike],
    local_states: Sequence[Mapping[str, ArrayLike]],
    adapter_pairs: Sequence[tuple[str, str]],
) -> tuple[
    dict[str, FloatArray],
    list[dict[str, FloatArray]],
    tuple[tuple[str, str], ...],
]:
    if not isinstance(global_state, Mapping) or not global_state:
        raise ValueError("global_state must be a non-empty mapping")
    if isinstance(local_states, (str, bytes)) or not isinstance(local_states, Sequence):
        raise TypeError("local_states must be a non-empty sequence of mappings")
    if not local_states:
        raise ValueError("local_states must not be empty")
    if isinstance(adapter_pairs, (str, bytes)) or not isinstance(
        adapter_pairs, Sequence
    ):
        raise TypeError("adapter_pairs must be a non-empty sequence")
    if not adapter_pairs:
        raise ValueError("adapter_pairs must not be empty")

    keys = set(global_state)
    if any(not isinstance(key, str) or not key for key in keys):
        raise ValueError("global_state keys must be non-empty strings")
    global_arrays = {
        key: _as_finite_array(value, name=f"global_state[{key!r}]")
        for key, value in global_state.items()
    }
    local_arrays: list[dict[str, FloatArray]] = []
    for client_index, state in enumerate(local_states):
        if not isinstance(state, Mapping):
            raise TypeError(f"local_states[{client_index}] must be a mapping")
        if set(state) != keys:
            raise ValueError(
                f"local_states[{client_index}] keys must match global_state"
            )
        converted = {
            key: _as_finite_array(
                state[key],
                name=f"local_states[{client_index}][{key!r}]",
            )
            for key in global_arrays
        }
        for key in global_arrays:
            if converted[key].shape != global_arrays[key].shape:
                raise ValueError(
                    f"local_states[{client_index}][{key!r}] shape must match global"
                )
        local_arrays.append(converted)

    pairs: list[tuple[str, str]] = []
    used_keys: set[str] = set()
    for pair_index, pair in enumerate(adapter_pairs):
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, Sequence)
            or len(pair) != 2
        ):
            raise TypeError(
                f"adapter_pairs[{pair_index}] must be a two-element key pair"
            )
        a_key, b_key = pair
        if not isinstance(a_key, str) or not a_key:
            raise ValueError(f"adapter_pairs[{pair_index}] A key must be non-empty")
        if not isinstance(b_key, str) or not b_key:
            raise ValueError(f"adapter_pairs[{pair_index}] B key must be non-empty")
        if a_key == b_key or a_key in used_keys or b_key in used_keys:
            raise ValueError("adapter_pairs must not duplicate or reuse keys")
        if a_key not in global_arrays or b_key not in global_arrays:
            raise KeyError(f"adapter_pairs[{pair_index}] contains an unknown key")
        _validate_factor_pair(global_arrays[a_key], global_arrays[b_key])
        used_keys.update((a_key, b_key))
        pairs.append((a_key, b_key))
    return global_arrays, local_arrays, tuple(pairs)


def _as_finite_array(values: ArrayLike, *, name: str) -> FloatArray:
    candidate = values
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    numpy_method = getattr(candidate, "numpy", None)
    if callable(numpy_method):
        candidate = numpy_method()
    array = np.asarray(candidate)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain real numeric values")
    result = np.asarray(array, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result.copy()


def _as_nonempty_finite_matrix(values: ArrayLike, *, name: str) -> FloatArray:
    array = _as_finite_array(values, name=name)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")
    return array


def _ineligible_result(
    *,
    reason: str,
    singular_values: ArrayLike = (),
    effective_rank: int = 0,
    alignment_singular_values: ArrayLike = (),
    alignment_margin: float | None = None,
) -> GaugeCanonicalizationResult:
    return GaugeCanonicalizationResult(
        A=None,
        B=None,
        singular_values=_readonly(singular_values),
        effective_rank=int(effective_rank),
        product_residual=None,
        balance_residual=None,
        aligned=False,
        alignment_singular_values=_readonly(alignment_singular_values),
        alignment_margin=alignment_margin,
        eligible=False,
        reason=reason,
    )


def _relative_tolerance(value: float | None, *, default: float, name: str) -> float:
    if value is None:
        return float(default)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    tolerance = float(value)
    if not np.isfinite(tolerance) or not 0.0 <= tolerance < 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1)")
    return tolerance


def _fix_joint_column_signs(left: FloatArray, right: FloatArray) -> None:
    """Choose a deterministic joint sign for each singular-vector pair."""

    for column in range(left.shape[1]):
        left_index = int(np.argmax(np.abs(left[:, column])))
        right_index = int(np.argmax(np.abs(right[:, column])))
        left_value = float(left[left_index, column])
        right_value = float(right[right_index, column])
        anchor = left_value if abs(left_value) >= abs(right_value) else right_value
        if anchor < 0.0:
            left[:, column] *= -1.0
            right[:, column] *= -1.0


def _relative_balance_residual(A: FloatArray, B: FloatArray) -> float:
    left_gram = B.T @ B
    right_gram = A @ A.T
    denominator = max(
        float(np.linalg.norm(left_gram, ord="fro")),
        float(np.linalg.norm(right_gram, ord="fro")),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(left_gram - right_gram, ord="fro") / denominator)


def _normalized_frames(A: FloatArray, B: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Return the left/right polar frames of a balanced full-rank pair."""

    gram = 0.5 * (B.T @ B + A @ A.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if float(eigenvalues[0]) <= 0.0:
        raise ValueError("eligible reference unexpectedly has a singular factor gram")
    inverse_root = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return B @ inverse_root, A.T @ inverse_root


def _relative_core_product_residual(
    A: FloatArray,
    B: FloatArray,
    source: _UnalignedCanonicalization,
) -> float:
    reconstructed_core = (source.input_left_basis.T @ B) @ (
        A @ source.input_right_basis
    )
    denominator = max(
        float(np.linalg.norm(source.input_core, ord="fro")),
        np.finfo(np.float64).tiny,
    )
    return float(
        np.linalg.norm(reconstructed_core - source.input_core, ord="fro") / denominator
    )


def _readonly(values: ArrayLike) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


__all__ = [
    "GaugeAlignedStatesResult",
    "GaugeCanonicalizationResult",
    "GaugePairAlignmentResult",
    "align_lora_states",
    "canonicalize_lora_pair",
]

r"""Solver-only certificate-balanced prototype for the v2 objective.

This module deliberately keeps the Cleaned-Encoder Robust VP-SPCA v2
objective unchanged:

.. math::

   \min_{A,B,S}\; \tfrac12\|(X-S)(I-BA^T)\|_F^2
       + \alpha\|B\|_1 + \kappa\|S\|_1,
   \qquad A^T A=I.

It changes only the block scheduling rule.  At every accepted micro-update,
all v2 block certificates are available at one common state.  The block with
the largest normalized certificate is selected unless a bounded-delay rule
must update a starved block.  B and S retain their monotone proximal
majorizers, and A retains the exact polar/Procrustes update.

The implementation is an isolated numerical prototype.  It does not define an
outlier scoring rule, an evaluation protocol, or a statistical success claim.
It is not imported by :mod:`rspca.__init__`; its only video-shaped entry point
is a separate truth-blind, observed-only solver-feasibility runner that makes
no computational use of a generator or statistical evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from .cleaned_encoder_v2 import (
    CleanedEncoderBlockStationarity,
    _B_gradient,
    _S_gradient,
    _initial_S_step_size,
    _smooth_from_cleaned,
    cleaned_encoder_block_stationarity,
    cleaned_encoder_objective,
    update_A_cleaned_encoder_procrustes,
)
from .vp import (
    FloatArray,
    _as_float_matrix,
    _initial_loadings,
    _pca_initial_loadings_and_step_size,
    _validate_n_components,
    _validate_nonnegative_scalar,
    _validate_positive_integer,
    _validate_positive_scalar,
    soft_threshold,
)


BlockName = Literal["B", "A", "S"]
SelectionReason = Literal["initial_coverage", "greedy", "bounded_delay"]
_BLOCKS: tuple[BlockName, ...] = ("B", "A", "S")


@dataclass(frozen=True)
class CertificateBalancedBlockCounts:
    """Accepted scheduled updates for each optimization block."""

    B: int
    A: int
    S: int

    @property
    def total(self) -> int:
        return self.B + self.A + self.S


@dataclass(frozen=True)
class CertificateBalancedWorkCounters:
    """Auditable work counters for the solver-only prototype.

    A full-stationarity evaluation internally computes one B gradient, one A
    cross product, and one S gradient.  Those common-state checks are recorded
    separately from gradients used to perform a selected block update.
    """

    objective_evaluations: int
    smooth_term_evaluations: int
    full_stationarity_evaluations: int
    B_update_gradient_evaluations: int
    S_update_gradient_evaluations: int
    A_polar_factorizations: int
    B_prox_backtracking_trials: int
    S_prox_backtracking_trials: int
    accepted_micro_updates: int


@dataclass(frozen=True)
class CertificateBalancedUpdate:
    """One accepted micro-update and the certificates that selected it."""

    update_index: int
    block: BlockName
    reason: SelectionReason
    B_certificate_before: float
    A_certificate_before: float
    S_certificate_before: float
    total_certificate_before: float
    B_certificate_step_size: float
    S_certificate_step_size: float
    objective_before: float
    objective_after: float
    accepted_step_size: float | None
    backtracking_trials: int


@dataclass(frozen=True)
class CertificateBalancedCleanedEncoderResult:
    """Result of :func:`fit_certificate_balanced_cleaned_encoder`."""

    A: FloatArray
    B: FloatArray
    S: FloatArray
    reconstruction: FloatArray
    objectives: tuple[float, ...]
    stationarity: tuple[float, ...]
    block_stationarity: tuple[CleanedEncoderBlockStationarity, ...]
    updates: tuple[CertificateBalancedUpdate, ...]
    block_counts: CertificateBalancedBlockCounts
    work: CertificateBalancedWorkCounters
    converged: bool
    n_updates: int
    alpha: float
    kappa: float
    tol: float
    max_block_delay: int
    B_certificate_step_size: float
    S_certificate_step_size: float
    status: str

    @property
    def block_choices(self) -> tuple[BlockName, ...]:
        return tuple(update.block for update in self.updates)

    @property
    def selection_reasons(self) -> tuple[SelectionReason, ...]:
        return tuple(update.reason for update in self.updates)


def _validate_delay(value: int) -> int:
    delay = _validate_positive_integer(value, name="max_block_delay")
    if delay < len(_BLOCKS):
        raise ValueError("max_block_delay must be at least the number of blocks (3)")
    return delay


def _certificate_scores(
    diagnostics: CleanedEncoderBlockStationarity,
) -> dict[BlockName, float]:
    return {
        "B": float(diagnostics.B_prox_mapping),
        "A": float(
            max(
                diagnostics.A_riemannian_kkt,
                diagnostics.A_global_optimality_gap,
                diagnostics.A_orthogonality,
            )
        ),
        "S": float(diagnostics.S_prox_mapping),
    }


def _largest_certificate(
    candidates: tuple[BlockName, ...],
    scores: dict[BlockName, float],
) -> BlockName:
    # ``max`` retains the first candidate on a tie, giving a stable B/A/S tie
    # rule rather than relying on hash or dictionary iteration order.
    return max(candidates, key=lambda block: scores[block])


def _select_block(
    *,
    update_index: int,
    scores: dict[BlockName, float],
    last_selected: dict[BlockName, int | None],
    max_block_delay: int,
) -> tuple[BlockName, SelectionReason]:
    if update_index < len(_BLOCKS):
        # A fixed B/A/S warm-up makes the first three micro-updates identical
        # to one cycle of the archived v2 solver.  It also avoids letting the
        # diagnostic scale choose its own initialization order.
        return _BLOCKS[update_index], "initial_coverage"

    overdue = tuple(
        block
        for block in _BLOCKS
        if update_index - int(last_selected[block]) >= max_block_delay
    )
    if overdue:
        oldest = min(int(last_selected[block]) for block in overdue)
        oldest_blocks = tuple(
            block for block in overdue if int(last_selected[block]) == oldest
        )
        return _largest_certificate(oldest_blocks, scores), "bounded_delay"
    return _largest_certificate(_BLOCKS, scores), "greedy"


def _initial_B_and_step(
    X: FloatArray,
    n_components: int,
    *,
    initial_B: ArrayLike | None,
    B_step_size: float | None,
) -> tuple[FloatArray, float]:
    if B_step_size is None and initial_B is None:
        return _pca_initial_loadings_and_step_size(X, n_components)
    B = _initial_loadings(X, n_components, initial_B)
    if B_step_size is not None:
        return B, _validate_positive_scalar(B_step_size, name="B_step_size")
    spectral_norm = float(linalg.norm(X, ord=2, check_finite=False))
    step = 1.0 / max(spectral_norm * spectral_norm, np.finfo(np.float64).eps)
    return B, step


def _canonical_certificate_step_sizes(
    X: FloatArray,
    B: FloatArray,
    S: FloatArray,
) -> tuple[float, float]:
    """Return current-state, path-independent diagnostic step sizes.

    These conservative bounds are deliberately separate from the proposal
    step sizes used by line search.  Otherwise two schedules that accept
    different historical steps would also change the scale of the certificate
    used to select and compare their next block.
    """

    cleaned = X - S
    cleaned_frobenius_squared = (
        float(linalg.norm(cleaned, ord="fro", check_finite=False)) ** 2
    )
    # The unit cap matters when ``cleaned`` is zero or tiny.  Although any
    # positive step majorizes a zero quadratic, an enormous diagnostic step
    # makes the usual prox-mapping norm tend to zero for a nonstationary L1
    # block.  Keeping eta <= 1 preserves a meaningful KKT certificate.
    B_step = 1.0 / max(1.0, cleaned_frobenius_squared)
    S_bound = 1.0 + float(linalg.norm(B, ord=2, check_finite=False))
    return float(B_step), float(1.0 / (S_bound * S_bound))


def fit_certificate_balanced_cleaned_encoder(
    X: ArrayLike,
    n_components: int,
    *,
    alpha: float = 0.0015,
    kappa: float = 0.001,
    max_updates: int = 600,
    tol: float = 1e-6,
    max_block_delay: int = 5,
    B_step_size: float | None = None,
    S_step_size: float | None = None,
    initial_B: ArrayLike | None = None,
    initial_S: ArrayLike | None = None,
    backtracking_expand: float = 1.2,
) -> CertificateBalancedCleanedEncoderResult:
    """Fit the unchanged v2 objective with greedy certificate scheduling.

    ``max_block_delay`` bounds the difference between consecutive scheduled
    indices for any block after the initial three-block coverage phase.  A
    starved block takes precedence over the greedy rule.  Formal convergence is
    assessed only after recomputing every v2 block certificate at one common
    final state.  ``max_updates`` must be at least three so that the fixed
    B/A/S warm-up cannot be truncated.
    """

    X_array = _as_float_matrix(X)
    n_components = _validate_n_components(X_array, n_components)
    alpha = _validate_nonnegative_scalar(alpha, name="alpha")
    kappa = _validate_positive_scalar(kappa, name="kappa")
    max_updates = _validate_positive_integer(max_updates, name="max_updates")
    if max_updates < len(_BLOCKS):
        raise ValueError("max_updates must allow the three-block B/A/S warm-up")
    tol = _validate_positive_scalar(tol, name="tol")
    max_block_delay = _validate_delay(max_block_delay)
    expand = _validate_positive_scalar(
        backtracking_expand,
        name="backtracking_expand",
    )
    if expand < 1.0:
        raise ValueError("backtracking_expand must be at least 1")

    B, initial_gamma_B = _initial_B_and_step(
        X_array,
        n_components,
        initial_B=initial_B,
        B_step_size=B_step_size,
    )
    if initial_S is None:
        S = np.zeros_like(X_array)
    else:
        S = _as_float_matrix(initial_S, name="initial_S").copy()
        if S.shape != X_array.shape:
            raise ValueError("initial_S must have the same shape as X")

    A = update_A_cleaned_encoder_procrustes(X_array, B, S)
    initial_gamma_S = (
        _initial_S_step_size(B)
        if S_step_size is None
        else _validate_positive_scalar(S_step_size, name="S_step_size")
    )
    gamma_B_proposal = initial_gamma_B
    gamma_S_proposal = initial_gamma_S
    certificate_steps = {"B": np.nan, "S": np.nan}

    counters = {
        "objective_evaluations": 0,
        "smooth_term_evaluations": 0,
        "full_stationarity_evaluations": 0,
        "B_update_gradient_evaluations": 0,
        "S_update_gradient_evaluations": 0,
        "A_polar_factorizations": 1,
        "B_prox_backtracking_trials": 0,
        "S_prox_backtracking_trials": 0,
        "accepted_micro_updates": 0,
    }

    def evaluate_objective() -> float:
        counters["objective_evaluations"] += 1
        counters["smooth_term_evaluations"] += 1
        return cleaned_encoder_objective(
            X_array,
            A,
            B,
            S,
            alpha=alpha,
            kappa=kappa,
        )

    def evaluate_stationarity() -> CleanedEncoderBlockStationarity:
        counters["full_stationarity_evaluations"] += 1
        eta_B, eta_S = _canonical_certificate_step_sizes(X_array, B, S)
        certificate_steps["B"] = eta_B
        certificate_steps["S"] = eta_S
        return cleaned_encoder_block_stationarity(
            X_array,
            A,
            B,
            S,
            alpha=alpha,
            kappa=kappa,
            B_step_size=eta_B,
            S_step_size=eta_S,
        )

    current_objective = evaluate_objective()
    current_diagnostics = evaluate_stationarity()
    objectives: list[float] = [float(current_objective)]
    stationarity: list[float] = [float(current_diagnostics.total)]
    block_stationarity: list[CleanedEncoderBlockStationarity] = [current_diagnostics]
    updates: list[CertificateBalancedUpdate] = []
    block_counts: dict[BlockName, int] = {"B": 0, "A": 0, "S": 0}
    last_selected: dict[BlockName, int | None] = {
        "B": None,
        "A": None,
        "S": None,
    }

    for update_index in range(max_updates):
        # Complete the fixed B/A/S warm-up before allowing termination.  This
        # keeps the initialization contract identical to one archived v2
        # cycle, even on degenerate inputs that become stationary after B.
        if update_index >= len(_BLOCKS) and current_diagnostics.total <= tol:
            break
        scores = _certificate_scores(current_diagnostics)
        block, reason = _select_block(
            update_index=update_index,
            scores=scores,
            last_selected=last_selected,
            max_block_delay=max_block_delay,
        )
        B_certificate_step_before = float(certificate_steps["B"])
        S_certificate_step_before = float(certificate_steps["S"])
        objective_before = float(current_objective)
        accepted_step_size: float | None = None
        backtracking_trials = 0

        if block == "B":
            cleaned = X_array - S
            gradient_B = _B_gradient(cleaned, A, B)
            counters["B_update_gradient_evaluations"] += 1
            smooth_current = _smooth_from_cleaned(cleaned, A, B)
            counters["smooth_term_evaluations"] += 1
            gamma_trial = gamma_B_proposal
            for trial_index in range(61):
                counters["B_prox_backtracking_trials"] += 1
                B_trial = soft_threshold(
                    B - gamma_trial * gradient_B,
                    gamma_trial * alpha,
                )
                difference = B_trial - B
                smooth_trial = _smooth_from_cleaned(cleaned, A, B_trial)
                counters["smooth_term_evaluations"] += 1
                upper = smooth_current + float(np.sum(gradient_B * difference))
                upper += 0.5 * float(np.sum(difference * difference)) / gamma_trial
                objective_trial = (
                    smooth_trial
                    + alpha * float(np.sum(np.abs(B_trial)))
                    + kappa * float(np.sum(np.abs(S)))
                )
                counters["objective_evaluations"] += 1
                tolerance = 1e-12 * max(1.0, abs(objective_before))
                if smooth_trial <= upper + tolerance and objective_trial <= (
                    objective_before + tolerance
                ):
                    break
                gamma_trial *= 0.5
            else:  # pragma: no cover
                raise RuntimeError(
                    "certificate-balanced B backtracking failed to terminate"
                )
            B = np.asarray(B_trial, dtype=np.float64)
            current_objective = float(objective_trial)
            gamma_B_proposal = gamma_trial * expand
            accepted_step_size = float(gamma_trial)
            backtracking_trials = trial_index + 1

        elif block == "A":
            A_trial = update_A_cleaned_encoder_procrustes(
                X_array,
                B,
                S,
                reference=A,
            )
            counters["A_polar_factorizations"] += 1
            cleaned = X_array - S
            objective_trial = (
                _smooth_from_cleaned(cleaned, A_trial, B)
                + alpha * float(np.sum(np.abs(B)))
                + kappa * float(np.sum(np.abs(S)))
            )
            counters["smooth_term_evaluations"] += 1
            counters["objective_evaluations"] += 1
            tolerance = 1e-11 * max(1.0, abs(objective_before))
            if objective_trial > objective_before + tolerance:
                raise RuntimeError("certificate-balanced A block increased objective")
            A = np.asarray(A_trial, dtype=np.float64)
            current_objective = float(objective_trial)

        else:
            cleaned = X_array - S
            gradient_S = _S_gradient(cleaned, A, B)
            counters["S_update_gradient_evaluations"] += 1
            smooth_current = _smooth_from_cleaned(cleaned, A, B)
            counters["smooth_term_evaluations"] += 1
            gamma_trial = gamma_S_proposal
            for trial_index in range(61):
                counters["S_prox_backtracking_trials"] += 1
                S_trial = soft_threshold(
                    S - gamma_trial * gradient_S,
                    gamma_trial * kappa,
                )
                difference = S_trial - S
                cleaned_trial = X_array - S_trial
                smooth_trial = _smooth_from_cleaned(cleaned_trial, A, B)
                counters["smooth_term_evaluations"] += 1
                upper = smooth_current + float(np.sum(gradient_S * difference))
                upper += 0.5 * float(np.sum(difference * difference)) / gamma_trial
                objective_trial = (
                    smooth_trial
                    + alpha * float(np.sum(np.abs(B)))
                    + kappa * float(np.sum(np.abs(S_trial)))
                )
                counters["objective_evaluations"] += 1
                tolerance = 1e-12 * max(1.0, abs(objective_before))
                if smooth_trial <= upper + tolerance and objective_trial <= (
                    objective_before + tolerance
                ):
                    break
                gamma_trial *= 0.5
            else:  # pragma: no cover
                raise RuntimeError(
                    "certificate-balanced S backtracking failed to terminate"
                )
            S = np.asarray(S_trial, dtype=np.float64)
            current_objective = float(objective_trial)
            gamma_S_proposal = gamma_trial * expand
            accepted_step_size = float(gamma_trial)
            backtracking_trials = trial_index + 1

        monotone_tolerance = 1e-10 * max(1.0, abs(objective_before))
        if current_objective > objective_before + monotone_tolerance:
            raise RuntimeError("certificate-balanced micro-update increased objective")

        current_diagnostics = evaluate_stationarity()
        updates.append(
            CertificateBalancedUpdate(
                update_index=update_index,
                block=block,
                reason=reason,
                B_certificate_before=scores["B"],
                A_certificate_before=scores["A"],
                S_certificate_before=scores["S"],
                total_certificate_before=float(max(scores.values())),
                B_certificate_step_size=B_certificate_step_before,
                S_certificate_step_size=S_certificate_step_before,
                objective_before=objective_before,
                objective_after=float(current_objective),
                accepted_step_size=accepted_step_size,
                backtracking_trials=backtracking_trials,
            )
        )
        block_counts[block] += 1
        last_selected[block] = update_index
        counters["accepted_micro_updates"] += 1
        objectives.append(float(current_objective))
        stationarity.append(float(current_diagnostics.total))
        block_stationarity.append(current_diagnostics)

    # Recompute both the objective and every block certificate at the same
    # final state.  This prevents a last-updated-block residual from being
    # mistaken for complete stationarity.
    final_objective = evaluate_objective()
    consistency_tolerance = 1e-11 * max(1.0, abs(final_objective))
    if abs(final_objective - current_objective) > consistency_tolerance:
        raise RuntimeError("final objective does not match the accepted trace")
    objectives[-1] = float(final_objective)
    final_diagnostics = evaluate_stationarity()
    stationarity[-1] = float(final_diagnostics.total)
    block_stationarity[-1] = final_diagnostics
    converged = bool(final_diagnostics.total <= tol)

    reconstruction = np.asarray(((X_array - S) @ B) @ A.T, dtype=np.float64)
    counts = CertificateBalancedBlockCounts(
        B=block_counts["B"],
        A=block_counts["A"],
        S=block_counts["S"],
    )
    work = CertificateBalancedWorkCounters(**counters)
    return CertificateBalancedCleanedEncoderResult(
        A=A,
        B=B,
        S=S,
        reconstruction=reconstruction,
        objectives=tuple(objectives),
        stationarity=tuple(stationarity),
        block_stationarity=tuple(block_stationarity),
        updates=tuple(updates),
        block_counts=counts,
        work=work,
        converged=converged,
        n_updates=len(updates),
        alpha=alpha,
        kappa=kappa,
        tol=tol,
        max_block_delay=max_block_delay,
        B_certificate_step_size=float(certificate_steps["B"]),
        S_certificate_step_size=float(certificate_steps["S"]),
        status="converged" if converged else "maximum micro-updates reached",
    )


__all__ = [
    "CertificateBalancedBlockCounts",
    "CertificateBalancedCleanedEncoderResult",
    "CertificateBalancedUpdate",
    "CertificateBalancedWorkCounters",
    "fit_certificate_balanced_cleaned_encoder",
]

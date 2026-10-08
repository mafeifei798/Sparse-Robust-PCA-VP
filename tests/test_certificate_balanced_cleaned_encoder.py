from __future__ import annotations

import inspect

import numpy as np
import pytest

import rspca.certificate_balanced_cleaned_encoder as balanced_module
from rspca.certificate_balanced_cleaned_encoder import (
    _canonical_certificate_step_sizes,
    fit_certificate_balanced_cleaned_encoder,
)
from rspca.cleaned_encoder_v2 import (
    cleaned_encoder_block_stationarity,
    cleaned_encoder_objective,
    cleaned_encoder_reconstruction,
    fit_cleaned_encoder_robust_vp_spca_v2,
)


def _corrupted_data(seed: int = 41) -> np.ndarray:
    rng = np.random.default_rng(seed)
    low_rank = rng.normal(size=(30, 2)) @ rng.normal(size=(2, 10))
    values = low_rank + 0.02 * rng.normal(size=low_rank.shape)
    indices = rng.choice(values.size, size=14, replace=False)
    values.flat[indices] += rng.choice((-5.0, 5.0), size=indices.size)
    return np.asarray(values, dtype=np.float64)


def _assert_bounded_selection_gaps(
    choices: tuple[str, ...],
    max_block_delay: int,
) -> None:
    for block in ("B", "A", "S"):
        positions = [index for index, choice in enumerate(choices) if choice == block]
        assert positions
        assert positions[0] < 3
        if len(positions) > 1:
            assert max(np.diff(positions)) <= max_block_delay


def test_balanced_solver_keeps_exact_v2_objective_and_monotonicity() -> None:
    X = _corrupted_data()
    result = fit_certificate_balanced_cleaned_encoder(
        X,
        2,
        alpha=0.04,
        kappa=0.35,
        max_updates=90,
        tol=1e-10,
        max_block_delay=6,
    )

    assert result.n_updates == len(result.updates)
    assert len(result.objectives) == result.n_updates + 1
    assert len(result.stationarity) == result.n_updates + 1
    assert len(result.block_stationarity) == result.n_updates + 1
    assert np.max(np.diff(result.objectives), initial=-np.inf) <= 1e-9
    assert all(
        update.objective_after
        <= update.objective_before + 1e-10 * max(1.0, abs(update.objective_before))
        for update in result.updates
    )

    np.testing.assert_allclose(result.A.T @ result.A, np.eye(2), atol=1e-11)
    np.testing.assert_array_equal(
        result.reconstruction,
        cleaned_encoder_reconstruction(X, result.A, result.B, result.S),
    )
    direct = 0.5 * float(np.sum(((X - result.S) - result.reconstruction) ** 2))
    direct += result.alpha * float(np.sum(np.abs(result.B)))
    direct += result.kappa * float(np.sum(np.abs(result.S)))
    assert result.objectives[-1] == pytest.approx(direct, rel=1e-11, abs=1e-11)
    assert result.objectives[-1] == pytest.approx(
        cleaned_encoder_objective(
            X,
            result.A,
            result.B,
            result.S,
            alpha=result.alpha,
            kappa=result.kappa,
        ),
        rel=1e-11,
        abs=1e-11,
    )

    final = result.block_stationarity[-1]
    assert final.total == max(
        final.B_prox_mapping,
        final.A_riemannian_kkt,
        final.A_global_optimality_gap,
        final.A_orthogonality,
        final.S_prox_mapping,
    )
    assert result.stationarity[-1] == final.total


def test_greedy_choices_and_bounded_delay_are_auditable() -> None:
    X = _corrupted_data(7)
    delay = 3
    result = fit_certificate_balanced_cleaned_encoder(
        X,
        2,
        alpha=0.03,
        kappa=0.25,
        max_updates=36,
        tol=1e-15,
        max_block_delay=delay,
    )

    assert result.n_updates == 36
    assert result.block_counts.total == result.n_updates
    assert result.block_counts.B > 0
    assert result.block_counts.A > 0
    assert result.block_counts.S > 0
    assert result.block_choices[:3] == ("B", "A", "S")
    _assert_bounded_selection_gaps(result.block_choices, delay)
    assert "bounded_delay" in result.selection_reasons

    for update in result.updates:
        certificates = {
            "B": update.B_certificate_before,
            "A": update.A_certificate_before,
            "S": update.S_certificate_before,
        }
        assert update.total_certificate_before == max(certificates.values())
        if update.reason == "greedy":
            assert certificates[update.block] == max(certificates.values())
        assert update.B_certificate_step_size > 0.0
        assert update.S_certificate_step_size > 0.0


def test_canonical_certificate_steps_depend_only_on_current_state() -> None:
    X = _corrupted_data(13)
    rng = np.random.default_rng(8)
    B = rng.normal(size=(X.shape[1], 2))
    S = 0.1 * rng.normal(size=X.shape)
    eta_B, eta_S = _canonical_certificate_step_sizes(X, B, S)

    expected_B = 1.0 / max(
        float(np.linalg.norm(X - S, ord="fro")) ** 2,
        1.0,
    )
    expected_S = 1.0 / (1.0 + float(np.linalg.norm(B, ord=2))) ** 2
    assert eta_B == pytest.approx(expected_B, rel=1e-15)
    assert eta_S == pytest.approx(expected_S, rel=1e-15)


def test_fixed_warmup_reproduces_one_archived_v2_cycle() -> None:
    X = _corrupted_data(23)
    cyclic = fit_cleaned_encoder_robust_vp_spca_v2(
        X,
        2,
        alpha=0.04,
        kappa=0.35,
        max_iter=1,
        tol=1e-15,
    )
    balanced = fit_certificate_balanced_cleaned_encoder(
        X,
        2,
        alpha=0.04,
        kappa=0.35,
        max_updates=3,
        tol=1e-15,
        max_block_delay=5,
    )

    assert balanced.block_choices == ("B", "A", "S")
    assert balanced.updates[1].backtracking_trials == 0
    np.testing.assert_array_equal(balanced.A, cyclic.A)
    np.testing.assert_array_equal(balanced.B, cyclic.B)
    np.testing.assert_array_equal(balanced.S, cyclic.S)
    assert balanced.objectives[-1] == cyclic.objectives[-1]


def test_zero_cleaned_matrix_does_not_report_false_initial_stationarity() -> None:
    X = np.zeros((7, 4), dtype=np.float64)
    initial_B = np.eye(4, 2, dtype=np.float64)
    alpha = 0.1
    cyclic = fit_cleaned_encoder_robust_vp_spca_v2(
        X,
        2,
        alpha=alpha,
        kappa=0.4,
        initial_B=initial_B,
        max_iter=1,
        tol=1e-15,
    )
    balanced = fit_certificate_balanced_cleaned_encoder(
        X,
        2,
        alpha=alpha,
        kappa=0.4,
        initial_B=initial_B,
        max_updates=3,
        tol=1e-15,
        max_block_delay=5,
    )

    assert balanced.stationarity[0] > 0.0
    assert balanced.n_updates > 0
    assert balanced.block_choices == ("B", "A", "S")
    np.testing.assert_array_equal(balanced.A, cyclic.A)
    np.testing.assert_array_equal(balanced.B, cyclic.B)
    np.testing.assert_array_equal(balanced.S, cyclic.S)
    assert balanced.objectives[-1] == cyclic.objectives[-1]
    assert balanced.objectives[-1] == 0.0


def test_small_rank_one_problem_reduces_stationarity_and_converges() -> None:
    spatial = np.array([1.0, 2.0, -1.5, 0.75, 3.0])[:, None]
    temporal = np.array([1.0, -0.5, 0.25, 0.75])[None, :]
    X = spatial @ temporal
    initial_B = np.array([[0.2], [0.8], [-0.3], [0.1]])
    result = fit_certificate_balanced_cleaned_encoder(
        X,
        1,
        alpha=0.0,
        kappa=10.0,
        initial_B=initial_B,
        max_updates=30,
        tol=1e-10,
        max_block_delay=6,
    )

    assert result.stationarity[0] > 1e-6
    assert result.stationarity[-1] < result.stationarity[0]
    assert result.converged
    assert result.stationarity[-1] <= result.tol
    np.testing.assert_allclose(result.reconstruction, X, atol=1e-10)


def test_final_stationarity_and_work_counters_recompute() -> None:
    X = _corrupted_data(19)
    result = fit_certificate_balanced_cleaned_encoder(
        X,
        2,
        alpha=0.05,
        kappa=0.4,
        max_updates=24,
        tol=1e-14,
        max_block_delay=5,
    )
    recomputed = cleaned_encoder_block_stationarity(
        X,
        result.A,
        result.B,
        result.S,
        alpha=result.alpha,
        kappa=result.kappa,
        B_step_size=result.B_certificate_step_size,
        S_step_size=result.S_certificate_step_size,
    )
    assert recomputed == result.block_stationarity[-1]

    assert result.work.accepted_micro_updates == result.n_updates
    assert result.work.full_stationarity_evaluations == result.n_updates + 2
    assert result.work.B_update_gradient_evaluations == result.block_counts.B
    assert result.work.S_update_gradient_evaluations == result.block_counts.S
    assert result.work.A_polar_factorizations == result.block_counts.A + 1
    assert result.work.B_prox_backtracking_trials >= result.block_counts.B
    assert result.work.S_prox_backtracking_trials >= result.block_counts.S
    assert result.work.objective_evaluations >= result.n_updates + 2


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"alpha": -0.1}, ValueError, "alpha"),
        ({"kappa": 0.0}, ValueError, "kappa"),
        ({"max_updates": 0}, ValueError, "max_updates"),
        ({"max_updates": 2}, ValueError, "max_updates"),
        ({"max_updates": True}, TypeError, "max_updates"),
        ({"tol": 0.0}, ValueError, "tol"),
        ({"max_block_delay": 2}, ValueError, "max_block_delay"),
        ({"max_block_delay": 3.5}, TypeError, "max_block_delay"),
        ({"B_step_size": 0.0}, ValueError, "B_step_size"),
        ({"S_step_size": 0.0}, ValueError, "S_step_size"),
        ({"backtracking_expand": 0.9}, ValueError, "backtracking_expand"),
        ({"initial_S": np.zeros((2, 2))}, ValueError, "initial_S"),
    ],
)
def test_balanced_solver_validates_inputs(
    kwargs: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    with pytest.raises(exception, match=match):
        fit_certificate_balanced_cleaned_encoder(
            _corrupted_data(3),
            2,
            **kwargs,
        )


def test_module_is_solver_only_and_not_a_video_runner() -> None:
    source = inspect.getsource(balanced_module)
    assert "experiments." not in source
    assert "corruption_support" not in source
    assert "outlier_detection" not in source
    assert "clean_truth" not in source

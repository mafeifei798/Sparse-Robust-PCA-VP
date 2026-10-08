from __future__ import annotations

import inspect
from dataclasses import asdict, replace

import numpy as np
import pytest
from scipy import linalg

import rspca.accelerated as accelerated_module
import rspca.randomized as randomized_module
import rspca.vp as vp_module
from rspca.randomized import (
    RandomizedVPSPCAWorkspace,
    certified_randomized_vp_spca,
    certified_randomized_vp_spca_from_workspace,
    fit_randomized_vp_spca,
    fit_randomized_vp_spca_from_workspace,
    prepare_randomized_vp_spca_workspace,
    randomized_range_finder,
    randomized_spca_vp,
    sketch_residual_diagnostic,
)
from rspca.vp import vp_objective


def _data(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.normal(size=(65, 7))
    right = rng.normal(size=(22, 7))
    X = left @ np.diag([9.0, 7.0, 5.0, 3.0, 2.0, 1.0, 0.4]) @ right.T
    X += 0.15 * rng.normal(size=X.shape)
    return X - X.mean(axis=0, keepdims=True)


def _legacy_reference_fit(
    X: np.ndarray,
    n_components: int,
    *,
    alpha: float,
    max_iter: int,
    tol: float,
    step_size: float,
    initial_B: np.ndarray,
    backtracking: bool = True,
    verbose: bool = False,
) -> vp_module.VPSPCAResult:
    """Independent pre-cache loop injected at randomized solver boundaries."""

    del n_components, verbose
    B = initial_B.copy()
    gamma = step_size
    A = vp_module.procrustes_factor(X, B)
    smooth = vp_module.smooth_value(X, A, B)
    objectives: list[float] = []
    stationarity: list[float] = []
    step_sizes: list[float] = []
    converged = False

    for _ in range(max_iter):
        gradient = vp_module.vp_gradient(X, A, B)
        gamma_trial = gamma
        for _ in range(61):
            B_trial = vp_module.soft_threshold(
                B - gamma_trial * gradient,
                gamma_trial * alpha,
            )
            A_trial = vp_module.procrustes_factor(X, B_trial, reference=A)
            smooth_trial = vp_module.smooth_value(X, A_trial, B_trial)
            difference = B_trial - B
            majorizer = smooth + float(np.sum(gradient * difference))
            majorizer += 0.5 * float(np.sum(difference * difference)) / gamma_trial
            if smooth_trial <= majorizer + 1e-12 * max(1.0, abs(smooth)):
                break
            if not backtracking:
                break
            gamma_trial *= 0.5
        else:  # pragma: no cover - same finite-loop guard as production
            raise RuntimeError("VP-SPCA backtracking failed to terminate")

        objective = smooth_trial + alpha * float(np.sum(np.abs(B_trial)))
        mapping = vp_module.proximal_gradient_mapping(
            X,
            A_trial,
            B_trial,
            alpha=alpha,
            step_size=gamma_trial,
        )
        criterion = float(linalg.norm(mapping, ord="fro", check_finite=False))
        objectives.append(objective)
        stationarity.append(criterion)
        step_sizes.append(gamma_trial)
        B, A, smooth = B_trial, A_trial, smooth_trial
        gamma = gamma_trial
        if criterion <= tol:
            converged = True
            break

    return vp_module.VPSPCAResult(
        A=A,
        B=B,
        objectives=objectives,
        stationarity=stationarity,
        step_sizes=step_sizes,
        converged=converged,
        n_iter=len(objectives),
        alpha=alpha,
        status="converged" if converged else "maximum iterations reached",
    )


def _assert_randomized_numerics_array_equal(
    actual: randomized_module.RandomizedVPSPCAResult,
    expected: randomized_module.RandomizedVPSPCAResult,
) -> None:
    for field in (
        "A",
        "B",
        "objectives",
        "stationarity",
        "step_sizes",
        "Q",
        "sketch_objectives",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(actual, field)),
            np.asarray(getattr(expected, field)),
        )
    for field in (
        "converged",
        "sketch_converged",
        "full_converged",
        "n_iter",
        "alpha",
        "status",
        "sketch_residual",
        "sketch_residual_relative",
        "full_objective",
        "full_stationarity",
        "full_diagnostic_step_size",
        "full_diagnostic_step_policy",
        "used_full_gradient",
        "sketch_iterations",
        "full_refinement_iterations",
    ):
        assert getattr(actual, field) == getattr(expected, field)
    assert (
        None if actual.certificate is None else asdict(actual.certificate)
    ) == (None if expected.certificate is None else asdict(expected.certificate))


def test_range_finder_reorthogonalizes_and_residual_is_exact() -> None:
    X = _data()
    Q = randomized_range_finder(X, 3, oversample=2, n_iter=3, random_state=11)
    np.testing.assert_allclose(Q.T @ Q, np.eye(5), atol=1e-11)

    residual, relative = sketch_residual_diagnostic(X, Q)
    direct = np.linalg.norm(X - Q @ (Q.T @ X))
    assert residual == pytest.approx(direct, rel=1e-10, abs=1e-10)
    assert relative == pytest.approx(direct / np.linalg.norm(X), rel=1e-10)
    Q_again = randomized_range_finder(X, 3, oversample=2, n_iter=3, random_state=11)
    np.testing.assert_allclose(np.abs(Q), np.abs(Q_again), atol=1e-12)


def test_projection_residual_does_not_cancel_near_exact_capture() -> None:
    epsilon = 1e-8
    X = np.array([[1.0, 2.0, 3.0], [epsilon, -epsilon, epsilon]])
    Q = np.array([[1.0], [0.0]])

    residual, relative = sketch_residual_diagnostic(X, Q)
    direct = np.linalg.norm(X - Q @ (Q.T @ X), ord="fro")

    assert direct == pytest.approx(np.sqrt(3.0) * epsilon, rel=1e-15)
    assert residual == pytest.approx(direct, rel=1e-15)
    assert residual > 0.0
    assert relative > 0.0


def test_sketch_only_full_diagnostic_avoids_exact_full_matrix_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = _data(17)
    original_svdvals = vp_module.linalg.svdvals
    seen_shapes: list[tuple[int, int]] = []

    def reject_full_matrix_svd(matrix, *args, **kwargs):
        shape = tuple(np.asarray(matrix).shape)
        seen_shapes.append(shape)
        if shape == X.shape:
            raise AssertionError(
                "randomized diagnostic requested an exact full-data SVD"
            )
        return original_svdvals(matrix, *args, **kwargs)

    monkeypatch.setattr(vp_module.linalg, "svdvals", reject_full_matrix_svd)
    result = fit_randomized_vp_spca(
        X,
        3,
        alpha=0.1,
        oversample=2,
        n_power_iter=1,
        max_iter=5,
        random_state=4,
    )

    exact_step = 1.0 / float(np.linalg.svd(X, compute_uv=False)[0] ** 2)
    assert X.shape not in seen_shapes
    assert result.full_diagnostic_step_policy == (
        "sketch_spectral_plus_projection_residual_upper_bound"
    )
    assert 0.0 < result.full_diagnostic_step_size <= exact_step * (1.0 + 1e-12)


def test_certificate_gap_bound_remains_positive_near_exact_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epsilon = 1e-8
    X = np.array([[1.0, 2.0, 3.0], [epsilon, -epsilon, epsilon]])
    Q = np.array([[1.0], [0.0]])
    monkeypatch.setattr(
        randomized_module,
        "_randomized_range_finder_prevalidated",
        lambda *args, **kwargs: Q.copy(),
    )

    result = certified_randomized_vp_spca(
        X,
        1,
        alpha=0.1,
        oversample=0,
        n_power_iter=0,
        max_iter=2,
        tol=1e-8,
        residual_tolerance=1.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
        random_state=1,
    )

    certificate = result.certificate
    assert certificate is not None
    assert certificate.projection_residual == pytest.approx(
        np.sqrt(3.0) * epsilon, rel=1e-15
    )
    assert certificate.loss_gap_upper_bound > 0.0


def test_full_gradient_flag_and_requested_full_finalization_are_effective() -> None:
    X = _data()
    common = dict(
        n_components=3,
        alpha=0.15,
        oversample=0,
        n_power_iter=0,
        max_iter=35,
        tol=1e-6,
        random_state=9,
    )
    sketch = fit_randomized_vp_spca(X, **common)
    full = fit_randomized_vp_spca(X, **common, use_full_gradient=True)
    refined = fit_randomized_vp_spca(X, **common, finalize_full_iters=6)

    assert not sketch.used_full_gradient
    assert full.used_full_gradient
    assert not np.allclose(sketch.B, full.B)
    assert full.full_objective < sketch.full_objective
    assert refined.full_refinement_iterations == 6
    assert refined.sketch_iterations == 35
    assert refined.full_objective <= sketch.full_objective
    assert sketch.full_objective == pytest.approx(
        vp_objective(X, sketch.A, sketch.B, sketch.alpha), rel=1e-12
    )


def test_workspace_reuses_alpha_independent_full_data_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = _data(23)
    original_range_finder = randomized_module._randomized_range_finder_prevalidated
    range_calls = 0

    def counted_range_finder(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal range_calls
        range_calls += 1
        return original_range_finder(*args, **kwargs)

    monkeypatch.setattr(
        randomized_module,
        "_randomized_range_finder_prevalidated",
        counted_range_finder,
    )
    workspace = prepare_randomized_vp_spca_workspace(
        X,
        3,
        oversample=3,
        n_power_iter=1,
        random_state=19,
    )
    assert isinstance(workspace, RandomizedVPSPCAWorkspace)
    assert workspace.X is X
    assert range_calls == 1

    first = fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=0.08,
        max_iter=8,
        tol=1e-8,
    )
    second = fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=0.16,
        max_iter=8,
        tol=1e-8,
    )

    assert range_calls == 1
    assert not np.allclose(first.B, second.B)
    for phase in (
        "range_finder",
        "compress",
        "residual_diagnostic",
        "initialization",
    ):
        assert first.timings[phase] == 0.0
        assert second.timings[phase] == 0.0
    assert workspace.timings["total"] > 0.0

    direct = fit_randomized_vp_spca(
        X,
        3,
        alpha=0.08,
        oversample=3,
        n_power_iter=1,
        max_iter=8,
        tol=1e-8,
        random_state=19,
    )
    assert range_calls == 2
    np.testing.assert_allclose(first.Q, direct.Q, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(first.B, direct.B, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(first.A, direct.A, atol=1e-13, rtol=1e-13)
    assert first.full_objective == pytest.approx(direct.full_objective, rel=1e-13)


def test_full_refinement_paths_do_not_request_exact_full_matrix_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = _data(29)
    original_svdvals = vp_module.linalg.svdvals
    seen_shapes: list[tuple[int, int]] = []

    def reject_full_matrix_svd(matrix, *args, **kwargs):
        shape = tuple(np.asarray(matrix).shape)
        seen_shapes.append(shape)
        if shape == X.shape:
            raise AssertionError("full refinement requested an exact full-data SVD")
        return original_svdvals(matrix, *args, **kwargs)

    monkeypatch.setattr(vp_module.linalg, "svdvals", reject_full_matrix_svd)
    refined = fit_randomized_vp_spca(
        X,
        3,
        alpha=0.1,
        oversample=2,
        n_power_iter=1,
        max_iter=5,
        finalize_full_iters=3,
        random_state=7,
    )
    certified = certified_randomized_vp_spca(
        X,
        3,
        alpha=0.1,
        oversample=2,
        n_power_iter=1,
        max_iter=5,
        residual_tolerance=0.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=3,
        random_state=7,
    )

    assert X.shape not in seen_shapes
    assert refined.full_refinement_iterations == 3
    assert certified.full_refinement_iterations > 0
    assert refined.full_diagnostic_step_policy == "accepted_full_data_solver_step"
    assert certified.full_diagnostic_step_policy == "accepted_full_data_solver_step"


def test_certified_workspace_matches_direct_entrypoint() -> None:
    X = _data(31)
    workspace = prepare_randomized_vp_spca_workspace(
        X,
        2,
        oversample=2,
        n_power_iter=1,
        random_state=13,
    )
    shared = certified_randomized_vp_spca_from_workspace(
        workspace,
        alpha=0.1,
        max_iter=6,
        tol=1e-5,
        residual_tolerance=1.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
    )
    direct = certified_randomized_vp_spca(
        X,
        2,
        alpha=0.1,
        oversample=2,
        n_power_iter=1,
        max_iter=6,
        tol=1e-5,
        residual_tolerance=1.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
        random_state=13,
    )

    np.testing.assert_allclose(shared.B, direct.B, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(shared.A, direct.A, atol=1e-13, rtol=1e-13)
    assert asdict(shared.certificate) == asdict(direct.certificate)
    assert shared.converged is direct.converged
    assert shared.full_converged is direct.full_converged
    assert shared.status == direct.status
    for result in (shared, direct):
        assert result.certificate is not None
        assert result.converged is result.full_converged
        assert result.full_converged is result.certificate.certified
        assert result.full_stationarity <= (
            result.certificate.full_stationarity_tolerance
        )


def test_explicit_certificate_tolerance_controls_all_final_convergence_fields() -> (
    None
):
    X = _data(32)
    result = certified_randomized_vp_spca(
        X,
        2,
        alpha=0.1,
        oversample=2,
        n_power_iter=1,
        max_iter=1,
        tol=1e-12,
        residual_tolerance=1.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
        random_state=14,
    )

    certificate = result.certificate
    assert certificate is not None
    assert not certificate.refinement_triggered
    assert result.full_stationarity > 1e-12
    assert result.full_stationarity <= certificate.full_stationarity_tolerance
    assert certificate.certified
    assert result.converged
    assert result.full_converged
    assert "passed at certificate tolerance" in result.status


def test_prevalidated_vp_kernel_is_exact_under_backtracking_and_nonconvergence() -> (
    None
):
    rng = np.random.default_rng(88)
    X = rng.normal(size=(70, 25))
    X -= X.mean(axis=0, keepdims=True)
    initial_B, exact_step = vp_module._pca_initial_loadings_and_step_size(X, 3)
    deliberately_large_step = 32.0 * exact_step
    common = dict(
        alpha=0.2,
        max_iter=7,
        tol=1e-14,
        step_size=deliberately_large_step,
        initial_B=initial_B,
        backtracking=True,
        verbose=False,
    )

    assert (
        randomized_module._fit_vp_spca_prevalidated
        is vp_module._fit_vp_spca_prevalidated
    )
    reference = _legacy_reference_fit(X, 3, **common)
    optimized = randomized_module._fit_vp_spca_prevalidated(X, 3, **common)

    assert reference.step_sizes[0] < deliberately_large_step
    assert reference.status == optimized.status == "maximum iterations reached"
    assert reference.n_iter == optimized.n_iter == 7
    assert reference.converged is optimized.converged is False
    for field in ("A", "B", "objectives", "stationarity", "step_sizes"):
        np.testing.assert_array_equal(
            np.asarray(getattr(optimized, field)),
            np.asarray(getattr(reference, field)),
        )


@pytest.mark.parametrize(
    "path",
    [
        "direct_sketch",
        "workspace_sketch",
        "direct_full",
        "workspace_full_refinement",
        "direct_certified_sketch",
        "workspace_certified_refinement",
    ],
)
def test_randomized_paths_are_bitwise_equivalent_to_legacy_vp_core(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    X = _data(404)
    workspace = prepare_randomized_vp_spca_workspace(
        X,
        3,
        oversample=2,
        n_power_iter=1,
        random_state=17,
    )

    def run() -> randomized_module.RandomizedVPSPCAResult:
        if path == "direct_sketch":
            return fit_randomized_vp_spca(
                X,
                3,
                alpha=0.13,
                oversample=2,
                n_power_iter=1,
                max_iter=6,
                tol=1e-12,
                random_state=17,
            )
        if path == "workspace_sketch":
            return fit_randomized_vp_spca_from_workspace(
                workspace,
                alpha=0.13,
                max_iter=6,
                tol=1e-12,
            )
        if path == "direct_full":
            return fit_randomized_vp_spca(
                X,
                3,
                alpha=0.13,
                oversample=2,
                n_power_iter=1,
                max_iter=6,
                tol=1e-12,
                use_full_gradient=True,
                random_state=17,
            )
        if path == "workspace_full_refinement":
            return fit_randomized_vp_spca_from_workspace(
                workspace,
                alpha=0.13,
                max_iter=6,
                tol=1e-12,
                finalize_full_iters=3,
            )
        if path == "direct_certified_sketch":
            return certified_randomized_vp_spca(
                X,
                3,
                alpha=0.13,
                oversample=2,
                n_power_iter=1,
                max_iter=6,
                tol=1e-12,
                residual_tolerance=1.0,
                full_stationarity_tolerance=1e12,
                full_refine_iters=2,
                random_state=17,
            )
        return certified_randomized_vp_spca_from_workspace(
            workspace,
            alpha=0.13,
            max_iter=6,
            tol=1e-12,
            residual_tolerance=0.0,
            full_stationarity_tolerance=1e12,
            full_refine_iters=2,
        )

    optimized = run()
    monkeypatch.setattr(
        randomized_module,
        "_fit_vp_spca_prevalidated",
        _legacy_reference_fit,
    )
    legacy = run()
    _assert_randomized_numerics_array_equal(optimized, legacy)


def test_sketch_and_full_convergence_are_reported_separately() -> None:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 20))
    X -= X.mean(axis=0, keepdims=True)

    result = fit_randomized_vp_spca(
        X,
        2,
        alpha=0.1,
        oversample=0,
        n_power_iter=0,
        max_iter=5,
        tol=0.5,
        random_state=9,
    )

    assert result.sketch_converged is True
    assert result.full_stationarity > 0.5
    assert result.full_converged is False
    assert result.converged is False
    assert "full stationarity diagnostic: failed" in result.status


def test_certified_hybrid_triggers_matrix_free_full_refinement() -> None:
    rng = np.random.default_rng(11)
    X = rng.normal(size=(50, 18))
    X -= X.mean(axis=0, keepdims=True)
    result = certified_randomized_vp_spca(
        X,
        2,
        alpha=0.1,
        oversample=0,
        n_power_iter=0,
        max_iter=20,
        tol=1e-4,
        residual_tolerance=0.01,
        full_stationarity_tolerance=0.5,
        full_refine_iters=25,
        random_state=2,
    )

    certificate = result.certificate
    assert certificate is not None
    assert certificate.refinement_triggered
    assert certificate.loss_gap_upper_bound >= 0.0
    assert certificate.full_stationarity_after < certificate.full_stationarity_before
    assert certificate.certified
    assert result.full_refinement_iterations > 0


def test_certificate_reports_zero_restarts_without_refinement() -> None:
    result = certified_randomized_vp_spca(
        _data(),
        3,
        alpha=0.1,
        oversample=4,
        n_power_iter=2,
        max_iter=5,
        tol=1e-4,
        residual_tolerance=1.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
        random_state=3,
    )

    certificate = result.certificate
    assert certificate is not None
    assert not certificate.refinement_triggered
    assert certificate.monotone_restarts == 0
    assert certificate.adaptive_restarts == 0
    assert certificate.restart_count == 0


def test_certificate_reconciles_refinement_restart_counts_and_serializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = accelerated_module.fit_accelerated_vp_spca

    def fit_with_known_restart_counts(*args: object, **kwargs: object):
        refined = original_fit(*args, **kwargs)
        return replace(
            refined,
            monotone_restarts=4,
            adaptive_restarts=3,
            restart_count=7,
        )

    monkeypatch.setattr(
        accelerated_module,
        "fit_accelerated_vp_spca",
        fit_with_known_restart_counts,
    )
    result = certified_randomized_vp_spca(
        _data(),
        2,
        alpha=0.1,
        oversample=0,
        n_power_iter=0,
        max_iter=3,
        tol=1e-4,
        residual_tolerance=0.0,
        full_stationarity_tolerance=1e12,
        full_refine_iters=2,
        random_state=2,
    )

    certificate = result.certificate
    assert certificate is not None
    assert certificate.refinement_triggered
    assert certificate.monotone_restarts == 4
    assert certificate.adaptive_restarts == 3
    assert certificate.restart_count == 7
    assert certificate.restart_count == (
        certificate.monotone_restarts + certificate.adaptive_restarts
    )
    serialized = asdict(certificate)
    assert serialized["monotone_restarts"] == 4
    assert serialized["adaptive_restarts"] == 3
    assert serialized["restart_count"] == 7


def test_randomized_legacy_wrapper_and_matrix_free_source() -> None:
    X = _data(5)
    B, A, objectives, timings = randomized_spca_vp(
        X,
        2,
        alpha=0.1,
        oversample=2,
        n_iter=1,
        max_iter=3,
        tol=1e-12,
        verbose=False,
        return_timings=True,
    )
    assert A.shape == B.shape == (X.shape[1], 2)
    assert len(objectives) == 3
    assert {"range_finder", "compress", "iter"} <= timings.keys()

    source = inspect.getsource(randomized_module)
    assert "X.T @ X" not in source
    assert "X_array.T @ X_array" not in source

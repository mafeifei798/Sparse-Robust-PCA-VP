from __future__ import annotations

import numpy as np
import pytest

from rspca.datasets import (
    array_sha256,
    canonical_sha256,
    generate_sparse_low_rank,
    make_seed_manifest,
    runtime_versions,
    verify_seed_manifest,
)

from experiments import spca_benchmark
from experiments.spca_benchmark import (
    BenchmarkConfig,
    run_benchmark,
    summarize_reconstruction_errors,
)


def test_sparse_low_rank_generator_is_deterministic_and_reproducible() -> None:
    arguments = dict(
        n_samples=40,
        n_features=24,
        rank=3,
        support_size=4,
        noise_std=0.1,
        corruption_fraction=0.125,
        corruption_scale=5.0,
        corruption_distribution="rademacher",
        seed=91,
    )
    first = generate_sparse_low_rank(**arguments)
    second = generate_sparse_low_rank(**arguments)

    assert np.array_equal(first.observed, second.observed)
    assert np.array_equal(first.feature_support, second.feature_support)
    assert np.array_equal(first.corruption_support, second.corruption_support)
    assert np.allclose(
        first.observed, first.low_rank + first.gaussian_noise + first.corruption
    )
    assert np.allclose(first.uncorrupted, first.low_rank + first.gaussian_noise)
    assert np.count_nonzero(first.feature_support, axis=0).tolist() == [4, 4, 4]
    assert np.count_nonzero(first.corruption_support) == round(0.125 * 40 * 24)
    assert np.array_equal(first.corruption_support, first.corruption != 0.0)
    assert np.linalg.matrix_rank(first.low_rank, tol=1e-9) == 3
    assert first.low_rank.mean(axis=0) == pytest.approx(np.zeros(24), abs=1e-14)
    assert first.gaussian_noise.mean(axis=0) == pytest.approx(np.zeros(24), abs=1e-14)
    assert first.uncorrupted.mean(axis=0) == pytest.approx(np.zeros(24), abs=1e-14)
    actual_increment = first.observed - first.uncorrupted
    assert actual_increment == pytest.approx(first.corruption, abs=1e-14)
    assert np.array_equal(actual_increment != 0.0, first.corruption_support)


def test_runtime_versions_records_numerical_environment() -> None:
    versions = runtime_versions()

    assert versions["python"]
    assert versions["numpy"] == np.__version__
    assert versions["scipy"]
    assert versions["vp_rspca"]
    assert versions["platform"]
    assert versions["machine"]


def test_post_corruption_centering_densifies_cellwise_support_counterexample() -> None:
    n_samples = 12
    n_features = 8
    dataset = generate_sparse_low_rank(
        n_samples=n_samples,
        n_features=n_features,
        rank=2,
        support_size=2,
        noise_std=0.1,
        corruption_fraction=1.0 / (n_samples * n_features),
        corruption_scale=4.0,
        corruption_distribution="rademacher",
        seed=12,
    )

    observed_centered = dataset.observed - dataset.observed.mean(axis=0, keepdims=True)
    clean_centered = dataset.uncorrupted - dataset.uncorrupted.mean(
        axis=0, keepdims=True
    )
    centered_increment = observed_centered - clean_centered

    assert np.count_nonzero(dataset.corruption_support) == 1
    assert np.count_nonzero(np.abs(centered_increment) > 1e-14) == n_samples
    assert np.any((np.abs(centered_increment) > 1e-14) & ~dataset.corruption_support)


def test_matched_sparsity_requires_predeclared_target() -> None:
    with pytest.raises(ValueError, match="explicit pre-declared target_sparsity"):
        run_benchmark(
            BenchmarkConfig(
                n_samples=20,
                n_features=10,
                rank=2,
                support_fraction=0.2,
                selection="matched-sparsity",
                target_sparsity=None,
                repeats=1,
                alphas=(0.1,),
                max_iter=2,
            )
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"rank": 20}, "smaller than n_samples"),
        ({"alphas": (0.1, float("nan"))}, "finite"),
        ({"alphas": (0.1, 0.1)}, "duplicates"),
        ({"support_fraction": float("nan")}, "support_fraction"),
        ({"support_fraction": 0.6}, "disjoint"),
        ({"corruption_fraction": 0.1, "corruption_scale": 0.0}, "corruption_scale"),
        ({"oversample": 1.5}, "oversample must be an integer"),
        ({"randomized_finalize_full_iters": -1}, "non-negative"),
        ({"tol": 0.0}, "tol must be positive"),
    ],
)
def test_benchmark_config_fails_before_fitting_invalid_controls(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "n_samples": 20,
        "n_features": 10,
        "rank": 2,
        "support_fraction": 0.2,
        "repeats": 1,
        "alphas": (0.1,),
        "max_iter": 2,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        run_benchmark(BenchmarkConfig(**values))  # type: ignore[arg-type]


def test_matched_sparsity_labels_thresholded_pca_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spca_benchmark, "_fit_candidates", lambda *_: [])
    rows, _ = run_benchmark(
        BenchmarkConfig(
            n_samples=20,
            n_features=10,
            rank=2,
            support_fraction=0.2,
            selection="matched-sparsity",
            target_sparsity=0.8,
            repeats=1,
            alphas=(0.1,),
            max_iter=2,
        )
    )

    assert rows[0]["method"] == "thresholded_pca_matched_sparsity"
    assert rows[0]["loading_postprocessing"] == (
        "hard_threshold_to_predeclared_target_sparsity"
    )


def test_benchmark_outlier_budget_is_declared_before_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(spca_benchmark, "_fit_candidates", lambda *_: [])
    config = BenchmarkConfig(
        n_samples=20,
        n_features=10,
        rank=2,
        support_fraction=0.2,
        corruption_fraction=0.075,
        repeats=1,
        alphas=(0.1,),
        max_iter=2,
    )
    rows, manifest = run_benchmark(config)

    assert len(rows) == 1
    assert rows[0]["outlier_selection_policy"] == (
        "predeclared_corruption_fraction_top_k"
    )
    assert rows[0]["outlier_budget_fraction"] == pytest.approx(0.075)
    assert rows[0]["outlier_budget_k"] == round(0.075 * 20 * 10)
    assert rows[0]["centering_policy"] == (
        "none_dgp_components_column_centered_before_corruption"
    )
    assert rows[0]["runtime_comparison_scope"] == (
        "exploratory_not_matvec_or_phase_matched"
    )
    assert rows[0]["peak_memory_scope"] == "not_measured"
    assert len(rows[0]["run_data_sha256"]) == 64
    assert len(rows[0]["observed_sha256"]) == 64
    assert manifest["run_provenance"][0]["run_data_sha256"] == (
        rows[0]["run_data_sha256"]
    )
    assert verify_seed_manifest(manifest)
    assert {"python", "rspca", "numpy", "scipy"} == set(manifest["software_versions"])


def test_certified_benchmark_restart_diagnostics_are_not_aliased() -> None:
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(14, 9))
    matrix -= matrix.mean(axis=0, keepdims=True)
    config = BenchmarkConfig(
        n_samples=14,
        n_features=9,
        rank=2,
        support_fraction=0.2,
        alphas=(0.1,),
        max_iter=2,
        oversample=2,
        n_power_iter=1,
        randomized_finalize_full_iters=1,
        certified_residual_tolerance=0.0,
        certified_full_refine_iters=2,
        repeats=1,
    )

    candidates = spca_benchmark._fit_candidates(matrix, config, seed=6)
    certified = next(
        candidate
        for candidate in candidates
        if candidate.method == "certified_randomized_vp_spca"
    )

    assert certified.diagnostics["restart_count"] == (
        certified.diagnostics["monotone_restarts"]
        + certified.diagnostics["adaptive_restarts"]
    )


def test_benchmark_does_not_report_sketch_convergence_as_full_convergence() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(60, 20))
    matrix -= matrix.mean(axis=0, keepdims=True)
    config = BenchmarkConfig(
        n_samples=60,
        n_features=20,
        rank=2,
        support_fraction=0.1,
        alphas=(0.1,),
        max_iter=5,
        tol=0.5,
        oversample=0,
        n_power_iter=0,
        randomized_finalize_full_iters=0,
        certified_full_refine_iters=2,
        repeats=1,
    )

    candidates = spca_benchmark._fit_candidates(matrix, config, seed=9)
    randomized = next(
        candidate
        for candidate in candidates
        if candidate.method == "randomized_vp_spca"
    )

    assert randomized.diagnostics["sketch_converged"] is True
    assert randomized.diagnostics["full_converged"] is False
    assert randomized.converged is False


def test_alpha_grid_summary_never_pools_distinct_alphas() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "method": "vp",
            "alpha": 0.1,
            "relative_reconstruction_error": 1.0,
        },
        {
            "run": 1,
            "seed": 11,
            "method": "vp",
            "alpha": 0.1,
            "relative_reconstruction_error": 3.0,
        },
        {
            "run": 0,
            "seed": 10,
            "method": "vp",
            "alpha": 1.0,
            "relative_reconstruction_error": 10.0,
        },
        {
            "run": 0,
            "seed": 10,
            "method": "pca",
            "alpha": None,
            "relative_reconstruction_error": 2.0,
        },
        {
            "run": 1,
            "seed": 11,
            "method": "pca",
            "alpha": None,
            "relative_reconstruction_error": 4.0,
        },
    ]
    fields, summary = summarize_reconstruction_errors(rows, "alpha-grid")

    assert fields == ("method", "alpha")
    keyed = {(row["method"], row["alpha"]): row for row in summary}
    assert keyed[("vp", 0.1)]["n"] == 2
    assert keyed[("vp", 0.1)]["mean_relative_reconstruction_error"] == 2.0
    assert keyed[("vp", 1.0)]["n"] == 1
    assert keyed[("vp", 1.0)]["mean_relative_reconstruction_error"] == 10.0
    assert keyed[("vp", 1.0)]["paired_error_difference_ci95_low"] is None
    assert keyed[("vp", 1.0)]["paired_error_difference_ci95_status"] == (
        "insufficient_pairs"
    )
    assert keyed[("vp", 0.1)]["n_pairs"] == 2
    assert keyed[("vp", 0.1)]["mean_paired_error_difference_vs_pca"] == -1.0
    assert keyed[("vp", 0.1)]["paired_error_difference_ci95_status"] == "ok"


def test_reconstruction_summary_pairs_by_run_and_seed_not_row_order() -> None:
    rows = [
        {
            "run": 0,
            "seed": 10,
            "method": "pca",
            "alpha": None,
            "relative_reconstruction_error": 0.0,
        },
        {
            "run": 1,
            "seed": 11,
            "method": "pca",
            "alpha": None,
            "relative_reconstruction_error": 100.0,
        },
        {
            "run": 1,
            "seed": 11,
            "method": "vp",
            "alpha": 0.1,
            "relative_reconstruction_error": 1.0,
        },
        {
            "run": 0,
            "seed": 10,
            "method": "vp",
            "alpha": 0.1,
            "relative_reconstruction_error": 101.0,
        },
    ]

    _, summary = summarize_reconstruction_errors(rows, "alpha-grid")
    vp = next(row for row in summary if row["method"] == "vp")

    assert vp["n_pairs"] == 2
    assert vp["mean_paired_error_difference_vs_pca"] == pytest.approx(1.0)
    assert vp["std_paired_error_difference_vs_pca"] == pytest.approx(np.sqrt(20000.0))
    assert vp["paired_error_difference_ci95_low"] < 1.0
    assert vp["paired_error_difference_ci95_high"] > 1.0


def test_generator_uses_samples_by_features_orientation() -> None:
    dataset = generate_sparse_low_rank(
        n_samples=31,
        n_features=20,
        rank=2,
        support_size=5,
        corruption_fraction=0.0,
        seed=2,
    )

    assert dataset.observed.shape == (31, 20)
    assert dataset.loadings.shape == (20, 2)
    assert dataset.scores.shape == (31, 2)
    assert dataset.feature_support.shape == (20, 2)


@pytest.mark.parametrize("distribution", ["gaussian", "laplace", "rademacher"])
def test_supported_corruption_distributions(distribution: str) -> None:
    dataset = generate_sparse_low_rank(
        n_samples=20,
        n_features=12,
        rank=2,
        support_size=3,
        corruption_fraction=0.1,
        corruption_distribution=distribution,
        seed=7,
    )
    assert np.count_nonzero(dataset.corruption_support) == 24


def test_generator_rejects_impossible_disjoint_support() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        generate_sparse_low_rank(
            n_samples=20,
            n_features=10,
            rank=3,
            support_size=4,
        )


def test_seed_manifest_is_stable_unique_and_self_verifying() -> None:
    first = make_seed_manifest(42, 5, namespace="benchmark", metadata={"rank": 3})
    second = make_seed_manifest(42, 5, namespace="benchmark", metadata={"rank": 3})

    assert first == second
    assert len(first["seeds"]) == len(set(first["seeds"])) == 5
    assert verify_seed_manifest(first)

    tampered = dict(first)
    tampered["base_seed"] = 43
    assert not verify_seed_manifest(tampered)


def test_canonical_hash_ignores_mapping_insertion_order() -> None:
    assert canonical_sha256({"b": 2, "a": [1, 3]}) == canonical_sha256(
        {"a": [1, 3], "b": 2}
    )


def test_array_hash_captures_shape_dtype_and_content() -> None:
    matrix = np.arange(6, dtype=np.float64).reshape(2, 3)

    assert array_sha256(matrix) == array_sha256(matrix.copy())
    assert array_sha256(matrix) != array_sha256(matrix.reshape(3, 2))
    assert array_sha256(matrix) != array_sha256(matrix.astype(np.float32))

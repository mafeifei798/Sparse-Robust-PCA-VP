from __future__ import annotations

import csv
import json
import sys

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

from experiments import genomics_1kg, spca_benchmark
from experiments.genomics_1kg import (
    GenomicsConfig,
    GenotypeDataset,
    fit_genotype_preprocessor,
    load_genotypes,
    run_genomics,
    stratified_train_test_split,
    support_overlap_diagnostics,
    transform_genotypes,
)
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


def test_training_preprocessor_does_not_use_held_out_values() -> None:
    train = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.0, np.nan],
            [1.0, 1.0, 2.0],
        ]
    )
    held_out = np.array([[101.0, 99.0, 20.0]])
    preprocessor = fit_genotype_preprocessor(train)
    transformed = transform_genotypes(held_out, preprocessor)

    assert preprocessor.means == pytest.approx([1.0, 1.0, 1.0])
    assert preprocessor.retained_features.tolist() == [True, False, True]
    assert transformed.shape == (1, 2)
    assert transformed[0, 0] > 50.0


def test_stratified_split_is_deterministic_and_keeps_every_population() -> None:
    labels = np.asarray(["A"] * 5 + ["B"] * 5 + ["C"] * 5)
    train_first, test_first = stratified_train_test_split(labels, 0.2, 11)
    train_second, test_second = stratified_train_test_split(labels, 0.2, 11)

    assert np.array_equal(train_first, train_second)
    assert np.array_equal(test_first, test_second)
    assert set(labels[train_first]) == {"A", "B", "C"}
    assert set(labels[test_first]) == {"A", "B", "C"}
    assert set(train_first).isdisjoint(test_first)


def test_dense_support_jaccard_is_flagged_and_chance_adjusted() -> None:
    base = np.ones(100, dtype=bool)
    bootstrap = np.ones(100, dtype=bool)
    base[0] = False
    bootstrap[1] = False
    diagnostics = support_overlap_diagnostics(base, bootstrap)

    assert diagnostics["support_jaccard"] > 0.95
    assert diagnostics["fixed_support_expected_jaccard"] > 0.95
    assert diagnostics["chance_model"] == (
        "independent_uniform_fixed_size_supports_hypergeometric"
    )
    assert diagnostics["chance_adjusted_jaccard"] < 0.1
    assert diagnostics["support_degenerate"] is True
    assert diagnostics["support_degeneracy_reason"] == "near_dense_support"


def test_fixed_support_jaccard_uses_exact_hypergeometric_expectation() -> None:
    diagnostics = support_overlap_diagnostics(
        np.asarray([True, False]),
        np.asarray([True, False]),
    )

    assert diagnostics["fixed_support_expected_jaccard"] == pytest.approx(0.5)
    assert diagnostics["support_jaccard"] == 1.0
    assert diagnostics["chance_adjusted_jaccard"] == 1.0
    assert diagnostics["support_degenerate"] is False


def test_empty_support_jaccard_is_explicitly_degenerate() -> None:
    diagnostics = support_overlap_diagnostics(
        np.zeros(5, dtype=bool),
        np.zeros(5, dtype=bool),
    )

    assert diagnostics["support_jaccard"] == 1.0
    assert diagnostics["fixed_support_expected_jaccard"] == 1.0
    assert diagnostics["chance_adjusted_jaccard"] == 0.0
    assert diagnostics["support_degenerate"] is True
    assert diagnostics["support_degeneracy_reason"] == "empty_support"


def test_genotype_loaders_preserve_ids_and_normalize_orientation(tmp_path) -> None:
    samples_by_snps = tmp_path / "samples_by_snps.tsv"
    samples_by_snps.write_text(
        "sample_id\trs1\trs2\n" "HG00001\t0\t1\n" "HG00002\t2\t0\n" "HG00003\t1\t2\n",
        encoding="utf-8",
    )
    loaded_tsv = load_genotypes(
        samples_by_snps,
        orientation="samples-by-snps",
        row_id_column="sample_id",
    )
    assert loaded_tsv.genotypes.shape == (3, 2)
    assert loaded_tsv.sample_ids.tolist() == ["HG00001", "HG00002", "HG00003"]
    assert loaded_tsv.snp_ids.tolist() == ["rs1", "rs2"]

    snps_by_samples = tmp_path / "snps_by_samples.npz"
    np.savez(
        snps_by_samples,
        X=loaded_tsv.genotypes.T,
        row_ids=np.asarray(["rs1", "rs2"]),
        column_ids=np.asarray(["HG00001", "HG00002", "HG00003"]),
    )
    loaded_npz = load_genotypes(
        snps_by_samples,
        orientation="snps-by-samples",
    )
    assert np.array_equal(loaded_npz.genotypes, loaded_tsv.genotypes)
    assert np.array_equal(loaded_npz.sample_ids, loaded_tsv.sample_ids)
    assert np.array_equal(loaded_npz.snp_ids, loaded_tsv.snp_ids)


@pytest.mark.parametrize("invalid", [float("inf"), float("-inf")])
def test_genotype_loader_rejects_infinite_values(tmp_path, invalid: float) -> None:
    path = tmp_path / "invalid.npy"
    matrix = np.zeros((3, 2), dtype=float)
    matrix[1, 1] = invalid
    np.save(path, matrix)

    with pytest.raises(ValueError, match="not infinity"):
        load_genotypes(path)


def test_missing_sentinel_is_consistent_across_loader_formats(tmp_path) -> None:
    matrix = np.asarray([[0.0, -9.0], [1.0, 2.0], [2.0, 0.0]])
    npy_path = tmp_path / "genotypes.npy"
    npz_path = tmp_path / "genotypes.npz"
    tsv_path = tmp_path / "genotypes.tsv"
    np.save(npy_path, matrix)
    np.savez(
        npz_path,
        X=matrix,
        row_ids=np.asarray(["s1", "s2", "s3"]),
        column_ids=np.asarray(["rs1", "rs2"]),
    )
    tsv_path.write_text(
        "sample_id\trs1\trs2\ns1\t0\t-9\ns2\t1\t2\ns3\t2\t0\n",
        encoding="utf-8",
    )

    loaded = [
        load_genotypes(npy_path, missing_value=-9.0),
        load_genotypes(npz_path, missing_value=-9.0),
        load_genotypes(tsv_path, row_id_column="sample_id", missing_value=-9.0),
    ]
    for dataset in loaded:
        assert np.isnan(dataset.genotypes[0, 1])
        assert np.count_nonzero(np.isnan(dataset.genotypes)) == 1

    disabled = load_genotypes(npy_path, missing_value=None)
    assert disabled.genotypes[0, 1] == -9.0


def test_single_population_is_rejected_before_held_out_scoring() -> None:
    with pytest.raises(ValueError, match="at least two populations"):
        stratified_train_test_split(np.asarray(["EUR"] * 6), 0.25, 2)
    dataset = GenotypeDataset(
        genotypes=np.tile(np.asarray([[0.0, 1.0], [1.0, 0.0]]), (3, 1)),
        sample_ids=np.asarray([f"s{index}" for index in range(6)]),
        snp_ids=np.asarray(["rs1", "rs2"]),
    )
    with pytest.raises(ValueError, match="at least two populations"):
        run_genomics(
            dataset,
            GenomicsConfig(rank=1, bootstrap_repeats=0, max_iter=1),
            population_labels=np.asarray(["EUR"] * 6),
        )


def test_bootstrap_refits_preprocessing_on_each_raw_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(4)
    raw = rng.binomial(2, 0.4, size=(18, 8)).astype(float)
    raw[0, 0] = np.nan
    dataset = GenotypeDataset(
        genotypes=raw,
        sample_ids=np.asarray([f"s{index}" for index in range(18)]),
        snp_ids=np.asarray([f"rs{index}" for index in range(8)]),
    )
    original = genomics_1kg.fit_genotype_preprocessor
    fit_inputs: list[np.ndarray] = []

    def recording_fit(matrix, **kwargs):
        fit_inputs.append(np.asarray(matrix).copy())
        return original(matrix, **kwargs)

    monkeypatch.setattr(genomics_1kg, "fit_genotype_preprocessor", recording_fit)
    rows, _ = run_genomics(
        dataset,
        GenomicsConfig(rank=1, max_iter=2, bootstrap_repeats=2, base_seed=14),
    )

    assert len(fit_inputs) == 3
    assert np.array_equal(fit_inputs[0], raw, equal_nan=True)
    assert all(sample.shape == raw.shape for sample in fit_inputs[1:])
    assert all(
        not np.array_equal(sample, raw, equal_nan=True) for sample in fit_inputs[1:]
    )
    bootstrap_rows = [
        row for row in rows if row["record_type"] == "bootstrap_stability"
    ]
    assert len(bootstrap_rows) == 2
    assert all(
        row["preprocessing_scope"] == "bootstrap_raw_resample_only"
        for row in bootstrap_rows
    )
    assert all(
        row["loading_coordinate_system"] == "original_snp_raw_dosage_coefficients"
        for row in bootstrap_rows
    )
    assert all(len(row["sampled_row_indices_sha256"]) == 64 for row in bootstrap_rows)
    assert all(row["status"] for row in bootstrap_rows)
    assert all("final_stationarity" in row for row in bootstrap_rows)


@pytest.mark.parametrize("method", ["pca", "vp", "randomized", "robust"])
def test_genomics_supported_solver_e2e_with_held_out_classification(
    method: str,
) -> None:
    rng = np.random.default_rng(731)
    labels = np.repeat(np.asarray(["AFR", "EUR", "EAS"]), 8)
    probabilities = np.asarray(
        [
            np.linspace(0.15, 0.75, 12),
            np.linspace(0.65, 0.20, 12),
            np.roll(np.linspace(0.20, 0.70, 12), 4),
        ]
    )
    class_indices = np.repeat(np.arange(3), 8)
    genotypes = rng.binomial(2, probabilities[class_indices]).astype(float)
    genotypes[0, 0] = np.nan
    dataset = GenotypeDataset(
        genotypes=genotypes,
        sample_ids=np.asarray([f"HG{index:05d}" for index in range(24)]),
        snp_ids=np.asarray([f"rs{index + 1}" for index in range(12)]),
    )
    config = GenomicsConfig(
        method=method,
        rank=2,
        alpha=0.05,
        max_iter=6,
        tol=1e-5,
        bootstrap_repeats=1,
        test_fraction=0.25,
        base_seed=991,
        oversample=2,
        n_power_iter=1,
        randomized_finalize_full_iters=1,
    )

    rows, manifest = run_genomics(dataset, config, population_labels=labels)

    assert len(manifest["seeds"]) == 3
    record_types = {row["record_type"] for row in rows}
    assert {
        "full_cohort_unsupervised_fit",
        "bootstrap_stability",
        "snp_stability",
        "held_out_population_classification",
    } <= record_types
    held_out = next(
        row
        for row in rows
        if row["record_type"] == "held_out_population_classification"
    )
    assert held_out["method"] == method
    assert held_out["train_samples"] + held_out["test_samples"] == 24
    assert held_out["preprocessing_fit_scope"] == "train_only"
    assert held_out["spca_fit_scope"] == "train_only"
    assert held_out["classifier_fit_scope"] == "train_only"
    assert held_out["test_used_for_model_or_feature_selection"] is False
    assert 0.0 <= held_out["accuracy"] <= 1.0
    assert 0.0 <= held_out["majority_baseline_accuracy"] <= 1.0
    assert isinstance(held_out["solver_converged"], bool)
    assert held_out["solver_status"]
    assert held_out["solver_n_iter"] >= 1
    assert len(held_out["train_indices_sha256"]) == 64
    assert len(held_out["test_indices_sha256"]) == 64
    assert len(held_out["retained_snp_mask_sha256"]) == 64
    bootstrap = next(row for row in rows if row["record_type"] == "bootstrap_stability")
    assert bootstrap["preprocessing_scope"] == "bootstrap_raw_resample_only"
    assert bootstrap["retained_snps"] >= config.rank
    assert bootstrap["loading_coordinate_system"] == (
        "original_snp_raw_dosage_coefficients"
    )


def test_genomics_cli_defaults_to_pca_and_serializes_runtime_provenance(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(804)
    labels = np.repeat(np.asarray(["AFR", "EUR", "EAS"]), 6)
    class_indices = np.repeat(np.arange(3), 6)
    probabilities = np.asarray(
        [
            np.linspace(0.15, 0.75, 8),
            np.linspace(0.70, 0.20, 8),
            np.roll(np.linspace(0.20, 0.65, 8), 3),
        ]
    )
    genotypes = rng.binomial(2, probabilities[class_indices]).astype(float)
    sample_ids = np.asarray([f"HG{index:05d}" for index in range(18)])
    snp_ids = np.asarray([f"rs{index + 1}" for index in range(8)])
    genotype_path = tmp_path / "tiny.npz"
    metadata_path = tmp_path / "populations.csv"
    output_path = tmp_path / "results.json"
    np.savez(
        genotype_path,
        X=genotypes,
        row_ids=sample_ids,
        column_ids=snp_ids,
    )
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "population"])
        writer.writerows(zip(sample_ids, labels))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "genomics_1kg.py",
            str(genotype_path),
            "--orientation",
            "samples-by-snps",
            "--rank",
            "2",
            "--bootstrap-repeats",
            "0",
            "--test-fraction",
            "0.25",
            "--metadata",
            str(metadata_path),
            "--output-json",
            str(output_path),
        ],
    )

    assert genomics_1kg.main() == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["config"]["method"] == "pca"
    assert payload["provenance"]["runtime_versions"]["numpy"] == np.__version__
    assert payload["provenance"]["genotype_sha256"]
    assert any(
        row["record_type"] == "held_out_population_classification"
        for row in payload["records"]
    )


def test_loading_comparison_inverse_scales_to_raw_dosage_directions() -> None:
    standardized = np.ones((2, 1), dtype=float)
    retained = np.ones(2, dtype=bool)
    first = genomics_1kg._embed_loadings_in_original_snp_coordinates(
        standardized,
        retained,
        np.asarray([1.0, 10.0]),
    )
    second = genomics_1kg._embed_loadings_in_original_snp_coordinates(
        standardized,
        retained,
        np.asarray([10.0, 1.0]),
    )

    assert first[:, 0] == pytest.approx([1.0, 0.1])
    assert second[:, 0] == pytest.approx([0.1, 1.0])
    raw_direction_correlation = float(
        (first[:, 0] @ second[:, 0])
        / (np.linalg.norm(first[:, 0]) * np.linalg.norm(second[:, 0]))
    )
    assert raw_direction_correlation == pytest.approx(0.19801980198019803)
    assert genomics_1kg._embed_support_in_original_snp_coordinates(
        standardized,
        retained,
        threshold=0.5,
    ).tolist() == [True, True]


def test_genomics_python_api_default_matches_cli_pca_policy() -> None:
    assert GenomicsConfig().method == "pca"


def test_numeric_sample_ids_are_recognized_from_canonical_header(tmp_path) -> None:
    path = tmp_path / "numeric_sample_ids.csv"
    path.write_text(
        "sample_id,rs1,rs2\n101,0,1\n102,2,0\n103,1,2\n",
        encoding="utf-8",
    )

    dataset = load_genotypes(path)

    assert dataset.sample_ids.tolist() == ["101", "102", "103"]
    assert dataset.snp_ids.tolist() == ["rs1", "rs2"]
    assert dataset.genotypes.shape == (3, 2)


def test_bootstrap_records_independent_resample_and_solver_seeds() -> None:
    rng = np.random.default_rng(876)
    dataset = GenotypeDataset(
        genotypes=rng.binomial(2, 0.4, size=(18, 8)).astype(float),
        sample_ids=np.asarray([f"s{index}" for index in range(18)]),
        snp_ids=np.asarray([f"rs{index}" for index in range(8)]),
    )

    rows, manifest = run_genomics(
        dataset,
        GenomicsConfig(rank=2, bootstrap_repeats=2, base_seed=41),
    )

    bootstrap_rows = [
        row for row in rows if row["record_type"] == "bootstrap_stability"
    ]
    assert len(bootstrap_rows) == 2
    assert len({row["resample_seed"] for row in bootstrap_rows}) == 2
    assert len({row["solver_seed"] for row in bootstrap_rows}) == 2
    assert all(row["resample_seed"] != row["solver_seed"] for row in bootstrap_rows)
    assert all(
        row["support_threshold_coordinate_system"]
        == "original_snp_raw_dosage_coefficients"
        for row in bootstrap_rows
    )
    assert manifest["metadata"]["seed_roles"]["index_2_onward"].startswith(
        "bootstrap_solver"
    )


def _randomized_alpha_selection_fixture() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    GenomicsConfig,
    int,
]:
    rng = np.random.default_rng(1907)
    labels = np.repeat(np.asarray(["AFR", "EUR", "EAS"]), 12)
    class_indices = np.repeat(np.arange(3), 12)
    probabilities = np.asarray(
        [
            np.linspace(0.10, 0.75, 18),
            np.linspace(0.70, 0.15, 18),
            np.roll(np.linspace(0.20, 0.65, 18), 5),
        ]
    )
    genotypes = rng.binomial(2, probabilities[class_indices]).astype(float)
    seed = 441
    outer_train, _ = stratified_train_test_split(labels, 0.2, seed)
    config = GenomicsConfig(
        method="randomized",
        rank=2,
        alpha=0.1,
        alpha_candidates=(0.01, 0.1, 0.5),
        validation_fraction=0.25,
        max_iter=7,
        tol=1e-8,
        bootstrap_repeats=0,
        test_fraction=0.2,
        base_seed=188,
        oversample=2,
        n_power_iter=1,
        randomized_finalize_full_iters=2,
    )
    return genotypes, labels, outer_train, config, seed


def test_randomized_alpha_grid_workspace_is_fieldwise_equivalent_to_cold_path() -> None:
    genotypes, labels, outer_train, config, seed = _randomized_alpha_selection_fixture()

    shared = genomics_1kg._select_alpha_on_training_partition(
        genotypes,
        labels,
        outer_train,
        config,
        seed,
        _reuse_randomized_workspace=True,
    )
    cold = genomics_1kg._select_alpha_on_training_partition(
        genotypes,
        labels,
        outer_train,
        config,
        seed,
        _reuse_randomized_workspace=False,
    )

    assert shared[0] == cold[0]
    assert shared[1] == cold[1]
    np.testing.assert_array_equal(shared[2], cold[2])
    np.testing.assert_array_equal(shared[3], cold[3])
    assert shared[4] == cold[4]


def test_randomized_alpha_grid_prepares_workspace_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import rspca.randomized as randomized_module

    genotypes, labels, outer_train, config, seed = _randomized_alpha_selection_fixture()
    original_prepare = randomized_module.prepare_randomized_vp_spca_workspace
    prepare_calls = 0

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        randomized_module,
        "prepare_randomized_vp_spca_workspace",
        counted_prepare,
    )
    result = genomics_1kg._select_alpha_on_training_partition(
        genotypes,
        labels,
        outer_train,
        config,
        seed,
    )

    assert prepare_calls == 1
    assert len(result[1]) == len(config.alpha_candidates)


def test_train_only_alpha_selection_is_unchanged_by_outer_test_values() -> None:
    rng = np.random.default_rng(912)
    labels = np.repeat(np.asarray(["AFR", "EUR", "EAS"]), 10)
    class_indices = np.repeat(np.arange(3), 10)
    probabilities = np.asarray(
        [
            np.linspace(0.10, 0.75, 14),
            np.linspace(0.70, 0.15, 14),
            np.roll(np.linspace(0.20, 0.65, 14), 5),
        ]
    )
    genotypes = rng.binomial(2, probabilities[class_indices]).astype(float)
    dataset = GenotypeDataset(
        genotypes=genotypes,
        sample_ids=np.asarray([f"HG{index:05d}" for index in range(30)]),
        snp_ids=np.asarray([f"rs{index + 1}" for index in range(14)]),
    )
    config = GenomicsConfig(
        method="vp",
        rank=2,
        alpha=0.1,
        alpha_candidates=(0.0, 0.05, 0.5),
        validation_fraction=0.25,
        max_iter=8,
        tol=1e-5,
        bootstrap_repeats=0,
        test_fraction=0.2,
        base_seed=188,
    )

    first_rows, manifest = run_genomics(
        dataset,
        config,
        population_labels=labels,
    )
    first = next(
        row
        for row in first_rows
        if row["record_type"] == "held_out_population_classification"
    )
    _, outer_test = stratified_train_test_split(
        labels,
        config.test_fraction,
        manifest["seeds"][1],
    )
    changed = genotypes.copy()
    changed[outer_test] = 1000.0 + np.arange(changed.shape[1])
    changed_dataset = GenotypeDataset(
        genotypes=changed,
        sample_ids=dataset.sample_ids,
        snp_ids=dataset.snp_ids,
    )
    second_rows, _ = run_genomics(
        changed_dataset,
        config,
        population_labels=labels,
    )
    second = next(
        row
        for row in second_rows
        if row["record_type"] == "held_out_population_classification"
    )

    assert first["alpha_selection_enabled"] is True
    assert first["validation_samples"] > 0
    assert first["selected_alpha"] in config.alpha_candidates
    assert len(first["alpha_selection_candidates"]) == len(config.alpha_candidates)
    assert first["test_used_for_model_or_feature_selection"] is False
    assert first["alpha_selection_status"] in {
        "selected_from_converged_candidates",
        "no_candidate_converged_fixed_budget_fallback",
    }
    assert first["alpha_selection_qualified"] is (
        first["alpha_selection_status"] == "selected_from_converged_candidates"
    )
    assert first["selected_alpha"] == second["selected_alpha"]
    assert first["alpha_selection_candidates"] == second["alpha_selection_candidates"]
    assert first["retained_snp_mask_sha256"] == second["retained_snp_mask_sha256"]
    assert first["validation_indices_sha256"] == second["validation_indices_sha256"]


@pytest.mark.parametrize(
    "config, message",
    [
        (GenomicsConfig(alpha=float("nan")), "alpha"),
        (GenomicsConfig(validation_fraction=1.0), "validation_fraction"),
        (
            GenomicsConfig(method="vp", alpha_candidates=(0.1, 0.1)),
            "duplicates",
        ),
        (GenomicsConfig(alpha_candidates=(0.1,)), "closed-form PCA"),
    ],
)
def test_genomics_config_rejects_invalid_selection_controls(
    config: GenomicsConfig,
    message: str,
) -> None:
    dataset = GenotypeDataset(
        genotypes=np.asarray([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]]),
        sample_ids=np.asarray(["s1", "s2", "s3", "s4"]),
        snp_ids=np.asarray(["rs1", "rs2"]),
    )
    with pytest.raises(ValueError, match=message):
        run_genomics(dataset, config)

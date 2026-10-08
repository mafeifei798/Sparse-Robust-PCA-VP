from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import rspca.genomics_aim as genomics_aim

from rspca.genomics_aim import (
    AIMProtocolConfig,
    MultiAutosomeGenotypeData,
    evaluate_aim_split,
    fit_smartpca_preprocessor,
    run_aim_protocol,
    select_loading_panel,
    stratified_split,
    transform_smartpca,
    validate_aim_protocol_result,
    validate_multiautosome_input,
)


def _three_population_data(
    *, input_scope: str = "synthetic_multi_autosome_smoke"
) -> MultiAutosomeGenotypeData:
    rng = np.random.default_rng(441)
    labels = np.repeat(np.asarray(["AFR", "EAS", "EUR"]), 16)
    frequencies = np.full((3, 18), 0.35)
    frequencies[0, :4] = 0.08
    frequencies[1, :4] = 0.88
    frequencies[2, :4] = 0.45
    frequencies[0, 4:8] = 0.82
    frequencies[1, 4:8] = 0.12
    frequencies[2, 4:8] = 0.48
    matrix = np.vstack(
        [rng.binomial(2, frequencies[index], size=(16, 18)) for index in range(3)]
    ).astype(float)
    chromosomes = (
        np.asarray(["20"] * 6 + ["21"] * 6 + ["22"] * 6)
        if input_scope == "legacy_chr20_22"
        else np.asarray(["1"] * 9 + ["2"] * 9)
    )
    return MultiAutosomeGenotypeData(
        genotypes=matrix,
        sample_ids=np.asarray(
            [f"sample_{index:03d}" for index in range(matrix.shape[0])]
        ),
        snp_ids=np.asarray([f"snp_{index:03d}" for index in range(matrix.shape[1])]),
        chromosomes=chromosomes,
        population_labels=labels,
        source_manifest_sha256="a" * 64,
        input_scope=input_scope,
        ld_pruned=True,
    )


def _config(*, methods: tuple[str, ...] = ("pca",)) -> AIMProtocolConfig:
    return AIMProtocolConfig(
        population_design="three_continent",
        methods=methods,
        n_components=2,
        panel_size=6,
        test_fraction=0.25,
        seeds=(71, 72),
        max_iter=4,
        tol=1e-3,
        randomized_oversample=3,
        randomized_power_iterations=1,
        randomized_finalize_full_iters=1,
        bootstrap_resamples=200,
        bootstrap_seed=991,
    )


def test_legacy_scope_is_bound_to_chr20_chr21_chr22() -> None:
    data = _three_population_data(input_scope="legacy_chr20_22")
    validate_multiautosome_input(data)
    invalid = replace(
        data, chromosomes=np.asarray(["19"] * 6 + ["21"] * 6 + ["22"] * 6)
    )
    with pytest.raises(ValueError, match="20, 21, and 22"):
        validate_multiautosome_input(invalid)


def test_input_contract_rejects_bad_dosage_identity_and_ld_declaration() -> None:
    data = _three_population_data()
    invalid_matrix = data.genotypes.copy()
    invalid_matrix[0, 0] = 2.5
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        validate_multiautosome_input(replace(data, genotypes=invalid_matrix))
    duplicate_ids = data.snp_ids.copy()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="snp_ids must be unique"):
        validate_multiautosome_input(replace(data, snp_ids=duplicate_ids))
    with pytest.raises(ValueError, match="LD-pruned"):
        validate_multiautosome_input(replace(data, ld_pruned=False))


def test_smartpca_statistics_are_fitted_once_and_reused() -> None:
    matrix = np.asarray(
        [
            [0.0, 0.0, np.nan],
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
        ]
    )
    preprocessor = fit_smartpca_preprocessor(matrix[:3])
    transformed = transform_smartpca(matrix[3:], preprocessor)
    np.testing.assert_allclose(preprocessor.mean_dosage, [1.0, 1.0, 1.5])
    np.testing.assert_allclose(
        transformed[0],
        (matrix[3] - preprocessor.mean_dosage) / preprocessor.scale,
    )
    changed_test = matrix[3:].copy()
    changed_test[:] = 2.0
    np.testing.assert_array_equal(
        preprocessor.mean_dosage,
        fit_smartpca_preprocessor(matrix[:3]).mean_dosage,
    )
    assert not np.array_equal(
        transform_smartpca(changed_test, preprocessor), transformed
    )


def test_loading_panel_is_exact_unsupervised_and_tie_deterministic() -> None:
    loadings = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.1, 0.1],
        ]
    )
    selected, importance = select_loading_panel(loadings, 2, score="l1")
    np.testing.assert_array_equal(selected, [0, 1])
    np.testing.assert_allclose(importance, [1.0, 1.0, 1.0, 0.2])
    with pytest.raises(ValueError, match="all loading importance"):
        select_loading_panel(np.zeros((4, 2)), 2)


def test_native_support_below_panel_size_cannot_be_qualified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _three_population_data()
    config = _config()

    def fake_fit(matrix, method, fit_config, seed):
        del method, seed
        loadings = np.zeros((matrix.shape[1], fit_config.n_components), dtype=float)
        if matrix.shape[1] > fit_config.panel_size:
            # The exact top-K selector will retain this one real loading and
            # deterministically tie-fill the remaining K-1 zero rows.
            loadings[0, 0] = 1.0
        else:
            loadings[: fit_config.n_components] = np.eye(fit_config.n_components)
        return genomics_aim._MethodFit(
            loadings=loadings,
            converged=True,
            status="test converged",
            n_iter=1,
            final_stationarity=0.0,
        )

    monkeypatch.setattr(genomics_aim, "_fit_method", fake_fit)
    train, test = stratified_split(
        data.population_labels, config.test_fraction, config.seeds[0]
    )
    records, _ = evaluate_aim_split(
        data, config, train, test, seed=config.seeds[0]
    )
    record = records[0]
    assert len(record["selected_snp_ids"]) == config.panel_size
    assert record["prepanel_native_support_count"] == 1
    assert record["prepanel_native_support_sufficient_for_panel"] is False
    assert record["prepanel_qualification_status"] == (
        "prepanel_native_support_below_panel_size"
    )
    assert record["prepanel_qualified"] is False
    assert record["overall_qualified"] is False


def test_pca_protocol_has_equal_budget_common_probe_and_stability() -> None:
    data = _three_population_data()
    config = _config()
    first = run_aim_protocol(data, config)
    second = run_aim_protocol(data, config)
    assert first.result_sha256 == second.result_sha256
    assert len(first.split_records) == 2
    assert len(first.method_summaries) == 1
    assert first.selection_frequencies["pca"].shape == (data.genotypes.shape[1],)
    for record in first.split_records:
        assert len(record["selected_snp_ids"]) == config.panel_size
        assert record["panel_selection_used_population_labels"] is False
        assert record["test_used_for_preprocessing_or_model_selection"] is False
        assert record["native_embedding_method"] == "pca"
        assert "full_width_native_balanced_accuracy" in record
        assert "native_panel_retention_delta_balanced_accuracy" in record
        assert record["native_embedding_balanced_accuracy"] == pytest.approx(
            record["common_pca_probe_balanced_accuracy"]
        )
        assert record["prepanel_native_support_count"] >= config.panel_size
    summary = first.method_summaries[0]
    assert 0.0 <= summary["mean_pairwise_panel_jaccard_across_splits"] <= 1.0
    assert summary["mean_panel_jaccard_with_equal_budget_pca"] == 1.0


def test_held_out_values_cannot_change_training_selected_panel() -> None:
    data = _three_population_data()
    config = _config()
    train, test = stratified_split(
        data.population_labels, config.test_fraction, config.seeds[0]
    )
    first, _ = evaluate_aim_split(data, config, train, test, seed=config.seeds[0])
    mutated = data.genotypes.copy()
    mutated[test] = 2.0 - mutated[test]
    second, _ = evaluate_aim_split(
        replace(data, genotypes=mutated),
        config,
        train,
        test,
        seed=config.seeds[0],
    )
    assert first[0]["selected_snp_ids"] == second[0]["selected_snp_ids"]
    assert first[0]["prepanel_loadings_sha256"] == second[0]["prepanel_loadings_sha256"]


@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_all_repository_methods_report_native_and_common_probe_evidence() -> None:
    data = _three_population_data()
    config = _config(methods=("pca", "vp", "randomized", "robust"))
    result = run_aim_protocol(data, config)
    assert {summary["method"] for summary in result.method_summaries} == {
        "pca",
        "vp",
        "randomized",
        "robust",
    }
    assert len(result.split_records) == 8
    for record in result.split_records:
        assert record["panel_size"] == 6
        assert "prepanel_solver_converged" in record
        assert "prepanel_native_support_fraction" in record
        assert "native_panel_refit_solver_converged" in record
        assert "native_embedding_balanced_accuracy" in record
        assert "common_pca_probe_balanced_accuracy" in record
        assert "full_width_native_per_population_metrics" in record
        assert "native_panel_retention_delta_balanced_accuracy" in record
        assert "paired_delta_vs_pca_common_pca_probe_balanced_accuracy" in record
        assert len(record["selected_snp_ids"]) == 6
        assert record["qualification_status"] in {
            "qualified",
            "prepanel_solver_not_converged",
            "prepanel_native_support_below_panel_size",
            "prepanel_native_support_degenerate",
            "native_panel_refit_solver_not_converged",
            "common_pca_probe_not_converged",
        }


@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_method_reordering_does_not_change_randomized_seed_or_fit() -> None:
    data = _three_population_data()
    first = run_aim_protocol(
        data,
        _config(methods=("pca", "vp", "randomized", "robust")),
    )
    reordered = run_aim_protocol(
        data,
        _config(methods=("robust", "randomized", "vp", "pca")),
    )
    first_randomized = {
        record["seed"]: record
        for record in first.split_records
        if record["method"] == "randomized"
    }
    reordered_randomized = {
        record["seed"]: record
        for record in reordered.split_records
        if record["method"] == "randomized"
    }
    for seed in first.config.seeds:
        for field in (
            "prepanel_loadings_sha256",
            "native_panel_loadings_sha256",
            "selected_snp_ids_sha256",
            "native_test_predictions_sha256",
        ):
            assert first_randomized[seed][field] == reordered_randomized[seed][field]


def test_each_fixed_panel_is_ranked_from_its_own_train_only_solver_loadings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.repeat(np.asarray(["AFR", "EAS", "EUR"]), 16)
    rows = np.arange(labels.size, dtype=float)[:, None]
    columns = np.arange(18, dtype=float)[None, :]
    matrix = np.mod(rows + columns, 3.0)
    data = MultiAutosomeGenotypeData(
        genotypes=matrix,
        sample_ids=np.asarray([f"lineage_sample_{index}" for index in range(48)]),
        snp_ids=np.asarray([f"lineage_snp_{index}" for index in range(18)]),
        chromosomes=np.asarray(["1"] * 9 + ["2"] * 9),
        population_labels=labels,
        source_manifest_sha256="b" * 64,
        input_scope="synthetic_multi_autosome_smoke",
        ld_pruned=True,
    )
    desired = {
        "pca": (0, 1),
        "vp": (2, 3),
        "randomized": (4, 5),
        "robust": (6, 7),
    }

    def fake_fit(matrix, method, config, seed):
        del seed
        loadings = np.zeros((matrix.shape[1], config.n_components), dtype=float)
        if matrix.shape[1] == 18:
            first, second = desired[method]
            loadings[first, 0] = 10.0
            loadings[second, 1] = 9.0
        else:
            loadings[: config.n_components, :] = np.eye(config.n_components)
        return genomics_aim._MethodFit(
            loadings=loadings,
            converged=True,
            status="controlled lineage test",
            n_iter=1,
            final_stationarity=0.0,
        )

    monkeypatch.setattr(genomics_aim, "_fit_method", fake_fit)
    config = replace(
        _config(methods=("pca", "vp", "randomized", "robust")),
        panel_size=2,
    )
    result = run_aim_protocol(data, config)
    for record in result.split_records:
        expected_ids = [data.snp_ids[index] for index in desired[record["method"]]]
        assert record["selected_snp_ids"] == expected_ids
        assert record["panel_selection_used_population_labels"] is False
    assert set(result.solver_lineage) == set(config.methods)
    assert all(
        lineage["panel_ranking_callable"] == "rspca.genomics_aim.select_loading_panel"
        and lineage["panel_selection_uses_population_labels"] is False
        and lineage["supervised_feature_ranking"] is False
        and len(lineage["source_sha256"]) == 64
        and len(lineage["qualification_role"]) > 0
        for lineage in result.solver_lineage.values()
    )


@pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")
def test_relabeling_fixed_train_test_rows_cannot_change_any_loading_panel() -> None:
    data = _three_population_data()
    config = _config(methods=("pca", "vp", "randomized", "robust"))
    train, test = stratified_split(
        data.population_labels,
        config.test_fraction,
        config.seeds[0],
    )
    first, _ = evaluate_aim_split(data, config, train, test, seed=config.seeds[0])
    relabeled = np.asarray(data.population_labels, dtype=str).copy()
    relabeled[train] = np.roll(relabeled[train], 1)
    relabeled[test] = np.roll(relabeled[test], 1)
    second, _ = evaluate_aim_split(
        replace(data, population_labels=relabeled),
        config,
        train,
        test,
        seed=config.seeds[0],
    )
    for first_record, second_record in zip(first, second, strict=True):
        assert first_record["method"] == second_record["method"]
        assert first_record["selected_snp_ids"] == second_record["selected_snp_ids"]
        assert (
            first_record["prepanel_loadings_sha256"]
            == second_record["prepanel_loadings_sha256"]
        )


def test_registered_coordinates_use_first_seed_only_and_are_auditable() -> None:
    data = _three_population_data()
    result = run_aim_protocol(data, _config())
    assert {row["seed"] for row in result.registered_coordinates} == {
        result.config.seeds[0]
    }
    assert {row["registered_seed_policy"] for row in result.registered_coordinates} == {
        "first_config_seed_not_best"
    }
    assert {row["evaluation"] for row in result.registered_coordinates} == {
        "full_width_native",
        "top_k_native_embedding",
        "top_k_common_pca_probe",
    }
    first_record = result.split_records[0]
    assert len(result.registered_coordinates) == 3 * first_record["test_samples"]
    assert all(np.isfinite(row["component_1"]) for row in result.registered_coordinates)
    assert all(np.isfinite(row["component_2"]) for row in result.registered_coordinates)


def test_summary_reports_sample_std_paired_bootstrap_and_qualification() -> None:
    result = run_aim_protocol(_three_population_data(), _config())
    summary = result.method_summaries[0]
    assert summary["std_native_embedding_balanced_accuracy"] >= 0.0
    assert summary["bootstrap_resamples"] == 200
    assert summary["bootstrap_seed"] == 991
    assert summary["runtime_bootstrapped"] is False
    assert summary["bootstrap_interval"] == "paired_seed_percentile_95"
    assert summary["mean_paired_delta_vs_pca_common_pca_probe_balanced_accuracy"] == 0.0
    assert (
        summary[
            "paired_delta_vs_pca_common_pca_probe_balanced_accuracy_bootstrap_ci95_low"
        ]
        == 0.0
    )
    assert (
        summary[
            "paired_delta_vs_pca_common_pca_probe_balanced_accuracy_bootstrap_ci95_high"
        ]
        == 0.0
    )
    assert summary["qualification_status"] in {
        "all_splits_qualified",
        "partially_qualified",
        "no_splits_qualified",
    }


def test_result_validator_rejects_identity_frequency_and_nonfinite_attacks() -> None:
    data = _three_population_data()
    result = run_aim_protocol(data, _config())
    bad_identity = dict(result.input_identity)
    bad_identity["genotypes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input_identity"):
        validate_aim_protocol_result(
            replace(result, input_identity=bad_identity),
            data,
        )

    with pytest.raises(ValueError, match="shape mismatch"):
        validate_aim_protocol_result(
            replace(
                result,
                selection_frequencies={"pca": result.selection_frequencies["pca"][:-1]},
            ),
            data,
        )
    bad_range = result.selection_frequencies["pca"].copy()
    bad_range[0] = 1.1
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        validate_aim_protocol_result(
            replace(result, selection_frequencies={"pca": bad_range}),
            data,
        )
    bad_nan = result.selection_frequencies["pca"].copy()
    bad_nan[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_aim_protocol_result(
            replace(result, selection_frequencies={"pca": bad_nan}),
            data,
        )

    bad_records = [dict(record) for record in result.split_records]
    bad_records[0]["native_embedding_balanced_accuracy"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_aim_protocol_result(
            replace(result, split_records=tuple(bad_records)),
            data,
        )


def test_result_validator_rejects_coordinate_and_digest_tampering() -> None:
    data = _three_population_data()
    result = run_aim_protocol(data, _config())
    coordinates = [dict(row) for row in result.registered_coordinates]
    coordinates[0]["sample_id"] = "unknown_sample"
    with pytest.raises(ValueError, match="coordinate"):
        validate_aim_protocol_result(
            replace(result, registered_coordinates=tuple(coordinates)),
            data,
        )
    with pytest.raises(ValueError, match="result_sha256"):
        validate_aim_protocol_result(
            replace(result, result_sha256="0" * 64),
            data,
        )


def test_three_continent_and_five_superpopulation_tracks_cannot_be_conflated() -> None:
    data = _three_population_data()
    with pytest.raises(ValueError, match="AFR, AMR, EAS, EUR, and SAS"):
        run_aim_protocol(
            data,
            replace(_config(), population_design="five_superpopulation"),
        )

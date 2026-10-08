from __future__ import annotations

import ast
import copy
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from experiments import genomics_pca_loading_all_autosomes as all_auto
from experiments import genomics_pca_loading_grid as base
from rspca.datasets import canonical_sha256
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    transform_smartpca,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def smoke_data() -> base.GenotypeInput:
    return all_auto._synthetic_smoke_data()


@pytest.fixture(scope="module")
def smoke_payload(smoke_data: base.GenotypeInput) -> dict[str, object]:
    config = all_auto.AllAutosomeConfig(
        seeds=(73, 74),
        analysis_mode="tiny_smoke",
        snp_chunk_size=257,
    )
    return all_auto.evaluate_all_autosome_stratification(
        smoke_data,
        config,
        input_identity=all_auto._synthetic_input_identity(smoke_data),
        source_identity=all_auto._source_identity(REPO_ROOT),
    )


def _resign(payload: dict[str, object]) -> None:
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(unsigned)


def _random_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(119)
    probabilities = rng.uniform(0.08, 0.82, size=(1, 520))
    raw = rng.binomial(2, probabilities, size=(85, 520)).astype(float)
    raw[1, 7] = np.nan
    raw[64, 19] = np.nan
    return raw, np.arange(65, dtype=np.int64), np.arange(65, 85, dtype=np.int64)


def test_protocol_is_fixed_wide_fraction_grid_without_target_mode() -> None:
    config = all_auto.AllAutosomeConfig()
    assert config.seeds == tuple(range(2025, 2045))
    assert config.component_grid == (4, 6, 8)
    assert config.panel_fractions == (0.05, 0.10, 0.25, 0.50, None)
    assert config.test_fraction == 0.2
    assert config.analysis_mode == "fixed_protocol"
    all_auto._validate_config(config)
    with pytest.raises(ValueError, match="component_grid is fixed"):
        all_auto._validate_config(replace(config, component_grid=(2, 4, 8)))
    with pytest.raises(ValueError, match="panel_fractions is fixed"):
        all_auto._validate_config(
            replace(config, panel_fractions=(0.01, 0.05, 0.10, None))
        )
    with pytest.raises(ValueError, match="test_fraction is fixed"):
        all_auto._validate_config(replace(config, test_fraction=0.3))
    with pytest.raises(ValueError, match="2025..2044"):
        all_auto._validate_config(
            replace(config, analysis_mode="registered_shard", seeds=(2045,))
        )
    with pytest.raises(ValueError, match="unique"):
        all_auto._validate_config(
            replace(config, seeds=tuple(float(seed) for seed in config.seeds))
        )
    with pytest.raises(ValueError, match="exact integers"):
        all_auto._validate_config(replace(config, component_grid=(4.0, 6.0, 8.0)))
    with pytest.raises(ValueError, match="frozen to 2048"):
        all_auto._validate_config(replace(config, snp_chunk_size=2049))
    with pytest.raises(ValueError, match="integer at least 32"):
        all_auto._validate_config(replace(config, snp_chunk_size=2048.9))


def test_config_payload_is_json_canonical_and_roundtrips_none_fraction() -> None:
    config = all_auto.AllAutosomeConfig(
        seeds=(2025, 2026),
        analysis_mode="registered_shard",
    )
    payload = all_auto._config_payload(config)
    assert payload["seeds"] == [2025, 2026]
    assert payload["component_grid"] == [4, 6, 8]
    assert payload["panel_fractions"] == [0.05, 0.10, 0.25, 0.50, None]
    assert all(
        isinstance(payload[field], list)
        for field in ("seeds", "component_grid", "panel_fractions")
    )
    assert json.loads(all_auto._strict_json_bytes(payload)) == payload
    assert all_auto._config_from_payload(payload) == config


def test_formal_blas_thread_environment_is_explicitly_single_threaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = {
        "threadpoolctl_version": "test-version",
        "blas_pools": [
            {
                "user_api": "blas",
                "internal_api": "openblas",
                "prefix": "libopenblas",
                "version": "test-openblas-version",
                "threading_layer": "pthreads",
                "architecture": "test-architecture",
                "num_threads": 1,
            }
        ],
    }
    monkeypatch.setattr(all_auto, "_blas_runtime_identity", lambda: runtime)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    assert all_auto._require_formal_blas_thread_environment() == runtime

    stale_runtime = copy.deepcopy(runtime)
    stale_runtime["blas_pools"][0]["num_threads"] = 10
    monkeypatch.setattr(all_auto, "_blas_runtime_identity", lambda: stale_runtime)
    with pytest.raises(ValueError, match="every loaded BLAS pool"):
        all_auto._require_formal_blas_thread_environment()

    undetected_runtime = {
        "threadpoolctl_version": "test-version",
        "blas_pools": [],
    }
    monkeypatch.setattr(all_auto, "_blas_runtime_identity", lambda: undetected_runtime)
    with pytest.raises(ValueError, match="detected BLAS runtime pool"):
        all_auto._require_formal_blas_thread_environment()

    monkeypatch.setattr(all_auto, "_blas_runtime_identity", lambda: runtime)
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="OPENBLAS_NUM_THREADS=1"):
        all_auto._require_formal_blas_thread_environment()


def test_completed_formal_input_anchors_are_code_locked() -> None:
    assert all_auto.FROZEN_FORMAL_SHAPE == (1000, 455_102)
    assert all_auto.FROZEN_FORMAL_MANIFEST_SHA256 == (
        "0abdf5f4b6c6724f30f539377f1244bb90d3932d7c3e1b9fd6510950bef61030"
    )
    assert all_auto.FROZEN_FORMAL_NPZ_SHA256 == (
        "6ef9864f1d15b8e294434d84b07443ba0d66a19bcfb38401cd1fcd67581e80bc"
    )
    assert all_auto.FROZEN_SELECTED_SAMPLE_IDS_SHA256 == (
        "fbb844e7d6277b15ebdd63d2a362f8254c0edf901bbbb82f6a7b81e7f6095497"
    )
    assert all_auto.FROZEN_PREPARED_ARRAY_SHA256["population_labels"] == (
        "6ca09a20bd51f5ada6c18be707ed3b7596b5d31fa95746b1e9c3cfc642832277"
    )


def test_fraction_resolution_is_floor_based_deduplicated_and_all_is_unselected() -> (
    None
):
    rows = all_auto._resolved_panels(10_003)
    assert [row["effective_snp_count"] for row in rows] == [
        500,
        1000,
        2500,
        5001,
        10_003,
    ]
    assert [row["panel_label"] for row in rows] == [
        "5pct",
        "10pct",
        "25pct",
        "50pct",
        "all",
    ]
    assert all(row["active_grid_cell"] for row in rows)
    assert rows[-1]["selection_rule"] == "all_train_retained_no_loading_selection"
    assert rows[-1]["resolution_uses_labels_or_outcomes"] is False

    tiny = all_auto._resolved_panels(8)
    assert [row["active_grid_cell"] for row in tiny] == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert tiny[1]["duplicate_effective_count_of"] == "5pct"
    assert tiny[-1]["panel_label"] == "all"
    assert tiny[-1]["active_grid_cell"] is True


def test_chunked_gram_pca_matches_full_svd_subspace_scores_and_predictions() -> None:
    raw, train, test = _random_matrix()
    observed = all_auto.fit_chunked_top_pca(
        raw,
        train,
        test,
        n_components=8,
        chunk_size=43,
    )
    preprocessor = fit_smartpca_preprocessor(raw[train])
    train_matrix = transform_smartpca(raw[train], preprocessor)
    test_matrix = transform_smartpca(raw[test], preprocessor)
    expected_loadings, expected_singular = base.fit_top_pca(train_matrix, 8)
    np.testing.assert_allclose(
        observed.singular_values,
        expected_singular,
        atol=2e-11,
        rtol=2e-12,
    )
    np.testing.assert_allclose(
        observed.loadings @ observed.loadings.T,
        expected_loadings @ expected_loadings.T,
        atol=3e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        observed.train_scores,
        train_matrix @ observed.loadings,
        atol=3e-11,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        observed.test_scores,
        test_matrix @ observed.loadings,
        atol=3e-11,
        rtol=1e-12,
    )
    labels = np.asarray(
        [population for population in all_auto.EXPECTED_POPULATIONS for _ in range(13)]
    )
    expected_predictions, _, _ = base._nearest_centroid(
        train_matrix @ expected_loadings,
        labels,
        test_matrix @ expected_loadings,
    )
    observed_predictions, _, _ = base._nearest_centroid(
        observed.train_scores,
        labels,
        observed.test_scores,
    )
    np.testing.assert_array_equal(observed_predictions, expected_predictions)


def test_incremental_nested_gram_matches_independent_panel_refits() -> None:
    raw, train, test = _random_matrix()
    full = all_auto.fit_chunked_top_pca(raw, train, test, n_components=8, chunk_size=47)
    ranking, _ = base.rank_by_pca_loadings(full.loadings, 8)
    ordered_original = full.retained_original[ranking]
    nested = all_auto.fit_nested_chunked_panels(
        raw,
        train,
        test,
        ordered_original_indices=ordered_original,
        panel_counts=(40, 90, 180),
        mean_dosage=full.mean_dosage,
        scale=full.scale,
        n_components=8,
        chunk_size=37,
    )
    labels = np.asarray(
        [population for population in all_auto.EXPECTED_POPULATIONS for _ in range(13)]
    )
    for count, observed in nested.items():
        mask = np.zeros(raw.shape[1], dtype=bool)
        mask[ordered_original[:count]] = True
        expected = all_auto.fit_chunked_panel_pca(
            raw,
            train,
            test,
            selected_mask=mask,
            mean_dosage=full.mean_dosage,
            scale=full.scale,
            n_components=8,
            chunk_size=41,
        )
        np.testing.assert_allclose(
            observed.loadings @ observed.loadings.T,
            expected.loadings @ expected.loadings.T,
            atol=3e-11,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            observed.singular_values,
            expected.singular_values,
            atol=3e-11,
            rtol=2e-12,
        )
        np.testing.assert_allclose(
            observed.train_scores @ observed.train_scores.T,
            expected.train_scores @ expected.train_scores.T,
            atol=3e-9,
            rtol=2e-12,
        )
        observed_predictions, _, _ = base._nearest_centroid(
            observed.train_scores, labels, observed.test_scores
        )
        expected_predictions, _, _ = base._nearest_centroid(
            expected.train_scores, labels, expected.test_scores
        )
        np.testing.assert_array_equal(observed_predictions, expected_predictions)


def test_runner_imports_no_forbidden_scientific_method() -> None:
    path = REPO_ROOT / "experiments/genomics_pca_loading_all_autosomes.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(
        module
        in {
            "rspca.vp",
            "rspca.robust",
            "rspca.randomized",
            "rspca.federated",
            "rspca.fedrpca_audit",
        }
        for module in modules
    )
    assert all_auto.METHOD_SCOPE == {
        "methods": ["pca_loading_ranking"],
        "robust_solver_run": False,
        "vp_solver_run": False,
        "randomized_solver_run": False,
        "federated_solver_run": False,
        "lpca_solver_run": False,
        "alpca_solver_run": False,
        "fisher_or_supervised_panel_ranking_run": False,
    }


def test_smoke_reports_every_cell_without_selection(
    smoke_payload: dict[str, object]
) -> None:
    validation = all_auto.validate_result_payload(smoke_payload)
    assert validation["records"] == 2 * 3 * 5
    assert validation["selection_masks"] == 2 * 3 * 5
    assert validation["sample_prediction_rows"] == 2 * 3 * 5 * 20
    policy = smoke_payload["analysis_policy"]
    assert policy["best_configuration_selected"] is False
    assert policy["headline_configuration_selected"] is False
    assert policy["target_metric_or_quality_gate_used"] is False
    assert policy["result_triggered_parameter_change_allowed"] is False
    assert policy["all_panel_uses_loading_selection"] is False
    all_rows = [
        row
        for row in smoke_payload["panel_selection_evidence"]
        if row["panel_label"] == "all"
    ]
    assert len(all_rows) == 2 * 3
    assert all(
        row["selection_rule"] == "all_train_retained_no_loading_selection"
        for row in all_rows
    )
    assert all(row["panel_feature_set_is_split_specific"] is True for row in all_rows)
    assert all(row["loading_selection_is_split_specific"] is False for row in all_rows)
    assert set(
        row["population"] for row in smoke_payload["population_summaries"]
    ) == set(all_auto.EXPECTED_POPULATIONS)


def test_result_validator_rejects_metric_and_mask_tampering(
    smoke_payload: dict[str, object],
) -> None:
    metric_attack = copy.deepcopy(smoke_payload)
    metric_attack["records"][0]["balanced_accuracy"] += 0.01
    _resign(metric_attack)
    with pytest.raises(ValueError, match="metrics do not match sample evidence"):
        all_auto.validate_result_payload(metric_attack)

    mask_attack = copy.deepcopy(smoke_payload)
    row = next(
        item
        for item in mask_attack["panel_selection_evidence"]
        if item["panel_label"] == "all"
    )
    mask = all_auto._unpack_mask(
        row["selected_mask_packbits_base64"], row["selected_mask_width"]
    )
    mask[np.flatnonzero(mask)[0]] = False
    row["selected_mask_packbits_base64"] = all_auto._pack_mask(mask)
    _resign(mask_attack)
    with pytest.raises(ValueError, match="hash/count mismatch"):
        all_auto.validate_result_payload(mask_attack)


def test_replay_mismatch_reports_first_changed_path(
    smoke_payload: dict[str, object]
) -> None:
    changed = copy.deepcopy(smoke_payload)
    changed["sample_prediction_evidence"][0]["prediction"] = "EUR"
    mismatch = all_auto._first_replay_mismatch(smoke_payload, changed)
    assert mismatch is not None
    assert "sample_prediction_evidence[0].prediction" in mismatch

    type_attack = copy.deepcopy(smoke_payload)
    type_attack["registered_seed_policy"]["panels"] = tuple(
        type_attack["registered_seed_policy"]["panels"]
    )
    mismatch = all_auto._first_replay_mismatch(smoke_payload, type_attack)
    assert mismatch is not None
    assert "registered_seed_policy.panels: type list != tuple" in mismatch


def _fake_registered_shard(seeds: tuple[int, ...], tag: str) -> dict[str, object]:
    width = 16
    panel_counts = {"5pct": 1, "10pct": 2, "25pct": 4, "50pct": 8, "all": 16}
    fractions = dict(
        zip(
            panel_counts,
            all_auto.PANEL_FRACTIONS,
            strict=True,
        )
    )
    records: list[dict[str, object]] = []
    evidence: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    for seed in seeds:
        for rank in all_auto.COMPONENT_GRID:
            rankings.append({"seed": seed, "n_components": rank})
            for label, count in panel_counts.items():
                mask = np.zeros(width, dtype=bool)
                mask[:count] = True
                indices = np.flatnonzero(mask).astype(np.int64)
                evidence.append(
                    {
                        "seed": seed,
                        "n_components": rank,
                        "panel_label": label,
                        "requested_panel_fraction": fractions[label],
                        "selected_snp_count": count,
                        "selected_mask_packbits_base64": all_auto._pack_mask(mask),
                        "selected_original_indices_sha256": all_auto.array_sha256(
                            indices
                        ),
                    }
                )
                records.append(
                    {
                        "seed": seed,
                        "n_components": rank,
                        "panel_label": label,
                        "requested_panel_fraction": fractions[label],
                        "selected_snp_count": count,
                        "effective_fraction_of_train_retained": count / width,
                        "accuracy": 0.8,
                        "balanced_accuracy": 0.8,
                        "macro_f1": 0.8,
                        "per_population_metrics": [
                            {
                                "population": population,
                                "support": 1,
                                "true_positive": 1,
                                "recall": 1.0,
                            }
                            for population in all_auto.EXPECTED_POPULATIONS
                        ],
                    }
                )
                sample_rows.append(
                    {
                        "seed": seed,
                        "n_components": rank,
                        "panel_label": label,
                        "sample_position": 0,
                    }
                )
    return {
        "analysis_policy": {"fixed": True},
        "config": asdict(
            all_auto.AllAutosomeConfig(
                seeds=seeds,
                analysis_mode="registered_shard",
                snp_chunk_size=2048,
            )
        ),
        "method_scope": dict(all_auto.METHOD_SCOPE),
        "input_identity": {"snps": width},
        "source_identity": {"snapshot": "same"},
        "runtime_versions": {"runtime": "same"},
        "memory_and_solver_policy": {"memory": "same"},
        "split_receipts": [{"seed": seed} for seed in seeds],
        "ranking_records": rankings,
        "panel_selection_evidence": evidence,
        "records": records,
        "sample_prediction_evidence": sample_rows,
        "registered_seed_policy": {"seed": seeds[0]},
        "registered_coordinates": [{"seed": seeds[0], "tag": tag}],
        "analysis_sha256": all_auto._bytes_sha256(tag.encode("utf-8")),
    }


def test_merge_requires_exact_cover_and_matches_direct_scientific_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(all_auto, "validate_result_payload", lambda _: {})
    groups = tuple(tuple(range(start, start + 5)) for start in (2025, 2030, 2035, 2040))
    shards = [
        _fake_registered_shard(seeds, f"shard-{index}")
        for index, seeds in enumerate(groups)
    ]
    merged = all_auto.merge_registered_shard_payloads(shards)
    assert isinstance(merged["config"]["seeds"], list)
    assert isinstance(merged["config"]["component_grid"], list)
    assert isinstance(merged["config"]["panel_fractions"], list)
    assert merged["config"]["panel_fractions"][-1] is None
    assert merged["execution_provenance"]["exact_registered_seed_cover"] is True
    assert len(merged["execution_provenance"]["source_shard_analysis_sha256"]) == 4
    assert [row["seed"] for row in merged["split_receipts"]] == list(
        all_auto.PROTOCOL_SEEDS
    )

    direct = copy.deepcopy(merged)
    direct["execution_provenance"] = {
        "execution_mode": "fixed_protocol",
        "registered_seed_bank": list(all_auto.PROTOCOL_SEEDS),
        "included_seeds": list(all_auto.PROTOCOL_SEEDS),
        "exact_registered_seed_cover": True,
        "merged_from_shards": False,
        "source_shard_analysis_sha256": [],
    }
    direct["analysis_sha256"] = "0" * 64
    assert all_auto._scientific_replay_mismatch(merged, direct) is None
    direct["records"][0]["balanced_accuracy"] = 0.7
    assert all_auto._scientific_replay_mismatch(merged, direct) is not None

    with pytest.raises(ValueError, match="duplicate registered seed"):
        all_auto.merge_registered_shard_payloads(
            [shards[0], shards[0], shards[2], shards[3]]
        )
    with pytest.raises(ValueError, match="exactly cover"):
        all_auto.merge_registered_shard_payloads(shards[:-1])
    incompatible = copy.deepcopy(shards)
    incompatible[-1]["runtime_versions"] = {"runtime": "different"}
    with pytest.raises(ValueError, match="immutable field"):
        all_auto.merge_registered_shard_payloads(incompatible)


def test_output_package_is_exclusive_exact_and_deterministic(
    smoke_data: base.GenotypeInput,
    smoke_payload: dict[str, object],
    tmp_path: Path,
) -> None:
    import matplotlib

    first = tmp_path / "first"
    second = tmp_path / "second"
    all_auto.write_output_package(smoke_payload, first)
    original_facecolor = matplotlib.rcParams["axes.facecolor"]
    try:
        matplotlib.rcParams["axes.facecolor"] = "magenta"
        all_auto.write_output_package(smoke_payload, second)
        assert matplotlib.rcParams["axes.facecolor"] == "magenta"
    finally:
        matplotlib.rcParams["axes.facecolor"] = original_facecolor
    expected = set(all_auto.ARTIFACT_FILENAMES) | {all_auto.MANIFEST_FILENAME}
    assert {path.name for path in first.iterdir()} == expected
    assert {path.name for path in second.iterdir()} == expected
    for filename in expected:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    reloaded = all_auto._strict_json_load(first / all_auto.RESULT_FILENAME)
    assert isinstance(reloaded["config"]["seeds"], list)
    assert isinstance(reloaded["config"]["component_grid"], list)
    assert isinstance(reloaded["config"]["panel_fractions"], list)
    assert reloaded["config"]["panel_fractions"][-1] is None
    replayed = all_auto.evaluate_all_autosome_stratification(
        smoke_data,
        all_auto._config_from_payload(reloaded["config"]),
        input_identity=reloaded["input_identity"],
        source_identity=reloaded["source_identity"],
    )
    assert all_auto._first_replay_mismatch(reloaded, replayed) is None

    with pytest.raises(FileExistsError, match="already exists"):
        all_auto.write_output_package(smoke_payload, first)
    extra = first / "unregistered.txt"
    extra.write_text("not part of the package\n", encoding="utf-8")
    with pytest.raises(ValueError, match="files must be exact"):
        all_auto.validate_output_package(first)


def test_registered_shard_write_reload_and_trusted_replay_use_same_config_schema(
    smoke_data: base.GenotypeInput,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_validate_input = all_auto._validate_input

    def validate_smoke_as_formal(
        data: base.GenotypeInput,
        *,
        require_formal_cohort: bool,
        chunk_size: int,
    ) -> None:
        assert require_formal_cohort is True
        original_validate_input(
            data,
            require_formal_cohort=False,
            chunk_size=chunk_size,
        )

    runtime_identity = all_auto._blas_runtime_identity()
    monkeypatch.setattr(all_auto, "_validate_input", validate_smoke_as_formal)
    monkeypatch.setattr(
        all_auto,
        "_require_formal_blas_thread_environment",
        lambda: runtime_identity,
    )
    monkeypatch.setattr(
        all_auto,
        "validate_result_payload",
        lambda payload: {"analysis_sha256": payload["analysis_sha256"]},
    )
    input_identity = all_auto._synthetic_input_identity(smoke_data)
    input_identity.update(
        {
            "input_scope": "all_autosomes",
            "external_source_manifest_anchor_verified": True,
        }
    )
    source_identity = all_auto._source_identity(REPO_ROOT)
    payload = all_auto.evaluate_all_autosome_stratification(
        smoke_data,
        all_auto.AllAutosomeConfig(
            seeds=(2025,),
            analysis_mode="registered_shard",
        ),
        input_identity=input_identity,
        source_identity=source_identity,
    )
    package = tmp_path / "registered_shard"
    all_auto.write_output_package(payload, package)
    reloaded = all_auto._strict_json_load(package / all_auto.RESULT_FILENAME)
    assert reloaded["config"] == payload["config"]
    assert reloaded["config"]["component_grid"] == [4, 6, 8]
    assert reloaded["config"]["panel_fractions"][-1] is None

    @contextmanager
    def fake_formal_input(*args: object, **kwargs: object):
        yield smoke_data, input_identity

    monkeypatch.setattr(all_auto, "load_formal_input", fake_formal_input)
    manifest = all_auto._strict_json_load(package / all_auto.MANIFEST_FILENAME)
    validation = all_auto.validate_output_package(
        package,
        repo_root=REPO_ROOT,
        expected_analysis_sha256=payload["analysis_sha256"],
        expected_package_anchor_sha256=manifest["package_anchor_sha256"],
        genotypes_path=tmp_path / "inert.npz",
        source_manifest_path=tmp_path / "inert.manifest.json",
        expected_source_manifest_sha256="0" * 64,
        scratch_root=tmp_path,
    )
    assert validation["trusted_input_replay_verified"] is True
    assert validation["trusted_scientific_replay_verified"] is True

    tampered = copy.deepcopy(payload)
    tampered["runtime_versions"]["python"] = "coherently-resigned-tamper"
    _resign(tampered)
    tampered_package = tmp_path / "registered_shard_tampered"
    all_auto.write_output_package(tampered, tampered_package)
    tampered_manifest = all_auto._strict_json_load(
        tampered_package / all_auto.MANIFEST_FILENAME
    )
    with pytest.raises(ValueError, match="trusted-input replay mismatch"):
        all_auto.validate_output_package(
            tampered_package,
            repo_root=REPO_ROOT,
            expected_analysis_sha256=tampered["analysis_sha256"],
            expected_package_anchor_sha256=tampered_manifest["package_anchor_sha256"],
            genotypes_path=tmp_path / "inert.npz",
            source_manifest_path=tmp_path / "inert.manifest.json",
            expected_source_manifest_sha256="0" * 64,
            scratch_root=tmp_path,
        )


def test_package_validator_rejects_csv_tampering(
    smoke_payload: dict[str, object], tmp_path: Path
) -> None:
    package = tmp_path / "package"
    all_auto.write_output_package(smoke_payload, package)
    path = package / all_auto.METRICS_FILENAME
    path.write_bytes(path.read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="artifact receipt mismatch"):
        all_auto.validate_output_package(package)


def test_package_anchor_rejects_resigned_source_provenance_tampering(
    smoke_payload: dict[str, object], tmp_path: Path
) -> None:
    package = tmp_path / "package"
    all_auto.write_output_package(smoke_payload, package)
    path = package / all_auto.RESULT_FILENAME
    attacked = json.loads(path.read_text(encoding="utf-8"))
    attacked["source_identity"]["git_identity"]["head_commit"] = "0" * 40
    _resign(attacked)
    path.write_bytes(all_auto._strict_json_bytes(attacked))
    with pytest.raises(ValueError, match="artifact receipt mismatch"):
        all_auto.validate_output_package(package)


def test_registered_shard_seed_parser_is_fail_closed() -> None:
    assert all_auto._parse_registered_seeds("2025,2027,2044") == (2025, 2027, 2044)
    for value in ("", "2027,2025", "2025,2025", "2045", "2025,x"):
        with pytest.raises(Exception):
            all_auto._parse_registered_seeds(value)


def test_source_mentions_original_lineage_and_no_claimed_import_exact_cover() -> None:
    source = all_auto._source_identity(REPO_ROOT)
    assert set(source["files"]) == set(all_auto.SOURCE_FILE_ROLES)
    assert source["import_graph_exact_cover_claimed"] is False
    assert source["roles"]["src/rspca/genomics_aim.py"] == (
        "historical_split_and_smartpca_lineage"
    )
    lineage = source["historical_github_lineage"]
    assert lineage["git_objects_and_hashes_verified"] is True
    assert lineage["historical_sparse_robust_randomized_branches_executed"] is False
    assert set(lineage["files"]) == set(all_auto.LEGACY_SOURCE_SHA256)


def test_plain_source_snapshot_records_weaker_lineage_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(all_auto, "_exact_git_toplevel", lambda _: None)
    source = all_auto._source_identity(REPO_ROOT)
    assert source["git_identity"]["git_metadata_available"] is False
    lineage = source["historical_github_lineage"]
    assert lineage["verification_mode"] == "embedded_lock_only_plain_source_snapshot"
    assert lineage["git_objects_and_hashes_verified"] is False
    assert lineage["embedded_lineage_lock_verified"] is True
    assert lineage["plain_snapshot_does_not_reauthenticate_historical_git_objects"]


def _write_canonical_npz(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with path.open("xb") as handle:
        with zipfile.ZipFile(
            handle, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for key in all_auto.NPZ_KEYS:
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                with archive.open(
                    info, mode="w", force_zip64=(key == "genotypes")
                ) as member:
                    import io

                    buffer = io.BytesIO()
                    np.lib.format.write_array(
                        buffer, np.asarray(arrays[key]), allow_pickle=False
                    )
                    payload = buffer.getvalue()
                    member.write(payload)
                    hashes[key] = all_auto._bytes_sha256(payload)
    return hashes


def test_formal_loader_verifies_canonical_npz_then_uses_genotype_memmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    labels = np.asarray(
        [
            population
            for population in all_auto.EXPECTED_POPULATIONS
            for _ in range(200)
        ],
        dtype="<U3",
    )
    chromosomes = np.asarray(
        [chromosome for chromosome in all_auto.EXPECTED_AUTOSOMES for _ in range(2)],
        dtype="<U2",
    )
    rng = np.random.default_rng(9001)
    genotypes = rng.binomial(2, 0.3, size=(1000, chromosomes.size)).astype("<f4")
    arrays = {
        "chromosomes": chromosomes,
        "genotypes": genotypes,
        "population_labels": labels,
        "sample_ids": np.asarray(
            [f"S{index:06d}" for index in range(1000)], dtype="<U7"
        ),
        "snp_ids": np.asarray(
            [
                f"{chromosome}:{index + 1}:A:G"
                for index, chromosome in enumerate(chromosomes)
            ],
            dtype="<U16",
        ),
    }
    npz_path = tmp_path / "genomics_aim_all_autosomes.npz"
    array_hashes = _write_canonical_npz(npz_path, arrays)
    per_chromosome = {
        chromosome: {
            "commands": [
                {
                    "argv": ["/inert/plink2", f"--receipt-{index}"],
                    "cwd": "/inert/work",
                    "environment_overrides": {
                        "LANG": "C",
                        "LC_ALL": "C",
                        "TZ": "UTC",
                    },
                    "returncode": 0,
                    "role": f"chr{chromosome}_receipt_{index}",
                    "stderr_sha256": "0" * 64,
                    "stdout_sha256": str(index) * 64,
                }
                for index in range(1, 5)
            ],
            "ephemeral_artifacts": {
                role: {"bytes": 1, "sha256": "5" * 64}
                for role in ("decompressed_pgen", "export_raw", "qc_pvar")
            },
            "ld_pruned_variant_count": 2,
            "qc_variant_count": 3,
            "retained_prune_in": {
                "path": f"prune_in/chr{chromosome}.prune.in",
                "bytes": 1,
                "sha256": "1" * 64,
                "variant_count": 2,
                "variant_ids_sha256": "2" * 64,
            },
            "source_inputs": {
                role: {"path": f"/inert/{role}", "bytes": 1, "sha256": "6" * 64}
                for role in ("pgen_zst", "psam", "pvar_zst")
            },
        }
        for chromosome in all_auto.EXPECTED_AUTOSOMES
    }
    output = {
        "path": "genomics_aim_all_autosomes.npz",
        "sha256": all_auto._file_sha256(npz_path),
        "bytes": npz_path.stat().st_size,
        "npz_keys": list(all_auto.NPZ_KEYS),
        "npz_container": {
            "format": "zip_npz",
            "compression": "ZIP_DEFLATED",
            "compresslevel": 6,
            "member_order": [f"{key}.npy" for key in all_auto.NPZ_KEYS],
            "member_timestamp": [1980, 1, 1, 0, 0, 0],
            "member_mode_octal": "0600",
            "genotypes_force_zip64": True,
        },
        "shape": [1000, int(chromosomes.size)],
        "dtype": "<f4",
        "missing_count": 0,
        "total_qc_variants": int(chromosomes.size),
        "array_sha256": array_hashes,
        "array_receipts": {
            key: {
                "sha256": array_hashes[key],
                "shape": list(arrays[key].shape),
                "dtype": arrays[key].dtype.str,
            }
            for key in all_auto.NPZ_KEYS
        },
    }
    monkeypatch.setattr(all_auto, "FROZEN_FORMAL_NPZ_SHA256", output["sha256"])
    monkeypatch.setattr(all_auto, "FROZEN_FORMAL_NPZ_BYTES", output["bytes"])
    monkeypatch.setattr(
        all_auto, "FROZEN_FORMAL_SHAPE", tuple(int(value) for value in output["shape"])
    )
    monkeypatch.setattr(
        all_auto, "FROZEN_FORMAL_TOTAL_QC_VARIANTS", output["total_qc_variants"]
    )
    monkeypatch.setattr(all_auto, "FROZEN_PREPARED_ARRAY_SHA256", dict(array_hashes))
    manifest = {
        "schema_version": all_auto.FORMAL_MANIFEST_SCHEMA,
        "preparation": {
            "tool": {
                "path": "/inert/prepare.py",
                "bytes": 1,
                "sha256": all_auto.FROZEN_PREPARER_SOURCE_SHA256,
            },
            "runtime_dependencies": {
                "legacy_preparer": {
                    "path": "/inert/legacy.py",
                    "bytes": 1,
                    "sha256": all_auto.FROZEN_LEGACY_PREPARER_SOURCE_SHA256,
                },
                "source_downloader": {
                    "path": "/inert/download.py",
                    "bytes": 1,
                    "sha256": all_auto.FROZEN_DOWNLOADER_SOURCE_SHA256,
                },
            },
            "large_intermediate_policy": (
                "one_chromosome_at_a_time; decompressed PGEN, QC fileset, RAW, "
                "prune.out, and logs deleted before the next chromosome"
            ),
            "memory_strategy": (
                "one chromosome float32 NPY memmap at a time; final genotype NPY "
                "member written row-wise across 22 disk blocks; validator extracts "
                "that member to a temporary memmap and checks it in row chunks"
            ),
            "disk_preflight_policy": (
                "per chromosome require 8*pgen_zst_bytes + 4*pvar_zst_bytes + "
                "8GiB free; before NPZ require 2*uncompressed genotype bytes + 3GiB; "
                "before validation extraction require member bytes + 1GiB"
            ),
            "retained_intermediates": "exact prune.in files only",
        },
        "dataset": {
            "input_scope": "all_autosomes",
            "autosomes_present": list(all_auto.EXPECTED_AUTOSOMES),
            "orientation": "samples_by_snps",
            "population_design": "five_superpopulation",
            "population_label_semantics": "1000_Genomes_superpopulation",
            "population_label_is_race": False,
            "ld_pruned": True,
        },
        "sources": {
            "external_input_anchors": {
                "expected_source_manifest_sha256": (
                    all_auto.FROZEN_UPSTREAM_SOURCE_MANIFEST_SHA256
                ),
                "expected_downloader_source_sha256": (
                    all_auto.FROZEN_DOWNLOADER_SOURCE_SHA256
                ),
                "expected_download_completion_sha256": (
                    all_auto.FROZEN_DOWNLOAD_COMPLETION_SHA256
                ),
            },
            "source_manifest": {
                "path": "/inert/source_manifest.json",
                "sha256": all_auto.FROZEN_UPSTREAM_SOURCE_MANIFEST_SHA256,
                "payload_sha256": "7" * 64,
            },
            "download_directory": {
                "path": "/inert/sources",
                "completion_manifest_sha256": (
                    all_auto.FROZEN_DOWNLOAD_COMPLETION_SHA256
                ),
                "completion_payload_sha256": "8" * 64,
                "validation_status": "valid_and_externally_anchored",
                "file_count": 45,
                "total_bytes": 1,
            },
            "authentication_boundary": "inert receipt text",
        },
        "cohort_anchor": {
            "legacy_preparation_manifest": {
                "path": "/inert/legacy_manifest.json",
                "sha256": all_auto.FROZEN_LEGACY_PREPARATION_MANIFEST_SHA256,
            },
            "selected_count": 1000,
            "selected_counts_by_group": {
                population: 200 for population in all_auto.EXPECTED_POPULATIONS
            },
            "selected_sample_ids_sha256": all_auto.FROZEN_SELECTED_SAMPLE_IDS_SHA256,
            "selected_records_sha256": all_auto.FROZEN_SELECTED_RECORDS_SHA256,
            "psam_check": {
                "rows": 1000,
                "selected_samples_present": 1000,
                "superpopulation_column_checked": True,
                "selected_sample_ids_sha256": (
                    all_auto.FROZEN_SELECTED_SAMPLE_IDS_SHA256
                ),
            },
        },
        "plink2": {
            "path": "/inert/plink2",
            "bytes": 1,
            "sha256": all_auto.FROZEN_PLINK2_SHA256,
            "version": all_auto.FROZEN_PLINK2_VERSION,
        },
        "variant_processing": {
            "policy": "genuine_qc_then_ld_prune_exact_prune_in_only",
            "cohort_scope": "same_frozen_1000_reference_samples_before_outer_split",
            "population_labels_used_for_variant_qc_or_ld": False,
            "variant_target": None,
            "padding_with_unpruned_variants": False,
            "hwe_filter": None,
            "maf": 0.05,
            "maximum_variant_missingness": 0.05,
            "biallelic_snv_only": True,
            "duplicate_variant_policy": "plink2_rm_dup_exclude_all",
            "ld_pruning": {
                "method": "plink2_indep_pairwise",
                "window_variants": 50,
                "step_variants": 5,
                "r2_threshold": 0.2,
                "indep_order": 2,
            },
            "per_chromosome": per_chromosome,
        },
        "downstream_label_boundary": {
            "allowed": [
                "stratified_train_test_split",
                "training_centroids_after_panel_freeze",
                "held_out_scoring",
            ],
            "forbidden": [
                "preprocessing_fit",
                "component_fit",
                "loading_based_panel_ranking",
                "panel_refit",
            ],
            "population_labels_used_for_variant_qc_or_ld": False,
            "panel_selection_used_population_labels": False,
        },
        "output": output,
    }
    manifest_path = tmp_path / "genomics_aim_all_autosomes.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = all_auto._file_sha256(manifest_path)
    monkeypatch.setattr(all_auto, "FROZEN_FORMAL_MANIFEST_SHA256", manifest_sha)
    empty_preparation = copy.deepcopy(manifest)
    empty_preparation["preparation"] = {}
    with pytest.raises(ValueError, match="preparation schema"):
        all_auto._validate_formal_manifest_structure(empty_preparation)
    with all_auto.load_formal_input(
        npz_path,
        manifest_path,
        expected_source_manifest_sha256=manifest_sha,
        scratch_root=tmp_path,
    ) as (data, identity):
        assert isinstance(data.genotypes, np.memmap)
        assert data.genotypes.shape == genotypes.shape
        assert data.genotypes.dtype.str == "<f4"
        assert identity["external_source_manifest_anchor_verified"] is True
        assert identity["raw_input_memory_strategy"].endswith("read by mmap")
        np.testing.assert_array_equal(data.genotypes[:3], genotypes[:3])
    with pytest.raises(ValueError, match="frozen formal anchor"):
        with all_auto.load_formal_input(
            npz_path,
            manifest_path,
            expected_source_manifest_sha256="0" * 64,
            scratch_root=tmp_path,
        ):
            pass

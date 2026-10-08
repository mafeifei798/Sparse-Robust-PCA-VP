from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import genomics_pca_loading_grid as diagnostic
from experiments.genomics_pca_loading_grid import (
    GenotypeInput,
    PCALoadingGridConfig,
    evaluate_loading_grid,
    fit_top_pca,
    rank_by_pca_loadings,
    validate_output_package,
    validate_result_payload,
    write_output_package,
)
from rspca.datasets import canonical_sha256
from rspca.genomics_aim import stratified_split


def _dataset() -> GenotypeInput:
    rng = np.random.default_rng(811)
    populations = diagnostic.EXPECTED_POPULATIONS
    labels = np.asarray(
        [population for population in populations for _ in range(12)], dtype=str
    )
    probabilities = np.full((labels.size, 24), 0.35)
    for index, population in enumerate(populations):
        rows = labels == population
        probabilities[rows, 2 * index : 2 * index + 2] = 0.82
        probabilities[rows, 10 + index : 15 + index] = 0.12 + 0.12 * index
    genotypes = rng.binomial(2, probabilities).astype(float)
    return GenotypeInput(
        genotypes=genotypes,
        sample_ids=np.asarray([f"sample_{index:03d}" for index in range(labels.size)]),
        snp_ids=np.asarray([f"22:{index + 1}:A:G" for index in range(genotypes.shape[1])]),
        chromosomes=np.asarray(["22"] * genotypes.shape[1]),
        population_labels=labels,
    )


def _config() -> PCALoadingGridConfig:
    return PCALoadingGridConfig(
        seeds=(2025, 2026),
        component_grid=(2, 3, 4),
        panel_sizes=(6, 10, None),
        test_fraction=0.25,
    )


def _payload(data: GenotypeInput | None = None) -> dict[str, object]:
    return evaluate_loading_grid(
        data or _dataset(),
        _config(),
        input_identity={"fixture": "synthetic_dosage"},
        source_identity={"fixture": "test"},
    )


def test_top_pca_matches_numpy_svd_subspaces() -> None:
    rng = np.random.default_rng(104)
    for shape in ((18, 31), (31, 18)):
        matrix = rng.normal(size=shape)
        loadings, singular_values = fit_top_pca(matrix, 4)
        _, expected_singular, expected_vt = np.linalg.svd(matrix, full_matrices=False)
        np.testing.assert_allclose(singular_values, expected_singular[:4], rtol=1e-10)
        np.testing.assert_allclose(
            loadings @ loadings.T,
            expected_vt[:4].T @ expected_vt[:4],
            atol=2e-10,
        )


def test_loading_ranking_is_label_free_and_deterministic() -> None:
    assert "label" not in inspect.signature(rank_by_pca_loadings).parameters
    loadings = np.asarray(
        [
            [2.0, 0.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, 3.0, 0.0],
        ]
    )
    first, scores = rank_by_pca_loadings(loadings, 2)
    second, repeated_scores = rank_by_pca_loadings(loadings.copy(), 2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(scores, repeated_scores)
    assert first[:2].tolist() == [0, 1]


def test_complete_grid_and_registered_coordinates_have_no_best_selection() -> None:
    payload = _payload()
    validation = validate_result_payload(payload)
    assert validation["records"] == 2 * 3 * 4
    assert validation["grid_summaries"] == 3 * 4
    assert payload["analysis_policy"]["status"] == "post_hoc_development_diagnostic"
    assert payload["analysis_policy"]["confirmatory"] is False
    assert payload["analysis_policy"]["best_configuration_selected"] is False
    assert all(
        ranking["population_labels_used"] is False
        for ranking in payload["ranking_records"]
    )
    panels = {row["panel_label"] for row in payload["registered_coordinates"]}
    assert panels == {"6", "10", "all"}
    assert {
        row["row_type"] for row in payload["registered_coordinates"]
    } == {"held_out_sample", "train_centroid"}
    assert {row["n_components"] for row in payload["registered_coordinates"]} == {
        2,
        4,
    }


def test_loading_panels_do_not_change_when_only_heldout_genotypes_change() -> None:
    data = _dataset()
    _, heldout = stratified_split(
        data.population_labels, _config().test_fraction, _config().seeds[0]
    )
    mutated_matrix = data.genotypes.copy()
    mutated_matrix[heldout] = 2.0 - mutated_matrix[heldout]
    mutated = replace(data, genotypes=mutated_matrix)
    first = _payload(data)
    second = _payload(mutated)
    first_rankings = [
        (
            row["seed"],
            row["n_components"],
            row["ordered_original_indices_sha256"],
            row["loading_importance_sha256"],
        )
        for row in first["ranking_records"]
        if row["seed"] == _config().seeds[0]
    ]
    second_rankings = [
        (
            row["seed"],
            row["n_components"],
            row["ordered_original_indices_sha256"],
            row["loading_importance_sha256"],
        )
        for row in second["ranking_records"]
        if row["seed"] == _config().seeds[0]
    ]
    assert first_rankings == second_rankings


def test_all_panel_is_exact_full_width_replay() -> None:
    payload = _payload()
    rows = payload["records"]
    for seed in _config().seeds:
        for components in _config().component_grid:
            full = next(
                row
                for row in rows
                if row["seed"] == seed
                and row["n_components"] == components
                and row["evaluation"] == "full_width"
            )
            all_panel = next(
                row
                for row in rows
                if row["seed"] == seed
                and row["n_components"] == components
                and row["evaluation"] == "loading_panel"
                and row["panel_label"] == "all"
            )
            assert full["test_predictions_sha256"] == all_panel["test_predictions_sha256"]
            assert full["confusion_matrix_counts"] == all_panel["confusion_matrix_counts"]
            assert full["balanced_accuracy"] == all_panel["balanced_accuracy"]


def test_negative_or_noninteger_confusion_counts_are_rejected() -> None:
    payload = json.loads(json.dumps(_payload()))
    payload["records"][0]["confusion_matrix_counts"][0][1] = -1
    with pytest.raises(ValueError, match="non-negative integer 5x5"):
        validate_result_payload(payload)
    payload = json.loads(json.dumps(_payload()))
    payload["records"][0]["confusion_matrix_counts"][0][1] = 1.5
    with pytest.raises(ValueError, match="non-negative integer 5x5"):
        validate_result_payload(payload)


def test_output_package_is_exact_and_rejects_artifact_tampering(tmp_path: Path) -> None:
    payload = _payload()
    output = tmp_path / "package"
    write_output_package(payload, output)
    validation = validate_output_package(output)
    assert validation["status"] == "valid"
    assert validation["package_files"] == 7
    with pytest.raises(ValueError, match="external analysis-hash anchor"):
        validate_output_package(
            output,
            expected_analysis_sha256="0" * 64,
            expected_package_anchor_sha256="0" * 64,
        )
    metrics = output / diagnostic.METRICS_FILENAME
    metrics.write_bytes(metrics.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        validate_output_package(output)


def test_resigned_policy_tampering_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "package"
    write_output_package(_payload(), output)
    result_path = output / diagnostic.RESULT_FILENAME
    payload = json.loads(result_path.read_text())
    payload["analysis_policy"]["confirmatory"] = True
    unsigned = dict(payload)
    unsigned.pop("analysis_sha256")
    payload["analysis_sha256"] = canonical_sha256(unsigned)
    result_path.write_bytes(diagnostic._strict_json_bytes(payload))
    manifest_path = output / diagnostic.MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["analysis_sha256"] = payload["analysis_sha256"]
    manifest["artifacts"][diagnostic.RESULT_FILENAME] = {
        "bytes": result_path.stat().st_size,
        "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
    }
    manifest["package_anchor_sha256"] = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    unsigned_manifest = dict(manifest)
    unsigned_manifest.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_sha256(unsigned_manifest)
    manifest_path.write_bytes(diagnostic._strict_json_bytes(manifest))
    with pytest.raises(ValueError, match="analysis-policy boundary"):
        validate_output_package(output)


def test_external_package_anchor_covers_resigned_png(tmp_path: Path) -> None:
    from PIL import Image

    output = tmp_path / "package"
    payload = _payload()
    write_output_package(payload, output)
    original_manifest = json.loads(
        (output / diagnostic.MANIFEST_FILENAME).read_text()
    )
    figure_path = output / diagnostic.PC12_FILENAME
    Image.new("RGB", (2500, 700), "white").save(figure_path, format="PNG")
    manifest = dict(original_manifest)
    manifest["artifacts"] = dict(manifest["artifacts"])
    manifest["artifacts"][diagnostic.PC12_FILENAME] = {
        "bytes": figure_path.stat().st_size,
        "sha256": hashlib.sha256(figure_path.read_bytes()).hexdigest(),
    }
    manifest["package_anchor_sha256"] = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    unsigned = dict(manifest)
    unsigned.pop("manifest_payload_sha256")
    manifest["manifest_payload_sha256"] = canonical_sha256(unsigned)
    (output / diagnostic.MANIFEST_FILENAME).write_bytes(
        diagnostic._strict_json_bytes(manifest)
    )
    with pytest.raises(ValueError, match="external package-hash anchor"):
        validate_output_package(
            output,
            expected_analysis_sha256=payload["analysis_sha256"],
            expected_package_anchor_sha256=original_manifest[
                "package_anchor_sha256"
            ],
        )


def test_package_rejects_extra_file_and_symlink(tmp_path: Path) -> None:
    output = tmp_path / "package"
    write_output_package(_payload(), output)
    (output / "extra.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="must be exact"):
        validate_output_package(output)
    (output / "extra.txt").unlink()
    metrics = output / diagnostic.METRICS_FILENAME
    metrics.unlink()
    metrics.symlink_to(output / diagnostic.RESULT_FILENAME)
    with pytest.raises(ValueError, match="symlink"):
        validate_output_package(output)


def test_config_rejects_non_posthoc_status() -> None:
    with pytest.raises(ValueError, match="post_hoc"):
        evaluate_loading_grid(
            _dataset(),
            replace(_config(), analysis_status="confirmatory"),
            input_identity={},
            source_identity={},
        )


def test_external_manifest_must_bind_npz_and_ld_pruning(tmp_path: Path) -> None:
    data = _dataset()
    npz = tmp_path / "input.npz"
    np.savez_compressed(
        npz,
        genotypes=data.genotypes,
        sample_ids=data.sample_ids,
        snp_ids=data.snp_ids,
        chromosomes=data.chromosomes,
        population_labels=data.population_labels,
    )
    manifest = {
        "schema_version": "rspca-1kg-phase3-v5b-chr20-22-preparation-v1",
        "dataset": {
            "population_design": "five_superpopulation",
            "orientation": "samples_by_snps",
            "ld_pruned": True,
            "input_scope": "legacy_chr20_22",
            "population_label_is_race": False,
        },
        "output": {
            "sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
            "bytes": npz.stat().st_size,
            "shape": list(data.genotypes.shape),
            "npz_keys": [
                "chromosomes",
                "genotypes",
                "population_labels",
                "sample_ids",
                "snp_ids",
            ],
            "array_sha256": {
                "chromosomes": diagnostic._npy_array_sha256(data.chromosomes),
                "genotypes": diagnostic._npy_array_sha256(data.genotypes),
                "population_labels": diagnostic._npy_array_sha256(
                    data.population_labels
                ),
                "sample_ids": diagnostic._npy_array_sha256(data.sample_ids),
                "snp_ids": diagnostic._npy_array_sha256(data.snp_ids),
            },
        },
        "preparation_tool": {
            "sha256": "1" * 64,
            "configuration": {
                "population_design": "five_superpopulation",
                "ld_window": 50,
                "ld_step": 5,
                "ld_r2": 0.2,
                "maf": 0.05,
                "geno": 0.05,
            },
        },
        "variant_processing": {
            "padding_with_unpruned_variants": False,
            "population_labels_used_for_variant_qc_or_ld": False,
            "ld_pruning": {
                "method": "plink2_indep_pairwise",
                "window": 50,
                "step_variants": 5,
                "r2_threshold": 0.2,
                "indep_order": 2,
            },
            "per_chromosome": {
                chromosome: {
                    "commands": [
                        {
                            "role": f"chr{chromosome}_import_qc",
                            "returncode": 0,
                            "stdout_sha256": "2" * 64,
                            "stderr_sha256": "3" * 64,
                            "argv": ["plink2", "--vcf"],
                        },
                        {
                            "role": f"chr{chromosome}_ld_prune",
                            "returncode": 0,
                            "stdout_sha256": "4" * 64,
                            "stderr_sha256": "5" * 64,
                            "argv": ["plink2", "--indep-pairwise"],
                        },
                        {
                            "role": f"chr{chromosome}_export_pruned",
                            "returncode": 0,
                            "stdout_sha256": "6" * 64,
                            "stderr_sha256": "7" * 64,
                            "argv": ["plink2", "--extract"],
                        },
                    ]
                }
                for chromosome in ("20", "21", "22")
            },
        },
        "plink2": {
            "sha256": "8" * 64,
            "version_command": {"returncode": 0},
        },
        "downstream_label_boundary": {
            "panel_selection_used_population_labels": False,
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    receipt = diagnostic.validate_external_source_manifest(
        manifest_path,
        npz,
        data,
        expected_manifest_sha256=manifest_sha,
    )
    assert receipt["manifest_ld_pruned_receipts_verified"] is True
    manifest["dataset"]["ld_pruned"] = False
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="required input"):
        diagnostic.validate_external_source_manifest(
            manifest_path,
            npz,
            data,
            expected_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )

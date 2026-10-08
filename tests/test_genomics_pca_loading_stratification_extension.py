from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import json
from pathlib import Path
import subprocess
import uuid

import numpy as np
import pytest

from experiments import genomics_pca_loading_stratification_extension as extension
from rspca.datasets import array_sha256, canonical_sha256
from rspca.genomics_aim import stratified_split


REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(*, seeds: tuple[int, ...] = (73, 74)) -> extension.StratificationExtensionConfig:
    return extension.StratificationExtensionConfig(
        seeds=seeds,
        panel_sizes=(400, 800, 1600, None),
        analysis_mode="tiny_smoke",
    )


def _identity(data: extension.base.GenotypeInput) -> dict[str, object]:
    return {
        "fixture": "synthetic_smoke_only",
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "orientation": "samples_by_snps",
    }


def _evaluate(
    data: extension.base.GenotypeInput | None = None,
    *,
    config: extension.StratificationExtensionConfig | None = None,
) -> dict[str, object]:
    current = data or extension._synthetic_smoke_data()
    return extension.evaluate_stratification_extension(
        current,
        config or _config(),
        input_identity=_identity(current),
        source_identity=extension._source_identity(REPO_ROOT),
    )


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return _evaluate()


def _resign(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("analysis_sha256")
    value["analysis_sha256"] = canonical_sha256(unsigned)


def _coherently_resign_seed_74_attack(
    clean: dict[str, object],
) -> dict[str, object]:
    """Alter seed 74 and rebuild every downstream numerical receipt."""

    attacked = json.loads(json.dumps(clean))
    classes = np.asarray(extension.EXPECTED_POPULATIONS, dtype=str)
    for record in attacked["records"]:
        if record["seed"] != 74:
            continue
        rows = sorted(
            (
                row
                for row in attacked["sample_prediction_evidence"]
                if row["seed"] == 74
                and row["n_components"] == record["n_components"]
                and row["panel_label"] == record["panel_label"]
            ),
            key=lambda row: row["sample_position"],
        )
        first = rows[0]
        old_position = extension.EXPECTED_POPULATIONS.index(first["prediction"])
        first["prediction"] = extension.EXPECTED_POPULATIONS[
            (old_position + 1) % len(extension.EXPECTED_POPULATIONS)
        ]
        first["correct"] = first["truth"] == first["prediction"]
        truth = np.asarray(
            [row["truth"] for row in rows], dtype=np.dtype(rows[0]["test_labels_dtype"])
        )
        predictions = np.asarray(
            [row["prediction"] for row in rows],
            dtype=np.dtype(record["test_predictions_dtype"]),
        )
        metrics = extension.base._classification_metrics(truth, predictions, classes)
        record.update(metrics)
        record["test_predictions_sha256"] = array_sha256(predictions)
        for row in rows:
            row["test_predictions_sha256"] = record["test_predictions_sha256"]

    config = _config()
    grid, populations = extension._summaries(attacked["records"], config)
    attacked["grid_summaries"] = grid
    attacked["population_summaries"] = populations
    attacked["selection_decision"] = extension._selection_decision(
        grid, populations, config
    )
    candidate = attacked["selection_decision"]["selected_candidate"]
    attacked["registered_seed_policy"]["rank"] = (
        int(candidate["n_components"])
        if candidate is not None
        else max(config.component_grid)
    )
    attacked["confirmation_proposal_policy"]["eligible"] = False
    _resign(attacked)
    return attacked


def test_runner_imports_no_alternative_scientific_solver() -> None:
    path = REPO_ROOT / "experiments/genomics_pca_loading_stratification_extension.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    assert not any(
        module in {
            "rspca.vp",
            "rspca.robust",
            "rspca.randomized",
            "rspca.federated",
            "rspca.fedrpca_audit",
        }
        for module in modules
    )
    assert extension.METHOD_SCOPE == {
        "methods": ["pca_loading_ranking"],
        "robust_solver_run": False,
        "vp_solver_run": False,
        "randomized_solver_run": False,
        "federated_solver_run": False,
    }


def test_complete_grid_selection_and_multi_pc_policy(payload: dict[str, object]) -> None:
    validation = extension.validate_result_payload(payload)
    assert validation["records"] == 2 * 3 * 4
    assert validation["grid_cells"] == 3 * 4
    assert payload["analysis_policy"][
        "population_labels_used_for_preprocessing_pca_or_loading_ranking"
    ] is False
    assert payload["analysis_policy"][
        "test_rows_used_for_preprocessing_pca_or_loading_ranking"
    ] is False
    assert "no single 2D projection" in payload["analysis_policy"][
        "multi_pc_interpretation"
    ]
    assert payload["selection_decision"]["candidate_order"] == (
        "smallest_eligible_panel_from_800_then_smallest_rank"
    )
    assert payload["selection_decision"]["top_400_role"] == (
        "compression_sensitivity_only_not_selection_eligible"
    )
    top_400 = [
        row
        for row in payload["selection_decision"]["candidate_evaluations"]
        if row["panel_label"] == "400"
    ]
    assert len(top_400) == 3
    assert all(row["selection_eligible"] is False for row in top_400)
    assert all(row["qualifies"] is False for row in top_400)
    assert len(payload["selection_decision"]["candidate_evaluations"]) == 12
    assert payload["confirmation_proposal_policy"]["eligible"] is False
    assert payload["confirmation_proposal_policy"]["confirmation_outcomes_accessed"] is False
    assert payload["source_identity"]["loaded_repo_python_files_exact_cover"] is True
    assert set(payload["source_identity"]["files"]) == set(extension.SOURCE_FILE_ROLES)
    assert payload["source_identity"]["loaded_repo_python_files"] == sorted(
        extension.SOURCE_FILE_ROLES
    )
    git_identity = payload["source_identity"]["git_identity"]
    assert len(git_identity["head_commit"]) == 40
    assert len(git_identity["head_tree"]) == 40
    assert git_identity["repo_root_binding"] == (
        "resolved_runner_parent_equals_git_toplevel"
    )
    assert isinstance(git_identity["tracked_dirty"], bool)
    assert isinstance(git_identity["untracked"], bool)
    expected_sample_rows = sum(
        sum(sum(row) for row in record["confusion_matrix_counts"])
        for record in payload["records"]
    )
    assert len(payload["sample_prediction_evidence"]) == expected_sample_rows


def test_default_development_grid_is_frozen() -> None:
    default = extension.StratificationExtensionConfig()
    assert default.seeds == tuple(range(2025, 2030))
    assert default.component_grid == (4, 6, 8)
    assert default.panel_sizes == (400, 800, 1600, 3200, None)
    assert default.test_fraction == 0.2
    extension._validate_config(default)
    with pytest.raises(ValueError, match="development_protocol configuration is frozen"):
        extension._validate_config(replace(default, seeds=(2025,)))
    with pytest.raises(ValueError, match="development_protocol configuration is frozen"):
        extension._validate_config(replace(default, panel_sizes=(800, 1600, None)))


def test_confirmation_seeds_are_rejected_for_every_analysis_profile() -> None:
    for mode in ("tiny_smoke", "development_protocol"):
        config = extension.StratificationExtensionConfig(
            seeds=(5001,),
            panel_sizes=(400, 800, 1600, None),
            analysis_mode=mode,
        )
        with pytest.raises(ValueError, match="reserved confirmation seeds"):
            extension._validate_config(config)
    data = extension._synthetic_smoke_data()
    with pytest.raises(ValueError, match="reserved confirmation seeds"):
        extension.evaluate_stratification_extension(
            data,
            _config(seeds=(73, 5007)),
            input_identity=_identity(data),
            source_identity=extension._source_identity(REPO_ROOT),
        )


def test_all_five_development_seed_ids_emit_sample_level_evidence() -> None:
    five_seed_payload = _evaluate(
        config=_config(seeds=extension.DEVELOPMENT_SEEDS)
    )
    evidence = five_seed_payload["sample_prediction_evidence"]
    assert {row["seed"] for row in evidence} == set(extension.DEVELOPMENT_SEEDS)
    cells = {
        (row["seed"], row["n_components"], row["panel_label"])
        for row in evidence
    }
    assert len(cells) == 5 * 3 * 4
    assert all(
        len(
            [
                row
                for row in evidence
                if (row["seed"], row["n_components"], row["panel_label"])
                == cell
            ]
        )
        == 20
        for cell in cells
    )


def test_selection_rule_prefers_panel_before_rank() -> None:
    summaries = []
    populations = []
    for rank in extension.COMPONENT_GRID:
        for panel in (400, 800, 1600, None):
            label = extension._panel_label(panel)
            all_ba = 0.95 + 0.001 * (rank - 4)
            mean_ba = all_ba if panel is None else all_ba - 0.005
            summaries.append(
                {
                    "n_components": rank,
                    "panel_label": label,
                    "mean_balanced_accuracy": mean_ba,
                }
            )
            for population in extension.EXPECTED_POPULATIONS:
                recall = 0.99
                if population == "AMR":
                    recall = 0.69 if (panel == 400 and rank < 8) else 0.71
                populations.append(
                    {
                        "n_components": rank,
                        "panel_label": label,
                        "population": population,
                        "mean_recall": recall,
                    }
                )
    decision = extension._selection_decision(summaries, populations, _config())
    assert decision["selected_candidate"]["panel_label"] == "800"
    assert decision["selected_candidate"]["n_components"] == 4
    assert decision["candidate_evaluations"][0]["qualifies"] is False
    assert decision["candidate_evaluations"][2]["quality_gates_pass"] is True
    assert decision["candidate_evaluations"][2]["selection_eligible"] is False
    assert decision["candidate_evaluations"][3]["qualifies"] is True


def test_rankings_are_invariant_to_heldout_genotype_changes() -> None:
    data = extension._synthetic_smoke_data()
    config = _config(seeds=(73,))
    _, heldout = stratified_split(data.population_labels, 0.2, 73)
    mutated_matrix = data.genotypes.copy()
    mutated_matrix[heldout] = 2.0 - mutated_matrix[heldout]
    mutated = replace(data, genotypes=mutated_matrix)
    first = _evaluate(data, config=config)
    second = _evaluate(mutated, config=config)
    first_hashes = [
        (
            row["n_components"],
            row["ordered_original_indices_sha256"],
            row["loading_importance_sha256"],
        )
        for row in first["ranking_records"]
    ]
    second_hashes = [
        (
            row["n_components"],
            row["ordered_original_indices_sha256"],
            row["loading_importance_sha256"],
        )
        for row in second["ranking_records"]
    ]
    assert first_hashes == second_hashes


def test_input_identity_must_match_loaded_arrays() -> None:
    data = extension._synthetic_smoke_data()
    identity = _identity(data)
    identity["genotypes_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="input identity disagrees"):
        extension.evaluate_stratification_extension(
            data,
            _config(seeds=(73,)),
            input_identity=identity,
            source_identity=extension._source_identity(REPO_ROOT),
        )


def test_formal_prepared_manifest_anchor_is_code_fixed() -> None:
    identity = {
        "external_source_manifest_anchor_verified": True,
        "ld_pruned_input_validated_from_external_manifest": True,
        "manifest_population_design": "five_superpopulation",
        "manifest_input_scope": "legacy_chr20_22",
        "orientation": "samples_by_snps",
        "external_source_manifest_sha256": (
            extension.PREPARED_SOURCE_MANIFEST_SHA256
        ),
        "expected_external_source_manifest_sha256": (
            extension.PREPARED_SOURCE_MANIFEST_SHA256
        ),
        "genotype_npz_sha256": "1" * 64,
        "genotypes_sha256": "2" * 64,
        "sample_ids_sha256": "3" * 64,
        "snp_ids_sha256": "4" * 64,
        "chromosomes_sha256": "5" * 64,
        "population_labels_sha256": "6" * 64,
        "samples": 1000,
        "snps": 3200,
    }
    assert extension._input_is_lock_eligible(identity) is True
    identity["expected_external_source_manifest_sha256"] = "0" * 64
    assert extension._input_is_lock_eligible(identity) is False

    data = extension._synthetic_smoke_data()
    with pytest.raises(PermissionError, match="restricted to tiny_smoke"):
        extension.evaluate_stratification_extension(
            data,
            extension.StratificationExtensionConfig(),
            input_identity=_identity(data),
            source_identity=extension._source_identity(REPO_ROOT),
        )
    signature = inspect.signature(extension.evaluate_formal_development_from_paths)
    assert list(signature.parameters) == ["genotypes_path", "source_manifest_path"]


def test_repo_root_and_git_source_identity_are_fail_closed(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match="repo_root must resolve exactly"):
        extension._source_identity(tmp_path)
    tampered = json.loads(json.dumps(payload))
    tampered["source_identity"]["git_identity"]["head_tree"] = "0" * 40
    git_identity = tampered["source_identity"]["git_identity"]
    git_identity["git_identity_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in git_identity.items()
            if key != "git_identity_sha256"
        }
    )
    source_identity = tampered["source_identity"]
    source_identity["source_manifest_sha256"] = canonical_sha256(
        {
            "files": source_identity["files"],
            "file_roles": extension.SOURCE_FILE_ROLES,
            "loaded_repo_python_files": source_identity[
                "loaded_repo_python_files"
            ],
            "git_identity": git_identity,
        }
    )
    _resign(tampered)
    output = tmp_path / "tampered_source"
    with pytest.raises(ValueError, match="source/Git identity differs"):
        extension.write_output_package(tampered, output, repo_root=REPO_ROOT)


def test_stability_and_selection_tampering_are_rejected(
    payload: dict[str, object],
) -> None:
    tampered = json.loads(json.dumps(payload))
    tampered["panel_stability_summaries"][0]["mean_pairwise_jaccard"] = 0.0
    _resign(tampered)
    with pytest.raises(ValueError, match="stability"):
        extension.validate_result_payload(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["selection_decision"]["selected_candidate"]["n_components"] = 8
    _resign(tampered)
    with pytest.raises(ValueError, match="selection decision"):
        extension.validate_result_payload(tampered)


def test_confusion_and_leakage_tampering_are_rejected(
    payload: dict[str, object],
) -> None:
    tampered = json.loads(json.dumps(payload))
    tampered["records"][0]["confusion_matrix_counts"][0][0] += 1
    _resign(tampered)
    with pytest.raises(
        ValueError,
        match="metrics disagree|population supports differ|not confusion-derived",
    ):
        extension.validate_result_payload(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["ranking_records"][0]["population_labels_used"] = True
    _resign(tampered)
    with pytest.raises(ValueError, match="leakage"):
        extension.validate_result_payload(tampered)


def test_sample_prediction_split_and_panel_attacks_are_rejected(
    payload: dict[str, object],
) -> None:
    tampered = json.loads(json.dumps(payload))
    row = tampered["sample_prediction_evidence"][0]
    row["prediction"] = "EUR" if row["prediction"] != "EUR" else "AMR"
    row["correct"] = row["truth"] == row["prediction"]
    _resign(tampered)
    with pytest.raises(ValueError, match="arrays disagree"):
        extension.validate_result_payload(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["sample_prediction_evidence"][0][
        "selected_original_indices_sha256"
    ] = "0" * 64
    _resign(tampered)
    with pytest.raises(ValueError, match="split or panel evidence"):
        extension.validate_result_payload(tampered)

    tampered = json.loads(json.dumps(payload))
    tampered["sample_prediction_evidence"][0]["sample_index"] += 1
    _resign(tampered)
    with pytest.raises(ValueError, match="arrays disagree|not unique"):
        extension.validate_result_payload(tampered)


def test_coherent_seed_74_resign_attack_requires_trusted_input_replay(
    payload: dict[str, object],
) -> None:
    attacked = _coherently_resign_seed_74_attack(payload)
    persisted_clean = json.loads(extension._strict_json_bytes(payload))
    # All row evidence, hashes, confusion matrices, metrics, summaries,
    # selection and the analysis digest were changed coherently.  Structural
    # self-consistency alone therefore cannot detect this attack.
    assert extension.validate_result_payload(attacked)["status"] == "valid"
    with pytest.raises(ValueError, match=r"trusted-input full replay mismatch"):
        extension._validate_trusted_input_replay(attacked, persisted_clean)
    assert extension._validate_trusted_input_replay(persisted_clean, payload)[
        "trusted_input_replay_verified"
    ] is True


def test_proposed_lock_never_authorizes_fresh_seed_execution(
    payload: dict[str, object], monkeypatch: pytest.MonkeyPatch,
) -> None:
    eligible = json.loads(json.dumps(payload))
    eligible["confirmation_proposal_policy"]["eligible"] = True
    proposal = extension._build_lock_proposal(eligible)
    with pytest.raises(ValueError, match="confirmation proposal policy"):
        extension.validate_lock_proposal(
            proposal,
            eligible,
            expected_development_analysis_sha256=eligible["analysis_sha256"],
        )

    # The exact-proposal checks below isolate proposal validation after the
    # preceding assertion has established that result validation happens first.
    monkeypatch.setattr(extension, "validate_result_payload", lambda _: {"status": "valid"})
    validation = extension.validate_lock_proposal(
        proposal,
        eligible,
        expected_development_analysis_sha256=eligible["analysis_sha256"],
    )
    assert validation["status"] == "valid"
    assert proposal["execution_authorized"] is False
    assert proposal["proposed_confirmation_protocol"]["seeds"] == list(
        range(5001, 5021)
    )
    proposal["execution_authorized"] = True
    proposal.pop("proposal_payload_sha256")
    proposal["proposal_payload_sha256"] = canonical_sha256(proposal)
    with pytest.raises(ValueError, match="exactly equal"):
        extension.validate_lock_proposal(
            proposal,
            eligible,
            expected_development_analysis_sha256=eligible["analysis_sha256"],
        )

    proposal = extension._build_lock_proposal(eligible)
    proposal["prepared_input_identity"]["genotypes_sha256"] = "0" * 64
    proposal.pop("proposal_payload_sha256")
    proposal["proposal_payload_sha256"] = canonical_sha256(proposal)
    with pytest.raises(ValueError, match="exactly equal"):
        extension.validate_lock_proposal(
            proposal,
            eligible,
            expected_development_analysis_sha256=eligible["analysis_sha256"],
        )
    with pytest.raises(ValueError, match="expected development analysis"):
        extension.validate_lock_proposal(
            extension._build_lock_proposal(eligible),
            eligible,
            expected_development_analysis_sha256="0" * 64,
        )


def test_output_package_is_exact_and_tamper_evident(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    output = tmp_path / "smoke_package"
    validation = extension.write_output_package(payload, output)
    assert validation["status"] == "valid"
    assert validation["self_consistency_verified"] is True
    assert validation["trusted_input_replay_verified"] is False
    assert "external_anchors_verified" not in validation
    assert validation["package_files"] == 12
    assert not (output / extension.LOCK_PROPOSAL_FILENAME).exists()
    assert (output / extension.SAMPLE_EVIDENCE_FILENAME).is_file()
    manifest = json.loads(
        (output / extension.MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["verification_scope"] == "self_consistency_verified"
    assert manifest["formal_execution"] is None
    for filename in (
        extension.PC12_FILENAME,
        extension.PC34_FILENAME,
        extension.PC56_FILENAME,
        extension.PC78_FILENAME,
        extension.CONFUSION_FILENAME,
    ):
        from PIL import Image

        with Image.open(output / filename) as image:
            assert image.width >= 2400
            assert image.height >= 700
    metrics = output / extension.METRICS_FILENAME
    metrics.write_bytes(metrics.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash/size mismatch"):
        extension.validate_output_package(output)


def test_output_package_rejects_extra_file_and_external_anchor_mismatch(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    output = tmp_path / "smoke_package"
    extension.write_output_package(payload, output)
    (output / "extra.txt").write_text("not allowed", encoding="utf-8")
    with pytest.raises(ValueError, match="files must be exact"):
        extension.validate_output_package(output)
    (output / "extra.txt").unlink()
    with pytest.raises(ValueError, match="analysis-hash anchor"):
        extension.validate_output_package(
            output,
            expected_analysis_sha256="0" * 64,
            expected_package_anchor_sha256="0" * 64,
        )


def test_external_pre_and_post_run_receipts_are_exact_and_tamper_evident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    genotypes_path = tmp_path / "prepared.npz"
    source_manifest_path = tmp_path / "prepared.manifest.json"
    genotypes_path.write_bytes(b"input-placeholder")
    source_manifest_path.write_bytes(b"manifest-placeholder")
    output = tmp_path / "formal_package"
    output.mkdir()
    receipt_directory = tmp_path / "receipts"
    intent_path = receipt_directory / "intent.json"
    post_path = receipt_directory / "post.json"
    source_bundle_path = tmp_path / "source.bundle"
    source_bundle_path.write_bytes(b"test-only-bundle-placeholder")
    monkeypatch.setattr(
        extension,
        "_validate_source_bundle_file",
        lambda path, source: {
            "source_bundle_format": "git_bundle_exact_18_file_snapshot_v1",
            "source_bundle_sha256": extension._file_sha256(Path(path)),
            "source_bundle_bytes": Path(path).stat().st_size,
        },
    )
    _, resolved_intent, _, _, _ = extension._validate_formal_artifact_locations(
        output_directory=output,
        intent_receipt_path=intent_path,
        post_run_receipt_path=post_path,
        source_bundle_path=source_bundle_path,
        repo_root=REPO_ROOT,
    )
    with pytest.raises(ValueError, match="outside the Git worktree"):
        extension._validate_formal_artifact_locations(
            output_directory=REPO_ROOT / "forbidden_formal_output",
            intent_receipt_path=intent_path,
            post_run_receipt_path=post_path,
            source_bundle_path=source_bundle_path,
            repo_root=REPO_ROOT,
        )

    input_identity = {
        "external_source_manifest_anchor_verified": True,
        "ld_pruned_input_validated_from_external_manifest": True,
        "manifest_population_design": "five_superpopulation",
        "manifest_input_scope": "legacy_chr20_22",
        "orientation": "samples_by_snps",
        "external_source_manifest_sha256": extension.PREPARED_SOURCE_MANIFEST_SHA256,
        "expected_external_source_manifest_sha256": extension.PREPARED_SOURCE_MANIFEST_SHA256,
        "genotype_npz_sha256": "1" * 64,
        "genotypes_sha256": "2" * 64,
        "sample_ids_sha256": "3" * 64,
        "snp_ids_sha256": "4" * 64,
        "chromosomes_sha256": "5" * 64,
        "population_labels_sha256": "6" * 64,
        "samples": 1000,
        "snps": 3200,
    }
    source_identity = extension._source_identity(REPO_ROOT)
    intent = extension._build_pre_run_intent_receipt(
        output_uuid=str(uuid.uuid4()),
        source_bundle_path=source_bundle_path,
        input_identity=input_identity,
        source_identity=source_identity,
    )
    assert intent["source_file_count"] == 18
    extension._write_json_exclusive(resolved_intent, intent)
    persisted_intent = extension._strict_json_load(resolved_intent)
    validated = extension._validate_pre_run_intent_receipt(
        persisted_intent,
        source_bundle_path=source_bundle_path,
        input_identity=input_identity,
        source_identity=source_identity,
    )
    binding = extension._formal_execution_binding(resolved_intent, validated)
    manifest = {
        "formal_execution": binding,
        "analysis_sha256": "7" * 64,
        "package_anchor_sha256": "8" * 64,
        "manifest_payload_sha256": "9" * 64,
    }
    (output / extension.MANIFEST_FILENAME).write_bytes(
        extension._strict_json_bytes(manifest)
    )
    post = extension._build_post_run_receipt(
        output_directory=output,
        intent_receipt_path=resolved_intent,
        intent_receipt=validated,
        manifest=manifest,
    )
    extension._write_json_exclusive(post_path, post)
    persisted_post = extension._strict_json_load(post_path)
    assert extension._validate_post_run_receipt(
        persisted_post,
        output_directory=output,
        intent_receipt_path=resolved_intent,
        intent_receipt=validated,
        manifest=manifest,
    )["post_run_payload_sha256"] == post["post_run_payload_sha256"]

    attacked_intent = json.loads(json.dumps(intent))
    attacked_intent["config"]["seeds"] = [2025]
    attacked_intent.pop("intent_payload_sha256")
    attacked_intent["intent_payload_sha256"] = canonical_sha256(attacked_intent)
    with pytest.raises(ValueError, match="does not match trusted inputs/source"):
        extension._validate_pre_run_intent_receipt(
            attacked_intent,
            source_bundle_path=source_bundle_path,
            input_identity=input_identity,
            source_identity=source_identity,
        )

    attacked_post = json.loads(json.dumps(persisted_post))
    attacked_post["analysis_sha256"] = "0" * 64
    attacked_post.pop("post_run_payload_sha256")
    attacked_post["post_run_payload_sha256"] = canonical_sha256(attacked_post)
    with pytest.raises(ValueError, match="does not match the completed package"):
        extension._validate_post_run_receipt(
            attacked_post,
            output_directory=output,
            intent_receipt_path=resolved_intent,
            intent_receipt=validated,
            manifest=manifest,
        )


def test_real_source_bundle_must_expose_exact_18_file_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    for relative in extension.SOURCE_FILE_ROLES:
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    for arguments in (
        ("init", "-q"),
        ("config", "user.email", "snapshot@example.invalid"),
        ("config", "user.name", "Snapshot Test"),
        ("add", "."),
        ("commit", "-q", "-m", "exact source snapshot"),
    ):
        subprocess.run(
            ["git", "-C", str(snapshot), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    bundle = tmp_path / "source.bundle"
    subprocess.run(
        ["git", "-C", str(snapshot), "bundle", "create", str(bundle), "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    git_identity = extension._git_identity(snapshot)
    monkeypatch.setattr(extension, "_runner_repo_root", lambda: snapshot)
    receipt = extension._validate_source_bundle_file(
        bundle, {"git_identity": git_identity}
    )
    assert receipt["source_bundle_sha256"] == extension._file_sha256(bundle)
    assert receipt["source_bundle_bytes"] == bundle.stat().st_size

    extra = snapshot / "unexpected.txt"
    extra.write_text("tracked extra\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(snapshot), "add", "unexpected.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(snapshot), "commit", "-q", "-m", "extra"],
        check=True,
    )
    with pytest.raises(ValueError, match="exactly the 18 source files"):
        extension._validate_source_bundle_file(
            bundle, {"git_identity": extension._git_identity(snapshot)}
        )


def test_parser_has_only_frozen_run_smoke_and_validate_commands() -> None:
    parser = extension.build_parser()
    run = parser.parse_args(
        [
            "run",
            "--genotypes",
            "input.npz",
            "--source-manifest",
            "input.json",
            "--expected-source-manifest-sha256",
            extension.PREPARED_SOURCE_MANIFEST_SHA256,
            "--output-dir",
            "result",
            "--source-bundle",
            "source.bundle",
            "--intent-receipt",
            "intent.json",
            "--post-run-receipt",
            "post.json",
        ]
    )
    assert run.command == "run"
    assert not hasattr(run, "seeds")
    assert run.expected_source_manifest_sha256 == (
        extension.PREPARED_SOURCE_MANIFEST_SHA256
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--genotypes",
                "input.npz",
                "--source-manifest",
                "input.json",
                "--expected-source-manifest-sha256",
                "0" * 64,
                "--output-dir",
                "result",
                "--source-bundle",
                "source.bundle",
                "--intent-receipt",
                "intent.json",
                "--post-run-receipt",
                "post.json",
            ]
        )
    smoke = parser.parse_args(["smoke", "--output-dir", "result"])
    assert smoke.command == "smoke"
    with pytest.raises(SystemExit):
        parser.parse_args(["validate", "result"])

    validate = parser.parse_args(
        [
            "validate",
            "result",
            "--repo-root",
            ".",
            "--expected-analysis-sha256",
            "1" * 64,
            "--expected-package-anchor-sha256",
            "2" * 64,
            "--genotypes",
            "input.npz",
            "--source-manifest",
            "input.json",
            "--expected-source-manifest-sha256",
            extension.PREPARED_SOURCE_MANIFEST_SHA256,
            "--intent-receipt",
            "intent.json",
            "--post-run-receipt",
            "post.json",
            "--source-bundle",
            "source.bundle",
        ]
    )
    assert validate.command == "validate"


def test_formal_validation_requires_source_closure_and_dual_anchors(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    output = tmp_path / "smoke_package"
    extension.write_output_package(payload, output)
    result_path = output / extension.RESULT_FILENAME
    raw = json.loads(result_path.read_text(encoding="utf-8"))
    raw["config"]["analysis_mode"] = "development_protocol"
    result_path.write_bytes(extension._strict_json_bytes(raw))
    with pytest.raises(
        ValueError,
        match="formal validation requires repo root, dual anchors, original NPZ/manifest",
    ):
        extension.validate_output_package(output)

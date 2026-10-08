from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from experiments import genomics_scale_aware_vp_confirmation as confirmation


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def source_identity() -> dict[str, object]:
    return confirmation.collect_source_identity(REPO_ROOT)


@pytest.fixture(scope="module")
def smoke_package(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("scale_aware_vp") / "smoke"
    confirmation.run_smoke(output, REPO_ROOT)
    return output


def _package_parts(path: Path) -> tuple[dict, dict, dict[str, np.ndarray], dict]:
    analysis = confirmation._strict_json_load(path / confirmation.ANALYSIS_FILENAME)
    predictions = confirmation._strict_json_load(
        path / confirmation.PREDICTIONS_FILENAME
    )
    arrays = confirmation._load_npz_bytes(
        (path / confirmation.ARRAYS_FILENAME).read_bytes()
    )
    protocol = confirmation._strict_json_load(
        path / confirmation.SMOKE_PROTOCOL_FILENAME
    )
    return analysis, predictions, arrays, protocol


def _resign(payload: dict, field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = confirmation.canonical_sha256(unsigned)


def _prepared_fixture_identity(data: confirmation.grid.GenotypeInput) -> dict:
    return {
        **confirmation._input_identity(data),
        "external_source_manifest_sha256": "1" * 64,
        "expected_external_source_manifest_sha256": "1" * 64,
        "external_source_manifest_anchor_verified": True,
        "external_source_manifest_bytes": 1,
        "genotype_npz_sha256": "2" * 64,
        "genotype_npz_bytes": 1,
        "manifest_population_design": "five_superpopulation",
        "manifest_input_scope": "legacy_chr20_22",
        "manifest_ld_pruned_receipts_verified": True,
        "population_design": "five_superpopulation",
        "input_scope": "legacy_chr20_22",
        "ld_pruned": True,
        "population_label_is_race": False,
    }


def _fake_development_identity(prepared: dict, source: dict) -> dict:
    unsigned = {
        "schema_version": confirmation.DEVELOPMENT_BUNDLE_SCHEMA_VERSION,
        "status": "validated_real_post_hoc_development_bundle",
        "bundle_manifest_file_sha256": "3" * 64,
        "bundle_manifest_payload_sha256": "4" * 64,
        "prepared_input_identity": prepared,
        "source_identity_aggregate_sha256": source["aggregate_sha256"],
        "selection_summary_sha256": "5" * 64,
        "selected_rho": confirmation.RHO,
        "selected_budget": confirmation.TOTAL_BUDGET,
        "package_anchors_by_budget": {
            budget: {
                "analysis_sha256": character * 64,
                "package_anchor_sha256": character.upper().lower() * 64,
            }
            for budget, character in (("3000", "6"), ("4000", "7"))
        },
        "formal_seeds_executed": [],
    }
    return {**unsigned, "identity_sha256": confirmation.canonical_sha256(unsigned)}


def _rewrite_resigned_smoke_package(
    path: Path,
    analysis: dict,
    predictions: dict,
    arrays: dict[str, np.ndarray],
) -> None:
    arrays_bytes = confirmation._deterministic_npz_bytes(arrays)
    analysis["solver_arrays_sha256"] = confirmation._bytes_sha256(arrays_bytes)
    _resign(analysis, "analysis_sha256")
    payloads = {
        confirmation.ANALYSIS_FILENAME: confirmation._strict_json_bytes(analysis),
        confirmation.PREDICTIONS_FILENAME: confirmation._strict_json_bytes(
            predictions
        ),
        confirmation.ARRAYS_FILENAME: arrays_bytes,
    }
    for filename, payload in payloads.items():
        (path / filename).write_bytes(payload)
    manifest = confirmation._strict_json_load(path / confirmation.MANIFEST_FILENAME)
    for filename in payloads:
        payload = (path / filename).read_bytes()
        manifest["artifacts"][filename] = {
            "bytes": len(payload),
            "sha256": confirmation._bytes_sha256(payload),
        }
    manifest["analysis_sha256"] = analysis["analysis_sha256"]
    manifest["package_anchor_sha256"] = confirmation.canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    _resign(manifest, "manifest_payload_sha256")
    (path / confirmation.MANIFEST_FILENAME).write_bytes(
        confirmation._strict_json_bytes(manifest)
    )


def test_frozen_profile_requires_external_real_development_bundle() -> None:
    config = confirmation.formal_config()
    assert config.rho == 0.0025
    assert config.total_budget == 3000
    assert config.randomized_sketch_budget == 1500
    assert config.randomized_full_budget == 1500
    assert config.minimum_mean_row_support_fraction == 0.75
    assert config.maximum_mean_row_support_fraction == 0.90
    assert config.minimum_split_row_support_fraction == 0.70
    assert config.maximum_split_row_support_fraction == 0.95
    assert config.maximum_entry_support_fraction == 0.30
    assert confirmation.MEAN_BA_NONINFERIORITY_MARGIN == 0.005
    assert confirmation.WORST_BA_NONINFERIORITY_MARGIN == 0.03
    assert confirmation.DEVELOPMENT_SEEDS == tuple(range(2025, 2030))
    assert confirmation.development_config(3000).total_budget == 3000
    assert confirmation.development_config(4000).total_budget == 4000
    assert not hasattr(confirmation, "development_evidence")


def test_source_identity_is_actual_eager_import_closure(
    source_identity: dict[str, object],
) -> None:
    files = source_identity["files"]
    assert "experiments/__init__.py" in files
    assert "src/rspca/__init__.py" in files
    assert "src/rspca/robust.py" in files
    assert "src/rspca/federated.py" in files
    assert "src/rspca/fedrpca_audit.py" in files
    for relative, digest in confirmation.FROZEN_SHARED_CORE_HASHES.items():
        assert files[relative]["sha256"] == digest
    tampered = copy.deepcopy(source_identity)
    tampered["files"]["src/rspca/robust.py"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="aggregate"):
        confirmation.validate_source_identity(tampered)

    seven_only = copy.deepcopy(source_identity)
    seven_only["files"] = {
        relative: receipt
        for relative, receipt in seven_only["files"].items()
        if relative in confirmation.SOURCE_FILE_ROLES
    }
    seven_only["exact_discovered_paths"] = list(seven_only["files"])
    unsigned = dict(seven_only)
    unsigned.pop("aggregate_sha256")
    seven_only["aggregate_sha256"] = confirmation.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="exact discovered set"):
        confirmation.validate_source_identity(seven_only)


def test_vp_timing_reuses_frozen_pca_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(902)
    matrix = rng.normal(size=(18, 8))
    initial, singular = confirmation.grid.fit_top_pca(matrix, 3)
    step = 1.0 / float(singular[0] ** 2)
    config = replace(confirmation.smoke_config(), rank=3, panel_size=8, total_budget=2)
    observed: dict[str, object] = {}
    original = confirmation.fit_vp_spca

    def wrapped(*args, **kwargs):
        observed["initial_B"] = np.asarray(kwargs["initial_B"])
        observed["step_size"] = kwargs["step_size"]
        return original(*args, **kwargs)

    monkeypatch.setattr(confirmation, "fit_vp_spca", wrapped)
    confirmation._fit_sparse_once(
        "vp",
        matrix,
        alpha=0.2,
        pca_initial_loadings=initial,
        reference_step=step,
        seed=7,
        config=config,
    )
    np.testing.assert_array_equal(observed["initial_B"], initial)
    assert observed["step_size"] == step


def test_explicit_pca_reuse_matches_default_vp_quality() -> None:
    rng = np.random.default_rng(1902)
    matrix = rng.normal(size=(22, 10))
    initial, singular = confirmation.grid.fit_top_pca(matrix, 3)
    step = 1.0 / float(singular[0] ** 2)
    default = confirmation.fit_vp_spca(matrix, 3, alpha=0.15, max_iter=8, tol=1e-300)
    reused = confirmation.fit_vp_spca(
        matrix,
        3,
        alpha=0.15,
        max_iter=8,
        tol=1e-300,
        initial_B=initial,
        step_size=step,
    )
    assert reused.objectives[-1] == pytest.approx(default.objectives[-1], rel=2e-12)
    assert reused.stationarity[-1] == pytest.approx(default.stationarity[-1], rel=2e-10)


def test_freshness_missing_and_empty_roots_hard_fail(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real directory"):
        confirmation.audit_fresh_seed_execution([tmp_path / "missing"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no auditable"):
        confirmation.audit_fresh_seed_execution([empty])


def test_freshness_invalid_json_and_symlink_fail_closed(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / "execution.json").write_bytes(b"\xffnot-json")
    with pytest.raises((UnicodeDecodeError, ValueError)):
        confirmation.audit_fresh_seed_execution([invalid])

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / "execution.json").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        confirmation.audit_fresh_seed_execution([linked])


@pytest.mark.parametrize(
    "container", ["cases", "case_records", "seed_records", "results"]
)
@pytest.mark.parametrize("seed_value", [4001, "4001", "04001.0"])
def test_freshness_finds_execution_containers_and_numeric_strings(
    tmp_path: Path, container: str, seed_value: object
) -> None:
    root = tmp_path / f"audit_{container}_{str(seed_value).replace('.', '_')}"
    root.mkdir()
    (root / "execution.json").write_text(
        json.dumps({container: [{"outer_seed": seed_value}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fresh-seed execution evidence"):
        confirmation.audit_fresh_seed_execution([root])


def test_freshness_does_not_treat_declared_schedule_as_execution(
    tmp_path: Path,
) -> None:
    root = tmp_path / "audit"
    root.mkdir()
    (root / "protocol.json").write_text(
        json.dumps({"protocol": {"fresh_seed_schedule": [4001, 4002]}}),
        encoding="utf-8",
    )
    receipt = confirmation.audit_fresh_seed_execution([root])
    assert receipt["passed"] is True
    assert receipt["hits"] == []
    assert receipt["root_snapshots"][0]["file_count"] == 1


def test_freshness_finds_top_level_execution_receipt_and_npz(tmp_path: Path) -> None:
    json_root = tmp_path / "json_execution"
    json_root.mkdir()
    (json_root / "receipt.json").write_text(
        json.dumps({"confirmation_seeds_executed": ["4001"]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="fresh-seed execution evidence"):
        confirmation.audit_fresh_seed_execution([json_root])

    npz_root = tmp_path / "npz_execution"
    npz_root.mkdir()
    np.savez(npz_root / "receipt.npz", executed_seeds=np.asarray([4001]))
    with pytest.raises(ValueError, match="fresh-seed execution evidence"):
        confirmation.audit_fresh_seed_execution([npz_root])


def test_canonical_freshness_roots_cannot_be_substituted(tmp_path: Path) -> None:
    roots = [
        tmp_path / relative
        for relative in confirmation.CANONICAL_FRESHNESS_RELATIVE_ROOTS
    ]
    for root in roots:
        root.mkdir(parents=True)
    assert (
        confirmation._validate_canonical_freshness_root_strings(
            [str(root) for root in roots]
        )
        == tmp_path
    )
    with pytest.raises(ValueError, match="canonical"):
        confirmation._validate_canonical_freshness_root_strings(
            [str(root) for root in reversed(roots)]
        )

    inventory = confirmation.build_freshness_scope_inventory(
        tmp_path, tmp_path / "new-scale-aware-package"
    )
    confirmation.validate_freshness_scope_inventory(
        inventory,
        freshness_roots=[str(root) for root in roots],
        exclusive_output_directory=str(tmp_path / "new-scale-aware-package"),
    )
    assert inventory["excluded_broad_roots"][0]["relative_path"] == "results"


def test_formal_lock_round_trip_does_not_execute_fresh_seeds(
    tmp_path: Path, source_identity: dict[str, object]
) -> None:
    base = confirmation.make_smoke_data()
    genotypes = np.tile(base.genotypes, (1, 8))
    data = confirmation.grid.GenotypeInput(
        genotypes=genotypes,
        sample_ids=base.sample_ids,
        snp_ids=np.asarray(
            [f"fixture_snp_{index}" for index in range(genotypes.shape[1])]
        ),
        chromosomes=np.asarray(["22"] * genotypes.shape[1]),
        population_labels=base.population_labels,
    )
    preparation = _prepared_fixture_identity(data)
    development = _fake_development_identity(preparation, source_identity)
    registry = tmp_path / "global-registry"
    registry.mkdir()
    lock = confirmation.build_formal_lock(
        repo_root=REPO_ROOT,
        data=data,
        preparation_identity=preparation,
        freshness_roots=confirmation.canonical_freshness_roots(REPO_ROOT),
        output_directory=tmp_path / "reserved-formal-output",
        expected_shared_source_sha256=source_identity["aggregate_sha256"],
        development_package_identity=development,
        execution_registry_directory=registry.resolve(),
    )
    confirmation.validate_formal_lock(lock)
    assert lock["freshness_audit"]["passed"] is True
    assert lock["freshness_audit"]["seeds"] == list(confirmation.FORMAL_SEEDS)
    assert (
        lock["freshness_scope_inventory"]["reserved_scale_aware_output"][
            "observed_exists"
        ]
        is False
    )
    tampered = copy.deepcopy(lock)
    tampered["freshness_scope_inventory"]["excluded_broad_roots"][0]["reason"] = ""
    _resign(tampered["freshness_scope_inventory"], "inventory_sha256")
    _resign(tampered, "lock_payload_sha256")
    with pytest.raises(ValueError, match="inventory policy"):
        confirmation.validate_formal_lock(tampered)


def test_smoke_package_round_trips_and_is_non_evidence(smoke_package: Path) -> None:
    validation = confirmation.validate_output_package(
        smoke_package, repo_root=REPO_ROOT
    )
    assert validation["profile"] == "smoke"
    assert validation["record_count"] == 3
    assert validation["scientific_claim_allowed"] is False
    analysis, _, _, _ = _package_parts(smoke_package)
    assert analysis["formal_fresh_seeds_executed"] == []
    assert analysis["status"] == "smoke_complete_non_evidence"


def test_output_closure_rejects_extra_file(smoke_package: Path, tmp_path: Path) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(smoke_package, copied)
    (copied / "untracked.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or extra"):
        confirmation.validate_output_package(copied)


def test_artifact_byte_tampering_is_detected(
    smoke_package: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(smoke_package, copied)
    with (copied / confirmation.ARRAYS_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="artifact receipt"):
        confirmation.validate_output_package(copied)


def test_resigned_unknown_record_field_is_rejected(smoke_package: Path) -> None:
    analysis, predictions, arrays, protocol = _package_parts(smoke_package)
    analysis["records"][0]["unknown"] = "attack"
    _resign(analysis, "analysis_sha256")
    with pytest.raises(ValueError, match="record schema"):
        confirmation.validate_analysis_payload(analysis, predictions, arrays, protocol)


def test_resigned_prediction_metric_forgery_is_rejected(smoke_package: Path) -> None:
    analysis, predictions, arrays, protocol = _package_parts(smoke_package)
    analysis["records"][0]["native"]["metrics"]["balanced_accuracy"] = 1.0
    analysis["summaries"] = confirmation.summarize_records(
        analysis["records"], confirmation.smoke_config()
    )
    analysis["gates"] = confirmation.compute_gates(
        analysis["records"], confirmation.smoke_config()
    )
    _resign(analysis, "analysis_sha256")
    with pytest.raises(ValueError, match="metrics do not match predictions"):
        confirmation.validate_analysis_payload(analysis, predictions, arrays, protocol)


def test_resigned_seed_substitution_is_rejected(smoke_package: Path) -> None:
    analysis, predictions, arrays, protocol = _package_parts(smoke_package)
    analysis["records"][0]["seed"] = 4001
    _resign(analysis, "analysis_sha256")
    with pytest.raises(ValueError, match="record identity"):
        confirmation.validate_analysis_payload(analysis, predictions, arrays, protocol)


def test_resigned_non_top_panel_substitution_is_rejected(smoke_package: Path) -> None:
    analysis, predictions, arrays, protocol = _package_parts(smoke_package)
    seed = confirmation.SMOKE_SEEDS[0]
    prefix = f"seed_{seed}"
    retained = arrays[f"{prefix}_retained_original_indices"]
    selected = arrays[f"{prefix}_selected_retained_indices"].copy()
    replacement = next(index for index in range(retained.size) if index not in selected)
    selected[0] = replacement
    selected = np.sort(selected)
    selected_original = np.sort(retained[selected])
    selected_ids = np.asarray(
        [f"smoke_snp_{index:03d}" for index in selected_original], dtype=str
    )
    arrays[f"{prefix}_selected_retained_indices"] = selected
    arrays[f"{prefix}_selected_original_indices"] = selected_original
    arrays[f"{prefix}_selected_snp_ids"] = selected_ids
    split = analysis["split_receipts"][0]
    split["selected_original_indices"] = selected_original.tolist()
    split["selected_snp_ids"] = selected_ids.tolist()
    split["selected_retained_indices_sha256"] = confirmation.array_sha256(selected)
    split["selected_original_indices_sha256"] = confirmation.array_sha256(
        selected_original
    )
    split["selected_snp_ids_sha256"] = confirmation.array_sha256(selected_ids)
    for record in analysis["records"]:
        record["selected_retained_indices_sha256"] = split[
            "selected_retained_indices_sha256"
        ]
        record["selected_original_indices_sha256"] = split[
            "selected_original_indices_sha256"
        ]
        record["selected_snp_ids_sha256"] = split["selected_snp_ids_sha256"]
    _resign(analysis, "analysis_sha256")
    with pytest.raises(ValueError, match="split receipt arrays mismatch"):
        confirmation.validate_analysis_payload(analysis, predictions, arrays, protocol)


def test_external_package_anchors_are_paired(smoke_package: Path) -> None:
    valid = confirmation.validate_output_package(smoke_package)
    with pytest.raises(ValueError, match="requires both"):
        confirmation.validate_output_package(
            smoke_package, expected_analysis_sha256=valid["analysis_sha256"]
        )
    anchored = confirmation.validate_output_package(
        smoke_package,
        expected_analysis_sha256=valid["analysis_sha256"],
        expected_package_anchor_sha256=valid["package_anchor_sha256"],
    )
    assert anchored["external_anchors_verified"] is True


def test_formal_execution_requires_explicit_authorization(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="execute-fresh-confirmation"):
        confirmation.run_formal_confirmation(
            repo_root=REPO_ROOT,
            genotype_npz=tmp_path / "never-read.npz",
            source_manifest=tmp_path / "never-read.json",
            expected_source_manifest_sha256="0" * 64,
            lock_path=tmp_path / "never-read-lock.json",
            expected_lock_file_sha256="0" * 64,
            output_directory=tmp_path / "formal-output",
            execute_fresh_confirmation=False,
        )


def test_public_evaluate_rejects_formal_before_first_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = False

    def forbidden_split(*args, **kwargs):
        nonlocal touched
        touched = True
        raise AssertionError("formal split must not be reached")

    monkeypatch.setattr(confirmation, "stratified_split", forbidden_split)
    with pytest.raises(PermissionError, match="public evaluate refuses formal"):
        confirmation.evaluate(
            confirmation.make_smoke_data(), confirmation.formal_config()
        )
    assert touched is False


def test_global_semantic_execution_registry_is_atomic_and_output_independent(
    tmp_path: Path,
) -> None:
    registry = (tmp_path / "registry").resolve()
    registry.mkdir()
    prepared = _prepared_fixture_identity(confirmation.make_smoke_data())
    lock = {
        "prepared_input_identity": prepared,
        "protocol": confirmation._protocol(confirmation.formal_config()),
        "lock_payload_sha256": "8" * 64,
    }
    capability = confirmation._consume_global_formal_execution(
        registry_directory=registry,
        lock=lock,
        lock_file_sha256="9" * 64,
        output_directory=tmp_path / "first-output",
    )
    receipt = capability._receipt
    spent = registry / (receipt["execution_key_sha256"] + ".spent.json")
    assert spent.is_file()
    assert receipt["fit_started_when_written"] is False
    confirmation.validate_execution_spend_receipt(receipt)
    with pytest.raises(FileExistsError):
        confirmation._consume_global_formal_execution(
            registry_directory=registry,
            lock=lock,
            lock_file_sha256="a" * 64,
            output_directory=tmp_path / "different-output-and-lock",
        )


def test_formal_capability_constructor_is_private() -> None:
    with pytest.raises(PermissionError, match="cannot be constructed publicly"):
        confirmation._FormalExecutionCapability(
            object(),
            execution_key="0" * 64,
            input_identity={},
            protocol_sha256="1" * 64,
            receipt={},
            spent_path=Path("/tmp/never-used"),
        )


def test_pre_numpy_thread_and_affinity_contract_in_fresh_process() -> None:
    script = (
        "import json; "
        "from experiments import genomics_scale_aware_vp_confirmation as c; "
        "print(json.dumps(c.runtime_execution_contract(), sort_keys=True))"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src:."
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt["numpy_preimported_before_contract"] is False
    assert receipt["thread_environment"] == {
        name: "1" for name in sorted(confirmation.THREAD_ENVIRONMENT)
    }
    if receipt["pre_numpy_cpu_affinity"]["status"] == "pinned":
        assert len(receipt["current_cpu_affinity"]) == 1
    assert all(pool["num_threads"] == 1 for pool in receipt["threadpools"])


def test_speed_gate_uses_prefrozen_paired_bootstrap_lower_bound(
    smoke_package: Path,
) -> None:
    analysis, _, _, _ = _package_parts(smoke_package)
    vp = [record for record in analysis["records"] if record["method"] == "vp"]
    randomized = [
        record for record in analysis["records"] if record["method"] == "randomized"
    ]
    first = confirmation._paired_speed_bootstrap(vp, randomized)
    second = confirmation._paired_speed_bootstrap(vp, randomized)
    assert first == second
    assert first["seed"] == confirmation.SPEED_BOOTSTRAP_SEED
    assert first["draws"] == confirmation.SPEED_BOOTSTRAP_DRAWS
    protocol = confirmation._protocol(confirmation.formal_config())
    assert protocol["speed_bootstrap"]["gate"].startswith(
        "percentile_lower_bound"
    )
    assert analysis["gates"]["randomized"][
        "paired_solver_speed_bootstrap"
    ]["ci_lower"] == pytest.approx(first["ci_lower"])


def test_coherently_resigned_support_forgery_is_recomputed_from_loadings(
    smoke_package: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "support-forged"
    shutil.copytree(smoke_package, copied)
    analysis, predictions, arrays, _ = _package_parts(copied)
    vp = next(record for record in analysis["records"] if record["method"] == "vp")
    vp["support"] = {
        "threshold": confirmation.SUPPORT_THRESHOLD,
        "entry_count": 4,
        "entry_fraction": 4 / (confirmation.SMOKE_PANEL_SIZE * confirmation.RANK),
        "row_count": 4,
        "row_fraction": 4 / confirmation.SMOKE_PANEL_SIZE,
        "component_counts": [1, 1, 1, 1],
        "matrix_rank": 4,
    }
    vp["support_qualified"] = True
    analysis["summaries"] = confirmation.summarize_records(
        analysis["records"], confirmation.smoke_config()
    )
    analysis["gates"] = confirmation.compute_gates(
        analysis["records"], confirmation.smoke_config()
    )
    _rewrite_resigned_smoke_package(copied, analysis, predictions, arrays)
    with pytest.raises(ValueError, match="objective/KKT/support/centroid"):
        confirmation.validate_output_package(copied, repo_root=REPO_ROOT)


def test_coherently_resigned_zero_objective_and_kkt_are_recomputed(
    smoke_package: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "kkt-forged"
    shutil.copytree(smoke_package, copied)
    analysis, predictions, arrays, _ = _package_parts(copied)
    vp = next(record for record in analysis["records"] if record["method"] == "vp")
    vp["common_objective"] = 0.0
    vp["common_stationarity"] = {
        "raw_frobenius": 0.0,
        "kkt_rms": 0.0,
        "eta": 0.0,
    }
    count = vp["native_stationarity"]["last_window_count"]
    zeros = np.zeros(count, dtype=float)
    native = vp["native_stationarity"]
    native.update(
        {
            "last_raw_frobenius": zeros.tolist(),
            "last_kkt_rms": zeros.tolist(),
            "last_eta": zeros.tolist(),
            "last_raw_sha256": confirmation.array_sha256(zeros),
            "last_kkt_rms_sha256": confirmation.array_sha256(zeros),
            "last_eta_sha256": confirmation.array_sha256(zeros),
            "final_raw_frobenius": 0.0,
            "final_kkt_rms": 0.0,
            "final_eta": 0.0,
            "last_window_max_eta": 0.0,
            "last_window_all_qualified": True,
        }
    )
    vp["normalized_kkt_qualified"] = True
    arrays[f"seed_{confirmation.SMOKE_SEEDS[0]}_vp_native_last_raw"] = zeros
    analysis["summaries"] = confirmation.summarize_records(
        analysis["records"], confirmation.smoke_config()
    )
    analysis["gates"] = confirmation.compute_gates(
        analysis["records"], confirmation.smoke_config()
    )
    _rewrite_resigned_smoke_package(copied, analysis, predictions, arrays)
    with pytest.raises(ValueError, match="objective/KKT/support/centroid"):
        confirmation.validate_output_package(copied, repo_root=REPO_ROOT)


def test_coherently_resigned_alternative_pca_screen_is_recomputed(
    smoke_package: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "screen-forged"
    shutil.copytree(smoke_package, copied)
    analysis, predictions, arrays, _ = _package_parts(copied)
    seed = confirmation.SMOKE_SEEDS[0]
    prefix = f"seed_{seed}"
    retained = arrays[f"{prefix}_retained_original_indices"]
    rng = np.random.default_rng(991_771)
    alternative, _ = np.linalg.qr(
        rng.normal(size=(retained.size, confirmation.RANK)), mode="reduced"
    )
    ranking, importance = confirmation.grid.rank_by_pca_loadings(
        alternative, confirmation.RANK
    )
    selected_retained = np.sort(ranking[: confirmation.SMOKE_PANEL_SIZE])
    selected_original = np.sort(retained[selected_retained])
    assert not np.array_equal(
        selected_original, arrays[f"{prefix}_selected_original_indices"]
    )
    arrays[f"{prefix}_full_width_pca_loadings"] = alternative
    arrays[f"{prefix}_loading_importance"] = importance
    arrays[f"{prefix}_selected_retained_indices"] = selected_retained
    arrays[f"{prefix}_selected_original_indices"] = selected_original
    split = analysis["split_receipts"][0]
    split["selected_original_indices"] = selected_original.tolist()
    split["ordered_original_ranking_sha256"] = confirmation.array_sha256(
        retained[ranking]
    )
    split["loading_importance_sha256"] = confirmation.array_sha256(importance)
    split["full_width_pca_loadings_sha256"] = confirmation.array_sha256(alternative)
    split["selected_retained_indices_sha256"] = confirmation.array_sha256(
        selected_retained
    )
    split["selected_original_indices_sha256"] = confirmation.array_sha256(
        selected_original
    )
    for record in analysis["records"]:
        record["selected_retained_indices_sha256"] = split[
            "selected_retained_indices_sha256"
        ]
        record["selected_original_indices_sha256"] = split[
            "selected_original_indices_sha256"
        ]
    _rewrite_resigned_smoke_package(copied, analysis, predictions, arrays)
    with pytest.raises(ValueError, match="split/preprocess/PCA/top-panel"):
        confirmation.validate_output_package(copied, repo_root=REPO_ROOT)


def test_formal_validation_cannot_omit_original_inputs_lock_or_manifest_anchor(
    smoke_package: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "formal-shaped"
    shutil.copytree(smoke_package, copied)
    manifest_path = copied / confirmation.MANIFEST_FILENAME
    manifest = confirmation._strict_json_load(manifest_path)
    manifest["profile"] = "formal"
    _resign(manifest, "manifest_payload_sha256")
    manifest_path.write_bytes(confirmation._strict_json_bytes(manifest))
    with pytest.raises(ValueError, match="requires original inputs/lock/repo/output"):
        confirmation.validate_output_package(copied)


def test_support_gate_checks_per_split_and_mean_bounds(smoke_package: Path) -> None:
    analysis, _, _, _ = _package_parts(smoke_package)
    template = {
        method: next(row for row in analysis["records"] if row["method"] == method)
        for method in confirmation.METHODS
    }
    records = []
    for seed in confirmation.FORMAL_SEEDS:
        for method in confirmation.METHODS:
            row = copy.deepcopy(template[method])
            row["seed"] = seed
            row["support_qualified"] = None if method == "pca" else True
            row["normalized_kkt_qualified"] = None if method == "pca" else True
            row["support"]["row_fraction"] = 0.74 if method == "vp" else 0.80
            row["native"]["metrics"]["balanced_accuracy"] = 0.90
            row["common_stationarity"]["eta"] = 0.0005
            records.append(row)
    gates = confirmation.compute_gates(records, confirmation.formal_config())
    assert gates["vp"]["support_per_split_bounds_all_splits"] is True
    assert gates["vp"]["row_support_mean_in_locked_range"] is False
    assert gates["vp"]["support_qualified"] is False

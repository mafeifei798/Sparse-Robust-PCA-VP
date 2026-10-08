from __future__ import annotations

import csv
import ast
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

import experiments.video_figure5_frozen_noise_stress as runner
from experiments.video_figure5_frozen_noise_stress import (
    CASES_NAME,
    DATA_MANIFEST_NAME,
    EXECUTION_RECEIPT_NAME,
    FACTORS_NAME,
    FIGURE_NAME,
    FORMAL_SNR_SEED_BANK,
    FRAMES_NAME,
    LOCK_NAME,
    OUTPUT_MANIFEST_NAME,
    REPORT_NAME,
    SEED_REGISTRY_NAME,
    SOURCE_MANIFEST_NAME,
    FROZEN_SHARED_CORE_SHA256,
    formal_config,
    generate_noisy_figure5_case,
    run_frozen_noise_stress,
    save_frozen_noise_stress,
    smoke_config,
    validate_config,
    validate_output_directory,
)
from experiments.video_spca_paper_reproduction import generate_figure5_dataset
from rspca.datasets import canonical_sha256
from tools.validate_video_figure5_frozen_noise_stress import (
    _validate_expected_formal_file_hashes,
    validate_frozen_noise_stress_output,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_source_manifest_ignores_unrelated_loaded_modules() -> None:
    import importlib

    root = Path(__file__).resolve().parents[1]
    before = runner._source_manifest(root)
    importlib.import_module("rspca.certificate_balanced_cleaned_encoder")
    after = runner._source_manifest(root)
    assert after["files"] == before["files"]
    assert set(after["files"]) == set(runner.EXPECTED_SOURCE_ROLE_SCHEMA)


def test_source_manifest_rejects_a_new_actual_dependency(monkeypatch) -> None:
    root = Path(__file__).resolve().parents[1]
    parse = ast.parse

    def changed_imports(source, filename="<unknown>", *args, **kwargs):
        if filename == "experiments/video_figure5_frozen_noise_stress.py":
            source += "\nimport rspca.certificate_balanced_cleaned_encoder\n"
        return parse(source, filename, *args, **kwargs)

    monkeypatch.setattr(runner.ast, "parse", changed_imports)
    with pytest.raises(RuntimeError, match="experiment dependency is outside fixed source schema"):
        runner._source_manifest(root)


def _write_json(payload: dict, path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _resign_json(path: Path, digest_field: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop(digest_field, None)
    payload[digest_field] = canonical_sha256(payload)
    _write_json(payload, path)
    return payload


def _refresh_output_manifest(output: Path, *names: str) -> None:
    path = output / OUTPUT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for name in names:
        payload = output / name
        manifest["files"][name] = {
            "sha256": _file_sha256(payload),
            "size_bytes": payload.stat().st_size,
        }
    manifest.pop("output_manifest_body_sha256", None)
    manifest["output_manifest_body_sha256"] = canonical_sha256(manifest)
    _write_json(manifest, path)


def _resign_linked_json_chain(output: Path) -> None:
    source = json.loads((output / SOURCE_MANIFEST_NAME).read_text(encoding="utf-8"))
    data = json.loads((output / DATA_MANIFEST_NAME).read_text(encoding="utf-8"))
    lock_path = output / LOCK_NAME
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["source_manifest_body_sha256"] = source["source_manifest_body_sha256"]
    lock["data_manifest_body_sha256"] = data["data_manifest_body_sha256"]
    lock.pop("protocol_lock_body_sha256")
    lock["protocol_lock_body_sha256"] = canonical_sha256(lock)
    _write_json(lock, lock_path)
    lock_file_sha = _file_sha256(lock_path)

    receipt_path = output / EXECUTION_RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["protocol_lock_file_sha256"] = lock_file_sha
    receipt["protocol_lock_body_sha256"] = lock["protocol_lock_body_sha256"]
    if "source_manifest_body_sha256" in receipt:
        receipt["source_manifest_body_sha256"] = source[
            "source_manifest_body_sha256"
        ]
    if "data_manifest_body_sha256" in receipt:
        receipt["data_manifest_body_sha256"] = data["data_manifest_body_sha256"]
    receipt.pop("execution_receipt_body_sha256")
    receipt["execution_receipt_body_sha256"] = canonical_sha256(receipt)
    _write_json(receipt, receipt_path)

    report_path = output / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["protocol_lock_file_sha256"] = lock_file_sha
    report["protocol_lock_body_sha256"] = lock["protocol_lock_body_sha256"]
    report["source_manifest_body_sha256"] = source["source_manifest_body_sha256"]
    report["data_manifest_body_sha256"] = data["data_manifest_body_sha256"]
    report["protocol"] = lock["protocol"]
    report["execution_authorization"] = lock["execution_authorization"]
    report["execution_receipt_body_sha256"] = receipt[
        "execution_receipt_body_sha256"
    ]
    report.pop("report_body_sha256")
    report["report_body_sha256"] = canonical_sha256(report)
    _write_json(report, report_path)
    _refresh_output_manifest(
        output,
        SOURCE_MANIFEST_NAME,
        DATA_MANIFEST_NAME,
        LOCK_NAME,
        EXECUTION_RECEIPT_NAME,
        REPORT_NAME,
    )


def test_frozen_formal_protocol_has_exact_parameters_and_new_seed_bank() -> None:
    config = formal_config()

    assert config.alpha == 9.0
    assert config.n_components == 3
    assert config.max_iter == 500
    assert config.tol == 1e-6
    assert (config.randomized_oversample, config.randomized_power_iters) == (100, 4)
    assert config.randomized_seed == 42
    assert [snr for snr, _ in config.snr_seed_bank] == [20.0, 10.0, 0.0]
    assert all(len(seeds) == 20 for _, seeds in config.snr_seed_bank)
    assert len({seed for _, seeds in config.snr_seed_bank for seed in seeds}) == 60
    assert config.snr_seed_bank == FORMAL_SNR_SEED_BANK
    assert FROZEN_SHARED_CORE_SHA256["src/rspca/vp.py"].startswith("88a84aa9")
    assert FROZEN_SHARED_CORE_SHA256["src/rspca/randomized.py"].startswith(
        "10183333"
    )
    with pytest.raises(ValueError, match="configuration was modified"):
        validate_config(replace(config, alpha=8.0))


def test_dense_noise_has_exact_snr_full_rank_witness_and_is_deterministic() -> None:
    config = smoke_config()
    clean = generate_figure5_dataset(runner._paper_config(config)).clean
    first = generate_noisy_figure5_case(clean, snr_db=10.0, noise_seed=83902)
    second = generate_noisy_figure5_case(clean, snr_db=10.0, noise_seed=83902)

    assert first.achieved_snr_db == pytest.approx(10.0, abs=1e-12)
    assert first.leading_square_rank == config.n_timesteps
    assert first.leading_square_min_singular_value > 0.0
    assert np.mean(first.noise != 0.0) == 1.0
    np.testing.assert_array_equal(first.noise, second.noise)
    np.testing.assert_array_equal(first.observed, second.observed)


def test_seed_history_audit_ignores_hash_substrings_but_rejects_seed_fields(
    tmp_path,
) -> None:
    results = tmp_path / "results"
    results.mkdir()
    incidental = results / "incidental.json"
    incidental.write_text(
        json.dumps({"content_sha256": "abc83001def"}), encoding="utf-8"
    )
    audit = runner._formal_seed_history_audit(tmp_path)
    assert audit["no_prior_outcome_artifact_matches"] is True

    (results / "prior_run.json").write_text(
        json.dumps({"config": {"noise_seed": 83001}}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="prior outcome artifacts"):
        runner._formal_seed_history_audit(tmp_path)


@pytest.fixture(scope="module")
def saved_smoke(tmp_path_factory):
    output = tmp_path_factory.mktemp("frozen-figure5-noise") / "smoke"
    completed = run_frozen_noise_stress("smoke", output_dir=output)
    paths = save_frozen_noise_stress(completed, output)
    validate_output_directory(output)
    validate_frozen_noise_stress_output(output)
    return completed, output, paths


def test_smoke_is_exact_closed_full_frame_source_bound_and_not_evidence(
    saved_smoke,
) -> None:
    completed, output, paths = saved_smoke
    assert set(paths) == {
        "report",
        "cases",
        "frames",
        "figure",
        "lock",
        "source_manifest",
        "data_manifest",
        "execution_receipt",
        "seed_registry",
        "factors",
        "output_manifest",
    }
    assert paths["figure"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["strict_qualification"]["claim_status"] == "smoke_not_evidence"
    assert report["formal_confirmation"] is False
    assert len(report["case_records"]) == 9
    assert len(completed.frame_records) == 3 * 3 * smoke_config().n_timesteps
    assert report["protocol"]["truth_used_during_fit"] is False
    assert report["protocol"]["post_hoc_amplitude_fit"] is False
    source = json.loads(paths["source_manifest"].read_text(encoding="utf-8"))
    assert source["package_eager_imports_bound"] is True
    assert source["files"]["src/rspca/federated.py"]["role"] == (
        "package_initialization_only"
    )
    assert source["federated_execution_path_used"] is False
    assert source["frozen_shared_core_sha256"] == FROZEN_SHARED_CORE_SHA256
    receipt = json.loads(paths["execution_receipt"].read_text(encoding="utf-8"))
    assert receipt["receipt_kind"] == "smoke_not_formal_lock_consumption"
    assert receipt["created_with_exclusive_open_before_any_fit"] is False
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_frozen_noise_stress(completed, output)


def test_runtime_gate_uses_balanced_order_and_conservative_complete_cost(
    saved_smoke,
) -> None:
    completed, _, _ = saved_smoke
    by_key = {
        (float(row["snr_db"]), int(row["noise_seed"]), row["method"]): row
        for row in completed.case_records
    }
    for snr_db, seeds in completed.config.snr_seed_bank:
        for seed in seeds:
            expected_order = (
                ["PCA", "VP-SPCA", "RND-VP-SPCA"]
                if seed % 2
                else ["PCA", "RND-VP-SPCA", "VP-SPCA"]
            )
            pca = by_key[(snr_db, seed, "PCA")]
            vp = by_key[(snr_db, seed, "VP-SPCA")]
            rnd = by_key[(snr_db, seed, "RND-VP-SPCA")]
            assert vp["fit_execution_order"] == expected_order
            assert rnd["fit_execution_order"] == expected_order
            assert vp["complete_method_runtime_seconds"] == pytest.approx(
                vp["raw_solver_fit_call_runtime_seconds"]
                + pca["raw_solver_fit_call_runtime_seconds"]
            )
            assert rnd["complete_method_runtime_seconds"] == pytest.approx(
                rnd["raw_solver_fit_call_runtime_seconds"]
            )


def test_formal_cannot_reach_fit_without_a_valid_external_lock(monkeypatch, tmp_path) -> None:
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("fit must not be reached")

    monkeypatch.setattr(runner, "_fit_observed_methods", forbidden)
    with pytest.raises(ValueError, match="requires output_dir"):
        run_frozen_noise_stress("formal")
    with pytest.raises(ValueError, match="--lock must name"):
        run_frozen_noise_stress(
            "formal",
            output_dir=tmp_path / "formal-output",
            lock_path=tmp_path / LOCK_NAME,
            expected_lock_sha256="0" * 64,
        )
    assert called is False


def test_small_formal_anchor_is_consumed_before_fit_and_cannot_be_reused(
    monkeypatch,
    tmp_path,
) -> None:
    small_bank = runner.SMOKE_SNR_SEED_BANK
    small_formal = replace(
        smoke_config(),
        profile="formal_confirmation",
        snr_seed_bank=small_bank,
    )
    monkeypatch.setattr(runner, "FORMAL_SNR_SEED_BANK", small_bank)
    monkeypatch.setattr(runner, "formal_config", lambda: small_formal)
    formal_root = tmp_path / "canonical-formal-root"
    anchor = formal_root / "anchor"
    output = formal_root / "formal_output"
    monkeypatch.setattr(
        runner,
        "_canonical_formal_paths",
        lambda repo_root: {
            "root": formal_root,
            "anchor": anchor,
            "output": output,
            "seed_registry": formal_root / SEED_REGISTRY_NAME,
        },
    )
    timing_body = {"formal_timing_qualified": True}
    timing = {
        **timing_body,
        "timing_environment_body_sha256": canonical_sha256(timing_body),
    }
    monkeypatch.setattr(
        runner,
        "_timing_environment_snapshot",
        lambda repo_root, *, enforce_formal: timing,
    )
    audit_body = {
        "audit_version": 1,
        "searched_roots": ["results", "assets/evaluation", "artifacts"],
        "root_states": [],
        "text_suffixes": [],
        "checked_files": 0,
        "inventory": [],
        "inventory_sha256": canonical_sha256({"files": []}),
        "excluded_directories_resolved": [str(formal_root.resolve())],
        "locked_formal_seeds": sorted(
            seed for _, seeds in small_bank for seed in seeds
        ),
        "matches": [],
        "no_prior_outcome_artifact_matches": True,
    }
    audit = {
        **audit_body,
        "seed_history_audit_body_sha256": canonical_sha256(audit_body),
    }
    monkeypatch.setattr(
        runner,
        "_formal_seed_history_audit",
        lambda repo_root, *, excluded_directories=(): audit,
    )
    frozen = runner.freeze_formal_anchor()
    lock_sha = str(frozen["protocol_lock_file_sha256"])
    assert not output.exists()
    assert not (anchor / EXECUTION_RECEIPT_NAME).exists()

    calls = 0

    def stop_at_first_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        assert (anchor / EXECUTION_RECEIPT_NAME).is_file()
        raise RuntimeError("intentional stop after receipt")

    monkeypatch.setattr(runner, "_fit_observed_methods", stop_at_first_fit)
    with pytest.raises(RuntimeError, match="intentional stop"):
        run_frozen_noise_stress(
            "formal",
            output_dir=output,
            lock_path=anchor / LOCK_NAME,
            expected_lock_sha256=lock_sha,
        )
    assert calls == 1
    receipt = json.loads(
        (anchor / EXECUTION_RECEIPT_NAME).read_text(encoding="utf-8")
    )
    assert receipt["exclusive_output_dir_resolved"] == str(output.resolve())
    with pytest.raises(ValueError, match="exact closure"):
        run_frozen_noise_stress(
            "formal",
            output_dir=output,
            lock_path=anchor / LOCK_NAME,
            expected_lock_sha256=lock_sha,
        )
    assert calls == 1


@pytest.mark.parametrize(
    "attack, expected_message",
    [
        ("gate", "paired gate vector mismatch"),
        ("case_csv", "case CSV clean_relative_reconstruction_error"),
        ("frame_csv", "frame CSV"),
        ("stationarity", "stationarity_met disagrees"),
    ],
)
def test_external_validator_rejects_resigned_semantic_attacks(
    saved_smoke,
    tmp_path,
    attack: str,
    expected_message: str,
) -> None:
    _, source, _ = saved_smoke
    output = tmp_path / attack
    shutil.copytree(source, output)
    if attack in {"gate", "stationarity"}:
        report_path = output / REPORT_NAME
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if attack == "gate":
            metric = report["stratified_randomized_vs_vp_gates"]["strata"]["20"][
                "metrics"
            ]["clean_relative_error_ratio"]
            metric["paired_values"][0] += 0.25
        else:
            vp = next(row for row in report["case_records"] if row["method"] == "VP-SPCA")
            vp["stationarity_met"] = not vp["stationarity_met"]
        report.pop("report_body_sha256")
        report["report_body_sha256"] = canonical_sha256(report)
        _write_json(report, report_path)
        _refresh_output_manifest(output, REPORT_NAME)
    elif attack == "case_csv":
        path = output / CASES_NAME
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["clean_relative_reconstruction_error"] = "123.0"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _refresh_output_manifest(output, CASES_NAME)
    else:
        path = output / FRAMES_NAME
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        rows[0]["normalized_rmse"] = str(float(rows[0]["normalized_rmse"]) + 5.0)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        _refresh_output_manifest(output, FRAMES_NAME)

    with pytest.raises(ValueError, match=expected_message):
        validate_frozen_noise_stress_output(output)


def test_external_validator_rejects_nonregular_extra_entry(saved_smoke, tmp_path) -> None:
    _, source, _ = saved_smoke
    output = tmp_path / "extra-entry"
    shutil.copytree(source, output)
    (output / "untracked").mkdir()

    with pytest.raises(ValueError, match="non-regular"):
        validate_frozen_noise_stress_output(output)


def test_freshness_inventory_covers_all_roots_and_parse_errors_fail_closed(
    tmp_path,
) -> None:
    for relative in ("results", "assets/evaluation", "artifacts"):
        (tmp_path / relative).mkdir(parents=True)
    (tmp_path / "results" / "valid.json").write_text(
        json.dumps({"noise_seed": 7}), encoding="utf-8"
    )
    (tmp_path / "artifacts" / "opaque.npz").write_bytes(b"opaque")
    audit = runner._formal_seed_history_audit(tmp_path)
    assert audit["searched_roots"] == ["results", "assets/evaluation", "artifacts"]
    assert {row["path"] for row in audit["inventory"]} == {
        "results/valid.json",
        "artifacts/opaque.npz",
    }
    assert audit["inventory_sha256"] == canonical_sha256(
        {"files": audit["inventory"]}
    )

    (tmp_path / "assets/evaluation" / "malformed.json").write_text(
        "{not-json", encoding="utf-8"
    )
    with pytest.raises(json.JSONDecodeError):
        runner._formal_seed_history_audit(tmp_path)


def test_global_seed_registry_is_create_once_inside_canonical_root(
    monkeypatch, tmp_path
) -> None:
    formal_root = tmp_path / "results" / "canonical-formal"
    monkeypatch.setattr(
        runner,
        "_canonical_formal_paths",
        lambda repo_root: {
            "root": formal_root,
            "anchor": formal_root / "anchor",
            "output": formal_root / "formal_output",
            "seed_registry": formal_root / SEED_REGISTRY_NAME,
        },
    )
    audit = {
        "inventory_sha256": "1" * 64,
        "seed_history_audit_body_sha256": "2" * 64,
    }
    timing = {"timing_environment_body_sha256": "3" * 64}
    payload, path = runner._claim_formal_seed_bank_registry(
        tmp_path, audit, timing
    )
    assert path.parent == formal_root
    assert payload["created_with_exclusive_open_before_anchor_and_any_fit"] is True
    with pytest.raises(FileExistsError, match="seed bank is consumed"):
        runner._claim_formal_seed_bank_registry(tmp_path, audit, timing)


@pytest.mark.parametrize(
    "attack, expected_message",
    [
        ("source_role", "source manifest exact path/role schema"),
        ("data_hash", "independently regenerated data array hash"),
    ],
)
def test_external_validator_rejects_fully_resigned_source_and_data_attacks(
    saved_smoke, tmp_path, attack: str, expected_message: str
) -> None:
    _, source_output, _ = saved_smoke
    output = tmp_path / attack
    shutil.copytree(source_output, output)
    if attack == "source_role":
        path = output / SOURCE_MANIFEST_NAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["files"]["src/rspca/vp.py"]["role"] = "package_initialization_only"
        payload.pop("source_manifest_body_sha256")
        payload["source_manifest_body_sha256"] = canonical_sha256(payload)
    else:
        path = output / DATA_MANIFEST_NAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["cases"][0]["array_sha256"]["observed"] = "0" * 64
        payload.pop("data_manifest_body_sha256")
        payload["data_manifest_body_sha256"] = canonical_sha256(payload)
    _write_json(payload, path)
    _resign_linked_json_chain(output)

    with pytest.raises(ValueError, match=expected_message):
        validate_frozen_noise_stress_output(output)


def test_external_validator_rejects_resigned_row_tolerance_attack(
    saved_smoke, tmp_path
) -> None:
    _, source_output, _ = saved_smoke
    output = tmp_path / "row-tolerance"
    shutil.copytree(source_output, output)
    report_path = output / REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["case_records"][0]["tolerance"] = 1e-5
    report.pop("report_body_sha256")
    report["report_body_sha256"] = canonical_sha256(report)
    _write_json(report, report_path)
    _refresh_output_manifest(output, REPORT_NAME)

    with pytest.raises(ValueError, match="tolerance must equal locked"):
        validate_frozen_noise_stress_output(output)


def test_external_validator_rejects_resigned_factor_npz_attack(
    saved_smoke, tmp_path
) -> None:
    _, source_output, _ = saved_smoke
    output = tmp_path / "factor-attack"
    shutil.copytree(source_output, output)
    report = json.loads((output / REPORT_NAME).read_text(encoding="utf-8"))
    key = report["case_records"][0]["factor_keys"]["B"]
    with np.load(output / FACTORS_NAME, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays[key][0, 0] += 0.25
    np.savez_compressed(output / FACTORS_NAME, **arrays)
    _refresh_output_manifest(output, FACTORS_NAME)

    with pytest.raises(ValueError, match="factor recomputation|factor A"):
        validate_frozen_noise_stress_output(output)


def test_external_validator_rejects_resigned_png_pixel_attack(
    saved_smoke, tmp_path
) -> None:
    _, source_output, _ = saved_smoke
    output = tmp_path / "png-attack"
    shutil.copytree(source_output, output)
    path = output / FIGURE_NAME
    with Image.open(path) as image:
        pixels = np.asarray(image).copy()
    pixels[0, 0, :3] = 0
    Image.fromarray(pixels, mode="RGBA").save(path)
    _refresh_output_manifest(output, FIGURE_NAME)

    with pytest.raises(ValueError, match="pixels differ"):
        validate_frozen_noise_stress_output(output)


def test_formal_external_validator_requires_both_out_of_band_hashes() -> None:
    with pytest.raises(ValueError, match="expected-lock-sha256"):
        _validate_expected_formal_file_hashes(
            "formal_confirmation",
            actual_lock_sha256="1" * 64,
            actual_output_manifest_sha256="2" * 64,
            expected_lock_sha256=None,
            expected_output_manifest_sha256=None,
        )
    with pytest.raises(ValueError, match="output manifest differs"):
        _validate_expected_formal_file_hashes(
            "formal_confirmation",
            actual_lock_sha256="1" * 64,
            actual_output_manifest_sha256="2" * 64,
            expected_lock_sha256="1" * 64,
            expected_output_manifest_sha256="3" * 64,
        )
    with pytest.raises(ValueError, match="output manifest differs"):
        _validate_expected_formal_file_hashes(
            "smoke_not_evidence",
            actual_lock_sha256="1" * 64,
            actual_output_manifest_sha256="2" * 64,
            expected_lock_sha256="1" * 64,
            expected_output_manifest_sha256="3" * 64,
        )


def test_local_environment_cannot_qualify_as_autodl_formal_timing() -> None:
    snapshot = runner._timing_environment_snapshot(
        Path(__file__).resolve().parents[1], enforce_formal=False
    )
    assert "threadpool_info" in snapshot
    assert "cpu_affinity" in snapshot
    assert "numpy_backend_configuration" in snapshot
    with pytest.raises(RuntimeError, match="formal timing environment is not qualified"):
        runner._timing_environment_snapshot(
            Path(__file__).resolve().parents[1], enforce_formal=True
        )

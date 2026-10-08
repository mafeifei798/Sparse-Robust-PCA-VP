from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from experiments.genomics_aim_protocol import (
    OUTPUT_FILE_NAMES,
    _ProgressReporter,
    _write_csv,
    _write_json,
    audit_output_package_claim_gates,
    load_protocol_input,
    validate_output_package,
    write_protocol_outputs,
)
from rspca.genomics_aim import AIMProtocolConfig, run_aim_protocol


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(923)
    labels = np.repeat(np.asarray(["AFR", "EAS", "EUR"]), 12)
    frequencies = np.full((3, 12), 0.35)
    frequencies[0, :3] = 0.08
    frequencies[1, :3] = 0.88
    frequencies[2, :3] = 0.45
    frequencies[0, 3:6] = 0.82
    frequencies[1, 3:6] = 0.12
    frequencies[2, 3:6] = 0.48
    matrix = np.vstack(
        [rng.binomial(2, frequencies[index], size=(12, 12)) for index in range(3)]
    ).astype(float)
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(matrix.shape[0])])
    snp_ids = np.asarray([f"snp_{index:03d}" for index in range(matrix.shape[1])])
    genotype_path = tmp_path / "genotypes.npz"
    np.savez(
        genotype_path,
        genotypes=matrix,
        sample_ids=sample_ids,
        snp_ids=snp_ids,
        chromosomes=np.asarray(["1"] * 6 + ["2"] * 6),
    )
    metadata_path = tmp_path / "populations.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "population"])
        writer.writeheader()
        writer.writerows(
            {
                "sample_id": sample_id,
                "population": population,
            }
            for sample_id, population in zip(sample_ids, labels, strict=True)
        )
    return genotype_path, metadata_path


def _cli_command(
    genotype_path: Path,
    metadata_path: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        "experiments/genomics_aim_protocol.py",
        "--genotypes",
        str(genotype_path),
        "--population-metadata",
        str(metadata_path),
        "--source-manifest-sha256",
        "a" * 64,
        "--input-scope",
        "synthetic_multi_autosome_smoke",
        "--population-design",
        "three_continent",
        "--ld-pruned",
        "--methods",
        "pca",
        "--panel-size",
        "4",
        "--seeds",
        "81,82",
        "--bootstrap-resamples",
        "100",
        "--bootstrap-seed",
        "717",
        "--output-dir",
        str(output),
    ]


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "src"
    environment["MPLBACKEND"] = "Agg"
    return environment


def _small_result(tmp_path: Path):
    genotype_path, metadata_path = _write_inputs(tmp_path)
    data = load_protocol_input(
        genotype_path,
        metadata_path,
        source_manifest_sha256="a" * 64,
        input_scope="synthetic_multi_autosome_smoke",
        ld_pruned=True,
    )
    config = AIMProtocolConfig(
        population_design="three_continent",
        methods=("pca",),
        panel_size=4,
        seeds=(81, 82),
        bootstrap_resamples=100,
        bootstrap_seed=717,
    )
    return data, run_aim_protocol(data, config)


def test_loader_requires_one_unambiguous_chromosome_source(tmp_path: Path) -> None:
    genotype_path, metadata_path = _write_inputs(tmp_path)
    variant_path = tmp_path / "variants.csv"
    variant_path.write_text("snp_id,chromosome\nsnp_000,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one place"):
        load_protocol_input(
            genotype_path,
            metadata_path,
            source_manifest_sha256="a" * 64,
            input_scope="synthetic_multi_autosome_smoke",
            ld_pruned=True,
            variant_metadata=variant_path,
        )


def test_loader_accepts_embedded_population_labels_and_rejects_two_sources(
    tmp_path: Path,
) -> None:
    genotype_path, metadata_path = _write_inputs(tmp_path)
    with np.load(genotype_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    labels = np.repeat(np.asarray(["AFR", "EAS", "EUR"]), 12)
    embedded_path = tmp_path / "embedded_labels.npz"
    np.savez(embedded_path, **arrays, population_labels=labels)
    data = load_protocol_input(
        embedded_path,
        None,
        source_manifest_sha256="a" * 64,
        input_scope="synthetic_multi_autosome_smoke",
        ld_pruned=True,
    )
    np.testing.assert_array_equal(data.population_labels, labels)
    with pytest.raises(ValueError, match="exactly one place"):
        load_protocol_input(
            embedded_path,
            metadata_path,
            source_manifest_sha256="a" * 64,
            input_scope="synthetic_multi_autosome_smoke",
            ld_pruned=True,
        )


def test_cli_writes_strict_non_overwriting_real_data_contract(tmp_path: Path) -> None:
    genotype_path, metadata_path = _write_inputs(tmp_path)
    output = tmp_path / "result"
    command = _cli_command(genotype_path, metadata_path, output)
    environment = _environment()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    receipt = json.loads(completed.stdout)
    assert "genomics_aim_progress" not in completed.stdout
    progress = [
        json.loads(line)
        for line in completed.stderr.splitlines()
        if '"event":"genomics_aim_progress"' in line
    ]
    assert [event["phase"] for event in progress] == [
        "start",
        "method_seed_complete",
        "method_seed_complete",
        "end",
    ]
    assert [(event["method"], event["seed"]) for event in progress[1:-1]] == [
        ("pca", 81),
        ("pca", 82),
    ]
    assert [event["completed"] for event in progress] == [0, 1, 2, 2]
    assert {event["total"] for event in progress} == {2}
    assert all(event["elapsed_seconds_descriptive"] >= 0.0 for event in progress)
    assert progress[-1]["status"] == "completed"
    assert len(receipt["result_sha256"]) == 64
    expected = set(OUTPUT_FILE_NAMES.values())
    assert {path.name for path in output.iterdir()} == expected
    validate_output_package(output)
    payload_text = (output / "aim_protocol_results.json").read_text(encoding="utf-8")
    assert "NaN" not in payload_text and "Infinity" not in payload_text
    payload = json.loads(payload_text)
    assert len(payload["split_records"]) == 2
    assert all(
        record["panel_selection_used_population_labels"] is False
        for record in payload["split_records"]
    )
    assert payload["registered_seed_policy"] == "first_config_seed_not_best"
    assert {row["seed"] for row in payload["registered_coordinates"]} == {81}
    assert payload["solver_lineage"]["pca"]["maintained_callable"].startswith(
        "numpy.linalg.svd"
    )
    assert payload["solver_lineage"]["pca"]["supervised_feature_ranking"] is False
    manifest = json.loads(
        (output / "aim_protocol_output_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["solver_lineage"] == payload["solver_lineage"]
    for filename in (
        "aim_protocol_registered_stratification.png",
        "aim_protocol_panel_stability_and_delta.png",
    ):
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )


def test_progress_observability_preserves_every_result_artifact_byte(
    tmp_path: Path,
) -> None:
    genotype_path, metadata_path = _write_inputs(tmp_path)
    data = load_protocol_input(
        genotype_path,
        metadata_path,
        source_manifest_sha256="a" * 64,
        input_scope="synthetic_multi_autosome_smoke",
        ld_pruned=True,
    )
    config = AIMProtocolConfig(
        population_design="three_continent",
        methods=("pca",),
        panel_size=4,
        seeds=(81, 82),
        bootstrap_resamples=100,
        bootstrap_seed=717,
    )
    reference = tmp_path / "reference_without_progress"
    write_protocol_outputs(run_aim_protocol(data, config), data, reference)

    observed = tmp_path / "observed_with_progress"
    completed = subprocess.run(
        _cli_command(genotype_path, metadata_path, observed),
        cwd=Path(__file__).resolve().parents[1],
        env=_environment(),
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"event":"genomics_aim_progress"' in completed.stderr
    assert json.loads(completed.stdout)["result_sha256"] == json.loads(
        (reference / OUTPUT_FILE_NAMES["results_json"]).read_text(encoding="utf-8")
    )["result_sha256"]
    for filename in OUTPUT_FILE_NAMES.values():
        assert (observed / filename).read_bytes() == (reference / filename).read_bytes()


def test_progress_json_lines_have_deterministic_encoding() -> None:
    config = AIMProtocolConfig(
        population_design="three_continent",
        methods=("pca",),
        panel_size=4,
        seeds=(81,),
        bootstrap_resamples=100,
    )
    timestamps = iter((10.0, 10.0, 11.2346, 12.0))
    stream = io.StringIO()
    reporter = _ProgressReporter(
        config,
        stream=stream,
        clock=lambda: next(timestamps),
    )
    reporter.start()
    reporter.method_seed_complete(method="pca", seed=81)
    reporter.end(succeeded=True)

    assert stream.getvalue().splitlines() == [
        '{"completed":0,"elapsed_seconds_descriptive":0.0,'
        '"event":"genomics_aim_progress","method":null,"phase":"start",'
        '"seed":null,"status":"running","total":1}',
        '{"completed":1,"elapsed_seconds_descriptive":1.235,'
        '"event":"genomics_aim_progress","method":"pca",'
        '"phase":"method_seed_complete","seed":81,"status":"running",'
        '"total":1}',
        '{"completed":1,"elapsed_seconds_descriptive":2.0,'
        '"event":"genomics_aim_progress","method":null,"phase":"end",'
        '"seed":null,"status":"completed","total":1}',
    ]


def test_read_side_native_support_claim_audit_accepts_current_and_flags_frozen_v1(
    tmp_path: Path,
) -> None:
    data, result = _small_result(tmp_path)
    base = tmp_path / "native-support-base"
    write_protocol_outputs(result, data, base)
    current = audit_output_package_claim_gates(base)
    assert current["all_split_methods_native_support_at_least_panel_size"] is True
    assert current["native_support_failures"] == []
    audit_environment = _environment()
    audit_environment["PYTHONPATH"] = "src:."
    current_cli = subprocess.run(
        [
            sys.executable,
            "tools/audit_genomics_aim_output.py",
            "--require-native-support",
            str(base),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=audit_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert current_cli.returncode == 0

    frozen = tmp_path / "native-support-frozen-v1"
    shutil.copytree(base, frozen)
    results_path = frozen / OUTPUT_FILE_NAMES["results_json"]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    for record in payload["split_records"]:
        record.pop("prepanel_native_support_sufficient_for_panel", None)
    first_record = payload["split_records"][0]
    first_record["prepanel_native_support_count"] = (
        int(payload["config"]["panel_size"]) - 1
    )
    first_record["prepanel_native_support_fraction"] = (
        first_record["prepanel_native_support_count"]
        / first_record["training_retained_snps"]
    )
    digest_body = {
        "config": payload["config"],
        "input_identity": payload["input_identity"],
        "split_records": payload["split_records"],
        "method_summaries": payload["method_summaries"],
        "selection_frequency_hashes": payload["selection_frequency_hashes"],
        "registered_coordinates": payload["registered_coordinates"],
        "solver_lineage": payload["solver_lineage"],
    }
    from rspca.datasets import canonical_sha256

    payload["result_sha256"] = canonical_sha256(digest_body)
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    manifest_path = frozen / OUTPUT_FILE_NAMES["output_manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["result_sha256"] = payload["result_sha256"]
    manifest["files"]["results_json"] = {
        "path": OUTPUT_FILE_NAMES["results_json"],
        "sha256": hashlib.sha256(results_path.read_bytes()).hexdigest(),
        "bytes": results_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    audit = audit_output_package_claim_gates(frozen)
    assert audit["all_split_methods_native_support_at_least_panel_size"] is False
    assert audit["claim_status"] == "unqualified_zero_loading_tie_fill_risk"
    assert audit["native_support_failures"] == [
        {
            "seed": payload["split_records"][0]["seed"],
            "method": payload["split_records"][0]["method"],
            "prepanel_native_support_count": payload["config"]["panel_size"] - 1,
            "panel_size": payload["config"]["panel_size"],
            "risk": "zero-loading SNPs can be tie-filled into exact top-K",
        }
    ]
    frozen_cli = subprocess.run(
        [
            sys.executable,
            "tools/audit_genomics_aim_output.py",
            "--require-native-support",
            str(frozen),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=audit_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert frozen_cli.returncode == 2
    assert json.loads(frozen_cli.stdout)["claim_status"] == (
        "unqualified_zero_loading_tie_fill_risk"
    )


@pytest.mark.parametrize("target_kind", ["directory", "file", "dangling_symlink"])
def test_cli_preflights_output_before_loading_inputs(
    tmp_path: Path,
    target_kind: str,
) -> None:
    output = tmp_path / "already_exists"
    if target_kind == "directory":
        output.mkdir()
    elif target_kind == "file":
        output.write_text("occupied\n", encoding="utf-8")
    else:
        output.symlink_to(tmp_path / "missing_target")
        assert os.path.lexists(output) and not output.exists()
    command = _cli_command(
        tmp_path / "missing_genotypes.npz",
        tmp_path / "missing_populations.csv",
        output,
    )
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "output path already exists" in completed.stderr
    assert "missing_genotypes" not in completed.stderr


def test_writer_rejects_result_data_mismatch_before_creating_output(
    tmp_path: Path,
) -> None:
    data, result = _small_result(tmp_path)
    changed = np.asarray(data.genotypes, dtype=float).copy()
    changed[0, 0] = 0.0 if changed[0, 0] != 0.0 else 2.0
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError, match="input_identity"):
        write_protocol_outputs(
            result,
            replace(data, genotypes=changed),
            output,
        )
    assert not os.path.lexists(output)


def test_json_and_csv_writers_reject_nonfinite_values_before_writing(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "bad.json"
    csv_path = tmp_path / "bad.csv"
    with pytest.raises(ValueError, match="non-finite"):
        _write_json(json_path, {"nested": [{"value": np.inf}]})
    with pytest.raises(ValueError, match="non-finite"):
        _write_csv(csv_path, [{"method": "pca", "value": np.nan}])
    assert not json_path.exists()
    assert not csv_path.exists()


def test_fixed_package_validator_rejects_tamper_unknown_symlink_and_nonregular(
    tmp_path: Path,
) -> None:
    data, result = _small_result(tmp_path)
    base = tmp_path / "base"
    write_protocol_outputs(result, data, base)
    validate_output_package(base)

    tampered = tmp_path / "tampered"
    shutil.copytree(base, tampered)
    with (tampered / OUTPUT_FILE_NAMES["results_json"]).open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="byte-count|SHA-256|final package write"):
        validate_output_package(tampered)

    unknown = tmp_path / "unknown"
    shutil.copytree(base, unknown)
    (unknown / "extra.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="file set mismatch"):
        validate_output_package(unknown)

    symlinked = tmp_path / "symlinked"
    shutil.copytree(base, symlinked)
    symlink_path = symlinked / OUTPUT_FILE_NAMES["split_metrics_csv"]
    symlink_path.unlink()
    symlink_path.symlink_to(base / OUTPUT_FILE_NAMES["split_metrics_csv"])
    with pytest.raises(ValueError, match="must not be a symlink"):
        validate_output_package(symlinked)

    nonregular = tmp_path / "nonregular"
    shutil.copytree(base, nonregular)
    nonregular_path = nonregular / OUTPUT_FILE_NAMES["population_metrics_csv"]
    nonregular_path.unlink()
    nonregular_path.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        validate_output_package(nonregular)


def test_package_validator_strictly_rejects_json_nan_even_if_file_hash_is_updated(
    tmp_path: Path,
) -> None:
    data, result = _small_result(tmp_path)
    output = tmp_path / "nan_attack"
    write_protocol_outputs(result, data, output)
    results_path = output / OUTPUT_FILE_NAMES["results_json"]
    text = results_path.read_text(encoding="utf-8")
    text = text.replace('"schema_version": 2', '"attack": NaN,\n  "schema_version": 2')
    results_path.write_text(text, encoding="utf-8")

    manifest_path = output / OUTPUT_FILE_NAMES["output_manifest_json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["results_json"]["bytes"] = results_path.stat().st_size
    manifest["files"]["results_json"]["sha256"] = hashlib.sha256(
        results_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        validate_output_package(output)


def test_registered_pngs_are_deterministic_for_the_same_result(tmp_path: Path) -> None:
    data, result = _small_result(tmp_path)
    first = write_protocol_outputs(result, data, tmp_path / "first")
    second = write_protocol_outputs(result, data, tmp_path / "second")
    for role in ("registered_stratification_png", "panel_stability_png"):
        assert (
            hashlib.sha256(first[role].read_bytes()).digest()
            == hashlib.sha256(second[role].read_bytes()).digest()
        )

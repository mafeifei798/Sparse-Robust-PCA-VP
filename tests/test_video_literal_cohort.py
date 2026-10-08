from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

import experiments.video_spca as video_experiment
import experiments.video_spca_literal_cohort as literal_runner
from experiments.video_spca import MethodReport, VideoExperimentConfig
from experiments.video_spca_benchmark import VideoBenchmarkConfig, _generation_manifest
from experiments.video_spca_literal_cohort import (
    LITERAL_SEEDS,
    SEED_SOURCE,
    LiteralVideoCohortConfig,
    LiteralVideoCohortResult,
    run_literal_video_cohort,
    validate_literal_video_cohort_output_directory,
    write_literal_video_cohort,
)
from rspca.datasets import canonical_sha256


_SEALED_RESULT_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "video_literal_2026-08-30"
)
_SEALED_SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "video_literal_2026-08-30_execution_source"
)


def _smoke_config() -> LiteralVideoCohortConfig:
    return LiteralVideoCohortConfig(
        video_config=VideoExperimentConfig(
            grid_size=8,
            n_timesteps=16,
            total_time=5.0,
            n_components=3,
            noise_std=0.02,
            corruption_fraction=0.01,
            corruption_scale=0.5,
            seed=91001,
            alpha=0.08,
            max_iter=1,
            tol=1e-4,
            randomized_oversample=1,
            randomized_power_iters=0,
            randomized_full_refine_iters=0,
        ),
        methods=("PCA", "Robust-VP-SPCA"),
        bootstrap_resamples=16,
        bootstrap_seed=7,
    )


@pytest.fixture(scope="module")
def literal_result() -> LiteralVideoCohortResult:
    return run_literal_video_cohort(_smoke_config())


def test_runner_uses_exact_literal_seeds_and_never_masquerades_as_derived(
    literal_result: LiteralVideoCohortResult,
) -> None:
    assert LITERAL_SEEDS == tuple(range(91001, 91021))
    manifest = literal_result.cohort_manifest
    assert manifest["seed_source"] == SEED_SOURCE == "literal_predeclared"
    assert manifest["seeds"] == list(range(91001, 91021))
    assert manifest["seed_count"] == 20
    assert manifest["manifest_derived_seed_generation"] is False
    assert manifest["distinct_from_manifest_derived_base_91001_cohort"] is True
    assert manifest["formal_default_config"]["exact"] is False
    assert literal_runner._formal_default_contract(LiteralVideoCohortConfig())[
        "exact"
    ] is True
    body = dict(manifest)
    assert body.pop("manifest_sha256") == canonical_sha256(body)
    closure = manifest["source_closure"]
    assert closure["package_initializer"] == "src/rspca/__init__.py"
    assert set(closure["eager_import_modules"]) == {
        "accelerated",
        "cross_modal_gates",
        "datasets",
        "federated",
        "fedrpca_audit",
        "gauge",
        "gauge_stress",
        "metrics",
        "product_aggregation",
        "randomized",
        "robust",
        "statistical_gates",
        "trust_audit",
        "vp",
    }
    sources = {record["path"]: record for record in manifest["source_files"]}
    assert sources["src/rspca/__init__.py"]["role"] == "package_initializer"
    assert sources["src/rspca/federated.py"]["role"] == (
        "package_initialization_only"
    )
    assert sources["src/rspca/federated.py"]["scientific_method_dependency"] is False
    assert sources["src/rspca/accelerated.py"]["role"] == (
        "scientific_or_provenance_dependency"
    )
    assert manifest["execution_environment"]["variables"] == {
        key: os.environ.get(key) for key in literal_runner._RUNTIME_ENVIRONMENT_KEYS
    }

    derived = _generation_manifest(
        VideoBenchmarkConfig(
            video_config=_smoke_config().video_config,
            repeats=20,
            base_seed=91001,
            methods=_smoke_config().methods,
            bootstrap_resamples=16,
            bootstrap_seed=7,
        )
    )
    assert manifest["seeds"] != derived["seeds"]
    assert derived["metadata"]["seed_source"] == "seed_manifest.seeds[run_index]"


def test_records_preserve_complete_method_reports_dataset_provenance_and_budget(
    literal_result: LiteralVideoCohortResult,
) -> None:
    assert len(literal_result.records) == 20 * 2
    report_fields = set(MethodReport.__dataclass_fields__)
    manifest_sha256 = literal_result.cohort_manifest["manifest_sha256"]
    for run_index, seed in enumerate(LITERAL_SEEDS):
        rows = [row for row in literal_result.records if row["run"] == run_index]
        assert len(rows) == 2
        assert {row["run_index"] for row in rows} == {run_index}
        assert {row["seed"] for row in rows} == {seed}
        assert {row["seed_source"] for row in rows} == {SEED_SOURCE}
        assert {row["cohort_manifest_sha256"] for row in rows} == {
            manifest_sha256
        }
        assert all(report_fields <= set(row) for row in rows)
        provenance = literal_result.cohort_manifest["run_provenance"][run_index]
        assert {row["data_sha256"] for row in rows} == {
            provenance["data_sha256"]
        }
        assert {row["dataset_manifest_sha256"] for row in rows} == {
            provenance["dataset_manifest_sha256"]
        }
        dataset_manifest = dict(provenance["dataset_manifest"])
        assert dataset_manifest.pop("manifest_sha256") == canonical_sha256(
            dataset_manifest
        )

    robust_records = [
        row for row in literal_result.records if row["method"] == "Robust-VP-SPCA"
    ]
    assert len(robust_records) == 20
    assert all(row["corruption_detection_at_budget"] is not None for row in robust_records)
    assert {
        summary["metric"] for summary in literal_result.corruption_budget_summaries
    } >= {
        "corruption_detection_at_budget.f1",
        "corruption_detection_at_budget.precision",
        "corruption_detection_at_budget.recall",
    }
    assert all(
        summary["n_seeds"] == 20
        for summary in literal_result.corruption_budget_summaries
    )


def test_writer_is_strict_exclusive_and_keeps_nested_budget_in_json_and_csv(
    tmp_path: Path,
    literal_result: LiteralVideoCohortResult,
) -> None:
    output_dir = tmp_path / "literal"
    paths = write_literal_video_cohort(literal_result, output_dir)
    original = {name: path.read_bytes() for name, path in paths.items()}
    assert set(paths) == {"json", "csv", "manifest"}
    completion = validate_literal_video_cohort_output_directory(output_dir)
    assert completion["self_excluding_manifest"] is True
    assert completion["formal_default_config_exact"] is False
    assert paths["manifest"].name not in {
        record["path"] for record in completion["artifacts"]
    }
    payload = json.loads(
        paths["json"].read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(f"non-standard JSON constant: {token}")
        ),
    )
    assert payload["config"]["seed_source"] == SEED_SOURCE
    assert payload["config"]["seeds"] == list(range(91001, 91021))
    assert payload["config"]["formal_default_config"]["exact"] is False
    assert payload["evaluation_protocol"]["manifest_derived_seed_generation"] is False
    assert payload["records"][0]["seed_source"] == SEED_SOURCE
    robust = next(
        record for record in payload["records"] if record["method"] == "Robust-VP-SPCA"
    )
    assert robust["corruption_detection_at_budget"]["budget_top_k"] > 0

    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["row_type"] for row in rows} == {
        "per_seed",
        "method_summary",
        "paired_summary",
        "corruption_budget_summary",
    }
    robust_csv = next(
        row
        for row in rows
        if row["row_type"] == "per_seed" and row["method"] == "Robust-VP-SPCA"
    )
    assert robust_csv["corruption_detection_at_budget_f1"]

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_literal_video_cohort(literal_result, output_dir)
    assert {name: path.read_bytes() for name, path in paths.items()} == original


def test_output_validator_rejects_tamper_and_undeclared_closure(
    tmp_path: Path,
    literal_result: LiteralVideoCohortResult,
) -> None:
    tampered_output = tmp_path / "tampered-output"
    paths = write_literal_video_cohort(literal_result, tampered_output)
    paths["csv"].write_bytes(paths["csv"].read_bytes() + b"tamper\n")
    with pytest.raises(ValueError, match="artifact hash/size mismatch"):
        validate_literal_video_cohort_output_directory(tampered_output)

    closure_output = tmp_path / "bad-closure"
    write_literal_video_cohort(literal_result, closure_output)
    (closure_output / "undeclared.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="file closure mismatch"):
        validate_literal_video_cohort_output_directory(closure_output)


def test_partial_literal_write_has_no_completion_manifest(
    tmp_path: Path,
    literal_result: LiteralVideoCohortResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "partial"
    real_open = video_experiment.os.open

    def fail_on_csv(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "literal_video_cohort.csv":
            raise OSError("synthetic literal CSV failure")
        if dir_fd is None:
            return real_open(path, flags, mode)  # type: ignore[arg-type]
        return real_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(video_experiment.os, "open", fail_on_csv)
    with pytest.raises(OSError, match="synthetic literal CSV failure"):
        write_literal_video_cohort(literal_result, output_dir)

    assert output_dir.is_dir()
    assert (output_dir / "literal_video_cohort.json").is_file()
    assert not (output_dir / "literal_video_cohort_output_manifest.json").exists()
    with pytest.raises(ValueError, match="completion manifest missing"):
        validate_literal_video_cohort_output_directory(output_dir)


def test_writer_rejects_nonfinite_tamper_before_claim(
    tmp_path: Path,
    literal_result: LiteralVideoCohortResult,
) -> None:
    records = deepcopy(literal_result.records)
    records[0]["extra"]["nonfinite"] = float("nan")
    tampered = replace(literal_result, records=records)
    output_dir = tmp_path / "tampered"

    with pytest.raises(ValueError, match="number must be finite"):
        write_literal_video_cohort(tampered, output_dir)
    assert not os.path.lexists(output_dir)


@pytest.mark.parametrize(
    "tampered_path",
    ["src/rspca/__init__.py", "src/rspca/federated.py"],
)
def test_writer_rejects_resigned_initializer_or_eager_dependency_hash_tamper(
    tmp_path: Path,
    literal_result: LiteralVideoCohortResult,
    tampered_path: str,
) -> None:
    manifest = deepcopy(literal_result.cohort_manifest)
    source_record = next(
        record for record in manifest["source_files"] if record["path"] == tampered_path
    )
    source_record["sha256"] = "0" * 64
    manifest_body = dict(manifest)
    manifest_body.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_sha256(manifest_body)
    records = deepcopy(literal_result.records)
    for record in records:
        record["cohort_manifest_sha256"] = manifest["manifest_sha256"]
    tampered = replace(
        literal_result,
        cohort_manifest=manifest,
        records=tuple(records),
    )
    output_dir = tmp_path / Path(tampered_path).stem

    with pytest.raises(ValueError, match="current sources/config/provenance"):
        write_literal_video_cohort(tampered, output_dir)
    assert not os.path.lexists(output_dir)


def test_cli_preflights_output_before_running_twenty_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    def runner_must_not_execute(*_: object, **__: object) -> object:
        raise AssertionError("literal cohort executed before output preflight")

    monkeypatch.setattr(literal_runner, "run_literal_video_cohort", runner_must_not_execute)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        literal_runner.main(["--output-dir", str(output_dir)])


def test_sealed_formal_literal_cohort_artifact_contract() -> None:
    expected_file_sha256 = {
        "literal_video_cohort.json": (
            "d21780a45084623f8d3e2dcfcc8b95899157d66ffd4a0054bf49a635cf474220"
        ),
        "literal_video_cohort.csv": (
            "50c2ad2848fbc0f507dd40d031c0d3baa086c29c7225e9c6cf9885e6dcefb44c"
        ),
        "literal_video_cohort_output_manifest.json": (
            "0cb3e28f57f7d40006400e1283a5779b641b9abb30b3e2d75895f2b07256d11a"
        ),
    }
    assert {path.name for path in _SEALED_RESULT_DIR.iterdir()} == set(
        expected_file_sha256
    )
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _SEALED_RESULT_DIR.iterdir()
    } == expected_file_sha256

    completion = validate_literal_video_cohort_output_directory(
        _SEALED_RESULT_DIR,
        source_root=_SEALED_SOURCE_DIR,
    )
    assert completion["manifest_sha256"] == (
        "d53bee44a5add4d88783581215094fa8394b1b92ca56fae849f92f40b0ca9307"
    )
    assert completion["cohort_manifest_sha256"] == (
        "819789e1e8365e89f8d13c34e1d0304283884461f887b73f45837378eca8c014"
    )
    assert completion["formal_default_config_exact"] is True

    result_path = _SEALED_RESULT_DIR / "literal_video_cohort.json"
    payload = json.loads(
        result_path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            AssertionError(f"non-standard JSON constant: {token}")
        ),
    )
    assert payload["config"]["seed_source"] == "literal_predeclared"
    assert payload["config"]["seeds"] == list(range(91001, 91021))
    assert payload["config"]["formal_default_config"]["exact"] is True
    assert payload["evaluation_protocol"]["best_seed_selection"] is False
    assert len(payload["records"]) == 20 * 4

    records_by_method = {
        method: [
            record for record in payload["records"] if record["method"] == method
        ]
        for method in ("PCA", "VP-SPCA", "RND-VP-SPCA", "Robust-VP-SPCA")
    }
    assert sum(record["converged"] for record in records_by_method["PCA"]) == 20
    assert {
        method: sum(record["converged"] for record in records_by_method[method])
        for method in ("VP-SPCA", "RND-VP-SPCA", "Robust-VP-SPCA")
    } == {"VP-SPCA": 0, "RND-VP-SPCA": 0, "Robust-VP-SPCA": 0}

    def method_mean(method: str, metric: str) -> float:
        summary = next(
            item
            for item in payload["method_summaries"]
            if item["method"] == method and item["metric"] == metric
        )
        return float(summary["mean"])

    pca_clean = method_mean("PCA", "clean_reconstruction_error")
    robust_clean = method_mean("Robust-VP-SPCA", "clean_reconstruction_error")
    assert pca_clean == pytest.approx(0.6042617845444261, abs=1e-15)
    assert robust_clean == pytest.approx(0.08575223661873115, abs=1e-15)
    assert (pca_clean - robust_clean) / pca_clean == pytest.approx(
        0.8580876057164815, abs=1e-15
    )

    budget_f1 = next(
        item
        for item in payload["corruption_budget_summaries"]
        if item["method"] == "Robust-VP-SPCA"
        and item["metric"] == "corruption_detection_at_budget.f1"
    )
    assert budget_f1["mean"] == pytest.approx(0.9989151071331707, abs=1e-15)
    assert budget_f1["converged_count"] == 0

    robust_records = records_by_method["Robust-VP-SPCA"]
    hard_case = next(record for record in robust_records if record["seed"] == 91007)
    assert hard_case["converged"] is False
    assert hard_case["status"] == "maximum iterations reached"
    assert hard_case["clean_reconstruction_error"] == pytest.approx(
        0.6195775062140684, abs=1e-15
    )
    assert hard_case["corruption_detection_at_budget"]["f1"] == pytest.approx(
        0.9834553837808516, abs=1e-15
    )
    assert hard_case["clean_reconstruction_error"] == max(
        record["clean_reconstruction_error"] for record in robust_records
    )
    assert max(
        record["clean_reconstruction_error"]
        for record in robust_records
        if record["seed"] != 91007
    ) < 0.059

    assert payload["paired_summaries"]
    assert all(
        summary["inference_qualified"] is False
        and summary["qualified_pairs"] == 0
        and summary["bootstrap_ci95_status"]
        == "unqualified_fixed_budget_runs_included"
        for summary in payload["paired_summaries"]
    )


def test_sealed_literal_cohort_uses_immutable_execution_snapshot(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="current source tree"):
        validate_literal_video_cohort_output_directory(_SEALED_RESULT_DIR)

    tampered_snapshot = tmp_path / "source_snapshot"
    shutil.copytree(_SEALED_SOURCE_DIR, tampered_snapshot)
    randomized = tampered_snapshot / "src" / "rspca" / "randomized.py"
    randomized.write_bytes(randomized.read_bytes() + b"\n# tampered snapshot\n")

    with pytest.raises(ValueError, match="packaged execution snapshot"):
        validate_literal_video_cohort_output_directory(
            _SEALED_RESULT_DIR,
            source_root=tampered_snapshot,
        )

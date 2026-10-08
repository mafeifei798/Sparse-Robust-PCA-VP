from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments import genomics_1kg
from experiments.genomics_1kg import GenomicsConfig, write_results


def _valid_inputs() -> tuple[
    list[dict[str, Any]], dict[str, Any], GenomicsConfig, dict[str, Any]
]:
    rows = [
        {
            "record_type": "fixture",
            "method": "pca",
            "metric": 0.75,
            "optional_diagnostic": None,
        }
    ]
    manifest = {
        "manifest_sha256": "a" * 64,
        "seeds": [2025],
        "optional_parent": None,
    }
    config = GenomicsConfig(bootstrap_repeats=0, missing_value=None)
    provenance = {
        "loader_schema_version": 2,
        "genotype_sha256": "b" * 64,
        "metadata_sha256": None,
        "sample_ids_sha256": "c" * 64,
        "snp_ids_sha256": "d" * 64,
        "input_orientation": "samples-by-snps",
        "missing_value_policy": "disabled",
        "full_cohort_retained_snp_ids_sha256": "e" * 64,
        "optional_nested": {"value": None},
    }
    return rows, manifest, config, provenance


def test_write_results_preserves_valid_none_in_strict_json_and_csv(
    tmp_path: Path,
) -> None:
    rows, manifest, config, provenance = _valid_inputs()
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "metrics.csv"

    assert (
        write_results(
            rows,
            manifest,
            config,
            provenance,
            output_json=json_path,
            output_csv=csv_path,
        )
        is None
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["records"][0]["optional_diagnostic"] is None
    assert payload["config"]["missing_value"] is None
    assert payload["provenance"]["optional_nested"]["value"] is None
    assert csv_path.is_file()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_write_results_rejects_nested_nonfinite_before_creating_any_output(
    tmp_path: Path, bad_value: float
) -> None:
    rows, manifest, config, provenance = _valid_inputs()
    rows[0]["nested"] = {"diagnostics": [0.0, np.float64(bad_value)]}
    json_path = tmp_path / "new" / "results.json"
    csv_path = tmp_path / "new" / "metrics.csv"

    with pytest.raises(
        ValueError,
        match=r"number must be finite at root\.records\[0\]\.nested\.diagnostics\[1\]",
    ):
        write_results(
            rows,
            manifest,
            config,
            provenance,
            output_json=json_path,
            output_csv=csv_path,
        )

    assert not json_path.exists()
    assert not csv_path.exists()
    assert not json_path.parent.exists()


def test_write_results_preflights_existing_targets_without_partial_writes(
    tmp_path: Path,
) -> None:
    rows, manifest, config, provenance = _valid_inputs()
    json_path = tmp_path / "results.json"
    json_path.write_text("owner data\n", encoding="utf-8")
    csv_path = tmp_path / "new" / "metrics.csv"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_results(
            rows,
            manifest,
            config,
            provenance,
            output_json=json_path,
            output_csv=csv_path,
        )

    assert json_path.read_text(encoding="utf-8") == "owner data\n"
    assert not csv_path.exists()
    assert not csv_path.parent.exists()


def test_write_results_rejects_same_resolved_json_and_csv_path(
    tmp_path: Path,
) -> None:
    rows, manifest, config, provenance = _valid_inputs()
    target = tmp_path / "same-output"

    with pytest.raises(ValueError, match="must be different paths"):
        write_results(
            rows,
            manifest,
            config,
            provenance,
            output_json=target,
            output_csv=target,
        )

    assert not target.exists()


def test_exclusive_create_still_refuses_a_target_created_after_preflight(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    rows, manifest, config, provenance = _valid_inputs()
    json_path = tmp_path / "results.json"
    original_resolver = genomics_1kg._resolved_output_paths

    def racing_resolver(
        output_json: str | Path | None,
        output_csv: str | Path | None,
    ) -> dict[str, Path]:
        paths = original_resolver(output_json, output_csv)
        json_path.write_text("racing owner\n", encoding="utf-8")
        return paths

    monkeypatch.setattr(genomics_1kg, "_resolved_output_paths", racing_resolver)
    with pytest.raises(FileExistsError):
        write_results(
            rows,
            manifest,
            config,
            provenance,
            output_json=json_path,
        )

    assert json_path.read_text(encoding="utf-8") == "racing owner\n"


def test_cli_preflights_outputs_before_loading_genotypes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    existing = tmp_path / "results.json"
    existing.write_text("owner data\n", encoding="utf-8")

    def unexpected_load(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("genotypes must not be loaded after failed output preflight")

    monkeypatch.setattr(genomics_1kg, "load_genotypes", unexpected_load)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        genomics_1kg.main(
            [str(tmp_path / "not-read.npy"), "--output-json", str(existing)]
        )

    assert existing.read_text(encoding="utf-8") == "owner data\n"

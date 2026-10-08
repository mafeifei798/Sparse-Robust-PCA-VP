from __future__ import annotations

import csv
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image
import pytest

from tools.render_genomics_aim_presentation import (
    ALL_METHODS,
    FORMAL_COORDINATES,
    FORMAL_FILENAMES,
    FORMAL_MANIFEST,
    FORMAL_METHOD_SUMMARIES,
    FORMAL_POPULATION_METRICS,
    OUTPUT_MANIFEST,
    PRIMARY_METHODS,
    README,
    RECALL_FIGURE,
    RECEIPT,
    ROBUST_FIGURE,
    SCATTER_FIGURE,
    SUMMARY_FIGURE,
    render_presentation,
    validate_formal_package,
    validate_presentation_output,
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write_formal(root: Path, populations: tuple[str, ...], *, five: bool) -> None:
    root.mkdir()
    coordinate_fields = [
        "registered_seed_policy",
        "seed",
        "method",
        "evaluation",
        "sample_index",
        "sample_id",
        "truth",
        "prediction",
        "component_1",
        "component_2",
        "qualification_status",
        "overall_qualified",
    ]
    with (root / FORMAL_COORDINATES).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=coordinate_fields)
        writer.writeheader()
        for method_index, method in enumerate(ALL_METHODS):
            for population_index, population in enumerate(populations):
                for sample in range(4):
                    incorrect = five and population in {"AMR", "SAS"} and sample == 0
                    writer.writerow(
                        {
                            "registered_seed_policy": "first_config_seed_not_best",
                            "seed": 2025,
                            "method": method,
                            "evaluation": "top_k_common_pca_probe",
                            "sample_index": population_index * 4 + sample,
                            "sample_id": f"sample_{population_index}_{sample}",
                            "truth": population,
                            "prediction": populations[(population_index + 1) % len(populations)] if incorrect else population,
                            "component_1": population_index * 2.0 + sample * 0.15 + method_index * 0.01,
                            "component_2": population_index * -1.4 + sample * 0.12,
                            "qualification_status": "qualified" if method == "pca" else "unqualified",
                            "overall_qualified": method == "pca",
                        }
                    )

    population_fields = [
        "seed",
        "method",
        "evaluation",
        "population",
        "support",
        "predicted_count",
        "true_positive",
        "precision",
        "recall",
        "f1",
    ]
    with (root / FORMAL_POPULATION_METRICS).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=population_fields)
        writer.writeheader()
        for seed in range(2025, 2045):
            for method_index, method in enumerate(ALL_METHODS):
                for population in populations:
                    base = 0.99
                    if population == "AMR":
                        base = 0.34 + (0.08 if method == "robust" else 0.0)
                    elif population == "SAS":
                        base = 0.62 + (0.04 if method == "robust" else 0.0)
                    recall = max(0.0, min(1.0, base + 0.002 * ((seed - 2025) % 3 - 1)))
                    writer.writerow(
                        {
                            "seed": seed,
                            "method": method,
                            "evaluation": "common_pca_probe",
                            "population": population,
                            "support": 40,
                            "predicted_count": 40,
                            "true_positive": round(40 * recall),
                            "precision": recall,
                            "recall": recall,
                            "f1": recall,
                        }
                    )

    summary_fields = [
        "method",
        "splits",
        "panel_size",
        "mean_full_width_native_balanced_accuracy",
        "mean_common_pca_probe_balanced_accuracy",
        "mean_panel_jaccard_with_equal_budget_pca",
        "mean_pairwise_panel_jaccard_across_splits",
        "qualified_splits",
        "qualification_status",
    ]
    with (root / FORMAL_METHOD_SUMMARIES).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for method_index, method in enumerate(ALL_METHODS):
            writer.writerow(
                {
                    "method": method,
                    "splits": 20,
                    "panel_size": 200,
                    "mean_full_width_native_balanced_accuracy": 0.995 if not five else (0.815 if method == "robust" else 0.831),
                    "mean_common_pca_probe_balanced_accuracy": 0.996 if not five else (0.808 if method == "robust" else 0.783 + 0.0002 * method_index),
                    "mean_panel_jaccard_with_equal_budget_pca": 1.0 if method == "pca" else (0.50 if method == "robust" else 0.999),
                    "mean_pairwise_panel_jaccard_across_splits": 0.61 if method == "robust" else 0.79,
                    "qualified_splits": 20 if method == "pca" else 0,
                    "qualification_status": "all_splits_qualified" if method == "pca" else "no_splits_qualified",
                }
            )

    for role, name in FORMAL_FILENAMES.items():
        path = root / name
        if path.exists():
            continue
        if name.endswith(".png"):
            path.write_bytes(b"synthetic png placeholder")
        elif name.endswith(".json"):
            path.write_text("{}\n", encoding="utf-8")
        else:
            path.write_text("field\nvalue\n", encoding="utf-8")

    expected_names = set(FORMAL_FILENAMES.values()) | {FORMAL_MANIFEST}
    manifest = {
        "schema_version": 2,
        "package_contract": "fixed_file_set_v1",
        "completion_status": "complete",
        "result_sha256": sha256(root.name.encode()).hexdigest(),
        "registered_seed": 2025,
        "registered_seed_policy": "first_config_seed_not_best",
        "solver_lineage": {},
        "expected_filenames": sorted(expected_names),
        "files": {
            role: {
                "path": name,
                "sha256": _digest(root / name),
                "bytes": (root / name).stat().st_size,
            }
            for role, name in FORMAL_FILENAMES.items()
        },
    }
    (root / FORMAL_MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_render_is_readable_closed_exclusive_and_source_preserving(tmp_path: Path) -> None:
    three = tmp_path / "formal_three"
    five = tmp_path / "formal_five"
    output = tmp_path / "presentation"
    _write_formal(three, ("AFR", "EAS", "EUR"), five=False)
    _write_formal(five, ("AFR", "AMR", "EAS", "EUR", "SAS"), five=True)
    before = {
        "three": {path.name: path.read_bytes() for path in three.iterdir()},
        "five": {path.name: path.read_bytes() for path in five.iterdir()},
    }

    render_presentation(three, five, output)
    manifest = validate_presentation_output(output)
    assert manifest["required_payloads"] == sorted(
        (SCATTER_FIGURE, RECALL_FIGURE, SUMMARY_FIGURE, ROBUST_FIGURE, RECEIPT, README)
    )
    assert {path.name for path in output.iterdir()} == set(manifest["required_payloads"]) | {OUTPUT_MANIFEST}
    assert before["three"] == {path.name: path.read_bytes() for path in three.iterdir()}
    assert before["five"] == {path.name: path.read_bytes() for path in five.iterdir()}

    for name in (SCATTER_FIGURE, RECALL_FIGURE, SUMMARY_FIGURE, ROBUST_FIGURE):
        with Image.open(output / name) as image:
            assert image.width >= 2500
            assert image.height >= 1200

    receipt = json.loads((output / RECEIPT).read_text(encoding="utf-8"))
    assert receipt["solver_called"] is False
    assert receipt["formal_sources_unchanged_during_render"] is True
    assert receipt["primary_methods"] == list(PRIMARY_METHODS)
    assert receipt["supplementary_methods"] == ["robust"]
    assert receipt["plotted_values"]["registered_scatter"]["registered_seed"] == 2025
    assert receipt["plotted_values"]["registered_scatter"]["methods"]["pca"]["misclassified"] == 2
    assert receipt["plotted_values"]["three_vs_five_summary"]["qualified_splits"]["5 groups"] == {
        "pca": 20,
        "randomized": 0,
        "vp": 0,
    }
    assert "not race" in (output / README).read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        render_presentation(three, five, output)


def test_formal_hash_tampering_is_rejected(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    _write_formal(formal, ("AFR", "EAS", "EUR"), five=False)
    with (formal / FORMAL_COORDINATES).open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="byte-count mismatch|SHA-256 mismatch"):
        validate_formal_package(formal)


def test_presentation_hash_tampering_is_rejected(tmp_path: Path) -> None:
    three = tmp_path / "formal_three"
    five = tmp_path / "formal_five"
    output = tmp_path / "presentation"
    _write_formal(three, ("AFR", "EAS", "EUR"), five=False)
    _write_formal(five, ("AFR", "AMR", "EAS", "EUR", "SAS"), five=True)
    render_presentation(three, five, output)
    with (output / README).open("a", encoding="utf-8") as stream:
        stream.write("tamper\n")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        validate_presentation_output(output)

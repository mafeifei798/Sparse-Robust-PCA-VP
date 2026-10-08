#!/usr/bin/env python3
"""Locked paired confirmation for the PCA-only Genomics loading diagnostic.

The development grid is consumed only to create a machine-readable lock.  A
separate command then compares exactly two frozen configurations on disjoint
outer splits: the historical 2-PC/top-200 loading panel and the locked
4-PC/top-400 loading panel.  Both use outer-training-only smartpca scaling,
unsupervised PCA-loading ranking, a panel PCA refit, training centroids, and
held-out scoring.  No VP-SPCA, randomized VP-SPCA, or Robust VP-SPCA solver is
called by this tool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

import numpy as np

if __package__:
    from . import genomics_pca_loading_grid as grid
else:  # Support direct execution from the repository root.
    import genomics_pca_loading_grid as grid  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    stratified_split,
    transform_smartpca,
)


BASELINE = {"name": "baseline_2pc_top200", "n_components": 2, "panel_size": 200}
LOCKED = {"name": "locked_4pc_top400", "n_components": 4, "panel_size": 400}
CONFIRMATION_SEEDS = tuple(range(3001, 3021))
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 410_400
EXPECTED_POPULATIONS = grid.EXPECTED_POPULATIONS
LOCK_SCHEMA_VERSION = 1
CONFIRMATION_SCHEMA_VERSION = 1
EXPECTED_PREPARED_SAMPLES = 1000
SELECTION_REASON = (
    "The complete PCA-only post-hoc development grid showed stronger separation "
    "for higher-component configurations. Four PCs and top-400 were locked as one "
    "combined moderate-budget configuration; this comparison does not isolate the "
    "PC-count effect from the panel-size effect, and Robust methods are out of scope."
)
PRE_OUTCOME_BOUNDARY = (
    "The externally anchored lock-file hash must exist before the run-start "
    "receipt. The semantic seed audit covers only evidence files in the supplied "
    "canonical repository and does not prove absence of unseen external runs."
)
FROZEN_CONFIRMATION_PROTOCOL = {
    "seeds": list(CONFIRMATION_SEEDS),
    "test_fraction": 0.2,
    "paired_same_outer_split": True,
    "preprocessing": "outer_train_only_smartpca_like",
    "ranking": "outer_train_only_sum_abs_pca_loadings",
    "panel_refit": "outer_train_only_closed_form_pca",
    "classifier": "nearest_outer_train_centroid",
    "held_out_use": "scoring_only",
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_unit": "paired_outer_split_seed",
    "bootstrap_interval": "percentile_95_two_sided",
    "estimand": (
        "mean paired performance over the frozen random 80/20 split mechanism, "
        "conditional on this fixed prepared 1000-person dataset"
    ),
    "repeated_individuals_across_splits": True,
    "cohort_or_population_generalization_interval": False,
    "registered_figure_seed": CONFIRMATION_SEEDS[0],
}
FROZEN_CLAIM_GATES = {
    "primary_balanced_accuracy": "paired_bootstrap_ci95_lower_gt_0",
    "amr_recall": "paired_bootstrap_ci95_lower_ge_0_and_mean_delta_gt_0",
    "sas_recall": "paired_bootstrap_ci95_lower_ge_0_and_mean_delta_gt_0",
    "solver_qualification": "closed_form_pca_all_40_fits",
    "overall": "all_four_gates",
}
FROZEN_METHOD_SCOPE = {
    "methods": ["pca"],
    "vp_spca_run": False,
    "randomized_vp_spca_run": False,
    "robust_vp_spca_run": False,
}
PREPARED_INPUT_IDENTITY_FIELDS = {
    "chromosomes_sha256",
    "expected_external_source_manifest_sha256",
    "external_source_manifest_anchor_verified",
    "external_source_manifest_bytes",
    "external_source_manifest_sha256",
    "genotype_npz_bytes",
    "genotype_npz_sha256",
    "genotypes_sha256",
    "ld_pruned_input_validated_from_external_manifest",
    "manifest_input_scope",
    "manifest_ld_pruned_receipts_verified",
    "manifest_population_design",
    "orientation",
    "population_labels_sha256",
    "sample_ids_sha256",
    "samples",
    "snp_ids_sha256",
    "snps",
}
LOCK_FILENAME = "pca_loading_confirmation_lock.json"
RESULT_FILENAME = "pca_loading_confirmation_results.json"
PAIRED_FILENAME = "pca_loading_confirmation_paired_metrics.csv"
SUMMARY_FILENAME = "pca_loading_confirmation_summary.csv"
COORDINATES_FILENAME = "pca_loading_confirmation_registered_coordinates.csv"
PC12_FILENAME = "pca_loading_confirmation_pc12.png"
PC34_FILENAME = "pca_loading_confirmation_pc34.png"
MANIFEST_FILENAME = "pca_loading_confirmation_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    PAIRED_FILENAME,
    SUMMARY_FILENAME,
    COORDINATES_FILENAME,
    PC12_FILENAME,
    PC34_FILENAME,
)
SOURCE_FILE_ROLES = {
    "experiments/genomics_pca_loading_confirmation.py": "locked_confirmation_runner",
    "experiments/genomics_pca_loading_grid.py": "development_protocol_and_shared_pca_helpers",
    "src/rspca/genomics_aim.py": "historical_genomics_preprocessing_and_split",
    "src/rspca/datasets.py": "hashing_and_runtime_identity",
    "src/rspca/__init__.py": "eager_package_initialization",
    "src/rspca/accelerated.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/cross_modal_gates.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/federated.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/fedrpca_audit.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/gauge.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/gauge_stress.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/metrics.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/product_aggregation.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/randomized.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/robust.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/statistical_gates.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/trust_audit.py": "eager_import_only_not_called_by_confirmation",
    "src/rspca/vp.py": "eager_import_only_not_called_by_confirmation",
}
SEED_FIELD_NAMES = {
    "seed",
    "seeds",
    "requested_seed",
    "requested_seeds",
    "split_seed",
    "split_seeds",
    "base_seed",
    "bootstrap_seed",
    "selection_seed",
    "random_state",
}


def _strict_json_load(path: Path) -> Any:
    return grid._strict_json_load(path)


def _strict_json_bytes(payload: Any) -> bytes:
    return grid._strict_json_bytes(payload)


def _file_sha256(path: Path) -> str:
    return grid._file_sha256(path)


def _is_sha256(value: Any) -> bool:
    return grid._is_sha256(value)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    return grid._regular_file(path, role=role)


def _seed_values(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, list):
        return tuple(int(item) for item in value if isinstance(item, int) and not isinstance(item, bool))
    return ()


def _json_seed_hits(
    value: Any,
    wanted: set[int],
    *,
    location: str = "root",
) -> tuple[list[dict[str, Any]], int]:
    hits: list[dict[str, Any]] = []
    fields_seen = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            child = f"{location}.{key}"
            if normalized in SEED_FIELD_NAMES or normalized.endswith("_seed") or normalized.endswith("_seeds"):
                fields_seen += 1
                for seed in _seed_values(item):
                    if seed in wanted:
                        hits.append({"location": child, "seed": seed})
            child_hits, child_fields = _json_seed_hits(item, wanted, location=child)
            hits.extend(child_hits)
            fields_seen += child_fields
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_hits, child_fields = _json_seed_hits(
                item, wanted, location=f"{location}[{index}]"
            )
            hits.extend(child_hits)
            fields_seen += child_fields
    return hits, fields_seen


def _is_genomics_evidence_path(relative: Path) -> bool:
    lowered = "/".join(relative.parts).lower()
    if "/source/" in f"/{lowered}/" or "/provenance/source_snapshot/" in f"/{lowered}/":
        return False
    return any(token in lowered for token in ("genom", "gene", "aim", "snp", "population"))


def audit_confirmation_seed_novelty(
    repo_root: Path,
    seeds: Sequence[int] = CONFIRMATION_SEEDS,
) -> dict[str, Any]:
    """Search semantic seed fields in existing Genomics result evidence."""

    root = Path(repo_root).resolve()
    wanted = {int(seed) for seed in seeds}
    if wanted != set(CONFIRMATION_SEEDS):
        raise ValueError("confirmation seed audit is fixed to 3001..3020")
    candidates: list[Path] = []
    for directory_name in ("results", "artifacts", "assets/evaluation"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in {".json", ".jsonl", ".csv"}
                and _is_genomics_evidence_path(path.relative_to(root))
            ):
                candidates.append(path)
    candidates = sorted(set(candidates))
    hits: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seed_fields_seen = 0
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        suffix = path.suffix.lower()
        file_hits: list[dict[str, Any]] = []
        fields_seen = 0
        if suffix == ".json":
            payload = _strict_json_load(path)
            file_hits, fields_seen = _json_seed_hits(payload, wanted)
        elif suffix == ".jsonl":
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                row_hits, row_fields = _json_seed_hits(
                    payload, wanted, location=f"line[{line_number}]"
                )
                file_hits.extend(row_hits)
                fields_seen += row_fields
        else:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                seed_columns = [
                    field
                    for field in (reader.fieldnames or [])
                    if field.lower() in SEED_FIELD_NAMES
                    or field.lower().endswith("_seed")
                    or field.lower().endswith("_seeds")
                ]
                fields_seen += len(seed_columns)
                for row_number, row in enumerate(reader, start=2):
                    for column in seed_columns:
                        raw = str(row[column]).strip()
                        if raw and raw.lstrip("-").isdigit() and int(raw) in wanted:
                            file_hits.append(
                                {
                                    "location": f"row[{row_number}].{column}",
                                    "seed": int(raw),
                                }
                            )
        for hit in file_hits:
            hits.append({"path": relative, **hit})
        seed_fields_seen += fields_seen
        receipts.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
                "semantic_seed_fields_seen": int(fields_seen),
            }
        )
    return {
        "scope": "existing_genomics_result_evidence_semantic_seed_fields",
        "scope_limitation": (
            "absence in the supplied repository is not proof that no external or "
            "unrecorded outcome has ever been viewed"
        ),
        "requested_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "files_scanned": len(receipts),
        "semantic_seed_fields_seen": int(seed_fields_seen),
        "scanned_file_receipts_sha256": canonical_sha256(receipts),
        "hits": hits,
        "passed": not hits,
        "cross_domain_numeric_reuse_is_out_of_scope": True,
    }


def _validate_source_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping) or set(identity) != {
        "files",
        "file_roles",
        "source_manifest_sha256",
        "loaded_repo_python_files_sha256",
        "loaded_repo_python_files_exact_cover",
        "formal_genomics_runner_modified",
        "vp_randomized_or_robust_scientific_code_used",
    }:
        raise ValueError("confirmation execution-source schema mismatch")
    if identity["file_roles"] != SOURCE_FILE_ROLES:
        raise ValueError("confirmation execution-source roles are not frozen")
    files = identity["files"]
    if not isinstance(files, Mapping) or set(files) != set(SOURCE_FILE_ROLES):
        raise ValueError("confirmation execution-source files are incomplete")
    if any(not _is_sha256(value) for value in files.values()):
        raise ValueError("confirmation execution-source file hash is invalid")
    if (
        identity["formal_genomics_runner_modified"] is not False
        or identity["vp_randomized_or_robust_scientific_code_used"] is not False
        or identity["loaded_repo_python_files_exact_cover"] is not True
        or identity["loaded_repo_python_files_sha256"]
        != canonical_sha256(
            {"loaded_repo_python_files": sorted(SOURCE_FILE_ROLES)}
        )
        or identity["source_manifest_sha256"]
        != canonical_sha256(
            {"files": dict(files), "file_roles": SOURCE_FILE_ROLES}
        )
    ):
        raise ValueError("confirmation execution-source identity is invalid")


def _validate_prepared_input_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping) or set(identity) != PREPARED_INPUT_IDENTITY_FIELDS:
        raise ValueError("locked prepared-input identity schema mismatch")
    for field in (
        "chromosomes_sha256",
        "expected_external_source_manifest_sha256",
        "external_source_manifest_sha256",
        "genotype_npz_sha256",
        "genotypes_sha256",
        "population_labels_sha256",
        "sample_ids_sha256",
        "snp_ids_sha256",
    ):
        if not _is_sha256(identity[field]):
            raise ValueError(f"locked prepared-input {field} is invalid")
    if (
        identity["expected_external_source_manifest_sha256"]
        != identity["external_source_manifest_sha256"]
        or identity["external_source_manifest_anchor_verified"] is not True
        or identity["ld_pruned_input_validated_from_external_manifest"] is not True
        or identity["manifest_ld_pruned_receipts_verified"] is not True
        or identity["manifest_population_design"] != "five_superpopulation"
        or identity["manifest_input_scope"] != "legacy_chr20_22"
        or identity["orientation"] != "samples_by_snps"
    ):
        raise ValueError("locked prepared-input provenance boundary is invalid")
    for field in ("external_source_manifest_bytes", "genotype_npz_bytes", "samples", "snps"):
        if not _is_plain_int(identity[field]) or int(identity[field]) <= 0:
            raise ValueError(f"locked prepared-input {field} is invalid")
    if int(identity["samples"]) != EXPECTED_PREPARED_SAMPLES:
        raise ValueError("locked prepared input is not the fixed 1000-person cohort")


def build_confirmation_lock(
    development_package: Path,
    *,
    expected_development_analysis_sha256: str,
    expected_development_package_anchor_sha256: str,
    development_source_root: Path,
    seed_audit_repo_root: Path,
) -> dict[str, Any]:
    """Freeze one comparison after the grid, before confirmation outcomes."""

    audited_root = Path(seed_audit_repo_root).resolve()
    for path, role in (
        (Path(development_package).resolve(), "development package"),
        (Path(development_source_root).resolve(), "development source snapshot"),
    ):
        try:
            path.relative_to(audited_root)
        except ValueError as error:
            raise ValueError(f"{role} must be inside the audited canonical repository") from error
    canonical_runner = audited_root / "experiments/genomics_pca_loading_confirmation.py"
    if canonical_runner.resolve() != Path(__file__).resolve():
        raise ValueError("seed-audit repository does not contain this confirmation runner")

    validation = grid.validate_output_package(
        development_package,
        repo_root=development_source_root,
        expected_analysis_sha256=expected_development_analysis_sha256,
        expected_package_anchor_sha256=expected_development_package_anchor_sha256,
    )
    development = _strict_json_load(development_package / grid.RESULT_FILENAME)
    if tuple(development["config"]["seeds"]) != grid.DEFAULT_SEEDS:
        raise ValueError("lock requires the fixed 2025..2029 development grid")
    if set(development["config"]["component_grid"]) != {2, 3, 4}:
        raise ValueError("development component grid is incomplete")
    if development["config"]["panel_sizes"] != [200, 400, 800, None]:
        raise ValueError("development panel grid is not the fixed full grid")
    if set(CONFIRMATION_SEEDS) & set(development["config"]["seeds"]):
        raise ValueError("development and confirmation seeds overlap")
    summary_snapshot = [
        {
            "n_components": row["n_components"],
            "panel_label": row["panel_label"],
            "mean_balanced_accuracy": row["mean_balanced_accuracy"],
        }
        for row in development["grid_summaries"]
        if row["evaluation"] == "loading_panel"
    ]
    baseline_development = next(
        row
        for row in development["grid_summaries"]
        if row["evaluation"] == "loading_panel"
        and row["n_components"] == BASELINE["n_components"]
        and row["panel_label"] == str(BASELINE["panel_size"])
    )
    locked_development = next(
        row
        for row in development["grid_summaries"]
        if row["evaluation"] == "loading_panel"
        and row["n_components"] == LOCKED["n_components"]
        and row["panel_label"] == str(LOCKED["panel_size"])
    )
    seed_audit = audit_confirmation_seed_novelty(seed_audit_repo_root)
    if not seed_audit["passed"]:
        raise ValueError(f"confirmation seeds already appear in Genomics evidence: {seed_audit['hits'][:3]}")
    lock: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "locked_before_confirmation_execution",
        "development_evidence": {
            "analysis_sha256": expected_development_analysis_sha256,
            "package_anchor_sha256": expected_development_package_anchor_sha256,
            "source_closure_verified": validation["source_closure_verified"],
            "external_anchors_verified": validation["external_anchors_verified"],
            "development_seeds": list(development["config"]["seeds"]),
            "prepared_input_identity": dict(development["input_identity"]),
            "complete_grid_balanced_accuracy_snapshot": summary_snapshot,
            "baseline_development_mean_balanced_accuracy": baseline_development[
                "mean_balanced_accuracy"
            ],
            "locked_development_mean_balanced_accuracy": locked_development[
                "mean_balanced_accuracy"
            ],
        },
        "selection": {
            "baseline": dict(BASELINE),
            "locked_candidate": dict(LOCKED),
            "reason": SELECTION_REASON,
            "selection_is_post_hoc_to_development_grid": True,
            "confirmation_outcomes_seen_before_lock": False,
            "confirmation_split_rows_accessed_before_lock": False,
            "no_best_seed_or_confirmation_seed_selected": True,
            "pre_outcome_boundary": PRE_OUTCOME_BOUNDARY,
        },
        "confirmation_protocol": dict(FROZEN_CONFIRMATION_PROTOCOL),
        "claim_gates": dict(FROZEN_CLAIM_GATES),
        "method_scope": dict(FROZEN_METHOD_SCOPE),
        "seed_novelty_audit": seed_audit,
        "execution_source_identity": _source_identity(seed_audit_repo_root),
        "lock_builder_sha256": _file_sha256(Path(__file__).resolve()),
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    validate_confirmation_lock(lock)
    return lock


def validate_confirmation_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "status",
        "development_evidence",
        "selection",
        "confirmation_protocol",
        "claim_gates",
        "method_scope",
        "seed_novelty_audit",
        "execution_source_identity",
        "lock_builder_sha256",
        "lock_payload_sha256",
    }
    if set(lock) != required or lock["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError("confirmation lock schema mismatch")
    grid._reject_nonfinite(lock)
    if lock["status"] != "locked_before_confirmation_execution":
        raise ValueError("confirmation lock status is invalid")
    development = lock["development_evidence"]
    development_fields = {
        "analysis_sha256",
        "package_anchor_sha256",
        "source_closure_verified",
        "external_anchors_verified",
        "development_seeds",
        "prepared_input_identity",
        "complete_grid_balanced_accuracy_snapshot",
        "baseline_development_mean_balanced_accuracy",
        "locked_development_mean_balanced_accuracy",
    }
    if not isinstance(development, Mapping) or set(development) != development_fields:
        raise ValueError("development-evidence lock schema mismatch")
    if (
        not _is_sha256(development["analysis_sha256"])
        or not _is_sha256(development["package_anchor_sha256"])
        or development["source_closure_verified"] is not True
        or development["external_anchors_verified"] is not True
        or tuple(development["development_seeds"]) != grid.DEFAULT_SEEDS
    ):
        raise ValueError("development evidence was not externally/source verified")
    _validate_prepared_input_identity(development["prepared_input_identity"])
    snapshot = development["complete_grid_balanced_accuracy_snapshot"]
    expected_snapshot_keys = {
        (components, panel)
        for components in (2, 3, 4)
        for panel in ("200", "400", "800", "all")
    }
    snapshot_map: dict[tuple[int, str], float] = {}
    if not isinstance(snapshot, list) or len(snapshot) != len(expected_snapshot_keys):
        raise ValueError("development grid snapshot is incomplete")
    for row in snapshot:
        if not isinstance(row, Mapping) or set(row) != {
            "n_components",
            "panel_label",
            "mean_balanced_accuracy",
        }:
            raise ValueError("development grid snapshot schema mismatch")
        if not _is_plain_int(row["n_components"]):
            raise ValueError("development component count is invalid")
        key = (int(row["n_components"]), str(row["panel_label"]))
        value = row["mean_balanced_accuracy"]
        if key in snapshot_map or isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError("development grid snapshot value is invalid")
        snapshot_map[key] = float(value)
    if set(snapshot_map) != expected_snapshot_keys:
        raise ValueError("development grid snapshot does not exact-cover the frozen grid")
    if (
        development["baseline_development_mean_balanced_accuracy"]
        != snapshot_map[(BASELINE["n_components"], str(BASELINE["panel_size"]))]
        or development["locked_development_mean_balanced_accuracy"]
        != snapshot_map[(LOCKED["n_components"], str(LOCKED["panel_size"]))]
    ):
        raise ValueError("development baseline/locked snapshot mismatch")

    selection = lock["selection"]
    if not isinstance(selection, Mapping) or set(selection) != {
        "baseline",
        "locked_candidate",
        "reason",
        "selection_is_post_hoc_to_development_grid",
        "confirmation_outcomes_seen_before_lock",
        "confirmation_split_rows_accessed_before_lock",
        "no_best_seed_or_confirmation_seed_selected",
        "pre_outcome_boundary",
    } or (
        selection["baseline"] != BASELINE
        or selection["locked_candidate"] != LOCKED
        or selection["reason"] != SELECTION_REASON
        or selection["selection_is_post_hoc_to_development_grid"] is not True
        or selection["confirmation_outcomes_seen_before_lock"] is not False
        or selection["confirmation_split_rows_accessed_before_lock"] is not False
        or selection["no_best_seed_or_confirmation_seed_selected"] is not True
        or selection["pre_outcome_boundary"] != PRE_OUTCOME_BOUNDARY
    ):
        raise ValueError("selection lock boundary is invalid")
    protocol = lock["confirmation_protocol"]
    if protocol != FROZEN_CONFIRMATION_PROTOCOL:
        raise ValueError("confirmation protocol is not the frozen plan")
    if set(protocol["seeds"]) & set(lock["development_evidence"]["development_seeds"]):
        raise ValueError("development and confirmation seed sets overlap")
    audit = lock["seed_novelty_audit"]
    if not isinstance(audit, Mapping) or set(audit) != {
        "scope",
        "scope_limitation",
        "requested_confirmation_seeds",
        "files_scanned",
        "semantic_seed_fields_seen",
        "scanned_file_receipts_sha256",
        "hits",
        "passed",
        "cross_domain_numeric_reuse_is_out_of_scope",
    }:
        raise ValueError("confirmation seed novelty audit schema mismatch")
    if (
        audit["scope"] != "existing_genomics_result_evidence_semantic_seed_fields"
        or audit["scope_limitation"]
        != (
            "absence in the supplied repository is not proof that no external or "
            "unrecorded outcome has ever been viewed"
        )
        or audit["requested_confirmation_seeds"] != list(CONFIRMATION_SEEDS)
        or not _is_plain_int(audit["files_scanned"])
        or int(audit["files_scanned"]) < 0
        or not _is_plain_int(audit["semantic_seed_fields_seen"])
        or int(audit["semantic_seed_fields_seen"]) < 0
        or not _is_sha256(audit["scanned_file_receipts_sha256"])
        or audit["passed"] is not True
        or audit["hits"] != []
        or audit["cross_domain_numeric_reuse_is_out_of_scope"] is not True
    ):
        raise ValueError("confirmation seed novelty audit did not pass")
    if lock["claim_gates"] != FROZEN_CLAIM_GATES:
        raise ValueError("confirmation claim gates are not frozen")
    if lock["method_scope"] != FROZEN_METHOD_SCOPE:
        raise ValueError("confirmation method scope is invalid")
    _validate_source_identity(lock["execution_source_identity"])
    if (
        not _is_sha256(lock["lock_builder_sha256"])
        or lock["lock_builder_sha256"]
        != lock["execution_source_identity"]["files"][
            "experiments/genomics_pca_loading_confirmation.py"
        ]
    ):
        raise ValueError("lock builder hash is invalid")
    unsigned = dict(lock)
    observed = unsigned.pop("lock_payload_sha256")
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError("lock_payload_sha256 mismatch")
    return {"status": "valid", "lock_payload_sha256": observed}


def write_confirmation_lock(lock: Mapping[str, Any], output_path: Path) -> None:
    validate_confirmation_lock(lock)
    output_path = Path(output_path)
    if os.path.lexists(output_path):
        raise FileExistsError(f"lock path already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.parent / f".{output_path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_bytes(_strict_json_bytes(lock))
    temporary.rename(output_path)


def load_anchored_lock(path: Path, expected_file_sha256: str) -> dict[str, Any]:
    _regular_file(path, role="confirmation lock")
    if not _is_sha256(expected_file_sha256) or _file_sha256(path) != expected_file_sha256:
        raise ValueError("confirmation lock does not match the external file-hash anchor")
    lock = _strict_json_load(path)
    validate_confirmation_lock(lock)
    return lock


def _paired_bootstrap_interval(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(CONFIRMATION_SEEDS),) or not np.isfinite(array).all():
        raise ValueError("paired bootstrap requires one finite value per confirmation seed")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_RESAMPLES, array.size),
        endpoint=False,
    )
    resampled_means = np.mean(array[indices], axis=1)
    lower, upper = np.quantile(resampled_means, [0.025, 0.975])
    return {
        "mean_delta": float(np.mean(array)),
        "sample_std_delta": float(np.std(array, ddof=1)),
        "ci95_lower": float(lower),
        "ci95_upper": float(upper),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "unit": "paired_outer_split_seed",
        "method": "percentile_two_sided",
        "paired_values_sha256": array_sha256(array),
    }


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    audit_code = """
import json
import sys
from pathlib import Path
import genomics_pca_loading_confirmation
root = Path.cwd().resolve()
paths = set()
for module in tuple(sys.modules.values()):
    raw = getattr(module, '__file__', None)
    if not raw:
        continue
    path = Path(raw).resolve()
    if path.suffix != '.py':
        continue
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        continue
    if relative.startswith('src/rspca/') or relative in {
        'experiments/genomics_pca_loading_confirmation.py',
        'experiments/genomics_pca_loading_grid.py',
    }:
        paths.add(relative)
print(json.dumps(sorted(paths)))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / "src"), str(root / "experiments")]
    )
    completed = subprocess.run(
        [sys.executable, "-c", audit_code],
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"confirmation fresh-import source audit failed: {completed.stderr[-500:]}"
        )
    loaded_repo_python_files = set(json.loads(completed.stdout))
    if loaded_repo_python_files != set(SOURCE_FILE_ROLES):
        missing = sorted(set(SOURCE_FILE_ROLES) - loaded_repo_python_files)
        unexpected = sorted(loaded_repo_python_files - set(SOURCE_FILE_ROLES))
        raise ValueError(
            f"confirmation loaded-source exact cover mismatch; missing={missing}, "
            f"unexpected={unexpected}"
        )
    files: dict[str, str] = {}
    for relative in SOURCE_FILE_ROLES:
        path = root / relative
        _regular_file(path, role=relative)
        files[relative] = _file_sha256(path)
    return {
        "files": files,
        "file_roles": dict(SOURCE_FILE_ROLES),
        "source_manifest_sha256": canonical_sha256(
            {"files": files, "file_roles": SOURCE_FILE_ROLES}
        ),
        "loaded_repo_python_files_sha256": canonical_sha256(
            {"loaded_repo_python_files": sorted(loaded_repo_python_files)}
        ),
        "loaded_repo_python_files_exact_cover": True,
        "formal_genomics_runner_modified": False,
        "vp_randomized_or_robust_scientific_code_used": False,
    }


def _population_recall(record: Mapping[str, Any], population: str) -> float:
    return float(
        next(
            row["recall"]
            for row in record["per_population_metrics"]
            if row["population"] == population
        )
    )


def _coordinate_rows(
    *,
    seed: int,
    specification: Mapping[str, Any],
    selected_snp_count: int,
    test_indices: np.ndarray,
    sample_ids: np.ndarray,
    test_labels: np.ndarray,
    train_labels: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_index, sample_index in enumerate(test_indices.tolist()):
        rows.append(
            {
                "row_type": "held_out_sample",
                "seed": int(seed),
                "configuration": str(specification["name"]),
                "n_components": int(specification["n_components"]),
                "panel_size": int(specification["panel_size"]),
                "selected_snp_count": int(selected_snp_count),
                "sample_index": int(sample_index),
                "sample_id": str(sample_ids[sample_index]),
                "population": str(test_labels[local_index]),
                "truth": str(test_labels[local_index]),
                "prediction": str(predictions[local_index]),
                "correct": bool(predictions[local_index] == test_labels[local_index]),
                **{
                    f"component_{component + 1}": float(test_scores[local_index, component])
                    for component in range(4)
                },
            }
        )
    classes = np.unique(train_labels)
    centroids = np.stack(
        [train_scores[train_labels == population].mean(axis=0) for population in classes],
        axis=0,
    )
    for class_index, population in enumerate(classes.tolist()):
        rows.append(
            {
                "row_type": "train_centroid",
                "seed": int(seed),
                "configuration": str(specification["name"]),
                "n_components": int(specification["n_components"]),
                "panel_size": int(specification["panel_size"]),
                "selected_snp_count": int(selected_snp_count),
                "sample_index": None,
                "sample_id": None,
                "population": population,
                "truth": None,
                "prediction": None,
                "correct": None,
                **{
                    f"component_{component + 1}": float(centroids[class_index, component])
                    for component in range(4)
                },
            }
        )
    return rows


def run_locked_confirmation(
    data: grid.GenotypeInput,
    lock: Mapping[str, Any],
    *,
    lock_file_sha256: str,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute only the two configurations frozen in ``lock``."""

    validate_confirmation_lock(lock)
    grid._validate_input(data)
    if not _is_sha256(lock_file_sha256):
        raise ValueError("confirmation execution requires an anchored lock-file hash")
    expected_input_identity = dict(
        lock["development_evidence"]["prepared_input_identity"]
    )
    expected_input_identity["ld_pruned_input_bound_to_anchored_receipts"] = True
    if dict(input_identity) != expected_input_identity:
        raise ValueError("confirmation input does not match the locked prepared dataset")
    _validate_source_identity(source_identity)
    if dict(source_identity) != lock["execution_source_identity"]:
        raise ValueError("confirmation execution source does not match the frozen lock")
    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    actual_input_fields = {
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "orientation": "samples_by_snps",
    }
    if any(input_identity[field] != value for field, value in actual_input_fields.items()):
        raise ValueError("confirmation input identity disagrees with loaded arrays")
    print(
        json.dumps(
            {
                "event": "locked_pca_confirmation_start",
                "lock_file_sha256": lock_file_sha256,
                "lock_payload_sha256": lock["lock_payload_sha256"],
                "execution_source_manifest_sha256": source_identity[
                    "source_manifest_sha256"
                ],
                "genotype_npz_sha256": input_identity["genotype_npz_sha256"],
                "confirmation_seeds_sha256": array_sha256(
                    np.asarray(CONFIRMATION_SEEDS, dtype=np.int64)
                ),
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    records: list[dict[str, Any]] = []
    ranking_records: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []

    for seed in CONFIRMATION_SEEDS:
        print(
            json.dumps({"event": "pca_loading_confirmation_seed_start", "seed": seed}),
            file=sys.stderr,
            flush=True,
        )
        train, test = stratified_split(labels, 0.2, seed)
        preprocessor = fit_smartpca_preprocessor(raw[train])
        train_matrix = transform_smartpca(raw[train], preprocessor)
        test_matrix = transform_smartpca(raw[test], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        if retained_original.size < LOCKED["panel_size"]:
            raise ValueError("confirmation split has fewer than 400 retained SNPs")
        split_hashes = {
            "train_indices_sha256": array_sha256(train),
            "test_indices_sha256": array_sha256(test),
            "train_sample_ids_sha256": array_sha256(sample_ids[train]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test]),
            "test_labels_sha256": array_sha256(labels[test]),
        }
        split_receipts.append(
            {
                "seed": int(seed),
                **split_hashes,
                "retained_original_indices_sha256": array_sha256(retained_original),
                "retained_snp_ids_sha256": array_sha256(data.snp_ids[retained_original]),
                "retained_snps": int(retained_original.size),
                "preprocessor_mean_sha256": array_sha256(preprocessor.mean_dosage),
                "preprocessor_scale_sha256": array_sha256(preprocessor.scale),
            }
        )
        full_loadings, _ = grid.fit_top_pca(train_matrix, 4)

        for specification in (BASELINE, LOCKED):
            n_components = int(specification["n_components"])
            panel_size = int(specification["panel_size"])
            ranking_retained, importance = grid.rank_by_pca_loadings(
                full_loadings, n_components
            )
            ranked_original = retained_original[ranking_retained]
            selected_retained = np.sort(ranking_retained[:panel_size]).astype(np.int64)
            selected_original = np.sort(retained_original[selected_retained]).astype(np.int64)
            panel_train = train_matrix[:, selected_retained]
            panel_test = test_matrix[:, selected_retained]
            panel_loadings, _ = grid.fit_top_pca(panel_train, 4)
            train_scores = panel_train @ panel_loadings
            test_scores = panel_test @ panel_loadings
            record, predictions, _ = grid._record(
                seed=seed,
                split_hashes=split_hashes,
                evaluation="loading_panel",
                n_components=n_components,
                panel_size=panel_size,
                selected_original=selected_original,
                retained_original=retained_original,
                loadings=panel_loadings,
                train_scores=train_scores,
                test_scores=test_scores,
                train_labels=labels[train],
                test_labels=labels[test],
                reused_full_width_fit=False,
            )
            record["configuration"] = str(specification["name"])
            record["held_out_scores_4pc_sha256"] = array_sha256(test_scores)
            full_centroids = np.stack(
                [
                    train_scores[labels[train] == population].mean(axis=0)
                    for population in EXPECTED_POPULATIONS
                ],
                axis=0,
            )
            record["outer_train_centroids_4pc_sha256"] = array_sha256(
                full_centroids
            )
            records.append(record)
            ranking_records.append(
                {
                    "seed": int(seed),
                    "configuration": str(specification["name"]),
                    "n_components": n_components,
                    "panel_size": panel_size,
                    "available_train_retained_snps": int(retained_original.size),
                    "ordered_original_indices": ranked_original.tolist(),
                    "ordered_original_indices_sha256": array_sha256(ranked_original),
                    "loading_importance_sha256": array_sha256(importance),
                    "selected_original_indices_sha256": array_sha256(selected_original),
                    "selection_rule": "descending_sum_abs_outer_train_pca_loadings",
                    "tie_break": "retained_column_index_ascending",
                    "population_labels_used": False,
                    "held_out_rows_used": False,
                }
            )
            if seed == CONFIRMATION_SEEDS[0]:
                coordinates.extend(
                    _coordinate_rows(
                        seed=seed,
                        specification=specification,
                        selected_snp_count=selected_original.size,
                        test_indices=test,
                        sample_ids=sample_ids,
                        test_labels=labels[test],
                        train_labels=labels[train],
                        train_scores=train_scores,
                        test_scores=test_scores,
                        predictions=predictions,
                    )
                )
        print(
            json.dumps(
                {
                    "event": "pca_loading_confirmation_seed_complete",
                    "seed": seed,
                    "full_width_pca_fits": 1,
                    "panel_pca_refits": 2,
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    record_map = {
        (int(record["seed"]), str(record["configuration"])): record
        for record in records
    }
    paired_records: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        baseline = record_map[(seed, BASELINE["name"])]
        locked = record_map[(seed, LOCKED["name"])]
        paired_records.append(
            {
                "seed": int(seed),
                "baseline_accuracy": baseline["accuracy"],
                "locked_accuracy": locked["accuracy"],
                "paired_accuracy_delta": locked["accuracy"] - baseline["accuracy"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "locked_balanced_accuracy": locked["balanced_accuracy"],
                "paired_balanced_accuracy_delta": locked["balanced_accuracy"]
                - baseline["balanced_accuracy"],
                "baseline_macro_f1": baseline["macro_f1"],
                "locked_macro_f1": locked["macro_f1"],
                "paired_macro_f1_delta": locked["macro_f1"] - baseline["macro_f1"],
                **{
                    f"baseline_{population.lower()}_recall": _population_recall(
                        baseline, population
                    )
                    for population in EXPECTED_POPULATIONS
                },
                **{
                    f"locked_{population.lower()}_recall": _population_recall(
                        locked, population
                    )
                    for population in EXPECTED_POPULATIONS
                },
                **{
                    f"paired_{population.lower()}_recall_delta": _population_recall(
                        locked, population
                    )
                    - _population_recall(baseline, population)
                    for population in EXPECTED_POPULATIONS
                },
            }
        )

    summary_rows: list[dict[str, Any]] = []
    metric_specs = [
        ("accuracy", "baseline_accuracy", "locked_accuracy", "paired_accuracy_delta"),
        (
            "balanced_accuracy",
            "baseline_balanced_accuracy",
            "locked_balanced_accuracy",
            "paired_balanced_accuracy_delta",
        ),
        ("macro_f1", "baseline_macro_f1", "locked_macro_f1", "paired_macro_f1_delta"),
        *[
            (
                f"{population}_recall",
                f"baseline_{population.lower()}_recall",
                f"locked_{population.lower()}_recall",
                f"paired_{population.lower()}_recall_delta",
            )
            for population in EXPECTED_POPULATIONS
        ],
    ]
    for metric, baseline_field, locked_field, delta_field in metric_specs:
        deltas = [float(row[delta_field]) for row in paired_records]
        interval = _paired_bootstrap_interval(deltas)
        summary_rows.append(
            {
                "metric": metric,
                "baseline_mean": float(
                    np.mean([row[baseline_field] for row in paired_records])
                ),
                "locked_mean": float(
                    np.mean([row[locked_field] for row in paired_records])
                ),
                **interval,
            }
        )
    summary_map = {row["metric"]: row for row in summary_rows}
    ba_gate = summary_map["balanced_accuracy"]["ci95_lower"] > 0.0
    amr_gate = (
        summary_map["AMR_recall"]["ci95_lower"] >= 0.0
        and summary_map["AMR_recall"]["mean_delta"] > 0.0
    )
    sas_gate = (
        summary_map["SAS_recall"]["ci95_lower"] >= 0.0
        and summary_map["SAS_recall"]["mean_delta"] > 0.0
    )
    solver_gate = bool(
        len(records) == 2 * len(CONFIRMATION_SEEDS)
        and all(record["solver_converged"] is True for record in records)
    )
    gates = {
        "primary_balanced_accuracy_superiority": bool(ba_gate),
        "amr_recall_nondecline_and_improvement": bool(amr_gate),
        "sas_recall_nondecline_and_improvement": bool(sas_gate),
        "closed_form_pca_all_fits_qualified": bool(solver_gate),
        "overall_locked_confirmation_passed": bool(
            ba_gate and amr_gate and sas_gate and solver_gate
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": CONFIRMATION_SCHEMA_VERSION,
        "status": "locked_paired_confirmation_complete",
        "lock_identity": {
            "lock_file_sha256": lock_file_sha256,
            "lock_payload_sha256": lock["lock_payload_sha256"],
            "confirmation_outcomes_seen_before_lock": False,
        },
        "protocol": dict(lock["confirmation_protocol"]),
        "configurations": {
            "baseline": dict(BASELINE),
            "locked": dict(LOCKED),
        },
        "method_scope": dict(lock["method_scope"]),
        "input_identity": dict(input_identity),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "split_receipts": split_receipts,
        "ranking_records": ranking_records,
        "records": records,
        "paired_records": paired_records,
        "metric_summaries": summary_rows,
        "claim_gates": gates,
        "registered_seed_policy": {
            "seed": CONFIRMATION_SEEDS[0],
            "selection": "first_locked_confirmation_seed_never_best_seed",
        },
        "registered_coordinates": coordinates,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_confirmation_payload(payload, lock)
    return payload


def _validate_confusion_record(record: Mapping[str, Any]) -> tuple[int, ...]:
    raw = record.get("confusion_matrix_counts")
    if (
        not isinstance(raw, list)
        or len(raw) != 5
        or any(
            not isinstance(row, list)
            or len(row) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
                for value in row
            )
            for row in raw
        )
    ):
        raise ValueError("confirmation confusion counts must be non-negative integer 5x5")
    confusion = np.asarray(raw, dtype=np.int64)
    supports = tuple(int(value) for value in confusion.sum(axis=1))
    if any(value <= 0 for value in supports):
        raise ValueError("confirmation populations require positive held-out support")
    derived = grid._derived_metrics(record)
    observed = (
        float(record["accuracy"]),
        float(record["balanced_accuracy"]),
        float(record["macro_f1"]),
    )
    if any(
        isinstance(record[field], bool)
        or not isinstance(record[field], (int, float))
        or not 0.0 <= float(record[field]) <= 1.0
        for field in ("accuracy", "balanced_accuracy", "macro_f1")
    ):
        raise ValueError("confirmation aggregate metric is invalid")
    if derived != observed:
        raise ValueError("confirmation metrics disagree with confusion counts")
    populations = record.get("per_population_metrics")
    if not isinstance(populations, list) or [row.get("population") for row in populations] != list(EXPECTED_POPULATIONS):
        raise ValueError("confirmation population metrics are incomplete")
    for index, row in enumerate(populations):
        if not isinstance(row, Mapping) or set(row) != {
            "population",
            "support",
            "predicted_count",
            "true_positive",
            "precision",
            "recall",
            "f1",
        }:
            raise ValueError("confirmation population metric schema mismatch")
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        true_positive = int(confusion[index, index])
        recall = float(true_positive / support)
        precision = float(true_positive / predicted) if predicted else 0.0
        f1 = (
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )
        expected = {
            "support": support,
            "predicted_count": predicted,
            "true_positive": true_positive,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        }
        for field, expected_value in expected.items():
            if isinstance(expected_value, float):
                if (
                    isinstance(row.get(field), bool)
                    or not isinstance(row.get(field), (int, float))
                    or float(row[field]) != expected_value
                ):
                    raise ValueError("confirmation population metric mismatch")
            elif not _is_plain_int(row.get(field)) or row.get(field) != expected_value:
                raise ValueError("confirmation population count mismatch")
    return supports


def _confusion_metric_view(record: Mapping[str, Any]) -> dict[str, float]:
    confusion = np.asarray(record["confusion_matrix_counts"], dtype=np.int64)
    accuracy, balanced_accuracy, macro_f1 = grid._derived_metrics(record)
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        **{
            f"{population.lower()}_recall": float(recalls[index])
            for index, population in enumerate(EXPECTED_POPULATIONS)
        },
    }


def validate_confirmation_payload(
    payload: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    expected_lock_file_sha256: str | None = None,
) -> dict[str, Any]:
    validate_confirmation_lock(lock)
    required = {
        "schema_version",
        "status",
        "lock_identity",
        "protocol",
        "configurations",
        "method_scope",
        "input_identity",
        "source_identity",
        "runtime_versions",
        "split_receipts",
        "ranking_records",
        "records",
        "paired_records",
        "metric_summaries",
        "claim_gates",
        "registered_seed_policy",
        "registered_coordinates",
        "analysis_sha256",
    }
    if set(payload) != required or payload["schema_version"] != CONFIRMATION_SCHEMA_VERSION:
        raise ValueError("confirmation payload schema mismatch")
    grid._reject_nonfinite(payload)
    lock_identity = payload["lock_identity"]
    if not isinstance(lock_identity, Mapping) or set(lock_identity) != {
        "lock_file_sha256",
        "lock_payload_sha256",
        "confirmation_outcomes_seen_before_lock",
    }:
        raise ValueError("confirmation lock-identity schema mismatch")
    if (
        payload["status"] != "locked_paired_confirmation_complete"
        or payload["protocol"] != lock["confirmation_protocol"]
        or payload["configurations"]
        != {"baseline": BASELINE, "locked": LOCKED}
        or payload["method_scope"] != lock["method_scope"]
        or lock_identity["lock_payload_sha256"] != lock["lock_payload_sha256"]
        or lock_identity["confirmation_outcomes_seen_before_lock"] is not False
        or not _is_sha256(lock_identity["lock_file_sha256"])
    ):
        raise ValueError("confirmation payload violates the lock")
    if expected_lock_file_sha256 is not None:
        if (
            not _is_sha256(expected_lock_file_sha256)
            or lock_identity["lock_file_sha256"] != expected_lock_file_sha256
        ):
            raise ValueError("confirmation payload does not match the trusted lock-file anchor")

    expected_input_identity = dict(
        lock["development_evidence"]["prepared_input_identity"]
    )
    expected_input_identity["ld_pruned_input_bound_to_anchored_receipts"] = True
    if payload["input_identity"] != expected_input_identity:
        raise ValueError("confirmation payload is not bound to the locked prepared input")
    _validate_source_identity(payload["source_identity"])
    if payload["source_identity"] != lock["execution_source_identity"]:
        raise ValueError("confirmation payload source differs from the frozen execution source")
    runtime = payload["runtime_versions"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "python",
        "numpy",
        "scipy",
        "vp_rspca",
        "platform",
        "machine",
    } or any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("confirmation runtime identity schema mismatch")

    receipt_fields = {
        "seed",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "retained_original_indices_sha256",
        "retained_snp_ids_sha256",
        "retained_snps",
        "preprocessor_mean_sha256",
        "preprocessor_scale_sha256",
    }
    receipt_map: dict[int, Mapping[str, Any]] = {}
    for receipt in payload["split_receipts"]:
        if not isinstance(receipt, Mapping) or set(receipt) != receipt_fields:
            raise ValueError("confirmation split receipt schema mismatch")
        if not _is_plain_int(receipt["seed"]):
            raise ValueError("confirmation split receipt seed is invalid")
        seed = int(receipt["seed"])
        if seed in receipt_map:
            raise ValueError("duplicate confirmation split receipt")
        for field in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
            "retained_original_indices_sha256",
            "retained_snp_ids_sha256",
            "preprocessor_mean_sha256",
            "preprocessor_scale_sha256",
        ):
            if not _is_sha256(receipt.get(field)):
                raise ValueError(f"invalid confirmation split receipt {field}")
        if not _is_plain_int(receipt["retained_snps"]) or int(receipt["retained_snps"]) < LOCKED["panel_size"]:
            raise ValueError("confirmation split receipt has insufficient SNPs")
        receipt_map[seed] = receipt
    if set(receipt_map) != set(CONFIRMATION_SEEDS):
        raise ValueError("confirmation split receipts do not exact-cover seeds")

    expected_keys = {
        (seed, specification["name"])
        for seed in CONFIRMATION_SEEDS
        for specification in (BASELINE, LOCKED)
    }
    ranking_map: dict[tuple[int, str], Mapping[str, Any]] = {}
    for ranking in payload["ranking_records"]:
        ranking_fields = {
            "seed",
            "configuration",
            "n_components",
            "panel_size",
            "available_train_retained_snps",
            "ordered_original_indices",
            "ordered_original_indices_sha256",
            "loading_importance_sha256",
            "selected_original_indices_sha256",
            "selection_rule",
            "tie_break",
            "population_labels_used",
            "held_out_rows_used",
        }
        if not isinstance(ranking, Mapping) or set(ranking) != ranking_fields:
            raise ValueError("confirmation ranking schema mismatch")
        if not _is_plain_int(ranking["seed"]):
            raise ValueError("confirmation ranking seed is invalid")
        key = (int(ranking["seed"]), str(ranking["configuration"]))
        if key in ranking_map:
            raise ValueError("duplicate confirmation ranking")
        specification = BASELINE if key[1] == BASELINE["name"] else LOCKED if key[1] == LOCKED["name"] else None
        if specification is None:
            raise ValueError("unknown confirmation ranking configuration")
        raw_ordered = ranking["ordered_original_indices"]
        if not isinstance(raw_ordered, list) or any(
            not _is_plain_int(value) for value in raw_ordered
        ):
            raise ValueError("confirmation ranking indices are not exact integers")
        ordered = np.asarray(raw_ordered, dtype=np.int64)
        receipt = receipt_map[key[0]]
        selected = np.sort(ordered[: int(specification["panel_size"])]).astype(np.int64)
        if (
            ranking.get("n_components") != specification["n_components"]
            or ranking.get("panel_size") != specification["panel_size"]
            or ranking.get("available_train_retained_snps") != receipt["retained_snps"]
            or ordered.size != receipt["retained_snps"]
            or np.unique(ordered).size != ordered.size
            or np.any(ordered < 0)
            or array_sha256(ordered) != ranking.get("ordered_original_indices_sha256")
            or array_sha256(np.sort(ordered))
            != receipt["retained_original_indices_sha256"]
            or array_sha256(selected) != ranking.get("selected_original_indices_sha256")
            or ranking.get("population_labels_used") is not False
            or ranking.get("held_out_rows_used") is not False
            or ranking.get("selection_rule")
            != "descending_sum_abs_outer_train_pca_loadings"
            or ranking.get("tie_break") != "retained_column_index_ascending"
            or not _is_sha256(ranking.get("loading_importance_sha256"))
        ):
            raise ValueError("confirmation ranking is invalid")
        ranking_map[key] = ranking
    if set(ranking_map) != expected_keys:
        raise ValueError("confirmation rankings do not cover the locked comparison")

    record_map: dict[tuple[int, str], Mapping[str, Any]] = {}
    support_map: dict[int, tuple[int, ...]] = {}
    record_fields = {
        "seed",
        "configuration",
        "evaluation",
        "n_components",
        "panel_label",
        "requested_panel_size",
        "available_train_retained_snps",
        "selected_snp_count",
        "selected_original_indices_sha256",
        "selected_snp_set_is_loading_only",
        "test_used_for_preprocessing_pca_or_ranking",
        "population_labels_used_for_panel_ranking",
        "preprocessing_scope",
        "pca_fit_scope",
        "panel_refit_scope",
        "classifier_fit_scope",
        "solver",
        "solver_converged",
        "solver_n_iter",
        "full_width_fit_reused",
        "loadings_sha256",
        "test_predictions_sha256",
        "held_out_scores_4pc_sha256",
        "outer_train_centroids_4pc_sha256",
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "class_order",
        "confusion_matrix_counts",
        "per_population_metrics",
    }
    for record in payload["records"]:
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("confirmation record schema mismatch")
        if not _is_plain_int(record["seed"]):
            raise ValueError("confirmation record seed is invalid")
        key = (int(record["seed"]), str(record["configuration"]))
        if key in record_map or key not in expected_keys:
            raise ValueError("duplicate or unknown confirmation record")
        specification = BASELINE if key[1] == BASELINE["name"] else LOCKED
        receipt = receipt_map[key[0]]
        ranking = ranking_map[key]
        for field in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if record.get(field) != receipt[field]:
                raise ValueError("confirmation record disagrees with split receipt")
        if (
            record.get("n_components") != specification["n_components"]
            or record.get("evaluation") != "loading_panel"
            or record.get("panel_label") != str(specification["panel_size"])
            or record.get("requested_panel_size") != specification["panel_size"]
            or record.get("available_train_retained_snps") != receipt["retained_snps"]
            or record.get("selected_snp_count") != specification["panel_size"]
            or record.get("selected_original_indices_sha256")
            != ranking["selected_original_indices_sha256"]
            or record.get("selected_snp_set_is_loading_only") is not True
            or record.get("population_labels_used_for_panel_ranking") is not False
            or record.get("test_used_for_preprocessing_pca_or_ranking") is not False
            or record.get("preprocessing_scope") != "outer_train_only"
            or record.get("pca_fit_scope") != "outer_train_only"
            or record.get("panel_refit_scope") != "outer_train_only"
            or record.get("classifier_fit_scope")
            != "outer_train_centroids_after_panel_freeze"
            or record.get("solver") != "closed_form_pca_smaller_gram_eigh"
            or record.get("solver_converged") is not True
            or record.get("solver_n_iter") != 1
            or record.get("full_width_fit_reused") is not False
            or record.get("class_order") != list(EXPECTED_POPULATIONS)
            or not _is_sha256(record.get("loadings_sha256"))
            or not _is_sha256(record.get("test_predictions_sha256"))
            or not _is_sha256(record.get("held_out_scores_4pc_sha256"))
            or not _is_sha256(record.get("outer_train_centroids_4pc_sha256"))
        ):
            raise ValueError("confirmation record violates the locked method")
        supports = _validate_confusion_record(record)
        prior = support_map.setdefault(key[0], supports)
        if supports != prior:
            raise ValueError("paired confirmation supports disagree within seed")
        record_map[key] = record
    if set(record_map) != expected_keys:
        raise ValueError("confirmation records do not exact-cover the locked comparison")
    if (
        len(set(support_map.values())) != 1
        or sum(next(iter(support_map.values())))
        != int(round(float(payload["input_identity"]["samples"]) * 0.2))
    ):
        raise ValueError("confirmation held-out supports are inconsistent across seeds")

    if len(payload["paired_records"]) != len(CONFIRMATION_SEEDS):
        raise ValueError("paired confirmation rows are incomplete")
    paired_fields = {
        "seed",
        "baseline_accuracy",
        "locked_accuracy",
        "paired_accuracy_delta",
        "baseline_balanced_accuracy",
        "locked_balanced_accuracy",
        "paired_balanced_accuracy_delta",
        "baseline_macro_f1",
        "locked_macro_f1",
        "paired_macro_f1_delta",
        *{
            f"{prefix}_{population.lower()}_recall"
            for population in EXPECTED_POPULATIONS
            for prefix in ("baseline", "locked")
        },
        *{
            f"paired_{population.lower()}_recall_delta"
            for population in EXPECTED_POPULATIONS
        },
    }
    paired_map: dict[int, Mapping[str, Any]] = {}
    for row in payload["paired_records"]:
        if (
            not isinstance(row, Mapping)
            or set(row) != paired_fields
            or not _is_plain_int(row["seed"])
            or int(row["seed"]) in paired_map
        ):
            raise ValueError("paired confirmation row schema/seed is invalid")
        paired_map[int(row["seed"])] = row
    if set(paired_map) != set(CONFIRMATION_SEEDS):
        raise ValueError("paired confirmation seed coverage is invalid")
    expected_paired: list[dict[str, Any]] = []
    for seed in CONFIRMATION_SEEDS:
        baseline = record_map[(seed, BASELINE["name"])]
        locked = record_map[(seed, LOCKED["name"])]
        baseline_metrics = _confusion_metric_view(baseline)
        locked_metrics = _confusion_metric_view(locked)
        expected = {
            "seed": seed,
            "baseline_accuracy": baseline_metrics["accuracy"],
            "locked_accuracy": locked_metrics["accuracy"],
            "paired_accuracy_delta": locked_metrics["accuracy"]
            - baseline_metrics["accuracy"],
            "baseline_balanced_accuracy": baseline_metrics["balanced_accuracy"],
            "locked_balanced_accuracy": locked_metrics["balanced_accuracy"],
            "paired_balanced_accuracy_delta": locked_metrics["balanced_accuracy"]
            - baseline_metrics["balanced_accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "locked_macro_f1": locked_metrics["macro_f1"],
            "paired_macro_f1_delta": locked_metrics["macro_f1"]
            - baseline_metrics["macro_f1"],
            **{
                f"baseline_{population.lower()}_recall": baseline_metrics[
                    f"{population.lower()}_recall"
                ]
                for population in EXPECTED_POPULATIONS
            },
            **{
                f"locked_{population.lower()}_recall": locked_metrics[
                    f"{population.lower()}_recall"
                ]
                for population in EXPECTED_POPULATIONS
            },
            **{
                f"paired_{population.lower()}_recall_delta": locked_metrics[
                    f"{population.lower()}_recall"
                ]
                - baseline_metrics[f"{population.lower()}_recall"]
                for population in EXPECTED_POPULATIONS
            },
        }
        if paired_map[seed] != expected:
            raise ValueError("paired confirmation row is not record-derived")
        expected_paired.append(expected)

    metric_specs = [
        ("accuracy", "baseline_accuracy", "locked_accuracy", "paired_accuracy_delta"),
        (
            "balanced_accuracy",
            "baseline_balanced_accuracy",
            "locked_balanced_accuracy",
            "paired_balanced_accuracy_delta",
        ),
        ("macro_f1", "baseline_macro_f1", "locked_macro_f1", "paired_macro_f1_delta"),
        *[
            (
                f"{population}_recall",
                f"baseline_{population.lower()}_recall",
                f"locked_{population.lower()}_recall",
                f"paired_{population.lower()}_recall_delta",
            )
            for population in EXPECTED_POPULATIONS
        ],
    ]
    if not isinstance(payload["metric_summaries"], list) or len(
        payload["metric_summaries"]
    ) != len(metric_specs):
        raise ValueError("confirmation metric summaries are incomplete")
    summary_map: dict[str, Mapping[str, Any]] = {}
    for row in payload["metric_summaries"]:
        if (
            not isinstance(row, Mapping)
            or set(row)
            != {
                "metric",
                "baseline_mean",
                "locked_mean",
                "mean_delta",
                "sample_std_delta",
                "ci95_lower",
                "ci95_upper",
                "resamples",
                "seed",
                "unit",
                "method",
                "paired_values_sha256",
            }
            or not isinstance(row["metric"], str)
            or row["metric"] in summary_map
        ):
            raise ValueError("confirmation metric summary schema is invalid")
        summary_map[row["metric"]] = row
    if set(summary_map) != {spec[0] for spec in metric_specs}:
        raise ValueError("confirmation metric summaries are incomplete")
    expected_summary_map: dict[str, dict[str, Any]] = {}
    for metric, baseline_field, locked_field, delta_field in metric_specs:
        expected_interval = _paired_bootstrap_interval(
            [row[delta_field] for row in expected_paired]
        )
        expected_summary = {
            "metric": metric,
            "baseline_mean": float(
                np.mean([row[baseline_field] for row in expected_paired])
            ),
            "locked_mean": float(
                np.mean([row[locked_field] for row in expected_paired])
            ),
            **expected_interval,
        }
        expected_summary_map[metric] = expected_summary
        observed_summary = summary_map[metric]
        for field, expected_value in expected_summary.items():
            observed = observed_summary[field]
            if isinstance(expected_value, float):
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, (int, float))
                    or float(observed) != expected_value
                ):
                    raise ValueError("confirmation summary is not paired-record-derived")
            elif observed != expected_value:
                raise ValueError("confirmation summary is not paired-record-derived")
    expected_gates = {
        "primary_balanced_accuracy_superiority": bool(
            expected_summary_map["balanced_accuracy"]["ci95_lower"] > 0.0
        ),
        "amr_recall_nondecline_and_improvement": bool(
            expected_summary_map["AMR_recall"]["ci95_lower"] >= 0.0
            and expected_summary_map["AMR_recall"]["mean_delta"] > 0.0
        ),
        "sas_recall_nondecline_and_improvement": bool(
            expected_summary_map["SAS_recall"]["ci95_lower"] >= 0.0
            and expected_summary_map["SAS_recall"]["mean_delta"] > 0.0
        ),
        "closed_form_pca_all_fits_qualified": True,
    }
    expected_gates["overall_locked_confirmation_passed"] = all(
        expected_gates.values()
    )
    if payload["claim_gates"] != expected_gates:
        raise ValueError("confirmation claim gates are not metric-derived")

    policy = payload["registered_seed_policy"]
    if policy != {
        "seed": CONFIRMATION_SEEDS[0],
        "selection": "first_locked_confirmation_seed_never_best_seed",
    }:
        raise ValueError("confirmation registered seed policy is invalid")
    coordinates = payload["registered_coordinates"]
    coordinate_fields = {
        "row_type",
        "seed",
        "configuration",
        "n_components",
        "panel_size",
        "selected_snp_count",
        "sample_index",
        "sample_id",
        "population",
        "truth",
        "prediction",
        "correct",
        "component_1",
        "component_2",
        "component_3",
        "component_4",
    }
    if not isinstance(coordinates, list) or any(
        not isinstance(row, Mapping) or set(row) != coordinate_fields
        for row in coordinates
    ):
        raise ValueError("confirmation registered-coordinate schema mismatch")
    expected_coordinate_count = 2 * (
        sum(support_map[CONFIRMATION_SEEDS[0]]) + len(EXPECTED_POPULATIONS)
    )
    if len(coordinates) != expected_coordinate_count or {
        row["configuration"] for row in coordinates
    } != {BASELINE["name"], LOCKED["name"]} or {
        row["row_type"] for row in coordinates
    } != {"held_out_sample", "train_centroid"}:
        raise ValueError("confirmation registered-coordinate closure mismatch")
    paired_coordinate_identity: tuple[tuple[int, str, str], ...] | None = None
    for configuration in (BASELINE["name"], LOCKED["name"]):
        specification = BASELINE if configuration == BASELINE["name"] else LOCKED
        rows = [row for row in coordinates if row.get("configuration") == configuration]
        samples = [row for row in rows if row.get("row_type") == "held_out_sample"]
        centroids = [row for row in rows if row.get("row_type") == "train_centroid"]
        record = record_map[(CONFIRMATION_SEEDS[0], configuration)]
        if (
            len(samples) != sum(sum(row) for row in record["confusion_matrix_counts"])
            or len(centroids) != 5
            or {row["population"] for row in centroids} != set(EXPECTED_POPULATIONS)
            or any(row.get("seed") != CONFIRMATION_SEEDS[0] for row in rows)
            or any(
                row.get("n_components") != specification["n_components"]
                or row.get("panel_size") != specification["panel_size"]
                or row.get("selected_snp_count") != specification["panel_size"]
                for row in rows
            )
            or any(
                isinstance(row[f"component_{component}"], bool)
                or not isinstance(row[f"component_{component}"], (int, float))
                or not math.isfinite(float(row[f"component_{component}"]))
                for row in rows
                for component in range(1, 5)
            )
        ):
            raise ValueError("confirmation registered coordinates are incomplete")
        if any(
            not _is_plain_int(row["sample_index"])
            or int(row["sample_index"]) < 0
            or not isinstance(row["sample_id"], str)
            or not row["sample_id"]
            or row["truth"] not in EXPECTED_POPULATIONS
            or row["prediction"] not in EXPECTED_POPULATIONS
            or not isinstance(row["correct"], bool)
            for row in samples
        ):
            raise ValueError("confirmation held-out coordinate identity is invalid")
        if (
            len({int(row["sample_index"]) for row in samples}) != len(samples)
            or len({str(row["sample_id"]) for row in samples}) != len(samples)
        ):
            raise ValueError("confirmation held-out coordinates contain duplicates")
        if any(
            row["sample_index"] is not None
            or row["sample_id"] is not None
            or row["truth"] is not None
            or row["prediction"] is not None
            or row["correct"] is not None
            for row in centroids
        ) or len({row["population"] for row in centroids}) != len(centroids):
            raise ValueError("confirmation centroid coordinate identity is invalid")
        coordinate_identity = tuple(
            (int(row["sample_index"]), str(row["sample_id"]), str(row["truth"]))
            for row in samples
        )
        if paired_coordinate_identity is None:
            paired_coordinate_identity = coordinate_identity
        elif coordinate_identity != paired_coordinate_identity:
            raise ValueError("baseline/locked coordinates do not use the same held-out rows")
        receipt = receipt_map[CONFIRMATION_SEEDS[0]]
        if (
            array_sha256(np.asarray([row["sample_index"] for row in samples], dtype=np.int64))
            != receipt["test_indices_sha256"]
            or array_sha256(np.asarray([row["sample_id"] for row in samples], dtype=str))
            != receipt["test_sample_ids_sha256"]
            or array_sha256(np.asarray([row["truth"] for row in samples], dtype=str))
            != receipt["test_labels_sha256"]
            or array_sha256(np.asarray([row["prediction"] for row in samples], dtype=str))
            != record["test_predictions_sha256"]
            or array_sha256(
                np.asarray(
                    [
                        [row[f"component_{component}"] for component in range(1, 5)]
                        for row in samples
                    ],
                    dtype=float,
                )
            )
            != record["held_out_scores_4pc_sha256"]
            or array_sha256(
                np.asarray(
                    [
                        [
                            next(
                                row
                                for row in centroids
                                if row["population"] == population
                            )[f"component_{component}"]
                            for component in range(1, 5)
                        ]
                        for population in EXPECTED_POPULATIONS
                    ],
                    dtype=float,
                )
            )
            != record["outer_train_centroids_4pc_sha256"]
        ):
            raise ValueError("confirmation coordinates do not match split/prediction receipts")
        positions = {population: index for index, population in enumerate(EXPECTED_POPULATIONS)}
        confusion = np.zeros((5, 5), dtype=np.int64)
        for row in samples:
            if (
                row.get("truth") not in EXPECTED_POPULATIONS
                or row.get("prediction") not in EXPECTED_POPULATIONS
                or row.get("population") != row.get("truth")
                or row.get("correct") is not (row.get("truth") == row.get("prediction"))
            ):
                raise ValueError("confirmation coordinate label mismatch")
            confusion[
                positions[str(row["truth"])], positions[str(row["prediction"])]
            ] += 1
        if confusion.tolist() != record["confusion_matrix_counts"]:
            raise ValueError("confirmation coordinates disagree with record")

    unsigned = dict(payload)
    observed_digest = unsigned.pop("analysis_sha256")
    if not _is_sha256(observed_digest) or canonical_sha256(unsigned) != observed_digest:
        raise ValueError("confirmation analysis_sha256 mismatch")
    return {
        "status": "valid",
        "analysis_sha256": observed_digest,
        "records": len(record_map),
        "paired_seeds": len(paired_map),
        "overall_gate_passed": payload["claim_gates"][
            "overall_locked_confirmation_passed"
        ],
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: "" if row.get(field) is None else row.get(field)
                for field in fieldnames
            }
        )
    return output.getvalue().encode("utf-8")


def _paired_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "seed",
        "baseline_accuracy",
        "locked_accuracy",
        "paired_accuracy_delta",
        "baseline_balanced_accuracy",
        "locked_balanced_accuracy",
        "paired_balanced_accuracy_delta",
        "baseline_macro_f1",
        "locked_macro_f1",
        "paired_macro_f1_delta",
        *[
            f"{prefix}_{population.lower()}_recall"
            for population in EXPECTED_POPULATIONS
            for prefix in ("baseline", "locked")
        ],
        *[
            f"paired_{population.lower()}_recall_delta"
            for population in EXPECTED_POPULATIONS
        ],
    )
    return _csv_bytes(payload["paired_records"], fields)


def _summary_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "metric",
        "baseline_mean",
        "locked_mean",
        "mean_delta",
        "sample_std_delta",
        "ci95_lower",
        "ci95_upper",
        "resamples",
        "seed",
        "unit",
        "method",
        "paired_values_sha256",
    )
    return _csv_bytes(payload["metric_summaries"], fields)


def _coordinates_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    fields = (
        "row_type",
        "seed",
        "configuration",
        "n_components",
        "panel_size",
        "selected_snp_count",
        "sample_index",
        "sample_id",
        "population",
        "truth",
        "prediction",
        "correct",
        "component_1",
        "component_2",
        "component_3",
        "component_4",
    )
    return _csv_bytes(payload["registered_coordinates"], fields)


def _scatter_configuration(
    axis: Any,
    payload: Mapping[str, Any],
    configuration: str,
    *,
    x_component: int,
    y_component: int,
) -> None:
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in payload["registered_coordinates"]
        if row["configuration"] == configuration
        and row["row_type"] == "held_out_sample"
    ]
    centroids = [
        row
        for row in payload["registered_coordinates"]
        if row["configuration"] == configuration
        and row["row_type"] == "train_centroid"
    ]
    palette = plt.get_cmap("tab10")
    colors = {
        population: palette(index)
        for index, population in enumerate(EXPECTED_POPULATIONS)
    }
    for population in EXPECTED_POPULATIONS:
        population_rows = [row for row in rows if row["truth"] == population]
        correct = [row for row in population_rows if row["correct"]]
        incorrect = [row for row in population_rows if not row["correct"]]
        if correct:
            axis.scatter(
                [row[f"component_{x_component}"] for row in correct],
                [row[f"component_{y_component}"] for row in correct],
                color=colors[population],
                s=34,
                alpha=0.8,
                edgecolors="white",
                linewidths=0.4,
                label=population,
            )
        if incorrect:
            axis.scatter(
                [row[f"component_{x_component}"] for row in incorrect],
                [row[f"component_{y_component}"] for row in incorrect],
                color=colors[population],
                s=52,
                marker="x",
                linewidths=1.5,
            )
        centroid = next(row for row in centroids if row["population"] == population)
        axis.scatter(
            [centroid[f"component_{x_component}"]],
            [centroid[f"component_{y_component}"]],
            color=colors[population],
            s=125,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
        )
    axis.axhline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    axis.axvline(0.0, color="#d1d5db", linewidth=0.7, zorder=0)
    axis.grid(alpha=0.12)
    axis.set_xlabel(f"PC{x_component}")
    axis.set_ylabel(f"PC{y_component}")


def _render_pc12(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=160)
    summary = {row["metric"]: row for row in payload["metric_summaries"]}
    first_seed = CONFIRMATION_SEEDS[0]
    for axis, specification, label in (
        (axes[0], BASELINE, "Baseline: 2 PCs / top 200"),
        (axes[1], LOCKED, "Locked: 4 PCs / top 400"),
    ):
        _scatter_configuration(
            axis,
            payload,
            specification["name"],
            x_component=1,
            y_component=2,
        )
        record = next(
            row
            for row in payload["records"]
            if row["seed"] == first_seed
            and row["configuration"] == specification["name"]
        )
        mean_key = "baseline_mean" if specification is BASELINE else "locked_mean"
        axis.set_title(
            f"{label}\nseed {first_seed} BA={record['balanced_accuracy']:.3f}; "
            f"20-seed mean={summary['balanced_accuracy'][mean_key]:.3f}",
            fontsize=11,
        )
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.suptitle(
        "Locked PCA confirmation — held-out PC1/PC2, first confirmation seed (not selected)",
        fontsize=14,
        y=1.0,
    )
    figure.text(
        0.5,
        0.01,
        "Outer-training-only scaling, loading ranking, panel refit, and centroids. "
        "x = held-out error; stars = training centroids. Each panel has its own PCA axes.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.88), w_pad=2.0)
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "vp-rspca", "Creation Time": None},
    )
    plt.close(figure)


def _render_pc34(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(1, 1, figsize=(9.4, 6.8), dpi=160)
    _scatter_configuration(
        axis,
        payload,
        LOCKED["name"],
        x_component=3,
        y_component=4,
    )
    first_seed_record = next(
        row
        for row in payload["records"]
        if row["seed"] == CONFIRMATION_SEEDS[0]
        and row["configuration"] == LOCKED["name"]
    )
    summary = next(
        row for row in payload["metric_summaries"] if row["metric"] == "balanced_accuracy"
    )
    axis.set_title(
        "Locked 4-PC / top-400 panel — held-out PC3/PC4\n"
        f"seed {CONFIRMATION_SEEDS[0]} BA={first_seed_record['balanced_accuracy']:.3f}; "
        f"20-seed mean={summary['locked_mean']:.3f}",
        fontsize=12,
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
    )
    figure.text(
        0.5,
        0.01,
        "First locked confirmation seed 3001, never best-seed selected. "
        "x = held-out error; stars = training centroids.",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0.02, 0.055, 0.98, 0.87))
    figure.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "vp-rspca", "Creation Time": None},
    )
    plt.close(figure)


def write_confirmation_package(
    payload: Mapping[str, Any],
    lock: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    validate_confirmation_payload(payload, lock)
    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        raise FileExistsError(f"confirmation output already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_directory.parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / PAIRED_FILENAME).write_bytes(_paired_csv_bytes(payload))
        (temporary / SUMMARY_FILENAME).write_bytes(_summary_csv_bytes(payload))
        (temporary / COORDINATES_FILENAME).write_bytes(_coordinates_csv_bytes(payload))
        _render_pc12(payload, temporary / PC12_FILENAME)
        _render_pc34(payload, temporary / PC34_FILENAME)
        artifacts: dict[str, dict[str, Any]] = {}
        for filename in ARTIFACT_FILENAMES:
            path = temporary / filename
            status = _regular_file(path, role=filename)
            artifacts[filename] = {
                "bytes": int(status.st_size),
                "sha256": _file_sha256(path),
            }
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "manifest_written_last": True,
            "analysis_status": "locked_paired_confirmation",
            "analysis_sha256": payload["analysis_sha256"],
            "lock_file_sha256": payload["lock_identity"]["lock_file_sha256"],
            "lock_payload_sha256": payload["lock_identity"]["lock_payload_sha256"],
            "artifact_count_excluding_manifest": len(ARTIFACT_FILENAMES),
            "exact_package_file_count": len(ARTIFACT_FILENAMES) + 1,
            "artifacts": artifacts,
        }
        manifest["package_anchor_sha256"] = canonical_sha256(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "lock_file_sha256": manifest["lock_file_sha256"],
                "artifacts": artifacts,
            }
        )
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        temporary.rename(output_directory)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    return validate_confirmation_package(output_directory, lock=lock)


def validate_confirmation_package(
    output_directory: Path,
    *,
    lock: Mapping[str, Any],
    repo_root: Path | None = None,
    expected_lock_file_sha256: str | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("confirmation package must be a real directory")
    expected_files = set(ARTIFACT_FILENAMES) | {MANIFEST_FILENAME}
    actual_files = {entry.name for entry in output_directory.iterdir()}
    if actual_files != expected_files:
        raise ValueError("confirmation package file closure mismatch")
    manifest_path = output_directory / MANIFEST_FILENAME
    _regular_file(manifest_path, role="confirmation manifest")
    manifest = _strict_json_load(manifest_path)
    required_manifest = {
        "schema_version",
        "status",
        "manifest_written_last",
        "analysis_status",
        "analysis_sha256",
        "lock_file_sha256",
        "lock_payload_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if set(manifest) != required_manifest:
        raise ValueError("confirmation manifest schema mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["status"] != "complete"
        or manifest["manifest_written_last"] is not True
        or manifest["analysis_status"] != "locked_paired_confirmation"
        or manifest["artifact_count_excluding_manifest"] != len(ARTIFACT_FILENAMES)
        or manifest["exact_package_file_count"] != len(expected_files)
        or set(manifest["artifacts"]) != set(ARTIFACT_FILENAMES)
        or manifest["lock_payload_sha256"] != lock["lock_payload_sha256"]
        or not _is_sha256(manifest["lock_file_sha256"])
    ):
        raise ValueError("confirmation manifest policy/closure mismatch")
    unsigned = dict(manifest)
    manifest_digest = unsigned.pop("manifest_payload_sha256")
    if not _is_sha256(manifest_digest) or canonical_sha256(unsigned) != manifest_digest:
        raise ValueError("confirmation manifest payload hash mismatch")
    derived_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "lock_file_sha256": manifest["lock_file_sha256"],
            "artifacts": manifest["artifacts"],
        }
    )
    if manifest["package_anchor_sha256"] != derived_anchor:
        raise ValueError("confirmation package anchor mismatch")
    for filename in ARTIFACT_FILENAMES:
        path = output_directory / filename
        status = _regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if (
            set(receipt) != {"bytes", "sha256"}
            or receipt["bytes"] != status.st_size
            or receipt["sha256"] != _file_sha256(path)
        ):
            raise ValueError(f"confirmation artifact receipt mismatch: {filename}")
    payload = _strict_json_load(output_directory / RESULT_FILENAME)
    result_validation = validate_confirmation_payload(
        payload,
        lock,
        expected_lock_file_sha256=expected_lock_file_sha256,
    )
    if (
        payload["analysis_sha256"] != manifest["analysis_sha256"]
        or payload["lock_identity"]["lock_file_sha256"]
        != manifest["lock_file_sha256"]
    ):
        raise ValueError("confirmation result/manifest identity mismatch")
    if expected_lock_file_sha256 is not None and (
        not _is_sha256(expected_lock_file_sha256)
        or manifest["lock_file_sha256"] != expected_lock_file_sha256
    ):
        raise ValueError("confirmation manifest does not match the trusted lock-file anchor")
    if (expected_analysis_sha256 is None) != (
        expected_package_anchor_sha256 is None
    ):
        raise ValueError("confirmation external verification requires both anchors")
    if expected_analysis_sha256 is not None:
        if (
            not _is_sha256(expected_analysis_sha256)
            or not _is_sha256(expected_package_anchor_sha256)
        ):
            raise ValueError("confirmation external result anchors are invalid")
        if (
            payload["analysis_sha256"] != expected_analysis_sha256
            or manifest["package_anchor_sha256"]
            != expected_package_anchor_sha256
        ):
            raise ValueError("confirmation package does not match external anchors")
    if repo_root is not None:
        source_files = payload["source_identity"]["files"]
        root = Path(repo_root).resolve()
        for relative, expected_hash in source_files.items():
            path = root / str(relative)
            _regular_file(path, role=f"confirmation source {relative}")
            if _file_sha256(path) != expected_hash:
                raise ValueError(f"confirmation source hash mismatch: {relative}")
    if (output_directory / PAIRED_FILENAME).read_bytes() != _paired_csv_bytes(payload):
        raise ValueError("paired CSV is not result-derived")
    if (output_directory / SUMMARY_FILENAME).read_bytes() != _summary_csv_bytes(payload):
        raise ValueError("summary CSV is not result-derived")
    if (output_directory / COORDINATES_FILENAME).read_bytes() != _coordinates_csv_bytes(payload):
        raise ValueError("coordinate CSV is not result-derived")
    from PIL import Image

    for filename in (PC12_FILENAME, PC34_FILENAME):
        path = output_directory / filename
        with Image.open(path) as figure:
            figure.verify()
        with Image.open(path) as figure:
            if figure.format != "PNG" or figure.width < 1800 or figure.height < 850:
                raise ValueError(f"confirmation figure below clarity floor: {filename}")
    return {
        **result_validation,
        "artifact_integrity_verified": True,
        "lock_file_anchor_verified": expected_lock_file_sha256 is not None,
        "external_anchors_verified": expected_analysis_sha256 is not None,
        "source_closure_verified": repo_root is not None,
        "manifest_payload_sha256": manifest_digest,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "package_files": len(expected_files),
    }


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock")
    lock_parser.add_argument("--development-package", type=Path, required=True)
    lock_parser.add_argument("--expected-development-analysis-sha256", required=True)
    lock_parser.add_argument("--expected-development-package-anchor-sha256", required=True)
    lock_parser.add_argument("--development-source-root", type=Path, required=True)
    lock_parser.add_argument("--seed-audit-repo-root", type=Path, default=Path("."))
    lock_parser.add_argument("--output-lock", type=Path, required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--genotypes", type=Path, required=True)
    run_parser.add_argument("--source-manifest", type=Path, required=True)
    run_parser.add_argument("--expected-source-manifest-sha256", required=True)
    run_parser.add_argument("--lock", type=Path, required=True)
    run_parser.add_argument("--expected-lock-sha256", required=True)
    run_parser.add_argument("--repo-root", type=Path, default=Path("."))
    run_parser.add_argument("--output-dir", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("output_dir", type=Path)
    validate_parser.add_argument("--lock", type=Path, required=True)
    validate_parser.add_argument("--expected-lock-sha256", required=True)
    validate_parser.add_argument("--repo-root", type=Path)
    validate_parser.add_argument("--expected-analysis-sha256")
    validate_parser.add_argument("--expected-package-anchor-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args().parse_args(argv)
    if args.command == "lock":
        lock = build_confirmation_lock(
            args.development_package,
            expected_development_analysis_sha256=args.expected_development_analysis_sha256,
            expected_development_package_anchor_sha256=args.expected_development_package_anchor_sha256,
            development_source_root=args.development_source_root,
            seed_audit_repo_root=args.seed_audit_repo_root,
        )
        write_confirmation_lock(lock, args.output_lock)
        print(
            json.dumps(
                {
                    "lock_file": str(args.output_lock),
                    "lock_file_sha256": _file_sha256(args.output_lock),
                    "lock_payload_sha256": lock["lock_payload_sha256"],
                    "confirmation_seeds": list(CONFIRMATION_SEEDS),
                    "seed_novelty_audit_passed": lock["seed_novelty_audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    lock = load_anchored_lock(args.lock, args.expected_lock_sha256)
    if args.command == "validate":
        print(
            json.dumps(
                validate_confirmation_package(
                    args.output_dir,
                    lock=lock,
                    repo_root=args.repo_root,
                    expected_lock_file_sha256=args.expected_lock_sha256,
                    expected_analysis_sha256=args.expected_analysis_sha256,
                    expected_package_anchor_sha256=args.expected_package_anchor_sha256,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if os.path.lexists(args.output_dir):
        raise SystemExit(f"confirmation output already exists: {args.output_dir}")
    data = grid.load_genotype_npz(args.genotypes)
    preparation_receipt = grid.validate_external_source_manifest(
        args.source_manifest,
        args.genotypes,
        data,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    input_identity = {
        **preparation_receipt,
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "orientation": "samples_by_snps",
        "ld_pruned_input_validated_from_external_manifest": True,
        "ld_pruned_input_bound_to_anchored_receipts": True,
    }
    payload = run_locked_confirmation(
        data,
        lock,
        lock_file_sha256=args.expected_lock_sha256,
        input_identity=input_identity,
        source_identity=_source_identity(args.repo_root),
    )
    write_confirmation_package(payload, lock, args.output_dir)
    validation = validate_confirmation_package(
        args.output_dir,
        lock=lock,
        repo_root=args.repo_root,
        expected_lock_file_sha256=args.expected_lock_sha256,
    )
    print(
        json.dumps(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "package_anchor_sha256": validation["package_anchor_sha256"],
                "overall_gate_passed": payload["claim_gates"][
                    "overall_locked_confirmation_passed"
                ],
                "output_directory": str(args.output_dir),
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

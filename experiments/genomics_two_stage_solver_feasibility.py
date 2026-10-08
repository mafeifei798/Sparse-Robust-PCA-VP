#!/usr/bin/env python3
"""Two-stage PCA-screened solver feasibility on a fixed top-400 panel.

Stage one is always outer-training-only PCA: smartpca-like preprocessing,
rank-four full-width PCA-loading ranking, and one frozen top-400 panel.  Stage
two compares PCA, VP-SPCA, and randomized VP-SPCA on that exact same panel.
This is a controlled solver comparison, not method-specific AIM discovery.
Robust, Fed, and FedRPCA solvers are never called.
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
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

if __package__:
    from . import genomics_pca_loading_grid as grid
else:
    import genomics_pca_loading_grid as grid  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    stratified_split,
    transform_smartpca,
)
from rspca.randomized import (
    fit_randomized_vp_spca_from_workspace,
    prepare_randomized_vp_spca_workspace,
)
from rspca.vp import fit_vp_spca, procrustes_factor, vp_objective
from rspca.vp import proximal_gradient_mapping


SCHEMA_VERSION = 1
LOCK_SCHEMA_VERSION = 1
DECISION_LOCK_SCHEMA_VERSION = 1
DEVELOPMENT_SEEDS = tuple(range(2025, 2030))
CONFIRMATION_SEEDS = tuple(range(4001, 4021))
RANK = 4
PANEL_SIZE = 400
ALPHA = 0.02
BUDGETS = (500, 1000)
TOL = 1e-5
SUPPORT_THRESHOLD = 1e-8
RANDOMIZED_OVERSAMPLE = 30
RANDOMIZED_POWER_ITERATIONS = 4
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 420_420
SPARSE_WARMUP_REPETITIONS = 1
SPARSE_TIMED_REPETITIONS = 3
NONINFERIORITY_MEAN_MARGIN = 0.01
NONINFERIORITY_WORST_SPLIT_MARGIN = 0.05
MATCHED_OBJECTIVE_RELATIVE_EXCESS = 1e-4
MATCHED_SPEEDUP_MINIMUM = 1.05
MATCHED_FASTER_FRACTION = 0.8
EXPECTED_POPULATIONS = grid.EXPECTED_POPULATIONS

FEASIBILITY_STATUS = "two_stage_controlled_solver_feasibility_development"
CONFIRMATION_STATUS = "two_stage_controlled_solver_confirmation"
LOCK_FILENAME = "two_stage_solver_feasibility_lock.json"
DECISION_LOCK_FILENAME = "two_stage_solver_confirmation_lock.json"
RESULT_FILENAME = "two_stage_solver_results.json"
RECORDS_FILENAME = "two_stage_solver_records.csv"
SUMMARY_FILENAME = "two_stage_solver_summaries.csv"
QUALITY_FILENAME = "two_stage_solver_matched_quality.csv"
PREDICTIONS_FILENAME = "two_stage_solver_heldout_predictions.csv"
FIGURE_FILENAME = "two_stage_solver_overview.png"
MANIFEST_FILENAME = "two_stage_solver_output_manifest.json"
ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    RECORDS_FILENAME,
    SUMMARY_FILENAME,
    QUALITY_FILENAME,
    PREDICTIONS_FILENAME,
    FIGURE_FILENAME,
)

SOURCE_FILE_ROLES = {
    "experiments/genomics_two_stage_solver_feasibility.py": "controlled_solver_runner",
    "experiments/genomics_pca_loading_grid.py": "shared_train_only_pca_and_metrics",
    "src/rspca/genomics_aim.py": "historical_genomics_preprocessing_and_split",
    "src/rspca/datasets.py": "hashing_and_runtime_identity",
    "src/rspca/__init__.py": "eager_package_initialization",
    "src/rspca/accelerated.py": "eager_import_only_not_called",
    "src/rspca/cross_modal_gates.py": "eager_import_only_not_called",
    "src/rspca/federated.py": "eager_import_only_not_called",
    "src/rspca/fedrpca_audit.py": "eager_import_only_not_called",
    "src/rspca/gauge.py": "eager_import_only_not_called",
    "src/rspca/gauge_stress.py": "eager_import_only_not_called",
    "src/rspca/metrics.py": "eager_import_only_not_called",
    "src/rspca/product_aggregation.py": "eager_import_only_not_called",
    "src/rspca/randomized.py": "stage_two_randomized_solver",
    "src/rspca/robust.py": "eager_import_only_not_called",
    "src/rspca/statistical_gates.py": "eager_import_only_not_called",
    "src/rspca/trust_audit.py": "eager_import_only_not_called",
    "src/rspca/vp.py": "stage_two_vp_solver_and_full_objective",
}

FROZEN_PROTOCOL = {
    "stage_one": "outer_train_only_rank4_pca_loading_top400",
    "stage_two_methods": ["pca", "vp", "randomized"],
    "development_seeds": list(DEVELOPMENT_SEEDS),
    "confirmation_seeds_if_eligible": list(CONFIRMATION_SEEDS),
    "rank": RANK,
    "panel_size": PANEL_SIZE,
    "alpha": ALPHA,
    "total_proximal_step_budgets": list(BUDGETS),
    "tol": TOL,
    "support_threshold": SUPPORT_THRESHOLD,
    "randomized_oversample": RANDOMIZED_OVERSAMPLE,
    "randomized_power_iterations": RANDOMIZED_POWER_ITERATIONS,
    "randomized_budget_allocation": "half_sketch_half_full_refinement",
    "randomized_use_full_gradient": False,
    "randomized_sketch_seed_rule": "outer_seed_plus_730000_same_across_budgets",
    "randomized_workspace_preparation_charged_to_solver_runtime": True,
    "budget_semantics": "maximum_total_accepted_proximal_steps_sketch_plus_full",
    "vp_budget_allocation": "all_full_data_steps",
    "pca_evaluated_once_per_split": True,
    "budget_cells_are_cold_independent": True,
    "pca_timed_repetitions": 1,
    "sparse_warmup_repetitions_per_cell": SPARSE_WARMUP_REPETITIONS,
    "sparse_timed_repetitions_per_cell": SPARSE_TIMED_REPETITIONS,
    "method_order": "alternating_by_outer_seed_budget_and_timed_repetition",
    "solver_runtime_aggregation": "median_of_three_cold_timed_repetitions_after_one_excluded_warmup",
    "solver_only_runtime": "native_initialization_plus_optimization_plus_native_diagnostics",
    "end_to_end_runtime": "shared_train_only_screen_plus_solver_only_plus_common_diagnostic_and_scoring",
    "common_full_stationarity_step": "exact_panel_pca_inverse_squared_sigma_max",
    "heldout_truth_use": "scoring_and_predeclared_development_ba_gate_only",
    "population_labels_used_for_panel_ranking": False,
    "alpha_or_initialization_selected_by_heldout_truth": False,
    "analysis_role": "two_stage_controlled_solver_feasibility_not_aim_discovery",
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_scope": "split_randomization_conditional_on_fixed_cohort",
}

FROZEN_ELIGIBILITY_GATES = {
    "candidate": "randomized_budget_cell_with_same_budget_full_vp_reference",
    "vp_reference_all_splits_common_stationarity_qualified": True,
    "vp_reference_balanced_accuracy_noninferior_to_pca": True,
    "randomized_candidate_balanced_accuracy_noninferior_to_pca": True,
    "vp_and_randomized_non_degenerate_support": "rank4_and_each_component_has_at_least_one_loading_above_1e-8",
    "balanced_accuracy_mean_noninferiority_margin": NONINFERIORITY_MEAN_MARGIN,
    "balanced_accuracy_worst_split_margin": NONINFERIORITY_WORST_SPLIT_MARGIN,
    "randomized_vs_vp_full_objective_relative_excess_max": MATCHED_OBJECTIVE_RELATIVE_EXCESS,
    "vp_reference_full_stationarity_at_most": TOL,
    "randomized_common_stationarity_relative_to_vp_max_factor": 1.05,
    "randomized_common_stationarity_match_policy": "at_most_max_absolute_tol_or_1.05x_vp",
    "randomized_vs_vp_solver_speedup_median_min": MATCHED_SPEEDUP_MINIMUM,
    "randomized_vs_vp_faster_fraction_min": MATCHED_FASTER_FRACTION,
    "budget_selection": "lowest_total_budget_passing_all_gates",
    "confirmation_policy": "run_4001_4020_only_if_development_gate_passes",
    "claim_if_randomized_not_converged": "matched_fixed_budget_quality_not_faster_convergence",
}

_SEED_FIELD_NAMES = {
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
_PLANNED_SEED_FIELDS = {
    "confirmation_seeds_if_eligible",
    "confirmation_seeds",
    "requested_confirmation_seeds",
}


def _seed_values(value: Any) -> tuple[int, ...]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, int):
        return (int(value),)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lstrip("-").isdigit():
            return (int(stripped),)
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                decoded = json.loads(stripped)
            except ValueError:
                return ()
            return _seed_values(decoded)
        return ()
    if isinstance(value, list):
        return tuple(
            seed
            for item in value
            for seed in _seed_values(item)
        )
    return ()


def _strict_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _strict_json_load(path: Path) -> Any:
    return grid._strict_json_load(path)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    return grid._regular_file(path, role=role)


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    audit_code = """
import json
import sys
from pathlib import Path
import genomics_two_stage_solver_feasibility
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
        'experiments/genomics_two_stage_solver_feasibility.py',
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
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"fresh import audit failed: {completed.stderr[-500:]}")
    loaded = set(json.loads(completed.stdout))
    if loaded != set(SOURCE_FILE_ROLES):
        raise ValueError(
            "loaded source closure mismatch: "
            f"missing={sorted(set(SOURCE_FILE_ROLES) - loaded)}, "
            f"unexpected={sorted(loaded - set(SOURCE_FILE_ROLES))}"
        )
    files: dict[str, str] = {}
    for relative in SOURCE_FILE_ROLES:
        path = root / relative
        _regular_file(path, role=relative)
        files[relative] = _file_sha256(path)
    identity = {
        "files": files,
        "file_roles": dict(SOURCE_FILE_ROLES),
        "loaded_repo_python_files_sha256": canonical_sha256(
            {"loaded_repo_python_files": sorted(loaded)}
        ),
        "loaded_repo_python_files_exact_cover": True,
        "formal_genomics_runner_modified": False,
        "robust_fed_or_fedrpca_solver_called": False,
    }
    identity["source_manifest_sha256"] = canonical_sha256(identity)
    return identity


def _validate_source_identity(identity: Any) -> None:
    if not isinstance(identity, Mapping) or set(identity) != {
        "files",
        "file_roles",
        "loaded_repo_python_files_sha256",
        "loaded_repo_python_files_exact_cover",
        "formal_genomics_runner_modified",
        "robust_fed_or_fedrpca_solver_called",
        "source_manifest_sha256",
    }:
        raise ValueError("source identity schema mismatch")
    if identity["file_roles"] != SOURCE_FILE_ROLES:
        raise ValueError("source roles mismatch")
    if not isinstance(identity["files"], Mapping) or set(identity["files"]) != set(SOURCE_FILE_ROLES):
        raise ValueError("source files do not exact-cover execution")
    if any(not _is_sha256(value) for value in identity["files"].values()):
        raise ValueError("source file hash is invalid")
    unsigned = dict(identity)
    observed = unsigned.pop("source_manifest_sha256")
    if (
        canonical_sha256(unsigned) != observed
        or identity["loaded_repo_python_files_exact_cover"] is not True
        or identity["formal_genomics_runner_modified"] is not False
        or identity["robust_fed_or_fedrpca_solver_called"] is not False
    ):
        raise ValueError("source identity digest/scope mismatch")


def _semantic_seed_hits(
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
            is_seed_field = (
                normalized in _SEED_FIELD_NAMES
                or normalized.endswith("_seed")
                or normalized.endswith("_seeds")
            )
            if is_seed_field and normalized not in _PLANNED_SEED_FIELDS:
                fields_seen += 1
                for seed in _seed_values(item):
                    if seed in wanted:
                        hits.append({"location": child, "seed": int(seed)})
            child_hits, child_fields = _semantic_seed_hits(
                item,
                wanted,
                location=child,
            )
            hits.extend(child_hits)
            fields_seen += child_fields
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_hits, child_fields = _semantic_seed_hits(
                item,
                wanted,
                location=f"{location}[{index}]",
            )
            hits.extend(child_hits)
            fields_seen += child_fields
    return hits, fields_seen


def _is_genomics_evidence_path(relative: Path) -> bool:
    lowered = "/".join(relative.parts).lower()
    if "/source/" in f"/{lowered}/" or "/provenance/source_snapshot/" in f"/{lowered}/":
        return False
    return any(
        token in lowered
        for token in (
            "genom",
            "gene",
            "aim",
            "snp",
            "population",
            "two_stage_solver",
        )
    )


def audit_confirmation_seed_novelty(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    wanted = set(CONFIRMATION_SEEDS)
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
    hits: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    seed_fields_seen = 0
    for path in sorted(set(candidates)):
        relative = path.relative_to(root).as_posix()
        file_hits: list[dict[str, Any]] = []
        fields_seen = 0
        parsed = True
        try:
            if path.suffix.lower() == ".json":
                file_hits, fields_seen = _semantic_seed_hits(
                    _strict_json_load(path),
                    wanted,
                )
            elif path.suffix.lower() == ".jsonl":
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    if not line.strip():
                        continue
                    row_hits, row_fields = _semantic_seed_hits(
                        json.loads(line),
                        wanted,
                        location=f"line[{line_number}]",
                    )
                    file_hits.extend(row_hits)
                    fields_seen += row_fields
            else:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    seed_columns = [
                        field
                        for field in (reader.fieldnames or [])
                        if (
                            field.lower() in _SEED_FIELD_NAMES
                            or field.lower().endswith("_seed")
                            or field.lower().endswith("_seeds")
                        )
                        and field.lower() not in _PLANNED_SEED_FIELDS
                    ]
                    fields_seen += len(seed_columns)
                    for row_number, row in enumerate(reader, start=2):
                        for column in seed_columns:
                            raw = str(row[column]).strip()
                            for seed in _seed_values(raw):
                                if seed in wanted:
                                    file_hits.append(
                                        {
                                            "location": f"row[{row_number}].{column}",
                                            "seed": int(seed),
                                        }
                                    )
        except (ValueError, UnicodeError, csv.Error):
            parsed = False
            file_hits = []
            fields_seen = 0
        hits.extend({"path": relative, **hit} for hit in file_hits)
        seed_fields_seen += fields_seen
        receipts.append(
            {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": _file_sha256(path),
                "semantic_seed_fields_seen": int(fields_seen),
                "parsed_as_declared_format": parsed,
            }
        )
    return {
        "scope": "supplied_repository_genomics_and_two_stage_solver_structured_result_semantic_seed_fields",
        "scope_limitation": "absence_here_does_not_prove_no_external_or_unrecorded_outcome_was_viewed",
        "requested_confirmation_seeds": list(CONFIRMATION_SEEDS),
        "planned_seed_declaration_fields_ignored": sorted(_PLANNED_SEED_FIELDS),
        "files_scanned": len(receipts),
        "semantic_seed_fields_seen": int(seed_fields_seen),
        "scanned_file_receipts": receipts,
        "scanned_file_receipts_sha256": canonical_sha256(receipts),
        "hits": hits,
        "passed": not hits and all(
            receipt["parsed_as_declared_format"] for receipt in receipts
        ),
    }


def _validate_seed_novelty_audit(audit: Any) -> None:
    if not isinstance(audit, Mapping) or set(audit) != {
        "scope",
        "scope_limitation",
        "requested_confirmation_seeds",
        "planned_seed_declaration_fields_ignored",
        "files_scanned",
        "semantic_seed_fields_seen",
        "scanned_file_receipts",
        "scanned_file_receipts_sha256",
        "hits",
        "passed",
    }:
        raise ValueError("confirmation seed-novelty audit schema mismatch")
    receipts = audit["scanned_file_receipts"]
    if not isinstance(receipts, list) or any(
        not isinstance(receipt, Mapping) for receipt in receipts
    ):
        raise ValueError("confirmation seed-novelty audit receipts are malformed")
    if (
        audit["scope"]
        != "supplied_repository_genomics_and_two_stage_solver_structured_result_semantic_seed_fields"
        or audit["scope_limitation"]
        != "absence_here_does_not_prove_no_external_or_unrecorded_outcome_was_viewed"
        or audit["requested_confirmation_seeds"] != list(CONFIRMATION_SEEDS)
        or audit["planned_seed_declaration_fields_ignored"]
        != sorted(_PLANNED_SEED_FIELDS)
        or isinstance(audit["files_scanned"], bool)
        or not isinstance(audit["files_scanned"], int)
        or audit["files_scanned"] != len(receipts)
        or isinstance(audit["semantic_seed_fields_seen"], bool)
        or not isinstance(audit["semantic_seed_fields_seen"], int)
        or audit["semantic_seed_fields_seen"] < 0
        or audit["semantic_seed_fields_seen"]
        != sum(receipt.get("semantic_seed_fields_seen", -1) for receipt in receipts)
        or audit["scanned_file_receipts_sha256"] != canonical_sha256(receipts)
        or len({receipt.get("path") for receipt in receipts}) != len(receipts)
        or any(not receipt["parsed_as_declared_format"] for receipt in receipts)
        or any(
            set(receipt)
            != {
                "path",
                "bytes",
                "sha256",
                "semantic_seed_fields_seen",
                "parsed_as_declared_format",
            }
            or not _is_sha256(receipt["sha256"])
            or not isinstance(receipt["bytes"], int)
            or receipt["bytes"] < 0
            or not isinstance(receipt["path"], str)
            or not receipt["path"]
            or isinstance(receipt["semantic_seed_fields_seen"], bool)
            or not isinstance(receipt["semantic_seed_fields_seen"], int)
            or receipt["semantic_seed_fields_seen"] < 0
            or not isinstance(receipt["parsed_as_declared_format"], bool)
            for receipt in receipts
        )
        or audit["hits"] != []
        or audit["passed"] is not True
    ):
        raise ValueError("confirmation seed-novelty audit failed or is malformed")


def build_feasibility_lock(
    parent_confirmation_lock: Path,
    *,
    expected_parent_lock_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    _regular_file(parent_confirmation_lock, role="parent confirmation lock")
    if _file_sha256(parent_confirmation_lock) != expected_parent_lock_sha256:
        raise ValueError("parent confirmation lock external anchor mismatch")
    parent = _strict_json_load(parent_confirmation_lock)
    if __package__:
        from . import genomics_pca_loading_confirmation as confirmation
    else:
        import genomics_pca_loading_confirmation as confirmation  # type: ignore[no-redef]

    confirmation.validate_confirmation_lock(parent)
    parent_validator_relative = "experiments/genomics_pca_loading_confirmation.py"
    parent_validator_sha256 = _file_sha256(Path(repo_root) / parent_validator_relative)
    if (
        parent_validator_sha256
        != parent["execution_source_identity"]["files"][parent_validator_relative]
    ):
        raise ValueError("current parent-lock validator differs from parent frozen source")
    prepared = parent.get("development_evidence", {}).get("prepared_input_identity")
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("samples") != 1000
        or prepared.get("manifest_population_design") != "five_superpopulation"
        or prepared.get("orientation") != "samples_by_snps"
        or prepared.get("ld_pruned_input_validated_from_external_manifest") is not True
        or not _is_sha256(prepared.get("genotype_npz_sha256"))
        or not _is_sha256(prepared.get("external_source_manifest_sha256"))
        or parent.get("selection", {}).get("locked_candidate")
        != {"name": "locked_4pc_top400", "n_components": 4, "panel_size": 400}
    ):
        raise ValueError("parent lock does not bind the required prepared input/configuration")
    source = _source_identity(repo_root)
    novelty_audit = audit_confirmation_seed_novelty(repo_root)
    _validate_seed_novelty_audit(novelty_audit)
    lock: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "status": "locked_before_solver_feasibility_execution",
        "parent_confirmation_lock_file_sha256": expected_parent_lock_sha256,
        "parent_confirmation_lock_payload_sha256": parent["lock_payload_sha256"],
        "parent_confirmation_lock_validation": {
            "validator_file_sha256": parent_validator_sha256,
            "matches_parent_frozen_source": True,
            "full_schema_and_digest_validated": True,
        },
        "confirmation_seed_novelty_audit": novelty_audit,
        "prepared_input_identity": dict(prepared),
        "protocol": dict(FROZEN_PROTOCOL),
        "eligibility_gates": dict(FROZEN_ELIGIBILITY_GATES),
        "selection_boundary": {
            "pca_loading_configuration_already_locked": True,
            "alpha_anchor": ALPHA,
            "complete_budget_grid_reported": True,
            "solver_outcomes_seen_before_lock": False,
            "heldout_truth_used_for_preprocessing_panel_solver_or_centroids": False,
            "development_heldout_truth_used_for_ba_gate_and_budget_selection": True,
            "development_only": True,
            "method_specific_aim_discovery_claim": False,
        },
        "execution_source_identity": source,
        "lock_builder_sha256": source["files"][
            "experiments/genomics_two_stage_solver_feasibility.py"
        ],
    }
    lock["lock_payload_sha256"] = canonical_sha256(lock)
    validate_feasibility_lock(lock)
    return lock


def validate_feasibility_lock(lock: Mapping[str, Any]) -> None:
    if set(lock) != {
        "schema_version",
        "status",
        "parent_confirmation_lock_file_sha256",
        "parent_confirmation_lock_payload_sha256",
        "parent_confirmation_lock_validation",
        "confirmation_seed_novelty_audit",
        "prepared_input_identity",
        "protocol",
        "eligibility_gates",
        "selection_boundary",
        "execution_source_identity",
        "lock_builder_sha256",
        "lock_payload_sha256",
    } or lock["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ValueError("feasibility lock schema mismatch")
    grid._reject_nonfinite(lock)
    parent_validation = lock["parent_confirmation_lock_validation"]
    if (
        lock["status"] != "locked_before_solver_feasibility_execution"
        or not _is_sha256(lock["parent_confirmation_lock_file_sha256"])
        or not _is_sha256(lock["parent_confirmation_lock_payload_sha256"])
        or not isinstance(parent_validation, Mapping)
        or parent_validation
        != {
            "validator_file_sha256": parent_validation.get("validator_file_sha256"),
            "matches_parent_frozen_source": True,
            "full_schema_and_digest_validated": True,
        }
        or not _is_sha256(parent_validation.get("validator_file_sha256"))
        or lock["protocol"] != FROZEN_PROTOCOL
        or lock["eligibility_gates"] != FROZEN_ELIGIBILITY_GATES
        or lock["selection_boundary"]
        != {
            "pca_loading_configuration_already_locked": True,
            "alpha_anchor": ALPHA,
            "complete_budget_grid_reported": True,
            "solver_outcomes_seen_before_lock": False,
            "heldout_truth_used_for_preprocessing_panel_solver_or_centroids": False,
            "development_heldout_truth_used_for_ba_gate_and_budget_selection": True,
            "development_only": True,
            "method_specific_aim_discovery_claim": False,
        }
    ):
        raise ValueError("feasibility lock policy mismatch")
    _validate_seed_novelty_audit(lock["confirmation_seed_novelty_audit"])
    prepared = lock["prepared_input_identity"]
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("samples") != 1000
        or prepared.get("manifest_population_design") != "five_superpopulation"
        or prepared.get("ld_pruned_input_validated_from_external_manifest") is not True
        or not _is_sha256(prepared.get("genotype_npz_sha256"))
    ):
        raise ValueError("feasibility prepared-input lock mismatch")
    _validate_source_identity(lock["execution_source_identity"])
    if lock["lock_builder_sha256"] != lock["execution_source_identity"]["files"][
        "experiments/genomics_two_stage_solver_feasibility.py"
    ]:
        raise ValueError("lock builder/source mismatch")
    unsigned = dict(lock)
    observed = unsigned.pop("lock_payload_sha256")
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError("feasibility lock payload digest mismatch")


def _write_new_json(payload: Mapping[str, Any], path: Path) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_bytes(_strict_json_bytes(payload))
    temporary.rename(path)


def load_anchored_feasibility_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    _regular_file(path, role="feasibility lock")
    if not _is_sha256(expected_sha256) or _file_sha256(path) != expected_sha256:
        raise ValueError("feasibility lock external anchor mismatch")
    lock = _strict_json_load(path)
    validate_feasibility_lock(lock)
    return lock


def _configuration_name(method: str, budget: int | None) -> str:
    return "pca" if method == "pca" else f"{method}_budget_{int(budget)}"


def _derived_randomized_seed(outer_seed: int) -> int:
    return int(outer_seed) + 730_000


def _bound_input_identity(
    data: grid.GenotypeInput,
    preparation_receipt: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        **dict(preparation_receipt),
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
    expected = dict(lock["prepared_input_identity"])
    expected["ld_pruned_input_bound_to_anchored_receipts"] = True
    if identity != expected:
        mismatches = sorted(
            key for key in set(identity) | set(expected) if identity.get(key) != expected.get(key)
        )
        raise ValueError(f"prepared input differs from frozen identity: {mismatches}")
    return identity


def _common_full_diagnostics(
    matrix: np.ndarray,
    loadings: np.ndarray,
    step_size: float,
) -> tuple[np.ndarray, float, float]:
    factor = procrustes_factor(matrix, loadings)
    objective = vp_objective(matrix, factor, loadings, ALPHA)
    mapping = proximal_gradient_mapping(
        matrix,
        factor,
        loadings,
        alpha=ALPHA,
        step_size=step_size,
    )
    return factor, float(objective), float(np.linalg.norm(mapping, ord="fro"))


def _support_receipt(loadings: np.ndarray) -> dict[str, Any]:
    support = np.abs(np.asarray(loadings, dtype=float)) > SUPPORT_THRESHOLD
    by_component = support.sum(axis=0).astype(np.int64)
    count = int(support.sum())
    return {
        "support_threshold": SUPPORT_THRESHOLD,
        "support_count": count,
        "support_fraction": float(count / support.size),
        "support_count_by_component": by_component.tolist(),
        "loading_matrix_rank": int(np.linalg.matrix_rank(loadings)),
    }


def _prediction_rows(
    *,
    seed: int,
    method: str,
    budget: int | None,
    test_indices: np.ndarray,
    sample_ids: np.ndarray,
    truth: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    configuration = _configuration_name(method, budget)
    return [
        {
            "seed": int(seed),
            "method": method,
            "budget": None if budget is None else int(budget),
            "configuration": configuration,
            "sample_index": int(sample_index),
            "sample_id": str(sample_ids[sample_index]),
            "truth": str(truth[local_index]),
            "prediction": str(predictions[local_index]),
        }
        for local_index, sample_index in enumerate(test_indices.tolist())
    ]


def _finish_record(
    *,
    seed: int,
    method: str,
    budget: int | None,
    loadings: np.ndarray,
    panel_train: np.ndarray,
    panel_test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    test_indices: np.ndarray,
    sample_ids: np.ndarray,
    panel_original: np.ndarray,
    split_hashes: Mapping[str, str],
    shared_screen_seconds: float,
    common_reference_seconds: float,
    solver_only_seconds: float,
    solver_metadata: Mapping[str, Any],
    common_step_size: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation_start = perf_counter()
    factor, objective, stationarity = _common_full_diagnostics(
        panel_train, loadings, common_step_size
    )
    train_scores = panel_train @ loadings
    test_scores = panel_test @ loadings
    predictions, classes, centroids = grid._nearest_centroid(
        train_scores,
        train_labels,
        test_scores,
    )
    metrics = grid._classification_metrics(test_labels, predictions, classes)
    evaluation_seconds = perf_counter() - evaluation_start
    reference_charge = 0.0 if method == "pca" else common_reference_seconds
    runtime = {
        "shared_train_only_screen_seconds": float(shared_screen_seconds),
        "common_pca_reference_seconds": float(common_reference_seconds),
        "common_reference_charged_to_end_to_end_seconds": float(reference_charge),
        "solver_only_seconds": float(solver_only_seconds),
        "common_diagnostic_and_scoring_seconds": float(evaluation_seconds),
        "end_to_end_seconds": float(
            shared_screen_seconds
            + reference_charge
            + solver_only_seconds
            + evaluation_seconds
        ),
    }
    rows = _prediction_rows(
        seed=seed,
        method=method,
        budget=budget,
        test_indices=test_indices,
        sample_ids=sample_ids,
        truth=test_labels,
        predictions=predictions,
    )
    record = {
        "seed": int(seed),
        "method": method,
        "budget": None if budget is None else int(budget),
        "configuration": _configuration_name(method, budget),
        "rank": RANK,
        "alpha": ALPHA,
        "panel_size": PANEL_SIZE,
        "selected_snp_count": int(panel_original.size),
        "selected_original_indices_sha256": array_sha256(panel_original),
        "loading_matrix_sha256": array_sha256(loadings),
        "procrustes_factor_sha256": array_sha256(factor),
        "train_scores_sha256": array_sha256(train_scores),
        "test_scores_sha256": array_sha256(test_scores),
        "train_centroids_sha256": array_sha256(centroids),
        "test_predictions_sha256": array_sha256(predictions),
        "full_objective_common": objective,
        "full_stationarity_common": stationarity,
        "common_stationarity_step_size": float(common_step_size),
        "common_stationarity_step_policy": "exact_panel_pca_inverse_squared_sigma_max",
        "common_stationarity_qualified": None if method == "pca" else bool(stationarity <= TOL),
        "panel_selected_by_train_only_pca_loadings": True,
        "same_panel_for_all_methods_within_split": True,
        "heldout_truth_used_for_solver_or_panel": False,
        "classifier_scope": "outer_train_nearest_centroid_after_solver_freeze",
        "runtime_seconds": runtime,
        **_support_receipt(loadings),
        **dict(split_hashes),
        **dict(solver_metadata),
        **metrics,
    }
    return record, rows


def _paired_bootstrap(values: Sequence[float], *, salt: int) -> dict[str, float]:
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("bootstrap values must be a nonempty finite vector")
    if not np.any(samples):
        lower = upper = 0.0
    else:
        rng = np.random.default_rng(BOOTSTRAP_SEED + int(salt))
        indices = rng.integers(0, samples.size, size=(BOOTSTRAP_RESAMPLES, samples.size))
        means = samples[indices].mean(axis=1)
        lower, upper = (float(value) for value in np.quantile(means, [0.025, 0.975]))
    return {
        "mean_delta": float(samples.mean()),
        "ci95_lower": lower,
        "ci95_upper": upper,
    }


def _population_recall(record: Mapping[str, Any], population: str) -> float:
    return float(
        next(
            row["recall"]
            for row in record["per_population_metrics"]
            if row["population"] == population
        )
    )


def _configuration_order(budgets: Sequence[int]) -> list[tuple[str, int | None]]:
    return [("pca", None)] + [
        (method, int(budget))
        for budget in budgets
        for method in ("vp", "randomized")
    ]


def _summarize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    index = {
        (int(record["seed"]), str(record["method"]), record["budget"]): record
        for record in records
    }
    summaries: list[dict[str, Any]] = []
    for configuration_index, (method, budget) in enumerate(_configuration_order(budgets)):
        rows = [index[(int(seed), method, budget)] for seed in seeds]
        baselines = [index[(int(seed), "pca", None)] for seed in seeds]
        ba_delta = [
            float(row["balanced_accuracy"]) - float(base["balanced_accuracy"])
            for row, base in zip(rows, baselines, strict=True)
        ]
        f1_delta = [
            float(row["macro_f1"]) - float(base["macro_f1"])
            for row, base in zip(rows, baselines, strict=True)
        ]
        summaries.append(
            {
                "configuration": _configuration_name(method, budget),
                "method": method,
                "budget": budget,
                "seed_count": len(rows),
                "balanced_accuracy_mean": float(np.mean([row["balanced_accuracy"] for row in rows])),
                "balanced_accuracy_std": float(
                    np.std(
                        [row["balanced_accuracy"] for row in rows],
                        ddof=1 if len(rows) > 1 else 0,
                    )
                ),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in rows])),
                "macro_f1_std": float(
                    np.std(
                        [row["macro_f1"] for row in rows],
                        ddof=1 if len(rows) > 1 else 0,
                    )
                ),
                "balanced_accuracy_paired_vs_pca": _paired_bootstrap(
                    ba_delta, salt=100 + configuration_index
                ),
                "macro_f1_paired_vs_pca": _paired_bootstrap(
                    f1_delta, salt=200 + configuration_index
                ),
                "per_population_recall_mean": {
                    population: float(np.mean([_population_recall(row, population) for row in rows]))
                    for population in EXPECTED_POPULATIONS
                },
                "common_stationarity_qualified_count": (
                    None
                    if method == "pca"
                    else int(sum(bool(row["common_stationarity_qualified"]) for row in rows))
                ),
                "support_count_mean": float(np.mean([row["support_count"] for row in rows])),
                "support_fraction_mean": float(np.mean([row["support_fraction"] for row in rows])),
                "solver_only_seconds_median": float(
                    np.median([row["runtime_seconds"]["solver_only_seconds"] for row in rows])
                ),
                "end_to_end_seconds_median": float(
                    np.median([row["runtime_seconds"]["end_to_end_seconds"] for row in rows])
                ),
            }
        )
    return summaries


def _matched_quality_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    index = {
        (int(record["seed"]), str(record["method"]), record["budget"]): record
        for record in records
    }
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        for seed in seeds:
            vp = index[(int(seed), "vp", int(budget))]
            randomized = index[(int(seed), "randomized", int(budget))]
            vp_objective_value = float(vp["full_objective_common"])
            randomized_objective_value = float(randomized["full_objective_common"])
            excess = (randomized_objective_value - vp_objective_value) / max(
                1.0, abs(vp_objective_value)
            )
            vp_seconds = float(vp["runtime_seconds"]["solver_only_seconds"])
            randomized_seconds = float(
                randomized["runtime_seconds"]["solver_only_seconds"]
            )
            stationarity_limit = max(
                TOL,
                1.05 * float(vp["full_stationarity_common"]),
            )
            rows.append(
                {
                    "seed": int(seed),
                    "budget": int(budget),
                    "vp_full_objective_common": vp_objective_value,
                    "randomized_full_objective_common": randomized_objective_value,
                    "randomized_minus_vp_objective": float(
                        randomized_objective_value - vp_objective_value
                    ),
                    "randomized_relative_objective_excess": float(excess),
                    "vp_full_stationarity_common": float(vp["full_stationarity_common"]),
                    "randomized_full_stationarity_common": float(
                        randomized["full_stationarity_common"]
                    ),
                    "randomized_stationarity_limit": float(stationarity_limit),
                    "full_quality_matched": bool(
                        excess <= MATCHED_OBJECTIVE_RELATIVE_EXCESS
                        and float(randomized["full_stationarity_common"])
                        <= stationarity_limit
                    ),
                    "vp_solver_only_seconds": vp_seconds,
                    "randomized_solver_only_seconds": randomized_seconds,
                    "randomized_speedup_vs_vp": float(vp_seconds / randomized_seconds),
                    "randomized_faster": bool(randomized_seconds < vp_seconds),
                }
            )
    return rows


def _eligibility_gates(
    records: Sequence[Mapping[str, Any]],
    quality_rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    budgets: Sequence[int],
) -> dict[str, Any]:
    index = {
        (int(record["seed"]), str(record["method"]), record["budget"]): record
        for record in records
    }
    by_budget: list[dict[str, Any]] = []
    passing: list[int] = []
    for budget in budgets:
        pca = [index[(int(seed), "pca", None)] for seed in seeds]
        vp = [index[(int(seed), "vp", int(budget))] for seed in seeds]
        randomized = [index[(int(seed), "randomized", int(budget))] for seed in seeds]
        vp_delta = np.asarray(
            [float(row["balanced_accuracy"]) - float(base["balanced_accuracy"]) for row, base in zip(vp, pca, strict=True)]
        )
        randomized_delta = np.asarray(
            [float(row["balanced_accuracy"]) - float(base["balanced_accuracy"]) for row, base in zip(randomized, pca, strict=True)]
        )
        quality = [row for row in quality_rows if int(row["budget"]) == int(budget)]
        speedups = np.asarray([float(row["randomized_speedup_vs_vp"]) for row in quality])
        vp_all = all(bool(row["common_stationarity_qualified"]) for row in vp)
        randomized_all = all(bool(row["common_stationarity_qualified"]) for row in randomized)
        vp_noninferior = bool(
            vp_delta.mean() >= -NONINFERIORITY_MEAN_MARGIN
            and vp_delta.min() >= -NONINFERIORITY_WORST_SPLIT_MARGIN
        )
        randomized_noninferior = bool(
            randomized_delta.mean() >= -NONINFERIORITY_MEAN_MARGIN
            and randomized_delta.min() >= -NONINFERIORITY_WORST_SPLIT_MARGIN
        )
        support_non_degenerate = all(
            int(row["loading_matrix_rank"]) == RANK
            and all(int(value) > 0 for value in row["support_count_by_component"])
            for row in (*vp, *randomized)
        )
        quality_all = all(bool(row["full_quality_matched"]) for row in quality)
        median_speedup = float(np.median(speedups))
        faster_fraction = float(np.mean([bool(row["randomized_faster"]) for row in quality]))
        speed_gate = bool(
            median_speedup >= MATCHED_SPEEDUP_MINIMUM
            and faster_fraction >= MATCHED_FASTER_FRACTION
        )
        passed = bool(
            vp_all
            and vp_noninferior
            and randomized_noninferior
            and support_non_degenerate
            and quality_all
            and speed_gate
        )
        if passed:
            passing.append(int(budget))
        by_budget.append(
            {
                "budget": int(budget),
                "vp_all_splits_common_stationarity_qualified": vp_all,
                "randomized_all_splits_common_stationarity_qualified": randomized_all,
                "at_least_one_sparse_solver_all_splits_qualified": bool(vp_all or randomized_all),
                "vp_balanced_accuracy_mean_delta_vs_pca": float(vp_delta.mean()),
                "vp_balanced_accuracy_worst_delta_vs_pca": float(vp_delta.min()),
                "vp_balanced_accuracy_noninferior": vp_noninferior,
                "randomized_balanced_accuracy_mean_delta_vs_pca": float(randomized_delta.mean()),
                "randomized_balanced_accuracy_worst_delta_vs_pca": float(randomized_delta.min()),
                "randomized_balanced_accuracy_noninferior": randomized_noninferior,
                "vp_and_randomized_non_degenerate_support": support_non_degenerate,
                "randomized_full_quality_matched_all_splits": quality_all,
                "randomized_solver_speedup_median": median_speedup,
                "randomized_faster_fraction": faster_fraction,
                "randomized_solver_speed_gate_passed": speed_gate,
                "candidate_bound_gate_passed": passed,
            }
        )
    selected = min(passing) if passing else None
    return {
        "gate_scope": "development_budget_selection" if tuple(seeds) == DEVELOPMENT_SEEDS else "fresh_confirmation_evaluation",
        "by_budget": by_budget,
        "selected_budget": selected,
        "confirmation_permitted": bool(selected is not None),
        "selection_rule": "lowest_total_budget_passing_all_candidate_bound_gates",
        "no_cross_cell_advantage_composition": True,
    }


def _fit_sparse_solver_once(
    method: str,
    panel_train: np.ndarray,
    *,
    seed: int,
    budget: int,
) -> tuple[np.ndarray, dict[str, Any], float]:
    solver_start = perf_counter()
    if method == "vp":
        result = fit_vp_spca(
            panel_train,
            RANK,
            alpha=ALPHA,
            max_iter=int(budget),
            tol=TOL,
            backtracking=True,
            verbose=False,
        )
        solver_seconds = perf_counter() - solver_start
        metadata = {
            "native_status": str(result.status),
            "native_converged": bool(result.converged),
            "native_iterations": int(result.n_iter),
            "native_full_objective": float(result.objectives[-1]),
            "native_full_stationarity": float(result.stationarity[-1]),
            "sketch_iterations": None,
            "full_refinement_iterations": None,
            "randomized_seed": None,
            "randomized_workspace_seconds": None,
            "randomized_optimization_seconds": None,
            "sketch_residual_relative": None,
            "randomized_workspace_timed_repetitions_seconds": None,
            "randomized_optimization_timed_repetitions_seconds": None,
        }
        return np.asarray(result.B, dtype=float), metadata, float(solver_seconds)
    if method != "randomized":
        raise ValueError(f"unsupported sparse method: {method}")
    randomized_seed = _derived_randomized_seed(int(seed))
    workspace_start = perf_counter()
    workspace = prepare_randomized_vp_spca_workspace(
        panel_train,
        RANK,
        oversample=RANDOMIZED_OVERSAMPLE,
        n_power_iter=RANDOMIZED_POWER_ITERATIONS,
        random_state=randomized_seed,
    )
    workspace_seconds = perf_counter() - workspace_start
    sketch_budget = int(budget) // 2
    full_budget = int(budget) - sketch_budget
    result = fit_randomized_vp_spca_from_workspace(
        workspace,
        alpha=ALPHA,
        max_iter=sketch_budget,
        tol=TOL,
        use_full_gradient=False,
        finalize_full_iters=full_budget,
        verbose=False,
    )
    solver_seconds = perf_counter() - solver_start
    metadata = {
        "native_status": str(result.status),
        "native_converged": bool(result.full_converged),
        "native_iterations": int(result.n_iter),
        "native_full_objective": float(result.full_objective),
        "native_full_stationarity": float(result.full_stationarity),
        "sketch_iterations": int(result.sketch_iterations),
        "full_refinement_iterations": int(result.full_refinement_iterations),
        "randomized_seed": randomized_seed,
        "randomized_workspace_seconds": float(workspace_seconds),
        "randomized_optimization_seconds": float(result.timings["optimization"]),
        "sketch_residual_relative": float(result.sketch_residual_relative),
        "randomized_workspace_timed_repetitions_seconds": None,
        "randomized_optimization_timed_repetitions_seconds": None,
    }
    return np.asarray(result.B, dtype=float), metadata, float(solver_seconds)


def evaluate_solver_experiment(
    data: grid.GenotypeInput,
    *,
    phase: str,
    seeds: Sequence[int],
    budgets: Sequence[int],
    lock: Mapping[str, Any],
    lock_file_sha256: str,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    decision_lock_file_sha256: str | None = None,
) -> dict[str, Any]:
    if phase not in {"feasibility", "confirmation"}:
        raise ValueError("phase must be feasibility or confirmation")
    expected_seeds = DEVELOPMENT_SEEDS if phase == "feasibility" else CONFIRMATION_SEEDS
    if tuple(seeds) != expected_seeds:
        raise ValueError("seed schedule differs from frozen protocol")
    if phase == "feasibility" and tuple(budgets) != BUDGETS:
        raise ValueError("development must report the complete budget grid")
    if phase == "confirmation" and len(tuple(budgets)) != 1:
        raise ValueError("confirmation consumes exactly one locked budget")
    if dict(source_identity) != dict(lock["execution_source_identity"]):
        raise ValueError("current execution source differs from frozen lock")
    expected_input = dict(lock["prepared_input_identity"])
    expected_input["ld_pruned_input_bound_to_anchored_receipts"] = True
    if dict(input_identity) != expected_input:
        raise ValueError("execution input differs from frozen lock")

    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    records: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []
    ranking_receipts: list[dict[str, Any]] = []

    for seed in seeds:
        print(
            json.dumps({"event": "two_stage_solver_seed_start", "phase": phase, "seed": int(seed)}),
            file=sys.stderr,
            flush=True,
        )
        shared_start = perf_counter()
        train_indices, test_indices = stratified_split(labels, 0.2, int(seed))
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        preprocessor = fit_smartpca_preprocessor(raw[train_indices])
        train_matrix = transform_smartpca(raw[train_indices], preprocessor)
        test_matrix = transform_smartpca(raw[test_indices], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        if retained_original.size < PANEL_SIZE:
            raise ValueError("train-retained SNP count is below the locked panel size")
        full_loadings, _ = grid.fit_top_pca(train_matrix, RANK)
        ranked_retained, importance = grid.rank_by_pca_loadings(full_loadings, RANK)
        selected_retained = np.sort(ranked_retained[:PANEL_SIZE]).astype(np.int64)
        selected_original = np.sort(retained_original[selected_retained]).astype(np.int64)
        panel_train = np.ascontiguousarray(train_matrix[:, selected_retained])
        panel_test = np.ascontiguousarray(test_matrix[:, selected_retained])
        shared_screen_seconds = perf_counter() - shared_start
        split_hashes = {
            "train_indices_sha256": array_sha256(train_indices),
            "test_indices_sha256": array_sha256(test_indices),
            "train_sample_ids_sha256": array_sha256(sample_ids[train_indices]),
            "test_sample_ids_sha256": array_sha256(sample_ids[test_indices]),
            "test_labels_sha256": array_sha256(labels[test_indices]),
        }
        split_receipts.append(
            {
                "seed": int(seed),
                "train_indices": train_indices.tolist(),
                "test_indices": test_indices.tolist(),
                **split_hashes,
                "retained_original_indices_sha256": array_sha256(retained_original),
                "retained_snps": int(retained_original.size),
                "selected_original_indices": selected_original.tolist(),
                "selected_original_indices_sha256": array_sha256(selected_original),
                "panel_train_sha256": array_sha256(panel_train),
                "panel_test_sha256": array_sha256(panel_test),
                "preprocessor_mean_sha256": array_sha256(preprocessor.mean_dosage),
                "preprocessor_scale_sha256": array_sha256(preprocessor.scale),
            }
        )
        ranking_receipts.append(
            {
                "seed": int(seed),
                "rank": RANK,
                "panel_size": PANEL_SIZE,
                "ordered_original_indices_sha256": array_sha256(retained_original[ranked_retained]),
                "loading_importance_sha256": array_sha256(importance),
                "full_width_loadings_sha256": array_sha256(full_loadings),
                "selected_original_indices_sha256": array_sha256(selected_original),
                "selection_rule": "descending_sum_abs_training_rank4_pca_loadings",
                "tie_break": "retained_column_index_ascending",
                "population_labels_used": False,
                "test_rows_used": False,
            }
        )

        pca_start = perf_counter()
        pca_loadings, panel_singular_values = grid.fit_top_pca(panel_train, RANK)
        pca_seconds = perf_counter() - pca_start
        sigma_max = float(panel_singular_values[0])
        common_step = 1.0 if sigma_max == 0.0 else 1.0 / (sigma_max * sigma_max)
        pca_record, pca_rows = _finish_record(
            seed=int(seed),
            method="pca",
            budget=None,
            loadings=pca_loadings,
            panel_train=panel_train,
            panel_test=panel_test,
            train_labels=labels[train_indices],
            test_labels=labels[test_indices],
            test_indices=test_indices,
            sample_ids=sample_ids,
            panel_original=selected_original,
            split_hashes=split_hashes,
            shared_screen_seconds=shared_screen_seconds,
            common_reference_seconds=pca_seconds,
            solver_only_seconds=pca_seconds,
            solver_metadata={
                "native_status": "closed_form_pca",
                "native_converged": True,
                "native_iterations": 1,
                "native_full_objective": None,
                "native_full_stationarity": None,
                "sketch_iterations": None,
                "full_refinement_iterations": None,
                "randomized_seed": None,
                "randomized_workspace_seconds": None,
                "randomized_optimization_seconds": None,
                "sketch_residual_relative": None,
                "warmup_repetitions": 0,
                "timed_repetitions": 1,
                "solver_timed_repetitions_seconds": [float(pca_seconds)],
                "timing_position_by_repetition": [0],
                "deterministic_repetition_outputs_identical": True,
                "randomized_workspace_timed_repetitions_seconds": None,
                "randomized_optimization_timed_repetitions_seconds": None,
            },
            common_step_size=common_step,
        )
        records.append(pca_record)
        predictions.extend(pca_rows)

        for budget in budgets:
            base_order = (
                ("vp", "randomized")
                if (int(seed) + int(budget) // 500) % 2 == 0
                else ("randomized", "vp")
            )
            # One excluded warmup for each sparse cell.  It is deliberately
            # cold and uses the same immutable seed as the timed fits.
            for method in base_order:
                _fit_sparse_solver_once(
                    method,
                    panel_train,
                    seed=int(seed),
                    budget=int(budget),
                )
            timed: dict[str, list[tuple[np.ndarray, dict[str, Any], float, int]]] = {
                "vp": [],
                "randomized": [],
            }
            for repetition in range(SPARSE_TIMED_REPETITIONS):
                order = base_order if repetition % 2 == 0 else tuple(reversed(base_order))
                for position, method in enumerate(order):
                    loadings, metadata, seconds = _fit_sparse_solver_once(
                        method,
                        panel_train,
                        seed=int(seed),
                        budget=int(budget),
                    )
                    timed[method].append((loadings, metadata, seconds, position))
            for method in base_order:
                repetitions = timed[method]
                loadings = repetitions[0][0]
                primary_metadata = dict(repetitions[0][1])
                output_hashes = [array_sha256(repetition[0]) for repetition in repetitions]
                metadata_hashes = [
                    canonical_sha256(
                        {
                            key: value
                            for key, value in repetition[1].items()
                            if key
                            not in {
                                "randomized_workspace_seconds",
                                "randomized_optimization_seconds",
                                "randomized_workspace_timed_repetitions_seconds",
                                "randomized_optimization_timed_repetitions_seconds",
                            }
                        }
                    )
                    for repetition in repetitions
                ]
                outputs_identical = bool(
                    len(set(output_hashes)) == 1 and len(set(metadata_hashes)) == 1
                )
                if not outputs_identical:
                    raise RuntimeError("cold timed repetitions changed deterministic solver output")
                solver_repetitions = [float(repetition[2]) for repetition in repetitions]
                primary_metadata.update(
                    {
                        "warmup_repetitions": SPARSE_WARMUP_REPETITIONS,
                        "timed_repetitions": SPARSE_TIMED_REPETITIONS,
                        "solver_timed_repetitions_seconds": solver_repetitions,
                        "timing_position_by_repetition": [
                            int(repetition[3]) for repetition in repetitions
                        ],
                        "deterministic_repetition_outputs_identical": outputs_identical,
                    }
                )
                if method == "randomized":
                    workspace_repetitions = [
                        float(repetition[1]["randomized_workspace_seconds"])
                        for repetition in repetitions
                    ]
                    optimization_repetitions = [
                        float(repetition[1]["randomized_optimization_seconds"])
                        for repetition in repetitions
                    ]
                    primary_metadata["randomized_workspace_timed_repetitions_seconds"] = workspace_repetitions
                    primary_metadata["randomized_optimization_timed_repetitions_seconds"] = optimization_repetitions
                    primary_metadata["randomized_workspace_seconds"] = float(
                        np.median(workspace_repetitions)
                    )
                    primary_metadata["randomized_optimization_seconds"] = float(
                        np.median(optimization_repetitions)
                    )
                record, rows = _finish_record(
                    seed=int(seed),
                    method=method,
                    budget=int(budget),
                    loadings=loadings,
                    panel_train=panel_train,
                    panel_test=panel_test,
                    train_labels=labels[train_indices],
                    test_labels=labels[test_indices],
                    test_indices=test_indices,
                    sample_ids=sample_ids,
                    panel_original=selected_original,
                    split_hashes=split_hashes,
                    shared_screen_seconds=shared_screen_seconds,
                    common_reference_seconds=pca_seconds,
                    solver_only_seconds=float(np.median(solver_repetitions)),
                    solver_metadata=primary_metadata,
                    common_step_size=common_step,
                )
                records.append(record)
                predictions.extend(rows)
        print(
            json.dumps({"event": "two_stage_solver_seed_complete", "phase": phase, "seed": int(seed)}),
            file=sys.stderr,
            flush=True,
        )

    records.sort(key=lambda row: (int(row["seed"]), _configuration_order(budgets).index((row["method"], row["budget"]))))
    predictions.sort(
        key=lambda row: (
            int(row["seed"]),
            _configuration_order(budgets).index((row["method"], row["budget"])),
        )
    )
    summaries = _summarize_records(records, seeds=seeds, budgets=budgets)
    quality = _matched_quality_rows(records, seeds=seeds, budgets=budgets)
    gates = _eligibility_gates(records, quality, seeds=seeds, budgets=budgets)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": FEASIBILITY_STATUS if phase == "feasibility" else CONFIRMATION_STATUS,
        "phase": phase,
        "claim_boundary": {
            "pca_screened_fixed_panel_solver_comparison": True,
            "method_specific_aim_discovery": False,
            "robust_fed_or_fedrpca_called": False,
            "development_budget_selection": phase == "feasibility",
            "conditional_on_this_fixed_1000_person_cohort": True,
            "population_generalization_claim": False,
        },
        "active_lock_file_sha256": lock_file_sha256,
        "active_lock_payload_sha256": lock["lock_payload_sha256"],
        "decision_lock_file_sha256": decision_lock_file_sha256,
        "input_identity": dict(input_identity),
        "source_identity": dict(source_identity),
        "runtime_environment": {
            **runtime_versions(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        },
        "protocol": dict(FROZEN_PROTOCOL),
        "eligibility_gates_frozen": dict(FROZEN_ELIGIBILITY_GATES),
        "seeds": [int(seed) for seed in seeds],
        "budgets": [int(budget) for budget in budgets],
        "split_receipts": split_receipts,
        "ranking_receipts": ranking_receipts,
        "records": records,
        "heldout_predictions": predictions,
        "summaries": summaries,
        "matched_quality": quality,
        "gate_results": gates,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_solver_payload(payload, lock=lock, expected_lock_file_sha256=lock_file_sha256)
    return payload


_METRIC_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "class_order",
    "confusion_matrix_counts",
    "per_population_metrics",
}

_RECORD_KEYS = {
    "seed",
    "method",
    "budget",
    "configuration",
    "rank",
    "alpha",
    "panel_size",
    "selected_snp_count",
    "selected_original_indices_sha256",
    "loading_matrix_sha256",
    "procrustes_factor_sha256",
    "train_scores_sha256",
    "test_scores_sha256",
    "train_centroids_sha256",
    "test_predictions_sha256",
    "full_objective_common",
    "full_stationarity_common",
    "common_stationarity_step_size",
    "common_stationarity_step_policy",
    "common_stationarity_qualified",
    "panel_selected_by_train_only_pca_loadings",
    "same_panel_for_all_methods_within_split",
    "heldout_truth_used_for_solver_or_panel",
    "classifier_scope",
    "runtime_seconds",
    "support_threshold",
    "support_count",
    "support_fraction",
    "support_count_by_component",
    "loading_matrix_rank",
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "native_status",
    "native_converged",
    "native_iterations",
    "native_full_objective",
    "native_full_stationarity",
    "sketch_iterations",
    "full_refinement_iterations",
    "randomized_seed",
    "randomized_workspace_seconds",
    "randomized_optimization_seconds",
    "sketch_residual_relative",
    "warmup_repetitions",
    "timed_repetitions",
    "solver_timed_repetitions_seconds",
    "timing_position_by_repetition",
    "deterministic_repetition_outputs_identical",
    "randomized_workspace_timed_repetitions_seconds",
    "randomized_optimization_timed_repetitions_seconds",
    *_METRIC_KEYS,
}

_SPLIT_KEYS = {
    "seed",
    "train_indices",
    "test_indices",
    "train_indices_sha256",
    "test_indices_sha256",
    "train_sample_ids_sha256",
    "test_sample_ids_sha256",
    "test_labels_sha256",
    "retained_original_indices_sha256",
    "retained_snps",
    "selected_original_indices",
    "selected_original_indices_sha256",
    "panel_train_sha256",
    "panel_test_sha256",
    "preprocessor_mean_sha256",
    "preprocessor_scale_sha256",
}

_RANKING_KEYS = {
    "seed",
    "rank",
    "panel_size",
    "ordered_original_indices_sha256",
    "loading_importance_sha256",
    "full_width_loadings_sha256",
    "selected_original_indices_sha256",
    "selection_rule",
    "tie_break",
    "population_labels_used",
    "test_rows_used",
}


def _validate_metric_block(record: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    truth = np.asarray([row["truth"] for row in rows], dtype=str)
    predicted = np.asarray([row["prediction"] for row in rows], dtype=str)
    classes = np.asarray(EXPECTED_POPULATIONS, dtype=str)
    derived = grid._classification_metrics(truth, predicted, classes)
    observed = {key: record[key] for key in _METRIC_KEYS}
    if canonical_sha256(observed) != canonical_sha256(derived):
        raise ValueError("record metrics do not derive exactly from held-out predictions")
    confusion = record["confusion_matrix_counts"]
    if (
        not isinstance(confusion, list)
        or len(confusion) != 5
        or any(not isinstance(row, list) or len(row) != 5 for row in confusion)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for row in confusion
            for value in row
        )
        or any(sum(row) <= 0 for row in confusion)
    ):
        raise ValueError("confusion matrix must be exact nonnegative 5x5 counts")


def validate_solver_payload(
    payload: Mapping[str, Any],
    *,
    lock: Mapping[str, Any],
    expected_lock_file_sha256: str,
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "phase",
        "claim_boundary",
        "active_lock_file_sha256",
        "active_lock_payload_sha256",
        "decision_lock_file_sha256",
        "input_identity",
        "source_identity",
        "runtime_environment",
        "protocol",
        "eligibility_gates_frozen",
        "seeds",
        "budgets",
        "split_receipts",
        "ranking_receipts",
        "records",
        "heldout_predictions",
        "summaries",
        "matched_quality",
        "gate_results",
        "analysis_sha256",
    }
    if set(payload) != expected_top or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("solver result schema mismatch")
    grid._reject_nonfinite(payload)
    phase = payload["phase"]
    if phase not in {"feasibility", "confirmation"}:
        raise ValueError("invalid solver experiment phase")
    expected_status = FEASIBILITY_STATUS if phase == "feasibility" else CONFIRMATION_STATUS
    seeds = DEVELOPMENT_SEEDS if phase == "feasibility" else CONFIRMATION_SEEDS
    budgets = tuple(payload["budgets"])
    if (
        payload["status"] != expected_status
        or payload["seeds"] != list(seeds)
        or (phase == "feasibility" and budgets != BUDGETS)
        or (phase == "confirmation" and len(budgets) != 1)
        or (
            phase == "confirmation"
            and budgets != (int(lock["selected_budget"]),)
        )
        or any(budget not in BUDGETS for budget in budgets)
        or payload["protocol"] != FROZEN_PROTOCOL
        or payload["eligibility_gates_frozen"] != FROZEN_ELIGIBILITY_GATES
        or payload["active_lock_file_sha256"] != expected_lock_file_sha256
        or payload["active_lock_payload_sha256"] != lock["lock_payload_sha256"]
        or (
            phase == "feasibility"
            and payload["decision_lock_file_sha256"] is not None
        )
        or (
            phase == "confirmation"
            and payload["decision_lock_file_sha256"] != expected_lock_file_sha256
        )
    ):
        raise ValueError("solver payload does not match the active frozen lock")
    if payload["claim_boundary"] != {
        "pca_screened_fixed_panel_solver_comparison": True,
        "method_specific_aim_discovery": False,
        "robust_fed_or_fedrpca_called": False,
        "development_budget_selection": phase == "feasibility",
        "conditional_on_this_fixed_1000_person_cohort": True,
        "population_generalization_claim": False,
    }:
        raise ValueError("solver claim boundary mismatch")
    expected_input = dict(lock["prepared_input_identity"])
    expected_input["ld_pruned_input_bound_to_anchored_receipts"] = True
    if payload["input_identity"] != expected_input:
        raise ValueError("solver payload input identity mismatch")
    _validate_source_identity(payload["source_identity"])
    if payload["source_identity"] != lock["execution_source_identity"]:
        raise ValueError("solver payload source differs from active lock")
    thread_fields = (
        "openblas_num_threads",
        "omp_num_threads",
        "mkl_num_threads",
        "veclib_maximum_threads",
    )
    if any(payload["runtime_environment"].get(field) != "1" for field in thread_fields):
        raise ValueError("solver runtime requires four explicit single-thread controls")

    splits = payload["split_receipts"]
    rankings = payload["ranking_receipts"]
    if (
        not isinstance(splits, list)
        or len(splits) != len(seeds)
        or {row.get("seed") for row in splits} != set(seeds)
        or not isinstance(rankings, list)
        or len(rankings) != len(seeds)
        or {row.get("seed") for row in rankings} != set(seeds)
    ):
        raise ValueError("split/ranking receipts do not exact-cover seeds")
    split_map: dict[int, Mapping[str, Any]] = {}
    ranking_map: dict[int, Mapping[str, Any]] = {}
    for receipt in splits:
        if set(receipt) != _SPLIT_KEYS:
            raise ValueError("split receipt schema mismatch")
        seed = int(receipt["seed"])
        train = np.asarray(receipt["train_indices"])
        test = np.asarray(receipt["test_indices"])
        selected = np.asarray(receipt["selected_original_indices"])
        all_sample_indices = np.concatenate([train, test])
        input_samples = int(payload["input_identity"]["samples"])
        input_snps = int(payload["input_identity"]["snps"])
        if (
            train.dtype.kind not in "iu"
            or test.dtype.kind not in "iu"
            or selected.dtype.kind not in "iu"
            or train.shape != (800,)
            or test.shape != (200,)
            or selected.shape != (PANEL_SIZE,)
            or np.unique(all_sample_indices).size != input_samples
            or not np.array_equal(
                np.sort(all_sample_indices.astype(np.int64)),
                np.arange(input_samples, dtype=np.int64),
            )
            or set(train.tolist()) & set(test.tolist())
            or np.any(train < 0)
            or np.any(test < 0)
            or np.any(selected < 0)
            or np.unique(selected).size != PANEL_SIZE
            or not np.all(selected[1:] > selected[:-1])
            or int(selected[-1]) >= input_snps
            or not PANEL_SIZE <= int(receipt["retained_snps"]) <= input_snps
            or array_sha256(train.astype(np.int64)) != receipt["train_indices_sha256"]
            or array_sha256(test.astype(np.int64)) != receipt["test_indices_sha256"]
            or array_sha256(selected.astype(np.int64))
            != receipt["selected_original_indices_sha256"]
        ):
            raise ValueError("split receipt index/hash closure failed")
        for key, value in receipt.items():
            if key.endswith("sha256") and not _is_sha256(value):
                raise ValueError("split receipt contains an invalid hash")
        split_map[seed] = receipt
    for ranking in rankings:
        if set(ranking) != _RANKING_KEYS:
            raise ValueError("ranking receipt schema mismatch")
        seed = int(ranking["seed"])
        if (
            ranking["rank"] != RANK
            or ranking["panel_size"] != PANEL_SIZE
            or ranking["selected_original_indices_sha256"]
            != split_map[seed]["selected_original_indices_sha256"]
            or ranking["selection_rule"]
            != "descending_sum_abs_training_rank4_pca_loadings"
            or ranking["tie_break"] != "retained_column_index_ascending"
            or ranking["population_labels_used"] is not False
            or ranking["test_rows_used"] is not False
            or any(
                not _is_sha256(ranking[field])
                for field in (
                    "ordered_original_indices_sha256",
                    "loading_importance_sha256",
                    "full_width_loadings_sha256",
                    "selected_original_indices_sha256",
                )
            )
        ):
            raise ValueError("ranking receipt policy/hash mismatch")
        ranking_map[seed] = ranking

    expected_configurations = set(_configuration_order(budgets))
    records = payload["records"]
    expected_record_count = len(seeds) * len(expected_configurations)
    if not isinstance(records, list) or len(records) != expected_record_count:
        raise ValueError("solver records do not exact-cover seeds/configurations")
    record_map: dict[tuple[int, str, int | None], Mapping[str, Any]] = {}
    for record in records:
        if set(record) != _RECORD_KEYS:
            raise ValueError("solver record schema mismatch")
        key = (int(record["seed"]), str(record["method"]), record["budget"])
        if key in record_map or key[0] not in seeds or key[1:] not in expected_configurations:
            raise ValueError("duplicate or unexpected solver record")
        method, budget = key[1], key[2]
        split = split_map[key[0]]
        integer_fields = (
            "seed",
            "rank",
            "panel_size",
            "selected_snp_count",
            "support_count",
            "loading_matrix_rank",
            "native_iterations",
            "warmup_repetitions",
            "timed_repetitions",
        )
        if any(
            isinstance(record[field], bool) or not isinstance(record[field], int)
            for field in integer_fields
        ):
            raise ValueError("solver record integer field type mismatch")
        numeric_fields = (
            "alpha",
            "full_objective_common",
            "full_stationarity_common",
            "common_stationarity_step_size",
            "support_threshold",
            "support_fraction",
        )
        if any(
            isinstance(record[field], bool)
            or not isinstance(record[field], (int, float))
            for field in numeric_fields
        ):
            raise ValueError("solver record numeric field type mismatch")
        if (
            float(record["full_objective_common"]) < 0.0
            or float(record["full_stationarity_common"]) < 0.0
            or float(record["common_stationarity_step_size"]) <= 0.0
            or not isinstance(record["native_status"], str)
            or not record["native_status"].strip()
            or not isinstance(record["native_converged"], bool)
        ):
            raise ValueError("solver record physical diagnostic domain mismatch")
        if (
            record["configuration"] != _configuration_name(method, budget)
            or record["rank"] != RANK
            or record["alpha"] != ALPHA
            or record["panel_size"] != PANEL_SIZE
            or record["selected_snp_count"] != PANEL_SIZE
            or record["selected_original_indices_sha256"]
            != split["selected_original_indices_sha256"]
            or any(record[field] != split[field] for field in (
                "train_indices_sha256",
                "test_indices_sha256",
                "train_sample_ids_sha256",
                "test_sample_ids_sha256",
                "test_labels_sha256",
            ))
            or record["panel_selected_by_train_only_pca_loadings"] is not True
            or record["same_panel_for_all_methods_within_split"] is not True
            or record["heldout_truth_used_for_solver_or_panel"] is not False
            or record["classifier_scope"]
            != "outer_train_nearest_centroid_after_solver_freeze"
            or record["common_stationarity_step_policy"]
            != "exact_panel_pca_inverse_squared_sigma_max"
            or record["support_threshold"] != SUPPORT_THRESHOLD
            or not _is_sha256(record["loading_matrix_sha256"])
            or not _is_sha256(record["procrustes_factor_sha256"])
            or not _is_sha256(record["train_scores_sha256"])
            or not _is_sha256(record["test_scores_sha256"])
            or not _is_sha256(record["train_centroids_sha256"])
            or not _is_sha256(record["test_predictions_sha256"])
        ):
            raise ValueError("solver record split/method identity mismatch")
        runtime = record["runtime_seconds"]
        if set(runtime) != {
            "shared_train_only_screen_seconds",
            "common_pca_reference_seconds",
            "common_reference_charged_to_end_to_end_seconds",
            "solver_only_seconds",
            "common_diagnostic_and_scoring_seconds",
            "end_to_end_seconds",
        } or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or float(value) < 0.0
            for value in runtime.values()
        ):
            raise ValueError("solver runtime schema/value mismatch")
        expected_end_to_end = (
            float(runtime["shared_train_only_screen_seconds"])
            + float(runtime["common_reference_charged_to_end_to_end_seconds"])
            + float(runtime["solver_only_seconds"])
            + float(runtime["common_diagnostic_and_scoring_seconds"])
        )
        if float(runtime["end_to_end_seconds"]) != expected_end_to_end:
            raise ValueError("end-to-end runtime does not equal its frozen components")
        timed_seconds = record["solver_timed_repetitions_seconds"]
        timing_positions = record["timing_position_by_repetition"]
        if (
            not isinstance(timed_seconds, list)
            or len(timed_seconds) != int(record["timed_repetitions"])
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or float(value) < 0.0
                for value in timed_seconds
            )
            or not isinstance(timing_positions, list)
            or len(timing_positions) != len(timed_seconds)
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value not in (0, 1)
                for value in timing_positions
            )
            or float(runtime["solver_only_seconds"])
            != float(np.median(np.asarray(timed_seconds, dtype=float)))
            or record["deterministic_repetition_outputs_identical"] is not True
        ):
            raise ValueError("solver timing repetition receipt mismatch")
        support_by_component = record["support_count_by_component"]
        if (
            not isinstance(support_by_component, list)
            or len(support_by_component) != RANK
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in support_by_component)
            or int(record["support_count"]) != sum(support_by_component)
            or not 0.0 <= float(record["support_fraction"]) <= 1.0
            or float(record["support_fraction"])
            != int(record["support_count"]) / float(PANEL_SIZE * RANK)
            or int(record["loading_matrix_rank"]) not in range(RANK + 1)
        ):
            raise ValueError("support receipt is invalid")
        common_qualified = bool(float(record["full_stationarity_common"]) <= TOL)
        if method == "pca":
            if (
                budget is not None
                or record["common_stationarity_qualified"] is not None
                or record["native_status"] != "closed_form_pca"
                or record["native_converged"] is not True
                or record["native_iterations"] != 1
                or record["warmup_repetitions"] != 0
                or record["timed_repetitions"] != 1
                or record["timing_position_by_repetition"] != [0]
                or any(record[field] is not None for field in (
                    "native_full_objective",
                    "native_full_stationarity",
                    "sketch_iterations",
                    "full_refinement_iterations",
                    "randomized_seed",
                    "randomized_workspace_seconds",
                    "randomized_optimization_seconds",
                    "sketch_residual_relative",
                    "randomized_workspace_timed_repetitions_seconds",
                    "randomized_optimization_timed_repetitions_seconds",
                ))
            ):
                raise ValueError("PCA record has solver-only fields")
        elif method == "vp":
            base_order = (
                ("vp", "randomized")
                if (key[0] + int(budget) // 500) % 2 == 0
                else ("randomized", "vp")
            )
            expected_positions = [
                (base_order if repetition % 2 == 0 else tuple(reversed(base_order))).index("vp")
                for repetition in range(SPARSE_TIMED_REPETITIONS)
            ]
            if (
                record["common_stationarity_qualified"] is not common_qualified
                or isinstance(record["native_full_objective"], bool)
                or not isinstance(record["native_full_objective"], (int, float))
                or float(record["native_full_objective"]) < 0.0
                or isinstance(record["native_full_stationarity"], bool)
                or not isinstance(record["native_full_stationarity"], (int, float))
                or float(record["native_full_stationarity"]) < 0.0
                or record["native_iterations"] < 1
                or record["native_iterations"] > int(budget)
                or record["warmup_repetitions"] != SPARSE_WARMUP_REPETITIONS
                or record["timed_repetitions"] != SPARSE_TIMED_REPETITIONS
                or record["timing_position_by_repetition"] != expected_positions
                or any(record[field] is not None for field in (
                    "sketch_iterations",
                    "full_refinement_iterations",
                    "randomized_seed",
                    "randomized_workspace_seconds",
                    "randomized_optimization_seconds",
                    "sketch_residual_relative",
                    "randomized_workspace_timed_repetitions_seconds",
                    "randomized_optimization_timed_repetitions_seconds",
                ))
            ):
                raise ValueError("VP record solver metadata mismatch")
        else:
            sketch_budget = int(budget) // 2
            full_budget = int(budget) - sketch_budget
            base_order = (
                ("vp", "randomized")
                if (key[0] + int(budget) // 500) % 2 == 0
                else ("randomized", "vp")
            )
            expected_positions = [
                (base_order if repetition % 2 == 0 else tuple(reversed(base_order))).index("randomized")
                for repetition in range(SPARSE_TIMED_REPETITIONS)
            ]
            if (
                record["common_stationarity_qualified"] is not common_qualified
                or isinstance(record["native_full_objective"], bool)
                or not isinstance(record["native_full_objective"], (int, float))
                or float(record["native_full_objective"]) < 0.0
                or isinstance(record["native_full_stationarity"], bool)
                or not isinstance(record["native_full_stationarity"], (int, float))
                or float(record["native_full_stationarity"]) < 0.0
                or record["randomized_seed"] != _derived_randomized_seed(key[0])
                or isinstance(record["randomized_seed"], bool)
                or not isinstance(record["randomized_seed"], int)
                or isinstance(record["sketch_iterations"], bool)
                or not isinstance(record["sketch_iterations"], int)
                or isinstance(record["full_refinement_iterations"], bool)
                or not isinstance(record["full_refinement_iterations"], int)
                or record["warmup_repetitions"] != SPARSE_WARMUP_REPETITIONS
                or record["timed_repetitions"] != SPARSE_TIMED_REPETITIONS
                or record["timing_position_by_repetition"] != expected_positions
                or not 0 <= int(record["sketch_iterations"]) <= sketch_budget
                or not 0 <= int(record["full_refinement_iterations"]) <= full_budget
                or int(record["native_iterations"])
                != int(record["sketch_iterations"]) + int(record["full_refinement_iterations"])
                or float(record["randomized_workspace_seconds"]) < 0.0
                or float(record["randomized_optimization_seconds"]) < 0.0
                or float(record["sketch_residual_relative"]) < 0.0
            ):
                raise ValueError("randomized record solver metadata mismatch")
            workspace_repetitions = record[
                "randomized_workspace_timed_repetitions_seconds"
            ]
            optimization_repetitions = record[
                "randomized_optimization_timed_repetitions_seconds"
            ]
            if (
                not isinstance(workspace_repetitions, list)
                or not isinstance(optimization_repetitions, list)
                or len(workspace_repetitions) != SPARSE_TIMED_REPETITIONS
                or len(optimization_repetitions) != SPARSE_TIMED_REPETITIONS
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or float(value) < 0.0
                    for value in (*workspace_repetitions, *optimization_repetitions)
                )
                or float(record["randomized_workspace_seconds"])
                != float(np.median(np.asarray(workspace_repetitions, dtype=float)))
                or float(record["randomized_optimization_seconds"])
                != float(np.median(np.asarray(optimization_repetitions, dtype=float)))
                or any(
                    float(total) < float(workspace) + float(optimization)
                    for total, workspace, optimization in zip(
                        timed_seconds,
                        workspace_repetitions,
                        optimization_repetitions,
                        strict=True,
                    )
                )
            ):
                raise ValueError("randomized timing subcomponent receipt mismatch")
        record_map[key] = record
    if set(record_map) != {
        (int(seed), method, budget)
        for seed in seeds
        for method, budget in expected_configurations
    }:
        raise ValueError("record grid exact cover failed")

    prediction_rows = payload["heldout_predictions"]
    if not isinstance(prediction_rows, list) or len(prediction_rows) != expected_record_count * 200:
        raise ValueError("held-out prediction rows do not exact-cover records")
    grouped: dict[tuple[int, str, int | None], list[Mapping[str, Any]]] = {}
    prediction_schema = {
        "seed", "method", "budget", "configuration", "sample_index", "sample_id", "truth", "prediction"
    }
    for row in prediction_rows:
        if set(row) != prediction_schema:
            raise ValueError("held-out prediction schema mismatch")
        key = (int(row["seed"]), str(row["method"]), row["budget"])
        if key not in record_map or row["configuration"] != _configuration_name(key[1], key[2]):
            raise ValueError("held-out prediction has an unknown record identity")
        if row["truth"] not in EXPECTED_POPULATIONS or row["prediction"] not in EXPECTED_POPULATIONS:
            raise ValueError("held-out prediction contains an invalid population")
        grouped.setdefault(key, []).append(row)
    for seed in seeds:
        baseline_key = (int(seed), "pca", None)
        baseline_rows = grouped.get(baseline_key, [])
        if len(baseline_rows) != 200:
            raise ValueError("PCA prediction rows missing")
        split = split_map[int(seed)]
        baseline_indices = [int(row["sample_index"]) for row in baseline_rows]
        baseline_ids = np.asarray([row["sample_id"] for row in baseline_rows], dtype=str)
        baseline_truth = np.asarray([row["truth"] for row in baseline_rows], dtype=str)
        if (
            baseline_indices != split["test_indices"]
            or len(set(baseline_indices)) != 200
            or array_sha256(baseline_ids) != split["test_sample_ids_sha256"]
            or array_sha256(baseline_truth) != split["test_labels_sha256"]
        ):
            raise ValueError("held-out sample identity/truth is not bound to split receipt")
        for method, budget in expected_configurations:
            key = (int(seed), method, budget)
            rows = grouped.get(key, [])
            if len(rows) != 200:
                raise ValueError("prediction group does not contain exactly 200 held-out rows")
            if (
                [int(row["sample_index"]) for row in rows] != baseline_indices
                or [row["sample_id"] for row in rows] != baseline_ids.tolist()
                or [row["truth"] for row in rows] != baseline_truth.tolist()
            ):
                raise ValueError("methods do not score the same held-out samples in the same order")
            predicted = np.asarray([row["prediction"] for row in rows], dtype=str)
            record = record_map[key]
            if array_sha256(predicted) != record["test_predictions_sha256"]:
                raise ValueError("prediction hash does not match record")
            _validate_metric_block(record, rows)

    expected_summaries = _summarize_records(records, seeds=seeds, budgets=budgets)
    expected_quality = _matched_quality_rows(records, seeds=seeds, budgets=budgets)
    expected_gates = _eligibility_gates(records, expected_quality, seeds=seeds, budgets=budgets)
    if canonical_sha256(payload["summaries"]) != canonical_sha256(expected_summaries):
        raise ValueError("summary table does not derive from raw records")
    if canonical_sha256(payload["matched_quality"]) != canonical_sha256(expected_quality):
        raise ValueError("matched-quality table does not derive from raw records")
    if canonical_sha256(payload["gate_results"]) != canonical_sha256(expected_gates):
        raise ValueError("gate decision does not derive from raw records")
    unsigned = dict(payload)
    observed_analysis = unsigned.pop("analysis_sha256")
    if not _is_sha256(observed_analysis) or canonical_sha256(unsigned) != observed_analysis:
        raise ValueError("solver analysis digest mismatch")
    return {
        "status": "structurally_valid",
        "phase": phase,
        "record_count": len(records),
        "prediction_row_count": len(prediction_rows),
        "analysis_sha256": observed_analysis,
        "selected_budget": expected_gates["selected_budget"],
        "confirmation_permitted": expected_gates["confirmation_permitted"],
    }


def _csv_value(value: Any) -> str:
    if isinstance(value, (Mapping, list)) or value is None or isinstance(value, bool):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return str(value)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ValueError("CSV rows cannot be empty")
    fields = sorted(set().union(*(set(row) for row in rows)))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _csv_value(row.get(field)) for field in fields})
    return stream.getvalue().encode("utf-8")


def _figure_bytes(payload: Mapping[str, Any]) -> bytes:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records = payload["records"]
    budgets = payload["budgets"]
    seeds = payload["seeds"]
    order = _configuration_order(budgets)
    names = [_configuration_name(method, budget).replace("_budget_", "\n") for method, budget in order]
    colors = ["#334155", "#2563eb", "#0f766e", "#7c3aed", "#c2410c"][: len(order)]
    record_map = {
        (record["seed"], record["method"], record["budget"]): record
        for record in records
    }
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    positions = np.arange(len(order))
    for position, ((method, budget), color) in enumerate(zip(order, colors, strict=True)):
        rows = [record_map[(seed, method, budget)] for seed in seeds]
        values = [row["balanced_accuracy"] for row in rows]
        axes[0, 0].scatter(np.full(len(values), position), values, color=color, s=28, alpha=0.8)
        axes[0, 0].plot([position - 0.22, position + 0.22], [np.mean(values)] * 2, color="black", lw=2)
    axes[0, 0].set_xticks(positions, names)
    axes[0, 0].set_ylim(0.0, 1.02)
    axes[0, 0].set_ylabel("Held-out balanced accuracy")
    axes[0, 0].set_title("Same train-only top-400 panel")
    sparse_order = [(method, budget) for method, budget in order if method != "pca"]
    sparse_names = [_configuration_name(method, budget).replace("_budget_", "\n") for method, budget in sparse_order]
    for position, (method, budget) in enumerate(sparse_order):
        values = [record_map[(seed, method, budget)]["full_stationarity_common"] for seed in seeds]
        axes[0, 1].scatter(np.full(len(values), position), values, s=28)
        axes[0, 1].plot([position - 0.22, position + 0.22], [np.median(values)] * 2, color="black", lw=2)
    axes[0, 1].axhline(TOL, color="#dc2626", ls="--", lw=1.5, label=f"tol={TOL:g}")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xticks(np.arange(len(sparse_order)), sparse_names)
    axes[0, 1].set_ylabel("Common full-data stationarity")
    axes[0, 1].set_title("One objective, one diagnostic step")
    axes[0, 1].legend(frameon=False)
    for position, (method, budget) in enumerate(sparse_order):
        values = [record_map[(seed, method, budget)]["runtime_seconds"]["solver_only_seconds"] for seed in seeds]
        axes[1, 0].scatter(np.full(len(values), position), values, s=28)
        axes[1, 0].plot([position - 0.22, position + 0.22], [np.median(values)] * 2, color="black", lw=2)
    axes[1, 0].set_xticks(np.arange(len(sparse_order)), sparse_names)
    axes[1, 0].set_ylabel("Solver-only seconds")
    axes[1, 0].set_title("Cold independent cells; RND includes workspace")
    quality = payload["matched_quality"]
    for position, budget in enumerate(budgets):
        rows = [row for row in quality if row["budget"] == budget]
        values = [row["randomized_relative_objective_excess"] for row in rows]
        axes[1, 1].scatter(np.full(len(values), position), values, s=28)
        axes[1, 1].plot([position - 0.22, position + 0.22], [np.median(values)] * 2, color="black", lw=2)
    axes[1, 1].axhline(MATCHED_OBJECTIVE_RELATIVE_EXCESS, color="#dc2626", ls="--", lw=1.5)
    axes[1, 1].set_xticks(np.arange(len(budgets)), [str(value) for value in budgets])
    axes[1, 1].set_ylabel("RND relative objective excess vs VP")
    axes[1, 1].set_title("Matched full-objective diagnostic")
    figure.suptitle(
        "Two-stage controlled solver comparison (development)"
        if payload["phase"] == "feasibility"
        else "Two-stage controlled solver comparison (fresh confirmation)",
        fontsize=15,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=180, metadata={"Software": "vp-rspca controlled solver companion"})
    plt.close(figure)
    return buffer.getvalue()


def _artifact_payloads(payload: Mapping[str, Any]) -> dict[str, bytes]:
    return {
        RESULT_FILENAME: _strict_json_bytes(payload),
        RECORDS_FILENAME: _csv_bytes(payload["records"]),
        SUMMARY_FILENAME: _csv_bytes(payload["summaries"]),
        QUALITY_FILENAME: _csv_bytes(payload["matched_quality"]),
        PREDICTIONS_FILENAME: _csv_bytes(payload["heldout_predictions"]),
        FIGURE_FILENAME: _figure_bytes(payload),
    }


def _all_package_relative_files(output_directory: Path) -> set[str]:
    relative: set[str] = set()
    for root, directories, files in os.walk(output_directory, followlinks=False):
        root_path = Path(root)
        for directory in directories:
            if (root_path / directory).is_symlink():
                raise ValueError("package contains a symlink directory")
        for filename in files:
            path = root_path / filename
            _regular_file(path, role="package artifact")
            relative.add(path.relative_to(output_directory).as_posix())
    return relative


def write_solver_package(
    payload: Mapping[str, Any],
    output_directory: Path,
    *,
    lock: Mapping[str, Any],
    expected_lock_file_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    validate_solver_payload(
        payload,
        lock=lock,
        expected_lock_file_sha256=expected_lock_file_sha256,
    )
    if os.path.lexists(output_directory):
        raise FileExistsError(f"output directory already exists: {output_directory}")
    output_directory.mkdir(parents=True, exist_ok=False)
    payloads = _artifact_payloads(payload)
    for relative, content in payloads.items():
        (output_directory / relative).write_bytes(content)
    snapshot_root = output_directory / "provenance" / "source_snapshot"
    snapshot_root.mkdir(parents=True)
    for relative in SOURCE_FILE_ROLES:
        destination = snapshot_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(repo_root) / relative, destination)
    expected_files = set(ARTIFACT_FILENAMES) | {
        f"provenance/source_snapshot/{relative}" for relative in SOURCE_FILE_ROLES
    }
    artifacts = []
    for relative in sorted(expected_files):
        path = output_directory / relative
        artifacts.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "bytes": int(path.stat().st_size),
                "role": (
                    SOURCE_FILE_ROLES[relative.removeprefix("provenance/source_snapshot/")]
                    if relative.startswith("provenance/source_snapshot/")
                    else "analysis_artifact"
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "phase": payload["phase"],
        "analysis_sha256": payload["analysis_sha256"],
        "active_lock_file_sha256": payload["active_lock_file_sha256"],
        "active_lock_payload_sha256": payload["active_lock_payload_sha256"],
        "source_manifest_sha256": payload["source_identity"]["source_manifest_sha256"],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    (output_directory / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
    return validate_solver_package(
        output_directory,
        lock=lock,
        expected_lock_file_sha256=expected_lock_file_sha256,
        repo_root=repo_root,
    )


def validate_solver_package(
    output_directory: Path,
    *,
    lock: Mapping[str, Any],
    expected_lock_file_sha256: str,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("solver package must be a real directory")
    manifest_path = output_directory / MANIFEST_FILENAME
    _regular_file(manifest_path, role="solver output manifest")
    package_anchor = _file_sha256(manifest_path)
    if expected_package_anchor_sha256 is not None and package_anchor != expected_package_anchor_sha256:
        raise ValueError("solver package external manifest anchor mismatch")
    manifest = _strict_json_load(manifest_path)
    if set(manifest) != {
        "schema_version",
        "phase",
        "analysis_sha256",
        "active_lock_file_sha256",
        "active_lock_payload_sha256",
        "source_manifest_sha256",
        "artifact_count",
        "artifacts",
        "manifest_payload_sha256",
    } or manifest["schema_version"] != 1:
        raise ValueError("solver package manifest schema mismatch")
    unsigned_manifest = dict(manifest)
    observed_manifest_payload = unsigned_manifest.pop("manifest_payload_sha256")
    if canonical_sha256(unsigned_manifest) != observed_manifest_payload:
        raise ValueError("solver package manifest payload digest mismatch")
    expected_files = set(ARTIFACT_FILENAMES) | {
        f"provenance/source_snapshot/{relative}" for relative in SOURCE_FILE_ROLES
    }
    actual_files = _all_package_relative_files(output_directory)
    if actual_files != expected_files | {MANIFEST_FILENAME}:
        raise ValueError(
            f"solver package exact file closure mismatch: missing={sorted((expected_files | {MANIFEST_FILENAME}) - actual_files)}, "
            f"extra={sorted(actual_files - (expected_files | {MANIFEST_FILENAME}))}"
        )
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != len(expected_files) or manifest["artifact_count"] != len(expected_files):
        raise ValueError("solver package artifact count mismatch")
    artifact_map: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if set(artifact) != {"path", "sha256", "bytes", "role"}:
            raise ValueError("solver package artifact schema mismatch")
        relative = artifact["path"]
        if relative in artifact_map or relative not in expected_files:
            raise ValueError("solver package artifact path is duplicate/unexpected")
        path = output_directory / relative
        expected_role = (
            SOURCE_FILE_ROLES[
                relative.removeprefix("provenance/source_snapshot/")
            ]
            if relative.startswith("provenance/source_snapshot/")
            else "analysis_artifact"
        )
        if (
            not _is_sha256(artifact["sha256"])
            or _file_sha256(path) != artifact["sha256"]
            or int(path.stat().st_size) != artifact["bytes"]
            or artifact["role"] != expected_role
        ):
            raise ValueError("solver package artifact receipt mismatch")
        artifact_map[relative] = artifact
    if set(artifact_map) != expected_files:
        raise ValueError("solver package artifacts do not exact-cover files")
    payload = _strict_json_load(output_directory / RESULT_FILENAME)
    payload_validation = validate_solver_payload(
        payload,
        lock=lock,
        expected_lock_file_sha256=expected_lock_file_sha256,
    )
    if (
        manifest["phase"] != payload["phase"]
        or manifest["analysis_sha256"] != payload["analysis_sha256"]
        or manifest["active_lock_file_sha256"] != expected_lock_file_sha256
        or manifest["active_lock_payload_sha256"] != lock["lock_payload_sha256"]
        or manifest["source_manifest_sha256"]
        != payload["source_identity"]["source_manifest_sha256"]
    ):
        raise ValueError("solver package manifest/result identity mismatch")
    if expected_analysis_sha256 is not None and payload["analysis_sha256"] != expected_analysis_sha256:
        raise ValueError("solver analysis external anchor mismatch")
    expected_payloads = _artifact_payloads(payload)
    for relative in ARTIFACT_FILENAMES:
        if relative == FIGURE_FILENAME:
            continue
        if (output_directory / relative).read_bytes() != expected_payloads[relative]:
            raise ValueError(f"derived artifact differs from result payload: {relative}")
    png = (output_directory / FIGURE_FILENAME).read_bytes()
    if (
        not png.startswith(b"\x89PNG\r\n\x1a\n")
        or len(png) < 24
        or png[12:16] != b"IHDR"
        or int.from_bytes(png[16:20], "big") < 1800
        or int.from_bytes(png[20:24], "big") < 1100
    ):
        raise ValueError("solver figure is not a sufficiently large decodable PNG")
    for relative, expected_hash in payload["source_identity"]["files"].items():
        snapshot = output_directory / "provenance" / "source_snapshot" / relative
        if _file_sha256(snapshot) != expected_hash:
            raise ValueError("source snapshot does not match frozen source identity")
        if repo_root is not None and _file_sha256(Path(repo_root) / relative) != expected_hash:
            raise ValueError("current repository differs from frozen execution source")
    externally_anchored = bool(
        expected_analysis_sha256 is not None
        and expected_package_anchor_sha256 is not None
    )
    return {
        **payload_validation,
        "status": "valid_and_externally_anchored" if externally_anchored else "structurally_valid_unanchored",
        "package_anchor_sha256": package_anchor,
        "analysis_anchor_verified": expected_analysis_sha256 is not None,
        "package_anchor_verified": expected_package_anchor_sha256 is not None,
        "active_lock_anchor_verified": manifest["active_lock_file_sha256"] == expected_lock_file_sha256,
        "source_snapshot_exact_cover_verified": True,
    }


def build_confirmation_decision_lock(
    *,
    feasibility_package: Path,
    feasibility_lock: Mapping[str, Any],
    feasibility_lock_file_sha256: str,
    expected_feasibility_analysis_sha256: str,
    expected_feasibility_package_anchor_sha256: str,
    repo_root: Path,
) -> dict[str, Any]:
    validation = validate_solver_package(
        feasibility_package,
        lock=feasibility_lock,
        expected_lock_file_sha256=feasibility_lock_file_sha256,
        expected_analysis_sha256=expected_feasibility_analysis_sha256,
        expected_package_anchor_sha256=expected_feasibility_package_anchor_sha256,
        repo_root=repo_root,
    )
    if validation["phase"] != "feasibility":
        raise ValueError("confirmation decision must consume a feasibility package")
    payload = _strict_json_load(feasibility_package / RESULT_FILENAME)
    selected_budget = payload["gate_results"]["selected_budget"]
    if (
        payload["gate_results"]["confirmation_permitted"] is not True
        or selected_budget not in BUDGETS
    ):
        raise ValueError("development gate failed; confirmation lock is forbidden")
    current_source = _source_identity(repo_root)
    if current_source != feasibility_lock["execution_source_identity"]:
        raise ValueError("source changed after feasibility lock")
    novelty_audit = audit_confirmation_seed_novelty(repo_root)
    _validate_seed_novelty_audit(novelty_audit)
    decision: dict[str, Any] = {
        "schema_version": DECISION_LOCK_SCHEMA_VERSION,
        "status": "locked_after_eligible_development_before_fresh_confirmation",
        "feasibility_lock_file_sha256": feasibility_lock_file_sha256,
        "feasibility_lock_payload_sha256": feasibility_lock["lock_payload_sha256"],
        "feasibility_analysis_sha256": expected_feasibility_analysis_sha256,
        "feasibility_package_anchor_sha256": expected_feasibility_package_anchor_sha256,
        "selected_budget": int(selected_budget),
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "protocol": dict(FROZEN_PROTOCOL),
        "eligibility_gates": dict(FROZEN_ELIGIBILITY_GATES),
        "confirmation_seed_novelty_audit": novelty_audit,
        "prepared_input_identity": dict(feasibility_lock["prepared_input_identity"]),
        "execution_source_identity": current_source,
        "selection_receipt": {
            "development_gate_passed": True,
            "selected_lowest_passing_budget": True,
            "development_outcomes_seen_for_budget_selection": True,
            "confirmation_outcomes_seen_before_lock": False,
            "fresh_confirmation_seed_schedule": True,
            "same_process_gate_to_confirmation_execution": False,
        },
    }
    decision["lock_payload_sha256"] = canonical_sha256(decision)
    validate_confirmation_decision_lock(decision)
    return decision


def validate_confirmation_decision_lock(lock: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "status",
        "feasibility_lock_file_sha256",
        "feasibility_lock_payload_sha256",
        "feasibility_analysis_sha256",
        "feasibility_package_anchor_sha256",
        "selected_budget",
        "confirmation_seeds",
        "protocol",
        "eligibility_gates",
        "confirmation_seed_novelty_audit",
        "prepared_input_identity",
        "execution_source_identity",
        "selection_receipt",
        "lock_payload_sha256",
    }
    if set(lock) != expected or lock["schema_version"] != DECISION_LOCK_SCHEMA_VERSION:
        raise ValueError("confirmation decision-lock schema mismatch")
    grid._reject_nonfinite(lock)
    if (
        lock["status"] != "locked_after_eligible_development_before_fresh_confirmation"
        or any(
            not _is_sha256(lock[field])
            for field in (
                "feasibility_lock_file_sha256",
                "feasibility_lock_payload_sha256",
                "feasibility_analysis_sha256",
                "feasibility_package_anchor_sha256",
            )
        )
        or lock["selected_budget"] not in BUDGETS
        or lock["confirmation_seeds"] != list(CONFIRMATION_SEEDS)
        or lock["protocol"] != FROZEN_PROTOCOL
        or lock["eligibility_gates"] != FROZEN_ELIGIBILITY_GATES
        or lock["selection_receipt"]
        != {
            "development_gate_passed": True,
            "selected_lowest_passing_budget": True,
            "development_outcomes_seen_for_budget_selection": True,
            "confirmation_outcomes_seen_before_lock": False,
            "fresh_confirmation_seed_schedule": True,
            "same_process_gate_to_confirmation_execution": False,
        }
    ):
        raise ValueError("confirmation decision-lock protocol mismatch")
    prepared = lock["prepared_input_identity"]
    _validate_seed_novelty_audit(lock["confirmation_seed_novelty_audit"])
    if (
        not isinstance(prepared, Mapping)
        or prepared.get("samples") != 1000
        or prepared.get("manifest_population_design") != "five_superpopulation"
        or prepared.get("ld_pruned_input_validated_from_external_manifest") is not True
        or not _is_sha256(prepared.get("genotype_npz_sha256"))
    ):
        raise ValueError("confirmation decision-lock prepared input mismatch")
    _validate_source_identity(lock["execution_source_identity"])
    unsigned = dict(lock)
    observed = unsigned.pop("lock_payload_sha256")
    if not _is_sha256(observed) or canonical_sha256(unsigned) != observed:
        raise ValueError("confirmation decision-lock digest mismatch")


def load_anchored_active_lock(path: Path, expected_sha256: str) -> dict[str, Any]:
    _regular_file(path, role="active solver lock")
    if not _is_sha256(expected_sha256) or _file_sha256(path) != expected_sha256:
        raise ValueError("active solver lock external anchor mismatch")
    lock = _strict_json_load(path)
    status = lock.get("status") if isinstance(lock, Mapping) else None
    if status == "locked_before_solver_feasibility_execution":
        validate_feasibility_lock(lock)
    elif status == "locked_after_eligible_development_before_fresh_confirmation":
        validate_confirmation_decision_lock(lock)
    else:
        raise ValueError("unknown active solver lock status")
    return lock


def _failure_receipt(
    *,
    feasibility_package: Path,
    expected_analysis_sha256: str,
    expected_package_anchor_sha256: str,
    feasibility_lock_file_sha256: str,
) -> dict[str, Any]:
    payload = _strict_json_load(feasibility_package / RESULT_FILENAME)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "development_gate_failed_confirmation_forbidden",
        "feasibility_analysis_sha256": expected_analysis_sha256,
        "feasibility_package_anchor_sha256": expected_package_anchor_sha256,
        "feasibility_lock_file_sha256": feasibility_lock_file_sha256,
        "gate_results": payload["gate_results"],
        "confirmation_seeds_executed": [],
        "confirmation_lock_created": False,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    lock_parser = subparsers.add_parser("lock", help="freeze development protocol before solver outcomes")
    lock_parser.add_argument("--parent-confirmation-lock", type=Path, required=True)
    lock_parser.add_argument("--expected-parent-lock-sha256", required=True)
    lock_parser.add_argument("--repo-root", type=Path, required=True)
    lock_parser.add_argument("--output-lock", type=Path, required=True)

    run_parser = subparsers.add_parser("run-feasibility", help="run the exact five-seed development grid")
    run_parser.add_argument("--lock", type=Path, required=True)
    run_parser.add_argument("--expected-lock-sha256", required=True)
    run_parser.add_argument("--genotypes", type=Path, required=True)
    run_parser.add_argument("--source-manifest", type=Path, required=True)
    run_parser.add_argument("--expected-source-manifest-sha256", required=True)
    run_parser.add_argument("--repo-root", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)

    validate_parser = subparsers.add_parser("validate", help="validate an anchored feasibility or confirmation package")
    validate_parser.add_argument("--lock", type=Path, required=True)
    validate_parser.add_argument("--expected-lock-sha256", required=True)
    validate_parser.add_argument("--output-dir", type=Path, required=True)
    validate_parser.add_argument("--expected-analysis-sha256", required=True)
    validate_parser.add_argument("--expected-package-anchor-sha256", required=True)
    validate_parser.add_argument("--repo-root", type=Path, required=True)

    decide_parser = subparsers.add_parser("decide", help="create a fresh-confirmation lock only after an anchored development pass")
    decide_parser.add_argument("--feasibility-lock", type=Path, required=True)
    decide_parser.add_argument("--expected-feasibility-lock-sha256", required=True)
    decide_parser.add_argument("--feasibility-package", type=Path, required=True)
    decide_parser.add_argument("--expected-analysis-sha256", required=True)
    decide_parser.add_argument("--expected-package-anchor-sha256", required=True)
    decide_parser.add_argument("--repo-root", type=Path, required=True)
    decide_parser.add_argument("--output-decision-lock", type=Path, required=True)
    decide_parser.add_argument("--failure-receipt", type=Path, required=True)

    confirmation_parser = subparsers.add_parser("run-confirmation", help="consume an externally anchored decision lock on fresh seeds")
    confirmation_parser.add_argument("--decision-lock", type=Path, required=True)
    confirmation_parser.add_argument("--expected-decision-lock-sha256", required=True)
    confirmation_parser.add_argument("--genotypes", type=Path, required=True)
    confirmation_parser.add_argument("--source-manifest", type=Path, required=True)
    confirmation_parser.add_argument("--expected-source-manifest-sha256", required=True)
    confirmation_parser.add_argument("--repo-root", type=Path, required=True)
    confirmation_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _require_single_thread_environment() -> None:
    names = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    mismatched = {name: os.environ.get(name) for name in names if os.environ.get(name) != "1"}
    if mismatched:
        raise ValueError(f"all BLAS thread controls must equal 1 before process start: {mismatched}")


def _load_bound_input(
    args: argparse.Namespace,
    lock: Mapping[str, Any],
) -> tuple[grid.GenotypeInput, dict[str, Any]]:
    data = grid.load_genotype_npz(args.genotypes)
    receipt = grid.validate_external_source_manifest(
        args.source_manifest,
        args.genotypes,
        data,
        expected_manifest_sha256=args.expected_source_manifest_sha256,
    )
    return data, _bound_input_identity(data, receipt, lock)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    if args.command == "lock":
        payload = build_feasibility_lock(
            args.parent_confirmation_lock,
            expected_parent_lock_sha256=args.expected_parent_lock_sha256,
            repo_root=args.repo_root,
        )
        _write_new_json(payload, args.output_lock)
        print(
            json.dumps(
                {
                    "event": "feasibility_lock_created_before_solver_outcomes",
                    "lock_file_sha256": _file_sha256(args.output_lock),
                    "lock_payload_sha256": payload["lock_payload_sha256"],
                    "output_lock": str(args.output_lock),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.command == "validate":
        lock = load_anchored_active_lock(args.lock, args.expected_lock_sha256)
        validation = validate_solver_package(
            args.output_dir,
            lock=lock,
            expected_lock_file_sha256=args.expected_lock_sha256,
            expected_analysis_sha256=args.expected_analysis_sha256,
            expected_package_anchor_sha256=args.expected_package_anchor_sha256,
            repo_root=args.repo_root,
        )
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 0

    if args.command == "decide":
        feasibility_lock = load_anchored_feasibility_lock(
            args.feasibility_lock,
            args.expected_feasibility_lock_sha256,
        )
        validation = validate_solver_package(
            args.feasibility_package,
            lock=feasibility_lock,
            expected_lock_file_sha256=args.expected_feasibility_lock_sha256,
            expected_analysis_sha256=args.expected_analysis_sha256,
            expected_package_anchor_sha256=args.expected_package_anchor_sha256,
            repo_root=args.repo_root,
        )
        if validation["confirmation_permitted"] is not True:
            receipt = _failure_receipt(
                feasibility_package=args.feasibility_package,
                expected_analysis_sha256=args.expected_analysis_sha256,
                expected_package_anchor_sha256=args.expected_package_anchor_sha256,
                feasibility_lock_file_sha256=args.expected_feasibility_lock_sha256,
            )
            _write_new_json(receipt, args.failure_receipt)
            print(
                json.dumps(
                    {
                        "event": "development_gate_failed_confirmation_forbidden",
                        "failure_receipt": str(args.failure_receipt),
                        "receipt_file_sha256": _file_sha256(args.failure_receipt),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return 2
        decision = build_confirmation_decision_lock(
            feasibility_package=args.feasibility_package,
            feasibility_lock=feasibility_lock,
            feasibility_lock_file_sha256=args.expected_feasibility_lock_sha256,
            expected_feasibility_analysis_sha256=args.expected_analysis_sha256,
            expected_feasibility_package_anchor_sha256=args.expected_package_anchor_sha256,
            repo_root=args.repo_root,
        )
        _write_new_json(decision, args.output_decision_lock)
        print(
            json.dumps(
                {
                    "event": "fresh_confirmation_lock_created",
                    "selected_budget": decision["selected_budget"],
                    "decision_lock_file_sha256": _file_sha256(args.output_decision_lock),
                    "decision_lock_payload_sha256": decision["lock_payload_sha256"],
                    "output_decision_lock": str(args.output_decision_lock),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    _require_single_thread_environment()
    if args.command == "run-feasibility":
        lock = load_anchored_feasibility_lock(args.lock, args.expected_lock_sha256)
        if _source_identity(args.repo_root) != lock["execution_source_identity"]:
            raise ValueError("current source differs from pre-outcome feasibility lock")
        data, input_identity = _load_bound_input(args, lock)
        payload = evaluate_solver_experiment(
            data,
            phase="feasibility",
            seeds=DEVELOPMENT_SEEDS,
            budgets=BUDGETS,
            lock=lock,
            lock_file_sha256=args.expected_lock_sha256,
            input_identity=input_identity,
            source_identity=_source_identity(args.repo_root),
        )
    else:
        lock = load_anchored_active_lock(
            args.decision_lock,
            args.expected_decision_lock_sha256,
        )
        if lock["status"] != "locked_after_eligible_development_before_fresh_confirmation":
            raise ValueError("run-confirmation requires a decision lock")
        if _source_identity(args.repo_root) != lock["execution_source_identity"]:
            raise ValueError("current source differs from pre-confirmation decision lock")
        data, input_identity = _load_bound_input(args, lock)
        payload = evaluate_solver_experiment(
            data,
            phase="confirmation",
            seeds=CONFIRMATION_SEEDS,
            budgets=(int(lock["selected_budget"]),),
            lock=lock,
            lock_file_sha256=args.expected_decision_lock_sha256,
            input_identity=input_identity,
            source_identity=_source_identity(args.repo_root),
            decision_lock_file_sha256=args.expected_decision_lock_sha256,
        )
    validation = write_solver_package(
        payload,
        args.output_dir,
        lock=lock,
        expected_lock_file_sha256=(
            args.expected_lock_sha256
            if args.command == "run-feasibility"
            else args.expected_decision_lock_sha256
        ),
        repo_root=args.repo_root,
    )
    print(
        json.dumps(
            {
                "event": f"two_stage_solver_{payload['phase']}_package_complete",
                "analysis_sha256": payload["analysis_sha256"],
                "package_anchor_sha256": validation["package_anchor_sha256"],
                "active_lock_file_sha256": payload["active_lock_file_sha256"],
                "selected_budget": payload["gate_results"]["selected_budget"],
                "confirmation_permitted": payload["gate_results"]["confirmation_permitted"],
                "output_directory": str(args.output_dir),
                "validation": validation,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

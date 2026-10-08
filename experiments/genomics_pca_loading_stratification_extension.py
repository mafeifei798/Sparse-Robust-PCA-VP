#!/usr/bin/env python3
"""PCA-loading extension for clearer five-superpopulation stratification.

This is an additive post-hoc development experiment on the repository's
historical Genomics path.  It uses outer-training-only smartpca-like scaling,
closed-form PCA, unsupervised loading-magnitude ranking, a PCA refit on each
panel, and nearest-training-centroid held-out scoring.  Population labels are
used only for the stratified outer split, centroid fitting, and evaluation;
they are never passed to preprocessing, PCA, or loading ranking.

The development grid spans 4/6/8 PCs crossed with
top-400/top-800/top-1600/top-3200/all retained SNPs on seeds 2025..2029.
Top-400 is compression-sensitivity evidence only and cannot be selected.  A
predeclared rule chooses the smallest eligible panel (starting at top-800),
then the smallest rank, that is within 0.01 balanced accuracy of the same-rank
all-SNP result and meets AMR/EUR/SAS recall floors.  Passing development can
produce a proposal for an independently audited fresh-seed confirmation; this
runner never executes confirmation seeds 5001..5020.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import io
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import uuid

import numpy as np
from numpy.typing import NDArray

if __package__:
    from . import genomics_pca_loading_grid as base
else:  # Support direct execution from the repository root.
    import genomics_pca_loading_grid as base  # type: ignore[no-redef]

from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.genomics_aim import (
    fit_smartpca_preprocessor,
    stratified_split,
    transform_smartpca,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

DEVELOPMENT_SEEDS = (2025, 2026, 2027, 2028, 2029)
COMPONENT_GRID = (4, 6, 8)
PANEL_GRID: tuple[int | None, ...] = (400, 800, 1600, 3200, None)
MINIMUM_SELECTION_PANEL_SIZE = 800
CONFIRMATION_SEEDS = tuple(range(5001, 5021))
PREPARED_SOURCE_MANIFEST_SHA256 = (
    "9378814177e0721e711ce2e6c3695effc5ff766d2529c17cc5c4e0de84f30342"
)
EXPECTED_POPULATIONS = base.EXPECTED_POPULATIONS
SELECTION_GATES = {
    "minimum_selection_panel_size": MINIMUM_SELECTION_PANEL_SIZE,
    "balanced_accuracy_noninferiority_margin_vs_same_rank_all": 0.01,
    "minimum_mean_amr_recall": 0.70,
    "minimum_mean_eur_recall": 0.98,
    "minimum_mean_sas_recall": 0.98,
}
METHOD_SCOPE = {
    "methods": ["pca_loading_ranking"],
    "robust_solver_run": False,
    "vp_solver_run": False,
    "randomized_solver_run": False,
    "federated_solver_run": False,
}

RESULT_FILENAME = "stratification_extension_results.json"
METRICS_FILENAME = "stratification_extension_grid_metrics.csv"
POPULATION_FILENAME = "stratification_extension_population_recall.csv"
STABILITY_FILENAME = "stratification_extension_panel_stability.csv"
SAMPLE_EVIDENCE_FILENAME = "stratification_extension_sample_predictions.csv"
COORDINATES_FILENAME = "stratification_extension_registered_coordinates.csv"
PC12_FILENAME = "stratification_extension_pc12.png"
PC34_FILENAME = "stratification_extension_pc34.png"
PC56_FILENAME = "stratification_extension_pc56.png"
PC78_FILENAME = "stratification_extension_pc78.png"
CONFUSION_FILENAME = "stratification_extension_confusion.png"
LOCK_PROPOSAL_FILENAME = "stratification_confirmation_lock_proposal.json"
MANIFEST_FILENAME = "stratification_extension_output_manifest.json"
BASE_ARTIFACT_FILENAMES = (
    RESULT_FILENAME,
    METRICS_FILENAME,
    POPULATION_FILENAME,
    STABILITY_FILENAME,
    SAMPLE_EVIDENCE_FILENAME,
    COORDINATES_FILENAME,
    PC12_FILENAME,
    PC34_FILENAME,
    PC56_FILENAME,
    PC78_FILENAME,
    CONFUSION_FILENAME,
)
SOURCE_FILE_ROLES = {
    "experiments/genomics_pca_loading_stratification_extension.py": (
        "independent_posthoc_development_runner"
    ),
    "experiments/genomics_pca_loading_grid.py": (
        "shared_pca_loading_and_input_validation_helpers"
    ),
    "src/rspca/genomics_aim.py": "historical_genomics_preprocessing_and_split",
    "src/rspca/datasets.py": "hashing_and_runtime_identity",
    "src/rspca/__init__.py": "package_initialization_only",
    "src/rspca/accelerated.py": "eager_import_only_not_called_by_extension",
    "src/rspca/cross_modal_gates.py": "eager_import_only_not_called_by_extension",
    "src/rspca/federated.py": "eager_import_only_not_called_by_extension",
    "src/rspca/fedrpca_audit.py": "eager_import_only_not_called_by_extension",
    "src/rspca/gauge.py": "eager_import_only_not_called_by_extension",
    "src/rspca/gauge_stress.py": "eager_import_only_not_called_by_extension",
    "src/rspca/metrics.py": "eager_import_only_not_called_by_extension",
    "src/rspca/product_aggregation.py": "eager_import_only_not_called_by_extension",
    "src/rspca/randomized.py": "eager_import_only_not_called_by_extension",
    "src/rspca/robust.py": "eager_import_only_not_called_by_extension",
    "src/rspca/statistical_gates.py": "eager_import_only_not_called_by_extension",
    "src/rspca/trust_audit.py": "eager_import_only_not_called_by_extension",
    "src/rspca/vp.py": "eager_import_only_not_called_by_extension",
}

# The public in-memory evaluator is deliberately limited to tiny structural
# smoke tests.  Formal execution can reach the same numerical implementation
# only through the path-loading entry point below, which validates the fixed
# prepared-input manifest before receiving this private capability.
_TINY_SMOKE_EVALUATION_CAPABILITY = object()
_FORMAL_DEVELOPMENT_EVALUATION_CAPABILITY = object()
_TINY_SMOKE_PACKAGE_CAPABILITY = object()
_FORMAL_DEVELOPMENT_PACKAGE_CAPABILITY = object()


@dataclass(frozen=True)
class StratificationExtensionConfig:
    seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    component_grid: tuple[int, ...] = COMPONENT_GRID
    panel_sizes: tuple[int | None, ...] = PANEL_GRID
    test_fraction: float = 0.2
    loading_score: str = "l1_abs_sum"
    population_design: str = "five_superpopulation"
    analysis_status: str = "post_hoc_development_selection"
    analysis_mode: str = "development_protocol"


def _file_sha256(path: Path) -> str:
    return base._file_sha256(path)


def _is_sha256(value: Any) -> bool:
    return base._is_sha256(value)


def _is_git_object_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_bytes(value: Any) -> bytes:
    return base._strict_json_bytes(value)


def _strict_json_load(path: Path) -> Any:
    return base._strict_json_load(path)


def _regular_file(path: Path, *, role: str) -> os.stat_result:
    if path.is_symlink():
        raise ValueError(f"{role} must not be a symlink: {path}")
    status = path.stat()
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{role} must be a regular file: {path}")
    return status


def _panel_label(panel_size: int | None) -> str:
    return "all" if panel_size is None else str(int(panel_size))


def _validate_config(config: StratificationExtensionConfig) -> None:
    if not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("seeds must be non-empty and unique")
    if any(isinstance(seed, bool) or int(seed) < 0 for seed in config.seeds):
        raise ValueError("seeds must be non-negative integers")
    forbidden_confirmation_seeds = sorted(
        set(int(seed) for seed in config.seeds) & set(CONFIRMATION_SEEDS)
    )
    if forbidden_confirmation_seeds:
        raise ValueError(
            "development/smoke execution may not use reserved confirmation seeds: "
            f"{forbidden_confirmation_seeds}"
        )
    if config.component_grid != COMPONENT_GRID:
        raise ValueError(f"component_grid is fixed to {COMPONENT_GRID}")
    finite_panels = tuple(value for value in config.panel_sizes if value is not None)
    if (
        config.panel_sizes.count(None) != 1
        or config.panel_sizes[-1] is not None
        or finite_panels != tuple(sorted(finite_panels))
        or len(set(finite_panels)) != len(finite_panels)
        or any(
            isinstance(value, bool) or int(value) < max(config.component_grid)
            for value in finite_panels
        )
    ):
        raise ValueError("panel_sizes must be increasing moderate budgets plus all")
    if not 0.0 < float(config.test_fraction) < 1.0:
        raise ValueError("test_fraction must lie strictly between zero and one")
    if config.loading_score != "l1_abs_sum":
        raise ValueError("loading_score is fixed to l1_abs_sum")
    if config.population_design != "five_superpopulation":
        raise ValueError("population_design is fixed to five_superpopulation")
    if config.analysis_status != "post_hoc_development_selection":
        raise ValueError("analysis_status must remain post_hoc_development_selection")
    if config.analysis_mode not in {"development_protocol", "tiny_smoke"}:
        raise ValueError("analysis_mode must be development_protocol or tiny_smoke")
    if config.analysis_mode == "development_protocol" and (
        config.seeds != DEVELOPMENT_SEEDS
        or config.panel_sizes != PANEL_GRID
        or not math.isclose(config.test_fraction, 0.2, abs_tol=0.0, rel_tol=0.0)
    ):
        raise ValueError("development_protocol configuration is frozen")


def _protocol_exact(config: StratificationExtensionConfig) -> bool:
    return config == StratificationExtensionConfig()


def _runner_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_text(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed: {completed.stderr[-500:]}"
        )
    return completed.stdout


def _git_identity(root: Path) -> dict[str, Any]:
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel").strip()).resolve()
    if top_level != root:
        raise ValueError("runner project root is not the Git top-level directory")
    remote_names = sorted(
        value for value in _git_text(root, "remote").splitlines() if value
    )
    remotes = [
        {
            "name": name,
            "urls": sorted(
                value
                for value in _git_text(root, "remote", "get-url", "--all", name).splitlines()
                if value
            ),
        }
        for name in remote_names
    ]
    tracked_dirty_entries = sorted(
        value
        for value in _git_text(
            root, "status", "--porcelain=v1", "--untracked-files=no", "-z"
        ).split("\0")
        if value
    )
    untracked_paths = sorted(
        value
        for value in _git_text(
            root, "ls-files", "--others", "--exclude-standard", "-z"
        ).split("\0")
        if value
    )
    identity: dict[str, Any] = {
        "head_commit": _git_text(root, "rev-parse", "HEAD").strip(),
        "head_tree": _git_text(root, "rev-parse", "HEAD^{tree}").strip(),
        "remotes": remotes,
        "tracked_dirty": bool(tracked_dirty_entries),
        "tracked_dirty_entries": tracked_dirty_entries,
        "tracked_dirty_entries_sha256": canonical_sha256(
            {"tracked_dirty_entries": tracked_dirty_entries}
        ),
        "untracked": bool(untracked_paths),
        "untracked_paths": untracked_paths,
        "untracked_paths_sha256": canonical_sha256(
            {"untracked_paths": untracked_paths}
        ),
        "worktree_dirty_including_untracked": bool(
            tracked_dirty_entries or untracked_paths
        ),
        "repo_root_binding": "resolved_runner_parent_equals_git_toplevel",
    }
    identity["git_identity_sha256"] = canonical_sha256(identity)
    return identity


def _source_identity(repo_root: Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    expected_root = _runner_repo_root()
    if root != expected_root:
        raise ValueError(
            "repo_root must resolve exactly to the repository containing this runner"
        )
    audit_code = """
import json
import sys
from pathlib import Path
import genomics_pca_loading_stratification_extension
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
        'experiments/genomics_pca_loading_stratification_extension.py',
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
            "stratification fresh-import source audit failed: "
            f"{completed.stderr[-500:]}"
        )
    loaded_repo_python_files = set(json.loads(completed.stdout))
    if loaded_repo_python_files != set(SOURCE_FILE_ROLES):
        missing = sorted(set(SOURCE_FILE_ROLES) - loaded_repo_python_files)
        unexpected = sorted(loaded_repo_python_files - set(SOURCE_FILE_ROLES))
        raise ValueError(
            "stratification loaded-source exact cover mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )
    files: dict[str, str] = {}
    for relative in SOURCE_FILE_ROLES:
        path = root / relative
        _regular_file(path, role=relative)
        files[relative] = _file_sha256(path)
    loaded_files = sorted(loaded_repo_python_files)
    git_identity = _git_identity(root)
    return {
        "files": files,
        "file_roles": dict(SOURCE_FILE_ROLES),
        "source_manifest_sha256": canonical_sha256(
            {
                "files": files,
                "file_roles": SOURCE_FILE_ROLES,
                "loaded_repo_python_files": loaded_files,
                "git_identity": git_identity,
            }
        ),
        "loaded_repo_python_files_sha256": canonical_sha256(
            {"loaded_repo_python_files": loaded_files}
        ),
        "loaded_repo_python_files": loaded_files,
        "loaded_repo_python_files_exact_cover": True,
        "git_identity": git_identity,
        "method_scope": dict(METHOD_SCOPE),
        "existing_formal_or_grid_runner_modified": False,
        "fed_or_fedrpca_scientific_code_used": False,
    }


def _input_is_lock_eligible(identity: Mapping[str, Any]) -> bool:
    return (
        identity.get("external_source_manifest_anchor_verified") is True
        and identity.get("ld_pruned_input_validated_from_external_manifest") is True
        and identity.get("manifest_population_design") == "five_superpopulation"
        and identity.get("manifest_input_scope") == "legacy_chr20_22"
        and identity.get("orientation") == "samples_by_snps"
        and identity.get("external_source_manifest_sha256")
        == PREPARED_SOURCE_MANIFEST_SHA256
        and identity.get("expected_external_source_manifest_sha256")
        == PREPARED_SOURCE_MANIFEST_SHA256
        and int(identity.get("samples", 0)) == 1000
        and int(identity.get("snps", 0)) >= 3200
        and all(
            _is_sha256(identity.get(field))
            for field in (
                "external_source_manifest_sha256",
                "expected_external_source_manifest_sha256",
                "genotype_npz_sha256",
                "genotypes_sha256",
                "sample_ids_sha256",
                "snp_ids_sha256",
                "chromosomes_sha256",
                "population_labels_sha256",
            )
        )
    )


def _validate_input_identity(
    data: base.GenotypeInput, identity: Mapping[str, Any]
) -> None:
    expected = {
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "orientation": "samples_by_snps",
    }
    for field, value in expected.items():
        if identity.get(field) != value:
            raise ValueError(f"input identity disagrees with loaded data: {field}")


def _source_bundle_descriptor(source_identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return the logical identity of the exact 18-file execution bundle."""

    git_identity = source_identity.get("git_identity")
    files = source_identity.get("files")
    if (
        not isinstance(git_identity, Mapping)
        or not isinstance(files, Mapping)
        or set(files) != set(SOURCE_FILE_ROLES)
    ):
        raise ValueError("source identity cannot define the exact source bundle")
    descriptor = {
        "format": "logical_exact_18_file_source_bundle_v1",
        "head_commit": git_identity.get("head_commit"),
        "head_tree": git_identity.get("head_tree"),
        "source_manifest_sha256": source_identity.get("source_manifest_sha256"),
        "files": dict(files),
    }
    if (
        not _is_git_object_id(descriptor["head_commit"])
        or not _is_git_object_id(descriptor["head_tree"])
        or not _is_sha256(descriptor["source_manifest_sha256"])
        or any(not _is_sha256(value) for value in descriptor["files"].values())
    ):
        raise ValueError("source bundle descriptor contains an invalid identity")
    return descriptor


def _source_bundle_sha256(source_identity: Mapping[str, Any]) -> str:
    return canonical_sha256(_source_bundle_descriptor(source_identity))


def _load_formal_development_input(
    genotypes_path: Path,
    source_manifest_path: Path,
) -> tuple[base.GenotypeInput, dict[str, Any]]:
    """Load and authenticate the one prepared input permitted for formal use."""

    genotype_path = Path(genotypes_path)
    manifest_path = Path(source_manifest_path)
    data = base.load_genotype_npz(genotype_path)
    preparation = base.validate_external_source_manifest(
        manifest_path,
        genotype_path,
        data,
        expected_manifest_sha256=PREPARED_SOURCE_MANIFEST_SHA256,
    )
    input_identity: dict[str, Any] = {
        **preparation,
        "genotypes_sha256": array_sha256(data.genotypes),
        "sample_ids_sha256": array_sha256(data.sample_ids),
        "snp_ids_sha256": array_sha256(data.snp_ids),
        "chromosomes_sha256": array_sha256(data.chromosomes),
        "population_labels_sha256": array_sha256(data.population_labels),
        "samples": int(data.genotypes.shape[0]),
        "snps": int(data.genotypes.shape[1]),
        "orientation": "samples_by_snps",
        "ld_pruned_input_validated_from_external_manifest": True,
    }
    _validate_input_identity(data, input_identity)
    if not _input_is_lock_eligible(input_identity):
        raise ValueError(
            "formal development input must match the fixed prepared-manifest anchor"
        )
    return data, input_identity


def _split_receipt(
    *,
    seed: int,
    train: IntArray,
    test: IntArray,
    sample_ids: StringArray,
    labels: StringArray,
    retained_original: IntArray,
    retained_snp_ids: StringArray,
    mean_dosage: FloatArray,
    scale: FloatArray,
) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "train_indices_sha256": array_sha256(train),
        "test_indices_sha256": array_sha256(test),
        "train_indices_dtype": str(train.dtype),
        "test_indices_dtype": str(test.dtype),
        "train_sample_ids_sha256": array_sha256(sample_ids[train]),
        "test_sample_ids_sha256": array_sha256(sample_ids[test]),
        "train_sample_ids_dtype": str(sample_ids[train].dtype),
        "test_sample_ids_dtype": str(sample_ids[test].dtype),
        "test_labels_sha256": array_sha256(labels[test]),
        "test_labels_dtype": str(labels[test].dtype),
        "retained_original_indices_sha256": array_sha256(retained_original),
        "retained_snp_ids_sha256": array_sha256(retained_snp_ids),
        "retained_snps": int(retained_original.size),
        "preprocessor_mean_sha256": array_sha256(mean_dosage),
        "preprocessor_scale_sha256": array_sha256(scale),
    }


def _summaries(
    records: Sequence[Mapping[str, Any]],
    config: StratificationExtensionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grid_summaries: list[dict[str, Any]] = []
    population_summaries: list[dict[str, Any]] = []
    for n_components in config.component_grid:
        all_rows = [
            row
            for row in records
            if row["n_components"] == n_components and row["panel_label"] == "all"
        ]
        all_by_seed = {int(row["seed"]): row for row in all_rows}
        for panel_size in config.panel_sizes:
            panel_label = _panel_label(panel_size)
            rows = [
                row
                for row in records
                if row["n_components"] == n_components
                and row["panel_label"] == panel_label
            ]
            if len(rows) != len(config.seeds):
                raise RuntimeError("incomplete stratification grid")
            balanced = np.asarray(
                [row["balanced_accuracy"] for row in rows], dtype=float
            )
            deltas = np.asarray(
                [
                    row["balanced_accuracy"]
                    - all_by_seed[int(row["seed"])]["balanced_accuracy"]
                    for row in rows
                ],
                dtype=float,
            )
            grid_summaries.append(
                {
                    "n_components": int(n_components),
                    "panel_label": panel_label,
                    "requested_panel_size": panel_size,
                    "seeds": len(rows),
                    "minimum_selected_snp_count": int(
                        min(row["selected_snp_count"] for row in rows)
                    ),
                    "maximum_selected_snp_count": int(
                        max(row["selected_snp_count"] for row in rows)
                    ),
                    "mean_accuracy": float(
                        np.mean([row["accuracy"] for row in rows])
                    ),
                    "mean_balanced_accuracy": float(np.mean(balanced)),
                    "sample_std_balanced_accuracy": float(
                        np.std(balanced, ddof=1) if len(rows) > 1 else 0.0
                    ),
                    "minimum_balanced_accuracy": float(np.min(balanced)),
                    "maximum_balanced_accuracy": float(np.max(balanced)),
                    "mean_macro_f1": float(
                        np.mean([row["macro_f1"] for row in rows])
                    ),
                    "mean_paired_balanced_accuracy_delta_vs_same_rank_all": float(
                        np.mean(deltas)
                    ),
                    "minimum_paired_balanced_accuracy_delta_vs_same_rank_all": float(
                        np.min(deltas)
                    ),
                    "qualification_status": "closed_form_pca_all_splits",
                }
            )
            for population in EXPECTED_POPULATIONS:
                metrics = [
                    next(
                        item
                        for item in row["per_population_metrics"]
                        if item["population"] == population
                    )
                    for row in rows
                ]
                recalls = np.asarray([item["recall"] for item in metrics], dtype=float)
                population_summaries.append(
                    {
                        "n_components": int(n_components),
                        "panel_label": panel_label,
                        "requested_panel_size": panel_size,
                        "population": population,
                        "seeds": len(rows),
                        "pooled_support": int(sum(item["support"] for item in metrics)),
                        "pooled_true_positive": int(
                            sum(item["true_positive"] for item in metrics)
                        ),
                        "mean_recall": float(np.mean(recalls)),
                        "minimum_recall": float(np.min(recalls)),
                        "maximum_recall": float(np.max(recalls)),
                    }
                )
    return grid_summaries, population_summaries


def _panel_sets(
    ranking_records: Sequence[Mapping[str, Any]],
    config: StratificationExtensionConfig,
    n_components: int,
    panel_size: int | None,
) -> list[set[int]]:
    by_seed = {
        int(row["seed"]): row
        for row in ranking_records
        if int(row["n_components"]) == n_components
    }
    sets: list[set[int]] = []
    for seed in config.seeds:
        ordered = [int(value) for value in by_seed[seed]["ordered_original_indices"]]
        selected = ordered if panel_size is None else ordered[: int(panel_size)]
        sets.append(set(selected))
    return sets


def _stability_summaries(
    ranking_records: Sequence[Mapping[str, Any]],
    config: StratificationExtensionConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_components in config.component_grid:
        for panel_size in config.panel_sizes:
            sets = _panel_sets(ranking_records, config, n_components, panel_size)
            jaccards: list[float] = []
            for left_index in range(len(sets)):
                for right_index in range(left_index + 1, len(sets)):
                    union = sets[left_index] | sets[right_index]
                    jaccards.append(
                        float(len(sets[left_index] & sets[right_index]) / len(union))
                    )
            counts = Counter(value for selected in sets for value in selected)
            consensus_floor = math.ceil(0.8 * len(sets))
            rows.append(
                {
                    "n_components": int(n_components),
                    "panel_label": _panel_label(panel_size),
                    "requested_panel_size": panel_size,
                    "seeds": len(sets),
                    "seed_pairs": len(jaccards),
                    "minimum_selected_snp_count": min(len(value) for value in sets),
                    "maximum_selected_snp_count": max(len(value) for value in sets),
                    "intersection_snp_count": len(set.intersection(*sets)),
                    "union_snp_count": len(set.union(*sets)),
                    "snps_selected_in_at_least_80_percent_of_seeds": sum(
                        count >= consensus_floor for count in counts.values()
                    ),
                    "mean_pairwise_jaccard": float(np.mean(jaccards))
                    if jaccards
                    else 1.0,
                    "minimum_pairwise_jaccard": float(np.min(jaccards))
                    if jaccards
                    else 1.0,
                    "maximum_pairwise_jaccard": float(np.max(jaccards))
                    if jaccards
                    else 1.0,
                }
            )
    return rows


def _selection_decision(
    grid_summaries: Sequence[Mapping[str, Any]],
    population_summaries: Sequence[Mapping[str, Any]],
    config: StratificationExtensionConfig,
) -> dict[str, Any]:
    metric_map = {
        (int(row["n_components"]), str(row["panel_label"])): row
        for row in grid_summaries
    }
    population_map = {
        (
            int(row["n_components"]),
            str(row["panel_label"]),
            str(row["population"]),
        ): row
        for row in population_summaries
    }
    cells: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for panel_size in config.panel_sizes:
        panel_label = _panel_label(panel_size)
        for n_components in config.component_grid:
            summary = metric_map[(n_components, panel_label)]
            all_summary = metric_map[(n_components, "all")]
            recalls = {
                population: float(
                    population_map[(n_components, panel_label, population)][
                        "mean_recall"
                    ]
                )
                for population in EXPECTED_POPULATIONS
            }
            gates = {
                "balanced_accuracy_within_0_01_of_same_rank_all": (
                    float(summary["mean_balanced_accuracy"])
                    >= float(all_summary["mean_balanced_accuracy"])
                    - SELECTION_GATES[
                        "balanced_accuracy_noninferiority_margin_vs_same_rank_all"
                    ]
                    - 1e-15
                ),
                "mean_amr_recall_at_least_0_70": recalls["AMR"]
                >= SELECTION_GATES["minimum_mean_amr_recall"] - 1e-15,
                "mean_eur_recall_at_least_0_98": recalls["EUR"]
                >= SELECTION_GATES["minimum_mean_eur_recall"] - 1e-15,
                "mean_sas_recall_at_least_0_98": recalls["SAS"]
                >= SELECTION_GATES["minimum_mean_sas_recall"] - 1e-15,
            }
            selection_eligible = (
                panel_size is None
                or int(panel_size) >= MINIMUM_SELECTION_PANEL_SIZE
            )
            cell = {
                "n_components": int(n_components),
                "panel_label": panel_label,
                "requested_panel_size": panel_size,
                "mean_balanced_accuracy": float(summary["mean_balanced_accuracy"]),
                "same_rank_all_mean_balanced_accuracy": float(
                    all_summary["mean_balanced_accuracy"]
                ),
                "mean_balanced_accuracy_delta_vs_same_rank_all": float(
                    summary["mean_balanced_accuracy"]
                    - all_summary["mean_balanced_accuracy"]
                ),
                "mean_population_recall": recalls,
                "selection_eligible": bool(selection_eligible),
                "selection_exclusion_reason": (
                    None
                    if selection_eligible
                    else "top_400_is_compression_sensitivity_only"
                ),
                "gates": gates,
                "quality_gates_pass": all(gates.values()),
                "qualifies": selection_eligible and all(gates.values()),
            }
            cells.append(cell)
            if selected is None and cell["qualifies"]:
                selected = {
                    key: value
                    for key, value in cell.items()
                    if key
                    not in {
                        "gates",
                        "quality_gates_pass",
                        "qualifies",
                        "selection_eligible",
                        "selection_exclusion_reason",
                    }
                }
    return {
        "selection_status": "candidate_selected" if selected else "no_candidate",
        "rule_declared_before_development_execution": True,
        "candidate_order": "smallest_eligible_panel_from_800_then_smallest_rank",
        "top_400_role": "compression_sensitivity_only_not_selection_eligible",
        "all_grid_cells_disclosed": True,
        "gates": dict(SELECTION_GATES),
        "candidate_evaluations": cells,
        "selected_candidate": selected,
        "selection_is_post_hoc_development_not_confirmation": True,
    }


def _evaluate_stratification_extension_internal(
    data: base.GenotypeInput,
    config: StratificationExtensionConfig,
    *,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    _capability: object,
) -> dict[str, Any]:
    """Shared numerical core, reachable only through a profile capability."""

    base._validate_input(data)
    _validate_config(config)
    if _capability is _TINY_SMOKE_EVALUATION_CAPABILITY:
        if config.analysis_mode != "tiny_smoke":
            raise PermissionError(
                "tiny-smoke capability cannot execute the formal development protocol"
            )
    elif _capability is _FORMAL_DEVELOPMENT_EVALUATION_CAPABILITY:
        if not _protocol_exact(config):
            raise PermissionError(
                "formal capability requires the exact frozen development protocol"
            )
    else:
        raise PermissionError("missing private stratification evaluation capability")
    _validate_input_identity(data, input_identity)
    if config.analysis_mode == "development_protocol" and not _input_is_lock_eligible(
        input_identity
    ):
        raise ValueError(
            "formal development input must match the fixed prepared-manifest anchor"
        )
    raw = np.asarray(data.genotypes, dtype=float)
    labels = np.asarray(data.population_labels, dtype=str)
    sample_ids = np.asarray(data.sample_ids, dtype=str)
    max_components = max(config.component_grid)
    largest_finite_panel = max(
        value for value in config.panel_sizes if value is not None
    )
    records: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    coordinates: list[dict[str, Any]] = []
    sample_prediction_evidence: list[dict[str, Any]] = []
    split_receipts: list[dict[str, Any]] = []

    for seed in config.seeds:
        print(
            json.dumps({"event": "stratification_extension_seed_start", "seed": seed}),
            file=sys.stderr,
            flush=True,
        )
        train, test = stratified_split(labels, config.test_fraction, seed)
        preprocessor = fit_smartpca_preprocessor(raw[train])
        train_matrix = transform_smartpca(raw[train], preprocessor)
        test_matrix = transform_smartpca(raw[test], preprocessor)
        retained_original = np.flatnonzero(preprocessor.retained_snps).astype(np.int64)
        if retained_original.size < largest_finite_panel:
            raise ValueError("a requested panel exceeds train-retained SNP count")
        receipt = _split_receipt(
            seed=seed,
            train=train,
            test=test,
            sample_ids=sample_ids,
            labels=labels,
            retained_original=retained_original,
            retained_snp_ids=data.snp_ids[retained_original],
            mean_dosage=preprocessor.mean_dosage,
            scale=preprocessor.scale,
        )
        split_receipts.append(receipt)
        split_hashes = {
            field: receipt[field]
            for field in (
                "train_indices_sha256",
                "test_indices_sha256",
                "train_sample_ids_sha256",
                "test_sample_ids_sha256",
                "test_labels_sha256",
            )
        }

        full_loadings, _ = base.fit_top_pca(train_matrix, max_components)
        full_train_scores = train_matrix @ full_loadings
        full_test_scores = test_matrix @ full_loadings
        panel_fit_cache: dict[
            str, tuple[IntArray, FloatArray, FloatArray, FloatArray]
        ] = {}

        for n_components in config.component_grid:
            ranking_retained, importance = base.rank_by_pca_loadings(
                full_loadings, n_components
            )
            ranked_original = retained_original[ranking_retained]
            rankings.append(
                {
                    "seed": int(seed),
                    "n_components": int(n_components),
                    "available_train_retained_snps": int(retained_original.size),
                    "ordered_original_indices": ranked_original.tolist(),
                    "ordered_original_indices_sha256": array_sha256(ranked_original),
                    "loading_importance_sha256": array_sha256(importance),
                    "full_width_loading_prefix_sha256": array_sha256(
                        full_loadings[:, :n_components]
                    ),
                    "selection_rule": "descending_sum_abs_training_pca_loadings",
                    "tie_break": "retained_column_index_ascending",
                    "population_labels_used": False,
                    "test_rows_used": False,
                }
            )

            for panel_size in config.panel_sizes:
                if panel_size is None:
                    selected_original = np.sort(retained_original)
                    panel_loadings = full_loadings
                    panel_train_scores = full_train_scores
                    panel_test_scores = full_test_scores
                    reused = True
                else:
                    selected_retained = np.sort(
                        ranking_retained[: int(panel_size)]
                    ).astype(np.int64)
                    selected_original = np.sort(
                        retained_original[selected_retained]
                    ).astype(np.int64)
                    cache_key = array_sha256(selected_original)
                    if cache_key not in panel_fit_cache:
                        panel_train = train_matrix[:, selected_retained]
                        panel_test = test_matrix[:, selected_retained]
                        panel_loadings, _ = base.fit_top_pca(
                            panel_train, max_components
                        )
                        panel_fit_cache[cache_key] = (
                            selected_original,
                            panel_loadings,
                            panel_train @ panel_loadings,
                            panel_test @ panel_loadings,
                        )
                    (
                        cached_original,
                        panel_loadings,
                        panel_train_scores,
                        panel_test_scores,
                    ) = panel_fit_cache[cache_key]
                    if not np.array_equal(cached_original, selected_original):
                        raise RuntimeError("panel-fit cache identity mismatch")
                    reused = False

                record, predictions, centroids = base._record(
                    seed=seed,
                    split_hashes=split_hashes,
                    evaluation="loading_panel",
                    n_components=n_components,
                    panel_size=panel_size,
                    selected_original=selected_original,
                    retained_original=retained_original,
                    loadings=panel_loadings,
                    train_scores=panel_train_scores,
                    test_scores=panel_test_scores,
                    train_labels=labels[train],
                    test_labels=labels[test],
                    reused_full_width_fit=reused,
                )
                record["test_predictions_dtype"] = str(predictions.dtype)
                records.append(record)

                panel_label = _panel_label(panel_size)
                for local_index, sample_index in enumerate(test.tolist()):
                    sample_prediction_evidence.append(
                        {
                            "seed": int(seed),
                            "n_components": int(n_components),
                            "panel_label": panel_label,
                            "requested_panel_size": panel_size,
                            "selected_snp_count": int(selected_original.size),
                            "selected_original_indices_sha256": record[
                                "selected_original_indices_sha256"
                            ],
                            "sample_position": int(local_index),
                            "sample_index": int(sample_index),
                            "sample_id": str(sample_ids[sample_index]),
                            "truth": str(labels[sample_index]),
                            "prediction": str(predictions[local_index]),
                            "correct": bool(
                                predictions[local_index] == labels[sample_index]
                            ),
                            "train_indices_sha256": receipt[
                                "train_indices_sha256"
                            ],
                            "test_indices_sha256": receipt["test_indices_sha256"],
                            "test_indices_dtype": receipt["test_indices_dtype"],
                            "train_sample_ids_sha256": receipt[
                                "train_sample_ids_sha256"
                            ],
                            "test_sample_ids_sha256": receipt[
                                "test_sample_ids_sha256"
                            ],
                            "test_sample_ids_dtype": receipt[
                                "test_sample_ids_dtype"
                            ],
                            "test_labels_sha256": receipt["test_labels_sha256"],
                            "test_labels_dtype": receipt["test_labels_dtype"],
                            "test_predictions_sha256": record[
                                "test_predictions_sha256"
                            ],
                            "test_predictions_dtype": record[
                                "test_predictions_dtype"
                            ],
                        }
                    )

                if seed == config.seeds[0]:
                    for local_index, sample_index in enumerate(test.tolist()):
                        coordinates.append(
                            {
                                "row_type": "held_out_sample",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": panel_label,
                                "requested_panel_size": panel_size,
                                "selected_snp_count": int(selected_original.size),
                                "sample_index": int(sample_index),
                                "sample_id": str(sample_ids[sample_index]),
                                "population": str(labels[sample_index]),
                                "truth": str(labels[sample_index]),
                                "prediction": str(predictions[local_index]),
                                "correct": bool(
                                    predictions[local_index] == labels[sample_index]
                                ),
                                **{
                                    f"component_{component + 1}": float(
                                        panel_test_scores[local_index, component]
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )
                    classes = np.asarray(EXPECTED_POPULATIONS, dtype=str)
                    if not np.array_equal(classes, np.unique(labels[train])):
                        raise RuntimeError("unexpected population order")
                    for class_index, population in enumerate(classes.tolist()):
                        coordinates.append(
                            {
                                "row_type": "train_centroid",
                                "seed": int(seed),
                                "n_components": int(n_components),
                                "panel_label": panel_label,
                                "requested_panel_size": panel_size,
                                "selected_snp_count": int(selected_original.size),
                                "sample_index": None,
                                "sample_id": None,
                                "population": population,
                                "truth": None,
                                "prediction": None,
                                "correct": None,
                                **{
                                    f"component_{component + 1}": float(
                                        centroids[class_index, component]
                                        if component < n_components
                                        else panel_train_scores[
                                            labels[train] == population, component
                                        ].mean()
                                    )
                                    for component in range(max_components)
                                },
                            }
                        )

        print(
            json.dumps(
                {
                    "event": "stratification_extension_seed_complete",
                    "seed": seed,
                    "full_width_pca_fits": 1,
                    "finite_panel_pca_fits": len(panel_fit_cache),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    grid_summaries, population_summaries = _summaries(records, config)
    stability_summaries = _stability_summaries(rankings, config)
    selection = _selection_decision(grid_summaries, population_summaries, config)
    lock_eligible = (
        _protocol_exact(config)
        and _input_is_lock_eligible(input_identity)
        and selection["selected_candidate"] is not None
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_policy": {
            "status": "post_hoc_development_selection",
            "confirmatory": False,
            "all_grid_cells_reported": True,
            "population_labels_are_race": False,
            "population_labels_used_for_split": True,
            "population_labels_used_for_training_centroids": True,
            "population_labels_used_for_held_out_scoring": True,
            "population_labels_used_for_preprocessing_pca_or_loading_ranking": False,
            "test_rows_used_for_preprocessing_pca_or_loading_ranking": False,
            "selection_rule": (
                "predeclared_smallest_eligible_panel_from_800_then_smallest_rank"
            ),
            "multi_pc_interpretation": (
                "classification uses all selected PCs; no single 2D projection can "
                "summarize separation in 4/6/8 dimensions"
            ),
        },
        "config": asdict(config),
        "method_scope": dict(METHOD_SCOPE),
        "input_identity": dict(input_identity),
        "source_identity": dict(source_identity),
        "runtime_versions": runtime_versions(),
        "split_receipts": split_receipts,
        "ranking_records": rankings,
        "records": records,
        "grid_summaries": grid_summaries,
        "population_summaries": population_summaries,
        "panel_stability_summaries": stability_summaries,
        "sample_prediction_evidence": sample_prediction_evidence,
        "selection_decision": selection,
        "confirmation_proposal_policy": {
            "eligible": bool(lock_eligible),
            "proposal_only_not_executed": True,
            "fresh_confirmation_seeds": list(CONFIRMATION_SEEDS),
            "confirmation_outcomes_accessed": False,
            "independent_source_and_protocol_audit_required_before_execution": True,
        },
        "registered_seed_policy": {
            "seed": int(config.seeds[0]),
            "selection": "first_configured_seed_never_best_seed",
            "rank": int(
                selection["selected_candidate"]["n_components"]
                if selection["selected_candidate"] is not None
                else max(config.component_grid)
            ),
            "rank_policy": (
                "selected_candidate_rank_or_max_rank_if_no_candidate"
            ),
            "plotted_component_pairs": [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ],
            "panels": [_panel_label(value) for value in config.panel_sizes],
        },
        "registered_coordinates": coordinates,
    }
    payload["analysis_sha256"] = canonical_sha256(payload)
    validate_result_payload(payload)
    return payload


def evaluate_stratification_extension(
    data: base.GenotypeInput,
    config: StratificationExtensionConfig,
    *,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a tiny structural smoke only; formal work must use authenticated paths."""

    if config.analysis_mode != "tiny_smoke":
        raise PermissionError(
            "public in-memory evaluation is restricted to tiny_smoke; "
            "use evaluate_formal_development_from_paths for formal execution"
        )
    return _evaluate_stratification_extension_internal(
        data,
        config,
        input_identity=input_identity,
        source_identity=source_identity,
        _capability=_TINY_SMOKE_EVALUATION_CAPABILITY,
    )


def evaluate_formal_development_from_paths(
    genotypes_path: Path,
    source_manifest_path: Path,
) -> dict[str, Any]:
    """Run the frozen development protocol from the fixed authenticated input."""

    data, input_identity = _load_formal_development_input(
        Path(genotypes_path), Path(source_manifest_path)
    )
    return _evaluate_stratification_extension_internal(
        data,
        StratificationExtensionConfig(),
        input_identity=input_identity,
        source_identity=_source_identity(_runner_repo_root()),
        _capability=_FORMAL_DEVELOPMENT_EVALUATION_CAPABILITY,
    )


def _candidate_map(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    candidate = payload["selection_decision"]["selected_candidate"]
    if candidate is not None:
        return candidate
    return {
        "n_components": max(payload["config"]["component_grid"]),
        "panel_label": "all",
        "requested_panel_size": None,
    }


def _build_lock_proposal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload["confirmation_proposal_policy"]["eligible"] is not True:
        raise ValueError("payload is not eligible for a confirmation lock proposal")
    selected = dict(payload["selection_decision"]["selected_candidate"])
    proposal: dict[str, Any] = {
        "schema_version": 1,
        "status": "proposal_ready_for_independent_audit_not_frozen_or_executed",
        "development_analysis_sha256": payload["analysis_sha256"],
        "development_source_manifest_sha256": payload["source_identity"][
            "source_manifest_sha256"
        ],
        "prepared_input_identity": dict(payload["input_identity"]),
        "selection_rule": {
            "candidate_order": (
                "smallest_eligible_panel_from_800_then_smallest_rank"
            ),
            "top_400_role": (
                "compression_sensitivity_only_not_selection_eligible"
            ),
            "gates": dict(SELECTION_GATES),
            "selected_candidate": selected,
        },
        "proposed_confirmation_protocol": {
            "seeds": list(CONFIRMATION_SEEDS),
            "test_fraction": 0.2,
            "same_prepared_chr20_22_five_superpopulation_input": True,
            "same_train_only_preprocessing_ranking_and_panel_refit": True,
            "classifier": "nearest_outer_train_centroid",
            "held_out_use": "scoring_only",
            "compare_to_same_rank_all_snp": True,
            "confirmation_outcomes_accessed": False,
        },
        "proposed_claim_gates": {
            "candidate_mean_balanced_accuracy_at_least_same_rank_all_minus_0_01": True,
            "candidate_mean_amr_recall_at_least_0_70": True,
            "candidate_mean_eur_recall_at_least_0_98": True,
            "candidate_mean_sas_recall_at_least_0_98": True,
            "closed_form_pca_all_splits": True,
            "overall": "all_five_gates",
        },
        "method_scope": dict(METHOD_SCOPE),
        "execution_authorized": False,
        "independent_audit_required": True,
    }
    proposal["proposal_payload_sha256"] = canonical_sha256(proposal)
    return proposal


def validate_lock_proposal(
    proposal: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    expected_development_analysis_sha256: str,
) -> dict[str, Any]:
    validate_result_payload(payload)
    if (
        not _is_sha256(expected_development_analysis_sha256)
        or expected_development_analysis_sha256 != payload["analysis_sha256"]
    ):
        raise ValueError("expected development analysis SHA-256 mismatch")
    if payload["confirmation_proposal_policy"]["eligible"] is not True:
        raise ValueError("an ineligible payload must not have a lock proposal")
    expected = _build_lock_proposal(payload)
    if dict(proposal) != expected:
        raise ValueError("lock proposal must exactly equal the payload-derived proposal")
    return {
        "status": "valid",
        "proposal_payload_sha256": expected["proposal_payload_sha256"],
    }


def _summary_maps(
    payload: Mapping[str, Any],
) -> tuple[dict[tuple[int, str], Mapping[str, Any]], dict[tuple[int, str, str], Mapping[str, Any]]]:
    summaries = {
        (int(row["n_components"]), str(row["panel_label"])): row
        for row in payload["grid_summaries"]
    }
    populations = {
        (
            int(row["n_components"]),
            str(row["panel_label"]),
            str(row["population"]),
        ): row
        for row in payload["population_summaries"]
    }
    return summaries, populations


def validate_result_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    required_top = {
        "schema_version",
        "analysis_policy",
        "config",
        "method_scope",
        "input_identity",
        "source_identity",
        "runtime_versions",
        "split_receipts",
        "ranking_records",
        "records",
        "grid_summaries",
        "population_summaries",
        "panel_stability_summaries",
        "sample_prediction_evidence",
        "selection_decision",
        "confirmation_proposal_policy",
        "registered_seed_policy",
        "registered_coordinates",
        "analysis_sha256",
    }
    if set(payload) != required_top or payload.get("schema_version") != 1:
        raise ValueError("result payload has an unexpected top-level schema")
    base._reject_nonfinite(payload)
    policy = payload["analysis_policy"]
    if (
        policy.get("status") != "post_hoc_development_selection"
        or policy.get("confirmatory") is not False
        or policy.get("all_grid_cells_reported") is not True
        or policy.get(
            "population_labels_used_for_preprocessing_pca_or_loading_ranking"
        )
        is not False
        or policy.get("test_rows_used_for_preprocessing_pca_or_loading_ranking")
        is not False
        or policy.get("selection_rule")
        != "predeclared_smallest_eligible_panel_from_800_then_smallest_rank"
        or "no single 2D projection" not in policy.get("multi_pc_interpretation", "")
    ):
        raise ValueError("analysis-policy boundary is invalid")
    if payload["method_scope"] != METHOD_SCOPE:
        raise ValueError("method scope is not PCA-loading-only")
    source = payload["source_identity"]
    git_identity = source.get("git_identity")
    git_required = {
        "head_commit",
        "head_tree",
        "remotes",
        "tracked_dirty",
        "tracked_dirty_entries",
        "tracked_dirty_entries_sha256",
        "untracked",
        "untracked_paths",
        "untracked_paths_sha256",
        "worktree_dirty_including_untracked",
        "repo_root_binding",
        "git_identity_sha256",
    }
    if (
        source.get("method_scope") != METHOD_SCOPE
        or source.get("existing_formal_or_grid_runner_modified") is not False
        or source.get("fed_or_fedrpca_scientific_code_used") is not False
        or set(source.get("files", {})) != set(SOURCE_FILE_ROLES)
        or source.get("file_roles") != SOURCE_FILE_ROLES
        or source.get("loaded_repo_python_files_exact_cover") is not True
        or source.get("loaded_repo_python_files") != sorted(SOURCE_FILE_ROLES)
        or source.get("loaded_repo_python_files_sha256")
        != canonical_sha256(
            {"loaded_repo_python_files": sorted(SOURCE_FILE_ROLES)}
        )
        or source.get("source_manifest_sha256")
        != canonical_sha256(
            {
                "files": source.get("files"),
                "file_roles": SOURCE_FILE_ROLES,
                "loaded_repo_python_files": source.get(
                    "loaded_repo_python_files"
                ),
                "git_identity": source.get("git_identity"),
            }
        )
        or any(not _is_sha256(value) for value in source.get("files", {}).values())
        or not isinstance(git_identity, Mapping)
        or set(git_identity) != git_required
        or git_identity.get("repo_root_binding")
        != "resolved_runner_parent_equals_git_toplevel"
        or not _is_sha256(git_identity.get("tracked_dirty_entries_sha256"))
        or git_identity.get("tracked_dirty_entries_sha256")
        != canonical_sha256(
            {"tracked_dirty_entries": git_identity.get("tracked_dirty_entries")}
        )
        or not _is_sha256(git_identity.get("untracked_paths_sha256"))
        or git_identity.get("untracked_paths_sha256")
        != canonical_sha256({"untracked_paths": git_identity.get("untracked_paths")})
        or not _is_sha256(git_identity.get("git_identity_sha256"))
        or git_identity.get("git_identity_sha256")
        != canonical_sha256(
            {key: value for key, value in git_identity.items() if key != "git_identity_sha256"}
        )
        or git_identity.get("tracked_dirty")
        is not bool(git_identity.get("tracked_dirty_entries"))
        or git_identity.get("untracked") is not bool(git_identity.get("untracked_paths"))
        or git_identity.get("worktree_dirty_including_untracked")
        is not bool(
            git_identity.get("tracked_dirty_entries")
            or git_identity.get("untracked_paths")
        )
        or not _is_git_object_id(git_identity.get("head_commit"))
        or not _is_git_object_id(git_identity.get("head_tree"))
        or not isinstance(git_identity.get("remotes"), list)
        or any(
            not isinstance(remote, Mapping)
            or set(remote) != {"name", "urls"}
            or not isinstance(remote["name"], str)
            or not remote["name"]
            or not isinstance(remote["urls"], list)
            or not remote["urls"]
            or any(not isinstance(url, str) or not url for url in remote["urls"])
            for remote in git_identity.get("remotes", [])
        )
        or not isinstance(git_identity.get("tracked_dirty_entries"), list)
        or git_identity.get("tracked_dirty_entries")
        != sorted(set(git_identity.get("tracked_dirty_entries", [])))
        or not isinstance(git_identity.get("untracked_paths"), list)
        or git_identity.get("untracked_paths")
        != sorted(set(git_identity.get("untracked_paths", [])))
    ):
        raise ValueError("source closure or method exclusivity mismatch")

    config = StratificationExtensionConfig(
        seeds=tuple(int(value) for value in payload["config"]["seeds"]),
        component_grid=tuple(
            int(value) for value in payload["config"]["component_grid"]
        ),
        panel_sizes=tuple(
            None if value is None else int(value)
            for value in payload["config"]["panel_sizes"]
        ),
        test_fraction=float(payload["config"]["test_fraction"]),
        loading_score=str(payload["config"]["loading_score"]),
        population_design=str(payload["config"]["population_design"]),
        analysis_status=str(payload["config"]["analysis_status"]),
        analysis_mode=str(payload["config"]["analysis_mode"]),
    )
    if set(payload["config"]) != set(asdict(config)):
        raise ValueError("config schema mismatch")
    _validate_config(config)
    if config.analysis_mode == "development_protocol" and not _input_is_lock_eligible(
        payload["input_identity"]
    ):
        raise ValueError("formal result input does not match the fixed manifest anchor")

    receipt_map: dict[int, Mapping[str, Any]] = {}
    receipt_hash_fields = {
        "train_indices_sha256",
        "test_indices_sha256",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_labels_sha256",
        "retained_original_indices_sha256",
        "retained_snp_ids_sha256",
        "preprocessor_mean_sha256",
        "preprocessor_scale_sha256",
    }
    receipt_fields = receipt_hash_fields | {"seed", "retained_snps"}
    receipt_dtype_fields = {
        "train_indices_dtype",
        "test_indices_dtype",
        "train_sample_ids_dtype",
        "test_sample_ids_dtype",
        "test_labels_dtype",
    }
    receipt_fields |= receipt_dtype_fields
    for receipt in payload["split_receipts"]:
        if set(receipt) != receipt_fields:
            raise ValueError("split receipt schema mismatch")
        seed = int(receipt["seed"])
        if seed in receipt_map or seed not in config.seeds:
            raise ValueError("duplicate or unexpected split receipt")
        if int(receipt["retained_snps"]) < max(
            value for value in config.panel_sizes if value is not None
        ) or any(not _is_sha256(receipt[field]) for field in receipt_hash_fields):
            raise ValueError("invalid split receipt")
        if any(
            not isinstance(receipt[field], str) or not receipt[field]
            for field in receipt_dtype_fields
        ):
            raise ValueError("invalid split receipt dtype identity")
        receipt_map[seed] = receipt
    if set(receipt_map) != set(config.seeds):
        raise ValueError("split receipts do not exactly cover configured seeds")

    ranking_map: dict[tuple[int, int], Mapping[str, Any]] = {}
    expected_ranking_keys = {
        (seed, rank) for seed in config.seeds for rank in config.component_grid
    }
    for ranking in payload["ranking_records"]:
        key = (int(ranking["seed"]), int(ranking["n_components"]))
        if key in ranking_map:
            raise ValueError("duplicate ranking record")
        if (
            ranking.get("population_labels_used") is not False
            or ranking.get("test_rows_used") is not False
            or ranking.get("selection_rule")
            != "descending_sum_abs_training_pca_loadings"
            or ranking.get("tie_break") != "retained_column_index_ascending"
        ):
            raise ValueError("ranking leakage boundary mismatch")
        ordered = np.asarray(ranking["ordered_original_indices"], dtype=np.int64)
        receipt = receipt_map[key[0]]
        if (
            ordered.ndim != 1
            or ordered.size != int(receipt["retained_snps"])
            or ordered.size != int(ranking["available_train_retained_snps"])
            or np.unique(ordered).size != ordered.size
            or np.any(ordered < 0)
            or array_sha256(ordered) != ranking["ordered_original_indices_sha256"]
            or array_sha256(np.sort(ordered))
            != receipt["retained_original_indices_sha256"]
            or not _is_sha256(ranking.get("loading_importance_sha256"))
            or not _is_sha256(ranking.get("full_width_loading_prefix_sha256"))
        ):
            raise ValueError("ranking permutation or hash mismatch")
        ranking_map[key] = ranking
    if set(ranking_map) != expected_ranking_keys:
        raise ValueError("ranking records do not exactly cover the grid")

    expected_record_keys = {
        (seed, rank, _panel_label(panel))
        for seed in config.seeds
        for rank in config.component_grid
        for panel in config.panel_sizes
    }
    record_map: dict[tuple[int, int, str], Mapping[str, Any]] = {}
    support_by_seed: dict[int, tuple[int, ...]] = {}
    for record in payload["records"]:
        key = (
            int(record["seed"]),
            int(record["n_components"]),
            str(record["panel_label"]),
        )
        if key in record_map:
            raise ValueError("duplicate grid record")
        if (
            record.get("evaluation") != "loading_panel"
            or record.get("solver") != "closed_form_pca_smaller_gram_eigh"
            or record.get("solver_converged") is not True
            or record.get("solver_n_iter") != 1
            or record.get("selected_snp_set_is_loading_only") is not True
            or record.get("population_labels_used_for_panel_ranking") is not False
            or record.get("test_used_for_preprocessing_pca_or_ranking") is not False
            or record.get("preprocessing_scope") != "outer_train_only"
            or record.get("pca_fit_scope") != "outer_train_only"
            or record.get("panel_refit_scope") != "outer_train_only"
            or record.get("classifier_fit_scope")
            != "outer_train_centroids_after_panel_freeze"
        ):
            raise ValueError("record violates the PCA-only fitting boundary")
        receipt = receipt_map[key[0]]
        for field in (
            "train_indices_sha256",
            "test_indices_sha256",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_labels_sha256",
        ):
            if record.get(field) != receipt[field]:
                raise ValueError(f"record {field} disagrees with split receipt")
        ordered = np.asarray(
            ranking_map[(key[0], key[1])]["ordered_original_indices"],
            dtype=np.int64,
        )
        requested = record["requested_panel_size"]
        if key[2] == "all":
            if requested is not None or record.get("full_width_fit_reused") is not True:
                raise ValueError("all-SNP replay boundary mismatch")
            selected = np.sort(ordered)
        else:
            if requested != int(key[2]) or record.get("full_width_fit_reused") is not False:
                raise ValueError("finite-panel metadata mismatch")
            selected = np.sort(ordered[: int(requested)])
        if (
            int(record["selected_snp_count"]) != selected.size
            or record["selected_original_indices_sha256"] != array_sha256(selected)
        ):
            raise ValueError("selected SNP panel does not match the train-only ranking")
        if record.get("class_order") != list(EXPECTED_POPULATIONS):
            raise ValueError("class order mismatch")
        raw_confusion = record.get("confusion_matrix_counts")
        if (
            not isinstance(raw_confusion, list)
            or len(raw_confusion) != 5
            or any(
                not isinstance(row, list)
                or len(row) != 5
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))
                    or int(value) < 0
                    for value in row
                )
                for row in raw_confusion
            )
        ):
            raise ValueError("confusion counts must be a non-negative integer 5x5 matrix")
        confusion = np.asarray(raw_confusion, dtype=np.int64)
        supports = tuple(int(value) for value in confusion.sum(axis=1))
        if any(value <= 0 for value in supports):
            raise ValueError("all populations require held-out support")
        prior = support_by_seed.setdefault(key[0], supports)
        if prior != supports:
            raise ValueError("held-out population supports differ within a seed")
        derived = base._derived_metrics(record)
        reported = tuple(
            float(record[field])
            for field in ("accuracy", "balanced_accuracy", "macro_f1")
        )
        if not np.allclose(derived, reported, atol=1e-14, rtol=0.0):
            raise ValueError("record metrics disagree with confusion counts")
        population_rows = record.get("per_population_metrics")
        if [row.get("population") for row in population_rows] != list(
            EXPECTED_POPULATIONS
        ):
            raise ValueError("per-population record order mismatch")
        for index, row in enumerate(population_rows):
            support = int(confusion[index].sum())
            predicted = int(confusion[:, index].sum())
            true_positive = int(confusion[index, index])
            recall = true_positive / support
            precision = true_positive / predicted if predicted else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
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
            for field, value in expected.items():
                observed = row.get(field)
                if isinstance(value, float):
                    if not math.isclose(float(observed), value, abs_tol=1e-14):
                        raise ValueError("per-population metrics are not confusion-derived")
                elif observed != value:
                    raise ValueError("per-population counts are not confusion-derived")
        for field in (
            "selected_original_indices_sha256",
            "loadings_sha256",
            "test_predictions_sha256",
        ):
            if not _is_sha256(record.get(field)):
                raise ValueError(f"invalid record hash: {field}")
        if (
            not isinstance(record.get("test_predictions_dtype"), str)
            or not record["test_predictions_dtype"]
            or np.dtype(record["test_predictions_dtype"]).hasobject
        ):
            raise ValueError("record prediction dtype identity is invalid")
        record_map[key] = record
    if set(record_map) != expected_record_keys:
        raise ValueError("records do not exactly cover the grid")

    evidence_fields = {
        "seed",
        "n_components",
        "panel_label",
        "requested_panel_size",
        "selected_snp_count",
        "selected_original_indices_sha256",
        "sample_position",
        "sample_index",
        "sample_id",
        "truth",
        "prediction",
        "correct",
        "train_indices_sha256",
        "test_indices_sha256",
        "test_indices_dtype",
        "train_sample_ids_sha256",
        "test_sample_ids_sha256",
        "test_sample_ids_dtype",
        "test_labels_sha256",
        "test_labels_dtype",
        "test_predictions_sha256",
        "test_predictions_dtype",
    }
    evidence_groups: dict[
        tuple[int, int, str], list[Mapping[str, Any]]
    ] = {}
    for row in payload["sample_prediction_evidence"]:
        if set(row) != evidence_fields:
            raise ValueError("sample-prediction evidence schema mismatch")
        if any(
            isinstance(row[field], bool) or not isinstance(row[field], int)
            for field in ("seed", "n_components", "sample_position", "sample_index")
        ):
            raise ValueError("sample-prediction index fields must be integers")
        key = (
            int(row["seed"]),
            int(row["n_components"]),
            str(row["panel_label"]),
        )
        if key not in record_map:
            raise ValueError("sample-prediction evidence references an unknown grid cell")
        evidence_groups.setdefault(key, []).append(row)
    if set(evidence_groups) != expected_record_keys:
        raise ValueError("sample-prediction evidence does not cover every grid cell")

    for key, unsorted_rows in evidence_groups.items():
        rows = sorted(unsorted_rows, key=lambda row: int(row["sample_position"]))
        record = record_map[key]
        receipt = receipt_map[key[0]]
        expected_count = int(np.asarray(record["confusion_matrix_counts"]).sum())
        if [int(row["sample_position"]) for row in rows] != list(
            range(expected_count)
        ):
            raise ValueError("sample-prediction positions are not an exact ordered cover")
        if len({int(row["sample_index"]) for row in rows}) != expected_count:
            raise ValueError("sample-prediction sample indices are not unique")
        for row in rows:
            expected_row_values = {
                "requested_panel_size": record["requested_panel_size"],
                "selected_snp_count": record["selected_snp_count"],
                "selected_original_indices_sha256": record[
                    "selected_original_indices_sha256"
                ],
                "train_indices_sha256": receipt["train_indices_sha256"],
                "test_indices_sha256": receipt["test_indices_sha256"],
                "test_indices_dtype": receipt["test_indices_dtype"],
                "train_sample_ids_sha256": receipt["train_sample_ids_sha256"],
                "test_sample_ids_sha256": receipt["test_sample_ids_sha256"],
                "test_sample_ids_dtype": receipt["test_sample_ids_dtype"],
                "test_labels_sha256": receipt["test_labels_sha256"],
                "test_labels_dtype": receipt["test_labels_dtype"],
                "test_predictions_sha256": record["test_predictions_sha256"],
                "test_predictions_dtype": record["test_predictions_dtype"],
            }
            if any(row.get(field) != value for field, value in expected_row_values.items()):
                raise ValueError("sample-prediction split or panel evidence mismatch")
            if (
                row["truth"] not in EXPECTED_POPULATIONS
                or row["prediction"] not in EXPECTED_POPULATIONS
                or row["correct"] is not (row["truth"] == row["prediction"])
            ):
                raise ValueError("sample-prediction truth/prediction evidence mismatch")
        try:
            indices_dtype = np.dtype(receipt["test_indices_dtype"])
            sample_ids_dtype = np.dtype(receipt["test_sample_ids_dtype"])
            labels_dtype = np.dtype(receipt["test_labels_dtype"])
            predictions_dtype = np.dtype(record["test_predictions_dtype"])
        except TypeError as error:
            raise ValueError("sample-prediction evidence contains an invalid dtype") from error
        if any(
            dtype.hasobject
            for dtype in (
                indices_dtype,
                sample_ids_dtype,
                labels_dtype,
                predictions_dtype,
            )
        ):
            raise ValueError("object dtype is forbidden in sample-prediction evidence")
        test_indices = np.asarray(
            [int(row["sample_index"]) for row in rows], dtype=indices_dtype
        )
        test_sample_ids = np.asarray(
            [str(row["sample_id"]) for row in rows], dtype=sample_ids_dtype
        )
        truth = np.asarray([str(row["truth"]) for row in rows], dtype=labels_dtype)
        predictions = np.asarray(
            [str(row["prediction"]) for row in rows], dtype=predictions_dtype
        )
        if (
            array_sha256(test_indices) != receipt["test_indices_sha256"]
            or array_sha256(test_sample_ids) != receipt["test_sample_ids_sha256"]
            or array_sha256(truth) != receipt["test_labels_sha256"]
            or array_sha256(predictions) != record["test_predictions_sha256"]
        ):
            raise ValueError("sample-prediction arrays disagree with split/prediction hashes")
        derived = base._classification_metrics(
            truth,
            predictions,
            np.asarray(EXPECTED_POPULATIONS, dtype=str),
        )
        if (
            derived["confusion_matrix_counts"] != record["confusion_matrix_counts"]
            or derived["class_order"] != record["class_order"]
            or derived["per_population_metrics"] != record["per_population_metrics"]
            or any(
                not math.isclose(
                    float(derived[field]), float(record[field]), abs_tol=1e-14
                )
                for field in ("accuracy", "balanced_accuracy", "macro_f1")
            )
        ):
            raise ValueError(
                "record confusion/metrics are not derivable from sample predictions"
            )

    expected_summary_keys = {
        (rank, _panel_label(panel))
        for rank in config.component_grid
        for panel in config.panel_sizes
    }
    summary_map, population_map = _summary_maps(payload)
    if set(summary_map) != expected_summary_keys or len(summary_map) != len(
        payload["grid_summaries"]
    ):
        raise ValueError("grid summaries are incomplete or duplicated")
    for (rank, panel_label), summary in summary_map.items():
        rows = [record_map[(seed, rank, panel_label)] for seed in config.seeds]
        all_rows = [record_map[(seed, rank, "all")] for seed in config.seeds]
        balanced = np.asarray([row["balanced_accuracy"] for row in rows], dtype=float)
        deltas = np.asarray(
            [
                row["balanced_accuracy"] - all_row["balanced_accuracy"]
                for row, all_row in zip(rows, all_rows, strict=True)
            ],
            dtype=float,
        )
        expected = {
            "seeds": len(rows),
            "minimum_selected_snp_count": min(
                int(row["selected_snp_count"]) for row in rows
            ),
            "maximum_selected_snp_count": max(
                int(row["selected_snp_count"]) for row in rows
            ),
            "mean_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "mean_balanced_accuracy": float(np.mean(balanced)),
            "sample_std_balanced_accuracy": float(
                np.std(balanced, ddof=1) if len(rows) > 1 else 0.0
            ),
            "minimum_balanced_accuracy": float(np.min(balanced)),
            "maximum_balanced_accuracy": float(np.max(balanced)),
            "mean_macro_f1": float(np.mean([row["macro_f1"] for row in rows])),
            "mean_paired_balanced_accuracy_delta_vs_same_rank_all": float(
                np.mean(deltas)
            ),
            "minimum_paired_balanced_accuracy_delta_vs_same_rank_all": float(
                np.min(deltas)
            ),
        }
        for field, value in expected.items():
            observed = summary.get(field)
            if isinstance(value, float):
                if not math.isclose(float(observed), value, abs_tol=1e-14):
                    raise ValueError(f"grid summary {field} is not record-derived")
            elif observed != value:
                raise ValueError(f"grid summary {field} is not record-derived")
        if summary.get("qualification_status") != "closed_form_pca_all_splits":
            raise ValueError("grid summary solver status mismatch")

    expected_population_keys = {
        (rank, _panel_label(panel), population)
        for rank in config.component_grid
        for panel in config.panel_sizes
        for population in EXPECTED_POPULATIONS
    }
    if set(population_map) != expected_population_keys or len(population_map) != len(
        payload["population_summaries"]
    ):
        raise ValueError("population summaries are incomplete or duplicated")
    for (rank, panel_label, population), summary in population_map.items():
        metrics = [
            next(
                row
                for row in record_map[(seed, rank, panel_label)][
                    "per_population_metrics"
                ]
                if row["population"] == population
            )
            for seed in config.seeds
        ]
        recalls = np.asarray([row["recall"] for row in metrics], dtype=float)
        expected = {
            "seeds": len(metrics),
            "pooled_support": sum(int(row["support"]) for row in metrics),
            "pooled_true_positive": sum(
                int(row["true_positive"]) for row in metrics
            ),
            "mean_recall": float(np.mean(recalls)),
            "minimum_recall": float(np.min(recalls)),
            "maximum_recall": float(np.max(recalls)),
        }
        for field, value in expected.items():
            observed = summary.get(field)
            if isinstance(value, float):
                if not math.isclose(float(observed), value, abs_tol=1e-14):
                    raise ValueError("population summary is not record-derived")
            elif observed != value:
                raise ValueError("population summary is not record-derived")

    derived_stability = _stability_summaries(payload["ranking_records"], config)
    if payload["panel_stability_summaries"] != derived_stability:
        raise ValueError("panel stability summaries are not ranking-derived")
    derived_selection = _selection_decision(
        payload["grid_summaries"], payload["population_summaries"], config
    )
    if payload["selection_decision"] != derived_selection:
        raise ValueError("selection decision does not follow the predeclared rule")
    expected_eligible = (
        _protocol_exact(config)
        and _input_is_lock_eligible(payload["input_identity"])
        and derived_selection["selected_candidate"] is not None
    )
    proposal_policy = payload["confirmation_proposal_policy"]
    if (
        proposal_policy.get("eligible") is not expected_eligible
        or proposal_policy.get("proposal_only_not_executed") is not True
        or proposal_policy.get("fresh_confirmation_seeds")
        != list(CONFIRMATION_SEEDS)
        or proposal_policy.get("confirmation_outcomes_accessed") is not False
        or proposal_policy.get(
            "independent_source_and_protocol_audit_required_before_execution"
        )
        is not True
    ):
        raise ValueError("confirmation proposal policy mismatch")

    registered = payload["registered_seed_policy"]
    candidate = derived_selection["selected_candidate"]
    expected_rank = (
        int(candidate["n_components"])
        if candidate is not None
        else max(config.component_grid)
    )
    if (
        registered.get("seed") != config.seeds[0]
        or registered.get("selection") != "first_configured_seed_never_best_seed"
        or registered.get("rank") != expected_rank
        or registered.get("rank_policy")
        != "selected_candidate_rank_or_max_rank_if_no_candidate"
        or registered.get("plotted_component_pairs")
        != [[1, 2], [3, 4], [5, 6], [7, 8]]
        or registered.get("panels")
        != [_panel_label(value) for value in config.panel_sizes]
    ):
        raise ValueError("registered figure policy mismatch")
    coordinates = payload["registered_coordinates"]
    coordinate_map: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in coordinates:
        key = (int(row["n_components"]), str(row["panel_label"]))
        coordinate_map.setdefault(key, []).append(row)
        if int(row["seed"]) != config.seeds[0]:
            raise ValueError("registered coordinates use a nonregistered seed")
        for component in range(1, 9):
            if not math.isfinite(float(row[f"component_{component}"])):
                raise ValueError("registered coordinate is non-finite")
    if set(coordinate_map) != expected_summary_keys:
        raise ValueError("registered coordinates do not cover every grid cell")
    common_samples: set[str] | None = None
    positions = {value: index for index, value in enumerate(EXPECTED_POPULATIONS)}
    for key, rows in coordinate_map.items():
        samples = [row for row in rows if row["row_type"] == "held_out_sample"]
        centroids = [row for row in rows if row["row_type"] == "train_centroid"]
        if (
            not samples
            or len(centroids) != 5
            or {row["population"] for row in centroids} != set(EXPECTED_POPULATIONS)
        ):
            raise ValueError("registered cell lacks samples or centroids")
        sample_ids = {str(row["sample_id"]) for row in samples}
        if len(sample_ids) != len(samples):
            raise ValueError("registered sample IDs are not unique")
        if common_samples is None:
            common_samples = sample_ids
        elif common_samples != sample_ids:
            raise ValueError("registered grid cells use different held-out samples")
        confusion = np.zeros((5, 5), dtype=np.int64)
        for row in samples:
            if (
                row.get("truth") not in EXPECTED_POPULATIONS
                or row.get("prediction") not in EXPECTED_POPULATIONS
                or row.get("population") != row.get("truth")
                or bool(row.get("correct"))
                != (row.get("truth") == row.get("prediction"))
            ):
                raise ValueError("registered predictions are inconsistent")
            confusion[
                positions[str(row["truth"])], positions[str(row["prediction"])]
            ] += 1
        record = record_map[(config.seeds[0], key[0], key[1])]
        if confusion.tolist() != record["confusion_matrix_counts"]:
            raise ValueError("registered predictions disagree with record")

    unsigned = dict(payload)
    digest = unsigned.pop("analysis_sha256")
    if not _is_sha256(digest) or canonical_sha256(unsigned) != digest:
        raise ValueError("analysis_sha256 mismatch")
    return {
        "status": "valid",
        "analysis_sha256": digest,
        "records": len(record_map),
        "grid_cells": len(summary_map),
        "sample_prediction_rows": len(payload["sample_prediction_evidence"]),
        "registered_coordinates": len(coordinates),
        "candidate_selected": candidate is not None,
        "lock_proposal_eligible": expected_eligible,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    ""
                    if row.get(field) is None
                    else json.dumps(row[field], separators=(",", ":"))
                    if isinstance(row.get(field), (list, dict))
                    else row.get(field)
                )
                for field in fields
            }
        )
    return output.getvalue().encode("utf-8")


def _metrics_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(
        payload["grid_summaries"],
        (
            "n_components",
            "panel_label",
            "requested_panel_size",
            "seeds",
            "minimum_selected_snp_count",
            "maximum_selected_snp_count",
            "mean_accuracy",
            "mean_balanced_accuracy",
            "sample_std_balanced_accuracy",
            "minimum_balanced_accuracy",
            "maximum_balanced_accuracy",
            "mean_macro_f1",
            "mean_paired_balanced_accuracy_delta_vs_same_rank_all",
            "minimum_paired_balanced_accuracy_delta_vs_same_rank_all",
            "qualification_status",
        ),
    )


def _population_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(
        payload["population_summaries"],
        (
            "n_components",
            "panel_label",
            "requested_panel_size",
            "population",
            "seeds",
            "pooled_support",
            "pooled_true_positive",
            "mean_recall",
            "minimum_recall",
            "maximum_recall",
        ),
    )


def _stability_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(
        payload["panel_stability_summaries"],
        (
            "n_components",
            "panel_label",
            "requested_panel_size",
            "seeds",
            "seed_pairs",
            "minimum_selected_snp_count",
            "maximum_selected_snp_count",
            "intersection_snp_count",
            "union_snp_count",
            "snps_selected_in_at_least_80_percent_of_seeds",
            "mean_pairwise_jaccard",
            "minimum_pairwise_jaccard",
            "maximum_pairwise_jaccard",
        ),
    )


def _sample_evidence_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(
        payload["sample_prediction_evidence"],
        (
            "seed",
            "n_components",
            "panel_label",
            "requested_panel_size",
            "selected_snp_count",
            "selected_original_indices_sha256",
            "sample_position",
            "sample_index",
            "sample_id",
            "truth",
            "prediction",
            "correct",
            "train_indices_sha256",
            "test_indices_sha256",
            "test_indices_dtype",
            "train_sample_ids_sha256",
            "test_sample_ids_sha256",
            "test_sample_ids_dtype",
            "test_labels_sha256",
            "test_labels_dtype",
            "test_predictions_sha256",
            "test_predictions_dtype",
        ),
    )


def _coordinates_csv_bytes(payload: Mapping[str, Any]) -> bytes:
    return _csv_bytes(
        payload["registered_coordinates"],
        (
            "row_type",
            "seed",
            "n_components",
            "panel_label",
            "requested_panel_size",
            "selected_snp_count",
            "sample_index",
            "sample_id",
            "population",
            "truth",
            "prediction",
            "correct",
            *(f"component_{index}" for index in range(1, 9)),
        ),
    )


def _render_pc_figure(
    payload: Mapping[str, Any],
    path: Path,
    *,
    x_component: int,
    y_component: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    candidate = _candidate_map(payload)
    rank = int(candidate["n_components"])
    config = payload["config"]
    panels = [_panel_label(value) for value in config["panel_sizes"]]
    summaries, _ = _summary_maps(payload)
    palette = plt.get_cmap("tab10")
    colors = {
        population: palette(index)
        for index, population in enumerate(EXPECTED_POPULATIONS)
    }
    with plt.rc_context(plt.rcParamsDefault):
        figure, axes = plt.subplots(
            1,
            len(panels),
            figsize=(4.5 * len(panels), 5.15),
            squeeze=False,
            dpi=150,
        )
        for panel_index, panel_label in enumerate(panels):
            axis = axes[0, panel_index]
            cell = [
                row
                for row in payload["registered_coordinates"]
                if row["n_components"] == rank and row["panel_label"] == panel_label
            ]
            samples = [row for row in cell if row["row_type"] == "held_out_sample"]
            centroids = [row for row in cell if row["row_type"] == "train_centroid"]
            for population in EXPECTED_POPULATIONS:
                population_rows = [row for row in samples if row["truth"] == population]
                correct = [row for row in population_rows if row["correct"]]
                errors = [row for row in population_rows if not row["correct"]]
                if correct:
                    axis.scatter(
                        [row[f"component_{x_component}"] for row in correct],
                        [row[f"component_{y_component}"] for row in correct],
                        color=colors[population],
                        s=28,
                        alpha=0.78,
                        edgecolors="white",
                        linewidths=0.35,
                        label=population,
                    )
                if errors:
                    axis.scatter(
                        [row[f"component_{x_component}"] for row in errors],
                        [row[f"component_{y_component}"] for row in errors],
                        color=colors[population],
                        s=45,
                        marker="x",
                        linewidths=1.35,
                    )
                centroid = next(
                    row for row in centroids if row["population"] == population
                )
                axis.scatter(
                    [centroid[f"component_{x_component}"]],
                    [centroid[f"component_{y_component}"]],
                    color=colors[population],
                    s=110,
                    marker="*",
                    edgecolors="black",
                    linewidths=0.75,
                    zorder=5,
                )
            record = next(
                row
                for row in payload["records"]
                if row["seed"] == config["seeds"][0]
                and row["n_components"] == rank
                and row["panel_label"] == panel_label
            )
            summary = summaries[(rank, panel_label)]
            title = "All retained" if panel_label == "all" else f"Top {panel_label}"
            axis.set_title(
                f"{title}\nseed BA={record['balanced_accuracy']:.3f}; "
                f"mean={summary['mean_balanced_accuracy']:.3f}",
                fontsize=10,
            )
            axis.axhline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
            axis.axvline(0.0, color="#d1d5db", linewidth=0.65, zorder=0)
            axis.set_xlabel(f"PC{x_component}")
            axis.set_ylabel(f"PC{y_component}")
            axis.grid(alpha=0.12)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.955),
        )
        auxiliary = ""
        if y_component > rank:
            auxiliary = "; these axes are diagnostic and are not used by the classifier"
        figure.suptitle(
            f"PCA-loading stratification extension — PC{x_component}/PC{y_component}, "
            f"registered seed {config['seeds'][0]}, selected rank {rank}{auxiliary}",
            fontsize=12.5,
            y=1.015,
        )
        figure.text(
            0.5,
            0.012,
            "Held-out samples only; stars are training centroids and x marks errors. "
            "Each panel has independently refit PCA axes. Multi-PC classification cannot be judged from one 2D view.",
            ha="center",
            fontsize=8.5,
        )
        figure.tight_layout(rect=(0.0, 0.07, 1.0, 0.90), w_pad=1.6)
        figure.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            metadata={"Software": "vp-rspca", "Creation Time": None},
        )
        plt.close(figure)


def _aggregate_confusion(
    payload: Mapping[str, Any], n_components: int, panel_label: str
) -> NDArray[np.int64]:
    matrices = [
        np.asarray(row["confusion_matrix_counts"], dtype=np.int64)
        for row in payload["records"]
        if int(row["n_components"]) == n_components
        and row["panel_label"] == panel_label
    ]
    if len(matrices) != len(payload["config"]["seeds"]):
        raise RuntimeError("cannot aggregate an incomplete confusion matrix")
    return np.sum(matrices, axis=0)


def _render_confusion(payload: Mapping[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    candidate = _candidate_map(payload)
    rank = int(candidate["n_components"])
    panel_label = str(candidate["panel_label"])
    panels = [panel_label] if panel_label == "all" else [panel_label, "all"]
    titles = [
        ("Selected candidate" if value == panel_label else "Same-rank all SNP")
        + f"\n{rank} PCs / {'all retained' if value == 'all' else 'top-' + value}"
        for value in panels
    ]
    with plt.rc_context(plt.rcParamsDefault):
        figure, axes = plt.subplots(
            1, len(panels), figsize=(6.7 * len(panels), 5.7), squeeze=False, dpi=150
        )
        for index, (current_panel, title) in enumerate(zip(panels, titles, strict=True)):
            axis = axes[0, index]
            confusion = _aggregate_confusion(payload, rank, current_panel)
            normalized = confusion / confusion.sum(axis=1, keepdims=True)
            image = axis.imshow(normalized, vmin=0.0, vmax=1.0, cmap="Blues")
            for row in range(5):
                for column in range(5):
                    value = normalized[row, column]
                    axis.text(
                        column,
                        row,
                        f"{int(confusion[row, column])}\n{value:.1%}",
                        ha="center",
                        va="center",
                        fontsize=8.8,
                        color="white" if value > 0.55 else "#111827",
                    )
            axis.set_xticks(range(5), EXPECTED_POPULATIONS)
            axis.set_yticks(range(5), EXPECTED_POPULATIONS)
            axis.set_xlabel("Predicted superpopulation")
            axis.set_ylabel("True superpopulation")
            axis.set_title(title)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Row recall")
        figure.suptitle(
            f"Five-development-seed held-out confusion (seeds {payload['config']['seeds'][0]}–"
            f"{payload['config']['seeds'][-1]})",
            fontsize=13,
        )
        figure.text(
            0.5,
            0.012,
            "Counts and row-normalized percentages. Superpopulation labels are dataset groups, not biological race categories.",
            ha="center",
            fontsize=8.7,
        )
        figure.tight_layout(rect=(0.0, 0.055, 1.0, 0.93))
        figure.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            metadata={"Software": "vp-rspca", "Creation Time": None},
        )
        plt.close(figure)


def _artifact_filenames(payload: Mapping[str, Any]) -> tuple[str, ...]:
    if payload["confirmation_proposal_policy"]["eligible"]:
        return (*BASE_ARTIFACT_FILENAMES, LOCK_PROPOSAL_FILENAME)
    return BASE_ARTIFACT_FILENAMES


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_formal_artifact_locations(
    *,
    output_directory: Path,
    intent_receipt_path: Path,
    post_run_receipt_path: Path,
    source_bundle_path: Path,
    repo_root: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    root = Path(repo_root).resolve()
    if root != _runner_repo_root():
        raise ValueError(
            "repo_root must resolve exactly to the repository containing this runner"
        )
    output = Path(output_directory).resolve(strict=False)
    intent = Path(intent_receipt_path).resolve(strict=False)
    post = Path(post_run_receipt_path).resolve(strict=False)
    source_bundle = Path(source_bundle_path).resolve(strict=False)
    if _path_is_within(output, root):
        raise ValueError("formal output must be outside the Git worktree")
    if intent == post:
        raise ValueError("pre-run and post-run receipt paths must be distinct")
    if source_bundle in {intent, post}:
        raise ValueError("source bundle and run receipt paths must be distinct")
    for receipt, role in ((intent, "pre-run intent"), (post, "post-run")):
        if _path_is_within(receipt, root):
            raise ValueError(f"{role} receipt must be outside the Git worktree")
        if _path_is_within(receipt, output):
            raise ValueError(f"{role} receipt must be external to the output package")
    if _path_is_within(source_bundle, root) or _path_is_within(
        source_bundle, output
    ):
        raise ValueError("source bundle must be outside the worktree and output package")
    return output, intent, post, source_bundle, root


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    if os.path.lexists(destination):
        raise FileExistsError(f"receipt already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = _strict_json_bytes(value)
    with destination.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    _regular_file(destination, role="external run receipt")


def _valid_output_uuid(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value


def _validate_source_bundle_file(
    source_bundle_path: Path,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a real minimal Git bundle to the executing 18-file checkout."""

    bundle_path = Path(source_bundle_path).resolve()
    status = _regular_file(bundle_path, role="18-file source Git bundle")
    root = _runner_repo_root()
    current_identity = _git_identity(root)
    if current_identity != source_identity.get("git_identity"):
        raise ValueError("source bundle check disagrees with the executing Git identity")
    tracked_head_files = sorted(
        value
        for value in _git_text(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        if value
    )
    if tracked_head_files != sorted(SOURCE_FILE_ROLES):
        raise ValueError("formal source checkout must contain exactly the 18 source files")
    verify = subprocess.run(
        ["git", "-C", str(root), "bundle", "verify", str(bundle_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        raise ValueError(f"source Git bundle verification failed: {verify.stderr[-500:]}")
    heads = subprocess.run(
        ["git", "bundle", "list-heads", str(bundle_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if heads.returncode != 0:
        raise ValueError(f"source Git bundle heads cannot be read: {heads.stderr[-500:]}")
    bundled_commits = {
        line.split(maxsplit=1)[0]
        for line in heads.stdout.splitlines()
        if line.strip()
    }
    if current_identity["head_commit"] not in bundled_commits:
        raise ValueError("source Git bundle does not expose the executing HEAD commit")
    return {
        "source_bundle_format": "git_bundle_exact_18_file_snapshot_v1",
        "source_bundle_sha256": _file_sha256(bundle_path),
        "source_bundle_bytes": int(status.st_size),
    }


def _build_pre_run_intent_receipt(
    *,
    output_uuid: str,
    source_bundle_path: Path,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    if not _valid_output_uuid(output_uuid):
        raise ValueError("formal output UUID must be a canonical UUID4")
    if not _input_is_lock_eligible(input_identity):
        raise ValueError("pre-run intent requires the fixed prepared input")
    source_bundle = _source_bundle_descriptor(source_identity)
    bundle_file = _validate_source_bundle_file(
        source_bundle_path, source_identity
    )
    serialized_config = json.loads(
        _strict_json_bytes(asdict(StratificationExtensionConfig())).decode("utf-8")
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "stratification_extension_pre_run_intent",
        "status": "intent_recorded_before_numerical_evaluation",
        "output_uuid": output_uuid,
        **bundle_file,
        "logical_source_closure_sha256": canonical_sha256(source_bundle),
        "source_head_commit": source_bundle["head_commit"],
        "source_head_tree": source_bundle["head_tree"],
        "execution_source_manifest_sha256": source_bundle[
            "source_manifest_sha256"
        ],
        "source_files": source_bundle["files"],
        "source_file_count": len(source_bundle["files"]),
        "input_genotype_npz_sha256": input_identity["genotype_npz_sha256"],
        "prepared_source_manifest_sha256": input_identity[
            "external_source_manifest_sha256"
        ],
        "config": serialized_config,
        "confirmation_seeds_executed": False,
    }
    receipt["intent_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_pre_run_intent_receipt(
    receipt: Mapping[str, Any],
    *,
    source_bundle_path: Path,
    input_identity: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    output_uuid = receipt.get("output_uuid")
    if not _valid_output_uuid(output_uuid):
        raise ValueError("pre-run intent has an invalid output UUID")
    expected = _build_pre_run_intent_receipt(
        output_uuid=str(output_uuid),
        source_bundle_path=source_bundle_path,
        input_identity=input_identity,
        source_identity=source_identity,
    )
    if dict(receipt) != expected:
        raise ValueError("pre-run intent receipt does not match trusted inputs/source")
    return expected


def _formal_execution_binding(
    intent_receipt_path: Path,
    intent_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    binding = {
        "output_uuid": intent_receipt["output_uuid"],
        "intent_payload_sha256": intent_receipt["intent_payload_sha256"],
        "intent_receipt_file_sha256": _file_sha256(Path(intent_receipt_path)),
        "source_bundle_sha256": intent_receipt["source_bundle_sha256"],
    }
    _validate_formal_execution_binding(binding)
    return binding


def _validate_formal_execution_binding(binding: Mapping[str, Any]) -> None:
    if set(binding) != {
        "output_uuid",
        "intent_payload_sha256",
        "intent_receipt_file_sha256",
        "source_bundle_sha256",
    } or not _valid_output_uuid(binding.get("output_uuid")):
        raise ValueError("formal execution binding schema or UUID mismatch")
    if any(
        not _is_sha256(binding.get(field))
        for field in (
            "intent_payload_sha256",
            "intent_receipt_file_sha256",
            "source_bundle_sha256",
        )
    ):
        raise ValueError("formal execution binding contains an invalid SHA-256")


def _build_post_run_receipt(
    *,
    output_directory: Path,
    intent_receipt_path: Path,
    intent_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    binding = manifest.get("formal_execution")
    if not isinstance(binding, Mapping):
        raise ValueError("formal package manifest lacks its execution binding")
    _validate_formal_execution_binding(binding)
    expected_binding = _formal_execution_binding(intent_receipt_path, intent_receipt)
    if dict(binding) != expected_binding:
        raise ValueError("package execution binding disagrees with pre-run intent")
    manifest_path = Path(output_directory) / MANIFEST_FILENAME
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_type": "stratification_extension_post_run_completion",
        "status": "completion_recorded_after_package_write",
        "output_uuid": binding["output_uuid"],
        "intent_receipt_file_sha256": binding["intent_receipt_file_sha256"],
        "intent_payload_sha256": binding["intent_payload_sha256"],
        "source_bundle_sha256": binding["source_bundle_sha256"],
        "input_genotype_npz_sha256": intent_receipt[
            "input_genotype_npz_sha256"
        ],
        "prepared_source_manifest_sha256": intent_receipt[
            "prepared_source_manifest_sha256"
        ],
        "analysis_sha256": manifest["analysis_sha256"],
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "package_manifest_file_sha256": _file_sha256(manifest_path),
        "confirmation_seeds_executed": False,
    }
    receipt["post_run_payload_sha256"] = canonical_sha256(receipt)
    return receipt


def _validate_post_run_receipt(
    receipt: Mapping[str, Any],
    *,
    output_directory: Path,
    intent_receipt_path: Path,
    intent_receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _build_post_run_receipt(
        output_directory=output_directory,
        intent_receipt_path=intent_receipt_path,
        intent_receipt=intent_receipt,
        manifest=manifest,
    )
    if dict(receipt) != expected:
        raise ValueError("post-run receipt does not match the completed package")
    return expected


def _first_replay_mismatch(expected: Any, observed: Any, path: str = "$") -> str | None:
    if type(expected) is not type(observed):
        return path
    if isinstance(expected, Mapping):
        if set(expected) != set(observed):
            return path
        for key in sorted(expected):
            mismatch = _first_replay_mismatch(
                expected[key], observed[key], f"{path}.{key}"
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return path
        for index, (left, right) in enumerate(zip(expected, observed, strict=True)):
            mismatch = _first_replay_mismatch(left, right, f"{path}[{index}]")
            if mismatch is not None:
                return mismatch
        return None
    return None if expected == observed else path


def _validate_trusted_input_replay(
    packaged_payload: Mapping[str, Any],
    replayed_payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact field-by-field equality after trusted-input recomputation."""

    validate_result_payload(packaged_payload)
    validate_result_payload(replayed_payload)
    # Compare the canonical persisted representation.  Python tuples in the
    # in-memory dataclass config become JSON arrays on disk; this normalization
    # preserves exact field/value comparison without treating that required
    # serialization step as a numerical mismatch.
    persisted_packaged = json.loads(_strict_json_bytes(packaged_payload))
    persisted_replayed = json.loads(_strict_json_bytes(replayed_payload))
    mismatch = _first_replay_mismatch(persisted_packaged, persisted_replayed)
    if mismatch is not None:
        raise ValueError(f"trusted-input full replay mismatch at {mismatch}")
    return {
        "trusted_input_replay_verified": True,
        "replayed_analysis_sha256": replayed_payload["analysis_sha256"],
    }


def _write_output_package_internal(
    payload: Mapping[str, Any],
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    formal_execution: Mapping[str, Any] | None = None,
    _capability: object,
) -> dict[str, Any]:
    validate_result_payload(payload)
    formal = payload["config"]["analysis_mode"] == "development_protocol"
    if formal:
        if _capability is not _FORMAL_DEVELOPMENT_PACKAGE_CAPABILITY:
            raise PermissionError("formal package writing requires its private capability")
        if not isinstance(formal_execution, Mapping):
            raise ValueError("formal package writing requires a pre-run intent binding")
        _validate_formal_execution_binding(formal_execution)
    else:
        if _capability is not _TINY_SMOKE_PACKAGE_CAPABILITY:
            raise PermissionError("tiny package writing requires its private capability")
        if formal_execution is not None:
            raise ValueError("tiny-smoke packages cannot contain formal run receipts")
    if formal and repo_root is None:
        raise ValueError("formal package writing requires mandatory source closure")
    if repo_root is not None and _source_identity(repo_root) != payload["source_identity"]:
        raise ValueError("current source/Git identity differs from the result payload")
    output_directory = Path(output_directory)
    if os.path.lexists(output_directory):
        raise FileExistsError(f"output directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_directory.parent / f".{output_directory.name}.tmp-{uuid.uuid4().hex}"
    if os.path.lexists(temporary):
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        (temporary / RESULT_FILENAME).write_bytes(_strict_json_bytes(payload))
        (temporary / METRICS_FILENAME).write_bytes(_metrics_csv_bytes(payload))
        (temporary / POPULATION_FILENAME).write_bytes(_population_csv_bytes(payload))
        (temporary / STABILITY_FILENAME).write_bytes(_stability_csv_bytes(payload))
        (temporary / SAMPLE_EVIDENCE_FILENAME).write_bytes(
            _sample_evidence_csv_bytes(payload)
        )
        (temporary / COORDINATES_FILENAME).write_bytes(_coordinates_csv_bytes(payload))
        for filename, pair in (
            (PC12_FILENAME, (1, 2)),
            (PC34_FILENAME, (3, 4)),
            (PC56_FILENAME, (5, 6)),
            (PC78_FILENAME, (7, 8)),
        ):
            _render_pc_figure(
                payload,
                temporary / filename,
                x_component=pair[0],
                y_component=pair[1],
            )
        _render_confusion(payload, temporary / CONFUSION_FILENAME)
        if payload["confirmation_proposal_policy"]["eligible"]:
            proposal = _build_lock_proposal(payload)
            validate_lock_proposal(
                proposal,
                payload,
                expected_development_analysis_sha256=payload["analysis_sha256"],
            )
            (temporary / LOCK_PROPOSAL_FILENAME).write_bytes(
                _strict_json_bytes(proposal)
            )

        artifact_names = _artifact_filenames(payload)
        artifacts: dict[str, dict[str, Any]] = {}
        for filename in artifact_names:
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
            "verification_scope": "self_consistency_verified",
            "analysis_status": "post_hoc_development_selection",
            "confirmatory": False,
            "confirmation_seeds_executed": False,
            "lock_proposal_included": bool(
                payload["confirmation_proposal_policy"]["eligible"]
            ),
            "analysis_sha256": payload["analysis_sha256"],
            "artifact_count_excluding_manifest": len(artifact_names),
            "exact_package_file_count": len(artifact_names) + 1,
            "artifacts": artifacts,
            "formal_execution": (
                dict(formal_execution) if formal_execution is not None else None
            ),
        }
        manifest["package_anchor_sha256"] = canonical_sha256(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "artifacts": artifacts,
                "formal_execution": manifest["formal_execution"],
            }
        )
        manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
        (temporary / MANIFEST_FILENAME).write_bytes(_strict_json_bytes(manifest))
        temporary.rename(output_directory)
    except BaseException:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        raise
    if formal:
        manifest = _strict_json_load(output_directory / MANIFEST_FILENAME)
        return _validate_output_package_self_consistency(
            output_directory,
            repo_root=repo_root,
            expected_analysis_sha256=payload["analysis_sha256"],
            expected_package_anchor_sha256=manifest["package_anchor_sha256"],
        )
    return _validate_output_package_self_consistency(
        output_directory, repo_root=repo_root
    )


def write_output_package(
    payload: Mapping[str, Any],
    output_directory: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Write a tiny-smoke package; formal writing is CLI-controlled."""

    if payload.get("config", {}).get("analysis_mode") != "tiny_smoke":
        raise PermissionError(
            "public package writer is restricted to tiny_smoke; use the controlled run CLI"
        )
    return _write_output_package_internal(
        payload,
        output_directory,
        repo_root=repo_root,
        formal_execution=None,
        _capability=_TINY_SMOKE_PACKAGE_CAPABILITY,
    )


def _validate_output_package_self_consistency(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("output package must be a real directory")
    result_path = output_directory / RESULT_FILENAME
    _regular_file(result_path, role=RESULT_FILENAME)
    payload = _strict_json_load(result_path)
    formal = (
        isinstance(payload, Mapping)
        and isinstance(payload.get("config"), Mapping)
        and payload["config"].get("analysis_mode") == "development_protocol"
    )
    if formal and (
        repo_root is None
        or expected_analysis_sha256 is None
        or expected_package_anchor_sha256 is None
    ):
        raise ValueError(
            "formal validation requires repo-root source closure and both external anchors"
        )
    result_validation = validate_result_payload(payload)
    artifact_names = _artifact_filenames(payload)
    expected_files = set(artifact_names) | {MANIFEST_FILENAME}
    actual_files = {path.name for path in output_directory.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            f"output package files must be exact; missing={sorted(expected_files-actual_files)}, "
            f"extra={sorted(actual_files-expected_files)}"
        )
    manifest_path = output_directory / MANIFEST_FILENAME
    _regular_file(manifest_path, role=MANIFEST_FILENAME)
    manifest = _strict_json_load(manifest_path)
    required_manifest = {
        "schema_version",
        "status",
        "manifest_written_last",
        "verification_scope",
        "analysis_status",
        "confirmatory",
        "confirmation_seeds_executed",
        "lock_proposal_included",
        "analysis_sha256",
        "artifact_count_excluding_manifest",
        "exact_package_file_count",
        "artifacts",
        "formal_execution",
        "package_anchor_sha256",
        "manifest_payload_sha256",
    }
    if set(manifest) != required_manifest:
        raise ValueError("completion manifest schema mismatch")
    if (
        manifest["schema_version"] != 1
        or manifest["status"] != "complete"
        or manifest["manifest_written_last"] is not True
        or manifest["verification_scope"] != "self_consistency_verified"
        or manifest["analysis_status"] != "post_hoc_development_selection"
        or manifest["confirmatory"] is not False
        or manifest["confirmation_seeds_executed"] is not False
        or manifest["lock_proposal_included"]
        is not payload["confirmation_proposal_policy"]["eligible"]
        or manifest["analysis_sha256"] != payload["analysis_sha256"]
        or manifest["artifact_count_excluding_manifest"] != len(artifact_names)
        or manifest["exact_package_file_count"] != len(expected_files)
        or set(manifest["artifacts"]) != set(artifact_names)
    ):
        raise ValueError("completion manifest policy mismatch")
    if formal:
        if not isinstance(manifest["formal_execution"], Mapping):
            raise ValueError("formal package lacks its pre-run execution binding")
        _validate_formal_execution_binding(manifest["formal_execution"])
    elif manifest["formal_execution"] is not None:
        raise ValueError("tiny-smoke package unexpectedly has a formal execution binding")
    unsigned_manifest = dict(manifest)
    manifest_digest = unsigned_manifest.pop("manifest_payload_sha256")
    if (
        not _is_sha256(manifest_digest)
        or canonical_sha256(unsigned_manifest) != manifest_digest
    ):
        raise ValueError("manifest payload digest mismatch")
    derived_anchor = canonical_sha256(
        {
            "analysis_sha256": manifest["analysis_sha256"],
            "artifacts": manifest["artifacts"],
            "formal_execution": manifest["formal_execution"],
        }
    )
    if (
        not _is_sha256(manifest["package_anchor_sha256"])
        or manifest["package_anchor_sha256"] != derived_anchor
    ):
        raise ValueError("package anchor mismatch")
    for filename in artifact_names:
        path = output_directory / filename
        status = _regular_file(path, role=filename)
        receipt = manifest["artifacts"][filename]
        if (
            set(receipt) != {"bytes", "sha256"}
            or receipt["bytes"] != status.st_size
            or receipt["sha256"] != _file_sha256(path)
        ):
            raise ValueError(f"artifact hash/size mismatch: {filename}")
    if (expected_analysis_sha256 is None) != (
        expected_package_anchor_sha256 is None
    ):
        raise ValueError("external validation requires both result/package anchors")
    if expected_analysis_sha256 is not None:
        if (
            not _is_sha256(expected_analysis_sha256)
            or payload["analysis_sha256"] != expected_analysis_sha256
        ):
            raise ValueError("external analysis-hash anchor mismatch")
        if (
            not _is_sha256(expected_package_anchor_sha256)
            or manifest["package_anchor_sha256"]
            != expected_package_anchor_sha256
        ):
            raise ValueError("external package-hash anchor mismatch")
    if repo_root is not None:
        current_source_identity = _source_identity(repo_root)
        if current_source_identity != payload["source_identity"]:
            raise ValueError("source closure or recorded Git identity mismatch")
    if (output_directory / METRICS_FILENAME).read_bytes() != _metrics_csv_bytes(payload):
        raise ValueError("metrics CSV is not an exact result serialization")
    if (output_directory / POPULATION_FILENAME).read_bytes() != _population_csv_bytes(
        payload
    ):
        raise ValueError("population CSV is not an exact result serialization")
    if (output_directory / STABILITY_FILENAME).read_bytes() != _stability_csv_bytes(
        payload
    ):
        raise ValueError("stability CSV is not an exact result serialization")
    if (
        output_directory / SAMPLE_EVIDENCE_FILENAME
    ).read_bytes() != _sample_evidence_csv_bytes(payload):
        raise ValueError("sample evidence CSV is not an exact result serialization")
    if (output_directory / COORDINATES_FILENAME).read_bytes() != _coordinates_csv_bytes(
        payload
    ):
        raise ValueError("coordinate CSV is not an exact result serialization")
    for filename in (
        PC12_FILENAME,
        PC34_FILENAME,
        PC56_FILENAME,
        PC78_FILENAME,
        CONFUSION_FILENAME,
    ):
        from PIL import Image

        with Image.open(output_directory / filename) as image:
            image.verify()
        with Image.open(output_directory / filename) as image:
            if (
                image.format != "PNG"
                or image.width < 2400
                or image.height < 700
            ):
                raise ValueError(f"PNG below clarity floor: {filename}")
    if payload["confirmation_proposal_policy"]["eligible"]:
        proposal = _strict_json_load(output_directory / LOCK_PROPOSAL_FILENAME)
        validate_lock_proposal(
            proposal,
            payload,
            expected_development_analysis_sha256=payload["analysis_sha256"],
        )
    return {
        **result_validation,
        "artifact_integrity_verified": True,
        "source_closure_verified": repo_root is not None,
        "self_consistency_verified": True,
        "dual_anchor_match_verified": expected_analysis_sha256 is not None,
        "trusted_input_replay_verified": False,
        "manifest_payload_sha256": manifest_digest,
        "package_anchor_sha256": manifest["package_anchor_sha256"],
        "package_files": len(expected_files),
    }


def validate_output_package(
    output_directory: Path,
    *,
    repo_root: Path | None = None,
    expected_analysis_sha256: str | None = None,
    expected_package_anchor_sha256: str | None = None,
    genotypes_path: Path | None = None,
    source_manifest_path: Path | None = None,
    source_bundle_path: Path | None = None,
    intent_receipt_path: Path | None = None,
    post_run_receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Validate smoke structurally or formally replay from trusted input paths."""

    output_directory = Path(output_directory)
    if output_directory.is_symlink() or not output_directory.is_dir():
        raise ValueError("output package must be a real directory")
    result_path = output_directory / RESULT_FILENAME
    _regular_file(result_path, role=RESULT_FILENAME)
    packaged_payload = _strict_json_load(result_path)
    formal = (
        isinstance(packaged_payload, Mapping)
        and isinstance(packaged_payload.get("config"), Mapping)
        and packaged_payload["config"].get("analysis_mode")
        == "development_protocol"
    )
    formal_arguments = (
        repo_root,
        expected_analysis_sha256,
        expected_package_anchor_sha256,
        genotypes_path,
        source_manifest_path,
        source_bundle_path,
        intent_receipt_path,
        post_run_receipt_path,
    )
    if formal and any(value is None for value in formal_arguments):
        raise ValueError(
            "formal validation requires repo root, dual anchors, original NPZ/manifest, "
            "the source Git bundle, and external pre-run/post-run receipts"
        )
    if not formal and any(
        value is not None
        for value in (
            genotypes_path,
            source_manifest_path,
            source_bundle_path,
            intent_receipt_path,
            post_run_receipt_path,
        )
    ):
        raise ValueError("trusted-input replay arguments are only valid for formal packages")

    self_check = _validate_output_package_self_consistency(
        output_directory,
        repo_root=repo_root,
        expected_analysis_sha256=expected_analysis_sha256,
        expected_package_anchor_sha256=expected_package_anchor_sha256,
    )
    if not formal:
        return self_check

    # Type narrowing follows the mandatory-argument check above.
    assert repo_root is not None
    assert genotypes_path is not None
    assert source_manifest_path is not None
    assert source_bundle_path is not None
    assert intent_receipt_path is not None
    assert post_run_receipt_path is not None
    _, resolved_intent, resolved_post, resolved_bundle, resolved_root = (
        _validate_formal_artifact_locations(
            output_directory=output_directory,
            intent_receipt_path=intent_receipt_path,
            post_run_receipt_path=post_run_receipt_path,
            source_bundle_path=source_bundle_path,
            repo_root=repo_root,
        )
    )
    _regular_file(resolved_intent, role="external pre-run intent receipt")
    _regular_file(resolved_post, role="external post-run receipt")
    intent_receipt = _strict_json_load(resolved_intent)
    post_run_receipt = _strict_json_load(resolved_post)
    if not isinstance(intent_receipt, Mapping) or not isinstance(
        post_run_receipt, Mapping
    ):
        raise ValueError("external run receipts must be JSON objects")

    _, trusted_input_identity = _load_formal_development_input(
        Path(genotypes_path), Path(source_manifest_path)
    )
    current_source_identity = _source_identity(resolved_root)
    validated_intent = _validate_pre_run_intent_receipt(
        intent_receipt,
        source_bundle_path=resolved_bundle,
        input_identity=trusted_input_identity,
        source_identity=current_source_identity,
    )
    manifest = _strict_json_load(output_directory / MANIFEST_FILENAME)
    expected_binding = _formal_execution_binding(resolved_intent, validated_intent)
    if manifest.get("formal_execution") != expected_binding:
        raise ValueError("package manifest does not bind the supplied pre-run intent")
    validated_post = _validate_post_run_receipt(
        post_run_receipt,
        output_directory=output_directory,
        intent_receipt_path=resolved_intent,
        intent_receipt=validated_intent,
        manifest=manifest,
    )

    # This performs a write-free, plot-free replay of every seed, split,
    # preprocessor, full PCA, loading ranking, panel refit, centroid and
    # prediction.  Exact recursive comparison covers every saved field.
    replayed_payload = evaluate_formal_development_from_paths(
        Path(genotypes_path), Path(source_manifest_path)
    )
    replay_validation = _validate_trusted_input_replay(
        packaged_payload, replayed_payload
    )
    return {
        **self_check,
        **replay_validation,
        "intent_receipt_payload_sha256": validated_intent[
            "intent_payload_sha256"
        ],
        "post_run_receipt_payload_sha256": validated_post[
            "post_run_payload_sha256"
        ],
        "run_receipts_bound_to_package": True,
    }


def _synthetic_smoke_data() -> base.GenotypeInput:
    rng = np.random.default_rng(7319)
    labels = np.asarray(
        [population for population in EXPECTED_POPULATIONS for _ in range(20)],
        dtype=str,
    )
    probabilities = np.full((labels.size, 1700), 0.24, dtype=float)
    for index, population in enumerate(EXPECTED_POPULATIONS):
        rows = labels == population
        probabilities[rows, 120 * index : 120 * (index + 1)] = 0.80
        probabilities[rows, 700 + 100 * index : 750 + 100 * index] = 0.10 + 0.15 * index
    genotypes = rng.binomial(2, probabilities).astype(float)
    return base.GenotypeInput(
        genotypes=genotypes,
        sample_ids=np.asarray([f"smoke_{index:03d}" for index in range(labels.size)]),
        snp_ids=np.asarray([f"22:{index + 1}:A:G" for index in range(1700)]),
        chromosomes=np.asarray(["22"] * 1700),
        population_labels=labels,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="execute the frozen development protocol")
    run.add_argument("--genotypes", type=Path, required=True)
    run.add_argument("--source-manifest", type=Path, required=True)
    run.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
        choices=(PREPARED_SOURCE_MANIFEST_SHA256,),
    )
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--repo-root", type=Path, default=Path("."))
    run.add_argument("--source-bundle", type=Path, required=True)
    run.add_argument("--intent-receipt", type=Path, required=True)
    run.add_argument("--post-run-receipt", type=Path, required=True)
    smoke = commands.add_parser("smoke", help="run a small synthetic structural smoke")
    smoke.add_argument("--output-dir", type=Path, required=True)
    smoke.add_argument("--repo-root", type=Path, default=Path("."))
    validate = commands.add_parser("validate", help="strictly validate a package")
    validate.add_argument("output_dir", type=Path)
    validate.add_argument("--repo-root", type=Path, required=True)
    validate.add_argument("--expected-analysis-sha256", required=True)
    validate.add_argument("--expected-package-anchor-sha256", required=True)
    validate.add_argument("--genotypes", type=Path, required=True)
    validate.add_argument("--source-manifest", type=Path, required=True)
    validate.add_argument(
        "--expected-source-manifest-sha256",
        required=True,
        choices=(PREPARED_SOURCE_MANIFEST_SHA256,),
    )
    validate.add_argument("--intent-receipt", type=Path, required=True)
    validate.add_argument("--post-run-receipt", type=Path, required=True)
    validate.add_argument("--source-bundle", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        print(
            json.dumps(
                validate_output_package(
                    args.output_dir,
                    repo_root=args.repo_root,
                    expected_analysis_sha256=args.expected_analysis_sha256,
                    expected_package_anchor_sha256=args.expected_package_anchor_sha256,
                    genotypes_path=args.genotypes,
                    source_manifest_path=args.source_manifest,
                    source_bundle_path=args.source_bundle,
                    intent_receipt_path=args.intent_receipt,
                    post_run_receipt_path=args.post_run_receipt,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if os.path.lexists(args.output_dir):
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    repo_root = args.repo_root.resolve()
    if args.command == "smoke":
        data = _synthetic_smoke_data()
        config = StratificationExtensionConfig(
            seeds=(73, 74),
            panel_sizes=(400, 800, 1600, None),
            analysis_mode="tiny_smoke",
        )
        input_identity = {
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
        payload = evaluate_stratification_extension(
            data,
            config,
            input_identity=input_identity,
            source_identity=_source_identity(repo_root),
        )
        validation = write_output_package(
            payload,
            args.output_dir,
            repo_root=repo_root,
        )
        writer_self_check = validation
    else:
        output_path, intent_path, post_path, bundle_path, repo_root = (
            _validate_formal_artifact_locations(
                output_directory=args.output_dir,
                intent_receipt_path=args.intent_receipt,
                post_run_receipt_path=args.post_run_receipt,
                source_bundle_path=args.source_bundle,
                repo_root=repo_root,
            )
        )
        if os.path.lexists(intent_path) or os.path.lexists(post_path):
            raise SystemExit("formal external receipt paths must not already exist")
        _, input_identity = _load_formal_development_input(
            args.genotypes, args.source_manifest
        )
        source_identity = _source_identity(repo_root)
        output_uuid = str(uuid.uuid4())
        intent_receipt = _build_pre_run_intent_receipt(
            output_uuid=output_uuid,
            source_bundle_path=bundle_path,
            input_identity=input_identity,
            source_identity=source_identity,
        )
        # This exclusive external write intentionally precedes numerical work.
        _write_json_exclusive(intent_path, intent_receipt)
        payload = evaluate_formal_development_from_paths(
            args.genotypes, args.source_manifest
        )
        if (
            payload["input_identity"] != input_identity
            or payload["source_identity"] != source_identity
        ):
            raise RuntimeError("source/input identity changed after pre-run intent")
        execution_binding = _formal_execution_binding(
            intent_path, intent_receipt
        )
        writer_self_check = _write_output_package_internal(
            payload,
            output_path,
            repo_root=repo_root,
            formal_execution=execution_binding,
            _capability=_FORMAL_DEVELOPMENT_PACKAGE_CAPABILITY,
        )
        manifest = _strict_json_load(output_path / MANIFEST_FILENAME)
        post_run_receipt = _build_post_run_receipt(
            output_directory=output_path,
            intent_receipt_path=intent_path,
            intent_receipt=intent_receipt,
            manifest=manifest,
        )
        _write_json_exclusive(post_path, post_run_receipt)
        validation = validate_output_package(
            output_path,
            repo_root=repo_root,
            expected_analysis_sha256=payload["analysis_sha256"],
            expected_package_anchor_sha256=manifest["package_anchor_sha256"],
            genotypes_path=args.genotypes,
            source_manifest_path=args.source_manifest,
            source_bundle_path=bundle_path,
            intent_receipt_path=intent_path,
            post_run_receipt_path=post_path,
        )
    print(
        json.dumps(
            {
                "analysis_sha256": payload["analysis_sha256"],
                "output_directory": str(args.output_dir),
                "selected_candidate": payload["selection_decision"][
                    "selected_candidate"
                ],
                "writer_self_check": writer_self_check,
                "validation": validation,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

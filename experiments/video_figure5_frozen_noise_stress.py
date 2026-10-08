"""Frozen full-rank-noise stress test for the original GitHub Figure 5 model.

This is an independent, one-shot confirmation runner.  It reuses the exact
Figure 5 generator, orientation, alignment, and localization metrics from
``video_spca_paper_reproduction.py``.  It does not tune any solver parameter:
the paper-scale profile fixes alpha=9, rank=3, 500 iterations, tolerance 1e-6,
and the historical randomized settings (oversample=100, q=4, seed=42).

Dense iid Gaussian noise is scaled to exactly 20, 10, or 0 dB Frobenius SNR.
Each SNR has twenty predeclared confirmation seeds.  Data/source manifests and
the protocol lock are constructed before the first fit.  A changed source file
during execution aborts the run.  Existing output directories are never
overwritten.

``--profile smoke`` exercises the same solver settings on three tiny cases.
Smoke output is explicitly ineligible for scientific claims.  The formal
profile has no parameter, SNR, or seed override surface.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from experiments.video_spca_paper_reproduction import (
    ARXIV_ID,
    ARXIV_PDF_SHA256,
    ARXIV_VERSION,
    LEGACY_SOURCE_COMMIT,
    LEGACY_SOURCE_SHA256,
    NativeFactorFit,
    PaperVideoConfig,
    _localization_metrics,
    _paired_cosines,
    _pca_fit,
    _randomized_fit,
    align_factor_pair_sign_permutation_only,
    generate_figure5_dataset,
    normalized_full_frame_errors,
    summarize_full_frame_errors,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions
from rspca.metrics import relative_reconstruction_error
from rspca.vp import fit_vp_spca, vp_objective


FloatArray = NDArray[np.float64]

PROTOCOL_VERSION = 1
CANONICAL_FORMAL_ROOT_RELATIVE = Path(
    "results/video_figure5_frozen_noise_stress_formal_v1"
)
AUTODL_TIMING_ROOT = Path("/root/autodl-tmp/dhy")
FORMAL_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
EXPECTED_SOURCE_ROLE_SCHEMA = {
    "experiments/video_figure5_frozen_noise_stress.py": "stress_runner",
    "experiments/video_spca_paper_reproduction.py": (
        "original_figure5_generator_alignment_and_fit_adapters"
    ),
    "src/rspca/__init__.py": "package_initialization",
    "src/rspca/accelerated.py": "package_initialization_only",
    "src/rspca/cross_modal_gates.py": "package_initialization_only",
    "src/rspca/datasets.py": "array_and_canonical_hashing",
    "src/rspca/federated.py": "package_initialization_only",
    "src/rspca/fedrpca_audit.py": "package_initialization_only",
    "src/rspca/gauge.py": "package_initialization_only",
    "src/rspca/gauge_stress.py": "package_initialization_only",
    "src/rspca/metrics.py": "reconstruction_metric",
    "src/rspca/product_aggregation.py": "package_initialization_only",
    "src/rspca/randomized.py": "randomized_vp_spca_solver",
    "src/rspca/robust.py": "package_initialization_only",
    "src/rspca/statistical_gates.py": "package_initialization_only",
    "src/rspca/trust_audit.py": "package_initialization_only",
    "src/rspca/vp.py": "deterministic_vp_spca_solver",
    "tools/validate_video_figure5_frozen_noise_stress.py": (
        "independent_external_anchor_validator"
    ),
}
FROZEN_SHARED_CORE_SHA256 = {
    "src/rspca/vp.py": (
        "88a84aa99825357f3a16650bf3e56bfef54a850aa17880342334ba7ad402b953"
    ),
    "src/rspca/randomized.py": (
        "101833335c80b178f35b696bdc28dd8a4c0eb6a9fc5a565af802d83088d0823b"
    ),
}
FORMAL_SNR_SEED_BANK: tuple[tuple[float, tuple[int, ...]], ...] = (
    (20.0, tuple(range(83001, 83021))),
    (10.0, tuple(range(83101, 83121))),
    (0.0, tuple(range(83201, 83221))),
)
SMOKE_SNR_SEED_BANK: tuple[tuple[float, tuple[int, ...]], ...] = (
    (20.0, (83901,)),
    (10.0, (83902,)),
    (0.0, (83903,)),
)

REPORT_NAME = "frozen_noise_stress_report.json"
CASES_NAME = "frozen_noise_stress_cases.csv"
FRAMES_NAME = "frozen_noise_stress_full_frame_errors.csv"
FIGURE_NAME = "frozen_noise_stress_summary.png"
LOCK_NAME = "frozen_noise_stress_protocol_lock.json"
SOURCE_MANIFEST_NAME = "frozen_noise_stress_source_manifest.json"
DATA_MANIFEST_NAME = "frozen_noise_stress_data_manifest.json"
OUTPUT_MANIFEST_NAME = "frozen_noise_stress_output_manifest.json"
ANCHOR_MANIFEST_NAME = "frozen_noise_stress_anchor_manifest.json"
EXECUTION_RECEIPT_NAME = "frozen_noise_stress_execution_started_receipt.json"
SEED_REGISTRY_NAME = "frozen_noise_stress_seed_bank_registry.json"
FACTORS_NAME = "frozen_noise_stress_factors.npz"

_PAYLOAD_NAMES = frozenset(
    {
        REPORT_NAME,
        CASES_NAME,
        FRAMES_NAME,
        FIGURE_NAME,
        LOCK_NAME,
        SOURCE_MANIFEST_NAME,
        DATA_MANIFEST_NAME,
        EXECUTION_RECEIPT_NAME,
        SEED_REGISTRY_NAME,
        FACTORS_NAME,
    }
)
_OUTPUT_NAMES = _PAYLOAD_NAMES | {OUTPUT_MANIFEST_NAME}
_ANCHOR_PAYLOAD_NAMES = frozenset(
    {LOCK_NAME, SOURCE_MANIFEST_NAME, DATA_MANIFEST_NAME}
)
_ANCHOR_OUTPUT_NAMES = _ANCHOR_PAYLOAD_NAMES | {ANCHOR_MANIFEST_NAME}

DATA_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "manifest_kind",
        "profile",
        "orientation",
        "clean_array_sha256",
        "clean_shape",
        "cases",
        "precomputed_before_first_fit",
        "regenerated_and_hash_checked_before_each_fit",
        "data_manifest_body_sha256",
    }
)
DATA_CASE_KEYS = frozenset(
    {
        "snr_db",
        "noise_seed",
        "shape",
        "distribution",
        "target_snr_db",
        "achieved_snr_db",
        "noise_nonzero_fraction",
        "full_column_rank_witness",
        "array_sha256",
    }
)
CASE_RECORD_KEYS = frozenset(
    {
        "snr_db",
        "noise_seed",
        "method",
        "raw_solver_fit_call_runtime_seconds",
        "complete_method_runtime_seconds",
        "runtime_components",
        "fit_execution_order",
        "fit_execution_position",
        "factor_keys",
        "factor_step_size",
        "converged",
        "full_converged",
        "n_iter",
        "status",
        "final_full_stationarity",
        "tolerance",
        "stationarity_tolerance",
        "stationarity_met",
        "final_full_objective_recomputed",
        "last_solver_objective",
        "clean_relative_reconstruction_error",
        "observed_relative_reconstruction_error",
        "full_frame_normalized_rmse",
        "loading_sparsity_at_threshold",
        "loading_zero_threshold",
        "mode_and_dynamics_recovery",
        "localization",
        "randomized_diagnostics",
    }
)


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} exact schema mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class FrozenNoiseStressConfig:
    """Immutable profile resolved by :func:`formal_config` or `smoke_config`."""

    profile: str
    grid_size: int
    n_timesteps: int
    total_time: float
    n_components: int
    alpha: float
    max_iter: int
    tol: float
    randomized_oversample: int
    randomized_power_iters: int
    randomized_full_refine_iters: int
    randomized_seed: int
    snr_seed_bank: tuple[tuple[float, tuple[int, ...]], ...]
    support_threshold: float
    loading_zero_threshold: float
    bootstrap_draws: int
    bootstrap_seed: int
    dpi: int


@dataclass(frozen=True)
class NoisyFigure5Case:
    """One dense-noise stress case; truth is retained only for evaluation."""

    snr_db: float
    noise_seed: int
    clean: FloatArray
    noise: FloatArray
    observed: FloatArray
    achieved_snr_db: float
    leading_square_rank: int
    leading_square_min_singular_value: float


@dataclass(frozen=True)
class FrozenNoiseStressRun:
    """All pre-fit locks and post-fit records needed by the writer."""

    config: FrozenNoiseStressConfig
    source_manifest: Mapping[str, Any]
    data_manifest: Mapping[str, Any]
    protocol_lock: Mapping[str, Any]
    case_records: tuple[Mapping[str, Any], ...]
    frame_records: tuple[Mapping[str, Any], ...]
    gates: Mapping[str, Any]
    protocol_lock_file_sha256: str
    factor_arrays: Mapping[str, FloatArray]
    anchor_paths: Mapping[str, Path] | None = None
    execution_receipt: Mapping[str, Any] | None = None
    execution_receipt_path: Path | None = None
    seed_registry: Mapping[str, Any] | None = None
    seed_registry_path: Path | None = None
    bound_output_dir: Path | None = None


def formal_config() -> FrozenNoiseStressConfig:
    """Return the only admissible paper-scale confirmation protocol."""

    return FrozenNoiseStressConfig(
        profile="formal_confirmation",
        grid_size=200,
        n_timesteps=300,
        total_time=150.0,
        n_components=3,
        alpha=9.0,
        max_iter=500,
        tol=1e-6,
        randomized_oversample=100,
        randomized_power_iters=4,
        randomized_full_refine_iters=0,
        randomized_seed=42,
        snr_seed_bank=FORMAL_SNR_SEED_BANK,
        support_threshold=0.05,
        loading_zero_threshold=1e-10,
        bootstrap_draws=10000,
        bootstrap_seed=84001,
        dpi=150,
    )


def smoke_config() -> FrozenNoiseStressConfig:
    """Return a tiny plumbing profile with unchanged solver parameters."""

    return FrozenNoiseStressConfig(
        profile="smoke_not_evidence",
        grid_size=10,
        n_timesteps=36,
        total_time=150.0,
        n_components=3,
        alpha=9.0,
        max_iter=500,
        tol=1e-6,
        randomized_oversample=100,
        randomized_power_iters=4,
        randomized_full_refine_iters=0,
        randomized_seed=42,
        snr_seed_bank=SMOKE_SNR_SEED_BANK,
        support_threshold=0.05,
        loading_zero_threshold=1e-10,
        bootstrap_draws=200,
        bootstrap_seed=84901,
        dpi=72,
    )


def _expected_config(profile: str) -> FrozenNoiseStressConfig:
    if profile == "formal_confirmation":
        return formal_config()
    if profile == "smoke_not_evidence":
        return smoke_config()
    raise ValueError("unknown frozen noise-stress profile")


def validate_config(config: FrozenNoiseStressConfig) -> None:
    """Reject every mutated profile, including apparently harmless overrides."""

    if not isinstance(config, FrozenNoiseStressConfig):
        raise TypeError("config must be FrozenNoiseStressConfig")
    expected = _expected_config(config.profile)
    if config != expected:
        raise ValueError("frozen noise-stress configuration was modified")
    seeds = [
        seed
        for _, stratum_seeds in config.snr_seed_bank
        for seed in stratum_seeds
    ]
    if len(seeds) != len(set(seeds)):
        raise ValueError("noise seeds must be unique across SNR strata")
    if config.grid_size**2 < config.n_timesteps:
        raise ValueError("rank witness requires at least as many pixels as frames")


def _paper_config(config: FrozenNoiseStressConfig) -> PaperVideoConfig:
    return PaperVideoConfig(
        grid_size=config.grid_size,
        n_timesteps=config.n_timesteps,
        total_time=config.total_time,
        n_components=config.n_components,
        figure5_alpha=config.alpha,
        randomized_oversample=config.randomized_oversample,
        randomized_power_iters=config.randomized_power_iters,
        randomized_full_refine_iters=config.randomized_full_refine_iters,
        randomized_seed=config.randomized_seed,
        max_iter=config.max_iter,
        tol=config.tol,
        figure6_frame_index=min(100, config.n_timesteps - 1),
        dpi=config.dpi,
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _strict_json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(
            stream,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-standard JSON constant: {value}")
            ),
        )


def _serialized_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _git_output(repo_root: Path, arguments: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def _source_manifest(repo_root: Path) -> dict[str, Any]:
    """Bind the exact source path/role schema before any fit starts."""

    # Bind imports made by this experiment, not every module imported by the
    # host process (pytest, notebooks, and other experiments share a process).
    # Check the declared files' rspca imports first so a genuinely new solver
    # dependency cannot disappear merely because we filter unrelated modules.
    for relative in EXPECTED_SOURCE_ROLE_SCHEMA:
        if relative.startswith("tools/"):
            continue
        source = repo_root / relative
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=relative)
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if node.level and relative.startswith("src/rspca/"):
                    module = "rspca." + module
                names = [module]
                if module.rstrip(".") == "rspca":
                    names += ["rspca." + alias.name for alias in node.names]
            for name in names:
                if not name.startswith("rspca."):
                    continue
                dependency = "src/" + name.replace(".", "/") + ".py"
                if (repo_root / dependency).is_file() and dependency not in EXPECTED_SOURCE_ROLE_SCHEMA:
                    raise RuntimeError(
                        f"experiment dependency is outside fixed source schema: {dependency}"
                    )

    loaded_modules_by_relative: dict[str, list[str]] = {}
    for module_name, module in sorted(tuple(sys.modules.items())):
        raw_path = getattr(module, "__file__", None)
        if raw_path is None:
            continue
        try:
            path = Path(raw_path).resolve()
            relative = path.relative_to(repo_root).as_posix()
        except (OSError, ValueError):
            continue
        if relative not in EXPECTED_SOURCE_ROLE_SCHEMA:
            continue
        if path.suffix != ".py" or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"loaded source is missing or invalid: {relative}")
        loaded_modules_by_relative.setdefault(relative, []).append(module_name)

    required_loaded = set(EXPECTED_SOURCE_ROLE_SCHEMA) - {
        "tools/validate_video_figure5_frozen_noise_stress.py"
    }
    missing_loaded = sorted(required_loaded - set(loaded_modules_by_relative))
    if missing_loaded:
        raise RuntimeError(f"fixed eager-import source schema is incomplete: {missing_loaded}")
    files: dict[str, Any] = {}
    for relative, role in EXPECTED_SOURCE_ROLE_SCHEMA.items():
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"required source is missing or a symlink: {relative}")
        files[relative] = {
            "sha256": _file_sha256(path),
            "size_bytes": int(path.stat().st_size),
            "role": role,
            "module_names": sorted(set(loaded_modules_by_relative.get(relative, []))),
        }
    if {path: record["role"] for path, record in files.items()} != (
        EXPECTED_SOURCE_ROLE_SCHEMA
    ):
        raise RuntimeError("fixed source path/role schema changed")
    for relative, expected_sha256 in FROZEN_SHARED_CORE_SHA256.items():
        if files[relative]["sha256"] != expected_sha256:
            raise RuntimeError(
                f"shared-core freeze hash mismatch for {relative}; refusing to run"
            )
    status = _git_output(repo_root, ["status", "--porcelain", "--untracked-files=no"])
    body: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "frozen_figure5_noise_stress_source_anchor",
        "files": files,
        "paper": {
            "arxiv_id": ARXIV_ID,
            "arxiv_version": ARXIV_VERSION,
            "official_pdf_sha256": ARXIV_PDF_SHA256,
            "section": "6.1 Multiscale Video Example, Figure 5",
        },
        "legacy_implementation": {
            "commit": LEGACY_SOURCE_COMMIT,
            "sha256": dict(LEGACY_SOURCE_SHA256),
        },
        "git_head": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "tracked_worktree_dirty": None if status is None else bool(status),
        "runtime_versions": runtime_versions(),
        "federated_execution_path_used": False,
        "package_eager_imports_bound": True,
        "package_initialization_only_is_not_algorithm_execution": True,
        "exact_source_path_role_schema": True,
        "source_path_role_schema_sha256": canonical_sha256(
            {"path_roles": EXPECTED_SOURCE_ROLE_SCHEMA}
        ),
        "frozen_shared_core_sha256": dict(FROZEN_SHARED_CORE_SHA256),
    }
    body["source_manifest_body_sha256"] = canonical_sha256(body)
    return body


def _assert_source_unchanged(source_manifest: Mapping[str, Any], repo_root: Path) -> None:
    for relative, record in source_manifest["files"].items():
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source changed during run: {relative}")
        if _file_sha256(path) != record["sha256"]:
            raise RuntimeError(f"source changed during run: {relative}")


def generate_noisy_figure5_case(
    clean: FloatArray,
    *,
    snr_db: float,
    noise_seed: int,
) -> NoisyFigure5Case:
    """Add iid Gaussian noise with exact Frobenius SNR and a rank witness."""

    clean_array = np.asarray(clean, dtype=np.float64)
    if clean_array.ndim != 2 or not np.isfinite(clean_array).all():
        raise ValueError("clean must be a finite two-dimensional matrix")
    if clean_array.shape[0] < clean_array.shape[1]:
        raise ValueError("a leading full-column-rank witness requires rows >= columns")
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, (int, np.integer)):
        raise TypeError("noise_seed must be an integer")
    if int(noise_seed) < 0:
        raise ValueError("noise_seed must be nonnegative")
    clean_norm = float(np.linalg.norm(clean_array))
    if clean_norm <= np.finfo(np.float64).eps:
        raise ValueError("clean Figure 5 video must have positive Frobenius norm")

    rng = np.random.default_rng(int(noise_seed))
    noise = np.asarray(rng.standard_normal(clean_array.shape), dtype=np.float64)
    noise_norm = float(np.linalg.norm(noise))
    target_noise_norm = clean_norm / (10.0 ** (float(snr_db) / 20.0))
    noise *= target_noise_norm / noise_norm
    observed = np.ascontiguousarray(clean_array + noise)
    achieved = float(20.0 * np.log10(clean_norm / np.linalg.norm(noise)))

    witness = np.ascontiguousarray(noise[: clean_array.shape[1], :])
    singular_values = np.linalg.svd(witness, compute_uv=False)
    tolerance = singular_values[0] * max(witness.shape) * np.finfo(np.float64).eps
    rank = int(np.count_nonzero(singular_values > tolerance))
    if rank != clean_array.shape[1]:
        raise RuntimeError("generated Gaussian noise failed the full-column-rank witness")
    return NoisyFigure5Case(
        snr_db=float(snr_db),
        noise_seed=int(noise_seed),
        clean=np.ascontiguousarray(clean_array),
        noise=np.ascontiguousarray(noise),
        observed=observed,
        achieved_snr_db=achieved,
        leading_square_rank=rank,
        leading_square_min_singular_value=float(singular_values[-1]),
    )


def _data_record(case: NoisyFigure5Case) -> dict[str, Any]:
    record = {
        "snr_db": case.snr_db,
        "noise_seed": case.noise_seed,
        "shape": list(case.observed.shape),
        "distribution": "iid_standard_normal_scaled_to_exact_frobenius_snr",
        "target_snr_db": case.snr_db,
        "achieved_snr_db": case.achieved_snr_db,
        "noise_nonzero_fraction": float(np.mean(case.noise != 0.0)),
        "full_column_rank_witness": {
            "leading_square_shape": [case.observed.shape[1], case.observed.shape[1]],
            "rank": case.leading_square_rank,
            "minimum_singular_value": case.leading_square_min_singular_value,
            "passed": bool(case.leading_square_rank == case.observed.shape[1]),
        },
        "array_sha256": {
            "clean": array_sha256(case.clean),
            "noise": array_sha256(case.noise),
            "observed": array_sha256(case.observed),
        },
    }
    _require_exact_keys(record, DATA_CASE_KEYS, label="data case")
    _require_exact_keys(
        record["full_column_rank_witness"],
        frozenset({"leading_square_shape", "rank", "minimum_singular_value", "passed"}),
        label="data case rank witness",
    )
    _require_exact_keys(
        record["array_sha256"],
        frozenset({"clean", "noise", "observed"}),
        label="data case array hashes",
    )
    return record


def _formal_seed_history_audit(
    repo_root: Path,
    *,
    excluded_directories: Sequence[Path] = (),
) -> dict[str, Any]:
    """Inventory semantic outcome roots and fail closed on any ambiguity."""

    seeds = sorted(seed for _, values in FORMAL_SNR_SEED_BANK for seed in values)
    seed_set = set(seeds)
    pattern = re.compile(
        r"(?<![0-9A-Za-z])(" + "|".join(str(seed) for seed in seeds) + r")(?![0-9A-Za-z])"
    )
    root_relatives = ("results", "assets/evaluation", "artifacts")
    roots = tuple(repo_root / relative for relative in root_relatives)
    text_suffixes = {
        ".csv",
        ".json",
        ".jsonl",
        ".log",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    matches: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    root_states: list[dict[str, Any]] = []
    exclusions = tuple(path.resolve() for path in excluded_directories)
    for root in roots:
        relative_root = root.relative_to(repo_root).as_posix()
        if not root.exists():
            root_states.append({"path": relative_root, "exists": False})
            continue
        if root.is_symlink() or not root.is_dir():
            raise RuntimeError(f"freshness root is not a real directory: {relative_root}")
        root_states.append({"path": relative_root, "exists": True})
        for path in sorted(root.rglob("*")):
            metadata = path.lstat()
            is_symlink = stat.S_ISLNK(metadata.st_mode)
            if is_symlink:
                link_target = os.readlink(path)
                try:
                    resolved_path = path.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise RuntimeError(
                        "freshness inventory contains an unreadable/broken symlink: "
                        f"{path.relative_to(repo_root).as_posix()}"
                    ) from error
                target_metadata = resolved_path.stat()
                if not stat.S_ISREG(target_metadata.st_mode):
                    raise RuntimeError(
                        "freshness symlink target is not a regular file: "
                        f"{path.relative_to(repo_root).as_posix()}"
                    )
            else:
                link_target = None
                resolved_path = path.resolve(strict=True)
                target_metadata = metadata
            if stat.S_ISDIR(metadata.st_mode):
                continue
            if not is_symlink and not stat.S_ISREG(metadata.st_mode):
                raise RuntimeError(
                    f"freshness inventory contains a non-regular entry: "
                    f"{path.relative_to(repo_root).as_posix()}"
                )
            lexical_path = path.absolute()
            if any(
                lexical_path == excluded or lexical_path.is_relative_to(excluded)
                for excluded in exclusions
            ):
                continue
            relative_path = path.relative_to(repo_root).as_posix()
            suffix = path.suffix.lower()
            with path.open("rb") as binary_stream:
                leading_magic = binary_stream.read(4)
            is_appledouble_binary = (
                path.name.startswith("._") and leading_magic == b"\x00\x05\x16\x07"
            )
            semantic_parse = suffix in text_suffixes and not is_appledouble_binary
            inventory_record: dict[str, Any] = {
                "path": relative_path,
                "entry_kind": "symlink" if is_symlink else "regular_file",
                "sha256": _file_sha256(path),
                "size_bytes": int(target_metadata.st_size),
                "semantic_parse": semantic_parse,
                "content_class": (
                    "appledouble_binary_metadata"
                    if is_appledouble_binary
                    else ("semantic_text" if semantic_parse else "opaque_binary")
                ),
            }
            if is_symlink:
                inventory_record.update(
                    {
                        "link_target": link_target,
                        "link_target_text_sha256": sha256(
                            str(link_target).encode("utf-8")
                        ).hexdigest(),
                        "resolved_target": str(resolved_path),
                        "symlink_lstat_size_bytes": int(metadata.st_size),
                    }
                )
            inventory.append(inventory_record)
            if not semantic_parse:
                continue
            # Decode and structured parse failures are protocol failures.  A
            # skipped unreadable file would make freshness a fail-open claim.
            text = path.read_text(encoding="utf-8")
            found_in_file: set[int] = set()

            def collect_seed_values(value: Any, *, seed_context: bool = False) -> None:
                if isinstance(value, Mapping):
                    for key, item in value.items():
                        child_context = seed_context or "seed" in str(key).lower()
                        collect_seed_values(item, seed_context=child_context)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        collect_seed_values(item, seed_context=seed_context)
                elif seed_context and not isinstance(value, (bool, np.bool_)):
                    candidate: int | None = None
                    if isinstance(value, (int, np.integer)):
                        candidate = int(value)
                    elif isinstance(value, (float, np.floating)) and float(value).is_integer():
                        candidate = int(value)
                    elif isinstance(value, str) and value.isdigit():
                        candidate = int(value)
                    if candidate in seed_set:
                        found_in_file.add(candidate)

            if suffix == ".json":
                collect_seed_values(json.loads(text))
            elif suffix == ".jsonl":
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"invalid JSONL at {relative_path}:{line_number}"
                        ) from error
                    collect_seed_values(payload)
            elif suffix == ".csv":
                rows = csv.DictReader(text.splitlines(), strict=True)
                seed_columns = [
                    name for name in (rows.fieldnames or []) if "seed" in name.lower()
                ]
                for row in rows:
                    if None in row:
                        raise ValueError(f"malformed CSV row in {relative_path}")
                    for name in seed_columns:
                        value = row.get(name, "")
                        if value is not None and value.isdigit():
                            candidate = int(value)
                            if candidate in seed_set:
                                found_in_file.add(candidate)
            else:
                for line in text.splitlines():
                    if "seed" not in line.lower():
                        continue
                    found_in_file.update(int(value) for value in pattern.findall(line))
            if found_in_file:
                matches.append(
                    {
                        "path": path.relative_to(repo_root).as_posix(),
                        "seeds": sorted(found_in_file),
                    }
                )
    body = {
        "audit_version": 1,
        "searched_roots": list(root_relatives),
        "root_states": root_states,
        "text_suffixes": sorted(text_suffixes),
        "checked_files": len(inventory),
        "inventory": inventory,
        "inventory_sha256": canonical_sha256({"files": inventory}),
        "excluded_directories_resolved": [str(path) for path in exclusions],
        "locked_formal_seeds": seeds,
        "matches": matches,
        "no_prior_outcome_artifact_matches": not matches,
    }
    if matches:
        raise RuntimeError(
            "formal noise-seed history audit found prior outcome artifacts"
        )
    body["seed_history_audit_body_sha256"] = canonical_sha256(body)
    return body


def _canonical_formal_paths(repo_root: Path) -> dict[str, Path]:
    root = (repo_root / CANONICAL_FORMAL_ROOT_RELATIVE).resolve()
    return {
        "root": root,
        "anchor": root / "anchor",
        "output": root / "formal_output",
        "seed_registry": root / SEED_REGISTRY_NAME,
    }


def _cpu_model_linux() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file() or cpuinfo.is_symlink():
        return ""
    for line in cpuinfo.read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return ""


def _timing_environment_snapshot(
    repo_root: Path,
    *,
    enforce_formal: bool,
) -> dict[str, Any]:
    """Capture the hardware/thread boundary used by the runtime gate."""

    try:
        import threadpoolctl
    except ImportError as error:  # pragma: no cover - formal fail-closed branch
        if enforce_formal:
            raise RuntimeError("formal timing requires threadpoolctl") from error
        threadpool_version = None
        pools: list[dict[str, Any]] = []
    else:
        threadpool_version = threadpoolctl.__version__
        pools = []
        for entry in threadpoolctl.threadpool_info():
            pools.append(
                {
                    "user_api": entry.get("user_api"),
                    "internal_api": entry.get("internal_api"),
                    "prefix": entry.get("prefix"),
                    "filepath": entry.get("filepath"),
                    "version": entry.get("version"),
                    "num_threads": int(entry.get("num_threads", -1)),
                    "threading_layer": entry.get("threading_layer"),
                    "architecture": entry.get("architecture"),
                }
            )
        pools.sort(key=lambda item: (str(item["filepath"]), str(item["prefix"])))
    affinity = (
        sorted(int(value) for value in os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    try:
        numpy_backend = np.__config__.show(mode="dicts")
    except TypeError:  # pragma: no cover - older NumPy cannot be formal-qualified
        numpy_backend = None
    numpy_backend_json = json.loads(json.dumps(numpy_backend, default=str))
    thread_environment = {name: os.environ.get(name) for name in FORMAL_THREAD_ENVIRONMENT}
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "timing_scope": "autodl_bjb2_single_cpu_single_blas_thread_only",
        "autodl_timing_marker": os.environ.get(
            "RSPCA_AUTODL_TIMING_CONFIRMATION"
        ),
        "repo_root_resolved": str(repo_root.resolve()),
        "required_autodl_root": str(AUTODL_TIMING_ROOT),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "hostname": platform.node(),
        "cpu_model": _cpu_model_linux(),
        "cpu_affinity": affinity,
        "thread_environment": thread_environment,
        "threadpoolctl_version": threadpool_version,
        "threadpool_info": pools,
        "numpy_backend_configuration": numpy_backend_json,
        "matplotlib_backend_environment": os.environ.get("MPLBACKEND"),
    }
    reasons: list[str] = []
    if snapshot["platform_system"] != "Linux":
        reasons.append("formal timing is restricted to Linux AutoDL")
    try:
        repo_root.resolve().relative_to(AUTODL_TIMING_ROOT)
    except ValueError:
        reasons.append("repository is outside /root/autodl-tmp/dhy")
    if snapshot["autodl_timing_marker"] != "1":
        reasons.append("RSPCA_AUTODL_TIMING_CONFIRMATION must equal 1")
    if str(snapshot["matplotlib_backend_environment"]).lower() != "agg":
        reasons.append("MPLBACKEND must equal Agg")
    for name, value in thread_environment.items():
        if value != "1":
            reasons.append(f"{name} must equal 1 before Python starts")
    if len(affinity) != 1:
        reasons.append("CPU affinity must contain exactly one logical CPU")
    if not snapshot["cpu_model"]:
        reasons.append("Linux CPU model is unavailable")
    if not pools:
        reasons.append("threadpoolctl found no numerical backend")
    elif any(int(entry["num_threads"]) != 1 for entry in pools):
        reasons.append("every loaded numerical thread pool must report one thread")
    snapshot["formal_timing_qualified"] = not reasons
    snapshot["qualification_failures"] = reasons
    snapshot["timing_environment_body_sha256"] = canonical_sha256(snapshot)
    if enforce_formal and reasons:
        raise RuntimeError("formal timing environment is not qualified: " + "; ".join(reasons))
    return snapshot


def _claim_formal_seed_bank_registry(
    repo_root: Path,
    freshness_audit: Mapping[str, Any],
    timing_environment: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Atomically and globally consume the only formal seed bank once."""

    paths = _canonical_formal_paths(repo_root)
    root = paths["root"]
    root.parent.mkdir(parents=True, exist_ok=True)
    try:
        root.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"canonical formal root already exists; seed bank is consumed: {root}"
        ) from error
    body: dict[str, Any] = {
        "schema_version": 1,
        "registry_kind": "create_once_global_formal_noise_seed_bank_claim",
        "canonical_formal_root_resolved": str(root),
        "canonical_anchor_dir_resolved": str(paths["anchor"]),
        "canonical_output_dir_resolved": str(paths["output"]),
        "formal_snr_seed_bank": [
            [snr, list(seeds)] for snr, seeds in FORMAL_SNR_SEED_BANK
        ],
        "freshness_inventory_sha256": freshness_audit["inventory_sha256"],
        "freshness_audit_body_sha256": freshness_audit[
            "seed_history_audit_body_sha256"
        ],
        "timing_environment_body_sha256": timing_environment[
            "timing_environment_body_sha256"
        ],
        "created_with_exclusive_open_before_anchor_and_any_fit": True,
        "reuse_or_reissue_forbidden": True,
    }
    body["seed_registry_body_sha256"] = canonical_sha256(body)
    registry_path = paths["seed_registry"]
    try:
        with registry_path.open("xb") as stream:
            stream.write(_serialized_json_bytes(body))
    except BaseException:
        # The create-once root intentionally remains as a consumed tombstone.
        raise
    return body, registry_path


def _preflight_data_manifest(
    config: FrozenNoiseStressConfig,
    clean: FloatArray,
) -> dict[str, Any]:
    """Hash every case before fitting, then regenerate-and-check during fitting."""

    cases: list[dict[str, Any]] = []
    for snr_db, seeds in config.snr_seed_bank:
        for noise_seed in seeds:
            case = generate_noisy_figure5_case(
                clean, snr_db=snr_db, noise_seed=noise_seed
            )
            cases.append(_data_record(case))
    body: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "frozen_figure5_full_rank_noise_data",
        "profile": config.profile,
        "orientation": "spatial_pixels_x_time_snapshots",
        "clean_array_sha256": array_sha256(clean),
        "clean_shape": list(clean.shape),
        "cases": cases,
        "precomputed_before_first_fit": True,
        "regenerated_and_hash_checked_before_each_fit": True,
    }
    body["data_manifest_body_sha256"] = canonical_sha256(body)
    _require_exact_keys(body, DATA_MANIFEST_KEYS, label="data manifest")
    return body


def _gate_specification() -> dict[str, Any]:
    """Return the predeclared, audit-recommended RND-versus-VP gates."""

    return {
        "comparison": "RND-VP-SPCA_minus_or_ratio_to_VP-SPCA",
        "speed_runtime_definition": {
            "VP-SPCA": "PCA_and_sigma_preparation_plus_VP_solver_fit_call",
            "RND-VP-SPCA": (
                "full_randomized_fit_call_including_range_compress_optimization_"
                "and_full_diagnostics"
            ),
            "raw_solver_fit_call_runtime_also_reported": True,
        },
        "stratification": "each_SNR_evaluated_separately",
        "formal_pairs_per_snr": 20,
        "paired_bootstrap": {
            "confidence_level": 0.95,
            "draws": 10000,
            "resampling_unit": "noise_seed_pair_within_snr",
            "mean_statistic_metrics": [
                "clean_relative_error_ratio",
                "max_frame_rmse_ratio",
                "spatial_mean_cosine_loss",
                "spatial_worst_cosine_loss",
                "temporal_mean_cosine_loss",
                "temporal_worst_cosine_loss",
                "off_support_absolute_increment",
                "cross_talk_absolute_increment",
                "loading_sparsity_absolute_difference",
                "relative_full_objective_excess",
            ],
            "median_statistic_metrics": ["complete_method_runtime_speedup"],
        },
        "thresholds": {
            "clean_relative_error_ratio": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 1.02,
            },
            "max_frame_rmse_ratio": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 1.02,
            },
            "spatial_mean_cosine_loss": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.002,
            },
            "spatial_worst_cosine_loss": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.002,
            },
            "temporal_mean_cosine_loss": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.002,
            },
            "temporal_worst_cosine_loss": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.002,
            },
            "off_support_absolute_increment": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.01,
            },
            "cross_talk_absolute_increment": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.01,
            },
            "loading_sparsity_absolute_difference": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 0.01,
            },
            "relative_full_objective_excess": {
                "ci_bound": "upper",
                "operator": "<=",
                "value": 1e-4,
            },
            "complete_method_runtime_speedup": {
                "ci_bound": "lower",
                "operator": ">",
                "value": 10.0,
            },
        },
        "failure_policy": {
            "stratum_failure": (
                "report the failed SNR boundary without pooling or relaxing a threshold"
            ),
            "solver_nonconvergence": (
                "quality comparisons are fixed-budget only; no strict converged claim"
            ),
        },
    }


def _protocol_lock(
    config: FrozenNoiseStressConfig,
    source_manifest: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    *,
    formal_anchor_dir: Path | None = None,
    formal_output_dir: Path | None = None,
    seed_history_audit: Mapping[str, Any] | None = None,
    seed_registry: Mapping[str, Any] | None = None,
    timing_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config.profile == "formal_confirmation":
        if formal_anchor_dir is None or formal_output_dir is None:
            raise ValueError("formal lock requires unique anchor and output paths")
        if (
            seed_history_audit is None
            or seed_history_audit.get("no_prior_outcome_artifact_matches") is not True
        ):
            raise ValueError("formal lock requires a passing fresh-seed history audit")
        if seed_registry is None or seed_registry.get(
            "created_with_exclusive_open_before_anchor_and_any_fit"
        ) is not True:
            raise ValueError("formal lock requires an atomically claimed seed registry")
        if timing_environment is None or timing_environment.get(
            "formal_timing_qualified"
        ) is not True:
            raise ValueError("formal lock requires a qualified AutoDL timing environment")
        execution_authorization: Mapping[str, Any] = {
            "formal_anchor_dir_resolved": str(formal_anchor_dir.resolve()),
            "exclusive_output_dir_resolved": str(formal_output_dir.resolve()),
            "pre_fit_consumption_receipt": EXECUTION_RECEIPT_NAME,
            "receipt_created_with_exclusive_open_before_any_fit": True,
            "reuse_after_receipt_exists": "forbidden_fail_closed",
        }
    else:
        if formal_anchor_dir is not None or formal_output_dir is not None:
            raise ValueError("smoke lock cannot contain formal path authorization")
        if seed_history_audit is not None or seed_registry is not None:
            raise ValueError("smoke lock cannot claim formal freshness or seed registry")
        if timing_environment is None:
            raise ValueError("smoke lock must disclose its nonqualifying timing environment")
        execution_authorization = {
            "formal_anchor_dir_resolved": None,
            "exclusive_output_dir_resolved": None,
            "pre_fit_consumption_receipt": None,
            "receipt_created_with_exclusive_open_before_any_fit": False,
            "reuse_after_receipt_exists": "not_applicable_smoke",
        }
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "lock_kind": "figure5_frozen_full_rank_noise_one_shot_confirmation",
        "profile": config.profile,
        "config": asdict(config),
        "source_manifest_body_sha256": source_manifest[
            "source_manifest_body_sha256"
        ],
        "data_manifest_body_sha256": data_manifest["data_manifest_body_sha256"],
        "gate_specification": _gate_specification(),
        "execution_authorization": execution_authorization,
        "fresh_seed_history_audit": seed_history_audit,
        "seed_registry_body_sha256": (
            None if seed_registry is None else seed_registry["seed_registry_body_sha256"]
        ),
        "seed_registry_file_sha256": (
            None
            if seed_registry is None
            else sha256(_serialized_json_bytes(seed_registry)).hexdigest()
        ),
        "timing_environment": timing_environment,
        "protocol": {
            "parameter_selection": "none",
            "truth_used_during_fit": False,
            "truth_used_for_metrics_only": True,
            "one_shot_confirmation": config.profile == "formal_confirmation",
            "formal_cli_override_surface": "none",
            "orientation": "spatial_pixels_x_time_snapshots",
            "generator": (
                "experiments.video_spca_paper_reproduction.generate_figure5_dataset"
            ),
            "alignment": "Hungarian permutation and signs only",
            "post_hoc_amplitude_fit": False,
            "fit_order": (
                "PCA first; VP-SPCA then RND-VP-SPCA for odd noise seeds; "
                "RND-VP-SPCA then VP-SPCA for even noise seeds"
            ),
            "sparse_method_order_balance_per_formal_snr": {
                "VP_before_RND": 10,
                "RND_before_VP": 10,
            },
            "runtime_definition": {
                "raw": "wall time of each solver fit call",
                "speed_gate": (
                    "complete method cost: VP includes PCA/sigma preparation; "
                    "RND includes range/compress/optimization/full diagnostics"
                ),
            },
            "execution_policy": "sequential_fits_without_parallel_runtime_contention",
            "runtime_gate_environment": (
                "AutoDL /root/autodl-tmp/dhy only; one CPU affinity; all BLAS/OpenMP "
                "thread pools fixed to one before Python startup"
            ),
            "formal_results_may_not_be_reused_for_parameter_selection": True,
        },
        "formal_eligibility": {
            "paper_scale_required": True,
            "exactly_twenty_paired_seeds_per_snr_required": True,
            "all_three_snr_strata_required": True,
            "smoke_is_evidence": False,
        },
    }
    body["protocol_lock_body_sha256"] = canonical_sha256(body)
    return body


def _verify_manifest_body(
    payload: Mapping[str, Any], digest_field: str, *, label: str
) -> None:
    body = dict(payload)
    recorded = body.pop(digest_field, None)
    if recorded != canonical_sha256(body):
        raise ValueError(f"{label} canonical digest mismatch")


def validate_formal_anchor_directory(
    anchor_dir: str | Path,
    *,
    verify_current_source: bool = True,
) -> dict[str, Any]:
    """Validate the independent pre-fit formal anchor directory."""

    anchor = Path(anchor_dir)
    repo_root = Path(__file__).resolve().parents[1]
    canonical = _canonical_formal_paths(repo_root)
    if anchor.resolve() != canonical["anchor"]:
        raise ValueError("formal anchor is outside the canonical create-once root")
    if anchor.is_symlink() or not anchor.is_dir():
        raise ValueError("anchor path must be a real directory, not a symlink")
    entries = list(anchor.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"anchor contains a non-regular entry: {entry.name}")
    if {entry.name for entry in entries} != _ANCHOR_OUTPUT_NAMES:
        raise ValueError("formal anchor does not have exact closure")
    anchor_manifest = _strict_json_load(anchor / ANCHOR_MANIFEST_NAME)
    _verify_manifest_body(
        anchor_manifest, "anchor_manifest_body_sha256", label="anchor manifest"
    )
    if set(anchor_manifest.get("files", {})) != _ANCHOR_PAYLOAD_NAMES:
        raise ValueError("anchor manifest payload set mismatch")
    for name, record in anchor_manifest["files"].items():
        path = anchor / name
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"formal anchor SHA-256 mismatch for {name}")
        if path.stat().st_size != record.get("size_bytes"):
            raise ValueError(f"formal anchor size mismatch for {name}")
    source = _strict_json_load(anchor / SOURCE_MANIFEST_NAME)
    data = _strict_json_load(anchor / DATA_MANIFEST_NAME)
    lock = _strict_json_load(anchor / LOCK_NAME)
    _verify_manifest_body(
        source, "source_manifest_body_sha256", label="source manifest"
    )
    _verify_manifest_body(data, "data_manifest_body_sha256", label="data manifest")
    _verify_manifest_body(lock, "protocol_lock_body_sha256", label="protocol lock")
    _require_exact_keys(data, DATA_MANIFEST_KEYS, label="formal data manifest")
    for case_record in data.get("cases", []):
        _require_exact_keys(case_record, DATA_CASE_KEYS, label="formal data case")
    if canonical_sha256(lock.get("config", {})) != canonical_sha256(
        asdict(formal_config())
    ):
        raise ValueError("formal anchor configuration changed")
    if lock.get("profile") != "formal_confirmation":
        raise ValueError("formal anchor profile changed")
    if lock.get("gate_specification") != _gate_specification():
        raise ValueError("formal anchor gate specification changed")
    seed_audit = lock.get("fresh_seed_history_audit")
    if not isinstance(seed_audit, Mapping):
        raise ValueError("formal fresh-seed history audit is missing")
    seed_audit_body = dict(seed_audit)
    seed_audit_digest = seed_audit_body.pop("seed_history_audit_body_sha256", None)
    if seed_audit_digest != canonical_sha256(seed_audit_body):
        raise ValueError("formal fresh-seed history audit digest mismatch")
    if seed_audit.get("no_prior_outcome_artifact_matches") is not True or seed_audit.get(
        "matches"
    ) != []:
        raise ValueError("formal seeds were previously present in outcome artifacts")
    if seed_audit.get("locked_formal_seeds") != sorted(
        seed for _, seeds in FORMAL_SNR_SEED_BANK for seed in seeds
    ):
        raise ValueError("formal fresh-seed audit seed bank changed")
    if seed_audit.get("searched_roots") != [
        "results",
        "assets/evaluation",
        "artifacts",
    ]:
        raise ValueError("formal freshness semantic roots changed")
    if seed_audit.get("inventory_sha256") != canonical_sha256(
        {"files": seed_audit.get("inventory")}
    ):
        raise ValueError("formal freshness inventory digest mismatch")
    timing_environment = lock.get("timing_environment")
    if not isinstance(timing_environment, Mapping):
        raise ValueError("formal timing environment is missing")
    timing_body = dict(timing_environment)
    timing_digest = timing_body.pop("timing_environment_body_sha256", None)
    if timing_digest != canonical_sha256(timing_body):
        raise ValueError("formal timing environment digest mismatch")
    if timing_environment.get("formal_timing_qualified") is not True:
        raise ValueError("formal timing environment is not qualified")
    registry_path = canonical["seed_registry"]
    if registry_path.is_symlink() or not registry_path.is_file():
        raise ValueError("canonical seed-bank registry is missing")
    seed_registry = _strict_json_load(registry_path)
    _verify_manifest_body(
        seed_registry, "seed_registry_body_sha256", label="seed registry"
    )
    if seed_registry.get("formal_snr_seed_bank") != [
        [snr, list(seeds)] for snr, seeds in FORMAL_SNR_SEED_BANK
    ]:
        raise ValueError("formal seed registry bank changed")
    if seed_registry.get("canonical_formal_root_resolved") != str(
        canonical["root"]
    ) or seed_registry.get("canonical_anchor_dir_resolved") != str(
        canonical["anchor"]
    ) or seed_registry.get("canonical_output_dir_resolved") != str(
        canonical["output"]
    ):
        raise ValueError("formal seed registry canonical paths changed")
    if seed_registry.get("freshness_inventory_sha256") != seed_audit.get(
        "inventory_sha256"
    ):
        raise ValueError("seed registry freshness inventory link mismatch")
    if seed_registry.get("freshness_audit_body_sha256") != seed_audit.get(
        "seed_history_audit_body_sha256"
    ):
        raise ValueError("seed registry freshness-audit link mismatch")
    if seed_registry.get("timing_environment_body_sha256") != timing_environment.get(
        "timing_environment_body_sha256"
    ):
        raise ValueError("seed registry timing-environment link mismatch")
    if lock.get("seed_registry_body_sha256") != seed_registry.get(
        "seed_registry_body_sha256"
    ):
        raise ValueError("formal lock-to-seed-registry body link mismatch")
    if lock.get("seed_registry_file_sha256") != _file_sha256(registry_path):
        raise ValueError("formal lock-to-seed-registry file link mismatch")
    authorization = lock.get("execution_authorization", {})
    if authorization.get("formal_anchor_dir_resolved") != str(anchor.resolve()):
        raise ValueError("formal anchor path differs from its lock")
    output_text = authorization.get("exclusive_output_dir_resolved")
    if not isinstance(output_text, str) or not Path(output_text).is_absolute():
        raise ValueError("formal output path is not bound in the lock")
    if output_text != str(canonical["output"]):
        raise ValueError("formal output path is not the canonical output")
    if authorization.get("pre_fit_consumption_receipt") != EXECUTION_RECEIPT_NAME:
        raise ValueError("formal lock consumption-receipt contract changed")
    if lock.get("source_manifest_body_sha256") != source.get(
        "source_manifest_body_sha256"
    ):
        raise ValueError("formal lock-to-source link mismatch")
    if lock.get("data_manifest_body_sha256") != data.get(
        "data_manifest_body_sha256"
    ):
        raise ValueError("formal lock-to-data link mismatch")
    expected_pairs = {
        (snr, seed)
        for snr, seeds in FORMAL_SNR_SEED_BANK
        for seed in seeds
    }
    actual_pairs = {
        (float(record["snr_db"]), int(record["noise_seed"]))
        for record in data.get("cases", [])
    }
    if actual_pairs != expected_pairs or len(data.get("cases", [])) != len(
        expected_pairs
    ):
        raise ValueError("formal data manifest seed bank changed")
    if source.get("frozen_shared_core_sha256") != FROZEN_SHARED_CORE_SHA256:
        raise ValueError("formal shared-core anchor changed")
    if {
        path: record.get("role") for path, record in source.get("files", {}).items()
    } != EXPECTED_SOURCE_ROLE_SCHEMA:
        raise ValueError("formal exact source path/role schema changed")
    if verify_current_source:
        _assert_source_unchanged(source, repo_root)
    return {
        "source_manifest": source,
        "data_manifest": data,
        "protocol_lock": lock,
        "protocol_lock_file_sha256": _file_sha256(anchor / LOCK_NAME),
        "anchor_manifest": anchor_manifest,
        "seed_registry": seed_registry,
        "seed_registry_path": registry_path,
    }


def freeze_formal_anchor() -> dict[str, Path | str]:
    """Atomically claim the global seed bank, then write the canonical anchor."""

    config = formal_config()
    validate_config(config)
    repo_root = Path(__file__).resolve().parents[1]
    canonical = _canonical_formal_paths(repo_root)
    destination = canonical["anchor"]
    bound_output = canonical["output"]
    canonical["root"].parent.mkdir(parents=True, exist_ok=True)
    if bound_output.exists():
        raise FileExistsError(
            f"formal output path must not exist when the lock is frozen: {bound_output}"
        )
    timing_environment = _timing_environment_snapshot(repo_root, enforce_formal=True)
    seed_history_audit = _formal_seed_history_audit(
        repo_root, excluded_directories=(canonical["root"],)
    )
    seed_registry, registry_path = _claim_formal_seed_bank_registry(
        repo_root, seed_history_audit, timing_environment
    )
    post_claim_audit = _formal_seed_history_audit(
        repo_root, excluded_directories=(canonical["root"],)
    )
    if post_claim_audit["inventory_sha256"] != seed_history_audit["inventory_sha256"]:
        raise RuntimeError("freshness inventory changed while claiming the seed bank")
    source = _source_manifest(repo_root)
    figure5 = generate_figure5_dataset(_paper_config(config))
    data = _preflight_data_manifest(config, figure5.clean)
    lock = _protocol_lock(
        config,
        source,
        data,
        formal_anchor_dir=destination,
        formal_output_dir=bound_output,
        seed_history_audit=seed_history_audit,
        seed_registry=seed_registry,
        timing_environment=timing_environment,
    )
    _assert_source_unchanged(source, repo_root)

    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing to overwrite existing anchor path: {destination}"
        ) from error
    paths = {
        "source_manifest": destination / SOURCE_MANIFEST_NAME,
        "data_manifest": destination / DATA_MANIFEST_NAME,
        "protocol_lock": destination / LOCK_NAME,
        "anchor_manifest": destination / ANCHOR_MANIFEST_NAME,
    }
    for key, payload in (
        ("source_manifest", source),
        ("data_manifest", data),
        ("protocol_lock", lock),
    ):
        with paths[key].open("xb") as stream:
            stream.write(_serialized_json_bytes(payload))
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "pre_fit_frozen_figure5_noise_stress_anchor",
        "created_before_any_formal_fit": True,
        "files": {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for key, path in paths.items()
            if key != "anchor_manifest"
        },
    }
    manifest["anchor_manifest_body_sha256"] = canonical_sha256(manifest)
    with paths["anchor_manifest"].open("xb") as stream:
        stream.write(_serialized_json_bytes(manifest))
    validated = validate_formal_anchor_directory(destination)
    return {
        **paths,
        "seed_registry": registry_path,
        "protocol_lock_file_sha256": validated["protocol_lock_file_sha256"],
    }


def _load_formal_anchor_for_run(
    lock_path: str | Path,
    expected_lock_sha256: str,
    formal_output_dir: str | Path,
) -> dict[str, Any]:
    lock_file = Path(lock_path)
    if lock_file.name != LOCK_NAME or lock_file.is_symlink() or not lock_file.is_file():
        raise ValueError("--lock must name the real frozen protocol-lock file")
    if (
        len(expected_lock_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_lock_sha256)
    ):
        raise ValueError("--expected-lock-sha256 must be 64 lowercase hex characters")
    actual_sha256 = _file_sha256(lock_file)
    if actual_sha256 != expected_lock_sha256:
        raise ValueError("protocol-lock file SHA-256 differs from the out-of-band value")
    validated = validate_formal_anchor_directory(lock_file.parent)
    if validated["protocol_lock_file_sha256"] != expected_lock_sha256:
        raise ValueError("validated anchor lock SHA-256 mismatch")
    output = Path(formal_output_dir).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    canonical = _canonical_formal_paths(repo_root)
    if lock_file.resolve() != canonical["anchor"] / LOCK_NAME:
        raise ValueError("formal lock is outside the canonical create-once root")
    if output != canonical["output"]:
        raise ValueError("formal output is outside the canonical create-once root")
    authorized_output = validated["protocol_lock"]["execution_authorization"][
        "exclusive_output_dir_resolved"
    ]
    if str(output) != authorized_output:
        raise ValueError("formal output path differs from the frozen authorization")
    if output.exists():
        raise FileExistsError("formal output path already exists before lock consumption")
    execution_seed_audit = _formal_seed_history_audit(
        repo_root,
        excluded_directories=(canonical["root"],),
    )
    frozen_audit = validated["protocol_lock"]["fresh_seed_history_audit"]
    if execution_seed_audit["inventory_sha256"] != frozen_audit["inventory_sha256"]:
        raise RuntimeError("freshness inventory changed after formal freeze")
    execution_timing = _timing_environment_snapshot(repo_root, enforce_formal=True)
    if execution_timing != validated["protocol_lock"]["timing_environment"]:
        raise RuntimeError("formal timing environment changed after freeze")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "receipt_kind": "formal_lock_consumed_before_any_solver_fit",
        "protocol_lock_file_sha256": expected_lock_sha256,
        "protocol_lock_body_sha256": validated["protocol_lock"][
            "protocol_lock_body_sha256"
        ],
        "source_manifest_body_sha256": validated["source_manifest"][
            "source_manifest_body_sha256"
        ],
        "data_manifest_body_sha256": validated["data_manifest"][
            "data_manifest_body_sha256"
        ],
        "seed_registry_body_sha256": validated["seed_registry"][
            "seed_registry_body_sha256"
        ],
        "seed_registry_file_sha256": _file_sha256(validated["seed_registry_path"]),
        "exclusive_output_dir_resolved": str(output),
        "created_with_exclusive_open_before_any_fit": True,
        "lock_reuse_forbidden": True,
        "pre_execution_fresh_seed_history_audit": execution_seed_audit,
        "execution_timing_environment": execution_timing,
    }
    receipt["execution_receipt_body_sha256"] = canonical_sha256(receipt)
    receipt_path = lock_file.parent / EXECUTION_RECEIPT_NAME
    try:
        with receipt_path.open("xb") as stream:
            stream.write(_serialized_json_bytes(receipt))
    except FileExistsError as error:
        raise FileExistsError("formal lock has already been consumed") from error
    validated["anchor_paths"] = {
        "source_manifest": lock_file.parent / SOURCE_MANIFEST_NAME,
        "data_manifest": lock_file.parent / DATA_MANIFEST_NAME,
        "protocol_lock": lock_file,
    }
    validated["execution_receipt"] = receipt
    validated["execution_receipt_path"] = receipt_path
    validated["bound_output_dir"] = output
    return validated


def _vp_fit_with_factor_step(
    data: FloatArray,
    pca: NativeFactorFit,
    sigma_max: float,
    config: PaperVideoConfig,
) -> NativeFactorFit:
    """Local adapter retaining the final accepted full-data VP step."""

    initial_step = 1.0 if sigma_max == 0.0 else 1.0 / sigma_max**2
    start = perf_counter()
    result = fit_vp_spca(
        data,
        config.n_components,
        alpha=config.figure5_alpha,
        max_iter=config.max_iter,
        tol=config.tol,
        step_size=initial_step,
        initial_B=pca.axes,
        backtracking=True,
        verbose=False,
    )
    runtime = perf_counter() - start
    if not result.step_sizes:
        raise RuntimeError("VP-SPCA returned no accepted step")
    spatial_modes = data @ result.B
    return NativeFactorFit(
        method="VP-SPCA",
        axes=np.asarray(result.A, dtype=np.float64),
        loadings=np.asarray(result.B, dtype=np.float64),
        spatial_modes=np.asarray(spatial_modes, dtype=np.float64),
        temporal_dynamics=np.asarray(result.A, dtype=np.float64),
        reconstruction=np.asarray(spatial_modes @ result.A.T, dtype=np.float64),
        runtime_seconds=float(runtime),
        converged=bool(result.converged),
        n_iter=int(result.n_iter),
        final_stationarity=float(result.stationarity[-1]),
        status=result.status,
        objectives=tuple(float(value) for value in result.objectives),
        extra={
            "alpha": float(result.alpha),
            "step_size_policy": "final_accepted_full_data_solver_step",
            "final_factor_step_size": float(result.step_sizes[-1]),
        },
    )


def _fit_observed_methods(
    observed: FloatArray,
    config: FrozenNoiseStressConfig,
    *,
    noise_seed: int,
) -> tuple[dict[str, NativeFactorFit], tuple[str, ...]]:
    """Fit from observed data only; evaluation truth is not an argument."""

    pca, sigma_max = _pca_fit(observed, method="PCA")
    paper = _paper_config(config)
    sparse_order = (
        ("VP-SPCA", "RND-VP-SPCA")
        if noise_seed % 2
        else ("RND-VP-SPCA", "VP-SPCA")
    )
    fits: dict[str, NativeFactorFit] = {"PCA": pca}
    for method in sparse_order:
        if method == "VP-SPCA":
            fits[method] = _vp_fit_with_factor_step(observed, pca, sigma_max, paper)
        else:
            fits[method] = _randomized_fit(observed, paper)
    return fits, ("PCA", *sparse_order)


def _fit_metric_record(
    *,
    case: NoisyFigure5Case,
    fit: NativeFactorFit,
    method: str,
    truth_modes: FloatArray,
    truth_dynamics: FloatArray,
    time: FloatArray,
    config: FrozenNoiseStressConfig,
    pca_prerequisite_runtime_seconds: float,
    fit_execution_order: Sequence[str],
) -> tuple[dict[str, Any], FloatArray]:
    aligned_modes, aligned_dynamics, permutation, signs = (
        align_factor_pair_sign_permutation_only(
            truth_modes, fit.spatial_modes, fit.temporal_dynamics
        )
    )
    spatial = _paired_cosines(truth_modes, aligned_modes)
    temporal = _paired_cosines(truth_dynamics, aligned_dynamics)
    localization = _localization_metrics(
        truth_modes,
        aligned_modes,
        relative_support_threshold=config.support_threshold,
    )
    frame_errors = normalized_full_frame_errors(case.clean, fit.reconstruction)
    frame_summary = summarize_full_frame_errors(frame_errors, time)
    full_objective = vp_objective(
        case.observed, fit.axes, fit.loadings, config.alpha
    )
    extra = {} if fit.extra is None else dict(fit.extra)
    sketch_residual = (
        float(extra["sketch_residual"])
        if method == "RND-VP-SPCA"
        else None
    )
    sketch_residual_relative = (
        float(extra["sketch_residual_relative"])
        if method == "RND-VP-SPCA"
        else None
    )
    randomized_full_objective = (
        float(extra["full_objective"])
        if method == "RND-VP-SPCA"
        else None
    )
    if randomized_full_objective is not None and not np.isclose(
        randomized_full_objective,
        full_objective,
        rtol=1e-10,
        atol=1e-8 * max(1.0, abs(full_objective)),
    ):
        raise RuntimeError("randomized full objective disagrees with recomputation")
    stationarity_met = (
        True
        if fit.final_stationarity is None
        else bool(fit.final_stationarity <= config.tol)
    )
    full_converged = (
        True
        if method == "PCA"
        else (
            bool(extra["full_converged"])
            if method == "RND-VP-SPCA"
            else bool(fit.converged)
        )
    )
    if method != "PCA" and not (
        bool(fit.converged) == full_converged == stationarity_met
    ):
        raise RuntimeError("sparse solver convergence semantics are inconsistent")
    if method == "PCA":
        largest = float(extra["largest_singular_value"])
        factor_step = 1.0 if largest == 0.0 else 1.0 / largest**2
    elif method == "VP-SPCA":
        factor_step = float(extra["final_factor_step_size"])
    else:
        factor_step = float(extra["full_diagnostic_step_size"])
    if method == "PCA":
        if not (
            fit.status == "closed-form truncated SVD"
            and fit.n_iter == 1
            and fit.converged
        ):
            raise RuntimeError("PCA status/convergence/iteration semantics changed")
    elif method == "VP-SPCA":
        expected_status = "converged" if full_converged else "maximum iterations reached"
        if fit.status != expected_status or not 1 <= fit.n_iter <= config.max_iter:
            raise RuntimeError("VP status/convergence/iteration semantics changed")
        if not full_converged and fit.n_iter != config.max_iter:
            raise RuntimeError("nonconverged VP must exhaust the locked iteration budget")
    else:
        sketch_converged = bool(extra["sketch_converged"])
        expected_status = (
            "sketch solve: "
            + ("converged" if sketch_converged else "maximum iterations reached")
            + "; full stationarity diagnostic: "
            + ("passed" if full_converged else "failed")
        )
        if fit.status != expected_status or not 1 <= fit.n_iter <= config.max_iter:
            raise RuntimeError("randomized status/convergence/iteration semantics changed")
        if not sketch_converged and fit.n_iter != config.max_iter:
            raise RuntimeError(
                "nonconverged randomized sketch must exhaust the locked iteration budget"
            )
    method_token = {"PCA": "pca", "VP-SPCA": "vp", "RND-VP-SPCA": "rnd"}[
        method
    ]
    snr_token = str(int(case.snr_db) if case.snr_db.is_integer() else case.snr_db)
    factor_prefix = f"snr_{snr_token}_seed_{case.noise_seed}_{method_token}"
    raw_runtime = float(fit.runtime_seconds)
    complete_method_runtime = (
        raw_runtime + float(pca_prerequisite_runtime_seconds)
        if method == "VP-SPCA"
        else raw_runtime
    )
    record: dict[str, Any] = {
        "snr_db": case.snr_db,
        "noise_seed": case.noise_seed,
        "method": method,
        "raw_solver_fit_call_runtime_seconds": raw_runtime,
        "complete_method_runtime_seconds": complete_method_runtime,
        "runtime_components": {
            "pca_and_sigma_preparation_seconds": (
                float(pca_prerequisite_runtime_seconds)
                if method == "VP-SPCA"
                else 0.0
            ),
            "solver_fit_call_seconds": raw_runtime,
            "randomized_fit_call_includes_range_compress_optimization_and_diagnostics": (
                method == "RND-VP-SPCA"
            ),
            "used_by_speed_gate": "complete_method_runtime_seconds",
        },
        "fit_execution_order": list(fit_execution_order),
        "fit_execution_position": int(fit_execution_order.index(method)),
        "factor_keys": {
            "A": f"{factor_prefix}__A",
            "B": f"{factor_prefix}__B",
            "step": f"{factor_prefix}__step",
        },
        "factor_step_size": factor_step,
        "converged": bool(fit.converged),
        "full_converged": full_converged,
        "n_iter": int(fit.n_iter),
        "status": fit.status,
        "final_full_stationarity": fit.final_stationarity,
        "tolerance": config.tol,
        "stationarity_tolerance": config.tol,
        "stationarity_met": stationarity_met,
        "final_full_objective_recomputed": float(full_objective),
        "last_solver_objective": (
            None if not fit.objectives else float(fit.objectives[-1])
        ),
        "clean_relative_reconstruction_error": relative_reconstruction_error(
            case.clean, fit.reconstruction
        ),
        "observed_relative_reconstruction_error": relative_reconstruction_error(
            case.observed, fit.reconstruction
        ),
        "full_frame_normalized_rmse": frame_summary,
        "loading_sparsity_at_threshold": float(
            np.mean(np.abs(fit.loadings) <= config.loading_zero_threshold)
        ),
        "loading_zero_threshold": config.loading_zero_threshold,
        "mode_and_dynamics_recovery": {
            "spatial_mode_cosine_per_component": spatial,
            "spatial_mode_mean_cosine": float(np.mean(spatial)),
            "spatial_mode_worst_cosine": float(np.min(spatial)),
            "temporal_dynamics_cosine_per_component": temporal,
            "temporal_dynamics_mean_cosine": float(np.mean(temporal)),
            "temporal_dynamics_worst_cosine": float(np.min(temporal)),
            "component_permutation": list(permutation),
            "component_signs": list(signs),
            "amplitude_refit": False,
        },
        "localization": localization,
        "randomized_diagnostics": {
            "sketch_residual": sketch_residual,
            "sketch_residual_relative": sketch_residual_relative,
            "reported_full_objective": randomized_full_objective,
            "reported_full_stationarity": (
                float(extra["full_stationarity"])
                if method == "RND-VP-SPCA"
                else None
            ),
            "sketch_converged": (
                bool(extra["sketch_converged"])
                if method == "RND-VP-SPCA"
                else None
            ),
            "full_converged": (
                bool(extra["full_converged"])
                if method == "RND-VP-SPCA"
                else None
            ),
            "used_full_gradient": (
                bool(extra["used_full_gradient"])
                if method == "RND-VP-SPCA"
                else None
            ),
            "internal_timing_diagnostics_not_used_for_speed_gate": (
                dict(extra["timings"])
                if method == "RND-VP-SPCA"
                else None
            ),
        },
    }
    _require_exact_keys(record, CASE_RECORD_KEYS, label="case record")
    return record, frame_errors


def _bootstrap_interval(
    values: FloatArray,
    *,
    statistic: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be a finite nonempty vector")
    if statistic not in {"mean", "median"}:
        raise ValueError("unsupported bootstrap statistic")
    observed = float(np.mean(array) if statistic == "mean" else np.median(array))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(draws, array.size))
    samples = array[indices]
    estimates = (
        np.mean(samples, axis=1)
        if statistic == "mean"
        else np.median(samples, axis=1)
    )
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return observed, float(lower), float(upper)


def _gate_metric(
    values: FloatArray,
    *,
    statistic: str,
    draws: int,
    seed: int,
    bound: str,
    operator: str,
    threshold: float,
) -> dict[str, Any]:
    estimate, lower, upper = _bootstrap_interval(
        values, statistic=statistic, draws=draws, seed=seed
    )
    if bound == "upper" and operator == "<=":
        passed = upper <= threshold
        evaluated_bound = upper
    elif bound == "lower" and operator == ">":
        passed = lower > threshold
        evaluated_bound = lower
    else:
        raise ValueError("unsupported gate direction")
    return {
        "paired_values": [float(value) for value in values],
        "statistic": statistic,
        "estimate": estimate,
        "ci95_lower": lower,
        "ci95_upper": upper,
        "evaluated_ci_bound": float(evaluated_bound),
        "required_operator": operator,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def _paired_stratified_gates(
    records: Sequence[Mapping[str, Any]],
    config: FrozenNoiseStressConfig,
) -> dict[str, Any]:
    spec = _gate_specification()
    thresholds = spec["thresholds"]
    strata: dict[str, Any] = {}
    for stratum_index, (snr_db, expected_seeds) in enumerate(config.snr_seed_bank):
        by_pair: dict[int, dict[str, Mapping[str, Any]]] = {}
        for record in records:
            if float(record["snr_db"]) == snr_db:
                by_pair.setdefault(int(record["noise_seed"]), {})[
                    str(record["method"])
                ] = record
        if tuple(sorted(by_pair)) != tuple(sorted(expected_seeds)):
            raise RuntimeError("case records do not match the locked seed bank")
        pairs: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for seed in expected_seeds:
            methods = by_pair[seed]
            if set(methods) != {"PCA", "VP-SPCA", "RND-VP-SPCA"}:
                raise RuntimeError("a paired case is missing a locked method")
            pairs.append((methods["VP-SPCA"], methods["RND-VP-SPCA"]))

        eps = np.finfo(np.float64).eps
        values: dict[str, FloatArray] = {
            "clean_relative_error_ratio": np.asarray(
                [
                    rnd["clean_relative_reconstruction_error"]
                    / max(vp["clean_relative_reconstruction_error"], eps)
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "max_frame_rmse_ratio": np.asarray(
                [
                    rnd["full_frame_normalized_rmse"]["max"]
                    / max(vp["full_frame_normalized_rmse"]["max"], eps)
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "spatial_mean_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"]["spatial_mode_mean_cosine"]
                    - rnd["mode_and_dynamics_recovery"]["spatial_mode_mean_cosine"]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "spatial_worst_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"]["spatial_mode_worst_cosine"]
                    - rnd["mode_and_dynamics_recovery"]["spatial_mode_worst_cosine"]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "temporal_mean_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"][
                        "temporal_dynamics_mean_cosine"
                    ]
                    - rnd["mode_and_dynamics_recovery"][
                        "temporal_dynamics_mean_cosine"
                    ]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "temporal_worst_cosine_loss": np.asarray(
                [
                    vp["mode_and_dynamics_recovery"][
                        "temporal_dynamics_worst_cosine"
                    ]
                    - rnd["mode_and_dynamics_recovery"][
                        "temporal_dynamics_worst_cosine"
                    ]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "off_support_absolute_increment": np.asarray(
                [
                    rnd["localization"]["off_support_energy_fraction_mean"]
                    - vp["localization"]["off_support_energy_fraction_mean"]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "cross_talk_absolute_increment": np.asarray(
                [
                    rnd["localization"]["cross_talk_energy_fraction_mean"]
                    - vp["localization"]["cross_talk_energy_fraction_mean"]
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "loading_sparsity_absolute_difference": np.asarray(
                [
                    abs(
                        rnd["loading_sparsity_at_threshold"]
                        - vp["loading_sparsity_at_threshold"]
                    )
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "relative_full_objective_excess": np.asarray(
                [
                    (
                        rnd["final_full_objective_recomputed"]
                        - vp["final_full_objective_recomputed"]
                    )
                    / max(abs(vp["final_full_objective_recomputed"]), eps)
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
            "complete_method_runtime_speedup": np.asarray(
                [
                    vp["complete_method_runtime_seconds"]
                    / max(rnd["complete_method_runtime_seconds"], eps)
                    for vp, rnd in pairs
                ],
                dtype=np.float64,
            ),
        }
        metric_results: dict[str, Any] = {}
        for metric_index, (name, metric_values) in enumerate(values.items()):
            rule = thresholds[name]
            statistic = (
                "median" if name == "complete_method_runtime_speedup" else "mean"
            )
            metric_results[name] = _gate_metric(
                metric_values,
                statistic=statistic,
                draws=config.bootstrap_draws,
                seed=config.bootstrap_seed + 100 * stratum_index + metric_index,
                bound=rule["ci_bound"],
                operator=rule["operator"],
                threshold=float(rule["value"]),
            )
        solver_converged_all = all(
            bool(vp["converged"])
            and bool(vp["stationarity_met"])
            and bool(rnd["converged"])
            and bool(rnd["stationarity_met"])
            for vp, rnd in pairs
        )
        eligible = bool(
            config.profile == "formal_confirmation" and len(pairs) == 20
        )
        thresholds_passed = bool(
            all(metric["passed"] for metric in metric_results.values())
        )
        strict_qualified = bool(
            eligible and thresholds_passed and solver_converged_all
        )
        if not eligible:
            claim_status = "smoke_not_evidence"
        elif not thresholds_passed:
            claim_status = "fixed_budget_gate_failed"
        elif not solver_converged_all:
            claim_status = "fixed_budget_quality_matched_nonconverged"
        else:
            claim_status = "strict_converged_quality_matched_and_speedup"
        strata[str(int(snr_db) if snr_db.is_integer() else snr_db)] = {
            "snr_db": snr_db,
            "n_paired_seeds": len(pairs),
            "formal_gate_eligible": eligible,
            "metrics": metric_results,
            "all_predeclared_thresholds_passed": thresholds_passed,
            "all_vp_and_randomized_solvers_converged": solver_converged_all,
            "strict_claim_qualified": strict_qualified,
            "claim_status": claim_status,
        }
    overall_strict = bool(
        config.profile == "formal_confirmation"
        and all(stratum["strict_claim_qualified"] for stratum in strata.values())
    )
    return {
        "gate_specification": spec,
        "profile": config.profile,
        "strata": strata,
        "pooling_across_snr_for_gate_forbidden": True,
        "overall_strict_claim_qualified": overall_strict,
        "overall_claim_status": (
            "strict_all_snr"
            if overall_strict
            else (
                "smoke_not_evidence"
                if config.profile != "formal_confirmation"
                else "report_each_failed_snr_boundary"
            )
        ),
    }


def _scalar_method_summaries(
    records: Sequence[Mapping[str, Any]],
    config: FrozenNoiseStressConfig,
) -> dict[str, Any]:
    extractors = {
        "clean_relative_reconstruction_error": lambda row: row[
            "clean_relative_reconstruction_error"
        ],
        "max_frame_normalized_rmse": lambda row: row[
            "full_frame_normalized_rmse"
        ]["max"],
        "spatial_mode_mean_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["spatial_mode_mean_cosine"],
        "spatial_mode_worst_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["spatial_mode_worst_cosine"],
        "temporal_dynamics_mean_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["temporal_dynamics_mean_cosine"],
        "temporal_dynamics_worst_cosine": lambda row: row[
            "mode_and_dynamics_recovery"
        ]["temporal_dynamics_worst_cosine"],
        "off_support_energy_fraction_mean": lambda row: row["localization"][
            "off_support_energy_fraction_mean"
        ],
        "cross_talk_energy_fraction_mean": lambda row: row["localization"][
            "cross_talk_energy_fraction_mean"
        ],
        "loading_sparsity_at_threshold": lambda row: row[
            "loading_sparsity_at_threshold"
        ],
        "complete_method_runtime_seconds": lambda row: row[
            "complete_method_runtime_seconds"
        ],
        "final_full_objective_recomputed": lambda row: row[
            "final_full_objective_recomputed"
        ],
    }
    output: dict[str, Any] = {}
    for snr_db, _ in config.snr_seed_bank:
        snr_key = str(int(snr_db) if snr_db.is_integer() else snr_db)
        output[snr_key] = {}
        for method in ("PCA", "VP-SPCA", "RND-VP-SPCA"):
            selected = [
                row
                for row in records
                if float(row["snr_db"]) == snr_db and row["method"] == method
            ]
            method_summary: dict[str, Any] = {
                "n_cases": len(selected),
                "converged_fraction": float(
                    np.mean([bool(row["converged"]) for row in selected])
                ),
            }
            for name, extractor in extractors.items():
                values = np.asarray([extractor(row) for row in selected], dtype=float)
                method_summary[name] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "p05": float(np.percentile(values, 5.0)),
                    "p95": float(np.percentile(values, 95.0)),
                }
            output[snr_key][method] = method_summary
    return output


def run_frozen_noise_stress(
    profile: str,
    *,
    output_dir: str | Path | None = None,
    lock_path: str | Path | None = None,
    expected_lock_sha256: str | None = None,
) -> FrozenNoiseStressRun:
    """Run smoke directly or consume an exclusive out-of-band formal lock."""

    repo_root = Path(__file__).resolve().parents[1]
    if profile == "formal":
        config = formal_config()
        if output_dir is None or lock_path is None or expected_lock_sha256 is None:
            raise ValueError(
                "formal execution requires output_dir, lock_path, and "
                "expected_lock_sha256"
            )
        anchored = _load_formal_anchor_for_run(
            lock_path, expected_lock_sha256, output_dir
        )
        source_manifest = anchored["source_manifest"]
        data_manifest = anchored["data_manifest"]
        protocol_lock = anchored["protocol_lock"]
        protocol_lock_file_sha256 = anchored["protocol_lock_file_sha256"]
        anchor_paths: Mapping[str, Path] | None = anchored["anchor_paths"]
        execution_receipt: Mapping[str, Any] | None = anchored[
            "execution_receipt"
        ]
        execution_receipt_path: Path | None = anchored[
            "execution_receipt_path"
        ]
        seed_registry: Mapping[str, Any] | None = anchored["seed_registry"]
        seed_registry_path: Path | None = anchored["seed_registry_path"]
        bound_output_dir: Path | None = anchored["bound_output_dir"]
    elif profile == "smoke":
        config = smoke_config()
        if lock_path is not None or expected_lock_sha256 is not None:
            raise ValueError("smoke cannot consume a formal lock")
        source_manifest = _source_manifest(repo_root)
        lock_figure5 = generate_figure5_dataset(_paper_config(config))
        data_manifest = _preflight_data_manifest(config, lock_figure5.clean)
        timing_environment = _timing_environment_snapshot(
            repo_root, enforce_formal=False
        )
        protocol_lock = _protocol_lock(
            config,
            source_manifest,
            data_manifest,
            timing_environment=timing_environment,
        )
        protocol_lock_file_sha256 = sha256(
            _serialized_json_bytes(protocol_lock)
        ).hexdigest()
        anchor_paths = None
        seed_registry = {
            "schema_version": 1,
            "registry_kind": "smoke_not_global_seed_bank_claim",
            "formal_snr_seed_bank": None,
            "created_with_exclusive_open_before_anchor_and_any_fit": False,
            "reuse_or_reissue_forbidden": False,
        }
        seed_registry["seed_registry_body_sha256"] = canonical_sha256(seed_registry)
        seed_registry_path = None
        execution_receipt = {
            "schema_version": 1,
            "receipt_kind": "smoke_not_formal_lock_consumption",
            "protocol_lock_file_sha256": protocol_lock_file_sha256,
            "protocol_lock_body_sha256": protocol_lock[
                "protocol_lock_body_sha256"
            ],
            "exclusive_output_dir_resolved": (
                None if output_dir is None else str(Path(output_dir).resolve())
            ),
            "created_with_exclusive_open_before_any_fit": False,
            "lock_reuse_forbidden": False,
            "execution_timing_environment": timing_environment,
        }
        execution_receipt["execution_receipt_body_sha256"] = canonical_sha256(
            execution_receipt
        )
        execution_receipt_path = None
        bound_output_dir = None if output_dir is None else Path(output_dir).resolve()
    else:
        raise ValueError("profile must be 'formal' or 'smoke'")
    validate_config(config)
    figure5 = generate_figure5_dataset(_paper_config(config))
    if array_sha256(figure5.clean) != data_manifest["clean_array_sha256"]:
        raise RuntimeError("current Figure 5 clean data differs from the pre-fit lock")
    expected_data = {
        (float(record["snr_db"]), int(record["noise_seed"])): record
        for record in data_manifest["cases"]
    }

    case_records: list[Mapping[str, Any]] = []
    frame_records: list[Mapping[str, Any]] = []
    factor_arrays: dict[str, FloatArray] = {}
    for snr_db, seeds in config.snr_seed_bank:
        for noise_seed in seeds:
            case = generate_noisy_figure5_case(
                figure5.clean, snr_db=snr_db, noise_seed=noise_seed
            )
            actual_data = _data_record(case)
            expected = expected_data[(snr_db, noise_seed)]
            if canonical_sha256(actual_data) != canonical_sha256(expected):
                raise RuntimeError("regenerated case does not match the pre-fit data lock")
            fits, execution_order = _fit_observed_methods(
                case.observed, config, noise_seed=noise_seed
            )
            pca_prerequisite_runtime = float(fits["PCA"].runtime_seconds)
            for method in ("PCA", "VP-SPCA", "RND-VP-SPCA"):
                record, errors = _fit_metric_record(
                    case=case,
                    fit=fits[method],
                    method=method,
                    truth_modes=figure5.spatial_modes,
                    truth_dynamics=figure5.temporal_scores,
                    time=figure5.time,
                    config=config,
                    pca_prerequisite_runtime_seconds=pca_prerequisite_runtime,
                    fit_execution_order=execution_order,
                )
                case_records.append(record)
                factor_keys = record["factor_keys"]
                factor_payloads = {
                    factor_keys["A"]: np.ascontiguousarray(
                        fits[method].axes, dtype=np.float64
                    ),
                    factor_keys["B"]: np.ascontiguousarray(
                        fits[method].loadings, dtype=np.float64
                    ),
                    factor_keys["step"]: np.asarray(
                        [record["factor_step_size"]], dtype=np.float64
                    ),
                }
                if set(factor_payloads) & set(factor_arrays):
                    raise RuntimeError("duplicate factor NPZ key")
                factor_arrays.update(factor_payloads)
                for frame_index, (timestamp, error) in enumerate(
                    zip(figure5.time, errors)
                ):
                    frame_records.append(
                        {
                            "snr_db": snr_db,
                            "noise_seed": noise_seed,
                            "method": method,
                            "frame_index": frame_index,
                            "time": float(timestamp),
                            "normalized_rmse": float(error),
                        }
                    )
    gates = _paired_stratified_gates(case_records, config)
    _assert_source_unchanged(source_manifest, repo_root)
    if anchor_paths is not None:
        anchored_payloads = {
            "source_manifest": source_manifest,
            "data_manifest": data_manifest,
            "protocol_lock": protocol_lock,
        }
        for key, payload in anchored_payloads.items():
            expected_sha256 = sha256(_serialized_json_bytes(payload)).hexdigest()
            if _file_sha256(anchor_paths[key]) != expected_sha256:
                raise RuntimeError(f"formal anchor changed during execution: {key}")
        if execution_receipt_path is None or execution_receipt is None:
            raise RuntimeError("formal execution receipt disappeared")
        if _file_sha256(execution_receipt_path) != sha256(
            _serialized_json_bytes(execution_receipt)
        ).hexdigest():
            raise RuntimeError("formal execution receipt changed during execution")
        if seed_registry_path is None or seed_registry is None:
            raise RuntimeError("formal seed registry disappeared during execution")
        if _file_sha256(seed_registry_path) != sha256(
            _serialized_json_bytes(seed_registry)
        ).hexdigest():
            raise RuntimeError("formal seed registry changed during execution")
    return FrozenNoiseStressRun(
        config=config,
        source_manifest=source_manifest,
        data_manifest=data_manifest,
        protocol_lock=protocol_lock,
        case_records=tuple(case_records),
        frame_records=tuple(frame_records),
        gates=gates,
        protocol_lock_file_sha256=protocol_lock_file_sha256,
        factor_arrays=factor_arrays,
        anchor_paths=anchor_paths,
        execution_receipt=execution_receipt,
        execution_receipt_path=execution_receipt_path,
        seed_registry=seed_registry,
        seed_registry_path=seed_registry_path,
        bound_output_dir=bound_output_dir,
    )


def _write_json_exclusive(payload: Mapping[str, Any], path: Path) -> None:
    with path.open("xb") as stream:
        stream.write(_serialized_json_bytes(payload))


def _write_cases_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "snr_db",
        "noise_seed",
        "method",
        "converged",
        "full_converged",
        "n_iter",
        "status",
        "final_full_stationarity",
        "tolerance",
        "stationarity_tolerance",
        "stationarity_met",
        "factor_step_size",
        "factor_A_key",
        "factor_B_key",
        "factor_step_key",
        "final_full_objective_recomputed",
        "last_solver_objective",
        "raw_solver_fit_call_runtime_seconds",
        "complete_method_runtime_seconds",
        "clean_relative_reconstruction_error",
        "observed_relative_reconstruction_error",
        "frame_rmse_median",
        "frame_rmse_p95",
        "frame_rmse_max",
        "worst_frame_index",
        "worst_frame_time",
        "loading_sparsity_at_threshold",
        "spatial_mode_mean_cosine",
        "spatial_mode_worst_cosine",
        "temporal_dynamics_mean_cosine",
        "temporal_dynamics_worst_cosine",
        "off_support_energy_fraction_mean",
        "cross_talk_energy_fraction_mean",
        "sketch_residual",
        "sketch_residual_relative",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            frame = record["full_frame_normalized_rmse"]
            recovery = record["mode_and_dynamics_recovery"]
            localization = record["localization"]
            randomized = record["randomized_diagnostics"]
            writer.writerow(
                {
                    "snr_db": f"{float(record['snr_db']):.17g}",
                    "noise_seed": int(record["noise_seed"]),
                    "method": record["method"],
                    "converged": str(bool(record["converged"])).lower(),
                    "full_converged": str(bool(record["full_converged"])).lower(),
                    "n_iter": int(record["n_iter"]),
                    "status": record["status"],
                    "final_full_stationarity": (
                        ""
                        if record["final_full_stationarity"] is None
                        else f"{float(record['final_full_stationarity']):.17g}"
                    ),
                    "tolerance": f"{float(record['tolerance']):.17g}",
                    "stationarity_tolerance": f"{float(record['stationarity_tolerance']):.17g}",
                    "stationarity_met": str(bool(record["stationarity_met"])).lower(),
                    "factor_step_size": f"{float(record['factor_step_size']):.17g}",
                    "factor_A_key": record["factor_keys"]["A"],
                    "factor_B_key": record["factor_keys"]["B"],
                    "factor_step_key": record["factor_keys"]["step"],
                    "final_full_objective_recomputed": f"{float(record['final_full_objective_recomputed']):.17g}",
                    "last_solver_objective": (
                        ""
                        if record["last_solver_objective"] is None
                        else f"{float(record['last_solver_objective']):.17g}"
                    ),
                    "raw_solver_fit_call_runtime_seconds": f"{float(record['raw_solver_fit_call_runtime_seconds']):.17g}",
                    "complete_method_runtime_seconds": f"{float(record['complete_method_runtime_seconds']):.17g}",
                    "clean_relative_reconstruction_error": f"{float(record['clean_relative_reconstruction_error']):.17g}",
                    "observed_relative_reconstruction_error": f"{float(record['observed_relative_reconstruction_error']):.17g}",
                    "frame_rmse_median": f"{float(frame['median']):.17g}",
                    "frame_rmse_p95": f"{float(frame['p95']):.17g}",
                    "frame_rmse_max": f"{float(frame['max']):.17g}",
                    "worst_frame_index": int(frame["worst_frame_index"]),
                    "worst_frame_time": f"{float(frame['worst_frame_time']):.17g}",
                    "loading_sparsity_at_threshold": f"{float(record['loading_sparsity_at_threshold']):.17g}",
                    "spatial_mode_mean_cosine": f"{float(recovery['spatial_mode_mean_cosine']):.17g}",
                    "spatial_mode_worst_cosine": f"{float(recovery['spatial_mode_worst_cosine']):.17g}",
                    "temporal_dynamics_mean_cosine": f"{float(recovery['temporal_dynamics_mean_cosine']):.17g}",
                    "temporal_dynamics_worst_cosine": f"{float(recovery['temporal_dynamics_worst_cosine']):.17g}",
                    "off_support_energy_fraction_mean": f"{float(localization['off_support_energy_fraction_mean']):.17g}",
                    "cross_talk_energy_fraction_mean": f"{float(localization['cross_talk_energy_fraction_mean']):.17g}",
                    "sketch_residual": (
                        ""
                        if randomized["sketch_residual"] is None
                        else f"{float(randomized['sketch_residual']):.17g}"
                    ),
                    "sketch_residual_relative": (
                        ""
                        if randomized["sketch_residual_relative"] is None
                        else f"{float(randomized['sketch_residual_relative']):.17g}"
                    ),
                }
            )


def _write_frames_csv(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    fields = (
        "snr_db",
        "noise_seed",
        "method",
        "frame_index",
        "time",
        "normalized_rmse",
    )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "snr_db": f"{float(record['snr_db']):.17g}",
                    "noise_seed": int(record["noise_seed"]),
                    "method": record["method"],
                    "frame_index": int(record["frame_index"]),
                    "time": f"{float(record['time']):.17g}",
                    "normalized_rmse": f"{float(record['normalized_rmse']):.17g}",
                }
            )


def _write_summary_figure(
    records: Sequence[Mapping[str, Any]],
    config: FrozenNoiseStressConfig,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = ("PCA", "VP-SPCA", "RND-VP-SPCA")
    colors = {"PCA": "#6b7280", "VP-SPCA": "#2563eb", "RND-VP-SPCA": "#15803d"}
    extractors = (
        (
            "Clean relative error ↓",
            lambda row: row["clean_relative_reconstruction_error"],
            False,
        ),
        (
            "Worst-frame normalized RMSE ↓",
            lambda row: row["full_frame_normalized_rmse"]["max"],
            False,
        ),
        (
            "Worst spatial-mode cosine ↑",
            lambda row: row["mode_and_dynamics_recovery"][
                "spatial_mode_worst_cosine"
            ],
            False,
        ),
        (
            "Worst temporal-dynamics cosine ↑",
            lambda row: row["mode_and_dynamics_recovery"][
                "temporal_dynamics_worst_cosine"
            ],
            False,
        ),
        (
            "Mean off-support energy ↓",
            lambda row: row["localization"]["off_support_energy_fraction_mean"],
            False,
        ),
        (
            "Mean cross-talk energy ↓",
            lambda row: row["localization"]["cross_talk_energy_fraction_mean"],
            False,
        ),
        (
            f"Loading sparsity (|B|≤{config.loading_zero_threshold:g})",
            lambda row: row["loading_sparsity_at_threshold"],
            False,
        ),
        (
            "Complete method runtime (s, log) ↓",
            lambda row: row["complete_method_runtime_seconds"],
            True,
        ),
    )
    snrs = [snr for snr, _ in config.snr_seed_bank]
    x = np.arange(len(snrs), dtype=float)
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    ):
        figure, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
        for axis, (title, extractor, log_scale) in zip(axes.ravel(), extractors):
            for method_index, method in enumerate(methods):
                means: list[float] = []
                lower: list[float] = []
                upper: list[float] = []
                offset = (method_index - 1) * 0.08
                for snr_index, snr_db in enumerate(snrs):
                    values = np.asarray(
                        [
                            extractor(row)
                            for row in records
                            if row["method"] == method
                            and float(row["snr_db"]) == snr_db
                        ],
                        dtype=float,
                    )
                    means.append(float(np.mean(values)))
                    lower.append(float(np.percentile(values, 5.0)))
                    upper.append(float(np.percentile(values, 95.0)))
                    axis.scatter(
                        np.full(values.size, x[snr_index] + offset),
                        values,
                        s=15,
                        alpha=0.28,
                        color=colors[method],
                    )
                mean_array = np.asarray(means)
                axis.errorbar(
                    x + offset,
                    mean_array,
                    yerr=np.vstack(
                        [mean_array - np.asarray(lower), np.asarray(upper) - mean_array]
                    ),
                    marker="o",
                    linewidth=1.6,
                    capsize=3,
                    color=colors[method],
                    label=method,
                )
            axis.set_title(title)
            axis.set_xticks(x, [f"{snr:g}" for snr in snrs])
            axis.set_xlabel("SNR (dB)")
            if log_scale:
                axis.set_yscale("log")
        axes[0, 0].legend(loc="best", fontsize=8)
        convergence = all(
            bool(row["converged"])
            for row in records
            if row["method"] in {"VP-SPCA", "RND-VP-SPCA"}
        )
        qualification = (
            "converged strict evaluation"
            if convergence
            else "nonconverged solver present: fixed-budget evidence only"
        )
        figure.suptitle(
            "Original Figure 5 under locked dense full-rank noise\n"
            f"alpha=9, rank=3, no truth tuning; {qualification}",
            fontsize=14,
        )
        figure.savefig(
            path,
            dpi=config.dpi,
            metadata={"Software": "vp-rspca frozen Figure 5 stress"},
        )
        plt.close(figure)


def _report_payload(run: FrozenNoiseStressRun) -> dict[str, Any]:
    summaries = _scalar_method_summaries(run.case_records, run.config)
    all_sparse_converged = all(
        bool(row["converged"]) and bool(row["stationarity_met"])
        for row in run.case_records
        if row["method"] in {"VP-SPCA", "RND-VP-SPCA"}
    )
    if run.config.profile != "formal_confirmation":
        claim_status = "smoke_not_evidence"
    elif run.gates["overall_strict_claim_qualified"]:
        claim_status = "strict_converged_quality_matched_and_speedup_all_snr"
    elif not all_sparse_converged:
        claim_status = "fixed_budget_only_due_to_solver_nonconvergence"
    else:
        claim_status = "predeclared_gate_failed_report_stratum_boundaries"
    body: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "profile": run.config.profile,
        "paper_scale": bool(
            run.config.grid_size == 200 and run.config.n_timesteps == 300
        ),
        "formal_confirmation": run.config.profile == "formal_confirmation",
        "config": asdict(run.config),
        "protocol_lock_file_sha256": run.protocol_lock_file_sha256,
        "protocol_lock_body_sha256": run.protocol_lock[
            "protocol_lock_body_sha256"
        ],
        "source_manifest_body_sha256": run.source_manifest[
            "source_manifest_body_sha256"
        ],
        "data_manifest_body_sha256": run.data_manifest[
            "data_manifest_body_sha256"
        ],
        "seed_registry_body_sha256": (
            None
            if run.seed_registry is None
            else run.seed_registry["seed_registry_body_sha256"]
        ),
        "protocol": run.protocol_lock["protocol"],
        "execution_authorization": run.protocol_lock["execution_authorization"],
        "execution_receipt_body_sha256": (
            None
            if run.execution_receipt is None
            else run.execution_receipt["execution_receipt_body_sha256"]
        ),
        "case_records": list(run.case_records),
        "method_summaries_by_snr": summaries,
        "stratified_randomized_vs_vp_gates": run.gates,
        "strict_qualification": {
            "all_vp_and_randomized_solvers_converged": all_sparse_converged,
            "overall_strict_claim_qualified": run.gates[
                "overall_strict_claim_qualified"
            ],
            "claim_status": claim_status,
            "failure_policy": (
                "any solver nonconvergence downgrades every quality statement to "
                "fixed-budget; failed gates are reported per SNR without relaxation"
            ),
        },
        "artifacts": {
            "cases_csv": CASES_NAME,
            "full_frame_errors_csv": FRAMES_NAME,
            "summary_figure": FIGURE_NAME,
            "protocol_lock": LOCK_NAME,
            "source_manifest": SOURCE_MANIFEST_NAME,
            "data_manifest": DATA_MANIFEST_NAME,
            "execution_started_receipt": EXECUTION_RECEIPT_NAME,
            "seed_bank_registry": SEED_REGISTRY_NAME,
            "factors_npz": FACTORS_NAME,
        },
    }
    body["report_body_sha256"] = canonical_sha256(body)
    return body


def save_frozen_noise_stress(
    run: FrozenNoiseStressRun,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write a new exact-closure output; an existing path is never replaced."""

    destination = Path(output_dir).resolve()
    if run.bound_output_dir is not None and destination != run.bound_output_dir:
        raise ValueError("save output path differs from the run's bound output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing path: {destination}") from error
    paths = {
        "report": destination / REPORT_NAME,
        "cases": destination / CASES_NAME,
        "frames": destination / FRAMES_NAME,
        "figure": destination / FIGURE_NAME,
        "lock": destination / LOCK_NAME,
        "source_manifest": destination / SOURCE_MANIFEST_NAME,
        "data_manifest": destination / DATA_MANIFEST_NAME,
        "execution_receipt": destination / EXECUTION_RECEIPT_NAME,
        "seed_registry": destination / SEED_REGISTRY_NAME,
        "factors": destination / FACTORS_NAME,
        "output_manifest": destination / OUTPUT_MANIFEST_NAME,
    }
    if run.anchor_paths is None:
        _write_json_exclusive(run.source_manifest, paths["source_manifest"])
        _write_json_exclusive(run.data_manifest, paths["data_manifest"])
        _write_json_exclusive(run.protocol_lock, paths["lock"])
    else:
        for key in ("source_manifest", "data_manifest", "protocol_lock"):
            output_key = "lock" if key == "protocol_lock" else key
            with paths[output_key].open("xb") as stream:
                stream.write(run.anchor_paths[key].read_bytes())
    if run.execution_receipt is None:
        raise RuntimeError("run lacks its pre-fit execution receipt")
    if run.execution_receipt_path is None:
        _write_json_exclusive(run.execution_receipt, paths["execution_receipt"])
    else:
        with paths["execution_receipt"].open("xb") as stream:
            stream.write(run.execution_receipt_path.read_bytes())
    if run.seed_registry is None:
        raise RuntimeError("run lacks its seed-registry disclosure")
    if run.seed_registry_path is None:
        _write_json_exclusive(run.seed_registry, paths["seed_registry"])
    else:
        with paths["seed_registry"].open("xb") as stream:
            stream.write(run.seed_registry_path.read_bytes())
    with paths["factors"].open("xb") as stream:
        np.savez_compressed(
            stream,
            **{
                key: np.asarray(value, dtype=np.float64)
                for key, value in sorted(run.factor_arrays.items())
            },
        )
    _write_cases_csv(run.case_records, paths["cases"])
    _write_frames_csv(run.frame_records, paths["frames"])
    _write_summary_figure(run.case_records, run.config, paths["figure"])
    _write_json_exclusive(_report_payload(run), paths["report"])
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "frozen_figure5_noise_stress_exact_output_closure",
        "self_excluding": True,
        "files": {
            path.name: {
                "sha256": _file_sha256(path),
                "size_bytes": int(path.stat().st_size),
            }
            for key, path in paths.items()
            if key != "output_manifest"
        },
    }
    manifest["output_manifest_body_sha256"] = canonical_sha256(manifest)
    _write_json_exclusive(manifest, paths["output_manifest"])
    return paths


def validate_output_directory(output_dir: str | Path) -> dict[str, Any]:
    """Internal closure check; the external tool adds semantic anchoring."""

    destination = Path(output_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError("output path must be a real directory, not a symlink")
    entries = list(destination.iterdir())
    for entry in entries:
        metadata = entry.lstat()
        if entry.is_symlink() or not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"output contains a non-regular entry: {entry.name}")
    if {entry.name for entry in entries} != _OUTPUT_NAMES:
        raise ValueError("output directory does not have exact frozen closure")
    manifest = _strict_json_load(destination / OUTPUT_MANIFEST_NAME)
    manifest_body = dict(manifest)
    digest = manifest_body.pop("output_manifest_body_sha256")
    if digest != canonical_sha256(manifest_body):
        raise ValueError("output manifest canonical digest mismatch")
    if set(manifest["files"]) != _PAYLOAD_NAMES:
        raise ValueError("output manifest payload set mismatch")
    for name, record in manifest["files"].items():
        path = destination / name
        if _file_sha256(path) != record["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"size mismatch for {name}")
    for filename, digest_field in (
        (REPORT_NAME, "report_body_sha256"),
        (LOCK_NAME, "protocol_lock_body_sha256"),
        (SOURCE_MANIFEST_NAME, "source_manifest_body_sha256"),
        (DATA_MANIFEST_NAME, "data_manifest_body_sha256"),
        (EXECUTION_RECEIPT_NAME, "execution_receipt_body_sha256"),
        (SEED_REGISTRY_NAME, "seed_registry_body_sha256"),
    ):
        payload = _strict_json_load(destination / filename)
        body = dict(payload)
        recorded = body.pop(digest_field)
        if recorded != canonical_sha256(body):
            raise ValueError(f"canonical digest mismatch for {filename}")
    report = _strict_json_load(destination / REPORT_NAME)
    expected_factor_keys = {
        key
        for row in report["case_records"]
        for key in row["factor_keys"].values()
    }
    with np.load(destination / FACTORS_NAME, allow_pickle=False) as factors:
        if set(factors.files) != expected_factor_keys:
            raise ValueError("factor NPZ exact key closure mismatch")
        for row in report["case_records"]:
            keys = row["factor_keys"]
            shape = (int(report["config"]["n_timesteps"]), 3)
            if factors[keys["A"]].shape != shape or factors[keys["B"]].shape != shape:
                raise ValueError("factor NPZ A/B shape mismatch")
            if factors[keys["step"]].shape != (1,):
                raise ValueError("factor NPZ step shape mismatch")
            if not all(
                np.isfinite(factors[key]).all() for key in keys.values()
            ):
                raise ValueError("factor NPZ contains a non-finite value")
        expected_nbytes = len(report["case_records"]) * (
            2 * int(report["config"]["n_timesteps"]) * 3 + 1
        ) * np.dtype("float64").itemsize
        if sum(factors[key].nbytes for key in factors.files) != expected_nbytes:
            raise ValueError("factor NPZ uncompressed payload size mismatch")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser(
        "freeze", help="write the formal pre-fit anchor; do not run a solver"
    )
    run = commands.add_parser("run", help="run smoke or consume a formal anchor")
    run.add_argument("--profile", choices=("smoke", "formal"), required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--lock", type=Path, default=None)
    run.add_argument("--expected-lock-sha256", default=None)
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    if arguments.command == "freeze":
        paths = freeze_formal_anchor()
        print(
            json.dumps(
                {name: str(value) for name, value in paths.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return
    completed_run = run_frozen_noise_stress(
        arguments.profile,
        output_dir=arguments.output_dir,
        lock_path=arguments.lock,
        expected_lock_sha256=arguments.expected_lock_sha256,
    )
    paths = save_frozen_noise_stress(completed_run, arguments.output_dir)
    validate_output_directory(arguments.output_dir)
    print(
        json.dumps(
            {
                "profile": completed_run.config.profile,
                "formal_started": arguments.profile == "formal",
                "output_dir": str(arguments.output_dir),
                "files": {name: str(path) for name, path in paths.items()},
                "external_validation_command": (
                    "python tools/validate_video_figure5_frozen_noise_stress.py "
                    f"--output-dir {arguments.output_dir}"
                    + (
                        ""
                        if arguments.profile == "smoke"
                        else (
                            f" --expected-lock-sha256 {completed_run.protocol_lock_file_sha256}"
                            f" --expected-output-manifest-sha256 "
                            f"{_file_sha256(paths['output_manifest'])}"
                        )
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

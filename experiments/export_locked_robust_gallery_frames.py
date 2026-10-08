"""Export exact sealed Robust VP-SPCA frames for the image-first gallery.

This runner is intentionally narrow: it replays only the already-selected
``pca_perturb_2`` continuation candidate for formal confirmation seed 51001.
It verifies the full clean/observed/corruption inputs and the full A/B/S,
reconstruction, and raw-outlier outputs against the sealed confirmation
report before writing anything.  Only the two predeclared display frames are
retained in the NPZ, which keeps the transfer small without weakening the
full-array replay receipt.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import platform
import stat
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.video_spca_paper_reproduction import generate_figure6_dataset
from experiments.video_spca_paper_robust_study import (
    RobustStudyConfig,
    _candidate_raw_arrays,
    _fit_candidate,
    _paper_video_config,
    build_initializations,
    default_start_specs,
    prepare_observed_case,
    validate_confirmation_output_directory,
)
from rspca.datasets import array_sha256, canonical_sha256, runtime_versions


NPZ_NAME = "locked_robust_gallery_frames.npz"
REPORT_NAME = "locked_robust_gallery_export_report.json"
MANIFEST_NAME = "locked_robust_gallery_export_manifest.json"
_PAYLOAD_NAMES = frozenset({NPZ_NAME, REPORT_NAME})
_OUTPUT_NAMES = _PAYLOAD_NAMES | {MANIFEST_NAME}

SEED = 51001
START_ID = "pca_perturb_2"
ARM_ID = "B_only_descending_kappa_multistart"
ALPHA = 0.0015
TARGET_KAPPA = 0.001
KAPPA_SCHEDULE = (0.004, 0.001)
STAGE_BUDGETS = (40, 160)
FRAME_INDICES = (100, 171)
GRID_SIZE = 200
N_TIMESTEPS = 300
TOTAL_TIME = 150.0


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_expectation(formal_dir: Path) -> dict[str, Any]:
    validate_confirmation_output_directory(formal_dir)
    report_path = formal_dir / "confirmation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    matches = [case for case in report["case_records"] if case["seed"] == SEED]
    if len(matches) != 1:
        raise RuntimeError("sealed confirmation seed is absent or duplicated")
    case = matches[0]
    arm = case["arms"][ARM_ID]
    selected = arm["candidate_records"][int(arm["selected_index"])]
    method = case["methods"]["continuation_multistart"]
    if arm["selected_start_id"] != START_ID or selected["start_id"] != START_ID:
        raise RuntimeError("sealed report does not select the locked start")
    if (
        float(selected["alpha"]) != ALPHA
        or float(selected["target_kappa"]) != TARGET_KAPPA
        or tuple(selected["kappa_schedule"]) != KAPPA_SCHEDULE
        or int(selected["allocated_iterations"]) != sum(STAGE_BUDGETS)
    ):
        raise RuntimeError("sealed selected candidate changed locked parameters")
    worst = method["truth_metrics_after_fit_selection_lock"][
        "full_frame_normalized_rmse"
    ]["worst_frame_index"]
    if int(worst) != FRAME_INDICES[1]:
        raise RuntimeError("sealed worst-frame index changed")
    return {
        "formal_report_sha256": _file_sha256(report_path),
        "formal_manifest_sha256": _file_sha256(
            formal_dir / "confirmation_output_manifest.json"
        ),
        "data": dict(case["data"]),
        "selected_start_id": arm["selected_start_id"],
        "selected_index": int(arm["selected_index"]),
        "selected_fit_sha256": arm["selected_fit_sha256"],
        "candidate": dict(selected),
        "method": dict(method),
    }


def _study_config() -> RobustStudyConfig:
    base = RobustStudyConfig()
    return replace(
        base,
        grid_size=GRID_SIZE,
        n_timesteps=N_TIMESTEPS,
        total_time=TOTAL_TIME,
        max_iter_per_start=sum(STAGE_BUDGETS),
        continuation_kappa_multipliers=tuple(
            value / TARGET_KAPPA for value in KAPPA_SCHEDULE
        ),
        continuation_stage_max_iters=STAGE_BUDGETS,
        representative_frame_index=FRAME_INDICES[0],
        dpi=200,
    )


def _source_provenance(source_root: Path) -> dict[str, Any]:
    roles = {
        "experiments/video_spca_paper_reproduction.py": "robust data generator",
        "experiments/video_spca_paper_robust_study.py": "sealed study helpers",
        "src/rspca/datasets.py": "hashing and dataset helpers",
        "src/rspca/robust.py": "current Robust VP-SPCA solver",
        "src/rspca/vp.py": "VP-SPCA core",
    }
    files: dict[str, Any] = {}
    for relative, role in roles.items():
        path = source_root / relative
        if not path.is_file():
            raise RuntimeError(f"required sealed source is missing: {relative}")
        files[relative] = {"sha256": _file_sha256(path), "role": role}
    return {
        "source_root": str(source_root.resolve()),
        "files": files,
        "runtime_versions": runtime_versions(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _verify_exact_replay(
    *,
    dataset: Any,
    candidate: Any,
    reconstruction: np.ndarray,
    outliers: np.ndarray,
    expectation: Mapping[str, Any],
) -> dict[str, str]:
    actual = {
        "clean": array_sha256(dataset.clean),
        "observed": array_sha256(dataset.observed),
        "corruption": array_sha256(dataset.corruption),
        "corruption_support": array_sha256(dataset.corruption_support),
        "A": array_sha256(candidate.result.A),
        "B": array_sha256(candidate.result.B),
        "S_normalized": array_sha256(candidate.result.S),
        "reconstruction": array_sha256(reconstruction),
        "outliers_raw_units": array_sha256(outliers),
    }
    data_expected = expectation["data"]
    expected = {
        "clean": data_expected["clean_sha256"],
        "observed": data_expected["observed_sha256"],
        "corruption": data_expected["corruption_sha256"],
        "corruption_support": data_expected["corruption_support_sha256"],
        "A": expectation["candidate"]["final_A_sha256"],
        "B": expectation["candidate"]["final_B_sha256"],
        "S_normalized": expectation["candidate"][
            "final_S_sha256_normalized"
        ],
        "reconstruction": expectation["method"]["reconstruction_sha256"],
        "outliers_raw_units": expectation["method"][
            "outlier_scores_sha256_raw_units"
        ],
    }
    mismatches = {
        name: {"actual": actual[name], "expected": expected[name]}
        for name in expected
        if actual[name] != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"sealed robust exact replay mismatch: {mismatches}")
    fit_sha = canonical_sha256(
        {"A": actual["A"], "B": actual["B"], "S": actual["S_normalized"]}
    )
    if fit_sha != expectation["selected_fit_sha256"]:
        raise RuntimeError("sealed selected fit hash mismatch")
    actual["selected_fit"] = fit_sha
    return actual


def export_locked_frames(
    *,
    formal_dir: str | Path,
    source_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    formal_path = Path(formal_dir)
    source_path = Path(source_root)
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing path: {destination}")

    expectation = _sealed_expectation(formal_path)
    study = _study_config()
    dataset = generate_figure6_dataset(_paper_video_config(study, SEED))
    observed = prepare_observed_case(dataset)
    specs = default_start_specs(study)
    spec_by_id = {spec.start_id: spec for spec in specs}
    if START_ID not in spec_by_id:
        raise RuntimeError("locked start is absent from the start bank")
    initializations = build_initializations(observed, specs)
    candidate = _fit_candidate(
        observed,
        start_id=START_ID,
        arm_id=ARM_ID,
        initial_B=initializations[START_ID],
        alpha=ALPHA,
        target_kappa=TARGET_KAPPA,
        kappa_schedule=KAPPA_SCHEDULE,
        stage_budgets=STAGE_BUDGETS,
        config=study,
    )
    reconstruction, outliers = _candidate_raw_arrays(observed, candidate)
    actual_hashes = _verify_exact_replay(
        dataset=dataset,
        candidate=candidate,
        reconstruction=reconstruction,
        outliers=outliers,
        expectation=expectation,
    )

    indices = np.asarray(FRAME_INDICES, dtype=np.int64)
    arrays = {
        "frame_indices": indices,
        "time": np.ascontiguousarray(dataset.time),
        "clean_frames": np.ascontiguousarray(dataset.clean[:, indices]),
        "observed_frames": np.ascontiguousarray(dataset.observed[:, indices]),
        "reconstruction_frames": np.ascontiguousarray(
            reconstruction[:, indices]
        ),
        "outlier_frames": np.ascontiguousarray(outliers[:, indices]),
        "A": np.ascontiguousarray(candidate.result.A),
        "B": np.ascontiguousarray(candidate.result.B),
        "normalization_sigma_max": np.asarray(
            observed.normalization_sigma_max, dtype=np.float64
        ),
    }
    source = _source_provenance(source_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "export_kind": "sealed_locked_robust_gallery_frames",
        "protocol": {
            "seed": SEED,
            "start_id": START_ID,
            "arm_id": ARM_ID,
            "alpha": ALPHA,
            "target_kappa": TARGET_KAPPA,
            "kappa_schedule": list(KAPPA_SCHEDULE),
            "stage_budgets": list(STAGE_BUDGETS),
            "frame_indices": list(FRAME_INDICES),
            "frame_selection": [
                "predeclared representative frame",
                "sealed full-frame-error worst frame",
            ],
            "native_reconstruction": True,
            "native_outliers_raw_units": True,
            "post_threshold": False,
            "truth_dependent_fit": False,
        },
        "sealed_expectation": expectation,
        "candidate_record": dict(candidate.record),
        "exact_full_array_replay_verified": True,
        "actual_full_array_hashes": actual_hashes,
        "exported_array_hashes": {
            name: array_sha256(values) for name, values in arrays.items()
        },
        "source_provenance": source,
    }

    destination.mkdir(parents=True, exist_ok=False)
    paths = {
        "npz": destination / NPZ_NAME,
        "report": destination / REPORT_NAME,
        "manifest": destination / MANIFEST_NAME,
    }
    np.savez_compressed(paths["npz"], **arrays)
    paths["report"].write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    files = {
        name: {
            "sha256": _file_sha256(destination / name),
            "size_bytes": (destination / name).stat().st_size,
        }
        for name in sorted(_PAYLOAD_NAMES)
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "sealed_locked_robust_gallery_export_exact_closure",
        "self_excluding": True,
        "files": files,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_locked_frames_export(destination)
    return paths


def validate_locked_frames_export(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError("output must be a real directory")
    entries = list(directory.iterdir())
    if {path.name for path in entries} != _OUTPUT_NAMES:
        raise ValueError("export does not have the exact required file set")
    if any(path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) for path in entries):
        raise ValueError("export contains a non-regular entry")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    claimed = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise ValueError("manifest self-hash mismatch")
    if set(manifest.get("files", {})) != _PAYLOAD_NAMES:
        raise ValueError("manifest payload set mismatch")
    for name, record in manifest["files"].items():
        path = directory / name
        if _file_sha256(path) != record["sha256"]:
            raise ValueError(f"payload hash mismatch: {name}")
        if path.stat().st_size != record["size_bytes"]:
            raise ValueError(f"payload size mismatch: {name}")
    report = json.loads((directory / REPORT_NAME).read_text(encoding="utf-8"))
    if report.get("exact_full_array_replay_verified") is not True:
        raise ValueError("exact full-array replay is not verified")
    with np.load(directory / NPZ_NAME, allow_pickle=False) as arrays:
        expected = report["exported_array_hashes"]
        if set(arrays.files) != set(expected):
            raise ValueError("exported array key set mismatch")
        for name, digest in expected.items():
            if array_sha256(arrays[name]) != digest:
                raise ValueError(f"exported array hash mismatch: {name}")
        if tuple(int(value) for value in arrays["frame_indices"]) != FRAME_INDICES:
            raise ValueError("exported frame indices changed")
        expected_shape = (GRID_SIZE * GRID_SIZE, len(FRAME_INDICES))
        for name in (
            "clean_frames",
            "observed_frames",
            "reconstruction_frames",
            "outlier_frames",
        ):
            if arrays[name].shape != expected_shape:
                raise ValueError(f"unexpected exported frame shape: {name}")
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Path]:
    args = _parse_args(argv)
    paths = export_locked_frames(
        formal_dir=args.formal_dir,
        source_root=args.source_root,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "exact_full_array_replay_verified": True,
                "manifest": str(paths["manifest"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return paths


if __name__ == "__main__":
    main()

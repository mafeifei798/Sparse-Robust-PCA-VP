#!/usr/bin/env python3
"""Strictly validate and combine disjoint Genomics benchmark seed shards.

Every input must be a schema-v1 JSON artifact emitted by
``experiments/genomics_1kg_benchmark``.  Shards are independently rehydrated
and passed through the benchmark's write-boundary validator before any records
are combined.  The combined artifact receives a new manifest, canonical
seed/method ordering, rebuilt record indices, and freshly computed summaries.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, fields, replace
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:  # Support direct execution from the repo root.
    repository_root = str(Path(__file__).resolve().parents[1])
    if repository_root not in sys.path:
        sys.path.insert(0, repository_root)

from experiments import genomics_1kg_benchmark as benchmark
from rspca.datasets import make_seed_manifest


_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "config",
        "runtime_versions",
        "manifest",
        "provenance",
        "evaluation_protocol",
        "records",
        "full_runs",
        "method_summaries",
        "paired_summaries",
    }
)
_BENCHMARK_CONFIG_FIELDS = frozenset(
    field.name for field in fields(benchmark.GenomicsBenchmarkConfig)
)
_SINGLE_CONFIG_FIELDS = frozenset(
    field.name for field in fields(benchmark.GenomicsConfig)
)
_FULL_RUN_FIELDS = frozenset(
    {
        "full_run_index",
        "run",
        "run_index",
        "requested_seed",
        "method",
        "runtime_seconds",
        "config",
        "seed_manifest",
        "records",
    }
)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {token}")


def _load_strict_json(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError as error:
        raise ValueError(f"{path}: input is not valid UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{path}: invalid strict JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: benchmark JSON root must be an object")
    benchmark._reject_nonfinite_numbers(payload, location=str(path))
    return payload


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be an object")
    return deepcopy(dict(value))


def _mapping_tuple(value: Any, *, location: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be an array")
    rows: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{location}[{index}] must be an object")
        rows.append(deepcopy(dict(item)))
    return tuple(rows)


def _config_from_json(
    value: Any, *, location: str
) -> benchmark.GenomicsBenchmarkConfig:
    raw = _mapping(value, location=location)
    if set(raw) != _BENCHMARK_CONFIG_FIELDS:
        missing = sorted(_BENCHMARK_CONFIG_FIELDS - set(raw))
        extra = sorted(set(raw) - _BENCHMARK_CONFIG_FIELDS)
        raise ValueError(
            f"{location} schema mismatch; missing={missing}, extra={extra}"
        )
    for field_name in ("seeds", "methods", "alpha_candidates"):
        if not isinstance(raw[field_name], list):
            raise ValueError(f"{location}.{field_name} must be an array")
        raw[field_name] = tuple(raw[field_name])
    config = benchmark.GenomicsBenchmarkConfig(**raw)
    benchmark._validate_config(config)
    return config


def _manifest_from_json(value: Any, *, location: str) -> Mapping[str, Any]:
    manifest = _mapping(value, location=location)
    if "configuration" not in manifest:
        raise ValueError(f"{location} is missing configuration")
    embedded_config = _config_from_json(
        manifest["configuration"],
        location=f"{location}.configuration",
    )
    # The in-memory writer contract contains tuples.  JSON necessarily turns
    # those into arrays, so rehydrate only the declared tuple fields.
    manifest["configuration"] = asdict(embedded_config)
    return manifest


def _full_runs_from_json(value: Any, *, location: str) -> tuple[Mapping[str, Any], ...]:
    runs = list(_mapping_tuple(value, location=location))
    normalized: list[Mapping[str, Any]] = []
    for index, source in enumerate(runs):
        run = dict(source)
        config_location = f"{location}[{index}].config"
        config = _mapping(run.get("config"), location=config_location)
        if set(config) != _SINGLE_CONFIG_FIELDS:
            missing = sorted(_SINGLE_CONFIG_FIELDS - set(config))
            extra = sorted(set(config) - _SINGLE_CONFIG_FIELDS)
            raise ValueError(
                f"{config_location} schema mismatch; missing={missing}, extra={extra}"
            )
        if not isinstance(config["alpha_candidates"], list):
            raise ValueError(f"{config_location}.alpha_candidates must be an array")
        config["alpha_candidates"] = tuple(config["alpha_candidates"])
        run["config"] = config
        normalized.append(run)
    return tuple(normalized)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        benchmark._json_safe(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _validate_record_round_trip(
    result: benchmark.GenomicsBenchmarkResult,
    *,
    label: str,
) -> None:
    grouped = benchmark._validated_records(
        result.records,
        methods=result.config.methods,
        seeds=result.config.seeds,
        benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
    )
    for run_index, requested_seed in enumerate(result.config.seeds):
        for method in result.config.methods:
            record = grouped[method][int(requested_seed)]
            full_run_index = record.get("full_run_index")
            if isinstance(full_run_index, bool) or not isinstance(full_run_index, int):
                raise ValueError(f"{label}: full_run_index must be an integer")
            full_run = result.full_runs[full_run_index]
            if set(full_run) != _FULL_RUN_FIELDS:
                missing = sorted(_FULL_RUN_FIELDS - set(full_run))
                extra = sorted(set(full_run) - _FULL_RUN_FIELDS)
                raise ValueError(
                    f"{label}: full run schema mismatch; missing={missing}, extra={extra}"
                )
            for source, field_name, expected in (
                (record, "run", run_index),
                (record, "run_index", run_index),
                (full_run, "full_run_index", full_run_index),
                (full_run, "run", run_index),
                (full_run, "run_index", run_index),
            ):
                observed = source.get(field_name)
                if (
                    isinstance(observed, bool)
                    or not isinstance(observed, int)
                    or observed != expected
                ):
                    raise ValueError(
                        f"{label}: {field_name} does not match canonical run ordering"
                    )
            rows = full_run.get("records")
            if not isinstance(rows, list):
                raise ValueError(f"{label}: full run records must be an array")
            classifications = [
                row
                for row in rows
                if isinstance(row, Mapping)
                and row.get("record_type") == "held_out_population_classification"
            ]
            if len(classifications) != 1:
                raise ValueError(
                    f"{label}: full run must contain exactly one held-out record"
                )
            for dimension in ("samples", "snps"):
                value = result.provenance.get(dimension)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(
                        f"{label}: provenance {dimension} must be a positive integer"
                    )
            single_config = benchmark._single_run_config(
                result.config,
                method=method,
                seed=int(requested_seed),
            )
            expected_single_manifest = make_seed_manifest(
                int(requested_seed),
                single_config.bootstrap_repeats + 2,
                namespace="genomics-1kg-v2",
                metadata={
                    **asdict(single_config),
                    "n_samples": result.provenance["samples"],
                    "n_snps": result.provenance["snps"],
                    "orientation": "samples-by-snps",
                    "loader_schema_version": 2,
                    "normalized_genotypes_sha256": result.provenance[
                        "normalized_genotypes_sha256"
                    ],
                    "sample_ids_sha256": result.provenance["sample_ids_sha256"],
                    "snp_ids_sha256": result.provenance["snp_ids_sha256"],
                    "seed_roles": {
                        "index_0": "full_cohort_solver",
                        "index_1": "held_out_split_and_selection_parent",
                        "index_2_onward": ("bootstrap_solver_parent_for_resample_seed"),
                    },
                },
            )
            if _canonical_json(full_run["seed_manifest"]) != _canonical_json(
                expected_single_manifest
            ):
                raise ValueError(
                    f"{label}: single-run seed manifest does not bind its config/data"
                )
            expected_record = benchmark._one_record(
                method=method,
                run_index=run_index,
                requested_seed=int(requested_seed),
                runtime_seconds=full_run["runtime_seconds"],
                classification=classifications[0],
                single_manifest=full_run["seed_manifest"],
                benchmark_manifest_sha256=str(result.manifest["manifest_sha256"]),
                provenance=result.provenance,
            )
            expected_record["full_run_index"] = full_run_index
            if dict(record) != expected_record:
                raise ValueError(
                    f"{label}: primary record does not exactly match its full run"
                )


def _validate_result(
    result: benchmark.GenomicsBenchmarkResult,
    *,
    label: str,
) -> None:
    try:
        benchmark._validate_result(result)
        _validate_record_round_trip(result, label=label)
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: invalid Genomics benchmark shard: {error}"
        ) from error


def load_genomics_shard(path: str | Path) -> benchmark.GenomicsBenchmarkResult:
    """Load one persisted shard and enforce its complete write contract."""

    source_path = Path(path)
    payload = _load_strict_json(source_path)
    if set(payload) != _TOP_LEVEL_FIELDS:
        missing = sorted(_TOP_LEVEL_FIELDS - set(payload))
        extra = sorted(set(payload) - _TOP_LEVEL_FIELDS)
        raise ValueError(
            f"{source_path}: top-level schema mismatch; missing={missing}, extra={extra}"
        )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ValueError(f"{source_path}: schema_version must be integer 1")
    runtime = payload["runtime_versions"]
    if (
        not isinstance(runtime, Mapping)
        or not runtime
        or any(not isinstance(key, str) for key in runtime)
        or any(not isinstance(value, str) for value in runtime.values())
    ):
        raise ValueError(f"{source_path}: runtime_versions must map strings to strings")

    config = _config_from_json(payload["config"], location=f"{source_path}.config")
    result = benchmark.GenomicsBenchmarkResult(
        config=config,
        manifest=_manifest_from_json(
            payload["manifest"], location=f"{source_path}.manifest"
        ),
        provenance=_mapping(
            payload["provenance"], location=f"{source_path}.provenance"
        ),
        records=_mapping_tuple(payload["records"], location=f"{source_path}.records"),
        full_runs=_full_runs_from_json(
            payload["full_runs"], location=f"{source_path}.full_runs"
        ),
        method_summaries=_mapping_tuple(
            payload["method_summaries"],
            location=f"{source_path}.method_summaries",
        ),
        paired_summaries=_mapping_tuple(
            payload["paired_summaries"],
            location=f"{source_path}.paired_summaries",
        ),
    )
    expected_protocol = benchmark._json_payload(result)["evaluation_protocol"]
    if payload["evaluation_protocol"] != expected_protocol:
        raise ValueError(f"{source_path}: evaluation protocol mismatch")
    if _canonical_json(payload["config"]) != _canonical_json(asdict(result.config)):
        raise ValueError(f"{source_path}: config failed JSON round-trip validation")
    if _canonical_json(payload["manifest"]) != _canonical_json(result.manifest):
        raise ValueError(f"{source_path}: manifest failed JSON round-trip validation")
    _validate_result(result, label=str(source_path))
    return result


def _config_without_seeds(
    config: benchmark.GenomicsBenchmarkConfig,
) -> Mapping[str, Any]:
    value = asdict(config)
    value.pop("seeds")
    return value


def _expected_seed_plan(seeds: Sequence[int]) -> tuple[int, ...]:
    if len(seeds) < 2:
        raise ValueError("expected seed plan must contain at least two seeds")
    normalized: list[int] = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("expected seeds must be non-negative integers")
        normalized.append(seed)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected seeds must not contain duplicates")
    return tuple(sorted(normalized))


def _combine_validated_shards(
    shards: Sequence[benchmark.GenomicsBenchmarkResult],
    *,
    expected_seeds: Sequence[int],
) -> benchmark.GenomicsBenchmarkResult:
    if len(shards) < 2:
        raise ValueError("at least two Genomics benchmark shards are required")
    expected_seed_plan = _expected_seed_plan(expected_seeds)
    expected_seed_set = set(expected_seed_plan)
    reference = shards[0]
    reference_config = _canonical_json(_config_without_seeds(reference.config))
    reference_provenance = _canonical_json(reference.provenance)
    seen_seeds: set[int] = set()
    source_by_identity: dict[
        tuple[int, str], tuple[Mapping[str, Any], Mapping[str, Any]]
    ] = {}

    for shard_index, shard in enumerate(shards):
        if _canonical_json(_config_without_seeds(shard.config)) != reference_config:
            raise ValueError(
                f"shard {shard_index} config differs outside the seeds field"
            )
        if _canonical_json(shard.provenance) != reference_provenance:
            raise ValueError(f"shard {shard_index} provenance differs")
        shard_seeds = {int(seed) for seed in shard.config.seeds}
        unexpected = shard_seeds - expected_seed_set
        if unexpected:
            raise ValueError(
                f"shard {shard_index} contains unexpected seeds: {sorted(unexpected)}"
            )
        overlap = seen_seeds & shard_seeds
        if overlap:
            raise ValueError(f"shard seed sets overlap: {sorted(overlap)}")
        seen_seeds.update(shard_seeds)
        for record in shard.records:
            seed = int(record["requested_seed"])
            method = str(record["method"])
            full_run_index = int(record["full_run_index"])
            source_by_identity[(seed, method)] = (
                record,
                shard.full_runs[full_run_index],
            )

    if seen_seeds != expected_seed_set:
        missing = sorted(expected_seed_set - seen_seeds)
        unexpected = sorted(seen_seeds - expected_seed_set)
        raise ValueError(
            "shard seed union does not match the expected seed plan; "
            f"missing={missing}, unexpected={unexpected}"
        )
    combined_seeds = expected_seed_plan
    combined_config = replace(reference.config, seeds=combined_seeds)
    combined_provenance = deepcopy(dict(reference.provenance))
    combined_manifest = benchmark._manifest(combined_config, combined_provenance)
    manifest_sha256 = str(combined_manifest["manifest_sha256"])
    records: list[Mapping[str, Any]] = []
    full_runs: list[Mapping[str, Any]] = []

    for run_index, requested_seed in enumerate(combined_seeds):
        for method in combined_config.methods:
            identity = (requested_seed, method)
            if identity not in source_by_identity:
                raise ValueError(f"combined shard matrix is missing {identity}")
            _, source_full_run = source_by_identity[identity]
            full_run = deepcopy(dict(source_full_run))
            full_run_index = len(full_runs)
            full_run["full_run_index"] = full_run_index
            full_run["run"] = run_index
            full_run["run_index"] = run_index
            full_runs.append(full_run)
            classifications = [
                row
                for row in full_run["records"]
                if row.get("record_type") == "held_out_population_classification"
            ]
            record = benchmark._one_record(
                method=method,
                run_index=run_index,
                requested_seed=requested_seed,
                runtime_seconds=full_run["runtime_seconds"],
                classification=classifications[0],
                single_manifest=full_run["seed_manifest"],
                benchmark_manifest_sha256=manifest_sha256,
                provenance=combined_provenance,
            )
            record["full_run_index"] = full_run_index
            records.append(record)

    method_summaries = benchmark.summarize_method_metrics(
        records,
        methods=combined_config.methods,
        seeds=combined_config.seeds,
        benchmark_manifest_sha256=manifest_sha256,
    )
    paired_summaries = benchmark.summarize_paired_differences(
        records,
        methods=combined_config.methods,
        seeds=combined_config.seeds,
        benchmark_manifest_sha256=manifest_sha256,
        bootstrap_resamples=combined_config.bootstrap_resamples,
        bootstrap_seed=combined_config.bootstrap_seed,
    )
    combined = benchmark.GenomicsBenchmarkResult(
        config=combined_config,
        manifest=combined_manifest,
        provenance=combined_provenance,
        records=tuple(records),
        full_runs=tuple(full_runs),
        method_summaries=tuple(method_summaries),
        paired_summaries=tuple(paired_summaries),
    )
    _validate_result(combined, label="combined result")
    return combined


def combine_shards(
    shards: Sequence[benchmark.GenomicsBenchmarkResult],
    *,
    expected_seeds: Sequence[int],
) -> benchmark.GenomicsBenchmarkResult:
    """Validate in-memory shards and combine their disjoint seed matrices."""

    for index, shard in enumerate(shards):
        _validate_result(shard, label=f"shard {index}")
    return _combine_validated_shards(shards, expected_seeds=expected_seeds)


def combine_shard_paths(
    paths: Sequence[str | Path],
    *,
    expected_seeds: Sequence[int],
) -> benchmark.GenomicsBenchmarkResult:
    """Load, validate, and combine persisted Genomics benchmark shards."""

    if len(paths) < 2:
        raise ValueError("at least two Genomics benchmark shards are required")
    shards = tuple(load_genomics_shard(path) for path in paths)
    return _combine_validated_shards(shards, expected_seeds=expected_seeds)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument(
        "--expected-seeds",
        type=benchmark._parse_seeds,
        required=True,
        help="Required full seed plan, e.g. 2025..2044",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    benchmark.validate_output_paths(
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    result = combine_shard_paths(
        args.shards,
        expected_seeds=args.expected_seeds,
    )
    paths = benchmark.write_results(
        result,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": result.manifest["manifest_sha256"],
                "seeds": list(result.config.seeds),
                "records": len(result.records),
                "full_runs": len(result.full_runs),
                "method_summaries": len(result.method_summaries),
                "paired_summaries": len(result.paired_summaries),
                "outputs": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "combine_shard_paths",
    "combine_shards",
    "load_genomics_shard",
    "main",
]

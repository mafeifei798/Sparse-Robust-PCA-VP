"""Dependency-light confirmatory statistics for the cross-modal audit.

These helpers deliberately operate on paired seed-level contrasts.  They do
not turn training seeds into independent data samples and they do not inspect
or select methods.  Callers must freeze the contrasts and hypothesis family
before using the functions for a confirmation gate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import itertools
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
DATASET_IDS = ("eurosat", "svhn", "20news")
VISION_DATASET_IDS = frozenset(("eurosat", "svhn"))


def _finite_vector(values: ArrayLike, *, name: str, minimum: int = 1) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if raw.size < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain real non-boolean values")
    result = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def paired_mean_percentile_interval(
    differences: ArrayLike,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 20261204,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Return a deterministic percentile interval for a paired seed mean.

    The interval characterizes variation over the supplied paired training
    seeds only; it is not a population-sampling confidence interval.
    """

    values = _finite_vector(differences, name="differences", minimum=2)
    repeats = _positive_integer(n_bootstrap, name="n_bootstrap")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, Real
    ):
        raise TypeError("confidence_level must be real")
    confidence = float(confidence_level)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")

    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(repeats, values.size))
    means = np.mean(values[indices], axis=1, dtype=np.float64)
    alpha = 0.5 * (1.0 - confidence)
    lower, upper = np.quantile(means, (alpha, 1.0 - alpha))
    return {
        "n_pairs": int(values.size),
        "n_bootstrap": repeats,
        "seed": int(seed),
        "confidence_level": confidence,
        "mean": float(np.mean(values, dtype=np.float64)),
        "sample_standard_deviation": float(np.std(values, ddof=1)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "scope": (
            "partition_initialization_and_training_randomness_not_"
            "population_sampling_uncertainty"
        ),
    }


def exact_paired_sign_flip_pvalue(
    differences: ArrayLike,
    *,
    alternative: str = "greater",
) -> float:
    """Compute an exact paired randomization p-value for up to 20 contrasts."""

    values = _finite_vector(differences, name="differences", minimum=1)
    if values.size > 20:
        raise ValueError("exact sign-flip enumeration is limited to 20 pairs")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    observed = float(np.mean(values, dtype=np.float64))
    null = np.asarray(
        [
            float(np.mean(values * np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=values.size)
        ],
        dtype=np.float64,
    )
    tolerance = 16.0 * np.finfo(np.float64).eps * max(1.0, abs(observed))
    if alternative == "greater":
        extreme = null >= observed - tolerance
    elif alternative == "less":
        extreme = null <= observed + tolerance
    else:
        extreme = np.abs(null) >= abs(observed) - tolerance
    return float(np.mean(extreme))


def holm_adjusted_pvalues(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise-error adjusted p-values by hypothesis name."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise ValueError("p_values must be a non-empty mapping")
    validated: list[tuple[str, float]] = []
    for name, value in p_values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("hypothesis names must be non-empty strings")
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"p-value for {name!r} must be real")
        probability = float(value)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"p-value for {name!r} must lie in [0, 1]")
        validated.append((name, probability))
    if len({name for name, _ in validated}) != len(validated):
        raise ValueError("hypothesis names must be unique")

    ordered = sorted(validated, key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, probability) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * probability)
        running = max(running, candidate)
        adjusted[name] = running
    return {name: adjusted[name] for name, _ in validated}


def cross_modal_requirement_met(
    passing_datasets: Sequence[str],
    *,
    datasets_required: int = 2,
) -> bool:
    """Require 20News and at least one vision task among passing datasets."""

    if isinstance(passing_datasets, (str, bytes)) or not isinstance(
        passing_datasets, Sequence
    ):
        raise TypeError("passing_datasets must be a sequence")
    required = _positive_integer(datasets_required, name="datasets_required")
    values = tuple(passing_datasets)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("dataset identifiers must be strings")
    if len(set(values)) != len(values):
        raise ValueError("passing_datasets must not contain duplicates")
    unknown = sorted(set(values) - set(DATASET_IDS))
    if unknown:
        raise ValueError(f"unknown dataset identifiers: {unknown}")
    passed = set(values)
    return (
        len(passed) >= required
        and "20news" in passed
        and bool(passed & VISION_DATASET_IDS)
    )


__all__ = [
    "cross_modal_requirement_met",
    "exact_paired_sign_flip_pvalue",
    "holm_adjusted_pvalues",
    "paired_mean_percentile_interval",
]

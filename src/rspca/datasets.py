"""Deterministic datasets and provenance manifests for SPCA experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
import scipy


@dataclass(frozen=True)
class SyntheticSparseLowRank:
    """A fully observed sparse low-rank simulation with recorded ground truth.

    All matrices use the conventional ``samples x features`` orientation.
    ``uncorrupted`` includes Gaussian measurement noise but excludes cellwise
    gross corruptions; ``low_rank`` is the noise-free target.
    """

    observed: NDArray[np.float64]
    uncorrupted: NDArray[np.float64]
    low_rank: NDArray[np.float64]
    gaussian_noise: NDArray[np.float64]
    corruption: NDArray[np.float64]
    scores: NDArray[np.float64]
    loadings: NDArray[np.float64]
    feature_support: NDArray[np.bool_]
    corruption_support: NDArray[np.bool_]
    singular_values: NDArray[np.float64]
    seed: int


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative")
    return integer


def generate_sparse_low_rank(
    n_samples: int = 240,
    n_features: int = 120,
    rank: int = 3,
    *,
    support_size: int | None = None,
    support_fraction: float = 0.15,
    singular_values: Sequence[float] | None = None,
    noise_std: float = 0.05,
    corruption_fraction: float = 0.05,
    corruption_scale: float = 8.0,
    corruption_distribution: str = "gaussian",
    disjoint_support: bool = True,
    seed: int = 0,
) -> SyntheticSparseLowRank:
    """Generate a sparse-factor, low-rank matrix with known cellwise outliers.

    Exactly ``round(corruption_fraction * n_samples * n_features)`` distinct
    cells are corrupted.  ``corruption_scale`` is measured relative to the
    empirical standard deviation of the noise-free low-rank matrix (or to
    ``noise_std`` when that is larger).  Supported corruption distributions are
    ``"gaussian"``, ``"laplace"``, and ``"rademacher"``.
    """

    n_samples = _positive_integer(n_samples, "n_samples")
    n_features = _positive_integer(n_features, "n_features")
    rank = _positive_integer(rank, "rank")
    if rank >= n_samples:
        raise ValueError("rank must be smaller than n_samples after centering")
    if rank > n_features:
        raise ValueError("rank must not exceed n_features")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if not 0.0 <= corruption_fraction <= 1.0:
        raise ValueError("corruption_fraction must lie in [0, 1]")
    if corruption_scale < 0 or (corruption_fraction > 0 and corruption_scale == 0):
        raise ValueError("corruption_scale must be positive when corruptions are used")
    distribution = corruption_distribution.lower()
    if distribution not in {"gaussian", "laplace", "rademacher"}:
        raise ValueError(
            "corruption_distribution must be gaussian, laplace, or rademacher"
        )
    if support_size is None:
        if not 0.0 < support_fraction <= 1.0:
            raise ValueError("support_fraction must lie in (0, 1]")
        support_size = max(1, int(round(support_fraction * n_features)))
    else:
        support_size = _positive_integer(support_size, "support_size")
    if support_size > n_features:
        raise ValueError("support_size must not exceed n_features")
    if disjoint_support and rank * support_size > n_features:
        raise ValueError(
            "disjoint component supports require rank * support_size <= n_features"
        )

    seed = _nonnegative_integer(seed, "seed")
    rng = np.random.default_rng(seed)

    support = np.zeros((n_features, rank), dtype=bool)
    if disjoint_support:
        shuffled_features = rng.permutation(n_features)
        for component in range(rank):
            start = component * support_size
            support[shuffled_features[start : start + support_size], component] = True
    else:
        for component in range(rank):
            selected = rng.choice(n_features, size=support_size, replace=False)
            support[selected, component] = True

    loadings = np.zeros((n_features, rank), dtype=float)
    for component in range(rank):
        values = rng.normal(size=support_size)
        norm = float(np.linalg.norm(values))
        while norm <= np.finfo(float).eps:
            values = rng.normal(size=support_size)
            norm = float(np.linalg.norm(values))
        loadings[support[:, component], component] = values / norm

    raw_scores = rng.normal(size=(n_samples, rank))
    raw_scores -= raw_scores.mean(axis=0, keepdims=True)
    score_basis, _ = np.linalg.qr(raw_scores, mode="reduced")
    if singular_values is None:
        spectrum = np.geomspace(12.0, 4.0, num=rank, dtype=float)
    else:
        spectrum = np.asarray(singular_values, dtype=float)
        if spectrum.shape != (rank,):
            raise ValueError(f"singular_values must contain exactly {rank} values")
        if not np.all(np.isfinite(spectrum)) or np.any(spectrum <= 0):
            raise ValueError("singular_values must be finite and positive")
    scores = score_basis[:, :rank] * spectrum
    low_rank = scores @ loadings.T

    gaussian_noise = rng.normal(scale=noise_std, size=low_rank.shape)
    # The latent scores are column-centered.  Center the Gaussian measurement
    # noise as a component of the DGP as well, before sparse corruptions are
    # injected.  Empirically centering the final observed matrix would spread a
    # single corrupted cell over its entire feature column and destroy the exact
    # cellwise-support contract used by robust-recovery metrics.
    gaussian_noise -= gaussian_noise.mean(axis=0, keepdims=True)
    uncorrupted = low_rank + gaussian_noise
    corruption = np.zeros_like(low_rank)
    corruption_support = np.zeros_like(low_rank, dtype=bool)
    n_corruptions = int(round(corruption_fraction * low_rank.size))
    if n_corruptions:
        flat_indices = rng.choice(low_rank.size, size=n_corruptions, replace=False)
        corruption_support.flat[flat_indices] = True
        reference_scale = max(float(np.std(low_rank)), float(noise_std), 1e-12)
        scale = corruption_scale * reference_scale
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("the effective corruption scale must be finite and positive")
        if distribution == "gaussian":
            corruptions = rng.normal(scale=scale, size=n_corruptions)
        elif distribution == "laplace":
            corruptions = rng.laplace(scale=scale / np.sqrt(2.0), size=n_corruptions)
        elif distribution == "rademacher":
            corruptions = scale * rng.choice(np.array([-1.0, 1.0]), size=n_corruptions)
        # The declared corruption budget is also the exact support budget.
        # Continuous RNGs produce an exact zero with probability zero, but make
        # that software contract deterministic rather than probabilistic.
        corruptions[corruptions == 0.0] = scale
        corruption.flat[flat_indices] = corruptions

    observed = uncorrupted + corruption
    return SyntheticSparseLowRank(
        observed=observed,
        uncorrupted=uncorrupted,
        low_rank=low_rank,
        gaussian_noise=gaussian_noise,
        corruption=corruption,
        scores=scores,
        loadings=loadings,
        feature_support=support,
        corruption_support=corruption_support,
        singular_values=spectrum,
        seed=seed,
    )


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a mapping using sorted, whitespace-free UTF-8 JSON."""

    compatible = _json_compatible(payload)
    encoded = json.dumps(
        compatible,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(values: Any) -> str:
    """Hash an array's shape, dtype, and canonical little-endian content."""

    array = np.asarray(values)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot be content-hashed deterministically")
    canonical_dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(canonical_dtype, copy=False))
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def runtime_versions() -> dict[str, str]:
    """Return the runtime identifiers needed to interpret saved artifacts.

    Numerical random generators and linear-algebra kernels can change across
    dependency versions.  Recording these identifiers does not promise
    cross-platform bitwise reproduction, but it makes that boundary explicit.
    """

    try:
        package_version = metadata.version("vp-rspca")
    except metadata.PackageNotFoundError:
        package_version = "uninstalled-source-tree"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "vp_rspca": package_version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def make_seed_manifest(
    base_seed: int,
    n_runs: int,
    *,
    namespace: str = "rspca",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a deterministic, content-addressed list of independent seeds."""

    base_seed = _nonnegative_integer(base_seed, "base_seed")
    n_runs = _positive_integer(n_runs, "n_runs")
    if not namespace:
        raise ValueError("namespace must not be empty")
    namespace_digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    namespace_words = np.frombuffer(namespace_digest[:8], dtype="<u4").astype(np.uint32)
    sequence = np.random.SeedSequence(
        [base_seed, int(namespace_words[0]), int(namespace_words[1])]
    )
    children = sequence.spawn(n_runs)
    seeds = [int(child.generate_state(1, dtype=np.uint64)[0]) for child in children]
    body: dict[str, Any] = {
        "manifest_version": 1,
        "namespace": namespace,
        "base_seed": base_seed,
        "seeds": seeds,
        "metadata": _json_compatible(dict(metadata or {})),
    }
    body["manifest_sha256"] = canonical_sha256(body)
    return body


def verify_seed_manifest(manifest: Mapping[str, Any]) -> bool:
    """Return whether a manifest's stored digest matches its contents."""

    if "manifest_sha256" not in manifest:
        return False
    body = dict(manifest)
    claimed = body.pop("manifest_sha256")
    return isinstance(claimed, str) and claimed == canonical_sha256(body)


__all__ = [
    "SyntheticSparseLowRank",
    "array_sha256",
    "canonical_sha256",
    "generate_sparse_low_rank",
    "make_seed_manifest",
    "runtime_versions",
    "verify_seed_manifest",
]

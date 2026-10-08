from __future__ import annotations

import numpy as np
import pytest

from rspca.datasets import (
    array_sha256,
    generate_sparse_low_rank,
    make_seed_manifest,
    verify_seed_manifest,
)


@pytest.mark.parametrize("invalid_seed", [True, np.bool_(True), 1.1, 1.9, "1"])
def test_generator_rejects_non_integer_seed_types(invalid_seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        generate_sparse_low_rank(
            n_samples=12,
            n_features=8,
            rank=2,
            support_size=2,
            seed=invalid_seed,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("invalid_seed", [True, np.bool_(True), 1.1, 1.9, "1"])
def test_manifest_rejects_non_integer_base_seed_types(invalid_seed: object) -> None:
    with pytest.raises(TypeError, match="base_seed must be an integer"):
        make_seed_manifest(invalid_seed, 3)  # type: ignore[arg-type]


@pytest.mark.parametrize("invalid_count", [True, np.bool_(True), 1.0, 1.9, "2"])
def test_positive_integer_parameters_reject_non_integer_types(
    invalid_count: object,
) -> None:
    with pytest.raises(TypeError, match="n_samples must be an integer"):
        generate_sparse_low_rank(
            n_samples=invalid_count,  # type: ignore[arg-type]
            n_features=8,
            rank=2,
            support_size=2,
            seed=1,
        )

    with pytest.raises(TypeError, match="n_runs must be an integer"):
        make_seed_manifest(1, invalid_count)  # type: ignore[arg-type]


def test_python_and_numpy_integer_seeds_keep_identical_content_digests() -> None:
    python_seed_data = generate_sparse_low_rank(
        n_samples=12,
        n_features=8,
        rank=2,
        support_size=2,
        seed=7,
    )
    numpy_seed_data = generate_sparse_low_rank(
        n_samples=np.int64(12),
        n_features=np.int64(8),
        rank=np.int64(2),
        support_size=np.int64(2),
        seed=np.int64(7),
    )
    assert array_sha256(python_seed_data.observed) == array_sha256(
        numpy_seed_data.observed
    )

    python_manifest = make_seed_manifest(7, 3, namespace="seed-contract")
    numpy_manifest = make_seed_manifest(
        np.int64(7),
        np.int64(3),
        namespace="seed-contract",
    )
    assert python_manifest == numpy_manifest
    assert verify_seed_manifest(numpy_manifest)


def test_fractional_seeds_cannot_alias_an_integer_manifest_digest() -> None:
    integer_manifest = make_seed_manifest(1, 2, namespace="no-seed-alias")
    assert verify_seed_manifest(integer_manifest)

    for fractional_seed in (1.1, 1.9):
        with pytest.raises(TypeError, match="base_seed must be an integer"):
            make_seed_manifest(fractional_seed, 2, namespace="no-seed-alias")

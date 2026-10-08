from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "spca_variable_projection_fista_bt_torch"


def test_legacy_torch_module_is_declared_for_wheel_install() -> None:
    """The import shim must remain a top-level module in built distributions."""

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"{MODULE_NAME}"' in pyproject


def test_legacy_gram_procrustes_contract_does_not_square_the_gram() -> None:
    """An old ``(G, B)`` call must compute polar(G B), not polar(G^2 B)."""

    torch = pytest.importorskip("torch")
    legacy = importlib.import_module(MODULE_NAME)
    canonical = importlib.import_module("rspca.torch_backend")

    signature = inspect.signature(legacy.update_A_procrustes_torch)
    assert tuple(signature.parameters) == ("G", "B")

    gram = torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    loadings = torch.tensor([[1.0], [1.0]], dtype=torch.float64)
    cross = gram @ loadings
    expected = cross / torch.linalg.vector_norm(cross)

    with pytest.deprecated_call(match="Gram-matrix contract"):
        actual = legacy.update_A_procrustes_torch(gram, loadings)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.mT @ actual, torch.eye(1, dtype=torch.float64))

    # The canonical helper intentionally has a different, matrix-free ``(X, B)``
    # contract. Passing G as X there demonstrates the exact silent regression
    # this compatibility shim prevents.
    squared_gram_result = canonical.update_A_procrustes_torch(gram, loadings)
    assert not torch.allclose(actual, squared_gram_result)


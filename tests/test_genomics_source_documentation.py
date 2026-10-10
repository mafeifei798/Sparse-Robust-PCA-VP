"""Fail-closed source checks for documentation-only solver revisions."""

from hashlib import sha256
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
shared = importlib.import_module("genomics_randomized_vp_5pct")


def source_pair(tmp_path, monkeypatch, *, allowed=True):
    name = "solver.py"
    original = b'"""Original documentation."""\ndef f(x=2):\n    """Evaluate x."""\n    return x * 3\n'
    monkeypatch.setattr(shared, "ROOT", tmp_path)
    monkeypatch.setattr(shared, "SOURCE_LOCKS", {name: sha256(original).hexdigest()})
    monkeypatch.setattr(shared, "DOCUMENTATION_REVISION_PATHS", {name} if allowed else set())
    monkeypatch.setattr(shared, "_frozen_source", lambda path: original)
    (tmp_path / name).write_bytes(original)
    return tmp_path / name, original


def test_pinned_sources_remain_authenticated():
    status = shared.verify_sources()
    assert set(status) == set(shared.SOURCE_LOCKS)
    assert {name for name, state in status.items() if state == "documentation_only"} == set(
        shared.DOCUMENTATION_REVISION_PATHS
    )


def test_byte_identical_source_needs_no_history(tmp_path, monkeypatch):
    path, _ = source_pair(tmp_path, monkeypatch)

    def unavailable(name):
        raise AssertionError("byte-identical source must not read Git history")

    monkeypatch.setattr(shared, "_frozen_source", unavailable)
    assert shared.verify_sources() == {path.name: "byte_identical"}


def test_comments_and_docstrings_are_accepted(tmp_path, monkeypatch):
    path, original = source_pair(tmp_path, monkeypatch)
    path.write_bytes(b"# Technical note.\n" + original.replace(b"Original documentation.", b"Revised documentation."))
    assert shared.verify_sources() == {path.name: "documentation_only"}


@pytest.mark.parametrize("old,new", [(b"x=2", b"x=4"), (b"x * 3", b"x + 3"), (b"x * 3", b"x * 4")])
def test_executable_changes_are_rejected(tmp_path, monkeypatch, old, new):
    path, original = source_pair(tmp_path, monkeypatch)
    path.write_bytes(original.replace(old, new))
    with pytest.raises(ValueError, match="executable source changed"):
        shared.verify_sources()


def test_original_blob_hash_is_verified(tmp_path, monkeypatch):
    path, original = source_pair(tmp_path, monkeypatch)
    path.write_bytes(b"# Note.\n" + original)
    monkeypatch.setattr(shared, "_frozen_source", lambda name: b"untrusted source")
    with pytest.raises(ValueError, match="historical source anchor mismatch"):
        shared.verify_sources()


def test_missing_historical_blob_is_rejected(monkeypatch):
    def missing(*args, **kwargs):
        raise subprocess.CalledProcessError(128, args[0])

    monkeypatch.setattr(shared.subprocess, "run", missing)
    with pytest.raises(ValueError, match="historical source unavailable"):
        shared._frozen_source("src/rspca/randomized.py")


def test_other_paths_keep_strict_byte_locks(tmp_path, monkeypatch):
    path, original = source_pair(tmp_path, monkeypatch, allowed=False)
    path.write_bytes(b"# Note.\n" + original)
    with pytest.raises(ValueError, match="frozen source changed"):
        shared.verify_sources()

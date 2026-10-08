from __future__ import annotations

from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEEP_LIST = ROOT / "experiments/configs/genomics_keep_5superpops_20260826.txt"
DOCUMENTATION = ROOT / "docs/genomics_1kg.md"


def test_frozen_genomics_keep_list_is_complete_unique_and_hash_bound() -> None:
    payload = KEEP_LIST.read_bytes()
    assert sha256(payload).hexdigest() == (
        "fc89d811dcde3becdd1a5b26a2ecd7150a75de0c42af23f3269f2157cfe89d9e"
    )

    rows = KEEP_LIST.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1000
    pairs = [row.split("\t") for row in rows]
    assert all(len(pair) == 2 and pair[0] == pair[1] for pair in pairs)
    assert len({pair[0] for pair in pairs}) == 1000


def test_genomics_reproduction_document_binds_inputs_outputs_and_command() -> None:
    text = DOCUMENTATION.read_text(encoding="utf-8")
    for digest in (
        "a90c16c4ff2b3196476d506ae13cb3047fae8670163c7c932c4b0239aef3daf5",
        "b4023dc6ee2d62ee89c8d4d347db4d348e65518d66d346574cdae7a4bbd76858",
        "fc89d811dcde3becdd1a5b26a2ecd7150a75de0c42af23f3269f2157cfe89d9e",
        "cb79b22e605a3b8a34968844dc32193ac4047a17dff77034f8a898807583759e",
        "55f351f5ac57e0fd18f6a4b2683e5df81a4fa197d48df3f564b72bf1911d86ac",
    ):
        assert digest in text
    for required in (
        "--keep experiments/configs/genomics_keep_5superpops_20260826.txt",
        "--bp-space 50000",
        "tools/plink_raw_to_npz.py",
        "population-structure benchmark",
        "not a race classifier",
    ):
        assert required in text

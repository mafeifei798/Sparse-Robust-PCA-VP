from __future__ import annotations

import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import rspca

from tools.generate_genomics_smoke_data import generate_smoke_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_public_api_is_importable() -> None:
    expected = {
        "fit_vp_spca",
        "fit_accelerated_vp_spca",
        "fit_randomized_vp_spca",
        "certified_randomized_vp_spca",
        "fit_robust_vp_spca",
        "principal_component_pursuit_admm",
        "fedrpca_aggregate",
        "reliability_gated_fedrpca",
    }
    assert expected <= set(rspca.__all__)
    for name in expected:
        assert callable(getattr(rspca, name))


def test_historical_algorithm_modules_are_thin_importable_compatibility_layers() -> None:
    sys.path.insert(0, str(ROOT))
    expected = {
        "spca_variable_projection": "spca_vp",
        "randomized_spca_vp": "randomized_spca_vp",
        "robust_spca_vp": "robust_spca_vp",
        "spca_variable_projection_fista_bt": "spca_algorithm_4_1_fista_bt",
    }
    for module_name, symbol in expected.items():
        module = importlib.import_module(module_name)
        assert symbol in module.__all__
        assert callable(getattr(module, symbol))


def test_maintained_python_files_parse_and_vp_code_has_no_explicit_gram() -> None:
    maintained_files = [
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "experiments").glob("*.py")),
        ROOT / "tools/generate_genomics_smoke_data.py",
        ROOT / "tools/execute_nonfed_notebooks.py",
    ]
    for path in maintained_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    vp_files = [
        ROOT / "src/rspca/vp.py",
        ROOT / "src/rspca/randomized.py",
        ROOT / "src/rspca/robust.py",
        ROOT / "src/rspca/accelerated.py",
        ROOT / "src/rspca/torch_backend.py",
    ]
    for path in vp_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.MatMult):
                continue
            left, right = node.left, node.right
            direct_gram = (
                isinstance(left, ast.Attribute)
                and left.attr in {"T", "mT"}
                and isinstance(left.value, ast.Name)
                and isinstance(right, ast.Name)
                and left.value.id == right.id
                and left.value.id.lower() in {"x", "x_array", "matrix"}
            )
            assert not direct_gram, f"explicit Gram matrix in {path}:{node.lineno}"


def test_all_committed_notebooks_are_output_free() -> None:
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            assert cell.get("execution_count") is None, str(path)
            assert cell.get("outputs", []) == [], str(path)


def test_canonical_notebooks_delegate_to_experiment_scripts() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook.get("cells", [])):
            expected_id = hashlib.sha256(
                f"{path.name}:{index}".encode("utf-8")
            ).hexdigest()[:12]
            assert cell.get("id") == expected_id
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
        )
        assert "%run" in source
        assert "def spca" not in source.lower()
        assert "except" not in source.lower()

    video_source = (ROOT / "notebooks/video_spca.ipynb").read_text(
        encoding="utf-8"
    )
    genomics_source = (ROOT / "notebooks/genomics_1kg.ipynb").read_text(
        encoding="utf-8"
    )
    assert "video_spca_benchmark.py" in video_source
    assert "--repeats 2" in video_source
    assert "--method pca" in genomics_source
    assert "--method vp" not in genomics_source


def test_genomics_smoke_generator_is_byte_deterministic(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_manifest = generate_smoke_dataset(first_dir)
    second_manifest = generate_smoke_dataset(second_dir)

    assert first_manifest == second_manifest
    for name in (
        "synthetic_genotypes.npz",
        "synthetic_populations.csv",
        "manifest.json",
    ):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()

    with np.load(first_dir / "synthetic_genotypes.npz", allow_pickle=False) as data:
        assert set(data.files) == {"X", "column_ids", "row_ids", "variant_ids"}
        assert data["X"].shape == (30, 18)
        assert data["row_ids"].shape == (30,)
        assert data["variant_ids"].shape == (18,)
        assert np.array_equal(data["column_ids"], data["variant_ids"])
        assert np.count_nonzero(data["X"] == -9) == first_manifest["matrix"][
            "missing_count"
        ]


def _make_recipe(makefile: str, target: str) -> str:
    lines = makefile.splitlines()
    start = lines.index(f"{target}:") + 1
    recipe: list[str] = []
    for line in lines[start:]:
        if not line.strip():
            break
        if not line.startswith("\t"):
            break
        recipe.append(line.strip())
    return "\n".join(recipe)


def test_nonfed_make_contract_excludes_federated_runners() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    test_files = makefile.split("NONFED_TESTS =", 1)[1].split("\n\n", 1)[0]
    test_recipe = _make_recipe(makefile, "test-nonfed")
    smoke_recipe = _make_recipe(makefile, "smoke-nonfed")
    notebook_recipe = _make_recipe(makefile, "smoke-notebooks-nonfed")

    assert "$(NONFED_TESTS)" in test_recipe
    assert "federated" not in test_files.lower()
    assert "federated" not in smoke_recipe.lower()
    assert "experiments/spca_benchmark.py" in smoke_recipe
    assert "experiments/video_spca.py" in smoke_recipe
    assert "experiments/video_spca_benchmark.py" in smoke_recipe
    assert "tools/generate_genomics_smoke_data.py" in smoke_recipe
    assert "experiments/genomics_1kg.py" in smoke_recipe
    assert "federated" not in notebook_recipe.lower()
    assert "tools/execute_nonfed_notebooks.py" in notebook_recipe

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "make test-nonfed" in readme
    assert "make smoke-nonfed" in readme
    assert "tools/generate_genomics_smoke_data.py" in readme

    notebook_smoke = (ROOT / "tools/execute_nonfed_notebooks.py").read_text(
        encoding="utf-8"
    )
    assert "spca_benchmark.ipynb" in notebook_smoke
    assert "video_spca.ipynb" in notebook_smoke
    assert "genomics_1kg.ipynb" in notebook_smoke
    assert "fedrpca.ipynb" not in notebook_smoke

    workflow = (ROOT / ".github/workflows/tests.yml").read_text(encoding="utf-8")
    assert "nonfed-optional-paths:" in workflow
    assert ".[experiments,torch,notebooks,dev]" in workflow
    assert "python tools/execute_nonfed_notebooks.py" in workflow

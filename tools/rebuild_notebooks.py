"""Build the small, output-free notebooks shipped with the repository.

The notebooks are generated so their source cannot silently diverge from the
canonical command-line experiments.  Run this file from the repository root.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def _write(name: str, title: str, markdown: str, commands: list[str]) -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook["metadata"]["language_info"] = {"name": "python", "version": "3.10"}
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\n{markdown}"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "ROOT = Path.cwd().resolve().parent if Path.cwd().name == 'notebooks' else Path.cwd().resolve()\n"
            "ROOT"
        ),
    ]
    notebook["cells"].extend(nbf.v4.new_code_cell(command) for command in commands)
    # nbformat otherwise generates random cell IDs at write time, causing a
    # source-identical rebuild to dirty every notebook.
    for index, cell in enumerate(notebook["cells"]):
        cell["id"] = hashlib.sha256(f"{name}:{index}".encode("utf-8")).hexdigest()[:12]
    path = NOTEBOOK_DIR / name
    nbf.write(notebook, path)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    _write(
        "spca_benchmark.ipynb",
        "Paired SPCA benchmark",
        "This notebook delegates to the tested benchmark runner. It adds noise and "
        "known cellwise corruptions, evaluates recovery metrics, and compares methods "
        "under paired seeds. Generated outputs are written under `artifacts/` and are "
        "not embedded in the notebook.",
        [
            "script = ROOT / 'experiments' / 'spca_benchmark.py'\n"
            "output_dir = ROOT / 'artifacts' / 'spca_benchmark'\n"
            "output_json = output_dir / 'results.json'\n"
            "output_csv = output_dir / 'metrics.csv'\n"
            "%run \"$script\" --repeats 3 --base-seed 2025 --n-samples 120 --n-features 80 --rank 3 --alphas 0.03,0.1 --max-iter 150 --output-json \"$output_json\" --output-csv \"$output_csv\"",
        ],
    )

    _write(
        "fedrpca.ipynb",
        "Audited FedRPCA aggregation",
        "This is a deterministic low-dimensional decomposition experiment, not a "
        "claim of end-to-end CLIP LoRA reproduction. It compares FedAvg, faithful "
        "FedRPCA, adaptive FedRPCA, and TIES on the same generated client updates "
        "and seed manifest. The evaluated near-null reliability gate is available "
        "only through an explicit CLI opt-in.",
        [
            "script = ROOT / 'experiments' / 'federated_synthetic.py'\n"
            "output_file = ROOT / 'artifacts' / 'federated_synthetic' / 'metrics.jsonl'\n"
            "%run \"$script\" --seeds 2025 2026 2027 --n-clients 12 --input-dim 32 --output-dim 24 --output \"$output_file\"",
        ],
    )

    _write(
        "video_spca.ipynb",
        "Paired spatially sparse video benchmark",
        "Rows are time samples and columns are pixel features, so the loading "
        "penalty acts spatially. Within each seed all methods receive the same "
        "noisy/corrupted movie. Ground truth is used only for post-fit scoring, "
        "never for hyperparameter or best-seed selection. Paired uncertainty is "
        "computed over seeds; wall-clock runtime is not bootstrapped.",
        [
            "script = ROOT / 'experiments' / 'video_spca_benchmark.py'\n"
            "output_dir = ROOT / 'artifacts' / 'video_spca'\n"
            "output_json = output_dir / 'paired_results.json'\n"
            "output_csv = output_dir / 'paired_metrics.csv'\n"
            "%run \"$script\" --repeats 2 --base-seed 2025 --methods pca,vp,robust --grid-size 10 --n-timesteps 30 --max-iter 10 --bootstrap-resamples 200 --bootstrap-seed 2025 --output-json \"$output_json\" --output-csv \"$output_csv\"",
        ],
    )

    _write(
        "genomics_1kg.ipynb",
        "1000 Genomes-style SPCA workflow (synthetic smoke)",
        "This notebook executes a tiny synthetic-genotype smoke, not a 1000 Genomes "
        "result. Replace `genotype_path` and `metadata_path` with verified prepared "
        "inputs for an actual analysis. The canonical matrix orientation is samples "
        "by SNPs. There is no fallback solver. Each raw bootstrap refits preprocessing, "
        "and population prediction uses one deterministic held-out split. Optional "
        "SPCA alpha grids are selected on a nested training-only validation split; "
        "the smoke uses the supported PCA default. Downloading and PLINK preprocessing "
        "remain an explicit external data-preparation step.",
        [
            "output_dir = ROOT / 'artifacts' / 'genomics_smoke'\n"
            "input_dir = output_dir / 'input'\n"
            "generator = ROOT / 'tools' / 'generate_genomics_smoke_data.py'\n"
            "%run \"$generator\" --output-dir \"$input_dir\"\n"
            "genotype_path = input_dir / 'synthetic_genotypes.npz'\n"
            "metadata_path = input_dir / 'synthetic_populations.csv'\n"
            "genotype_path, metadata_path",
            "script = ROOT / 'experiments' / 'genomics_1kg.py'\n"
            "output_json = output_dir / 'results.json'\n"
            "output_csv = output_dir / 'metrics.csv'\n"
            "%run \"$script\" \"$genotype_path\" --orientation samples-by-snps --missing-value -9 --method pca --rank 2 --bootstrap-repeats 1 --metadata \"$metadata_path\" --output-json \"$output_json\" --output-csv \"$output_csv\"",
        ],
    )


if __name__ == "__main__":
    main()

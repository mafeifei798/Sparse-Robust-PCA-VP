# Variable Projection SPCA Toolkit

## Project Overview

This repository collects a series of prototype implementations and experiments on sparse principal component analysis (SPCA) built around the variable projection paradigm. Besides the baseline VP-SPCA solver, it includes variants that target robustness, randomized acceleration, FISTA-style momentum, and adaptive regularization. The codebase also ships with simulation scripts, analysis notebooks, and pre-rendered figures that showcase modelling skills and experimental design.

## Highlights

- **Variable-projection SPCA**: `spca_variable_projection.py` implements the VP-SPCA proximal updates together with a non-stationarity monitor.
- **Robust / randomized extensions**: `robust_spca_vp.py` separates sparse outliers, while `randomized_spca_vp.py` leverages randomized range finding for scalability.
- **Acceleration studies**: `spca_variable_projection_fista_bt.py` and `spca_variable_projection_fista_bt_torch.py` add FISTA momentum plus backtracking; the `spca_variable_projection_vs_*` scripts compare different strategies.
- **Multiscale video case study**: `spca_multiscale_video.py` synthesizes a multiscale video to visualize recovered spatial modes and temporal activations.
- **Federated & large-scale notebooks**: `fedrpca-*.ipynb` and `gene*.ipynb` document experiments on satellite, remote sensing and gene-expression datasets.
- **Ready-to-use visuals**: multiple PNG files illustrate convergence behavior, performance comparisons, and recovered structures.

## Repository Tour

- `spca_variable_projection.py`: core VP-SPCA algorithm and demo.
- `robust_spca_vp.py`: robust VP-SPCA with explicit sparse outlier estimates.
- `randomized_spca_vp.py`: randomized variable projection plus multiscale video demo.
- `spca_variable_projection_fista_bt.py` / `_torch.py`: FISTA + backtracking variants (NumPy / PyTorch backends).
- `spca_multiscale_video.py`: data generator and visualization utilities.
- `spca_video_comparison.py`: compares PCA, VP-SPCA, and randomized VP-SPCA.
- `spca_variable_projection_vs_adaptive_compare.py`, `spca_variable_projection_vs_fista_compare.py`: benchmark different algorithmic variants.
- `*.ipynb`: notebooks covering federated RPCA and gene-expression experiments.
- `*.png`: publication-ready plots and reconstructions.

## Environment Setup

Python 3.10+ is recommended. Create a virtual environment and install core dependencies:

```bash
python -m venv .venv
source .venv/bin/activate         # On Windows use .venv\Scripts\activate
pip install numpy scipy matplotlib scikit-learn torch
```

> Some notebooks additionally rely on packages such as `pandas`, `seaborn`, or `torchvision`. Install them on demand following each notebook’s instructions.

## Quick Start

1. **VP-SPCA demo**
   ```bash
   python spca_multiscale_video.py
   ```
   Produces `multiscale_spca_6_1_results.png` and `multiscale_spca_6_1_convergence.png`, illustrating recovered spatial modes and convergence diagnostics.

2. **Compare algorithmic variants**
   ```bash
   python spca_video_comparison.py
   python spca_variable_projection_vs_adaptive_compare.py
   python spca_variable_projection_vs_fista_compare.py
   ```
   These scripts save multiple comparison plots (for example `spca_video_compare.png`, `alg41_vs_adaptive_time.png`) and print timing plus reconstruction error in the console.

3. **Randomized / robust variants**
   ```bash
   python randomized_spca_vp.py
   python spca_multiscale_video.py       # includes the robust VP-SPCA call
   ```
   Observe how randomized and robust strategies behave on high-dimensional video data.

4. **Deep learning backend (optional)**
   To run the FISTA + Torch implementation on GPU or Apple MPS, install the matching PyTorch build and execute:
   ```bash
   python spca_variable_projection_vs_fista_compare.py
   ```
   The script auto-detects `cuda` and `mps` devices when available.

## Jupyter Notebooks

- `fedrpca-eurosat-3.ipynb`, `fedrpca-svhn.ipynb`: federated / distributed RPCA experiments on satellite imagery, and vision datasets.
- `gene.ipynb`, `gene large scale.ipynb`: RPCA/SPCA analysis on gene-expression matrices.
- Before committing or sharing, consider clearing notebook outputs to keep revisions focused on final figures and conclusions. Follow the dependency and data-download hints at the top of each notebook when reproducing experiments.

## Result Figures

Representative PNG files such as `robust_spca_vp_fig6.png`, and `spca_video_compare.png` are included for immediate presentation.



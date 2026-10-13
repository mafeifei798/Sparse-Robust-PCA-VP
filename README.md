# Sparse and Robust PCA with Variable Projection

## Overview

Implementations and saved results for sparse dimensionality reduction,
synthetic video decomposition, and genomic population stratification.

## Methods

- [VP-SPCA](src/rspca/vp.py): sparse loadings through variable projection.
- [Accelerated VP-SPCA](src/rspca/accelerated.py): restarted FISTA and backtracking.
- [Randomized VP-SPCA](src/rspca/randomized.py): matrix-free randomized sketches.
- [Robust VP-SPCA](src/rspca/robust.py): Huber loss and sparse outlier separation.
- [Genomics pipeline](experiments/genomics_randomized_vp_selected5.py): SNP selection and independent refitting.

Based on Erichson et al., [Sparse Principal Component Analysis via Variable Projection](https://arxiv.org/abs/1804.00341).

## Results

| Study | Main result | Numerical records |
|---|---|---|
| Corrupted synthetic data | 30.7% lower mean subspace error than default scikit-learn PCA, across 12 paired seeds | [Summary](results/synthetic/default_pca_comparison/paired_replay/summary.json), [per-seed records](results/synthetic/default_pca_comparison/paired_replay/records.json) |
| Clean synthetic video | Three recovered spatial modes and their temporal dynamics | [Results](results/video/clean_modes/three_mode_recovery/results.json) |
| Corrupted synthetic video | Single-start Robust VP-SPCA reduced mean clean reconstruction error by 22.1% versus PCA across 20 paired seeds | [Results](results/video/robust_decomposition/confirmation/results.json), [per-frame errors](results/video/robust_decomposition/confirmation/confirmation_full_frame_errors.csv) |
| 1000 Genomes | Selected 22,755 of 455,102 SNPs (5%); balanced accuracy 97.475% versus 98.175% for all-SNP PCA; median within-split modeling-runtime reduction 29.1% across 20 paired splits | [Summary](results/genomics/selected_snps/paired_splits/summary.json) |

Figures: [spatial modes and dynamics](results/video/clean_modes/three_mode_recovery/spatial_modes_and_dynamics.png),
[robust decomposition](results/video/robust_decomposition/representative_frame/robust_video_decomposition.png),
[population PC1/PC2](results/genomics/selected_snps/paired_splits/seed_2025/pc12.png),
[population PC3/PC4](results/genomics/selected_snps/paired_splits/seed_2025/pc34.png).

The synthetic PCA comparison is not sparsity-matched and does not isolate
sketching effects. The clean-video fit reached 500 iterations without strict
convergence. Robust-video nominal-budget outlier F1 was lower than PCA's, and
the combined confirmation criterion was not met. Its representative figure
uses continuation and multiple starts, not the single-start aggregate fit.

Genomics timing includes preprocessing, selection, refitting, projection, and
readouts; it excludes downloads, upstream QC/LD processing, rendering, and
separate validation. The plots show the first split. A fixed 50-step VP budget
and repeated splits of one 1,000-person cohort do not establish strict
stationarity or external-population generalization.

## Run

Python 3.11 or newer; use a full Git clone.

```bash
python -m pip install -e '.[experiments]'
python -m experiments.spca_pca_comparison --help
python -m experiments.video_clean_mode_recovery --help
python -m experiments.video_robust_decomposition --help
PYTHONPATH=src python experiments/genomics_randomized_vp_selected5.py --help
```

These commands show interfaces without running experiments. Full historical
replay requires the original source/input archives and numerical environment.
Experiment parameters remain with the results; package dependencies are in
`pyproject.toml`. Genotypes, participant-level predictions, and checkpoints
are not distributed. [MIT License](LICENSE).

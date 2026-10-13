# Sparse and Robust PCA with Variable Projection

## Overview

Sparse dimensionality reduction, synthetic video decomposition, and genomic
population stratification using variable-projection PCA methods. This repository
contains implementations and saved numerical results. [MIT License](LICENSE).

## Methods

- [VP-SPCA](src/rspca/vp.py) with sparse loadings.
- [Accelerated VP-SPCA](src/rspca/accelerated.py) with restarted FISTA and backtracking.
- [Randomized VP-SPCA](src/rspca/randomized.py) with matrix-free sketches.
- [Robust VP-SPCA](src/rspca/robust.py) with Huber loss and a sparse outlier component.
- [Genomics pipeline](experiments/genomics_randomized_vp_selected5.py) for SNP selection and independent refitting.

Based on Erichson et al.,
[Sparse Principal Component Analysis via Variable Projection](https://arxiv.org/abs/1804.00341).

## Results

Accelerated VP-SPCA reduced mean subspace error by **30.7%** versus default
scikit-learn PCA on corrupted synthetic data across 12 paired seeds.
[Summary](results/synthetic/default_pca_comparison/paired_replay/summary.json) ·
[Per-seed records](results/synthetic/default_pca_comparison/paired_replay/records.json).
This is not a sparsity-matched comparison and does not isolate sketching effects.

VP-SPCA recovered three spatial modes and their temporal dynamics in clean
synthetic video without truth-dependent amplitude refitting.
[Results](results/video/clean_modes/three_mode_recovery/results.json).
The fit reached its 500-iteration budget without strict convergence.

![Recovered spatial modes and temporal dynamics](results/video/clean_modes/three_mode_recovery/spatial_modes_and_dynamics.png)

Single-start Robust VP-SPCA reduced mean clean relative reconstruction error by
**22.1%** versus PCA across 20 paired seeds. Nominal-budget outlier F1 was lower
than PCA's, and the combined confirmatory effect criterion was not met.
[Results](results/video/robust_decomposition/confirmation/results.json) ·
[Full-frame errors](results/video/robust_decomposition/confirmation/confirmation_full_frame_errors.csv).

![Truth, corrupted input, robust reconstruction, and sparse outliers](results/video/robust_decomposition/representative_frame/robust_video_decomposition.png)

This representative frame uses continuation and multiple starts; it illustrates
the decomposition rather than the single-start aggregate comparison.

The 1000 Genomes pipeline selected **22,755 of 455,102 SNPs (5%)**, with balanced
accuracy of **97.475%** versus **98.175%** for all-SNP PCA. Across 20 paired splits,
the median within-split modeling-runtime reduction was **29.1%**.
[Summary](results/genomics/selected_snps/paired_splits/summary.json).
Timing covers preprocessing, selection, refitting, projection, and readouts,
excluding downloads, upstream QC/LD processing, rendering, and separate validation.

![Population separation on PC1 and PC2](results/genomics/selected_snps/paired_splits/seed_2025/pc12.png)

![Population separation on PC3 and PC4](results/genomics/selected_snps/paired_splits/seed_2025/pc34.png)

The plots show the first split, not aggregate scores. A fixed 50-step VP budget
and repeated splits of one 1,000-person cohort do not establish strict
stationarity or external-population generalization.

## Run

Use Python 3.11 or newer and a full Git clone:

```bash
python -m pip install -e '.[experiments]'
python -m experiments.spca_benchmark --help
python -m experiments.video_clean_mode_recovery --help
python -m experiments.video_robust_decomposition --help
PYTHONPATH=src python experiments/genomics_randomized_vp_selected5.py --help
```

`pyproject.toml` records package dependencies; saved experiment parameters are
retained with the numerical results. The commands above show the experiment
interfaces without running a benchmark. Exact replay requires the original inputs
and numerical environment. Downloaded genotypes, participant-level predictions,
and checkpoints are not distributed.

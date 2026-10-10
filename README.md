# Sparse and Robust PCA with Variable Projection

University of Macau summer research | Jun-Oct 2025

Code and saved results for sparse dimensionality reduction, synthetic video
decomposition, and genomic population stratification. The period above describes
the original project; subsequent implementation and validation are included.

## Code and results

| Task | Implementation | Saved results |
| --- | --- | --- |
| Sparse dimensionality reduction | [accelerated VP-SPCA](src/rspca/accelerated.py), [randomized solver](src/rspca/randomized.py) | [summary](results/synthetic/default_pca_comparison/paired_replay/summary.json), [per-seed records](results/synthetic/default_pca_comparison/paired_replay/records.json) |
| Clean-video mode recovery | [VP-SPCA](src/rspca/vp.py) | [results](results/video/clean_modes/historical_github_current_v2/results.json), [three-mode figure](results/video/clean_modes/historical_github_current_v2/integrity_current_vp_spca_no_refit.png) |
| Robust video decomposition | [Huber-based solver](src/rspca/robust.py) | [20-seed results](results/video/robust_decomposition/formal_confirm_v1/results.json), [figures and CSV measurements](results/video/robust_decomposition/presentation_figures) |
| Genomic population stratification | [VP-loading SNP selection and refit](experiments/genomics_randomized_vp_selected5.py) | [20-split summary](results/genomics/selected_snps/paired_20seeds/summary.json), [PC plots](results/genomics/selected_snps/paired_20seeds/seed_2025) |
| Cleaned-encoder comparison, including negative findings | [experiment](experiments/video_cleaned_encoder_v2.py) | [paired results](results/video/cleaned_encoder/formal_fresh_v1/results.json) |

## Measurement scope

- Synthetic recovery: accelerated VP-SPCA reduces subspace error by 30.7%
  against default dense scikit-learn PCA across 12 paired seeds. This is not
  a sparsity-matched comparison or an isolated effect of random sketches.
- Clean-video recovery and corrupted-video reconstruction are separate tasks.
  The clean fit recovers three native modes without truth-dependent amplitude
  refitting, but reaches its 500-iteration limit without strict convergence.
- In the 20-seed corrupted-video comparison, mean clean relative reconstruction
  error is 0.4332993 for PCA and 0.3373443 for robust VP-SPCA, a 22.1452%
  reduction. Nominal-budget outlier F1 is lower than PCA's.
- The reference genomic configuration, RVP-select5-c0.001, selects 22,755 of
  455,102 SNPs. Balanced accuracy is 97.475% versus 98.175% for all-SNP PCA;
  AMR recall is 88.5% versus 92.0%. All predeclared penalty summaries remain.
- The 29.1% genomic speedup is the median within-split workflow-time reduction
  over 20 paired splits. Timing includes preprocessing, selection, panel
  materialization, independent refitting, projection, and readouts; downloads,
  upstream QC/LD processing, rendering, and independent audits are excluded.
  A fixed 50-step VP budget does not establish strict full-data stationarity.
- Genomic splits reuse one 1,000-person cohort; they do not establish external
  population generalization. These are task-specific results, not universal
  superiority or claims of a new variable-projection PCA algorithm.

## Install and source-only checks

Python 3.11 or newer:

```bash
python -m pip install -e '.[experiments,dev]'
make test-nonfed-unit
```

Rows are samples and columns are centered features:

```python
from rspca import fit_accelerated_vp_spca

result = fit_accelerated_vp_spca(X, n_components=3, alpha=0.1)
X_hat = X @ result.B @ result.A.T
```

This repository includes selected code and saved results, not the full
replication archive. Downloaded cohorts, participant-level predictions, and
checkpoints are not included. Historical rendering and archive-dependent
validation require the original research archive. Source-only checks do not
reproduce the saved benchmarks.

## Reference and license

N. B. Erichson et al.,
[Sparse Principal Component Analysis via Variable Projection](https://arxiv.org/abs/1804.00341).

[MIT License](LICENSE). [Citation metadata](CITATION.cff).

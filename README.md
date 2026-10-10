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
| Robust video decomposition | [Huber-based solver](src/rspca/robust.py) | [Figure-6-style decomposition](results/video/robust_decomposition/figure6_style/robust_vp_spca_fig6.png), [20-seed results](results/video/robust_decomposition/formal_confirm_v1/results.json), [audit figures](results/video/robust_decomposition/presentation_figures) |
| Genomic population stratification | [VP-loading SNP selection and refit](experiments/genomics_randomized_vp_selected5.py) | [20-split summary](results/genomics/selected_snps/paired_20seeds/summary.json), [PC plots](results/genomics/selected_snps/paired_20seeds/seed_2025) |

## Robust video: Figure-6-style decomposition

![Clean truth, corrupted input, native robust reconstruction, and native outliers](results/video/robust_decomposition/figure6_style/robust_vp_spca_fig6.png)

The four panels show **truth -> corrupted input -> reconstruction -> outliers**.
This is a saved native output from the continuation/multistart variant of our
robust VP-SPCA implementation: confirmation seed 51001, predeclared frame 100.
It is not the old repository's image and was not rerun or denoised for this update.
[Figure provenance and display limits](results/video/robust_decomposition/figure6_style/provenance.json)
record the existing fit and image hashes.

Truth, input and reconstruction share one signal-range color scale; the sparse
outlier layer uses a separate signed scale. The audit overview used a shared
range of +/-12.62 set by gross corruption, which made a roughly 0.26 signal
nearly invisible. The saved four-panel view uses the clean frame's 1st/99th
percentiles for its common signal color limits. Values outside the limits
saturate in the colormap; numerical arrays are unchanged.

This illustrative frame is not the single-start method underlying the 22.1452%
aggregate reduction quoted below, nor proof of whole-sequence recovery or
superiority to the original legacy example. The existing
[20-seed results](results/video/robust_decomposition/formal_confirm_v1/results.json),
[full-frame CSV](results/video/robust_decomposition/formal_confirm_v1/confirmation_full_frame_errors.csv)
and [worst-frame audit views](results/video/robust_decomposition/presentation_figures)
remain available.

## Measurement scope

- Synthetic recovery: accelerated VP-SPCA reduces subspace error by 30.7%
  against default dense scikit-learn PCA across 12 paired seeds. This is not
  a sparsity-matched comparison or an isolated effect of random sketches.
- Clean-video recovery and corrupted-video reconstruction are separate tasks.
  The clean fit recovers three native modes without truth-dependent amplitude
  refitting, but reaches its 500-iteration limit without strict convergence.
- In the 20-seed corrupted-video comparison, mean clean relative reconstruction
  error is 0.4332993 for PCA and 0.3373443 for robust VP-SPCA, a 22.1452%
  reduction. Nominal-budget outlier F1 is lower than PCA's. The confirmatory
  effect gate failed, and continuation adds no meaningful mean reconstruction
  benefit over direct multistart. These limits remain in the saved results.
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
make test
make cli-check
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
validation require the original research archive. In particular, the exact
synthetic replay requires frozen source snapshots not included here. Tests
check source provenance hashes when historical Git objects are available and otherwise
require an explicit unverified status; they do not certify unavailable history.
The strict array-hash replay is retained separately as `make test-historical`;
it requires the original numerical environment and is excluded from portable
source-only CI. Source-only checks do not reproduce the saved benchmarks.

Historical source hashes remain unchanged. Documentation-only revisions in two
solver files are checked against authenticated blobs from the pinned commit;
executable changes are rejected. This check requires a full Git clone.

## Reference and license

N. B. Erichson et al.,
[Sparse Principal Component Analysis via Variable Projection](https://arxiv.org/abs/1804.00341).

[MIT License](LICENSE). [Citation metadata](CITATION.cff).

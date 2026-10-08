"""Memory-bounded equivalents for matched full-width genomics development.

These are not literal sklearn PCA.fit timings. The randomized PCA control
calls sklearn's pinned 1.7.0 randomized-SVD kernel on a validated linear
operator, with the same auto-transpose, oversampling and q policy. Exact PCA
uses the sample Gram matrix, never a feature-by-feature Gram matrix.
"""

from time import perf_counter

import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator
import sklearn

from .genomics_aim import SmartPCAPreprocessor
from .randomized import _randomized_range_finder_prevalidated
from .randomized_reduced import ReducedVPSPCAResult, solve_reduced_vp
from .vp import _thin_svd


def fit_block_preprocessor(raw, train, *, chunk_size=2048):
    """Fit the existing training-only scaling rule one SNP chunk at a time."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    width = raw.shape[1]
    mean, frequency, scale = [np.empty(width, dtype=float) for _ in range(3)]
    retained = np.zeros(width, dtype=bool)
    for start in range(0, width, chunk_size):
        stop = min(start + chunk_size, width)
        block = np.asarray(raw[train, start:stop], dtype=float)
        # A globally non-polymorphic block may be dropped even if a different
        # block retains SNPs; the dense helper intentionally rejects a wholly
        # empty matrix, so reproduce its elementwise formula here.
        if np.isinf(block).any():
            raise ValueError("infinite genotype")
        observed = np.isfinite(block)
        if np.any(observed & ((block < 0) | (block > 2))):
            raise ValueError("genotype dosage outside [0,2]")
        counts = observed.sum(axis=0)
        sums = np.where(observed, block, 0).sum(axis=0)
        means = np.divide(sums, counts, out=np.zeros(stop-start), where=counts > 0)
        freqs = means / 2
        scales = np.sqrt(2 * freqs * (1 - freqs))
        mask = (counts > 0) & np.isfinite(scales) & (scales > 1e-12)
        mean[start:stop], frequency[start:stop] = means, freqs
        scale[start:stop], retained[start:stop] = np.where(mask, scales, 1), mask
    if not retained.any():
        raise ValueError("no polymorphic training SNPs")
    return SmartPCAPreprocessor(mean, frequency, scale, retained)


class GenotypeOperator(LinearOperator):
    """Read-only, centered/scaled samples-by-retained-SNPs operator."""

    def __init__(self, raw, rows, preprocessing, *, chunk_size=2048):
        self.raw = raw
        self.rows = np.asarray(rows, dtype=np.int64)
        self.preprocessing = preprocessing
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1 or self.rows.ndim != 1 or not self.rows.size:
            raise ValueError("invalid chunk size or row selection")
        if len(np.unique(self.rows)) != len(self.rows) or np.any(self.rows < 0) or np.any(self.rows >= raw.shape[0]):
            raise ValueError("row indices must be unique and within the input")
        if raw.ndim != 2 or raw.shape[1] != len(preprocessing.retained_snps):
            raise ValueError("input and preprocessing shapes differ")
        self.passes = 0
        super().__init__(dtype=np.dtype("float64"), shape=(len(self.rows), int(preprocessing.retained_snps.sum())))

    def blocks(self):
        self.passes += 1
        offset = 0
        pre = self.preprocessing
        for start in range(0, self.raw.shape[1], self.chunk_size):
            stop = min(start + self.chunk_size, self.raw.shape[1])
            mask = pre.retained_snps[start:stop]
            count = int(mask.sum())
            if not count:
                continue
            block = np.asarray(self.raw[self.rows, start:stop], dtype=float)
            if np.isinf(block).any():
                raise ValueError("infinite genotype")
            finite = np.isfinite(block)
            if np.any(finite & ((block < 0) | (block > 2))):
                raise ValueError("genotype dosage outside [0,2]")
            block = np.where(finite, block, pre.mean_dosage[start:stop])
            block -= pre.mean_dosage[start:stop]
            block /= pre.scale[start:stop]
            if not mask.all():
                block = block[:, mask]
            yield offset, offset + count, block
            offset += count
        if offset != self.shape[1]:
            raise RuntimeError("retained feature offset mismatch")

    def _matmat(self, other):
        other = np.asarray(other, dtype=float)
        if other.ndim != 2 or other.shape[0] != self.shape[1]:
            raise ValueError("right-product dimensions differ")
        result = np.zeros((self.shape[0], other.shape[1]))
        for start, stop, block in self.blocks():
            result += block @ other[start:stop]
        return result

    def _rmatmat(self, other):
        other = np.asarray(other, dtype=float)
        if other.ndim != 2 or other.shape[0] != self.shape[0]:
            raise ValueError("transpose-product dimensions differ")
        result = np.empty((self.shape[1], other.shape[1]))
        for start, stop, block in self.blocks():
            result[start:stop] = block.T @ other
        return result

    def _matvec(self, other):
        return self._matmat(np.asarray(other).reshape(-1, 1)).ravel()

    def _rmatvec(self, other):
        return self._rmatmat(np.asarray(other).reshape(-1, 1)).ravel()

    def frobenius_norm(self):
        return float(np.sqrt(sum(float(np.sum(block * block)) for _, _, block in self.blocks())))


def fit_block_exact_pca(X, rank):
    """Exact rank-r PCA through XX.T; input must be training-centered."""
    if not 1 <= rank <= min(X.shape):
        raise ValueError("invalid PCA rank")
    gram = np.zeros((X.shape[0], X.shape[0]))
    for _, _, block in X.blocks():
        gram += block @ block.T
    eigenvalues, U = linalg.eigh((gram + gram.T) / 2, subset_by_index=(X.shape[0]-rank, X.shape[0]-1), check_finite=False)
    singular = np.sqrt(np.maximum(eigenvalues[::-1], 0))
    if singular[-1] <= np.finfo(float).eps * max(X.shape) * singular[0]:
        raise ValueError("numerical rank below requested PCA rank")
    B = X.T @ (U[:, ::-1] / singular)
    return B, singular


def fit_block_randomized_pca(X, rank, *, n_iter="auto", random_state=2025):
    """Pinned sklearn SVD kernel on an operator, not the public PCA.fit API."""
    if sklearn.__version__ != "1.7.0":
        raise RuntimeError("this private-kernel adapter is validated for sklearn 1.7.0 only")
    # Older supported project environments may not expose this private name.
    # Keep preprocessing/exact PCA/VP-SPCA importable there, while failing
    # explicitly if this version-pinned experimental control is requested.
    from sklearn.utils.extmath import _randomized_svd
    if not 1 <= rank <= min(X.shape):
        raise ValueError("invalid PCA rank")
    _, singular, right = _randomized_svd(X, n_components=rank, n_oversamples=10,
                                        n_iter=n_iter, random_state=random_state)
    return np.ascontiguousarray(right.T), singular


def fit_block_randomized_vp_spca(X, rank, *, penalty_fraction, max_iter=50,
                               n_power_iter=2, random_state=2025):
    """Same randomized VP objective/solver as the dense development arm."""
    if not 1 <= rank <= min(X.shape) or not np.isfinite(penalty_fraction) or penalty_fraction <= 0 or max_iter < 1:
        raise ValueError("invalid rank, penalty or iteration count")
    if not isinstance(n_power_iter, (int, np.integer)) or isinstance(n_power_iter, bool) or n_power_iter < 0:
        raise ValueError("power iteration count must be a nonnegative integer")
    begin = perf_counter()
    Q = _randomized_range_finder_prevalidated(X, rank, 10, n_power_iter, random_state)
    sketch = Q.T @ X
    _, singular, right = _thin_svd(sketch)
    scale = float(singular[0] ** 2)
    if scale <= np.finfo(float).tiny:
        raise ValueError("zero sketch")
    weights = singular**2 / scale
    V = np.ascontiguousarray(right.T)
    rho = penalty_fraction / np.sqrt(X.shape[1])
    prepared = perf_counter()
    B, A, trace, eta, converged, it, restarts = solve_reduced_vp(V, weights, rank, rho, max_iter=max_iter)
    finished = perf_counter()
    return ReducedVPSPCAResult(A=A, B=B, Q=Q, alpha=rho*scale, lipschitz_sketch=scale,
        sketch_converged=converged, sketch_relative_stationarity=eta,
        sketch_objectives_scaled=trace, n_iter=it, restarts=restarts,
        timings={"preparation_seconds": prepared-begin, "iteration_seconds": finished-prepared, "fit_seconds": finished-begin})

"""Task-owned read-only float64 caches shared fairly by PCA and VP-SPCA."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from .genomics_blocked import GenotypeOperator, fit_block_preprocessor


@dataclass(frozen=True)
class CachedGenotypes:
    train: np.memmap
    test: np.memmap
    preprocessing: object
    preparation_seconds: float
    timings: dict[str, float]
    identity: dict


@contextmanager
def standardized_genotype_cache(raw, train, test, *, scratch_root=None, chunk_size=2048):
    """Prepare once; charge the same complete preparation cost to every arm.

    Raw genotypes and the caller's indices are never mutated. The temporary
    directory is exclusively owned here and removed only after the read-only
    mappings are closed. The caller must not use returned mappings afterwards.
    """
    if np.intersect1d(train, test).size:
        raise ValueError("training and test rows must be disjoint")
    started = perf_counter()
    pre = fit_block_preprocessor(raw, train, chunk_size=chunk_size)
    statistics_done = perf_counter()
    mappings = []
    with tempfile.TemporaryDirectory(prefix="rspca-standardized-", dir=scratch_root) as owned:
        directory = Path(owned)
        identities = {}
        try:
            for label, rows in (("train", train), ("test", test)):
                op = GenotypeOperator(raw, rows, pre, chunk_size=chunk_size)
                target = directory / f"{label}.npy"
                writable = np.lib.format.open_memmap(target, mode="w+", dtype=np.float64,
                                                     shape=op.shape, fortran_order=False)
                block_digest = hashlib.sha256()
                try:
                    for start, stop, block in op.blocks():
                        writable[:, start:stop] = block
                        block_digest.update(np.asarray([start, stop], dtype="<i8").tobytes())
                        block_digest.update(np.ascontiguousarray(block, dtype="<f8").tobytes())
                    writable.flush()
                finally:
                    writable._mmap.close()
                readonly = np.load(target, mmap_mode="r", allow_pickle=False)
                if readonly.flags.writeable or not readonly.flags.c_contiguous:
                    readonly._mmap.close()
                    raise RuntimeError("cache must be read-only contiguous float64")
                mappings.append(readonly)
                identities[label] = {"shape": list(readonly.shape), "dtype": str(readonly.dtype),
                    "standardized_block_stream_sha256": block_digest.hexdigest(),
                    "block_digest_contract": "for each increasing retained-feature block: little-endian int64 start/stop then little-endian float64 C-order block bytes",
                    "chunk_size": chunk_size, "bytes": int(readonly.nbytes)}
            finished = perf_counter()
            yield CachedGenotypes(mappings[0], mappings[1], pre, finished-started,
                                  {"training_statistics_seconds": statistics_done-started,
                                   "cache_materialization_seconds": finished-statistics_done}, identities)
        finally:
            for mapping in mappings:
                mapping._mmap.close()

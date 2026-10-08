"""A cache may save passes, not alter preprocessing or give one arm free work."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from sklearn.decomposition import PCA

from rspca.genomics_aim import fit_smartpca_preprocessor, transform_smartpca
from rspca.genomics_cached import standardized_genotype_cache
from rspca.randomized_reduced_fast import fit_fast_randomized_vp_spca


class CachedGenomicsTests(unittest.TestCase):
    def test_cache_matches_dense_and_is_readonly_with_scoped_cleanup(self):
        rng = np.random.default_rng(34)
        raw = rng.integers(0, 3, (90, 137)).astype(np.float32)
        raw[:, 0] = 0
        raw[3:12, 8] = np.nan
        train, test = np.arange(70), np.arange(70, 90)
        original = raw.copy()
        pre = fit_smartpca_preprocessor(raw[train])
        with tempfile.TemporaryDirectory() as parent:
            with standardized_genotype_cache(raw, train, test, scratch_root=parent, chunk_size=19) as cache:
                np.testing.assert_array_equal(cache.train, transform_smartpca(raw[train], pre))
                np.testing.assert_array_equal(cache.test, transform_smartpca(raw[test], pre))
                self.assertFalse(cache.train.flags.writeable)
                self.assertEqual(cache.train.dtype, np.dtype("float64"))
                self.assertTrue(cache.train.flags.c_contiguous)
                self.assertGreater(cache.preparation_seconds, 0)
                # Actual public PCA with untouched defaults can consume it.
                pca = PCA(n_components=4, random_state=2025).fit(cache.train)
                reference = PCA(n_components=4, random_state=2025).fit(np.asarray(cache.train).copy())
                np.testing.assert_allclose(pca.components_, reference.components_, atol=1e-12)
                model = fit_fast_randomized_vp_spca(cache.train, 4, max_iter=10)
                self.assertTrue(np.isfinite(model.B).all())
                np.testing.assert_array_equal(cache.train, transform_smartpca(raw[train], pre))
            self.assertEqual(list(Path(parent).iterdir()), [])
        np.testing.assert_array_equal(raw, original)

    def test_cleanup_after_caller_failure(self):
        raw = np.tile(np.arange(3), (20, 10)).astype(float)
        with tempfile.TemporaryDirectory() as parent:
            with self.assertRaisesRegex(RuntimeError, "caller"):
                with standardized_genotype_cache(raw, np.arange(15), np.arange(15, 20), scratch_root=parent):
                    raise RuntimeError("caller failed")
            self.assertEqual(list(Path(parent).iterdir()), [])

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            with standardized_genotype_cache(np.ones((5, 7)), [0, 1], [1, 2]):
                pass


if __name__ == "__main__":
    unittest.main()

"""Blocked algebra must agree with dense preprocessing and fitted methods."""

import unittest

import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA
import sklearn

from rspca.genomics_aim import fit_smartpca_preprocessor, transform_smartpca
from rspca.genomics_blocked import (GenotypeOperator, fit_block_preprocessor,
    fit_block_exact_pca, fit_block_randomized_pca, fit_block_randomized_vp_spca)
from rspca.randomized import _full_diagnostics
from rspca.randomized_reduced import fit_reduced_randomized_vp_spca


class BlockedGenomicsTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(41)
        self.G = self.rng.integers(0, 3, (55, 93)).astype(float)
        self.G[0:10, 1] = np.nan
        self.G[:, 0] = np.nan
        self.G[:, 2] = 0
        self.G[:, 3] = 2
        self.train = np.arange(40)
        self.test = np.arange(40, 55)
        self.pre = fit_smartpca_preprocessor(self.G[self.train])
        self.X = transform_smartpca(self.G[self.train], self.pre)

    def test_preprocessing_and_matrix_products(self):
        for chunk in (1, 7, 35, 100):
            pre = fit_block_preprocessor(self.G, self.train, chunk_size=chunk)
            for field in ("mean_dosage", "allele_frequency", "scale", "retained_snps"):
                np.testing.assert_array_equal(getattr(pre, field), getattr(self.pre, field))
            op = GenotypeOperator(self.G, self.train, pre, chunk_size=chunk)
            B = self.rng.normal(size=(self.X.shape[1], 5))
            Q = self.rng.normal(size=(self.X.shape[0], 5))
            np.testing.assert_allclose(op @ B, self.X @ B, atol=1e-13)
            np.testing.assert_allclose(op.T @ Q, self.X.T @ Q, atol=1e-13)
            np.testing.assert_allclose(Q.T @ op, Q.T @ self.X, atol=1e-13)
            np.testing.assert_allclose(op @ B[:, 0], self.X @ B[:, 0], atol=1e-13)
            self.assertAlmostEqual(op.frobenius_norm(), linalg.norm(self.X), places=11)

    def test_test_rows_use_frozen_training_statistics(self):
        op = GenotypeOperator(self.G, self.test, self.pre, chunk_size=13)
        dense = transform_smartpca(self.G[self.test], self.pre)
        B = self.rng.normal(size=(dense.shape[1], 3))
        np.testing.assert_allclose(op @ B, dense @ B, atol=1e-13)

    def test_exact_and_sklearn_randomized_pca_equivalence(self):
        op = GenotypeOperator(self.G, self.train, self.pre, chunk_size=13)
        for rank in (4, 6, 8):
            B, s = fit_block_exact_pca(op, rank)
            reference = PCA(n_components=rank, svd_solver="full").fit(self.X)
            self.assertLess(linalg.norm(linalg.subspace_angles(B, reference.components_.T)), 1e-10)
            np.testing.assert_allclose(s, reference.singular_values_, rtol=1e-12)
            if sklearn.__version__ != "1.7.0":
                with self.assertRaisesRegex(RuntimeError, "sklearn 1.7.0 only"):
                    fit_block_randomized_pca(op, rank)
                continue
            for iterations in (2, 4, "auto"):
                B, s = fit_block_randomized_pca(op, rank, n_iter=iterations, random_state=31)
                ref = PCA(n_components=rank, svd_solver="randomized", iterated_power=iterations, random_state=31).fit(self.X)
                signs = np.sign(np.sum(B * ref.components_.T, axis=0))
                np.testing.assert_allclose(B * signs, ref.components_.T, atol=3e-10, rtol=3e-9)
                np.testing.assert_allclose(s, ref.singular_values_, rtol=1e-11)

    def test_vp_solution_and_full_diagnostics_match_dense(self):
        op = GenotypeOperator(self.G, self.train, self.pre, chunk_size=13)
        for rank in (4, 6, 8):
            dense = fit_reduced_randomized_vp_spca(self.X, rank, penalty_fraction=.001, max_iter=50)
            blocked = fit_block_randomized_vp_spca(op, rank, penalty_fraction=.001, max_iter=50)
            signs = np.sign(np.sum(blocked.B * dense.B, axis=0))
            np.testing.assert_allclose(blocked.B * signs, dense.B, atol=3e-9, rtol=3e-8)
            np.testing.assert_allclose(blocked.sketch_objectives_scaled, dense.sketch_objectives_scaled, atol=3e-10)
            step = 1/linalg.svdvals(self.X)[0]**2
            norm = linalg.norm(self.X)
            d = _full_diagnostics(self.X, dense.A, dense.B, dense.alpha, step, norm)
            b = _full_diagnostics(op, dense.A, dense.B, dense.alpha, step, norm)
            np.testing.assert_allclose(b[0], d[0], atol=1e-12)
            np.testing.assert_allclose(b[1:], d[1:], rtol=1e-11, atol=1e-11)

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            GenotypeOperator(self.G, [0, 0], self.pre)
        with self.assertRaises(ValueError):
            fit_block_preprocessor(np.zeros((5, 7)), np.arange(4))
        with self.assertRaises(ValueError):
            fit_block_preprocessor(self.G, self.train, chunk_size=0)

    def test_q4_vp_matches_dense_and_pass_accounting(self):
        op = GenotypeOperator(self.G, self.train, self.pre, chunk_size=13)
        dense = fit_reduced_randomized_vp_spca(self.X, 8, penalty_fraction=.001,
                                             max_iter=50, n_power_iter=4)
        blocked = fit_block_randomized_vp_spca(op, 8, penalty_fraction=.001,
                                             max_iter=50, n_power_iter=4)
        self.assertEqual(op.passes, 10)
        signs = np.sign(np.sum(blocked.B * dense.B, axis=0))
        np.testing.assert_allclose(blocked.B * signs, dense.B, atol=3e-9, rtol=3e-8)
        np.testing.assert_allclose(blocked.sketch_objectives_scaled,
                                   dense.sketch_objectives_scaled, atol=3e-10)
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                fit_block_randomized_vp_spca(op, 8, penalty_fraction=.001,
                                             n_power_iter=invalid)


if __name__ == "__main__":
    unittest.main()

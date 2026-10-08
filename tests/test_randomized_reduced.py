"""Algebra and failure-boundary tests for the isolated development solver."""

import unittest

import numpy as np
from scipy import linalg

from rspca.randomized_reduced import (
    fit_reduced_randomized_vp_spca,
    reduced_value_polar,
    solve_reduced_vp,
)
from rspca.vp import procrustes_factor, smooth_value, vp_gradient


class ReducedVPTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(64)
        self.Z = self.rng.normal(size=(9, 23))
        _, s, vt = linalg.svd(self.Z, full_matrices=False)
        self.V, self.L, self.w = vt.T, s[0]**2, s**2 / s[0]**2
        self.B = self.rng.normal(size=(23, 3))

    def test_value_factor_gradient_equal_direct(self):
        coordinates = self.V.T @ self.B
        value, polar = reduced_value_polar(self.w, coordinates)
        A = self.V @ polar
        direct = procrustes_factor(self.Z, self.B)
        np.testing.assert_allclose(A, direct, atol=2e-14)
        self.assertAlmostEqual(value + 0.5 * np.sum(self.Z**2) / self.L,
                               smooth_value(self.Z, direct, self.B) / self.L, places=12)
        gradient = self.V @ (self.w[:, None] * (coordinates - polar))
        np.testing.assert_allclose(gradient, vp_gradient(self.Z, direct, self.B) / self.L, atol=2e-14)

    def test_finite_difference(self):
        W = self.V.T @ self.B
        _, P = reduced_value_polar(self.w, W)
        direction = self.rng.normal(size=self.B.shape)
        grad = self.V @ (self.w[:, None] * (W - P))
        eps = 1e-5
        plus = reduced_value_polar(self.w, self.V.T @ (self.B + eps * direction))[0]
        minus = reduced_value_polar(self.w, self.V.T @ (self.B - eps * direction))[0]
        self.assertAlmostEqual((plus-minus)/(2*eps), np.sum(grad*direction), places=8)

    def test_rank_deficient_procrustes_remains_optimal(self):
        for B in (np.zeros_like(self.B), np.tile(self.B[:, :1], (1, 3))):
            W = self.V.T @ B
            value, P = reduced_value_polar(self.w, W)
            A = self.V @ P
            np.testing.assert_allclose(A.T @ A, np.eye(3), atol=2e-14)
            self.assertAlmostEqual(value + 0.5*np.sum(self.Z**2)/self.L,
                                   smooth_value(self.Z, A, B)/self.L, places=12)

    def test_descent_and_original_coordinate_sparsity(self):
        B, A, trace, eta, converged, it, restarts = solve_reduced_vp(
            self.V, self.w, 3, 0.035, max_iter=300, check_every=10)
        self.assertTrue(np.all(np.diff(trace) <= 1e-10))
        self.assertGreater(np.count_nonzero(B == 0), 0)
        self.assertGreater(linalg.norm(B - self.V @ (self.V.T @ B)), 1e-4)
        np.testing.assert_allclose(A.T @ A, np.eye(3), atol=2e-14)
        direct_gradient = vp_gradient(self.Z, procrustes_factor(self.Z, B), B) / self.L
        shifted = B - direct_gradient
        prox = np.sign(shifted) * np.maximum(np.abs(shifted)-0.035, 0)
        self.assertAlmostEqual(eta, linalg.norm(B-prox)/(0.035*np.sqrt(B.size)), places=10)

    def test_one_step_equals_direct_vp(self):
        rho = .01
        B0 = self.V[:, :3]
        A0 = procrustes_factor(self.Z, B0)
        shifted = B0 - vp_gradient(self.Z, A0, B0)/self.L
        expected = np.sign(shifted) * np.maximum(np.abs(shifted)-rho, 0)
        observed = solve_reduced_vp(self.V, self.w, 3, rho, max_iter=1)[0]
        np.testing.assert_allclose(observed, expected, atol=2e-14)

    def test_reproducible_and_no_dense_disguise(self):
        X = self.rng.normal(size=(30, 70))
        X -= X.mean(axis=0)
        a = fit_reduced_randomized_vp_spca(X, 3, max_iter=30)
        b = fit_reduced_randomized_vp_spca(X, 3, max_iter=30)
        np.testing.assert_array_equal(a.B, b.B)
        self.assertGreater(a.alpha, 0)
        self.assertFalse(hasattr(a, "full_converged"))
        with self.assertRaises(ValueError):
            fit_reduced_randomized_vp_spca(X, 3, penalty_fraction=0)
        with self.assertRaises(ValueError):
            fit_reduced_randomized_vp_spca(np.zeros_like(X), 3)


if __name__ == "__main__":
    unittest.main()

"""Same-objective and numerical-guard tests; timing is not a unit-test gate."""

import unittest

import numpy as np
from scipy import linalg

from rspca.randomized import _randomized_range_finder_prevalidated
from rspca.randomized_reduced import fit_reduced_randomized_vp_spca, solve_reduced_vp
from rspca.randomized_reduced_fast import (checked_range_finder, checked_sketch_factorization,
    fit_fast_randomized_vp_spca, solve_reduced_vp_buffered)


class FastReducedVPTests(unittest.TestCase):
    def test_buffered_steps_match_reference_including_restarts(self):
        rng = np.random.default_rng(194)
        z = rng.normal(size=(15, 120))
        _, s, vt = linalg.svd(z, full_matrices=False)
        V, weights = np.ascontiguousarray(vt.T), s**2/s[0]**2
        for rho in (.0001, .01, .2):
            for steps in (1, 7, 50, 200):
                for acceleration in (False, True):
                    kwargs = dict(max_iter=steps, acceleration=acceleration)
                    reference = solve_reduced_vp(V, weights, 4, rho, **kwargs)
                    fast = solve_reduced_vp_buffered(V, weights, 4, rho, **kwargs)
                    for i in (0, 1, 2):
                        np.testing.assert_allclose(fast[i], reference[i], atol=1e-11, rtol=1e-10)
                    self.assertAlmostEqual(fast[3], reference[3], places=10)
                    self.assertEqual(fast[4:], reference[4:])
                    np.testing.assert_array_equal(fast[0] != 0, reference[0] != 0)

    def test_checked_range_subspace_and_orthogonality(self):
        rng = np.random.default_rng(53)
        for shape in ((43, 97), (97, 43)):
            X = rng.normal(size=shape)
            for power in (0, 2, 4):
                q, diagnostic = checked_range_finder(X, 4, 10, power, 21)
                reference = _randomized_range_finder_prevalidated(X, 4, 10, power, 21)
                self.assertLess(linalg.norm(linalg.subspace_angles(q, reference)), 1e-10)
                np.testing.assert_allclose(q.T @ q, np.eye(14), atol=2e-14)
                self.assertEqual(diagnostic["qr_calls"], 2*power+1)
        X = rng.normal(size=(50, 2)) @ rng.normal(size=(2, 80))
        q, diagnostic = checked_range_finder(X, 4, 10, 4, 21)
        self.assertGreater(diagnostic["qr_repairs"], 0)
        np.testing.assert_allclose(q.T @ q, np.eye(14), atol=2e-14)

    def test_small_gram_factorization_and_ill_conditioned_fallback(self):
        rng = np.random.default_rng(77)
        z = rng.normal(size=(13, 151))
        for source, expected in ((z, "checked_small_row_gram"),
                                 (np.diag(np.geomspace(1, 1e-12, 13)) @ z, "original_svd_fallback"),
                                 (np.zeros_like(z), "original_svd_fallback")):
            s, V, diagnostic = checked_sketch_factorization(source)
            _, reference_s, reference_vt = linalg.svd(source, full_matrices=False)
            self.assertEqual(diagnostic["factorization"], expected)
            np.testing.assert_allclose(s, reference_s, atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose((V*s) @ V.T, (reference_vt.T*reference_s) @ reference_vt,
                                       atol=1e-11, rtol=1e-9)
            np.testing.assert_allclose(V.T @ V, np.eye(13), atol=1e-11)

    def test_fit_equivalence_reproducibility_and_input_is_readonly(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=(90, 180))
        X -= X.mean(axis=0)
        X.setflags(write=False)
        for rank in (4, 6, 8):
            for penalty in (.0002, .001, .005):
                kwargs = dict(penalty_fraction=penalty, n_power_iter=4, max_iter=50,
                              random_state=2025)
                reference = fit_reduced_randomized_vp_spca(X, rank, **kwargs)
                fast = fit_fast_randomized_vp_spca(X, rank, **kwargs)
                signs = np.sign(np.sum(reference.B*fast.B, axis=0))
                np.testing.assert_allclose(fast.B*signs, reference.B, atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(fast.A*signs, reference.A, atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(fast.sketch_objectives_scaled,
                                           reference.sketch_objectives_scaled, atol=1e-9, rtol=1e-9)
                np.testing.assert_array_equal(fast.B != 0, reference.B != 0)
                self.assertAlmostEqual(fast.alpha/reference.alpha, 1, places=10)
                np.testing.assert_allclose(fast.sketch_relative_stationarity,
                    reference.sketch_relative_stationarity, atol=1e-8, rtol=1e-6)
                self.assertEqual(fast.n_iter, reference.n_iter)
                self.assertEqual(fast.sketch_converged, reference.sketch_converged)
        first = fit_fast_randomized_vp_spca(X, 4, n_power_iter=4, max_iter=30)
        second = fit_fast_randomized_vp_spca(X, 4, n_power_iter=4, max_iter=30)
        np.testing.assert_array_equal(first.B, second.B)

    def test_invalid_parameters_and_zero_input_fail(self):
        X = np.ones((20, 30))
        for kwargs in ({"penalty_fraction": 0}, {"n_power_iter": -1},
                       {"oversample": True}, {"max_iter": 0}):
            with self.assertRaises((TypeError, ValueError)):
                fit_fast_randomized_vp_spca(X, 3, **kwargs)
        with self.assertRaises(ValueError):
            fit_fast_randomized_vp_spca(np.zeros_like(X), 3)


if __name__ == "__main__":
    unittest.main()

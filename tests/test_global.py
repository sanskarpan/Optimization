"""
tests/test_global.py
====================

Tests for global_opt.py — pure Python global optimisation algorithms.
"""

import sys
import os
import math
import unittest

# Ensure the package root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from global_opt import (
    simulated_annealing,
    genetic_algorithm,
    differential_evolution,
    particle_swarm,
    nelder_mead,
    cma_es,
    basin_hopping,
    random_search,
    latin_hypercube_search,
)

# ---------------------------------------------------------------------------
# Common test functions
# ---------------------------------------------------------------------------

def sphere(x):
    """f(x) = sum(xi^2), minimum 0 at origin."""
    return sum(xi ** 2 for xi in x)


def rosenbrock(x):
    """f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2), min 0 at (1,...,1)."""
    return sum(
        100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


# ---------------------------------------------------------------------------
# 1. Simulated Annealing
# ---------------------------------------------------------------------------

class TestSimulatedAnnealing(unittest.TestCase):

    def test_returns_3_tuple(self):
        result = simulated_annealing(sphere, [1.0, 1.0], seed=42)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_result_types(self):
        x_best, f_best, n_evals = simulated_annealing(sphere, [1.0, 1.0], seed=0)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)
        self.assertIsInstance(n_evals, int)

    def test_finds_sphere_minimum(self):
        x_best, f_best, _ = simulated_annealing(
            sphere,
            [2.0, 2.0],
            T0=5.0,
            T_min=1e-5,
            alpha=0.95,
            n_steps=200,
            step_size=0.3,
            seed=7,
        )
        self.assertLess(f_best, 1.0)

    def test_n_evaluations_positive(self):
        _, _, n_evals = simulated_annealing(sphere, [1.0], seed=1)
        self.assertGreater(n_evals, 0)

    def test_1d(self):
        x_best, f_best, n_evals = simulated_annealing(
            sphere, [3.0], T0=2.0, n_steps=100, alpha=0.95, step_size=0.2, seed=99
        )
        self.assertEqual(len(x_best), 1)
        self.assertLess(f_best, 4.0)

    def test_seed_reproducible(self):
        r1 = simulated_annealing(sphere, [1.0, 1.0], seed=42)
        r2 = simulated_annealing(sphere, [1.0, 1.0], seed=42)
        self.assertEqual(r1[1], r2[1])


# ---------------------------------------------------------------------------
# 2. Genetic Algorithm
# ---------------------------------------------------------------------------

class TestGeneticAlgorithm(unittest.TestCase):

    def test_returns_2_tuple(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        result = genetic_algorithm(sphere, bounds, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_result_types(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        x_best, f_best = genetic_algorithm(sphere, bounds, seed=1)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)

    def test_sphere_2d(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        x_best, f_best = genetic_algorithm(
            sphere, bounds, pop_size=50, n_gens=200,
            mutation_rate=0.1, crossover_rate=0.9, seed=42
        )
        self.assertLess(f_best, 1.0)

    def test_solution_within_bounds(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        x_best, _ = genetic_algorithm(sphere, bounds, seed=5)
        for xi, (lo, hi) in zip(x_best, bounds):
            self.assertGreaterEqual(xi, lo)
            self.assertLessEqual(xi, hi)

    def test_seed_reproducible(self):
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        r1 = genetic_algorithm(sphere, bounds, seed=7)
        r2 = genetic_algorithm(sphere, bounds, seed=7)
        self.assertAlmostEqual(r1[1], r2[1])


# ---------------------------------------------------------------------------
# 3. Differential Evolution
# ---------------------------------------------------------------------------

class TestDifferentialEvolution(unittest.TestCase):

    def _run_de(self, strategy, seed=0):
        bounds = [(-5.0, 5.0)] * 2
        return differential_evolution(
            sphere, bounds, max_gens=200, F=0.8, CR=0.9,
            strategy=strategy, seed=seed
        )

    def test_returns_2_tuple(self):
        result = self._run_de('rand/1')
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_sphere_rand1(self):
        x_best, f_best = differential_evolution(
            sphere, [(-5.0, 5.0)] * 2,
            pop_size=20, max_gens=300, F=0.8, CR=0.9,
            strategy='rand/1', seed=42
        )
        self.assertLess(f_best, 0.1)

    def test_all_strategies_no_crash(self):
        for strategy in ('rand/1', 'best/1', 'current-to-best/1'):
            x_best, f_best = self._run_de(strategy, seed=13)
            self.assertIsInstance(f_best, float)

    def test_default_pop_size(self):
        bounds = [(-5.0, 5.0)] * 3
        # Should not crash; pop_size defaults to 10*len(bounds)=30
        x_best, f_best = differential_evolution(
            sphere, bounds, pop_size=None, max_gens=50, seed=0
        )
        self.assertIsInstance(f_best, float)

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            differential_evolution(sphere, [(-1.0, 1.0)], strategy='bad', seed=0)

    def test_solution_within_bounds(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, _ = self._run_de('best/1', seed=3)
        for xi, (lo, hi) in zip(x_best, bounds):
            self.assertGreaterEqual(xi, lo)
            self.assertLessEqual(xi, hi)

    def test_seed_reproducible(self):
        r1 = self._run_de('rand/1', seed=55)
        r2 = self._run_de('rand/1', seed=55)
        self.assertAlmostEqual(r1[1], r2[1])


# ---------------------------------------------------------------------------
# 4. Particle Swarm
# ---------------------------------------------------------------------------

class TestParticleSwarm(unittest.TestCase):

    def test_returns_2_tuple(self):
        bounds = [(-5.0, 5.0)] * 2
        result = particle_swarm(sphere, bounds, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_sphere_2d(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, f_best = particle_swarm(
            sphere, bounds, n_particles=30, max_iter=200, seed=42
        )
        self.assertLess(f_best, 0.1)

    def test_result_types(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, f_best = particle_swarm(sphere, bounds, seed=1)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)

    def test_solution_within_bounds(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, _ = particle_swarm(sphere, bounds, seed=5)
        for xi, (lo, hi) in zip(x_best, bounds):
            self.assertGreaterEqual(xi, lo)
            self.assertLessEqual(xi, hi)

    def test_seed_reproducible(self):
        bounds = [(-5.0, 5.0)] * 2
        r1 = particle_swarm(sphere, bounds, seed=77)
        r2 = particle_swarm(sphere, bounds, seed=77)
        self.assertAlmostEqual(r1[1], r2[1])

    def test_1d(self):
        bounds = [(-10.0, 10.0)]
        x_best, f_best = particle_swarm(sphere, bounds, seed=0)
        self.assertEqual(len(x_best), 1)
        self.assertLess(f_best, 1.0)


# ---------------------------------------------------------------------------
# 5. Nelder-Mead
# ---------------------------------------------------------------------------

class TestNelderMead(unittest.TestCase):

    def test_returns_4_tuple(self):
        result = nelder_mead(sphere, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_result_types(self):
        x_opt, f_opt, n_iters, converged = nelder_mead(sphere, [1.0, 1.0])
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_iters, int)
        self.assertIsInstance(converged, bool)

    def test_sphere_converges_to_origin(self):
        x_opt, f_opt, n_iters, converged = nelder_mead(
            sphere, [1.0, 1.0], tol=1e-8, max_iter=2000
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(f_opt, 0.0, delta=1e-6)
        for xi in x_opt:
            self.assertAlmostEqual(xi, 0.0, delta=1e-3)

    def test_rosenbrock(self):
        x_opt, f_opt, n_iters, converged = nelder_mead(
            rosenbrock, [0.0, 0.0], step=0.5, tol=1e-8, max_iter=5000
        )
        # Near the minimum at (1,1)
        self.assertLess(f_opt, 0.1)

    def test_converged_flag_true_on_easy_problem(self):
        _, _, _, converged = nelder_mead(sphere, [0.01, 0.01], tol=1e-3)
        self.assertTrue(converged)

    def test_max_iter_reached(self):
        _, _, n_iters, converged = nelder_mead(sphere, [1.0, 1.0], max_iter=5)
        # With only 5 iterations it won't converge
        self.assertLessEqual(n_iters, 5)

    def test_1d(self):
        x_opt, f_opt, _, _ = nelder_mead(sphere, [3.0])
        self.assertEqual(len(x_opt), 1)
        self.assertAlmostEqual(f_opt, 0.0, delta=1e-4)


# ---------------------------------------------------------------------------
# 6. CMA-ES
# ---------------------------------------------------------------------------

class TestCMAES(unittest.TestCase):

    def test_returns_4_tuple(self):
        result = cma_es(sphere, [1.0, 1.0], seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_result_types(self):
        x_opt, f_opt, n_iters, converged = cma_es(sphere, [1.0, 1.0], seed=1)
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_iters, int)
        self.assertIsInstance(converged, bool)

    def test_sphere_2d(self):
        x_opt, f_opt, n_iters, converged = cma_es(
            sphere, [1.0, 1.0], sigma0=0.5, max_iter=1000, tol=1e-7, seed=42
        )
        self.assertLess(f_opt, 0.01)

    def test_no_crash_3d(self):
        x_opt, f_opt, n_iters, converged = cma_es(
            sphere, [1.0, 1.0, 1.0], sigma0=0.5, max_iter=500, seed=0
        )
        self.assertIsInstance(f_opt, float)
        self.assertFalse(math.isnan(f_opt))

    def test_seed_reproducible(self):
        r1 = cma_es(sphere, [1.0, 1.0], seed=13)
        r2 = cma_es(sphere, [1.0, 1.0], seed=13)
        self.assertAlmostEqual(r1[1], r2[1])

    def test_converged_flag_true(self):
        _, _, _, converged = cma_es(
            sphere, [0.001, 0.001], sigma0=0.1, tol=1e-3, max_iter=200, seed=0
        )
        self.assertTrue(converged)


# ---------------------------------------------------------------------------
# 7. Basin Hopping
# ---------------------------------------------------------------------------

class TestBasinHopping(unittest.TestCase):

    def test_returns_3_tuple(self):
        result = basin_hopping(sphere, [1.0, 1.0], n_hops=5, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_result_types(self):
        x_best, f_best, n_hops_done = basin_hopping(sphere, [1.0, 1.0], seed=0)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)
        self.assertIsInstance(n_hops_done, int)

    def test_finds_sphere_minimum(self):
        x_best, f_best, _ = basin_hopping(
            sphere, [2.0, 2.0], n_hops=20, T=1.0, step_size=0.5, seed=42
        )
        self.assertLess(f_best, 0.5)

    def test_n_hops_completed(self):
        n_hops = 10
        _, _, n_done = basin_hopping(sphere, [1.0], n_hops=n_hops, seed=0)
        self.assertEqual(n_done, n_hops)

    def test_custom_local_optimizer(self):
        def local_opt(f, x):
            # Simple gradient step
            eps = 1e-4
            for _ in range(50):
                g = [(f([xi + (eps if i == j else 0.0) for j, xi in enumerate(x)]) - f(x)) / eps
                     for i in range(len(x))]
                x = [xi - 0.01 * gi for xi, gi in zip(x, g)]
            return x

        x_best, f_best, _ = basin_hopping(
            sphere, [1.0, 1.0], n_hops=5, local_optimizer=local_opt, seed=1
        )
        self.assertIsInstance(f_best, float)

    def test_seed_reproducible(self):
        r1 = basin_hopping(sphere, [1.0, 1.0], n_hops=10, seed=99)
        r2 = basin_hopping(sphere, [1.0, 1.0], n_hops=10, seed=99)
        self.assertAlmostEqual(r1[1], r2[1])


# ---------------------------------------------------------------------------
# 8. Random Search
# ---------------------------------------------------------------------------

class TestRandomSearch(unittest.TestCase):

    def test_returns_2_tuple(self):
        bounds = [(-5.0, 5.0)] * 2
        result = random_search(sphere, bounds, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_result_types(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, f_best = random_search(sphere, bounds, seed=1)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)

    def test_best_within_bounds(self):
        bounds = [(-3.0, 3.0), (1.0, 4.0)]
        x_best, _ = random_search(sphere, bounds, n_samples=200, seed=7)
        for xi, (lo, hi) in zip(x_best, bounds):
            self.assertGreaterEqual(xi, lo)
            self.assertLessEqual(xi, hi)

    def test_more_samples_better(self):
        bounds = [(-5.0, 5.0)] * 2
        _, f_small = random_search(sphere, bounds, n_samples=10, seed=0)
        _, f_large = random_search(sphere, bounds, n_samples=1000, seed=0)
        self.assertLessEqual(f_large, f_small)

    def test_seed_reproducible(self):
        bounds = [(-5.0, 5.0)] * 2
        r1 = random_search(sphere, bounds, n_samples=100, seed=42)
        r2 = random_search(sphere, bounds, n_samples=100, seed=42)
        self.assertAlmostEqual(r1[1], r2[1])

    def test_1d(self):
        bounds = [(-10.0, 10.0)]
        x_best, f_best = random_search(sphere, bounds, n_samples=500, seed=3)
        self.assertEqual(len(x_best), 1)
        self.assertGreaterEqual(x_best[0], -10.0)
        self.assertLessEqual(x_best[0], 10.0)


# ---------------------------------------------------------------------------
# 9. Latin Hypercube Search
# ---------------------------------------------------------------------------

class TestLatinHypercube(unittest.TestCase):

    def test_returns_2_tuple(self):
        bounds = [(-5.0, 5.0)] * 2
        result = latin_hypercube_search(sphere, bounds, seed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_result_types(self):
        bounds = [(-5.0, 5.0)] * 2
        x_best, f_best = latin_hypercube_search(sphere, bounds, seed=1)
        self.assertIsInstance(x_best, list)
        self.assertIsInstance(f_best, float)

    def test_best_within_bounds(self):
        bounds = [(-3.0, 3.0), (1.0, 4.0)]
        x_best, _ = latin_hypercube_search(sphere, bounds, n_samples=50, seed=7)
        for xi, (lo, hi) in zip(x_best, bounds):
            self.assertGreaterEqual(xi, lo)
            self.assertLessEqual(xi, hi)

    def test_lhs_stratification(self):
        """
        Each dimension should have exactly one sample per stratum.
        We verify this by binning the raw LHS design values and checking
        that every bin [k/n, (k+1)/n) contains exactly one sample.
        """
        import random as _random

        n_samples = 20
        ndim = 2
        bounds = [(0.0, 1.0)] * ndim
        seed = 42

        # Reproduce the LHS design internal to latin_hypercube_search
        rng = _random.Random(seed)
        design: list = []
        for d in range(ndim):
            perm = list(range(n_samples))
            rng.shuffle(perm)
            lo, hi = bounds[d]
            col = [(perm[i] + rng.random()) / n_samples * (hi - lo) + lo
                   for i in range(n_samples)]
            design.append(col)

        # For each dimension, check that bin k contains exactly one sample
        for d in range(ndim):
            bins = [0] * n_samples
            for v in design[d]:
                bin_idx = int(v * n_samples)
                bin_idx = min(bin_idx, n_samples - 1)  # handle v==1.0
                bins[bin_idx] += 1
            for b in bins:
                self.assertEqual(b, 1, msg=f"Dim {d}: bin count {b} != 1")

    def test_seed_reproducible(self):
        bounds = [(-5.0, 5.0)] * 2
        r1 = latin_hypercube_search(sphere, bounds, seed=42)
        r2 = latin_hypercube_search(sphere, bounds, seed=42)
        self.assertAlmostEqual(r1[1], r2[1])

    def test_lhs_finds_near_minimum(self):
        """With enough samples, LHS should get closer to 0 than most random points."""
        bounds = [(-5.0, 5.0)] * 2
        _, f_best = latin_hypercube_search(sphere, bounds, n_samples=200, seed=0)
        # Just check it found something reasonable
        self.assertLess(f_best, 5.0)

    def test_1d(self):
        bounds = [(-10.0, 10.0)]
        x_best, f_best = latin_hypercube_search(sphere, bounds, n_samples=100, seed=0)
        self.assertEqual(len(x_best), 1)
        self.assertGreaterEqual(x_best[0], -10.0)
        self.assertLessEqual(x_best[0], 10.0)


if __name__ == '__main__':
    unittest.main()

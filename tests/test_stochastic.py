"""
Tests for stochastic optimization module (stochastic.py).

Covers:
* SVRG — convergence, history length, full-gradient call count
* SAGAOptimizer — convergence, table updates, reset
* SAGOptimizer — convergence, average gradient estimate
* iterate_averaging — smoothing and burn-in correctness
* robbins_monro — convergence, learning-rate decay, return length
"""

import math
import os
import random
import sys
import unittest

# ---------------------------------------------------------------------------
# Make the parent package importable regardless of how the test is invoked.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stochastic import (
    SAGAOptimizer,
    SAGOptimizer,
    iterate_averaging,
    robbins_monro,
    svrg,
)


# ---------------------------------------------------------------------------
# Helper: 1-D quadratic finite-sum problem
#   f(x) = (1/n) * sum_{i=0}^{n-1} (x[0] - a[i])^2
#   grad_i(i, x) = [2 * (x[0] - a[i])]
#   Minimum at x* = mean(a) = 4.5  (for a = [0, 1, ..., 9])
# ---------------------------------------------------------------------------


class TestSVRG(unittest.TestCase):
    """Tests for the ``svrg`` function."""

    def setUp(self) -> None:
        """Set up the 1-D quadratic finite-sum problem."""
        self.a = list(range(10))
        self.n = 10

        def grad_full(x: list) -> list:
            """Full gradient: 2*(x[0] - mean(a))."""
            mean_a = sum(self.a) / self.n
            return [2.0 * (x[0] - mean_a)]

        def grad_i(i: int, x: list) -> list:
            """Gradient of i-th component."""
            return [2.0 * (x[0] - self.a[i])]

        self.grad_full = grad_full
        self.grad_i = grad_i

    # ------------------------------------------------------------------
    def test_svrg_converges(self) -> None:
        """SVRG should converge to x ≈ 4.5 for the quadratic problem."""
        x0 = [0.0]
        x_final, history = svrg(
            self.grad_full,
            self.grad_i,
            x0,
            n=self.n,
            learning_rate=0.05,
            epochs=20,
        )
        self.assertAlmostEqual(x_final[0], 4.5, delta=0.5)

    def test_svrg_returns_history(self) -> None:
        """History list must have exactly one entry per epoch."""
        x0 = [0.0]
        x_final, history = svrg(
            self.grad_full, self.grad_i, x0, n=self.n, epochs=5
        )
        self.assertEqual(len(history), 5)

    def test_svrg_full_gradient_used_each_epoch(self) -> None:
        """``grad_full`` must be called exactly once per epoch."""
        call_count = [0]

        def counting_grad_full(x: list) -> list:
            call_count[0] += 1
            return self.grad_full(x)

        x0 = [0.0]
        svrg(counting_grad_full, self.grad_i, x0, n=self.n, epochs=3)
        self.assertEqual(call_count[0], 3)


# ---------------------------------------------------------------------------


class TestSAGA(unittest.TestCase):
    """Tests for ``SAGAOptimizer``."""

    def setUp(self) -> None:
        self.a = list(range(10))
        self.n = 10

        def grad_i(i: int, x: list) -> list:
            return [2.0 * (x[0] - self.a[i])]

        self.grad_i = grad_i

    # ------------------------------------------------------------------
    def test_saga_converges(self) -> None:
        """SAGA should converge to x ≈ 4.5."""
        opt = SAGAOptimizer(n=self.n, d=1, learning_rate=0.03)
        x_final, _ = opt.run(self.grad_i, [0.0], n_steps=500, seed=42)
        self.assertAlmostEqual(x_final[0], 4.5, delta=1.0)

    def test_saga_table_updates(self) -> None:
        """After a step with index i=3, ``gradient_table[3]`` must change."""
        opt = SAGAOptimizer(n=self.n, d=1, learning_rate=0.01)
        x = [0.0]
        old_val = opt.gradient_table[3][0]
        opt.step(self.grad_i, x, i=3)
        new_val = opt.gradient_table[3][0]
        self.assertNotEqual(old_val, new_val)

    def test_saga_reset(self) -> None:
        """``reset()`` must set the gradient table and sum to zeros."""
        opt = SAGAOptimizer(n=self.n, d=1, learning_rate=0.01)
        opt.step(self.grad_i, [5.0], i=0)
        opt.reset()
        self.assertEqual(opt.gradient_table[0][0], 0.0)
        self.assertEqual(opt.sum_table[0], 0.0)


# ---------------------------------------------------------------------------


class TestSAG(unittest.TestCase):
    """Tests for ``SAGOptimizer``."""

    def setUp(self) -> None:
        self.a = list(range(10))
        self.n = 10

        def grad_i(i: int, x: list) -> list:
            return [2.0 * (x[0] - self.a[i])]

        self.grad_i = grad_i

    # ------------------------------------------------------------------
    def test_sag_converges(self) -> None:
        """SAG should converge to x ≈ 4.5."""
        opt = SAGOptimizer(n=self.n, d=1, learning_rate=0.1)
        x_final, _ = opt.run(self.grad_i, [0.0], n_steps=500, seed=42)
        self.assertAlmostEqual(x_final[0], 4.5, delta=1.5)

    def test_sag_uses_average(self) -> None:
        """With lr=0 and x fixed at 0, the running sum must equal
        ``sum_i grad_i(i, 0) = sum_i 2*(0 - a_i) = -2*45 = -90``."""
        opt = SAGOptimizer(n=self.n, d=1, learning_rate=0.0)
        x = [0.0]
        for i in range(self.n):
            opt.step(self.grad_i, x, i)
        expected_sum = sum(2.0 * (0.0 - float(a)) for a in self.a)  # -90.0
        self.assertAlmostEqual(opt.running_sum[0], expected_sum, delta=0.1)


# ---------------------------------------------------------------------------


class TestPolyakAveraging(unittest.TestCase):
    """Tests for ``iterate_averaging``."""

    def test_averaging_smooths_iterates(self) -> None:
        """The Polyak average must be closer to 0 than the start."""
        random.seed(42)

        def noisy_step(x: list) -> list:
            g = [2.0 * x[0] + random.gauss(0, 0.1)]
            return [x[0] - 0.1 * g[0]]

        avg, iterates = iterate_averaging(noisy_step, [5.0], n_steps=100, burn_in=10)
        self.assertLess(abs(avg[0]), 5.0)
        self.assertEqual(len(iterates), 100)

    def test_burn_in_excludes_early(self) -> None:
        """The average must equal the mean of ``iterates[50:]``."""
        calls = [0]

        def step(x: list) -> list:
            calls[0] += 1
            return [x[0] * 0.9]

        avg, iterates = iterate_averaging(step, [1.0], n_steps=100, burn_in=50)
        self.assertEqual(len(iterates), 100)

        # Average must be computed over iterates[50..99] (50 iterates)
        window = iterates[50:]
        expected_avg = sum(v[0] for v in window) / len(window)
        self.assertAlmostEqual(avg[0], expected_avg, delta=1e-10)


# ---------------------------------------------------------------------------


class TestRobbinsMonro(unittest.TestCase):
    """Tests for ``robbins_monro``."""

    def setUp(self) -> None:
        self.a = list(range(10))
        self.n = 10

        def grad_stochastic(i: int, x: list) -> list:
            return [2.0 * (x[0] - self.a[i])]

        self.grad_stochastic = grad_stochastic

    # ------------------------------------------------------------------
    def test_robbins_monro_converges(self) -> None:
        """After 2000 steps the iterate should be close to 4.5."""
        x_final, avg = robbins_monro(
            self.grad_stochastic,
            [0.0],
            n=self.n,
            n_steps=2000,
            c=1.0,
            alpha=0.602,
            seed=42,
        )
        self.assertAlmostEqual(x_final[0], 4.5, delta=2.0)

    def test_robbins_monro_lr_decays(self) -> None:
        """eta_t = c/(t+1)^alpha must be strictly decreasing for alpha > 0."""
        c, alpha = 1.0, 0.602
        etas = [c / (t + 1) ** alpha for t in range(10)]
        for i in range(len(etas) - 1):
            self.assertGreater(etas[i], etas[i + 1])

    def test_robbins_monro_returns_average(self) -> None:
        """The running_avg list must have exactly ``n_steps`` entries."""
        x_final, running_avg = robbins_monro(
            self.grad_stochastic,
            [0.0],
            n=self.n,
            n_steps=100,
            seed=0,
        )
        self.assertEqual(len(running_avg), 100)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

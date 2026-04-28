"""
tests/test_constrained.py — Unit tests for constrained.py
"""

import sys
import math
import unittest

sys.path.insert(0, "/Users/sanskar/dev/Research/Phase0_Core/Optimization")

from constrained import (
    project_box,
    project_simplex,
    project_l2_ball,
    project_l1_ball,
    project_linf_ball,
    projected_gradient,
    penalty_method,
    augmented_lagrangian,
    frank_wolfe,
    admm,
    barrier_method,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm2(x):
    return math.sqrt(sum(xi * xi for xi in x))


def _norm1(x):
    return sum(abs(xi) for xi in x)


def _norminf(x):
    return max(abs(xi) for xi in x)


# ---------------------------------------------------------------------------
# Projection tests
# ---------------------------------------------------------------------------

class TestProjections(unittest.TestCase):

    # --- project_box -------------------------------------------------------

    def test_project_box_clips_below(self):
        x = [-1.0, 0.5, 3.0]
        lb = [0.0, 0.0, 0.0]
        ub = [2.0, 2.0, 2.0]
        result = project_box(x, lb, ub)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 2.0)

    def test_project_box_clips_above(self):
        x = [5.0, -1.0]
        lb = [1.0, 0.0]
        ub = [3.0, 1.0]
        result = project_box(x, lb, ub)
        self.assertAlmostEqual(result[0], 3.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_project_box_inside_unchanged(self):
        x = [1.5, 0.5]
        lb = [0.0, 0.0]
        ub = [2.0, 1.0]
        result = project_box(x, lb, ub)
        for r, xi in zip(result, x):
            self.assertAlmostEqual(r, xi)

    def test_project_box_asymmetric_bounds(self):
        x = [0.0]
        lb = [2.0]
        ub = [5.0]
        result = project_box(x, lb, ub)
        self.assertAlmostEqual(result[0], 2.0)

    # --- project_simplex ---------------------------------------------------

    def test_project_simplex_sum_to_one(self):
        x = [3.0, 1.0, -2.0, 0.5]
        result = project_simplex(x)
        self.assertAlmostEqual(sum(result), 1.0, places=10)

    def test_project_simplex_non_negative(self):
        x = [0.1, -0.5, 0.3]
        result = project_simplex(x)
        for r in result:
            self.assertGreaterEqual(r, -1e-12)

    def test_project_simplex_already_on_simplex(self):
        x = [0.3, 0.5, 0.2]
        result = project_simplex(x)
        self.assertAlmostEqual(sum(result), 1.0, places=10)
        for r in result:
            self.assertGreaterEqual(r, -1e-12)

    def test_project_simplex_single_element(self):
        x = [3.7]
        result = project_simplex(x)
        self.assertAlmostEqual(result[0], 1.0, places=10)

    def test_project_simplex_uniform(self):
        # All equal → projection is [1/n, ..., 1/n]
        n = 4
        x = [0.5] * n
        result = project_simplex(x)
        for r in result:
            self.assertAlmostEqual(r, 1.0 / n, places=10)

    # --- project_l2_ball ---------------------------------------------------

    def test_project_l2_ball_inside_unchanged(self):
        x = [0.3, 0.4]
        result = project_l2_ball(x, radius=1.0)
        for r, xi in zip(result, x):
            self.assertAlmostEqual(r, xi)

    def test_project_l2_ball_outside_normalized(self):
        x = [3.0, 4.0]  # norm = 5
        result = project_l2_ball(x, radius=1.0)
        self.assertAlmostEqual(_norm2(result), 1.0, places=10)

    def test_project_l2_ball_custom_radius(self):
        x = [6.0, 8.0]  # norm = 10
        result = project_l2_ball(x, radius=2.0)
        self.assertAlmostEqual(_norm2(result), 2.0, places=10)

    def test_project_l2_ball_on_boundary(self):
        x = [0.6, 0.8]  # norm exactly 1
        result = project_l2_ball(x, radius=1.0)
        for r, xi in zip(result, x):
            self.assertAlmostEqual(r, xi, places=10)

    # --- project_l1_ball ---------------------------------------------------

    def test_project_l1_ball_inside_unchanged(self):
        x = [0.2, 0.3]
        result = project_l1_ball(x, radius=1.0)
        for r, xi in zip(result, x):
            self.assertAlmostEqual(r, xi)

    def test_project_l1_ball_norm_le_radius(self):
        x = [1.0, 2.0, -3.0]
        result = project_l1_ball(x, radius=2.0)
        self.assertLessEqual(_norm1(result), 2.0 + 1e-10)

    def test_project_l1_ball_preserves_signs(self):
        x = [-2.0, 3.0]
        result = project_l1_ball(x, radius=1.0)
        self.assertLessEqual(result[0], 0.0)
        self.assertGreaterEqual(result[1], 0.0)

    def test_project_l1_ball_large_vector(self):
        x = [5.0, 5.0, 5.0]
        result = project_l1_ball(x, radius=1.0)
        self.assertLessEqual(_norm1(result), 1.0 + 1e-10)

    # --- project_linf_ball -------------------------------------------------

    def test_project_linf_ball_inside_unchanged(self):
        x = [0.5, -0.3, 0.9]
        result = project_linf_ball(x, radius=1.0)
        for r, xi in zip(result, x):
            self.assertAlmostEqual(r, xi)

    def test_project_linf_ball_clips(self):
        x = [2.0, -3.0, 0.5]
        result = project_linf_ball(x, radius=1.0)
        self.assertAlmostEqual(result[0], 1.0)
        self.assertAlmostEqual(result[1], -1.0)
        self.assertAlmostEqual(result[2], 0.5)

    def test_project_linf_ball_custom_radius(self):
        x = [5.0, -10.0]
        result = project_linf_ball(x, radius=3.0)
        self.assertAlmostEqual(result[0], 3.0)
        self.assertAlmostEqual(result[1], -3.0)

    def test_project_linf_ball_max_norm(self):
        x = [1.5, 0.5, -2.0]
        result = project_linf_ball(x, radius=1.0)
        self.assertLessEqual(_norminf(result), 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# Projected gradient descent
# ---------------------------------------------------------------------------

class TestProjectedGradient(unittest.TestCase):
    """Minimise x^2 + y^2 subject to x >= 1.  Optimal: [1, 0]."""

    def setUp(self):
        self.f = lambda v: v[0] ** 2 + v[1] ** 2
        self.grad_f = lambda v: [2.0 * v[0], 2.0 * v[1]]
        # Feasible set: x >= 1, y unconstrained
        def project(v):
            return [max(v[0], 1.0), v[1]]
        self.project = project

    def test_converges_to_opt(self):
        x0 = [3.0, 3.0]
        x_opt, f_opt, n_iters, converged = projected_gradient(
            self.f, self.grad_f, self.project, x0,
            lr=0.1, max_iter=2000, tol=1e-6,
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=4)
        self.assertAlmostEqual(x_opt[1], 0.0, places=4)

    def test_converged_flag(self):
        x0 = [3.0, 3.0]
        _, _, _, converged = projected_gradient(
            self.f, self.grad_f, self.project, x0,
            lr=0.1, max_iter=2000, tol=1e-6,
        )
        self.assertTrue(converged)

    def test_already_feasible_opt(self):
        # Starting at optimal → should stay
        x0 = [1.0, 0.0]
        x_opt, f_opt, _, _ = projected_gradient(
            self.f, self.grad_f, self.project, x0,
            lr=0.1, max_iter=200, tol=1e-8,
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=4)
        self.assertAlmostEqual(x_opt[1], 0.0, places=4)

    def test_return_types(self):
        x_opt, f_opt, n_iters, converged = projected_gradient(
            self.f, self.grad_f, self.project, [2.0, 2.0],
            lr=0.1, max_iter=100,
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_iters, int)
        self.assertIsInstance(converged, bool)


# ---------------------------------------------------------------------------
# Penalty method
# ---------------------------------------------------------------------------

class TestPenaltyMethod(unittest.TestCase):
    """Minimise x^2 + y^2  s.t.  x + y >= 1.  Optimal: [0.5, 0.5]."""

    def setUp(self):
        self.f = lambda v: v[0] ** 2 + v[1] ** 2
        self.grad_f = lambda v: [2.0 * v[0], 2.0 * v[1]]
        # c(x) = 1 - x - y <= 0
        self.constraints = [lambda v: 1.0 - v[0] - v[1]]

    def test_converges_near_opt(self):
        x0 = [0.0, 0.0]
        x_opt, f_opt, n_outer, converged = penalty_method(
            self.f, self.grad_f, self.constraints, x0,
            mu0=1.0, mu_factor=10.0, max_outer=15, max_inner=200, tol=1e-5,
        )
        self.assertAlmostEqual(x_opt[0], 0.5, places=2)
        self.assertAlmostEqual(x_opt[1], 0.5, places=2)

    def test_feasibility(self):
        x0 = [0.0, 0.0]
        x_opt, _, _, _ = penalty_method(
            self.f, self.grad_f, self.constraints, x0,
            mu0=1.0, mu_factor=10.0, max_outer=15, max_inner=200, tol=1e-5,
        )
        self.assertGreaterEqual(x_opt[0] + x_opt[1], 1.0 - 1e-3)

    def test_return_types(self):
        x_opt, f_opt, n_outer, converged = penalty_method(
            self.f, self.grad_f, self.constraints, [1.0, 1.0],
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_outer, int)
        self.assertIsInstance(converged, bool)

    def test_with_provided_grad_constraints(self):
        grad_c = [lambda v: [-1.0, -1.0]]
        x0 = [0.0, 0.0]
        x_opt, _, _, _ = penalty_method(
            self.f, self.grad_f, self.constraints, x0,
            mu0=1.0, mu_factor=10.0, max_outer=15, max_inner=200, tol=1e-5,
            grad_constraints=grad_c,
        )
        self.assertAlmostEqual(x_opt[0], 0.5, places=2)
        self.assertAlmostEqual(x_opt[1], 0.5, places=2)


# ---------------------------------------------------------------------------
# Augmented Lagrangian
# ---------------------------------------------------------------------------

class TestAugmentedLagrangian(unittest.TestCase):
    """Minimise x^2 + y^2  s.t.  x + y = 1.  Optimal: x = y = 0.5."""

    def setUp(self):
        self.f = lambda v: v[0] ** 2 + v[1] ** 2
        self.grad_f = lambda v: [2.0 * v[0], 2.0 * v[1]]
        # h(x) = x + y - 1 = 0
        self.eq_constraints = [lambda v: v[0] + v[1] - 1.0]

    def test_converges_to_opt(self):
        x0 = [0.0, 0.0]
        x_opt, f_opt, n_outer, converged = augmented_lagrangian(
            self.f, self.grad_f, self.eq_constraints, x0,
            mu0=1.0, mu_factor=2.0, max_outer=30, max_inner=200, tol=1e-6,
        )
        self.assertAlmostEqual(x_opt[0], 0.5, places=3)
        self.assertAlmostEqual(x_opt[1], 0.5, places=3)

    def test_equality_satisfied(self):
        x0 = [0.0, 0.0]
        x_opt, _, _, _ = augmented_lagrangian(
            self.f, self.grad_f, self.eq_constraints, x0,
            mu0=1.0, mu_factor=2.0, max_outer=30, max_inner=200, tol=1e-6,
        )
        self.assertAlmostEqual(x_opt[0] + x_opt[1], 1.0, places=3)

    def test_optimal_value(self):
        x0 = [0.0, 0.0]
        _, f_opt, _, _ = augmented_lagrangian(
            self.f, self.grad_f, self.eq_constraints, x0,
            mu0=1.0, mu_factor=2.0, max_outer=30, max_inner=200, tol=1e-6,
        )
        # Minimum is 0.5^2 + 0.5^2 = 0.5
        self.assertAlmostEqual(f_opt, 0.5, places=3)

    def test_return_types(self):
        x_opt, f_opt, n_outer, converged = augmented_lagrangian(
            self.f, self.grad_f, self.eq_constraints, [0.5, 0.5],
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_outer, int)
        self.assertIsInstance(converged, bool)

    def test_with_initial_multipliers(self):
        x0 = [0.0, 0.0]
        x_opt, _, _, _ = augmented_lagrangian(
            self.f, self.grad_f, self.eq_constraints, x0,
            lam0=[0.0], mu0=1.0, mu_factor=2.0, max_outer=30, max_inner=200, tol=1e-6,
        )
        self.assertAlmostEqual(x_opt[0] + x_opt[1], 1.0, places=3)


# ---------------------------------------------------------------------------
# Frank-Wolfe
# ---------------------------------------------------------------------------

class TestFrankWolfe(unittest.TestCase):
    """Minimise x^2 + y^2 over the L1 ball (radius 1).

    The minimum of a convex quadratic over the L1 ball is 0, achieved at the
    origin; but the origin is in the interior of the L1 ball, so Frank-Wolfe
    will converge to very small values near 0.
    """

    def setUp(self):
        self.f = lambda v: v[0] ** 2 + v[1] ** 2
        self.grad_f = lambda v: [2.0 * v[0], 2.0 * v[1]]

        # LP oracle for L1 ball: argmin_{||s||_1 <= 1} <g, s>
        # Solution: put all weight on coordinate with most negative g_i
        def lp_oracle(g):
            n = len(g)
            s = [0.0] * n
            idx = min(range(n), key=lambda i: g[i])
            if g[idx] < 0:
                s[idx] = 1.0
            else:
                idx2 = max(range(n), key=lambda i: g[i])
                s[idx2] = -1.0
            return s

        self.lp_oracle = lp_oracle

    def test_objective_decreases(self):
        x0 = [1.0, 0.0]  # vertex of L1 ball
        x_opt, f_opt, n_iters, converged = frank_wolfe(
            self.f, self.grad_f, self.lp_oracle, x0,
            max_iter=500, tol=1e-4,
        )
        # Minimum of x^2+y^2 over L1 ball is 0 (at origin)
        self.assertLess(f_opt, self.f(x0))

    def test_stays_in_l1_ball(self):
        x0 = [1.0, 0.0]
        x_opt, _, _, _ = frank_wolfe(
            self.f, self.grad_f, self.lp_oracle, x0,
            max_iter=500, tol=1e-4,
        )
        self.assertLessEqual(_norm1(x_opt), 1.0 + 1e-10)

    def test_converges_toward_zero(self):
        x0 = [0.5, 0.5]
        x_opt, f_opt, n_iters, _ = frank_wolfe(
            self.f, self.grad_f, self.lp_oracle, x0,
            max_iter=1000, tol=1e-6,
        )
        # Frank-Wolfe converges at O(1/k); f_opt should be small
        self.assertLess(f_opt, 0.1)

    def test_return_types(self):
        x_opt, f_opt, n_iters, converged = frank_wolfe(
            self.f, self.grad_f, self.lp_oracle, [1.0, 0.0],
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_iters, int)
        self.assertIsInstance(converged, bool)


# ---------------------------------------------------------------------------
# ADMM
# ---------------------------------------------------------------------------

class TestADMM(unittest.TestCase):
    """Consensus ADMM: min ||x-a||^2 + ||z-b||^2  s.t.  x = z.
    Optimal: x = z = (a + b) / 2.
    """

    def setUp(self):
        self.a = [1.0, 3.0]
        self.b = [3.0, 1.0]
        # prox of ||·-a||^2 with parameter rho: (v + rho*a) / (1 + rho)
        a = self.a
        b = self.b

        def prox_f(v, rho):
            return [(vi + rho * ai) / (1.0 + rho) for vi, ai in zip(v, a)]

        def prox_g(v, rho):
            return [(vi + rho * bi) / (1.0 + rho) for vi, bi in zip(v, b)]

        self.prox_f = prox_f
        self.prox_g = prox_g

    def test_converges_to_mean(self):
        x0 = [0.0, 0.0]
        x_opt, z_opt, n_iters, converged = admm(
            self.prox_f, self.prox_g, x0,
            rho=1.0, max_iter=500, tol=1e-6,
        )
        expected = [(ai + bi) / 2.0 for ai, bi in zip(self.a, self.b)]
        for r, e in zip(x_opt, expected):
            self.assertAlmostEqual(r, e, places=4)
        for r, e in zip(z_opt, expected):
            self.assertAlmostEqual(r, e, places=4)

    def test_x_equals_z_at_convergence(self):
        x0 = [0.0, 0.0]
        x_opt, z_opt, _, _ = admm(
            self.prox_f, self.prox_g, x0,
            rho=1.0, max_iter=500, tol=1e-6,
        )
        for xi, zi in zip(x_opt, z_opt):
            self.assertAlmostEqual(xi, zi, places=4)

    def test_return_types(self):
        x_opt, z_opt, n_iters, converged = admm(
            self.prox_f, self.prox_g, [0.0, 0.0],
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(z_opt, list)
        self.assertIsInstance(n_iters, int)
        self.assertIsInstance(converged, bool)

    def test_with_z0(self):
        x0 = [0.0, 0.0]
        z0 = [5.0, 5.0]
        x_opt, z_opt, _, _ = admm(
            self.prox_f, self.prox_g, x0, z0=z0,
            rho=1.0, max_iter=500, tol=1e-6,
        )
        expected = [(ai + bi) / 2.0 for ai, bi in zip(self.a, self.b)]
        for r, e in zip(x_opt, expected):
            self.assertAlmostEqual(r, e, places=4)


# ---------------------------------------------------------------------------
# Barrier method
# ---------------------------------------------------------------------------

class TestBarrierMethod(unittest.TestCase):
    """Minimise x^2 via log-barrier -log(x - 1) for x > 1.

    As t -> inf the barrier solution converges to x = 1 (the boundary).
    With large enough t the solution should be very close to 1.
    """

    def setUp(self):
        # Objective: f(x) = x^2,  constraint: x >= 1  i.e. c(x) = 1 - x <= 0
        # Log barrier: -log(x - 1)
        self.f = lambda v: v[0] ** 2
        self.grad_f = lambda v: [2.0 * v[0]]

        def barrier_f(v):
            if v[0] <= 1.0:
                raise ValueError("Outside domain")
            return -math.log(v[0] - 1.0)

        def barrier_grad(v):
            return [-1.0 / (v[0] - 1.0)]

        self.barrier_f = barrier_f
        self.barrier_grad = barrier_grad

    def test_converges_near_boundary(self):
        x0 = [2.0]  # strictly feasible
        x_opt, f_opt, n_outer, converged = barrier_method(
            self.f, self.grad_f, self.barrier_f, self.barrier_grad, x0,
            t0=1.0, mu=10.0, max_outer=20, max_inner=200, tol=1e-6,
        )
        # x should be close to 1 (and >= 1)
        self.assertGreater(x_opt[0], 1.0 - 1e-3)
        self.assertLess(x_opt[0], 1.5)

    def test_solution_feasible(self):
        x0 = [3.0]
        x_opt, _, _, _ = barrier_method(
            self.f, self.grad_f, self.barrier_f, self.barrier_grad, x0,
            t0=1.0, mu=10.0, max_outer=20, max_inner=200, tol=1e-6,
        )
        self.assertGreater(x_opt[0], 1.0 - 1e-6)

    def test_objective_at_opt(self):
        x0 = [2.0]
        x_opt, f_opt, _, _ = barrier_method(
            self.f, self.grad_f, self.barrier_f, self.barrier_grad, x0,
            t0=1.0, mu=10.0, max_outer=20, max_inner=200, tol=1e-6,
        )
        # f_opt = x_opt^2 should be close to 1
        self.assertLess(f_opt, 2.0)

    def test_return_types(self):
        x0 = [2.0]
        x_opt, f_opt, n_outer, converged = barrier_method(
            self.f, self.grad_f, self.barrier_f, self.barrier_grad, x0,
        )
        self.assertIsInstance(x_opt, list)
        self.assertIsInstance(f_opt, float)
        self.assertIsInstance(n_outer, int)
        self.assertIsInstance(converged, bool)


if __name__ == "__main__":
    unittest.main()

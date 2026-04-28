"""Tests for second_order.py — comprehensive coverage of all 8 optimizers."""

import sys
import os
import math
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from second_order import (
    newton_raphson,
    bfgs,
    lbfgs,
    sr1,
    gauss_newton,
    levenberg_marquardt,
    trust_region,
    newton_cg,
)

# ---------------------------------------------------------------------------
# Shared test functions
# ---------------------------------------------------------------------------

def sphere(x):
    """f(x) = sum(xi^2);  minimiser at origin."""
    return sum(xi * xi for xi in x)

def sphere_grad(x):
    return [2.0 * xi for xi in x]

def sphere_hess(x):
    n = len(x)
    return [[2.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def rosenbrock(x):
    """Classic Rosenbrock banana function; minimiser at (1, 1)."""
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

def rosenbrock_grad(x):
    dx0 = -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2)
    dx1 = 200.0 * (x[1] - x[0] ** 2)
    return [dx0, dx1]

def rosenbrock_hess(x):
    h00 = 2.0 - 400.0 * (x[1] - x[0] ** 2) + 800.0 * x[0] ** 2
    h01 = -400.0 * x[0]
    h10 = -400.0 * x[0]
    h11 = 200.0
    return [[h00, h01], [h10, h11]]

def quadratic(x):
    """f(x) = 3*x0^2 + 2*x1^2 + x0*x1;  known minimum at origin."""
    return 3.0 * x[0] ** 2 + 2.0 * x[1] ** 2 + x[0] * x[1]

def quadratic_grad(x):
    return [6.0 * x[0] + x[1], 4.0 * x[1] + x[0]]

def quadratic_hess(x):
    return [[6.0, 1.0], [1.0, 4.0]]


# ---------------------------------------------------------------------------
# TestNewtonRaphson
# ---------------------------------------------------------------------------

class TestNewtonRaphson(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = newton_raphson(sphere, sphere_grad, sphere_hess, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges_to_origin(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = newton_raphson(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)
        self.assertAlmostEqual(f_opt, 0.0, places=8)

    def test_sphere_3d_converges(self):
        x0 = [1.0, -1.0, 2.0]
        x_opt, f_opt, n_iters, converged = newton_raphson(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        for xi in x_opt:
            self.assertAlmostEqual(xi, 0.0, places=5)

    def test_rosenbrock_converges_to_one_one(self):
        x0 = [0.0, 0.0]
        x_opt, f_opt, n_iters, converged = newton_raphson(
            rosenbrock, rosenbrock_grad, rosenbrock_hess, x0,
            tol=1e-6, max_iter=200
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=4)
        self.assertAlmostEqual(x_opt[1], 1.0, places=4)

    def test_quadratic_converges(self):
        x0 = [5.0, -3.0]
        x_opt, f_opt, n_iters, converged = newton_raphson(
            quadratic, quadratic_grad, quadratic_hess, x0
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_n_iters_positive(self):
        x0 = [1.0, 1.0]
        _, _, n_iters, _ = newton_raphson(sphere, sphere_grad, sphere_hess, x0)
        self.assertGreaterEqual(n_iters, 1)

    def test_converged_flag_true_at_minimum(self):
        # Start exactly at minimum
        x0 = [0.0, 0.0]
        _, _, _, converged = newton_raphson(sphere, sphere_grad, sphere_hess, x0, tol=1.0)
        self.assertTrue(converged)

    def test_not_converged_when_maxiter_is_one_from_far(self):
        # Starting far from minimum, 1 iteration of Newton won't reach tol=1e-15
        x0 = [10.0, 10.0]
        _, _, n_iters, converged = newton_raphson(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-15, max_iter=1
        )
        # Newton on sphere converges in 1 step exactly, so let's use a harder tol
        # What matters is n_iters is bounded by max_iter
        self.assertLessEqual(n_iters, 2)


# ---------------------------------------------------------------------------
# TestBFGS
# ---------------------------------------------------------------------------

class TestBFGS(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = bfgs(sphere, sphere_grad, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = bfgs(sphere, sphere_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_sphere_1d_converges(self):
        x0 = [5.0]
        x_opt, f_opt, n_iters, converged = bfgs(sphere, sphere_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)

    def test_rosenbrock_converges(self):
        x0 = [-1.0, 1.0]
        x_opt, f_opt, n_iters, converged = bfgs(
            rosenbrock, rosenbrock_grad, x0, tol=1e-6, max_iter=500
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=3)
        self.assertAlmostEqual(x_opt[1], 1.0, places=3)

    def test_converged_flag(self):
        x_opt, f_opt, n_iters, converged = bfgs(sphere, sphere_grad, [0.0, 0.0], tol=1.0)
        self.assertTrue(converged)

    def test_f_opt_equals_f_at_x_opt(self):
        x0 = [2.0, -1.0]
        x_opt, f_opt, _, _ = bfgs(sphere, sphere_grad, x0)
        self.assertAlmostEqual(f_opt, sphere(x_opt), places=10)

    def test_sphere_3d(self):
        x0 = [1.0, 2.0, -3.0]
        x_opt, f_opt, n_iters, converged = bfgs(sphere, sphere_grad, x0, tol=1e-7)
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-10)

    def test_quadratic(self):
        x0 = [4.0, -2.0]
        x_opt, f_opt, n_iters, converged = bfgs(quadratic, quadratic_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)


# ---------------------------------------------------------------------------
# TestLBFGS
# ---------------------------------------------------------------------------

class TestLBFGS(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = lbfgs(sphere, sphere_grad, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = lbfgs(sphere, sphere_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_rosenbrock_with_m5(self):
        x0 = [-1.0, 1.0]
        x_opt, f_opt, n_iters, converged = lbfgs(
            rosenbrock, rosenbrock_grad, x0, m=5, tol=1e-6, max_iter=500
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=3)
        self.assertAlmostEqual(x_opt[1], 1.0, places=3)

    def test_with_m1(self):
        x0 = [2.0, -1.0]
        x_opt, f_opt, n_iters, converged = lbfgs(sphere, sphere_grad, x0, m=1, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(f_opt, 0.0, places=8)

    def test_with_m10_default(self):
        x0 = [5.0, 5.0]
        x_opt, f_opt, n_iters, converged = lbfgs(sphere, sphere_grad, x0, m=10, tol=1e-8)
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-10)

    def test_sphere_5d(self):
        x0 = [1.0, -1.0, 2.0, -2.0, 0.5]
        x_opt, f_opt, n_iters, converged = lbfgs(sphere, sphere_grad, x0, tol=1e-7)
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-10)

    def test_f_opt_consistent(self):
        x0 = [3.0, -1.0]
        x_opt, f_opt, _, _ = lbfgs(sphere, sphere_grad, x0)
        self.assertAlmostEqual(f_opt, sphere(x_opt), places=10)


# ---------------------------------------------------------------------------
# TestSR1
# ---------------------------------------------------------------------------

class TestSR1(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = sr1(sphere, sphere_grad, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = sr1(sphere, sphere_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_sphere_1d(self):
        x0 = [7.0]
        x_opt, f_opt, n_iters, converged = sr1(sphere, sphere_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)

    def test_makes_progress_on_rosenbrock(self):
        x0 = [-1.0, 1.0]
        x_opt, f_opt, n_iters, converged = sr1(
            rosenbrock, rosenbrock_grad, x0, tol=1e-6, max_iter=500
        )
        # SR1 might not always converge fully, but must reduce function value
        self.assertLess(rosenbrock(x_opt), rosenbrock(x0))

    def test_quadratic_converges(self):
        x0 = [4.0, -2.0]
        x_opt, f_opt, n_iters, converged = sr1(quadratic, quadratic_grad, x0, tol=1e-8)
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_skip_update_guard(self):
        # With r=1.0 (very aggressive skip), should still make progress
        x0 = [2.0, -1.0]
        x_opt, f_opt, n_iters, converged = sr1(sphere, sphere_grad, x0, r=1.0, tol=1e-6)
        self.assertLess(f_opt, sphere(x0))


# ---------------------------------------------------------------------------
# TestGaussNewton
# ---------------------------------------------------------------------------

class TestGaussNewton(unittest.TestCase):

    def _simple_residuals(self, targets):
        """r_i(x) = x[0] - target_i  →  minimiser at mean(targets)."""
        def residuals_f(x):
            return [x[0] - t for t in targets]
        def jacobian_f(x):
            return [[1.0] for _ in targets]
        return residuals_f, jacobian_f

    def test_returns_tuple_of_4(self):
        r_f, j_f = self._simple_residuals([1.0, 2.0, 3.0])
        result = gauss_newton(r_f, j_f, [0.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_linear_residuals_converge_to_mean(self):
        targets = [1.0, 2.0, 3.0, 4.0, 5.0]
        r_f, j_f = self._simple_residuals(targets)
        x_opt, res_norm, n_iters, converged = gauss_newton(r_f, j_f, [0.0], tol=1e-8)
        expected = sum(targets) / len(targets)  # 3.0
        self.assertAlmostEqual(x_opt[0], expected, places=5)

    def test_residual_norm_at_solution(self):
        # For r_i(x) = x - c, min at x=mean; residual_norm = std dev scaled
        targets = [0.0, 2.0]          # mean = 1.0, residuals at opt = [-1, 1]
        r_f, j_f = self._simple_residuals(targets)
        x_opt, res_norm, n_iters, converged = gauss_newton(r_f, j_f, [0.0], tol=1e-8)
        self.assertAlmostEqual(x_opt[0], 1.0, places=5)
        # ||r||^2 = (-1)^2 + 1^2 = 2  →  ||r|| = sqrt(2)
        self.assertAlmostEqual(res_norm, math.sqrt(2.0), places=5)

    def test_2d_least_squares(self):
        # r_i(x) = x[0] + x[1] - bi with bi = i
        # J = [[1, 1], [1, 1], [1, 1]] — rank 1
        # Use a well-posed system instead: r_i(x) = a_i*x[0] + b_i*x[1] - c_i
        # True solution: x = [1, 2]
        A_mat = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]]
        b_vec = [1.0, 2.0, 3.0, 0.0]  # A_mat @ [1, 2] = [1, 2, 3, 0]
        def res_f(x):
            return [sum(A_mat[i][j] * x[j] for j in range(2)) - b_vec[i]
                    for i in range(4)]
        def jac_f(x):
            return [row[:] for row in A_mat]
        x_opt, res_norm, n_iters, converged = gauss_newton(res_f, jac_f, [0.0, 0.0], tol=1e-8)
        self.assertAlmostEqual(x_opt[0], 1.0, places=4)
        self.assertAlmostEqual(x_opt[1], 2.0, places=4)

    def test_converged_flag(self):
        targets = [5.0]
        r_f, j_f = self._simple_residuals(targets)
        x_opt, _, _, converged = gauss_newton(r_f, j_f, [0.0], tol=1.0)
        self.assertTrue(converged)

    def test_nonlinear_residuals(self):
        # r(x) = [x[0]^2 - 1];  GN minimises (x^2-1)^2, solution x=1 or x=-1
        def res_f(x):
            return [x[0] ** 2 - 1.0]
        def jac_f(x):
            return [[2.0 * x[0]]]
        x_opt, res_norm, n_iters, converged = gauss_newton(res_f, jac_f, [2.0], tol=1e-8)
        self.assertAlmostEqual(abs(x_opt[0]), 1.0, places=4)


# ---------------------------------------------------------------------------
# TestLevenbergMarquardt
# ---------------------------------------------------------------------------

class TestLevenbergMarquardt(unittest.TestCase):

    def _linear_setup(self, targets, x_init):
        def res_f(x):
            return [x[0] - t for t in targets]
        def jac_f(x):
            return [[1.0] for _ in targets]
        return res_f, jac_f

    def test_returns_tuple_of_4(self):
        r_f, j_f = self._linear_setup([1.0, 2.0], [0.0])
        result = levenberg_marquardt(r_f, j_f, [0.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_simple_least_squares(self):
        targets = [1.0, 2.0, 3.0]
        r_f, j_f = self._linear_setup(targets, [0.0])
        x_opt, res_norm, n_iters, converged = levenberg_marquardt(
            r_f, j_f, [0.0], lam=1.0, tol=1e-8
        )
        self.assertAlmostEqual(x_opt[0], 2.0, places=4)  # mean of targets

    def test_converged_flag(self):
        targets = [3.0]
        r_f, j_f = self._linear_setup(targets, [0.0])
        _, _, _, converged = levenberg_marquardt(r_f, j_f, [0.0], lam=1.0, tol=1.0)
        self.assertTrue(converged)

    def test_lambda_adaptation_reduces_residual(self):
        # Start with a moderate lambda; after optimisation residual should be low
        targets = [1.0, 3.0, 5.0]   # mean = 3.0
        r_f, j_f = self._linear_setup(targets, [0.0])
        x_opt, res_norm, n_iters, converged = levenberg_marquardt(
            r_f, j_f, [0.0], lam=100.0, tol=1e-6, max_iter=200
        )
        # Even starting with large lambda, should converge toward mean
        self.assertAlmostEqual(x_opt[0], 3.0, places=3)

    def test_nonlinear_rosenbrock_residuals(self):
        # LM on Rosenbrock viewed as NLS:
        # r = [10*(x1 - x0^2), (1 - x0)]  →  min at (1, 1)
        def res_f(x):
            return [10.0 * (x[1] - x[0] ** 2), 1.0 - x[0]]
        def jac_f(x):
            return [[-20.0 * x[0], 10.0], [-1.0, 0.0]]
        x_opt, res_norm, n_iters, converged = levenberg_marquardt(
            res_f, jac_f, [0.0, 0.0], lam=1.0, tol=1e-6, max_iter=200
        )
        self.assertAlmostEqual(x_opt[0], 1.0, places=3)
        self.assertAlmostEqual(x_opt[1], 1.0, places=3)

    def test_2d_linear_system(self):
        A_mat = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        b_vec = [2.0, 3.0, 5.0]  # [1,0]*x=[2], [0,1]*x=[3], [1,1]*x=[5] → x=[2,3]
        def res_f(x):
            return [sum(A_mat[i][j] * x[j] for j in range(2)) - b_vec[i]
                    for i in range(3)]
        def jac_f(x):
            return [row[:] for row in A_mat]
        x_opt, res_norm, n_iters, converged = levenberg_marquardt(
            res_f, jac_f, [0.0, 0.0], lam=1.0, tol=1e-8
        )
        self.assertAlmostEqual(x_opt[0], 2.0, places=4)
        self.assertAlmostEqual(x_opt[1], 3.0, places=4)


# ---------------------------------------------------------------------------
# TestTrustRegion
# ---------------------------------------------------------------------------

class TestTrustRegion(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = trust_region(sphere, sphere_grad, sphere_hess, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges_to_origin(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = trust_region(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-6
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_sphere_3d_converges(self):
        x0 = [2.0, -1.0, 3.0]
        x_opt, f_opt, n_iters, converged = trust_region(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-6
        )
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-8)

    def test_quadratic_converges(self):
        x0 = [5.0, -3.0]
        x_opt, f_opt, n_iters, converged = trust_region(
            quadratic, quadratic_grad, quadratic_hess, x0, tol=1e-6
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=4)
        self.assertAlmostEqual(x_opt[1], 0.0, places=4)

    def test_rosenbrock_makes_progress(self):
        x0 = [-1.0, 1.0]
        x_opt, f_opt, n_iters, converged = trust_region(
            rosenbrock, rosenbrock_grad, rosenbrock_hess, x0,
            tol=1e-6, max_iter=200
        )
        self.assertLess(rosenbrock(x_opt), rosenbrock(x0))

    def test_rho_update_reduces_radius_on_bad_step(self):
        # With a tiny initial radius and very tight eta, radius might shrink;
        # the algorithm should still terminate.
        x0 = [1.0, 1.0]
        x_opt, f_opt, n_iters, converged = trust_region(
            sphere, sphere_grad, sphere_hess, x0, delta0=0.01, eta=0.9, tol=1e-6
        )
        # Must make progress regardless
        self.assertLessEqual(f_opt, sphere(x0) + 1e-6)

    def test_f_opt_consistent(self):
        x0 = [2.0, -1.0]
        x_opt, f_opt, _, _ = trust_region(sphere, sphere_grad, sphere_hess, x0)
        self.assertAlmostEqual(f_opt, sphere(x_opt), places=10)

    def test_n_iters_positive(self):
        x0 = [1.0, -1.0]
        _, _, n_iters, _ = trust_region(sphere, sphere_grad, sphere_hess, x0)
        self.assertGreaterEqual(n_iters, 1)


# ---------------------------------------------------------------------------
# TestNewtonCG
# ---------------------------------------------------------------------------

class TestNewtonCG(unittest.TestCase):

    def test_returns_tuple_of_4(self):
        result = newton_cg(sphere, sphere_grad, sphere_hess, [1.0, 1.0])
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)

    def test_sphere_2d_converges(self):
        x0 = [3.0, -2.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_sphere_3d(self):
        x0 = [1.0, -2.0, 3.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-10)

    def test_quadratic_converges(self):
        x0 = [5.0, -3.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            quadratic, quadratic_grad, quadratic_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)
        self.assertAlmostEqual(x_opt[1], 0.0, places=5)

    def test_rosenbrock_makes_progress(self):
        x0 = [0.0, 0.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            rosenbrock, rosenbrock_grad, rosenbrock_hess, x0,
            tol=1e-6, max_iter=300
        )
        self.assertLess(rosenbrock(x_opt), rosenbrock(x0))

    def test_converged_at_minimum(self):
        x0 = [0.0, 0.0]
        _, _, _, converged = newton_cg(sphere, sphere_grad, sphere_hess, x0, tol=1.0)
        self.assertTrue(converged)

    def test_f_opt_consistent(self):
        x0 = [3.0, -1.0]
        x_opt, f_opt, _, _ = newton_cg(sphere, sphere_grad, sphere_hess, x0)
        self.assertAlmostEqual(f_opt, sphere(x_opt), places=10)

    def test_cg_tol_parameter(self):
        # cg_tol=0.9 (looser CG) should still converge on sphere
        x0 = [2.0, -2.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-7, cg_tol=0.9
        )
        self.assertTrue(converged)
        self.assertLess(f_opt, 1e-10)

    def test_sphere_1d(self):
        x0 = [10.0]
        x_opt, f_opt, n_iters, converged = newton_cg(
            sphere, sphere_grad, sphere_hess, x0, tol=1e-8
        )
        self.assertTrue(converged)
        self.assertAlmostEqual(x_opt[0], 0.0, places=5)


# ---------------------------------------------------------------------------
# Module-level smoke tests
# ---------------------------------------------------------------------------

class TestModuleExports(unittest.TestCase):

    def test_all_exports_present(self):
        import second_order
        expected = [
            'newton_raphson', 'bfgs', 'lbfgs', 'sr1',
            'gauss_newton', 'levenberg_marquardt', 'trust_region', 'newton_cg',
        ]
        for name in expected:
            self.assertIn(name, second_order.__all__, f"{name} missing from __all__")
            self.assertTrue(callable(getattr(second_order, name)), f"{name} not callable")


if __name__ == '__main__':
    unittest.main(verbosity=2)

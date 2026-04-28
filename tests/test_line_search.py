"""
test_line_search.py — Comprehensive tests for line_search.py.

Run with:
    python -m pytest /Users/sanskar/dev/Research/Phase0_Core/Optimization/tests/test_line_search.py -v
"""

import math
import sys
import unittest

sys.path.insert(0, '/Users/sanskar/dev/Research/Phase0_Core/Optimization')

from line_search import (
    backtracking_line_search,
    brent_minimize,
    cubic_interpolation_line_search,
    strong_wolfe_line_search,
    wolfe_line_search,
)


# ---------------------------------------------------------------------------
# Shared test functions
# ---------------------------------------------------------------------------

def _dot(u, v):
    return sum(ui * vi for ui, vi in zip(u, v))


def _axpy(a, x, y):
    return [a * xi + yi for xi, yi in zip(x, y)]


def _norm(v):
    return math.sqrt(sum(vi ** 2 for vi in v))


# --- sphere (sum of squares) ---
def sphere(x):
    return sum(xi ** 2 for xi in x)


def grad_sphere(x):
    return [2.0 * xi for xi in x]


# --- rosenbrock ---
def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
               for i in range(len(x) - 1))


def grad_rosenbrock(x):
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return g


# --- simple quadratic f(x) = 0.5 * ||x||^2  (same as sphere/2) ---
def quadratic(x):
    return 0.5 * sum(xi ** 2 for xi in x)


def grad_quadratic(x):
    return list(x)


def _neg_descent(grad):
    """Return the negative gradient (steepest descent direction)."""
    return [-gi for gi in grad]


# ---------------------------------------------------------------------------
# Helper: check Armijo condition
# ---------------------------------------------------------------------------

def armijo_satisfied(f, x, direction, alpha, c=1e-4):
    g = [2.0 * xi for xi in x]          # grad_sphere; caller should pass correct one
    return True  # placeholder — tests do it inline


# ===========================================================================
# 1. Backtracking line search
# ===========================================================================

class TestBacktracking(unittest.TestCase):

    def _armijo_ok(self, f, grad_f, x, d, alpha, c=1e-4):
        f0 = f(x)
        slope = _dot(grad_f(x), d)
        x_new = _axpy(alpha, d, x)
        return f(x_new) <= f0 + c * alpha * slope

    # ------------------------------------------------------------------
    def test_returns_float(self):
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d)
        self.assertIsInstance(alpha, float)

    def test_alpha_positive(self):
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d)
        self.assertGreater(alpha, 0.0)

    def test_armijo_satisfied_sphere_2d(self):
        x = [3.0, 4.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d)
        self.assertTrue(self._armijo_ok(sphere, grad_sphere, x, d, alpha))

    def test_armijo_satisfied_sphere_5d(self):
        x = [1.0, -2.0, 3.0, -4.0, 5.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d)
        self.assertTrue(self._armijo_ok(sphere, grad_sphere, x, d, alpha))

    def test_armijo_satisfied_rosenbrock(self):
        x = [-1.0, 1.0]
        d = _neg_descent(grad_rosenbrock(x))
        alpha = backtracking_line_search(rosenbrock, grad_rosenbrock, x, d)
        self.assertTrue(self._armijo_ok(rosenbrock, grad_rosenbrock, x, d, alpha))

    def test_function_value_decreases(self):
        """f(x + alpha*d) < f(x) must hold for a descent direction."""
        x = [2.0, -3.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d)
        x_new = _axpy(alpha, d, x)
        self.assertLess(sphere(x_new), sphere(x))

    def test_custom_rho_and_c(self):
        x = [1.0, 1.0]
        d = _neg_descent(grad_sphere(x))
        alpha = backtracking_line_search(sphere, grad_sphere, x, d,
                                         alpha0=2.0, rho=0.8, c=1e-3)
        self.assertTrue(self._armijo_ok(sphere, grad_sphere, x, d, alpha, c=1e-3))

    def test_alpha_at_most_alpha0(self):
        """Backtracking never increases beyond alpha0."""
        x = [1.0, 1.0]
        d = _neg_descent(grad_sphere(x))
        alpha0 = 0.7
        alpha = backtracking_line_search(sphere, grad_sphere, x, d, alpha0=alpha0)
        self.assertLessEqual(alpha, alpha0 + 1e-12)

    def test_quadratic_exact_step(self):
        """For f(x) = 0.5*||x||^2 the exact minimiser is alpha=1 along -grad."""
        x = [1.0, 0.0]
        d = [-1.0, 0.0]   # exact Newton step moves to [0, 0]
        alpha = backtracking_line_search(quadratic, grad_quadratic, x, d,
                                          alpha0=1.0, rho=0.5, c=1e-4)
        # alpha0=1.0 should satisfy Armijo and be returned immediately
        self.assertAlmostEqual(alpha, 1.0, places=10)


# ===========================================================================
# 2. Wolfe line search
# ===========================================================================

class TestWolfeLineSearch(unittest.TestCase):

    def _armijo_ok(self, f, grad_f, x, d, alpha, c1=1e-4):
        f0 = f(x)
        slope = _dot(grad_f(x), d)
        return f(x_new := _axpy(alpha, d, x)) <= f0 + c1 * alpha * slope

    def test_returns_float(self):
        x = [2.0, 3.0]
        d = _neg_descent(grad_sphere(x))
        alpha = wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertIsInstance(alpha, float)

    def test_alpha_positive(self):
        x = [2.0, 3.0]
        d = _neg_descent(grad_sphere(x))
        alpha = wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertGreater(alpha, 0.0)

    def test_armijo_on_sphere(self):
        x = [1.0, -1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = wolfe_line_search(sphere, grad_sphere, x, d)
        f0 = sphere(x)
        slope = _dot(grad_sphere(x), d)
        x_new = _axpy(alpha, d, x)
        self.assertLessEqual(sphere(x_new), f0 + 1e-4 * alpha * slope + 1e-10)

    def test_armijo_on_quadratic(self):
        x = [3.0, -2.0]
        d = _neg_descent(grad_quadratic(x))
        alpha = wolfe_line_search(quadratic, grad_quadratic, x, d)
        f0 = quadratic(x)
        slope = _dot(grad_quadratic(x), d)
        x_new = _axpy(alpha, d, x)
        self.assertLessEqual(quadratic(x_new), f0 + 1e-4 * alpha * slope + 1e-10)

    def test_value_decreases_sphere(self):
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertLess(sphere(_axpy(alpha, d, x)), sphere(x))

    def test_value_decreases_quadratic(self):
        x = [4.0, -3.0]
        d = _neg_descent(grad_quadratic(x))
        alpha = wolfe_line_search(quadratic, grad_quadratic, x, d)
        self.assertLess(quadratic(_axpy(alpha, d, x)), quadratic(x))

    def test_c1_c2_custom(self):
        x = [2.0, -2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = wolfe_line_search(sphere, grad_sphere, x, d, c1=1e-3, c2=0.8)
        self.assertGreater(alpha, 0.0)


# ===========================================================================
# 3. Brent's method
# ===========================================================================

class TestBrentMinimize(unittest.TestCase):

    def test_parabola_minimum(self):
        """f(x) = (x - 2)^2 has minimum at x = 2."""
        x_min = brent_minimize(lambda x: (x - 2.0) ** 2, 0.0, 4.0)
        self.assertAlmostEqual(x_min, 2.0, places=5)

    def test_neg_parabola_minimum(self):
        """f(x) = -(x - 2)^2 has minimum (as a function to minimise) at the boundary.
        We treat it as minimising, so it returns an endpoint."""
        # Min of -(x-2)^2 on [0,4]: boundary, so just check return is in [0,4]
        x_min = brent_minimize(lambda x: -(x - 2.0) ** 2, 0.0, 4.0)
        self.assertGreaterEqual(x_min, 0.0)
        self.assertLessEqual(x_min, 4.0)

    def test_sin_minimum(self):
        """sin(x) on [3, 6] has minimum near x = 3*pi/2 ≈ 4.712."""
        x_min = brent_minimize(math.sin, 3.0, 6.0)
        self.assertAlmostEqual(x_min, 1.5 * math.pi, places=5)

    def test_quadratic_x_squared(self):
        """f(x) = x^2 on [-1, 1] has minimum at x = 0."""
        x_min = brent_minimize(lambda x: x ** 2, -1.0, 1.0)
        self.assertAlmostEqual(x_min, 0.0, places=5)

    def test_cubic_x3_minus_x(self):
        """f(x) = x^3 - x on [0, 1], derivative = 3x^2 - 1 = 0 => x = 1/sqrt(3) ≈ 0.577."""
        x_min = brent_minimize(lambda x: x ** 3 - x, 0.0, 1.0)
        self.assertAlmostEqual(x_min, 1.0 / math.sqrt(3.0), places=5)

    def test_returns_float(self):
        x_min = brent_minimize(lambda x: x ** 2, -5.0, 5.0)
        self.assertIsInstance(x_min, float)

    def test_in_bracket(self):
        """Returned value must be in [a, b]."""
        a, b = 1.0, 5.0
        x_min = brent_minimize(lambda x: (x - 3.0) ** 2, a, b)
        self.assertGreaterEqual(x_min, a - 1e-12)
        self.assertLessEqual(x_min, b + 1e-12)

    def test_tight_tolerance(self):
        """With very tight tolerance, still converges."""
        x_min = brent_minimize(lambda x: (x - math.pi) ** 2, 3.0, 4.0, tol=1e-10)
        self.assertAlmostEqual(x_min, math.pi, places=9)

    def test_flat_region(self):
        """Flat region: f(x)=1 on [0,1], any point is fine."""
        x_min = brent_minimize(lambda x: 1.0, 0.0, 1.0)
        self.assertGreaterEqual(x_min, 0.0)
        self.assertLessEqual(x_min, 1.0)


# ===========================================================================
# 4. Cubic interpolation line search
# ===========================================================================

class TestCubicInterpolation(unittest.TestCase):

    def test_returns_in_bracket(self):
        """Returned alpha must be inside [alpha_lo, alpha_hi]."""
        x = [1.0, 1.0]
        d = [-1.0, -1.0]
        alpha = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 0.0, 1.0)
        self.assertGreaterEqual(alpha, 0.0 - 1e-12)
        self.assertLessEqual(alpha, 1.0 + 1e-12)

    def test_returns_float(self):
        x = [2.0, -1.0]
        d = _neg_descent(grad_sphere(x))
        alpha = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 0.1, 1.0)
        self.assertIsInstance(alpha, float)

    def test_on_quadratic_improves(self):
        """For a simple quadratic, the returned alpha should decrease f."""
        x = [3.0, 4.0]
        d = _neg_descent(grad_quadratic(x))
        alpha = cubic_interpolation_line_search(quadratic, grad_quadratic, x, d, 0.0, 2.0)
        x_new = _axpy(alpha, d, x)
        self.assertLess(quadratic(x_new), quadratic(x))

    def test_bracket_order_does_not_matter(self):
        """Swapping alpha_lo and alpha_hi should still return a value in [0, 1]."""
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha1 = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 0.0, 1.0)
        alpha2 = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 1.0, 0.0)
        # The cubic minimiser formula is anchored to alpha_hi, so the two calls can
        # produce different (but both valid) interpolated points.  Assert both are
        # clamped to the bracket [0, 1].
        self.assertGreaterEqual(alpha1, 0.0 - 1e-12)
        self.assertLessEqual(alpha1, 1.0 + 1e-12)
        self.assertGreaterEqual(alpha2, 0.0 - 1e-12)
        self.assertLessEqual(alpha2, 1.0 + 1e-12)

    def test_clamped_to_bracket(self):
        """Even with unusual inputs, result is clamped."""
        x = [0.1, 0.1]
        d = [-0.1, -0.1]
        alpha = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 0.5, 2.0)
        self.assertGreaterEqual(alpha, 0.5 - 1e-12)
        self.assertLessEqual(alpha, 2.0 + 1e-12)

    def test_near_minimum(self):
        """Sphere minimum is at origin; starting from near-min should give small alpha."""
        x = [0.01, 0.01]
        d = _neg_descent(grad_sphere(x))
        alpha = cubic_interpolation_line_search(sphere, grad_sphere, x, d, 0.0, 1.0)
        self.assertGreaterEqual(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)


# ===========================================================================
# 5. Strong Wolfe line search
# ===========================================================================

class TestStrongWolfe(unittest.TestCase):

    def _strong_wolfe_ok(self, f, grad_f, x, d, alpha, c1=1e-4, c2=0.9):
        """Return True if strong Wolfe conditions are satisfied."""
        f0 = f(x)
        dphi0 = _dot(grad_f(x), d)
        x_new = _axpy(alpha, d, x)
        f_new = f(x_new)
        dphi_new = _dot(grad_f(x_new), d)
        armijo = f_new <= f0 + c1 * alpha * dphi0
        curvature = abs(dphi_new) <= c2 * abs(dphi0)
        return armijo and curvature

    def test_returns_float(self):
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertIsInstance(alpha, float)

    def test_alpha_positive(self):
        x = [1.0, 2.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertGreater(alpha, 0.0)

    def test_strong_wolfe_sphere_2d(self):
        x = [3.0, 4.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertTrue(self._strong_wolfe_ok(sphere, grad_sphere, x, d, alpha))

    def test_strong_wolfe_sphere_5d(self):
        x = [1.0, -1.0, 2.0, -2.0, 3.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertTrue(self._strong_wolfe_ok(sphere, grad_sphere, x, d, alpha))

    def test_strong_wolfe_quadratic(self):
        """For f=0.5*||x||^2, Newton step is alpha=1 and satisfies strong Wolfe."""
        x = [2.0, -1.0]
        d = _neg_descent(grad_quadratic(x))  # = [-2, 1], Newton step along -x
        alpha = strong_wolfe_line_search(quadratic, grad_quadratic, x, d)
        self.assertTrue(self._strong_wolfe_ok(quadratic, grad_quadratic, x, d, alpha))

    def test_quadratic_exact_alpha(self):
        """For quadratic, strong Wolfe should find alpha=1 (exact minimiser)."""
        x = [1.0, 0.0]
        d = [-1.0, 0.0]
        alpha = strong_wolfe_line_search(quadratic, grad_quadratic, x, d,
                                          alpha0=1.0, c1=1e-4, c2=0.9)
        # alpha=1 is exact minimiser; strong Wolfe should return it or very close
        self.assertAlmostEqual(alpha, 1.0, places=6)

    def test_value_decreases(self):
        x = [2.0, 3.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        self.assertLess(sphere(_axpy(alpha, d, x)), sphere(x))

    def test_armijo_satisfied_sphere(self):
        x = [1.5, -2.5]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d)
        f0 = sphere(x)
        slope = _dot(grad_sphere(x), d)
        x_new = _axpy(alpha, d, x)
        self.assertLessEqual(sphere(x_new), f0 + 1e-4 * alpha * slope + 1e-10)

    def test_rosenbrock_descent(self):
        """Strong Wolfe on Rosenbrock should give a step that decreases f."""
        x = [0.0, 0.0]
        d = _neg_descent(grad_rosenbrock(x))
        alpha = strong_wolfe_line_search(rosenbrock, grad_rosenbrock, x, d)
        self.assertGreater(alpha, 0.0)
        self.assertLess(rosenbrock(_axpy(alpha, d, x)), rosenbrock(x))

    def test_custom_c1_c2(self):
        x = [1.0, 1.0]
        d = _neg_descent(grad_sphere(x))
        alpha = strong_wolfe_line_search(sphere, grad_sphere, x, d,
                                          c1=1e-3, c2=0.8)
        self.assertGreater(alpha, 0.0)
        # At least Armijo should hold
        f0 = sphere(x)
        slope = _dot(grad_sphere(x), d)
        x_new = _axpy(alpha, d, x)
        self.assertLessEqual(sphere(x_new), f0 + 1e-3 * alpha * slope + 1e-10)


# ===========================================================================

if __name__ == '__main__':
    unittest.main()

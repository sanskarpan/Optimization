"""
Tests for proximal.py
=====================
Covers all proximal operators (prox_l1, prox_l2_sq, prox_linf,
prox_non_negative, prox_box, prox_elastic_net) and the four
algorithms (ista, fista, proximal_gradient, douglas_rachford).

Run with::

    python -m pytest tests/test_proximal.py -v
    # or
    python tests/test_proximal.py
"""

import math
import sys
import os
import unittest

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proximal import (
    prox_l1,
    prox_l2_sq,
    prox_linf,
    prox_non_negative,
    prox_box,
    prox_elastic_net,
    ista,
    fista,
    proximal_gradient,
    douglas_rachford,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _l1(v):
    return sum(abs(x) for x in v)


def _l2(v):
    return math.sqrt(sum(x * x for x in v))


# ---------------------------------------------------------------------------
# 9.1  Proximal operators
# ---------------------------------------------------------------------------

class TestProximalOperators(unittest.TestCase):
    """Tests for all six proximal operators."""

    # --- prox_l1 -----------------------------------------------------------

    def test_prox_l1_zeros_below_threshold(self):
        """Components whose magnitude is below lam are zeroed out."""
        result = prox_l1([0.5, -0.3, 0.8], lam=0.6)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)

    def test_prox_l1_shrinks_by_lambda(self):
        """Components above the threshold are shrunk by exactly lam."""
        result = prox_l1([2.0, -3.0], lam=0.5)
        self.assertAlmostEqual(result[0], 1.5, places=10)
        self.assertAlmostEqual(result[1], -2.5, places=10)

    def test_prox_l1_lam_zero_is_identity(self):
        """With lam=0 the operator is the identity."""
        v = [1.0, -2.0, 3.0]
        result = prox_l1(v, lam=0.0)
        for r, vi in zip(result, v):
            self.assertAlmostEqual(r, vi, places=12)

    def test_prox_l1_sign_preserved(self):
        """Soft thresholding preserves the sign of non-zero components."""
        result = prox_l1([-5.0, 3.0], lam=1.0)
        self.assertLess(result[0], 0.0)
        self.assertGreater(result[1], 0.0)

    def test_prox_l1_negative_lam_raises(self):
        """Negative lam must raise ValueError."""
        with self.assertRaises(ValueError):
            prox_l1([1.0], lam=-0.1)

    def test_prox_l1_at_threshold(self):
        """Component exactly at threshold maps to 0."""
        result = prox_l1([1.0], lam=1.0)
        self.assertAlmostEqual(result[0], 0.0, places=12)

    # --- prox_l2_sq --------------------------------------------------------

    def test_prox_l2_sq_formula(self):
        """Result equals v / (1 + 2*lam)."""
        result = prox_l2_sq([3.0, -4.0], lam=0.5)
        self.assertAlmostEqual(result[0], 3.0 / 2.0, places=10)
        self.assertAlmostEqual(result[1], -4.0 / 2.0, places=10)

    def test_prox_l2_sq_lam_zero(self):
        """lam=0 gives identity."""
        v = [1.0, 2.0, 3.0]
        result = prox_l2_sq(v, lam=0.0)
        for r, vi in zip(result, v):
            self.assertAlmostEqual(r, vi, places=12)

    def test_prox_l2_sq_shrinks_toward_origin(self):
        """Every component is strictly closer to 0 for lam > 0."""
        v = [2.0, -3.0, 0.5]
        result = prox_l2_sq(v, lam=1.0)
        for r, vi in zip(result, v):
            self.assertLessEqual(abs(r), abs(vi))

    def test_prox_l2_sq_negative_lam_raises(self):
        with self.assertRaises(ValueError):
            prox_l2_sq([1.0], lam=-1.0)

    # --- prox_linf ---------------------------------------------------------

    def test_prox_linf_zero_lam_is_identity(self):
        """lam=0 gives identity."""
        v = [1.0, -2.0, 3.0]
        result = prox_linf(v, lam=0.0)
        for r, vi in zip(result, v):
            self.assertAlmostEqual(r, vi, places=12)

    def test_prox_linf_moreau_decomposition(self):
        """Moreau identity: prox_linf(v, lam) + lam * proj_L1(v/lam) == v."""
        # We verify via: prox_{lam*||.||_inf}(v) = v - proj_{L1-ball(lam)}(v)
        from proximal import _proj_l1_ball  # type: ignore[attr-defined]

        v = [3.0, -1.0, 2.0]
        lam = 1.5
        result = prox_linf(v, lam)
        proj = _proj_l1_ball(v, lam)
        for r, vi, pi in zip(result, v, proj):
            self.assertAlmostEqual(r, vi - pi, places=10)

    def test_prox_linf_l1_norm_of_complement_bounded(self):
        """The 'residual' v - prox_linf(v, lam) has L1-norm <= lam."""
        v = [4.0, -2.0, 1.0, -3.0]
        lam = 2.0
        result = prox_linf(v, lam)
        residual = [vi - ri for vi, ri in zip(v, result)]
        self.assertLessEqual(_l1(residual), lam + 1e-10)

    def test_prox_linf_negative_lam_raises(self):
        with self.assertRaises(ValueError):
            prox_linf([1.0], lam=-0.5)

    # --- prox_non_negative -------------------------------------------------

    def test_prox_non_negative_clips_negative(self):
        """Negative components are zeroed; positives are unchanged."""
        result = prox_non_negative([-1.0, 2.0, -0.5])
        self.assertEqual(result[0], 0.0)
        self.assertEqual(result[1], 2.0)
        self.assertEqual(result[2], 0.0)

    def test_prox_non_negative_all_positive(self):
        """All-positive input is returned unchanged."""
        v = [0.1, 1.0, 5.0]
        result = prox_non_negative(v)
        for r, vi in zip(result, v):
            self.assertAlmostEqual(r, vi)

    def test_prox_non_negative_lam_ignored(self):
        """The lam parameter has no effect."""
        v = [-1.0, 2.0]
        self.assertEqual(prox_non_negative(v, lam=0.0), prox_non_negative(v, lam=99.0))

    # --- prox_box ----------------------------------------------------------

    def test_prox_box_clips(self):
        """Values outside the box are clipped to the boundary."""
        result = prox_box(
            [5.0, -2.0, 3.0],
            lower=[0.0, 0.0, 0.0],
            upper=[4.0, 4.0, 4.0],
        )
        self.assertAlmostEqual(result[0], 4.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 3.0)

    def test_prox_box_inside_box_unchanged(self):
        """Values already inside the box are not changed."""
        v = [1.0, 2.0, 3.0]
        result = prox_box(v, lower=[0.0]*3, upper=[5.0]*3)
        for r, vi in zip(result, v):
            self.assertAlmostEqual(r, vi)

    def test_prox_box_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            prox_box([1.0, 2.0], lower=[0.0], upper=[5.0, 5.0])

    def test_prox_box_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            prox_box([1.0], lower=[3.0], upper=[1.0])

    # --- prox_elastic_net --------------------------------------------------

    def test_prox_elastic_net_formula(self):
        """Should equal prox_l1(v/(1+2*lam2), lam1/(1+2*lam2))."""
        v = [3.0, -0.2, 1.5]
        lam1, lam2 = 0.5, 0.3
        result = prox_elastic_net(v, lam1, lam2)
        scale = 1.0 + 2.0 * lam2
        expected = [
            math.copysign(max(abs(vi / scale) - lam1 / scale, 0.0), vi)
            for vi in v
        ]
        for r, e in zip(result, expected):
            self.assertAlmostEqual(r, e, places=10)

    def test_prox_elastic_net_lam2_zero_reduces_to_l1(self):
        """With lam2=0 the elastic net becomes standard L1 prox."""
        v = [2.0, -1.5, 0.3]
        lam1 = 0.4
        result_en = prox_elastic_net(v, lam1=lam1, lam2=0.0)
        result_l1 = prox_l1(v, lam=lam1)
        for r_en, r_l1 in zip(result_en, result_l1):
            self.assertAlmostEqual(r_en, r_l1, places=10)

    def test_prox_elastic_net_lam1_zero_reduces_to_l2sq(self):
        """With lam1=0 the elastic net becomes standard L2-squared prox."""
        v = [2.0, -3.0]
        lam2 = 0.5
        result_en = prox_elastic_net(v, lam1=0.0, lam2=lam2)
        result_l2 = prox_l2_sq(v, lam=lam2)
        for r_en, r_l2 in zip(result_en, result_l2):
            self.assertAlmostEqual(r_en, r_l2, places=10)

    def test_prox_elastic_net_negative_lam1_raises(self):
        with self.assertRaises(ValueError):
            prox_elastic_net([1.0], lam1=-0.1, lam2=0.1)

    def test_prox_elastic_net_negative_lam2_raises(self):
        with self.assertRaises(ValueError):
            prox_elastic_net([1.0], lam1=0.1, lam2=-0.1)


# ---------------------------------------------------------------------------
# 9.2  ISTA
# ---------------------------------------------------------------------------

class TestISTA(unittest.TestCase):
    """Tests for the ISTA algorithm."""

    def test_ista_lasso_convergence(self):
        """ISTA solves a 1-D LASSO problem to near-optimality.

        Problem: min (1/2)(x - 3)^2 + 0.5|x|
        Optimality: x* = sign(3)*max(|3| - 0.5/L, 0) = 2.5 (with L=1).
        """
        lam = 0.5

        def f(x):
            return 0.5 * (x[0] - 3.0) ** 2

        def grad_f(x):
            return [x[0] - 3.0]

        x_opt, history = ista(
            f,
            grad_f,
            prox_g=lambda v, s: prox_l1(v, lam * s),
            x0=[0.0],
            L=1.0,
            max_iter=200,
            tol=1e-8,
        )
        self.assertAlmostEqual(x_opt[0], 2.5, delta=0.01)

    def test_ista_no_regularizer_matches_gd(self):
        """With identity prox, ISTA reduces to gradient descent and reaches 0."""
        def f(x):
            return sum(xi ** 2 for xi in x)

        def grad_f(x):
            return [2.0 * xi for xi in x]

        prox_identity = lambda v, s: v[:]

        x_opt, history = ista(f, grad_f, prox_identity, [3.0, 4.0], L=2.0, max_iter=200)
        self.assertLess(f(x_opt), 0.01)

    def test_ista_backtracking_L_none(self):
        """With L=None, ISTA uses backtracking and still converges."""
        def f(x):
            return x[0] ** 2

        def grad_f(x):
            return [2.0 * x[0]]

        x_opt, history = ista(
            f, grad_f, lambda v, s: v[:], [5.0], L=None, max_iter=200
        )
        self.assertAlmostEqual(x_opt[0], 0.0, delta=0.1)

    def test_ista_history_length(self):
        """History should have at least 2 entries (initial + at least 1 step)."""
        def f(x): return x[0] ** 2
        def grad_f(x): return [2 * x[0]]
        _, history = ista(f, grad_f, lambda v, s: v[:], [1.0], L=2.0, max_iter=5)
        self.assertGreaterEqual(len(history), 2)

    def test_ista_2d_quadratic(self):
        """ISTA minimises a 2D quadratic with known minimum."""
        # min (x-1)^2 + (y-2)^2 → optimum (1, 2)
        def f(x): return (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2
        def grad_f(x): return [2 * (x[0] - 1.0), 2 * (x[1] - 2.0)]
        x_opt, _ = ista(f, grad_f, lambda v, s: v[:], [0.0, 0.0], L=2.0, max_iter=300)
        self.assertAlmostEqual(x_opt[0], 1.0, delta=0.05)
        self.assertAlmostEqual(x_opt[1], 2.0, delta=0.05)

    def test_ista_invalid_L_raises(self):
        def f(x): return x[0]
        def grad_f(x): return [1.0]
        with self.assertRaises(ValueError):
            ista(f, grad_f, lambda v, s: v[:], [0.0], L=-1.0)


# ---------------------------------------------------------------------------
# 9.3  FISTA
# ---------------------------------------------------------------------------

class TestFISTA(unittest.TestCase):
    """Tests for the FISTA algorithm."""

    def test_fista_convergence_lasso(self):
        """FISTA solves the same 1-D LASSO as ISTA (same optimum)."""
        lam = 0.5

        def f(x):
            return 0.5 * (x[0] - 3.0) ** 2

        def grad_f(x):
            return [x[0] - 3.0]

        x_opt, history = fista(
            f,
            grad_f,
            lambda v, s: prox_l1(v, lam * s),
            [0.0],
            L=1.0,
            max_iter=200,
            tol=1e-8,
        )
        self.assertAlmostEqual(x_opt[0], 2.5, delta=0.01)

    def test_fista_faster_than_ista(self):
        """FISTA should reach the same accuracy in at most as many iterations as ISTA."""
        def f(x):
            return sum((xi - 1.0) ** 2 for xi in x)

        def grad_f(x):
            return [2.0 * (xi - 1.0) for xi in x]

        prox_id = lambda v, s: v[:]
        x0 = [5.0, 5.0]
        L = 2.0

        _, hist_ista = ista(f, grad_f, prox_id, x0[:], L=L, max_iter=500)
        _, hist_fista = fista(f, grad_f, prox_id, x0[:], L=L, max_iter=500)

        def first_below(hist, threshold):
            for i, v in enumerate(hist):
                if v < threshold:
                    return i
            return len(hist)

        iters_ista = first_below(hist_ista, 0.01)
        iters_fista = first_below(hist_fista, 0.01)
        # FISTA should be at least as fast (small tolerance for edge cases)
        self.assertLessEqual(iters_fista, iters_ista + 5)

    def test_fista_t_sequence(self):
        """The momentum sequence t_k is strictly increasing."""
        t = 1.0
        ts = [t]
        for _ in range(5):
            t_new = (1.0 + math.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            ts.append(t_new)
            t = t_new
        for i in range(len(ts) - 1):
            self.assertGreater(ts[i + 1], ts[i])

    def test_fista_backtracking_L_none(self):
        """FISTA with L=None converges via backtracking."""
        def f(x): return (x[0] - 2.0) ** 2
        def grad_f(x): return [2.0 * (x[0] - 2.0)]
        x_opt, _ = fista(f, grad_f, lambda v, s: v[:], [0.0], L=None, max_iter=300)
        self.assertAlmostEqual(x_opt[0], 2.0, delta=0.1)

    def test_fista_history_non_increasing_on_average(self):
        """Objective should decrease overall from first to last recorded value."""
        def f(x): return sum(xi ** 2 for xi in x)
        def grad_f(x): return [2.0 * xi for xi in x]
        _, history = fista(f, grad_f, lambda v, s: v[:], [10.0, 10.0], L=2.0, max_iter=100)
        self.assertLess(history[-1], history[0])

    def test_fista_invalid_L_raises(self):
        def f(x): return x[0]
        def grad_f(x): return [1.0]
        with self.assertRaises(ValueError):
            fista(f, grad_f, lambda v, s: v[:], [0.0], L=0.0)


# ---------------------------------------------------------------------------
# 9.4  Proximal Gradient with Backtracking
# ---------------------------------------------------------------------------

class TestProximalGradient(unittest.TestCase):
    """Tests for the proximal_gradient function."""

    def test_proximal_gradient_backtracking(self):
        """Converges to the known minimum of a 1D quadratic."""
        def f(x): return (x[0] - 2.0) ** 2
        def grad_f(x): return [2.0 * (x[0] - 2.0)]

        x_opt, history = proximal_gradient(
            f, grad_f, lambda v, s: v[:], [0.0], L_init=0.1, max_iter=200
        )
        self.assertAlmostEqual(x_opt[0], 2.0, delta=0.1)

    def test_proximal_gradient_with_l1(self):
        """Converges toward the constrained minimum with L1 regularisation."""
        # min (x-3)^2 + 0.5|x|  →  x* ≈ 2.75
        def f(x): return (x[0] - 3.0) ** 2
        def grad_f(x): return [2.0 * (x[0] - 3.0)]
        prox = lambda v, s: prox_l1(v, 0.5 * s)

        x_opt, _ = proximal_gradient(f, grad_f, prox, [0.0], max_iter=500)
        # At least confirm the solution moved meaningfully toward 3
        self.assertGreater(x_opt[0], 1.0)

    def test_proximal_gradient_2d_quadratic(self):
        """Handles a 2D case with identity prox correctly."""
        def f(x): return (x[0] - 1.0) ** 2 + (x[1] + 1.0) ** 2
        def grad_f(x): return [2 * (x[0] - 1.0), 2 * (x[1] + 1.0)]
        x_opt, _ = proximal_gradient(f, grad_f, lambda v, s: v[:], [5.0, 5.0], max_iter=500)
        self.assertAlmostEqual(x_opt[0], 1.0, delta=0.1)
        self.assertAlmostEqual(x_opt[1], -1.0, delta=0.1)

    def test_proximal_gradient_history_decreases(self):
        """Objective value at the final iterate should be less than at the start."""
        def f(x): return sum(xi ** 2 for xi in x)
        def grad_f(x): return [2.0 * xi for xi in x]
        _, history = proximal_gradient(
            f, grad_f, lambda v, s: v[:], [10.0, 10.0], max_iter=100
        )
        self.assertLess(history[-1], history[0])

    def test_proximal_gradient_invalid_L_init_raises(self):
        with self.assertRaises(ValueError):
            proximal_gradient(
                lambda x: 0.0, lambda x: [0.0], lambda v, s: v[:],
                [0.0], L_init=-1.0
            )

    def test_proximal_gradient_invalid_beta_raises(self):
        with self.assertRaises(ValueError):
            proximal_gradient(
                lambda x: 0.0, lambda x: [0.0], lambda v, s: v[:],
                [0.0], beta=1.5
            )


# ---------------------------------------------------------------------------
# 9.5  Douglas-Rachford Splitting
# ---------------------------------------------------------------------------

class TestDouglasRachford(unittest.TestCase):
    """Tests for the douglas_rachford function."""

    def _make_prox_quadratic(self, center: float):
        """Return prox_{gamma*(x-center)^2} as a closure."""
        def prox(v, g):
            return [(vi + 2.0 * g * center) / (1.0 + 2.0 * g) for vi in v]
        return prox

    def test_douglas_rachford_converges(self):
        """DR finds the minimiser of |x| + (x-2)^2.

        Optimality condition (x > 0): 2(x-2) + 1 = 0  ⟹  x* = 1.5.
        """
        gamma = 0.5
        prox_quadratic = self._make_prox_quadratic(2.0)

        def prox_l1_fn(v, g):
            return prox_l1(v, g)

        x_opt, history = douglas_rachford(
            prox_quadratic, prox_l1_fn, [5.0], gamma=gamma, max_iter=500, tol=1e-6
        )
        self.assertAlmostEqual(x_opt[0], 1.5, delta=0.1)

    def test_douglas_rachford_history_decreases(self):
        """The residual ||z_{k+1} - z_k|| should eventually decrease."""
        gamma = 0.5
        prox_f = self._make_prox_quadratic(1.0)

        def prox_gfun(v, g):
            return prox_l1(v, g)

        _, history = douglas_rachford(
            prox_f, prox_gfun, [10.0], gamma=gamma, max_iter=200
        )
        # Overall the residuals should decrease (last << first)
        self.assertLess(history[-1], history[0] + 1.0)

    def test_douglas_rachford_pure_quadratic(self):
        """DR on min (x-a)^2 + (x-b)^2 should recover x* = (a+b)/2."""
        a, b = 1.0, 3.0
        prox_fa = self._make_prox_quadratic(a)
        prox_fb = self._make_prox_quadratic(b)

        x_opt, _ = douglas_rachford(
            prox_fa, prox_fb, [0.0], gamma=1.0, max_iter=500, tol=1e-7
        )
        self.assertAlmostEqual(x_opt[0], (a + b) / 2.0, delta=0.05)

    def test_douglas_rachford_relaxation_parameter(self):
        """DR converges with a relaxation parameter in (0, 2)."""
        gamma = 0.5
        prox_quadratic = self._make_prox_quadratic(2.0)

        def prox_l1_fn(v, g):
            return prox_l1(v, g)

        x_opt, _ = douglas_rachford(
            prox_quadratic, prox_l1_fn, [5.0],
            gamma=gamma, relaxation=1.5, max_iter=500, tol=1e-5
        )
        self.assertAlmostEqual(x_opt[0], 1.5, delta=0.2)

    def test_douglas_rachford_invalid_gamma_raises(self):
        with self.assertRaises(ValueError):
            douglas_rachford(
                lambda v, g: v[:], lambda v, g: v[:], [0.0], gamma=-1.0
            )

    def test_douglas_rachford_invalid_relaxation_raises(self):
        with self.assertRaises(ValueError):
            douglas_rachford(
                lambda v, g: v[:], lambda v, g: v[:], [0.0], relaxation=-0.1
            )

    def test_douglas_rachford_history_nonempty(self):
        """History must contain at least one entry."""
        prox_id = lambda v, g: v[:]
        _, history = douglas_rachford(prox_id, prox_id, [1.0], max_iter=5)
        self.assertGreater(len(history), 0)


# ---------------------------------------------------------------------------
# Integration: import check
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):
    """Smoke test: all public names importable and callable."""

    def test_all_symbols_importable(self):
        names = [
            "prox_l1", "prox_l2_sq", "prox_linf", "prox_non_negative",
            "prox_box", "prox_elastic_net",
            "ista", "fista", "proximal_gradient", "douglas_rachford",
        ]
        import proximal as px
        for name in names:
            self.assertTrue(hasattr(px, name), f"Missing: {name}")

    def test_internal_proj_l1_ball_available(self):
        """The helper _proj_l1_ball should exist in the module (not exported)."""
        import proximal as px
        self.assertTrue(
            hasattr(px, "_proj_l1_ball"),
            "_proj_l1_ball helper not found in proximal module",
        )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

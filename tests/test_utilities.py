"""
Unit tests for utilities.py.

Covers every public symbol in the module:

* Gradient / Jacobian checking (check_gradient, check_jacobian)
* Numerical derivatives (numerical_gradient, numerical_hessian)
* Standard test functions (sphere, rosenbrock, rastrigin, ackley,
  himmelblau, beale, booth, styblinski_tang)
* Callbacks (EarlyStopping, GradientMonitor, LossLogger,
  DivergenceDetector, CallbackList)
* Checkpointing (save_state / load_state round-trip)
* NaN/Inf guards (check_finite)
* Benchmarking framework (benchmark_optimizer, compare_optimizers)
"""

import math
import os
import sys
import tempfile
import unittest

# ---------------------------------------------------------------------------
# Make the parent package importable regardless of how the test is invoked.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilities import (
    BenchmarkResult,
    Callback,
    CallbackList,
    DivergenceDetector,
    EarlyStopping,
    GradientMonitor,
    LossLogger,
    ackley,
    ackley_grad,
    beale,
    beale_grad,
    benchmark_optimizer,
    booth,
    booth_grad,
    check_finite,
    check_gradient,
    check_jacobian,
    compare_optimizers,
    himmelblau,
    himmelblau_grad,
    load_state,
    numerical_gradient,
    numerical_hessian,
    rastrigin,
    rastrigin_grad,
    rosenbrock,
    rosenbrock_grad,
    safe_step,
    save_state,
    sphere,
    sphere_grad,
    styblinski_tang,
)


# ===========================================================================
# Helper: a tiny gradient-descent optimizer for benchmarking tests
# ===========================================================================


def _simple_gd(
    f,
    grad_f,
    x0,
    lr: float = 0.01,
    max_iter: int = 200,
):
    """Vanilla gradient descent used only inside tests."""
    x = list(x0)
    history = []
    for _ in range(max_iter):
        g = grad_f(x)
        x = [xi - lr * gi for xi, gi in zip(x, g)]
        history.append(f(x))
    return x, history


# ===========================================================================
# 1.  TestGradientChecking
# ===========================================================================


class TestGradientChecking(unittest.TestCase):
    """Tests for check_gradient and check_jacobian."""

    def test_check_gradient_correct(self):
        """Sphere gradient passes with rtol=1e-3."""
        x = [1.0, 2.0, 3.0]
        max_abs_err, rel_err, passed = check_gradient(
            sphere, sphere_grad, x, h=1e-5, rtol=1e-3
        )
        self.assertTrue(
            passed,
            msg=f"Expected check to pass; rel_err={rel_err:.2e}",
        )
        self.assertLess(max_abs_err, 1e-6)

    def test_check_gradient_wrong(self):
        """A deliberately wrong gradient (all-ones instead of 2x) must fail."""
        x = [1.0, 2.0, 3.0]

        def wrong_grad(x):
            return [1.0] * len(x)

        _, _, passed = check_gradient(sphere, wrong_grad, x, rtol=1e-3)
        self.assertFalse(
            passed,
            msg="Expected check to fail for wrong gradient",
        )

    def test_check_jacobian_linear(self):
        """f(x) = [x0+x1, x0-x1] has exact Jacobian [[1,1],[1,-1]]."""

        def f_vec(x):
            return [x[0] + x[1], x[0] - x[1]]

        def jac(x):
            return [[1.0, 1.0], [1.0, -1.0]]

        x = [2.0, 3.0]
        jac_a, jac_fd, max_err = check_jacobian(f_vec, jac, x, h=1e-5)

        self.assertLess(max_err, 1e-7, msg=f"max_abs_err={max_err:.2e}")
        # Verify shape
        self.assertEqual(len(jac_fd), 2)
        self.assertEqual(len(jac_fd[0]), 2)
        # Verify values match expected Jacobian
        expected = [[1.0, 1.0], [1.0, -1.0]]
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(jac_fd[i][j], expected[i][j], places=5)


# ===========================================================================
# 2.  TestNumericalDerivatives
# ===========================================================================


class TestNumericalDerivatives(unittest.TestCase):
    """Tests for numerical_gradient and numerical_hessian."""

    def test_numerical_gradient_sphere(self):
        """Numerical gradient of sphere matches sphere_grad at [1, 2, 3]."""
        x = [1.0, 2.0, 3.0]
        ng = numerical_gradient(sphere, x, h=1e-5)
        ag = sphere_grad(x)
        for i, (ngi, agi) in enumerate(zip(ng, ag)):
            self.assertAlmostEqual(
                ngi,
                agi,
                places=5,
                msg=f"Component {i}: numerical={ngi}, analytical={agi}",
            )

    def test_numerical_hessian_sphere(self):
        """Hessian of sphere is 2*I (within 1e-4 tolerance)."""
        x = [1.0, 2.0, 3.0]
        H = numerical_hessian(sphere, x, h=1e-4)
        n = len(x)
        for i in range(n):
            for j in range(n):
                expected = 2.0 if i == j else 0.0
                self.assertAlmostEqual(
                    H[i][j],
                    expected,
                    delta=1e-4,
                    msg=f"H[{i}][{j}]: got {H[i][j]}, expected {expected}",
                )

    def test_numerical_hessian_symmetric(self):
        """Numerical Hessian of Rosenbrock is symmetric: H[i][j] == H[j][i]."""
        x = [1.5, 0.8]
        H = numerical_hessian(rosenbrock, x, h=1e-4)
        n = len(x)
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(
                    H[i][j],
                    H[j][i],
                    places=5,
                    msg=f"Hessian not symmetric at [{i}][{j}]",
                )


# ===========================================================================
# 3.  TestTestFunctions
# ===========================================================================


class TestTestFunctions(unittest.TestCase):
    """Tests for all standard benchmark functions and their gradients."""

    # -- sphere --------------------------------------------------------------

    def test_sphere_minimum(self):
        """sphere([0, 0, 0]) == 0."""
        self.assertEqual(sphere([0.0, 0.0, 0.0]), 0.0)

    def test_sphere_grad(self):
        """sphere_grad([1, 2]) == [2, 4]."""
        g = sphere_grad([1.0, 2.0])
        self.assertAlmostEqual(g[0], 2.0)
        self.assertAlmostEqual(g[1], 4.0)

    # -- rosenbrock ----------------------------------------------------------

    def test_rosenbrock_minimum(self):
        """rosenbrock([1, 1]) == 0."""
        self.assertAlmostEqual(rosenbrock([1.0, 1.0]), 0.0, places=12)

    def test_rosenbrock_grad_at_ones(self):
        """rosenbrock_grad([1, 1]) == [0, 0]."""
        g = rosenbrock_grad([1.0, 1.0])
        self.assertAlmostEqual(g[0], 0.0, places=12)
        self.assertAlmostEqual(g[1], 0.0, places=12)

    # -- rastrigin -----------------------------------------------------------

    def test_rastrigin_minimum(self):
        """rastrigin([0, 0]) == 0."""
        self.assertAlmostEqual(rastrigin([0.0, 0.0]), 0.0, places=12)

    # -- ackley --------------------------------------------------------------

    def test_ackley_minimum(self):
        """ackley([0, 0]) < 1e-10."""
        val = ackley([0.0, 0.0])
        self.assertLess(
            val,
            1e-10,
            msg=f"ackley at origin should be ~0, got {val}",
        )

    # -- himmelblau ----------------------------------------------------------

    def test_himmelblau_minima(self):
        """Himmelblau has four global minima, each with f ≈ 0."""
        minima = [
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ]
        for pt in minima:
            val = himmelblau(pt)
            self.assertAlmostEqual(
                val,
                0.0,
                places=4,
                msg=f"himmelblau at {pt} = {val}, expected ≈ 0",
            )

    # -- beale ---------------------------------------------------------------

    def test_beale_minimum(self):
        """beale([3, 0.5]) ≈ 0."""
        val = beale([3.0, 0.5])
        self.assertAlmostEqual(val, 0.0, places=10)

    # -- booth ---------------------------------------------------------------

    def test_booth_minimum(self):
        """booth([1, 3]) == 0."""
        self.assertAlmostEqual(booth([1.0, 3.0]), 0.0, places=12)

    # -- styblinski-tang -----------------------------------------------------

    def test_styblinski_minimum(self):
        """styblinski_tang([-2.9035, -2.9035]) ≈ -78.332."""
        val = styblinski_tang([-2.9035, -2.9035])
        self.assertAlmostEqual(val, -78.332, delta=0.01)


# ===========================================================================
# 4.  TestCallbacks
# ===========================================================================


class TestCallbacks(unittest.TestCase):
    """Tests for the callback system."""

    def test_early_stopping_triggers(self):
        """EarlyStopping stops after patience=2 steps with no improvement."""
        cb = EarlyStopping(patience=2, min_delta=1e-6, mode="min")
        dummy_x = [0.0]

        # Step 0 — sets best
        cb.on_step_end(0, dummy_x, 1.0)
        self.assertFalse(cb.should_stop())

        # Steps 1 & 2 — no improvement → wait reaches patience
        cb.on_step_end(1, dummy_x, 1.0)
        self.assertFalse(cb.should_stop())  # wait == 1

        cb.on_step_end(2, dummy_x, 1.0)
        self.assertTrue(
            cb.should_stop(),
            msg="should_stop() should be True after patience=2 non-improvements",
        )

    def test_early_stopping_no_trigger(self):
        """EarlyStopping does not stop when the objective keeps improving."""
        cb = EarlyStopping(patience=3, min_delta=1e-6, mode="min")
        dummy_x = [0.0]
        for step, fval in enumerate([5.0, 4.0, 3.0, 2.0, 1.0]):
            cb.on_step_end(step, dummy_x, fval)
        self.assertFalse(
            cb.should_stop(),
            msg="should_stop() should be False when improving every step",
        )

    def test_loss_logger_records(self):
        """LossLogger correctly records f values in self.losses."""
        cb = LossLogger(log_every=0)
        dummy_x = [0.0]
        expected = [3.0, 2.0, 1.0]
        for step, fval in enumerate(expected):
            cb.on_step_end(step, dummy_x, fval)
        self.assertEqual(cb.losses, expected)

    def test_gradient_monitor_norms(self):
        """GradientMonitor records correct L2 norms for each step."""
        cb = GradientMonitor()
        grads = [[3.0, 4.0], [0.0, 1.0], [1.0, 0.0]]
        expected_norms = [5.0, 1.0, 1.0]
        dummy_x = [0.0, 0.0]
        for step, g in enumerate(grads):
            cb.on_step_begin(step, dummy_x, g)

        for i, (got, exp) in enumerate(zip(cb.norms, expected_norms)):
            self.assertAlmostEqual(
                got,
                exp,
                places=10,
                msg=f"norm[{i}]: got {got}, expected {exp}",
            )
        self.assertAlmostEqual(cb.mean_norm, sum(expected_norms) / 3, places=10)
        self.assertAlmostEqual(cb.max_norm, 5.0, places=10)

    def test_divergence_detector_raises(self):
        """DivergenceDetector raises RuntimeError when f = 1e11 > threshold."""
        cb = DivergenceDetector(threshold=1e10)
        with self.assertRaises(RuntimeError) as ctx:
            cb.on_step_end(0, [0.0], 1e11)
        self.assertIn("Divergence detected", str(ctx.exception))
        # Python may render 1e11 as "1e+11" or "100000000000.0" depending
        # on the platform; check the step number and the key phrase instead.
        self.assertIn("step 0", str(ctx.exception))

    def test_callback_list_delegates(self):
        """CallbackList correctly forwards calls to both sub-callbacks."""
        logger1 = LossLogger()
        logger2 = LossLogger()
        cb_list = CallbackList([logger1, logger2])

        dummy_x = [0.0]
        cb_list.on_step_begin(0, dummy_x, [1.0])  # GradientMonitor-style
        cb_list.on_step_end(0, dummy_x, 7.5)
        cb_list.on_step_end(1, dummy_x, 4.2)

        self.assertEqual(logger1.losses, [7.5, 4.2])
        self.assertEqual(logger2.losses, [7.5, 4.2])
        self.assertFalse(cb_list.should_stop())


# ===========================================================================
# 5.  TestCheckpointing
# ===========================================================================


class _MockOptimizer:
    """Minimal mock optimizer that supports get_state / load_state."""

    def __init__(self) -> None:
        self.lr: float = 0.01
        self.iterations: int = 0
        self.velocity: list = [0.1, -0.2, 0.3]

    def get_state(self):
        """Return a JSON-serialisable state dict."""
        return {
            "lr": self.lr,
            "iterations": self.iterations,
            "velocity": list(self.velocity),
        }

    def load_state(self, state: dict) -> None:
        """Restore state from a dict."""
        self.lr = state["lr"]
        self.iterations = state["iterations"]
        self.velocity = state["velocity"]

    def step(self, params, grads):
        """Trivial step: subtract grads (used in safe_step tests)."""
        return [p - g for p, g in zip(params, grads)]


class TestCheckpointing(unittest.TestCase):
    """Tests for save_state / load_state and NaN/Inf guards."""

    def test_save_load_roundtrip(self):
        """save_state then load_state restores the optimizer state exactly."""
        opt = _MockOptimizer()
        opt.lr = 0.001
        opt.iterations = 42
        opt.velocity = [1.5, -3.0, 0.0]

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            save_state(opt, tmp_path)

            # Create a fresh optimizer and restore into it.
            opt2 = _MockOptimizer()
            opt2.lr = 99.0
            opt2.iterations = 0
            opt2.velocity = [0.0, 0.0, 0.0]

            load_state(opt2, tmp_path)

            self.assertAlmostEqual(opt2.lr, 0.001, places=10)
            self.assertEqual(opt2.iterations, 42)
            self.assertEqual(opt2.velocity, [1.5, -3.0, 0.0])
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_nan_guard_raises(self):
        """check_finite raises ValueError on [1.0, float('nan'), 2.0]."""
        with self.assertRaises(ValueError) as ctx:
            check_finite([1.0, float("nan"), 2.0])
        self.assertIn("[1]", str(ctx.exception))

    def test_nan_guard_inf_raises(self):
        """check_finite raises ValueError on [float('inf')]."""
        with self.assertRaises(ValueError) as ctx:
            check_finite([float("inf")])
        self.assertIn("[0]", str(ctx.exception))

    def test_nan_guard_passes_clean(self):
        """check_finite([1.0, 2.0]) raises no exception."""
        try:
            check_finite([1.0, 2.0])
        except ValueError as exc:
            self.fail(f"check_finite raised unexpectedly: {exc}")

    def test_safe_step_clean(self):
        """safe_step returns updated params when everything is finite."""
        opt = _MockOptimizer()
        result = safe_step(opt, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        self.assertEqual(len(result), 3)
        for r in result:
            self.assertTrue(math.isfinite(r))

    def test_safe_step_nan_params_raises(self):
        """safe_step raises ValueError when params contain NaN."""
        opt = _MockOptimizer()
        with self.assertRaises(ValueError):
            safe_step(opt, [float("nan"), 1.0], [0.0, 0.0])

    def test_safe_step_nan_grads_raises(self):
        """safe_step raises ValueError when grads contain Inf."""
        opt = _MockOptimizer()
        with self.assertRaises(ValueError):
            safe_step(opt, [1.0, 2.0], [0.0, float("inf")])


# ===========================================================================
# 6.  TestBenchmark
# ===========================================================================


class TestBenchmark(unittest.TestCase):
    """Tests for benchmark_optimizer and compare_optimizers."""

    # Common set of test functions (sphere only, for speed)
    _test_fns = [
        ("sphere", sphere, sphere_grad, [1.0, 1.0]),
    ]

    def _make_optimizer(self, lr: float = 0.1):
        """Return a simple GD wrapper that matches the required signature."""
        def opt_fn(f, grad_f, x0):
            return _simple_gd(f, grad_f, x0, lr=lr, max_iter=100)
        return opt_fn

    def test_benchmark_returns_keys(self):
        """benchmark_optimizer result has required keys for each function."""
        opt_fn = self._make_optimizer()
        results = benchmark_optimizer(opt_fn, self._test_fns, n_runs=3, tol=1.0)

        required_keys = {"mean_f", "std_f", "mean_iters", "success_rate", "all_f_values"}
        self.assertIn("sphere", results)
        sphere_result = results["sphere"]
        self.assertTrue(
            required_keys.issubset(sphere_result.keys()),
            msg=f"Missing keys: {required_keys - sphere_result.keys()}",
        )
        # Sanity-check types
        self.assertIsInstance(sphere_result["mean_f"], float)
        self.assertIsInstance(sphere_result["std_f"], float)
        self.assertIsInstance(sphere_result["mean_iters"], float)
        self.assertIsInstance(sphere_result["success_rate"], float)
        self.assertIsInstance(sphere_result["all_f_values"], list)
        self.assertEqual(len(sphere_result["all_f_values"]), 3)

    def test_compare_optimizers_shape(self):
        """compare_optimizers returns a nested dict with the right structure."""
        optimizers = {
            "gd_fast": self._make_optimizer(lr=0.1),
            "gd_slow": self._make_optimizer(lr=0.01),
        }
        results = compare_optimizers(optimizers, self._test_fns)

        # Outer keys: optimizer names
        self.assertIn("gd_fast", results)
        self.assertIn("gd_slow", results)

        # Inner keys: function names
        for opt_name in ("gd_fast", "gd_slow"):
            self.assertIn("sphere", results[opt_name])
            inner = results[opt_name]["sphere"]
            # Each inner value must be a BenchmarkResult dict
            self.assertIsInstance(inner, dict)
            self.assertIn("mean_f", inner)
            self.assertIn("success_rate", inner)

    def test_benchmark_success_rate_sphere(self):
        """Gradient descent on sphere with lr=0.1 achieves 100% success."""
        opt_fn = self._make_optimizer(lr=0.1)
        results = benchmark_optimizer(
            opt_fn, self._test_fns, n_runs=5, tol=1e-4
        )
        rate = results["sphere"]["success_rate"]
        self.assertAlmostEqual(rate, 1.0, places=5)

    def test_benchmark_mean_iters(self):
        """mean_iters equals the number of history entries returned."""
        opt_fn = self._make_optimizer()
        results = benchmark_optimizer(opt_fn, self._test_fns, n_runs=2)
        # _simple_gd runs exactly 100 steps, so history has 100 entries.
        self.assertAlmostEqual(results["sphere"]["mean_iters"], 100.0, places=5)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    unittest.main()

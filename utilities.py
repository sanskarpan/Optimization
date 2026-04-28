"""
Utilities — gradient checking, numerical derivatives, test functions,
benchmarking, callbacks, checkpointing, NaN/Inf guards.

This module provides a comprehensive toolkit that complements the core
optimization algorithms in the Phase0_Core/Optimization library:

* Gradient / Jacobian verification via finite differences
* Numerical gradient and Hessian approximation
* Classic benchmark / test functions with analytical gradients
* A benchmarking framework for comparing optimizers
* An event-driven callback system (early stopping, logging, etc.)
* JSON-based optimizer state checkpointing
* NaN/Inf guards for safe parameter updates
"""

import math
import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 1.1  Gradient / Jacobian Checking
# ---------------------------------------------------------------------------


def check_gradient(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    h: float = 1e-5,
    rtol: float = 1e-3,
) -> Tuple[float, float, bool]:
    """Verify an analytical gradient against a central-difference estimate.

    For each component *i*, the finite-difference approximation is::

        fd_i = (f(x + h*e_i) - f(x - h*e_i)) / (2 * h)

    Parameters
    ----------
    f:
        Scalar-valued objective function ``f(x) -> float``.
    grad_f:
        Analytical gradient function ``grad_f(x) -> List[float]``.
    x:
        Point at which the check is performed.
    h:
        Step size for finite differences (default ``1e-5``).
    rtol:
        Relative-error tolerance; the check passes when
        ``rel_err < rtol`` (default ``1e-3``).

    Returns
    -------
    max_abs_err:
        Maximum absolute difference ``max |grad_f(x)[i] - fd[i]|``.
    rel_err:
        Relative error ``max_abs_err / (max(|fd|) + 1e-8)``.
    passed:
        ``True`` when ``rel_err < rtol``.
    """
    n = len(x)
    analytical = grad_f(x)
    fd = []
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        fd.append((f(x_plus) - f(x_minus)) / (2.0 * h))

    max_abs_err = max(abs(analytical[i] - fd[i]) for i in range(n))
    max_fd = max(abs(v) for v in fd)
    rel_err = max_abs_err / (max_fd + 1e-8)
    passed = rel_err < rtol
    return max_abs_err, rel_err, passed


def check_jacobian(
    f_vec: Callable[[List[float]], List[float]],
    jac_f: Callable[[List[float]], List[List[float]]],
    x: List[float],
    h: float = 1e-5,
) -> Tuple[List[List[float]], List[List[float]], float]:
    """Verify an analytical Jacobian against a central-difference estimate.

    For a function ``f_vec : R^n -> R^m``, the finite-difference Jacobian
    element is::

        J_fd[i][j] = (f_vec(x + h*e_j)[i] - f_vec(x - h*e_j)[i]) / (2 * h)

    Parameters
    ----------
    f_vec:
        Vector-valued function ``f_vec(x) -> List[float]``.
    jac_f:
        Analytical Jacobian ``jac_f(x) -> List[List[float]]``
        of shape ``(m, n)``.
    x:
        Point at which the check is performed (length *n*).
    h:
        Step size for finite differences (default ``1e-5``).

    Returns
    -------
    jac_analytical:
        Result of ``jac_f(x)``, shape ``(m, n)``.
    jac_fd:
        Finite-difference Jacobian, shape ``(m, n)``.
    max_abs_err:
        Element-wise maximum absolute difference.
    """
    n = len(x)
    f0 = f_vec(x)
    m = len(f0)

    # Build FD Jacobian column by column (perturb e_j).
    jac_fd: List[List[float]] = [[0.0] * n for _ in range(m)]
    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[j] += h
        x_minus[j] -= h
        fp = f_vec(x_plus)
        fm = f_vec(x_minus)
        for i in range(m):
            jac_fd[i][j] = (fp[i] - fm[i]) / (2.0 * h)

    jac_analytical = jac_f(x)

    max_abs_err = 0.0
    for i in range(m):
        for j in range(n):
            diff = abs(jac_analytical[i][j] - jac_fd[i][j])
            if diff > max_abs_err:
                max_abs_err = diff

    return jac_analytical, jac_fd, max_abs_err


# ---------------------------------------------------------------------------
# 1.2  Numerical Derivatives
# ---------------------------------------------------------------------------


def numerical_gradient(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-5,
) -> List[float]:
    """Compute the gradient of *f* at *x* via central differences.

    Each component is approximated as::

        g_i = (f(x + h*e_i) - f(x - h*e_i)) / (2 * h)

    Parameters
    ----------
    f:
        Scalar-valued function.
    x:
        Point at which to evaluate the gradient.
    h:
        Finite-difference step size (default ``1e-5``).

    Returns
    -------
    List[float]
        Numerical gradient, same length as *x*.
    """
    n = len(x)
    grad = []
    for i in range(n):
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2.0 * h))
    return grad


def numerical_hessian(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-4,
) -> List[List[float]]:
    """Compute the Hessian of *f* at *x* via central differences.

    Diagonal elements use the second-order formula::

        H[i][i] = (f(x + h*e_i) - 2*f(x) + f(x - h*e_i)) / h^2

    Off-diagonal elements use the mixed partial formula::

        H[i][j] = (f(x+h*ei+h*ej) - f(x+h*ei-h*ej)
                   - f(x-h*ei+h*ej) + f(x-h*ei-h*ej)) / (4 * h^2)

    Parameters
    ----------
    f:
        Scalar-valued function.
    x:
        Point at which to evaluate the Hessian.
    h:
        Finite-difference step size (default ``1e-4``).

    Returns
    -------
    List[List[float]]
        Symmetric *n x n* Hessian matrix.
    """
    n = len(x)
    f0 = f(x)
    H: List[List[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        # Diagonal
        x_plus = list(x)
        x_minus = list(x)
        x_plus[i] += h
        x_minus[i] -= h
        H[i][i] = (f(x_plus) - 2.0 * f0 + f(x_minus)) / (h * h)

        # Off-diagonal
        for j in range(i + 1, n):
            xpp = list(x); xpp[i] += h; xpp[j] += h
            xpm = list(x); xpm[i] += h; xpm[j] -= h
            xmp = list(x); xmp[i] -= h; xmp[j] += h
            xmm = list(x); xmm[i] -= h; xmm[j] -= h
            val = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4.0 * h * h)
            H[i][j] = val
            H[j][i] = val

    return H


# ---------------------------------------------------------------------------
# 1.3  Standard Test Functions
# ---------------------------------------------------------------------------


def sphere(x: List[float]) -> float:
    """Sphere function: ``f(x) = sum(xi^2)``.

    Global minimum is 0 at the origin.

    Parameters
    ----------
    x:
        Input vector of any dimension.

    Returns
    -------
    float
        Function value.
    """
    return sum(xi * xi for xi in x)


def sphere_grad(x: List[float]) -> List[float]:
    """Analytical gradient of the sphere function.

    ``grad_i = 2 * x_i``

    Parameters
    ----------
    x:
        Input vector.

    Returns
    -------
    List[float]
        Gradient vector.
    """
    return [2.0 * xi for xi in x]


def rosenbrock(x: List[float]) -> float:
    """Rosenbrock (banana) function.

    ``f(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]``

    Global minimum is 0 at the all-ones vector.

    Parameters
    ----------
    x:
        Input vector of dimension >= 2.

    Returns
    -------
    float
        Function value.
    """
    total = 0.0
    for i in range(len(x) - 1):
        total += 100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1.0 - x[i]) ** 2
    return total


def rosenbrock_grad(x: List[float]) -> List[float]:
    """Analytical gradient of the Rosenbrock function.

    Parameters
    ----------
    x:
        Input vector of dimension >= 2.

    Returns
    -------
    List[float]
        Gradient vector.
    """
    n = len(x)
    grad = [0.0] * n
    for i in range(n - 1):
        # derivative w.r.t. x[i]
        grad[i] += -400.0 * x[i] * (x[i + 1] - x[i] ** 2) - 2.0 * (1.0 - x[i])
        # derivative w.r.t. x[i+1]
        grad[i + 1] += 200.0 * (x[i + 1] - x[i] ** 2)
    return grad


def rastrigin(x: List[float]) -> float:
    """Rastrigin function.

    ``f(x) = 10*n + sum(xi^2 - 10*cos(2*pi*xi))``

    Highly multimodal; global minimum is 0 at the origin.

    Parameters
    ----------
    x:
        Input vector of any dimension.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    return 10.0 * n + sum(xi ** 2 - 10.0 * math.cos(2.0 * math.pi * xi) for xi in x)


def rastrigin_grad(x: List[float]) -> List[float]:
    """Analytical gradient of the Rastrigin function.

    ``grad_i = 2*xi + 20*pi*sin(2*pi*xi)``

    Parameters
    ----------
    x:
        Input vector.

    Returns
    -------
    List[float]
        Gradient vector.
    """
    return [2.0 * xi + 20.0 * math.pi * math.sin(2.0 * math.pi * xi) for xi in x]


def ackley(x: List[float]) -> float:
    """Ackley function.

    ``f(x) = -20*exp(-0.2*sqrt(mean(xi^2))) - exp(mean(cos(2*pi*xi))) + 20 + e``

    Uses constants ``a=20``, ``b=0.2``, ``c=2*pi``.
    Global minimum is 0 at the origin.

    Parameters
    ----------
    x:
        Input vector of any dimension.

    Returns
    -------
    float
        Function value.
    """
    n = len(x)
    a, b, c = 20.0, 0.2, 2.0 * math.pi
    sum_sq = sum(xi * xi for xi in x) / n
    sum_cos = sum(math.cos(c * xi) for xi in x) / n
    return -a * math.exp(-b * math.sqrt(sum_sq)) - math.exp(sum_cos) + a + math.e


def ackley_grad(x: List[float]) -> List[float]:
    """Analytical gradient of the Ackley function.

    Derived by differentiating each of the two exponential terms.

    Parameters
    ----------
    x:
        Input vector of any dimension.

    Returns
    -------
    List[float]
        Gradient vector.
    """
    n = len(x)
    a, b, c = 20.0, 0.2, 2.0 * math.pi

    sum_sq = sum(xi * xi for xi in x) / n
    sqrt_sum_sq = math.sqrt(sum_sq) if sum_sq > 0.0 else 0.0
    sum_cos = sum(math.cos(c * xi) for xi in x) / n

    exp1 = math.exp(-b * sqrt_sum_sq)
    exp2 = math.exp(sum_cos)

    grad = []
    for xi in x:
        # Derivative of -a*exp(-b*sqrt(mean(xi^2)))
        if sqrt_sum_sq > 0.0:
            d_term1 = a * b * exp1 * xi / (n * sqrt_sum_sq)
        else:
            d_term1 = 0.0
        # Derivative of -exp(mean(cos(c*xi)))
        d_term2 = exp2 * math.sin(c * xi) * c / n
        grad.append(d_term1 + d_term2)
    return grad


def himmelblau(x: List[float]) -> float:
    """Himmelblau's function (2-D only).

    ``f(x) = (x0^2 + x1 - 11)^2 + (x0 + x1^2 - 7)^2``

    Has four global minima, each with ``f ≈ 0``:

    * (3, 2)
    * (-2.805118, 3.131312)
    * (-3.779310, -3.283186)
    * (3.584428, -1.848126)

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    float
        Function value.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"himmelblau requires 2-D input, got {len(x)}")
    return (x[0] ** 2 + x[1] - 11.0) ** 2 + (x[0] + x[1] ** 2 - 7.0) ** 2


def himmelblau_grad(x: List[float]) -> List[float]:
    """Analytical gradient of Himmelblau's function.

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    List[float]
        2-element gradient vector.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"himmelblau_grad requires 2-D input, got {len(x)}")
    a = x[0] ** 2 + x[1] - 11.0
    b = x[0] + x[1] ** 2 - 7.0
    dx0 = 4.0 * x[0] * a + 2.0 * b
    dx1 = 2.0 * a + 4.0 * x[1] * b
    return [dx0, dx1]


def beale(x: List[float]) -> float:
    """Beale's function (2-D only).

    ``f(x) = (1.5 - x0 + x0*x1)^2``
    ``     + (2.25 - x0 + x0*x1^2)^2``
    ``     + (2.625 - x0 + x0*x1^3)^2``

    Global minimum is 0 at (3, 0.5).

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    float
        Function value.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"beale requires 2-D input, got {len(x)}")
    t1 = 1.5 - x[0] + x[0] * x[1]
    t2 = 2.25 - x[0] + x[0] * x[1] ** 2
    t3 = 2.625 - x[0] + x[0] * x[1] ** 3
    return t1 ** 2 + t2 ** 2 + t3 ** 2


def beale_grad(x: List[float]) -> List[float]:
    """Analytical gradient of Beale's function.

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    List[float]
        2-element gradient vector.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"beale_grad requires 2-D input, got {len(x)}")
    t1 = 1.5 - x[0] + x[0] * x[1]
    t2 = 2.25 - x[0] + x[0] * x[1] ** 2
    t3 = 2.625 - x[0] + x[0] * x[1] ** 3

    # Partial derivatives w.r.t. x0
    dt1_dx0 = -1.0 + x[1]
    dt2_dx0 = -1.0 + x[1] ** 2
    dt3_dx0 = -1.0 + x[1] ** 3
    dx0 = 2.0 * t1 * dt1_dx0 + 2.0 * t2 * dt2_dx0 + 2.0 * t3 * dt3_dx0

    # Partial derivatives w.r.t. x1
    dt1_dx1 = x[0]
    dt2_dx1 = 2.0 * x[0] * x[1]
    dt3_dx1 = 3.0 * x[0] * x[1] ** 2
    dx1 = 2.0 * t1 * dt1_dx1 + 2.0 * t2 * dt2_dx1 + 2.0 * t3 * dt3_dx1

    return [dx0, dx1]


def booth(x: List[float]) -> float:
    """Booth's function (2-D only).

    ``f(x) = (x0 + 2*x1 - 7)^2 + (2*x0 + x1 - 5)^2``

    Global minimum is 0 at (1, 3).

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    float
        Function value.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"booth requires 2-D input, got {len(x)}")
    return (x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2


def booth_grad(x: List[float]) -> List[float]:
    """Analytical gradient of Booth's function.

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    List[float]
        2-element gradient vector.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"booth_grad requires 2-D input, got {len(x)}")
    a = x[0] + 2.0 * x[1] - 7.0
    b = 2.0 * x[0] + x[1] - 5.0
    dx0 = 2.0 * a + 4.0 * b
    dx1 = 4.0 * a + 2.0 * b
    return [dx0, dx1]


def matyas(x: List[float]) -> float:
    """Matyas function (2-D only).

    ``f(x) = 0.26*(x0^2 + x1^2) - 0.48*x0*x1``

    Global minimum is 0 at the origin.

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    float
        Function value.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"matyas requires 2-D input, got {len(x)}")
    return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]


def three_hump_camel(x: List[float]) -> float:
    """Three-hump camel function (2-D only).

    ``f(x) = 2*x0^2 - 1.05*x0^4 + x0^6/6 + x0*x1 + x1^2``

    Global minimum is 0 at the origin.

    Parameters
    ----------
    x:
        2-D input vector.

    Returns
    -------
    float
        Function value.

    Raises
    ------
    ValueError
        If ``len(x) != 2``.
    """
    if len(x) != 2:
        raise ValueError(f"three_hump_camel requires 2-D input, got {len(x)}")
    return (
        2.0 * x[0] ** 2
        - 1.05 * x[0] ** 4
        + x[0] ** 6 / 6.0
        + x[0] * x[1]
        + x[1] ** 2
    )


def styblinski_tang(x: List[float]) -> float:
    """Styblinski-Tang function.

    ``f(x) = 0.5 * sum(xi^4 - 16*xi^2 + 5*xi)``

    Global minimum ≈ -39.166 * n near ``xi ≈ -2.9035``.

    Parameters
    ----------
    x:
        Input vector of any dimension.

    Returns
    -------
    float
        Function value.
    """
    return 0.5 * sum(xi ** 4 - 16.0 * xi ** 2 + 5.0 * xi for xi in x)


# ---------------------------------------------------------------------------
# 1.4  Benchmarking Framework
# ---------------------------------------------------------------------------

# Type alias for readability.
BenchmarkResult = Dict[str, Any]


def benchmark_optimizer(
    optimizer_fn: Callable,
    test_fns: List[Tuple[str, Callable, Callable, List[float]]],
    n_runs: int = 5,
    tol: float = 1.0,
) -> Dict[str, BenchmarkResult]:
    """Benchmark a single optimizer across multiple test functions.

    Parameters
    ----------
    optimizer_fn:
        A callable with signature
        ``optimizer_fn(f, grad_f, x0) -> (x_opt, history)``
        where *history* is any sequence (used only for its length).
    test_fns:
        List of ``(name, f, grad_f, x0)`` tuples.
    n_runs:
        Number of independent runs per test function (default 5).
    tol:
        Success threshold: a run is "successful" if
        ``f(x_opt) < tol`` (default 1.0).

    Returns
    -------
    Dict[str, BenchmarkResult]
        Keys are function names; values are dicts with:

        * ``mean_f``      — mean final objective value across runs
        * ``std_f``       — standard deviation of final objective values
        * ``mean_iters``  — mean number of iterations / history length
        * ``success_rate``— fraction of runs where ``f(x_opt) < tol``
        * ``all_f_values``— raw list of final objective values
    """
    results: Dict[str, BenchmarkResult] = {}
    for name, f, grad_f, x0 in test_fns:
        f_vals: List[float] = []
        iter_counts: List[float] = []
        successes = 0
        for _ in range(n_runs):
            x_opt, history = optimizer_fn(f, grad_f, list(x0))
            fv = f(x_opt)
            f_vals.append(fv)
            iter_counts.append(float(len(history)))
            if fv < tol:
                successes += 1

        mean_f = sum(f_vals) / n_runs
        variance = sum((v - mean_f) ** 2 for v in f_vals) / n_runs
        std_f = math.sqrt(variance)
        mean_iters = sum(iter_counts) / n_runs
        success_rate = successes / n_runs

        results[name] = {
            "mean_f": mean_f,
            "std_f": std_f,
            "mean_iters": mean_iters,
            "success_rate": success_rate,
            "all_f_values": f_vals,
        }
    return results


def compare_optimizers(
    optimizers: Dict[str, Callable],
    test_fns: List[Tuple[str, Callable, Callable, List[float]]],
) -> Dict[str, Dict[str, BenchmarkResult]]:
    """Compare multiple optimizers on the same set of test functions.

    Parameters
    ----------
    optimizers:
        Mapping of ``{optimizer_name: optimizer_fn}`` where each
        ``optimizer_fn`` has signature
        ``optimizer_fn(f, grad_f, x0) -> (x_opt, history)``.
    test_fns:
        List of ``(name, f, grad_f, x0)`` tuples (same format as
        :func:`benchmark_optimizer`).

    Returns
    -------
    Dict[str, Dict[str, BenchmarkResult]]
        Nested mapping ``{optimizer_name: {fn_name: BenchmarkResult}}``.
    """
    comparison: Dict[str, Dict[str, BenchmarkResult]] = {}
    for opt_name, opt_fn in optimizers.items():
        comparison[opt_name] = benchmark_optimizer(opt_fn, test_fns)
    return comparison


# ---------------------------------------------------------------------------
# 1.5  Callback System
# ---------------------------------------------------------------------------


class Callback(ABC):
    """Abstract base class for optimizer callbacks.

    Subclasses override :meth:`on_step_begin`, :meth:`on_step_end`, and/or
    :meth:`should_stop` to hook into the optimizer loop.
    """

    def on_step_begin(
        self, step: int, x: List[float], grad: List[float]
    ) -> None:
        """Called at the beginning of each optimization step.

        Parameters
        ----------
        step:
            Current step index (0-based).
        x:
            Current parameter vector.
        grad:
            Current gradient vector.
        """

    def on_step_end(
        self, step: int, x: List[float], f_val: float
    ) -> None:
        """Called at the end of each optimization step.

        Parameters
        ----------
        step:
            Current step index (0-based).
        x:
            Updated parameter vector.
        f_val:
            Objective value at the updated parameters.
        """

    def should_stop(self) -> bool:
        """Return ``True`` to signal early termination.

        Returns
        -------
        bool
            Default implementation always returns ``False``.
        """
        return False


class CallbackList(Callback):
    """Composite callback that delegates to a list of sub-callbacks.

    All sub-callbacks receive every event. :meth:`should_stop` returns
    ``True`` if *any* sub-callback requests early stopping.

    Parameters
    ----------
    callbacks:
        Ordered list of :class:`Callback` instances.
    """

    def __init__(self, callbacks: List[Callback]) -> None:
        """Initialise with a list of callbacks."""
        self.callbacks = callbacks

    def on_step_begin(
        self, step: int, x: List[float], grad: List[float]
    ) -> None:
        """Delegate to all sub-callbacks."""
        for cb in self.callbacks:
            cb.on_step_begin(step, x, grad)

    def on_step_end(
        self, step: int, x: List[float], f_val: float
    ) -> None:
        """Delegate to all sub-callbacks."""
        for cb in self.callbacks:
            cb.on_step_end(step, x, f_val)

    def should_stop(self) -> bool:
        """Return ``True`` if any sub-callback signals a stop."""
        return any(cb.should_stop() for cb in self.callbacks)


class EarlyStopping(Callback):
    """Stop optimization when no improvement occurs for *patience* steps.

    Parameters
    ----------
    patience:
        Number of steps to wait without improvement before stopping
        (default 10).
    min_delta:
        Minimum change in the monitored value that qualifies as an
        improvement (default ``1e-6``).
    mode:
        ``'min'`` (lower is better) or ``'max'`` (higher is better).
        Default ``'min'``.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        """Initialise early-stopping callback."""
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._best: Optional[float] = None
        self._wait: int = 0
        self._stop: bool = False

    def on_step_end(
        self, step: int, x: List[float], f_val: float
    ) -> None:
        """Update patience counter based on the latest objective value."""
        if self._best is None:
            self._best = f_val
            return

        if self.mode == "min":
            improved = f_val < self._best - self.min_delta
        else:
            improved = f_val > self._best + self.min_delta

        if improved:
            self._best = f_val
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                self._stop = True

    def should_stop(self) -> bool:
        """Return ``True`` once patience has been exhausted."""
        return self._stop

    def reset(self) -> None:
        """Reset the callback to its initial state."""
        self._best = None
        self._wait = 0
        self._stop = False


class GradientMonitor(Callback):
    """Record the L2 norm of the gradient at each step.

    Attributes
    ----------
    norms:
        List of per-step gradient L2 norms accumulated during a run.
    mean_norm:
        Mean gradient norm (available after at least one step).
    max_norm:
        Maximum gradient norm (available after at least one step).
    """

    def __init__(self) -> None:
        """Initialise with empty norm history."""
        self.norms: List[float] = []
        self.mean_norm: float = 0.0
        self.max_norm: float = 0.0

    def on_step_begin(
        self, step: int, x: List[float], grad: List[float]
    ) -> None:
        """Record the L2 norm of *grad* and update summary statistics."""
        norm = math.sqrt(sum(g * g for g in grad))
        self.norms.append(norm)
        self.mean_norm = sum(self.norms) / len(self.norms)
        self.max_norm = max(self.norms)


class LossLogger(Callback):
    """Record (and optionally print) the objective value at each step.

    Parameters
    ----------
    log_every:
        Print the loss every this many steps.  ``0`` (default) means
        silent — values are recorded in :attr:`losses` but not printed.

    Attributes
    ----------
    losses:
        List of objective values recorded via :meth:`on_step_end`.
    """

    def __init__(self, log_every: int = 0) -> None:
        """Initialise the logger."""
        self.log_every = log_every
        self.losses: List[float] = []

    def on_step_end(
        self, step: int, x: List[float], f_val: float
    ) -> None:
        """Append *f_val* to :attr:`losses` and optionally print it."""
        self.losses.append(f_val)
        if self.log_every > 0 and (step + 1) % self.log_every == 0:
            print(f"Step {step + 1}: f = {f_val:.6g}")


class DivergenceDetector(Callback):
    """Raise :exc:`RuntimeError` when the objective diverges.

    A value is considered diverged if ``|f(x)| > threshold``, or if
    it is NaN or Inf.

    Parameters
    ----------
    threshold:
        Absolute value above which divergence is declared
        (default ``1e10``).
    """

    def __init__(self, threshold: float = 1e10) -> None:
        """Initialise with the divergence threshold."""
        self.threshold = threshold

    def on_step_end(
        self, step: int, x: List[float], f_val: float
    ) -> None:
        """Check *f_val* and raise if divergence is detected.

        Raises
        ------
        RuntimeError
            When ``|f_val| > threshold`` or *f_val* is NaN/Inf.
        """
        if (
            math.isnan(f_val)
            or math.isinf(f_val)
            or abs(f_val) > self.threshold
        ):
            raise RuntimeError(
                f"Divergence detected at step {step}: f={f_val}"
            )


# ---------------------------------------------------------------------------
# 1.6  Optimizer State Checkpointing
# ---------------------------------------------------------------------------


def _make_serializable(obj: Any) -> Any:
    """Recursively convert an object to a JSON-serialisable form.

    Handles nested lists, dicts, None, bool, int, and float.
    Unrecognised types are converted via ``str()``.

    Parameters
    ----------
    obj:
        Arbitrary Python object returned by :meth:`get_state`.

    Returns
    -------
    Any
        A JSON-serialisable representation.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        # Preserve NaN / Inf as strings so they round-trip through JSON.
        if math.isnan(obj):
            return "__nan__"
        if math.isinf(obj):
            return "__inf__" if obj > 0 else "__-inf__"
        return obj
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    # Fallback for arbitrary objects.
    return str(obj)


def _restore_special_floats(obj: Any) -> Any:
    """Inverse of :func:`_make_serializable` for float sentinels.

    Replaces the string sentinels ``'__nan__'``, ``'__inf__'``, and
    ``'__-inf__'`` with their ``float`` equivalents.

    Parameters
    ----------
    obj:
        Deserialised JSON object.

    Returns
    -------
    Any
        Object with float sentinels replaced.
    """
    if isinstance(obj, str):
        if obj == "__nan__":
            return float("nan")
        if obj == "__inf__":
            return float("inf")
        if obj == "__-inf__":
            return float("-inf")
        return obj
    if isinstance(obj, list):
        return [_restore_special_floats(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _restore_special_floats(v) for k, v in obj.items()}
    return obj


def save_state(optimizer: Any, path: str) -> None:
    """Save an optimizer's state to a JSON file.

    The optimizer must implement a ``get_state() -> dict`` method.
    The file is written atomically (written to a temporary path first,
    then renamed) to avoid leaving a partially-written file on disk.

    Parameters
    ----------
    optimizer:
        Any optimizer that exposes ``get_state() -> dict``.
    path:
        Destination file path (will be created or overwritten).

    Raises
    ------
    AttributeError
        If *optimizer* does not have a ``get_state`` method.
    """
    state = optimizer.get_state()
    serialisable = _make_serializable(state)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2)
    os.replace(tmp_path, path)


def load_state(optimizer: Any, path: str) -> None:
    """Load an optimizer's state from a JSON file.

    The optimizer must implement a ``load_state(state_dict: dict) -> None``
    method.  Float sentinels written by :func:`save_state` are converted
    back to their ``float`` values before being passed to the optimizer.

    Parameters
    ----------
    optimizer:
        Any optimizer that exposes ``load_state(dict) -> None``.
    path:
        Path to the JSON checkpoint written by :func:`save_state`.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    AttributeError
        If *optimizer* does not have a ``load_state`` method.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    state = _restore_special_floats(raw)
    optimizer.load_state(state)


# ---------------------------------------------------------------------------
# 1.7  NaN/Inf Guards
# ---------------------------------------------------------------------------


def check_finite(x: List[float], name: str = "x") -> None:
    """Raise :exc:`ValueError` if any element of *x* is NaN or Inf.

    Parameters
    ----------
    x:
        Vector to validate.
    name:
        Human-readable name used in the error message (default ``'x'``).

    Raises
    ------
    ValueError
        With message ``"<name>[<i>] is <val>"`` for the first invalid
        element encountered.
    """
    for i, val in enumerate(x):
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"{name}[{i}] is {val}")


def safe_step(
    optimizer: Any, params: List[float], grads: List[float]
) -> List[float]:
    """Perform a guarded optimizer step that validates all vectors.

    Calls :func:`check_finite` on *params* and *grads* before the step,
    and on the result afterwards.  The optimizer must expose a
    ``step(params, grads) -> List[float]`` method.

    Parameters
    ----------
    optimizer:
        Optimizer object with a ``step(params, grads) -> List[float]``
        method.
    params:
        Current parameter vector.
    grads:
        Current gradient vector.

    Returns
    -------
    List[float]
        Updated parameter vector after the optimizer step.

    Raises
    ------
    ValueError
        If any input or output vector contains NaN or Inf.
    """
    check_finite(params, "params")
    check_finite(grads, "grads")
    result = optimizer.step(params, grads)
    check_finite(result, "updated_params")
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # 1.1 Gradient / Jacobian checking
    "check_gradient",
    "check_jacobian",
    # 1.2 Numerical derivatives
    "numerical_gradient",
    "numerical_hessian",
    # 1.3 Test functions
    "sphere",
    "sphere_grad",
    "rosenbrock",
    "rosenbrock_grad",
    "rastrigin",
    "rastrigin_grad",
    "ackley",
    "ackley_grad",
    "himmelblau",
    "himmelblau_grad",
    "beale",
    "beale_grad",
    "booth",
    "booth_grad",
    "matyas",
    "three_hump_camel",
    "styblinski_tang",
    # 1.4 Benchmarking
    "BenchmarkResult",
    "benchmark_optimizer",
    "compare_optimizers",
    # 1.5 Callbacks
    "Callback",
    "CallbackList",
    "EarlyStopping",
    "GradientMonitor",
    "LossLogger",
    "DivergenceDetector",
    # 1.6 Checkpointing
    "save_state",
    "load_state",
    # 1.7 NaN/Inf guards
    "check_finite",
    "safe_step",
]

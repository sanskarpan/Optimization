"""
constrained.py — Pure-Python constrained optimisation toolkit.

Algorithms
----------
Projection utilities  : project_box, project_simplex, project_l2_ball,
                        project_l1_ball, project_linf_ball
Descent methods       : projected_gradient, penalty_method,
                        augmented_lagrangian, frank_wolfe
Splitting / barriers  : admm, barrier_method
"""

from __future__ import annotations

__all__ = [
    # Projection utilities
    "project_box",
    "project_simplex",
    "project_l2_ball",
    "project_l1_ball",
    "project_linf_ball",
    # Constrained optimisation algorithms
    "projected_gradient",
    "penalty_method",
    "augmented_lagrangian",
    "frank_wolfe",
    "admm",
    # Barrier method
    "barrier_method",
]

import math
import warnings
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm2(x: List[float]) -> float:
    return math.sqrt(sum(xi * xi for xi in x))


def _axpy(alpha: float, x: List[float], y: List[float]) -> List[float]:
    """Return alpha*x + y."""
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def _sub(a: List[float], b: List[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _scale(alpha: float, x: List[float]) -> List[float]:
    return [alpha * xi for xi in x]


def _fd_grad(
    f: Callable[[List[float]], float],
    x: List[float],
    eps: float = 1e-7,
) -> List[float]:
    """Finite-difference gradient of f at x (central differences)."""
    n = len(x)
    g = [0.0] * n
    for i in range(n):
        xp = list(x)
        xm = list(x)
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2.0 * eps)
    return g


# ---------------------------------------------------------------------------
# Projection utilities
# ---------------------------------------------------------------------------

def project_box(
    x: List[float],
    lb: List[float],
    ub: List[float],
) -> List[float]:
    """Project *x* onto the axis-aligned box [lb_i, ub_i].

    Parameters
    ----------
    x:  point to project
    lb: lower bounds (same length as x)
    ub: upper bounds (same length as x)

    Returns
    -------
    Projected point (new list).
    """
    return [max(li, min(xi, ui)) for xi, li, ui in zip(x, lb, ub)]


def project_simplex(x: List[float]) -> List[float]:
    """Project *x* onto the probability simplex {v : sum(v)=1, v>=0}.

    Uses the O(n log n) algorithm of Duchi et al. (2008).

    Parameters
    ----------
    x: point to project

    Returns
    -------
    Projected point.
    """
    n = len(x)
    # Sort descending
    u = sorted(x, reverse=True)
    # Find rho
    cumsum = 0.0
    rho = 0
    for j in range(n):
        cumsum += u[j]
        if u[j] - (cumsum - 1.0) / (j + 1) > 0:
            rho = j
    theta = (sum(u[: rho + 1]) - 1.0) / (rho + 1)
    return [max(xi - theta, 0.0) for xi in x]


def project_l2_ball(x: List[float], radius: float = 1.0) -> List[float]:
    """Project *x* onto the L2 ball of given *radius*.

    Parameters
    ----------
    x:      point to project
    radius: ball radius (default 1.0)

    Returns
    -------
    Projected point.
    """
    nrm = _norm2(x)
    if nrm <= radius:
        return list(x)
    return _scale(radius / nrm, x)


def project_l1_ball(x: List[float], radius: float = 1.0) -> List[float]:
    """Project *x* onto the L1 ball ||v||_1 <= *radius*.

    Algorithm: project |x| onto the simplex scaled by *radius*, then restore
    the original signs.

    Parameters
    ----------
    x:      point to project
    radius: ball radius (default 1.0)

    Returns
    -------
    Projected point.
    """
    # Already inside?
    if sum(abs(xi) for xi in x) <= radius:
        return list(x)
    # Work with magnitudes; project onto simplex, scale
    u = [abs(xi) / radius for xi in x]
    v_unit = project_simplex(u)
    v = _scale(radius, v_unit)
    # Restore signs
    return [vi if xi >= 0 else -vi for xi, vi in zip(x, v)]


def project_linf_ball(x: List[float], radius: float = 1.0) -> List[float]:
    """Project *x* onto the L-infinity ball ||v||_inf <= *radius*.

    Parameters
    ----------
    x:      point to project
    radius: ball radius (default 1.0)

    Returns
    -------
    Projected point.
    """
    return [max(-radius, min(xi, radius)) for xi in x]


# ---------------------------------------------------------------------------
# Projected gradient descent
# ---------------------------------------------------------------------------

def projected_gradient(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    project: Callable[[List[float]], List[float]],
    x0: List[float],
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[List[float], float, int, bool]:
    """Projected gradient descent.

    Iteration: x_{k+1} = project(x_k - lr * grad_f(x_k))

    Parameters
    ----------
    f:       objective function
    grad_f:  gradient of objective
    project: projection onto feasible set
    x0:      starting point
    lr:      step size
    max_iter: maximum iterations
    tol:     convergence tolerance (step norm)

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    x = list(x0)
    converged = False
    k = 0
    for k in range(max_iter):
        g = grad_f(x)
        x_new = project(_axpy(-lr, g, x))
        step = _norm2(_sub(x_new, x))
        x = x_new
        if step < tol:
            converged = True
            break
    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# Penalty method
# ---------------------------------------------------------------------------

def penalty_method(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    constraints: List[Callable[[List[float]], float]],
    x0: List[float],
    mu0: float = 1.0,
    mu_factor: float = 10.0,
    max_outer: int = 20,
    max_inner: int = 100,
    tol: float = 1e-6,
    grad_constraints: Optional[List[Callable[[List[float]], List[float]]]] = None,
) -> Tuple[List[float], float, int, bool]:
    """Quadratic penalty method for inequality constraints c_i(x) <= 0.

    Penalised objective: F(x) = f(x) + mu * sum_i max(0, c_i(x))^2

    Parameters
    ----------
    f:              objective function
    grad_f:         gradient of objective
    constraints:    list of callables, each returns c_i(x) (feasible when <= 0)
    x0:             starting point
    mu0:            initial penalty parameter
    mu_factor:      multiplicative increase of mu per outer iteration
    max_outer:      maximum outer iterations
    max_inner:      maximum inner gradient-descent steps per outer iteration
    tol:            convergence tolerance
    grad_constraints: optional list of gradient functions for each constraint;
                    if None, finite differences are used

    Returns
    -------
    (x_opt, f_opt, n_outer_iters, converged)
    """
    n = len(x0)
    x = list(x0)
    mu = mu0
    converged = False

    for outer in range(max_outer):
        # Build penalised objective and gradient
        def penalised(xv: List[float], _mu: float = mu) -> float:
            val = f(xv)
            for c in constraints:
                cv = c(xv)
                if cv > 0:
                    val += _mu * cv * cv
            return val

        def penalised_grad(xv: List[float], _mu: float = mu) -> List[float]:
            g = list(grad_f(xv))
            for idx, c in enumerate(constraints):
                cv = c(xv)
                if cv > 0:
                    if grad_constraints is not None:
                        gc = grad_constraints[idx](xv)
                    else:
                        gc = _fd_grad(c, xv)
                    for i in range(n):
                        g[i] += 2.0 * _mu * cv * gc[i]
            return g

        # Inner gradient descent with backtracking line search
        lr = 1.0 / (mu + 1.0)  # adaptive initial step
        for _ in range(max_inner):
            g = penalised_grad(x)
            gnorm = _norm2(g)
            if gnorm < tol:
                break
            # Simple backtracking
            step = lr
            f0 = penalised(x)
            for _ in range(50):
                x_try = _axpy(-step, g, x)
                if penalised(x_try) <= f0 - 0.5 * step * gnorm * gnorm:
                    break
                step *= 0.5
            x_new = _axpy(-step, g, x)
            dx = _norm2(_sub(x_new, x))
            x = x_new
            if dx < tol * 0.1:
                break

        # Check feasibility
        max_viol = max((max(0.0, c(x)) for c in constraints), default=0.0)
        if max_viol < tol:
            converged = True
            break

        mu *= mu_factor

    return x, f(x), outer + 1, converged


# ---------------------------------------------------------------------------
# Augmented Lagrangian (equality constraints)
# ---------------------------------------------------------------------------

def augmented_lagrangian(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    eq_constraints: List[Callable[[List[float]], float]],
    x0: List[float],
    lam0: Optional[List[float]] = None,
    mu0: float = 1.0,
    mu_factor: float = 2.0,
    max_outer: int = 20,
    max_inner: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[float], float, int, bool]:
    """Augmented Lagrangian method for equality constraints h_i(x) = 0.

    L_aug(x, lam, mu) = f(x) + sum_i lam_i*h_i(x) + (mu/2)*sum_i h_i(x)^2

    Parameters
    ----------
    f:              objective function
    grad_f:         gradient of objective
    eq_constraints: list of callables, each returns h_i(x) (feasible when = 0)
    x0:             starting point
    lam0:           initial multipliers (zeros if None)
    mu0:            initial penalty parameter
    mu_factor:      multiplicative increase of mu per outer iteration
    max_outer:      maximum outer iterations
    max_inner:      maximum inner gradient-descent steps
    tol:            convergence tolerance

    Returns
    -------
    (x_opt, f_opt, n_outer_iters, converged)
    """
    m = len(eq_constraints)
    n = len(x0)
    x = list(x0)
    lam = list(lam0) if lam0 is not None else [0.0] * m
    mu = mu0
    converged = False

    for outer in range(max_outer):
        # Snapshot multipliers & penalty for the closure
        _lam = list(lam)
        _mu = mu

        def aug_obj(xv: List[float]) -> float:
            val = f(xv)
            for i, h in enumerate(eq_constraints):
                hv = h(xv)
                val += _lam[i] * hv + (_mu / 2.0) * hv * hv
            return val

        def aug_grad(xv: List[float]) -> List[float]:
            g = list(grad_f(xv))
            for i, h in enumerate(eq_constraints):
                hv = h(xv)
                gh = _fd_grad(h, xv)
                coeff = _lam[i] + _mu * hv
                for j in range(n):
                    g[j] += coeff * gh[j]
            return g

        # Inner gradient descent
        lr = 1.0 / (_mu + 1.0)
        for _ in range(max_inner):
            g = aug_grad(x)
            gnorm = _norm2(g)
            if gnorm < tol * 0.1:
                break
            step = lr
            f0 = aug_obj(x)
            for _ in range(50):
                x_try = _axpy(-step, g, x)
                if aug_obj(x_try) <= f0 - 0.5 * step * gnorm * gnorm:
                    break
                step *= 0.5
            x_new = _axpy(-step, g, x)
            dx = _norm2(_sub(x_new, x))
            x = x_new
            if dx < tol * 0.01:
                break

        # Dual ascent: update multipliers
        h_vals = [h(x) for h in eq_constraints]
        for i in range(m):
            lam[i] += mu * h_vals[i]

        # Check feasibility
        feas = math.sqrt(sum(hv * hv for hv in h_vals))
        if feas < tol:
            converged = True
            break

        mu *= mu_factor

    return x, f(x), outer + 1, converged


# ---------------------------------------------------------------------------
# Frank-Wolfe (conditional gradient)
# ---------------------------------------------------------------------------

def frank_wolfe(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    lp_oracle: Callable[[List[float]], List[float]],
    x0: List[float],
    max_iter: int = 200,
    tol: float = 1e-6,
) -> Tuple[List[float], float, int, bool]:
    """Frank-Wolfe (conditional gradient) method.

    Per iteration:
        d_k  = lp_oracle(grad_f(x_k))   # min_{s in C} <grad, s>
        gap  = <grad, x_k - d_k>
        if gap < tol → converged
        gamma_k = 2 / (k + 2)
        x_{k+1} = (1 - gamma_k) * x_k + gamma_k * d_k

    Parameters
    ----------
    f:         objective function
    grad_f:    gradient of objective
    lp_oracle: given gradient g, returns argmin_{s in C} <g, s>
    x0:        starting point (must be in feasible set C)
    max_iter:  maximum iterations
    tol:       Frank-Wolfe gap convergence tolerance

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    x = list(x0)
    converged = False
    k = 0
    for k in range(max_iter):
        g = grad_f(x)
        d = lp_oracle(g)
        # Frank-Wolfe gap
        gap = _dot(g, _sub(x, d))
        if gap < tol:
            converged = True
            break
        gamma = 2.0 / (k + 2.0)
        # x = (1-gamma)*x + gamma*d
        x = _axpy(gamma, _sub(d, x), x)
    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# ADMM
# ---------------------------------------------------------------------------

def admm(
    f_prox: Callable[[List[float], float], List[float]],
    g_prox: Callable[[List[float], float], List[float]],
    x0: List[float],
    z0: Optional[List[float]] = None,
    rho: float = 1.0,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> Tuple[List[float], List[float], int, bool]:
    """ADMM for min f(x) + g(z) subject to x = z.

    Updates:
        x_{k+1} = prox_{f/rho}(z_k - u_k)
        z_{k+1} = prox_{g/rho}(x_{k+1} + u_k)
        u_{k+1} = u_k + x_{k+1} - z_{k+1}

    Parameters
    ----------
    f_prox:   proximal operator of f; signature (v, rho) -> x
    g_prox:   proximal operator of g; signature (v, rho) -> z
    x0:       initial x
    z0:       initial z (copy of x0 if None)
    rho:      ADMM penalty parameter
    max_iter: maximum iterations
    tol:      convergence tolerance for primal and dual residuals

    Returns
    -------
    (x_opt, z_opt, n_iters, converged)
    """
    n = len(x0)
    x = list(x0)
    z = list(z0) if z0 is not None else list(x0)
    u = [0.0] * n
    converged = False
    k = 0
    for k in range(max_iter):
        z_old = list(z)
        # x-update
        x = f_prox(_sub(z, u), rho)
        # z-update
        z = g_prox(_axpy(1.0, u, x), rho)
        # u-update
        u = _axpy(1.0, _sub(x, z), u)
        # Residuals
        primal_res = _norm2(_sub(x, z))
        dual_res = rho * _norm2(_sub(z, z_old))
        if primal_res < tol and dual_res < tol:
            converged = True
            break
    return x, z, k + 1, converged


# ---------------------------------------------------------------------------
# Barrier / interior-point method
# ---------------------------------------------------------------------------

def barrier_method(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    barrier_f: Callable[[List[float]], float],
    barrier_grad: Callable[[List[float]], List[float]],
    x0: List[float],
    t0: float = 1.0,
    mu: float = 10.0,
    max_outer: int = 20,
    max_inner: int = 100,
    tol: float = 1e-6,
) -> Tuple[List[float], float, int, bool]:
    """Log-barrier / interior-point method.

    Minimises (1/t)*f(x) + barrier_f(x) with increasing t.

    Parameters
    ----------
    f:            objective function
    grad_f:       gradient of objective
    barrier_f:    barrier function (e.g. -sum log(-c_i(x)))
    barrier_grad: gradient of barrier function
    x0:           strictly feasible starting point
    t0:           initial scaling parameter
    mu:           multiplicative increase of t per outer iteration
    max_outer:    maximum outer iterations
    max_inner:    maximum inner gradient-descent steps
    tol:          convergence tolerance (gradient norm of barrier subproblem)

    Returns
    -------
    (x_opt, f_opt, n_outer_iters, converged)
    """
    x = list(x0)
    t = t0
    converged = False

    for outer in range(max_outer):
        _t = t

        def barrier_obj(xv: List[float]) -> float:
            try:
                bval = barrier_f(xv)
            except (ValueError, ZeroDivisionError, OverflowError):
                return math.inf
            return (1.0 / _t) * f(xv) + bval

        def barrier_obj_grad(xv: List[float]) -> List[float]:
            gf = _scale(1.0 / _t, grad_f(xv))
            gb = barrier_grad(xv)
            return [a + b for a, b in zip(gf, gb)]

        # Inner gradient descent with backtracking
        for _ in range(max_inner):
            g = barrier_obj_grad(x)
            gnorm = _norm2(g)
            if gnorm < tol:
                break
            # Backtracking line search (stay in domain)
            step = 1.0
            f0 = barrier_obj(x)
            for _ in range(100):
                x_try = _axpy(-step, g, x)
                fval = barrier_obj(x_try)
                if math.isfinite(fval) and fval <= f0 - 0.5 * step * gnorm * gnorm:
                    break
                step *= 0.5
            else:
                # Could not make progress; stop inner loop
                break
            x_new = _axpy(-step, g, x)
            dx = _norm2(_sub(x_new, x))
            x = x_new
            if dx < tol * 0.01:
                break

        # Convergence: duality gap proxy = n_constraints / t
        # We use gradient norm of the full barrier subproblem as stopping criterion
        g_final = barrier_obj_grad(x)
        if _norm2(g_final) < tol:
            converged = True
            break

        t *= mu

    return x, f(x), outer + 1, converged

"""
Proximal Operators and Algorithms
===================================
Proximal operators for common regularizers, ISTA, FISTA,
proximal gradient with backtracking, Douglas-Rachford splitting.

All routines operate on plain Python lists of floats and use only
the standard library (math, typing).  No numpy or scipy dependency.

Public API
----------
Proximal operators:
    prox_l1, prox_l2_sq, prox_linf, prox_non_negative,
    prox_box, prox_elastic_net

Algorithms:
    ista, fista, proximal_gradient, douglas_rachford
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dot(a: List[float], b: List[float]) -> float:
    """Return the dot product of two equal-length lists."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm2(v: List[float]) -> float:
    """Return the Euclidean (L2) norm of *v*."""
    return math.sqrt(sum(vi * vi for vi in v))


def _norm1(v: List[float]) -> float:
    """Return the L1 norm of *v*."""
    return sum(abs(vi) for vi in v)


def _proj_l1_ball(v: List[float], radius: float) -> List[float]:
    """Project *v* onto the L1 ball {x : ||x||_1 <= radius}.

    Uses the efficient algorithm of Duchi et al. (2008):

    1. If ||v||_1 <= radius the point is already feasible; return *v*.
    2. Otherwise sort the absolute values in descending order and find
       a threshold ``theta`` such that::

           sum_i max(|v_i| - theta, 0) = radius

       Then return ``sign(v_i) * max(|v_i| - theta, 0)``.

    Parameters
    ----------
    v:
        Input vector (list of floats).
    radius:
        Radius of the L1 ball (must be > 0).

    Returns
    -------
    List[float]
        Projection of *v* onto the L1 ball of the given radius.
    """
    if radius <= 0.0:
        return [0.0] * len(v)

    abs_v = [abs(vi) for vi in v]
    if sum(abs_v) <= radius:
        return v[:]

    # Sort absolute values descending
    u = sorted(abs_v, reverse=True)
    n = len(u)
    cumsum = 0.0
    theta = 0.0
    for j in range(n):
        cumsum += u[j]
        t = (cumsum - radius) / (j + 1)
        if t < u[j]:
            theta = t
        else:
            break

    return [math.copysign(max(abs(vi) - theta, 0.0), vi) for vi in v]


# ---------------------------------------------------------------------------
# 9.1  Proximal Operators
# ---------------------------------------------------------------------------


def prox_l1(v: List[float], lam: float) -> List[float]:
    """Proximal operator for ``lambda * ||x||_1`` (soft thresholding).

    Computes element-wise::

        prox_l1(v, lam)[i] = sign(v[i]) * max(|v[i]| - lam, 0)

    This is the solution to::

        argmin_x  lam * ||x||_1  +  (1/2) * ||x - v||_2^2

    Parameters
    ----------
    v:
        Input vector.
    lam:
        Non-negative regularisation strength.

    Returns
    -------
    List[float]
        Soft-thresholded vector.

    Raises
    ------
    ValueError
        If *lam* is negative.
    """
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    return [math.copysign(max(abs(vi) - lam, 0.0), vi) for vi in v]


def prox_l2_sq(v: List[float], lam: float) -> List[float]:
    """Proximal operator for ``lambda * ||x||_2^2`` (ridge / L2-squared).

    Computes element-wise::

        prox_l2_sq(v, lam)[i] = v[i] / (1 + 2 * lam)

    This is the solution to::

        argmin_x  lam * ||x||_2^2  +  (1/2) * ||x - v||_2^2

    Parameters
    ----------
    v:
        Input vector.
    lam:
        Non-negative regularisation strength.

    Returns
    -------
    List[float]
        Shrunk vector.

    Raises
    ------
    ValueError
        If *lam* is negative.
    """
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    scale = 1.0 / (1.0 + 2.0 * lam)
    return [vi * scale for vi in v]


def prox_linf(v: List[float], lam: float) -> List[float]:
    """Proximal operator for ``lambda * ||x||_inf``.

    Uses the Moreau decomposition::

        prox_{lam * ||.||_inf}(v) = v - lam * proj_{L1-ball(lam)}(v / lam) * lam
                                  = v - lam * proj_{L1-ball(1)}(v / lam)

    Equivalently (and more directly)::

        prox_{lam * ||.||_inf}(v) = v - proj_{L1-ball(lam)}(v)

    because the Moreau identity states::

        prox_{lam * g}(v) + lam * prox_{g* / lam}(v / lam) = v

    and the conjugate of ``||.||_inf`` is the indicator of the L1 ball, whose
    proximal operator is the projection onto the L1 ball.

    Implementation:
        1. Compute ``w = proj_{L1-ball(lam)}(v)`` via :func:`_proj_l1_ball`.
        2. Return ``v[i] - w[i]`` for each component.

    Parameters
    ----------
    v:
        Input vector.
    lam:
        Non-negative regularisation strength (scales the L-inf norm).

    Returns
    -------
    List[float]
        Result of the proximal operator.

    Raises
    ------
    ValueError
        If *lam* is negative.
    """
    if lam < 0.0:
        raise ValueError(f"lam must be non-negative, got {lam}")
    if lam == 0.0:
        return v[:]
    proj = _proj_l1_ball(v, lam)
    return [vi - pi for vi, pi in zip(v, proj)]


def prox_non_negative(v: List[float], lam: float = 0.0) -> List[float]:
    """Proximal operator for the indicator of the non-negative orthant.

    Returns the projection of *v* onto ``{x : x_i >= 0}``::

        prox_non_negative(v)[i] = max(v[i], 0)

    The parameter *lam* is accepted for API uniformity but is ignored.

    Parameters
    ----------
    v:
        Input vector.
    lam:
        Ignored (kept for uniform proximal-operator API).

    Returns
    -------
    List[float]
        Vector clipped to non-negative values.
    """
    return [max(vi, 0.0) for vi in v]


def prox_box(
    v: List[float],
    lower: List[float],
    upper: List[float],
    lam: float = 0.0,
) -> List[float]:
    """Proximal operator for the indicator of a box constraint.

    Returns the projection of *v* onto the box ``[lower_i, upper_i]``::

        prox_box(v)[i] = clip(v[i], lower[i], upper[i])

    The parameter *lam* is accepted for API uniformity but is ignored.

    Parameters
    ----------
    v:
        Input vector (length *n*).
    lower:
        Lower bounds (length *n*).
    upper:
        Upper bounds (length *n*).
    lam:
        Ignored (kept for uniform proximal-operator API).

    Returns
    -------
    List[float]
        Clipped vector.

    Raises
    ------
    ValueError
        If *lower*, *upper*, and *v* do not have the same length, or if
        any ``lower[i] > upper[i]``.
    """
    n = len(v)
    if len(lower) != n or len(upper) != n:
        raise ValueError("v, lower, and upper must have the same length.")
    for i in range(n):
        if lower[i] > upper[i]:
            raise ValueError(
                f"lower[{i}]={lower[i]} > upper[{i}]={upper[i]}."
            )
    return [max(lower[i], min(upper[i], v[i])) for i in range(n)]


def prox_elastic_net(v: List[float], lam1: float, lam2: float) -> List[float]:
    """Proximal operator for the elastic-net penalty.

    Minimises::

        lam1 * ||x||_1  +  lam2 * ||x||_2^2  +  (1/2) * ||x - v||_2^2

    The closed-form solution is::

        prox_elastic_net(v, lam1, lam2)[i]
            = sign(v[i]) * max(|v[i]| / (1 + 2*lam2) - lam1 / (1 + 2*lam2), 0)

    Derivation: ridge shrinkage first (scale ``v`` by ``1/(1+2*lam2)``),
    then soft-threshold with ``lam1/(1+2*lam2)``.

    Parameters
    ----------
    v:
        Input vector.
    lam1:
        L1 regularisation weight (non-negative).
    lam2:
        L2-squared regularisation weight (non-negative).

    Returns
    -------
    List[float]
        Elastic-net proximal result.

    Raises
    ------
    ValueError
        If *lam1* or *lam2* is negative.
    """
    if lam1 < 0.0:
        raise ValueError(f"lam1 must be non-negative, got {lam1}")
    if lam2 < 0.0:
        raise ValueError(f"lam2 must be non-negative, got {lam2}")
    scale = 1.0 + 2.0 * lam2
    threshold = lam1 / scale
    return [math.copysign(max(abs(vi) / scale - threshold, 0.0), vi) for vi in v]


# ---------------------------------------------------------------------------
# 9.2  ISTA
# ---------------------------------------------------------------------------


def ista(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    prox_g: Callable[[List[float], float], List[float]],
    x0: List[float],
    L: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """Iterative Shrinkage-Thresholding Algorithm (ISTA).

    Solves the composite minimisation problem::

        min_x  f(x) + g(x)

    where *f* is convex and *L*-smooth (has a Lipschitz-continuous gradient)
    and *g* is convex but possibly non-smooth with a tractable proximal
    operator.

    Each iterate is::

        x_{k+1} = prox_{g / L}(x_k - (1/L) * grad_f(x_k))

    Convergence rate is ``O(1/k)`` for convex *f*.

    **Backtracking line search** (when ``L is None``):
        An estimate ``L_est`` is maintained and doubled whenever the
        sufficient-decrease condition::

            f(x_new) <= f(x) + <grad_f(x), x_new - x> + (L_est/2)*||x_new - x||^2

        is violated.  ``L_est`` is initialised to ``1.0``.

    Parameters
    ----------
    f:
        Smooth part of the objective; maps ``List[float]`` -> ``float``.
    grad_f:
        Gradient of *f*; maps ``List[float]`` -> ``List[float]``.
    prox_g:
        Proximal operator of *g*; signature ``prox_g(v, step) -> x``.
        Here *step* = ``1/L``.
    x0:
        Starting point (list of floats, length *n*).
    L:
        Lipschitz constant of ``grad_f``.  If ``None`` backtracking is
        used to estimate it adaptively each iteration.
    max_iter:
        Maximum number of gradient steps.
    tol:
        Convergence tolerance on ``||x_{k+1} - x_k||_2``.

    Returns
    -------
    x_opt : List[float]
        Approximate minimiser.
    history : List[float]
        Sequence of objective values ``f(x_k) + ...`` — here recorded as
        ``f(x_k)`` (the smooth part) after each proximal step.

    Raises
    ------
    ValueError
        If *L* is provided and is not positive.
    """
    if L is not None and L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")

    n = len(x0)
    x: List[float] = x0[:]
    history: List[float] = [f(x)]

    # Backtracking parameters
    L_est: float = 1.0 if L is None else L
    beta: float = 2.0  # multiplicative increase factor

    for _k in range(max_iter):
        grad = grad_f(x)
        f_x = f(x)

        if L is None:
            # Backtracking: find smallest L_est (up to 50 doublings) such that
            # the descent-lemma condition holds.
            for _ in range(50):
                step = 1.0 / L_est
                x_new = prox_g([x[i] - step * grad[i] for i in range(n)], step)
                diff = [x_new[i] - x[i] for i in range(n)]
                rhs = (
                    f_x
                    + _dot(grad, diff)
                    + (L_est / 2.0) * sum(d * d for d in diff)
                )
                if f(x_new) <= rhs + 1e-10:
                    break
                L_est *= beta
            # Slight decrease to allow L to shrink over time (optional)
            L_est = max(L_est * 0.99, 1e-12)
        else:
            step = 1.0 / L_est
            x_new = prox_g([x[i] - step * grad[i] for i in range(n)], step)

        diff_norm = _norm2([x_new[i] - x[i] for i in range(n)])
        x = x_new
        history.append(f(x))

        if diff_norm < tol:
            break

    return x, history


# ---------------------------------------------------------------------------
# 9.3  FISTA
# ---------------------------------------------------------------------------


def fista(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    prox_g: Callable[[List[float], float], List[float]],
    x0: List[float],
    L: Optional[float] = None,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """Fast Iterative Shrinkage-Thresholding Algorithm (FISTA).

    Extends :func:`ista` with Nesterov momentum to achieve an ``O(1/k^2)``
    convergence rate (Beck & Teboulle, 2009).

    Algorithm::

        t_1 = 1,  y_1 = x_0
        For k = 1, 2, ...:
            x_{k+1} = prox_{g/L}(y_k - (1/L) * grad_f(y_k))
            t_{k+1} = (1 + sqrt(1 + 4 * t_k^2)) / 2
            y_{k+1} = x_{k+1} + ((t_k - 1) / t_{k+1}) * (x_{k+1} - x_k)

    The momentum coefficient ``(t_k - 1) / t_{k+1}`` increases with *k*,
    accelerating convergence compared to ISTA.

    **Backtracking** (when ``L is None``): same adaptive doubling scheme as
    :func:`ista`, applied to the momentum iterate *y_k*.

    Parameters
    ----------
    f:
        Smooth objective; maps ``List[float]`` -> ``float``.
    grad_f:
        Gradient of *f*; maps ``List[float]`` -> ``List[float]``.
    prox_g:
        Proximal operator of *g*; signature ``prox_g(v, step) -> x``.
    x0:
        Starting point.
    L:
        Lipschitz constant of ``grad_f``.  If ``None``, backtracking is used.
    max_iter:
        Maximum number of momentum steps.
    tol:
        Convergence tolerance on ``||x_{k+1} - x_k||_2``.

    Returns
    -------
    x_opt : List[float]
        Approximate minimiser.
    history : List[float]
        Sequence of ``f(x_k)`` values after each step.

    Raises
    ------
    ValueError
        If *L* is provided and is not positive.
    """
    if L is not None and L <= 0.0:
        raise ValueError(f"L must be positive, got {L}.")

    n = len(x0)
    x: List[float] = x0[:]
    y: List[float] = x0[:]   # momentum point
    t: float = 1.0
    history: List[float] = [f(x)]

    L_est: float = 1.0 if L is None else L
    beta: float = 2.0

    for _k in range(max_iter):
        grad_y = grad_f(y)
        f_y = f(y)

        if L is None:
            # Backtracking on the momentum iterate y
            for _ in range(50):
                step = 1.0 / L_est
                x_new = prox_g([y[i] - step * grad_y[i] for i in range(n)], step)
                diff = [x_new[i] - y[i] for i in range(n)]
                rhs = (
                    f_y
                    + _dot(grad_y, diff)
                    + (L_est / 2.0) * sum(d * d for d in diff)
                )
                if f(x_new) <= rhs + 1e-10:
                    break
                L_est *= beta
            L_est = max(L_est * 0.99, 1e-12)
        else:
            step = 1.0 / L_est
            x_new = prox_g([y[i] - step * grad_y[i] for i in range(n)], step)

        # Update momentum sequence
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        momentum = (t - 1.0) / t_new

        diff_norm = _norm2([x_new[i] - x[i] for i in range(n)])

        # Momentum-extrapolated point for next iteration
        y_new = [x_new[i] + momentum * (x_new[i] - x[i]) for i in range(n)]

        x = x_new
        y = y_new
        t = t_new
        history.append(f(x))

        if diff_norm < tol:
            break

    return x, history


# ---------------------------------------------------------------------------
# 9.4  Proximal Gradient with Backtracking
# ---------------------------------------------------------------------------


def proximal_gradient(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    prox_g: Callable[[List[float], float], List[float]],
    x0: List[float],
    L_init: float = 1.0,
    beta: float = 0.5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """Proximal gradient method with adaptive backtracking.

    Identical in spirit to :func:`ista` but **always** performs backtracking:
    there is no option for a fixed Lipschitz constant.  The step size
    ``1/L`` adapts per iteration — when the descent-lemma condition is
    violated, *L* is multiplied by ``1/beta`` (i.e. the step shrinks).

    The current *L* is carried across iterations (warm-starting), which
    often means fewer backtracking sub-iterations as the algorithm matures.
    *L* is capped at ``1e12`` to avoid numerical blow-up.

    Parameters
    ----------
    f:
        Smooth objective.
    grad_f:
        Gradient of *f*.
    prox_g:
        Proximal operator of *g*.
    x0:
        Starting point.
    L_init:
        Initial estimate of the Lipschitz constant (positive).
    beta:
        Backtracking contraction factor in ``(0, 1)``.  When the descent
        condition fails *L* is replaced by ``L / beta``.
    max_iter:
        Maximum iterations.
    tol:
        Convergence tolerance on ``||x_{k+1} - x_k||_2``.

    Returns
    -------
    x_opt : List[float]
        Approximate minimiser.
    history : List[float]
        Sequence of ``f(x_k)`` values.

    Raises
    ------
    ValueError
        If *L_init* <= 0 or *beta* not in (0, 1).
    """
    if L_init <= 0.0:
        raise ValueError(f"L_init must be positive, got {L_init}.")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}.")

    n = len(x0)
    x: List[float] = x0[:]
    L_est: float = L_init
    history: List[float] = [f(x)]
    _L_cap: float = 1e12

    for _k in range(max_iter):
        grad = grad_f(x)
        f_x = f(x)

        # Backtracking: increase L until descent condition satisfied
        for _ in range(100):
            step = 1.0 / L_est
            x_new = prox_g([x[i] - step * grad[i] for i in range(n)], step)
            diff = [x_new[i] - x[i] for i in range(n)]
            rhs = (
                f_x
                + _dot(grad, diff)
                + (L_est / 2.0) * sum(d * d for d in diff)
            )
            if f(x_new) <= rhs + 1e-10:
                break
            L_est = min(L_est / beta, _L_cap)
            if L_est >= _L_cap:
                warnings.warn(
                    "proximal_gradient: L hit cap 1e12; "
                    "check gradient or objective.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                break

        diff_norm = _norm2([x_new[i] - x[i] for i in range(n)])
        x = x_new
        history.append(f(x))

        if diff_norm < tol:
            break

    return x, history


# ---------------------------------------------------------------------------
# 9.5  Douglas-Rachford Splitting
# ---------------------------------------------------------------------------


def douglas_rachford(
    prox_f: Callable[[List[float], float], List[float]],
    prox_g: Callable[[List[float], float], List[float]],
    x0: List[float],
    gamma: float = 1.0,
    relaxation: float = 1.0,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> Tuple[List[float], List[float]]:
    """Douglas-Rachford Splitting for ``min f(x) + g(x)``.

    Requires tractable proximal operators for **both** *f* and *g*; no
    gradient information is needed.

    Algorithm (relaxed Peaceman-Rachford / Douglas-Rachford)::

        z_0 = x0
        For k = 0, 1, ...:
            x_{k+1} = prox_{gamma * g}(z_k)
            y_{k+1} = prox_{gamma * f}(2 * x_{k+1} - z_k)
            z_{k+1} = z_k + relaxation * (y_{k+1} - x_{k+1})

    At convergence the primal solution is::

        x* = prox_{gamma * g}(z*)

    Setting ``relaxation = 1`` recovers the standard Douglas-Rachford
    iteration; ``relaxation = 2`` gives Peaceman-Rachford (convergence
    requires strong convexity in that case).

    Parameters
    ----------
    prox_f:
        Proximal operator of *f*; signature ``prox_f(v, gamma) -> x``.
    prox_g:
        Proximal operator of *g*; signature ``prox_g(v, gamma) -> x``.
    x0:
        Starting point for the dual variable *z*.
    gamma:
        Step-size / regularisation parameter (positive).
    relaxation:
        Over-relaxation coefficient, typically in ``(0, 2)``.
    max_iter:
        Maximum number of splitting iterations.
    tol:
        Convergence tolerance on ``||z_{k+1} - z_k||_2``.

    Returns
    -------
    x_opt : List[float]
        Approximate primal minimiser ``prox_{gamma * g}(z*)``.
    history : List[float]
        Sequence of ``||z_{k+1} - z_k||_2`` values (one per iteration).

    Raises
    ------
    ValueError
        If *gamma* <= 0 or *relaxation* <= 0.
    """
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}.")
    if relaxation <= 0.0:
        raise ValueError(f"relaxation must be positive, got {relaxation}.")

    n = len(x0)
    z: List[float] = x0[:]
    history: List[float] = []

    for _k in range(max_iter):
        # Proximal step w.r.t. g
        x_new = prox_g(z, gamma)
        # Reflection of x_new through z, then proximal step w.r.t. f
        reflected = [2.0 * x_new[i] - z[i] for i in range(n)]
        y_new = prox_f(reflected, gamma)
        # Relaxed update of dual variable
        z_new = [z[i] + relaxation * (y_new[i] - x_new[i]) for i in range(n)]

        residual = _norm2([z_new[i] - z[i] for i in range(n)])
        history.append(residual)
        z = z_new

        if residual < tol:
            break

    # Recover primal solution
    x_opt = prox_g(z, gamma)
    return x_opt, history


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "prox_l1",
    "prox_l2_sq",
    "prox_linf",
    "prox_non_negative",
    "prox_box",
    "prox_elastic_net",
    "ista",
    "fista",
    "proximal_gradient",
    "douglas_rachford",
]

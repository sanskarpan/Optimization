"""
line_search.py — Line search methods for optimization.

Pure Python (stdlib only: math, typing).

All vector operations work on List[float].
"""

import math
from typing import Callable, List, Tuple

__all__ = [
    'backtracking_line_search',
    'wolfe_line_search',
    'brent_minimize',
    'cubic_interpolation_line_search',
    'strong_wolfe_line_search',
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dot(u: List[float], v: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))


def _axpy(alpha: float, x: List[float], y: List[float]) -> List[float]:
    """Return alpha*x + y (BLAS axpy style)."""
    return [alpha * xi + yi for xi, yi in zip(x, y)]


# ---------------------------------------------------------------------------
# 1. Backtracking line search (Armijo condition)
# ---------------------------------------------------------------------------

def backtracking_line_search(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Backtracking line search satisfying the Armijo sufficient-decrease condition.

    Starts from ``alpha0`` and repeatedly multiplies by ``rho`` until

        f(x + alpha * direction) <= f(x) + c * alpha * dot(grad_f(x), direction)

    or ``max_iter`` reductions have been performed.

    Parameters
    ----------
    f:
        Objective function ``f(x) -> float``.
    grad_f:
        Gradient ``grad_f(x) -> List[float]``.
    x:
        Current iterate (List[float]).
    direction:
        Search direction (List[float]).  Should satisfy dot(grad_f(x), direction) < 0
        for a descent direction.
    alpha0:
        Initial step size (default 1.0).
    rho:
        Contraction factor in (0, 1) (default 0.5).
    c:
        Armijo constant, typically 1e-4 (default 1e-4).
    max_iter:
        Maximum number of contractions (default 50).

    Returns
    -------
    float
        Step size satisfying the Armijo condition, or the smallest alpha tried
        if the condition is never met.
    """
    f0 = f(x)
    g0 = grad_f(x)
    slope = _dot(g0, direction)

    alpha = alpha0
    for _ in range(max_iter):
        x_new = _axpy(alpha, direction, x)
        if f(x_new) <= f0 + c * alpha * slope:
            return alpha
        alpha *= rho

    return alpha


# ---------------------------------------------------------------------------
# 2. Basic Wolfe line search (bracket + bisection)
# ---------------------------------------------------------------------------

def wolfe_line_search(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
) -> float:
    """Line search satisfying the (weak) Wolfe conditions.

    The two Wolfe conditions are:

    * **Armijo** (sufficient decrease):
      ``phi(alpha) <= phi(0) + c1 * alpha * phi'(0)``
    * **Curvature**:
      ``phi'(alpha) >= c2 * phi'(0)``

    Uses a bracket-then-bisection strategy.

    Parameters
    ----------
    f:
        Objective function.
    grad_f:
        Gradient function.
    x:
        Current iterate.
    direction:
        Search direction (should be a descent direction).
    alpha0:
        Initial trial step (default 1.0).
    c1:
        Armijo constant (default 1e-4).
    c2:
        Curvature constant, 0 < c1 < c2 < 1 (default 0.9).
    max_iter:
        Maximum iterations (default 20).

    Returns
    -------
    float
        Step size approximately satisfying Wolfe conditions.
    """
    phi0 = f(x)
    dphi0 = _dot(grad_f(x), direction)  # phi'(0)

    def phi(a: float) -> float:
        return f(_axpy(a, direction, x))

    def dphi(a: float) -> float:
        return _dot(grad_f(_axpy(a, direction, x)), direction)

    alpha_lo = 0.0
    alpha_hi = alpha0
    phi_lo = phi0
    dphi_lo = dphi0

    alpha = alpha0

    for i in range(max_iter):
        phi_a = phi(alpha)

        # Armijo violated → must reduce; set upper bound
        if phi_a > phi0 + c1 * alpha * dphi0 or (i > 0 and phi_a >= phi_lo):
            alpha_hi = alpha
            # bisect
            alpha = 0.5 * (alpha_lo + alpha_hi)
            continue

        dphi_a = dphi(alpha)

        # Curvature condition met → done
        if dphi_a >= c2 * dphi0:
            return alpha

        # Slope positive → upper bound found; zoom from below
        alpha_lo = alpha
        phi_lo = phi_a
        dphi_lo = dphi_a

        if alpha_hi is None or math.isinf(alpha_hi):
            alpha = 2.0 * alpha
        else:
            alpha = 0.5 * (alpha_lo + alpha_hi)

    return alpha


# ---------------------------------------------------------------------------
# 3. Brent's method for scalar minimization
# ---------------------------------------------------------------------------

def brent_minimize(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Brent's method for finding the minimum of a scalar function on [a, b].

    Combines golden-section search with parabolic interpolation for
    superlinear convergence near a smooth minimum.

    Parameters
    ----------
    f:
        Scalar objective function ``f(x) -> float``.
    a:
        Left endpoint of the search interval.
    b:
        Right endpoint of the search interval.
    tol:
        Convergence tolerance (default 1e-6).
    max_iter:
        Maximum number of iterations (default 100).

    Returns
    -------
    float
        The x value (not f(x)) at the approximate minimum.
    """
    _golden = 1.0 - (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.381966

    # Initialise: x is the best point so far; w and v are the two next best.
    x = w = v = a + _golden * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0  # d = last step, e = step before last

    for _ in range(max_iter):
        midpoint = 0.5 * (a + b)
        tol1 = tol * abs(x) + 1e-10
        tol2 = 2.0 * tol1

        # Convergence check
        if abs(x - midpoint) <= tol2 - 0.5 * (b - a):
            return x

        # Try parabolic interpolation
        use_golden = True
        if abs(e) > tol1:
            # Fit parabola through x, w, v (if distinct)
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            else:
                q = -q
            r = e
            e = d

            # Accept parabolic step if it lies in [a, b] and is small enough
            if abs(p) < abs(0.5 * q * r) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                # Don't evaluate too close to the endpoints
                if (u - a) < tol2 or (b - u) < tol2:
                    d = math.copysign(tol1, midpoint - x)
                use_golden = False

        if use_golden:
            # Golden-section step
            if x >= midpoint:
                e = a - x
            else:
                e = b - x
            d = _golden * e

        # Evaluate at new trial point u
        u = x + (d if abs(d) >= tol1 else math.copysign(tol1, d))
        fu = f(u)

        # Update brackets and best points
        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, fv = w, fw
            w, fw = x, fx
            x, fx = u, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, fv = w, fw
                w, fw = u, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu

    return x


# ---------------------------------------------------------------------------
# 4. Cubic interpolation line search
# ---------------------------------------------------------------------------

def cubic_interpolation_line_search(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    alpha_lo: float,
    alpha_hi: float,
) -> float:
    """Find a step via cubic interpolation within the bracket [alpha_lo, alpha_hi].

    Uses function and directional-derivative values at both endpoints to fit
    a cubic polynomial and return its minimiser (clamped to the bracket).

    Let ``phi(alpha) = f(x + alpha * direction)``.  The cubic is fit through
    ``(alpha_lo, phi(alpha_lo), phi'(alpha_lo))`` and
    ``(alpha_hi, phi(alpha_hi), phi'(alpha_hi))``.

    Parameters
    ----------
    f:
        Objective function.
    grad_f:
        Gradient function.
    x:
        Current iterate.
    direction:
        Search direction.
    alpha_lo:
        Lower endpoint of the bracket.
    alpha_hi:
        Upper endpoint of the bracket.

    Returns
    -------
    float
        Interpolated step size, clamped to [min(alpha_lo, alpha_hi),
        max(alpha_lo, alpha_hi)].
    """
    lo, hi = min(alpha_lo, alpha_hi), max(alpha_lo, alpha_hi)

    x_lo = _axpy(alpha_lo, direction, x)
    x_hi = _axpy(alpha_hi, direction, x)

    phi_lo = f(x_lo)
    phi_hi = f(x_hi)
    d_lo = _dot(grad_f(x_lo), direction)   # phi'(alpha_lo)
    d_hi = _dot(grad_f(x_hi), direction)   # phi'(alpha_hi)

    delta = alpha_hi - alpha_lo

    # Cubic minimiser formula (Nocedal & Wright, equation 3.59)
    # Compute discriminant safely
    s = d_lo + d_hi - 3.0 * (phi_hi - phi_lo) / delta
    disc = s * s - d_lo * d_hi

    if disc < 0.0:
        # Imaginary roots — fall back to midpoint
        return 0.5 * (alpha_lo + alpha_hi)

    sqrt_disc = math.sqrt(disc)

    # Minimiser of the cubic (Nocedal & Wright eq. 3.59)
    denom = d_hi - d_lo + 2.0 * sqrt_disc
    if abs(denom) < 1e-15:
        # Degenerate cubic — fall back to midpoint
        return 0.5 * (alpha_lo + alpha_hi)

    alpha_star = alpha_hi - delta * (d_hi + sqrt_disc - s) / denom

    # Clamp to bracket
    return max(lo, min(hi, alpha_star))


# ---------------------------------------------------------------------------
# 5. Strong Wolfe line search (Nocedal & Wright Algorithm 3.5 / 3.6)
# ---------------------------------------------------------------------------

def strong_wolfe_line_search(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    alpha0: float = 1.0,
    alpha_max: float = 50.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20,
) -> float:
    """Strong Wolfe line search (Nocedal & Wright Algorithms 3.5 and 3.6).

    Finds a step length satisfying the **strong** Wolfe conditions:

    * **Armijo**: ``phi(alpha) <= phi(0) + c1 * alpha * phi'(0)``
    * **Strong curvature**: ``|phi'(alpha)| <= c2 * |phi'(0)|``

    The algorithm has two phases:

    1. **Bracket phase**: increase alpha until a suitable bracket is found.
    2. **Zoom phase**: bisect/interpolate within the bracket.

    Parameters
    ----------
    f:
        Objective function.
    grad_f:
        Gradient function.
    x:
        Current iterate.
    direction:
        Search direction.
    alpha0:
        Initial trial step (default 1.0).
    alpha_max:
        Maximum allowable step (default 50.0).
    c1:
        Armijo constant (default 1e-4).
    c2:
        Curvature constant, 0 < c1 < c2 < 1 (default 0.9).
    max_iter:
        Maximum iterations per phase (default 20).

    Returns
    -------
    float
        Step satisfying the strong Wolfe conditions, or the best step found.
    """
    def phi(a: float) -> float:
        return f(_axpy(a, direction, x))

    def dphi(a: float) -> float:
        return _dot(grad_f(_axpy(a, direction, x)), direction)

    phi0 = phi(0.0)
    dphi0 = dphi(0.0)

    # ------------------------------------------------------------------
    # Zoom phase (Algorithm 3.6)
    # ------------------------------------------------------------------
    def _zoom(alpha_lo: float, alpha_hi: float, phi_lo: float, phi_hi: float) -> float:
        """Zoom into bracket [alpha_lo, alpha_hi] to find strong-Wolfe step."""
        for _ in range(max_iter):
            # Try cubic interpolation, fall back to bisection
            try:
                alpha_j = cubic_interpolation_line_search(
                    f, grad_f, x, direction, alpha_lo, alpha_hi
                )
            except Exception:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)

            # Safety: if interpolation lands outside or too close to boundary, bisect
            lo_b = min(alpha_lo, alpha_hi)
            hi_b = max(alpha_lo, alpha_hi)
            margin = 0.1 * (hi_b - lo_b)
            if alpha_j <= lo_b + margin or alpha_j >= hi_b - margin:
                alpha_j = 0.5 * (alpha_lo + alpha_hi)

            phi_j = phi(alpha_j)

            if phi_j > phi0 + c1 * alpha_j * dphi0 or phi_j >= phi_lo:
                # Armijo violated or not improving — shrink from hi side
                alpha_hi = alpha_j
                phi_hi = phi_j
            else:
                dphi_j = dphi(alpha_j)

                # Strong curvature condition met → done
                if abs(dphi_j) <= c2 * abs(dphi0):
                    return alpha_j

                # Slope points toward current lo → flip hi
                if dphi_j * (alpha_hi - alpha_lo) >= 0.0:
                    alpha_hi = alpha_lo
                    phi_hi = phi_lo

                alpha_lo = alpha_j
                phi_lo = phi_j

            # Convergence guard
            if abs(alpha_hi - alpha_lo) < 1e-12:
                break

        return alpha_lo

    # ------------------------------------------------------------------
    # Phase 1: bracket (Algorithm 3.5)
    # ------------------------------------------------------------------
    alpha_prev = 0.0
    phi_prev = phi0
    alpha_i = alpha0

    for i in range(max_iter):
        phi_i = phi(alpha_i)

        if phi_i > phi0 + c1 * alpha_i * dphi0 or (i > 0 and phi_i >= phi_prev):
            # Armijo violated or not improving — zoom in [alpha_prev, alpha_i]
            return _zoom(alpha_prev, alpha_i, phi_prev, phi_i)

        dphi_i = dphi(alpha_i)

        # Strong curvature satisfied → done
        if abs(dphi_i) <= c2 * abs(dphi0):
            return alpha_i

        # Positive derivative → bracket is [alpha_i, alpha_prev]; zoom
        if dphi_i >= 0.0:
            return _zoom(alpha_i, alpha_prev, phi_i, phi_prev)

        # Continue expanding
        alpha_prev = alpha_i
        phi_prev = phi_i
        alpha_i = min(2.0 * alpha_i, alpha_max)

        if alpha_i >= alpha_max:
            return alpha_i

    return alpha_i

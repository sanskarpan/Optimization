"""
Line Search Methods
===================

Line search algorithms for determining optimal step size.

Applications:
- Newton's method and quasi-Newton methods
- Trust region methods
- Conjugate gradient
"""

import math
from typing import Callable, Tuple, Optional, List


def armijo_condition(
    f: Callable[[List[float]], float],
    x: List[float],
    direction: List[float],
    gradient: List[float],
    alpha: float,
    c1: float = 1e-4
) -> bool:
    """
    Check Armijo condition (sufficient decrease).

    f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd

    Args:
        f: Objective function
        x: Current point
        direction: Search direction
        gradient: Gradient at x
        alpha: Step size to check
        c1: Armijo constant (typically 1e-4)

    Returns:
        True if Armijo condition satisfied
    """
    # Compute x + alpha * direction
    x_new = [xi + alpha * di for xi, di in zip(x, direction)]

    # Left side: f(x_new)
    f_new = f(x_new)

    # Right side: f(x) + c1 * alpha * grad^T * direction
    grad_dot_dir = sum(gi * di for gi, di in zip(gradient, direction))
    f_expected = f(x) + c1 * alpha * grad_dot_dir

    return f_new <= f_expected


def backtracking_line_search(
    f: Callable[[List[float]], float],
    x: List[float],
    direction: List[float],
    gradient: List[float],
    alpha_init: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    max_iter: int = 50
) -> float:
    """
    Backtracking line search satisfying Armijo condition.

    Starts with alpha_init and reduces by factor rho until
    Armijo condition is satisfied.

    Args:
        f: Objective function
        x: Current point
        direction: Search direction (typically -gradient or Newton direction)
        gradient: Gradient at x
        alpha_init: Initial step size
        rho: Reduction factor (0 < rho < 1)
        c1: Armijo constant
        max_iter: Maximum iterations

    Returns:
        Step size alpha

    Example:
        >>> f = lambda x: sum(xi**2 for xi in x)
        >>> x = [1.0, 2.0]
        >>> grad = [2.0, 4.0]
        >>> direction = [-g for g in grad]
        >>> alpha = backtracking_line_search(f, x, direction, grad)
    """
    alpha = alpha_init

    for _ in range(max_iter):
        if armijo_condition(f, x, direction, gradient, alpha, c1):
            return alpha
        alpha *= rho

    return alpha


def wolfe_conditions(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    gradient: List[float],
    alpha: float,
    c1: float = 1e-4,
    c2: float = 0.9
) -> bool:
    """
    Check Wolfe conditions (sufficient decrease + curvature).

    Condition 1 (Armijo): f(x + α*d) ≤ f(x) + c₁*α*∇f(x)ᵀd
    Condition 2 (Curvature): ∇f(x + α*d)ᵀd ≥ c₂*∇f(x)ᵀd

    Args:
        f: Objective function
        grad_f: Gradient function
        x: Current point
        direction: Search direction
        gradient: Gradient at x
        alpha: Step size
        c1: Armijo constant
        c2: Curvature constant

    Returns:
        True if both Wolfe conditions satisfied
    """
    # Check Armijo condition
    if not armijo_condition(f, x, direction, gradient, alpha, c1):
        return False

    # Compute new point
    x_new = [xi + alpha * di for xi, di in zip(x, direction)]

    # Compute new gradient
    grad_new = grad_f(x_new)

    # Check curvature condition
    grad_dot_dir = sum(gi * di for gi, di in zip(gradient, direction))
    grad_new_dot_dir = sum(gi * di for gi, di in zip(grad_new, direction))

    return grad_new_dot_dir >= c2 * grad_dot_dir


def wolfe_line_search(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x: List[float],
    direction: List[float],
    gradient: List[float],
    alpha_init: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 20
) -> float:
    """
    Line search satisfying Wolfe conditions.

    Uses bracketing and zoom procedure.

    Args:
        f: Objective function
        grad_f: Gradient function
        x: Current point
        direction: Search direction
        gradient: Gradient at x
        alpha_init: Initial step size
        c1: Armijo constant
        c2: Curvature constant
        max_iter: Maximum iterations

    Returns:
        Step size alpha

    Example:
        >>> f = lambda x: sum(xi**2 for xi in x)
        >>> grad_f = lambda x: [2*xi for xi in x]
        >>> x = [1.0, 2.0]
        >>> direction = [-2.0, -4.0]
        >>> grad = [2.0, 4.0]
        >>> alpha = wolfe_line_search(f, grad_f, x, direction, grad)
    """
    # Bracket-and-bisect Wolfe line search.
    # alpha_lo satisfies Armijo; alpha_hi violates it (or is the upper sentinel).
    alpha = alpha_init
    alpha_lo = 0.0
    alpha_hi = None          # unknown upper bound initially
    ALPHA_MAX = 1e8          # safety cap to prevent unbounded growth

    for _ in range(max_iter):
        if wolfe_conditions(f, grad_f, x, direction, gradient, alpha, c1, c2):
            return alpha

        if not armijo_condition(f, x, direction, gradient, alpha, c1):
            # Armijo violated: alpha is too large; shrink the bracket
            alpha_hi = alpha
            alpha = (alpha_lo + alpha_hi) / 2.0
        else:
            # Armijo satisfied but curvature not: need a larger alpha
            alpha_lo = alpha
            if alpha_hi is not None:
                # Zoom between lo and hi
                alpha = (alpha_lo + alpha_hi) / 2.0
            else:
                # No upper bound yet; double with a hard cap
                alpha = min(alpha * 2.0, ALPHA_MAX)
                if alpha >= ALPHA_MAX:
                    break

    return alpha


def exact_line_search_quadratic(
    A: List[List[float]],
    b: List[float],
    x: List[float],
    direction: List[float]
) -> float:
    """
    Exact line search for quadratic function.

    For f(x) = 0.5 * x^T A x - b^T x,
    optimal step size is α = (r^T d) / (d^T A d)

    where r = b - Ax is the residual.

    Args:
        A: Matrix in quadratic form
        b: Vector in quadratic form
        x: Current point
        direction: Search direction

    Returns:
        Exact step size
    """
    n = len(x)
    if len(A) != n or len(b) != n or any(len(row) != n for row in A):
        raise ValueError(
            f"exact_line_search_quadratic: A must be {n}×{n} and b must have "
            f"length {n} to match x (got A: {len(A)}×{len(A[0]) if A else 0}, "
            f"b: {len(b)})"
        )

    # Compute Ax
    Ax = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]

    # Residual: r = b - Ax
    r = [b[i] - Ax[i] for i in range(len(x))]

    # Compute A * direction
    Ad = [sum(A[i][j] * direction[j] for j in range(len(direction)))
          for i in range(len(A))]

    # Numerator: r^T * d
    numerator = sum(r[i] * direction[i] for i in range(len(r)))

    # Denominator: d^T * A * d
    denominator = sum(direction[i] * Ad[i] for i in range(len(direction)))

    if abs(denominator) < 1e-10:
        return 0.0

    alpha = numerator / denominator
    # A negative alpha would move in the ascent direction; clamp to zero.
    return max(alpha, 0.0)


def golden_section_search(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-5
) -> float:
    """
    Golden section search for 1D minimization.

    Finds minimum of f on interval [a, b].

    Args:
        f: Univariate function
        a: Left endpoint
        b: Right endpoint
        tol: Tolerance

    Returns:
        Approximate minimizer

    Example:
        >>> f = lambda x: (x - 2)**2
        >>> x_min = golden_section_search(f, 0, 5)
        >>> # Returns value close to 2
    """
    if a >= b:
        raise ValueError(
            f"golden_section_search: require a < b, got a={a}, b={b}"
        )

    golden_ratio = (math.sqrt(5) - 1) / 2

    # Initial points
    c = b - golden_ratio * (b - a)
    d = a + golden_ratio * (b - a)

    while abs(b - a) > tol:
        if f(c) < f(d):
            b = d
            d = c
            c = b - golden_ratio * (b - a)
        else:
            a = c
            c = d
            d = a + golden_ratio * (b - a)

    return (a + b) / 2

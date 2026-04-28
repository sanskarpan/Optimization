"""
Constrained Optimization
========================

Lagrange multipliers, KKT conditions, projected gradient descent.
"""

import math
from typing import List, Callable, Tuple, Optional


def lagrange_multiplier(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    constraint: Callable[[List[float]], float],
    grad_constraint: Callable[[List[float]], List[float]],
    x0: List[float],
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[List[float], float]:
    """
    Minimize f(x) subject to g(x) = 0 using Lagrange multipliers.

    L(x, λ) = f(x) + λ*g(x)

    Args:
        f: Objective function
        grad_f: Gradient of objective
        constraint: Constraint function g(x) = 0
        grad_constraint: Gradient of constraint
        x0: Initial point
        learning_rate: Step size
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        (optimal_point, lagrange_multiplier)
    """
    x = x0[:]
    lam = 0.0

    for iteration in range(max_iter):
        grad_obj = grad_f(x)
        grad_con = grad_constraint(x)
        con_val = constraint(x)

        # Update x: minimize Lagrangian
        x = [xi - learning_rate * (grad_obj[i] + lam * grad_con[i])
             for i, xi in enumerate(x)]

        # Update lambda: enforce constraint
        lam = lam + learning_rate * con_val

        # Recompute at updated x/lam for an accurate convergence check.
        # Using pre-update gradients (stale) could declare false convergence.
        grad_obj_new = grad_f(x)
        grad_con_new = grad_constraint(x)
        con_val_new = constraint(x)
        lagrangian_grad = [grad_obj_new[i] + lam * grad_con_new[i]
                           for i in range(len(x))]
        if abs(con_val_new) < tol and all(abs(g) < tol for g in lagrangian_grad):
            break

    return x, lam


def kkt_conditions(
    grad_f: List[float],
    grad_constraints: List[List[float]],
    multipliers: List[float],
    constraints: List[float],
    tol: float = 1e-6
) -> bool:
    """
    Check if KKT conditions are satisfied.

    For min f(x) s.t. g_i(x) <= 0:

    1. Stationarity: ∇f + Σ λ_i ∇g_i = 0
    2. Primal feasibility: g_i(x) <= 0
    3. Dual feasibility: λ_i >= 0
    4. Complementary slackness: λ_i * g_i(x) = 0

    Args:
        grad_f: Gradient of objective at point
        grad_constraints: Gradients of constraints at point
        multipliers: Lagrange multipliers
        constraints: Constraint values at point
        tol: Tolerance

    Returns:
        True if KKT conditions satisfied
    """
    n = len(grad_f)
    m = len(constraints)

    # 1. Stationarity
    gradient_lagrangian = grad_f[:]
    for i in range(m):
        for j in range(n):
            gradient_lagrangian[j] += multipliers[i] * grad_constraints[i][j]

    if any(abs(g) > tol for g in gradient_lagrangian):
        return False

    # 2. Primal feasibility
    if any(c > tol for c in constraints):
        return False

    # 3. Dual feasibility
    if any(lam < -tol for lam in multipliers):
        return False

    # 4. Complementary slackness
    if any(abs(lam * c) > tol for lam, c in zip(multipliers, constraints)):
        return False

    return True


def projected_gradient_descent(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    projection: Callable[[List[float]], List[float]],
    x0: List[float],
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-6
) -> Tuple[List[float], List[float]]:
    """
    Projected gradient descent for constrained optimization.

    At each step:
    1. Take gradient step
    2. Project back onto feasible set

    Args:
        f: Objective function
        grad_f: Gradient function
        projection: Function that projects point onto feasible set
        x0: Initial point (must be feasible)
        learning_rate: Step size
        max_iter: Maximum iterations
        tol: Convergence tolerance

    Returns:
        (optimal_point, history)
    """
    x = projection(x0)  # Ensure initial point is feasible
    history = [f(x)]

    for iteration in range(max_iter):
        grad = grad_f(x)

        # Gradient step
        x_new = [xi - learning_rate * gi for xi, gi in zip(x, grad)]

        # Project onto feasible set
        x_new = projection(x_new)

        # Check convergence
        diff = math.sqrt(sum((xn - xi)**2 for xn, xi in zip(x_new, x)))
        x = x_new          # accept the step before the break so we return x_new
        history.append(f(x))
        if diff < tol:
            break

    return x, history


def box_projection(x: List[float], lower: List[float], upper: List[float]) -> List[float]:
    """
    Project point onto box constraints.

    lower_i <= x_i <= upper_i

    Args:
        x: Point to project
        lower: Lower bounds (same length as x)
        upper: Upper bounds (same length as x)

    Returns:
        Projected point

    Example:
        >>> x = [5.0, -2.0, 3.0]
        >>> lower = [0.0, 0.0, 0.0]
        >>> upper = [4.0, 4.0, 4.0]
        >>> box_projection(x, lower, upper)
        [4.0, 0.0, 3.0]
    """
    if len(x) != len(lower) or len(x) != len(upper):
        raise ValueError(
            f"box_projection: x, lower, upper must have the same length "
            f"(got {len(x)}, {len(lower)}, {len(upper)})"
        )
    return [max(l, min(u, xi)) for xi, l, u in zip(x, lower, upper)]


def simplex_projection(x: List[float], z: float = 1.0) -> List[float]:
    """
    Project point onto probability simplex.

    Σ x_i = z, x_i >= 0

    Args:
        x: Point to project
        z: Sum constraint (default 1.0 for probability simplex)

    Returns:
        Projected point

    Example:
        >>> x = [0.5, 0.7, -0.2]
        >>> simplex_projection(x, z=1.0)
        # Returns valid probability distribution
    """
    if z <= 0:
        raise ValueError(
            f"simplex_projection: z must be > 0 (got z={z})"
        )
    if len(x) == 0:
        return []
    n = len(x)
    u = sorted(x, reverse=True)

    # Find threshold
    cumsum = 0.0
    rho = 0
    for i in range(n):
        cumsum += u[i]
        if u[i] - (cumsum - z) / (i + 1) > 0:
            rho = i

    # Compute threshold
    theta = (sum(u[:rho + 1]) - z) / (rho + 1)

    # Project
    return [max(xi - theta, 0.0) for xi in x]


def barrier_method(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    constraints: List[Callable[[List[float]], float]],
    x0: List[float],
    t: float = 1.0,
    mu: float = 10.0,
    tol: float = 1e-6,
    max_outer: int = 50
) -> Tuple[List[float], List[float]]:
    """
    Barrier method for inequality constraints.

    Approximates min f(x) s.t. g_i(x) <= 0
    using barrier: min t*f(x) - Σ log(-g_i(x))

    Args:
        f: Objective function
        grad_f: Gradient of objective
        constraints: List of constraint functions (g_i(x) <= 0)
        x0: Initial feasible point
        t: Initial barrier parameter
        mu: Barrier parameter multiplier
        tol: Tolerance
        max_outer: Maximum outer iterations

    Returns:
        (optimal_point, history)
    """
    x = x0[:]
    history = [f(x)]

    for outer in range(max_outer):
        # Inner loop: minimise barrier function using gradient descent + backtracking
        for inner in range(100):
            # Compute barrier gradient: ∇(t·f - Σ log(-g_i))
            barrier_grad = [t * g for g in grad_f(x)]

            for constraint_func in constraints:
                # Use the actual constraint value at x for both the denominator and
                # the finite-difference baseline (fixes BUG-005).
                con_val_at_x = constraint_func(x)
                # Clamp to a small negative number to keep log defined when x is on
                # or just outside the boundary; the large repulsive gradient will
                # push x back into the interior.
                denom = min(con_val_at_x, -1e-10)

                # Numerical gradient of constraint (∇g_i)
                h = 1e-5
                con_grad = []
                for i in range(len(x)):
                    x_plus = x[:]
                    x_plus[i] += h
                    # Use actual con_val_at_x as baseline (fixes BUG-005)
                    con_grad.append((constraint_func(x_plus) - con_val_at_x) / h)

                # Barrier contribution: -∇g_i / g_i
                for i in range(len(x)):
                    barrier_grad[i] -= con_grad[i] / denom

            # Backtracking line search: enforce feasibility AND sufficient decrease.
            # The initial step size is normalised by the gradient magnitude so it
            # remains stable even for large t (where t·∇f can be very large).
            grad_norm = math.sqrt(sum(gi ** 2 for gi in barrier_grad))
            lr = min(0.01, 1.0 / (grad_norm + 1e-8))
            x_new = None
            barrier_val_x = t * f(x) - sum(
                math.log(-min(cf(x), -1e-10)) for cf in constraints)
            for _ in range(30):
                candidate = [xi - lr * gi for xi, gi in zip(x, barrier_grad)]
                feasible = all(cf(candidate) < 0 for cf in constraints)
                if feasible:
                    barrier_val_new = t * f(candidate) - sum(
                        math.log(-cf(candidate)) for cf in constraints)
                    # Armijo sufficient-decrease condition for barrier objective
                    if barrier_val_new <= barrier_val_x - 1e-4 * lr * (grad_norm ** 2):
                        x_new = candidate
                        break
                lr *= 0.5

            if x_new is None:
                # Cannot find a feasible step; leave x unchanged and end inner loop.
                break

            # Check convergence
            diff = math.sqrt(sum((xn - xi)**2 for xn, xi in zip(x_new, x)))
            x = x_new
            if diff < tol / t:
                break

        history.append(f(x))

        # Check outer convergence
        if len(constraints) / t < tol:
            break

        # Increase barrier parameter
        t *= mu

    return x, history

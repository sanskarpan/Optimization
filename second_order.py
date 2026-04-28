"""
Second-Order Optimization Methods
==================================

Newton's method, Quasi-Newton methods (BFGS, L-BFGS), Conjugate Gradient.

Uses second-order information (Hessian) for faster convergence.
"""

import math
from collections import deque
from typing import List, Callable, Tuple, Optional


def newton_step(gradient: List[float], hessian: List[List[float]]) -> List[float]:
    """
    Compute Newton direction: direction = -H^(-1) * gradient

    Uses Gaussian elimination with partial pivoting and Tikhonov regularisation
    (diagonal damping λ = 1e-8) to handle near-singular or indefinite Hessians.
    A RuntimeWarning is raised if the matrix is numerically singular so callers
    can detect unreliable directions.

    Args:
        gradient: Gradient vector
        hessian: Hessian matrix

    Returns:
        Newton direction
    """
    import warnings
    n = len(gradient)

    # Copy inputs; apply small diagonal regularisation to improve conditioning.
    REG = 1e-8
    H = [row[:] for row in hessian]
    for i in range(n):
        H[i][i] += REG
    g = [-gi for gi in gradient]

    # Forward elimination with partial pivoting
    for i in range(n):
        # Find pivot
        max_row = i
        for k in range(i + 1, n):
            if abs(H[k][i]) > abs(H[max_row][i]):
                max_row = k
        H[i], H[max_row] = H[max_row], H[i]
        g[i], g[max_row] = g[max_row], g[i]

        if abs(H[i][i]) < 1e-10:
            # Column is numerically zero even after pivoting; skip elimination
            # for this column but warn the caller.
            warnings.warn(
                f"newton_step: near-singular Hessian at pivot {i} "
                "(direction component set to 0); result may be unreliable.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Explicitly zero the sub-diagonal entries of column i so that
            # back-substitution sees a properly upper-triangular matrix.
            # Without this, H[k][i] for k > i remain non-zero and corrupt
            # the back-substitution sums for rows above i.
            for k in range(i + 1, n):
                H[k][i] = 0.0
            continue

        for k in range(i + 1, n):
            factor = H[k][i] / H[i][i]
            g[k] -= factor * g[i]
            for j in range(i, n):
                H[k][j] -= factor * H[i][j]

    # Back substitution
    direction = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(H[i][i]) < 1e-10:
            direction[i] = 0.0
        else:
            direction[i] = (
                g[i] - sum(H[i][j] * direction[j] for j in range(i + 1, n))
            ) / H[i][i]

    return direction


def _so_line_search(
    f: Callable[[List[float]], float],
    x: List[float],
    direction: List[float],
    grad: List[float],
    rho: float = 0.9,
    c: float = 1e-4,
    max_iter: int = 20
) -> float:
    """Backtracking Armijo line search shared by all second-order methods.

    Uses rho=0.9 (slow shrinkage) because quasi-Newton / conjugate-gradient
    search directions are well-scaled and alpha=1 is typically accepted on
    the first or second try.  The public ``backtracking_line_search`` in
    line_search.py uses rho=0.5 (faster shrinkage) for a more conservative
    general-purpose default.
    """
    alpha = 1.0
    gTd = sum(g * d for g, d in zip(grad, direction))
    f_x = f(x)
    for _ in range(max_iter):
        x_new = [xi + alpha * di for xi, di in zip(x, direction)]
        if f(x_new) <= f_x + c * alpha * gTd:
            return alpha
        alpha *= rho
    return alpha


class NewtonMethod:
    """
    Newton's Method for optimization.

    Uses second-order Taylor approximation.

    θ_new = θ - H^(-1) * ∇f(θ)

    where H is the Hessian matrix.
    """

    def __init__(
        self,
        learning_rate: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6
    ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        f: Callable[[List[float]], float],
        grad_f: Callable[[List[float]], List[float]],
        hess_f: Callable[[List[float]], List[List[float]]],
        x0: List[float]
    ) -> Tuple[List[float], List[float]]:
        """
        Minimize function using Newton's method.

        Args:
            f: Objective function
            grad_f: Gradient function
            hess_f: Hessian function
            x0: Initial point

        Returns:
            (optimal_point, history_of_function_values)
        """
        x = x0[:]
        history = [f(x)]

        for iteration in range(self.max_iter):
            grad = grad_f(x)

            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in grad))
            if grad_norm < self.tol:
                break

            # Compute Newton direction
            hess = hess_f(x)
            direction = newton_step(grad, hess)

            # Update
            x = [xi + self.learning_rate * di for xi, di in zip(x, direction)]
            history.append(f(x))

        return x, history

    def reset(self):
        """No persistent state between optimize() calls; provided for API consistency."""
        pass


class BFGS:
    """
    Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.

    Quasi-Newton method that approximates Hessian inverse.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        f: Callable[[List[float]], float],
        grad_f: Callable[[List[float]], List[float]],
        x0: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Minimize using BFGS."""
        n = len(x0)
        x = x0[:]

        # Initialize inverse Hessian approximation as identity
        H_inv = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

        grad = grad_f(x)
        history = [f(x)]

        for iteration in range(self.max_iter):
            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in grad))
            if grad_norm < self.tol:
                break

            # Compute search direction
            direction = [-sum(H_inv[i][j] * grad[j] for j in range(n)) for i in range(n)]

            # Line search (simplified - use fixed step)
            alpha = self._line_search(f, x, direction, grad)

            # Update x
            x_new = [xi + alpha * di for xi, di in zip(x, direction)]

            # Compute gradient at new point
            grad_new = grad_f(x_new)

            # BFGS update
            s = [alpha * di for di in direction]  # x_new - x
            y = [grad_new[i] - grad[i] for i in range(n)]

            # Update inverse Hessian approximation
            H_inv = self._bfgs_update(H_inv, s, y)

            x = x_new
            grad = grad_new
            history.append(f(x))

        return x, history

    def _line_search(self, f, x, direction, grad):
        """Delegate to shared second-order backtracking line search."""
        return _so_line_search(f, x, direction, grad)

    def reset(self):
        """No persistent state between optimize() calls; provided for API consistency."""
        pass

    def _bfgs_update(self, H_inv, s, y):
        """BFGS update formula for inverse Hessian."""
        n = len(s)

        # Compute s^T y
        s_dot_y = sum(s[i] * y[i] for i in range(n))

        # Curvature condition: s·y must be strictly positive for H_inv to remain PD.
        # Using abs() previously allowed negative s·y to corrupt the approximation.
        if s_dot_y <= 1e-10:
            return H_inv  # Skip update

        # Compute H_inv * y
        Hy = [sum(H_inv[i][j] * y[j] for j in range(n)) for i in range(n)]

        # Compute y^T * H_inv * y
        y_H_y = sum(y[i] * Hy[i] for i in range(n))

        # BFGS update
        rho = 1.0 / s_dot_y

        H_new = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                term1 = H_inv[i][j]
                term2 = rho * s[i] * s[j] * (1 + rho * y_H_y)
                term3 = rho * (Hy[i] * s[j] + s[i] * Hy[j])
                H_new[i][j] = term1 + term2 - term3

        return H_new


class LBFGS:
    """
    Limited-memory BFGS.

    Memory-efficient version of BFGS for large-scale problems.
    Stores only recent updates instead of full Hessian approximation.
    """

    def __init__(self, m: int = 10, max_iter: int = 100, tol: float = 1e-6):
        """
        Args:
            m: Number of correction pairs to store
            max_iter: Maximum iterations
            tol: Tolerance
        """
        self.m = m
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        f: Callable[[List[float]], float],
        grad_f: Callable[[List[float]], List[float]],
        x0: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Minimize using L-BFGS."""
        x = x0[:]
        grad = grad_f(x)
        history = [f(x)]

        # Use deque with maxlen for O(1) eviction of oldest pairs (fixes QUALITY-003).
        s_list: deque = deque(maxlen=self.m)  # s_k = x_{k+1} - x_k
        y_list: deque = deque(maxlen=self.m)  # y_k = grad_{k+1} - grad_k

        for iteration in range(self.max_iter):
            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in grad))
            if grad_norm < self.tol:
                break

            # Compute search direction using two-loop recursion
            direction = self._two_loop_recursion(grad, s_list, y_list)

            # Line search
            alpha = self._line_search(f, x, direction, grad)

            # Update
            x_new = [xi + alpha * di for xi, di in zip(x, direction)]
            grad_new = grad_f(x_new)

            # Store correction pair only when curvature condition s·y > 0 holds.
            # Negative or zero curvature would corrupt the two-loop recursion.
            s = [x_new[i] - x[i] for i in range(len(x))]
            y = [grad_new[i] - grad[i] for i in range(len(grad))]
            s_dot_y = sum(s[j] * y[j] for j in range(len(s)))
            if s_dot_y > 1e-10:
                # deque(maxlen=m) automatically evicts the oldest pair; no pop(0) needed
                s_list.append(s)
                y_list.append(y)

            x = x_new
            grad = grad_new
            history.append(f(x))

        return x, history

    def _two_loop_recursion(self, grad, s_list, y_list):
        """L-BFGS two-loop recursion."""
        q = grad[:]
        n = len(q)
        m = len(s_list)

        alpha_list = []

        # First loop (backward)
        for i in range(m - 1, -1, -1):
            s_dot_y = sum(s_list[i][j] * y_list[i][j] for j in range(n))
            if abs(s_dot_y) < 1e-10:
                alpha_i = 0.0
            else:
                rho_i = 1.0 / s_dot_y
                alpha_i = rho_i * sum(s_list[i][j] * q[j] for j in range(n))
                q = [q[j] - alpha_i * y_list[i][j] for j in range(n)]
            alpha_list.append(alpha_i)

        # Scaling
        if m > 0:
            s_dot_y = sum(s_list[-1][j] * y_list[-1][j] for j in range(n))
            y_dot_y = sum(y_list[-1][j] * y_list[-1][j] for j in range(n))
            if abs(y_dot_y) > 1e-10:
                gamma = s_dot_y / y_dot_y
                r = [gamma * q[j] for j in range(n)]
            else:
                r = q
        else:
            r = q

        # Second loop (forward)
        alpha_list.reverse()
        for i in range(m):
            s_dot_y = sum(s_list[i][j] * y_list[i][j] for j in range(n))
            if abs(s_dot_y) < 1e-10:
                continue
            rho_i = 1.0 / s_dot_y
            beta = rho_i * sum(y_list[i][j] * r[j] for j in range(n))
            r = [r[j] + s_list[i][j] * (alpha_list[i] - beta) for j in range(n)]

        return [-ri for ri in r]

    def _line_search(self, f, x, direction, grad):
        """Delegate to shared second-order backtracking line search."""
        return _so_line_search(f, x, direction, grad)

    def reset(self):
        """No persistent state between optimize() calls; provided for API consistency."""
        pass


class ConjugateGradient:
    """
    Conjugate Gradient method.

    Efficient for quadratic functions, extends to non-quadratic via
    nonlinear conjugate gradient.
    """

    def __init__(self, max_iter: int = 100, tol: float = 1e-6):
        self.max_iter = max_iter
        self.tol = tol

    def optimize(
        self,
        f: Callable[[List[float]], float],
        grad_f: Callable[[List[float]], List[float]],
        x0: List[float]
    ) -> Tuple[List[float], List[float]]:
        """Minimize using Conjugate Gradient."""
        x = x0[:]
        grad = grad_f(x)
        direction = [-g for g in grad]  # Initial direction: steepest descent
        history = [f(x)]

        for iteration in range(self.max_iter):
            # Check convergence
            grad_norm = math.sqrt(sum(g**2 for g in grad))
            if grad_norm < self.tol:
                break

            # Line search
            alpha = self._line_search(f, x, direction, grad)

            # Update x
            x = [xi + alpha * di for xi, di in zip(x, direction)]

            # New gradient
            grad_new = grad_f(x)

            # Compute beta (Fletcher-Reeves)
            grad_norm_new_sq = sum(g**2 for g in grad_new)
            grad_norm_sq = sum(g**2 for g in grad)

            if abs(grad_norm_sq) > 1e-10:
                beta = grad_norm_new_sq / grad_norm_sq
            else:
                beta = 0.0

            # Periodic restart every n steps (Powell's criterion): reset to
            # steepest descent to prevent non-conjugate direction accumulation
            # on non-quadratic objectives.
            if (iteration + 1) % len(x0) == 0:
                beta = 0.0

            # Update direction
            direction = [-g + beta * d for g, d in zip(grad_new, direction)]

            grad = grad_new
            history.append(f(x))

        return x, history

    def _line_search(self, f, x, direction, grad):
        """Delegate to shared second-order backtracking line search."""
        return _so_line_search(f, x, direction, grad)

    def reset(self):
        """No persistent state between optimize() calls; provided for API consistency."""
        pass

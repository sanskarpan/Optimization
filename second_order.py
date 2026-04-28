"""
Second-Order Optimization Methods
===================================
Pure-Python implementations of second-order and quasi-Newton optimization
algorithms. No external dependencies — stdlib only (math, random, typing,
warnings).

All matrix operations are performed on plain Python ``List[List[float]]``
objects using private helper functions defined at the top of this module.

Exported functions
------------------
* ``newton_raphson``       — Exact Newton with Cholesky solver + ridge fallback
* ``bfgs``                 — Full BFGS with inverse Hessian update
* ``lbfgs``                — Limited-memory BFGS (two-loop recursion)
* ``sr1``                  — Symmetric Rank-1 quasi-Newton
* ``gauss_newton``         — Gauss-Newton for nonlinear least squares
* ``levenberg_marquardt``  — Levenberg-Marquardt damped least squares
* ``trust_region``         — Trust-region with Steihaug CG subproblem
* ``newton_cg``            — Truncated Newton (CG to solve Newton system)

References
----------
* Nocedal & Wright (2006) — Numerical Optimization, 2nd ed.
* Conn, Gould & Toint (2000) — Trust-Region Methods.
"""

import math
import warnings
from typing import Callable, List, Optional, Tuple

__all__ = [
    'newton_raphson',
    'bfgs',
    'lbfgs',
    'sr1',
    'gauss_newton',
    'levenberg_marquardt',
    'trust_region',
    'newton_cg',
]

# ---------------------------------------------------------------------------
# Private matrix / linear-algebra helpers
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    """Dot product of two vectors."""
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    """Matrix-vector product  A @ v."""
    return [_dot(row, v) for row in A]


def _outer(a: List[float], b: List[float]) -> List[List[float]]:
    """Outer product  a * b^T."""
    return [[ai * bj for bj in b] for ai in a]


def _eye(n: int) -> List[List[float]]:
    """n x n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Element-wise matrix addition."""
    n = len(A)
    m = len(A[0])
    return [[A[i][j] + B[i][j] for j in range(m)] for i in range(n)]


def _mat_scale(A: List[List[float]], s: float) -> List[List[float]]:
    """Scalar multiplication of a matrix."""
    return [[A[i][j] * s for j in range(len(A[i]))] for i in range(len(A))]


def _mat_mul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Matrix-matrix product  A @ B."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for k in range(p):
            if A[i][k] == 0.0:
                continue
            for j in range(m):
                C[i][j] += A[i][k] * B[k][j]
    return C


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    """Element-wise vector addition."""
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    """Element-wise vector subtraction."""
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_scale(a: List[float], s: float) -> List[float]:
    """Scale vector by scalar."""
    return [ai * s for ai in a]


def _norm(v: List[float]) -> float:
    """Euclidean norm of a vector."""
    return math.sqrt(sum(vi * vi for vi in v))


def _cholesky(A: List[List[float]]) -> List[List[float]]:
    """Cholesky decomposition: return lower-triangular L such that A = L L^T.

    Raises ValueError if A is not positive-definite.
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = A[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError(
                        f"Matrix is not positive-definite (diagonal entry {s} at index {i})"
                    )
                L[i][j] = math.sqrt(s)
            else:
                L[i][j] = s / L[j][j]
    return L


def _chol_solve(L: List[List[float]], b: List[float]) -> List[float]:
    """Solve the system A x = b given the Cholesky factor L (A = L L^T).

    Uses forward substitution (L y = b) then back substitution (L^T x = y).
    """
    n = len(L)
    # Forward substitution: L y = b
    y = [0.0] * n
    for i in range(n):
        y[i] = (b[i] - sum(L[i][j] * y[j] for j in range(i))) / L[i][i]
    # Back substitution: L^T x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x


def _lu_decompose(A: List[List[float]]) -> Tuple[List[List[float]], List[int]]:
    """LU decomposition with partial pivoting.

    Returns (LU, piv) where LU stores both L (strictly lower) and U (upper)
    in-place, and piv is the pivot index array.
    """
    n = len(A)
    # Work on a copy
    M = [row[:] for row in A]
    piv = list(range(n))
    for k in range(n):
        # Find pivot
        max_val = abs(M[k][k])
        max_row = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_row = i
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
            piv[k], piv[max_row] = piv[max_row], piv[k]
        if abs(M[k][k]) < 1e-15:
            # Singular; leave as-is (caller handles)
            continue
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            M[i][k] = factor  # store L below diagonal
            for j in range(k + 1, n):
                M[i][j] -= factor * M[k][j]
    return M, piv


def _lu_solve(LU: List[List[float]], piv: List[int], b: List[float]) -> List[float]:
    """Solve A x = b given an LU factorisation with pivoting."""
    n = len(LU)
    # Apply permutation
    pb = [b[piv[i]] for i in range(n)]
    # Forward substitution: L y = pb (L has implicit 1s on diagonal)
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(LU[i][j] * y[j] for j in range(i))
    # Back substitution: U x = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(LU[i][i]) < 1e-15:
            x[i] = 0.0
        else:
            x[i] = (y[i] - sum(LU[i][j] * x[j] for j in range(i + 1, n))) / LU[i][i]
    return x


def _solve_linear(A: List[List[float]], b: List[float]) -> List[float]:
    """Solve the linear system A x = b.

    Tries Cholesky first (faster, requires PD); falls back to LU.
    """
    try:
        L = _cholesky(A)
        return _chol_solve(L, b)
    except ValueError:
        LU, piv = _lu_decompose(A)
        return _lu_solve(LU, piv, b)


# ---------------------------------------------------------------------------
# Backtracking line search (Armijo condition) — used internally
# ---------------------------------------------------------------------------

def _backtracking(
    f: Callable[[List[float]], float],
    x: List[float],
    d: List[float],
    g: List[float],
    alpha: float = 1.0,
    rho: float = 0.5,
    c1: float = 1e-4,
    max_ls: int = 50,
) -> float:
    """Return step size satisfying the Armijo sufficient-decrease condition.

    f(x + alpha*d) <= f(x) + c1 * alpha * grad^T d
    """
    f0 = f(x)
    slope = _dot(g, d)
    for _ in range(max_ls):
        x_new = _vec_add(x, _vec_scale(d, alpha))
        if f(x_new) <= f0 + c1 * alpha * slope:
            return alpha
        alpha *= rho
    return alpha


# ---------------------------------------------------------------------------
# 1. Newton-Raphson
# ---------------------------------------------------------------------------

def newton_raphson(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    hess_f: Callable[[List[float]], List[List[float]]],
    x0: List[float],
    tol: float = 1e-6,
    max_iter: int = 100,
    alpha: float = 1.0,
) -> Tuple[List[float], float, int, bool]:
    """Pure Newton optimisation with exact Hessian.

    Parameters
    ----------
    f:
        Objective function ``f(x) -> float``.
    grad_f:
        Gradient function ``grad_f(x) -> List[float]``.
    hess_f:
        Hessian function ``hess_f(x) -> List[List[float]]``.
    x0:
        Initial point.
    tol:
        Convergence tolerance on gradient norm (default ``1e-6``).
    max_iter:
        Maximum number of Newton steps (default ``100``).
    alpha:
        Fixed step size multiplier (default ``1.0``).

    Returns
    -------
    x_opt:
        Approximate minimiser.
    f_opt:
        Objective value at ``x_opt``.
    n_iters:
        Number of iterations performed.
    converged:
        ``True`` if ``||grad|| < tol`` at termination.

    Notes
    -----
    The Newton direction solves  H d = -g  via Cholesky factorisation when
    the Hessian is positive-definite.  If Cholesky fails (H not PD), ridge
    regularisation (λ I) is added and increased until the modified system is
    solvable.
    """
    x = x0[:]
    n = len(x)
    converged = False
    n_iters = 0

    for _ in range(max_iter):
        n_iters += 1
        g = grad_f(x)
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        H = hess_f(x)
        neg_g = [-gi for gi in g]

        # Try Cholesky; if H is not PD, add increasing ridge regularisation
        lam = 0.0
        d = None
        for _r in range(50):
            H_reg = [
                [H[i][j] + (lam if i == j else 0.0) for j in range(n)]
                for i in range(n)
            ]
            try:
                L = _cholesky(H_reg)
                d = _chol_solve(L, neg_g)
                break
            except ValueError:
                lam = max(1e-6, lam * 10) if lam > 0 else 1e-6

        if d is None:
            warnings.warn("Newton-Raphson: could not find a descent direction; stopping.")
            break

        x = _vec_add(x, _vec_scale(d, alpha))

    return x, f(x), n_iters, converged


# ---------------------------------------------------------------------------
# 2. BFGS
# ---------------------------------------------------------------------------

def bfgs(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    tol: float = 1e-6,
    max_iter: int = 200,
    alpha0: float = 1.0,
) -> Tuple[List[float], float, int, bool]:
    """Full BFGS with inverse Hessian approximation.

    Implements the standard BFGS update for the *inverse* Hessian H_k:

    .. math::
        H_{k+1} = (I - \\rho_k s_k y_k^T) H_k (I - \\rho_k y_k s_k^T)
                  + \\rho_k s_k s_k^T

    with  :math:`\\rho_k = 1 / y_k^T s_k`.

    The search direction is  d_k = -H_k g_k  and the step size is found by
    backtracking Armijo line search.

    Parameters
    ----------
    f, grad_f, x0, tol, max_iter, alpha0:
        See ``newton_raphson`` for common parameters.

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    n = len(x0)
    x = x0[:]
    g = grad_f(x)
    H = _eye(n)  # inverse Hessian approximation
    converged = False

    for k in range(max_iter):
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        # Search direction
        d = _mat_vec(H, [-gi for gi in g])

        # Ensure descent direction
        if _dot(d, g) >= 0:
            d = [-gi for gi in g]  # fall back to steepest descent

        # Backtracking line search (Armijo)
        step = _backtracking(f, x, d, g, alpha=alpha0)

        # Update x
        x_new = _vec_add(x, _vec_scale(d, step))
        g_new = grad_f(x_new)

        s = _vec_sub(x_new, x)
        y = _vec_sub(g_new, g)
        sy = _dot(s, y)

        # BFGS inverse Hessian update (skip if curvature condition fails)
        if sy > 1e-10:
            rho = 1.0 / sy
            # H_{k+1} = (I - rho s y^T) H_k (I - rho y s^T) + rho s s^T
            # Compute  v = H_k y  first
            Hy = _mat_vec(H, y)
            # Build rank-2 update directly
            # H_new = H - rho*(H*y*s^T + s*y^T*H) + rho*(s^T*H*y + 1)*s*s^T
            yTHy = _dot(y, Hy)
            # H_{k+1}[i][j] = H[i][j]
            #   - rho*(Hy[i]*s[j] + s[i]*Hy[j])      [symmetrised cross terms]
            #   + rho*(rho*yTHy + 1)*s[i]*s[j]
            H_new = [
                [
                    H[i][j]
                    - rho * (Hy[i] * s[j] + s[i] * Hy[j])
                    + rho * (rho * yTHy + 1.0) * s[i] * s[j]
                    for j in range(n)
                ]
                for i in range(n)
            ]
            H = H_new

        x = x_new
        g = g_new

    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# 3. L-BFGS
# ---------------------------------------------------------------------------

def lbfgs(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    m: int = 10,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[List[float], float, int, bool]:
    """Limited-memory BFGS.

    Stores the last ``m`` curvature pairs (s_k, y_k) and applies the
    two-loop recursion to compute the search direction  d = -H_k g  without
    ever forming H_k explicitly.

    Parameters
    ----------
    f, grad_f, x0:
        See ``bfgs``.
    m:
        History size (default ``10``).
    tol, max_iter:
        See ``newton_raphson``.

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    x = x0[:]
    g = grad_f(x)
    converged = False

    # History stored as plain lists (we rotate manually)
    s_hist: List[List[float]] = []
    y_hist: List[List[float]] = []
    rho_hist: List[float] = []

    for k in range(max_iter):
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        # Two-loop recursion
        q = g[:]
        alphas = []
        hist_len = len(s_hist)
        for i in range(hist_len - 1, -1, -1):
            ai = rho_hist[i] * _dot(s_hist[i], q)
            alphas.append(ai)
            q = _vec_sub(q, _vec_scale(y_hist[i], ai))
        alphas.reverse()

        # Initial Hessian scaling: gamma = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}
        if hist_len > 0:
            sy = _dot(s_hist[-1], y_hist[-1])
            yy = _dot(y_hist[-1], y_hist[-1])
            gamma = sy / yy if yy > 1e-15 else 1.0
        else:
            gamma = 1.0

        r = _vec_scale(q, gamma)

        for i in range(hist_len):
            beta = rho_hist[i] * _dot(y_hist[i], r)
            r = _vec_add(r, _vec_scale(s_hist[i], alphas[i] - beta))

        d = [-ri for ri in r]  # search direction

        # Ensure descent
        if _dot(d, g) >= 0:
            d = [-gi for gi in g]

        # Backtracking line search
        step = _backtracking(f, x, d, g)

        x_new = _vec_add(x, _vec_scale(d, step))
        g_new = grad_f(x_new)

        s = _vec_sub(x_new, x)
        y = _vec_sub(g_new, g)
        sy = _dot(s, y)

        if sy > 1e-10:
            if len(s_hist) >= m:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s)
            y_hist.append(y)
            rho_hist.append(1.0 / sy)

        x = x_new
        g = g_new

    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# 4. SR1
# ---------------------------------------------------------------------------

def sr1(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    x0: List[float],
    tol: float = 1e-6,
    max_iter: int = 200,
    r: float = 1e-8,
) -> Tuple[List[float], float, int, bool]:
    """Symmetric Rank-1 quasi-Newton (inverse Hessian form).

    The SR1 update for the inverse Hessian approximation H_k is:

    .. math::
        H_{k+1} = H_k + \\frac{(s - H y)(s - H y)^T}{(s - H y)^T y}

    The update is skipped when the denominator is small relative to the
    norms involved (safeguard controlled by ``r``).

    Parameters
    ----------
    r:
        Skip-update threshold: update is skipped when
        ``|(s - H y)^T y| < r * ||s - H y|| * ||y||`` (default ``1e-8``).

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    n = len(x0)
    x = x0[:]
    g = grad_f(x)
    H = _eye(n)  # inverse Hessian approximation
    converged = False

    for k in range(max_iter):
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        d = _mat_vec(H, [-gi for gi in g])

        # Ensure descent
        if _dot(d, g) >= 0:
            d = [-gi for gi in g]

        step = _backtracking(f, x, d, g)

        x_new = _vec_add(x, _vec_scale(d, step))
        g_new = grad_f(x_new)

        s = _vec_sub(x_new, x)
        y = _vec_sub(g_new, g)

        # SR1 inverse Hessian update
        Hy = _mat_vec(H, y)
        v = _vec_sub(s, Hy)               # s - H y
        denom = _dot(v, y)                  # (s - H y)^T y
        v_norm = _norm(v)
        y_norm = _norm(y)
        # Skip if denominator is too small
        if abs(denom) >= r * v_norm * y_norm and abs(denom) > 1e-15:
            scale = 1.0 / denom
            # H_{k+1} = H_k + scale * v v^T
            for i in range(n):
                for j in range(n):
                    H[i][j] += scale * v[i] * v[j]

        x = x_new
        g = g_new

    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# 5. Gauss-Newton
# ---------------------------------------------------------------------------

def gauss_newton(
    residuals_f: Callable[[List[float]], List[float]],
    jacobian_f: Callable[[List[float]], List[List[float]]],
    x0: List[float],
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[List[float], float, int, bool]:
    """Gauss-Newton method for nonlinear least-squares.

    Minimises  ``0.5 * ||r(x)||^2``  by iterating the linearised subproblem:

    .. math::
        (J^T J) d = -J^T r

    Parameters
    ----------
    residuals_f:
        Callable ``r(x) -> List[float]`` returning the residual vector.
    jacobian_f:
        Callable ``J(x) -> List[List[float]]`` returning the m×n Jacobian.
    x0:
        Initial point (length n).
    tol:
        Convergence tolerance on ``||J^T r||`` (default ``1e-6``).
    max_iter:
        Maximum number of iterations (default ``100``).

    Returns
    -------
    x_opt:
        Approximate minimiser.
    residual_norm:
        ``||r(x_opt)||`` at termination.
    n_iters:
        Number of iterations performed.
    converged:
        ``True`` if ``||J^T r|| < tol`` at termination.
    """
    x = x0[:]
    n = len(x)
    converged = False

    for k in range(max_iter):
        r = residuals_f(x)
        J = jacobian_f(x)   # m x n

        # J^T r
        JTr = [sum(J[i][j] * r[i] for i in range(len(r))) for j in range(n)]
        grad_norm = _norm(JTr)
        if grad_norm < tol:
            converged = True
            break

        # J^T J  (n x n)
        JTJ = [[sum(J[i][row] * J[i][col] for i in range(len(r)))
                 for col in range(n)] for row in range(n)]

        neg_JTr = [-v for v in JTr]
        try:
            d = _solve_linear(JTJ, neg_JTr)
        except Exception:
            warnings.warn("Gauss-Newton: singular J^T J; stopping.")
            break

        # Simple backtracking on ||r||^2
        f_ls = lambda z: sum(ri * ri for ri in residuals_f(z))
        g_ls = [2 * v for v in JTr]
        step = _backtracking(f_ls, x, d, g_ls)

        x = _vec_add(x, _vec_scale(d, step))

    r_final = residuals_f(x)
    return x, _norm(r_final), k + 1, converged


# ---------------------------------------------------------------------------
# 6. Levenberg-Marquardt
# ---------------------------------------------------------------------------

def levenberg_marquardt(
    residuals_f: Callable[[List[float]], List[float]],
    jacobian_f: Callable[[List[float]], List[List[float]]],
    x0: List[float],
    lam: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
    lam_factor: float = 10.0,
) -> Tuple[List[float], float, int, bool]:
    """Levenberg-Marquardt algorithm for nonlinear least-squares.

    Solves a damped linear system at each iteration:

    .. math::
        (J^T J + \\lambda I) d = -J^T r

    ``lambda`` is increased when a trial step fails to reduce the residual
    norm and decreased on success, providing an interpolation between
    Gauss-Newton (small λ) and steepest descent (large λ).

    Parameters
    ----------
    lam:
        Initial damping parameter (default ``1.0``).
    lam_factor:
        Multiplicative factor for adapting λ (default ``10.0``).

    Returns
    -------
    (x_opt, residual_norm, n_iters, converged)
    """
    x = x0[:]
    n = len(x)
    converged = False

    for k in range(max_iter):
        r = residuals_f(x)
        J = jacobian_f(x)   # m x n
        m_res = len(r)

        JTr = [sum(J[i][j] * r[i] for i in range(m_res)) for j in range(n)]
        grad_norm = _norm(JTr)
        if grad_norm < tol:
            converged = True
            break

        JTJ = [[sum(J[i][row] * J[i][col] for i in range(m_res))
                 for col in range(n)] for row in range(n)]

        # Damped system: (J^T J + lam * I) d = -J^T r
        for _ in range(50):   # try increasing lam until system is solvable
            JTJ_lam = [[JTJ[i][j] + (lam if i == j else 0.0) for j in range(n)]
                        for i in range(n)]
            neg_JTr = [-v for v in JTr]
            try:
                d = _solve_linear(JTJ_lam, neg_JTr)
            except Exception:
                lam *= lam_factor
                continue

            x_new = _vec_add(x, d)
            r_new = residuals_f(x_new)
            cost_old = sum(ri * ri for ri in r)
            cost_new = sum(ri * ri for ri in r_new)

            if cost_new < cost_old:
                x = x_new
                lam = max(lam / lam_factor, 1e-15)
                break
            else:
                lam *= lam_factor
        else:
            # Lam blew up without progress; accept last d anyway to avoid stall
            pass

    r_final = residuals_f(x)
    return x, _norm(r_final), k + 1, converged


# ---------------------------------------------------------------------------
# 7. Trust-Region (Steihaug CG subproblem)
# ---------------------------------------------------------------------------

def _steihaug_cg(
    g: List[float],
    H: List[List[float]],
    delta: float,
    tol: float = 1e-10,
    max_cg: int = 200,
) -> List[float]:
    """Steihaug truncated CG for the trust-region subproblem.

    Minimise  m(d) = g^T d + 0.5 d^T H d  subject to  ||d|| <= delta.

    Returns the step d.
    """
    n = len(g)
    d = [0.0] * n
    r = g[:]          # residual = g + H d = g when d=0
    p = [-ri for ri in r]  # search direction

    r_norm_sq = _dot(r, r)
    if math.sqrt(r_norm_sq) < tol:
        return d

    for _ in range(max_cg):
        Hp = _mat_vec(H, p)
        pHp = _dot(p, Hp)

        if pHp <= 0:
            # Negative curvature: go to boundary
            # Solve ||d + tau*p||^2 = delta^2 for tau
            d_norm_sq = _dot(d, d)
            dp = _dot(d, p)
            p_norm_sq = _dot(p, p)
            discriminant = dp * dp - p_norm_sq * (d_norm_sq - delta * delta)
            tau = (-dp + math.sqrt(max(0.0, discriminant))) / p_norm_sq
            return _vec_add(d, _vec_scale(p, tau))

        alpha = r_norm_sq / pHp
        d_new = _vec_add(d, _vec_scale(p, alpha))

        if _norm(d_new) >= delta:
            # Hit boundary: go to boundary
            d_norm_sq = _dot(d, d)
            dp = _dot(d, p)
            p_norm_sq = _dot(p, p)
            discriminant = dp * dp - p_norm_sq * (d_norm_sq - delta * delta)
            tau = (-dp + math.sqrt(max(0.0, discriminant))) / p_norm_sq
            return _vec_add(d, _vec_scale(p, tau))

        d = d_new
        r_new = _vec_add(r, _vec_scale(Hp, alpha))
        r_norm_sq_new = _dot(r_new, r_new)

        if math.sqrt(r_norm_sq_new) < tol:
            return d

        beta = r_norm_sq_new / r_norm_sq
        p = _vec_add([-ri for ri in r_new], _vec_scale(p, beta))
        r = r_new
        r_norm_sq = r_norm_sq_new

    return d


def trust_region(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    hess_f: Callable[[List[float]], List[List[float]]],
    x0: List[float],
    delta0: float = 1.0,
    delta_max: float = 100.0,
    eta: float = 0.125,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[List[float], float, int, bool]:
    """Trust-region method with Steihaug CG subproblem solver.

    At each iteration the algorithm:

    1. Solves the trust-region subproblem approximately using Steihaug CG.
    2. Computes the ratio  ρ = (f(x) - f(x+d)) / (m(0) - m(d)),
       where m(d) = f + g^T d + 0.5 d^T H d is the quadratic model.
    3. Accepts or rejects the step and updates the trust-region radius δ.

    Parameters
    ----------
    delta0:
        Initial trust-region radius (default ``1.0``).
    delta_max:
        Maximum trust-region radius (default ``100.0``).
    eta:
        Minimum acceptable ratio for step acceptance (default ``0.125``).

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    x = x0[:]
    delta = delta0
    converged = False

    for k in range(max_iter):
        g = grad_f(x)
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        H = hess_f(x)
        d = _steihaug_cg(g, H, delta)

        f_x = f(x)
        f_xd = f(_vec_add(x, d))

        # Actual reduction
        actual = f_x - f_xd
        # Predicted reduction from quadratic model: -(g^T d + 0.5 d^T H d)
        Hd = _mat_vec(H, d)
        predicted = -((_dot(g, d) + 0.5 * _dot(d, Hd)))

        if abs(predicted) < 1e-15:
            rho = 1.0 if actual >= 0 else 0.0
        else:
            rho = actual / predicted

        # Update trust radius
        d_norm = _norm(d)
        if rho < 0.25:
            delta = 0.25 * d_norm
            delta = max(delta, 1e-10)
        elif rho > 0.75 and abs(d_norm - delta) < 1e-10:
            delta = min(2.0 * delta, delta_max)

        # Accept or reject step
        if rho > eta:
            x = _vec_add(x, d)

    return x, f(x), k + 1, converged


# ---------------------------------------------------------------------------
# 8. Newton-CG (Truncated Newton)
# ---------------------------------------------------------------------------

def newton_cg(
    f: Callable[[List[float]], float],
    grad_f: Callable[[List[float]], List[float]],
    hess_f: Callable[[List[float]], List[List[float]]],
    x0: List[float],
    tol: float = 1e-6,
    max_iter: int = 100,
    cg_tol: float = 0.5,
) -> Tuple[List[float], float, int, bool]:
    """Newton-CG (truncated Newton) optimisation.

    Uses the conjugate gradient method to *approximately* solve the Newton
    system  H d = -g  at each outer iteration.  The CG solver terminates
    when the residual norm satisfies the forcing-sequence condition:

    .. math::
        ||r|| \\le \\min(cg\\_tol,\\, \\sqrt{||g||}) \\cdot ||g||

    This avoids over-solving and is appropriate for inexact Newton methods.
    A backtracking line search is used on the resulting direction.

    Parameters
    ----------
    cg_tol:
        Coefficient in the forcing sequence (default ``0.5``).

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    x = x0[:]
    converged = False

    for k in range(max_iter):
        g = grad_f(x)
        g_norm = _norm(g)
        if g_norm < tol:
            converged = True
            break

        H = hess_f(x)
        n = len(x)

        # CG stopping tolerance (forcing sequence)
        eta_k = min(cg_tol, math.sqrt(g_norm)) * g_norm

        # CG to solve H d = -g
        d = [0.0] * n
        r = g[:]            # residual r = g + H*0 = g
        p = [-ri for ri in r]
        r_norm_sq = _dot(r, r)

        for _ in range(n * 2 + 10):
            if math.sqrt(r_norm_sq) <= eta_k:
                break
            Hp = _mat_vec(H, p)
            pHp = _dot(p, Hp)
            if pHp <= 0:
                # Negative curvature: use steepest-descent as fallback
                if _norm(d) < 1e-12:
                    d = [-gi for gi in g]
                break
            alpha_cg = r_norm_sq / pHp
            d = _vec_add(d, _vec_scale(p, alpha_cg))
            r = _vec_add(r, _vec_scale(Hp, alpha_cg))
            r_norm_sq_new = _dot(r, r)
            beta = r_norm_sq_new / r_norm_sq
            p = _vec_add([-ri for ri in r], _vec_scale(p, beta))
            r_norm_sq = r_norm_sq_new

        # Ensure d is a descent direction
        if _dot(d, g) >= 0:
            d = [-gi for gi in g]

        # Backtracking line search
        step = _backtracking(f, x, d, g)
        x = _vec_add(x, _vec_scale(d, step))

    return x, f(x), k + 1, converged

"""
global_opt.py
=============

Global optimization algorithms implemented in pure Python (stdlib only).

Algorithms
----------
- simulated_annealing   : Classic SA with geometric cooling
- genetic_algorithm     : Real-valued GA with SBX crossover & polynomial mutation
- differential_evolution: DE with rand/1, best/1, current-to-best/1 strategies
- particle_swarm        : Standard PSO
- nelder_mead           : Full Nelder-Mead simplex method
- cma_es                : CMA-ES (Cholesky-based, Hansen 2006)
- basin_hopping         : Perturbation + local opt + Metropolis accept
- random_search         : Pure random search over bounds
- latin_hypercube_search: LHS stratified sampling
"""

__all__ = [
    'simulated_annealing',
    'genetic_algorithm',
    'differential_evolution',
    'particle_swarm',
    'nelder_mead',
    'cma_es',
    'basin_hopping',
    'random_search',
    'latin_hypercube_search',
]

import math
import random
import warnings
from typing import Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

def _box_muller(rng: random.Random) -> float:
    """Return one standard-normal sample via Box-Muller transform."""
    while True:
        u1 = rng.random()
        u2 = rng.random()
        if u1 > 0.0:
            break
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _randn_vec(n: int, rng: random.Random) -> List[float]:
    """Return a list of n iid N(0,1) samples."""
    return [_box_muller(rng) for _ in range(n)]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _clip_vec(x: List[float], bounds: List[Tuple[float, float]]) -> List[float]:
    return [_clip(xi, lo, hi) for xi, (lo, hi) in zip(x, bounds)]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [ai + bi for ai, bi in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [ai - bi for ai, bi in zip(a, b)]


def _vec_scale(s: float, a: List[float]) -> List[float]:
    return [s * ai for ai in a]


def _vec_dot(a: List[float], b: List[float]) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _mat_vec(M: List[List[float]], v: List[float]) -> List[float]:
    """Multiply matrix M (row-major list-of-lists) by vector v."""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _cholesky(A: List[List[float]]) -> Optional[List[List[float]]]:
    """
    Lower-triangular Cholesky factor L of positive-definite matrix A.
    Returns None if A is not positive-definite (fall back to diagonal).
    """
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                val = A[i][i] - s
                if val <= 0.0:
                    return None
                L[i][j] = math.sqrt(val)
            else:
                L[i][j] = (A[i][j] - s) / L[j][j]
    return L


def _outer(a: List[float], b: List[float]) -> List[List[float]]:
    """Outer product a ⊗ b."""
    return [[ai * bi for bi in b] for ai in a]


def _mat_add(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(n)] for i in range(n)]


def _mat_scale(s: float, A: List[List[float]]) -> List[List[float]]:
    return [[s * A[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def _identity(n: int) -> List[List[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Simulated Annealing
# ---------------------------------------------------------------------------

def simulated_annealing(
    f: Callable[[List[float]], float],
    x0: List[float],
    T0: float = 1.0,
    T_min: float = 1e-5,
    alpha: float = 0.95,
    n_steps: int = 100,
    step_size: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[List[float], float, int]:
    """
    Simulated annealing with geometric cooling.

    Parameters
    ----------
    f         : Objective function (minimisation).
    x0        : Initial point.
    T0        : Initial temperature.
    T_min     : Minimum temperature (stopping criterion).
    alpha     : Cooling rate (T *= alpha each epoch).
    n_steps   : Number of proposals per temperature level.
    step_size : Standard deviation of the Normal proposal.
    seed      : Random seed.

    Returns
    -------
    (x_best, f_best, n_evaluations)
    """
    rng = random.Random(seed)
    n = len(x0)
    x_cur = list(x0)
    f_cur = f(x_cur)
    x_best = list(x_cur)
    f_best = f_cur
    n_evals = 1
    T = T0

    while T > T_min:
        for _ in range(n_steps):
            # Box-Muller proposal
            x_new = [x_cur[i] + step_size * _box_muller(rng) for i in range(n)]
            f_new = f(x_new)
            n_evals += 1
            delta = f_new - f_cur
            if delta < 0.0 or rng.random() < math.exp(-delta / T):
                x_cur = x_new
                f_cur = f_new
                if f_cur < f_best:
                    x_best = list(x_cur)
                    f_best = f_cur
        T *= alpha

    return x_best, f_best, n_evals


# ---------------------------------------------------------------------------
# 2. Genetic Algorithm
# ---------------------------------------------------------------------------

def genetic_algorithm(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    n_gens: int = 100,
    mutation_rate: float = 0.01,
    crossover_rate: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Real-valued GA with tournament selection, SBX crossover, polynomial mutation.

    Parameters
    ----------
    f             : Objective function (minimisation).
    bounds        : [(lo, hi)] per dimension.
    pop_size      : Population size.
    n_gens        : Number of generations.
    mutation_rate : Probability of mutating each gene.
    crossover_rate: Probability of applying crossover to a pair.
    seed          : Random seed.

    Returns
    -------
    (x_best, f_best)
    """
    rng = random.Random(seed)
    ndim = len(bounds)
    eta_c = 20.0   # SBX distribution index
    eta_m = 20.0   # Polynomial mutation distribution index

    def random_individual() -> List[float]:
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    def sbx_crossover(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        c1, c2 = list(p1), list(p2)
        if rng.random() > crossover_rate:
            return c1, c2
        for i in range(ndim):
            if rng.random() < 0.5:
                lo, hi = bounds[i]
                if abs(p2[i] - p1[i]) > 1e-14:
                    y1, y2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                    rand = rng.random()
                    # beta
                    beta = 1.0 + 2.0 * (y1 - lo) / (y2 - y1)
                    alpha = 2.0 - beta ** (-(eta_c + 1.0))
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    c1[i] = _clip(0.5 * ((y1 + y2) - beta_q * (y2 - y1)), lo, hi)

                    beta = 1.0 + 2.0 * (hi - y2) / (y2 - y1)
                    alpha = 2.0 - beta ** (-(eta_c + 1.0))
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta_c + 1.0))
                    c2[i] = _clip(0.5 * ((y1 + y2) + beta_q * (y2 - y1)), lo, hi)
        return c1, c2

    def poly_mutate(x: List[float]) -> List[float]:
        x = list(x)
        for i in range(ndim):
            if rng.random() < mutation_rate:
                lo, hi = bounds[i]
                delta1 = (x[i] - lo) / (hi - lo + 1e-300)
                delta2 = (hi - x[i]) / (hi - lo + 1e-300)
                rand = rng.random()
                mut_pow = 1.0 / (eta_m + 1.0)
                if rand < 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1.0))
                    delta_q = 1.0 - val ** mut_pow
                x[i] = _clip(x[i] + delta_q * (hi - lo), lo, hi)
        return x

    def tournament_select(pop: List[List[float]], fits: List[float]) -> List[float]:
        a, b = rng.randrange(pop_size), rng.randrange(pop_size)
        return list(pop[a] if fits[a] < fits[b] else pop[b])

    # Initialise
    pop = [random_individual() for _ in range(pop_size)]
    fits = [f(ind) for ind in pop]

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    x_best, f_best = list(pop[best_idx]), fits[best_idx]

    for _ in range(n_gens):
        new_pop: List[List[float]] = []
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits)
            p2 = tournament_select(pop, fits)
            c1, c2 = sbx_crossover(p1, p2)
            new_pop.append(poly_mutate(c1))
            if len(new_pop) < pop_size:
                new_pop.append(poly_mutate(c2))
        pop = new_pop
        fits = [f(ind) for ind in pop]
        bi = min(range(pop_size), key=lambda i: fits[i])
        if fits[bi] < f_best:
            x_best, f_best = list(pop[bi]), fits[bi]

    return x_best, f_best


# ---------------------------------------------------------------------------
# 3. Differential Evolution
# ---------------------------------------------------------------------------

def differential_evolution(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: Optional[int] = None,
    max_gens: int = 200,
    F: float = 0.8,
    CR: float = 0.9,
    strategy: str = 'rand/1',
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Differential Evolution.

    Strategies: 'rand/1', 'best/1', 'current-to-best/1'.

    Returns
    -------
    (x_best, f_best)
    """
    rng = random.Random(seed)
    ndim = len(bounds)
    if pop_size is None:
        pop_size = 10 * ndim

    valid_strategies = ('rand/1', 'best/1', 'current-to-best/1')
    if strategy not in valid_strategies:
        raise ValueError(f"strategy must be one of {valid_strategies}")

    def rand_ind() -> List[float]:
        return [rng.uniform(lo, hi) for lo, hi in bounds]

    pop = [rand_ind() for _ in range(pop_size)]
    fits = [f(ind) for ind in pop]
    best_idx = min(range(pop_size), key=lambda i: fits[i])

    for _ in range(max_gens):
        best_idx = min(range(pop_size), key=lambda i: fits[i])
        x_best_vec = pop[best_idx]
        for i in range(pop_size):
            # Select distinct random indices ≠ i
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = rng.sample(candidates, 3)

            if strategy == 'rand/1':
                v = _vec_add(pop[r1], _vec_scale(F, _vec_sub(pop[r2], pop[r3])))
            elif strategy == 'best/1':
                v = _vec_add(x_best_vec, _vec_scale(F, _vec_sub(pop[r1], pop[r2])))
            else:  # current-to-best/1
                v = _vec_add(
                    _vec_add(pop[i], _vec_scale(F, _vec_sub(x_best_vec, pop[i]))),
                    _vec_scale(F, _vec_sub(pop[r1], pop[r2])),
                )

            # Clip to bounds
            v = _clip_vec(v, bounds)

            # Binomial crossover
            j_rand = rng.randrange(ndim)
            u = [
                v[j] if rng.random() < CR or j == j_rand else pop[i][j]
                for j in range(ndim)
            ]

            f_u = f(u)
            if f_u <= fits[i]:
                pop[i] = u
                fits[i] = f_u

    best_idx = min(range(pop_size), key=lambda i: fits[i])
    return list(pop[best_idx]), fits[best_idx]


# ---------------------------------------------------------------------------
# 4. Particle Swarm Optimisation
# ---------------------------------------------------------------------------

def particle_swarm(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    max_iter: int = 200,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Standard Particle Swarm Optimisation.

    v_{k+1} = w*v_k + c1*r1*(pbest-x) + c2*r2*(gbest-x)

    Returns
    -------
    (x_best, f_best)
    """
    rng = random.Random(seed)
    ndim = len(bounds)

    # Initialise positions and velocities
    pos = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(n_particles)]
    v_max = [(hi - lo) * 0.1 for lo, hi in bounds]
    vel = [
        [rng.uniform(-v_max[j], v_max[j]) for j in range(ndim)]
        for _ in range(n_particles)
    ]

    fits = [f(pos[i]) for i in range(n_particles)]
    pbest_pos = [list(pos[i]) for i in range(n_particles)]
    pbest_fit = list(fits)

    gbest_idx = min(range(n_particles), key=lambda i: pbest_fit[i])
    gbest_pos = list(pbest_pos[gbest_idx])
    gbest_fit = pbest_fit[gbest_idx]

    for _ in range(max_iter):
        for i in range(n_particles):
            r1 = [rng.random() for _ in range(ndim)]
            r2 = [rng.random() for _ in range(ndim)]
            for j in range(ndim):
                vel[i][j] = (
                    w * vel[i][j]
                    + c1 * r1[j] * (pbest_pos[i][j] - pos[i][j])
                    + c2 * r2[j] * (gbest_pos[j] - pos[i][j])
                )
                # Clamp velocity
                vel[i][j] = _clip(vel[i][j], -v_max[j], v_max[j])
                pos[i][j] = _clip(pos[i][j] + vel[i][j], bounds[j][0], bounds[j][1])

            fi = f(pos[i])
            if fi < pbest_fit[i]:
                pbest_pos[i] = list(pos[i])
                pbest_fit[i] = fi
                if fi < gbest_fit:
                    gbest_pos = list(pos[i])
                    gbest_fit = fi

    return gbest_pos, gbest_fit


# ---------------------------------------------------------------------------
# 5. Nelder-Mead
# ---------------------------------------------------------------------------

def nelder_mead(
    f: Callable[[List[float]], float],
    x0: List[float],
    step: float = 0.1,
    tol: float = 1e-6,
    max_iter: int = 1000,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = 0.5,
    sigma: float = 0.5,
) -> Tuple[List[float], float, int, bool]:
    """
    Full Nelder-Mead simplex method.

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    n = len(x0)

    # 1. Build initial simplex
    simplex = [list(x0)]
    for i in range(n):
        v = list(x0)
        v[i] += step
        simplex.append(v)

    fvals = [f(v) for v in simplex]
    n_iters = 0
    converged = False

    for n_iters in range(1, max_iter + 1):
        # 2. Order
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]

        # Convergence check: std of f values
        f_mean = sum(fvals) / (n + 1)
        f_std = math.sqrt(sum((fv - f_mean) ** 2 for fv in fvals) / (n + 1))
        if f_std < tol:
            converged = True
            break

        # 3. Centroid (exclude worst)
        x_bar = [
            sum(simplex[i][j] for i in range(n)) / n for j in range(n)
        ]

        x_worst = simplex[n]
        f_worst = fvals[n]
        x_second_worst_f = fvals[n - 1]
        f_best = fvals[0]

        # 4. Reflection
        x_r = _vec_add(x_bar, _vec_scale(alpha, _vec_sub(x_bar, x_worst)))
        f_r = f(x_r)

        if f_best <= f_r < x_second_worst_f:
            simplex[n] = x_r
            fvals[n] = f_r
        elif f_r < f_best:
            # Expansion
            x_e = _vec_add(x_bar, _vec_scale(gamma, _vec_sub(x_r, x_bar)))
            f_e = f(x_e)
            if f_e < f_r:
                simplex[n] = x_e
                fvals[n] = f_e
            else:
                simplex[n] = x_r
                fvals[n] = f_r
        else:
            # Contraction
            if f_r < f_worst:
                # Outside contraction
                x_c = _vec_add(x_bar, _vec_scale(rho, _vec_sub(x_r, x_bar)))
                f_c = f(x_c)
                if f_c < f_r:
                    simplex[n] = x_c
                    fvals[n] = f_c
                else:
                    # Shrink
                    x_best_v = simplex[0]
                    for i in range(1, n + 1):
                        simplex[i] = _vec_add(
                            x_best_v,
                            _vec_scale(sigma, _vec_sub(simplex[i], x_best_v)),
                        )
                        fvals[i] = f(simplex[i])
            else:
                # Inside contraction
                x_c = _vec_add(x_bar, _vec_scale(rho, _vec_sub(x_worst, x_bar)))
                f_c = f(x_c)
                if f_c < f_worst:
                    simplex[n] = x_c
                    fvals[n] = f_c
                else:
                    # Shrink
                    x_best_v = simplex[0]
                    for i in range(1, n + 1):
                        simplex[i] = _vec_add(
                            x_best_v,
                            _vec_scale(sigma, _vec_sub(simplex[i], x_best_v)),
                        )
                        fvals[i] = f(simplex[i])

    # Re-order to return best
    order = sorted(range(n + 1), key=lambda i: fvals[i])
    return list(simplex[order[0]]), fvals[order[0]], n_iters, converged


# ---------------------------------------------------------------------------
# 6. CMA-ES
# ---------------------------------------------------------------------------

def cma_es(
    f: Callable[[List[float]], float],
    x0: List[float],
    sigma0: float = 0.3,
    max_iter: int = 1000,
    tol: float = 1e-6,
    seed: Optional[int] = None,
) -> Tuple[List[float], float, int, bool]:
    """
    CMA-ES with Cholesky-based sampling (Hansen, 2006).

    Returns
    -------
    (x_opt, f_opt, n_iters, converged)
    """
    rng = random.Random(seed)
    n = len(x0)

    # Strategy parameters
    lam = 4 + int(3 * math.log(n))
    mu = lam // 2

    # Recombination weights
    raw_w = [math.log((lam + 1) / 2.0) - math.log(i + 1) for i in range(mu)]
    w_sum = sum(raw_w)
    weights = [wi / w_sum for wi in raw_w]
    mu_eff = 1.0 / sum(wi ** 2 for wi in weights)

    # Adaptation parameters
    c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
    d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
    c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
    c_1 = 2.0 / ((n + 1.3) ** 2 + mu_eff)
    c_mu = min(
        1.0 - c_1,
        2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff),
    )
    chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

    # State
    m = list(x0)
    sigma = sigma0
    p_sigma = [0.0] * n
    p_c = [0.0] * n
    # Covariance C = I initially; maintain as full matrix
    C = _identity(n)

    x_best = list(m)
    f_best = f(m)
    converged = False
    n_iters = 0

    def _safe_cholesky(C_mat: List[List[float]]) -> List[List[float]]:
        """Return Cholesky factor, falling back to sqrt(diag) if not PD."""
        L = _cholesky(C_mat)
        if L is not None:
            return L
        # Fallback: diagonal sqrt
        warnings.warn("CMA-ES: Covariance not positive-definite; resetting to identity.")
        return _identity(n)

    for n_iters in range(1, max_iter + 1):
        L = _safe_cholesky(C)

        # Sample lam offspring
        zs: List[List[float]] = []
        ys: List[List[float]] = []
        xs: List[List[float]] = []
        for _ in range(lam):
            z = _randn_vec(n, rng)
            y = _mat_vec(L, z)
            x = _vec_add(m, _vec_scale(sigma, y))
            zs.append(z)
            ys.append(y)
            xs.append(x)

        fvals = [f(xi) for xi in xs]

        # Sort by fitness
        order = sorted(range(lam), key=lambda i: fvals[i])
        f_min = fvals[order[0]]
        if f_min < f_best:
            f_best = f_min
            x_best = list(xs[order[0]])

        # Update mean
        m_old = list(m)
        m = [
            sum(weights[k] * xs[order[k]][j] for k in range(mu))
            for j in range(n)
        ]

        # Step-size path p_sigma
        # C^{-1/2} * (m - m_old) / sigma  ≈ weighted sum of z's
        y_w = [sum(weights[k] * ys[order[k]][j] for k in range(mu)) for j in range(n)]
        # invsqrt(C) * y_w — for Cholesky approach: solve L @ v = y_w via forward sub
        # z_w = L^{-1} y_w
        z_w = [0.0] * n
        for i in range(n):
            s = y_w[i] - sum(L[i][j] * z_w[j] for j in range(i))
            z_w[i] = s / L[i][i]

        p_sigma = [
            (1.0 - c_sigma) * p_sigma[j] + math.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * z_w[j]
            for j in range(n)
        ]

        # Sigma update
        p_sigma_norm = math.sqrt(sum(x ** 2 for x in p_sigma))
        sigma = sigma * math.exp((c_sigma / d_sigma) * (p_sigma_norm / chi_n - 1.0))

        # Anisotropic evolution path p_c
        h_sigma = 1.0 if (
            p_sigma_norm / math.sqrt(1.0 - (1.0 - c_sigma) ** (2.0 * (n_iters + 1)))
            < (1.4 + 2.0 / (n + 1.0)) * chi_n
        ) else 0.0

        p_c = [
            (1.0 - c_c) * p_c[j] + h_sigma * math.sqrt(c_c * (2.0 - c_c) * mu_eff) * y_w[j]
            for j in range(n)
        ]

        # Covariance update
        delta_h = (1.0 - h_sigma) * c_c * (2.0 - c_c)

        # Rank-1 update term
        pc_outer = _outer(p_c, p_c)

        # Rank-mu update term
        rank_mu = [[0.0] * n for _ in range(n)]
        for k in range(mu):
            yo = ys[order[k]]
            oo = _outer(yo, yo)
            for i in range(n):
                for j in range(n):
                    rank_mu[i][j] += weights[k] * oo[i][j]

        for i in range(n):
            for j in range(n):
                C[i][j] = (
                    (1.0 - c_1 - c_mu) * C[i][j]
                    + c_1 * (pc_outer[i][j] + delta_h * C[i][j])
                    + c_mu * rank_mu[i][j]
                )

        # Enforce symmetry
        for i in range(n):
            for j in range(i):
                C[i][j] = C[j][i] = 0.5 * (C[i][j] + C[j][i])

        # Convergence: sigma below tolerance
        if sigma < tol:
            converged = True
            break

    return x_best, f_best, n_iters, converged


# ---------------------------------------------------------------------------
# 7. Basin Hopping
# ---------------------------------------------------------------------------

def _finite_diff_gradient(
    f: Callable[[List[float]], float],
    x: List[float],
    h: float = 1e-5,
) -> List[float]:
    """Central-difference gradient approximation."""
    n = len(x)
    grad = []
    for i in range(n):
        xp = list(x)
        xm = list(x)
        xp[i] += h
        xm[i] -= h
        grad.append((f(xp) - f(xm)) / (2.0 * h))
    return grad


def _simple_local_opt(
    f: Callable[[List[float]], float],
    x: List[float],
    n_steps: int = 100,
    lr: float = 0.01,
) -> List[float]:
    """Simple gradient descent with finite differences."""
    x = list(x)
    for _ in range(n_steps):
        g = _finite_diff_gradient(f, x)
        g_norm = math.sqrt(sum(gi ** 2 for gi in g)) + 1e-300
        x = [xi - lr * gi for xi, gi in zip(x, g)]
    return x


def basin_hopping(
    f: Callable[[List[float]], float],
    x0: List[float],
    n_hops: int = 100,
    T: float = 1.0,
    step_size: float = 0.5,
    local_optimizer: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> Tuple[List[float], float, int]:
    """
    Basin hopping: random perturbation + local optimisation + Metropolis accept.

    Parameters
    ----------
    f               : Objective function (minimisation).
    x0              : Initial point.
    n_hops          : Number of basin-hopping steps.
    T               : Temperature for Metropolis criterion.
    step_size       : Standard deviation of the Normal perturbation.
    local_optimizer : Callable(f, x) -> x_local_opt. Defaults to simple GD.
    seed            : Random seed.

    Returns
    -------
    (x_best, f_best, n_hops_completed)
    """
    rng = random.Random(seed)
    n = len(x0)

    if local_optimizer is None:
        local_optimizer = lambda func, x: _simple_local_opt(func, x)

    x_cur = local_optimizer(f, list(x0))
    f_cur = f(x_cur)
    x_best = list(x_cur)
    f_best = f_cur

    hops_done = 0
    for hop in range(n_hops):
        # Perturb
        x_new = [x_cur[i] + step_size * _box_muller(rng) for i in range(n)]
        # Local opt
        x_new = local_optimizer(f, x_new)
        f_new = f(x_new)
        hops_done = hop + 1

        # Metropolis
        delta = f_new - f_cur
        if delta < 0.0 or (T > 0.0 and rng.random() < math.exp(-delta / T)):
            x_cur = x_new
            f_cur = f_new

        if f_new < f_best:
            x_best = list(x_new)
            f_best = f_new

    return x_best, f_best, hops_done


# ---------------------------------------------------------------------------
# 8. Random Search
# ---------------------------------------------------------------------------

def random_search(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Pure random search: sample uniformly from bounds, track best.

    Returns
    -------
    (x_best, f_best)
    """
    rng = random.Random(seed)
    ndim = len(bounds)

    x_best: Optional[List[float]] = None
    f_best = math.inf

    for _ in range(n_samples):
        x = [rng.uniform(lo, hi) for lo, hi in bounds]
        fv = f(x)
        if fv < f_best:
            f_best = fv
            x_best = list(x)

    if x_best is None:
        x_best = [lo for lo, _ in bounds]

    return x_best, f_best


# ---------------------------------------------------------------------------
# 9. Latin Hypercube Search
# ---------------------------------------------------------------------------

def latin_hypercube_search(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> Tuple[List[float], float]:
    """
    Latin Hypercube Sampling followed by function evaluation.

    Each dimension is stratified into n_samples equal strata.
    One sample per stratum (permuted independently per dimension).

    Returns
    -------
    (x_best, f_best)
    """
    rng = random.Random(seed)
    ndim = len(bounds)

    # Generate LHS design: for each dimension, create a permutation of [0..n-1]
    design: List[List[float]] = []
    for d in range(ndim):
        perm = list(range(n_samples))
        rng.shuffle(perm)
        lo, hi = bounds[d]
        col = [(perm[i] + rng.random()) / n_samples * (hi - lo) + lo for i in range(n_samples)]
        design.append(col)

    x_best: Optional[List[float]] = None
    f_best = math.inf

    for i in range(n_samples):
        x = [design[d][i] for d in range(ndim)]
        fv = f(x)
        if fv < f_best:
            f_best = fv
            x_best = list(x)

    if x_best is None:
        x_best = [lo for lo, _ in bounds]

    return x_best, f_best

"""
Multi-Objective Optimization
=============================
Pareto dominance, NSGA-II, weighted-sum scalarization.

Algorithms implemented
----------------------
* Pareto dominance check
* Pareto-front extraction
* Crowding-distance computation (NSGA-II diversity metric)
* Fast non-dominated sorting (NSGA-II ranking)
* NSGA-II multi-objective evolutionary algorithm (Deb et al., 2002)
* Weighted-sum scalarization
* Uniform simplex weight-vector generation

All implementations are pure Python (stdlib only: math, random, collections,
typing).  No third-party dependencies are required.

References
----------
* Deb, Pratap, Agarwal & Meyarivan (2002) — "A fast and elitist multiobjective
  genetic algorithm: NSGA-II." IEEE TEVC 6(2):182-197.
"""

import math
import random
from typing import Callable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 10.1  Pareto Utilities
# ---------------------------------------------------------------------------


def dominates(a: List[float], b: List[float]) -> bool:
    """Return True if objective vector *a* Pareto-dominates *b*.

    Assumes **minimization** of all objectives.

    *a* dominates *b* iff:

    * ``a[i] <= b[i]`` for every objective *i*  (a is no worse in any objective), **and**
    * ``a[i] < b[i]``  for at least one *i*     (a is strictly better somewhere).

    Args:
        a: Objective vector for solution A.
        b: Objective vector for solution B (same length as *a*).

    Returns:
        ``True`` if *a* dominates *b*, ``False`` otherwise.

    Examples::

        >>> dominates([1.0, 2.0], [3.0, 4.0])
        True
        >>> dominates([1.0, 2.0], [1.0, 2.0])
        False
        >>> dominates([1.0, 5.0], [3.0, 2.0])
        False
    """
    at_least_one_strictly_better = False
    for ai, bi in zip(a, b):
        if ai > bi:
            return False  # a is worse in this objective — cannot dominate
        if ai < bi:
            at_least_one_strictly_better = True
    return at_least_one_strictly_better


def pareto_front(
    solutions: List[List[float]],
    objectives: List[List[float]],
) -> Tuple[List[List[float]], List[List[float]]]:
    """Compute the Pareto front (non-dominated set).

    A solution *i* is non-dominated if no other solution *j* satisfies
    ``dominates(objectives[j], objectives[i])``.

    Args:
        solutions:  List of decision-variable vectors.
        objectives: List of objective vectors corresponding to each solution
                    (must be the same length as *solutions*).

    Returns:
        A tuple ``(front_solutions, front_objectives)`` containing only the
        non-dominated solutions and their objective vectors, in the original
        order.

    Raises:
        ValueError: If *solutions* and *objectives* have different lengths.
    """
    if len(solutions) != len(objectives):
        raise ValueError(
            f"solutions and objectives must have the same length, "
            f"got {len(solutions)} and {len(objectives)}"
        )

    n = len(solutions)
    non_dominated: List[bool] = [True] * n

    for i in range(n):
        if not non_dominated[i]:
            continue
        for j in range(n):
            if i == j or not non_dominated[j]:
                continue
            if dominates(objectives[j], objectives[i]):
                non_dominated[i] = False
                break  # i is dominated; no need to check further

    front_solutions: List[List[float]] = []
    front_objectives: List[List[float]] = []
    for i in range(n):
        if non_dominated[i]:
            front_solutions.append(solutions[i])
            front_objectives.append(objectives[i])

    return front_solutions, front_objectives


def crowding_distance(
    front_objectives: List[List[float]],
) -> List[float]:
    """Compute crowding distance for NSGA-II diversity preservation.

    Boundary solutions (the extreme points for any objective) receive an
    infinite crowding distance.  Interior solutions accumulate distance
    contributions from each objective::

        distance[i] += (obj[i+1][m] - obj[i-1][m]) / (max_m - min_m)

    where the indexing refers to the position of solution *i* in the list
    sorted by objective *m*.  If ``max_m == min_m`` the contribution for that
    objective is 0.

    Args:
        front_objectives: List of objective vectors for solutions on a single
                          Pareto front.  May be in any order.

    Returns:
        List of crowding distances in the **same order** as
        *front_objectives*.  Boundary points get ``float('inf')``.
        A single-point front returns ``[float('inf')]``.

    Note:
        The function does not modify *front_objectives* in place.
    """
    n = len(front_objectives)
    if n == 0:
        return []
    if n == 1:
        return [float("inf")]

    distances: List[float] = [0.0] * n
    n_objectives = len(front_objectives[0])

    for m in range(n_objectives):
        # Sort indices by objective m
        sorted_indices = sorted(range(n), key=lambda i: front_objectives[i][m])

        obj_min = front_objectives[sorted_indices[0]][m]
        obj_max = front_objectives[sorted_indices[-1]][m]

        # Boundary points always get infinite distance
        distances[sorted_indices[0]] = float("inf")
        distances[sorted_indices[-1]] = float("inf")

        obj_range = obj_max - obj_min
        if obj_range == 0.0:
            # All solutions have the same value for this objective
            continue

        for k in range(1, n - 1):
            prev_obj = front_objectives[sorted_indices[k - 1]][m]
            next_obj = front_objectives[sorted_indices[k + 1]][m]
            # Only accumulate if this point has not already been made infinite
            if distances[sorted_indices[k]] != float("inf"):
                distances[sorted_indices[k]] += (next_obj - prev_obj) / obj_range

    return distances


def fast_non_dominated_sort(
    objectives: List[List[float]],
) -> List[List[int]]:
    """NSGA-II fast non-dominated sorting.

    Partitions solution indices into a sequence of fronts.  Front 0 is the
    Pareto front; front 1 contains solutions that are non-dominated once the
    Pareto front is removed, and so on.

    Algorithm (Deb et al., 2002, Section III-A)::

        For each i:
            n_i  = number of solutions that dominate i
            S_i  = set of solutions that i dominates
        F_0 = {i : n_i == 0}
        Repeat:
            Q = {}
            For each i in current front:
                For each j in S_i:
                    n_j -= 1
                    If n_j == 0: add j to Q
            next front = Q

    Args:
        objectives: List of objective vectors for all solutions.

    Returns:
        A list of fronts.  Each front is a list of **indices** into
        *objectives*, ordered as encountered (not sorted).

    Raises:
        ValueError: If *objectives* is empty.
    """
    if not objectives:
        return []

    n = len(objectives)
    # domination_count[i] = number of solutions that dominate i
    domination_count: List[int] = [0] * n
    # dominated_set[i]   = list of solutions that i dominates
    dominated_set: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                domination_count[i] += 1

    fronts: List[List[int]] = []
    current_front: List[int] = [i for i in range(n) if domination_count[i] == 0]

    while current_front:
        fronts.append(current_front)
        next_front: List[int] = []
        for i in current_front:
            for j in dominated_set[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front = next_front

    return fronts


# ---------------------------------------------------------------------------
# 10.2  NSGA-II
# ---------------------------------------------------------------------------


def nsga2(
    objectives: List[Callable[[List[float]], float]],
    bounds: List[Tuple[float, float]],
    pop_size: int = 100,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.9,
    seed: Optional[int] = None,
) -> Tuple[List[List[float]], List[List[float]]]:
    """NSGA-II multi-objective evolutionary algorithm (Deb et al., 2002).

    Minimizes each objective in *objectives* simultaneously over the search
    space defined by *bounds*.

    Algorithm overview::

        1. Initialise population P (size pop_size) randomly inside bounds.
        2. Evaluate all objective functions for every individual.
        3. For each generation:
            a. Create offspring Q via binary tournament selection,
               single-point crossover, and uniform mutation.
            b. Form combined pool R = P ∪ Q (size 2 * pop_size).
            c. Apply fast non-dominated sort to R → fronts F_0, F_1, ...
            d. Compute crowding distances within each front.
            e. Fill next generation P by greedily adding complete fronts;
               the last (critical) front is truncated using crowding distance.
        4. Return the Pareto front of the final population.

    Selection: Binary tournament — two individuals are drawn at random and
    compared by ``(rank ASC, crowding_distance DESC)``.

    Crossover: Single-point — the chromosome is split at a uniformly random
    index, and genes are swapped between the two parents when
    ``random() < crossover_rate``.

    Mutation: Uniform per-gene — each gene is re-drawn uniformly within its
    bounds with probability *mutation_rate*.

    Args:
        objectives:     List of objective functions; each takes a
                        ``List[float]`` of decision variables and returns a
                        ``float``.
        bounds:         ``[(lower, upper)]`` per dimension.
        pop_size:       Population size.  Automatically rounded up to the
                        nearest even number.
        generations:    Number of evolutionary generations.
        mutation_rate:  Per-gene probability of mutation.
        crossover_rate: Probability that a crossover is applied to a parent
                        pair (otherwise offspring is a copy of the first
                        parent).
        seed:           Optional integer seed for ``random`` to allow
                        reproducible runs.

    Returns:
        ``(pareto_solutions, pareto_objectives)`` — the Pareto-front
        solutions and their objective vectors from the final population.

    Raises:
        ValueError: If *bounds* is empty or any bound has ``lower > upper``.
    """
    if seed is not None:
        random.seed(seed)

    if not bounds:
        raise ValueError("bounds must not be empty")
    for lo, hi in bounds:
        if lo > hi:
            raise ValueError(f"Invalid bound: lower={lo} > upper={hi}")

    # Ensure pop_size is even (required for pairing in crossover)
    if pop_size % 2 != 0:
        pop_size += 1

    n_dims = len(bounds)
    n_obj = len(objectives)

    # ------------------------------------------------------------------
    # Helper: random individual
    # ------------------------------------------------------------------
    def _random_individual() -> List[float]:
        return [random.uniform(lo, hi) for lo, hi in bounds]

    # ------------------------------------------------------------------
    # Helper: evaluate objectives for one individual
    # ------------------------------------------------------------------
    def _evaluate(x: List[float]) -> List[float]:
        return [f(x) for f in objectives]

    # ------------------------------------------------------------------
    # Helper: single-point crossover
    # ------------------------------------------------------------------
    def _crossover(p1: List[float], p2: List[float]) -> Tuple[List[float], List[float]]:
        if random.random() >= crossover_rate or n_dims == 1:
            # No crossover — offspring are copies of parents
            return list(p1), list(p2)
        point = random.randint(1, n_dims - 1)
        c1 = p1[:point] + p2[point:]
        c2 = p2[:point] + p1[point:]
        return c1, c2

    # ------------------------------------------------------------------
    # Helper: uniform mutation
    # ------------------------------------------------------------------
    def _mutate(x: List[float]) -> List[float]:
        return [
            random.uniform(lo, hi) if random.random() < mutation_rate else xi
            for xi, (lo, hi) in zip(x, bounds)
        ]

    # ------------------------------------------------------------------
    # Helper: tournament selection
    # Selects one individual from the population.
    # rank_map[i] = rank (front index) for individual i in the pool.
    # cd_map[i]   = crowding distance for individual i in the pool.
    # ------------------------------------------------------------------
    def _tournament(
        pool_size: int,
        rank_map: List[int],
        cd_map: List[float],
    ) -> int:
        a = random.randrange(pool_size)
        b = random.randrange(pool_size)
        # Lower rank is better; higher crowding distance is better
        if rank_map[a] < rank_map[b]:
            return a
        if rank_map[b] < rank_map[a]:
            return b
        # Same rank: prefer higher crowding distance (more diversity)
        if cd_map[a] >= cd_map[b]:
            return a
        return b

    # ------------------------------------------------------------------
    # Initialise population
    # ------------------------------------------------------------------
    population: List[List[float]] = [_random_individual() for _ in range(pop_size)]
    pop_obj: List[List[float]] = [_evaluate(x) for x in population]

    # ------------------------------------------------------------------
    # Main evolutionary loop
    # ------------------------------------------------------------------
    for _gen in range(generations):
        # --- Rank and crowding distance for current population -----------
        fronts = fast_non_dominated_sort(pop_obj)

        # Build per-individual rank and crowding-distance maps
        rank_map: List[int] = [0] * pop_size
        cd_map: List[float] = [0.0] * pop_size

        for rank, front in enumerate(fronts):
            front_objs = [pop_obj[i] for i in front]
            dists = crowding_distance(front_objs)
            for local_idx, global_idx in enumerate(front):
                rank_map[global_idx] = rank
                cd_map[global_idx] = dists[local_idx]

        # --- Generate offspring via selection, crossover, mutation -------
        offspring: List[List[float]] = []
        offspring_obj: List[List[float]] = []

        while len(offspring) < pop_size:
            p1_idx = _tournament(pop_size, rank_map, cd_map)
            p2_idx = _tournament(pop_size, rank_map, cd_map)

            c1, c2 = _crossover(population[p1_idx], population[p2_idx])
            c1 = _mutate(c1)
            c2 = _mutate(c2)

            offspring.append(c1)
            offspring_obj.append(_evaluate(c1))
            if len(offspring) < pop_size:
                offspring.append(c2)
                offspring_obj.append(_evaluate(c2))

        # --- Combine parent + offspring pools ----------------------------
        combined = population + offspring
        combined_obj = pop_obj + offspring_obj
        combined_size = len(combined)  # 2 * pop_size

        # --- Fast non-dominated sort on combined pool --------------------
        combined_fronts = fast_non_dominated_sort(combined_obj)

        # --- Select next generation --------------------------------------
        new_pop: List[List[float]] = []
        new_obj: List[List[float]] = []

        for front in combined_fronts:
            if len(new_pop) + len(front) <= pop_size:
                # Entire front fits — add all
                for idx in front:
                    new_pop.append(combined[idx])
                    new_obj.append(combined_obj[idx])
            else:
                # Partial front — rank by crowding distance (descending)
                front_objs = [combined_obj[i] for i in front]
                dists = crowding_distance(front_objs)
                # Sort by crowding distance descending
                sorted_pairs = sorted(
                    zip(front, dists), key=lambda t: t[1], reverse=True
                )
                slots_left = pop_size - len(new_pop)
                for idx, _dist in sorted_pairs[:slots_left]:
                    new_pop.append(combined[idx])
                    new_obj.append(combined_obj[idx])
                break  # Population full

        population = new_pop
        pop_obj = new_obj

    # ------------------------------------------------------------------
    # Return Pareto front of final population
    # ------------------------------------------------------------------
    return pareto_front(population, pop_obj)


# ---------------------------------------------------------------------------
# 10.3  Weighted-Sum Scalarization
# ---------------------------------------------------------------------------


def weighted_sum(
    objectives: List[Callable[[List[float]], float]],
    weights: List[float],
) -> Callable[[List[float]], float]:
    """Create a scalarized single objective from multiple objectives.

    The returned function computes::

        f(x) = sum_i  weights[i] * objectives[i](x)

    Args:
        objectives: List of objective functions, each ``(List[float]) -> float``.
        weights:    Non-negative weight for each objective.  Weights do not
                    need to sum to 1.

    Returns:
        A single callable ``f(x) -> float``.

    Raises:
        ValueError: If *objectives* and *weights* have different lengths.

    Examples::

        >>> def f1(x): return x[0]
        >>> def f2(x): return x[1]
        >>> scal = weighted_sum([f1, f2], [0.5, 0.5])
        >>> scal([2.0, 4.0])
        3.0
    """
    if len(objectives) != len(weights):
        raise ValueError(
            f"objectives and weights must have the same length, "
            f"got {len(objectives)} and {len(weights)}"
        )

    # Capture by value at definition time
    _objectives = list(objectives)
    _weights = list(weights)

    def _scalarized(x: List[float]) -> float:
        return sum(w * f(x) for w, f in zip(_weights, _objectives))

    return _scalarized


def generate_weight_vectors(
    n_objectives: int,
    n_vectors: int,
    seed: Optional[int] = None,
) -> List[List[float]]:
    """Generate *n_vectors* weight vectors uniformly on the unit simplex.

    Sampling method (Dirichlet-like via the exponential trick):

    1. For each vector draw *n_objectives* independent samples
       ``e_i ~ Exp(1)``, i.e., ``e_i = -log(U)`` where ``U ~ Uniform(0, 1)``.
    2. Normalise: ``w_i = e_i / sum(e_j)``.

    This produces a distribution that is uniform on the
    ``(n_objectives - 1)``-simplex.

    Args:
        n_objectives: Dimensionality of the weight vectors (number of
                      objectives).
        n_vectors:    Number of weight vectors to generate.
        seed:         Optional integer seed for reproducibility.

    Returns:
        A list of *n_vectors* weight vectors; each is a ``List[float]`` of
        length *n_objectives* whose entries are non-negative and sum to 1.

    Raises:
        ValueError: If *n_objectives* < 1 or *n_vectors* < 1.
    """
    if n_objectives < 1:
        raise ValueError(f"n_objectives must be >= 1, got {n_objectives}")
    if n_vectors < 1:
        raise ValueError(f"n_vectors must be >= 1, got {n_vectors}")

    if seed is not None:
        random.seed(seed)

    vectors: List[List[float]] = []
    for _ in range(n_vectors):
        # Draw Exp(1) samples: e_i = -log(U), guard against log(0)
        exps: List[float] = [-math.log(random.random()) for _ in range(n_objectives)]
        total = sum(exps)
        vectors.append([e / total for e in exps])

    return vectors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "dominates",
    "pareto_front",
    "crowding_distance",
    "fast_non_dominated_sort",
    "nsga2",
    "weighted_sum",
    "generate_weight_vectors",
]

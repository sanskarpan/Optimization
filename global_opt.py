"""
Global Optimization Methods
============================

Simulated annealing, genetic algorithms, particle swarm optimization.

For finding global optima in non-convex problems.
"""

import random
import math
from typing import List, Callable, Tuple, Optional


def simulated_annealing(
    f: Callable[[List[float]], float],
    x0: List[float],
    bounds: List[Tuple[float, float]],
    T_init: float = 100.0,
    T_min: float = 0.001,
    alpha: float = 0.95,
    max_iter: int = 10000
) -> Tuple[List[float], float]:
    """
    Simulated Annealing for global optimization.

    Probabilistically accepts worse solutions to escape local minima.

    Args:
        f: Objective function to minimize
        x0: Initial solution
        bounds: List of (min, max) for each dimension
        T_init: Initial temperature
        T_min: Minimum temperature (stopping criterion)
        alpha: Cooling rate
        max_iter: Maximum iterations

    Returns:
        (best_solution, best_value)

    Example:
        >>> f = lambda x: sum(xi**2 for xi in x)
        >>> x0 = [5.0, 5.0]
        >>> bounds = [(-10, 10), (-10, 10)]
        >>> x_opt, f_opt = simulated_annealing(f, x0, bounds)
    """
    x_current = x0[:]
    f_current = f(x_current)

    x_best = x_current[:]
    f_best = f_current

    T = T_init
    iteration = 0

    while T > T_min and iteration < max_iter:
        # Generate neighbour solution.
        # Scale perturbation with temperature so exploration is wide at high T
        # and narrows as the algorithm cools (proportional SA step size).
        x_new = []
        for i, (lower, upper) in enumerate(bounds):
            scale = (upper - lower) * math.sqrt(T / T_init)
            step = scale * random.gauss(0, 1)
            x_i = x_current[i] + step
            # Ensure within bounds
            x_i = max(lower, min(upper, x_i))
            x_new.append(x_i)

        f_new = f(x_new)

        # Accept or reject
        delta_f = f_new - f_current

        if delta_f < 0:
            # Always accept better solutions
            x_current = x_new
            f_current = f_new

            if f_current < f_best:
                x_best = x_current[:]
                f_best = f_current
        else:
            # Accept worse solutions with probability exp(-ΔE/T)
            acceptance_prob = math.exp(-delta_f / T)
            if random.random() < acceptance_prob:
                x_current = x_new
                f_current = f_new

        # Cool down
        T *= alpha
        iteration += 1

    return x_best, f_best


def genetic_algorithm(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    generations: int = 100,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8
) -> Tuple[List[float], float]:
    """
    Genetic Algorithm for optimization.

    Mimics natural selection: selection, crossover, mutation.

    Args:
        f: Objective function to minimize
        bounds: List of (min, max) for each dimension
        pop_size: Population size
        generations: Number of generations
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover

    Returns:
        (best_solution, best_value)

    Example:
        >>> f = lambda x: sum((xi - 5)**2 for xi in x)
        >>> bounds = [(0, 10), (0, 10)]
        >>> x_opt, f_opt = genetic_algorithm(f, bounds, generations=100)
    """
    if pop_size < 3:
        raise ValueError(
            f"genetic_algorithm: pop_size must be >= 3 for tournament "
            f"selection of size 3 (got {pop_size})"
        )

    n_dim = len(bounds)

    # Initialize population
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(lower, upper) for lower, upper in bounds]
        population.append(individual)

    best_individual = None
    best_fitness = float('inf')

    for generation in range(generations):
        # Evaluate fitness
        fitness = [f(ind) for ind in population]

        # Track best
        min_fitness = min(fitness)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_individual = population[fitness.index(min_fitness)][:]

        # Selection (tournament selection)
        selected = []
        for _ in range(pop_size):
            # Tournament of size 3
            tournament = random.sample(list(range(pop_size)), 3)
            winner = min(tournament, key=lambda i: fitness[i])
            selected.append(population[winner][:])

        # Crossover and mutation — track offspring fitness inline to avoid
        # a costly second full pass over the population for elitism.
        next_population = []
        next_fitness_list = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < pop_size else selected[0]

            if random.random() < crossover_rate and n_dim > 1:
                # Single-point crossover (requires at least 2 dimensions)
                point = random.randint(1, n_dim - 1)
                child1 = parent1[:point] + parent2[point:]
                child2 = parent2[:point] + parent1[point:]
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            for child in [child1, child2]:
                for j in range(n_dim):
                    if random.random() < mutation_rate:
                        lower, upper = bounds[j]
                        child[j] = random.uniform(lower, upper)
                next_population.append(child)
                next_fitness_list.append(f(child))   # compute once, reuse for elitism

        population = next_population[:pop_size]
        next_fitness_list = next_fitness_list[:pop_size]

        # Elitism: guarantee the all-time best individual survives by replacing
        # the worst member of the new population (no extra f evaluations needed).
        if best_individual is not None:
            worst_idx = max(range(len(population)), key=lambda i: next_fitness_list[i])
            if next_fitness_list[worst_idx] > best_fitness:
                population[worst_idx] = best_individual[:]

    return best_individual, best_fitness


def particle_swarm_optimization(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    n_particles: int = 30,
    max_iter: int = 100,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    v_max: Optional[float] = None
) -> Tuple[List[float], float]:
    """
    Particle Swarm Optimization.

    Particles move in search space influenced by personal best
    and global best positions.

    Args:
        f: Objective function to minimize
        bounds: List of (min, max) for each dimension
        n_particles: Number of particles
        max_iter: Maximum iterations
        w: Inertia weight
        c1: Cognitive parameter (personal best weight)
        c2: Social parameter (global best weight)
        v_max: Optional per-component velocity clamp (|v_j| <= v_max).
            None (default) disables clamping; set to a positive float to
            prevent velocity explosion on large search spaces.

    Returns:
        (best_position, best_value)

    Example:
        >>> f = lambda x: sum(xi**2 for xi in x)
        >>> bounds = [(-5, 5), (-5, 5)]
        >>> x_opt, f_opt = particle_swarm_optimization(f, bounds)
    """
    n_dim = len(bounds)

    # Initialize particles
    positions = []
    velocities = []
    personal_best_positions = []
    personal_best_values = []

    for _ in range(n_particles):
        # Random position
        pos = [random.uniform(lower, upper) for lower, upper in bounds]
        positions.append(pos)

        # Random velocity
        vel = [random.uniform(-1, 1) * (upper - lower) * 0.1
               for lower, upper in bounds]
        velocities.append(vel)

        # Initialize personal best
        personal_best_positions.append(pos[:])
        personal_best_values.append(f(pos))

    # Find global best
    global_best_idx = min(range(n_particles), key=lambda i: personal_best_values[i])
    global_best_position = personal_best_positions[global_best_idx][:]
    global_best_value = personal_best_values[global_best_idx]

    # Main loop
    for iteration in range(max_iter):
        for i in range(n_particles):
            # Update velocity
            for j in range(n_dim):
                r1, r2 = random.random(), random.random()

                cognitive = c1 * r1 * (personal_best_positions[i][j] - positions[i][j])
                social = c2 * r2 * (global_best_position[j] - positions[i][j])

                velocities[i][j] = w * velocities[i][j] + cognitive + social

                # Optional velocity clamping to prevent explosion on large spaces
                if v_max is not None:
                    velocities[i][j] = max(-v_max, min(v_max, velocities[i][j]))

            # Update position
            for j in range(n_dim):
                positions[i][j] += velocities[i][j]

                # Enforce bounds
                lower, upper = bounds[j]
                if positions[i][j] < lower:
                    positions[i][j] = lower
                    velocities[i][j] = 0
                elif positions[i][j] > upper:
                    positions[i][j] = upper
                    velocities[i][j] = 0

            # Evaluate
            value = f(positions[i])

            # Update personal best
            if value < personal_best_values[i]:
                personal_best_values[i] = value
                personal_best_positions[i] = positions[i][:]

                # Update global best
                if value < global_best_value:
                    global_best_value = value
                    global_best_position = positions[i][:]

    return global_best_position, global_best_value


def differential_evolution(
    f: Callable[[List[float]], float],
    bounds: List[Tuple[float, float]],
    pop_size: int = 50,
    max_iter: int = 100,
    F: float = 0.8,
    CR: float = 0.9
) -> Tuple[List[float], float]:
    """
    Differential Evolution algorithm.

    Args:
        f: Objective function
        bounds: Parameter bounds
        pop_size: Population size
        max_iter: Maximum iterations
        F: Differential weight
        CR: Crossover probability

    Returns:
        (best_solution, best_value)
    """
    n_dim = len(bounds)

    # Initialize population
    population = []
    for _ in range(pop_size):
        individual = [random.uniform(lower, upper) for lower, upper in bounds]
        population.append(individual)

    # Evaluate
    fitness = [f(ind) for ind in population]
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_solution = population[best_idx][:]
    best_value = fitness[best_idx]

    for iteration in range(max_iter):
        for i in range(pop_size):
            # Mutation: select three random individuals
            candidates = [j for j in range(pop_size) if j != i]
            a, b, c = random.sample(candidates, 3)

            # Mutant vector
            mutant = []
            for j in range(n_dim):
                value = population[a][j] + F * (population[b][j] - population[c][j])
                # Enforce bounds
                lower, upper = bounds[j]
                value = max(lower, min(upper, value))
                mutant.append(value)

            # Crossover
            trial = []
            j_rand = random.randint(0, n_dim - 1)
            for j in range(n_dim):
                if random.random() < CR or j == j_rand:
                    trial.append(mutant[j])
                else:
                    trial.append(population[i][j])

            # Selection
            trial_fitness = f(trial)
            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

                if trial_fitness < best_value:
                    best_value = trial_fitness
                    best_solution = trial[:]

    return best_solution, best_value

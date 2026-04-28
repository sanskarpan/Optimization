# Changelog

All notable changes to the Optimization library are documented here.

## [2.0.0] - 2026-04-28

### Added

#### New Optimizers (`optimizers.py`)
- **AdamW** — Adam with decoupled weight decay (Loshchilov & Hutter, 2019)
- **RAdam** — Rectified Adam with automatic warm-up detection (Liu et al., 2020)
- **Adadelta** — Adaptive learning rate without global LR (Zeiler, 2012)
- **Lookahead** — Meta-optimizer wrapper; slow/fast weight interplay (Zhang et al., 2019)
- **Lion** — EvoLved Sign Momentum; memory-efficient sign-based update (Chen et al., 2023)

#### Enhanced Optimizer Protocol
- `get_state() -> dict` — serialise full optimiser state to JSON-compatible dict
- `load_state(state)` — restore optimiser from saved dict
- `__repr__()` — human-readable summary for all optimisers
- NaN/Inf guard in `step()` — raises `ValueError` on bad gradients

#### New LR Schedules (`learning_rate.py`)
- **LinearWarmupLR** — linear ramp from `start_lr` to `initial_lr` over `warmup_steps`
- **CyclicLR** — triangular / triangular2 / exp_range cycling (Smith, 2017)
- **NoamLR** — Transformer warm-up schedule (Vaswani et al., 2017)
- **ComposedLR** — chain arbitrary schedules sequentially

#### Enhanced Schedule Protocol
- `get_state() / load_state()` and `__repr__()` on all 8 existing schedules

#### Stochastic Methods (`stochastic.py`) — new module
- **`svrg()`** — Stochastic Variance Reduced Gradient (Johnson & Zhang, 2013)
- **`SAGAOptimizer`** — SAGA with per-sample gradient table (Defazio et al., 2014)
- **`SAGOptimizer`** — SAG with memory-efficient average gradient (Schmidt et al., 2013)
- **`iterate_averaging()`** — Polyak–Ruppert suffix averaging (Polyak & Juditsky, 1992)
- **`robbins_monro()`** — Classical stochastic approximation with 1/t^α decay

#### Proximal Operators & Algorithms (`proximal.py`) — new module
- **Proximal operators**: `prox_l1`, `prox_l2_sq`, `prox_linf`, `prox_non_negative`, `prox_box`, `prox_elastic_net`
- **ISTA** — Iterative Shrinkage-Thresholding Algorithm
- **FISTA** — Fast ISTA with Nesterov momentum (Beck & Teboulle, 2009)
- **`proximal_gradient()`** — Backtracking line search for composite objectives
- **`douglas_rachford()`** — Douglas–Rachford splitting for sum of two prox-friendly functions

#### Multi-Objective Optimization (`multi_objective.py`) — new module
- **`dominates()`** / **`pareto_front()`** — Pareto dominance and front extraction
- **`crowding_distance()`** — NSGA-II diversity metric
- **`fast_non_dominated_sort()`** — O(M·N²) ranking for NSGA-II
- **`nsga2()`** — Full NSGA-II algorithm (Deb et al., 2002)
- **`weighted_sum()`** — Scalarization wrapper
- **`generate_weight_vectors()`** — Uniform simplex weight sampling

#### Utilities (`utilities.py`) — new module
- **Gradient checking**: `check_gradient`, `check_jacobian`, `numerical_gradient`, `numerical_hessian`
- **Test functions**: sphere, rosenbrock, rastrigin, ackley, himmelblau, beale, booth, matyas, three_hump_camel, styblinski_tang
- **Benchmarking**: `benchmark_optimizer`, `compare_optimizers`
- **Callbacks**: `Callback`, `CallbackList`, `EarlyStopping`, `GradientMonitor`, `LossLogger`, `DivergenceDetector`
- **Checkpointing**: `save_state`, `load_state` (JSON-based)
- **Guards**: `check_finite`, `safe_step`

#### Line Search (`line_search.py`) — new module
- **`backtracking_line_search()`** — Armijo sufficient decrease
- **`wolfe_line_search()`** — Weak Wolfe conditions (sufficient decrease + curvature)
- **`brent_minimize()`** — Brent's method for scalar minimisation
- **`cubic_interpolation_line_search()`** — Cubic interpolation within a bracket
- **`strong_wolfe_line_search()`** — Full strong Wolfe with zoom phase (Nocedal & Wright, 2006)

#### Second-Order Methods (`second_order.py`) — new module
- **`newton_raphson()`** — Pure Newton with exact Hessian (ridge regularisation fallback)
- **`bfgs()`** — Full BFGS with inverse Hessian approximation
- **`lbfgs()`** — Limited-memory BFGS with two-loop recursion
- **`sr1()`** — Symmetric Rank-1 quasi-Newton update
- **`gauss_newton()`** — Gauss–Newton for nonlinear least squares
- **`levenberg_marquardt()`** — LM with adaptive damping parameter
- **`trust_region()`** — Trust-region with Steihaug CG subproblem solver
- **`newton_cg()`** — Truncated Newton with CG inner solver (forcing sequence)

#### Constrained Optimization (`constrained.py`) — new module
- **Projections**: `project_box`, `project_simplex`, `project_l2_ball`, `project_l1_ball`, `project_linf_ball`
- **`projected_gradient()`** — Projected gradient descent for constraint sets
- **`penalty_method()`** — Quadratic penalty for inequality constraints (optional analytical gradient)
- **`augmented_lagrangian()`** — Augmented Lagrangian for equality constraints
- **`frank_wolfe()`** — Conditional gradient / Frank–Wolfe algorithm
- **`admm()`** — Alternating Direction Method of Multipliers
- **`barrier_method()`** — Log-barrier interior-point method

#### Global Optimization (`global_opt.py`) — new module
- **`simulated_annealing()`** — Geometric cooling with Box–Muller Gaussian proposals
- **`genetic_algorithm()`** — Real-valued GA with SBX crossover and polynomial mutation
- **`differential_evolution()`** — DE with strategy parameter: rand/1, best/1, current-to-best/1
- **`particle_swarm()`** — Standard PSO with inertia weight
- **`nelder_mead()`** — Full simplex method (reflection, expansion, contraction, shrink)
- **`cma_es()`** — Full CMA-ES with Cholesky covariance updates (Hansen, 2006)
- **`basin_hopping()`** — Perturbation + local optimization + Metropolis acceptance
- **`random_search()`** — Uniform random sampling baseline
- **`latin_hypercube_search()`** — Stratified LHS for better space coverage

#### Build & Packaging
- `pyproject.toml` — PEP 517/518 build configuration with setuptools backend

## [1.0.0] - Initial Release

- Gradient-based optimizers: SGD, Momentum, NAG, Adagrad, RMSprop, Adam, AdaMax, NAdam, AMSGrad
- LR schedules: ConstantLR, StepDecayLR, ExponentialDecayLR, CosineAnnealingLR, WarmRestartLR, PolynomialDecayLR, OneCycleLR, ReduceLROnPlateau
- Gradient clipping: `clip_gradients` (L2 norm), `clip_gradients_value` (element-wise)

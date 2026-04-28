"""
Optimization Module
===================

Comprehensive pure-Python optimization library (stdlib only).

Modules
-------
- optimizers       : Gradient-based first-order optimizers
- learning_rate    : Learning rate schedules and decay strategies
- line_search      : Line search methods (Armijo, Wolfe, Brent, strong Wolfe)
- second_order     : Second-order methods (Newton, BFGS, L-BFGS, SR1, LM, trust-region)
- constrained      : Constrained optimization (projections, penalty, AL, Frank-Wolfe, ADMM)
- global_opt       : Global/metaheuristic optimization (SA, GA, DE, PSO, Nelder-Mead, CMA-ES)
- stochastic       : Variance-reduced stochastic methods (SVRG, SAGA, SAG, Polyak averaging)
- proximal         : Proximal operators and algorithms (ISTA, FISTA, Douglas-Rachford)
- multi_objective  : Multi-objective optimization (NSGA-II, Pareto front, weighted sum)
- utilities        : Gradient checking, test functions, benchmarking, callbacks, checkpointing
"""

__version__ = '2.0.0'

# ---------------------------------------------------------------------------
# Learning rate schedules
# ---------------------------------------------------------------------------
from .learning_rate import (
    LRSchedule,
    ConstantLR,
    StepDecayLR,
    ExponentialDecayLR,
    CosineAnnealingLR,
    WarmRestartLR,
    PolynomialDecayLR,
    OneCycleLR,
    ReduceLROnPlateau,
    LinearWarmupLR,
    CyclicLR,
    NoamLR,
    ComposedLR,
)

# ---------------------------------------------------------------------------
# Gradient-based optimizers
# ---------------------------------------------------------------------------
from .optimizers import (
    clip_gradients,
    clip_gradients_value,
    Optimizer,
    SGD,
    Momentum,
    NesterovMomentum,
    Adagrad,
    RMSprop,
    Adam,
    AdaMax,
    NAdam,
    AMSGrad,
    AdamW,
    RAdam,
    Adadelta,
    Lookahead,
    Lion,
)

# ---------------------------------------------------------------------------
# Line search
# ---------------------------------------------------------------------------
from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    brent_minimize,
    cubic_interpolation_line_search,
    strong_wolfe_line_search,
)

# ---------------------------------------------------------------------------
# Second-order methods
# ---------------------------------------------------------------------------
from .second_order import (
    newton_raphson,
    bfgs,
    lbfgs,
    sr1,
    gauss_newton,
    levenberg_marquardt,
    trust_region,
    newton_cg,
)

# ---------------------------------------------------------------------------
# Constrained optimization
# ---------------------------------------------------------------------------
from .constrained import (
    project_box,
    project_simplex,
    project_l2_ball,
    project_l1_ball,
    project_linf_ball,
    projected_gradient,
    penalty_method,
    augmented_lagrangian,
    frank_wolfe,
    admm,
    barrier_method,
)

# ---------------------------------------------------------------------------
# Global / metaheuristic optimization
# ---------------------------------------------------------------------------
from .global_opt import (
    simulated_annealing,
    genetic_algorithm,
    differential_evolution,
    particle_swarm,
    nelder_mead,
    cma_es,
    basin_hopping,
    random_search,
    latin_hypercube_search,
)

# ---------------------------------------------------------------------------
# Stochastic variance-reduced methods
# ---------------------------------------------------------------------------
from .stochastic import (
    svrg,
    SAGAOptimizer,
    SAGOptimizer,
    iterate_averaging,
    robbins_monro,
)

# ---------------------------------------------------------------------------
# Proximal operators and algorithms
# ---------------------------------------------------------------------------
from .proximal import (
    prox_l1,
    prox_l2_sq,
    prox_linf,
    prox_non_negative,
    prox_box,
    prox_elastic_net,
    ista,
    fista,
    proximal_gradient,
    douglas_rachford,
)

# ---------------------------------------------------------------------------
# Multi-objective optimization
# ---------------------------------------------------------------------------
from .multi_objective import (
    dominates,
    pareto_front,
    crowding_distance,
    fast_non_dominated_sort,
    nsga2,
    weighted_sum,
    generate_weight_vectors,
)

# ---------------------------------------------------------------------------
# Utilities: gradient checking, test functions, benchmarking, callbacks
# ---------------------------------------------------------------------------
from .utilities import (
    check_gradient,
    check_jacobian,
    numerical_gradient,
    numerical_hessian,
    sphere,
    sphere_grad,
    rosenbrock,
    rosenbrock_grad,
    rastrigin,
    rastrigin_grad,
    ackley,
    ackley_grad,
    himmelblau,
    himmelblau_grad,
    beale,
    beale_grad,
    booth,
    booth_grad,
    matyas,
    three_hump_camel,
    styblinski_tang,
    BenchmarkResult,
    benchmark_optimizer,
    compare_optimizers,
    Callback,
    CallbackList,
    EarlyStopping,
    GradientMonitor,
    LossLogger,
    DivergenceDetector,
    save_state,
    load_state,
    check_finite,
    safe_step,
)

__all__ = [
    # ---- version ----
    '__version__',

    # ---- LR schedules ----
    'LRSchedule',
    'ConstantLR',
    'StepDecayLR',
    'ExponentialDecayLR',
    'CosineAnnealingLR',
    'WarmRestartLR',
    'PolynomialDecayLR',
    'OneCycleLR',
    'ReduceLROnPlateau',
    'LinearWarmupLR',
    'CyclicLR',
    'NoamLR',
    'ComposedLR',

    # ---- optimizers ----
    'clip_gradients',
    'clip_gradients_value',
    'Optimizer',
    'SGD',
    'Momentum',
    'NesterovMomentum',
    'Adagrad',
    'RMSprop',
    'Adam',
    'AdaMax',
    'NAdam',
    'AMSGrad',
    'AdamW',
    'RAdam',
    'Adadelta',
    'Lookahead',
    'Lion',

    # ---- line search ----
    'backtracking_line_search',
    'wolfe_line_search',
    'brent_minimize',
    'cubic_interpolation_line_search',
    'strong_wolfe_line_search',

    # ---- second-order ----
    'newton_raphson',
    'bfgs',
    'lbfgs',
    'sr1',
    'gauss_newton',
    'levenberg_marquardt',
    'trust_region',
    'newton_cg',

    # ---- constrained ----
    'project_box',
    'project_simplex',
    'project_l2_ball',
    'project_l1_ball',
    'project_linf_ball',
    'projected_gradient',
    'penalty_method',
    'augmented_lagrangian',
    'frank_wolfe',
    'admm',
    'barrier_method',

    # ---- global / metaheuristic ----
    'simulated_annealing',
    'genetic_algorithm',
    'differential_evolution',
    'particle_swarm',
    'nelder_mead',
    'cma_es',
    'basin_hopping',
    'random_search',
    'latin_hypercube_search',

    # ---- stochastic ----
    'svrg',
    'SAGAOptimizer',
    'SAGOptimizer',
    'iterate_averaging',
    'robbins_monro',

    # ---- proximal ----
    'prox_l1',
    'prox_l2_sq',
    'prox_linf',
    'prox_non_negative',
    'prox_box',
    'prox_elastic_net',
    'ista',
    'fista',
    'proximal_gradient',
    'douglas_rachford',

    # ---- multi-objective ----
    'dominates',
    'pareto_front',
    'crowding_distance',
    'fast_non_dominated_sort',
    'nsga2',
    'weighted_sum',
    'generate_weight_vectors',

    # ---- utilities ----
    'check_gradient',
    'check_jacobian',
    'numerical_gradient',
    'numerical_hessian',
    'sphere',
    'sphere_grad',
    'rosenbrock',
    'rosenbrock_grad',
    'rastrigin',
    'rastrigin_grad',
    'ackley',
    'ackley_grad',
    'himmelblau',
    'himmelblau_grad',
    'beale',
    'beale_grad',
    'booth',
    'booth_grad',
    'matyas',
    'three_hump_camel',
    'styblinski_tang',
    'BenchmarkResult',
    'benchmark_optimizer',
    'compare_optimizers',
    'Callback',
    'CallbackList',
    'EarlyStopping',
    'GradientMonitor',
    'LossLogger',
    'DivergenceDetector',
    'save_state',
    'load_state',
    'check_finite',
    'safe_step',
]

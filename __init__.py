"""
Optimization Module
===================

Comprehensive optimization algorithms for machine learning and deep learning.

Modules:
--------
- optimizers: Gradient-based optimizers (SGD, Adam, RMSProp, etc.)
- learning_rate: Learning rate schedules and decay strategies
- line_search: Line search methods (backtracking, Wolfe conditions)
- second_order: Second-order methods (Newton, BFGS, L-BFGS)
- constrained: Constrained optimization (Lagrange, KKT, projected gradient)
- global_opt: Global optimization (simulated annealing, genetic algorithms)
"""

__version__ = "1.0.0"
# Public API count: 42 symbols (see __all__ below)

from .optimizers import (
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
    clip_gradients,
    clip_gradients_value,
)

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
)

from .line_search import (
    backtracking_line_search,
    wolfe_line_search,
    wolfe_conditions,
    armijo_condition,
    exact_line_search_quadratic,
    golden_section_search,
)

from .second_order import (
    newton_step,
    NewtonMethod,
    BFGS,
    LBFGS,
    ConjugateGradient,
)

from .constrained import (
    lagrange_multiplier,
    kkt_conditions,
    projected_gradient_descent,
    barrier_method,
    box_projection,
    simplex_projection,
)

from .global_opt import (
    simulated_annealing,
    genetic_algorithm,
    particle_swarm_optimization,
    differential_evolution,
)

__all__ = [
    # Base classes (for subclassing / isinstance checks)
    'Optimizer',
    'LRSchedule',

    # Optimizers
    'SGD',
    'Momentum',
    'NesterovMomentum',
    'Adagrad',
    'RMSprop',
    'Adam',
    'AdaMax',
    'NAdam',
    'AMSGrad',
    'clip_gradients',
    'clip_gradients_value',

    # Learning Rate Schedules
    'ConstantLR',
    'StepDecayLR',
    'ExponentialDecayLR',
    'CosineAnnealingLR',
    'WarmRestartLR',
    'PolynomialDecayLR',
    'OneCycleLR',
    'ReduceLROnPlateau',

    # Line Search
    'backtracking_line_search',
    'wolfe_line_search',
    'wolfe_conditions',
    'armijo_condition',
    'exact_line_search_quadratic',
    'golden_section_search',

    # Second-Order Methods
    'newton_step',
    'NewtonMethod',
    'BFGS',
    'LBFGS',
    'ConjugateGradient',

    # Constrained Optimization
    'lagrange_multiplier',
    'kkt_conditions',
    'projected_gradient_descent',
    'barrier_method',
    'box_projection',
    'simplex_projection',

    # Global Optimization
    'simulated_annealing',
    'genetic_algorithm',
    'particle_swarm_optimization',
    'differential_evolution',
]

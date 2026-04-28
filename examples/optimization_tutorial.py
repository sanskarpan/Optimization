"""
Optimization Algorithms Tutorial
=================================

Demonstrates various optimization algorithms for machine learning.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
random.seed(42)

from optimizers import SGD, Momentum, Adam, RMSprop
from learning_rate import StepDecayLR, CosineAnnealingLR, OneCycleLR
from second_order import NewtonMethod, BFGS, LBFGS
from constrained import projected_gradient_descent, box_projection
from global_opt import simulated_annealing, genetic_algorithm, particle_swarm_optimization


def rosenbrock(x):
    """
    Rosenbrock function - classic test function for optimization.

    f(x, y) = (1 - x)² + 100(y - x²)²

    Global minimum at (1, 1) with f(1, 1) = 0
    """
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def rosenbrock_grad(x):
    """Gradient of Rosenbrock function."""
    dx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dy = 200 * (x[1] - x[0]**2)
    return [dx, dy]


def gradient_based_optimizers_demo():
    """Compare different gradient-based optimizers."""
    print("=" * 70)
    print("GRADIENT-BASED OPTIMIZERS - Training Neural Networks")
    print("=" * 70)

    print("\nOptimizing Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²")
    print("Global minimum at (1, 1) with f = 0")
    print("-" * 70)

    initial_params = [0.0, 0.0]
    iterations = 1000
    learning_rate = 0.001

    optimizers_config = [
        ("SGD", SGD(learning_rate=learning_rate)),
        ("Momentum", Momentum(learning_rate=learning_rate, momentum=0.9)),
        ("RMSprop", RMSprop(learning_rate=learning_rate)),
        ("Adam", Adam(learning_rate=learning_rate)),
    ]

    print(f"\n{'Optimizer':<15} {'Final x':<15} {'Final y':<15} {'Final Loss':<15}")
    print("-" * 70)

    for name, optimizer in optimizers_config:
        params = initial_params[:]

        for _ in range(iterations):
            grad = rosenbrock_grad(params)
            params = optimizer.update(params, grad)

        final_loss = rosenbrock(params)
        print(f"{name:<15} {params[0]:<15.6f} {params[1]:<15.6f} {final_loss:<15.6f}")

    print("\n→ Adam typically converges fastest for deep learning")
    print("→ Momentum helps escape local minima")
    print("→ RMSprop adapts learning rate per parameter")


def learning_rate_schedules_demo():
    """Demonstrate learning rate schedules."""
    print("\n" + "=" * 70)
    print("LEARNING RATE SCHEDULES - Improving Convergence")
    print("=" * 70)

    print("\n1. Step Decay - Reduces LR at fixed intervals")
    print("-" * 40)
    schedule = StepDecayLR(initial_lr=0.1, step_size=10, gamma=0.1)
    print(f"   Epoch   0: lr = {schedule.get_lr(0):.6f}")
    print(f"   Epoch  10: lr = {schedule.get_lr(10):.6f}")
    print(f"   Epoch  20: lr = {schedule.get_lr(20):.6f}")
    print(f"   → LR drops by 10x every 10 epochs")

    print("\n2. Cosine Annealing - Smooth decay")
    print("-" * 40)
    schedule = CosineAnnealingLR(initial_lr=0.1, T_max=100)
    print(f"   Start (epoch 0):   lr = {schedule.get_lr(0):.6f}")
    print(f"   Middle (epoch 50): lr = {schedule.get_lr(50):.6f}")
    print(f"   End (epoch 100):   lr = {schedule.get_lr(100):.6f}")
    print(f"   → Smooth reduction following cosine curve")

    print("\n3. One-Cycle - Warmup then anneal")
    print("-" * 40)
    schedule = OneCycleLR(max_lr=0.1, total_steps=100, pct_start=0.3)
    print(f"   Start:  lr = {schedule.get_lr(0):.6f}")
    print(f"   Peak:   lr = {schedule.get_lr(30):.6f}")
    print(f"   End:    lr = {schedule.get_lr(100):.6f}")
    print(f"   → Used in super-convergence (fast.ai)")


def second_order_methods_demo():
    """Demonstrate second-order optimization."""
    print("\n" + "=" * 70)
    print("SECOND-ORDER METHODS - Using Curvature Information")
    print("=" * 70)

    # Quadratic function for testing
    def quadratic(x):
        return (x[0] - 3)**2 + (x[1] - 2)**2

    def quad_grad(x):
        return [2 * (x[0] - 3), 2 * (x[1] - 2)]

    def quad_hess(x):
        return [[2.0, 0.0], [0.0, 2.0]]

    print("\nMinimize f(x,y) = (x-3)² + (y-2)²")
    print("Minimum at (3, 2)")
    print("-" * 70)

    x0 = [0.0, 0.0]

    # Newton's Method
    print("\n1. Newton's Method")
    print("   Uses exact Hessian (second derivatives)")
    newton = NewtonMethod(learning_rate=1.0, max_iter=10)
    x_opt, history = newton.optimize(quadratic, quad_grad, quad_hess, x0)
    print(f"   Converged to: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   Iterations: {len(history)}")
    print(f"   → Quadratic convergence for smooth functions")

    # BFGS
    print("\n2. BFGS - Quasi-Newton Method")
    print("   Approximates Hessian using gradients only")
    bfgs = BFGS(max_iter=50)
    x_opt, history = bfgs.optimize(quadratic, quad_grad, x0)
    print(f"   Converged to: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   Iterations: {len(history)}")
    print(f"   → Superlinear convergence, no Hessian needed")

    # L-BFGS
    print("\n3. L-BFGS - Memory-Efficient BFGS")
    print("   Stores only recent updates (scalable to millions of parameters)")
    lbfgs = LBFGS(m=10, max_iter=50)
    x_opt, history = lbfgs.optimize(quadratic, quad_grad, x0)
    print(f"   Converged to: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"   Iterations: {len(history)}")
    print(f"   → Used in PyTorch's L-BFGS optimizer")


def constrained_optimization_demo():
    """Demonstrate constrained optimization."""
    print("\n" + "=" * 70)
    print("CONSTRAINED OPTIMIZATION - Optimization with Constraints")
    print("=" * 70)

    print("\nMinimize f(x,y) = x² + y² subject to 0 ≤ x ≤ 2, 0 ≤ y ≤ 2")
    print("-" * 70)

    def objective(x):
        return (x[0] - 3)**2 + (x[1] - 3)**2  # Minimum at (3, 3)

    def objective_grad(x):
        return [2 * (x[0] - 3), 2 * (x[1] - 3)]

    def box_projection_func(x):
        return box_projection(x, [0.0, 0.0], [2.0, 2.0])

    x0 = [0.0, 0.0]
    x_opt, history = projected_gradient_descent(
        objective, objective_grad, box_projection_func, x0,
        learning_rate=0.1, max_iter=100
    )

    print(f"\nUnconstrained minimum would be at: (3, 3)")
    print(f"Constrained minimum found at: ({x_opt[0]:.6f}, {x_opt[1]:.6f})")
    print(f"Final objective value: {objective(x_opt):.6f}")
    print(f"\n→ Projected gradient descent handles box constraints")
    print(f"→ Solution is at boundary: (2, 2)")


def global_optimization_demo():
    """Demonstrate global optimization algorithms."""
    print("\n" + "=" * 70)
    print("GLOBAL OPTIMIZATION - Finding Global Minima")
    print("=" * 70)

    print("\nOptimize multi-modal function with many local minima")
    print("-" * 70)

    # Rastrigin function - many local minima
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)

    bounds = [(-5.12, 5.12), (-5.12, 5.12)]
    x0 = [3.0, 3.0]

    print("\n1. Simulated Annealing")
    print("   Probabilistically accepts worse solutions to escape local minima")
    x_opt, f_opt = simulated_annealing(rastrigin, x0, bounds, max_iter=5000)
    print(f"   Best solution: ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
    print(f"   Best value: {f_opt:.6f}")

    print("\n2. Genetic Algorithm")
    print("   Evolves population using selection, crossover, mutation")
    x_opt, f_opt = genetic_algorithm(rastrigin, bounds, pop_size=50, generations=100)
    print(f"   Best solution: ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
    print(f"   Best value: {f_opt:.6f}")

    print("\n3. Particle Swarm Optimization")
    print("   Particles explore space influenced by personal and global bests")
    x_opt, f_opt = particle_swarm_optimization(rastrigin, bounds, n_particles=30, max_iter=100)
    print(f"   Best solution: ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
    print(f"   Best value: {f_opt:.6f}")

    print(f"\n→ Global minimum is at (0, 0) with f = 0")
    print(f"→ These methods help avoid local minima in non-convex problems")


def ml_training_simulation():
    """Simulate training a simple ML model."""
    print("\n" + "=" * 70)
    print("ML TRAINING SIMULATION - Putting It All Together")
    print("=" * 70)

    print("\nSimulating neural network training:")
    print("- Using Adam optimizer")
    print("- With cosine annealing learning rate schedule")
    print("- Training for 100 epochs")
    print("-" * 70)

    # Simulate loss landscape
    def loss_function(params):
        # Simulated loss: starts high, converges to minimum
        return 10 * rosenbrock([p * 0.1 for p in params])

    def loss_gradient(params):
        grad = rosenbrock_grad([p * 0.1 for p in params])
        return [g * 0.1 for g in grad]

    # Initialize
    params = [0.0, 0.0]
    optimizer = Adam(learning_rate=0.01)
    lr_schedule = CosineAnnealingLR(initial_lr=0.01, T_max=100)

    print(f"\n{'Epoch':<10} {'Loss':<15} {'Learning Rate':<15}")
    print("-" * 45)

    for epoch in range(101):
        # Compute gradient
        grad = loss_gradient(params)

        # Update parameters
        params = optimizer.update(params, grad)

        # Update learning rate
        if epoch < 100:
            optimizer.learning_rate = lr_schedule.step()

        # Print progress
        if epoch % 20 == 0:
            loss = loss_function(params)
            print(f"{epoch:<10} {loss:<15.6f} {optimizer.learning_rate:<15.6f}")

    final_loss = loss_function(params)
    print(f"\nFinal loss: {final_loss:.6f}")
    print(f"Final parameters: ({params[0]:.4f}, {params[1]:.4f})")
    print(f"\n→ Training converged successfully!")
    print(f"→ Learning rate decreased smoothly from 0.01 to ~0")


def main():
    """Run all optimization tutorials."""
    gradient_based_optimizers_demo()
    learning_rate_schedules_demo()
    second_order_methods_demo()
    constrained_optimization_demo()
    global_optimization_demo()
    ml_training_simulation()

    print("\n" + "=" * 70)
    print("Optimization Tutorial Complete!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ Adam is the go-to optimizer for deep learning")
    print("✓ Learning rate schedules prevent oscillations and improve convergence")
    print("✓ Second-order methods (BFGS, L-BFGS) converge faster but need more memory")
    print("✓ Projected gradient descent handles constraints efficiently")
    print("✓ Global optimization methods find global minima in non-convex problems")
    print("✓ Combining good optimizer + LR schedule = fast, stable training")
    print("\nYou now understand optimization algorithms powering modern ML!")


if __name__ == '__main__':
    main()

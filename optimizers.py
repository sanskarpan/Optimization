"""
Gradient-Based Optimizers
==========================

Implementation of optimizers used in deep learning from scratch.

Includes: SGD, Momentum, Nesterov, Adagrad, RMSprop, Adam, and variants.

Applications in ML/DL:
- Training neural networks
- Parameter updates during backpropagation
- Adaptive learning rates
- Momentum for faster convergence
"""

import math
from typing import List, Callable, Optional, Tuple
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        """
        Args:
            learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate
        self.iterations = 0

    @abstractmethod
    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """
        Update parameters given gradients.

        Args:
            params: Current parameter values
            gradients: Gradients of loss w.r.t. parameters

        Returns:
            Updated parameters
        """
        pass

    def reset(self):
        """Reset optimizer state."""
        self.iterations = 0


class SGD(Optimizer):
    """
    Stochastic Gradient Descent

    θ_new = θ_old - η * ∇L(θ)

    Most basic optimizer. Updates parameters in direction of negative gradient.

    Args:
        learning_rate: Step size
        weight_decay: L2 regularization coefficient
    """

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        super().__init__(learning_rate)
        self.weight_decay = weight_decay

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using vanilla SGD."""
        updated_params = []

        for param, grad in zip(params, gradients):
            # Add weight decay (L2 regularization)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update: θ = θ - η * ∇L
            updated_param = param - self.learning_rate * grad
            updated_params.append(updated_param)

        self.iterations += 1
        return updated_params


class Momentum(Optimizer):
    """
    SGD with Momentum

    v_t = β * v_{t-1} + ∇L(θ)   (undampened form, matches PyTorch default)
    θ_new = θ_old - η * v_t

    Accelerates SGD by accumulating velocity in consistent gradient directions.
    Note: uses the undampened form (no (1-β) factor on the gradient); for the
    dampened form use `v_t = β*v_{t-1} + (1-β)*∇L`.

    Args:
        learning_rate: Step size
        momentum: Momentum coefficient (typically 0.9)
        weight_decay: L2 regularization
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using momentum."""
        # Initialize velocity on first iteration
        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update velocity: v = β * v + ∇L
            self.velocity[i] = self.momentum * self.velocity[i] + grad

            # Update parameters: θ = θ - η * v
            updated_param = param - self.learning_rate * self.velocity[i]
            updated_params.append(updated_param)

        self.iterations += 1
        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.velocity = None


class NesterovMomentum(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG) — PyTorch-equivalent form.

    PyTorch / Sutskever form (what this class implements):
      v_t = β * v_{t-1} + g_t
      θ   = θ - η * (g_t + β * v_t)

    This is mathematically equivalent to evaluating the gradient at the
    lookahead position θ - η*β*v_{t-1}, but expressed in terms of the
    current-position gradient g_t for computational convenience.
    The class-level formula is NOT the "raw" NAG formula; the inner
    ``update`` docstring describes what the code actually computes.

    Args:
        learning_rate: Step size
        momentum: Momentum coefficient
        weight_decay: L2 regularization
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """
        Update parameters using Nesterov momentum.

        Equivalent to PyTorch's nesterov=True form:
          v_t = β * v_{t-1} + g
          θ   = θ - η * (g + β * v_t)
        """
        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update velocity: v_t = β * v_{t-1} + g
            self.velocity[i] = self.momentum * self.velocity[i] + grad

            # Nesterov update: θ = θ - η * (g + β * v_t)
            updated_param = param - self.learning_rate * (
                grad + self.momentum * self.velocity[i]
            )
            updated_params.append(updated_param)

        self.iterations += 1
        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.velocity = None


class Adagrad(Optimizer):
    """
    Adaptive Gradient Algorithm

    G_t = G_{t-1} + (∇L)²
    θ_new = θ_old - (η / √(G_t + ε)) * ∇L

    Adapts learning rate for each parameter based on historical gradients.
    Good for sparse data.

    Args:
        learning_rate: Initial learning rate
        epsilon: Small constant for numerical stability
    """

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.accumulated_gradients: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using Adagrad."""
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [0.0] * len(params)

        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Accumulate squared gradients
            self.accumulated_gradients[i] += grad ** 2

            # Adaptive learning rate
            adapted_lr = self.learning_rate / (
                math.sqrt(self.accumulated_gradients[i]) + self.epsilon
            )

            # Update parameter
            updated_param = param - adapted_lr * grad
            updated_params.append(updated_param)

        self.iterations += 1
        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.accumulated_gradients = None


class RMSprop(Optimizer):
    """
    Root Mean Square Propagation

    E[g²]_t = β * E[g²]_{t-1} + (1-β) * (∇L)²
    θ_new = θ_old - (η / √(E[g²]_t + ε)) * ∇L

    Uses exponential moving average of squared gradients.
    Addresses Adagrad's diminishing learning rate problem.

    Args:
        learning_rate: Learning rate
        rho: Decay rate for moving average (typically 0.9)
        epsilon: Small constant for numerical stability
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        rho: float = 0.9,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.rho = rho
        self.epsilon = epsilon
        self.squared_gradients: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using RMSprop."""
        if self.squared_gradients is None:
            self.squared_gradients = [0.0] * len(params)

        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update moving average of squared gradients
            self.squared_gradients[i] = (
                self.rho * self.squared_gradients[i] +
                (1 - self.rho) * grad ** 2
            )

            # Adaptive learning rate
            adapted_lr = self.learning_rate / (
                math.sqrt(self.squared_gradients[i]) + self.epsilon
            )

            # Update parameter
            updated_param = param - adapted_lr * grad
            updated_params.append(updated_param)

        self.iterations += 1
        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.squared_gradients = None


class Adam(Optimizer):
    """
    Adaptive Moment Estimation

    m_t = β₁ * m_{t-1} + (1-β₁) * ∇L
    v_t = β₂ * v_{t-1} + (1-β₂) * (∇L)²
    m̂_t = m_t / (1 - β₁^t)  # Bias correction
    v̂_t = v_t / (1 - β₂^t)  # Bias correction
    θ_new = θ_old - η * m̂_t / (√v̂_t + ε)

    Combines momentum and RMSprop. Most popular optimizer for deep learning.

    Args:
        learning_rate: Learning rate (default 0.001)
        beta1: Exponential decay rate for first moment (default 0.9)
        beta2: Exponential decay rate for second moment (default 0.999)
        epsilon: Small constant for numerical stability
        weight_decay: L2 regularization coefficient
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m: Optional[List[float]] = None  # First moment
        self.v: Optional[List[float]] = None  # Second moment

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using Adam."""
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.iterations += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Add weight decay
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.iterations)
            v_hat = self.v[i] / (1 - self.beta2 ** self.iterations)

            # Update parameter
            updated_param = param - self.learning_rate * m_hat / (
                math.sqrt(v_hat) + self.epsilon
            )
            updated_params.append(updated_param)

        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m = None
        self.v = None


class AdaMax(Optimizer):
    """
    AdaMax - Variant of Adam based on infinity norm

    m_t = β₁ * m_{t-1} + (1-β₁) * ∇L
    u_t = max(β₂ * u_{t-1}, |∇L|)
    θ_new = θ_old - (η/(1-β₁^t)) * m_t / u_t

    More stable than Adam in some cases.

    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for infinity norm
        epsilon: Small constant
    """

    def __init__(
        self,
        learning_rate: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[List[float]] = None
        self.u: Optional[List[float]] = None  # Exponentially weighted infinity norm

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using AdaMax."""
        if self.m is None:
            self.m = [0.0] * len(params)
            self.u = [0.0] * len(params)

        self.iterations += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update first moment
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # Update infinity norm
            self.u[i] = max(self.beta2 * self.u[i], abs(grad))

            # Bias-corrected learning rate
            lr_t = self.learning_rate / (1 - self.beta1 ** self.iterations)

            # Update parameter
            updated_param = param - lr_t * self.m[i] / (self.u[i] + self.epsilon)
            updated_params.append(updated_param)

        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m = None
        self.u = None


class NAdam(Optimizer):
    """
    Nesterov-accelerated Adaptive Moment Estimation

    Combines Adam and Nesterov momentum.

    Args:
        learning_rate: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        epsilon: Small constant
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using NAdam."""
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.iterations += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction for second moment
            v_hat = self.v[i] / (1 - self.beta2 ** self.iterations)

            # NAdam lookahead: m_bar approximates m_{t+1} bias-corrected.
            # The first term uses (1 - β₁^{t+1}) — the next step's denominator —
            # while the raw-gradient term uses (1 - β₁^t), matching Dozat (2016).
            m_hat_next = self.m[i] / (1 - self.beta1 ** (self.iterations + 1))
            m_bar = self.beta1 * m_hat_next + (1 - self.beta1) * grad / (
                1 - self.beta1 ** self.iterations
            )

            # Update parameter
            updated_param = param - self.learning_rate * m_bar / (
                math.sqrt(v_hat) + self.epsilon
            )
            updated_params.append(updated_param)

        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m = None
        self.v = None


class AMSGrad(Optimizer):
    """
    AMSGrad - Variant of Adam with guaranteed convergence

    Fixes potential non-convergence issue in Adam by maintaining the running
    maximum of the second moment estimates, ensuring a non-increasing effective
    learning rate per parameter.

    Note: This implementation uses the *bias-corrected* AMSGrad variant
    (v_hat = v / (1 - β₂^t) before taking the max), which departs from the
    original Reddi et al. (2018) formulation.  The bias-corrected v̂ is
    *larger* than the raw v_t in early training, so the denominator is
    larger and the effective step size is *smaller* (more conservative)
    in early steps compared to the original paper's formulation.

    Args:
        learning_rate: Learning rate
        beta1: First moment decay
        beta2: Second moment decay
        epsilon: Small constant
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None
        self.v_hat_max: Optional[List[float]] = None

    def update(self, params: List[float], gradients: List[float]) -> List[float]:
        """Update parameters using AMSGrad."""
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
            self.v_hat_max = [0.0] * len(params)

        self.iterations += 1
        updated_params = []

        for i, (param, grad) in enumerate(zip(params, gradients)):
            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction for second moment
            v_hat = self.v[i] / (1 - self.beta2 ** self.iterations)

            # Maintain maximum
            self.v_hat_max[i] = max(self.v_hat_max[i], v_hat)

            # Bias correction for first moment
            m_hat = self.m[i] / (1 - self.beta1 ** self.iterations)

            # Update parameter using max of v_hat
            updated_param = param - self.learning_rate * m_hat / (
                math.sqrt(self.v_hat_max[i]) + self.epsilon
            )
            updated_params.append(updated_param)

        return updated_params

    def reset(self):
        """Reset optimizer state."""
        super().reset()
        self.m = None
        self.v = None
        self.v_hat_max = None


def clip_gradients(gradients: List[float], max_norm: float) -> List[float]:
    """
    Gradient clipping to prevent exploding gradients.

    Args:
        gradients: Gradients to clip
        max_norm: Maximum allowed L2 norm

    Returns:
        Clipped gradients

    Example:
        >>> grads = [5.0, 10.0, 15.0]
        >>> clipped = clip_gradients(grads, max_norm=5.0)
    """
    # Compute L2 norm
    norm = math.sqrt(sum(g ** 2 for g in gradients))

    if norm > max_norm:
        # Scale down
        scale = max_norm / norm
        return [g * scale for g in gradients]
    else:
        return gradients[:]   # return a copy so caller mutations don't alias source


def clip_gradients_value(
    gradients: List[float],
    clip_value: float
) -> List[float]:
    """
    Clip gradients by value (element-wise).

    Args:
        gradients: Gradients to clip
        clip_value: Maximum absolute value

    Returns:
        Clipped gradients

    Example:
        >>> grads = [5.0, -10.0, 2.0]
        >>> clipped = clip_gradients_value(grads, clip_value=3.0)
        >>> # Returns [3.0, -3.0, 2.0]
    """
    return [max(min(g, clip_value), -clip_value) for g in gradients]

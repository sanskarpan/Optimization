"""
Gradient-Based Optimizers
==========================
First-order adaptive optimizers for iterative parameter updates.

All implementations are pure Python (stdlib only: math, random, warnings,
typing, abc).  No third-party dependencies are required.

Each optimizer follows the same protocol:

* ``step(params, grads) -> new_params``   — one update step
* ``reset()``                              — reinitialise all state
* ``get_state() -> dict``                  — serialise hyperparams + state
* ``load_state(state)``                    — restore from dict
* ``__repr__()``                           — human-readable summary

References
----------
* Rumelhart et al. (1986)  — SGD with momentum.
* Duchi et al. (2011)      — Adagrad.
* Tieleman & Hinton (2012) — RMSprop.
* Zeiler (2012)            — Adadelta.
* Kingma & Ba (2015)       — Adam.
* Dozat (2016)             — NAdam.
* Reddi et al. (2018)      — AMSGrad.
* Loshchilov & Hutter (2019) — AdamW (decoupled weight decay).
* Liu et al. (2020)        — RAdam (Rectified Adam).
* Zhang et al. (2019)      — Lookahead.
* Chen et al. (2023)       — Lion (EvoLved Sign Momentum).
"""

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sign(x: float) -> float:
    """Return the sign of *x*: +1.0, -1.0, or 0.0."""
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    return 0.0


# ---------------------------------------------------------------------------
# Gradient clipping helpers
# ---------------------------------------------------------------------------

def clip_gradients(grads: List[float], max_norm: float) -> List[float]:
    """Clip gradients by global L2 norm.

    If the L2 norm of *grads* exceeds *max_norm*, all elements are scaled
    down proportionally so that the resulting norm equals *max_norm*.

    Parameters
    ----------
    grads:
        Gradient vector.
    max_norm:
        Maximum allowed L2 norm (must be > 0).

    Returns
    -------
    List[float]
        Clipped gradient vector (new list; input is not modified).

    Raises
    ------
    ValueError
        If *max_norm* <= 0.
    """
    if max_norm <= 0:
        raise ValueError(f"max_norm must be > 0, got {max_norm}")
    norm = math.sqrt(sum(g * g for g in grads))
    if norm > max_norm:
        scale = max_norm / norm
        return [g * scale for g in grads]
    return list(grads)


def clip_gradients_value(grads: List[float], clip_value: float) -> List[float]:
    """Clip gradients element-wise to the range ``[-clip_value, +clip_value]``.

    Parameters
    ----------
    grads:
        Gradient vector.
    clip_value:
        Absolute clipping bound (must be > 0).

    Returns
    -------
    List[float]
        Clipped gradient vector (new list; input is not modified).

    Raises
    ------
    ValueError
        If *clip_value* <= 0.
    """
    if clip_value <= 0:
        raise ValueError(f"clip_value must be > 0, got {clip_value}")
    return [max(-clip_value, min(clip_value, g)) for g in grads]


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class Optimizer(ABC):
    """Abstract base class for all gradient-based optimizers.

    Subclasses must implement :meth:`step` and :meth:`reset`.

    Attributes
    ----------
    iterations:
        Number of :meth:`step` calls made since the last :meth:`reset`.
    """

    def __init__(self) -> None:
        self.iterations: int = 0

    @abstractmethod
    def step(self, params: List[float], grads: List[float]) -> List[float]:
        """Perform one parameter update.

        Parameters
        ----------
        params:
            Current parameter vector (not modified in-place).
        grads:
            Gradient vector, same length as *params*.

        Returns
        -------
        List[float]
            Updated parameter vector.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset all internal state (moments, accumulators, step count)."""

    def get_state(self) -> dict:
        """Return a JSON-serialisable snapshot of all hyperparameters and state."""
        return {'iterations': self.iterations}

    def load_state(self, state: dict) -> None:
        """Restore from a dict previously returned by :meth:`get_state`."""
        self.iterations = state.get('iterations', 0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# SGD
# ---------------------------------------------------------------------------

class SGD(Optimizer):
    """Stochastic Gradient Descent (with optional L2 weight decay).

    Update rule::

        θ ← θ - lr * (g + weight_decay * θ)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.01).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        wd = self.weight_decay
        new_params = [
            p - lr * (g + wd * p)
            for p, g in zip(params, grads)
        ]
        return new_params

    def reset(self) -> None:
        self.iterations = 0

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.weight_decay = state.get('weight_decay', self.weight_decay)

    def __repr__(self) -> str:
        return (
            f"SGD(lr={self.learning_rate}, weight_decay={self.weight_decay})"
        )


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

class Momentum(Optimizer):
    """SGD with momentum (PyTorch / undampened style).

    Update rule::

        v ← β*v + g
        θ ← θ - lr * v

    Parameters
    ----------
    learning_rate:
        Step size (default 0.01).
    momentum:
        Momentum coefficient β (default 0.9).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        beta = self.momentum
        wd = self.weight_decay

        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        new_v: List[float] = []
        new_params: List[float] = []
        for v, p, g in zip(self.velocity, params, grads):
            g_reg = g + wd * p
            v_new = beta * v + g_reg
            new_v.append(v_new)
            new_params.append(p - lr * v_new)

        self.velocity = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.velocity = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocity': list(self.velocity) if self.velocity is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.momentum = state.get('momentum', self.momentum)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        v = state.get('velocity')
        self.velocity = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"Momentum(lr={self.learning_rate}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay})"
        )


# ---------------------------------------------------------------------------
# Nesterov Momentum
# ---------------------------------------------------------------------------

class NesterovMomentum(Optimizer):
    """SGD with Nesterov momentum (lookahead gradient).

    Update rule::

        v ← β*v + g
        θ ← θ - lr * (g + β*v)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.01).
    momentum:
        Momentum coefficient β (default 0.9).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        beta = self.momentum
        wd = self.weight_decay

        if self.velocity is None:
            self.velocity = [0.0] * len(params)

        new_v: List[float] = []
        new_params: List[float] = []
        for v, p, g in zip(self.velocity, params, grads):
            g_reg = g + wd * p
            v_new = beta * v + g_reg
            new_v.append(v_new)
            # Nesterov: use g + beta * v_new  (lookahead)
            new_params.append(p - lr * (g_reg + beta * v_new))

        self.velocity = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.velocity = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'velocity': list(self.velocity) if self.velocity is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.momentum = state.get('momentum', self.momentum)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        v = state.get('velocity')
        self.velocity = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"NesterovMomentum(lr={self.learning_rate}, momentum={self.momentum}, "
            f"weight_decay={self.weight_decay})"
        )


# ---------------------------------------------------------------------------
# Adagrad
# ---------------------------------------------------------------------------

class Adagrad(Optimizer):
    """Adagrad — adaptive learning rates via accumulated squared gradients.

    Update rule::

        G_t ← G_{t-1} + g_t²
        θ ← θ - (lr / sqrt(G_t + ε)) * g

    Parameters
    ----------
    learning_rate:
        Initial step size (default 0.01).
    eps:
        Numerical stability (default 1e-8).
    """

    def __init__(self, learning_rate: float = 0.01, eps: float = 1e-8) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.eps = eps
        self.G: Optional[List[float]] = None  # accumulated squared grads

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        eps = self.eps

        if self.G is None:
            self.G = [0.0] * len(params)

        new_G: List[float] = []
        new_params: List[float] = []
        for G_i, p, g in zip(self.G, params, grads):
            G_new = G_i + g * g
            new_G.append(G_new)
            new_params.append(p - lr / math.sqrt(G_new + eps) * g)

        self.G = new_G
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.G = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'eps': self.eps,
            'G': list(self.G) if self.G is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.eps = state.get('eps', self.eps)
        G = state.get('G')
        self.G = list(G) if G is not None else None

    def __repr__(self) -> str:
        return f"Adagrad(lr={self.learning_rate}, eps={self.eps})"


# ---------------------------------------------------------------------------
# RMSprop
# ---------------------------------------------------------------------------

class RMSprop(Optimizer):
    """RMSprop — exponential moving average of squared gradients.

    Update rule::

        E[g²]_t ← decay * E[g²]_{t-1} + (1-decay) * g_t²
        θ ← θ - (lr / sqrt(E[g²]_t + ε)) * g

    Parameters
    ----------
    learning_rate:
        Step size (default 0.001).
    decay:
        Decay rate for running average (default 0.9).
    eps:
        Numerical stability (default 1e-8).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        decay: float = 0.9,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.decay = decay
        self.eps = eps
        self.weight_decay = weight_decay
        self.E_g2: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        rho = self.decay
        eps = self.eps
        wd = self.weight_decay

        if self.E_g2 is None:
            self.E_g2 = [0.0] * len(params)

        new_E: List[float] = []
        new_params: List[float] = []
        for E_i, p, g in zip(self.E_g2, params, grads):
            g_reg = g + wd * p
            E_new = rho * E_i + (1.0 - rho) * g_reg * g_reg
            new_E.append(E_new)
            new_params.append(p - lr / math.sqrt(E_new + eps) * g_reg)

        self.E_g2 = new_E
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.E_g2 = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'decay': self.decay,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'E_g2': list(self.E_g2) if self.E_g2 is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.decay = state.get('decay', self.decay)
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        E = state.get('E_g2')
        self.E_g2 = list(E) if E is not None else None

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.learning_rate}, decay={self.decay}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# Adam
# ---------------------------------------------------------------------------

class Adam(Optimizer):
    """Adam — Adaptive Moment Estimation (Kingma & Ba, 2015).

    Update rule::

        m_t ← β₁*m_{t-1} + (1-β₁)*g_t
        v_t ← β₂*v_{t-1} + (1-β₂)*g_t²
        m̂_t = m_t / (1 - β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ ← θ - lr * m̂_t / (sqrt(v̂_t) + ε)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.001).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    weight_decay:
        L2 regularisation coefficient (default 0.0).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t: int = 0
        self.m: Optional[List[float]] = None  # first moment
        self.v: Optional[List[float]] = None  # second moment

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        wd = self.weight_decay
        t = self.t

        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        new_m: List[float] = []
        new_v: List[float] = []
        new_params: List[float] = []
        for m_i, v_i, p, g in zip(self.m, self.v, params, grads):
            g_reg = g + wd * p
            m_new = b1 * m_i + (1.0 - b1) * g_reg
            v_new = b2 * v_i + (1.0 - b2) * g_reg * g_reg
            new_m.append(m_new)
            new_v.append(v_new)
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            new_params.append(p - lr * m_hat / (math.sqrt(v_hat) + eps))

        self.m = new_m
        self.v = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.v = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': list(self.m) if self.m is not None else None,
            'v': list(self.v) if self.v is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        m = state.get('m')
        self.m = list(m) if m is not None else None
        v = state.get('v')
        self.v = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"Adam(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# AdaMax
# ---------------------------------------------------------------------------

class AdaMax(Optimizer):
    """AdaMax — L∞-norm variant of Adam (Kingma & Ba, 2015).

    Uses the infinity norm for the second moment instead of L2::

        m_t ← β₁*m_{t-1} + (1-β₁)*g_t
        u_t ← max(β₂*u_{t-1}, |g_t|)
        θ ← θ - (lr / (1 - β₁^t)) * m_t / (u_t + ε)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.002).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Infinity-norm decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 0.002,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0
        self.m: Optional[List[float]] = None
        self.u: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        t = self.t

        if self.m is None:
            self.m = [0.0] * len(params)
            self.u = [0.0] * len(params)

        bc1 = 1.0 - b1 ** t

        new_m: List[float] = []
        new_u: List[float] = []
        new_params: List[float] = []
        for m_i, u_i, p, g in zip(self.m, self.u, params, grads):
            m_new = b1 * m_i + (1.0 - b1) * g
            u_new = max(b2 * u_i, abs(g))
            new_m.append(m_new)
            new_u.append(u_new)
            new_params.append(p - (lr / bc1) * m_new / (u_new + eps))

        self.m = new_m
        self.u = new_u
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.u = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': list(self.m) if self.m is not None else None,
            'u': list(self.u) if self.u is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        m = state.get('m')
        self.m = list(m) if m is not None else None
        u = state.get('u')
        self.u = list(u) if u is not None else None

    def __repr__(self) -> str:
        return (
            f"AdaMax(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# NAdam
# ---------------------------------------------------------------------------

class NAdam(Optimizer):
    """NAdam — Nesterov-accelerated Adam (Dozat, 2016).

    Incorporates a Nesterov lookahead into the first-moment update::

        m_t ← β₁*m_{t-1} + (1-β₁)*g_t
        v_t ← β₂*v_{t-1} + (1-β₂)*g_t²
        m̂_t = β₁*m_t/(1-β₁^{t+1}) + (1-β₁)*g_t/(1-β₁^t)
        v̂_t = v_t / (1 - β₂^t)
        θ ← θ - lr * m̂_t / (sqrt(v̂_t) + ε)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.001).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        t = self.t

        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        bc1_t = 1.0 - b1 ** t
        bc1_t1 = 1.0 - b1 ** (t + 1)
        bc2 = 1.0 - b2 ** t

        new_m: List[float] = []
        new_v: List[float] = []
        new_params: List[float] = []
        for m_i, v_i, p, g in zip(self.m, self.v, params, grads):
            m_new = b1 * m_i + (1.0 - b1) * g
            v_new = b2 * v_i + (1.0 - b2) * g * g
            new_m.append(m_new)
            new_v.append(v_new)
            # Nesterov bias-corrected first moment
            m_hat = b1 * m_new / bc1_t1 + (1.0 - b1) * g / bc1_t
            v_hat = v_new / bc2
            new_params.append(p - lr * m_hat / (math.sqrt(v_hat) + eps))

        self.m = new_m
        self.v = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.v = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': list(self.m) if self.m is not None else None,
            'v': list(self.v) if self.v is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        m = state.get('m')
        self.m = list(m) if m is not None else None
        v = state.get('v')
        self.v = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"NAdam(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# AMSGrad
# ---------------------------------------------------------------------------

class AMSGrad(Optimizer):
    """AMSGrad — Adam with guaranteed convergence via max second moment.

    Tracks the running maximum of the bias-corrected second moment::

        m_t ← β₁*m_{t-1} + (1-β₁)*g_t
        v_t ← β₂*v_{t-1} + (1-β₂)*g_t²
        v̂_t_max = max(v̂_{t-1}_max, v_t / (1-β₂^t))
        θ ← θ - lr * m_hat / (sqrt(v̂_t_max) + ε)

    Parameters
    ----------
    learning_rate:
        Step size (default 0.001).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None
        self.v_hat_max: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        t = self.t

        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)
            self.v_hat_max = [0.0] * len(params)

        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        new_m: List[float] = []
        new_v: List[float] = []
        new_vhm: List[float] = []
        new_params: List[float] = []
        for m_i, v_i, vhm_i, p, g in zip(
            self.m, self.v, self.v_hat_max, params, grads
        ):
            m_new = b1 * m_i + (1.0 - b1) * g
            v_new = b2 * v_i + (1.0 - b2) * g * g
            new_m.append(m_new)
            new_v.append(v_new)
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            vhm_new = max(vhm_i, v_hat)
            new_vhm.append(vhm_new)
            new_params.append(p - lr * m_hat / (math.sqrt(vhm_new) + eps))

        self.m = new_m
        self.v = new_v
        self.v_hat_max = new_vhm
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.v = None
        self.v_hat_max = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': list(self.m) if self.m is not None else None,
            'v': list(self.v) if self.v is not None else None,
            'v_hat_max': list(self.v_hat_max) if self.v_hat_max is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        m = state.get('m')
        self.m = list(m) if m is not None else None
        v = state.get('v')
        self.v = list(v) if v is not None else None
        vhm = state.get('v_hat_max')
        self.v_hat_max = list(vhm) if vhm is not None else None

    def __repr__(self) -> str:
        return (
            f"AMSGrad(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# AdamW
# ---------------------------------------------------------------------------

class AdamW(Optimizer):
    """AdamW: Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Key difference from Adam(weight_decay=λ): weight decay is applied
    DIRECTLY to parameters BEFORE the gradient moment update::

        θ ← θ * (1 - lr * weight_decay)
        then standard Adam update on gradient g

    This decoupling prevents weight decay from being adapted by the
    second moment (as in L2-regularized Adam), which the paper shows
    leads to worse generalization.

    Parameters
    ----------
    learning_rate:
        Step size (default 1e-3).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    weight_decay:
        Decoupled L2 penalty (default 0.01).
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t: int = 0
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        wd = self.weight_decay
        t = self.t

        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        new_m: List[float] = []
        new_v: List[float] = []
        new_params: List[float] = []
        for m_i, v_i, p, g in zip(self.m, self.v, params, grads):
            # Step 1: decoupled weight decay
            p_decayed = p * (1.0 - lr * wd)
            # Step 2: standard Adam update on raw gradient (no L2 in gradient)
            m_new = b1 * m_i + (1.0 - b1) * g
            v_new = b2 * v_i + (1.0 - b2) * g * g
            new_m.append(m_new)
            new_v.append(v_new)
            m_hat = m_new / bc1
            v_hat = v_new / bc2
            new_params.append(p_decayed - lr * m_hat / (math.sqrt(v_hat) + eps))

        self.m = new_m
        self.v = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.v = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'weight_decay': self.weight_decay,
            'm': list(self.m) if self.m is not None else None,
            'v': list(self.v) if self.v is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        m = state.get('m')
        self.m = list(m) if m is not None else None
        v = state.get('v')
        self.v = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"AdamW(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps}, "
            f"weight_decay={self.weight_decay})"
        )


# ---------------------------------------------------------------------------
# RAdam
# ---------------------------------------------------------------------------

class RAdam(Optimizer):
    """RAdam: Rectified Adam (Liu et al., 2020).

    Automatically handles warm-up by detecting when the variance of the
    adaptive learning rate is too large (early in training).

    Algorithm::

        rho_max = 2/(1 - beta2) - 1
        rho_t = rho_max - 2*t*beta2^t / (1 - beta2^t)
        If rho_t > 4:  (adaptive mode: variance tractable)
            r_t = sqrt((rho_t-4)*(rho_t-2)*rho_max /
                       ((rho_max-4)*(rho_max-2)*rho_t))
            theta -= lr * r_t * m_hat / (sqrt(v_hat) + eps)
        Else:  (SGD mode: use bias-corrected first moment only)
            theta -= lr * m_hat

    Parameters
    ----------
    learning_rate:
        Step size (default 1e-3).
    beta1:
        First moment decay (default 0.9).
    beta2:
        Second moment decay (default 0.999).
    eps:
        Numerical stability (default 1e-8).
    """

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t: int = 0
        self.m: Optional[List[float]] = None
        self.v: Optional[List[float]] = None
        # rho_max is a constant for this optimizer instance
        self._rho_max: float = 2.0 / (1.0 - beta2) - 1.0

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        self.t += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        eps = self.eps
        t = self.t
        rho_max = self._rho_max

        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        bc1 = 1.0 - b1 ** t
        bc2 = 1.0 - b2 ** t

        # Compute rho_t
        rho_t = rho_max - 2.0 * t * (b2 ** t) / bc2

        new_m: List[float] = []
        new_v: List[float] = []
        new_params: List[float] = []

        if rho_t > 4.0:
            # Adaptive mode: rectification term
            r_t = math.sqrt(
                (rho_t - 4.0) * (rho_t - 2.0) * rho_max
                / ((rho_max - 4.0) * (rho_max - 2.0) * rho_t)
            )
            for m_i, v_i, p, g in zip(self.m, self.v, params, grads):
                m_new = b1 * m_i + (1.0 - b1) * g
                v_new = b2 * v_i + (1.0 - b2) * g * g
                new_m.append(m_new)
                new_v.append(v_new)
                m_hat = m_new / bc1
                v_hat = v_new / bc2
                new_params.append(
                    p - lr * r_t * m_hat / (math.sqrt(v_hat) + eps)
                )
        else:
            # SGD mode: use bias-corrected first moment only
            for m_i, v_i, p, g in zip(self.m, self.v, params, grads):
                m_new = b1 * m_i + (1.0 - b1) * g
                v_new = b2 * v_i + (1.0 - b2) * g * g
                new_m.append(m_new)
                new_v.append(v_new)
                m_hat = m_new / bc1
                new_params.append(p - lr * m_hat)

        self.m = new_m
        self.v = new_v
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.t = 0
        self.m = None
        self.v = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            't': self.t,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'eps': self.eps,
            'm': list(self.m) if self.m is not None else None,
            'v': list(self.v) if self.v is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.t = state.get('t', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.eps = state.get('eps', self.eps)
        self._rho_max = 2.0 / (1.0 - self.beta2) - 1.0
        m = state.get('m')
        self.m = list(m) if m is not None else None
        v = state.get('v')
        self.v = list(v) if v is not None else None

    def __repr__(self) -> str:
        return (
            f"RAdam(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, eps={self.eps})"
        )


# ---------------------------------------------------------------------------
# Adadelta
# ---------------------------------------------------------------------------

class Adadelta(Optimizer):
    """Adadelta (Zeiler, 2012). No global learning rate needed.

    Maintains::

        E[g²]_t = rho * E[g²]_{t-1} + (1-rho) * g_t²
        RMS_g_t = sqrt(E[g²]_t + eps)
        delta_t = -(RMS_delta_{t-1} / RMS_g_t) * g_t
        E[delta²]_t = rho * E[delta²]_{t-1} + (1-rho) * delta_t²

    Update: theta <- theta + delta_t

    Note: E[delta²]_{t-1} uses the PREVIOUS step's accumulated delta
    (initialized to epsilon so RMS_delta is nonzero from the start).

    Parameters
    ----------
    rho:
        Decay rate for running averages (default 0.95).
    eps:
        Numerical stability (default 1e-6).
    """

    def __init__(self, rho: float = 0.95, eps: float = 1e-6) -> None:
        super().__init__()
        self.rho = rho
        self.eps = eps
        self.E_g2: Optional[List[float]] = None   # E[g²]
        self.E_d2: Optional[List[float]] = None   # E[delta²]

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        rho = self.rho
        eps = self.eps

        if self.E_g2 is None:
            # Initialize E[delta²] to eps so RMS_delta starts finite
            self.E_g2 = [0.0] * len(params)
            self.E_d2 = [eps] * len(params)

        new_E_g2: List[float] = []
        new_E_d2: List[float] = []
        new_params: List[float] = []

        for E_g2_i, E_d2_i, p, g in zip(self.E_g2, self.E_d2, params, grads):
            E_g2_new = rho * E_g2_i + (1.0 - rho) * g * g
            rms_g = math.sqrt(E_g2_new + eps)
            rms_d = math.sqrt(E_d2_i + eps)
            delta = -(rms_d / rms_g) * g
            E_d2_new = rho * E_d2_i + (1.0 - rho) * delta * delta
            new_E_g2.append(E_g2_new)
            new_E_d2.append(E_d2_new)
            new_params.append(p + delta)

        self.E_g2 = new_E_g2
        self.E_d2 = new_E_d2
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.E_g2 = None
        self.E_d2 = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'rho': self.rho,
            'eps': self.eps,
            'E_g2': list(self.E_g2) if self.E_g2 is not None else None,
            'E_d2': list(self.E_d2) if self.E_d2 is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.rho = state.get('rho', self.rho)
        self.eps = state.get('eps', self.eps)
        E_g2 = state.get('E_g2')
        self.E_g2 = list(E_g2) if E_g2 is not None else None
        E_d2 = state.get('E_d2')
        self.E_d2 = list(E_d2) if E_d2 is not None else None

    def __repr__(self) -> str:
        return f"Adadelta(rho={self.rho}, eps={self.eps})"


# ---------------------------------------------------------------------------
# Lookahead
# ---------------------------------------------------------------------------

class Lookahead(Optimizer):
    """Lookahead meta-optimizer (Zhang et al., 2019).

    Maintains "slow weights" phi. Every k steps::

        phi <- phi + alpha * (theta - phi)   (slow update)
        theta <- phi                          (fast weights reset to slow)

    Between slow updates: fast weights are updated by the inner optimizer.

    Parameters
    ----------
    optimizer:
        Any Optimizer instance (the "inner" / fast optimizer).
    k:
        Sync period (default 5).
    alpha:
        Slow weight interpolation rate (default 0.5).
    """

    def __init__(
        self,
        optimizer: 'Optimizer',
        k: int = 5,
        alpha: float = 0.5,
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.slow_weights: Optional[List[float]] = None
        self._step_count: int = 0

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1

        # Initialize slow weights on first call
        if self.slow_weights is None:
            self.slow_weights = list(params)

        # Delegate to inner optimizer
        fast_params = self.optimizer.step(params, grads)
        self._step_count += 1

        # Every k steps: sync slow/fast weights
        if self._step_count % self.k == 0:
            alpha = self.alpha
            new_slow: List[float] = []
            for phi, theta in zip(self.slow_weights, fast_params):
                phi_new = phi + alpha * (theta - phi)
                new_slow.append(phi_new)
            self.slow_weights = new_slow
            # Reset fast weights to slow weights
            fast_params = list(new_slow)

        return fast_params

    def reset(self) -> None:
        self.optimizer.reset()
        self.slow_weights = None
        self._step_count = 0
        self.iterations = 0

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'k': self.k,
            'alpha': self.alpha,
            '_step_count': self._step_count,
            'slow_weights': (
                list(self.slow_weights)
                if self.slow_weights is not None
                else None
            ),
            'inner_state': self.optimizer.get_state(),
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.k = state.get('k', self.k)
        self.alpha = state.get('alpha', self.alpha)
        self._step_count = state.get('_step_count', 0)
        sw = state.get('slow_weights')
        self.slow_weights = list(sw) if sw is not None else None
        inner = state.get('inner_state')
        if inner is not None:
            self.optimizer.load_state(inner)

    def __repr__(self) -> str:
        return (
            f"Lookahead(optimizer={self.optimizer!r}, k={self.k}, "
            f"alpha={self.alpha})"
        )


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------

class Lion(Optimizer):
    """Lion optimizer: EvoLved Sign Momentum (Chen et al., 2023).

    Memory-efficient: only tracks first moment (no second moment).
    Uses SIGN of interpolated gradient — all updates are ±lr.

    Algorithm::

        update = sign(beta1 * m_{t-1} + (1-beta1) * g_t)
        m_t = beta2 * m_{t-1} + (1-beta2) * g_t   (AFTER computing update)
        theta <- theta - lr * (update + weight_decay * theta)

    Note: m is updated AFTER computing the update signal.
    sign(0) = 0 (zero gradient produces zero update).

    Parameters
    ----------
    learning_rate:
        Step size (default 1e-4; Lion uses smaller LR than Adam).
    beta1:
        Coefficient for update computation (default 0.9).
    beta2:
        Coefficient for momentum update (default 0.99).
    weight_decay:
        L2 regularisation applied to parameters (default 0.0).
    """

    def __init__(
        self,
        learning_rate: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.m: Optional[List[float]] = None  # first moment

    def step(self, params: List[float], grads: List[float]) -> List[float]:
        if any(math.isnan(g) or math.isinf(g) for g in grads):
            raise ValueError("NaN or Inf detected in gradients")
        self.iterations += 1
        lr = self.learning_rate
        b1 = self.beta1
        b2 = self.beta2
        wd = self.weight_decay

        if self.m is None:
            self.m = [0.0] * len(params)

        new_m: List[float] = []
        new_params: List[float] = []
        for m_i, p, g in zip(self.m, params, grads):
            # Compute update signal using CURRENT moment (before update)
            interp = b1 * m_i + (1.0 - b1) * g
            update = _sign(interp)
            # Update moment AFTER computing update signal
            m_new = b2 * m_i + (1.0 - b2) * g
            new_m.append(m_new)
            # Parameter update
            new_params.append(p - lr * (update + wd * p))

        self.m = new_m
        return new_params

    def reset(self) -> None:
        self.iterations = 0
        self.m = None

    def get_state(self) -> dict:
        return {
            'iterations': self.iterations,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'weight_decay': self.weight_decay,
            'm': list(self.m) if self.m is not None else None,
        }

    def load_state(self, state: dict) -> None:
        self.iterations = state.get('iterations', 0)
        self.learning_rate = state.get('learning_rate', self.learning_rate)
        self.beta1 = state.get('beta1', self.beta1)
        self.beta2 = state.get('beta2', self.beta2)
        self.weight_decay = state.get('weight_decay', self.weight_decay)
        m = state.get('m')
        self.m = list(m) if m is not None else None

    def __repr__(self) -> str:
        return (
            f"Lion(lr={self.learning_rate}, beta1={self.beta1}, "
            f"beta2={self.beta2}, weight_decay={self.weight_decay})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: List[str] = [
    # Gradient clipping
    "clip_gradients",
    "clip_gradients_value",
    # Base class
    "Optimizer",
    # Original optimizers
    "SGD",
    "Momentum",
    "NesterovMomentum",
    "Adagrad",
    "RMSprop",
    "Adam",
    "AdaMax",
    "NAdam",
    "AMSGrad",
    # New optimizers
    "AdamW",
    "RAdam",
    "Adadelta",
    "Lookahead",
    "Lion",
]

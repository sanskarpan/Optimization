"""
Learning Rate Schedules
========================

Learning rate scheduling strategies for optimization.

Applications in ML/DL:
- Preventing oscillations in late training
- Escaping local minima
- Fine-tuning near convergence
- Warm restarts for better generalization
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

__all__ = [
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
]


class LRSchedule(ABC):
    """Base class for learning rate schedules."""

    def __init__(self, initial_lr: float) -> None:
        """
        Args:
            initial_lr: Initial learning rate
        """
        self.initial_lr = initial_lr
        self.current_step = 0

    @abstractmethod
    def get_lr(self, step: Optional[int] = None) -> float:
        """
        Get learning rate for current step.

        Args:
            step: Training step (optional, uses internal counter if not provided)

        Returns:
            Current learning rate
        """
        return None

    def step(self) -> float:
        """
        Increment step counter and return new learning rate.

        Returns:
            Learning rate for next step
        """
        self.current_step += 1
        return self.get_lr(self.current_step)

    def reset(self) -> None:
        """Reset schedule to initial state."""
        self.current_step = 0

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(initial_lr={self.initial_lr})"


class ConstantLR(LRSchedule):
    """
    Constant learning rate (no decay).

    lr(t) = lr₀

    Args:
        initial_lr: Learning rate (constant)
    """

    def __init__(self, initial_lr: float = 0.01) -> None:
        super().__init__(initial_lr)

    def get_lr(self, step: Optional[int] = None) -> float:
        """Return constant learning rate."""
        return self.initial_lr

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"ConstantLR(initial_lr={self.initial_lr})"


class StepDecayLR(LRSchedule):
    """
    Step decay learning rate schedule.

    lr(t) = lr₀ * γ^⌊t/step_size⌋

    Reduces learning rate by factor γ every step_size epochs.

    Args:
        initial_lr: Initial learning rate
        step_size: Number of epochs between decays
        gamma: Multiplicative factor (default 0.1)

    Example:
        >>> lr_schedule = StepDecayLR(initial_lr=0.1, step_size=10, gamma=0.1)
        >>> lr_schedule.get_lr(0)   # 0.1
        >>> lr_schedule.get_lr(10)  # 0.01
        >>> lr_schedule.get_lr(20)  # 0.001
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        step_size: int = 10,
        gamma: float = 0.1,
    ) -> None:
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step
        decay_steps = step // self.step_size
        return self.initial_lr * (self.gamma ** decay_steps)

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"StepDecayLR(initial_lr={self.initial_lr}, "
            f"step_size={self.step_size}, gamma={self.gamma})"
        )


class ExponentialDecayLR(LRSchedule):
    """
    Exponential decay learning rate schedule.

    lr(t) = lr₀ * e^(-λt)

    or

    lr(t) = lr₀ * γ^t

    Args:
        initial_lr: Initial learning rate
        decay_rate: Decay rate λ (for exponential) or γ (for geometric)
        decay_type: 'exponential' or 'geometric'

    Example:
        >>> lr_schedule = ExponentialDecayLR(initial_lr=0.1, decay_rate=0.95)
        >>> lr_schedule.get_lr(10)  # 0.1 * 0.95^10
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        decay_rate: float = 0.96,
        decay_type: str = 'geometric',
    ) -> None:
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_type = decay_type

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step
        if self.decay_type == 'geometric':
            return self.initial_lr * (self.decay_rate ** step)
        else:
            return self.initial_lr * math.exp(-self.decay_rate * step)

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"ExponentialDecayLR(initial_lr={self.initial_lr}, "
            f"decay_rate={self.decay_rate}, decay_type={self.decay_type!r})"
        )


class CosineAnnealingLR(LRSchedule):
    """
    Cosine annealing learning rate schedule.

    lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(πt/T)) / 2

    Smoothly decreases learning rate following cosine curve.
    Popular in modern deep learning.

    Args:
        initial_lr: Maximum learning rate
        T_max: Maximum number of iterations
        eta_min: Minimum learning rate

    Example:
        >>> lr_schedule = CosineAnnealingLR(initial_lr=0.1, T_max=100)
        >>> lr_schedule.get_lr(0)    # 0.1 (maximum)
        >>> lr_schedule.get_lr(50)   # ~0.05 (middle)
        >>> lr_schedule.get_lr(100)  # ~0.0 (minimum)
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        T_max: int = 100,
        eta_min: float = 0.0,
    ) -> None:
        if T_max <= 0:
            raise ValueError(
                f"CosineAnnealingLR: T_max must be > 0 (got {T_max})"
            )
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """
        Get learning rate at given step.

        Note: steps beyond T_max are clamped to T_max, so eta_min is
        returned for all steps > T_max (the schedule does not repeat).
        """
        if step is None:
            step = self.current_step
        step = min(step, self.T_max)
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * step / self.T_max)
        ) / 2
        return lr

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"CosineAnnealingLR(initial_lr={self.initial_lr}, "
            f"T_max={self.T_max}, eta_min={self.eta_min})"
        )


class WarmRestartLR(LRSchedule):
    """
    Cosine annealing with warm restarts (SGDR).

    Periodically restarts learning rate to escape local minima.

    lr(t) = lr_min + (lr_max - lr_min) * (1 + cos(πt_i/T_i)) / 2

    where t_i is iterations since last restart, T_i is period length.

    Args:
        initial_lr: Initial/maximum learning rate
        T_0: Initial restart period
        T_mult: Period multiplier after each restart
        eta_min: Minimum learning rate

    Example:
        >>> # Restart every 10, 20, 40, 80, ... steps
        >>> lr_schedule = WarmRestartLR(initial_lr=0.1, T_0=10, T_mult=2)
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 0.0,
    ) -> None:
        if T_0 <= 0:
            raise ValueError(f"WarmRestartLR: T_0 must be > 0, got {T_0}")
        if T_mult <= 0:
            raise ValueError(f"WarmRestartLR: T_mult must be > 0, got {T_mult}")
        super().__init__(initial_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step
        T_cur = step
        T_i = self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * T_cur / T_i)
        ) / 2
        return lr

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"WarmRestartLR(initial_lr={self.initial_lr}, "
            f"T_0={self.T_0}, T_mult={self.T_mult}, eta_min={self.eta_min})"
        )


class PolynomialDecayLR(LRSchedule):
    """
    Polynomial decay learning rate schedule.

    lr(t) = (lr₀ - lr_end) * (1 - t/T)^power + lr_end

    Args:
        initial_lr: Initial learning rate
        total_steps: Total number of training steps
        end_lr: Final learning rate
        power: Polynomial power (1 = linear decay)

    Example:
        >>> # Linear decay from 0.1 to 0.0 over 100 steps
        >>> lr_schedule = PolynomialDecayLR(initial_lr=0.1, total_steps=100, power=1.0)
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        total_steps: int = 1000,
        end_lr: float = 0.0,
        power: float = 1.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(
                f"PolynomialDecayLR: total_steps must be > 0 (got {total_steps})"
            )
        super().__init__(initial_lr)
        self.total_steps = total_steps
        self.end_lr = end_lr
        self.power = power

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step
        if step >= self.total_steps:
            return self.end_lr
        decay_factor = (1 - step / self.total_steps) ** self.power
        lr = (self.initial_lr - self.end_lr) * decay_factor + self.end_lr
        return lr

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"PolynomialDecayLR(initial_lr={self.initial_lr}, "
            f"total_steps={self.total_steps}, end_lr={self.end_lr}, power={self.power})"
        )


class OneCycleLR(LRSchedule):
    """
    One-cycle learning rate policy.

    Increases learning rate from low to high, then decreases.
    Popularized by fast.ai and super-convergence paper.

    Phase 1 (0 to pct_start): Linear increase to max_lr
    Phase 2 (pct_start to 1.0): Cosine annealing to min_lr

    Args:
        max_lr: Maximum learning rate
        total_steps: Total training steps
        pct_start: Percentage of training for warmup phase
        div_factor: Initial lr = max_lr / div_factor
        final_div_factor: Final lr = max_lr / final_div_factor

    Example:
        >>> # Warmup for 30% of training, then anneal
        >>> lr_schedule = OneCycleLR(max_lr=0.1, total_steps=100, pct_start=0.3)
    """

    def __init__(
        self,
        max_lr: float = 0.1,
        total_steps: int = 1000,
        pct_start: float = 0.3,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(
                f"OneCycleLR: total_steps must be > 0 (got {total_steps})"
            )
        if not (0.0 < pct_start < 1.0):
            raise ValueError(
                f"OneCycleLR: pct_start must be in (0, 1) (got {pct_start})"
            )
        initial_lr = max_lr / div_factor
        super().__init__(initial_lr)
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.min_lr_start = initial_lr
        self.min_lr_end = max_lr / final_div_factor

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step
        progress = step / self.total_steps
        if progress < self.pct_start:
            lr = self.min_lr_start + (self.max_lr - self.min_lr_start) * (
                progress / self.pct_start
            )
        else:
            progress_phase2 = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.min_lr_end + (self.max_lr - self.min_lr_end) * (
                1 + math.cos(math.pi * progress_phase2)
            ) / 2
        return lr

    def get_state(self) -> dict:
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"OneCycleLR(max_lr={self.max_lr}, "
            f"total_steps={self.total_steps}, pct_start={self.pct_start})"
        )


class ReduceLROnPlateau:
    """
    Reduce learning rate when metric has stopped improving.

    Monitors a metric (e.g., validation loss) and reduces learning rate
    when no improvement is observed for a patience period.

    Args:
        initial_lr: Initial learning rate
        mode: 'min' for loss, 'max' for accuracy
        factor: Factor to reduce learning rate
        patience: Number of epochs with no improvement before reducing
        threshold: Minimum change to qualify as improvement (absolute delta,
            not relative; improvement requires metric < best - threshold for
            mode='min', or metric > best + threshold for mode='max')
        min_lr: Minimum learning rate (floor; lr is never reduced below this)

    Example:
        >>> scheduler = ReduceLROnPlateau(initial_lr=0.1, patience=5)
        >>> for epoch in range(100):
        ...     val_loss = train_epoch()
        ...     new_lr = scheduler.step(val_loss)
    """

    def __init__(
        self,
        initial_lr: float = 0.1,
        mode: str = 'min',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
    ) -> None:
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.min_lr = min_lr
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.num_bad_epochs = 0

    def step(self, metric: float) -> float:
        """
        Update learning rate based on metric.

        Args:
            metric: Current metric value (e.g., validation loss)

        Returns:
            New learning rate
        """
        if self.mode == 'min':
            improved = metric < self.best_metric - self.threshold
        else:
            improved = metric > self.best_metric + self.threshold

        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self) -> None:
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.num_bad_epochs = 0

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"ReduceLROnPlateau(initial_lr={self.initial_lr}, "
            f"mode={self.mode!r}, factor={self.factor}, patience={self.patience})"
        )


# ---------------------------------------------------------------------------
# New schedule classes
# ---------------------------------------------------------------------------


class LinearWarmupLR(LRSchedule):
    """
    Linear warmup learning rate schedule.

    During warmup (step < warmup_steps):
        lr(t) = start_lr + (initial_lr - start_lr) * t / warmup_steps

    After warmup (step >= warmup_steps):
        lr(t) = initial_lr  (constant)

    Args:
        initial_lr: target learning rate after warmup
        warmup_steps: number of steps to ramp up (must be > 0)
        start_lr: starting learning rate (default 0.0)

    Example:
        >>> sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=100)
        >>> sched.get_lr(0)    # 0.0
        >>> sched.get_lr(50)   # 0.005
        >>> sched.get_lr(100)  # 0.01
        >>> sched.get_lr(200)  # 0.01
    """

    def __init__(
        self,
        initial_lr: float,
        warmup_steps: int,
        start_lr: float = 0.0,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError(
                f"LinearWarmupLR: warmup_steps must be > 0, got {warmup_steps}"
            )
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.start_lr = start_lr
        self._step = 0

    def get_lr(self, step: Optional[int] = None) -> float:
        t = step if step is not None else self._step
        if t >= self.warmup_steps:
            return self.initial_lr
        return self.start_lr + (self.initial_lr - self.start_lr) * t / self.warmup_steps

    def step(self) -> float:
        lr = self.get_lr(self._step)
        self._step += 1
        return lr

    def reset(self) -> None:
        self._step = 0

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"LinearWarmupLR(initial_lr={self.initial_lr}, "
            f"warmup_steps={self.warmup_steps}, start_lr={self.start_lr})"
        )


class CyclicLR(LRSchedule):
    """
    Cyclical Learning Rate schedule (Smith, 2017).

    Cycles learning rate between base_lr and max_lr.
    Cycle length = 2 * step_size steps.

    Modes:
        'triangular': constant amplitude triangle wave
            cycle = floor(1 + step / (2*step_size))
            x = |step/step_size - 2*cycle + 1|
            lr = base_lr + (max_lr - base_lr) * max(0, 1-x)

        'triangular2': amplitude halves each cycle
            lr = base_lr + (max_lr - base_lr) * max(0, 1-x) / (2^(cycle-1))

        'exp_range': exponential decay of amplitude
            lr = base_lr + (max_lr - base_lr) * max(0, 1-x) * gamma^step

    Args:
        base_lr: minimum learning rate
        max_lr: maximum learning rate
        step_size: half-cycle length in steps (default 2000)
        mode: 'triangular', 'triangular2', or 'exp_range' (default 'triangular')
        gamma: decay factor for 'exp_range' mode (default 1.0)

    Example:
        >>> sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        >>> [sched.step() for _ in range(9)]
        # Should cycle from 0.001 up to 0.01 and back
    """

    def __init__(
        self,
        base_lr: float,
        max_lr: float,
        step_size: int = 2000,
        mode: str = 'triangular',
        gamma: float = 1.0,
    ) -> None:
        if mode not in ('triangular', 'triangular2', 'exp_range'):
            raise ValueError(f"CyclicLR: unknown mode '{mode}'")
        if step_size <= 0:
            raise ValueError(f"CyclicLR: step_size must be > 0, got {step_size}")
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self._step = 0
        # Set initial_lr for ABC compatibility
        self.initial_lr = base_lr

    def get_lr(self, step: Optional[int] = None) -> float:
        t = step if step is not None else self._step
        cycle = math.floor(1 + t / (2 * self.step_size))
        x = abs(t / self.step_size - 2 * cycle + 1)
        scale = max(0.0, 1.0 - x)

        if self.mode == 'triangular':
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale
        elif self.mode == 'triangular2':
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale / (2 ** (cycle - 1))
        else:  # exp_range
            lr = self.base_lr + (self.max_lr - self.base_lr) * scale * (self.gamma ** t)
        return lr

    def step(self) -> float:
        lr = self.get_lr(self._step)
        self._step += 1
        return lr

    def reset(self) -> None:
        self._step = 0

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"CyclicLR(base_lr={self.base_lr}, max_lr={self.max_lr}, "
            f"step_size={self.step_size}, mode={self.mode!r})"
        )


class NoamLR(LRSchedule):
    """
    Noam learning rate schedule from "Attention Is All You Need" (Vaswani et al., 2017).

    lr(t) = scale * d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))

    Increases linearly for t < warmup_steps, then decreases proportionally to 1/sqrt(t).
    Peaks at step t = warmup_steps.

    Args:
        d_model: model dimension (e.g., 512)
        warmup_steps: warmup period (default 4000)
        scale: global scale factor (default 1.0)

    Note: step=0 is treated as step=1 to avoid division by zero.

    Example:
        >>> sched = NoamLR(d_model=512, warmup_steps=4000)
        >>> sched.get_lr(4000)  # peak learning rate ≈ 0.00138
    """

    def __init__(
        self,
        d_model: int,
        warmup_steps: int = 4000,
        scale: float = 1.0,
    ) -> None:
        if d_model <= 0:
            raise ValueError(f"NoamLR: d_model must be > 0, got {d_model}")
        if warmup_steps <= 0:
            raise ValueError(f"NoamLR: warmup_steps must be > 0, got {warmup_steps}")
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.scale = scale
        self._step = 0
        # Set initial_lr for repr/compatibility
        self.initial_lr = scale * (d_model ** (-0.5)) * (warmup_steps ** (-0.5))

    def get_lr(self, step: Optional[int] = None) -> float:
        t = step if step is not None else self._step
        t = max(t, 1)  # avoid division by zero
        arg1 = t ** (-0.5)
        arg2 = t * (self.warmup_steps ** (-1.5))
        return self.scale * (self.d_model ** (-0.5)) * min(arg1, arg2)

    def step(self) -> float:
        lr = self.get_lr(self._step)
        self._step += 1
        return lr

    def reset(self) -> None:
        self._step = 0

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return (
            f"NoamLR(d_model={self.d_model}, "
            f"warmup_steps={self.warmup_steps}, scale={self.scale})"
        )


class ComposedLR(LRSchedule):
    """
    Chain multiple LR schedules sequentially.

    Constructor: ComposedLR([(sched1, n_steps1), (sched2, n_steps2), ...])

    For step t:
        - Determine which segment it falls in (accumulate n_steps)
        - Compute local step offset within that segment
        - Delegate to that segment's schedule with local step

    The last segment runs indefinitely (if step exceeds total steps,
    stays in last segment with offset continuing to grow).

    Example:
        >>> warmup = LinearWarmupLR(initial_lr=0.01, warmup_steps=100)
        >>> cosine = CosineAnnealingLR(initial_lr=0.01, T_max=900)
        >>> sched = ComposedLR([(warmup, 100), (cosine, 900)])
        >>> sched.get_lr(50)    # uses warmup schedule, local step 50
        >>> sched.get_lr(150)   # uses cosine schedule, local step 50

    Args:
        segments: list of (LRSchedule, n_steps) tuples
    """

    def __init__(self, segments: List[Tuple['LRSchedule', int]]) -> None:
        if not segments:
            raise ValueError("ComposedLR: segments list must not be empty")
        self.segments = segments
        self._step = 0
        # Set initial_lr for repr/compatibility
        self.initial_lr = segments[0][0].get_lr(0)

    def get_lr(self, step: Optional[int] = None) -> float:
        t = step if step is not None else self._step
        cumulative = 0
        for sched, n in self.segments[:-1]:
            if t < cumulative + n:
                local_step = t - cumulative
                return sched.get_lr(local_step)
            cumulative += n
        # Last segment
        local_step = t - cumulative
        return self.segments[-1][0].get_lr(local_step)

    def step(self) -> float:
        lr = self.get_lr(self._step)
        self._step += 1
        return lr

    def reset(self) -> None:
        self._step = 0
        for sched, _ in self.segments:
            sched.reset()

    def get_state(self) -> dict:
        """Return a copy of the scheduler's current state."""
        return self.__dict__.copy()

    def load_state(self, state: dict) -> None:
        """Restore scheduler state from a dict."""
        self.__dict__.update(state)

    def __repr__(self) -> str:
        seg_repr = ", ".join(
            f"({sched!r}, {n})" for sched, n in self.segments
        )
        return f"ComposedLR(segments=[{seg_repr}])"

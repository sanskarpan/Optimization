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
from typing import Optional
from abc import ABC, abstractmethod


class LRSchedule(ABC):
    """Base class for learning rate schedules."""

    def __init__(self, initial_lr: float):
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
        pass

    def step(self) -> float:
        """
        Increment step counter and return new learning rate.

        Returns:
            Learning rate for next step
        """
        self.current_step += 1
        return self.get_lr(self.current_step)

    def reset(self):
        """Reset schedule to initial state."""
        self.current_step = 0


class ConstantLR(LRSchedule):
    """
    Constant learning rate (no decay).

    lr(t) = lr₀

    Args:
        initial_lr: Learning rate (constant)
    """

    def __init__(self, initial_lr: float = 0.01):
        super().__init__(initial_lr)

    def get_lr(self, step: Optional[int] = None) -> float:
        """Return constant learning rate."""
        return self.initial_lr


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
        gamma: float = 0.1
    ):
        super().__init__(initial_lr)
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step

        # Number of complete decay periods
        decay_steps = step // self.step_size

        return self.initial_lr * (self.gamma ** decay_steps)


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
        decay_type: str = 'geometric'
    ):
        super().__init__(initial_lr)
        self.decay_rate = decay_rate
        self.decay_type = decay_type

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step

        if self.decay_type == 'geometric':
            # lr = lr₀ * γ^t
            return self.initial_lr * (self.decay_rate ** step)
        else:  # exponential
            # lr = lr₀ * e^(-λt)
            return self.initial_lr * math.exp(-self.decay_rate * step)


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
        eta_min: float = 0.0
    ):
        if T_max <= 0:
            raise ValueError(f"CosineAnnealingLR: T_max must be > 0 (got {T_max})")
        super().__init__(initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step.

        Note: steps beyond T_max are clamped to T_max, so eta_min is
        returned for all steps > T_max (the schedule does not repeat).
        """
        if step is None:
            step = self.current_step

        # Clamp step to [0, T_max] — holds eta_min for all steps beyond T_max
        step = min(step, self.T_max)

        # Cosine annealing formula
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * step / self.T_max)
        ) / 2

        return lr


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
        eta_min: float = 0.0
    ):
        if T_0 <= 0:
            raise ValueError(
                f"WarmRestartLR: T_0 must be > 0, got {T_0}"
            )
        if T_mult <= 0:
            raise ValueError(
                f"WarmRestartLR: T_mult must be > 0, got {T_mult}"
            )
        super().__init__(initial_lr)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min

    def get_lr(self, step: Optional[int] = None) -> float:
        """Get learning rate at given step."""
        if step is None:
            step = self.current_step

        # Determine position within current period
        T_cur = step
        T_i = self.T_0

        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult

        # Cosine annealing within period
        lr = self.eta_min + (self.initial_lr - self.eta_min) * (
            1 + math.cos(math.pi * T_cur / T_i)
        ) / 2

        return lr


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
        power: float = 1.0
    ):
        if total_steps <= 0:
            raise ValueError(f"PolynomialDecayLR: total_steps must be > 0 (got {total_steps})")
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

        # Polynomial decay
        decay_factor = (1 - step / self.total_steps) ** self.power
        lr = (self.initial_lr - self.end_lr) * decay_factor + self.end_lr

        return lr


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
        final_div_factor: float = 1e4
    ):
        if total_steps <= 0:
            raise ValueError(f"OneCycleLR: total_steps must be > 0 (got {total_steps})")
        if not (0.0 < pct_start < 1.0):
            raise ValueError(f"OneCycleLR: pct_start must be in (0, 1) (got {pct_start})")
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

        # Normalize step to [0, 1]
        progress = step / self.total_steps

        if progress < self.pct_start:
            # Phase 1: Linear warmup
            lr = self.min_lr_start + (self.max_lr - self.min_lr_start) * (
                progress / self.pct_start
            )
        else:
            # Phase 2: Cosine annealing
            progress_phase2 = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.min_lr_end + (self.max_lr - self.min_lr_end) * (
                1 + math.cos(math.pi * progress_phase2)
            ) / 2

        return lr


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
        min_lr: float = 0.0
    ):
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
        # Check if improved
        if self.mode == 'min':
            improved = metric < self.best_metric - self.threshold
        else:  # max
            improved = metric > self.best_metric + self.threshold

        if improved:
            self.best_metric = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if patience exceeded
        if self.num_bad_epochs >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.num_bad_epochs = 0

        return self.current_lr

    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

    def reset(self):
        """Reset scheduler to initial state."""
        self.current_lr = self.initial_lr
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.num_bad_epochs = 0

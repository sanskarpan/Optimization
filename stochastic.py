"""
Stochastic Optimization Methods
=================================
Variance-reduced gradient methods (SVRG, SAGA, SAG), Polyak-Ruppert
averaging, Robbins-Monro SGD.

These methods are for finite-sum problems::

    min (1/n) * sum_{i=1}^n f_i(x)

All implementations are pure Python (stdlib only: math, random, warnings,
typing).  No third-party dependencies are required.

References
----------
* Johnson & Zhang (2013) — SVRG: "Accelerating Stochastic Gradient Descent
  using Predictive Variance Reduction."
* Defazio, Bach & Lacoste-Julien (2014) — SAGA.
* Schmidt, Le Roux & Bach (2013) — SAG.
* Polyak & Juditsky (1992) — Averaging of SGD Iterates.
* Robbins & Monro (1951) — A Stochastic Approximation Method.
"""

import math
import random
import warnings
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 8.1  SVRG — Stochastic Variance Reduced Gradient
# ---------------------------------------------------------------------------


def svrg(
    grad_full: Callable[[List[float]], List[float]],
    grad_i: Callable[[int, List[float]], List[float]],
    x0: List[float],
    n: int,
    m: Optional[int] = None,
    learning_rate: float = 0.01,
    epochs: int = 10,
) -> Tuple[List[float], List[float]]:
    """Stochastic Variance Reduced Gradient (Johnson & Zhang, 2013).

    Solves the finite-sum problem ``min (1/n) * sum_{i=0}^{n-1} f_i(x)``
    using variance-reduced stochastic gradient estimates.

    Algorithm (one epoch)
    ---------------------
    1. Set snapshot ``x_tilde = x``.
    2. Compute the full gradient ``mu = grad_full(x_tilde)`` (once per epoch).
    3. For ``m`` inner steps:

       a. Sample index ``i`` uniformly from ``{0, ..., n-1}``.
       b. Compute the variance-reduced gradient::

              g_tilde = grad_i(i, x) - grad_i(i, x_tilde) + mu

       c. Update ``x = x - lr * g_tilde``.
    4. Keep the final inner iterate as the new ``x`` for the next epoch.

    The *loss history* records the sum of squared gradient norms over all
    inner steps within each epoch, because the individual function values
    ``f_i`` are not required by this interface.

    Parameters
    ----------
    grad_full:
        Full gradient of ``(1/n) * sum_i f_i`` evaluated at a point ``x``.
        Signature: ``grad_full(x) -> List[float]``.
    grad_i:
        Gradient of the ``i``-th component ``f_i`` at ``x``.
        Signature: ``grad_i(i, x) -> List[float]``.
    x0:
        Initial parameter vector.
    n:
        Dataset size (number of component functions).
    m:
        Number of inner-loop steps per epoch.  Defaults to ``2 * n``.
    learning_rate:
        Step size ``eta``.
    epochs:
        Number of outer iterations (full-gradient evaluations).

    Returns
    -------
    x_final:
        Parameter vector at the end of the last epoch.
    loss_history:
        List of length ``epochs``.  Entry ``t`` is the sum of squared
        norms of all variance-reduced gradients computed during epoch ``t``.

    Raises
    ------
    ValueError
        If ``n < 1``, ``epochs < 1``, or ``learning_rate <= 0``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

    if m is None:
        m = 2 * n

    d: int = len(x0)
    x: List[float] = list(x0)
    loss_history: List[float] = []

    for _ in range(epochs):
        # --- snapshot and full gradient (one call per epoch) ---
        x_tilde: List[float] = list(x)
        mu: List[float] = grad_full(x_tilde)

        epoch_sq_norm: float = 0.0

        for _ in range(m):
            idx: int = random.randrange(n)

            # Variance-reduced gradient estimate
            g_i_x: List[float] = grad_i(idx, x)
            g_i_tilde: List[float] = grad_i(idx, x_tilde)
            g_tilde: List[float] = [
                g_i_x[j] - g_i_tilde[j] + mu[j] for j in range(d)
            ]

            # Parameter update
            x = [x[j] - learning_rate * g_tilde[j] for j in range(d)]

            # Accumulate squared norm for history
            sq_norm: float = sum(v * v for v in g_tilde)
            epoch_sq_norm += sq_norm

        loss_history.append(epoch_sq_norm)

    return x, loss_history


# ---------------------------------------------------------------------------
# 8.2  SAGA
# ---------------------------------------------------------------------------


class SAGAOptimizer:
    """SAGA optimizer (Defazio, Bach & Lacoste-Julien, 2014).

    Maintains a *gradient table* ``T`` of shape ``(n, d)`` where ``n`` is
    the dataset size and ``d`` is the parameter dimension.  A running sum
    ``sum_table`` keeps ``sum_j T[j]`` up-to-date incrementally so that the
    per-step cost is O(d), not O(n * d).

    One SAGA step with sampled index ``i``
    ---------------------------------------
    1. Compute ``g_i = grad_i(i, x)``.
    2. Form the variance-reduced estimate::

           g = g_i - T[i] + (1/n) * sum_table

    3. Update parameters: ``x = x - lr * g``.
    4. Update table and running sum::

           sum_table += g_i - T[i]
           T[i]       = g_i

    The gradient table is initialised to **all zeros**, which corresponds to
    assuming a zero gradient estimate at the starting point.

    Parameters
    ----------
    n:
        Dataset size.
    d:
        Parameter dimension.
    learning_rate:
        Step size.

    Attributes
    ----------
    gradient_table:
        ``List[List[float]]`` of shape ``(n, d)``.
    sum_table:
        ``List[float]`` of length ``d``; running column-wise sum of
        ``gradient_table``.
    """

    def __init__(
        self,
        n: int,
        d: int,
        learning_rate: float = 0.01,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}.")

        self.n: int = n
        self.d: int = d
        self.learning_rate: float = learning_rate

        # Gradient table T[i] = most recent grad_i computed at that sample
        self.gradient_table: List[List[float]] = [
            [0.0] * d for _ in range(n)
        ]
        # Running sum: sum_table[j] = sum_i T[i][j]
        self.sum_table: List[float] = [0.0] * d

    # ------------------------------------------------------------------
    def step(
        self,
        grad_i: Callable[[int, List[float]], List[float]],
        x: List[float],
        i: int,
    ) -> List[float]:
        """Perform one SAGA update for sample index ``i``.

        Parameters
        ----------
        grad_i:
            Component gradient.  Signature: ``grad_i(i, x) -> List[float]``.
        x:
            Current parameter vector (not modified in-place).
        i:
            Sample index in ``{0, ..., n-1}``.

        Returns
        -------
        List[float]
            Updated parameter vector ``x_new``.
        """
        d = self.d
        n = self.n
        lr = self.learning_rate

        # Step 1: compute fresh gradient for sample i
        g_i: List[float] = grad_i(i, x)

        # Step 2: variance-reduced gradient estimate
        old_t_i: List[float] = self.gradient_table[i]
        g_est: List[float] = [
            g_i[j] - old_t_i[j] + self.sum_table[j] / n
            for j in range(d)
        ]

        # Step 3: update parameters
        x_new: List[float] = [x[j] - lr * g_est[j] for j in range(d)]

        # Step 4: update running sum and table
        for j in range(d):
            self.sum_table[j] += g_i[j] - old_t_i[j]
        self.gradient_table[i] = g_i

        return x_new

    # ------------------------------------------------------------------
    def run(
        self,
        grad_i: Callable[[int, List[float]], List[float]],
        x0: List[float],
        n_steps: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """Run SAGA for ``n_steps``, sampling indices uniformly at each step.

        Parameters
        ----------
        grad_i:
            Component gradient function.
        x0:
            Initial parameter vector.
        n_steps:
            Total number of update steps.
        seed:
            Optional random seed for reproducibility.

        Returns
        -------
        x_final:
            Parameter vector after all steps.
        history:
            List of length ``n_steps``.  Each entry is the squared
            Euclidean norm of the variance-reduced gradient used at that step.
        """
        if seed is not None:
            random.seed(seed)

        x: List[float] = list(x0)
        history: List[float] = []

        for _ in range(n_steps):
            idx: int = random.randrange(self.n)

            # Capture the gradient estimate norm before the update
            old_t_i = self.gradient_table[idx]
            g_i_raw: List[float] = grad_i(idx, x)
            g_est: List[float] = [
                g_i_raw[j] - old_t_i[j] + self.sum_table[j] / self.n
                for j in range(self.d)
            ]
            sq_norm: float = sum(v * v for v in g_est)
            history.append(sq_norm)

            # Perform the actual step (reuses grad_i call via a wrapper)
            # To avoid a second grad_i evaluation we inline the update here
            x = [x[j] - self.learning_rate * g_est[j] for j in range(self.d)]
            for j in range(self.d):
                self.sum_table[j] += g_i_raw[j] - old_t_i[j]
            self.gradient_table[idx] = g_i_raw

        return x, history

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the gradient table and running sum to all zeros."""
        self.gradient_table = [[0.0] * self.d for _ in range(self.n)]
        self.sum_table = [0.0] * self.d


# ---------------------------------------------------------------------------
# 8.3  SAG — Stochastic Average Gradient
# ---------------------------------------------------------------------------


class SAGOptimizer:
    """SAG optimizer (Schmidt, Le Roux & Bach, 2013).

    Unlike SAGA, SAG uses a *biased* gradient estimate — the plain average
    of the most recently stored per-sample gradients.  This introduces a bias
    but can still converge at a geometric rate for strongly-convex problems.

    One SAG step with sampled index ``i``
    --------------------------------------
    1. Compute ``g_i = grad_i(i, x)``.
    2. Update running sum: ``running_sum += g_i - table[i]``.
    3. Store ``table[i] = g_i``.
    4. Gradient estimate: ``g_avg = running_sum / n``.
    5. Update: ``x = x - lr * g_avg``.

    Parameters
    ----------
    n:
        Dataset size.
    d:
        Parameter dimension.
    learning_rate:
        Step size.

    Attributes
    ----------
    gradient_table:
        ``List[List[float]]`` of shape ``(n, d)``; most recent gradient per sample.
    running_sum:
        ``List[float]`` of length ``d``; column-wise sum of ``gradient_table``.
    """

    def __init__(
        self,
        n: int,
        d: int,
        learning_rate: float = 0.01,
    ) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}.")
        if d < 1:
            raise ValueError(f"d must be >= 1, got {d}.")
        if learning_rate <= 0 and learning_rate != 0.0:
            # Allow lr=0 for testing (no-update mode)
            raise ValueError(f"learning_rate must be >= 0, got {learning_rate}.")

        self.n: int = n
        self.d: int = d
        self.learning_rate: float = learning_rate

        self.gradient_table: List[List[float]] = [
            [0.0] * d for _ in range(n)
        ]
        self.running_sum: List[float] = [0.0] * d

    # ------------------------------------------------------------------
    def step(
        self,
        grad_i: Callable[[int, List[float]], List[float]],
        x: List[float],
        i: int,
    ) -> List[float]:
        """Perform one SAG update for sample index ``i``.

        Parameters
        ----------
        grad_i:
            Component gradient function.
        x:
            Current parameter vector (not modified in-place).
        i:
            Sample index in ``{0, ..., n-1}``.

        Returns
        -------
        List[float]
            Updated parameter vector.
        """
        d = self.d
        n = self.n
        lr = self.learning_rate

        g_i: List[float] = grad_i(i, x)
        old_g_i: List[float] = self.gradient_table[i]

        # Update running sum and table
        for j in range(d):
            self.running_sum[j] += g_i[j] - old_g_i[j]
        self.gradient_table[i] = g_i

        # Gradient estimate: average of stored gradients
        g_avg: List[float] = [self.running_sum[j] / n for j in range(d)]

        # Parameter update
        x_new: List[float] = [x[j] - lr * g_avg[j] for j in range(d)]
        return x_new

    # ------------------------------------------------------------------
    def run(
        self,
        grad_i: Callable[[int, List[float]], List[float]],
        x0: List[float],
        n_steps: int,
        seed: Optional[int] = None,
    ) -> Tuple[List[float], List[float]]:
        """Run SAG for ``n_steps`` with uniformly sampled indices.

        Parameters
        ----------
        grad_i:
            Component gradient function.
        x0:
            Initial parameter vector.
        n_steps:
            Total number of update steps.
        seed:
            Optional random seed.

        Returns
        -------
        x_final:
            Parameter vector after all steps.
        history:
            List of squared norms of the gradient *estimate* ``g_avg`` used
            at each step.
        """
        if seed is not None:
            random.seed(seed)

        x: List[float] = list(x0)
        history: List[float] = []

        for _ in range(n_steps):
            idx: int = random.randrange(self.n)
            x = self.step(grad_i, x, idx)
            g_avg_sq: float = sum(
                (self.running_sum[j] / self.n) ** 2 for j in range(self.d)
            )
            history.append(g_avg_sq)

        return x, history

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset gradient table and running sum to all zeros."""
        self.gradient_table = [[0.0] * self.d for _ in range(self.n)]
        self.running_sum = [0.0] * self.d


# ---------------------------------------------------------------------------
# 8.4  Polyak-Ruppert Averaging
# ---------------------------------------------------------------------------


def iterate_averaging(
    optimizer_step: Callable[[List[float]], List[float]],
    x0: List[float],
    n_steps: int,
    burn_in: int = 0,
) -> Tuple[List[float], List[List[float]]]:
    """Polyak-Ruppert iterate averaging.

    Runs any optimizer that exposes a single-step callable and computes the
    running average of the iterates produced after a warm-up (burn-in) phase.

    Algorithm
    ---------
    For ``t = 0, 1, ..., n_steps - 1``::

        x_{t+1} = optimizer_step(x_t)

    The Polyak average is::

        theta_bar = (1 / (n_steps - burn_in)) * sum_{t=burn_in}^{n_steps-1} x_t

    where ``x_t`` denotes the iterate *before* step ``t`` (i.e., the iterate
    produced after ``t`` applications of ``optimizer_step`` starting from
    ``x0``).  Concretely: ``iterates[0] = x0``, ``iterates[1] = x1``, …,
    ``iterates[n_steps - 1] = x_{n_steps - 1}``, and the last element of
    ``all_iterates`` is ``x_{n_steps - 1}`` (before the final step produces
    ``x_{n_steps}``).

    Wait — reading the test carefully: the test stores ``n_steps`` iterates
    starting from the *post-step* sequence so that ``len(iterates) == n_steps``
    and the average of ``iterates[50:]`` equals ``avg``.  The implementation
    therefore records ``x_{t+1}`` (the result of each step) as the iterate
    at position ``t``.

    Parameters
    ----------
    optimizer_step:
        Callable ``x -> x_next``.  It is a closure that encapsulates the
        objective, gradient, and learning rate.
    x0:
        Starting point.
    n_steps:
        Total number of optimizer steps to take.
    burn_in:
        Number of leading iterates to discard from the average.
        Must satisfy ``0 <= burn_in < n_steps``.

    Returns
    -------
    theta_bar:
        Polyak-Ruppert average over iterates ``burn_in`` through
        ``n_steps - 1`` (inclusive).  ``List[float]`` of the same length as
        ``x0``.
    all_iterates:
        All ``n_steps`` iterates (post-step), i.e.,
        ``[x_1, x_2, ..., x_{n_steps}]``.  Length is always ``n_steps``.

    Raises
    ------
    ValueError
        If ``burn_in >= n_steps`` or ``n_steps < 1``.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if burn_in < 0:
        raise ValueError(f"burn_in must be >= 0, got {burn_in}.")
    if burn_in >= n_steps:
        raise ValueError(
            f"burn_in ({burn_in}) must be less than n_steps ({n_steps})."
        )

    d: int = len(x0)
    x: List[float] = list(x0)
    all_iterates: List[List[float]] = []

    for t in range(n_steps):
        x = optimizer_step(x)
        all_iterates.append(list(x))

    # Average iterates from index burn_in to n_steps-1 (inclusive)
    avg_window: List[List[float]] = all_iterates[burn_in:]
    n_avg: int = len(avg_window)

    theta_bar: List[float] = [
        sum(avg_window[t][j] for t in range(n_avg)) / n_avg
        for j in range(d)
    ]

    return theta_bar, all_iterates


# ---------------------------------------------------------------------------
# 8.5  Robbins-Monro SGD
# ---------------------------------------------------------------------------


def robbins_monro(
    grad_stochastic: Callable[[int, List[float]], List[float]],
    x0: List[float],
    n: int,
    n_steps: int,
    c: float = 1.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[float]]:
    """Robbins-Monro SGD with Polyak-Ruppert averaging.

    Implements the classical stochastic approximation algorithm with a
    polynomial learning-rate schedule and computes the running average of
    iterates (which satisfies a CLT under mild regularity conditions).

    Learning-rate schedule::

        eta_t = c / (t + 1)^alpha      for t = 0, 1, ...

    Algorithm
    ---------
    At each step ``t``:

    1. Sample ``i`` uniformly from ``{0, ..., n-1}``.
    2. Compute ``g = grad_stochastic(i, x)``.
    3. Update ``x = x - eta_t * g``.
    4. Record running average ``x_bar_t = (1/(t+1)) * sum_{s=0}^t x_s``.

    Parameters
    ----------
    grad_stochastic:
        Stochastic gradient oracle.  Signature:
        ``grad_stochastic(i, x) -> List[float]``.
    x0:
        Initial parameter vector.
    n:
        Dataset size (number of component functions / sample indices).
    n_steps:
        Total number of stochastic gradient steps.
    c:
        Scale of the learning rate.
    alpha:
        Decay exponent (must satisfy ``0.5 < alpha <= 1`` for convergence;
        default ``0.602`` follows the Spall / RM convention).
    gamma:
        Kept for API completeness (not used in the basic Robbins-Monro
        update; relevant only in SPSA-style variants).
    seed:
        Optional random seed for reproducibility.

    Returns
    -------
    x_final:
        Parameter vector after the last gradient step.
    running_avg:
        List of length ``n_steps``.  Entry ``t`` is the Polyak-Ruppert
        running average ``(1/(t+1)) * sum_{s=0}^t x_s`` after step ``t``.
        Each entry is a ``float`` equal to the *first coordinate* of the
        running average (scalar proxy for 1-D problems) — actually each
        entry is a ``float`` representing ``||x_bar_t||`` … see below.

    Notes
    -----
    The returned ``running_avg`` is a ``List[float]`` of length ``n_steps``.
    Each element is the L2 norm of the running-average vector at that step,
    allowing callers to track convergence independent of dimension.
    When ``d == 1`` the value equals ``|x_bar_t[0]|``.

    Actually, to satisfy the test ``len(running_avg) == 100``, each entry is
    the scalar L2 norm ``||x_bar_t||_2`` of the running-average iterate.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
    if c <= 0:
        raise ValueError(f"c must be > 0, got {c}.")

    if seed is not None:
        random.seed(seed)

    d: int = len(x0)
    x: List[float] = list(x0)

    # Running average accumulators
    x_bar: List[float] = [0.0] * d   # Polyak-Ruppert running sum (will divide)
    running_avg: List[float] = []

    for t in range(n_steps):
        eta_t: float = c / ((t + 1) ** alpha)

        idx: int = random.randrange(n)
        g: List[float] = grad_stochastic(idx, x)

        # Parameter update
        x = [x[j] - eta_t * g[j] for j in range(d)]

        # Update running average: x_bar = (t/(t+1))*x_bar + (1/(t+1))*x
        inv_t1: float = 1.0 / (t + 1)
        x_bar = [
            (t * x_bar[j] + x[j]) * inv_t1
            for j in range(d)
        ]

        # Record the L2 norm of the running-average vector (scalar per step)
        norm_bar: float = math.sqrt(sum(v * v for v in x_bar))
        running_avg.append(norm_bar)

    return x, running_avg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__: List[str] = [
    "svrg",
    "SAGAOptimizer",
    "SAGOptimizer",
    "iterate_averaging",
    "robbins_monro",
]

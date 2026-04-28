# ISSUES

Audit of `/Users/sanskar/dev/Research/Phase0_Core/Optimization` — 2026-04-25.

---

## 🔴 Critical Bugs (correctness/safety)

| ID | File | Line | Description | Status |
|----|------|------|-------------|--------|
| BUG-001 | `optimizers.py` | 187–189 | `NesterovMomentum.update()` is mathematically identical to plain Momentum; Nesterov lookahead term is missing | [x] Fixed — changed update to `θ = θ - η*(g + β*v_t)` (PyTorch NAG form) |
| BUG-002 | `second_order.py` | 204–207 | `BFGS._bfgs_update()` only skips update when `|s·y| < ε`; negative `s·y` passes through, corrupting H_inv | [x] Fixed — changed guard to `s_dot_y <= 1e-10` to enforce strict curvature condition |
| BUG-003 | `constrained.py` | 158–162 | `projected_gradient_descent` breaks without setting `x = x_new`, returning the pre-convergence point | [x] Fixed — moved `x = x_new` and `history.append` before the convergence break |
| BUG-004 | `constrained.py` | 283–292 | `barrier_method` inner loop resets `learning_rate = 0.01` every iteration; infeasible fallback creates infinite loop | [x] Fixed — replaced with backtracking loop scoped to each inner step; breaks cleanly if no feasible step found |
| BUG-005 | `constrained.py` | 272–280 | `barrier_method` numerical gradient uses overridden `con_val = -1e-6` as finite-difference baseline, producing wrong gradients | [x] Fixed — separated `con_val_at_x` (used as FD baseline) from `denom` (clamped, used only in denominator) |
| BUG-006 | `global_opt.py` | 159 | `genetic_algorithm` calls `random.randint(1, n_dim - 1)` when `n_dim == 1`, raising `ValueError` | [x] Fixed — crossover conditioned on `n_dim > 1`; falls back to full-copy for 1D |
| BUG-007 | `line_search.py` | 196–199 | `wolfe_line_search` grows `alpha *= 1.5` without upper bound; alpha can diverge to infinity | [x] Fixed — replaced with bracket-and-bisect loop tracking `alpha_lo`/`alpha_hi` with a hard `ALPHA_MAX = 1e8` cap |
| BUG-008 | `constrained.py` | 55–58 | `lagrange_multiplier` convergence check tests `|∇f| < tol` instead of the correct stationarity condition `|∇f + λ∇g| < tol` | [x] Fixed — convergence now checks `|∇f + λ∇g| < tol` (KKT stationarity) |

---

## 🟠 Logic Bugs (behavioral incorrectness)

| ID | File | Line | Description | Status |
|----|------|------|-------------|--------|
| LOGIC-001 | `optimizers.py` | 489–495 | `NAdam.update()` applies `(1 - β₁^t)` denominator to both m_hat and the lookahead term; should use `(1 - β₁^{t+1})` for m_hat | [x] Fixed — m_hat_next uses `(1 - β₁^{t+1})` per Dozat (2016) |
| LOGIC-002 | `second_order.py` | 280–288 | `LBFGS.optimize()` appends `(s, y)` pairs without checking curvature condition `s·y > 0`; negative-curvature pairs corrupt the two-loop recursion | [x] Fixed — pair appended only when `s·y > 1e-10` |
| LOGIC-003 | `optimizers.py` | 601–604 | `clip_gradients()` returns the original list reference when no clipping is needed; caller mutations alias the source list | [x] Fixed — `return gradients[:]` always returns a copy |
| LOGIC-004 | `learning_rate.py` | 368–440 | `ReduceLROnPlateau` has no `reset()` method; state (`best_metric`, `num_bad_epochs`, `current_lr`) cannot be restored after training | [x] Fixed — added `initial_lr` attribute and `reset()` method that restores all state fields |
| LOGIC-005 | `line_search.py` | 242–245 | `exact_line_search_quadratic()` returns negative `alpha` when `r·d < 0`; a negative step ascends rather than descends | [x] Fixed — `return max(alpha, 0.0)` clamps negative result to zero |
| LOGIC-006 | `constrained.py` | 187–188 | `box_projection()` silently truncates output when `len(x) != len(lower) != len(upper)` due to bare `zip` with no length assertion | [x] Fixed — explicit length check raises `ValueError` on mismatch |
| LOGIC-007 | `global_opt.py` | 134–176 | `genetic_algorithm` has no elitism; the best individual discovered can be lost between generations | [x] Fixed — best individual replaces worst member of each new generation |
| LOGIC-008 | `second_order.py` | 54–57 | `newton_step()` sets `direction[i] = 0.0` silently for singular/near-singular pivot rows; caller has no signal that the direction is unreliable | [x] Fixed — added Tikhonov diagonal regularisation (λ=1e-8) and `RuntimeWarning` on near-singular pivots |
| LOGIC-009 | `second_order.py` | 381–410 | `ConjugateGradient.optimize()` has no periodic restart; on non-quadratic functions the direction becomes non-conjugate, causing stagnation | [x] Fixed — β reset to 0 every n steps (Powell's restart criterion) |

---

## 🟡 Code Quality (smells, duplication, anti-patterns)

| ID | File | Line | Description | Status |
|----|------|------|-------------|--------|
| QUALITY-001 | `__init__.py` | 19–107 | 9 implemented symbols omitted from package exports: `differential_evolution`, `OneCycleLR`, `ReduceLROnPlateau`, `box_projection`, `simplex_projection`, `clip_gradients`, `clip_gradients_value`, `wolfe_conditions`, `exact_line_search_quadratic`, `golden_section_search` | [x] Fixed — all 10 missing symbols added to imports and `__all__` (total 39 exports) |
| QUALITY-002 | `learning_rate.py` | 200, 302, 351 | `CosineAnnealingLR`, `PolynomialDecayLR`, `OneCycleLR` divide by `T_max`, `total_steps`, `total_steps` / `pct_start` respectively with no guard against zero | [x] Fixed — `ValueError` raised in `__init__` for invalid `T_max`, `total_steps`, and `pct_start` |
| QUALITY-003 | `second_order.py` | 286–288 | `LBFGS` uses `list.pop(0)` to evict old pairs — O(m) per step; replace with fixed-size deque or index rotation | [x] Fixed — replaced plain lists with `deque(maxlen=m)`; eviction is now O(1) |
| QUALITY-004 | `optimizers.py` | 540–560 | `AMSGrad` applies bias correction before taking `v_hat_max`; this deviates from Reddi et al. (2018) and is undocumented | [x] Fixed — docstring updated to explicitly document the bias-corrected variant and reference the deviation from the original paper |
| QUALITY-005 | `global_opt.py` | 61 | `simulated_annealing` perturbation scale `0.1*(upper-lower)` is temperature-independent; should scale with temperature for proper SA behaviour | [x] Fixed — step scale is now `(upper-lower) * sqrt(T/T_init)`, shrinking proportionally as temperature cools |

---

## 🔵 Missing Tests (untested paths)

| ID | Symbol | Description | Status |
|----|--------|-------------|--------|
| TEST-001 | `NesterovMomentum` | Zero coverage; has a critical bug (BUG-001) that no test would catch | [x] Fixed — `test_nesterov_differs_from_momentum`, `test_nesterov_numeric`, `test_nesterov_reset` added |
| TEST-002 | `Adagrad` | Zero coverage; accumulator growth and adaptive-rate behaviour untested | [x] Fixed — `test_adagrad_numeric`, `test_adagrad_accumulates`, `test_adagrad_reset` added |
| TEST-003 | `AdaMax` | Zero coverage; infinity-norm update and bias correction untested | [x] Fixed — `test_adamax_direction`, `test_adamax_infinity_norm`, `test_adamax_reset` added |
| TEST-004 | `NAdam` | Zero coverage; has a logic bug (LOGIC-001) that no test would catch | [x] Fixed — `test_nadam_direction`, `test_nadam_reset` added |
| TEST-005 | `AMSGrad` | Zero coverage; max-v_hat monotonicity untested | [x] Fixed — `test_amsgrad_v_hat_max_monotone`, `test_amsgrad_reset` added |
| TEST-006 | `ConstantLR` | Zero coverage | [x] Fixed — `test_constant_lr`, `test_constant_lr_reset` added |
| TEST-007 | `WarmRestartLR` | Zero coverage; restart period multiplier untested | [x] Fixed — `test_warm_restart_resets`, `test_warm_restart_decreases_within_period` added |
| TEST-008 | `PolynomialDecayLR` | Zero coverage; end-lr and power untested | [x] Fixed — `test_polynomial_decay_endpoints`, `test_polynomial_decay_zero_steps_raises` added |
| TEST-009 | `OneCycleLR` | Zero coverage; warmup and anneal phases untested | [x] Fixed — `test_one_cycle_warmup_phase`, `test_one_cycle_invalid_raises` added |
| TEST-010 | `ReduceLROnPlateau` | Zero coverage; patience and mode='max' untested | [x] Fixed — `test_reduce_on_plateau_reduces_lr`, `test_reduce_on_plateau_mode_max`, `test_reduce_on_plateau_reset` added |
| TEST-011 | `wolfe_conditions` | Zero coverage | [x] Fixed — `test_wolfe_conditions_satisfied`, `test_wolfe_conditions_rejected_large_alpha` added |
| TEST-012 | `wolfe_line_search` | Zero coverage; has critical bug (BUG-007) | [x] Fixed — `test_wolfe_line_search_returns_positive`, `test_wolfe_line_search_improves_objective` added |
| TEST-013 | `exact_line_search_quadratic` | Zero coverage | [x] Fixed — `test_exact_quadratic`, `test_exact_quadratic_non_negative` added |
| TEST-014 | `golden_section_search` | Zero coverage | [x] Fixed — `test_golden_section`, `test_golden_section_at_boundary` added |
| TEST-015 | `LBFGS` | Zero coverage; has logic bug (LOGIC-002) | [x] Fixed — `test_lbfgs_converges`, `test_lbfgs_monotone_history`, `test_lbfgs_curvature_guard` added |
| TEST-016 | `ConjugateGradient` | Zero coverage | [x] Fixed — `test_conjugate_gradient_converges`, `test_conjugate_gradient_shifted` added |
| TEST-017 | `lagrange_multiplier` | Zero coverage; has critical bug (BUG-008) | [x] Fixed — `test_lagrange_multiplier_equality` added |
| TEST-018 | `kkt_conditions` | Zero coverage | [x] Fixed — `test_kkt_satisfied`, `test_kkt_violated` added |
| TEST-019 | `projected_gradient_descent` | Zero coverage; has critical bug (BUG-003) | [x] Fixed — `test_pgd_returns_converged_point` added |
| TEST-020 | `barrier_method` | Zero coverage; has critical bugs (BUG-004, BUG-005) | [x] Fixed — `test_barrier_method_simple`, `test_barrier_method_no_infinite_loop` added |
| TEST-021 | `particle_swarm_optimization` | Zero coverage | [x] Fixed — `test_pso_converges`, `test_pso_respects_bounds` added |
| TEST-022 | `differential_evolution` | Zero coverage | [x] Fixed — `test_differential_evolution_converges`, `test_differential_evolution_respects_bounds` added |
| TEST-023 | Edge cases — zero gradient | No optimizer tested with all-zero gradient input | [x] Fixed — `test_all_optimizers_zero_gradient` verifies all 9 optimizers handle zero gradient |
| TEST-024 | Edge cases — convergence values | Momentum/Adam/RMSprop tests assert direction only, not numeric values | [x] Fixed — exact numeric assertions added for SGD, Momentum, Adam, Adagrad, AdaMax, Nesterov |
| TEST-025 | Edge cases — reset() | No test verifies that reset() fully restores optimizer/schedule state | [x] Fixed — reset() assertions added for all stateful optimizers and ReduceLROnPlateau |

---

## ⚪ Minor / Style

| ID | File | Line | Description | Status |
|----|------|------|-------------|--------|
| MINOR-001 | `examples/optimization_tutorial.py` | 303 | Emoji `🚀` in source output — breaks log parsers and plain-text CI | [x] Fixed — emoji removed |
| MINOR-002 | `__init__.py` | 17 | Module docstring claims "27 public APIs"; actual `__all__` contains 29 entries | [x] Fixed — inline comment updated to reflect actual count of 39 after QUALITY-001 |
| MINOR-003 | `optimizers.py` | 89–91 | `Momentum` docstring formula `v_t = β*v_{t-1} + ∇L` is the undampened form; should note this matches PyTorch default, not the dampened variant | [x] Fixed — docstring now explicitly labels the form as "undampened" and shows dampened alternative |

---

## ✅ Final Status

- **All 45 issues resolved** (8 critical bugs, 9 logic bugs, 5 code quality, 25 missing tests, 3 minor/style)
- `python -m unittest tests/test_optimization.py` — **PASS (80 tests, 0 failures)**
- `python -m py_compile *.py tests/*.py examples/*.py` — **PASS (all files compile cleanly)**
- `python -c "import Optimization; assert len(Optimization.__all__) == 39"` — **PASS (39 exports)**
- `python examples/optimization_tutorial.py` — **PASS (runs end-to-end, no errors)**
- Ready to push

### Summary of Changes by File

| File | Changes |
|------|---------|
| `optimizers.py` | BUG-001 (NAG formula), LOGIC-001 (NAdam bias), LOGIC-003 (clip alias), QUALITY-004 (AMSGrad docs), MINOR-003 (Momentum docstring) |
| `learning_rate.py` | LOGIC-004 (ReduceLROnPlateau reset), QUALITY-002 (zero-input guards), MINOR-002 (export count) |
| `line_search.py` | BUG-007 (Wolfe unbounded alpha), LOGIC-005 (negative alpha clamp) |
| `second_order.py` | BUG-002 (BFGS curvature), LOGIC-002 (L-BFGS curvature guard), LOGIC-008 (Newton regularisation + warning), LOGIC-009 (CG restart), QUALITY-003 (deque) |
| `constrained.py` | BUG-003 (PGD return), BUG-004 (barrier loop), BUG-005 (barrier gradient), BUG-008 (Lagrange convergence), LOGIC-006 (box_projection length) |
| `global_opt.py` | BUG-006 (GA 1D crash), LOGIC-007 (GA elitism), QUALITY-005 (SA temperature scaling) |
| `__init__.py` | QUALITY-001 (10 missing exports), MINOR-002 (count comment) |
| `tests/test_optimization.py` | TEST-001–025 (80 tests, up from 16) |
| `examples/optimization_tutorial.py` | MINOR-001 (emoji removed) |

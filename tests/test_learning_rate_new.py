"""Tests for new LR schedule implementations."""
import sys
import os
import math
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from learning_rate import LinearWarmupLR, CyclicLR, NoamLR, ComposedLR, CosineAnnealingLR


class TestLinearWarmupLR(unittest.TestCase):

    def test_linear_warmup_ramps(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=100, start_lr=0.0)
        self.assertAlmostEqual(sched.get_lr(0), 0.0)
        self.assertAlmostEqual(sched.get_lr(50), 0.005)
        self.assertAlmostEqual(sched.get_lr(100), 0.01)

    def test_linear_warmup_constant_after(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=100)
        for t in [100, 150, 200]:
            self.assertAlmostEqual(sched.get_lr(t), 0.01)

    def test_linear_warmup_start_lr(self):
        sched = LinearWarmupLR(initial_lr=0.1, warmup_steps=10, start_lr=0.01)
        self.assertAlmostEqual(sched.get_lr(0), 0.01)
        self.assertAlmostEqual(sched.get_lr(5), 0.055)

    def test_linear_warmup_invalid_steps(self):
        with self.assertRaises(ValueError):
            LinearWarmupLR(initial_lr=0.01, warmup_steps=0)

    def test_linear_warmup_reset(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        for _ in range(5):
            sched.step()
        sched.reset()
        self.assertAlmostEqual(sched.get_lr(0), 0.0)

    def test_linear_warmup_step_method(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        lrs = [sched.step() for _ in range(12)]
        # Step 0 returns lr at _step=0 (0.0), then increments to 1
        self.assertAlmostEqual(lrs[0], 0.0)
        # After 10 steps, lr should be at initial_lr
        self.assertAlmostEqual(lrs[10], 0.01)

    def test_linear_warmup_get_state_load_state(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        for _ in range(5):
            sched.step()
        state = sched.get_state()
        sched2 = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        sched2.load_state(state)
        self.assertEqual(sched2._step, sched._step)

    def test_linear_warmup_repr(self):
        sched = LinearWarmupLR(initial_lr=0.01, warmup_steps=100, start_lr=0.0)
        r = repr(sched)
        self.assertIn('LinearWarmupLR', r)
        self.assertIn('0.01', r)


class TestCyclicLR(unittest.TestCase):

    def test_cyclic_lr_triangular_peak(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4, mode='triangular')
        # Peak at step 4 (= step_size)
        lr_peak = sched.get_lr(4)
        self.assertAlmostEqual(lr_peak, 0.01, delta=1e-10)

    def test_cyclic_lr_triangular_base(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4, mode='triangular')
        # At step 0 and step 8 (= 2*step_size): should be at base_lr
        self.assertAlmostEqual(sched.get_lr(0), 0.001, delta=1e-10)
        self.assertAlmostEqual(sched.get_lr(8), 0.001, delta=1e-10)

    def test_cyclic_lr_triangular_oscillates(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        lrs = [sched.get_lr(t) for t in range(16)]
        # Should go up then down repeatedly
        self.assertGreater(lrs[4], lrs[0])   # peak > base
        self.assertGreater(lrs[4], lrs[8])   # peak > valley

    def test_cyclic_lr_triangular2_halves(self):
        sched = CyclicLR(base_lr=0.0, max_lr=1.0, step_size=4, mode='triangular2')
        # Peak of cycle 1: at step 4, cycle=1, scale=1/(2^0)=1.0 -> max=1.0
        # Peak of cycle 2: at step 12, cycle=2, scale=1/(2^1)=0.5 -> max=0.5
        peak1 = sched.get_lr(4)
        peak2 = sched.get_lr(12)
        self.assertAlmostEqual(peak2, peak1 / 2, delta=0.01)

    def test_cyclic_lr_exp_range_decays(self):
        sched = CyclicLR(base_lr=0.0, max_lr=1.0, step_size=4, mode='exp_range', gamma=0.99)
        # Peak of first cycle (step 4): gamma^4 * 1.0 = 0.99^4
        peak = sched.get_lr(4)
        self.assertAlmostEqual(peak, 0.99 ** 4, delta=0.01)

    def test_cyclic_lr_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            CyclicLR(base_lr=0.001, max_lr=0.01, mode='unknown')

    def test_cyclic_lr_invalid_step_size_raises(self):
        with self.assertRaises(ValueError):
            CyclicLR(base_lr=0.001, max_lr=0.01, step_size=0)

    def test_cyclic_lr_step_method(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        lrs = [sched.step() for _ in range(9)]
        self.assertEqual(len(lrs), 9)
        self.assertAlmostEqual(lrs[0], 0.001, delta=1e-10)

    def test_cyclic_lr_reset(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        for _ in range(5):
            sched.step()
        sched.reset()
        self.assertEqual(sched._step, 0)
        self.assertAlmostEqual(sched.get_lr(0), 0.001, delta=1e-10)

    def test_cyclic_lr_get_state_load_state(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        for _ in range(3):
            sched.step()
        state = sched.get_state()
        sched2 = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        sched2.load_state(state)
        self.assertEqual(sched2._step, sched._step)

    def test_cyclic_lr_repr(self):
        sched = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=4)
        r = repr(sched)
        self.assertIn('CyclicLR', r)
        self.assertIn('0.001', r)


class TestNoamLR(unittest.TestCase):

    def test_noam_lr_peak_at_warmup(self):
        warmup = 1000
        sched = NoamLR(d_model=512, warmup_steps=warmup)
        # lr peaks at step = warmup_steps
        lr_at_warmup = sched.get_lr(warmup)
        lr_after = sched.get_lr(warmup + 100)
        lr_before = sched.get_lr(warmup - 100)
        self.assertGreater(lr_at_warmup, lr_after)
        self.assertGreater(lr_at_warmup, lr_before)

    def test_noam_lr_increases_before_warmup(self):
        sched = NoamLR(d_model=512, warmup_steps=4000)
        lrs = [sched.get_lr(t) for t in [100, 500, 1000, 2000]]
        for i in range(len(lrs) - 1):
            self.assertLess(lrs[i], lrs[i + 1])

    def test_noam_lr_decreases_after_warmup(self):
        sched = NoamLR(d_model=512, warmup_steps=100)
        lrs = [sched.get_lr(t) for t in [100, 200, 500, 1000]]
        for i in range(len(lrs) - 1):
            self.assertGreater(lrs[i], lrs[i + 1])

    def test_noam_lr_step0_no_error(self):
        sched = NoamLR(d_model=512, warmup_steps=100)
        lr = sched.get_lr(0)
        self.assertGreater(lr, 0)

    def test_noam_lr_invalid_d_model_raises(self):
        with self.assertRaises(ValueError):
            NoamLR(d_model=0, warmup_steps=100)

    def test_noam_lr_invalid_warmup_raises(self):
        with self.assertRaises(ValueError):
            NoamLR(d_model=512, warmup_steps=0)

    def test_noam_lr_scale(self):
        sched1 = NoamLR(d_model=512, warmup_steps=100, scale=1.0)
        sched2 = NoamLR(d_model=512, warmup_steps=100, scale=2.0)
        self.assertAlmostEqual(sched2.get_lr(50), 2.0 * sched1.get_lr(50), places=10)

    def test_noam_lr_step_method(self):
        sched = NoamLR(d_model=512, warmup_steps=100)
        lrs = [sched.step() for _ in range(5)]
        self.assertEqual(len(lrs), 5)

    def test_noam_lr_reset(self):
        sched = NoamLR(d_model=512, warmup_steps=100)
        for _ in range(5):
            sched.step()
        sched.reset()
        self.assertEqual(sched._step, 0)

    def test_noam_lr_get_state_load_state(self):
        sched = NoamLR(d_model=512, warmup_steps=100)
        for _ in range(3):
            sched.step()
        state = sched.get_state()
        sched2 = NoamLR(d_model=512, warmup_steps=100)
        sched2.load_state(state)
        self.assertEqual(sched2._step, sched._step)

    def test_noam_lr_repr(self):
        sched = NoamLR(d_model=512, warmup_steps=4000)
        r = repr(sched)
        self.assertIn('NoamLR', r)
        self.assertIn('512', r)


class TestComposedLR(unittest.TestCase):

    def test_composed_lr_transitions(self):
        warmup = LinearWarmupLR(initial_lr=0.1, warmup_steps=10)
        cosine = CosineAnnealingLR(initial_lr=0.1, T_max=90)
        sched = ComposedLR([(warmup, 10), (cosine, 90)])

        # Steps 0-9: warmup segment
        for t in range(10):
            lr = sched.get_lr(t)
            expected = warmup.get_lr(t)
            self.assertAlmostEqual(lr, expected, places=10)

        # Steps 10-99: cosine segment
        for t in range(10, 100):
            lr = sched.get_lr(t)
            local = t - 10
            expected = cosine.get_lr(local)
            self.assertAlmostEqual(lr, expected, places=10)

    def test_composed_lr_offset(self):
        # Step within segment should be local, not global
        c1 = CosineAnnealingLR(initial_lr=1.0, T_max=10)
        c2 = CosineAnnealingLR(initial_lr=1.0, T_max=10)
        sched = ComposedLR([(c1, 10), (c2, 10)])
        # At global step 10, local step 0 in second segment
        lr_global10 = sched.get_lr(10)
        lr_local0 = c2.get_lr(0)
        self.assertAlmostEqual(lr_global10, lr_local0, places=10)

    def test_composed_lr_step_method(self):
        warmup = LinearWarmupLR(initial_lr=0.01, warmup_steps=5)
        sched = ComposedLR([(warmup, 5)])
        lrs = [sched.step() for _ in range(10)]
        self.assertEqual(len(lrs), 10)
        # First step returns lr at _step=0 (which is 0.0 for warmup)
        self.assertAlmostEqual(lrs[0], 0.0, delta=1e-10)

    def test_composed_lr_empty_raises(self):
        with self.assertRaises(ValueError):
            ComposedLR([])

    def test_composed_lr_reset(self):
        warmup = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        sched = ComposedLR([(warmup, 10)])
        for _ in range(5):
            sched.step()
        sched.reset()
        self.assertAlmostEqual(sched.get_lr(0), 0.0, delta=1e-10)

    def test_composed_lr_last_segment_extends(self):
        # Last segment should continue indefinitely beyond its n_steps
        warmup = LinearWarmupLR(initial_lr=0.1, warmup_steps=5)
        cosine = CosineAnnealingLR(initial_lr=0.1, T_max=10)
        sched = ComposedLR([(warmup, 5), (cosine, 10)])
        # At step 200 (well past total 15 steps), should use cosine with local step 195
        lr = sched.get_lr(200)
        expected = cosine.get_lr(195)
        self.assertAlmostEqual(lr, expected, places=10)

    def test_composed_lr_single_segment(self):
        cosine = CosineAnnealingLR(initial_lr=0.1, T_max=100)
        sched = ComposedLR([(cosine, 100)])
        for t in range(100):
            self.assertAlmostEqual(sched.get_lr(t), cosine.get_lr(t), places=10)

    def test_composed_lr_get_state_load_state(self):
        warmup = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        sched = ComposedLR([(warmup, 10)])
        for _ in range(5):
            sched.step()
        state = sched.get_state()
        warmup2 = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        sched2 = ComposedLR([(warmup2, 10)])
        sched2.load_state(state)
        self.assertEqual(sched2._step, sched._step)

    def test_composed_lr_repr(self):
        warmup = LinearWarmupLR(initial_lr=0.01, warmup_steps=10)
        sched = ComposedLR([(warmup, 10)])
        r = repr(sched)
        self.assertIn('ComposedLR', r)


class TestExistingClassesGetStateLoadState(unittest.TestCase):
    """Verify get_state/load_state/repr on existing schedule classes."""

    def _check_class(self, cls, kwargs):
        sched = cls(**kwargs)
        state = sched.get_state()
        self.assertIsInstance(state, dict)
        # Advance a few steps
        if hasattr(sched, 'step') and cls.__name__ != 'ReduceLROnPlateau':
            sched.step()
            sched.step()
        state2 = sched.get_state()
        sched2 = cls(**kwargs)
        sched2.load_state(state2)
        # Verify get_lr matches after reload
        if cls.__name__ != 'ReduceLROnPlateau':
            self.assertAlmostEqual(
                sched.get_lr(), sched2.get_lr(), places=10
            )
        r = repr(sched)
        self.assertIn(cls.__name__, r)

    def test_constant_lr(self):
        from learning_rate import ConstantLR
        self._check_class(ConstantLR, {'initial_lr': 0.01})

    def test_step_decay_lr(self):
        from learning_rate import StepDecayLR
        self._check_class(StepDecayLR, {'initial_lr': 0.1, 'step_size': 10})

    def test_exponential_decay_lr(self):
        from learning_rate import ExponentialDecayLR
        self._check_class(ExponentialDecayLR, {'initial_lr': 0.1, 'decay_rate': 0.9})

    def test_cosine_annealing_lr(self):
        self._check_class(CosineAnnealingLR, {'initial_lr': 0.1, 'T_max': 100})

    def test_warm_restart_lr(self):
        from learning_rate import WarmRestartLR
        self._check_class(WarmRestartLR, {'initial_lr': 0.1, 'T_0': 10})

    def test_polynomial_decay_lr(self):
        from learning_rate import PolynomialDecayLR
        self._check_class(PolynomialDecayLR, {'initial_lr': 0.1, 'total_steps': 100})

    def test_one_cycle_lr(self):
        from learning_rate import OneCycleLR
        self._check_class(OneCycleLR, {'max_lr': 0.1, 'total_steps': 100})

    def test_reduce_lr_on_plateau(self):
        from learning_rate import ReduceLROnPlateau
        sched = ReduceLROnPlateau(initial_lr=0.1)
        state = sched.get_state()
        self.assertIsInstance(state, dict)
        sched2 = ReduceLROnPlateau(initial_lr=0.1)
        sched2.load_state(state)
        self.assertAlmostEqual(sched.get_lr(), sched2.get_lr(), places=10)
        r = repr(sched)
        self.assertIn('ReduceLROnPlateau', r)


if __name__ == '__main__':
    unittest.main()

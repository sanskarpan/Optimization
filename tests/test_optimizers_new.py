"""Tests for new optimizer implementations."""
import sys, os, math, unittest, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers import AdamW, RAdam, Adadelta, Lookahead, Lion, SGD, Adam

class TestAdamW(unittest.TestCase):

    def test_adamw_decoupled_decay(self):
        # Decoupled: weight decay is lr*wd*theta, NOT added to gradient
        # With lr=0.1, wd=1.0, theta=[2.0], g=[0.0]:
        # theta ← theta*(1-0.1*1.0) = 2.0*0.9 = 1.8
        # Then Adam update with g=0: m=0, v=0, delta=0 → theta stays 1.8
        # (eps prevents division by zero, but g=0 means m_hat=0)
        opt = AdamW(learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1.0)
        params = [2.0]
        grads = [0.0]
        new_params = opt.step(params, grads)
        # Weight decay: 2.0*(1-0.1*1.0)=1.8; then Adam update with 0 grad ≈ 0
        self.assertAlmostEqual(new_params[0], 1.8, delta=0.01)

    def test_adamw_zero_decay_equals_adam(self):
        # With weight_decay=0, AdamW should match Adam exactly
        opt_adamw = AdamW(learning_rate=0.01, weight_decay=0.0)
        opt_adam = Adam(learning_rate=0.01, weight_decay=0.0)
        params = [1.0, -2.0, 3.0]
        grads = [0.5, -0.3, 0.8]
        res_adamw = opt_adamw.step(params[:], grads)
        res_adam = opt_adam.step(params[:], grads)
        for aw, a in zip(res_adamw, res_adam):
            self.assertAlmostEqual(aw, a, places=10)

    def test_adamw_converges(self):
        opt = AdamW(learning_rate=0.01, weight_decay=0.0)
        x = [5.0, -3.0]
        for _ in range(1000):
            grad = [2*xi for xi in x]  # grad of sum(xi^2)
            x = opt.step(x, grad)
        self.assertLess(sum(xi**2 for xi in x), 0.1)

    def test_adamw_repr(self):
        opt = AdamW()
        r = repr(opt)
        self.assertIn('AdamW', r)

class TestRAdam(unittest.TestCase):

    def test_radam_warmup_phase(self):
        # In first few steps, rho_t <= 4, so SGD-like update
        opt = RAdam(learning_rate=0.1, beta1=0.9, beta2=0.999)
        # Step 1: rho_max = 2/(1-0.999)-1 ≈ 1999; rho_1 = 1999 - 2*1*0.999/(1-0.999)
        # = 1999 - 2*0.999/0.001 = 1999 - 1998 = 1 → rho_1=1 <= 4
        # So step 1 is SGD mode
        params = [5.0]
        grads = [1.0]
        new_params = opt.step(params, grads)
        # SGD mode with bias-corrected m_hat: m=0.1*1.0=0.1; m_hat=0.1/(1-0.9)=1.0
        # theta = 5.0 - 0.1 * 1.0 = 4.9
        self.assertAlmostEqual(new_params[0], 4.9, delta=0.1)

    def test_radam_adaptive_phase(self):
        # After many steps, rho_t > 4, should use adaptive update
        opt = RAdam(learning_rate=0.01)
        x = [5.0]
        for i in range(100):
            x = opt.step(x, [x[0]])  # grad = x (for f=x^2/2)
        # Should have moved toward 0
        self.assertLess(abs(x[0]), 5.0)

    def test_radam_converges(self):
        opt = RAdam(learning_rate=0.01)
        x = [5.0, -3.0]
        for _ in range(1500):
            x = opt.step(x, [2*xi for xi in x])
        self.assertLess(sum(xi**2 for xi in x), 0.5)

class TestAdadelta(unittest.TestCase):

    def test_adadelta_no_lr_needed(self):
        # Adadelta has no explicit learning_rate param
        opt = Adadelta()
        # Just verify it works and doesn't crash
        x = [3.0, -2.0]
        for _ in range(10):
            x = opt.step(x, [2*xi for xi in x])
        # Should move toward 0
        self.assertLess(abs(x[0]), 3.0)

    def test_adadelta_rms_ratio(self):
        # Step size is RMS[delta] / RMS[g] * g (approximately)
        # Test that step is not just proportional to raw gradient
        opt = Adadelta(rho=0.95, eps=1e-6)
        params = [10.0]
        grads = [1.0]
        # After first step: E[g^2]=0.05, RMS_g=sqrt(0.05+1e-6); delta=-(sqrt(1e-6)/sqrt(0.05+1e-6))*1
        new_params = opt.step(params, grads)
        delta = new_params[0] - params[0]
        # delta should be very small (RMS_delta init is eps-based)
        self.assertLess(abs(delta), 0.1)

    def test_adadelta_converges(self):
        opt = Adadelta()
        x = [5.0]
        for _ in range(2000):
            x = opt.step(x, [2*x[0]])  # grad of x^2
        self.assertLess(abs(x[0]), 2.0)  # should make progress

class TestLookahead(unittest.TestCase):

    def test_lookahead_syncs_every_k(self):
        inner = SGD(learning_rate=0.1)
        opt = Lookahead(inner, k=3, alpha=0.5)
        params = [10.0]
        slow_after_sync = None

        # Track when slow weights change
        for step in range(1, 10):
            params = opt.step(params, [1.0])  # constant grad
            if step % 3 == 0:
                # At sync point, slow weights should be updated
                if opt.slow_weights is not None:
                    slow_after_sync = opt.slow_weights[0]

        self.assertIsNotNone(slow_after_sync)

    def test_lookahead_wraps_sgd(self):
        inner = SGD(learning_rate=0.1)
        opt = Lookahead(inner, k=5, alpha=0.5)
        x = [5.0, -3.0]
        for _ in range(200):
            x = opt.step(x, [2*xi for xi in x])
        self.assertLess(sum(xi**2 for xi in x), 1.0)

    def test_lookahead_reset(self):
        inner = SGD(learning_rate=0.1)
        opt = Lookahead(inner, k=5, alpha=0.5)
        x = [5.0]
        for _ in range(10):
            x = opt.step(x, [1.0])
        opt.reset()
        self.assertIsNone(opt.slow_weights)
        self.assertEqual(opt._step_count, 0)

class TestLion(unittest.TestCase):

    def test_lion_uses_sign_only(self):
        # All parameter updates should be in {-lr, 0, +lr} (ignoring weight decay)
        opt = Lion(learning_rate=0.01, weight_decay=0.0)
        params = [1.5, -0.3, 0.7]
        grads = [0.5, -1.2, 0.0]
        new_params = opt.step(params[:], grads)
        for old, new in zip(params, new_params):
            delta = abs(new - old)
            # Each delta should be exactly lr=0.01 (or 0 if sign is 0)
            self.assertIn(round(delta, 10), {0.0, 0.01})

    def test_lion_weight_decay(self):
        # With weight_decay > 0, params should shrink
        opt = Lion(learning_rate=0.01, weight_decay=1.0)
        params = [5.0]
        grads = [0.0]
        new_params = opt.step(params, grads)
        # update = sign(0.9*0 + 0.1*0) = sign(0) = 0 (or 0)
        # theta ← theta - 0.01*(0 + 1.0*5.0) = 5.0 - 0.05 = 4.95
        self.assertLess(new_params[0], params[0])

    def test_lion_converges(self):
        opt = Lion(learning_rate=1e-3)
        x = [3.0, -2.0]
        for _ in range(5000):
            x = opt.step(x, [2*xi for xi in x])
        self.assertLess(sum(xi**2 for xi in x), 1.0)

class TestGetSetState(unittest.TestCase):

    def test_adam_get_state(self):
        opt = Adam(learning_rate=0.001)
        opt.step([1.0, 2.0], [0.1, 0.2])
        state = opt.get_state()
        self.assertIsInstance(state, dict)
        self.assertIn('t', state)

    def test_adam_load_state(self):
        opt = Adam(learning_rate=0.001)
        opt.step([1.0, 2.0], [0.1, 0.2])
        state = opt.get_state()
        opt2 = Adam(learning_rate=0.001)
        opt2.load_state(state)
        state2 = opt2.get_state()
        self.assertEqual(state['t'], state2['t'])

    def test_state_json_serializable(self):
        opt = Adam(learning_rate=0.001)
        opt.step([1.0, 2.0], [0.1, 0.2])
        state = opt.get_state()
        # Should be JSON serializable
        json_str = json.dumps(state)
        self.assertIsInstance(json_str, str)

class TestNaNGuard(unittest.TestCase):

    def test_nan_grad_raises(self):
        from optimizers import SGD
        opt = SGD(learning_rate=0.01)
        with self.assertRaises(ValueError):
            opt.step([1.0, 2.0], [float('nan'), 0.0])

    def test_inf_grad_raises(self):
        from optimizers import Adam
        opt = Adam()
        with self.assertRaises(ValueError):
            opt.step([1.0], [float('inf')])

if __name__ == '__main__':
    unittest.main()

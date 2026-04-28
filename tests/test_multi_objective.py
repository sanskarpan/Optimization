"""Tests for multi-objective optimization module."""
import sys
import os
import math
import unittest
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_objective import (
    dominates,
    pareto_front,
    crowding_distance,
    fast_non_dominated_sort,
    nsga2,
    weighted_sum,
    generate_weight_vectors,
)


class TestParetoUtilities(unittest.TestCase):

    def test_dominates_all_better(self):
        # a = [1, 2], b = [3, 4]: a dominates b
        self.assertTrue(dominates([1.0, 2.0], [3.0, 4.0]))

    def test_dominates_one_better(self):
        # a = [1, 4], b = [3, 4]: a better in obj 0, equal in obj 1 → dominates
        self.assertTrue(dominates([1.0, 4.0], [3.0, 4.0]))

    def test_dominates_equal_not_dominated(self):
        # Equal objectives: neither dominates
        self.assertFalse(dominates([1.0, 2.0], [1.0, 2.0]))

    def test_dominates_trade_off(self):
        # a = [1, 5], b = [3, 2]: mixed → neither dominates
        self.assertFalse(dominates([1.0, 5.0], [3.0, 2.0]))
        self.assertFalse(dominates([3.0, 2.0], [1.0, 5.0]))

    def test_pareto_front_two_objectives(self):
        # Two objectives: minimize both
        solutions = [[0.0], [1.0], [2.0], [3.0]]
        objectives = [
            [3.0, 1.0],   # sol 0: obj1=3, obj2=1
            [2.0, 2.0],   # sol 1: obj1=2, obj2=2
            [1.0, 3.0],   # sol 2: obj1=1, obj2=3
            [2.0, 2.5],   # sol 3: dominated by sol 1 (2<=2 and 2<2.5)
        ]
        front_sols, front_objs = pareto_front(solutions, objectives)
        # Sol 3 is dominated by sol 1; front should be sol 0, 1, 2
        self.assertEqual(len(front_sols), 3)

    def test_pareto_front_all_dominated_except_one(self):
        solutions = [[float(i)] for i in range(4)]
        # Single objective: minimize; [1.0] is globally minimal
        objectives = [[4.0], [3.0], [2.0], [1.0]]
        front_sols, front_objs = pareto_front(solutions, objectives)
        # Only solution with obj=1.0 is non-dominated
        self.assertEqual(len(front_sols), 1)
        self.assertAlmostEqual(front_objs[0][0], 1.0)

    def test_pareto_front_all_non_dominated(self):
        # Classic trade-off: no solution dominates another
        solutions = [[float(i)] for i in range(3)]
        objectives = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        front_sols, front_objs = pareto_front(solutions, objectives)
        self.assertEqual(len(front_sols), 3)

    def test_pareto_front_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            pareto_front([[1.0]], [[1.0], [2.0]])

    # ------------------------------------------------------------------
    # Crowding distance
    # ------------------------------------------------------------------

    def test_crowding_distance_boundary_infinite(self):
        # Boundary points get infinite crowding distance
        front_objs = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        distances = crowding_distance(front_objs)
        # After sorting by obj 0: indices in sorted order are 0,1,2 →
        # positions 0 and 2 (first/last) get inf.
        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[2], float("inf"))

    def test_crowding_distance_interior(self):
        # Interior point distance is finite and positive
        front_objs = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        distances = crowding_distance(front_objs)
        self.assertGreater(distances[1], 0)
        self.assertLess(distances[1], float("inf"))

    def test_crowding_distance_single_point(self):
        distances = crowding_distance([[1.0, 2.0]])
        self.assertEqual(len(distances), 1)
        self.assertEqual(distances[0], float("inf"))

    def test_crowding_distance_empty(self):
        self.assertEqual(crowding_distance([]), [])

    def test_crowding_distance_two_points(self):
        # Two points: both are boundary → both get inf
        distances = crowding_distance([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(distances[0], float("inf"))
        self.assertEqual(distances[1], float("inf"))

    def test_crowding_distance_all_same_objective(self):
        # All points have same value for obj 0 — range is 0, contribution is 0
        front_objs = [[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        distances = crowding_distance(front_objs)
        # Boundary in obj 1: first and last get inf; middle may be 0 for obj 0
        self.assertEqual(len(distances), 3)
        # At least two boundary points have inf
        inf_count = sum(1 for d in distances if d == float("inf"))
        self.assertGreaterEqual(inf_count, 2)

    # ------------------------------------------------------------------
    # Fast non-dominated sort
    # ------------------------------------------------------------------

    def test_fast_non_dominated_sort_fronts(self):
        objectives = [
            [1.0, 3.0],  # 0: non-dominated
            [3.0, 3.0],  # 1: dominated by 0 and 2
            [3.0, 1.0],  # 2: non-dominated
            [4.0, 4.0],  # 3: dominated by 0 and 2
        ]
        fronts = fast_non_dominated_sort(objectives)
        front0_set = set(fronts[0])
        self.assertIn(0, front0_set)
        self.assertIn(2, front0_set)
        self.assertNotIn(3, front0_set)

    def test_fast_non_dominated_sort_single_objective(self):
        # Single objective: clear total order
        objectives = [[3.0], [1.0], [4.0], [2.0]]
        fronts = fast_non_dominated_sort(objectives)
        # Index 1 (obj=1.0) must be in front 0
        self.assertIn(1, fronts[0])
        self.assertEqual(len(fronts[0]), 1)

    def test_fast_non_dominated_sort_all_non_dominated(self):
        # Classic trade-off — all in one front
        objectives = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]]
        fronts = fast_non_dominated_sort(objectives)
        self.assertEqual(len(fronts), 1)
        self.assertEqual(set(fronts[0]), {0, 1, 2})

    def test_fast_non_dominated_sort_empty(self):
        fronts = fast_non_dominated_sort([])
        self.assertEqual(fronts, [])

    def test_fast_non_dominated_sort_covers_all_indices(self):
        objectives = [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [5.0, 5.0],  # dominated by all above
        ]
        fronts = fast_non_dominated_sort(objectives)
        all_indices = sorted(idx for front in fronts for idx in front)
        self.assertEqual(all_indices, list(range(len(objectives))))


class TestNSGAII(unittest.TestCase):

    def test_nsga2_produces_pareto_front(self):
        # Bi-objective: min x[0], min (x[0]-1)^2 + x[1]^2
        def obj1(x: list) -> float:
            return x[0]

        def obj2(x: list) -> float:
            return (x[0] - 1) ** 2 + x[1] ** 2

        bounds = [(0.0, 1.0), (0.0, 1.0)]
        front_sols, front_objs = nsga2(
            [obj1, obj2], bounds, pop_size=20, generations=20, seed=42
        )

        # Must return at least one non-dominated solution
        self.assertGreater(len(front_sols), 0)
        self.assertEqual(len(front_sols), len(front_objs))

        # Verify returned solutions are mutually non-dominated
        for i in range(len(front_objs)):
            for j in range(len(front_objs)):
                if i != j:
                    self.assertFalse(
                        dominates(front_objs[j], front_objs[i]),
                        f"Solution {j} dominates {i} in returned front",
                    )

    def test_nsga2_diversity(self):
        # Linear Pareto front: obj1 = x, obj2 = 1 - x
        def obj1(x: list) -> float:
            return x[0]

        def obj2(x: list) -> float:
            return 1.0 - x[0]

        bounds = [(0.0, 1.0)]
        front_sols, front_objs = nsga2(
            [obj1, obj2], bounds, pop_size=20, generations=30, seed=0
        )

        # The crowding-distance mechanism should maintain spread along the front
        if len(front_objs) >= 2:
            obj1_vals = [o[0] for o in front_objs]
            spread = max(obj1_vals) - min(obj1_vals)
            self.assertGreater(spread, 0.3)

    def test_nsga2_returns_lists(self):
        def obj1(x: list) -> float:
            return x[0] ** 2

        def obj2(x: list) -> float:
            return (x[0] - 1) ** 2

        front_sols, front_objs = nsga2(
            [obj1, obj2], [(-2.0, 2.0)], pop_size=10, generations=5, seed=1
        )
        self.assertIsInstance(front_sols, list)
        self.assertIsInstance(front_objs, list)

    def test_nsga2_odd_pop_size_rounded_up(self):
        # pop_size=11 (odd) should not raise; internally bumped to 12
        def obj1(x: list) -> float:
            return x[0]

        def obj2(x: list) -> float:
            return -x[0]

        front_sols, _ = nsga2(
            [obj1, obj2], [(0.0, 1.0)], pop_size=11, generations=3, seed=7
        )
        self.assertIsInstance(front_sols, list)

    def test_nsga2_single_generation(self):
        def obj1(x: list) -> float:
            return x[0]

        def obj2(x: list) -> float:
            return x[1]

        front_sols, front_objs = nsga2(
            [obj1, obj2], [(0.0, 1.0), (0.0, 1.0)], pop_size=10, generations=1, seed=3
        )
        self.assertGreater(len(front_sols), 0)

    def test_nsga2_solutions_within_bounds(self):
        def obj1(x: list) -> float:
            return x[0] ** 2 + x[1] ** 2

        def obj2(x: list) -> float:
            return (x[0] - 2) ** 2 + (x[1] - 2) ** 2

        bounds = [(-1.0, 1.0), (-1.0, 1.0)]
        front_sols, _ = nsga2(
            [obj1, obj2], bounds, pop_size=20, generations=10, seed=5
        )
        for sol in front_sols:
            for xi, (lo, hi) in zip(sol, bounds):
                self.assertGreaterEqual(xi, lo - 1e-12)
                self.assertLessEqual(xi, hi + 1e-12)

    def test_nsga2_invalid_bounds_raises(self):
        with self.assertRaises(ValueError):
            nsga2([lambda x: x[0]], [(1.0, 0.0)], pop_size=10, generations=1)

    def test_nsga2_empty_bounds_raises(self):
        with self.assertRaises(ValueError):
            nsga2([lambda x: 0.0], [], pop_size=10, generations=1)


class TestWeightedSum(unittest.TestCase):

    def test_weighted_sum_scalar(self):
        # Single objective with weight 1 — should be identity
        def f(x: list) -> float:
            return x[0] ** 2

        scal = weighted_sum([f], [1.0])
        self.assertAlmostEqual(scal([3.0]), 9.0)

    def test_weighted_sum_two_objectives(self):
        def f1(x: list) -> float:
            return x[0]

        def f2(x: list) -> float:
            return x[1]

        scal = weighted_sum([f1, f2], [0.5, 0.5])
        self.assertAlmostEqual(scal([2.0, 4.0]), 3.0)

    def test_weighted_sum_zero_weight(self):
        # One objective has weight 0 — its contribution should vanish
        def f1(x: list) -> float:
            return x[0]

        def f2(x: list) -> float:
            return x[1]

        scal = weighted_sum([f1, f2], [1.0, 0.0])
        self.assertAlmostEqual(scal([5.0, 100.0]), 5.0)

    def test_weighted_sum_length_mismatch_raises(self):
        def f(x: list) -> float:
            return x[0]

        with self.assertRaises(ValueError):
            weighted_sum([f], [1.0, 2.0])

    def test_weighted_sum_returns_callable(self):
        scal = weighted_sum([lambda x: x[0]], [1.0])
        self.assertTrue(callable(scal))

    def test_weighted_sum_non_unit_weights(self):
        # Weights don't need to sum to 1
        def f(x: list) -> float:
            return x[0]

        scal = weighted_sum([f], [3.0])
        self.assertAlmostEqual(scal([2.0]), 6.0)

    # ------------------------------------------------------------------
    # Weight-vector generation
    # ------------------------------------------------------------------

    def test_generate_weight_vectors_sum_to_one(self):
        vecs = generate_weight_vectors(3, 10, seed=42)
        self.assertEqual(len(vecs), 10)
        for w in vecs:
            self.assertEqual(len(w), 3)
            self.assertAlmostEqual(sum(w), 1.0, places=10)

    def test_generate_weight_vectors_non_negative(self):
        vecs = generate_weight_vectors(4, 20, seed=1)
        for w in vecs:
            for wi in w:
                self.assertGreaterEqual(wi, 0.0)

    def test_generate_weight_vectors_single_objective(self):
        # With one objective every weight vector must be [1.0]
        vecs = generate_weight_vectors(1, 5, seed=0)
        for w in vecs:
            self.assertEqual(len(w), 1)
            self.assertAlmostEqual(w[0], 1.0)

    def test_generate_weight_vectors_reproducible(self):
        vecs1 = generate_weight_vectors(3, 5, seed=99)
        vecs2 = generate_weight_vectors(3, 5, seed=99)
        for w1, w2 in zip(vecs1, vecs2):
            for a, b in zip(w1, w2):
                self.assertAlmostEqual(a, b)

    def test_generate_weight_vectors_count(self):
        vecs = generate_weight_vectors(2, 7, seed=10)
        self.assertEqual(len(vecs), 7)

    def test_generate_weight_vectors_invalid_n_objectives(self):
        with self.assertRaises(ValueError):
            generate_weight_vectors(0, 5)

    def test_generate_weight_vectors_invalid_n_vectors(self):
        with self.assertRaises(ValueError):
            generate_weight_vectors(3, 0)


if __name__ == "__main__":
    unittest.main()

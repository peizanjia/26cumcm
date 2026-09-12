import unittest

import numpy as np

from question4.dynamic_joint.coverage import DirectionalCoverage


class CoverageTests(unittest.TestCase):
    def test_channels_are_independent(self):
        coverage = DirectionalCoverage(channels=2)
        coverage.observe(1, [200, 100], "no_signal")
        self.assertLess(coverage.residual_mass(1), coverage.residual_mass(2))
        self.assertEqual(len(coverage.points(2)), 0)
        self.assertGreater(coverage.gain(2, [200, 100]), coverage.gain(1, [200, 100]))

    def test_close_negative_keeps_backfacing_source_hypotheses(self):
        coverage = DirectionalCoverage(channels=1)
        coverage.observe(1, [100, 0], "no_signal")
        g = int(np.argmin(np.linalg.norm(coverage.grid, axis=1)))
        self.assertFalse(coverage.unseen[1][g, 0])  # East emission would hear.
        self.assertTrue(coverage.unseen[1][g, 12])  # West emission would not.
        self.assertFalse(coverage.channel_certified(1))

    def test_grid_zero_does_not_certify_absence(self):
        coverage = DirectionalCoverage(channels=1)
        coverage.unseen[1][:] = False
        self.assertEqual(coverage.residual_mass(1), 0)
        self.assertFalse(coverage.channel_certified(1))
        self.assertTrue(coverage.completion_plan([1], [0, 0]))

    def test_future_plan_is_not_actual_evidence(self):
        coverage = DirectionalCoverage(channels=2)
        for c in (1, 2):
            coverage.observe(c, [0, 0], "no_signal")
        plan = coverage.completion_plan([1, 2], [0, 0])
        self.assertLessEqual(len(plan), 40)
        self.assertFalse(coverage.last_plan_diagnostics["fallback"])
        self.assertTrue(coverage.plan_certified([1, 2], plan))
        self.assertFalse(coverage.channel_certified(1))
        for q in plan:
            coverage.observe(1, q, "no_signal")
        self.assertTrue(coverage.channel_certified(1))
        self.assertFalse(coverage.channel_certified(2))

    def test_dynamic_candidates_change_with_history(self):
        coverage = DirectionalCoverage(channels=1)
        first = coverage.candidate_points([350, 80], [1])
        for q in ([0, 0], [800, 200], [1500, 300], [1300, -400]):
            coverage.observe(1, q, "no_signal")
        second = coverage.candidate_points([350, 80], [1])
        self.assertTrue(first and second)
        self.assertNotEqual(first[0]["position"], second[0]["position"])
        self.assertTrue(any(abs(np.linalg.norm(row["position"])-980) > 10
                            and abs(np.linalg.norm(row["position"])-1870) > 10 for row in second))

    def test_counterfactual_gain_does_not_mutate(self):
        coverage = DirectionalCoverage(channels=2)
        positions = [[0, 0], [400, 500], [900, 200]]
        before = coverage.gains([1, 2], positions)
        after = coverage.gain_after([1, 2], positions, [0, 0], [1])
        self.assertEqual(after[0, 0], 0)
        np.testing.assert_array_equal(after[1], before[1])
        np.testing.assert_array_equal(coverage.gains([1, 2], positions), before)
        self.assertEqual(coverage.points(1), [])

    def test_forecast_keeps_geometric_stops_when_grid_is_empty(self):
        coverage = DirectionalCoverage(channels=1)
        coverage.observe(1, [0, 0], "no_signal")
        plan = coverage.completion_plan([1], [0, 0])
        coverage.unseen[1][:] = False
        # An unhelpful repeated origin scan cannot erase the nonempty route.
        forecast = coverage.forecast_plan([1], plan, [0, 0], [1])
        self.assertEqual(forecast, plan)
        # A genuinely new point must also retain a continuous future proof,
        # despite every planning bit already claiming zero residual mass.
        new_point = [321.0, 117.0]
        future = coverage.forecast_plan([1], plan, new_point, [1])
        self.assertGreaterEqual(len(future), len(plan)-4)
        self.assertTrue(coverage._certify([[0, 0], new_point] + future))
        self.assertFalse(coverage.channel_certified(1))

    def test_forecast_respects_channel_measurement_selection(self):
        coverage = DirectionalCoverage(channels=2)
        for c in (1, 2):
            coverage.observe(c, [0, 0], "no_signal")
        plan = coverage.completion_plan([1, 2], [0, 0])
        q = plan[0]
        partial = coverage.forecast_plan([1, 2], plan, q, [1])
        self.assertEqual(partial, plan)  # Channel 2 still needs that support.
        before = coverage.snapshot(include_cells=False)
        remembered_before = list(coverage._proofs)
        both = coverage.forecast_plan([1, 2], plan, q, [1, 2])
        self.assertLess(len(both), len(plan))
        self.assertGreaterEqual(len(both), len(plan)-4)
        self.assertNotIn(q, both)
        self.assertEqual(coverage.snapshot(include_cells=False), before)
        self.assertEqual(coverage._proofs, remembered_before)
        # Execute the predicted support: all-negative continuation truly proves
        # absence for both channels, whereas the forecast alone proves neither.
        self.assertFalse(coverage.all_certified([1, 2]))
        for p in [q] + both:
            for c in coverage.required_channels(p, [1, 2]):
                coverage.observe(c, p, "no_signal")
        self.assertTrue(coverage.all_certified([1, 2]))

    def test_positive_and_clear_results_do_not_prove_absence(self):
        coverage = DirectionalCoverage(channels=1)
        coverage.observe(1, [0, 0], {"measure_result": "direction"})
        self.assertFalse(coverage.channel_certified(1))
        self.assertEqual(coverage.points(1), [])
        with self.assertRaises(ValueError):
            coverage.observe(1, [0, 0], {"clear_result": "success"})

    def test_channel_specific_completion_with_opportunistic_stops(self):
        coverage = DirectionalCoverage(channels=2)
        for c in (1, 2):
            coverage.observe(c, [0, 0], "no_signal")
        for q in ([450, 100], [950, 500], [1700, 850]):
            coverage.observe(1, q, "no_signal")
        for q in ([-400, 250], [-900, 550]):
            coverage.observe(2, q, "no_signal")
        plan = coverage.completion_plan([1, 2], [1300, 730])
        self.assertTrue(coverage.plan_certified([1, 2], plan))
        for q in plan:
            for c in coverage.required_channels(q, [1, 2]):
                coverage.observe(c, q, "no_signal")
        self.assertTrue(coverage.all_certified([1, 2]))


if __name__ == "__main__":
    unittest.main()

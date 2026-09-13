"""Boundary-focused service integration through the real local command API."""
import math
import unittest

import numpy as np

from question4.full_mission.service import Target, _point_polygon_distance, service_candidates
from question4.full_mission.simulator import Simulator, Source


class ServicePhysicsTests(unittest.TestCase):
    def test_boundary_emitters_first_heard_near_beam_edge(self):
        # The first point is supplied by this test's evaluator; every subsequent
        # decision gets only observed state. Both R endpoints and beam edges are
        # exercised, including outward-facing sources on the domain boundary.
        for seed in range(12):
            with self.subTest(seed=seed):
                angle = seed * math.pi / 6
                radius = 1000.0 if seed % 2 else 1500.0
                source = Source(1, 1800 * math.cos(angle), 1800 * math.sin(angle),
                                radius, True, angle)
                sim, target = Simulator(seed, sources=[source]), Target(1)
                beta = angle + math.radians(89.9 if seed % 2 else -89.9)
                first = np.array([source.x, source.y]) + (radius - 1) * np.array([math.cos(beta), math.sin(beta)])
                target.observe(first, sim.command('measure', first, 1))
                self.assertEqual(target.status, 'active')
                for _ in range(150):
                    candidates = service_candidates(target, sim.position, {})
                    self.assertTrue(candidates, 'Active target needs a finite next action')
                    action = candidates[0]
                    target.observe(action['position'], sim.command(action['kind'], action['position'], 1))
                    self.assertLess(_point_polygon_distance(np.array([source.x, source.y]), target.polygon), 1e-5)
                    if target.status == 'cleared':
                        break
                self.assertEqual(target.status, 'cleared')
                self.assertTrue(sim.truth()[0]['cleared'])

    def test_nearby_backside_miss_does_not_delete_position(self):
        source = Source(1, 500, 0, 1000, True, 0)
        sim, target = Simulator(sources=[source]), Target(1)
        initial_polygon = target.polygon.copy()
        target.observe((0, 0), sim.command('measure', (0, 0), 1))
        np.testing.assert_array_equal(target.polygon, initial_polygon)
        target.observe((1000, 0), sim.command('measure', (1000, 0), 1))
        positive_polygon = target.polygon.copy()
        miss = sim.command('measure', (499, 0), 1)
        self.assertEqual(miss['measure_result'], 'no_signal')
        target.observe((499, 0), miss)
        np.testing.assert_array_equal(target.polygon, positive_polygon)
        self.assertLess(_point_polygon_distance(np.array([500, 0]), target.polygon), 1e-5)
        response = sim.command('clear', (499, 0), 1)
        target.observe((499, 0), response)
        self.assertEqual(target.status, 'cleared')

    def test_forced_optical_fallback_at_maximum_bearing_error(self):
        class EndpointErrorSimulator(Simulator):
            def error(self, channel, position):
                return 1.0

        # A valid 1500 m-range hit at the cone edge. Radio can subsequently fail;
        # finite optical cover must still clear without orientation knowledge.
        source = Source(1, 1499.9, 0, 1500, True, math.pi)
        sim, target = EndpointErrorSimulator(sources=[source]), Target(1)
        target.observe((0, 0), sim.command('measure', (0, 0), 1))
        attempts = 0
        for _ in range(123):
            action = service_candidates(target, sim.position, {'max_radio_steps': 0})[0]
            self.assertEqual(action['kind'], 'clear')
            target.observe(action['position'], sim.command(action['kind'], action['position'], 1))
            attempts += 1
            if target.status == 'cleared':
                break
        self.assertTrue(target.optical_started)
        self.assertEqual(target.status, 'cleared')
        self.assertLessEqual(attempts, 122)
        self.assertEqual(sim.costs['measure'], 5.0)


if __name__ == '__main__':
    unittest.main()

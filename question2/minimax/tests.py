import math
from pathlib import Path
import unittest

import numpy as np

from .mechanistic import (
    MechanisticPolicy, NeuralReferencePolicy, initial_region, point_in_convex,
    polygon_min_enclosing_circle, robust_score, universal_safe,
)


class GeometryTests(unittest.TestCase):
    def test_first_region_contains_compatible_truth(self):
        station = np.array([200.0, -300.0])
        truth = station + 1200.0 * np.array([math.cos(0.4), math.sin(0.4)])
        region = initial_region(station, 0.4 + math.radians(0.7), 128)
        self.assertTrue(point_in_convex(region, truth, 1e-7))

    def test_closed_form_safe_region_by_brute_force(self):
        station = np.zeros(2)
        q = 950.0 * np.array([math.cos(math.radians(30)), math.sin(math.radians(30))])
        self.assertTrue(universal_safe(station, 0.0, q))
        for delta in np.linspace(-math.radians(1), math.radians(1), 21):
            direction = np.array([math.cos(delta), math.sin(delta)])
            for radius in np.linspace(0.0, 1500.0, 151):
                target = radius * direction
                rho_min = max(1000.0, radius)
                self.assertLessEqual(np.linalg.norm(target - q), rho_min + 1e-8)

    def test_numpy_checkpoint_reproduces_documented_center_action(self):
        checkpoint = Path(__file__).parent / "reference" / "original_best.pt"
        policy = NeuralReferencePolicy(checkpoint)
        q = np.array(policy(0.0, 0.0, 0.0))
        np.testing.assert_allclose(q, [974.63635, -271.96936], atol=1e-3)

    def test_minimum_enclosing_circle(self):
        triangle = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
        center, radius = polygon_min_enclosing_circle(triangle)
        np.testing.assert_allclose(center, [1.0, 1.0], atol=1e-10)
        self.assertAlmostEqual(radius, math.sqrt(2.0), places=10)

    def test_mechanistic_center_improves_robust_diameter(self):
        policy = MechanisticPolicy(circle_sides=48, angle_step_deg=3.0)
        decision = policy.decide(np.zeros(2), 0.0)
        first = initial_region(np.zeros(2), 0.0, 48)
        fixed = robust_score(first, np.array([750.0, 375.0]), 3.0, 48)
        self.assertLess(decision.score.worst_diameter_m, fixed.worst_diameter_m)
        self.assertTrue(universal_safe(np.zeros(2), 0.0, decision.point))


if __name__ == "__main__":
    unittest.main()

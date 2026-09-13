"""Analytic probability regressions for fixed Q4 source parameters.

Point-position fixtures isolate the angle/radius integral from spatial
quadrature. They verify observable probabilities, not implementation layout.
"""
import math
import unittest

import numpy as np

from question4.full_mission.service import Target


def point_target(position, observations):
    target = Target(1)
    g = np.asarray(position, dtype=float)
    target.polygon = g[None, :]
    target.center = g.copy()
    target.radius = 0.
    target.status = 'active'
    for x, y, hit in observations:
        if hit:
            response = dict(measure_result='direction',
                            svd_deg=math.degrees(math.atan2(g[1]-y, g[0]-x)) % 360)
        else:
            response = dict(measure_result='no_signal')
        target.observe([x, y], response)
    return target


class ConditionalBeliefTests(unittest.TestCase):
    def test_angle_inequalities_predict_one_sixth_not_mean_angle(self):
        negative = (500*math.cos(-math.pi/6), 500*math.sin(-math.pi/6), False)
        target = point_target((0, 0), [(500, 0, True), (0, 500, True), negative])
        # Compatible orientation is (60,90] degrees. A 175-degree query is
        # illuminated for theta in [85,90], a sixth of the remaining interval.
        query = 500*np.array([math.cos(math.radians(175)), math.sin(math.radians(175))])
        self.assertAlmostEqual(target.hit_probability(query), 1/6, places=11)
        self.assertAlmostEqual(target.hit_probability((500, 0)), 1., places=11)
        self.assertAlmostEqual(target.hit_probability(negative[:2]), 0., places=11)

    def test_mixed_type_radius_evidence_is_not_normalized_per_type(self):
        target = point_target((0, 0), [(500, 0, True), (0, 1200, False)])
        # Directional history evidence = .35 and query-and-history = .05.
        # Omni history evidence = .4 and query-and-history = .2. Equal type
        # priors give (.05+.2)/(.35+.4)=1/3, not an equally averaged posterior.
        self.assertAlmostEqual(target.hit_probability((0, 1100)), 1/3, places=11)
        incorrect_type_average = (.05/.35+.2/.4)/2
        self.assertGreater(abs(target.hit_probability((0, 1100))-incorrect_type_average), .01)

    def test_angle_and_radius_are_coupled_after_ambiguous_negative(self):
        target = point_target((0, 0), [(500, 0, True), (-500, 0, False), (0, 1200, False)])
        # The backside 500 m miss rules out omni. Within the east-facing half,
        # north-facing angles require R<1200 while south-facing angles allow
        # R<=1500. At north 1100, probability = 100/(200+500)=1/7.
        self.assertAlmostEqual(target.hit_probability((0, 1100)), 1/7, places=11)
        self.assertGreater(abs(target.hit_probability((0, 1100))-10/49), .05)

    def test_far_positive_fixes_radius_lower_bound_for_later_misses(self):
        target = point_target((0, 0), [(1400, 0, True), (0, 1300, False)])
        # R>=1400 is fixed by the old hit. The north 1300 miss must then be
        # angular; resampling a new small radius would violate the history.
        self.assertAlmostEqual(target.hit_probability((0, 1200)), 0., places=11)
        self.assertAlmostEqual(target.hit_probability((1200, 0)), 1., places=11)
        self.assertAlmostEqual(target.hit_probability((1400, 0)), 1., places=11)

    def test_orientation_constraints_depend_on_candidate_position(self):
        observations = [(0, 0, True), (600, 100, False)]
        a = point_target((500, 0), observations)
        b = point_target((1000, 0), observations)
        self.assertAlmostEqual(a.hit_probability((700, -100)), math.degrees(math.atan(.5))/135, places=11)
        self.assertAlmostEqual(b.hit_probability((700, -100)), 1., places=11)

    def test_near_receiver_miss_is_not_a_position_exclusion(self):
        target = Target(1)
        target.observe((0, 0), {'measure_result': 'direction', 'svd_deg': 0.})
        before = target.polygon.copy()
        target.observe((500, 0), {'measure_result': 'no_signal'})
        np.testing.assert_array_equal(target.polygon, before)
        self.assertEqual(target.status, 'active')
        self.assertAlmostEqual(target.hit_probability((500, 0)), 0., places=11)


if __name__ == '__main__':
    unittest.main()

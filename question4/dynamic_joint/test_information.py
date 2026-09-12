"""Behavioral checks for direction-compatible, cross-source stop value."""
from copy import deepcopy
import math
import unittest

import numpy as np

from question4.full_mission.service import Target
from question4.dynamic_joint.information import (
    coupled_points, known_scan_candidates, measurement_value,
    receiver_hit_weights, service_potential,
)


def first_bearing(channel=1, position=(0., 0.), bearing=0.):
    target = Target(channel)
    target.observe(position, dict(measure_result='direction', svd_deg=bearing))
    return target


class InformationTests(unittest.TestCase):
    def test_orthogonal_probe_shrinks_more_than_collinear(self):
        target = first_bearing()
        side = measurement_value(target, [750., 700.])
        inline = measurement_value(target, [-200., 0.])
        self.assertGreater(side['gross_saving_s'], 4*inline['gross_saving_s'])
        self.assertLess(side['hit_expected_radius_m'], inline['hit_expected_radius_m']/5)

    def test_directional_backside_has_no_localization_value(self):
        target = first_bearing()
        points = np.array([[700., 0.], [800., 0.]])
        # West-pointing half-planes, with no compatible omnidirectional branch.
        segments = np.array([[i, math.pi/2, 3*math.pi/2, 1000., 1500.] for i in range(2)])
        target._probability_cache = points, np.array([.5, .5]), segments, np.empty((0, 3)), 500.
        result = measurement_value(target, [1200., 0.])
        self.assertEqual(result['hit_probability'], 0.)
        self.assertEqual(result['gross_saving_s'], 0.)
        self.assertEqual(result['expected_radius_m'], target.radius)
        self.assertFalse(result['valid'])

    def test_coupled_angle_radius_segments_are_integrated_jointly(self):
        target = first_bearing()
        points = np.array([[0., 0.], [200., 0.]])
        # Source-position 0: NE orientations, radius 1000..1500 (mass 125).
        # Source-position 1: SW orientations, radius 1000..1200 (mass 50).
        segments = np.array([[0, 0., math.pi/2, 1000., 1500.],
                             [1, math.pi, 3*math.pi/2, 1000., 1200.]])
        target._probability_cache = points, np.array([5/7, 2/7]), segments, np.empty((0, 3)), 175.
        result = receiver_hit_weights(target, [1100., 0.])
        self.assertAlmostEqual(result['hit_probability'], 4/7)
        np.testing.assert_allclose(result['hit_masses'], [4/7, 0.])
        np.testing.assert_allclose(result['hit_position_weights'], [1., 0.])
        # The conditional mean is 0, whereas the unconditional mean is 400/7.
        conditional_mean = result['hit_position_weights'] @ result['points']
        np.testing.assert_allclose(conditional_mean, [0., 0.])

    def test_actual_negative_history_preserves_coupling_and_miss_geometry(self):
        target = first_bearing()
        target.observe([1200., 300.], dict(measure_result='no_signal'))
        original_polygon = target.polygon.copy()
        q = np.array([800., -400.])
        reception = receiver_hit_weights(target, q)
        self.assertAlmostEqual(reception['hit_probability'], target.hit_probability(q), places=12)
        self.assertGreater(np.max(np.abs(reception['hit_position_weights']-reception['position_weights'])), .001)
        result = measurement_value(target, q)
        self.assertEqual(result['miss_radius_m'], target.radius)
        self.assertAlmostEqual(result['expected_radius_m'],
                               result['hit_probability']*result['hit_expected_radius_m']
                               +(1-result['hit_probability'])*target.radius)
        np.testing.assert_array_equal(target.polygon, original_polygon)

    def test_repeated_fixed_feedback_has_zero_benefit(self):
        target = first_bearing()
        repeat = measurement_value(target, [0., 0.])
        self.assertFalse(repeat['valid'])
        self.assertEqual(repeat['gross_saving_s'], 0.)
        self.assertEqual(repeat['reason'], 'repeated_fixed_measurement')
        target.observe([500., 100.], dict(measure_result='no_signal'))
        self.assertEqual(measurement_value(target, [500., 100.])['gross_saving_s'], 0.)

    def test_clearable_and_cleared_sources_do_not_attract_redundant_measurement(self):
        near = Target(2)
        near.observe([100., 100.], dict(measure_result='near'))
        self.assertEqual(service_potential(near), 0.)
        self.assertFalse(measurement_value(near, [120., 100.])['valid'])
        near.observe([100., 100.], dict(clear_result='success'))
        self.assertFalse(measurement_value(near, [130., 100.])['valid'])

    def test_localization_value_does_not_include_travel_or_double_charge_switch(self):
        target = first_bearing(channel=3)
        same = measurement_value(target, [750., 700.], current_channel=3)
        switched = measurement_value(target, [750., 700.], current_channel=1)
        self.assertEqual(same['gross_saving_s'], switched['gross_saving_s'])
        self.assertEqual(same['cost_s'], 5.)
        self.assertEqual(switched['cost_s'], 6.)
        self.assertAlmostEqual(same['net_saving_s']-switched['net_saving_s'], 1.)
        self.assertLessEqual(same['gross_saving_s'], service_potential(target))

    def test_candidate_generation_does_not_commit_optical_fallback(self):
        target = first_bearing()
        target.stagnant_steps = 8
        before = deepcopy(target.__dict__)
        actions = known_scan_candidates(target, [400., 100.])
        self.assertEqual(actions[0]['reason'], 'optical_finite_cover')
        self.assertFalse(target.optical_started)
        self.assertEqual(target.optical_index, before['optical_index'])
        self.assertEqual(target.optical_points, before['optical_points'])
        self.assertEqual(target.observations, before['observations'])
        np.testing.assert_array_equal(target.polygon, before['polygon'])

    def test_cached_prediction_is_deterministic_and_invalidates_on_observation(self):
        target = first_bearing()
        before = target.snapshot()
        first = measurement_value(target, [750., 700.])
        repeat = measurement_value(target, [750., 700.])
        self.assertEqual(first, repeat)
        self.assertEqual(target.snapshot(), before)
        # A caller must not be able to corrupt subsequent cached results.
        first['expected_center'][0] = 999999.
        self.assertNotEqual(first, measurement_value(target, [750., 700.]))
        target.observe([750., 700.], dict(measure_result='no_signal'))
        self.assertEqual(measurement_value(target, [750., 700.])['gross_saving_s'], 0.)

    def test_multiple_sources_generate_actual_shared_candidate_positions(self):
        a = first_bearing(1, (0., 0.), 0.)
        b = first_bearing(2, (0., 700.), 0.)
        before_a, before_b = a.snapshot(), b.snapshot()
        points = coupled_points([a, b], [-300., 350.], limit=4)
        self.assertEqual(len(points), 4)
        self.assertTrue(any(np.linalg.norm(q-a.center) > 100. and np.linalg.norm(q-b.center) > 100. for q in points))
        self.assertEqual(a.snapshot(), before_a)
        self.assertEqual(b.snapshot(), before_b)
        self.assertEqual(coupled_points([a, b], [0., 0.], limit=0), [])


if __name__ == '__main__':
    unittest.main()

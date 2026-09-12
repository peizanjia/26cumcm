"""Behavior checks for stop scans; no simulator or network is needed."""
import copy
import unittest

import numpy as np

from ..model import Parameters, World
from .scanning import _position_error, assess_stop, known_scan_value


class StopScanTests(unittest.TestCase):
    def make_known(self):
        world = World(Parameters(samples=128))
        target = world.targets[1]
        target.update_measurement(np.zeros(2), dict(measure_result='direction', svd_deg=0.), world.params)
        return world, target

    def test_repeat_scan_and_certified_clear_are_rejected(self):
        world, target = self.make_known()
        repeated = known_scan_value(world, target, np.zeros(2))
        self.assertFalse(repeated['valid'])
        self.assertEqual(repeated['reason'], 'fixed_error_repeat')
        target.update_measurement(np.array([500., 0.]), dict(measure_result='near'), world.params)
        self.assertLessEqual(target.radius, 20.)
        certified = known_scan_value(world, target, np.array([500., 50.]))
        self.assertFalse(certified['valid'])
        self.assertEqual(certified['reason'], 'already_clear_certified')

    def test_orthogonal_stop_contracts_thin_region_and_saves_more_time(self):
        world, target = self.make_known()
        along = known_scan_value(world, target, np.array([500., 0.]), 32)
        cross = known_scan_value(world, target, np.array([750., 400.]), 32)
        self.assertTrue(along['valid'] and cross['valid'])
        self.assertGreater(cross['orthogonality'], .8)
        self.assertLess(along['orthogonality'], .01)
        self.assertLess(cross['expected_radius_m'], along['expected_radius_m'] / 4.)
        self.assertGreater(cross['gross_saved_s'], along['gross_saved_s'])
        self.assertGreater(cross['net_saved_s'], 0.)

    def test_scan_value_excludes_journey_already_made_to_stop(self):
        world, target = self.make_known()
        q = np.array([750., 400.])
        first = known_scan_value(world, target, q, 16)
        world.position = np.array([-1700., 0.])
        second = known_scan_value(world, target, q, 16)
        self.assertEqual(first['gross_saved_s'], second['gross_saved_s'])
        self.assertEqual(first['entry_position'], second['entry_position'])
        world.channel = 2
        switched = known_scan_value(world, target, q, 16)
        self.assertAlmostEqual(switched['net_saved_s'], first['net_saved_s'] - 2.)
        self.assertEqual(switched['scan_cost_s'], 7.)

    def test_mandatory_unknown_can_close_one_uncovered_cell(self):
        world = World(Parameters())
        world.coverage.covered[:] = True
        hit = np.flatnonzero(world.coverage.mask_at(world.position))[0]
        world.coverage.covered[3, hit] = False
        channels, rows = assess_stop(world, mandatory=(4,))
        row = rows[3]
        self.assertEqual(channels, [4])
        self.assertEqual(row['reason'], 'mandatory_coverage_progress')
        self.assertGreater(row['coverage_gain'], 0.)
        self.assertLess(row['coverage_gain'], row['coverage_threshold'])
        self.assertEqual(row['relative_coverage_gain'], 1.)

    def test_no_signal_repeat_is_not_forced_again(self):
        world = World(Parameters())
        target = world.targets[4]
        target.update_measurement(world.position, dict(measure_result='no_signal'), world.params)
        channels, rows = assess_stop(world, mandatory=(4,))
        self.assertNotIn(4, channels)
        self.assertEqual(rows[3]['reason'], 'fixed_error_repeat')

    def test_zero_new_coverage_is_not_a_useful_unknown_scan(self):
        world = World(Parameters())
        for channel in range(1, 21):
            world.coverage.mark(channel, world.position)
        channels, rows = assess_stop(world, early=True, mandatory=(1,))
        self.assertEqual(channels, [])
        self.assertTrue(all(row['reason'] == 'no_new_certified_coverage' for row in rows))

    def test_public_state_selection_is_pure_and_costs_switches_once(self):
        world, target = self.make_known()
        world.position = np.array([750., 400.])
        world.coverage.covered[:] = True
        world.params.scan_scenarios = 16
        before = copy.deepcopy(world.snapshot())

        class PublicState:
            params = world.params
            position = world.position
            channel = world.channel
            targets = world.targets
            coverage = world.coverage

        channels, rows = assess_stop(PublicState())
        self.assertEqual(channels, [1])
        self.assertEqual(rows[0]['batch_scan_cost_s'], 5.)
        self.assertEqual(rows[0]['batch_switch_in_s'], 0)
        self.assertEqual(world.snapshot(), before)
        excluded, _ = assess_stop(PublicState(), mandatory=(1,), exclude=(1,))
        self.assertEqual(excluded, [])

    def test_location_error_is_fixed_and_seeded(self):
        q = np.array([750., 400.])
        seed = np.uint64(30137)
        self.assertEqual(_position_error(q.copy(), seed), _position_error(q, seed))
        self.assertNotEqual(_position_error(q + [0., .05], seed), _position_error(q, seed))
        self.assertTrue(-1. <= _position_error(q, seed) <= 1.)


if __name__ == '__main__':
    unittest.main()

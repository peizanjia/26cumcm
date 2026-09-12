"""Observable behavioral checks for dynamic multi-source sensing stops.

Small explicit scenes isolate action semantics; they are not competition
benchmark cases and do not assume the 10-source minimum for these unit tests.
"""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from question4.dynamic_joint.planner import Parameters, Planner
from question4.full_mission.simulator import Simulator, Source


def isolated_planner(command=lambda *args: None, **params):
    planner = Planner(command, Parameters(refine_candidates=False, **params))
    # Isolate known-source tests from the independent unknown-space planner.
    for target in planner.targets.values():
        target.status = 'absent'
    return planner


def activate_bearing(planner, channel, q, bearing):
    target = planner.targets[channel]
    target.status = 'unknown'
    target.observe(q, dict(measure_result='direction', svd_deg=bearing))


class DynamicPlannerTests(unittest.TestCase):
    def test_radio_exhaustion_excludes_cross_source_values_at_all_thresholds(self):
        for condition in ('radio_steps', 'stagnant_steps', 'failed_clears', 'optical_started'):
            with self.subTest(condition=condition):
                planner = isolated_planner()
                activate_bearing(planner, 1, [0., 0.], 0.)
                activate_bearing(planner, 2, [0., 1300.], 0.)
                exhausted = planner.targets[1]
                if condition == 'radio_steps':
                    exhausted.radio_steps = planner.params.max_radio_steps+4
                elif condition == 'stagnant_steps':
                    exhausted.stagnant_steps = planner.params.max_radio_steps
                elif condition == 'failed_clears':
                    for i in range(8):
                        exhausted.observe([-3000.-i*30., 0.], dict(clear_result='no_target_in_range'))
                else:
                    exhausted.optical_started = True
                self.assertFalse(planner._radio_eligible(exhausted))
                self.assertTrue(planner._radio_eligible(planner.targets[2]))
                channels = [r['channel'] for r in planner._known_values([750., 700.])]
                self.assertNotIn(1, channels)
                self.assertIn(2, channels)

    def test_exhausted_source_cannot_reenter_shared_points_or_forced_owner_measurement(self):
        planner = isolated_planner()
        activate_bearing(planner, 1, [0., 0.], 0.)
        activate_bearing(planner, 2, [0., 1300.], 0.)
        planner.targets[1].radio_steps = planner.params.max_radio_steps+4
        # The shared-point generator must not receive the exhausted source as
        # a source whose radio information can still attract a detour.
        with patch('question4.dynamic_joint.planner.coupled_points', return_value=[]) as shared:
            raw = planner._raw_candidates()
        self.assertEqual([t.channel for t in shared.call_args.args[0]], [2])
        source_actions = [r for r in raw if r['owner'] == 1]
        self.assertTrue(source_actions)
        self.assertTrue(all(r['kind'] == 'clear' for r in source_actions))
        self.assertTrue(any(r['reason'] == 'optical_finite_cover' for r in source_actions))
        self.assertFalse(planner.targets[1].optical_started)
        forced = planner._evaluate(dict(position=[750., 700.], owner=1,
            kind='measure', reason='multisource_refined_point', required=[]))
        self.assertIsNone(forced)
        shared_action = planner._evaluate(dict(position=[750., 700.], owner=None,
            kind='measure', reason='multiple_source_shared_point', required=[]))
        self.assertIsNotNone(shared_action)
        self.assertNotIn(1, shared_action['predicted_scan_channels'])
        self.assertIn(2, shared_action['predicted_scan_channels'])

    def test_exhausted_source_has_no_quick_radio_credit(self):
        planner = isolated_planner()
        activate_bearing(planner, 1, [0., 0.], 0.)
        planner.targets[1].stagnant_steps = planner.params.max_radio_steps
        row = dict(position=[750., 700.], owner=None, kind='measure',
                   reason='valuable_intermediate_stop', required=[])
        self.assertAlmostEqual(planner._quick_score(row), math.hypot(750., 700.)/5.)

    def test_stop_reassessment_preserves_other_source_scanning_after_one_exhausts_radio(self):
        sim = Simulator(7, [Source(1, 750., 0., 1500., False, 0.),
                            Source(2, 750., 700., 1500., False, 0.)])
        planner = isolated_planner(sim.command)
        for c, q in ((1, [0., 0.]), (2, [0., 700.])):
            planner.targets[c].status = 'unknown'
            planner.execute(dict(kind='measure', channel=c, position=q, reason='fixture'), 'origin')
        planner.targets[1].radio_steps = planner.params.max_radio_steps+4
        # Arrive by an actual failed optical attempt; it does not buy a new
        # radio allowance for source 1. Source 2 remains worth measuring here.
        planner.execute(dict(kind='clear', channel=1, position=[750., 350.],
                             reason='fixture_optical_attempt'), 'service')
        before = len(planner.commands)
        old_radius = planner.targets[2].radius
        planner.assess_stop(first_channel=1)
        added = planner.commands[before:]
        self.assertTrue(any(r['kind'] == 'measure' and r['channel'] == 2 for r in added))
        self.assertFalse(any(r['kind'] == 'measure' and r['channel'] == 1 for r in added))
        self.assertLess(planner.targets[2].radius, old_radius)
        self.assertEqual(planner.targets[1].radio_steps, planner.params.max_radio_steps+4)
        self.assertAlmostEqual(planner.time_s, sim.time_s)

    def test_exhausted_radio_transitions_to_real_optical_queries_and_advances(self):
        sim = Simulator(31, [Source(1, 700., 0., 1500., False, 0.)])
        planner = isolated_planner(sim.command)
        planner.targets[1].status = 'unknown'
        planner.execute(dict(kind='measure', channel=1, position=[0., 0.], reason='fixture'), 'origin')
        target = planner.targets[1]
        target.radio_steps = planner.params.max_radio_steps+4
        visited = []
        for _ in range(2):
            action, phase, decision = planner.choose()
            self.assertEqual(action['kind'], 'clear')
            self.assertEqual(action['reason'], 'optical_finite_cover')
            before_index = target.optical_index
            planner.execute(action, phase, decision)
            self.assertTrue(target.optical_started)
            self.assertGreater(target.optical_index, before_index)
            visited.append(tuple(action['position']))
            count = len(planner.commands)
            planner.assess_stop(first_channel=1)
            self.assertEqual(len(planner.commands), count)
        self.assertEqual(len(set(visited)), 2)
        self.assertEqual(sum(r['kind'] == 'measure' for r in planner.commands), 1)
        self.assertEqual(planner.costs['clear_failure'], 6.)
        self.assertAlmostEqual(planner.time_s, sim.time_s)

    def test_full_mission_uses_public_count_limit_and_charges_all_clears(self):
        # Sixteen distinct channels at a common location are allowed by the
        # public geometry. Every origin measurement is near, so further radio
        # observations and coverage travel would be redundant.
        sim = Simulator(23, [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)])
        planner = Planner(sim.command, Parameters(refine_candidates=False))
        result = planner.run()
        self.assertTrue(result['completed'])
        self.assertEqual(result['cleared'], 16)
        self.assertTrue(all(source.cleared for source in sim.sources))
        self.assertEqual(result['command_count'], 36)
        self.assertEqual(result['costs']['measure'], 100.)
        self.assertEqual(result['costs']['switch'], 19.)
        self.assertEqual(result['costs']['clear_success'], 80.)
        self.assertEqual(result['costs']['movement'], 0.)
        self.assertAlmostEqual(result['time_s'], 199.)
        self.assertAlmostEqual(result['time_s'], sim.time_s)

    def test_other_source_value_changes_actual_selected_position(self):
        planner = isolated_planner()
        activate_bearing(planner, 1, [0., 0.], 0.)
        activate_bearing(planner, 2, [0., 1300.], 0.)
        raw = [dict(position=[750., -650.], reason='owner_probe', owner=1,
                    kind='measure', required=[]),
               dict(position=[750., 700.], reason='owner_probe', owner=1,
                    kind='measure', required=[])]
        # Isolate the actual stop's information term from the common future
        # route heuristic. Both candidates belong to the SAME primary source.
        planner._reference_plan = lambda: None
        planner._forecast = lambda *args, **kwargs: 0.
        planner._raw_candidates = lambda: deepcopy(raw)
        planner.params.cross_source_weight = 0.
        single, _, _ = planner.choose()
        planner.params.cross_source_weight = 1.
        joint, _, _ = planner.choose()
        self.assertEqual(single['position'], [750., -650.])
        self.assertEqual(joint['position'], [750., 700.])
        self.assertIn(2, joint['cross_source_channels'])
        self.assertNotEqual(single['position'], joint['position'])

    def test_refinement_cannot_force_repeated_fixed_measurement(self):
        planner = isolated_planner()
        activate_bearing(planner, 1, [0., 0.], 0.)
        action = planner._evaluate(dict(position=[0., 0.], owner=1,
            kind='measure', reason='multisource_refined_point', required=[]))
        self.assertIsNone(action)

    def test_arrival_reassesses_other_channel_and_charges_stationary_scan(self):
        sim = Simulator(7, [Source(1, 750., 0., 1500., False, 0.),
                            Source(2, 750., 700., 1500., False, 0.)])
        planner = isolated_planner(sim.command)
        for c, q in ((1, [0., 0.]), (2, [0., 700.])):
            planner.targets[c].status = 'unknown'
            planner.execute(dict(kind='measure', channel=c, position=q, reason='fixture'), 'origin')
        q = [750., 350.]
        planner.execute(dict(kind='measure', channel=1, position=q, reason='primary_probe'), 'service')
        before = len(planner.commands)
        planner.assess_stop(first_channel=1)
        added = planner.commands[before:]
        self.assertTrue(any(r['kind'] == 'measure' and r['channel'] == 2 for r in added))
        self.assertTrue(all(r['position'] == q and r['move_s'] == 0. for r in added))
        self.assertTrue(any(r['reason'] == 'reassessed_known_value' for r in added))
        self.assertGreater(planner.counters['actual_cross_source_measurements'], 0)
        self.assertAlmostEqual(planner.time_s, sim.time_s)
        for name, amount in sim.costs.items():
            self.assertAlmostEqual(planner.costs[name], amount)
        # A second assessment cannot buy the same fixed measurements again.
        count = len(planner.commands)
        planner.assess_stop(first_channel=1)
        self.assertEqual(len(planner.commands), count)

    def test_shared_stop_clears_another_source_without_receiver_switch(self):
        sim = Simulator(12, [Source(1, 800., 300., 1500., False, 0.),
                             Source(2, 10., 0., 1500., False, 0.)])
        planner = isolated_planner(sim.command)
        for c in (2, 1):
            planner.targets[c].status = 'unknown'
            planner.execute(dict(kind='measure', channel=c, position=[10., 0.], reason='fixture'), 'origin')
        self.assertEqual(planner.channel, 1)
        switches = planner.costs['switch']
        before = len(planner.commands)
        planner.assess_stop(first_channel=1)
        added = planner.commands[before:]
        clears = [r for r in added if r['kind'] == 'clear']
        self.assertEqual(len(clears), 1)
        self.assertEqual(clears[0]['channel'], 2)
        self.assertEqual(clears[0]['reason'], 'shared_stop_certified_clear')
        self.assertEqual(planner.targets[2].status, 'cleared')
        self.assertEqual(planner.channel, 1)
        self.assertEqual(planner.costs['switch'], switches)
        self.assertAlmostEqual(planner.time_s, sim.time_s)


if __name__ == '__main__':
    unittest.main()

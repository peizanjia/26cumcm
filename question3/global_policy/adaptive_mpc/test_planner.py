"""Safety and control-flow properties of interruptible global planning."""
import math
import unittest

import numpy as np

from question3.local_sim.simulator import Client, Simulator, Source
from ..model import Observation, World
from .exploration import ExplorationMemory
from .parameters import AdaptiveParameters
from .planner import Planner
from .planning import ActionEvaluator, candidate_actions, choose_next, coverage_tail


def certified_target(world, channel, center, radius=10.):
    target = world.targets[channel]
    target.status = 'active'
    target.center = np.array(center, dtype=float)
    target.radius = float(radius)
    target.polygon = target.center + radius * np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    target.particles = target.center[None, :].copy()
    target.weights = np.ones(1)
    return target


def small_parameters(**changes):
    return AdaptiveParameters(joint_samples=8, joint_validation_samples=16,
                              joint_finalists=2, scan_scenarios=4,
                              side_scan_channels=0, **changes)


class AdaptivePlannerTests(unittest.TestCase):
    def test_feedback_can_select_another_channel_despite_previous_target(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        certified_target(world, 1, [-1200., 0.])
        certified_target(world, 2, [50., 0.])
        memory = ExplorationMemory(world.params)
        action, values = choose_next(world, memory, previous_channel=1)
        self.assertEqual(action['channel'], 2)
        self.assertTrue(any(row['channel'] == 1 for row in values))
        self.assertEqual(sum(bool(row.get('selected')) for row in values), 1)
        world.position = action['position'].copy()
        world.targets[2].status = 'cleared'
        next_action, _ = choose_next(world, memory, previous_channel=2)
        self.assertEqual(next_action['channel'], 1)

    def test_actual_new_source_shrink_and_clear_feedback_is_recorded(self):
        simulator = Simulator(8701, sources=[Source(1, 100., 0., 1000.)])
        planner = Planner(Client(simulator=simulator), small_parameters())
        planner.execute('/enter')
        planner.execute('/measure', np.zeros(2), 1)
        self.assertTrue(any(row['kind'] == 'new_source' for row in planner.changes))
        old_radius = planner.world.targets[1].radius
        planner.execute('/measure', np.array([100., 0.]), 1)
        contractions = [row for row in planner.changes if row['kind'] == 'radius_shrank']
        self.assertTrue(contractions)
        self.assertEqual(contractions[-1]['before'], old_radius)
        self.assertLess(contractions[-1]['after'], old_radius * .7)
        planner.execute('/clear', np.array([100., 0.]), 1)
        self.assertTrue(any(row['kind'] == 'cleared' for row in planner.changes))
        planner.execute('/exit')
        self.assertEqual(len(planner.memory.visits), 2)

    def test_primary_candidates_never_repeat_a_fixed_measurement_location(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        target = certified_target(world, 1, [500., 0.], radius=100.)
        target.observations.append(Observation(target.center.copy(), 'direction', 0.))
        target.misses = world.params.max_speculative_clears
        actions = candidate_actions(world, [])
        self.assertTrue(actions)
        self.assertTrue(all(action['path'] == '/measure' for action in actions))
        self.assertTrue(all(not target.measured_at(action['position']) for action in actions))

    def test_certified_clear_is_not_blocked_or_penalized_by_previous_visits(self):
        world = World(small_parameters(revisit_penalty_s=120.))
        world.coverage.covered[:] = True
        certified_target(world, 1, [50., 0.])
        memory = ExplorationMemory(world.params)
        memory.observe('/measure', [40., 0.], 1, 'direction', 1)
        memory.observe('/measure', [0., 0.], 2, 'no_signal', 2)
        self.assertGreater(memory.penalty([40., 0.], channel=1), 0.)
        action, values = choose_next(world, memory)
        self.assertEqual(action['path'], '/clear')
        selected = next(row for row in values if row.get('selected'))
        self.assertEqual(selected['revisit_penalty_s'], 0.)

    def test_unknown_scans_are_charged_even_after_certified_clear(self):
        world = World(small_parameters())
        world.coverage.covered[:] = True
        world.coverage.covered[1] = False
        certified_target(world, 1, [100., 0.], radius=5.)
        action = candidate_actions(world, [])[0]
        evaluator = ActionEvaluator(world, ExplorationMemory(world.params))
        value = evaluator.value(action, n=8)
        self.assertEqual(value['unknown_channels'], [2])
        # Clearing needs 5 s, measuring unknown channel 2 needs 5 s + switch.
        minimum = np.linalg.norm(action['position'] - world.position) / 5 + 5 + 6
        self.assertGreaterEqual(value['service_rollout_s'] + 1e-9, minimum)

    def test_coverage_tail_is_public_pure_and_certifies_all_remaining_cells(self):
        world = World(small_parameters())
        for channel in range(1, 21):
            world.coverage.mark(channel, np.zeros(2))
        before = world.coverage.covered.copy()
        position = world.position.copy()
        stations = coverage_tail(world)
        self.assertTrue(stations)
        np.testing.assert_array_equal(world.coverage.covered, before)
        np.testing.assert_array_equal(world.position, position)
        for station in stations:
            for channel in range(1, 21):
                world.coverage.mark(channel, station)
        self.assertTrue(world.coverage.covered.all())

    def test_no_initial_sources_found_still_covers_map_with_scan_at_every_stop(self):
        simulator = Simulator(8702, sources=[])
        delegate = Client(simulator=simulator)

        class PublicClient:
            # No simulator, source list or truth interface is exposed to Planner.
            __slots__ = ()

            def command(self, *args, **kwargs):
                return delegate.command(*args, **kwargs)

        class AuditedPlanner(Planner):
            def __init__(self, client, parameters):
                super().__init__(client, parameters)
                self.scanned_stops = []

            def scan_here(self, *args, **kwargs):
                self.scanned_stops.append(tuple(self.world.position))
                return super().scan_here(*args, **kwargs)

        planner = AuditedPlanner(PublicClient(), small_parameters())
        result = planner.run()
        self.assertTrue(result['complete'], result.get('failure'))
        self.assertEqual(result['cleared'], 0)
        self.assertTrue(planner.world.coverage.covered.all())
        self.assertTrue(planner.world.finished())
        self.assertGreater(result['replans'], 0)
        self.assertFalse(any(t.status == 'active' for t in planner.world.targets.values()))
        stops = {tuple(row['position']) for row in planner.memory.visits if np.linalg.norm(row['position']) > 1e-7}
        self.assertTrue(stops.issubset(set(planner.scanned_stops)))
        seen = set()
        for command in planner.world.commands:
            if command['path'] != '/measure':
                continue
            identity = (command['channel'], *command['position'])
            self.assertNotIn(identity, seen)
            seen.add(identity)

    def test_boundary_sources_invisible_at_origin_are_all_discovered_and_cleared(self):
        sources = [Source(i + 1, 1800 * math.cos(i * math.tau / 10),
                          1800 * math.sin(i * math.tau / 10), 1000.) for i in range(10)]
        simulator = Simulator(8703, sources=sources)
        delegate = Client(simulator=simulator)

        class PublicClient:
            __slots__ = ()

            def command(self, *args, **kwargs):
                return delegate.command(*args, **kwargs)

        planner = Planner(PublicClient(), small_parameters())
        result = planner.run()
        self.assertTrue(result['complete'], result.get('failure'))
        origin = [command for command in planner.world.commands
                  if command['path'] == '/measure' and command['reason'] == 'origin_all_channels']
        self.assertEqual(len(origin), 20)
        self.assertTrue(all(command['response']['measure_result'] == 'no_signal' for command in origin))
        self.assertEqual(result['cleared'], 10)
        self.assertTrue(all(source.cleared for source in sources))
        self.assertTrue(all(planner.world.coverage.complete(channel) for channel in range(11, 21)))
        self.assertTrue(any(any(trigger['kind'] == 'new_source' for trigger in event['triggers'])
                            for event in planner.events if event['phase'] == 'global_replan'))


if __name__ == '__main__':
    unittest.main()

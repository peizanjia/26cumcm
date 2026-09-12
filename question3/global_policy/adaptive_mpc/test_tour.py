"""Bounded regression checks for the route-first controller's public contract."""
import unittest
import sys
from unittest.mock import patch

import numpy as np

from question3.local_sim.simulator import Client, Simulator, Source
from .anticipated_tail import anticipated_tail
from .planning import coverage_tail
from .sweep_planner import SweepParameters
from .tour_planner import Planner


def small_parameters():
    return SweepParameters(samples=32, search_scenarios=16, validation_scenarios=32,
                           refine_iterations=0, scan_scenarios=4, coverage_cell=150.,
                           frontier_spacing=500., early_geometry_mapping=False,
                           early_mapping=False, initial_map_probe=False,
                           unknown_scan_gain=.08, early_unknown_scan_gain=.08,
                           revisit_penalty_s=0.)


def public_client(simulator):
    delegate = Client(simulator=simulator)

    class PublicClient:
        __slots__ = ()

        def command(self, *args, **kwargs):
            return delegate.command(*args, **kwargs)

    return PublicClient()


class TourPlannerTests(unittest.TestCase):
    def test_feedback_reorders_route_before_finishing_first_source(self):
        """Controller sequencing uses real feedback, with routing/service stubs.

        Source 1's first operation is a probe and leaves it active. The next
        routing call prefers source 2. A stale route or complete-source service
        loop would therefore violate the required [1, 2, 1] execution sequence.
        """
        simulator = Simulator(9821, sources=[Source(1, 100., 0., 1000.),
                                             Source(2, 0., 150., 1000.)])
        planner = Planner(public_client(simulator), small_parameters())
        # This sequencing fixture begins with a complete absence certificate
        # for channels other than the two sources revealed by the origin scan.
        planner.world.coverage.covered[:] = True
        selected = []
        route_states = []

        def route_from_public_feedback(world, *args, **kwargs):
            route_states.append((world.targets[1].status, world.targets[1].radius,
                                 world.targets[2].status))
            if world.targets[1].radius > 20.:
                order = [1, 2]
            elif world.targets[2].status == 'active':
                order = [2, 1]
            else:
                order = [1]
            return 0., [world.targets[channel].center.tolist() for channel in order]

        def scripted_single_command(channel):
            selected.append(channel)
            if len(selected) == 1:
                planner.execute('/measure', np.array([100., 50.]), channel, 'fixture_probe')
            else:
                point = [0., 150.] if channel == 2 else [100., 0.]
                planner.execute('/clear', np.asarray(point), channel, 'fixture_clear')

        tour_module = sys.modules[Planner.__module__]
        with patch.object(tour_module, 'coverage_tail', return_value=[]), \
             patch.object(tour_module, 'anticipated_tail', side_effect=route_from_public_feedback) as route, \
             patch.object(planner, 'service_target', side_effect=scripted_single_command), \
             patch.object(planner, 'scan_here') as scan:
            result = planner.run()
        self.assertTrue(result['complete'], result['failure'])
        self.assertEqual(selected, [1, 2, 1])
        self.assertEqual(route.call_count, 3)
        self.assertEqual(scan.call_count, 3)
        self.assertEqual(route_states[1][0], 'active')
        self.assertLessEqual(route_states[1][1], 20.)
        self.assertEqual(route_states[1][2], 'active')

    def test_known_center_and_missing_coverage_share_a_pure_route(self):
        planner = Planner(None, small_parameters())
        world = planner.world
        world.targets[1].update_measurement(np.array([1200., 0.]),
                                            dict(measure_result='near'), world.params)
        world.coverage.covered[:] = True
        cell = int(np.argmin(np.linalg.norm(world.coverage.centers - [-1200., 0.], axis=1)))
        world.coverage.covered[2, cell] = False
        original_bits = world.coverage.covered.copy()
        original_position = world.position.copy()
        stations = coverage_tail(world)
        _, route = anticipated_tail(world, {1: 0.}, start=world.position, removed=None,
                                     y=world.position, unknown_channels=[], stations=stations)
        center = world.targets[1].center
        self.assertTrue(any(np.linalg.norm(np.asarray(point) - center) < 1e-5 for point in route))
        coverage_points = [point for point in route if world.coverage.gain(3, point) > 0.]
        self.assertTrue(coverage_points)
        self.assertEqual(planner._route_action([center])['kind'], 'known')
        coverage_action = planner._route_action([coverage_points[0], center])
        self.assertEqual(coverage_action['kind'], 'coverage')
        self.assertEqual(coverage_action['channel'], 3)
        np.testing.assert_array_equal(world.coverage.covered, original_bits)
        np.testing.assert_array_equal(world.position, original_position)
        self.assertFalse(world.coverage.complete(3))
        self.assertFalse(world.finished())

    def test_public_client_completes_empty_arena_with_actual_measurements(self):
        simulator = Simulator(9822, sources=[])
        client = public_client(simulator)
        self.assertFalse(hasattr(client, 'simulator'))
        self.assertFalse(hasattr(client, 'sources'))
        planner = Planner(client, small_parameters())
        result = planner.run()
        self.assertTrue(result['complete'], result['failure'])
        self.assertEqual(result['cleared'], 0)
        self.assertTrue(planner.world.coverage.covered.all())
        self.assertTrue(planner.world.finished())
        self.assertGreater(result['tour_replans'], 0)
        observations = set()
        for command in planner.world.commands:
            if command['path'] != '/measure':
                continue
            self.assertTrue(command['response']['accepted'])
            identity = (command['channel'], *command['position'])
            self.assertNotIn(identity, observations)
            observations.add(identity)
        disks = sum(len(disks) for disks in planner.memory.scan_disks.values())
        self.assertEqual(disks, len(observations))
        moved_stops = {tuple(visit['position']) for visit in planner.memory.visits
                       if np.linalg.norm(visit['position']) > 1e-7}
        scanned_stops = {tuple(event['position']) for event in planner.events if event['phase'] == 'stop_scan'}
        self.assertTrue(moved_stops.issubset(scanned_stops))


if __name__ == '__main__':
    unittest.main()

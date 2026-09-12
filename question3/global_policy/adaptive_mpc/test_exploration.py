import json
import unittest

import numpy as np

from ..dynamic.parameters import DynamicParameters
from ..model import ARENA_R, World
from .exploration import ExplorationMemory, frontier_candidates


class ExplorationTests(unittest.TestCase):
    def test_disks_are_channel_specific_and_travel_is_not_scanning(self):
        memory = ExplorationMemory()
        memory.observe('/measure', [0., 0.], 1, {'measure_result': 'no_signal'}, 1)
        memory.observe('/clear', [1600., 0.], 2, {'clear_result': 'fail'}, 2)
        self.assertEqual(len(memory.visits), 2)
        self.assertEqual(len(memory.segments), 1)
        self.assertEqual(len(memory.scan_disks[1]), 1)
        self.assertEqual(memory.scan_disks[1][0]['radius'], 1000.)
        self.assertEqual(memory.scan_disks[2], [])
        self.assertEqual(memory.visits[-1]['channels'], [])
        memory.observe('/measure', [1600., 0.], 2, {'measure_result': 'direction'}, 3)
        self.assertEqual(len(memory.scan_disks[2]), 1)
        self.assertEqual(memory.scan_disks[2][0]['center'], [1600., 0.])
        # The unmeasured midpoint is not added as another 1000 m disk.
        self.assertEqual(sum(map(len, memory.scan_disks.values())), 2)

    def test_remeasure_at_same_stop_does_not_accumulate_visits_or_penalty(self):
        memory = ExplorationMemory()
        for index, channel in enumerate([1, 1, 2, 3]):
            memory.observe('/measure', [0., 0.], channel, 'no_signal', index)
        self.assertEqual(len(memory.visits), 1)
        self.assertEqual(memory.visits[0]['channels'], [1, 2, 3])
        self.assertEqual(len(memory.scan_disks[1]), 1)
        self.assertEqual(memory.penalty([0., 0.]), 0.)
        saved = memory.snapshot()
        memory.observe('/measure', [0., 0.], 4, 'no_signal', 5)
        self.assertEqual(saved['visits'][0]['channels'], [1, 2, 3])
        json.dumps(saved, allow_nan=False)

    def test_rejected_commands_leave_no_visit_or_disk(self):
        memory = ExplorationMemory()
        memory.observe('/measure', [100., 0.], 1,
                       {'accepted': False, 'measure_result': 'no_signal'}, 1)
        self.assertEqual(memory.visits, [])
        self.assertEqual(memory.scan_disks[1], [])

    def test_revisits_are_soft_and_useful_revisits_can_be_exempt(self):
        memory = ExplorationMemory()
        memory.observe('/measure', [0., 0.], 1, 'no_signal', 1)
        memory.observe('/measure', [500., 0.], 1, 'no_signal', 2)
        cost = memory.penalty([20., 0.], channel=1)
        self.assertGreater(cost, 0.)
        self.assertLessEqual(cost, 15.)
        self.assertTrue(np.isfinite(cost))
        self.assertEqual(memory.penalty([20., 0.], channel=2), 0.)
        self.assertEqual(memory.penalty([20., 0.], useful=True), 0.)
        self.assertEqual(memory.penalty([500., 0.]), 0.)

    def test_frontiers_use_each_channels_actual_holes(self):
        world = World(DynamicParameters())
        world.coverage.covered[:] = True
        left = world.coverage.centers[:, 0] < -1500.
        right = world.coverage.centers[:, 0] > 1500.
        world.coverage.covered[0, left] = False
        world.coverage.covered[1, right] = False
        world.position = np.array([-1300., 0.])
        rows = frontier_candidates(world, ExplorationMemory(), limit=4)
        self.assertTrue(rows)
        self.assertIn(1, rows[0]['channels'])
        self.assertNotIn(2, rows[0]['channels'])
        for row in rows:
            mask = world.coverage.mask_at(row['position'])
            self.assertGreater(row['gain_cells'], 0)
            for channel in row['channels']:
                self.assertTrue(np.any(mask & ~world.coverage.covered[channel - 1]))
            self.assertLessEqual(np.linalg.norm(row['position']), ARENA_R)

    def test_tiny_boundary_gap_and_required_revisit_still_make_progress(self):
        world = World(DynamicParameters())
        world.coverage.covered[:] = True
        cell = int(np.argmax(np.linalg.norm(world.coverage.centers, axis=1)))
        world.coverage.covered[0, cell] = False
        center = world.coverage.centers[cell]
        self.assertGreater(np.linalg.norm(center), ARENA_R)
        q = center * ((ARENA_R - 1e-4) / np.linalg.norm(center))
        memory = ExplorationMemory()
        memory.observe('/measure', q, 2, 'no_signal', 1)
        memory.observe('/measure', [0., 0.], 2, 'no_signal', 2)
        rows = frontier_candidates(world, memory)
        self.assertTrue(rows)
        self.assertEqual(rows[0]['channels'], [1])
        self.assertEqual(rows[0]['closes'], 1)
        self.assertEqual(rows[0]['penalty_s'], 0.)
        world.coverage.mark(1, rows[0]['position'])
        self.assertTrue(world.coverage.complete(1))
        self.assertEqual(frontier_candidates(world, memory), [])


if __name__ == '__main__':
    unittest.main()

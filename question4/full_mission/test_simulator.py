"""Physical boundary and accounting checks for the independent Q4 simulator."""
import math
import unittest

import numpy as np

from question4.benchmark import make_scene
from question4.full_mission.simulator import Simulator, Source


def source(channel=1, x=0.0, y=0.0, radius=1000.0, directional=True, orientation=0.0):
    return Source(channel, x, y, radius, directional, orientation)


class SimulatorTests(unittest.TestCase):
    def test_closed_directional_halfplane_radius_and_near(self):
        sim = Simulator(sources=[source()])
        for position, expected in [((1000, 0), "direction"), ((1000.001, 0), "no_signal"),
                                   ((0, 1000), "direction"), ((0, -1000), "direction"),
                                   ((-1, 0), "no_signal"), ((5, 0), "near"),
                                   ((5.001, 0), "direction"), ((0, 0), "near")]:
            with self.subTest(position=position):
                self.assertEqual(sim.command("measure", position, 1)["measure_result"], expected)

    def test_rotated_halfplane_and_omni(self):
        for directional in (True, False):
            sim = Simulator(sources=[source(directional=directional, orientation=math.pi / 2)])
            self.assertEqual(sim.command("measure", (0, 600), 1)["measure_result"], "direction")
            self.assertEqual(sim.command("measure", (0, -600), 1)["measure_result"],
                             "no_signal" if directional else "direction")
            self.assertEqual(sim.command("measure", (700, 0), 1)["measure_result"], "direction")

    def test_backside_optical_clear_and_switch_accounting(self):
        sim = Simulator(sources=[source(channel=3)])
        sim.command("/measure", (-20, 0), 2)  # 4 move + 5 measure + 1 switch
        response = sim.command("/clear", (-20, 0), 3)  # 5 success, no switch
        self.assertEqual(response["clear_result"], "success")
        self.assertEqual(sim.channel, 2)
        self.assertEqual(sim.time_s, 15.0)
        self.assertEqual(sim.costs, dict(movement=4.0, measure=5.0, switch=1.0,
                                        clear_success=5.0, clear_failure=0.0))
        self.assertEqual(sim.command("measure", (0, 0), 3)["measure_result"], "no_signal")
        self.assertEqual(sim.command("clear", (0, 0), 3)["clear_result"], "no_target_in_range")
        self.assertEqual(sim.costs["clear_failure"], 3.0)

    def test_clear_radius_failure_does_not_remove_source(self):
        sim = Simulator(sources=[source()])
        self.assertEqual(sim.command("clear", (20.001, 0), 1)["clear_result"], "no_target_in_range")
        self.assertFalse(sim.truth()[0]["cleared"])
        self.assertEqual(sim.command("clear", (20, 0), 1)["clear_result"], "success")

    def test_fixed_bearing_error_and_rounding(self):
        sim = Simulator(8472, sources=[source(x=500, y=40, directional=False)])
        initial = sim.command("measure", (0, 0), 1)["svd_deg"]
        sim.command("measure", (100, 100), 2)
        repeat = sim.command("measure", (-0.0, 0.0), 1)["svd_deg"]
        self.assertEqual(initial, repeat)
        true_bearing = math.degrees(math.atan2(40, 500))
        self.assertLessEqual(abs((initial - true_bearing + 180) % 360 - 180), 1.005)
        self.assertEqual(initial, round(initial, 2))
        self.assertEqual(sim.error(1, (0, 0)), sim.error(1, (-0.0, 0)))
        self.assertNotEqual(sim.error(1, (0, 0)), sim.error(1, (0.01, 0)))

    def test_empty_channel_and_no_sensing_during_movement(self):
        sim = Simulator(sources=[source()])
        # The segment crosses the emission footprint but arrival is outside it.
        self.assertEqual(sim.command("measure", (2100, 0), 1)["measure_result"], "no_signal")
        self.assertEqual(sim.command("measure", (0, 0), 20)["measure_result"], "no_signal")

    def test_initial_twenty_scan_cost(self):
        sim = Simulator(sources=[])
        for channel in range(1, 21):
            sim.command("measure", (0, 0), channel)
        self.assertEqual(sim.time_s, 119.0)
        self.assertEqual(sim.costs["measure"], 100.0)
        self.assertEqual(sim.costs["switch"], 19.0)

    def test_history_and_cost_reconstruction(self):
        sim = Simulator(sources=[source(channel=4, x=100)])
        for action in [("measure", (12, 16), 4), ("clear", (12, 16), 4),
                       ("measure", (88, 16), 2), ("clear", (88, 16), 4)]:
            sim.command(*action)
        self.assertAlmostEqual(sum(sim.costs.values()), sim.time_s)
        cursor = 0.0
        for row in sim.history:
            self.assertAlmostEqual(cursor, row["time_before_s"])
            self.assertAlmostEqual(row["move_s"], math.dist(row["from"], row["position"]) / 5.0)
            cursor += row["move_s"] + row["action_s"] + row["switch_s"]
            self.assertAlmostEqual(cursor, row["end_time_s"])
            self.assertAlmostEqual(cursor, row["response"]["virtual_time_s"])

    def test_independent_scene_copies_and_truth(self):
        shared = [source()]
        a, b = Simulator(sources=shared), Simulator(sources=shared)
        a.command("clear", (0, 0), 1)
        self.assertFalse(shared[0].cleared)
        self.assertFalse(b.truth()[0]["cleared"])
        detached = b.truth()
        detached[0]["cleared"] = True
        self.assertFalse(b.truth()[0]["cleared"])
        expected = make_scene(20279200)
        generated = Simulator(20279200)
        self.assertEqual([(s.channel, s.x, s.y, s.radius, s.directional, s.orientation) for s in expected],
                         [(s.channel, s.x, s.y, s.radius, s.directional, s.orientation) for s in generated.sources])

    def test_invalid_commands_leave_state_unchanged(self):
        sim = Simulator(sources=[])
        for args in [("move", (0, 0), 1), ("measure", (math.nan, 0), 1),
                     ("clear", (2e6 + 1, 0), 1), ("measure", (0, 0), 1.5),
                     ("clear", (0, 0), True), ("measure", (0, 0), 21)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                sim.command(*args)
        self.assertEqual(sim.time_s, 0.0)
        self.assertEqual(sim.history, [])
        np.testing.assert_array_equal(sim.position, (0, 0))


if __name__ == "__main__":
    unittest.main()

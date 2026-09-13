"""Behavioral checks for the search-only zigzag benchmark.

Run from the repository root with:
    python -m unittest question4.zigzag_study.test_study -v
These tests exercise physical counterexamples, conservative certificates, and
the simulator's observable stopping/charging rules, not complete Q4 clearing.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from question4.benchmark import Source, measure
from question4.zigzag_study.routes import ring, zigzag
from question4.zigzag_study.study import (
    certificate_details, certify_cells, certify_domain, make_cells, prepare_route,
    raw_plan, simulate,
)


def source(x: float, y: float, theta: float = 0.0, channel: int = 1,
           radius: float = 1000.0, directional: bool = True) -> Source:
    return Source(channel, x, y, radius, directional, theta)


class PhysicsAndBlindSpots(unittest.TestCase):
    def test_hemisphere_inclusive_boundary_and_backside_near(self):
        s = source(0.0, 0.0)
        self.assertEqual(measure(s, (0.0, 1000.0)), "direction")
        self.assertEqual(measure(s, (0.0, -1000.0)), "direction")
        self.assertEqual(measure(s, (1000.0, 0.0)), "direction")
        self.assertEqual(measure(s, (1000.01, 0.0)), "no_signal")
        self.assertEqual(measure(s, (-1.0, 0.0)), "no_signal")
        self.assertEqual(measure(s, (1.0, 0.0)), "near")

    def test_boundary_outward_requires_external_peak(self):
        s = source(1800.0, 0.0)
        # Offset phases avoid measuring at the source itself.
        inner = ring(1799.0, 720, math.pi / 720)
        self.assertTrue(all(measure(s, tuple(p)) == "no_signal" for p in inner))
        self.assertEqual(measure(s, (1900.0, 0.0)), "direction")

    def test_large_outer_loop_requires_inner_valley(self):
        s = source(500.0, 0.0)
        self.assertEqual(measure(s, (0.0, 0.0)), "no_signal")
        self.assertTrue(all(measure(s, tuple(p)) == "no_signal" for p in ring(1900.0, 720)))
        self.assertEqual(measure(s, (990.0, 0.0)), "direction")

    def test_old_two_layer_has_radial_gap_despite_sample_success(self):
        s = source(925.0, 0.0)
        old_points = np.vstack(([0.0, 0.0], ring(900.0, 720), ring(1950.0, 720)))
        self.assertTrue(all(measure(s, tuple(p)) == "no_signal" for p in old_points))
        self.assertEqual(measure(s, (990.0, 0.0)), "direction")


class CoverageCertificates(unittest.TestCase):
    def test_twenty_five_point_staggered_annulus_certifies_continuous_disk(self):
        points = zigzag(990.0, 1900.0, 24)
        self.assertEqual(len(points), 25)
        self.assertTrue(certify_domain(points))

    def test_adding_old_two_layer_points_preserves_valid_zigzag_subset_proof(self):
        old_two_layer = np.vstack(([0.0, 0.0], ring(900.0, 24), ring(1950.0, 32)))
        proven_subset = zigzag(990.0, 1900.0, 24)
        union = np.vstack((old_two_layer, proven_subset))
        self.assertEqual(len(old_two_layer), 57)
        # Adding actual same-channel measurements cannot invalidate coverage,
        # even when the Delaunay triangulation of the union changes its edges.
        self.assertTrue(certificate_details(union)["certified"])

    def test_global_hull_alone_is_not_a_local_receive_certificate(self):
        far_points = np.vstack(([0.0, 0.0], ring(4000.0, 24)))
        self.assertFalse(certify_domain(far_points))
        cells = np.array([[[450.0, -25.0], [550.0, -25.0],
                           [550.0, 25.0], [450.0, 25.0]]])
        self.assertFalse(bool(certify_cells(far_points, cells)[0]))

    def test_repeating_same_measurement_cannot_create_coverage(self):
        repeated = np.tile([0.0, 0.0], (25, 1))
        self.assertFalse(certify_domain(repeated))
        cells = make_cells(side=175.0)
        self.assertFalse(bool(np.all(certify_cells(repeated, cells))))


class SearchExecution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = prepare_route({
            "id": "test_25_point_certificate",
            "points": zigzag(990.0, 1900.0, 24).tolist(),
        })

    def test_certified_zigzag_needs_no_repair_points(self):
        self.assertEqual(self.prepared["repair_count"], 0)
        self.assertEqual(self.prepared["certificate_prefix"], 25)

    def test_empty_scene_does_not_trigger_hidden_source_count_stopping(self):
        result = simulate([], self.prepared, keep_replay=True)
        self.assertEqual(result["executed_stops"], 25)
        self.assertEqual(result["observable_stop"], "geometry_certificate")
        self.assertEqual(result["measurements"], 500)
        self.assertIsNone(result["s_per_source"])
        self.assertGreater(result["time_s"], 119.0)

    def test_fifteen_found_sources_still_need_unknown_channel_certificate(self):
        scene = [source(0.0, 0.0, channel=c) for c in range(1, 16)]
        result = simulate(scene, self.prepared)
        self.assertEqual(result["found"], 15)
        self.assertEqual(result["executed_stops"], 25)
        self.assertEqual(result["observable_stop"], "geometry_certificate")
        self.assertGreater(result["time_s"], result["all_found_s"])

    def test_sixteen_found_sources_allow_observable_early_stop(self):
        scene = [source(0.0, 0.0, channel=c) for c in range(1, 17)]
        result = simulate(scene, self.prepared, keep_replay=True)
        self.assertEqual(result["observable_stop"], "sixteen_found")
        self.assertEqual(result["executed_stops"], 1)
        # The common origin initialization still scans every channel.
        self.assertEqual(result["measurements"], 20)
        self.assertEqual(result["stops"][0]["channels"], list(range(1, 21)))
        self.assertEqual(result["time_s"], 119.0)

    def test_origin_and_repeated_stop_have_exact_sensor_charges(self):
        plan = raw_plan({"points": [[0.0, 0.0], [0.0, 0.0], [300.0, 400.0]]})
        result = simulate([], plan, scan_mode="all", keep_replay=True)
        self.assertEqual(result["stops"][0]["end_s"], 119.0)
        self.assertEqual(result["stops"][1]["end_s"], 239.0)
        self.assertEqual(result["movement_s"], 100.0)
        self.assertEqual(result["measure_s"], 300.0)
        self.assertEqual(result["switch_s"], 59.0)
        self.assertEqual(result["time_s"], 459.0)
        self.assertEqual(result["time_s"], sum(result[k] for k in ("movement_s", "measure_s", "switch_s")))

    def test_already_found_channel_skips_later_search_scans(self):
        plan = raw_plan({"points": [[0.0, 0.0], [0.0, 0.0]]})
        result = simulate([source(0.0, 0.0)], plan, keep_replay=True)
        self.assertEqual(result["stops"][0]["channels"], list(range(1, 21)))
        self.assertEqual(result["stops"][1]["channels"], list(range(2, 21)))
        self.assertEqual(result["measurements"], 39)
        self.assertEqual(result["time_s"], 233.0)

    def test_passing_through_source_without_stopping_is_not_a_measurement(self):
        plan = raw_plan({"points": [[0.0, 0.0], [3000.0, 0.0]]})
        s = source(1500.0, 0.0)
        self.assertEqual(measure(s, (1500.0, 0.0)), "near")
        result = simulate([s], plan, keep_replay=True)
        self.assertEqual(result["found"], 0)
        self.assertEqual(result["missed_channels"], [1])
        self.assertEqual(result["observable_stop"], "plan_exhausted")

    def test_raw_legacy_plan_is_not_shortened_by_sixteen_detections(self):
        scene = [source(0.0, 0.0, channel=c) for c in range(1, 17)]
        plan = raw_plan({"points": [[0.0, 0.0], [300.0, 400.0]]})
        result = simulate(scene, plan, scan_mode="all")
        self.assertEqual(result["executed_stops"], 2)
        self.assertEqual(result["measurements"], 40)
        self.assertEqual(result["time_s"], 339.0)

    def test_fixed_scene_replay_is_reproducible(self):
        scene = [source(1800.0, 0.0), source(925.0, 0.0, channel=2)]
        a = simulate(scene, self.prepared, keep_replay=True)
        b = simulate(scene, self.prepared, keep_replay=True)
        self.assertEqual(a, b)
        self.assertEqual(a["found"], 2)
        self.assertAlmostEqual(a["s_per_source"], a["time_s"] / 2)


if __name__ == "__main__":
    unittest.main()

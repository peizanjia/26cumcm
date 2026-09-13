"""Continuous certificate retention, planned/observed separation and stopping."""
import unittest

import numpy as np

from question4.full_mission.planner import Parameters, Planner, SearchMap
from question4.full_mission.simulator import Simulator, Source
from question4.zigzag_study.routes import ring
from question4.zigzag_study.study import certify_domain


def base_plan():
    return np.vstack(([0., 0.], ring(980., 12), ring(1870., 12, -np.pi/12)))


class CoverageTests(unittest.TestCase):
    def test_valid_support_survives_a_failing_new_delaunay_mesh(self):
        base = base_plan()
        extra = np.array([1076.8205022343918, -934.2145246922546])
        # A real counterexample to treating one selected mesh as the complete
        # geometry: it changes after point insertion, although coverage cannot.
        self.assertTrue(certify_domain(base))
        self.assertFalse(certify_domain(np.vstack((base, extra))))
        coverage = SearchMap()
        self.assertTrue(coverage.certify_plan(base, remember=True))
        self.assertFalse(coverage.certified, 'A future plan is not an observation')
        coverage.mark(extra)
        for q in base[:-1]:
            coverage.mark(q)
        self.assertFalse(coverage.certified)
        coverage.mark(base[-1])
        self.assertTrue(coverage.certified)
        self.assertTrue(coverage.certify_plan(np.vstack((base, extra))))

    def test_deleting_unmeasured_proof_vertex_needs_a_new_proof(self):
        coverage = SearchMap()
        base = base_plan()
        self.assertTrue(coverage.certify_plan(base, remember=True))
        self.assertFalse(coverage.certify_plan(base[:-1], remember=True))
        self.assertFalse(coverage.certify_plan([[0., 0.]], remember=True))
        coverage.unseen[:] = False
        coverage.mark((0., 0.))
        self.assertFalse(coverage.certified, 'Finite angle bits cannot certify absence')

    def test_empty_joint_task_list_restores_finite_work_without_false_completion(self):
        sim = Simulator(sources=[])
        planner = Planner(sim.command)
        planner.coverage.mark((0., 0.))
        planner.stations = []
        action, phase, _ = planner.choose()
        self.assertEqual(phase, 'search')
        self.assertEqual(action['kind'], 'measure')
        self.assertTrue(planner.stations)
        self.assertFalse(planner.coverage.certified)
        self.assertFalse(planner.finished())
        self.assertEqual(planner.counters['coverage_fallbacks'], 1)

    def test_regression_seed_finishes_without_repair_or_uncleared_sources(self):
        sim = Simulator(20283000)
        planner = Planner(sim.command)
        result = planner.run()
        self.assertTrue(result['completed'])
        self.assertTrue(all(s.cleared for s in sim.sources))
        self.assertTrue(result['coverage_certified'])
        self.assertEqual(result['counters']['coverage_fallbacks'], 0)

    def test_sixteenth_hit_interrupts_unknown_sweep_without_marking_partial_stop(self):
        sources = [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)]
        sim = Simulator(sources=sources)
        planner = Planner(sim.command)
        for c in range(1, 16):
            planner.execute(dict(kind='measure', position=[0., 0.], channel=c), 'test')
        count_before = len(planner.commands)
        self.assertTrue(planner.scan_unknowns(force=True))
        self.assertEqual(len(planner.commands) - count_before, 1)
        self.assertEqual(len(planner.active()), 16)
        self.assertEqual(planner.coverage.points, [])
        self.assertFalse(planner.coverage.certified)

    def test_sixteen_discoveries_end_search_but_require_actual_clears(self):
        sources = [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)]
        sim = Simulator(sources=sources)
        planner = Planner(sim.command)
        for c in range(1, 17):
            planner.execute(dict(kind='measure', position=[0., 0.], channel=c), 'test')
        self.assertEqual(len(planner.active()), 16)
        self.assertFalse(planner.finished())
        self.assertFalse(planner.search_needed())
        count_before = len(planner.commands)
        self.assertFalse(planner.scan_unknowns(force=True))
        self.assertEqual(len(planner.commands), count_before)
        planner.prune()
        self.assertEqual(planner.stations, [])
        self.assertFalse(planner.coverage.certified)
        absent = [r for r in planner.snapshots() if r['channel'] > 16]
        self.assertTrue(all(r['status'] == 'absent' and r['absence_reason'] == 'source_count_upper_bound' for r in absent))
        for c in range(1, 17):
            planner.execute(dict(kind='clear', position=[0., 0.], channel=c), 'test')
        self.assertTrue(planner.finished())
        self.assertTrue(all(s.cleared for s in sim.sources))


if __name__ == '__main__':
    unittest.main()

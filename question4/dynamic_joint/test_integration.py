"""Behavioral checks for experiment integrity; mission cases added with policy."""
from __future__ import annotations

import copy
import unittest

from .evaluate import aggregate, old_parameters, paired, resolve_configs


def example(seed, name="dynamic_joint", value=100., completed=True):
    return dict(seed=seed, strategy=name, label=name, source_count=10, cleared=10 if completed else 1,
                completed=completed, per_source_s=value if completed else None,
                time_s=value * 10, costs=dict(movement=value * 10, measure=0., switch=0., clear_success=0., clear_failure=0.),
                counters={}, phases={}, search_after_last_clear_s=0.)


class EvaluationIntegrityTests(unittest.TestCase):
    def test_failed_short_run_cannot_improve_complete_mean(self):
        rows = [example(1, value=100.), example(2, value=.001, completed=False)]
        score = aggregate(rows, "dynamic_joint")
        self.assertIsNone(score["mean_per_source_s"])
        self.assertIsNone(score["pooled_per_source_s"])
        self.assertEqual(score["failed_count"], 1)
        self.assertEqual(score["completed_only_mean_per_source_s"], 100.)

    def test_pairing_aligns_seed_and_marks_incomplete_pairs(self):
        rows = [example(2, "a", 130.), example(1, "b", 110.), example(1, "a", 100.),
                example(2, "b", 125.), example(3, "a", 0., False), example(3, "b", 150.)]
        comparison = paired(rows, "a", "b")
        self.assertEqual(comparison["mean_difference_s"], -2.5)
        self.assertEqual(comparison["paired_count"], 3)
        self.assertEqual(comparison["complete_pairs"], 2)
        self.assertFalse(comparison["complete_group"])

    def test_frozen_baseline_cannot_be_overridden(self):
        prior = old_parameters()
        altered = copy.deepcopy(prior)
        altered["side_gain"] += .001
        with self.assertRaises(ValueError):
            resolve_configs({"old_joint": {"policy": "old_joint", "parameters": altered}})
        resolved = resolve_configs({"old_joint": {"policy": "old_joint"}})
        self.assertEqual(resolved["old_joint"]["parameters"], prior)


class MissionIntegrationTests(unittest.TestCase):
    def test_public_sixteen_discoveries_end_search_but_still_clear_all(self):
        from question4.full_mission.simulator import Simulator, Source
        from .planner import Parameters, Planner
        sources = [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)]
        simulator = Simulator(910, sources)
        policy = Planner(simulator.command, Parameters(refine_candidates=False))
        result = policy.run()
        self.assertTrue(result["completed"])
        self.assertEqual(result["cleared"], 16)
        self.assertTrue(all(source.cleared for source in simulator.sources))
        self.assertEqual(result["command_count"], 36)
        self.assertAlmostEqual(result["time_s"], 199.)
        self.assertEqual(result["costs"]["movement"], 0.)
        self.assertEqual(result["costs"]["clear_success"], 80.)
        self.assertEqual([r["channel"] for r in policy.commands[:20]], list(range(1, 21)))
        self.assertTrue(all(r["kind"] == "measure" for r in policy.commands[:20]))
        self.assertTrue(all(r["kind"] == "clear" for r in policy.commands[20:]))

    def test_recording_does_not_change_sixteen_source_actions(self):
        from question4.full_mission.simulator import Simulator, Source
        from .planner import Parameters, Planner
        sources = [Source(c, 0., 0., 1000., False, 0.) for c in range(1, 17)]
        histories = []
        for record in (False, True):
            simulator = Simulator(911, sources)
            policy = Planner(simulator.command, Parameters(refine_candidates=False), record=record)
            result = policy.run()
            histories.append([(r["kind"], r["channel"], r["position"], r["response"]) for r in policy.commands])
            self.assertEqual(len(policy.frames), result["command_count"] if record else 0)
        self.assertEqual(histories[0], histories[1])


if __name__ == "__main__":
    unittest.main()

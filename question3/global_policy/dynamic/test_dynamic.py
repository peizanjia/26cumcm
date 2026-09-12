import math
import unittest
from unittest.mock import patch
import numpy as np
from question3.local_sim.simulator import Simulator,Client,Source
from ..model import World,Target
from .parameters import DynamicParameters
from .frontier import Sweep,possible_cells,nearby_targets
from .local_rollout import choose_action
from .runner import Planner


class DynamicTests(unittest.TestCase):
    def test_optimizer_objective_penalizes_incomplete_cases(self):
        from .evaluate import evaluate_parameters
        with patch('question3.global_policy.dynamic.evaluate.run_local',return_value=({'complete':False,'average_time_s':None},None)):
            score,rows=evaluate_parameters(DynamicParameters(),[123])
            self.assertEqual(score,1100000);self.assertFalse(rows[0]['complete'])
        with self.assertRaises(ValueError):evaluate_parameters(DynamicParameters(),[])

    def test_shared_origin_does_not_assign_opposite_bearing_to_same_region(self):
        w=World(DynamicParameters());t=w.targets[1]
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0),w.params)
        sweep=Sweep(w)
        self.assertTrue(sweep.target_in_region(t,0))
        self.assertFalse(sweep.target_in_region(t,4))

    def test_boundary_cells_belong_to_both_adjacent_coverage_regions(self):
        w=World(DynamicParameters());s=Sweep(w)
        angle=s.heading+s.sign*s.width/2
        x=1700*np.array([math.cos(angle),math.sin(angle)])
        cells=np.all(np.abs(w.coverage.centers-x)<=w.coverage.half+1e-8,axis=1)
        self.assertTrue(cells.any())
        self.assertTrue(np.all(s.region_masks[0][cells]))
        self.assertTrue(np.all(s.region_masks[1][cells]))

    def test_frontier_responds_to_holes_and_is_not_a_ring(self):
        w=World(DynamicParameters());s=Sweep(w)
        for channel in range(1,21):w.coverage.mark(channel,[0,0])
        _,first,_=s.next_point()
        self.assertGreater(abs(np.linalg.norm(first)-1400),10)
        for channel in range(1,21):w.coverage.mark(channel,first)
        second=s.next_point()
        self.assertTrue(second is None or np.linalg.norm(second[1]-first)>1)

    def test_nearby_large_uncertainty_is_not_skipped(self):
        w=World(DynamicParameters());t=w.targets[1]
        t.status='active';t.center=np.array([300.,0]);t.radius=700.
        self.assertIn(t,nearby_targets(w))

    def test_geometry_cell_contains_boundary_truth_after_negative_measurement(self):
        w=World(DynamicParameters());t=w.targets[1];x=np.array([1799.,0.])
        t.update_measurement(np.zeros(2),dict(measure_result='no_signal'),w.params)
        q=np.array([900.,100.]);angle=math.degrees(math.atan2(*(x-q)[::-1]))
        t.update_measurement(q,dict(measure_result='direction',svd_deg=round(angle+1,2)%360),w.params)
        truth_cells=np.all(np.abs(w.coverage.centers-x)<=w.coverage.half+1e-8,axis=1)
        self.assertTrue(truth_cells.any());self.assertTrue(np.all(possible_cells(t,w.coverage)[truth_cells]))
        self.assertFalse(np.any(possible_cells(t,w.coverage)&w.coverage.mask_at([0,0])))

    def test_local_actions_compare_total_time_above_60(self):
        w=World(DynamicParameters());t=w.targets[1]
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0),w.params)
        path,q,name,values=choose_action(w,t)
        self.assertGreater(t.radius,60)
        self.assertTrue(any(v['name'].startswith('diagonal') for v in values))
        self.assertTrue(any(v['name']=='direct_measure' for v in values))
        costs=[v['expected_remaining_s'] for v in values]
        selected=next(v for v in values if v['name']==name)
        self.assertLessEqual(selected['expected_remaining_s'],min(costs)+w.params.action_margin_s+1e-9)

    def test_20_to_60_is_not_unconditional_measurement(self):
        w=World(DynamicParameters());t=w.targets[1];t.status='active'
        t.center=np.array([100.,0.]);t.radius=40
        t.polygon=np.array([[60.,0],[100.,-40],[140.,0],[100.,40]])
        t.particles=np.array([[100.,0.]]);t.weights=np.array([1.])
        self.assertEqual(choose_action(w,t)[0],'/clear')
        t.misses=w.params.max_speculative_clears
        self.assertEqual(choose_action(w,t)[0],'/measure')

    def test_three_is_a_planning_budget_not_a_false_geometry_guarantee(self):
        w=World(DynamicParameters());t=w.targets[1]
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0),w.params)
        _,_,_,values=choose_action(w,t,probe_count=3)
        self.assertFalse(any(v['name'].startswith('diagonal') for v in values))
        self.assertGreater(t.radius,20)

    def test_invalid_parameters(self):
        for args in [dict(sector_count=8.5),dict(rollout_samples=0),dict(nearby_distance=float('nan'))]:
            with self.assertRaises(ValueError):DynamicParameters(**args).validate()

    def test_end_to_end_and_closure_invariants_adversarial_layouts(self):
        cases=[
            [Source(i+1,1800*math.cos(i*2*math.pi/16),1800*math.sin(i*2*math.pi/16),1000.) for i in range(16)],
            [Source(i+1,2.,1.,1000.) for i in range(10)],
            [Source(i+1,1600+5*i,10+3*i,1000.) for i in range(10)]]
        for sources in cases:
            sim=Simulator(20260912,sources=sources)
            class PublicClient:
                def command(self,*a,**kw):return ClientProxy.command(*a,**kw)
            ClientProxy=Client(simulator=sim)
            class AuditedPlanner(Planner):
                def event(self,phase,**extra):
                    if phase=='coverage_frontier':
                        assert not nearby_targets(self.world)
                    if phase=='sector_closed':
                        assert self.sweep.can_close()
                    if self.sweep and self.sweep.closed:
                        for t in self.world.active():
                            assert not any(self.sweep.target_in_region(t,i) for i in self.sweep.closed)
                    super().event(phase,**extra)
            planner=AuditedPlanner(PublicClient());summary=planner.run()
            self.assertTrue(summary['complete'],summary['failure']);self.assertTrue(all(s.cleared for s in sim.sources))
            self.assertEqual(summary['closed_sectors'],list(range(8)))


if __name__=='__main__':unittest.main()

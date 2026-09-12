import unittest
import numpy as np
from question1.enclosing_circle import minimum_enclosing_circle
from ..model import Target,Observation
from .kernel import mec,exclude_disk,update,evaluate
from .parameters import JointParameters
from .planner import Planner,run_local
from .service import closest_certified


class JointTests(unittest.TestCase):
    def test_compiled_mec(self):
        rng=np.random.default_rng(77)
        for n in (1,2,3,4,8,15):
            for _ in range(12):
                points=rng.normal(size=(n,2))*500
                center,r=mec(points);reference=minimum_enclosing_circle(points)
                self.assertAlmostEqual(r,reference.radius,places=5)
                self.assertLess(np.linalg.norm(center-reference.center),1e-4)

    def test_negative_evidence_retains_remaining_points(self):
        poly=np.array([[-100.,-100.],[100.,-100.],[100.,100.],[-100.,100.]])
        rng=np.random.default_rng(4)
        for q in (np.array([100.,100.]),np.zeros(2),np.array([80.,0.])):
            result=exclude_disk(poly,q,80.)
            points=rng.uniform(-100,100,(2000,2));points=points[np.linalg.norm(points-q,axis=1)>=80]
            for a,b in zip(result,np.roll(result,-1,axis=0)):
                d=b-a;cross=d[0]*(points[:,1]-a[1])-d[1]*(points[:,0]-a[0])
                self.assertGreaterEqual(float(cross.min()),-1e-6)

    def test_miss_has_expected_cost_and_no_switch_on_success(self):
        poly=np.array([[95.,-5.],[105.,-5.],[105.,5.],[95.,5.]])
        x=np.array([[100.,0.]])
        costs,_=evaluate(poly,np.zeros(2),x[0],True,x,np.array([1200.]),np.zeros((1,16)),np.empty((0,2)),1,32)
        self.assertAlmostEqual(costs[0],25.)
        # Certain miss at origin, then already-certified safe clear, no measurement/switch.
        costs,_=evaluate(poly,np.zeros(2),np.zeros(2),True,x,np.array([1200.]),np.zeros((1,16)),np.empty((0,2)),1,32)
        self.assertLess(costs[0],28.)
        self.assertGreater(costs[0],20.)

    def test_projection_is_certified(self):
        t=Target(1);t.polygon=np.array([[-15.,0.],[15.,0.],[0.,5.]])
        c=minimum_enclosing_circle(t.polygon);t.center,t.radius=c.center,c.radius
        current=np.array([0.,100.]);q=closest_certified(t,current)
        self.assertLessEqual(np.max(np.linalg.norm(t.polygon-q,axis=1)),20+1e-8)
        self.assertLessEqual(np.linalg.norm(q-current),np.linalg.norm(t.robust_clear_point(current)-current)+1e-6)

    def test_baseline_exact_regression(self):
        row,_=run_local(20260911,JointParameters(linked_terminal=False,continuous_service=False,samples=256))
        self.assertTrue(row['complete'],row['failure'])
        self.assertAlmostEqual(row['average_time_s'],229.2241286875,places=5)

    def test_terminal_uses_future_without_mutating_sweep(self):
        from ..dynamic.frontier import Sweep
        planner=Planner(None);w=planner.world
        for c,point in [(1,[400.,0.]),(2,[0.,1000.])]:
            t=w.targets[c];t.status='active';t.center=np.array(point);t.radius=1.
            t.polygon=t.center+np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]])
        w.coverage.covered[:]=True;planner.sweep=Sweep(w);planner.sweep.index=0
        opts=planner.terminal_options([w.targets[1]])
        self.assertEqual(planner.sweep.index,0)
        self.assertTrue(opts);self.assertTrue(any(o['sector']>0 for o in opts))

    def test_simulated_observation_never_excludes_generator(self):
        poly=np.array([[-1800.,-1800.],[1800.,-1800.],[1800.,1800.],[-1800.,1800.]])
        rng=np.random.default_rng(172)
        for _ in range(60):
            x=rng.uniform(-900,900,2);q=rng.uniform(-1200,1200,2)
            result=update(poly,q,x,rng.uniform(1000,1500),rng.uniform(-1,1),32)
            self.assertGreater(len(result),0)
            center,r=mec(result)
            self.assertLessEqual(np.linalg.norm(x-center),r+1e-5)

    def test_continuous_selects_validation_minimum_without_side_effect(self):
        from ..model import World
        from .service import choose_action
        p=JointParameters(search_scenarios=16,validation_scenarios=32,refine_iterations=1)
        w=World(p);t=w.targets[1]
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0.),p)
        before=t.polygon.copy();path,q,reason,values=choose_action(w,t)
        self.assertLessEqual(np.linalg.norm(q),1800)
        chosen=next(v for v in values if v.get('selected'))
        self.assertEqual(chosen['expected_remaining_s'],min(v['expected_remaining_s'] for v in values))
        np.testing.assert_array_equal(before,t.polygon)
        self.assertEqual(len(t.observations),1)
        self.assertEqual(len(w.commands),0)


if __name__=='__main__':unittest.main()

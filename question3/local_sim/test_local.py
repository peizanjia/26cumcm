import json
import math
import unittest
import numpy as np
from .simulator import Simulator, Source, Client, random_scene
from .solve_single import prior, outcomes, likelihood, sweep_cost, fallback_points, Belief


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.sim = Simulator(123,sources=[Source(1,100,0,1000),Source(2,0,100,1500)])
        self.client = Client(simulator=self.sim)
        self.client.command('/enter')

    def payload(self,rid='test',**extra):
        return dict(arena_id='default',robot_id='local',request_id=rid,**extra)

    def test_generator_constraints(self):
        for seed in range(100):
            s=random_scene(seed)
            self.assertTrue(10<=len(s)<=16)
            self.assertEqual(len(s),len({x.channel for x in s}))
            self.assertTrue(all(1<=x.channel<=20 and math.hypot(x.x,x.y)<=1800 and 1000<=x.radius<=1500 for x in s))

    def test_document_time_example(self):
        for path,xy,c,t in [('/measure',(300,400),1,105),('/measure',(300,400),2,111),
                            ('/clear',(300,0),3,194),('/measure',(300,0),2,199)]:
            self.assertEqual(self.client.command(path,xy,c)['virtual_time_s'],t)

    def test_clear_boundaries_and_repeat(self):
        self.assertEqual(self.client.command('/clear',(79.999,0),1)['clear_result'],'no_target_in_range')
        self.assertEqual(self.client.command('/clear',(80,0),1)['clear_result'],'success')
        self.assertEqual(self.client.command('/clear',(100,0),1)['clear_result'],'no_target_in_range')
        self.assertEqual(self.client.command('/measure',(100,0),1)['measure_result'],'no_signal')

    def test_detection_boundary(self):
        for x,expected in [(95,'near'),(94.999,'direction'),(-900,'direction'),(-900.001,'no_signal')]:
            self.assertEqual(self.client.command('/measure',(x,0),1)['measure_result'],expected)

    def test_repeat_bearing(self):
        a=self.client.command('/measure',(10,20),1)
        self.client.command('/measure',(30,40),2)
        b=self.client.command('/measure',(10,20),1)
        self.assertEqual(a['svd_deg'],b['svd_deg'])

    def test_idempotent_conflict(self):
        p=self.payload(position=dict(x=0,y=0),channel=1)
        first=self.sim.handle('/measure',p)
        self.assertEqual(first,self.sim.handle('/measure',p))
        self.assertEqual(self.sim.virtual_time,5)
        self.assertEqual(self.sim.handle('/clear',p)[0],409)

    def test_invalid_does_not_mutate_or_reserve(self):
        p=self.payload(position=dict(x=float('nan'),y=0),channel=1)
        self.assertEqual(self.sim.handle('/measure',p)[0],400)
        p['position']['x']=0
        p['unknown']=1
        self.assertFalse(self.sim.handle('/measure',p)[1]['accepted'])
        del p['unknown']
        self.assertTrue(self.sim.handle('/measure',p)[1]['accepted'])
        self.assertEqual(self.sim.virtual_time,5)

    def test_clear_keeps_channel(self):
        self.client.command('/clear',(0,100),2)
        self.assertEqual(self.sim.channel,1)
        before=self.sim.virtual_time
        self.client.command('/measure',(0,100),1)
        self.assertEqual(self.sim.virtual_time-before,5)

    def test_unknown_source_not_disclosed(self):
        r=self.client.command('/measure',(0,0),20)
        self.assertEqual(set(r),{'accepted','real_timestamp_ms','virtual_time_s','measure_result'})

    def test_probability_partition(self):
        p=prior(1000,np.random.default_rng(5))
        for q in (np.array([500.,80.]),np.array([1400.,0.]),np.array([-500.,20.])):
            total=np.zeros(len(p))
            for _,r,bw in outcomes(p,q,4): total+=likelihood(p,q,r,bw or .01)
            np.testing.assert_allclose(total,1,atol=1e-12)

    def test_terminal_cost(self):
        p=np.array([[100.,0.,1000.],[100.,0.,1500.]])
        self.assertAlmostEqual(sweep_cost(p,np.array([.3,.7]),np.zeros(2)),25.)

    def test_geometric_fallback_covers_prior(self):
        p=prior(10000,np.random.default_rng(7))[:,:2]
        grid=np.array(list(fallback_points()))
        self.assertLess(np.linalg.norm(p[:,None,:]-grid[None,:,:],axis=2).min(axis=1).max(),20)

    def test_belief_failure_excludes_disk(self):
        b=Belief(1000)
        b.update(np.array([800.,0.]),dict(clear_result='no_target_in_range'))
        self.assertTrue(np.all(np.linalg.norm(b.p[b.w>0,:2]-[800,0],axis=1)>20))


if __name__=='__main__': unittest.main()

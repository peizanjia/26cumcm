import unittest
import numpy as np
from ..model import World
from .parameters import Parameters
from .routing import refine_route,mask,route_length,reorder


class StructuralTests(unittest.TestCase):
    def test_flexible_nodes_preserve_forecast_union_without_real_observations(self):
        w=World(Parameters(refine_coverage=True,exact_route=True))
        for c in range(1,21):w.coverage.mark(c,[0.,0.])
        angle=np.arange(6)*np.pi/3
        old=list(1500*np.column_stack((np.cos(angle),np.sin(angle))))
        bits=w.coverage.covered.copy()
        original=np.logical_or.reduce([mask(w,q,[]) for q in old])
        route,info=refine_route(w,old)
        after=np.logical_or.reduce([mask(w,q,[]) for q in route])
        self.assertTrue(np.all((bits[0]|after)[original]))
        np.testing.assert_array_equal(bits,w.coverage.covered)
        self.assertLessEqual(info['after_m'],info['before_m']+1e-4)
        self.assertTrue(all(np.linalg.norm(q)<=1800 for q in route))

    def test_exact_route_keeps_free_endpoint(self):
        start=np.array([0.,0.]);route=[np.array(q,dtype=float) for q in [(10,0),(0,10),(10,10)]]
        result=reorder(start,route,True)
        self.assertAlmostEqual(route_length(start,result),30.)

    def test_invalid_tuning_parameters_rejected(self):
        for values in [dict(defer_overlap=.1),dict(mapping_radius_ratio=2),dict(coupled_scan_weight=-1)]:
            with self.assertRaises(ValueError):Parameters(**values).validate()

    def test_disabled_structural_changes_match_previous_commands(self):
        from question3.local_sim.simulator import Simulator,Client,Source
        from ..adaptive_mpc.tour_planner import Planner as Old
        from ..adaptive_mpc.sweep_planner import SweepParameters
        from .planner import Planner
        fields=dict(search_scenarios=16,validation_scenarios=32,refine_iterations=0,samples=64,
                    early_geometry_mapping=True,early_unknown_scan_gain=.08,revisit_penalty_s=15.)
        runs=[]
        for cls,params in [(Old,SweepParameters(**fields)),(Planner,Parameters(**fields))]:
            simulator=Simulator(911,sources=[Source(1,650,100,1250),Source(2,-1300,0,1000)])
            plan=cls(Client(simulator=simulator),params);row=plan.run()
            self.assertTrue(row['complete'],row['failure'])
            runs.append([(c['path'],c['channel'],c['position']) for c in plan.world.commands])
        self.assertEqual(runs[0],runs[1])

    def test_improved_policy_clears_boundary_sources_using_only_public_commands(self):
        from question3.local_sim.simulator import Simulator,Client,Source
        from .planner import Planner
        angles=np.arange(10)*2*np.pi/10
        sources=[Source(i+1,float(1800*np.cos(angle)),float(1800*np.sin(angle)),1000.)
                 for i,angle in enumerate(angles)]
        simulator=Simulator(319,sources=sources);delegate=Client(simulator=simulator)
        class PublicClient:
            __slots__=()
            def command(self,*args,**kwargs):return delegate.command(*args,**kwargs)
        params=Parameters(refine_coverage=True,exact_route=True,coupled_scan_weight=.6,
                          search_scenarios=16,validation_scenarios=32,refine_iterations=0,
                          scan_scenarios=4,samples=64,coverage_cell=150.,frontier_spacing=500.)
        planner=Planner(PublicClient(),params);row=planner.run()
        self.assertTrue(row['complete'],row['failure'])
        self.assertTrue(all(source.cleared for source in sources))
        measured=set()
        for command in planner.world.commands:
            if command['path']=='/measure':
                key=(command['channel'],*command['position'])
                self.assertNotIn(key,measured);measured.add(key)
        for channel in range(11,21):self.assertTrue(planner.world.coverage.complete(channel))


if __name__=='__main__':unittest.main()

import itertools
import unittest
import numpy as np
from question3.local_sim.simulator import Source
from ..model import World
from .parameters import StudyParameters
from .routes import held_karp,plan_route
from .bounds import scene_bounds
from .information import scan_value
from .planner import run_local
from .graph_algorithms import length


class StudyTests(unittest.TestCase):
    def test_graph_heuristics_are_valid_orders_and_annealing_keeps_incumbent(self):
        w=World(StudyParameters());w.position=np.array([5.,9.]);rng=np.random.default_rng(8);targets=[]
        for channel,point in enumerate(rng.uniform(-1000,1000,(7,2)),1):
            t=w.targets[channel];t.status='active';t.center=point;targets.append(t)
        points=np.array([t.center for t in targets]);destination=np.array([-900.,200.]);costs={}
        for method in ('insertion_2opt','nearest','exact_dp','mst_2opt','christofides','annealing'):
            route=plan_route(w,targets,destination,method)
            self.assertEqual(sorted(route),list(range(1,8)))
            costs[method]=length(points,w.position,destination,[c-1 for c in route])
        self.assertLessEqual(costs['annealing'],costs['insertion_2opt']+1e-6)
        for cost in costs.values():self.assertGreaterEqual(cost+1e-6,costs['exact_dp'])

    def test_mapping_first_and_global_graph_complete(self):
        for p in [StudyParameters(global_mode='block_first',route_method='exact_open'),
                  StudyParameters(route_scope='global_guided',route_method='exact_open'),
                  StudyParameters(global_mode='global_graph',route_method='exact_open')]:
            row,planner=run_local(20260911,p)
            self.assertTrue(row['complete'],row['failure']);self.assertEqual(row['true_cleared'],16)
            if p.global_mode=='block_first':self.assertTrue(any(e['phase']=='frozen_block_order' for e in planner.events))
            if p.route_scope=='global_guided':self.assertTrue(any(e['phase']=='global_route_guidance' for e in planner.events))

    def test_no_frontier_can_use_free_end_instead_of_fictitious_return(self):
        from .planner import Planner
        class NoTransport:
            def command(self,*args):raise AssertionError('Pure planning must not send a command')
        planner=Planner(NoTransport(),StudyParameters(route_method='exact_dp',open_if_no_frontier=True))
        targets=[]
        for channel,x in [(1,100.),(2,300.)]:
            t=planner.world.targets[channel];t.status='active';t.center=np.array([x,0.]);targets.append(t)
        self.assertEqual(planner.route(targets,np.zeros(2),open_end=True),[1,2])
        self.assertEqual(planner.events[-1]['method'],'exact_open')

    def test_full_coverage_bound_counts_extra_absent_channel_scans(self):
        row=scene_bounds([Source(i+1,0,0,1000) for i in range(10)])
        self.assertAlmostEqual(row['full_absent_coverage_lower_s_per_source']-row['mandatory_origin_scan_lower_s_per_source'],15.)

    def test_dp_matches_all_orders_small_asymmetric_problem(self):
        rng=np.random.default_rng(7);n=5
        start=rng.random(n);edges=rng.random((n,n));end=rng.random(n)
        exact,order=held_karp(start,edges,end)
        cost=lambda r:start[r[0]]+sum(edges[a,b] for a,b in zip(r,r[1:]))+end[r[-1]]
        self.assertAlmostEqual(exact,min(cost(r) for r in itertools.permutations(range(n))))
        self.assertAlmostEqual(exact,cost(order))

    def test_neighborhood_lower_bound_and_oracle_have_different_meanings(self):
        row=scene_bounds([Source(1,100,0,1000),Source(2,300,0,1000)])
        self.assertEqual(row['relaxed_distance_m'],240.)
        self.assertEqual(row['oracle_center_distance_m'],300.)
        # True neighborhood route ends at x=280, between the relaxed bound and center route.
        self.assertLess(row['relaxed_distance_m'],280);self.assertLess(280,row['oracle_center_distance_m'])

    def test_bound_for_coincident_sources_is_only_clear_time(self):
        row=scene_bounds([Source(i+1,0,0,1000) for i in range(10)])
        self.assertEqual(row['general_lower_s_per_source'],5.)

    def test_repeated_measurement_has_no_new_information(self):
        w=World(StudyParameters());t=w.targets[1]
        t.update_measurement(w.position,dict(measure_result='direction',svd_deg=0),w.params)
        self.assertIsNone(scan_value(w,t))

    def test_no_signal_station_has_negative_local_value(self):
        w=World(StudyParameters());t=w.targets[1]
        t.update_measurement(w.position,dict(measure_result='direction',svd_deg=0),w.params)
        w.position=np.array([-1800.,0.]);value=scan_value(w,t)
        self.assertLess(value['net_saving_s'],0)

    def test_parameters(self):
        for kwargs in [dict(route_method='bad'),dict(inner_nodes=7.5),dict(voi_samples=0),dict(inner_probability=2)]:
            with self.assertRaises(ValueError):StudyParameters(**kwargs).validate()

    def test_baseline_reproduced_and_alternatives_complete(self):
        from ..dynamic.runner import run_local as dynamic_run
        baseline,reference=dynamic_run(20260911)
        for params in [StudyParameters(),StudyParameters(global_mode='inner_first'),StudyParameters(global_mode='adaptive_sector'),
                       StudyParameters(route_method='exact_dp',scan_mode='voi',scan_each_probe=True)]:
            row,planner=run_local(20260911,params)
            self.assertTrue(row['complete'],row['failure']);self.assertEqual(row['true_cleared'],16)
            if params==StudyParameters():
                # Historical absolute time3667.586059 is not portable between
                # numerical runtimes; compare unchanged strategy/actions here.
                self.assertEqual(row['virtual_time_s'],baseline['virtual_time_s'])
                actions=lambda p:[(c['path'],c['channel'],c['position']) for c in p.world.commands]
                self.assertEqual(actions(planner),actions(reference))


if __name__=='__main__':unittest.main()

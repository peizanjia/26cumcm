import math
import unittest
import numpy as np
from question3.local_sim.simulator import Simulator,Source,Client
from .model import Coverage,Parameters,Target,World,CLEAR_R
from .decisions import choose_clear_or_probe,localization_value,plan_cleanup_route
from .runner import Planner


class PolicyTests(unittest.TestCase):
    def test_origin_does_not_close_unknown_channels(self):
        w=World(Parameters());w.coverage.mark(1,[0,0])
        self.assertFalse(w.coverage.complete(1));self.assertFalse(w.finished())

    def test_hexagonal_ring_closes_continuous_cell_certificate(self):
        for phase in [0.,.173,math.pi/6]:
            coverage=Coverage(60);coverage.mark(1,[0,0])
            for a in phase+np.arange(6)*math.pi/3:
                coverage.mark(1,1400*np.array([math.cos(a),math.sin(a)]))
            self.assertTrue(coverage.complete(1))
            self.assertFalse(coverage.complete(2))

    def test_covered_cells_have_all_corners_in_detection_disk(self):
        coverage=Coverage(60);q=np.array([600,400])
        selected=coverage.centers[coverage.mask_at(q)]
        for sign in [[1,1],[1,-1],[-1,1],[-1,-1]]:
            self.assertTrue(np.all(np.linalg.norm(selected+coverage.half*np.array(sign)-q,axis=1)<=1000))

    def test_bearing_outer_polygon_contains_true_source(self):
        params=Parameters();target=Target(1);truth=np.array([1500.,-200.])
        for q,error in [(np.zeros(2),1.),(np.array([900.,100.]),-1.)]:
            a=math.degrees(math.atan2(truth[1]-q[1],truth[0]-q[0]))
            # The first point is outside 1500m for this truth: use a valid closer point.
            if np.linalg.norm(truth-q)>1500:q=q+np.array([100.,0.]);a=math.degrees(math.atan2(truth[1]-q[1],truth[0]-q[0]))
            target.update_measurement(q,dict(measure_result='direction',svd_deg=round((a+error)%360,2)),params)
            edges=np.roll(target.polygon,-1,axis=0)-target.polygon;delta=truth-target.polygon
            self.assertTrue(np.all(edges[:,0]*delta[:,1]-edges[:,1]*delta[:,0]>=-1e-6))

    def test_miss_retains_outer_polygon_and_excludes_particles(self):
        t=Target(1);p=Parameters()
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0.),p)
        poly=t.polygon.copy();q=t.center.copy()
        t.update_clear(q,dict(clear_result='no_target_in_range'),p)
        np.testing.assert_allclose(poly,t.polygon)
        self.assertEqual(t.misses,1)
        if t.particles is not None:self.assertTrue(np.all(np.linalg.norm(t.particles-q,axis=1)>20))

    def test_repeat_measurement_does_not_add_evidence(self):
        t=Target(1);p=Parameters();r=dict(measure_result='direction',svd_deg=1.)
        t.update_measurement(np.zeros(2),r,p);before=t.radius
        t.update_measurement(np.zeros(2),r,p)
        self.assertEqual(len(t.observations),1);self.assertEqual(t.radius,before)

    def test_zero_signal_keeps_known_target_active(self):
        t=Target(1);p=Parameters()
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0.),p)
        t.update_measurement(np.array([2000.,0.]),dict(measure_result='no_signal'),p)
        self.assertEqual(t.status,'active');self.assertEqual(t.exclusions[-1][2],'no_signal')

    def test_r60_does_not_imply_two_shots(self):
        w=World(Parameters());t=w.targets[1];t.status='active';t.center=np.zeros(2);t.radius=60.
        t.polygon=np.array([[-60.,0.],[0.,-60.],[60.,0.],[0.,60.]])
        t.particles=np.array([[0.,0.]]);t.weights=np.ones(1)
        t.misses=w.params.max_speculative_clears
        self.assertEqual(choose_clear_or_probe(w,t)[0],'/measure')

    def test_certified_clear_destination_covers_polygon(self):
        t=Target(1);t.center=np.array([100.,0.]);t.radius=10.
        t.polygon=np.array([[90.,0.],[100.,10.],[110.,0.],[100.,-10.]])
        q=t.robust_clear_point(np.zeros(2))
        self.assertLessEqual(np.linalg.norm(t.polygon-q,axis=1).max(),20.)

    def test_collinear_approach_still_has_value(self):
        w=World(Parameters());t=w.targets[1]
        t.update_measurement(np.zeros(2),dict(measure_result='direction',svd_deg=0.),w.params)
        self.assertGreater(localization_value(t,np.array([550.,0.]),w),0.)

    def test_count_bound_is_optional(self):
        w=World(Parameters())
        for c in range(1,17):w.targets[c].status='cleared'
        self.assertFalse(w.finished());w.params.count_bound_stop=True;self.assertTrue(w.finished())

    def test_threshold_validation(self):
        with self.assertRaises(ValueError):Parameters(trial_probability_min=1.1).validate()
        with self.assertRaises(ValueError):Parameters(ring_nodes=6.5).validate()

    def test_tuning_vector_roundtrip_and_default_objective(self):
        from .tune import SEARCH_SPACE,parameters_from_vector
        from .evaluate import evaluate_parameters
        base=Parameters();params=parameters_from_vector([getattr(base,k) for k in SEARCH_SPACE])
        self.assertEqual(base,params)
        score,rows=evaluate_parameters(params,[20267000])
        self.assertTrue(rows[0]['complete'])
        self.assertAlmostEqual(score,rows[0]['average_time_s'])

    def test_boundary_cluster_and_coincident_sources(self):
        cases=[]
        cases.append([Source(i+1,1800*math.cos(i*2*math.pi/16),1800*math.sin(i*2*math.pi/16),1000.) for i in range(16)])
        cases.append([Source(i+1,2.,1.,1000.) for i in range(10)])
        cases.append([Source(i+1,1600+5*i,10+3*i,1000.) for i in range(10)])
        for sources in cases:
            sim=Simulator(9182,sources=sources);planner=Planner(Client(simulator=sim))
            result=planner.run()
            self.assertTrue(result['complete'],result['failure'])
            self.assertTrue(all(s.cleared for s in sim.sources))

    def test_parameter_changed_ring_has_coverage_patch_fallback(self):
        sim=Simulator(8765);params=Parameters(ring_nodes=4,ring_radius=1700,coverage_cell=150)
        planner=Planner(Client(simulator=sim),params);result=planner.run()
        self.assertTrue(result['complete'],result['failure'])
        self.assertTrue(all(s.cleared for s in sim.sources))

    def test_hidden_truth_not_needed_by_planner(self):
        sim=Simulator(4567);client=Client(simulator=sim)
        class PublicOnly:
            def command(self,*a,**kw):return client.command(*a,**kw)
        planner=Planner(PublicOnly());result=planner.run()
        self.assertTrue(result['complete'],result['failure'])

    def test_official_port_and_unconfirmed_practice_block_before_client(self):
        import contextlib,io
        from unittest.mock import patch
        from .runner import main
        for flags in [['--mode','http','--base-url','http://127.0.0.1:2026'],
                      ['--mode','http','--http-context','practice']]:
            with patch('sys.argv',['runner',*flags]),patch('question3.global_policy.runner.HTTPClient') as http:
                with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as raised:main()
                self.assertEqual(raised.exception.code,2)
                http.assert_not_called()


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from .gaussian_process import GaussianProcess,expected_improvement
from .bayesian_search import fields_from_vector,vector,BOUNDS,scores


class SearchTests(unittest.TestCase):
    def test_gp_interpolates_observed_values_with_small_noise(self):
        X=np.array([[0.],[.3],[.6],[1.]])
        y=np.array([8.,3.,1.,6.])
        gp=GaussianProcess(X,y,np.full(4,1e-8))
        mu,sd=gp.predict(X)
        np.testing.assert_allclose(mu,y,atol=.003)
        self.assertTrue(np.all(sd>=0))
        self.assertLess(float(sd.max()),.03)

    def test_noise_increases_uncertainty_and_ei_values_improvement(self):
        X=np.array([[0.],[.5],[1.]])
        clean=GaussianProcess(X,[0,3,0],[1e-7]*3)
        noisy=GaussianProcess(X,[0,3,0],[5.]*3)
        self.assertGreater(noisy.predict(X)[1].mean(),clean.predict(X)[1].mean())
        ei=expected_improvement(np.array([-3.,3.]),np.array([.01,.01]),0.)
        self.assertGreater(ei[0],2.9);self.assertLess(ei[1],1e-8)

    def test_failure_never_becomes_fast_parameter(self):
        baseline=[dict(seed=1,complete=True,average_time_s=250.),dict(seed=2,complete=True,average_time_s=250.)]
        failed=[dict(seed=1,complete=True,average_time_s=100.),dict(seed=2,complete=False,average_time_s=1.)]
        row=scores(failed,baseline)
        self.assertGreater(row['objective'],1e5);self.assertFalse(row['complete'])

    def test_parameter_roundtrip_and_integer_bounds(self):
        for x in [np.zeros(len(BOUNDS)),np.ones(len(BOUNDS)),np.full(len(BOUNDS),.5)]:
            fields=fields_from_vector(x);values=vector(fields)
            self.assertTrue(np.all((values>=0)&(values<=1)))
            self.assertIsInstance(fields['early_geometry_stops'],int)
            self.assertEqual(fields['unknown_scan_gain'],fields['early_unknown_scan_gain'])


if __name__=='__main__':unittest.main()

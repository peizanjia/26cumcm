import math
import unittest
import numpy as np
from question1.enclosing_circle import minimum_enclosing_circle
from .optimize import EPS, FACTOR, Y, certificate, condition_polygon, initial_polygon, run_case


class CertificateTests(unittest.TestCase):
    def test_constructive_certificate(self):
        result=certificate([750,600])
        self.assertTrue(result['feasible'])
        self.assertLess(result['receive_bound_m'],1000)
        self.assertLess(result['first_radius_bound_m'],69.342)
        self.assertLess(result['final_radius_bound_m'],17.341)

    def test_sector_enclosing_disk(self):
        r=100
        theta=np.linspace(-EPS,EPS,101)
        for radius in np.linspace(0,r,51):
            points=radius*np.column_stack([np.cos(theta),np.sin(theta)])
            center=np.array([r/FACTOR,0])
            self.assertLessEqual(np.linalg.norm(points-center,axis=1).max(),r/FACTOR+1e-10)

    def test_three_bad_probes_do_not_localize(self):
        poly=initial_polygon()
        for q in ([100,1],[200,2],[300,3]):
            poly=condition_polygon(poly,np.array(q),0.)
            for g in ([700,0],[1400,0]):
                bearing=math.degrees(math.atan2(g[1]-q[1],g[0]-q[0]))
                self.assertLess(abs(bearing),1.)
        self.assertGreaterEqual(minimum_enclosing_circle(poly).radius,350.)

    def test_extreme_cases(self):
        for d in (5.0001,750.,1499.999):
            for a in (-1.,0.,1.):
                truth=np.array([d*math.cos(math.radians(a)),d*math.sin(math.radians(a)),1500.])
                for code in range(8):
                    errors=np.array([1 if code & (1<<i) else -1 for i in range(3)])
                    result=run_case([750,600],truth,errors)
                    self.assertTrue(result['success'])
                    self.assertLessEqual(result['probes'],3)
                    self.assertLessEqual(result['final_radius_m'],20)

    def test_prior_contains_sector_boundary(self):
        poly=initial_polygon()
        # Convex CCW polygon edge orientations include true sector endpoints.
        for a in [-EPS,0,EPS]:
            p=1500*np.array([math.cos(a),math.sin(a)])
            edges=np.roll(poly,-1,axis=0)-poly
            deltas=p-poly
            cross=edges[:,0]*deltas[:,1]-edges[:,1]*deltas[:,0]
            self.assertTrue(np.all(cross>=-1e-7))


if __name__=='__main__':unittest.main()

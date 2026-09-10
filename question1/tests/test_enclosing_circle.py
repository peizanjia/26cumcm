import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

from question1.enclosing_circle import minimum_enclosing_circle, diameter_circle_metrics


def test_equilateral_extreme():
    points = np.array([[0., 0.], [2., 0.], [1., np.sqrt(3)]])
    result = diameter_circle_metrics(points)
    assert result['failure']
    assert result['radius_ratio'] == pytest.approx(2/np.sqrt(3))
    assert result['mec_radius_m'] == pytest.approx(2/np.sqrt(3))
    assert len(result['mec_support_indices']) == 3


def test_rectangle_and_obtuse_triangle_are_covered():
    for points in ([[0,0],[4,0],[4,3],[0,3]], [[0,0],[4,0],[1,1]]):
        result = diameter_circle_metrics(points)
        assert not result['failure']
        assert result['radius_ratio'] == pytest.approx(1)


def test_more_vertices_does_not_imply_coverage():
    for m in (3,4,5,6,7,8):
        angle = np.arange(m)*2*np.pi/m
        p = np.column_stack((np.cos(angle),np.sin(angle)))
        result = diameter_circle_metrics(p)
        assert result['failure'] == bool(m%2)


def test_degenerate_and_translation():
    assert minimum_enclosing_circle([[1,2]]).radius == 0
    circle = minimum_enclosing_circle([[0,0],[2,0],[1,0]])
    np.testing.assert_allclose(circle.center,[1,0])
    assert circle.radius == pytest.approx(1)
    p = np.array([[0,0],[2,0],[1,np.sqrt(3)]])
    a = minimum_enclosing_circle(p)
    b = minimum_enclosing_circle(p*1000+[1e6,-1e6])
    np.testing.assert_allclose(b.center,a.center*1000+[1e6,-1e6])
    assert b.radius == pytest.approx(a.radius*1000)


@pytest.mark.parametrize('seed',range(30))
def test_against_independent_convex_optimization(seed):
    rng=np.random.default_rng(seed)
    p=rng.normal(size=(20,2))
    p=p[ConvexHull(p).vertices]
    circle=minimum_enclosing_circle(p)
    # Independent convex optimization of squared radius with analytic Jacobian.
    def constraints(z):
        return z[2]-np.sum((p-z[:2])**2,axis=1)
    def jac(z):
        return np.column_stack((2*(p-z[:2]),np.ones(len(p))))
    start_center=p.mean(axis=0)
    start=np.r_[start_center,np.max(np.sum((p-start_center)**2,axis=1))+1.]
    ref=minimize(lambda z:z[2],start,jac=lambda z:np.array([0.,0.,1.]),
                 constraints={'type':'ineq','fun':constraints,'jac':jac},
                 method='SLSQP',options={'ftol':1e-9,'maxiter':1000})
    assert ref.success
    assert circle.radius**2 == pytest.approx(ref.fun,abs=2e-7)
    assert np.max(np.linalg.norm(p-circle.center,axis=1)) <= circle.radius+1e-9
    result=diameter_circle_metrics(p)
    assert 1-1e-9 <= result['radius_ratio'] <= 2/np.sqrt(3)+1e-9


def test_bad_input():
    with pytest.raises(ValueError):
        minimum_enclosing_circle([])
    with pytest.raises(ValueError):
        minimum_enclosing_circle([[float('nan'),0]])


def test_valid_bearing_counterexample_reaches_bound():
    from question1.search_diameter_circle import symmetric_counterexample
    case=symmetric_counterexample()
    assert case['metrics']['radius_ratio']==pytest.approx(2/np.sqrt(3))
    assert case['polygon_inside_arena']
    assert all(abs(m['error_deg'])<=1 for m in case['measurements'])


def test_four_vertices_can_also_reach_bound():
    p=[[0,0],[.5,-.1],[1,0],[.5,np.sqrt(3)/2]]
    assert diameter_circle_metrics(p)['radius_ratio']==pytest.approx(2/np.sqrt(3))

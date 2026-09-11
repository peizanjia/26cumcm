import numpy as np
import pytest
from question1.enclosing_circle import minimum_enclosing_circle
from .core import mec, score, marginalized_cost


@pytest.mark.parametrize('seed',range(12))
def test_reference(seed):
    rng=np.random.default_rng(seed)
    p=rng.normal(size=(8,2))*rng.uniform(1,1500)
    assert mec(p)==pytest.approx(minimum_enclosing_circle(p).radius,rel=1e-8,abs=1e-7)


def test_equilateral_not_half_diameter():
    p=np.array([[0.,0.],[40.,0.],[20.,20*np.sqrt(3)]])
    assert mec(p)==pytest.approx(40/np.sqrt(3))
    assert mec(p)>20


def test_outcome_branches():
    p=np.array([[[0.,-10.],[100.,-10.],[100.,10.],[0.,10.]]])
    g=np.array([[[80.,0.]]]); e=np.zeros((1,1));r=np.ones((1,1))*1000;c=np.array([4])
    assert score(p,c,g,e,r,np.array([[80.,0.]]))[0,0]==0
    assert score(p,c,g,e,r,np.array([[2000.,0.]]))[0,0]==pytest.approx(mec(p[0]))
    assert score(p,c,g,e,r,np.array([[0.,0.]]))[0,0]==pytest.approx(mec(p[0]))


def test_marginalized_penalty_matches_radius_quadrature():
    p=np.array([[[0.,-10.],[300.,-10.],[300.,10.],[0.,10.]]]);c=np.array([4])
    g=np.array([[[200.,0.]]]);e=np.zeros((1,1));q=np.array([[1400.,-200.]])
    expected=marginalized_cost(p,c,g,e,q,True)[0]
    n=16384
    radii=(1000+500*(np.arange(n)+.5)/n)[None]
    rr=score(p,c,np.repeat(g,n,axis=1),np.zeros((1,n)),radii,q)[0]
    actual=np.mean(rr/20+4*np.maximum(rr/20-1,0)**2)+.15*np.linalg.norm(q[0])/1500
    assert actual==pytest.approx(expected,abs=.02)

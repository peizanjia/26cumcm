import numpy as np
import pytest
from question2.最小包围圆思路.core import score,mec
from .core import components
from .policy import candidate_strategy
from .robust import conservative_bounds


def fixture():
    p=np.array([[[0.,-10.],[300.,-10.],[300.,10.],[0.,10.]]])
    return p,np.array([4]),np.array([[[200.,0.]]]),np.zeros((1,1))


def test_target20_is_expected_absolute_deviation_not_deviation_of_mean():
    p,c,g,e=fixture();q=np.array([[1400.,-200.]])
    got=components(p,c,g,e,q)[0]
    n=32768;radii=(1000+500*(np.arange(n)+.5)/n)[None]
    r=score(p,c,np.repeat(g,n,axis=1),np.zeros((1,n)),radii,q)[0]
    assert got[0]==pytest.approx(np.abs(r-20).mean()/20,abs=.001)
    assert got[2]==pytest.approx(r.mean(),abs=.02)


def test_optical_radius_zero_has_literal_target20_penalty():
    p,c,g,e=fixture()
    result=components(p,c,g,e,np.array([[200.,0.]]))[0]
    assert result.tolist()==[1.,0.,0.,0.]


def test_repeated_measurement_is_not_independent():
    p,c,g,e=fixture();result=components(p,c,g,e,np.zeros((1,2)))[0]
    assert result[0]==pytest.approx(abs(mec(p[0])-20)/20)
    assert result[1]==1


def test_orthogonality_integrates_uniform_angle():
    p,c,g,e=fixture();q=np.array([[250.,-200.]])
    got=components(p,c,g,e,q)[0,1]
    a=np.arctan2(200.,-50.)
    eps=((np.arange(100000)+.5)/100000*2-1)*np.pi/180
    assert got==pytest.approx(np.mean(np.cos(a+eps)**2),abs=1e-10)


def test_mirror_geometry_and_guidance():
    p,c,g,e=fixture();q=np.array([[250.,-200.]])
    expected=components(p,c,g,e,q)
    reflected=components((p*np.array([1,-1]))[:,::-1].copy(),c,g*np.array([1,-1]),-e,q*np.array([1,-1]))
    assert np.allclose(expected,reflected,atol=1e-8)


def test_candidate_strategy_explicitly_returns_empty_feasible_set():
    result=candidate_strategy(0,0,0,[[5000,5000],[0,0]],events=32)
    assert not result['feasible_mask'].any()
    assert result['selected_point'] is None
    assert result['certified'] is False


def test_candidate_strategy_needs_no_true_target_input():
    result=candidate_strategy(1500,0,0,[[1650,-100],[1650,100]],events=32,probability=.5)
    assert result['metrics'].shape==(2,4)
    assert np.isfinite(result['metrics']).all()


def test_conservative_angular_bound_covers_dense_bearings():
    from question2.黑箱思路.geometry import second_cpu
    poly=np.array([[0.,0.],[100.,-1.75],[100.,1.75]])
    q=np.array([[70.,-35.]])
    bound=conservative_bounds(poly,q,bins=720)[0]
    radii=[mec(second_cpu(poly,q[0],a)) for a in np.linspace(-np.pi,np.pi,1000)]
    assert max(radii)<=bound+1e-7
    assert bound<=mec(poly)+1e-7

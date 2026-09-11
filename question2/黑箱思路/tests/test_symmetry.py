import importlib
import numpy as np
import pytest
import torch

mod=importlib.import_module("question2.黑箱思路.symmetry.policy")
env=importlib.import_module("question2.黑箱思路.environment")
obj=importlib.import_module("question2.黑箱思路.objective")


@pytest.mark.parametrize("scaled",[False,True])
def test_exact_reflection_and_side_swap(scaled):
    net=mod.TwoSidedNet(seed=55,boundary_scaled=scaled)
    p=torch.tensor([[0.,0.],[.5,.6],[.1,1.1],[-.7,-.3]])
    flip=p.new_tensor([1.,-1.])
    torch.testing.assert_close(net(p*flip),net(p).flip(-2)*flip,atol=0,rtol=0)
    assert (net(p)[:,0,1]<0).all() and (net(p)[:,1,1]>0).all()


def test_reflected_simulator_scores_match_with_reflected_random_draws():
    data=env.make_dataset(25,12,394)
    d=obj.tensor_dataset(data,"cpu")
    reflected={k:v.clone() for k,v in d.items()}
    for name in ("targets","polygons"):
        reflected[name][...,1]*=-1
    reflected["states"][:,4]*=-1
    reflected["errors"]*=-1
    q=torch.tensor([[.55,.22]]*25)
    a=obj.outcomes(d,q)["loss"]
    b=obj.outcomes(reflected,q*torch.tensor([1.,-1.]))["loss"]
    torch.testing.assert_close(a,b,rtol=.0005,atol=.04)


def test_public_bool_interface_and_rotation(tmp_path):
    net=mod.TwoSidedNet(26)
    path=tmp_path/"network.pt"
    torch.save(dict(state_dict=net.state_dict(),boundary_scaled=True),path)
    policy=mod.SidePolicy(path)
    choices=policy.options(0,0,0)
    assert choices[False][1]<0 and choices[True][1]>0
    assert policy(0,0,0,True)==choices[True]
    assert choices[False][0]==pytest.approx(choices[True][0])
    assert choices[False][1]==pytest.approx(-choices[True][1])
    with pytest.raises(TypeError):policy(0,0,0,1)
    q=policy(300,400,23,True)
    qrot=policy(-400,300,113,True)
    assert qrot==pytest.approx([-q[1],q[0]],abs=.001)

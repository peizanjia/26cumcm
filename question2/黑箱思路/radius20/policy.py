"""Point policy plus an explicitly empirical candidate feasible-set strategy."""
from pathlib import Path
import numpy as np
from ..symmetry.policy import SidePolicy
from ..geometry import first_polygon,DELTA
from ..environment import sample_posterior
from question2.最小包围圆思路.core import evaluate

ROOT=Path(__file__).parent


def load_policy(guided=True):
    return SidePolicy(ROOT/'checkpoints'/('orthogonal.pt' if guided else 'plain.pt'))


def pi(x,y,theta_deg,left=False,guided=True):
    if not isinstance(left,(bool,np.bool_)):raise TypeError('left must be boolean')
    return load_policy(guided).options(x,y,theta_deg)[bool(left)]


def candidate_strategy(x,y,theta_deg,candidates,events=2048,seed=40260911,probability=.95,validate=True):
    """Predictive sample feasible set, then minimize MAE20 inside it.

    Inputs contain no true target. Returns no selected point if grid has none.
    Candidate coordinates are global metres, and can extend beyond the arena.
    This is NOT a continuous or certified feasible domain.
    """
    if not np.isfinite([x,y,theta_deg]).all() or np.hypot(x,y)>1800:raise ValueError('first point must be in arena')
    if not 0<probability<=1 or events<2:raise ValueError('invalid probability/events')
    candidates=np.asarray(candidates,dtype=float)
    if candidates.ndim!=2 or candidates.shape[1]!=2 or not np.isfinite(candidates).all():raise ValueError('candidates must be finite (n,2)')
    t=np.deg2rad(theta_deg);rot=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
    p=np.array([x,y]);local_p=p@rot/1500
    g=sample_posterior(local_p[None],events,seed)[0]*1500
    rng=np.random.default_rng(seed+1);low=np.maximum(1000,np.linalg.norm(g,axis=1))
    poly=first_polygon(local_p,1024)*1500
    d=dict(polygons=poly[None],counts=np.array([len(poly)]),targets=g[None],
           errors=rng.uniform(-DELTA,DELTA,(1,events)),radii=(low+(1500-low)*rng.random(events))[None])
    values=[]
    for start in range(0,len(candidates),64):
        qq=(candidates[start:start+64]-p)@rot
        dd={k:np.repeat(v,len(qq),axis=0) for k,v in d.items()}
        r=evaluate(dd,qq)
        values.append(np.column_stack([r.mean(1),np.abs(r-20).mean(1),(r<=20).mean(1),r.max(1)]))
    z=np.concatenate(values) if values else np.empty((0,4))
    mask=z[:,2]>=probability
    ids=np.flatnonzero(mask)
    selected=None
    if len(ids):
        order=np.lexsort((np.linalg.norm(candidates[ids]-p,axis=1),z[ids,1]))
        selected=int(ids[order[0]])
    proposal=selected;validation=None
    if selected is not None and validate:
        check=candidate_strategy(x,y,theta_deg,candidates[selected:selected+1],
            events=max(4096,events),seed=seed+100000,probability=probability,validate=False)
        validation=dict(events=check['events'],seed=check['seed'],metrics=check['metrics'][0],
                        passes=check['selected_index'] is not None)
        if not validation['passes']:selected=None
    return dict(candidates=candidates,metrics=z,metric_columns=['mean_radius','mae20','p_le20','sample_max'],
        feasible_mask=mask,selected_index=selected,
        selected_point=None if selected is None else candidates[selected],
        proposal_index=proposal,independent_validation=validation,
        events=events,seed=seed,probability=probability,certified=False)

"""Continuous first-action SAA optimization, independent finalist evaluation."""
import math
import time
import numpy as np
from scipy.optimize import minimize
from .kernel import evaluate,exclude_disk,mec


def scenarios(target,n,seed):
    rng=np.random.default_rng(seed)
    u=(np.arange(n)+rng.random(n))/n
    ids=np.minimum(np.searchsorted(np.cumsum(target.weights),u),len(target.particles)-1)
    points=target.particles[ids].copy();lo=np.full(n,1000.);hi=np.full(n,1500.)
    for obs in target.observations:
        d=np.linalg.norm(points-obs.position,axis=1)
        if obs.result=='no_signal':hi=np.minimum(hi,d)
        else:lo=np.maximum(lo,d)
    return points,lo+rng.random(n)*np.maximum(0,hi-lo),rng.uniform(-1,1,(n,16))


def closest_certified(target,current):
    """Projection onto intersection of radius-20 disks about polygon vertices."""
    initial=target.robust_clear_point(current)
    if np.max(np.linalg.norm(target.polygon-current,axis=1))<=20-1e-7:return current.copy()
    result=minimize(lambda y: .5*np.sum((y-current)**2),initial,jac=lambda y:y-current,
        constraints=[dict(type='ineq',fun=lambda y:400-np.sum((target.polygon-y)**2,axis=1),
                          jac=lambda y:2*(target.polygon-y))],method='SLSQP',
        options={'maxiter':80,'ftol':1e-9})
    if result.success and np.max(np.linalg.norm(target.polygon-result.x,axis=1))<=20-1e-8:
        return result.x
    return initial


def choose_action(world,target,probe_count=0):
    started=time.perf_counter();p=world.params;current=world.position.copy()
    if target.radius<=20:
        q=closest_certified(target,current)
        return '/clear',q,'joint_certified_clear',[dict(action='/clear',name='certified_projection',
            destination=q.tolist(),expected_remaining_s=float(np.linalg.norm(q-current)/5+5),
            scenario_se_s=0.,hit_probability=1.,certified=True)]
    if target.particles is None:
        from ..dynamic.local_rollout import choose_action as old
        return old(world,target,probe_count)
    poly=target.polygon.copy()
    for q,r,kind in target.exclusions:
        poly=exclude_disk(poly,np.asarray(q),float(r))
    if not len(poly):poly=target.polygon.copy() # Conservative original if numerical exclusion fails.
    center,_=mec(poly)
    history=np.array([o.position for o in target.observations]).reshape(-1,2)
    seed=p.model_seed+1009*target.channel+31*len(history)+target.misses
    samples=scenarios(target,p.search_scenarios,seed+170003)
    switch=int(world.channel!=target.channel)
    candidates=[];lookup={}
    def add(y,clear=False,name='continuous_measure'):
        y=np.asarray(y,dtype=float)
        if not np.isfinite(y).all() or np.linalg.norm(y)>1800:return 1e8
        if not clear and target.measured_at(y):return 1e8
        if clear and (target.misses>=p.max_speculative_clears or target.hit_probability(y)<=0):return 1e8
        key=(clear,*np.round(y,5))
        if key in lookup:return candidates[lookup[key]]['search_mean']
        costs,_=evaluate(poly,current,y,clear,*samples,history,switch,p.circle_sides)
        value=float(costs.mean());lookup[key]=len(candidates)
        candidates.append(dict(q=y.copy(),clear=bool(clear),name=name,search_mean=value))
        return value
    direct=center.copy()
    while target.measured_at(direct):direct=direct+np.array([.05,0.])
    add(direct,name='direct_measure');direct_index=0
    delta=center-current;distance=max(1.,np.linalg.norm(delta));u=delta/distance;n=np.array([-u[1],u[0]])
    for fraction in (.2,.35,.5,.65,.8,1.):
        for lateral in (-280.,-140.,-70.,-35.,0.,35.,70.,140.,280.):
            add(current+fraction*delta+min(1.,distance/250)*lateral*n,name='grid_measure')
    # Retain original candidates as an incumbent family.
    for fraction in (p.probe_fraction_near,p.probe_fraction_far):
        for sign in (-1,1):add(current+fraction*delta+sign*min(p.probe_lateral,.3*distance)*n,name='legacy_measure')
    trial=[center,current,target.center]
    ids=np.linspace(0,len(target.particles)-1,min(48,len(target.particles)),dtype=int)
    points=target.particles[ids]
    ranking=sorted(range(len(points)),key=lambda i:target.hit_probability(points[i]),reverse=True)
    trial += [points[i] for i in ranking[:12]]
    for y in trial:add(y,True,'trial_clear')
    initial=list(candidates)
    # Powell refines both measurement and trial-clear locations in continuous 2D.
    for clear in (False,True):
        best=sorted([c for c in initial if c['clear']==clear],key=lambda c:c['search_mean'])[:2]
        for c in best:
            if p.refine_iterations:
                minimize(lambda y:add(y,clear,'refined_clear' if clear else 'refined_measure'),c['q'],
                         method='Powell',bounds=[(-1800,1800),(-1800,1800)],
                         options={'maxiter':p.refine_iterations,'maxfev':120,'xtol':.5,'ftol':1e-4})
    selected=sorted(range(len(candidates)),key=lambda i:candidates[i]['search_mean'])[:p.refine_finalists]
    selected += [direct_index]
    for clear in (False,True):
        family=[i for i,c in enumerate(candidates) if c['clear']==clear]
        if family:selected.append(min(family,key=lambda i:candidates[i]['search_mean']))
    selected=list(dict.fromkeys(selected))
    independent=scenarios(target,p.validation_scenarios,seed+910009)
    values=[];all_costs=[]
    for i in selected:
        c=candidates[i]
        costs,endpoints=evaluate(poly,current,c['q'],c['clear'],*independent,history,switch,p.circle_sides)
        all_costs.append(costs)
        values.append(dict(action='/clear' if c['clear'] else '/measure',name=c['name'],destination=c['q'].tolist(),
            expected_remaining_s=float(costs.mean()),scenario_se_s=float(costs.std(ddof=1)/math.sqrt(len(costs))),
            hit_probability=target.hit_probability(c['q']),search_expected_s=c['search_mean'],
            validation_samples=len(costs),search_samples=p.search_scenarios,
            guard_probability=float(np.mean(costs>=1e6)),predicted_end_position=endpoints.mean(axis=0).tolist()))
    reference=all_costs[selected.index(direct_index)]
    for costs,v in zip(all_costs,values):
        d=costs-reference;v['paired_delta_to_direct_s']=float(d.mean())
        v['paired_se_to_direct_s']=float(d.std(ddof=1)/math.sqrt(len(d)))
    winner=int(np.argmin([v['expected_remaining_s'] for v in values]))
    v=values[winner];v['selected']=True
    v['candidate_evaluations']=len(candidates);v['planning_wall_s']=time.perf_counter()-started
    v['search_catalog']=[dict(action='/clear' if c['clear'] else '/measure',destination=c['q'].tolist(),
                             expected_s=c['search_mean']) for c in candidates]
    return v['action'],np.asarray(v['destination']),'joint_'+v['name'],values

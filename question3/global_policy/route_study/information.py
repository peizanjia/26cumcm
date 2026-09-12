"""Local proxy value of an optional stationary measurement, in seconds."""
from dataclasses import replace
import numpy as np
from question1.enclosing_circle import minimum_enclosing_circle
from ..dynamic.local_rollout import posterior_scenarios,observe_polygon,continuation
from ..decisions import localization_value


def scan_value(world,target):
    q=world.position;p=world.params
    if target.radius<=20 or target.measured_at(q):return None
    scenarios=posterior_scenarios(target,replace(p,rollout_samples=p.voi_samples))
    if scenarios is None:return None
    points,radii,errors=scenarios;gains=[];new_radii=[]
    for i,(x,r) in enumerate(zip(points,radii)):
        seen={tuple(np.round(obs.position,8)):None for obs in target.observations}
        baseline=continuation(target.polygon.copy(),q,x,r,errors[i,1:],p.circle_sides,dict(seen))
        updated=observe_polygon(target.polygon.copy(),q,x,r,errors[i,0],p.circle_sides)
        seen[tuple(np.round(q,8))]=None
        after=continuation(updated,q,x,r,errors[i,1:],p.circle_sides,seen)
        gains.append(baseline-after)
        new_radii.append(float(minimum_enclosing_circle(updated).radius))
    gross=float(np.mean(gains));se=float(np.std(gains,ddof=1)/np.sqrt(len(gains)))
    cost=5+int(world.channel!=target.channel)
    # No credit is claimed for a possible future switch saved: conservative local proxy.
    return dict(channel=target.channel,radius=target.radius,gross_saving_s=gross,
                scan_cost_s=cost,net_saving_s=gross-cost,paired_se_s=se,
                score_s=gross-cost-p.voi_uncertainty_weight*se,
                expected_radius=float(np.mean(new_radii)),probability_r20=float(np.mean(np.array(new_radii)<=20)))


def rank_known_scans(world):
    eligible=[t for t in world.active() if t.radius>20 and not t.measured_at(world.position)]
    eligible.sort(key=lambda t:localization_value(t,world.position,world),reverse=True)
    rows=[]
    for i,target in enumerate(eligible):
        if i>=world.params.voi_candidates:
            rows.append(dict(channel=target.channel,reason='outside_computation_budget'));continue
        value=scan_value(world,target)
        rows.append(value if value is not None else dict(channel=target.channel,reason='no_posterior'))
    accepted=[r for r in rows if r.get('score_s',-1e20)>=world.params.voi_gain_threshold_s]
    accepted.sort(key=lambda r:r['score_s'],reverse=True)
    return [r['channel'] for r in accepted[:world.params.max_known_scans]],rows

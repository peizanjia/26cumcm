"""One-step sampled rollout; actions use public belief, never simulator truth."""
import math
import numpy as np
from question1.geometry import _clip, bearing_halfplanes
from question1.enclosing_circle import minimum_enclosing_circle
from ..model import CLEAR_R, ERROR_DEG, SPEED, clip_circle_outer


def observe_polygon(poly,q,x,receive_radius,error,sides):
    """Synthetic feedback for one posterior scenario, not an actual command."""
    distance=np.linalg.norm(x-q)
    if distance>receive_radius:return poly.copy()
    if distance<=5:return clip_circle_outer(poly,q,5,sides)
    bearing=round((math.degrees(math.atan2(*(x-q)[::-1]))+error)%360,2)%360
    a,b=bearing_halfplanes([q],[bearing],ERROR_DEG)
    for normal,offset in zip(a,b):poly=_clip(poly,normal,offset,1e-8)
    return clip_circle_outer(poly,q,1500,sides)


def certified_point(center,radius,current):
    delta=current-center;distance=np.linalg.norm(delta)
    return center+delta*min(1.,max(0.,20-radius-1e-6)/distance) if distance else center.copy()


def continuation(poly,current,x,receive_radius,errors,sides,seen):
    """Base policy: measure enclosing-circle center until a certified clear.

    Latent x generates feedback only. Choice of q depends exclusively on polygon.
    Same-position feedback is memoized for each scenario, as in the simulator.
    """
    cost=0.
    for step in range(12):
        circle=minimum_enclosing_circle(poly)
        if circle.radius<=CLEAR_R:
            q=certified_point(circle.center,circle.radius,current)
            return cost+np.linalg.norm(q-current)/SPEED+5
        q=circle.center.copy()
        key=tuple(np.round(q,8))
        # Avoid revisiting a fixed-error station if a degenerate MEC is unchanged.
        while key in seen:
            q=q+np.array([.05,0.]);key=tuple(np.round(q,8))
        cost+=np.linalg.norm(q-current)/SPEED+5
        if key not in seen:
            seen[key]=observe_polygon(poly,q,x,receive_radius,errors[step],sides)
        poly=seen[key].copy();current=q
        if not len(poly):return 1e6
    return 1e6  # Not a claimed finite-time proof for an arbitrary input polygon.


def posterior_scenarios(target,params):
    if target.particles is None:return None
    rng=np.random.default_rng(params.model_seed+target.channel*1009+len(target.observations)*31+target.misses)
    n=params.rollout_samples
    # Stratified weighted draws and common random numbers across all candidate actions.
    u=(np.arange(n)+rng.random(n))/n
    ids=np.minimum(np.searchsorted(np.cumsum(target.weights),u),len(target.particles)-1)
    points=target.particles[ids]
    lower=np.full(n,1000.);upper=np.full(n,1500.)
    for obs in target.observations:
        distance=np.linalg.norm(points-obs.position,axis=1)
        if obs.result=='no_signal':upper=np.minimum(upper,distance)
        else:lower=np.maximum(lower,distance)
    radii=lower+rng.random(n)*np.maximum(0,upper-lower)
    return points,radii,rng.uniform(-1,1,(n,14))


def choose_action(world,target,probe_count=0):
    p=world.params;current=world.position;center=target.center
    if target.radius<=CLEAR_R:
        q=target.robust_clear_point(current)
        return '/clear',q,'certified_clear',[]
    direct=center.copy()
    while target.measured_at(direct):direct=direct+np.array([.05,0.])
    candidates=[('/measure',direct,'direct_measure')]
    delta=center-current;distance=np.linalg.norm(delta)
    if distance>30 and probe_count<p.rollout_probe_limit:
        normal=np.array([-delta[1],delta[0]])/distance
        for fraction in (p.probe_fraction_near,p.probe_fraction_far):
            for sign in (-1,1):
                q=current+fraction*delta+sign*min(p.probe_lateral,.3*distance)*normal
                if np.linalg.norm(q)<=1800 and not target.measured_at(q):
                    candidates.append(('/measure',q,f'diagonal_{fraction}_{sign}'))
    if target.misses<p.max_speculative_clears:
        clear_points=[center]
        if target.particles is not None:
            # Weighted local 20m mass, no hidden location or clearance outcome.
            ids=np.linspace(0,len(target.particles)-1,min(24,len(target.particles)),dtype=int)
            trial=target.particles[ids]
            mass=np.array([target.hit_probability(q) for q in trial])
            clear_points.append(trial[int(np.argmax(mass))])
        for i,q in enumerate(clear_points):
            if target.hit_probability(q)>0 and not any(k=='clear_miss' and np.linalg.norm(q-v)<1e-7 for v,r,k in target.exclusions):
                candidates.append(('/clear',np.asarray(q),f'trial_clear_{i}'))
    scenarios=posterior_scenarios(target,p)
    if scenarios is None:return '/measure',direct,'direct_measure_no_posterior',[]
    points,radii,errors=scenarios;values=[]
    for path,q,name in candidates:
        costs=[]
        for i,(x,receive_radius) in enumerate(zip(points,radii)):
            travel=np.linalg.norm(q-current)/SPEED
            if path=='/clear' and np.linalg.norm(x-q)<=CLEAR_R:
                costs.append(travel+5);continue
            poly=target.polygon.copy()
            seen={tuple(np.round(obs.position,8)):None for obs in target.observations}
            if path=='/measure':
                action=5+int(world.channel!=target.channel)
                poly=observe_polygon(poly,q,x,receive_radius,errors[i,0],p.circle_sides)
                seen[tuple(np.round(q,8))]=poly.copy()
            else:
                # Base policy continues by measurement. Clears do not switch channels.
                action=3+int(world.channel!=target.channel)
            rest=continuation(poly,q,x,receive_radius,errors[i,1:],p.circle_sides,seen)
            costs.append(travel+action+rest)
        values.append(dict(action=path,name=name,destination=q.tolist(),
                           expected_remaining_s=float(np.mean(costs)),
                           scenario_se_s=float(np.std(costs,ddof=1)/math.sqrt(len(costs))),
                           hit_probability=target.hit_probability(q)))
    winner=int(np.argmin([v['expected_remaining_s'] for v in values]))
    # Small noisy estimated differences do not justify an extra diagonal deviation.
    if values[0]['expected_remaining_s']-values[winner]['expected_remaining_s']<p.action_margin_s:winner=0
    path,q,name=candidates[winner]
    return path,q,name,values

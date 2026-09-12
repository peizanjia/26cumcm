"""Compare interruptible actions using a common remaining-work time surrogate.

The selected source is completed only by the *simulated base continuation*;
the live controller executes one command and can select any other source next.
Unknown discoveries are not simulated by the coarse coverage tail.
"""
from copy import copy
import math
import numpy as np
from ..joint_rollout.kernel import evaluate, exclude_disk
from ..joint_rollout.service import scenarios, closest_certified
from ..decisions import localization_value
from .exploration import frontier_candidates, ExplorationMemory
from .scanning import known_scan_value


def open_route(start, points):
    """Nearest insertion / 2-opt open path, never an artificial return to start."""
    points=np.asarray(points,dtype=float).reshape(-1,2)
    if not len(points): return 0., []
    remaining=list(range(len(points))); order=[];q=np.asarray(start)
    while remaining:
        i=min(remaining,key=lambda j:float(np.linalg.norm(points[j]-q)))
        order.append(i);remaining.remove(i);q=points[i]
    def length(ids):
        route=np.vstack((start,points[ids]))
        return float(np.linalg.norm(np.diff(route,axis=0),axis=1).sum())
    value=length(order)
    for _ in range(3):
        changed=False
        for i in range(len(order)-1):
            for j in range(i+1,len(order)):
                trial=order[:i]+order[i:j+1][::-1]+order[j+1:]
                cost=length(trial)
                if cost<value-1e-6: value,order,changed=cost,trial,True
        if not changed:break
    return value,order


def coverage_tail(world):
    """Finite coarse covering stations from actual per-channel missing bits."""
    w=copy(world);w.coverage=copy(world.coverage)
    w.coverage.covered=world.coverage.covered.copy();memory=ExplorationMemory(world.params)
    stations=[]
    for _ in range(24):
        if not w.unknown():break
        options=frontier_candidates(w,memory,limit=1)
        if not options:break
        item=options[0];q=np.asarray(item['position'])
        stations.append(q.copy())
        for channel in item['channels']:w.coverage.mark(channel,q)
        w.position=q.copy()
    if w.unknown():raise RuntimeError('Coverage-tail construction made insufficient progress')
    return stations


def posterior_input(target):
    poly=target.polygon.copy()
    for q,r,_ in target.exclusions:
        updated=exclude_disk(poly,np.asarray(q),float(r))
        if len(updated):poly=updated
    history=np.array([o.position for o in target.observations]).reshape(-1,2)
    return poly,history


def candidate_actions(world, frontiers):
    """All known channels enter the competition; no fixed target commitment."""
    p=world.params;q=world.position;items=[]
    for t in world.active():
        if t.radius<=20:
            items.append(dict(channel=t.channel,path='/clear',position=closest_certified(t,q),name='certified_clear'))
            continue
        center=t.center.copy();delta=center-q;d=np.linalg.norm(delta)
        normal=np.array([-delta[1],delta[0]])/max(d,1.)
        trial=[(center,'center')]
        for fraction in (.5,.8):
            for side in (-1,1):
                trial.append((q+fraction*delta+side*min(70.,.3*d)*normal,'diagonal'))
        for dest,name in trial:
            if np.linalg.norm(dest)>1800 or t.measured_at(dest):continue
            items.append(dict(channel=t.channel,path='/measure',position=dest,name=name))
        if t.misses<p.max_speculative_clears:
            clear_points=[center]
            if t.particles is not None:
                pts=t.particles[np.linspace(0,len(t.particles)-1,min(16,len(t.particles)),dtype=int)]
                clear_points.append(max(pts,key=t.hit_probability))
            for dest in clear_points:
                if t.hit_probability(dest)>0 and not any(k=='clear_miss' and np.linalg.norm(dest-v)<1e-6 for v,_,k in t.exclusions):
                    items.append(dict(channel=t.channel,path='/clear',position=np.asarray(dest),name='trial_clear'))
    for frontier in frontiers:
        items.append(dict(channel=frontier['channels'][0],path='/measure',position=np.asarray(frontier['position']),
                          name='coverage_frontier',mandatory=list(frontier['channels'])))
    return items


def bounded_waypoint(world,action,early=False):
    """Value intermediate sensing BEFORE selection, never alter a scored action."""
    a=action.copy();q=np.asarray(a['position']);d=np.linalg.norm(q-world.position)
    if d<=world.params.max_waypoint_step:return a
    y=world.position+(q-world.position)*world.params.max_waypoint_step/d
    threshold=world.params.early_unknown_scan_gain if early else world.params.unknown_scan_gain
    known=[t for t in world.active() if not t.measured_at(y) and t.radius>20 and localization_value(t,y,world)>.05]
    unknown=[t for t in world.unknown() if not t.measured_at(y) and world.coverage.gain(t.channel,y)>=threshold]
    if known:channel=max(known,key=lambda t:localization_value(t,y,world)).channel
    elif unknown:channel=max(unknown,key=lambda t:world.coverage.gain(t.channel,y)).channel
    else:return a
    return dict(path='/measure',position=y,channel=channel,name='enroute_probe',mandatory=[],
                intended_destination=q.tolist(),intended_channel=a['channel'])


class ActionEvaluator:
    def __init__(self,world,memory,early=False):
        self.w=world;self.memory=memory;self.early=early
        self.stations=coverage_tail(world);self.cache={};self.route_cache={}
        self.residual={}
        for t in world.active():
            if t.radius<=20:self.residual[t.channel]=5.;continue
            if t.particles is None:self.residual[t.channel]=5.+t.radius/5;continue
            data=self.samples(t,world.params.joint_samples,0)
            center=t.center.copy()
            while t.measured_at(center):center=center+np.array([.05,0.])
            costs,_=evaluate(*self.poly_current(t,t.center,center),False,*data,
                             self.cache[t.channel][1],0,world.params.circle_sides)
            self.residual[t.channel]=float(np.mean(costs))

    def poly_current(self,t,current,dest):
        if t.channel not in self.cache:self.cache[t.channel]=posterior_input(t)
        return self.cache[t.channel][0],current,dest

    def samples(self,t,n,stage):
        if t.channel not in self.cache:self.cache[t.channel]=posterior_input(t)
        seed=self.w.params.model_seed+1009*t.channel+31*len(t.observations)+t.misses+700001+stage*100003
        return scenarios(t,n,seed)

    def unknown_channels(self,y,mandatory):
        w=self.w;p=w.params
        threshold=p.early_unknown_scan_gain if self.early else p.unknown_scan_gain
        if not p.stop_scans:
            return [t.channel for t in w.unknown() if t.channel in mandatory and not t.measured_at(y)]
        selected=[]
        for t in w.unknown():
            gain=w.coverage.gain(t.channel,y);remaining=float(np.mean(~w.coverage.covered[t.channel-1]))
            if not t.measured_at(y) and gain>0 and (t.channel in mandatory or gain>=threshold or
                    (remaining<=threshold and gain>=.85*remaining)):selected.append(t.channel)
        return selected

    def tail(self,start,removed,y,unknown_channels):
        if self.w.params.anticipate_coverage:
            from .anticipated_tail import anticipated_tail
            return anticipated_tail(self.w,self.residual,start,removed,y,unknown_channels,self.stations,self.early)
        w=self.w;bits=w.coverage.covered.copy()
        mask=w.coverage.mask_at(y)
        for c in unknown_channels:bits[c-1]|=mask
        points=[t.center for t in w.active() if t.channel!=removed]
        ids=[t.channel for t in w.unknown()];scan_s=0.
        for station in self.stations:
            cover=w.coverage.mask_at(station)
            needed=[c for c in ids if np.any(cover&~bits[c-1])]
            if not needed:continue
            points.append(station);scan_s+=6*len(needed)
            for c in needed:bits[c-1]|=cover
        length,order=open_route(start,points)
        residual=sum(v for c,v in self.residual.items() if c!=removed)
        return length/5+scan_s+residual, [np.asarray(points[i]).tolist() for i in order]

    def value(self,a,n,stage=0,side_scans=False):
        w=self.w;y=a['position'];t=w.targets[a['channel']];p=w.params
        unknown=self.unknown_channels(y,a.get('mandatory',()))
        travel=float(np.linalg.norm(y-w.position)/5)
        if t.status=='unknown':
            if t.channel not in unknown:unknown.append(t.channel)
            costs=np.array([travel+6*len(unknown)-int(w.channel in unknown)])
            ends=np.array([y]);removed=None
        elif t.radius<=20:
            costs=np.array([travel+5]);ends=np.array([y]);removed=t.channel
        elif t.particles is not None:
            costs,ends=evaluate(*self.poly_current(t,w.position,y),a['path']=='/clear',*self.samples(t,n,stage),
                               self.cache[t.channel][1],int(w.channel!=t.channel),p.circle_sides)
            removed=t.channel
        else:
            costs=np.array([travel+6+t.radius/5]);ends=np.array([t.center]);removed=t.channel
        if t.status=='active' and unknown:
            # These scans are NOT in the one-source rollout, including certified
            # clears and missing-particle fallbacks. Pay before crediting coverage.
            costs=costs+6*len(unknown)+int(t.radius>20)
        endpoint=ends.mean(axis=0)
        tail,route=self.tail(endpoint,removed,y,unknown)
        # Use E distance to first tail node, not distance of E endpoint.
        if route:
            node=np.array(route[0]);tail+=float(np.mean(np.linalg.norm(ends-node,axis=1))-np.linalg.norm(endpoint-node))/5
        benefit=0.;details=[]
        if side_scans and p.stop_scans and p.side_scan_channels:
            others=[v for v in w.active() if v.channel!=removed and not v.measured_at(y) and v.radius>20]
            others.sort(key=lambda v:localization_value(v,y,w),reverse=True)
            for other in others[:p.side_scan_channels]:
                row=known_scan_value(w,other,y,n=p.scan_scenarios)
                if row and row.get('valid',True):
                    net=row.get('net_saved_s',row.get('net_saving_s',0.))
                    threshold=p.early_scan_min_net_s if self.early else p.scan_min_net_s
                    adjusted=row.get('risk_adjusted_net_s',net)
                    credit=min(max(0.,net),max(0.,self.residual[other.channel]-5.)) if adjusted>threshold else 0.
                    benefit+=credit;details.append(dict(channel=other.channel,credited_s=credit))
        useful=(a['path']=='/clear' and t.radius<=20) or bool(unknown)
        penalty=self.memory.penalty(y,channel=t.channel,useful=useful)
        mean=float(costs.mean())+tail-benefit+penalty
        angular=0.
        if p.angular_bias_s and np.linalg.norm(w.position)>300 and np.linalg.norm(y-w.position)>500:
            delta=(getattr(self.memory,'sweep_sign',1)*(math.atan2(y[1],y[0])-math.atan2(w.position[1],w.position[0]))+math.pi)%(2*math.pi)-math.pi
            angular=p.angular_bias_s*max(0.,-delta)
        se=float(costs.std(ddof=1)/math.sqrt(len(costs))) if len(costs)>1 else 0.
        return dict(channel=t.channel,action=a['path'],name=a['name'],destination=y.tolist(),
            expected_remaining_s=mean,service_rollout_s=float(costs.mean()),tail_s=tail,
            side_scan_credit_s=benefit,side_scan_values=details,revisit_penalty_s=penalty,
            angular_bias_s=angular,scenario_se_s=se,guard_probability=float(np.mean(costs>=1e6)),
            unknown_channels=unknown,predicted_end_position=endpoint.tolist(),tail_route=route,
            value_kind='full_source_base_rollout_plus_remaining_work_proxy')


def choose_next(world,memory,previous_channel=None,early=False):
    frontiers=frontier_candidates(world,memory,world.params.frontier_candidates)
    original=candidate_actions(world,frontiers)
    actions=original+[bounded_waypoint(world,a,early) for a in original]
    unique={}
    for a in actions:
        key=(a['path'],a['channel'],*np.round(a['position'],5))
        if key not in unique or a.get('mandatory'):unique[key]=a
    actions=list(unique.values())
    if not actions:raise RuntimeError('Incomplete state without a productive action')
    evaluator=ActionEvaluator(world,memory,early)
    coarse=[evaluator.value(a,world.params.joint_samples) for a in actions]
    rank=lambda i:coarse[i]['expected_remaining_s']+coarse[i]['angular_bias_s']
    ids=sorted(range(len(actions)),key=rank)[:world.params.joint_finalists]
    # Always retain the best available coverage action and previous-channel action.
    for predicate in (lambda a:a['name']=='coverage_frontier',lambda a:a['channel']==previous_channel):
        family=[i for i,a in enumerate(actions) if predicate(a)]
        if family:ids.append(min(family,key=rank))
    ids=list(dict.fromkeys(ids))
    values=[evaluator.value(actions[i],world.params.joint_validation_samples,1,True) for i in ids]
    for row in values:
        row['switch_hysteresis_s']=world.params.target_switch_margin_s if previous_channel is not None and row['channel']!=previous_channel else 0.
        row['selection_score_s']=row['expected_remaining_s']+row['switch_hysteresis_s']+row['angular_bias_s']
    best=min(range(len(values)),key=lambda i:values[i]['selection_score_s'])
    values[best]['selected']=True
    result=actions[ids[best]].copy()
    result['predicted_unknown_channels']=values[best]['unknown_channels']
    return result,values

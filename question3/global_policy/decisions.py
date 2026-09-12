"""Pure, parameterized decision functions. No network calls or hidden truth."""
import math
import numpy as np
from .model import CLEAR_R, ERROR_DEG, RECEIVE_MIN, RECEIVE_MAX, SPEED


def angle_difference(a,b):return (a-b+math.pi)%(2*math.pi)-math.pi


def choose_dense_heading(world):
    bearings=[]
    for target in world.active():
        directions=[o for o in target.observations if o.result=='direction']
        if directions:bearings.append(math.radians(directions[0].bearing))
    if not bearings:return 0.
    angles=np.arange(world.params.heading_candidates)*2*math.pi/world.params.heading_candidates
    delta=angle_difference(angles[:,None],np.asarray(bearings)[None,:])
    sigma=math.radians(world.params.density_sigma_deg)
    scores=np.exp(-.5*(delta/sigma)**2).sum(axis=1)
    return float(angles[np.argmax(scores)])


def localization_value(target,q,world):
    """Heuristic gain mixes parallax and certified approach contraction."""
    if target.status!='active' or target.radius<=CLEAR_R or target.measured_at(q):return 0.
    vector=target.center-q;distance=max(1.,np.linalg.norm(vector))
    directions=[o for o in target.observations if o.result=='direction']
    cross=0.
    for obs in directions:
        previous=target.center-obs.position
        denominator=max(1e-9,np.linalg.norm(previous)*distance)
        sine=(previous[0]*vector[1]-previous[1]*vector[0])/denominator
        cross=max(cross,float(sine*sine))
    max_distance=float(np.linalg.norm(target.polygon-q,axis=1).max())
    contracted=max_distance/(2*math.cos(math.radians(ERROR_DEG)))
    approach=max(0.,1-contracted/max(target.radius,1.)) if max_distance<=RECEIVE_MIN else 0.
    if target.particles is not None:
        d=np.linalg.norm(target.particles-q,axis=1)
        # Proxy: radius marginal is not a full learned model of official cases.
        visibility=float(target.weights@np.clip((RECEIVE_MAX-d)/(RECEIVE_MAX-RECEIVE_MIN),0,1))
    else:visibility=float(max_distance<=RECEIVE_MIN)
    p=world.params
    return visibility*max(p.information_weight*cross,p.proximity_weight*approach)*min(4.,target.radius/60)


def choose_initial_probe(world):
    heading=choose_dense_heading(world);world.dense_heading=heading
    along=np.array([math.cos(heading),math.sin(heading)])
    normal=np.array([-along[1],along[0]])
    candidates=[world.params.initial_step*along+sign*world.params.initial_lateral*normal for sign in (-1,1)]
    # Small lateral motion selected jointly for several known targets.
    scores=[sum(localization_value(t,q,world) for t in world.active()) for q in candidates]
    return candidates[int(np.argmax(scores))]


def select_known_channels(world,q,heading,exclude=()):
    forward=np.array([math.cos(heading),math.sin(heading)])
    scored=[]
    for target in world.active():
        if target.channel in exclude:continue
        score=localization_value(target,q,world)
        if target.center is not None and np.dot(target.center-q,forward)<0:
            score*=world.params.rear_scan_weight
        cost=5+(target.channel!=world.channel)
        score/=cost
        if score>=world.params.known_score_min:scored.append((score,target.channel))
    scored.sort(reverse=True)
    return [c for _,c in scored[:world.params.max_known_scans]]


def select_unknown_channels(world,q,force=False):
    scores=[]
    for target in world.unknown():
        if target.measured_at(q):continue
        gain=world.coverage.gain(target.channel,q)
        if force or world.params.coverage_weight*gain>=world.params.unknown_gain_min:
            scores.append((world.params.coverage_weight*gain/(5+(target.channel!=world.channel)),target.channel))
    scores.sort(reverse=True)
    return [c for _,c in scores]


def choose_ring_route(world):
    p=world.params
    candidates=[]
    for sign in (-1,1):
        angles=world.dense_heading+sign*np.arange(p.ring_nodes)*2*math.pi/p.ring_nodes
        points=p.ring_radius*np.column_stack([np.cos(angles),np.sin(angles)])
        # Same coverage, prefer early access to currently known target clusters.
        score=0.
        for target in world.active():
            d=np.linalg.norm(points-target.center,axis=1)
            nearest=int(np.argmin(d))
            score+=(nearest+1)*np.clip(target.radius/60,.5,3)+d[nearest]/1000
        candidates.append((score,points))
    return min(candidates,key=lambda x:x[0])[1]


def cleanup_destination(target,current):
    return target.robust_clear_point(current) if target.radius<=CLEAR_R else target.center.copy()


def path_length(points):
    return float(sum(np.linalg.norm(b-a) for a,b in zip(points,points[1:])))


def plan_cleanup_route(world,destination,force=False):
    """Greedy minimum-insertion detour, followed by endpoint-fixed 2-opt."""
    candidates=[t for t in world.active() if force or t.radius<=max(world.params.service_radius,world.params.probe_service_radius)]
    route=[];nodes=[world.position.copy(),np.asarray(destination).copy()]
    base=path_length(nodes)
    while candidates and len(route)<world.params.route_batch:
        best=None
        for target in candidates:
            q=cleanup_destination(target,world.position)
            for i in range(len(nodes)-1):
                extra=np.linalg.norm(nodes[i]-q)+np.linalg.norm(q-nodes[i+1])-np.linalg.norm(nodes[i]-nodes[i+1])
                limit=world.params.max_detour if target.radius<=world.params.service_radius else world.params.probe_detour_max
                if not force and (extra/world.params.cleanup_weight>limit or
                    (path_length(nodes)+extra-base)/world.params.cleanup_weight>world.params.route_detour_budget):continue
                # Service effort estimates stop detours for poorly localized distant targets.
                cost=(extra/SPEED+5+5*max(0,math.log2(max(1,target.radius/20))))/world.params.cleanup_weight
                if best is None or cost<best[0]:best=(cost,target,i,q)
        if best is None:break
        _,target,i,q=best
        nodes.insert(i+1,q);route.insert(i,target.channel);candidates.remove(target)
    improved=True
    while improved and len(route)>2:
        improved=False
        for i in range(1,len(nodes)-2):
            for j in range(i+1,len(nodes)-1):
                trial=nodes[:i]+list(reversed(nodes[i:j+1]))+nodes[j+1:]
                if path_length(trial)+1e-8<path_length(nodes):
                    nodes=trial;route[i-1:j]=reversed(route[i-1:j]);improved=True
    return route


def choose_clear_or_probe(world,target):
    """r60 admits service, not a two-shot guarantee."""
    if target.radius<=CLEAR_R:
        return '/clear',target.robust_clear_point(world.position),'certified_20m'
    q=target.center.copy()
    probability=target.hit_probability(q)
    # Repeating a failed clear at the same point cannot add information.
    repeated=any(kind=='clear_miss' and np.linalg.norm(q-old)<1e-6 for old,_,kind in target.exclusions)
    if (target.radius<=world.params.service_radius and not repeated
        and target.misses<world.params.max_speculative_clears
        and probability>=world.params.trial_probability_min):
        return '/clear',q,f'speculative_p={probability:.3f}'
    return '/measure',q,'reduce_uncertainty'


def choose_coverage_patch(world):
    """Fill any actual residual certificate holes, including after parameter changes."""
    unknown=world.unknown()
    if not unknown:return None
    counts=sum((~world.coverage.covered[t.channel-1]).astype(int) for t in unknown)
    ids=np.flatnonzero(counts)
    # Candidate pruning affects travel efficiency, never the closure test.
    ids=ids[np.argsort(-counts[ids])[:128]]
    best=None
    for i in ids:
        q=world.coverage.centers[i]
        gain=sum(np.count_nonzero(world.coverage.mask_at(q)&~world.coverage.covered[t.channel-1]) for t in unknown)
        score=gain/(np.linalg.norm(q-world.position)/SPEED+6*len(unknown))
        if best is None or score>best[0]:best=(score,q.copy())
    return best[1]

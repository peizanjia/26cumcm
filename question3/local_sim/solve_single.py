"""One conditional single-source scenario: finite-belief tree rollout optimization.

Run from repository root: python -m question3.local_sim.solve_single --episodes 100
No official simulator access. No claim of continuous/global optimality.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from .simulator import Client, Simulator, Source


def prior(n, rng):
    # R hidden; target area-uniform in sector conditional on a direction (d>5).
    chunks = []
    while sum(len(x) for x in chunks) < n:
        R = rng.uniform(1000, 1500, n)
        d = R*np.sqrt(rng.random(n))
        a = np.deg2rad(rng.uniform(-1, 1, n))
        x = np.column_stack((d*np.cos(a), d*np.sin(a), R))
        chunks.append(x[d > 5])
    return np.concatenate(chunks)[:n]


def likelihood(particles, pos, response, bin_width=.01):
    dxy = particles[:, :2]-pos
    d = np.linalg.norm(dxy, axis=1)
    kind = response.get('measure_result', response.get('clear_result'))
    if kind == 'success': return (d <= 20).astype(float)
    if kind == 'no_target_in_range': return (d > 20).astype(float)
    if kind == 'near': return ((d <= 5) & (d <= particles[:,2])).astype(float)
    if kind == 'no_signal': return (d > particles[:,2]).astype(float)
    angle = np.rad2deg(np.arctan2(dxy[:,1], dxy[:,0])) % 360
    delta = (response['svd_deg']-angle+180) % 360-180
    # Integrate uniform error over returned-angle rounding/bin interval.
    overlap = np.maximum(0, np.minimum(1, delta+bin_width/2)-np.maximum(-1, delta-bin_width/2))/2
    return overlap*((d > 5) & (d <= particles[:,2]))


class Belief:
    def __init__(self, n=4096, seed=0):
        self.rng = np.random.default_rng(seed)
        self.p = prior(n, self.rng)
        self.w = np.full(n, 1/n)
        self.history = []
        self.recoveries = 0

    def update(self, pos, response):
        pos = np.asarray(pos)
        # Same-point errors are fixed: do not multiply a repeated observation likelihood.
        if any(np.array_equal(pos, q) and response.get('measure_result') is not None and old.get('measure_result') is not None for q,old in self.history):
            return
        self.history.append((pos.copy(), response.copy()))
        self.w *= likelihood(self.p, pos, response)
        if self.w.sum() == 0:
            self.recoveries += 1
            # Rejection refresh uses only public history, never simulator truth.
            for _ in range(30):
                points = prior(32768, self.rng)
                weights = np.ones(len(points))
                for q,r in self.history:
                    weights *= likelihood(points,q,r)
                    if weights.max() > 0: weights /= weights.max()
                keep = weights > 0
                if keep.any():
                    self.p, self.w = points[keep], weights[keep]
                    break
            else:
                raise RuntimeError('Particle depletion; invoke geometric coverage fallback')
        self.w /= self.w.sum()

    def planning(self, n):
        indices = np.searchsorted(np.cumsum(self.w), (np.arange(n)+.5)/n)
        indices = np.minimum(indices, len(self.p)-1)
        ids, counts = np.unique(indices, return_counts=True)
        return self.p[ids], counts/counts.sum()


def sweep_cost(p, w, pos):
    """Feasible empirical terminal policy: clear particles in either x-order.

    All support points are covered. This is NOT a guarantee for continuous belief.
    """
    if len(p) == 0 or w.sum() == 0: return 0.
    values = []
    for order in (np.argsort(p[:,0]), np.argsort(-p[:,0])):
        alive = w.copy()
        q = np.asarray(pos)
        cost = 0.
        for i in order:
            if alive[i] <= 0: continue
            target = p[i,:2]
            hit = np.linalg.norm(p[:,:2]-target, axis=1) <= 20
            mass, success = alive.sum(), alive[hit].sum()
            cost += mass*(np.linalg.norm(target-q)/5+3)+2*success
            alive[hit] = 0
            q = target
        values.append(cost)
    return float(min(values))


def candidate_actions(p, w, pos, visited=(), dense=False):
    center = w @ p[:,:2]
    v = center-pos
    length = np.linalg.norm(v)
    unit = v/max(length, 1e-12)
    normal = np.array([-unit[1], unit[0]])
    clears = [center, center-15*unit]
    for quantile in (.2,.5,.8):
        order = np.argsort(p[:,0])
        i = order[min(np.searchsorted(np.cumsum(w[order]),quantile),len(order)-1)]
        clears.append(p[i,:2])
    actions = [('clear', x) for x in clears]
    fractions = (.2,.35,.5,.65,.8,1.) if dense else (.4,.7,1.)
    offsets = (0., 15., 40., 80., 160.) if dense else (0.,20.,60.)
    for f in fractions:
        for lateral in offsets:
            for sign in ((1,-1) if lateral else (1,)):
                q = pos+f*v+sign*lateral*normal
                if np.linalg.norm(q-pos) > .01 and not any(np.linalg.norm(q-old) < 1e-8 for old in visited):
                    actions.append(('measure',q))
    return actions


def outcomes(p, q, width):
    yield 'no_signal', dict(measure_result='no_signal'), None
    yield 'near', dict(measure_result='near'), None
    delta = p[:,:2]-q
    angles = np.rad2deg(np.arctan2(delta[:,1],delta[:,0])) % 360
    # Only bins overlapping at least one particle's [-1,1] degree support.
    count = round(360/width)
    bins = set()
    for a in angles:
        for k in range(math.floor((a-1)/width), math.floor((a+1)/width)+1):
            bins.add(k % count)
    for k in sorted(bins):
        a = (k+.5)*width
        yield f'direction [{k*width:g},{(k+1)*width:g})', dict(measure_result='direction',svd_deg=a), width


def action_value(action, p, w, pos, width=4, depth=1, visited=()):
    kind, q = action
    travel = float(np.linalg.norm(q-pos)/5)
    branches = []
    if kind == 'clear':
        hit = np.linalg.norm(p[:,:2]-q, axis=1) <= 20
        probability = float(w[hit].sum())
        remaining = w*(~hit)
        continuation = 0.
        if remaining.sum() > 1e-12:
            continuation = leaf_value(p,remaining/remaining.sum(),q,width,depth-1,visited)
        cost = travel+3+2*probability+(1-probability)*continuation
        branches = [dict(observation='success',probability=probability, continuation_s=0.),
                    dict(observation='no_target_in_range',probability=1-probability,continuation_s=continuation)]
    else:
        cost = travel+5
        for label,response,bw in outcomes(p,q,width):
            weights = w*likelihood(p,q,response,bw or .01)
            probability = float(weights.sum())
            if probability <= 1e-12: continue
            keep = weights > 0
            continuation = 5. if label == 'near' else leaf_value(p[keep],weights[keep]/probability,q,width,depth-1,(*visited,q))
            cost += probability*continuation
            branches.append(dict(observation=label,probability=probability,continuation_s=continuation))
    return dict(kind=kind,position=q.tolist(),expected_s=float(cost),branches=branches)


def leaf_value(p,w,pos,width,depth,visited):
    if depth <= 0: return sweep_cost(p,w,pos)
    return min(action_value(a,p,w,pos,width,depth,visited)['expected_s']
               for a in candidate_actions(p,w,pos,visited))


def choose(belief, pos, planning_n=96, width=4, depth=1, dense=False):
    p,w = belief.planning(planning_n)
    visited = [q for q,r in belief.history if 'measure_result' in r]
    values = [action_value(a,p,w,np.asarray(pos),width,depth,visited)
              for a in candidate_actions(p,w,np.asarray(pos),visited,dense)]
    values.sort(key=lambda r:r['expected_s'])
    return values[0],values


def fallback_points():
    # Cover the entire original +/-1 deg, r<=1500 sector by a 25m square grid.
    # Rectangle [0,1500] x [-26.18,26.18]; furthest point to grid <=sqrt(2)*12.5<20.
    for i,x in enumerate(np.arange(0,1525,25)):
        ys = (-25.,0.,25.) if i%2==0 else (25.,0.,-25.)
        for y in ys: yield np.array([x,y])


def episode(seed, policy, args, fixed_first=None):
    truth = prior(1,np.random.default_rng(seed))[0]
    sim = Simulator(seed,sources=[Source(1,*truth)])
    client = Client(simulator=sim)
    client.command('/enter')
    # Time origin is AFTER a supplied initial bearing 0; do not make a second initial measurement.
    belief = Belief(args.particles, seed+100000)
    belief.history.append((np.zeros(2),dict(measure_result='direction',svd_deg=0.)))
    pos = np.zeros(2)
    fallback = False
    for step in range(20):
        try:
            if step == 0 and fixed_first is not None:
                action = fixed_first
            elif policy == 'tree':
                action,_ = choose(belief,pos,args.planning,args.bin_width,args.depth)
            else:
                p,w = belief.planning(args.planning)
                center = w @ p[:,:2]
                covered = w[np.linalg.norm(p[:,:2]-center,axis=1)<=20].sum()
                action = dict(kind='clear' if covered >= .999 else 'measure',
                              position=(center if covered >= .999 else pos+.7*(center-pos)).tolist())
            pos = np.asarray(action['position'])
            result = client.command('/'+action['kind'],pos,1)
            if result.get('clear_result') == 'success': break
            if result.get('measure_result') == 'near':
                result = client.command('/clear',pos,1)
                break
            belief.update(pos,result)
        except RuntimeError as e:
            if 'Particle depletion' not in str(e): raise
            fallback = True
            break
    else:
        fallback = True
    if not sim.sources[0].cleared:
        fallback = True
        for q in fallback_points():
            result = client.command('/clear',q,1)
            if result['clear_result'] == 'success': break
    client.command('/exit')
    return dict(seed=seed,time_s=sim.virtual_time,success=sim.sources[0].cleared,
                measures=sum(h['path']=='/measure' for h in sim.history),
                clears=sum(h['path']=='/clear' for h in sim.history),fallback=fallback,
                recoveries=belief.recoveries,truth=truth.tolist(),commands=sim.history)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=20260911)
    p.add_argument('--particles',type=int,default=4096)
    p.add_argument('--planning',type=int,default=96)
    p.add_argument('--bin-width',type=float,default=4.)
    p.add_argument('--depth',type=int,default=1)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--output',default='question3/local_sim/outputs/single')
    args=p.parse_args()
    if args.episodes < 1 or args.planning < 2 or args.particles < args.planning or args.depth < 1 or args.bin_width <= 0 or abs(360/args.bin_width-round(360/args.bin_width)) > 1e-9:
        p.error('Require episodes>=1, particles>=planning>=2, depth>=1, bin width dividing 360')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    belief=Belief(args.particles,args.seed)
    belief.history.append((np.zeros(2),dict(measure_result='direction',svd_deg=0.)))
    best,values=choose(belief,np.zeros(2),args.planning,args.bin_width,args.depth,True)
    (out/'tree.json').write_text(json.dumps(dict(configuration=vars(args),best=best,candidates=values),indent=2),encoding='utf8')
    print('Finite-tree optimized first action:',best['kind'],best['position'],best['expected_s'],flush=True)
    policies = {'optimized_first':best,'straight_first':dict(kind='measure',position=[best['position'][0],0.]),
                'conservative':None}
    samples={}; summary={}
    for name,first in policies.items():
        results=[]
        for i in range(args.episodes):
            results.append(episode(args.seed+1000+i,'conservative' if name=='conservative' else 'tree',args,first))
            if (i+1)%10==0: print(name,i+1,'/',args.episodes,flush=True)
        times=np.array([r['time_s'] for r in results])
        summary[name]=dict(n=len(results),mean_s=float(times.mean()),
            se_s=float(times.std(ddof=1)/np.sqrt(len(times))) if len(times)>1 else None,
            p90_s=float(np.quantile(times,.9)),successes=sum(r['success'] for r in results),
            fallbacks=sum(r['fallback'] for r in results),mean_measures=float(np.mean([r['measures'] for r in results])),
            mean_clears=float(np.mean([r['clears'] for r in results])))
        samples[name]=results
        (out/'episodes.json').write_text(json.dumps(samples,indent=2),encoding='utf8')
    delta=np.array([a['time_s']-b['time_s'] for a,b in zip(samples['optimized_first'],samples['straight_first'])])
    summary['paired_optimized_minus_straight']=dict(mean_s=float(delta.mean()),se_s=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None)
    summary['wall_time_s']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()

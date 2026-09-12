"""Continuous first-point optimization with a three-additional-probe certificate.

python -m question3.local_sim.three_probe.optimize --train 96 --test 1000
The first point is optimized; subsequent branching uses a certified base policy.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.optimize import differential_evolution, minimize
from question1.geometry import _clip, bearing_halfplanes
from question1.enclosing_circle import minimum_enclosing_circle
from ..solve_single import prior
from ..simulator import Simulator, Source, Client

EPS = math.radians(1.005)
Y = 1500*math.sin(EPS)
FACTOR = 2*math.cos(EPS)
THRESHOLD = 20*FACTOR**2


def certificate(q):
    a,h = map(float,q)
    h=abs(h)
    if h <= Y:
        return dict(receive_bound_m=math.inf,first_radius_bound_m=math.inf,final_radius_bound_m=math.inf,feasible=False)
    X=max(abs(a),abs(1500-a))
    M=X/(h-Y)
    width=2*Y*M+2*EPS*(h+Y)*(1+M*M)
    radius=math.hypot(width,2*Y)/2
    receive=math.hypot(X,h+Y)
    return dict(receive_bound_m=receive,first_radius_bound_m=radius,
                final_radius_bound_m=radius/FACTOR**2,
                feasible=receive<=1000 and radius<=THRESHOLD)


def initial_polygon():
    poly=np.array([[0.,-Y],[1500.,-Y],[1500.,Y],[0.,Y]])
    return condition_polygon(poly,np.zeros(2),0.)


def condition_polygon(poly,q,bearing):
    a,b=bearing_halfplanes([q],[bearing],error_deg=math.degrees(EPS))
    for normal,offset in zip(a,b): poly=_clip(poly,normal,offset,1e-9)
    if not len(poly): raise ArithmeticError('Empty conservative support')
    return poly


def safe_clear_point(circle,current):
    # Inner ball B(c,20-rho) is inside the robust clear set.
    delta=current-circle.center
    d=np.linalg.norm(delta)
    slack=max(0.,20-circle.radius-1e-7)
    return circle.center+delta*(min(1.,slack/d) if d else 0.)


def run_case(first,truth,errors=None,seed=None,trace=False):
    if not certificate(first)['feasible']: raise ValueError('Uncertified first point')
    poly=initial_polygon(); pos=np.zeros(2); q=np.asarray(first,dtype=float)
    elapsed=0.; probes=0; radii=[]; history=[]; final_radius=None
    client=None
    if seed is not None:
        sim=Simulator(seed,sources=[Source(1,*truth)])
        client=Client(simulator=sim);client.command('/enter')
    for k in range(3):
        probes+=1
        elapsed+=np.linalg.norm(q-pos)/5+5
        pos=q.copy()
        if client is not None:
            response=client.command('/measure',q,1)
        else:
            distance=np.linalg.norm(truth[:2]-q)
            if distance>truth[2]: response=dict(measure_result='no_signal')
            elif distance<=5: response=dict(measure_result='near')
            else:
                angle=math.degrees(math.atan2(truth[1]-q[1],truth[0]-q[0]))
                response=dict(measure_result='direction',svd_deg=round((angle+errors[k])%360,2)%360)
        if trace: history.append(dict(kind='measure',position=q.tolist(),response=response))
        if response['measure_result']=='no_signal': raise AssertionError('Reception certificate failed')
        if response['measure_result']=='near':
            final_radius=5.;clear=q.copy();break
        poly=condition_polygon(poly,q,response['svd_deg'])
        circle=minimum_enclosing_circle(poly)
        radii.append(circle.radius)
        final_radius=circle.radius
        if k==0 and circle.radius>certificate(first)['first_radius_bound_m']+1e-6:
            raise AssertionError('First radius bound failed')
        if k>0 and circle.radius>radii[-2]/FACTOR+1e-6:
            raise AssertionError('Contraction bound failed')
        if circle.radius<=20:
            clear=safe_clear_point(circle,q);break
        if k==2: raise AssertionError('Three-probe certificate failed')
        q=circle.center.copy()
    elapsed+=np.linalg.norm(clear-pos)/5+5
    success=np.linalg.norm(clear-truth[:2])<=20+1e-7
    if not success: raise AssertionError('Clear point does not cover truth')
    if client is not None:
        response=client.command('/clear',clear,1)
        if response['clear_result']!='success': raise AssertionError(response)
        client.command('/exit')
        elapsed=sim.virtual_time
    if trace: history.append(dict(kind='clear',position=clear.tolist(),success=bool(success)))
    return dict(time_s=float(elapsed),probes=probes,final_radius_m=final_radius,
                radii_m=radii,success=bool(success),history=history)


def statistics(cases):
    t=np.array([x['time_s'] for x in cases])
    return dict(n=len(cases),mean_s=float(t.mean()),se_s=float(t.std(ddof=1)/math.sqrt(len(t))) if len(t)>1 else None,
                max_s=float(t.max()),p90_s=float(np.quantile(t,.9)),
                probe_counts={str(k):sum(x['probes']==k for x in cases) for k in (1,2,3)},
                largest_terminal_radius_m=max(x['final_radius_m'] for x in cases),
                successes=sum(x['success'] for x in cases))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=int,default=96)
    parser.add_argument('--test',type=int,default=1000)
    parser.add_argument('--iterations',type=int,default=24)
    parser.add_argument('--seed',type=int,default=20260912)
    parser.add_argument('--output',default='question3/local_sim/three_probe/outputs')
    args=parser.parse_args()
    if min(args.train,args.test,args.iterations)<1: parser.error('Counts must be positive')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(args.seed)
    truth=prior(args.train,rng);errors=rng.uniform(-1,1,(args.train,3))
    calls=[];start=time.perf_counter()

    def objective(q):
        cert=certificate(q)
        violation=max(0,cert['receive_bound_m']-1000)+10*max(0,cert['first_radius_bound_m']-THRESHOLD)
        if violation>0: return 10000+violation
        cost=float(np.mean([run_case(q,t,e)['time_s'] for t,e in zip(truth,errors)]))
        calls.append(dict(first=np.asarray(q).tolist(),training_mean_s=cost))
        return cost

    print('Constructive certificate',certificate([750,600]),flush=True)
    # A separate smooth constrained problem minimizes only first-leg distance.
    # It supplies an initial feasible point; it is not the expected-time optimizer.
    sqp=minimize(lambda q:np.linalg.norm(q),[700,550],method='SLSQP',bounds=[(500,1000),(Y+1,750)],
                 constraints=[dict(type='ineq',fun=lambda q:999.999-certificate(q)['receive_bound_m']),
                              dict(type='ineq',fun=lambda q:THRESHOLD-1e-5-certificate(q)['first_radius_bound_m'])],
                 options=dict(maxiter=200,ftol=1e-9))
    initial=sqp.x if certificate(sqp.x)['feasible'] else np.array([750.,600.])
    print('Feasible shortest-first-leg point',initial.tolist(),flush=True)
    result=differential_evolution(objective,[(500,1000),(Y+1,750)],seed=args.seed,popsize=8,
                                  maxiter=args.iterations,polish=False,tol=.0002,atol=.01,x0=initial,
                                  callback=lambda x,convergence:print('DE best',np.round(x,3).tolist(),flush=True) or False)
    fixed=np.array([750.,600.])
    values=[(result.x,objective(result.x)),(fixed,objective(fixed)),(initial,objective(initial))]
    best,cost=min(values,key=lambda x:x[1])
    optimization=dict(configuration=vars(args),algorithm='sample average approximation + differential evolution; fixed certified continuation',
                      first=best.tolist(),training_mean_s=cost,certificate=certificate(best),
                      scipy_success=bool(result.success),scipy_message=str(result.message),nfev=int(result.nfev),
                      shortest_first_leg=initial.tolist(),shortest_first_leg_sqp_success=bool(sqp.success),
                      trace=calls)
    (out/'optimization.json').write_text(json.dumps(optimization,indent=2),encoding='utf8')
    test_truth=prior(args.test,np.random.default_rng(args.seed+500000))
    results={}
    for label,first in [('optimized',best),('constructive_750_600',fixed)]:
        cases=[run_case(first,t,seed=args.seed+600000+i,trace=i<3) for i,t in enumerate(test_truth)]
        results[label]=dict(first=first.tolist(),statistics=statistics(cases),cases=cases)
        print(label,results[label]['statistics'],flush=True)
    delta=np.array([a['time_s']-b['time_s'] for a,b in zip(results['optimized']['cases'],results['constructive_750_600']['cases'])])
    # Endpoint errors and targets test numerical implementation, not a substitute for proof.
    stress=[]
    for d in [5.0001,20.,300.,750.,1000.,1499.999]:
        for angle in [-1.,0.,1.]:
            g=np.array([d*math.cos(math.radians(angle)),d*math.sin(math.radians(angle)),1500.])
            for index in range(8):
                e=np.array([1 if (index>>k)&1 else -1 for k in range(3)])
                stress.append(run_case(best,g,e))
    summary=dict(optimized=results['optimized']['statistics'],constructive=results['constructive_750_600']['statistics'],
                 paired_difference_s=float(delta.mean()),paired_se_s=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None,
                 endpoint_stress=statistics(stress),wall_time_s=time.perf_counter()-start)
    (out/'evaluation.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()

"""Refine forecast coverage nodes without altering accepted observations."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from ..model import ARENA_R, RECEIVE_MIN
from ..adaptive_mpc.planning import open_route
from ..route_study.routes import held_karp


def route_length(start, route):
    return float(np.linalg.norm(np.diff(np.vstack([start, *route]),axis=0),axis=1).sum())


def reorder(start, route, exact=False):
    if not route:return []
    points=np.asarray(route)
    if exact and len(points)<=13:
        _,ids=held_karp(np.linalg.norm(points-start,axis=1),
                       np.linalg.norm(points[:,None]-points[None,:],axis=2),np.zeros(len(points)))
    else:_,ids=open_route(start,points)
    return [points[i].copy() for i in ids]


def mask(world, point, known):
    radius=RECEIVE_MIN-20 if any(np.linalg.norm(point-q)<1e-6 for q in known) else RECEIVE_MIN
    far=np.abs(world.coverage.centers-point)+world.coverage.half
    return np.sum(far*far,axis=1)<=radius**2-1e-5


def refine_route(world, original):
    """Coordinate descent in intersections of disks around necessary cell corners.

    Only dedicated coverage nodes move. Required corners are computed assuming
    all other forecast nodes remain. Retain an update only if the entire union
    still covers every previously predicted cell for all unfinished channels.
    This is a planning certificate, never an update to the real coverage map.
    """
    route=[np.asarray(q).copy() for q in original]
    p=world.params;start=world.position
    before=route_length(start,route)
    known=[t.center for t in world.active()]
    unknown=[t.channel-1 for t in world.unknown()]
    missing=(~world.coverage.covered[unknown]).any(axis=0) if unknown else np.zeros(len(world.coverage.centers),bool)
    accepted=removed=0
    if p.refine_coverage:
        for _ in range(p.coverage_passes):
            changed=False
            for i in range(len(route)-1,-1,-1):
                q=route[i]
                if any(np.linalg.norm(q-k)<1e-5 for k in known):continue
                others=np.zeros(len(missing),bool)
                for j,other in enumerate(route):
                    if j!=i:others |= mask(world,other,known)
                required=missing & ~others & mask(world,q,known)
                if not required.any():
                    route.pop(i);removed+=1;changed=True;continue
                centers=world.coverage.centers[required]
                h=world.coverage.half
                corners=np.vstack([centers+[a*h,b*h] for a,b in [(-1,-1),(-1,1),(1,-1),(1,1)]])
                corners=np.unique(corners,axis=0)
                try:corners=corners[ConvexHull(corners).vertices]
                except QhullError:pass
                left=start if i==0 else route[i-1]
                right=None if i+1==len(route) else route[i+1]
                def objective(y):
                    return np.linalg.norm(y-left)+(np.linalg.norm(y-right) if right is not None else 0.)
                def jac(y):
                    a=(y-left)/max(1e-9,np.linalg.norm(y-left))
                    return a+((y-right)/max(1e-9,np.linalg.norm(y-right)) if right is not None else 0.)
                result=minimize(objective,q,jac=jac,method='SLSQP',constraints=[
                    dict(type='ineq',fun=lambda y:(RECEIVE_MIN-1e-4)**2-np.sum((corners-y)**2,axis=1),
                         jac=lambda y:2*(corners-y)),
                    dict(type='ineq',fun=lambda y:(ARENA_R-1e-5)**2-np.sum(y*y),jac=lambda y:-2*y)],
                    options={'maxiter':45,'ftol':1e-5})
                y=result.x
                if (np.isfinite(y).all() and np.linalg.norm(y)<=ARENA_R and
                        objective(y)<objective(q)-1e-4 and np.all(mask(world,y,known)[required])):
                    route[i]=y;accepted+=1;changed=True
            route=reorder(start,route,p.exact_route)
            if not changed:break
    elif p.exact_route:
        route=reorder(start,route,True)
    return route,dict(before_m=before,after_m=route_length(start,route),moved=accepted,removed=removed)

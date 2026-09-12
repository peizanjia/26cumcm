"""Open path algorithms. DP optimizes the supplied surrogate, not unknown truth."""
import numpy as np
from numba import njit
from ..dynamic.frontier import short_service_route


@njit(cache=True)
def held_karp(start_cost,edges,end_cost):
    n=len(start_cost)
    if n==0:return 0.,np.empty(0,np.int64)
    size=1<<n
    dp=np.full((size,n),np.inf)
    parent=np.full((size,n),-1,np.int16)
    for j in range(n):dp[1<<j,j]=start_cost[j]
    for mask in range(1,size):
        for last in range(n):
            if not (mask&(1<<last)):continue
            prev=mask^(1<<last)
            if not prev:continue
            for before in range(n):
                if not(prev&(1<<before)):continue
                cost=dp[prev,before]+edges[before,last]
                if cost<dp[mask,last]:dp[mask,last]=cost;parent[mask,last]=before
    mask=size-1;last=np.argmin(dp[mask]+end_cost);value=dp[mask,last]+end_cost[last]
    order=np.empty(n,np.int64)
    for i in range(n-1,-1,-1):
        order[i]=last;before=parent[mask,last];mask^=1<<last;last=before
    return value,order


def plan_route(world,targets,destination,method):
    if method=='insertion_2opt':return short_service_route(world,targets,destination)
    if not targets:return []
    points=np.array([t.center for t in targets])
    if method=='nearest':
        current=world.position.copy();remaining=list(range(len(points)));order=[]
        while remaining:
            i=min(remaining,key=lambda k:np.linalg.norm(points[k]-current))
            order.append(i);remaining.remove(i);current=points[i]
    elif method in ('exact_dp','exact_open'):
        start=np.linalg.norm(points-world.position,axis=1)
        edges=np.linalg.norm(points[:,None,:]-points[None,:,:],axis=2)
        end=np.linalg.norm(points-destination,axis=1) if method=='exact_dp' else np.zeros(len(points))
        _,order=held_karp(start,edges,end)
    elif method in ('mst_2opt','christofides'):
        if len(points)==1:return [targets[0].channel]
        from .graph_algorithms import tree_or_christofides
        order=tree_or_christofides(points,world.position,destination,method)
    elif method=='annealing':
        from .graph_algorithms import anneal
        channels=[t.channel for t in targets]
        initial=[channels.index(c) for c in short_service_route(world,targets,destination)]
        order=anneal(points,world.position,destination,initial,world.params,channels)
    else:raise ValueError(f'Unknown route method: {method}')
    return [targets[i].channel for i in order]

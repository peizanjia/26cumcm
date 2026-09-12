"""Classical graph heuristics on a frozen public-position surrogate."""
import math
import numpy as np


def length(points,start,end,order):
    path=np.vstack((start,points[list(order)],end))
    return float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())


def two_opt(points,start,end,order):
    order=list(order);changed=True
    while changed:
        changed=False
        for i in range(len(order)):
            for j in range(i+1,len(order)):
                trial=order[:i]+order[i:j+1][::-1]+order[j+1:]
                if length(points,start,end,trial)<length(points,start,end,order)-1e-7:order=trial;changed=True
    return order


def tree_or_christofides(points,start,end,method):
    import networkx as nx
    all_points=np.vstack((start,points));graph=nx.Graph()
    for i in range(len(all_points)):
        for j in range(i):graph.add_edge(j,i,weight=float(np.linalg.norm(all_points[i]-all_points[j])))
    if method=='mst_2opt':
        tree=nx.minimum_spanning_tree(graph)
        sequence=list(nx.dfs_preorder_nodes(tree,source=0))
    elif method=='christofides':
        sequence=nx.approximation.christofides(graph,weight='weight')[:-1]
        pivot=sequence.index(0);sequence=sequence[pivot:]+sequence[:pivot]
    else:raise ValueError(method)
    order=[i-1 for i in sequence if i!=0]
    # Opening a Christofides cycle destroys its closed-tour 3/2 guarantee.
    # Both orientations are evaluated on the actual frozen open-path objective.
    if length(points,start,end,order[::-1])<length(points,start,end,order):order=order[::-1]
    return two_opt(points,start,end,order) if method=='mst_2opt' else order


def anneal(points,start,end,initial,params,channels):
    if len(initial)<3:return list(initial)
    seed=int(params.model_seed+sum((i+1)*c for i,c in enumerate(channels))+abs(start.sum())*10)%2147483647
    rng=np.random.default_rng(seed);current=list(initial);best=list(initial)
    current_cost=best_cost=length(points,start,end,current)
    for step in range(params.annealing_steps):
        i,j=sorted(rng.choice(len(current),2,replace=False))
        if rng.random()<.5:trial=current[:i]+current[i:j+1][::-1]+current[j+1:]
        else:
            trial=current.copy();node=trial.pop(i);trial.insert(j,node)
        value=length(points,start,end,trial)
        temperature=params.annealing_temperature*.02**(step/max(1,params.annealing_steps-1))
        if value<current_cost or rng.random()<math.exp(min(0.,(current_cost-value)/temperature)):
            current,current_cost=trial,value
        if value<best_cost:best,best_cost=trial,value
    return two_opt(points,start,end,best)

"""Continuous coverage by an adaptive partition and local receiver hulls.

For a whole convex source cell P, choose measured sites Q whose distance to
EVERY vertex of P is below 1000 m. If P is contained in conv(Q), every source
in P hears at least one of Q for every closed emitting halfplane. Cells form
a partition of a circumscribed polygon containing the true radius-1800 disk.
This does not require every edge of a global Delaunay mesh to be below 1000.
"""
from functools import lru_cache
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from question1.geometry import _clip
from question4.zigzag_study.study import certificate_details as triangle_certificate


def _split(poly):
    axis=int(np.argmax(np.ptp(poly,axis=0)))
    middle=float((poly[:,axis].min()+poly[:,axis].max())/2)
    normal=np.eye(2)[axis]
    return [_clip(poly,normal,middle,0.),_clip(poly,-normal,-middle,0.)]


@lru_cache(maxsize=1)
def initial_cells():
    angles=(np.arange(128)+.5)*2*np.pi/128
    poly=(1800.01/np.cos(np.pi/128))*np.column_stack((np.cos(angles),np.sin(angles)))
    todo=[poly];cells=[]
    while todo:
        p=todo.pop()
        if np.ptp(p,axis=0).max()<=455:cells.append(p)
        else:todo.extend(q for q in _split(p) if len(q)>=3)
    return tuple(cells)


def _inside_hull(points, vertices, margin=1e-7):
    if len(points)<3:return False
    try:hull=ConvexHull(points)
    except QhullError:return False
    return bool(np.max(vertices@hull.equations[:,:2].T+hull.equations[:,2])<=-margin)


def witness_normals(points, center):
    """Midpoint of the widest angular gap of sites within reception range."""
    delta=np.asarray(points).reshape(-1,2)-center
    delta=delta[(np.linalg.norm(delta,axis=1)<999.99)&(np.linalg.norm(delta,axis=1)>1e-7)]
    if not len(delta):return np.eye(2).tolist()+(-np.eye(2)).tolist()
    angles=np.sort(np.arctan2(delta[:,1],delta[:,0]))
    gaps=np.diff(np.r_[angles,angles[0]+2*np.pi]);ids=np.argsort(-gaps)
    return [np.array([np.cos(angles[i]+gaps[i]/2),np.sin(angles[i]+gaps[i]/2)]).tolist()
            for i in ids[:2]]


def partition_certificate(points, minimum_side=7., details=False, stop_on_gap=False):
    sites=np.unique(np.asarray(points,dtype=float).reshape(-1,2),axis=0)
    todo=list(initial_cells()); unresolved=[]; accepted=[];checks=0
    while todo:
        poly=todo.pop();checks+=1
        distances=np.linalg.norm(sites[:,None,:]-poly[None,:,:],axis=2)
        near=sites[np.max(distances,axis=1)<1000.-1e-6] if len(sites) else sites
        if _inside_hull(near,poly):
            if details:accepted.append(poly.tolist())
            continue
        center=poly.mean(axis=0)
        local=sites[np.linalg.norm(sites-center,axis=1)<1000.-1e-6]
        center_covered=_inside_hull(local,center[None,:])
        if center_covered and np.ptp(poly,axis=0).max()>minimum_side:
            todo.extend(q for q in _split(poly) if len(q)>=3)
            continue
        unresolved.append(dict(center=center.tolist(),polygon=poly.tolist(),
            side_m=float(np.ptp(poly,axis=0).max()),normals=witness_normals(sites,center)))
        if stop_on_gap:break
    return dict(certified=not unresolved,proof_kind='adaptive_local_hull_partition',
                cells=accepted if details else [],unresolved=unresolved,cell_checks=checks,
                domain_outer_radius_m=1800.01/np.cos(np.pi/128),minimum_cell_side_m=minimum_side)


def certificate_details(points):
    quick=triangle_certificate(points)
    if quick['certified']:return quick
    # A convex hull enclosing the source disk is a necessary condition.
    if quick['hull_inradius_m']<1800.:return quick
    result=partition_certificate(points,stop_on_gap=True)
    return dict(quick,**{k:v for k,v in result.items() if k not in ('triangles',)})

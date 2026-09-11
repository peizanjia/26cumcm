"""Numba MEC radius and physical outcome scoring, coordinates in metres."""
import numpy as np
from numba import njit, prange
from question2.黑箱思路.geometry import second_cpu


@njit(cache=True)
def mec(p):
    n=len(p)
    if n == 0:
        return 0.
    best=1e300
    # Candidate diameters; an enclosing diameter is already optimal if it
    # attains the farthest pair, but enumeration keeps the implementation clear.
    for i in range(n):
        for j in range(i,n):
            c=(p[i]+p[j])*.5
            r2=np.sum((p[i]-c)**2)
            if r2 >= best: continue
            m=0.
            for k in range(n): m=max(m,np.sum((p[k]-c)**2))
            if m <= r2+1e-8: best=max(r2,m)
    for i in range(n):
        for j in range(i+1,n):
            for k in range(j+1,n):
                u=p[j]-p[i];v=p[k]-p[i]
                cross=u[0]*v[1]-u[1]*v[0]
                if abs(cross)<1e-12: continue
                u2=np.sum(u*u);v2=np.sum(v*v)
                c=p[i]+np.array([u2*v[1]-v2*u[1],u[0]*v2-v[0]*u2])/(2*cross)
                r2=np.sum((p[i]-c)**2)
                if r2>=best: continue
                m=0.
                for z in range(n): m=max(m,np.sum((p[z]-c)**2))
                if m <= r2+1e-7: best=max(r2,m)
    return np.sqrt(best)


@njit(cache=True,parallel=True)
def score(polys,counts,targets,errors,radii,q):
    """Each row has an action and posterior events; R shared with first read.

    Returns realized radius (not conditional mean radius per event). No signal
    retains first MEC, near is zero AFTER successful optical localization.
    """
    out=np.empty(errors.shape)
    for i in prange(len(q)):
        poly=polys[i,:counts[i]]
        r0=mec(poly)
        for j in range(errors.shape[1]):
            d=np.linalg.norm(targets[i,j]-q[i])
            if np.linalg.norm(q[i])<1e-9: out[i,j]=r0
            elif d<=5: out[i,j]=0.
            elif d>radii[i,j]: out[i,j]=r0
            else:
                a=np.arctan2(targets[i,j,1]-q[i,1],targets[i,j,0]-q[i,0])+errors[i,j]
                out[i,j]=mec(second_cpu(poly,q[i],a))
    return out


def statistics(x,threshold=True):
    x=np.asarray(x,dtype=float).ravel()
    result=dict(n=len(x),mean=float(x.mean()),min=float(x.min()),max=float(x.max()),
        variance=float(x.var()),std=float(x.std()),median=float(np.median(x)),
        p05=float(np.quantile(x,.05)),p25=float(np.quantile(x,.25)),p75=float(np.quantile(x,.75)),
        p90=float(np.quantile(x,.90)),p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),
        rms=float(np.sqrt(np.mean(x*x))),success_le20=float(np.mean(x<=20)),
        mean_excess20=float(np.maximum(x-20,0).mean()))
    if not threshold:
        result.pop('success_le20');result.pop('mean_excess20')
    return result


@njit(cache=True,parallel=True)
def marginalized_cost(polys,counts,targets,errors,q,guided):
    """Integrate shared unknown R analytically BEFORE differentiating penalty.

    E[f(radius)] is used, never f(E[radius]); avoids rare sampled detection
    jumps overwhelming finite-difference policy updates.
    """
    out=np.zeros(len(q))
    for i in prange(len(q)):
        poly=polys[i,:counts[i]];r0=mec(poly)
        v0=r0/20
        if guided:v0+=4*max(r0/20-1,0)**2
        for j in range(errors.shape[1]):
            g=targets[i,j];d=np.linalg.norm(g-q[i])
            if np.linalg.norm(q[i])<1e-9:out[i]+=v0
            elif d<=5:continue
            else:
                low=max(1000.,np.linalg.norm(g))
                prob=max(0.,min(1.,(1500-max(low,d))/max(1e-9,1500-low)))
                v=0.
                if prob>0:
                    a=np.arctan2(g[1]-q[i,1],g[0]-q[i,0])+errors[i,j]
                    rad=mec(second_cpu(poly,q[i],a));v=rad/20
                    if guided:v+=4*max(rad/20-1,0)**2
                out[i]+=prob*v+(1-prob)*v0
        out[i]/=errors.shape[1]
        if guided:out[i]+=.15*np.linalg.norm(q[i])/1500
    return out


def expected_cost(d,q,guided):
    return marginalized_cost(np.ascontiguousarray(d['polygons']),d['counts'],
        np.ascontiguousarray(d['targets']),np.ascontiguousarray(d['errors']),
        np.ascontiguousarray(q,dtype=float),guided)


def dataset(n,m,seed,sides=512):
    from question2.黑箱思路.environment import make_dataset
    d=make_dataset(n,m,seed,sides=sides)
    d['targets']*=1500; d['polygons']*=1500
    low=np.maximum(1000,np.linalg.norm(d['targets'],axis=-1))
    d['radii']=low+(1500-low)*np.random.default_rng(seed+3).random(low.shape)
    return d


def evaluate(d,q):
    return score(np.ascontiguousarray(d['polygons']),d['counts'],np.ascontiguousarray(d['targets']),
                 np.ascontiguousarray(d['errors']),np.ascontiguousarray(d['radii']),np.ascontiguousarray(q,dtype=float))

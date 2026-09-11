"""Exact polygon MEC, shared-radius marginalization and posterior angle guidance.

All geometry inputs are metres. Network never receives any target coordinates.
Finite-difference action derivatives are used by train.py, not autograd of Numba.
"""
import numpy as np
from numba import njit, prange
from question2.最小包围圆思路.core import mec
from ..geometry import second_cpu, DELTA


@njit(cache=True, parallel=True)
def components(polys, counts, targets, errors, q):
    # Columns: E|rho-20|/20, E[available-direction * cos²], E[rho], P(no signal).
    out=np.zeros((len(q),4))
    sinc=np.sin(2*DELTA)/(2*DELTA)
    for i in prange(len(q)):
        poly=polys[i,:counts[i]]
        r0=mec(poly)
        for j in range(errors.shape[1]):
            g=targets[i,j];dx=g[0]-q[i,0];dy=g[1]-q[i,1]
            d2=dx*dx+dy*dy;d=np.sqrt(d2)
            if np.linalg.norm(q[i])<1e-9:
                out[i,0]+=abs(r0-20)/20;out[i,1]+=1;out[i,2]+=r0
            elif d<=5:
                out[i,0]+=1 # optical terminal radius zero; literal target-20 deviation
            else:
                low=max(1000.,np.linalg.norm(g))
                prob=max(0.,min(1.,(1500-max(low,d))/max(1e-9,1500-low)))
                rad=0.
                if prob>0:
                    angle=np.arctan2(dy,dx)+errors[i,j]
                    rad=mec(second_cpu(poly,q[i],angle))
                out[i,0]+=prob*abs(rad-20)/20+(1-prob)*abs(r0-20)/20
                out[i,1]+=prob*(.5+.5*sinc*(dx*dx-dy*dy)/d2)
                out[i,2]+=prob*rad+(1-prob)*r0
                out[i,3]+=1-prob
        out[i]/=errors.shape[1]
    return out


def expected_components(d,q):
    return components(np.ascontiguousarray(d['polygons']),d['counts'],
        np.ascontiguousarray(d['targets']),np.ascontiguousarray(d['errors']),
        np.ascontiguousarray(q,dtype=np.float64))


def cost(d,q,weight=0.):
    c=expected_components(d,q)
    return c[:,0]+weight*c[:,1]

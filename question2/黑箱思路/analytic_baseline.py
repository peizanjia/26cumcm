"""Small-angle analytical baseline; uses posterior moments, no hidden truth."""
import numpy as np
from .geometry import ARENA, DELTA, NEAR


def moment_action(local_p):
    """a=E[r²]/E[r], |b|=sqrt(E[r³]/E[r]-a²).

    Posterior moments include the exact arena ray boundary, radius survival
    weighting, and 64-point Gauss-Legendre angular quadrature. The objective
    approximation itself ignores wedge curvature and first-circle clipping.
    """
    p = np.atleast_2d(local_p).astype(float)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    angle = nodes*DELTA
    dot = p[:, :1]*np.cos(angle)[None]+p[:, 1:]*np.sin(angle)[None]
    upper = np.minimum(1., -dot+np.sqrt(np.maximum(0, dot**2+ARENA**2-(p*p).sum(1)[:, None])))
    upper = np.maximum(NEAR, upper)
    cutoff = 2/3
    integrals = []
    for k in (1, 2, 3):
        # Integrate r^(k+1) * w(r) dr, w=1 below 2/3, 3(1-r) above.
        m = k+2
        low = (np.minimum(upper, cutoff)**m-NEAR**m)/m
        high = 3*((upper**m-cutoff**m)/m-(upper**(m+1)-cutoff**(m+1))/(m+1))
        radial = low+np.where(upper>cutoff, high, 0)
        integrals.append((radial*weights[None]).sum(1))
    z1, z2, z3 = integrals
    a = z2/np.maximum(z1, 1e-15)
    b = np.sqrt(np.maximum(0, z3/np.maximum(z1, 1e-15)-a*a))
    return np.column_stack([a, -b])

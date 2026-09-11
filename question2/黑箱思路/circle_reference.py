"""Independent continuous-circle area integral for numerical verification.

No circle polygonization: integrate feasible radial intervals along the first
bearing wedge. Coordinates use the same 1500 m scaling as the main model.
"""
import numpy as np
from scipy.integrate import quad
from .geometry import ARENA, DELTA


def exact_circle_area(local_p, q=None, theta2=None):
    local_p = np.asarray(local_p, float)
    normals = []
    if q is not None:
        q = np.asarray(q, float)
        for sign in [-1, 1]:
            a = theta2 + sign*DELTA
            n = np.array([-sign*np.sin(a), sign*np.cos(a)])
            normals.append((n, float(np.dot(n, q))))

    def integrand(alpha):
        u = np.array([np.cos(alpha), np.sin(alpha)])
        dot = float(np.dot(local_p, u))
        discriminant = dot*dot+ARENA*ARENA-float(np.dot(local_p, local_p))
        if discriminant < 0:
            return 0.
        lo = max(0., -dot-np.sqrt(discriminant))
        hi = min(1., -dot+np.sqrt(discriminant))
        for n, b in normals:
            coefficient = float(np.dot(n, u))
            if abs(coefficient) < 1e-14:
                if b < 0:
                    return 0.
            elif coefficient > 0:
                hi = min(hi, b/coefficient)
            else:
                lo = max(lo, b/coefficient)
        if hi <= lo:
            return 0.
        return .5*(hi*hi-lo*lo)
    return quad(integrand, -DELTA, DELTA, epsabs=1e-10, epsrel=1e-7,
                limit=250, points=np.linspace(-DELTA, DELTA, 17)[1:-1])[0]

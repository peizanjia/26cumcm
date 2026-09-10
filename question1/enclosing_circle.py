"""Smallest enclosing circle and diameter-circle diagnostics for small polygons.

The minimum circle is supported by at most three vertices. Exhaustive triples
are deliberate here: localization polygons are small and auditability matters.
Coordinates are normalized before circumcenter calculation.
"""

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np

from .geometry import polygon_area, polygon_diameter


@dataclass(frozen=True)
class EnclosingCircle:
    center: np.ndarray
    radius: float
    support_indices: tuple[int, ...]


@lru_cache(maxsize=128)
def _triples(count):
    return np.array(list(combinations(range(count), 3)), dtype=int).reshape(-1, 3)


def minimum_enclosing_circle(vertices) -> EnclosingCircle:
    """Exact candidate enumeration in floating point; O(m^4) time, O(m^3) memory.

    For a two-point supported minimum circle, those points must be a diameter
    pair of the entire point set. Otherwise enumerate all noncollinear triples.
    Intended for small sets (normally 3-12 localization vertices), not clouds.
    """
    p = np.asarray(vertices, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0 or not np.isfinite(p).all():
        raise ValueError("vertices must be a nonempty finite (m,2) array")
    origin = p[0].copy()
    scale = float(np.max(np.ptp(p, axis=0)))
    if scale == 0:
        return EnclosingCircle(origin, 0.0, (0,))
    q = (p - origin) / scale
    distances2 = np.sum((q[:, None] - q[None, :])**2, axis=2)
    i, j = np.unravel_index(np.argmax(distances2), distances2.shape)
    center = (q[i] + q[j]) / 2
    radius2 = distances2[i, j] / 4
    if np.max(np.sum((q - center)**2, axis=1)) <= radius2 + 1e-12:
        # Inflate only to remove floating-point undercoverage.
        radius = float(np.max(np.linalg.norm(q-center, axis=1))) * scale
        return EnclosingCircle(origin + scale*center, radius, (int(i), int(j)))

    indices = _triples(len(p))
    base = q[indices[:, 0]]
    u = q[indices[:, 1]] - base
    v = q[indices[:, 2]] - base
    cross = u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]
    valid = np.abs(cross) > 1e-14
    indices, base, u, v, cross = (x[valid] for x in (indices, base, u, v, cross))
    u2, v2 = np.sum(u*u, axis=1), np.sum(v*v, axis=1)
    centers = base + np.column_stack((u2*v[:, 1] - v2*u[:, 1],
                                      u[:, 0]*v2 - v[:, 0]*u2)) / (2*cross[:, None])
    radii2 = np.sum((centers-base)**2, axis=1)
    # Stream over vertices, avoiding a triples-by-vertices temporary array.
    maxd2 = np.zeros(len(centers))
    for point in q:
        maxd2 = np.maximum(maxd2, np.sum((centers-point)**2, axis=1))
    covers = maxd2 <= radii2 + 1e-11
    if not covers.any():
        raise ArithmeticError("No valid enclosing circle found; inspect numerical conditioning")
    score = np.where(covers, np.maximum(radii2, maxd2), np.inf)
    k = int(np.argmin(score))
    return EnclosingCircle(origin + scale*centers[k],
                           float(np.sqrt(score[k]))*scale,
                           tuple(int(x) for x in indices[k]))


def diameter_circle_metrics(vertices, source=None, *, rtol=1e-7, atol=1e-8):
    """Certify coverage and quantify the minimum necessary radius increase.

    For ANY diameter pair A,B, a circle of radius D/2 containing both must have
    center (A+B)/2. Thus failure of this one circle disproves all radius-D/2
    circles, not just a particular choice of center. Every polygon vertex is
    tested; convexity of a disk then guarantees coverage of the entire polygon.
    """
    p = np.asarray(vertices, dtype=float)
    if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol <= 0:
        raise ValueError("invalid tolerances")
    diameter, endpoints = polygon_diameter(p)
    if len(endpoints) == 0 or diameter == 0:
        raise ValueError("metrics require a nonzero-diameter polygon")
    center = endpoints.mean(axis=0)
    radial = np.linalg.norm(p-center, axis=1)
    outside = radial - diameter/2
    tol = max(atol, rtol*diameter)
    circle = minimum_enclosing_circle(p)
    perimeter = float(np.linalg.norm(np.roll(p, -1, axis=0)-p, axis=1).sum())
    area = polygon_area(p)
    failure = bool(np.max(outside) > tol)
    # The MEC and midpoint tests are equivalent analytically. Points extremely
    # near the threshold can differ numerically, so record both margins.
    result = {"diameter_m": diameter, "diameter_endpoints": endpoints.tolist(),
              "diameter_center": center.tolist(), "diameter_radius_m": diameter/2,
              "mec_center": circle.center.tolist(), "mec_radius_m": circle.radius,
              "mec_support_indices": list(circle.support_indices),
              "radius_ratio": 2*circle.radius/diameter,
              "mec_radius_excess_m": circle.radius-diameter/2,
              "max_outside_m": float(np.max(outside)),
              "outside_vertex_indices": np.flatnonzero(outside > tol).tolist(),
              "failure": failure, "tolerance_m": tol,
              "area_m2": area, "perimeter_m": perimeter,
              "circularity": float(4*np.pi*area/perimeter**2),
              "vertices_count": len(p)}
    if source is not None:
        s = np.asarray(source, dtype=float)
        if s.shape != (2,) or not np.isfinite(s).all():
            raise ValueError("source must be a finite coordinate pair")
        result["source_outside_m"] = float(np.linalg.norm(s-center)-diameter/2)
        result["source_outside"] = bool(result["source_outside_m"] > tol)
    return result

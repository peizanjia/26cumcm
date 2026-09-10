"""Intersect closed bearing wedges without gridding or slope singularities.

Angles are degrees counterclockwise from east. A wedge is two half-planes.
The usual bounded case uses convex polygon clipping; LP is a fallback only
when a temporary computational box cannot certify the full intersection.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class IntersectionResult:
    status: Literal["bounded", "empty", "unbounded"]
    vertices: FloatArray
    # True only if the caller explicitly supplied a finite search box.
    search_box_applied: bool = False


def bearing_halfplanes(points: ArrayLike, bearings_deg: ArrayLike,
                       error_deg: ArrayLike = 1.0) -> tuple[FloatArray, FloatArray]:
    """Return unit-normal inequalities A @ x <= b for forward bearing wedges."""
    p = np.asarray(points, dtype=float)
    theta = np.asarray(bearings_deg, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) == 0:
        raise ValueError("points must have shape (n, 2), n >= 1")
    if theta.shape != (len(p),):
        raise ValueError("bearings_deg must have shape (n,)")
    try:
        error = np.broadcast_to(np.asarray(error_deg, dtype=float), theta.shape)
    except ValueError as exc:
        raise ValueError("error_deg must be scalar or shape (n,)") from exc
    if not all(np.isfinite(v).all() for v in (p, theta, error)):
        raise ValueError("all inputs must be finite")
    if np.any((error <= 0) | (error >= 90)):
        raise ValueError("each error half-width must be strictly between 0 and 90 degrees")
    lower, upper = np.deg2rad((theta % 360) - error), np.deg2rad((theta % 360) + error)
    a = np.empty((2 * len(p), 2))
    a[0::2] = np.column_stack((np.sin(lower), -np.cos(lower)))
    a[1::2] = np.column_stack((-np.sin(upper), np.cos(upper)))
    b = np.einsum("ij,ij->i", a, np.repeat(p, 2, axis=0))
    return a, b


def _box_vertices(bounds: ArrayLike) -> FloatArray:
    box = np.asarray(bounds, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("bounds must be finite (xmin, ymin, xmax, ymax)")
    x0, y0, x1, y1 = box
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bounds must have positive width and height")
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


def _deduplicate(vertices: list | FloatArray, tol: float) -> FloatArray:
    result = []
    for v in vertices:
        if not result or np.linalg.norm(v - result[-1]) > tol:
            result.append(v)
    if len(result) > 1 and np.linalg.norm(result[0] - result[-1]) <= tol:
        result.pop()
    return np.asarray(result, dtype=float).reshape(-1, 2)


def _clip(poly: FloatArray, normal: FloatArray, offset: float, tol: float) -> FloatArray:
    if not len(poly):
        return poly
    distance = poly @ normal - offset
    # Snap only rounding-level signed distances to zero. Intersections otherwise
    # lie on the actual boundary, not on an expanded tolerance strip.
    distance[np.abs(distance) <= tol] = 0.0
    inside = distance <= 0
    if inside.all():
        return poly
    if not inside.any():
        return np.empty((0, 2))
    out = []
    for i in range(len(poly)):
        j = (i - 1) % len(poly)
        if inside[i] != inside[j]:
            t = distance[j] / (distance[j] - distance[i])
            out.append(poly[j] + t * (poly[i] - poly[j]))
        if inside[i]:
            out.append(poly[i])
    return _deduplicate(out, tol)


def _clip_all(bounds: ArrayLike, a: FloatArray, b: FloatArray, tol: float) -> FloatArray:
    poly = _box_vertices(bounds)
    for normal, offset in zip(a, b):
        poly = _clip(poly, normal, offset, tol)
        if not len(poly):
            break
    return poly


def intersect_bearings(points: ArrayLike, bearings_deg: ArrayLike,
                       error_deg: ArrayLike = 1.0, *,
                       bounds: ArrayLike | None = None,
                       atol: float = 1e-8) -> IntersectionResult:
    """Compute the complete intersection of bearing wedges.

    Parameters
    ----------
    points : (n, 2) monitoring coordinates in metres.
    bearings_deg : (n,) measured directions, east=0, north=90.
    error_deg : scalar or (n,) positive error half-widths, default 1 degree.
    bounds : optional (xmin, ymin, xmax, ymax) physical search rectangle.
        None means the complete wedge intersection, with no arena truncation.
    atol : positive absolute geometric tolerance in metres.

    Returns
    -------
    IntersectionResult : bounded vertices in counterclockwise order; empty and
        unbounded results have shape-(0,2) vertices. A point or segment is a
        bounded degenerate result. Vertices represent the closed feasible set;
        bearing itself is undefined exactly at a monitoring point.

    Complexity
    ----------
    Clipping costs O(n*v), worst-case O(n^2), where v is
    the largest intermediate vertex count. This is intended for small/moderate
    monitoring sets, not claimed to be asymptotically optimal half-plane
    intersection. Including stored constraints, memory is O(n+v).
    Rare ambiguous cases use up to five 2-variable linear programs.
    """
    if not np.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be positive and finite")
    a, b = bearing_halfplanes(points, bearings_deg, error_deg)
    p = np.asarray(points, dtype=float)
    if bounds is not None:
        poly = _clip_all(bounds, a, b, atol)
        return IntersectionResult("bounded" if len(poly) else "empty", poly, True)

    # Translate to reduce cancellation for coordinates far from the origin.
    origin = p.mean(axis=0)
    local_b = np.einsum("ij,ij->i", a, np.repeat(p - origin, 2, axis=0))
    span = max(3600.0, float(np.max(np.abs(p - origin))) * 4)
    poly = _clip_all((-span, -span, span, span), a, local_b, atol)
    if len(poly) and np.max(np.abs(poly)) < span - 10 * atol:
        return IntersectionResult("bounded", poly + origin)

    # A finite artificial box is NOT evidence of boundedness or emptiness.
    # Resolve cases touching/outside it with LP on the actual half-planes.
    from scipy.optimize import linprog

    lp_options = {"primal_feasibility_tolerance": 1e-9,
                  "dual_feasibility_tolerance": 1e-9}

    def solve(c):
        result = linprog(c, A_ub=a, b_ub=local_b,
                         bounds=[(None, None), (None, None)],
                         method="highs", options=lp_options)
        if result.status not in (0, 2, 3):
            raise ArithmeticError(f"Half-plane LP failed: {result.message}")
        return result

    feasible = solve([0, 0])
    if feasible.status == 2:
        return IntersectionResult("empty", np.empty((0, 2)))
    if feasible.status != 0:
        raise ArithmeticError("Could not establish feasibility")
    limits = []
    for objective in ([1, 0], [0, 1], [-1, 0], [0, -1]):
        result = solve(objective)
        if result.status == 3:
            return IntersectionResult("unbounded", np.empty((0, 2)))
        if result.status != 0:
            raise ArithmeticError("Inconsistent LP feasibility results")
        limits.append(float(result.fun))
    low = np.array(limits[:2])
    high = -np.array(limits[2:])
    pad = max(1.0, float(np.max(high - low)) * 0.01)
    poly = _clip_all((*(low - pad), *(high + pad)), a, local_b, atol)
    if not len(poly):
        raise ArithmeticError("Clipping disagrees with LP; rescale coordinates or tolerance")
    return IntersectionResult("bounded", poly + origin)


def polygon_area(vertices: ArrayLike) -> float:
    p = np.asarray(vertices, dtype=float).reshape(-1, 2)
    if len(p) < 3:
        return 0.0
    p = p - p[0]
    return float(abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                            - p[:, 1] * np.roll(p[:, 0], -1))) / 2)


def polygon_diameter(vertices: ArrayLike) -> tuple[float, FloatArray]:
    """O(v) rotating calipers for an ordered convex polygon (CW or CCW).

    Return diameter and its two endpoints. Empty input returns (0, empty).
    """
    p = np.asarray(vertices, dtype=float).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("vertices must be finite")
    n = len(p)
    if n == 0:
        return 0.0, np.empty((0, 2))
    if n == 1:
        return 0.0, np.repeat(p, 2, axis=0)
    if n == 2:
        return float(np.linalg.norm(p[1] - p[0])), p.copy()
    q = p - p[0]
    signed_area = np.sum(q[:, 0] * np.roll(q[:, 1], -1)
                         - q[:, 1] * np.roll(q[:, 0], -1))
    if signed_area < 0:
        p = p[::-1]
    if signed_area == 0:
        axis = int(np.argmax(np.ptp(p, axis=0)))
        pair = p[[np.argmin(p[:, axis]), np.argmax(p[:, axis])]]
        return float(np.linalg.norm(pair[1] - pair[0])), pair
    best2, pair = 0.0, p[[0, 0]].copy()
    j = 1
    for i in range(n):
        nxt = (i + 1) % n
        edge = p[nxt] - p[i]

        def height(k):
            v = p[k % n] - p[i]
            return edge[0] * v[1] - edge[1] * v[0]

        for _ in range(n):
            if height(j + 1) <= height(j):
                break
            j = (j + 1) % n
        for u, v in ((i, j), (nxt, j), (i, (j + 1) % n), (nxt, (j + 1) % n)):
            d2 = float(np.dot(p[u] - p[v], p[u] - p[v]))
            if d2 > best2:
                best2, pair = d2, p[[u, v]].copy()
    return float(np.sqrt(best2)), pair

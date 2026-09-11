"""CPU reference geometry and differentiable, batched convex clipping.

Internal coordinates are measured in 1500 m units, with the first observed
bearing along +x and the first monitor at (0, 0). Circles are approximated
OUTWARDS by tangent polygons; source samples always use the exact circles.
"""
import numpy as np
import torch
from numba import njit, prange

SCALE = 1500.0
ARENA = 1800.0 / SCALE
DELTA = np.pi / 180
NEAR = 5.0 / SCALE


@njit(cache=True)
def clip_cpu(poly, normal, offset):
    out = np.empty((len(poly) + 2, 2), np.float64)
    k = 0
    for i in range(len(poly)):
        j = (i - 1) % len(poly)
        da = np.dot(poly[j], normal) - offset
        db = np.dot(poly[i], normal) - offset
        if (da <= 0) != (db <= 0):
            out[k] = poly[j] + da / (da - db) * (poly[i] - poly[j])
            k += 1
        if db <= 0:
            out[k] = poly[i]
            k += 1
    return out[:k].copy()


@njit(cache=True)
def area_cpu(poly):
    if len(poly) < 3:
        return 0.0
    total = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        total += ((poly[i, 0] - poly[0, 0]) * (poly[j, 1] - poly[0, 1])
                  - (poly[i, 1] - poly[0, 1]) * (poly[j, 0] - poly[0, 0]))
    return abs(total) / 2


@njit(cache=True)
def first_polygon(local_p, sides=256):
    # x<=1 is tangent to the unit receiving disc. Extra tangents at the
    # wedge edges reduce the local receiving-circle approximation error.
    poly = np.array([[0., 0.], [1., -np.tan(DELTA)], [1., np.tan(DELTA)]])
    for angle in (-DELTA, DELTA):
        poly = clip_cpu(poly, np.array([np.cos(angle), np.sin(angle)]), 1.)
    for i in range(sides):
        angle = 2 * np.pi * i / sides
        normal = np.array([np.cos(angle), np.sin(angle)])
        poly = clip_cpu(poly, normal, ARENA - np.dot(normal, local_p))
    return poly


@njit(cache=True, parallel=True)
def first_polygons(local_p, sides=256, capacity=64):
    vertices = np.zeros((len(local_p), capacity, 2))
    counts = np.zeros(len(local_p), np.int64)
    areas = np.zeros(len(local_p))
    for i in prange(len(local_p)):
        p = first_polygon(local_p[i], sides)
        counts[i] = len(p)
        if len(p) <= capacity:
            vertices[i, :len(p)] = p
        areas[i] = area_cpu(p)
    return vertices, counts, areas


def clip_torch(poly, counts, normal, offset):
    """Differentiable Sutherland-Hodgman clipping, one plane per batch row.

    Topology is discrete, but active intersections and shoelace areas retain
    their exact derivatives away from topology changes. No soft raster mask.
    """
    batch, cap, _ = poly.shape
    index = torch.arange(cap, device=poly.device)[None].expand(batch, -1)
    valid = index < counts[:, None]
    previous = torch.where(index == 0, (counts - 1).clamp_min(0)[:, None], index - 1)
    a = poly.gather(1, previous[..., None].expand(-1, -1, 2))
    da = (a * normal[:, None]).sum(-1) - offset[:, None]
    db = (poly * normal[:, None]).sum(-1) - offset[:, None]
    crossing = ((da <= 0) != (db <= 0)) & valid
    keep = (db <= 0) & valid
    denominator = torch.where(crossing, da - db, torch.ones_like(da))
    t = torch.where(crossing, da / denominator, torch.zeros_like(da))
    intersection = a + t[..., None] * (poly - a)
    emit = torch.stack((crossing, keep), -1).reshape(batch, 2 * cap)
    values = torch.stack((intersection, poly), 2).reshape(batch, 2 * cap, 2)
    positions = (emit.long().cumsum(1) - 1).clamp_min(0)
    new_count = emit.sum(1)
    # A convex polygon gains at most one vertex from one half-plane.
    out = torch.zeros((batch, cap + 1, 2), device=poly.device, dtype=poly.dtype)
    out.scatter_add_(1, positions[..., None].expand(-1, -1, 2),
                     torch.where(emit[..., None], values, torch.zeros_like(values)))
    return out, new_count


def area_torch(poly, counts):
    batch, cap, _ = poly.shape
    index = torch.arange(cap, device=poly.device)[None].expand(batch, -1)
    nxt = torch.where(index + 1 < counts[:, None], index + 1, 0)
    # Translate to improve float32 accuracy for long, narrow polygons.
    p = poly - poly[:, :1]
    q = p.gather(1, nxt[..., None].expand(-1, -1, 2))
    terms = p[..., 0] * q[..., 1] - p[..., 1] * q[..., 0]
    return (terms * (index < counts[:, None])).sum(1).abs() / 2


def second_area(poly, counts, q, angle):
    for sign in (-1, 1):
        a = angle + sign * DELTA
        normal = torch.stack((-sign * torch.sin(a), sign * torch.cos(a)), -1)
        poly, counts = clip_torch(poly, counts, normal, (normal * q).sum(-1))
    return area_torch(poly, counts)


@njit(cache=True)
def second_cpu(poly, q, angle):
    for sign in (-1, 1):
        a = angle + sign * DELTA
        normal = np.array([-sign * np.sin(a), sign * np.cos(a)])
        poly = clip_cpu(poly, normal, np.dot(normal, q))
    return poly


@njit(cache=True)
def robust_margin(poly, q):
    """Conservative margin for ||q-g||<=max(1000,||g||), in local units.

    Uses the stronger, easily certified sufficient condition ||q-g||<=1000
    for every point of the outer convex polygon.
    """
    worst = 0.0
    for g in poly:
        worst = max(worst, np.linalg.norm(g - q))
    return 1000.0 / SCALE - worst

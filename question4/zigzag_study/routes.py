"""Scene-independent Q4 search routes and finite development candidates.

Only stops are measured: the straight segments between them are not observations.
All routes start at the origin and are open (no forced return).  Geometry-based
coverage claims apply to a channel only after that channel was measured at every
required stop.  This module neither uses hidden source truth nor runs a simulator.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

R_DOMAIN = 1800.0
R_MIN = 1000.0


def ring(radius: float, n: int, phase: float = 0.0) -> np.ndarray:
    angles = phase + np.arange(n, dtype=float) * (2 * math.pi / n)
    return radius * np.column_stack((np.cos(angles), np.sin(angles)))


def path_length(points: Iterable) -> float:
    p = np.asarray(points, dtype=float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def with_origin(points: Iterable) -> np.ndarray:
    p = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(p) and np.linalg.norm(p[0]) < 1e-10:
        return p.copy()
    return np.vstack((np.zeros((1, 2)), p))


def nearest_neighbor_open_2opt(points: Iterable, max_passes: int = 80) -> np.ndarray:
    """Preserve every supplied point; fix the origin and optimize an open path.

    Suffix reversals are allowed, so the last point is not artificially fixed.
    Strict improvements and deterministic tie-breaking make results reproducible.
    """
    p = with_origin(points)
    remaining = list(range(1, len(p)))
    order = [0]
    while remaining:
        j = min(remaining, key=lambda k: (float(np.sum((p[k] - p[order[-1]]) ** 2)), k))
        order.append(j)
        remaining.remove(j)
    p = p[order].copy()
    for _ in range(max_passes):
        best_delta, best_pair = -1e-8, None
        for i in range(1, len(p) - 1):
            for j in range(i + 1, len(p)):
                before = float(np.linalg.norm(p[i - 1] - p[i]))
                after = float(np.linalg.norm(p[i - 1] - p[j]))
                if j + 1 < len(p):
                    before += float(np.linalg.norm(p[j] - p[j + 1]))
                    after += float(np.linalg.norm(p[i] - p[j + 1]))
                delta = after - before
                if delta < best_delta:
                    best_delta, best_pair = delta, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        p[i:j + 1] = p[i:j + 1][::-1]
    return p


def zigzag(inner: float, outer: float, n: int, phase: float = 0.0) -> np.ndarray:
    """One polar revolution with alternating radial stop positions."""
    if n < 4 or n % 2:
        raise ValueError("zigzag requires an even number of at least four stops")
    theta = phase + np.arange(n) * (2 * math.pi / n)
    radii = np.where(np.arange(n) % 2 == 0, inner, outer)
    return with_origin(radii[:, None] * np.column_stack((np.cos(theta), np.sin(theta))))


def wave(inner: float, outer: float, lobes: int, n: int) -> np.ndarray:
    """Sinusoidal radial stop placement; actual robot motion is piecewise linear."""
    theta = np.arange(n) * (2 * math.pi / n)
    radii = (inner + outer) / 2 - (outer - inner) / 2 * np.cos(lobes * theta)
    return with_origin(radii[:, None] * np.column_stack((np.cos(theta), np.sin(theta))))


def spiral(outer: float, turns: float, n: int, inner: float = 80.0) -> np.ndarray:
    theta = np.linspace(0.0, 2 * math.pi * turns, n)
    radii = np.linspace(inner, outer, n)
    return with_origin(radii[:, None] * np.column_stack((np.cos(theta), np.sin(theta))))


def _legacy_farthest() -> np.ndarray:
    # Intentionally reproduce the old static ordering, including its huge jumps.
    pool = [(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n))
            for r, n in ((700, 16), (1200, 24), (1750, 32), (2050, 32))
            for i in range(n)]
    pool.extend((float(x), float(y)) for x in (-1000, 0, 1000)
                for y in (-1000, 0, 1000))
    selected = []
    while pool and len(selected) < 70:
        if not selected:
            k = max(range(len(pool)), key=lambda i: math.hypot(*pool[i]))
        else:
            k = max(range(len(pool)), key=lambda i: min(math.dist(pool[i], q) for q in selected))
        selected.append(pool.pop(k))
    return with_origin(selected)


def scaffold_points() -> np.ndarray:
    """45 vertices of all 700 m squares intersecting the source disk.

    Every such square has diameter 700 sqrt(2) < 1000.  Its vertices certify
    visibility for every source position and every 180-degree orientation in
    that square.  Routing the same vertices preserves this continuous guarantee.
    """
    axis = np.arange(-2100.0, 2101.0, 700.0)
    vertices = set()
    for x in axis[:-1]:
        for y in axis[:-1]:
            lo = np.array([x, y])
            hi = lo + 700.0
            closest = np.maximum(lo, np.minimum(np.zeros(2), hi))
            if float(closest @ closest) <= R_DOMAIN ** 2:
                vertices.update(((x, y), (x + 700, y), (x + 700, y + 700), (x, y + 700)))
    ordered = [(0.0, 0.0)] + [p for p in sorted(vertices) if p != (0.0, 0.0)]
    return nearest_neighbor_open_2opt(ordered)


def zigzag_certificate(inner: float, outer: float, n: int) -> dict:
    """Sufficient continuous certificate via a triangulated staggered annulus.

    This is deliberately sufficient only: a false result means uncertified,
    not a proven miss.  n/2 inner and n/2 outer vertices are staggered by 2pi/n.
    """
    if n < 6 or n % 2 or not 0 < inner < outer:
        return {"certified": False, "reason": "invalid staggered-annulus parameters"}
    m = n // 2
    step = 2 * math.pi / m
    diameters = [inner, 2 * inner * math.sin(step / 2),
                 2 * outer * math.sin(step / 2),
                 math.sqrt(inner ** 2 + outer ** 2 - 2 * inner * outer * math.cos(step / 2))]
    inradius = outer * math.cos(step / 2)
    diameter = max(diameters)
    return {"certified": bool(inradius >= R_DOMAIN and diameter <= R_MIN),
            "method": "origin_fan_plus_staggered_annulus_triangles",
            "outer_polygon_inradius_m": inradius,
            "maximum_triangle_diameter_m": diameter,
            "distance_margin_m": R_MIN - diameter,
            "required_channel_scans": n + 1}


def _route(route_id: str, name: str, family: str, points: Iterable,
           params: dict | None = None, **extra) -> dict:
    p = with_origin(points)
    return {"id": route_id, "name": name, "family": family,
            "params": params or {}, "points": p.tolist(),
            "path_length_m": path_length(p), "stops": len(p), **extra}


def baseline_routes() -> list[dict]:
    farthest = _legacy_farthest()
    zz = zigzag(990.0, 1900.0, 24)
    cert = zigzag_certificate(990, 1900, 24)
    # Preserve the exact old spiral: t_i = i*pi/9, i=0,...,119.
    old_spiral = spiral(2120.0, 119 / 18, 120)
    return [
        _route("single_ring", "旧单圈", "single_ring", ring(1800, 36),
               {"radius": 1800, "n": 36}, legacy_id="single_ring"),
        _route("two_layer", "旧双层", "two_layer", np.vstack((ring(900, 24), ring(1950, 32))),
               {"inner": 900, "outer": 1950, "n_inner": 24, "n_outer": 32}, legacy_id="two_layer"),
        _route("spiral", "旧螺旋", "spiral", old_spiral,
               {"inner": 80, "outer": 2120, "turns": 119 / 18, "n": 120}, legacy_id="spiral"),
        _route("farthest_static", "旧最远点次序", "farthest_static", farthest,
               {"n": 70}, legacy_id="gap_candidates"),
        _route("farthest_reordered", "旧最远点集重排", "farthest_reordered",
               nearest_neighbor_open_2opt(farthest), {"n": 70, "ordering": "nearest_neighbor_open_2opt"}),
        _route("zigzag_certified", "锯齿单次绕行（25停点证书）", "zigzag", zz,
               {"inner": 990, "outer": 1900, "n": 24}, certificate=cert),
        _route("zigzag_certified_reordered", "同锯齿点集重排", "zigzag_reordered",
               nearest_neighbor_open_2opt(zz),
               {"inner": 990, "outer": 1900, "n": 24, "ordering": "nearest_neighbor_open_2opt"},
               certificate=cert),
        _route("wave_baseline", "六瓣平滑起伏", "wave", wave(900, 2050, 6, 48),
               {"inner": 900, "outer": 2050, "lobes": 6, "n": 48}),
    ]


def candidate_routes() -> list[dict]:
    """Finite parameter menu fixed before inspecting development scenes.

    25 single circles, 40 two-layer routes, 36 spirals, 40 zigzags, and 36 waves.
    Parameter selection belongs to the benchmark driver, never this module.
    """
    out = []
    for radius in (1600, 1800, 1900, 2050, 2200):
        for n in (12, 20, 28, 36, 48):
            out.append(_route(f"ring_r{radius}_n{n}", f"单圈 r={radius}, n={n}",
                              "single_ring", ring(radius, n), {"radius": radius, "n": n}))
    for inner in (600, 800, 950):
        for outer in (1900, 2050, 2200):
            for ni in (12, 18):
                for no in (18, 26):
                    phase = -math.pi / no
                    p = np.vstack((ring(inner, ni), ring(outer, no, phase)))
                    params = {"inner": inner, "outer": outer, "n_inner": ni,
                              "n_outer": no, "outer_phase": phase}
                    out.append(_route(f"two_{inner}_{outer}_{ni}_{no}",
                                      f"双层 {inner}/{outer}, {ni}+{no}点", "two_layer", p, params))
    # Four small certified point sets also traversed inner-ring then outer-ring.
    for inner, outer, n in ((990, 1900, 24), (980, 1870, 24), (980, 1930, 28), (990, 1950, 32)):
        m = n // 2
        p = np.vstack((ring(inner, m), ring(outer, m, -math.pi / m)))
        params = {"inner": inner, "outer": outer, "n_inner": m, "n_outer": m,
                  "outer_phase": -math.pi / m}
        out.append(_route(f"two_staggered_{inner}_{outer}_{n}", f"交错双层 {inner}/{outer}, {n}点",
                          "two_layer", p, params, certificate=zigzag_certificate(inner, outer, n)))
    for outer in (1950, 2150, 2350):
        for turns in (1.5, 2.5, 3.5, 4.5):
            for n in (28, 40, 56):
                out.append(_route(f"spiral_{outer}_{turns}_{n}", f"螺旋 {turns}圈, {n}点",
                                  "spiral", spiral(outer, turns, n),
                                  {"inner": 80, "outer": outer, "turns": turns, "n": n}))
    for inner in (600, 900, 1200):
        for outer in (1950, 2150, 2350):
            for n in (20, 28, 36, 44):
                out.append(_route(f"zigzag_{inner}_{outer}_{n}", f"锯齿 {inner}/{outer}, {n}点",
                                  "zigzag", zigzag(inner, outer, n),
                                  {"inner": inner, "outer": outer, "n": n},
                                  certificate=zigzag_certificate(inner, outer, n)))
    for inner, outer, n in ((990, 1900, 24), (980, 1870, 24), (980, 1930, 28), (990, 1950, 32)):
        out.append(_route(f"zigzag_{inner}_{outer}_{n}", f"紧凑锯齿 {inner}/{outer}, {n}点",
                          "zigzag", zigzag(inner, outer, n), {"inner": inner, "outer": outer, "n": n},
                          certificate=zigzag_certificate(inner, outer, n)))
    for inner in (700, 1000):
        for outer in (1950, 2150):
            for lobes in (4, 6, 8):
                for n in (32, 48, 64):
                    out.append(_route(f"wave_{inner}_{outer}_{lobes}_{n}",
                                      f"波浪 {inner}/{outer}, {lobes}瓣, {n}点", "wave",
                                      wave(inner, outer, lobes, n),
                                      {"inner": inner, "outer": outer, "lobes": lobes, "n": n}))
    return out

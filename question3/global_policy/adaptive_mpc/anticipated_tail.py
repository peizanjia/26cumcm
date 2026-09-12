"""Predict shared unknown scans at future known-source service stops.

This is a continuation-value surrogate, not a coverage certificate.  It routes
through current known-source centers, credits paid scans there, and covers only
the residual holes with dedicated stations.  Predicted masks never enter World.
"""
from copy import copy

import numpy as np

from ..model import ARENA_R, RECEIVE_MIN
from .exploration import frontier_candidates


def _insertion(route, start, point):
    """Cheapest insertion into an open path; a free last endpoint is allowed."""
    choices = []
    for index in range(len(route) + 1):
        before = np.asarray(start if index == 0 else route[index - 1])
        extra = float(np.linalg.norm(before - point))
        if index < len(route):
            after = np.asarray(route[index])
            extra += float(np.linalg.norm(point - after) - np.linalg.norm(before - after))
        choices.append((extra, index))
    return min(choices)


def _guaranteed_station_tail(world, residual, start, removed, y,
                             unknown_channels, stations):
    """Bounded baseline continuation when optional anticipation cannot progress.

    Discard optimistic future-center credits and reuse the caller's full-disk
    station certificate. Each station is an explicit exact-position scan in
    this fallback route, so it is not subject to a predicted center's erosion.
    A stale station list is repaired with uncovered-cell centers; each repair
    covers at least one previously missing whole cell and therefore terminates.
    """
    from .planning import open_route

    coverage = world.coverage
    bits = coverage.covered.copy()
    for channel in unknown_channels:
        bits[channel-1] |= coverage.mask_at(y)
    ids = [target.channel for target in world.unknown()]
    route = [target.center.copy() for target in world.active() if target.channel != removed]
    seconds = 0.

    def add_station(point):
        nonlocal seconds
        mask = coverage.mask_at(point)
        needed = [channel for channel in ids if np.any(mask & ~bits[channel-1])]
        if not needed:
            return False
        if not any(np.linalg.norm(point-old) < 1e-7 for old in route):
            route.append(np.asarray(point).copy())
        for channel in needed:
            bits[channel-1] |= mask
            seconds += 6.
        return True

    for station in stations:
        add_station(np.asarray(station))
    # At least one entire square disappears per repair. Coverage validation
    # bounds square sides at150 m, so projecting a boundary cell center into
    # R1800 still puts all four corners well within the 1000 m detection disk.
    for _ in range(len(coverage.centers)):
        holes = ~bits[np.asarray(ids)-1] if ids else np.zeros((0,len(coverage.centers)),dtype=bool)
        if not holes.any():
            break
        cell = int(np.flatnonzero(holes.any(axis=0))[0])
        point = coverage.centers[cell].copy()
        point *= min(1.,(ARENA_R-1e-5)/max(1.,float(np.linalg.norm(point))))
        if not add_station(point):
            raise ArithmeticError('Coverage cell-center fallback violated its progress invariant')
    else:
        if ids and (~bits[np.asarray(ids)-1]).any():
            raise ArithmeticError('Coverage cell-center fallback exhausted its finite cell bound')
    length,order = open_route(start,route)
    cost = length/5 + seconds + sum(value for channel,value in residual.items() if channel != removed)
    return float(cost), [np.asarray(route[index]).tolist() for index in order]


def anticipated_tail(world, residual, start, removed, y, unknown_channels,
                     stations, early=False):
    """Return (predicted remaining seconds, route), leaving public state intact.

    The base continuation visits current known-source centers.  Its predicted
    detection circles are eroded by 20 m to tolerate small final clear-point
    offsets; larger future route/posterior changes are still an approximation.
    Real termination always uses actual accepted scans and World.coverage.

    Unknown discoveries are not sampled here. All still-unknown channels are
    conservatively budgeted as requiring a whole-map coverage certificate.
    Every predicted extra channel measurement is charged six seconds.
    """
    from .planning import open_route

    p = world.params
    coverage = world.coverage
    bits = coverage.covered.copy()
    ids = [target.channel for target in world.unknown()]
    mask_at_y = coverage.mask_at(y)
    for channel in unknown_channels:
        bits[channel - 1] |= mask_at_y
    known = [target for target in world.active() if target.channel != removed]
    points = [target.center.copy() for target in known]
    _, order = open_route(start, points)
    route = [points[index] for index in order]
    threshold = p.early_unknown_scan_gain if early else p.unknown_scan_gain
    scan_seconds = 0.

    if p.stop_scans:
        for index in order:
            center = points[index]
            far = np.abs(coverage.centers - center) + coverage.half
            # Conservative for <=20 m displacement of the forecast stop. This
            # deliberately does not claim a current-radius uncertainty guarantee.
            predicted_mask = np.sum(far * far, axis=1) <= (RECEIVE_MIN - 20.) ** 2 - 1e-5
            for channel in ids:
                target = world.targets[channel]
                if target.measured_at(center):
                    continue
                holes = ~bits[channel - 1]
                gain_cells = int(np.count_nonzero(predicted_mask & holes))
                if not gain_cells:
                    continue
                gain = gain_cells / len(coverage.centers)
                remaining = float(np.mean(holes))
                if gain >= threshold or (remaining <= threshold and gain >= .85 * remaining):
                    bits[channel - 1] |= predicted_mask
                    scan_seconds += 6.

    candidates = [np.asarray(q, dtype=float).copy() for q in stations]
    # Existing known stops can receive a mandatory small-gap scan without
    # additional travel, even if the normal gain threshold excluded that scan.
    candidates += [point.copy() for point in points]
    if ids:
        needed = (~bits[np.asarray(ids) - 1]).sum(axis=0)
        missing = coverage.centers[needed > 0]
        if len(missing):
            centroid = np.average(coverage.centers, axis=0, weights=needed)
            # A few cheap residual-hole candidates allow an obsolete station to
            # be replaced; there is no repeated whole-arena search per action.
            sample = missing[np.linspace(0, len(missing)-1, min(8, len(missing)), dtype=int)]
            candidates += [centroid] + list(sample)
    if candidates:
        candidates = np.asarray(candidates)
        norms = np.linalg.norm(candidates, axis=1)
        candidates *= np.minimum(1., (ARENA_R-1e-5)/np.maximum(norms, 1.))[:, None]
        candidates = list(np.unique(np.round(candidates, 7), axis=0))
    else:
        candidates = []
    def forecast_mask(point):
        if any(np.linalg.norm(point-known_point) < 1e-6 for known_point in points):
            far = np.abs(coverage.centers-point)+coverage.half
            return np.sum(far*far,axis=1) <= (RECEIVE_MIN-20.)**2-1e-5
        return coverage.mask_at(point)

    masks = [forecast_mask(point) for point in candidates]
    remaining_candidates = list(range(len(candidates)))
    for _ in range(80):
        holes = ~bits[np.asarray(ids) - 1] if ids else np.zeros((0, len(coverage.centers)), dtype=bool)
        if not holes.any():
            break
        best = None
        total_holes = float(holes.sum())
        totals = holes.sum(axis=1)
        for index in remaining_candidates:
            gains = np.sum(holes & masks[index], axis=1)
            selected = [channel for channel, gain in zip(ids, gains) if gain > 0]
            if not selected:
                continue
            distance, insertion = _insertion(route, start, candidates[index])
            closes = np.count_nonzero((totals > 0) & (gains == totals))
            merit = (float(gains.sum()) / total_holes + p.frontier_closure_bonus * closes / max(1,len(ids)))
            score = merit / (max(0., distance)/5 + 6*len(selected) + 1.)
            value = (score, -distance, index, insertion, selected)
            if best is None or value[:2] > best[:2]:
                best = value
        if best is None:
            # Defensive repair if caller supplied an incomplete/stale station
            # list. Only a private coverage copy receives these forecast bits.
            forecast = copy(world)
            forecast.coverage = copy(coverage)
            forecast.coverage.covered = bits.copy()
            forecast.position = np.asarray(route[-1] if route else start).copy()
            options = frontier_candidates(forecast, None, limit=1)
            if not options:
                return _guaranteed_station_tail(world,residual,start,removed,y,unknown_channels,stations)
            point = np.asarray(options[0]['position'])
            mask = forecast_mask(point)
            # A full1000 m frontier may sit on a known center whose conservative
            # forecast is only980 m. It can then have no *forecast* gain at all.
            # Do not append the identical zero-progress repair until the guard.
            if not np.any(holes & mask):
                return _guaranteed_station_tail(world,residual,start,removed,y,unknown_channels,stations)
            candidates.append(point)
            masks.append(mask)
            remaining_candidates.append(len(candidates)-1)
            continue
        _, _, index, insertion, selected = best
        point = candidates[index]
        if not any(np.linalg.norm(point-existing) < 1e-6 for existing in route):
            route.insert(insertion, point.copy())
        for channel in selected:
            bits[channel-1] |= masks[index]
            scan_seconds += 6.
        remaining_candidates.remove(index)
    else:
        return _guaranteed_station_tail(world,residual,start,removed,y,unknown_channels,stations)

    length, final_order = open_route(start, route)
    route = [np.asarray(route[index]).tolist() for index in final_order]
    seconds = length/5 + scan_seconds + sum(value for channel,value in residual.items() if channel != removed)
    return float(seconds), route

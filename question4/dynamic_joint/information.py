"""Read-only, direction-compatible value of measuring any known Q4 source.

The quadrature is a planning approximation, not a feasible-set certificate.
Positive observations are weighted by the *joint* compatible position,
orientation and reception-radius mass. A missed measurement retains the
current conservative position polygon. Seconds saved use the explicit
localization potential ``0.30 * max(MEC_radius - 19.8, 0)``; travel is excluded
so the global planner can charge movement once for all sources at a stop.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import math
import weakref

import numpy as np

from question1.enclosing_circle import minimum_enclosing_circle
from question1.geometry import _clip, bearing_halfplanes
from question4.full_mission.service import (
    BEARING_ERROR, TAU, Target, _clip_disk, service_candidates,
)

CLEAR_RADIUS_M = 19.8
LOCALIZATION_SECONDS_PER_METRE = .30
MEASUREMENT_SECONDS = 5.
_CACHE: OrderedDict = OrderedDict()


def _cache(target):
    """Cache numerical predictions without modifying the target's state."""
    key = id(target)
    # Observation contents are immutable after append in Target.observe.
    signature = (len(target.observations), target.status,
                 float(target.radius), target.polygon.tobytes())
    item = _CACHE.get(key)
    if item is None or item[0]() is not target or item[1] != signature:
        item = (weakref.ref(target), signature, {})
        _CACHE[key] = item
    _CACHE.move_to_end(key)
    while len(_CACHE) > 64:
        _CACHE.popitem(last=False)
    return item[2]


def radius_potential(radius_m):
    """Estimated remaining localization effort in seconds, excluding travel."""
    return LOCALIZATION_SECONDS_PER_METRE * max(0., float(radius_m)-CLEAR_RADIUS_M)


def service_potential(target):
    return radius_potential(target.radius) if target.status == 'active' else 0.


def receiver_hit_weights(target: Target, q):
    """Integrate reception mass separately at every position sample.

    ``hit_masses[i]`` is P(position sample i AND reception at q), normalized
    by all mass compatible with history. Its sum is ``hit_probability``.
    ``hit_position_weights`` is P(position sample i | reception at q).
    Returned arrays are copies; neither probabilities nor zero quadrature mass
    may be used to declare a source absent.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('measurement position must be a finite coordinate pair')
    cached = _cache(target)
    key = ('reception', float(q[0]), float(q[1]))
    if key in cached:
        return {k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in cached[key].items()}
    # _quadrature caches pure numerical integration on Target; it does not
    # change geometry, observation history or the optical fallback state.
    points, weights, segments, omnis, total = target._quadrature()
    distances = np.linalg.norm(q-points, axis=1)
    joint = np.zeros(len(points))
    consistent = total > 1e-12
    if consistent:
        if len(segments):
            idx = segments[:, 0].astype(int)
            beta = np.arctan2(q[1]-points[idx, 1], q[0]-points[idx, 0])
            start = (beta-math.pi/2) % TAU
            overlap = np.zeros(len(idx))
            for shift in (-TAU, 0., TAU):
                overlap += np.maximum(0., np.minimum(segments[:, 2], start+math.pi+shift)
                                      - np.maximum(segments[:, 1], start+shift))
            at_source = distances[idx] < 1e-10
            overlap[at_source] = segments[at_source, 2]-segments[at_source, 1]
            # Each angular interval has its OWN compatible radius interval.
            widths = np.maximum(0., segments[:, 4]-np.maximum(segments[:, 3], distances[idx]))
            np.add.at(joint, idx, overlap*widths/TAU)
        if len(omnis):
            idx = omnis[:, 0].astype(int)
            widths = np.maximum(0., omnis[:, 2]-np.maximum(omnis[:, 1], distances[idx]))
            np.add.at(joint, idx, widths)
        joint /= total
        # Roundoff cannot create joint hit mass above the full position mass.
        joint = np.minimum(np.maximum(joint, 0.), weights)
    else:
        # Finite integration missed a feasible remnant. Use a deliberately
        # labelled heuristic, never an impossibility or absence conclusion.
        joint = .5*weights*(distances <= 1500.)
    hit = float(np.clip(joint.sum(), 0., 1.))
    conditional = joint/hit if hit > 1e-12 else np.zeros(len(points))
    result = dict(points=points.copy(), position_weights=weights.copy(),
                  hit_masses=joint, hit_probability=hit,
                  hit_position_weights=conditional, consistent=consistent)
    if len(cached) > 768:
        cached.clear()
    cached[key] = result
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in result.items()}


def _hypothetical_positive(target, q, g, error_deg):
    if np.linalg.norm(g-q) <= 5.:
        # A near response alone already gives a radius < 19.8 m. This outer
        # disk matches the 48-sided conservative disk used by Target.observe.
        return min(float(target.radius), 5./math.cos(math.pi/48)), q.copy()
    delta = g-q
    angle = math.degrees(math.atan2(delta[1], delta[0]))+error_deg
    a, b = bearing_halfplanes([q], [angle], BEARING_ERROR)
    poly = target.polygon.copy()
    for normal, offset in zip(a, b):
        poly = _clip(poly, normal, offset, 1e-8)
    if not len(poly):
        return float(target.radius), target.center.copy()
    if np.max(np.linalg.norm(poly-q, axis=1)) > 1500.+1e-8:
        poly = _clip_disk(poly, q, 1500.)
    if not len(poly):
        return float(target.radius), target.center.copy()
    circle = minimum_enclosing_circle(poly)
    return min(float(target.radius), circle.radius), circle.center


def measurement_value(target: Target, q, current_channel=None, samples=5):
    """Approximate one-stop localization benefit for one active source.

    Five deterministic position quantiles (configurable) are taken from the
    *conditional-on-hit* distribution. Each uses the same three uniform-error
    quantiles (-2/3, 0, +2/3 degree). No-signal leaves the geometric radius and
    center unchanged. ``gross_saving_s`` excludes both action and travel cost;
    ``net_saving_s`` subtracts measurement and the supplied receiver switch.
    Repeated radio observations at the same position have zero new value.
    """
    q = np.asarray(q, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all():
        raise ValueError('measurement position must be a finite coordinate pair')
    if int(samples) != samples or samples < 1:
        raise ValueError('samples must be a positive integer')
    count = int(samples)
    cost = MEASUREMENT_SECONDS + float(current_channel is not None and current_channel != target.channel)
    base = dict(valid=False, channel=int(target.channel), position=q.tolist(),
                hit_probability=0., expected_radius_m=float(target.radius),
                expected_center=target.center.tolist(), gross_saving_s=0.,
                cost_s=cost, net_saving_s=-cost,
                old_potential_s=service_potential(target),
                expected_potential_s=service_potential(target),
                hit_expected_radius_m=float(target.radius),
                hit_expected_center=target.center.tolist(),
                miss_radius_m=float(target.radius), samples=count, error_samples=3,
                model='conditional_hit_quadrature_linear_localization_potential')
    if target.status != 'active':
        return dict(base, reason='target_not_active')
    if target.radius <= CLEAR_RADIUS_M:
        return dict(base, reason='already_certifiably_clearable')
    if any(o['result'] in ('direction', 'near', 'no_signal')
           and np.linalg.norm(q-np.asarray(o['position'])) <= .02
           for o in target.observations):
        return dict(base, reason='repeated_fixed_measurement')
    cached = _cache(target)
    key = ('value', float(q[0]), float(q[1]), current_channel, count)
    if key in cached:
        return deepcopy(cached[key])
    reception = receiver_hit_weights(target, q)
    probability = reception['hit_probability']
    if probability <= 1e-12:
        result = dict(base, reason='no_compatible_reception_mass',
                      quadrature_consistent=reception['consistent'])
        cached[key] = result
        return deepcopy(result)
    quantiles = (np.arange(count)+.5)/count
    selected = np.minimum(np.searchsorted(np.cumsum(reception['hit_position_weights']), quantiles),
                          len(reception['points'])-1)
    radii, centers, potentials = [], [], []
    for index in selected:
        for error in (-2./3., 0., 2./3.):
            radius, center = _hypothetical_positive(target, q, reception['points'][index], error)
            radii.append(radius)
            centers.append(center)
            potentials.append(radius_potential(radius))
    hit_radius = float(np.mean(radii))
    hit_center = np.mean(centers, axis=0)
    expected_potential = probability*float(np.mean(potentials))+(1-probability)*base['old_potential_s']
    gross = max(0., base['old_potential_s']-expected_potential)
    expected_radius = probability*hit_radius+(1-probability)*target.radius
    expected_center = probability*hit_center+(1-probability)*target.center
    result = dict(base, valid=gross > 1e-9, reason='positive_localization_value' if gross > 1e-9 else 'no_geometric_gain',
                  hit_probability=probability, gross_saving_s=float(gross),
                  net_saving_s=float(gross-cost), expected_radius_m=float(expected_radius),
                  expected_center=expected_center.tolist(), hit_expected_radius_m=hit_radius,
                  hit_expected_center=hit_center.tolist(), expected_potential_s=float(expected_potential),
                  quadrature_consistent=reception['consistent'])
    cached[key] = result
    return deepcopy(result)


def known_scan_candidates(target, current, params=None, next_hint=None):
    """Enumerate legacy service candidates without committing optical state.

    If the selected action has reason optical_finite_cover, the caller must
    commit the corresponding fallback on the real target before executing it.
    Candidate enumeration itself does not start or advance the real cover.
    """
    shadow = deepcopy(target)
    return service_candidates(shadow, current, params or {}, next_hint)


def coupled_points(targets, current, base_points=(), limit=12):
    """Actual non-fixed points that may help multiple nearby active sources.

    Return candidate coordinates, not actions or promises of information gain.
    The planner evaluates every returned point against every source. Pair
    centers and offsets toward the current path deliberately include locations
    distinct from either source's individual service point.
    """
    if int(limit) <= 0:
        return []
    current = np.asarray(current, float)
    active = [t for t in targets if t.status == 'active' and t.radius > CLEAR_RADIUS_M]
    raw = []
    for i, left in enumerate(active):
        for right in active[i+1:]:
            delta = right.center-left.center
            distance = float(np.linalg.norm(delta))
            if distance > 2200.:
                continue
            midpoint = (left.center+right.center)/2
            cross = np.array([-delta[1], delta[0]])/max(distance, 1.)
            offset = float(np.clip(.2*distance, 50., 220.))
            for q in (midpoint, midpoint+offset*cross, midpoint-offset*cross,
                      .75*midpoint+.25*current):
                raw.append((float(np.linalg.norm(q-current)), q))
    # Bend individual candidate points slightly toward another useful source.
    for item in base_points:
        q = np.asarray(item['position'] if isinstance(item, dict) else item, float)
        for target in active:
            distance = np.linalg.norm(target.center-q)
            if 80. < distance < 900.:
                blended = .8*q+.2*target.center
                raw.append((float(np.linalg.norm(blended-current)), blended))
    unique = []
    for _, q in sorted(raw, key=lambda pair: pair[0]):
        if np.linalg.norm(q-current) < 1.:
            continue
        if all(np.linalg.norm(q-p) >= 35. for p in unique):
            unique.append(np.round(q, 2))
            if len(unique) >= int(limit):
                break
    return unique

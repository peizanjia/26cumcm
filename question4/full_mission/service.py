"""Observation-only localization and optical clearance for Q4.

Geometry is a conservative outer bound, while the finite probability quadrature
is only a planning heuristic. In particular no-signal never removes a 1000 m
position disk: the source may be pointing away. This module cannot read truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from question1.geometry import _clip, bearing_halfplanes, polygon_area
from question1.enclosing_circle import minimum_enclosing_circle

TAU = 2 * math.pi
BEARING_ERROR = 1.005


def _outer_disk(center, radius, count=48):
    angles = (np.arange(count) + .5) * TAU / count
    return np.asarray(center) + radius / math.cos(math.pi / count) * np.column_stack((np.cos(angles), np.sin(angles)))


def _clip_disk(poly, center, radius, count=48):
    result = poly.copy()
    for theta in np.arange(count) * TAU / count:
        normal = np.array([math.cos(theta), math.sin(theta)])
        result = _clip(result, normal, float(np.dot(normal, center) + radius), 1e-8)
    return result


def _bearing(response):
    for key in ('svd_deg', 'bearing_deg', 'direction_deg', 'bearing'):
        if key in response:
            return float(response[key])
    raise ValueError('direction feedback must include svd_deg or bearing_deg')


def _point_polygon_distance(q, poly):
    if not len(poly):
        return math.inf
    if len(poly) == 1:
        return float(np.linalg.norm(q-poly[0]))
    edges = np.roll(poly, -1, axis=0)-poly
    rel = q-poly
    cross = edges[:, 0]*rel[:, 1]-edges[:, 1]*rel[:, 0]
    if len(poly) >= 3 and (np.all(cross >= -1e-7) or np.all(cross <= 1e-7)):
        return 0.0
    norm = np.sum(edges*edges, axis=1)
    t = np.clip(np.divide(np.sum(rel*edges, axis=1), norm,
                          out=np.zeros_like(norm), where=norm > 0), 0, 1)
    return float(np.min(np.linalg.norm(q-(poly+t[:, None]*edges), axis=1)))


def _sample_polygon(poly, n=32):
    """Deterministic equal-area triangle quadrature, not a coverage certificate."""
    if len(poly) <= 2 or polygon_area(poly) < 1e-7:
        t = (np.arange(n)+.5)/n
        return poly[0][None, :]*(1-t[:, None])+poly[-1][None, :]*t[:, None]
    center = poly.mean(axis=0)
    nxt = np.roll(poly, -1, axis=0)
    rel, rel2 = poly-center, nxt-center
    areas = np.abs(rel[:, 0]*rel2[:, 1]-rel[:, 1]*rel2[:, 0])/2
    cdf = np.cumsum(areas)/areas.sum()
    bins = np.minimum(np.searchsorted(cdf, (np.arange(n)+.5)/n), len(poly)-1)
    u = np.sqrt((np.arange(n)*.6180339887498949+.31) % 1)
    v = (np.arange(n)*.4142135623730951+.23) % 1
    return (1-u[:, None])*center + u[:, None]*((1-v[:, None])*poly[bins]+v[:, None]*nxt[bins])


@dataclass
class Target:
    channel: int
    status: str = 'unknown'
    polygon: np.ndarray = field(default_factory=lambda: _outer_disk([0., 0.], 1800.))
    center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    radius: float = 1800. / math.cos(math.pi/48)
    observations: list[dict] = field(default_factory=list)
    radio_steps: int = 0
    stagnant_steps: int = 0
    first_bearing: dict | None = None
    optical_points: list[list[float]] = field(default_factory=list)
    optical_index: int = 0
    optical_started: bool = False
    _probability_cache: Any = field(default=None, repr=False)

    def observe(self, q, response):
        q = np.asarray(q, dtype=float)
        kind = response.get('measure_result', response.get('clear_result'))
        if kind is None:
            raise ValueError('observation must have measure_result or clear_result')
        old_radius = self.radius
        was_active = self.status == 'active'
        record = dict(position=q.tolist(), result=str(kind))
        if kind == 'direction':
            record['bearing_deg'] = _bearing(response)
        self.observations.append(record)
        self._probability_cache = None
        if kind == 'success':
            self.status = 'cleared'
            return
        if kind in ('no_target_in_range', 'failed', 'failure'):
            # The exclusion disk is not convex. Retain it in history and apply
            # it to probability samples; never fill a hole by unsafe clipping.
            self._advance_optical(q)
            return
        if kind not in ('direction', 'near', 'no_signal'):
            raise ValueError(f'unsupported feedback: {kind}')
        if was_active:
            self.radio_steps += 1
        if kind in ('direction', 'near'):
            self.status = 'active'
            poly = _clip_disk(self.polygon, q, 5. if kind == 'near' else 1500.)
            if kind == 'direction':
                a, b = bearing_halfplanes([q], [record['bearing_deg']], BEARING_ERROR)
                for normal, offset in zip(a, b):
                    poly = _clip(poly, normal, offset, 1e-8)
                if self.first_bearing is None:
                    self.first_bearing = record.copy()
            if not len(poly):
                raise ArithmeticError(f'channel {self.channel}: inconsistent positive geometry')
            self.polygon = poly
            circle = minimum_enclosing_circle(poly)
            self.center, self.radius = circle.center, circle.radius
        if was_active:
            self.stagnant_steps = self.stagnant_steps+1 if self.radius >= old_radius*.88 else 0

    def _advance_optical(self, q):
        if self.optical_started and self.optical_index < len(self.optical_points):
            if np.linalg.norm(q-np.array(self.optical_points[self.optical_index])) < .02:
                self.optical_index += 1

    def snapshot(self):
        return dict(channel=self.channel, status=self.status, center=self.center.tolist(),
                    radius_m=float(self.radius), polygon=self.polygon.tolist(),
                    observations=len(self.observations), radio_steps=self.radio_steps,
                    stagnant_steps=self.stagnant_steps, optical_started=self.optical_started,
                    optical_index=self.optical_index, optical_total=len(self.optical_points))

    def _quadrature(self):
        if self._probability_cache is not None:
            return self._probability_cache
        samples = _sample_polygon(self.polygon)
        failures = [o for o in self.observations if o['result'] in ('no_target_in_range', 'failed', 'failure')]
        keep = np.linalg.norm(samples, axis=1) <= 1800.+1e-7
        for o in failures:
            keep &= np.linalg.norm(samples-o['position'], axis=1) > 20.
        samples = samples[keep]
        # A finite quadrature can miss a narrow feasible remainder. This is
        # numerical scarcity, never a statement that the source is absent.
        if not len(samples):
            samples = _sample_polygon(self.polygon, 128)
            for o in failures:
                samples = samples[np.linalg.norm(samples-o['position'], axis=1) > 20.]
        if not len(samples):
            samples = self.center[None, :]
        radio = [o for o in self.observations if o['result'] in ('direction', 'near', 'no_signal')]
        rows = []
        omni = []
        # Equal prior mass for omni and directional types is a modelling choice.
        for index, g in enumerate(samples):
            geometry = [(float(np.linalg.norm(np.array(o['position'])-g)),
                         math.atan2(o['position'][1]-g[1], o['position'][0]-g[0]),
                         o['result'] != 'no_signal') for o in radio]
            lower = max([1000.] + [d for d, _, hit in geometry if hit])
            if lower > 1500.+1e-7:
                continue
            ends = {0., TAU}
            for d, beta, _ in geometry:
                if d > 1e-10:
                    ends.update(((beta-math.pi/2) % TAU, (beta+math.pi/2) % TAU))
            ends = sorted(ends)
            for start, end in zip(ends, ends[1:]):
                angle = (start+end)/2
                upper, allowed = 1500., True
                for d, beta, hit in geometry:
                    illuminated = d < 1e-10 or math.cos(angle-beta) >= 0
                    if hit and not illuminated:
                        allowed = False
                        break
                    if not hit and illuminated:
                        upper = min(upper, d)
                if allowed and upper > lower:
                    rows.append((index, start, end, lower, upper))
            upper = min([1500.] + [d for d, _, hit in geometry if not hit])
            if upper > lower:
                omni.append((index, lower, upper))
        segments = np.asarray(rows, dtype=float).reshape(-1, 5)
        omnis = np.asarray(omni, dtype=float).reshape(-1, 3)
        masses = np.zeros(len(samples))
        if len(segments):
            np.add.at(masses, segments[:, 0].astype(int),
                      (segments[:, 2]-segments[:, 1])*(segments[:, 4]-segments[:, 3])/TAU)
        if len(omnis):
            np.add.at(masses, omnis[:, 0].astype(int), omnis[:, 2]-omnis[:, 1])
        total = masses.sum()
        weights = masses/total if total > 1e-12 else np.ones(len(samples))/len(samples)
        result = (samples, weights, segments, omnis, float(total))
        self._probability_cache = result
        return result

    def hit_probability(self, q):
        """Integrate compatible angular/radius intervals at position samples."""
        q = np.asarray(q, float)
        samples, _, segments, omnis, total = self._quadrature()
        distance = np.linalg.norm(q-samples, axis=1)
        if total <= 1e-12:
            # Conservative planning fallback, not an impossibility certificate.
            return .5 if distance.min() <= 1500 else 0.
        mass = 0.
        if len(segments):
            idx = segments[:, 0].astype(int)
            angle = np.arctan2(q[1]-samples[idx, 1], q[0]-samples[idx, 0])
            start = (angle-math.pi/2) % TAU
            overlap = np.zeros(len(idx))
            for shift in (-TAU, 0., TAU):
                overlap += np.maximum(0., np.minimum(segments[:, 2], start+math.pi+shift)
                                       - np.maximum(segments[:, 1], start+shift))
            overlap[distance[idx] < 1e-10] = segments[distance[idx] < 1e-10, 2]-segments[distance[idx] < 1e-10, 1]
            width = np.maximum(0., segments[:, 4]-np.maximum(segments[:, 3], distance[idx]))
            mass += float(np.dot(overlap, width)/TAU)
        if len(omnis):
            mass += float(np.maximum(0., omnis[:, 2]-np.maximum(omnis[:, 1], distance[omnis[:, 0].astype(int)])).sum())
        return float(np.clip(mass/total, 0., 1.))

    def clear_probability(self, q):
        points, weights, _, _, _ = self._quadrature()
        return float(weights[np.linalg.norm(points-np.asarray(q), axis=1) <= 20.].sum())


def _optical_candidate(target, current):
    if not target.optical_started:
        first = target.first_bearing
        if first is None:
            return None
        theta = math.radians(first['bearing_deg'])
        direction = np.array([math.cos(theta), math.sin(theta)])
        side = np.array([-direction[1], direction[0]])
        origin = np.array(first['position'])
        xs = np.arange(0., 1500.+.01, 25.)
        paths = []
        for xx in (xs, xs[::-1]):
            for sign in (-1, 1):
                path = [origin+x*direction+sign*15*side for x in xx]
                path += [origin+x*direction-sign*15*side for x in xx[::-1]]
                paths.append(path)
        # A serpentine reversal alone starts at the same longitudinal end.
        # Compare all four corner entries to avoid an unnecessary 1500 m return.
        local = min(paths, key=lambda path: np.linalg.norm(current-path[0]))
        target.optical_points = [p.tolist() for p in local]
        target.optical_started = True
    tried = [np.array(o['position']) for o in target.observations if o['result'] in ('no_target_in_range', 'failed', 'failure')]
    while target.optical_index < len(target.optical_points):
        q = np.array(target.optical_points[target.optical_index])
        if _point_polygon_distance(q, target.polygon) > 20.+1e-6 or any(np.linalg.norm(q-p) < .02 for p in tried):
            target.optical_index += 1
            continue
        return dict(kind='clear', channel=target.channel, position=q.tolist(),
                    reason='optical_finite_cover', score_s=float(np.linalg.norm(q-current)/5+3),
                    hit_probability=target.clear_probability(q), optical_index=target.optical_index)
    return None


def service_candidates(target: Target, current, params, next_hint=None):
    """Return one-step actions ranked by heuristic remaining service seconds.

    score_s includes current-to-action travel, measurement/clear and approximate
    future localization/clearance. It excludes a possible channel switch.
    """
    if target.status != 'active':
        return []
    current = np.asarray(current, float)
    params = params or {}
    if target.radius <= 19.8:
        return [dict(kind='clear', channel=target.channel, position=target.center.tolist(),
                     reason='certified_mec_clear', score_s=float(np.linalg.norm(current-target.center)/5+5),
                     hit_probability=1., certified=True)]
    max_radio = int(params.get('max_radio_steps', 8))
    failed_count = sum(o['result'] in ('no_target_in_range', 'failed', 'failure') for o in target.observations)
    if target.optical_started or target.stagnant_steps >= max_radio or target.radio_steps >= max_radio+4 or failed_count >= 8:
        fallback = _optical_candidate(target, current)
        return [fallback] if fallback else []
    points, weights, _, _, _ = target._quadrature()
    mean = np.sum(points*weights[:, None], axis=0)
    vector = mean-current
    length = np.linalg.norm(vector)
    unit = vector/max(length, 1e-9)
    if length < 1e-7:
        positive = next(o for o in reversed(target.observations) if o['result'] in ('direction', 'near'))
        vector = mean-np.asarray(positive['position'])
        unit = vector/max(np.linalg.norm(vector), 1e-9)
    side = np.array([-unit[1], unit[0]])
    fraction = float(params.get('probe_fraction', .72))
    lateral = float(np.clip(target.radius*float(params.get('lateral_ratio', .25)), 18., 230.))
    raw = [(current+fraction*vector+sign*lateral*side, 'forward_lateral') for sign in (-1, 1)]
    raw += [(mean+sign*lateral*side, 'center_lateral') for sign in (-1, 1)]
    raw += [(current+.4*vector+sign*.5*lateral*side, 'short_probe') for sign in (-1, 1)]
    raw += [(mean, 'posterior_center'), (target.center, 'mec_center')]
    positive = next(o for o in reversed(target.observations) if o['result'] in ('direction', 'near'))
    anchor = np.asarray(positive['position'])
    anchor_vector = mean-anchor
    anchor_axis = anchor_vector/max(np.linalg.norm(anchor_vector), 1e-9)
    anchor_side = np.array([-anchor_axis[1], anchor_axis[0]])
    raw += [(anchor+.35*anchor_vector+sign*.6*lateral*anchor_side,
             'positive_anchor_probe') for sign in (-1, 1)]
    if next_hint is not None:
        onward = np.asarray(next_hint)-mean
        onward /= max(np.linalg.norm(onward), 1e-9)
        raw.append((mean+min(lateral, 120.)*onward, 'onward_service_probe'))
    measured = [np.array(o['position']) for o in target.observations if o['result'] in ('direction', 'near', 'no_signal')]
    failed = [np.array(o['position']) for o in target.observations if o['result'] in ('no_target_in_range', 'failed', 'failure')]
    candidates = []
    clear_limit = float(params.get('clear_try_radius', 48.))
    if target.radius <= clear_limit:
        for q in (target.center, mean):
            if any(np.linalg.norm(q-p) < .02 for p in failed):
                continue
            chance = target.clear_probability(q)
            if chance >= .16:
                candidates.append(dict(kind='clear', channel=target.channel, position=q.tolist(),
                                       reason='probabilistic_optical_try', hit_probability=chance,
                                       score_s=float(np.linalg.norm(current-q)/5 + 3+2*chance+(1-chance)*(14+target.radius/5)),
                                       certified=False))
    unique = []
    # Small quadrature of hypothetical bearings; measurements are deterministic
    # at a repeated site, which is excluded above. Exact planning is not claimed.
    quantiles = (np.arange(7)+.5)/7
    selected = np.minimum(np.searchsorted(np.cumsum(weights), quantiles), len(points)-1)
    for q, reason in raw:
        q = np.round(q, 2)
        if any(np.linalg.norm(q-p) < 1. for p in measured+unique):
            continue
        unique.append(q)
        chance = target.hit_probability(q)
        if chance < .04:
            continue
        posterior_radii = []
        posterior_centers = []
        for g in points[selected]:
            delta = g-q
            if np.linalg.norm(delta) <= 5.:
                posterior_radii.append(5.)
                posterior_centers.append(q)
                continue
            angle = math.degrees(math.atan2(delta[1], delta[0]))
            a, b = bearing_halfplanes([q], [angle], BEARING_ERROR)
            poly = target.polygon.copy()
            for normal, offset in zip(a, b):
                poly = _clip(poly, normal, offset, 1e-8)
            if len(poly):
                circle = minimum_enclosing_circle(poly)
                posterior_radii.append(circle.radius)
                posterior_centers.append(circle.center)
        after_radius = float(np.mean(posterior_radii)) if posterior_radii else target.radius
        center_travel = float(np.mean(np.linalg.norm(np.asarray(posterior_centers)-q, axis=1)))/5 if posterior_centers else length/5
        # The miss branch retains full geometric uncertainty but changes the
        # probability model on the next real observation. Penalize its likely
        # extra lateral move and measurement rather than claiming exact MPC.
        hit_tail = center_travel+5.+max(0., after_radius-19.8)*.18
        miss_tail = np.linalg.norm(mean-q)/5+18.+min(target.radius, 250.)*.32
        score = np.linalg.norm(q-current)/5+5.+chance*hit_tail+(1-chance)*miss_tail
        candidates.append(dict(kind='measure', channel=target.channel, position=q.tolist(),
                               reason=reason, score_s=float(score), hit_probability=chance,
                               expected_radius_m=after_radius, score_model='short_service_heuristic'))
    if not candidates:
        fallback = _optical_candidate(target, current)
        return [fallback] if fallback else []
    return sorted(candidates, key=lambda row: row['score_s'])

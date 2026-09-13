"""Adaptive deterministic integration of position/heading/radius constraints.

The convex position bound remains conservative. Numerical mass is used ONLY
for planning; absence and successful clearance never follow from quadrature.
"""
from __future__ import annotations

import numpy as np

from question4.full_mission.service import Target as BaseTarget, _sample_polygon, TAU


def integrate(samples, observations):
    """Vectorized exact heading/radius integral at each supplied position."""
    samples = np.asarray(samples, float)
    radio = [o for o in observations if o['result'] in ('direction', 'near', 'no_signal')]
    n = len(samples)
    if not radio:
        return (np.empty((0, 5)), np.column_stack((np.arange(n), np.full(n, 1000.),
                                                 np.full(n, 1500.))), np.full(n, 500.))
    pos = np.array([o['position'] for o in radio])
    hit = np.array([o['result'] != 'no_signal' for o in radio])
    delta = pos[None, :, :] - samples[:, None, :]
    d = np.linalg.norm(delta, axis=2)
    beta = np.arctan2(delta[:, :, 1], delta[:, :, 0])
    lower = np.maximum(1000., d[:, hit].max(axis=1) if hit.any() else 1000.)
    lower = np.broadcast_to(lower, (n,))
    ends = np.sort(np.column_stack((np.zeros(n), np.full(n, TAU),
                                   (beta-np.pi/2) % TAU, (beta+np.pi/2) % TAU)), axis=1)
    start, end = ends[:, :-1], ends[:, 1:]
    middle = (start+end)/2
    allowed = np.ones(middle.shape, bool)
    upper = np.full(middle.shape, 1500.)
    for j in range(len(radio)):
        illuminated = (np.cos(middle-beta[:, j, None]) >= 0) | (d[:, j, None] < 1e-10)
        if hit[j]:
            allowed &= illuminated
        else:
            upper = np.minimum(upper, np.where(illuminated, d[:, j, None], 1500.))
    good = allowed & (upper > lower[:, None]) & (end-start > 1e-14)
    i, j = np.nonzero(good)
    segments = np.column_stack((i, start[i, j], end[i, j], lower[i], upper[i, j]))
    oupper = np.minimum(1500., d[:, ~hit].min(axis=1) if (~hit).any() else 1500.)
    oupper = np.broadcast_to(oupper, (n,))
    idx = np.flatnonzero(oupper > lower)
    omnis = np.column_stack((idx, lower[idx], oupper[idx]))
    masses = np.zeros(n)
    np.add.at(masses, i, (end[i, j]-start[i, j])*(upper[i, j]-lower[i])/TAU)
    masses[idx] += oupper[idx]-lower[idx]
    return segments, omnis, masses


class Target(BaseTarget):
    integration_min = 128
    integration_max = 2048

    def _quadrature(self):
        if self._probability_cache is not None:
            return self._probability_cache
        failures = [o for o in self.observations if o['result'] in ('no_target_in_range', 'failed', 'failure')]
        count = self.integration_min
        while True:
            points = _sample_polygon(self.polygon, count)
            keep = np.linalg.norm(points, axis=1) <= 1800.+1e-7
            for o in failures:
                keep &= np.linalg.norm(points-o['position'], axis=1) > 20.
            points = points[keep]
            if len(points):
                segments, omnis, masses = integrate(points, self.observations)
                total = float(masses.sum())
                effective = total*total/max(float(np.dot(masses, masses)), 1e-30)
            else:
                total, effective = 0., 0.
            if effective >= 24 or count >= self.integration_max:
                break
            count = min(count*4, self.integration_max)
        if not len(points):
            points = self.center[None, :]
            segments, omnis, masses = integrate(points, self.observations)
            total = float(masses.sum())
        weights = masses/total if total > 1e-12 else np.ones(len(points))/len(points)
        self.integration_diagnostics = dict(requested=count, retained=len(points),
            effective=effective, consistent=total > 1e-12)
        self._probability_cache = points, weights, segments, omnis, total
        return self._probability_cache

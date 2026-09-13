"""Observation-only free support and directional recovery extension."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .frozen_dynamic.planner import Planner as BasePlanner, Parameters as BaseParameters
from question4.full_mission.planner import open_order
from .belief import Target
from .coverage import DirectionalCoverage


@dataclass
class Parameters(BaseParameters):
    strategy: str = 'free_joint'
    unknown_credit_cap_s: float = 60.
    adaptive_belief: bool = True
    elastic_coverage: bool = True
    near_anchor_probes: bool = True
    stationary_unknown_cap_s: float = 0.
    stationary_unknown_gain: float = .015
    integration_min: int = 128
    integration_max: int = 2048
    posterior_route_mix: float = 0.
    approach_clear_boundary: bool = False


class Planner(BasePlanner):
    def __init__(self, command, params=None, record=False):
        super().__init__(command, params or Parameters(), record)
        if self.params.adaptive_belief:
            self.targets = {c: Target(c) for c in self.targets}
            for t in self.targets.values():
                t.integration_min = self.params.integration_min
                t.integration_max = self.params.integration_max
        self.unknown_map = DirectionalCoverage(spacing=self.params.coverage_spacing,
                                               directions=self.params.coverage_directions)
        self.unknown_map.elastic_enabled = self.params.elastic_coverage
        self._stationary = False

    def _reference_plan(self):
        self.unknown_map.service_positions = [self._service_destination(t) for t in self.active()]
        super()._reference_plan()

    def _service_destination(self, target):
        mix = self.params.posterior_route_mix
        if not mix or target.radius <= 19.8:
            return target.center
        points, weights, _, _, mass = target._quadrature()
        if mass <= 1e-12:
            return target.center
        mean = (points*weights[:, None]).sum(axis=0)
        return (1-mix)*target.center+mix*mean

    def _forecast(self, q, known_values, selected_channels, clearing=None):
        if not self.params.posterior_route_mix:
            return super()._forecast(q, known_values, selected_channels, clearing)
        predicted = {r['channel']: r['expected_center'] for r in known_values
                     if r['channel'] in selected_channels}
        known = [np.asarray(predicted.get(t.channel, self._service_destination(t)))
                 for t in self.active() if t.channel != clearing]
        plan = [np.asarray(p) for p in self.search_plan]
        channels = [t.channel for t in self.unknown()] if self.search_needed() else []
        if plan and channels and selected_channels:
            plan = self.unknown_map.forecast_plan(channels, plan, q, selected_channels)
        nodes = known+plan
        if not nodes:
            return 0.
        order = open_order(q, nodes, passes=self.params.forecast_passes)
        points = np.vstack((q, [nodes[i] for i in order]))
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()/5)

    def _unknown_values(self, q, required=()):
        rows = super()._unknown_values(q, required)
        if self._stationary and self.params.stationary_unknown_cap_s:
            for r in rows:
                if r['gain'] < self.params.stationary_unknown_gain:
                    continue
                raw = r['gross_saving_s']/max(r['coverage_credit_scale'], 1e-12)
                r['gross_saving_s'] = min(self.params.stationary_unknown_cap_s, raw)
                r['net_saving_s'] = r['gross_saving_s']-r['cost_s']
                r['model'] = 'stationary_marginal_scan_no_additional_travel'
        return rows

    def assess_stop(self, required=(), first_channel=None):
        self._stationary = True
        try:
            super().assess_stop(required, first_channel)
        finally:
            self._stationary = False

    def _raw_candidates(self):
        rows = super()._raw_candidates()
        if self.params.approach_clear_boundary:
            for t in self.active():
                if t.radius > 19.8:
                    continue
                directions = [self.position-t.center]
                if self.search_plan:
                    directions.append(np.asarray(self.search_plan[0])-t.center)
                for d in directions:
                    length = np.linalg.norm(d)
                    if length < 1e-8:
                        continue
                    unit = d/length
                    rel = t.center-t.polygon
                    projection = rel@unit
                    # Furthest point along this ray in the intersection of
                    # 20m disks around EVERY polygon vertex. Convexity covers
                    # the whole conservative polygon, independent of belief.
                    limit = np.min(-projection+np.sqrt(np.maximum(0., projection**2+
                        (20.-1e-5)**2-np.sum(rel*rel, axis=1))))
                    q = t.center+min(length, max(0., limit))*unit
                    rows.append(dict(position=q.tolist(), reason='near_boundary_certified_clear',
                                     owner=t.channel, kind='clear', required=[]))
        if not self.params.near_anchor_probes:
            return rows
        for t in self.active():
            if not self._radio_eligible(t):
                continue
            positive = next((o for o in reversed(t.observations) if o['result'] == 'direction'), None)
            if positive is None:
                continue
            angle = math.radians(positive['bearing_deg'])
            direction = np.array([math.cos(angle), math.sin(angle)])
            side = np.array([-direction[1], direction[0]])
            anchor = np.array(positive['position'])
            # Anchor-relative distances include the previously undersampled
            # near end of a long first-bearing wedge, on both receiver sides.
            distance = np.linalg.norm(t.center-anchor)
            lengths = (min(100., .2*distance), min(220., .4*distance), min(400., .6*distance))
            for length in lengths:
                if length < 10.:
                    continue
                for sign in (-1, 1):
                    q = np.round(anchor+length*direction+sign*max(15., .55*length)*side, 2)
                    if self._covered_at(t, q):
                        continue
                    rows.append(dict(position=q.tolist(), reason='near_positive_anchor',
                                     owner=t.channel, kind='measure', required=[]))
        return rows

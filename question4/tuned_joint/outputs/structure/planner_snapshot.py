"""Public-observation planner with jointly priced coverage scan bundles.

Future scans are a conditional all-negative continuation, not observations or
an expectation over undiscovered emitters. Only inherited execute updates truth.
"""
from dataclasses import dataclass
import numpy as np

from question4.free_joint.planner import Parameters as BaseParameters, Planner as BasePlanner
from question4.full_mission.planner import open_order


@dataclass
class Parameters(BaseParameters):
    strategy: str = 'tuned_joint'
    future_scan_weight: float = 1.0
    bundle_scans: bool = True
    replacement_candidates: int = 6


class Planner(BasePlanner):
    def __init__(self, command, params=None, record=False):
        super().__init__(command, params or Parameters(), record)

    def _future_plan(self, q, selected_channels):
        channels = [t.channel for t in self.unknown()] if self.search_needed() else []
        plan = [np.asarray(p) for p in self.search_plan]
        if plan and channels and selected_channels:
            plan = self.unknown_map.forecast_plan(channels, plan, q, selected_channels)
        return channels, plan

    def _forecast(self, q, known_values, selected_channels, clearing=None):
        if not self.params.future_scan_weight:
            return super()._forecast(q, known_values, selected_channels, clearing)
        predicted = {r['channel']: r['expected_center'] for r in known_values
                     if r['channel'] in selected_channels}
        known = [np.asarray(predicted.get(t.channel, self._service_destination(t)))
                 for t in self.active() if t.channel != clearing]
        channels, plan = self._future_plan(q, selected_channels)
        nodes = known + plan
        if not nodes:
            return 0.
        order = open_order(q, nodes, passes=self.params.forecast_passes)
        points = np.vstack((q, [nodes[i] for i in order]))
        movement = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()/5.)
        # Price the explicit unknown-channel continuation. Service-center nodes
        # are routing proxies: future localization measurements remain represented
        # by the existing localization potential rather than invented commands.
        receiver = selected_channels[-1] if selected_channels else self.channel
        scans = 0.
        for i in order:
            if i < len(known):
                continue
            needed = self.unknown_map.required_channels(nodes[i], channels)
            if receiver in needed:
                needed = [receiver] + [c for c in needed if c != receiver]
            for c in needed:
                scans += 5. + float(c != receiver)
                receiver = c
        return movement + self.params.future_scan_weight*scans

    def _evaluate(self, row):
        ordinary = super()._evaluate(row)
        if not self.params.bundle_scans or row['required'] or not self.search_needed():
            return ordinary
        q = np.asarray(row['position'])
        channels, plan = self._future_plan(q, [])
        required = self.unknown_map.required_channels(q, channels)
        if not required or (ordinary is not None and set(required) <= set(ordinary['predicted_scan_channels'])):
            return ordinary
        after = self.unknown_map.forecast_plan(channels, plan, q, required)
        if len(after) >= len(plan):
            return ordinary
        bundled = super()._evaluate(dict(row, required=required))
        if bundled is None:
            return ordinary
        bundled.update(bundle_scan=True, replaced_future_stops=len(plan)-len(after),
                       score_model='joint_coverage_bundle_and_priced_negative_continuation')
        if ordinary is None or bundled['score_s'] < ordinary['score_s']:
            return bundled
        return ordinary

    def _raw_candidates(self):
        rows = super()._raw_candidates()
        if not self.params.replacement_candidates or not self.search_needed() or not self.search_plan:
            return rows
        channels = [t.channel for t in self.unknown()]
        anchors = sorted((r for r in rows if r['owner'] is not None and r['kind'] == 'measure'),
                         key=self._quick_score)[:4]
        added = 0
        for row in anchors:
            q = np.asarray(row['position'])
            support = min(self.search_plan, key=lambda p: np.linalg.norm(np.asarray(p)-q))
            for fraction in (.25, .5, .75):
                trial = np.round((1.-fraction)*q + fraction*np.asarray(support), 2)
                remaining = self.unknown_map.forecast_plan(channels, self.search_plan, trial, channels)
                if len(remaining) >= len(self.search_plan):
                    continue
                rows.append(dict(position=trial.tolist(), reason='joint_coverage_replacement',
                                 owner=row['owner'], kind='measure',
                                 required=self.unknown_map.required_channels(trial, channels)))
                added += 1
                if added >= self.params.replacement_candidates:
                    return rows
        return rows

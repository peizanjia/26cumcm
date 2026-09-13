"""Continuously movable coverage support, conditioned on actual observations."""
from __future__ import annotations

import numpy as np

from .frozen_dynamic.coverage import DirectionalCoverage as BaseCoverage, _array, _order
from question4.full_mission.planner import open_order


class DirectionalCoverage(BaseCoverage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.service_positions = []
        self.elastic_enabled = True
        self.elastic_changes = 0

    def completion_plan(self, channels, current):
        plan = _array(super().completion_plan(channels, current))
        if not len(plan) or not self.elastic_enabled:
            return plan.tolist()
        current = np.asarray(current)
        services = _array(self.service_positions)
        def cost(p):
            nodes = np.vstack((p, services))
            route = np.vstack((current, nodes[open_order(current, nodes, passes=2)]))
            return np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
        best = cost(plan)
        changed = 0
        # All positions may move; the first few receive more search effort at
        # this rolling step. Actual observations are immutable proof supports.
        for i in range(min(4, len(plan))):
            q = plan[i].copy()
            prev = current if i == 0 else plan[i-1]
            nxt = plan[min(i+1, len(plan)-1)]
            attractors = [prev, (prev+nxt)/2]
            if len(services):
                attractors.append(services[np.linalg.norm(services-q, axis=1).argmin()])
            for target in attractors:
                for fraction in (.25, .10, .04):
                    new = q+fraction*(target-q)
                    if np.linalg.norm(new-q) < 2.:
                        continue
                    trial = plan.copy(); trial[i] = new
                    value = cost(trial)
                    if value >= best-1.:
                        continue
                    if self.plan_certified(channels, trial, remember=True):
                        plan, best = trial, value
                        changed += 1
                        break
        self.elastic_changes += changed
        self._plan = [p.copy() for p in plan]
        self.last_plan_diagnostics.update(elastic_updates=changed,
            elastic_updates_total=self.elastic_changes,
            positions_fixed=False, continuous_guard='short_edge_triangles')
        return _order(current, plan)

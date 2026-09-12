"""Full Q4 mission policy using public responses only.

Search-stop geometry is shared with actual service stops. Approximate service
values and an open task tour choose one action, followed by replanning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from question4.zigzag_study.routes import ring, zigzag, scaffold_points
from question4.zigzag_study.study import certificate_details
from .service import Target, service_candidates


@dataclass
class Parameters:
    strategy: str = "joint"
    inner: float = 980.0
    outer: float = 1870.0
    service_detour: float = 650.0
    side_gain: float = .025
    side_scans: bool = True
    known_side_scans: bool = True
    prune_coverage: bool = True
    clear_try_radius: float = 48.0
    probe_fraction: float = .72
    lateral_ratio: float = .25
    max_radio_steps: int = 8
    service_tail_weight: float = .30
    max_commands: int = 7000


def open_order(current, positions, passes=4):
    """Nearest-neighbor open tour with distance-matrix 2-opt; no fixed endpoint."""
    p = np.vstack((current, np.asarray(positions).reshape(-1, 2)))
    n = len(p)
    if n < 2:
        return []
    distance = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=2)
    left = list(range(1, n))
    order = [0]
    while left:
        j = min(left, key=lambda k: (distance[order[-1], k], k))
        order.append(j); left.remove(j)
    for _ in range(passes):
        best, pair = -1e-7, None
        for i in range(1, n-1):
            for j in range(i+1, n):
                delta = distance[order[i-1], order[j]] - distance[order[i-1], order[i]]
                if j+1 < n:
                    delta += distance[order[i], order[j+1]] - distance[order[j], order[j+1]]
                if delta < best:
                    best, pair = delta, (i, j)
        if pair is None:
            break
        i, j = pair
        order[i:j+1] = reversed(order[i:j+1])
    return [i-1 for i in order[1:]]


class SearchMap:
    def __init__(self):
        axis = np.arange(-1800.0, 1800.1, 225.0)
        self.grid = np.array([[x, y] for x in axis for y in axis if x*x+y*y <= 1800.0**2])
        angles = np.arange(16) * 2*np.pi/16
        self.normals = np.column_stack((np.cos(angles), np.sin(angles)))
        self.unseen = np.ones((len(self.grid), len(angles)), dtype=bool)
        self.points = []
        self.certified = False
        # A Delaunay mesh is a sufficient proof, not a monotone test: inserting
        # another measured point can change its diagonals and make the NEW mesh
        # fail the diameter check. Preserve earlier valid supports explicitly.
        # Future points may enter a proof, but absence is certified only when
        # every point of one complete proof has actually been measured.
        self._proofs = []

    @staticmethod
    def _keys(points):
        return frozenset((float(q[0]) or 0., float(q[1]) or 0.) for q in points)

    def certify_plan(self, points, remember=False):
        """Conservative continuous proof, monotone for remembered supports.

        ``remember`` stores an independently successful proof for a planned or
        observed point set. It never marks those planned points as observed.
        A later station deletion is allowed only if the remaining plan contains
        an existing complete support or establishes a new complete proof.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        keys = self._keys(points)
        if any(proof['support'] <= keys for proof in self._proofs):
            return True
        proof = certificate_details(points)
        if not proof['certified']:
            return False
        if remember:
            support_points = np.asarray(proof.get('proof_points', points), dtype=float)
            support = self._keys(support_points)
            if not support <= keys:
                raise ArithmeticError('Coverage proof uses a point outside its supplied plan')
            self._proofs = [old for old in self._proofs if not support <= old['support']]
            self._proofs.append(dict(support=support, points=support_points.copy(),
                                    triangles=proof['triangles']))
        return True

    def mask(self, q):
        delta = np.asarray(q)-self.grid
        return (np.linalg.norm(delta, axis=1)[:, None] <= 1000.0) & (delta @ self.normals.T >= 0)

    def gain(self, q):
        return float(np.count_nonzero(self.unseen & self.mask(q))/self.unseen.size)

    def mark(self, q):
        if any(np.linalg.norm(np.asarray(q)-p) < 1e-6 for p in self.points):
            return
        self.points.append(np.asarray(q).copy())
        self.unseen &= ~self.mask(q)
        # Bits only guide the scan choice. They can never certify absence.
        self.certified = self.certified or self.certify_plan(self.points, remember=True)


class Planner:
    def __init__(self, command, params=None, record=False):
        self.command = command  # Only public command transport, not a simulator.
        self.params = params or Parameters()
        self.position = np.zeros(2)
        self.channel = 1
        self.time_s = 0.0
        self.targets = {c: Target(c) for c in range(1, 21)}
        self.coverage = SearchMap()
        if self.params.strategy == "zigzag":
            points = zigzag(self.params.inner, self.params.outer, 24)
        else:
            points = np.vstack(([0.0, 0.0], ring(self.params.inner, 12), ring(self.params.outer, 12, -np.pi/12)))
        self.stations = [np.array(q) for q in points[1:]]
        if not self.coverage.certify_plan(points, remember=True):
            raise ValueError("Base route must have a continuous coverage certificate")
        self.record = record
        self.frames = []
        self.commands = []
        self.costs = dict(movement=0., measure=0., switch=0., clear_success=0., clear_failure=0.)
        self.route_hint = []
        self.counters = dict(measures=0, clears=0, clear_failures=0, no_signal=0, known_no_signal=0,
                             search_stops=0, side_search_stops=0, side_known_measures=0,
                             pruned_stations=0, replans=0, optical_fallback_clears=0,
                             coverage_fallbacks=0)
        self.target_times = {c: dict(found_s=None, cleared_s=None, service_time_s=0.,
                                     service_move_m=0., measures=0, clears=0, failed_clears=0) for c in range(1, 21)}
        self.last_prune_count = -1

    def active(self):
        return [t for t in self.targets.values() if t.status == "active"]

    def unknown(self):
        return [t for t in self.targets.values() if t.status == "unknown"]

    def search_needed(self):
        # Every discovered source occupies a distinct channel. Sixteen public
        # discoveries exhaust the stated count upper bound, even before clear.
        discovered = sum(t.status in ("active", "cleared") for t in self.targets.values())
        return bool(self.unknown()) and not self.coverage.certified and discovered < 16

    def finished(self):
        return sum(t.status == "cleared" for t in self.targets.values()) >= 16 or (
            not self.active() and (not self.unknown() or self.coverage.certified))

    def snapshots(self):
        result = []
        maximum_found = sum(t.status in ("active", "cleared") for t in self.targets.values()) >= 16
        for t in self.targets.values():
            row = t.snapshot()
            if t.status == "unknown":
                row.update(polygon=None, center=None, radius=None,
                           status="absent" if self.coverage.certified or maximum_found else "unknown",
                           absence_reason="source_count_upper_bound" if maximum_found else
                           "continuous_coverage" if self.coverage.certified else None)
            result.append(row)
        return result

    def execute(self, action, phase, decision=None):
        if len(self.commands) >= self.params.max_commands:
            raise RuntimeError("Actual command budget exceeded")
        kind, c = action["kind"], int(action["channel"])
        q = np.asarray(action["position"], dtype=float)
        before, old_time = self.position.copy(), self.time_s
        active_before = self.targets[c].status == "active"
        response = self.command(kind, q, c)
        movement = float(np.linalg.norm(q-before))/5
        switching = float(kind == "measure" and c != self.channel)
        if kind == "measure":
            self.channel = c
            fee = 5.
            self.costs["measure"] += fee
            self.counters["measures"] += 1
            if response["measure_result"] == "no_signal":
                self.counters["no_signal"] += 1
                self.counters["known_no_signal"] += int(active_before)
        else:
            success = response["clear_result"] == "success"
            fee = 5. if success else 3.
            self.costs["clear_success" if success else "clear_failure"] += fee
            self.counters["clears"] += 1
            self.counters["clear_failures"] += int(not success)
            self.counters["optical_fallback_clears"] += int(action.get("reason") == "optical_finite_cover")
        self.costs["movement"] += movement
        self.costs["switch"] += switching
        self.position = q
        self.time_s = float(response["virtual_time_s"])
        if abs(self.time_s-old_time-movement-switching-fee) > 1e-6:
            raise ArithmeticError("Public response and planner cost ledger disagree")
        self.targets[c].observe(q, response)
        times = self.target_times[c]
        if response.get("measure_result") in ("direction", "near") and times["found_s"] is None:
            times["found_s"] = self.time_s
        if response.get("clear_result") == "success":
            times["cleared_s"] = self.time_s
        if kind == "measure": times["measures"] += 1
        else:
            times["clears"] += 1
            times["failed_clears"] += int(response["clear_result"] != "success")
        if phase in ("service", "side_known"):
            times["service_time_s"] += self.time_s-old_time
            times["service_move_m"] += movement*5
        row = dict(index=len(self.commands), time_s=self.time_s, from_position=before.tolist(),
            position=q.tolist(), channel=c, kind=kind, phase=phase, reason=action.get("reason", phase),
            response=response, move_s=movement, action_s=fee+switching,
            costs=dict(self.costs), coverage_certified=self.coverage.certified,
            coverage_state_mass=float(self.coverage.unseen.mean()))
        row["from"] = row.pop("from_position")
        if decision is not None:
            row["decision"] = decision
        self.commands.append(row)
        if self.record:
            self.frames.append(dict(row, targets=self.snapshots(), route_hint=list(self.route_hint)))

    def scan_unknowns(self, force=False, phase="side_search"):
        unknown = self.unknown()
        if not self.search_needed():
            return False
        q = self.position.copy()
        if any(np.linalg.norm(q-p) < 1e-6 for p in self.coverage.points):
            return False
        gain = self.coverage.gain(q)
        if not force and (not self.params.side_scans or gain < self.params.side_gain):
            return False
        full_scan = True
        for target in unknown:
            if not self.search_needed():
                full_scan = False
                break
            # All still-unknown channels see the same stop. A newly found source
            # leaves this set; its history is retained by Target.
            self.execute(dict(kind="measure", channel=target.channel, position=q.tolist(),
                              reason="mandatory_search" if force else "directional_gain_side_scan"), phase)
        if full_scan:
            self.coverage.mark(q)
        self.counters["search_stops"] += 1
        self.counters["side_search_stops"] += int(phase == "side_search")
        if self.record and self.frames:
            self.frames[-1].update(coverage_certified=self.coverage.certified,
                                   coverage_state_mass=float(self.coverage.unseen.mean()), targets=self.snapshots())
        return True

    def side_known(self, exclude=None):
        if not self.params.known_side_scans:
            return
        rows = []
        for target in self.active():
            if target.channel == exclude or target.radius <= 19.8:
                continue
            positives = [o for o in target.observations if o["result"] in ("direction", "near")]
            if not positives or any(np.linalg.norm(self.position-np.array(o["position"])) < 120. for o in target.observations):
                continue
            center = target.center
            a, b = np.array(positives[-1]["position"])-center, self.position-center
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1 or nb < 1 or nb > 1450:
                continue
            cross = abs(a[0]*b[1]-a[1]*b[0])/(na*nb)
            if cross < .25:
                continue
            probability = target.hit_probability(self.position)
            if probability < .45:
                continue
            # Geometry-based optional scan screening, not an exact time VOI.
            rows.append((probability*cross*min(target.radius, 500), target.channel))
        for _, channel in sorted(rows, reverse=True)[:2]:
            self.execute(dict(kind="measure", channel=channel, position=self.position.tolist(),
                              reason="known_geometry_side_scan"), "side_known")
            self.counters["side_known_measures"] += 1

    def prune(self):
        if not self.search_needed():
            self.stations = []
            return
        self.stations = [q for q in self.stations if not any(np.linalg.norm(q-p) < 1e-6 for p in self.coverage.points)]
        if self.params.strategy != "joint" or not self.params.prune_coverage:
            return
        if len(self.coverage.points) == self.last_prune_count:
            return
        self.last_prune_count = len(self.coverage.points)
        # Every deletion keeps an independently verifiable future search plan.
        for q in sorted(self.stations, key=lambda p: -float(np.linalg.norm(p-self.position))):
            kept = [p for p in self.stations if np.linalg.norm(p-q) > 1e-6]
            if self.coverage.certify_plan([*self.coverage.points, *kept], remember=True):
                self.stations = kept
                self.counters["pruned_stations"] += 1

    def restore_search_plan(self):
        """Finite geometry-only recovery; planned measurements stay unobserved."""
        self.stations = [q for q in scaffold_points() if not any(
            np.linalg.norm(q-p) < 1e-6 for p in self.coverage.points)]
        if not self.stations:
            # The same complete scaffold was observed; try remembered supports
            # once before reporting a genuine invariant violation.
            self.coverage.certified = self.coverage.certify_plan(self.coverage.points, remember=True)
            if not self.coverage.certified:
                raise RuntimeError("Coverage fallback exhausted without a certificate")
            return
        if not self.coverage.certify_plan([*self.coverage.points, *self.stations], remember=True):
            raise ArithmeticError("Finite coverage recovery must have a continuous proof")
        self.counters["coverage_fallbacks"] += 1

    def choose(self):
        self.prune()
        active = self.active()
        if not active and not self.stations and self.search_needed():
            # Recover before the joint task list is sorted. The old branch at
            # the bottom was unreachable when joint work was already empty.
            self.restore_search_plan()
        station = self.stations[0] if self.stations else None
        hint = station
        chosen = None
        if self.params.strategy == "joint":
            work = [("service", t.channel, t.center) for t in active]
            work += [("search", None, q) for q in self.stations]
            ordered = [work[j] for j in open_order(self.position, [w[2] for w in work])]
            self.route_hint = [w[2].tolist() for w in ordered]
            if not ordered:
                raise RuntimeError("Unfinished mission without work")
            kind, c, q = ordered[0]
            hint = ordered[1][2] if len(ordered) > 1 else None
            if kind == "search": station = q
            else: chosen = self.targets[c]
        else:
            self.route_hint = [q.tolist() for q in self.stations]
            if active:
                if station is None:
                    chosen = min(active, key=lambda t: np.linalg.norm(t.center-self.position))
                else:
                    def detour(t):
                        return np.linalg.norm(t.center-self.position)+np.linalg.norm(t.center-station)-np.linalg.norm(station-self.position)
                    t = min(active, key=detour)
                    if detour(t) <= self.params.service_detour:
                        chosen = t
        if chosen is not None:
            candidates = service_candidates(chosen, self.position, asdict(self.params), next_hint=hint)
            if not candidates:
                raise RuntimeError(f"Source {chosen.channel}: no finite service action")
            for a in candidates:
                onward = np.linalg.norm(np.array(a["position"])-hint)/5 if hint is not None else 0.
                a["selection_s"] = a["score_s"] + int(a["kind"] == "measure" and chosen.channel != self.channel) + self.params.service_tail_weight*onward
            action = min(candidates, key=lambda a: a["selection_s"])
            return action, "service", dict(candidates=candidates, selected=action, score_model="service_heuristic_plus_onward_distance")
        if station is None:
            # Finite fallback only if a conservative dynamic pruning test was
            # inconclusive; never infer completion from an empty numeric belief.
            self.restore_search_plan()
            if not self.stations:
                raise RuntimeError("Action requested after coverage completion")
            station = min(self.stations, key=lambda q: np.linalg.norm(q-self.position))
        channel = self.unknown()[0].channel
        action = dict(kind="measure", position=station.tolist(), channel=channel, reason="search_route_stop")
        return action, "search", dict(candidates=[action], selected=action, score_model="open_search_and_source_task_order")

    def run(self):
        started = time.perf_counter()
        for c in range(1, 21):
            self.execute(dict(kind="measure", position=[0., 0.], channel=c, reason="origin_all_channels"), "origin")
        self.coverage.mark([0., 0.])
        self.counters["search_stops"] += 1
        while not self.finished():
            self.counters["replans"] += 1
            action, phase, decision = self.choose()
            self.execute(action, phase, decision)
            if phase == "search":
                # The first channel was just observed by the moving command.
                # Avoid paying for it twice if it remained unknown.
                q = self.position.copy()
                full_scan = True
                for target in self.unknown():
                    if not self.search_needed():
                        full_scan = False
                        break
                    if target.channel == action["channel"]:
                        continue
                    self.execute(dict(kind="measure", position=q.tolist(), channel=target.channel, reason="mandatory_search"), "search")
                if full_scan:
                    self.coverage.mark(q)
                self.counters["search_stops"] += 1
                self.stations = [p for p in self.stations if np.linalg.norm(p-q) > 1e-6]
            else:
                self.scan_unknowns()
            self.side_known(exclude=action["channel"] if phase == "service" else None)
            if self.record and self.frames:
                self.frames[-1].update(coverage_certified=self.coverage.certified, targets=self.snapshots())
        return dict(time_s=self.time_s, costs=self.costs, counters=self.counters,
                    completed=True, cleared=sum(t.status == "cleared" for t in self.targets.values()),
                    source_times=self.target_times, coverage_certified=self.coverage.certified,
                    planning_wall_s=time.perf_counter()-started, command_count=len(self.commands),
                    parameters=asdict(self.params))

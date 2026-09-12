"""Per-channel directional unknown regions and adaptive geometric completion.

The finite position/normal grid is a planning quadrature, never an absence
certificate. A negative result removes only hypotheses guaranteed to hear at
R >= 1000 m; the possible source position behind that stop is retained.
Continuous absence uses an actually measured short-edge triangulation covering
the whole source disk. Future proof supports are kept separate from history.
"""
from __future__ import annotations

from collections import OrderedDict
import math

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, QhullError

from question4.zigzag_study.study import certificate_details, triangle_min_radius
from question4.zigzag_study.routes import zigzag

DOMAIN = 1800.0
RECEIVE_MIN = 1000.0


def _key(q):
    return (float(q[0]) or 0.0, float(q[1]) or 0.0)


def _array(points):
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _order(current, points):
    """Short open order, used only to price/return a future completion plan."""
    if not len(points):
        return []
    p = np.vstack((current, _array(points)))
    d = np.linalg.norm(p[:, None] - p[None, :], axis=2)
    left, order = list(range(1, len(p))), [0]
    while left:
        j = min(left, key=lambda k: (d[order[-1], k], k))
        order.append(j)
        left.remove(j)
    for _ in range(3):
        best, pair = -1e-6, None
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                delta = d[order[i - 1], order[j]] - d[order[i - 1], order[i]]
                if j + 1 < len(order):
                    delta += d[order[i], order[j + 1]] - d[order[j], order[j + 1]]
                if delta < best:
                    best, pair = delta, (i, j)
        if pair is None:
            break
        i, j = pair
        order[i:j + 1] = reversed(order[i:j + 1])
    return p[order[1:]].tolist()


class DirectionalCoverage:
    """Unknown position/normal states for channels numbered 1..channels.

    ``gain`` is removed fraction of the INITIAL position x normal mass. It is
    not a calibrated source-existence probability or fraction of remaining
    mass. Thus gains from different histories have a common denominator.
    """

    def __init__(self, channels=20, spacing=225.0, directions=24):
        self.channels = tuple(range(1, int(channels) + 1))
        self.spacing = float(spacing)
        axis = np.arange(-DOMAIN, DOMAIN + 1e-7, self.spacing)
        self.grid = np.array([(x, y) for x in axis for y in axis
                              if x*x + y*y <= DOMAIN**2 + 1e-7])
        angles = np.arange(int(directions)) * 2*np.pi/int(directions)
        self.normals = np.column_stack((np.cos(angles), np.sin(angles)))
        self.unseen = {c: np.ones((len(self.grid), len(angles)), dtype=bool)
                       for c in self.channels}
        self._points = {c: [] for c in self.channels}
        self._keys = {c: set() for c in self.channels}
        self._positive = {c: False for c in self.channels}
        self._certified = {c: False for c in self.channels}
        self._proofs = []
        self._mask_cache = OrderedDict()
        self._revision = 0
        self._plan_revision = None
        self._plan = []
        self._mesh_common_count = -1
        self.last_plan_diagnostics = dict(kind="not_built", fallback=False)

    def mask(self, position):
        q = np.asarray(position, dtype=float)
        key = _key(q)
        if key in self._mask_cache:
            return self._mask_cache[key]
        delta = q - self.grid
        mask = ((np.einsum("ij,ij->i", delta, delta)[:, None] <= RECEIVE_MIN**2)
                & (delta @ self.normals.T >= 0.0))
        self._mask_cache[key] = mask
        if len(self._mask_cache) > 256:
            self._mask_cache.popitem(last=False)
        return mask

    def observe(self, channel, position, result):
        """Apply one actual measurement; clear feedback is not accepted."""
        channel = int(channel)
        if isinstance(result, dict):
            result = result.get("measure_result")
        if result not in ("no_signal", "direction", "near"):
            raise ValueError("Coverage accepts actual measure feedback only")
        if result != "no_signal":
            self._positive[channel] = True
            return
        q = np.asarray(position, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Expected a finite planar measurement position")
        key = _key(q)
        if key in self._keys[channel]:
            return
        self._points[channel].append(q.copy())
        self._keys[channel].add(key)
        self.unseen[channel] &= ~self.mask(q)
        self._revision += 1
        # Planned points never enter this call or these keys.
        if not self._certified[channel]:
            self._certified[channel] = self._certify(self._points[channel], remember=True)

    def points(self, channel):
        """Copies of actual negative measurement positions for this channel."""
        return [p.copy() for p in self._points[int(channel)]]

    def residual_mass(self, channel):
        return float(self.unseen[int(channel)].mean())

    def gain(self, channel, position):
        c = int(channel)
        if self._certified[c] or self._positive[c]:
            return 0.0
        return float(np.count_nonzero(self.unseen[c] & self.mask(position)) / self.unseen[c].size)

    def gains(self, channels, positions):
        """Shape (number of channels, number of positions)."""
        return self.gain_after(channels, positions, None)

    def gain_after(self, channels, positions, extra_position, measured_channels=None):
        """Counterfactual negative-result gains; does not mutate actual state.

        This is a conditional no-detection continuation, not an expectation
        over positive detections. The caller must keep that distinction.
        """
        channels = tuple(int(c) for c in channels)
        p = _array(positions)
        if not channels or not len(p):
            return np.zeros((len(channels), len(p)))
        masks = np.asarray([self.mask(q) for q in p])
        update = set(channels if measured_channels is None else measured_channels)
        extra = None if extra_position is None else self.mask(extra_position)
        output = np.zeros((len(channels), len(p)))
        # Identical histories are common: avoid repeating the large boolean sum.
        cache = {}
        for i, c in enumerate(channels):
            if self._certified[c] or self._positive[c]:
                continue
            key = (frozenset(self._keys[c]), c in update and extra is not None)
            if key not in cache:
                states = self.unseen[c]
                if key[1]:
                    states = states & ~extra
                cache[key] = np.count_nonzero(masks & states[None], axis=(1, 2)) / states.size
            output[i] = cache[key]
        return output

    def channel_certified(self, channel):
        return bool(self._certified[int(channel)])

    def all_certified(self, channels):
        return all(self.channel_certified(c) for c in channels)

    def required_channels(self, position, channels):
        key = _key(position)
        return [int(c) for c in channels
                if not self._certified[int(c)] and key not in self._keys[int(c)]]

    def _certify(self, points, remember=False):
        p = _array(points)
        keys = frozenset(_key(q) for q in p)
        if any(support <= keys for support in self._proofs):
            return True
        details = certificate_details(p)
        if not details["certified"]:
            return False
        if remember:
            support = frozenset(_key(q) for q in details.get("proof_points", p))
            if not support <= keys:
                raise ArithmeticError("A geometric proof contains an unprovided point")
            self._proofs = [old for old in self._proofs if not support <= old]
            self._proofs.append(support)
        return True

    def plan_certified(self, channels, future_points, remember=True):
        """Proof for actual history + future points, without certifying absence."""
        future = _array(future_points)
        checked = set()
        for c in channels:
            c = int(c)
            if self._certified[c]:
                continue
            signature = frozenset(self._keys[c])
            if signature in checked:
                continue
            checked.add(signature)
            p = np.vstack((_array(self._points[c]), future))
            if not self._certify(p, remember=remember):
                return False
        return True

    def forecast_plan(self, channels, plan, position, measured_channels):
        """Remove only continuously redundant stops after a hypothetical scan.

        For each still-unknown channel, the support is its actual negative
        history, the candidate point IF that channel will be measured there,
        and the retained future common scan stops. This is the all-negative
        continuation; a positive result instead removes a discovered channel
        from the unknown-search problem.

        At most the four nearest stops are tried. Actual history, grid state,
        certificate flags, and remembered proof supports are not modified.
        The finite grid is deliberately not consulted, even when it is empty.
        """
        channels = tuple(int(c) for c in channels
                         if not self._certified[int(c)] and not self._positive[int(c)])
        kept = _array(plan).copy()
        if not channels:
            return []
        if not len(kept):
            return []
        q = np.asarray(position, dtype=float)
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("Expected a finite candidate position")
        measured = set(int(c) for c in measured_channels)
        qkey = _key(q)
        if not any(c in measured and qkey not in self._keys[c] for c in channels):
            return kept.tolist()
        # Group equal counterfactual histories; twenty identical channels need
        # only one geometric check per attempted deletion.
        histories = {}
        for c in channels:
            keys = self._keys[c] | ({qkey} if c in measured else set())
            signature = frozenset(keys)
            if signature not in histories:
                histories[signature] = _array(sorted(keys))
        candidates = sorted(kept, key=lambda p: float(np.linalg.norm(p-q)))[:4]
        for remove in candidates:
            trial = _array([p for p in kept if _key(p) != _key(remove)])
            if all(self._certify(np.vstack((history, trial)), remember=False)
                   for history in histories.values()):
                kept = trial
        return kept.tolist()

    def candidate_points(self, current, channels, limit=18):
        """Free positions from residual direction mass, with no route template.

        Each normal's weighted source centroid yields two points in its still
        unobserved emitting half-plane. Spatially separated residual hotspots
        contribute local alternatives. Positions move when real history changes.
        """
        channels = [int(c) for c in channels if not self._certified[int(c)]
                    and not self._positive[int(c)]]
        if not channels:
            return []
        current = np.asarray(current, dtype=float)
        weights = np.mean([self.unseen[c] for c in channels], axis=0)
        mass = weights.sum(axis=0)
        raw = []
        for k, normal in enumerate(self.normals):
            if mass[k] <= 0:
                continue
            center = np.sum(self.grid * weights[:, k, None], axis=0) / mass[k]
            for distance in (700.0, 950.0):
                raw.append((center + distance * normal, "remaining_direction_centroid"))
        spatial = weights.sum(axis=1)
        picked = []
        for j in np.argsort(-spatial, kind="stable"):
            if spatial[j] <= 0 or len(picked) >= 6:
                break
            g = self.grid[j]
            if any(np.linalg.norm(g-self.grid[k]) < 600 for k in picked):
                continue
            picked.append(j)
            angle_order = np.argsort(-weights[j], kind="stable")
            for k in angle_order[:2]:
                raw.append((g + 820*self.normals[k], "remaining_direction_hotspot"))
        if not raw:
            # A zero quadrature mass still requires geometric completion.
            return []
        p = np.array([q for q, _ in raw])
        gains = self.gains(channels, p).sum(axis=0)
        travel = np.linalg.norm(p-current, axis=1)
        score = gains / (60.0 + travel/5.0 + len(channels)*5.0)
        chosen, result = [], []
        observed = np.array([q for c in channels for q in self._points[c]])
        for j in np.argsort(-score, kind="stable"):
            if gains[j] <= 1e-10:
                continue
            if any(np.linalg.norm(p[j]-p[k]) < 175 for k in chosen):
                continue
            chosen.append(int(j))
            nearest = float(np.linalg.norm(observed-p[j], axis=1).min()) if len(observed) else None
            result.append(dict(position=p[j].tolist(), reason=raw[j][1],
                               gain=float(gains[j]), gain_per_channel=float(gains[j]/len(channels)),
                               travel_m=float(travel[j]), nearest_scan_m=nearest,
                               proposal_score=float(score[j])))
            if len(result) >= limit:
                break
        return result

    @staticmethod
    def _hull_covers(points):
        if len(points) < 3:
            return False
        try:
            hull = ConvexHull(_array(points))
            return bool(np.min(-hull.equations[:, 2]) >= DOMAIN + 1.0)
        except QhullError:
            return False

    def _mesh_completion(self, actual, current):
        """Create a short-edge mesh from actual points, refining geometric holes."""
        actual = _array(actual)
        p = actual.copy()
        additions = []
        if not self._hull_covers(p):
            # This circumscribed boundary is a geometric support pool, not a
            # prescribed tour. Real boundary observations replace its vertices.
            far = current if np.linalg.norm(current) > 100 else (
                actual[np.argmax(np.linalg.norm(actual, axis=1))] if len(actual) else np.zeros(2))
            phase = math.atan2(float(far[1]), float(far[0]))
            radius = (DOMAIN + 2.0) / math.cos(math.pi/12)
            angles = phase + np.arange(12)*2*np.pi/12
            boundary = np.column_stack((np.cos(angles), np.sin(angles))) * radius
            extra = [q for q in boundary if not len(p) or np.linalg.norm(p-q, axis=1).min() > 1e-7]
            # Delete supports already made unnecessary by actual outer stops.
            for q in list(extra):
                kept = [r for r in extra if _key(r) != _key(q)]
                if self._hull_covers(np.vstack((p, _array(kept)))):
                    extra = kept
            additions.extend(extra)
            p = np.vstack((p, _array(extra)))
        for _ in range(7):
            if self._certify(p, remember=True):
                return additions, False
            try:
                triangles = p[Delaunay(p).simplices]
            except QhullError:
                break
            triangles = triangles[triangle_min_radius(triangles) <= DOMAIN + 1e-7]
            edges = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
            bad = np.flatnonzero(edges.max(axis=1) >= RECEIVE_MIN - 1e-6)
            new = []
            for j in bad:
                tri = triangles[j]
                u = tri[1:] - tri[0]
                try:
                    q = tri[0] + np.linalg.solve(2*u, np.sum(u*u, axis=1))
                except np.linalg.LinAlgError:
                    q = tri.mean(axis=0)
                # Obtuse or nearly degenerate triangles use their long edge.
                try:
                    bary = np.linalg.solve(u.T, q-tri[0])
                    inside = bool(np.min(bary) >= -1e-7 and bary.sum() <= 1+1e-7)
                except np.linalg.LinAlgError:
                    inside = False
                if not inside or np.linalg.norm(q) > 2600:
                    k = int(np.argmax(edges[j]))
                    q = (tri[k] + tri[(k+1) % 3])/2
                if np.linalg.norm(p-q, axis=1).min() < 1e-5:
                    continue
                if any(np.linalg.norm(r-q) < 80 for r in new):
                    continue
                new.append(q)
            if not new or len(p) + len(new) > 120:
                break
            additions.extend(new)
            p = np.vstack((p, new))
        # Finite certified fallback. It is explicitly identified in diagnostics.
        fallback = zigzag(990.0, 1900.0, 24)
        if not self._certify(fallback, remember=True):
            raise ArithmeticError("Known finite coverage construction did not certify")
        additions = [q for q in fallback if not len(actual) or np.linalg.norm(actual-q, axis=1).min() > 1e-7]
        return additions, True

    def _prune(self, channels, current, plan):
        plan = [np.asarray(q) for q in plan
                if self.required_channels(q, channels)]
        for q in sorted(plan, key=lambda q: -float(np.linalg.norm(q-current))):
            kept = [r for r in plan if _key(r) != _key(q)]
            if self.plan_certified(channels, kept, remember=True):
                plan = kept
        return plan

    def completion_plan(self, channels, current):
        """A finite continuously certified FUTURE common scan plan.

        Fresh supports come from the actual geometry. Between complete rebuilds
        the proven plan is pruned as actual opportunistic scans replace points;
        it is not restarted at every source service or channel discovery.
        """
        channels = tuple(sorted(int(c) for c in channels if not self._certified[int(c)]
                                and not self._positive[int(c)]))
        if not channels:
            self.last_plan_diagnostics = dict(kind="already_certified", fallback=False, stops=0)
            return []
        current = np.asarray(current, dtype=float)
        signature = (channels, tuple(frozenset(self._keys[c]) for c in channels))
        if signature == self._plan_revision:
            return _order(current, self._plan)
        previous = self._plan
        used_warm = bool(previous) and self.plan_certified(channels, previous, remember=True)
        fallback = False
        common = set.intersection(*(self._keys[c] for c in channels))
        actual = _array(sorted(common))
        regenerated = False
        if used_warm:
            plan = self._prune(channels, current, previous)
            kind = "actual_stop_replacement"
            fallback = bool(self.last_plan_diagnostics.get("fallback", False))
            # New actual geometry can also change future coordinates, rather
            # than merely deleting vertices from the first plan. Rebuild at a
            # bounded cadence and retain the shorter certified continuation.
            if len(common) >= self._mesh_common_count + 3:
                regenerated = True
                self._mesh_common_count = len(common)
                fresh, fresh_fallback = self._mesh_completion(actual, current)
                if self.plan_certified(channels, fresh, remember=True):
                    fresh = self._prune(channels, current, fresh)
                    def cost(stops):
                        route = np.vstack((current, _array(_order(current, stops))))
                        # All-negative common scan continuation: movement plus
                        # 5 seconds per unresolved channel, in distance units.
                        return float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum()
                                     + len(stops)*len(channels)*25.0)
                    if cost(fresh) + 1e-6 < cost(plan):
                        plan, fallback = fresh, fresh_fallback
                        kind = "regenerated_adaptive_triangles"
        else:
            # A common observed subset is a conservative shared foundation.
            # Per-channel differences are still used in pruning and all gains.
            plan, fallback = self._mesh_completion(actual, current)
            if not self.plan_certified(channels, plan, remember=True):
                raise ArithmeticError("Adaptive completion is not a continuous proof")
            plan = self._prune(channels, current, plan)
            kind = "adaptive_triangle_completion"
            self._mesh_common_count = len(common)
            regenerated = True
        self._plan = [q.copy() for q in plan]
        self._plan_revision = signature
        self.last_plan_diagnostics = dict(kind=kind, fallback=fallback, stops=len(plan),
                                          mesh_regenerated=regenerated,
                                          common_actual_points=len(common),
                                          channels=len(channels), actual_points_min=min(len(self._points[c]) for c in channels),
                                          proof_kind="continuous_short_edge_triangles")
        return _order(current, plan)

    def snapshot(self, channel=None, include_cells=True):
        if channel is None:
            return dict(channels={str(c): self.snapshot(c, include_cells=include_cells)
                                  for c in self.channels},
                        last_plan=dict(self.last_plan_diagnostics),
                        planned_points=[q.tolist() for q in self._plan],
                        grid_is_certificate=False)
        c = int(channel)
        fraction = self.unseen[c].mean(axis=1)
        result = dict(channel=c, certified=self.channel_certified(c),
                      remaining_mass=self.residual_mass(c), uncovered_positions=int(np.count_nonzero(fraction)),
                      position_samples=len(self.grid), direction_samples=len(self.normals),
                      grid_spacing_m=self.spacing, negative_measurement_count=len(self._points[c]),
                      negative_points=[q.tolist() for q in self._points[c]],
                      grid_is_certificate=False)
        if self._certified[c]:
            supports = [s for s in self._proofs if s <= self._keys[c]]
            if not supports:
                raise ArithmeticError("Actual coverage certificate lost its supporting observations")
            support = min(supports, key=len)
            result["proof_points"] = [list(q) for q in sorted(support)]
        if include_cells:
            result["cells"] = [[float(q[0]), float(q[1]), float(f)]
                               for q, f in zip(self.grid, fraction) if f > 0]
        return result

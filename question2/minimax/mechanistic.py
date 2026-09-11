"""Deterministic set-membership geometry and second-monitor policies.

All geometry is expressed in metres and radians.  Circular constraints are
represented by circumscribed regular polygons, so reported feasible regions
are conservative outer approximations of the exact circular intersections.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import io
import math
import pickle
from pathlib import Path
import zipfile

import numpy as np


ARENA_RADIUS = 1800.0
RECEIVE_MIN = 1000.0
RECEIVE_MAX = 1500.0
ANGLE_ERROR = math.radians(1.0)
NEAR_RADIUS = 5.0


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def clip_halfplane(poly: np.ndarray, normal: np.ndarray, offset: float,
                   tol: float = 1e-10) -> np.ndarray:
    """Sutherland-Hodgman clipping by normal dot x <= offset."""
    if len(poly) == 0:
        return poly.copy()
    out: list[np.ndarray] = []
    for i, b in enumerate(poly):
        a = poly[i - 1]
        da = float(np.dot(a, normal) - offset)
        db = float(np.dot(b, normal) - offset)
        inside_a, inside_b = da <= tol, db <= tol
        if inside_a != inside_b:
            denominator = da - db
            if abs(denominator) > tol:
                out.append(a + da / denominator * (b - a))
        if inside_b:
            out.append(b.copy())
    return np.asarray(out, dtype=float).reshape(-1, 2)


def outer_disk_polygon(center: np.ndarray, radius: float,
                       sides: int = 128) -> np.ndarray:
    """Vertices of a regular polygon circumscribed about a disk."""
    angles = (np.arange(sides) + 0.5) * 2.0 * math.pi / sides
    scale = radius / math.cos(math.pi / sides)
    return np.asarray(center, float) + scale * np.column_stack(
        (np.cos(angles), np.sin(angles)))


def clip_disk(poly: np.ndarray, center: np.ndarray, radius: float,
              sides: int = 128) -> np.ndarray:
    center = np.asarray(center, float)
    for angle in np.arange(sides) * 2.0 * math.pi / sides:
        normal = np.array([math.cos(angle), math.sin(angle)])
        poly = clip_halfplane(poly, normal, radius + float(np.dot(normal, center)))
        if len(poly) == 0:
            break
    return poly


def clip_wedge(poly: np.ndarray, station: np.ndarray, bearing: float,
               alpha: float = ANGLE_ERROR) -> np.ndarray:
    station = np.asarray(station, float)
    for sign in (-1.0, 1.0):
        angle = bearing + sign * alpha
        normal = np.array([-sign * math.sin(angle), sign * math.cos(angle)])
        poly = clip_halfplane(poly, normal, float(np.dot(normal, station)))
        if len(poly) == 0:
            break
    return poly


def initial_region(station: np.ndarray, measured_bearing: float,
                   circle_sides: int = 128) -> np.ndarray:
    """First feasible target set: arena, 1500 m disk and +/-1 degree cone."""
    poly = outer_disk_polygon(np.zeros(2), ARENA_RADIUS, circle_sides)
    poly = clip_disk(poly, np.asarray(station, float), RECEIVE_MAX, circle_sides)
    return clip_wedge(poly, np.asarray(station, float), measured_bearing)


def second_region(first: np.ndarray, station: np.ndarray, measured_bearing: float,
                  circle_sides: int = 128, include_receive_disk: bool = True) -> np.ndarray:
    poly = first
    if include_receive_disk:
        poly = clip_disk(poly, np.asarray(station, float), RECEIVE_MAX, circle_sides)
    return clip_wedge(poly, np.asarray(station, float), measured_bearing)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return float(abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                     - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))) / 2.0)


def polygon_diameter(poly: np.ndarray) -> float:
    if len(poly) < 2:
        return 0.0
    delta = poly[:, None, :] - poly[None, :, :]
    return float(np.sqrt(np.max(np.sum(delta * delta, axis=-1))))


def polygon_min_enclosing_circle(poly: np.ndarray,
                                 tol: float = 1e-7) -> tuple[np.ndarray, float]:
    """Exact minimum enclosing circle of a convex polygon's vertices.

    A planar minimum enclosing circle is supported by either two or three
    points.  The feasible regions here have few vertices, so enumerating those
    supports is deterministic and keeps this validation metric independent of
    the collaborator's Numba implementation.
    """
    points = np.asarray(poly, dtype=float)
    if len(points) == 0:
        return np.array([math.nan, math.nan]), 0.0
    if len(points) == 1:
        return points[0].copy(), 0.0

    best_center = points[0].copy()
    best_r2 = math.inf

    def consider(center: np.ndarray, radius2: float) -> None:
        nonlocal best_center, best_r2
        if not np.isfinite(radius2) or radius2 >= best_r2:
            return
        required = float(np.max(np.sum((points - center) ** 2, axis=1)))
        allowance = tol * max(1.0, radius2)
        if required <= radius2 + allowance:
            best_center = center.copy()
            best_r2 = max(radius2, required)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            center = 0.5 * (points[i] + points[j])
            consider(center, float(np.sum((points[i] - center) ** 2)))

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            u = points[j] - points[i]
            for k in range(j + 1, len(points)):
                v = points[k] - points[i]
                cross = float(u[0] * v[1] - u[1] * v[0])
                if abs(cross) < 1e-12:
                    continue
                u2, v2 = float(np.dot(u, u)), float(np.dot(v, v))
                center = points[i] + np.array([
                    u2 * v[1] - v2 * u[1],
                    u[0] * v2 - v[0] * u2,
                ]) / (2.0 * cross)
                consider(center, float(np.sum((points[i] - center) ** 2)))

    if not np.isfinite(best_r2):
        # This can only arise from a fully coincident degenerate polygon.
        return points[0].copy(), 0.0
    return best_center, math.sqrt(best_r2)


def polygon_mec_radius(poly: np.ndarray) -> float:
    """Minimum-enclosing-circle radius, matching the new NN's metric."""
    return polygon_min_enclosing_circle(poly)[1]


def point_in_convex(poly: np.ndarray, point: np.ndarray, tol: float = 1e-8) -> bool:
    if len(poly) < 3:
        return False
    edges = np.roll(poly, -1, axis=0) - poly
    rel = np.asarray(point) - poly
    cross = edges[:, 0] * rel[:, 1] - edges[:, 1] * rel[:, 0]
    return bool(np.all(cross >= -tol) or np.all(cross <= tol))


def sample_boundary(poly: np.ndarray, subdivisions: int = 4) -> np.ndarray:
    if len(poly) == 0:
        return poly.copy()
    fractions = np.arange(subdivisions, dtype=float) / subdivisions
    pieces = []
    for i, a in enumerate(poly):
        b = poly[(i + 1) % len(poly)]
        pieces.append(a + fractions[:, None] * (b - a))
    return np.concatenate(pieces)


def universal_safe(station: np.ndarray, measured_bearing: float, q: np.ndarray,
                   alpha: float = ANGLE_ERROR, tol: float = 1e-7) -> bool:
    """Closed-form sufficient reception certificate for the full first cone.

    This conservatively permits target range zero.  It therefore remains safe
    when the true first target range is greater than the 5 m near threshold.
    """
    delta = np.asarray(q, float) - np.asarray(station, float)
    s = float(np.linalg.norm(delta))
    if s <= tol:
        return True
    beta = abs(float(wrap_angle(math.atan2(delta[1], delta[0]) - measured_bearing)))
    if beta + alpha >= math.pi / 2.0:
        return False
    return (s <= RECEIVE_MIN + tol
            and s <= 2.0 * RECEIVE_MIN * math.cos(beta + alpha) + tol)


def project_to_universal_safe(station: np.ndarray, measured_bearing: float,
                              q: np.ndarray) -> np.ndarray:
    """Project an arbitrary action toward a known-safe 750 m forward anchor."""
    station = np.asarray(station, float)
    q = np.asarray(q, float)
    if universal_safe(station, measured_bearing, q):
        return q.copy()
    anchor = station + 750.0 * np.array([math.cos(measured_bearing),
                                         math.sin(measured_bearing)])
    lo, hi = 0.0, 1.0
    for _ in range(55):
        mid = (lo + hi) / 2.0
        trial = anchor + mid * (q - anchor)
        if universal_safe(station, measured_bearing, trial):
            lo = mid
        else:
            hi = mid
    return anchor + lo * (q - anchor)


def sampled_reception_margin(first: np.ndarray, station: np.ndarray, q: np.ndarray,
                              subdivisions: int = 8) -> float:
    """Sampled check of the exact shared-radius reception condition."""
    points = sample_boundary(first, subdivisions)
    if len(points) == 0:
        return -math.inf
    r1 = np.linalg.norm(points - station, axis=1)
    allowance = np.maximum(RECEIVE_MIN, r1)
    return float(np.min(allowance - np.linalg.norm(points - q, axis=1)))


def possible_bearings(first: np.ndarray, q: np.ndarray,
                      step_deg: float = 1.0) -> np.ndarray:
    """Deterministic grid covering every bearing direction subtended by first."""
    boundary = sample_boundary(first, 3)
    if len(boundary) == 0:
        return np.empty(0)
    if point_in_convex(first, q):
        return np.arange(-math.pi, math.pi, math.radians(step_deg))
    raw = np.arctan2(boundary[:, 1] - q[1], boundary[:, 0] - q[0])
    reference = math.atan2(float(np.sin(raw).mean()), float(np.cos(raw).mean()))
    relative = wrap_angle(raw - reference)
    lo = float(np.min(relative) - ANGLE_ERROR)
    hi = float(np.max(relative) + ANGLE_ERROR)
    if hi - lo > math.pi:
        return np.arange(-math.pi, math.pi, math.radians(step_deg))
    count = max(2, int(math.ceil((hi - lo) / math.radians(step_deg))) + 1)
    grid = reference + np.linspace(lo, hi, count)
    critical = np.concatenate((raw - ANGLE_ERROR, raw, raw + ANGLE_ERROR))
    return np.unique(np.round(wrap_angle(np.concatenate((grid, critical))), 10))


@dataclass(frozen=True)
class RobustScore:
    worst_diameter_m: float
    worst_area_m2: float
    worst_bearing_rad: float


def robust_score(first: np.ndarray, q: np.ndarray, angle_step_deg: float = 1.0,
                 circle_sides: int = 128) -> RobustScore:
    """Worst second-measurement region over all compatible future bearings."""
    q = np.asarray(q, float)
    base = clip_disk(first, q, RECEIVE_MAX, circle_sides)
    worst_diameter = -1.0
    worst_area = -1.0
    worst_angle = math.nan
    for angle in possible_bearings(base, q, angle_step_deg):
        poly = clip_wedge(base, q, float(angle))
        if len(poly) == 0:
            continue
        diameter = polygon_diameter(poly)
        area = polygon_area(poly)
        if diameter > worst_diameter + 1e-9:
            worst_diameter, worst_angle = diameter, float(angle)
        worst_area = max(worst_area, area)
    if worst_diameter < 0:
        return RobustScore(0.0, 0.0, math.nan)
    return RobustScore(worst_diameter, worst_area, worst_angle)


def candidate_offsets(radial_step: float = 100.0, angle_step_deg: float = 6.0,
                      min_radius: float = 600.0) -> np.ndarray:
    values: list[tuple[float, float]] = []
    for radius in np.arange(min_radius, RECEIVE_MIN + 0.1, radial_step):
        for beta_deg in np.arange(-54.0, 54.1, angle_step_deg):
            if abs(beta_deg) < 6.0:
                continue
            beta = math.radians(float(beta_deg))
            if radius <= 2.0 * RECEIVE_MIN * math.cos(abs(beta) + ANGLE_ERROR):
                values.append((radius * math.cos(beta), radius * math.sin(beta)))
    # Closed-form far-range, first-order optimum: cos(beta)=12/13.
    values.extend([(12000.0 / 13.0, 5000.0 / 13.0),
                   (12000.0 / 13.0, -5000.0 / 13.0)])
    return np.unique(np.round(np.asarray(values), 8), axis=0)


@dataclass
class PolicyDecision:
    point: np.ndarray
    score: RobustScore
    candidate_points: np.ndarray
    candidate_scores: np.ndarray
    near_optimal_mask: np.ndarray


class MechanisticPolicy:
    """Grid-refined minimax-diameter policy over a certified reception set."""

    def __init__(self, circle_sides: int = 96, angle_step_deg: float = 1.5,
                 diameter_tolerance_m: float = 5.0):
        self.circle_sides = circle_sides
        self.angle_step_deg = angle_step_deg
        self.diameter_tolerance_m = diameter_tolerance_m
        self._coarse = candidate_offsets()

    @staticmethod
    def _world_offsets(local: np.ndarray, bearing: float) -> np.ndarray:
        c, s = math.cos(bearing), math.sin(bearing)
        rotation = np.array([[c, -s], [s, c]])
        return local @ rotation.T

    def _score_candidates(self, first: np.ndarray, station: np.ndarray,
                          bearing: float, local: np.ndarray) -> tuple[np.ndarray, list[RobustScore]]:
        world = station + self._world_offsets(local, bearing)
        keep = np.array([universal_safe(station, bearing, q) for q in world])
        world = world[keep]
        scores = [robust_score(first, q, self.angle_step_deg, self.circle_sides)
                  for q in world]
        return world, scores

    def decide(self, station: np.ndarray, measured_bearing: float) -> PolicyDecision:
        station = np.asarray(station, float)
        first = initial_region(station, measured_bearing, self.circle_sides)
        coarse_points, coarse_scores = self._score_candidates(
            first, station, measured_bearing, self._coarse)
        coarse_values = np.array([[s.worst_diameter_m, s.worst_area_m2]
                                  for s in coarse_scores])
        best_index = int(np.lexsort((coarse_values[:, 1], coarse_values[:, 0]))[0])
        best_delta = coarse_points[best_index] - station
        c, s = math.cos(measured_bearing), math.sin(measured_bearing)
        forward = c * best_delta[0] + s * best_delta[1]
        lateral = -s * best_delta[0] + c * best_delta[1]
        best_radius = math.hypot(forward, lateral)
        best_beta = math.atan2(lateral, forward)

        local_refined = []
        for radius in np.arange(max(450.0, best_radius - 100.0),
                                min(RECEIVE_MIN, best_radius + 100.0) + 0.1, 25.0):
            for beta in np.arange(best_beta - math.radians(5.0),
                                  best_beta + math.radians(5.0) + 1e-9,
                                  math.radians(1.0)):
                if radius <= 2.0 * RECEIVE_MIN * math.cos(abs(beta) + ANGLE_ERROR):
                    local_refined.append((radius * math.cos(beta), radius * math.sin(beta)))
        refined_points, refined_scores = self._score_candidates(
            first, station, measured_bearing, np.asarray(local_refined))
        points = np.vstack((coarse_points, refined_points))
        scores = coarse_scores + refined_scores
        values = np.array([[score.worst_diameter_m, score.worst_area_m2]
                           for score in scores])
        best_index = int(np.lexsort((values[:, 1], values[:, 0]))[0])
        best = scores[best_index]
        mask = values[:, 0] <= best.worst_diameter_m + self.diameter_tolerance_m
        return PolicyDecision(points[best_index], best, points, values, mask)

    def __call__(self, x: float, y: float, theta_deg: float) -> tuple[float, float]:
        decision = self.decide(np.array([x, y]), math.radians(theta_deg))
        return float(decision.point[0]), float(decision.point[1])


@dataclass(frozen=True)
class _Storage:
    key: str


@dataclass(frozen=True)
class _Tensor:
    storage: _Storage
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def _rebuild_tensor(storage, offset, size, stride, *unused):
    return _Tensor(storage, int(offset), tuple(size), tuple(stride))


class _TorchCheckpointUnpickler(pickle.Unpickler):
    def persistent_load(self, value):
        if value[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {value!r}")
        return _Storage(str(value[2]))

    def find_class(self, module, name):
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return object
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        return super().find_class(module, name)


def load_torch_state_dict_numpy(path: str | Path) -> dict[str, np.ndarray]:
    """Read this repository's simple float32 PyTorch state_dict without torch."""
    with zipfile.ZipFile(path) as archive:
        prefix = archive.namelist()[0].split("/")[0]
        payload = _TorchCheckpointUnpickler(
            io.BytesIO(archive.read(f"{prefix}/data.pkl"))).load()
        state = payload["state_dict"]
        arrays: dict[str, np.ndarray] = {}
        for name, tensor in state.items():
            raw = archive.read(f"{prefix}/data/{tensor.storage.key}")
            flat = np.frombuffer(raw, dtype="<f4")
            item_strides = tuple(value * flat.dtype.itemsize for value in tensor.stride)
            view = np.ndarray(tensor.size, dtype=flat.dtype,
                              buffer=flat.data, offset=tensor.offset * flat.dtype.itemsize,
                              strides=item_strides)
            arrays[name] = np.array(view, copy=True)
        return arrays


class NeuralReferencePolicy:
    """NumPy reproduction of the collaborator repository's original network."""

    SCALE = 1500.0
    ARENA_SCALED = 1800.0 / SCALE

    def __init__(self, checkpoint: str | Path):
        self.state = load_torch_state_dict_numpy(checkpoint)

    def local_action(self, local_monitor: np.ndarray) -> np.ndarray:
        x = np.asarray(local_monitor, np.float32) / self.ARENA_SCALED
        for index in (0, 2):
            x = np.tanh(self.state[f"layers.{index}.weight"] @ x
                        + self.state[f"layers.{index}.bias"])
        x = self.state["layers.4.weight"] @ x + self.state["layers.4.bias"]
        return 2.0 * np.tanh(x.astype(np.float32))

    def __call__(self, x: float, y: float, theta_deg: float) -> tuple[float, float]:
        theta = math.radians(theta_deg % 360.0)
        c, s = math.cos(theta), math.sin(theta)
        p = np.array([x, y], dtype=float) / self.SCALE
        local_p = np.array([c * p[0] + s * p[1], -s * p[0] + c * p[1]])
        q = self.local_action(local_p)
        delta = self.SCALE * np.array([c * q[0] - s * q[1],
                                       s * q[0] + c * q[1]])
        result = np.array([x, y]) + delta
        return float(result[0]), float(result[1])

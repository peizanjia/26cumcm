"""Exact-circle conditional sampling; probability laws are modeling choices."""
import numpy as np
from scipy.special import ndtr, ndtri
from .geometry import SCALE, ARENA, DELTA, NEAR, first_polygons


def noise(rng, shape, kind="uniform"):
    if kind == "uniform":
        return rng.uniform(-DELTA, DELTA, shape)
    if kind == "truncnorm":
        return ndtri(rng.uniform(ndtr(-3), ndtr(3), shape)) * DELTA / 3
    raise ValueError("noise must be uniform or truncnorm")


def disk(rng, n):
    angle = rng.uniform(-np.pi, np.pi, n)
    r = ARENA * np.sqrt(rng.random(n))
    return r[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))


def sample_states(n, seed, noise_kind="uniform", radius_mode="uniform"):
    """Independently propose monitor/source, THEN condition on direction.

    First monitor is drawn in the arena for this training distribution only;
    the second action is not restricted to the arena.
    """
    rng = np.random.default_rng(seed)
    result = []
    remaining = n
    while remaining:
        count = max(remaining * 4, 256)
        p, g = disk(rng, count), disk(rng, count)
        r = (rng.uniform(1000, 1500, count) if radius_mode == "uniform"
             else np.full(count, float(radius_mode))) / SCALE
        d = np.linalg.norm(g - p, axis=1)
        accepted = np.flatnonzero((d > NEAR) & (d <= r))[:remaining]
        pp, gg = p[accepted], g[accepted]
        theta = np.arctan2((gg - pp)[:, 1], (gg - pp)[:, 0])
        theta += noise(rng, len(accepted), noise_kind)
        c, s = np.cos(theta), np.sin(theta)
        local_p = np.column_stack((c * pp[:, 0] + s * pp[:, 1],
                                   -s * pp[:, 0] + c * pp[:, 1]))
        result.append(np.column_stack((pp, theta, local_p)))
        remaining -= len(accepted)
    return np.concatenate(result)


def sample_posterior(local_p, samples, seed, noise_kind="uniform", radius_mode="uniform"):
    """p(g | p,theta,direction), in monitor/bearing coordinates.

    Proposal: angle from the error density and r^2 uniform over (5,1500]^2.
    Reject outside the true arena and accept with P(R>=r). This includes
    the polar Jacobian and never reads a hidden source in the policy.
    """
    rng = np.random.default_rng(seed)
    n = len(local_p)
    out = np.empty((n, samples, 2))
    missing = np.ones((n, samples), dtype=bool)
    radius_max = 1. if radius_mode == "uniform" else float(radius_mode) / SCALE
    # A state-specific upper radius keeps sampling efficient near the arena
    # boundary. The largest ray exit occurs at the smallest p dot direction.
    toward_center = np.arctan2(-local_p[:, 1], -local_p[:, 0])
    dot_min = np.minimum(local_p[:, 0] * np.cos(DELTA) + local_p[:, 1] * np.sin(DELTA),
                         local_p[:, 0] * np.cos(DELTA) - local_p[:, 1] * np.sin(DELTA))
    dot_min = np.where(np.abs(toward_center) <= DELTA, -np.linalg.norm(local_p, axis=1), dot_min)
    exit_max = -dot_min + np.sqrt(np.maximum(0, dot_min**2 + ARENA**2 - (local_p**2).sum(1)))
    upper = np.minimum(radius_max, exit_max)
    if np.any(upper <= NEAR):
        raise ValueError("No valid first-direction posterior for this input")
    for _ in range(10000):
        rows, cols = np.where(missing)
        if len(rows) == 0:
            return out
        r = np.sqrt(NEAR**2 + rng.random(len(rows)) * (upper[rows]**2 - NEAR**2))
        angle = noise(rng, len(rows), noise_kind)
        g = r[:, None] * np.column_stack((np.cos(angle), np.sin(angle)))
        w = (np.clip((1 - r) / (500 / SCALE), 0, 1)
             if radius_mode == "uniform" else np.ones(len(rows)))
        accept = (np.linalg.norm(g + local_p[rows], axis=1) <= ARENA)
        accept &= rng.random(len(rows)) < w
        out[rows[accept], cols[accept]] = g[accept]
        missing[rows[accept], cols[accept]] = False
    raise RuntimeError("Posterior sampling stalled: impossible or extremely rare input state")


def make_dataset(n, samples, seed, sides=256, noise_kind="uniform", radius_mode="uniform"):
    states = sample_states(n, seed, noise_kind, radius_mode)
    targets = sample_posterior(states[:, 3:5], samples, seed + 1, noise_kind, radius_mode)
    polygons, counts, areas = first_polygons(states[:, 3:5].copy(), sides, 64)
    if counts.max() > 64 or counts.min() < 3:
        raise RuntimeError(f"Invalid polygon capacity/count: {counts.min()}..{counts.max()}")
    # Strip unused padded vertices globally; clipping adds at most two more.
    polygons = polygons[:, :counts.max()]
    errors = noise(np.random.default_rng(seed + 2), (n, samples), noise_kind)
    return dict(states=states, targets=targets, polygons=polygons, counts=counts,
                areas=areas, errors=errors)

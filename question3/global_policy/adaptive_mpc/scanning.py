"""Stationary, public-state-only selection of useful channel measurements.

Known-channel value is expressed in seconds, using paired sampled continuations
from the *same later entry point* before and after a hypothetical observation.
The journey already made to the present stop is not a benefit of scanning.
This is a small deterministic base-policy rollout, not the optimal value function.
Unknown-channel selection instead uses a clearly identified coverage policy;
coverage gain is not mislabelled as a measured expected time saving.
"""
import math

import numpy as np
from numba import njit

from ..joint_rollout.kernel import exclude_disk, mec, update
from ..joint_rollout.service import scenarios
from ..model import CLEAR_R, RECEIVE_MIN, RECEIVE_MAX, SPEED


@njit(cache=True)
def _position_error(q, seed):
    """One latent error per rounded position and source, shared by both branches."""
    # SplitMix64 supplies a reproducible uniform draw without evaluating a hidden
    # simulator. Reusing a position reuses its error, including across branches.
    a = np.uint64(np.int64(round(q[0] * 1e6)))
    b = np.uint64(np.int64(round(q[1] * 1e6)))
    z = seed ^ (a * np.uint64(0x9E3779B97F4A7C15)) ^ (b * np.uint64(0xBF58476D1CE4E5B9))
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return 2. * float(z >> np.uint64(11)) / 9007199254740992. - 1.


@njit(cache=True)
def _continuation(poly, entry, x, receive_radius, error_seed, history, sides):
    """Direct-MEC base policy, including the certified-clear branch at entry."""
    p = poly.copy()
    q = entry.copy()
    seen = np.empty((len(history) + 16, 2))
    for i in range(len(history)):
        seen[i] = history[i]
    used = len(history)
    cost = 0.
    for step in range(14):
        if not len(p):
            break
        center, radius = mec(p)
        if not math.isfinite(radius):
            break
        if radius <= CLEAR_R:
            delta = q - center
            distance = np.linalg.norm(delta)
            dest = center.copy()
            if distance > 0:
                dest += delta * min(1., max(0., CLEAR_R - radius - 1e-6) / distance)
            return cost + np.linalg.norm(dest - q) / SPEED + 5.
        dest = center.copy()
        # Old observations are actual observations, never fictitious markers to
        # force this policy to leave the present point. A new scan is appended
        # only to the AFTER branch that actually receives it.
        for attempt in range(24):
            repeated = False
            for j in range(used):
                if np.linalg.norm(dest - seen[j]) < 1e-7:
                    repeated = True
                    break
            if not repeated:
                break
            dest[0] += .05
        cost += np.linalg.norm(dest - q) / SPEED + 5.
        p = update(p, dest, x, receive_radius, _position_error(dest, error_seed), sides)
        seen[used] = dest
        used += 1
        q = dest
    return 1e6


@njit(cache=True)
def _paired_values(poly, entry, stop, points, radii, seeds, history, sides):
    n = len(points)
    before = np.empty(n)
    after = np.empty(n)
    posterior_radius = np.empty(n)
    after_history = np.empty((len(history) + 1, 2))
    for j in range(len(history)):
        after_history[j] = history[j]
    after_history[-1] = stop
    for i in range(n):
        x = points[i]
        p = update(poly.copy(), stop, x, radii[i], _position_error(stop, seeds[i]), sides)
        _, posterior_radius[i] = mec(p)
        before[i] = _continuation(poly, entry, x, radii[i], seeds[i], history, sides)
        after[i] = _continuation(p, entry, x, radii[i], seeds[i], after_history, sides)
    return before, after, posterior_radius


def _row(target, q):
    return dict(channel=int(target.channel), status=target.status, position=np.asarray(q).tolist(),
                selected=False, valid=False, reason='not_assessed',
                gross_saved_s=0., net_saved_s=0., expected_radius_m=None,
                probability_r20=0., detection_probability=0., orthogonality=0.,
                anisotropy=1., fisher_information_gain=0., scenario_se_s=0.,
                coverage_gain=0., remaining_fraction=0., relative_coverage_gain=0.)


def _known_geometry(target, q):
    points = target.particles
    weights = target.weights
    center = weights @ points
    delta = points - center
    covariance = (delta * weights[:, None]).T @ delta
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ray = center - q
    distance = max(float(np.linalg.norm(ray)), 1.)
    orthogonality = float(np.clip(1. - (eigenvectors[:, -1] @ (ray / distance)) ** 2, 0., 1.))
    gradient = np.array([-ray[1], ray[0]]) / distance ** 2
    noise_variance = math.radians(1.) ** 2 / 3.
    fisher = .5 * math.log1p(max(0., float(gradient @ covariance @ gradient)) / noise_variance)
    lo = np.full(len(points), RECEIVE_MIN)
    hi = np.full(len(points), RECEIVE_MAX)
    for obs in target.observations:
        d = np.linalg.norm(points - obs.position, axis=1)
        if obs.result == 'no_signal':
            hi = np.minimum(hi, d)
        else:
            lo = np.maximum(lo, d)
    distance_to_stop = np.linalg.norm(points - q, axis=1)
    probability = np.clip((hi - np.maximum(lo, distance_to_stop)) / np.maximum(hi - lo, 1e-12), 0., 1.)
    visibility = float(weights @ probability)
    # Coarse but useful for avoiding full rollouts on all 20 channels at each
    # stop. Final selection still uses seconds, never this information score.
    score = visibility * (fisher + .15 * math.log1p(target.radius / CLEAR_R))
    return dict(orthogonality=orthogonality,
                anisotropy=float(eigenvalues[-1] / max(eigenvalues[0], 1e-9)),
                fisher_information_gain=fisher, detection_probability=visibility,
                geometry_screen_score=float(score))


def known_scan_value(world, target, q, n=None):
    """Return the incremental value of a scan at q; never mutate world/target.

    entry is fixed at the existing posterior center in both branches. The value
    therefore captures localization/service savings, but omits possible future
    global route changes and shared observations with other channels. A scan
    outside the current channel reserves one switch in and one later switch
    back; assess_stop recomputes costs for its selected multichannel order.
    """
    q = np.asarray(q, dtype=float)
    row = _row(target, q)
    if target.status != 'active':
        row['reason'] = 'not_known_active'
        return row
    if target.radius <= CLEAR_R:
        row['reason'] = 'already_clear_certified'
        row['expected_radius_m'] = float(target.radius)
        return row
    if target.measured_at(q):
        row['reason'] = 'fixed_error_repeat'
        return row
    if target.particles is None or target.weights is None or not len(target.particles):
        row['reason'] = 'posterior_samples_unavailable'
        return row
    row.update(_known_geometry(target, q))
    p = world.params
    n = int(n if n is not None else getattr(p, 'scan_scenarios', 24))
    if n < 2:
        raise ValueError('scan scenarios must be at least 2')
    poly = target.polygon.copy()
    for center, radius, _ in target.exclusions:
        poly = exclude_disk(poly, np.asarray(center), float(radius))
    if not len(poly):
        poly = target.polygon.copy()
    history = np.asarray([o.position for o in target.observations], dtype=float).reshape(-1, 2)
    seed = p.model_seed + 1009 * target.channel + 31 * len(history) + target.misses + 781003
    points, radii, _ = scenarios(target, n, seed)
    seeds = np.random.default_rng(seed + 3011).integers(1, np.iinfo(np.int64).max, size=n, dtype=np.uint64)
    entry = np.asarray(target.center, dtype=float)
    before, after, radii_after = _paired_values(poly, entry, q, points, radii, seeds, history, p.circle_sides)
    valid = (before < 1e6) & (after < 1e6) & np.isfinite(radii_after)
    row.update(scenarios=n, entry_position=entry.tolist(),
               guard_probability=float(1. - valid.mean()),
               expected_radius_m=float(radii_after.mean()),
               probability_r20=float(np.mean(radii_after <= CLEAR_R)),
               expected_radius_ratio=float(radii_after.mean() / max(target.radius, 1e-9)),
               scan_cost_s=float(5 + 2 * (world.channel != target.channel)),
               switch_in_s=int(world.channel != target.channel),
               reserved_restore_switch_s=int(world.channel != target.channel))
    if not valid.all():
        row['reason'] = 'continuation_guard'
        return row
    savings = before - after
    gross = float(savings.mean())
    se = float(savings.std(ddof=1) / math.sqrt(n))
    row.update(valid=True, reason='paired_continuation', gross_saved_s=gross,
               net_saved_s=gross - row['scan_cost_s'], scenario_se_s=se,
               risk_adjusted_net_s=gross - row['scan_cost_s'] - getattr(p, 'scan_risk_weight', 0.) * se,
               baseline_future_s=float(before.mean()), scanned_future_s=float(after.mean()))
    return row


def assess_stop(world, early=False, mandatory=(), exclude=()):
    """Return (ordered channel IDs, diagnostics for every public channel).

    mandatory may force an unknown scan for even a single uncovered cell, but
    cannot override repeated-measurement, cleared, or certified-clear checks.
    It is meant for the coverage frontier's progress guarantee. Known mandatory
    measurements are evaluated and reported, even if over the normal scan cap.
    """
    p = world.params
    q = np.asarray(world.position, dtype=float)
    mandatory = set(mandatory)
    exclude = set(exclude)
    rows = {}
    known = []
    threshold = getattr(p, 'early_scan_min_net_s', 0.) if early else getattr(p, 'scan_min_net_s', 1.)
    unknown_threshold = (getattr(p, 'early_unknown_scan_gain', .015) if early
                         else getattr(p, 'unknown_scan_gain', .035))
    for target in world.targets.values():
        c = target.channel
        row = rows[c] = _row(target, q)
        row['mandatory'] = c in mandatory
        if c in exclude:
            row['reason'] = 'excluded'
        elif target.status == 'cleared':
            row['reason'] = 'cleared'
        elif target.measured_at(q):
            row['reason'] = 'fixed_error_repeat'
        elif target.status == 'active':
            if target.radius <= CLEAR_R:
                row['reason'] = 'already_clear_certified'
                row['expected_radius_m'] = float(target.radius)
            elif target.particles is None or target.weights is None:
                row['reason'] = 'posterior_samples_unavailable'
            else:
                row.update(_known_geometry(target, q))
                known.append(target)
        elif world.coverage.complete(c):
            row['reason'] = 'absence_certified'
        else:
            gain = float(world.coverage.gain(c, q))
            remaining = float(np.mean(~world.coverage.covered[c - 1]))
            relative = gain / remaining if remaining else 0.
            row.update(valid=True, coverage_gain=gain, remaining_fraction=remaining,
                       relative_coverage_gain=relative, coverage_threshold=unknown_threshold)
            if gain <= 0:
                row['reason'] = 'no_new_certified_coverage'
            elif c in mandatory:
                row.update(selected=True, reason='mandatory_coverage_progress')
            elif gain >= unknown_threshold:
                row.update(selected=True, reason='coverage_gain_policy')
            elif remaining <= unknown_threshold and relative >= .85:
                row.update(selected=True, reason='close_small_remaining_patch')
            else:
                row['reason'] = 'coverage_below_policy_threshold'
    known.sort(key=lambda target: (target.channel in mandatory,
                                  rows[target.channel]['geometry_screen_score']), reverse=True)
    budget = int(getattr(p, 'scan_candidates', 8))
    evaluated = []
    for index, target in enumerate(known):
        c = target.channel
        if index >= budget and c not in mandatory:
            rows[c]['reason'] = 'geometry_screen_budget'
            continue
        row = known_scan_value(world, target, q)
        row['mandatory'] = c in mandatory
        row['selection_threshold_s'] = float(threshold)
        rows[c] = row
        if row['valid']:
            evaluated.append(row)
    evaluated.sort(key=lambda row: (row['mandatory'], row['risk_adjusted_net_s']), reverse=True)
    count = 0
    for row in evaluated:
        if row['mandatory']:
            row.update(selected=True, reason='mandatory_known_measurement')
            count += 1
        elif row['risk_adjusted_net_s'] <= threshold:
            row['reason'] = 'insufficient_expected_seconds_saved'
        elif count >= int(getattr(p, 'scan_max_known', 5)):
            row['reason'] = 'known_scan_budget'
        else:
            row.update(selected=True, reason='positive_expected_seconds_saved')
            count += 1
    selected = [row for row in rows.values() if row['selected']]
    # Starting with the tuned channel, when selected, saves one switch. Channel
    # numbers otherwise have no physical switching distance, so value breaks ties.
    selected.sort(key=lambda row: (row['channel'] != world.channel,
                                  not row['mandatory'],
                                  -row['gross_saved_s'], -row['coverage_gain'], row['channel']))
    previous = world.channel
    for index, row in enumerate(selected):
        switch = int(previous != row['channel'])
        restore = int(index == len(selected) - 1 and row['channel'] != world.channel)
        row.update(batch_order=index, batch_switch_in_s=switch,
                   batch_reserved_restore_s=restore, batch_scan_cost_s=float(5 + switch + restore))
        previous = row['channel']
    return [row['channel'] for row in selected], [rows[c] for c in sorted(rows)]

"""Public scan disks, stop history, and unrestricted channel coverage frontiers.

A travelled segment is a visit record, never a detection certificate.  Only an
accepted measurement creates a disk for its channel; ``World.coverage`` remains
the conservative whole-cell certificate used to decide completion.
"""
from functools import lru_cache

import numpy as np

from ..model import ARENA_R, RECEIVE_MIN, SPEED


class ExplorationMemory:
    """Serializable public history with a bounded, optional revisit cost."""

    def __init__(self, params=None):
        self.radius = float(getattr(params, 'revisit_radius', 120.))
        self.maximum_penalty = float(getattr(params, 'revisit_penalty_s', 15.))
        self.visits = []
        self.segments = []
        self.scan_disks = {channel: [] for channel in range(1, 21)}

    def observe(self, path, q, channel=None, result=None, command_index=None):
        """Record an accepted command, coalescing scans at one physical stop.

        ``path`` is the public command name, e.g. ``'/measure'`` or ``'/clear'``.
        Repeating a channel at the identical location does not create a second
        disk: the simulator's location-dependent direction error is fixed.
        """
        if q is None or (isinstance(result, dict) and result.get('accepted') is False):
            return
        position = np.asarray(q, dtype=float)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError('A visited stop must have two finite coordinates')
        same_stop = bool(self.visits and np.linalg.norm(
            position - self.visits[-1]['position']) < 1e-7)
        if not same_stop:
            if self.visits:
                self.segments.append(dict(
                    start=list(self.visits[-1]['position']), end=position.tolist(),
                    command_index=command_index))
            self.visits.append(dict(
                position=position.tolist(), first_command_index=command_index,
                last_command_index=command_index, channels=[], commands=[]))
        visit = self.visits[-1]
        visit['last_command_index'] = command_index
        visit['commands'].append(dict(path=path, channel=channel,
                                      command_index=command_index))
        kind = result.get('measure_result') if isinstance(result, dict) else result
        if path != '/measure' or kind not in ('no_signal', 'direction', 'near'):
            return
        if channel not in self.scan_disks:
            raise ValueError('Measurement channel must be in 1..20')
        channel = int(channel)
        if channel not in visit['channels']:
            visit['channels'].append(channel)
        previous = self.scan_disks[channel]
        if any(np.linalg.norm(position - disk['center']) < 1e-7 for disk in previous):
            return
        previous.append(dict(center=position.tolist(), radius=RECEIVE_MIN,
                             result=kind, command_index=command_index))

    def penalty(self, q, channel=None, useful=False):
        """Seconds of soft revisit discouragement; useful actions can waive it.

        The current physical stop is free, so deciding to measure another
        channel there never looks like a second journey.  A caller can exempt
        certified clearing, substantial information, or necessary coverage.
        """
        if useful or len(self.visits) < 2 or self.maximum_penalty <= 0:
            return 0.
        q = np.asarray(q, dtype=float)
        if np.linalg.norm(q - self.visits[-1]['position']) < 1e-7:
            return 0.
        old = [visit['position'] for visit in self.visits[:-1]
               if channel is None or channel in visit['channels']]
        if not old:
            return 0.
        distance_squared = np.sum((np.asarray(old) - q) ** 2, axis=1)
        kernel = np.exp(-distance_squared / (2 * max(self.radius, 1e-6) ** 2))
        return float(self.maximum_penalty * min(1., float(kernel.sum())))

    def snapshot(self):
        # Build independent plain-JSON containers; future observations must not
        # mutate an earlier event's snapshot.
        return dict(
            detection_radius_m=RECEIVE_MIN,
            revisit_radius_m=self.radius,
            revisit_penalty_cap_s=self.maximum_penalty,
            visits=[dict(position=list(v['position']),
                         first_command_index=v['first_command_index'],
                         last_command_index=v['last_command_index'],
                         channels=list(v['channels']),
                         commands=[dict(c) for c in v['commands']])
                    for v in self.visits],
            segments=[dict(start=list(s['start']), end=list(s['end']),
                           command_index=s['command_index']) for s in self.segments],
            scan_disks_by_channel={str(channel): [dict(
                center=list(disk['center']), radius=disk['radius'],
                result=disk['result'], command_index=disk['command_index'])
                for disk in disks] for channel, disks in self.scan_disks.items()})


def _project(points):
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1)
    return points * np.minimum(1., (ARENA_R - 1e-5) / np.maximum(norms, 1.))[:, None]


@lru_cache(maxsize=8)
def _arena_grid(spacing):
    axis = np.arange(-ARENA_R, ARENA_R + spacing / 2, spacing)
    x, y = np.meshgrid(axis, axis)
    points = np.column_stack((x.ravel(), y.ravel()))
    return points[np.linalg.norm(points, axis=1) <= ARENA_R]


def frontier_candidates(world, memory, limit=4):
    """Rank whole-arena coverage opportunities without fixed ring or sectors.

    Returned positions guarantee at least one newly certified whole cell for
    one unknown channel.  Including projected uncovered-cell centers prevents
    a coarse candidate grid from missing tiny or boundary residual holes.
    Scores are a coverage scheduling heuristic, not expected source clear time.
    The main planner compares service actions and their downstream time costs.
    """
    if limit <= 0:
        return []
    channels = [target.channel for target in world.unknown()]
    if not channels:
        return []
    coverage = world.coverage
    holes = ~coverage.covered[np.asarray(channels) - 1]
    needed = holes.sum(axis=0)
    missing = coverage.centers[needed > 0]
    if not len(missing):
        return []
    spacing = float(getattr(world.params, 'frontier_spacing', 240.))
    samples = missing[np.linspace(0, len(missing) - 1, min(80, len(missing)), dtype=int)]
    mean = np.average(coverage.centers, axis=0, weights=needed)
    toward_mean = world.position + np.array([.65, .85, 1.])[:, None] * (mean - world.position)
    nearby = [target.center for target in world.active()
              if target.center is not None and np.linalg.norm(target.center - world.position)
              <= float(getattr(world.params, 'nearby_distance', 500.))]
    blocks = [_arena_grid(spacing), samples, toward_mean, np.asarray(world.position)[None, :]]
    if nearby:
        blocks.append(np.asarray(nearby))
    points = np.unique(np.round(_project(np.vstack(blocks)), decimals=7), axis=0)

    # Batch masks rather than asking Coverage.mask_at for every channel.  Every
    # bit uses the farthest square corner, exactly like the coverage certificate.
    gains = np.empty((len(points), len(channels)), dtype=np.int32)
    holes_int = holes.astype(np.float32)
    for start in range(0, len(points), 64):
        far = np.abs(coverage.centers[None, :, :] - points[start:start + 64, None, :]) + coverage.half
        masks = np.sum(far * far, axis=2) <= RECEIVE_MIN ** 2 - 1e-5
        gains[start:start + len(masks)] = masks.astype(np.float32) @ holes_int.T
    remaining = holes.sum(axis=1)
    total = float(remaining.sum())
    rows = []
    for q, channel_gains in zip(points, gains):
        useful_channels = [channel for channel, gain in zip(channels, channel_gains) if gain > 0]
        if not useful_channels:
            continue
        gain = float(channel_gains.sum() / total)
        closes = int(np.count_nonzero(channel_gains == remaining))
        travel = float(np.linalg.norm(q - world.position) / SPEED)
        # A return that closes a channel's last hole is necessary progress and
        # can be exempt.  Ordinary marginal recoverage remains softly discouraged.
        revisit = memory.penalty(q, useful=bool(closes)) if memory is not None else 0.
        closure_fraction = closes / len(channels)
        bonus = float(getattr(world.params, 'frontier_closure_bonus', .35)) * closure_fraction
        score = (gain + bonus) / (travel + 6. * len(useful_channels) + revisit + 1.)
        rows.append(dict(position=q.copy(), channels=useful_channels, gain=gain,
                         gain_cells=int(channel_gains.sum()),
                         channel_gain_cells={str(channel): int(value)
                                             for channel, value in zip(channels, channel_gains) if value > 0},
                         score=float(score), travel_s=travel, penalty_s=float(revisit),
                         closes=closes, closure=closure_fraction))
    rows.sort(key=lambda row: (-row['score'], row['travel_s']))
    selected = []
    separation = min(150., spacing * .6)
    for row in rows:
        if all(np.linalg.norm(row['position'] - other['position']) >= separation for other in selected):
            selected.append(row)
            if len(selected) >= limit:
                break
    return selected

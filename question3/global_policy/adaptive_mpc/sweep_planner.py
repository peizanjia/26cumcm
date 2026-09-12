"""Isolated local comparison: interruptible service with a dynamic sweep backbone.

The sectors order unfinished work, not fixed hexagon vertices. Each real probe
returns to the outer route planner unless the explicit lock-service ablation is
enabled. Only the per-channel public coverage certificate can close a sector.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import gzip
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ..joint_rollout.planner import Planner as JointPlanner
from ..joint_rollout.service import choose_action
from ..dynamic.frontier import Sweep, nearby_targets
from ..decisions import choose_initial_probe, localization_value
from .parameters import AdaptiveParameters
from .exploration import ExplorationMemory
from .scanning import assess_stop, known_scan_value


@dataclass
class SweepParameters(AdaptiveParameters):
    unlock_service: bool = True
    value_scans: bool = True
    initial_map_probe: bool = True
    early_geometry_mapping: bool = False
    early_geometry_stops: int = 6
    # Development-v1 exposed excessive unknown-channel scanning. These policy
    # thresholds affect optional scans only; mandatory frontier scans survive.
    unknown_scan_gain: float = .08
    early_unknown_scan_gain: float = .035

    def validate(self):
        super().validate()
        for name in ('unlock_service', 'value_scans', 'initial_map_probe', 'early_geometry_mapping'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(name)
        if (isinstance(self.early_geometry_stops, bool) or not isinstance(self.early_geometry_stops, int)
                or not 1 <= self.early_geometry_stops <= 20):
            raise ValueError('early_geometry_stops')
        return self


class Planner(JointPlanner):
    def __init__(self, client, parameters=None):
        super().__init__(client, parameters or SweepParameters())
        self.memory = ExplorationMemory(self.world.params)
        self.service_steps = {}
        self.last_target = None
        self.target_switches = 0
        self.replans = 0

    def execute(self, path, q=None, channel=None, reason=''):
        result = super().execute(path, q, channel, reason)
        if q is not None:
            self.memory.observe(path, q, channel, result, len(self.world.commands) - 1)
        return result

    def early(self):
        return self.world.params.early_mapping and len(self.memory.visits) <= self.world.params.early_stops

    def scan_here(self, mandatory=(), reason='opportunistic_stop'):
        w = self.world
        if not w.params.value_scans:
            return super().scan_here(mandatory, reason)
        channels, rows = assess_stop(w, early=self.early(), mandatory=mandatory)
        geometric = []
        nonorigin_stops = len(self.memory.visits) - 1
        if (w.params.early_geometry_mapping and w.params.stop_scans
                and 1 <= nonorigin_stops <= w.params.early_geometry_stops):
            by_channel = {row['channel']: row for row in rows}
            for target in w.active():
                if (target.channel in channels or target.radius <= 20
                        or target.measured_at(w.position)):
                    continue
                row = by_channel[target.channel]
                if not row.get('valid'):
                    row = known_scan_value(w, target, w.position)
                    by_channel[target.channel] = row
                if (row.get('valid') and row.get('expected_radius_ratio', math.inf) < .6
                        and row.get('detection_probability', 0.) > .5):
                    row.update(selected=True, reason='early_geometry_mapping_spend',
                               geometry_mapping=True, geometry_mapping_stop=nonorigin_stops,
                               value_kind='geometry_map_investment_not_positive_time_voi')
                    geometric.append(target.channel)
                    channels.append(target.channel)
            rows = [by_channel[c] for c in sorted(by_channel)]
        if not w.params.stop_scans:
            channels = [c for c in channels if c in mandatory]
        self.event('stop_scan', selected=channels, values=rows, early=self.early(),
                   geometric_mapping_channels=geometric, nonorigin_stop=nonorigin_stops)
        for c in channels:
            if not w.targets[c].measured_at(w.position):
                self.execute('/measure', w.position, c, 'early_geometry_mapping_spend' if c in geometric else reason)

    def service_target(self, channel):
        w = self.world
        target = w.targets[channel]
        limit = 1 if w.params.unlock_service else 24
        for _ in range(limit):
            if target.status == 'cleared':
                return
            steps = self.service_steps.get(channel, 0)
            if steps >= 80:
                raise RuntimeError(f'Channel {channel}: interruptible service progress guard')
            path, q, reason, values = choose_action(w, target, steps)
            # Keep original continuation times and expose the separate optional
            # revisit cost. Useful geometry and guaranteed clears can revisit.
            if values:
                for row in values:
                    y = np.asarray(row['destination'])
                    useful = ((row['action'] == '/clear' and target.radius <= 20)
                              or localization_value(target, y, w) > .15)
                    row['revisit_penalty_s'] = self.memory.penalty(y, channel=channel, useful=useful)
                    row['selection_score_s'] = row['expected_remaining_s'] + row['revisit_penalty_s']
                    row.pop('selected', None)
                best = min(values, key=lambda row: row['selection_score_s'])
                best['selected'] = True
                path, q, reason = best['action'], np.asarray(best['destination']), 'sweep_' + best['name']
            self.replans += 1
            if self.last_target is not None and channel != self.last_target:
                self.target_switches += 1
            self.last_target = channel
            self.event('local_decision', channel=channel, action=path, destination=q.tolist(),
                       radius=target.radius, hit_probability=target.hit_probability(q),
                       reason=reason, candidates=values, sector=self.sweep.index,
                       interruptible=w.params.unlock_service)
            self.execute(path, q, channel, reason)
            self.service_steps[channel] = steps + 1
            # In the unlocked branch the outer loop invokes scan_here exactly
            # once before replanning its route. Locked+VOI is a separate ablation.
            if not w.params.unlock_service and w.params.value_scans:
                self.scan_here(reason='locked_probe_stop')
        if not w.params.unlock_service and target.status != 'cleared':
            raise RuntimeError(f'Channel {channel}: locked service guard')

    def clean_neighborhood(self):
        w = self.world
        for _ in range(160):
            nearby = nearby_targets(w)
            if not nearby:
                return
            frontier = self.sweep.next_point()
            route = self.route(nearby, frontier[1] if frontier else w.position, open_end=frontier is None)
            self.event('nearby_cleanup_route', channels=route, sector=self.sweep.index, inner_phase=False)
            self.service_target(route[0])
            self.scan_here()
        raise RuntimeError('Interruptible neighborhood progress guard')

    def plan(self):
        w = self.world
        self.event('origin_scan')
        for c in range(1, 21):
            self.execute('/measure', np.zeros(2), c, 'origin_all_channels')
        self.sweep = Sweep(w)
        self.event('sweep_start', heading=self.sweep.heading, sign=self.sweep.sign,
                   sectors=w.params.sector_count)
        if w.params.initial_map_probe and w.active():
            q = choose_initial_probe(w)
            target = max(w.active(), key=lambda t: localization_value(t, q, w))
            self.event('initial_probe_uncommitted', channel=target.channel, destination=q.tolist())
            self.execute('/measure', q, target.channel, 'initial_map_probe')
            self.scan_here()
        remaining = set(range(w.params.sector_count))
        while remaining:
            index = self.next_sector(remaining)
            self.sweep.index = index
            for _ in range(240):
                self.clean_neighborhood()
                if self.sweep.can_close():
                    self.sweep.closed.append(index)
                    remaining.remove(index)
                    self.event('sector_closed', sector=index)
                    break
                targets = self.sweep.outstanding()
                frontier = self.sweep.next_point()
                if targets:
                    route = self.route(targets, frontier[1] if frontier else w.position,
                                       open_end=frontier is None)
                    self.event('region_cleanup_route', channels=route, sector=index)
                    self.service_target(route[0])
                    self.scan_here()
                    continue
                if frontier is None:
                    raise RuntimeError('Incomplete sector has no productive frontier')
                score, q, channels = frontier
                self.event('coverage_frontier', sector=index, destination=q.tolist(),
                           gain_per_second=score, channels=channels, nearby_left=[])
                self.execute('/measure', q, channels[0], 'coverage_frontier')
                self.scan_here(channels[1:], 'coverage_frontier')
            else:
                raise RuntimeError('Interruptible sector progress guard')
        if not w.finished():
            raise RuntimeError('Invalid public completion certificate')
        self.event('coverage_closed')

    def run(self):
        row = super().run()
        w = self.world
        components = dict(movement=w.distance / 5., measure=0, switch=0, clear=0)
        previous = 1
        for command in w.commands:
            if command['path'] == '/measure':
                components['measure'] += 5
                components['switch'] += int(command['channel'] != previous)
                previous = command['channel']
            elif command['path'] == '/clear':
                components['clear'] += 5 if command['response']['clear_result'] == 'success' else 3
        row.update(policy='adaptive_sweep', replans=self.replans, target_switches=self.target_switches,
                   stop_count=len(self.memory.visits), time_components_s=components)
        row['geometric_mapping_scans'] = sum(c.get('reason') == 'early_geometry_mapping_spend' for c in w.commands)
        return row

    def save(self, directory, summary):
        super().save(directory, summary)
        (Path(directory) / 'exploration_memory.json').write_text(
            json.dumps(self.memory.snapshot(), ensure_ascii=False), encoding='utf8')


def run_local(seed, params=None, output=None):
    from question3.local_sim.simulator import Simulator, Client
    simulator = Simulator(seed)
    planner = Planner(Client(simulator=simulator), params)
    row = planner.run()
    row.update(seed=seed, evaluation='local_synthetic', true_total=len(simulator.sources),
               true_cleared=sum(source.cleared for source in simulator.sources))
    if row['complete'] and not all(source.cleared for source in simulator.sources):
        row.update(complete=False, failure='Evaluator found an uncleared source')
    if output:
        planner.save(output, row)
        simulator.dump(Path(output) / 'evaluator')
    return row, planner


VARIANTS = {
    'unlock_old': dict(unlock_service=True, value_scans=False, initial_map_probe=False, revisit_penalty_s=0.),
    'unlock_voi': dict(unlock_service=True, value_scans=True, initial_map_probe=False, revisit_penalty_s=0.),
    'lock_voi': dict(unlock_service=False, value_scans=True, initial_map_probe=False, revisit_penalty_s=0.),
    'unlock_voi_initial': dict(unlock_service=True, value_scans=True, initial_map_probe=True, revisit_penalty_s=0.),
    'geometry_off': dict(unlock_service=True, value_scans=True, initial_map_probe=True,
                         early_geometry_mapping=False, unknown_scan_gain=.08,
                         early_unknown_scan_gain=.08, revisit_penalty_s=0.),
    'geometry_6': dict(unlock_service=True, value_scans=True, initial_map_probe=True,
                       early_geometry_mapping=True, early_geometry_stops=6,
                       unknown_scan_gain=.08, early_unknown_scan_gain=.08, revisit_penalty_s=0.),
    'geometry_3': dict(unlock_service=True, value_scans=True, initial_map_probe=True,
                       early_geometry_mapping=True, early_geometry_stops=3,
                       unknown_scan_gain=.08, early_unknown_scan_gain=.08, revisit_penalty_s=0.),
}
for _sector_count in (4, 5, 6, 8):
    VARIANTS[f'sector_{_sector_count}'] = dict(VARIANTS['geometry_6'], sector_count=_sector_count)
for _gain in (.06, .12):
    VARIANTS[f'sector_4_gain_{round(100 * _gain):02d}'] = dict(
        VARIANTS['sector_4'], unknown_scan_gain=_gain, early_unknown_scan_gain=_gain)


def _screen_task(arguments):
    seed, variant, output = arguments
    row, planner = run_local(seed, SweepParameters(**VARIANTS[variant]))
    row['variant'] = variant
    directory = Path(output) / variant
    directory.mkdir(parents=True, exist_ok=True)
    package = dict(summary=row, commands=planner.world.commands, events=planner.events,
                   exploration_memory=planner.memory.snapshot(),
                   truth=[asdict(source) for source in planner.client.simulator.sources])
    with gzip.open(directory / f'{seed}.json.gz', 'wt', encoding='utf8', compresslevel=3) as handle:
        json.dump(package, handle, ensure_ascii=False)
    (directory / f'{seed}.summary.json').write_text(json.dumps(row), encoding='utf8')
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--seed-start', type=int, default=20272300)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS),
                        default=['unlock_old', 'unlock_voi', 'lock_voi', 'unlock_voi_initial'])
    parser.add_argument('--output', default='question3/global_policy/adaptive_mpc/outputs/sweep_screen')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(seed, variant, str(output)) for seed in range(args.seed_start, args.seed_start + args.count)
             for variant in args.variants]
    existing = [output / variant / f'{seed}.summary.json' for seed, variant, _ in tasks]
    if any(path.exists() for path in existing):
        raise RuntimeError('Choose a fresh output directory; do not overwrite recorded experiments')
    (output / 'manifest.json').write_text(json.dumps(dict(
        seed_start=args.seed_start, count=args.count, variants={v: asdict(SweepParameters(**VARIANTS[v])) for v in args.variants},
        evaluation='local_synthetic_development'), indent=2), encoding='utf8')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_screen_task, task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(done=len(rows), total=len(tasks), seed=row['seed'], variant=row['variant'],
                                  average_time_s=row['average_time_s'], complete=row['complete'], failure=row['failure'])), flush=True)
    summary = {}
    for variant in args.variants:
        subset = [row for row in rows if row['variant'] == variant]
        summary[variant] = dict(count=len(subset), complete=sum(row['complete'] for row in subset),
            mean_s_per_source=float(np.mean([row['average_time_s'] for row in subset if row['complete']])),
            mean_components_s={key: float(np.mean([row['time_components_s'][key] for row in subset]))
                               for key in ('movement', 'measure', 'switch', 'clear')})
    (output / 'cases.json').write_text(json.dumps(rows), encoding='utf8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

"""Route-first development experiment joining known sources and coverage stops.

The entire open tour is rebuilt from public feedback after every service action.
Anticipated future coverage affects routing only: actual accepted measurements
and World.finished() are the sole completion certificate.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import gzip
import hashlib
import json
import os

import numpy as np

from ..decisions import choose_initial_probe, localization_value
from .anticipated_tail import anticipated_tail
from .exploration import frontier_candidates
from .planning import coverage_tail
from .sweep_planner import Planner as SweepPlanner, SweepParameters


def default_parameters():
    return SweepParameters(unlock_service=True, value_scans=True, initial_map_probe=True,
                           early_geometry_mapping=True, early_geometry_stops=6,
                           unknown_scan_gain=.08, early_unknown_scan_gain=.08,
                           revisit_penalty_s=0.)


class Planner(SweepPlanner):
    def __init__(self, client, parameters=None):
        super().__init__(client, parameters or default_parameters())
        if not self.world.params.unlock_service:
            raise ValueError('The route-first experiment requires one-action unlocked service')
        self.tour_replans = 0

    def _route_action(self, route):
        """First still-productive real action on the freshly constructed tour."""
        w = self.world
        skipped = []
        for point in route:
            q = np.asarray(point, dtype=float)
            targets = [target for target in w.active() if np.linalg.norm(target.center - q) < 1e-5]
            if targets:
                target = min(targets, key=lambda target: (target.radius, target.channel))
                return dict(kind='known', position=q, channel=target.channel, skipped=skipped)
            channels = [target.channel for target in w.unknown()
                        if not target.measured_at(q) and w.coverage.gain(target.channel, q) > 0]
            if channels:
                channels.sort(key=lambda channel: (channel != w.channel, -w.coverage.gain(channel, q), channel))
                return dict(kind='coverage', position=q, channel=channels[0],
                            mandatory=channels[1:], skipped=skipped)
            skipped.append(q.tolist())
        # Forecast station lists may become redundant as knowledge changes. An
        # actual positive whole-cell gain remains a valid finite-progress repair.
        frontiers = frontier_candidates(w, self.memory, limit=1)
        if frontiers:
            frontier = frontiers[0]
            return dict(kind='coverage', position=np.asarray(frontier['position']),
                        channel=frontier['channels'][0], mandatory=frontier['channels'][1:],
                        skipped=skipped, repaired=True)
        raise RuntimeError('Unfinished world has no productive point on the route')

    def plan(self):
        w = self.world
        # Compatibility metadata for inherited service diagnostics; no angular
        # mask or sector closure is constructed or used by the route planner.
        self.sweep = SimpleNamespace(index=-1, closed=[], heading=0., sign=1., width=2*np.pi)
        self.event('origin_scan')
        for channel in range(1, 21):
            self.execute('/measure', np.zeros(2), channel, 'origin_all_channels')
        if w.params.initial_map_probe and w.active():
            q = choose_initial_probe(w)
            target = max(w.active(), key=lambda target: localization_value(target, q, w))
            self.event('initial_probe_uncommitted', channel=target.channel, destination=q.tolist())
            self.execute('/measure', q, target.channel, 'initial_map_probe')
            self.scan_here()
        for cycle in range(240):
            if w.finished():
                self.event('coverage_closed')
                return
            self.tour_replans += 1
            stations = coverage_tail(w)
            # With zero source residuals, this is a travel/coverage tour, not a
            # speculative full-source action value that removes the source first.
            cost, route = anticipated_tail(w, {target.channel: 0. for target in w.active()},
                                           start=w.position, removed=None, y=w.position,
                                           unknown_channels=[], stations=stations, early=self.early())
            action = self._route_action(route)
            self.event('global_tour_replan', cycle=cycle, route=route,
                       forecast_route_seconds=cost, channel=action['channel'],
                       kind=action['kind'], destination=action['position'].tolist(),
                       skipped_nonproductive=action['skipped'], repaired=action.get('repaired', False),
                       value_kind='open_route_plus_paid_forecast_scans_without_source_service_residual')
            if action['kind'] == 'known':
                self.service_target(action['channel'])
                self.scan_here()
            else:
                self.event('coverage_frontier', sector=-1, destination=action['position'].tolist(),
                           channels=[action['channel']] + action['mandatory'], nearby_left=[])
                self.execute('/measure', action['position'], action['channel'], 'tour_coverage_frontier')
                self.scan_here(action['mandatory'], 'tour_coverage_frontier')
        raise RuntimeError('Route-first 240-cycle progress guard reached')

    def run(self):
        row = super().run()
        row.update(policy='adaptive_tour', tour_replans=self.tour_replans,
                   route_architecture='open_tour_of_known_centers_and_residual_coverage')
        return row


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


def _screen_task(arguments):
    seed, output = arguments
    row, planner = run_local(seed)
    directory = Path(output)
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
    parser.add_argument('--output', default='question3/global_policy/adaptive_mpc/outputs/tour_screen')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.count))
    if any((output / f'{seed}.summary.json').exists() for seed in seeds):
        raise RuntimeError('Choose a fresh output directory')
    source = Path(__file__).resolve()
    dependencies = ['tour_planner.py', 'sweep_planner.py', 'anticipated_tail.py', 'planning.py', 'scanning.py']
    (output / 'manifest.json').write_text(json.dumps(dict(
        seeds=seeds, parameters=asdict(default_parameters()), evaluation='local_synthetic_development',
        source_sha256={name: hashlib.sha256((source.parent / name).read_bytes()).hexdigest() for name in dependencies},
        completion='actual accepted channel coverage and actual source clearing only',
        forecast='anticipated_tail constructs route; no source removed before actual feedback'), indent=2), encoding='utf8')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
        os.environ[name] = '1'
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_screen_task, (seed, str(output))) for seed in seeds]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(dict(done=len(rows), total=len(seeds), seed=row['seed'],
                                  average_time_s=row['average_time_s'], complete=row['complete'],
                                  failure=row['failure'])), flush=True)
    successful = [row for row in rows if row['complete']]
    summary = dict(count=len(rows), complete=len(successful),
                   mean_s_per_source=float(np.mean([row['average_time_s'] for row in successful])) if successful else None,
                   mean_components_s={key: float(np.mean([row['time_components_s'][key] for row in rows]))
                                      for key in ('movement', 'measure', 'switch', 'clear')})
    (output / 'cases.json').write_text(json.dumps(rows), encoding='utf8')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

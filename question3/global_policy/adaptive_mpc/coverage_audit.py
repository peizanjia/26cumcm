"""Read-only movement/coverage diagnosis of saved local synthetic replays.

Replay truth is used only for evaluator denominators. Public response streams
determine discovered/cleared states. No audited route is fed back into a run.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from ..model import Coverage, World


def inspect_pack(pack):
    summary = pack['summary']
    commands = pack['commands']
    coverage = Coverage(summary['parameters']['coverage_cell'])
    discovered = set()
    cleared = set()
    position = np.zeros(2)
    last_time = 0.
    rows = []
    stops = []
    source_discoveries = {}
    counters = defaultdict(float)
    active_empty_entries = 0
    previous_active_empty = False
    previous_mode = None
    mode_switches = 0
    for index, command in enumerate(commands):
        if command['path'] not in ('/measure', '/clear'):
            continue
        channel = command['channel']
        response = command['response']
        q = np.array(command['position'])
        distance = float(np.linalg.norm(q - position))
        unknown = channel not in discovered
        active = discovered - cleared
        clock = response['virtual_time_s']
        elapsed = clock - last_time
        mode = 'unknown_exploration' if unknown else ('clear' if command['path'] == '/clear' else 'known_measure')
        if distance > 1e-7:
            counters['movement_' + mode + '_s'] += distance / 5
            counters['stops_' + mode] += 1
            if not active:
                counters['movement_while_no_known_active_s'] += distance / 5
            if previous_mode is not None and previous_mode != mode:
                mode_switches += 1
            previous_mode = mode
            revisit = any(np.linalg.norm(q - old['q']) < 120. for old in stops[:-2])
            if revisit:
                counters['revisit_stops_120m'] += 1
                counters['movement_ending_near_old_stop_s'] += distance / 5
            stops.append(dict(q=q.copy(), index=index, time=clock, mode=mode,
                              active_count=len(active), move_s=distance / 5))
        gained = 0
        if command['path'] == '/measure':
            mask = coverage.mask_at(q)
            gained = int(np.count_nonzero(mask & ~coverage.covered[channel - 1]))
            coverage.mark(channel, q)
            if unknown:
                counters['unknown_scans'] += 1
                if gained == 0:
                    counters['unknown_scans_zero_certified_gain'] += 1
            else:
                counters['known_scans'] += 1
            if response['measure_result'] in ('direction', 'near') and unknown:
                discovered.add(channel)
                source_discoveries[channel] = dict(time=clock, index=index, position=q.tolist())
        if command['path'] == '/clear' and response['clear_result'] == 'success':
            cleared.add(channel)
        empty = not (discovered - cleared)
        if empty and not previous_active_empty:
            active_empty_entries += 1
        previous_active_empty = empty
        rows.append(dict(index=index, time=clock, elapsed_s=elapsed, move_s=distance / 5,
                         active_after=len(discovered-cleared), discovered=len(discovered),
                         cleared=len(cleared), mode=mode, position=q.tolist(),
                         gained=gained, path=command['path'], channel=channel))
        position = q
        last_time = clock
    final_time = summary['virtual_time_s']
    n = summary['true_total']
    last_clear = max((row for row in rows if row['cleared'] == n), key=lambda row: -row['time'])
    last_discovery = max(source_discoveries.values(), key=lambda item: item['time'])
    suffix = [row for row in rows if row['index'] > last_clear['index']]
    discovery_times = sorted(item['time'] for item in source_discoveries.values())
    eligible_stops = [stop for stop in stops if np.linalg.norm(stop['q']) >= 400.]
    angle_reversal = 0
    angle_travel = 0.
    if len(eligible_stops) > 2:
        angles = np.array([np.arctan2(stop['q'][1], stop['q'][0]) for stop in eligible_stops])
        changes = (np.diff(angles) + np.pi) % (2*np.pi) - np.pi
        signs = np.sign(changes[np.abs(changes) >= .08])
        angle_reversal = int(np.count_nonzero(signs[1:] != signs[:-1]))
        angle_travel = float(np.abs(changes).sum())
    components = summary['time_components_s']
    result = dict(seed=summary['seed'], variant=summary['variant'], complete=summary['complete'],
                  source_count=n, time_s=final_time, average_s=summary['average_time_s'],
                  movement_s=components['movement'], measure_s=components['measure'],
                  switch_s=components['switch'], clear_s=components['clear'],
                  final_source_clear_time_s=last_clear['time'],
                  post_clear_coverage_time_s=final_time-last_clear['time'],
                  post_clear_coverage_movement_s=sum(row['move_s'] for row in suffix),
                  post_clear_coverage_commands=len(suffix),
                  last_discovery_time_s=last_discovery['time'],
                  last_discovery_fraction=last_discovery['time']/final_time,
                  half_discovered_time_s=discovery_times[(len(discovery_times)-1)//2],
                  active_empty_entries=active_empty_entries,
                  move_purpose_switches=mode_switches,
                  angular_direction_changes=angle_reversal,
                  angular_total_radians=angle_travel,
                  long_reversals=summary.get('long_reversals'),
                  counters=dict(counters))
    for key in ('post_clear_coverage_time_s', 'post_clear_coverage_movement_s', 'movement_s',
                'measure_s', 'switch_s', 'clear_s'):
        result[key + '_per_source'] = result[key] / n
    for key in ('movement_unknown_exploration_s', 'movement_clear_s', 'movement_known_measure_s'):
        result[key + '_per_source'] = counters[key] / n
    return result


def mean_rows(rows):
    scalar = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key != 'seed']
    result = {key:float(np.mean([row[key] for row in rows])) for key in scalar}
    names = sorted(set().union(*(row['counters'] for row in rows)))
    result['counters'] = {key:float(np.mean([row['counters'].get(key,0) for row in rows])) for key in names}
    return result


def post_clear_counterfactual(pack):
    """Public-state continuation estimate from the logged final clear point.

    Uses no hidden source coordinates; evaluator only selects the checkpoint.
    It is an offline route substitute, not a newly executed simulator score.
    """
    from .anticipated_tail import anticipated_tail
    from .parameters import AdaptiveParameters
    from .planning import coverage_tail

    params = AdaptiveParameters(**pack['summary']['parameters'])
    world = World(params)
    cleared = 0
    checkpoint = 0.
    for command in pack['commands']:
        q = command['position']
        if q is None:
            continue
        channel = command['channel']
        response = command['response']
        world.position = np.array(q)
        if command['path'] == '/measure':
            world.coverage.mark(channel,q)
            world.channel = channel
            if response['measure_result'] in ('direction','near'):
                world.targets[channel].status = 'active'
        elif command['path'] == '/clear' and response['clear_result'] == 'success':
            world.targets[channel].status = 'cleared'
            cleared += 1
        if cleared == pack['summary']['true_total']:
            checkpoint = response['virtual_time_s']
            break
    stations = coverage_tail(world)
    cost, route = anticipated_tail(world,{},world.position,None,world.position,[],stations)
    before = world.coverage.covered.copy()
    for q in route:
        for target in world.unknown():
            world.coverage.mark(target.channel,q)
    complete = not world.unknown()
    world.coverage.covered = before
    actual = pack['summary']['virtual_time_s'] - checkpoint
    return dict(seed=pack['summary']['seed'],variant=pack['summary']['variant'],
                source_count=cleared,actual_tail_s=actual,estimated_alternative_tail_s=cost,
                estimated_saved_s=actual-cost,estimated_saved_s_per_source=(actual-cost)/cleared,
                route=route,predicted_coverage_complete=complete)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='question3/global_policy/adaptive_mpc/outputs/development_v2')
    parser.add_argument('--output', default='question3/global_policy/adaptive_mpc/outputs/coverage_audit_v2.json')
    parser.add_argument('--post-clear-counterfactual', action='store_true')
    args = parser.parse_args()
    root = Path(args.input)
    rows = []
    counterfactuals = []
    for variant in ('combined', 'adaptive'):
        for path in sorted((root / variant).glob('*.json.gz')):
            with gzip.open(path, 'rt', encoding='utf8') as stream:
                pack = json.load(stream)
                rows.append(inspect_pack(pack))
                if args.post_clear_counterfactual:
                    counterfactuals.append(post_clear_counterfactual(pack))
    by_variant = {variant: [row for row in rows if row['variant'] == variant]
                  for variant in ('combined','adaptive')}
    means = {variant:mean_rows(items) for variant,items in by_variant.items()}
    paired = []
    control = {row['seed']:row for row in by_variant['combined']}
    for row in by_variant['adaptive']:
        before = control[row['seed']]
        paired.append(dict(seed=row['seed'],source_count=row['source_count'],
                           delta_s_per_source=row['average_s']-before['average_s'],
                           adaptive_s=row['average_s'],combined_s=before['average_s'],
                           adaptive_post_clear_s_per_source=row['post_clear_coverage_time_s_per_source'],
                           combined_post_clear_s_per_source=before['post_clear_coverage_time_s_per_source'],
                           adaptive_unknown_move_s_per_source=row['movement_unknown_exploration_s_per_source'],
                           combined_unknown_move_s_per_source=before['movement_unknown_exploration_s_per_source']))
    report = dict(evaluation='read_only_local_synthetic_replay_audit', input=str(root),
                  definitions=dict(post_clear='time strictly after last actual source clear; unavoidable absence proof may remain',
                                   movement_unknown='moving command addressed to a channel not yet discovered at that moment',
                                   revisit='endpoint within120m of a stop at least two distinct stops earlier; not necessarily waste',
                                   angular_changes='sign changes of polar movement outside400m; not necessarily waste'),
                  means=means,low_source_means={variant:mean_rows([row for row in items if row['source_count']<=11])
                                               for variant,items in by_variant.items()},
                  paired=sorted(paired,key=lambda row:-row['delta_s_per_source']),rows=rows)
    if counterfactuals:
        report['post_clear_counterfactuals'] = counterfactuals
        report['counterfactual_means'] = {variant: {key: float(np.mean([r[key] for r in counterfactuals if r['variant']==variant]))
                                                    for key in ('actual_tail_s','estimated_alternative_tail_s','estimated_saved_s','estimated_saved_s_per_source')}
                                           for variant in ('combined','adaptive')}
    Path(args.output).write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(dict(means=means,low_source_means=report['low_source_means'],worst=report['paired'][:5],
                         counterfactual_means=report.get('counterfactual_means')),indent=2))


if __name__ == '__main__':
    main()

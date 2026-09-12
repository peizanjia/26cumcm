"""Independently audit persisted full-mission histories without running a policy.

Uses evaluator truth to verify actual outcomes and costs after execution. No
strategy or simulator execution is imported. Continuous certificate algorithms
have separate geometry tests; this audit verifies their recorded terminal flag
and that every still-unseen channel really measured each recorded support stop.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
COST_KEYS = ('movement', 'measure', 'switch', 'clear_success', 'clear_failure')


class AuditFailure(AssertionError):
    pass


def require(condition, text):
    if not condition:
        raise AuditFailure(text)


def close(actual, expected, text, tolerance=2e-6):
    require(math.isfinite(float(actual)) and abs(float(actual)-float(expected)) <= tolerance,
            f'{text}: recorded={actual!r}, reconstructed={expected!r}')


def point_key(q):
    return tuple(float(x) or 0. for x in q)


def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def truth_signature(truth):
    return json.dumps(sorted(truth, key=lambda r: r['channel']), sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def audit_history(history, outer_summary):
    summary, commands, truth = history['summary'], history['commands'], history['truth']
    require(summary['completed'] and not summary.get('error'), 'Saved run is incomplete or failed')
    require(len(truth) == summary['source_count'], 'Source count differs from evaluator truth')
    sources = {int(row['channel']): row for row in truth}
    require(len(sources) == len(truth), 'Truth contains duplicate channels')
    require(all(not row['cleared'] for row in truth), 'History must contain initial uncleared truth')
    state = {c: dict(found_s=None, cleared_s=None, service_time_s=0., service_move_m=0.,
                     measures=0, clears=0, failed_clears=0) for c in range(1, 21)}
    cleared = set()
    negatives = defaultdict(set)
    repeats = {}
    optical_sites = defaultdict(set)
    cost = {k: 0. for k in COST_KEYS}
    phase_cost = defaultdict(lambda: dict(time_s=0., move_s=0., commands=0))
    channel, position, elapsed = 1, (0., 0.), 0.
    counts = dict(measures=0, clears=0, clear_failures=0, no_signal=0,
                  known_no_signal=0, optical_fallback_clears=0)
    repeated_measures = 0
    positive_measures = 0
    for index, row in enumerate(commands):
        context = f'action {index}'
        require(row['index'] == index, f'{context}: index discontinuity')
        require(point_key(row['from']) == point_key(position), f'{context}: movement does not start at previous stop')
        q, c, kind = point_key(row['position']), int(row['channel']), row['kind']
        require(len(q) == 2 and all(math.isfinite(v) and abs(v) <= 2e6 for v in q), f'{context}: invalid coordinates')
        require(1 <= c <= 20 and kind in ('measure', 'clear'), f'{context}: invalid action')
        response = row['response']
        require(response.get('accepted') is True, f'{context}: unaccepted command')
        source = sources.get(c) if c not in cleared else None
        d = distance(q, (source['x'], source['y'])) if source else math.inf
        move = distance(q, position)/5.
        before = elapsed
        switching = 0.
        was_known = state[c]['found_s'] is not None and c not in cleared
        if kind == 'measure':
            switching = float(c != channel)
            channel = c
            action_fee = 5.
            cost['measure'] += action_fee
            cost['switch'] += switching
            counts['measures'] += 1
            state[c]['measures'] += 1
            illuminated = False
            if source:
                projection = ((q[0]-source['x'])*math.cos(source['orientation'])+
                              (q[1]-source['y'])*math.sin(source['orientation']))
                illuminated = not source['directional'] or projection >= -1e-9
            expected = ('no_signal' if source is None or d > source['radius'] or not illuminated
                        else 'near' if d <= 5. else 'direction')
            require(response.get('measure_result') == expected, f'{context}: measurement violates range/orientation/near physics')
            key = (c, q)
            if c not in cleared:
                comparable = (expected, response.get('svd_deg'))
                if key in repeats:
                    repeated_measures += 1
                    require(repeats[key] == comparable, f'{context}: repeated-site bearing error changed')
                repeats[key] = comparable
            if expected == 'direction':
                exact = math.degrees(math.atan2(source['y']-q[1], source['x']-q[0]))
                error = abs((float(response['svd_deg'])-exact+180.) % 360.-180.)
                require(error <= 1.00500001, f'{context}: bearing error exceeds one degree plus rounding')
            if expected == 'no_signal':
                negatives[c].add(q)
                counts['no_signal'] += 1
                counts['known_no_signal'] += int(was_known)
            else:
                positive_measures += 1
        else:
            success = source is not None and d <= 20.
            expected = 'success' if success else 'no_target_in_range'
            require(response.get('clear_result') == expected, f'{context}: clear response violates actual 20 m optical range')
            action_fee = 5. if success else 3.
            cost['clear_success' if success else 'clear_failure'] += action_fee
            counts['clears'] += 1
            counts['clear_failures'] += int(not success)
            state[c]['clears'] += 1
            state[c]['failed_clears'] += int(not success)
            if success:
                require(c not in cleared, f'{context}: duplicate source clear')
                cleared.add(c)
            if row.get('reason') == 'optical_finite_cover':
                require(q not in optical_sites[c], f'{context}: finite optical fallback repeated a stop')
                optical_sites[c].add(q)
                require(len(optical_sites[c]) <= 122, f'{context}: optical fallback exceeds finite site bound')
                counts['optical_fallback_clears'] += 1
            # Receiver channel intentionally remains unchanged on clear.
        cost['movement'] += move
        elapsed += move+action_fee+switching
        close(row['move_s'], move, f'{context}: movement fee')
        close(row['action_s'], action_fee+switching, f'{context}: action/switch fee')
        close(row['time_s'], elapsed, f'{context}: cumulative elapsed time')
        close(response['virtual_time_s'], elapsed, f'{context}: response clock')
        for key in COST_KEYS:
            close(row['costs'][key], cost[key], f'{context}: cumulative {key} cost')
        if response.get('measure_result') in ('near', 'direction') and state[c]['found_s'] is None:
            state[c]['found_s'] = elapsed
        if response.get('clear_result') == 'success':
            state[c]['cleared_s'] = elapsed
        if row['phase'] in ('service', 'side_known'):
            state[c]['service_time_s'] += elapsed-before
            state[c]['service_move_m'] += move*5.
        phase = phase_cost[row['phase']]
        phase['time_s'] += move+action_fee+switching
        phase['move_s'] += move
        phase['commands'] += 1
        position = q
    require(len(commands) == summary['command_count'], 'Command count mismatch')
    require(set(sources) == cleared, 'Policy terminated while a real source remained uncleared')
    require(summary['cleared'] == len(cleared), 'Summary clear count mismatch')
    require(len(cleared) == 16 or summary['coverage_certified'], 'Termination lacks 16 clears or continuous coverage flag')
    require(all(state[c]['found_s'] is not None for c in sources), 'Cleared source has no first observation')
    absent_channels = set(range(1, 21))-set(sources)
    if summary['coverage_certified']:
        require(bool(history['search_points']), 'Coverage flag has no measured support stops')
        for q in history['search_points']:
            key = point_key(q)
            for c in absent_channels:
                require(key in negatives[c], f'Coverage stop {key} was not actually measured negatively on absent channel {c}')
    close(summary['time_s'], elapsed, 'Final elapsed time')
    close(sum(cost.values()), elapsed, 'Fee sum')
    close(summary['per_source_s'], elapsed/len(sources), 'Per-source time')
    for key in COST_KEYS:
        close(summary['costs'][key], cost[key], f'Final {key} fee')
        close(summary['costs_per_source'][key], cost[key]/len(sources), f'Per-source {key} fee')
    for c in range(1, 21):
        actual = summary['source_times'][str(c)]
        for key, expected in state[c].items():
            if expected is None:
                require(actual[key] is None, f'Channel {c} {key} has no corresponding action')
            else:
                close(actual[key], expected, f'Channel {c} {key}')
    for key, expected in counts.items():
        require(summary['counters'][key] == expected, f'Counter {key} mismatch')
    for phase, values in phase_cost.items():
        for key, expected in values.items():
            close(summary['phases'][phase][key], expected, f'Phase {phase}: {key}')
    close(summary['search_after_last_clear_s'], elapsed-max(state[c]['cleared_s'] for c in sources),
          'Search after last clear')
    close(summary['max_source_service_s'], max(s['service_time_s'] for s in state.values()),
          'Maximum source service time')
    for key in ('time_s', 'per_source_s', 'command_count', 'source_count', 'cleared'):
        close(outer_summary[key], summary[key], f'Validation summary versus history: {key}')
    require(outer_summary['completed'] == summary['completed'], 'Validation completion differs from history')
    return dict(seed=summary['seed'], strategy=summary['strategy'], commands=len(commands),
                sources=len(sources), cleared=len(cleared), time_s=elapsed,
                repeated_measures=repeated_measures, positive_measures=positive_measures,
                optical_fallback_clears=counts['optical_fallback_clears'],
                stop_reason='sixteen_clears' if len(cleared) == 16 else 'continuous_coverage_and_all_cleared')


def audit(validation_path, output_path):
    validation_path, output_path = Path(validation_path), Path(output_path)
    validation = json.loads(validation_path.read_text(encoding='utf-8'))
    issues, results, truth_by_seed, seen = [], [], {}, set()
    expected_files = set()
    for row in validation['rows']:
        key = (row['strategy'], row['seed'])
        if key in seen:
            issues.append(dict(strategy=key[0], seed=key[1], error='Duplicate validation row'))
            continue
        seen.add(key)
        path = BASE/row['history_file']
        expected_files.add(path.resolve())
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                history = json.load(stream)
            signature = truth_signature(history['truth'])
            if row['seed'] in truth_by_seed:
                require(truth_by_seed[row['seed']] == signature, 'Same seed has different hidden scene across strategies')
            else:
                truth_by_seed[row['seed']] = signature
            results.append(audit_history(history, row))
        except Exception as error:
            issues.append(dict(strategy=row['strategy'], seed=row['seed'], history=str(path),
                               error=f'{type(error).__name__}: {error}'))
    configs = set(validation['configs']['parameters'])
    expected_pairs = {(name, seed) for name in configs for seed in validation['seeds']}
    if seen != expected_pairs:
        issues.append(dict(error='Validation pair matrix is incomplete or has extra rows',
                           missing=sorted(expected_pairs-seen), extra=sorted(seen-expected_pairs)))
    actual_files = {p.resolve() for p in (BASE/'outputs'/'histories').glob('*/*.json.gz')}
    if actual_files != expected_files:
        issues.append(dict(error='History directory differs from validation index',
                           missing=sorted(str(p) for p in expected_files-actual_files),
                           extra=sorted(str(p) for p in actual_files-expected_files)))
    for name, expected_hash in validation.get('code_sha256', {}).items():
        actual_hash = hashlib.sha256((BASE/name).read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            issues.append(dict(error=f'Executed code changed since validation: {name}',
                               validation_hash=expected_hash, current_hash=actual_hash))
    by_strategy = []
    for strategy in sorted(configs):
        rows = [r for r in results if r['strategy'] == strategy]
        by_strategy.append(dict(strategy=strategy, runs=len(rows), commands=sum(r['commands'] for r in rows),
                                sources=sum(r['sources'] for r in rows), cleared=sum(r['cleared'] for r in rows),
                                optical_fallback_clears=sum(r['optical_fallback_clears'] for r in rows)))
    result = dict(passed=not issues, kind='independent_persisted_history_audit_not_strategy_run',
                  validation_file=str(validation_path), validation_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  expected_runs=len(validation['rows']), audited_runs=len(results),
                  unique_scenarios=len(truth_by_seed), commands=sum(r['commands'] for r in results),
                  physical_sources=sum(len(json.loads(v)) for v in truth_by_seed.values()),
                  clears_across_all_runs=sum(r['cleared'] for r in results),
                  repeated_measurements_checked=sum(r['repeated_measures'] for r in results),
                  positive_bearings_or_near_checked=sum(r['positive_measures'] for r in results),
                  by_strategy=by_strategy, issues=issues,
                  checks=['Straight-line movement / 5 m/s', 'Measurement 5 s and switching 1 s',
                          'Optical success 5 s / failure 3 s; clear leaves receiver channel unchanged',
                          'Every actual measurement obeys fixed range and 180-degree orientation',
                          'Every successful clear is <=20 m; failures and clear state reconstructed',
                          'Bearing errors <=1.005 degrees and repeated-site observations fixed',
                          'Per-source first-hit, clear and service accounting reconstructed',
                          'All source truth identical across matched seeds',
                          'Stop requires 16 successes or recorded continuous coverage and every real source cleared',
                          'Absent channels measured negatively at every recorded coverage support point',
                          'Finite optical fallback has <=122 distinct sites per channel',
                          'Complete paired seed matrix and unchanged executed source hashes'],
                  limits=['Geometry certificate construction is covered by separate continuous-coverage regression tests; '
                          'this audit verifies its recorded terminal flag and actually observed support points.'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('passed', 'audited_runs', 'unique_scenarios', 'commands',
                                            'physical_sources', 'clears_across_all_runs', 'by_strategy', 'issues')},
                     ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=BASE/'outputs'/'validation.json')
    parser.add_argument('--output', type=Path, default=BASE/'outputs'/'audit.json')
    args = parser.parse_args()
    result = audit(args.validation, args.output)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

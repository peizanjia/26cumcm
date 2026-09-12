"""Audit real Q4 actions and measure the realized benefit of side scans.

Geometry reconstruction accepts ONLY saved public commands and feedback. It
does not read truth, predict measurements or rerun a policy. Evaluator truth
is used separately by audit_history to verify physical outcomes, costs and
completion, and by paired-scene equality checks; it never updates geometry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

# Support direct execution as well as python -m invocation.
PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from question4.dynamic_joint.audit import audit_history
from question4.full_mission.audit_results import truth_signature
from question4.full_mission.service import Target

MOVE_REASONS = {
    'multiple_source_shared_point': 'shared_point_moves',
    'valuable_intermediate_stop': 'intermediate_stops',
    'multisource_refined_point': 'refined_stops',
    'dynamic_direction_frontier': 'direction_frontier_moves',
    'dynamic_coverage_completion': 'coverage_completion_moves',
}
COST_KEYS = ('movement', 'measure', 'switch', 'clear_success', 'clear_failure')


def reconstruct_behavior(commands):
    """Rebuild conservative MECs from actual public responses, without truth."""
    targets = {c: Target(c) for c in range(1, 21)}
    move_reasons, selected_features = Counter(), Counter()
    side_scans = []
    shared_clears = []
    for row in commands:
        target = targets[int(row['channel'])]
        before_active = target.status == 'active'
        before_radius = float(target.radius)
        before_center = target.center.tolist()
        # This is the only state update in geometry reconstruction.
        target.observe(row['position'], row['response'])
        if row['move_s'] > 1e-9:
            move_reasons[row['reason']] += 1
            if row['reason'] in MOVE_REASONS:
                selected_features[MOVE_REASONS[row['reason']]] += 1
        decision = row.get('decision') or {}
        selected = decision.get('selected')
        if isinstance(selected, dict):
            selected_features['selected_stops_with_decision'] += 1
            cross = selected.get('cross_source_channels') or []
            selected_features['selected_stops_with_predicted_cross_source_value'] += bool(cross)
        if row['kind'] == 'measure' and row['phase'] == 'side_known' and before_active:
            after_radius = float(target.radius)
            side_scans.append(dict(
                command_index=int(row['index']), channel=int(row['channel']),
                position=row['position'], reason=row['reason'],
                result=row['response']['measure_result'],
                before_radius_m=before_radius, after_radius_m=after_radius,
                before_center=before_center, after_center=target.center.tolist(),
                radius_reduction_m=max(0., before_radius-after_radius),
                fractional_radius_reduction=max(0., 1.-after_radius/max(before_radius, 1e-12)),
                radius_at_least_halved=after_radius <= .5*before_radius,
                newly_certifiably_clearable=before_radius > 19.8 and after_radius <= 19.8,
                move_s=float(row['move_s']), action_and_switch_s=float(row['action_s']),
            ))
        if row['reason'] == 'shared_stop_certified_clear':
            shared_clears.append(dict(command_index=int(row['index']),
                                     channel=int(row['channel']), position=row['position'],
                                     result=row['response']['clear_result']))
    return dict(
        feature_counts=dict(selected_features), move_reasons=dict(move_reasons),
        side_scan_count=len(side_scans),
        side_scan_halves=sum(r['radius_at_least_halved'] for r in side_scans),
        side_scan_no_signal=sum(r['result'] == 'no_signal' for r in side_scans),
        side_scan_new_clearable=sum(r['newly_certifiably_clearable'] for r in side_scans),
        side_scan_radius_reduction_sum_m=sum(r['radius_reduction_m'] for r in side_scans),
        side_scan_action_and_switch_s=sum(r['action_and_switch_s'] for r in side_scans),
        shared_stop_clear_count=len(shared_clears), side_scans=side_scans,
        shared_stop_clears=shared_clears,
    )


def analyze_history(history, path):
    summary = history['summary']
    try:
        # Independent physical audit: no Simulator or Planner execution.
        physical = dict(passed=True, result=audit_history(history, summary))
    except Exception as error:
        physical = dict(passed=False, error=f'{type(error).__name__}: {error}')
    behavior = reconstruct_behavior(history['commands'])
    return dict(
        strategy=summary['strategy'], seed=int(summary['seed']),
        history_file=str(Path(path).resolve()), completed=bool(summary['completed']),
        source_count=int(summary['source_count']), cleared=int(summary['cleared']),
        per_source_s=summary.get('per_source_s'), time_s=float(summary['time_s']),
        costs=dict(summary['costs']), costs_per_source=dict(summary['costs_per_source']),
        command_count=len(history['commands']), physical_audit=physical,
        geometry_source='actual command positions and public responses only',
        **behavior,
    )


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['strategy']].append(row)
    results = []
    for strategy, members in sorted(groups.items()):
        features, reasons = Counter(), Counter()
        for row in members:
            features.update(row['feature_counts'])
            reasons.update(row['move_reasons'])
        complete = all(r['completed'] and r['physical_audit']['passed']
                       and r['per_source_s'] is not None for r in members)
        side_count = sum(r['side_scan_count'] for r in members)
        halves = sum(r['side_scan_halves'] for r in members)
        results.append(dict(
            strategy=strategy, cases=len(members),
            seeds=sorted(r['seed'] for r in members),
            completed_count=sum(r['completed'] for r in members),
            physical_audit_passed_count=sum(r['physical_audit']['passed'] for r in members),
            mean_per_source_s=statistics.mean(r['per_source_s'] for r in members) if complete else None,
            costs_per_source={k: statistics.mean(r['costs_per_source'][k] for r in members) for k in COST_KEYS},
            feature_counts=dict(features), move_reasons=dict(reasons),
            side_scan_count=side_count, side_scan_halves=halves,
            side_scan_halves_fraction=halves/side_count if side_count else None,
            side_scan_no_signal=sum(r['side_scan_no_signal'] for r in members),
            side_scan_new_clearable=sum(r['side_scan_new_clearable'] for r in members),
            side_scan_radius_reduction_sum_m=sum(r['side_scan_radius_reduction_sum_m'] for r in members),
            side_scan_action_and_switch_s=sum(r['side_scan_action_and_switch_s'] for r in members),
            shared_stop_clear_count=sum(r['shared_stop_clear_count'] for r in members),
        ))
    return results


def analyze(paths, baseline='old_joint'):
    rows, issues, scenes = [], [], {}
    for path in sorted(set(Path(p) for p in paths)):
        try:
            with gzip.open(path, 'rt', encoding='utf-8') as stream:
                history = json.load(stream)
            seed = int(history['summary']['seed'])
            signature = truth_signature(history['truth'])
            if seed in scenes and scenes[seed] != signature:
                issues.append(dict(history_file=str(path), error='Paired initial scenes differ'))
            else:
                scenes[seed] = signature
            rows.append(analyze_history(history, path))
        except Exception as error:
            issues.append(dict(history_file=str(path), error=f'{type(error).__name__}: {error}'))
    old = {r['seed']: r for r in rows if r['strategy'] == baseline}
    comparisons = []
    for strategy in sorted({r['strategy'] for r in rows} - {baseline}):
        pairs = []
        for row in sorted((r for r in rows if r['strategy'] == strategy), key=lambda r: r['seed']):
            previous = old.get(row['seed'])
            if previous is None:
                continue
            complete = (row['completed'] and previous['completed'] and
                        row['physical_audit']['passed'] and previous['physical_audit']['passed'])
            difference = row['per_source_s']-previous['per_source_s'] if complete else None
            pairs.append(dict(seed=row['seed'], complete=complete,
                              baseline_per_source_s=previous['per_source_s'],
                              current_per_source_s=row['per_source_s'], difference_s=difference))
        differences = [p['difference_s'] for p in pairs if p['complete']]
        comparisons.append(dict(
            strategy=strategy, baseline=baseline, paired_count=len(pairs),
            complete_pair_count=len(differences),
            mean_difference_s=statistics.mean(differences) if differences and len(differences) == len(pairs) else None,
            faster_count=sum(x < 0 for x in differences), pairs=pairs,
        ))
    return dict(
        scope='Read-only behavior analysis of saved local synthetic full-mission histories',
        geometry_reconstruction='Only actual command positions and public responses update Target; truth is not used',
        truth_usage='Separate offline physical audit and equality of paired initial scenes only',
        side_scan_definition='Any actual measure command with phase side_known and an already active target; includes old and new policies',
        benefit_definition='Observed change of conservative MEC radius after that real side measurement; not a counterfactual total-time saving',
        selected_feature_definition='Shared/intermediate/refined counts require actual nonzero movement; predicted cross-source choices are counted from the selected decision',
        count_caveat='Feature occurrence proves execution, not that a feature alone caused a total-time improvement',
        complete_matrix_caveat='A histories-directory snapshot does not establish that the intended benchmark matrix has finished',
        passed=bool(rows) and not issues and all(r['physical_audit']['passed'] for r in rows),
        analyzed_runs=len(rows), command_count=sum(r['command_count'] for r in rows),
        paired_scenes_equal=not any(i['error'] == 'Paired initial scenes differ' for i in issues),
        summary=aggregate(rows), comparisons=comparisons, issues=issues, runs=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--histories', type=Path, help='Directory containing strategy/seed.json.gz')
    group.add_argument('--validation', type=Path, help='Completed validate.json with rows/history_file')
    parser.add_argument('--strategies', help='Optional comma-separated strategy directory names')
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--seed-end', type=int)
    parser.add_argument('--baseline', default='old_joint')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.validation:
        data = json.loads(args.validation.read_text(encoding='utf-8'))
        paths = [Path(r['history_file']) for r in data['rows']]
    else:
        paths = list(args.histories.glob('*/*.json.gz'))
    wanted = set(args.strategies.split(',')) if args.strategies else None
    paths = [p for p in paths if (wanted is None or p.parent.name in wanted)
             and (args.seed_start is None or int(p.name.split('.')[0]) >= args.seed_start)
             and (args.seed_end is None or int(p.name.split('.')[0]) <= args.seed_end)]
    result = analyze(paths, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(passed=result['passed'], analyzed_runs=result['analyzed_runs'],
                          command_count=result['command_count'], summary=result['summary'],
                          comparisons=result['comparisons'], issues=result['issues'],
                          output=str(args.output.resolve())), ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

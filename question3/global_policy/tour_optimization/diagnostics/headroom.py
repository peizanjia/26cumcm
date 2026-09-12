"""Offline diagnosis of frozen traces; hidden truth never enters the policy.

The center-tour relaxation is a lower bound, not an attainable sensing policy
or a forecast of the improvement from a proposed implementation.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from ...route_study.routes import held_karp


def analyze(folder):
    audit_path = folder / 'audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    source_rows = audit['replay_audit']['optimized']['rows']
    rows = []
    for saved in source_rows:
        seed = saved['seed']
        trace = json.loads(gzip.decompress((folder / 'optimized' / f'{seed}.json.gz').read_bytes()))
        points = np.array([[s['x'], s['y']] for s in trace['truth']], dtype=float)
        n = len(points)
        center_length, order = held_karp(
            np.linalg.norm(points, axis=1),
            np.linalg.norm(points[:, None] - points[None, :], axis=2),
            np.zeros(n))
        # A successful clearance lies within 20 m of each true center. Replace
        # the first endpoint by its center (<=20 m) and every subsequent edge's
        # two endpoints by their centers (<=40 m per edge).
        movement_lb = max(0., center_length - (40 * n - 20))
        row = dict(seed=seed, source_count=n, exact_center_tour_m=float(center_length),
                   clearance_disk_movement_lower_bound_m=float(movement_lb),
                   time_lower_bound_s_per_source=float(movement_lb / (5 * n) + 5),
                   actual_s_per_source=saved['average_s'],
                   actual_movement_s_per_source=saved['movement_s_per_source'],
                   actual_movement_m=float(trace['summary']['distance_m']),
                   center_tour_order_channels=[trace['truth'][i]['channel'] for i in order])
        local_count = changed = positive_credit = 0
        for event in trace['events']:
            if event['phase'] != 'local_decision':
                continue
            candidates = event.get('candidates', [])
            selected = [v for v in candidates if v.get('selected')]
            if not selected:
                continue
            local_count += 1
            best = selected[0]
            changed += best is not min(candidates, key=lambda v: v['expected_remaining_s'])
            positive_credit += best.get('side_scan_credit_s', 0.) > 0.
        row.update(local_decisions=local_count, reranked_decisions=changed,
                   selected_with_positive_side_credit=positive_credit,
                   geometry_investment_scans=sum(c['path'] == '/measure' and
                       c.get('reason') == 'early_geometry_mapping_spend' for c in trace['commands']))
        rows.append(row)
    group = audit['groups']['optimized']
    costs = {k: v['mean'] for k, v in group['component_s_per_source'].items()}
    actual = group['time_s_per_source']['mean']
    means = {k: float(np.mean([r[k] for r in rows])) for k in (
        'exact_center_tour_m', 'clearance_disk_movement_lower_bound_m',
        'time_lower_bound_s_per_source', 'actual_movement_m', 'local_decisions',
        'reranked_decisions', 'selected_with_positive_side_credit', 'geometry_investment_scans')}
    replay = audit['replay_audit']['optimized']['means']
    fixed_other_cost_floor = (means['time_lower_bound_s_per_source'] - 5
                              + sum(v for k, v in costs.items() if k != 'movement'))
    return dict(evaluation='offline_diagnosis_of_already_used_local_validation_traces',
                audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                count=len(rows), costs_s_per_source=costs,
                movement_fraction=costs['movement'] / actual,
                target_gap_s_per_source=actual-200,
                required_total_reduction_fraction=(actual-200)/actual,
                required_movement_reduction_if_other_costs_fixed=(actual-200)/costs['movement'],
                lower_bound_if_nonmovement_costs_fixed_s_per_source=fixed_other_cost_floor,
                minimum_nonmovement_reduction_required_under_relaxation=max(0., fixed_other_cost_floor-200),
                post_clear_tail_s_per_source=replay['post_clear_coverage_time_s_per_source'],
                average_unknown_scans=replay['counters']['unknown_scans'],
                average_known_scans=replay['counters']['known_scans'],
                last_discovery_fraction=replay['last_discovery_fraction'], means=means, rows=rows,
                limitation='Lower bound uses evaluator-only true positions, relaxes all sensing and '
                    'absence certification, and counts only movement and successful clearance. '
                    'The gap is not a prediction of achievable improvement. Re-ranking differences '
                    'combine side credit and revisit penalties, not a causal effect of either alone. '
                    'No new policy run or new holdout validation was performed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='question3/global_policy/tour_optimization/outputs/release_validation')
    parser.add_argument('--output', default='question3/global_policy/tour_optimization/outputs/followup_diagnosis.json')
    args = parser.parse_args()
    result = analyze(Path(args.input))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

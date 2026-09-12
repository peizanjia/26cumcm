"""Recompute exploratory whole-scene paired effects from complete raw case sets.

Never read report/case_selection.json or selected HTML examples as a sample.
All intervals are marginal Student-t intervals, not multiplicity adjusted.
Different development folders often reuse the same seeds and are not pooled.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t


DATASETS = ('development_v1', 'development_v2', 'factor_screen', 'sweep_screen',
            'early_geometry_screen', 'soft_sweep_screen', 'anticipated_screen', 'sector_screen', 'tour_screen')
CONTRASTS = {
    'v1_all_changes': ('development_v1', 'adaptive', 'development_v1', 'combined'),
    'v2_all_changes': ('development_v2', 'adaptive', 'development_v2', 'combined'),
    'v2_vs_v1_implementation': ('development_v2', 'adaptive', 'development_v1', 'adaptive'),
    'stop_scans_on': ('factor_screen', 'adaptive', 'factor_screen', 'no_stop_scans'),
    'unknown_scans_lean': ('factor_screen', 'lean', 'factor_screen', 'adaptive'),
    'early_mapping_on': ('factor_screen', 'lean', 'factor_screen', 'lean_no_early'),
    'revisit_penalty_on': ('factor_screen', 'lean', 'factor_screen', 'lean_no_revisit'),
    'waypoint_candidates_on': ('factor_screen', 'lean', 'factor_screen', 'lean_no_waypoints'),
    'lean_vs_combined': ('factor_screen', 'lean', 'factor_screen', 'combined'),
    'unlock_sweep_service': ('sweep_screen', 'unlock_voi', 'sweep_screen', 'lock_voi'),
    'sweep_value_scans': ('sweep_screen', 'unlock_voi', 'sweep_screen', 'unlock_old'),
    'sweep_initial_map_probe': ('sweep_screen', 'unlock_voi_initial', 'sweep_screen', 'unlock_voi'),
    'early_geometry_6_on': ('early_geometry_screen', 'geometry_6', 'early_geometry_screen', 'geometry_off'),
    'early_geometry_3_on': ('early_geometry_screen', 'geometry_3', 'early_geometry_screen', 'geometry_off'),
    'early_geometry_6_vs_3': ('early_geometry_screen', 'geometry_6', 'early_geometry_screen', 'geometry_3'),
    'early_geometry_6_vs_combined': ('early_geometry_screen', 'geometry_6', 'development_v2', 'combined'),
    'soft_sweep_80_on': ('soft_sweep_screen', 'sweep80', 'soft_sweep_screen', 'map_global'),
    'soft_sweep_160_on': ('soft_sweep_screen', 'sweep160', 'soft_sweep_screen', 'map_global'),
    'soft_sweep_160_vs_combined': ('soft_sweep_screen', 'sweep160', 'soft_sweep_screen', 'combined'),
    'map_global_vs_combined': ('soft_sweep_screen', 'map_global', 'soft_sweep_screen', 'combined'),
    'anticipated_vs_combined': ('anticipated_screen', 'anticipated', 'anticipated_screen', 'combined'),
    'anticipated_frequent_vs_combined': ('anticipated_screen', 'anticipated_frequent', 'anticipated_screen', 'combined'),
    'tour_vs_combined': ('tour_screen', 'tour', 'development_v2', 'combined'),
    'tour_vs_geometry_6': ('tour_screen', 'tour', 'early_geometry_screen', 'geometry_6'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return dict(n=0, mean=None, standard_error=None, ci95=None)
    average = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    half = float(student_t.ppf(.975, n - 1) * se) if n > 1 else None
    return dict(n=n, mean=average, standard_error=se,
                ci95=[average - half, average + half] if half is not None else None)


def normalize_cases(raw):
    if isinstance(raw, dict):
        return {name: sorted(items, key=lambda row: row['seed']) for name, items in raw.items()}
    grouped = defaultdict(list)
    for row in raw:
        variant = row.get('variant') or ('tour' if row.get('policy') == 'adaptive_tour' else row.get('policy', 'unnamed'))
        grouped[variant].append(row)
    return {name: sorted(items, key=lambda row: row['seed']) for name, items in grouped.items()}


def profiles(rows):
    result = {}
    for row in rows:
        params = row.get('parameters', {})
        key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
        item = result.setdefault(key, dict(parameters=params, seeds=[]))
        item['seeds'].append(row['seed'])
    return result


def compare(left_rows, right_rows):
    left = {r['seed']: r for r in left_rows}
    right = {r['seed']: r for r in right_rows}
    common = sorted(set(left) & set(right))
    good = [s for s in common if left[s].get('complete') and right[s].get('complete')
            and isinstance(left[s].get('average_time_s'), (float, int))
            and isinstance(right[s].get('average_time_s'), (float, int))]
    differences = [left[s]['average_time_s'] - right[s]['average_time_s'] for s in good]
    report = dict(sign='left_minus_right; negative means left faster',
        common_seeds=common, paired_complete_seeds=good,
        excluded_incomplete_seeds=[s for s in common if s not in good],
        timing_conditional_on_both_complete=len(good) < len(common),
        left_complete=sum(bool(left[s].get('complete')) for s in common),
        right_complete=sum(bool(right[s].get('complete')) for s in common),
        paired_time_s_per_source=mean_ci(differences),
        left_mean_s_per_source=mean_ci([left[s]['average_time_s'] for s in good])['mean'],
        right_mean_s_per_source=mean_ci([right[s]['average_time_s'] for s in good])['mean'],
        faster=sum(d < -1e-9 for d in differences), same=sum(abs(d) <= 1e-9 for d in differences),
        slower=sum(d > 1e-9 for d in differences),
        per_seed=[dict(seed=s, left=left[s]['average_time_s'], right=right[s]['average_time_s'],
                      difference=left[s]['average_time_s'] - right[s]['average_time_s']) for s in good])
    components = {}
    for key in ('movement', 'measure', 'switch', 'clear'):
        valid = [s for s in good if key in left[s].get('time_components_s', {})
                 and key in right[s].get('time_components_s', {})]
        if valid:
            components[key] = mean_ci([left[s]['time_components_s'][key] / left[s]['true_total']
                                     - right[s]['time_components_s'][key] / right[s]['true_total']
                                     for s in valid])
    report['component_differences_s_per_source'] = components
    lp, rp = profiles(left_rows), profiles(right_rows)
    if len(lp) == len(rp) == 1:
        a = next(iter(lp.values()))['parameters']
        b = next(iter(rp.values()))['parameters']
        report['parameter_differences'] = {k: dict(left=a.get(k), right=b.get(k))
                                         for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
    return report


def summarize_group(rows):
    complete = [r for r in rows if r.get('complete')]
    return dict(count=len(rows), complete=len(complete), seeds=[r['seed'] for r in rows],
        mean_s_per_source=mean_ci([r['average_time_s'] for r in complete])['mean'],
        mean_measures=mean_ci([r['measures'] for r in rows])['mean'],
        mean_distance_m=mean_ci([r['distance_m'] for r in rows])['mean'],
        cleared_sources=sum(r.get('true_cleared', r.get('cleared', 0)) for r in rows),
        failures=[dict(seed=r['seed'], failure=r.get('failure')) for r in rows if not r.get('complete')],
        parameter_profiles=profiles(rows),
        raw_case_metrics=[{k: r.get(k) for k in ('seed', 'variant', 'complete', 'average_time_s',
            'virtual_time_s', 'true_total', 'true_cleared', 'distance_m', 'measures', 'misses',
            'failure', 'time_components_s', 'replans', 'target_switches', 'geometric_mapping_scans',
            'policy', 'tour_replans')}
                          for r in rows])


def tail_decomposition(root):
    source = root / 'coverage_audit_v2.json'
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding='utf-8'))
    groups = defaultdict(dict)
    for row in data['rows']:
        groups[row['variant']][row['seed']] = row
    common = sorted(set(groups['combined']) & set(groups['adaptive']))
    total, tail, details = [], [], []
    for seed in common:
        a, b = groups['adaptive'][seed], groups['combined'][seed]
        dt = a['average_s'] - b['average_s']
        dh = a['post_clear_coverage_time_s'] / a['source_count'] - b['post_clear_coverage_time_s'] / b['source_count']
        total.append(dt)
        tail.append(dh)
        details.append(dict(seed=seed, total_difference=dt, post_clear_difference=dh,
                            pre_final_clear_difference=dt - dh))
    return dict(source=str(source), sha256=digest(source), seeds=common,
        total_difference_s_per_source=mean_ci(total), post_clear_difference_s_per_source=mean_ci(tail),
        share_of_mean_regression=float(np.mean(tail) / np.mean(total)),
        interpretation='Accounting decomposition, not causal savings: absence proof remains mandatory.',
        raw_per_seed=details)


def geometry_investment_logs(root):
    """Read all geometry_6 traces, not just cases selected for presentation."""
    folder = root / 'early_geometry_screen' / 'geometry_6'
    sources, scans = [], []
    for path in sorted(folder.glob('*.json.gz')):
        with gzip.open(path, 'rt', encoding='utf-8') as handle:
            pack = json.load(handle)
        seed = pack['summary']['seed']
        sources.append(dict(path=str(path), seed=seed, sha256=digest(path)))
        for event in pack.get('events', []):
            if event.get('phase') != 'stop_scan':
                continue
            for row in event.get('values', []):
                if row.get('reason') == 'early_geometry_mapping_spend':
                    scans.append(dict(seed=seed, command_index=event['command_index'],
                        channel=row['channel'], net_saved_s=row.get('net_saved_s'),
                        expected_radius_ratio=row.get('expected_radius_ratio'),
                        detection_probability=row.get('detection_probability')))
    values = [row['net_saved_s'] for row in scans if row['net_saved_s'] is not None]
    return dict(source_traces=sources, scan_count=len(scans),
                negative_local_net_voi_count=sum(value < 0 for value in values),
                mean_local_net_voi_s=float(np.mean(values)) if values else None,
                scans=scans,
                note='Local proxy predictions at logged stops, not independent measured causal effects.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='question3/global_policy/adaptive_mpc/outputs')
    parser.add_argument('--output', help='Default: <root>/six_effects.json')
    args = parser.parse_args()
    root = Path(args.root)
    output = Path(args.output) if args.output else root / 'six_effects.json'
    groups = {}
    result = dict(evaluation='exploratory_local_synthetic_development',
        statistical_unit='whole_scene', interval='marginal paired Student-t 95 percent',
        caveats=[
            'Repeated development seeds across folders are not independent new evaluations.',
            'Multiple exploratory intervals are not multiplicity-adjusted.',
            'Do not add the effects: actions, scans and route changes interact.',
            'Completion failures are excluded only from time comparisons and reported explicitly.',
            'Selected report cases are never read as evaluation samples.',
            'Current code is not substituted for historical parameter profiles.',
        ], datasets={}, named_contrasts={}, unavailable=[])
    for name in DATASETS:
        case_path = root / name / 'cases.json'
        if not case_path.exists():
            result['unavailable'].append(name)
            continue
        cases = normalize_cases(json.loads(case_path.read_text(encoding='utf-8')))
        groups[name] = cases
        manifest_path = root / name / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        rawhash = manifest.get('source_hash')
        warnings = []
        if rawhash == hashlib.sha256(b'').hexdigest():
            warnings.append('Recorded source_hash is SHA256(empty): it does not freeze the historical implementation.')
        elif not rawhash and not manifest.get('source_sha256'):
            warnings.append('No source_hash in manifest; parameter/seed evidence exists but exact historical code is not frozen.')
        item = dict(cases_path=str(case_path), cases_sha256=digest(case_path),
            manifest=manifest, warnings=warnings,
            variants={v: summarize_group(rows) for v, rows in cases.items()}, pairwise={})
        for left, right in combinations(cases, 2):
            item['pairwise'][f'{left}_minus_{right}'] = compare(cases[left], cases[right])
        result['datasets'][name] = item
    for key, (ld, lv, rd, rv) in CONTRASTS.items():
        if ld not in groups or rd not in groups or lv not in groups[ld] or rv not in groups[rd]:
            result['unavailable'].append(key)
            continue
        result['named_contrasts'][key] = dict(left=dict(dataset=ld, variant=lv),
            right=dict(dataset=rd, variant=rv), cross_version_comparison=ld != rd,
            **compare(groups[ld][lv], groups[rd][rv]))
    result['v2_tail_decomposition'] = tail_decomposition(root)
    result['early_geometry_investment_logs'] = geometry_investment_logs(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(output), datasets=len(result['datasets']),
                         contrasts=len(result['named_contrasts']), unavailable=result['unavailable'])))
    for key, row in result['named_contrasts'].items():
        stats = row['paired_time_s_per_source']
        print(f'{key}: n={stats["n"]} delta={stats["mean"]:.6f} CI={stats["ci95"]}')


if __name__ == '__main__':
    main()
